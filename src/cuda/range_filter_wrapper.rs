//! CUDA wrapper for the Range Filter indicator.
//!
//! Parity goals (mirrors ALMA/CWMA wrappers):
//! - PTX load via DetermineTargetFromContext + OptLevel O2 with fallback
//! - NON_BLOCKING stream
//! - Policy enums (simple 1D launch for batch; 1D blocks per series for many-series)
//! - VRAM checks + grid.y chunking (<= 65_535) for batch
//! - Warmup/NaN semantics identical to scalar: warm = first_valid + max(range_period, smooth? smooth_period : 0)
//! - Double-precision accumulators in kernels; FP32 I/O

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::range_filter::{RangeFilterBatchRange, RangeFilterParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaRangeFilterError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaRangeFilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaRangeFilterError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaRangeFilterError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaRangeFilterError {}

/// Three VRAM-backed arrays for (filter, high, low)
pub struct DeviceRangeFilterTrio {
    pub filter: DeviceBuffer<f32>,
    pub high: DeviceBuffer<f32>,
    pub low: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl DeviceRangeFilterTrio {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}
impl Default for BatchKernelPolicy {
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        ManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaRangeFilterPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaRangeFilter {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaRangeFilterPolicy,
    debug_logged: std::sync::atomic::AtomicBool,
}

impl CudaRangeFilter {
    pub fn new(device_id: usize) -> Result<Self, CudaRangeFilterError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let context =
            Context::new(device).map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/range_filter_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])
                    .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaRangeFilterPolicy::default(),
            debug_logged: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub fn set_policy(&mut self, p: CudaRangeFilterPolicy) {
        self.policy = p;
    }
    pub fn synchronize(&self) -> Result<(), CudaRangeFilterError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))
    }

    #[inline]
    fn headroom_bytes() -> usize {
        env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024)
    }
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() {
            bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    // ---------------- Batch: one series × many params ----------------
    pub fn range_filter_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &RangeFilterBatchRange,
    ) -> Result<(DeviceRangeFilterTrio, Vec<RangeFilterParams>), CudaRangeFilterError> {
        if data_f32.is_empty() {
            return Err(CudaRangeFilterError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaRangeFilterError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaRangeFilterError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        // Validate periods and warmup coverage like scalar
        let max_needed = combos
            .iter()
            .map(|c| {
                let rp = c.range_period.unwrap_or(14);
                let sp = if c.smooth_range.unwrap_or(true) {
                    c.smooth_period.unwrap_or(27)
                } else {
                    0
                };
                rp.max(sp)
            })
            .max()
            .unwrap_or(0);
        let valid = len - first_valid;
        if valid < max_needed {
            return Err(CudaRangeFilterError::InvalidInput(format!(
                "not enough valid data: needed = {}, valid = {}",
                max_needed, valid
            )));
        }
        for p in &combos {
            let rs = p.range_size.unwrap_or(2.618);
            if !rs.is_finite() || rs <= 0.0 {
                return Err(CudaRangeFilterError::InvalidInput(
                    "invalid range_size".into(),
                ));
            }
            let rp = p.range_period.unwrap_or(14);
            if rp == 0 || rp > len {
                return Err(CudaRangeFilterError::InvalidInput(
                    "invalid range_period".into(),
                ));
            }
            let sr = p.smooth_range.unwrap_or(true);
            let sp = p.smooth_period.unwrap_or(27);
            if sr && (sp == 0 || sp > len) {
                return Err(CudaRangeFilterError::InvalidInput(
                    "invalid smooth_period".into(),
                ));
            }
        }

        // VRAM estimate
        let rows = combos.len();
        let in_bytes = len * std::mem::size_of::<f32>();
        let params_bytes = rows * (std::mem::size_of::<f32>() + 3 * std::mem::size_of::<i32>());
        let out_bytes = 3 * rows * len * std::mem::size_of::<f32>();
        let required = in_bytes + params_bytes + out_bytes;
        if !Self::will_fit(required, Self::headroom_bytes()) {
            return Err(CudaRangeFilterError::InvalidInput(
                "insufficient device memory for range_filter batch".into(),
            ));
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let range_sizes_f32: Vec<f32> = combos
            .iter()
            .map(|c| c.range_size.unwrap_or(2.618) as f32)
            .collect();
        let range_periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.range_period.unwrap_or(14) as i32)
            .collect();
        let smooth_flags_i32: Vec<i32> = combos
            .iter()
            .map(|c| if c.smooth_range.unwrap_or(true) { 1 } else { 0 })
            .collect();
        let smooth_periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.smooth_period.unwrap_or(27) as i32)
            .collect();
        let d_rs = DeviceBuffer::from_slice(&range_sizes_f32)
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let d_rp = DeviceBuffer::from_slice(&range_periods_i32)
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let d_sf = DeviceBuffer::from_slice(&smooth_flags_i32)
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let d_sp = DeviceBuffer::from_slice(&smooth_periods_i32)
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;

        let mut d_f: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let mut d_h: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let mut d_l: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * len) }
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;

        // Kernel launch with grid.y chunking
        let func = self
            .module
            .get_function("range_filter_batch_f32")
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
            _ => 1,
        };
        const MAX_GRID_Y: usize = 65_535;
        let mut start = 0usize;
        while start < rows {
            let count = (rows - start).min(MAX_GRID_Y);
            let grid: GridSize = (1u32, count as u32, 1u32).into();
            let block: BlockSize = (block_x, 1, 1).into();
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut rs_ptr = d_rs.as_device_ptr().add(start).as_raw();
                let mut rp_ptr = d_rp.as_device_ptr().add(start).as_raw();
                let mut sf_ptr = d_sf.as_device_ptr().add(start).as_raw();
                let mut sp_ptr = d_sp.as_device_ptr().add(start).as_raw();
                let mut len_i = len as i32;
                let mut nrows_i = count as i32;
                let mut first_i = first_valid as i32;
                let base = start * len;
                let mut f_ptr = d_f.as_device_ptr().add(base).as_raw();
                let mut h_ptr = d_h.as_device_ptr().add(base).as_raw();
                let mut l_ptr = d_l.as_device_ptr().add(base).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut rs_ptr as *mut _ as *mut c_void,
                    &mut rp_ptr as *mut _ as *mut c_void,
                    &mut sf_ptr as *mut _ as *mut c_void,
                    &mut sp_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut nrows_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut f_ptr as *mut _ as *mut c_void,
                    &mut h_ptr as *mut _ as *mut c_void,
                    &mut l_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
            }
            start += count;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;

        Ok((
            DeviceRangeFilterTrio {
                filter: d_f,
                high: d_h,
                low: d_l,
                rows,
                cols: len,
            },
            combos,
        ))
    }

    // ------------- Many series × one param (time‑major) -------------
    pub fn range_filter_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &RangeFilterParams,
    ) -> Result<DeviceRangeFilterTrio, CudaRangeFilterError> {
        if rows == 0 || cols == 0 {
            return Err(CudaRangeFilterError::InvalidInput("empty dims".into()));
        }
        if data_tm_f32.len() != rows * cols {
            return Err(CudaRangeFilterError::InvalidInput(
                "time-major input must be rows*cols".into(),
            ));
        }
        let rs_f64 = params.range_size.unwrap_or(2.618);
        if !rs_f64.is_finite() || rs_f64 <= 0.0 {
            return Err(CudaRangeFilterError::InvalidInput(
                "invalid range_size".into(),
            ));
        }
        let rs = rs_f64 as f32;
        let rp = params.range_period.unwrap_or(14) as i32;
        let sr = params.smooth_range.unwrap_or(true);
        let sp = params.smooth_period.unwrap_or(27) as i32;
        if rp <= 0 || (sr && sp <= 0) {
            return Err(CudaRangeFilterError::InvalidInput("invalid period".into()));
        }

        // first_valid per series (column) scanning time-major layout
        let mut first_valids = vec![cols as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                if data_tm_f32[idx].is_finite() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }

        let in_bytes = rows * cols * std::mem::size_of::<f32>();
        let out_bytes = 3 * rows * cols * std::mem::size_of::<f32>();
        let aux_bytes = cols * std::mem::size_of::<i32>();
        if !Self::will_fit(in_bytes + out_bytes + aux_bytes, Self::headroom_bytes()) {
            return Err(CudaRangeFilterError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        let d_data = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let mut d_f: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * cols) }
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let mut d_h: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * cols) }
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let mut d_l: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(rows * cols) }
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("range_filter_many_series_one_param_f32")
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
            _ => 1,
        };
        let grid: GridSize = (cols as u32, 1u32, 1u32).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut data_ptr = d_data.as_device_ptr().as_raw();
            let mut rs_f = rs;
            let mut rp_i = rp;
            let mut sf_i = if sr { 1i32 } else { 0i32 };
            let mut sp_i = sp;
            let mut cols_i = cols as i32; // num_series
            let mut rows_i = rows as i32; // series_len
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut f_ptr = d_f.as_device_ptr().as_raw();
            let mut h_ptr = d_h.as_device_ptr().as_raw();
            let mut l_ptr = d_l.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut data_ptr as *mut _ as *mut c_void,
                &mut rs_f as *mut _ as *mut c_void,
                &mut rp_i as *mut _ as *mut c_void,
                &mut sf_i as *mut _ as *mut c_void,
                &mut sp_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut f_ptr as *mut _ as *mut c_void,
                &mut h_ptr as *mut _ as *mut c_void,
                &mut l_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaRangeFilterError::Cuda(e.to_string()))?;

        Ok(DeviceRangeFilterTrio {
            filter: d_f,
            high: d_h,
            low: d_l,
            rows,
            cols,
        })
    }
}

// ---------------- Helpers ----------------
#[inline]
fn expand_grid(r: &RangeFilterBatchRange) -> Vec<RangeFilterParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    let range_sizes = axis_f64(r.range_size);
    let range_periods = axis_usize(r.range_period);
    let mut out = Vec::with_capacity(range_sizes.len() * range_periods.len());
    for &rs in &range_sizes {
        for &rp in &range_periods {
            out.push(RangeFilterParams {
                range_size: Some(rs),
                range_period: Some(rp),
                smooth_range: r.smooth_range,
                smooth_period: r.smooth_period,
            });
        }
    }
    out
}

// ---------------- Bench profiles ----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    struct RfBatchState {
        cuda: CudaRangeFilter,
    }
    impl CudaBenchState for RfBatchState {
        fn launch(&mut self) {
            let _ = &self.cuda;
        }
    }

    fn prep_rf_batch() -> Box<dyn CudaBenchState> {
        let len = 120_000usize;
        let mut data = vec![f32::NAN; len];
        for i in 5..len {
            let x = i as f32;
            data[i] = (x * 0.0021).sin() + 0.00021 * x;
        }
        let sweep = RangeFilterBatchRange {
            range_size: (2.0, 4.0, 0.2),
            range_period: (8, 64, 8),
            smooth_range: Some(true),
            smooth_period: Some(27),
        };
        let cuda = CudaRangeFilter::new(0).unwrap();
        let _ = cuda.range_filter_batch_dev(&data, &sweep).unwrap();
        Box::new(RfBatchState { cuda })
    }

    struct RfManySeriesState {
        cuda: CudaRangeFilter,
    }
    impl CudaBenchState for RfManySeriesState {
        fn launch(&mut self) {
            let _ = &self.cuda;
        }
    }

    fn prep_rf_many_series() -> Box<dyn CudaBenchState> {
        let cols = 256usize; // series
        let rows = 600_000usize; // time
        let mut tm = vec![f32::NAN; rows * cols];
        for s in 0..cols {
            for t in s..rows {
                let idx = t * cols + s;
                let x = t as f32 + s as f32 * 0.01;
                tm[idx] = (x * 0.0013).sin() + 0.00011 * x;
            }
        }
        let params = RangeFilterParams {
            range_size: Some(2.618),
            range_period: Some(14),
            smooth_range: Some(true),
            smooth_period: Some(27),
        };
        let cuda = CudaRangeFilter::new(0).unwrap();
        let _ = cuda
            .range_filter_many_series_one_param_time_major_dev(&tm, cols, rows, &params)
            .unwrap();
        Box::new(RfManySeriesState { cuda })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "range_filter",
                "one_series_many_params",
                "cuda/range_filter",
                "batch",
                prep_rf_batch,
            )
            .with_sample_size(25),
            CudaBenchScenario::new(
                "range_filter",
                "many_series_one_param",
                "cuda/range_filter",
                "many_series",
                prep_rf_many_series,
            )
            .with_sample_size(15),
        ]
    }
}
