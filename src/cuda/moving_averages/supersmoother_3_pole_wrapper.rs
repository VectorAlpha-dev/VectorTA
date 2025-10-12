//! CUDA wrapper for the 3-pole SuperSmoother filter kernels.
//!
//! Mirrors the ALMA/CWMA-style wrapper: VRAM-first device handles, simple
//! policy selection (plain kernels only), JIT options for PTX loading,
//! VRAM checks, and chunked launches to respect grid limits. Kernels operate
//! in FP32 with FP64 intermediates to match f64 scalar behavior.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::supersmoother_3_pole::{
    SuperSmoother3PoleBatchRange, SuperSmoother3PoleParams,
};
use cust::context::{CacheConfig, Context};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, CopyDestination, AsyncCopyDestination, LockedBuffer, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[inline(always)]
fn div_up(a: usize, b: usize) -> usize { (a + b - 1) / b }

#[derive(Debug)]
pub enum CudaSuperSmoother3PoleError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaSuperSmoother3PoleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaSuperSmoother3PoleError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaSuperSmoother3PoleError::InvalidInput(msg) => {
                write!(f, "Invalid input: {}", msg)
            }
        }
    }
}

impl std::error::Error for CudaSuperSmoother3PoleError {}

// -------- Kernel policy + introspection (kept minimal; no tiled variants) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaSupersmoother3PolePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaSupersmoother3PolePolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { Plain { block_x: u32 } }
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }

pub struct CudaSupersmoother3Pole {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaSupersmoother3PolePolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaSupersmoother3Pole {
    pub fn new(device_id: usize) -> Result<Self, CudaSuperSmoother3PoleError> {
        cust::init(CudaFlags::empty())
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        let context =
            Context::new(device).map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/supersmoother_3_pole_kernel.ptx"));
        // Use context-targeted JIT; default to O4 (most optimized), unless overridden.
        let opt = match std::env::var("CUDA_JIT_OPT").ok().as_deref() {
            Some("O0") => OptLevel::O0,
            Some("O1") => OptLevel::O1,
            Some("O2") => OptLevel::O2,
            Some("O3") => OptLevel::O3,
            _ => OptLevel::O4,
        };
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(opt),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaSupersmoother3PolePolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaSupersmoother3PolePolicy,
    ) -> Result<Self, CudaSuperSmoother3PoleError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaSupersmoother3PolePolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaSupersmoother3PolePolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    pub fn synchronize(&self) -> Result<(), CudaSuperSmoother3PoleError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))
    }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }

    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    fn expand_periods(range: &SuperSmoother3PoleBatchRange) -> Vec<SuperSmoother3PoleParams> {
        let (start, end, step) = range.period;
        let periods = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect::<Vec<_>>()
        };
        periods
            .into_iter()
            .map(|p| SuperSmoother3PoleParams { period: Some(p) })
            .collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SuperSmoother3PoleBatchRange,
    ) -> Result<(Vec<SuperSmoother3PoleParams>, usize, usize), CudaSuperSmoother3PoleError> {
        if data_f32.is_empty() {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "price data is empty".into(),
            ));
        }

        let first_valid = data_f32.iter().position(|v| !v.is_nan()).ok_or_else(|| {
            CudaSuperSmoother3PoleError::InvalidInput("all values are NaN".into())
        })?;

        let combos = Self::expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "no period combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaSuperSmoother3PoleError::InvalidInput(
                    "period must be >= 1".into(),
                ));
            }
            if period > series_len {
                return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                    "period {} exceeds series length {}",
                    period, series_len
                )));
            }
            let valid = series_len - first_valid;
            if valid < period {
                return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                    "not enough valid data: need >= {}, valid = {}",
                    period, valid
                )));
            }
        }

        Ok((combos, first_valid, series_len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSuperSmoother3PoleError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "series_len and n_combos must be > 0".into(),
            ));
        }

        // Kernel + prefer L1
        let mut func = self
            .module
            .get_function("supersmoother_3_pole_batch_f32")
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        // Occupancy-based block size for Auto
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => match func.suggested_launch_configuration(0, (0, 0, 0).into()) {
                Ok((_min_grid, block)) => block.max(64),
                Err(_) => 256,
            },
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
        };
        unsafe {
            (*(self as *const _ as *mut CudaSupersmoother3Pole)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }

        // True device max grid.x
        let device = Device::get_device(self.device_id)
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))? as usize;

        let tpb = block_x as usize;
        let chunk_capacity = max_grid_x.saturating_mul(tpb);

        let mut launched = 0usize;
        while launched < n_combos {
            let launch_elems = (n_combos - launched).min(chunk_capacity);
            let blocks = (launch_elems + tpb - 1) / tpb;

            let grid: GridSize = (blocks as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(launched).as_raw();
                let mut series_len_i = series_len as i32;
                let mut n_elems_i = launch_elems as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .add(launched * series_len)
                    .as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_elems_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
            }

            launched += launch_elems;
        }

        self.maybe_log_batch_debug();
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[SuperSmoother3PoleParams],
        first_valid: usize,
        series_len: usize,
    ) -> Result<DeviceArrayF32, CudaSuperSmoother3PoleError> {
        let n_combos = combos.len();
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64MB cushion
        if !Self::will_fit(required, headroom) {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream) }
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            series_len,
            n_combos,
            first_valid,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: series_len })
    }

    pub fn supersmoother_3_pole_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SuperSmoother3PoleBatchRange,
    ) -> Result<DeviceArrayF32, CudaSuperSmoother3PoleError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, first_valid, series_len)
    }

    pub fn supersmoother_3_pole_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &SuperSmoother3PoleBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<SuperSmoother3PoleParams>), CudaSuperSmoother3PoleError> {
        let (combos, first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(data_f32, &combos, first_valid, series_len)?;
        let total = expected;
        let mut pinned = unsafe { LockedBuffer::<f32>::uninitialized(total) }
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        unsafe {
            arr.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        out.copy_from_slice(pinned.as_slice());
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn supersmoother_3_pole_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSuperSmoother3PoleError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "series_len and n_combos must be > 0".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            series_len,
            n_combos,
            first_valid,
            d_out,
        )
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SuperSmoother3PoleParams,
    ) -> Result<(Vec<i32>, usize), CudaSuperSmoother3PoleError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "period must be >= 1".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let found = fv.ok_or_else(|| {
                CudaSuperSmoother3PoleError::InvalidInput(format!(
                    "series {} contains only NaNs",
                    series
                ))
            })?;
            if (rows as i32 - found) < period as i32 {
                return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                    "series {} lacks data: need >= {}, valid = {}",
                    series,
                    period,
                    rows as i32 - found
                )));
            }
            first_valids[series] = found;
        }

        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSuperSmoother3PoleError> {
        if period == 0 {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "period must be >= 1".into(),
            ));
        }

        let mut func = self
            .module
            .get_function("supersmoother_3_pole_many_series_one_param_time_major_f32")
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => match func.suggested_launch_configuration(0, (0, 0, 0).into()) {
                Ok((_min_grid, block)) => block.max(64),
                Err(_) => 256,
            },
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
        };
        unsafe {
            (*(self as *const _ as *mut CudaSupersmoother3Pole)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }

        let device = Device::get_device(self.device_id)
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        let max_grid_x = device
            .get_attribute(DeviceAttribute::MaxGridDimX)
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))? as usize;

        let tpb = block_x as usize;
        let chunk_capacity = max_grid_x.saturating_mul(tpb);

        let mut launched = 0usize;
        while launched < cols {
            let launch_elems = (cols - launched).min(chunk_capacity);
            let blocks = (launch_elems + tpb - 1) / tpb;

            let grid: GridSize = (blocks as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().add(launched).as_raw();
                let mut period_i = period as i32;
                let mut cols_i = cols as i32; // full stride stays constant
                let mut rows_i = rows as i32;
                let mut first_ptr = d_first_valids.as_device_ptr().add(launched).as_raw();
                let mut out_ptr = d_out.as_device_ptr().add(launched).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut first_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
            }

            launched += launch_elems;
        }

        self.maybe_log_many_debug();
        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
    ) -> Result<DeviceArrayF32, CudaSuperSmoother3PoleError> {
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let first_valid_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        let d_first_valids = unsafe { DeviceBuffer::from_slice_async(first_valids, &self.stream) }
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(&d_prices, period, cols, rows, &d_first_valids, &mut d_out)?;

        self.stream
            .synchronize()
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    pub fn supersmoother_3_pole_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SuperSmoother3PoleParams,
    ) -> Result<DeviceArrayF32, CudaSuperSmoother3PoleError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn supersmoother_3_pole_many_series_one_param_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SuperSmoother3PoleParams,
        out: &mut [f32],
    ) -> Result<(), CudaSuperSmoother3PoleError> {
        if out.len() != cols * rows {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                cols * rows
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let arr = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)?;
        let total = cols * rows;
        let mut pinned = unsafe { LockedBuffer::<f32>::uninitialized(total) }
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        unsafe {
            arr.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaSuperSmoother3PoleError::Cuda(e.to_string()))?;
        out.copy_from_slice(pinned.as_slice());
        Ok(())
    }

    pub fn supersmoother_3_pole_many_series_one_param_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSuperSmoother3PoleError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSuperSmoother3PoleError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        self.launch_many_series_kernel(d_prices, period, cols, rows, d_first_valids, d_out)
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] SS3P batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaSupersmoother3Pole)).debug_batch_logged = true; }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] SS3P many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaSupersmoother3Pole)).debug_many_logged = true; }
            }
        }
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        supersmoother_3_pole_benches,
        CudaSupersmoother3Pole,
        crate::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleBatchRange,
        crate::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleParams,
        supersmoother_3_pole_batch_dev,
        supersmoother_3_pole_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::supersmoother_3_pole::SuperSmoother3PoleParams { period: Some(64) },
        "supersmoother_3_pole",
        "supersmoother_3_pole"
    );
    pub use supersmoother_3_pole_benches::bench_profiles;
}
