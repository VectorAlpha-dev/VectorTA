//! CUDA wrapper for CCI (Commodity Channel Index)
//!
//! Parity goals:
//! - API mirrors ALMA-style wrappers: PTX load via OUT_DIR, NON_BLOCKING stream
//! - VRAM estimation and basic headroom guard
//! - Batch (one series × many params) and many-series × one-param (time-major)
//! - Warmup/NaN semantics identical to scalar CCI

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::cci::{CciBatchRange, CciParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaCciError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaCciError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaCciError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaCciError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaCciError {}

pub struct CudaCci {
    module: Module,
    stream: Stream,
    _context: Context,
    debug_batch_logged: bool,
}

impl CudaCci {
    pub fn new(device_id: usize) -> Result<Self, CudaCciError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaCciError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaCciError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaCciError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cci_kernel.ptx"));
        let opt = match env::var("CCI_JIT_OPT").ok().as_deref() {
            Some("O0") => OptLevel::O0,
            Some("O1") => OptLevel::O1,
            Some("O2") => OptLevel::O2,
            Some("O3") => OptLevel::O3,
            Some("O4") => OptLevel::O4,
            _ => OptLevel::O2,
        };
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(opt),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaCciError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaCciError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            debug_batch_logged: false,
        })
    }

    #[inline]
    fn use_async() -> bool {
        match env::var("CCI_ASYNC") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _)) = cust::memory::mem_get_info() {
            required_bytes.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    fn expand_periods(range: &CciBatchRange) -> Vec<CciParams> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return vec![CciParams {
                period: Some(start),
            }];
        }
        let mut v = Vec::new();
        let mut p = start;
        while p <= end {
            v.push(CciParams { period: Some(p) });
            p = p.saturating_add(step);
        }
        v
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &CciBatchRange,
    ) -> Result<(Vec<CciParams>, usize, usize), CudaCciError> {
        if data_f32.is_empty() {
            return Err(CudaCciError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaCciError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaCciError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        for c in &combos {
            let p = c.period.unwrap_or(0);
            if p == 0 {
                return Err(CudaCciError::InvalidInput("period must be >=1".into()));
            }
            if p > len {
                return Err(CudaCciError::InvalidInput(format!(
                    "period {} > len {}",
                    p, len
                )));
            }
            if len - first_valid < p {
                return Err(CudaCciError::InvalidInput(format!(
                    "not enough valid data for period {} (tail={})",
                    p,
                    len - first_valid
                )));
            }
        }
        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
        periods_offset: usize,
        out_offset_elems: usize,
    ) -> Result<(), CudaCciError> {
        let mut func: Function = self
            .module
            .get_function("cci_batch_f32")
            .map_err(|e| CudaCciError::Cuda(e.to_string()))?;

        // Use CUDA-suggested block size for occupancy; override with CCI_BLOCK_X
        let block_x: u32 = match env::var("CCI_BLOCK_X").ok().as_deref() {
            Some("auto") => {
                let (_mg, suggest) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
                suggest
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
            None => {
                let (_mg, suggest) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
                suggest
            }
        };
        // One block per combo (kernel uses 1 active lane per block for simplicity)
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x.max(64), 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw()
                + (periods_offset * std::mem::size_of::<i32>()) as u64;
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw()
                + (out_offset_elems * std::mem::size_of::<f32>()) as u64;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn cci_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &CciBatchRange,
    ) -> Result<DeviceArrayF32, CudaCciError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let rows = combos.len();
        // VRAM estimate: inputs + params + outputs + headroom
        let bytes = len * std::mem::size_of::<f32>()
            + rows * std::mem::size_of::<i32>()
            + rows * len * std::mem::size_of::<f32>();
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaCciError::InvalidInput(format!(
                "insufficient VRAM: need ~{} MB (incl headroom)",
                (bytes + headroom) / (1024 * 1024)
            )));
        }

        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();

        if Self::use_async() {
            let h_prices = LockedBuffer::from_slice(data_f32)
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            let h_periods = LockedBuffer::from_slice(&periods)
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            let mut d_prices =
                unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            let mut d_periods =
                unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            let mut d_out =
                unsafe { DeviceBuffer::<f32>::uninitialized_async(rows * len, &self.stream) }
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            unsafe {
                d_prices
                    .async_copy_from(&h_prices, &self.stream)
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
                d_periods
                    .async_copy_from(&h_periods, &self.stream)
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            }
            // Chunk launches if rows is extremely large; keep <= 65_535 blocks as a conservative limit
            let max_blocks: usize = 65_535;
            let mut launched = 0usize;
            while launched < rows {
                let n_this = std::cmp::min(max_blocks, rows - launched);
                let periods_off = launched;
                let out_off = launched * len;
                self.launch_batch_kernel(
                    &d_prices,
                    &d_periods,
                    len,
                    n_this,
                    first_valid,
                    &mut d_out,
                    periods_off,
                    out_off,
                )?;
                launched += n_this;
            }
            self.stream
                .synchronize()
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            if !self.debug_batch_logged && env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
                eprintln!(
                    "[cci] batch kernel: Plain, block_x=auto, chunked={} rows",
                    rows
                );
            }
            Ok(DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: len,
            })
        } else {
            let d_prices = DeviceBuffer::from_slice(data_f32)
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            let d_periods = DeviceBuffer::from_slice(&periods)
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(rows * len) }
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            let max_blocks: usize = 65_535;
            let mut launched = 0usize;
            while launched < rows {
                let n_this = std::cmp::min(max_blocks, rows - launched);
                self.launch_batch_kernel(
                    &d_prices,
                    &d_periods,
                    len,
                    n_this,
                    first_valid,
                    &mut d_out,
                    launched,
                    launched * len,
                )?;
                launched += n_this;
            }
            if !self.debug_batch_logged && env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
                eprintln!(
                    "[cci] batch kernel: Plain, block_x=auto, chunked={} rows",
                    rows
                );
            }
            Ok(DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: len,
            })
        }
    }

    fn prepare_many_series(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<(Vec<i32>, usize), CudaCciError> {
        if data_tm_f32.len() != cols * rows {
            return Err(CudaCciError::InvalidInput(
                "time-major buffer shape mismatch".into(),
            ));
        }
        if period == 0 || period > rows {
            return Err(CudaCciError::InvalidInput("invalid period".into()));
        }
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for r in 0..rows {
                if !data_tm_f32[r * cols + s].is_nan() {
                    fv = Some(r);
                    break;
                }
            }
            let fv =
                fv.ok_or_else(|| CudaCciError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv < period {
                return Err(CudaCciError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail={})",
                    s,
                    period,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }
        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCciError> {
        let mut func: Function = self
            .module
            .get_function("cci_many_series_one_param_f32")
            .map_err(|e| CudaCciError::Cuda(e.to_string()))?;

        let block_x: u32 = match env::var("CCI_MS_BLOCK_X").ok().as_deref() {
            Some("auto") => {
                let (_mg, s) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
                s
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
            None => {
                let (_mg, s) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
                s
            }
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn cci_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
    ) -> Result<DeviceArrayF32, CudaCciError> {
        let (first_valids, period) = Self::prepare_many_series(data_tm_f32, cols, rows, period)?;
        if Self::use_async() {
            let h_prices = LockedBuffer::from_slice(data_tm_f32)
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            let h_first = LockedBuffer::from_slice(&first_valids)
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            let mut d_prices =
                unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            let mut d_first =
                unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream) }
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            let mut d_out =
                unsafe { DeviceBuffer::<f32>::uninitialized_async(cols * rows, &self.stream) }
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            unsafe {
                d_prices
                    .async_copy_from(&h_prices, &self.stream)
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
                d_first
                    .async_copy_from(&h_first, &self.stream)
                    .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            }
            self.launch_many_series_kernel(&d_prices, &d_first, cols, rows, period, &mut d_out)?;
            self.stream
                .synchronize()
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            Ok(DeviceArrayF32 {
                buf: d_out,
                rows,
                cols,
            })
        } else {
            let d_prices = DeviceBuffer::from_slice(data_tm_f32)
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            let d_first = DeviceBuffer::from_slice(&first_valids)
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
                .map_err(|e| CudaCciError::Cuda(e.to_string()))?;
            self.launch_many_series_kernel(&d_prices, &d_first, cols, rows, period, &mut d_out)?;
            Ok(DeviceArrayF32 {
                buf: d_out,
                rows,
                cols,
            })
        }
    }
}

// ---------- Benches (wrapper-owned) ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 200; // keep runtime reasonable (O(period) MAD per step)

    fn bytes_required() -> usize {
        let in_bytes = SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct CciBatchState {
        cuda: CudaCci,
        data: Vec<f32>,
        sweep: CciBatchRange,
    }
    impl CudaBenchState for CciBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .cci_batch_dev(&self.data, &self.sweep)
                .expect("cci batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaCci::new(0).expect("cuda cci");
        let data = gen_series(SERIES_LEN);
        let sweep = CciBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        Box::new(CciBatchState { cuda, data, sweep })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![CudaBenchScenario::new(
            "cci",
            "one_series_many_params",
            "cci_cuda_batch_dev",
            "1m_x_200",
            prep_one_series_many_params,
        )
        .with_sample_size(8)
        .with_mem_required(bytes_required())]
    }
}
