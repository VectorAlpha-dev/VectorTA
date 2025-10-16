//! CUDA wrapper for Percentile Nearest Rank (PNR).
//!
//! Parity goals with ALMA/CWMA wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/percentile_nearest_rank_kernel.ptx"))
//! - NON_BLOCKING stream
//! - Simple policy surface with introspection + one-time BENCH_DEBUG logging
//! - VRAM estimate + headroom check; chunking not needed for this baseline
//! - Public device entry points:
//!     - one-series × many-params (batch)
//!     - many-series × one-param (time-major)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::percentile_nearest_rank::{
    PercentileNearestRankBatchRange, PercentileNearestRankParams,
};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaPnrError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaPnrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaPnrError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaPnrError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaPnrError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaPnrPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaPnrPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaPercentileNearestRank {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaPnrPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaPercentileNearestRank {
    pub fn new(device_id: usize) -> Result<Self, CudaPnrError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaPnrError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/percentile_nearest_rank_kernel.ptx"));
        // Stable JIT options with fallbacks
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaPnrError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaPnrPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaPnrError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaPnrError::Cuda(e.to_string()))
    }

    pub fn set_policy(&mut self, policy: CudaPnrPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaPnrPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    #[inline]
    fn maybe_log(sel: Option<impl fmt::Debug>, which: &str, once_flag: &AtomicBool, printed: &mut bool) {
        if *printed { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(s) = sel {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !once_flag.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] PNR {} selected kernel: {:?}", which, s);
                }
                *printed = true;
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") { Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"), Err(_) => true }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() { required.saturating_add(headroom) <= free } else { true }
    }

    // -------- Helpers --------

    fn axis_usize(axis: (usize, usize, usize)) -> Vec<usize> {
        let (start, end, step) = axis;
        if step == 0 || start == end { return vec![start]; }
        if start > end { return Vec::new(); }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64(axis: (f64, f64, f64)) -> Vec<f64> {
        let (start, end, step) = axis;
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return vec![start]; }
        if start > end { return Vec::new(); }
        let mut out = Vec::new();
        let mut v = start;
        let lim = end + step.abs() * 1e-12;
        while v <= lim { out.push(v); v += step; }
        out
    }

    fn expand_grid(r: &PercentileNearestRankBatchRange) -> Vec<PercentileNearestRankParams> {
        let lengths = Self::axis_usize(r.length);
        let percentages = Self::axis_f64(r.percentage);
        let mut combos = Vec::with_capacity(lengths.len() * percentages.len());
        for &l in &lengths {
            for &p in &percentages {
                combos.push(PercentileNearestRankParams { length: Some(l), percentage: Some(p) });
            }
        }
        combos
    }

    // -------- Batch: one-series × many params --------
    pub fn pnr_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &PercentileNearestRankBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<PercentileNearestRankParams>), CudaPnrError> {
        if data_f32.is_empty() { return Err(CudaPnrError::InvalidInput("empty data".into())); }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaPnrError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() { return Err(CudaPnrError::InvalidInput("no parameter combinations".into())); }
        let max_len = combos.iter().map(|c| c.length.unwrap_or(15)).max().unwrap();
        if len - first_valid < max_len { return Err(CudaPnrError::InvalidInput("not enough valid data".into())); }

        // Inputs
        let periods: Vec<i32> = combos.iter().map(|c| c.length.unwrap_or(15) as i32).collect();
        let percs: Vec<f32> = combos.iter().map(|c| c.percentage.unwrap_or(50.0) as f32).collect();

        let prices_bytes = len * core::mem::size_of::<f32>();
        let periods_bytes = periods.len() * core::mem::size_of::<i32>();
        let percs_bytes = percs.len() * core::mem::size_of::<f32>();
        let out_bytes = combos.len() * len * core::mem::size_of::<f32>();
        let scratch_bytes = combos.len() * max_len * core::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + percs_bytes + out_bytes + scratch_bytes;
        let headroom = 64 * 1024 * 1024; // ~64MB
        if !Self::will_fit(required, headroom) {
            return Err(CudaPnrError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Pinned host buffers
        let h_prices = LockedBuffer::from_slice(data_f32).map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        let h_periods = LockedBuffer::from_slice(&periods).map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        let h_percs = LockedBuffer::from_slice(&percs).map_err(|e| CudaPnrError::Cuda(e.to_string()))?;

        // Device buffers
        let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }.map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        let mut d_periods: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(periods.len()) }.map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        let mut d_percs: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(percs.len()) }.map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(combos.len() * len) }.map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        let mut d_scratch: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(combos.len() * max_len) }.map_err(|e| CudaPnrError::Cuda(e.to_string()))?;

        // Async H2D
        unsafe {
            d_prices.async_copy_from(h_prices.as_slice(), &self.stream).map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
            d_periods.async_copy_from(h_periods.as_slice(), &self.stream).map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
            d_percs.async_copy_from(h_percs.as_slice(), &self.stream).map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        }

        // Launch
        let func = self
            .module
            .get_function("percentile_nearest_rank_batch_f32")
            .map_err(|e| CudaPnrError::Cuda(e.to_string()))?;

        let block_x = match self.policy.batch { BatchKernelPolicy::Auto => 128, BatchKernelPolicy::OneD { block_x } => block_x };
        let grid_x = ((combos.len() as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut percs_ptr = d_percs.as_device_ptr().as_raw();
            let mut series_len_i = len as i32;
            let mut combos_i = combos.len() as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut scratch_ptr = d_scratch.as_device_ptr().as_raw();
            let mut max_len_i = max_len as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut percs_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
                &mut scratch_ptr as *mut _ as *mut c_void,
                &mut max_len_i as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        }
        // Track selection for debugging
        let sel = BatchKernelSelected::OneD { block_x };
        // Track selection for introspection and optional BENCH_DEBUG logging
        unsafe { (*(self as *const _ as *mut CudaPercentileNearestRank)).last_batch = Some(sel); }
        static ONCE: AtomicBool = AtomicBool::new(false);
        let mut printed = false;
        Self::maybe_log(Some(sel), "batch", &ONCE, &mut printed);
        // Wait for completion
        self.stream
            .synchronize()
            .map_err(|e| CudaPnrError::Cuda(e.to_string()))?;

        Ok((DeviceArrayF32 { buf: d_out, rows: combos.len(), cols: len }, combos))
    }

    // -------- Many-series × one-param (time-major) --------
    pub fn pnr_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        length: usize,
        percentage: f64,
    ) -> Result<DeviceArrayF32, CudaPnrError> {
        if cols == 0 || rows == 0 { return Err(CudaPnrError::InvalidInput("empty shape".into())); }
        if data_tm_f32.len() != cols * rows { return Err(CudaPnrError::InvalidInput("time-major input shape mismatch".into())); }
        if length == 0 || length > rows { return Err(CudaPnrError::InvalidInput("invalid length".into())); }

        // Per-series first_valid (index of first non-NaN)
        let mut firsts = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() { fv = t as i32; break; }
            }
            if fv < 0 { return Err(CudaPnrError::InvalidInput("all values NaN for a series".into())); }
            if (rows as i32 - fv) < (length as i32) {
                return Err(CudaPnrError::InvalidInput("not enough valid data for a series".into()));
            }
            firsts[s] = fv;
        }

        let prices_bytes = cols * rows * core::mem::size_of::<f32>();
        let out_bytes = cols * rows * core::mem::size_of::<f32>();
        let firsts_bytes = cols * core::mem::size_of::<i32>();
        let scratch_bytes = cols * length * core::mem::size_of::<f32>();
        let required = prices_bytes + out_bytes + firsts_bytes + scratch_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaPnrError::InvalidInput("insufficient free VRAM for workload".into()));
        }

        // Pinned host buffers
        let h_prices = LockedBuffer::from_slice(data_tm_f32).map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        let h_firsts = LockedBuffer::from_slice(&firsts).map_err(|e| CudaPnrError::Cuda(e.to_string()))?;

        let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        let mut d_firsts: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(cols) }.map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        let mut d_scratch: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * length) }.map_err(|e| CudaPnrError::Cuda(e.to_string()))?;

        unsafe {
            d_prices.async_copy_from(h_prices.as_slice(), &self.stream).map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
            d_firsts.async_copy_from(h_firsts.as_slice(), &self.stream).map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        }

        let func = self
            .module
            .get_function("percentile_nearest_rank_many_series_one_param_time_major_f32")
            .map_err(|e| CudaPnrError::Cuda(e.to_string()))?;

        let block_x = match self.policy.many_series { ManySeriesKernelPolicy::Auto => 128, ManySeriesKernelPolicy::OneD { block_x } => block_x };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut length_i = length as i32;
            let mut perc_f = percentage as f32;
            let mut firsts_ptr = d_firsts.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut scratch_ptr = d_scratch.as_device_ptr().as_raw();
            let mut max_len_i = length as i32;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut length_i as *mut _ as *mut c_void,
                &mut perc_f as *mut _ as *mut c_void,
                &mut firsts_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
                &mut scratch_ptr as *mut _ as *mut c_void,
                &mut max_len_i as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaPnrError::Cuda(e.to_string()))?;
        }

        let sel = ManySeriesKernelSelected::OneD { block_x };
        unsafe { (*(self as *const _ as *mut CudaPercentileNearestRank)).last_many = Some(sel); }
        static ONCE: AtomicBool = AtomicBool::new(false);
        let mut printed = false;
        Self::maybe_log(Some(sel), "many-series", &ONCE, &mut printed);

        self.stream
            .synchronize()
            .map_err(|e| CudaPnrError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }
}

// ---------------- Bench integration ----------------
#[cfg(test)]
mod benches_dummy_compile_only {}

#[cfg(feature = "cuda")]
pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    struct PnrBatchState {
        cuda: CudaPercentileNearestRank,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_percs: DeviceBuffer<f32>,
        d_scratch: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        n_combos: usize,
        first_valid: i32,
        max_len: i32,
        block: BlockSize,
        grid: GridSize,
    }
    impl CudaBenchState for PnrBatchState {
        fn launch(&mut self) { unsafe {
            let func = self
                .cuda
                .module
                .get_function("percentile_nearest_rank_batch_f32")
                .expect("get_function pnr batch");
            let mut prices_ptr = self.d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = self.d_periods.as_device_ptr().as_raw();
            let mut percs_ptr = self.d_percs.as_device_ptr().as_raw();
            let mut series_len_i = self.len as i32;
            let mut combos_i = self.n_combos as i32;
            let mut first_valid_i = self.first_valid;
            let mut out_ptr = self.d_out.as_device_ptr().as_raw();
            let mut scratch_ptr = self.d_scratch.as_device_ptr().as_raw();
            let mut max_len_i = self.max_len;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut percs_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
                &mut scratch_ptr as *mut _ as *mut c_void,
                &mut max_len_i as *mut _ as *mut c_void,
            ];
            let _ = self.cuda.stream.launch(&func, self.grid, self.block, 0, args);
            let _ = self.cuda.stream.synchronize();
        }}
    }

    struct PnrManyState {
        cuda: CudaPercentileNearestRank,
        d_prices: DeviceBuffer<f32>,
        d_firsts: DeviceBuffer<i32>,
        d_scratch: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        length: i32,
        perc: f32,
        block: BlockSize,
        grid: GridSize,
    }
    impl CudaBenchState for PnrManyState {
        fn launch(&mut self) { unsafe {
            let func = self
                .cuda
                .module
                .get_function("percentile_nearest_rank_many_series_one_param_time_major_f32")
                .expect("get_function pnr many");
            let mut prices_ptr = self.d_prices.as_device_ptr().as_raw();
            let mut cols_i = self.cols as i32;
            let mut rows_i = self.rows as i32;
            let mut length_i = self.length;
            let mut perc_f = self.perc;
            let mut firsts_ptr = self.d_firsts.as_device_ptr().as_raw();
            let mut out_ptr = self.d_out.as_device_ptr().as_raw();
            let mut scratch_ptr = self.d_scratch.as_device_ptr().as_raw();
            let mut max_len_i = self.length;
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut length_i as *mut _ as *mut c_void,
                &mut perc_f as *mut _ as *mut c_void,
                &mut firsts_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
                &mut scratch_ptr as *mut _ as *mut c_void,
                &mut max_len_i as *mut _ as *mut c_void,
            ];
            let _ = self.cuda.stream.launch(&func, self.grid, self.block, 0, args);
            let _ = self.cuda.stream.synchronize();
        }}
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
        let mut v = Vec::new();

        // Batch: 1 series × many params (moderate sizes to avoid OOM/timeouts)
        v.push(CudaBenchScenario::new(
            "pnr",
            "one_series_many_params",
            "pnr/batch",
            "pnr_batch/100k",
            || {
                let len = 100_000usize;
                let prices = gen_series(len);
                let periods: Vec<i32> = (10..=50).step_by(10).flat_map(|l| std::iter::repeat(l as i32).take(3)).collect();
                let percs: Vec<f32> = [25.0f32, 50.0, 75.0].repeat(5);
                let n_combos = periods.len();
                let first_valid = prices.iter().position(|v| !v.is_nan()).unwrap_or(0) as i32;
                let max_len = *periods.iter().max().unwrap_or(&10);

                let cuda = CudaPercentileNearestRank::new(0).expect("cuda ctx");
                let block_x: u32 = 128;
                let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
                let grid: GridSize = (grid_x.max(1), 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();

                let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }.unwrap();
                let mut d_periods: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(n_combos) }.unwrap();
                let mut d_percs: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos) }.unwrap();
                let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * len) }.unwrap();
                let mut d_scratch: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * (max_len as usize)) }.unwrap();

                let hp = LockedBuffer::from_slice(&prices).unwrap();
                let hperiods = LockedBuffer::from_slice(&periods).unwrap();
                let hpercs = LockedBuffer::from_slice(&percs).unwrap();
                unsafe {
                    d_prices.async_copy_from(hp.as_slice(), &cuda.stream).unwrap();
                    d_periods.async_copy_from(hperiods.as_slice(), &cuda.stream).unwrap();
                    d_percs.async_copy_from(hpercs.as_slice(), &cuda.stream).unwrap();
                }
                let _ = cuda.stream.synchronize();

                Box::new(PnrBatchState { cuda, d_prices, d_periods, d_percs, d_scratch, d_out, len, n_combos, first_valid, max_len, block, grid }) as Box<dyn CudaBenchState>
            }
        ).with_mem_required( (100_000 * 4) + (15 * 100_000 * 4) + (15 * 64 * 4) ));

        // Many-series: time-major, one param
        v.push(CudaBenchScenario::new(
            "pnr",
            "many_series_one_param",
            "pnr/many_series",
            "pnr_many/cols=128,rows=4096",
            || {
                let cols = 128usize;
                let rows = 4096usize;
                let prices = gen_time_major_prices(cols, rows);
                let length = 21i32;
                let perc = 50.0f32;
                let mut firsts = vec![0i32; cols];
                for s in 0..cols {
                    firsts[s] = (0..rows).find(|&t| !prices[t * cols + s].is_nan()).unwrap_or(0) as i32;
                }
                let cuda = CudaPercentileNearestRank::new(0).expect("cuda ctx");
                let block_x: u32 = 128;
                let grid_x = ((cols as u32) + block_x - 1) / block_x;
                let grid: GridSize = (grid_x.max(1), 1, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();

                let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.unwrap();
                let mut d_firsts: DeviceBuffer<i32> = unsafe { DeviceBuffer::uninitialized(cols) }.unwrap();
                let mut d_scratch: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * (length as usize)) }.unwrap();
                let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.unwrap();

                let hp = LockedBuffer::from_slice(&prices).unwrap();
                let hf = LockedBuffer::from_slice(&firsts).unwrap();
                unsafe {
                    d_prices.async_copy_from(hp.as_slice(), &cuda.stream).unwrap();
                    d_firsts.async_copy_from(hf.as_slice(), &cuda.stream).unwrap();
                }
                let _ = cuda.stream.synchronize();

                Box::new(PnrManyState { cuda, d_prices, d_firsts, d_scratch, d_out, cols, rows, length, perc, block, grid }) as Box<dyn CudaBenchState>
            }
        ).with_mem_required( (128*4096*4) + (128*4096*4) + (128*21*4) ));

        v
    }
}
