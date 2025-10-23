//! CUDA wrapper for Reverse RSI (price level achieving a target RSI level).
//!
//! Parity points with ALMA/Cg/Cmo wrappers:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/reverse_rsi_kernel.ptx"))
//! - Stream NON_BLOCKING, O2 JIT + fallbacks
//! - Lightweight policy + introspection hooks
//! - VRAM checks and simple chunking guards
//! - Public device entry points that return VRAM-resident DeviceArrayF32

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::reverse_rsi::{ReverseRsiBatchRange, ReverseRsiParams};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaReverseRsiError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaReverseRsiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaReverseRsiError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaReverseRsiError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaReverseRsiError {}

// Minimal policy surface mirroring other oscillators
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
pub struct CudaReverseRsiPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaReverseRsiPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
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

pub struct CudaReverseRsi {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaReverseRsiPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaReverseRsi {
    pub fn new(device_id: usize) -> Result<Self, CudaReverseRsiError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/reverse_rsi_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;

        // Favor L1 as working sets are read-mostly
        let _ = cust::context::CurrentContext::set_cache_config(CacheConfig::PreferL1);

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaReverseRsiPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaReverseRsiPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaReverseRsiPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaReverseRsiError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scen =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scen || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] ReverseRSI batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaReverseRsi)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scen =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scen || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] ReverseRSI many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaReverseRsi)).debug_many_logged = true;
                }
            }
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    // ---- Batch (one series × many params) ----

    fn expand_grid(sweep: &ReverseRsiBatchRange) -> Vec<ReverseRsiParams> {
        let (ls, le, lp) = sweep.rsi_length_range;
        let (vs, ve, vp) = sweep.rsi_level_range;
        let lengths: Vec<usize> = if lp == 0 {
            vec![ls]
        } else {
            (ls..=le).step_by(lp).collect()
        };
        let mut levels: Vec<f64> = Vec::new();
        if vp == 0.0 {
            levels.push(vs)
        } else {
            let mut x = vs;
            while x <= ve {
                levels.push(x);
                x += vp;
            }
        }
        let mut combos = Vec::with_capacity(lengths.len() * levels.len());
        for &l in &lengths {
            for &v in &levels {
                combos.push(ReverseRsiParams {
                    rsi_length: Some(l),
                    rsi_level: Some(v),
                });
            }
        }
        combos
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &ReverseRsiBatchRange,
    ) -> Result<(Vec<ReverseRsiParams>, usize, usize), CudaReverseRsiError> {
        if prices.is_empty() {
            return Err(CudaReverseRsiError::InvalidInput("empty data".into()));
        }
        let len = prices.len();
        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaReverseRsiError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaReverseRsiError::InvalidInput(
                "no parameter combos".into(),
            ));
        }
        // Validate feasibility for at least max length
        let max_len = combos
            .iter()
            .map(|p| p.rsi_length.unwrap_or(14))
            .max()
            .unwrap_or(14);
        let ema_len = (2 * max_len).saturating_sub(1);
        if len - first_valid <= ema_len {
            return Err(CudaReverseRsiError::InvalidInput(format!(
                "not enough valid data: needed > {}, have {}",
                ema_len,
                len - first_valid
            )));
        }
        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_lengths: &DeviceBuffer<i32>,
        d_levels: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaReverseRsiError> {
        let mut func: Function = self
            .module
            .get_function("reverse_rsi_batch_f32")
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        // Dynamic shared memory required by the kernel's async tiling path.
        // Keep TILE consistent with the kernel (TILE=256).
        const TILE: usize = 256;
        let shmem_bytes: usize = 4 * TILE * std::mem::size_of::<f32>();

        // Block size via occupancy suggestion or env override RRSI_BLOCK_X.
        let block_x: u32 = match std::env::var("RRSI_BLOCK_X").ok().as_deref() {
            Some("auto") | None => {
                // Feed dynamic shared memory into occupancy so suggested block size
                // respects per-block SMEM usage.
                let (_min_grid, suggested) = func
                    .suggested_launch_configuration(shmem_bytes, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
                suggested
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
        };
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        // Record selection for introspection
        unsafe {
            (*(self as *const _ as *mut CudaReverseRsi)).last_batch =
                Some(BatchKernelSelected::OneD { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut lengths_ptr = d_lengths.as_device_ptr().as_raw();
            let mut levels_ptr = d_levels.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut lengths_ptr as *mut _ as *mut c_void,
                &mut levels_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            // Pass dynamic shared memory size for kernels using extern __shared__
            self.stream
                .launch(&func, grid, block, (shmem_bytes as u32), args)
                .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    pub fn reverse_rsi_batch_dev(
        &self,
        prices: &[f32],
        sweep: &ReverseRsiBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<ReverseRsiParams>), CudaReverseRsiError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(prices, sweep)?;
        let rows = combos.len();

        // VRAM estimate: prices + lengths + levels + out
        let bytes = len * std::mem::size_of::<f32>()
            + rows * std::mem::size_of::<i32>()
            + rows * std::mem::size_of::<f32>()
            + rows * len * std::mem::size_of::<f32>();
        let headroom = 64 * 1024 * 1024; // 64MB
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaReverseRsiError::InvalidInput(
                "insufficient VRAM for reverse_rsi batch".into(),
            ));
        }

        let lengths_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.rsi_length.unwrap_or(14) as i32)
            .collect();
        let levels_f32: Vec<f32> = combos
            .iter()
            .map(|c| c.rsi_level.unwrap_or(50.0) as f32)
            .collect();

        // Pinned host buffers + async H2D
        let h_prices = LockedBuffer::from_slice(prices)
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        let h_lens = LockedBuffer::from_slice(&lengths_i32)
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        let h_lvls = LockedBuffer::from_slice(&levels_f32)
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;

        let mut d_prices = unsafe { DeviceBuffer::<f32>::uninitialized_async(len, &self.stream) }
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        let mut d_lengths = unsafe { DeviceBuffer::<i32>::uninitialized_async(rows, &self.stream) }
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        let mut d_levels = unsafe { DeviceBuffer::<f32>::uninitialized_async(rows, &self.stream) }
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        let mut d_out =
            unsafe { DeviceBuffer::<f32>::uninitialized_async(rows * len, &self.stream) }
                .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;

        unsafe {
            d_prices
                .async_copy_from(&h_prices, &self.stream)
                .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
            d_lengths
                .async_copy_from(&h_lens, &self.stream)
                .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
            d_levels
                .async_copy_from(&h_lvls, &self.stream)
                .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        }

        self.launch_batch_kernel(
            &d_prices,
            &d_lengths,
            &d_levels,
            len,
            rows,
            first_valid,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;

        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows,
                cols: len,
            },
            combos,
        ))
    }

    // ---- Many-series × one-param (time-major) ----

    fn prepare_many_series_inputs(
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &ReverseRsiParams,
    ) -> Result<(Vec<i32>, i32, f32), CudaReverseRsiError> {
        if prices_tm.len() != cols * rows {
            return Err(CudaReverseRsiError::InvalidInput(
                "time-major input has wrong size".into(),
            ));
        }
        let period = params.rsi_length.unwrap_or(14) as i32;
        let level = params.rsi_level.unwrap_or(50.0) as f32;
        if !(level > 0.0 && level < 100.0) || period <= 0 {
            return Err(CudaReverseRsiError::InvalidInput("invalid params".into()));
        }
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for r in 0..rows {
                let v = prices_tm[r * cols + s];
                if !v.is_nan() {
                    fv = r as i32;
                    break;
                }
            }
            if fv < 0 {
                fv = 0;
            }
            first_valids[s] = fv;
        }
        Ok((first_valids, period, level))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: i32,
        level: f32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaReverseRsiError> {
        let mut func: Function = self
            .module
            .get_function("reverse_rsi_many_series_one_param_f32")
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        let _ = func.set_cache_config(CacheConfig::PreferL1);

        let block_x: u32 = match std::env::var("RRSI_MANY_BLOCK_X").ok().as_deref() {
            Some("auto") | None => {
                let (_min, suggested) = func
                    .suggested_launch_configuration(0, BlockSize::xyz(0, 0, 0))
                    .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
                suggested
            }
            Some(s) => s.parse::<u32>().ok().filter(|&v| v > 0).unwrap_or(128),
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            (*(self as *const _ as *mut CudaReverseRsi)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period;
            let mut level_f = level;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut level_f as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn reverse_rsi_many_series_one_param_time_major_dev(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &ReverseRsiParams,
    ) -> Result<DeviceArrayF32, CudaReverseRsiError> {
        let (first_valids, period, level) =
            Self::prepare_many_series_inputs(prices_tm, cols, rows, params)?;

        // VRAM estimate
        let elems = cols * rows;
        let bytes = elems * std::mem::size_of::<f32>() /* in */
            + cols * std::mem::size_of::<i32>()        /* firsts */
            + elems * std::mem::size_of::<f32>(); /* out */
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(bytes, headroom) {
            return Err(CudaReverseRsiError::InvalidInput(
                "insufficient VRAM for reverse_rsi many-series".into(),
            ));
        }

        // Use pinned host buffers + async copies for higher throughput and true async behavior
        let h_prices_tm = LockedBuffer::from_slice(prices_tm)
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        let h_first = LockedBuffer::from_slice(&first_valids)
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;

        let mut d_prices_tm = unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        let mut d_first = unsafe { DeviceBuffer::<i32>::uninitialized_async(cols, &self.stream) }
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        let mut d_out_tm = unsafe { DeviceBuffer::<f32>::uninitialized_async(elems, &self.stream) }
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;

        unsafe {
            d_prices_tm
                .async_copy_from(&h_prices_tm, &self.stream)
                .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
            d_first
                .async_copy_from(&h_first, &self.stream)
                .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;
        }

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_first,
            cols,
            rows,
            period,
            level,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaReverseRsiError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP_L: usize = 100; // lengths
    const PARAM_SWEEP_V: usize = 50; // levels
    const MANY_SERIES_COLS: usize = 256;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let rows = PARAM_SWEEP_L * PARAM_SWEEP_V;
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let params_bytes = rows * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let out_bytes = rows * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + params_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let first_bytes = MANY_SERIES_COLS * std::mem::size_of::<i32>();
        in_bytes + out_bytes + first_bytes + 64 * 1024 * 1024
    }

    struct BatchState {
        cuda: CudaReverseRsi,
        price: Vec<f32>,
        sweep: ReverseRsiBatchRange,
    }
    impl CudaBenchState for BatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .reverse_rsi_batch_dev(&self.price, &self.sweep)
                .expect("reverse_rsi_batch_dev");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaReverseRsi::new(0).expect("cuda");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = ReverseRsiBatchRange {
            rsi_length_range: (5, 5 + PARAM_SWEEP_L as usize - 1, 1),
            rsi_level_range: (10.0, 10.0 + PARAM_SWEEP_V as f64 - 1.0, 1.0),
        };
        Box::new(BatchState { cuda, price, sweep })
    }

    struct ManyState {
        cuda: CudaReverseRsi,
        data_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        params: ReverseRsiParams,
    }
    impl CudaBenchState for ManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .reverse_rsi_many_series_one_param_time_major_dev(
                    &self.data_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                )
                .expect("reverse_rsi_many_series_one_param_time_major_dev");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaReverseRsi::new(0).expect("cuda");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = ReverseRsiParams {
            rsi_length: Some(14),
            rsi_level: Some(50.0),
        };
        Box::new(ManyState {
            cuda,
            data_tm,
            cols,
            rows,
            params,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "reverse_rsi",
                "one_series_many_params",
                "reverse_rsi_cuda_batch_dev",
                "1m_x_5000",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "reverse_rsi",
                "many_series_one_param",
                "reverse_rsi_cuda_many_series_one_param",
                "256x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
