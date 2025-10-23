//! CUDA wrapper for QQE (Quantitative Qualitative Estimation).
//!
//! Parity goals with ALMA/CWMA wrappers:
//! - PTX load via DetermineTargetFromContext + OptLevel O2, with conservative fallbacks.
//! - NON_BLOCKING stream.
//! - Policy enums and last-selected kernel logging when BENCH_DEBUG=1.
//! - VRAM checks with ~64MB headroom and grid.y chunking (≤ 65_535) for batch.
//! - Output packing: rows = 2 * n_combos (FAST row, then SLOW row per combo).

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;
use crate::indicators::qqe::{QqeBatchRange, QqeParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaQqeError {
    Cuda(String),
    InvalidInput(String),
}
impl fmt::Display for CudaQqeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaQqeError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaQqeError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaQqeError {}

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
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaQqe {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy_batch: BatchKernelPolicy,
    policy_many: ManySeriesKernelPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaQqe {
    // warp-friendly clamp/alignment for block sizes (32..=1024)
    #[inline]
    fn warp_align(x: u32) -> u32 {
        let clamped = x.clamp(32, 1024);
        ((clamped + 31) / 32) * 32
    }
    pub fn new(device_id: usize) -> Result<Self, CudaQqeError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/qqe_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaQqeError::Cuda(e.to_string()))?
                }
            }
        };
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy_batch: BatchKernelPolicy::Auto,
            policy_many: ManySeriesKernelPolicy::Auto,
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    pub fn set_batch_policy(&mut self, p: BatchKernelPolicy) { self.policy_batch = p; }
    #[inline]
    pub fn set_many_series_policy(&mut self, p: ManySeriesKernelPolicy) { self.policy_many = p; }
    #[inline]
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    #[inline]
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaQqeError> {
        self.stream.synchronize().map_err(|e| CudaQqeError::Cuda(e.to_string()))
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] QQE batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaQqe)).debug_batch_logged = true; }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] QQE many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaQqe)).debug_many_logged = true; }
            }
        }
    }

    // ---- Helpers ----
    fn device_mem_ok(bytes: usize) -> bool {
        match mem_get_info() { Ok((free, _)) => bytes.saturating_add(64 * 1024 * 1024) <= free, Err(_) => true }
    }
    fn first_valid_f32(series: &[f32]) -> Result<usize, CudaQqeError> {
        if series.is_empty() { return Err(CudaQqeError::InvalidInput("empty series".into())); }
        series.iter().position(|x| x.is_finite()).ok_or_else(|| CudaQqeError::InvalidInput("all values are NaN".into()))
    }
    fn expand_grid(range: &QqeBatchRange) -> Vec<QqeParams> {
        fn axis_usize(t: (usize, usize, usize)) -> Vec<usize> { if t.2 == 0 || t.0 == t.1 { vec![t.0] } else { (t.0..=t.1).step_by(t.2).collect() } }
        fn axis_f64(t: (f64, f64, f64)) -> Vec<f64> {
            if t.2.abs() < 1e-12 || (t.0 - t.1).abs() < 1e-12 { vec![t.0] } else {
                let mut v = Vec::new(); let mut x = t.0; while x <= t.1 + 1e-12 { v.push(x); x += t.2; } v
            }
        }
        let rs = axis_usize(range.rsi_period);
        let sm = axis_usize(range.smoothing_factor);
        let ff = axis_f64(range.fast_factor);
        let mut out = Vec::with_capacity(rs.len() * sm.len() * ff.len());
        for &r in &rs { for &s in &sm { for &k in &ff { out.push(QqeParams { rsi_period: Some(r), smoothing_factor: Some(s), fast_factor: Some(k) }); } } }
        out
    }

    // ---- Batch: one series × many params ----
    pub fn qqe_batch_dev(&self, prices_f32: &[f32], sweep: &QqeBatchRange) -> Result<(DeviceArrayF32, Vec<QqeParams>), CudaQqeError> {
        if prices_f32.is_empty() { return Err(CudaQqeError::InvalidInput("empty price input".into())); }
        let first_valid = Self::first_valid_f32(prices_f32)?;
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() { return Err(CudaQqeError::InvalidInput("empty parameter sweep".into())); }
        let len = prices_f32.len();

        // Validate: need at least rsi+ema-1 samples after first_valid
        let mut worst_needed = 0usize;
        for c in &combos { let need = c.rsi_period.unwrap() + c.smoothing_factor.unwrap(); worst_needed = worst_needed.max(need); }
        if len - first_valid < worst_needed { return Err(CudaQqeError::InvalidInput("not enough valid data for warmup".into())); }

        // VRAM: prices + (rsi+ema+fast arrays) + outputs (2 * rows * len)
        let rows = combos.len();
        let req_bytes = len*4 + rows*(4+4+4) + (2*rows*len)*4;
        if !Self::device_mem_ok(req_bytes) { return Err(CudaQqeError::InvalidInput("insufficient device memory".into())); }

        let d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::from_slice_async(prices_f32, &self.stream) }
            .map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        let rsi_i32: Vec<i32> = combos.iter().map(|c| c.rsi_period.unwrap() as i32).collect();
        let ema_i32: Vec<i32> = combos.iter().map(|c| c.smoothing_factor.unwrap() as i32).collect();
        let fast_f32: Vec<f32> = combos.iter().map(|c| c.fast_factor.unwrap() as f32).collect();
        // async param uploads to avoid implicit syncs
        let d_rsi: DeviceBuffer<i32> = unsafe { DeviceBuffer::from_slice_async(&rsi_i32, &self.stream) }
            .map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        let d_ema: DeviceBuffer<i32> = unsafe { DeviceBuffer::from_slice_async(&ema_i32, &self.stream) }
            .map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        let d_fast: DeviceBuffer<f32> = unsafe { DeviceBuffer::from_slice_async(&fast_f32, &self.stream) }
            .map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(2 * rows * len, &self.stream) }
            .map_err(|e| CudaQqeError::Cuda(e.to_string()))?;

        // Kernel launch: grid = (grid_x=1, grid_y=chunk_rows), block.x warp-aligned
        let mut block_x = match self.policy_batch { BatchKernelPolicy::Plain { block_x } => block_x, BatchKernelPolicy::Auto => 256 };
        block_x = Self::warp_align(block_x);
        // Cache function lookup
        let func = self.module.get_function("qqe_batch_f32").map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        const MAX_Y: usize = 65_535;
        let mut base = 0usize;
        while base < rows {
            let take = (rows - base).min(MAX_Y);
            unsafe {
                let mut f_prices = d_prices.as_device_ptr().as_raw();
                let mut f_rsi    = d_rsi .as_device_ptr().add(base).as_raw();
                let mut f_ema    = d_ema .as_device_ptr().add(base).as_raw();
                let mut f_fast   = d_fast.as_device_ptr().add(base).as_raw();
                let mut series_len_i = len as i32;
                let mut n_combos_i = take as i32;
                let mut first_i = first_valid as i32;
                let row_offset_elems = 2 * base * len;
                let mut f_out = d_out.as_device_ptr().add(row_offset_elems).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut f_prices as *mut _ as *mut c_void,
                    &mut f_rsi as *mut _ as *mut c_void,
                    &mut f_ema as *mut _ as *mut c_void,
                    &mut f_fast as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut f_out as *mut _ as *mut c_void,
                ];
                let grid: GridSize = (1u32, take as u32, 1u32).into();
                let block: BlockSize = (block_x, 1, 1).into();
                self.stream.launch(&func, grid, block, 0, args).map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
            }
            unsafe { (*(self as *const _ as *mut CudaQqe)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }
            self.maybe_log_batch_debug();
            base += take;
        }
        self.stream.synchronize().map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        Ok((DeviceArrayF32 { buf: d_out, rows: 2 * rows, cols: len }, combos))
    }

    // ---- Many-series: time-major, one param ----
    pub fn qqe_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &QqeParams,
    ) -> Result<DeviceArrayF32, CudaQqeError> {
        if cols == 0 || rows == 0 { return Err(CudaQqeError::InvalidInput("cols/rows must be > 0".into())); }
        if prices_tm_f32.len() != cols * rows { return Err(CudaQqeError::InvalidInput("data length != cols*rows".into())); }
        let rsi_p = params.rsi_period.unwrap_or(0); let ema_p = params.smoothing_factor.unwrap_or(0); let fast_k = params.fast_factor.unwrap_or(4.236) as f32;
        if rsi_p == 0 || ema_p == 0 { return Err(CudaQqeError::InvalidInput("invalid rsi/ema period".into())); }

        // Per-series first_valid
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = prices_tm_f32[t * cols + s];
                if v.is_finite() { fv = Some(t); break; }
            }
            let fv = fv.ok_or_else(|| CudaQqeError::InvalidInput(format!("series {} all NaN", s)))?;
            if rows - fv < rsi_p + ema_p { return Err(CudaQqeError::InvalidInput("not enough valid data per series".into())); }
            first_valids[s] = fv as i32;
        }

        let req = cols*rows*4 + cols*4 + 2*cols*rows*4 + 64*1024*1024usize;
        if !Self::device_mem_ok(req) { return Err(CudaQqeError::InvalidInput("insufficient VRAM".into())); }
        let d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::from_slice_async(prices_tm_f32, &self.stream) }
            .map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        let d_first: DeviceBuffer<i32> = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }
            .map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(rows * (2 * cols), &self.stream) }
            .map_err(|e| CudaQqeError::Cuda(e.to_string()))?;

        // Heuristic + warp alignment for block size: kernel uses block.x to prefill warmup
        let warm_max = (0..cols)
            .map(|s| (first_valids[s] as usize).saturating_add(rsi_p + ema_p).saturating_sub(2))
            .max().unwrap_or(0);
        let mut block_x = match self.policy_many {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => { if warm_max < 128 { 128 } else if warm_max < 512 { 256 } else { 512 } }
        };
        block_x = Self::warp_align(block_x);
        // Cache function lookup
        let func = self.module.get_function("qqe_many_series_one_param_time_major_f32").map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut rsi_i = rsi_p as i32;
            let mut ema_i = ema_p as i32;
            let mut fast_k_f = fast_k as f32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut rsi_i as *mut _ as *mut c_void,
                &mut ema_i as *mut _ as *mut c_void,
                &mut fast_k_f as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            // Critical: grid.x must be 1; one block per series along Y
            let grid: GridSize = (1u32, cols as u32, 1u32).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.stream.launch(&func, grid, block, 0, args).map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        }
        unsafe { (*(self as *const _ as *mut CudaQqe)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
        self.maybe_log_many_debug();
        self.stream.synchronize().map_err(|e| CudaQqeError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols: 2 * cols })
    }
}

// -------- Benches --------
pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 192;
    const MANY_COLS: usize = 256;
    const MANY_ROWS: usize = 16 * 1024;

    fn gen_series(n: usize) -> Vec<f32> {
        let mut v = vec![f32::NAN; n];
        for i in 64..n { let x = i as f32; v[i] = (x * 0.00123).sin() + 0.00025 * x; }
        v
    }

    struct BatchState { cuda: CudaQqe, prices: Vec<f32>, sweep: QqeBatchRange }
    impl CudaBenchState for BatchState { fn launch(&mut self) { let _ = self.cuda.qqe_batch_dev(&self.prices, &self.sweep).unwrap(); } }

    struct ManyState { cuda: CudaQqe, prices_tm: Vec<f32>, cols: usize, rows: usize, params: QqeParams }
    impl CudaBenchState for ManyState { fn launch(&mut self) { let _ = self.cuda.qqe_many_series_one_param_time_major_dev(&self.prices_tm, self.cols, self.rows, &self.params).unwrap(); } }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaQqe::new(0).expect("cuda qqe");
        let prices = gen_series(ONE_SERIES_LEN);
        let sweep = QqeBatchRange { rsi_period: (8, 8 + PARAM_SWEEP - 1, 1), smoothing_factor: (5, 5, 0), fast_factor: (4.236, 4.236, 0.0) };
        Box::new(BatchState { cuda, prices, sweep })
    }
    fn prep_many() -> Box<dyn CudaBenchState> {
        let cuda = CudaQqe::new(0).expect("cuda qqe");
        let cols = MANY_COLS; let rows = MANY_ROWS;
        let mut tm = vec![f32::NAN; cols * rows];
        for s in 0..cols { for t in s..rows { let x = (t as f32) + 0.1 * (s as f32); tm[t*cols + s] = (0.002*x).sin() + 0.0003*x; } }
        let params = QqeParams { rsi_period: Some(14), smoothing_factor: Some(5), fast_factor: Some(4.236) };
        Box::new(ManyState { cuda, prices_tm: tm, cols, rows, params })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new("qqe", "one_series_many_params", "qqe_cuda_batch_dev", "1m_x_192", prep_batch)
                .with_sample_size(12)
                .with_mem_required(ONE_SERIES_LEN * 4 + (2 * PARAM_SWEEP * ONE_SERIES_LEN) * 4 + 64*1024*1024),
            CudaBenchScenario::new("qqe", "many_series_one_param", "qqe_cuda_many_series_one_param_dev", "256x16k", prep_many)
                .with_sample_size(12)
                .with_mem_required(MANY_COLS * MANY_ROWS * 3 * 4 + 64*1024*1024),
        ]
    }
}

