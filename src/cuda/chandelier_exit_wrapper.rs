//! CUDA scaffolding for the Chandelier Exit (CE) indicator.
//!
//! Math category: recurrence/time-scan per parameter (ATR + rolling
//! extremums + trailing logic). We parallelize across rows (batch) or
//! across series (many-series) and scan time within each thread to
//! preserve scalar semantics. No sizable row-shared precompute beyond
//! ATR; use_close is static across a batch sweep.

#![cfg(feature = "cuda")]

use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

use crate::indicators::chandelier_exit::{CeBatchRange, ChandelierExitParams};
use crate::cuda::moving_averages::alma_wrapper::DeviceArrayF32;

#[derive(Debug)]
pub enum CudaCeError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaCeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaCeError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaCeError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaCeError {}

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
pub enum BatchKernelSelected { Plain { block_x: u32 } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }

pub struct CudaChandelierExit {
    module: Module,
    stream: Stream,
    _ctx: Context,
    policy_batch: BatchKernelPolicy,
    policy_many: ManySeriesKernelPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaChandelierExit {
    pub fn new(device_id: usize) -> Result<Self, CudaCeError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        let ctx = Context::new(device).map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/chandelier_exit_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaCeError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _ctx: ctx,
            policy_batch: BatchKernelPolicy::Auto,
            policy_many: ManySeriesKernelPolicy::Auto,
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Ok((free, _)) = cust::memory::mem_get_info() {
            required_bytes.saturating_add(headroom) <= free
        } else { true }
    }

    fn expand_grid(range: &CeBatchRange) -> Vec<ChandelierExitParams> {
        // Mirror scalar expand_ce (use_close is static in the sweep)
        fn axis_usize(t: (usize, usize, usize)) -> Vec<usize> {
            if t.2 == 0 || t.0 == t.1 { return vec![t.0]; }
            (t.0..=t.1).step_by(t.2).collect()
        }
        fn axis_f64(t: (f64, f64, f64)) -> Vec<f64> {
            if t.2.abs() < 1e-12 || (t.0 - t.1).abs() < 1e-12 { return vec![t.0]; }
            let mut v = Vec::new();
            let mut x = t.0; while x <= t.1 + 1e-12 { v.push(x); x += t.2; }
            v
        }
        let periods = axis_usize(range.period);
        let mults = axis_f64(range.mult);
        let use_close = range.use_close.0;
        let mut out = Vec::with_capacity(periods.len() * mults.len());
        for &p in &periods { for &m in &mults {
            out.push(ChandelierExitParams { period: Some(p), mult: Some(m), use_close: Some(use_close) });
        } }
        out
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] CE batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaChandelierExit)).debug_batch_logged = true; }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] CE many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaChandelierExit)).debug_many_logged = true; }
            }
        }
    }

    // Public policy controls for parity with ALMA/CWMA wrappers
    pub fn set_batch_policy(&mut self, p: BatchKernelPolicy) { self.policy_batch = p; }
    pub fn set_many_series_policy(&mut self, p: ManySeriesKernelPolicy) { self.policy_many = p; }
    pub fn batch_policy(&self) -> BatchKernelPolicy { self.policy_batch }
    pub fn many_series_policy(&self) -> ManySeriesKernelPolicy { self.policy_many }

    fn first_valid(use_close: bool, high: &[f32], low: &[f32], close: &[f32]) -> Result<usize, CudaCeError> {
        let len = close.len().min(high.len()).min(low.len());
        if len == 0 { return Err(CudaCeError::InvalidInput("empty input".into())); }
        let fv = if use_close {
            (0..len).find(|&i| !close[i].is_nan())
        } else {
            (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        };
        fv.ok_or_else(|| CudaCeError::InvalidInput("all values are NaN".into()))
    }

    fn launch_batch(
        &self,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_periods: &DeviceBuffer<i32>,
        d_mults: &DeviceBuffer<f32>,
        n_combos: usize,
        use_close: bool,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCeError> {
        let func = self
            .module
            .get_function("chandelier_exit_batch_f32")
            .map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        // Environment override wins; else policy; else 256
        let block_x_env = std::env::var("CE_BLOCK_X").ok().and_then(|v| v.parse::<u32>().ok());
        let block_x = block_x_env
            .or_else(|| match self.policy_batch { BatchKernelPolicy::Plain { block_x } => Some(block_x), _ => None })
            .unwrap_or(256)
            .max(32);
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut h = d_high.as_device_ptr().as_raw();
            let mut l = d_low.as_device_ptr().as_raw();
            let mut c = d_close.as_device_ptr().as_raw();
            let mut n = len as i32;
            let mut fv = first_valid as i32;
            let mut p = d_periods.as_device_ptr().as_raw();
            let mut m = d_mults.as_device_ptr().as_raw();
            let mut r = n_combos as i32;
            let mut u = if use_close { 1i32 } else { 0i32 };
            let mut o = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut m as *mut _ as *mut c_void,
                &mut r as *mut _ as *mut c_void,
                &mut u as *mut _ as *mut c_void,
                &mut o as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaCeError::Cuda(e.to_string()))?;
            // Record selection for debug parity
            (*(self as *const _ as *mut CudaChandelierExit)).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    pub fn chandelier_exit_batch_dev(
        &self,
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &CeBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<ChandelierExitParams>), CudaCeError> {
        let len = high.len().min(low.len()).min(close.len());
        if len == 0 { return Err(CudaCeError::InvalidInput("empty input".into())); }
        let combos = Self::expand_grid(sweep);
        if combos.is_empty() { return Err(CudaCeError::InvalidInput("no parameter combinations".into())); }
        let use_close = sweep.use_close.0;
        let first_valid = Self::first_valid(use_close, high, low, close)?;
        // Ensure all combos are feasible
        for prm in &combos {
            let p = prm.period.unwrap_or(22);
            if p == 0 { return Err(CudaCeError::InvalidInput("period must be >=1".into())); }
            if len - first_valid < p {
                return Err(CudaCeError::InvalidInput(format!(
                    "not enough valid data (need >= {}, have {})",
                    p, len - first_valid
                )));
            }
        }

        // VRAM estimate: 3 inputs + params + outputs (2 rows per combo)
        let rows = combos.len();
        let req = (3 * len * std::mem::size_of::<f32>())
            + (rows * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>()))
            + (2 * rows * len * std::mem::size_of::<f32>());
        let headroom = std::env::var("CUDA_MEM_HEADROOM").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(64 * 1024 * 1024);
        if !Self::will_fit(req, headroom) {
            return Err(CudaCeError::InvalidInput("insufficient VRAM for CE batch".into()));
        }

        // Device buffers
        let d_high = unsafe { DeviceBuffer::from_slice_async(&high[..len], &self.stream) }.map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        let d_low  = unsafe { DeviceBuffer::from_slice_async(&low[..len],  &self.stream) }.map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        let d_close= unsafe { DeviceBuffer::from_slice_async(&close[..len],&self.stream) }.map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        let periods_host: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let mults_host:   Vec<f32> = combos.iter().map(|c| c.mult.unwrap()   as f32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods_host).map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        let d_mults   = DeviceBuffer::from_slice(&mults_host).map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(2 * rows * len, &self.stream) }.map_err(|e| CudaCeError::Cuda(e.to_string()))?;

        self.launch_batch(&d_high, &d_low, &d_close, len, first_valid, &d_periods, &d_mults, rows, use_close, &mut d_out)?;
        self.stream.synchronize().map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        Ok((DeviceArrayF32 { buf: d_out, rows: 2 * rows, cols: len }, combos))
    }

    fn launch_many_series(
        &self,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        d_first_valids: &DeviceBuffer<i32>,
        use_close: bool,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCeError> {
        let func = self
            .module
            .get_function("chandelier_exit_many_series_one_param_time_major_f32")
            .map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        let block_x_env = std::env::var("CE_MANY_BLOCK_X").ok().and_then(|v| v.parse::<u32>().ok());
        let block_x = block_x_env
            .or_else(|| match self.policy_many { ManySeriesKernelPolicy::OneD { block_x } => Some(block_x), _ => None })
            .unwrap_or(256)
            .max(32);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut h = d_high_tm.as_device_ptr().as_raw();
            let mut l = d_low_tm.as_device_ptr().as_raw();
            let mut c = d_close_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut p = period as i32;
            let mut m = mult as f32;
            let mut fv = d_first_valids.as_device_ptr().as_raw();
            let mut u = if use_close { 1i32 } else { 0i32 };
            let mut o = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut h as *mut _ as *mut c_void,
                &mut l as *mut _ as *mut c_void,
                &mut c as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
                &mut m as *mut _ as *mut c_void,
                &mut fv as *mut _ as *mut c_void,
                &mut u as *mut _ as *mut c_void,
                &mut o as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaCeError::Cuda(e.to_string()))?;
            // Record selection for debug parity
            (*(self as *const _ as *mut CudaChandelierExit)).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    pub fn chandelier_exit_many_series_one_param_time_major_dev(
        &self,
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
        period: usize,
        mult: f32,
        use_close: bool,
    ) -> Result<DeviceArrayF32, CudaCeError> {
        if cols == 0 || rows == 0 { return Err(CudaCeError::InvalidInput("empty matrix".into())); }
        if high_tm.len() != cols * rows || low_tm.len() != cols * rows || close_tm.len() != cols * rows {
            return Err(CudaCeError::InvalidInput("matrix shape mismatch".into()));
        }
        if period == 0 { return Err(CudaCeError::InvalidInput("period must be >=1".into())); }

        // Per-series first_valid
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let idx = t * cols + s;
                let ok = if use_close {
                    !close_tm[idx].is_nan()
                } else {
                    !high_tm[idx].is_nan() && !low_tm[idx].is_nan() && !close_tm[idx].is_nan()
                };
                if ok { first_valids[s] = t as i32; break; }
            }
            if (rows as i32 - first_valids[s]) < period as i32 {
                return Err(CudaCeError::InvalidInput("not enough valid data for at least one series".into()));
            }
        }

        // VRAM: 3 inputs + first_valids + outputs (2 matrices)
        let req = (3 * cols * rows + cols + 2 * cols * rows) * std::mem::size_of::<f32>();
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(req, headroom) {
            return Err(CudaCeError::InvalidInput("insufficient VRAM for CE many-series".into()));
        }

        let d_high = unsafe { DeviceBuffer::from_slice_async(high_tm, &self.stream) }.map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        let d_low  = unsafe { DeviceBuffer::from_slice_async(low_tm, &self.stream) }.map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        let d_close= unsafe { DeviceBuffer::from_slice_async(close_tm, &self.stream) }.map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids).map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(2 * cols * rows, &self.stream) }.map_err(|e| CudaCeError::Cuda(e.to_string()))?;

        self.launch_many_series(&d_high, &d_low, &d_close, cols, rows, period, mult, &d_first, use_close, &mut d_out)?;
        self.stream.synchronize().map_err(|e| CudaCeError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out, rows: 2 * rows, cols })
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::cuda::bench::helpers::gen_series;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const COLS_256: usize = 256;
    const ROWS_8K: usize = 8 * 1024;

    fn synth_hlc_from_close(close: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut high = close.to_vec();
        let mut low = close.to_vec();
        for i in 0..close.len() {
            let v = close[i];
            if v.is_nan() { continue; }
            let x = i as f32 * 0.0025;
            let off = (0.002 * x.sin()).abs() + 0.15;
            high[i] = v + off;
            low[i] = v - off;
        }
        (high, low)
    }

    struct BatchState { cuda: CudaChandelierExit, high: Vec<f32>, low: Vec<f32>, close: Vec<f32>, sweep: CeBatchRange }
    impl CudaBenchState for BatchState { fn launch(&mut self) { let _ = self.cuda.chandelier_exit_batch_dev(&self.high, &self.low, &self.close, &self.sweep).unwrap(); } }

    struct ManySeriesState { cuda: CudaChandelierExit, high_tm: Vec<f32>, low_tm: Vec<f32>, close_tm: Vec<f32>, cols: usize, rows: usize, period: usize, mult: f32 }
    impl CudaBenchState for ManySeriesState { fn launch(&mut self) { let _ = self.cuda.chandelier_exit_many_series_one_param_time_major_dev(&self.high_tm, &self.low_tm, &self.close_tm, self.cols, self.rows, self.period, self.mult, true).unwrap(); } }

    fn prep_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaChandelierExit::new(0).expect("cuda ce");
        let close = gen_series(ONE_SERIES_LEN);
        let (high, low) = synth_hlc_from_close(&close);
        let sweep = CeBatchRange { period: (10, 50, 10), mult: (2.0, 3.0, 0.5), use_close: (true, true, false) };
        Box::new(BatchState { cuda, high, low, close, sweep })
    }
    fn prep_many() -> Box<dyn CudaBenchState> {
        let cuda = CudaChandelierExit::new(0).expect("cuda ce");
        let cols = COLS_256; let rows = ROWS_8K;
        let close_tm = {
            let mut v = vec![f32::NAN; cols * rows];
            for s in 0..cols { for t in s..rows { let x = (t as f32) + (s as f32) * 0.2; v[t*cols + s] = (x * 0.002).sin() + 0.0003 * x; } }
            v
        };
        let (high_tm, low_tm) = synth_hlc_from_close(&close_tm);
        Box::new(ManySeriesState { cuda, high_tm, low_tm, close_tm, cols, rows, period: 22, mult: 3.0 })
    }

    fn bytes_batch() -> usize {
        // 3 inputs + params + 2*output + 64MB headroom
        (3 * ONE_SERIES_LEN + (ONE_SERIES_LEN / 10) + 2 * ONE_SERIES_LEN) * std::mem::size_of::<f32>() + 64 * 1024 * 1024
    }
    fn bytes_many() -> usize {
        (3 * COLS_256 * ROWS_8K + COLS_256 + 2 * COLS_256 * ROWS_8K) * std::mem::size_of::<f32>() + 64 * 1024 * 1024
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new("chandelier_exit", "batch", "ce_cuda_batch", "1m", prep_batch)
                .with_mem_required(bytes_batch()),
            CudaBenchScenario::new("chandelier_exit", "many_series_one_param", "ce_cuda_many_series", "8k x 256", prep_many)
                .with_mem_required(bytes_many()),
        ]
    }
}
