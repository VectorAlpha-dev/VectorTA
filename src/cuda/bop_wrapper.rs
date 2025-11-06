//! CUDA wrapper for Balance of Power (BOP): (close - open) / (high - low)
//!
//! Pattern classification: elementwise ratio per time index. No parameters.
//!
//! Semantics (parity with scalar):
//! - Warmup: write NaN up to `first_valid` (first index where all OHLC are non-NaN).
//! - Runtime rule: if (high - low) <= 0.0 → 0.0 at that index; otherwise ratio.
//! - Mid-stream NaNs are not masked (match CPU path).

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaBopError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaBopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaBopError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaBopError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaBopError {}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaBopPolicy {
    pub batch_block_x: Option<u32>,
    pub many_block_x: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaBop {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaBopPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    sm_count: u32,
}

impl CudaBop {
    #[inline]
    fn copy_h2d_maybe_async_f32(
        &self,
        src: &[f32],
    ) -> Result<(DeviceBuffer<f32>, Option<LockedBuffer<f32>>), CudaBopError> {
        const ASYNC_PIN_THRESHOLD_BYTES: usize = 1 << 20; // 1 MiB
        let bytes = src.len() * std::mem::size_of::<f32>();
        if bytes >= ASYNC_PIN_THRESHOLD_BYTES {
            let h_locked =
                LockedBuffer::from_slice(src).map_err(|e| CudaBopError::Cuda(e.to_string()))?;
            let mut d = unsafe {
                DeviceBuffer::uninitialized_async(src.len(), &self.stream)
                    .map_err(|e| CudaBopError::Cuda(e.to_string()))?
            };
            unsafe {
                d.async_copy_from(&h_locked, &self.stream)
                    .map_err(|e| CudaBopError::Cuda(e.to_string()))?;
            }
            Ok((d, Some(h_locked)))
        } else {
            DeviceBuffer::from_slice(src)
                .map(|d| (d, None))
                .map_err(|e| CudaBopError::Cuda(e.to_string()))
        }
    }
    pub fn new(device_id: usize) -> Result<Self, CudaBopError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaBopError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaBopError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaBopError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaBopError::Cuda(e.to_string()))?;

        // SM count for light grid clamping
        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)
            .map_err(|e| CudaBopError::Cuda(e.to_string()))? as u32;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/bop_kernel.ptx"));
        // Keep target-from-context; default JIT OptLevel is O4 (max optimised)
        let jit_opts = &[ModuleJitOption::DetermineTargetFromContext];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaBopError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaBopError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaBopPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            sm_count,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, p: CudaBopPolicy) {
        self.policy = p;
    }

    // ---------- Batch (one series × many params [degenerate: 1]) ----------
    pub fn bop_batch_dev(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<DeviceArrayF32, CudaBopError> {
        let (first_valid, len) = Self::validate_ohlc_slices(open, high, low, close)?;

        // VRAM estimate and check (best-effort)
        if let Ok((free, _)) = mem_get_info() {
            let bytes = (5 * len) * std::mem::size_of::<f32>(); // 4 inputs + 1 output
            let headroom = 64usize * 1024 * 1024;
            if bytes.saturating_add(headroom) > free {
                return Err(CudaBopError::InvalidInput(
                    "estimated device memory exceeds free VRAM".into(),
                ));
            }
        }

        let mut _pinned_guards: Vec<LockedBuffer<f32>> = Vec::new();
        let (d_open, p0) = self.copy_h2d_maybe_async_f32(open)?;  if let Some(h) = p0 { _pinned_guards.push(h); }
        let (d_high, p1) = self.copy_h2d_maybe_async_f32(high)?;  if let Some(h) = p1 { _pinned_guards.push(h); }
        let (d_low,  p2) = self.copy_h2d_maybe_async_f32(low)?;   if let Some(h) = p2 { _pinned_guards.push(h); }
        let (d_close,p3) = self.copy_h2d_maybe_async_f32(close)?; if let Some(h) = p3 { _pinned_guards.push(h); }
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream) }
                .map_err(|e| CudaBopError::Cuda(e.to_string()))?;

        self.launch_batch(
            &d_open,
            &d_high,
            &d_low,
            &d_close,
            len,
            first_valid,
            &mut d_out,
        )?;
        self.launch_batch(
            &d_open,
            &d_high,
            &d_low,
            &d_close,
            len,
            first_valid,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: 1,
            cols: len,
        })
    }

    fn launch_batch(
        &self,
        d_open: &DeviceBuffer<f32>,
        d_high: &DeviceBuffer<f32>,
        d_low: &DeviceBuffer<f32>,
        d_close: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaBopError> {
        let func = self
            .module
            .get_function("bop_batch_f32")
            .map_err(|e| CudaBopError::Cuda(e.to_string()))?;

        let block_x = self.policy.batch_block_x.unwrap_or(256);
        // Match kernel ILP (4 items per thread) to size grid tightly.
        const ILP: u32 = 4;
        let work = ((len as u32) + block_x * ILP - 1) / (block_x * ILP);
        // Clamp grid to a conservative multiple of SMs to avoid oversubscription on huge inputs
        let max_grid = (self.sm_count.max(1)) * 32;
        let grid_x = work.min(max_grid).max(1);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut open_ptr = d_open.as_device_ptr().as_raw();
            let mut high_ptr = d_high.as_device_ptr().as_raw();
            let mut low_ptr = d_low.as_device_ptr().as_raw();
            let mut close_ptr = d_close.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut first_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut open_ptr as *mut _ as *mut c_void,
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaBopError::Cuda(e.to_string()))?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaBop)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        self.stream
            .synchronize()
            .map_err(|e| CudaBopError::Cuda(e.to_string()))
    }

    // ---------- Many-series × one-param (time-major) ----------
    pub fn bop_many_series_one_param_time_major_dev(
        &self,
        open_tm: &[f32],
        high_tm: &[f32],
        low_tm: &[f32],
        close_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<DeviceArrayF32, CudaBopError> {
        if cols == 0 || rows == 0 {
            return Err(CudaBopError::InvalidInput("invalid dims".into()));
        }
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaBopError::InvalidInput("rows*cols overflow".into()))?;
        if open_tm.len() != expected
            || high_tm.len() != expected
            || low_tm.len() != expected
            || close_tm.len() != expected
        {
            return Err(CudaBopError::InvalidInput(
                "time-major inputs length mismatch".into(),
            ));
        }

        // Per-series first_valids (allow all-NaN series; kernel fills NaNs)
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for t in 0..rows {
                let idx = t * cols + s;
                let o = open_tm[idx];
                let h = high_tm[idx];
                let l = low_tm[idx];
                let c = close_tm[idx];
                if !o.is_nan() && !h.is_nan() && !l.is_nan() && !c.is_nan() {
                    fv = t as i32;
                    break;
                }
            }
            first_valids[s] = fv;
        }

        // VRAM estimate and check (best-effort)
        if let Ok((free, _)) = mem_get_info() {
            let n = expected;
            let bytes = (5 * n) * std::mem::size_of::<f32>() + cols * std::mem::size_of::<i32>();
            let headroom = 64usize * 1024 * 1024;
            if bytes.saturating_add(headroom) > free {
                return Err(CudaBopError::InvalidInput(
                    "estimated device memory exceeds free VRAM".into(),
                ));
            }
        }

        let mut _pinned_guards: Vec<LockedBuffer<f32>> = Vec::new();
        let (d_open,  p0) = self.copy_h2d_maybe_async_f32(open_tm)?;  if let Some(h) = p0 { _pinned_guards.push(h); }
        let (d_high,  p1) = self.copy_h2d_maybe_async_f32(high_tm)?;  if let Some(h) = p1 { _pinned_guards.push(h); }
        let (d_low,   p2) = self.copy_h2d_maybe_async_f32(low_tm)?;   if let Some(h) = p2 { _pinned_guards.push(h); }
        let (d_close, p3) = self.copy_h2d_maybe_async_f32(close_tm)?; if let Some(h) = p3 { _pinned_guards.push(h); }
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaBopError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(expected, &self.stream) }
                .map_err(|e| CudaBopError::Cuda(e.to_string()))?;

        self.launch_many_series(
            &d_open, &d_high, &d_low, &d_close, &d_first, cols, rows, &mut d_out,
        )?;
        self.launch_many_series(
            &d_open, &d_high, &d_low, &d_close, &d_first, cols, rows, &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_many_series(
        &self,
        d_open_tm: &DeviceBuffer<f32>,
        d_high_tm: &DeviceBuffer<f32>,
        d_low_tm: &DeviceBuffer<f32>,
        d_close_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaBopError> {
        let func = self
            .module
            .get_function("bop_many_series_one_param_f32")
            .map_err(|e| CudaBopError::Cuda(e.to_string()))?;
        let block_x = self.policy.many_block_x.unwrap_or(256);
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut open_ptr = d_open_tm.as_device_ptr().as_raw();
            let mut high_ptr = d_high_tm.as_device_ptr().as_raw();
            let mut low_ptr = d_low_tm.as_device_ptr().as_raw();
            let mut close_ptr = d_close_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut nseries_i = cols as i32;
            let mut slen_i = rows as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut open_ptr as *mut _ as *mut c_void,
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut nseries_i as *mut _ as *mut c_void,
                &mut slen_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaBopError::Cuda(e.to_string()))?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaBop)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        self.stream
            .synchronize()
            .map_err(|e| CudaBopError::Cuda(e.to_string()))
    }

    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] BOP batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBop)).debug_batch_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBop)).debug_batch_logged = true;
                }
            }
        }
    }
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] BOP many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBop)).debug_many_logged = true;
                }
                unsafe {
                    (*(self as *const _ as *mut CudaBop)).debug_many_logged = true;
                }
            }
        }
    }

    fn validate_ohlc_slices(
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
    ) -> Result<(usize, usize), CudaBopError> {
        let len = open.len();
        if len == 0 || high.len() != len || low.len() != len || close.len() != len {
            return Err(CudaBopError::InvalidInput(
                "input slices are empty or mismatched".into(),
            ));
        }
        let first_valid = (0..len)
            .find(|&i| {
                !open[i].is_nan() && !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan()
            })
            .ok_or_else(|| CudaBopError::InvalidInput("all values are NaN".into()))?;
        Ok((first_valid, len))
    }
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::gen_series;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_COLS: usize = 1024;
    const MANY_ROWS: usize = 8192;

    fn bytes_one_series() -> usize {
        // 4 inputs + 1 output + headroom
        let in_bytes = 4 * ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series() -> usize {
        let n = MANY_COLS * MANY_ROWS;
        let in_bytes = 4 * n * std::mem::size_of::<f32>();
        let out_bytes = n * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct BopBatchState {
        cuda: CudaBop,
        open: Vec<f32>,
        high: Vec<f32>,
        low: Vec<f32>,
        close: Vec<f32>,
    }
    impl CudaBenchState for BopBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .bop_batch_dev(&self.open, &self.high, &self.low, &self.close)
                .expect("bop batch");
        }
    }
    fn prep_one_series_batch() -> Box<dyn CudaBenchState> {
        let cuda = CudaBop::new(0).expect("cuda bop");
        let mut open = gen_series(ONE_SERIES_LEN);
        let mut high = open.clone();
        let mut low = open.clone();
        let mut close = open.clone();
        for i in 4..ONE_SERIES_LEN {
            let x = i as f32 * 0.0023;
            open[i] = open[i] + 0.001 * x.sin();
            high[i] = open[i] + (0.5 + 0.05 * x.cos()).abs();
            low[i] = open[i] - (0.5 + 0.05 * x.sin()).abs();
            close[i] = open[i] + 0.2 * (x).sin();
        }
        Box::new(BopBatchState {
            cuda,
            open,
            high,
            low,
            close,
        })
    }

    
    struct BopManyState {
        cuda: CudaBop,
        open_tm: Vec<f32>,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        close_tm: Vec<f32>,
    }
    impl CudaBenchState for BopManyState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .bop_many_series_one_param_time_major_dev(
                    &self.open_tm,
                    &self.high_tm,
                    &self.low_tm,
                    &self.close_tm,
                    MANY_COLS,
                    MANY_ROWS,
                )
                .expect("bop many");
        }
    }
    fn prep_many_series() -> Box<dyn CudaBenchState> {
        let cuda = CudaBop::new(0).expect("cuda bop");
        let n = MANY_COLS * MANY_ROWS;
        let mut base = gen_series(n);
        let mut open = vec![f32::NAN; n];
        let mut high = vec![f32::NAN; n];
        let mut low = vec![f32::NAN; n];
        let mut close = vec![f32::NAN; n];
        for s in 0..MANY_COLS {
            for t in s..MANY_ROWS {
                let idx = t * MANY_COLS + s;
                let x = (t as f32) * 0.002 + (s as f32) * 0.01;
                let b = base[idx];
                open[idx] = b + 0.001 * x.cos();
                high[idx] = b + 0.3 + 0.02 * x.sin();
                low[idx] = b - 0.3 - 0.02 * x.cos();
                close[idx] = b + 0.05 * x.sin();
            }
        }
        Box::new(BopManyState {
            cuda,
            open_tm: open,
            high_tm: high,
            low_tm: low,
            close_tm: close,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "bop",
                "one_series_many_params",
                "bop_cuda_batch_dev",
                "1m_x_1",
                prep_one_series_batch,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series()),
            CudaBenchScenario::new(
                "bop",
                "many_series_one_param",
                "bop_cuda_many_series_one_param_dev",
                "1024x8192",
                prep_many_series,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series()),
        ]
    }
}
