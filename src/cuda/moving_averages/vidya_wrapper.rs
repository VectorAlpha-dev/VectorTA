//! CUDA wrapper for VIDYA (Variable Index Dynamic Average).
//!
//! Parity with ALMA wrapper style:
//! - PTX load via include_str!(concat!(env!("OUT_DIR"), "/vidya_kernel.ptx"))
//!   using DetermineTargetFromContext + OptLevel(O2) with safe fallbacks.
//! - NON_BLOCKING stream.
//! - Lightweight kernel policies and one-time debug logging when BENCH_DEBUG=1.
//! - VRAM estimation with mem_get_info() and 64MB headroom; chunking across grid.x if needed.
//!
//! Math pattern: Recurrence/IIR per parameter. No host precompute required beyond
//! expanding the parameter grid. Warmup/NaN semantics match src/indicators/vidya.rs:
//! first defined output index = first_valid + long_period - 2; prefix is NaN.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::vidya::{VidyaBatchRange, VidyaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaVidyaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaVidyaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaVidyaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaVidyaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaVidyaError {}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}
impl Default for BatchKernelPolicy { fn default() -> Self { Self::Auto } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}
impl Default for ManySeriesKernelPolicy { fn default() -> Self { Self::Auto } }

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaVidyaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaVidya {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaVidyaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaVidya {
    pub fn new(device_id: usize) -> Result<Self, CudaVidyaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vidya_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])
                    .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaVidyaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaVidyaPolicy) -> Result<Self, CudaVidyaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaVidyaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))
    }

    // ---------- Utilities ----------
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Ok((free, _)) = mem_get_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] VIDYA batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaVidya)).debug_batch_logged = true; }
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
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] VIDYA many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaVidya)).debug_many_logged = true; }
            }
        }
    }

    // ---------- Host prep ----------
    fn expand_grid(range: &VidyaBatchRange) -> Vec<VidyaParams> {
        fn axis_usize(a: (usize, usize, usize)) -> Vec<usize> {
            let (start, end, step) = a;
            if step == 0 || start == end {
                return vec![start];
            }
            let mut v = Vec::new();
            let mut x = start;
            while x <= end {
                v.push(x);
                x = x.saturating_add(step);
            }
            v
        }
        fn axis_f64(a: (f64, f64, f64)) -> Vec<f64> {
            let (start, end, step) = a;
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
        let shorts = axis_usize(range.short_period);
        let longs = axis_usize(range.long_period);
        let alphas = axis_f64(range.alpha);
        let mut combos = Vec::with_capacity(shorts.len() * longs.len() * alphas.len());
        for &sp in &shorts {
            for &lp in &longs {
                for &a in &alphas {
                    combos.push(VidyaParams {
                        short_period: Some(sp),
                        long_period: Some(lp),
                        alpha: Some(a),
                    });
                }
            }
        }
        combos
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &VidyaBatchRange,
    ) -> Result<(Vec<VidyaParams>, usize, usize, usize), CudaVidyaError> {
        if data_f32.is_empty() {
            return Err(CudaVidyaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaVidyaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaVidyaError::InvalidInput("no parameter combinations".into()));
        }
        let series_len = data_f32.len();
        let max_long = combos
            .iter()
            .map(|c| c.long_period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_long == 0 || series_len - first_valid < max_long {
            return Err(CudaVidyaError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_long,
                series_len - first_valid
            )));
        }
        Ok((combos, first_valid, series_len, max_long))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VidyaParams,
    ) -> Result<(Vec<i32>, usize, usize, i32, i32, f32), CudaVidyaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaVidyaError::InvalidInput("cols or rows is zero".into()));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaVidyaError::InvalidInput("data length mismatch".into()));
        }
        let sp = params.short_period.unwrap_or(2) as i32;
        let lp = params.long_period.unwrap_or(5) as i32;
        let a = params.alpha.unwrap_or(0.2) as f32;
        if sp <= 0 || lp <= 0 || !(a.is_finite()) {
            return Err(CudaVidyaError::InvalidInput("invalid params".into()));
        }
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = 0i32;
            let mut found = false;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = t as i32;
                    found = true;
                    break;
                }
            }
            first_valids[s] = if found { fv } else { rows as i32 };
        }
        Ok((first_valids, cols, rows, sp, lp, a))
    }

    // ---------- Launchers ----------
    pub fn vidya_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &VidyaBatchRange,
    ) -> Result<DeviceArrayF32, CudaVidyaError> {
        let (combos, first_valid, series_len, _max_long) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        // VRAM estimate
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let params_bytes = n_combos
            * (std::mem::size_of::<i32>()   // shorts
                + std::mem::size_of::<i32>() // longs
                + std::mem::size_of::<f32>()); // alphas
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64MB
        if !Self::will_fit(required, headroom) {
            return Err(CudaVidyaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Build small param arrays
        let mut shorts_i32 = vec![0i32; n_combos];
        let mut longs_i32 = vec![0i32; n_combos];
        let mut alphas_f32 = vec![0f32; n_combos];
        for (i, prm) in combos.iter().enumerate() {
            shorts_i32[i] = prm.short_period.unwrap() as i32;
            longs_i32[i] = prm.long_period.unwrap() as i32;
            alphas_f32[i] = prm.alpha.unwrap() as f32;
        }

        // Allocate and copy
        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_f32, &self.stream)
                .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?
        };
        let d_shorts = DeviceBuffer::from_slice(&shorts_i32)
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
        let d_longs = DeviceBuffer::from_slice(&longs_i32)
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
        let d_alphas = DeviceBuffer::from_slice(&alphas_f32)
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;

        // Select policy
        let (block_x, grid_x) = match self.policy.batch {
            BatchKernelPolicy::Auto => (64u32, n_combos as u32),
            BatchKernelPolicy::Plain { block_x } => (block_x.max(1), n_combos as u32),
        };
        unsafe {
            (*(self as *const _ as *mut CudaVidya)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        // Launch
        let func = self
            .module
            .get_function("vidya_batch_f32")
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
        let stream = &self.stream; // satisfy cust::launch! macro stream ident
        unsafe {
            launch!(
                func<<<GridSize::xy(grid_x, 1), BlockSize::xyz(block_x, 1, 1), 0, stream>>>(
                    d_prices.as_device_ptr(),
                    d_shorts.as_device_ptr(),
                    d_longs.as_device_ptr(),
                    d_alphas.as_device_ptr(),
                    series_len as i32,
                    first_valid as i32,
                    n_combos as i32,
                    d_out.as_device_ptr()
                )
            )
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: series_len })
    }

    pub fn vidya_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VidyaParams,
    ) -> Result<DeviceArrayF32, CudaVidyaError> {
        let (first_valids, cols, rows, sp, lp, a) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let stride = cols;
        let total = cols * rows;

        // VRAM estimate
        let input_bytes = total * std::mem::size_of::<f32>();
        let params_bytes = cols * std::mem::size_of::<i32>(); // first_valids
        let out_bytes = total * std::mem::size_of::<f32>();
        let required = input_bytes + params_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaVidyaError::InvalidInput("insufficient device memory".into()));
        }

        // Copies
        let d_prices_tm = unsafe {
            DeviceBuffer::from_slice_async(data_tm_f32, &self.stream)
                .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?
        };
        let d_firsts = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total) }
                .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;

        // Policy
        let (block_x, grid_x) = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => (64u32, cols as u32),
            ManySeriesKernelPolicy::OneD { block_x } => (block_x.max(1), cols as u32),
        };
        unsafe {
            (*(self as *const _ as *mut CudaVidya)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        // Launch
        let func = self
            .module
            .get_function("vidya_many_series_one_param_f32")
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<GridSize::xy(grid_x, 1), BlockSize::xyz(block_x, 1, 1), 0, stream>>>(
                    d_prices_tm.as_device_ptr(),
                    d_firsts.as_device_ptr(),
                    sp,
                    lp,
                    a,
                    cols as i32,
                    rows as i32,
                    d_out.as_device_ptr()
                )
            )
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaVidyaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }
}

// ---------- Bench registration ----------
pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        vidya_benches,
        CudaVidya,
        crate::indicators::vidya::VidyaBatchRange,
        crate::indicators::vidya::VidyaParams,
        vidya_batch_dev,
        vidya_many_series_one_param_time_major_dev,
        crate::indicators::vidya::VidyaBatchRange {
            short_period: (2, 2 + PARAM_SWEEP - 1, 1),
            long_period: (10, 10 + PARAM_SWEEP - 1, 1),
            alpha: (0.2, 0.2, 0.0)
        },
        crate::indicators::vidya::VidyaParams {
            short_period: Some(4),
            long_period: Some(32),
            alpha: Some(0.2)
        },
        "vidya",
        "vidya"
    );
    pub use vidya_benches::bench_profiles;
}
