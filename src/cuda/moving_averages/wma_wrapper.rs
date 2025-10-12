//! CUDA wrapper for the Weighted Moving Average (WMA) kernels.
//!
//! Parity notes with ALMA/CWMA:
//! - Policy enums for kernel selection (Auto/Plain/etc.)
//! - Non-blocking stream, VRAM checks, and Y-chunking for batch grid ≤ 65_535
//! - PTX JIT options: DetermineTargetFromContext + OptLevel(O2) with fallbacks
//! - Warmup/NaN semantics match scalar; FP32 compute on device
//! - API returns VRAM handles (`DeviceArrayF32`) for staged host copies

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::wma::{WmaBatchRange, WmaParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel, Symbol};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::{c_void, CStr, CString};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

// Keep in sync with device constant C_WMA_RAMP size in the CUDA TU.
const WMA_MAX_PERIOD: usize = 8192;

#[derive(Debug)]
pub enum CudaWmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaWmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaWmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaWmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaWmaError {}

// -------- Kernel selection policy (parity with ALMA/CWMA subset) --------

#[derive(Clone, Copy, Debug)]
pub enum WmaBatchThreadsPerOutput {
    One,
    Two,
}

#[derive(Clone, Copy, Debug)]
pub enum WmaBatchKernelPolicy {
    Auto,
    Plain {
        block_x: u32,
    },
    // Reserved for future: tiled WMA kernels are not implemented today
    Tiled {
        tile: u32,
        per_thread: WmaBatchThreadsPerOutput,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum WmaManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    // Reserved for future
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaWmaPolicy {
    pub batch: WmaBatchKernelPolicy,
    pub many_series: WmaManySeriesKernelPolicy,
}

impl Default for CudaWmaPolicy {
    fn default() -> Self {
        Self {
            batch: WmaBatchKernelPolicy::Auto,
            many_series: WmaManySeriesKernelPolicy::Auto,
        }
    }
}

// -------- Introspection (selected kernel) --------

#[derive(Clone, Copy, Debug)]
pub enum WmaBatchKernelSelected {
    Plain { block_x: u32 },
    Rolling { block_x: u32 },
    Prefix { block_x: u32 },
    Tiled1x { tile: u32 },
    Tiled2x { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum WmaManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

pub struct CudaWma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaWmaPolicy,
    last_batch: Option<WmaBatchKernelSelected>,
    last_many: Option<WmaManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    ramp_inited: bool,
}

impl CudaWma {
    /// Initialize the device constant ramp C_WMA_RAMP with 1..=WMA_MAX_PERIOD.
    fn init_constant_ramp(&mut self) -> Result<(), CudaWmaError> {
        // Try binding by CStr literal first; fall back to CString path.
        unsafe {
            if let Ok(mut sym) = self.module.get_global::<[f32; WMA_MAX_PERIOD]>(
                CStr::from_bytes_with_nul_unchecked(b"C_WMA_RAMP\0"),
            ) {
                let mut host = [0f32; WMA_MAX_PERIOD];
                for i in 0..WMA_MAX_PERIOD {
                    host[i] = (i as f32) + 1.0;
                }
                sym.copy_from(&host)
                    .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
                self.ramp_inited = true;
                return Ok(());
            }
            let name = CString::new("C_WMA_RAMP").unwrap();
            if let Ok(mut sym) = self
                .module
                .get_global::<[f32; WMA_MAX_PERIOD]>(name.as_c_str())
            {
                let mut host = [0f32; WMA_MAX_PERIOD];
                for i in 0..WMA_MAX_PERIOD {
                    host[i] = (i as f32) + 1.0;
                }
                sym.copy_from(&host)
                    .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
                self.ramp_inited = true;
            }
        }
        Ok(())
    }
    pub fn new(device_id: usize) -> Result<Self, CudaWmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaWmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/wma_kernel.ptx"));
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaWmaError::Cuda(e.to_string()))?
                }
            }
        };

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;

        let mut s = Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaWmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            ramp_inited: false,
        };
        let _ = s.init_constant_ramp();
        Ok(s)
    }

    pub fn new_with_policy(device_id: usize, policy: CudaWmaPolicy) -> Result<Self, CudaWmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaWmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaWmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<WmaBatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<WmaManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaWmaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaWmaError::Cuda(e.to_string()))
    }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

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

    #[inline]
    fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX_GRID_Y: usize = 65_535;
        (0..n).step_by(MAX_GRID_Y).map(move |start| {
            let len = (n - start).min(MAX_GRID_Y);
            (start, len)
        })
    }

    #[inline]
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
                    eprintln!("[DEBUG] WMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaWma)).debug_batch_logged = true;
                }
            }
        }
    }

    #[inline]
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
                    eprintln!("[DEBUG] WMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaWma)).debug_many_logged = true;
                }
            }
        }
    }

    fn expand_periods(range: &WmaBatchRange) -> Vec<WmaParams> {
        let (start, end, step) = range.period;
        let periods = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect::<Vec<_>>()
        };
        periods
            .into_iter()
            .map(|p| WmaParams { period: Some(p) })
            .collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &WmaBatchRange,
    ) -> Result<(Vec<WmaParams>, usize, usize, usize), CudaWmaError> {
        if data_f32.is_empty() {
            return Err(CudaWmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaWmaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaWmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period <= 1 {
                return Err(CudaWmaError::InvalidInput(format!(
                    "invalid period {} (must be > 1)",
                    period
                )));
            }
            if period > series_len {
                return Err(CudaWmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, series_len
                )));
            }
            let valid = series_len - first_valid;
            if valid < period {
                return Err(CudaWmaError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    period, valid
                )));
            }
            max_period = max_period.max(period);
        }

        Ok((combos, first_valid, series_len, max_period))
    }

    // (top-level WMA_MAX_PERIOD constant is used)

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWmaError> {
        if max_period == 0 {
            return Err(CudaWmaError::InvalidInput(
                "max_period must be positive".into(),
            ));
        }

        // Policy (only Plain path implemented)
        let block_x: u32 = match self.policy.batch {
            WmaBatchKernelPolicy::Plain { block_x } => block_x.max(1),
            _ => 256,
        };
        unsafe {
            let this = self as *const _ as *mut CudaWma;
            (*this).last_batch = Some(WmaBatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        // Zero SMEM when constant ramp is available and period within cap.
        let shared_bytes: u32 = if self.ramp_inited && max_period <= WMA_MAX_PERIOD {
            0
        } else {
            max_period
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| CudaWmaError::InvalidInput("shared memory size overflow".into()))?
                as u32
        };
        let max_smem = Device::get_device(self.device_id)
            .ok()
            .and_then(|d| {
                d.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)
                    .ok()
            })
            .unwrap_or(96 * 1024) as usize;
        if (shared_bytes as usize) > max_smem {
            return Err(CudaWmaError::InvalidInput(format!(
                "period {} requires {} bytes shared memory (exceeds limit {})",
                max_period, shared_bytes, max_smem
            )));
        }

        let func = self
            .module
            .get_function("wma_batch_f32")
            .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;

        for (start, len) in Self::grid_y_chunks(n_combos) {
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut series_len_i = series_len as i32;
                let mut combos_i = len as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
            }
        }
        Ok(())
    }

    fn launch_batch_kernel_rolling(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWmaError> {
        // Policy (block size)
        let block_x: u32 = match self.policy.batch {
            WmaBatchKernelPolicy::Plain { block_x } => block_x.max(1),
            _ => 256,
        };
        unsafe {
            let this = self as *const _ as *mut CudaWma;
            (*this).last_batch = Some(WmaBatchKernelSelected::Rolling { block_x });
        }
        self.maybe_log_batch_debug();

        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let func = self
            .module
            .get_function("wma_batch_rolling_f32")
            .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;

        for (start, len) in Self::grid_y_chunks(n_combos) {
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut series_len_i = series_len as i32;
                let mut combos_i = len as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
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
                    .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
            }
        }
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[WmaParams],
        first_valid: usize,
        series_len: usize,
        max_period: usize,
    ) -> Result<DeviceArrayF32, CudaWmaError> {
        let n_combos = combos.len();
        let has_prefix = self.module.get_function("wma_batch_prefix_f32").is_ok();
        let has_rolling = self.module.get_function("wma_batch_rolling_f32").is_ok();
        let prefer_prefix_env = matches!(std::env::var("WMA_BATCH_PREFIX"), Ok(ref v) if v == "1" || v.eq_ignore_ascii_case("true"));
        let force_path = std::env::var("WMA_FORCE_PATH").ok();

        // Heuristic defaults: prefer rolling when available and sizes warrant it
        let rolling_min_p: usize = std::env::var("WMA_ROLLING_MIN_PERIOD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let min_period = combos
            .iter()
            .map(|p| p.period.unwrap() as usize)
            .min()
            .unwrap_or(max_period);
        let may_use_rolling = has_rolling
            && self.ramp_inited
            && max_period <= WMA_MAX_PERIOD
            && min_period >= rolling_min_p
            && series_len >= (min_period + 8);

        enum Path {
            Plain,
            Rolling,
            Prefix,
        }
        let path = match force_path.as_deref() {
            Some("prefix") if has_prefix => Path::Prefix,
            Some("rolling") if has_rolling => Path::Rolling,
            Some("plain") => Path::Plain,
            _ => {
                if prefer_prefix_env && has_prefix {
                    Path::Prefix
                } else if may_use_rolling {
                    Path::Rolling
                } else {
                    Path::Plain
                }
            }
        };

        // Accurate VRAM estimate: only account for prefixes when we will use them
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let prefix_bytes = 2 * (series_len + 1) * std::mem::size_of::<f32>();
        let required = match path {
            Path::Prefix => prices_bytes + periods_bytes + prefix_bytes + out_bytes,
            _ => prices_bytes + periods_bytes + out_bytes,
        };
        let headroom = if matches!(path, Path::Prefix) { 64 } else { 32 } * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaWmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;

        match path {
            Path::Prefix => {
                // Build prefixes A and B on host (leading NaNs treated as 0.0)
                let mut pref_a = vec![0f32; series_len + 1];
                let mut pref_b = vec![0f32; series_len + 1];
                for i in 0..series_len {
                    let x = if i < first_valid { 0.0 } else { data_f32[i] };
                    pref_a[i + 1] = pref_a[i] + x;
                    pref_b[i + 1] = pref_b[i] + (i as f32) * x;
                }
                let d_pref_a = DeviceBuffer::from_slice(&pref_a)
                    .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
                let d_pref_b = DeviceBuffer::from_slice(&pref_b)
                    .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
                self.launch_batch_kernel_prefix(
                    &d_pref_a,
                    &d_pref_b,
                    &d_periods,
                    series_len,
                    n_combos,
                    first_valid,
                    &mut d_out,
                )?;
                // Introspection
                let block_x = match self.policy.batch {
                    WmaBatchKernelPolicy::Plain { block_x } => block_x.max(1),
                    _ => 256,
                };
                unsafe {
                    (*(self as *const _ as *mut CudaWma)).last_batch =
                        Some(WmaBatchKernelSelected::Prefix { block_x });
                }
                self.maybe_log_batch_debug();
            }
            Path::Rolling => {
                self.launch_batch_kernel_rolling(
                    &d_prices,
                    &d_periods,
                    series_len,
                    n_combos,
                    first_valid,
                    &mut d_out,
                )?;
            }
            Path::Plain => {
                self.launch_batch_kernel(
                    &d_prices,
                    &d_periods,
                    series_len,
                    n_combos,
                    first_valid,
                    max_period,
                    &mut d_out,
                )?;
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn wma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &WmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaWmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)
    }

    pub fn wma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &WmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<WmaParams>), CudaWmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaWmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn wma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        max_period: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWmaError> {
        if series_len <= 0 || n_combos <= 0 || max_period <= 1 {
            return Err(CudaWmaError::InvalidInput(
                "series_len, n_combos must be positive and period > 1".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            series_len as usize,
            n_combos as usize,
            first_valid.max(0) as usize,
            max_period as usize,
            d_out,
        )
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WmaParams,
    ) -> Result<(Vec<i32>, usize), CudaWmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaWmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaWmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        if period <= 1 {
            return Err(CudaWmaError::InvalidInput(format!(
                "invalid period {} (must be > 1)",
                period
            )));
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
            let found =
                fv.ok_or_else(|| CudaWmaError::InvalidInput(format!("series {} all NaN", series)))?;
            if (rows as i32 - found) < period as i32 {
                return Err(CudaWmaError::InvalidInput(format!(
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
    ) -> Result<(), CudaWmaError> {
        // Policy selection and introspection
        let block_x: u32 = match self.policy.many_series {
            WmaManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
            _ => 128,
        };
        unsafe {
            let this = self as *const _ as *mut CudaWma;
            (*this).last_many = Some(WmaManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes = if self.ramp_inited && period <= WMA_MAX_PERIOD {
            0
        } else {
            period
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| CudaWmaError::InvalidInput("shared memory size overflow".into()))?
        };
        let max_smem = Device::get_device(self.device_id)
            .ok()
            .and_then(|d| {
                d.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)
                    .ok()
            })
            .unwrap_or(96 * 1024) as usize;
        if shared_bytes > max_smem {
            return Err(CudaWmaError::InvalidInput(format!(
                "period {} requires {} bytes shared memory (exceeds limit {})",
                period, shared_bytes, max_smem
            )));
        }

        let func = self
            .module
            .get_function("wma_multi_series_one_param_time_major_f32")
            .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)
                .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn launch_batch_kernel_prefix(
        &self,
        d_pref_a: &DeviceBuffer<f32>,
        d_pref_b: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWmaError> {
        // Policy (only Plain path implemented)
        let block_x: u32 = match self.policy.batch {
            WmaBatchKernelPolicy::Plain { block_x } => block_x.max(1),
            _ => 256,
        };
        unsafe {
            let this = self as *const _ as *mut CudaWma;
            (*this).last_batch = Some(WmaBatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let func = self
            .module
            .get_function("wma_batch_prefix_f32")
            .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
        for (start, len) in Self::grid_y_chunks(n_combos) {
            unsafe {
                let mut pref_a_ptr = d_pref_a.as_device_ptr().as_raw();
                let mut pref_b_ptr = d_pref_b.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut series_len_i = series_len as i32;
                let mut combos_i = len as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                let args: &mut [*mut c_void] = &mut [
                    &mut pref_a_ptr as *mut _ as *mut c_void,
                    &mut pref_b_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
            }
        }
        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
    ) -> Result<DeviceArrayF32, CudaWmaError> {
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let first_valid_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaWmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_tm_f32).map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(&d_prices, period, cols, rows, &d_first_valids, &mut d_out)?;

        self.stream
            .synchronize()
            .map_err(|e| CudaWmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn wma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WmaParams,
    ) -> Result<DeviceArrayF32, CudaWmaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn wma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WmaParams,
        out: &mut [f32],
    ) -> Result<(), CudaWmaError> {
        if out.len() != cols * rows {
            return Err(CudaWmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                cols * rows
            )));
        }
        let arr =
            self.wma_multi_series_one_param_time_major_dev(data_tm_f32, cols, rows, params)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaWmaError::Cuda(e.to_string()))
    }

    pub fn wma_multi_series_one_param_time_major_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWmaError> {
        if period <= 1 || num_series <= 0 || series_len <= 0 {
            return Err(CudaWmaError::InvalidInput(
                "period must be > 1 and dimensions positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out,
        )
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        wma_benches,
        CudaWma,
        crate::indicators::moving_averages::wma::WmaBatchRange,
        crate::indicators::moving_averages::wma::WmaParams,
        wma_batch_dev,
        wma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::wma::WmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::wma::WmaParams { period: Some(64) },
        "wma",
        "wma"
    );
    pub use wma_benches::bench_profiles;
}
