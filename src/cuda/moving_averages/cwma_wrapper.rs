//! CUDA support for the Cubic Weighted Moving Average (CWMA).
//!
//! This mirrors the public API offered by the ALMA CUDA wrapper but is tailored
//! to the simpler parameter surface of CWMA (period only). Kernels operate in
//! single precision and return VRAM-resident buffers so higher layers (Python,
//! bindings, etc.) can decide when to stage back to host memory.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::cwma::{CwmaBatchRange, CwmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, CopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaCwmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaCwmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaCwmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaCwmaError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}

impl std::error::Error for CudaCwmaError {}

// -------- Kernel selection policy (mirrors ALMA) --------

/// Whether each thread computes one output or two outputs in fused fashion.
#[derive(Clone, Copy, Debug)]
pub enum BatchThreadsPerOutput { One, Two }

/// Policy to select the batch (one-series × many-params) kernel implementation.
#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
    Tiled { tile: u32, per_thread: BatchThreadsPerOutput },
}

/// Policy to select the many-series (time-major) kernel implementation.
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

/// Aggregate CUDA policy used by CudaCwma.
#[derive(Clone, Copy, Debug)]
pub struct CudaCwmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaCwmaPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

// -------- Introspection (selected kernel) --------

/// Introspection for the selected batch kernel at last launch.
#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
    Tiled1x { tile: u32 },
    Tiled2x { tile: u32 },
}

/// Introspection for the selected many-series kernel at last launch.
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

/// CUDA CWMA wrapper providing batched and many-series execution paths.
///
/// - FP32 compute; uses host-precomputed weights per parameter and pre-scales
///   them to remove per-output multiplies.
/// - Exposes kernel selection policies similar to ALMA for deterministic
///   benches and debugging.
pub struct CudaCwma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaCwmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaCwma {
    pub fn new(device_id: usize) -> Result<Self, CudaCwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cwma_kernel.ptx"));
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaCwmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaCwmaPolicy) -> Result<Self, CudaCwmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaCwmaPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaCwmaPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    pub fn synchronize(&self) -> Result<(), CudaCwmaError> {
        self.stream.synchronize().map_err(|e| CudaCwmaError::Cuda(e.to_string()))
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
    fn grid_y_chunks(n_combos: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX_GRID_Y: usize = 65_535;
        (0..n_combos)
            .step_by(MAX_GRID_Y)
            .map(move |start| {
                let len = (n_combos - start).min(MAX_GRID_Y);
                (start, len)
            })
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] CWMA batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaCwma)).debug_batch_logged = true; }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] CWMA many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaCwma)).debug_many_logged = true; }
            }
        }
    }

    fn expand_periods(range: &CwmaBatchRange) -> Vec<CwmaParams> {
        let (start, end, step) = range.period;
        let periods = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect::<Vec<_>>()
        };
        periods
            .into_iter()
            .map(|p| CwmaParams { period: Some(p) })
            .collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &CwmaBatchRange,
    ) -> Result<(Vec<CwmaParams>, usize, usize, usize), CudaCwmaError> {
        if data_f32.is_empty() {
            return Err(CudaCwmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaCwmaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaCwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period <= 1 {
                return Err(CudaCwmaError::InvalidInput(format!(
                    "invalid period {} (must be > 1)",
                    period
                )));
            }
            if period > series_len {
                return Err(CudaCwmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, series_len
                )));
            }
            if series_len - first_valid < period {
                return Err(CudaCwmaError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    period,
                    series_len - first_valid
                )));
            }
            max_period = max_period.max(period);
        }

        Ok((combos, first_valid, series_len, max_period))
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[CwmaParams],
        first_valid: usize,
        series_len: usize,
        max_period: usize,
    ) -> Result<DeviceArrayF32, CudaCwmaError> {
        let n_combos = combos.len();
        let weights_stride = max_period;
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let weights_bytes = n_combos * weights_stride * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let inv_norm_bytes = n_combos * std::mem::size_of::<f32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + periods_bytes + inv_norm_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaCwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let mut periods_i32 = vec![0i32; n_combos];
        let mut weights_flat = vec![0f32; n_combos * weights_stride];
        let mut inv_norms = vec![0f32; n_combos];

        for (idx, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap();
            let weight_len = period - 1;
            let mut norm = 0.0f32;
            for k in 0..weight_len {
                let weight = ((period - k) as f32).powi(3);
                weights_flat[idx * weights_stride + k] = weight;
                norm += weight;
            }
            if norm == 0.0 {
                return Err(CudaCwmaError::InvalidInput(format!(
                    "period {} produced zero normalization",
                    period
                )));
            }
            // Pre-scale weights by inv_norm to drop per-output multiply in kernels
            let inv = 1.0 / norm;
            for k in 0..weight_len {
                let base = idx * weights_stride + k;
                weights_flat[base] *= inv;
            }
            periods_i32[idx] = period as i32;
            inv_norms[idx] = 1.0; // weights already scaled
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let d_inv_norms =
            DeviceBuffer::from_slice(&inv_norms).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_weights,
            &d_periods,
            &d_inv_norms,
            series_len,
            n_combos,
            first_valid,
            max_period,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_inv_norms: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCwmaError> {
        // Decide kernel by policy: plain vs tiled (1x or 2x)
        let mut use_tiled = series_len > 8192;
        let mut tile_x: u32 = 256;
        let mut use_two = false;
        match self.policy.batch {
            BatchKernelPolicy::Auto => {}
            BatchKernelPolicy::Plain { block_x } => { use_tiled = false; tile_x = block_x; }
            BatchKernelPolicy::Tiled { tile, per_thread } => {
                use_tiled = true; tile_x = tile; use_two = matches!(per_thread, BatchThreadsPerOutput::Two);
            }
        }

        // Helper: compute dynamic shared memory size for tiled batch kernels:
        // bytes = align16(wlen*4) + (tile_x + wlen)*4, with wlen <= max_period-1 and using worst-case wlen.
        let align16 = |x: usize| (x + 15) & !15usize;
        if use_tiled {
            // Choose function name
            let func_name = if use_two {
                match tile_x { 128 => "cwma_batch_tiled_f32_2x_tile128", _ => "cwma_batch_tiled_f32_2x_tile256" }
            } else {
                match tile_x { 128 => "cwma_batch_tiled_f32_tile128",  _ => "cwma_batch_tiled_f32_tile256" }
            };
            let func = self
                .module
                .get_function(func_name)
                .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

            // Introspection
            unsafe {
                let this = self as *const _ as *mut CudaCwma;
                (*this).last_batch = Some(if use_two { BatchKernelSelected::Tiled2x { tile: tile_x } } else { BatchKernelSelected::Tiled1x { tile: tile_x } });
            }
            self.maybe_log_batch_debug();

            let block_x = if use_two { (tile_x / 2).max(1) } else { tile_x };
            let grid_x = ((series_len as u32) + tile_x - 1) / tile_x;
            let block: BlockSize = (block_x, 1, 1).into();
            // Shared: weights (<= wlen=max_period-1) aligned to 16B + tile (tile_x + wlen)
            let wlen = max_period.saturating_sub(1);
            let shared_bytes = (align16(wlen * std::mem::size_of::<f32>())
                + (tile_x as usize + wlen) * std::mem::size_of::<f32>()) as u32;

            for (start, len) in Self::grid_y_chunks(n_combos) {
                let grid: GridSize = (grid_x, len as u32, 1).into();
                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    // Offset parameter arrays to this Y-slice
                    let mut weights_ptr = unsafe { d_weights.as_device_ptr().add(start * max_period).as_raw() };
                    let mut periods_ptr = unsafe { d_periods.as_device_ptr().add(start).as_raw() };
                    let mut inv_ptr = unsafe { d_inv_norms.as_device_ptr().add(start).as_raw() };
                    let mut max_period_i = max_period as i32;
                    let mut series_len_i = series_len as i32;
                    let mut n_combos_i = len as i32;
                    let mut first_valid_i = first_valid as i32;
                    // Offset output pointer by start * series_len elements
                    let mut out_ptr = unsafe { d_out.as_device_ptr().add(start * series_len).as_raw() };
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut weights_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut inv_ptr as *mut _ as *mut c_void,
                        &mut max_period_i as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut n_combos_i as *mut _ as *mut c_void,
                        &mut first_valid_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream
                        .launch(&func, grid, block, shared_bytes, args)
                        .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
                }
            }
        } else {
            // Plain kernel path
            let block_x: u32 = tile_x.max(1);
            let grid_x = ((series_len as u32) + block_x - 1) / block_x;
            let block: BlockSize = (block_x, 1, 1).into();
            let shared_bytes = ((max_period.saturating_sub(1)) * std::mem::size_of::<f32>()) as u32;

            let func = self
                .module
                .get_function("cwma_batch_f32")
                .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

            for (start, len) in Self::grid_y_chunks(n_combos) {
                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut weights_ptr = d_weights.as_device_ptr().add(start * max_period).as_raw();
                    let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                    let mut inv_ptr = d_inv_norms.as_device_ptr().add(start).as_raw();
                    let mut max_period_i = max_period as i32;
                    let mut series_len_i = series_len as i32;
                    let mut n_combos_i = len as i32;
                    let mut first_valid_i = first_valid as i32;
                    let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                    let grid: GridSize = (grid_x, len as u32, 1).into();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut weights_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut inv_ptr as *mut _ as *mut c_void,
                        &mut max_period_i as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut n_combos_i as *mut _ as *mut c_void,
                        &mut first_valid_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream
                        .launch(&func, grid, block, shared_bytes, args)
                        .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
                }
            }
            unsafe {
                let this = self as *const _ as *mut CudaCwma;
                (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
            }
            self.maybe_log_batch_debug();
        }
        Ok(())
    }

    pub fn cwma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_inv_norms: &DeviceBuffer<f32>,
        max_period: i32,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCwmaError> {
        if max_period <= 1 || series_len <= 0 || n_combos <= 0 {
            return Err(CudaCwmaError::InvalidInput(
                "max_period, series_len, and n_combos must be positive".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_weights,
            d_periods,
            d_inv_norms,
            series_len as usize,
            n_combos as usize,
            first_valid.max(0) as usize,
            max_period as usize,
            d_out,
        )
    }

    pub fn cwma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &CwmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaCwmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)
    }

    /// Convenience: run batch when prices are already on device. Precomputes
    /// weights & periods on host (or could be extended to use on-device
    /// precompute). Returns a VRAM-backed array.
    pub fn cwma_batch_from_device_prices(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        sweep: &CwmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaCwmaError> {
        if series_len == 0 { return Err(CudaCwmaError::InvalidInput("series_len is zero".into())); }
        let combos = Self::expand_periods(sweep);
        if combos.is_empty() { return Err(CudaCwmaError::InvalidInput("no parameter combinations".into())); }
        let n_combos = combos.len();
        let max_period = combos.iter().map(|c| c.period.unwrap_or(0)).max().unwrap_or(0);
        if max_period <= 1 || series_len - first_valid < max_period {
            return Err(CudaCwmaError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                series_len - first_valid
            )));
        }

        let mut periods_i32 = vec![0i32; n_combos];
        let mut inv_norms = vec![1.0f32; n_combos];
        let mut weights_flat = vec![0f32; n_combos * max_period];
        for (idx, prm) in combos.iter().enumerate() {
            let p = prm.period.unwrap();
            let wlen = p - 1;
            let mut norm = 0.0f32;
            for k in 0..wlen { let w = ((p - k) as f32).powi(3); weights_flat[idx * max_period + k] = w; norm += w; }
            let inv = 1.0 / norm.max(1e-20);
            for k in 0..wlen { weights_flat[idx * max_period + k] *= inv; }
            periods_i32[idx] = p as i32;
        }

        let d_weights = DeviceBuffer::from_slice(&weights_flat).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        self.launch_batch_kernel(d_prices, &d_weights, &d_periods, &d_inv_norms, series_len, n_combos, first_valid, max_period, &mut d_out)?;
        self.synchronize()?;
        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: series_len })
    }

    pub fn cwma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &CwmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<CwmaParams>), CudaCwmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaCwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CwmaParams,
    ) -> Result<(Vec<i32>, usize, Vec<f32>, f32), CudaCwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaCwmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaCwmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        if period <= 1 {
            return Err(CudaCwmaError::InvalidInput(format!(
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
            let found = fv
                .ok_or_else(|| CudaCwmaError::InvalidInput(format!("series {} all NaN", series)))?;
            if (rows as i32 - found) < period as i32 {
                return Err(CudaCwmaError::InvalidInput(format!(
                    "series {} lacks data: need >= {}, valid = {}",
                    series,
                    period,
                    rows as i32 - found
                )));
            }
            first_valids[series] = found;
        }

        let weight_len = period - 1;
        let mut weights = vec![0f32; weight_len];
        let mut norm = 0.0f32;
        for k in 0..weight_len {
            let w = ((period - k) as f32).powi(3);
            weights[k] = w;
            norm += w;
        }
        if norm == 0.0 {
            return Err(CudaCwmaError::InvalidInput(format!(
                "period {} produced zero normalization",
                period
            )));
        }
        let inv_norm = 1.0 / norm;

        Ok((first_valids, period, weights, inv_norm))
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
        weights: &[f32],
        inv_norm: f32,
    ) -> Result<DeviceArrayF32, CudaCwmaError> {
        let weights_bytes = weights.len() * std::mem::size_of::<f32>();
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let first_valid_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = weights_bytes + prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaCwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(weights).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_weights,
            period,
            inv_norm,
            cols,
            rows,
            &d_first_valids,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: usize,
        inv_norm: f32,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCwmaError> {
        // Decide policy for many-series
        let mut use_tiled2d = (cols >= 64) && (rows >= 4096);
        let mut tx = 128u32; let mut ty = 4u32;
        match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => {}
            ManySeriesKernelPolicy::OneD { block_x } => { use_tiled2d = false; tx = block_x; }
            ManySeriesKernelPolicy::Tiled2D { tx: txx, ty: tyy } => { use_tiled2d = true; tx = txx; ty = tyy; }
        }

        let func = if use_tiled2d {
            // Pick among a small set of tiled 2D kernels
            let name = match (tx, ty) {
                (128, 4) => "cwma_ms1p_tiled_f32_tx128_ty4",
                (128, 2) => "cwma_ms1p_tiled_f32_tx128_ty2",
                _ => "cwma_ms1p_tiled_f32_tx128_ty4",
            };
            let f = self.module.get_function(name).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
            unsafe { let this = self as *const _ as *mut CudaCwma; (*this).last_many = Some(ManySeriesKernelSelected::Tiled2D { tx, ty }); }
            self.maybe_log_many_debug();
            f
        } else {
            let f = self.module.get_function("cwma_multi_series_one_param_time_major_f32").map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
            let block1d_x: u32 = match self.policy.many_series { ManySeriesKernelPolicy::OneD { block_x } => block_x, _ => 128 };
            unsafe { let this = self as *const _ as *mut CudaCwma; (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x: block1d_x }); }
            self.maybe_log_many_debug();
            f
        };

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut inv = inv_norm;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fvalid_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut inv as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fvalid_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            if use_tiled2d {
                // Shared bytes = align16(wlen*4) + ( (tx + wlen) * ty * 4 )
                let wlen = period.saturating_sub(1);
                let align16 = |x: usize| (x + 15) & !15usize;
                let total = tx as usize + wlen;
                let shared_bytes = (align16(wlen * std::mem::size_of::<f32>())
                    + total * ty as usize * std::mem::size_of::<f32>()) as u32;
                let grid_x = ((rows as u32) + tx - 1) / tx;
                let grid_y = ((cols as u32) + ty - 1) / ty;
                let grid: GridSize = (grid_x, grid_y, 1).into();
                let block: BlockSize = (tx, ty, 1).into();
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
            } else {
                let block_x: u32 = match self.policy.many_series { ManySeriesKernelPolicy::OneD { block_x } => block_x, _ => 128 };
                let grid_x = ((rows as u32) + block_x - 1) / block_x;
                let grid: GridSize = (grid_x, cols as u32, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                let shared_bytes = (period.saturating_sub(1) * std::mem::size_of::<f32>()) as u32;
                self.stream
                    .launch(&func, grid, block, shared_bytes, args)
                    .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
            }
        }
        Ok(())
    }

    pub fn cwma_multi_series_one_param_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        inv_norm: f32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCwmaError> {
        if period <= 1 || num_series <= 0 || series_len <= 0 {
            return Err(CudaCwmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices,
            d_weights,
            period as usize,
            inv_norm,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out,
        )
    }

    pub fn cwma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CwmaParams,
    ) -> Result<DeviceArrayF32, CudaCwmaError> {
        let (first_valids, period, weights, inv_norm) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            &weights,
            inv_norm,
        )
    }

    pub fn cwma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CwmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaCwmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaCwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period, weights, inv_norm) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let arr = self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            &weights,
            inv_norm,
        )?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        cwma_benches,
        CudaCwma,
        crate::indicators::moving_averages::cwma::CwmaBatchRange,
        crate::indicators::moving_averages::cwma::CwmaParams,
        cwma_batch_dev,
        cwma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::cwma::CwmaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::cwma::CwmaParams { period: Some(64) },
        "cwma",
        "cwma"
    );
    pub use cwma_benches::bench_profiles;
}
