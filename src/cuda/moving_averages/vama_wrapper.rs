//! CUDA support for the Volatility Adjusted Moving Average (VAMA) indicator.
//!
//! Parity with ALMA wrapper for API shape and policy while keeping
//! VAMA-specific math patterns (EMA + rolling max/min deviation window).
//!
//! - Batch (one series × many params): `vama_batch_f32`
//! - Many-series × one param (time-major): `vama_many_series_one_param_f32`
//!
//! Notes:
//! - JIT uses DetermineTargetFromContext + OptLevel O2 with fallbacks, mirroring ALMA.
//! - Stream is NON_BLOCKING.
//! - Adds light policy and introspection, VRAM checks, and chunking of large combo counts.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::volatility_adjusted_ma::{VamaBatchRange, VamaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

// -------- Kernel selection policy (kept simple; no tiled variants for VAMA) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    /// Plain 1D grid over combos; `block_x` threads per block
    Plain {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    /// 2D grid (x over time, y over series); `block_x` threads
    OneD {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaVamaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for BatchKernelPolicy {
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
}
impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        ManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}
#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Debug)]
pub enum CudaVamaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaVamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaVamaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaVamaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaVamaError {}

pub struct CudaVama {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaVamaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

struct PreparedVamaBatch {
    combos: Vec<VamaParams>,
    first_valid: usize,
    series_len: usize,
    base_periods: Vec<i32>,
    vol_periods: Vec<i32>,
    alphas: Vec<f32>,
    betas: Vec<f32>,
}

struct PreparedVamaManySeries {
    first_valids: Vec<i32>,
    base_period: usize,
    vol_period: usize,
    alpha: f32,
    beta: f32,
}

impl CudaVama {
    /// Create a new `CudaVama` on `device_id` and load the PTX module with ALMA-style JIT options.
    pub fn new(device_id: usize) -> Result<Self, CudaVamaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaVamaError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/vama_kernel.ptx"));
        // Keep conservative default JIT options; ALMA-style O2 is available but disabled here
        // to ensure bitwise-stable behavior across drivers for this numerically sensitive path.
        // If desired, we can revisit enabling O2 with driver guards.
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaVamaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create using an explicit policy.
    pub fn new_with_policy(
        device_id: usize,
        policy: CudaVamaPolicy,
    ) -> Result<Self, CudaVamaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaVamaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaVamaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
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
                    eprintln!("[DEBUG] VAMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaVama)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] VAMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaVama)).debug_many_logged = true;
                }
            }
        }
    }

    // ---------- VRAM utilities ----------
    #[inline]
    fn mem_check_enabled() -> bool {
        match std::env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
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
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    pub fn vama_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &VamaBatchRange,
    ) -> Result<DeviceArrayF32, CudaVamaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        // VRAM estimation: inputs + params + ema + outputs
        let headroom = 64usize * 1024 * 1024; // ~64MB
        let price_bytes = prepared.series_len * std::mem::size_of::<f32>();
        let params_bytes =
            n_combos * (std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f32>() * 2);
        let work_bytes = n_combos * prepared.series_len * std::mem::size_of::<f32>() * 2; // ema + out
        let total_est = price_bytes + params_bytes + work_bytes;
        if !Self::will_fit(total_est, headroom) {
            return Err(CudaVamaError::Cuda(format!(
                "insufficient VRAM for VAMA batch: need ~{} MB",
                (total_est as f64 / (1024.0 * 1024.0)).ceil() as u64
            )));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
        let d_base = DeviceBuffer::from_slice(&prepared.base_periods)
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
        let d_vol = DeviceBuffer::from_slice(&prepared.vol_periods)
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
        let d_alphas = DeviceBuffer::from_slice(&prepared.alphas)
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
        let d_betas = DeviceBuffer::from_slice(&prepared.betas)
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
        let mut d_ema: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaVamaError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaVamaError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_base,
            &d_vol,
            &d_alphas,
            &d_betas,
            prepared.series_len,
            prepared.first_valid,
            n_combos,
            &mut d_ema,
            &mut d_out,
        )?;

        // Ensure completion before returning VRAM handle for consistency
        self.stream
            .synchronize()
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn vama_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_base_periods: &DeviceBuffer<i32>,
        d_vol_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        d_betas: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_ema: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVamaError> {
        if series_len == 0 {
            return Err(CudaVamaError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaVamaError::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, series_len
            )));
        }
        if n_combos == 0 {
            return Err(CudaVamaError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if d_base_periods.len() != n_combos
            || d_vol_periods.len() != n_combos
            || d_alphas.len() != n_combos
            || d_betas.len() != n_combos
        {
            return Err(CudaVamaError::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }
        if d_ema.len() != n_combos * series_len || d_out.len() != n_combos * series_len {
            return Err(CudaVamaError::InvalidInput(
                "output buffers must equal combos * series_len".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_base_periods,
            d_vol_periods,
            d_alphas,
            d_betas,
            series_len,
            first_valid,
            n_combos,
            d_ema,
            d_out,
        )
    }

    pub fn vama_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &VamaParams,
    ) -> Result<DeviceArrayF32, CudaVamaError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
        let mut d_ema: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)
                .map_err(|e| CudaVamaError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)
                .map_err(|e| CudaVamaError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first_valids,
            prepared.base_period,
            prepared.vol_period,
            prepared.alpha,
            prepared.beta,
            num_series,
            series_len,
            &mut d_ema,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
        })
    }

    pub fn vama_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &VamaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaVamaError> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaVamaError::InvalidInput(format!(
                "output slice wrong length: got {}, expected {}",
                out_tm.len(),
                num_series * series_len
            )));
        }

        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
        let mut d_ema: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)
                .map_err(|e| CudaVamaError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)
                .map_err(|e| CudaVamaError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first_valids,
            prepared.base_period,
            prepared.vol_period,
            prepared.alpha,
            prepared.beta,
            num_series,
            series_len,
            &mut d_ema,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;

        d_out
            .copy_to(out_tm)
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))
    }

    pub fn vama_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        base_period: i32,
        vol_period: i32,
        alpha: f32,
        beta: f32,
        num_series: i32,
        series_len: i32,
        d_ema: &mut DeviceBuffer<f32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVamaError> {
        if base_period <= 0 || vol_period <= 0 {
            return Err(CudaVamaError::InvalidInput(
                "base_period and vol_period must be positive".into(),
            ));
        }
        if num_series <= 0 || series_len <= 0 {
            return Err(CudaVamaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if d_first_valids.len() != num_series as usize {
            return Err(CudaVamaError::InvalidInput(
                "first_valids length mismatch".into(),
            ));
        }
        if d_ema.len() != (num_series as usize) * (series_len as usize)
            || d_out_tm.len() != (num_series as usize) * (series_len as usize)
        {
            return Err(CudaVamaError::InvalidInput(
                "output buffers must match num_series * series_len".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            base_period as usize,
            vol_period as usize,
            alpha,
            beta,
            num_series as usize,
            series_len as usize,
            d_ema,
            d_out_tm,
        )
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &VamaBatchRange,
    ) -> Result<PreparedVamaBatch, CudaVamaError> {
        if data_f32.is_empty() {
            return Err(CudaVamaError::InvalidInput("input data is empty".into()));
        }
        let combos = expand_vama_grid(sweep);
        if combos.is_empty() {
            return Err(CudaVamaError::InvalidInput(
                "no parameter combinations provided".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaVamaError::InvalidInput("all values are NaN".into()))?;

        let mut base_periods = Vec::with_capacity(combos.len());
        let mut vol_periods = Vec::with_capacity(combos.len());
        let mut alphas = Vec::with_capacity(combos.len());
        let mut betas = Vec::with_capacity(combos.len());

        for params in &combos {
            let base = params.base_period.unwrap_or(0);
            let vol = params.vol_period.unwrap_or(0);
            if base == 0 || vol == 0 {
                return Err(CudaVamaError::InvalidInput(
                    "periods must be positive".into(),
                ));
            }
            let needed = base.max(vol);
            if series_len - first_valid < needed {
                return Err(CudaVamaError::InvalidInput(format!(
                    "not enough valid data: need >= {}, have {}",
                    needed,
                    series_len - first_valid
                )));
            }

            base_periods.push(base as i32);
            vol_periods.push(vol as i32);
            let alpha = 2.0f32 / (base as f32 + 1.0f32);
            alphas.push(alpha);
            betas.push(1.0f32 - alpha);
        }

        Ok(PreparedVamaBatch {
            combos,
            first_valid,
            series_len,
            base_periods,
            vol_periods,
            alphas,
            betas,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_base: &DeviceBuffer<i32>,
        d_vol: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        d_betas: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_ema: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVamaError> {
        if n_combos == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("vama_batch_f32")
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;

        // Policy selection (simple; no tiled variants). Allow override via env VAMA_BLOCK_X.
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => std::env::var("VAMA_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|&v| v > 0)
                .unwrap_or(256),
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        unsafe {
            let this = self as *const _ as *mut CudaVama;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        // Large combo counts: chunk launches to avoid extreme grid.x sizes (safety parity with ALMA's grid.y chunking guidance)
        const MAX_GRID_DIM: usize = 65_535; // conservative bound per dimension for portability
        let mut launch_chunk =
            |start_combo: usize, chunk_len: usize| -> Result<(), CudaVamaError> {
                unsafe {
                    // Launch covers [start_combo, start_combo + chunk_len).
                    // Offset param pointers and ema/out bases to the chunk start.
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut base_ptr = d_base.as_device_ptr().add(start_combo).as_raw();
                    let mut vol_ptr = d_vol.as_device_ptr().add(start_combo).as_raw();
                    let mut alpha_ptr = d_alphas.as_device_ptr().add(start_combo).as_raw();
                    let mut beta_ptr = d_betas.as_device_ptr().add(start_combo).as_raw();
                    let mut series_len_i = series_len as i32;
                    let mut first_valid_i = first_valid as i32;
                    let mut combos_i = chunk_len as i32;
                    let mut ema_ptr = d_ema.as_device_ptr().add(start_combo * series_len).as_raw();
                    let mut out_ptr = d_out.as_device_ptr().add(start_combo * series_len).as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut base_ptr as *mut _ as *mut c_void,
                        &mut vol_ptr as *mut _ as *mut c_void,
                        &mut alpha_ptr as *mut _ as *mut c_void,
                        &mut beta_ptr as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut first_valid_i as *mut _ as *mut c_void,
                        &mut combos_i as *mut _ as *mut c_void,
                        &mut ema_ptr as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    let grid: GridSize = (chunk_len as u32, 1, 1).into();
                    let block: BlockSize = (block_x, 1, 1).into();
                    self.stream
                        .launch(&func, grid, block, 0, args)
                        .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
                }
                Ok(())
            };

        let mut remaining = n_combos;
        let mut start = 0usize;
        while remaining > 0 {
            let take = remaining.min(MAX_GRID_DIM);
            launch_chunk(start, take)?;
            start += take;
            remaining -= take;
        }
        Ok(())
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &VamaParams,
    ) -> Result<PreparedVamaManySeries, CudaVamaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaVamaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaVamaError::InvalidInput(format!(
                "time-major slice length mismatch: got {}, expected {}",
                data_tm_f32.len(),
                num_series * series_len
            )));
        }

        let base_period = params.base_period.unwrap_or(113);
        let vol_period = params.vol_period.unwrap_or(51);
        if base_period == 0 || vol_period == 0 {
            return Err(CudaVamaError::InvalidInput(
                "base_period and vol_period must be positive".into(),
            ));
        }
        if params.smoothing.unwrap_or(true) {
            return Err(CudaVamaError::InvalidInput(
                "CUDA VAMA many-series path does not support smoothing".into(),
            ));
        }

        let needed = base_period.max(vol_period);
        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut first_valid: Option<usize> = None;
            for t in 0..series_len {
                let value = data_tm_f32[t * num_series + series];
                if value.is_finite() {
                    first_valid = Some(t);
                    break;
                }
            }
            let fv = first_valid.ok_or_else(|| {
                CudaVamaError::InvalidInput(format!("series {} is entirely NaN", series))
            })?;

            if series_len - fv < needed {
                return Err(CudaVamaError::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    series,
                    needed,
                    series_len - fv
                )));
            }

            first_valids.push(fv as i32);
        }

        let alpha = 2.0f32 / (base_period as f32 + 1.0f32);
        let beta = 1.0f32 - alpha;

        Ok(PreparedVamaManySeries {
            first_valids,
            base_period,
            vol_period,
            alpha,
            beta,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        base_period: usize,
        vol_period: usize,
        alpha: f32,
        beta: f32,
        num_series: usize,
        series_len: usize,
        d_ema: &mut DeviceBuffer<f32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVamaError> {
        if num_series == 0 || series_len == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("vama_many_series_one_param_f32")
            .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => std::env::var("VAMA_MANY_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|&v| v > 0)
                .unwrap_or(128),
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        unsafe {
            let this = self as *const _ as *mut CudaVama;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), num_series as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut base_i = base_period as i32;
            let mut vol_i = vol_period as i32;
            let mut alpha_f = alpha;
            let mut beta_f = beta;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut ema_ptr = d_ema.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut base_i as *mut _ as *mut c_void,
                &mut vol_i as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
                &mut beta_f as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut ema_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaVamaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        vama_benches,
        CudaVama,
        crate::indicators::moving_averages::volatility_adjusted_ma::VamaBatchRange,
        crate::indicators::moving_averages::volatility_adjusted_ma::VamaParams,
        vama_batch_dev,
        vama_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::volatility_adjusted_ma::VamaBatchRange {
            base_period: (16, 16 + PARAM_SWEEP - 1, 1),
            vol_period: (51, 51, 0)
        },
        crate::indicators::moving_averages::volatility_adjusted_ma::VamaParams {
            base_period: Some(64),
            vol_period: Some(51),
            smoothing: Some(false),
            smooth_type: Some(3),
            smooth_period: Some(5)
        },
        "vama",
        "vama"
    );
    pub use vama_benches::bench_profiles;
}

fn expand_vama_grid(range: &VamaBatchRange) -> Vec<VamaParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let base = axis(range.base_period);
    let vol = axis(range.vol_period);
    let mut out = Vec::with_capacity(base.len() * vol.len());
    for &b in &base {
        for &v in &vol {
            out.push(VamaParams {
                base_period: Some(b),
                vol_period: Some(v),
                smoothing: Some(false),
                smooth_type: Some(3),
                smooth_period: Some(5),
            });
        }
    }
    out
}
