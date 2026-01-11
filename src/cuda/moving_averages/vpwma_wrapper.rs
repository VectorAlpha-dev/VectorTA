//! CUDA scaffolding for VPWMA (Variable Power Weighted Moving Average).
//!
//! Mirrors the ALMA/CWMA GPU integration pattern: parameter sweeps are expanded
//! on the host, weights are precomputed in FP32 (category-appropriate), and the
//! kernel evaluates each row independently while sharing the flattened weight
//! buffer. Warmup/NaN semantics match the scalar path: write NaN up to
//! `first_valid + (period - 1)` per row/series.

#![cfg(feature = "cuda")]

use crate::indicators::moving_averages::vpwma::{expand_grid_vpwma, VpwmaBatchRange, VpwmaParams};
use cust::error::CudaError;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[inline]
fn vpwma_tile_t() -> usize {
    const DEFAULT_TILE_T: usize = 256;
    option_env!("VPWMA_TILE_T")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_TILE_T)
        .max(DEFAULT_TILE_T)
}

/// VRAM-backed array handle owned by VPWMA CUDA wrapper.
///
/// Keeps the CUDA context alive via `Arc<Context>` so that the device buffer
/// remains valid as long as this handle is live.
pub struct DeviceArrayF32 {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub(crate) ctx: Arc<Context>,
    pub(crate) device_id: u32,
}
impl DeviceArrayF32 {
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.buf.as_device_ptr().as_raw() as u64
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

#[derive(Debug, Error)]
pub enum CudaVpwmaError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("launch configuration too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("device mismatch: buffer on device {buf}, current device {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

// -------- Kernel selection policy (plain variants; mirrors ALMA/CWMA shape) --------

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
pub struct CudaVpwmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaVpwmaPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
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

pub struct CudaVpwma {
    module: Module,
    stream: Stream,
    ctx: Arc<Context>,
    device_id: u32,
    policy: CudaVpwmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaVpwma {
    pub fn new(device_id: usize) -> Result<Self, CudaVpwmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vpwma_kernel.ptx"));
        // Prefer context-targeted and O2; gracefully fall back for brittle drivers
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
                    Module::from_ptx(ptx, &[])?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            ctx: Arc::new(context),
            device_id: device_id as u32,
            policy: CudaVpwmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaVpwmaPolicy,
    ) -> Result<Self, CudaVpwmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaVpwmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaVpwmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaVpwmaError> {
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline]
    fn ensure_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaVpwmaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaVpwmaError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
        }
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
                    eprintln!("[DEBUG] VPWMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaVpwma)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] VPWMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaVpwma)).debug_many_logged = true;
                }
            }
        }
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &VpwmaBatchRange,
    ) -> Result<(Vec<VpwmaParams>, usize, usize), CudaVpwmaError> {
        if data_f32.is_empty() {
            return Err(CudaVpwmaError::InvalidInput("empty data".into()));
        }

        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaVpwmaError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_vpwma(sweep);
        if combos.is_empty() {
            return Err(CudaVpwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = data_f32.len();
        let mut max_period = 0usize;
        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            let power = combo.power.unwrap_or(f64::NAN);
            if period < 2 {
                return Err(CudaVpwmaError::InvalidInput(
                    "period must be at least 2".into(),
                ));
            }
            if period > len {
                return Err(CudaVpwmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if power.is_nan() || power.is_infinite() {
                return Err(CudaVpwmaError::InvalidInput("power must be finite".into()));
            }
            max_period = max_period.max(period);
        }

        if len - first_valid < max_period {
            return Err(CudaVpwmaError::InvalidInput(format!(
                "not enough valid data (need >= {}, have {} after first valid)",
                max_period,
                len - first_valid
            )));
        }

        Ok((combos, first_valid, len))
    }

    fn compute_weights(period: usize, power: f64) -> Result<(Vec<f32>, f32), CudaVpwmaError> {
        if !power.is_finite() {
            return Err(CudaVpwmaError::InvalidInput("power must be finite".into()));
        }
        if period < 2 {
            return Err(CudaVpwmaError::InvalidInput(
                "period must be at least 2".into(),
            ));
        }
        let win_len = period - 1;
        let mut weights = vec![0f32; win_len];
        let mut norm = 0.0f64;
        for k in 0..win_len {
            let w = (period as f64 - k as f64).powf(power);
            weights[k] = w as f32;
            norm += w;
        }
        if !norm.is_finite() || norm == 0.0 {
            return Err(CudaVpwmaError::InvalidInput(format!(
                "invalid normalization for period {} power {}",
                period, power
            )));
        }
        Ok((weights, (1.0 / norm) as f32))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VpwmaParams,
    ) -> Result<(Vec<i32>, usize, Vec<f32>, f32), CudaVpwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaVpwmaError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaVpwmaError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }

        let period = params.period.unwrap_or(0);
        let power = params.power.unwrap_or(f64::NAN);
        if period < 2 {
            return Err(CudaVpwmaError::InvalidInput(
                "period must be at least 2".into(),
            ));
        }
        if !power.is_finite() {
            return Err(CudaVpwmaError::InvalidInput("power must be finite".into()));
        }

        let stride = cols;
        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let val = data_tm_f32[row * stride + series];
                if !val.is_nan() {
                    found = Some(row as i32);
                    break;
                }
            }
            let first = found.ok_or_else(|| {
                CudaVpwmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - (first as usize) < period {
                return Err(CudaVpwmaError::InvalidInput(format!(
                    "series {} does not have enough data for period {}",
                    series, period
                )));
            }
            first_valids[series] = first;
        }

        let (weights, inv_norm) = Self::compute_weights(period, power)?;
        Ok((first_valids, period, weights, inv_norm))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_win_lengths: &DeviceBuffer<i32>,
        d_weights: &DeviceBuffer<f32>,
        d_inv_norms: &DeviceBuffer<f32>,
        len: usize,
        stride: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVpwmaError> {
        use cust::context::CacheConfig;
        let mut func = self.module.get_function("vpwma_batch_f32").map_err(|_| CudaVpwmaError::MissingKernelSymbol { name: "vpwma_batch_f32" })?;
        let _ = func.set_cache_config(CacheConfig::PreferShared);

        // One block per combo (CTA-per-row tiled kernel path)
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 128,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
        };
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        // Dynamic shared memory requirement per CTA = weights(stride) + tile(T) + (stride-1)
        // in floats, then *4 bytes.
        let t_tile = vpwma_tile_t();
        let smem_floats = t_tile
            .checked_add(2usize.saturating_mul(stride))
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| {
                CudaVpwmaError::InvalidInput("dynamic shared memory size overflow".into())
            })?;
        let smem_bytes_u32: u32 = smem_floats
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|b| u32::try_from(b).ok())
            .ok_or_else(|| {
                CudaVpwmaError::InvalidInput("dynamic shared memory size overflow".into())
            })?;

        if let Ok(avail) = func.available_dynamic_shared_memory_per_block(grid, block) {
            if (smem_bytes_u32 as usize) > avail {
                return Err(CudaVpwmaError::LaunchConfigTooLarge { gx: grid.x, gy: grid.y, gz: grid.z, bx: block.x, by: block.y, bz: block.z });
            }
        }

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut win_ptr = d_win_lengths.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut inv_ptr = d_inv_norms.as_device_ptr().as_raw();
            let mut series_len_i = len as i32;
            let mut stride_i = stride as i32;
            let mut first_valid_i = first_valid as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut win_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut inv_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut stride_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, smem_bytes_u32, args)?;
        }
        // Record introspection for debug once-per-run
        unsafe {
            (*(self as *const _ as *mut CudaVpwma)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        d_weights: &DeviceBuffer<f32>,
        inv_norm: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVpwmaError> {
        use cust::context::CacheConfig;
        let mut func = self.module.get_function("vpwma_many_series_one_param_f32").map_err(|_| CudaVpwmaError::MissingKernelSymbol { name: "vpwma_many_series_one_param_f32" })?;
        let _ = func.set_cache_config(CacheConfig::PreferShared);

        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
        };
        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        // Dynamic shared memory for per-CTA weight cache: win_len floats
        let win_len = period.saturating_sub(1);
        let smem_bytes_u32: u32 = win_len
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|b| u32::try_from(b).ok())
            .ok_or_else(|| {
                CudaVpwmaError::InvalidInput("dynamic shared memory size overflow".into())
            })?;

        if let Ok(avail) = func.available_dynamic_shared_memory_per_block(grid, block) {
            if (smem_bytes_u32 as usize) > avail {
                return Err(CudaVpwmaError::LaunchConfigTooLarge { gx: grid.x, gy: grid.y, gz: grid.z, bx: block.x, by: block.y, bz: block.z });
            }
        }

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut period_i = period as i32;
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut inv = inv_norm;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut inv as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, smem_bytes_u32, args)?;
        }
        // Record introspection for debug once-per-run
        unsafe {
            (*(self as *const _ as *mut CudaVpwma)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[VpwmaParams],
        first_valid: usize,
        len: usize,
    ) -> Result<DeviceArrayF32, CudaVpwmaError> {
        let n_combos = combos.len();
        let mut periods = Vec::with_capacity(n_combos);
        let mut win_lengths = Vec::with_capacity(n_combos);
        let mut inv_norms = Vec::with_capacity(n_combos);

        let stride = combos
            .iter()
            .map(|c| c.period.unwrap() - 1)
            .max()
            .unwrap_or(1);

        // VRAM estimate (prices + params + weights + out) with ~64MB headroom
        let prices_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaVpwmaError::InvalidInput("byte-size overflow".into()))?;
        let weights_bytes = n_combos
            .checked_mul(stride)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaVpwmaError::InvalidInput("byte-size overflow".into()))?;
        let periods_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaVpwmaError::InvalidInput("byte-size overflow".into()))?;
        let winlens_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaVpwmaError::InvalidInput("byte-size overflow".into()))?;
        let invnorm_bytes = n_combos
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaVpwmaError::InvalidInput("byte-size overflow".into()))?;
        let out_bytes = n_combos
            .checked_mul(len)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaVpwmaError::InvalidInput("byte-size overflow".into()))?;
        let required = prices_bytes
            .checked_add(weights_bytes)
            .and_then(|v| v.checked_add(periods_bytes))
            .and_then(|v| v.checked_add(winlens_bytes))
            .and_then(|v| v.checked_add(invnorm_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaVpwmaError::InvalidInput("byte-size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::ensure_fit(required, headroom)?;

        let d_prices = DeviceBuffer::from_slice(data_f32)?;

        let mut weights_flat = vec![0f32; n_combos * stride];

        for (idx, combo) in combos.iter().enumerate() {
            let period = combo.period.unwrap();
            let power = combo.power.unwrap();
            let win_len = period - 1;

            periods.push(period as i32);
            win_lengths.push(win_len as i32);

            let mut norm = 0.0f64;
            for k in 0..win_len {
                let weight = (period as f64 - k as f64).powf(power);
                weights_flat[idx * stride + k] = weight as f32;
                norm += weight;
            }

            if !norm.is_finite() || norm == 0.0 {
                return Err(CudaVpwmaError::InvalidInput(format!(
                    "invalid normalization for period {} power {}",
                    period, power
                )));
            }

            inv_norms.push((1.0 / norm) as f32);
        }

        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_win_lengths = DeviceBuffer::from_slice(&win_lengths)?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat)?;
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms)?;

        let elems = n_combos
            .checked_mul(len)
            .ok_or_else(|| CudaVpwmaError::InvalidInput("element count overflow".into()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_win_lengths,
            &d_weights,
            &d_inv_norms,
            len,
            stride,
            first_valid,
            n_combos,
            &mut d_out,
        )?;
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: len,
            ctx: self.ctx.clone(),
            device_id: self.device_id,
        })
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VpwmaParams,
    ) -> Result<DeviceArrayF32, CudaVpwmaError> {
        let (first_valids, period, weights, inv_norm) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;
        let d_weights = DeviceBuffer::from_slice(&weights)?;

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaVpwmaError::InvalidInput("element count overflow".into()))?;
        let mut d_out_tm = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_first_valids,
            period,
            &d_weights,
            inv_norm,
            cols,
            rows,
            &mut d_out_tm,
        )?;
        // Synchronize producing stream so Python CAI can omit stream safely
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
            ctx: self.ctx.clone(),
            device_id: self.device_id,
        })
    }

    pub fn vpwma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &VpwmaBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<VpwmaParams>), CudaVpwmaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        Ok((dev, combos))
    }

    pub fn vpwma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &VpwmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<VpwmaParams>), CudaVpwmaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaVpwmaError::InvalidInput(format!(
                "output slice length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }

        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        dev.buf.copy_to(out)?;
        Ok((combos.len(), len, combos))
    }

    pub fn vpwma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        inv_norm: f32,
        num_series: i32,
        series_len: i32,
        d_weights: &DeviceBuffer<f32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVpwmaError> {
        if period < 2 || num_series <= 0 || series_len <= 0 {
            return Err(CudaVpwmaError::InvalidInput(
                "invalid period/series configuration".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            period as usize,
            d_weights,
            inv_norm,
            num_series as usize,
            series_len as usize,
            d_out_tm,
        )
    }

    pub fn vpwma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VpwmaParams,
    ) -> Result<DeviceArrayF32, CudaVpwmaError> {
        self.run_many_series_kernel(data_tm_f32, cols, rows, params)
    }

    pub fn vpwma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VpwmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaVpwmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaVpwmaError::InvalidInput(format!(
                "output slice length mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }

        let arr = self.run_many_series_kernel(data_tm_f32, cols, rows, params)?;
        arr.buf.copy_to(out_tm)?;
        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::vpwma::{VpwmaBatchRange, VpwmaParams};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct VpwmaBatchDevState {
        cuda: CudaVpwma,
        d_prices: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_win_lengths: DeviceBuffer<i32>,
        d_weights: DeviceBuffer<f32>,
        d_inv_norms: DeviceBuffer<f32>,
        len: usize,
        stride: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for VpwmaBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_periods,
                    &self.d_win_lengths,
                    &self.d_weights,
                    &self.d_inv_norms,
                    self.len,
                    self.stride,
                    self.first_valid,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("vpwma batch kernel");
            self.cuda.stream.synchronize().expect("vpwma sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaVpwma::new(0).expect("cuda vpwma");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = VpwmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            power: (2.0, 2.0, 0.0),
        };
        let (combos, first_valid, len) =
            CudaVpwma::prepare_batch_inputs(&price, &sweep).expect("vpwma prepare batch inputs");
        let n_combos = combos.len();

        let stride = combos
            .iter()
            .map(|c| c.period.unwrap() - 1)
            .max()
            .unwrap_or(1);

        let mut periods = Vec::with_capacity(n_combos);
        let mut win_lengths = Vec::with_capacity(n_combos);
        let mut inv_norms = Vec::with_capacity(n_combos);
        let mut weights_flat = vec![0f32; n_combos * stride];

        for (idx, combo) in combos.iter().enumerate() {
            let period = combo.period.unwrap();
            let power = combo.power.unwrap();
            let win_len = period - 1;

            periods.push(period as i32);
            win_lengths.push(win_len as i32);

            let mut norm = 0.0f64;
            for k in 0..win_len {
                let weight = (period as f64 - k as f64).powf(power);
                weights_flat[idx * stride + k] = weight as f32;
                norm += weight;
            }
            if !norm.is_finite() || norm == 0.0 {
                panic!("vpwma invalid norm for period={} power={}", period, power);
            }
            inv_norms.push((1.0 / norm) as f32);
        }

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
        let d_win_lengths = DeviceBuffer::from_slice(&win_lengths).expect("d_win_lengths");
        let d_weights = DeviceBuffer::from_slice(&weights_flat).expect("d_weights");
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms).expect("d_inv_norms");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * len) }.expect("d_out");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(VpwmaBatchDevState {
            cuda,
            d_prices,
            d_periods,
            d_win_lengths,
            d_weights,
            d_inv_norms,
            len,
            stride,
            first_valid,
            n_combos,
            d_out,
        })
    }

    struct VpwmaManyDevState {
        cuda: CudaVpwma,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        d_weights: DeviceBuffer<f32>,
        period: usize,
        inv_norm: f32,
        cols: usize,
        rows: usize,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for VpwmaManyDevState {
        fn launch(&mut self) {
            self.cuda
                .vpwma_multi_series_one_param_device(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    self.period as i32,
                    self.inv_norm,
                    self.cols as i32,
                    self.rows as i32,
                    &self.d_weights,
                    &mut self.d_out_tm,
                )
                .expect("vpwma many-series kernel");
            self.cuda.stream.synchronize().expect("vpwma sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaVpwma::new(0).expect("cuda vpwma");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = VpwmaParams {
            period: Some(64),
            power: Some(2.0),
        };
        let (first_valids, period, weights, inv_norm) =
            CudaVpwma::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("vpwma prepare many-series inputs");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_weights = DeviceBuffer::from_slice(&weights).expect("d_weights");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");

        cuda.stream.synchronize().expect("sync after prep");
        Box::new(VpwmaManyDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            d_weights,
            period,
            inv_norm,
            cols,
            rows,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "vpwma",
                "one_series_many_params",
                "vpwma_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "vpwma",
                "many_series_one_param",
                "vpwma_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
