//! CUDA wrapper for EHMA (Ehlers Hann Moving Average) kernels.
//!
//! Upgraded to ALMA "gold standard" parity:
//! - Policy selection with introspection (batch and many-series)
//! - VRAM checks and BENCH_DEBUG kernel selection logs
//! - Plain batch (on-device weights) and tiled batch 2x (precomputed weights)
//! - Many-series 1D and 2D tiled (time-major)

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::ehma::{expand_grid, EhmaBatchRange, EhmaParams};
use cust::context::Context;
use cust::context::{CacheConfig, SharedMemoryConfig};
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::AsyncCopyDestination;
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::ffi::c_void;
use std::fmt;
use thiserror::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Error)]
pub enum CudaEhmaError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Out of memory on device: required={required} bytes (including headroom={headroom}), free={free} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Launch configuration too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchThreadsPerOutput {
    One,
    Two,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain {
        block_x: u32,
    },
    Tiled {
        tile: u32,
        per_thread: BatchThreadsPerOutput,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaEhmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaEhmaPolicy {
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
    Tiled2x { tile: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

pub struct CudaEhma {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaEhmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaEhma {
    pub fn new(device_id: usize) -> Result<Self, CudaEhmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ehma_kernel.ptx"));
        // Prefer O2 and target-from-context for stability across drivers
        let jit_opts = &[
            cust::module::ModuleJitOption::DetermineTargetFromContext,
            cust::module::ModuleJitOption::OptLevel(cust::module::OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts).or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // Best-effort: reserve a portion of L2 for persisting cache if supported
        Self::reserve_l2_persisting_quota_once(device_id as u32);

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaEhmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Clone of the underlying CUDA context to help ensure VRAM lifetime when needed by callers.
    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }

    /// Create with an explicit policy.
    pub fn new_with_policy(
        device_id: usize,
        policy: CudaEhmaPolicy,
    ) -> Result<Self, CudaEhmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaEhmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaEhmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    /// Explicit synchronize for deterministic timing in benches
    pub fn synchronize(&self) -> Result<(), CudaEhmaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaEhmaError::Cuda(e))
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
                    eprintln!("[DEBUG] EHMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaEhma)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] EHMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaEhma)).debug_many_logged = true;
                }
            }
        }
    }

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
    fn bytes_for<T>(elems: usize) -> Result<usize, CudaEhmaError> {
        elems
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| CudaEhmaError::InvalidInput("byte size overflow".into()))
    }

    #[inline]
    fn will_fit_checked(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaEhmaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            let need = required_bytes.saturating_add(headroom_bytes);
            if need <= free {
                Ok(())
            } else {
                Err(CudaEhmaError::OutOfMemory { required: need, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
        }
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

    #[inline(always)]
    fn align_up_16(bytes: usize) -> usize {
        (bytes + 15) & !15
    }

    #[inline]
    fn set_kernel_launch_prefs(&self, func: &mut Function, dyn_smem_bytes: usize) {
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        let _ = func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize);
        unsafe {
            let _ = cu::cuFuncSetAttribute(
                func.to_raw(),
                cu::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                dyn_smem_bytes as i32,
            );
            let _ = cu::cuFuncSetAttribute(
                func.to_raw(),
                cu::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                100,
            );
        }
    }

    fn reserve_l2_persisting_quota_once(device_id: u32) {
        if std::env::var("EHMA_L2_HINT").ok().as_deref() == Some("0") {
            return;
        }
        unsafe {
            if let Ok(dev) = cust::device::Device::get_device(device_id) {
                let mut max_persist: i32 = 0;
                let _ = cu::cuDeviceGetAttribute(
                    &mut max_persist as *mut _,
                    cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
                    dev.as_raw(),
                );
                if max_persist > 0 {
                    let want = ((max_persist as usize) * 3) / 4;
                    let _ = cu::cuCtxSetLimit(
                        cu::CUlimit_enum::CU_LIMIT_PERSISTING_L2_CACHE_SIZE,
                        want,
                    );
                }
            }
        }
    }

    fn hint_stream_access_policy_window(&self, base_dev_ptr: u64, num_bytes: usize) {
        if std::env::var("EHMA_L2_HINT").ok().as_deref() == Some("0") {
            return;
        }
        unsafe {
            let mut max_window: i32 = num_bytes as i32;
            if let Ok(dev) = cust::device::Device::get_device(self.device_id) {
                let _ = cu::cuDeviceGetAttribute(
                    &mut max_window as *mut _,
                    cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
                    dev.as_raw(),
                );
            }
            let window = (num_bytes as i32).min(max_window.max(0)) as usize;

            let mut val: cu::CUstreamAttrValue = std::mem::zeroed();
            let apw = cu::CUaccessPolicyWindow_v1 {
                base_ptr: base_dev_ptr as usize as *mut std::ffi::c_void,
                num_bytes: window,
                hitRatio: 0.60f32,
                hitProp: cu::CUaccessProperty_enum::CU_ACCESS_PROPERTY_PERSISTING,
                missProp: cu::CUaccessProperty_enum::CU_ACCESS_PROPERTY_NORMAL,
            };
            val.accessPolicyWindow = apw;
            let _ = cu::cuStreamSetAttribute(
                self.stream.as_inner(),
                cu::CUstreamAttrID_enum::CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &val as *const _ as *mut _,
            );
        }
    }

    #[inline]
    fn batch_tiled_symbol(&self, tile: u32) -> &'static str {
        match tile {
            128 => {
                if self
                    .module
                    .get_function("ehma_batch_tiled_f32_2x_tile128_async")
                    .is_ok()
                {
                    "ehma_batch_tiled_f32_2x_tile128_async"
                } else {
                    "ehma_batch_tiled_f32_2x_tile128"
                }
            }
            512 => {
                if self
                    .module
                    .get_function("ehma_batch_tiled_f32_2x_tile512_async")
                    .is_ok()
                {
                    "ehma_batch_tiled_f32_2x_tile512_async"
                } else {
                    "ehma_batch_tiled_f32_2x_tile512"
                }
            }
            _ => {
                if self
                    .module
                    .get_function("ehma_batch_tiled_f32_2x_tile256_async")
                    .is_ok()
                {
                    "ehma_batch_tiled_f32_2x_tile256_async"
                } else {
                    "ehma_batch_tiled_f32_2x_tile256"
                }
            }
        }
    }

    #[inline]
    fn ms2d_symbol(&self, tx: u32, ty: u32) -> Result<&'static str, CudaEhmaError> {
        let (a, b) = match (tx, ty) {
            (128, 4) => (
                "ehma_ms1p_tiled_f32_tx128_ty4_async",
                "ehma_ms1p_tiled_f32_tx128_ty4",
            ),
            (128, 2) => (
                "ehma_ms1p_tiled_f32_tx128_ty2_async",
                "ehma_ms1p_tiled_f32_tx128_ty2",
            ),
            _ => {
                return Err(CudaEhmaError::InvalidInput(format!(
                    "unsupported 2D tile tx={}, ty={}",
                    tx, ty
                )))
            }
        };
        Ok(if self.module.get_function(a).is_ok() {
            a
        } else {
            b
        })
    }

    #[inline]
    fn pick_tiled_block(&self, series_len: usize) -> u32 {
        if let Ok(v) = std::env::var("EHMA_TILE") {
            if let Ok(tile) = v.parse::<u32>() {
                let name = match tile {
                    128 => Some("ehma_batch_tiled_f32_2x_tile128"),
                    256 => Some("ehma_batch_tiled_f32_2x_tile256"),
                    512 => Some("ehma_batch_tiled_f32_2x_tile512"),
                    _ => None,
                };
                if let Some(fname) = name {
                    if self.module.get_function(fname).is_ok() {
                        return tile;
                    }
                }
            }
        }
        // Default: 256 unless very short series
        if series_len < 8192 {
            if self
                .module
                .get_function("ehma_batch_tiled_f32_2x_tile128")
                .is_ok()
            {
                return 128;
            }
        }
        256
    }

    #[inline]
    fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX_Y: usize = 65_535;
        (0..n)
            .step_by(MAX_Y)
            .map(move |start| (start, (n - start).min(MAX_Y)))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &EhmaBatchRange,
    ) -> Result<(Vec<EhmaParams>, usize, usize, usize), CudaEhmaError> {
        if data_f32.is_empty() {
            return Err(CudaEhmaError::InvalidInput("empty data".into()));
        }

        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaEhmaError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaEhmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaEhmaError::InvalidInput("period must be > 0".into()));
            }
            if period > len {
                return Err(CudaEhmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaEhmaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
            max_period = max_period.max(period);
        }

        Ok((combos, first_valid, len, max_period))
    }

    fn compute_normalized_weights(period: usize) -> Vec<f32> {
        // Use exact Hann normalization: sum_{i=1..P} (1 - cos(2*pi*i/(P+1))) = P+1
        // Generate weights via 2*sin^2(pi*x) with f64 trig, then normalize by 1/(P+1) exactly.
        let mut weights = vec![0.0f32; period];
        if period == 0 {
            return weights;
        }
        let inv = 1.0f32 / (period as f32 + 1.0f32);
        for idx in 0..period {
            let i = (period - idx) as f64; // map oldest..newest to i=P..1
            let x = i / ((period as f64) + 1.0);
            let s = (std::f64::consts::PI * x).sin();
            let wt = 2.0 * s * s; // 1 - cos = 2*sin^2
            weights[idx] = (wt as f32) * inv;
        }
        weights
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhmaParams,
    ) -> Result<(Vec<i32>, usize, Vec<f32>), CudaEhmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaEhmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaEhmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(CudaEhmaError::InvalidInput("period must be > 0".into()));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series;
                let v = data_tm_f32[idx];
                if !v.is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fv_row = fv.ok_or_else(|| {
                CudaEhmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv_row < period {
                return Err(CudaEhmaError::InvalidInput(format!(
                    "series {} lacks enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv_row
                )));
            }
            first_valids[series] = fv_row as i32;
        }

        let weights = Self::compute_normalized_weights(period);
        Ok((first_valids, period, weights))
    }

    fn launch_batch_kernel_plain(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhmaError> {
        if series_len == 0 {
            return Err(CudaEhmaError::InvalidInput("series_len is zero".into()));
        }
        if n_combos == 0 {
            return Err(CudaEhmaError::InvalidInput("no parameter combos".into()));
        }
        if max_period == 0 {
            return Err(CudaEhmaError::InvalidInput("max_period is zero".into()));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaEhmaError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }

        let mut func = self
            .module
            .get_function("ehma_batch_f32")
            .map_err(|_| CudaEhmaError::MissingKernelSymbol { name: "ehma_batch_f32" })?;

        const BLOCK_X: u32 = 256;
        let grid_x = ((series_len as u32) + BLOCK_X - 1) / BLOCK_X;
        let block: BlockSize = (BLOCK_X, 1, 1).into();
        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;
        self.set_kernel_launch_prefs(&mut func, shared_bytes as usize);

        // Launch in chunks along grid.y to respect the 65,535 limit
        for (start, len) in Self::grid_y_chunks(n_combos) {
            let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                // Offset periods/warm/out by the chunk start
                let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                let mut warms_ptr = d_warms.as_device_ptr().add(start).as_raw();
                let out_offset = d_out.as_device_ptr().add(start * series_len);
                let mut out_ptr = out_offset.as_raw();
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = len as i32;
                let mut max_period_i = max_period as i32;
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut warms_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut max_period_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, shared_bytes, args)?;
            }
        }

        // Introspection/log
        unsafe {
            (*(self as *const _ as *mut CudaEhma)).last_batch =
                Some(BatchKernelSelected::Plain { block_x: BLOCK_X });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    fn launch_many_series_kernel_1d(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhmaError> {
        if period == 0 {
            return Err(CudaEhmaError::InvalidInput("period is zero".into()));
        }
        if num_series == 0 || series_len == 0 {
            return Err(CudaEhmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaEhmaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let mut func = self
            .module
            .get_function("ehma_multi_series_one_param_f32")
            .map_err(|_| CudaEhmaError::MissingKernelSymbol { name: "ehma_multi_series_one_param_f32" })?;

        const BLOCK_X: u32 = 256;
        let grid_x = ((series_len as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x.max(1), num_series as u32, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();
        let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;
        self.set_kernel_launch_prefs(&mut func, shared_bytes as usize);

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, shared_bytes, args)?;
        }

        // Introspection/log
        unsafe {
            (*(self as *const _ as *mut CudaEhma)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x: BLOCK_X });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    fn launch_many_series_kernel_2d(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
        tx: u32,
        ty: u32,
    ) -> Result<(), CudaEhmaError> {
        let fname = self.ms2d_symbol(tx, ty)?;
        let mut func = self
            .module
            .get_function(fname)
            .map_err(|_| CudaEhmaError::MissingKernelSymbol { name: fname })?;
        let grid_x = ((series_len as u32) + tx - 1) / tx;
        let grid_y = ((num_series as u32) + ty - 1) / ty;
        let grid: GridSize = (grid_x.max(1), grid_y.max(1), 1).into();
        let block: BlockSize = (tx, ty, 1).into();
        // Shared layout: align16(period*4) + (TX+period-1)*TY*4
        let period_aligned = Self::align_up_16(period * std::mem::size_of::<f32>());
        let tile_elems = (tx as usize + period - 1) * (ty as usize);
        let shared_bytes = (period_aligned + tile_elems * std::mem::size_of::<f32>()) as u32;
        self.set_kernel_launch_prefs(&mut func, shared_bytes as usize);

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut inv_norm = 1.0f32; // ignored in kernels
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut inv_norm as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, shared_bytes, args)?;
        }
        unsafe {
            (*(self as *const _ as *mut CudaEhma)).last_many =
                Some(ManySeriesKernelSelected::Tiled2D { tx, ty });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    pub fn ehma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhmaError> {
        self.launch_batch_kernel_plain(
            d_prices, d_periods, d_warms, series_len, n_combos, max_period, d_out,
        )
    }

    pub fn ehma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &EhmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaEhmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        // VRAM preflight (prices + weights_flat + out) with checked arithmetic
        let prices_bytes = Self::bytes_for::<f32>(series_len)?;
        let weights_elems = n_combos
            .checked_mul(max_period)
            .ok_or_else(|| CudaEhmaError::InvalidInput("weights elems overflow".into()))?;
        let weights_bytes = Self::bytes_for::<f32>(weights_elems)?;
        let out_elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaEhmaError::InvalidInput("output elems overflow".into()))?;
        let out_bytes = Self::bytes_for::<f32>(out_elems)?;
        let required = prices_bytes
            .checked_add(weights_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaEhmaError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit_checked(required, 64 * 1024 * 1024)?;

        // Upload prices once
        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }?;

        // Build per-combo weights (pre-normalized) and periods; warms for plain fallback
        let mut periods_i32 = vec![0i32; n_combos];
        let mut warms_i32 = vec![0i32; n_combos];
        let mut weights_flat = vec![0f32; n_combos * max_period];
        for (i, prm) in combos.iter().enumerate() {
            let p = prm.period.unwrap();
            periods_i32[i] = p as i32;
            warms_i32[i] = (first_valid + p - 1) as i32;
            let w = Self::compute_normalized_weights(p);
            let base = i * max_period;
            weights_flat[base..base + p].copy_from_slice(&w);
        }
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream) }?;
        let d_warms = unsafe { DeviceBuffer::from_slice_async(&warms_i32, &self.stream) }?;
        let d_weights = unsafe { DeviceBuffer::from_slice_async(&weights_flat, &self.stream) }?;

        // Output
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream) }?;

        // Hint L2 persistence for the prices region (best-effort)
        self.hint_stream_access_policy_window(
            d_prices.as_device_ptr().as_raw(),
            series_len * std::mem::size_of::<f32>(),
        );

        // Choose kernel per policy
        let mut use_tiled = series_len > 8192;
        let mut force_tile: Option<u32> = None;
        match self.policy.batch {
            BatchKernelPolicy::Auto => {}
            BatchKernelPolicy::Plain { .. } => use_tiled = false,
            BatchKernelPolicy::Tiled { tile, .. } => {
                use_tiled = true;
                force_tile = Some(tile);
            }
        }

        if use_tiled {
            let tile = force_tile.unwrap_or_else(|| self.pick_tiled_block(series_len));
            let fname = self.batch_tiled_symbol(tile);
            if let Ok(mut func) = self.module.get_function(fname) {
                let grid_x = ((series_len as u32) + tile - 1) / tile;
                let block_x = (tile / 2) as u32; // 2 outputs per thread
                let block: BlockSize = (block_x, 1, 1).into();
                // Chunk over combos in grid.y
                for (start, len) in Self::grid_y_chunks(n_combos) {
                    let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
                    // Exact dynamic shared memory layout
                    let period_aligned = Self::align_up_16(max_period * std::mem::size_of::<f32>());
                    let tile_elems = (tile as usize) + max_period - 1;
                    let shared_bytes =
                        (period_aligned + tile_elems * std::mem::size_of::<f32>()) as u32;
                    self.set_kernel_launch_prefs(&mut func, shared_bytes as usize);
                    unsafe {
                        let out_ptr = d_out.as_device_ptr().add(start * series_len);
                        let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                        let mut wflat_ptr = d_weights.as_device_ptr().as_raw();
                        let mut periods_ptr = d_periods.as_device_ptr().as_raw();
                        let mut inv_ptr: *const f32 = std::ptr::null();
                        let mut maxp_i = max_period as i32;
                        let mut len_i = series_len as i32;
                        let mut ncomb_i = len as i32;
                        let mut fv_i = first_valid as i32;
                        let mut out_raw = out_ptr.as_raw();
                        let args: &mut [*mut c_void] = &mut [
                            &mut prices_ptr as *mut _ as *mut c_void,
                            &mut wflat_ptr as *mut _ as *mut c_void,
                            &mut periods_ptr as *mut _ as *mut c_void,
                            &mut inv_ptr as *mut _ as *mut c_void,
                            &mut maxp_i as *mut _ as *mut c_void,
                            &mut len_i as *mut _ as *mut c_void,
                            &mut ncomb_i as *mut _ as *mut c_void,
                            &mut fv_i as *mut _ as *mut c_void,
                            &mut out_raw as *mut _ as *mut c_void,
                        ];
                        self.stream
                            .launch(&func, grid, block, shared_bytes, args)
                            .map_err(|e| CudaEhmaError::Cuda(e))?;
                    }
                }
                // Introspection
                unsafe {
                    (*(self as *const _ as *mut CudaEhma)).last_batch =
                        Some(BatchKernelSelected::Tiled2x { tile });
                }
                self.maybe_log_batch_debug();
            } else {
                // Fallback to plain
                self.launch_batch_kernel_plain(
                    &d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out,
                )?;
            }
        } else {
            self.launch_batch_kernel_plain(
                &d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out,
            )?;
        }

        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    /// Device input → VRAM output, avoids host price copies. Caller supplies first_valid.
    pub fn ehma_batch_from_device_prices(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        sweep: &EhmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaEhmaError> {
        let (combos, _fv, _len, max_period) =
            Self::prepare_batch_inputs(&vec![0f32; series_len], sweep)?;
        let n_combos = combos.len();
        let mut periods_i32 = vec![0i32; n_combos];
        let mut warms_i32 = vec![0i32; n_combos];
        let mut weights_flat = vec![0f32; n_combos * max_period];
        // inv_norms unused by tiled path
        for (i, prm) in combos.iter().enumerate() {
            let p = prm.period.unwrap();
            periods_i32[i] = p as i32;
            warms_i32[i] = (first_valid + p - 1) as i32;
            let w = Self::compute_normalized_weights(p);
            let base = i * max_period;
            weights_flat[base..base + p].copy_from_slice(&w);
        }
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream) }
            .map_err(|e| CudaEhmaError::Cuda(e))?;
        let d_warms = unsafe { DeviceBuffer::from_slice_async(&warms_i32, &self.stream) }
            .map_err(|e| CudaEhmaError::Cuda(e))?;
        let d_weights = unsafe { DeviceBuffer::from_slice_async(&weights_flat, &self.stream) }
            .map_err(|e| CudaEhmaError::Cuda(e))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n_combos * series_len, &self.stream) }
                .map_err(|e| CudaEhmaError::Cuda(e))?;
        // Hint L2 persistence for the prices region (best-effort)
        self.hint_stream_access_policy_window(
            d_prices.as_device_ptr().as_raw(),
            series_len * std::mem::size_of::<f32>(),
        );

        // Prefer tiled if available
        let tile = self.pick_tiled_block(series_len);
        let fname = self.batch_tiled_symbol(tile);
        if let Ok(mut func) = self.module.get_function(fname) {
            let grid_x = ((series_len as u32) + tile - 1) / tile;
            let block_x = (tile / 2) as u32;
            let block: BlockSize = (block_x, 1, 1).into();
            for (start, len) in Self::grid_y_chunks(n_combos) {
                let grid: GridSize = (grid_x.max(1), len as u32, 1).into();
                let period_aligned = Self::align_up_16(max_period * std::mem::size_of::<f32>());
                let tile_elems = (tile as usize) + max_period - 1;
                let shared_bytes =
                    (period_aligned + tile_elems * std::mem::size_of::<f32>()) as u32;
                self.set_kernel_launch_prefs(&mut func, shared_bytes as usize);
                unsafe {
                    let out_ptr = d_out.as_device_ptr().add(start * series_len);
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut wflat_ptr = d_weights.as_device_ptr().as_raw();
                    let mut periods_ptr = d_periods.as_device_ptr().as_raw();
                    let mut inv_ptr: *const f32 = std::ptr::null();
                    let mut maxp_i = max_period as i32;
                    let mut len_i = series_len as i32;
                    let mut ncomb_i = len as i32;
                    let mut fv_i = first_valid as i32;
                    let mut out_raw = out_ptr.as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut wflat_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut inv_ptr as *mut _ as *mut c_void,
                        &mut maxp_i as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut ncomb_i as *mut _ as *mut c_void,
                        &mut fv_i as *mut _ as *mut c_void,
                        &mut out_raw as *mut _ as *mut c_void,
                    ];
                    self.stream.launch(&func, grid, block, shared_bytes, args)?;
                }
            }
            unsafe {
                (*(self as *const _ as *mut CudaEhma)).last_batch =
                    Some(BatchKernelSelected::Tiled2x { tile });
            }
            self.maybe_log_batch_debug();
        } else {
            self.launch_batch_kernel_plain(
                d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out,
            )?;
        }
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn ehma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &EhmaBatchRange,
        out: &mut [f32],
    ) -> Result<Vec<EhmaParams>, CudaEhmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();
        if out.len() != n_combos * series_len {
            return Err(CudaEhmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                n_combos * series_len
            )));
        }

        // VRAM preflight (prices + periods + warms + out) with checked arithmetic
        let prices_bytes = Self::bytes_for::<f32>(series_len)?;
        let period_elems = n_combos;
        let warms_elems = n_combos;
        let periods_bytes = Self::bytes_for::<i32>(period_elems)?;
        let warms_bytes = Self::bytes_for::<i32>(warms_elems)?;
        let out_elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaEhmaError::InvalidInput("output elems overflow".into()))?;
        let out_bytes = Self::bytes_for::<f32>(out_elems)?;
        let required = prices_bytes
            .checked_add(periods_bytes)
            .and_then(|x| x.checked_add(warms_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaEhmaError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit_checked(required, 64 * 1024 * 1024)?;

        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut warms_i32 = Vec::with_capacity(n_combos);
        for prm in &combos {
            let period = prm.period.unwrap();
            if period > i32::MAX as usize {
                return Err(CudaEhmaError::InvalidInput(
                    "period exceeds i32::MAX".into(),
                ));
            }
            let warm = first_valid + period - 1;
            if warm > i32::MAX as usize {
                return Err(CudaEhmaError::InvalidInput(
                    "warm index exceeds i32::MAX".into(),
                ));
            }
            periods_i32.push(period as i32);
            warms_i32.push(warm as i32);
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }?;
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods_i32, &self.stream) }?;
        let d_warms = unsafe { DeviceBuffer::from_slice_async(&warms_i32, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n_combos * series_len, &self.stream) }?;

        self.launch_batch_kernel_plain(
            &d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out,
        )?;
        self.stream.synchronize()?;

        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(out.len()) }?;
        unsafe {
            d_out.async_copy_to(pinned.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        out.copy_from_slice(pinned.as_slice());

        Ok(combos)
    }

    pub fn ehma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhmaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaEhmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel_1d(
            d_prices_tm,
            d_weights,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn ehma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhmaParams,
    ) -> Result<DeviceArrayF32, CudaEhmaError> {
        let (first_valids, period, weights) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // VRAM preflight with checked arithmetic
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaEhmaError::InvalidInput("tm elems overflow".into()))?;
        let prices_bytes = Self::bytes_for::<f32>(elems)?;
        let weights_bytes = Self::bytes_for::<f32>(period)?;
        let out_bytes = Self::bytes_for::<f32>(elems)?;
        let required = prices_bytes
            .checked_add(weights_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaEhmaError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit_checked(required, 64 * 1024 * 1024)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)?;
        // Hint L2 persistence for time-major prices if supported
        self.hint_stream_access_policy_window(
            d_prices_tm.as_device_ptr().as_raw(),
            cols * rows * std::mem::size_of::<f32>(),
        );
        let d_weights = DeviceBuffer::from_slice(&weights)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }?;

        // Select kernel
        match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => {
                // Prefer 2D tiled for larger problems
                if cols >= 16
                    && rows >= 8192
                    && self
                        .module
                        .get_function("ehma_ms1p_tiled_f32_tx128_ty4")
                        .is_ok()
                {
                    self.launch_many_series_kernel_2d(
                        &d_prices_tm,
                        &d_weights,
                        period,
                        cols,
                        rows,
                        &d_first_valids,
                        &mut d_out_tm,
                        128,
                        4,
                    )?;
                } else if self
                    .module
                    .get_function("ehma_ms1p_tiled_f32_tx128_ty2")
                    .is_ok()
                    && (rows >= 8192)
                {
                    self.launch_many_series_kernel_2d(
                        &d_prices_tm,
                        &d_weights,
                        period,
                        cols,
                        rows,
                        &d_first_valids,
                        &mut d_out_tm,
                        128,
                        2,
                    )?;
                } else {
                    self.launch_many_series_kernel_1d(
                        &d_prices_tm,
                        &d_weights,
                        period,
                        cols,
                        rows,
                        &d_first_valids,
                        &mut d_out_tm,
                    )?;
                }
            }
            ManySeriesKernelPolicy::OneD { .. } => {
                self.launch_many_series_kernel_1d(
                    &d_prices_tm,
                    &d_weights,
                    period,
                    cols,
                    rows,
                    &d_first_valids,
                    &mut d_out_tm,
                )?;
            }
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                self.launch_many_series_kernel_2d(
                    &d_prices_tm,
                    &d_weights,
                    period,
                    cols,
                    rows,
                    &d_first_valids,
                    &mut d_out_tm,
                    tx,
                    ty,
                )?;
            }
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaEhmaError::Cuda(e))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn ehma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaEhmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaEhmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period, weights) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_weights = DeviceBuffer::from_slice(&weights)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }?;

        match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => {
                if cols >= 16
                    && rows >= 8192
                    && self
                        .module
                        .get_function("ehma_ms1p_tiled_f32_tx128_ty4")
                        .is_ok()
                {
                    self.launch_many_series_kernel_2d(
                        &d_prices_tm,
                        &d_weights,
                        period,
                        cols,
                        rows,
                        &d_first_valids,
                        &mut d_out_tm,
                        128,
                        4,
                    )?;
                } else if self
                    .module
                    .get_function("ehma_ms1p_tiled_f32_tx128_ty2")
                    .is_ok()
                    && (rows >= 8192)
                {
                    self.launch_many_series_kernel_2d(
                        &d_prices_tm,
                        &d_weights,
                        period,
                        cols,
                        rows,
                        &d_first_valids,
                        &mut d_out_tm,
                        128,
                        2,
                    )?;
                } else {
                    self.launch_many_series_kernel_1d(
                        &d_prices_tm,
                        &d_weights,
                        period,
                        cols,
                        rows,
                        &d_first_valids,
                        &mut d_out_tm,
                    )?;
                }
            }
            ManySeriesKernelPolicy::OneD { .. } => {
                self.launch_many_series_kernel_1d(
                    &d_prices_tm,
                    &d_weights,
                    period,
                    cols,
                    rows,
                    &d_first_valids,
                    &mut d_out_tm,
                )?;
            }
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                self.launch_many_series_kernel_2d(
                    &d_prices_tm,
                    &d_weights,
                    period,
                    cols,
                    rows,
                    &d_first_valids,
                    &mut d_out_tm,
                    tx,
                    ty,
                )?;
            }
        }
        self.stream.synchronize()?;

        // Pinned D2H for determinism and throughput
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(out_tm.len()) }?;
        unsafe {
            d_out_tm.async_copy_to(pinned.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        out_tm.copy_from_slice(pinned.as_slice());

        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        ehma_benches,
        CudaEhma,
        crate::indicators::moving_averages::ehma::EhmaBatchRange,
        crate::indicators::moving_averages::ehma::EhmaParams,
        ehma_batch_dev,
        ehma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::ehma::EhmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::ehma::EhmaParams { period: Some(64) },
        "ehma",
        "ehma"
    );
    pub use ehma_benches::bench_profiles;
}
