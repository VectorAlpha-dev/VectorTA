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
use cust::context::{CacheConfig, SharedMemoryConfig};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{
    mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer, LockedBuffer,
};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaCwmaError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("device out of memory: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("launch configuration too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buffer device {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}



/// Whether each thread computes one output or two outputs in fused fashion.
#[derive(Clone, Copy, Debug)]
pub enum BatchThreadsPerOutput {
    One,
    Two,
}

/// Policy to select the batch (one-series × many-params) kernel implementation.
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
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}



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
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaCwmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaCwma {
    pub fn new(device_id: usize) -> Result<Self, CudaCwmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cwma_kernel.ptx"));
        
        let mut jit_vec: Vec<ModuleJitOption> = vec![
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        if let Ok(v) = std::env::var("CWMA_JIT_MAXREGS") {
            if let Ok(cap) = v.parse::<u32>() {
                jit_vec.push(ModuleJitOption::MaxRegisters(cap));
            }
        }
        let module = match Module::from_ptx(ptx, &jit_vec) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
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
            _context: context,
            device_id: device_id as u32,
            policy: CudaCwmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaCwmaPolicy,
    ) -> Result<Self, CudaCwmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaCwmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaCwmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaCwmaError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    fn will_fit_checked(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaCwmaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaCwmaError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn get_function_checked(&self, name: &'static str) -> Result<Function, CudaCwmaError> {
        self.module
            .get_function(name)
            .map_err(|_| CudaCwmaError::MissingKernelSymbol { name })
    }

    #[inline]
    fn prefer_shared_and_optin_smem(&self, func: &mut Function, requested_dynamic_smem: usize) {
        // Prefer shared L1 partitioning for shared-heavy kernels
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        let _ = func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize);
        // Best-effort: request larger opt-in dynamic shared memory and shared carve-out
        unsafe {
            use cust::sys::{cuFuncSetAttribute, CUfunction_attribute_enum as Attr};
            let raw = func.to_raw();
            // Request up to requested_dynamic_smem bytes
            let _ = cuFuncSetAttribute(
                raw,
                Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                requested_dynamic_smem as i32,
            );
            // Hint: prefer allocating on-chip storage to shared (0..=100)
            let _ = cuFuncSetAttribute(
                raw,
                Attr::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                100,
            );
        }
    }

    #[inline]
    fn grid_y_chunks(n_combos: usize) -> impl Iterator<Item = (usize, usize)> {
        const MAX_GRID_Y: usize = 65_535;
        (0..n_combos).step_by(MAX_GRID_Y).map(move |start| {
            let len = (n_combos - start).min(MAX_GRID_Y);
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
                    eprintln!("[DEBUG] CWMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaCwma)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] CWMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaCwma)).debug_many_logged = true;
                }
            }
        }
    }

    fn expand_periods(range: &CwmaBatchRange) -> Vec<CwmaParams> {
        let (start, end, step) = range.period;
        let periods = if step == 0 || start == end {
            vec![start]
        } else {
            let (lo, hi) = if start <= end { (start, end) } else { (end, start) };
            (lo..=hi).step_by(step).collect::<Vec<_>>()
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
        let sz_f32 = std::mem::size_of::<f32>();
        let prices_bytes = series_len
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaCwmaError::InvalidInput("size overflow: series_len".into()))?;
        let weights_bytes = n_combos
            .checked_mul(weights_stride)
            .and_then(|e| e.checked_mul(sz_f32))
            .ok_or_else(|| CudaCwmaError::InvalidInput("size overflow: weights".into()))?;
        let periods_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaCwmaError::InvalidInput("size overflow: periods".into()))?;
        let inv_norm_bytes = n_combos
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaCwmaError::InvalidInput("size overflow: inv_norms".into()))?;
        let out_bytes = n_combos
            .checked_mul(series_len)
            .and_then(|e| e.checked_mul(sz_f32))
            .ok_or_else(|| CudaCwmaError::InvalidInput("size overflow: out".into()))?;
        let required = prices_bytes
            .checked_add(weights_bytes)
            .and_then(|x| x.checked_add(periods_bytes))
            .and_then(|x| x.checked_add(inv_norm_bytes))
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or_else(|| CudaCwmaError::InvalidInput("size overflow: required".into()))?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit_checked(required, headroom)?;
        // Common device buffers: page-locked host -> async copy to device
        let h_prices = LockedBuffer::from_slice(data_f32)?;
        let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(series_len) }?;
        unsafe {
            d_prices.async_copy_from(&*h_prices, &self.stream)?;
        }
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }?;

        // Prepare periods on host (shared across paths)
        let mut periods_i32 = vec![0i32; n_combos];
        for (idx, prm) in combos.iter().enumerate() {
            periods_i32[idx] = prm.period.unwrap() as i32;
        }
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;

        // Optional: on-device weight precompute path (env CWMA_BATCH_ONDEV=1 or small sweeps)
        let has_precompute = self
            .module
            .get_function("cwma_precompute_weights_f32")
            .is_ok();
        let env_force_ondev = matches!(std::env::var("CWMA_BATCH_ONDEV"), Ok(ref v) if v == "1" || v.eq_ignore_ascii_case("true"));
        let prefer_ondev = env_force_ondev || (n_combos <= 16);

        if has_precompute && prefer_ondev {
            // Allocate device weights/inv_norms and build them on-device
            let mut d_weights: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(n_combos * weights_stride) }?;
            let mut d_inv_norms: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized(n_combos) }?;

            // Launch precompute: one block per combo
            let func = self.get_function_checked("cwma_precompute_weights_oldest_first_f32")?;
            let grid: GridSize = (n_combos as u32, 1, 1).into();
            let block: BlockSize = (128, 1, 1).into();
            unsafe {
                let mut periods_ptr = d_periods.as_device_ptr().as_raw();
                let mut n_combos_i = n_combos as i32;
                let mut max_period_i = max_period as i32;
                let mut weights_ptr = d_weights.as_device_ptr().as_raw();
                let mut inv_ptr = d_inv_norms.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut max_period_i as *mut _ as *mut c_void,
                    &mut weights_ptr as *mut _ as *mut c_void,
                    &mut inv_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }

            // Now launch batch kernel using on-device precomputed weights
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
        } else {
            // Host precompute (default): build oldest-first weights and pre-scale by inv-norm
            let mut weights_flat = vec![0f32; n_combos * weights_stride];
            let mut inv_norms = vec![0f32; n_combos];
            for (idx, prm) in combos.iter().enumerate() {
                let period = prm.period.unwrap();
                let weight_len = period - 1;
                let mut norm = 0.0f32;
                for k in 0..weight_len {
                    let weight = ((k + 2) as f32).powi(3);
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
                inv_norms[idx] = 1.0; // weights already scaled
            }
            let d_weights = DeviceBuffer::from_slice(&weights_flat)?;
            let d_inv_norms = DeviceBuffer::from_slice(&inv_norms)?;

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
        }

        self.stream.synchronize()?;

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
            BatchKernelPolicy::Plain { block_x } => {
                use_tiled = false;
                tile_x = block_x;
            }
            BatchKernelPolicy::Tiled { tile, per_thread } => {
                use_tiled = true;
                tile_x = tile;
                use_two = matches!(per_thread, BatchThreadsPerOutput::Two);
            }
        }

        // Helper: compute dynamic shared memory size for tiled batch kernels:
        // bytes = align16(wlen*4) + (tile_x + wlen)*4, with wlen <= max_period-1 and using worst-case wlen.
        let align16 = |x: usize| (x + 15) & !15usize;
        if use_tiled {
            // Heuristic: prefer 2x-per-thread for larger windows when policy is Auto
            let wlen = max_period.saturating_sub(1);
            if matches!(self.policy.batch, BatchKernelPolicy::Auto) {
                use_two = wlen >= 32;
            }
            // Choose function name. If 2x per-thread, prefer async variant when present
            let func_name = if use_two {
                match tile_x {
                    128 => {
                        if self
                            .module
                            .get_function("cwma_batch_tiled_async_f32_2x_tile128")
                            .is_ok()
                        {
                            "cwma_batch_tiled_async_f32_2x_tile128"
                        } else {
                            "cwma_batch_tiled_f32_2x_tile128"
                        }
                    }
                    _ => {
                        if self
                            .module
                            .get_function("cwma_batch_tiled_async_f32_2x_tile256")
                            .is_ok()
                        {
                            "cwma_batch_tiled_async_f32_2x_tile256"
                        } else {
                            "cwma_batch_tiled_f32_2x_tile256"
                        }
                    }
                }
            } else {
                match tile_x {
                    128 => "cwma_batch_tiled_f32_tile128",
                    _ => "cwma_batch_tiled_f32_tile256",
                }
            };
            let mut func = self.get_function_checked(func_name)?;

            // Introspection
            unsafe {
                let this = self as *const _ as *mut CudaCwma;
                (*this).last_batch = Some(if use_two {
                    BatchKernelSelected::Tiled2x { tile: tile_x }
                } else {
                    BatchKernelSelected::Tiled1x { tile: tile_x }
                });
            }
            self.maybe_log_batch_debug();

            // Shared: weights (<= wlen=max_period-1) aligned to 16B + tile (tile_x + wlen)
            let wlen = max_period.saturating_sub(1);
            let tile_stages: usize = if func_name.contains("_async_") { 2 } else { 1 };
            let shared_for_tile = |tile: u32| -> u32 {
                (align16(wlen * std::mem::size_of::<f32>())
                    + tile_stages * (tile as usize + wlen) * std::mem::size_of::<f32>())
                    as u32
            };
            let mut tile_x_used = tile_x;
            let mut shared_bytes = shared_for_tile(tile_x_used);
            // Downshift tile if shared memory exceeds device capacity
            let dev = Device::get_device(self.device_id).ok();
            let max_smem_default: usize = dev
                .and_then(|d| {
                    d.get_attribute(cust::device::DeviceAttribute::MaxSharedMemoryPerBlock)
                        .ok()
                })
                .unwrap_or(48 * 1024) as usize;
            let max_smem_optin: usize =
                dev.and_then(|d| {
                    d.get_attribute(cust::device::DeviceAttribute::MaxSharedMemoryPerBlock)
                        .ok()
                })
                .unwrap_or(max_smem_default as i32) as usize;
            let avail = max_smem_optin.max(max_smem_default);
            while (shared_bytes as usize) > avail && tile_x_used > 64 {
                tile_x_used /= 2;
                shared_bytes = shared_for_tile(tile_x_used);
            }
            // Hint kernel for larger dynamic shared memory when available
            self.prefer_shared_and_optin_smem(&mut func, shared_bytes as usize);
            let block_x = if use_two {
                (tile_x_used / 2).max(1)
            } else {
                tile_x_used
            };
            let grid_x = ((series_len as u32) + tile_x_used - 1) / tile_x_used;
            let block: BlockSize = (block_x, 1, 1).into();

            for (start, len) in Self::grid_y_chunks(n_combos) {
                let grid: GridSize = (grid_x, len as u32, 1).into();
                // Validate grid X does not exceed device limits (best-effort)
                let max_gx = Device::get_device(self.device_id)
                    .and_then(|d| d.get_attribute(DeviceAttribute::MaxGridDimX))?
                    as u32;
                if grid_x > max_gx {
                    return Err(CudaCwmaError::LaunchConfigTooLarge {
                        gx: grid_x,
                        gy: len as u32,
                        gz: 1,
                        bx: block_x,
                        by: 1,
                        bz: 1,
                    });
                }
                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    // Offset parameter arrays to this Y-slice
                    let mut weights_ptr =
                        unsafe { d_weights.as_device_ptr().add(start * max_period).as_raw() };
                    let mut periods_ptr = unsafe { d_periods.as_device_ptr().add(start).as_raw() };
                    let mut inv_ptr = unsafe { d_inv_norms.as_device_ptr().add(start).as_raw() };
                    let mut max_period_i = max_period as i32;
                    let mut series_len_i = series_len as i32;
                    let mut n_combos_i = len as i32;
                    let mut first_valid_i = first_valid as i32;
                    // Offset output pointer by start * series_len elements
                    let mut out_ptr =
                        unsafe { d_out.as_device_ptr().add(start * series_len).as_raw() };
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
                    self.stream.launch(&func, grid, block, shared_bytes, args)?;
                }
            }
            // Synchronize producing stream before returning a VRAM handle
            self.synchronize()?;
        } else {
            // Plain kernel path (use occupancy suggestion when Auto)
            let shared_bytes = ((max_period.saturating_sub(1)) * std::mem::size_of::<f32>()) as u32;
            let mut func = self.get_function_checked("cwma_batch_f32")?;
            let block_x: u32 = match self.policy.batch {
                BatchKernelPolicy::Plain { block_x } => block_x,
                _ => func
                    .suggested_launch_configuration(shared_bytes as usize, BlockSize::xyz(0, 0, 0))
                    .map(|(_, bx)| bx)
                    .unwrap_or(256)
                    .max(64)
                    .min(1024),
            };
            let grid_x = ((series_len as u32) + block_x - 1) / block_x;
            let block: BlockSize = (block_x, 1, 1).into();

            for (start, len) in Self::grid_y_chunks(n_combos) {
                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut weights_ptr =
                        d_weights.as_device_ptr().add(start * max_period).as_raw();
                    let mut periods_ptr = d_periods.as_device_ptr().add(start).as_raw();
                    let mut inv_ptr = d_inv_norms.as_device_ptr().add(start).as_raw();
                    let mut max_period_i = max_period as i32;
                    let mut series_len_i = series_len as i32;
                    let mut n_combos_i = len as i32;
                    let mut first_valid_i = first_valid as i32;
                    let mut out_ptr = d_out.as_device_ptr().add(start * series_len).as_raw();
                    let grid: GridSize = (grid_x, len as u32, 1).into();
                    let max_gx = Device::get_device(self.device_id)
                        .and_then(|d| d.get_attribute(DeviceAttribute::MaxGridDimX))?
                        as u32;
                    if grid_x > max_gx {
                        return Err(CudaCwmaError::LaunchConfigTooLarge {
                            gx: grid_x,
                            gy: len as u32,
                            gz: 1,
                            bx: block_x,
                            by: 1,
                            bz: 1,
                        });
                    }
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
                    self.stream.launch(&func, grid, block, shared_bytes, args)?;
                }
            }
            unsafe {
                let this = self as *const _ as *mut CudaCwma;
                (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
            }
            self.maybe_log_batch_debug();
            // Synchronize producing stream before returning a VRAM handle
            self.synchronize()?;
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
        if series_len == 0 {
            return Err(CudaCwmaError::InvalidInput("series_len is zero".into()));
        }
        let combos = Self::expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaCwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let n_combos = combos.len();
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
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
            for k in 0..wlen {
                let w = ((p - k) as f32).powi(3);
                weights_flat[idx * max_period + k] = w;
                norm += w;
            }
            let inv = 1.0 / norm.max(1e-20);
            for k in 0..wlen {
                weights_flat[idx * max_period + k] *= inv;
            }
            periods_i32[idx] = p as i32;
        }

        let d_weights = DeviceBuffer::from_slice(&weights_flat)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }?;
        self.launch_batch_kernel(
            d_prices,
            &d_weights,
            &d_periods,
            &d_inv_norms,
            series_len,
            n_combos,
            first_valid,
            max_period,
            &mut d_out,
        )?;
        self.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
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
        // Async device->pinned copy, then slice copy
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(out.len()) }?;
        unsafe {
            arr.buf.async_copy_to(&mut pinned, &self.stream)?;
        }
        self.stream.synchronize()?;
        out.copy_from_slice(pinned.as_slice());
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
            let w = ((k + 2) as f32).powi(3);
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
        Self::will_fit_checked(required, headroom)?;

        // Pinned + async HtoD for prices
        let h_prices = LockedBuffer::from_slice(data_tm_f32)?;
        let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }?;
        unsafe {
            d_prices.async_copy_from(&*h_prices, &self.stream)?;
        }
        let d_first_valids = DeviceBuffer::from_slice(first_valids)?;
        let d_weights = DeviceBuffer::from_slice(weights)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }?;

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

        self.stream.synchronize()?;

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
        let mut tx = 128u32;
        let mut ty = 4u32;
        match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => {}
            ManySeriesKernelPolicy::OneD { block_x } => {
                use_tiled2d = false;
                tx = block_x;
            }
            ManySeriesKernelPolicy::Tiled2D { tx: txx, ty: tyy } => {
                use_tiled2d = true;
                tx = txx;
                ty = tyy;
            }
        }

        let mut func = if use_tiled2d {
            // Pick among a small set of tiled 2D kernels
            let name = match (tx, ty) {
                (128, 4) => "cwma_ms1p_tiled_f32_tx128_ty4",
                (128, 2) => "cwma_ms1p_tiled_f32_tx128_ty2",
                _ => "cwma_ms1p_tiled_f32_tx128_ty4",
            };
            let f = self.get_function_checked(name)?;
            unsafe {
                let this = self as *const _ as *mut CudaCwma;
                (*this).last_many = Some(ManySeriesKernelSelected::Tiled2D { tx, ty });
            }
            self.maybe_log_many_debug();
            f
        } else {
            let f = self.get_function_checked("cwma_multi_series_one_param_time_major_f32")?;
            let block1d_x: u32 = match self.policy.many_series {
                ManySeriesKernelPolicy::OneD { block_x } => block_x,
                _ => 128,
            };
            unsafe {
                let this = self as *const _ as *mut CudaCwma;
                (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x: block1d_x });
            }
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
                // Account for optional 1-column padding when TY divides 32 (CWMA_PAD_2D=1)
                let ty_pad = if (32 % (ty as usize)) == 0 {
                    (ty + 1) as usize
                } else {
                    ty as usize
                };
                let shared_bytes = (align16(wlen * std::mem::size_of::<f32>())
                    + total * ty_pad * std::mem::size_of::<f32>())
                    as u32;
                // Prefer shared and request opt-in dynamic smem
                self.prefer_shared_and_optin_smem(&mut func, shared_bytes as usize);
                // If dynamic shared memory exceeds capacity, fall back to 1D kernel
                let dev = Device::get_device(self.device_id).ok();
                let max_smem_default: usize = dev
                    .and_then(|d| {
                        d.get_attribute(cust::device::DeviceAttribute::MaxSharedMemoryPerBlock)
                            .ok()
                    })
                    .unwrap_or(48 * 1024) as usize;
                let max_smem_optin: usize =
                    dev.and_then(|d| {
                        d.get_attribute(cust::device::DeviceAttribute::MaxSharedMemoryPerBlock)
                            .ok()
                    })
                    .unwrap_or(max_smem_default as i32) as usize;
                let avail = max_smem_optin.max(max_smem_default);
                if (shared_bytes as usize) > avail {
                    // Try smaller TY variant if we started with TY=4
                    if ty == 4 {
                        if let Ok(mut func2) =
                            self.module.get_function("cwma_ms1p_tiled_f32_tx128_ty2")
                        {
                            let ty2: u32 = 2;
                            let ty2_pad = if (32 % (ty2 as usize)) == 0 {
                                (ty2 + 1) as usize
                            } else {
                                ty2 as usize
                            };
                            let shared_bytes2 = (align16(wlen * std::mem::size_of::<f32>())
                                + total * ty2_pad * std::mem::size_of::<f32>())
                                as u32;
                            if (shared_bytes2 as usize) <= avail {
                                self.prefer_shared_and_optin_smem(
                                    &mut func2,
                                    shared_bytes2 as usize,
                                );
                                let grid_x = ((rows as u32) + tx - 1) / tx;
                                let grid_y = ((cols as u32) + ty2 - 1) / ty2;
                                let grid: GridSize = (grid_x, grid_y, 1).into();
                                let block: BlockSize = (tx, ty2, 1).into();
                                self.stream.launch(&func2, grid, block, shared_bytes2, args)?;
                                unsafe {
                                    let this = self as *const _ as *mut CudaCwma;
                                    (*this).last_many =
                                        Some(ManySeriesKernelSelected::Tiled2D { tx, ty: ty2 });
                                }
                                self.maybe_log_many_debug();
                                return Ok(());
                            }
                        }
                    }
                    // Fallback: 1D mapping
                    let func1d =
                        self.get_function_checked("cwma_multi_series_one_param_time_major_f32")?;
                    let block_x: u32 = match self.policy.many_series {
                        ManySeriesKernelPolicy::OneD { block_x } => block_x,
                        _ => 128,
                    };
                    let grid_x = ((rows as u32) + block_x - 1) / block_x;
                    let grid: GridSize = (grid_x, cols as u32, 1).into();
                    let block: BlockSize = (block_x, 1, 1).into();
                    let shared_1d = (wlen * std::mem::size_of::<f32>()) as u32;
                    self.stream.launch(&func1d, grid, block, shared_1d, args)?;
                    unsafe {
                        let this = self as *const _ as *mut CudaCwma;
                        (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
                    }
                    self.maybe_log_many_debug();
                    return Ok(());
                }
                let grid_x = ((rows as u32) + tx - 1) / tx;
                let grid_y = ((cols as u32) + ty - 1) / ty;
                let grid: GridSize = (grid_x, grid_y, 1).into();
                let block: BlockSize = (tx, ty, 1).into();
                self.stream.launch(&func, grid, block, shared_bytes, args)?;
            } else {
                let wlen = period.saturating_sub(1);
                let shared_bytes = (wlen * std::mem::size_of::<f32>()) as u32;
                let block_x: u32 = match self.policy.many_series {
                    ManySeriesKernelPolicy::OneD { block_x } => block_x,
                    _ => func
                        .suggested_launch_configuration(
                            shared_bytes as usize,
                            BlockSize::xyz(0, 0, 0),
                        )
                        .map(|(_, bx)| bx)
                        .unwrap_or(128)
                        .max(64)
                        .min(1024),
                };
                let grid_x = ((rows as u32) + block_x - 1) / block_x;
                let grid: GridSize = (grid_x, cols as u32, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                let shared_bytes = shared_bytes;
                self.stream.launch(&func, grid, block, shared_bytes, args)?;
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
        // Async device->pinned copy, then slice copy
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(out_tm.len()) }?;
        unsafe {
            arr.buf.async_copy_to(&mut pinned, &self.stream)?;
        }
        self.stream.synchronize()?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::moving_averages::cwma::CwmaParams;

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

    struct CwmaBatchDevState {
        cuda: CudaCwma,
        d_prices: DeviceBuffer<f32>,
        d_weights: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_inv_norms: DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for CwmaBatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_weights,
                    &self.d_periods,
                    &self.d_inv_norms,
                    self.series_len,
                    self.n_combos,
                    self.first_valid,
                    self.max_period,
                    &mut self.d_out,
                )
                .expect("cwma batch kernel");
            self.cuda.stream.synchronize().expect("cwma sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaCwma::new(0).expect("cuda cwma");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = CwmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let (combos, first_valid, series_len, max_period) =
            CudaCwma::prepare_batch_inputs(&price, &sweep).expect("cwma prepare batch inputs");
        let n_combos = combos.len();

        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();

        // Host precompute: build oldest-first weights and pre-scale by inv-norm.
        let weights_stride = max_period;
        let mut weights_flat = vec![0f32; n_combos * weights_stride];
        let mut inv_norms = vec![0f32; n_combos];
        for (idx, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap();
            let weight_len = period - 1;
            let mut norm = 0.0f32;
            for k in 0..weight_len {
                let weight = ((k + 2) as f32).powi(3);
                weights_flat[idx * weights_stride + k] = weight;
                norm += weight;
            }
            if norm == 0.0 {
                panic!("cwma: period {period} produced zero normalization");
            }
            let inv = 1.0 / norm;
            for k in 0..weight_len {
                let base = idx * weights_stride + k;
                weights_flat[base] *= inv;
            }
            inv_norms[idx] = 1.0;
        }

        let d_prices = DeviceBuffer::from_slice(&price).expect("d_prices");
        let d_weights = DeviceBuffer::from_slice(&weights_flat).expect("d_weights");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms).expect("d_inv_norms");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(CwmaBatchDevState {
            cuda,
            d_prices,
            d_weights,
            d_periods,
            d_inv_norms,
            series_len,
            n_combos,
            first_valid,
            max_period,
            d_out,
        })
    }

    struct CwmaManyDevState {
        cuda: CudaCwma,
        d_prices_tm: DeviceBuffer<f32>,
        d_weights: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        inv_norm: f32,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for CwmaManyDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_weights,
                    self.period,
                    self.inv_norm,
                    self.cols,
                    self.rows,
                    &self.d_first_valids,
                    &mut self.d_out_tm,
                )
                .expect("cwma many-series kernel");
            self.cuda.stream.synchronize().expect("cwma sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaCwma::new(0).expect("cuda cwma");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = CwmaParams { period: Some(64) };
        let (first_valids, period, weights, inv_norm) =
            CudaCwma::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("cwma prepare many-series inputs");

        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_weights = DeviceBuffer::from_slice(&weights).expect("d_weights");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(CwmaManyDevState {
            cuda,
            d_prices_tm,
            d_weights,
            d_first_valids,
            cols,
            rows,
            period,
            inv_norm,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "cwma",
                "one_series_many_params",
                "cwma_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "cwma",
                "many_series_one_param",
                "cwma_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
