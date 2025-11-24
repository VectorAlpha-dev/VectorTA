//! CUDA scaffolding for the ALMA (Arnaud Legoux Moving Average) kernels.
//!
//! VRAM-first design with optional on-device weight generation, async copies,
//! and chunked launches that respect grid limits. Falls back to the legacy
//! precomputed-weights kernels if on-device variants are not present.
//!
//! Kernel symbols expected (any missing ones are auto-fallbacks to the others):
//! - "alma_batch_f32_ondev"                          // one-series × many-params, weights computed on device
//! - "alma_batch_f32"                                // one-series × many-params, precomputed weights
//! - "alma_batch_tiled_f32_tile128" / "..._tile256"  // tiled variant for long series, precomputed weights
//! - "alma_multi_series_one_param_f32"               // many-series × one-param, precomputed weights
//!
//! Notes:
//! - Async allocations/copies use the stream-ordered allocator and pinned
//!   host buffers for throughput.
//! - Grid-Y is chunked to <= 65_535 to avoid launch limit.
//! - Uses occupancy-based suggestion for non-tiled batch kernel.
//!
//! This file is a drop-in replacement for the previous wrapper.

#![cfg(feature = "cuda")]

use crate::indicators::moving_averages::alma::{AlmaBatchRange, AlmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::AsyncCopyDestination; // for async_copy_to on DeviceBuffer
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

// -------- Kernel selection policy (explicit for tests; Auto for production) --------

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

// Single-variant wrapper (fast FP32 path preserved)

#[derive(Clone, Copy, Debug)]
pub struct CudaAlmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaAlmaPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

// -------- Introspection (selected kernel) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
    Tiled1x { tile: u32 },
    Tiled2x { tile: u32 },
    OnDevice { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(thiserror::Error, Debug)]
pub enum CudaAlmaError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

/// VRAM-backed array handle returned to higher layers (Python bindings, etc.).
pub struct DeviceArrayF32 {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
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

pub struct CudaAlma {
    module: Module,
    stream: Stream,
    context: std::sync::Arc<Context>, // keep context alive and clonable
    device_id: u32,
    policy: CudaAlmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    // Debug: ensure we only print selection once per instance when BENCH_DEBUG=1
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaAlma {
    /// Create a new `CudaAlma` on `device_id` and load the ALMA PTX module.
    pub fn new(device_id: usize) -> Result<Self, CudaAlmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = std::sync::Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/alma_kernel.ptx"));
        // High optimization, target from current context
        // Use a slightly lower JIT opt level for improved stability across drivers
        // while keeping high performance in practice.
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                // Retry with progressively simpler JIT options to avoid brittle drivers.
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
            context,
            device_id: device_id as u32,
            policy: CudaAlmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Expose a simple synchronize for callers (e.g., benches) that manage
    /// their own device buffers and want deterministic timing.
    pub fn synchronize(&self) -> Result<(), CudaAlmaError> {
        self.stream.synchronize().map_err(Into::into)
    }

    pub fn context_arc(&self) -> std::sync::Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    /// Best-effort: request an L2 persisting cache window for read-mostly spans.
    /// Enabled by default; opt-out via ALMA_L2_PERSIST=0.
    fn try_enable_persisting_l2(&self, base_dev_ptr: u64, bytes: usize) {
        if std::env::var("ALMA_L2_PERSIST").ok().as_deref() == Some("0") {
            return;
        }
        unsafe {
            use cust::device::Device as CuDevice;
            use cust::sys::{
                cuCtxSetLimit, cuDeviceGetAttribute, cuStreamSetAttribute,
                CUaccessPolicyWindow_v1 as CUaccessPolicyWindow,
                CUaccessProperty_enum as AccessProp, CUdevice_attribute_enum as DevAttr,
                CUlimit_enum as CULimit, CUstreamAttrID_enum as StreamAttrId,
                CUstreamAttrValue_v1 as CUstreamAttrValue,
            };

            // Max window size supported by device
            let mut max_window_bytes_i32: i32 = 0;
            if let Ok(dev) = CuDevice::get_device(self.device_id) {
                let _ = cuDeviceGetAttribute(
                    &mut max_window_bytes_i32 as *mut _,
                    DevAttr::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
                    dev.as_raw(),
                );
            }
            let max_window_bytes = (max_window_bytes_i32.max(0) as usize).min(bytes);

            // Best-effort set-aside for L2 persistence
            let _ = cuCtxSetLimit(CULimit::CU_LIMIT_PERSISTING_L2_CACHE_SIZE, max_window_bytes);

            // Configure the stream access policy window
            let mut val: CUstreamAttrValue = std::mem::zeroed();
            val.accessPolicyWindow = CUaccessPolicyWindow {
                base_ptr: base_dev_ptr as *mut std::ffi::c_void,
                num_bytes: max_window_bytes,
                hitRatio: 0.6f32,
                hitProp: AccessProp::CU_ACCESS_PROPERTY_PERSISTING,
                missProp: AccessProp::CU_ACCESS_PROPERTY_STREAMING,
            };
            let _ = cuStreamSetAttribute(
                self.stream.as_inner(),
                StreamAttrId::CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &mut val as *mut _,
            );
        }
    }

    /// Create using an explicit policy.
    pub fn new_with_policy(
        device_id: usize,
        policy: CudaAlmaPolicy,
    ) -> Result<Self, CudaAlmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaAlmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaAlmaPolicy {
        &self.policy
    }
    /// Last selected batch kernel (if any) for debugging/inspection.
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    /// Last selected many-series kernel (if any) for debugging/inspection.
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        // Throttle: once per process unless BENCH_DEBUG_SCOPE=scenario
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] ALMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaAlma)).debug_batch_logged = true;
                }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        // Throttle: once per process unless BENCH_DEBUG_SCOPE=scenario
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] ALMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaAlma)).debug_many_logged = true;
                }
            }
        }
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

    #[inline]
    fn pick_tiled_block(&self, max_period: usize, series_len: usize, n_combos: usize) -> u32 {
        // Env override: ALMA_TILE=128|256|512
        if let Ok(v) = std::env::var("ALMA_TILE") {
            if let Ok(tile) = v.parse::<u32>() {
                let name = match tile {
                    128 => Some("alma_batch_tiled_f32_tile128"),
                    256 => Some("alma_batch_tiled_f32_tile256"),
                    512 => Some("alma_batch_tiled_f32_tile512"),
                    _ => None,
                };
                if let Some(fname) = name {
                    if self.has_function(fname) {
                        return tile;
                    }
                }
            }
        }
        // Default preference: 256 is best for typical long series and moderate combo counts.
        // Fall back to 128 only for very short series where blocks would be mostly idle.
        let prefer_256 = self.has_function("alma_batch_tiled_f32_tile256");
        if prefer_256 {
            // If series is extremely short, 128 can reduce tail waste.
            if series_len < 8192 {
                if self.has_function("alma_batch_tiled_f32_tile128") {
                    return 128;
                }
            }
            return 256;
        }
        // Next best: 128
        if self.has_function("alma_batch_tiled_f32_tile128") {
            return 128;
        }
        // Otherwise 512 if available
        if self.has_function("alma_batch_tiled_f32_tile512") {
            return 512;
        }
        // Fallback
        256
    }

    #[inline]
    fn has_function(&self, name: &str) -> bool {
        self.module.get_function(name).is_ok()
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
    fn validate_launch(&self, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32) -> Result<(), CudaAlmaError> {
        let dev = Device::get_device(self.device_id)?;
        let max_threads = dev
            .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)
            .unwrap_or(1024) as u32;
        let max_bx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimX)
            .unwrap_or(1024) as u32;
        let max_by = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimY)
            .unwrap_or(1024) as u32;
        let max_bz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxBlockDimZ)
            .unwrap_or(64) as u32;
        let max_gx = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimX)
            .unwrap_or(2_147_483_647) as u32;
        let max_gy = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimY)
            .unwrap_or(65_535) as u32;
        let max_gz = dev
            .get_attribute(cust::device::DeviceAttribute::MaxGridDimZ)
            .unwrap_or(65_535) as u32;

        let threads = bx.saturating_mul(by).saturating_mul(bz);
        if threads > max_threads || bx > max_bx || by > max_by || bz > max_bz || gx > max_gx || gy > max_gy || gz > max_gz {
            return Err(CudaAlmaError::LaunchConfigTooLarge { gx, gy, gz, bx, by, bz });
        }
        Ok(())
    }

    // ---------- Host helpers ----------

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &AlmaBatchRange,
    ) -> Result<(Vec<AlmaParams>, usize, usize, usize), CudaAlmaError> {
        if data_f32.is_empty() {
            return Err(CudaAlmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaAlmaError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaAlmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);

        if max_period == 0 || series_len - first_valid < max_period {
            return Err(CudaAlmaError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                series_len - first_valid
            )));
        }

        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            let offset = prm.offset.unwrap_or(0.85);
            let sigma = prm.sigma.unwrap_or(6.0);
            if period == 0 || sigma <= 0.0 || !(0.0..=1.0).contains(&offset) {
                return Err(CudaAlmaError::InvalidInput(format!(
                    "invalid params: period={}, offset={}, sigma={}",
                    period, offset, sigma
                )));
            }
        }

        Ok((combos, first_valid, series_len, max_period))
    }

    fn prepare_batch_inputs_device(
        series_len: usize,
        first_valid: usize,
        sweep: &AlmaBatchRange,
    ) -> Result<(Vec<AlmaParams>, usize), CudaAlmaError> {
        if series_len == 0 {
            return Err(CudaAlmaError::InvalidInput("series_len is zero".into()));
        }
        let combos = expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaAlmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_period == 0 || series_len - first_valid < max_period {
            return Err(CudaAlmaError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                series_len - first_valid
            )));
        }
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            let offset = prm.offset.unwrap_or(0.85);
            let sigma = prm.sigma.unwrap_or(6.0);
            if period == 0 || sigma <= 0.0 || !(0.0..=1.0).contains(&offset) {
                return Err(CudaAlmaError::InvalidInput(format!(
                    "invalid params: period={}, offset={}, sigma={}",
                    period, offset, sigma
                )));
            }
        }
        Ok((combos, max_period))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &AlmaParams,
    ) -> Result<(Vec<i32>, usize, f32, f32), CudaAlmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaAlmaError::InvalidInput("cols or rows is zero".into()));
        }
        if cols.checked_mul(rows).map(|n| n != data_tm_f32.len()).unwrap_or(true) {
            return Err(CudaAlmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols.checked_mul(rows).unwrap_or(usize::MAX)
            )));
        }

        let period = params.period.unwrap_or(0);
        let offset = params.offset.unwrap_or(0.85);
        let sigma = params.sigma.unwrap_or(6.0);
        if period == 0 || sigma <= 0.0 || !(0.0..=1.0).contains(&offset) {
            return Err(CudaAlmaError::InvalidInput(format!(
                "invalid params: period={}, offset={}, sigma={}",
                period, offset, sigma
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + series];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv
                .ok_or_else(|| CudaAlmaError::InvalidInput(format!("series {} all NaN", series)))?;
            if rows - fv < period {
                return Err(CudaAlmaError::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    series,
                    period,
                    rows - fv
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period, offset as f32, sigma as f32))
    }

    // ---------- Public API: one-series × many-params ----------

    /// Launch using precomputed device buffers (legacy performance path).
    pub fn alma_batch_device(
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
    ) -> Result<(), CudaAlmaError> {
        if series_len <= 0 || n_combos <= 0 || max_period <= 0 {
            return Err(CudaAlmaError::InvalidInput(
                "series_len, n_combos, and max_period must be positive".into(),
            ));
        }
        self.launch_batch_kernel_precomputed(
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

    /// Host input → VRAM output (on-device weights if available, else legacy).
    pub fn alma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &AlmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaAlmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;

        let prices_bytes = series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAlmaError::InvalidInput("series_len bytes overflow".into()))?;
        // Upper bound for legacy path; on-device path avoids weight traffic
        let weights_bytes = combos
            .len()
            .checked_mul(max_period)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaAlmaError::InvalidInput("weights bytes overflow".into()))?;
        let out_bytes = combos
            .len()
            .checked_mul(series_len)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaAlmaError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(weights_bytes)
            .and_then(|n| n.checked_add(out_bytes))
            .ok_or_else(|| CudaAlmaError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024; // 64MB safety
        if !Self::will_fit(required, headroom) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaAlmaError::OutOfMemory { required, free, headroom });
            } else {
                return Err(CudaAlmaError::InvalidInput("insufficient device memory".into()));
            }
        }

        // H2D async copy for prices
        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_f32, &self.stream)?
        };

        self.run_batch_with_prices_device(&d_prices, series_len, first_valid, &combos, max_period)
    }

    /// Device input → VRAM output, avoids host copies of prices.
    /// Caller supplies `first_valid` computed elsewhere.
    pub fn alma_batch_from_device_prices(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        sweep: &AlmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaAlmaError> {
        let (combos, max_period) =
            Self::prepare_batch_inputs_device(series_len, first_valid, sweep)?;
        self.run_batch_with_prices_device(d_prices, series_len, first_valid, &combos, max_period)
    }

    fn run_batch_with_prices_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        combos: &[AlmaParams],
        max_period: usize,
    ) -> Result<DeviceArrayF32, CudaAlmaError> {
        let n_combos = combos.len();

        // Best-effort: keep the large, read-mostly prices region warm in L2 for this stream
        self.try_enable_persisting_l2(
            d_prices.as_device_ptr().as_raw(),
            series_len
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| CudaAlmaError::InvalidInput("series_len bytes overflow".into()))?,
        );

        let out_elems = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaAlmaError::InvalidInput("n_combos*series_len overflow".into()))?;

        // Output buffer
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(out_elems, &self.stream)? };

        // Prefer precomputed host weights for large sweeps; allow opt-in to on-device
        // weights via env ALMA_BATCH_ONDEV=1.
        let has_ondev = self.module.get_function("alma_batch_f32_ondev").is_ok()
            || self.module.get_function("alma_batch_f32_onthefly").is_ok();
        let has_pre = self.has_function("alma_batch_f32")
            || self.has_function("alma_batch_tiled_f32_tile128")
            || self.has_function("alma_batch_tiled_f32_tile256");

        let env_force_ondev = matches!(
            std::env::var("ALMA_BATCH_ONDEV"),
            Ok(ref v) if v == "1" || v.eq_ignore_ascii_case("true")
        );

        // Heuristic: on-device weight generation is only beneficial for very small
        // combo counts; otherwise precomputing once on host is faster.
        let prefer_ondev = env_force_ondev || (n_combos <= 16);

        if has_pre && (!has_ondev || !prefer_ondev) {
            // Legacy: build CPU weights + inv_norms per-combo
            let mut periods_i32 = vec![0i32; n_combos];
            let mut inv_norms = vec![0f32; n_combos];
            let weights_cap = n_combos
                .checked_mul(max_period)
                .ok_or_else(|| CudaAlmaError::InvalidInput("n_combos*max_period overflow".into()))?;
            let mut weights_flat = vec![0f32; weights_cap];

            for (idx, prm) in combos.iter().enumerate() {
                let period = prm.period.unwrap() as usize;
                let offset = prm.offset.unwrap();
                let sigma = prm.sigma.unwrap();
                let (mut weights, inv_norm) = compute_weights_cpu_f32(period, offset, sigma);
                periods_i32[idx] = period as i32;
                // Pre-scale weights to eliminate per-output multiply in kernels
                if inv_norm != 0.0 {
                    for w in &mut weights {
                        *w *= inv_norm;
                    }
                }
                inv_norms[idx] = 1.0; // now a dummy in kernels
                let base = idx * max_period;
                weights_flat[base..base + period].copy_from_slice(&weights);
            }

            let d_weights = DeviceBuffer::from_slice(&weights_flat)?;
            let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
            let d_inv_norms = DeviceBuffer::from_slice(&inv_norms)?;

            self.launch_batch_kernel_precomputed(
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
        } else if has_ondev {
            self.launch_batch_kernel_ondev(
                d_prices,
                combos,
                series_len,
                n_combos,
                first_valid,
                max_period,
                &mut d_out,
            )?;
        } else {
            return Err(CudaAlmaError::NotImplemented);
        }

        // Ensure completion for VRAM handle consistency.
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: series_len })
    }

    /// Host-copy helper that writes into `out` (FP32) while returning metadata.
    pub fn alma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &AlmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<AlmaParams>), CudaAlmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected_len = combos
            .len()
            .checked_mul(series_len)
            .ok_or_else(|| CudaAlmaError::InvalidInput("combos*series_len overflow".into()))?;
        if out.len() != expected_len {
            return Err(CudaAlmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected_len
            )));
        }
        let arr = self.run_batch_with_prices_device(
            &unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream)? },
            series_len,
            first_valid,
            &combos,
            max_period,
        )?;

        // Pinned buffer for faster D2H
        let mut pinned: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(arr.len())?
        };
        unsafe {
            arr.buf
                .async_copy_to(pinned.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        out.copy_from_slice(pinned.as_slice());
        Ok((arr.rows, arr.cols, combos))
    }

    // ---------- Public API: many-series × one-parameter ----------

    /// Many-series × one-parameter (time-major). Returns a VRAM handle.
    pub fn alma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &AlmaParams,
    ) -> Result<DeviceArrayF32, CudaAlmaError> {
        let (first_valids, period, offset, sigma) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaAlmaError::InvalidInput("cols*rows overflow".into()))?;
        let prices_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAlmaError::InvalidInput("prices bytes overflow".into()))?;
        let weights_bytes = period
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAlmaError::InvalidInput("weights bytes overflow".into()))?; // legacy upper bound
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAlmaError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(weights_bytes)
            .and_then(|n| n.checked_add(out_bytes))
            .ok_or_else(|| CudaAlmaError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            if let Some((free, _total)) = Self::device_mem_info() {
                return Err(CudaAlmaError::OutOfMemory { required, free, headroom });
            } else {
                return Err(CudaAlmaError::InvalidInput("insufficient device memory".into()));
            }
        }

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream)? };
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;

        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream)? };

        // Best-effort: persist the time-major price matrix in L2 for this stream
        self.try_enable_persisting_l2(
            d_prices.as_device_ptr().as_raw(),
            elems
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| CudaAlmaError::InvalidInput("prices bytes overflow".into()))?,
        );

        // Prefer host-precomputed weights/normalization for accuracy.
        // Pre-scale weights to eliminate per-output multiply in kernels.
        let (mut weights_host, inv_norm) =
            compute_weights_cpu_f32(period, offset as f64, sigma as f64);
        if inv_norm != 0.0 {
            for w in &mut weights_host {
                *w *= inv_norm;
            }
        }
        let d_weights = DeviceBuffer::from_slice(&weights_host)?;
        self.launch_many_series_kernel_precomputed(
            &d_prices,
            &d_weights,
            period,
            1.0,
            cols,
            rows,
            &d_first_valids,
            &mut d_out,
        )?;

        self.stream.synchronize()?;

        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    /// Host-copy helper for many-series × one-param (FP32 output).
    pub fn alma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &AlmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaAlmaError> {
        let expected_len = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaAlmaError::InvalidInput("cols*rows overflow".into()))?;
        if out_tm.len() != expected_len {
            return Err(CudaAlmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                expected_len
            )));
        }
        let arr =
            self.alma_multi_series_one_param_time_major_dev(data_tm_f32, cols, rows, params)?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(arr.len())? };
        unsafe {
            arr.buf.async_copy_to(pinned.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }

    // ---------- Kernel launches (internal) ----------

    fn launch_batch_kernel_ondev(
        &self,
        d_prices: &DeviceBuffer<f32>,
        combos: &[AlmaParams],
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAlmaError> {
        // Compact params
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let offsets: Vec<f32> = combos.iter().map(|c| c.offset.unwrap() as f32).collect();
        let sigmas: Vec<f32> = combos.iter().map(|c| c.sigma.unwrap() as f32).collect();

            let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_offsets = DeviceBuffer::from_slice(&offsets)?;
        let d_sigmas = DeviceBuffer::from_slice(&sigmas)?;

        let func = match self.module.get_function("alma_batch_f32_ondev") {
            Ok(f) => f,
            Err(_) => self
                .module
                .get_function("alma_batch_f32_onthefly")
                .map_err(|_| CudaAlmaError::MissingKernelSymbol {
                    name: "alma_batch_f32_ondev/alma_batch_f32_onthefly",
                })?,
        };

        // Shared memory needed for weights (dynamic) ~ max_period * sizeof(f32)
        let shared_bytes = max_period
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAlmaError::InvalidInput("shared bytes overflow".into()))?
            as u32;

        // Ask occupancy for a good block size
        let (_, suggested_block) = func
            .suggested_launch_configuration(shared_bytes as usize, BlockSize::xyz(0, 0, 0))?;
        let block_x = if suggested_block > 0 {
            suggested_block
        } else {
            256
        };

        // Introspection
        unsafe {
            let this = self as *const _ as *mut CudaAlma;
            (*this).last_batch = Some(BatchKernelSelected::OnDevice { block_x });
        }
        self.maybe_log_batch_debug();

        // Chunk over grid.y if needed
        for (start, len) in Self::grid_y_chunks(n_combos) {
            let grid_x = ((series_len as u32) + block_x - 1) / block_x;
            let grid: GridSize = (grid_x, len as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.validate_launch(grid_x, len as u32, 1, block_x, 1, 1)?;

            // Offset output pointer by start * series_len elements
            let out_ptr = unsafe { d_out.as_device_ptr().add(start * series_len) };

            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, shared_bytes, stream>>>(
                        d_prices.as_device_ptr(),
                        d_periods.as_device_ptr(),
                        d_offsets.as_device_ptr(),
                        d_sigmas.as_device_ptr(),
                        series_len as i32,
                        len as i32,
                        (first_valid as i32),
                        out_ptr
                    )
                )?;
            }
        }

        Ok(())
    }

    fn launch_batch_kernel_precomputed(
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
    ) -> Result<(), CudaAlmaError> {
        // Decide between plain and tiled kernels (policy first)
        let mut use_tiled = series_len > 8192;
        let mut block_x: u32 = 256;
        let mut force_tile: Option<u32> = None;
        let mut force_two_per_thread: Option<bool> = None;
        match self.policy.batch {
            BatchKernelPolicy::Auto => {}
            BatchKernelPolicy::Plain { block_x: bx } => {
                use_tiled = false;
                block_x = bx;
            }
            BatchKernelPolicy::Tiled { tile, per_thread } => {
                use_tiled = true;
                force_tile = Some(tile);
                force_two_per_thread = Some(matches!(per_thread, BatchThreadsPerOutput::Two));
            }
        }

        if use_tiled {
            block_x = force_tile
                .unwrap_or_else(|| self.pick_tiled_block(max_period, series_len, n_combos));
            let tile = block_x as usize; // outputs per block
                                         // Automatic 2x-per-thread selection for larger periods if kernel exists
            let two_x_name = match block_x {
                128 => "alma_batch_tiled_f32_2x_tile128",
                256 => "alma_batch_tiled_f32_2x_tile256",
                512 => "alma_batch_tiled_f32_2x_tile512",
                _ => "",
            };
            let two_x_available = !two_x_name.is_empty() && self.has_function(two_x_name);
            // Decide 1x vs 2x per-thread: prefer 2x whenever available unless explicitly forced off.
            let use_two_per_thread = if let Some(v) = force_two_per_thread {
                v
            } else {
                let force_2x = matches!(std::env::var("ALMA_FORCE_2X"), Ok(v) if v == "1" || v.eq_ignore_ascii_case("true"));
                let force_1x = matches!(std::env::var("ALMA_FORCE_1X"), Ok(v) if v == "1" || v.eq_ignore_ascii_case("true"));
                if force_2x && two_x_available {
                    true
                } else if force_1x {
                    false
                } else {
                    two_x_available
                }
            };
            let threads_x = if use_two_per_thread {
                (block_x / 2).max(1)
            } else {
                block_x
            };
            // Dynamic shared: weights + tile_overlap buffer sized by outputs per block
            let elems = max_period + (tile + max_period - 1);
            let shared_bytes = (elems * std::mem::size_of::<f32>()) as u32;

            let base = if use_two_per_thread {
                if block_x == 128 {
                    "alma_batch_tiled_f32_2x_tile128"
                } else if block_x == 256 {
                    "alma_batch_tiled_f32_2x_tile256"
                } else if block_x == 512 {
                    "alma_batch_tiled_f32_2x_tile512"
                } else {
                    "alma_batch_tiled_f32_2x_tile256"
                }
            } else {
                if block_x == 128 {
                    "alma_batch_tiled_f32_tile128"
                } else if block_x == 256 {
                    "alma_batch_tiled_f32_tile256"
                } else if block_x == 512 {
                    "alma_batch_tiled_f32_tile512"
                } else {
                    "alma_batch_tiled_f32_tile256"
                }
            };
            let func = self
                .module
                .get_function(base)
                .map_err(|_| CudaAlmaError::MissingKernelSymbol { name: base })?;

            // Introspection
            unsafe {
                let this = self as *const _ as *mut CudaAlma;
                (*this).last_batch = Some(if use_two_per_thread {
                    BatchKernelSelected::Tiled2x { tile: block_x }
                } else {
                    BatchKernelSelected::Tiled1x { tile: block_x }
                });
            }
            self.maybe_log_batch_debug();

            for (start, len) in Self::grid_y_chunks(n_combos) {
                let grid_x = ((series_len as u32) + (block_x - 1)) / block_x; // divide by outputs per block
                let grid: GridSize = (grid_x, len as u32, 1).into();
                let block: BlockSize = (threads_x, 1, 1).into();
                self.validate_launch(grid_x, len as u32, 1, threads_x, 1, 1)?;
                let out_ptr = unsafe { d_out.as_device_ptr().add(start * series_len) };

                let stream = &self.stream;
                unsafe {
                    launch!(
                        func<<<grid, block, shared_bytes, stream>>>(
                            d_prices.as_device_ptr(),
                            d_weights.as_device_ptr(),
                            d_periods.as_device_ptr(),
                            d_inv_norms.as_device_ptr(),
                            (max_period as i32),
                            (series_len as i32),
                            (len as i32),
                            (first_valid as i32),
                            out_ptr
                        )
                    )?;
                }
            }
        } else {
            // Plain kernel: dynamic shared = weights only
            let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;
            let func = self
                .module
                .get_function("alma_batch_f32")
                .map_err(|_| CudaAlmaError::MissingKernelSymbol { name: "alma_batch_f32" })?;

            // Fixed block size; allow override when Auto only.
            block_x = match self.policy.batch {
                BatchKernelPolicy::Plain { block_x } => block_x,
                _ => match std::env::var("ALMA_BLOCK_X")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    Some(v) if v == 128 || v == 256 || v == 512 => v,
                    _ => 256,
                },
            };

            // Introspection
            unsafe {
                let this = self as *const _ as *mut CudaAlma;
                (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
            }
            self.maybe_log_batch_debug();

            for (start, len) in Self::grid_y_chunks(n_combos) {
                let grid_x = ((series_len as u32) + block_x - 1) / block_x;
                let grid: GridSize = (grid_x, len as u32, 1).into();
                let block: BlockSize = (block_x, 1, 1).into();
                self.validate_launch(grid_x, len as u32, 1, block_x, 1, 1)?;
                let out_ptr = unsafe { d_out.as_device_ptr().add(start * series_len) };

                let stream = &self.stream;
                unsafe {
                    launch!(
                        func<<<grid, block, shared_bytes, stream>>>(
                            d_prices.as_device_ptr(),
                            d_weights.as_device_ptr(),
                            d_periods.as_device_ptr(),
                            d_inv_norms.as_device_ptr(),
                            (max_period as i32),
                            (series_len as i32),
                            (len as i32),
                            (first_valid as i32),
                            out_ptr
                        )
                    )?;
                }
            }
        }

        Ok(())
    }

    // (removed) launch_many_series_kernel_ondev: many-series on-the-fly weights path was unused

    fn launch_many_series_kernel_precomputed(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: usize,
        inv_norm: f32,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAlmaError> {
        // Try 2D tiled variants first (TX=128, TY=4 then TY=2), fall back if not suitable.
        // We prefer 2D whenever the symbol exists and shared memory fits.
        let dev = Device::get_device(self.device_id).ok();
        let max_smem: usize = dev
            .and_then(|d| {
                d.get_attribute(cust::device::DeviceAttribute::MaxSharedMemoryPerBlock)
                    .ok()
            })
            .unwrap_or(48 * 1024) as usize;

        // Optional test/bench override: ALMA_MS_FORCE=1d|2d_ty4|2d_ty2 (only used when policy=Auto)
        let force_ms = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => std::env::var("ALMA_MS_FORCE").ok(),
            _ => None,
        };

        let try_2d = |tx: u32, ty: u32| -> Option<()> {
            let total = tx as usize + period - 1;
            // Shared = weights + tile
            let shared_elems = period.checked_add(total.checked_mul(ty as usize)? )?;
            let shared_bytes = shared_elems.checked_mul(std::mem::size_of::<f32>())? as u32;
            if (shared_bytes as usize) > max_smem {
                return None;
            }
            // Make sure symbol exists
            let fname = match (tx, ty) {
                (128, 4) => "alma_ms1p_tiled_f32_tx128_ty4",
                (128, 2) => "alma_ms1p_tiled_f32_tx128_ty2",
                _ => return None,
            };
            let func = match self.module.get_function(fname) {
                Ok(f) => f,
                Err(_) => return None,
            };
            let grid_x = ((rows as u32) + tx - 1) / tx;
            let grid_y = ((cols as u32) + ty - 1) / ty;
            let grid: GridSize = (grid_x, grid_y, 1).into();
            let block: BlockSize = (tx, ty, 1).into();
            if self
                .validate_launch(grid_x, grid_y, 1, tx, ty, 1)
                .is_err()
            {
                return None;
            }
            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, shared_bytes, stream>>>(
                        d_prices.as_device_ptr(),
                        d_weights.as_device_ptr(),
                        (period as i32),
                        inv_norm,
                        (cols as i32),
                        (rows as i32),
                        d_first_valids.as_device_ptr(),
                        d_out.as_device_ptr()
                    )
                )
                .map_err(CudaAlmaError::from)
                .ok()?;
            }
            // Introspection
            unsafe {
                let this = self as *const _ as *mut CudaAlma;
                (*this).last_many = Some(ManySeriesKernelSelected::Tiled2D { tx, ty });
            }
            self.maybe_log_many_debug();
            Some(())
        };

        // Respect explicit policy first
        match self.policy.many_series {
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                if try_2d(tx as u32, ty as u32).is_some() {
                    return Ok(());
                }
            }
            ManySeriesKernelPolicy::OneD { .. } => { /* fall through to 1D below */ }
            ManySeriesKernelPolicy::Auto => {
                // For very small column counts, 1D tends to be numerically
                // cleaner and plenty fast. Prefer 1D when cols are tiny.
                if cols < 16 {
                    // fall through to 1D mapping below
                } else if let Some(ref v) = force_ms {
                    if v.eq_ignore_ascii_case("2d_ty4") {
                        if try_2d(128, 4).is_some() {
                            return Ok(());
                        }
                    } else if v.eq_ignore_ascii_case("2d_ty2") {
                        if try_2d(128, 2).is_some() {
                            return Ok(());
                        }
                    }
                } else {
                    // Prefer TY=4 when columns are reasonably wide; else TY=2.
                    if cols >= 128 {
                        if try_2d(128, 4).is_some() {
                            return Ok(());
                        }
                        if try_2d(128, 2).is_some() {
                            return Ok(());
                        }
                    } else {
                        if try_2d(128, 2).is_some() {
                            return Ok(());
                        }
                        if try_2d(128, 4).is_some() {
                            return Ok(());
                        }
                    }
                }
            }
        }

        // Fallback: 1D mapping (current kernel)
        let func = self
            .module
            .get_function("alma_multi_series_one_param_f32")
            .map_err(|_| CudaAlmaError::MissingKernelSymbol { name: "alma_multi_series_one_param_f32" })?;

        let shared_bytes = period
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaAlmaError::InvalidInput("shared bytes overflow".into()))?
            as u32;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => 128,
        };
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x, cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        self.validate_launch(grid_x, cols as u32, 1, block_x, 1, 1)?;
        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, shared_bytes, stream>>>(
                    d_prices.as_device_ptr(),
                    d_weights.as_device_ptr(),
                    (period as i32),
                    inv_norm,
                    (cols as i32),
                    (rows as i32),
                    d_first_valids.as_device_ptr(),
                    d_out.as_device_ptr()
                )
            )?;
        }
        // Introspection
        unsafe {
            let this = self as *const _ as *mut CudaAlma;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    /// Public: many-series × one-param with device-side inputs and precomputed weights.
    pub fn alma_many_series_one_param_time_major_device_precomputed(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: usize,
        inv_norm: f32,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaAlmaError> {
        if cols == 0 || rows == 0 || period == 0 {
            return Err(CudaAlmaError::InvalidInput(
                "cols, rows, and period must be positive".into(),
            ));
        }
        self.launch_many_series_kernel_precomputed(
            d_prices_tm,
            d_weights,
            period,
            inv_norm,
            cols,
            rows,
            d_first_valids,
            d_out_tm,
        )
    }
}

// ---------- Grid expansion + CPU weights (fallback) ----------

fn expand_grid(r: &AlmaBatchRange) -> Result<Vec<AlmaParams>, CudaAlmaError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaAlmaError> {
        if step == 0 || start == end { return Ok(vec![start]); }
        if start < end { return Ok((start..=end).step_by(step.max(1)).collect()); }
        let mut v = Vec::new();
        let mut x = start as isize; let end_i = end as isize; let st = (step as isize).max(1);
        while x >= end_i { v.push(x as usize); x -= st; }
        if v.is_empty() { return Err(CudaAlmaError::InvalidInput(format!("Invalid range: start={}, end={}, step={}", start, end, step))); }
        Ok(v)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaAlmaError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return Ok(vec![start]); }
        if start < end {
            let mut v = Vec::new(); let mut x = start; let st = step.abs();
            while x <= end + 1e-12 { v.push(x); x += st; }
            if v.is_empty() { return Err(CudaAlmaError::InvalidInput(format!("Invalid range: start={}, end={}, step={}", start, end, step))); }
            return Ok(v);
        }
        let mut v = Vec::new(); let mut x = start; let st = step.abs();
        while x + 1e-12 >= end { v.push(x); x -= st; }
        if v.is_empty() { return Err(CudaAlmaError::InvalidInput(format!("Invalid range: start={}, end={}, step={}", start, end, step))); }
        Ok(v)
    }

    let periods = axis_usize(r.period)?;
    let offsets = axis_f64(r.offset)?;
    let sigmas = axis_f64(r.sigma)?;
    let cap = periods.len().checked_mul(offsets.len()).and_then(|x| x.checked_mul(sigmas.len()))
        .ok_or_else(|| CudaAlmaError::InvalidInput("range size overflow".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &p in &periods { for &o in &offsets { for &s in &sigmas {
        out.push(AlmaParams { period: Some(p), offset: Some(o), sigma: Some(s) });
    }}}
    Ok(out)
}

fn compute_weights_cpu_f32(period: usize, offset: f64, sigma: f64) -> (Vec<f32>, f32) {
    let mut weights = vec![0f32; period];
    if period == 0 {
        return (weights, 0.0);
    }
    let m = offset * (period.saturating_sub(1)) as f64;
    let s = (period as f64) / sigma;
    let s2 = 2.0 * s * s;
    let mut norm = 0.0f64;
    for i in 0..period {
        let diff = i as f64 - m;
        let w = (-((diff * diff) / s2)).exp() as f32;
        weights[i] = w;
        norm += w as f64;
    }
    let inv = if norm == 0.0 {
        0.0
    } else {
        (1.0 / norm) as f32
    };
    (weights, inv)
}

// ---------- Bench profiles (wrapper-owned) ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;
    // Additional sizes for coverage
    const ONE_SERIES_LEN_SMALL: usize = 250_000;
    const PARAM_SWEEP_SMALL: usize = 128;
    const ONE_SERIES_LEN_LARGE: usize = 2_000_000;
    const PARAM_SWEEP_LARGE: usize = 250;
    const ONE_SERIES_LEN_XL: usize = 4_000_000;
    const MANY_SERIES_COLS_SMALL: usize = 128;
    const MANY_SERIES_LEN_SMALL: usize = 500_000;
    const MANY_SERIES_COLS_LARGE: usize = 512;
    const MANY_SERIES_LEN_LARGE: usize = 2_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        // small overhead for params + 64MB headroom
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    // Preallocated, device-resident batch state to avoid per-iteration allocs/copies.
    struct AlmaBatchDeviceState {
        cuda: CudaAlma,
        d_prices: DeviceBuffer<f32>,
        d_weights: DeviceBuffer<f32>,
        d_periods: DeviceBuffer<i32>,
        d_inv_norms: DeviceBuffer<f32>,
        d_out: DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        first_valid: usize,
        warmed: bool,
    }

    impl CudaBenchState for AlmaBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .alma_batch_device(
                    &self.d_prices,
                    &self.d_weights,
                    &self.d_periods,
                    &self.d_inv_norms,
                    self.max_period as i32,
                    self.series_len as i32,
                    self.n_combos as i32,
                    self.first_valid as i32,
                    &mut self.d_out,
                )
                .expect("launch alma_batch_device");
            self.cuda.synchronize().expect("sync");
            if !self.warmed {
                self.warmed = true;
            }
        }
    }

    fn prep_alma_one_series_many_params() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaAlma::new(0).expect("cuda alma");
        // Bench pin (fast path)
        cuda.set_policy(CudaAlmaPolicy {
            batch: BatchKernelPolicy::Tiled {
                tile: 256,
                per_thread: BatchThreadsPerOutput::Two,
            },
            many_series: ManySeriesKernelPolicy::Auto,
        });
        let price = gen_series(ONE_SERIES_LEN);
        let start_period = 10usize;
        let end_period = start_period + PARAM_SWEEP - 1;
        let sweep = AlmaBatchRange {
            period: (start_period, end_period, 1),
            offset: (0.85, 0.85, 0.0),
            sigma: (6.0, 6.0, 0.0),
        };

        // Compute first_valid on host once
        let first_valid = price.iter().position(|x| !x.is_nan()).unwrap_or(0);

        // Expand combos and host-precompute weights once
        let combos = super::expand_grid(&sweep).expect("expand_grid");
        let n_combos = combos.len();
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);

        let mut periods_i32 = vec![0i32; n_combos];
        let mut inv_norms = vec![0f32; n_combos];
        let mut weights_flat = vec![0f32; n_combos * max_period];
        for (idx, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap() as usize;
            let offset = prm.offset.unwrap();
            let sigma = prm.sigma.unwrap();
            let (mut weights, inv_norm) = super::compute_weights_cpu_f32(period, offset, sigma);
            periods_i32[idx] = period as i32;
            if inv_norm != 0.0 {
                for w in &mut weights {
                    *w *= inv_norm;
                }
            }
            inv_norms[idx] = 1.0;
            let base = idx * max_period;
            weights_flat[base..base + period].copy_from_slice(&weights);
        }

        // Upload once
        let d_prices =
            unsafe { DeviceBuffer::from_slice_async(&price, &cuda.stream) }.expect("d_prices");
        let d_weights = DeviceBuffer::from_slice(&weights_flat).expect("d_weights");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms).expect("d_inv_norms");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n_combos * ONE_SERIES_LEN, &cuda.stream) }
                .expect("d_out");
        cuda.synchronize().expect("sync after prep");

        Box::new(AlmaBatchDeviceState {
            cuda,
            d_prices,
            d_weights,
            d_periods,
            d_inv_norms,
            d_out,
            series_len: ONE_SERIES_LEN,
            n_combos,
            max_period,
            first_valid,
            warmed: false,
        })
    }

    // Generic prep for batch with (LEN, SWEEP)
    fn prep_batch_len_sweep<const LEN: usize, const SWEEP: usize>() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaAlma::new(0).expect("cuda alma");
        // Bench pin (fast path)
        cuda.set_policy(CudaAlmaPolicy {
            batch: BatchKernelPolicy::Tiled {
                tile: 256,
                per_thread: BatchThreadsPerOutput::Two,
            },
            many_series: ManySeriesKernelPolicy::Auto,
        });
        let price = gen_series(LEN);
        let start_period = 10usize;
        let end_period = start_period + SWEEP - 1;
        let sweep = AlmaBatchRange {
            period: (start_period, end_period, 1),
            offset: (0.85, 0.85, 0.0),
            sigma: (6.0, 6.0, 0.0),
        };
        let first_valid = price.iter().position(|x| !x.is_nan()).unwrap_or(0);
        let combos = super::expand_grid(&sweep).expect("expand_grid");
        let n_combos = combos.len();
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        let mut periods_i32 = vec![0i32; n_combos];
        let mut inv_norms = vec![0f32; n_combos];
        let mut weights_flat = vec![0f32; n_combos * max_period];
        for (idx, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap() as usize;
            let (mut weights, inv_norm) =
                super::compute_weights_cpu_f32(period, prm.offset.unwrap(), prm.sigma.unwrap());
            periods_i32[idx] = period as i32;
            if inv_norm != 0.0 {
                for w in &mut weights {
                    *w *= inv_norm;
                }
            }
            inv_norms[idx] = 1.0;
            let base = idx * max_period;
            weights_flat[base..base + period].copy_from_slice(&weights);
        }
        let d_prices =
            unsafe { DeviceBuffer::from_slice_async(&price, &cuda.stream) }.expect("d_prices");
        let d_weights = DeviceBuffer::from_slice(&weights_flat).expect("d_weights");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms).expect("d_inv_norms");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n_combos * LEN, &cuda.stream) }
                .expect("d_out");
        cuda.synchronize().expect("sync after prep");
        Box::new(AlmaBatchDeviceState {
            cuda,
            d_prices,
            d_weights,
            d_periods,
            d_inv_norms,
            d_out,
            series_len: LEN,
            n_combos,
            max_period,
            first_valid,
            warmed: false,
        })
    }

    // Fast-mode variants of the prep functions
    fn prep_alma_one_series_many_params_fast() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaAlma::new(0).expect("cuda alma");
        cuda.set_policy(CudaAlmaPolicy {
            batch: BatchKernelPolicy::Tiled {
                tile: 256,
                per_thread: BatchThreadsPerOutput::Two,
            },
            many_series: ManySeriesKernelPolicy::Auto,
        });
        let price = gen_series(ONE_SERIES_LEN);
        let start_period = 10usize;
        let end_period = start_period + PARAM_SWEEP - 1;
        let sweep = AlmaBatchRange {
            period: (start_period, end_period, 1),
            offset: (0.85, 0.85, 0.0),
            sigma: (6.0, 6.0, 0.0),
        };
        let first_valid = price.iter().position(|x| !x.is_nan()).unwrap_or(0);
        let combos = super::expand_grid(&sweep).expect("expand_grid");
        let n_combos = combos.len();
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        let mut periods_i32 = vec![0i32; n_combos];
        let mut inv_norms = vec![0f32; n_combos];
        let mut weights_flat = vec![0f32; n_combos * max_period];
        for (idx, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap() as usize;
            let offset = prm.offset.unwrap();
            let sigma = prm.sigma.unwrap();
            let (mut weights, inv_norm) = super::compute_weights_cpu_f32(period, offset, sigma);
            periods_i32[idx] = period as i32;
            if inv_norm != 0.0 {
                for w in &mut weights {
                    *w *= inv_norm;
                }
            }
            inv_norms[idx] = 1.0;
            let base = idx * max_period;
            weights_flat[base..base + period].copy_from_slice(&weights);
        }
        let d_prices =
            unsafe { DeviceBuffer::from_slice_async(&price, &cuda.stream) }.expect("d_prices");
        let d_weights = DeviceBuffer::from_slice(&weights_flat).expect("d_weights");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms).expect("d_inv_norms");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n_combos * ONE_SERIES_LEN, &cuda.stream) }
                .expect("d_out");
        cuda.synchronize().expect("sync after prep");
        Box::new(AlmaBatchDeviceState {
            cuda,
            d_prices,
            d_weights,
            d_periods,
            d_inv_norms,
            d_out,
            series_len: ONE_SERIES_LEN,
            n_combos,
            max_period,
            first_valid,
            warmed: false,
        })
    }

    // removed fast-only prep

    struct AlmaManySeriesDeviceState {
        cuda: CudaAlma,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        d_weights: DeviceBuffer<f32>,
        d_out_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        inv_norm: f32,
        warmed: bool,
    }

    impl CudaBenchState for AlmaManySeriesDeviceState {
        fn launch(&mut self) {
            self.cuda
                .alma_many_series_one_param_time_major_device_precomputed(
                    &self.d_prices_tm,
                    &self.d_weights,
                    self.period,
                    self.inv_norm,
                    self.cols,
                    self.rows,
                    &self.d_first_valids,
                    &mut self.d_out_tm,
                )
                .expect("alma many-series precomputed");
            self.cuda.synchronize().expect("sync");
            if !self.warmed {
                self.warmed = true;
            }
        }
    }

    // Generic prep for many-series (COLS, ROWS)
    fn prep_many_series_generic<const COLS: usize, const ROWS: usize>() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaAlma::new(0).expect("cuda alma");
        // Bench pin (fast path)
        cuda.set_policy(CudaAlmaPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Tiled2D { tx: 128, ty: 4 },
        });
        let prices_tm = gen_time_major_prices(COLS, ROWS);
        let period = 64usize;
        let (mut weights, inv_norm) = super::compute_weights_cpu_f32(period, 0.85, 6.0);
        if inv_norm != 0.0 {
            for w in &mut weights {
                *w *= inv_norm;
            }
        }
        let mut first_valids = vec![0i32; COLS];
        for s in 0..COLS {
            let mut fv = 0usize;
            for t in 0..ROWS {
                if !prices_tm[t * COLS + s].is_nan() {
                    fv = t;
                    break;
                }
            }
            first_valids[s] = fv as i32;
        }
        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(&prices_tm, &cuda.stream) }
            .expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_weights = DeviceBuffer::from_slice(&weights).expect("d_weights");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(COLS * ROWS, &cuda.stream) }
                .expect("d_out_tm");
        cuda.synchronize().expect("sync after prep");
        Box::new(AlmaManySeriesDeviceState {
            cuda,
            d_prices_tm,
            d_first_valids,
            d_weights,
            d_out_tm,
            cols: COLS,
            rows: ROWS,
            period,
            inv_norm: 1.0,
            warmed: false,
        })
    }

    // removed fast-only prep

    fn prep_alma_many_series_one_param() -> Box<dyn CudaBenchState> {
        let mut cuda = CudaAlma::new(0).expect("cuda alma");
        // Bench pin (fast path)
        cuda.set_policy(CudaAlmaPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Tiled2D { tx: 128, ty: 4 },
        });
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let prices_tm = gen_time_major_prices(cols, rows);

        // Params fixed
        let period = 64usize;
        let offset = 0.85f64;
        let sigma = 6.0f64;
        let (mut weights, inv_norm) = super::compute_weights_cpu_f32(period, offset, sigma);
        if inv_norm != 0.0 {
            for w in &mut weights {
                *w *= inv_norm;
            }
        }

        // Compute first_valids
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = 0usize;
            for t in 0..rows {
                if !prices_tm[t * cols + s].is_nan() {
                    fv = t;
                    break;
                }
            }
            first_valids[s] = fv as i32;
        }

        // Upload once
        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(&prices_tm, &cuda.stream) }
            .expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_weights = DeviceBuffer::from_slice(&weights).expect("d_weights");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &cuda.stream) }
                .expect("d_out_tm");
        cuda.synchronize().expect("sync after prep");

        Box::new(AlmaManySeriesDeviceState {
            cuda,
            d_prices_tm,
            d_first_valids,
            d_weights,
            d_out_tm,
            cols,
            rows,
            period,
            inv_norm: 1.0,
            warmed: false,
        })
    }

    // removed fast-only prep

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            // Accurate scenarios
            CudaBenchScenario::new(
                "alma",
                "one_series_many_params",
                "alma_cuda_batch_dev",
                "250k_x_250",
                prep_batch_len_sweep::<{ ONE_SERIES_LEN_SMALL }, { PARAM_SWEEP }>,
            )
            .with_sample_size(12)
            .with_mem_required(
                ONE_SERIES_LEN_SMALL * 4
                    + ONE_SERIES_LEN_SMALL * PARAM_SWEEP * 4
                    + 64 * 1024 * 1024,
            ),
            CudaBenchScenario::new(
                "alma",
                "one_series_many_params",
                "alma_cuda_batch_dev",
                "1m_x_250",
                prep_alma_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "alma",
                "one_series_many_params",
                "alma_cuda_batch_dev",
                "4m_x_250",
                prep_batch_len_sweep::<{ ONE_SERIES_LEN_XL }, { PARAM_SWEEP }>,
            )
            .with_sample_size(6)
            .with_mem_required(
                ONE_SERIES_LEN_XL * 4 + ONE_SERIES_LEN_XL * PARAM_SWEEP * 4 + 64 * 1024 * 1024,
            ),
            CudaBenchScenario::new(
                "alma",
                "many_series_one_param",
                "alma_cuda_many_series_one_param",
                "128x500k",
                prep_many_series_generic::<{ MANY_SERIES_COLS_SMALL }, { MANY_SERIES_LEN_SMALL }>,
            )
            .with_sample_size(12)
            .with_mem_required(
                MANY_SERIES_COLS_SMALL * MANY_SERIES_LEN_SMALL * 4 * 2 + 64 * 1024 * 1024,
            ),
            CudaBenchScenario::new(
                "alma",
                "many_series_one_param",
                "alma_cuda_many_series_one_param",
                "250x1m",
                prep_alma_many_series_one_param,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series_one_param()),
            CudaBenchScenario::new(
                "alma",
                "many_series_one_param",
                "alma_cuda_many_series_one_param",
                "512x2m",
                prep_many_series_generic::<{ MANY_SERIES_COLS_LARGE }, { MANY_SERIES_LEN_LARGE }>,
            )
            .with_sample_size(8)
            .with_mem_required(
                MANY_SERIES_COLS_LARGE * MANY_SERIES_LEN_LARGE * 4 * 2 + 64 * 1024 * 1024,
            ),
        ]
    }
}
