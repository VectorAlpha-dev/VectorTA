//! CUDA wrapper for the New Adaptive Moving Average (NAMA) kernels.
//!
//! Aligned with the ALMA wrapper patterns:
//! - PTX JIT with DetermineTargetFromContext and O2 fallback.
//! - NON_BLOCKING stream.
//! - Policy enums for kernel selection with last-selected introspection and
//!   optional BENCH_DEBUG output.
//! - VRAM estimation with mem_get_info and 64MiB headroom.
//! - Batch grid chunking over parameter rows (combos) to respect grid limits.
//!
//! Kernels are recurrence/IIR style (one thread per series/row does the time
//! scan). Block size mainly impacts warmup NaN writes; default is 128 threads.
//! We expose the same policy surface as ALMA for a consistent user experience.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::nama::{NamaBatchRange, NamaParams};
use cust::context::{CacheConfig, Context, SharedMemoryConfig};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::AsyncCopyDestination;
use cust::memory::{mem_get_info, CopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys::{
    cuDeviceGetAttribute, cuFuncSetAttribute, CUdevice_attribute, CUfunction_attribute,
};
use std::env;
use std::ffi::c_void;
use std::fmt;

// -------- Kernel selection policy (mirror ALMA surface) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchThreadsPerOutput {
    One,
    Two,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    /// Recurrence kernels are 1D (combo-per-block). Only block_x matters.
    Plain {
        block_x: u32,
    },
    /// Tiled hint accepted for API symmetry; mapped to `Plain { block_x: tile }`.
    Tiled {
        tile: u32,
        per_thread: BatchThreadsPerOutput,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    /// 1D series-per-block mapping; block_x controls warmup fill speed.
    OneD {
        block_x: u32,
    },
    /// 2D hint accepted for API symmetry; mapped to `OneD { block_x: tx }`.
    Tiled2D {
        tx: u32,
        ty: u32,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaNamaPolicy {
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

// -------- Introspection (selected kernel) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

#[derive(Debug)]
pub enum CudaNamaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaNamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaNamaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaNamaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaNamaError {}

pub struct CudaNama {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaNamaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaNama {
    // ---- Device/Driver helpers -------------------------------------------------
    #[inline]
    fn query_max_grid_x(&self) -> usize {
        // Prefer driver attribute; fallback to historical 65_535
        let dev = Device::get_device(self.device_id).ok();
        if let Some(d) = dev {
            unsafe {
                let mut v: i32 = 0;
                let _ = cuDeviceGetAttribute(
                    &mut v as *mut _,
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                    d.as_raw(),
                );
                if v > 0 {
                    return v as usize;
                }
            }
        }
        65_535
    }

    // Return (default_per_block, optin_per_block). If opt-in isn't supported, returns (default, default).
    #[inline]
    fn query_smem_per_block_limits(&self) -> (usize, usize) {
        let dev = match Device::get_device(self.device_id) {
            Ok(d) => d,
            Err(_) => return (48 * 1024, 48 * 1024),
        };
        let default = dev
            .get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock)
            .unwrap_or(48 * 1024) as usize;
        let mut optin = default as i32;
        unsafe {
            let _ = cuDeviceGetAttribute(
                &mut optin as *mut _,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                dev.as_raw(),
            );
        }
        (default, optin.max(default as i32) as usize)
    }

    // Try to raise the per‑block dynamic smem limit for this function.
    #[inline]
    fn try_enable_large_dynamic_smem(func: &Function, bytes: usize) {
        unsafe {
            let _ = cuFuncSetAttribute(
                func.to_raw(),
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes as i32,
            );
            let _ = cuFuncSetAttribute(
                func.to_raw(),
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                100,
            );
        }
    }
    #[inline]
    fn has_function(&self, name: &str) -> bool {
        self.module.get_function(name).is_ok()
    }
    pub fn new(device_id: usize) -> Result<Self, CudaNamaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/nama_kernel.ptx"));
        let module = match Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        ) {
            Ok(m) => m,
            Err(_) => Module::from_ptx(ptx, &[]).map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaNamaPolicy {
                batch: BatchKernelPolicy::Auto,
                many_series: ManySeriesKernelPolicy::Auto,
            },
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create with explicit policy.
    pub fn new_with_policy(
        device_id: usize,
        policy: CudaNamaPolicy,
    ) -> Result<Self, CudaNamaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaNamaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaNamaPolicy {
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
        use std::sync::atomic::{AtomicBool, Ordering};
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] NAMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaNama)).debug_batch_logged = true;
                }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] NAMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaNama)).debug_many_logged = true;
                }
            }
        }
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

    fn expand_periods(range: &NamaBatchRange) -> Vec<NamaParams> {
        let (start, end, step) = range.period;
        let periods = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step.max(1)).collect::<Vec<_>>()
        };
        periods
            .into_iter()
            .map(|p| NamaParams { period: Some(p) })
            .collect()
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        high: Option<&[f32]>,
        low: Option<&[f32]>,
        close: Option<&[f32]>,
        sweep: &NamaBatchRange,
    ) -> Result<(Vec<NamaParams>, usize, usize, usize, bool), CudaNamaError> {
        if prices.is_empty() {
            return Err(CudaNamaError::InvalidInput("price data is empty".into()));
        }
        let has_ohlc = high.is_some() || low.is_some() || close.is_some();
        if has_ohlc {
            if high.is_none() || low.is_none() || close.is_none() {
                return Err(CudaNamaError::InvalidInput(
                    "when providing OHLC data, high/low/close must all be present".into(),
                ));
            }
            let len = prices.len();
            if high.unwrap().len() != len
                || low.unwrap().len() != len
                || close.unwrap().len() != len
            {
                return Err(CudaNamaError::InvalidInput(
                    "price/high/low/close slices must have equal length".into(),
                ));
            }
        }

        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaNamaError::InvalidInput("all price values are NaN".into()))?;

        let combos = Self::expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaNamaError::InvalidInput(
                "no parameter combinations generated".into(),
            ));
        }

        let series_len = prices.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaNamaError::InvalidInput("period must be >= 1".into()));
            }
            if period > series_len {
                return Err(CudaNamaError::InvalidInput(format!(
                    "period {} exceeds series length {}",
                    period, series_len
                )));
            }
            let valid = series_len - first_valid;
            if valid < period {
                return Err(CudaNamaError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    period, valid
                )));
            }
            max_period = max_period.max(period);
        }

        Ok((combos, first_valid, series_len, max_period, has_ohlc))
    }

    fn launch_batch_kernel_chunk(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_high: Option<&DeviceBuffer<f32>>,
        d_low: Option<&DeviceBuffer<f32>>,
        d_close: Option<&DeviceBuffer<f32>>,
        d_prefix_tr: Option<&DeviceBuffer<f32>>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_chunk: usize,
        first_valid: usize,
        chunk_max_period: usize,
        has_ohlc: bool,
        start_combo: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaNamaError> {
        if chunk_max_period == 0 {
            return Err(CudaNamaError::InvalidInput(
                "chunk_max_period must be positive".into(),
            ));
        }

        // Shared memory layout: [dq_max:int[cap] | dq_min:int[cap] | tr_ring:float[period]]
        let shared_bytes = (chunk_max_period + 1)
            .checked_mul(2 * std::mem::size_of::<i32>())
            .ok_or_else(|| CudaNamaError::InvalidInput("shared memory size overflow".into()))?;
        let shared_bytes = shared_bytes
            .checked_add(
                chunk_max_period
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        CudaNamaError::InvalidInput("shared memory size overflow".into())
                    })?,
            )
            .ok_or_else(|| CudaNamaError::InvalidInput("shared memory size overflow".into()))?;
        // Select function (prefix or full) first
        let (mut func, is_prefix) = if let Some(_) = d_prefix_tr {
            (
                self.module
                    .get_function("nama_batch_prefix_f32")
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
                true,
            )
        } else {
            (
                self.module
                    .get_function("nama_batch_f32")
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
                false,
            )
        };

        // Shared/occupancy preferences
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        let _ = func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize);
        Self::try_enable_large_dynamic_smem(&func, shared_bytes);

        // Block size: Auto uses driver suggestion, otherwise honor policy
        let user_block = match self.policy.batch {
            BatchKernelPolicy::Auto => None,
            BatchKernelPolicy::Plain { block_x }
            | BatchKernelPolicy::Tiled { tile: block_x, .. } => Some(block_x.max(32)),
        };
        let mut block_x = if let Some(bx) = user_block {
            bx
        } else {
            let (_, suggested) = func
                .suggested_launch_configuration(shared_bytes as usize, BlockSize::xyz(0, 0, 0))
                .unwrap_or((0, 128));
            if suggested > 0 {
                suggested
            } else {
                128
            }
        };
        block_x = block_x.clamp(32, 1024);
        let grid: GridSize = (n_chunk as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        // If available dynamic smem is still lower, try once more to opt-in
        if let Ok(avail) = func.available_dynamic_shared_memory_per_block(grid, block) {
            if shared_bytes > avail {
                Self::try_enable_large_dynamic_smem(&func, shared_bytes);
            }
        }

        // Launch
        if is_prefix {
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut prefix_ptr = d_prefix_tr.unwrap().as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().add(start_combo).as_raw();
                let mut series_len_i = series_len as i32;
                let mut combos_i = n_chunk as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start_combo * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut prefix_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes as u32, args)
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
            }
        } else {
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut high_ptr = d_high.map(|buf| buf.as_device_ptr().as_raw()).unwrap_or(0);
                let mut low_ptr = d_low.map(|buf| buf.as_device_ptr().as_raw()).unwrap_or(0);
                let mut close_ptr = d_close.map(|buf| buf.as_device_ptr().as_raw()).unwrap_or(0);
                let mut has_ohlc_i = if has_ohlc { 1i32 } else { 0i32 };
                let mut periods_ptr = d_periods.as_device_ptr().add(start_combo).as_raw();
                let mut series_len_i = series_len as i32;
                let mut combos_i = n_chunk as i32;
                let mut first_valid_i = first_valid as i32;
                let mut out_ptr = d_out.as_device_ptr().add(start_combo * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut high_ptr as *mut _ as *mut c_void,
                    &mut low_ptr as *mut _ as *mut c_void,
                    &mut close_ptr as *mut _ as *mut c_void,
                    &mut has_ohlc_i as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, shared_bytes as u32, args)
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
            }
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        high: Option<&[f32]>,
        low: Option<&[f32]>,
        close: Option<&[f32]>,
        combos: &[NamaParams],
        first_valid: usize,
        series_len: usize,
        max_period: usize,
        has_ohlc: bool,
    ) -> Result<DeviceArrayF32, CudaNamaError> {
        let n_combos = combos.len();
        if n_combos == 0 {
            return Err(CudaNamaError::InvalidInput("no period combinations".into()));
        }

        let mut periods_i32 = Vec::with_capacity(n_combos);
        for prm in combos {
            let period = prm.period.unwrap();
            periods_i32.push(period as i32);
        }

        // Decide once whether we’ll use the prefix kernel
        let use_prefix = !has_ohlc && self.has_function("nama_batch_prefix_f32");

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let ohlc_bytes = if has_ohlc {
            3 * series_len * std::mem::size_of::<f32>()
        } else {
            0
        };
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        // Budget prefix only when it will be used
        let prefix_bytes = if use_prefix {
            (series_len + 1) * std::mem::size_of::<f32>()
        } else {
            0
        };
        let required = prices_bytes + ohlc_bytes + periods_bytes + out_bytes + prefix_bytes;
        let headroom = 64 * 1024 * 1024; // 64 MiB safety margin
        if !Self::will_fit(required, headroom) {
            return Err(CudaNamaError::InvalidInput(
                "not enough free device memory".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(prices).map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        let d_high = if has_ohlc {
            Some(
                DeviceBuffer::from_slice(high.unwrap())
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
            )
        } else {
            None
        };
        let d_low = if has_ohlc {
            Some(
                DeviceBuffer::from_slice(low.unwrap())
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
            )
        } else {
            None
        };
        let d_close = if has_ohlc {
            Some(
                DeviceBuffer::from_slice(close.unwrap())
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
            )
        } else {
            None
        };

        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        // Optional prefix path: only when no OHLC is provided and kernel exists
        let d_prefix = if use_prefix {
            // Build degenerate TR prefix (NaN-insensitive): prefix[0]=0, prefix[t]=sum_{k=1..t}|p[k]-p[k-1]|
            let mut prefix: Vec<f32> = Vec::with_capacity(series_len + 1);
            prefix.push(0.0f32);
            let mut acc = 0.0f32;
            for i in 1..series_len {
                let cur = prices[i];
                let prev = prices[i - 1];
                let diff = if cur.is_nan() || prev.is_nan() {
                    0.0f32
                } else {
                    (cur - prev).abs()
                };
                acc += diff;
                prefix.push(acc);
            }
            // pad to series_len + 1
            prefix.push(acc);
            Some(
                DeviceBuffer::from_slice(&prefix)
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
            )
        } else {
            None
        };

        // Launch in chunks to respect grid size limits
        let max_grid_x = self.query_max_grid_x();
        let mut launched_any = false;
        let mut start = 0usize;
        while start < n_combos {
            let len = (n_combos - start).min(max_grid_x);
            // Periods are generated in ascending order; last in chunk is max
            let chunk_max_period = periods_i32[start + len - 1] as usize;
            self.launch_batch_kernel_chunk(
                &d_prices,
                d_high.as_ref(),
                d_low.as_ref(),
                d_close.as_ref(),
                d_prefix.as_ref(),
                &d_periods,
                series_len,
                len,
                first_valid,
                chunk_max_period,
                has_ohlc,
                start,
                &mut d_out,
            )?;
            launched_any = true;
            start += len;
        }
        if launched_any {
            unsafe {
                let this = self as *const _ as *mut CudaNama;
                let bx = match self.policy.batch {
                    BatchKernelPolicy::Plain { block_x } => block_x,
                    _ => 128,
                };
                (*this).last_batch = Some(BatchKernelSelected::Plain { block_x: bx });
            }
            self.maybe_log_batch_debug();
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn nama_batch_dev(
        &self,
        prices: &[f32],
        sweep: &NamaBatchRange,
    ) -> Result<DeviceArrayF32, CudaNamaError> {
        let (combos, first_valid, series_len, max_period, has_ohlc) =
            Self::prepare_batch_inputs(prices, None, None, None, sweep)?;
        debug_assert!(!has_ohlc);
        self.run_batch_kernel(
            prices,
            None,
            None,
            None,
            &combos,
            first_valid,
            series_len,
            max_period,
            false,
        )
    }

    pub fn nama_batch_with_ohlc_dev(
        &self,
        prices: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
        sweep: &NamaBatchRange,
    ) -> Result<DeviceArrayF32, CudaNamaError> {
        let (combos, first_valid, series_len, max_period, has_ohlc) =
            Self::prepare_batch_inputs(prices, Some(high), Some(low), Some(close), sweep)?;
        self.run_batch_kernel(
            prices,
            Some(high),
            Some(low),
            Some(close),
            &combos,
            first_valid,
            series_len,
            max_period,
            has_ohlc,
        )
    }

    pub fn nama_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &NamaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<NamaParams>), CudaNamaError> {
        let (combos, first_valid, series_len, max_period, has_ohlc) =
            Self::prepare_batch_inputs(prices, None, None, None, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaNamaError::InvalidInput(format!(
                "output slice len {} != expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(
            prices,
            None,
            None,
            None,
            &combos,
            first_valid,
            series_len,
            max_period,
            has_ohlc,
        )?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn nama_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_high: Option<&DeviceBuffer<f32>>,
        d_low: Option<&DeviceBuffer<f32>>,
        d_close: Option<&DeviceBuffer<f32>>,
        d_periods: &DeviceBuffer<i32>,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        max_period: i32,
        has_ohlc: bool,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaNamaError> {
        if series_len <= 0 || n_combos <= 0 || max_period <= 0 {
            return Err(CudaNamaError::InvalidInput(
                "series_len, n_combos, and max_period must be positive".into(),
            ));
        }
        self.launch_batch_kernel_chunk(
            d_prices,
            d_high,
            d_low,
            d_close,
            None,
            d_periods,
            series_len as usize,
            n_combos as usize,
            first_valid.max(0) as usize,
            max_period as usize,
            has_ohlc,
            0,
            d_out,
        )
    }

    fn run_many_series_kernel(
        &self,
        prices_tm: &[f32],
        high_tm: Option<&[f32]>,
        low_tm: Option<&[f32]>,
        close_tm: Option<&[f32]>,
        num_series: usize,
        series_len: usize,
        first_valids: &[i32],
        period: usize,
        has_ohlc: bool,
    ) -> Result<DeviceArrayF32, CudaNamaError> {
        let total = num_series * series_len;
        let prices_bytes = total * std::mem::size_of::<f32>();
        let ohlc_bytes = if has_ohlc {
            3 * total * std::mem::size_of::<f32>()
        } else {
            0
        };
        let fv_bytes = first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes = total * std::mem::size_of::<f32>();
        let required = prices_bytes + ohlc_bytes + fv_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaNamaError::InvalidInput(
                "not enough free device memory".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(prices_tm).map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        let d_high = if has_ohlc {
            Some(
                DeviceBuffer::from_slice(high_tm.unwrap())
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
            )
        } else {
            None
        };
        let d_low = if has_ohlc {
            Some(
                DeviceBuffer::from_slice(low_tm.unwrap())
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
            )
        } else {
            None
        };
        let d_close = if has_ohlc {
            Some(
                DeviceBuffer::from_slice(close_tm.unwrap())
                    .map_err(|e| CudaNamaError::Cuda(e.to_string()))?,
            )
        } else {
            None
        };
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(total) }
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            d_high.as_ref(),
            d_low.as_ref(),
            d_close.as_ref(),
            num_series,
            series_len,
            period,
            &d_first_valids,
            has_ohlc,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
        })
    }

    pub fn nama_many_series_one_param_time_major_dev(
        &self,
        prices_tm: &[f32],
        num_series: usize,
        series_len: usize,
        params: &NamaParams,
    ) -> Result<DeviceArrayF32, CudaNamaError> {
        let (first_valids, period, has_ohlc) = Self::prepare_many_series_inputs(
            prices_tm, None, None, None, num_series, series_len, params,
        )?;
        self.run_many_series_kernel(
            prices_tm,
            None,
            None,
            None,
            num_series,
            series_len,
            &first_valids,
            period,
            has_ohlc,
        )
    }

    pub fn nama_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm: &[f32],
        num_series: usize,
        series_len: usize,
        params: &NamaParams,
        out: &mut [f32],
    ) -> Result<(), CudaNamaError> {
        if out.len() != num_series * series_len {
            return Err(CudaNamaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                num_series * series_len
            )));
        }
        let arr = self
            .nama_many_series_one_param_time_major_dev(prices_tm, num_series, series_len, params)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))
    }

    pub fn nama_many_series_one_param_time_major_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_high: Option<&DeviceBuffer<f32>>,
        d_low: Option<&DeviceBuffer<f32>>,
        d_close: Option<&DeviceBuffer<f32>>,
        num_series: i32,
        series_len: i32,
        period: i32,
        d_first_valids: &DeviceBuffer<i32>,
        has_ohlc: bool,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaNamaError> {
        if num_series <= 0 || series_len <= 0 || period <= 0 {
            return Err(CudaNamaError::InvalidInput(
                "num_series, series_len, and period must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices,
            d_high,
            d_low,
            d_close,
            num_series as usize,
            series_len as usize,
            period as usize,
            d_first_valids,
            has_ohlc,
            d_out,
        )
    }

    fn prepare_many_series_inputs(
        prices_tm: &[f32],
        high_tm: Option<&[f32]>,
        low_tm: Option<&[f32]>,
        close_tm: Option<&[f32]>,
        num_series: usize,
        series_len: usize,
        params: &NamaParams,
    ) -> Result<(Vec<i32>, usize, bool), CudaNamaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaNamaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if prices_tm.len() != num_series * series_len {
            return Err(CudaNamaError::InvalidInput(format!(
                "price tensor length {} != num_series*series_len {}",
                prices_tm.len(),
                num_series * series_len
            )));
        }
        let has_ohlc = high_tm.is_some() || low_tm.is_some() || close_tm.is_some();
        if has_ohlc {
            if high_tm.is_none() || low_tm.is_none() || close_tm.is_none() {
                return Err(CudaNamaError::InvalidInput(
                    "when supplying OHLC tensors, high/low/close must all be provided".into(),
                ));
            }
            let expected = num_series * series_len;
            if high_tm.unwrap().len() != expected
                || low_tm.unwrap().len() != expected
                || close_tm.unwrap().len() != expected
            {
                return Err(CudaNamaError::InvalidInput(
                    "price/high/low/close tensors must share the same length".into(),
                ));
            }
        }

        let period = params
            .period
            .ok_or_else(|| CudaNamaError::InvalidInput("period must be specified".into()))?;
        if period == 0 || period > series_len {
            return Err(CudaNamaError::InvalidInput(format!(
                "invalid period {} for series_len {}",
                period, series_len
            )));
        }

        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let idx = t * num_series + series;
                if !prices_tm[idx].is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv_i = fv.ok_or_else(|| {
                CudaNamaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            let valid = series_len - fv_i as usize;
            if valid < period {
                return Err(CudaNamaError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    series, period, valid
                )));
            }
            first_valids.push(fv_i);
        }

        Ok((first_valids, period, has_ohlc))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_high: Option<&DeviceBuffer<f32>>,
        d_low: Option<&DeviceBuffer<f32>>,
        d_close: Option<&DeviceBuffer<f32>>,
        num_series: usize,
        series_len: usize,
        period: usize,
        d_first_valids: &DeviceBuffer<i32>,
        has_ohlc: bool,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaNamaError> {
        if period == 0 {
            return Err(CudaNamaError::InvalidInput(
                "period must be positive".into(),
            ));
        }
        // Shared memory: two deques (int) + TR ring (float)
        let shared_bytes = (period + 1)
            .checked_mul(2 * std::mem::size_of::<i32>())
            .ok_or_else(|| CudaNamaError::InvalidInput("shared memory size overflow".into()))?;
        let shared_bytes = shared_bytes
            .checked_add(
                period
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        CudaNamaError::InvalidInput("shared memory size overflow".into())
                    })?,
            )
            .ok_or_else(|| CudaNamaError::InvalidInput("shared memory size overflow".into()))?;
        let mut func = self
            .module
            .get_function("nama_many_series_one_param_time_major_f32")
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;

        // Preferences and opt-in
        let _ = func.set_cache_config(CacheConfig::PreferShared);
        let _ = func.set_shared_memory_config(SharedMemoryConfig::FourByteBankSize);
        Self::try_enable_large_dynamic_smem(&func, shared_bytes);

        // Policy block size or suggested
        let user_block = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => None,
            ManySeriesKernelPolicy::OneD { block_x }
            | ManySeriesKernelPolicy::Tiled2D { tx: block_x, .. } => Some(block_x.max(32)),
        };
        let mut block_x = if let Some(bx) = user_block {
            bx
        } else {
            let (_, suggested) = func
                .suggested_launch_configuration(shared_bytes as usize, BlockSize::xyz(0, 0, 0))
                .unwrap_or((0, 128));
            if suggested > 0 {
                suggested
            } else {
                128
            }
        };
        block_x = block_x.clamp(32, 1024);
        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        // Check availability with this launch shape
        if let Ok(avail) = func.available_dynamic_shared_memory_per_block(grid, block) {
            if shared_bytes > avail {
                Self::try_enable_large_dynamic_smem(&func, shared_bytes);
            }
        }

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut high_ptr = d_high.map(|buf| buf.as_device_ptr().as_raw()).unwrap_or(0);
            let mut low_ptr = d_low.map(|buf| buf.as_device_ptr().as_raw()).unwrap_or(0);
            let mut close_ptr = d_close.map(|buf| buf.as_device_ptr().as_raw()).unwrap_or(0);
            let mut has_ohlc_i = if has_ohlc { 1i32 } else { 0i32 };
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut period_i = period as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut high_ptr as *mut _ as *mut c_void,
                &mut low_ptr as *mut _ as *mut c_void,
                &mut close_ptr as *mut _ as *mut c_void,
                &mut has_ohlc_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (num_series as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)
                .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        }

        unsafe {
            let this = self as *const _ as *mut CudaNama;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        Ok(())
    }
}

// ---- Optional pinned I/O helpers (additive, no API break) ---------------------
impl CudaNama {
    /// Return page-locked output for batch, useful for overlapping copies.
    pub fn nama_batch_into_pinned_host_f32(
        &self,
        prices: &[f32],
        sweep: &NamaBatchRange,
    ) -> Result<(LockedBuffer<f32>, usize, usize, Vec<NamaParams>), CudaNamaError> {
        let (combos, first_valid, series_len, max_period, has_ohlc) =
            Self::prepare_batch_inputs(prices, None, None, None, sweep)?;
        let arr = self.run_batch_kernel(
            prices,
            None,
            None,
            None,
            &combos,
            first_valid,
            series_len,
            max_period,
            has_ohlc,
        )?;
        let mut pinned = unsafe { LockedBuffer::<f32>::uninitialized(arr.rows * arr.cols) }
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        unsafe {
            arr.buf
                .async_copy_to(&mut pinned, &self.stream)
                .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        Ok((pinned, arr.rows, arr.cols, combos))
    }

    /// Return page-locked output for many-series time-major run.
    pub fn nama_many_series_one_param_time_major_into_pinned_host_f32(
        &self,
        prices_tm: &[f32],
        num_series: usize,
        series_len: usize,
        params: &NamaParams,
    ) -> Result<LockedBuffer<f32>, CudaNamaError> {
        let arr = self
            .nama_many_series_one_param_time_major_dev(prices_tm, num_series, series_len, params)?;
        let mut pinned = unsafe { LockedBuffer::<f32>::uninitialized(num_series * series_len) }
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        unsafe {
            arr.buf
                .async_copy_to(&mut pinned, &self.stream)
                .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaNamaError::Cuda(e.to_string()))?;
        Ok(pinned)
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        nama_benches,
        CudaNama,
        crate::indicators::moving_averages::nama::NamaBatchRange,
        crate::indicators::moving_averages::nama::NamaParams,
        nama_batch_dev,
        nama_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::nama::NamaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::nama::NamaParams { period: Some(64) },
        "nama",
        "nama"
    );
    pub use nama_benches::bench_profiles;
}
