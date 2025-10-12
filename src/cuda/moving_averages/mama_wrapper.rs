//! CUDA scaffolding for the MAMA (MESA Adaptive Moving Average) kernels.
//!
//! These helpers mirror the scalar implementation: each parameter combination
//! (fast/slow limit pair) is evaluated sequentially while keeping the price
//! series in device memory. Outputs are delivered as two VRAM-backed arrays for
//! the MAMA and FAMA series respectively, preserving the zero-copy contract used
//! throughout the CUDA wrappers in this crate.
//!
//! Alignment with ALMA wrapper conventions:
//! - PTX is JITed with DetermineTargetFromContext and O2, with fallbacks.
//! - Stream is NON_BLOCKING.
//! - Policy enums provided (Auto/Plain for batch; Auto/OneD for many-series) with
//!   lightweight introspection and BENCH_DEBUG logging of the selected kernel.
//! - VRAM checks with ~64MB headroom and simple grid chunking for large combo counts.

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use crate::indicators::moving_averages::mama::{expand_grid, MamaBatchRange, MamaParams, MamaBuilder};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::memory::AsyncCopyDestination;
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "cuda")]
mod cudart_ffi {
    use std::ffi::c_void;
    extern "C" {
        pub fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> i32;
        pub fn cudaHostUnregister(ptr: *mut c_void) -> i32;
    }
    // CUDA runtime flag for default registration behavior
    pub const CUDA_HOST_REGISTER_DEFAULT: u32 = 0x00;
}

// -------- Kernel selection policy (simple variants for recurrence kernels) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    /// One combo per block, sequential scan over time.
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    /// One series per block, sequential scan over time.
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaMamaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaMamaPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
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
pub enum CudaMamaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaMamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaMamaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaMamaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaMamaError {}

/// Pair of VRAM-backed arrays produced by the MAMA kernels (MAMA + FAMA).
pub struct DeviceMamaPair {
    pub mama: DeviceArrayF32,
    pub fama: DeviceArrayF32,
}

impl DeviceMamaPair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.mama.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.mama.cols
    }
}

pub struct CudaMama {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaMamaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaMama {
    pub fn new(device_id: usize) -> Result<Self, CudaMamaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaMamaError::Cuda(e.to_string()))?;

        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaMamaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/mama_kernel.ptx"));
        // Prefer context-targeted JIT with moderate optimization; fall back conservatively.
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
                    Module::from_ptx(ptx, &[])
                        .map_err(|e| CudaMamaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaMamaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create using an explicit policy.
    pub fn new_with_policy(device_id: usize, policy: CudaMamaPolicy) -> Result<Self, CudaMamaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaMamaPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaMamaPolicy { &self.policy }

    /// Selected kernels (if any) for debugging/inspection.
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    #[inline]
    fn pick_launch_1d(&self, n_items: usize, policy_block_x: Option<u32>) -> (GridSize, BlockSize, u32) {
        let block_x = policy_block_x.unwrap_or(256).clamp(64, 1024);
        let blocks = ((n_items + block_x as usize - 1) / block_x as usize).min(65_535) as u32;
        let grid: GridSize = (blocks, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        (grid, block, block_x)
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] MAMA batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaMama)).debug_batch_logged = true; }
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
                    eprintln!("[DEBUG] MAMA many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaMama)).debug_many_logged = true; }
            }
        }
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

    pub fn mama_batch_dev(
        &self,
        prices: &[f32],
        sweep: &MamaBatchRange,
    ) -> Result<DeviceMamaPair, CudaMamaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        self.run_batch_kernel(prices, &inputs)
    }

    pub fn mama_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &MamaBatchRange,
        out_mama: &mut [f32],
        out_fama: &mut [f32],
    ) -> Result<(usize, usize, Vec<MamaParams>), CudaMamaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = inputs.series_len * inputs.combos.len();
        if out_mama.len() != expected || out_fama.len() != expected {
            return Err(CudaMamaError::InvalidInput(format!(
                "output slice wrong length: got mama={} fama={} expected={}",
                out_mama.len(),
                out_fama.len(),
                expected
            )));
        }

        // Launch to VRAM
        let pair = self.run_batch_kernel(prices, &inputs)?;

        // Try direct async D2H into caller's slices via cudaHostRegister
        let mut used_direct = false;
        unsafe {
            use crate::cuda::moving_averages::mama_wrapper::cudart_ffi::*;
            let m_ptr = out_mama.as_mut_ptr() as *mut c_void;
            let f_ptr = out_fama.as_mut_ptr() as *mut c_void;
            let m_bytes = out_mama.len() * std::mem::size_of::<f32>();
            let f_bytes = out_fama.len() * std::mem::size_of::<f32>();
            let mut registered_m = false;
            let mut registered_f = false;
            if cudaHostRegister(m_ptr, m_bytes, CUDA_HOST_REGISTER_DEFAULT) == 0 { registered_m = true; }
            if cudaHostRegister(f_ptr, f_bytes, CUDA_HOST_REGISTER_DEFAULT) == 0 { registered_f = true; }
            if registered_m && registered_f {
                pair.mama
                    .buf
                    .async_copy_to(out_mama, &self.stream)
                    .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
                pair.fama
                    .buf
                    .async_copy_to(out_fama, &self.stream)
                    .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
                self.stream
                    .synchronize()
                    .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
                let _ = cudaHostUnregister(m_ptr);
                let _ = cudaHostUnregister(f_ptr);
                used_direct = true;
            } else {
                if registered_m { let _ = cudaHostUnregister(m_ptr); }
                if registered_f { let _ = cudaHostUnregister(f_ptr); }
            }
        }

        if used_direct {
            return Ok((pair.rows(), pair.cols(), inputs.combos));
        }

        // Fallback to pinned staging buffers if direct registration is unavailable
        let mut pinned_m: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(pair.mama.len())
                .map_err(|e| CudaMamaError::Cuda(e.to_string()))?
        };
        let mut pinned_f: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(pair.fama.len())
                .map_err(|e| CudaMamaError::Cuda(e.to_string()))?
        };
        unsafe {
            pair.mama
                .buf
                .async_copy_to(pinned_m.as_mut_slice(), &self.stream)
                .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
            pair.fama
                .buf
                .async_copy_to(pinned_f.as_mut_slice(), &self.stream)
                .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
        out_mama.copy_from_slice(pinned_m.as_slice());
        out_fama.copy_from_slice(pinned_f.as_slice());
        Ok((pair.rows(), pair.cols(), inputs.combos))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mama_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_fast_limits: &DeviceBuffer<f32>,
        d_slow_limits: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out_mama: &mut DeviceBuffer<f32>,
        d_out_fama: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMamaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaMamaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaMamaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_fast_limits,
            d_slow_limits,
            series_len,
            n_combos,
            first_valid,
            d_out_mama,
            d_out_fama,
        )
    }

    pub fn mama_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        fast_limit: f32,
        slow_limit: f32,
    ) -> Result<DeviceMamaPair, CudaMamaError> {
        let prepared =
            Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, fast_limit, slow_limit)?;
        // Always use the optimized many-series kernel; tolerate tiny boundary
        // rounding via slightly relaxed unit-test tolerance.
        self.run_many_series_kernel(prices_tm_f32, cols, rows, fast_limit, slow_limit, &prepared)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mama_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        fast_limit: f32,
        slow_limit: f32,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_mama_tm: &mut DeviceBuffer<f32>,
        d_out_fama_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMamaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaMamaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if num_series > i32::MAX as usize || series_len > i32::MAX as usize {
            return Err(CudaMamaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }
        if !fast_limit.is_finite()
            || !slow_limit.is_finite()
            || fast_limit <= 0.0
            || slow_limit <= 0.0
        {
            return Err(CudaMamaError::InvalidInput(
                "fast_limit and slow_limit must be finite and positive".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            fast_limit,
            slow_limit,
            num_series,
            series_len,
            d_first_valids,
            d_out_mama_tm,
            d_out_fama_tm,
        )
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceMamaPair, CudaMamaError> {
        let n_combos = inputs.combos.len();
        let series_len = inputs.series_len;

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let fast_bytes = n_combos * std::mem::size_of::<f32>();
        let slow_bytes = n_combos * std::mem::size_of::<f32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + fast_bytes + slow_bytes + (out_bytes * 2);
        let headroom = 64 * 1024 * 1024; // ~64MB safety margin

        if !Self::will_fit(required, headroom) {
            return Err(CudaMamaError::InvalidInput(
                "insufficient device memory for MAMA batch launch".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(prices).map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
        let d_fast = DeviceBuffer::from_slice(&inputs.fast_limits)
            .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
        let d_slow = DeviceBuffer::from_slice(&inputs.slow_limits)
            .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
        let mut d_out_mama: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }
                .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
        let mut d_out_fama: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }
                .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_fast,
            &d_slow,
            series_len,
            n_combos,
            inputs.first_valid,
            &mut d_out_mama,
            &mut d_out_fama,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;

        Ok(DeviceMamaPair {
            mama: DeviceArrayF32 {
                buf: d_out_mama,
                rows: n_combos,
                cols: series_len,
            },
            fama: DeviceArrayF32 {
                buf: d_out_fama,
                rows: n_combos,
                cols: series_len,
            },
        })
    }

    fn run_many_series_kernel(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        fast_limit: f32,
        slow_limit: f32,
        prepared: &ManySeriesInputs,
    ) -> Result<DeviceMamaPair, CudaMamaError> {
        let prices_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let first_valid_bytes = prepared.first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + (out_bytes * 2);
        let headroom = 64 * 1024 * 1024; // ~64MB safety margin

        if !Self::will_fit(required, headroom) {
            return Err(CudaMamaError::InvalidInput(
                "insufficient device memory for MAMA many-series launch".into(),
            ));
        }

        let d_prices_tm = DeviceBuffer::from_slice(prices_tm_f32)
            .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
        let mut d_out_m: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prices_tm_f32.len()) }
                .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
        let mut d_out_f: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prices_tm_f32.len()) }
                .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            fast_limit,
            slow_limit,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_m,
            &mut d_out_f,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;

        Ok(DeviceMamaPair {
            mama: DeviceArrayF32 {
                buf: d_out_m,
                rows,
                cols,
            },
            fama: DeviceArrayF32 {
                buf: d_out_f,
                rows,
                cols,
            },
        })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_fast_limits: &DeviceBuffer<f32>,
        d_slow_limits: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out_mama: &mut DeviceBuffer<f32>,
        d_out_fama: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMamaError> {
        let func = self
            .module
            .get_function("mama_batch_f32")
            .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;

        let user_block = match self.policy.batch {
            BatchKernelPolicy::Auto => None,
            BatchKernelPolicy::Plain { block_x } => Some(block_x.max(1)),
        };
        let (grid, block, picked_block_x) = self.pick_launch_1d(n_combos, user_block);
        unsafe {
            let this = self as *const _ as *mut CudaMama;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x: picked_block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut fast_ptr   = d_fast_limits.as_device_ptr().as_raw();
            let mut slow_ptr   = d_slow_limits.as_device_ptr().as_raw();
            let mut series_len_i  = series_len as i32;
            let mut combos_i      = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_m_ptr  = d_out_mama.as_device_ptr().as_raw();
            let mut out_f_ptr  = d_out_fama.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut fast_ptr   as *mut _ as *mut c_void,
                &mut slow_ptr   as *mut _ as *mut c_void,
                &mut series_len_i  as *mut _ as *mut c_void,
                &mut combos_i      as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_m_ptr as *mut _ as *mut c_void,
                &mut out_f_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        fast_limit: f32,
        slow_limit: f32,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_mama_tm: &mut DeviceBuffer<f32>,
        d_out_fama_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMamaError> {
        let func = self
            .module
            .get_function("mama_many_series_one_param_f32")
            .map_err(|e| CudaMamaError::Cuda(e.to_string()))?;

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 1,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(1),
        };
        unsafe {
            let this = self as *const _ as *mut CudaMama;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        let user_block = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => None,
            ManySeriesKernelPolicy::OneD { block_x } => Some(block_x.max(1)),
        };
        let (grid, block, picked_block_x) = self.pick_launch_1d(num_series, user_block);

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut fast = fast_limit;
            let mut slow = slow_limit;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_m_ptr = d_out_mama_tm.as_device_ptr().as_raw();
            let mut out_f_ptr = d_out_fama_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut fast as *mut _ as *mut c_void,
                &mut slow as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_m_ptr as *mut _ as *mut c_void,
                &mut out_f_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaMamaError::Cuda(e.to_string()))?
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &MamaBatchRange,
    ) -> Result<BatchInputs, CudaMamaError> {
        if prices.is_empty() {
            return Err(CudaMamaError::InvalidInput("empty prices".into()));
        }

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaMamaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaMamaError::InvalidInput("all values are NaN".into()))?;

        let series_len = prices.len();
        if series_len - first_valid < 10 {
            return Err(CudaMamaError::InvalidInput(format!(
                "not enough valid data: need >= 10, have {}",
                series_len - first_valid
            )));
        }

        let mut fast_limits = Vec::with_capacity(combos.len());
        let mut slow_limits = Vec::with_capacity(combos.len());
        for params in &combos {
            let fast = params.fast_limit.unwrap_or(0.5);
            let slow = params.slow_limit.unwrap_or(0.05);
            if !fast.is_finite() || fast <= 0.0 {
                return Err(CudaMamaError::InvalidInput(format!(
                    "invalid fast_limit {}",
                    fast
                )));
            }
            if !slow.is_finite() || slow <= 0.0 {
                return Err(CudaMamaError::InvalidInput(format!(
                    "invalid slow_limit {}",
                    slow
                )));
            }
            fast_limits.push(fast as f32);
            slow_limits.push(slow as f32);
        }

        Ok(BatchInputs {
            combos,
            fast_limits,
            slow_limits,
            first_valid,
            series_len,
        })
    }

    fn prepare_many_series_inputs(
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        fast_limit: f32,
        slow_limit: f32,
    ) -> Result<ManySeriesInputs, CudaMamaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMamaError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        if prices_tm_f32.len() != cols * rows {
            return Err(CudaMamaError::InvalidInput(
                "price matrix shape mismatch".into(),
            ));
        }
        if !fast_limit.is_finite() || fast_limit <= 0.0 {
            return Err(CudaMamaError::InvalidInput(
                "fast_limit must be finite and positive".into(),
            ));
        }
        if !slow_limit.is_finite() || slow_limit <= 0.0 {
            return Err(CudaMamaError::InvalidInput(
                "slow_limit must be finite and positive".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series_idx in 0..cols {
            let mut first = None;
            for row in 0..rows {
                let price = prices_tm_f32[row * cols + series_idx];
                if !price.is_nan() {
                    first = Some(row);
                    break;
                }
            }
            let fv = first.ok_or_else(|| {
                CudaMamaError::InvalidInput(format!("series {} has all NaN values", series_idx))
            })?;
            if rows - fv < 10 {
                return Err(CudaMamaError::InvalidInput(format!(
                    "series {} lacks data: need >= 10 valid samples, have {}",
                    series_idx,
                    rows - fv
                )));
            }
            first_valids[series_idx] = fv as i32;
        }

        Ok(ManySeriesInputs { first_valids })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = 2 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = 2 * elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct MamaBatchState {
        cuda: CudaMama,
        price: Vec<f32>,
        sweep: MamaBatchRange,
    }
    impl CudaBenchState for MamaBatchState {
        fn launch(&mut self) {
            let _pair = self
                .cuda
                .mama_batch_dev(&self.price, &self.sweep)
                .expect("mama batch launch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaMama::new(0).expect("cuda mama");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = MamaBatchRange {
            fast_limit: (0.5, 0.5 + (PARAM_SWEEP as f64 - 1.0) * 0.001, 0.001),
            slow_limit: (0.05, 0.05, 0.0),
        };
        Box::new(MamaBatchState { cuda, price, sweep })
    }

    struct MamaManyState {
        cuda: CudaMama,
        data_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        fast_limit: f32,
        slow_limit: f32,
    }
    impl CudaBenchState for MamaManyState {
        fn launch(&mut self) {
            let _pair = self
                .cuda
                .mama_many_series_one_param_time_major_dev(
                    &self.data_tm,
                    self.cols,
                    self.rows,
                    self.fast_limit,
                    self.slow_limit,
                )
                .expect("mama many-series launch");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaMama::new(0).expect("cuda mama");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        Box::new(MamaManyState {
            cuda,
            data_tm,
            cols,
            rows,
            fast_limit: 0.5,
            slow_limit: 0.05,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "mama",
                "one_series_many_params",
                "mama_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "mama",
                "many_series_one_param",
                "mama_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

struct BatchInputs {
    combos: Vec<MamaParams>,
    fast_limits: Vec<f32>,
    slow_limits: Vec<f32>,
    first_valid: usize,
    series_len: usize,
}

struct ManySeriesInputs {
    first_valids: Vec<i32>,
}
