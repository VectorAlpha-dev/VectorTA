//! CUDA scaffolding for the Hull Moving Average (HMA).
//!
//! Aligns with the ALMA CUDA wrapper patterns:
//! - Stream: NON_BLOCKING; PTX JIT with DetermineTargetFromContext and O2→fallback.
//! - Policy enums and introspection (batch and many-series), with BENCH_DEBUG logging.
//! - VRAM estimation and early failure if insufficient; grid.y chunking (<= 65_535).
//! - Public device entry points mirror ALMA naming and return `DeviceArrayF32`.
//!
//! Kernels implement a sequential (recurrence-style) update per-thread; we keep
//! the simple 1D kernels and rely on host-side chunking for large sweeps.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::hma::{HmaBatchRange, HmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

// Reuse ALMA/CWMA policy enums for a consistent public API.
use super::cwma_wrapper::{BatchKernelPolicy, BatchThreadsPerOutput, ManySeriesKernelPolicy};

#[derive(Debug)]
pub enum CudaHmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaHmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaHmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaHmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaHmaError {}

/// Selected kernel variants (introspection only)
#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaHma {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaHmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaHmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaHmaPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

impl CudaHma {
    #[inline]
    fn ring_in_shared() -> bool {
        true
    }
    #[inline]
    fn assume_out_prefilled() -> bool {
        true
    }
    pub fn new(device_id: usize) -> Result<Self, CudaHmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/hma_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaHmaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaHmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaHmaPolicy) -> Result<Self, CudaHmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaHmaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaHmaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaHmaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))
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
                    eprintln!("[DEBUG] HMA batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaHma)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] HMA many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaHma)).debug_many_logged = true;
                }
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

    fn expand_range(range: &HmaBatchRange) -> Vec<HmaParams> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return vec![HmaParams {
                period: Some(start),
            }];
        }
        (start..=end)
            .step_by(step)
            .map(|period| HmaParams {
                period: Some(period),
            })
            .collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &HmaBatchRange,
    ) -> Result<(Vec<HmaParams>, usize, usize, usize), CudaHmaError> {
        if data_f32.is_empty() {
            return Err(CudaHmaError::InvalidInput("empty data".into()));
        }

        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaHmaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_range(sweep);
        if combos.is_empty() {
            return Err(CudaHmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let tail_len = len - first_valid;
        let mut max_sqrt_len = 0usize;
        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period < 2 {
                return Err(CudaHmaError::InvalidInput(
                    "period must be at least 2".into(),
                ));
            }
            if period > len {
                return Err(CudaHmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            let half = period / 2;
            if half == 0 {
                return Err(CudaHmaError::InvalidInput(format!(
                    "period {} results in zero half-window",
                    period
                )));
            }
            let sqrt_len = ((period as f64).sqrt().floor() as usize).max(1);
            if tail_len < period + sqrt_len - 1 {
                return Err(CudaHmaError::InvalidInput(format!(
                    "not enough valid data for period {} (tail = {}, need >= {})",
                    period,
                    tail_len,
                    period + sqrt_len - 1
                )));
            }
            max_sqrt_len = max_sqrt_len.max(sqrt_len);
        }

        Ok((combos, first_valid, len, max_sqrt_len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods_ptr: u64,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_sqrt_len: usize,
        d_ring_ptr: u64,
        d_out_ptr: u64,
        block_x: u32,
        shared_bytes: usize,
    ) -> Result<(), CudaHmaError> {
        let func = self
            .module
            .get_function("hma_batch_f32")
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods_ptr;
            let mut len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut max_sqrt_i = max_sqrt_len as i32;
            let mut ring_ptr = d_ring_ptr;
            let mut out_ptr = d_out_ptr;

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut max_sqrt_i as *mut _ as *mut c_void,
                &mut ring_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)
                .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[HmaParams],
        first_valid: usize,
        len: usize,
        max_sqrt_len: usize,
    ) -> Result<DeviceArrayF32, CudaHmaError> {
        // VRAM estimate: prices + periods + ring + output (allocate ring to be robust
        // to mismatched build macros; kernel may ignore it when using shared memory)
        let n = combos.len();
        let prices_bytes = len * std::mem::size_of::<f32>();
        let periods_bytes = n * std::mem::size_of::<i32>();
        let ring_bytes = n * max_sqrt_len * std::mem::size_of::<f32>();
        let out_bytes = n * len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + ring_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64MB
        if !Self::will_fit(required, headroom) {
            return Err(CudaHmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        // Async HtoD
        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&periods, &self.stream) }
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        let elems = n * len;
        let mut d_ring: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n * max_sqrt_len) }
                .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(elems) }
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        // Device NaN prefill if kernel assumes prefilled output
        if Self::assume_out_prefilled() {
            unsafe {
                let ptr: cu::CUdeviceptr = d_out.as_device_ptr().as_raw();
                let n32: usize = d_out.len();
                let st: cu::CUstream = self.stream.as_inner();
                let res = cu::cuMemsetD32Async(ptr, 0x7FFF_FFFFu32, n32, st);
                if res != cu::CUresult::CUDA_SUCCESS {
                    return Err(CudaHmaError::Cuda(format!(
                        "cuMemsetD32Async failed: {:?}",
                        res
                    )));
                }
            }
        }

        // Policy: currently only Plain batch kernel; allow user override of block_x
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            _ => match std::env::var("HMA_BLOCK_X")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
            {
                Some(v) if v == 128 || v == 256 || v == 512 => v,
                _ => 256,
            },
        };
        unsafe {
            let this = self as *const _ as *mut CudaHma;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        // Single grid-stride launch
        let periods_ptr = unsafe { d_periods.as_device_ptr().as_raw() };
        let ring_ptr = unsafe { d_ring.as_device_ptr().as_raw() };
        let out_ptr = unsafe { d_out.as_device_ptr().as_raw() };
        let shared_bytes: usize = if Self::ring_in_shared() {
            max_sqrt_len * (block_x as usize) * std::mem::size_of::<f32>()
        } else {
            0
        };

        self.launch_batch_kernel(
            &d_prices,
            periods_ptr,
            len,
            n,
            first_valid,
            max_sqrt_len,
            ring_ptr,
            out_ptr,
            block_x,
            shared_bytes,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n,
            cols: len,
        })
    }

    pub fn hma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &HmaBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<HmaParams>), CudaHmaError> {
        let (combos, first_valid, len, max_sqrt_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len, max_sqrt_len)?;
        Ok((dev, combos))
    }

    pub fn hma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &HmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<HmaParams>), CudaHmaError> {
        let (combos, first_valid, len, max_sqrt_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaHmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len, max_sqrt_len)?;
        // Async D2H with optional pinned staging for large outputs
        let n_elems = out.len();
        if n_elems >= (1 << 20) {
            let mut pinned: LockedBuffer<f32> = unsafe {
                LockedBuffer::uninitialized(n_elems)
                    .map_err(|e| CudaHmaError::Cuda(e.to_string()))?
            };
            unsafe {
                dev.buf
                    .async_copy_to(pinned.as_mut_slice(), &self.stream)
                    .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
            }
            self.stream
                .synchronize()
                .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
            out.copy_from_slice(pinned.as_slice());
        } else {
            unsafe {
                dev.buf
                    .async_copy_to(out, &self.stream)
                    .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
            }
            self.stream
                .synchronize()
                .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        }
        Ok((combos.len(), len, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HmaParams,
    ) -> Result<(Vec<i32>, usize, usize), CudaHmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaHmaError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaHmaError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }

        let period = params.period.unwrap_or(0);
        if period < 2 {
            return Err(CudaHmaError::InvalidInput(
                "period must be at least 2".into(),
            ));
        }
        if period > rows {
            return Err(CudaHmaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }
        let half = period / 2;
        if half == 0 {
            return Err(CudaHmaError::InvalidInput(format!(
                "period {} results in zero half-window",
                period
            )));
        }
        let sqrt_len = ((period as f64).sqrt().floor() as usize).max(1);

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaHmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < period + sqrt_len - 1 {
                return Err(CudaHmaError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {}, need >= {})",
                    series,
                    period,
                    rows - fv,
                    period + sqrt_len - 1
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period, sqrt_len))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        num_series: usize,
        series_len: usize,
        period: usize,
        max_sqrt_len: usize,
        d_ring_ptr: u64,
        d_out_ptr: u64,
        block_x: u32,
        shared_bytes: usize,
    ) -> Result<(), CudaHmaError> {
        let func = self
            .module
            .get_function("hma_many_series_one_param_f32")
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = num_series as i32;
            let mut rows_i = series_len as i32;
            let mut period_i = period as i32;
            let mut max_sqrt_i = max_sqrt_len as i32;
            let mut ring_ptr = d_ring_ptr;
            let mut out_ptr = d_out_ptr;

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut max_sqrt_i as *mut _ as *mut c_void,
                &mut ring_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, shared_bytes as u32, args)
                .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
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
        sqrt_len: usize,
    ) -> Result<DeviceArrayF32, CudaHmaError> {
        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(first_valids, &self.stream) }
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        let elems = cols * rows;
        let mut d_ring: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * sqrt_len) }
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        if Self::assume_out_prefilled() {
            unsafe {
                let ptr: cu::CUdeviceptr = d_out.as_device_ptr().as_raw();
                let n32: usize = d_out.len();
                let st: cu::CUstream = self.stream.as_inner();
                let res = cu::cuMemsetD32Async(ptr, 0x7FFF_FFFFu32, n32, st);
                if res != cu::CUresult::CUDA_SUCCESS {
                    return Err(CudaHmaError::Cuda(format!(
                        "cuMemsetD32Async failed: {:?}",
                        res
                    )));
                }
            }
        }

        // Policy: currently only 1D many-series kernel; allow override of block_x
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            _ => match std::env::var("HMA_MS_BLOCK_X")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
            {
                Some(v) if v == 128 || v == 256 || v == 512 => v,
                _ => 256,
            },
        };
        unsafe {
            let this = self as *const _ as *mut CudaHma;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        let ring_ptr = unsafe { d_ring.as_device_ptr().as_raw() };
        let out_ptr = unsafe { d_out.as_device_ptr().as_raw() };
        let shared_bytes: usize = if Self::ring_in_shared() {
            sqrt_len * (block_x as usize) * std::mem::size_of::<f32>()
        } else {
            0
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            cols,
            rows,
            period,
            sqrt_len,
            ring_ptr,
            out_ptr,
            block_x,
            shared_bytes,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn hma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HmaParams,
    ) -> Result<DeviceArrayF32, CudaHmaError> {
        let (first_valids, period, sqrt_len) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period, sqrt_len)
    }

    pub fn hma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaHmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaHmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }
        let (first_valids, period, sqrt_len) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev =
            self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period, sqrt_len)?;
        // Async D2H with optional pinned staging
        let n_elems = out_tm.len();
        if n_elems >= (1 << 20) {
            let mut pinned: LockedBuffer<f32> = unsafe {
                LockedBuffer::uninitialized(n_elems)
                    .map_err(|e| CudaHmaError::Cuda(e.to_string()))?
            };
            unsafe {
                dev.buf
                    .async_copy_to(pinned.as_mut_slice(), &self.stream)
                    .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
            }
            self.stream
                .synchronize()
                .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
            out_tm.copy_from_slice(pinned.as_slice());
        } else {
            unsafe {
                dev.buf
                    .async_copy_to(out_tm, &self.stream)
                    .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
            }
            self.stream
                .synchronize()
                .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        hma_benches,
        CudaHma,
        crate::indicators::moving_averages::hma::HmaBatchRange,
        crate::indicators::moving_averages::hma::HmaParams,
        hma_batch_dev,
        hma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::hma::HmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::hma::HmaParams { period: Some(64) },
        "hma",
        "hma"
    );
    pub use hma_benches::bench_profiles;
}
