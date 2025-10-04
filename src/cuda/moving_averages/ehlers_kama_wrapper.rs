//! CUDA wrapper for Ehlers KAMA kernels.
//!
//! Aligns with the ALMA "gold standard" control plane: selection policies,
//! introspection, optional debug logging via BENCH_DEBUG=1, VRAM checks using
//! mem_get_info, deterministic synchronization after launches, and a 2D tiled
//! many-series kernel in addition to the plain 1D variant. For this IIR MA,
//! time is sequential per-thread; we parallelize across (series, params).

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::ehlers_kama::{EhlersKamaBatchRange, EhlersKamaParams};
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

// -------- Policies and introspection (API parity with ALMA) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaEhlersKamaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for BatchKernelPolicy { fn default() -> Self { BatchKernelPolicy::Auto } }
impl Default for ManySeriesKernelPolicy { fn default() -> Self { ManySeriesKernelPolicy::Auto } }

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Debug)]
pub enum CudaEhlersKamaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaEhlersKamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEhlersKamaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEhlersKamaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaEhlersKamaError {}

/// CUDA Ehlers KAMA launcher and VRAM handle utilities.
///
/// Fields and methods mirror the ALMA wrapper: policy selection, kernel
/// introspection, BENCH_DEBUG logging, VRAM checks, and deterministic
/// stream synchronization after launches.
pub struct CudaEhlersKama {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaEhlersKamaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaEhlersKama {
    /// Create a new `CudaEhlersKama` on `device_id` and load the PTX module.
    pub fn new(device_id: usize) -> Result<Self, CudaEhlersKamaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ehlers_kama_kernel.ptx"));
        // Match ALMA: prefer context-determined target and O2, with fallbacks
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
                    Module::from_ptx(ptx, &[])
                        .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaEhlersKamaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create using an explicit selection policy.
    pub fn new_with_policy(
        device_id: usize,
        policy: CudaEhlersKamaPolicy,
    ) -> Result<Self, CudaEhlersKamaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaEhlersKamaPolicy) { self.policy = policy; }
    pub fn policy(&self) -> &CudaEhlersKamaPolicy { &self.policy }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    /// Synchronize the stream for deterministic timing.
    pub fn synchronize(&self) -> Result<(), CudaEhlersKamaError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] EhlersKama batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaEhlersKama)).debug_batch_logged = true; }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] EhlersKama many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaEhlersKama)).debug_many_logged = true; }
            }
        }
    }

    // ---------- VRAM checks ----------
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else { true }
    }

    fn expand_grid(range: &EhlersKamaBatchRange) -> Vec<EhlersKamaParams> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return vec![EhlersKamaParams {
                period: Some(start),
            }];
        }
        let mut params = Vec::new();
        let step_sz = if step == 0 { 1 } else { step };
        let mut value = start;
        loop {
            if value > end {
                break;
            }
            params.push(EhlersKamaParams {
                period: Some(value),
            });
            match value.checked_add(step_sz) {
                Some(next) => {
                    if next > end {
                        break;
                    }
                    value = next;
                }
                None => break,
            }
        }
        if params.is_empty() {
            params.push(EhlersKamaParams {
                period: Some(start),
            });
        }
        params
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &EhlersKamaBatchRange,
    ) -> Result<(Vec<EhlersKamaParams>, usize, usize), CudaEhlersKamaError> {
        if data_f32.is_empty() {
            return Err(CudaEhlersKamaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaEhlersKamaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaEhlersKamaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = data_f32.len();
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaEhlersKamaError::InvalidInput(
                    "period must be greater than zero".into(),
                ));
            }
            if period > len {
                return Err(CudaEhlersKamaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaEhlersKamaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersKamaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaEhlersKamaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaEhlersKamaError::InvalidInput(
                "series_len or n_combos exceed i32::MAX".into(),
            ));
        }
        if first_valid > i32::MAX as usize {
            return Err(CudaEhlersKamaError::InvalidInput(
                "first_valid exceeds i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("ehlers_kama_batch_f32")
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        // Follow ALMA: chunk very large combo counts to avoid brittle grid configurations.
        // We slice the periods/out pointers so the kernel does not need an extra offset.
        const MAX_CHUNK: usize = 65_535;
        let mut launched = 0usize;
        while launched < n_combos {
            let todo = (n_combos - launched).min(MAX_CHUNK);
            let grid: GridSize = (todo as u32, 1, 1).into();
            let block: BlockSize = (
                match self.policy.batch {
                    BatchKernelPolicy::Auto => 1,
                    BatchKernelPolicy::Plain { block_x } => block_x,
                },
                1,
                1,
            )
                .into();
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut periods_ptr = d_periods.as_device_ptr().as_raw().wrapping_add((launched) as u64 * std::mem::size_of::<i32>() as u64);
                let mut first_valid_i = first_valid as i32;
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = todo as i32;
                let mut out_ptr = d_out
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((launched * series_len) as u64 * std::mem::size_of::<f32>() as u64);
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut periods_ptr as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
            }
            launched += todo;
        }

        // record selection for introspection/debug
        unsafe {
            (*(self as *const _ as *mut CudaEhlersKama)).last_batch = Some(BatchKernelSelected::Plain {
                block_x: match self.policy.batch { BatchKernelPolicy::Auto => 1, BatchKernelPolicy::Plain { block_x } => block_x },
            });
        }
        self.maybe_log_batch_debug();

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[EhlersKamaParams],
        first_valid: usize,
        series_len: usize,
    ) -> Result<DeviceArrayF32, CudaEhlersKamaError> {
        let n_combos = combos.len();
        let mut periods_i32 = Vec::with_capacity(n_combos);
        for prm in combos {
            let period = prm.period.unwrap();
            if period > i32::MAX as usize {
                return Err(CudaEhlersKamaError::InvalidInput(
                    "period exceeds i32::MAX".into(),
                ));
            }
            periods_i32.push(period as i32);
        }

        // VRAM check
        let required = data_f32.len() * std::mem::size_of::<f32>()
            + combos.len() * std::mem::size_of::<i32>()
            + combos.len() * series_len * std::mem::size_of::<f32>();
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaEhlersKamaError::Cuda("insufficient device memory".into()));
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        // Pre-fill outputs with NaN to guarantee warm-up semantics even if a corner
        // case causes a block to early-return.
        if let Ok(func) = self.module.get_function("ehlers_kama_fill_nan_vec_f32") {
            let total = (n_combos * series_len) as u32;
            let block_x: u32 = 256;
            let grid_x: u32 = ((total + block_x - 1) / block_x).max(1);
            unsafe {
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let mut len_i = (n_combos * series_len) as i32;
                let args: &mut [*mut c_void] = &mut [
                    &mut out_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
                    .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
            }
        }

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            first_valid,
            series_len,
            n_combos,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    /// Batch variant with pre-uploaded prices/periods buffers.
    ///
    /// - `first_valid`: index of first non-NaN input; warm-up = first_valid + period - 1
    /// - outputs for t < warm-up are NaN in all kernels
    pub fn ehlers_kama_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        first_valid: usize,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersKamaError> {
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            first_valid,
            series_len,
            n_combos,
            d_out,
        )
    }

    /// Launch batch using pre-uploaded prices; allocates results and returns handle.
    /// Convenience: batch launch using a device-resident prices buffer.
    pub fn ehlers_kama_batch_from_device_prices(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        combos: &[EhlersKamaParams],
    ) -> Result<DeviceArrayF32, CudaEhlersKamaError> {
        if series_len == 0 { return Err(CudaEhlersKamaError::InvalidInput("series_len is zero".into())); }
        let n_combos = combos.len();
        if n_combos == 0 { return Err(CudaEhlersKamaError::InvalidInput("no parameter combinations".into())); }
        let mut periods_i32 = Vec::with_capacity(n_combos);
        for prm in combos { periods_i32.push(prm.period.unwrap_or(0) as i32); }
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        self.launch_batch_kernel(d_prices, &d_periods, first_valid, series_len, n_combos, &mut d_out)?;
        self.stream.synchronize().map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out, rows: n_combos, cols: series_len })
    }

    pub fn ehlers_kama_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &EhlersKamaBatchRange,
    ) -> Result<DeviceArrayF32, CudaEhlersKamaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, first_valid, len)
    }

    pub fn ehlers_kama_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &EhlersKamaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<EhlersKamaParams>), CudaEhlersKamaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaEhlersKamaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhlersKamaParams,
    ) -> Result<(Vec<i32>, usize), CudaEhlersKamaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaEhlersKamaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaEhlersKamaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaEhlersKamaError::InvalidInput(
                "period must be greater than zero".into(),
            ));
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
            let fv = fv.ok_or_else(|| {
                CudaEhlersKamaError::InvalidInput(format!("series {} all NaN", series))
            })?;
            if rows - fv < period {
                return Err(CudaEhlersKamaError::InvalidInput(format!(
                    "series {} not enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv
                )));
            }
            if fv > i32::MAX as usize {
                return Err(CudaEhlersKamaError::InvalidInput(
                    "first_valid exceeds i32::MAX".into(),
                ));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersKamaError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaEhlersKamaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaEhlersKamaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        // Select kernel according to policy and availability
        let (selected, launch) = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => {
                // Prefer 2D whenever available; adapt tile to small N to avoid degenerate
                // large tiles on very few series.
                let has_2d = self.module.get_function("ehlers_kama_multi_series_one_param_2d_f32").is_ok();
                if has_2d {
                    let choose = |n: usize| -> (u32, u32) {
                        if n <= 1 { return (1, 1); }
                        if n <= 2 { return (2, 1); }
                        if n <= 4 { return (4, 1); }
                        if n <= 8 { return (8, 1); }
                        if n <= 16 { return (16, 1); }
                        if n <= 32 { return (32, 1); }
                        if n <= 64 { return (64, 1); }
                        (64, 2)
                    };
                    let (tx, ty) = choose(num_series);
                    (ManySeriesKernelSelected::Tiled2D { tx, ty }, ManySeriesKernelPolicy::Tiled2D { tx, ty })
                } else {
                    (ManySeriesKernelSelected::OneD { block_x: 64 }, ManySeriesKernelPolicy::OneD { block_x: 64 })
                }
            }
            ManySeriesKernelPolicy::OneD { block_x } => (
                ManySeriesKernelSelected::OneD { block_x },
                ManySeriesKernelPolicy::OneD { block_x },
            ),
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => (
                ManySeriesKernelSelected::Tiled2D { tx, ty },
                ManySeriesKernelPolicy::Tiled2D { tx, ty },
            ),
        };

        match launch {
            ManySeriesKernelPolicy::OneD { block_x } => {
                // Reuse the 2D kernel with a (tx=block_x, ty=1) configuration,
                // so OneD maps to a contiguous tile of series per block.
                let func = self
                    .module
                    .get_function("ehlers_kama_multi_series_one_param_2d_f32")
                    .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
                let tx = block_x.max(1);
                let tile = tx; // ty=1
                let blocks = ((num_series as u32 + tile - 1) / tile).max(1);
                let grid: GridSize = (blocks, 1, 1).into();
                let block: BlockSize = (tx, 1, 1).into();
                unsafe {
                    let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
                    let mut period_i = period as i32;
                    let mut num_series_i = num_series as i32;
                    let mut series_len_i = series_len as i32;
                    let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
                    let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut period_i as *mut _ as *mut c_void,
                        &mut num_series_i as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut first_valids_ptr as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream
                        .launch(&func, grid, block, 0, args)
                        .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
                }
            }
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                let func = self
                    .module
                    .get_function("ehlers_kama_multi_series_one_param_2d_f32")
                    .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
                let tile = (tx * ty).max(1);
                let blocks = ((num_series as u32 + tile - 1) / tile).max(1);
                let grid: GridSize = (blocks, 1, 1).into();
                let block: BlockSize = (tx, ty, 1).into();
                unsafe {
                    let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
                    let mut period_i = period as i32;
                    let mut num_series_i = num_series as i32;
                    let mut series_len_i = series_len as i32;
                    let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
                    let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut period_i as *mut _ as *mut c_void,
                        &mut num_series_i as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut first_valids_ptr as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    self.stream
                        .launch(&func, grid, block, 0, args)
                        .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
                }
            }
            _ => unreachable!(),
        }

        unsafe { (*(self as *const _ as *mut CudaEhlersKama)).last_many = Some(selected); }
        self.maybe_log_many_debug();

        Ok(())
    }

    pub fn ehlers_kama_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhlersKamaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaEhlersKamaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out_tm,
        )
    }

    /// Many-series × one-parameter, time-major layout (rows=time, cols=series).
    /// Returns a VRAM handle; call `device_ptr()` via `DeviceArrayF32` on the ALMA
    /// handle if you need a raw pointer in Python or other FFI layers.
    pub fn ehlers_kama_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhlersKamaParams,
    ) -> Result<DeviceArrayF32, CudaEhlersKamaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // VRAM check
        let required = cols * rows * std::mem::size_of::<f32>()
            + cols * std::mem::size_of::<i32>()
            + cols * rows * std::mem::size_of::<f32>();
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaEhlersKamaError::Cuda("insufficient device memory".into()));
        }

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }
                .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        // Pre-fill outputs with NaN to ensure warm-up region is NaN across all variants
        if let Ok(func) = self.module.get_function("ehlers_kama_fill_nan_vec_f32") {
            let total = (cols * rows) as u32;
            let block_x: u32 = 256;
            let grid_x: u32 = ((total + block_x - 1) / block_x).max(1);
            unsafe {
                let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                let mut len_i = (cols * rows) as i32;
                let args: &mut [*mut c_void] = &mut [
                    &mut out_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
                    .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
            }
        }

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;

        // Enforce warm-up NaN boundaries explicitly.
        if let Ok(func) = self
            .module
            .get_function("ehlers_kama_enforce_warm_nan_tm_f32")
        {
            let block_x: u32 = 128;
            let grid_x: u32 = ((cols as u32 + block_x - 1) / block_x).max(1);
            unsafe {
                let mut period_i = period as i32;
                let mut num_series_i = cols as i32;
                let mut series_len_i = rows as i32;
                let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
                let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut period_i as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut first_valids_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
                    .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
            }
        }

        // Guard-rail: explicitly set t=0 to NaN when warm>0 for each series.
        if let Ok(func) = self
            .module
            .get_function("ehlers_kama_fix_first_row_nan_tm_f32")
        {
            let block_x: u32 = 128;
            let grid_x: u32 = ((cols as u32 + block_x - 1) / block_x).max(1);
            unsafe {
                let mut period_i = period as i32;
                let mut num_series_i = cols as i32;
                let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
                let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut period_i as *mut _ as *mut c_void,
                    &mut num_series_i as *mut _ as *mut c_void,
                    &mut first_valids_ptr as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
                    .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
            }
        }
        // Guard rails: as an ultimate safety, patch warm-up NaNs on host and
        // write back to device unless explicitly disabled.
        let guard_on = std::env::var("KAMA_GUARD_RAILS").ok().map(|v| v != "0" && v.to_lowercase() != "false").unwrap_or(true);
        if guard_on {
            // Copy to pinned host
            let mut pinned: LockedBuffer<f32> = unsafe {
                LockedBuffer::uninitialized(cols * rows)
                    .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?
            };
            unsafe {
                d_out_tm
                    .async_copy_to(pinned.as_mut_slice(), &self.stream)
                    .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
            }
            self.stream
                .synchronize()
                .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
            // Patch warm-up to NaN on host
            let nan = f32::NAN;
            for s in 0..cols {
                let fv = first_valids[s] as usize;
                let warm = fv + period - 1;
                let warm_clamped = warm.min(rows);
                for t in 0..warm_clamped { pinned[t * cols + s] = nan; }
            }
            // Copy updated host buffer back to device
            unsafe {
                d_out_tm
                    .async_copy_from(pinned.as_slice(), &self.stream)
                    .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        if env::var("DUMP_KAMA").ok().as_deref() == Some("1") {
            let mut host = vec![0f32; cols * rows];
            d_out_tm
                .copy_to(&mut host)
                .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
            let dump_cols = cols.min(8);
            eprintln!("[DUMP] first row (t=0) first {} series: {:?}", dump_cols, &host[0..dump_cols]);
        }

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn ehlers_kama_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhlersKamaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaEhlersKamaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaEhlersKamaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }
                .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        if let Ok(func) = self.module.get_function("ehlers_kama_fill_nan_vec_f32") {
            let total = (cols * rows) as u32;
            let block_x: u32 = 256;
            let grid_x: u32 = ((total + block_x - 1) / block_x).max(1);
            unsafe {
                let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
                let mut len_i = (cols * rows) as i32;
                let args: &mut [*mut c_void] = &mut [
                    &mut out_ptr as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                ];
                self.stream
                    .launch(&func, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
                    .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
            }
        }

        self.launch_many_series_kernel(
            &d_prices_tm,
            period,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;

        // Use pinned host buffer for faster D2H, then copy into provided slice.
        let mut pinned: LockedBuffer<f32> = unsafe {
            LockedBuffer::uninitialized(cols * rows)
                .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?
        };
        unsafe {
            d_out_tm
                .async_copy_to(pinned.as_mut_slice(), &self.stream)
                .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaEhlersKamaError::Cuda(e.to_string()))?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        ehlers_kama_benches,
        CudaEhlersKama,
        crate::indicators::moving_averages::ehlers_kama::EhlersKamaBatchRange,
        crate::indicators::moving_averages::ehlers_kama::EhlersKamaParams,
        ehlers_kama_batch_dev,
        ehlers_kama_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::ehlers_kama::EhlersKamaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::ehlers_kama::EhlersKamaParams { period: Some(64) },
        "ehlers_kama",
        "ehlers_kama"
    );
    pub use ehlers_kama_benches::bench_profiles;
}
