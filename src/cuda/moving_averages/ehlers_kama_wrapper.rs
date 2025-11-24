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
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::AsyncCopyDestination;
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;
use std::sync::Arc;

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
    Tiled2D { tx: u32, ty: u32 },
}

#[derive(Debug, Error)]
pub enum CudaEhlersKamaError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("insufficient device memory: required={required}B free={free}B headroom={headroom}B")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("arithmetic overflow computing sizes/bytes")]
    SizeOverflow,
    #[error("not implemented")]
    NotImplemented,
    #[error("device mismatch: buf on {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("invalid range: start={start} end={end} step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
}

/// CUDA Ehlers KAMA launcher and VRAM handle utilities.
///
/// Fields and methods mirror the ALMA wrapper: policy selection, kernel
/// introspection, BENCH_DEBUG logging, VRAM checks, and deterministic
/// stream synchronization after launches.
pub struct CudaEhlersKama {
    module: Module,
    stream: Stream,
    ctx: Arc<Context>,
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
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

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
                    Module::from_ptx(ptx, &[])?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            ctx: context,
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
    pub fn set_policy(&mut self, policy: CudaEhlersKamaPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaEhlersKamaPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { Arc::clone(&self.ctx) }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }

    /// Synchronize the stream for deterministic timing.
    pub fn synchronize(&self) -> Result<(), CudaEhlersKamaError> {
        self.stream.synchronize()?;
        Ok(())
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] EhlersKama batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaEhlersKama)).debug_batch_logged = true;
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
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per = env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] EhlersKama many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaEhlersKama)).debug_many_logged = true;
                }
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
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn will_fit_checked(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaEhlersKamaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            let need = required_bytes
                .checked_add(headroom_bytes)
                .ok_or(CudaEhlersKamaError::SizeOverflow)?;
            if need <= free {
                Ok(())
            } else {
                Err(CudaEhlersKamaError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                })
            }
        } else {
            Ok(())
        }
    }

    #[inline]
    fn bytes_for<T>(elems: usize) -> Result<usize, CudaEhlersKamaError> {
        elems
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(CudaEhlersKamaError::SizeOverflow)
    }

    fn expand_grid(range: &EhlersKamaBatchRange) -> Vec<EhlersKamaParams> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return vec![EhlersKamaParams { period: Some(start) }];
        }
        let mut params = Vec::new();
        if start < end {
            let mut value = start;
            let step_sz = step.max(1);
            while value <= end {
                params.push(EhlersKamaParams { period: Some(value) });
                match value.checked_add(step_sz) {
                    Some(next) => value = next,
                    None => break,
                }
            }
        } else {
            // reversed bounds supported
            let mut value = start;
            let step_sz = step.max(1);
            loop {
                params.push(EhlersKamaParams { period: Some(value) });
                match value.checked_sub(step_sz) {
                    Some(next) => {
                        if next < end { break; }
                        value = next;
                    }
                    None => break,
                }
            }
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
            let (s, e, st) = sweep.period;
            return Err(CudaEhlersKamaError::InvalidRange { start: s, end: e, step: st });
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

        // Grid-stride launch: single kernel launch covers all combos.
        let func = self
            .module
            .get_function("ehlers_kama_batch_f32")
            .map_err(|_| CudaEhlersKamaError::MissingKernelSymbol { name: "ehlers_kama_batch_f32" })?;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(1),
        };
        let grid_x: u32 = ((n_combos as u32 + block_x - 1) / block_x).max(1);
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut first_valid_i = first_valid as i32;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }

        unsafe {
            (*(self as *const _ as *mut CudaEhlersKama)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
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
        let required =
            Self::bytes_for::<f32>(data_f32.len())?
                .checked_add(Self::bytes_for::<i32>(combos.len())?)
                .ok_or(CudaEhlersKamaError::SizeOverflow)?
                .checked_add(Self::bytes_for::<f32>(combos.len().checked_mul(series_len).ok_or(CudaEhlersKamaError::SizeOverflow)?)?)
                .ok_or(CudaEhlersKamaError::SizeOverflow)?;
        Self::will_fit_checked(required, 64 * 1024 * 1024)?;

        let d_prices = DeviceBuffer::from_slice(data_f32)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }?;

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
                self.stream.launch(&func, (grid_x, 1, 1), (block_x, 1, 1), 0, args)?;
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
        self.stream.synchronize()?;

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
        // Pre-fill outputs with NaN to guarantee warm-up semantics.
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
                self.stream.launch(&func, (grid_x, 1, 1), (block_x, 1, 1), 0, args)?;
            }
        }
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
        if series_len == 0 {
            return Err(CudaEhlersKamaError::InvalidInput(
                "series_len is zero".into(),
            ));
        }
        let n_combos = combos.len();
        if n_combos == 0 {
            return Err(CudaEhlersKamaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let mut periods_i32 = Vec::with_capacity(n_combos);
        for prm in combos {
            periods_i32.push(prm.period.unwrap_or(0) as i32);
        }
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }?;
        // Pre-fill outputs with NaN to guarantee warm-up semantics.
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
                self.stream.launch(&func, (grid_x, 1, 1), (block_x, 1, 1), 0, args)?;
            }
        }
        self.launch_batch_kernel(
            d_prices,
            &d_periods,
            first_valid,
            series_len,
            n_combos,
            &mut d_out,
        )?;
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
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
        arr.buf.copy_to(out)?;
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
                let has_2d = self
                    .module
                    .get_function("ehlers_kama_multi_series_one_param_2d_f32")
                    .is_ok();
                if has_2d {
                    let choose = |n: usize| -> (u32, u32) {
                        if n <= 1 {
                            return (1, 1);
                        }
                        if n <= 2 {
                            return (2, 1);
                        }
                        if n <= 4 {
                            return (4, 1);
                        }
                        if n <= 8 {
                            return (8, 1);
                        }
                        if n <= 16 {
                            return (16, 1);
                        }
                        if n <= 32 {
                            return (32, 1);
                        }
                        if n <= 64 {
                            return (64, 1);
                        }
                        (64, 2)
                    };
                    let (tx, ty) = choose(num_series);
                    (
                        ManySeriesKernelSelected::Tiled2D { tx, ty },
                        ManySeriesKernelPolicy::Tiled2D { tx, ty },
                    )
                } else {
                    (
                        ManySeriesKernelSelected::OneD { block_x: 64 },
                        ManySeriesKernelPolicy::OneD { block_x: 64 },
                    )
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
                // Use the dedicated 1D kernel for one-param many-series.
                // Grid.x maps to series, each block processes one full series sequentially.
                let func = self
                    .module
                    .get_function("ehlers_kama_multi_series_one_param_f32")
                    .map_err(|_| CudaEhlersKamaError::MissingKernelSymbol { name: "ehlers_kama_multi_series_one_param_f32" })?;
                let tx = block_x.max(1);
                let blocks = (num_series as u32).max(1);
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
                    self.stream.launch(&func, grid, block, 0, args)?;
                }
            }
            ManySeriesKernelPolicy::Tiled2D { tx, ty } => {
                let mut func = self
                    .module
                    .get_function("ehlers_kama_multi_series_one_param_2d_f32")
                    .map_err(|_| CudaEhlersKamaError::MissingKernelSymbol { name: "ehlers_kama_multi_series_one_param_2d_f32" })?;
                // Prefer shared when the ring buffer is sizable (advisory).
                let _ = func.set_cache_config(CacheConfig::PreferShared);

                let tile = (tx * ty).max(1);
                let blocks = ((num_series as u32 + tile - 1) / tile).max(1);
                let grid: GridSize = (blocks, 1, 1).into();
                let block: BlockSize = (tx, ty, 1).into();
                // Dynamic shared memory for per-thread ring: (period-1) * tile_series * sizeof(f32)
                let mut shmem_bytes: u32 = ((period.saturating_sub(1) * tile as usize)
                    .saturating_mul(std::mem::size_of::<f32>()))
                .try_into()
                .map_err(|_| {
                    CudaEhlersKamaError::InvalidInput("shared memory bytes overflow".into())
                })?;
                // Soft-guard: if requested dynamic shared exceeds device limit, fall back to 0.
                if let Ok(dev) = Device::get_device(self.device_id) {
                    let max_dyn = dev
                        .get_attribute(cust::device::DeviceAttribute::MaxSharedMemoryPerBlock)
                        .map(|v| v as u32)
                        .unwrap_or(0);
                    if max_dyn > 0 && shmem_bytes > max_dyn {
                        shmem_bytes = 0;
                    }
                }

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
                    self.stream.launch(&func, grid, block, shmem_bytes, args)?;
                }
            }
            _ => unreachable!(),
        }

        unsafe {
            (*(self as *const _ as *mut CudaEhlersKama)).last_many = Some(selected);
        }
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
        let required =
            Self::bytes_for::<f32>(cols.checked_mul(rows).ok_or(CudaEhlersKamaError::SizeOverflow)?)?
                .checked_add(Self::bytes_for::<i32>(cols)?)
                .ok_or(CudaEhlersKamaError::SizeOverflow)?
                .checked_add(Self::bytes_for::<f32>(cols.checked_mul(rows).ok_or(CudaEhlersKamaError::SizeOverflow)?)?)
                .ok_or(CudaEhlersKamaError::SizeOverflow)?;
        Self::will_fit_checked(required, 64 * 1024 * 1024)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }?;

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
                self.stream.launch(&func, (grid_x, 1, 1), (block_x, 1, 1), 0, args)?;
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
                self.stream.launch(&func, (grid_x, 1, 1), (block_x, 1, 1), 0, args)?;
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
                self.stream.launch(&func, (grid_x, 1, 1), (block_x, 1, 1), 0, args)?;
            }
        }
        // Guard rails: as an ultimate safety, patch warm-up NaNs on host and
        // No host-side warm-up enforcement/patching needed: compute kernels skip warm region
        // and outputs were prefilled with NaN.

        self.stream.synchronize()?;

        if env::var("DUMP_KAMA").ok().as_deref() == Some("1") {
            let mut host = vec![0f32; cols * rows];
            d_out_tm.copy_to(&mut host)?;
            let dump_cols = cols.min(8);
            eprintln!(
                "[DUMP] first row (t=0) first {} series: {:?}",
                dump_cols,
                &host[0..dump_cols]
            );
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

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }?;

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
                self.stream.launch(&func, (grid_x, 1, 1), (block_x, 1, 1), 0, args)?;
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
        self.stream.synchronize()?;

        // Use pinned host buffer for faster D2H, then copy into provided slice.
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(cols * rows)? };
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
        ehlers_kama_benches,
        CudaEhlersKama,
        crate::indicators::moving_averages::ehlers_kama::EhlersKamaBatchRange,
        crate::indicators::moving_averages::ehlers_kama::EhlersKamaParams,
        ehlers_kama_batch_dev,
        ehlers_kama_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::ehlers_kama::EhlersKamaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::ehlers_kama::EhlersKamaParams { period: Some(64) },
        "ehlers_kama",
        "ehlers_kama"
    );
    pub use ehlers_kama_benches::bench_profiles;
}
