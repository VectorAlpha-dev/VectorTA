//! CUDA support for the two-pole high-pass filter (highpass_2_pole).
//!
//! Provides zero-copy device entry points that mirror the ALMA CUDA surface:
//! - one-series × many-parameter sweeps via `highpass2_batch_f32`
//! - time-major many-series × one-parameter execution via
//!   `highpass2_many_series_one_param_f32`
//!
//! Coefficients derived from `(period, k)` are precomputed on the host so each
//! output row reuses shared data instead of recomputing expensive trig values
//! on the device.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::highpass_2_pole::{HighPass2BatchRange, HighPass2Params};
use cust::context::{CacheConfig, Context};
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;
use cust::error::CudaError;

#[derive(Debug, Error)]
pub enum CudaHighPass2Error {
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Out of device memory (required={required}B, free={free}B, headroom={headroom}B)")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Launch config too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("Device mismatch: buffer on device {buf}, current device {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("Invalid usize range: start={start}, end={end}, step={step}")]
    InvalidRangeUsize { start: usize, end: usize, step: usize },
    #[error("Invalid f64 range: start={start}, end={end}, step={step}")]
    InvalidRangeF64 { start: f64, end: f64, step: f64 },
    #[error("size overflow while computing {what}")]
    SizeOverflow { what: &'static str },
    #[error("Not implemented")]
    NotImplemented,
}

// -------- Kernel policy + introspection (ALMA parity adapted to recurrence) --------

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
pub struct CudaHighPass2Policy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaHighPass2Policy {
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

pub struct CudaHighPass2 {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaHighPass2Policy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

struct PreparedHighPass2Batch {
    combos: Vec<HighPass2Params>,
    first_valid: usize,
    series_len: usize,
    periods_i32: Vec<i32>,
    c_vals: Vec<f32>,
    cm2_vals: Vec<f32>,
    two_1m_vals: Vec<f32>,
    neg_oma_sq_vals: Vec<f32>,
}

struct PreparedHighPass2ManySeries {
    first_valids: Vec<i32>,
    period: i32,
    c: f32,
    cm2: f32,
    two_1m: f32,
    neg_oma_sq: f32,
}

/// Reusable device-side parameter buffers for batch highpass2 runs.
pub struct HighPass2BatchDeviceParams {
    pub n_combos: usize,
    pub first_valid: usize,
    pub series_len: usize,
    pub d_periods: DeviceBuffer<i32>,
    pub d_c: DeviceBuffer<f32>,
    pub d_cm2: DeviceBuffer<f32>,
    pub d_two_1m: DeviceBuffer<f32>,
    pub d_neg_oma_sq: DeviceBuffer<f32>,
}

impl CudaHighPass2 {
    pub fn new(device_id: usize) -> Result<Self, CudaHighPass2Error> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/highpass2_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            // Default JIT opt is O4; keep default or make explicit with OptLevel(O4).
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
        // Prefer L1 cache for memory-bound recurrences (hint; driver may override).
        if let Ok(mut f) = module.get_function("highpass2_batch_f32") {
            let _ = f.set_cache_config(CacheConfig::PreferL1);
        }
        if let Ok(mut f) = module.get_function("highpass2_many_series_one_param_f32") {
            let _ = f.set_cache_config(CacheConfig::PreferL1);
        }
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaHighPass2Policy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaHighPass2Policy,
    ) -> Result<Self, CudaHighPass2Error> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaHighPass2Policy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaHighPass2Policy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaHighPass2Error> { self.stream.synchronize().map_err(Into::into) }
    pub fn context_arc(&self) -> Arc<Context> { self._context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn calc_launch_1d(
        &self,
        func: &Function,
        n_items: usize,
        policy_block_x: Option<u32>,
    ) -> (BlockSize, GridSize) {
        if let Some(bx) = policy_block_x {
            let bx = bx.max(32);
            let gx = ((n_items as u32 + bx - 1) / bx).max(1);
            return ((bx, 1, 1).into(), (gx, 1, 1).into());
        }
        // dynamic shared mem = 0, no block-size limit
        let (min_grid, block_x) = func
            .suggested_launch_configuration(0usize, BlockSize::xyz(0, 0, 0))
            .unwrap_or((1, 256));
        let block_x = block_x.max(128); // small floor for latency hiding
        let mut grid_x = ((n_items as u32 + block_x - 1) / block_x).max(min_grid.max(1));
        if let Ok(dev) = Device::get_device(self.device_id) {
            if let Ok(max_gx) = dev.get_attribute(DeviceAttribute::MaxGridDimX) {
                grid_x = grid_x.min(max_gx as u32);
            }
        }
        ((block_x, 1, 1).into(), (grid_x, 1, 1).into())
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline]
    fn will_fit_checked(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaHighPass2Error> {
        if !Self::mem_check_enabled() { return Ok(()); }
        if let Some((free, _total)) = Self::device_mem_info() {
            let need = required_bytes
                .checked_add(headroom_bytes)
                .ok_or(CudaHighPass2Error::OutOfMemory { required: usize::MAX, free, headroom: headroom_bytes })?;
            if need <= free { Ok(()) } else { Err(CudaHighPass2Error::OutOfMemory { required: need, free, headroom: headroom_bytes }) }
        } else { Ok(()) }
    }

    #[inline]
    fn assert_current_device(&self) -> Result<(), CudaHighPass2Error> {
        unsafe {
            let mut dev: i32 = -1;
            let _ = cust::sys::cuCtxGetDevice(&mut dev);
            if dev < 0 { return Ok(()); }
            let cur = dev as u32;
            if cur != self.device_id {
                return Err(CudaHighPass2Error::DeviceMismatch { buf: self.device_id, current: cur });
            }
        }
        Ok(())
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] HIGHPASS2 batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaHighPass2)).debug_batch_logged = true;
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
                let per = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] HIGHPASS2 many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaHighPass2)).debug_many_logged = true;
                }
            }
        }
    }

    pub fn highpass2_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &HighPass2BatchRange,
    ) -> Result<DeviceArrayF32, CudaHighPass2Error> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        // VRAM estimate: prices + param arrays + out (guard overflow)
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let prices_bytes = prepared
            .series_len
            .checked_mul(sz_f32)
            .ok_or(CudaHighPass2Error::SizeOverflow { what: "prices bytes" })?;
        let per_combo_bytes = sz_i32
            .checked_add(4usize.checked_mul(sz_f32).ok_or(CudaHighPass2Error::SizeOverflow { what: "param bytes" })?)
            .ok_or(CudaHighPass2Error::SizeOverflow { what: "param bytes" })?;
        let params_bytes = n_combos
            .checked_mul(per_combo_bytes)
            .ok_or(CudaHighPass2Error::SizeOverflow { what: "params bytes" })?;
        let out_elems = prepared
            .series_len
            .checked_mul(n_combos)
            .ok_or(CudaHighPass2Error::SizeOverflow { what: "output elements" })?;
        let out_bytes = out_elems
            .checked_mul(sz_f32)
            .ok_or(CudaHighPass2Error::SizeOverflow { what: "output bytes" })?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or(CudaHighPass2Error::SizeOverflow { what: "total bytes" })?;
        Self::will_fit_checked(required, 64 * 1024 * 1024)?;

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }?;
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&prepared.periods_i32, &self.stream) }?;
        let d_c       = unsafe { DeviceBuffer::from_slice_async(&prepared.c_vals, &self.stream) }?;
        let d_cm2     = unsafe { DeviceBuffer::from_slice_async(&prepared.cm2_vals, &self.stream) }?;
        let d_two     = unsafe { DeviceBuffer::from_slice_async(&prepared.two_1m_vals, &self.stream) }?;
        let d_neg     = unsafe { DeviceBuffer::from_slice_async(&prepared.neg_oma_sq_vals, &self.stream) }?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(prepared.series_len * n_combos, &self.stream) }?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_c,
            &d_cm2,
            &d_two,
            &d_neg,
            prepared.series_len,
            n_combos,
            prepared.first_valid,
            &mut d_out,
        )?;

        // Synchronize so returned handle is immediately consumable (CAI stream omitted)
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn highpass2_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_c: &DeviceBuffer<f32>,
        d_cm2: &DeviceBuffer<f32>,
        d_two_1m: &DeviceBuffer<f32>,
        d_neg_oma_sq: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHighPass2Error> {
        if series_len == 0 {
            return Err(CudaHighPass2Error::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if n_combos == 0 {
            return Err(CudaHighPass2Error::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaHighPass2Error::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, series_len
            )));
        }
        let expected = n_combos;
        if d_periods.len() != expected
            || d_c.len() != expected
            || d_cm2.len() != expected
            || d_two_1m.len() != expected
            || d_neg_oma_sq.len() != expected
        {
            return Err(CudaHighPass2Error::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaHighPass2Error::InvalidInput(
                "prices length must equal series_len".into(),
            ));
        }
        if d_out.len() != series_len * expected {
            return Err(CudaHighPass2Error::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        // Ensure we are on the same device/context as the buffers were allocated for
        let _ = self.assert_current_device();

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_c,
            d_cm2,
            d_two_1m,
            d_neg_oma_sq,
            series_len,
            n_combos,
            first_valid,
            d_out,
        )
    }

    pub fn highpass2_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &HighPass2BatchRange,
        out_flat: &mut [f32],
    ) -> Result<(), CudaHighPass2Error> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        if out_flat.len() != prepared.series_len * prepared.combos.len() {
            return Err(CudaHighPass2Error::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.highpass2_batch_dev(data_f32, sweep)?;
        // Use pinned host memory for faster D2H, then copy into out slice
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(handle.len())? };
        unsafe { handle.buf.async_copy_to(pinned.as_mut_slice(), &self.stream)?; }
        self.stream.synchronize()?;
        out_flat.copy_from_slice(pinned.as_slice());
        Ok(())
    }

    pub fn highpass2_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &HighPass2Params,
    ) -> Result<DeviceArrayF32, CudaHighPass2Error> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        // VRAM estimate: input + out + first_valids (guard overflow)
        let sz_f32 = std::mem::size_of::<f32>();
        let sz_i32 = std::mem::size_of::<i32>();
        let elems = num_series
            .checked_mul(series_len)
            .ok_or(CudaHighPass2Error::SizeOverflow { what: "many-series elements" })?;
        let in_bytes = elems
            .checked_mul(sz_f32)
            .ok_or(CudaHighPass2Error::SizeOverflow { what: "input bytes" })?;
        let out_bytes = in_bytes;
        let param_bytes = num_series
            .checked_mul(sz_i32)
            .ok_or(CudaHighPass2Error::SizeOverflow { what: "first_valids bytes" })?;
        let required = in_bytes
            .checked_add(out_bytes)
            .and_then(|x| x.checked_add(param_bytes))
            .ok_or(CudaHighPass2Error::SizeOverflow { what: "total bytes" })?;
        Self::will_fit_checked(required, 64 * 1024 * 1024)?;

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }?;
        let d_first = DeviceBuffer::from_slice(&prepared.first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(num_series * series_len, &self.stream) }?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            prepared.period,
            prepared.c,
            prepared.cm2,
            prepared.two_1m,
            prepared.neg_oma_sq,
            num_series,
            series_len,
            &mut d_out,
        )?;

        // Sync to make returned buffer immediately consumable
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn highpass2_many_series_one_param_time_major_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        c: f32,
        cm2: f32,
        two_1m: f32,
        neg_oma_sq: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHighPass2Error> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaHighPass2Error::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if d_prices_tm.len() != num_series * series_len {
            return Err(CudaHighPass2Error::InvalidInput(
                "prices_tm length mismatch".into(),
            ));
        }
        if d_out_tm.len() != num_series * series_len {
            return Err(CudaHighPass2Error::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }
        if d_first_valids.len() != num_series {
            return Err(CudaHighPass2Error::InvalidInput(
                "first_valids length mismatch".into(),
            ));
        }

        // Ensure current context matches allocation device
        let _ = self.assert_current_device();

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            period,
            c,
            cm2,
            two_1m,
            neg_oma_sq,
            num_series,
            series_len,
            d_out_tm,
        )
    }

    pub fn highpass2_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &HighPass2Params,
        out_tm: &mut [f32],
    ) -> Result<(), CudaHighPass2Error> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaHighPass2Error::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.highpass2_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(handle.len())? };
        unsafe { handle.buf.async_copy_to(pinned.as_mut_slice(), &self.stream)?; }
        self.stream.synchronize()?;
        out_tm.copy_from_slice(pinned.as_slice());
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_c: &DeviceBuffer<f32>,
        d_cm2: &DeviceBuffer<f32>,
        d_two_1m: &DeviceBuffer<f32>,
        d_neg_oma_sq: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHighPass2Error> {
        let func = self
            .module
            .get_function("highpass2_batch_f32")
            .map_err(|_| CudaHighPass2Error::MissingKernelSymbol { name: "highpass2_batch_f32" })?;

        let (block, grid) = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => {
                let block: BlockSize = (block_x, 1, 1).into();
                let grid_x = ((n_combos as u32 + block_x - 1) / block_x).max(1);
                (block, (grid_x, 1, 1).into())
            }
            BatchKernelPolicy::Auto => self.calc_launch_1d(&func, n_combos, None),
        };

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut c_ptr = d_c.as_device_ptr().as_raw();
            let mut cm2_ptr = d_cm2.as_device_ptr().as_raw();
            let mut two_ptr = d_two_1m.as_device_ptr().as_raw();
            let mut neg_ptr = d_neg_oma_sq.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut c_ptr as *mut _ as *mut c_void,
                &mut cm2_ptr as *mut _ as *mut c_void,
                &mut two_ptr as *mut _ as *mut c_void,
                &mut neg_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        unsafe {
            let bx = match self.policy.batch {
                BatchKernelPolicy::Plain { block_x } => block_x,
                BatchKernelPolicy::Auto => block.x,
            };
            (*(self as *const _ as *mut CudaHighPass2)).last_batch =
                Some(BatchKernelSelected::Plain { block_x: bx });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        c: f32,
        cm2: f32,
        two_1m: f32,
        neg_oma_sq: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHighPass2Error> {
        let func = self
            .module
            .get_function("highpass2_many_series_one_param_f32")
            .map_err(|_| CudaHighPass2Error::MissingKernelSymbol { name: "highpass2_many_series_one_param_f32" })?;

        let (block, grid) = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => {
                let block: BlockSize = (block_x, 1, 1).into();
                let grid_x = ((num_series as u32 + block_x - 1) / block_x).max(1);
                (block, (grid_x, 1, 1).into())
            }
            ManySeriesKernelPolicy::Auto => self.calc_launch_1d(&func, num_series, None),
        };

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut period_i = period;
            let mut c_val = c;
            let mut cm2_val = cm2;
            let mut two_val = two_1m;
            let mut neg_val = neg_oma_sq;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut c_val as *mut _ as *mut c_void,
                &mut cm2_val as *mut _ as *mut c_void,
                &mut two_val as *mut _ as *mut c_void,
                &mut neg_val as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, &mut args)?;
        }
        unsafe {
            let bx = match self.policy.many_series {
                ManySeriesKernelPolicy::OneD { block_x } => block_x,
                ManySeriesKernelPolicy::Auto => block.x,
            };
            (*(self as *const _ as *mut CudaHighPass2)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x: bx });
        }
        self.maybe_log_many_debug();
        Ok(())
    }

    // ---------- Optional: parameter reuse across sweeps ----------

    /// Uploads and returns reusable device-side parameter buffers for batch runs.
    pub fn upload_batch_params(
        &self,
        data_f32: &[f32],
        sweep: &HighPass2BatchRange,
    ) -> Result<HighPass2BatchDeviceParams, CudaHighPass2Error> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n = prepared.combos.len();

        let d_periods = unsafe { DeviceBuffer::from_slice_async(&prepared.periods_i32, &self.stream) }?;
        let d_c       = unsafe { DeviceBuffer::from_slice_async(&prepared.c_vals, &self.stream) }?;
        let d_cm2     = unsafe { DeviceBuffer::from_slice_async(&prepared.cm2_vals, &self.stream) }?;
        let d_two     = unsafe { DeviceBuffer::from_slice_async(&prepared.two_1m_vals, &self.stream) }?;
        let d_neg     = unsafe { DeviceBuffer::from_slice_async(&prepared.neg_oma_sq_vals, &self.stream) }?;

        Ok(HighPass2BatchDeviceParams {
            n_combos: n,
            first_valid: prepared.first_valid,
            series_len: prepared.series_len,
            d_periods,
            d_c,
            d_cm2,
            d_two_1m: d_two,
            d_neg_oma_sq: d_neg,
        })
    }

    /// Runs batch kernel using pre-uploaded parameter buffers; returns device handle.
    pub fn highpass2_batch_with_params_dev(
        &self,
        d_prices: &DeviceBuffer<f32>,
        params: &HighPass2BatchDeviceParams,
    ) -> Result<DeviceArrayF32, CudaHighPass2Error> {
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(params.series_len * params.n_combos, &self.stream) }?;
        self.launch_batch_kernel(
            d_prices,
            &params.d_periods,
            &params.d_c,
            &params.d_cm2,
            &params.d_two_1m,
            &params.d_neg_oma_sq,
            params.series_len,
            params.n_combos,
            params.first_valid,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: params.n_combos,
            cols: params.series_len,
        })
    }

    /// Convenience helper to run batch using pre-uploaded params and copy to a host slice.
    pub fn highpass2_batch_with_params_into_host_f32(
        &self,
        d_prices: &DeviceBuffer<f32>,
        params: &HighPass2BatchDeviceParams,
        out_flat: &mut [f32],
    ) -> Result<(), CudaHighPass2Error> {
        if out_flat.len() != params.series_len * params.n_combos {
            return Err(CudaHighPass2Error::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let arr = self.highpass2_batch_with_params_dev(d_prices, params)?;
        let mut pinned: LockedBuffer<f32> = unsafe { LockedBuffer::uninitialized(arr.len())? };
        unsafe { arr.buf.async_copy_to(pinned.as_mut_slice(), &self.stream)?; }
        self.stream.synchronize()?;
        out_flat.copy_from_slice(pinned.as_slice());
        Ok(())
    }

    /// Optional: run batch directly from a pinned host input buffer.
    pub fn highpass2_batch_dev_locked(
        &self,
        data_pinned: &LockedBuffer<f32>,
        sweep: &HighPass2BatchRange,
    ) -> Result<DeviceArrayF32, CudaHighPass2Error> {
        let prepared = Self::prepare_batch_inputs(data_pinned.as_slice(), sweep)?;

        let prices_bytes = prepared.series_len * std::mem::size_of::<f32>();
        let params_bytes =
            prepared.combos.len() * (std::mem::size_of::<i32>() + 4 * std::mem::size_of::<f32>());
        let out_bytes = prepared.series_len * prepared.combos.len() * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        Self::will_fit_checked(required, 64 * 1024 * 1024)?;

        // H2D from pinned source
        let mut d_prices: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized_async(prepared.series_len, &self.stream) }?;
        unsafe { d_prices.async_copy_from(data_pinned.as_slice(), &self.stream)?; }

        let params_dev = self.upload_batch_params(data_pinned.as_slice(), sweep)?;
        self.highpass2_batch_with_params_dev(&d_prices, &params_dev)
    }

    /// Optional: many-series output into pinned host buffer.
    pub fn highpass2_many_series_one_param_time_major_into_pinned_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &HighPass2Params,
        out_tm_pinned: &mut LockedBuffer<f32>,
    ) -> Result<(), CudaHighPass2Error> {
        if out_tm_pinned.len() != num_series * series_len {
            return Err(CudaHighPass2Error::InvalidInput(
                "out pinned buffer wrong length".into(),
            ));
        }
        let arr = self.highpass2_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        unsafe { arr.buf.async_copy_to(out_tm_pinned.as_mut_slice(), &self.stream)?; }
        self.stream.synchronize()?;
        Ok(())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &HighPass2BatchRange,
    ) -> Result<PreparedHighPass2Batch, CudaHighPass2Error> {
        if data_f32.is_empty() {
            return Err(CudaHighPass2Error::InvalidInput(
                "input data is empty".into(),
            ));
        }
        let combos = expand_grid(sweep)?;
        if combos.is_empty() {
            return Err(CudaHighPass2Error::InvalidInput(
                "no parameter combinations provided".into(),
            ));
        }

        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaHighPass2Error::InvalidInput("all values are NaN".into()))?;
        let series_len = data_f32.len();

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut c_vals = Vec::with_capacity(combos.len());
        let mut cm2_vals = Vec::with_capacity(combos.len());
        let mut two_vals = Vec::with_capacity(combos.len());
        let mut neg_vals = Vec::with_capacity(combos.len());

        for params in &combos {
            let period = params.period.unwrap_or(0);
            let k = params.k.unwrap_or(0.707);
            if period < 2 {
                return Err(CudaHighPass2Error::InvalidInput(
                    "period must be >= 2".into(),
                ));
            }
            if !(k > 0.0) || !k.is_finite() {
                return Err(CudaHighPass2Error::InvalidInput(format!(
                    "invalid k: {}",
                    k
                )));
            }
            if series_len - first_valid < period {
                return Err(CudaHighPass2Error::InvalidInput(format!(
                    "not enough valid data: needed >= {}, have {}",
                    period,
                    series_len - first_valid
                )));
            }

            let coeffs = compute_coefficients(period, k);
            periods_i32.push(period as i32);
            c_vals.push(coeffs.c);
            cm2_vals.push(coeffs.cm2);
            two_vals.push(coeffs.two_1m);
            neg_vals.push(coeffs.neg_oma_sq);
        }

        Ok(PreparedHighPass2Batch {
            combos,
            first_valid,
            series_len,
            periods_i32,
            c_vals,
            cm2_vals,
            two_1m_vals: two_vals,
            neg_oma_sq_vals: neg_vals,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &HighPass2Params,
    ) -> Result<PreparedHighPass2ManySeries, CudaHighPass2Error> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaHighPass2Error::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaHighPass2Error::InvalidInput(format!(
                "time-major slice length mismatch: got {}, expected {}",
                data_tm_f32.len(),
                num_series * series_len
            )));
        }

        let period = params.period.unwrap_or(48) as i32;
        let k = params.k.unwrap_or(0.707);
        if period < 2 {
            return Err(CudaHighPass2Error::InvalidInput(
                "period must be >= 2".into(),
            ));
        }
        if !(k > 0.0) || !k.is_finite() {
            return Err(CudaHighPass2Error::InvalidInput(format!(
                "invalid k: {}",
                k
            )));
        }

        let needed = period as usize;
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
                CudaHighPass2Error::InvalidInput(format!("series {} is entirely NaN", series))
            })?;
            if series_len - fv < needed {
                return Err(CudaHighPass2Error::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    series,
                    needed,
                    series_len - fv
                )));
            }
            first_valids.push(fv as i32);
        }

        let coeffs = compute_coefficients(period as usize, k);

        Ok(PreparedHighPass2ManySeries {
            first_valids,
            period,
            c: coeffs.c,
            cm2: coeffs.cm2,
            two_1m: coeffs.two_1m,
            neg_oma_sq: coeffs.neg_oma_sq,
        })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        highpass2_benches,
        CudaHighPass2,
        crate::indicators::moving_averages::highpass_2_pole::HighPass2BatchRange,
        crate::indicators::moving_averages::highpass_2_pole::HighPass2Params,
        highpass2_batch_dev,
        highpass2_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::highpass_2_pole::HighPass2BatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            k: (0.5, 0.5, 0.0)
        },
        crate::indicators::moving_averages::highpass_2_pole::HighPass2Params {
            period: Some(64),
            k: Some(0.5)
        },
        "highpass2",
        "highpass2"
    );
    pub use highpass2_benches::bench_profiles;
}

struct Coefficients {
    c: f32,
    cm2: f32,
    two_1m: f32,
    neg_oma_sq: f32,
}

fn compute_coefficients(period: usize, k: f64) -> Coefficients {
    use std::f64::consts::PI;

    let theta = 2.0 * PI * k / period as f64;
    let sin_v = theta.sin();
    let cos_v = theta.cos();
    let alpha = 1.0 + ((sin_v - 1.0) / cos_v);
    let c = (1.0 - 0.5 * alpha).powi(2);
    let cm2 = -2.0 * c;
    let one_minus_alpha = 1.0 - alpha;
    let two_1m = 2.0 * one_minus_alpha;
    let neg_oma_sq = -(one_minus_alpha * one_minus_alpha);

    Coefficients {
        c: c as f32,
        cm2: cm2 as f32,
        two_1m: two_1m as f32,
        neg_oma_sq: neg_oma_sq as f32,
    }
}

fn expand_grid(range: &HighPass2BatchRange) -> Result<Vec<HighPass2Params>, CudaHighPass2Error> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaHighPass2Error> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let (lo, hi) = if start <= end { (start, end) } else { (end, start) };
        let v: Vec<usize> = (lo..=hi).step_by(step).collect();
        if v.is_empty() {
            return Err(CudaHighPass2Error::InvalidRangeUsize { start, end, step });
        }
        Ok(v)
    }

    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaHighPass2Error> {
        const EPS: f64 = 1e-12;
        if step.abs() < EPS || (start - end).abs() < EPS {
            return Ok(vec![start]);
        }
        let step_eff = if start <= end { step.abs() } else { -step.abs() };
        let mut v = Vec::new();
        let mut x = start;
        if step_eff > 0.0 {
            while x <= end + EPS {
                v.push(x);
                x += step_eff;
            }
        } else {
            while x >= end - EPS {
                v.push(x);
                x += step_eff;
            }
        }
        if v.is_empty() {
            return Err(CudaHighPass2Error::InvalidRangeF64 { start, end, step });
        }
        Ok(v)
    }

    let periods = axis_usize(range.period)?;
    let ks = axis_f64(range.k)?;
    let total = periods
        .len()
        .checked_mul(ks.len())
        .ok_or(CudaHighPass2Error::SizeOverflow { what: "parameter grid" })?;
    let mut out = Vec::with_capacity(total);
    for &p in &periods {
        for &k in &ks {
            out.push(HighPass2Params { period: Some(p), k: Some(k) });
        }
    }
    Ok(out)
}
