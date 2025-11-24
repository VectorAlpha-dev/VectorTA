//! CUDA wrapper for OTTO (Optimized Trend Tracker Oscillator).
//!
//! Parity goals with ALMA wrapper:
//! - PTX embed and JIT options (DetermineTargetFromContext + O2 fallback)
//! - NON_BLOCKING stream, optional VRAM checks + chunked launches
//! - Batch (one series × many params) and Many-series × one param (time-major)
//!
//! Math category: Recurrence/IIR (per-row sequential). Batch path precomputes
//! abs(CMO(9)) on host once per series and reuses it for all rows, mirroring
//! the scalar batch path in `indicators::otto`.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use super::{BatchKernelPolicy, ManySeriesKernelPolicy};
use crate::indicators::otto::{OttoBatchRange, OttoParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(thiserror::Error, Debug)]
pub enum CudaOttoError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("otto: out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("otto: missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("otto: invalid input: {0}")]
    InvalidInput(String),
    #[error("otto: invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("otto: launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("otto: device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("otto: not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaOttoPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaOttoPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain1D { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaOtto {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaOttoPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaOtto {
    pub fn new(device_id: usize) -> Result<Self, CudaOttoError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/otto_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaOttoPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaOttoPolicy,
    ) -> Result<Self, CudaOttoError> {
        let mut me = Self::new(device_id)?;
        me.policy = policy;
        Ok(me)
    }
    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }
    #[inline]
    pub fn set_policy(&mut self, policy: CudaOttoPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaOttoPolicy {
        &self.policy
    }
    #[inline]
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    #[inline]
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaOttoError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaOttoError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaOttoError::OutOfMemory {
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
    fn validate_launch(&self, grid: (u32, u32, u32), block: (u32, u32, u32)) -> Result<(), CudaOttoError> {
        let dev = Device::get_device(self.device_id)?;
        let max_bx = dev.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        let max_by = dev.get_attribute(DeviceAttribute::MaxBlockDimY)? as u32;
        let max_bz = dev.get_attribute(DeviceAttribute::MaxBlockDimZ)? as u32;
        let max_gx = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_gy = dev.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        let max_gz = dev.get_attribute(DeviceAttribute::MaxGridDimZ)? as u32;
        if block.0 == 0 || block.1 == 0 || block.2 == 0
            || grid.0 == 0 || grid.1 == 0 || grid.2 == 0
            || block.0 > max_bx || block.1 > max_by || block.2 > max_bz
            || grid.0 > max_gx || grid.1 > max_gy || grid.2 > max_gz
        {
            return Err(CudaOttoError::LaunchConfigTooLarge {
                gx: grid.0,
                gy: grid.1,
                gz: grid.2,
                bx: block.0,
                by: block.1,
                bz: block.2,
            });
        }
        Ok(())
    }

    // ------- Batch (one series × many params) -------

    pub fn otto_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &OttoBatchRange,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, Vec<OttoParams>), CudaOttoError> {
        let inputs = Self::prepare_batch_inputs(prices_f32, sweep)?;

        let elem_size = std::mem::size_of::<f32>();
        let prices_bytes = inputs
            .series_len
            .checked_mul(elem_size)
            .ok_or_else(|| CudaOttoError::InvalidInput("series_len overflow".into()))?;
        let cabs_bytes = inputs
            .series_len
            .checked_mul(elem_size)
            .ok_or_else(|| CudaOttoError::InvalidInput("series_len overflow".into()))?;
        let params_scalars = 5usize;
        let params_bytes = inputs
            .combos
            .len()
            .checked_mul(params_scalars)
            .and_then(|v| v.checked_mul(elem_size))
            .ok_or_else(|| CudaOttoError::InvalidInput("params_bytes overflow".into()))?;
        let elems = inputs
            .series_len
            .checked_mul(inputs.combos.len())
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| CudaOttoError::InvalidInput("rows*cols overflow".into()))?;
        let out_bytes = elems
            .checked_mul(elem_size)
            .ok_or_else(|| CudaOttoError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(cabs_bytes)
            .and_then(|v| v.checked_add(params_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaOttoError::InvalidInput("total bytes overflow".into()))?;
        let headroom = 64 * 1024 * 1024usize;
        Self::will_fit(required, headroom)?;

        // H2D copies (pinned/async for prices + cabs)
        let d_prices = self.htod_copy_f32(prices_f32)?;
        let d_cabs = self.htod_copy_f32(inputs.cabs.as_slice())?;
        let d_ott_periods = DeviceBuffer::from_slice(&inputs.ott_periods).map_err(CudaOttoError::Cuda)?;
        let d_ott_percents = DeviceBuffer::from_slice(&inputs.ott_percents).map_err(CudaOttoError::Cuda)?;
        let d_fast = DeviceBuffer::from_slice(&inputs.fast).map_err(CudaOttoError::Cuda)?;
        let d_slow = DeviceBuffer::from_slice(&inputs.slow).map_err(CudaOttoError::Cuda)?;
        let d_coco = DeviceBuffer::from_slice(&inputs.coco).map_err(CudaOttoError::Cuda)?;

        let elems = inputs
            .series_len
            .checked_mul(inputs.combos.len())
            .ok_or_else(|| CudaOttoError::InvalidInput("rows*cols overflow".into()))?;
        let mut d_hott: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }.map_err(CudaOttoError::Cuda)?;
        let mut d_lott: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }.map_err(CudaOttoError::Cuda)?;

        self.launch_batch_kernel(
            &d_prices,
            &d_cabs,
            &d_ott_periods,
            &d_ott_percents,
            &d_fast,
            &d_slow,
            &d_coco,
            inputs.series_len,
            inputs.combos.len(),
            0,
            &mut d_hott,
            &mut d_lott,
        )?;

        self.stream.synchronize().map_err(CudaOttoError::Cuda)?;

        Ok((
            DeviceArrayF32 {
                buf: d_hott,
                rows: inputs.combos.len(),
                cols: inputs.series_len,
            },
            DeviceArrayF32 {
                buf: d_lott,
                rows: inputs.combos.len(),
                cols: inputs.series_len,
            },
            inputs.combos,
        ))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_cabs: &DeviceBuffer<f32>,
        d_ott_periods: &DeviceBuffer<i32>,
        d_ott_percents: &DeviceBuffer<f32>,
        d_fast: &DeviceBuffer<i32>,
        d_slow: &DeviceBuffer<i32>,
        d_coco: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_hott: &mut DeviceBuffer<f32>,
        d_lott: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaOttoError> {
        let func = self
            .module
            .get_function("otto_batch_f32")
            .map_err(|_| CudaOttoError::MissingKernelSymbol { name: "otto_batch_f32" })?;

        unsafe {
            let this = self as *const _ as *mut CudaOtto;
            (*this).last_batch = Some(BatchKernelSelected::Plain1D { block_x: 1 });
        }
        self.maybe_log_batch_debug();

        const MAX_Y: usize = 65_535;
        let block: BlockSize = (1, 1, 1).into();
        self.validate_launch((1, 1, 1), (1, 1, 1))?;
        let mut start = 0usize;
        while start < n_combos {
            let chunk = (n_combos - start).min(MAX_Y);
            let grid: GridSize = (chunk as u32, 1, 1).into();
            unsafe {
                let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                let mut cabs_ptr = d_cabs.as_device_ptr().as_raw();
                let mut ottp_ptr = d_ott_periods.as_device_ptr().add(start).as_raw();
                let mut ottk_ptr = d_ott_percents.as_device_ptr().add(start).as_raw();
                let mut fast_ptr = d_fast.as_device_ptr().add(start).as_raw();
                let mut slow_ptr = d_slow.as_device_ptr().add(start).as_raw();
                let mut coco_ptr = d_coco.as_device_ptr().add(start).as_raw();
                let mut series_len_i = series_len as i32;
                let mut n_combos_i = chunk as i32;
                let mut first_valid_i = first_valid as i32;
                let mut hott_ptr = d_hott.as_device_ptr().add(start * series_len).as_raw();
                let mut lott_ptr = d_lott.as_device_ptr().add(start * series_len).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut prices_ptr as *mut _ as *mut c_void,
                    &mut cabs_ptr as *mut _ as *mut c_void,
                    &mut ottp_ptr as *mut _ as *mut c_void,
                    &mut ottk_ptr as *mut _ as *mut c_void,
                    &mut fast_ptr as *mut _ as *mut c_void,
                    &mut slow_ptr as *mut _ as *mut c_void,
                    &mut coco_ptr as *mut _ as *mut c_void,
                    &mut series_len_i as *mut _ as *mut c_void,
                    &mut n_combos_i as *mut _ as *mut c_void,
                    &mut first_valid_i as *mut _ as *mut c_void,
                    &mut hott_ptr as *mut _ as *mut c_void,
                    &mut lott_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args).map_err(CudaOttoError::Cuda)?;
            }
            start += chunk;
        }
        Ok(())
    }

    // ------- Many-series × one param (time-major) -------

    pub fn otto_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &OttoParams,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32), CudaOttoError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaOttoError::InvalidInput("cols*rows overflow".into()))?;
        if prices_tm_f32.len() != expected {
            return Err(CudaOttoError::InvalidInput(
                "prices_tm length must equal cols*rows".into(),
            ));
        }
        let ott_period = params.ott_period.unwrap_or(2).max(1);
        let ott_percent = params.ott_percent.unwrap_or(0.6) as f32;
        let fast = params.fast_vidya_length.unwrap_or(10).max(1);
        let slow = params.slow_vidya_length.unwrap_or(25).max(1);
        let coco = params.correcting_constant.unwrap_or(100000.0) as f32;

        let elem_size = std::mem::size_of::<f32>();
        let prices_bytes = expected
            .checked_mul(elem_size)
            .ok_or_else(|| CudaOttoError::InvalidInput("prices bytes overflow".into()))?;
        let out_elems = expected
            .checked_mul(2)
            .ok_or_else(|| CudaOttoError::InvalidInput("output elems overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(elem_size)
            .ok_or_else(|| CudaOttoError::InvalidInput("output bytes overflow".into()))?;
        let required = prices_bytes
            .checked_add(out_bytes)
            .ok_or_else(|| CudaOttoError::InvalidInput("total bytes overflow".into()))?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_prices_tm = self.htod_copy_f32(prices_tm_f32)?;
        let mut d_hott_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }.map_err(CudaOttoError::Cuda)?;
        let mut d_lott_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(cols * rows, &self.stream) }.map_err(CudaOttoError::Cuda)?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            cols,
            rows,
            ott_period,
            ott_percent,
            fast,
            slow,
            coco,
            &mut d_hott_tm,
            &mut d_lott_tm,
        )?;

        self.stream.synchronize().map_err(CudaOttoError::Cuda)?;

        Ok((
            DeviceArrayF32 {
                buf: d_hott_tm,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_lott_tm,
                rows,
                cols,
            },
        ))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        ott_period: usize,
        ott_percent: f32,
        fast: usize,
        slow: usize,
        coco: f32,
        d_hott_tm: &mut DeviceBuffer<f32>,
        d_lott_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaOttoError> {
        let func = self
            .module
            .get_function("otto_many_series_one_param_f32")
            .map_err(|_| CudaOttoError::MissingKernelSymbol { name: "otto_many_series_one_param_f32" })?;
        unsafe {
            let this = self as *const _ as *mut CudaOtto;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x: 1 });
        }
        self.maybe_log_many_debug();

        let grid_x = rows as u32;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();
        self.validate_launch((grid_x.max(1), 1, 1), (1, 1, 1))?;
        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut ott_p_i = ott_period as i32;
            let mut ott_k_f = ott_percent as f32;
            let mut fast_i = fast as i32;
            let mut slow_i = slow as i32;
            let mut coco_f = coco as f32;
            let mut hott_ptr = d_hott_tm.as_device_ptr().as_raw();
            let mut lott_ptr = d_lott_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut ott_p_i as *mut _ as *mut c_void,
                &mut ott_k_f as *mut _ as *mut c_void,
                &mut fast_i as *mut _ as *mut c_void,
                &mut slow_i as *mut _ as *mut c_void,
                &mut coco_f as *mut _ as *mut c_void,
                &mut hott_ptr as *mut _ as *mut c_void,
                &mut lott_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args).map_err(CudaOttoError::Cuda)?;
        }
        Ok(())
    }

    // ---------- Helpers ----------

    fn htod_copy_f32(&self, host: &[f32]) -> Result<DeviceBuffer<f32>, CudaOttoError> {
        let mut d: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(host.len(), &self.stream) }?;
        unsafe {
            d.async_copy_from(host, &self.stream)?;
        }
        Ok(d)
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] OTTO batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut Self)).debug_batch_logged = true;
                }
            }
        }
    }
    #[inline]
    fn maybe_log_many_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] OTTO many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut Self)).debug_many_logged = true;
                }
            }
        }
    }

    fn prepare_batch_inputs(
        prices_f32: &[f32],
        sweep: &OttoBatchRange,
    ) -> Result<BatchInputs, CudaOttoError> {
        if prices_f32.is_empty() {
            return Err(CudaOttoError::InvalidInput("empty price input".into()));
        }
        let combos = {
            let v = expand_grid_otto(sweep);
            if v.is_empty() {
                return Err(CudaOttoError::InvalidInput("no parameter combinations".into()));
            }
            v
        };

        // First valid (Pine-style nz -> treat NaN as 0.0 for diffs); still detect all-NaN stream
        let first_valid = prices_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaOttoError::InvalidInput("all values are NaN".into()))?;

        let cabs = build_cmo_abs9(prices_f32);

        let series_len = prices_f32.len();
        let mut ott_periods = Vec::with_capacity(combos.len());
        let mut ott_percents = Vec::with_capacity(combos.len());
        let mut fast = Vec::with_capacity(combos.len());
        let mut slow = Vec::with_capacity(combos.len());
        let mut coco = Vec::with_capacity(combos.len());
        for p in &combos {
            ott_periods.push(p.ott_period.unwrap_or(2) as i32);
            ott_percents.push(p.ott_percent.unwrap_or(0.6) as f32);
            fast.push(p.fast_vidya_length.unwrap_or(10) as i32);
            slow.push(p.slow_vidya_length.unwrap_or(25) as i32);
            coco.push(p.correcting_constant.unwrap_or(100000.0) as f32);
        }

        Ok(BatchInputs {
            combos,
            series_len,
            first_valid,
            cabs,
            ott_periods,
            ott_percents,
            fast,
            slow,
            coco,
        })
    }
}

fn expand_grid_otto(r: &OttoBatchRange) -> Vec<OttoParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        if start < end {
            (start..=end).step_by(step).collect()
        } else {
            let mut v = Vec::new();
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if cur - end < step {
                    break;
                }
                cur -= step;
            }
            v
        }
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut out = Vec::new();
        if start < end {
            let st = if step > 0.0 { step } else { -step };
            let mut x = start;
            while x <= end + 1e-12 {
                out.push(x);
                x += st;
            }
        } else {
            let st = if step > 0.0 { -step } else { step };
            if st.abs() < 1e-12 {
                return vec![start];
            }
            let mut x = start;
            while x >= end - 1e-12 {
                out.push(x);
                x += st;
            }
        }
        out
    }
    let p = axis_usize(r.ott_period);
    let op = axis_f64(r.ott_percent);
    let fv = axis_usize(r.fast_vidya);
    let sv = axis_usize(r.slow_vidya);
    let cc = axis_f64(r.correcting_constant);
    let mt = &r.ma_types;
    let mut v = Vec::with_capacity(p.len() * op.len() * fv.len() * sv.len() * cc.len() * mt.len());
    for &pp in &p {
        for &oo in &op {
            for &ff in &fv {
                for &ss in &sv {
                    for &ccv in &cc {
                        for m in mt {
                            v.push(OttoParams {
                                ott_period: Some(pp),
                                ott_percent: Some(oo),
                                fast_vidya_length: Some(ff),
                                slow_vidya_length: Some(ss),
                                correcting_constant: Some(ccv),
                                ma_type: Some(m.clone()),
                            });
                        }
                    }
                }
            }
        }
    }
    v
}

fn build_cmo_abs9(prices: &[f32]) -> Vec<f32> {
    // Match scalar path semantics bit-for-bit by computing in f64 with the
    // same ring/update ordering, then cast once to f32.
    const P: usize = 9;
    let n = prices.len();
    let mut out = vec![0.0f32; n];
    if n == 0 {
        return out;
    }
    let mut ring_up = [0.0f64; P];
    let mut ring_dn = [0.0f64; P];
    let mut sum_up = 0.0f64;
    let mut sum_dn = 0.0f64;
    let mut head = 0usize;
    let mut prev = prices[0] as f64;
    for i in 0..n {
        let x = prices[i] as f64;
        if i > 0 {
            let mut d = x - prev;
            if !x.is_finite() || !prev.is_finite() {
                d = 0.0;
            }
            if i >= P {
                sum_up -= ring_up[head];
                sum_dn -= ring_dn[head];
            }
            let (up, dn) = if d > 0.0 { (d, 0.0) } else { (0.0, -d) };
            ring_up[head] = up;
            ring_dn[head] = dn;
            sum_up += up;
            sum_dn += dn;
            head += 1;
            if head == P {
                head = 0;
            }
        }
        prev = x;
        let denom = sum_up + sum_dn;
        let c = if i >= P && denom != 0.0 {
            ((sum_up - sum_dn) / denom).abs()
        } else {
            0.0
        };
        out[i] = c as f32;
    }
    out
}

struct BatchInputs {
    combos: Vec<OttoParams>,
    series_len: usize,
    first_valid: usize,
    cabs: Vec<f32>,
    ott_periods: Vec<i32>,
    ott_percents: Vec<f32>,
    fast: Vec<i32>,
    slow: Vec<i32>,
    coco: Vec<f32>,
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "otto",
                "batch_dev",
                "otto_cuda_batch_dev",
                "60k_x_64combos",
                prep_otto_batch_box,
            )
            .with_inner_iters(6),
            CudaBenchScenario::new(
                "otto",
                "many_series_one_param",
                "otto_cuda_many_series_one_param",
                "512x64k",
                prep_otto_many_series_box,
            )
            .with_inner_iters(4),
        ]
    }

    struct OttoBatchState {
        cuda: CudaOtto,
        d_prices: DeviceBuffer<f32>,
        d_cabs: DeviceBuffer<f32>,
        d_ottp: DeviceBuffer<i32>,
        d_ottk: DeviceBuffer<f32>,
        d_fast: DeviceBuffer<i32>,
        d_slow: DeviceBuffer<i32>,
        d_coco: DeviceBuffer<f32>,
        d_hott: DeviceBuffer<f32>,
        d_lott: DeviceBuffer<f32>,
        len: usize,
        rows: usize,
    }
    impl CudaBenchState for OttoBatchState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_prices,
                    &self.d_cabs,
                    &self.d_ottp,
                    &self.d_ottk,
                    &self.d_fast,
                    &self.d_slow,
                    &self.d_coco,
                    self.len,
                    self.rows,
                    0,
                    &mut self.d_hott,
                    &mut self.d_lott,
                )
                .expect("otto launch batch");
            self.cuda.synchronize().unwrap();
        }
    }

    fn prep_otto_batch() -> OttoBatchState {
        let mut cuda = CudaOtto::new(0).expect("cuda otto");
        cuda.set_policy(CudaOttoPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        });

        let len = 60_000usize;
        let mut price = vec![f32::NAN; len];
        for i in 1..len {
            let x = i as f32;
            price[i] = (x * 0.0011).sin() + 0.0002 * x;
        }

        let sweep = OttoBatchRange {
            ott_period: (2, 33, 5),
            ott_percent: (0.6, 0.6, 0.0),
            fast_vidya: (10, 10, 0),
            slow_vidya: (25, 25, 0),
            correcting_constant: (100000.0, 100000.0, 0.0),
            ma_types: vec!["VAR".into()],
        };
        let inputs = CudaOtto::prepare_batch_inputs(&price, &sweep).expect("inputs");
        let rows = inputs.combos.len();

        let d_prices = DeviceBuffer::from_slice(&price).expect("prices");
        let d_cabs = DeviceBuffer::from_slice(&inputs.cabs).expect("cabs");
        let d_ottp = DeviceBuffer::from_slice(&inputs.ott_periods).expect("ottp");
        let d_ottk = DeviceBuffer::from_slice(&inputs.ott_percents).expect("ottk");
        let d_fast = DeviceBuffer::from_slice(&inputs.fast).expect("fast");
        let d_slow = DeviceBuffer::from_slice(&inputs.slow).expect("slow");
        let d_coco = DeviceBuffer::from_slice(&inputs.coco).expect("coco");
        let d_hott: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len * rows) }.expect("hott");
        let d_lott: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len * rows) }.expect("lott");

        OttoBatchState {
            cuda,
            d_prices,
            d_cabs,
            d_ottp,
            d_ottk,
            d_fast,
            d_slow,
            d_coco,
            d_hott,
            d_lott,
            len,
            rows,
        }
    }
    fn prep_otto_batch_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_otto_batch())
    }

    struct OttoManySeriesState {
        cuda: CudaOtto,
        d_prices_tm: DeviceBuffer<f32>,
        d_hott_tm: DeviceBuffer<f32>,
        d_lott_tm: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        params: OttoParams,
    }
    impl CudaBenchState for OttoManySeriesState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    self.cols,
                    self.rows,
                    self.params.ott_period.unwrap_or(2),
                    self.params.ott_percent.unwrap_or(0.6) as f32,
                    self.params.fast_vidya_length.unwrap_or(10),
                    self.params.slow_vidya_length.unwrap_or(25),
                    self.params.correcting_constant.unwrap_or(100000.0) as f32,
                    &mut self.d_hott_tm,
                    &mut self.d_lott_tm,
                )
                .expect("otto many-series");
            self.cuda.synchronize().unwrap();
        }
    }

    fn prep_otto_many_series() -> OttoManySeriesState {
        let mut cuda = CudaOtto::new(0).expect("cuda otto");
        cuda.set_policy(CudaOttoPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        });
        let cols = 512usize;
        let rows = 64_000usize;
        let mut tm = vec![f32::NAN; cols * rows];
        for s in 0..rows {
            for t in 0..cols {
                let x = (t as f32) + (s as f32) * 0.01;
                tm[t * rows + s] = (x * 0.002).sin() + 0.0001 * x;
            }
        }
        let params = OttoParams::default();

        let d_prices_tm = DeviceBuffer::from_slice(&tm).expect("tm");
        let d_hott_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("hott");
        let d_lott_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("lott");
        OttoManySeriesState {
            cuda,
            d_prices_tm,
            d_hott_tm,
            d_lott_tm,
            cols,
            rows,
            params,
        }
    }
    fn prep_otto_many_series_box() -> Box<dyn CudaBenchState> {
        Box::new(prep_otto_many_series())
    }
}
