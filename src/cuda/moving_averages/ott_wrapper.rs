#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;

use super::{BatchKernelPolicy, ManySeriesKernelPolicy};
use crate::cuda::moving_averages::{
    CudaEma, CudaKama, CudaNama, CudaSma, CudaVpwma, CudaVwma, CudaWilders, CudaZlema,
};
use crate::cuda::moving_averages::{
    CudaMaData, CudaMaDeviceDataRef, CudaMaSelector, CudaMaSelectorError,
};
use crate::cuda::runtime::CudaSession;
use crate::cuda::CudaDeviceSliceF32Ref;
use crate::indicators::ott::{OttBatchRange, OttParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, Function, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

fn ott_batch_var_block_x() -> u32 {
    env::var("OTT_BATCH_VAR_BLOCK_X")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .map(|v| v.clamp(1, 1024))
        .unwrap_or(1)
}

const OTT_VCMO_SHARED_MIN_COMBOS: usize = 96;

#[cfg(test)]
static OTT_SINGLE_PERIOD_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static OTT_SINGLE_PERCENT_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static OTT_BATCH_PERIOD_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static OTT_BATCH_PERCENT_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static OTT_VCMO_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[derive(Default)]
struct CudaOttScratch {
    single_period: Option<DeviceBuffer<i32>>,
    single_percent: Option<DeviceBuffer<f32>>,
    batch_periods: Option<DeviceBuffer<i32>>,
    batch_percents: Option<DeviceBuffer<f32>>,
    batch_len: usize,
    vcmo: Option<DeviceBuffer<f32>>,
    vcmo_len: usize,
}

impl CudaOttScratch {
    fn ensure_single_params(
        &mut self,
        stream: &Stream,
        period: i32,
        percent: f32,
    ) -> Result<(), CudaOttError> {
        if self.single_period.is_none() {
            #[cfg(test)]
            OTT_SINGLE_PERIOD_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            self.single_period = Some(unsafe { DeviceBuffer::uninitialized(1) }?);
        }
        if self.single_percent.is_none() {
            #[cfg(test)]
            OTT_SINGLE_PERCENT_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            self.single_percent = Some(unsafe { DeviceBuffer::uninitialized(1) }?);
        }
        unsafe {
            self.single_period
                .as_mut()
                .expect("single_period")
                .async_copy_from(&[period], stream)?;
            self.single_percent
                .as_mut()
                .expect("single_percent")
                .async_copy_from(&[percent], stream)?;
        }
        Ok(())
    }

    fn ensure_batch_params(
        &mut self,
        stream: &Stream,
        periods: &[i32],
        percents: &[f32],
    ) -> Result<(), CudaOttError> {
        let rows = periods.len();
        if self.batch_len != rows || self.batch_periods.is_none() || self.batch_percents.is_none() {
            #[cfg(test)]
            {
                OTT_BATCH_PERIOD_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
                OTT_BATCH_PERCENT_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            }
            self.batch_periods = Some(unsafe { DeviceBuffer::uninitialized(rows) }?);
            self.batch_percents = Some(unsafe { DeviceBuffer::uninitialized(rows) }?);
            self.batch_len = rows;
        }
        unsafe {
            self.batch_periods
                .as_mut()
                .expect("batch_periods")
                .async_copy_from(periods, stream)?;
            self.batch_percents
                .as_mut()
                .expect("batch_percents")
                .async_copy_from(percents, stream)?;
        }
        Ok(())
    }

    fn ensure_vcmo(&mut self, len: usize) -> Result<(), CudaOttError> {
        if self.vcmo_len != len || self.vcmo.is_none() {
            #[cfg(test)]
            OTT_VCMO_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            self.vcmo = Some(unsafe { DeviceBuffer::uninitialized(len) }?);
            self.vcmo_len = len;
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum CudaOttError {
    #[error("CUDA error: {0}")]
    Cuda(String),
    #[error("out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

impl From<cust::error::CudaError> for CudaOttError {
    fn from(e: cust::error::CudaError) -> Self {
        CudaOttError::Cuda(e.to_string())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaOttPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaOttPolicy {
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

pub struct CudaOtt {
    module: Module,
    stream: Arc<Stream>,
    context: Arc<Context>,
    device_id: u32,
    scratch: RefCell<CudaOttScratch>,
    policy: CudaOttPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaOtt {
    pub fn new(device_id: usize) -> Result<Self, CudaOttError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ott_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = crate::load_cuda_embedded_module!("ott_kernel")?;

        let stream = Arc::new(Stream::new(StreamFlags::NON_BLOCKING, None)?);

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            scratch: RefCell::new(CudaOttScratch::default()),
            policy: CudaOttPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn from_session(session: Arc<CudaSession>) -> Result<Self, CudaOttError> {
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ott_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = crate::load_cuda_embedded_module!("ott_kernel")?;

        Ok(Self {
            module,
            stream: session.stream_arc(),
            context: session.context_arc(),
            device_id: session.device_id(),
            scratch: RefCell::new(CudaOttScratch::default()),
            policy: CudaOttPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaOttError> {
        self.stream.synchronize()?;
        Ok(())
    }

    #[cfg(test)]
    fn debug_reset_scratch_alloc_counters() {
        OTT_SINGLE_PERIOD_ALLOCATIONS.store(0, Ordering::Relaxed);
        OTT_SINGLE_PERCENT_ALLOCATIONS.store(0, Ordering::Relaxed);
        OTT_BATCH_PERIOD_ALLOCATIONS.store(0, Ordering::Relaxed);
        OTT_BATCH_PERCENT_ALLOCATIONS.store(0, Ordering::Relaxed);
        OTT_VCMO_ALLOCATIONS.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn debug_scratch_alloc_counters() -> (usize, usize, usize, usize, usize) {
        (
            OTT_SINGLE_PERIOD_ALLOCATIONS.load(Ordering::Relaxed),
            OTT_SINGLE_PERCENT_ALLOCATIONS.load(Ordering::Relaxed),
            OTT_BATCH_PERIOD_ALLOCATIONS.load(Ordering::Relaxed),
            OTT_BATCH_PERCENT_ALLOCATIONS.load(Ordering::Relaxed),
            OTT_VCMO_ALLOCATIONS.load(Ordering::Relaxed),
        )
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    #[inline]
    fn shared_session(&self) -> Arc<CudaSession> {
        Arc::new(CudaSession::from_parts(
            self.context.clone(),
            self.stream.clone(),
            self.device_id,
        ))
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        mem_get_info().ok()
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaOttError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            let required = required_bytes
                .checked_add(headroom_bytes)
                .ok_or_else(|| CudaOttError::InvalidInput("byte size overflow".into()))?;
            if required > free {
                return Err(CudaOttError::OutOfMemory {
                    required,
                    free,
                    headroom: headroom_bytes,
                });
            }
            return Ok(());
        }
        Ok(())
    }

    #[inline]
    fn memset_nan32_async(&self, dst_ptr_raw: u64, n_elems: usize) -> Result<(), CudaOttError> {
        const QNAN_BITS: u32 = 0x7FC0_0000;
        unsafe {
            use cust::sys::cuMemsetD32Async;
            let err = cuMemsetD32Async(
                dst_ptr_raw as cust::sys::CUdeviceptr,
                QNAN_BITS,
                n_elems,
                self.stream.as_inner(),
            );
            if err != cust::sys::CUresult::CUDA_SUCCESS {
                return Err(CudaOttError::Cuda(format!(
                    "cuMemsetD32Async failed: {:?}",
                    err
                )));
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
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] OTT batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaOtt)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] OTT many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaOtt)).debug_many_logged = true;
                }
            }
        }
    }

    pub fn ott_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &OttBatchRange,
    ) -> Result<DeviceArrayF32, CudaOttError> {
        if prices_f32.is_empty() {
            return Err(CudaOttError::InvalidInput("empty price input".into()));
        }
        let cols = prices_f32.len();

        let mut any_finite = false;
        let mut all_finite = true;
        for &v in prices_f32 {
            if v.is_finite() {
                any_finite = true;
            } else {
                all_finite = false;
            }
        }
        if !any_finite {
            return Err(CudaOttError::InvalidInput("all values are NaN".into()));
        }
        let first_valid = prices_f32.iter().position(|v| v.is_finite()).unwrap_or(0);

        let combos = expand_combos(sweep)?;
        let rows = combos.len();
        let sz_f32 = std::mem::size_of::<f32>();
        let out_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaOttError::InvalidInput("rows * cols overflow".into()))?;
        let prices_bytes = cols
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaOttError::InvalidInput("byte size overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(sz_f32)
            .ok_or_else(|| CudaOttError::InvalidInput("byte size overflow".into()))?;
        let bytes = prices_bytes
            .checked_add(out_bytes)
            .ok_or_else(|| CudaOttError::InvalidInput("byte size overflow".into()))?;
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::will_fit(bytes, headroom)?;

        if cols > i32::MAX as usize {
            return Err(CudaOttError::InvalidInput(
                "series length exceeds kernel limits".into(),
            ));
        }
        if rows > u32::MAX as usize {
            return Err(CudaOttError::InvalidInput(
                "combo count exceeds kernel launch limits".into(),
            ));
        }

        let mut d_prices = unsafe { DeviceBuffer::<f32>::uninitialized(cols) }?;
        unsafe { d_prices.async_copy_from(prices_f32, &self.stream) }?;

        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(out_elems) }?;
        self.memset_nan32_async(d_out.as_device_ptr().as_raw() as u64, out_elems)?;

        let mut f_var: Option<Function> = self.module.get_function("ott_from_var_batch_f32").ok();
        let mut f_var_all_finite: Option<Function> = self
            .module
            .get_function("ott_from_var_batch_f32_all_finite")
            .ok();
        let mut f_var_vcmo_all_finite: Option<Function> = self
            .module
            .get_function("ott_from_vcmo_batch_f32_all_finite")
            .ok();
        let mut f_cmo_all_finite: Option<Function> = self
            .module
            .get_function("cmo9_from_prices_f32_all_finite")
            .ok();
        let mut f_apply = self
            .module
            .get_function("ott_apply_single_f32")
            .map_err(|_| CudaOttError::MissingKernelSymbol {
                name: "ott_apply_single_f32",
            })?;

        let all_var = combos.iter().all(|p| {
            p.ma_type
                .as_deref()
                .unwrap_or("VAR")
                .eq_ignore_ascii_case("VAR")
        });
        if all_var {
            let mut periods_host: Vec<i32> = Vec::with_capacity(rows);
            let mut percents_host: Vec<f32> = Vec::with_capacity(rows);
            for p in &combos {
                let period = p.period.unwrap_or(2);
                let percent = p.percent.unwrap_or(1.4) as f32;
                if period == 0 {
                    return Err(CudaOttError::InvalidInput("period must be positive".into()));
                }
                if !percent.is_finite() {
                    return Err(CudaOttError::InvalidInput("percent must be finite".into()));
                }
                if period > i32::MAX as usize {
                    return Err(CudaOttError::InvalidInput(
                        "period exceeds CUDA i32 range".into(),
                    ));
                }
                periods_host.push(period as i32);
                percents_host.push(percent);
            }

            let mut scratch = self.scratch.borrow_mut();
            scratch.ensure_batch_params(&self.stream, &periods_host, &percents_host)?;

            if all_finite && rows >= OTT_VCMO_SHARED_MIN_COMBOS {
                if let (Some(cmo_func), Some(var_vcmo_func)) =
                    (f_cmo_all_finite.as_mut(), f_var_vcmo_all_finite.as_mut())
                {
                    scratch.ensure_vcmo(cols)?;
                    unsafe {
                        let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                        let mut series_len_i = cols as i32;
                        let mut vcmo_ptr = scratch
                            .vcmo
                            .as_ref()
                            .expect("vcmo")
                            .as_device_ptr()
                            .as_raw();
                        let cmo_args: &mut [*mut c_void] = &mut [
                            &mut prices_ptr as *mut _ as *mut c_void,
                            &mut series_len_i as *mut _ as *mut c_void,
                            &mut vcmo_ptr as *mut _ as *mut c_void,
                        ];
                        let cmo_grid: GridSize = (1, 1, 1).into();
                        let cmo_block: BlockSize = (1, 1, 1).into();
                        self.stream
                            .launch(cmo_func, cmo_grid, cmo_block, 0, cmo_args)
                            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;

                        let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                        let mut vcmo_ptr = scratch
                            .vcmo
                            .as_ref()
                            .expect("vcmo")
                            .as_device_ptr()
                            .as_raw();
                        let mut periods_ptr = scratch
                            .batch_periods
                            .as_ref()
                            .expect("batch_periods")
                            .as_device_ptr()
                            .as_raw();
                        let mut percents_ptr = scratch
                            .batch_percents
                            .as_ref()
                            .expect("batch_percents")
                            .as_device_ptr()
                            .as_raw();
                        let mut series_len_i = cols as i32;
                        let mut n_combos_i = rows as i32;
                        let mut out_ptr = d_out.as_device_ptr().as_raw();
                        let args: &mut [*mut c_void] = &mut [
                            &mut prices_ptr as *mut _ as *mut c_void,
                            &mut vcmo_ptr as *mut _ as *mut c_void,
                            &mut periods_ptr as *mut _ as *mut c_void,
                            &mut percents_ptr as *mut _ as *mut c_void,
                            &mut series_len_i as *mut _ as *mut c_void,
                            &mut n_combos_i as *mut _ as *mut c_void,
                            &mut out_ptr as *mut _ as *mut c_void,
                        ];
                        let block_x = ott_batch_var_block_x();
                        let grid_x = ((rows as u32).saturating_add(block_x - 1)) / block_x;
                        let grid: GridSize = (grid_x.max(1), 1, 1).into();
                        let block: BlockSize = (block_x, 1, 1).into();
                        self.stream
                            .launch(var_vcmo_func, grid, block, 0, args)
                            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
                    }

                    self.stream.synchronize()?;
                    return Ok(DeviceArrayF32 {
                        buf: d_out,
                        rows,
                        cols,
                    });
                }
            }

            let var_func = if all_finite {
                if f_var_all_finite.is_some() {
                    f_var_all_finite.as_mut()
                } else {
                    f_var.as_mut()
                }
            } else {
                f_var.as_mut()
            };
            if let Some(func) = var_func {
                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut periods_ptr = scratch
                        .batch_periods
                        .as_ref()
                        .expect("batch_periods")
                        .as_device_ptr()
                        .as_raw();
                    let mut percents_ptr = scratch
                        .batch_percents
                        .as_ref()
                        .expect("batch_percents")
                        .as_device_ptr()
                        .as_raw();
                    let mut series_len_i = cols as i32;
                    let mut n_combos_i = rows as i32;
                    let mut out_ptr = d_out.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut percents_ptr as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut n_combos_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    let block_x = ott_batch_var_block_x();
                    let grid_x = ((rows as u32).saturating_add(block_x - 1)) / block_x;
                    let grid: GridSize = (grid_x.max(1), 1, 1).into();
                    let block: BlockSize = (block_x, 1, 1).into();
                    self.stream
                        .launch(func, grid, block, 0, args)
                        .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
                }

                self.stream.synchronize()?;
                return Ok(DeviceArrayF32 {
                    buf: d_out,
                    rows,
                    cols,
                });
            }
        }

        let selector = CudaMaSelector::from_session(self.shared_session());
        let device_selector = selector.device_native();
        let prices_view = unsafe {
            CudaDeviceSliceF32Ref::from_raw_parts(
                d_prices.as_device_ptr().as_raw(),
                cols,
                self.device_id,
            )
            .map_err(|e| CudaOttError::InvalidInput(e.to_string()))?
        };

        let selector_dev = |ma_type: &str, period: usize| -> Result<DeviceArrayF32, CudaOttError> {
            match device_selector.ma_to_device_ref(
                ma_type,
                CudaMaDeviceDataRef::Slice(prices_view),
                first_valid,
                period,
            ) {
                Ok(dev) => Ok(dev),
                Err(CudaMaSelectorError::Unsupported(_)) => selector
                    .ma_to_device(ma_type, CudaMaData::SliceF32(prices_f32), period)
                    .map_err(|e| CudaOttError::Cuda(e.to_string())),
                Err(e) => Err(CudaOttError::Cuda(e.to_string())),
            }
        };

        for (row_idx, p) in combos.iter().enumerate() {
            let period = p.period.unwrap_or(2);
            let percent = p.percent.unwrap_or(1.4) as f32;
            let ma_type = p.ma_type.as_deref().unwrap_or("VAR");
            let row_offset = row_idx
                .checked_mul(cols)
                .ok_or_else(|| CudaOttError::InvalidInput("row offset overflow".into()))?;
            let out_row_ptr = unsafe { d_out.as_device_ptr().offset(row_offset as isize) };

            if ma_type.eq_ignore_ascii_case("VAR") {
                if let Some(ref mut func) = f_var {
                    let mut scratch = self.scratch.borrow_mut();
                    scratch.ensure_single_params(&self.stream, period as i32, percent)?;

                    unsafe {
                        let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                        let mut periods_ptr = scratch
                            .single_period
                            .as_ref()
                            .expect("single_period")
                            .as_device_ptr()
                            .as_raw();
                        let mut percents_ptr = scratch
                            .single_percent
                            .as_ref()
                            .expect("single_percent")
                            .as_device_ptr()
                            .as_raw();
                        let mut series_len_i = cols as i32;
                        let mut n_combos_i = 1i32;
                        let mut out_ptr = out_row_ptr.as_raw();
                        let args: &mut [*mut c_void] = &mut [
                            &mut prices_ptr as *mut _ as *mut c_void,
                            &mut periods_ptr as *mut _ as *mut c_void,
                            &mut percents_ptr as *mut _ as *mut c_void,
                            &mut series_len_i as *mut _ as *mut c_void,
                            &mut n_combos_i as *mut _ as *mut c_void,
                            &mut out_ptr as *mut _ as *mut c_void,
                        ];
                        let grid: GridSize = (1, 1, 1).into();
                        let block: BlockSize = (1, 1, 1).into();
                        self.stream
                            .launch(func, grid, block, 0, args)
                            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
                    }
                } else {
                    let dev = selector_dev("VAR", period)?;
                    self.launch_apply_single(
                        &mut f_apply,
                        &dev.buf,
                        cols,
                        percent,
                        out_row_ptr.as_raw(),
                    )?;

                    self.stream.synchronize()?;
                }
            } else {
                let dev = selector_dev(ma_type, period)?;
                self.launch_apply_single(
                    &mut f_apply,
                    &dev.buf,
                    cols,
                    percent,
                    out_row_ptr.as_raw(),
                )?;
                self.stream.synchronize()?;
            }
        }

        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn ott_batch_dev_from_device_prices(
        &self,
        d_prices: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        sweep: &OttBatchRange,
    ) -> Result<DeviceArrayF32, CudaOttError> {
        if series_len == 0 {
            return Err(CudaOttError::InvalidInput("empty price input".into()));
        }
        if first_valid >= series_len {
            return Err(CudaOttError::InvalidInput(format!(
                "invalid first_valid {} for series length {}",
                first_valid, series_len
            )));
        }
        if series_len > i32::MAX as usize {
            return Err(CudaOttError::InvalidInput(
                "series length exceeds kernel limits".into(),
            ));
        }

        let combos = expand_combos(sweep)?;
        let rows = combos.len();
        if rows == 0 {
            return Err(CudaOttError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        if rows > u32::MAX as usize {
            return Err(CudaOttError::InvalidInput(
                "combo count exceeds kernel launch limits".into(),
            ));
        }

        let max_period = combos
            .iter()
            .map(|combo| combo.period.unwrap_or(2))
            .max()
            .unwrap_or(0);
        if max_period == 0 || max_period > series_len {
            return Err(CudaOttError::InvalidInput(format!(
                "invalid max period {} for series length {}",
                max_period, series_len
            )));
        }
        if series_len.saturating_sub(first_valid) < max_period {
            return Err(CudaOttError::InvalidInput(format!(
                "not enough valid data after first_valid {} for max period {}",
                first_valid, max_period
            )));
        }

        let out_elems = rows
            .checked_mul(series_len)
            .ok_or_else(|| CudaOttError::InvalidInput("rows * cols overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaOttError::InvalidInput("byte size overflow".into()))?;
        let headroom = env::var("CUDA_MEM_HEADROOM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024 * 1024);
        Self::will_fit(out_bytes, headroom)?;

        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(out_elems) }?;
        self.memset_nan32_async(d_out.as_device_ptr().as_raw() as u64, out_elems)?;

        let mut f_var: Option<Function> = self.module.get_function("ott_from_var_batch_f32").ok();
        let mut f_apply = self
            .module
            .get_function("ott_apply_single_f32")
            .map_err(|_| CudaOttError::MissingKernelSymbol {
                name: "ott_apply_single_f32",
            })?;

        let all_var = combos.iter().all(|p| {
            p.ma_type
                .as_deref()
                .unwrap_or("VAR")
                .eq_ignore_ascii_case("VAR")
        });
        if all_var {
            let mut periods_host: Vec<i32> = Vec::with_capacity(rows);
            let mut percents_host: Vec<f32> = Vec::with_capacity(rows);
            for p in &combos {
                let period = p.period.unwrap_or(2);
                let percent = p.percent.unwrap_or(1.4) as f32;
                if period == 0 {
                    return Err(CudaOttError::InvalidInput("period must be positive".into()));
                }
                if !percent.is_finite() {
                    return Err(CudaOttError::InvalidInput("percent must be finite".into()));
                }
                if period > i32::MAX as usize {
                    return Err(CudaOttError::InvalidInput(
                        "period exceeds CUDA i32 range".into(),
                    ));
                }
                periods_host.push(period as i32);
                percents_host.push(percent);
            }

            if let Some(func) = f_var.as_mut() {
                let mut scratch = self.scratch.borrow_mut();
                scratch.ensure_batch_params(&self.stream, &periods_host, &percents_host)?;

                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut periods_ptr = scratch
                        .batch_periods
                        .as_ref()
                        .expect("batch_periods")
                        .as_device_ptr()
                        .as_raw();
                    let mut percents_ptr = scratch
                        .batch_percents
                        .as_ref()
                        .expect("batch_percents")
                        .as_device_ptr()
                        .as_raw();
                    let mut series_len_i = series_len as i32;
                    let mut n_combos_i = rows as i32;
                    let mut out_ptr = d_out.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut percents_ptr as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut n_combos_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    let block_x = ott_batch_var_block_x();
                    let grid_x = ((rows as u32).saturating_add(block_x - 1)) / block_x;
                    let grid: GridSize = (grid_x.max(1), 1, 1).into();
                    let block: BlockSize = (block_x, 1, 1).into();
                    self.stream
                        .launch(func, grid, block, 0, args)
                        .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
                }

                return Ok(DeviceArrayF32 {
                    buf: d_out,
                    rows,
                    cols: series_len,
                });
            }
        }

        let selector = CudaMaSelector::from_session(self.shared_session());
        let device_selector = selector.device_native();
        let prices_view = unsafe {
            CudaDeviceSliceF32Ref::from_raw_parts(
                d_prices.as_device_ptr().as_raw(),
                series_len,
                self.device_id,
            )
            .map_err(|e| CudaOttError::InvalidInput(e.to_string()))?
        };
        let selector_data = CudaMaDeviceDataRef::Slice(prices_view);
        let sweep_plan = device_selector
            .create_sweep_plan(sweep.period.0, sweep.period.1, sweep.period.2)
            .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
        let mut period_row_by_value = HashMap::with_capacity(sweep_plan.len());
        for (row_idx, &period) in sweep_plan.periods().iter().enumerate() {
            period_row_by_value.insert(period, row_idx);
        }
        let mut selector_ma_rows: HashMap<String, DeviceArrayF32> = HashMap::new();
        for ma_type in combos
            .iter()
            .filter_map(|combo| combo.ma_type.as_deref())
            .filter(|ma_type| !ma_type.eq_ignore_ascii_case("VAR"))
        {
            let key = ma_type.to_ascii_lowercase();
            if selector_ma_rows.contains_key(&key) {
                continue;
            }
            let dev = device_selector
                .ma_sweep_plan_to_device_ref(ma_type, selector_data, first_valid, &sweep_plan)
                .map_err(|e| match e {
                    CudaMaSelectorError::Unsupported(reason) => CudaOttError::InvalidInput(
                        format!(
                            "ott device path does not support ma_type '{}' without host fallback: {}",
                            ma_type, reason
                        ),
                    ),
                    other => CudaOttError::Cuda(other.to_string()),
                })?;
            selector_ma_rows.insert(key, dev);
        }

        for (row_idx, combo) in combos.iter().enumerate() {
            let period = combo.period.unwrap_or(2);
            let percent = combo.percent.unwrap_or(1.4) as f32;
            let ma_type = combo.ma_type.as_deref().unwrap_or("VAR");
            let row_offset = row_idx
                .checked_mul(series_len)
                .ok_or_else(|| CudaOttError::InvalidInput("row offset overflow".into()))?;
            let out_row_ptr = unsafe { d_out.as_device_ptr().offset(row_offset as isize) };

            if ma_type.eq_ignore_ascii_case("VAR") {
                let func = f_var.as_mut().ok_or_else(|| {
                    CudaOttError::InvalidInput(
                        "ott VAR device path requires ott_from_var_batch_f32 kernel".into(),
                    )
                })?;
                let mut scratch = self.scratch.borrow_mut();
                scratch.ensure_single_params(&self.stream, period as i32, percent)?;
                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut periods_ptr = scratch
                        .single_period
                        .as_ref()
                        .expect("single_period")
                        .as_device_ptr()
                        .as_raw();
                    let mut percents_ptr = scratch
                        .single_percent
                        .as_ref()
                        .expect("single_percent")
                        .as_device_ptr()
                        .as_raw();
                    let mut series_len_i = series_len as i32;
                    let mut n_combos_i = 1i32;
                    let mut out_ptr = out_row_ptr.as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut percents_ptr as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut n_combos_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    let grid: GridSize = (1, 1, 1).into();
                    let block: BlockSize = (1, 1, 1).into();
                    self.stream
                        .launch(func, grid, block, 0, args)
                        .map_err(|e| CudaOttError::Cuda(e.to_string()))?;
                }
            } else {
                let ma_rows = selector_ma_rows
                    .get(&ma_type.to_ascii_lowercase())
                    .ok_or_else(|| {
                        CudaOttError::InvalidInput(format!(
                            "missing borrowed-device MA sweep rows for ma_type '{}'",
                            ma_type
                        ))
                    })?;
                let row_idx_in_ma = period_row_by_value.get(&period).copied().ok_or_else(|| {
                    CudaOttError::InvalidInput(format!(
                        "period {} missing from OTT borrowed-device sweep plan",
                        period
                    ))
                })?;
                let ma_row_ptr = unsafe {
                    ma_rows
                        .buf
                        .as_device_ptr()
                        .add(row_idx_in_ma * series_len)
                        .as_raw()
                };
                self.launch_apply_single_from_ptr(
                    &mut f_apply,
                    ma_row_ptr,
                    series_len,
                    percent,
                    out_row_ptr.as_raw(),
                )?;
            }
        }

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols: series_len,
        })
    }

    fn launch_apply_single(
        &self,
        func: &mut Function,
        d_ma: &DeviceBuffer<f32>,
        len: usize,
        percent: f32,
        out_ptr_raw: u64,
    ) -> Result<(), CudaOttError> {
        self.launch_apply_single_from_ptr(
            func,
            d_ma.as_device_ptr().as_raw(),
            len,
            percent,
            out_ptr_raw,
        )
    }

    fn launch_apply_single_from_ptr(
        &self,
        func: &mut Function,
        ma_ptr_raw: u64,
        len: usize,
        percent: f32,
        out_ptr_raw: u64,
    ) -> Result<(), CudaOttError> {
        unsafe {
            let mut ma_ptr = ma_ptr_raw;
            let mut series_len_i = len as i32;
            let mut pct = percent;
            let mut out_ptr = out_ptr_raw;
            let args: &mut [*mut c_void] = &mut [
                &mut ma_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut pct as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let grid: GridSize = (1, 1, 1).into();
            let block: BlockSize = (1, 1, 1).into();
            self.stream.launch(func, grid, block, 0, args)?;
        }
        Ok(())
    }

    pub fn ott_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &OttParams,
    ) -> Result<DeviceArrayF32, CudaOttError> {
        if cols == 0 || rows == 0 {
            return Err(CudaOttError::InvalidInput("empty input".into()));
        }
        let expected_elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaOttError::InvalidInput("rows * cols overflow".into()))?;
        if data_tm_f32.len() != expected_elems {
            return Err(CudaOttError::InvalidInput("shape mismatch".into()));
        }
        if cols > i32::MAX as usize || rows > i32::MAX as usize {
            return Err(CudaOttError::InvalidInput(
                "rows/cols exceed kernel launch limits".into(),
            ));
        }
        let period = params.period.unwrap_or(2);
        let percent = params.percent.unwrap_or(1.4) as f32;
        let ma_type = params.ma_type.as_deref().unwrap_or("VAR");

        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(expected_elems) }?;
        self.memset_nan32_async(d_out.as_device_ptr().as_raw() as u64, expected_elems)?;

        if ma_type.eq_ignore_ascii_case("VAR") {
            let mut d_in = unsafe { DeviceBuffer::<f32>::uninitialized(expected_elems) }?;
            unsafe { d_in.async_copy_from(data_tm_f32, &self.stream) }?;
            let mut func = self
                .module
                .get_function("ott_from_var_many_series_one_param_f32")
                .map_err(|_| CudaOttError::MissingKernelSymbol {
                    name: "ott_from_var_many_series_one_param_f32",
                })?;
            unsafe {
                let mut in_ptr = d_in.as_device_ptr().as_raw();
                let mut cols_i = cols as i32;
                let mut rows_i = rows as i32;
                let mut period_i = period as i32;
                let mut pct = percent;
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut in_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut pct as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                let grid: GridSize = ((cols as u32).max(1), 1, 1).into();
                let block: BlockSize = (1, 1, 1).into();
                self.stream.launch(&mut func, grid, block, 0, args)?;
            }
        } else {
            let ma_dev = if ma_type.eq_ignore_ascii_case("EMA") {
                let p = crate::indicators::moving_averages::ema::EmaParams {
                    period: Some(period),
                };
                CudaEma::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .ema_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("SMA") {
                let p = crate::indicators::moving_averages::sma::SmaParams {
                    period: Some(period),
                };
                CudaSma::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .sma_multi_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("ZLEMA") {
                let p = crate::indicators::moving_averages::zlema::ZlemaParams {
                    period: Some(period),
                };
                CudaZlema::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .zlema_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("WILDERS")
                || ma_type.eq_ignore_ascii_case("WWMA")
            {
                let p = crate::indicators::moving_averages::wilders::WildersParams {
                    period: Some(period),
                };
                CudaWilders::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .wilders_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map(|h| super::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("KAMA") {
                let p = crate::indicators::moving_averages::kama::KamaParams {
                    period: Some(period),
                };
                CudaKama::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .kama_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map(|h| super::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("VWMA") {
                return Err(CudaOttError::InvalidInput(
                    "vwma requires candles+volume; not supported in this path".into(),
                ));
            } else if ma_type.eq_ignore_ascii_case("VPWMA") {
                let p = crate::indicators::moving_averages::vpwma::VpwmaParams {
                    period: Some(period),
                    power: Some(0.382),
                };
                CudaVpwma::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .vpwma_multi_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map(|h| super::DeviceArrayF32 {
                        buf: h.buf,
                        rows: h.rows,
                        cols: h.cols,
                    })
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("NAMA") {
                let p = crate::indicators::moving_averages::nama::NamaParams {
                    period: Some(period),
                    ..Default::default()
                };
                CudaNama::new(self.device_id as usize)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
                    .nama_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, &p)
                    .map_err(|e| CudaOttError::Cuda(e.to_string()))?
            } else if ma_type.eq_ignore_ascii_case("VWMA") || ma_type.eq_ignore_ascii_case("VWAP") {
                return Err(CudaOttError::InvalidInput(
                    "volume/anchor-based MA not supported in ott_many_series path".into(),
                ));
            } else {
                return Err(CudaOttError::InvalidInput(format!(
                    "unsupported ma_type '{}' for OTT CUDA many-series",
                    ma_type
                )));
            };

            let mut func = self
                .module
                .get_function("ott_many_series_one_param_f32")
                .map_err(|_| CudaOttError::MissingKernelSymbol {
                    name: "ott_many_series_one_param_f32",
                })?;
            unsafe {
                let mut in_ptr = ma_dev.buf.as_device_ptr().as_raw();
                let mut cols_i = cols as i32;
                let mut rows_i = rows as i32;
                let mut pct = percent;
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut in_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut pct as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                let grid: GridSize = ((cols as u32).max(1), 1, 1).into();
                let block: BlockSize = (1, 1, 1).into();
                self.stream.launch(&mut func, grid, block, 0, args)?;
            }
        }

        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::ott::OttParams;
    use std::ffi::c_void;

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let params_bytes = PARAM_SWEEP * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + params_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct OttBatchVarDevState {
        cuda: CudaOtt,
        d_prices: DeviceBuffer<f32>,
        d_vcmo: Option<DeviceBuffer<f32>>,
        d_periods: DeviceBuffer<i32>,
        d_percents: DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for OttBatchVarDevState {
        fn launch(&mut self) {
            let out_elems = self.series_len * self.n_combos;
            self.cuda
                .memset_nan32_async(self.d_out.as_device_ptr().as_raw() as u64, out_elems)
                .expect("ott memset nan");

            if let (Some(d_vcmo), Ok(mut func)) = (
                self.d_vcmo.as_ref(),
                self.cuda
                    .module
                    .get_function("ott_from_vcmo_batch_f32_all_finite"),
            ) {
                unsafe {
                    let mut prices_ptr = self.d_prices.as_device_ptr().as_raw();
                    let mut vcmo_ptr = d_vcmo.as_device_ptr().as_raw();
                    let mut periods_ptr = self.d_periods.as_device_ptr().as_raw();
                    let mut percents_ptr = self.d_percents.as_device_ptr().as_raw();
                    let mut series_len_i = self.series_len as i32;
                    let mut n_combos_i = self.n_combos as i32;
                    let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut vcmo_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut percents_ptr as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut n_combos_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    let block_x = ott_batch_var_block_x();
                    let grid_x = ((self.n_combos as u32).saturating_add(block_x - 1)) / block_x;
                    let grid: GridSize = (grid_x.max(1), 1, 1).into();
                    let block: BlockSize = (block_x, 1, 1).into();
                    self.cuda
                        .stream
                        .launch(&mut func, grid, block, 0, args)
                        .expect("ott_from_vcmo_batch_f32_all_finite launch");
                }
            } else {
                let mut func = self
                    .cuda
                    .module
                    .get_function("ott_from_var_batch_f32_all_finite")
                    .or_else(|_| self.cuda.module.get_function("ott_from_var_batch_f32"))
                    .expect("ott_from_var_batch_f32(_all_finite)");
                unsafe {
                    let mut prices_ptr = self.d_prices.as_device_ptr().as_raw();
                    let mut periods_ptr = self.d_periods.as_device_ptr().as_raw();
                    let mut percents_ptr = self.d_percents.as_device_ptr().as_raw();
                    let mut series_len_i = self.series_len as i32;
                    let mut n_combos_i = self.n_combos as i32;
                    let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                    let args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut percents_ptr as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut n_combos_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    let block_x = ott_batch_var_block_x();
                    let grid_x = ((self.n_combos as u32).saturating_add(block_x - 1)) / block_x;
                    let grid: GridSize = (grid_x.max(1), 1, 1).into();
                    let block: BlockSize = (block_x, 1, 1).into();
                    self.cuda
                        .stream
                        .launch(&mut func, grid, block, 0, args)
                        .expect("ott_from_var_batch_f32 launch");
                }
            }
            self.cuda.stream.synchronize().expect("ott sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaOtt::new(0).expect("cuda ott");
        let prices = gen_series(ONE_SERIES_LEN);
        let sweep = crate::indicators::ott::OttBatchRange {
            period: (2, 2 + PARAM_SWEEP - 1, 1),
            percent: (1.4, 1.4, 0.0),
            ma_types: vec!["VAR".to_string()],
        };
        let combos = expand_combos(&sweep).expect("ott expand combos");
        let n_combos = combos.len();
        let periods_host: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(2) as i32)
            .collect();
        let percents_host: Vec<f32> = combos
            .iter()
            .map(|p| p.percent.unwrap_or(1.4) as f32)
            .collect();

        let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
        let d_vcmo: Option<DeviceBuffer<f32>> =
            if let Ok(mut cmo_func) = cuda.module.get_function("cmo9_from_prices_f32_all_finite") {
                let mut tmp: DeviceBuffer<f32> =
                    unsafe { DeviceBuffer::uninitialized(prices.len()) }.expect("d_vcmo");
                unsafe {
                    let mut prices_ptr = d_prices.as_device_ptr().as_raw();
                    let mut series_len_i = prices.len() as i32;
                    let mut vcmo_ptr = tmp.as_device_ptr().as_raw();
                    let cmo_args: &mut [*mut c_void] = &mut [
                        &mut prices_ptr as *mut _ as *mut c_void,
                        &mut series_len_i as *mut _ as *mut c_void,
                        &mut vcmo_ptr as *mut _ as *mut c_void,
                    ];
                    let cmo_grid: GridSize = (1, 1, 1).into();
                    let cmo_block: BlockSize = (1, 1, 1).into();
                    cuda.stream
                        .launch(&mut cmo_func, cmo_grid, cmo_block, 0, cmo_args)
                        .expect("cmo9_from_prices_f32_all_finite launch");
                }
                Some(tmp)
            } else {
                None
            };
        let d_periods = DeviceBuffer::from_slice(&periods_host).expect("d_periods");
        let d_percents = DeviceBuffer::from_slice(&percents_host).expect("d_percents");
        let d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prices.len().checked_mul(n_combos).expect("out size"))
        }
        .expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(OttBatchVarDevState {
            cuda,
            d_prices,
            d_vcmo,
            d_periods,
            d_percents,
            series_len: prices.len(),
            n_combos,
            d_out,
        })
    }

    struct OttBorrowedDeviceBatchState {
        cuda: CudaOtt,
        d_prices: DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        sweep: crate::indicators::ott::OttBatchRange,
    }
    impl CudaBenchState for OttBorrowedDeviceBatchState {
        fn launch(&mut self) {
            let out = self
                .cuda
                .ott_batch_dev_from_device_prices(
                    &self.d_prices,
                    self.series_len,
                    self.first_valid,
                    &self.sweep,
                )
                .expect("ott_batch_dev_from_device_prices");
            std::hint::black_box((out.rows, out.cols));
            self.cuda.stream.synchronize().expect("ott borrowed sync");
        }
    }

    fn prep_one_series_many_params_borrowed_device() -> Box<dyn CudaBenchState> {
        let cuda = CudaOtt::new(0).expect("cuda ott");
        let prices = gen_series(ONE_SERIES_LEN);
        let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
        let first_valid = prices.iter().position(|v| v.is_finite()).unwrap_or(0);
        let sweep = crate::indicators::ott::OttBatchRange {
            period: (2, 2 + PARAM_SWEEP - 1, 1),
            percent: (1.4, 1.4, 0.0),
            ma_types: vec!["VAR".to_string()],
        };
        cuda.stream.synchronize().expect("sync after borrowed prep");

        Box::new(OttBorrowedDeviceBatchState {
            cuda,
            d_prices,
            series_len: ONE_SERIES_LEN,
            first_valid,
            sweep,
        })
    }

    struct OttManyVarDevState {
        cuda: CudaOtt,
        d_in: DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        period: usize,
        percent: f32,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for OttManyVarDevState {
        fn launch(&mut self) {
            let out_elems = self.cols * self.rows;
            self.cuda
                .memset_nan32_async(self.d_out.as_device_ptr().as_raw() as u64, out_elems)
                .expect("ott memset nan");

            let mut func = self
                .cuda
                .module
                .get_function("ott_from_var_many_series_one_param_f32")
                .expect("ott_from_var_many_series_one_param_f32");
            unsafe {
                let mut in_ptr = self.d_in.as_device_ptr().as_raw();
                let mut cols_i = self.cols as i32;
                let mut rows_i = self.rows as i32;
                let mut period_i = self.period as i32;
                let mut pct = self.percent;
                let mut out_ptr = self.d_out.as_device_ptr().as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut in_ptr as *mut _ as *mut c_void,
                    &mut cols_i as *mut _ as *mut c_void,
                    &mut rows_i as *mut _ as *mut c_void,
                    &mut period_i as *mut _ as *mut c_void,
                    &mut pct as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                let grid: GridSize = ((self.cols as u32).max(1), 1, 1).into();
                let block: BlockSize = (1, 1, 1).into();
                self.cuda
                    .stream
                    .launch(&mut func, grid, block, 0, args)
                    .expect("ott_from_var_many_series_one_param_f32 launch");
            }
            self.cuda.stream.synchronize().expect("ott sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaOtt::new(0).expect("cuda ott");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = OttParams {
            period: Some(16),
            percent: Some(1.4),
            ma_type: Some("VAR".to_string()),
        };

        let d_in = DeviceBuffer::from_slice(&data_tm).expect("d_in");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols.checked_mul(rows).expect("out size")) }
                .expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(OttManyVarDevState {
            cuda,
            d_in,
            cols,
            rows,
            period: params.period.unwrap_or(16),
            percent: params.percent.unwrap_or(1.4) as f32,
            d_out,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "ott",
                "one_series_many_params",
                "ott_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "ott",
                "one_series_many_params",
                "ott_cuda_batch_dev_from_device_prices",
                "1m_x_250",
                prep_one_series_many_params_borrowed_device,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "ott",
                "many_series_one_param",
                "ott_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}

fn expand_combos(range: &OttBatchRange) -> Result<Vec<OttParams>, CudaOttError> {
    fn axis_usize(axis: (usize, usize, usize)) -> Result<Vec<usize>, CudaOttError> {
        let (start, end, step) = axis;
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            return Ok((start..=end).step_by(step.max(1)).collect());
        }
        let mut out = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            out.push(x as usize);
            x -= st;
        }
        if out.is_empty() {
            return Err(CudaOttError::InvalidInput(format!(
                "Invalid range: start={}, end={}, step={}",
                start, end, step
            )));
        }
        Ok(out)
    }
    fn axis_f64(axis: (f64, f64, f64)) -> Result<Vec<f64>, CudaOttError> {
        let (start, end, step) = axis;
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        if start < end {
            let mut out = Vec::new();
            let mut x = start;
            let st = step.abs();
            while x <= end + 1e-12 {
                out.push(x);
                x += st;
            }
            if out.is_empty() {
                return Err(CudaOttError::InvalidInput(format!(
                    "Invalid range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            return Ok(out);
        }
        let mut out = Vec::new();
        let mut x = start;
        let st = step.abs();
        while x + 1e-12 >= end {
            out.push(x);
            x -= st;
        }
        if out.is_empty() {
            return Err(CudaOttError::InvalidInput(format!(
                "Invalid range: start={}, end={}, step={}",
                start, end, step
            )));
        }
        Ok(out)
    }

    let periods = axis_usize(range.period)?;
    let percents = axis_f64(range.percent)?;
    let types = if range.ma_types.is_empty() {
        vec!["VAR".to_string()]
    } else {
        range.ma_types.clone()
    };
    let cap = periods
        .len()
        .checked_mul(percents.len())
        .and_then(|x| x.checked_mul(types.len()))
        .ok_or_else(|| CudaOttError::InvalidInput("range size overflow".into()))?;
    let mut combos = Vec::with_capacity(cap);
    for &p in &periods {
        for &q in &percents {
            for t in &types {
                combos.push(OttParams {
                    period: Some(p),
                    percent: Some(q),
                    ma_type: Some(t.clone()),
                });
            }
        }
    }
    if combos.is_empty() {
        return Err(CudaOttError::InvalidInput(
            "no parameter combinations".into(),
        ));
    }
    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaRuntime;

    #[test]
    fn borrowed_device_ott_reuses_wrapper_scratch_for_repeated_var_sweeps() {
        if !crate::cuda::cuda_available() {
            return;
        }

        let len = 2048usize;
        let prices: Vec<f32> = (0..len)
            .map(|i| {
                let x = i as f32;
                (x * 0.00123).sin() + 0.00017 * x
            })
            .collect();
        let first_valid = prices.iter().position(|v| v.is_finite()).unwrap_or(0);
        let sweep = OttBatchRange {
            period: (2, 2 + OTT_VCMO_SHARED_MIN_COMBOS - 1, 1),
            percent: (1.4, 1.4, 0.0),
            ma_types: vec!["VAR".to_string()],
        };
        let runtime = CudaRuntime::new(0).expect("runtime");
        let d_prices = runtime.upload_f32(&prices).expect("upload");
        let cuda = CudaOtt::new(0).expect("CudaOtt::new");

        CudaOtt::debug_reset_scratch_alloc_counters();
        let first = cuda
            .ott_batch_dev_from_device_prices(d_prices.buffer(), len, first_valid, &sweep)
            .expect("first borrowed-device ott sweep");
        cuda.synchronize().expect("sync first borrowed ott");
        let first_counts = CudaOtt::debug_scratch_alloc_counters();
        assert_eq!(first_counts.2, 1, "expected one batch-period allocation");
        assert_eq!(first_counts.3, 1, "expected one batch-percent allocation");
        assert_eq!(
            first_counts.4, 0,
            "borrowed-device OTT path currently should not allocate VCMO scratch"
        );

        let second = cuda
            .ott_batch_dev_from_device_prices(d_prices.buffer(), len, first_valid, &sweep)
            .expect("second borrowed-device ott sweep");
        cuda.synchronize().expect("sync second borrowed ott");
        let second_counts = CudaOtt::debug_scratch_alloc_counters();
        assert_eq!(
            second_counts, first_counts,
            "repeated same-shape borrowed-device ott sweep should reuse scratch buffers"
        );

        let mut first_host = vec![0f32; first.len()];
        let mut second_host = vec![0f32; second.len()];
        first.buf.copy_to(&mut first_host).expect("copy first");
        second.buf.copy_to(&mut second_host).expect("copy second");
        assert_eq!(first.rows, second.rows);
        assert_eq!(first.cols, second.cols);
        assert_eq!(first_host, second_host);
    }
}
