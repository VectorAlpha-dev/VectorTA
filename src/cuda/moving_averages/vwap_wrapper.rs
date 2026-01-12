#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32 as SharedDeviceArrayF32;
use crate::indicators::moving_averages::vwap::{
    expand_grid_vwap, first_valid_vwap_index, parse_anchor, VwapBatchRange, VwapParams,
};
use cust::context::{CacheConfig, Context};
use cust::device::Device;
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::convert::TryFrom;
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaVwapError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("out of memory: required={required}B, free={free}B, headroom={headroom}B")]
    OutOfMemory {
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("launch config too large: grid=({gx},{gy},{gz}), block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
    #[error("device mismatch: buffer device={buf}, current device={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct VwapDeviceArrayF32 {
    pub buf: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchThreadsPerOutput {
    One,
    Two,
}

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
pub struct CudaVwapPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaVwapPolicy {
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

pub struct CudaVwap {
    module: Module,
    stream: Stream,
    _context: Arc<Context>,
    device_id: u32,
    policy: CudaVwapPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

struct PreparedBatch {
    combos: Vec<VwapParams>,
    counts: Vec<i32>,
    unit_codes: Vec<i32>,
    divisors: Vec<i64>,
    first_valids: Vec<i32>,
    month_ids: Option<Vec<i32>>,
    series_len: usize,
}

impl CudaVwap {
    pub fn new(device_id: usize) -> Result<Self, CudaVwapError> {
        cust::init(CudaFlags::empty())?;

        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vwap_kernel.ptx"));

        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))?;

        if let Ok(mut f) = module.get_function("vwap_batch_f32") {
            let _ = f.set_cache_config(CacheConfig::PreferL1);
        }
        if let Ok(mut f) = module.get_function("vwap_multi_series_one_param_f32") {
            let _ = f.set_cache_config(CacheConfig::PreferL1);
        }

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaVwapPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn synchronize(&self) -> Result<(), CudaVwapError> {
        Ok(())
    }

    pub fn set_policy(&mut self, policy: CudaVwapPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaVwapPolicy {
        &self.policy
    }
    pub fn context_arc(&self) -> Arc<Context> {
        self._context.clone()
    }
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
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
                    eprintln!("[DEBUG] VWAP batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaVwap)).debug_batch_logged = true;
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
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario =
                    std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] VWAP many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaVwap)).debug_many_logged = true;
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
        if let Some((free, _)) = Self::device_mem_info() {
            return required_bytes.saturating_add(headroom_bytes) <= free;
        }
        true
    }
    #[inline]
    fn ensure_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaVwapError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) > free {
                return Err(CudaVwapError::OutOfMemory {
                    required: required_bytes,
                    free,
                    headroom: headroom_bytes,
                });
            }
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        timestamps: &[i64],
        volumes: &[f64],
        prices: &[f64],
        sweep: &VwapBatchRange,
    ) -> Result<PreparedBatch, CudaVwapError> {
        if timestamps.len() != volumes.len() || volumes.len() != prices.len() {
            return Err(CudaVwapError::InvalidInput(
                "timestamps, volumes, and prices must have equal length".into(),
            ));
        }
        if timestamps.is_empty() {
            return Err(CudaVwapError::InvalidInput("empty input series".into()));
        }

        let combos = expand_grid_vwap(sweep);
        if combos.is_empty() {
            return Err(CudaVwapError::InvalidInput(
                "no parameter combinations after anchor expansion".into(),
            ));
        }

        let mut counts = Vec::with_capacity(combos.len());
        let mut unit_codes = Vec::with_capacity(combos.len());
        let mut divisors = Vec::with_capacity(combos.len());
        let mut first_valids = Vec::with_capacity(combos.len());
        let mut needs_months = false;

        for params in &combos {
            let anchor = params.anchor.as_deref().unwrap_or("1d");
            let (count_u32, unit_char) =
                parse_anchor(anchor).map_err(|e| CudaVwapError::InvalidInput(e.to_string()))?;
            if count_u32 == 0 {
                return Err(CudaVwapError::InvalidInput(format!(
                    "anchor '{}' resolved to zero count",
                    anchor
                )));
            }
            let count = i32::try_from(count_u32)
                .map_err(|_| CudaVwapError::InvalidInput("count exceeds i32::MAX".into()))?;

            let (unit_code, divisor) = match unit_char {
                'm' => (0, (count as i64).saturating_mul(60_000)),
                'h' => (1, (count as i64).saturating_mul(3_600_000)),
                'd' => (2, (count as i64).saturating_mul(86_400_000)),
                'M' => {
                    needs_months = true;
                    (3, count as i64)
                }
                other => {
                    return Err(CudaVwapError::InvalidInput(format!(
                        "unsupported anchor unit '{}'",
                        other
                    )))
                }
            };

            if divisor <= 0 {
                return Err(CudaVwapError::InvalidInput(format!(
                    "non-positive divisor derived from anchor '{}'",
                    anchor
                )));
            }

            let warm = first_valid_vwap_index(timestamps, volumes, count_u32, unit_char);
            let warm_i32 = i32::try_from(warm).unwrap_or(i32::MAX);

            counts.push(count);
            unit_codes.push(unit_code);
            divisors.push(divisor);
            first_valids.push(warm_i32);
        }

        let month_ids = if needs_months {
            Some(Self::compute_month_ids(timestamps)?)
        } else {
            None
        };

        Ok(PreparedBatch {
            combos,
            counts,
            unit_codes,
            divisors,
            first_valids,
            month_ids,
            series_len: prices.len(),
        })
    }

    fn compute_month_ids(timestamps: &[i64]) -> Result<Vec<i32>, CudaVwapError> {
        use crate::indicators::moving_averages::vwap::floor_to_month;

        let mut out = Vec::with_capacity(timestamps.len());
        for &ts in timestamps {
            let month = match floor_to_month(ts, 1) {
                Ok(v) => v,
                Err(_) => i64::MIN,
            };
            let clamped = month.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            out.push(clamped);
        }
        Ok(out)
    }

    fn launch_kernel(
        &self,
        d_timestamps: &DeviceBuffer<i64>,
        d_volumes: &DeviceBuffer<f32>,
        d_prices: &DeviceBuffer<f32>,
        d_counts: &DeviceBuffer<i32>,
        d_unit_codes: &DeviceBuffer<i32>,
        d_divisors: &DeviceBuffer<i64>,
        d_first_valids: &DeviceBuffer<i32>,
        month_ids_ptr: u64,
        d_out: &mut DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
    ) -> Result<(), CudaVwapError> {
        if series_len > i32::MAX as usize {
            return Err(CudaVwapError::InvalidInput(
                "series length exceeds i32::MAX (unsupported by kernel)".into(),
            ));
        }
        if n_combos > i32::MAX as usize {
            return Err(CudaVwapError::InvalidInput(
                "number of parameter combos exceeds i32::MAX".into(),
            ));
        }

        let func = self.module.get_function("vwap_batch_f32").map_err(|_| {
            CudaVwapError::MissingKernelSymbol {
                name: "vwap_batch_f32",
            }
        })?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => 256,
        };
        if block_x != 256 {
            return Err(CudaVwapError::InvalidPolicy(
                "vwap batch kernel requires block_x=256",
            ));
        }
        let grid_x = n_combos as u32;

        let series_len_i = series_len as i32;
        let n_combos_i = n_combos as i32;

        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        if block_x > 1024 {
            return Err(CudaVwapError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        unsafe {
            let mut ts_ptr = d_timestamps.as_device_ptr().as_raw();
            let mut vol_ptr = d_volumes.as_device_ptr().as_raw();
            let mut price_ptr = d_prices.as_device_ptr().as_raw();
            let mut count_ptr = d_counts.as_device_ptr().as_raw();
            let mut unit_ptr = d_unit_codes.as_device_ptr().as_raw();
            let mut div_ptr = d_divisors.as_device_ptr().as_raw();
            let mut warm_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut month_ptr = month_ids_ptr;
            let mut series_len_i = series_len_i;
            let mut n_combos_i = n_combos_i;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut ts_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut price_ptr as *mut _ as *mut c_void,
                &mut count_ptr as *mut _ as *mut c_void,
                &mut unit_ptr as *mut _ as *mut c_void,
                &mut div_ptr as *mut _ as *mut c_void,
                &mut warm_ptr as *mut _ as *mut c_void,
                &mut month_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, 0, args)?;
        }

        unsafe {
            let this = self as *const _ as *mut CudaVwap;
            (*this).last_batch = Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();
        Ok(())
    }

    pub fn vwap_batch_dev(
        &self,
        timestamps: &[i64],
        volumes: &[f64],
        prices: &[f64],
        sweep: &VwapBatchRange,
    ) -> Result<SharedDeviceArrayF32, CudaVwapError> {
        let PreparedBatch {
            combos,
            counts,
            unit_codes,
            divisors,
            first_valids,
            month_ids,
            series_len,
        } = Self::prepare_batch_inputs(timestamps, volumes, prices, sweep)?;
        let n_combos = combos.len();

        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let volumes_f32: Vec<f32> = volumes.iter().map(|&v| v as f32).collect();

        let in_bytes = series_len * (std::mem::size_of::<i64>() + 2 * std::mem::size_of::<f32>());
        let param_bytes = n_combos
            * (2 * std::mem::size_of::<i32>()
                + std::mem::size_of::<i64>()
                + std::mem::size_of::<i32>());
        let month_bytes = month_ids.as_ref().map(|v| v.len() * 4).unwrap_or(0);
        let out_bytes = n_combos
            .checked_mul(series_len)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let required = in_bytes
            .checked_add(param_bytes)
            .and_then(|v| v.checked_add(month_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaVwapError::OutOfMemory {
                required,
                free: Self::device_mem_info().map(|(f, _)| f).unwrap_or(0),
                headroom,
            });
        }

        let d_timestamps = unsafe { DeviceBuffer::from_slice_async(timestamps, &self.stream) }?;
        let d_volumes = unsafe { DeviceBuffer::from_slice_async(&volumes_f32, &self.stream) }?;
        let d_prices = unsafe { DeviceBuffer::from_slice_async(&prices_f32, &self.stream) }?;
        let d_counts = unsafe { DeviceBuffer::from_slice_async(&counts, &self.stream) }?;
        let d_unit_codes = unsafe { DeviceBuffer::from_slice_async(&unit_codes, &self.stream) }?;
        let d_divisors = unsafe { DeviceBuffer::from_slice_async(&divisors, &self.stream) }?;
        let d_first_valids =
            unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
        let mut d_month_ids = if let Some(ids) = month_ids {
            Some(unsafe { DeviceBuffer::from_slice_async(&ids, &self.stream) }?)
        } else {
            None
        };
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n_combos * series_len, &self.stream) }?;

        let month_ptr = d_month_ids
            .as_mut()
            .map(|buf| buf.as_device_ptr().as_raw())
            .unwrap_or(0);

        self.launch_kernel(
            &d_timestamps,
            &d_volumes,
            &d_prices,
            &d_counts,
            &d_unit_codes,
            &d_divisors,
            &d_first_valids,
            month_ptr,
            &mut d_out,
            series_len,
            n_combos,
        )?;

        self.stream.synchronize()?;
        Ok(SharedDeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn vwap_batch_dev_retaining_ctx(
        &self,
        timestamps: &[i64],
        volumes: &[f64],
        prices: &[f64],
        sweep: &VwapBatchRange,
    ) -> Result<VwapDeviceArrayF32, CudaVwapError> {
        let PreparedBatch {
            combos,
            counts,
            unit_codes,
            divisors,
            first_valids,
            month_ids,
            series_len,
        } = Self::prepare_batch_inputs(timestamps, volumes, prices, sweep)?;
        let n_combos = combos.len();
        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let volumes_f32: Vec<f32> = volumes.iter().map(|&v| v as f32).collect();

        let in_bytes = series_len * (std::mem::size_of::<i64>() + 2 * std::mem::size_of::<f32>());
        let param_bytes = n_combos
            * (2 * std::mem::size_of::<i32>()
                + std::mem::size_of::<i64>()
                + std::mem::size_of::<i32>());
        let month_bytes = month_ids.as_ref().map(|v| v.len() * 4).unwrap_or(0);
        let out_bytes = n_combos
            .checked_mul(series_len)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let required = in_bytes
            .checked_add(param_bytes)
            .and_then(|v| v.checked_add(month_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaVwapError::OutOfMemory {
                required,
                free: Self::device_mem_info().map(|(f, _)| f).unwrap_or(0),
                headroom,
            });
        }

        let d_timestamps = unsafe { DeviceBuffer::from_slice_async(timestamps, &self.stream) }?;
        let d_volumes = unsafe { DeviceBuffer::from_slice_async(&volumes_f32, &self.stream) }?;
        let d_prices = unsafe { DeviceBuffer::from_slice_async(&prices_f32, &self.stream) }?;
        let d_counts = unsafe { DeviceBuffer::from_slice_async(&counts, &self.stream) }?;
        let d_unit_codes = unsafe { DeviceBuffer::from_slice_async(&unit_codes, &self.stream) }?;
        let d_divisors = unsafe { DeviceBuffer::from_slice_async(&divisors, &self.stream) }?;
        let d_first_valids =
            unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
        let mut d_month_ids = if let Some(ids) = month_ids {
            Some(unsafe { DeviceBuffer::from_slice_async(&ids, &self.stream) }?)
        } else {
            None
        };
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(n_combos * series_len, &self.stream) }?;

        let month_ptr = d_month_ids
            .as_mut()
            .map(|buf| buf.as_device_ptr().as_raw())
            .unwrap_or(0);
        self.launch_kernel(
            &d_timestamps,
            &d_volumes,
            &d_prices,
            &d_counts,
            &d_unit_codes,
            &d_divisors,
            &d_first_valids,
            month_ptr,
            &mut d_out,
            series_len,
            n_combos,
        )?;
        self.stream.synchronize()?;
        Ok(VwapDeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
            _ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    pub fn vwap_batch_into_host_f32(
        &self,
        timestamps: &[i64],
        volumes: &[f64],
        prices: &[f64],
        sweep: &VwapBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<VwapParams>), CudaVwapError> {
        let PreparedBatch {
            combos,
            counts,
            unit_codes,
            divisors,
            first_valids,
            month_ids,
            series_len,
        } = Self::prepare_batch_inputs(timestamps, volumes, prices, sweep)?;
        let n_combos = combos.len();
        let expected = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaVwapError::InvalidInput("rows*cols overflow".into()))?;
        if out.len() != expected {
            return Err(CudaVwapError::InvalidInput(format!(
                "output slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
        let volumes_f32: Vec<f32> = volumes.iter().map(|&v| v as f32).collect();

        let d_timestamps = unsafe { DeviceBuffer::from_slice_async(timestamps, &self.stream) }?;
        let d_volumes = unsafe { DeviceBuffer::from_slice_async(&volumes_f32, &self.stream) }?;
        let d_prices = unsafe { DeviceBuffer::from_slice_async(&prices_f32, &self.stream) }?;
        let d_counts = unsafe { DeviceBuffer::from_slice_async(&counts, &self.stream) }?;
        let d_unit_codes = unsafe { DeviceBuffer::from_slice_async(&unit_codes, &self.stream) }?;
        let d_divisors = unsafe { DeviceBuffer::from_slice_async(&divisors, &self.stream) }?;
        let d_first_valids =
            unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
        let mut d_month_ids = if let Some(ids) = month_ids {
            Some(unsafe { DeviceBuffer::from_slice_async(&ids, &self.stream) }?)
        } else {
            None
        };
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(expected, &self.stream) }?;

        let month_ptr = d_month_ids
            .as_mut()
            .map(|buf| buf.as_device_ptr().as_raw())
            .unwrap_or(0);

        self.launch_kernel(
            &d_timestamps,
            &d_volumes,
            &d_prices,
            &d_counts,
            &d_unit_codes,
            &d_divisors,
            &d_first_valids,
            month_ptr,
            &mut d_out,
            series_len,
            n_combos,
        )?;

        let mut pinned_out = unsafe { LockedBuffer::<f32>::uninitialized(expected) }?;
        unsafe {
            d_out.async_copy_to(pinned_out.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        out.copy_from_slice(pinned_out.as_slice());

        Ok((n_combos, series_len, combos))
    }

    pub fn vwap_batch_device(
        &self,
        d_timestamps: &DeviceBuffer<i64>,
        d_volumes: &DeviceBuffer<f32>,
        d_prices: &DeviceBuffer<f32>,
        counts: &[i32],
        unit_codes: &[i32],
        divisors: &[i64],
        first_valids: &[i32],
        month_ids: Option<&DeviceBuffer<i32>>,
        series_len: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVwapError> {
        let n_combos = counts.len();
        let d_counts = unsafe { DeviceBuffer::from_slice_async(counts, &self.stream) }?;
        let d_unit_codes = unsafe { DeviceBuffer::from_slice_async(unit_codes, &self.stream) }?;
        let d_divisors = unsafe { DeviceBuffer::from_slice_async(divisors, &self.stream) }?;
        let d_first_valids = unsafe { DeviceBuffer::from_slice_async(first_valids, &self.stream) }?;

        self.vwap_batch_device_with_params(
            d_timestamps,
            d_volumes,
            d_prices,
            &d_counts,
            &d_unit_codes,
            &d_divisors,
            &d_first_valids,
            month_ids,
            series_len,
            n_combos,
            d_out,
        )
    }

    pub fn vwap_batch_device_with_params(
        &self,
        d_timestamps: &DeviceBuffer<i64>,
        d_volumes: &DeviceBuffer<f32>,
        d_prices: &DeviceBuffer<f32>,
        d_counts: &DeviceBuffer<i32>,
        d_unit_codes: &DeviceBuffer<i32>,
        d_divisors: &DeviceBuffer<i64>,
        d_first_valids: &DeviceBuffer<i32>,
        month_ids: Option<&DeviceBuffer<i32>>,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVwapError> {
        if d_counts.len() != n_combos
            || d_unit_codes.len() != n_combos
            || d_divisors.len() != n_combos
            || d_first_valids.len() != n_combos
        {
            return Err(CudaVwapError::InvalidInput(
                "parameter buffer length mismatch".into(),
            ));
        }
        let expected = n_combos
            .checked_mul(series_len)
            .ok_or_else(|| CudaVwapError::InvalidInput("rows*cols overflow".into()))?;
        if d_out.len() != expected {
            return Err(CudaVwapError::InvalidInput(format!(
                "output buffer wrong length: got {}, expected {}",
                d_out.len(),
                expected
            )));
        }

        let month_ptr = month_ids
            .map(|buf| buf.as_device_ptr().as_raw())
            .unwrap_or(0);

        self.launch_kernel(
            d_timestamps,
            d_volumes,
            d_prices,
            d_counts,
            d_unit_codes,
            d_divisors,
            d_first_valids,
            month_ptr,
            d_out,
            series_len,
            n_combos,
        )?;
        self.stream.synchronize()?;
        Ok(())
    }

    fn compute_first_valids_many_series(
        timestamps: &[i64],
        volumes_tm: &[f64],
        cols: usize,
        rows: usize,
        count: u32,
        unit_char: char,
    ) -> Result<Vec<i32>, CudaVwapError> {
        if timestamps.len() != rows {
            return Err(CudaVwapError::InvalidInput(
                "timestamps len must equal rows".into(),
            ));
        }
        if volumes_tm.len() != rows * cols {
            return Err(CudaVwapError::InvalidInput(
                "volumes_tm wrong length".into(),
            ));
        }
        let mut out = vec![0i32; cols];
        let bucket_ms: i64 = match unit_char {
            'm' => (count as i64) * 60_000,
            'h' => (count as i64) * 3_600_000,
            'd' => (count as i64) * 86_400_000,
            'M' => 0,
            _ => 0,
        };
        if unit_char == 'M' {
            let months: Vec<i32> = Self::compute_month_ids(timestamps)?;
            for s in 0..cols {
                let mut cur_gid = i64::MIN;
                let mut vsum = 0.0f64;
                for t in 0..rows {
                    let gid = (months[t] as i64) / (count as i64);
                    if gid != cur_gid {
                        cur_gid = gid;
                        vsum = 0.0;
                    }
                    let v = volumes_tm[t * cols + s];
                    vsum += v;
                    if vsum > 0.0 {
                        out[s] = t as i32;
                        break;
                    }
                }
            }
        } else {
            let div = bucket_ms.max(1);
            for s in 0..cols {
                let mut cur_gid = i64::MIN;
                let mut vsum = 0.0f64;
                for t in 0..rows {
                    let gid = timestamps[t] / div;
                    if gid != cur_gid {
                        cur_gid = gid;
                        vsum = 0.0;
                    }
                    let v = volumes_tm[t * cols + s];
                    vsum += v;
                    if vsum > 0.0 {
                        out[s] = t as i32;
                        break;
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn vwap_many_series_one_param_time_major_dev(
        &self,
        timestamps: &[i64],
        volumes_tm_f64: &[f64],
        prices_tm_f64: &[f64],
        cols: usize,
        rows: usize,
        anchor: &str,
    ) -> Result<SharedDeviceArrayF32, CudaVwapError> {
        if cols == 0 || rows == 0 {
            return Err(CudaVwapError::InvalidInput("empty matrix".into()));
        }
        if timestamps.len() != rows {
            return Err(CudaVwapError::InvalidInput("timestamps len != rows".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaVwapError::InvalidInput("empty matrix".into()));
        }
        if timestamps.len() != rows {
            return Err(CudaVwapError::InvalidInput("timestamps len != rows".into()));
        }
        if volumes_tm_f64.len() != rows * cols || prices_tm_f64.len() != rows * cols {
            return Err(CudaVwapError::InvalidInput(
                "prices/volumes len != rows*cols".into(),
            ));
            return Err(CudaVwapError::InvalidInput(
                "prices/volumes len != rows*cols".into(),
            ));
        }

        let (count, unit_char) =
            parse_anchor(anchor).map_err(|e| CudaVwapError::InvalidInput(e.to_string()))?;
        let (count, unit_char) =
            parse_anchor(anchor).map_err(|e| CudaVwapError::InvalidInput(e.to_string()))?;

        let first_valids = Self::compute_first_valids_many_series(
            timestamps,
            volumes_tm_f64,
            cols,
            rows,
            count,
            unit_char,
        )?;

        let prices_tm_f32: Vec<f32> = prices_tm_f64.iter().map(|&v| v as f32).collect();
        let volumes_tm_f32: Vec<f32> = volumes_tm_f64.iter().map(|&v| v as f32).collect();

        let in_bytes_ts = rows
            .checked_mul(std::mem::size_of::<i64>())
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let in_bytes_floats = rows
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(2 * std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let in_bytes_first_valids = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let in_bytes = in_bytes_ts
            .checked_add(in_bytes_floats)
            .and_then(|v| v.checked_add(in_bytes_first_valids))
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let month_bytes = if unit_char == 'M' {
            rows * std::mem::size_of::<i32>()
        } else {
            0
        };
        let month_bytes = if unit_char == 'M' {
            rows * std::mem::size_of::<i32>()
        } else {
            0
        };
        let out_bytes = rows
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let required = in_bytes
            .checked_add(month_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaVwapError::OutOfMemory {
                required,
                free: Self::device_mem_info().map(|(f, _)| f).unwrap_or(0),
                headroom,
            });
        }

        let d_timestamps = unsafe { DeviceBuffer::from_slice_async(timestamps, &self.stream) }?;
        let d_volumes_tm =
            unsafe { DeviceBuffer::from_slice_async(&volumes_tm_f32, &self.stream) }?;
        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(&prices_tm_f32, &self.stream) }?;
        let d_first_valids =
            unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
        let mut d_month_ids = if unit_char == 'M' {
            let ids = Self::compute_month_ids(timestamps)?;
            Some(unsafe { DeviceBuffer::from_slice_async(&ids, &self.stream) }?)
        } else {
            None
        };
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(rows * cols, &self.stream) }?;

        self.launch_many_series_kernel(
            &d_timestamps,
            &d_volumes_tm,
            &d_prices_tm,
            count,
            unit_char,
            &d_first_valids,
            d_month_ids.as_mut().map(|b| b),
            cols,
            rows,
            &mut d_out_tm,
        )?;
        self.stream.synchronize()?;
        Ok(SharedDeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn vwap_many_series_one_param_time_major_dev_retaining_ctx(
        &self,
        timestamps: &[i64],
        volumes_tm_f64: &[f64],
        prices_tm_f64: &[f64],
        cols: usize,
        rows: usize,
        anchor: &str,
    ) -> Result<VwapDeviceArrayF32, CudaVwapError> {
        if cols == 0 || rows == 0 {
            return Err(CudaVwapError::InvalidInput("empty matrix".into()));
        }
        if timestamps.len() != rows {
            return Err(CudaVwapError::InvalidInput("timestamps len != rows".into()));
        }
        if volumes_tm_f64.len() != rows * cols || prices_tm_f64.len() != rows * cols {
            return Err(CudaVwapError::InvalidInput(
                "prices/volumes len != rows*cols".into(),
            ));
        }
        let (count, unit_char) =
            parse_anchor(anchor).map_err(|e| CudaVwapError::InvalidInput(e.to_string()))?;
        let first_valids = Self::compute_first_valids_many_series(
            timestamps,
            volumes_tm_f64,
            cols,
            rows,
            count,
            unit_char,
        )?;
        let prices_tm_f32: Vec<f32> = prices_tm_f64.iter().map(|&v| v as f32).collect();
        let volumes_tm_f32: Vec<f32> = volumes_tm_f64.iter().map(|&v| v as f32).collect();
        let in_bytes_ts = rows
            .checked_mul(std::mem::size_of::<i64>())
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let in_bytes_floats = rows
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(2 * std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let in_bytes_first_valids = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let in_bytes = in_bytes_ts
            .checked_add(in_bytes_floats)
            .and_then(|v| v.checked_add(in_bytes_first_valids))
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let month_bytes = if unit_char == 'M' {
            rows * std::mem::size_of::<i32>()
        } else {
            0
        };
        let out_bytes = rows
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let required = in_bytes
            .checked_add(month_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaVwapError::InvalidInput("byte size overflow".into()))?;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaVwapError::OutOfMemory {
                required,
                free: Self::device_mem_info().map(|(f, _)| f).unwrap_or(0),
                headroom,
            });
        }

        let d_timestamps = unsafe { DeviceBuffer::from_slice_async(timestamps, &self.stream) }?;
        let d_volumes_tm =
            unsafe { DeviceBuffer::from_slice_async(&volumes_tm_f32, &self.stream) }?;
        let d_prices_tm = unsafe { DeviceBuffer::from_slice_async(&prices_tm_f32, &self.stream) }?;
        let d_first_valids =
            unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
        let mut d_month_ids = if unit_char == 'M' {
            let ids = Self::compute_month_ids(timestamps)?;
            Some(unsafe { DeviceBuffer::from_slice_async(&ids, &self.stream) }?)
        } else {
            None
        };
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(rows * cols, &self.stream) }?;

        self.launch_many_series_kernel(
            &d_timestamps,
            &d_volumes_tm,
            &d_prices_tm,
            count,
            unit_char,
            &d_first_valids,
            d_month_ids.as_mut().map(|b| b),
            cols,
            rows,
            &mut d_out_tm,
        )?;
        self.stream.synchronize()?;
        Ok(VwapDeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
            _ctx: self._context.clone(),
            device_id: self.device_id,
        })
    }

    fn launch_many_series_kernel(
        &self,
        d_timestamps: &DeviceBuffer<i64>,
        d_volumes_tm: &DeviceBuffer<f32>,
        d_prices_tm: &DeviceBuffer<f32>,
        count: u32,
        unit_char: char,
        d_first_valids: &DeviceBuffer<i32>,
        d_month_ids: Option<&mut DeviceBuffer<i32>>,
        cols: usize,
        rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVwapError> {
        if cols == 0 || rows == 0 {
            return Err(CudaVwapError::InvalidInput("empty matrix".into()));
        }
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaVwapError::InvalidInput("rows*cols overflow".into()))?;
        if d_out_tm.len() != expected {
            return Err(CudaVwapError::InvalidInput("out buf wrong length".into()));
        }
        if cols == 0 || rows == 0 {
            return Err(CudaVwapError::InvalidInput("empty matrix".into()));
        }
        let expected2 = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaVwapError::InvalidInput("rows*cols overflow".into()))?;
        if d_out_tm.len() != expected2 {
            return Err(CudaVwapError::InvalidInput("out buf wrong length".into()));
        }

        let func = self
            .module
            .get_function("vwap_multi_series_one_param_f32")
            .map_err(|_| CudaVwapError::MissingKernelSymbol {
                name: "vwap_multi_series_one_param_f32",
            })?;

        let (unit_code, divisor_ms, month_ptr): (i32, i64, u64) = match unit_char {
            'm' => (0, (count as i64) * 60_000, 0),
            'h' => (1, (count as i64) * 3_600_000, 0),
            'd' => (2, (count as i64) * 86_400_000, 0),
            'M' => (
                3,
                0,
                d_month_ids.map(|b| b.as_device_ptr().as_raw()).unwrap_or(0),
            ),
            'M' => (
                3,
                0,
                d_month_ids.map(|b| b.as_device_ptr().as_raw()).unwrap_or(0),
            ),
            _ => return Err(CudaVwapError::InvalidInput("unsupported unit".into())),
        };

        let (_grid_suggest, block_suggest) = func
            .suggested_launch_configuration(0, (0, 0, 0).into())
            .unwrap_or((0, 128));
        let auto_block = block_suggest.max(128).min(1024);
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
            ManySeriesKernelPolicy::Auto => auto_block,
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;

        let count_i = count as i32;
        let num_series_i = cols as i32;
        let series_len_i = rows as i32;

        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        if block_x > 1024 {
            return Err(CudaVwapError::LaunchConfigTooLarge {
                gx: grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        unsafe {
            let mut ts_ptr = d_timestamps.as_device_ptr().as_raw();
            let mut vol_ptr = d_volumes_tm.as_device_ptr().as_raw();
            let mut price_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut count_i = count_i;
            let mut unit_i = unit_code as i32;
            let mut divisor_i = divisor_ms as i64;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut month_ptr_u = month_ptr;
            let mut num_series_i = num_series_i;
            let mut series_len_i = series_len_i;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut ts_ptr as *mut _ as *mut c_void,
                &mut vol_ptr as *mut _ as *mut c_void,
                &mut price_ptr as *mut _ as *mut c_void,
                &mut count_i as *mut _ as *mut c_void,
                &mut unit_i as *mut _ as *mut c_void,
                &mut divisor_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut month_ptr_u as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, 0, args)?;
        }

        unsafe {
            let this = self as *const _ as *mut CudaVwap;
            (*this).last_many = Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }
}

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes =
            ONE_SERIES_LEN * (std::mem::size_of::<i64>() + 2 * std::mem::size_of::<f32>());
        let param_bytes = PARAM_SWEEP
            * (2 * std::mem::size_of::<i32>()
                + std::mem::size_of::<i64>()
                + std::mem::size_of::<i32>());
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + param_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn synth_vwap_inputs(len: usize) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
        let mut ts = vec![0i64; len];
        for i in 0..len {
            ts[i] = (i as i64) * 60_000;
        }

        let mut prices = vec![f64::NAN; len];
        for i in 3..len {
            let x = i as f64;
            prices[i] = (x * 0.001).sin() + 0.0001 * x;
        }

        let mut vols = vec![f64::NAN; len];
        for i in 5..len {
            let x = i as f64 * 0.007;
            vols[i] = (x.cos().abs() + 1.2) * 950.0;
        }
        (ts, vols, prices)
    }

    struct VwapBatchDevState {
        cuda: CudaVwap,
        series_len: usize,
        n_combos: usize,
        d_timestamps: DeviceBuffer<i64>,
        d_volumes: DeviceBuffer<f32>,
        d_prices: DeviceBuffer<f32>,
        d_counts: DeviceBuffer<i32>,
        d_unit_codes: DeviceBuffer<i32>,
        d_divisors: DeviceBuffer<i64>,
        d_first_valids: DeviceBuffer<i32>,
        d_month_ids: Option<DeviceBuffer<i32>>,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for VwapBatchDevState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .vwap_batch_device_with_params(
                    &self.d_timestamps,
                    &self.d_volumes,
                    &self.d_prices,
                    &self.d_counts,
                    &self.d_unit_codes,
                    &self.d_divisors,
                    &self.d_first_valids,
                    self.d_month_ids.as_ref(),
                    self.series_len,
                    self.n_combos,
                    &mut self.d_out,
                )
                .expect("vwap batch launch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaVwap::new(0).expect("cuda vwap");
        let (ts, vol, price) = synth_vwap_inputs(ONE_SERIES_LEN);

        let sweep = VwapBatchRange {
            anchor: ("1d".to_string(), "250d".to_string(), 1),
        };
        let PreparedBatch {
            counts,
            unit_codes,
            divisors,
            first_valids,
            month_ids,
            series_len,
            combos: _,
        } = CudaVwap::prepare_batch_inputs(&ts, &vol, &price, &sweep)
            .expect("vwap prepare batch inputs");
        let n_combos = counts.len();

        let prices_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
        let volumes_f32: Vec<f32> = vol.iter().map(|&v| v as f32).collect();

        let d_timestamps = unsafe { DeviceBuffer::from_slice_async(&ts, &cuda.stream) }
            .expect("vwap upload timestamps");
        let d_volumes = unsafe { DeviceBuffer::from_slice_async(&volumes_f32, &cuda.stream) }
            .expect("vwap upload volumes");
        let d_prices = unsafe { DeviceBuffer::from_slice_async(&prices_f32, &cuda.stream) }
            .expect("vwap upload prices");

        let d_counts = unsafe { DeviceBuffer::from_slice_async(&counts, &cuda.stream) }
            .expect("vwap upload counts");
        let d_unit_codes = unsafe { DeviceBuffer::from_slice_async(&unit_codes, &cuda.stream) }
            .expect("vwap upload unit_codes");
        let d_divisors = unsafe { DeviceBuffer::from_slice_async(&divisors, &cuda.stream) }
            .expect("vwap upload divisors");
        let d_first_valids = unsafe { DeviceBuffer::from_slice_async(&first_valids, &cuda.stream) }
            .expect("vwap upload first_valids");
        let d_month_ids = if let Some(ids) = month_ids {
            Some(
                unsafe { DeviceBuffer::from_slice_async(&ids, &cuda.stream) }
                    .expect("vwap upload month_ids"),
            )
        } else {
            None
        };

        let expected = n_combos * series_len;
        let d_out = unsafe { DeviceBuffer::uninitialized(expected) }.expect("vwap allocate output");

        cuda.stream.synchronize().expect("vwap prep sync");

        Box::new(VwapBatchDevState {
            cuda,
            series_len,
            n_combos,
            d_timestamps,
            d_volumes,
            d_prices,
            d_counts,
            d_unit_codes,
            d_divisors,
            d_first_valids,
            d_month_ids,
            d_out,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        struct VwapManySeriesState {
            cuda: CudaVwap,
            d_timestamps: DeviceBuffer<i64>,
            d_vol_tm: DeviceBuffer<f32>,
            d_price_tm: DeviceBuffer<f32>,
            d_first_valids: DeviceBuffer<i32>,
            d_month_ids: Option<DeviceBuffer<i32>>,
            cols: usize,
            rows: usize,
            count: u32,
            unit_char: char,
            d_out_tm: DeviceBuffer<f32>,
        }
        impl CudaBenchState for VwapManySeriesState {
            fn launch(&mut self) {
                self.cuda
                    .launch_many_series_kernel(
                        &self.d_timestamps,
                        &self.d_vol_tm,
                        &self.d_price_tm,
                        self.count,
                        self.unit_char,
                        &self.d_first_valids,
                        self.d_month_ids.as_mut(),
                        self.cols,
                        self.rows,
                        &mut self.d_out_tm,
                    )
                    .expect("vwap many-series");
                self.cuda.synchronize().expect("vwap many sync");
            }
        }

        let mut out = vec![CudaBenchScenario::new(
            "vwap",
            "one_series_many_params",
            "vwap_cuda_batch_dev",
            "1m_x_250",
            prep_one_series_many_params,
        )
        .with_sample_size(10)
        .with_mem_required(bytes_one_series_many_params())];

        fn synth_many_series(rows: usize, cols: usize) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
            let mut ts = vec![0i64; rows];
            for t in 0..rows {
                ts[t] = (t as i64) * 60_000;
            }
            let mut price_tm = vec![f32::NAN; rows * cols];
            let mut vol_tm = vec![0f32; rows * cols];
            for s in 0..cols {
                for t in (s % 7)..rows {
                    let x = (t as f32) + (s as f32) * 0.1;
                    price_tm[t * cols + s] = (x * 0.002).sin() + 0.0002 * x;
                    vol_tm[t * cols + s] = (x * 0.003).cos().abs() + 0.8;
                }
            }
            (ts, vol_tm, price_tm)
        }

        const MS_COLS: usize = 256;
        const MS_ROWS: usize = 500_000;
        fn prep_vwap_many_series_256x500k() -> Box<dyn CudaBenchState> {
            let cuda = CudaVwap::new(0).expect("cuda vwap");
            let (ts, vol_tm, price_tm) = synth_many_series(MS_ROWS, MS_COLS);
            let (count, unit_char) = parse_anchor("1d").expect("parse_anchor");
            let div = ((count as i64) * 86_400_000).max(1);
            let mut first_valids = vec![MS_ROWS as i32; MS_COLS];
            for s in 0..MS_COLS {
                let mut cur_gid = i64::MIN;
                let mut vsum = 0.0f64;
                for t in 0..MS_ROWS {
                    let gid = ts[t] / div;
                    if gid != cur_gid {
                        cur_gid = gid;
                        vsum = 0.0;
                    }
                    let v = vol_tm[t * MS_COLS + s];
                    if v.is_finite() {
                        vsum += v as f64;
                    }
                    if vsum > 0.0 {
                        first_valids[s] = t as i32;
                        break;
                    }
                }
            }

            let d_timestamps =
                unsafe { DeviceBuffer::from_slice_async(&ts, &cuda.stream) }.expect("d_ts");
            let d_vol_tm =
                unsafe { DeviceBuffer::from_slice_async(&vol_tm, &cuda.stream) }.expect("d_vol");
            let d_price_tm =
                unsafe { DeviceBuffer::from_slice_async(&price_tm, &cuda.stream) }.expect("d_px");
            let d_first_valids =
                unsafe { DeviceBuffer::from_slice_async(&first_valids, &cuda.stream) }
                    .expect("d_first_valids");
            let d_out_tm: DeviceBuffer<f32> =
                unsafe { DeviceBuffer::uninitialized_async(MS_ROWS * MS_COLS, &cuda.stream) }
                    .expect("d_out_tm");
            cuda.stream.synchronize().expect("vwap many prep sync");
            Box::new(VwapManySeriesState {
                cuda,
                d_timestamps,
                d_vol_tm,
                d_price_tm,
                d_first_valids,
                d_month_ids: None,
                cols: MS_COLS,
                rows: MS_ROWS,
                count,
                unit_char,
                d_out_tm,
            })
        }
        out.push(
            CudaBenchScenario::new(
                "vwap",
                "many_series_one_param",
                "vwap_cuda_many_series_one_param",
                "256x500k_1d",
                prep_vwap_many_series_256x500k,
            )
            .with_sample_size(10)
            .with_mem_required(
                MS_ROWS * 8 + MS_ROWS * MS_COLS * 4 * 3 + MS_COLS * 4 + 64 * 1024 * 1024,
            ),
        );

        out
    }
}
