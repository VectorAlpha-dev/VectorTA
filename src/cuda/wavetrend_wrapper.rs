#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::wavetrend::{WavetrendBatchRange, WavetrendParams};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, AsyncCopyDestination, CopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaWavetrendError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
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

// -------- Kernel selection policy (parity with ALMA style) --------

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
pub struct CudaWavetrendPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaWavetrendPolicy {
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

pub struct CudaWavetrend {
    module: Module,
    stream: Stream,
    ctx: Arc<Context>,
    device_id: u32,
    policy: CudaWavetrendPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

pub struct CudaWavetrendBatch {
    pub wt1: DeviceArrayF32,
    pub wt2: DeviceArrayF32,
    pub wt_diff: DeviceArrayF32,
    pub combos: Vec<WavetrendParams>,
}

struct PreparedBatch {
    combos: Vec<WavetrendParams>,
    first_valid: usize,
    series_len: usize,
}

impl CudaWavetrend {
    pub fn new(device_id: usize) -> Result<Self, CudaWavetrendError> {
        cust::init(CudaFlags::empty())?;

        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/wavetrend_kernel.ptx"));
        // High optimization with context-derived target, then graceful fallbacks.
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O4),
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
            policy: CudaWavetrendPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaWavetrendPolicy,
    ) -> Result<Self, CudaWavetrendError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }
    pub fn set_policy(&mut self, policy: CudaWavetrendPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaWavetrendPolicy {
        &self.policy
    }
    pub fn context_arc(&self) -> Arc<Context> { self.ctx.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario =
                    env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] Wavetrend batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaWavetrend)).debug_batch_logged = true;
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
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario =
                    env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] Wavetrend many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaWavetrend)).debug_many_logged = true;
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
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaWavetrendError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Ok((free, _total)) = mem_get_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaWavetrendError::OutOfMemory {
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
    fn checked_mul(a: usize, b: usize, what: &'static str) -> Result<usize, CudaWavetrendError> {
        a.checked_mul(b)
            .ok_or_else(|| CudaWavetrendError::InvalidInput(format!("{what} overflow")))
    }

    #[inline]
    fn checked_add(a: usize, b: usize, what: &'static str) -> Result<usize, CudaWavetrendError> {
        a.checked_add(b)
            .ok_or_else(|| CudaWavetrendError::InvalidInput(format!("{what} overflow")))
    }

    pub fn wavetrend_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &WavetrendBatchRange,
    ) -> Result<CudaWavetrendBatch, CudaWavetrendError> {
        let PreparedBatch {
            combos,
            first_valid,
            series_len,
        } = Self::prepare_batch_inputs(data_f32, sweep)?;
        let rows = combos.len();

        // VRAM estimate and guard
        let sizeof_f32 = std::mem::size_of::<f32>();
        let sizeof_i32 = std::mem::size_of::<i32>();
        let prices_bytes = Self::checked_mul(series_len, sizeof_f32, "prices_bytes")?;
        let param_i32_count = Self::checked_mul(3, rows, "params_i32_count")?;
        let param_i32_bytes =
            Self::checked_mul(param_i32_count, sizeof_i32, "params_i32_bytes")?;
        let param_f32_bytes = Self::checked_mul(rows, sizeof_f32, "params_f32_bytes")?;
        let params_bytes = Self::checked_add(param_i32_bytes, param_f32_bytes, "params_bytes")?;
        let elems_per_out = Self::checked_mul(rows, series_len, "rows*series_len")?;
        let out_elems = Self::checked_mul(3, elems_per_out, "out_elems")?;
        let out_bytes = Self::checked_mul(out_elems, sizeof_f32, "out_bytes")?;
        let required = Self::checked_add(
            Self::checked_add(prices_bytes, params_bytes, "prices+params_bytes")?,
            out_bytes,
            "total_bytes",
        )?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit(required, headroom)?;

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream) }?;

        let (channels, averages, mas, factors) = Self::build_param_arrays(&combos)?;
        let d_channels = unsafe { DeviceBuffer::from_slice_async(&channels, &self.stream) }?;
        let d_averages = unsafe { DeviceBuffer::from_slice_async(&averages, &self.stream) }?;
        let d_mas = unsafe { DeviceBuffer::from_slice_async(&mas, &self.stream) }?;
        let d_factors = unsafe { DeviceBuffer::from_slice_async(&factors, &self.stream) }?;

        let elems = Self::checked_mul(rows, series_len, "rows*series_len")?;
        let mut d_wt1: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_wt2: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_wt_diff: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;

        self.launch_kernel(
            &d_prices,
            &d_channels,
            &d_averages,
            &d_mas,
            &d_factors,
            first_valid,
            series_len,
            rows,
            &mut d_wt1,
            &mut d_wt2,
            &mut d_wt_diff,
        )?;

        self.stream.synchronize()?;

        Ok(CudaWavetrendBatch {
            wt1: DeviceArrayF32 {
                buf: d_wt1,
                rows,
                cols: series_len,
            },
            wt2: DeviceArrayF32 {
                buf: d_wt2,
                rows,
                cols: series_len,
            },
            wt_diff: DeviceArrayF32 {
                buf: d_wt_diff,
                rows,
                cols: series_len,
            },
            combos,
        })
    }

    pub fn wavetrend_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &WavetrendBatchRange,
        out_wt1: &mut [f32],
        out_wt2: &mut [f32],
        out_wt_diff: &mut [f32],
    ) -> Result<(usize, usize, Vec<WavetrendParams>), CudaWavetrendError> {
        let batch = self.wavetrend_batch_dev(data_f32, sweep)?;
        let rows = batch.wt1.rows;
        let cols = batch.wt1.cols;
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaWavetrendError::InvalidInput("rows*cols overflow".into()))?;
        if out_wt1.len() != expected || out_wt2.len() != expected || out_wt_diff.len() != expected {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "output slices have wrong length (expected {})",
                expected
            )));
        }

        batch.wt1.buf.copy_to(out_wt1)?;
        batch.wt2.buf.copy_to(out_wt2)?;
        batch.wt_diff.buf.copy_to(out_wt_diff)?;

        Ok((rows, cols, batch.combos))
    }

    // Optional pinned-host fast path (keeps API semantics identical elsewhere)
    pub fn wavetrend_batch_into_host_locked_f32(
        &self,
        data_f32: &[f32],
        sweep: &WavetrendBatchRange,
        out_wt1: &mut LockedBuffer<f32>,
        out_wt2: &mut LockedBuffer<f32>,
        out_wt_diff: &mut LockedBuffer<f32>,
    ) -> Result<(usize, usize, Vec<WavetrendParams>), CudaWavetrendError> {
        let batch = self.wavetrend_batch_dev(data_f32, sweep)?;
        let rows = batch.wt1.rows;
        let cols = batch.wt1.cols;
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| CudaWavetrendError::InvalidInput("rows*cols overflow".into()))?;
        if out_wt1.len() != expected || out_wt2.len() != expected || out_wt_diff.len() != expected {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "pinned output buffers have wrong length (expected {})",
                expected
            )));
        }
        unsafe {
            batch
                .wt1
                .buf
                .async_copy_to(out_wt1.as_mut_slice(), &self.stream)?;
            batch
                .wt2
                .buf
                .async_copy_to(out_wt2.as_mut_slice(), &self.stream)?;
            batch
                .wt_diff
                .buf
                .async_copy_to(out_wt_diff.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        Ok((rows, cols, batch.combos))
    }

    pub fn wavetrend_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        combos: &[WavetrendParams],
        first_valid: usize,
        series_len: usize,
        d_wt1: &mut DeviceBuffer<f32>,
        d_wt2: &mut DeviceBuffer<f32>,
        d_wt_diff: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWavetrendError> {
        if combos.is_empty() {
            return Err(CudaWavetrendError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        if series_len == 0 {
            return Err(CudaWavetrendError::InvalidInput(
                "series_len is zero".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "price buffer len {} != series_len {}",
                d_prices.len(),
                series_len
            )));
        }

        let (channels, averages, mas, factors) = Self::build_param_arrays(combos)?;
        let d_channels = unsafe { DeviceBuffer::from_slice_async(&channels, &self.stream) }?;
        let d_averages = unsafe { DeviceBuffer::from_slice_async(&averages, &self.stream) }?;
        let d_mas = unsafe { DeviceBuffer::from_slice_async(&mas, &self.stream) }?;
        let d_factors = unsafe { DeviceBuffer::from_slice_async(&factors, &self.stream) }?;

        self.launch_kernel(
            d_prices,
            &d_channels,
            &d_averages,
            &d_mas,
            &d_factors,
            first_valid,
            series_len,
            combos.len(),
            d_wt1,
            d_wt2,
            d_wt_diff,
        )?;

        self.stream.synchronize()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_channels: &DeviceBuffer<i32>,
        d_averages: &DeviceBuffer<i32>,
        d_mas: &DeviceBuffer<i32>,
        d_factors: &DeviceBuffer<f32>,
        first_valid: usize,
        series_len: usize,
        rows: usize,
        d_wt1: &mut DeviceBuffer<f32>,
        d_wt2: &mut DeviceBuffer<f32>,
        d_wt_diff: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWavetrendError> {
        if series_len == 0 {
            return Err(CudaWavetrendError::InvalidInput("series_len is zero".into()));
        }
        if d_prices.len() != series_len {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "price buffer len {} != series_len {}",
                d_prices.len(),
                series_len
            )));
        }
        if d_channels.len() != rows || d_averages.len() != rows || d_mas.len() != rows || d_factors.len() != rows {
            return Err(CudaWavetrendError::InvalidInput(
                "parameter buffers must match number of combinations".into(),
            ));
        }
        let expected = rows * series_len;
        if d_wt1.len() != expected || d_wt2.len() != expected || d_wt_diff.len() != expected {
            return Err(CudaWavetrendError::InvalidInput(format!(
                "output buffer mismatch: expected {} entries per output",
                expected
            )));
        }
        if series_len > i32::MAX as usize {
            return Err(CudaWavetrendError::InvalidInput("series length exceeds i32::MAX".into()));
        }
        if rows > i32::MAX as usize {
            return Err(CudaWavetrendError::InvalidInput("row count exceeds i32::MAX".into()));
        }

        let func = self
            .module
            .get_function("wavetrend_batch_f32")
            .map_err(|_| CudaWavetrendError::MissingKernelSymbol { name: "wavetrend_batch_f32" })?;

        // Occupancy-based suggestion
        let auto_block_x = {
            let (bs, _mg) = func.suggested_launch_configuration(0, (0, 0, 0).into())?;
            bs.clamp(32, 1024)
        };
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
            BatchKernelPolicy::Auto => auto_block_x,
        };

        // Full rows in one launch; enforce device limit
        let dev = Device::get_device(self.device_id)?;
        let max_grid_x = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let wanted_grid_x = ((rows as u32) + block_x - 1) / block_x;
        if wanted_grid_x == 0 || wanted_grid_x > max_grid_x {
            return Err(CudaWavetrendError::LaunchConfigTooLarge {
                gx: wanted_grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        let grid_x = wanted_grid_x;
        let grid: GridSize = (grid_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        // Record selection for diagnostics
        unsafe {
            (*(self as *const _ as *mut CudaWavetrend)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut first_i = first_valid as i32;
            let mut rows_i = rows as i32;
            let mut ch_ptr = d_channels.as_device_ptr().as_raw();
            let mut avg_ptr = d_averages.as_device_ptr().as_raw();
            let mut ma_ptr = d_mas.as_device_ptr().as_raw();
            let mut factor_ptr = d_factors.as_device_ptr().as_raw();
            let mut wt1_ptr = d_wt1.as_device_ptr().as_raw();
            let mut wt2_ptr = d_wt2.as_device_ptr().as_raw();
            let mut diff_ptr = d_wt_diff.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut ch_ptr as *mut _ as *mut c_void,
                &mut avg_ptr as *mut _ as *mut c_void,
                &mut ma_ptr as *mut _ as *mut c_void,
                &mut factor_ptr as *mut _ as *mut c_void,
                &mut wt1_ptr as *mut _ as *mut c_void,
                &mut wt2_ptr as *mut _ as *mut c_void,
                &mut diff_ptr as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        data: &[f32],
        sweep: &WavetrendBatchRange,
    ) -> Result<PreparedBatch, CudaWavetrendError> {
        if data.is_empty() {
            return Err(CudaWavetrendError::InvalidInput("empty data".into()));
        }
        let first_valid = data
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaWavetrendError::InvalidInput("all values are NaN".into()))?;
        let series_len = data.len();
        let combos = Self::expand_range(sweep)?;
        if combos.is_empty() {
            return Err(CudaWavetrendError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        if series_len > i32::MAX as usize {
            return Err(CudaWavetrendError::InvalidInput(
                "series length exceeds i32::MAX (unsupported)".into(),
            ));
        }
        if combos.len() > i32::MAX as usize {
            return Err(CudaWavetrendError::InvalidInput(
                "combination count exceeds i32::MAX (unsupported)".into(),
            ));
        }

        for (idx, combo) in combos.iter().enumerate() {
            let ch = combo.channel_length.unwrap_or(0);
            let avg = combo.average_length.unwrap_or(0);
            let ma = combo.ma_length.unwrap_or(0);
            if ch == 0 || avg == 0 || ma == 0 {
                return Err(CudaWavetrendError::InvalidInput(format!(
                    "invalid periods at combo {} (ch={}, avg={}, ma={})",
                    idx, ch, avg, ma
                )));
            }
            if ch > series_len || avg > series_len || ma > series_len {
                return Err(CudaWavetrendError::InvalidInput(format!(
                    "period exceeds series length at combo {}",
                    idx
                )));
            }
            let needed = ch.max(avg).max(ma);
            let valid = series_len - first_valid;
            if valid < needed {
                return Err(CudaWavetrendError::InvalidInput(format!(
                    "not enough valid data for combo {} (needed {}, valid {})",
                    idx, needed, valid
                )));
            }
        }

        Ok(PreparedBatch {
            combos,
            first_valid,
            series_len,
        })
    }

    fn build_param_arrays(
        combos: &[WavetrendParams],
    ) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>, Vec<f32>), CudaWavetrendError> {
        let mut channels = Vec::with_capacity(combos.len());
        let mut averages = Vec::with_capacity(combos.len());
        let mut mas = Vec::with_capacity(combos.len());
        let mut factors = Vec::with_capacity(combos.len());
        for combo in combos {
            channels.push(combo.channel_length.unwrap_or(0) as i32);
            averages.push(combo.average_length.unwrap_or(0) as i32);
            mas.push(combo.ma_length.unwrap_or(0) as i32);
            factors.push(combo.factor.unwrap_or(0.015) as f32);
        }
        Ok((channels, averages, mas, factors))
    }

    fn expand_range(sweep: &WavetrendBatchRange) -> Result<Vec<WavetrendParams>, CudaWavetrendError> {
        fn axis_usize(axis: (usize, usize, usize)) -> Result<Vec<usize>, CudaWavetrendError> {
            let (start, end, step) = axis;
            if step == 0 || start == end {
                return Ok(vec![start]);
            }
            if start < end {
                let st = step.max(1);
                return Ok((start..=end).step_by(st).collect());
            }
            let st = step.max(1) as isize;
            let mut v = Vec::new();
            let mut x = start as isize;
            let end_i = end as isize;
            while x >= end_i {
                v.push(x as usize);
                x -= st;
            }
            if v.is_empty() {
                return Err(CudaWavetrendError::InvalidInput(format!(
                    "invalid usize range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            Ok(v)
        }
        fn axis_f64(axis: (f64, f64, f64)) -> Result<Vec<f64>, CudaWavetrendError> {
            let (start, end, step) = axis;
            if step.abs() < f64::EPSILON || (start - end).abs() < f64::EPSILON {
                return Ok(vec![start]);
            }
            if start < end {
                let mut out = Vec::new();
                let mut v = start;
                let st = step.abs();
                while v <= end + 1e-12 {
                    out.push(v);
                    v += st;
                }
                if out.is_empty() {
                    return Err(CudaWavetrendError::InvalidInput(format!(
                        "invalid f64 range: start={}, end={}, step={}",
                        start, end, step
                    )));
                }
                return Ok(out);
            }
            let mut out = Vec::new();
            let mut v = start;
            let st = step.abs();
            while v + 1e-12 >= end {
                out.push(v);
                v -= st;
            }
            if out.is_empty() {
                return Err(CudaWavetrendError::InvalidInput(format!(
                    "invalid f64 range: start={}, end={}, step={}",
                    start, end, step
                )));
            }
            Ok(out)
        }

        let channels = axis_usize(sweep.channel_length)?;
        let averages = axis_usize(sweep.average_length)?;
        let mas = axis_usize(sweep.ma_length)?;
        let factors = axis_f64(sweep.factor)?;

        let cap = channels
            .len()
            .checked_mul(averages.len())
            .and_then(|x| x.checked_mul(mas.len()))
            .and_then(|x| x.checked_mul(factors.len()))
            .ok_or_else(|| CudaWavetrendError::InvalidInput("combo capacity overflow".into()))?;

        let mut combos = Vec::with_capacity(cap);
        for &ch in &channels {
            for &avg in &averages {
                for &ma in &mas {
                    for &f in &factors {
                        combos.push(WavetrendParams {
                            channel_length: Some(ch),
                            average_length: Some(avg),
                            ma_length: Some(ma),
                            factor: Some(f),
                        });
                    }
                }
            }
        }
        Ok(combos)
    }

    // ---------- Many-series × one-param (time-major) ----------
    pub fn wavetrend_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WavetrendParams,
    ) -> Result<(DeviceArrayF32, DeviceArrayF32, DeviceArrayF32), CudaWavetrendError> {
        let (first_valids, ch, avg, ma, factor) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // VRAM estimate: inputs + first_valids + 3 outputs
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaWavetrendError::InvalidInput("rows*cols overflow".into()))?;
        let sizeof_f32 = std::mem::size_of::<f32>();
        let sizeof_i32 = std::mem::size_of::<i32>();
        let in_bytes = Self::checked_mul(elems, sizeof_f32, "in_bytes")?;
        let fv_bytes = Self::checked_mul(cols, sizeof_i32, "first_valids_bytes")?;
        let out_elems = Self::checked_mul(3, elems, "out_elems")?;
        let out_bytes = Self::checked_mul(out_elems, sizeof_f32, "out_bytes")?;
        let required = Self::checked_add(
            Self::checked_add(in_bytes, fv_bytes, "in+fv_bytes")?,
            out_bytes,
            "total_bytes",
        )?;
        Self::will_fit(required, 64 * 1024 * 1024)?;

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream) }?;
        let d_first = unsafe { DeviceBuffer::from_slice_async(&first_valids, &self.stream) }?;
        let mut d_wt1: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_wt2: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;
        let mut d_wt_diff: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(elems, &self.stream) }?;

        self.launch_many_series_kernel(
            &d_prices,
            cols,
            rows,
            ch,
            avg,
            ma,
            factor,
            &d_first,
            &mut d_wt1,
            &mut d_wt2,
            &mut d_wt_diff,
        )?;
        self.stream.synchronize()?;

        Ok((
            DeviceArrayF32 {
                buf: d_wt1,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_wt2,
                rows,
                cols,
            },
            DeviceArrayF32 {
                buf: d_wt_diff,
                rows,
                cols,
            },
        ))
    }

    pub fn wavetrend_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WavetrendParams,
        wt1_tm: &mut [f32],
        wt2_tm: &mut [f32],
        wt_diff_tm: &mut [f32],
    ) -> Result<(), CudaWavetrendError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaWavetrendError::InvalidInput("size overflow".into()))?;
        if wt1_tm.len() != expected || wt2_tm.len() != expected || wt_diff_tm.len() != expected {
            return Err(CudaWavetrendError::InvalidInput(
                "output slices must be cols*rows".into(),
            ));
        }
        let (wt1, wt2, wt_diff) =
            self.wavetrend_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, params)?;
        wt1.buf.copy_to(wt1_tm)?;
        wt2.buf.copy_to(wt2_tm)?;
        wt_diff.buf.copy_to(wt_diff_tm)?;
        Ok(())
    }

    // Pinned-host fast path for many-series time-major outputs
    pub fn wavetrend_many_series_one_param_time_major_into_host_locked_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WavetrendParams,
        out_wt1_tm: &mut LockedBuffer<f32>,
        out_wt2_tm: &mut LockedBuffer<f32>,
        out_wt_diff_tm: &mut LockedBuffer<f32>,
    ) -> Result<(), CudaWavetrendError> {
        let expected = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaWavetrendError::InvalidInput("size overflow".into()))?;
        if out_wt1_tm.len() != expected || out_wt2_tm.len() != expected || out_wt_diff_tm.len() != expected {
            return Err(CudaWavetrendError::InvalidInput(
                "pinned output buffers must be cols*rows".into(),
            ));
        }
        let (wt1, wt2, wt_diff) =
            self.wavetrend_many_series_one_param_time_major_dev(data_tm_f32, cols, rows, params)?;
        unsafe {
            wt1.buf
                .async_copy_to(out_wt1_tm.as_mut_slice(), &self.stream)?;
            wt2.buf
                .async_copy_to(out_wt2_tm.as_mut_slice(), &self.stream)?;
            wt_diff
                .buf
                .async_copy_to(out_wt_diff_tm.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &WavetrendParams,
    ) -> Result<(Vec<i32>, i32, i32, i32, f32), CudaWavetrendError> {
        if cols == 0 || rows == 0 {
            return Err(CudaWavetrendError::InvalidInput(
                "cols or rows is zero".into(),
            ));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaWavetrendError::InvalidInput("size overflow".into()))?;
        if data_tm_f32.len() != elems {
            return Err(CudaWavetrendError::InvalidInput(
                "data length must be time-major cols*rows".into(),
            ));
        }
        let ch = params.channel_length.unwrap_or(9) as i32;
        let avg = params.average_length.unwrap_or(12) as i32;
        let ma = params.ma_length.unwrap_or(3) as i32;
        let factor = params.factor.unwrap_or(0.015) as f32;
        if ch <= 0 || avg <= 0 || ma <= 0 {
            return Err(CudaWavetrendError::InvalidInput(
                "periods must be positive".into(),
            ));
        }
        let need = ch.max(avg).max(ma) as usize;
        // Per-series first-valid indices
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv: Option<i32> = None;
            for t in 0..rows {
                if !data_tm_f32[t * cols + s].is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv = fv
                .ok_or_else(|| CudaWavetrendError::InvalidInput(format!("series {} all NaN", s)))?;
            if (rows as i32) - fv < (need as i32) {
                return Err(CudaWavetrendError::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    s,
                    need,
                    (rows as i32) - fv
                )));
            }
            first_valids[s] = fv;
        }
        Ok((first_valids, ch, avg, ma, factor))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        cols: usize,
        rows: usize,
        ch: i32,
        avg: i32,
        ma: i32,
        factor: f32,
        d_first_valids: &DeviceBuffer<i32>,
        d_wt1: &mut DeviceBuffer<f32>,
        d_wt2: &mut DeviceBuffer<f32>,
        d_wt_diff: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaWavetrendError> {
        if cols == 0 || rows == 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("wavetrend_many_series_one_param_time_major_f32")
            .map_err(|_| CudaWavetrendError::MissingKernelSymbol {
                name: "wavetrend_many_series_one_param_time_major_f32",
            })?;

        // Occupancy-based suggestion for block size
        let auto_block_x = {
            let (bs, _mg) = func.suggested_launch_configuration(0, (0, 0, 0).into())?;
            bs.clamp(32, 1024)
        };
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
            ManySeriesKernelPolicy::Auto => auto_block_x,
        };
        let dev = Device::get_device(self.device_id)?;
        let max_grid_x = dev.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let wanted_grid_x = ((cols as u32) + block_x - 1) / block_x;
        if wanted_grid_x == 0 || wanted_grid_x > max_grid_x {
            return Err(CudaWavetrendError::LaunchConfigTooLarge {
                gx: wanted_grid_x,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }
        let grid_x = wanted_grid_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            let mut p_prices = d_prices_tm.as_device_ptr().as_raw();
            let mut p_cols = cols as i32;
            let mut p_rows = rows as i32;
            let mut p_ch = ch;
            let mut p_avg = avg;
            let mut p_ma = ma;
            let mut p_factor = factor;
            let mut p_first = d_first_valids.as_device_ptr().as_raw();
            let mut p_wt1 = d_wt1.as_device_ptr().as_raw();
            let mut p_wt2 = d_wt2.as_device_ptr().as_raw();
            let mut p_diff = d_wt_diff.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_prices as *mut _ as *mut c_void,
                &mut p_cols as *mut _ as *mut c_void,
                &mut p_rows as *mut _ as *mut c_void,
                &mut p_ch as *mut _ as *mut c_void,
                &mut p_avg as *mut _ as *mut c_void,
                &mut p_ma as *mut _ as *mut c_void,
                &mut p_factor as *mut _ as *mut c_void,
                &mut p_first as *mut _ as *mut c_void,
                &mut p_wt1 as *mut _ as *mut c_void,
                &mut p_wt2 as *mut _ as *mut c_void,
                &mut p_diff as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        // Record selection and maybe log
        unsafe {
            (*(self as *const _ as *mut CudaWavetrend)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();
        Ok(())
    }
}

// ---------- Bench profiles (batch only) ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        // 3 outputs
        let out_bytes = 3 * ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = 3 * elems * std::mem::size_of::<f32>();
        in_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct WtBatchState {
        cuda: CudaWavetrend,
        price: Vec<f32>,
        sweep: WavetrendBatchRange,
    }
    impl CudaBenchState for WtBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .wavetrend_batch_dev(&self.price, &self.sweep)
                .expect("wavetrend batch");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaWavetrend::new(0).expect("cuda wavetrend");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = WavetrendBatchRange {
            channel_length: (10, 10 + PARAM_SWEEP - 1, 1),
            average_length: (21, 21, 0),
            ma_length: (4, 4, 0),
            factor: (0.015, 0.015, 0.0),
        };
        Box::new(WtBatchState { cuda, price, sweep })
    }

    struct WtManySeriesState {
        cuda: CudaWavetrend,
        tm: Vec<f32>,
    }
    impl CudaBenchState for WtManySeriesState {
        fn launch(&mut self) {
            let params = WavetrendParams {
                channel_length: Some(10),
                average_length: Some(21),
                ma_length: Some(4),
                factor: Some(0.015),
            };
            let _ = self
                .cuda
                .wavetrend_many_series_one_param_time_major_dev(
                    &self.tm,
                    MANY_SERIES_COLS,
                    MANY_SERIES_LEN,
                    &params,
                )
                .expect("wavetrend many-series");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaWavetrend::new(0).expect("cuda wavetrend");
        let tm = gen_time_major_prices(MANY_SERIES_COLS, MANY_SERIES_LEN);
        Box::new(WtManySeriesState { cuda, tm })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "wavetrend",
                "one_series_many_params",
                "wavetrend_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "wavetrend",
                "many_series_one_param",
                "wavetrend_cuda_many_series_one_param_dev",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
