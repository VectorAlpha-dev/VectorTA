//! CUDA support for the Exponential Moving Average (EMA).
//!
//! Brought in line with the ALMA wrapper design: policy-driven launch choices,
//! VRAM estimates with headroom checks, chunked launches to respect grid
//! limits, NON_BLOCKING stream, and JIT target-from-context PTX loading with
//! stable fallbacks. EMA is a recurrence/IIR pattern: one thread per combo
//! (or per series) executes the sequential scan; optional block size knobs are
//! provided for experimentation.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::ema::{EmaBatchRange, EmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::error::CudaError;
use thiserror::Error;
use std::sync::Arc;
use std::env;
use std::ffi::c_void;
use std::fmt;
// remove duplicate import

#[derive(Debug, Error)]
pub enum CudaEmaError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Out of memory on device: required={required} bytes, free={free} bytes, headroom={headroom} bytes")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Launch configuration too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("arithmetic overflow computing {0}")]
    ArithmeticOverflow(&'static str),
    #[error("not implemented")]
    NotImplemented,
}

// `thiserror::Error` derive already provides `std::error::Error` impl.

// -------- Kernel selection policy (explicit for tests; Auto for production) --------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    /// Plain 1D grid, one block per combo; thread 0 performs the scan.
    Plain {
        block_x: u32,
    },
}

impl Default for BatchKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    /// One-dimensional time-major mapping (grid.x = num_series).
    OneD {
        block_x: u32,
    },
}

impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaEmaPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected {
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected {
    OneD { block_x: u32 },
}

pub struct CudaEma {
    module: Module,
    stream: Stream,
    _ctx: Arc<Context>,
    device_id: u32,
    policy: CudaEmaPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    // Device-derived limits & features
    max_grid_x: usize,
    warp_size: u32,
    max_threads_per_block: u32,
    has_coalesced_ms: bool,
}

struct PreparedEmaBatch {
    combos: Vec<EmaParams>,
    first_valid: usize,
    series_len: usize,
    periods_i32: Vec<i32>,
    alphas_f32: Vec<f32>,
}

struct PreparedEmaManySeries {
    first_valids: Vec<i32>,
    period: i32,
    alpha: f32,
    num_series: usize,
    series_len: usize,
}

impl CudaEma {
    pub fn new(device_id: usize) -> Result<Self, CudaEmaError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Context::new(device)?;
        let ctx = Arc::new(context);

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/ema_kernel.ptx"));
        // Match ALMA: prefer DetermineTargetFromContext + O2; fall back to simpler modes.
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => {
                    Module::from_ptx(ptx, &[])?
                }
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // Device attributes for sane defaults & launch sizing
        let max_grid_x = device.get_attribute(cust::device::DeviceAttribute::MaxGridDimX)? as usize;
        let warp_size = device.get_attribute(cust::device::DeviceAttribute::WarpSize)? as u32;
        let max_threads_per_block =
            device.get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)? as u32;

        // Detect presence of coalesced kernel symbol in PTX
        let has_coalesced_ms = module
            .get_function("ema_many_series_one_param_f32_coalesced")
            .is_ok();

        Ok(Self {
            module,
            stream,
            _ctx: ctx,
            device_id: device_id as u32,
            policy: CudaEmaPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            max_grid_x,
            warp_size,
            max_threads_per_block,
            has_coalesced_ms,
        })
    }

    /// Optional: override policy (e.g., benches/tests).
    pub fn new_with_policy(device_id: usize, policy: CudaEmaPolicy) -> Result<Self, CudaEmaError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaEmaError> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[inline]
    pub(crate) fn context_arc(&self) -> Arc<Context> { self._ctx.clone() }

    #[inline]
    pub(crate) fn device_id(&self) -> u32 { self.device_id }

    pub fn ema_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &EmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaEmaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        // VRAM estimate and async H2D copy (fixed alphas sizing)
        let prices_bytes = prepared
            .series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaEmaError::ArithmeticOverflow("prices_bytes"))?;
        let params_count = prepared
            .periods_i32
            .len()
            .checked_add(prepared.alphas_f32.len())
            .ok_or(CudaEmaError::ArithmeticOverflow("params_count"))?;
        let params_bytes = params_count
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(CudaEmaError::ArithmeticOverflow("params_bytes"))?;
        let out_elems = n_combos
            .checked_mul(prepared.series_len)
            .ok_or(CudaEmaError::ArithmeticOverflow("out_elems"))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaEmaError::ArithmeticOverflow("out_bytes"))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or(CudaEmaError::ArithmeticOverflow("required_bytes"))?;
        let headroom = 64 * 1024 * 1024; // 64MB safety
        Self::will_fit_checked(required, headroom)?;

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_f32, &self.stream)? };
        // Make small param copies async as well to avoid host stalls
        let d_periods = unsafe { DeviceBuffer::from_slice_async(&prepared.periods_i32, &self.stream)? };
        let d_alphas = unsafe { DeviceBuffer::from_slice_async(&prepared.alphas_f32, &self.stream)? };
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(prepared.series_len * n_combos, &self.stream)? };

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_alphas,
            prepared.series_len,
            prepared.first_valid,
            n_combos,
            &mut d_out,
        )?;

        // Ensure completion for VRAM handle consistency
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ema_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEmaError> {
        if series_len == 0 {
            return Err(CudaEmaError::InvalidInput("series_len must be positive".into()));
        }
        if first_valid >= series_len {
            return Err(CudaEmaError::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, series_len
            )));
        }
        if n_combos == 0 {
            return Err(CudaEmaError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if d_periods.len() != n_combos || d_alphas.len() != n_combos {
            return Err(CudaEmaError::InvalidInput(
                "period/alpha buffer length mismatch".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaEmaError::InvalidInput(
                "prices length must match series_len".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaEmaError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_alphas,
            series_len,
            first_valid,
            n_combos,
            d_out,
        )
    }

    pub fn ema_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &EmaBatchRange,
        out_flat: &mut [f32],
    ) -> Result<(), CudaEmaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        if out_flat.len() != prepared.series_len * prepared.combos.len() {
            return Err(CudaEmaError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.ema_batch_dev(data_f32, sweep)?;
        handle.buf.copy_to(out_flat).map_err(Into::into)
    }

    pub fn ema_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &EmaParams,
    ) -> Result<DeviceArrayF32, CudaEmaError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        // VRAM estimate and async H2D copy
        let prices_bytes = num_series
            .checked_mul(series_len)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(CudaEmaError::ArithmeticOverflow("prices_bytes"))?;
        let params_bytes = prepared
            .first_valids
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(CudaEmaError::ArithmeticOverflow("params_bytes"))?;
        let out_bytes = num_series
            .checked_mul(series_len)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(CudaEmaError::ArithmeticOverflow("out_bytes"))?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or(CudaEmaError::ArithmeticOverflow("required_bytes"))?;
        let headroom = 64 * 1024 * 1024; // 64MB safety
        Self::will_fit_checked(required, headroom)?;

        let d_prices = unsafe { DeviceBuffer::from_slice_async(data_tm_f32, &self.stream)? };
        let d_first = unsafe { DeviceBuffer::from_slice_async(&prepared.first_valids, &self.stream)? };
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized_async(num_series * series_len, &self.stream)? };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            prepared.period,
            prepared.alpha,
            num_series,
            series_len,
            &mut d_out,
        )?;

        // Ensure completion for VRAM handle consistency.
        self.stream.synchronize()?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ema_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEmaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaEmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if period <= 0 {
            return Err(CudaEmaError::InvalidInput("period must be positive".into()));
        }
        let total = num_series * series_len;
        if d_prices_tm.len() != total || d_out_tm.len() != total {
            return Err(CudaEmaError::InvalidInput(
                "time-major buffer length mismatch".into(),
            ));
        }
        if d_first_valids.len() != num_series {
            return Err(CudaEmaError::InvalidInput(
                "first_valids buffer length mismatch".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            period,
            alpha,
            num_series,
            series_len,
            d_out_tm,
        )
    }

    pub fn ema_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &EmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaEmaError> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaEmaError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.ema_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        handle
            .buf
            .copy_to(out_tm)
            .map_err(CudaEmaError::Cuda)
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEmaError> {
        if n_combos == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("ema_batch_f32")
            .map_err(|_| CudaEmaError::MissingKernelSymbol { name: "ema_batch_f32" })?;

        // Policy/env block size selection
        let mut block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } => block_x,
            BatchKernelPolicy::Auto => env::var("EMA_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(256),
        };
        if block_x == 0 {
            block_x = 256;
        }
        if block_x > self.max_threads_per_block {
            return Err(CudaEmaError::LaunchConfigTooLarge {
                gx: n_combos as u32,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        // Introspection (once per scenario when BENCH_DEBUG=1)
        unsafe {
            (*(self as *const _ as *mut CudaEma)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        // Grid limit guard: chunk by device max grid.x
        let cap = self.max_grid_x.max(1).min(usize::MAX / 2);
        for (start, len) in Self::grid_chunks(n_combos, cap) {
            let grid: GridSize = (len as u32, 1, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();

            let out_ptr = unsafe { d_out.as_device_ptr().add(start * series_len) };
            let periods_ptr = unsafe { d_periods.as_device_ptr().add(start) };
            let alphas_ptr = unsafe { d_alphas.as_device_ptr().add(start) };

            let series_len_i = series_len as i32;
            let first_valid_i = first_valid as i32;
            let n_combos_i = len as i32;

            let stream = &self.stream;
            unsafe {
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        d_prices.as_device_ptr(),
                        periods_ptr,
                        alphas_ptr,
                        series_len_i,
                        first_valid_i,
                        n_combos_i,
                        out_ptr
                    )
                )?;
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEmaError> {
        if num_series == 0 {
            return Ok(());
        }

        // Resolve kernel function (standard one-block-per-series variant)
        let func = self
            .module
            .get_function("ema_many_series_one_param_f32")
            .map_err(|_| CudaEmaError::MissingKernelSymbol { name: "ema_many_series_one_param_f32" })?;

        // Decide block size from policy/env, then normalize to warp multiple and cap
        let mut block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } => block_x,
            ManySeriesKernelPolicy::Auto => env::var("EMA_MS_BLOCK_X")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(128),
        };
        if block_x == 0 {
            block_x = 128;
        }
        if block_x > self.max_threads_per_block {
            return Err(CudaEmaError::LaunchConfigTooLarge {
                gx: num_series as u32,
                gy: 1,
                gz: 1,
                bx: block_x,
                by: 1,
                bz: 1,
            });
        }

        // Introspection
        unsafe {
            (*(self as *const _ as *mut CudaEma)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        let stream = &self.stream;
        unsafe {
            launch!(
                func<<<grid, block, 0, stream>>>(
                    d_prices_tm.as_device_ptr(),
                    d_first_valids.as_device_ptr(),
                    period,
                    alpha,
                    num_series as i32,
                    series_len as i32,
                    d_out_tm.as_device_ptr()
                )
            )?;
        }

        Ok(())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &EmaBatchRange,
    ) -> Result<PreparedEmaBatch, CudaEmaError> {
        if data_f32.is_empty() {
            return Err(CudaEmaError::InvalidInput("input data is empty".into()));
        }

        let combos = expand_grid(sweep)?;

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaEmaError::InvalidInput("all values are NaN".into()))?;

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut alphas_f32 = Vec::with_capacity(combos.len());

        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaEmaError::InvalidInput("period must be positive".into()));
            }
            if series_len - first_valid < period {
                return Err(CudaEmaError::InvalidInput(format!(
                    "not enough valid data: need {} valid samples, have {}",
                    period,
                    series_len - first_valid
                )));
            }
            periods_i32.push(period as i32);
            alphas_f32.push(2.0f32 / (period as f32 + 1.0f32));
        }

        Ok(PreparedEmaBatch {
            combos,
            first_valid,
            series_len,
            periods_i32,
            alphas_f32,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &EmaParams,
    ) -> Result<PreparedEmaManySeries, CudaEmaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaEmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaEmaError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }

        let period = params.period.unwrap_or(0) as i32;
        if period <= 0 {
            return Err(CudaEmaError::InvalidInput("period must be positive".into()));
        }

        let alpha = 2.0f32 / (period as f32 + 1.0f32);

        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + series];
                if v.is_finite() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaEmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            let remaining = series_len - fv as usize;
            if remaining < period as usize {
                return Err(CudaEmaError::InvalidInput(format!(
                    "series {} does not have enough valid data: need {} valid samples, have {}",
                    series, period, remaining
                )));
            }
            first_valids.push(fv);
        }

        Ok(PreparedEmaManySeries {
            first_valids,
            period,
            alpha,
            num_series,
            series_len,
        })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        ema_benches,
        CudaEma,
        crate::indicators::moving_averages::ema::EmaBatchRange,
        crate::indicators::moving_averages::ema::EmaParams,
        ema_batch_dev,
        ema_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::ema::EmaBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::ema::EmaParams { period: Some(64) },
        "ema",
        "ema"
    );
    pub use ema_benches::bench_profiles;
}

fn expand_grid(range: &EmaBatchRange) -> Result<Vec<EmaParams>, CudaEmaError> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        let (lo, hi) = if start <= end { (start, end) } else { (end, start) };
        (lo..=hi).step_by(step).collect()
    }

    let vals = axis(range.period);
    if vals.is_empty() {
        return Err(CudaEmaError::InvalidInput(format!(
            "invalid range: start={} end={} step={}",
            range.period.0, range.period.1, range.period.2
        )));
    }
    Ok(vals.into_iter().map(|p| EmaParams { period: Some(p) }).collect())
}

// ---------- Utilities (VRAM + debug) ----------

impl CudaEma {
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit_checked(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaEmaError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        match mem_get_info() {
            Ok((free, _total)) => {
                let need = required_bytes
                    .checked_add(headroom_bytes)
                    .ok_or(CudaEmaError::ArithmeticOverflow("required+headroom"))?;
                if need <= free {
                    Ok(())
                } else {
                    Err(CudaEmaError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
                }
            }
            Err(_) => Ok(()),
        }
    }

    #[inline]
    fn grid_chunks(n: usize, max_chunk: usize) -> impl Iterator<Item = (usize, usize)> {
        (0..n).step_by(max_chunk).map(move |start| {
            let len = (n - start).min(max_chunk);
            (start, len)
        })
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        if self.debug_batch_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                eprintln!("[DEBUG] EMA batch selected kernel: {:?}", sel);
                unsafe {
                    (*(self as *const _ as *mut CudaEma)).debug_batch_logged = true;
                }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        if self.debug_many_logged {
            return;
        }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                eprintln!("[DEBUG] EMA many-series selected kernel: {:?}", sel);
                unsafe {
                    (*(self as *const _ as *mut CudaEma)).debug_many_logged = true;
                }
            }
        }
    }
}
