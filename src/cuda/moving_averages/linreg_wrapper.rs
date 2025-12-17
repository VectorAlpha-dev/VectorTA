//! CUDA scaffolding for the Linear Regression (LINREG) indicator.
//!
//! ALMA-aligned wrapper with:
//! - Policy enums for kernel selection (kept simple for LINREG: 1D kernels).
//! - PTX JIT options with DetermineTargetFromContext and O2, with fallbacks.
//! - VRAM estimation with 64MB headroom and early failure on OOM risk.
//! - Warmup/NaN behavior identical to the scalar reference.
//! - Public APIs for one-series×many-params and many-series×one-param.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::linreg::{
    expand_grid_linreg, LinRegBatchRange, LinRegParams,
};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{AsyncCopyDestination, CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use thiserror::Error;
use std::env;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// ---------------- Kernel policy & selection (mirrors ALMA structure) ----------------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    /// 1D launch, `block_x` threads per block
    Plain {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    /// 1D launch across series, `block_x` threads per block
    OneD {
        block_x: u32,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaLinregPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaLinregPolicy {
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

#[derive(Debug, Error)]
pub enum CudaLinregError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Out of memory on device: required={required}B, free={free}B, headroom={headroom}B")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("Missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("Invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("Launch config too large (grid=({gx},{gy},{gz}), block=({bx},{by},{bz}))")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("device mismatch: buf on {buf}, current {current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
    #[error("arithmetic overflow when computing {what}")]
    ArithmeticOverflow { what: &'static str },
}

pub struct CudaLinreg {
    module: Module,
    stream: Stream,
    ctx: Arc<Context>,
    device_id: u32,
    policy: CudaLinregPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    sm_count: i32, // number of SMs on the active device
}

impl CudaLinreg {
    pub fn new(device_id: usize) -> Result<Self, CudaLinregError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let sm_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)?;
        let context = Context::new(device)?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/linreg_kernel.ptx"));
        // Prefer DetermineTargetFromContext with OptLevel O4 (most optimized).
        let module = match Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O4),
            ],
        ) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            ctx: Arc::new(context),
            device_id: device_id as u32,
            policy: CudaLinregPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            sm_count,
        })
    }

    /// Create with explicit policy.
    pub fn new_with_policy(
        device_id: usize,
        policy: CudaLinregPolicy,
    ) -> Result<Self, CudaLinregError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaLinregPolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn ctx(&self) -> Arc<Context> { Arc::clone(&self.ctx) }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }
    #[inline]
    pub fn policy(&self) -> &CudaLinregPolicy {
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

    /// Optional synchronizer for benches/tests.
    pub fn synchronize(&self) -> Result<(), CudaLinregError> {
        self.stream.synchronize().map_err(CudaLinregError::from)
    }

    /// Expose the producing stream handle for CUDA Array Interface (v3) interop.
    pub fn stream_handle_u64(&self) -> u64 {
        self.stream.as_inner() as u64
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
                    eprintln!("[DEBUG] LINREG batch selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaLinreg)).debug_batch_logged = true;
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
                    eprintln!("[DEBUG] LINREG many-series selected kernel: {:?}", sel);
                }
                unsafe {
                    (*(self as *const _ as *mut CudaLinreg)).debug_many_logged = true;
                }
            }
        }
    }

    // --------------- VRAM checks ---------------
    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }
    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        cust::memory::mem_get_info().ok()
    }
    #[inline]
    fn will_fit_checked(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaLinregError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        let (free, _total) = match Self::device_mem_info() {
            Some(v) => v,
            None => return Ok(()),
        };
        let need = required_bytes
            .checked_add(headroom_bytes)
            .ok_or(CudaLinregError::ArithmeticOverflow { what: "required_bytes + headroom_bytes" })?;
        if need <= free {
            Ok(())
        } else {
            Err(CudaLinregError::OutOfMemory { required: required_bytes, free, headroom: headroom_bytes })
        }
    }

    #[allow(clippy::type_complexity)]
    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &LinRegBatchRange,
    ) -> Result<
        (
            Vec<LinRegParams>,
            usize,
            usize,
            Vec<i32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
        ),
        CudaLinregError,
    > {
        if data_f32.is_empty() {
            return Err(CudaLinregError::InvalidInput("empty data".into()));
        }

        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaLinregError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_linreg(sweep);
        if combos.is_empty() {
            let (s, e, t) = sweep.period;
            return Err(CudaLinregError::InvalidInput(format!(
                "no parameter combinations (start={}, end={}, step={})",
                s, e, t
            )));
        }

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut x_sums = Vec::with_capacity(combos.len());
        let mut denom_invs = Vec::with_capacity(combos.len());
        let mut inv_periods = Vec::with_capacity(combos.len());

        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaLinregError::InvalidInput(
                    "period must be at least 1".into(),
                ));
            }
            if period > len {
                return Err(CudaLinregError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaLinregError::InvalidInput(format!(
                    "not enough valid data for period {} (tail = {})",
                    period,
                    len - first_valid
                )));
            }

            let period_f = period as f64;
            let x_sum = period_f * (period_f + 1.0) * 0.5;
            let x2_sum = period_f * (period_f + 1.0) * (2.0 * period_f + 1.0) / 6.0;
            let denom = period_f * x2_sum - x_sum * x_sum;
            let denom_inv = 1.0 / denom;
            let inv_period = 1.0 / period_f;

            periods_i32.push(period as i32);
            x_sums.push(x_sum as f32);
            denom_invs.push(denom_inv as f32);
            inv_periods.push(inv_period as f32);
        }

        Ok((
            combos,
            first_valid,
            len,
            periods_i32,
            x_sums,
            denom_invs,
            inv_periods,
        ))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_x_sums: &DeviceBuffer<f32>,
        d_denom_invs: &DeviceBuffer<f32>,
        d_inv_periods: &DeviceBuffer<f32>,
        series_len: usize,
        combos_len: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaLinregError> {
        let func = self
            .module
            .get_function("linreg_batch_f32")
            .map_err(|_| CudaLinregError::MissingKernelSymbol { name: "linreg_batch_f32" })?;
        // Kernel policy selection (only Plain supported for LINREG currently)
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
        };
        let grid: GridSize = self.grid_1d_for(combos_len, block_x);
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            (*(self as *const _ as *mut CudaLinreg)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut x_sums_ptr = d_x_sums.as_device_ptr().as_raw();
            let mut denom_ptr = d_denom_invs.as_device_ptr().as_raw();
            let mut inv_periods_ptr = d_inv_periods.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = combos_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut x_sums_ptr as *mut _ as *mut c_void,
                &mut denom_ptr as *mut _ as *mut c_void,
                &mut inv_periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, 0, args)?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        periods_i32: &[i32],
        x_sums: &[f32],
        denom_invs: &[f32],
        inv_periods: &[f32],
        combos_len: usize,
        first_valid: usize,
        len: usize,
    ) -> Result<DeviceArrayF32, CudaLinregError> {
        // VRAM estimate and early check
        let prices_bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaLinregError::ArithmeticOverflow { what: "len * sizeof(f32)" })?;
        let per_combo = std::mem::size_of::<i32>()
            .checked_add(std::mem::size_of::<f32>() * 3)
            .ok_or(CudaLinregError::ArithmeticOverflow { what: "per_combo bytes" })?;
        let params_bytes = combos_len
            .checked_mul(per_combo)
            .ok_or(CudaLinregError::ArithmeticOverflow { what: "combos_len * per_combo" })?;
        let out_elems = combos_len
            .checked_mul(len)
            .ok_or(CudaLinregError::ArithmeticOverflow { what: "combos_len * len" })?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaLinregError::ArithmeticOverflow { what: "out_elems * sizeof(f32)" })?;
        let required = prices_bytes
            .checked_add(params_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or(CudaLinregError::ArithmeticOverflow { what: "total required bytes" })?;
        let headroom = 64 * 1024 * 1024; // 64MB safety margin
        Self::will_fit_checked(required, headroom)?;

        let d_prices = DeviceBuffer::from_slice(data_f32)?;
        let d_periods = DeviceBuffer::from_slice(periods_i32)?;
        let d_x_sums = DeviceBuffer::from_slice(x_sums)?;
        let d_denom_invs = DeviceBuffer::from_slice(denom_invs)?;
        let d_inv_periods = DeviceBuffer::from_slice(inv_periods)?;

        let elems = combos_len
            .checked_mul(len)
            .ok_or(CudaLinregError::ArithmeticOverflow { what: "combos_len * len" })?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_x_sums,
            &d_denom_invs,
            &d_inv_periods,
            len,
            combos_len,
            first_valid,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos_len,
            cols: len,
        })
    }

    pub fn linreg_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &LinRegBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<LinRegParams>), CudaLinregError> {
        let (combos, first_valid, len, periods_i32, x_sums, denom_invs, inv_periods) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(
            data_f32,
            &periods_i32,
            &x_sums,
            &denom_invs,
            &inv_periods,
            combos.len(),
            first_valid,
            len,
        )?;
        // Preserve prior semantics: synchronize before returning device buffer
        self.stream.synchronize()?;
        Ok((dev, combos))
    }

    pub fn linreg_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &LinRegBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<LinRegParams>), CudaLinregError> {
        let (combos, first_valid, len, periods_i32, x_sums, denom_invs, inv_periods) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaLinregError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }

        let dev = self.run_batch_kernel(
            data_f32,
            &periods_i32,
            &x_sums,
            &denom_invs,
            &inv_periods,
            combos.len(),
            first_valid,
            len,
        )?;
        // Single sync at the API boundary, then a blocking pageable copy
        self.stream.synchronize()?;
        dev.buf.copy_to(out)?;
        Ok((combos.len(), len, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &LinRegParams,
    ) -> Result<(Vec<i32>, usize, f32, f32, f32), CudaLinregError> {
        if cols == 0 || rows == 0 {
            return Err(CudaLinregError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaLinregError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaLinregError::InvalidInput(
                "period must be at least 1".into(),
            ));
        }
        if period > rows {
            return Err(CudaLinregError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

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
                CudaLinregError::InvalidInput(format!("series {} all NaN", series))
            })?;
            if rows - fv < period {
                return Err(CudaLinregError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {})",
                    series,
                    period,
                    rows - fv
                )));
            }
            first_valids[series] = fv as i32;
        }

        let period_f = period as f64;
        let x_sum = period_f * (period_f + 1.0) * 0.5;
        let x2_sum = period_f * (period_f + 1.0) * (2.0 * period_f + 1.0) / 6.0;
        let denom = period_f * x2_sum - x_sum * x_sum;
        let denom_inv = 1.0 / denom;
        let inv_period = 1.0 / period_f;

        Ok((
            first_valids,
            period,
            x_sum as f32,
            denom_inv as f32,
            inv_period as f32,
        ))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        x_sum: f32,
        denom_inv: f32,
        inv_period: f32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaLinregError> {
        let func = self
            .module
            .get_function("linreg_many_series_one_param_f32")
            .map_err(|_| CudaLinregError::MissingKernelSymbol { name: "linreg_many_series_one_param_f32" })?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
        };
        let grid: GridSize = self.grid_1d_for(cols, block_x);
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            (*(self as *const _ as *mut CudaLinreg)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        self.maybe_log_many_debug();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut x_sum_f = x_sum;
            let mut denom_inv_f = denom_inv;
            let mut inv_period_f = inv_period;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut x_sum_f as *mut _ as *mut c_void,
                &mut denom_inv_f as *mut _ as *mut c_void,
                &mut inv_period_f as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream.launch(&func, grid, block, 0, args)?;
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
        x_sum: f32,
        denom_inv: f32,
        inv_period: f32,
    ) -> Result<DeviceArrayF32, CudaLinregError> {
        // VRAM estimate and check
        let elems = cols
            .checked_mul(rows)
            .ok_or(CudaLinregError::ArithmeticOverflow { what: "cols * rows" })?;
        let prices_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaLinregError::ArithmeticOverflow { what: "elems * sizeof(f32)" })?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(CudaLinregError::ArithmeticOverflow { what: "cols * sizeof(i32)" })?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaLinregError::ArithmeticOverflow { what: "elems * sizeof(f32)" })?;
        let required = prices_bytes
            .checked_add(first_bytes)
            .and_then(|x| x.checked_add(out_bytes))
            .ok_or(CudaLinregError::ArithmeticOverflow { what: "total required bytes" })?;
        let headroom = 64 * 1024 * 1024;
        Self::will_fit_checked(required, headroom)?;
        let d_prices = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(first_valids)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        self.launch_many_series_kernel(
            &d_prices, &d_first, cols, rows, period, x_sum, denom_inv, inv_period, &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn linreg_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &LinRegParams,
    ) -> Result<DeviceArrayF32, CudaLinregError> {
        let (first_valids, period, x_sum, denom_inv, inv_period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            x_sum,
            denom_inv,
            inv_period,
        )?;
        // Synchronize before returning device buffer
        self.stream.synchronize()?;
        Ok(dev)
    }

    pub fn linreg_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &LinRegParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaLinregError> {
        if out_tm.len() != cols * rows {
            return Err(CudaLinregError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }
        let (first_valids, period, x_sum, denom_inv, inv_period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            x_sum,
            denom_inv,
            inv_period,
        )?;
        // Single sync then blocking pageable copy
        self.stream.synchronize()?;
        dev.buf
            .copy_to(out_tm)
            .map_err(|e| CudaLinregError::Cuda(e))
    }

    // Compute a right-sized grid for grid-stride 1D kernels
    #[inline]
    fn grid_1d_for(&self, work_items: usize, block_x: u32) -> GridSize {
        let blocks_needed = ((work_items as u32).saturating_add(block_x - 1)) / block_x;
        let max_blocks = (self.sm_count as u32).saturating_mul(32).max(1);
        let grid_x = core::cmp::min(blocks_needed.max(1), max_blocks);
        (grid_x, 1, 1).into()
    }

    /// Optional fast path: copy Device -> pinned host asynchronously, then sync once.
    /// Caller benefits when they can reuse `LockedBuffer` across runs.
    pub fn linreg_batch_into_locked_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &LinRegBatchRange,
        pinned_out: &mut cust::memory::LockedBuffer<f32>,
    ) -> Result<(usize, usize, Vec<LinRegParams>), CudaLinregError> {
        let (combos, first_valid, len, periods_i32, x_sums, denom_invs, inv_periods) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if pinned_out.len() != expected {
            return Err(CudaLinregError::InvalidInput(format!(
                "pinned output length mismatch: expected {}, got {}",
                expected,
                pinned_out.len()
            )));
        }
        let dev = self.run_batch_kernel(
            data_f32,
            &periods_i32,
            &x_sums,
            &denom_invs,
            &inv_periods,
            combos.len(),
            first_valid,
            len,
        )?;
        unsafe {
            dev.buf.async_copy_to(pinned_out.as_mut_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        Ok((combos.len(), len, combos))
    }

    /// Skip CPU scan for first_valids when already computed by the caller.
    pub fn linreg_multi_series_one_param_time_major_dev_with_first_valids(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &LinRegParams,
        first_valids: &[i32],
    ) -> Result<DeviceArrayF32, CudaLinregError> {
        if first_valids.len() != cols {
            return Err(CudaLinregError::InvalidInput(format!(
                "first_valids length mismatch: expected {}, got {}",
                cols,
                first_valids.len()
            )));
        }
        let period = params
            .period
            .ok_or_else(|| CudaLinregError::InvalidInput("period must be at least 1".into()))?
            as usize;
        if period == 0 || period > rows {
            return Err(CudaLinregError::InvalidInput(format!(
                "period {} invalid for rows {}",
                period, rows
            )));
        }
        for (s, &fv) in first_valids.iter().enumerate() {
            let fv = fv as isize;
            if fv < 0 || fv as usize >= rows || (rows - (fv as usize)) < period {
                return Err(CudaLinregError::InvalidInput(format!(
                    "series {} insufficient data for period {} (first_valid={}, rows={})",
                    s, period, fv, rows
                )));
            }
        }
        let pf = period as f64;
        let x_sum = (pf * (pf + 1.0) * 0.5) as f32;
        let x2_sum = (pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0) as f32;
        let denom_inv = (1.0f64 / (pf * x2_sum as f64 - (x_sum as f64) * (x_sum as f64))) as f32;
        let inv_period = (1.0f64 / pf) as f32;

        let dev = self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            first_valids,
            period,
            x_sum,
            denom_inv,
            inv_period,
        )?;
        self.stream.synchronize()?;
        Ok(dev)
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        linreg_benches,
        CudaLinreg,
        crate::indicators::moving_averages::linreg::LinRegBatchRange,
        crate::indicators::moving_averages::linreg::LinRegParams,
        linreg_batch_dev,
        linreg_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::linreg::LinRegBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::moving_averages::linreg::LinRegParams { period: Some(64) },
        "linreg",
        "linreg"
    );
    pub use linreg_benches::bench_profiles;
}
