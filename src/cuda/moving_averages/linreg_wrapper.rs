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
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------- Kernel policy & selection (mirrors ALMA structure) ----------------

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    /// 1D launch, `block_x` threads per block
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    /// 1D launch across series, `block_x` threads per block
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaLinregPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaLinregPolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { Plain { block_x: u32 } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }

#[derive(Debug)]
pub enum CudaLinregError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaLinregError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaLinregError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaLinregError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaLinregError {}

pub struct CudaLinreg {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaLinregPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaLinreg {
    pub fn new(device_id: usize) -> Result<Self, CudaLinregError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaLinregError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaLinregError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/linreg_kernel.ptx"));
        // Match ALMA-style JIT preference: O2 + DetermineTargetFromContext with fallbacks.
        let module = match Module::from_ptx(
            ptx,
            &[ModuleJitOption::DetermineTargetFromContext, ModuleJitOption::OptLevel(OptLevel::O2)],
        ) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])
                    .map_err(|e| CudaLinregError::Cuda(e.to_string()))?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaLinregPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    /// Create with explicit policy.
    pub fn new_with_policy(device_id: usize, policy: CudaLinregPolicy) -> Result<Self, CudaLinregError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaLinregPolicy) { self.policy = policy; }
    #[inline]
    pub fn policy(&self) -> &CudaLinregPolicy { &self.policy }
    #[inline]
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    #[inline]
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }

    /// Optional synchronizer for benches/tests.
    pub fn synchronize(&self) -> Result<(), CudaLinregError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))
    }

    #[inline]
    fn maybe_log_batch_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_batch_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_batch {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] LINREG batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaLinreg)).debug_batch_logged = true; }
            }
        }
    }

    #[inline]
    fn maybe_log_many_debug(&self) {
        static GLOBAL_ONCE: AtomicBool = AtomicBool::new(false);
        if self.debug_many_logged { return; }
        if std::env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            if let Some(sel) = self.last_many {
                let per_scenario = std::env::var("BENCH_DEBUG_SCOPE").ok().as_deref() == Some("scenario");
                if per_scenario || !GLOBAL_ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!("[DEBUG] LINREG many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaLinreg)).debug_many_logged = true; }
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
    fn device_mem_info() -> Option<(usize, usize)> { cust::memory::mem_get_info().ok() }
    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else { true }
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
            return Err(CudaLinregError::InvalidInput(
                "no parameter combinations".into(),
            ));
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
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;
        // Kernel policy selection (only Plain supported for LINREG currently)
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 128,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
        };
        let grid_x = ((combos_len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe { (*(self as *const _ as *mut CudaLinreg)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }
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

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;
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
        let prices_bytes = len * std::mem::size_of::<f32>();
        let params_bytes = combos_len
            * (std::mem::size_of::<i32>()
                + std::mem::size_of::<f32>() * 3);
        let out_bytes = combos_len * len * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64MB safety margin
        if !Self::will_fit(required, headroom) {
            return Err(CudaLinregError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaLinregError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(periods_i32)
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;
        let d_x_sums =
            DeviceBuffer::from_slice(x_sums).map_err(|e| CudaLinregError::Cuda(e.to_string()))?;
        let d_denom_invs = DeviceBuffer::from_slice(denom_invs)
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;
        let d_inv_periods = DeviceBuffer::from_slice(inv_periods)
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;

        let elems = combos_len * len;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;

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

        // Ensure completion before returning VRAM handle
        self.stream
            .synchronize()
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;

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
        dev.buf
            .copy_to(out)
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;
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
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe { (*(self as *const _ as *mut CudaLinreg)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
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

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;
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
        let elems = cols * rows;
        let prices_bytes = elems * std::mem::size_of::<f32>();
        let first_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let required = prices_bytes + first_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaLinregError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }
        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices, &d_first, cols, rows, period, x_sum, denom_inv, inv_period, &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))?;

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
        self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            x_sum,
            denom_inv,
            inv_period,
        )
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
        dev.buf
            .copy_to(out_tm)
            .map_err(|e| CudaLinregError::Cuda(e.to_string()))
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
        crate::indicators::moving_averages::linreg::LinRegBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::linreg::LinRegParams { period: Some(64) },
        "linreg",
        "linreg"
    );
    pub use linreg_benches::bench_profiles;
}
