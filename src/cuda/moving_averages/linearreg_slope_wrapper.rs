//! CUDA scaffolding for Linear Regression Slope (outputs the slope `b`).
//!
//! Mirrors the LINREG wrapper structure and ALMA policies: PTX load with
//! DetermineTargetFromContext + OptLevel, NON_BLOCKING stream, VRAM checks,
//! and one-series×many-params + many-series×one-param device entrypoints.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::linearreg_slope::{
    LinearRegSlopeBatchRange, LinearRegSlopeParams,
};
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::memory::{AsyncCopyDestination, CopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------- Kernel policy (simple 1D for both paths) ----------------

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
pub struct CudaLinearregSlopePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaLinearregSlopePolicy {
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

#[derive(Debug)]
pub enum CudaLinearregSlopeError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaLinearregSlopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaLinearregSlopeError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaLinearregSlopeError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}
impl std::error::Error for CudaLinearregSlopeError {}

pub struct CudaLinearregSlope {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaLinearregSlopePolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    sm_count: i32,
}

impl CudaLinearregSlope {
    pub fn new(device_id: usize) -> Result<Self, CudaLinearregSlopeError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        let context = Context::new(device)
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/linearreg_slope_kernel.ptx"));
        let module = match Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        ) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])
                    .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?,
            },
        };

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaLinearregSlopePolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            sm_count,
        })
    }

    #[inline]
    pub fn set_policy(&mut self, policy: CudaLinearregSlopePolicy) {
        self.policy = policy;
    }
    #[inline]
    pub fn policy(&self) -> &CudaLinearregSlopePolicy {
        &self.policy
    }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaLinearregSlopeError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))
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
                    eprintln!("[DEBUG] LRS batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut Self)).debug_batch_logged = true; }
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
                    eprintln!("[DEBUG] LRS many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut Self)).debug_many_logged = true; }
            }
        }
    }

    // ---------------- VRAM helpers ----------------
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
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    // ---------------- Input prep (batch) ----------------
    #[allow(clippy::type_complexity)]
    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &LinearRegSlopeBatchRange,
    ) -> Result<(Vec<LinearRegSlopeParams>, usize, usize, Vec<i32>, Vec<f32>, Vec<f32>), CudaLinearregSlopeError>
    {
        if data_f32.is_empty() {
            return Err(CudaLinearregSlopeError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaLinearregSlopeError::InvalidInput("all values are NaN".into()))?;

        let combos = {
            // replicate expansion from indicator
            let (start, end, step) = sweep.period;
            let periods: Vec<usize> = if step == 0 || start == end {
                vec![start]
            } else {
                (start..=end).step_by(step).collect()
            };
            periods
                .into_iter()
                .map(|p| LinearRegSlopeParams { period: Some(p) })
                .collect::<Vec<_>>()
        };
        if combos.is_empty() {
            return Err(CudaLinearregSlopeError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut x_sums = Vec::with_capacity(combos.len());
        let mut denom_invs = Vec::with_capacity(combos.len());

        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period < 2 {
                return Err(CudaLinearregSlopeError::InvalidInput(
                    "period must be >= 2".into(),
                ));
            }
            if period > len {
                return Err(CudaLinearregSlopeError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaLinearregSlopeError::InvalidInput(format!(
                    "not enough valid data for period {} (tail={})",
                    period,
                    len - first_valid
                )));
            }

            let pf = period as f64;
            let x_sum = pf * (pf + 1.0) * 0.5; // x=1..period
            let x2_sum = pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0;
            let denom = pf * x2_sum - x_sum * x_sum;
            let denom_inv = 1.0 / denom;

            periods_i32.push(period as i32);
            x_sums.push(x_sum as f32);
            denom_invs.push(denom_inv as f32);
        }

        Ok((combos, first_valid, len, periods_i32, x_sums, denom_invs))
    }

    fn grid_1d_for(&self, work_items: usize, block_x: u32) -> GridSize {
        let gx = ((work_items as u32) + block_x - 1) / block_x;
        (gx.max(self.sm_count as u32), 1, 1).into()
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_x_sums: &DeviceBuffer<f32>,
        d_denom_invs: &DeviceBuffer<f32>,
        series_len: usize,
        combos_len: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaLinearregSlopeError> {
        let func = self
            .module
            .get_function("linearreg_slope_batch_f32")
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
        };
        let grid = self.grid_1d_for(combos_len, block_x);
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut Self)).last_batch = Some(BatchKernelSelected::Plain {
                block_x,
            });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut xs_ptr = d_x_sums.as_device_ptr().as_raw();
            let mut dinv_ptr = d_denom_invs.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_len_i = combos_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut xs_ptr as *mut _ as *mut c_void,
                &mut dinv_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        periods_i32: &[i32],
        x_sums: &[f32],
        denom_invs: &[f32],
        len: usize,
        first_valid: usize,
    ) -> Result<DeviceArrayF32, CudaLinearregSlopeError> {
        let nrows = periods_i32.len();
        let elems = nrows * len;
        let prices_bytes = len * std::mem::size_of::<f32>();
        let periods_bytes = nrows * std::mem::size_of::<i32>();
        let consts_bytes = nrows * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + consts_bytes * 2 + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaLinearregSlopeError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }
        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(periods_i32)
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        let d_xs = DeviceBuffer::from_slice(x_sums)
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        let d_dinv = DeviceBuffer::from_slice(denom_invs)
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices, &d_periods, &d_xs, &d_dinv, len, nrows, first_valid, &mut d_out,
        )?;
        Ok(DeviceArrayF32 { buf: d_out, rows: nrows, cols: len })
    }

    pub fn linearreg_slope_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &LinearRegSlopeBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<LinearRegSlopeParams>), CudaLinearregSlopeError> {
        let (combos, first_valid, len, periods_i32, x_sums, denom_invs) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(
            data_f32,
            &periods_i32,
            &x_sums,
            &denom_invs,
            len,
            first_valid,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        Ok((dev, combos))
    }

    pub fn linearreg_slope_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &LinearRegSlopeBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<LinearRegSlopeParams>), CudaLinearregSlopeError> {
        let (combos, first_valid, len, periods_i32, x_sums, denom_invs) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let nrows = combos.len();
        if out.len() != nrows * len {
            return Err(CudaLinearregSlopeError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                nrows * len,
                out.len()
            )));
        }
        let dev = self.run_batch_kernel(
            data_f32,
            &periods_i32,
            &x_sums,
            &denom_invs,
            len,
            first_valid,
        )?;
        dev.buf
            .copy_to(out)
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        self.stream
            .synchronize()
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        Ok((nrows, len, combos))
    }

    // ---------------- Many-series one param ----------------
    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &LinearRegSlopeParams,
    ) -> Result<(Vec<i32>, usize, f32, f32), CudaLinearregSlopeError> {
        if data_tm_f32.len() != cols * rows {
            return Err(CudaLinearregSlopeError::InvalidInput(
                "invalid time-major shape".into(),
            ));
        }
        let period = params.period.unwrap_or(0);
        if period < 2 || period > rows {
            return Err(CudaLinearregSlopeError::InvalidInput(format!(
                "invalid period {} for rows {}",
                period, rows
            )));
        }
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for r in 0..rows {
                let v = data_tm_f32[r * cols + s];
                if !v.is_nan() {
                    fv = r as i32;
                    break;
                }
            }
            if fv < 0 {
                return Err(CudaLinearregSlopeError::InvalidInput(format!(
                    "series {} contains only NaN",
                    s
                )));
            }
            first_valids[s] = fv;
            if (rows as i32 - fv) < period as i32 {
                return Err(CudaLinearregSlopeError::InvalidInput(format!(
                    "series {} insufficient data for period {} (first_valid={}, rows={})",
                    s, period, fv, rows
                )));
            }
        }
        let pf = period as f64;
        let x_sum = (pf * (pf + 1.0) * 0.5) as f32;
        let x2_sum = (pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0) as f32;
        let denom_inv = (1.0f64 / (pf * x2_sum as f64 - (x_sum as f64) * (x_sum as f64))) as f32;
        Ok((first_valids, period, x_sum, denom_inv))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        x_sum: f32,
        denom_inv: f32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaLinearregSlopeError> {
        let func = self
            .module
            .get_function("linearreg_slope_many_series_one_param_f32")
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
        };
        let grid = self.grid_1d_for(cols, block_x);
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe { (*(self as *const _ as *mut Self)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }
        self.maybe_log_many_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut x_sum_f = x_sum;
            let mut denom_inv_f = denom_inv;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut x_sum_f as *mut _ as *mut c_void,
                &mut denom_inv_f as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
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
    ) -> Result<DeviceArrayF32, CudaLinearregSlopeError> {
        let elems = cols * rows;
        let prices_bytes = elems * std::mem::size_of::<f32>();
        let first_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let required = prices_bytes + first_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaLinearregSlopeError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }
        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            cols,
            rows,
            period,
            x_sum,
            denom_inv,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }

    pub fn linearreg_slope_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &LinearRegSlopeParams,
    ) -> Result<DeviceArrayF32, CudaLinearregSlopeError> {
        let (first_valids, period, x_sum, denom_inv) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(
            data_tm_f32, cols, rows, &first_valids, period, x_sum, denom_inv,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaLinearregSlopeError::Cuda(e.to_string()))?;
        Ok(dev)
    }
}

// ---------------- Benches ----------------
pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        linearreg_slope_benches,
        CudaLinearregSlope,
        crate::indicators::linearreg_slope::LinearRegSlopeBatchRange,
        crate::indicators::linearreg_slope::LinearRegSlopeParams,
        linearreg_slope_batch_dev,
        linearreg_slope_many_series_one_param_time_major_dev,
        crate::indicators::linearreg_slope::LinearRegSlopeBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::linearreg_slope::LinearRegSlopeParams { period: Some(64) },
        "linearreg_slope",
        "linearreg_slope"
    );
    pub use linearreg_slope_benches::bench_profiles;
}

