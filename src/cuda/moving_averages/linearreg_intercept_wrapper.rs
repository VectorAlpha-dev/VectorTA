//! CUDA scaffolding for the Linear Regression Intercept indicator.
//!
//! Parity with ALMA wrapper conventions:
//! - PTX load via DetermineTargetFromContext with O4 (fallbacks to simpler)
//! - NON_BLOCKING stream
//! - Policy enums for batch and many-series 1D kernels
//! - VRAM estimate + ~64MB headroom checks
//! - Warmup/NaN semantics identical to scalar implementation
//! - Public device entry points for one-series×many-params and many-series×one-param

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::linearreg_intercept::{
    LinearRegInterceptBatchRange, LinearRegInterceptParams,
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

// ---------------- Kernel policy & selection ----------------

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
pub struct CudaLinregInterceptPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaLinregInterceptPolicy {
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
pub enum CudaLinregInterceptError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaLinregInterceptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaLinregInterceptError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaLinregInterceptError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}
impl std::error::Error for CudaLinregInterceptError {}

pub struct CudaLinregIntercept {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaLinregInterceptPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
    sm_count: i32,
}

impl CudaLinregIntercept {
    pub fn new(device_id: usize) -> Result<Self, CudaLinregInterceptError> {
        cust::init(CudaFlags::empty())
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        let context =
            Context::new(device).map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/linearreg_intercept_kernel.ptx"));
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
                    .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaLinregInterceptPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
            sm_count,
        })
    }

    pub fn set_policy(&mut self, policy: CudaLinregInterceptPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaLinregInterceptPolicy {
        &self.policy
    }
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> {
        self.last_batch
    }
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> {
        self.last_many
    }
    pub fn synchronize(&self) -> Result<(), CudaLinregInterceptError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))
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
                    eprintln!(
                        "[DEBUG] LINEARREG_INTERCEPT batch selected kernel: {:?}",
                        sel
                    );
                }
                unsafe {
                    (*(self as *const _ as *mut CudaLinregIntercept)).debug_batch_logged = true;
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
                    eprintln!(
                        "[DEBUG] LINEARREG_INTERCEPT many-series selected kernel: {:?}",
                        sel
                    );
                }
                unsafe {
                    (*(self as *const _ as *mut CudaLinregIntercept)).debug_many_logged = true;
                }
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
        if let Some((free, _)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }

    // ---------------- Batch: prepare inputs ----------------
    #[allow(clippy::type_complexity)]
    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &LinearRegInterceptBatchRange,
    ) -> Result<
        (
            Vec<LinearRegInterceptParams>,
            usize,
            usize,
            Vec<i32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
        ),
        CudaLinregInterceptError,
    > {
        if data_f32.is_empty() {
            return Err(CudaLinregInterceptError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaLinregInterceptError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_params(sweep);
        if combos.is_empty() {
            return Err(CudaLinregInterceptError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut x_sums = Vec::with_capacity(combos.len());
        let mut denom_invs = Vec::with_capacity(combos.len());
        let mut inv_periods = Vec::with_capacity(combos.len());

        for c in &combos {
            let p = c.period.unwrap_or(0);
            if p == 0 {
                return Err(CudaLinregInterceptError::InvalidInput(
                    "period must be at least 1".into(),
                ));
            }
            if p > len {
                return Err(CudaLinregInterceptError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    p, len
                )));
            }
            if len - first_valid < p {
                return Err(CudaLinregInterceptError::InvalidInput(format!(
                    "not enough valid data for period {} (tail = {})",
                    p,
                    len - first_valid
                )));
            }

            let pf = p as f64;
            let x_sum = pf * (pf + 1.0) * 0.5;
            let x2_sum = pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0;
            let denom = pf * x2_sum - x_sum * x_sum;
            periods_i32.push(p as i32);
            x_sums.push(x_sum as f32);
            denom_invs.push((1.0 / denom) as f32);
            inv_periods.push((1.0 / pf) as f32);
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
    ) -> Result<(), CudaLinregInterceptError> {
        let func = self
            .module
            .get_function("linearreg_intercept_batch_f32")
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
        };
        let grid: GridSize = self.grid_1d_for(combos_len, block_x);
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaLinregIntercept)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        self.maybe_log_batch_debug();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut xs_ptr = d_x_sums.as_device_ptr().as_raw();
            let mut dinv_ptr = d_denom_invs.as_device_ptr().as_raw();
            let mut invp_ptr = d_inv_periods.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_len_i = combos_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut xs_ptr as *mut _ as *mut c_void,
                &mut dinv_ptr as *mut _ as *mut c_void,
                &mut invp_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
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
    ) -> Result<DeviceArrayF32, CudaLinregInterceptError> {
        // VRAM estimate
        let prices_bytes = len * std::mem::size_of::<f32>();
        let params_bytes =
            combos_len * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>() * 3);
        let out_bytes = combos_len * len * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaLinregInterceptError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(periods_i32)
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        let d_x_sums = DeviceBuffer::from_slice(x_sums)
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        let d_denom_invs = DeviceBuffer::from_slice(denom_invs)
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        let d_inv_periods = DeviceBuffer::from_slice(inv_periods)
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        let elems = combos_len * len;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
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

    pub fn linearreg_intercept_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &LinearRegInterceptBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<LinearRegInterceptParams>), CudaLinregInterceptError> {
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
        self.stream
            .synchronize()
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        Ok((dev, combos))
    }

    pub fn linearreg_intercept_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &LinearRegInterceptBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<LinearRegInterceptParams>), CudaLinregInterceptError> {
        let (combos, first_valid, len, periods_i32, x_sums, denom_invs, inv_periods) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaLinregInterceptError::InvalidInput(format!(
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
        self.stream
            .synchronize()
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        dev.buf
            .copy_to(out)
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        Ok((combos.len(), len, combos))
    }

    // ---------------- Many-series × one-param ----------------
    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &LinearRegInterceptParams,
    ) -> Result<(Vec<i32>, usize, f32, f32, f32), CudaLinregInterceptError> {
        if cols == 0 || rows == 0 {
            return Err(CudaLinregInterceptError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaLinregInterceptError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaLinregInterceptError::InvalidInput(
                "period must be at least 1".into(),
            ));
        }
        if period > rows {
            return Err(CudaLinregInterceptError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for r in 0..rows {
                let idx = r * cols + s;
                if !data_tm_f32[idx].is_nan() {
                    fv = Some(r);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaLinregInterceptError::InvalidInput(format!("series {} all NaN", s))
            })?;
            if rows - fv < period {
                return Err(CudaLinregInterceptError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {})",
                    s,
                    period,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }

        let pf = period as f64;
        let x_sum = pf * (pf + 1.0) * 0.5;
        let x2_sum = pf * (pf + 1.0) * (2.0 * pf + 1.0) / 6.0;
        let denom = pf * x2_sum - x_sum * x_sum;
        Ok((
            first_valids,
            period,
            x_sum as f32,
            (1.0 / denom) as f32,
            (1.0 / pf) as f32,
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
    ) -> Result<(), CudaLinregInterceptError> {
        let func = self
            .module
            .get_function("linearreg_intercept_many_series_one_param_f32")
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
        };
        let grid: GridSize = self.grid_1d_for(cols, block_x);
        let block: BlockSize = (block_x, 1, 1).into();
        unsafe {
            (*(self as *const _ as *mut CudaLinregIntercept)).last_many =
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
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
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
    ) -> Result<DeviceArrayF32, CudaLinregInterceptError> {
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let params_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaLinregInterceptError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(cols * rows) }
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_first_valids,
            cols,
            rows,
            period,
            x_sum,
            denom_inv,
            inv_period,
            &mut d_out,
        )?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn linearreg_intercept_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &LinearRegInterceptParams,
    ) -> Result<DeviceArrayF32, CudaLinregInterceptError> {
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
        self.stream
            .synchronize()
            .map_err(|e| CudaLinregInterceptError::Cuda(e.to_string()))?;
        Ok(dev)
    }

    // grid helper
    #[inline]
    fn grid_1d_for(&self, work_items: usize, block_x: u32) -> GridSize {
        let blocks_needed = ((work_items as u32).saturating_add(block_x - 1)) / block_x;
        let max_blocks = (self.sm_count as u32).saturating_mul(32).max(1);
        let grid_x = core::cmp::min(blocks_needed.max(1), max_blocks);
        (grid_x, 1, 1).into()
    }
}

#[inline]
fn expand_grid_params(r: &LinearRegInterceptBatchRange) -> Vec<LinearRegInterceptParams> {
    let (start, end, step) = r.period;
    let periods: Vec<usize> = if step == 0 || start == end {
        vec![start]
    } else {
        (start..=end).step_by(step).collect()
    };
    let mut out = Vec::with_capacity(periods.len());
    for p in periods {
        out.push(LinearRegInterceptParams { period: Some(p) });
    }
    out
}

// ---------- Benches ----------
pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        linearreg_intercept_benches,
        CudaLinregIntercept,
        crate::indicators::linearreg_intercept::LinearRegInterceptBatchRange,
        crate::indicators::linearreg_intercept::LinearRegInterceptParams,
        linearreg_intercept_batch_dev,
        linearreg_intercept_many_series_one_param_time_major_dev,
        crate::indicators::linearreg_intercept::LinearRegInterceptBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::linearreg_intercept::LinearRegInterceptParams { period: Some(64) },
        "linearreg_intercept",
        "linearreg_intercept"
    );
    pub use linearreg_intercept_benches::bench_profiles;
}
