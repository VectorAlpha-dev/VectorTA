//! CUDA scaffolding for Linear Regression Angle (LRA).
//!
//! Parity with ALMA wrapper conventions:
//! - PTX loaded via include_str!(concat!(env!("OUT_DIR"), "/linearreg_angle_kernel.ptx"))
//! - Stream NON_BLOCKING
//! - Simple policies for 1D kernels (batch + many-series)
//! - VRAM checks with ~64MB headroom and grid.y chunking for batch
//! - Warmup/NaN semantics identical to scalar (prefix NaN handling on batch;
//!   O(1) slide + O(period) rebuild on many-series)

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::linearreg_angle::{Linearreg_angleBatchRange, Linearreg_angleParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Clone, Debug)]
struct Combo { period: usize }

// ---------------- Kernel policy & selection (ALMA-style, simplified) ----------------

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
pub struct CudaLinearregAnglePolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaLinearregAnglePolicy {
    fn default() -> Self {
        Self { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Auto }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelSelected { Plain { block_x: u32 } }

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelSelected { OneD { block_x: u32 } }

#[derive(Debug)]
pub enum CudaLinearregAngleError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaLinearregAngleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaLinearregAngleError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaLinearregAngleError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaLinearregAngleError {}

pub struct CudaLinearregAngle {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaLinearregAnglePolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
}

impl CudaLinearregAngle {
    pub fn new(device_id: usize) -> Result<Self, CudaLinearregAngleError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/linearreg_angle_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        )
        .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
        .or_else(|_| Module::from_ptx(ptx, &[]))
        .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaLinearregAnglePolicy::default(),
            last_batch: None,
            last_many: None,
        })
    }

    pub fn new_with_policy(
        device_id: usize,
        policy: CudaLinearregAnglePolicy,
    ) -> Result<Self, CudaLinearregAngleError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline] pub fn set_policy(&mut self, policy: CudaLinearregAnglePolicy) { self.policy = policy; }
    #[inline] pub fn policy(&self) -> &CudaLinearregAnglePolicy { &self.policy }
    #[inline] pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    #[inline] pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    #[inline] pub fn synchronize(&self) -> Result<(), CudaLinearregAngleError> {
        self.stream.synchronize().map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))
    }

    // -------- VRAM checks --------
    #[inline] fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") { Ok(v) => v != "0" && v.to_lowercase() != "false", Err(_) => true }
    }
    #[inline] fn device_mem_info() -> Option<(usize, usize)> { mem_get_info().ok() }
    #[inline] fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() { return true; }
        if let Some((free, _)) = Self::device_mem_info() { required_bytes.saturating_add(headroom_bytes) <= free } else { true }
    }

    // -------- Inputs prep --------

    fn expand_combos(range: &Linearreg_angleBatchRange) -> Vec<Combo> {
        fn axis_usize((s, e, st): (usize, usize, usize)) -> Vec<usize> {
            if st == 0 || s == e { vec![s] } else { (s..=e).step_by(st).collect() }
        }
        axis_usize(range.period)
            .into_iter()
            .map(|p| Combo { period: p })
            .collect()
    }

    fn build_prefixes_lra(data: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
        let n = data.len();
        let mut ps = vec![0.0f64; n + 1]; // Σy
        let mut pk = vec![0.0f64; n + 1]; // Σ(k*y)
        let mut pn = vec![0i32; n + 1];  // count NaN
        let mut s = 0.0f64; let mut ksum = 0.0f64; let mut cn = 0i32;
        for i in 0..n {
            let v = data[i];
            if v.is_nan() { cn += 1; }
            else { let dv = v as f64; s += dv; ksum += (i as f64) * dv; }
            ps[i + 1] = s; pk[i + 1] = ksum; pn[i + 1] = cn;
        }
        (ps, pk, pn)
    }

    #[allow(clippy::type_complexity)]
    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &Linearreg_angleBatchRange,
    ) -> Result<(
        Vec<Combo>, usize, usize, Vec<i32>, Vec<f32>, Vec<f32>, Vec<f64>, Vec<f64>, Vec<i32>
    ), CudaLinearregAngleError> {
        if data_f32.is_empty() {
            return Err(CudaLinearregAngleError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32.iter().position(|v| !v.is_nan())
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_combos(sweep);
        if combos.is_empty() {
            return Err(CudaLinearregAngleError::InvalidInput("no parameter combinations".into()));
        }
        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut sum_x = Vec::with_capacity(combos.len());
        let mut inv_div = Vec::with_capacity(combos.len());
        for c in &combos { let p = c.period; if p < 2 || p > len {
                return Err(CudaLinearregAngleError::InvalidInput(format!("invalid period {} for len {}", p, len)));
            }
            if len - first_valid < p {
                return Err(CudaLinearregAngleError::InvalidInput(format!(
                    "not enough valid data for period {} (tail after first {} is {})", p, first_valid, len - first_valid
                )));
            }
            let pf = p as f64; let sx = (p * (p - 1)) as f64 * 0.5; let sx2 = (p * (p - 1) * (2 * p - 1)) as f64 / 6.0;
            let denom = sx * sx - pf * sx2; let invd = 1.0 / denom;
            periods_i32.push(p as i32); sum_x.push(sx as f32); inv_div.push(invd as f32);
        }
        let (ps, pk, pn) = Self::build_prefixes_lra(data_f32);
        Ok((combos, first_valid, len, periods_i32, sum_x, inv_div, ps, pk, pn))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_ps: &DeviceBuffer<f64>,
        d_pk: &DeviceBuffer<f64>,
        d_pn: &DeviceBuffer<i32>,
        d_periods: &DeviceBuffer<i32>,
        d_sumx: &DeviceBuffer<f32>,
        d_invd: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaLinearregAngleError> {
        let func = self.module
            .get_function("linearreg_angle_batch_f32")
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
        };
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe { (*(self as *const _ as *mut CudaLinearregAngle)).last_batch = Some(BatchKernelSelected::Plain { block_x }); }

        const MAX_GRID_Y: usize = 65_535;
        let mut start = 0usize;
        while start < combos {
            let chunk = (combos - start).min(MAX_GRID_Y);
            let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
            unsafe {
                let mut p_prices = d_prices.as_device_ptr().as_raw();
                let mut p_ps = d_ps.as_device_ptr().as_raw();
                let mut p_pk = d_pk.as_device_ptr().as_raw();
                let mut p_pn = d_pn.as_device_ptr().as_raw();
                let mut len_i = len as i32; let mut first_i = first_valid as i32;
                let mut p_per = d_periods.as_device_ptr().as_raw()
                    .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                let mut p_sx  = d_sumx.as_device_ptr().as_raw()
                    .wrapping_add((start * std::mem::size_of::<f32>()) as u64);
                let mut p_id  = d_invd.as_device_ptr().as_raw()
                    .wrapping_add((start * std::mem::size_of::<f32>()) as u64);
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let mut n_cmb = chunk as i32;
                let args: &mut [*mut c_void] = &mut [
                    &mut p_prices as *mut _ as *mut c_void,
                    &mut p_ps as *mut _ as *mut c_void,
                    &mut p_pk as *mut _ as *mut c_void,
                    &mut p_pn as *mut _ as *mut c_void,
                    &mut len_i as *mut _ as *mut c_void,
                    &mut first_i as *mut _ as *mut c_void,
                    &mut p_per as *mut _ as *mut c_void,
                    &mut p_sx as *mut _ as *mut c_void,
                    &mut p_id as *mut _ as *mut c_void,
                    &mut n_cmb as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)
                    .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
            }
            start += chunk;
        }
        Ok(())
    }

    pub fn linearreg_angle_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &Linearreg_angleBatchRange,
    ) -> Result<DeviceArrayF32, CudaLinearregAngleError> {
        let (combos, first_valid, len, periods_i32, sum_x, inv_div, ps, pk, pn) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let rows = combos.len();

        // VRAM estimate: prices + (ps, pk, pn) + params + output
        let req = len * std::mem::size_of::<f32>()
            + (len + 1) * (std::mem::size_of::<f64>() * 2 + std::mem::size_of::<i32>())
            + rows * (std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f32>())
            + rows * len * std::mem::size_of::<f32>();
        if !Self::will_fit(req, 64 * 1024 * 1024) {
            return Err(CudaLinearregAngleError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (req as f64) / (1024.0 * 1024.0)
            )));
        }

        // H2D copies
        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        let d_ps = DeviceBuffer::from_slice(&ps)
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        let d_pk = DeviceBuffer::from_slice(&pk)
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        let d_pn = DeviceBuffer::from_slice(&pn)
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        let d_sumx = DeviceBuffer::from_slice(&sum_x)
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        let d_invd = DeviceBuffer::from_slice(&inv_div)
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(rows * len) }
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices, &d_ps, &d_pk, &d_pn, &d_periods, &d_sumx, &d_invd, len, first_valid, rows, &mut d_out,
        )?;

        self.stream.synchronize().map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols: len })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32], cols: usize, rows: usize, params: &Linearreg_angleParams,
    ) -> Result<(Vec<i32>, usize, f32, f32), CudaLinearregAngleError> {
        if cols == 0 || rows == 0 { return Err(CudaLinearregAngleError::InvalidInput("empty matrix".into())); }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaLinearregAngleError::InvalidInput(format!(
                "length mismatch: {} != {}*{}", data_tm_f32.len(), cols, rows
            )));
        }
        let period = params.period.unwrap_or(14);
        if period < 2 || period > rows { return Err(CudaLinearregAngleError::InvalidInput("invalid period".into())); }
        // first_valid per column
        let mut first = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for r in 0..rows { let v = data_tm_f32[r * cols + s]; if !v.is_nan() { fv = r as i32; break; } }
            first[s] = fv;
            if fv >= 0 { let tail = rows - fv as usize; if tail < period {
                return Err(CudaLinearregAngleError::InvalidInput(format!(
                    "not enough valid data in series {} (tail {}) for period {}", s, tail, period
                )));
            }}
        }
        let p = period; let sx = (p * (p - 1)) as f64 * 0.5; let sx2 = (p * (p - 1) * (2 * p - 1)) as f64 / 6.0;
        let denom = sx * sx - (p as f64) * sx2; let invd = 1.0 / denom;
        Ok((first, period, sx as f32, invd as f32))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        sum_x: f32,
        inv_div: f32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaLinearregAngleError> {
        let func = self.module
            .get_function("linearreg_angle_many_series_one_param_f32")
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
        };
        let grid: GridSize = (((cols as u32) + block_x - 1) / block_x, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe { (*(self as *const _ as *mut CudaLinearregAngle)).last_many = Some(ManySeriesKernelSelected::OneD { block_x }); }

        unsafe {
            let mut p_prices = d_prices_tm.as_device_ptr().as_raw();
            let mut p_first  = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = cols as i32; let mut rows_i = rows as i32; let mut period_i = period as i32;
            let mut sx_f = sum_x; let mut invd_f = inv_div; let mut p_out = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_prices as *mut _ as *mut c_void,
                &mut p_first as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut sx_f as *mut _ as *mut c_void,
                &mut invd_f as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)
                .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn linearreg_angle_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &Linearreg_angleParams,
    ) -> Result<DeviceArrayF32, CudaLinearregAngleError> {
        let (first_valids, period, sum_x, inv_div) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let elems = cols * rows;
        let req = elems * std::mem::size_of::<f32>() * 2 + cols * std::mem::size_of::<i32>();
        if !Self::will_fit(req, 64 * 1024 * 1024) {
            return Err(CudaLinearregAngleError::InvalidInput("insufficient VRAM".into()));
        }
        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(&d_prices, &d_first, cols, rows, period, sum_x, inv_div, &mut d_out)?;
        self.stream.synchronize().map_err(|e| CudaLinearregAngleError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }
}

// ---------------- Benches ----------------
pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        linearreg_angle_benches,
        CudaLinearregAngle,
        crate::indicators::linearreg_angle::Linearreg_angleBatchRange,
        crate::indicators::linearreg_angle::Linearreg_angleParams,
        linearreg_angle_batch_dev,
        linearreg_angle_many_series_one_param_time_major_dev,
        crate::indicators::linearreg_angle::Linearreg_angleBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::linearreg_angle::Linearreg_angleParams { period: Some(32) },
        "linearreg_angle",
        "linearreg_angle"
    );
    pub use linearreg_angle_benches::bench_profiles;
}
