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
use cust::device::{Device, DeviceAttribute};
use cust::error::CudaError;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

// Module-level alias matching CUDA float2 (8-byte aligned)
type Float2 = [f32; 2];

#[inline] fn f2(x: f32, y: f32) -> Float2 { [x, y] }
// Compensated primitives (Dekker/Kahan style)
#[inline] fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}
#[inline] fn df_add_f(mut acc: Float2, x: f32) -> Float2 {
    let (s, mut e) = two_sum(acc[0], x);
    e += acc[1];
    let (s2, e2) = two_sum(s, e);
    acc[0] = s2; acc[1] = e2; acc
}
#[inline] fn df_add_prod(acc: Float2, a: f32, b: f32) -> Float2 {
    let p   = a * b;
    let err = a.mul_add(b, -p);
    df_add_f(df_add_f(acc, p), err)
}

#[derive(Clone, Debug)]
struct Combo {
    period: usize,
}

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
pub enum CudaLinearregAngleError {
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
    #[error("device mismatch: buf={buf}, current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("not implemented")]
    NotImplemented,
}

pub struct CudaLinearregAngle {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaLinearregAnglePolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    sm_count: u32,
}

impl CudaLinearregAngle {
    pub fn new(device_id: usize) -> Result<Self, CudaLinearregAngleError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/linearreg_angle_kernel.ptx"));
        let module = Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        );
        let module = match Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        ) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) =
                    Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[])?
                }
            }
        };

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let sm_count = device
            .get_attribute(DeviceAttribute::MultiprocessorCount)
            .unwrap_or(64) as u32;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaLinearregAnglePolicy::default(),
            last_batch: None,
            last_many: None,
            sm_count,
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

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    #[inline]
    pub fn device_id(&self) -> u32 { self.device_id }
    #[inline]
    pub fn set_policy(&mut self, policy: CudaLinearregAnglePolicy) { self.policy = policy; }
    #[inline]
    pub fn policy(&self) -> &CudaLinearregAnglePolicy { &self.policy }
    #[inline]
    pub fn selected_batch_kernel(&self) -> Option<BatchKernelSelected> { self.last_batch }
    #[inline]
    pub fn selected_many_series_kernel(&self) -> Option<ManySeriesKernelSelected> { self.last_many }
    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaLinearregAngleError> {
        self.stream.synchronize()?;
        Ok(())
    }

    // -------- VRAM checks --------
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
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaLinearregAngleError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        if let Some((free, _)) = Self::device_mem_info() {
            if required_bytes.saturating_add(headroom_bytes) <= free {
                Ok(())
            } else {
                Err(CudaLinearregAngleError::OutOfMemory {
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
    fn validate_launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), CudaLinearregAngleError> {
        let dev = Device::get_device(self.device_id)?;
        let max_bx = dev
            .get_attribute(DeviceAttribute::MaxBlockDimX)?
            as u32;
        let max_gx = dev
            .get_attribute(DeviceAttribute::MaxGridDimX)?
            as u32;
        if block.0 == 0
            || block.0 > max_bx
            || grid.0 == 0
            || grid.0 > max_gx
        {
            return Err(CudaLinearregAngleError::LaunchConfigTooLarge {
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

    // -------- Inputs prep --------

    fn expand_combos(range: &Linearreg_angleBatchRange) -> Result<Vec<Combo>, CudaLinearregAngleError> {
        fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
            if step == 0 || start == end {
                return vec![start];
            }
            let mut vals = Vec::new();
            if start < end {
                let mut x = start;
                while x <= end {
                    vals.push(x);
                    let next = x.saturating_add(step);
                    if next == x {
                        break;
                    }
                    x = next;
                }
            } else {
                let mut x = start;
                loop {
                    vals.push(x);
                    if x <= end {
                        break;
                    }
                    let next = x.saturating_sub(step);
                    if next >= x {
                        break;
                    }
                    x = next;
                }
            }
            vals
        }
        let vals = axis_usize(range.period);
        if vals.is_empty() {
            return Err(CudaLinearregAngleError::InvalidInput(format!(
                "invalid range: start={}, end={}, step={}",
                range.period.0, range.period.1, range.period.2
            )));
        }
        Ok(vals.into_iter().map(|p| Combo { period: p }).collect())
    }

    fn build_prefixes_lra_f2(data: &[f32]) -> (Vec<Float2>, Vec<Float2>, Vec<i32>) {
        let n = data.len();
        let mut ps = vec![f2(0.0, 0.0); n + 1]; // Σy
        let mut pk = vec![f2(0.0, 0.0); n + 1]; // Σ(k_abs * y)
        let mut pn = vec![0i32; n + 1];               // count NaN

        let mut s  = f2(0.0, 0.0);
        let mut kd = f2(0.0, 0.0);
        let mut cn = 0i32;

        for i in 0..n {
            let v = data[i];
            if v.is_nan() {
                cn += 1; // exclude NaN by not changing sums
            } else {
                s  = df_add_f(s, v);
                kd = df_add_prod(kd, i as f32, v);
            }
            ps[i + 1] = s; pk[i + 1] = kd; pn[i + 1] = cn;
        }
        (ps, pk, pn)
    }

    #[allow(clippy::type_complexity)]
    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &Linearreg_angleBatchRange,
    ) -> Result<(
        Vec<Combo>, usize, usize, Vec<i32>, Vec<f32>, Vec<f32>, Vec<Float2>, Vec<Float2>, Vec<i32>
    ), CudaLinearregAngleError> {
        if data_f32.is_empty() {
            return Err(CudaLinearregAngleError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("all values are NaN".into()))?;
        let combos = Self::expand_combos(sweep)?;
        if combos.is_empty() {
            return Err(CudaLinearregAngleError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let max_p = combos.iter().map(|c| c.period).max().unwrap();
        let _ = combos
            .len()
            .checked_mul(max_p)
            .ok_or_else(|| {
                CudaLinearregAngleError::InvalidInput("n_combos * max_period overflow".into())
            })?;
        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut sum_x = Vec::with_capacity(combos.len());
        let mut inv_div = Vec::with_capacity(combos.len());
        for c in &combos {
            let p = c.period;
            if p < 2 || p > len {
                return Err(CudaLinearregAngleError::InvalidInput(format!(
                    "invalid period {} for len {}",
                    p, len
                )));
            }
            if len - first_valid < p {
                return Err(CudaLinearregAngleError::InvalidInput(format!(
                    "not enough valid data for period {} (tail after first {} is {})",
                    p, first_valid, len - first_valid
                )));
            }
            // Keep denominator consistent with scalar (reversed-x form): Σx² - p·Σx²
            let pf = p as f64; let sx = (p * (p - 1)) as f64 * 0.5; let sx2 = (p * (p - 1) * (2 * p - 1)) as f64 / 6.0;
            let denom = sx * sx - pf * sx2; let invd = 1.0 / denom;
            periods_i32.push(p as i32); sum_x.push(sx as f32); inv_div.push(invd as f32);
        }
        let (ps2, pk2, pn) = Self::build_prefixes_lra_f2(data_f32);
        Ok((combos, first_valid, len, periods_i32, sum_x, inv_div, ps2, pk2, pn))
    }

    fn launch_batch_kernel(
        &self,
        d_ps2: &DeviceBuffer<Float2>,
        d_pk2: &DeviceBuffer<Float2>,
        d_pn: &DeviceBuffer<i32>,
        d_periods: &DeviceBuffer<i32>,
        d_sumx: &DeviceBuffer<f32>,
        d_invd: &DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaLinearregAngleError> {
        let func = &self
            .module
            .get_function("linearreg_angle_batch_f32")
            .map_err(|_| CudaLinearregAngleError::MissingKernelSymbol {
                name: "linearreg_angle_batch_f32",
            })?;

        let block_x: u32 = match self.policy.batch {
            BatchKernelPolicy::Auto => 256,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
        };
        // Grid-stride loop: cap blocks.x to ~8×SMs
        let blocks_needed = ((len as u32) + block_x - 1) / block_x;
        let max_blocks_x  = self.sm_count.saturating_mul(8).max(1);
        let grid_x        = blocks_needed.min(max_blocks_x).max(1);
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            (*(self as *const _ as *mut CudaLinearregAngle)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaLinearregAngle)).last_batch =
                Some(BatchKernelSelected::Plain { block_x });
        }

        const MAX_GRID_Y: usize = 65_535;
        let mut start = 0usize;
        while start < combos {
            let chunk = (combos - start).min(MAX_GRID_Y);
            let gy = chunk as u32;
            let grid_tuple = (grid_x.max(1), gy, 1);
            self.validate_launch(grid_tuple, (block_x, 1, 1))?;
            let grid: GridSize = grid_tuple.into();
            unsafe {
                // prices pointer unused in kernel; pass NULL
                let mut p_prices: u64 = 0;
                let mut p_ps = d_ps2.as_device_ptr().as_raw();
                let mut p_pk = d_pk2.as_device_ptr().as_raw();
                let mut p_pn = d_pn.as_device_ptr().as_raw();
                let mut len_i: i32 = len
                    .try_into()
                    .map_err(|_| CudaLinearregAngleError::InvalidInput("length exceeds i32".into()))?;
                let mut first_i: i32 = first_valid
                    .try_into()
                    .map_err(|_| CudaLinearregAngleError::InvalidInput("first_valid exceeds i32".into()))?;
                let mut p_per = d_periods
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * std::mem::size_of::<i32>()) as u64);
                let mut p_sx = d_sumx
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * std::mem::size_of::<f32>()) as u64);
                let mut p_id = d_invd
                    .as_device_ptr()
                    .as_raw()
                    .wrapping_add((start * std::mem::size_of::<f32>()) as u64);
                let mut out_ptr = d_out.as_device_ptr().as_raw();
                let mut n_cmb: i32 = chunk
                    .try_into()
                    .map_err(|_| CudaLinearregAngleError::InvalidInput("combo count exceeds i32".into()))?;
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
                self.stream.launch(&func, grid, block, 0, args)?;
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
        let (combos, first_valid, len, periods_i32, sum_x, inv_div, ps2, pk2, pn) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let rows = combos.len();

        // VRAM estimate: (ps2, pk2, pn) + params + output (no price upload needed)
        let len_p1 = len
            .checked_add(1)
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("len+1 overflow".into()))?;
        let prefix_stride = std::mem::size_of::<Float2>() * 2 + std::mem::size_of::<i32>();
        let prefix_bytes = len_p1
            .checked_mul(prefix_stride)
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("prefix bytes overflow".into()))?;
        let params_stride = std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f32>();
        let params_bytes = rows
            .checked_mul(params_stride)
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("params bytes overflow".into()))?;
        let out_elems = rows
            .checked_mul(len)
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("rows*len overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("output bytes overflow".into()))?;
        let req = prefix_bytes
            .checked_add(params_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("VRAM size overflow".into()))?;
        Self::will_fit(req, 64 * 1024 * 1024)?;

        // H2D copies (no price upload for batch kernel)
        let d_ps2 = DeviceBuffer::from_slice(&ps2)?;
        let d_pk2 = DeviceBuffer::from_slice(&pk2)?;
        let d_pn = DeviceBuffer::from_slice(&pn)?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let d_sumx = DeviceBuffer::from_slice(&sum_x)?;
        let d_invd = DeviceBuffer::from_slice(&inv_div)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(rows * len) }?;

        self.launch_batch_kernel(
            &d_ps2, &d_pk2, &d_pn, &d_periods, &d_sumx, &d_invd, len, first_valid, rows, &mut d_out,
        )?;

        self.stream.synchronize()?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols: len })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &Linearreg_angleParams,
    ) -> Result<(Vec<i32>, usize, f32, f32), CudaLinearregAngleError> {
        if cols == 0 || rows == 0 {
            return Err(CudaLinearregAngleError::InvalidInput("empty matrix".into()));
        }
        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("cols*rows overflow".into()))?;
        if data_tm_f32.len() != elems {
            return Err(CudaLinearregAngleError::InvalidInput(format!(
                "length mismatch: {} != {}*{}",
                data_tm_f32.len(),
                cols,
                rows
            )));
        }
        let period = params.period.unwrap_or(14);
        if period < 2 || period > rows {
            return Err(CudaLinearregAngleError::InvalidInput(
                "invalid period".into(),
            ));
        }
        // first_valid per column
        let mut first = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = -1i32;
            for r in 0..rows {
                let v = data_tm_f32[r * cols + s];
                if !v.is_nan() {
                    fv = r as i32;
                    break;
                }
            }
            first[s] = fv;
            if fv >= 0 {
                let tail = rows - fv as usize;
                if tail < period {
                    return Err(CudaLinearregAngleError::InvalidInput(format!(
                        "not enough valid data in series {} (tail {}) for period {}",
                        s, tail, period
                    )));
                }
            }
        }
        let p = period;
        let sx = (p * (p - 1)) as f64 * 0.5;
        let sx2 = (p * (p - 1) * (2 * p - 1)) as f64 / 6.0;
        let denom = sx * sx - (p as f64) * sx2;
        let invd = 1.0 / denom;
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
        let func = self
            .module
            .get_function("linearreg_angle_many_series_one_param_f32")
            .map_err(|_| CudaLinearregAngleError::MissingKernelSymbol {
                name: "linearreg_angle_many_series_one_param_f32",
            })?;
        let block_x: u32 = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
        };
        let blocks_needed = ((cols as u32) + block_x - 1) / block_x;
        let max_blocks_x  = self.sm_count.saturating_mul(8).max(1);
        let gx = blocks_needed.min(max_blocks_x).max(1);
        self.validate_launch((gx, 1, 1), (block_x, 1, 1))?;
        let grid: GridSize = (gx, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            (*(self as *const _ as *mut CudaLinearregAngle)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }
        unsafe {
            (*(self as *const _ as *mut CudaLinearregAngle)).last_many =
                Some(ManySeriesKernelSelected::OneD { block_x });
        }

        unsafe {
            let mut p_prices = d_prices_tm.as_device_ptr().as_raw();
            let mut p_first = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i: i32 = cols
                .try_into()
                .map_err(|_| CudaLinearregAngleError::InvalidInput("cols exceeds i32".into()))?;
            let mut rows_i: i32 = rows
                .try_into()
                .map_err(|_| CudaLinearregAngleError::InvalidInput("rows exceeds i32".into()))?;
            let mut period_i: i32 = period
                .try_into()
                .map_err(|_| CudaLinearregAngleError::InvalidInput("period exceeds i32".into()))?;
            let mut sx_f = sum_x;
            let mut invd_f = inv_div;
            let mut p_out = d_out_tm.as_device_ptr().as_raw();
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
            self.stream.launch(&func, grid, block, 0, args)?;
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

        let elems = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("cols*rows overflow".into()))?;
        let prices_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("prices bytes overflow".into()))?;
        let first_bytes = cols
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("first_valids bytes overflow".into()))?;
        let out_bytes = elems
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("output bytes overflow".into()))?;
        let req = prices_bytes
            .checked_add(first_bytes)
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaLinearregAngleError::InvalidInput("VRAM size overflow".into()))?;
        Self::will_fit(req, 64 * 1024 * 1024)?;

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_first = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }?;

        self.launch_many_series_kernel(
            &d_prices, &d_first, cols, rows, period, sum_x, inv_div, &mut d_out,
        )?;
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 { buf: d_out, rows, cols })
    }
}

// ---------------- Benches ----------------
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use crate::indicators::linearreg_angle::{Linearreg_angleBatchRange, Linearreg_angleParams};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250;
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_LEN: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let len_p1 = ONE_SERIES_LEN + 1;
        let prefix_stride = std::mem::size_of::<Float2>() * 2 + std::mem::size_of::<i32>();
        let prefix_bytes = len_p1 * prefix_stride;
        let params_stride = std::mem::size_of::<i32>() + 2 * std::mem::size_of::<f32>();
        let params_bytes = PARAM_SWEEP * params_stride;
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        prefix_bytes + params_bytes + out_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_LEN;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let first_bytes = MANY_SERIES_COLS * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        in_bytes + first_bytes + out_bytes + 64 * 1024 * 1024
    }

    struct BatchDevState {
        cuda: CudaLinearregAngle,
        d_ps2: DeviceBuffer<Float2>,
        d_pk2: DeviceBuffer<Float2>,
        d_pn: DeviceBuffer<i32>,
        d_periods: DeviceBuffer<i32>,
        d_sumx: DeviceBuffer<f32>,
        d_invd: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        rows: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for BatchDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_ps2,
                    &self.d_pk2,
                    &self.d_pn,
                    &self.d_periods,
                    &self.d_sumx,
                    &self.d_invd,
                    self.len,
                    self.first_valid,
                    self.rows,
                    &mut self.d_out,
                )
                .expect("linearreg_angle batch kernel");
            self.cuda.stream.synchronize().expect("linearreg_angle sync");
        }
    }

    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaLinearregAngle::new(0).expect("cuda linearreg_angle");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = Linearreg_angleBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
        };
        let (_combos, first_valid, len, periods_i32, sum_x, inv_div, ps2, pk2, pn) =
            CudaLinearregAngle::prepare_batch_inputs(&price, &sweep).expect("prepare_batch_inputs");
        let rows = periods_i32.len();

        let d_ps2 = DeviceBuffer::from_slice(&ps2).expect("d_ps2");
        let d_pk2 = DeviceBuffer::from_slice(&pk2).expect("d_pk2");
        let d_pn = DeviceBuffer::from_slice(&pn).expect("d_pn");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_sumx = DeviceBuffer::from_slice(&sum_x).expect("d_sumx");
        let d_invd = DeviceBuffer::from_slice(&inv_div).expect("d_invd");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(rows * len) }.expect("d_out");
        cuda.stream.synchronize().expect("sync after prep");

        Box::new(BatchDevState {
            cuda,
            d_ps2,
            d_pk2,
            d_pn,
            d_periods,
            d_sumx,
            d_invd,
            len,
            first_valid,
            rows,
            d_out,
        })
    }

    struct ManySeriesDevState {
        cuda: CudaLinearregAngle,
        d_prices_tm: DeviceBuffer<f32>,
        d_first_valids: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        sum_x: f32,
        inv_div: f32,
        d_out_tm: DeviceBuffer<f32>,
    }
    impl CudaBenchState for ManySeriesDevState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_prices_tm,
                    &self.d_first_valids,
                    self.cols,
                    self.rows,
                    self.period,
                    self.sum_x,
                    self.inv_div,
                    &mut self.d_out_tm,
                )
                .expect("linearreg_angle many-series kernel");
            self.cuda.stream.synchronize().expect("linearreg_angle sync");
        }
    }

    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaLinearregAngle::new(0).expect("cuda linearreg_angle");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_LEN;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = Linearreg_angleParams { period: Some(32) };
        let (first_valids, period, sum_x, inv_div) =
            CudaLinearregAngle::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("prepare_many_series_inputs");
        let d_prices_tm = DeviceBuffer::from_slice(&data_tm).expect("d_prices_tm");
        let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
        let d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("d_out_tm");
        cuda.stream.synchronize().expect("sync after prep");
        Box::new(ManySeriesDevState {
            cuda,
            d_prices_tm,
            d_first_valids,
            cols,
            rows,
            period,
            sum_x,
            inv_div,
            d_out_tm,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "linearreg_angle",
                "one_series_many_params",
                "linearreg_angle_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "linearreg_angle",
                "many_series_one_param",
                "linearreg_angle_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
