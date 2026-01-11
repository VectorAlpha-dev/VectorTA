//! CUDA support for Chande Forecast Oscillator (CFO).
//!
//! Math pattern classification: prefix-sum/rational.
//! - Batch (one series × many params): host builds prefix sums P (sum_y) and Q (weighted sum)
//!   over the valid segment [first_valid..), and the kernel computes O(1) window outputs.
//! - Many-series × one-param (time-major): host builds time-major P/Q per series with respect to
//!   the per-series first_valid; the kernel mirrors the same window logic.
//!
//! Semantics match the scalar CFO implementation:
//! - Warmup per row/series: warm = first_valid + period - 1
//! - Warmup prefix is filled with NaN
//! - If current value is NaN or 0.0, output is NaN
//! - Critical accumulations use f64; outputs are f32

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::cfo::{CfoBatchRange, CfoParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaCfoError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
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
    #[error("invalid range (usize): start={start} end={end} step={step}")]
    InvalidRangeUsize { start: usize, end: usize, step: usize },
    #[error("invalid range (f64): start={start} end={end} step={step}")]
    InvalidRangeF64 { start: f64, end: f64, step: f64 },
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct CudaCfoPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaCfoPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaCfo {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaCfoPolicy,
    last_batch_block: Option<u32>,
    last_many_block: Option<u32>,
}

impl CudaCfo {
    fn will_fit(required_bytes: usize, headroom: usize) -> Result<(), CudaCfoError> {
        if let Ok((free, _)) = mem_get_info() {
            if required_bytes > free.saturating_sub(headroom) {
                return Err(CudaCfoError::OutOfMemory { required: required_bytes, free, headroom });
            }
        }
        Ok(())
    }
    pub fn new(device_id: usize) -> Result<Self, CudaCfoError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cfo_kernel.ptx"));
        let module = match Module::from_ptx(
            ptx,
            &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O2),
            ],
        ) {
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
            context,
            device_id: device_id as u32,
            policy: CudaCfoPolicy::default(),
            last_batch_block: None,
            last_many_block: None,
        })
    }

    pub fn context_arc(&self) -> Arc<Context> { self.context.clone() }
    pub fn device_id(&self) -> u32 { self.device_id }

    pub fn set_policy(&mut self, policy: CudaCfoPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaCfoPolicy {
        &self.policy
    }

    // ---------- One-series × many-params ----------

    pub fn cfo_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &CfoBatchRange,
    ) -> Result<DeviceArrayF32, CudaCfoError> {
        let (periods, scalars, first_valid) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let len = data_f32.len();
        let n_combos = periods.len();

        // Build f64 prefixes over [first_valid..)
        let (ps, pw) = build_prefixes_from_first(data_f32, first_valid);

        // VRAM: data + prefixes (2*f64 arrays) + params + out + headroom
        let bytes_data = len.checked_mul(std::mem::size_of::<f32>()).ok_or(CudaCfoError::InvalidInput("size overflow".into()))?;
        let bytes_prefix = (len + 1).checked_mul(std::mem::size_of::<f64>()).and_then(|b| b.checked_mul(2)).ok_or(CudaCfoError::InvalidInput("size overflow".into()))?;
        let bytes_params = n_combos.checked_mul(std::mem::size_of::<i32>() + std::mem::size_of::<f32>()).ok_or(CudaCfoError::InvalidInput("size overflow".into()))?;
        let bytes_out = len.checked_mul(n_combos).and_then(|e| e.checked_mul(std::mem::size_of::<f32>())).ok_or(CudaCfoError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024usize;
        let required = bytes_data.checked_add(bytes_prefix).and_then(|v| v.checked_add(bytes_params)).and_then(|v| v.checked_add(bytes_out)).and_then(|v| v.checked_add(headroom)).ok_or(CudaCfoError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, headroom)?;

        let d_data = DeviceBuffer::from_slice(data_f32)?;
        let d_ps: DeviceBuffer<f64> = DeviceBuffer::from_slice(&ps)?;
        let d_pw: DeviceBuffer<f64> = DeviceBuffer::from_slice(&pw)?;
        let d_periods = DeviceBuffer::from_slice(&periods)?;
        let d_scalars = DeviceBuffer::from_slice(&scalars)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(len * n_combos)?
        };

        self.launch_batch_kernel(
            &d_data,
            &d_ps,
            &d_pw,
            len as i32,
            first_valid as i32,
            &d_periods,
            &d_scalars,
            n_combos as i32,
            &mut d_out,
        )?;

        // synchronize producing stream to simplify CAI interop
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_ps: &DeviceBuffer<f64>,
        d_pw: &DeviceBuffer<f64>,
        len: i32,
        first_valid: i32,
        d_periods: &DeviceBuffer<i32>,
        d_scalars: &DeviceBuffer<f32>,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCfoError> {
        if len <= 0 || n_combos <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("cfo_batch_f32")
            .map_err(|_| CudaCfoError::MissingKernelSymbol { name: "cfo_batch_f32" })?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        // Launch in grid.y chunks to respect 65_535 limit
        for (start, count) in grid_y_chunks(n_combos as usize) {
            let grid: GridSize = (grid_x.max(1), count as u32, 1).into();
            let block: BlockSize = (block_x, 1, 1).into();
            // Optional validation against device limits
            let dev = Device::get_device(self.device_id)?;
            let max_gx = dev.get_attribute(cust::device::DeviceAttribute::MaxGridDimX)? as u32;
            let max_gy = dev.get_attribute(cust::device::DeviceAttribute::MaxGridDimY)? as u32;
            let max_bx = dev.get_attribute(cust::device::DeviceAttribute::MaxBlockDimX)? as u32;
            if grid_x > max_gx || (count as u32) > max_gy || block_x > max_bx {
                return Err(CudaCfoError::LaunchConfigTooLarge { gx: grid_x, gy: count as u32, gz: 1, bx: block_x, by: 1, bz: 1 });
            }
            unsafe {
                let mut p_data = d_data.as_device_ptr().as_raw();
                let mut p_ps = d_ps.as_device_ptr().as_raw();
                let mut p_pw = d_pw.as_device_ptr().as_raw();
                let mut p_len = len;
                let mut p_first = first_valid;
                let mut p_periods = d_periods.as_device_ptr().add(start).as_raw();
                let mut p_scalars = d_scalars.as_device_ptr().add(start).as_raw();
                let mut p_n = count as i32;
                let mut p_out = d_out.as_device_ptr().add(start * (len as usize)).as_raw();
                let args: &mut [*mut c_void] = &mut [
                    &mut p_data as *mut _ as *mut c_void,
                    &mut p_ps as *mut _ as *mut c_void,
                    &mut p_pw as *mut _ as *mut c_void,
                    &mut p_len as *mut _ as *mut c_void,
                    &mut p_first as *mut _ as *mut c_void,
                    &mut p_periods as *mut _ as *mut c_void,
                    &mut p_scalars as *mut _ as *mut c_void,
                    &mut p_n as *mut _ as *mut c_void,
                    &mut p_out as *mut _ as *mut c_void,
                ];
                self.stream.launch(&func, grid, block, 0, args)?;
            }
        }
        Ok(())
    }

    // ---------- Many-series × one-param (time-major) ----------

    pub fn cfo_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CfoParams,
    ) -> Result<DeviceArrayF32, CudaCfoError> {
        let (first_valids, period, scalar) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // Build time-major f64 prefixes P/Q per series
        let (ps_tm, pw_tm) = build_prefixes_time_major(data_tm_f32, cols, rows, &first_valids);

        // VRAM estimate
        let elems = cols.checked_mul(rows).ok_or(CudaCfoError::InvalidInput("size overflow".into()))?;
        let bytes_data = elems.checked_mul(std::mem::size_of::<f32>()).ok_or(CudaCfoError::InvalidInput("size overflow".into()))?;
        let bytes_prefix = (elems + 1).checked_mul(std::mem::size_of::<f64>()).and_then(|b| b.checked_mul(2)).ok_or(CudaCfoError::InvalidInput("size overflow".into()))?;
        let bytes_fv = cols.checked_mul(std::mem::size_of::<i32>()).ok_or(CudaCfoError::InvalidInput("size overflow".into()))?;
        let bytes_out = elems.checked_mul(std::mem::size_of::<f32>()).ok_or(CudaCfoError::InvalidInput("size overflow".into()))?;
        let headroom = 64 * 1024 * 1024usize;
        let required = bytes_data.checked_add(bytes_prefix).and_then(|v| v.checked_add(bytes_fv)).and_then(|v| v.checked_add(bytes_out)).and_then(|v| v.checked_add(headroom)).ok_or(CudaCfoError::InvalidInput("size overflow".into()))?;
        Self::will_fit(required, headroom)?;

        let d_data = DeviceBuffer::from_slice(data_tm_f32)?;
        let d_ps: DeviceBuffer<f64> = DeviceBuffer::from_slice(&ps_tm)?;
        let d_pw: DeviceBuffer<f64> = DeviceBuffer::from_slice(&pw_tm)?;
        let d_fv = DeviceBuffer::from_slice(&first_valids)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(elems)?
        };

        self.launch_many_series_kernel(
            &d_data,
            &d_ps,
            &d_pw,
            &d_fv,
            cols as i32,
            rows as i32,
            period as i32,
            scalar as f32,
            &mut d_out,
        )?;

        self.stream.synchronize()?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_data: &DeviceBuffer<f32>,
        d_ps: &DeviceBuffer<f64>,
        d_pw: &DeviceBuffer<f64>,
        d_fv: &DeviceBuffer<i32>,
        cols: i32,
        rows: i32,
        period: i32,
        scalar: f32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCfoError> {
        if cols <= 0 || rows <= 0 {
            return Ok(());
        }
        let func = self
            .module
            .get_function("cfo_many_series_one_param_time_major_f32")
            .map_err(|_| CudaCfoError::MissingKernelSymbol { name: "cfo_many_series_one_param_time_major_f32" })?;

        // Original mapping: stride over time (x), grid.y enumerates series
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        let grid_x = ((rows as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut p_data = d_data.as_device_ptr().as_raw();
            let mut p_ps = d_ps.as_device_ptr().as_raw();
            let mut p_pw = d_pw.as_device_ptr().as_raw();
            let mut p_fv = d_fv.as_device_ptr().as_raw();
            let mut p_cols = cols;
            let mut p_rows = rows;
            let mut p_period = period;
            let mut p_scalar = scalar;
            let mut p_out = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_data as *mut _ as *mut c_void,
                &mut p_ps as *mut _ as *mut c_void,
                &mut p_pw as *mut _ as *mut c_void,
                &mut p_fv as *mut _ as *mut c_void,
                &mut p_cols as *mut _ as *mut c_void,
                &mut p_rows as *mut _ as *mut c_void,
                &mut p_period as *mut _ as *mut c_void,
                &mut p_scalar as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            self.stream.launch(&func, grid, block, 0, args)?;
        }
        Ok(())
    }

    // ----- Prep helpers -----

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &CfoBatchRange,
    ) -> Result<(Vec<i32>, Vec<f32>, usize), CudaCfoError> {
        if data_f32.is_empty() {
            return Err(CudaCfoError::InvalidInput("empty data".into()));
        }
        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaCfoError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep)?;

        let mut periods = Vec::with_capacity(combos.len());
        let mut scalars = Vec::with_capacity(combos.len());
        for c in combos {
            let p = c.period.unwrap_or(0);
            if p == 0 || p > len {
                return Err(CudaCfoError::InvalidInput(format!(
                    "invalid period {} for data length {}",
                    p, len
                )));
            }
            if len - first_valid < p {
                return Err(CudaCfoError::InvalidInput(format!(
                    "not enough valid data: needed {}, valid {}",
                    p,
                    len - first_valid
                )));
            }
            periods.push(p as i32);
            scalars.push(c.scalar.unwrap_or(100.0) as f32);
        }
        Ok((periods, scalars, first_valid))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CfoParams,
    ) -> Result<(Vec<i32>, usize, f64), CudaCfoError> {
        if cols == 0 || rows == 0 {
            return Err(CudaCfoError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols.checked_mul(rows).ok_or(CudaCfoError::InvalidInput("size overflow".into()))? {
            return Err(CudaCfoError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }
        let period = params.period.unwrap_or(14);
        if period == 0 || period > rows {
            return Err(CudaCfoError::InvalidInput(format!(
                "invalid period {} for series length {}",
                period, rows
            )));
        }
        let mut first_valids = vec![0i32; cols];
        for s in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaCfoError::InvalidInput(format!("series {} consists entirely of NaNs", s))
            })?;
            if rows - fv < period {
                return Err(CudaCfoError::InvalidInput(format!(
                    "series {} lacks data: needed {}, valid {}",
                    s,
                    period,
                    rows - fv
                )));
            }
            first_valids[s] = fv as i32;
        }
        Ok((first_valids, period, params.scalar.unwrap_or(100.0)))
    }
}

// ---- Prefix builders (f64) ----
fn build_prefixes_from_first(data: &[f32], first_valid: usize) -> (Vec<f64>, Vec<f64>) {
    let len = data.len();
    let mut ps = vec![0.0f64; len + 1];
    let mut pw = vec![0.0f64; len + 1];
    let mut acc_s = 0.0f64;
    let mut acc_w = 0.0f64;
    let mut weight = 0.0f64;
    for i in 0..len {
        if i >= first_valid {
            let v = data[i] as f64;
            weight += 1.0;
            acc_s += v;
            acc_w += v * weight;
        }
        let w = i + 1;
        ps[w] = acc_s;
        pw[w] = acc_w;
    }
    (ps, pw)
}

fn build_prefixes_time_major(
    data_tm: &[f32],
    cols: usize,
    rows: usize,
    first_valids: &[i32],
) -> (Vec<f64>, Vec<f64>) {
    let total = data_tm.len();
    let mut ps = vec![0.0f64; total + 1];
    let mut pw = vec![0.0f64; total + 1];
    for s in 0..cols {
        let fv = first_valids[s].max(0) as usize;
        let mut acc_s = 0.0f64;
        let mut acc_w = 0.0f64;
        let mut weight = 0.0f64;
        for t in 0..rows {
            if t >= fv {
                let v = data_tm[t * cols + s] as f64;
                weight += 1.0;
                acc_s += v;
                acc_w += v * weight;
            }
            let idx = (t * cols + s) + 1;
            ps[idx] = acc_s;
            pw[idx] = acc_w;
        }
    }
    (ps, pw)
}

// ---- Grid expand (mirror indicator) ----

fn expand_grid(r: &CfoBatchRange) -> Result<Vec<CfoParams>, CudaCfoError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, CudaCfoError> {
        if step == 0 || start == end { return Ok(vec![start]); }
        let mut vals = Vec::new();
        if start < end {
            let mut cur = start; while cur <= end { vals.push(cur); if let Some(n) = cur.checked_add(step) { cur = n } else { break } }
        } else {
            let mut cur = start; while cur >= end { vals.push(cur); cur = cur.saturating_sub(step); if vals.last() == Some(&cur) { break; } }
            if let Some(&last) = vals.last() { if last < end { vals.pop(); } }
        }
        if vals.is_empty() { return Err(CudaCfoError::InvalidRangeUsize { start, end, step }); }
        Ok(vals)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, CudaCfoError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return Ok(vec![start]); }
        let mut vals = Vec::new();
        let delta = if start <= end { step.abs() } else { -step.abs() };
        if delta.is_sign_positive() { let mut x = start; while x <= end + 1e-12 { vals.push(x); x += delta; if !x.is_finite() { break; } } }
        else { let mut x = start; while x >= end - 1e-12 { vals.push(x); x += delta; if !x.is_finite() { break; } } }
        if vals.is_empty() { return Err(CudaCfoError::InvalidRangeF64 { start, end, step }); }
        Ok(vals)
    }
    let periods = axis_usize(r.period)?;
    let scalars = axis_f64(r.scalar)?;
    let combos_len = periods.len().checked_mul(scalars.len()).ok_or(CudaCfoError::InvalidInput("size overflow".into()))?;
    let mut out = Vec::with_capacity(combos_len);
    for &p in &periods { for &s in &scalars { out.push(CfoParams { period: Some(p), scalar: Some(s) }); } }
    Ok(out)
}

#[inline(always)]
fn grid_y_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
    struct YChunks {
        n: usize,
        launched: usize,
    }
    impl Iterator for YChunks {
        type Item = (usize, usize);
        fn next(&mut self) -> Option<Self::Item> {
            const MAX: usize = 65_535;
            if self.launched >= self.n {
                return None;
            }
            let start = self.launched;
            let len = (self.n - self.launched).min(MAX);
            self.launched += len;
            Some((start, len))
        }
    }
    YChunks { n, launched: 0 }
}

// ---------- Benches ----------

pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const PARAM_SWEEP: usize = 250; // vary periods only
    const MANY_SERIES_COLS: usize = 250;
    const MANY_SERIES_ROWS: usize = 1_000_000;

    fn bytes_one_series_many_params() -> usize {
        let in_bytes = ONE_SERIES_LEN * std::mem::size_of::<f32>();
        let out_bytes = ONE_SERIES_LEN * PARAM_SWEEP * std::mem::size_of::<f32>();
        // include ~2x f64 prefix
        let prefix_bytes = (ONE_SERIES_LEN + 1) * 2 * std::mem::size_of::<f64>();
        in_bytes + out_bytes + prefix_bytes + 64 * 1024 * 1024
    }
    fn bytes_many_series_one_param() -> usize {
        let elems = MANY_SERIES_COLS * MANY_SERIES_ROWS;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let prefix_bytes = (elems + 1) * 2 * std::mem::size_of::<f64>();
        let fv_bytes = MANY_SERIES_COLS * std::mem::size_of::<i32>();
        in_bytes + out_bytes + prefix_bytes + fv_bytes + 64 * 1024 * 1024
    }

    struct CfoBatchDeviceState {
        cuda: CudaCfo,
        d_data: DeviceBuffer<f32>,
        d_ps: DeviceBuffer<f64>,
        d_pw: DeviceBuffer<f64>,
        d_periods: DeviceBuffer<i32>,
        d_scalars: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for CfoBatchDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_batch_kernel(
                    &self.d_data,
                    &self.d_ps,
                    &self.d_pw,
                    self.len as i32,
                    self.first_valid as i32,
                    &self.d_periods,
                    &self.d_scalars,
                    self.n_combos as i32,
                    &mut self.d_out,
                )
                .expect("cfo launch");
            self.cuda.stream.synchronize().expect("cfo sync");
        }
    }
    fn prep_one_series_many_params() -> Box<dyn CudaBenchState> {
        let cuda = CudaCfo::new(0).expect("cuda cfo");
        let price = gen_series(ONE_SERIES_LEN);
        let sweep = CfoBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1),
            scalar: (100.0, 100.0, 0.0),
        };

        let (periods, scalars, first_valid) =
            CudaCfo::prepare_batch_inputs(&price, &sweep).expect("prepare_batch_inputs");
        let len = price.len();
        let n_combos = periods.len();
        let (ps, pw) = build_prefixes_from_first(&price, first_valid);

        let d_data = DeviceBuffer::from_slice(&price).expect("cfo d_data H2D");
        let d_ps: DeviceBuffer<f64> = DeviceBuffer::from_slice(&ps).expect("cfo d_ps H2D");
        let d_pw: DeviceBuffer<f64> = DeviceBuffer::from_slice(&pw).expect("cfo d_pw H2D");
        let d_periods = DeviceBuffer::from_slice(&periods).expect("cfo d_periods H2D");
        let d_scalars = DeviceBuffer::from_slice(&scalars).expect("cfo d_scalars H2D");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(len * n_combos) }.expect("cfo out alloc");

        Box::new(CfoBatchDeviceState {
            cuda,
            d_data,
            d_ps,
            d_pw,
            d_periods,
            d_scalars,
            len,
            first_valid,
            n_combos,
            d_out,
        })
    }

    struct CfoManyDeviceState {
        cuda: CudaCfo,
        d_data: DeviceBuffer<f32>,
        d_ps: DeviceBuffer<f64>,
        d_pw: DeviceBuffer<f64>,
        d_fv: DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        scalar: f32,
        d_out: DeviceBuffer<f32>,
    }
    impl CudaBenchState for CfoManyDeviceState {
        fn launch(&mut self) {
            self.cuda
                .launch_many_series_kernel(
                    &self.d_data,
                    &self.d_ps,
                    &self.d_pw,
                    &self.d_fv,
                    self.cols as i32,
                    self.rows as i32,
                    self.period as i32,
                    self.scalar,
                    &mut self.d_out,
                )
                .expect("cfo many-series launch");
            self.cuda.stream.synchronize().expect("cfo many sync");
        }
    }
    fn prep_many_series_one_param() -> Box<dyn CudaBenchState> {
        let cuda = CudaCfo::new(0).expect("cuda cfo");
        let cols = MANY_SERIES_COLS;
        let rows = MANY_SERIES_ROWS;
        let data_tm = gen_time_major_prices(cols, rows);
        let params = CfoParams {
            period: Some(14),
            scalar: Some(100.0),
        };

        let (first_valids, period, scalar) =
            CudaCfo::prepare_many_series_inputs(&data_tm, cols, rows, &params)
                .expect("prepare_many_series_inputs");
        let (ps_tm, pw_tm) = build_prefixes_time_major(&data_tm, cols, rows, &first_valids);

        let d_data = DeviceBuffer::from_slice(&data_tm).expect("cfo d_data_tm H2D");
        let d_ps: DeviceBuffer<f64> = DeviceBuffer::from_slice(&ps_tm).expect("cfo d_ps_tm H2D");
        let d_pw: DeviceBuffer<f64> = DeviceBuffer::from_slice(&pw_tm).expect("cfo d_pw_tm H2D");
        let d_fv = DeviceBuffer::from_slice(&first_valids).expect("cfo d_fv H2D");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(cols * rows) }.expect("cfo d_out alloc");

        Box::new(CfoManyDeviceState {
            cuda,
            d_data,
            d_ps,
            d_pw,
            d_fv,
            cols,
            rows,
            period,
            scalar: scalar as f32,
            d_out,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "cfo",
                "one_series_many_params",
                "cfo_cuda_batch_dev",
                "1m_x_250",
                prep_one_series_many_params,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_one_series_many_params()),
            CudaBenchScenario::new(
                "cfo",
                "many_series_one_param",
                "cfo_cuda_many_series_one_param",
                "250x1m",
                prep_many_series_one_param,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many_series_one_param()),
        ]
    }
}
