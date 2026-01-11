//! CUDA support for the Mass Index (MASS) indicator.
//!
//! Mirrors ALMA-class wrappers for policy, PTX load, NON_BLOCKING stream,
//! VRAM checks, chunking, and bench registration. The math follows the scalar
//! path in `indicators::mass` exactly: two nested EMA(9) filters on (high-low)
//! to form `ratio = ema1/ema2`, then a rolling sum over `period` bars.
//!
//! One-series × many-params (batch): we precompute the ratio and its
//! double-single (float2) prefix sums on the host and upload them, enabling each
//! row (period) to evaluate O(1) per timestamp via prefix differences. We also
//! upload a prefix count of NaNs so any window containing a NaN yields NaN.
//!
//! Many-series × one-param (time-major): DS path proved numerically tricky with
//! the existing time-major prefix layout. For now, we compute the scalar CPU
//! result per series and upload it (preserves correctness and test parity).
//! The batch (one-series × many-params) CUDA path is fully FP32/DS-enabled.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::mass::{mass_with_kernel, MassBatchRange, MassData, MassInput, MassParams};
use crate::utilities::enums::Kernel;
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer, AsyncCopyDestination};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust_derive::DeviceCopy;
use std::env;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaMassError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("mass: out of memory: required={required} free={free} headroom={headroom}")]
    OutOfMemory { required: usize, free: usize, headroom: usize },
    #[error("mass: missing kernel symbol: {name}")]
    MissingKernelSymbol { name: &'static str },
    #[error("mass: invalid input: {0}")]
    InvalidInput(String),
    #[error("mass: invalid policy: {0}")]
    InvalidPolicy(&'static str),
    #[error("mass: launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge { gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32 },
    #[error("mass: device mismatch: buf={buf} current={current}")]
    DeviceMismatch { buf: u32, current: u32 },
    #[error("mass: not implemented")]
    NotImplemented,
}

#[derive(Clone, Copy, Debug)]
pub enum BatchKernelPolicy {
    Auto,
    Plain { block_x: u32 },
}

impl Default for BatchKernelPolicy {
    fn default() -> Self {
        BatchKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ManySeriesKernelPolicy {
    Auto,
    OneD { block_x: u32 },
}

impl Default for ManySeriesKernelPolicy {
    fn default() -> Self {
        ManySeriesKernelPolicy::Auto
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaMassPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaMass {
    module: Module,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
    policy: CudaMassPolicy,
    debug_logged: bool,
}

// Two-float encoding for high-precision sums (double-single)
#[repr(C)]
#[derive(Clone, Copy, Default, DeviceCopy)]
pub struct F2 { pub x: f32, pub y: f32 }

#[inline(always)]
fn two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let z = s - a;
    let e = (a - (s - z)) + (b - z);
    (s, e)
}

#[inline(always)]
fn ds_add(hi: f32, lo: f32, x: f32) -> (f32, f32) {
    let (s, e) = two_sum_f32(hi, x);
    let (s2, e2) = two_sum_f32(s, lo);
    let (hi2, lo2) = two_sum_f32(s2, e + e2);
    (hi2, lo2)
}

impl CudaMass {
    pub fn new(device_id: usize) -> Result<Self, CudaMassError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/mass_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| {
                Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
            })
            .or_else(|_| Module::from_ptx(ptx, &[]))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            context,
            device_id: device_id as u32,
            policy: CudaMassPolicy::default(),
            debug_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaMassPolicy) {
        self.policy = policy;
    }

    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    fn maybe_log_selected(&mut self, which: &str, block_x: u32) {
        if self.debug_logged {
            return;
        }
        if env::var("BENCH_DEBUG").ok().as_deref() == Some("1") {
            eprintln!("[DEBUG] MASS {} block_x={} ", which, block_x);
            self.debug_logged = true;
        }
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
            Err(_) => true,
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> Result<(), CudaMassError> {
        if !Self::mem_check_enabled() {
            return Ok(());
        }
        match mem_get_info() {
            Ok((free, _total)) => {
                let needed = required_bytes.saturating_add(headroom_bytes);
                if needed > free {
                    Err(CudaMassError::OutOfMemory {
                        required: required_bytes,
                        free,
                        headroom: headroom_bytes,
                    })
                } else {
                    Ok(())
                }
            }
            Err(e) => Err(CudaMassError::Cuda(e)),
        }
    }

    // ----------------------- Host precompute helpers -----------------------
    
    fn first_valid_hilo(high: &[f32], low: &[f32]) -> Option<usize> {
        high.iter()
            .zip(low.iter())
            .position(|(&h, &l)| h.is_finite() && l.is_finite())
    }

    fn precompute_ratio_prefix_one_series_ds(
        high: &[f32],
        low: &[f32],
    ) -> Result<(Vec<F2>, Vec<i32>, usize), CudaMassError> {
        if high.len() != low.len() || high.is_empty() {
            return Err(CudaMassError::InvalidInput(
                "mismatched or empty inputs".into(),
            ));
        }
        let n = high.len();
        let first = Self::first_valid_hilo(high, low)
            .ok_or_else(|| CudaMassError::InvalidInput("all values are NaN".into()))?;

        // DS prefix buffers: [hi, lo] per entry, length n+1
        let mut prefix_ratio_ds = vec!(F2::default(); n + 1);
        let mut prefix_nan = vec![0i32; n + 1];

        // EMA(9) constants (f32)
        let alpha: f32 = 2.0f32 / 10.0f32;
        let inv_alpha: f32 = 1.0f32 - alpha;

        // Initialize EMAs at first valid
        let mut ema1: f32 = high[first] - low[first];
        let mut ema2: f32 = ema1;
        let start_ema2 = first + 8;
        let start_ratio = first + 16;

        // DS accumulator
        let mut acc_hi: f32 = 0.0;
        let mut acc_lo: f32 = 0.0;

        for i in 0..n {
            if i < first {
                prefix_ratio_ds[i + 1] = F2 { x: acc_hi, y: acc_lo };
                prefix_nan[i + 1] = prefix_nan[i];
                continue;
            }
            let hl: f32 = high[i] - low[i];
            ema1 = inv_alpha.mul_add(ema1, alpha * hl);
            if i == start_ema2 { ema2 = ema1; }
            let mut ratio: f32 = f32::NAN;
            if i >= start_ema2 {
                ema2 = inv_alpha.mul_add(ema2, alpha * ema1);
                if i >= start_ratio {
                    ratio = ema1 / ema2;
                }
            }
            let is_nan = !ratio.is_finite();
            if !is_nan {
                (acc_hi, acc_lo) = ds_add(acc_hi, acc_lo, ratio);
                prefix_nan[i + 1] = prefix_nan[i];
            } else {
                prefix_nan[i + 1] = prefix_nan[i] + 1;
            }
            prefix_ratio_ds[i + 1] = F2 { x: acc_hi, y: acc_lo };
        }

        Ok((prefix_ratio_ds, prefix_nan, first))
    }

    #[allow(dead_code)]
    fn precompute_ratio_prefix_time_major_ds(
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<(Vec<F2>, Vec<i32>, Vec<i32>), CudaMassError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMassError::InvalidInput("cols/rows zero".into()));
        }
        let total = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMassError::InvalidInput("grid size overflow".into()))?;
        if high_tm.len() != total || low_tm.len() != total {
            return Err(CudaMassError::InvalidInput(
                "time-major inputs wrong length".into(),
            ));
        }
        let mut prefix_ratio_tm_ds = vec!(F2::default(); total + 1);
        let mut prefix_nan_tm = vec![0i32; total + 1];
        let mut first_valids = vec![0i32; cols];

        let alpha: f32 = 2.0f32 / 10.0f32;
        let inv_alpha: f32 = 1.0f32 - alpha;

        for s in 0..cols {
            // find first valid row for this series
            let fv = (0..rows)
                .find(|&t| high_tm[t * cols + s].is_finite() && low_tm[t * cols + s].is_finite())
                .unwrap_or(rows);
            first_valids[s] = fv as i32;

            let mut acc_hi: f32 = 0.0;
            let mut acc_lo: f32 = 0.0;
            let mut nan_cnt: i32 = 0;

            let mut ema1: f32 = 0.0;
            let mut ema2: f32 = 0.0;
            let start_ema2 = fv + 8;
            let start_ratio = fv + 16;
            if fv < rows {
                ema1 = high_tm[fv * cols + s] - low_tm[fv * cols + s];
                ema2 = ema1;
            }

            for t in 0..rows {
                let idx = t * cols + s;
                let mut ratio = f32::NAN;
                if t >= fv {
                    let hl = high_tm[idx] - low_tm[idx];
                    ema1 = inv_alpha.mul_add(ema1, alpha * hl);
                    if t == start_ema2 { ema2 = ema1; }
                    if t >= start_ema2 {
                        ema2 = inv_alpha.mul_add(ema2, alpha * ema1);
                        if t >= start_ratio {
                            ratio = ema1 / ema2;
                        }
                    }
                }
                if ratio.is_finite() {
                    (acc_hi, acc_lo) = ds_add(acc_hi, acc_lo, ratio);
                } else {
                    nan_cnt += 1;
                }
                prefix_ratio_tm_ds[idx + 1] = F2 { x: acc_hi, y: acc_lo };
                prefix_nan_tm[idx + 1] = nan_cnt;
            }
        }

        Ok((prefix_ratio_tm_ds, prefix_nan_tm, first_valids))
    }

    fn precompute_ratio_prefix_time_major(
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
    ) -> Result<(Vec<f64>, Vec<i32>, Vec<i32>), CudaMassError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMassError::InvalidInput("cols/rows zero".into()));
        }
        let total = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMassError::InvalidInput("grid size overflow".into()))?;
        if high_tm.len() != total || low_tm.len() != total {
            return Err(CudaMassError::InvalidInput(
                "time-major inputs wrong length".into(),
            ));
        }
        let mut prefix_nan_tm = vec![0i32; total + 1];
        let mut first_valids = vec![0i32; cols];

        // Build double prefixes exactly like the original path (time-major flattened)
        let mut prefix_ratio_tm_f64 = vec![0.0f64; total + 1];
        let alpha = 2.0f64 / 10.0f64;
        let inv_alpha = 1.0f64 - alpha;

        for s in 0..cols {
            // find first valid
            let mut fv: Option<usize> = None;
            for t in 0..rows {
                let h = high_tm[t * cols + s];
                let l = low_tm[t * cols + s];
                if h.is_finite() && l.is_finite() {
                    fv = Some(t);
                    break;
                }
                if h.is_finite() && l.is_finite() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = match fv {
                Some(x) => x,
                None => {
                    first_valids[s] = rows as i32;
                    continue;
                }
            };
            first_valids[s] = fv as i32;

            let mut ema1 = (high_tm[fv * cols + s] as f64) - (low_tm[fv * cols + s] as f64);
            let mut ema2 = ema1;
            let start_ema2 = fv + 8;
            let start_ratio = fv + 16;

            for t in 0..rows {
                let idx = t * cols + s;
                if t < fv {
                    prefix_ratio_tm_f64[idx + 1] = prefix_ratio_tm_f64[idx];
                    prefix_nan_tm[idx + 1] = prefix_nan_tm[idx];
                    continue;
                }
                let hl = (high_tm[idx] as f64) - (low_tm[idx] as f64);
                ema1 = ema1.mul_add(inv_alpha, hl * alpha);
                if t == start_ema2 {
                    ema2 = ema1;
                }
                if t == start_ema2 {
                    ema2 = ema1;
                }
                let mut ratio = f64::NAN;
                if t >= start_ema2 {
                    ema2 = ema2.mul_add(inv_alpha, ema1 * alpha);
                    if t >= start_ratio {
                        ratio = ema1 / ema2;
                    }
                }
                let is_nan = !ratio.is_finite();
                prefix_ratio_tm_f64[idx + 1] = prefix_ratio_tm_f64[idx] + if is_nan { 0.0 } else { ratio };
                prefix_nan_tm[idx + 1] = prefix_nan_tm[idx] + if is_nan { 1 } else { 0 };
            }
        }

        Ok((prefix_ratio_tm_f64, prefix_nan_tm, first_valids))
    }

    // ----------------------- Public device entry points -----------------------

    pub fn mass_batch_dev(
        &mut self,
        high: &[f32],
        low: &[f32],
        sweep: &MassBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<MassParams>), CudaMassError> {
        if high.is_empty() || low.is_empty() || high.len() != low.len() {
            return Err(CudaMassError::InvalidInput(
                "mismatched or empty inputs".into(),
            ));
        }

        // Expand parameter grid
        let combos = expand_mass_combos(sweep)?;
        if combos.is_empty() {
            return Err(CudaMassError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let (prefix_ratio_ds, prefix_nan, first_valid) =
            Self::precompute_ratio_prefix_one_series_ds(high, low)?;

        let len = high.len();
        let max_period = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_period == 0 || len - (first_valid as usize) < max_period {
            return Err(CudaMassError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                len - first_valid
            )));
        }

        // VRAM check with headroom and checked arithmetic
        let size_f2 = std::mem::size_of::<F2>();
        let size_i32 = std::mem::size_of::<i32>();
        let size_f32 = std::mem::size_of::<f32>();

        let ratio_bytes = prefix_ratio_ds
            .len()
            .checked_mul(size_f2)
            .ok_or_else(|| CudaMassError::InvalidInput("size overflow (ratio)".into()))?;
        let nan_bytes = prefix_nan
            .len()
            .checked_mul(size_i32)
            .ok_or_else(|| CudaMassError::InvalidInput("size overflow (nan prefix)".into()))?;
        let periods_bytes = combos
            .len()
            .checked_mul(size_i32)
            .ok_or_else(|| CudaMassError::InvalidInput("size overflow (periods)".into()))?;
        let out_elems = len
            .checked_mul(combos.len())
            .ok_or_else(|| CudaMassError::InvalidInput("output size overflow".into()))?;
        let out_bytes = out_elems
            .checked_mul(size_f32)
            .ok_or_else(|| CudaMassError::InvalidInput("size overflow (output bytes)".into()))?;

        let bytes_needed = ratio_bytes
            .checked_add(nan_bytes)
            .and_then(|v| v.checked_add(periods_bytes))
            .and_then(|v| v.checked_add(out_bytes))
            .ok_or_else(|| CudaMassError::InvalidInput("total VRAM size overflow".into()))?;

        let headroom = 64usize << 20; // ~64MB
        Self::will_fit(bytes_needed, headroom)?;

        // Upload inputs
        let d_prefix_ratio = DeviceBuffer::from_slice(&prefix_ratio_ds)?;
        let d_prefix_nan = DeviceBuffer::from_slice(&prefix_nan)?;
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.period.unwrap_or(0) as i32)
            .collect();
        let d_periods = DeviceBuffer::from_slice(&periods_i32)?;
        let mut d_out: DeviceBuffer<f32> = DeviceBuffer::zeroed(out_elems)?;

        // Launch config
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        self.maybe_log_selected("batch", block_x);
        let grid_x = ((len as u32) + block_x - 1) / block_x;
        let block: BlockSize = (block_x, 1, 1).into();
        let stream = &self.stream;

        // Chunk Y if needed
        let mut launched = 0usize;
        const MAX_GRID_Y: usize = 65_535;
        while launched < combos.len() {
            let chunk = (combos.len() - launched).min(MAX_GRID_Y);
            let func = self
                .module
                .get_function("mass_batch_f32")
                .map_err(|_| CudaMassError::MissingKernelSymbol { name: "mass_batch_f32" })?;
            let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
            unsafe {
                launch!(
                    func<<<grid, block, 0, stream>>>(
                        d_prefix_ratio.as_device_ptr(),
                        d_prefix_nan.as_device_ptr(),
                        len as i32,
                        first_valid as i32,
                        d_periods.as_device_ptr().offset(launched as isize),
                        chunk as i32,
                        d_out.as_device_ptr().offset((launched * len) as isize)
                    )
                )
                ?;
            }
            launched += chunk;
        }
        self.stream.synchronize()?;
        Ok((
            DeviceArrayF32 {
                buf: d_out,
                rows: combos.len(),
                cols: len,
            },
            combos,
        ))
    }

    pub fn mass_many_series_one_param_time_major_dev(
        &mut self,
        high_tm: &[f32],
        low_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &MassParams,
    ) -> Result<DeviceArrayF32, CudaMassError> {
        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaMassError::InvalidInput("period=0".into()));
        }
        let total = cols
            .checked_mul(rows)
            .ok_or_else(|| CudaMassError::InvalidInput("grid size overflow".into()))?;
        if high_tm.len() != total || low_tm.len() != total {
            return Err(CudaMassError::InvalidInput(
                "time-major inputs wrong length".into(),
            ));
        }

        // Host buffer used only for upload; still guard VRAM for the final DeviceBuffer.
        let bytes_out = total
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMassError::InvalidInput("output bytes overflow".into()))?;
        Self::will_fit(bytes_out, 64usize << 20)?;

        // Fallback to CPU compute for correctness, then upload to device.
        let mut host_tm = vec![0f32; total];
        for s in 0..cols {
            let mut h = vec![f64::NAN; rows];
            let mut l = vec![f64::NAN; rows];
            for t in 0..rows {
                h[t] = high_tm[t * cols + s] as f64;
                l[t] = low_tm[t * cols + s] as f64;
            }
            let p = MassParams { period: Some(period) };
            let input = MassInput { data: MassData::Slices { high: &h, low: &l }, params: p };
            let out = mass_with_kernel(&input, Kernel::Scalar)
                .map_err(|e| CudaMassError::InvalidInput(format!("cpu mass error: {}", e)))?;
            for t in 0..rows {
                host_tm[t * cols + s] = out.values[t] as f32;
            }
        }

        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(total) }?;
        unsafe {
            d_out_tm
                .async_copy_from(host_tm.as_slice(), &self.stream)?;
        }
        self.stream.synchronize()?;
        Ok(DeviceArrayF32 { buf: d_out_tm, rows, cols })
    }
}

// ----------------------- helpers -----------------------

fn expand_mass_combos(r: &MassBatchRange) -> Result<Vec<MassParams>, CudaMassError> {
    #[inline]
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, CudaMassError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let v: Vec<usize> = (start..=end).step_by(step).collect();
            if v.is_empty() {
                return Err(CudaMassError::InvalidInput(
                    "invalid range (empty expansion)".into(),
                ));
            }
            Ok(v)
        } else {
            // Reversed bounds supported: walk downward including both ends.
            let mut v = Vec::new();
            let mut cur = start;
            loop {
                v.push(cur);
                if cur <= end {
                    break;
                }
                match cur.checked_sub(step) {
                    Some(next) => {
                        cur = next;
                    }
                    None => break,
                }
            }
            if v.is_empty() {
                Err(CudaMassError::InvalidInput(
                    "invalid range (empty expansion)".into(),
                ))
            } else {
                Ok(v)
            }
        }
    }

    let periods = axis_usize(r.period)?;
    if periods.is_empty() {
        return Err(CudaMassError::InvalidInput(
            "invalid range (empty expansion)".into(),
        ));
    }
    let mut v = Vec::with_capacity(periods.len());
    for p in periods {
        v.push(MassParams { period: Some(p) });
    }
    Ok(v)
}

// ----------------------- benches registration -----------------------

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};
    use std::ffi::c_void;

    struct MassBatchState {
        cuda: CudaMass,
        d_prefix_ratio: DeviceBuffer<F2>,
        d_prefix_nan: DeviceBuffer<i32>,
        d_periods: DeviceBuffer<i32>,
        d_out: DeviceBuffer<f32>,
        len: usize,
        first_valid: usize,
        n_combos: usize,
    }
    impl CudaBenchState for MassBatchState {
        fn launch(&mut self) {
            let block_x: u32 = 256;
            let grid_x = ((self.len as u32) + block_x - 1) / block_x;
            let block: BlockSize = (block_x, 1, 1).into();
            let stream = &self.cuda.stream;

            const MAX_GRID_Y: usize = 65_535;
            let mut launched = 0usize;
            while launched < self.n_combos {
                let chunk = (self.n_combos - launched).min(MAX_GRID_Y);
                let grid: GridSize = (grid_x.max(1), chunk as u32, 1).into();
                unsafe {
                    let func = self
                        .cuda
                        .module
                        .get_function("mass_batch_f32")
                        .expect("mass_batch_f32");
                    let mut prefix_ptr = self.d_prefix_ratio.as_device_ptr().as_raw();
                    let mut nan_ptr = self.d_prefix_nan.as_device_ptr().as_raw();
                    let mut len_i = self.len as i32;
                    let mut first_i = self.first_valid as i32;
                    let mut periods_ptr = self
                        .d_periods
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add((launched * std::mem::size_of::<i32>()) as u64);
                    let mut combos_i = chunk as i32;
                    let mut out_ptr = self
                        .d_out
                        .as_device_ptr()
                        .as_raw()
                        .wrapping_add(((launched * self.len) * std::mem::size_of::<f32>()) as u64);
                    let args: &mut [*mut c_void] = &mut [
                        &mut prefix_ptr as *mut _ as *mut c_void,
                        &mut nan_ptr as *mut _ as *mut c_void,
                        &mut len_i as *mut _ as *mut c_void,
                        &mut first_i as *mut _ as *mut c_void,
                        &mut periods_ptr as *mut _ as *mut c_void,
                        &mut combos_i as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    stream.launch(&func, grid, block, 0, args).expect("launch");
                }
                launched += chunk;
            }
            self.cuda.stream.synchronize().expect("mass sync");
        }
    }

    fn prep_mass_batch() -> Box<dyn CudaBenchState> {
        let mut high = vec![f32::NAN; 120_000];
        let mut low = vec![f32::NAN; 120_000];
        for i in 20..high.len() {
            let x = i as f32;
            high[i] = (x * 0.0023).sin().abs() + 1.0;
            low[i] = high[i] - (0.5 + (x * 0.0017).cos().abs());
        }
        let sweep = MassBatchRange { period: (2, 32, 2) };
        let combos = expand_mass_combos(&sweep).expect("expand_mass_combos");
        let n_combos = combos.len();
        let len = high.len();

        let (prefix_ratio_ds, prefix_nan, first_valid) =
            CudaMass::precompute_ratio_prefix_one_series_ds(&high, &low)
                .expect("precompute_ratio_prefix_one_series_ds");
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.period.unwrap_or(0) as i32)
            .collect();

        let cuda = CudaMass::new(0).expect("cuda mass");
        let d_prefix_ratio = DeviceBuffer::from_slice(&prefix_ratio_ds).expect("d_prefix_ratio");
        let d_prefix_nan = DeviceBuffer::from_slice(&prefix_nan).expect("d_prefix_nan");
        let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
        let d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * len) }.expect("d_out");
        cuda.stream.synchronize().expect("mass prep sync");

        Box::new(MassBatchState {
            cuda,
            d_prefix_ratio,
            d_prefix_nan,
            d_periods,
            d_out,
            len,
            first_valid,
            n_combos,
        })
    }

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        vec![
            CudaBenchScenario::new(
                "mass",
                "batch_dev",
                "mass_cuda_batch_dev",
                "120k_x_16combos",
                prep_mass_batch,
            )
            .with_inner_iters(4),
        ]
    }
}
