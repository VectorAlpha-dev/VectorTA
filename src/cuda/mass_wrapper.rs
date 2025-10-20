//! CUDA support for the Mass Index (MASS) indicator.
//!
//! Mirrors ALMA-class wrappers for policy, PTX load, NON_BLOCKING stream,
//! VRAM checks, chunking, and bench registration. The math follows the scalar
//! path in `indicators::mass` exactly: two nested EMA(9) filters on (high-low)
//! to form `ratio = ema1/ema2`, then a rolling sum over `period` bars.
//!
//! One-series × many-params (batch): we precompute the ratio and its
//! double-precision prefix sums on the host and upload them, enabling each
//! row (period) to evaluate O(1) per timestamp via prefix differences. We also
//! upload a prefix count of NaNs so any window containing a NaN yields NaN.
//!
//! Many-series × one-param (time-major): similarly uses time-major prefixes
//! of `ratio` and a prefix NaN-count buffer.

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::mass::{MassBatchRange, MassParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::fmt;

#[derive(Debug)]
pub enum CudaMassError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaMassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaMassError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaMassError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaMassError {}

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
    _context: Context,
    policy: CudaMassPolicy,
    debug_logged: bool,
}

impl CudaMass {
    pub fn new(device_id: usize) -> Result<Self, CudaMassError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaMassError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaMassError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaMassError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/mass_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaMassError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaMassError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaMassPolicy::default(),
            debug_logged: false,
        })
    }

    pub fn set_policy(&mut self, policy: CudaMassPolicy) {
        self.policy = policy;
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

    // ----------------------- Host precompute helpers -----------------------
    fn first_valid_hilo(high: &[f32], low: &[f32]) -> Option<usize> {
        high.iter()
            .zip(low.iter())
            .position(|(&h, &l)| h.is_finite() && l.is_finite())
    }

    fn precompute_ratio_prefix_one_series(
        high: &[f32],
        low: &[f32],
    ) -> Result<(Vec<f64>, Vec<i32>, usize), CudaMassError> {
        if high.len() != low.len() || high.is_empty() {
            return Err(CudaMassError::InvalidInput(
                "mismatched or empty inputs".into(),
            ));
        }
        let n = high.len();
        let first = Self::first_valid_hilo(high, low)
            .ok_or_else(|| CudaMassError::InvalidInput("all values are NaN".into()))?;

        let mut prefix_ratio = vec![0f64; n + 1];
        let mut prefix_nan = vec![0i32; n + 1];

        // EMA(9) constants
        let alpha = 2.0f64 / 10.0;
        let inv_alpha = 1.0f64 - alpha;

        let mut ema1 = (high[first] as f64) - (low[first] as f64);
        let mut ema2 = ema1;
        let start_ema2 = first + 8;
        let start_ratio = first + 16;

        for i in 0..n {
            if i < first {
                prefix_ratio[i + 1] = prefix_ratio[i];
                prefix_nan[i + 1] = prefix_nan[i];
                continue;
            }
            let hl = (high[i] as f64) - (low[i] as f64);
            ema1 = ema1.mul_add(inv_alpha, hl * alpha);
            if i == start_ema2 {
                ema2 = ema1;
            }
            let mut ratio = f64::NAN;
            if i >= start_ema2 {
                ema2 = ema2.mul_add(inv_alpha, ema1 * alpha);
                if i >= start_ratio {
                    ratio = ema1 / ema2;
                }
            }
            let is_nan = !ratio.is_finite();
            prefix_ratio[i + 1] = prefix_ratio[i] + if is_nan { 0.0 } else { ratio };
            prefix_nan[i + 1] = prefix_nan[i] + if is_nan { 1 } else { 0 };
        }

        Ok((prefix_ratio, prefix_nan, first))
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
        if high_tm.len() != cols * rows || low_tm.len() != cols * rows {
            return Err(CudaMassError::InvalidInput(
                "time-major inputs wrong length".into(),
            ));
        }
        let mut prefix_ratio_tm = vec![0f64; cols * rows + 1];
        let mut prefix_nan_tm = vec![0i32; cols * rows + 1];
        let mut first_valids = vec![0i32; cols];

        // Per series EMA precompute
        let alpha = 2.0f64 / 10.0;
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
                    prefix_ratio_tm[idx + 1] = prefix_ratio_tm[idx];
                    prefix_nan_tm[idx + 1] = prefix_nan_tm[idx];
                    continue;
                }
                let hl = (high_tm[idx] as f64) - (low_tm[idx] as f64);
                ema1 = ema1.mul_add(inv_alpha, hl * alpha);
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
                prefix_ratio_tm[idx + 1] = prefix_ratio_tm[idx] + if is_nan { 0.0 } else { ratio };
                prefix_nan_tm[idx + 1] = prefix_nan_tm[idx] + if is_nan { 1 } else { 0 };
            }
        }

        Ok((prefix_ratio_tm, prefix_nan_tm, first_valids))
    }

    // ----------------------- Public device entry points -----------------------

    pub fn mass_batch_dev(
        &mut self,
        high: &[f32],
        low: &[f32],
        sweep: &MassBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<MassParams>), CudaMassError> {
        if high.len() != low.len() || high.is_empty() {
            return Err(CudaMassError::InvalidInput(
                "mismatched or empty inputs".into(),
            ));
        }

        // Expand parameter grid
        let combos = expand_mass_combos(sweep);
        if combos.is_empty() {
            return Err(CudaMassError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let (prefix_ratio, prefix_nan, first_valid) =
            Self::precompute_ratio_prefix_one_series(high, low)?;

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

        // VRAM check with headroom
        let bytes_needed = prefix_ratio.len() * std::mem::size_of::<f64>()
            + prefix_nan.len() * std::mem::size_of::<i32>()
            + combos.len() * std::mem::size_of::<i32>()
            + (len * combos.len()) * std::mem::size_of::<f32>();
        if let Ok((free, _)) = mem_get_info() {
            let headroom = 64usize << 20; // ~64MB
            if bytes_needed + headroom > free {
                return Err(CudaMassError::InvalidInput(
                    "insufficient device memory for mass batch".into(),
                ));
            }
        }

        // Upload inputs
        let d_prefix_ratio = DeviceBuffer::from_slice(&prefix_ratio)
            .map_err(|e| CudaMassError::Cuda(e.to_string()))?;
        let d_prefix_nan = DeviceBuffer::from_slice(&prefix_nan)
            .map_err(|e| CudaMassError::Cuda(e.to_string()))?;
        let periods_i32: Vec<i32> = combos
            .iter()
            .map(|c| c.period.unwrap_or(0) as i32)
            .collect();
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaMassError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = DeviceBuffer::zeroed(len * combos.len())
            .map_err(|e| CudaMassError::Cuda(e.to_string()))?;

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
        while launched < combos.len() {
            let chunk = (combos.len() - launched).min(65_535);
            let func = self
                .module
                .get_function("mass_batch_f32")
                .map_err(|e| CudaMassError::Cuda(e.to_string()))?;
            let grid: GridSize = (grid_x, chunk as u32, 1).into();
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
                .map_err(|e| CudaMassError::Cuda(e.to_string()))?;
            }
            launched += chunk;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaMassError::Cuda(e.to_string()))?;

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
        let (prefix_ratio_tm, prefix_nan_tm, first_valids) =
            Self::precompute_ratio_prefix_time_major(high_tm, low_tm, cols, rows)?;

        // VRAM check
        let bytes_needed = prefix_ratio_tm.len() * std::mem::size_of::<f64>()
            + prefix_nan_tm.len() * std::mem::size_of::<i32>()
            + first_valids.len() * std::mem::size_of::<i32>()
            + (cols * rows) * std::mem::size_of::<f32>();
        if let Ok((free, _)) = mem_get_info() {
            let headroom = 64usize << 20;
            if bytes_needed + headroom > free {
                return Err(CudaMassError::InvalidInput(
                    "insufficient device memory for mass many-series".into(),
                ));
            }
        }

        // Upload inputs
        let d_prefix_ratio_tm = DeviceBuffer::from_slice(&prefix_ratio_tm)
            .map_err(|e| CudaMassError::Cuda(e.to_string()))?;
        let d_prefix_nan_tm = DeviceBuffer::from_slice(&prefix_nan_tm)
            .map_err(|e| CudaMassError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaMassError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> =
            DeviceBuffer::zeroed(cols * rows).map_err(|e| CudaMassError::Cuda(e.to_string()))?;

        // Launch config
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x,
            _ => 256,
        };
        self.maybe_log_selected("many_series", block_x);
        let grid_x = ((rows as u32) + block_x - 1) / block_x; // along time
        let grid_y = (cols as u32).min(65_535);
        let grid: GridSize = (grid_x, grid_y, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let stream = &self.stream;

        let func = self
            .module
            .get_function("mass_many_series_one_param_time_major_f32")
            .map_err(|e| CudaMassError::Cuda(e.to_string()))?;
        unsafe {
            launch!(
                func<<<grid, block, 0, stream>>>(
                    d_prefix_ratio_tm.as_device_ptr(),
                    d_prefix_nan_tm.as_device_ptr(),
                    period as i32,
                    cols as i32,
                    rows as i32,
                    d_first_valids.as_device_ptr(),
                    d_out_tm.as_device_ptr()
                )
            )
            .map_err(|e| CudaMassError::Cuda(e.to_string()))?;
        }
        self.stream
            .synchronize()
            .map_err(|e| CudaMassError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }
}

// ----------------------- helpers -----------------------

fn expand_mass_combos(r: &MassBatchRange) -> Vec<MassParams> {
    let (start, end, step) = r.period;
    let mut v = Vec::new();
    if step == 0 {
        v.push(MassParams {
            period: Some(start),
        });
        return v;
    }
    if start == 0 || end == 0 || start > end || step == 0 {
        return v;
    }
    let mut p = start;
    while p <= end {
        v.push(MassParams { period: Some(p) });
        match p.checked_add(step) {
            Some(nxt) => p = nxt,
            None => break,
        }
    }
    v
}

// ----------------------- benches registration -----------------------

pub mod benches {
    use super::*;
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    struct MassBatchState {
        cuda: CudaMass,
        high: Vec<f32>,
        low: Vec<f32>,
        sweep: MassBatchRange,
    }
    impl CudaBenchState for MassBatchState {
        fn launch(&mut self) {
            let _ = self
                .cuda
                .mass_batch_dev(&self.high, &self.low, &self.sweep)
                .expect("mass batch");
        }
    }

    struct MassManySeriesState {
        cuda: CudaMass,
        high_tm: Vec<f32>,
        low_tm: Vec<f32>,
        cols: usize,
        rows: usize,
        period: usize,
    }
    impl CudaBenchState for MassManySeriesState {
        fn launch(&mut self) {
            let p = MassParams {
                period: Some(self.period),
            };
            let _ = self
                .cuda
                .mass_many_series_one_param_time_major_dev(
                    &self.high_tm,
                    &self.low_tm,
                    self.cols,
                    self.rows,
                    &p,
                )
                .expect("mass many-series");
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
        Box::new(MassBatchState {
            cuda: CudaMass::new(0).expect("cuda mass"),
            high,
            low,
            sweep,
        })
    }

    fn prep_mass_many_series() -> Box<dyn CudaBenchState> {
        let cols = 64usize;
        let rows = 120_000usize;
        let mut high_tm = vec![f32::NAN; cols * rows];
        let mut low_tm = vec![f32::NAN; cols * rows];
        for s in 0..cols {
            for t in s..rows {
                let x = (t as f32) + (s as f32) * 0.1;
                let h = (x * 0.002).sin().abs() + 1.1;
                let l = h - (0.4 + (x * 0.0013).cos().abs());
                high_tm[t * cols + s] = h;
                low_tm[t * cols + s] = l;
            }
        }
        Box::new(MassManySeriesState {
            cuda: CudaMass::new(0).expect("cuda mass"),
            high_tm,
            low_tm,
            cols,
            rows,
            period: 9,
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
            CudaBenchScenario::new(
                "mass",
                "many_series_one_param",
                "mass_cuda_many_series_one_param",
                "64x120k",
                prep_mass_many_series,
            )
            .with_inner_iters(2),
        ]
    }
}
