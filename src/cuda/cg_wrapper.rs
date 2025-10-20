//! CUDA wrapper for Center of Gravity (CG).
//!
//! Mirrors the ALMA wrapper surface where useful while keeping the
//! implementation simple (no tiling variants needed). Numeric semantics match
//! the scalar CG implementation:
//! - Warmup index: `first_valid + period`
//! - Denominator near-zero or NaN => write 0.0
//! - NaN prefix before warmup
//! - FP32 compute with FP32 I/O

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::cg::{CgBatchRange, CgParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
pub enum CudaCgError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaCgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaCgError::Cuda(e) => write!(f, "CUDA error: {e}"),
            CudaCgError::InvalidInput(e) => write!(f, "Invalid input: {e}"),
        }
    }
}
impl std::error::Error for CudaCgError {}

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
pub struct CudaCgPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

impl Default for CudaCgPolicy {
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

pub struct CudaCg {
    module: Module,
    stream: Stream,
    _context: Context,
    device_id: u32,
    policy: CudaCgPolicy,
    last_batch: Option<BatchKernelSelected>,
    last_many: Option<ManySeriesKernelSelected>,
    debug_batch_logged: bool,
    debug_many_logged: bool,
}

impl CudaCg {
    pub fn new(device_id: usize) -> Result<Self, CudaCgError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaCgError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaCgError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaCgError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cg_kernel.ptx"));
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
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaCgError::Cuda(e.to_string()))?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaCgError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            device_id: device_id as u32,
            policy: CudaCgPolicy::default(),
            last_batch: None,
            last_many: None,
            debug_batch_logged: false,
            debug_many_logged: false,
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaCgPolicy) -> Result<Self, CudaCgError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    pub fn synchronize(&self) -> Result<(), CudaCgError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaCgError::Cuda(e.to_string()))
    }

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
                    eprintln!("[DEBUG] CG batch selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaCg)).debug_batch_logged = true };
            }
        }
    }

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
                    eprintln!("[DEBUG] CG many-series selected kernel: {:?}", sel);
                }
                unsafe { (*(self as *const _ as *mut CudaCg)).debug_many_logged = true };
            }
        }
    }

    // -------- Batch (one series × many params) --------
    pub fn cg_batch_dev(
        &self,
        prices_f32: &[f32],
        sweep: &CgBatchRange,
    ) -> Result<DeviceArrayF32, CudaCgError> {
        let len = prices_f32.len();
        if len == 0 {
            return Err(CudaCgError::InvalidInput("empty input".into()));
        }

        let combos = expand_grid_cg(sweep);
        if combos.is_empty() {
            return Err(CudaCgError::InvalidInput("no parameter combos".into()));
        }

        let first_valid = prices_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaCgError::InvalidInput("all values are NaN".into()))?;
        let max_p = combos
            .iter()
            .map(|c| c.period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_p == 0 {
            return Err(CudaCgError::InvalidInput("period must be positive".into()));
        }
        if len - first_valid < (max_p + 1) {
            return Err(CudaCgError::InvalidInput(format!(
                "not enough valid data: need >= {}, have {}",
                max_p + 1,
                len - first_valid
            )));
        }

        // Upload inputs
        let d_prices =
            DeviceBuffer::from_slice(prices_f32).map_err(|e| CudaCgError::Cuda(e.to_string()))?;
        let periods: Vec<i32> = combos
            .iter()
            .map(|p| p.period.unwrap_or(0) as i32)
            .collect();
        let d_periods =
            DeviceBuffer::from_slice(&periods).map_err(|e| CudaCgError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(len * combos.len())
                .map_err(|e| CudaCgError::Cuda(e.to_string()))?
        };

        // Launch
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 256u32,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32).min(1024),
        };
        let grid_x = ((combos.len() as u32) + block_x - 1) / block_x;
        let func = self
            .module
            .get_function("cg_batch_f32")
            .map_err(|e| CudaCgError::Cuda(e.to_string()))?;
        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut combos_i = combos.len() as i32;
            let mut first_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let bs = (block_x, 1, 1);
            let gs = (grid_x, 1, 1);
            self.stream
                .launch(&func, gs, bs, 0, args)
                .map_err(|e| CudaCgError::Cuda(e.to_string()))?;
        }
        self.maybe_log_batch_debug();

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    // -------- Many-series × one param (time-major) --------
    pub fn cg_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CgParams,
    ) -> Result<DeviceArrayF32, CudaCgError> {
        if cols == 0 || rows == 0 {
            return Err(CudaCgError::InvalidInput("empty matrix shape".into()));
        }
        if prices_tm_f32.len() != cols * rows {
            return Err(CudaCgError::InvalidInput(
                "time-major input size mismatch".into(),
            ));
        }
        let period = params.period.unwrap_or(10);
        if period == 0 || period > rows {
            return Err(CudaCgError::InvalidInput("invalid period".into()));
        }

        // Compute per-series first_valids over time-major input
        let first_valids = compute_first_valids_time_major(prices_tm_f32, cols, rows);

        let d_prices = DeviceBuffer::from_slice(prices_tm_f32)
            .map_err(|e| CudaCgError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaCgError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(cols * rows)
                .map_err(|e| CudaCgError::Cuda(e.to_string()))?
        };

        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 256u32,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32).min(1024),
        };
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let func = self
            .module
            .get_function("cg_many_series_one_param_f32")
            .map_err(|e| CudaCgError::Cuda(e.to_string()))?;
        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut first_ptr = d_first.as_device_ptr().as_raw();
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut period_i = period as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            let bs = (block_x, 1, 1);
            let gs = (grid_x, 1, 1);
            self.stream
                .launch(&func, gs, bs, 0, args)
                .map_err(|e| CudaCgError::Cuda(e.to_string()))?;
        }
        self.maybe_log_many_debug();

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }
}

fn expand_grid_cg(r: &CgBatchRange) -> Vec<CgParams> {
    let (start, end, step) = r.period;
    let vals: Vec<usize> = if step == 0 || start == end {
        vec![start]
    } else {
        (start..=end).step_by(step).collect()
    };
    vals.into_iter()
        .map(|p| CgParams { period: Some(p) })
        .collect()
}

fn compute_first_valids_time_major(data_tm: &[f32], cols: usize, rows: usize) -> Vec<i32> {
    let mut v = vec![-1i32; cols];
    for c in 0..cols {
        let mut fv = -1i32;
        for r in 0..rows {
            let val = data_tm[r * cols + c];
            if !val.is_nan() {
                fv = r as i32;
                break;
            }
        }
        v[c] = fv;
    }
    v
}

// ---- Minimal benches registration to integrate with cuda_bench.rs ----
pub mod benches {
    use super::*;
    use crate::cuda::{CudaBenchScenario, CudaBenchState};

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        let mut v = Vec::new();
        // Batch: one series × many params
        v.push(
            CudaBenchScenario::new(
                "cg",
                "one_series_many_params",
                "cg",
                "cg_batch/1x-many",
                || {
                    struct St {
                        cuda: CudaCg,
                        prices: Vec<f32>,
                        sweep: CgBatchRange,
                    }
                    impl CudaBenchState for St {
                        fn launch(&mut self) {
                            let _ = self
                                .cuda
                                .cg_batch_dev(&self.prices, &self.sweep)
                                .expect("cg_batch_dev");
                        }
                    }
                    let prices = (0..100_000).map(|i| (i as f32).sin()).collect::<Vec<_>>();
                    let sweep = CgBatchRange {
                        period: (10, 40, 10),
                    };
                    let cuda = CudaCg::new(0).expect("cuda cg");
                    Box::new(St {
                        cuda,
                        prices,
                        sweep,
                    })
                },
            )
            .with_sample_size(20)
            .with_inner_iters(1),
        );

        // Many-series: 512 series × 8192 rows, single param
        v.push(
            CudaBenchScenario::new(
                "cg",
                "many_series_one_param",
                "cg",
                "cg_many/series-major",
                || {
                    struct St {
                        cuda: CudaCg,
                        tm: Vec<f32>,
                        cols: usize,
                        rows: usize,
                        p: CgParams,
                    }
                    impl CudaBenchState for St {
                        fn launch(&mut self) {
                            let _ = self
                                .cuda
                                .cg_many_series_one_param_time_major_dev(
                                    &self.tm, self.cols, self.rows, &self.p,
                                )
                                .expect("cg_many_series_one_param_time_major_dev");
                        }
                    }
                    let cols = 512usize;
                    let rows = 8_192usize;
                    let mut tm = vec![f32::NAN; cols * rows];
                    for r in 0..rows {
                        for c in 0..cols {
                            tm[r * cols + c] = ((r as f32) * 0.001 + (c as f32) * 0.0001).sin();
                        }
                    }
                    let cuda = CudaCg::new(0).expect("cuda cg");
                    let p = CgParams { period: Some(20) };
                    Box::new(St {
                        cuda,
                        tm,
                        cols,
                        rows,
                        p,
                    })
                },
            )
            .with_sample_size(20)
            .with_inner_iters(1),
        );
        v
    }
}
