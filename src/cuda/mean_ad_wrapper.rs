//! CUDA wrapper for Mean Absolute Deviation (MeanAd).
//!
//! Matches ALMA-style API for policy, PTX load, stream, VRAM checks, and
//! public entrypoints. Math pattern is a recurrence per time step:
//! - Rolling SMA via window sum
//! - Rolling mean(|x - SMA|) via a period-length ring buffer
//!
//! Kernels expected:
//! - "mean_ad_batch_f32"
//! - "mean_ad_many_series_one_param_f32"

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32; // reuse common VRAM handle
use crate::indicators::mean_ad::{MeanAdBatchRange, MeanAdParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer, LockedBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaMeanAdError {
    Cuda(String),
    InvalidInput(String),
    NotImplemented,
}

impl fmt::Display for CudaMeanAdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaMeanAdError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaMeanAdError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
            CudaMeanAdError::NotImplemented => write!(f, "CUDA MeanAd not implemented"),
        }
    }
}
impl std::error::Error for CudaMeanAdError {}

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
pub struct CudaMeanAdPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}
impl Default for CudaMeanAdPolicy {
    fn default() -> Self {
        Self {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Auto,
        }
    }
}

pub struct CudaMeanAd {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaMeanAdPolicy,
}

impl CudaMeanAd {
    pub fn new(device_id: usize) -> Result<Self, CudaMeanAdError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/mean_ad_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = Module::from_ptx(ptx, jit_opts)
            .or_else(|_| Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]))
            .or_else(|_| Module::from_ptx(ptx, &[]))
            .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaMeanAdPolicy::default(),
        })
    }

    pub fn set_policy(&mut self, policy: CudaMeanAdPolicy) {
        self.policy = policy;
    }
    pub fn policy(&self) -> &CudaMeanAdPolicy {
        &self.policy
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &MeanAdBatchRange,
    ) -> Result<(Vec<MeanAdParams>, usize, usize, usize), CudaMeanAdError> {
        if data_f32.is_empty() {
            return Err(CudaMeanAdError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaMeanAdError::InvalidInput("all values are NaN".into()))?;
        // Local expansion: (start, end, step) -> [start..=end step-by-step]; if step==0 or start==end => [start]
        let combos: Vec<MeanAdParams> = {
            let (s, e, st) = sweep.period;
            if st == 0 || s == e {
                vec![MeanAdParams { period: Some(s) }]
            } else {
                let mut v = Vec::new();
                let mut p = s;
                while p <= e {
                    v.push(MeanAdParams { period: Some(p) });
                    p += st;
                }
                v
            }
        };
        if combos.is_empty() {
            return Err(CudaMeanAdError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let p = prm.period.unwrap_or(0);
            if p == 0 {
                return Err(CudaMeanAdError::InvalidInput("period must be > 0".into()));
            }
            if p > len {
                return Err(CudaMeanAdError::InvalidInput(
                    "period exceeds data length".into(),
                ));
            }
            if len - first_valid < p {
                return Err(CudaMeanAdError::InvalidInput(
                    "not enough valid data for period".into(),
                ));
            }
            max_period = max_period.max(p);
        }
        Ok((combos, first_valid, len, max_period))
    }

    pub fn mean_ad_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &MeanAdBatchRange,
    ) -> Result<DeviceArrayF32, CudaMeanAdError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        // VRAM estimate: prices + periods + warms + out
        let prices_bytes = series_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;
        let periods_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;
        let warms_bytes = n_combos
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;
        let out_bytes = n_combos
            .checked_mul(series_len)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| CudaMeanAdError::InvalidInput("size overflow".into()))?;
        if let Ok((free, _)) = mem_get_info() {
            let required = prices_bytes + periods_bytes + warms_bytes + out_bytes;
            let headroom = 64usize * 1024 * 1024;
            if required.saturating_add(headroom) > free {
                return Err(CudaMeanAdError::InvalidInput(
                    "insufficient free VRAM".into(),
                ));
            }
        }

        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut warms_i32 = Vec::with_capacity(n_combos);
        for prm in &combos {
            let p = prm.period.unwrap();
            periods_i32.push(p as i32);
            let warm = first_valid + 2 * p - 2;
            warms_i32.push(warm as i32);
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        let d_warms = DeviceBuffer::from_slice(&warms_i32)
            .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;

        let mut func = self
            .module
            .get_function("mean_ad_batch_f32")
            .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;

        let block_x = match self.policy.batch {
            BatchKernelPolicy::Plain { block_x } if block_x > 0 => block_x,
            _ => 128,
        };
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32; // per-block ring capacity

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut warms_ptr = d_warms.as_device_ptr().as_raw();
            let mut first_valid_i = first_valid as i32;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 7] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut warms_ptr as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&mut func, grid, block, shared_bytes, &mut args)
                .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MeanAdParams,
    ) -> Result<(Vec<i32>, usize), CudaMeanAdError> {
        if cols == 0 || rows == 0 {
            return Err(CudaMeanAdError::InvalidInput("empty grid".into()));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaMeanAdError::InvalidInput("data length mismatch".into()));
        }
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(CudaMeanAdError::InvalidInput("period must be > 0".into()));
        }
        if period > rows {
            return Err(CudaMeanAdError::InvalidInput(
                "period exceeds series length".into(),
            ));
        }
        // first_valid per series (column)
        let mut firsts = vec![0i32; cols];
        for s in 0..cols {
            let mut f = -1;
            for t in 0..rows {
                let v = data_tm_f32[t * cols + s];
                if !v.is_nan() {
                    f = t as i32;
                    break;
                }
            }
            firsts[s] = f;
        }
        Ok((firsts, period))
    }

    pub fn mean_ad_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &MeanAdParams,
    ) -> Result<DeviceArrayF32, CudaMeanAdError> {
        let (firsts, period) = Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        // Heuristic for dynamic shared memory: ring per thread => period * block_x * 4 bytes
        let max_shmem: usize = 48 * 1024; // conservative default
        let mut block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::OneD { block_x } if block_x > 0 => block_x as usize,
            _ => 128,
        };
        block_x = block_x
            .min(max_shmem / (period * std::mem::size_of::<f32>()))
            .max(1);
        let grid_x = ((cols + block_x - 1) / block_x) as u32;
        let block: BlockSize = (block_x as u32, 1, 1).into();
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let shared_bytes = (period * block_x * std::mem::size_of::<f32>()) as u32;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        let d_firsts =
            DeviceBuffer::from_slice(&firsts).map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;

        let mut func = self
            .module
            .get_function("mean_ad_many_series_one_param_f32")
            .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut firsts_ptr = d_firsts.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 6] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut firsts_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&mut func, grid, block, shared_bytes, &mut args)
                .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaMeanAdError::Cuda(e.to_string()))?;
        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }
}

// ---------- Bench profiles ----------
pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        mean_ad_benches,
        CudaMeanAd,
        crate::indicators::mean_ad::MeanAdBatchRange,
        crate::indicators::mean_ad::MeanAdParams,
        mean_ad_batch_dev,
        mean_ad_many_series_one_param_time_major_dev,
        crate::indicators::mean_ad::MeanAdBatchRange {
            period: (10, 10 + PARAM_SWEEP - 1, 1)
        },
        crate::indicators::mean_ad::MeanAdParams { period: Some(32) },
        "mean_ad",
        "mean_ad"
    );
    pub use mean_ad_benches::bench_profiles;
}
