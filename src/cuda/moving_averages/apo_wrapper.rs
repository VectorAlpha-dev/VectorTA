//! CUDA support for the Absolute Price Oscillator (APO).
//!
//! Parity with ALMA/EMA wrappers:
//! - NON_BLOCKING stream, PTX JIT with DetermineTargetFromContext, O2 fallback
//! - VRAM estimates with ~64MB headroom guard
//! - Policy enums for explicit configuration (kept simple here)
//! - Warmup/NaN semantics identical to scalar APO:
//!   prefix [0..first_valid) = NaN, out[first_valid] = 0.0, then EMA(se)-EMA(le)

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::apo::{ApoBatchRange, ApoParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, AsyncCopyDestination, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaApoError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaApoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaApoError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaApoError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}
impl std::error::Error for CudaApoError {}

#[derive(Clone, Copy, Debug, Default)]
pub enum BatchKernelPolicy {
    #[default]
    Auto,
    Plain { block_x: u32 },
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ManySeriesKernelPolicy {
    #[default]
    Auto,
    OneD { block_x: u32 },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaApoPolicy {
    pub batch: BatchKernelPolicy,
    pub many_series: ManySeriesKernelPolicy,
}

pub struct CudaApo {
    module: Module,
    stream: Stream,
    _context: Context,
    policy: CudaApoPolicy,
}

impl CudaApo {
    pub fn new(device_id: usize) -> Result<Self, CudaApoError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaApoError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaApoError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaApoError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/apo_kernel.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => Module::from_ptx(ptx, &[])
                    .map_err(|e| CudaApoError::Cuda(e.to_string()))?,
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaApoError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
            policy: CudaApoPolicy::default(),
        })
    }

    pub fn new_with_policy(device_id: usize, policy: CudaApoPolicy) -> Result<Self, CudaApoError> {
        let mut s = Self::new(device_id)?;
        s.policy = policy;
        Ok(s)
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaApoError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaApoError::Cuda(e.to_string()))
    }

    // -------------------- Public API: One series × many params --------------------
    pub fn apo_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &ApoBatchRange,
    ) -> Result<DeviceArrayF32, CudaApoError> {
        let prep = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n = prep.combos.len();

        // VRAM estimate
        let prices_bytes = prep.series_len * std::mem::size_of::<f32>();
        let params_bytes = (prep.short_periods.len() * std::mem::size_of::<i32>())
            + (prep.long_periods.len() * std::mem::size_of::<i32>())
            + (prep.short_alphas.len() * std::mem::size_of::<f32>())
            + (prep.long_alphas.len() * std::mem::size_of::<f32>());
        let out_bytes = n * prep.series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + params_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaApoError::InvalidInput(
                "insufficient device memory for APO batch".into(),
            ));
        }

        // Async H2D
        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_f32, &self.stream)
                .map_err(|e| CudaApoError::Cuda(e.to_string()))?
        };
        let d_sp = unsafe {
            DeviceBuffer::from_slice_async(&prep.short_periods, &self.stream)
                .map_err(|e| CudaApoError::Cuda(e.to_string()))?
        };
        let d_lp = unsafe {
            DeviceBuffer::from_slice_async(&prep.long_periods, &self.stream)
                .map_err(|e| CudaApoError::Cuda(e.to_string()))?
        };
        let d_sa = unsafe {
            DeviceBuffer::from_slice_async(&prep.short_alphas, &self.stream)
                .map_err(|e| CudaApoError::Cuda(e.to_string()))?
        };
        let d_la = unsafe {
            DeviceBuffer::from_slice_async(&prep.long_alphas, &self.stream)
                .map_err(|e| CudaApoError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(prep.series_len * n, &self.stream)
                .map_err(|e| CudaApoError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_sp,
            &d_sa,
            &d_lp,
            &d_la,
            prep.series_len,
            prep.first_valid,
            n,
            &mut d_out,
        )?;
        self.synchronize()?;
        Ok(DeviceArrayF32 { buf: d_out, rows: n, cols: prep.series_len })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apo_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_short_periods: &DeviceBuffer<i32>,
        d_short_alphas: &DeviceBuffer<f32>,
        d_long_periods: &DeviceBuffer<i32>,
        d_long_alphas: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaApoError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaApoError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if d_out.len() != series_len * n_combos {
            return Err(CudaApoError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_short_periods,
            d_short_alphas,
            d_long_periods,
            d_long_alphas,
            series_len,
            first_valid,
            n_combos,
            d_out,
        )?;
        self.synchronize()
    }

    pub fn apo_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &ApoParams,
    ) -> Result<DeviceArrayF32, CudaApoError> {
        let (first_valids, sp, lp, a_s, a_l) =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        // VRAM estimate
        let elems = num_series * series_len;
        let in_bytes = elems * std::mem::size_of::<f32>();
        let first_bytes = first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes = elems * std::mem::size_of::<f32>();
        let required = in_bytes + first_bytes + out_bytes;
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaApoError::InvalidInput(
                "insufficient device memory for APO many-series".into(),
            ));
        }

        let d_prices = unsafe {
            DeviceBuffer::from_slice_async(data_tm_f32, &self.stream)
                .map_err(|e| CudaApoError::Cuda(e.to_string()))?
        };
        let d_first = unsafe {
            DeviceBuffer::from_slice_async(&first_valids, &self.stream)
                .map_err(|e| CudaApoError::Cuda(e.to_string()))?
        };
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized_async(elems, &self.stream)
                .map_err(|e| CudaApoError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(&d_prices, &d_first, sp, a_s, lp, a_l, num_series, series_len, &mut d_out)?;
        self.synchronize()?;
        Ok(DeviceArrayF32 { buf: d_out, rows: series_len, cols: num_series })
    }

    // -------------------- Launch helpers --------------------
    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_sp: &DeviceBuffer<i32>,
        d_sa: &DeviceBuffer<f32>,
        d_lp: &DeviceBuffer<i32>,
        d_la: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaApoError> {
        let func = self
            .module
            .get_function("apo_batch_f32")
            .map_err(|e| CudaApoError::Cuda(e.to_string()))?;

        // Default launch config
        let block_x = match self.policy.batch {
            BatchKernelPolicy::Auto => 128u32,
            BatchKernelPolicy::Plain { block_x } => block_x.max(32),
        };
        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut p_ptr = d_prices.as_device_ptr().as_raw();
            let mut sp_ptr = d_sp.as_device_ptr().as_raw();
            let mut sa_ptr = d_sa.as_device_ptr().as_raw();
            let mut lp_ptr = d_lp.as_device_ptr().as_raw();
            let mut la_ptr = d_la.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut first_i = first_valid as i32;
            let mut n_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut p_ptr as *mut _ as *mut c_void,
                &mut sp_ptr as *mut _ as *mut c_void,
                &mut sa_ptr as *mut _ as *mut c_void,
                &mut lp_ptr as *mut _ as *mut c_void,
                &mut la_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut first_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaApoError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        short_period: i32,
        short_alpha: f32,
        long_period: i32,
        long_alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaApoError> {
        let func = self
            .module
            .get_function("apo_many_series_one_param_f32")
            .map_err(|e| CudaApoError::Cuda(e.to_string()))?;
        let block_x = match self.policy.many_series {
            ManySeriesKernelPolicy::Auto => 128u32,
            ManySeriesKernelPolicy::OneD { block_x } => block_x.max(32),
        };
        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut sp_i = short_period;
            let mut sa_f = short_alpha;
            let mut lp_i = long_period;
            let mut la_f = long_alpha;
            let mut ns_i = num_series as i32;
            let mut sl_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut sp_i as *mut _ as *mut c_void,
                &mut sa_f as *mut _ as *mut c_void,
                &mut lp_i as *mut _ as *mut c_void,
                &mut la_f as *mut _ as *mut c_void,
                &mut ns_i as *mut _ as *mut c_void,
                &mut sl_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaApoError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    // -------------------- Preparation helpers --------------------
    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &ApoBatchRange,
    ) -> Result<PreparedApoBatch, CudaApoError> {
        if data_f32.is_empty() {
            return Err(CudaApoError::InvalidInput("input data is empty".into()));
        }
        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaApoError::InvalidInput(
                "no valid parameter combinations".into(),
            ));
        }
        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaApoError::InvalidInput("all values are NaN".into()))?;

        // Validate sufficient valid samples for the longest period
        let max_long = combos
            .iter()
            .map(|c| c.long_period.unwrap_or(0))
            .max()
            .unwrap_or(0);
        if max_long == 0 || series_len - first_valid < max_long {
            return Err(CudaApoError::InvalidInput(format!(
                "not enough valid data: need >= {}, have {}",
                max_long,
                series_len - first_valid
            )));
        }

        let mut short_periods = Vec::with_capacity(combos.len());
        let mut long_periods = Vec::with_capacity(combos.len());
        let mut short_alphas = Vec::with_capacity(combos.len());
        let mut long_alphas = Vec::with_capacity(combos.len());
        for p in &combos {
            let s = p.short_period.unwrap_or(0);
            let l = p.long_period.unwrap_or(0);
            if s == 0 || l == 0 || s >= l {
                return Err(CudaApoError::InvalidInput("invalid short/long periods".into()));
            }
            short_periods.push(s as i32);
            long_periods.push(l as i32);
            short_alphas.push(2.0f32 / (s as f32 + 1.0f32));
            long_alphas.push(2.0f32 / (l as f32 + 1.0f32));
        }

        Ok(PreparedApoBatch {
            combos,
            first_valid,
            series_len,
            short_periods,
            short_alphas,
            long_periods,
            long_alphas,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &ApoParams,
    ) -> Result<(Vec<i32>, i32, i32, f32, f32), CudaApoError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaApoError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaApoError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }
        let sp = params.short_period.unwrap_or(0) as i32;
        let lp = params.long_period.unwrap_or(0) as i32;
        if sp <= 0 || lp <= 0 || sp >= lp {
            return Err(CudaApoError::InvalidInput(
                "invalid short/long period".into(),
            ));
        }
        let a_s = 2.0f32 / (sp as f32 + 1.0f32);
        let a_l = 2.0f32 / (lp as f32 + 1.0f32);

        let mut first_valids = Vec::with_capacity(num_series);
        for s in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + s];
                if v.is_finite() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv = fv
                .ok_or_else(|| CudaApoError::InvalidInput(format!("series {} all NaN", s)))?;
            let remaining = series_len - fv as usize;
            if remaining < lp as usize {
                return Err(CudaApoError::InvalidInput(format!(
                    "series {} not enough valid data (need >= {}, have {})",
                    s, lp, remaining
                )));
            }
            first_valids.push(fv);
        }

        Ok((first_valids, sp, lp, a_s, a_l))
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        match mem_get_info() {
            Ok((free, _)) => required_bytes.saturating_add(headroom_bytes) <= free,
            Err(_) => true,
        }
    }
}

struct PreparedApoBatch {
    combos: Vec<ApoParams>,
    first_valid: usize,
    series_len: usize,
    short_periods: Vec<i32>,
    short_alphas: Vec<f32>,
    long_periods: Vec<i32>,
    long_alphas: Vec<f32>,
}

fn expand_grid(r: &ApoBatchRange) -> Vec<ApoParams> {
    fn axis_u((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let shorts = axis_u(r.short);
    let longs = axis_u(r.long);
    let mut out = Vec::with_capacity(shorts.len() * longs.len());
    for &s in &shorts {
        for &l in &longs {
            if s > 0 && l > 0 && s < l {
                out.push(ApoParams { short_period: Some(s), long_period: Some(l) });
            }
        }
    }
    out
}

// -------------------- Benches --------------------
pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        apo_benches,
        CudaApo,
        crate::indicators::apo::ApoBatchRange,
        crate::indicators::apo::ApoParams,
        apo_batch_dev,
        apo_many_series_one_param_time_major_dev,
        crate::indicators::apo::ApoBatchRange { short: (5, 5 + PARAM_SWEEP - 1, 1), long: (20, 20 + PARAM_SWEEP - 1, 1) },
        crate::indicators::apo::ApoParams { short_period: Some(10), long_period: Some(20) },
        "apo",
        "apo"
    );
    pub use apo_benches::bench_profiles;
}

