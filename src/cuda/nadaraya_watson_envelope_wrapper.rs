//! CUDA wrapper for Nadaraya–Watson Envelope (NWE).
//!
//! API mirrors ALMA/CWMA-style wrappers:
//! - PTX load with DetermineTargetFromContext + O2 fallback
//! - Non-blocking stream
//! - VRAM estimates and guardrails
//! - Batch: one series × many params → returns upper/lower device arrays
//! - Many series × one param (time-major) → returns upper/lower device arrays

#![cfg(feature = "cuda")]

use crate::cuda::moving_averages::DeviceArrayF32;
use crate::indicators::nadaraya_watson_envelope::{NweBatchRange, NweParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{Module, ModuleJitOption, OptLevel};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaNweError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaNweError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaNweError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaNweError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}
impl std::error::Error for CudaNweError {}

pub struct DeviceNwePair {
    pub upper: DeviceArrayF32,
    pub lower: DeviceArrayF32,
}

impl DeviceNwePair {
    #[inline]
    pub fn rows(&self) -> usize {
        self.upper.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.upper.cols
    }
}

pub struct CudaNwe {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaNwe {
    pub fn new(device_id: usize) -> Result<Self, CudaNweError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        let ptx = include_str!(concat!(
            env!("OUT_DIR"),
            "/nadaraya_watson_envelope_kernel.ptx"
        ));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => match Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext]) {
                Ok(m) => m,
                Err(_) => {
                    Module::from_ptx(ptx, &[]).map_err(|e| CudaNweError::Cuda(e.to_string()))?
                }
            },
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn will_fit(required: usize, headroom: usize) -> bool {
        if let Ok((free, _)) = mem_get_info() {
            required.saturating_add(headroom) <= free
        } else {
            true
        }
    }

    fn expand_grid(r: &NweBatchRange) -> Vec<NweParams> {
        let mut bw = Vec::new();
        let mut m = Vec::new();
        let mut lb = Vec::new();
        // bandwidth
        if (r.bandwidth.2.abs() < 1e-12) || (r.bandwidth.0 == r.bandwidth.1) {
            bw.push(r.bandwidth.0);
        } else {
            let mut v = r.bandwidth.0;
            while v <= r.bandwidth.1 + 1e-12 {
                bw.push(v);
                v += r.bandwidth.2.max(1e-12);
            }
        }
        // multiplier
        if (r.multiplier.2.abs() < 1e-12) || (r.multiplier.0 == r.multiplier.1) {
            m.push(r.multiplier.0);
        } else {
            let mut v = r.multiplier.0;
            while v <= r.multiplier.1 + 1e-12 {
                m.push(v);
                v += r.multiplier.2.max(1e-12);
            }
        }
        // lookback
        let step_lb = r.lookback.2.max(1);
        for v in (r.lookback.0..=r.lookback.1).step_by(step_lb) {
            lb.push(v);
        }
        let mut out = Vec::with_capacity(bw.len() * m.len() * lb.len());
        for &b in &bw {
            for &mm in &m {
                for &l in &lb {
                    out.push(NweParams {
                        bandwidth: Some(b),
                        multiplier: Some(mm),
                        lookback: Some(l),
                    });
                }
            }
        }
        out
    }

    fn compute_weights_row(bandwidth: f64, lookback: usize) -> (Vec<f32>, usize) {
        let mut w = Vec::with_capacity(lookback);
        let mut den = 0.0f64;
        for k in 0..lookback {
            let wk = (-(k as f64) * (k as f64) / (2.0 * bandwidth * bandwidth)).exp();
            w.push(wk as f32);
            den += wk;
        }
        let inv_den = if den != 0.0 {
            1.0f32 / (den as f32)
        } else {
            0.0f32
        };
        for x in &mut w {
            *x *= inv_den;
        }
        (w, lookback)
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &NweBatchRange,
    ) -> Result<
        (
            Vec<NweParams>,
            usize,
            usize,
            Vec<i32>,
            Vec<f32>,
            Vec<f32>,
            usize,
        ),
        CudaNweError,
    > {
        if prices.is_empty() {
            return Err(CudaNweError::InvalidInput("empty series".into()));
        }
        let len = prices.len();
        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaNweError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaNweError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let mut lookbacks = Vec::with_capacity(combos.len());
        let mut multipliers = Vec::with_capacity(combos.len());
        let mut max_lb = 1usize;
        for prm in &combos {
            let lb = prm.lookback.unwrap_or(500);
            if lb == 0 {
                return Err(CudaNweError::InvalidInput("lookback must be > 0".into()));
            }
            if len - first_valid < lb {
                return Err(CudaNweError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    lb,
                    len - first_valid
                )));
            }
            lookbacks.push(lb as i32);
            multipliers.push(prm.multiplier.unwrap_or(3.0) as f32);
            max_lb = max_lb.max(lb);
        }

        let mut weights_flat = vec![0f32; combos.len() * max_lb];
        for (row, prm) in combos.iter().enumerate() {
            let lb = prm.lookback.unwrap_or(500);
            let (row_w, _l) = Self::compute_weights_row(prm.bandwidth.unwrap_or(8.0), lb);
            let base = row * max_lb;
            weights_flat[base..base + lb].copy_from_slice(&row_w);
        }

        Ok((
            combos,
            first_valid,
            len,
            lookbacks,
            multipliers,
            weights_flat,
            max_lb,
        ))
    }

    pub fn nwe_batch_dev(
        &self,
        prices: &[f32],
        sweep: &NweBatchRange,
    ) -> Result<(DeviceNwePair, Vec<NweParams>), CudaNweError> {
        let (combos, first_valid, len, lookbacks, multipliers, weights_flat, max_lb) =
            Self::prepare_batch_inputs(prices, sweep)?;
        let n = combos.len();

        // VRAM estimate (2 outputs)
        let required = len * std::mem::size_of::<f32>()
            + n * max_lb * std::mem::size_of::<f32>()
            + n * std::mem::size_of::<i32>()
            + n * std::mem::size_of::<f32>()
            + 2 * n * len * std::mem::size_of::<f32>();
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaNweError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(prices).map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat)
            .map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        let d_looks =
            DeviceBuffer::from_slice(&lookbacks).map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        let d_mults = DeviceBuffer::from_slice(&multipliers)
            .map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n * len) }
            .map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n * len) }
            .map_err(|e| CudaNweError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("nadaraya_watson_envelope_batch_f32")
            .map_err(|e| CudaNweError::Cuda(e.to_string()))?;

        // Single-thread per combo; grid.y = n
        let grid = GridSize::xy(1, n as u32);
        let block = BlockSize::xyz(1, 1, 1);

        unsafe {
            let mut prices_p = d_prices.as_device_ptr().as_raw();
            let mut weights_p = d_weights.as_device_ptr().as_raw();
            let mut looks_p = d_looks.as_device_ptr().as_raw();
            let mut mults_p = d_mults.as_device_ptr().as_raw();
            let mut len_i = len as i32;
            let mut n_i = n as i32;
            let mut fv_i = first_valid as i32;
            let mut max_lb_i = max_lb as i32;
            let mut upper_p = d_upper.as_device_ptr().as_raw();
            let mut lower_p = d_lower.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_p as *mut _ as *mut c_void,
                &mut weights_p as *mut _ as *mut c_void,
                &mut looks_p as *mut _ as *mut c_void,
                &mut mults_p as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut fv_i as *mut _ as *mut c_void,
                &mut max_lb_i as *mut _ as *mut c_void,
                &mut upper_p as *mut _ as *mut c_void,
                &mut lower_p as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaNweError::Cuda(e.to_string()))?;

        let pair = DeviceNwePair {
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows: n,
                cols: len,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows: n,
                cols: len,
            },
        };
        Ok((pair, combos))
    }

    pub fn nwe_many_series_one_param_time_major_dev(
        &self,
        data_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &NweParams,
    ) -> Result<DeviceNwePair, CudaNweError> {
        if cols == 0 || rows == 0 {
            return Err(CudaNweError::InvalidInput("empty matrix".into()));
        }
        if data_tm.len() != cols * rows {
            return Err(CudaNweError::InvalidInput(
                "matrix shape mismatch (time-major)".into(),
            ));
        }
        let bandwidth = params.bandwidth.unwrap_or(8.0);
        let lookback = params.lookback.unwrap_or(500);
        let multiplier = params.multiplier.unwrap_or(3.0) as f32;
        if lookback == 0 {
            return Err(CudaNweError::InvalidInput("lookback must be > 0".into()));
        }

        // first_valid per series
        let mut first_valids = vec![rows as i32; cols];
        for s in 0..cols {
            for t in 0..rows {
                let v = data_tm[t * cols + s];
                if !v.is_nan() {
                    first_valids[s] = t as i32;
                    break;
                }
            }
        }
        // weights pre-scaled
        let (w_row, _l) = Self::compute_weights_row(bandwidth, lookback);

        let required = data_tm.len() * std::mem::size_of::<f32>()
            + w_row.len() * std::mem::size_of::<f32>()
            + first_valids.len() * std::mem::size_of::<i32>()
            + 2 * data_tm.len() * std::mem::size_of::<f32>();
        if !Self::will_fit(required, 64 * 1024 * 1024) {
            return Err(CudaNweError::InvalidInput(
                "insufficient device memory".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(data_tm).map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(&w_row).map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        let mut d_upper: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        let mut d_lower: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaNweError::Cuda(e.to_string()))?;

        let func = self
            .module
            .get_function("nadaraya_watson_envelope_many_series_one_param_f32")
            .map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        let grid = GridSize::xy(1, cols as u32);
        let block = BlockSize::xyz(1, 1, 1);

        unsafe {
            let mut prices_p = d_prices.as_device_ptr().as_raw();
            let mut weights_p = d_weights.as_device_ptr().as_raw();
            let mut lookback_i = lookback as i32;
            let mut mult_f = multiplier;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut first_p = d_first.as_device_ptr().as_raw();
            let mut out_u_p = d_upper.as_device_ptr().as_raw();
            let mut out_l_p = d_lower.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_p as *mut _ as *mut c_void,
                &mut weights_p as *mut _ as *mut c_void,
                &mut lookback_i as *mut _ as *mut c_void,
                &mut mult_f as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_p as *mut _ as *mut c_void,
                &mut out_u_p as *mut _ as *mut c_void,
                &mut out_l_p as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaNweError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaNweError::Cuda(e.to_string()))?;

        Ok(DeviceNwePair {
            upper: DeviceArrayF32 {
                buf: d_upper,
                rows,
                cols,
            },
            lower: DeviceArrayF32 {
                buf: d_lower,
                rows,
                cols,
            },
        })
    }
}

// -------- Benches registration (lightweight) --------
#[cfg(feature = "cuda")]
pub mod benches {
    use super::*;
    use crate::cuda::bench::helpers::{gen_series, gen_time_major_prices};
    use crate::cuda::bench::{CudaBenchScenario, CudaBenchState};

    const ONE_SERIES_LEN: usize = 1_000_000;
    const MANY_SERIES_COLS: usize = 256;
    const MANY_SERIES_LEN: usize = 1_000_000;

    pub fn bench_profiles() -> Vec<CudaBenchScenario> {
        // Batch
        struct BatchState {
            cuda: CudaNwe,
            price: Vec<f32>,
            sweep: NweBatchRange,
        }
        impl CudaBenchState for BatchState {
            fn launch(&mut self) {
                let _ = self.cuda.nwe_batch_dev(&self.price, &self.sweep);
            }
        }
        let prep_batch = || {
            let cuda = CudaNwe::new(0).expect("cuda");
            let price = gen_series(ONE_SERIES_LEN);
            let sweep = NweBatchRange {
                bandwidth: (6.0, 12.0, 2.0),
                multiplier: (2.0, 3.0, 0.5),
                lookback: (128, 512, 64),
            };
            Box::new(BatchState { cuda, price, sweep }) as Box<dyn CudaBenchState>
        };
        // Many-series
        struct ManyState {
            cuda: CudaNwe,
            data_tm: Vec<f32>,
            cols: usize,
            rows: usize,
            params: NweParams,
        }
        impl CudaBenchState for ManyState {
            fn launch(&mut self) {
                let _ = self.cuda.nwe_many_series_one_param_time_major_dev(
                    &self.data_tm,
                    self.cols,
                    self.rows,
                    &self.params,
                );
            }
        }
        let prep_many = || {
            let cuda = CudaNwe::new(0).expect("cuda");
            let cols = MANY_SERIES_COLS;
            let rows = MANY_SERIES_LEN;
            let data_tm = gen_time_major_prices(cols, rows);
            let params = NweParams {
                bandwidth: Some(8.0),
                multiplier: Some(3.0),
                lookback: Some(256),
            };
            Box::new(ManyState {
                cuda,
                data_tm,
                cols,
                rows,
                params,
            }) as Box<dyn CudaBenchState>
        };
        let bytes_batch = ONE_SERIES_LEN * std::mem::size_of::<f32>()
            + (ONE_SERIES_LEN * 256) * std::mem::size_of::<f32>()
            + 64 * 1024 * 1024;
        let bytes_many =
            MANY_SERIES_COLS * MANY_SERIES_LEN * 3 * std::mem::size_of::<f32>() + 64 * 1024 * 1024;
        vec![
            CudaBenchScenario::new(
                "nwe",
                "one_series_many_params",
                "nwe_cuda_batch_dev",
                "1m_x_grid",
                prep_batch,
            )
            .with_sample_size(10)
            .with_mem_required(bytes_batch),
            CudaBenchScenario::new(
                "nwe",
                "many_series_one_param",
                "nwe_cuda_many_series_one_param",
                "256x1m",
                prep_many,
            )
            .with_sample_size(5)
            .with_mem_required(bytes_many),
        ]
    }
}
