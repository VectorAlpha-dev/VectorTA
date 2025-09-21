//! CUDA scaffolding for the Jurik Moving Average (JMA) kernels.
//!
//! The GPU implementation mirrors the scalar recurrence exactly: each parameter
//! sweep (one price series × many parameter combinations) and the many-series
//! variant execute sequentially per series/combo while the surrounding wrapper
//! keeps data resident on device for zero-copy workflows.

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use crate::indicators::moving_averages::jma::{expand_grid_jma, JmaBatchRange, JmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaJmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaJmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaJmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaJmaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaJmaError {}

pub struct CudaJma {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaJma {
    pub fn new(device_id: usize) -> Result<Self, CudaJmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaJmaError::Cuda(e.to_string()))?;

        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaJmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaJmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/jma_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaJmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn jma_batch_dev(
        &self,
        prices: &[f32],
        sweep: &JmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaJmaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        self.run_batch_kernel(prices, &inputs)
    }

    pub fn jma_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &JmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<JmaParams>), CudaJmaError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = inputs.series_len * inputs.combos.len();
        if out.len() != expected {
            return Err(CudaJmaError::InvalidInput(format!(
                "output slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(prices, &inputs)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, inputs.combos))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn jma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_alphas: &DeviceBuffer<f32>,
        d_one_minus_betas: &DeviceBuffer<f32>,
        d_phase_ratios: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJmaError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaJmaError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaJmaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_alphas,
            d_one_minus_betas,
            d_phase_ratios,
            series_len,
            n_combos,
            first_valid,
            d_out,
        )
    }

    pub fn jma_many_series_one_param_time_major_dev(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &JmaParams,
    ) -> Result<DeviceArrayF32, CudaJmaError> {
        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, params)?;
        let consts = Self::compute_params_consts(params)?;
        self.run_many_series_kernel(prices_tm_f32, cols, rows, &prepared, &consts)
    }

    pub fn jma_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &JmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaJmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaJmaError::InvalidInput(format!(
                "output slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }

        let prepared = Self::prepare_many_series_inputs(prices_tm_f32, cols, rows, params)?;
        let consts = Self::compute_params_consts(params)?;
        let arr = self.run_many_series_kernel(prices_tm_f32, cols, rows, &prepared, &consts)?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn jma_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        alpha: f32,
        one_minus_beta: f32,
        phase_ratio: f32,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJmaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaJmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if num_series > i32::MAX as usize || series_len > i32::MAX as usize {
            return Err(CudaJmaError::InvalidInput(
                "arguments exceed kernel limits".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            alpha,
            one_minus_beta,
            phase_ratio,
            num_series,
            series_len,
            d_first_valids,
            d_out_tm,
        )
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32, CudaJmaError> {
        let n_combos = inputs.combos.len();
        let series_len = inputs.series_len;

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let alpha_bytes = n_combos * std::mem::size_of::<f32>();
        let beta_bytes = n_combos * std::mem::size_of::<f32>();
        let phase_bytes = n_combos * std::mem::size_of::<f32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + alpha_bytes + beta_bytes + phase_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64 MB safety margin

        if !Self::will_fit(required, headroom) {
            return Err(CudaJmaError::InvalidInput(
                "insufficient device memory for JMA batch launch".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(prices).map_err(|e| CudaJmaError::Cuda(e.to_string()))?;
        let d_alphas = DeviceBuffer::from_slice(&inputs.alphas)
            .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;
        let d_one_minus_betas = DeviceBuffer::from_slice(&inputs.one_minus_betas)
            .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;
        let d_phase_ratios = DeviceBuffer::from_slice(&inputs.phase_ratios)
            .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }
                .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_alphas,
            &d_one_minus_betas,
            &d_phase_ratios,
            series_len,
            n_combos,
            inputs.first_valid,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn run_many_series_kernel(
        &self,
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        prepared: &ManySeriesInputs,
        consts: &JmaConsts,
    ) -> Result<DeviceArrayF32, CudaJmaError> {
        let num_series = cols;
        let series_len = rows;

        let prices_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let first_valid_bytes = prepared.first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes = prices_tm_f32.len() * std::mem::size_of::<f32>();
        let required = prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024; // 32 MB safety margin

        if !Self::will_fit(required, headroom) {
            return Err(CudaJmaError::InvalidInput(
                "insufficient device memory for JMA many-series launch".into(),
            ));
        }

        let d_prices_tm = DeviceBuffer::from_slice(prices_tm_f32)
            .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(prices_tm_f32.len()) }
                .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            consts.alpha,
            consts.one_minus_beta,
            consts.phase_ratio,
            num_series,
            series_len,
            &d_first_valids,
            &mut d_out_tm,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows: series_len,
            cols: num_series,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_alphas: &DeviceBuffer<f32>,
        d_one_minus_betas: &DeviceBuffer<f32>,
        d_phase_ratios: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJmaError> {
        let func = self
            .module
            .get_function("jma_batch_f32")
            .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut alphas_ptr = d_alphas.as_device_ptr().as_raw();
            let mut beta_ptr = d_one_minus_betas.as_device_ptr().as_raw();
            let mut phase_ptr = d_phase_ratios.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut alphas_ptr as *mut _ as *mut c_void,
                &mut beta_ptr as *mut _ as *mut c_void,
                &mut phase_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        alpha: f32,
        one_minus_beta: f32,
        phase_ratio: f32,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJmaError> {
        let func = self
            .module
            .get_function("jma_many_series_one_param_f32")
            .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;

        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut alpha_f = alpha;
            let mut one_minus_beta_f = one_minus_beta;
            let mut phase_ratio_f = phase_ratio;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
                &mut one_minus_beta_f as *mut _ as *mut c_void,
                &mut phase_ratio_f as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaJmaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &JmaBatchRange,
    ) -> Result<BatchInputs, CudaJmaError> {
        if prices.is_empty() {
            return Err(CudaJmaError::InvalidInput("empty price series".into()));
        }

        let combos = expand_grid_jma(sweep);
        if combos.is_empty() {
            return Err(CudaJmaError::InvalidInput(
                "no parameter combinations provided".into(),
            ));
        }

        let series_len = prices.len();
        let first_valid = prices
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaJmaError::InvalidInput("all price values are NaN".into()))?;

        let mut alphas = Vec::with_capacity(combos.len());
        let mut one_minus_betas = Vec::with_capacity(combos.len());
        let mut phase_ratios = Vec::with_capacity(combos.len());

        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            let phase = prm.phase.unwrap_or(50.0);
            let power = prm.power.unwrap_or(2);
            if period == 0 {
                return Err(CudaJmaError::InvalidInput("period must be positive".into()));
            }
            if period > i32::MAX as usize {
                return Err(CudaJmaError::InvalidInput(
                    "period exceeds kernel limits".into(),
                ));
            }
            if !phase.is_finite() {
                return Err(CudaJmaError::InvalidInput(format!(
                    "phase must be finite (got {})",
                    phase
                )));
            }
            let consts = Self::compute_consts(period, phase, power)?;
            alphas.push(consts.alpha);
            one_minus_betas.push(consts.one_minus_beta);
            phase_ratios.push(consts.phase_ratio);
            max_period = max_period.max(period);
        }

        if series_len - first_valid < max_period {
            return Err(CudaJmaError::InvalidInput(format!(
                "not enough valid data (needed >= {}, valid = {})",
                max_period,
                series_len - first_valid
            )));
        }

        Ok(BatchInputs {
            combos,
            alphas,
            one_minus_betas,
            phase_ratios,
            first_valid,
            series_len,
        })
    }

    fn prepare_many_series_inputs(
        prices_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &JmaParams,
    ) -> Result<ManySeriesInputs, CudaJmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaJmaError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        if prices_tm_f32.len() != cols * rows {
            return Err(CudaJmaError::InvalidInput("matrix shape mismatch".into()));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaJmaError::InvalidInput("period must be positive".into()));
        }
        if period > i32::MAX as usize {
            return Err(CudaJmaError::InvalidInput(
                "period exceeds kernel limits".into(),
            ));
        }
        let phase = params.phase.unwrap_or(50.0);
        if !phase.is_finite() {
            return Err(CudaJmaError::InvalidInput(format!(
                "phase must be finite (got {})",
                phase
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series_idx in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series_idx;
                if !prices_tm_f32[idx].is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let first = fv.ok_or_else(|| {
                CudaJmaError::InvalidInput(format!(
                    "series {} contains only NaN values",
                    series_idx
                ))
            })?;
            if rows - first < period {
                return Err(CudaJmaError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    series_idx,
                    period,
                    rows - first
                )));
            }
            first_valids[series_idx] = first as i32;
        }

        Ok(ManySeriesInputs { first_valids })
    }

    fn compute_params_consts(params: &JmaParams) -> Result<JmaConsts, CudaJmaError> {
        let period = params.period.unwrap_or(0);
        let phase = params.phase.unwrap_or(50.0);
        let power = params.power.unwrap_or(2);
        if period == 0 {
            return Err(CudaJmaError::InvalidInput("period must be positive".into()));
        }
        if !phase.is_finite() {
            return Err(CudaJmaError::InvalidInput(format!(
                "phase must be finite (got {})",
                phase
            )));
        }
        Self::compute_consts(period, phase, power)
    }

    fn compute_consts(period: usize, phase: f64, power: u32) -> Result<JmaConsts, CudaJmaError> {
        let phase_ratio = if phase < -100.0 {
            0.5
        } else if phase > 100.0 {
            2.5
        } else {
            phase / 100.0 + 1.5
        };

        let numerator = 0.45 * (period as f64 - 1.0);
        let denominator = numerator + 2.0;
        if denominator.abs() < f64::EPSILON {
            return Err(CudaJmaError::InvalidInput(
                "invalid period leading to zero denominator in beta".into(),
            ));
        }
        let beta = numerator / denominator;
        let alpha = beta.powi(power as i32);
        let one_minus_beta = 1.0 - beta;

        Ok(JmaConsts {
            alpha: alpha as f32,
            one_minus_beta: one_minus_beta as f32,
            phase_ratio: phase_ratio as f32,
        })
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
    fn device_mem_info() -> Option<(usize, usize)> {
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let res = cu::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
            if res == cu::CUresult::CUDA_SUCCESS {
                Some((free, total))
            } else {
                None
            }
        }
    }

    #[inline]
    fn will_fit(required_bytes: usize, headroom_bytes: usize) -> bool {
        if !Self::mem_check_enabled() {
            return true;
        }
        if let Some((free, _total)) = Self::device_mem_info() {
            required_bytes.saturating_add(headroom_bytes) <= free
        } else {
            true
        }
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        jma_benches,
        CudaJma,
        crate::indicators::moving_averages::jma::JmaBatchRange,
        crate::indicators::moving_averages::jma::JmaParams,
        jma_batch_dev,
        jma_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::jma::JmaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1), phase: (50.0, 50.0, 0.0), power: (2, 2, 0) },
        crate::indicators::moving_averages::jma::JmaParams { period: Some(64), phase: Some(50.0), power: Some(2) },
        "jma",
        "jma"
    );
    pub use jma_benches::bench_profiles;
}

struct BatchInputs {
    combos: Vec<JmaParams>,
    alphas: Vec<f32>,
    one_minus_betas: Vec<f32>,
    phase_ratios: Vec<f32>,
    first_valid: usize,
    series_len: usize,
}

struct ManySeriesInputs {
    first_valids: Vec<i32>,
}

struct JmaConsts {
    alpha: f32,
    one_minus_beta: f32,
    phase_ratio: f32,
}
