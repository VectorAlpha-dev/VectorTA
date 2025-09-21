//! CUDA support for the Square Root Weighted Moving Average (SRWMA).
//!
//! Provides zero-copy device entry points matching the ALMA CUDA API:
//!  * one-series × many-params batch execution, reusing precomputed
//!    square-root weights staged per combination;
//!  * time-major many-series × one-param execution that shares the same
//!    weights across series.
//!
//! Kernels operate purely in FP32 to keep VRAM usage compact while mirroring
//! the scalar CPU semantics (warm-up handling, NaN propagation).

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::srwma::{SrwmaBatchRange, SrwmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaSrwmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaSrwmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaSrwmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaSrwmaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaSrwmaError {}

pub struct CudaSrwma {
    module: Module,
    stream: Stream,
    _context: Context,
}

struct PreparedSrwmaBatch {
    combos: Vec<SrwmaParams>,
    first_valid: usize,
    series_len: usize,
    max_wlen: usize,
    periods_i32: Vec<i32>,
    warm_indices: Vec<i32>,
    inv_norms: Vec<f32>,
    weights_flat: Vec<f32>,
}

struct PreparedSrwmaManySeries {
    first_valids: Vec<i32>,
    period: usize,
    weights: Vec<f32>,
    inv_norm: f32,
}

impl CudaSrwma {
    pub fn new(device_id: usize) -> Result<Self, CudaSrwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/srwma_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn srwma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SrwmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaSrwmaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&prepared.weights_flat)
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&prepared.periods_i32)
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let d_warm = DeviceBuffer::from_slice(&prepared.warm_indices)
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let d_inv = DeviceBuffer::from_slice(&prepared.inv_norms)
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_weights,
            &d_periods,
            &d_warm,
            &d_inv,
            prepared.series_len,
            prepared.max_wlen,
            n_combos,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn srwma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights_flat: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warm_indices: &DeviceBuffer<i32>,
        d_inv_norms: &DeviceBuffer<f32>,
        series_len: usize,
        _first_valid: usize,
        max_wlen: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSrwmaError> {
        if series_len == 0 {
            return Err(CudaSrwmaError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if n_combos == 0 {
            return Err(CudaSrwmaError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if max_wlen == 0 {
            return Err(CudaSrwmaError::InvalidInput(
                "max_wlen must be positive".into(),
            ));
        }
        if d_periods.len() != n_combos
            || d_warm_indices.len() != n_combos
            || d_inv_norms.len() != n_combos
        {
            return Err(CudaSrwmaError::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }
        if d_weights_flat.len() != n_combos * max_wlen {
            return Err(CudaSrwmaError::InvalidInput(
                "weights buffer must be combos * max_wlen".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaSrwmaError::InvalidInput(
                "prices buffer length must equal series_len".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaSrwmaError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_weights_flat,
            d_periods,
            d_warm_indices,
            d_inv_norms,
            series_len,
            max_wlen,
            n_combos,
            d_out,
        )
    }

    pub fn srwma_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &SrwmaParams,
    ) -> Result<DeviceArrayF32, CudaSrwmaError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&prepared.weights)
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)
                .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first_valids,
            &d_weights,
            prepared.period,
            prepared.inv_norm,
            num_series,
            series_len,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: series_len,
            cols: num_series,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn srwma_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        inv_norm: f32,
        num_series: i32,
        series_len: i32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSrwmaError> {
        if period <= 1 {
            return Err(CudaSrwmaError::InvalidInput("period must be >= 2".into()));
        }
        if num_series <= 0 || series_len <= 0 {
            return Err(CudaSrwmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if d_weights.len() != (period as usize - 1) {
            return Err(CudaSrwmaError::InvalidInput(
                "weights length must equal period - 1".into(),
            ));
        }
        if d_first_valids.len() != num_series as usize {
            return Err(CudaSrwmaError::InvalidInput(
                "first_valids length mismatch".into(),
            ));
        }
        if d_prices_tm.len() != (num_series as usize * series_len as usize)
            || d_out_tm.len() != (num_series as usize * series_len as usize)
        {
            return Err(CudaSrwmaError::InvalidInput(
                "time-major buffer length mismatch".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            d_weights,
            period as usize,
            inv_norm,
            num_series as usize,
            series_len as usize,
            d_out_tm,
        )
    }

    pub fn srwma_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &SrwmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaSrwmaError> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaSrwmaError::InvalidInput(
                "output slice wrong length".into(),
            ));
        }
        let handle = self.srwma_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        handle
            .buf
            .copy_to(out_tm)
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights_flat: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warm_indices: &DeviceBuffer<i32>,
        d_inv_norms: &DeviceBuffer<f32>,
        series_len: usize,
        max_wlen: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSrwmaError> {
        let block: BlockSize = (128, 1, 1).into();
        let grid_x = ((series_len as u32) + block.x - 1) / block.x;
        let grid: GridSize = (grid_x.max(1), n_combos as u32, 1).into();
        let shared_bytes = (max_wlen * std::mem::size_of::<f32>()) as u32;

        let func = self
            .module
            .get_function("srwma_batch_f32")
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights_flat.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut warm_ptr = d_warm_indices.as_device_ptr().as_raw();
            let mut inv_ptr = d_inv_norms.as_device_ptr().as_raw();
            let mut max_wlen_i = max_wlen as i32;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 9] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut warm_ptr as *mut _ as *mut c_void,
                &mut inv_ptr as *mut _ as *mut c_void,
                &mut max_wlen_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, &mut args)
                .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        d_weights: &DeviceBuffer<f32>,
        period: usize,
        inv_norm: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSrwmaError> {
        let block: BlockSize = (128, 1, 1).into();
        let grid_x = ((series_len as u32) + block.x - 1) / block.x;
        let grid: GridSize = (grid_x.max(1), num_series as u32, 1).into();
        let shared_bytes = ((period - 1) * std::mem::size_of::<f32>()) as u32;

        let func = self
            .module
            .get_function("srwma_many_series_one_param_f32")
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut inv = inv_norm;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 8] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut inv as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, &mut args)
                .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaSrwmaError::Cuda(e.to_string()))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SrwmaBatchRange,
    ) -> Result<PreparedSrwmaBatch, CudaSrwmaError> {
        if data_f32.is_empty() {
            return Err(CudaSrwmaError::InvalidInput("input data is empty".into()));
        }
        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaSrwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }
        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaSrwmaError::InvalidInput("all values are NaN".into()))?;

        let mut max_wlen = 0usize;
        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period < 2 {
                return Err(CudaSrwmaError::InvalidInput(format!(
                    "invalid period {} (must be >= 2)",
                    period
                )));
            }
            if series_len - first_valid < period + 1 {
                return Err(CudaSrwmaError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    period + 1,
                    series_len - first_valid
                )));
            }
            max_wlen = max_wlen.max(period - 1);
        }

        let n_combos = combos.len();
        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut warm_indices = Vec::with_capacity(n_combos);
        let mut inv_norms = Vec::with_capacity(n_combos);
        let mut weights_flat = vec![0f32; n_combos * max_wlen];

        for (idx, params) in combos.iter().enumerate() {
            let period = params.period.unwrap();
            let wlen = period - 1;
            let mut norm = 0f32;
            for k in 0..wlen {
                let weight = ((period - k) as f32).sqrt();
                weights_flat[idx * max_wlen + k] = weight;
                norm += weight;
            }
            if norm <= 0.0 {
                return Err(CudaSrwmaError::InvalidInput(format!(
                    "period {} produced non-positive norm",
                    period
                )));
            }
            periods_i32.push(period as i32);
            warm_indices.push((first_valid + period + 1) as i32);
            inv_norms.push(1.0f32 / norm);
        }

        Ok(PreparedSrwmaBatch {
            combos,
            first_valid,
            series_len,
            max_wlen,
            periods_i32,
            warm_indices,
            inv_norms,
            weights_flat,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &SrwmaParams,
    ) -> Result<PreparedSrwmaManySeries, CudaSrwmaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaSrwmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaSrwmaError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }
        let period = params.period.unwrap_or(14);
        if period < 2 {
            return Err(CudaSrwmaError::InvalidInput(format!(
                "invalid period {} (must be >= 2)",
                period
            )));
        }
        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + series];
                if v.is_finite() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaSrwmaError::InvalidInput(format!("series {} all NaN", series))
            })?;
            if series_len - fv < period + 1 {
                return Err(CudaSrwmaError::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    series,
                    period + 1,
                    series_len - fv
                )));
            }
            first_valids.push(fv as i32);
        }

        let wlen = period - 1;
        let mut weights = Vec::with_capacity(wlen);
        let mut norm = 0f32;
        for k in 0..wlen {
            let weight = ((period - k) as f32).sqrt();
            weights.push(weight);
            norm += weight;
        }
        if norm <= 0.0 {
            return Err(CudaSrwmaError::InvalidInput(
                "computed weight norm <= 0".into(),
            ));
        }
        let inv_norm = 1.0f32 / norm;

        Ok(PreparedSrwmaManySeries {
            first_valids,
            period,
            weights,
            inv_norm,
        })
    }
}

fn expand_grid(range: &SrwmaBatchRange) -> Vec<SrwmaParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis(range.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(SrwmaParams { period: Some(p) });
    }
    out
}
