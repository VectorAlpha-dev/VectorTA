//! CUDA support for the Midway Weighted Exponential (MWDX) indicator.
//!
//! Mirrors the ALMA CUDA API surface by exposing zero-copy device entry points
//! for both batching a single series across many factor values and processing a
//! time-major matrix of many series with a shared factor. Kernels operate in
//! FP32, while the host wrapper handles validation and staging of parameters.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::mwdx::{expand_grid_mwdx, MwdxBatchRange, MwdxParams};
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
pub enum CudaMwdxError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaMwdxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaMwdxError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaMwdxError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaMwdxError {}

pub struct CudaMwdx {
    module: Module,
    stream: Stream,
    _context: Context,
}

struct PreparedMwdxBatch {
    combos: Vec<MwdxParams>,
    first_valid: usize,
    series_len: usize,
    factors_f32: Vec<f32>,
}

struct PreparedMwdxManySeries {
    first_valids: Vec<i32>,
    factor: f32,
}

impl CudaMwdx {
    pub fn new(device_id: usize) -> Result<Self, CudaMwdxError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaMwdxError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaMwdxError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaMwdxError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/mwdx_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaMwdxError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaMwdxError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn mwdx_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &MwdxBatchRange,
    ) -> Result<DeviceArrayF32, CudaMwdxError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaMwdxError::Cuda(e.to_string()))?;
        let d_factors = DeviceBuffer::from_slice(&prepared.factors_f32)
            .map_err(|e| CudaMwdxError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaMwdxError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_factors,
            prepared.series_len,
            prepared.first_valid,
            n_combos,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    pub fn mwdx_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_factors: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMwdxError> {
        if series_len == 0 {
            return Err(CudaMwdxError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if n_combos == 0 {
            return Err(CudaMwdxError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaMwdxError::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, series_len
            )));
        }
        if d_prices.len() != series_len {
            return Err(CudaMwdxError::InvalidInput(
                "prices buffer length mismatch".into(),
            ));
        }
        if d_factors.len() != n_combos {
            return Err(CudaMwdxError::InvalidInput(
                "factors buffer length mismatch".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaMwdxError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_factors,
            series_len,
            first_valid,
            n_combos,
            d_out,
        )
    }

    pub fn mwdx_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &MwdxBatchRange,
        out_flat: &mut [f32],
    ) -> Result<(), CudaMwdxError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        if out_flat.len() != prepared.series_len * prepared.combos.len() {
            return Err(CudaMwdxError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.mwdx_batch_dev(data_f32, sweep)?;
        handle
            .buf
            .copy_to(out_flat)
            .map_err(|e| CudaMwdxError::Cuda(e.to_string()))
    }

    pub fn mwdx_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &MwdxParams,
    ) -> Result<DeviceArrayF32, CudaMwdxError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaMwdxError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaMwdxError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)
                .map_err(|e| CudaMwdxError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first_valids,
            prepared.factor,
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

    pub fn mwdx_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        factor: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMwdxError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaMwdxError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if d_prices_tm.len() != num_series * series_len {
            return Err(CudaMwdxError::InvalidInput(
                "time-major prices length mismatch".into(),
            ));
        }
        if d_first_valids.len() != num_series {
            return Err(CudaMwdxError::InvalidInput(
                "first_valids length must equal num_series".into(),
            ));
        }
        if d_out_tm.len() != num_series * series_len {
            return Err(CudaMwdxError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }
        if !factor.is_finite() || factor <= 0.0 {
            return Err(CudaMwdxError::InvalidInput(
                "factor must be positive and finite".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            factor,
            num_series,
            series_len,
            d_out_tm,
        )
    }

    pub fn mwdx_many_series_one_param_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &MwdxParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaMwdxError> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaMwdxError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.mwdx_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        handle
            .buf
            .copy_to(out_tm)
            .map_err(|e| CudaMwdxError::Cuda(e.to_string()))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_factors: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMwdxError> {
        if n_combos == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("mwdx_batch_f32")
            .map_err(|e| CudaMwdxError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (256, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut facs_ptr = d_factors.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 6] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut facs_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaMwdxError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaMwdxError::Cuda(e.to_string()))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        factor: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaMwdxError> {
        let func = self
            .module
            .get_function("mwdx_many_series_one_param_f32")
            .map_err(|e| CudaMwdxError::Cuda(e.to_string()))?;

        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (256, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut factor_f32 = factor;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 6] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut factor_f32 as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaMwdxError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaMwdxError::Cuda(e.to_string()))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &MwdxBatchRange,
    ) -> Result<PreparedMwdxBatch, CudaMwdxError> {
        if data_f32.is_empty() {
            return Err(CudaMwdxError::InvalidInput("input data is empty".into()));
        }

        let combos = expand_grid_mwdx(sweep);
        if combos.is_empty() {
            return Err(CudaMwdxError::InvalidInput(
                "no factor combinations provided".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaMwdxError::InvalidInput("all values are NaN".into()))?;
        if first_valid >= series_len {
            return Err(CudaMwdxError::InvalidInput(
                "first_valid index equals series length".into(),
            ));
        }

        let mut factors_f32 = Vec::with_capacity(combos.len());
        for params in &combos {
            let factor = params.factor.unwrap_or(0.2);
            if !factor.is_finite() || factor <= 0.0 {
                return Err(CudaMwdxError::InvalidInput(format!(
                    "factor must be positive and finite (got {})",
                    factor
                )));
            }
            let val2 = (2.0 / factor) - 1.0;
            if val2 + 1.0 <= 0.0 {
                return Err(CudaMwdxError::InvalidInput(format!(
                    "invalid denominator for factor {}",
                    factor
                )));
            }
            factors_f32.push(factor as f32);
        }

        Ok(PreparedMwdxBatch {
            combos,
            first_valid,
            series_len,
            factors_f32,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &MwdxParams,
    ) -> Result<PreparedMwdxManySeries, CudaMwdxError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaMwdxError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaMwdxError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }

        let factor = params.factor.unwrap_or(0.2);
        if !factor.is_finite() || factor <= 0.0 {
            return Err(CudaMwdxError::InvalidInput(
                "factor must be positive and finite".into(),
            ));
        }
        let val2 = (2.0 / factor) - 1.0;
        if val2 + 1.0 <= 0.0 {
            return Err(CudaMwdxError::InvalidInput(
                "invalid denominator derived from factor".into(),
            ));
        }

        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut first = None;
            for t in 0..series_len {
                let value = data_tm_f32[t * num_series + series];
                if value.is_finite() {
                    first = Some(t as i32);
                    break;
                }
            }
            let idx = first.ok_or_else(|| {
                CudaMwdxError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            first_valids.push(idx);
        }

        Ok(PreparedMwdxManySeries {
            first_valids,
            factor: factor as f32,
        })
    }
}
