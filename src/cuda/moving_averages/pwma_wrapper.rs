//! CUDA wrapper for PWMA (Pascal Weighted Moving Average) kernels.
//!
//! Mirrors the ALMA/SWMA scaffold: validate host inputs, upload FP32 data and
//! Pascal weights once, then launch kernels that keep the dot products entirely
//! on device. Supports both the single-series × many-parameter sweep and the
//! many-series × one-parameter path operating on time-major inputs.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::pwma::{expand_grid, PwmaBatchRange, PwmaParams};
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
pub enum CudaPwmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaPwmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaPwmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaPwmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaPwmaError {}

pub struct CudaPwma {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaPwma {
    pub fn new(device_id: usize) -> Result<Self, CudaPwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/pwma_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn pascal_weights_f32(period: usize) -> Result<Vec<f32>, CudaPwmaError> {
        if period == 0 {
            return Err(CudaPwmaError::InvalidInput(
                "period must be greater than zero".into(),
            ));
        }
        let n = period - 1;
        let mut row = Vec::with_capacity(period);
        let mut sum = 0.0f64;
        for r in 0..=n {
            let mut val = 1.0f64;
            for i in 0..r {
                val *= (n - i) as f64;
                val /= (i + 1) as f64;
            }
            row.push(val);
            sum += val;
        }
        if sum == 0.0 {
            return Err(CudaPwmaError::InvalidInput(format!(
                "Pascal weights sum to zero for period {}",
                period
            )));
        }
        let inv = 1.0 / sum;
        Ok(row.into_iter().map(|v| (v * inv) as f32).collect())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &PwmaBatchRange,
    ) -> Result<(Vec<PwmaParams>, usize, usize, usize, Vec<f32>), CudaPwmaError> {
        if data_f32.is_empty() {
            return Err(CudaPwmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaPwmaError::InvalidInput("all values are NaN".into()))?;
        let len = data_f32.len();

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaPwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaPwmaError::InvalidInput("period must be > 0".into()));
            }
            if period > len {
                return Err(CudaPwmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaPwmaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
            if period > max_period {
                max_period = period;
            }
        }

        let n_combos = combos.len();
        let mut weights_flat = vec![0.0f32; n_combos * max_period];
        for (row, prm) in combos.iter().enumerate() {
            let weights = Self::pascal_weights_f32(prm.period.unwrap())?;
            let base = row * max_period;
            for (idx, w) in weights.iter().enumerate() {
                weights_flat[base + idx] = *w;
            }
        }

        Ok((combos, first_valid, len, max_period, weights_flat))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPwmaError> {
        if series_len == 0 || n_combos == 0 || max_period == 0 {
            return Err(CudaPwmaError::InvalidInput(
                "series_len, n_combos, and max_period must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaPwmaError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("pwma_batch_f32")
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        let block_x: u32 = 256;
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut warms_ptr = d_warms.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut max_period_i = max_period as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut warms_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut max_period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    pub fn pwma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPwmaError> {
        self.launch_batch_kernel(
            d_prices, d_weights, d_periods, d_warms, series_len, n_combos, max_period, d_out,
        )
    }

    pub fn pwma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &PwmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaPwmaError> {
        let (combos, first_valid, series_len, max_period, weights_flat) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let warms_i32: Vec<i32> = combos
            .iter()
            .map(|p| (first_valid + p.period.unwrap() - 1) as i32)
            .collect();

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat)
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let d_warms =
            DeviceBuffer::from_slice(&warms_i32).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices, &d_weights, &d_periods, &d_warms, series_len, n_combos, max_period,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

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
        params: &PwmaParams,
    ) -> Result<(Vec<i32>, Vec<f32>, usize), CudaPwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaPwmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaPwmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(CudaPwmaError::InvalidInput("period must be > 0".into()));
        }
        if period > rows {
            return Err(CudaPwmaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }

        let weights = Self::pascal_weights_f32(period)?;

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let idx = row * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    found = Some(row);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaPwmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < period {
                return Err(CudaPwmaError::InvalidInput(format!(
                    "series {} lacks enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv
                )));
            }
            if fv > i32::MAX as usize {
                return Err(CudaPwmaError::InvalidInput(
                    "first_valid exceeds i32::MAX".into(),
                ));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, weights, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPwmaError> {
        if period == 0 || num_series == 0 || series_len == 0 {
            return Err(CudaPwmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaPwmaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("pwma_multi_series_one_param_f32")
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        const BLOCK_X: u32 = 128;
        let grid_x = ((series_len as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x.max(1), num_series as u32, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();
        let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    pub fn pwma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaPwmaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaPwmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            d_weights,
            d_first_valids,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_out_tm,
        )
    }

    pub fn pwma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &PwmaParams,
    ) -> Result<DeviceArrayF32, CudaPwmaError> {
        let (first_valids, weights, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(&weights).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_weights,
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn pwma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &PwmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaPwmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaPwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, weights, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(&weights).map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_weights,
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;

        d_out_tm
            .copy_to(out_tm)
            .map_err(|e| CudaPwmaError::Cuda(e.to_string()))?;
        Ok(())
    }
}
