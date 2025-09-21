//! CUDA wrapper for SWMA (Symmetric Weighted Moving Average) kernels.
//!
//! Matches the ALMA/TRIMA scaffolding: validate host inputs, upload FP32
//! buffers, and launch device kernels that precompute the symmetric weights in
//! shared memory.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::swma::{SwmaBatchRange, SwmaParams};
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
pub enum CudaSwmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaSwmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaSwmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaSwmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaSwmaError {}

pub struct CudaSwma {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaSwma {
    pub fn new(device_id: usize) -> Result<Self, CudaSwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/swma_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn expand_periods(range: &SwmaBatchRange) -> Vec<usize> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return vec![start];
        }
        if start > end {
            return Vec::new();
        }
        (start..=end).step_by(step).collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &SwmaBatchRange,
    ) -> Result<(Vec<usize>, usize), CudaSwmaError> {
        if data_f32.is_empty() {
            return Err(CudaSwmaError::InvalidInput("empty data".into()));
        }

        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaSwmaError::InvalidInput("all values are NaN".into()))?;

        let periods = Self::expand_periods(sweep);
        if periods.is_empty() {
            return Err(CudaSwmaError::InvalidInput("no periods in sweep".into()));
        }

        let len = data_f32.len();
        for &period in &periods {
            if period == 0 {
                return Err(CudaSwmaError::InvalidInput("period must be > 0".into()));
            }
            if period > len {
                return Err(CudaSwmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaSwmaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
        }

        Ok((periods, first_valid))
    }

    fn compute_weights(period: usize) -> Vec<f32> {
        let mut weights = vec![0.0f32; period];
        if period == 0 {
            return weights;
        }
        let norm = if period <= 2 {
            period as f32
        } else if period % 2 == 0 {
            let half = (period / 2) as f32;
            half * (half + 1.0f32)
        } else {
            let half_plus = ((period + 1) / 2) as f32;
            half_plus * half_plus
        };
        let inv_norm = 1.0f32 / norm.max(f32::EPSILON);
        for idx in 0..period {
            let left = idx + 1;
            let right = period - idx;
            let w = left.min(right) as f32;
            weights[idx] = w * inv_norm;
        }
        weights
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SwmaParams,
    ) -> Result<(Vec<i32>, usize), CudaSwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaSwmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaSwmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(CudaSwmaError::InvalidInput("period must be > 0".into()));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series;
                let v = data_tm_f32[idx];
                if !v.is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fv_row = fv.ok_or_else(|| {
                CudaSwmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv_row < period {
                return Err(CudaSwmaError::InvalidInput(format!(
                    "series {} lacks enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv_row
                )));
            }
            first_valids[series] = fv_row as i32;
        }

        Ok((first_valids, period))
    }

    fn launch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSwmaError> {
        if series_len == 0 {
            return Err(CudaSwmaError::InvalidInput("series_len is zero".into()));
        }
        if n_combos == 0 {
            return Err(CudaSwmaError::InvalidInput("no parameter combos".into()));
        }
        if max_period == 0 {
            return Err(CudaSwmaError::InvalidInput("max_period is zero".into()));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaSwmaError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("swma_batch_f32")
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        const BLOCK_X: u32 = 256;
        let grid_x = ((series_len as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x.max(1), n_combos as u32, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();
        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut warms_ptr = d_warms.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut max_period_i = max_period as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut warms_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut max_period_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSwmaError> {
        if period == 0 || cols == 0 || rows == 0 {
            return Err(CudaSwmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        if period > i32::MAX as usize || cols > i32::MAX as usize || rows > i32::MAX as usize {
            return Err(CudaSwmaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("swma_multi_series_one_param_f32")
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        const BLOCK_X: u32 = 128;
        let grid_x = ((rows as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x.max(1), cols as u32, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();
        let shared_bytes = (period * std::mem::size_of::<f32>()) as u32;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
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
                .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    pub fn swma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSwmaError> {
        self.launch_kernel(
            d_prices, d_periods, d_warms, series_len, n_combos, max_period, d_out,
        )
    }

    pub fn swma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &SwmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaSwmaError> {
        let (periods, first_valid) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let series_len = data_f32.len();
        let n_combos = periods.len();
        let max_period = periods.iter().copied().max().unwrap_or(0);

        let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
        let warms_i32: Vec<i32> = periods
            .iter()
            .map(|&p| (first_valid + p - 1) as i32)
            .collect();

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let d_warms =
            DeviceBuffer::from_slice(&warms_i32).map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        self.launch_kernel(
            &d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
    ) -> Result<DeviceArrayF32, CudaSwmaError> {
        let weights = Self::compute_weights(period);
        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(&weights).map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_weights,
            &d_first_valids,
            period,
            cols,
            rows,
            &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn swma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaSwmaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaSwmaError::InvalidInput(
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

    pub fn swma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SwmaParams,
    ) -> Result<DeviceArrayF32, CudaSwmaError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn swma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &SwmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaSwmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaSwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let arr = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaSwmaError::Cuda(e.to_string()))?;
        Ok(())
    }
}
