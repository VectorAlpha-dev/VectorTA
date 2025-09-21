//! CUDA scaffolding for the Hull Moving Average (HMA).
//!
//! Mirrors the VRAM-first integrations used for ALMA/SMA: host validation
//! expands parameter sweeps, kernels operate purely in FP32, and results are
//! returned through zero-copy `DeviceArrayF32` handles with optional helpers to
//! stream data back to the host.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::hma::{HmaBatchRange, HmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaHmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaHmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaHmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaHmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaHmaError {}

pub struct CudaHma {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaHma {
    pub fn new(device_id: usize) -> Result<Self, CudaHmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/hma_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn expand_range(range: &HmaBatchRange) -> Vec<HmaParams> {
        let (start, end, step) = range.period;
        if step == 0 || start == end {
            return vec![HmaParams {
                period: Some(start),
            }];
        }
        (start..=end)
            .step_by(step)
            .map(|period| HmaParams {
                period: Some(period),
            })
            .collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &HmaBatchRange,
    ) -> Result<(Vec<HmaParams>, usize, usize, usize), CudaHmaError> {
        if data_f32.is_empty() {
            return Err(CudaHmaError::InvalidInput("empty data".into()));
        }

        let len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaHmaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_range(sweep);
        if combos.is_empty() {
            return Err(CudaHmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let tail_len = len - first_valid;
        let mut max_sqrt_len = 0usize;
        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period < 2 {
                return Err(CudaHmaError::InvalidInput(
                    "period must be at least 2".into(),
                ));
            }
            if period > len {
                return Err(CudaHmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            let half = period / 2;
            if half == 0 {
                return Err(CudaHmaError::InvalidInput(format!(
                    "period {} results in zero half-window",
                    period
                )));
            }
            let sqrt_len = ((period as f64).sqrt().floor() as usize).max(1);
            if tail_len < period + sqrt_len - 1 {
                return Err(CudaHmaError::InvalidInput(format!(
                    "not enough valid data for period {} (tail = {}, need >= {})",
                    period,
                    tail_len,
                    period + sqrt_len - 1
                )));
            }
            max_sqrt_len = max_sqrt_len.max(sqrt_len);
        }

        Ok((combos, first_valid, len, max_sqrt_len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_sqrt_len: usize,
        d_ring: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHmaError> {
        let func = self
            .module
            .get_function("hma_batch_f32")
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        let block_x: u32 = 128;
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut max_sqrt_i = max_sqrt_len as i32;
            let mut ring_ptr = d_ring.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut max_sqrt_i as *mut _ as *mut c_void,
                &mut ring_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[HmaParams],
        first_valid: usize,
        len: usize,
        max_sqrt_len: usize,
    ) -> Result<DeviceArrayF32, CudaHmaError> {
        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods =
            DeviceBuffer::from_slice(&periods).map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        let elems = combos.len() * len;
        let mut d_ring = unsafe { DeviceBuffer::<f32>::uninitialized(combos.len() * max_sqrt_len) }
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            len,
            combos.len(),
            first_valid,
            max_sqrt_len,
            &mut d_ring,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn hma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &HmaBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<HmaParams>), CudaHmaError> {
        let (combos, first_valid, len, max_sqrt_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len, max_sqrt_len)?;
        Ok((dev, combos))
    }

    pub fn hma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &HmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<HmaParams>), CudaHmaError> {
        let (combos, first_valid, len, max_sqrt_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaHmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len, max_sqrt_len)?;
        dev.buf
            .copy_to(out)
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        Ok((combos.len(), len, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HmaParams,
    ) -> Result<(Vec<i32>, usize, usize), CudaHmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaHmaError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaHmaError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }

        let period = params.period.unwrap_or(0);
        if period < 2 {
            return Err(CudaHmaError::InvalidInput(
                "period must be at least 2".into(),
            ));
        }
        if period > rows {
            return Err(CudaHmaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }
        let half = period / 2;
        if half == 0 {
            return Err(CudaHmaError::InvalidInput(format!(
                "period {} results in zero half-window",
                period
            )));
        }
        let sqrt_len = ((period as f64).sqrt().floor() as usize).max(1);

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaHmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv < period + sqrt_len - 1 {
                return Err(CudaHmaError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {}, need >= {})",
                    series,
                    period,
                    rows - fv,
                    period + sqrt_len - 1
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period, sqrt_len))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        num_series: usize,
        series_len: usize,
        period: usize,
        max_sqrt_len: usize,
        d_ring: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHmaError> {
        let func = self
            .module
            .get_function("hma_many_series_one_param_f32")
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        let block_x: u32 = 128;
        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut cols_i = num_series as i32;
            let mut rows_i = series_len as i32;
            let mut period_i = period as i32;
            let mut max_sqrt_i = max_sqrt_len as i32;
            let mut ring_ptr = d_ring.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut max_sqrt_i as *mut _ as *mut c_void,
                &mut ring_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
        sqrt_len: usize,
    ) -> Result<DeviceArrayF32, CudaHmaError> {
        let d_prices =
            DeviceBuffer::from_slice(data_tm_f32).map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        let elems = cols * rows;
        let mut d_ring = unsafe { DeviceBuffer::<f32>::uninitialized(cols * sqrt_len) }
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            cols,
            rows,
            period,
            sqrt_len,
            &mut d_ring,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn hma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HmaParams,
    ) -> Result<DeviceArrayF32, CudaHmaError> {
        let (first_valids, period, sqrt_len) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period, sqrt_len)
    }

    pub fn hma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaHmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaHmaError::InvalidInput(format!(
                "output length mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }
        let (first_valids, period, sqrt_len) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev =
            self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period, sqrt_len)?;
        dev.buf
            .copy_to(out_tm)
            .map_err(|e| CudaHmaError::Cuda(e.to_string()))
    }
}
