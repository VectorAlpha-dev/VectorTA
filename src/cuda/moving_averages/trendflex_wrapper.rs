//! CUDA scaffolding for the TrendFlex filter.
//!
//! Mirrors the VRAM-first integrations used by ALMA/ZLEMA/VPWMA: host-side
//! validation expands parameter sweeps and the device kernels emit `f32`
//! outputs either into GPU buffers or directly into host slices.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::trendflex::{
    expand_grid_trendflex, TrendFlexBatchRange, TrendFlexParams,
};
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
pub enum CudaTrendflexError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaTrendflexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaTrendflexError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaTrendflexError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaTrendflexError {}

pub struct CudaTrendflex {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaTrendflex {
    pub fn new(device_id: usize) -> Result<Self, CudaTrendflexError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/trendflex_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &TrendFlexBatchRange,
    ) -> Result<(Vec<TrendFlexParams>, usize, usize), CudaTrendflexError> {
        if data_f32.is_empty() {
            return Err(CudaTrendflexError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTrendflexError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_trendflex(sweep);
        if combos.is_empty() {
            return Err(CudaTrendflexError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = data_f32.len();
        let tail_len = len - first_valid;
        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaTrendflexError::InvalidInput(
                    "period must be at least 1".into(),
                ));
            }
            if period >= len {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            let ss_period = ((period as f64) / 2.0).round() as usize;
            if tail_len < period {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "not enough valid data for period {} (valid tail = {})",
                    period, tail_len
                )));
            }
            if tail_len < ss_period {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "not enough valid data for smoother period {} (valid tail = {})",
                    ss_period, tail_len
                )));
            }
        }

        Ok((combos, first_valid, len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_ssf: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrendflexError> {
        let func = self
            .module
            .get_function("trendflex_batch_f32")
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;

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
            let mut ssf_ptr = d_ssf.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut ssf_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[TrendFlexParams],
        first_valid: usize,
        len: usize,
    ) -> Result<DeviceArrayF32, CudaTrendflexError> {
        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let periods: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;

        let elems = combos.len() * len;
        let mut d_ssf = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            len,
            combos.len(),
            first_valid,
            &mut d_ssf,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: combos.len(),
            cols: len,
        })
    }

    pub fn trendflex_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &TrendFlexBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<TrendFlexParams>), CudaTrendflexError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        Ok((dev, combos))
    }

    pub fn trendflex_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &TrendFlexBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<TrendFlexParams>), CudaTrendflexError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "output slice length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        dev.buf
            .copy_to(out)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        Ok((combos.len(), len, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrendFlexParams,
    ) -> Result<(Vec<i32>, usize), CudaTrendflexError> {
        if cols == 0 || rows == 0 {
            return Err(CudaTrendflexError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }
        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaTrendflexError::InvalidInput(
                "period must be at least 1".into(),
            ));
        }
        if period >= rows {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, rows
            )));
        }
        let ss_period = ((period as f64) / 2.0).round() as usize;
        if ss_period == 0 {
            return Err(CudaTrendflexError::InvalidInput(
                "smoother period must be positive".into(),
            ));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let idx = row * cols + series;
                let val = data_tm_f32[idx];
                if !val.is_nan() {
                    found = Some(row);
                    break;
                }
            }
            let fv = found.ok_or_else(|| {
                CudaTrendflexError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            let tail = rows - fv;
            if tail < period {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "series {} insufficient data for period {} (tail = {})",
                    series, period, tail
                )));
            }
            if tail < ss_period {
                return Err(CudaTrendflexError::InvalidInput(format!(
                    "series {} insufficient data for smoother {} (tail = {})",
                    series, ss_period, tail
                )));
            }
            first_valids[series] = fv as i32;
        }

        Ok((first_valids, period))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        cols: usize,
        rows: usize,
        period: usize,
        d_ssf: &mut DeviceBuffer<f32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTrendflexError> {
        let func = self
            .module
            .get_function("trendflex_many_series_one_param_f32")
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;

        let block_x: u32 = 128;
        let grid_x = ((cols as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = cols as i32;
            let mut series_len_i = rows as i32;
            let mut period_i = period as i32;
            let mut ssf_ptr = d_ssf.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut ssf_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
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
    ) -> Result<DeviceArrayF32, CudaTrendflexError> {
        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let elems = cols * rows;
        let mut d_ssf = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_first_valids,
            cols,
            rows,
            period,
            &mut d_ssf,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn trendflex_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrendFlexParams,
    ) -> Result<DeviceArrayF32, CudaTrendflexError> {
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)
    }

    pub fn trendflex_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &TrendFlexParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaTrendflexError> {
        if out_tm.len() != cols * rows {
            return Err(CudaTrendflexError::InvalidInput(format!(
                "output slice mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }
        let (first_valids, period) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let dev = self.run_many_series_kernel(data_tm_f32, cols, rows, &first_valids, period)?;
        dev.buf
            .copy_to(out_tm)
            .map_err(|e| CudaTrendflexError::Cuda(e.to_string()))
    }
}
