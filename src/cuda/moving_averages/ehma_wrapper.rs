//! CUDA wrapper for EHMA (Ehlers Hann Moving Average) kernels.
//!
//! Mirrors the ALMA/SWMA scaffolding: validate host inputs, upload FP32
//! buffers, and launch device kernels that either precompute Hann weights on the
//! GPU (batch path) or consume pre-normalized weights for the many-series path.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::ehma::{expand_grid, EhmaBatchRange, EhmaParams};
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
pub enum CudaEhmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaEhmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEhmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEhmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaEhmaError {}

pub struct CudaEhma {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaEhma {
    pub fn new(device_id: usize) -> Result<Self, CudaEhmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/ehma_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &EhmaBatchRange,
    ) -> Result<(Vec<EhmaParams>, usize, usize, usize), CudaEhmaError> {
        if data_f32.is_empty() {
            return Err(CudaEhmaError::InvalidInput("empty data".into()));
        }

        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaEhmaError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaEhmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaEhmaError::InvalidInput("period must be > 0".into()));
            }
            if period > len {
                return Err(CudaEhmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if len - first_valid < period {
                return Err(CudaEhmaError::InvalidInput(format!(
                    "not enough valid data: needed {}, have {}",
                    period,
                    len - first_valid
                )));
            }
            max_period = max_period.max(period);
        }

        Ok((combos, first_valid, len, max_period))
    }

    fn compute_normalized_weights(period: usize) -> Vec<f32> {
        let mut weights = vec![0.0f32; period];
        if period == 0 {
            return weights;
        }
        let mut sum = 0.0f32;
        let pi = std::f32::consts::PI;
        for idx in 0..period {
            let i = (period - idx) as f32;
            let angle = (2.0f32 * pi * i) / (period as f32 + 1.0f32);
            let wt = 1.0f32 - angle.cos();
            weights[idx] = wt;
            sum += wt;
        }
        if sum > 0.0 {
            let inv = 1.0f32 / sum;
            for w in &mut weights {
                *w *= inv;
            }
        }
        weights
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhmaParams,
    ) -> Result<(Vec<i32>, usize, Vec<f32>), CudaEhmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaEhmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaEhmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(CudaEhmaError::InvalidInput("period must be > 0".into()));
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
                CudaEhmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - fv_row < period {
                return Err(CudaEhmaError::InvalidInput(format!(
                    "series {} lacks enough valid data: needed {}, have {}",
                    series,
                    period,
                    rows - fv_row
                )));
            }
            first_valids[series] = fv_row as i32;
        }

        let weights = Self::compute_normalized_weights(period);
        Ok((first_valids, period, weights))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhmaError> {
        if series_len == 0 {
            return Err(CudaEhmaError::InvalidInput("series_len is zero".into()));
        }
        if n_combos == 0 {
            return Err(CudaEhmaError::InvalidInput("no parameter combos".into()));
        }
        if max_period == 0 {
            return Err(CudaEhmaError::InvalidInput("max_period is zero".into()));
        }
        if series_len > i32::MAX as usize
            || n_combos > i32::MAX as usize
            || max_period > i32::MAX as usize
        {
            return Err(CudaEhmaError::InvalidInput(
                "series_len, n_combos, or max_period exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("ehma_batch_f32")
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

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
                .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhmaError> {
        if period == 0 {
            return Err(CudaEhmaError::InvalidInput("period is zero".into()));
        }
        if num_series == 0 || series_len == 0 {
            return Err(CudaEhmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if period > i32::MAX as usize
            || num_series > i32::MAX as usize
            || series_len > i32::MAX as usize
        {
            return Err(CudaEhmaError::InvalidInput(
                "period, num_series, or series_len exceed i32::MAX".into(),
            ));
        }

        let func = self
            .module
            .get_function("ehma_multi_series_one_param_f32")
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        const BLOCK_X: u32 = 256;
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
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    pub fn ehma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warms: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhmaError> {
        self.launch_batch_kernel(
            d_prices, d_periods, d_warms, series_len, n_combos, max_period, d_out,
        )
    }

    pub fn ehma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &EhmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaEhmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();

        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut warms_i32 = Vec::with_capacity(n_combos);
        for prm in &combos {
            let period = prm.period.unwrap();
            if period > i32::MAX as usize {
                return Err(CudaEhmaError::InvalidInput(
                    "period exceeds i32::MAX".into(),
                ));
            }
            let warm = first_valid + period - 1;
            if warm > i32::MAX as usize {
                return Err(CudaEhmaError::InvalidInput(
                    "warm index exceeds i32::MAX".into(),
                ));
            }
            periods_i32.push(period as i32);
            warms_i32.push(warm as i32);
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_warms =
            DeviceBuffer::from_slice(&warms_i32).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn ehma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &EhmaBatchRange,
        out: &mut [f32],
    ) -> Result<Vec<EhmaParams>, CudaEhmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = combos.len();
        if out.len() != n_combos * series_len {
            return Err(CudaEhmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                n_combos * series_len
            )));
        }

        let mut periods_i32 = Vec::with_capacity(n_combos);
        let mut warms_i32 = Vec::with_capacity(n_combos);
        for prm in &combos {
            let period = prm.period.unwrap();
            if period > i32::MAX as usize {
                return Err(CudaEhmaError::InvalidInput(
                    "period exceeds i32::MAX".into(),
                ));
            }
            let warm = first_valid + period - 1;
            if warm > i32::MAX as usize {
                return Err(CudaEhmaError::InvalidInput(
                    "warm index exceeds i32::MAX".into(),
                ));
            }
            periods_i32.push(period as i32);
            warms_i32.push(warm as i32);
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_warms =
            DeviceBuffer::from_slice(&warms_i32).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices, &d_periods, &d_warms, series_len, n_combos, max_period, &mut d_out,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        d_out
            .copy_to(out)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        Ok(combos)
    }

    pub fn ehma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEhmaError> {
        if period <= 0 || num_series <= 0 || series_len <= 0 {
            return Err(CudaEhmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            d_weights,
            period as usize,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out_tm,
        )
    }

    pub fn ehma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhmaParams,
    ) -> Result<DeviceArrayF32, CudaEhmaError> {
        let (first_valids, period, weights) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(&weights).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_weights,
            period,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn ehma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &EhmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaEhmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaEhmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period, weights) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(&weights).map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_weights,
            period,
            cols,
            rows,
            &d_first_valids,
            &mut d_out_tm,
        )?;
        self.stream
            .synchronize()
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        d_out_tm
            .copy_to(out_tm)
            .map_err(|e| CudaEhmaError::Cuda(e.to_string()))?;

        Ok(())
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        ehma_benches,
        CudaEhma,
        crate::indicators::moving_averages::ehma::EhmaBatchRange,
        crate::indicators::moving_averages::ehma::EhmaParams,
        ehma_batch_dev,
        ehma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::ehma::EhmaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::ehma::EhmaParams { period: Some(64) },
        "ehma",
        "ehma"
    );
    pub use ehma_benches::bench_profiles;
}
