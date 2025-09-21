//! CUDA scaffolding for VPWMA (Variable Power Weighted Moving Average).
//!
//! Mirrors the ALMA/ZLEMA GPU integration pattern: parameter sweeps are expanded
//! on the host, weights are precomputed in FP32, and the kernel evaluates each
//! row independently while sharing the flattened weight buffer.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::vpwma::{expand_grid_vpwma, VpwmaBatchRange, VpwmaParams};
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
pub enum CudaVpwmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaVpwmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaVpwmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaVpwmaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaVpwmaError {}

pub struct CudaVpwma {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaVpwma {
    pub fn new(device_id: usize) -> Result<Self, CudaVpwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/vpwma_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &VpwmaBatchRange,
    ) -> Result<(Vec<VpwmaParams>, usize, usize), CudaVpwmaError> {
        if data_f32.is_empty() {
            return Err(CudaVpwmaError::InvalidInput("empty data".into()));
        }

        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaVpwmaError::InvalidInput("all values are NaN".into()))?;

        let combos = expand_grid_vpwma(sweep);
        if combos.is_empty() {
            return Err(CudaVpwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let len = data_f32.len();
        let mut max_period = 0usize;
        for combo in &combos {
            let period = combo.period.unwrap_or(0);
            let power = combo.power.unwrap_or(f64::NAN);
            if period < 2 {
                return Err(CudaVpwmaError::InvalidInput(
                    "period must be at least 2".into(),
                ));
            }
            if period > len {
                return Err(CudaVpwmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, len
                )));
            }
            if power.is_nan() || power.is_infinite() {
                return Err(CudaVpwmaError::InvalidInput("power must be finite".into()));
            }
            max_period = max_period.max(period);
        }

        if len - first_valid < max_period {
            return Err(CudaVpwmaError::InvalidInput(format!(
                "not enough valid data (need >= {}, have {} after first valid)",
                max_period,
                len - first_valid
            )));
        }

        Ok((combos, first_valid, len))
    }

    fn compute_weights(period: usize, power: f64) -> Result<(Vec<f32>, f32), CudaVpwmaError> {
        if !power.is_finite() {
            return Err(CudaVpwmaError::InvalidInput("power must be finite".into()));
        }
        if period < 2 {
            return Err(CudaVpwmaError::InvalidInput(
                "period must be at least 2".into(),
            ));
        }
        let win_len = period - 1;
        let mut weights = vec![0f32; win_len];
        let mut norm = 0.0f64;
        for k in 0..win_len {
            let w = (period as f64 - k as f64).powf(power);
            weights[k] = w as f32;
            norm += w;
        }
        if !norm.is_finite() || norm == 0.0 {
            return Err(CudaVpwmaError::InvalidInput(format!(
                "invalid normalization for period {} power {}",
                period, power
            )));
        }
        Ok((weights, (1.0 / norm) as f32))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VpwmaParams,
    ) -> Result<(Vec<i32>, usize, Vec<f32>, f32), CudaVpwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaVpwmaError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaVpwmaError::InvalidInput(format!(
                "data length mismatch: expected {}, got {}",
                cols * rows,
                data_tm_f32.len()
            )));
        }

        let period = params.period.unwrap_or(0);
        let power = params.power.unwrap_or(f64::NAN);
        if period < 2 {
            return Err(CudaVpwmaError::InvalidInput(
                "period must be at least 2".into(),
            ));
        }
        if !power.is_finite() {
            return Err(CudaVpwmaError::InvalidInput("power must be finite".into()));
        }

        let stride = cols;
        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut found = None;
            for row in 0..rows {
                let val = data_tm_f32[row * stride + series];
                if !val.is_nan() {
                    found = Some(row as i32);
                    break;
                }
            }
            let first = found.ok_or_else(|| {
                CudaVpwmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if rows - (first as usize) < period {
                return Err(CudaVpwmaError::InvalidInput(format!(
                    "series {} does not have enough data for period {}",
                    series, period
                )));
            }
            first_valids[series] = first;
        }

        let (weights, inv_norm) = Self::compute_weights(period, power)?;
        Ok((first_valids, period, weights, inv_norm))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_win_lengths: &DeviceBuffer<i32>,
        d_weights: &DeviceBuffer<f32>,
        d_inv_norms: &DeviceBuffer<f32>,
        len: usize,
        stride: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVpwmaError> {
        let func = self
            .module
            .get_function("vpwma_batch_f32")
            .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;

        let block_x: u32 = 128;
        let grid_x = ((n_combos as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut win_ptr = d_win_lengths.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut inv_ptr = d_inv_norms.as_device_ptr().as_raw();
            let mut series_len_i = len as i32;
            let mut stride_i = stride as i32;
            let mut first_valid_i = first_valid as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut win_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut inv_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut stride_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        d_weights: &DeviceBuffer<f32>,
        inv_norm: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVpwmaError> {
        let func = self
            .module
            .get_function("vpwma_many_series_one_param_f32")
            .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;

        let block_x: u32 = 128;
        let grid_x = ((num_series as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x.max(1), 1, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut period_i = period as i32;
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut inv = inv_norm;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();

            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut inv as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];

            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[VpwmaParams],
        first_valid: usize,
        len: usize,
    ) -> Result<DeviceArrayF32, CudaVpwmaError> {
        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;

        let n_combos = combos.len();
        let mut periods = Vec::with_capacity(n_combos);
        let mut win_lengths = Vec::with_capacity(n_combos);
        let mut inv_norms = Vec::with_capacity(n_combos);

        let stride = combos
            .iter()
            .map(|c| c.period.unwrap() - 1)
            .max()
            .unwrap_or(1);

        let mut weights_flat = vec![0f32; n_combos * stride];

        for (idx, combo) in combos.iter().enumerate() {
            let period = combo.period.unwrap();
            let power = combo.power.unwrap();
            let win_len = period - 1;

            periods.push(period as i32);
            win_lengths.push(win_len as i32);

            let mut norm = 0.0f64;
            for k in 0..win_len {
                let weight = (period as f64 - k as f64).powf(power);
                weights_flat[idx * stride + k] = weight as f32;
                norm += weight;
            }

            if !norm.is_finite() || norm == 0.0 {
                return Err(CudaVpwmaError::InvalidInput(format!(
                    "invalid normalization for period {} power {}",
                    period, power
                )));
            }

            inv_norms.push((1.0 / norm) as f32);
        }

        let d_periods =
            DeviceBuffer::from_slice(&periods).map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;
        let d_win_lengths = DeviceBuffer::from_slice(&win_lengths)
            .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat)
            .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;
        let d_inv_norms = DeviceBuffer::from_slice(&inv_norms)
            .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;

        let elems = n_combos * len;
        let mut d_out = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_win_lengths,
            &d_weights,
            &d_inv_norms,
            len,
            stride,
            first_valid,
            n_combos,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: len,
        })
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VpwmaParams,
    ) -> Result<DeviceArrayF32, CudaVpwmaError> {
        let (first_valids, period, weights, inv_norm) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let d_prices_tm = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&first_valids)
            .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(&weights).map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;

        let elems = cols * rows;
        let mut d_out_tm = unsafe { DeviceBuffer::<f32>::uninitialized(elems) }
            .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_first_valids,
            period,
            &d_weights,
            inv_norm,
            cols,
            rows,
            &mut d_out_tm,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows,
            cols,
        })
    }

    pub fn vpwma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &VpwmaBatchRange,
    ) -> Result<(DeviceArrayF32, Vec<VpwmaParams>), CudaVpwmaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        Ok((dev, combos))
    }

    pub fn vpwma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &VpwmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<VpwmaParams>), CudaVpwmaError> {
        let (combos, first_valid, len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * len;
        if out.len() != expected {
            return Err(CudaVpwmaError::InvalidInput(format!(
                "output slice length mismatch: expected {}, got {}",
                expected,
                out.len()
            )));
        }

        let dev = self.run_batch_kernel(data_f32, &combos, first_valid, len)?;
        dev.buf
            .copy_to(out)
            .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))?;
        Ok((combos.len(), len, combos))
    }

    pub fn vpwma_multi_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        inv_norm: f32,
        num_series: i32,
        series_len: i32,
        d_weights: &DeviceBuffer<f32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaVpwmaError> {
        if period < 2 || num_series <= 0 || series_len <= 0 {
            return Err(CudaVpwmaError::InvalidInput(
                "invalid period/series configuration".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            period as usize,
            d_weights,
            inv_norm,
            num_series as usize,
            series_len as usize,
            d_out_tm,
        )
    }

    pub fn vpwma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VpwmaParams,
    ) -> Result<DeviceArrayF32, CudaVpwmaError> {
        self.run_many_series_kernel(data_tm_f32, cols, rows, params)
    }

    pub fn vpwma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &VpwmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaVpwmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaVpwmaError::InvalidInput(format!(
                "output slice length mismatch: expected {}, got {}",
                cols * rows,
                out_tm.len()
            )));
        }

        let arr = self.run_many_series_kernel(data_tm_f32, cols, rows, params)?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaVpwmaError::Cuda(e.to_string()))
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        vpwma_benches,
        CudaVpwma,
        crate::indicators::moving_averages::vpwma::VpwmaBatchRange,
        crate::indicators::moving_averages::vpwma::VpwmaParams,
        vpwma_batch_dev,
        vpwma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::vpwma::VpwmaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1), power: (2.0, 2.0, 0.0) },
        crate::indicators::moving_averages::vpwma::VpwmaParams { period: Some(64), power: Some(2.0) },
        "vpwma",
        "vpwma"
    );
    pub use vpwma_benches::bench_profiles;
}
