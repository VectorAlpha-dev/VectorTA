//! CUDA support for the Cubic Weighted Moving Average (CWMA).
//!
//! This mirrors the public API offered by the ALMA CUDA wrapper but is tailored
//! to the simpler parameter surface of CWMA (period only). Kernels operate in
//! single precision and return VRAM-resident buffers so higher layers (Python,
//! bindings, etc.) can decide when to stage back to host memory.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::cwma::{CwmaBatchRange, CwmaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaCwmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaCwmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaCwmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaCwmaError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}

impl std::error::Error for CudaCwmaError {}

pub struct CudaCwma {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaCwma {
    pub fn new(device_id: usize) -> Result<Self, CudaCwmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/cwma_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

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

    fn expand_periods(range: &CwmaBatchRange) -> Vec<CwmaParams> {
        let (start, end, step) = range.period;
        let periods = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect::<Vec<_>>()
        };
        periods
            .into_iter()
            .map(|p| CwmaParams { period: Some(p) })
            .collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &CwmaBatchRange,
    ) -> Result<(Vec<CwmaParams>, usize, usize, usize), CudaCwmaError> {
        if data_f32.is_empty() {
            return Err(CudaCwmaError::InvalidInput("empty data".into()));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaCwmaError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaCwmaError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        let mut max_period = 0usize;
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period <= 1 {
                return Err(CudaCwmaError::InvalidInput(format!(
                    "invalid period {} (must be > 1)",
                    period
                )));
            }
            if period > series_len {
                return Err(CudaCwmaError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, series_len
                )));
            }
            if series_len - first_valid < period {
                return Err(CudaCwmaError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    period,
                    series_len - first_valid
                )));
            }
            max_period = max_period.max(period);
        }

        Ok((combos, first_valid, series_len, max_period))
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[CwmaParams],
        first_valid: usize,
        series_len: usize,
        max_period: usize,
    ) -> Result<DeviceArrayF32, CudaCwmaError> {
        let n_combos = combos.len();
        let weights_stride = max_period;
        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let weights_bytes = n_combos * weights_stride * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let inv_norm_bytes = n_combos * std::mem::size_of::<f32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + weights_bytes + periods_bytes + inv_norm_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaCwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let mut periods_i32 = vec![0i32; n_combos];
        let mut weights_flat = vec![0f32; n_combos * weights_stride];
        let mut inv_norms = vec![0f32; n_combos];

        for (idx, prm) in combos.iter().enumerate() {
            let period = prm.period.unwrap();
            let weight_len = period - 1;
            let mut norm = 0.0f32;
            for k in 0..weight_len {
                let weight = ((period - k) as f32).powi(3);
                weights_flat[idx * weights_stride + k] = weight;
                norm += weight;
            }
            if norm == 0.0 {
                return Err(CudaCwmaError::InvalidInput(format!(
                    "period {} produced zero normalization",
                    period
                )));
            }
            periods_i32[idx] = period as i32;
            inv_norms[idx] = 1.0 / norm;
        }

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let d_weights = DeviceBuffer::from_slice(&weights_flat)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let d_inv_norms =
            DeviceBuffer::from_slice(&inv_norms).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_weights,
            &d_periods,
            &d_inv_norms,
            series_len,
            n_combos,
            first_valid,
            max_period,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_inv_norms: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        max_period: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCwmaError> {
        let block_x: u32 = 256;
        let grid_x = ((series_len as u32) + block_x - 1) / block_x;
        let grid: GridSize = (grid_x, n_combos as u32, 1).into();
        let block: BlockSize = (block_x, 1, 1).into();
        let shared_bytes = (max_period * std::mem::size_of::<f32>()) as u32;

        let func = self
            .module
            .get_function("cwma_batch_f32")
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut inv_ptr = d_inv_norms.as_device_ptr().as_raw();
            let mut max_period_i = max_period as i32;
            let mut series_len_i = series_len as i32;
            let mut n_combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut inv_ptr as *mut _ as *mut c_void,
                &mut max_period_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn cwma_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_inv_norms: &DeviceBuffer<f32>,
        max_period: i32,
        series_len: i32,
        n_combos: i32,
        first_valid: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCwmaError> {
        if max_period <= 1 || series_len <= 0 || n_combos <= 0 {
            return Err(CudaCwmaError::InvalidInput(
                "max_period, series_len, and n_combos must be positive".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_weights,
            d_periods,
            d_inv_norms,
            series_len as usize,
            n_combos as usize,
            first_valid.max(0) as usize,
            max_period as usize,
            d_out,
        )
    }

    pub fn cwma_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &CwmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaCwmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)
    }

    pub fn cwma_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &CwmaBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<CwmaParams>), CudaCwmaError> {
        let (combos, first_valid, series_len, max_period) =
            Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaCwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(data_f32, &combos, first_valid, series_len, max_period)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, combos))
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CwmaParams,
    ) -> Result<(Vec<i32>, usize, Vec<f32>, f32), CudaCwmaError> {
        if cols == 0 || rows == 0 {
            return Err(CudaCwmaError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaCwmaError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        if period <= 1 {
            return Err(CudaCwmaError::InvalidInput(format!(
                "invalid period {} (must be > 1)",
                period
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series in 0..cols {
            let mut fv = None;
            for t in 0..rows {
                let idx = t * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let found = fv
                .ok_or_else(|| CudaCwmaError::InvalidInput(format!("series {} all NaN", series)))?;
            if (rows as i32 - found) < period as i32 {
                return Err(CudaCwmaError::InvalidInput(format!(
                    "series {} lacks data: need >= {}, valid = {}",
                    series,
                    period,
                    rows as i32 - found
                )));
            }
            first_valids[series] = found;
        }

        let weight_len = period - 1;
        let mut weights = vec![0f32; weight_len];
        let mut norm = 0.0f32;
        for k in 0..weight_len {
            let w = ((period - k) as f32).powi(3);
            weights[k] = w;
            norm += w;
        }
        if norm == 0.0 {
            return Err(CudaCwmaError::InvalidInput(format!(
                "period {} produced zero normalization",
                period
            )));
        }
        let inv_norm = 1.0 / norm;

        Ok((first_valids, period, weights, inv_norm))
    }

    fn run_many_series_kernel(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        first_valids: &[i32],
        period: usize,
        weights: &[f32],
        inv_norm: f32,
    ) -> Result<DeviceArrayF32, CudaCwmaError> {
        let weights_bytes = weights.len() * std::mem::size_of::<f32>();
        let prices_bytes = cols * rows * std::mem::size_of::<f32>();
        let first_valid_bytes = cols * std::mem::size_of::<i32>();
        let out_bytes = cols * rows * std::mem::size_of::<f32>();
        let required = weights_bytes + prices_bytes + first_valid_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaCwmaError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(first_valids)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let d_weights =
            DeviceBuffer::from_slice(weights).map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(
            &d_prices,
            &d_weights,
            period,
            inv_norm,
            cols,
            rows,
            &d_first_valids,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: usize,
        inv_norm: f32,
        cols: usize,
        rows: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCwmaError> {
        const BLOCK_X: u32 = 128;
        let grid_x = ((rows as u32) + BLOCK_X - 1) / BLOCK_X;
        let grid: GridSize = (grid_x, cols as u32, 1).into();
        let block: BlockSize = (BLOCK_X, 1, 1).into();
        let shared_bytes = (period.saturating_sub(1) * std::mem::size_of::<f32>()) as u32;

        let func = self
            .module
            .get_function("cwma_multi_series_one_param_time_major_f32")
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut weights_ptr = d_weights.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut inv = inv_norm;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut fvalid_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut weights_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut inv as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut fvalid_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, shared_bytes, args)
                .map_err(|e| CudaCwmaError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn cwma_multi_series_one_param_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_weights: &DeviceBuffer<f32>,
        period: i32,
        inv_norm: f32,
        num_series: i32,
        series_len: i32,
        d_first_valids: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaCwmaError> {
        if period <= 1 || num_series <= 0 || series_len <= 0 {
            return Err(CudaCwmaError::InvalidInput(
                "period, num_series, and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices,
            d_weights,
            period as usize,
            inv_norm,
            num_series as usize,
            series_len as usize,
            d_first_valids,
            d_out,
        )
    }

    pub fn cwma_multi_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CwmaParams,
    ) -> Result<DeviceArrayF32, CudaCwmaError> {
        let (first_valids, period, weights, inv_norm) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            &weights,
            inv_norm,
        )
    }

    pub fn cwma_multi_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &CwmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaCwmaError> {
        if out_tm.len() != cols * rows {
            return Err(CudaCwmaError::InvalidInput(format!(
                "out slice wrong length: got {}, expected {}",
                out_tm.len(),
                cols * rows
            )));
        }
        let (first_valids, period, weights, inv_norm) =
            Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;
        let arr = self.run_many_series_kernel(
            data_tm_f32,
            cols,
            rows,
            &first_valids,
            period,
            &weights,
            inv_norm,
        )?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaCwmaError::Cuda(e.to_string()))
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        cwma_benches,
        CudaCwma,
        crate::indicators::moving_averages::cwma::CwmaBatchRange,
        crate::indicators::moving_averages::cwma::CwmaParams,
        cwma_batch_dev,
        cwma_multi_series_one_param_time_major_dev,
        crate::indicators::moving_averages::cwma::CwmaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::cwma::CwmaParams { period: Some(64) },
        "cwma",
        "cwma"
    );
    pub use cwma_benches::bench_profiles;
}
