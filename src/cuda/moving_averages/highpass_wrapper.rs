//! CUDA wrapper for the single-pole High-Pass filter kernels.
//!
//! Mirrors the VRAM-first approach used by other moving averages: callers pass
//! FP32 slices, kernels operate entirely on the device, and results are exposed
//! via `DeviceArrayF32` handles to avoid implicit host transfers.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::highpass::{HighPassBatchRange, HighPassParams};
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
pub enum CudaHighpassError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaHighpassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaHighpassError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaHighpassError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CudaHighpassError {}

pub struct CudaHighpass {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaHighpass {
    pub fn new(device_id: usize) -> Result<Self, CudaHighpassError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/highpass_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;

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

    fn expand_periods(range: &HighPassBatchRange) -> Vec<HighPassParams> {
        let (start, end, step) = range.period;
        let periods = if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect::<Vec<_>>()
        };
        periods
            .into_iter()
            .map(|p| HighPassParams { period: Some(p) })
            .collect()
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &HighPassBatchRange,
    ) -> Result<(Vec<HighPassParams>, usize, usize), CudaHighpassError> {
        if data_f32.is_empty() {
            return Err(CudaHighpassError::InvalidInput("empty data".into()));
        }
        if data_f32.len() < 2 {
            return Err(CudaHighpassError::InvalidInput(
                "series must contain at least two samples".into(),
            ));
        }
        let first_valid = data_f32
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaHighpassError::InvalidInput("all values are NaN".into()))?;

        let combos = Self::expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaHighpassError::InvalidInput(
                "no parameter combinations".into(),
            ));
        }

        let series_len = data_f32.len();
        for prm in &combos {
            let period = prm.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaHighpassError::InvalidInput(
                    "period must be >= 1".into(),
                ));
            }
            if period > series_len {
                return Err(CudaHighpassError::InvalidInput(format!(
                    "period {} exceeds data length {}",
                    period, series_len
                )));
            }
            let valid = series_len - first_valid;
            if valid < period {
                return Err(CudaHighpassError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    period, valid
                )));
            }
            let theta = 2.0 * std::f64::consts::PI / period as f64;
            let cos_val = theta.cos();
            if cos_val.abs() < 1e-12 {
                return Err(CudaHighpassError::InvalidInput(format!(
                    "period {} yields unstable alpha (cos(theta) ≈ 0)",
                    period
                )));
            }
        }

        Ok((combos, first_valid, series_len))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHighpassError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaHighpassError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }

        let func = self
            .module
            .get_function("highpass_batch_f32")
            .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn run_batch_kernel(
        &self,
        data_f32: &[f32],
        combos: &[HighPassParams],
        series_len: usize,
    ) -> Result<DeviceArrayF32, CudaHighpassError> {
        let n_combos = combos.len();

        let prices_bytes = series_len * std::mem::size_of::<f32>();
        let periods_bytes = n_combos * std::mem::size_of::<i32>();
        let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
        let required = prices_bytes + periods_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaHighpassError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;
        let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
        let d_periods = DeviceBuffer::from_slice(&periods_i32)
            .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
                .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(&d_prices, &d_periods, series_len, n_combos, &mut d_out)?;
        self.stream
            .synchronize()
            .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: series_len,
        })
    }

    pub fn highpass_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &HighPassBatchRange,
    ) -> Result<DeviceArrayF32, CudaHighpassError> {
        let (combos, _first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        self.run_batch_kernel(data_f32, &combos, series_len)
    }

    pub fn highpass_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &HighPassBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<HighPassParams>), CudaHighpassError> {
        let (combos, _first_valid, series_len) = Self::prepare_batch_inputs(data_f32, sweep)?;
        let expected = combos.len() * series_len;
        if out.len() != expected {
            return Err(CudaHighpassError::InvalidInput(format!(
                "out slice length {} != expected {}",
                out.len(),
                expected
            )));
        }
        let arr = self.run_batch_kernel(data_f32, &combos, series_len)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn highpass_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        series_len: i32,
        n_combos: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHighpassError> {
        if series_len <= 0 || n_combos <= 0 {
            return Err(CudaHighpassError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        self.launch_batch_kernel(
            d_prices,
            d_periods,
            series_len as usize,
            n_combos as usize,
            d_out,
        )
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HighPassParams,
    ) -> Result<usize, CudaHighpassError> {
        if cols == 0 || rows == 0 {
            return Err(CudaHighpassError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }
        if rows < 2 {
            return Err(CudaHighpassError::InvalidInput(
                "series must contain at least two samples".into(),
            ));
        }
        if data_tm_f32.len() != cols * rows {
            return Err(CudaHighpassError::InvalidInput(format!(
                "data length {} != cols*rows {}",
                data_tm_f32.len(),
                cols * rows
            )));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaHighpassError::InvalidInput(
                "period must be >= 1".into(),
            ));
        }
        if period > rows {
            return Err(CudaHighpassError::InvalidInput(format!(
                "period {} exceeds series_len {}",
                period, rows
            )));
        }

        let theta = 2.0 * std::f64::consts::PI / period as f64;
        if theta.cos().abs() < 1e-12 {
            return Err(CudaHighpassError::InvalidInput(format!(
                "period {} yields unstable alpha (cos(theta) ≈ 0)",
                period
            )));
        }

        for series in 0..cols {
            let mut first = None;
            for t in 0..rows {
                let idx = t * cols + series;
                if !data_tm_f32[idx].is_nan() {
                    first = Some(t);
                    break;
                }
            }
            let fv = first.ok_or_else(|| {
                CudaHighpassError::InvalidInput(format!("series {} all NaN", series))
            })?;
            if rows - fv < period {
                return Err(CudaHighpassError::InvalidInput(format!(
                    "series {} lacks valid samples: need >= {}, valid = {}",
                    series,
                    period,
                    rows - fv
                )));
            }
        }

        Ok(period)
    }

    fn launch_many_series_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: usize,
        cols: usize,
        rows: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHighpassError> {
        if cols == 0 || rows == 0 {
            return Err(CudaHighpassError::InvalidInput(
                "num_series or series_len is zero".into(),
            ));
        }

        let func = self
            .module
            .get_function("highpass_many_series_one_param_time_major_f32")
            .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;

        let grid: GridSize = (cols as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut cols_i = cols as i32;
            let mut rows_i = rows as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut cols_i as *mut _ as *mut c_void,
                &mut rows_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    pub fn highpass_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        cols: usize,
        rows: usize,
        params: &HighPassParams,
    ) -> Result<DeviceArrayF32, CudaHighpassError> {
        let period = Self::prepare_many_series_inputs(data_tm_f32, cols, rows, params)?;

        let prices_bytes = data_tm_f32.len() * std::mem::size_of::<f32>();
        let out_bytes = data_tm_f32.len() * std::mem::size_of::<f32>();
        let required = prices_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024;
        if !Self::will_fit(required, headroom) {
            return Err(CudaHighpassError::InvalidInput(format!(
                "estimated device memory {:.2} MB exceeds free VRAM",
                (required as f64) / (1024.0 * 1024.0)
            )));
        }

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }
            .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;

        self.launch_many_series_kernel(&d_prices, period, cols, rows, &mut d_out)?;
        self.stream
            .synchronize()
            .map_err(|e| CudaHighpassError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    pub fn highpass_many_series_one_param_time_major_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        period: i32,
        cols: i32,
        rows: i32,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHighpassError> {
        if period <= 0 || cols <= 0 || rows <= 0 {
            return Err(CudaHighpassError::InvalidInput(
                "period, num_series and series_len must be positive".into(),
            ));
        }
        self.launch_many_series_kernel(
            d_prices,
            period as usize,
            cols as usize,
            rows as usize,
            d_out,
        )
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        highpass_benches,
        CudaHighpass,
        crate::indicators::moving_averages::highpass::HighPassBatchRange,
        crate::indicators::moving_averages::highpass::HighPassParams,
        highpass_batch_dev,
        highpass_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::highpass::HighPassBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::highpass::HighPassParams { period: Some(64) },
        "highpass",
        "highpass"
    );
    pub use highpass_benches::bench_profiles;
}
