//! CUDA support for the Exponential Moving Average (EMA).
//!
//! Mirrors the ALMA CUDA API by providing zero-copy device entry points for
//! both the one-series × many-parameter sweep and the time-major
//! many-series × one-parameter scenario.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::ema::{EmaBatchRange, EmaParams};
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
pub enum CudaEmaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaEmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaEmaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaEmaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaEmaError {}

pub struct CudaEma {
    module: Module,
    stream: Stream,
    _context: Context,
}

struct PreparedEmaBatch {
    combos: Vec<EmaParams>,
    first_valid: usize,
    series_len: usize,
    periods_i32: Vec<i32>,
    alphas_f32: Vec<f32>,
}

struct PreparedEmaManySeries {
    first_valids: Vec<i32>,
    period: i32,
    alpha: f32,
    num_series: usize,
    series_len: usize,
}

impl CudaEma {
    pub fn new(device_id: usize) -> Result<Self, CudaEmaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaEmaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaEmaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaEmaError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/ema_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaEmaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaEmaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn ema_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &EmaBatchRange,
    ) -> Result<DeviceArrayF32, CudaEmaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaEmaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&prepared.periods_i32)
            .map_err(|e| CudaEmaError::Cuda(e.to_string()))?;
        let d_alphas = DeviceBuffer::from_slice(&prepared.alphas_f32)
            .map_err(|e| CudaEmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaEmaError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_alphas,
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

    #[allow(clippy::too_many_arguments)]
    pub fn ema_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEmaError> {
        if series_len == 0 {
            return Err(CudaEmaError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaEmaError::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, series_len
            )));
        }
        if n_combos == 0 {
            return Err(CudaEmaError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if d_periods.len() != n_combos || d_alphas.len() != n_combos {
            return Err(CudaEmaError::InvalidInput(
                "period/alpha buffer length mismatch".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaEmaError::InvalidInput(
                "prices length must match series_len".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaEmaError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_alphas,
            series_len,
            first_valid,
            n_combos,
            d_out,
        )
    }

    pub fn ema_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &EmaBatchRange,
        out_flat: &mut [f32],
    ) -> Result<(), CudaEmaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        if out_flat.len() != prepared.series_len * prepared.combos.len() {
            return Err(CudaEmaError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.ema_batch_dev(data_f32, sweep)?;
        handle
            .buf
            .copy_to(out_flat)
            .map_err(|e| CudaEmaError::Cuda(e.to_string()))
    }

    pub fn ema_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &EmaParams,
    ) -> Result<DeviceArrayF32, CudaEmaError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        let d_prices =
            DeviceBuffer::from_slice(data_tm_f32).map_err(|e| CudaEmaError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaEmaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)
                .map_err(|e| CudaEmaError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            prepared.period,
            prepared.alpha,
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
    pub fn ema_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEmaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaEmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if period <= 0 {
            return Err(CudaEmaError::InvalidInput("period must be positive".into()));
        }
        let total = num_series * series_len;
        if d_prices_tm.len() != total || d_out_tm.len() != total {
            return Err(CudaEmaError::InvalidInput(
                "time-major buffer length mismatch".into(),
            ));
        }
        if d_first_valids.len() != num_series {
            return Err(CudaEmaError::InvalidInput(
                "first_valids buffer length mismatch".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            period,
            alpha,
            num_series,
            series_len,
            d_out_tm,
        )
    }

    pub fn ema_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &EmaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaEmaError> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaEmaError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.ema_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        handle
            .buf
            .copy_to(out_tm)
            .map_err(|e| CudaEmaError::Cuda(e.to_string()))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_alphas: &DeviceBuffer<f32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEmaError> {
        if n_combos == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("ema_batch_f32")
            .map_err(|e| CudaEmaError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (256, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut alphas_ptr = d_alphas.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut n_combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut alphas_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut n_combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaEmaError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaEmaError::Cuda(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        alpha: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaEmaError> {
        if num_series == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("ema_many_series_one_param_f32")
            .map_err(|e| CudaEmaError::Cuda(e.to_string()))?;

        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (128, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut period_i = period;
            let mut alpha_f = alpha;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut alpha_f as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaEmaError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaEmaError::Cuda(e.to_string()))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &EmaBatchRange,
    ) -> Result<PreparedEmaBatch, CudaEmaError> {
        if data_f32.is_empty() {
            return Err(CudaEmaError::InvalidInput("input data is empty".into()));
        }

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaEmaError::InvalidInput(
                "no parameter combinations provided".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaEmaError::InvalidInput("all values are NaN".into()))?;

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut alphas_f32 = Vec::with_capacity(combos.len());

        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaEmaError::InvalidInput("period must be positive".into()));
            }
            if series_len - first_valid < period {
                return Err(CudaEmaError::InvalidInput(format!(
                    "not enough valid data: need {} valid samples, have {}",
                    period,
                    series_len - first_valid
                )));
            }
            periods_i32.push(period as i32);
            alphas_f32.push(2.0f32 / (period as f32 + 1.0f32));
        }

        Ok(PreparedEmaBatch {
            combos,
            first_valid,
            series_len,
            periods_i32,
            alphas_f32,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &EmaParams,
    ) -> Result<PreparedEmaManySeries, CudaEmaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaEmaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaEmaError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }

        let period = params.period.unwrap_or(0) as i32;
        if period <= 0 {
            return Err(CudaEmaError::InvalidInput("period must be positive".into()));
        }

        let alpha = 2.0f32 / (period as f32 + 1.0f32);

        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let v = data_tm_f32[t * num_series + series];
                if v.is_finite() {
                    fv = Some(t as i32);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaEmaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            let remaining = series_len - fv as usize;
            if remaining < period as usize {
                return Err(CudaEmaError::InvalidInput(format!(
                    "series {} does not have enough valid data: need {} valid samples, have {}",
                    series, period, remaining
                )));
            }
            first_valids.push(fv);
        }

        Ok(PreparedEmaManySeries {
            first_valids,
            period,
            alpha,
            num_series,
            series_len,
        })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        ema_benches,
        CudaEma,
        crate::indicators::moving_averages::ema::EmaBatchRange,
        crate::indicators::moving_averages::ema::EmaParams,
        ema_batch_dev,
        ema_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::ema::EmaBatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1) },
        crate::indicators::moving_averages::ema::EmaParams { period: Some(64) },
        "ema",
        "ema"
    );
    pub use ema_benches::bench_profiles;
}

fn expand_grid(range: &EmaBatchRange) -> Vec<EmaParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    axis(range.period)
        .into_iter()
        .map(|p| EmaParams { period: Some(p) })
        .collect()
}
