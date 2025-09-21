//! CUDA support for the Jump Step Average (JSA).
//!
//! Provides zero-copy device entry points that mirror the ALMA CUDA surface,
//! covering both the classic "one series × many periods" sweep and the
//! time-major "many series × one period" flow. Kernels operate entirely in
//! FP32 and rely on host-prepared warm-up indices so that GPU work is limited to
//! simple averaging.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::jsa::{JsaBatchRange, JsaParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::convert::TryFrom;
use std::ffi::c_void;
use std::fmt;

#[derive(Debug)]
pub enum CudaJsaError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaJsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaJsaError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaJsaError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaJsaError {}

pub struct CudaJsa {
    module: Module,
    stream: Stream,
    _context: Context,
}

struct PreparedJsaBatch {
    combos: Vec<JsaParams>,
    first_valid: usize,
    series_len: usize,
    periods_i32: Vec<i32>,
    warm_indices: Vec<i32>,
}

struct PreparedJsaManySeries {
    first_valids: Vec<i32>,
    warm_indices: Vec<i32>,
    period: i32,
}

impl CudaJsa {
    pub fn new(device_id: usize) -> Result<Self, CudaJsaError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaJsaError::Cuda(e.to_string()))?;
        let device =
            Device::get_device(device_id as u32).map_err(|e| CudaJsaError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaJsaError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/jsa_kernel.ptx"));
        let module = Module::from_ptx(ptx, &[]).map_err(|e| CudaJsaError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaJsaError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn jsa_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &JsaBatchRange,
    ) -> Result<DeviceArrayF32, CudaJsaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaJsaError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&prepared.periods_i32)
            .map_err(|e| CudaJsaError::Cuda(e.to_string()))?;
        let d_warm = DeviceBuffer::from_slice(&prepared.warm_indices)
            .map_err(|e| CudaJsaError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaJsaError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_warm,
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
    pub fn jsa_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warm: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJsaError> {
        if series_len == 0 {
            return Err(CudaJsaError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaJsaError::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, series_len
            )));
        }
        if n_combos == 0 {
            return Err(CudaJsaError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaJsaError::InvalidInput(
                "prices buffer length mismatch".into(),
            ));
        }
        if d_periods.len() != n_combos || d_warm.len() != n_combos {
            return Err(CudaJsaError::InvalidInput(
                "period or warm buffer length mismatch".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaJsaError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_warm,
            series_len,
            first_valid,
            n_combos,
            d_out,
        )
    }

    pub fn jsa_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &JsaBatchRange,
        out_flat: &mut [f32],
    ) -> Result<(), CudaJsaError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        if out_flat.len() != prepared.series_len * prepared.combos.len() {
            return Err(CudaJsaError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.jsa_batch_dev(data_f32, sweep)?;
        handle
            .buf
            .copy_to(out_flat)
            .map_err(|e| CudaJsaError::Cuda(e.to_string()))
    }

    pub fn jsa_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &JsaParams,
    ) -> Result<DeviceArrayF32, CudaJsaError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        let d_prices_tm =
            DeviceBuffer::from_slice(data_tm_f32).map_err(|e| CudaJsaError::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaJsaError::Cuda(e.to_string()))?;
        let d_warm = DeviceBuffer::from_slice(&prepared.warm_indices)
            .map_err(|e| CudaJsaError::Cuda(e.to_string()))?;
        let mut d_out_tm: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)
                .map_err(|e| CudaJsaError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_first,
            &d_warm,
            prepared.period,
            num_series,
            series_len,
            &mut d_out_tm,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out_tm,
            rows: series_len,
            cols: num_series,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn jsa_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        d_warm_indices: &DeviceBuffer<i32>,
        period: i32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJsaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaJsaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if period <= 0 {
            return Err(CudaJsaError::InvalidInput("period must be positive".into()));
        }
        let expected = num_series * series_len;
        if d_prices_tm.len() != expected || d_out_tm.len() != expected {
            return Err(CudaJsaError::InvalidInput(
                "time-major buffer length mismatch".into(),
            ));
        }
        if d_first_valids.len() != num_series || d_warm_indices.len() != num_series {
            return Err(CudaJsaError::InvalidInput(
                "first_valid or warm index buffer length mismatch".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            d_warm_indices,
            period,
            num_series,
            series_len,
            d_out_tm,
        )
    }

    pub fn jsa_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &JsaParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaJsaError> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaJsaError::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.jsa_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        handle
            .buf
            .copy_to(out_tm)
            .map_err(|e| CudaJsaError::Cuda(e.to_string()))
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_warm: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJsaError> {
        let block: BlockSize = (256, 1, 1).into();
        let grid: GridSize = (n_combos as u32, 1, 1).into();

        let func = self
            .module
            .get_function("jsa_batch_f32")
            .map_err(|e| CudaJsaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut warm_ptr = d_warm.as_device_ptr().as_raw();
            let mut first_valid_i = first_valid as i32;
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 7] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut warm_ptr as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaJsaError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaJsaError::Cuda(e.to_string()))
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        d_warm: &DeviceBuffer<i32>,
        period: i32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaJsaError> {
        let block: BlockSize = (256, 1, 1).into();
        let grid: GridSize = (num_series as u32, 1, 1).into();

        let func = self
            .module
            .get_function("jsa_many_series_one_param_f32")
            .map_err(|e| CudaJsaError::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut warm_ptr = d_warm.as_device_ptr().as_raw();
            let mut period_i = period;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 7] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut warm_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaJsaError::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaJsaError::Cuda(e.to_string()))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &JsaBatchRange,
    ) -> Result<PreparedJsaBatch, CudaJsaError> {
        if data_f32.is_empty() {
            return Err(CudaJsaError::InvalidInput("input data is empty".into()));
        }

        let combos = expand_periods(sweep);
        if combos.is_empty() {
            return Err(CudaJsaError::InvalidInput(
                "no period combinations provided".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| v.is_finite())
            .ok_or_else(|| CudaJsaError::InvalidInput("all values are NaN".into()))?;

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut warm_indices = Vec::with_capacity(combos.len());

        for params in &combos {
            let period = params.period.unwrap_or(0);
            if period == 0 {
                return Err(CudaJsaError::InvalidInput("period must be positive".into()));
            }
            if series_len - first_valid < period {
                return Err(CudaJsaError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, have {}",
                    period,
                    series_len - first_valid
                )));
            }
            let period_i32 = i32::try_from(period)
                .map_err(|_| CudaJsaError::InvalidInput("period exceeds i32".into()))?;
            let warm = first_valid + period;
            periods_i32.push(period_i32);
            warm_indices.push(warm as i32);
        }

        Ok(PreparedJsaBatch {
            combos,
            first_valid,
            series_len,
            periods_i32,
            warm_indices,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &JsaParams,
    ) -> Result<PreparedJsaManySeries, CudaJsaError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaJsaError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaJsaError::InvalidInput(
                "time-major slice length mismatch".into(),
            ));
        }

        let period = params.period.unwrap_or(0);
        if period == 0 {
            return Err(CudaJsaError::InvalidInput("period must be positive".into()));
        }
        if period > series_len {
            return Err(CudaJsaError::InvalidInput(format!(
                "period {} exceeds series length {}",
                period, series_len
            )));
        }

        let period_i32 = i32::try_from(period)
            .map_err(|_| CudaJsaError::InvalidInput("period exceeds i32".into()))?;

        let mut first_valids = Vec::with_capacity(num_series);
        let mut warm_indices = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut fv = None;
            for t in 0..series_len {
                let value = data_tm_f32[t * num_series + series];
                if value.is_finite() {
                    fv = Some(t);
                    break;
                }
            }
            let fv = fv.ok_or_else(|| {
                CudaJsaError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if series_len - fv < period {
                return Err(CudaJsaError::InvalidInput(format!(
                    "series {} does not have enough valid data (needed >= {}, have {})",
                    series,
                    period,
                    series_len - fv
                )));
            }
            first_valids.push(fv as i32);
            warm_indices.push((fv + period) as i32);
        }

        Ok(PreparedJsaManySeries {
            first_valids,
            warm_indices,
            period: period_i32,
        })
    }
}

fn expand_periods(range: &JsaBatchRange) -> Vec<JsaParams> {
    let (start, end, step) = range.period;
    if start > end {
        return Vec::new();
    }
    if step == 0 || start == end {
        return vec![JsaParams {
            period: Some(start),
        }];
    }

    let mut out = Vec::new();
    let mut value = start;
    while value <= end {
        out.push(JsaParams {
            period: Some(value),
        });
        value = value.saturating_add(step);
        if step == 0 {
            break;
        }
    }
    out
}
