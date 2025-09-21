//! CUDA support for the Tilson T3 moving average indicator.
//!
//! Mirrors the CPU batching API by preparing per-combination coefficients on the
//! host and executing the sequential six-stage EMA recurrence on the device in
//! FP32. Includes both the classic "one series × many params" flow and a
//! time-major "many series × one param" variant for parity with ALMA.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::tilson::{TilsonBatchRange, TilsonParams};
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
pub enum CudaTilsonError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaTilsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaTilsonError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaTilsonError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaTilsonError {}

pub struct CudaTilson {
    module: Module,
    stream: Stream,
    _context: Context,
}

struct PreparedTilsonBatch {
    combos: Vec<TilsonParams>,
    first_valid: usize,
    series_len: usize,
    periods_i32: Vec<i32>,
    ks_f32: Vec<f32>,
    c1_f32: Vec<f32>,
    c2_f32: Vec<f32>,
    c3_f32: Vec<f32>,
    c4_f32: Vec<f32>,
    lookbacks_i32: Vec<i32>,
}

struct PreparedTilsonManySeries {
    first_valids: Vec<i32>,
    period: usize,
    k_f32: f32,
    c1_f32: f32,
    c2_f32: f32,
    c3_f32: f32,
    c4_f32: f32,
    lookback_i32: i32,
}

impl CudaTilson {
    pub fn new(device_id: usize) -> Result<Self, CudaTilsonError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/tilson_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn tilson_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &TilsonBatchRange,
    ) -> Result<DeviceArrayF32, CudaTilsonError> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        let d_prices =
            DeviceBuffer::from_slice(data_f32).map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&prepared.periods_i32)
            .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        let d_ks = DeviceBuffer::from_slice(&prepared.ks_f32)
            .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        let d_c1 = DeviceBuffer::from_slice(&prepared.c1_f32)
            .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        let d_c2 = DeviceBuffer::from_slice(&prepared.c2_f32)
            .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        let d_c3 = DeviceBuffer::from_slice(&prepared.c3_f32)
            .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        let d_c4 = DeviceBuffer::from_slice(&prepared.c4_f32)
            .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        let d_lookbacks = DeviceBuffer::from_slice(&prepared.lookbacks_i32)
            .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_ks,
            &d_c1,
            &d_c2,
            &d_c3,
            &d_c4,
            &d_lookbacks,
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
    pub fn tilson_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_ks: &DeviceBuffer<f32>,
        d_c1: &DeviceBuffer<f32>,
        d_c2: &DeviceBuffer<f32>,
        d_c3: &DeviceBuffer<f32>,
        d_c4: &DeviceBuffer<f32>,
        d_lookbacks: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTilsonError> {
        if series_len == 0 {
            return Err(CudaTilsonError::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaTilsonError::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, series_len
            )));
        }
        if n_combos == 0 {
            return Err(CudaTilsonError::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if d_periods.len() != n_combos
            || d_ks.len() != n_combos
            || d_c1.len() != n_combos
            || d_c2.len() != n_combos
            || d_c3.len() != n_combos
            || d_c4.len() != n_combos
            || d_lookbacks.len() != n_combos
        {
            return Err(CudaTilsonError::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }
        if d_out.len() != n_combos * series_len {
            return Err(CudaTilsonError::InvalidInput(
                "output buffer has incorrect length".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_ks,
            d_c1,
            d_c2,
            d_c3,
            d_c4,
            d_lookbacks,
            series_len,
            first_valid,
            n_combos,
            d_out,
        )
    }

    pub fn tilson_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &TilsonParams,
    ) -> Result<DeviceArrayF32, CudaTilsonError> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)
                .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first_valids,
            prepared.period,
            prepared.k_f32,
            prepared.c1_f32,
            prepared.c2_f32,
            prepared.c3_f32,
            prepared.c4_f32,
            prepared.lookback_i32,
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
    pub fn tilson_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        k: f32,
        c1: f32,
        c2: f32,
        c3: f32,
        c4: f32,
        lookback: i32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTilsonError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaTilsonError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if d_first_valids.len() != num_series {
            return Err(CudaTilsonError::InvalidInput(
                "first_valids length mismatch".into(),
            ));
        }
        if d_out_tm.len() != num_series * series_len {
            return Err(CudaTilsonError::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            period,
            k,
            c1,
            c2,
            c3,
            c4,
            lookback,
            num_series,
            series_len,
            d_out_tm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_ks: &DeviceBuffer<f32>,
        d_c1: &DeviceBuffer<f32>,
        d_c2: &DeviceBuffer<f32>,
        d_c3: &DeviceBuffer<f32>,
        d_c4: &DeviceBuffer<f32>,
        d_lookbacks: &DeviceBuffer<i32>,
        series_len: usize,
        first_valid: usize,
        n_combos: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTilsonError> {
        if n_combos == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("tilson_batch_f32")
            .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (256, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut ks_ptr = d_ks.as_device_ptr().as_raw();
            let mut c1_ptr = d_c1.as_device_ptr().as_raw();
            let mut c2_ptr = d_c2.as_device_ptr().as_raw();
            let mut c3_ptr = d_c3.as_device_ptr().as_raw();
            let mut c4_ptr = d_c4.as_device_ptr().as_raw();
            let mut lookbacks_ptr = d_lookbacks.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut first_valid_i = first_valid as i32;
            let mut combos_i = n_combos as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut ks_ptr as *mut _ as *mut c_void,
                &mut c1_ptr as *mut _ as *mut c_void,
                &mut c2_ptr as *mut _ as *mut c_void,
                &mut c3_ptr as *mut _ as *mut c_void,
                &mut c4_ptr as *mut _ as *mut c_void,
                &mut lookbacks_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: usize,
        k: f32,
        c1: f32,
        c2: f32,
        c3: f32,
        c4: f32,
        lookback: i32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaTilsonError> {
        if num_series == 0 || series_len == 0 {
            return Ok(());
        }

        let func = self
            .module
            .get_function("tilson_many_series_one_param_f32")
            .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;

        let grid: GridSize = (1, num_series as u32, 1).into();
        let block: BlockSize = (128, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut k_f = k;
            let mut c1_f = c1;
            let mut c2_f = c2;
            let mut c3_f = c3;
            let mut c4_f = c4;
            let mut lookback_i = lookback;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut k_f as *mut _ as *mut c_void,
                &mut c1_f as *mut _ as *mut c_void,
                &mut c2_f as *mut _ as *mut c_void,
                &mut c3_f as *mut _ as *mut c_void,
                &mut c4_f as *mut _ as *mut c_void,
                &mut lookback_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaTilsonError::Cuda(e.to_string()))?;
        }

        Ok(())
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &TilsonBatchRange,
    ) -> Result<PreparedTilsonBatch, CudaTilsonError> {
        if data_f32.is_empty() {
            return Err(CudaTilsonError::InvalidInput("input data is empty".into()));
        }

        let combos = expand_combos(sweep);
        if combos.is_empty() {
            return Err(CudaTilsonError::InvalidInput(
                "no parameter combinations provided".into(),
            ));
        }

        let series_len = data_f32.len();
        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaTilsonError::InvalidInput("all values are NaN".into()))?;

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut ks_f32 = Vec::with_capacity(combos.len());
        let mut c1_f32 = Vec::with_capacity(combos.len());
        let mut c2_f32 = Vec::with_capacity(combos.len());
        let mut c3_f32 = Vec::with_capacity(combos.len());
        let mut c4_f32 = Vec::with_capacity(combos.len());
        let mut lookbacks_i32 = Vec::with_capacity(combos.len());

        for params in &combos {
            let period = params.period.unwrap_or(0);
            let volume_factor = params.volume_factor.unwrap_or(0.0);
            if period == 0 {
                return Err(CudaTilsonError::InvalidInput(
                    "period must be positive".into(),
                ));
            }
            if !volume_factor.is_finite() {
                return Err(CudaTilsonError::InvalidInput(
                    "volume_factor must be finite".into(),
                ));
            }
            if period > i32::MAX as usize {
                return Err(CudaTilsonError::InvalidInput(
                    "period exceeds CUDA i32 range".into(),
                ));
            }
            let lookback = 6usize
                .checked_mul(period.saturating_sub(1))
                .ok_or_else(|| CudaTilsonError::InvalidInput("lookback overflow".into()))?;
            if lookback > i32::MAX as usize {
                return Err(CudaTilsonError::InvalidInput(
                    "lookback exceeds CUDA i32 range".into(),
                ));
            }

            if first_valid + lookback >= series_len {
                return Err(CudaTilsonError::InvalidInput(format!(
                    "not enough valid data: need >= {}, have {}",
                    lookback + 1,
                    series_len - first_valid
                )));
            }
            if first_valid + period > series_len {
                return Err(CudaTilsonError::InvalidInput(
                    "period exceeds remaining data".into(),
                ));
            }

            let k = 2.0f32 / (period as f32 + 1.0f32);
            let vf = volume_factor as f32;
            let temp = vf * vf;
            let c1 = -(temp * vf);
            let c2 = 3.0f32 * (temp - c1);
            let c3 = -6.0f32 * temp - 3.0f32 * (vf - c1);
            let c4 = 1.0f32 + 3.0f32 * vf - c1 + 3.0f32 * temp;

            periods_i32.push(period as i32);
            ks_f32.push(k);
            c1_f32.push(c1);
            c2_f32.push(c2);
            c3_f32.push(c3);
            c4_f32.push(c4);
            lookbacks_i32.push(lookback as i32);
            for coeff in [c1, c2, c3, c4] {
                if !coeff.is_finite() {
                    return Err(CudaTilsonError::InvalidInput(
                        "computed coefficient is not finite".into(),
                    ));
                }
            }
        }

        Ok(PreparedTilsonBatch {
            combos,
            first_valid,
            series_len,
            periods_i32,
            ks_f32,
            c1_f32,
            c2_f32,
            c3_f32,
            c4_f32,
            lookbacks_i32,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &TilsonParams,
    ) -> Result<PreparedTilsonManySeries, CudaTilsonError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaTilsonError::InvalidInput(
                "series dimensions must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaTilsonError::InvalidInput(
                "time-major buffer length mismatch".into(),
            ));
        }

        let period = params.period.unwrap_or(5);
        let v_factor = params.volume_factor.unwrap_or(0.0);
        if period == 0 {
            return Err(CudaTilsonError::InvalidInput(
                "period must be positive".into(),
            ));
        }
        if !v_factor.is_finite() {
            return Err(CudaTilsonError::InvalidInput(
                "volume_factor must be finite".into(),
            ));
        }
        if period > i32::MAX as usize {
            return Err(CudaTilsonError::InvalidInput(
                "period exceeds CUDA i32 range".into(),
            ));
        }
        let lookback = 6usize
            .checked_mul(period.saturating_sub(1))
            .ok_or_else(|| CudaTilsonError::InvalidInput("lookback overflow".into()))?;
        if lookback > i32::MAX as usize {
            return Err(CudaTilsonError::InvalidInput(
                "lookback exceeds CUDA i32 range".into(),
            ));
        }

        let stride = num_series;
        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut fv_opt = None;
            for t in 0..series_len {
                let value = data_tm_f32[t * stride + series];
                if !value.is_nan() {
                    fv_opt = Some(t);
                    break;
                }
            }
            let fv = fv_opt.ok_or_else(|| {
                CudaTilsonError::InvalidInput(format!("series {} contains only NaNs", series))
            })?;
            if series_len - fv <= lookback {
                return Err(CudaTilsonError::InvalidInput(format!(
                    "series {} not enough valid data (need >= {}, have {})",
                    series,
                    lookback + 1,
                    series_len - fv
                )));
            }
            first_valids.push(fv as i32);
        }

        let k = 2.0f32 / (period as f32 + 1.0f32);
        let vf = v_factor as f32;
        let temp = vf * vf;
        let c1 = -(temp * vf);
        let c2 = 3.0f32 * (temp - c1);
        let c3 = -6.0f32 * temp - 3.0f32 * (vf - c1);
        let c4 = 1.0f32 + 3.0f32 * vf - c1 + 3.0f32 * temp;

        for coeff in [c1, c2, c3, c4] {
            if !coeff.is_finite() {
                return Err(CudaTilsonError::InvalidInput(
                    "computed coefficient is not finite".into(),
                ));
            }
        }

        Ok(PreparedTilsonManySeries {
            first_valids,
            period,
            k_f32: k,
            c1_f32: c1,
            c2_f32: c2,
            c3_f32: c3,
            c4_f32: c4,
            lookback_i32: lookback as i32,
        })
    }
}

fn expand_combos(range: &TilsonBatchRange) -> Vec<TilsonParams> {
    fn axis_usize(axis: (usize, usize, usize)) -> Vec<usize> {
        let (start, end, step) = axis;
        if step == 0 || start == end {
            return vec![start];
        }
        let mut out = Vec::new();
        let mut value = start;
        while value <= end {
            out.push(value);
            value = value.saturating_add(step);
        }
        out
    }

    fn axis_f64(axis: (f64, f64, f64)) -> Vec<f64> {
        let (start, end, step) = axis;
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut out = Vec::new();
        let mut value = start;
        while value <= end + 1e-12 {
            out.push(value);
            value += step;
        }
        out
    }

    let periods = axis_usize(range.period);
    let volume_factors = axis_f64(range.volume_factor);
    let mut combos = Vec::with_capacity(periods.len() * volume_factors.len());
    for &period in &periods {
        for &vf in &volume_factors {
            combos.push(TilsonParams {
                period: Some(period),
                volume_factor: Some(vf),
            });
        }
    }
    combos
}
