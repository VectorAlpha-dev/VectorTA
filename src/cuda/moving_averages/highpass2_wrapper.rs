//! CUDA support for the two-pole high-pass filter (highpass_2_pole).
//!
//! Provides zero-copy device entry points that mirror the ALMA CUDA surface:
//! - one-series × many-parameter sweeps via `highpass2_batch_f32`
//! - time-major many-series × one-parameter execution via
//!   `highpass2_many_series_one_param_f32`
//!
//! Coefficients derived from `(period, k)` are precomputed on the host so each
//! output row reuses shared data instead of recomputing expensive trig values
//! on the device.

#![cfg(feature = "cuda")]

use super::alma_wrapper::DeviceArrayF32;
use crate::indicators::moving_averages::highpass_2_pole::{HighPass2BatchRange, HighPass2Params};
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
pub enum CudaHighPass2Error {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaHighPass2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaHighPass2Error::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaHighPass2Error::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaHighPass2Error {}

pub struct CudaHighPass2 {
    module: Module,
    stream: Stream,
    _context: Context,
}

struct PreparedHighPass2Batch {
    combos: Vec<HighPass2Params>,
    first_valid: usize,
    series_len: usize,
    periods_i32: Vec<i32>,
    c_vals: Vec<f32>,
    cm2_vals: Vec<f32>,
    two_1m_vals: Vec<f32>,
    neg_oma_sq_vals: Vec<f32>,
}

struct PreparedHighPass2ManySeries {
    first_valids: Vec<i32>,
    period: i32,
    c: f32,
    cm2: f32,
    two_1m: f32,
    neg_oma_sq: f32,
}

impl CudaHighPass2 {
    pub fn new(device_id: usize) -> Result<Self, CudaHighPass2Error> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;
        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;

        let ptx = include_str!(concat!(env!("OUT_DIR"), "/highpass2_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn highpass2_batch_dev(
        &self,
        data_f32: &[f32],
        sweep: &HighPass2BatchRange,
    ) -> Result<DeviceArrayF32, CudaHighPass2Error> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        let n_combos = prepared.combos.len();

        let d_prices = DeviceBuffer::from_slice(data_f32)
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&prepared.periods_i32)
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;
        let d_c = DeviceBuffer::from_slice(&prepared.c_vals)
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;
        let d_cm2 = DeviceBuffer::from_slice(&prepared.cm2_vals)
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;
        let d_two = DeviceBuffer::from_slice(&prepared.two_1m_vals)
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;
        let d_neg = DeviceBuffer::from_slice(&prepared.neg_oma_sq_vals)
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(prepared.series_len * n_combos)
                .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?
        };

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_c,
            &d_cm2,
            &d_two,
            &d_neg,
            prepared.series_len,
            n_combos,
            prepared.first_valid,
            &mut d_out,
        )?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: prepared.series_len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn highpass2_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_c: &DeviceBuffer<f32>,
        d_cm2: &DeviceBuffer<f32>,
        d_two_1m: &DeviceBuffer<f32>,
        d_neg_oma_sq: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHighPass2Error> {
        if series_len == 0 {
            return Err(CudaHighPass2Error::InvalidInput(
                "series_len must be positive".into(),
            ));
        }
        if n_combos == 0 {
            return Err(CudaHighPass2Error::InvalidInput(
                "n_combos must be positive".into(),
            ));
        }
        if first_valid >= series_len {
            return Err(CudaHighPass2Error::InvalidInput(format!(
                "first_valid {} out of range for len {}",
                first_valid, series_len
            )));
        }
        let expected = n_combos;
        if d_periods.len() != expected
            || d_c.len() != expected
            || d_cm2.len() != expected
            || d_two_1m.len() != expected
            || d_neg_oma_sq.len() != expected
        {
            return Err(CudaHighPass2Error::InvalidInput(
                "device buffer length mismatch".into(),
            ));
        }
        if d_prices.len() != series_len {
            return Err(CudaHighPass2Error::InvalidInput(
                "prices length must equal series_len".into(),
            ));
        }
        if d_out.len() != series_len * expected {
            return Err(CudaHighPass2Error::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_c,
            d_cm2,
            d_two_1m,
            d_neg_oma_sq,
            series_len,
            n_combos,
            first_valid,
            d_out,
        )
    }

    pub fn highpass2_batch_into_host_f32(
        &self,
        data_f32: &[f32],
        sweep: &HighPass2BatchRange,
        out_flat: &mut [f32],
    ) -> Result<(), CudaHighPass2Error> {
        let prepared = Self::prepare_batch_inputs(data_f32, sweep)?;
        if out_flat.len() != prepared.series_len * prepared.combos.len() {
            return Err(CudaHighPass2Error::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.highpass2_batch_dev(data_f32, sweep)?;
        handle
            .buf
            .copy_to(out_flat)
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))
    }

    pub fn highpass2_many_series_one_param_time_major_dev(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &HighPass2Params,
    ) -> Result<DeviceArrayF32, CudaHighPass2Error> {
        let prepared =
            Self::prepare_many_series_inputs(data_tm_f32, num_series, series_len, params)?;

        let d_prices = DeviceBuffer::from_slice(data_tm_f32)
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;
        let d_first = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::uninitialized(num_series * series_len)
                .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?
        };

        self.launch_many_series_kernel(
            &d_prices,
            &d_first,
            prepared.period,
            prepared.c,
            prepared.cm2,
            prepared.two_1m,
            prepared.neg_oma_sq,
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
    pub fn highpass2_many_series_one_param_time_major_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        c: f32,
        cm2: f32,
        two_1m: f32,
        neg_oma_sq: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHighPass2Error> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaHighPass2Error::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if d_prices_tm.len() != num_series * series_len {
            return Err(CudaHighPass2Error::InvalidInput(
                "prices_tm length mismatch".into(),
            ));
        }
        if d_out_tm.len() != num_series * series_len {
            return Err(CudaHighPass2Error::InvalidInput(
                "output buffer length mismatch".into(),
            ));
        }
        if d_first_valids.len() != num_series {
            return Err(CudaHighPass2Error::InvalidInput(
                "first_valids length mismatch".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_first_valids,
            period,
            c,
            cm2,
            two_1m,
            neg_oma_sq,
            num_series,
            series_len,
            d_out_tm,
        )
    }

    pub fn highpass2_many_series_one_param_time_major_into_host_f32(
        &self,
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &HighPass2Params,
        out_tm: &mut [f32],
    ) -> Result<(), CudaHighPass2Error> {
        if out_tm.len() != num_series * series_len {
            return Err(CudaHighPass2Error::InvalidInput(
                "output slice length mismatch".into(),
            ));
        }
        let handle = self.highpass2_many_series_one_param_time_major_dev(
            data_tm_f32,
            num_series,
            series_len,
            params,
        )?;
        handle
            .buf
            .copy_to(out_tm)
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_c: &DeviceBuffer<f32>,
        d_cm2: &DeviceBuffer<f32>,
        d_two_1m: &DeviceBuffer<f32>,
        d_neg_oma_sq: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHighPass2Error> {
        let block: BlockSize = (256, 1, 1).into();
        let grid: GridSize = (n_combos as u32, 1, 1).into();

        let func = self
            .module
            .get_function("highpass2_batch_f32")
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut c_ptr = d_c.as_device_ptr().as_raw();
            let mut cm2_ptr = d_cm2.as_device_ptr().as_raw();
            let mut two_ptr = d_two_1m.as_device_ptr().as_raw();
            let mut neg_ptr = d_neg_oma_sq.as_device_ptr().as_raw();
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut c_ptr as *mut _ as *mut c_void,
                &mut cm2_ptr as *mut _ as *mut c_void,
                &mut two_ptr as *mut _ as *mut c_void,
                &mut neg_ptr as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_first_valids: &DeviceBuffer<i32>,
        period: i32,
        c: f32,
        cm2: f32,
        two_1m: f32,
        neg_oma_sq: f32,
        num_series: usize,
        series_len: usize,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaHighPass2Error> {
        let block: BlockSize = (256, 1, 1).into();
        let grid: GridSize = (num_series as u32, 1, 1).into();

        let func = self
            .module
            .get_function("highpass2_many_series_one_param_f32")
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut first_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut period_i = period;
            let mut c_val = c;
            let mut cm2_val = cm2;
            let mut two_val = two_1m;
            let mut neg_val = neg_oma_sq;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let mut args: [*mut c_void; 10] = [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut first_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut c_val as *mut _ as *mut c_void,
                &mut cm2_val as *mut _ as *mut c_void,
                &mut two_val as *mut _ as *mut c_void,
                &mut neg_val as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, &mut args)
                .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| CudaHighPass2Error::Cuda(e.to_string()))
    }

    fn prepare_batch_inputs(
        data_f32: &[f32],
        sweep: &HighPass2BatchRange,
    ) -> Result<PreparedHighPass2Batch, CudaHighPass2Error> {
        if data_f32.is_empty() {
            return Err(CudaHighPass2Error::InvalidInput(
                "input data is empty".into(),
            ));
        }
        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaHighPass2Error::InvalidInput(
                "no parameter combinations provided".into(),
            ));
        }

        let first_valid = data_f32
            .iter()
            .position(|v| !v.is_nan())
            .ok_or_else(|| CudaHighPass2Error::InvalidInput("all values are NaN".into()))?;
        let series_len = data_f32.len();

        let mut periods_i32 = Vec::with_capacity(combos.len());
        let mut c_vals = Vec::with_capacity(combos.len());
        let mut cm2_vals = Vec::with_capacity(combos.len());
        let mut two_vals = Vec::with_capacity(combos.len());
        let mut neg_vals = Vec::with_capacity(combos.len());

        for params in &combos {
            let period = params.period.unwrap_or(0);
            let k = params.k.unwrap_or(0.707);
            if period < 2 {
                return Err(CudaHighPass2Error::InvalidInput(
                    "period must be >= 2".into(),
                ));
            }
            if !(k > 0.0) || !k.is_finite() {
                return Err(CudaHighPass2Error::InvalidInput(format!(
                    "invalid k: {}",
                    k
                )));
            }
            if series_len - first_valid < period {
                return Err(CudaHighPass2Error::InvalidInput(format!(
                    "not enough valid data: needed >= {}, have {}",
                    period,
                    series_len - first_valid
                )));
            }

            let coeffs = compute_coefficients(period, k);
            periods_i32.push(period as i32);
            c_vals.push(coeffs.c);
            cm2_vals.push(coeffs.cm2);
            two_vals.push(coeffs.two_1m);
            neg_vals.push(coeffs.neg_oma_sq);
        }

        Ok(PreparedHighPass2Batch {
            combos,
            first_valid,
            series_len,
            periods_i32,
            c_vals,
            cm2_vals,
            two_1m_vals: two_vals,
            neg_oma_sq_vals: neg_vals,
        })
    }

    fn prepare_many_series_inputs(
        data_tm_f32: &[f32],
        num_series: usize,
        series_len: usize,
        params: &HighPass2Params,
    ) -> Result<PreparedHighPass2ManySeries, CudaHighPass2Error> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaHighPass2Error::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if data_tm_f32.len() != num_series * series_len {
            return Err(CudaHighPass2Error::InvalidInput(format!(
                "time-major slice length mismatch: got {}, expected {}",
                data_tm_f32.len(),
                num_series * series_len
            )));
        }

        let period = params.period.unwrap_or(48) as i32;
        let k = params.k.unwrap_or(0.707);
        if period < 2 {
            return Err(CudaHighPass2Error::InvalidInput(
                "period must be >= 2".into(),
            ));
        }
        if !(k > 0.0) || !k.is_finite() {
            return Err(CudaHighPass2Error::InvalidInput(format!(
                "invalid k: {}",
                k
            )));
        }

        let needed = period as usize;
        let mut first_valids = Vec::with_capacity(num_series);
        for series in 0..num_series {
            let mut first_valid: Option<usize> = None;
            for t in 0..series_len {
                let value = data_tm_f32[t * num_series + series];
                if value.is_finite() {
                    first_valid = Some(t);
                    break;
                }
            }
            let fv = first_valid.ok_or_else(|| {
                CudaHighPass2Error::InvalidInput(format!("series {} is entirely NaN", series))
            })?;
            if series_len - fv < needed {
                return Err(CudaHighPass2Error::InvalidInput(format!(
                    "series {} not enough valid data (needed >= {}, valid = {})",
                    series,
                    needed,
                    series_len - fv
                )));
            }
            first_valids.push(fv as i32);
        }

        let coeffs = compute_coefficients(period as usize, k);

        Ok(PreparedHighPass2ManySeries {
            first_valids,
            period,
            c: coeffs.c,
            cm2: coeffs.cm2,
            two_1m: coeffs.two_1m,
            neg_oma_sq: coeffs.neg_oma_sq,
        })
    }
}

// ---------- Bench profiles ----------

pub mod benches {
    use super::*;
    use crate::define_ma_period_benches;

    define_ma_period_benches!(
        highpass2_benches,
        CudaHighPass2,
        crate::indicators::moving_averages::highpass_2_pole::HighPass2BatchRange,
        crate::indicators::moving_averages::highpass_2_pole::HighPass2Params,
        highpass2_batch_dev,
        highpass2_many_series_one_param_time_major_dev,
        crate::indicators::moving_averages::highpass_2_pole::HighPass2BatchRange { period: (10, 10 + PARAM_SWEEP - 1, 1), k: (0.5, 0.5, 0.0) },
        crate::indicators::moving_averages::highpass_2_pole::HighPass2Params { period: Some(64), k: Some(0.5) },
        "highpass2",
        "highpass2"
    );
    pub use highpass2_benches::bench_profiles;
}

struct Coefficients {
    c: f32,
    cm2: f32,
    two_1m: f32,
    neg_oma_sq: f32,
}

fn compute_coefficients(period: usize, k: f64) -> Coefficients {
    use std::f64::consts::PI;

    let theta = 2.0 * PI * k / period as f64;
    let sin_v = theta.sin();
    let cos_v = theta.cos();
    let alpha = 1.0 + ((sin_v - 1.0) / cos_v);
    let c = (1.0 - 0.5 * alpha).powi(2);
    let cm2 = -2.0 * c;
    let one_minus_alpha = 1.0 - alpha;
    let two_1m = 2.0 * one_minus_alpha;
    let neg_oma_sq = -(one_minus_alpha * one_minus_alpha);

    Coefficients {
        c: c as f32,
        cm2: cm2 as f32,
        two_1m: two_1m as f32,
        neg_oma_sq: neg_oma_sq as f32,
    }
}

fn expand_grid(range: &HighPass2BatchRange) -> Vec<HighPass2Params> {
    fn axis_usize(axis: (usize, usize, usize)) -> Vec<usize> {
        let (start, end, step) = axis;
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    fn axis_f64(axis: (f64, f64, f64)) -> Vec<f64> {
        let (start, end, step) = axis;
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut values = Vec::new();
        let mut current = start;
        while current <= end + 1e-12 {
            values.push(current);
            current += step;
        }
        values
    }

    let periods = axis_usize(range.period);
    let ks = axis_f64(range.k);
    let mut out = Vec::with_capacity(periods.len() * ks.len());
    for &p in &periods {
        for &k in &ks {
            out.push(HighPass2Params {
                period: Some(p),
                k: Some(k),
            });
        }
    }
    out
}
