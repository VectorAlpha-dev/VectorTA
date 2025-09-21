//! CUDA scaffolding for the Gaussian moving average kernels.
//!
//! This mirrors the scalar Gaussian filter by precomputing the cascaded single
//! pole coefficients on the host, uploading them once, and executing the
//! sequential recurrence per parameter combination (or per series) entirely on
//! the GPU. The wrapper exposes zero-copy device handles plus host-copy helper
//! methods to follow the conventions established by the other CUDA indicators.

#![cfg(feature = "cuda")]

use super::DeviceArrayF32;
use crate::indicators::moving_averages::gaussian::{GaussianBatchRange, GaussianParams};
use cust::context::Context;
use cust::device::Device;
use cust::function::{BlockSize, GridSize};
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::sys as cu;
use std::env;
use std::ffi::c_void;
use std::fmt;

const COEFF_STRIDE: usize = 5; // coefficient slots per combo (max poles = 4)

#[derive(Debug)]
pub enum CudaGaussianError {
    Cuda(String),
    InvalidInput(String),
}

impl fmt::Display for CudaGaussianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaGaussianError::Cuda(e) => write!(f, "CUDA error: {}", e),
            CudaGaussianError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl std::error::Error for CudaGaussianError {}

pub struct CudaGaussian {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaGaussian {
    pub fn new(device_id: usize) -> Result<Self, CudaGaussianError> {
        cust::init(CudaFlags::empty()).map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;

        let device = Device::get_device(device_id as u32)
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;
        let context = Context::new(device).map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/gaussian_kernel.ptx"));
        let module =
            Module::from_ptx(ptx, &[]).map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn gaussian_batch_dev(
        &self,
        prices: &[f32],
        sweep: &GaussianBatchRange,
    ) -> Result<DeviceArrayF32, CudaGaussianError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        self.run_batch_kernel(prices, &inputs)
    }

    pub fn gaussian_batch_into_host_f32(
        &self,
        prices: &[f32],
        sweep: &GaussianBatchRange,
        out: &mut [f32],
    ) -> Result<(usize, usize, Vec<GaussianParams>), CudaGaussianError> {
        let inputs = Self::prepare_batch_inputs(prices, sweep)?;
        let expected = inputs.series_len * inputs.combos.len();
        if out.len() != expected {
            return Err(CudaGaussianError::InvalidInput(format!(
                "output slice length mismatch: got {}, expected {}",
                out.len(),
                expected
            )));
        }

        let arr = self.run_batch_kernel(prices, &inputs)?;
        arr.buf
            .copy_to(out)
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;
        let BatchInputs { combos, .. } = inputs;
        Ok((arr.rows, arr.cols, combos))
    }

    pub fn gaussian_batch_device(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_poles: &DeviceBuffer<i32>,
        d_coeffs: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaGaussianError> {
        if series_len == 0 || n_combos == 0 {
            return Err(CudaGaussianError::InvalidInput(
                "series_len and n_combos must be positive".into(),
            ));
        }
        if series_len > i32::MAX as usize || n_combos > i32::MAX as usize {
            return Err(CudaGaussianError::InvalidInput(
                "arguments exceed kernel launch limits".into(),
            ));
        }

        self.launch_batch_kernel(
            d_prices,
            d_periods,
            d_poles,
            d_coeffs,
            series_len,
            n_combos,
            first_valid,
            d_out,
        )
    }

    pub fn gaussian_many_series_one_param_time_major_dev(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &GaussianParams,
    ) -> Result<DeviceArrayF32, CudaGaussianError> {
        let prepared = Self::prepare_many_series_inputs(prices_tm, cols, rows, params)?;
        self.run_many_series_kernel(prices_tm, cols, rows, params, &prepared)
    }

    pub fn gaussian_many_series_one_param_time_major_into_host_f32(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &GaussianParams,
        out_tm: &mut [f32],
    ) -> Result<(), CudaGaussianError> {
        if out_tm.len() != prices_tm.len() {
            return Err(CudaGaussianError::InvalidInput(format!(
                "output slice length mismatch: got {}, expected {}",
                out_tm.len(),
                prices_tm.len()
            )));
        }

        let prepared = Self::prepare_many_series_inputs(prices_tm, cols, rows, params)?;
        let arr = self.run_many_series_kernel(prices_tm, cols, rows, params, &prepared)?;
        arr.buf
            .copy_to(out_tm)
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;
        Ok(())
    }

    pub fn gaussian_many_series_one_param_device(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_coeffs: &DeviceBuffer<f32>,
        period: usize,
        poles: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaGaussianError> {
        if num_series == 0 || series_len == 0 {
            return Err(CudaGaussianError::InvalidInput(
                "num_series and series_len must be positive".into(),
            ));
        }
        if period < 2 || !(1..=4).contains(&poles) {
            return Err(CudaGaussianError::InvalidInput(
                "period >= 2 and poles within 1..=4 are required".into(),
            ));
        }
        if num_series > i32::MAX as usize || series_len > i32::MAX as usize {
            return Err(CudaGaussianError::InvalidInput(
                "dimensions exceed kernel launch limits".into(),
            ));
        }

        self.launch_many_series_kernel(
            d_prices_tm,
            d_coeffs,
            period,
            poles,
            num_series,
            series_len,
            d_first_valids,
            d_out_tm,
        )
    }

    fn run_batch_kernel(
        &self,
        prices: &[f32],
        inputs: &BatchInputs,
    ) -> Result<DeviceArrayF32, CudaGaussianError> {
        let n_combos = inputs.combos.len();
        let price_bytes = prices.len() * std::mem::size_of::<f32>();
        let period_bytes = inputs.periods.len() * std::mem::size_of::<i32>();
        let pole_bytes = inputs.poles.len() * std::mem::size_of::<i32>();
        let coeff_bytes = inputs.coeffs.len() * std::mem::size_of::<f32>();
        let out_bytes = n_combos * inputs.series_len * std::mem::size_of::<f32>();
        let required = price_bytes + period_bytes + pole_bytes + coeff_bytes + out_bytes;
        let headroom = 64 * 1024 * 1024; // 64 MiB safety margin
        if !Self::will_fit(required, headroom) {
            return Err(CudaGaussianError::InvalidInput(
                "insufficient free device memory for Gaussian batch kernel".into(),
            ));
        }

        let d_prices =
            DeviceBuffer::from_slice(prices).map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;
        let d_periods = DeviceBuffer::from_slice(&inputs.periods)
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;
        let d_poles = DeviceBuffer::from_slice(&inputs.poles)
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;
        let d_coeffs = DeviceBuffer::from_slice(&inputs.coeffs)
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> =
            unsafe { DeviceBuffer::uninitialized(inputs.series_len * n_combos) }
                .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;

        self.launch_batch_kernel(
            &d_prices,
            &d_periods,
            &d_poles,
            &d_coeffs,
            inputs.series_len,
            n_combos,
            inputs.first_valid,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows: n_combos,
            cols: inputs.series_len,
        })
    }

    fn run_many_series_kernel(
        &self,
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &GaussianParams,
        prepared: &ManySeriesInputs,
    ) -> Result<DeviceArrayF32, CudaGaussianError> {
        let price_bytes = prices_tm.len() * std::mem::size_of::<f32>();
        let coeff_bytes = prepared.coeffs.len() * std::mem::size_of::<f32>();
        let fv_bytes = prepared.first_valids.len() * std::mem::size_of::<i32>();
        let out_bytes = price_bytes;
        let required = price_bytes + coeff_bytes + fv_bytes + out_bytes;
        let headroom = 32 * 1024 * 1024; // lighter workload, smaller margin
        if !Self::will_fit(required, headroom) {
            return Err(CudaGaussianError::InvalidInput(
                "insufficient free device memory for Gaussian many-series kernel".into(),
            ));
        }

        let d_prices_tm = DeviceBuffer::from_slice(prices_tm)
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;
        let d_coeffs = DeviceBuffer::from_slice(&prepared.coeffs)
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;
        let d_first_valids = DeviceBuffer::from_slice(&prepared.first_valids)
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;
        let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(prices_tm.len()) }
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;

        let period = params.period.unwrap_or(14);
        let poles = params.poles.unwrap_or(4);

        self.launch_many_series_kernel(
            &d_prices_tm,
            &d_coeffs,
            period,
            poles,
            cols,
            rows,
            &d_first_valids,
            &mut d_out,
        )?;

        self.stream
            .synchronize()
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;

        Ok(DeviceArrayF32 {
            buf: d_out,
            rows,
            cols,
        })
    }

    fn launch_batch_kernel(
        &self,
        d_prices: &DeviceBuffer<f32>,
        d_periods: &DeviceBuffer<i32>,
        d_poles: &DeviceBuffer<i32>,
        d_coeffs: &DeviceBuffer<f32>,
        series_len: usize,
        n_combos: usize,
        first_valid: usize,
        d_out: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaGaussianError> {
        let func = self
            .module
            .get_function("gaussian_batch_f32")
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;

        let grid: GridSize = (n_combos as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices.as_device_ptr().as_raw();
            let mut periods_ptr = d_periods.as_device_ptr().as_raw();
            let mut poles_ptr = d_poles.as_device_ptr().as_raw();
            let mut coeffs_ptr = d_coeffs.as_device_ptr().as_raw();
            let mut coeff_stride_i = COEFF_STRIDE as i32;
            let mut series_len_i = series_len as i32;
            let mut combos_i = n_combos as i32;
            let mut first_valid_i = first_valid as i32;
            let mut out_ptr = d_out.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut periods_ptr as *mut _ as *mut c_void,
                &mut poles_ptr as *mut _ as *mut c_void,
                &mut coeffs_ptr as *mut _ as *mut c_void,
                &mut coeff_stride_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut combos_i as *mut _ as *mut c_void,
                &mut first_valid_i as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn launch_many_series_kernel(
        &self,
        d_prices_tm: &DeviceBuffer<f32>,
        d_coeffs: &DeviceBuffer<f32>,
        period: usize,
        poles: usize,
        num_series: usize,
        series_len: usize,
        d_first_valids: &DeviceBuffer<i32>,
        d_out_tm: &mut DeviceBuffer<f32>,
    ) -> Result<(), CudaGaussianError> {
        let func = self
            .module
            .get_function("gaussian_many_series_one_param_f32")
            .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;

        let grid: GridSize = (num_series as u32, 1, 1).into();
        let block: BlockSize = (1, 1, 1).into();

        unsafe {
            let mut prices_ptr = d_prices_tm.as_device_ptr().as_raw();
            let mut coeffs_ptr = d_coeffs.as_device_ptr().as_raw();
            let mut period_i = period as i32;
            let mut poles_i = poles as i32;
            let mut num_series_i = num_series as i32;
            let mut series_len_i = series_len as i32;
            let mut first_valids_ptr = d_first_valids.as_device_ptr().as_raw();
            let mut out_ptr = d_out_tm.as_device_ptr().as_raw();
            let args: &mut [*mut c_void] = &mut [
                &mut prices_ptr as *mut _ as *mut c_void,
                &mut coeffs_ptr as *mut _ as *mut c_void,
                &mut period_i as *mut _ as *mut c_void,
                &mut poles_i as *mut _ as *mut c_void,
                &mut num_series_i as *mut _ as *mut c_void,
                &mut series_len_i as *mut _ as *mut c_void,
                &mut first_valids_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
            ];
            self.stream
                .launch(&func, grid, block, 0, args)
                .map_err(|e| CudaGaussianError::Cuda(e.to_string()))?;
        }
        Ok(())
    }

    fn prepare_batch_inputs(
        prices: &[f32],
        sweep: &GaussianBatchRange,
    ) -> Result<BatchInputs, CudaGaussianError> {
        if prices.is_empty() {
            return Err(CudaGaussianError::InvalidInput("empty price series".into()));
        }

        let combos = expand_grid(sweep);
        if combos.is_empty() {
            return Err(CudaGaussianError::InvalidInput(
                "Gaussian sweep produced no parameter combinations".into(),
            ));
        }

        let series_len = prices.len();
        let first_valid = prices
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| CudaGaussianError::InvalidInput("all values are NaN".into()))?;

        let mut periods = Vec::with_capacity(combos.len());
        let mut poles = Vec::with_capacity(combos.len());
        let mut coeffs = Vec::with_capacity(combos.len() * COEFF_STRIDE);

        for prm in &combos {
            let period = prm.period.unwrap_or(14);
            let pole = prm.poles.unwrap_or(4);

            if period < 2 {
                return Err(CudaGaussianError::InvalidInput(format!(
                    "period must be >= 2 (got {})",
                    period
                )));
            }
            if !(1..=4).contains(&pole) {
                return Err(CudaGaussianError::InvalidInput(format!(
                    "poles must be in 1..=4 (got {})",
                    pole
                )));
            }
            if period > i32::MAX as usize {
                return Err(CudaGaussianError::InvalidInput(
                    "period exceeds i32::MAX".into(),
                ));
            }
            if series_len - first_valid < period {
                return Err(CudaGaussianError::InvalidInput(format!(
                    "not enough valid data: needed >= {}, valid = {}",
                    period,
                    series_len - first_valid
                )));
            }

            let coeff = compute_gaussian_coeffs(period, pole)?;
            periods.push(period as i32);
            poles.push(pole as i32);
            coeffs.extend_from_slice(&coeff);
        }

        Ok(BatchInputs {
            combos,
            periods,
            poles,
            coeffs,
            first_valid,
            series_len,
        })
    }

    fn prepare_many_series_inputs(
        prices_tm: &[f32],
        cols: usize,
        rows: usize,
        params: &GaussianParams,
    ) -> Result<ManySeriesInputs, CudaGaussianError> {
        if cols == 0 || rows == 0 {
            return Err(CudaGaussianError::InvalidInput(
                "matrix dimensions must be positive".into(),
            ));
        }
        if prices_tm.len() != cols * rows {
            return Err(CudaGaussianError::InvalidInput(
                "matrix shape mismatch for time-major layout".into(),
            ));
        }

        let period = params.period.unwrap_or(14);
        let poles = params.poles.unwrap_or(4);
        if period < 2 {
            return Err(CudaGaussianError::InvalidInput(format!(
                "period must be >= 2 (got {})",
                period
            )));
        }
        if !(1..=4).contains(&poles) {
            return Err(CudaGaussianError::InvalidInput(format!(
                "poles must be in 1..=4 (got {})",
                poles
            )));
        }

        let mut first_valids = vec![0i32; cols];
        for series_idx in 0..cols {
            let mut fv = None;
            for row in 0..rows {
                let idx = row * cols + series_idx;
                let price = prices_tm[idx];
                if !price.is_nan() {
                    fv = Some(row);
                    break;
                }
            }
            let val = fv.ok_or_else(|| {
                CudaGaussianError::InvalidInput(format!(
                    "series {} has no valid price values",
                    series_idx
                ))
            })?;
            if rows - val < period {
                return Err(CudaGaussianError::InvalidInput(format!(
                    "series {} lacks data: needed >= {}, valid = {}",
                    series_idx,
                    period,
                    rows - val
                )));
            }
            first_valids[series_idx] = val as i32;
        }

        let coeffs = compute_gaussian_coeffs(period, poles)?;
        Ok(ManySeriesInputs {
            first_valids,
            coeffs,
        })
    }

    #[inline]
    fn mem_check_enabled() -> bool {
        match env::var("CUDA_MEM_CHECK") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => true,
        }
    }

    #[inline]
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

    #[inline]
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
}

struct BatchInputs {
    combos: Vec<GaussianParams>,
    periods: Vec<i32>,
    poles: Vec<i32>,
    coeffs: Vec<f32>,
    first_valid: usize,
    series_len: usize,
}

struct ManySeriesInputs {
    first_valids: Vec<i32>,
    coeffs: [f32; COEFF_STRIDE],
}

fn compute_gaussian_coeffs(
    period: usize,
    poles: usize,
) -> Result<[f32; COEFF_STRIDE], CudaGaussianError> {
    use std::f64::consts::PI;

    if period < 2 {
        return Err(CudaGaussianError::InvalidInput(
            "period must be >= 2 for Gaussian coefficients".into(),
        ));
    }
    if !(1..=4).contains(&poles) {
        return Err(CudaGaussianError::InvalidInput(
            "poles must be within 1..=4 for Gaussian coefficients".into(),
        ));
    }

    let period_f = period as f64;
    let poles_f = poles as f64;

    let beta_num = 1.0 - (2.0 * PI / period_f).cos();
    let beta_den = (2.0f64).powf(1.0 / poles_f) - 1.0;
    if beta_den.abs() < 1e-12 {
        return Err(CudaGaussianError::InvalidInput(
            "beta denominator too small, coefficients unstable".into(),
        ));
    }
    let beta = beta_num / beta_den;
    let discr = beta * beta + 2.0 * beta;
    if discr < 0.0 {
        return Err(CudaGaussianError::InvalidInput(
            "negative discriminant while computing Gaussian alpha".into(),
        ));
    }
    let alpha = -beta + discr.sqrt();
    let one = 1.0 - alpha;

    let mut coeffs = [0f32; COEFF_STRIDE];
    match poles {
        1 => {
            coeffs[0] = alpha as f32;
            coeffs[1] = one as f32;
        }
        2 => {
            let one_sq = one * one;
            coeffs[0] = (alpha * alpha) as f32;
            coeffs[1] = (2.0 * one) as f32;
            coeffs[2] = (-one_sq) as f32;
        }
        3 => {
            let one_sq = one * one;
            coeffs[0] = (alpha * alpha * alpha) as f32;
            coeffs[1] = (3.0 * one) as f32;
            coeffs[2] = (-3.0 * one_sq) as f32;
            coeffs[3] = (one_sq * one) as f32;
        }
        4 => {
            let one_sq = one * one;
            let one_cu = one_sq * one;
            coeffs[0] = (alpha * alpha * alpha * alpha) as f32;
            coeffs[1] = (4.0 * one) as f32;
            coeffs[2] = (-6.0 * one_sq) as f32;
            coeffs[3] = (4.0 * one_cu) as f32;
            coeffs[4] = (-(one_cu * one)) as f32;
        }
        _ => unreachable!(),
    }

    if coeffs.iter().any(|c| !c.is_finite()) {
        return Err(CudaGaussianError::InvalidInput(
            "non-finite Gaussian coefficients produced".into(),
        ));
    }
    Ok(coeffs)
}

fn expand_grid(range: &GaussianBatchRange) -> Vec<GaussianParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        }
    }

    let periods = axis(range.period);
    let poles = axis(range.poles);
    let mut combos = Vec::with_capacity(periods.len() * poles.len());
    for &p in &periods {
        for &k in &poles {
            combos.push(GaussianParams {
                period: Some(p),
                poles: Some(k),
            });
        }
    }
    combos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_gaussian_coeffs_matches_known_values() {
        let coeffs = compute_gaussian_coeffs(10, 2).expect("coeffs");
        assert!(coeffs[0].is_finite());
        assert!(coeffs[1].is_finite());
        assert!(coeffs[2].is_finite());
    }
}
