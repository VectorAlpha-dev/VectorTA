#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaOtt;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::{make_device_array_py, DeviceArrayF32Py};
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};

use crate::indicators::moving_averages::{
    ema::{ema_with_kernel, EmaInput, EmaParams},
    linreg::{linreg_with_kernel, LinRegInput, LinRegParams},
    sma::{sma_with_kernel, SmaInput, SmaParams},
    wma::{wma_with_kernel, WmaInput, WmaParams},
    zlema::{zlema_with_kernel, ZlemaInput, ZlemaParams},
};
use crate::indicators::tsf::{tsf_with_kernel, TsfInput, TsfParams};

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use std::collections::HashMap;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for OttInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            OttData::Slice(slice) => slice,
            OttData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum OttData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct OttOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct OttParams {
    pub period: Option<usize>,
    pub percent: Option<f64>,
    pub ma_type: Option<String>,
}

impl Default for OttParams {
    fn default() -> Self {
        Self {
            period: Some(2),
            percent: Some(1.4),
            ma_type: Some("VAR".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OttInput<'a> {
    pub data: OttData<'a>,
    pub params: OttParams,
}

impl<'a> OttInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: OttParams) -> Self {
        Self {
            data: OttData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: OttParams) -> Self {
        Self {
            data: OttData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", OttParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(2)
    }

    #[inline]
    pub fn get_percent(&self) -> f64 {
        self.params.percent.unwrap_or(1.4)
    }

    #[inline]
    pub fn get_ma_type(&self) -> &str {
        match &self.params.ma_type {
            Some(s) => s.as_str(),
            None => "VAR",
        }
    }
}

#[derive(Clone, Debug)]
pub struct OttBuilder {
    period: Option<usize>,
    percent: Option<f64>,
    ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for OttBuilder {
    fn default() -> Self {
        Self {
            period: None,
            percent: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl OttBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, val: usize) -> Self {
        self.period = Some(val);
        self
    }

    #[inline(always)]
    pub fn percent(mut self, val: f64) -> Self {
        self.percent = Some(val);
        self
    }

    #[inline(always)]
    pub fn ma_type(mut self, val: String) -> Self {
        self.ma_type = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<OttOutput, OttError> {
        let p = OttParams {
            period: self.period,
            percent: self.percent,
            ma_type: self.ma_type,
        };
        let i = OttInput::from_candles(c, "close", p);
        ott_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<OttOutput, OttError> {
        let p = OttParams {
            period: self.period,
            percent: self.percent,
            ma_type: self.ma_type,
        };
        let i = OttInput::from_slice(d, p);
        ott_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_candles(self, c: &Candles, source: &str) -> Result<OttOutput, OttError> {
        let p = OttParams {
            period: self.period,
            percent: self.percent,
            ma_type: self.ma_type,
        };
        let i = OttInput::from_candles(c, source, p);
        ott_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<OttStream, OttError> {
        let p = OttParams {
            period: self.period,
            percent: self.percent,
            ma_type: self.ma_type,
        };
        OttStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum OttError {
    #[error("ott: Input data slice is empty.")]
    EmptyInputData,
    #[error("ott: All values are NaN.")]
    AllValuesNaN,
    #[error("ott: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("ott: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ott: Invalid percent: {percent}")]
    InvalidPercent { percent: f64 },
    #[error("ott: Invalid moving average type: {ma_type}")]
    InvalidMaType { ma_type: String },
    #[error("ott: Moving average calculation failed: {reason}")]
    MaCalculationFailed { reason: String },
    #[error("ott: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ott: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("ott: Invalid kernel for batch operation. Expected batch kernel, got: {0:?}")]
    InvalidKernelForBatch(Kernel),

    #[error("ott: Invalid kernel for batch operation")]
    InvalidBatchKernel,
}

#[inline]
pub fn ott(input: &OttInput) -> Result<OttOutput, OttError> {
    ott_with_kernel(input, Kernel::Auto)
}

pub fn ott_with_kernel(input: &OttInput, kernel: Kernel) -> Result<OttOutput, OttError> {
    let (data, period, percent, ma_type, first, chosen) = ott_prepare(input, kernel)?;

    if false
        && chosen == Kernel::Scalar
        && period == 2
        && percent == 1.4
        && ma_type.to_uppercase() == "VAR"
    {
        let mut out = vec![f64::NAN; data.len()];
        unsafe {
            ott_scalar_classic(data, period, percent, first, &mut out)?;
        }
        return Ok(OttOutput { values: out });
    }

    let ma_values = calculate_moving_average(data, period, ma_type, chosen)?;

    let ma_first = ma_values
        .iter()
        .position(|&x| !x.is_nan())
        .unwrap_or(data.len());

    let mut out = alloc_with_nan_prefix(data.len(), ma_first);

    ott_compute_into(
        data, &ma_values, percent, ma_first, period, chosen, &mut out,
    );

    Ok(OttOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn ott_into(input: &OttInput, out: &mut [f64]) -> Result<(), OttError> {
    ott_into_slice(out, input, Kernel::Auto)
}

#[inline]
pub fn ott_into_slice(dst: &mut [f64], input: &OttInput, kern: Kernel) -> Result<(), OttError> {
    let (data, period, percent, ma_type, first, chosen) = ott_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(OttError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    let ma_values = calculate_moving_average(data, period, ma_type, chosen)?;

    let ma_first = ma_values
        .iter()
        .position(|&x| !x.is_nan())
        .unwrap_or(data.len());

    ott_compute_into(data, &ma_values, percent, ma_first, period, chosen, dst);

    for v in &mut dst[..ma_first] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline(always)]
fn ott_prepare<'a>(
    input: &'a OttInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, f64, &'a str, usize, Kernel), OttError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(OttError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(OttError::AllValuesNaN)?;

    let period = input.get_period();
    let percent = input.get_percent();
    let ma_type = input.get_ma_type();

    if period == 0 || period > data.len() {
        return Err(OttError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    if data.len() - first < period {
        return Err(OttError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first,
        });
    }

    if percent < 0.0 || percent.is_nan() || percent.is_infinite() {
        return Err(OttError::InvalidPercent { percent });
    }

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    Ok((data, period, percent, ma_type, first, chosen))
}

fn calculate_moving_average(
    data: &[f64],
    period: usize,
    ma_type: &str,
    kernel: Kernel,
) -> Result<Vec<f64>, OttError> {
    match ma_type.to_uppercase().as_str() {
        "SMA" => {
            let params = SmaParams {
                period: Some(period),
            };
            let input = SmaInput::from_slice(data, params);
            sma_with_kernel(&input, kernel)
                .map(|o| o.values)
                .map_err(|e| OttError::MaCalculationFailed {
                    reason: e.to_string(),
                })
        }
        "EMA" => {
            let params = EmaParams {
                period: Some(period),
            };
            let input = EmaInput::from_slice(data, params);
            ema_with_kernel(&input, kernel)
                .map(|o| o.values)
                .map_err(|e| OttError::MaCalculationFailed {
                    reason: e.to_string(),
                })
        }
        "WMA" => {
            let params = WmaParams {
                period: Some(period),
            };
            let input = WmaInput::from_slice(data, params);
            wma_with_kernel(&input, kernel)
                .map(|o| o.values)
                .map_err(|e| OttError::MaCalculationFailed {
                    reason: e.to_string(),
                })
        }
        "TMA" => calculate_tma(data, period, kernel),
        "VAR" => calculate_var_ma(data, period),
        "WWMA" => calculate_wwma(data, period),
        "ZLEMA" => {
            let params = ZlemaParams {
                period: Some(period),
            };
            let input = ZlemaInput::from_slice(data, params);
            zlema_with_kernel(&input, kernel)
                .map(|o| o.values)
                .map_err(|e| OttError::MaCalculationFailed {
                    reason: e.to_string(),
                })
        }
        "TSF" => {
            let params = TsfParams {
                period: Some(period),
            };
            let input = TsfInput::from_slice(data, params);
            tsf_with_kernel(&input, kernel)
                .map(|o| o.values)
                .map_err(|e| OttError::MaCalculationFailed {
                    reason: e.to_string(),
                })
        }
        _ => Err(OttError::InvalidMaType {
            ma_type: ma_type.to_string(),
        }),
    }
}

fn calculate_tma(data: &[f64], period: usize, kernel: Kernel) -> Result<Vec<f64>, OttError> {
    let half_period = (period + 1) / 2;
    let floor_half = period / 2 + 1;

    let params1 = SmaParams {
        period: Some(half_period),
    };
    let input1 = SmaInput::from_slice(data, params1);
    let sma1 = sma_with_kernel(&input1, kernel).map_err(|e| OttError::MaCalculationFailed {
        reason: e.to_string(),
    })?;

    let params2 = SmaParams {
        period: Some(floor_half),
    };
    let input2 = SmaInput::from_slice(&sma1.values, params2);
    let sma2 = sma_with_kernel(&input2, kernel).map_err(|e| OttError::MaCalculationFailed {
        reason: e.to_string(),
    })?;

    Ok(sma2.values)
}

fn calculate_wwma(data: &[f64], period: usize) -> Result<Vec<f64>, OttError> {
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(OttError::AllValuesNaN)?;

    if data.len() - first < period {
        return Err(OttError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first,
        });
    }

    let mut out = alloc_with_nan_prefix(data.len(), first);
    let alpha = 1.0 / period as f64;

    let mut wwma = alpha * data[first];
    out[first] = wwma;

    for i in (first + 1)..data.len() {
        let xi = data[i];
        if xi.is_nan() {
            continue;
        }
        wwma = alpha * xi + (1.0 - alpha) * wwma;
        out[i] = wwma;
    }
    Ok(out)
}

fn calculate_var_ma(data: &[f64], period: usize) -> Result<Vec<f64>, OttError> {
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(OttError::AllValuesNaN)?;

    let mut out = alloc_with_nan_prefix(data.len(), first);
    let valpha = 2.0 / (period as f64 + 1.0);

    let mut ring_u = [0.0f64; 9];
    let mut ring_d = [0.0f64; 9];
    let mut u_sum = 0.0;
    let mut d_sum = 0.0;
    let mut idx = 0usize;

    let mut var = 0.0;
    out[first] = var;

    let start = first + 1;
    let pre_end = (first + 8).min(data.len().saturating_sub(1));
    for i in start..=pre_end {
        let a = data[i - 1];
        let b = data[i];
        if a.is_nan() || b.is_nan() {
            continue;
        }
        let up = (b - a).max(0.0);
        let down = (a - b).max(0.0);
        ring_u[idx] = up;
        u_sum += up;
        ring_d[idx] = down;
        d_sum += down;
        idx = (idx + 1) % 9;
        out[i] = var;
    }

    if data.len() - first <= 9 {
        return Ok(out);
    }

    for i in (first + 9)..data.len() {
        let a = data[i - 1];
        let b = data[i];
        if a.is_nan() || b.is_nan() {
            continue;
        }

        let old_u = ring_u[idx];
        let old_d = ring_d[idx];
        let up = (b - a).max(0.0);
        let down = (a - b).max(0.0);

        u_sum += up - old_u;
        d_sum += down - old_d;

        ring_u[idx] = up;
        ring_d[idx] = down;
        idx = (idx + 1) % 9;

        let denom = u_sum + d_sum;
        let vcmo = if denom != 0.0 {
            (u_sum - d_sum) / denom
        } else {
            0.0
        };

        var = valpha * vcmo.abs() * b + (1.0 - valpha * vcmo.abs()) * var;
        out[i] = var;
    }

    Ok(out)
}

#[inline(always)]
fn ott_compute_into(
    data: &[f64],
    ma_values: &[f64],
    percent: f64,
    first: usize,
    period: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                ott_simd128(data, ma_values, percent, first, period, out);
                return;
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                ott_scalar(data, ma_values, percent, first, period, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                ott_avx2(data, ma_values, percent, first, period, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                ott_avx512(data, ma_values, percent, first, period, out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                ott_scalar(data, ma_values, percent, first, period, out)
            }
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
pub unsafe fn ott_scalar_classic(
    data: &[f64],
    period: usize,
    percent: f64,
    first: usize,
    out: &mut [f64],
) -> Result<(), OttError> {
    let len = data.len();

    let valpha = 2.0 / (period as f64 + 1.0);

    let mut ring_u = [0.0f64; 9];
    let mut ring_d = [0.0f64; 9];
    let mut u_sum = 0.0;
    let mut d_sum = 0.0;
    let mut idx = 0usize;

    let mut var = 0.0;
    let mut var_ma = vec![f64::NAN; len];
    var_ma[first] = var;

    let start = first + 1;
    let pre_end = (first + 8).min(len.saturating_sub(1));
    for i in start..=pre_end {
        let a = data[i - 1];
        let b = data[i];
        if !a.is_nan() && !b.is_nan() {
            let up = (b - a).max(0.0);
            let down = (a - b).max(0.0);
            ring_u[idx] = up;
            u_sum += up;
            ring_d[idx] = down;
            d_sum += down;
            idx = (idx + 1) % 9;
        }
        var_ma[i] = var;
    }

    for i in (first + 9)..len {
        let a = data[i - 1];
        let b = data[i];
        if !a.is_nan() && !b.is_nan() {
            let old_u = ring_u[idx];
            let old_d = ring_d[idx];
            let up = (b - a).max(0.0);
            let down = (a - b).max(0.0);

            u_sum += up - old_u;
            d_sum += down - old_d;

            ring_u[idx] = up;
            ring_d[idx] = down;
            idx = (idx + 1) % 9;

            let denom = u_sum + d_sum;
            let vcmo = if denom != 0.0 {
                (u_sum - d_sum) / denom
            } else {
                0.0
            };

            var = valpha * vcmo.abs() * b + (1.0 - valpha * vcmo.abs()) * var;
            var_ma[i] = var;
        } else if i > first {
            var_ma[i] = var_ma[i - 1];
        }
    }

    let fark = percent * 0.01;
    let ma_first = first;

    for i in 0..ma_first {
        out[i] = f64::NAN;
    }

    let mut dir = 1i32;
    let mut long_stop = f64::NAN;
    let mut short_stop = f64::NAN;

    for i in ma_first..len {
        let mavg = var_ma[i];

        if mavg.is_nan() {
            continue;
        }

        let offset = mavg * fark;

        let long_stop_prev = if long_stop.is_nan() {
            mavg - offset
        } else {
            long_stop
        };
        long_stop = if mavg > long_stop_prev {
            (mavg - offset).max(long_stop_prev)
        } else {
            mavg - offset
        };

        let short_stop_prev = if short_stop.is_nan() {
            mavg + offset
        } else {
            short_stop
        };
        short_stop = if mavg < short_stop_prev {
            (mavg + offset).min(short_stop_prev)
        } else {
            mavg + offset
        };

        let prev_dir = dir;
        if mavg > short_stop_prev {
            dir = 1;
        } else if mavg <= long_stop_prev {
            dir = -1;
        }

        out[i] = if dir == -1 { short_stop } else { long_stop };
    }

    Ok(())
}

#[inline]
pub fn ott_scalar(
    _data: &[f64],
    ma_values: &[f64],
    percent: f64,
    first_val: usize,
    _period: usize,
    out: &mut [f64],
) {
    let len = ma_values.len();
    if first_val >= len {
        return;
    }

    let fark = percent * 0.01;
    let scale_minus = 1.0 - (percent * 0.005);

    let mut i = first_val;
    let mut m = ma_values[i];
    if m.is_nan() {
        if let Some(next) = ma_values[first_val..].iter().position(|x| !x.is_nan()) {
            i = first_val + next;
            m = ma_values[i];
        } else {
            return;
        }
    }

    let mut long_stop = m.mul_add(-fark, m);
    let mut short_stop = m.mul_add(fark, m);
    let mut dir: i32 = 1;

    let mt0 = long_stop;
    let scale0 = if m > mt0 {
        scale_minus + fark
    } else {
        scale_minus
    };
    out[i] = mt0 * scale0;
    i += 1;

    while i < len {
        let mavg = ma_values[i];
        if !mavg.is_nan() {
            let cand_long = mavg.mul_add(-fark, mavg);
            let cand_short = mavg.mul_add(fark, mavg);

            let lprev = long_stop;
            let sprev = short_stop;

            if mavg > lprev {
                long_stop = if cand_long > lprev { cand_long } else { lprev };
            } else {
                long_stop = cand_long;
            }

            if mavg < sprev {
                short_stop = if cand_short < sprev {
                    cand_short
                } else {
                    sprev
                };
            } else {
                short_stop = cand_short;
            }

            if dir == -1 && mavg > sprev {
                dir = 1;
            } else if dir == 1 && mavg < lprev {
                dir = -1;
            }

            let mt = if dir == 1 { long_stop } else { short_stop };
            let scale = if mavg > mt {
                scale_minus + fark
            } else {
                scale_minus
            };
            out[i] = mt * scale;
        }
        i += 1;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn ott_simd128(
    data: &[f64],
    ma_values: &[f64],
    percent: f64,
    first_val: usize,
    period: usize,
    out: &mut [f64],
) {
    ott_scalar(data, ma_values, percent, first_val, period, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn ott_avx2(
    data: &[f64],
    ma_values: &[f64],
    percent: f64,
    first_val: usize,
    period: usize,
    out: &mut [f64],
) {
    ott_scalar(data, ma_values, percent, first_val, period, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn ott_avx512(
    data: &[f64],
    ma_values: &[f64],
    percent: f64,
    first_val: usize,
    period: usize,
    out: &mut [f64],
) {
    ott_scalar(data, ma_values, percent, first_val, period, out);
}

#[derive(Debug, Clone)]
pub struct OttStream {
    period: usize,
    percent: f64,
    ma_type: String,

    buf: Vec<f64>,
    pos: usize,
    count: usize,

    long_stop: f64,
    short_stop: f64,
    dir: i32,

    fark: f64,
    scale_plus: f64,
    scale_minus: f64,

    sma_sum: f64,

    ema_alpha: f64,
    ema_state: Option<f64>,

    ww_alpha: f64,
    wwma_state: Option<f64>,

    wma_simple_sum: f64,
    wma_weighted_sum: f64,
    wma_inv_norm: f64,

    zlema_alpha: f64,
    zlema_state: Option<f64>,
    zlema_lag: usize,

    var_alpha_base: f64,
    var_state: f64,
    var_u_ring: [f64; 9],
    var_d_ring: [f64; 9],
    var_idx: usize,
    var_u_sum: f64,
    var_d_sum: f64,
    var_seen_diffs: usize,
}

impl OttStream {
    pub fn try_new(params: OttParams) -> Result<Self, OttError> {
        let period = params.period.unwrap_or(2);
        let percent = params.percent.unwrap_or(1.4);
        let ma_type = params.ma_type.unwrap_or_else(|| "VAR".to_string());

        if period == 0 {
            return Err(OttError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        if percent < 0.0 || !percent.is_finite() {
            return Err(OttError::InvalidPercent { percent });
        }

        let need = if ma_type.eq_ignore_ascii_case("VAR") {
            period.max(10)
        } else {
            period.max(1)
        };

        let fark = percent * 0.01;
        let scale_minus = 1.0 - (percent * 0.005);
        let scale_plus = 1.0 + (percent * 0.005);

        let ema_alpha = 2.0 / (period as f64 + 1.0);
        let ww_alpha = 1.0 / period as f64;
        let zlema_alpha = ema_alpha;
        let zlema_lag = ((period.saturating_sub(1)) as f64 / 2.0).floor() as usize;

        let n = period as f64;
        let wma_inv_norm = if period > 1 {
            2.0 / (n * (n + 1.0))
        } else {
            1.0
        };

        Ok(Self {
            period,
            percent,
            ma_type,

            buf: vec![f64::NAN; need],
            pos: 0,
            count: 0,

            long_stop: f64::NAN,
            short_stop: f64::NAN,
            dir: 1,

            fark,
            scale_plus,
            scale_minus,

            sma_sum: 0.0,

            ema_alpha,
            ema_state: None,

            ww_alpha,
            wwma_state: None,

            wma_simple_sum: 0.0,
            wma_weighted_sum: 0.0,
            wma_inv_norm,

            zlema_alpha,
            zlema_state: None,
            zlema_lag,

            var_alpha_base: ema_alpha,
            var_state: 0.0,
            var_u_ring: [0.0; 9],
            var_d_ring: [0.0; 9],
            var_idx: 0,
            var_u_sum: 0.0,
            var_d_sum: 0.0,
            var_seen_diffs: 0,
        })
    }

    #[inline]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        let cap = self.buf.len();

        let old = self.buf[self.pos];
        self.buf[self.pos] = x;
        self.pos = (self.pos + 1) % cap;
        if self.count < cap {
            self.count += 1;
        }

        let ma = self.calculate_ma(x, old);

        if !ma.is_finite() || self.count < cap {
            return None;
        }

        let offset = ma * self.fark;

        let lprev = if self.long_stop.is_nan() {
            ma - offset
        } else {
            self.long_stop
        };
        let sprev = if self.short_stop.is_nan() {
            ma + offset
        } else {
            self.short_stop
        };

        let cand_long = ma - offset;
        self.long_stop = if ma > lprev {
            if cand_long > lprev {
                cand_long
            } else {
                lprev
            }
        } else {
            cand_long
        };

        let cand_short = ma + offset;
        self.short_stop = if ma < sprev {
            if cand_short < sprev {
                cand_short
            } else {
                sprev
            }
        } else {
            cand_short
        };

        if self.dir == -1 && ma > sprev {
            self.dir = 1;
        } else if self.dir == 1 && ma < lprev {
            self.dir = -1;
        }

        let mt = if self.dir == 1 {
            self.long_stop
        } else {
            self.short_stop
        };
        let scaled = if ma > mt {
            mt * self.scale_plus
        } else {
            mt * self.scale_minus
        };

        Some(scaled)
    }

    #[inline]
    fn calculate_ma(&mut self, x: f64, old: f64) -> f64 {
        match self.ma_type.as_str() {
            "VAR" => self.update_var(x),

            "WWMA" => self.update_wwma(x),

            "EMA" => self.update_ema(x),

            "SMA" => self.update_sma(x, old),

            "WMA" => self.update_wma(x, old),

            "ZLEMA" => self.update_zlema(x),

            _ => self.update_sma(x, old),
        }
    }

    #[inline]
    fn update_sma(&mut self, x: f64, old: f64) -> f64 {
        if old.is_finite() {
            self.sma_sum += x - old;
        } else {
            self.sma_sum += x;
        }
        if self.count < self.period {
            f64::NAN
        } else {
            self.sma_sum / self.period as f64
        }
    }

    #[inline]
    fn update_ema(&mut self, x: f64) -> f64 {
        let ema = match self.ema_state {
            Some(prev) => self.ema_alpha.mul_add(x - prev, prev),
            None => x,
        };
        self.ema_state = Some(ema);
        ema
    }

    #[inline]
    fn update_wwma(&mut self, x: f64) -> f64 {
        let ww = match self.wwma_state {
            Some(prev) => self.ww_alpha.mul_add(x - prev, prev),
            None => self.ww_alpha * x,
        };
        self.wwma_state = Some(ww);
        ww
    }

    #[inline]
    fn update_wma(&mut self, x: f64, old: f64) -> f64 {
        if self.count <= self.period {
            let w = self.count as f64;
            self.wma_simple_sum += x;
            self.wma_weighted_sum += w * x;

            if self.count < self.period {
                return f64::NAN;
            }
            return self.wma_weighted_sum * self.wma_inv_norm;
        }

        let s_prev = self.wma_simple_sum;
        self.wma_weighted_sum += self.period as f64 * x - s_prev;

        let x_out = old;
        self.wma_simple_sum += x - x_out;

        self.wma_weighted_sum * self.wma_inv_norm
    }

    #[inline]
    fn update_zlema(&mut self, x: f64) -> f64 {
        let cap = self.buf.len();
        let lag_idx = (self.pos + cap - 1 - self.zlema_lag % cap) % cap;
        let lagged = self.buf[lag_idx];

        let de_lagged = if lagged.is_finite() {
            x + (x - lagged)
        } else {
            x
        };
        let z = match self.zlema_state {
            Some(prev) => self.zlema_alpha.mul_add(de_lagged - prev, prev),
            None => de_lagged,
        };
        self.zlema_state = Some(z);
        z
    }

    #[inline]
    fn update_var(&mut self, x: f64) -> f64 {
        let cap = self.buf.len();
        if self.count == 0 {
            return self.var_state;
        }
        let prev_idx = (self.pos + cap - 2) % cap;
        let prev = self.buf[prev_idx];
        if !x.is_finite() || !prev.is_finite() {
            return self.var_state;
        }

        let up = (x - prev).max(0.0);
        let dn = (prev - x).max(0.0);

        let old_u = self.var_u_ring[self.var_idx];
        let old_d = self.var_d_ring[self.var_idx];
        self.var_u_ring[self.var_idx] = up;
        self.var_d_ring[self.var_idx] = dn;

        if self.var_seen_diffs < 9 {
            self.var_seen_diffs += 1;
        }

        self.var_u_sum += up - old_u;
        self.var_d_sum += dn - old_d;
        self.var_idx = (self.var_idx + 1) % 9;

        if self.count < 10 || self.var_seen_diffs < 9 {
            return self.var_state;
        }

        let denom = self.var_u_sum + self.var_d_sum;
        let cmo_abs = if denom != 0.0 {
            (self.var_u_sum - self.var_d_sum).abs() / denom
        } else {
            0.0
        };

        let alpha = cmo_abs * self.var_alpha_base;

        self.var_state = alpha.mul_add(x - self.var_state, self.var_state);
        self.var_state
    }
}

#[derive(Clone, Debug)]
pub struct OttBatchRange {
    pub period: (usize, usize, usize),
    pub percent: (f64, f64, f64),
    pub ma_types: Vec<String>,
}

impl Default for OttBatchRange {
    fn default() -> Self {
        Self {
            period: (2, 251, 1),
            percent: (1.4, 1.4, 0.0),
            ma_types: vec!["VAR".to_string()],
        }
    }
}

#[derive(Clone, Debug)]
pub struct OttBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<OttParams>,
    pub rows: usize,
    pub cols: usize,
}

impl OttBatchOutput {
    pub fn row_for_params(&self, p: &OttParams) -> Option<usize> {
        let tp = p.period.unwrap_or(2);
        let tq = p.percent.unwrap_or(1.4);
        let tt = p.ma_type.as_deref().unwrap_or("VAR");
        self.combos.iter().position(|c| {
            c.period.unwrap_or(2) == tp
                && (c.percent.unwrap_or(1.4) - tq).abs() < 1e-12
                && c.ma_type.as_deref().unwrap_or("VAR") == tt
        })
    }

    pub fn values_for(&self, p: &OttParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct OttBatchBuilder {
    range: OttBatchRange,
    kernel: Kernel,
}

impl OttBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<OttBatchOutput, OttError> {
        OttBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn with_default_candles(c: &Candles) -> Result<OttBatchOutput, OttError> {
        OttBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }

    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    #[inline]
    pub fn period_static(mut self, val: usize) -> Self {
        self.range.period = (val, val, 0);
        self
    }

    #[inline]
    pub fn percent_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.percent = (start, end, step);
        self
    }

    #[inline]
    pub fn percent_static(mut self, val: f64) -> Self {
        self.range.percent = (val, val, 0.0);
        self
    }

    #[inline]
    pub fn ma_types(mut self, types: Vec<String>) -> Self {
        self.range.ma_types = types;
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<OttBatchOutput, OttError> {
        ott_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_batch(self, data: &[f64]) -> Result<OttBatchOutput, OttError> {
        ott_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<OttBatchOutput, OttError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
}

#[inline(always)]
fn expand_grid_ott(r: &OttBatchRange) -> Result<Vec<OttParams>, OttError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, OttError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            return Ok((start..=end).step_by(step.max(1)).collect());
        }
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(OttError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, OttError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            let st = step.abs();
            while x <= end + 1e-12 {
                v.push(x);
                x += st;
            }
            if v.is_empty() {
                return Err(OttError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut x = start;
        let st = step.abs();
        while x + 1e-12 >= end {
            v.push(x);
            x -= st;
        }
        if v.is_empty() {
            return Err(OttError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(v)
    }

    let periods = axis_usize(r.period)?;
    let percents = axis_f64(r.percent)?;
    let types = if r.ma_types.is_empty() {
        vec!["VAR".to_string()]
    } else {
        r.ma_types.clone()
    };
    let cap = periods
        .len()
        .checked_mul(percents.len())
        .and_then(|x| x.checked_mul(types.len()))
        .ok_or_else(|| OttError::InvalidRange {
            start: "cap".into(),
            end: "overflow".into(),
            step: "mul".into(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &pct in &percents {
            for mt in &types {
                out.push(OttParams {
                    period: Some(p),
                    percent: Some(pct),
                    ma_type: Some(mt.clone()),
                });
            }
        }
    }
    if out.is_empty() {
        return Err(OttError::InvalidRange {
            start: r.period.0.to_string(),
            end: r.period.1.to_string(),
            step: r.period.2.to_string(),
        });
    }
    Ok(out)
}

#[inline(always)]
fn ott_batch_inner_into(
    data: &[f64],
    sweep: &OttBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<OttParams>, OttError> {
    let combos = expand_grid_ott(sweep)?;

    let cols = data.len();
    if cols == 0 {
        return Err(OttError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(OttError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if max_p == 0 || max_p > cols {
        return Err(OttError::InvalidPeriod {
            period: max_p,
            data_len: cols,
        });
    }
    if cols - first < max_p {
        return Err(OttError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }

    let row_kern = match kern {
        Kernel::Auto => match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => Kernel::Scalar,
        },
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        k => k,
    };

    let mut ma_cache: HashMap<(usize, String), (Vec<f64>, usize)> = HashMap::new();
    for prm in &combos {
        let p = prm.period.unwrap();
        if p == 0 || p > cols {
            return Err(OttError::InvalidPeriod {
                period: p,
                data_len: cols,
            });
        }
        let pct = prm.percent.unwrap();
        if pct < 0.0 || !pct.is_finite() {
            return Err(OttError::InvalidPercent { percent: pct });
        }
        let mt = prm.ma_type.as_deref().unwrap().to_uppercase();
        if !ma_cache.contains_key(&(p, mt.clone())) {
            let ma = calculate_moving_average(data, p, &mt, row_kern).map_err(|e| {
                OttError::MaCalculationFailed {
                    reason: e.to_string(),
                }
            })?;
            let ma_first = ma.iter().position(|&x| !x.is_nan()).unwrap_or(cols);
            ma_cache.insert((p, mt), (ma, ma_first));
        }
    }

    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    let do_row = |r: usize, dst_mu: &mut [MaybeUninit<f64>]| -> Result<(), OttError> {
        let prm = &combos[r];
        let p = prm.period.unwrap();
        let pct = prm.percent.unwrap();
        let mt = prm.ma_type.as_deref().unwrap();

        let key = (p, mt.to_uppercase());
        let (ma, ma_first) = ma_cache.get(&key).expect("missing MA cache entry");

        let row: &mut [f64] = unsafe {
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len())
        };

        for v in &mut row[..(*ma_first).min(cols)] {
            *v = f64::NAN;
        }

        ott_compute_into(data, ma, pct, *ma_first, p, row_kern, row);
        Ok(())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .try_for_each(|(r, sl)| do_row(r, sl))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, sl) in out_mu.chunks_mut(cols).enumerate() {
                do_row(r, sl)?;
            }
        }
    } else {
        for (r, sl) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, sl)?;
        }
    }

    Ok(combos)
}

#[inline(always)]
pub fn ott_batch_slice(
    data: &[f64],
    sweep: &OttBatchRange,
    kern: Kernel,
) -> Result<OttBatchOutput, OttError> {
    ott_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ott_batch_par_slice(
    data: &[f64],
    sweep: &OttBatchRange,
    kern: Kernel,
) -> Result<OttBatchOutput, OttError> {
    ott_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn ott_batch_inner(
    data: &[f64],
    sweep: &OttBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<OttBatchOutput, OttError> {
    let combos = expand_grid_ott(sweep)?;
    let cols = data.len();
    if cols == 0 {
        return Err(OttError::EmptyInputData);
    }

    let rows = combos.len();
    rows.checked_mul(cols)
        .ok_or_else(|| OttError::InvalidRange {
            start: sweep.period.0.to_string(),
            end: sweep.period.1.to_string(),
            step: sweep.period.2.to_string(),
        })?;
    let mut buf_mu = make_uninit_matrix(rows, cols);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let combos = ott_batch_inner_into(data, sweep, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(OttBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn ott_batch_with_kernel(
    data: &[f64],
    sweep: &OttBatchRange,
    k: Kernel,
) -> Result<OttBatchOutput, OttError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(OttError::InvalidKernelForBatch(k)),
    };

    ott_batch_par_slice(data, sweep, kernel)
}

#[cfg(feature = "python")]
#[pyfunction(name = "ott")]
#[pyo3(signature = (data, period=2, percent=1.4, ma_type="VAR", kernel=None))]
pub fn ott_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    percent: f64,
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = OttParams {
        period: Some(period),
        percent: Some(percent),
        ma_type: Some(ma_type.to_string()),
    };
    let input = OttInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| ott_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "OttStream")]
pub struct OttStreamPy {
    stream: OttStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl OttStreamPy {
    #[new]
    fn new(period: usize, percent: f64, ma_type: &str) -> PyResult<Self> {
        let params = OttParams {
            period: Some(period),
            percent: Some(percent),
            ma_type: Some(ma_type.to_string()),
        };
        let stream =
            OttStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(OttStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ott_batch")]
#[pyo3(signature = (data, period_range, percent_range, ma_types, kernel=None))]
pub fn ott_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    percent_range: (f64, f64, f64),
    ma_types: Vec<String>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{PyArray1, PyArrayMethods};
    let slice_in = data.as_slice()?;
    let sweep = OttBatchRange {
        period: period_range,
        percent: percent_range,
        ma_types,
    };
    let kern = validate_kernel(kernel, true)?;

    let combos = expand_grid_ott(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows * cols overflow"))?;

    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => unreachable!(),
            };
            ott_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "percents",
        combos
            .iter()
            .map(|p| p.percent.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    let types = PyList::new(py, combos.iter().map(|p| p.ma_type.as_deref().unwrap()))?;
    dict.set_item("ma_types", types)?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ott_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, percent_range, ma_types, device_id=0))]
pub fn ott_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    percent_range: (f64, f64, f64),
    ma_types: Vec<String>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_f32.as_slice()?;
    let sweep = OttBatchRange {
        period: period_range,
        percent: percent_range,
        ma_types,
    };

    let combos = expand_grid_ott(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let cols = slice_in.len();
    for prm in &combos {
        let p = prm.period.unwrap();
        if p == 0 || p > cols {
            return Err(PyValueError::new_err(
                OttError::InvalidPeriod {
                    period: p,
                    data_len: cols,
                }
                .to_string(),
            ));
        }
        let pct = prm.percent.unwrap();
        if pct < 0.0 || !pct.is_finite() {
            return Err(PyValueError::new_err(
                OttError::InvalidPercent { percent: pct }.to_string(),
            ));
        }
    }
    let inner = py.allow_threads(|| {
        let cuda = CudaOtt::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.ott_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let handle = make_device_array_py(device_id, inner)?;
    Ok(handle)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ott_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, percent, ma_type="VAR", device_id=0))]
pub fn ott_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    percent: f64,
    ma_type: &str,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let flat = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = OttParams {
        period: Some(period),
        percent: Some(percent),
        ma_type: Some(ma_type.to_string()),
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaOtt::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.ott_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let handle = make_device_array_py(device_id, inner)?;
    Ok(handle)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ott_js(
    data: &[f64],
    period: usize,
    percent: f64,
    ma_type: &str,
) -> Result<Vec<f64>, JsValue> {
    let params = OttParams {
        period: Some(period),
        percent: Some(percent),
        ma_type: Some(ma_type.to_string()),
    };
    let input = OttInput::from_slice(data, params);

    let mut out = vec![f64::NAN; data.len()];

    let kernel = if cfg!(target_arch = "wasm32") {
        Kernel::Scalar
    } else {
        detect_best_kernel()
    };
    ott_into_slice(&mut out, &input, kernel).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ott_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ott_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ott_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    percent: f64,
    ma_type: &str,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to ott_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let params = OttParams {
            period: Some(period),
            percent: Some(percent),
            ma_type: Some(ma_type.to_string()),
        };
        let input = OttInput::from_slice(data, params);

        let kernel = if cfg!(target_arch = "wasm32") {
            Kernel::Scalar
        } else {
            detect_best_kernel()
        };

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            ott_into_slice(&mut temp, &input, kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ott_into_slice(out, &input, kernel).map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[deprecated(
    since = "1.0.0",
    note = "For reuse, prefer fast/unsafe API with persistent buffers"
)]
pub struct OttContext {
    period: usize,
    percent: f64,
    ma_type: String,
    kernel: Kernel,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
#[allow(deprecated)]
impl OttContext {
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize, percent: f64, ma_type: &str) -> Result<OttContext, JsValue> {
        if period == 0 {
            return Err(JsValue::from_str("Invalid period: 0"));
        }
        if !percent.is_finite() || percent < 0.0 {
            return Err(JsValue::from_str("Invalid percent"));
        }
        Ok(OttContext {
            period,
            percent,
            ma_type: ma_type.to_string(),
            kernel: if cfg!(target_arch = "wasm32") {
                Kernel::Scalar
            } else {
                detect_best_kernel()
            },
        })
    }

    pub fn update_into(
        &self,
        in_ptr: *const f64,
        out_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if in_ptr.is_null() || out_ptr.is_null() {
            return Err(JsValue::from_str("null pointer"));
        }
        unsafe {
            let data = std::slice::from_raw_parts(in_ptr, len);
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            let params = OttParams {
                period: Some(self.period),
                percent: Some(self.percent),
                ma_type: Some(self.ma_type.clone()),
            };
            let input = OttInput::from_slice(data, params);
            ott_into_slice(out, &input, self.kernel).map_err(|e| JsValue::from_str(&e.to_string()))
        }
    }

    pub fn get_warmup_period(&self) -> usize {
        self.period.saturating_sub(1)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct OttBatchConfig {
    pub period_range: (usize, usize, usize),
    pub percent_range: (f64, f64, f64),
    pub ma_types: Vec<String>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct OttBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<OttParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = ott_batch)]
pub fn ott_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: OttBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = OttBatchRange {
        period: cfg.period_range,
        percent: cfg.percent_range,
        ma_types: cfg.ma_types,
    };

    let kernel = if cfg!(target_arch = "wasm32") {
        Kernel::ScalarBatch
    } else {
        detect_best_batch_kernel()
    };
    let out = ott_batch_with_kernel(data, &sweep, kernel)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = OttBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ott_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    p_start: usize,
    p_end: usize,
    p_step: usize,
    q_start: f64,
    q_end: f64,
    q_step: f64,
    ma_types: JsValue,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to ott_batch_into"));
    }
    let types: Vec<String> = serde_wasm_bindgen::from_value(ma_types)
        .map_err(|e| JsValue::from_str(&format!("Invalid ma_types: {}", e)))?;

    let sweep = OttBatchRange {
        period: (p_start, p_end, p_step),
        percent: (q_start, q_end, q_step),
        ma_types: types,
    };
    let combos = expand_grid_ott(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    if combos.is_empty() {
        return Err(JsValue::from_str("no parameter combinations"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let rows = combos.len();
        let cols = len;

        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("rows * cols overflow"))?;

        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        let row_kern = match detect_best_batch_kernel() {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        };

        for (r, prm) in combos.iter().enumerate() {
            let p = prm.period.unwrap();
            let pct = prm.percent.unwrap();
            let mt = prm.ma_type.as_deref().unwrap();
            if p == 0 || p > cols {
                return Err(JsValue::from_str(
                    &OttError::InvalidPeriod {
                        period: p,
                        data_len: cols,
                    }
                    .to_string(),
                ));
            }
            if pct < 0.0 || !pct.is_finite() {
                return Err(JsValue::from_str(
                    &OttError::InvalidPercent { percent: pct }.to_string(),
                ));
            }

            let ma = calculate_moving_average(data, p, mt, row_kern)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let ma_first = ma.iter().position(|&x| !x.is_nan()).unwrap_or(cols);

            let row = &mut out[r * cols..(r + 1) * cols];
            for v in &mut row[..ma_first.min(cols)] {
                *v = f64::NAN;
            }

            ott_compute_into(data, &ma, pct, ma_first, p, row_kern, row);
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;
    use std::error::Error;

    fn check_ott_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = OttInput::from_candles(&candles, "close", OttParams::default());
        let result = ott_with_kernel(&input, kernel)?;

        let expected_last_five = [
            59719.89457348,
            59719.89457348,
            59719.89457348,
            59719.89457348,
            59649.80599569,
        ];

        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] OTT {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_ott_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = OttParams {
            period: None,
            percent: None,
            ma_type: None,
        };
        let input = OttInput::from_candles(&candles, "close", default_params);
        let output = ott_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_ott_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = OttInput::with_default_candles(&candles);
        match input.data {
            OttData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected OttData::Candles"),
        }
        let output = ott_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_ott_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = OttParams {
            period: Some(0),
            percent: None,
            ma_type: None,
        };
        let input = OttInput::from_slice(&input_data, params);
        let res = ott_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] OTT should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_ott_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = OttParams {
            period: Some(10),
            percent: None,
            ma_type: None,
        };
        let input = OttInput::from_slice(&data_small, params);
        let res = ott_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] OTT should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_ott_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = OttParams::default();
        let input = OttInput::from_slice(&single_point, params);
        let res = ott_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] OTT should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_ott_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = OttParams::default();
        let input = OttInput::from_slice(&empty, params);
        let res = ott_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] OTT should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_ott_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = OttParams::default();
        let input = OttInput::from_slice(&nan_data, params);
        let res = ott_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] OTT should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    #[cfg(feature = "proptest")]
    proptest! {
        #[test]
        fn test_ott_no_panic(data: Vec<f64>, period in 1usize..100) {
            let params = OttParams {
                period: Some(period),
                percent: Some(1.4),
                ma_type: Some("VAR".to_string()),
            };
            let input = OttInput::from_slice(&data, params);
            let _ = ott(&input);
        }

        #[test]
        fn test_ott_length_preservation(size in 10usize..100) {
            let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
            let params = OttParams::default();
            let input = OttInput::from_slice(&data, params);

            if let Ok(output) = ott(&input) {
                prop_assert_eq!(output.values.len(), size);
            }
        }
    }

    fn check_ott_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = OttInput::from_candles(&candles, "close", OttParams::default());
        let first_result = ott_with_kernel(&input, kernel)?;

        let input2 = OttInput::from_slice(&first_result.values, OttParams::default());
        let second_result = ott_with_kernel(&input2, kernel)?;

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "[{}] OTT reinput length mismatch",
            test_name
        );
        Ok(())
    }

    fn check_ott_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = OttInput::from_candles(&candles, "close", OttParams::default());
        let result = ott_with_kernel(&input, kernel)?;

        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "[{}] OTT length mismatch",
            test_name
        );

        let first_valid = result
            .values
            .iter()
            .position(|x| !x.is_nan())
            .unwrap_or(result.values.len());

        if result.values.len() > first_valid + 100 {
            for i in (first_valid + 100)..result.values.len() {
                if candles.close[i].is_nan() {
                    continue;
                }
                assert!(
                    !result.values[i].is_nan(),
                    "[{}] Unexpected NaN at index {} after warmup",
                    test_name,
                    i
                );
            }
        }

        assert!(
            first_valid <= candles.close.len(),
            "[{}] First valid index out of range",
            test_name
        );
        Ok(())
    }

    fn check_ott_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close = &candles.close;

        let input = OttInput::from_candles(&candles, "close", OttParams::default());
        let batch_result = ott_with_kernel(&input, kernel)?;

        let mut stream = OttStream::try_new(OttParams::default())?;
        let mut stream_values = Vec::new();

        for &price in close {
            let result = stream.update(price);
            stream_values.push(result.unwrap_or(f64::NAN));
        }

        assert_eq!(
            batch_result.values.len(),
            stream_values.len(),
            "[{}] OTT streaming length mismatch",
            test_name
        );

        let warmup = OttParams::default().period.unwrap_or(2);
        if batch_result.values.len() > warmup + 10 {
            for i in (warmup + 10)..batch_result.values.len() {
                if batch_result.values[i].is_nan() || stream_values[i].is_nan() {
                    continue;
                }

                let diff = (batch_result.values[i] - stream_values[i]).abs();
                let tolerance = batch_result.values[i].abs() * 0.05;
                assert!(
                    diff <= tolerance.max(1.0),
                    "[{}] OTT streaming mismatch at index {}: batch={}, stream={}, diff={}",
                    test_name,
                    i,
                    batch_result.values[i],
                    stream_values[i],
                    diff
                );
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_ott_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let out = OttBuilder::new().kernel(kernel).apply(&c)?;
        for &v in &out.values {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(
                b, 0x11111111_11111111,
                "alloc_with_nan_prefix poison leaked"
            );
            assert_ne!(b, 0x22222222_22222222, "init_matrix_prefixes poison leaked");
            assert_ne!(b, 0x33333333_33333333, "make_uninit_matrix poison leaked");
        }
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ott_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_ott_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    fn check_ott_invalid_percent(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let params = OttParams {
                period: Some(2),
                percent: Some(bad),
                ma_type: Some("VAR".to_string()),
            };
            let input = OttInput::from_slice(&data, params);
            let res = ott_with_kernel(&input, kernel);
            assert!(matches!(res, Err(OttError::InvalidPercent { .. })));
        }
        Ok(())
    }

    generate_all_ott_tests!(
        check_ott_partial_params,
        check_ott_accuracy,
        check_ott_default_candles,
        check_ott_zero_period,
        check_ott_period_exceeds_length,
        check_ott_very_small_dataset,
        check_ott_empty_input,
        check_ott_all_nan,
        check_ott_reinput,
        check_ott_nan_handling,
        check_ott_streaming,
        check_ott_no_poison,
        check_ott_invalid_percent
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let out = OttBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = OttParams::default();
        let row = out.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());

        let single_kernel = match kernel {
            Kernel::ScalarBatch => Kernel::Scalar,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2Batch => Kernel::Avx2,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Auto => Kernel::Scalar,
            _ => Kernel::Scalar,
        };
        let single =
            ott_with_kernel(&OttInput::from_slice(&c.close, def.clone()), single_kernel)?.values;

        assert_eq!(single.len(), row.len());
        for i in 0..row.len() {
            if row[i].is_nan() || single[i].is_nan() {
                continue;
            }
            assert!(
                (row[i] - single[i]).abs() <= 1e-9,
                "[{test}] mismatch at {i}"
            );
        }
        Ok(())
    }

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let data = vec![1.0; 100];

        let out = OttBatchBuilder::new()
            .kernel(kernel)
            .period_range(10, 20, 10)
            .percent_range(1.0, 2.0, 1.0)
            .ma_types(vec!["VAR".to_string(), "WWMA".to_string()])
            .apply_slice(&data)?;

        assert_eq!(
            out.rows, 8,
            "[{}] Expected 8 rows for parameter sweep",
            test
        );
        assert_eq!(out.cols, 100, "[{}] Column count mismatch", test);

        assert_eq!(out.combos.len(), 8, "[{}] Combos count mismatch", test);
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let out = OttBatchBuilder::new()
            .kernel(kernel)
            .period_range(40, 60, 20)
            .percent_range(1.0, 2.0, 1.0)
            .ma_types(vec!["VAR".to_string()])
            .apply_slice(&c.close)?;

        for &v in &out.values {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            assert_ne!(
                b, 0x11111111_11111111,
                "[{}] alloc_with_nan_prefix poison",
                test
            );
            assert_ne!(
                b, 0x22222222_22222222,
                "[{}] init_matrix_prefixes poison",
                test
            );
            assert_ne!(
                b, 0x33333333_33333333,
                "[{}] make_uninit_matrix poison",
                test
            );
        }
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);

    fn check_batch_helpers_and_row_lookup(
        _test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];

        let out1 = OttBatchBuilder::with_default_slice(&data, kernel)?;
        assert_eq!(out1.rows, 1);
        assert_eq!(out1.cols, data.len());

        let builder = OttBatchBuilder::new()
            .kernel(kernel)
            .period_static(3)
            .percent_static(1.4)
            .ma_types(vec!["VAR".to_string()]);

        let out2 = builder.apply_batch(&data)?;
        assert_eq!(out2.rows, 1);
        assert_eq!(out2.cols, data.len());

        let params = OttParams {
            period: Some(3),
            percent: Some(1.4),
            ma_type: Some("VAR".to_string()),
        };

        let row_idx = out2.row_for_params(&params);
        assert_eq!(row_idx, Some(0));

        let values = out2.values_for(&params);
        assert!(values.is_some());
        assert_eq!(values.unwrap().len(), data.len());

        let default_params = OttParams::default();
        let default_row_idx = out1.row_for_params(&default_params);
        assert_eq!(default_row_idx, Some(0));

        let invalid_params = OttParams {
            period: Some(999),
            percent: Some(999.0),
            ma_type: Some("INVALID".to_string()),
        };
        assert_eq!(out2.row_for_params(&invalid_params), None);
        assert_eq!(out2.values_for(&invalid_params), None);

        Ok(())
    }

    gen_batch_tests!(check_batch_helpers_and_row_lookup);
}

#[cfg(test)]
#[test]
fn test_ott_into_matches_api() {
    let n = 256usize;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64;
        data.push((x * 0.1).sin() * 100.0 + x * 0.05);
    }

    let input = OttInput::from_slice(&data, OttParams::default());

    let baseline = ott(&input).expect("baseline ott() failed");

    let mut into_out = vec![0.0; data.len()];

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    {
        ott_into(&input, &mut into_out).expect("ott_into failed");
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    {
        ott_into_slice(&mut into_out, &input, Kernel::Scalar).expect("ott_into_slice failed");
    }

    assert_eq!(baseline.values.len(), into_out.len());

    fn eq_or_both_nan(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= 1e-12
    }

    for (i, (&a, &b)) in baseline.values.iter().zip(into_out.iter()).enumerate() {
        assert!(
            eq_or_both_nan(a, b),
            "parity mismatch at {}: vec_api={} vs into_api={}",
            i,
            a,
            b
        );
    }
}
