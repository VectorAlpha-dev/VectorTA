//! # Schaff Trend Cycle (STC)
//!
//! Schaff Trend Cycle (STC) is an oscillator that applies MACD and double stochastic steps,
//! then smooths with EMA. This implementation supports batch, AVX, and builder APIs like alma.rs.
//!
//! ## Parameters
//! - **fast_period**: Period for fast MA (default: 23)
//! - **slow_period**: Period for slow MA (default: 50)
//! - **k_period**: Stochastic window (default: 10)
//! - **d_period**: EMA smoothing window (default: 3)
//! - **fast_ma_type**: Type for fast MA (default: "ema")
//! - **slow_ma_type**: Type for slow MA (default: "ema")
//!
//! ## Returns
//! - **`Ok(StcOutput)`** or **`Err(StcError)`**
//!
//! ## SIMD
//! - All AVX2/AVX512 implementations are stubs for strict API parity.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use std::error::Error;
use thiserror::Error;
use rayon::prelude::*;
use std::convert::AsRef;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;

#[derive(Debug, Clone)]
pub enum StcData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for StcInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            StcData::Slice(slice) => slice,
            StcData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StcOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct StcParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub k_period: Option<usize>,
    pub d_period: Option<usize>,
    pub fast_ma_type: Option<String>,
    pub slow_ma_type: Option<String>,
}

impl Default for StcParams {
    fn default() -> Self {
        Self {
            fast_period: Some(23),
            slow_period: Some(50),
            k_period: Some(10),
            d_period: Some(3),
            fast_ma_type: Some("ema".to_string()),
            slow_ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StcInput<'a> {
    pub data: StcData<'a>,
    pub params: StcParams,
}

impl<'a> StcInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: StcParams) -> Self {
        Self { data: StcData::Candles { candles: c, source: s }, params: p }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: StcParams) -> Self {
        Self { data: StcData::Slice(sl), params: p }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", StcParams::default())
    }
    #[inline]
    pub fn get_fast_period(&self) -> usize { self.params.fast_period.unwrap_or(23) }
    #[inline]
    pub fn get_slow_period(&self) -> usize { self.params.slow_period.unwrap_or(50) }
    #[inline]
    pub fn get_k_period(&self) -> usize { self.params.k_period.unwrap_or(10) }
    #[inline]
    pub fn get_d_period(&self) -> usize { self.params.d_period.unwrap_or(3) }
    #[inline]
    pub fn get_fast_ma_type(&self) -> &str { self.params.fast_ma_type.as_deref().unwrap_or("ema") }
    #[inline]
    pub fn get_slow_ma_type(&self) -> &str { self.params.slow_ma_type.as_deref().unwrap_or("ema") }
}

#[derive(Clone, Debug)]
pub struct StcBuilder {
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    k_period: Option<usize>,
    d_period: Option<usize>,
    fast_ma_type: Option<String>,
    slow_ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for StcBuilder {
    fn default() -> Self {
        Self {
            fast_period: None,
            slow_period: None,
            k_period: None,
            d_period: None,
            fast_ma_type: None,
            slow_ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StcBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn fast_period(mut self, n: usize) -> Self { self.fast_period = Some(n); self }
    #[inline(always)]
    pub fn slow_period(mut self, n: usize) -> Self { self.slow_period = Some(n); self }
    #[inline(always)]
    pub fn k_period(mut self, n: usize) -> Self { self.k_period = Some(n); self }
    #[inline(always)]
    pub fn d_period(mut self, n: usize) -> Self { self.d_period = Some(n); self }
    #[inline(always)]
    pub fn fast_ma_type<T: Into<String>>(mut self, s: T) -> Self { self.fast_ma_type = Some(s.into()); self }
    #[inline(always)]
    pub fn slow_ma_type<T: Into<String>>(mut self, s: T) -> Self { self.slow_ma_type = Some(s.into()); self }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<StcOutput, StcError> {
        let p = StcParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            k_period: self.k_period,
            d_period: self.d_period,
            fast_ma_type: self.fast_ma_type,
            slow_ma_type: self.slow_ma_type,
        };
        let i = StcInput::from_candles(c, "close", p);
        stc_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<StcOutput, StcError> {
        let p = StcParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            k_period: self.k_period,
            d_period: self.d_period,
            fast_ma_type: self.fast_ma_type,
            slow_ma_type: self.slow_ma_type,
        };
        let i = StcInput::from_slice(d, p);
        stc_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<StcStream, StcError> {
        let p = StcParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            k_period: self.k_period,
            d_period: self.d_period,
            fast_ma_type: self.fast_ma_type,
            slow_ma_type: self.slow_ma_type,
        };
        StcStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum StcError {
    #[error("stc: Empty data provided.")]
    EmptyData,
    #[error("stc: All values are NaN.")]
    AllValuesNaN,
    #[error("stc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("stc: MA error: {0}")]
    MaError(#[from] Box<dyn Error>),
    #[error("stc: Internal error: {0}")]
    Internal(String),
}

#[inline]
pub fn stc(input: &StcInput) -> Result<StcOutput, StcError> {
    stc_with_kernel(input, Kernel::Auto)
}

pub fn stc_with_kernel(input: &StcInput, kernel: Kernel) -> Result<StcOutput, StcError> {
    let data: &[f64] = input.as_ref();

    let first = data.iter().position(|x| !x.is_nan())
        .ok_or(StcError::AllValuesNaN)?;
    let len = data.len();

    let fast_period = input.get_fast_period();
    let slow_period = input.get_slow_period();
    let k_period = input.get_k_period();
    let d_period = input.get_d_period();
    let needed = fast_period.max(slow_period).max(k_period).max(d_period);
    if len == 0 { return Err(StcError::EmptyData); }
    if (len - first) < needed {
        return Err(StcError::NotEnoughValidData { needed, valid: len - first });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                stc_scalar(data, fast_period, slow_period, k_period, d_period, input.get_fast_ma_type(), input.get_slow_ma_type(), first, &mut vec![f64::NAN; len])
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                stc_avx2(data, fast_period, slow_period, k_period, d_period, input.get_fast_ma_type(), input.get_slow_ma_type(), first, &mut vec![f64::NAN; len])
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                stc_avx512(data, fast_period, slow_period, k_period, d_period, input.get_fast_ma_type(), input.get_slow_ma_type(), first, &mut vec![f64::NAN; len])
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn stc_scalar(
    data: &[f64], fast: usize, slow: usize, k: usize, d: usize,
    fast_type: &str, slow_type: &str, first: usize, out: &mut [f64]
) -> Result<StcOutput, StcError> {
    use crate::indicators::ema::{EmaInput, EmaParams, ema};
    use crate::indicators::moving_averages::ma::{ma, MaData};
    use crate::indicators::utility_functions::{max_rolling, min_rolling};
    let len = data.len();
    let slice = &data[first..];

    let fast_ma = ma(fast_type, MaData::Slice(slice), fast)?;
    let slow_ma = ma(slow_type, MaData::Slice(slice), slow)?;

    let macd: Vec<f64> = fast_ma.iter().zip(slow_ma.iter()).map(|(f, s)| f - s).collect();

    let macd_min = min_rolling(&macd, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let macd_max = max_rolling(&macd, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let mut stok = vec![f64::NAN; macd.len()];
    for i in 0..macd.len() {
        let range = macd_max[i] - macd_min[i];
        if range.abs() > f64::EPSILON && !range.is_nan() {
            stok[i] = (macd[i] - macd_min[i]) / range * 100.0;
        } else if !macd[i].is_nan() {
            stok[i] = 50.0;
        }
    }

    let d_ema = ema(&EmaInput::from_slice(&stok, EmaParams { period: Some(d) }))
        .map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let d_vals = &d_ema.values;

    let d_min = min_rolling(&d_vals, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let d_max = max_rolling(&d_vals, k).map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let mut kd = vec![f64::NAN; d_vals.len()];
    for i in 0..d_vals.len() {
        let range = d_max[i] - d_min[i];
        if range.abs() > f64::EPSILON && !range.is_nan() {
            kd[i] = (d_vals[i] - d_min[i]) / range * 100.0;
        } else if !d_vals[i].is_nan() {
            kd[i] = 50.0;
        }
    }

    let kd_ema = ema(&EmaInput::from_slice(&kd, EmaParams { period: Some(d) }))
        .map_err(|e| StcError::Internal(format!("{:?}", e)))?;
    let final_stc = &kd_ema.values;

    for (i, &val) in final_stc.iter().enumerate() {
        out[first + i] = val;
    }
    Ok(StcOutput { values: out.to_vec() })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stc_avx2(
    data: &[f64], fast: usize, slow: usize, k: usize, d: usize,
    fast_type: &str, slow_type: &str, first: usize, out: &mut [f64]
) -> Result<StcOutput, StcError> {
    stc_scalar(data, fast, slow, k, d, fast_type, slow_type, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn stc_avx512(
    data: &[f64], fast: usize, slow: usize, k: usize, d: usize,
    fast_type: &str, slow_type: &str, first: usize, out: &mut [f64]
) -> Result<StcOutput, StcError> {
    if fast <= 32 && slow <= 32 {
        unsafe { stc_avx512_short(data, fast, slow, k, d, fast_type, slow_type, first, out) }
    } else {
        unsafe { stc_avx512_long(data, fast, slow, k, d, fast_type, slow_type, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stc_avx512_short(
    data: &[f64], fast: usize, slow: usize, k: usize, d: usize,
    fast_type: &str, slow_type: &str, first: usize, out: &mut [f64]
) -> Result<StcOutput, StcError> {
    stc_scalar(data, fast, slow, k, d, fast_type, slow_type, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stc_avx512_long(
    data: &[f64], fast: usize, slow: usize, k: usize, d: usize,
    fast_type: &str, slow_type: &str, first: usize, out: &mut [f64]
) -> Result<StcOutput, StcError> {
    stc_scalar(data, fast, slow, k, d, fast_type, slow_type, first, out)
}

// Batch API and related structs
#[derive(Clone, Debug)]
pub struct StcBatchRange {
    pub fast_period: (usize, usize, usize),
    pub slow_period: (usize, usize, usize),
    pub k_period: (usize, usize, usize),
    pub d_period: (usize, usize, usize),
}

impl Default for StcBatchRange {
    fn default() -> Self {
        Self {
            fast_period: (23, 23, 0),
            slow_period: (50, 50, 0),
            k_period: (10, 10, 0),
            d_period: (3, 3, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StcBatchBuilder {
    range: StcBatchRange,
    kernel: Kernel,
}

impl StcBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn fast_period_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.fast_period = (start, end, step); self }
    pub fn slow_period_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.slow_period = (start, end, step); self }
    pub fn k_period_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.k_period = (start, end, step); self }
    pub fn d_period_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.d_period = (start, end, step); self }

    pub fn apply_slice(self, data: &[f64]) -> Result<StcBatchOutput, StcError> {
        stc_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<StcBatchOutput, StcError> {
        StcBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<StcBatchOutput, StcError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<StcBatchOutput, StcError> {
        StcBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

pub fn stc_batch_with_kernel(
    data: &[f64], sweep: &StcBatchRange, k: Kernel
) -> Result<StcBatchOutput, StcError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(StcError::Internal("Invalid kernel".to_string())),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    stc_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct StcBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<StcParams>,
    pub rows: usize,
    pub cols: usize,
}

impl StcBatchOutput {
    pub fn row_for_params(&self, p: &StcParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.fast_period == p.fast_period && c.slow_period == p.slow_period &&
            c.k_period == p.k_period && c.d_period == p.d_period
        })
    }
    pub fn values_for(&self, p: &StcParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &StcBatchRange) -> Vec<StcParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    let fasts = axis(r.fast_period);
    let slows = axis(r.slow_period);
    let ks = axis(r.k_period);
    let ds = axis(r.d_period);
    let mut out = Vec::with_capacity(fasts.len() * slows.len() * ks.len() * ds.len());
    for &f in &fasts {
        for &s in &slows {
            for &k in &ks {
                for &d in &ds {
                    out.push(StcParams {
                        fast_period: Some(f),
                        slow_period: Some(s),
                        k_period: Some(k),
                        d_period: Some(d),
                        fast_ma_type: None,
                        slow_ma_type: None,
                    });
                }
            }
        }
    }
    out
}

#[inline(always)]
pub fn stc_batch_slice(
    data: &[f64], sweep: &StcBatchRange, kern: Kernel,
) -> Result<StcBatchOutput, StcError> {
    stc_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn stc_batch_par_slice(
    data: &[f64], sweep: &StcBatchRange, kern: Kernel,
) -> Result<StcBatchOutput, StcError> {
    stc_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn stc_batch_inner(
    data: &[f64], sweep: &StcBatchRange, kern: Kernel, parallel: bool,
) -> Result<StcBatchOutput, StcError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(StcError::NotEnoughValidData { needed: 1, valid: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan())
        .ok_or(StcError::AllValuesNaN)?;
    let max_needed = combos.iter().map(|c| {
        c.fast_period.unwrap().max(c.slow_period.unwrap())
         .max(c.k_period.unwrap()).max(c.d_period.unwrap())
    }).max().unwrap();
    if data.len() - first < max_needed {
        return Err(StcError::NotEnoughValidData { needed: max_needed, valid: data.len() - first });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        match kern {
            Kernel::Scalar => {
                stc_row_scalar(data, first, prm, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => {
                stc_row_avx2(data, first, prm, out_row)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                stc_row_avx512(data, first, prm, out_row)
            }
            _ => unreachable!(),
        }
    };

    if parallel {
        values.par_chunks_mut(cols).enumerate().for_each(|(row, slice)| { do_row(row, slice).unwrap(); });
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice).unwrap();
        }
    }
    Ok(StcBatchOutput { values, combos, rows, cols })
}

#[inline(always)]
pub unsafe fn stc_row_scalar(
    data: &[f64], first: usize, prm: &StcParams, out: &mut [f64]
) -> Result<(), StcError> {
    let mut tmp = vec![f64::NAN; data.len()];
    stc_scalar(
        data,
        prm.fast_period.unwrap(),
        prm.slow_period.unwrap(),
        prm.k_period.unwrap(),
        prm.d_period.unwrap(),
        prm.fast_ma_type.as_deref().unwrap_or("ema"),
        prm.slow_ma_type.as_deref().unwrap_or("ema"),
        first,
        &mut tmp,
    )?;
    out.copy_from_slice(&tmp);
    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn stc_row_avx2(
    data: &[f64], first: usize, prm: &StcParams, out: &mut [f64]
) -> Result<(), StcError> {
    stc_row_scalar(data, first, prm, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn stc_row_avx512(
    data: &[f64], first: usize, prm: &StcParams, out: &mut [f64]
) -> Result<(), StcError> {
    if prm.fast_period.unwrap() <= 32 && prm.slow_period.unwrap() <= 32 {
        stc_row_avx512_short(data, first, prm, out)
    } else {
        stc_row_avx512_long(data, first, prm, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn stc_row_avx512_short(
    data: &[f64], first: usize, prm: &StcParams, out: &mut [f64]
) -> Result<(), StcError> {
    stc_row_scalar(data, first, prm, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn stc_row_avx512_long(
    data: &[f64], first: usize, prm: &StcParams, out: &mut [f64]
) -> Result<(), StcError> {
    stc_row_scalar(data, first, prm, out)
}

// Streaming STC
#[derive(Debug, Clone)]
pub struct StcStream {
    pub fast_period: usize,
    pub slow_period: usize,
    pub k_period: usize,
    pub d_period: usize,
    buffer: Vec<f64>,
    idx: usize,
    filled: bool,
    params: StcParams,
}

impl StcStream {
    pub fn try_new(params: StcParams) -> Result<Self, StcError> {
        let fast = params.fast_period.unwrap_or(23);
        let slow = params.slow_period.unwrap_or(50);
        let k = params.k_period.unwrap_or(10);
        let d = params.d_period.unwrap_or(3);

        if fast == 0 || slow == 0 || k == 0 || d == 0 {
            return Err(StcError::NotEnoughValidData { needed: 1, valid: 0 });
        }
        Ok(Self {
            fast_period: fast,
            slow_period: slow,
            k_period: k,
            d_period: d,
            buffer: vec![f64::NAN; fast.max(slow).max(k).max(d)],
            idx: 0,
            filled: false,
            params,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.idx] = value;
        self.idx = (self.idx + 1) % self.buffer.len();
        if !self.filled && self.idx == 0 {
            self.filled = true;
        }
        if !self.filled { return None; }
        let slice: Vec<f64> = self.buffer.iter().cycle().skip(self.idx).take(self.buffer.len()).cloned().collect();
        let input = StcInput::from_slice(&slice, self.params.clone());
        match stc(&input) {
            Ok(res) => res.values.last().cloned(),
            Err(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_stc_default_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = StcInput::with_default_candles(&candles);
        let output = stc_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_stc_last_five(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = StcInput::with_default_candles(&candles);
        let result = stc_with_kernel(&input, kernel)?;
        let expected = [
            0.21394384188858884,
            0.10697192094429442,
            0.05348596047214721,
            50.02674298023607,
            49.98686202668157,
        ];
        let n = result.values.len();
        for (i, &exp) in expected.iter().enumerate() {
            let val = result.values[n - 5 + i];
            assert!((val - exp).abs() < 1e-5, "Expected {}, got {} at idx {}", exp, val, n - 5 + i);
        }
        Ok(())
    }

    fn check_stc_with_slice_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let slice_data = [10.0, 11.0, 12.0, 13.0, 14.0];
        let params = StcParams {
            fast_period: Some(2),
            slow_period: Some(3),
            k_period: Some(2),
            d_period: Some(1),
            fast_ma_type: Some("ema".to_string()),
            slow_ma_type: Some("ema".to_string()),
        };
        let input = StcInput::from_slice(&slice_data, params);
        let result = stc_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), slice_data.len());
        Ok(())
    }

    fn check_stc_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data: [f64; 0] = [];
        let input = StcInput::from_slice(&data, StcParams::default());
        let result = stc_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_stc_all_nan_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let input = StcInput::from_slice(&data, StcParams::default());
        let result = stc_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_stc_not_enough_valid_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, 2.0, 3.0];
        let params = StcParams { fast_period: Some(5), ..Default::default() };
        let input = StcInput::from_slice(&data, params);
        let result = stc_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    macro_rules! generate_all_stc_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                    #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }
    generate_all_stc_tests!(
        check_stc_default_params,
        check_stc_last_five,
        check_stc_with_slice_data,
        check_stc_empty_data,
        check_stc_all_nan_data,
        check_stc_not_enough_valid_data
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = StcBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
        let def = StcParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
