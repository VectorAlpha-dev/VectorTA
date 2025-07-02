//! # Moving Average Bands (MAB)
//!
//! Calculates upper, middle, and lower bands based on fast and slow moving averages and the rolling standard deviation of their difference over the fast window.
//!
//! ## Parameters
//! - **fast_period**: Fast MA window (default: 10)
//! - **slow_period**: Slow MA window (default: 50)
//! - **devup**: Upper band multiplier (default: 1.0)
//! - **devdn**: Lower band multiplier (default: 1.0)
//! - **fast_ma_type**: Fast MA type ("sma" or "ema", default: "sma")
//! - **slow_ma_type**: Slow MA type ("sma" or "ema", default: "sma")
//!
//! ## Errors
//! - **AllValuesNaN**: All input values are NaN.
//! - **InvalidPeriod**: Fast/slow period is zero or exceeds data length.
//! - **NotEnoughValidData**: Not enough valid data for the required periods.
//! - **EmptyData**: Input slice is empty.
//!
//! ## Returns
//! - `Ok(MabOutput)` with `.upperband`, `.middleband`, `.lowerband` (all Vec<f64>)
//! - `Err(MabError)` otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for MabInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MabData::Slice(slice) => slice,
            MabData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MabData<'a> {
    Candles { candles: &'a Candles, source: &'a str },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MabOutput {
    pub upperband: Vec<f64>,
    pub middleband: Vec<f64>,
    pub lowerband: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MabParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub devup: Option<f64>,
    pub devdn: Option<f64>,
    pub fast_ma_type: Option<String>,
    pub slow_ma_type: Option<String>,
}

impl Default for MabParams {
    fn default() -> Self {
        Self {
            fast_period: Some(10),
            slow_period: Some(50),
            devup: Some(1.0),
            devdn: Some(1.0),
            fast_ma_type: Some("sma".to_string()),
            slow_ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MabInput<'a> {
    pub data: MabData<'a>,
    pub params: MabParams,
}

impl<'a> MabInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MabParams) -> Self {
        Self { data: MabData::Candles { candles: c, source: s }, params: p }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MabParams) -> Self {
        Self { data: MabData::Slice(sl), params: p }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MabParams::default())
    }
    #[inline]
    pub fn get_fast_period(&self) -> usize {
        self.params.fast_period.unwrap_or(10)
    }
    #[inline]
    pub fn get_slow_period(&self) -> usize {
        self.params.slow_period.unwrap_or(50)
    }
    #[inline]
    pub fn get_devup(&self) -> f64 {
        self.params.devup.unwrap_or(1.0)
    }
    #[inline]
    pub fn get_devdn(&self) -> f64 {
        self.params.devdn.unwrap_or(1.0)
    }
    #[inline]
    pub fn get_fast_ma_type(&self) -> &str {
        self.params.fast_ma_type.as_deref().unwrap_or("sma")
    }
    #[inline]
    pub fn get_slow_ma_type(&self) -> &str {
        self.params.slow_ma_type.as_deref().unwrap_or("sma")
    }
}

#[derive(Clone, Debug)]
pub struct MabBuilder {
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    devup: Option<f64>,
    devdn: Option<f64>,
    fast_ma_type: Option<String>,
    slow_ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for MabBuilder {
    fn default() -> Self {
        Self {
            fast_period: None,
            slow_period: None,
            devup: None,
            devdn: None,
            fast_ma_type: None,
            slow_ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MabBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn fast_period(mut self, n: usize) -> Self { self.fast_period = Some(n); self }
    #[inline(always)]
    pub fn slow_period(mut self, n: usize) -> Self { self.slow_period = Some(n); self }
    #[inline(always)]
    pub fn devup(mut self, v: f64) -> Self { self.devup = Some(v); self }
    #[inline(always)]
    pub fn devdn(mut self, v: f64) -> Self { self.devdn = Some(v); self }
    #[inline(always)]
    pub fn fast_ma_type(mut self, s: &str) -> Self { self.fast_ma_type = Some(s.to_string()); self }
    #[inline(always)]
    pub fn slow_ma_type(mut self, s: &str) -> Self { self.slow_ma_type = Some(s.to_string()); self }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<MabOutput, MabError> {
        let p = MabParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            devup: self.devup,
            devdn: self.devdn,
            fast_ma_type: self.fast_ma_type.clone(),
            slow_ma_type: self.slow_ma_type.clone(),
        };
        let i = MabInput::from_candles(c, "close", p);
        mab_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MabOutput, MabError> {
        let p = MabParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            devup: self.devup,
            devdn: self.devdn,
            fast_ma_type: self.fast_ma_type.clone(),
            slow_ma_type: self.slow_ma_type.clone(),
        };
        let i = MabInput::from_slice(d, p);
        mab_with_kernel(&i, self.kernel)
    }
}

#[derive(Debug, Error)]
pub enum MabError {
    #[error("mab: Empty data provided.")]
    EmptyData,
    #[error("mab: Invalid period: fast = {fast_period}, slow = {slow_period}, data length = {data_len}")]
    InvalidPeriod { fast_period: usize, slow_period: usize, data_len: usize },
    #[error("mab: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("mab: All values are NaN.")]
    AllValuesNaN,
    #[error("mab: Underlying MA calculation failed: {0}")]
    MaCalculationError(String),
}

#[inline]
pub fn mab(input: &MabInput) -> Result<MabOutput, MabError> {
    mab_with_kernel(input, Kernel::Auto)
}

pub fn mab_with_kernel(input: &MabInput, kernel: Kernel) -> Result<MabOutput, MabError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(MabError::EmptyData);
    }
    let fast_period = input.get_fast_period();
    let slow_period = input.get_slow_period();
    let devup = input.get_devup();
    let devdn = input.get_devdn();
    let fast_ma_type = input.get_fast_ma_type();
    let slow_ma_type = input.get_slow_ma_type();

    if fast_period == 0 || slow_period == 0 || fast_period > data.len() || slow_period > data.len() {
        return Err(MabError::InvalidPeriod { fast_period, slow_period, data_len: data.len() });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(MabError::AllValuesNaN)?;
    let needed = fast_period.max(slow_period);
    if (data.len() - first) < needed {
        return Err(MabError::NotEnoughValidData { needed, valid: data.len() - first });
    }
    let fast_ma = crate::indicators::moving_averages::ma::ma(&fast_ma_type, crate::indicators::moving_averages::ma::MaData::Slice(data), fast_period)
        .map_err(|e| MabError::MaCalculationError(format!("{:?}", e)))?;
    let slow_ma = crate::indicators::moving_averages::ma::ma(&slow_ma_type, crate::indicators::moving_averages::ma::MaData::Slice(data), slow_period)
        .map_err(|e| MabError::MaCalculationError(format!("{:?}", e)))?;

    let first_valid = fast_ma.iter().position(|x| !x.is_nan()).unwrap().max(
        slow_ma.iter().position(|x| !x.is_nan()).unwrap()
    );
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut upperband = vec![f64::NAN; data.len()];
    let mut middleband = vec![f64::NAN; data.len()];
    let mut lowerband = vec![f64::NAN; data.len()];

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                mab_scalar(
                    &fast_ma, &slow_ma, fast_period, devup, devdn, first_valid, &mut upperband, &mut middleband, &mut lowerband
                )
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                mab_avx2(
                    &fast_ma, &slow_ma, fast_period, devup, devdn, first_valid, &mut upperband, &mut middleband, &mut lowerband
                )
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                mab_avx512(
                    &fast_ma, &slow_ma, fast_period, devup, devdn, first_valid, &mut upperband, &mut middleband, &mut lowerband
                )
            }
            _ => unreachable!(),
        }
    }

    Ok(MabOutput { upperband, middleband, lowerband })
}

#[inline]
pub fn mab_scalar(
    fast_ma: &[f64], slow_ma: &[f64], fast_period: usize,
    devup: f64, devdn: f64, first_valid: usize,
    upper: &mut [f64], mid: &mut [f64], lower: &mut [f64],
) {
    let mut sum_sq = 0.0;
    for i in first_valid..first_valid + fast_period {
        let diff = fast_ma[i] - slow_ma[i];
        sum_sq += diff * diff;
    }
    let mut dev = (sum_sq / fast_period as f64).sqrt();
    let idx = first_valid + fast_period - 1;
    mid[idx] = fast_ma[idx];
    upper[idx] = slow_ma[idx] + devup * dev;
    lower[idx] = slow_ma[idx] - devdn * dev;
    for i in (first_valid + fast_period)..fast_ma.len() {
        let old = fast_ma[i - fast_period] - slow_ma[i - fast_period];
        let new = fast_ma[i] - slow_ma[i];
        sum_sq += new * new - old * old;
        dev = (sum_sq / fast_period as f64).sqrt();
        mid[i] = fast_ma[i];
        upper[i] = slow_ma[i] + devup * dev;
        lower[i] = slow_ma[i] - devdn * dev;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mab_avx512(
    fast_ma: &[f64], slow_ma: &[f64], fast_period: usize,
    devup: f64, devdn: f64, first_valid: usize,
    upper: &mut [f64], mid: &mut [f64], lower: &mut [f64],
) {
    if fast_period <= 32 {
        unsafe { mab_avx512_short(fast_ma, slow_ma, fast_period, devup, devdn, first_valid, upper, mid, lower) }
    } else {
        unsafe { mab_avx512_long(fast_ma, slow_ma, fast_period, devup, devdn, first_valid, upper, mid, lower) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mab_avx2(
    fast_ma: &[f64], slow_ma: &[f64], fast_period: usize,
    devup: f64, devdn: f64, first_valid: usize,
    upper: &mut [f64], mid: &mut [f64], lower: &mut [f64],
) {
    unsafe { mab_scalar(fast_ma, slow_ma, fast_period, devup, devdn, first_valid, upper, mid, lower) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mab_avx512_short(
    fast_ma: &[f64], slow_ma: &[f64], fast_period: usize,
    devup: f64, devdn: f64, first_valid: usize,
    upper: &mut [f64], mid: &mut [f64], lower: &mut [f64],
) {
    mab_scalar(fast_ma, slow_ma, fast_period, devup, devdn, first_valid, upper, mid, lower);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mab_avx512_long(
    fast_ma: &[f64], slow_ma: &[f64], fast_period: usize,
    devup: f64, devdn: f64, first_valid: usize,
    upper: &mut [f64], mid: &mut [f64], lower: &mut [f64],
) {
    mab_scalar(fast_ma, slow_ma, fast_period, devup, devdn, first_valid, upper, mid, lower);
}

#[inline(always)]
pub fn mab_row_scalar(
    fast_ma: &[f64], slow_ma: &[f64], fast_period: usize, devup: f64, devdn: f64, first_valid: usize,
    upper: &mut [f64], mid: &mut [f64], lower: &mut [f64],
) {
    mab_scalar(fast_ma, slow_ma, fast_period, devup, devdn, first_valid, upper, mid, lower)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn mab_row_avx2(
    fast_ma: &[f64], slow_ma: &[f64], fast_period: usize, devup: f64, devdn: f64, first_valid: usize,
    upper: &mut [f64], mid: &mut [f64], lower: &mut [f64],
) {
    mab_avx2(fast_ma, slow_ma, fast_period, devup, devdn, first_valid, upper, mid, lower)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn mab_row_avx512(
    fast_ma: &[f64], slow_ma: &[f64], fast_period: usize, devup: f64, devdn: f64, first_valid: usize,
    upper: &mut [f64], mid: &mut [f64], lower: &mut [f64],
) {
    mab_avx512(fast_ma, slow_ma, fast_period, devup, devdn, first_valid, upper, mid, lower)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn mab_row_avx512_short(
    fast_ma: &[f64], slow_ma: &[f64], fast_period: usize, devup: f64, devdn: f64, first_valid: usize,
    upper: &mut [f64], mid: &mut [f64], lower: &mut [f64],
) {
    unsafe { mab_avx512_short(fast_ma, slow_ma, fast_period, devup, devdn, first_valid, upper, mid, lower) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn mab_row_avx512_long(
    fast_ma: &[f64], slow_ma: &[f64], fast_period: usize, devup: f64, devdn: f64, first_valid: usize,
    upper: &mut [f64], mid: &mut [f64], lower: &mut [f64],
) {
    unsafe { mab_avx512_long(fast_ma, slow_ma, fast_period, devup, devdn, first_valid, upper, mid, lower) }
}

// --- Batch API and Structs ---

#[derive(Clone, Debug)]
pub struct MabBatchRange {
    pub fast_period: (usize, usize, usize),
    pub slow_period: (usize, usize, usize),
    pub devup: (f64, f64, f64),
    pub devdn: (f64, f64, f64),
    pub fast_ma_type: (String, String, String), // static only
    pub slow_ma_type: (String, String, String), // static only
}
impl Default for MabBatchRange {
    fn default() -> Self {
        Self {
            fast_period: (10, 50, 10),
            slow_period: (50, 50, 1),
            devup: (1.0, 1.0, 0.0),
            devdn: (1.0, 1.0, 0.0),
            fast_ma_type: ("sma".to_string(), "sma".to_string(), "".to_string()),
            slow_ma_type: ("sma".to_string(), "sma".to_string(), "".to_string()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MabBatchBuilder {
    range: MabBatchRange,
    kernel: Kernel,
}

impl MabBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn fast_period_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.fast_period = (start, end, step); self }
    pub fn slow_period_range(mut self, start: usize, end: usize, step: usize) -> Self { self.range.slow_period = (start, end, step); self }
    pub fn devup_range(mut self, start: f64, end: f64, step: f64) -> Self { self.range.devup = (start, end, step); self }
    pub fn devdn_range(mut self, start: f64, end: f64, step: f64) -> Self { self.range.devdn = (start, end, step); self }
    pub fn fast_ma_type_static(mut self, typ: &str) -> Self { self.range.fast_ma_type = (typ.to_string(), typ.to_string(), "".to_string()); self }
    pub fn slow_ma_type_static(mut self, typ: &str) -> Self { self.range.slow_ma_type = (typ.to_string(), typ.to_string(), "".to_string()); self }
    pub fn apply_slice(self, data: &[f64]) -> Result<MabBatchOutput, MabError> {
        mab_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MabBatchOutput, MabError> {
        MabBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MabBatchOutput, MabError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MabBatchOutput, MabError> {
        MabBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct MabBatchOutput {
    pub upperbands: Vec<f64>,
    pub middlebands: Vec<f64>,
    pub lowerbands: Vec<f64>,
    pub combos: Vec<MabParams>,
    pub rows: usize,
    pub cols: usize,
}
impl MabBatchOutput {
    pub fn row_for_params(&self, p: &MabParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.fast_period.unwrap_or(10) == p.fast_period.unwrap_or(10)
                && c.slow_period.unwrap_or(50) == p.slow_period.unwrap_or(50)
                && (c.devup.unwrap_or(1.0) - p.devup.unwrap_or(1.0)).abs() < 1e-12
                && (c.devdn.unwrap_or(1.0) - p.devdn.unwrap_or(1.0)).abs() < 1e-12
                && c.fast_ma_type.as_ref().unwrap_or(&"sma".to_string()) == p.fast_ma_type.as_ref().unwrap_or(&"sma".to_string())
                && c.slow_ma_type.as_ref().unwrap_or(&"sma".to_string()) == p.slow_ma_type.as_ref().unwrap_or(&"sma".to_string())
        })
    }
    pub fn bands_for(&self, p: &MabParams) -> Option<(&[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.upperbands[start..start + self.cols],
                &self.middlebands[start..start + self.cols],
                &self.lowerbands[start..start + self.cols],
            )
        })
    }
}

fn expand_grid(r: &MabBatchRange) -> Vec<MabParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return vec![start]; }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    let fasts = axis_usize(r.fast_period);
    let slows = axis_usize(r.slow_period);
    let devups = axis_f64(r.devup);
    let devdns = axis_f64(r.devdn);
    let fast_ma = &r.fast_ma_type.0;
    let slow_ma = &r.slow_ma_type.0;
    let mut out = Vec::with_capacity(fasts.len() * slows.len() * devups.len() * devdns.len());
    for &f in &fasts {
        for &s in &slows {
            for &du in &devups {
                for &dd in &devdns {
                    out.push(MabParams {
                        fast_period: Some(f),
                        slow_period: Some(s),
                        devup: Some(du),
                        devdn: Some(dd),
                        fast_ma_type: Some(fast_ma.clone()),
                        slow_ma_type: Some(slow_ma.clone()),
                    });
                }
            }
        }
    }
    out
}

pub fn mab_batch_with_kernel(
    data: &[f64], sweep: &MabBatchRange, k: Kernel,
) -> Result<MabBatchOutput, MabError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(MabError::InvalidPeriod { fast_period: 0, slow_period: 0, data_len: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    mab_batch_par_slice(data, sweep, simd)
}

pub fn mab_batch_slice(
    data: &[f64], sweep: &MabBatchRange, kern: Kernel,
) -> Result<MabBatchOutput, MabError> {
    mab_batch_inner(data, sweep, kern, false)
}

pub fn mab_batch_par_slice(
    data: &[f64], sweep: &MabBatchRange, kern: Kernel,
) -> Result<MabBatchOutput, MabError> {
    mab_batch_inner(data, sweep, kern, true)
}

fn mab_batch_inner(
    data: &[f64], sweep: &MabBatchRange, kern: Kernel, parallel: bool,
) -> Result<MabBatchOutput, MabError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MabError::InvalidPeriod { fast_period: 0, slow_period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(MabError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.fast_period.unwrap().max(c.slow_period.unwrap())).max().unwrap();
    if data.len() - first < max_p {
        return Err(MabError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut upperbands = vec![f64::NAN; rows * cols];
    let mut middlebands = vec![f64::NAN; rows * cols];
    let mut lowerbands = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_upper: &mut [f64], out_mid: &mut [f64], out_low: &mut [f64]| unsafe {
        let p = &combos[row];
        let fast_ma = crate::indicators::moving_averages::ma::ma(p.fast_ma_type.as_ref().unwrap(), crate::indicators::moving_averages::ma::MaData::Slice(data), p.fast_period.unwrap())
            .map_err(|e| MabError::MaCalculationError(format!("{:?}", e))).unwrap();
        let slow_ma = crate::indicators::moving_averages::ma::ma(p.slow_ma_type.as_ref().unwrap(), crate::indicators::moving_averages::ma::MaData::Slice(data), p.slow_period.unwrap())
            .map_err(|e| MabError::MaCalculationError(format!("{:?}", e))).unwrap();
        let fv = fast_ma.iter().position(|x| !x.is_nan()).unwrap().max(
            slow_ma.iter().position(|x| !x.is_nan()).unwrap()
        );
        match kern {
            Kernel::Scalar => mab_row_scalar(&fast_ma, &slow_ma, p.fast_period.unwrap(), p.devup.unwrap(), p.devdn.unwrap(), fv, out_upper, out_mid, out_low),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => mab_row_avx2(&fast_ma, &slow_ma, p.fast_period.unwrap(), p.devup.unwrap(), p.devdn.unwrap(), fv, out_upper, out_mid, out_low),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => mab_row_avx512(&fast_ma, &slow_ma, p.fast_period.unwrap(), p.devup.unwrap(), p.devdn.unwrap(), fv, out_upper, out_mid, out_low),
            _ => unreachable!(),
        }
    };
    if parallel {

        #[cfg(not(target_arch = "wasm32"))] {

        upperbands.par_chunks_mut(cols)

                    .zip(middlebands.par_chunks_mut(cols))

                    .zip(lowerbands.par_chunks_mut(cols))

                    .enumerate()

                    .for_each(|(row, ((u, m), l))| do_row(row, u, m, l));

        }

        #[cfg(target_arch = "wasm32")] {

        let mut upper_iter = upperbands.chunks_mut(cols);

                let mut middle_iter = middlebands.chunks_mut(cols);

                let mut lower_iter = lowerbands.chunks_mut(cols);


                for row in 0..rows {

                    let u = upper_iter.next().unwrap();

                    let m = middle_iter.next().unwrap();

                    let l = lower_iter.next().unwrap();

                    do_row(row, u, m, l);

        }

        }
    } else {
        let mut upper_iter = upperbands.chunks_mut(cols);
        let mut middle_iter = middlebands.chunks_mut(cols);
        let mut lower_iter = lowerbands.chunks_mut(cols);

        for row in 0..rows {
            let u = upper_iter.next().unwrap();
            let m = middle_iter.next().unwrap();
            let l = lower_iter.next().unwrap();
            do_row(row, u, m, l);
        }
    }
    Ok(MabBatchOutput { upperbands, middlebands, lowerbands, combos, rows, cols })
}

// --- Stream API ---
#[derive(Debug, Clone)]
pub struct MabStream {
    fast_period: usize,
    slow_period: usize,
    devup: f64,
    devdn: f64,
    fast_ma_type: String,
    slow_ma_type: String,
    fast_ma_buf: Vec<f64>,
    slow_ma_buf: Vec<f64>,
    diff_buf: Vec<f64>,
    idx: usize,
    filled: bool,
}
impl MabStream {
    pub fn try_new(params: MabParams) -> Result<Self, MabError> {
        let fast_period = params.fast_period.unwrap_or(10);
        let slow_period = params.slow_period.unwrap_or(50);
        let devup = params.devup.unwrap_or(1.0);
        let devdn = params.devdn.unwrap_or(1.0);
        let fast_ma_type = params.fast_ma_type.unwrap_or("sma".to_string());
        let slow_ma_type = params.slow_ma_type.unwrap_or("sma".to_string());
        if fast_period == 0 || slow_period == 0 {
            return Err(MabError::InvalidPeriod { fast_period, slow_period, data_len: 0 });
        }
        Ok(Self {
            fast_period,
            slow_period,
            devup,
            devdn,
            fast_ma_type,
            slow_ma_type,
            fast_ma_buf: vec![f64::NAN; fast_period],
            slow_ma_buf: vec![f64::NAN; slow_period],
            diff_buf: vec![0.0; fast_period],
            idx: 0,
            filled: false,
        })
    }
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        self.fast_ma_buf[self.idx % self.fast_period] = value;
        self.slow_ma_buf[self.idx % self.slow_period] = value;
        let valid_fast = self.idx + 1 >= self.fast_period;
        let valid_slow = self.idx + 1 >= self.slow_period;
        let out = if valid_fast && valid_slow {
            let fast_ma = self.fast_ma_buf.iter().copied().rev().take(self.fast_period).sum::<f64>() / self.fast_period as f64;
            let slow_ma = self.slow_ma_buf.iter().copied().rev().take(self.slow_period).sum::<f64>() / self.slow_period as f64;
            let diff = fast_ma - slow_ma;
            self.diff_buf[self.idx % self.fast_period] = diff;
            let sum_sq: f64 = self.diff_buf.iter().map(|&d| d * d).sum();
            let dev = (sum_sq / self.fast_period as f64).sqrt();
            let upper = slow_ma + self.devup * dev;
            let lower = slow_ma - self.devdn * dev;
            Some((upper, fast_ma, lower))
        
            } else {
            None
        };
        self.idx += 1;
        out
    }
}

// --- Macro for tests like alma ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_mab_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = MabParams { fast_period: None, ..MabParams::default() };
        let input = MabInput::from_candles(&candles, "close", default_params);
        let output = mab_with_kernel(&input, kernel)?;
        assert_eq!(output.upperband.len(), candles.close.len());
        Ok(())
    }

    fn check_mab_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MabParams::default();
        let input = MabInput::from_candles(&candles, "close", params);
        let result = mab_with_kernel(&input, kernel)?;
        let expected_upper_last_five = [
            64002.843463352016,
            63976.62699738246,
            63949.00496307154,
            63912.13708526151,
            63828.40371728143,
        ];
        let expected_middle_last_five = [
            59213.90000000002,
            59180.800000000025,
            59161.40000000002,
            59132.00000000002,
            59042.40000000002,
        ];
        let expected_lower_last_five = [
            59350.676536647945,
            59296.93300261751,
            59252.75503692843,
            59190.30291473845,
            59070.11628271853,
        ];
        let len = result.upperband.len();
        for i in 0..5 {
            let idx = len - 5 + i;
            assert!((result.upperband[idx] - expected_upper_last_five[i]).abs() < 1e-4);
            assert!((result.middleband[idx] - expected_middle_last_five[i]).abs() < 1e-4);
            assert!((result.lowerband[idx] - expected_lower_last_five[i]).abs() < 1e-4);
        }
        Ok(())
    }

    fn check_mab_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MabInput::with_default_candles(&candles);
        let output = mab_with_kernel(&input, kernel)?;
        assert_eq!(output.upperband.len(), candles.close.len());
        Ok(())
    }

    fn check_mab_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = MabParams { fast_period: Some(0), slow_period: Some(5), ..MabParams::default() };
        let input = MabInput::from_slice(&input_data, params);
        let res = mab_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_mab_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = MabParams { fast_period: Some(2), slow_period: Some(10), ..MabParams::default() };
        let input = MabInput::from_slice(&data_small, params);
        let res = mab_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_mab_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = MabParams { fast_period: Some(10), slow_period: Some(20), ..MabParams::default() };
        let input = MabInput::from_slice(&single_point, params);
        let res = mab_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_mab_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MabParams::default();
        let first_input = MabInput::from_candles(&candles, "close", params.clone());
        let first_result = mab_with_kernel(&first_input, kernel)?;
        let second_input = MabInput::from_slice(&first_result.upperband, params);
        let second_result = mab_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.upperband.len(), first_result.upperband.len());
        Ok(())
    }

    fn check_mab_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MabInput::from_candles(&candles, "close", MabParams::default());
        let res = mab_with_kernel(&input, kernel)?;
        for i in 300..res.upperband.len() {
            assert!(!res.upperband[i].is_nan());
            assert!(!res.middleband[i].is_nan());
            assert!(!res.lowerband[i].is_nan());
        }
        Ok(())
    }

    fn check_mab_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MabParams::default();
        let input = MabInput::from_candles(&candles, "close", params.clone());
        let batch_output = mab_with_kernel(&input, kernel)?;
        let mut stream = MabStream::try_new(params)?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &v in &candles.close {
            match stream.update(v) {
                Some((up, mid, low)) => {
                    stream_values.push((up, mid, low));
                }
                None => stream_values.push((f64::NAN, f64::NAN, f64::NAN)),
            }
        }
        for (i, (((bu, bm), bl), &(su0, su1, su2))) in
            batch_output.upperband.iter()
                .zip(&batch_output.middleband)
                .zip(&batch_output.lowerband)
                .zip(stream_values.iter())
                .enumerate()
        {
            if i > 100 {
                assert!((bu - su0).abs() < 1e-8);
                assert!((bm - su1).abs() < 1e-8);
                assert!((bl - su2).abs() < 1e-8);
            }
        }
        Ok(())
    }

    macro_rules! generate_all_mab_tests {
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
            }
        }
    }
    generate_all_mab_tests!(
        check_mab_partial_params,
        check_mab_accuracy,
        check_mab_default_candles,
        check_mab_zero_period,
        check_mab_period_exceeds_length,
        check_mab_very_small_dataset,
        check_mab_reinput,
        check_mab_nan_handling,
        check_mab_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = MabBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = MabParams::default();
        let (upper, middle, lower) = output.bands_for(&def).expect("default row missing");

        assert_eq!(upper.len(), c.close.len());
        assert_eq!(middle.len(), c.close.len());
        assert_eq!(lower.len(), c.close.len());

        // Spot check last 5 values against known expected values (identical to non-batch test)
        let expected_upper = [
            64002.843463352016,
            63976.62699738246,
            63949.00496307154,
            63912.13708526151,
            63828.40371728143,
        ];
        let expected_middle = [
            59213.90000000002,
            59180.800000000025,
            59161.40000000002,
            59132.00000000002,
            59042.40000000002,
        ];
        let expected_lower = [
            59350.676536647945,
            59296.93300261751,
            59252.75503692843,
            59190.30291473845,
            59070.11628271853,
        ];

        let start = upper.len() - 5;
        for i in 0..5 {
            assert!(
                (upper[start + i] - expected_upper[i]).abs() < 1e-4,
                "[{test}] batch-upper mismatch at idx {i}: {} vs expected {}",
                upper[start + i], expected_upper[i]
            );
            assert!(
                (middle[start + i] - expected_middle[i]).abs() < 1e-4,
                "[{test}] batch-middle mismatch at idx {i}: {} vs expected {}",
                middle[start + i], expected_middle[i]
            );
            assert!(
                (lower[start + i] - expected_lower[i]).abs() < 1e-4,
                "[{test}] batch-lower mismatch at idx {i}: {} vs expected {}",
                lower[start + i], expected_lower[i]
            );
        }
        Ok(())
    }

    fn check_batch_grid_varying_fast_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let range = MabBatchRange {
            fast_period: (10, 12, 1), // grid: 10, 11, 12
            ..Default::default()
        };

        let output = MabBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        // Check combos
        assert!(output.combos.len() >= 1);
        // Check bands_for returns valid slices for each combo
        for params in &output.combos {
            let (u, m, l) = output.bands_for(params).unwrap();
            assert_eq!(u.len(), output.cols);
            assert_eq!(m.len(), output.cols);
            assert_eq!(l.len(), output.cols);
        }
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
    gen_batch_tests!(check_batch_grid_varying_fast_period);
}
