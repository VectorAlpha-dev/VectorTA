//! # Percentage Price Oscillator (PPO)
//!
//! Expresses the difference between two moving averages as a percentage of the slower MA.
//!
//! ## Parameters
//! - **data**: Input price data
//! - **fast_period**: Short-term MA period (default: 12)
//! - **slow_period**: Long-term MA period (default: 26)
//! - **ma_type**: Moving average type (default: "sma")
//!
//! ## Returns
//! - `Vec<f64>` - PPO values as percentage, matching input length
//!
//! ## Developer Status
//! **AVX2**: Stub (calls scalar)
//! **AVX512**: Has short/long variants but all stubs
//! **Streaming**: O(n) - Recalculates full MA on each update
//! **Memory**: Good - Uses `alloc_with_nan_prefix` and `make_uninit_matrix`

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for PpoInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PpoData::Slice(slice) => slice,
            PpoData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PpoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PpoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct PpoParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for PpoParams {
    fn default() -> Self {
        Self {
            fast_period: Some(12),
            slow_period: Some(26),
            ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PpoInput<'a> {
    pub data: PpoData<'a>,
    pub params: PpoParams,
}

impl<'a> PpoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: PpoParams) -> Self {
        Self {
            data: PpoData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: PpoParams) -> Self {
        Self {
            data: PpoData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", PpoParams::default())
    }
    #[inline]
    pub fn get_fast_period(&self) -> usize {
        self.params.fast_period.unwrap_or(12)
    }
    #[inline]
    pub fn get_slow_period(&self) -> usize {
        self.params.slow_period.unwrap_or(26)
    }
    #[inline]
    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .clone()
            .unwrap_or_else(|| "sma".to_string())
    }
}

#[derive(Clone, Debug)]
pub struct PpoBuilder {
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for PpoBuilder {
    fn default() -> Self {
        Self {
            fast_period: None,
            slow_period: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PpoBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn fast_period(mut self, n: usize) -> Self {
        self.fast_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn slow_period(mut self, n: usize) -> Self {
        self.slow_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn ma_type<S: Into<String>>(mut self, s: S) -> Self {
        self.ma_type = Some(s.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<PpoOutput, PpoError> {
        let p = PpoParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            ma_type: self.ma_type,
        };
        let i = PpoInput::from_candles(c, "close", p);
        ppo_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<PpoOutput, PpoError> {
        let p = PpoParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            ma_type: self.ma_type,
        };
        let i = PpoInput::from_slice(d, p);
        ppo_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<PpoStream, PpoError> {
        let p = PpoParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            ma_type: self.ma_type,
        };
        PpoStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum PpoError {
    #[error("ppo: All values are NaN.")]
    AllValuesNaN,
    #[error("ppo: Invalid period: fast = {fast}, slow = {slow}, data length = {data_len}")]
    InvalidPeriod {
        fast: usize,
        slow: usize,
        data_len: usize,
    },
    #[error("ppo: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ppo: MA error: {0}")]
    MaError(String),
}

#[inline]
pub fn ppo(input: &PpoInput) -> Result<PpoOutput, PpoError> {
    ppo_with_kernel(input, Kernel::Auto)
}

pub fn ppo_with_kernel(input: &PpoInput, kernel: Kernel) -> Result<PpoOutput, PpoError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PpoError::AllValuesNaN)?;

    let fast = input.get_fast_period();
    let slow = input.get_slow_period();
    let ma_type = input.get_ma_type();

    if fast == 0 || slow == 0 || fast > len || slow > len {
        return Err(PpoError::InvalidPeriod {
            fast,
            slow,
            data_len: len,
        });
    }
    if (len - first) < slow {
        return Err(PpoError::NotEnoughValidData {
            needed: slow,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    let mut out = alloc_with_nan_prefix(len, first + slow - 1);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                ppo_scalar(data, fast, slow, &ma_type, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                ppo_avx2(data, fast, slow, &ma_type, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                ppo_avx512(data, fast, slow, &ma_type, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(PpoOutput { values: out })
}

/// Write PPO result directly to output slice - no allocations
pub fn ppo_into_slice(dst: &mut [f64], input: &PpoInput, kern: Kernel) -> Result<(), PpoError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(PpoError::AllValuesNaN);
    }

    let fast = input.get_fast_period();
    let slow = input.get_slow_period();
    let ma_type = input.get_ma_type();

    if fast == 0 || slow == 0 || fast > data.len() || slow > data.len() {
        return Err(PpoError::InvalidPeriod {
            fast,
            slow,
            data_len: data.len(),
        });
    }
    if dst.len() != data.len() {
        return Err(PpoError::InvalidPeriod {
            fast,
            slow,
            data_len: data.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PpoError::AllValuesNaN)?;
    if data.len() - first < slow {
        return Err(PpoError::NotEnoughValidData {
            needed: slow,
            valid: data.len() - first,
        });
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                ppo_scalar(data, fast, slow, &ma_type, first, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ppo_avx2(data, fast, slow, &ma_type, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                ppo_avx512(data, fast, slow, &ma_type, first, dst)
            }
            _ => unreachable!(),
        }
    }

    let warmup_end = first + slow - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline]
pub unsafe fn ppo_scalar(
    data: &[f64],
    fast: usize,
    slow: usize,
    ma_type: &str,
    first: usize,
    out: &mut [f64],
) {
    // Check for classic kernel optimization
    if ma_type == "ema" {
        return ppo_scalar_classic_ema(data, fast, slow, first, out);
    } else if ma_type == "sma" {
        return ppo_scalar_classic_sma(data, fast, slow, first, out);
    }

    // Fall back to regular implementation for other MA types
    // MA failures should be impossible after validation; if they occur, write NaN and return.
    let fast_ma = match ma(ma_type, MaData::Slice(data), fast) {
        Ok(v) => v,
        Err(_) => {
            for i in (first + slow - 1)..data.len() {
                out[i] = f64::NAN;
            }
            return;
        }
    };
    let slow_ma = match ma(ma_type, MaData::Slice(data), slow) {
        Ok(v) => v,
        Err(_) => {
            for i in (first + slow - 1)..data.len() {
                out[i] = f64::NAN;
            }
            return;
        }
    };

    for i in (first + slow - 1)..data.len() {
        let sf = slow_ma[i];
        let ff = fast_ma[i];
        out[i] = if sf.is_nan() || ff.is_nan() || sf == 0.0 {
            f64::NAN
        } else {
            100.0 * (ff - sf) / sf
        };
    }
}

// Classic kernel with inline EMA calculations
#[inline]
pub unsafe fn ppo_scalar_classic_ema(
    data: &[f64],
    fast: usize,
    slow: usize,
    first: usize,
    out: &mut [f64],
) {
    // EMA alpha factors
    let fast_alpha = 2.0 / (fast as f64 + 1.0);
    let slow_alpha = 2.0 / (slow as f64 + 1.0);

    // Initialize EMAs with SMA
    let mut fast_sum = 0.0;
    let mut slow_sum = 0.0;

    // Calculate initial SMAs for EMA initialization
    for i in first..first + fast.min(data.len() - first) {
        fast_sum += data[i];
        if i < first + slow {
            slow_sum += data[i];
        }
    }

    for i in first + fast..first + slow.min(data.len() - first) {
        slow_sum += data[i];
    }

    let mut fast_ema = fast_sum / fast as f64;
    let mut slow_ema = slow_sum / slow as f64;

    // Process data with inline EMA calculations
    for i in first..data.len() {
        if i >= first + fast - 1 {
            if i == first + fast - 1 {
                // First EMA value is the SMA
                fast_ema = fast_sum / fast as f64;
            } else {
                // Update EMA
                fast_ema = fast_alpha * data[i] + (1.0 - fast_alpha) * fast_ema;
            }
        }

        if i >= first + slow - 1 {
            if i == first + slow - 1 {
                // First EMA value is the SMA
                slow_ema = slow_sum / slow as f64;
            } else {
                // Update EMA
                slow_ema = slow_alpha * data[i] + (1.0 - slow_alpha) * slow_ema;
            }

            // Calculate PPO
            out[i] = if slow_ema == 0.0 || slow_ema.is_nan() || fast_ema.is_nan() {
                f64::NAN
            } else {
                100.0 * (fast_ema - slow_ema) / slow_ema
            };
        }
    }
}

// Classic kernel with inline SMA calculations
#[inline]
pub unsafe fn ppo_scalar_classic_sma(
    data: &[f64],
    fast: usize,
    slow: usize,
    first: usize,
    out: &mut [f64],
) {
    // SMA calculation logic matching the exact behavior of sma_scalar

    // Calculate slow SMA sum starting from 'first'
    let mut slow_sum = 0.0;
    for i in 0..slow {
        slow_sum += data[first + i];
    }

    // For fast SMA at index (first + slow - 1), we need the window
    // from (first + slow - fast) to (first + slow - 1)
    let mut fast_sum = 0.0;
    let fast_start = first + slow - fast;
    for i in 0..fast {
        fast_sum += data[fast_start + i];
    }

    // First valid index for both SMAs is at (first + slow - 1)
    let start_idx = first + slow - 1;

    // Calculate first PPO value
    let fast_ma = fast_sum / fast as f64;
    let slow_ma = slow_sum / slow as f64;
    out[start_idx] = if slow_ma == 0.0 || slow_ma.is_nan() || fast_ma.is_nan() {
        f64::NAN
    } else {
        100.0 * (fast_ma - slow_ma) / slow_ma
    };

    // Process remaining data with rolling window
    for i in start_idx + 1..data.len() {
        // Update fast SMA rolling sum
        fast_sum += data[i] - data[i - fast];

        // Update slow SMA rolling sum
        slow_sum += data[i] - data[i - slow];

        // Calculate new SMAs
        let fast_ma = fast_sum / fast as f64;
        let slow_ma = slow_sum / slow as f64;

        // Calculate PPO
        out[i] = if slow_ma == 0.0 || slow_ma.is_nan() || fast_ma.is_nan() {
            f64::NAN
        } else {
            100.0 * (fast_ma - slow_ma) / slow_ma
        };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ppo_avx2(
    data: &[f64],
    fast: usize,
    slow: usize,
    ma_type: &str,
    first: usize,
    out: &mut [f64],
) {
    ppo_scalar(data, fast, slow, ma_type, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ppo_avx512(
    data: &[f64],
    fast: usize,
    slow: usize,
    ma_type: &str,
    first: usize,
    out: &mut [f64],
) {
    if slow <= 32 {
        ppo_avx512_short(data, fast, slow, ma_type, first, out)
    } else {
        ppo_avx512_long(data, fast, slow, ma_type, first, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ppo_avx512_short(
    data: &[f64],
    fast: usize,
    slow: usize,
    ma_type: &str,
    first: usize,
    out: &mut [f64],
) {
    ppo_scalar(data, fast, slow, ma_type, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ppo_avx512_long(
    data: &[f64],
    fast: usize,
    slow: usize,
    ma_type: &str,
    first: usize,
    out: &mut [f64],
) {
    ppo_scalar(data, fast, slow, ma_type, first, out)
}

#[derive(Clone, Debug)]
pub struct PpoBatchRange {
    pub fast_period: (usize, usize, usize),
    pub slow_period: (usize, usize, usize),
    pub ma_type: String,
}

impl Default for PpoBatchRange {
    fn default() -> Self {
        Self {
            fast_period: (12, 12, 0),
            slow_period: (26, 26, 0),
            ma_type: "sma".to_string(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PpoBatchBuilder {
    range: PpoBatchRange,
    kernel: Kernel,
}

impl PpoBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn fast_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_period = (start, end, step);
        self
    }
    pub fn slow_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_period = (start, end, step);
        self
    }
    pub fn ma_type<S: Into<String>>(mut self, t: S) -> Self {
        self.range.ma_type = t.into();
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<PpoBatchOutput, PpoError> {
        ppo_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<PpoBatchOutput, PpoError> {
        PpoBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<PpoBatchOutput, PpoError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<PpoBatchOutput, PpoError> {
        PpoBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct PpoBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PpoParams>,
    pub rows: usize,
    pub cols: usize,
}

impl PpoBatchOutput {
    pub fn row_for_params(&self, p: &PpoParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.fast_period.unwrap_or(12) == p.fast_period.unwrap_or(12)
                && c.slow_period.unwrap_or(26) == p.slow_period.unwrap_or(26)
                && c.ma_type.as_ref().unwrap_or(&"sma".to_string())
                    == p.ma_type.as_ref().unwrap_or(&"sma".to_string())
        })
    }
    pub fn values_for(&self, p: &PpoParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &PpoBatchRange) -> Vec<PpoParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let fasts = axis_usize(r.fast_period);
    let slows = axis_usize(r.slow_period);
    let ma_type = r.ma_type.clone();

    let mut out = Vec::with_capacity(fasts.len() * slows.len());
    for &f in &fasts {
        for &s in &slows {
            out.push(PpoParams {
                fast_period: Some(f),
                slow_period: Some(s),
                ma_type: Some(ma_type.clone()),
            });
        }
    }
    out
}

#[inline(always)]
pub fn ppo_batch_with_kernel(
    data: &[f64],
    sweep: &PpoBatchRange,
    k: Kernel,
) -> Result<PpoBatchOutput, PpoError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(PpoError::InvalidPeriod {
                fast: 0,
                slow: 0,
                data_len: 0,
            })
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    ppo_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn ppo_batch_slice(
    data: &[f64],
    sweep: &PpoBatchRange,
    kern: Kernel,
) -> Result<PpoBatchOutput, PpoError> {
    ppo_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ppo_batch_par_slice(
    data: &[f64],
    sweep: &PpoBatchRange,
    kern: Kernel,
) -> Result<PpoBatchOutput, PpoError> {
    ppo_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn ppo_batch_inner_into(
    data: &[f64],
    sweep: &PpoBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<PpoParams>, PpoError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(PpoError::InvalidPeriod {
            fast: 0,
            slow: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PpoError::AllValuesNaN)?;
    let max_slow = combos.iter().map(|c| c.slow_period.unwrap()).max().unwrap();
    if data.len() - first < max_slow {
        return Err(PpoError::NotEnoughValidData {
            needed: max_slow,
            valid: data.len() - first,
        });
    }

    let cols = data.len();
    // Treat the destination as uninitialized like ALMA
    let out_uninit: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let p = &combos[row];
        let out_row: &mut [f64] =
            std::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());
        match kern {
            Kernel::Scalar => ppo_row_scalar(
                data,
                first,
                p.fast_period.unwrap(),
                p.slow_period.unwrap(),
                p.ma_type.as_ref().unwrap(),
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ppo_row_avx2(
                data,
                first,
                p.fast_period.unwrap(),
                p.slow_period.unwrap(),
                p.ma_type.as_ref().unwrap(),
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ppo_row_avx512(
                data,
                first,
                p.fast_period.unwrap(),
                p.slow_period.unwrap(),
                p.ma_type.as_ref().unwrap(),
                out_row,
            ),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_uninit
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[inline(always)]
fn ppo_batch_inner(
    data: &[f64],
    sweep: &PpoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<PpoBatchOutput, PpoError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(PpoError::InvalidPeriod {
            fast: 0,
            slow: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PpoError::AllValuesNaN)?;
    let max_slow = combos.iter().map(|c| c.slow_period.unwrap()).max().unwrap();
    if data.len() - first < max_slow {
        return Err(PpoError::NotEnoughValidData {
            needed: max_slow,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Create uninitialized matrix
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each parameter combination
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.slow_period.unwrap() - 1)
        .collect();

    // Initialize matrix with NaN prefixes
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    // Convert to mutable slice for computation
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let values: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let p = &combos[row];
        match kern {
            Kernel::Scalar => ppo_row_scalar(
                data,
                first,
                p.fast_period.unwrap(),
                p.slow_period.unwrap(),
                p.ma_type.as_ref().unwrap(),
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ppo_row_avx2(
                data,
                first,
                p.fast_period.unwrap(),
                p.slow_period.unwrap(),
                p.ma_type.as_ref().unwrap(),
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ppo_row_avx512(
                data,
                first,
                p.fast_period.unwrap(),
                p.slow_period.unwrap(),
                p.ma_type.as_ref().unwrap(),
                out_row,
            ),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in values.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // Convert buffer back to Vec
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(PpoBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn ppo_row_scalar(
    data: &[f64],
    first: usize,
    fast: usize,
    slow: usize,
    ma_type: &str,
    out: &mut [f64],
) {
    // Check for classic kernel optimization
    if ma_type == "ema" {
        ppo_row_scalar_classic_ema(data, first, fast, slow, out);
    } else if ma_type == "sma" {
        ppo_row_scalar_classic_sma(data, first, fast, slow, out);
    } else {
        ppo_scalar(data, fast, slow, ma_type, first, out);
    }
}

// Classic row kernel with inline EMA for batch processing
#[inline(always)]
pub unsafe fn ppo_row_scalar_classic_ema(
    data: &[f64],
    first: usize,
    fast: usize,
    slow: usize,
    out: &mut [f64],
) {
    ppo_scalar_classic_ema(data, fast, slow, first, out);
}

// Classic row kernel with inline SMA for batch processing
#[inline(always)]
pub unsafe fn ppo_row_scalar_classic_sma(
    data: &[f64],
    first: usize,
    fast: usize,
    slow: usize,
    out: &mut [f64],
) {
    ppo_scalar_classic_sma(data, fast, slow, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ppo_row_avx2(
    data: &[f64],
    first: usize,
    fast: usize,
    slow: usize,
    ma_type: &str,
    out: &mut [f64],
) {
    ppo_scalar(data, fast, slow, ma_type, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ppo_row_avx512(
    data: &[f64],
    first: usize,
    fast: usize,
    slow: usize,
    ma_type: &str,
    out: &mut [f64],
) {
    if slow <= 32 {
        ppo_row_avx512_short(data, first, fast, slow, ma_type, out)
    } else {
        ppo_row_avx512_long(data, first, fast, slow, ma_type, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ppo_row_avx512_short(
    data: &[f64],
    first: usize,
    fast: usize,
    slow: usize,
    ma_type: &str,
    out: &mut [f64],
) {
    ppo_scalar(data, fast, slow, ma_type, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ppo_row_avx512_long(
    data: &[f64],
    first: usize,
    fast: usize,
    slow: usize,
    ma_type: &str,
    out: &mut [f64],
) {
    ppo_scalar(data, fast, slow, ma_type, first, out)
}

pub struct PpoStream {
    fast_period: usize,
    slow_period: usize,
    ma_type: String,
    data: Vec<f64>,
}

impl PpoStream {
    pub fn try_new(params: PpoParams) -> Result<Self, PpoError> {
        Ok(Self {
            fast_period: params.fast_period.unwrap_or(12),
            slow_period: params.slow_period.unwrap_or(26),
            ma_type: params.ma_type.clone().unwrap_or_else(|| "sma".to_string()),
            data: Vec::new(),
        })
    }

    /// Update the stream with a new value and return the latest PPO if available.
    ///
    /// Returns `None` until enough data has been supplied for the slow moving
    /// average. Once both averages are ready, it returns `Some(ppo)` where
    /// `ppo` is `100 * (fast_ma - slow_ma) / slow_ma`.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.data.push(value);
        if self.data.len() < self.slow_period {
            return None;
        }
        let fast_ma = ma(&self.ma_type, MaData::Slice(&self.data), self.fast_period).ok()?;
        let slow_ma = ma(&self.ma_type, MaData::Slice(&self.data), self.slow_period).ok()?;
        let ff = *fast_ma.last()?;
        let sf = *slow_ma.last()?;
        if ff.is_nan() || sf.is_nan() || sf == 0.0 {
            Some(f64::NAN)
        } else {
            Some(100.0 * (ff - sf) / sf)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_ppo_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = PpoParams {
            fast_period: None,
            slow_period: None,
            ma_type: None,
        };
        let input = PpoInput::from_candles(&candles, "close", default_params);
        let output = ppo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ppo_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PpoInput::from_candles(&candles, "close", PpoParams::default());
        let result = ppo_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        let expected_last_five = [
            -0.8532313608928664,
            -0.8537562894550523,
            -0.6821291938174874,
            -0.5620008722078592,
            -0.4101724140910927,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-7,
                "[{}] PPO {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_ppo_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PpoInput::with_default_candles(&candles);
        match input.data {
            PpoData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected PpoData::Candles"),
        }
        let output = ppo_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ppo_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = PpoParams {
            fast_period: Some(0),
            slow_period: None,
            ma_type: None,
        };
        let input = PpoInput::from_slice(&input_data, params);
        let res = ppo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PPO should fail with zero fast period",
            test_name
        );
        Ok(())
    }

    fn check_ppo_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = PpoParams {
            fast_period: Some(12),
            slow_period: Some(26),
            ma_type: None,
        };
        let input = PpoInput::from_slice(&data_small, params);
        let res = ppo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PPO should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_ppo_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = PpoParams {
            fast_period: Some(12),
            slow_period: Some(26),
            ma_type: None,
        };
        let input = PpoInput::from_slice(&single_point, params);
        let res = ppo_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PPO should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_ppo_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PpoInput::from_candles(
            &candles,
            "close",
            PpoParams {
                fast_period: Some(12),
                slow_period: Some(26),
                ma_type: Some("sma".to_string()),
            },
        );
        let res = ppo_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 30 {
            for (i, &val) in res.values[30..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    30 + i
                );
            }
        }
        Ok(())
    }

    fn check_ppo_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let fast = 12;
        let slow = 26;
        let ma_type = "sma".to_string();
        let input = PpoInput::from_candles(
            &candles,
            "close",
            PpoParams {
                fast_period: Some(fast),
                slow_period: Some(slow),
                ma_type: Some(ma_type.clone()),
            },
        );
        let batch_output = ppo_with_kernel(&input, kernel)?.values;
        let mut stream = PpoStream::try_new(PpoParams {
            fast_period: Some(fast),
            slow_period: Some(slow),
            ma_type: Some(ma_type),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(ppo_val) => stream_values.push(ppo_val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] PPO streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_ppo_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            // Default parameters
            PpoParams::default(),
            // Minimum viable periods
            PpoParams {
                fast_period: Some(2),
                slow_period: Some(3),
                ma_type: Some("sma".to_string()),
            },
            // Small periods
            PpoParams {
                fast_period: Some(5),
                slow_period: Some(10),
                ma_type: Some("sma".to_string()),
            },
            // Different MA types with default periods
            PpoParams {
                fast_period: Some(12),
                slow_period: Some(26),
                ma_type: Some("ema".to_string()),
            },
            PpoParams {
                fast_period: Some(12),
                slow_period: Some(26),
                ma_type: Some("wma".to_string()),
            },
            // Medium periods
            PpoParams {
                fast_period: Some(20),
                slow_period: Some(40),
                ma_type: Some("sma".to_string()),
            },
            // Large periods
            PpoParams {
                fast_period: Some(50),
                slow_period: Some(100),
                ma_type: Some("sma".to_string()),
            },
            // Edge case: fast and slow close together
            PpoParams {
                fast_period: Some(10),
                slow_period: Some(11),
                ma_type: Some("sma".to_string()),
            },
            // Different ratios
            PpoParams {
                fast_period: Some(3),
                slow_period: Some(21),
                ma_type: Some("ema".to_string()),
            },
            // More edge cases
            PpoParams {
                fast_period: Some(7),
                slow_period: Some(14),
                ma_type: Some("wma".to_string()),
            },
            PpoParams {
                fast_period: Some(9),
                slow_period: Some(21),
                ma_type: Some("sma".to_string()),
            },
            // Very large period
            PpoParams {
                fast_period: Some(100),
                slow_period: Some(200),
                ma_type: Some("ema".to_string()),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = PpoInput::from_candles(&candles, "close", params.clone());
            let output = ppo_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: fast_period={}, slow_period={}, ma_type={:?} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.fast_period.unwrap_or(12),
                        params.slow_period.unwrap_or(26),
                        params.ma_type.as_ref().unwrap_or(&"sma".to_string()),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: fast_period={}, slow_period={}, ma_type={:?} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.fast_period.unwrap_or(12),
                        params.slow_period.unwrap_or(26),
                        params.ma_type.as_ref().unwrap_or(&"sma".to_string()),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: fast_period={}, slow_period={}, ma_type={:?} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.fast_period.unwrap_or(12),
                        params.slow_period.unwrap_or(26),
                        params.ma_type.as_ref().unwrap_or(&"sma".to_string()),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ppo_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_ppo_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::indicators::moving_averages::ma::{ma, MaData};
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy to generate test parameters
        let strat = (2usize..=64).prop_flat_map(|slow_period| {
            (
                // Data vector with length from slow_period to 400
                prop::collection::vec(
                    (10f64..100000f64)
                        .prop_filter("positive finite", |x| x.is_finite() && *x > 0.0),
                    slow_period..400,
                ),
                // Fast period must be less than or equal to slow period
                2usize..=slow_period,
                Just(slow_period),
                // MA type - focus on SMA for mathematical verification
                Just("sma"),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, fast_period, slow_period, ma_type)| {
                let params = PpoParams {
                    fast_period: Some(fast_period),
                    slow_period: Some(slow_period),
                    ma_type: Some(ma_type.to_string()),
                };
                let input = PpoInput::from_slice(&data, params);

                // Get output from kernel under test
                let PpoOutput { values: out } = ppo_with_kernel(&input, kernel).unwrap();
                // Get reference output from scalar kernel
                let PpoOutput { values: ref_out } =
                    ppo_with_kernel(&input, Kernel::Scalar).unwrap();

                // Calculate MAs independently for verification
                let fast_ma = ma(&ma_type, MaData::Slice(&data), fast_period).unwrap();
                let slow_ma = ma(&ma_type, MaData::Slice(&data), slow_period).unwrap();

                // Property 1: Check warmup period consistency
                for i in 0..(slow_period - 1).min(data.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                // Check properties for each valid output value
                for i in (slow_period - 1)..data.len() {
                    let y = out[i];
                    let r = ref_out[i];
                    let fast_val = fast_ma[i];
                    let slow_val = slow_ma[i];

                    // Property 2: Verify mathematical formula
                    // PPO = 100 * (fast_ma - slow_ma) / slow_ma
                    if !fast_val.is_nan() && !slow_val.is_nan() && slow_val != 0.0 {
                        let expected_ppo = 100.0 * (fast_val - slow_val) / slow_val;

                        if y.is_finite() && expected_ppo.is_finite() {
                            prop_assert!(
								(y - expected_ppo).abs() < 1e-9,
								"PPO formula mismatch at index {}: got {}, expected {} (fast={}, slow={})",
								i, y, expected_ppo, fast_val, slow_val
							);
                        }
                    }

                    // Property 3: Consistency between kernels
                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch idx {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);
                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "kernel mismatch idx {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );

                    // Property 4: Sign correctness
                    if y.is_finite()
                        && fast_val.is_finite()
                        && slow_val.is_finite()
                        && slow_val > 0.0
                    {
                        if fast_val > slow_val {
                            prop_assert!(
                                y > 0.0,
                                "PPO should be positive when fast > slow at index {}",
                                i
                            );
                        } else if fast_val < slow_val {
                            prop_assert!(
                                y < 0.0,
                                "PPO should be negative when fast < slow at index {}",
                                i
                            );
                        } else {
                            prop_assert!(
                                y.abs() < 1e-9,
                                "PPO should be ~0 when fast == slow at index {}",
                                i
                            );
                        }
                    }

                    // Property 5: Special case - equal periods
                    if fast_period == slow_period && y.is_finite() {
                        prop_assert!(
                            y.abs() < 1e-9,
                            "PPO should be ~0 when fast_period == slow_period at index {}: got {}",
                            i,
                            y
                        );
                    }

                    // Property 6: Constant data check
                    if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12) && y.is_finite() {
                        prop_assert!(
                            y.abs() < 1e-6,
                            "PPO should be ~0 for constant data at index {}: got {}",
                            i,
                            y
                        );
                    }

                    // Property 7: Reasonable bounds for price data
                    // For realistic price data (10-100000 range), PPO rarely exceeds ±200%
                    // unless there's extreme volatility in a small window
                    if y.is_finite() {
                        // Calculate data volatility in the window
                        let window_start = i.saturating_sub(slow_period - 1);
                        let window = &data[window_start..=i];
                        let min_val = window.iter().cloned().fold(f64::INFINITY, f64::min);
                        let max_val = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let volatility_ratio = if min_val > 0.0 {
                            max_val / min_val
                        } else {
                            1.0
                        };

                        // PPO should be bounded by the volatility of the underlying data
                        // High volatility (10x price change) could produce PPO around ±900%
                        let max_expected_ppo = 100.0 * (volatility_ratio - 1.0);

                        prop_assert!(
							y.abs() <= max_expected_ppo * 1.5, // Allow some margin for MA lag
							"PPO exceeds expected bounds at index {}: got {}%, max expected ~{}% (volatility ratio {})",
							i, y, max_expected_ppo, volatility_ratio
						);
                    }

                    // Property 8: Handle near-zero slow_ma correctly
                    if slow_val.abs() < 1e-10 && slow_val != 0.0 {
                        // When slow_ma is very close to zero, PPO should be NaN or very large
                        prop_assert!(
							y.is_nan() || y.abs() > 1000.0,
							"PPO should be NaN or very large when slow_ma ~0 at index {}: slow_ma={}, ppo={}",
							i, slow_val, y
						);
                    }
                }

                // Property 9: Monotonic data behavior
                let is_monotonic_increasing = data.windows(2).all(|w| w[1] >= w[0]);
                let is_monotonic_decreasing = data.windows(2).all(|w| w[1] <= w[0]);

                if (is_monotonic_increasing || is_monotonic_decreasing)
                    && data.len() > slow_period * 2
                {
                    // After sufficient data, PPO should stabilize
                    let last_values = &out[out.len() - slow_period / 2..];
                    let valid_last: Vec<f64> = last_values
                        .iter()
                        .filter(|x| x.is_finite())
                        .cloned()
                        .collect();

                    if valid_last.len() > 2 {
                        if is_monotonic_increasing && fast_period < slow_period {
                            // For increasing data with fast < slow, PPO should be positive
                            let avg = valid_last.iter().sum::<f64>() / valid_last.len() as f64;
                            prop_assert!(
                                avg > -1e-6,
                                "PPO should be positive for monotonic increasing data: avg={}",
                                avg
                            );
                        } else if is_monotonic_decreasing && fast_period < slow_period {
                            // For decreasing data with fast < slow, PPO should be negative
                            let avg = valid_last.iter().sum::<f64>() / valid_last.len() as f64;
                            prop_assert!(
                                avg < 1e-6,
                                "PPO should be negative for monotonic decreasing data: avg={}",
                                avg
                            );
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_ppo_tests {
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

    generate_all_ppo_tests!(
        check_ppo_partial_params,
        check_ppo_accuracy,
        check_ppo_default_candles,
        check_ppo_zero_period,
        check_ppo_period_exceeds_length,
        check_ppo_very_small_dataset,
        check_ppo_nan_handling,
        check_ppo_streaming,
        check_ppo_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_ppo_tests!(check_ppo_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = PpoBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = PpoParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -0.8532313608928664,
            -0.8537562894550523,
            -0.6821291938174874,
            -0.5620008722078592,
            -0.4101724140910927,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-7,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
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

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (fast_start, fast_end, fast_step, slow_start, slow_end, slow_step, ma_type)
            (2, 10, 2, 12, 30, 3, "sma"),     // Small periods
            (5, 25, 5, 26, 50, 5, "sma"),     // Medium periods
            (30, 60, 15, 65, 100, 10, "sma"), // Large periods
            (2, 5, 1, 6, 10, 1, "ema"),       // Dense small range with EMA
            (10, 20, 2, 21, 40, 4, "wma"),    // Medium range with WMA
            (3, 9, 3, 12, 21, 3, "sma"),      // Classic ratios
            (7, 14, 7, 21, 28, 7, "ema"),     // Weekly/monthly periods
        ];

        for (cfg_idx, &(f_start, f_end, f_step, s_start, s_end, s_step, ma_type)) in
            test_configs.iter().enumerate()
        {
            let output = PpoBatchBuilder::new()
                .kernel(kernel)
                .fast_period_range(f_start, f_end, f_step)
                .slow_period_range(s_start, s_end, s_step)
                .ma_type(ma_type)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                // Check all three poison patterns with detailed context
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: fast_period={}, slow_period={}, ma_type={:?}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_period.unwrap_or(12),
                        combo.slow_period.unwrap_or(26),
                        combo.ma_type.as_ref().unwrap_or(&"sma".to_string())
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: fast_period={}, slow_period={}, ma_type={:?}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_period.unwrap_or(12),
                        combo.slow_period.unwrap_or(26),
                        combo.ma_type.as_ref().unwrap_or(&"sma".to_string())
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: fast_period={}, slow_period={}, ma_type={:?}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fast_period.unwrap_or(12),
                        combo.slow_period.unwrap_or(26),
                        combo.ma_type.as_ref().unwrap_or(&"sma".to_string())
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

// ┌────────────────────────────────────────────────────────────────────────────┐
// │                          Python Bindings                                   │
// └────────────────────────────────────────────────────────────────────────────┘

#[cfg(feature = "python")]
#[pyfunction(name = "ppo")]
#[pyo3(signature = (data, fast_period=None, slow_period=None, ma_type=None, kernel=None))]
pub fn ppo_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    ma_type: Option<&str>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = PpoParams {
        fast_period,
        slow_period,
        ma_type: ma_type.map(|s| s.to_string()),
    };
    let input = PpoInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| ppo_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "PpoStream")]
pub struct PpoStreamPy {
    stream: PpoStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PpoStreamPy {
    #[new]
    fn new(
        fast_period: Option<usize>,
        slow_period: Option<usize>,
        ma_type: Option<&str>,
    ) -> PyResult<Self> {
        let params = PpoParams {
            fast_period,
            slow_period,
            ma_type: ma_type.map(|s| s.to_string()),
        };
        let stream =
            PpoStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PpoStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ppo_batch")]
#[pyo3(signature = (data, fast_period_range, slow_period_range, ma_type=None, kernel=None))]
pub fn ppo_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    fast_period_range: (usize, usize, usize),
    slow_period_range: (usize, usize, usize),
    ma_type: Option<&str>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;

    let sweep = PpoBatchRange {
        fast_period: fast_period_range,
        slow_period: slow_period_range,
        ma_type: ma_type.unwrap_or("sma").to_string(),
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Initialize NaN prefixes directly on the NumPy buffer using helpers
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| {
            let first = slice_in.iter().position(|x| !x.is_nan()).unwrap_or(0);
            first + c.slow_period.unwrap() - 1
        })
        .collect();

    unsafe {
        let mu: &mut [MaybeUninit<f64>] = std::slice::from_raw_parts_mut(
            slice_out.as_mut_ptr() as *mut MaybeUninit<f64>,
            slice_out.len(),
        );
        init_matrix_prefixes(mu, cols, &warm);
    }

    let kern = validate_kernel(kernel, true)?;
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
                k if !k.is_batch() => k,
                _ => Kernel::Scalar,
            };
            ppo_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "fast_periods",
        combos
            .iter()
            .map(|p| p.fast_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slow_periods",
        combos
            .iter()
            .map(|p| p.slow_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ma_types",
        combos
            .iter()
            .map(|p| p.ma_type.as_ref().unwrap().clone())
            .collect::<Vec<_>>(),
    )?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_ppo_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ppo_py, m)?)?;
    m.add_function(wrap_pyfunction!(ppo_batch_py, m)?)?;
    m.add_class::<PpoStreamPy>()?;
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────────
// WASM Bindings
// ────────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ppo_js(
    data: &[f64],
    fast_period: usize,
    slow_period: usize,
    ma_type: &str,
) -> Result<Vec<f64>, JsValue> {
    let params = PpoParams {
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
        ma_type: Some(ma_type.to_string()),
    };
    let input = PpoInput::from_slice(data, params);
    let mut out = vec![0.0; data.len()];
    ppo_into_slice(&mut out, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ppo_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ppo_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ppo_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    fast_period: usize,
    slow_period: usize,
    ma_type: &str,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to ppo_into"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = PpoParams {
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
            ma_type: Some(ma_type.to_string()),
        };
        let input = PpoInput::from_slice(data, params);
        if in_ptr == out_ptr {
            let mut tmp = vec![0.0; len];
            ppo_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ppo_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PpoBatchConfig {
    pub fast_period_range: (usize, usize, usize),
    pub slow_period_range: (usize, usize, usize),
    pub ma_type: String,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PpoBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PpoParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "ppo_batch")]
pub fn ppo_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: PpoBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = PpoBatchRange {
        fast_period: cfg.fast_period_range,
        slow_period: cfg.slow_period_range,
        ma_type: cfg.ma_type,
    };
    let out = ppo_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = PpoBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ppo_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    fast_start: usize,
    fast_end: usize,
    fast_step: usize,
    slow_start: usize,
    slow_end: usize,
    slow_step: usize,
    ma_type: &str,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to ppo_batch_into"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = PpoBatchRange {
            fast_period: (fast_start, fast_end, fast_step),
            slow_period: (slow_start, slow_end, slow_step),
            ma_type: ma_type.to_string(),
        };
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
        ppo_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}
