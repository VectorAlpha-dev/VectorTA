//! # Ehlers Simple Decycler
//!
//! A noise-reduction filter based on John Ehlers's High-Pass Filter concept.
//! Removes high-frequency components from the original data, leaving a "decycled" output
//! that helps highlight underlying trends.
//!
//! ## Parameters
//! - **hp_period**: Window size used for the embedded high-pass filter (minimum of 2). Defaults to 125.
//! - **k**: Frequency coefficient for the high-pass filter. Defaults to 0.707.
//!
//! ## Errors
//! - **EmptyData**: decycler: Input data slice is empty.
//! - **InvalidPeriod**: decycler: `hp_period` is zero, less than 2, or exceeds the data length.
//! - **NotEnoughValidData**: decycler: Fewer than `hp_period` valid (non-`NaN`) data points remain after the first valid index.
//! - **AllValuesNaN**: decycler: All input data values are `NaN`.
//! - **InvalidK**: decycler: `k` is non-positive or NaN.
//!
//! ## Returns
//! - **`Ok(DecyclerOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(DecyclerError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<[f64]> for DecyclerInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DecyclerData::Slice(slice) => slice,
            DecyclerData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DecyclerData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DecyclerOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DecyclerParams {
    pub hp_period: Option<usize>,
    pub k: Option<f64>,
}

impl Default for DecyclerParams {
    fn default() -> Self {
        Self {
            hp_period: Some(125),
            k: Some(0.707),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecyclerInput<'a> {
    pub data: DecyclerData<'a>,
    pub params: DecyclerParams,
}

impl<'a> DecyclerInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: DecyclerParams) -> Self {
        Self {
            data: DecyclerData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: DecyclerParams) -> Self {
        Self {
            data: DecyclerData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", DecyclerParams::default())
    }
    #[inline]
    pub fn get_hp_period(&self) -> usize {
        self.params.hp_period.unwrap_or(125)
    }
    #[inline]
    pub fn get_k(&self) -> f64 {
        self.params.k.unwrap_or(0.707)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DecyclerBuilder {
    hp_period: Option<usize>,
    k: Option<f64>,
    kernel: Kernel,
}

impl Default for DecyclerBuilder {
    fn default() -> Self {
        Self {
            hp_period: None,
            k: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DecyclerBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn hp_period(mut self, n: usize) -> Self {
        self.hp_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn k(mut self, x: f64) -> Self {
        self.k = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<DecyclerOutput, DecyclerError> {
        let p = DecyclerParams {
            hp_period: self.hp_period,
            k: self.k,
        };
        let i = DecyclerInput::from_candles(c, "close", p);
        decycler_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<DecyclerOutput, DecyclerError> {
        let p = DecyclerParams {
            hp_period: self.hp_period,
            k: self.k,
        };
        let i = DecyclerInput::from_slice(d, p);
        decycler_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<DecyclerStream, DecyclerError> {
        let p = DecyclerParams {
            hp_period: self.hp_period,
            k: self.k,
        };
        DecyclerStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DecyclerError {
    #[error("decycler: Empty data provided for Decycler.")]
    EmptyData,
    #[error("decycler: Invalid period: hp_period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("decycler: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("decycler: All values are NaN.")]
    AllValuesNaN,
    #[error("decycler: Invalid k: k = {k}")]
    InvalidK { k: f64 },
}

#[inline]
pub fn decycler(input: &DecyclerInput) -> Result<DecyclerOutput, DecyclerError> {
    decycler_with_kernel(input, Kernel::Auto)
}

pub fn decycler_with_kernel(
    input: &DecyclerInput,
    kernel: Kernel,
) -> Result<DecyclerOutput, DecyclerError> {
    let data: &[f64] = match &input.data {
        DecyclerData::Candles { candles, source } => source_type(candles, source),
        DecyclerData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(DecyclerError::EmptyData);
    }
    let hp_period = input.get_hp_period();
    let k = input.get_k();
    if hp_period < 2 || hp_period > data.len() {
        return Err(DecyclerError::InvalidPeriod {
            period: hp_period,
            data_len: data.len(),
        });
    }
    if !(k.is_finite()) || k <= 0.0 {
        return Err(DecyclerError::InvalidK { k });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(DecyclerError::AllValuesNaN)?;
    if data.len() - first < hp_period {
        return Err(DecyclerError::NotEnoughValidData {
            needed: hp_period,
            valid: data.len() - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => decycler_scalar(data, hp_period, k, first),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => decycler_avx2(data, hp_period, k, first),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => decycler_avx512(data, hp_period, k, first),
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn decycler_scalar(
    data: &[f64],
    hp_period: usize,
    k: f64,
    first: usize,
) -> Result<DecyclerOutput, DecyclerError> {
    use std::f64::consts::PI;
    let mut out = vec![f64::NAN; data.len()];
    let mut hp = vec![0.0; data.len()];
    let angle = 2.0 * PI * k / (hp_period as f64);
    let sin_val = angle.sin();
    let cos_val = angle.cos();
    let alpha = 1.0 + ((sin_val - 1.0) / cos_val);
    let one_minus_alpha_half = 1.0 - alpha / 2.0;
    let c = one_minus_alpha_half * one_minus_alpha_half;
    let one_minus_alpha = 1.0 - alpha;
    let one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;

    if data.len() > first {
        hp[first] = data[first];
        out[first] = data[first] - hp[first];
    }
    if data.len() > (first + 1) {
        hp[first + 1] = data[first + 1];
        out[first + 1] = data[first + 1] - hp[first + 1];
    }
    for i in (first + 2)..data.len() {
        let current = data[i];
        let prev1 = data[i - 1];
        let prev2 = data[i - 2];
        let hp_prev1 = hp[i - 1];
        let hp_prev2 = hp[i - 2];
        let val = c * current - 2.0 * c * prev1 + c * prev2
            + 2.0 * one_minus_alpha * hp_prev1
            - one_minus_alpha_sq * hp_prev2;
        hp[i] = val;
        out[i] = current - val;
    }
    Ok(DecyclerOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn decycler_avx512(
    data: &[f64],
    hp_period: usize,
    k: f64,
    first: usize,
) -> Result<DecyclerOutput, DecyclerError> {
    // STUB: AVX512 not implemented, fallback to scalar
    decycler_scalar(data, hp_period, k, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn decycler_avx2(
    data: &[f64],
    hp_period: usize,
    k: f64,
    first: usize,
) -> Result<DecyclerOutput, DecyclerError> {
    // STUB: AVX2 not implemented, fallback to scalar
    decycler_scalar(data, hp_period, k, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn decycler_avx512_short(
    data: &[f64],
    hp_period: usize,
    k: f64,
    first: usize,
) -> Result<DecyclerOutput, DecyclerError> {
    decycler_scalar(data, hp_period, k, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn decycler_avx512_long(
    data: &[f64],
    hp_period: usize,
    k: f64,
    first: usize,
) -> Result<DecyclerOutput, DecyclerError> {
    decycler_scalar(data, hp_period, k, first)
}

#[inline]
pub fn decycler_batch_with_kernel(
    data: &[f64],
    sweep: &DecyclerBatchRange,
    k: Kernel,
) -> Result<DecyclerBatchOutput, DecyclerError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(DecyclerError::InvalidPeriod {
                period: 0,
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
    decycler_batch_par_slice(data, sweep, simd)
}

#[derive(Debug, Clone)]
pub struct DecyclerStream {
    hp_period: usize,
    k: f64,
    buffer: Vec<f64>,
    hp: Vec<f64>,
    idx: usize,
    filled: bool,
    angle: f64,
    c: f64,
    one_minus_alpha: f64,
    one_minus_alpha_sq: f64,
}

impl DecyclerStream {
    pub fn try_new(params: DecyclerParams) -> Result<Self, DecyclerError> {
        let hp_period = params.hp_period.unwrap_or(125);
        let k = params.k.unwrap_or(0.707);
        if hp_period < 2 {
            return Err(DecyclerError::InvalidPeriod {
                period: hp_period,
                data_len: 0,
            });
        }
        if !(k.is_finite()) || k <= 0.0 {
            return Err(DecyclerError::InvalidK { k });
        }
        use std::f64::consts::PI;
        let angle = 2.0 * PI * k / (hp_period as f64);
        let sin_val = angle.sin();
        let cos_val = angle.cos();
        let alpha = 1.0 + ((sin_val - 1.0) / cos_val);
        let one_minus_alpha_half = 1.0 - alpha / 2.0;
        let c = one_minus_alpha_half * one_minus_alpha_half;
        let one_minus_alpha = 1.0 - alpha;
        let one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;
        Ok(Self {
            hp_period,
            k,
            buffer: vec![f64::NAN; hp_period],
            hp: vec![0.0; hp_period],
            idx: 0,
            filled: false,
            angle,
            c,
            one_minus_alpha,
            one_minus_alpha_sq,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let idx0 = self.idx % self.hp_period;
        self.buffer[idx0] = value;
        if self.idx == 0 {
            self.hp[idx0] = value;
        } else if self.idx == 1 {
            self.hp[idx0] = value;
        
            } else {
            let prev1 = self.buffer[(self.idx + self.hp_period - 1) % self.hp_period];
            let prev2 = self.buffer[(self.idx + self.hp_period - 2) % self.hp_period];
            let hp_prev1 = self.hp[(self.idx + self.hp_period - 1) % self.hp_period];
            let hp_prev2 = self.hp[(self.idx + self.hp_period - 2) % self.hp_period];
            let val = self.c * value
                - 2.0 * self.c * prev1
                + self.c * prev2
                + 2.0 * self.one_minus_alpha * hp_prev1
                - self.one_minus_alpha_sq * hp_prev2;
            self.hp[idx0] = val;
        }
        let out = if self.idx < 2 {
            value - self.hp[idx0]
        
            } else {
            value - self.hp[idx0]
        };
        self.idx += 1;
        if self.idx >= self.hp_period {
            self.filled = true;
        }
        Some(out)
    }
}

#[derive(Clone, Debug)]
pub struct DecyclerBatchRange {
    pub hp_period: (usize, usize, usize),
    pub k: (f64, f64, f64),
}

impl Default for DecyclerBatchRange {
    fn default() -> Self {
        Self {
            hp_period: (125, 125, 0),
            k: (0.707, 0.707, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DecyclerBatchBuilder {
    range: DecyclerBatchRange,
    kernel: Kernel,
}

impl DecyclerBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn hp_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.hp_period = (start, end, step);
        self
    }
    #[inline]
    pub fn hp_period_static(mut self, p: usize) -> Self {
        self.range.hp_period = (p, p, 0);
        self
    }
    #[inline]
    pub fn k_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.k = (start, end, step);
        self
    }
    #[inline]
    pub fn k_static(mut self, v: f64) -> Self {
        self.range.k = (v, v, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<DecyclerBatchOutput, DecyclerError> {
        decycler_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<DecyclerBatchOutput, DecyclerError> {
        DecyclerBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<DecyclerBatchOutput, DecyclerError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<DecyclerBatchOutput, DecyclerError> {
        DecyclerBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct DecyclerBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DecyclerParams>,
    pub rows: usize,
    pub cols: usize,
}
impl DecyclerBatchOutput {
    pub fn row_for_params(&self, p: &DecyclerParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.hp_period.unwrap_or(125) == p.hp_period.unwrap_or(125)
                && (c.k.unwrap_or(0.707) - p.k.unwrap_or(0.707)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &DecyclerParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &DecyclerBatchRange) -> Vec<DecyclerParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    let hp_periods = axis_usize(r.hp_period);
    let ks = axis_f64(r.k);
    let mut out = Vec::with_capacity(hp_periods.len() * ks.len());
    for &p in &hp_periods {
        for &k in &ks {
            out.push(DecyclerParams {
                hp_period: Some(p),
                k: Some(k),
            });
        }
    }
    out
}

#[inline(always)]
pub fn decycler_batch_slice(
    data: &[f64],
    sweep: &DecyclerBatchRange,
    kern: Kernel,
) -> Result<DecyclerBatchOutput, DecyclerError> {
    decycler_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn decycler_batch_par_slice(
    data: &[f64],
    sweep: &DecyclerBatchRange,
    kern: Kernel,
) -> Result<DecyclerBatchOutput, DecyclerError> {
    decycler_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn decycler_batch_inner(
    data: &[f64],
    sweep: &DecyclerBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DecyclerBatchOutput, DecyclerError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DecyclerError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(DecyclerError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.hp_period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(DecyclerError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let hp_period = combos[row].hp_period.unwrap();
        let k = combos[row].k.unwrap();
        match kern {
            Kernel::Scalar => decycler_row_scalar(data, first, hp_period, k, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => decycler_row_avx2(data, first, hp_period, k, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => decycler_row_avx512(data, first, hp_period, k, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {


        #[cfg(not(target_arch = "wasm32"))] {


        values


                    .par_chunks_mut(cols)


                    .enumerate()


                    .for_each(|(row, slice)| do_row(row, slice));


        }


        #[cfg(target_arch = "wasm32")] {


        for (row, slice) in values.chunks_mut(cols).enumerate() {


                    do_row(row, slice);


        }


    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(DecyclerBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn decycler_row_scalar(
    data: &[f64],
    first: usize,
    hp_period: usize,
    k: f64,
    out: &mut [f64],
) {
    use std::f64::consts::PI;
    let mut hp = vec![0.0; data.len()];
    let angle = 2.0 * PI * k / (hp_period as f64);
    let sin_val = angle.sin();
    let cos_val = angle.cos();
    let alpha = 1.0 + ((sin_val - 1.0) / cos_val);
    let one_minus_alpha_half = 1.0 - alpha / 2.0;
    let c = one_minus_alpha_half * one_minus_alpha_half;
    let one_minus_alpha = 1.0 - alpha;
    let one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;

    if data.len() > first {
        hp[first] = data[first];
        out[first] = data[first] - hp[first];
    }
    if data.len() > (first + 1) {
        hp[first + 1] = data[first + 1];
        out[first + 1] = data[first + 1] - hp[first + 1];
    }
    for i in (first + 2)..data.len() {
        let current = data[i];
        let prev1 = data[i - 1];
        let prev2 = data[i - 2];
        let hp_prev1 = hp[i - 1];
        let hp_prev2 = hp[i - 2];
        let val = c * current - 2.0 * c * prev1 + c * prev2
            + 2.0 * one_minus_alpha * hp_prev1
            - one_minus_alpha_sq * hp_prev2;
        hp[i] = val;
        out[i] = current - val;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn decycler_row_avx2(
    data: &[f64],
    first: usize,
    hp_period: usize,
    k: f64,
    out: &mut [f64],
) {
    decycler_row_scalar(data, first, hp_period, k, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn decycler_row_avx512(
    data: &[f64],
    first: usize,
    hp_period: usize,
    k: f64,
    out: &mut [f64],
) {
    if hp_period <= 32 {
        decycler_row_avx512_short(data, first, hp_period, k, out)
    
        } else {
        decycler_row_avx512_long(data, first, hp_period, k, out)
    }
    _mm_sfence();
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn decycler_row_avx512_short(
    data: &[f64],
    first: usize,
    hp_period: usize,
    k: f64,
    out: &mut [f64],
) {
    decycler_row_scalar(data, first, hp_period, k, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn decycler_row_avx512_long(
    data: &[f64],
    first: usize,
    hp_period: usize,
    k: f64,
    out: &mut [f64],
) {
    decycler_row_scalar(data, first, hp_period, k, out)
}

#[inline(always)]
pub fn expand_grid_decycler(r: &DecyclerBatchRange) -> Vec<DecyclerParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_decycler_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = DecyclerParams { hp_period: None, k: None };
        let input_default = DecyclerInput::from_candles(&candles, "close", default_params);
        let output_default = decycler_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_hp_50 = DecyclerParams { hp_period: Some(50), k: None };
        let input_hp_50 = DecyclerInput::from_candles(&candles, "hl2", params_hp_50);
        let output_hp_50 = decycler_with_kernel(&input_hp_50, kernel)?;
        assert_eq!(output_hp_50.values.len(), candles.close.len());

        let params_custom = DecyclerParams { hp_period: Some(30), k: None };
        let input_custom = DecyclerInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = decycler_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());
        Ok(())
    }

    fn check_decycler_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let params = DecyclerParams { hp_period: Some(125), k: None };
        let input = DecyclerInput::from_candles(&candles, "close", params);
        let decycler_result = decycler_with_kernel(&input, kernel)?;
        assert_eq!(decycler_result.values.len(), close_prices.len());
        let test_values = [
            60289.96384058519,
            60204.010366691065,
            60114.255563805666,
            60028.535266555904,
            59934.26876964316,
        ];
        assert!(decycler_result.values.len() >= test_values.len());
        let start_index = decycler_result.values.len() - test_values.len();
        let result_last_values = &decycler_result.values[start_index..];
        for (i, &value) in result_last_values.iter().enumerate() {
            let expected_value = test_values[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "Decycler mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        Ok(())
    }

    fn check_decycler_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = DecyclerParams { hp_period: Some(0), k: None };
        let input = DecyclerInput::from_slice(&input_data, params);
        let result = decycler_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_decycler_period_exceed_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = DecyclerParams { hp_period: Some(10), k: None };
        let input = DecyclerInput::from_slice(&input_data, params);
        let result = decycler_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_decycler_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [42.0];
        let params = DecyclerParams { hp_period: Some(2), k: None };
        let input = DecyclerInput::from_slice(&input_data, params);
        let result = decycler_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_decycler_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = DecyclerParams { hp_period: Some(30), k: None };
        let first_input = DecyclerInput::from_candles(&candles, "close", first_params);
        let first_result = decycler_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = DecyclerParams { hp_period: Some(30), k: None };
        let second_input = DecyclerInput::from_slice(&first_result.values, second_params);
        let second_result = decycler_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_decycler_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = &candles.close;
        let period = 125;
        let params = DecyclerParams { hp_period: Some(period), k: None };
        let input = DecyclerInput::from_candles(&candles, "close", params);
        let decycler_result = decycler_with_kernel(&input, kernel)?;
        assert_eq!(decycler_result.values.len(), close_prices.len());
        if decycler_result.values.len() > 240 {
            for i in 240..decycler_result.values.len() {
                assert!(
                    !decycler_result.values[i].is_nan(),
                    "Expected no NaN after index 240, found NaN at {}",
                    i
                );
            }
        }
        Ok(())
    }

    macro_rules! generate_all_decycler_tests {
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
    generate_all_decycler_tests!(
        check_decycler_partial_params,
        check_decycler_accuracy,
        check_decycler_zero_period,
        check_decycler_period_exceed_length,
        check_decycler_very_small_dataset,
        check_decycler_reinput,
        check_decycler_nan_handling
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = DecyclerBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = DecyclerParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            60289.96384058519,
            60204.010366691065,
            60114.255563805666,
            60028.535266555904,
            59934.26876964316,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-6,
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                     Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
