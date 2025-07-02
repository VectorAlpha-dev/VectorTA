//! # Parabolic SAR (SAR)
//!
//! The Parabolic SAR is a trend-following indicator that uses a system of
//! progressively rising (in an uptrend) or falling (in a downtrend) dots.
//!
//! ## Parameters
//! - **acceleration**: Acceleration factor. Defaults to 0.02.
//! - **maximum**: Maximum acceleration. Defaults to 0.2.
//!
//! ## Errors
//! - **EmptyData**: sar: Input data slice is empty.
//! - **AllValuesNaN**: sar: All high/low values are `NaN`.
//! - **NotEnoughValidData**: Fewer than 2 valid (non-`NaN`) data points after the first valid index.
//! - **InvalidAcceleration**: sar: acceleration ≤ 0.0 or NaN.
//! - **InvalidMaximum**: sar: maximum ≤ 0.0 or NaN.
//!
//! ## Returns
//! - **`Ok(SarOutput)`** on success, containing a `Vec<f64>` matching the input length,
//!   with leading `NaN`s until the calculation starts.
//! - **`Err(SarError)`** otherwise.

use crate::utilities::data_loader::{Candles};
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel};
use crate::utilities::enums::Kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

impl<'a> AsRef<(Vec<f64>, Vec<f64>)> for SarInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &(Vec<f64>, Vec<f64>) {
        unreachable!("Do not use AsRef for SarInput directly")
    }
}

#[derive(Debug, Clone)]
pub enum SarData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct SarOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SarParams {
    pub acceleration: Option<f64>,
    pub maximum: Option<f64>,
}

impl Default for SarParams {
    fn default() -> Self {
        Self {
            acceleration: Some(0.02),
            maximum: Some(0.2),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SarInput<'a> {
    pub data: SarData<'a>,
    pub params: SarParams,
}

impl<'a> SarInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: SarParams) -> Self {
        Self {
            data: SarData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        params: SarParams,
    ) -> Self {
        Self {
            data: SarData::Slices { high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SarData::Candles { candles },
            params: SarParams::default(),
        }
    }

    #[inline]
    pub fn get_acceleration(&self) -> f64 {
        self.params.acceleration.unwrap_or(0.02)
    }

    #[inline]
    pub fn get_maximum(&self) -> f64 {
        self.params.maximum.unwrap_or(0.2)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SarBuilder {
    acceleration: Option<f64>,
    maximum: Option<f64>,
    kernel: Kernel,
}

impl Default for SarBuilder {
    fn default() -> Self {
        Self {
            acceleration: None,
            maximum: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SarBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn acceleration(mut self, v: f64) -> Self {
        self.acceleration = Some(v);
        self
    }
    #[inline(always)]
    pub fn maximum(mut self, v: f64) -> Self {
        self.maximum = Some(v);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<SarOutput, SarError> {
        let params = SarParams {
            acceleration: self.acceleration,
            maximum: self.maximum,
        };
        let input = SarInput::from_candles(c, params);
        sar_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<SarOutput, SarError> {
        let params = SarParams {
            acceleration: self.acceleration,
            maximum: self.maximum,
        };
        let input = SarInput::from_slices(high, low, params);
        sar_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<SarStream, SarError> {
        let params = SarParams {
            acceleration: self.acceleration,
            maximum: self.maximum,
        };
        SarStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum SarError {
    #[error("sar: Empty data provided for SAR.")]
    EmptyData,
    #[error("sar: All values are NaN.")]
    AllValuesNaN,
    #[error("sar: Not enough valid data. needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("sar: Invalid acceleration: {acceleration}")]
    InvalidAcceleration { acceleration: f64 },
    #[error("sar: Invalid maximum: {maximum}")]
    InvalidMaximum { maximum: f64 },
}

#[inline]
pub fn sar(input: &SarInput) -> Result<SarOutput, SarError> {
    sar_with_kernel(input, Kernel::Auto)
}

pub fn sar_with_kernel(input: &SarInput, kernel: Kernel) -> Result<SarOutput, SarError> {
    let (high, low) = match &input.data {
        SarData::Candles { candles } => (candles.high.as_slice(), candles.low.as_slice()),
        SarData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(SarError::EmptyData);
    }

    let first_valid_idx = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan());
    let first = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(SarError::AllValuesNaN),
    };

    if (high.len() - first) < 2 {
        return Err(SarError::NotEnoughValidData {
            needed: 2,
            valid: high.len() - first,
        });
    }

    let acceleration = input.get_acceleration();
    let maximum = input.get_maximum();

    if !(acceleration > 0.0) || acceleration.is_nan() || acceleration.is_infinite() {
        return Err(SarError::InvalidAcceleration { acceleration });
    }
    if !(maximum > 0.0) || maximum.is_nan() || maximum.is_infinite() {
        return Err(SarError::InvalidMaximum { maximum });
    }

    let mut out = vec![f64::NAN; high.len()];

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                sar_scalar(high, low, first, acceleration, maximum, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                sar_avx2(high, low, first, acceleration, maximum, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                sar_avx512(high, low, first, acceleration, maximum, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(SarOutput { values: out })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn sar_avx512(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    if high.len() <= 32 {
        unsafe { sar_avx512_short(high, low, first_valid, acceleration, maximum, out) }
    } else {
        unsafe { sar_avx512_long(high, low, first_valid, acceleration, maximum, out) }
    }
}

#[inline]
pub fn sar_scalar(
    high: &[f64],
    low: &[f64],
    first: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    let len = high.len();
    let mut trend_up;
    let mut sar;
    let mut ep;
    let i0 = first;
    let i1 = i0 + 1;
    if high[i1] > high[i0] {
        trend_up = true;
        sar = low[i0];
        ep = high[i1];
    
        } else {
        trend_up = false;
        sar = high[i0];
        ep = low[i1];
    }
    let mut acc = acceleration;
    out[i0] = f64::NAN;
    out[i1] = sar;

    for i in (i1..len).skip(1) {
        let mut next_sar = sar + acc * (ep - sar);
        if trend_up {
            if low[i] < next_sar {
                trend_up = false;
                next_sar = ep;
                ep = low[i];
                acc = acceleration;
            } else {
                if high[i] > ep {
                    ep = high[i];
                    acc = (acc + acceleration).min(maximum);
                }
                let prev = i.saturating_sub(1);
                let pre_prev = i.saturating_sub(2);
                if prev < len {
                    next_sar = next_sar.min(low[prev]);
                }
                if pre_prev < len {
                    next_sar = next_sar.min(low[pre_prev]);
                }
            }
        } else {
                if high[i] > next_sar {
                    trend_up = true;
                    next_sar = ep;
                    ep = high[i];
                    acc = acceleration;
                } else {
                    if low[i] < ep {
                        ep = low[i];
                        acc = (acc + acceleration).min(maximum);
                    }
                    let prev = i.saturating_sub(1);
                    let pre_prev = i.saturating_sub(2);
                    if prev < len {
                        next_sar = next_sar.max(high[prev]);
                    }
                    if pre_prev < len {
                        next_sar = next_sar.max(high[pre_prev]);
                    }
                }
            }
        out[i] = next_sar;
        sar = next_sar;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn sar_avx2(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    // AVX2 stub points to scalar
    sar_scalar(high, low, first_valid, acceleration, maximum, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn sar_avx512_short(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    // AVX512 short stub points to scalar
    sar_scalar(high, low, first_valid, acceleration, maximum, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn sar_avx512_long(
    high: &[f64],
    low: &[f64],
    first_valid: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    // AVX512 long stub points to scalar
    sar_scalar(high, low, first_valid, acceleration, maximum, out)
}

// Streaming

#[derive(Debug, Clone)]
pub struct SarStream {
    acceleration: f64,
    maximum: f64,
    state: Option<StreamState>,
    idx: usize,
}

#[derive(Debug, Clone)]
struct StreamState {
    trend_up: bool,
    sar: f64,
    ep: f64,
    acc: f64,
    prev: Option<f64>,
    prev2: Option<f64>,
}

impl SarStream {
    pub fn try_new(params: SarParams) -> Result<Self, SarError> {
        let acceleration = params.acceleration.unwrap_or(0.02);
        let maximum = params.maximum.unwrap_or(0.2);

        if !(acceleration > 0.0) || acceleration.is_nan() || acceleration.is_infinite() {
            return Err(SarError::InvalidAcceleration { acceleration });
        }
        if !(maximum > 0.0) || maximum.is_nan() || maximum.is_infinite() {
            return Err(SarError::InvalidMaximum { maximum });
        }

        Ok(Self {
            acceleration,
            maximum,
            state: None,
            idx: 0,
        })
    }

    pub fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        self.idx += 1;

        match self.state.as_mut() {
            None => {
                self.state = Some(StreamState {
                    trend_up: false,
                    sar: f64::NAN,
                    ep: f64::NAN,
                    acc: self.acceleration,
                    prev: Some(high),
                    prev2: None,
                });
                None
            }
            Some(st) if self.idx == 2 => {
                let prev_high = st.prev.unwrap();
                let prev_low = low;
                let (trend_up, sar, ep) = if high > prev_high {
                    (true, prev_low, high)
} else {
                    (false, prev_high, low)
                };
                *st = StreamState {
                    trend_up,
                    sar,
                    ep,
                    acc: self.acceleration,
                    prev: Some(high),
                    prev2: st.prev,
                };
                Some(sar)
            }
            Some(st) => {
                let mut next_sar = st.sar + st.acc * (st.ep - st.sar);
                if st.trend_up {
                    if low < next_sar {
                        st.trend_up = false;
                        next_sar = st.ep;
                        st.ep = low;
                        st.acc = self.acceleration;
                    } else {
                        if high > st.ep {
                            st.ep = high;
                            st.acc = (st.acc + self.acceleration).min(self.maximum);
                        }
                        if let Some(p) = st.prev {
                            next_sar = next_sar.min(p);
                        }
                        if let Some(p2) = st.prev2 {
                            next_sar = next_sar.min(p2);
                        }
                    }
                } else {
                    if high > next_sar {
                        st.trend_up = true;
                        next_sar = st.ep;
                        st.ep = high;
                        st.acc = self.acceleration;
                    } else {
                        if low < st.ep {
                            st.ep = low;
                            st.acc = (st.acc + self.acceleration).min(self.maximum);
                        }
                        if let Some(p) = st.prev {
                            next_sar = next_sar.max(p);
                        }
                        if let Some(p2) = st.prev2 {
                            next_sar = next_sar.max(p2);
                        }
                    }
                }
                st.prev2 = st.prev;
                st.prev = Some(high);
                st.sar = next_sar;
                Some(next_sar)
            }
        }
    }
}

// Batch

#[derive(Clone, Debug)]
pub struct SarBatchRange {
    pub acceleration: (f64, f64, f64),
    pub maximum: (f64, f64, f64),
}

impl Default for SarBatchRange {
    fn default() -> Self {
        Self {
            acceleration: (0.02, 0.2, 0.02),
            maximum: (0.2, 0.2, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SarBatchBuilder {
    range: SarBatchRange,
    kernel: Kernel,
}

impl SarBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn acceleration_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.acceleration = (start, end, step);
        self
    }
    pub fn acceleration_static(mut self, x: f64) -> Self {
        self.range.acceleration = (x, x, 0.0);
        self
    }
    pub fn maximum_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.maximum = (start, end, step);
        self
    }
    pub fn maximum_static(mut self, x: f64) -> Self {
        self.range.maximum = (x, x, 0.0);
        self
    }

    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<SarBatchOutput, SarError> {
        sar_batch_with_kernel(high, low, &self.range, self.kernel)
    }

    pub fn with_default_slices(high: &[f64], low: &[f64], k: Kernel) -> Result<SarBatchOutput, SarError> {
        SarBatchBuilder::new().kernel(k).apply_slices(high, low)
    }

    pub fn apply_candles(self, c: &Candles) -> Result<SarBatchOutput, SarError> {
        self.apply_slices(&c.high, &c.low)
    }

    pub fn with_default_candles(c: &Candles) -> Result<SarBatchOutput, SarError> {
        SarBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

pub fn sar_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &SarBatchRange,
    k: Kernel,
) -> Result<SarBatchOutput, SarError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(SarError::EmptyData),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    sar_batch_par_slice(high, low, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SarBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SarParams>,
    pub rows: usize,
    pub cols: usize,
}
impl SarBatchOutput {
    pub fn row_for_params(&self, p: &SarParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            (c.acceleration.unwrap_or(0.02) - p.acceleration.unwrap_or(0.02)).abs() < 1e-12
                && (c.maximum.unwrap_or(0.2) - p.maximum.unwrap_or(0.2)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &SarParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &SarBatchRange) -> Vec<SarParams> {
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

    let accs = axis_f64(r.acceleration);
    let maxs = axis_f64(r.maximum);

    let mut out = Vec::with_capacity(accs.len() * maxs.len());
    for &a in &accs {
        for &m in &maxs {
            out.push(SarParams {
                acceleration: Some(a),
                maximum: Some(m),
            });
        }
    }
    out
}

#[inline(always)]
pub fn sar_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &SarBatchRange,
    kern: Kernel,
) -> Result<SarBatchOutput, SarError> {
    sar_batch_inner(high, low, sweep, kern, false)
}

#[inline(always)]
pub fn sar_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &SarBatchRange,
    kern: Kernel,
) -> Result<SarBatchOutput, SarError> {
    sar_batch_inner(high, low, sweep, kern, true)
}

#[inline(always)]
fn sar_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &SarBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SarBatchOutput, SarError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SarError::EmptyData);
    }
    let first = high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
        .ok_or(SarError::AllValuesNaN)?;

    if high.len() - first < 2 {
        return Err(SarError::NotEnoughValidData {
            needed: 2,
            valid: high.len() - first,
        });
    }
    let rows = combos.len();
    let cols = high.len();
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let p = &combos[row];
        match kern {
            Kernel::Scalar => sar_row_scalar(high, low, first, p.acceleration.unwrap(), p.maximum.unwrap(), out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => sar_row_avx2(high, low, first, p.acceleration.unwrap(), p.maximum.unwrap(), out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => sar_row_avx512(high, low, first, p.acceleration.unwrap(), p.maximum.unwrap(), out_row),
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

        }
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(SarBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn sar_row_scalar(
    high: &[f64],
    low: &[f64],
    first: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    sar_scalar(high, low, first, acceleration, maximum, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sar_row_avx2(
    high: &[f64],
    low: &[f64],
    first: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    sar_scalar(high, low, first, acceleration, maximum, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn sar_row_avx512(
    high: &[f64],
    low: &[f64],
    first: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    if high.len() <= 32 {
        sar_row_avx512_short(high, low, first, acceleration, maximum, out);
    
        } else {
        sar_row_avx512_long(high, low, first, acceleration, maximum, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn sar_row_avx512_short(
    high: &[f64],
    low: &[f64],
    first: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    sar_scalar(high, low, first, acceleration, maximum, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn sar_row_avx512_long(
    high: &[f64],
    low: &[f64],
    first: usize,
    acceleration: f64,
    maximum: f64,
    out: &mut [f64],
) {
    sar_scalar(high, low, first, acceleration, maximum, out)
}

#[inline(always)]
fn expand_grid_for_test(r: &SarBatchRange) -> Vec<SarParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_sar_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = SarParams {
            acceleration: None,
            maximum: None,
        };
        let input = SarInput::from_candles(&candles, default_params);
        let output = sar_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_sar_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = SarInput::from_candles(&candles, SarParams::default());
        let result = sar_with_kernel(&input, kernel)?;
        let expected_last_five = [
            60370.00224209362,
            60220.362107568006,
            60079.70038111392,
            59947.478358247085,
            59823.189656752256,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-4,
                "[{}] SAR {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_sar_from_slices(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [50000.0, 50500.0, 51000.0];
        let low = [49000.0, 49500.0, 49900.0];
        let params = SarParams::default();
        let input = SarInput::from_slices(&high, &low, params);
        let result = sar_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), high.len());
        Ok(())
    }

    fn check_sar_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let params = SarParams::default();
        let input = SarInput::from_slices(&high, &low, params);
        let result = sar_with_kernel(&input, kernel);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
        Ok(())
    }

    macro_rules! generate_all_sar_tests {
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

    generate_all_sar_tests!(
        check_sar_partial_params,
        check_sar_accuracy,
        check_sar_from_slices,
        check_sar_all_nan
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = SarBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;

        let def = SarParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            60370.00224209362,
            60220.362107568006,
            60079.70038111392,
            59947.478358247085,
            59823.189656752256,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-4,
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
    gen_batch_tests!(check_batch_default_row);
}
