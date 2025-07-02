//! # Polarized Fractal Efficiency (PFE)
//!
//! Measures the efficiency of price movement over a period, producing signed values
//! (positive = upward efficiency, negative = downward), then smooths with EMA.
//!
//! ## Parameters
//! - **period**: Lookback window (default: 10)
//! - **smoothing**: EMA smoothing window (default: 5)
//!
//! ## Errors
//! - **AllValuesNaN**: pfe: All input data values are `NaN`.
//! - **InvalidPeriod**: pfe: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: pfe: Not enough valid data points for the requested `period`.
//!
//! ## Returns
//! - **`Ok(PfeOutput)`**: `Vec<f64>` matching input length (leading NaN for non-computable values)
//! - **`Err(PfeError)`** otherwise
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

impl<'a> AsRef<[f64]> for PfeInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PfeData::Slice(slice) => slice,
            PfeData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PfeData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PfeOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PfeParams {
    pub period: Option<usize>,
    pub smoothing: Option<usize>,
}

impl Default for PfeParams {
    fn default() -> Self {
        Self {
            period: Some(10),
            smoothing: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PfeInput<'a> {
    pub data: PfeData<'a>,
    pub params: PfeParams,
}

impl<'a> PfeInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: PfeParams) -> Self {
        Self {
            data: PfeData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: PfeParams) -> Self {
        Self {
            data: PfeData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", PfeParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(10)
    }
    #[inline]
    pub fn get_smoothing(&self) -> usize {
        self.params.smoothing.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PfeBuilder {
    period: Option<usize>,
    smoothing: Option<usize>,
    kernel: Kernel,
}

impl Default for PfeBuilder {
    fn default() -> Self {
        Self {
            period: None,
            smoothing: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PfeBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline(always)]
    pub fn smoothing(mut self, s: usize) -> Self {
        self.smoothing = Some(s);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<PfeOutput, PfeError> {
        let p = PfeParams {
            period: self.period,
            smoothing: self.smoothing,
        };
        let i = PfeInput::from_candles(c, "close", p);
        pfe_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<PfeOutput, PfeError> {
        let p = PfeParams {
            period: self.period,
            smoothing: self.smoothing,
        };
        let i = PfeInput::from_slice(d, p);
        pfe_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<PfeStream, PfeError> {
        let p = PfeParams {
            period: self.period,
            smoothing: self.smoothing,
        };
        PfeStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum PfeError {
    #[error("pfe: All values are NaN.")]
    AllValuesNaN,
    #[error("pfe: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("pfe: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("pfe: Invalid smoothing: {smoothing}")]
    InvalidSmoothing { smoothing: usize },
}

#[inline]
pub fn pfe(input: &PfeInput) -> Result<PfeOutput, PfeError> {
    pfe_with_kernel(input, Kernel::Auto)
}

pub fn pfe_with_kernel(input: &PfeInput, kernel: Kernel) -> Result<PfeOutput, PfeError> {
    let data: &[f64] = match &input.data {
        PfeData::Candles { candles, source } => source_type(candles, source),
        PfeData::Slice(sl) => sl,
    };

    let first = data.iter().position(|x| !x.is_nan()).ok_or(PfeError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    let smoothing = input.get_smoothing();

    if period == 0 || period > len {
        return Err(PfeError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(PfeError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if smoothing == 0 {
        return Err(PfeError::InvalidSmoothing { smoothing });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                pfe_scalar(data, period, smoothing, first, &mut vec![])
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                pfe_avx2(data, period, smoothing, first, &mut vec![])
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                pfe_avx512(data, period, smoothing, first, &mut vec![])
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn pfe_scalar(
    data: &[f64],
    period: usize,
    smoothing: usize,
    _first_valid: usize,
    _out: &mut Vec<f64>,
) -> Result<PfeOutput, PfeError> {
    // 1. Use a full 'period'-bar gap (not period - 1)
    let ln = period;
    let len = data.len();
    if period > len {
        // (though upstream should already catch this)
        return Err(PfeError::InvalidPeriod { period, data_len: len });
    }

    // diff_len = number of indices i for which i + period < data.len()
    let diff_len = len.saturating_sub(period);

    // 2. Build diff_array[i] = data[i+period] - data[i]
    let mut diff_array = vec![f64::NAN; diff_len];
    for i in 0..diff_len {
        diff_array[i] = data[i + period] - data[i];
    }

    // 3. Compute the "long leg" for each i: sqrt( ΔP² + (period)² )
    let mut a_array = vec![f64::NAN; diff_len];
    for i in 0..diff_len {
        let d = diff_array[i];
        a_array[i] = (d.powi(2) + (period as f64).powi(2)).sqrt();
    }

    // 4. Compute the "short leg" by summing 'period' single-bar distances
    //    for j from i..i+period (exclusive of i+period), doing sqrt( (ΔP)² + 1 ).
    let mut b_array = vec![f64::NAN; diff_len];
    for i in 0..diff_len {
        let mut b_sum = 0.0;
        let start = i;
        // j runs from i..(i + period - 1), so that j+1 runs i+1..i+period
        for j in start..(start + period) {
            let step_diff = data[j + 1] - data[j];
            b_sum += (1.0 + step_diff.powi(2)).sqrt();
        }
        b_array[i] = b_sum;
    }

    // 5. Form raw PFE = 100 * (long_leg / short_leg)
    //    If denom is zero (should never happen with real candles),
    //    we set rawPFE = 0 to avoid division by zero.
    let mut pfe_tmp = vec![f64::NAN; diff_len];
    for i in 0..diff_len {
        if b_array[i].abs() < f64::EPSILON {
            pfe_tmp[i] = 0.0;
        
            } else {
            pfe_tmp[i] = 100.0 * a_array[i] / b_array[i];
        }
    }

    // 6. Apply sign: if diff_array[i] < 0, then negate
    let mut signed_pfe = vec![f64::NAN; diff_len];
    for i in 0..diff_len {
        let d = diff_array[i];
        if d.is_nan() {
            signed_pfe[i] = f64::NAN;
        } else if d > 0.0 {
            signed_pfe[i] = pfe_tmp[i];
        
            } else {
            signed_pfe[i] = -pfe_tmp[i];
        }
    }

    // 7. EMA‐smooth the signed PFE (period = 'smoothing')
    let alpha = 2.0 / (smoothing as f64 + 1.0);
    let mut ema_array = vec![f64::NAN; diff_len];
    let mut started = false;
    let mut ema_val = 0.0;
    for i in 0..diff_len {
        let val = signed_pfe[i];
        if val.is_nan() {
            ema_array[i] = f64::NAN;
        } else if !started {
            // first non‐NaN seeds the EMA
            ema_val = val;
            ema_array[i] = val;
            started = true;
        
            } else {
            ema_val = alpha * val + (1.0 - alpha) * ema_val;
            ema_array[i] = ema_val;
        }
    }

    // 8. Build final output vector of length = data.len(), with leading NaNs
    let mut pfe_values = vec![f64::NAN; len];
    for (i, &val) in ema_array.iter().enumerate() {
        let out_idx = i + ln;  // i + period
        if out_idx < pfe_values.len() {
            pfe_values[out_idx] = val;
        }
    }

    Ok(PfeOutput { values: pfe_values })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pfe_avx512(
    data: &[f64],
    period: usize,
    smoothing: usize,
    first_valid: usize,
    out: &mut Vec<f64>,
) -> Result<PfeOutput, PfeError> {
    pfe_scalar(data, period, smoothing, first_valid, out)
}

#[inline]
pub fn pfe_avx2(
    data: &[f64],
    period: usize,
    smoothing: usize,
    first_valid: usize,
    out: &mut Vec<f64>,
) -> Result<PfeOutput, PfeError> {
    pfe_scalar(data, period, smoothing, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pfe_avx512_short(
    data: &[f64],
    period: usize,
    smoothing: usize,
    first_valid: usize,
    out: &mut Vec<f64>,
) -> Result<PfeOutput, PfeError> {
    pfe_scalar(data, period, smoothing, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pfe_avx512_long(
    data: &[f64],
    period: usize,
    smoothing: usize,
    first_valid: usize,
    out: &mut Vec<f64>,
) -> Result<PfeOutput, PfeError> {
    pfe_scalar(data, period, smoothing, first_valid, out)
}

#[inline]
pub fn pfe_batch_with_kernel(
    data: &[f64],
    sweep: &PfeBatchRange,
    k: Kernel,
) -> Result<PfeBatchOutput, PfeError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(PfeError::InvalidPeriod {
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
    pfe_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct PfeBatchRange {
    pub period: (usize, usize, usize),
    pub smoothing: (usize, usize, usize),
}

impl Default for PfeBatchRange {
    fn default() -> Self {
        Self {
            period: (10, 40, 1),
            smoothing: (5, 10, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PfeBatchBuilder {
    range: PfeBatchRange,
    kernel: Kernel,
}

impl PfeBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    #[inline]
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    #[inline]
    pub fn smoothing_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smoothing = (start, end, step);
        self
    }
    #[inline]
    pub fn smoothing_static(mut self, s: usize) -> Self {
        self.range.smoothing = (s, s, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<PfeBatchOutput, PfeError> {
        pfe_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<PfeBatchOutput, PfeError> {
        PfeBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<PfeBatchOutput, PfeError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<PfeBatchOutput, PfeError> {
        PfeBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct PfeBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PfeParams>,
    pub rows: usize,
    pub cols: usize,
}
impl PfeBatchOutput {
    pub fn row_for_params(&self, p: &PfeParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(10) == p.period.unwrap_or(10)
                && c.smoothing.unwrap_or(5) == p.smoothing.unwrap_or(5)
        })
    }
    pub fn values_for(&self, p: &PfeParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &PfeBatchRange) -> Vec<PfeParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis(r.period);
    let smoothings = axis(r.smoothing);

    let mut out = Vec::with_capacity(periods.len() * smoothings.len());
    for &p in &periods {
        for &s in &smoothings {
            out.push(PfeParams {
                period: Some(p),
                smoothing: Some(s),
            });
        }
    }
    out
}

#[inline(always)]
pub fn pfe_batch_slice(
    data: &[f64],
    sweep: &PfeBatchRange,
    kern: Kernel,
) -> Result<PfeBatchOutput, PfeError> {
    pfe_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn pfe_batch_par_slice(
    data: &[f64],
    sweep: &PfeBatchRange,
    kern: Kernel,
) -> Result<PfeBatchOutput, PfeError> {
    pfe_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn pfe_batch_inner(
    data: &[f64],
    sweep: &PfeBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<PfeBatchOutput, PfeError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(PfeError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data.iter().position(|x| !x.is_nan()).ok_or(PfeError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(PfeError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let smoothing = combos[row].smoothing.unwrap();
        match kern {
            Kernel::Scalar => {
                let out = pfe_row_scalar(data, first, period, smoothing, out_row);
                out;
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => {
                let out = pfe_row_avx2(data, first, period, smoothing, out_row);
                out;
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                let out = pfe_row_avx512(data, first, period, smoothing, out_row);
                out;
            }
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
    Ok(PfeBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn pfe_row_scalar(
    data: &[f64],
    _first_valid: usize,
    period: usize,
    smoothing: usize,
    out: &mut [f64],
) {
    // 1. Use a full 'period'-bar gap (not period - 1)
    let ln = period;
    let len = data.len();
    let diff_len = len.saturating_sub(ln);

    // 2. Build diff_array[i] = data[i + period] - data[i]
    let mut diff_array = Vec::with_capacity(diff_len);
    diff_array.set_len(diff_len);
    for i in 0..diff_len {
        diff_array[i] = data[i + ln] - data[i];
    }

    // 3. Compute the "long leg" for each i: sqrt( ΔP² + (period)² )
    let mut a_array = Vec::with_capacity(diff_len);
    a_array.set_len(diff_len);
    for i in 0..diff_len {
        let d = diff_array[i];
        a_array[i] = (d.powi(2) + (period as f64).powi(2)).sqrt();
    }

    // 4. Compute the "short leg" by summing 'period' single-bar distances
    //    for j in i..(i+period), summing sqrt(1 + (ΔP)²) over exactly 'period' intervals.
    let mut b_array = Vec::with_capacity(diff_len);
    b_array.set_len(diff_len);
    for i in 0..diff_len {
        let mut b_sum = 0.0;
        let start = i;
        // j runs from i..(i + period - 1), so that j+1 runs i+1..i+period
        for j in start..(start + ln) {
            let step_diff = data[j + 1] - data[j];
            b_sum += (1.0 + step_diff.powi(2)).sqrt();
        }
        b_array[i] = b_sum;
    }

    // 5. Form raw PFE = 100 * (long_leg / short_leg)
    let mut pfe_tmp = Vec::with_capacity(diff_len);
    pfe_tmp.set_len(diff_len);
    for i in 0..diff_len {
        if b_array[i].abs() < f64::EPSILON {
            pfe_tmp[i] = 0.0;
        
            } else {
            pfe_tmp[i] = 100.0 * a_array[i] / b_array[i];
        }
    }

    // 6. Apply sign: if diff_array[i] < 0, then negate
    let mut signed_pfe = Vec::with_capacity(diff_len);
    signed_pfe.set_len(diff_len);
    for i in 0..diff_len {
        let d = diff_array[i];
        if d.is_nan() {
            signed_pfe[i] = f64::NAN;
        } else if d > 0.0 {
            signed_pfe[i] = pfe_tmp[i];
        
            } else {
            signed_pfe[i] = -pfe_tmp[i];
        }
    }

    // 7. EMA‐smooth the signed PFE (period = 'smoothing')
    let alpha = 2.0 / (smoothing as f64 + 1.0);
    let mut ema_array = Vec::with_capacity(diff_len);
    ema_array.set_len(diff_len);
    let mut started = false;
    let mut ema_val = 0.0;
    for i in 0..diff_len {
        let val = signed_pfe[i];
        if val.is_nan() {
            ema_array[i] = f64::NAN;
        } else if !started {
            ema_val = val;
            ema_array[i] = val;
            started = true;
        
            } else {
            ema_val = alpha * val + (1.0 - alpha) * ema_val;
            ema_array[i] = ema_val;
        }
    }

    // 8. Write results into `out`, offset by period
    for (i, &val) in ema_array.iter().enumerate() {
        let out_idx = i + ln;
        if out_idx < out.len() {
            out[out_idx] = val;
        }
    }
}


#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pfe_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    smoothing: usize,
    out: &mut [f64],
) {
    pfe_row_scalar(data, first, period, smoothing, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pfe_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    smoothing: usize,
    out: &mut [f64],
) {
    pfe_row_scalar(data, first, period, smoothing, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pfe_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    smoothing: usize,
    out: &mut [f64],
) {
    pfe_row_scalar(data, first, period, smoothing, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn pfe_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    smoothing: usize,
    out: &mut [f64],
) {
    pfe_row_scalar(data, first, period, smoothing, out)
}

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct PfeStream {
    period: usize,
    smoothing: usize,
    // We keep period+1 elements so that we can always refer to [0] = P_{t-period}
    // and [period] = P_t.  When length < period+1, return None.
    buffer: VecDeque<f64>,

    // EMA state
    ema_val: f64,
    started: bool,
}

impl PfeStream {
    pub fn try_new(params: PfeParams) -> Result<Self, PfeError> {
        let period = params.period.unwrap_or(10);
        if period == 0 {
            return Err(PfeError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let smoothing = params.smoothing.unwrap_or(5);
        if smoothing == 0 {
            return Err(PfeError::InvalidSmoothing { smoothing });
        }

        Ok(Self {
            period,
            smoothing,
            buffer: VecDeque::with_capacity(period + 1),
            ema_val: 0.0,
            started: false,
        })
    }

    /// Pushes one new price into the stream.  Returns `None` until we have
    /// collected (period+1) values.  Once we have exactly period+1 values,
    /// we compute:
    ///   diff = P_t - P_{t-period},
    ///   numerator = sqrt(diff² + period²),
    ///   denominator = sum_{i=0..period-1} sqrt(1 + (ΔP)^2) over the sliding window,
    ///   raw_pfe = 100 * (numerator / denominator) with correct sign,
    ///   then EMA-smooth that raw value.
    pub fn update(&mut self, price: f64) -> Option<f64> {
        // 1) Push new price, and ensure we never keep more than (period + 1) elements.
        self.buffer.push_back(price);
        if self.buffer.len() > self.period + 1 {
            self.buffer.pop_front();
        }

        // 2) If we don’t yet have (period+1) points, return None
        if self.buffer.len() < self.period + 1 {
            return None;
        }

        // 3) Now buffer.len() == period+1.  Let:
        //      front = P_{t-period},   // buffer[0]
        //      back  = P_t,            // buffer[period]
        let front = self.buffer[0];
        let back = *self.buffer.get(self.period).unwrap();

        // 4) Compute diff = P_t - P_{t-period}
        let diff = back - front;

        // 5) Long leg = sqrt(diff² + period²)
        let long_leg = (diff.powi(2) + (self.period as f64).powi(2)).sqrt();

        // 6) Short leg = sum_{i=0..period-1} sqrt(1 + (window[i+1]-window[i])²)
        let mut short_leg = 0.0;
        for i in 0..self.period {
            let p_i = self.buffer[i];
            let p_next = self.buffer[i + 1];
            let step_diff = p_next - p_i;
            short_leg += (1.0 + step_diff.powi(2)).sqrt();
        }

        // 7) raw PFE = 100 * (long_leg / short_leg), or 0 if denominator ≈ 0
        let raw_pfe = if short_leg.abs() < f64::EPSILON {
            0.0
        
            } else {
            100.0 * long_leg / short_leg
        };

        // 8) Apply sign based on diff
        let signed = if diff > 0.0 { raw_pfe } else { -raw_pfe };

        // 9) EMA‐smooth using alpha = 2/(smoothing+1)
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        let out_val = if !self.started {
            // seed the EMA on the first available raw value
            self.ema_val = signed;
            self.started = true;
            signed
        
            } else {
            self.ema_val = alpha * signed + (1.0 - alpha) * self.ema_val;
            self.ema_val
        };

        Some(out_val)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_pfe_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = PfeParams {
            period: None,
            smoothing: None,
        };
        let input = PfeInput::from_candles(&candles, "close", default_params);
        let output = pfe_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_pfe_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = &candles.close;

        let params = PfeParams {
            period: Some(10),
            smoothing: Some(5),
        };
        let input = PfeInput::from_candles(&candles, "close", params);
        let pfe_result = pfe_with_kernel(&input, kernel)?;

        assert_eq!(pfe_result.values.len(), close_prices.len());

        let expected_last_five_pfe = [
            -13.03562252,
            -11.93979855,
            -9.94609862,
            -9.73372410,
            -14.88374798,
        ];
        let start_index = pfe_result.values.len() - 5;
        let result_last_five_pfe = &pfe_result.values[start_index..];
        for (i, &value) in result_last_five_pfe.iter().enumerate() {
            let expected_value = expected_last_five_pfe[i];
            assert!(
                (value - expected_value).abs() < 1e-8,
                "[{}] PFE mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                value,
                expected_value
            );
        }

        for i in 0..(10 - 1) {
            assert!(pfe_result.values[i].is_nan());
        }

        Ok(())
    }

    fn check_pfe_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = PfeInput::with_default_candles(&candles);
        match input.data {
            PfeData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected PfeData::Candles"),
        }
        let output = pfe_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_pfe_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = PfeParams {
            period: Some(0),
            smoothing: Some(5),
        };
        let input = PfeInput::from_slice(&input_data, params);
        let res = pfe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PFE should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_pfe_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = PfeParams {
            period: Some(10),
            smoothing: Some(2),
        };
        let input = PfeInput::from_slice(&data_small, params);
        let res = pfe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PFE should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_pfe_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = PfeParams {
            period: Some(10),
            smoothing: Some(2),
        };
        let input = PfeInput::from_slice(&single_point, params);
        let res = pfe_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] PFE should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_pfe_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = PfeParams {
            period: Some(10),
            smoothing: Some(5),
        };
        let first_input = PfeInput::from_candles(&candles, "close", first_params);
        let first_result = pfe_with_kernel(&first_input, kernel)?;

        let second_params = PfeParams {
            period: Some(10),
            smoothing: Some(5),
        };
        let second_input = PfeInput::from_slice(&first_result.values, second_params);
        let second_result = pfe_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 20..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] Expected no NaN after index 20, but found NaN at index {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    fn check_pfe_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = PfeInput::from_candles(
            &candles,
            "close",
            PfeParams {
                period: Some(10),
                smoothing: Some(5),
            },
        );
        let res = pfe_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    fn check_pfe_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 10;
        let smoothing = 5;

        let input = PfeInput::from_candles(
            &candles,
            "close",
            PfeParams {
                period: Some(period),
                smoothing: Some(smoothing),
            },
        );
        let batch_output = pfe_with_kernel(&input, kernel)?.values;

        let mut stream = PfeStream::try_new(PfeParams {
            period: Some(period),
            smoothing: Some(smoothing),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
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
                "[{}] PFE streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_pfe_tests {
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

    generate_all_pfe_tests!(
        check_pfe_partial_params,
        check_pfe_accuracy,
        check_pfe_default_candles,
        check_pfe_zero_period,
        check_pfe_period_exceeds_length,
        check_pfe_very_small_dataset,
        check_pfe_reinput,
        check_pfe_nan_handling,
        check_pfe_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = PfeBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = PfeParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            -13.03562252,
            -11.93979855,
            -9.94609862,
            -9.73372410,
            -14.88374798,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
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
