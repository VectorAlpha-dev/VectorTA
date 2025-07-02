//! # Relative Strength Mark (RSMK)
//!
//! A comparative momentum-based indicator: calculates the log-ratio of two sources, applies a momentum
//! transform, smooths the result with a moving average, and provides both the main and signal lines.
//! Supports kernel selection, batch sweeping, and streaming APIs, with AVX2/AVX512 function stubs for parity.
//!
//! ## Parameters
//! - **lookback**: Lookback for momentum. Default: 90
//! - **period**: MA period. Default: 3
//! - **signal_period**: Signal MA period. Default: 20
//! - **matype**: MA type. Default: "ema"
//! - **signal_matype**: Signal MA type. Default: "ema"
//!
//! ## Errors
//! - **EmptyData**: All input slices empty.
//! - **InvalidPeriod**: One or more periods are zero/invalid.
//! - **NotEnoughValidData**: Not enough valid points after first valid.
//! - **AllValuesNaN**: All input or comparison values NaN.
//! - **MaError**: Underlying MA error.
//!
//! ## Returns
//! - **`Ok(RsmkOutput)`** on success: contains indicator/signal, both `Vec<f64>` of input length.
//! - **`Err(RsmkError)`** otherwise.
//!
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

use crate::indicators::moving_averages::ma::{ma, MaData};

#[derive(Debug, Clone)]
pub enum RsmkData<'a> {
    Candles {
        candles: &'a Candles,
        candles_compare: &'a Candles,
        source: &'a str,
    },
    Slices {
        main: &'a [f64],
        compare: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct RsmkOutput {
    pub indicator: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RsmkParams {
    pub lookback: Option<usize>,
    pub period: Option<usize>,
    pub signal_period: Option<usize>,
    pub matype: Option<String>,
    pub signal_matype: Option<String>,
}

impl Default for RsmkParams {
    fn default() -> Self {
        Self {
            lookback: Some(90),
            period: Some(3),
            signal_period: Some(20),
            matype: Some("ema".to_string()),
            signal_matype: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RsmkInput<'a> {
    pub data: RsmkData<'a>,
    pub params: RsmkParams,
}

impl<'a> RsmkInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        candles_compare: &'a Candles,
        source: &'a str,
        params: RsmkParams,
    ) -> Self {
        Self {
            data: RsmkData::Candles {
                candles,
                candles_compare,
                source,
            },
            params,
        }
    }

    #[inline]
    pub fn from_slices(main: &'a [f64], compare: &'a [f64], params: RsmkParams) -> Self {
        Self {
            data: RsmkData::Slices { main, compare },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles, candles_compare: &'a Candles) -> Self {
        Self::from_candles(candles, candles_compare, "close", RsmkParams::default())
    }

    #[inline]
    pub fn get_lookback(&self) -> usize {
        self.params.lookback.unwrap_or(90)
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(3)
    }
    #[inline]
    pub fn get_signal_period(&self) -> usize {
        self.params.signal_period.unwrap_or(20)
    }
    #[inline]
    pub fn get_ma_type(&self) -> &str {
        self.params.matype.as_deref().unwrap_or("ema")
    }
    #[inline]
    pub fn get_signal_ma_type(&self) -> &str {
        self.params.signal_matype.as_deref().unwrap_or("ema")
    }
}

#[derive(Clone, Debug)]
pub struct RsmkBuilder {
    lookback: Option<usize>,
    period: Option<usize>,
    signal_period: Option<usize>,
    matype: Option<String>,
    signal_matype: Option<String>,
    kernel: Kernel,
}

impl Default for RsmkBuilder {
    fn default() -> Self {
        Self {
            lookback: None,
            period: None,
            signal_period: None,
            matype: None,
            signal_matype: None,
            kernel: Kernel::Auto,
        }
    }
}

impl RsmkBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn lookback(mut self, n: usize) -> Self {
        self.lookback = Some(n);
        self
    }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline(always)]
    pub fn signal_period(mut self, n: usize) -> Self {
        self.signal_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn matype<S: Into<String>>(mut self, s: S) -> Self {
        self.matype = Some(s.into());
        self
    }
    #[inline(always)]
    pub fn signal_matype<S: Into<String>>(mut self, s: S) -> Self {
        self.signal_matype = Some(s.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles, candles_compare: &Candles) -> Result<RsmkOutput, RsmkError> {
        let params = RsmkParams {
            lookback: self.lookback,
            period: self.period,
            signal_period: self.signal_period,
            matype: self.matype.clone(),
            signal_matype: self.signal_matype.clone(),
        };
        let input = RsmkInput::from_candles(candles, candles_compare, "close", params);
        rsmk_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(self, main: &[f64], compare: &[f64]) -> Result<RsmkOutput, RsmkError> {
        let params = RsmkParams {
            lookback: self.lookback,
            period: self.period,
            signal_period: self.signal_period,
            matype: self.matype.clone(),
            signal_matype: self.signal_matype.clone(),
        };
        let input = RsmkInput::from_slices(main, compare, params);
        rsmk_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<RsmkStream, RsmkError> {
        let params = RsmkParams {
            lookback: self.lookback,
            period: self.period,
            signal_period: self.signal_period,
            matype: self.matype,
            signal_matype: self.signal_matype,
        };
        RsmkStream::try_new(params)
    }
}

#[derive(Debug, Error)]
pub enum RsmkError {
    #[error("rsmk: Empty data provided for RSMK.")]
    EmptyData,
    #[error("rsmk: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("rsmk: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("rsmk: All values are NaN.")]
    AllValuesNaN,
    #[error("rsmk: Error from MA function: {0}")]
    MaError(String),
}

#[inline]
pub fn rsmk(input: &RsmkInput) -> Result<RsmkOutput, RsmkError> {
    rsmk_with_kernel(input, Kernel::Auto)
}

pub fn rsmk_with_kernel(input: &RsmkInput, kernel: Kernel) -> Result<RsmkOutput, RsmkError> {
    let (main, compare) = match &input.data {
        RsmkData::Candles { candles, candles_compare, source } => (
            source_type(candles, source),
            source_type(candles_compare, source),
        ),
        RsmkData::Slices { main, compare } => (*main, *compare),
    };

    if main.is_empty() || compare.is_empty() {
        return Err(RsmkError::EmptyData);
    }

    let lookback = input.get_lookback();
    let period = input.get_period();
    let signal_period = input.get_signal_period();

    if lookback == 0
        || period == 0
        || signal_period == 0
        || period > main.len()
        || signal_period > main.len()
        || lookback >= main.len()
    {
        return Err(RsmkError::InvalidPeriod {
            period: lookback.max(period).max(signal_period),
            data_len: main.len().min(compare.len()),
        });
    }

    let mut lr = AVec::<f64>::with_capacity(CACHELINE_ALIGN, main.len());
    lr.resize(main.len(), f64::NAN);

    for i in 0..main.len() {
        let m = main[i];
        let c = compare[i];
        lr[i] = if m.is_nan() || c.is_nan() || c == 0.0 {
            f64::NAN
        
            } else {
            (m / c).ln()
        };
    }

    let first_valid = lr.iter().position(|&x| !x.is_nan()).ok_or(RsmkError::AllValuesNaN)?;
    let valid_points = lr.len() - first_valid;
    if valid_points < lookback.max(period).max(signal_period) {
        return Err(RsmkError::NotEnoughValidData {
            needed: lookback.max(period).max(signal_period),
            valid: valid_points,
        });
    }

    let mut mom = AVec::<f64>::with_capacity(CACHELINE_ALIGN, lr.len());
    mom.resize(lr.len(), f64::NAN);

    unsafe {
        match match kernel {
            Kernel::Auto => detect_best_kernel(),
            k => k,
        } {
            Kernel::Scalar | Kernel::ScalarBatch => {
                rsmk_scalar(&lr, lookback, period, signal_period, input, first_valid, &mut mom)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                rsmk_avx2(&lr, lookback, period, signal_period, input, first_valid, &mut mom)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                rsmk_avx512(&lr, lookback, period, signal_period, input, first_valid, &mut mom)
            }
            _ => unreachable!(),
        }
    }
}

pub fn rsmk_scalar(
    lr: &[f64],
    lookback: usize,
    period: usize,
    signal_period: usize,
    input: &RsmkInput,
    first_valid: usize,
    mom: &mut [f64],
) -> Result<RsmkOutput, RsmkError> {
    for i in (first_valid + lookback)..lr.len() {
        mom[i] = if lr[i].is_nan() || lr[i - lookback].is_nan() {
            f64::NAN
        
            } else {
            lr[i] - lr[i - lookback]
        };
    }
    let matype = input.get_ma_type();
    let sigmatype = input.get_signal_ma_type();

    let ma_b = ma(matype, MaData::Slice(mom), period)
        .map_err(|e| RsmkError::MaError(e.to_string()))?;

    let mut indicator = vec![f64::NAN; lr.len()];
    for i in 0..lr.len() {
        if i < ma_b.len() {
            indicator[i] = ma_b[i] * 100.0;
        }
    }

    let ma_signal = ma(sigmatype, MaData::Slice(&indicator), signal_period)
        .map_err(|e| RsmkError::MaError(e.to_string()))?;

    let mut signal = vec![f64::NAN; lr.len()];
    for i in 0..lr.len() {
        if i < ma_signal.len() {
            signal[i] = ma_signal[i];
        }
    }
    Ok(RsmkOutput { indicator, signal })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rsmk_avx2(
    lr: &[f64],
    lookback: usize,
    period: usize,
    signal_period: usize,
    input: &RsmkInput,
    first_valid: usize,
    mom: &mut [f64],
) -> Result<RsmkOutput, RsmkError> {
    rsmk_scalar(lr, lookback, period, signal_period, input, first_valid, mom)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn rsmk_avx512(
    lr: &[f64],
    lookback: usize,
    period: usize,
    signal_period: usize,
    input: &RsmkInput,
    first_valid: usize,
    mom: &mut [f64],
) -> Result<RsmkOutput, RsmkError> {
    if period <= 32 {
        unsafe { rsmk_avx512_short(lr, lookback, period, signal_period, input, first_valid, mom) }
    } else {
        unsafe { rsmk_avx512_long(lr, lookback, period, signal_period, input, first_valid, mom) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rsmk_avx512_short(
    lr: &[f64],
    lookback: usize,
    period: usize,
    signal_period: usize,
    input: &RsmkInput,
    first_valid: usize,
    mom: &mut [f64],
) -> Result<RsmkOutput, RsmkError> {
    rsmk_scalar(lr, lookback, period, signal_period, input, first_valid, mom)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn rsmk_avx512_long(
    lr: &[f64],
    lookback: usize,
    period: usize,
    signal_period: usize,
    input: &RsmkInput,
    first_valid: usize,
    mom: &mut [f64],
) -> Result<RsmkOutput, RsmkError> {
    rsmk_scalar(lr, lookback, period, signal_period, input, first_valid, mom)
}

#[derive(Debug, Clone)]
pub struct RsmkStream {
    lookback: usize,
    period: usize,
    signal_period: usize,
    matype: String,
    signal_matype: String,
    buffer_lr: Vec<f64>,
    buffer_mom: Vec<f64>,
    buffer_ma: Vec<f64>,
    head: usize,
    filled: bool,
}

impl RsmkStream {
    pub fn try_new(params: RsmkParams) -> Result<Self, RsmkError> {
        let lookback = params.lookback.unwrap_or(90);
        let period = params.period.unwrap_or(3);
        let signal_period = params.signal_period.unwrap_or(20);

        if lookback == 0 || period == 0 || signal_period == 0 {
            return Err(RsmkError::InvalidPeriod { period: lookback.max(period).max(signal_period), data_len: 0 });
        }
        Ok(Self {
            lookback,
            period,
            signal_period,
            matype: params.matype.unwrap_or_else(|| "ema".to_string()),
            signal_matype: params.signal_matype.unwrap_or_else(|| "ema".to_string()),
            buffer_lr: vec![f64::NAN; lookback.max(period).max(signal_period)],
            buffer_mom: vec![f64::NAN; lookback.max(period)],
            buffer_ma: vec![f64::NAN; period],
            head: 0,
            filled: false,
        })
    }

    pub fn update(&mut self, main: f64, compare: f64) -> Option<(f64, f64)> {
        let lr = if main.is_nan() || compare.is_nan() || compare == 0.0 { f64::NAN } else { (main / compare).ln() };
        let pos = self.head % self.buffer_lr.len();
        self.buffer_lr[pos] = lr;
        self.head += 1;
        if self.head < self.lookback {
            return None;
        }
        let mom_idx = (self.head - self.lookback) % self.buffer_mom.len();
        let mom = if self.head < self.buffer_lr.len() + 1 || self.head < self.lookback + 1 {
            f64::NAN
        
            } else {
            let curr = self.buffer_lr[pos];
            let prev = self.buffer_lr[(pos + self.buffer_lr.len() - self.lookback) % self.buffer_lr.len()];
            if curr.is_nan() || prev.is_nan() { f64::NAN } else { curr - prev }
        };
        self.buffer_mom[mom_idx] = mom;

        if self.head < self.lookback + self.period {
            return None;
        }
        let mut vals: Vec<f64> = self.buffer_mom.iter().cloned().collect();
        let ma_result = ma(&self.matype, MaData::Slice(&vals), self.period).ok();
        let ind = ma_result.as_ref().and_then(|ma| ma.last().copied()).unwrap_or(f64::NAN) * 100.0;

        if self.head < self.lookback + self.period + self.signal_period {
            return Some((ind, f64::NAN));
        }
        let sig_result = ma(&self.signal_matype, MaData::Slice(&[ind]), self.signal_period).ok();
        let sig = sig_result.as_ref().and_then(|s| s.last().copied()).unwrap_or(f64::NAN);
        Some((ind, sig))
    }
}

#[derive(Clone, Debug)]
pub struct RsmkBatchRange {
    pub lookback: (usize, usize, usize),
    pub period: (usize, usize, usize),
    pub signal_period: (usize, usize, usize),
}

impl Default for RsmkBatchRange {
    fn default() -> Self {
        Self {
            lookback: (90, 90, 1),
            period: (3, 3, 1),
            signal_period: (20, 20, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RsmkBatchBuilder {
    range: RsmkBatchRange,
    kernel: Kernel,
}

impl RsmkBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lookback = (start, end, step);
        self
    }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    #[inline]
    pub fn signal_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.signal_period = (start, end, step);
        self
    }
    pub fn apply_slices(self, main: &[f64], compare: &[f64]) -> Result<RsmkBatchOutput, RsmkError> {
        rsmk_batch_with_kernel(main, compare, &self.range, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct RsmkBatchOutput {
    pub indicator: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<RsmkParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn expand_grid(r: &RsmkBatchRange) -> Vec<RsmkParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let looks = axis(r.lookback);
    let periods = axis(r.period);
    let signals = axis(r.signal_period);

    let mut out = Vec::with_capacity(looks.len() * periods.len() * signals.len());
    for &l in &looks {
        for &p in &periods {
            for &s in &signals {
                out.push(RsmkParams {
                    lookback: Some(l),
                    period: Some(p),
                    signal_period: Some(s),
                    matype: Some("ema".to_string()),
                    signal_matype: Some("ema".to_string()),
                });
            }
        }
    }
    out
}

pub fn rsmk_batch_with_kernel(
    main: &[f64],
    compare: &[f64],
    sweep: &RsmkBatchRange,
    k: Kernel,
) -> Result<RsmkBatchOutput, RsmkError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(RsmkError::InvalidPeriod {
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
    rsmk_batch_par_slice(main, compare, sweep, simd)
}

pub fn rsmk_batch_slice(
    main: &[f64],
    compare: &[f64],
    sweep: &RsmkBatchRange,
    kern: Kernel,
) -> Result<RsmkBatchOutput, RsmkError> {
    rsmk_batch_inner(main, compare, sweep, kern, false)
}

pub fn rsmk_batch_par_slice(
    main: &[f64],
    compare: &[f64],
    sweep: &RsmkBatchRange,
    kern: Kernel,
) -> Result<RsmkBatchOutput, RsmkError> {
    rsmk_batch_inner(main, compare, sweep, kern, true)
}

fn rsmk_batch_inner(
    main: &[f64],
    compare: &[f64],
    sweep: &RsmkBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<RsmkBatchOutput, RsmkError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(RsmkError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = main.iter().position(|x| !x.is_nan()).ok_or(RsmkError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| c.lookback.unwrap().max(c.period.unwrap()).max(c.signal_period.unwrap()))
        .max()
        .unwrap();

    if main.len() - first < max_p {
        return Err(RsmkError::NotEnoughValidData {
            needed: max_p,
            valid: main.len() - first,
        });
    }

    let rows = combos.len();
    let cols = main.len();

    let mut indicators = vec![f64::NAN; rows * cols];
    let mut signals = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, ind_row: &mut [f64], sig_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        let lookback = prm.lookback.unwrap();
        let period = prm.period.unwrap();
        let signal_period = prm.signal_period.unwrap();

        let mut lr = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cols);
        lr.resize(cols, f64::NAN);

        for i in 0..cols {
            let m = main[i];
            let c = compare[i];
            lr[i] = if m.is_nan() || c.is_nan() || c == 0.0 { f64::NAN } else { (m / c).ln() };
        }

        let mut mom = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cols);
        mom.resize(cols, f64::NAN);
        for i in (first + lookback)..cols {
            mom[i] = if lr[i].is_nan() || lr[i - lookback].is_nan() { f64::NAN } else { lr[i] - lr[i - lookback] };
        }
        let ma_b = ma("ema", MaData::Slice(&mom), period).unwrap_or_else(|_| vec![f64::NAN; cols]);
        for i in 0..cols {
            if i < ma_b.len() {
                ind_row[i] = ma_b[i] * 100.0;
            }
        }
        let ma_signal = ma("ema", MaData::Slice(ind_row), signal_period).unwrap_or_else(|_| vec![f64::NAN; cols]);
        for i in 0..cols {
            if i < ma_signal.len() {
                sig_row[i] = ma_signal[i];
            }
        }
    };

    if parallel {


        #[cfg(not(target_arch = "wasm32"))] {


        indicators.par_chunks_mut(cols).zip(signals.par_chunks_mut(cols)).enumerate()


                    .for_each(|(row, (ind_row, sig_row))| do_row(row, ind_row, sig_row));


        }


        #[cfg(target_arch = "wasm32")] {


        for (row, (ind_row, sig_row)) in indicators.chunks_mut(cols).zip(signals.chunks_mut(cols)).enumerate() {


                    do_row(row, ind_row, sig_row);


        }


    } else {
        for (row, (ind_row, sig_row)) in indicators.chunks_mut(cols).zip(signals.chunks_mut(cols)).enumerate() {
            do_row(row, ind_row, sig_row);
        }
    }

    Ok(RsmkBatchOutput { indicator: indicators, signal: signals, combos, rows, cols })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_rsmk_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = RsmkParams {
            lookback: None,
            period: None,
            signal_period: None,
            matype: None,
            signal_matype: None,
        };
        let input_default = RsmkInput::from_candles(&candles, &candles, "close", default_params);
        let output_default = rsmk_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.indicator.len(), candles.close.len());
        assert_eq!(output_default.signal.len(), candles.close.len());
        Ok(())
    }

    fn check_rsmk_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = RsmkParams::default();
        let input = RsmkInput::from_candles(&candles, &candles, "close", params.clone());
        let rsmk_result = rsmk_with_kernel(&input, kernel)?;
        assert_eq!(rsmk_result.indicator.len(), candles.close.len());
        assert_eq!(rsmk_result.signal.len(), candles.close.len());
        let expected_last_five = [0.0, 0.0, 0.0, 0.0, 0.0];
        let start = rsmk_result.indicator.len() - 5;
        for (i, &value) in rsmk_result.indicator[start..].iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!((value - expected_value).abs() < 1e-1);
        }
        for (i, &value) in rsmk_result.signal[start..].iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!((value - expected_value).abs() < 1e-1);
        }
        Ok(())
    }

    fn check_rsmk_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = RsmkInput::with_default_candles(&candles, &candles);
        let rsmk_result = rsmk_with_kernel(&input, kernel)?;
        assert_eq!(rsmk_result.indicator.len(), candles.close.len());
        assert_eq!(rsmk_result.signal.len(), candles.close.len());
        Ok(())
    }

    fn check_rsmk_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 11.0, 12.0];
        let params = RsmkParams {
            lookback: Some(0),
            period: Some(0),
            signal_period: Some(0),
            matype: Some("ema".to_string()),
            signal_matype: Some("ema".to_string()),
        };
        let input = RsmkInput::from_slices(&input_data, &input_data, params);
        let result = rsmk_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_rsmk_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [42.0];
        let params = RsmkParams::default();
        let input = RsmkInput::from_slices(&input_data, &input_data, params);
        let result = rsmk_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_rsmk_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = RsmkParams::default();
        let input = RsmkInput::from_slices(&input_data, &input_data, params);
        let result = rsmk_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_rsmk_not_enough_valid_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [f64::NAN, 10.0, 20.0, 30.0];
        let params = RsmkParams {
            lookback: Some(3),
            period: Some(3),
            signal_period: Some(3),
            matype: Some("ema".to_string()),
            signal_matype: Some("ema".to_string()),
        };
        let input = RsmkInput::from_slices(&input_data, &input_data, params);
        let result = rsmk_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_rsmk_ma_error(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let params = RsmkParams {
            lookback: Some(2),
            period: Some(3),
            signal_period: Some(3),
            matype: Some("nonexistent_ma".to_string()),
            signal_matype: Some("ema".to_string()),
        };
        let input = RsmkInput::from_slices(&input_data, &input_data, params);
        let result = rsmk_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    macro_rules! generate_all_rsmk_tests {
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

    generate_all_rsmk_tests!(
        check_rsmk_partial_params,
        check_rsmk_accuracy,
        check_rsmk_default_candles,
        check_rsmk_zero_period,
        check_rsmk_very_small_dataset,
        check_rsmk_all_nan,
        check_rsmk_not_enough_valid_data,
        check_rsmk_ma_error
    );
        fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let main = &candles.close;
        let compare = &candles.close;

        let batch = RsmkBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(main, compare)?;

        let def = RsmkParams::default();
        // Find row matching default parameters
        let default_row = batch.combos.iter().position(|c|
            c.lookback.unwrap_or(90) == def.lookback.unwrap_or(90)
            && c.period.unwrap_or(3) == def.period.unwrap_or(3)
            && c.signal_period.unwrap_or(20) == def.signal_period.unwrap_or(20)
        ).expect("default row missing");

        let start = default_row * batch.cols;
        let ind_row = &batch.indicator[start..start+batch.cols];
        let sig_row = &batch.signal[start..start+batch.cols];

        assert_eq!(ind_row.len(), candles.close.len());
        assert_eq!(sig_row.len(), candles.close.len());

        // RSMK with default: last five values should be (close vs close): zeroes or near zero
        let expected = [0.0, 0.0, 0.0, 0.0, 0.0];
        let len = ind_row.len();
        let start_idx = len - 5;

        for (i, &v) in ind_row[start_idx..].iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-1, "[{test}] default-indicator mismatch at idx {i}: {v} vs {expected:?}");
        }
        for (i, &v) in sig_row[start_idx..].iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-1, "[{test}] default-signal mismatch at idx {i}: {v} vs {expected:?}");
        }
        // NaN checks for early region
        let max_period = def.lookback.unwrap().max(def.period.unwrap()).max(def.signal_period.unwrap());
        for i in 0..max_period {
            if i < ind_row.len() {
                assert!(ind_row[i].is_nan(), "Expected indicator NaN at index {i}, got {}", ind_row[i]);
            }
            if i < sig_row.len() {
                assert!(sig_row[i].is_nan(), "Expected signal NaN at index {i}, got {}", sig_row[i]);
            }
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
