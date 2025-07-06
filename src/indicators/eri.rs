//! # Elder Ray Index (ERI)
//!
//! The Elder Ray Index (ERI) measures bullish and bearish pressure using an MA as a baseline.
//! Parameters include `period` (MA window size) and `ma_type` (e.g., "ema", "sma").
//! Batch/grid computation and SIMD stubs provided for API parity with alma.rs.
//!
//! ## Parameters
//! - **period**: Window size for MA (default 13)
//! - **ma_type**: MA type (default "ema")
//!
//! ## Errors
//! - **AllValuesNaN**: eri: All input values are `NaN`.
//! - **InvalidPeriod**: eri: `period` is zero or exceeds data length.
//! - **NotEnoughValidData**: eri: Not enough valid data points for `period`.
//!
//! ## Returns
//! - **`Ok(EriOutput)`** on success, containing `bull` and `bear` vectors of length equal to the input.
//! - **`Err(EriError)`** otherwise.

use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;
use std::convert::AsRef;

#[derive(Debug, Clone)]
pub enum EriData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        source: &'a [f64],
    },
}

impl<'a> AsRef<[f64]> for EriInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EriData::Candles { candles, source } => source_type(candles, source),
            EriData::Slices { source, .. } => source,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EriOutput {
    pub bull: Vec<f64>,
    pub bear: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EriParams {
    pub period: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for EriParams {
    fn default() -> Self {
        Self {
            period: Some(13),
            ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EriInput<'a> {
    pub data: EriData<'a>,
    pub params: EriParams,
}

impl<'a> EriInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: EriParams) -> Self {
        Self {
            data: EriData::Candles { candles, source },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        source: &'a [f64],
        params: EriParams,
    ) -> Self {
        Self {
            data: EriData::Slices { high, low, source },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", EriParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(13)
    }
    #[inline]
    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .clone()
            .unwrap_or_else(|| "ema".to_string())
    }
}

#[derive(Clone, Debug)]
pub struct EriBuilder {
    period: Option<usize>,
    ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for EriBuilder {
    fn default() -> Self {
        Self {
            period: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EriBuilder {
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
    pub fn ma_type<S: Into<String>>(mut self, t: S) -> Self {
        self.ma_type = Some(t.into());
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EriOutput, EriError> {
        let p = EriParams {
            period: self.period,
            ma_type: self.ma_type,
        };
        let i = EriInput::from_candles(c, "close", p);
        eri_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        src: &[f64],
    ) -> Result<EriOutput, EriError> {
        let p = EriParams {
            period: self.period,
            ma_type: self.ma_type,
        };
        let i = EriInput::from_slices(high, low, src, p);
        eri_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<EriStream, EriError> {
        let p = EriParams {
            period: self.period,
            ma_type: self.ma_type,
        };
        EriStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum EriError {
    #[error("eri: All values are NaN.")]
    AllValuesNaN,
    #[error("eri: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("eri: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("eri: MA calculation error: {0}")]
    MaCalculationError(String),
    #[error("eri: Empty data provided.")]
    EmptyData,
}

#[inline]
pub fn eri(input: &EriInput) -> Result<EriOutput, EriError> {
    eri_with_kernel(input, Kernel::Auto)
}

pub fn eri_with_kernel(input: &EriInput, kernel: Kernel) -> Result<EriOutput, EriError> {
    let (high, low, source_data) = match &input.data {
        EriData::Candles { candles, source } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| EriError::EmptyData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| EriError::EmptyData)?;
            let src = source_type(candles, source);
            (high, low, src)
        }
        EriData::Slices { high, low, source } => (*high, *low, *source),
    };

    if source_data.is_empty() || high.is_empty() || low.is_empty() {
        return Err(EriError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > source_data.len() {
        return Err(EriError::InvalidPeriod {
            period,
            data_len: source_data.len(),
        });
    }

    let mut first_valid_idx = None;
    for i in 0..source_data.len() {
        if !(source_data[i].is_nan() || high[i].is_nan() || low[i].is_nan()) {
            first_valid_idx = Some(i);
            break;
        }
    }
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(EriError::AllValuesNaN),
    };

    if (source_data.len() - first_valid_idx) < period {
        return Err(EriError::NotEnoughValidData {
            needed: period,
            valid: source_data.len() - first_valid_idx,
        });
    }

    let ma_type = input.get_ma_type();
    let full_ma = ma(&ma_type, MaData::Slice(&source_data), period)
        .map_err(|e| EriError::MaCalculationError(e.to_string()))?;

    let mut bull = vec![f64::NAN; source_data.len()];
    let mut bear = vec![f64::NAN; source_data.len()];

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                eri_scalar(high, low, &full_ma, period, first_valid_idx, &mut bull, &mut bear)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                eri_avx2(high, low, &full_ma, period, first_valid_idx, &mut bull, &mut bear)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                eri_avx512(high, low, &full_ma, period, first_valid_idx, &mut bull, &mut bear)
            }
            _ => unreachable!(),
        }
    }

    Ok(EriOutput { bull, bear })
}

#[inline]
pub fn eri_scalar(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    period: usize,
    first_valid: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    for i in (first_valid + period - 1)..high.len() {
        bull[i] = high[i] - ma[i];
        bear[i] = low[i] - ma[i];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn eri_avx512(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    period: usize,
    first_valid: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    unsafe { eri_avx512_long(high, low, ma, period, first_valid, bull, bear) }
}

#[inline]
pub fn eri_avx2(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    period: usize,
    first_valid: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    eri_scalar(high, low, ma, period, first_valid, bull, bear)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn eri_avx512_short(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    period: usize,
    first_valid: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    eri_scalar(high, low, ma, period, first_valid, bull, bear)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn eri_avx512_long(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    period: usize,
    first_valid: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    eri_scalar(high, low, ma, period, first_valid, bull, bear)
}

#[inline]
pub fn eri_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &EriBatchRange,
    k: Kernel,
) -> Result<EriBatchOutput, EriError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(EriError::InvalidPeriod {
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
    eri_batch_par_slice(high, low, source, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct EriBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for EriBatchRange {
    fn default() -> Self {
        Self {
            period: (13, 60, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EriBatchBuilder {
    range: EriBatchRange,
    kernel: Kernel,
}

impl EriBatchBuilder {
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
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        source: &[f64],
    ) -> Result<EriBatchOutput, EriError> {
        eri_batch_with_kernel(high, low, source, &self.range, self.kernel)
    }
}

#[derive(Clone, Debug)]
pub struct EriBatchOutput {
    pub bull: Vec<f64>,
    pub bear: Vec<f64>,
    pub params: Vec<EriParams>,
    pub rows: usize,
    pub cols: usize,
}

impl EriBatchOutput {
    pub fn row_for_params(&self, p: &EriParams) -> Option<usize> {
        self.params.iter().position(|c| c.period == p.period)
    }
    pub fn values_for_bull(&self, p: &EriParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.bull[start..start + self.cols]
        })
    }
    pub fn values_for_bear(&self, p: &EriParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.bear[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &EriBatchRange) -> Vec<EriParams> {
    let (start, end, step) = r.period;
    if step == 0 || start == end {
        return vec![EriParams {
            period: Some(start),
            ma_type: None,
        }];
    }
    (start..=end)
        .step_by(step)
        .map(|p| EriParams {
            period: Some(p),
            ma_type: None,
        })
        .collect()
}

#[inline(always)]
pub fn eri_batch_slice(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &EriBatchRange,
    kern: Kernel,
) -> Result<EriBatchOutput, EriError> {
    eri_batch_inner(high, low, source, sweep, kern, false)
}

#[inline(always)]
pub fn eri_batch_par_slice(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &EriBatchRange,
    kern: Kernel,
) -> Result<EriBatchOutput, EriError> {
    eri_batch_inner(high, low, source, sweep, kern, true)
}

#[inline(always)]
fn eri_batch_inner(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &EriBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EriBatchOutput, EriError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(EriError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = source
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EriError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if source.len() - first < max_p {
        return Err(EriError::NotEnoughValidData {
            needed: max_p,
            valid: source.len() - first,
        });
    }
    let rows = combos.len();
    let cols = source.len();
    let mut bull = vec![f64::NAN; rows * cols];
    let mut bear = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, bull_row: &mut [f64], bear_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        let ma_type = combos[row]
            .ma_type
            .clone()
            .unwrap_or_else(|| "ema".to_string());
        let ma_vec = ma(&ma_type, MaData::Slice(source), period)
            .map_err(|e| EriError::MaCalculationError(e.to_string()))?;
        match kern {
            Kernel::Scalar => eri_row_scalar(
                high,
                low,
                &ma_vec,
                first,
                period,
                bull_row,
                bear_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => eri_row_avx2(
                high,
                low,
                &ma_vec,
                first,
                period,
                bull_row,
                bear_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => eri_row_avx512(
                high,
                low,
                &ma_vec,
                first,
                period,
                bull_row,
                bear_row,
            ),
            _ => unreachable!(),
        }
        Ok::<(), EriError>(())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            bull.par_chunks_mut(cols)
                .zip(bear.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (bull_row, bear_row))| {
                    let _ = do_row(row, bull_row, bear_row);
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, (bull_row, bear_row)) in bull
                .chunks_mut(cols)
                .zip(bear.chunks_mut(cols))
                .enumerate()
            {
                let _ = do_row(row, bull_row, bear_row);
            }
        }
    } else {
        for (row, (bull_row, bear_row)) in bull
            .chunks_mut(cols)
            .zip(bear.chunks_mut(cols))
            .enumerate()
        {
            let _ = do_row(row, bull_row, bear_row);
        }
    }

    Ok(EriBatchOutput {
        bull,
        bear,
        params: combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn eri_row_scalar(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    first: usize,
    period: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    for i in (first + period - 1)..high.len() {
        bull[i] = high[i] - ma[i];
        bear[i] = low[i] - ma[i];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn eri_row_avx2(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    first: usize,
    period: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    eri_row_scalar(high, low, ma, first, period, bull, bear)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn eri_row_avx512(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    first: usize,
    period: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    if period <= 32 {
        eri_row_avx512_short(high, low, ma, first, period, bull, bear);
    } else {
        eri_row_avx512_long(high, low, ma, first, period, bull, bear);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn eri_row_avx512_short(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    first: usize,
    period: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    eri_row_scalar(high, low, ma, first, period, bull, bear)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn eri_row_avx512_long(
    high: &[f64],
    low: &[f64],
    ma: &[f64],
    first: usize,
    period: usize,
    bull: &mut [f64],
    bear: &mut [f64],
) {
    eri_row_scalar(high, low, ma, first, period, bull, bear)
}

#[derive(Debug, Clone)]
pub struct EriStream {
    period: usize,
    ma_type: String,
    ma_buffer: Vec<f64>,
    high_buffer: Vec<f64>,
    low_buffer: Vec<f64>,
    idx: usize,
    filled: bool,
}

impl EriStream {
    pub fn try_new(params: EriParams) -> Result<Self, EriError> {
        let period = params.period.unwrap_or(13);
        if period == 0 {
            return Err(EriError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let ma_type = params.ma_type.unwrap_or_else(|| "ema".to_string());
        Ok(Self {
            period,
            ma_type,
            ma_buffer: vec![f64::NAN; period],
            high_buffer: vec![f64::NAN; period],
            low_buffer: vec![f64::NAN; period],
            idx: 0,
            filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, source: f64) -> Option<(f64, f64)> {
        self.ma_buffer[self.idx] = source;
        self.high_buffer[self.idx] = high;
        self.low_buffer[self.idx] = low;
        self.idx = (self.idx + 1) % self.period;

        if !self.filled && self.idx == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        let ring_ma = ma(
            &self.ma_type,
            MaData::Slice(&self.ma_buffer),
            self.period,
        )
        .ok()
        .and_then(|ma_v| ma_v.last().copied())
        .unwrap_or(f64::NAN);

        let hi = self.high_buffer[self.idx];
        let lo = self.low_buffer[self.idx];
        Some((hi - ring_ma, lo - ring_ma))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_eri_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = EriParams {
            period: None,
            ma_type: None,
        };
        let input_default = EriInput::from_candles(&candles, "close", default_params);
        let output_default = eri_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.bull.len(), candles.close.len());
        assert_eq!(output_default.bear.len(), candles.close.len());

        let params_period_14 = EriParams {
            period: Some(14),
            ma_type: Some("ema".to_string()),
        };
        let input_period_14 = EriInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 = eri_with_kernel(&input_period_14, kernel)?;
        assert_eq!(output_period_14.bull.len(), candles.close.len());
        assert_eq!(output_period_14.bear.len(), candles.close.len());

        let params_custom = EriParams {
            period: Some(20),
            ma_type: Some("sma".to_string()),
        };
        let input_custom = EriInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = eri_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.bull.len(), candles.close.len());
        assert_eq!(output_custom.bear.len(), candles.close.len());

        Ok(())
    }

    fn check_eri_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = EriParams {
            period: Some(13),
            ma_type: Some("ema".to_string()),
        };
        let input = EriInput::from_candles(&candles, "close", params);
        let eri_result = eri_with_kernel(&input, kernel)?;

        assert_eq!(eri_result.bull.len(), close_prices.len());
        assert_eq!(eri_result.bear.len(), close_prices.len());

        let expected_bull_last_five = [
            -103.35343557205488,
            6.839912366813223,
            -42.851503685589705,
            -9.444146016219747,
            11.476446271808527,
        ];
        let expected_bear_last_five = [
            -433.3534355720549,
            -314.1600876331868,
            -414.8515036855897,
            -336.44414601621975,
            -925.5235537281915,
        ];

        let start_index = eri_result.bull.len() - 5;
        for i in 0..5 {
            let actual_bull = eri_result.bull[start_index + i];
            let actual_bear = eri_result.bear[start_index + i];
            let expected_bull = expected_bull_last_five[i];
            let expected_bear = expected_bear_last_five[i];
            assert!(
                (actual_bull - expected_bull).abs() < 1e-2,
                "ERI bull mismatch at index {}: expected {}, got {}",
                i, expected_bull, actual_bull
            );
            assert!(
                (actual_bear - expected_bear).abs() < 1e-2,
                "ERI bear mismatch at index {}: expected {}, got {}",
                i, expected_bear, actual_bear
            );
        }
        Ok(())
    }

    fn check_eri_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EriInput::with_default_candles(&candles);
        match input.data {
            EriData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EriData::Candles"),
        }
        let output = eri_with_kernel(&input, kernel)?;
        assert_eq!(output.bull.len(), candles.close.len());
        assert_eq!(output.bear.len(), candles.close.len());

        Ok(())
    }

    fn check_eri_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [8.0, 18.0, 28.0];
        let src = [9.0, 19.0, 29.0];
        let params = EriParams {
            period: Some(0),
            ma_type: Some("ema".to_string()),
        };
        let input = EriInput::from_slices(&high, &low, &src, params);
        let res = eri_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] ERI should fail with zero period", test_name);
        Ok(())
    }

    fn check_eri_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 20.0, 30.0];
        let low = [8.0, 18.0, 28.0];
        let src = [9.0, 19.0, 29.0];
        let params = EriParams {
            period: Some(10),
            ma_type: Some("ema".to_string()),
        };
        let input = EriInput::from_slices(&high, &low, &src, params);
        let res = eri_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] ERI should fail with period exceeding length", test_name);
        Ok(())
    }

    fn check_eri_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [42.0];
        let low = [40.0];
        let src = [41.0];
        let params = EriParams {
            period: Some(9),
            ma_type: Some("ema".to_string()),
        };
        let input = EriInput::from_slices(&high, &low, &src, params);
        let res = eri_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] ERI should fail with insufficient data", test_name);
        Ok(())
    }

    fn check_eri_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = EriParams {
            period: Some(14),
            ma_type: Some("ema".to_string()),
        };
        let first_input = EriInput::from_candles(&candles, "close", first_params);
        let first_result = eri_with_kernel(&first_input, kernel)?;

        assert_eq!(first_result.bull.len(), candles.close.len());
        assert_eq!(first_result.bear.len(), candles.close.len());

        let second_params = EriParams {
            period: Some(14),
            ma_type: Some("ema".to_string()),
        };
        let second_input = EriInput::from_slices(
            &first_result.bull,
            &first_result.bear,
            &first_result.bull,
            second_params,
        );
        let second_result = eri_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.bull.len(), first_result.bull.len());
        assert_eq!(second_result.bear.len(), first_result.bear.len());

        for i in 28..second_result.bull.len() {
            assert!(
                !second_result.bull[i].is_nan(),
                "Expected no NaN in bull after index 28, but found NaN at index {}",
                i
            );
            assert!(
                !second_result.bear[i].is_nan(),
                "Expected no NaN in bear after index 28, but found NaN at index {}",
                i
            );
        }
        Ok(())
    }

    fn check_eri_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EriInput::from_candles(
            &candles,
            "close",
            EriParams {
                period: Some(13),
                ma_type: Some("ema".to_string()),
            },
        );
        let res = eri_with_kernel(&input, kernel)?;
        assert_eq!(res.bull.len(), candles.close.len());
        if res.bull.len() > 240 {
            for (i, &val) in res.bull[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at bull-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        if res.bear.len() > 240 {
            for (i, &val) in res.bear[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at bear-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }

    // Macro to generate kernel-variant tests as in alma.rs
    macro_rules! generate_all_eri_tests {
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

    generate_all_eri_tests!(
        check_eri_partial_params,
        check_eri_accuracy,
        check_eri_default_candles,
        check_eri_zero_period,
        check_eri_period_exceeds_length,
        check_eri_very_small_dataset,
        check_eri_reinput,
        check_eri_nan_handling
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let high = c.select_candle_field("high").unwrap();
        let low = c.select_candle_field("low").unwrap();
        let src = c.select_candle_field("close").unwrap();

        let output = EriBatchBuilder::new()
            .kernel(kernel)
            .period_static(13)
            .apply_slices(high, low, src)?;

        let def = EriParams::default();
        let row = output.values_for_bull(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            -103.35343557205488,
            6.839912366813223,
            -42.851503685589705,
            -9.444146016219747,
            11.476446271808527,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-2,
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

