//! # Average True Range (ATR)
//!
//! Measures volatility using the average of true ranges over a period. API closely mirrors alma.rs for consistency.
//!
//! ## Parameters
//! - **length**: Window size for smoothing (defaults to 14).
//!
//! ## Errors
//! - **InvalidLength**: ATR: Length is zero.
//! - **InconsistentSliceLengths**: ATR: Provided slices have differing lengths.
//! - **NoCandlesAvailable**: ATR: No data.
//! - **NotEnoughData**: ATR: Data length too short for requested window.
//!
//! ## Returns
//! - **`Ok(AtrOutput)`** on success, containing a `Vec<f64>` matching input length.
//! - **`Err(AtrError)`** otherwise.
//!

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AtrData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct AtrOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AtrParams {
    pub length: Option<usize>,
}

impl Default for AtrParams {
    fn default() -> Self {
        Self { length: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AtrInput<'a> {
    pub data: AtrData<'a>,
    pub params: AtrParams,
}

impl<'a> AtrInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AtrParams) -> Self {
        Self {
            data: AtrData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: AtrParams,
    ) -> Self {
        Self {
            data: AtrData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, AtrParams::default())
    }
    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(14)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AtrBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for AtrBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AtrBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn length(mut self, n: usize) -> Self {
        self.length = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AtrOutput, AtrError> {
        let p = AtrParams {
            length: self.length,
        };
        let i = AtrInput::from_candles(c, p);
        atr_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AtrOutput, AtrError> {
        let p = AtrParams {
            length: self.length,
        };
        let i = AtrInput::from_slices(high, low, close, p);
        atr_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<AtrStream, AtrError> {
        let p = AtrParams {
            length: self.length,
        };
        AtrStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AtrError {
    #[error("Invalid length for ATR calculation (length={length}).")]
    InvalidLength { length: usize },
    #[error("Inconsistent slice lengths for ATR calculation: high={high_len}, low={low_len}, close={close_len}")]
    InconsistentSliceLengths {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error("No candles available for ATR calculation.")]
    NoCandlesAvailable,
    #[error("Not enough data to calculate ATR: length={length}, data length={data_len}")]
    NotEnoughData { length: usize, data_len: usize },
}

#[inline]
pub fn atr(input: &AtrInput) -> Result<AtrOutput, AtrError> {
    atr_with_kernel(input, Kernel::Auto)
}

pub fn atr_with_kernel(input: &AtrInput, kernel: Kernel) -> Result<AtrOutput, AtrError> {
    let (high, low, close) = match &input.data {
        AtrData::Candles { candles } => (
            candles
                .select_candle_field("high")
                .map_err(|_| AtrError::NoCandlesAvailable)?,
            candles
                .select_candle_field("low")
                .map_err(|_| AtrError::NoCandlesAvailable)?,
            candles
                .select_candle_field("close")
                .map_err(|_| AtrError::NoCandlesAvailable)?,
        ),
        AtrData::Slices { high, low, close } => {
            if high.len() != low.len() || low.len() != close.len() {
                return Err(AtrError::InconsistentSliceLengths {
                    high_len: high.len(),
                    low_len: low.len(),
                    close_len: close.len(),
                });
            }
            (*high, *low, *close)
        }
    };

    let len = close.len();
    let length = input.get_length();
    if length == 0 {
        return Err(AtrError::InvalidLength { length });
    }
    if len == 0 {
        return Err(AtrError::NoCandlesAvailable);
    }
    if length > len {
        return Err(AtrError::NotEnoughData {
            length,
            data_len: len,
        });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let mut out = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => atr_scalar(high, low, close, length, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => atr_avx2(high, low, close, length, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => atr_avx512(high, low, close, length, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(AtrOutput { values: out })
}

#[inline]
pub fn atr_scalar(high: &[f64], low: &[f64], close: &[f64], length: usize, out: &mut [f64]) {
    let len = close.len();
    let alpha = 1.0 / length as f64;
    let mut sum_tr = 0.0;
    let mut rma = f64::NAN;
    for i in 0..len {
        let tr = if i == 0 {
            high[0] - low[0]
        } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };
        if i < length {
            sum_tr += tr;
            if i == length - 1 {
                rma = sum_tr / length as f64;
                out[i] = rma;
            } else {
                out[i] = f64::NAN;
            }
        } else {
            rma += alpha * (tr - rma);
            out[i] = rma;
        }
    }
}

// -- AVX2/AVX512 always point to scalar; structuring for parity/stub compatibility --
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn atr_avx512(high: &[f64], low: &[f64], close: &[f64], length: usize, out: &mut [f64]) {
    atr_scalar(high, low, close, length, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn atr_avx2(high: &[f64], low: &[f64], close: &[f64], length: usize, out: &mut [f64]) {
    atr_scalar(high, low, close, length, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn atr_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    out: &mut [f64],
) {
    atr_scalar(high, low, close, length, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn atr_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    out: &mut [f64],
) {
    atr_scalar(high, low, close, length, out)
}

// Streaming object for rolling ATR (like AlmaStream)
#[derive(Debug, Clone)]
pub struct AtrStream {
    length: usize,
    buffer: Vec<f64>,
    idx: usize,
    filled: bool,
    prev_close: f64,
    rma: f64,
    sum: f64,
}

impl AtrStream {
    pub fn try_new(params: AtrParams) -> Result<Self, AtrError> {
        let length = params.length.unwrap_or(14);
        if length == 0 {
            return Err(AtrError::InvalidLength { length });
        }
        Ok(Self {
            length,
            buffer: vec![0.0; length],
            idx: 0,
            filled: false,
            prev_close: f64::NAN,
            rma: f64::NAN,
            sum: 0.0,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = if self.prev_close.is_nan() {
            high - low
        } else {
            let hl = high - low;
            let hc = (high - self.prev_close).abs();
            let lc = (low - self.prev_close).abs();
            hl.max(hc).max(lc)
        };
        self.prev_close = close;
        self.sum += tr - self.buffer[self.idx];
        self.buffer[self.idx] = tr;
        self.idx = (self.idx + 1) % self.length;
        if !self.filled && self.idx == 0 {
            self.filled = true;
            self.rma = self.sum / self.length as f64;
            return Some(self.rma);
        }
        if !self.filled {
            None
        } else {
            let alpha = 1.0 / self.length as f64;
            self.rma += alpha * (tr - self.rma);
            Some(self.rma)
        }
    }
}

// Batch/param sweep types for parity with alma.rs

#[derive(Clone, Debug)]
pub struct AtrBatchRange {
    pub length: (usize, usize, usize),
}
impl Default for AtrBatchRange {
    fn default() -> Self {
        Self {
            length: (14, 30, 1),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct AtrBatchBuilder {
    range: AtrBatchRange,
    kernel: Kernel,
}
impl AtrBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }
    #[inline]
    pub fn length_static(mut self, p: usize) -> Self {
        self.range.length = (p, p, 0);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<AtrBatchOutput, AtrError> {
        atr_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<AtrBatchOutput, AtrError> {
        let high = c
            .select_candle_field("high")
            .map_err(|_| AtrError::NoCandlesAvailable)?;
        let low = c
            .select_candle_field("low")
            .map_err(|_| AtrError::NoCandlesAvailable)?;
        let close = c
            .select_candle_field("close")
            .map_err(|_| AtrError::NoCandlesAvailable)?;
        self.apply_slices(high, low, close)
    }
}

#[derive(Clone, Debug)]
pub struct AtrBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AtrParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AtrBatchOutput {
    pub fn row_for_params(&self, p: &AtrParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.length.unwrap_or(14) == p.length.unwrap_or(14))
    }
    pub fn values_for(&self, p: &AtrParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &AtrBatchRange) -> Vec<AtrParams> {
    let (start, end, step) = r.length;
    if step == 0 || start == end {
        return vec![AtrParams {
            length: Some(start),
        }];
    }
    (start..=end)
        .step_by(step)
        .map(|l| AtrParams { length: Some(l) })
        .collect()
}

pub fn atr_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrBatchRange,
    k: Kernel,
) -> Result<AtrBatchOutput, AtrError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(AtrError::InvalidLength { length: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    atr_batch_par_slice(high, low, close, sweep, simd)
}

#[inline(always)]
pub fn atr_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrBatchRange,
    kern: Kernel,
) -> Result<AtrBatchOutput, AtrError> {
    atr_batch_inner(high, low, close, sweep, kern, false)
}
#[inline(always)]
pub fn atr_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrBatchRange,
    kern: Kernel,
) -> Result<AtrBatchOutput, AtrError> {
    atr_batch_inner(high, low, close, sweep, kern, true)
}

fn atr_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &AtrBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AtrBatchOutput, AtrError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(AtrError::InvalidLength { length: 0 });
    }
    let len = close.len();
    let rows = combos.len();
    let cols = len;
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let length = combos[row].length.unwrap();
        match kern {
            Kernel::Scalar => atr_row_scalar(high, low, close, length, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => atr_row_avx2(high, low, close, length, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => atr_row_avx512(high, low, close, length, out_row),
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
    Ok(AtrBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn atr_row_scalar(high: &[f64], low: &[f64], close: &[f64], length: usize, out: &mut [f64]) {
    atr_scalar(high, low, close, length, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn atr_row_avx2(high: &[f64], low: &[f64], close: &[f64], length: usize, out: &mut [f64]) {
    atr_scalar(high, low, close, length, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn atr_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    out: &mut [f64],
) {
    if length <= 32 {
        atr_row_avx512_short(high, low, close, length, out);
    } else {
        atr_row_avx512_long(high, low, close, length, out);
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn atr_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    out: &mut [f64],
) {
    atr_scalar(high, low, close, length, out);
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn atr_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
    out: &mut [f64],
) {
    atr_scalar(high, low, close, length, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_atr_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = AtrParams { length: None };
        let input_partial = AtrInput::from_candles(&candles, partial_params);
        let result_partial = atr_with_kernel(&input_partial, kernel)?;
        assert_eq!(result_partial.values.len(), candles.close.len());
        let zero_and_none_params = AtrParams { length: Some(14) };
        let input_zero_and_none = AtrInput::from_candles(&candles, zero_and_none_params);
        let result_zero_and_none = atr_with_kernel(&input_zero_and_none, kernel)?;
        assert_eq!(result_zero_and_none.values.len(), candles.close.len());
        Ok(())
    }

    fn check_atr_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AtrInput::with_default_candles(&candles);
        let result = atr_with_kernel(&input, kernel)?;
        let expected_last_five = [916.89, 874.33, 838.45, 801.92, 811.57];
        assert!(result.values.len() >= 5, "Not enough ATR values");
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "ATR output length does not match input length!"
        );
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &value) in last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-2,
                "ATR value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }
        let length = 14;
        for val in result.values.iter().skip(length - 1) {
            if !val.is_nan() {
                assert!(
                    val.is_finite(),
                    "ATR output should be finite after RMA stabilizes"
                );
            }
        }
        Ok(())
    }

    fn check_atr_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AtrInput::with_default_candles(&candles);
        match input.data {
            AtrData::Candles { .. } => {}
            _ => panic!("Expected AtrData::Candles variant"),
        }
        let default_params = AtrParams::default();
        assert_eq!(input.params.length, default_params.length);
        let output = atr_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_atr_zero_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let zero_length_params = AtrParams { length: Some(0) };
        let input_zero_length = AtrInput::from_candles(&candles, zero_length_params);
        let result_zero_length = atr_with_kernel(&input_zero_length, kernel);
        assert!(result_zero_length.is_err());
        Ok(())
    }

    fn check_atr_length_exceeding_data_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let too_long_params = AtrParams {
            length: Some(candles.close.len() + 10),
        };
        let input_too_long = AtrInput::from_candles(&candles, too_long_params);
        let result_too_long = atr_with_kernel(&input_too_long, kernel);
        assert!(result_too_long.is_err());
        Ok(())
    }

    fn check_atr_very_small_data_set(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0];
        let low = [5.0];
        let close = [7.0];
        let params = AtrParams { length: Some(14) };
        let input = AtrInput::from_slices(&high, &low, &close, params);
        let result = atr_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_atr_with_slice_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = AtrParams { length: Some(14) };
        let first_input = AtrInput::from_candles(&candles, first_params);
        let first_result = atr_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = AtrParams { length: Some(5) };
        let second_input = AtrInput::from_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            second_params,
        );
        let second_result = atr_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_atr_accuracy_nan_check(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = AtrParams { length: Some(14) };
        let input = AtrInput::from_candles(&candles, params);
        let result = atr_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
        Ok(())
    }

    macro_rules! generate_all_atr_tests {
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

    generate_all_atr_tests!(
        check_atr_partial_params,
        check_atr_accuracy,
        check_atr_default_candles,
        check_atr_zero_length,
        check_atr_length_exceeding_data_length,
        check_atr_very_small_data_set,
        check_atr_with_slice_data_reinput,
        check_atr_accuracy_nan_check
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = AtrBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = AtrParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [916.89, 874.33, 838.45, 801.92, 811.57];
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

// ============= Python and WASM Bindings =============

#[cfg(feature = "python")]
use pyo3::create_exception;

// Custom Python exceptions for each AtrError variant
#[cfg(feature = "python")]
create_exception!(atr, InvalidLengthError, PyValueError);
#[cfg(feature = "python")]
create_exception!(atr, InconsistentSliceLengthsError, PyValueError);
#[cfg(feature = "python")]
create_exception!(atr, NoCandlesAvailableError, PyValueError);
#[cfg(feature = "python")]
create_exception!(atr, NotEnoughDataError, PyValueError);

#[cfg(feature = "python")]
impl From<AtrError> for PyErr {
    fn from(err: AtrError) -> PyErr {
        match err {
            AtrError::InvalidLength { length } => {
                InvalidLengthError::new_err(format!("Invalid length for ATR calculation (length={}).", length))
            }
            AtrError::InconsistentSliceLengths { high_len, low_len, close_len } => {
                InconsistentSliceLengthsError::new_err(format!(
                    "Inconsistent slice lengths for ATR calculation: high={}, low={}, close={}",
                    high_len, low_len, close_len
                ))
            }
            AtrError::NoCandlesAvailable => {
                NoCandlesAvailableError::new_err("No candles available for ATR calculation.")
            }
            AtrError::NotEnoughData { length, data_len } => {
                NotEnoughDataError::new_err(format!(
                    "Not enough data to calculate ATR: length={}, data length={}",
                    length, data_len
                ))
            }
        }
    }
}

/// Helper function to prepare ATR calculation parameters
#[inline(always)]
fn atr_prepare<'a>(
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    length: usize,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize), AtrError> {
    // Check for mismatched lengths
    if high.len() != low.len() || low.len() != close.len() {
        return Err(AtrError::InconsistentSliceLengths {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    // Check for empty data
    if close.is_empty() {
        return Err(AtrError::NoCandlesAvailable);
    }

    // Check for zero length
    if length == 0 {
        return Err(AtrError::InvalidLength { length });
    }

    // Check if we have enough data
    if length > close.len() {
        return Err(AtrError::NotEnoughData {
            length,
            data_len: close.len(),
        });
    }

    Ok((high, low, close, length))
}

#[cfg(feature = "python")]
#[pyfunction(name = "atr")]
#[pyo3(signature = (high, low, close, length=14, kernel=None))]
pub fn atr_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let kernel_enum = match kernel {
        Some(k) => validate_kernel(k)?,
        None => Kernel::Auto,
    };

    // Get slices from numpy arrays - zero copy
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    // Validate inputs using prepare function
    let (high_valid, low_valid, close_valid, length_valid) = 
        atr_prepare(high_slice, low_slice, close_slice, length)?;

    let len = close_valid.len();

    // Pre-allocate output array
    let output = unsafe { PyArray1::new(py, [len], false) };

    // Get mutable slice for writing
    let out_slice = unsafe { output.as_slice_mut()? };

    // Choose kernel
    let chosen = match kernel_enum {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // SAFETY: We have validated that inputs are the same length
    // and output buffer is allocated with correct size
    py.allow_threads(|| unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                atr_scalar(high_valid, low_valid, close_valid, length_valid, out_slice)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                atr_avx2(high_valid, low_valid, close_valid, length_valid, out_slice)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                atr_avx512(high_valid, low_valid, close_valid, length_valid, out_slice)
            }
            _ => unreachable!(),
        }
    });

    Ok(output.into())
}

#[cfg(feature = "python")]
#[pyclass(name = "AtrStream")]
pub struct AtrStreamPy {
    stream: AtrStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AtrStreamPy {
    #[new]
    pub fn new(length: Option<usize>) -> PyResult<Self> {
        let params = AtrParams { length };
        let stream = AtrStream::try_new(params)?;
        Ok(Self { stream })
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "atr_batch")]
#[pyo3(signature = (high, low, close, length_range, kernel=None))]
pub fn atr_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<&'py PyDict> {
    let kernel_enum = match kernel {
        Some(k) => validate_kernel(k, true)?,  // true for batch operation
        None => Kernel::Auto,
    };

    let batch_kernel = match kernel_enum {
        Kernel::Auto => detect_best_batch_kernel(),
        Kernel::Scalar => Kernel::ScalarBatch,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => Kernel::Avx2Batch,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => Kernel::Avx512Batch,
        k if k.is_batch() => k,
        _ => return Err(PyValueError::new_err("Invalid kernel for batch operation")),
    };

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    let range = AtrBatchRange {
        length: length_range,
    };

    // First expand grid to know dimensions
    let combos = expand_grid(&range);
    let rows = combos.len();
    let cols = close_slice.len();

    // Pre-allocate 2D NumPy array
    let values_2d = unsafe { numpy::PyArray2::new(py, [rows, cols], false) };
    
    // Ensure the array is contiguous (should always be true for new arrays)
    debug_assert!(values_2d.is_contiguous());
    
    // Get a pointer to the raw data - need to cast from *mut c_void to *mut f64
    let raw_ptr = values_2d.data() as *mut f64;

    // Clone combos to avoid recomputing expand_grid
    let combos_cloned = combos.clone();

    // Compute directly into the pre-allocated array
    py.allow_threads(|| {
        // Create a mutable slice view of the entire 2D array as flat
        let output_slice = unsafe {
            std::slice::from_raw_parts_mut(raw_ptr, rows * cols)
        };
        
        // Call the inner batch function that works with a flat output slice
        let simd = match batch_kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };
        
        // Process each row
        for (row_idx, combo) in combos_cloned.iter().enumerate() {
            let length = combo.length.unwrap_or(14);
            let row_start = row_idx * cols;
            let row_slice = &mut output_slice[row_start..row_start + cols];
            
            // SAFETY: We have validated inputs and allocated correct output size
            unsafe {
                match simd {
                    Kernel::Scalar => atr_scalar(high_slice, low_slice, close_slice, length, row_slice),
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx2 => atr_avx2(high_slice, low_slice, close_slice, length, row_slice),
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    Kernel::Avx512 => atr_avx512(high_slice, low_slice, close_slice, length, row_slice),
                    _ => unreachable!(),
                }
            }
        }
    });

    let dict = PyDict::new(py);
    dict.set_item("values", values_2d)?;

    // Extract lengths from combos
    let lengths: Vec<usize> = combos.iter()
        .map(|p| p.length.unwrap_or(14))
        .collect();
    dict.set_item("lengths", PyList::new(py, &lengths)?)?;

    Ok(dict)
}

// ============= WASM Bindings =============

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "atr")]
pub fn atr_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length: usize,
) -> Result<Vec<f64>, JsError> {
    let params = AtrParams {
        length: Some(length),
    };
    let input = AtrInput::from_slices(high, low, close, params);
    let output = atr(&input).map_err(|e| JsError::new(&e.to_string()))?;
    Ok(output.values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "atrBatch")]
pub fn atr_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<Vec<f64>, JsError> {
    let range = AtrBatchRange {
        length: (length_start, length_end, length_step),
    };
    let output = atr_batch_with_kernel(high, low, close, &range, Kernel::Auto)
        .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(output.values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "atrBatchMetadata")]
pub fn atr_batch_metadata_js(
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Vec<f64> {
    let range = AtrBatchRange {
        length: (length_start, length_end, length_step),
    };
    let combos = expand_grid(&range);
    
    // Return flat array of lengths
    combos.iter()
        .map(|p| p.length.unwrap_or(14) as f64)
        .collect()
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "atrBatch", skip_jsdoc)]
pub fn atr_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsError> {
    #[derive(Deserialize)]
    struct BatchConfig {
        length_range: [usize; 3],
    }

    let config: BatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsError::new(&e.to_string()))?;

    let range = AtrBatchRange {
        length: (config.length_range[0], config.length_range[1], config.length_range[2]),
    };

    let output = atr_batch_with_kernel(high, low, close, &range, Kernel::Auto)
        .map_err(|e| JsError::new(&e.to_string()))?;

    #[derive(Serialize)]
    struct BatchResult {
        values: Vec<f64>,
        combos: Vec<AtrParams>,
        rows: usize,
        cols: usize,
    }

    let result = BatchResult {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsError::new(&e.to_string()))
}

// Python module registration for custom exceptions
// This should be called from the main module initialization
#[cfg(feature = "python")]
pub fn register_atr_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("InvalidLengthError", m.py().get_type::<InvalidLengthError>())?;
    m.add("InconsistentSliceLengthsError", m.py().get_type::<InconsistentSliceLengthsError>())?;
    m.add("NoCandlesAvailableError", m.py().get_type::<NoCandlesAvailableError>())?;
    m.add("NotEnoughDataError", m.py().get_type::<NotEnoughDataError>())?;
    Ok(())
}
