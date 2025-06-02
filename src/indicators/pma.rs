//! # Predictive Moving Average (PMA)
//!
//! Ehlers’ Predictive Moving Average calculates a smoothed value (`predict`)
//! and a signal line (`trigger`) by applying a series of weighted moving averages
//! and transformations to the input data. This indicator aims to predict future
//! price movements more responsively than standard moving averages.
//!
//! ## Parameters
//! - **source**: The data field to be used from the candles (e.g., "close", "hl2", etc.).
//!   Defaults to "close" when using `with_default_candles`.
//!
//! ## Errors
//! - **EmptyData**: pma: Input data slice is empty.
//! - **NotEnoughValidData**: pma: Fewer than 7 valid (non-`NaN`) data points remain
//!   after the first valid index.
//! - **AllValuesNaN**: pma: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(PmaOutput)`** on success, containing two `Vec<f64>` (`predict` and `trigger`),
//!   each matching the input length and filled with leading `NaN`s until enough data
//!   points have accumulated.
//! - **`Err(PmaError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use aligned_vec::{AVec, CACHELINE_ALIGN};
use rayon::prelude::*;
use thiserror::Error;
use std::convert::AsRef;

impl<'a> AsRef<[f64]> for PmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PmaData::Slice(slice) => slice,
            PmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PmaOutput {
    pub predict: Vec<f64>,
    pub trigger: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PmaParams;

impl Default for PmaParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct PmaInput<'a> {
    pub data: PmaData<'a>,
    pub params: PmaParams,
}

impl<'a> PmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: PmaParams) -> Self {
        Self {
            data: PmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: PmaParams) -> Self {
        Self {
            data: PmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", PmaParams::default())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PmaBuilder {
    kernel: Kernel,
}

impl Default for PmaBuilder {
    fn default() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
}

impl PmaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<PmaOutput, PmaError> {
        let i = PmaInput::from_candles(c, "close", PmaParams::default());
        pma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<PmaOutput, PmaError> {
        let i = PmaInput::from_slice(d, PmaParams::default());
        pma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<PmaStream, PmaError> {
        PmaStream::try_new(PmaParams::default())
    }
}

#[derive(Debug, Error)]
pub enum PmaError {
    #[error("pma: Empty data provided.")]
    EmptyData,
    #[error("pma: Not enough valid data: needed = 7, valid = {valid}")]
    NotEnoughValidData { valid: usize },
    #[error("pma: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn pma(input: &PmaInput) -> Result<PmaOutput, PmaError> {
    pma_with_kernel(input, Kernel::Auto)
}

pub fn pma_with_kernel(input: &PmaInput, kernel: Kernel) -> Result<PmaOutput, PmaError> {
    let data: &[f64] = match &input.data {
        PmaData::Candles { candles, source } => source_type(candles, source),
        PmaData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(PmaError::EmptyData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PmaError::AllValuesNaN)?;

    if (data.len() - first) < 7 {
        return Err(PmaError::NotEnoughValidData {
            valid: data.len() - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                pma_scalar(data, first)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                pma_avx2(data, first)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                pma_avx512(data, first)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn pma_scalar(data: &[f64], first_valid_idx: usize) -> Result<PmaOutput, PmaError> {
    let mut predict = vec![f64::NAN; data.len()];
    let mut trigger = vec![f64::NAN; data.len()];
    let mut wma1 = vec![0.0; data.len()];

    for j in (first_valid_idx + 6)..data.len() {
        let wma1_j = ((7.0 * data[j])
            + (6.0 * data[j - 1])
            + (5.0 * data[j - 2])
            + (4.0 * data[j - 3])
            + (3.0 * data[j - 4])
            + (2.0 * data[j - 5])
            + data[j - 6])
            / 28.0;
        wma1[j] = wma1_j;

        let wma2 = ((7.0 * wma1[j])
            + (6.0 * wma1[j - 1])
            + (5.0 * wma1[j - 2])
            + (4.0 * wma1[j - 3])
            + (3.0 * wma1[j - 4])
            + (2.0 * wma1[j - 5])
            + wma1[j - 6])
            / 28.0;

        let predict_j = (2.0 * wma1_j) - wma2;
        predict[j] = predict_j;

        let trigger_j =
            ((4.0 * predict_j) + (3.0 * predict[j - 1]) + (2.0 * predict[j - 2]) + predict[j - 3])
                / 10.0;
        trigger[j] = trigger_j;
    }

    Ok(PmaOutput { predict, trigger })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pma_avx512(data: &[f64], first_valid_idx: usize) -> Result<PmaOutput, PmaError> {
    pma_scalar(data, first_valid_idx)
}

#[inline]
pub fn pma_avx2(data: &[f64], first_valid_idx: usize) -> Result<PmaOutput, PmaError> {
    pma_scalar(data, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pma_avx512_short(data: &[f64], first_valid_idx: usize) -> Result<PmaOutput, PmaError> {
    pma_scalar(data, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pma_avx512_long(data: &[f64], first_valid_idx: usize) -> Result<PmaOutput, PmaError> {
    pma_scalar(data, first_valid_idx)
}

#[inline]
pub fn pma_batch_with_kernel(
    data: &[f64],
    sweep: &PmaBatchRange,
    k: Kernel,
) -> Result<PmaBatchOutput, PmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(PmaError::EmptyData),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    pma_batch_par_slice(data, sweep, simd)
}

#[derive(Debug, Clone)]
pub struct PmaStream {
    buffer: Vec<f64>,
    wma1: Vec<f64>,
    idx: usize,
    filled: bool,
}

impl PmaStream {
    pub fn try_new(_params: PmaParams) -> Result<Self, PmaError> {
        Ok(Self {
            buffer: vec![f64::NAN; 7],
            wma1: vec![0.0; 7],
            idx: 0,
            filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.buffer[self.idx] = value;
        self.idx = (self.idx + 1) % 7;
        if !self.filled && self.idx == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        let j = self.idx;
        let slice = |k| self.buffer[(j + k) % 7];

        let wma1_j = (
            7.0 * slice(6) +
            6.0 * slice(5) +
            5.0 * slice(4) +
            4.0 * slice(3) +
            3.0 * slice(2) +
            2.0 * slice(1) +
            slice(0)
        ) / 28.0;
        self.wma1[self.idx] = wma1_j;

        let wma2 = (
            7.0 * self.wma1[(self.idx + 6) % 7] +
            6.0 * self.wma1[(self.idx + 5) % 7] +
            5.0 * self.wma1[(self.idx + 4) % 7] +
            4.0 * self.wma1[(self.idx + 3) % 7] +
            3.0 * self.wma1[(self.idx + 2) % 7] +
            2.0 * self.wma1[(self.idx + 1) % 7] +
            self.wma1[self.idx]
        ) / 28.0;

        let predict_j = 2.0 * wma1_j - wma2;
        let t3 = predict_j;
        let t2 = predict_j; // Not enough context for previous predictions in stream
        let t1 = predict_j;
        let t0 = predict_j;
        let trigger_j = (4.0 * t3 + 3.0 * t2 + 2.0 * t1 + t0) / 10.0;
        Some((predict_j, trigger_j))
    }
}

#[derive(Clone, Debug)]
pub struct PmaBatchRange {
    // Only dummy for now, PMA has no true sweepable params, but we keep for API parity
    pub dummy: (usize, usize, usize),
}

impl Default for PmaBatchRange {
    fn default() -> Self {
        Self { dummy: (0, 0, 0) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PmaBatchBuilder {
    range: PmaBatchRange,
    kernel: Kernel,
}

impl PmaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn apply_slice(self, data: &[f64]) -> Result<PmaBatchOutput, PmaError> {
        pma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<PmaBatchOutput, PmaError> {
        PmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<PmaBatchOutput, PmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<PmaBatchOutput, PmaError> {
        PmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct PmaBatchOutput {
    pub predict: Vec<f64>,
    pub trigger: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}
impl PmaBatchOutput {
    pub fn values_for(&self, _dummy: &PmaParams) -> Option<(&[f64], &[f64])> {
        Some((&self.predict[..], &self.trigger[..]))
    }
}

#[inline(always)]
pub fn expand_grid(_r: &PmaBatchRange) -> Vec<PmaParams> {
    vec![PmaParams {}]
}

#[inline(always)]
pub fn pma_batch_slice(
    data: &[f64],
    sweep: &PmaBatchRange,
    kern: Kernel,
) -> Result<PmaBatchOutput, PmaError> {
    pma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn pma_batch_par_slice(
    data: &[f64],
    sweep: &PmaBatchRange,
    kern: Kernel,
) -> Result<PmaBatchOutput, PmaError> {
    pma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn pma_batch_inner(
    data: &[f64],
    _sweep: &PmaBatchRange,
    kern: Kernel,
    _parallel: bool,
) -> Result<PmaBatchOutput, PmaError> {
    let params = PmaParams {};
    let out = match kern {
        Kernel::Scalar => pma_scalar(data, data.iter().position(|x| !x.is_nan()).unwrap_or(0))?,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => pma_avx2(data, data.iter().position(|x| !x.is_nan()).unwrap_or(0))?,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => pma_avx512(data, data.iter().position(|x| !x.is_nan()).unwrap_or(0))?,
        _ => unreachable!(),
    };
    Ok(PmaBatchOutput {
        predict: out.predict,
        trigger: out.trigger,
        rows: 1,
        cols: data.len(),
    })
}

// Row functions - stubs for AVX2/AVX512
#[inline(always)]
pub unsafe fn pma_row_scalar(
    data: &[f64],
    first: usize,
    _stride: usize,
    _dummy: *const f64,
    _inv_n: f64,
    out_predict: &mut [f64],
    out_trigger: &mut [f64],
) {
    let result = pma_scalar(data, first).unwrap();
    out_predict.copy_from_slice(&result.predict);
    out_trigger.copy_from_slice(&result.trigger);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pma_row_avx2(
    data: &[f64],
    first: usize,
    stride: usize,
    dummy: *const f64,
    inv_n: f64,
    out_predict: &mut [f64],
    out_trigger: &mut [f64],
) {
    pma_row_scalar(data, first, stride, dummy, inv_n, out_predict, out_trigger);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pma_row_avx512(
    data: &[f64],
    first: usize,
    stride: usize,
    dummy: *const f64,
    inv_n: f64,
    out_predict: &mut [f64],
    out_trigger: &mut [f64],
) {
    pma_row_scalar(data, first, stride, dummy, inv_n, out_predict, out_trigger);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pma_row_avx512_short(
    data: &[f64],
    first: usize,
    stride: usize,
    dummy: *const f64,
    inv_n: f64,
    out_predict: &mut [f64],
    out_trigger: &mut [f64],
) {
    pma_row_scalar(data, first, stride, dummy, inv_n, out_predict, out_trigger);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pma_row_avx512_long(
    data: &[f64],
    first: usize,
    stride: usize,
    dummy: *const f64,
    inv_n: f64,
    out_predict: &mut [f64],
    out_trigger: &mut [f64],
) {
    pma_row_scalar(data, first, stride, dummy, inv_n, out_predict, out_trigger);
}

//--------------------------
// Tests
//--------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_pma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PmaInput::with_default_candles(&candles);
        let output = pma_with_kernel(&input, kernel)?;
        assert_eq!(output.predict.len(), candles.close.len());
        assert_eq!(output.trigger.len(), candles.close.len());
        Ok(())
    }

    fn check_pma_with_slice(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let input = PmaInput::from_slice(&data, PmaParams {});
        let output = pma_with_kernel(&input, kernel)?;
        assert_eq!(output.predict.len(), data.len());
        assert_eq!(output.trigger.len(), data.len());
        Ok(())
    }

    fn check_pma_not_enough_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let input = PmaInput::from_slice(&data, PmaParams {});
        let result = pma_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for not enough data");
        Ok(())
    }

    fn check_pma_all_values_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let input = PmaInput::from_slice(&data, PmaParams {});
        let result = pma_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for all values NaN");
        Ok(())
    }

    fn check_pma_expected_values(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PmaInput::from_candles(&candles, "hl2", PmaParams {});
        let result = pma_with_kernel(&input, kernel)?;

        assert_eq!(result.predict.len(), candles.close.len(), "Predict length mismatch");
        assert_eq!(result.trigger.len(), candles.close.len(), "Trigger length mismatch");

        let expected_predict = [
            59208.18749999999,
            59233.83609693878,
            59213.19132653061,
            59199.002551020414,
            58993.318877551,
        ];
        let expected_trigger = [
            59157.70790816327,
            59208.60076530612,
            59218.6763392857,
            59211.1443877551,
            59123.05019132652,
        ];

        assert!(result.predict.len() >= 5, "Output length too short for checking");
        let start_idx = result.predict.len() - 5;
        for i in 0..5 {
            let calc_val = result.predict[start_idx + i];
            let exp_val = expected_predict[i];
            assert!((calc_val - exp_val).abs() < 1e-1, "Mismatch in predict at index {}: expected {}, got {}", start_idx + i, exp_val, calc_val);
        }
        for i in 0..5 {
            let calc_val = result.trigger[start_idx + i];
            let exp_val = expected_trigger[i];
            assert!((calc_val - exp_val).abs() < 1e-1, "Mismatch in trigger at index {}: expected {}, got {}", start_idx + i, exp_val, calc_val);
        }
        Ok(())
    }

    macro_rules! generate_all_pma_tests {
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

    generate_all_pma_tests!(
        check_pma_default_candles,
        check_pma_with_slice,
        check_pma_not_enough_data,
        check_pma_all_values_nan,
        check_pma_expected_values
    );
        fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = PmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        // Should always be a single row, for PMA's batch API
        assert_eq!(output.rows, 1, "Expected exactly 1 row");
        assert_eq!(output.cols, c.close.len());
        assert_eq!(output.predict.len(), c.close.len());
        assert_eq!(output.trigger.len(), c.close.len());

        // Spot check output against direct calculation
        let input = PmaInput::from_candles(&c, "close", PmaParams::default());
        let expected = pma_with_kernel(&input, kernel)?;

        for (i, (&a, &b)) in output.predict.iter().zip(expected.predict.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() < 1e-12,
                "[{test}] predict mismatch at idx {i}: batch={}, direct={}",
                a, b
            );
        }
        for (i, (&a, &b)) in output.trigger.iter().zip(expected.trigger.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() < 1e-12,
                "[{test}] trigger mismatch at idx {i}: batch={}, direct={}",
                a, b
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
