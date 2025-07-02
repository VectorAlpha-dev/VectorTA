//! # Balance of Power (BOP)
//!
//! (Close - Open) / (High - Low)
//!
//! If (High - Low) <= 0.0, output is 0.0.
//!
//! ## Parameters
//! None currently required; see `BopParams` for future extensibility.
//!
//! ## Errors
//! - **EmptyData**: bop: No data provided.
//! - **InconsistentLengths**: bop: Input arrays have different lengths.
//!
//! ## Returns
//! - **`Ok(BopOutput)`** on success, containing a `Vec<f64>` with the BOP values.
//! - **`Err(BopError)`** otherwise.

use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum BopData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BopParams {}

#[derive(Debug, Clone)]
pub struct BopInput<'a> {
    pub data: BopData<'a>,
    pub params: BopParams,
}

impl<'a> BopInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: BopParams) -> Self {
        Self {
            data: BopData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: BopParams,
    ) -> Self {
        Self {
            data: BopData::Slices {
                open,
                high,
                low,
                close,
            },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: BopData::Candles { candles },
            params: BopParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BopOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum BopError {
    #[error("bop: Data is empty.")]
    EmptyData,
    #[error("bop: Inconsistent lengths.")]
    InconsistentLengths,
    #[error("bop: Candle field error: {0}")]
    CandleFieldError(String),
}

#[derive(Copy, Clone, Debug)]
pub struct BopBuilder {
    kernel: Kernel,
}

impl Default for BopBuilder {
    fn default() -> Self {
        Self { kernel: Kernel::Auto }
    }
}

impl BopBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<BopOutput, BopError> {
        let i = BopInput::from_candles(c, BopParams::default());
        bop_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<BopOutput, BopError> {
        let i = BopInput::from_slices(open, high, low, close, BopParams::default());
        bop_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<BopStream, BopError> {
        BopStream::try_new()
    }
}

#[inline]
pub fn bop(input: &BopInput) -> Result<BopOutput, BopError> {
    bop_with_kernel(input, Kernel::Auto)
}

pub fn bop_with_kernel(input: &BopInput, kernel: Kernel) -> Result<BopOutput, BopError> {
    let (open, high, low, close): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        BopData::Candles { candles } => {
            let open = candles.select_candle_field("open").map_err(|e| BopError::CandleFieldError(e.to_string()))?;
            let high = candles.select_candle_field("high").map_err(|e| BopError::CandleFieldError(e.to_string()))?;
            let low  = candles.select_candle_field("low").map_err(|e| BopError::CandleFieldError(e.to_string()))?;
            let close= candles.select_candle_field("close").map_err(|e| BopError::CandleFieldError(e.to_string()))?;
            (open, high, low, close)
        }
        BopData::Slices { open, high, low, close } => (open, high, low, close),
    };

    let len = open.len();
    if len == 0 {
        return Err(BopError::EmptyData);
    }
    if len != high.len() || len != low.len() || len != close.len() {
        return Err(BopError::InconsistentLengths);
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    let mut out = vec![0.0; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                bop_scalar(open, high, low, close, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                bop_avx2(open, high, low, close, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                bop_avx512(open, high, low, close, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(BopOutput { values: out })
}

#[inline]
pub unsafe fn bop_scalar(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out: &mut [f64],
) {
    for i in 0..open.len() {
        let denom = high[i] - low[i];
        out[i] = if denom <= 0.0 { 0.0 } else { (close[i] - open[i]) / denom };
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bop_avx2(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out: &mut [f64],
) {
    bop_scalar(open, high, low, close, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bop_avx512(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out: &mut [f64],
) {
    bop_scalar(open, high, low, close, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bop_avx512_short(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out: &mut [f64],
) {
    bop_scalar(open, high, low, close, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn bop_avx512_long(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out: &mut [f64],
) {
    bop_scalar(open, high, low, close, out)
}

#[inline]
pub fn bop_row_scalar(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out: &mut [f64],
) {
    unsafe { bop_scalar(open, high, low, close, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn bop_row_avx2(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out: &mut [f64],
) {
    unsafe { bop_avx2(open, high, low, close, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn bop_row_avx512(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out: &mut [f64],
) {
    unsafe { bop_avx512(open, high, low, close, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn bop_row_avx512_short(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out: &mut [f64],
) {
    unsafe { bop_avx512_short(open, high, low, close, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn bop_row_avx512_long(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    out: &mut [f64],
) {
    unsafe { bop_avx512_long(open, high, low, close, out) }
}

// ---- Batch/Streaming structs and functions ----

#[derive(Clone, Debug)]
pub struct BopStream {
    pub last: Option<f64>,
}

impl BopStream {
    pub fn try_new() -> Result<Self, BopError> {
        Ok(Self { last: None })
    }
    #[inline(always)]
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> f64 {
        let denom = high - low;
        let val = if denom <= 0.0 { 0.0 } else { (close - open) / denom };
        self.last = Some(val);
        val
    }
}

// ---- Batch processing API ----

#[derive(Clone, Debug)]
pub struct BopBatchRange {
    pub dummy: (u8, u8, u8),
}

impl Default for BopBatchRange {
    fn default() -> Self {
        Self { dummy: (0, 0, 0) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BopBatchBuilder {
    range: BopBatchRange,
    kernel: Kernel,
}

impl BopBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<BopBatchOutput, BopError> {
        bop_batch_with_kernel(open, high, low, close, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<BopBatchOutput, BopError> {
        let open = source_type(c, "open");
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(open, high, low, close)
    }
    pub fn with_default_candles(c: &Candles) -> Result<BopBatchOutput, BopError> {
        BopBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct BopBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

pub fn bop_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kernel: Kernel,
) -> Result<BopBatchOutput, BopError> {
    let len = open.len();
    if len == 0 || high.len() != len || low.len() != len || close.len() != len {
        return Err(BopError::InconsistentLengths);
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let mut values = vec![0.0; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => bop_scalar(open, high, low, close, &mut values),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => bop_avx2(open, high, low, close, &mut values),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => bop_avx512(open, high, low, close, &mut values),
            _ => unreachable!(),
        }
    }
    Ok(BopBatchOutput {
        values,
        rows: 1,
        cols: len,
    })
}

#[inline(always)]
pub fn bop_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
) -> Result<BopBatchOutput, BopError> {
    bop_batch_with_kernel(open, high, low, close, kern)
}

#[inline(always)]
pub fn bop_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
) -> Result<BopBatchOutput, BopError> {
    // BOP is cheap, so just run the regular batch; use rayon for real batch ops if needed.
    bop_batch_with_kernel(open, high, low, close, kern)
}

#[inline(always)]
fn expand_grid(_r: &BopBatchRange) -> Vec<BopParams> {
    vec![BopParams {}]
}

// ---- Unit tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_bop_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = BopInput::with_default_candles(&candles);
        let output = bop_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_bop_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = BopInput::with_default_candles(&candles);
        let bop_result = bop_with_kernel(&input, kernel)?;

        let expected_last_five = [
            0.045454545454545456,
            -0.32398753894080995,
            -0.3844086021505376,
            0.3547400611620795,
            -0.5336179295624333,
        ];
        let start_index = bop_result.values.len().saturating_sub(5);
        let result_last_five = &bop_result.values[start_index..];
        for (i, &v) in result_last_five.iter().enumerate() {
            assert!(
                (v - expected_last_five[i]).abs() < 1e-10,
                "[{}] BOP mismatch at idx {}: got {}, expected {}",
                test_name, i, v, expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_bop_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = BopInput::with_default_candles(&candles);
        match input.data {
            BopData::Candles { .. } => {},
            _ => panic!("Expected BopData::Candles"),
        }
        let output = bop_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_bop_with_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = BopParams::default();
        let input = BopInput::from_slices(&empty, &empty, &empty, &empty, params);
        let result = bop_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] Expected an error for empty data", test_name);
        Ok(())
    }

    fn check_bop_with_inconsistent_lengths(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let open = [1.0, 2.0, 3.0];
        let high = [1.5, 2.5];
        let low = [0.8, 1.8, 2.8];
        let close = [1.2, 2.2, 3.2];
        let params = BopParams::default();
        let input = BopInput::from_slices(&open, &high, &low, &close, params);
        let result = bop_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] Expected an error for inconsistent input lengths", test_name);
        Ok(())
    }

    fn check_bop_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let open = [10.0];
        let high = [12.0];
        let low = [9.5];
        let close = [11.0];
        let params = BopParams::default();
        let input = BopInput::from_slices(&open, &high, &low, &close, params);
        let result = bop_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), 1);
        assert!((result.values[0] - 0.4).abs() < 1e-10);
        Ok(())
    }

    fn check_bop_with_slice_data_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_input = BopInput::with_default_candles(&candles);
        let first_result = bop_with_kernel(&first_input, kernel)?;

        let dummy = vec![0.0; first_result.values.len()];
        let second_input = BopInput::from_slices(&dummy, &dummy, &dummy, &first_result.values, BopParams::default());
        let second_result = bop_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for (i, &val) in second_result.values.iter().enumerate() {
            assert!(
                (val - 0.0).abs() < f64::EPSILON,
                "[{}] Expected BOP=0.0 for dummy data at idx {}, got {}", test_name, i, val
            );
        }
        Ok(())
    }

    fn check_bop_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = BopInput::with_default_candles(&candles);
        let bop_result = bop_with_kernel(&input, kernel)?;
        if bop_result.values.len() > 240 {
            for i in 240..bop_result.values.len() {
                assert!(
                    !bop_result.values[i].is_nan(),
                    "[{}] Found NaN at idx {}", test_name, i
                );
            }
        }
        Ok(())
    }

    fn check_bop_streaming(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        // streaming BOP is trivial (no state), just check that update matches scalar formula
        let open = [10.0, 5.0, 6.0, 10.0, 11.0];
        let high = [15.0, 6.0, 9.0, 20.0, 13.0];
        let low  = [10.0, 5.0, 4.0, 10.0, 11.0];
        let close= [14.0, 6.0, 7.0, 12.0, 12.0];
        let mut s = BopStream::try_new()?;
        for i in 0..open.len() {
            let val = s.update(open[i], high[i], low[i], close[i]);
            let denom = high[i] - low[i];
            let expected = if denom <= 0.0 { 0.0 } else { (close[i] - open[i]) / denom };
            assert!((val - expected).abs() < 1e-12, "stream mismatch");
        }
        Ok(())
    }

    macro_rules! generate_all_bop_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                })*
            }
        }
    }

    generate_all_bop_tests!(
        check_bop_partial_params,
        check_bop_accuracy,
        check_bop_default_candles,
        check_bop_with_empty_data,
        check_bop_with_inconsistent_lengths,
        check_bop_very_small_dataset,
        check_bop_with_slice_data_reinput,
        check_bop_nan_handling,
        check_bop_streaming
    );
        fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let open = source_type(&c, "open");
        let high = source_type(&c, "high");
        let low = source_type(&c, "low");
        let close = source_type(&c, "close");

        let batch_output = BopBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(open, high, low, close)?;

        assert_eq!(batch_output.cols, c.close.len());
        assert_eq!(batch_output.rows, 1);

        // Confirm that batch output matches scalar indicator
        let input = BopInput::from_slices(open, high, low, close, BopParams::default());
        let scalar = bop_with_kernel(&input, kernel)?;

        for (i, &v) in batch_output.values.iter().enumerate() {
            assert!(
                (v - scalar.values[i]).abs() < 1e-12,
                "[{test}] batch value mismatch at idx {i}: {v} vs {scalar:?}"
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
