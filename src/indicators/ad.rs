//! # Chaikin Accumulation/Distribution (AD)
//!
//! Volume-based cumulative money flow indicator using high, low, close, and volume.
//!
//! ## Parameters
//! - No adjustable parameters beyond input data.
//!
//! ## Errors
//! - **CandleFieldError**: ad: Failure retrieving required candle fields.
//! - **DataLengthMismatch**: ad: Provided slices are not the same length.
//! - **NotEnoughData**: ad: Data length is zero.
//!
//! ## Returns
//! - **Ok(AdOutput)** on success, with AD values.
//! - **Err(AdError)** otherwise.

use crate::utilities::data_loader::{Candles, source_type};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use aligned_vec::{AVec, CACHELINE_ALIGN};
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AdData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone, Default)]
pub struct AdParams {}

#[derive(Debug, Clone)]
pub struct AdInput<'a> {
    pub data: AdData<'a>,
    pub params: AdParams,
}

impl<'a> AdInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AdParams) -> Self {
        Self { data: AdData::Candles { candles }, params }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64], low: &'a [f64], close: &'a [f64], volume: &'a [f64], params: AdParams
    ) -> Self {
        Self { data: AdData::Slices { high, low, close, volume }, params }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, AdParams::default())
    }
}

#[derive(Debug, Clone)]
pub struct AdOutput {
    pub values: Vec<f64>,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct AdBuilder {
    kernel: Kernel,
}

impl AdBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self { kernel: Kernel::Auto } }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AdOutput, AdError> {
        let input = AdInput::from_candles(c, AdParams::default());
        ad_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64], low: &[f64], close: &[f64], volume: &[f64]
    ) -> Result<AdOutput, AdError> {
        let input = AdInput::from_slices(high, low, close, volume, AdParams::default());
        ad_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AdStream, AdError> {
        AdStream::try_new()
    }
}

#[derive(Debug, Error)]
pub enum AdError {
    #[error(transparent)]
    CandleFieldError(#[from] Box<dyn std::error::Error>),
    #[error("ad: Data length mismatch for AD calculation: high={high_len}, low={low_len}, close={close_len}, volume={volume_len}")]
    DataLengthMismatch { high_len: usize, low_len: usize, close_len: usize, volume_len: usize },
    #[error("ad: Not enough data points to calculate AD. Length={len}")]
    NotEnoughData { len: usize },
}

#[inline]
pub fn ad(input: &AdInput) -> Result<AdOutput, AdError> {
    ad_with_kernel(input, Kernel::Auto)
}

pub fn ad_with_kernel(input: &AdInput, kernel: Kernel) -> Result<AdOutput, AdError> {
    let (high, low, close, volume): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        AdData::Candles { candles } => {
            let high = candles.select_candle_field("high")?;
            let low = candles.select_candle_field("low")?;
            let close = candles.select_candle_field("close")?;
            let volume = candles.select_candle_field("volume")?;
            (high, low, close, volume)
        }
        AdData::Slices { high, low, close, volume } => (*high, *low, *close, *volume),
    };

    if high.len() != low.len() || high.len() != close.len() || high.len() != volume.len() {
        return Err(AdError::DataLengthMismatch {
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }

    let size = high.len();
    if size < 1 {
        return Err(AdError::NotEnoughData { len: size });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    let mut out = vec![0.0; size];

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                ad_scalar(high, low, close, volume, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                ad_avx2(high, low, close, volume, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                ad_avx512(high, low, close, volume, &mut out)
            }
            _ => unreachable!(),
        }
    }
    Ok(AdOutput { values: out })
}

#[inline]
pub fn ad_scalar(
    high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64],
) {
    let size = high.len();
    let mut sum = 0.0;
    for i in 0..size {
        let hl = high[i] - low[i];
        if hl != 0.0 {
            let mfm = ((close[i] - low[i]) - (high[i] - close[i])) / hl;
            let mfv = mfm * volume[i];
            sum += mfv;
        }
        out[i] = sum;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx2(
    high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64],
) {
    ad_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx512(
    high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64],
) {
    ad_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx512_short(
    high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64],
) {
    ad_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ad_avx512_long(
    high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64],
) {
    ad_scalar(high, low, close, volume, out)
}

#[inline]
pub fn ad_batch_with_kernel(
    data: &AdBatchInput,
    k: Kernel,
) -> Result<AdBatchOutput, AdError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch,
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    ad_batch_par_slice(data, simd)
}

#[derive(Clone, Debug)]
pub struct AdBatchInput<'a> {
    pub highs: &'a [&'a [f64]],
    pub lows: &'a [&'a [f64]],
    pub closes: &'a [&'a [f64]],
    pub volumes: &'a [&'a [f64]],
}

#[derive(Clone, Debug)]
pub struct AdBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
pub fn ad_batch_slice(
    data: &AdBatchInput,
    kern: Kernel,
) -> Result<AdBatchOutput, AdError> {
    ad_batch_inner(data, kern, false)
}

#[inline(always)]
pub fn ad_batch_par_slice(
    data: &AdBatchInput,
    kern: Kernel,
) -> Result<AdBatchOutput, AdError> {
    ad_batch_inner(data, kern, true)
}

fn ad_batch_inner(
    data: &AdBatchInput,
    kern: Kernel,
    parallel: bool,
) -> Result<AdBatchOutput, AdError> {
    let rows = data.highs.len();
    let cols = if rows > 0 { data.highs[0].len() } else { 0 };
    let mut values = vec![0.0; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        ad_row_scalar(
            data.highs[row],
            data.lows[row],
            data.closes[row],
            data.volumes[row],
            out_row,
        );
    };

    if parallel {
        values
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(AdBatchOutput { values, rows, cols })
}

#[inline(always)]
pub unsafe fn ad_row_scalar(
    high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64],
) {
    ad_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ad_row_avx2(
    high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64],
) {
    ad_row_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ad_row_avx512(
    high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64],
) {
    ad_row_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ad_row_avx512_short(
    high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64],
) {
    ad_row_scalar(high, low, close, volume, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ad_row_avx512_long(
    high: &[f64], low: &[f64], close: &[f64], volume: &[f64], out: &mut [f64],
) {
    ad_row_scalar(high, low, close, volume, out)
}

#[derive(Debug, Clone)]
pub struct AdStream {
    sum: f64,
}

impl AdStream {
    pub fn try_new() -> Result<Self, AdError> {
        Ok(Self { sum: 0.0 })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let hl = high - low;
        if hl != 0.0 {
            let mfm = ((close - low) - (high - close)) / hl;
            let mfv = mfm * volume;
            self.sum += mfv;
        }
        self.sum
    }
}

// Batch Builder for parity with Alma
#[derive(Clone, Debug, Default)]
pub struct AdBatchBuilder {
    pub kernel: Kernel,
}

impl AdBatchBuilder {
    pub fn new() -> Self { Self { kernel: Kernel::Auto } }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_slices(
        self,
        highs: &[&[f64]], lows: &[&[f64]], closes: &[&[f64]], volumes: &[&[f64]],
    ) -> Result<AdBatchOutput, AdError> {
        let batch = AdBatchInput { highs, lows, closes, volumes };
        ad_batch_with_kernel(&batch, self.kernel)
    }
}

// Grid expansion for batch (no param sweep in AD, but kept for parity)
#[inline(always)]
fn expand_grid_ad<'a>(
    highs: &'a [&'a [f64]], lows: &'a [&'a [f64]], closes: &'a [&'a [f64]], volumes: &'a [&'a [f64]],
) -> Vec<(&'a [f64], &'a [f64], &'a [f64], &'a [f64])> {
    let mut out = Vec::with_capacity(highs.len());
    for i in 0..highs.len() {
        out.push((highs[i], lows[i], closes[i], volumes[i]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_ad_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = AdParams::default();
        let input = AdInput::from_candles(&candles, default_params);
        let output = ad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ad_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdInput::with_default_candles(&candles);
        let ad_result = ad_with_kernel(&input, kernel)?;
        assert_eq!(ad_result.values.len(), candles.close.len());
        let expected_last_five = [1645918.16, 1645876.11, 1645824.27, 1645828.87, 1645728.78];
        let start = ad_result.values.len() - 5;
        let actual = &ad_result.values[start..];
        for (i, &val) in actual.iter().enumerate() {
            assert!(
                (val - expected_last_five[i]).abs() < 1e-1,
                "[{}] AD mismatch at idx {}: got {}, expected {}",
                test_name, i, val, expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_ad_with_slice_data_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_input = AdInput::with_default_candles(&candles);
        let first_result = ad_with_kernel(&first_input, kernel)?;
        let second_input = AdInput::from_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            &first_result.values,
            AdParams::default(),
        );
        let second_result = ad_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 50..second_result.values.len() {
            assert!(!second_result.values[i].is_nan());
        }
        Ok(())
    }

    fn check_ad_input_with_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdInput::with_default_candles(&candles);
        match input.data {
            AdData::Candles { .. } => {}
            _ => panic!("Expected AdData::Candles variant"),
        }
        Ok(())
    }

    fn check_ad_accuracy_nan_check(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdInput::with_default_candles(&candles);
        let ad_result = ad_with_kernel(&input, kernel)?;
        assert_eq!(ad_result.values.len(), candles.close.len());
        if ad_result.values.len() > 50 {
            for i in 50..ad_result.values.len() {
                assert!(
                    !ad_result.values[i].is_nan(),
                    "[{}] Expected no NaN after index 50, but found NaN at index {}",
                    test_name, i
                );
            }
        }
        Ok(())
    }

    fn check_ad_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AdInput::with_default_candles(&candles);
        let batch = ad_with_kernel(&input, kernel)?.values;
        let mut stream = AdStream::try_new()?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for i in 0..candles.close.len() {
            let val = stream.update(
                candles.high[i],
                candles.low[i],
                candles.close[i],
                candles.volume[i]
            );
            stream_values.push(val);
        }
        assert_eq!(batch.len(), stream_values.len());
        for (b, s) in batch.iter().zip(stream_values.iter()) {
            if b.is_nan() && s.is_nan() { continue; }
            assert!((b - s).abs() < 1e-9, "[{}] AD streaming mismatch", test_name);
        }
        Ok(())
    }

    macro_rules! generate_all_ad_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test] fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(#[test] fn [<$test_fn _avx2_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(#[test] fn [<$test_fn _avx512_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                })*
            }
        }
    }

    generate_all_ad_tests!(
        check_ad_partial_params,
        check_ad_accuracy,
        check_ad_input_with_default_candles,
        check_ad_with_slice_data_reinput,
        check_ad_accuracy_nan_check,
        check_ad_streaming
    );
        fn check_batch_single_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        // Prepare slices
        let highs: Vec<&[f64]> = vec![&candles.high];
        let lows: Vec<&[f64]> = vec![&candles.low];
        let closes: Vec<&[f64]> = vec![&candles.close];
        let volumes: Vec<&[f64]> = vec![&candles.volume];

        // Individual calculation
        let single = ad_with_kernel(&AdInput::from_candles(&candles, AdParams::default()), kernel)?.values;

        // Batch calculation
        let batch = AdBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(&highs, &lows, &closes, &volumes)?;

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, candles.close.len());
        assert_eq!(batch.values.len(), candles.close.len());

        for (i, (a, b)) in single.iter().zip(&batch.values).enumerate() {
            assert!(
                (a - b).abs() < 1e-8,
                "[{}] AD batch single row mismatch at {}: {} vs {}",
                test, i, a, b
            );
        }
        Ok(())
    }

    fn check_batch_multi_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        // Multi-row: repeat candle data 3 times as separate rows (for test)
        let highs: Vec<&[f64]> = vec![&candles.high, &candles.high, &candles.high];
        let lows: Vec<&[f64]> = vec![&candles.low, &candles.low, &candles.low];
        let closes: Vec<&[f64]> = vec![&candles.close, &candles.close, &candles.close];
        let volumes: Vec<&[f64]> = vec![&candles.volume, &candles.volume, &candles.volume];

        // Individual calculation (should match every row)
        let single = ad_with_kernel(&AdInput::from_candles(&candles, AdParams::default()), kernel)?.values;

        let batch = AdBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(&highs, &lows, &closes, &volumes)?;

        assert_eq!(batch.rows, 3);
        assert_eq!(batch.cols, candles.close.len());
        assert_eq!(batch.values.len(), 3 * candles.close.len());

        for row in 0..3 {
            let row_slice = &batch.values[row * batch.cols..(row + 1) * batch.cols];
            for (i, (a, b)) in single.iter().zip(row_slice.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-8,
                    "[{}] AD batch multi row mismatch row {} idx {}: {} vs {}",
                    test, row, i, a, b
                );
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

    gen_batch_tests!(check_batch_single_row);
    gen_batch_tests!(check_batch_multi_row);

}
