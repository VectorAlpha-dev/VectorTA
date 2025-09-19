//! # Negative Volume Index (NVI)
//!
//! Tracks price changes on days when volume decreases from the previous day.
//!
//! ## Parameters
//! - **close**: Close price data
//! - **volume**: Volume data
//!
//! ## Returns
//! - `Vec<f64>` - NVI values starting at 1000, matching input length
//!
//! ## Developer Status
//! **AVX2**: Stub (calls scalar)
//! **AVX512**: Has short/long variants but all stubs
//! **Streaming**: O(1) - Simple state tracking
//! **Memory**: Good - Uses `alloc_with_nan_prefix` and `make_uninit_matrix`

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

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
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum NviData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct NviOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct NviParams; // no params

#[derive(Debug, Clone)]
pub struct NviInput<'a> {
    pub data: NviData<'a>,
    pub params: NviParams,
}

impl<'a> NviInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, close_source: &'a str, params: NviParams) -> Self {
        Self {
            data: NviData::Candles {
                candles,
                close_source,
            },
            params,
        }
    }
    #[inline]
    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: NviParams) -> Self {
        Self {
            data: NviData::Slices { close, volume },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", NviParams)
    }
}

#[derive(Debug, Error)]
pub enum NviError {
    #[error("nvi: Empty data provided.")]
    EmptyData,
    #[error("nvi: All close values are NaN.")]
    AllCloseValuesNaN,
    #[error("nvi: All volume values are NaN.")]
    AllVolumeValuesNaN,
    #[error("nvi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("nvi: Close and volume length mismatch: close={close_len}, volume={volume_len}")]
    MismatchedLength { close_len: usize, volume_len: usize },
    #[error(
        "nvi: Destination length mismatch: dst={dst_len}, close={close_len}, volume={volume_len}"
    )]
    DestinationLengthMismatch {
        dst_len: usize,
        close_len: usize,
        volume_len: usize,
    },
}

#[derive(Copy, Clone, Debug, Default)]
pub struct NviBuilder {
    kernel: Kernel,
}
impl NviBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<NviOutput, NviError> {
        let i = NviInput::with_default_candles(c);
        nvi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, close: &[f64], volume: &[f64]) -> Result<NviOutput, NviError> {
        let i = NviInput::from_slices(close, volume, NviParams);
        nvi_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<NviStream, NviError> {
        NviStream::try_new()
    }
}

#[derive(Debug, Clone)]
pub struct NviStream {
    prev_close: Option<f64>,
    prev_volume: Option<f64>,
    nvi_val: f64,
    started: bool,
}
impl NviStream {
    pub fn try_new() -> Result<Self, NviError> {
        Ok(Self {
            prev_close: None,
            prev_volume: None,
            nvi_val: 1000.0,
            started: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        if !self.started && !close.is_nan() && !volume.is_nan() {
            self.prev_close = Some(close);
            self.prev_volume = Some(volume);
            self.started = true;
            return Some(self.nvi_val);
        }
        if !self.started {
            return None;
        }
        let prev_c = self.prev_close?;
        let prev_v = self.prev_volume?;
        let mut new_nvi = self.nvi_val;
        if volume < prev_v {
            let pct = (close - prev_c) / prev_c;
            new_nvi += new_nvi * pct;
        }
        self.nvi_val = new_nvi;
        self.prev_close = Some(close);
        self.prev_volume = Some(volume);
        Some(self.nvi_val)
    }
}

#[derive(Clone, Debug)]
pub struct NviBatchOutput {
    pub values: Vec<f64>, // flattened 1 × cols
    pub rows: usize,      // 1
    pub cols: usize,      // data length
}

#[inline]
pub fn nvi(input: &NviInput) -> Result<NviOutput, NviError> {
    nvi_with_kernel(input, Kernel::Auto)
}
pub fn nvi_with_kernel(input: &NviInput, kernel: Kernel) -> Result<NviOutput, NviError> {
    let (close, volume): (&[f64], &[f64]) = match &input.data {
        NviData::Candles {
            candles,
            close_source,
        } => {
            let close = source_type(candles, close_source);
            let volume = candles
                .select_candle_field("volume")
                .map_err(|_| NviError::EmptyData)?;
            (close, volume)
        }
        NviData::Slices { close, volume } => (*close, *volume),
    };

    if close.is_empty() || volume.is_empty() {
        return Err(NviError::EmptyData);
    }
    if close.len() != volume.len() {
        return Err(NviError::MismatchedLength {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let first = close
        .iter()
        .zip(volume)
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or_else(|| {
            if close.iter().all(|&c| c.is_nan()) {
                NviError::AllCloseValuesNaN
            } else {
                NviError::AllVolumeValuesNaN
            }
        })?;
    if close.len() - first < 2 {
        return Err(NviError::NotEnoughValidData {
            needed: 2,
            valid: close.len() - first,
        });
    }
    let mut out = alloc_with_nan_prefix(close.len(), first);
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => nvi_scalar(close, volume, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => nvi_avx2(close, volume, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => nvi_avx512(close, volume, first, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(NviOutput { values: out })
}

#[inline]
pub fn nvi_into_slice(
    dst: &mut [f64],
    close: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<(), NviError> {
    if close.is_empty() || volume.is_empty() {
        return Err(NviError::EmptyData);
    }
    if close.len() != volume.len() {
        return Err(NviError::MismatchedLength {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    if dst.len() != close.len() {
        return Err(NviError::DestinationLengthMismatch {
            dst_len: dst.len(),
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }

    let first = close
        .iter()
        .zip(volume)
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or_else(|| {
            if close.iter().all(|&c| c.is_nan()) {
                NviError::AllCloseValuesNaN
            } else {
                NviError::AllVolumeValuesNaN
            }
        })?;

    if close.len() - first < 2 {
        return Err(NviError::NotEnoughValidData {
            needed: 2,
            valid: close.len() - first,
        });
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => nvi_scalar(close, volume, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => nvi_avx2(close, volume, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => nvi_avx512(close, volume, first, dst),
            _ => unreachable!(),
        }
    }

    // Fill warmup period with NaN
    for v in &mut dst[..first] {
        *v = f64::NAN;
    }

    Ok(())
}

pub fn nvi_scalar(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    assert!(
        close.len() == volume.len() && volume.len() == out.len(),
        "Input slices must all have the same length."
    );
    let len = close.len();
    if len == 0 || first_valid >= len {
        return;
    }

    // Start NVI at first valid index
    let mut nvi_val = 1000.0;
    if first_valid < len {
        out[first_valid] = nvi_val;
    }

    if first_valid + 1 >= len {
        return;
    }

    // Track previous day's close & volume for bar-to-bar comparison
    let mut prev_close = close[first_valid];
    let mut prev_volume = volume[first_valid];

    // For each subsequent bar
    for i in (first_valid + 1)..len {
        // 3a) Only update when volume has decreased from the prior bar
        if volume[i] < prev_volume {
            // Percentage change in price from yesterday --> today
            let pct_change = (close[i] - prev_close) / prev_close; // :contentReference[oaicite:4]{index=4}
                                                                   // Apply Fosback formula: NVI_t = NVI_{t-1} + (pct_change)*NVI_{t-1}
            nvi_val += nvi_val * pct_change; // :contentReference[oaicite:5]{index=5}
        }
        // 3b) Otherwise, carry forward the same NVI
        out[i] = nvi_val;

        // 3c) Update “previous bar” references
        prev_close = close[i];
        prev_volume = volume[i];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx2(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    nvi_scalar(close, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx512(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    nvi_scalar(close, volume, first_valid, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx512_short(close: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    nvi_scalar(close, volume, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn nvi_avx512_long(close: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    nvi_scalar(close, volume, first, out)
}

#[inline(always)]
pub fn nvi_batch_with_kernel(
    close: &[f64],
    volume: &[f64],
    k: Kernel,
) -> Result<NviBatchOutput, NviError> {
    if close.is_empty() || volume.is_empty() {
        return Err(NviError::EmptyData);
    }
    if close.len() != volume.len() {
        return Err(NviError::MismatchedLength {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }

    let cols = close.len();
    let first = close
        .iter()
        .zip(volume)
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
        .ok_or_else(|| {
            if close.iter().all(|&c| c.is_nan()) {
                NviError::AllCloseValuesNaN
            } else {
                NviError::AllVolumeValuesNaN
            }
        })?;
    if cols - first < 2 {
        return Err(NviError::NotEnoughValidData {
            needed: 2,
            valid: cols - first,
        });
    }

    // 1×N matrix, prefill warm prefix with NaN without copying the rest.
    let mut buf_mu = make_uninit_matrix(1, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &[first]);

    // Convert to &mut [f64] without copy.
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // Compute into row 0.
    let chosen = match k {
        Kernel::Auto => detect_best_batch_kernel(), // maps to Scalar/Avx2/Avx512 row kernels
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => nvi_row_scalar(close, volume, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => nvi_row_scalar(close, volume, first, out), // stubbed
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => nvi_row_scalar(close, volume, first, out), // stubbed
            _ => unreachable!(),
        }
    }

    // Reclaim Vec<f64> without copy.
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    Ok(NviBatchOutput {
        values,
        rows: 1,
        cols,
    })
}

#[inline(always)]
unsafe fn nvi_row_scalar(close: &[f64], volume: &[f64], first: usize, row_out_flat: &mut [f64]) {
    let len = close.len();
    let out = &mut row_out_flat[..len]; // single row
    let mut nvi_val = 1000.0;
    out[first] = nvi_val;

    let mut prev_close = close[first];
    let mut prev_volume = volume[first];

    for i in (first + 1)..len {
        if volume[i] < prev_volume {
            let pct = (close[i] - prev_close) / prev_close;
            nvi_val += nvi_val * pct;
        }
        out[i] = nvi_val;
        prev_close = close[i];
        prev_volume = volume[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_nvi_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = NviInput::with_default_candles(&candles);
        let output = nvi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_nvi_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = NviInput::with_default_candles(&candles);
        let result = nvi_with_kernel(&input, kernel)?;
        let expected_last_five = [
            154243.6925373456,
            153973.11239019397,
            153973.11239019397,
            154275.63921207888,
            154275.63921207888,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-5,
                "[{}] NVI {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_nvi_empty_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data: [f64; 0] = [];
        let volume_data: [f64; 0] = [];
        let input = NviInput::from_slices(&close_data, &volume_data, NviParams);
        let res = nvi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NVI should fail with empty data",
            test_name
        );
        Ok(())
    }

    fn check_nvi_not_enough_valid_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close_data = [f64::NAN, 100.0];
        let volume_data = [f64::NAN, 120.0];
        let input = NviInput::from_slices(&close_data, &volume_data, NviParams);
        let res = nvi_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] NVI should fail with not enough valid data",
            test_name
        );
        Ok(())
    }

    fn check_nvi_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close = candles.select_candle_field("close")?;
        let volume = candles.select_candle_field("volume")?;
        let input = NviInput::from_slices(close, volume, NviParams);
        let batch_output = nvi_with_kernel(&input, kernel)?.values;
        let mut stream = NviStream::try_new()?;

        // Find first valid index for proper comparison
        let first_valid = close
            .iter()
            .zip(volume.iter())
            .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
            .unwrap_or(0);

        // Use alloc_with_nan_prefix for zero-copy allocation
        let mut stream_values = alloc_with_nan_prefix(close.len(), first_valid);

        for (i, (&c, &v)) in close.iter().zip(volume.iter()).enumerate() {
            if let Some(nvi_val) = stream.update(c, v) {
                stream_values[i] = nvi_val;
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
                "[{}] NVI streaming mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_nvi_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Since NVI has no parameters, we test with different data scenarios
        // Test with default candles data
        let test_scenarios = vec![
            ("default_candles", NviInput::with_default_candles(&candles)),
            (
                "close_source",
                NviInput::from_candles(&candles, "close", NviParams),
            ),
            (
                "high_source",
                NviInput::from_candles(&candles, "high", NviParams),
            ),
            (
                "low_source",
                NviInput::from_candles(&candles, "low", NviParams),
            ),
            (
                "open_source",
                NviInput::from_candles(&candles, "open", NviParams),
            ),
        ];

        for (scenario_idx, (scenario_name, input)) in test_scenarios.iter().enumerate() {
            let output = nvi_with_kernel(input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with scenario: {} (scenario set {})",
                        test_name, val, bits, i, scenario_name, scenario_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with scenario: {} (scenario set {})",
                        test_name, val, bits, i, scenario_name, scenario_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with scenario: {} (scenario set {})",
                        test_name, val, bits, i, scenario_name, scenario_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_nvi_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(test)]
    fn check_nvi_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Since NVI has no parameters, we focus on generating various price and volume scenarios
        let strat = (50usize..=500)
            .prop_flat_map(|len| {
                (
                    // Generate realistic price data
                    prop::collection::vec(
                        prop::strategy::Union::new(vec![
                            (0.001f64..0.1f64).boxed(), // Very small prices
                            (10f64..10000f64).boxed(),  // Normal prices
                            (1e6f64..1e8f64).boxed(),   // Very large prices
                        ])
                        .prop_filter("finite", |x| x.is_finite()),
                        len,
                    ),
                    // Generate realistic volume data
                    prop::collection::vec(
                        prop::strategy::Union::new(vec![
                            (100f64..1000f64).boxed(), // Small volume
                            (1000f64..1e6f64).boxed(), // Normal volume
                            (1e6f64..1e9f64).boxed(),  // Large volume
                        ])
                        .prop_filter("finite", |x| x.is_finite()),
                        len,
                    ),
                    // Scenario selector for different volume patterns
                    0usize..=7,
                )
            })
            .prop_map(|(mut prices, mut volumes, scenario)| {
                // Create different test scenarios
                match scenario {
                    0 => {
                        // Random realistic data (already generated)
                    }
                    1 => {
                        // Constant volume - NVI should never change from 1000.0
                        let const_vol = volumes[0];
                        volumes.iter_mut().for_each(|v| *v = const_vol);
                    }
                    2 => {
                        // Always decreasing volume - NVI should track all price changes
                        volumes.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    }
                    3 => {
                        // Always increasing volume - NVI should stay at 1000.0
                        volumes.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    }
                    4 => {
                        // Alternating volume - predictable pattern
                        for i in 0..volumes.len() {
                            volumes[i] = if i % 2 == 0 { 1000.0 } else { 500.0 };
                        }
                    }
                    5 => {
                        // Constant prices with varying volume
                        let const_price = prices[0];
                        prices.iter_mut().for_each(|p| *p = const_price);
                    }
                    6 => {
                        // Trending prices with random volume
                        let start = prices[0];
                        let trend = 0.01f64; // 1% per bar
                        for i in 0..prices.len() {
                            prices[i] = start * (1.0 + trend).powi(i as i32);
                        }
                    }
                    7 => {
                        // Oscillating prices with decreasing volume trend
                        let base = prices[0];
                        for i in 0..prices.len() {
                            prices[i] = base * (1.0 + 0.1 * ((i as f64 * 0.5).sin()));
                        }
                        // Add decreasing trend to volume
                        for i in 0..volumes.len() {
                            volumes[i] *= (1.0 - (i as f64 / volumes.len() as f64) * 0.5);
                        }
                    }
                    _ => unreachable!(),
                }
                (prices, volumes, scenario)
            });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(close_data, volume_data, scenario)| {
                let input = NviInput::from_slices(&close_data, &volume_data, NviParams);

                // Test with the specified kernel
                let NviOutput { values: out } = nvi_with_kernel(&input, kernel)?;

                // Also get scalar reference for kernel consistency check
                let NviOutput { values: ref_out } = nvi_with_kernel(&input, Kernel::Scalar)?;

                // Find first valid index
                let first_valid = close_data
                    .iter()
                    .zip(volume_data.iter())
                    .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
                    .unwrap_or(close_data.len());

                if first_valid >= close_data.len() {
                    return Ok(()); // No valid data
                }

                // Property 1: NVI should start at 1000.0 at first valid index
                prop_assert!(
                    (out[first_valid] - 1000.0).abs() < 1e-9,
                    "NVI should start at 1000.0, got {} at index {} (scenario {})",
                    out[first_valid],
                    first_valid,
                    scenario
                );

                // Property 2 & 3: NVI only changes when volume decreases
                let mut prev_nvi = 1000.0;
                let mut prev_close = close_data[first_valid];
                let mut prev_volume = volume_data[first_valid];

                for i in (first_valid + 1)..close_data.len() {
                    let curr_close = close_data[i];
                    let curr_volume = volume_data[i];
                    let curr_nvi = out[i];

                    if curr_volume < prev_volume {
                        // Property 2: NVI should change based on price change
                        let expected_pct = (curr_close - prev_close) / prev_close;
                        let expected_nvi = prev_nvi + prev_nvi * expected_pct;

                        prop_assert!(
							(curr_nvi - expected_nvi).abs() < 1e-9 || 
							(curr_nvi - expected_nvi).abs() / expected_nvi.abs() < 1e-9,
							"NVI calculation error at index {} (scenario {}): expected {}, got {}, \
							prev_nvi={}, pct_change={}, volume {} -> {}",
							i, scenario, expected_nvi, curr_nvi, prev_nvi, expected_pct,
							prev_volume, curr_volume
						);
                    } else {
                        // Property 3: NVI should stay constant when volume doesn't decrease
                        prop_assert!(
							(curr_nvi - prev_nvi).abs() < 1e-9,
							"NVI should not change when volume doesn't decrease at index {} (scenario {}): \
							prev_nvi={}, curr_nvi={}, volume {} -> {}",
							i, scenario, prev_nvi, curr_nvi, prev_volume, curr_volume
						);
                    }

                    prev_nvi = curr_nvi;
                    prev_close = curr_close;
                    prev_volume = curr_volume;
                }

                // Property 4: Kernel consistency - all kernels should produce identical results
                for i in first_valid..close_data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "Kernel finite/NaN mismatch at index {} (scenario {}): {} vs {}",
                            i,
                            scenario,
                            y,
                            r
                        );
                    } else {
                        let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "Kernel mismatch at index {} (scenario {}): {} vs {} (ULP={})",
                            i,
                            scenario,
                            y,
                            r,
                            ulp_diff
                        );
                    }
                }

                // Property 5: Special case validations based on scenario
                match scenario {
                    1 => {
                        // Constant volume - NVI should never change from 1000.0
                        for i in (first_valid + 1)..out.len() {
                            prop_assert!(
								(out[i] - 1000.0).abs() < 1e-9,
								"NVI should stay at 1000.0 with constant volume, got {} at index {}",
								out[i], i
							);
                        }
                    }
                    3 => {
                        // Always increasing volume - NVI should stay at 1000.0
                        for i in (first_valid + 1)..out.len() {
                            prop_assert!(
								(out[i] - 1000.0).abs() < 1e-9,
								"NVI should stay at 1000.0 with always increasing volume, got {} at index {}",
								out[i], i
							);
                        }
                    }
                    5 => {
                        // Constant prices - NVI should remain constant (no price change means pct_change = 0)
                        // NVI stays at whatever value it has reached based on volume changes
                        if first_valid + 1 < out.len() {
                            let mut expected_nvi = out[first_valid]; // Start with initial NVI value (1000.0)
                            for i in (first_valid + 1)..out.len() {
                                // With constant prices, pct_change is always 0, so NVI doesn't change
                                // regardless of volume changes
                                prop_assert!(
									(out[i] - expected_nvi).abs() < 1e-9,
									"NVI should stay constant at {} with constant prices, got {} at index {}",
									expected_nvi, out[i], i
								);
                            }
                        }
                    }
                    _ => {}
                }

                // Property 6: Streaming should match batch processing
                let mut stream = NviStream::try_new()?;
                for i in 0..close_data.len() {
                    if let Some(stream_val) = stream.update(close_data[i], volume_data[i]) {
                        let batch_val = out[i];
                        if !batch_val.is_nan() {
                            prop_assert!(
                                (stream_val - batch_val).abs() < 1e-9,
                                "Streaming mismatch at index {} (scenario {}): stream={}, batch={}",
                                i,
                                scenario,
                                stream_val,
                                batch_val
                            );
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_nvi_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); } )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test] fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); } )*
            }
        }
    }

    generate_all_nvi_tests!(
        check_nvi_partial_params,
        check_nvi_accuracy,
        check_nvi_empty_data,
        check_nvi_not_enough_valid_data,
        check_nvi_streaming,
        check_nvi_no_poison
    );

    #[cfg(test)]
    generate_all_nvi_tests!(check_nvi_property);
}

#[cfg(feature = "python")]
#[pyclass(name = "NviStream")]
pub struct NviStreamPy {
    stream: NviStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl NviStreamPy {
    #[new]
    fn new() -> PyResult<Self> {
        let stream = NviStream::try_new().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(NviStreamPy { stream })
    }

    fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        self.stream.update(close, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "nvi")]
#[pyo3(signature = (close, volume, kernel=None))]
pub fn nvi_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let input = NviInput::from_slices(close_slice, volume_slice, NviParams);

    let result_vec: Vec<f64> = py
        .allow_threads(|| nvi_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "nvi_batch")]
#[pyo3(signature = (close, volume, kernel=None))]
pub fn nvi_batch_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    // Allocate NumPy buffer once and fill without extra copies.
    let rows = 1usize;
    let cols = close_slice.len();
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    // Compute inside GIL-free region. Prefill warm prefix manually since we hold f64, not MaybeUninit.
    py.allow_threads(|| -> Result<(), NviError> {
        if close_slice.len() != volume_slice.len() {
            return Err(NviError::MismatchedLength {
                close_len: close_slice.len(),
                volume_len: volume_slice.len(),
            });
        }
        let first = close_slice
            .iter()
            .zip(volume_slice)
            .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
            .ok_or_else(|| {
                if close_slice.iter().all(|&c| c.is_nan()) {
                    NviError::AllCloseValuesNaN
                } else {
                    NviError::AllVolumeValuesNaN
                }
            })?;
        if cols - first < 2 {
            return Err(NviError::NotEnoughValidData {
                needed: 2,
                valid: cols - first,
            });
        }
        // Warm prefix NaNs for row 0 only.
        for v in &mut out_slice[..first] {
            *v = f64::NAN;
        }

        // Row compute.
        unsafe { nvi_row_scalar(close_slice, volume_slice, first, out_slice) };
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("values", out_arr.reshape((rows, cols))?)?;
    d.set_item("rows", rows)?;
    d.set_item("cols", cols)?;
    Ok(d)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nvi_js(close: &[f64], volume: &[f64]) -> Result<Vec<f64>, JsValue> {
    let mut output = vec![0.0; close.len()];

    nvi_into_slice(&mut output, close, volume, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nvi_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<(), JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        // Check for aliasing with either input pointer
        if close_ptr == out_ptr as *const f64 || volume_ptr == out_ptr as *const f64 {
            // Handle aliasing by using a temporary buffer
            let mut temp = vec![0.0; len];
            nvi_into_slice(&mut temp, close, volume, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            nvi_into_slice(out, close, volume, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nvi_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nvi_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn nvi_batch_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<usize, JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len); // 1×len

        if close.len() != volume.len() {
            return Err(JsValue::from_str("Length mismatch"));
        }
        let first = close
            .iter()
            .zip(volume)
            .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
            .ok_or_else(|| JsValue::from_str("All values NaN in one or both inputs"))?;
        if len - first < 2 {
            return Err(JsValue::from_str("Not enough valid data"));
        }

        // Warm prefix NaNs then compute row.
        for v in &mut out[..first] {
            *v = f64::NAN;
        }
        nvi_row_scalar(close, volume, first, out);
        Ok(1) // rows
    }
}
