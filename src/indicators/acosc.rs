//! # Accelerator Oscillator (ACOSC)
//!
//! Bill Williams’ AC Oscillator: measures median price acceleration via SMA5, SMA34, and further SMA5 smoothing.
//!
//! ## Parameters
//! - None (fixed: periods are 5 and 34)
//!
//! ## Errors
//! - **CandleFieldError**: Failed to get high/low from candles
//! - **LengthMismatch**: Slices have different lengths
//! - **NotEnoughData**: Less than 39 data points
//!
//! ## Returns
//! - `Ok(AcoscOutput)` with vectors of `osc` and `change`
//!

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AcoscData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone, Default)]
pub struct AcoscParams {} // ACOSC is fixed (no param grid), but kept for parity

#[derive(Debug, Clone)]
pub struct AcoscInput<'a> {
    pub data: AcoscData<'a>,
    pub params: AcoscParams,
}
impl<'a> AcoscInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AcoscParams) -> Self {
        Self { data: AcoscData::Candles { candles }, params }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: AcoscParams) -> Self {
        Self { data: AcoscData::Slices { high, low }, params }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self { data: AcoscData::Candles { candles }, params: AcoscParams::default() }
    }
}

#[derive(Debug, Clone)]
pub struct AcoscOutput {
    pub osc: Vec<f64>,
    pub change: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum AcoscError {
    #[error("acosc: Failed to get high/low fields from candles: {msg}")]
    CandleFieldError { msg: String },
    #[error("acosc: Mismatch in high/low candle data lengths: high_len={high_len}, low_len={low_len}")]
    LengthMismatch { high_len: usize, low_len: usize },
    #[error("acosc: Not enough data points: required={required}, actual={actual}")]
    NotEnoughData { required: usize, actual: usize },
}

#[inline]
pub fn acosc(input: &AcoscInput) -> Result<AcoscOutput, AcoscError> {
    acosc_with_kernel(input, Kernel::Auto)
}
pub fn acosc_with_kernel(input: &AcoscInput, kernel: Kernel) -> Result<AcoscOutput, AcoscError> {
    let (high, low) = match &input.data {
        AcoscData::Candles { candles } => {
            let h = candles.select_candle_field("high")
                .map_err(|e| AcoscError::CandleFieldError { msg: e.to_string() })?;
            let l = candles.select_candle_field("low")
                .map_err(|e| AcoscError::CandleFieldError { msg: e.to_string() })?;
            (h, l)
        }
        AcoscData::Slices { high, low } => (*high, *low),
    };
    if high.len() != low.len() {
        return Err(AcoscError::LengthMismatch { high_len: high.len(), low_len: low.len() });
    }
    let len = low.len();
    const PERIOD_SMA5: usize = 5;
    const PERIOD_SMA34: usize = 34;
    const REQUIRED_LENGTH: usize = PERIOD_SMA34 + PERIOD_SMA5;
    if len < REQUIRED_LENGTH {
        return Err(AcoscError::NotEnoughData { required: REQUIRED_LENGTH, actual: len });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let mut osc = vec![f64::NAN; len];
    let mut change = vec![f64::NAN; len];
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                acosc_scalar(high, low, &mut osc, &mut change)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                acosc_avx2(high, low, &mut osc, &mut change)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                acosc_avx512(high, low, &mut osc, &mut change)
            }
            _ => unreachable!(),
        }
    }
    Ok(AcoscOutput { osc, change })
}
#[inline]
pub fn acosc_scalar(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
    // SCALAR LOGIC UNCHANGED
    const PERIOD_SMA5: usize = 5;
    const PERIOD_SMA34: usize = 34;
    const INV5: f64 = 1.0 / 5.0;
    const INV34: f64 = 1.0 / 34.0;
    let len = high.len();
    let mut queue5 = [0.0; PERIOD_SMA5];
    let mut queue34 = [0.0; PERIOD_SMA34];
    let mut queue5_ao = [0.0; PERIOD_SMA5];
    let mut sum5 = 0.0;
    let mut sum34 = 0.0;
    let mut sum5_ao = 0.0;
    let mut idx5 = 0;
    let mut idx34 = 0;
    let mut idx5_ao = 0;
    for i in 0..PERIOD_SMA34 {
        let med = (high[i] + low[i]) * 0.5;
        sum34 += med;
        queue34[i] = med;
        if i < PERIOD_SMA5 {
            sum5 += med;
            queue5[i] = med;
        }
    }
    for i in PERIOD_SMA34..(PERIOD_SMA34 + PERIOD_SMA5 - 1) {
        let med = (high[i] + low[i]) * 0.5;
        sum34 += med - queue34[idx34];
        queue34[idx34] = med;
        idx34 = (idx34 + 1) % PERIOD_SMA34;
        let sma34 = sum34 * INV34;
        sum5 += med - queue5[idx5];
        queue5[idx5] = med;
        idx5 = (idx5 + 1) % PERIOD_SMA5;
        let sma5 = sum5 * INV5;
        let ao = sma5 - sma34;
        sum5_ao += ao;
        queue5_ao[idx5_ao] = ao;
        idx5_ao += 1;
    }
    if idx5_ao == PERIOD_SMA5 {
        idx5_ao = 0;
    }
    let mut prev_res = 0.0;
    for i in (PERIOD_SMA34 + PERIOD_SMA5 - 1)..len {
        let med = (high[i] + low[i]) * 0.5;
        sum34 += med - queue34[idx34];
        queue34[idx34] = med;
        idx34 = (idx34 + 1) % PERIOD_SMA34;
        let sma34 = sum34 * INV34;
        sum5 += med - queue5[idx5];
        queue5[idx5] = med;
        idx5 = (idx5 + 1) % PERIOD_SMA5;
        let sma5 = sum5 * INV5;
        let ao = sma5 - sma34;
        let old_ao = queue5_ao[idx5_ao];
        sum5_ao += ao - old_ao;
        queue5_ao[idx5_ao] = ao;
        idx5_ao = (idx5_ao + 1) % PERIOD_SMA5;
        let sma5_ao = sum5_ao * INV5;
        let res = ao - sma5_ao;
        let mom = res - prev_res;
        prev_res = res;
        osc[i] = res;
        change[i] = mom;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn acosc_avx512(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
    acosc_scalar(high, low, osc, change)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn acosc_avx2(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
    acosc_scalar(high, low, osc, change)
}
#[inline]
pub fn acosc_avx512_short(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
    acosc_scalar(high, low, osc, change)
}
#[inline]
pub fn acosc_avx512_long(high: &[f64], low: &[f64], osc: &mut [f64], change: &mut [f64]) {
    acosc_scalar(high, low, osc, change)
}

// Stream API (stateful tick-by-tick)
#[derive(Debug, Clone)]
pub struct AcoscStream {
    queue5: [f64; 5],
    queue34: [f64; 34],
    queue5_ao: [f64; 5],
    sum5: f64,
    sum34: f64,
    sum5_ao: f64,
    idx5: usize,
    idx34: usize,
    idx5_ao: usize,
    filled: usize,
    prev_res: f64,
}
impl AcoscStream {
    pub fn try_new(_params: AcoscParams) -> Result<Self, AcoscError> {
        Ok(Self {
            queue5: [0.0; 5],
            queue34: [0.0; 34],
            queue5_ao: [0.0; 5],
            sum5: 0.0,
            sum34: 0.0,
            sum5_ao: 0.0,
            idx5: 0,
            idx34: 0,
            idx5_ao: 0,
            filled: 0,
            prev_res: 0.0,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        let med = (high + low) * 0.5;
        self.filled += 1;
        if self.filled <= 34 {
            self.sum34 += med;
            self.queue34[self.filled - 1] = med;
            if self.filled <= 5 {
                self.sum5 += med;
                self.queue5[self.filled - 1] = med;
            }
            return None;
        }
        if self.filled < 39 {
            self.sum34 += med - self.queue34[self.idx34];
            self.queue34[self.idx34] = med;
            self.idx34 = (self.idx34 + 1) % 34;
            let sma34 = self.sum34 / 34.0;
            self.sum5 += med - self.queue5[self.idx5];
            self.queue5[self.idx5] = med;
            self.idx5 = (self.idx5 + 1) % 5;
            let sma5 = self.sum5 / 5.0;
            let ao = sma5 - sma34;
            self.sum5_ao += ao;
            self.queue5_ao[self.idx5_ao] = ao;
            self.idx5_ao = (self.idx5_ao + 1) % 5;
            return None;
        }
        self.sum34 += med - self.queue34[self.idx34];
        self.queue34[self.idx34] = med;
        self.idx34 = (self.idx34 + 1) % 34;
        let sma34 = self.sum34 / 34.0;
        self.sum5 += med - self.queue5[self.idx5];
        self.queue5[self.idx5] = med;
        self.idx5 = (self.idx5 + 1) % 5;
        let sma5 = self.sum5 / 5.0;
        let ao = sma5 - sma34;
        let old_ao = self.queue5_ao[self.idx5_ao];
        self.sum5_ao += ao - old_ao;
        self.queue5_ao[self.idx5_ao] = ao;
        self.idx5_ao = (self.idx5_ao + 1) % 5;
        let sma5_ao = self.sum5_ao / 5.0;
        let res = ao - sma5_ao;
        let mom = res - self.prev_res;
        self.prev_res = res;
        Some((res, mom))
    }
}

// --- Batch/Builder API ---

#[derive(Clone, Debug)]
pub struct AcoscBatchRange {} // For parity only

impl Default for AcoscBatchRange {
    fn default() -> Self { Self {} }
}

#[derive(Clone, Debug, Default)]
pub struct AcoscBatchBuilder {
    kernel: Kernel,
}
impl AcoscBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn apply_slice(self, high: &[f64], low: &[f64]) -> Result<AcoscBatchOutput, AcoscError> {
        acosc_batch_with_kernel(high, low, self.kernel)
    }
    pub fn with_default_slice(high: &[f64], low: &[f64], k: Kernel) -> Result<AcoscBatchOutput, AcoscError> {
        AcoscBatchBuilder::new().kernel(k).apply_slice(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<AcoscBatchOutput, AcoscError> {
        let high = c.select_candle_field("high")
            .map_err(|e| AcoscError::CandleFieldError { msg: e.to_string() })?;
        let low = c.select_candle_field("low")
            .map_err(|e| AcoscError::CandleFieldError { msg: e.to_string() })?;
        self.apply_slice(high, low)
    }
    pub fn with_default_candles(c: &Candles) -> Result<AcoscBatchOutput, AcoscError> {
        AcoscBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
    }
}
#[derive(Clone, Debug)]
pub struct AcoscBatchOutput {
    pub osc: Vec<f64>,
    pub change: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}
pub fn acosc_batch_with_kernel(
    high: &[f64], low: &[f64], k: Kernel,
) -> Result<AcoscBatchOutput, AcoscError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(AcoscError::NotEnoughData { required: 39, actual: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    acosc_batch_par_slice(high, low, simd)
}
#[inline(always)]
pub fn acosc_batch_slice(
    high: &[f64], low: &[f64], kern: Kernel,
) -> Result<AcoscBatchOutput, AcoscError> {
    acosc_batch_inner(high, low, kern, false)
}
#[inline(always)]
pub fn acosc_batch_par_slice(
    high: &[f64], low: &[f64], kern: Kernel,
) -> Result<AcoscBatchOutput, AcoscError> {
    acosc_batch_inner(high, low, kern, true)
}
#[inline(always)]
fn acosc_batch_inner(
    high: &[f64], low: &[f64], kern: Kernel, _parallel: bool,
) -> Result<AcoscBatchOutput, AcoscError> {
    let mut osc = vec![f64::NAN; high.len()];
    let mut change = vec![f64::NAN; high.len()];
    unsafe {
        match kern {
            Kernel::Scalar => acosc_scalar(high, low, &mut osc, &mut change),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => acosc_avx2(high, low, &mut osc, &mut change),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => acosc_avx512(high, low, &mut osc, &mut change),
            _ => unreachable!(),
        }
    }
    Ok(AcoscBatchOutput { osc, change, rows: 1, cols: high.len() })
}
#[inline(always)]
pub fn expand_grid(_r: &AcoscBatchRange) -> Vec<AcoscParams> {
    vec![AcoscParams::default()]
}

// --- Row kernel API (batch) ---
#[inline(always)]
pub unsafe fn acosc_row_scalar(
    high: &[f64], low: &[f64], out_osc: &mut [f64], out_change: &mut [f64]
) {
    acosc_scalar(high, low, out_osc, out_change)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn acosc_row_avx2(
    high: &[f64], low: &[f64], out_osc: &mut [f64], out_change: &mut [f64]
) {
    acosc_avx2(high, low, out_osc, out_change)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn acosc_row_avx512(
    high: &[f64], low: &[f64], out_osc: &mut [f64], out_change: &mut [f64]
) {
    acosc_avx512(high, low, out_osc, out_change)
}
#[inline(always)]
pub fn acosc_row_avx512_short(
    high: &[f64], low: &[f64], out_osc: &mut [f64], out_change: &mut [f64]
) {
    acosc_scalar(high, low, out_osc, out_change)
}
#[inline(always)]
pub fn acosc_row_avx512_long(
    high: &[f64], low: &[f64], out_osc: &mut [f64], out_change: &mut [f64]
) {
    acosc_scalar(high, low, out_osc, out_change)
}

// --- Optional: AcoscBuilder for strict parity ---
#[derive(Copy, Clone, Debug, Default)]
pub struct AcoscBuilder {
    kernel: Kernel,
}
impl AcoscBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline(always)]
    pub fn apply_candles(self, candles: &Candles) -> Result<AcoscOutput, AcoscError> {
        let input = AcoscInput::with_default_candles(candles);
        acosc_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<AcoscOutput, AcoscError> {
        let input = AcoscInput::from_slices(high, low, AcoscParams::default());
        acosc_with_kernel(&input, self.kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_acosc_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = AcoscParams::default();
        let input = AcoscInput::from_candles(&candles, default_params);
        let output = acosc_with_kernel(&input, kernel)?;
        assert_eq!(output.osc.len(), candles.close.len());
        assert_eq!(output.change.len(), candles.close.len());
        Ok(())
    }

    fn check_acosc_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AcoscInput::with_default_candles(&candles);
        let result = acosc_with_kernel(&input, kernel)?;
        assert_eq!(result.osc.len(), candles.close.len());
        assert_eq!(result.change.len(), candles.close.len());
        let expected_last_five_acosc_osc = [273.30, 383.72, 357.7, 291.25, 176.84];
        let expected_last_five_acosc_change = [49.6, 110.4, -26.0, -66.5, -114.4];
        let start = result.osc.len().saturating_sub(5);
        for (i, &val) in result.osc[start..].iter().enumerate() {
            assert!((val - expected_last_five_acosc_osc[i]).abs() < 1e-1,
                "[{}] ACOSC {:?} osc mismatch idx {}: got {}, expected {}",
                test_name, kernel, i, val, expected_last_five_acosc_osc[i]
            );
        }
        for (i, &val) in result.change[start..].iter().enumerate() {
            assert!((val - expected_last_five_acosc_change[i]).abs() < 1e-1,
                "[{}] ACOSC {:?} change mismatch idx {}: got {}, expected {}",
                test_name, kernel, i, val, expected_last_five_acosc_change[i]
            );
        }
        Ok(())
    }

    fn check_acosc_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AcoscInput::with_default_candles(&candles);
        match input.data {
            AcoscData::Candles { .. } => {}
            _ => panic!("Expected AcoscData::Candles variant"),
        }
        let output = acosc_with_kernel(&input, kernel)?;
        assert_eq!(output.osc.len(), candles.close.len());
        Ok(())
    }

    fn check_acosc_too_short(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [100.0, 101.0];
        let low = [99.0, 98.0];
        let params = AcoscParams::default();
        let input = AcoscInput::from_slices(&high, &low, params);
        let result = acosc_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] Should fail with not enough data", test_name);
        Ok(())
    }

    fn check_acosc_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AcoscInput::with_default_candles(&candles);
        let first_result = acosc_with_kernel(&input, kernel)?;
        assert_eq!(first_result.osc.len(), candles.close.len());
        assert_eq!(first_result.change.len(), candles.close.len());
        let input2 = AcoscInput::from_slices(&candles.high, &candles.low, AcoscParams::default());
        let second_result = acosc_with_kernel(&input2, kernel)?;
        assert_eq!(second_result.osc.len(), candles.close.len());
        for (a, b) in second_result.osc.iter().zip(first_result.osc.iter()) {
            if a.is_nan() && b.is_nan() { continue; }
            assert!((a - b).abs() < 1e-8, "Reinput values mismatch: {} vs {}", a, b);
        }
        Ok(())
    }

    fn check_acosc_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AcoscInput::with_default_candles(&candles);
        let result = acosc_with_kernel(&input, kernel)?;
        if result.osc.len() > 240 {
            for i in 240..result.osc.len() {
                assert!(!result.osc[i].is_nan(), "Found NaN in osc at {}", i);
                assert!(!result.change[i].is_nan(), "Found NaN in change at {}", i);
            }
        }
        Ok(())
    }

    fn check_acosc_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AcoscInput::with_default_candles(&candles);
        let batch = acosc_with_kernel(&input, kernel)?;
        let mut stream = AcoscStream::try_new(AcoscParams::default())?;
        let mut osc_stream = Vec::with_capacity(candles.close.len());
        let mut change_stream = Vec::with_capacity(candles.close.len());
        for (&h, &l) in candles.high.iter().zip(candles.low.iter()) {
            match stream.update(h, l) {
                Some((o, c)) => { osc_stream.push(o); change_stream.push(c); }
                None => { osc_stream.push(f64::NAN); change_stream.push(f64::NAN); }
            }
        }
        assert_eq!(batch.osc.len(), osc_stream.len());
        assert_eq!(batch.change.len(), change_stream.len());
        for (i, (&a, &b)) in batch.osc.iter().zip(osc_stream.iter()).enumerate() {
            if a.is_nan() && b.is_nan() { continue; }
            assert!((a - b).abs() < 1e-9, "Streaming osc mismatch at idx {}: {} vs {}", i, a, b);
        }
        for (i, (&a, &b)) in batch.change.iter().zip(change_stream.iter()).enumerate() {
            if a.is_nan() && b.is_nan() { continue; }
            assert!((a - b).abs() < 1e-9, "Streaming change mismatch at idx {}: {} vs {}", i, a, b);
        }
        Ok(())
    }

    macro_rules! generate_all_acosc_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(#[test]
                  fn [<$test_fn _scalar_f64>]() {
                      let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                  })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(#[test]
                  fn [<$test_fn _avx2_f64>]() {
                      let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                  }
                  #[test]
                  fn [<$test_fn _avx512_f64>]() {
                      let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                  })*
            }
        }
    }
    generate_all_acosc_tests!(
        check_acosc_partial_params,
        check_acosc_accuracy,
        check_acosc_default_candles,
        check_acosc_too_short,
        check_acosc_reinput,
        check_acosc_nan_handling,
        check_acosc_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = AcoscBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        assert_eq!(output.osc.len(), c.close.len());
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
