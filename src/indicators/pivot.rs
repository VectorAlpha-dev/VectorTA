//! # Pivot Points (PIVOT)
//!
//! Support (S) and resistance (R) levels from High, Low, Close, Open prices.
//! Multiple calculation modes supported (Standard, Fibonacci, Demark, Camarilla, Woodie).
//!
//! ## Parameters
//! - **mode**: Calculation method. 0=Standard, 1=Fibonacci, 2=Demark, 3=Camarilla (default), 4=Woodie
//!
//! ## Errors
//! - **EmptyData**: Required field missing
//! - **AllValuesNaN**: All values are NaN
//! - **NotEnoughValidData**: Not enough valid data for calculation
//!
//! ## Returns
//! - **Ok(PivotOutput)** with 9 Vec<f64> levels, each input length
//! - **Err(PivotError)** on error

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

// ========== DATA/INPUT/OUTPUT STRUCTS ==========

#[derive(Debug, Clone)]
pub enum PivotData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64], close: &'a [f64], open: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct PivotParams {
    pub mode: Option<usize>,
}
impl Default for PivotParams {
    fn default() -> Self { Self { mode: Some(3) } }
}

#[derive(Debug, Clone)]
pub struct PivotInput<'a> {
    pub data: PivotData<'a>,
    pub params: PivotParams,
}
impl<'a> PivotInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: PivotParams) -> Self {
        Self { data: PivotData::Candles { candles }, params }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64], low: &'a [f64], close: &'a [f64], open: &'a [f64], params: PivotParams
    ) -> Self {
        Self { data: PivotData::Slices { high, low, close, open }, params }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, PivotParams::default())
    }
    #[inline]
    pub fn get_mode(&self) -> usize {
        self.params.mode.unwrap_or_else(|| PivotParams::default().mode.unwrap())
    }
}
impl<'a> AsRef<PivotData<'a>> for PivotInput<'a> {
    fn as_ref(&self) -> &PivotData<'a> { &self.data }
}

#[derive(Debug, Clone)]
pub struct PivotOutput {
    pub r4: Vec<f64>, pub r3: Vec<f64>, pub r2: Vec<f64>, pub r1: Vec<f64>, pub pp: Vec<f64>,
    pub s1: Vec<f64>, pub s2: Vec<f64>, pub s3: Vec<f64>, pub s4: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum PivotError {
    #[error("pivot: One or more required fields is empty.")]
    EmptyData,
    #[error("pivot: All values are NaN.")]
    AllValuesNaN,
    #[error("pivot: Not enough valid data after the first valid index.")]
    NotEnoughValidData,
}

// ========== BUILDER ==========

#[derive(Copy, Clone, Debug)]
pub struct PivotBuilder {
    mode: Option<usize>,
    kernel: Kernel,
}
impl Default for PivotBuilder {
    fn default() -> Self { Self { mode: None, kernel: Kernel::Auto } }
}
impl PivotBuilder {
    #[inline(always)] pub fn new() -> Self { Self::default() }
    #[inline(always)] pub fn mode(mut self, mode: usize) -> Self { self.mode = Some(mode); self }
    #[inline(always)] pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<PivotOutput, PivotError> {
        let params = PivotParams { mode: self.mode };
        let input = PivotInput::from_candles(candles, params);
        pivot_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self, high: &[f64], low: &[f64], close: &[f64], open: &[f64]
    ) -> Result<PivotOutput, PivotError> {
        let params = PivotParams { mode: self.mode };
        let input = PivotInput::from_slices(high, low, close, open, params);
        pivot_with_kernel(&input, self.kernel)
    }
}

// ========== MAIN INTERFACE FUNCTIONS ==========

#[inline]
pub fn pivot(input: &PivotInput) -> Result<PivotOutput, PivotError> {
    pivot_with_kernel(input, Kernel::Auto)
}

pub fn pivot_with_kernel(input: &PivotInput, kernel: Kernel) -> Result<PivotOutput, PivotError> {
    let (high, low, close, open) = match &input.data {
        PivotData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            let open = source_type(candles, "open");
            (high, low, close, open)
        }
        PivotData::Slices { high, low, close, open } => (*high, *low, *close, *open),
    };
    let len = high.len();
    if high.is_empty() || low.is_empty() || close.is_empty() { return Err(PivotError::EmptyData); }
    if low.len() != len || close.len() != len || open.len() != len { return Err(PivotError::EmptyData); }
    let mode = input.get_mode();

    let mut r4 = vec![f64::NAN; len];
    let mut r3 = vec![f64::NAN; len];
    let mut r2 = vec![f64::NAN; len];
    let mut r1 = vec![f64::NAN; len];
    let mut pp = vec![f64::NAN; len];
    let mut s1 = vec![f64::NAN; len];
    let mut s2 = vec![f64::NAN; len];
    let mut s3 = vec![f64::NAN; len];
    let mut s4 = vec![f64::NAN; len];

    let mut first_valid_idx = None;
    for i in 0..len {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if !(h.is_nan() || l.is_nan() || c.is_nan()) { first_valid_idx = Some(i); break; }
    }
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(PivotError::AllValuesNaN),
    };
    if first_valid_idx >= len { return Err(PivotError::NotEnoughValidData); }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                pivot_scalar(high, low, close, open, mode, first_valid_idx, &mut r4, &mut r3, &mut r2, &mut r1, &mut pp, &mut s1, &mut s2, &mut s3, &mut s4)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                pivot_avx2(high, low, close, open, mode, first_valid_idx, &mut r4, &mut r3, &mut r2, &mut r1, &mut pp, &mut s1, &mut s2, &mut s3, &mut s4)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                pivot_avx512(high, low, close, open, mode, first_valid_idx, &mut r4, &mut r3, &mut r2, &mut r1, &mut pp, &mut s1, &mut s2, &mut s3, &mut s4)
            }
            _ => unreachable!(),
        }
    }
    Ok(PivotOutput { r4, r3, r2, r1, pp, s1, s2, s3, s4 })
}

#[inline]
pub unsafe fn pivot_scalar(
    high: &[f64], low: &[f64], close: &[f64], open: &[f64], mode: usize, first: usize,
    r4: &mut [f64], r3: &mut [f64], r2: &mut [f64], r1: &mut [f64], pp: &mut [f64],
    s1: &mut [f64], s2: &mut [f64], s3: &mut [f64], s4: &mut [f64]
) {
    let len = high.len();
    for i in first..len {
        let h = high[i]; let l = low[i]; let c = close[i]; let o = open[i];
        if h.is_nan() || l.is_nan() || c.is_nan() { continue; }
        let p = match mode {
            2 => {
                if c < o { (h + 2.0 * l + c) / 4.0 }
                else if c > o { (2.0 * h + l + c) / 4.0 }
                else { (h + l + 2.0 * c) / 4.0 }
            }
            4 => (h + l + (2.0 * o)) / 4.0,
            _ => (h + l + c) / 3.0,
        };
        pp[i] = p;
        match mode {
            0 => {
                r1[i] = 2.0 * p - l; r2[i] = p + (h - l);
                s1[i] = 2.0 * p - h; s2[i] = p - (h - l);
            }
            1 => {
                r1[i] = p + 0.382 * (h - l); r2[i] = p + 0.618 * (h - l);
                r3[i] = p + 1.0 * (h - l);
                s1[i] = p - 0.382 * (h - l); s2[i] = p - 0.618 * (h - l);
                s3[i] = p - 1.0 * (h - l);
            }
            2 => {
                s1[i] = if c < o { (h + 2.0 * l + c) / 2.0 - h }
                else if c > o { (2.0 * h + l + c) / 2.0 - h }
                else { (h + l + 2.0 * c) / 2.0 - h };
                r1[i] = if c < o { (h + 2.0 * l + c) / 2.0 - l }
                else if c > o { (2.0 * h + l + c) / 2.0 - l }
                else { (h + l + 2.0 * c) / 2.0 - l };
            }
            3 => {
                r4[i] = (0.55 * (h - l)) + c;
                r3[i] = (0.275 * (h - l)) + c;
                r2[i] = (0.183 * (h - l)) + c;
                r1[i] = (0.0916 * (h - l)) + c;
                s1[i] = c - (0.0916 * (h - l));
                s2[i] = c - (0.183 * (h - l));
                s3[i] = c - (0.275 * (h - l));
                s4[i] = c - (0.55 * (h - l));
            }
            4 => {
                r3[i] = h + 2.0 * (p - l);
                r4[i] = r3[i] + (h - l);
                r2[i] = p + (h - l);
                r1[i] = 2.0 * p - l;
                s1[i] = 2.0 * p - h;
                s2[i] = p - (h - l);
                s3[i] = l - 2.0 * (h - p);
                s4[i] = s3[i] - (h - l);
            }
            _ => {},
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pivot_avx2(
    high: &[f64], low: &[f64], close: &[f64], open: &[f64], mode: usize, first: usize,
    r4: &mut [f64], r3: &mut [f64], r2: &mut [f64], r1: &mut [f64], pp: &mut [f64],
    s1: &mut [f64], s2: &mut [f64], s3: &mut [f64], s4: &mut [f64]
) {
    // AVX2 stub fallback to scalar
    pivot_scalar(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pivot_avx512(
    high: &[f64], low: &[f64], close: &[f64], open: &[f64], mode: usize, first: usize,
    r4: &mut [f64], r3: &mut [f64], r2: &mut [f64], r1: &mut [f64], pp: &mut [f64],
    s1: &mut [f64], s2: &mut [f64], s3: &mut [f64], s4: &mut [f64]
) {
    if high.len() <= 32 {
        pivot_avx512_short(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
    } else {
        pivot_avx512_long(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pivot_avx512_short(
    high: &[f64], low: &[f64], close: &[f64], open: &[f64], mode: usize, first: usize,
    r4: &mut [f64], r3: &mut [f64], r2: &mut [f64], r1: &mut [f64], pp: &mut [f64],
    s1: &mut [f64], s2: &mut [f64], s3: &mut [f64], s4: &mut [f64]
) {
    // AVX512 short stub fallback to scalar
    pivot_scalar(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn pivot_avx512_long(
    high: &[f64], low: &[f64], close: &[f64], open: &[f64], mode: usize, first: usize,
    r4: &mut [f64], r3: &mut [f64], r2: &mut [f64], r1: &mut [f64], pp: &mut [f64],
    s1: &mut [f64], s2: &mut [f64], s3: &mut [f64], s4: &mut [f64]
) {
    // AVX512 long stub fallback to scalar
    pivot_scalar(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}

// ========== ROW "BATCH" VECTORIZED API ==========

#[inline(always)]
pub unsafe fn pivot_row_scalar(
    high: &[f64], low: &[f64], close: &[f64], open: &[f64], mode: usize, first: usize,
    r4: &mut [f64], r3: &mut [f64], r2: &mut [f64], r1: &mut [f64], pp: &mut [f64],
    s1: &mut [f64], s2: &mut [f64], s3: &mut [f64], s4: &mut [f64]
) {
    pivot_scalar(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx2(
    high: &[f64], low: &[f64], close: &[f64], open: &[f64], mode: usize, first: usize,
    r4: &mut [f64], r3: &mut [f64], r2: &mut [f64], r1: &mut [f64], pp: &mut [f64],
    s1: &mut [f64], s2: &mut [f64], s3: &mut [f64], s4: &mut [f64]
) {
    pivot_avx2(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx512(
    high: &[f64], low: &[f64], close: &[f64], open: &[f64], mode: usize, first: usize,
    r4: &mut [f64], r3: &mut [f64], r2: &mut [f64], r1: &mut [f64], pp: &mut [f64],
    s1: &mut [f64], s2: &mut [f64], s3: &mut [f64], s4: &mut [f64]
) {
    pivot_avx512(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx512_short(
    high: &[f64], low: &[f64], close: &[f64], open: &[f64], mode: usize, first: usize,
    r4: &mut [f64], r3: &mut [f64], r2: &mut [f64], r1: &mut [f64], pp: &mut [f64],
    s1: &mut [f64], s2: &mut [f64], s3: &mut [f64], s4: &mut [f64]
) {
    pivot_avx512_short(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pivot_row_avx512_long(
    high: &[f64], low: &[f64], close: &[f64], open: &[f64], mode: usize, first: usize,
    r4: &mut [f64], r3: &mut [f64], r2: &mut [f64], r1: &mut [f64], pp: &mut [f64],
    s1: &mut [f64], s2: &mut [f64], s3: &mut [f64], s4: &mut [f64]
) {
    pivot_avx512_long(high, low, close, open, mode, first, r4, r3, r2, r1, pp, s1, s2, s3, s4)
}

// ========== BATCH (RANGE) API ==========

#[derive(Clone, Debug)]
pub struct PivotBatchRange {
    pub mode: (usize, usize, usize),
}
impl Default for PivotBatchRange {
    fn default() -> Self { Self { mode: (3, 3, 1) } }
}
#[derive(Clone, Debug, Default)]
pub struct PivotBatchBuilder {
    range: PivotBatchRange,
    kernel: Kernel,
}
impl PivotBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    #[inline]
    pub fn mode_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.mode = (start, end, step); self
    }
    #[inline]
    pub fn mode_static(mut self, m: usize) -> Self {
        self.range.mode = (m, m, 1); self
    }
    pub fn apply_slice(
        self, high: &[f64], low: &[f64], close: &[f64], open: &[f64]
    ) -> Result<PivotBatchOutput, PivotError> {
        pivot_batch_with_kernel(high, low, close, open, &self.range, self.kernel)
    }
    pub fn apply_candles(
        self, candles: &Candles
    ) -> Result<PivotBatchOutput, PivotError> {
        let high = source_type(candles, "high");
        let low = source_type(candles, "low");
        let close = source_type(candles, "close");
        let open = source_type(candles, "open");
        self.apply_slice(high, low, close, open)
    }
    pub fn with_default_candles(
        candles: &Candles
    ) -> Result<PivotBatchOutput, PivotError> {
        PivotBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(candles)
    }
}

pub fn pivot_batch_with_kernel(
    high: &[f64], low: &[f64], close: &[f64], open: &[f64], sweep: &PivotBatchRange, k: Kernel
) -> Result<PivotBatchOutput, PivotError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(PivotError::EmptyData),
    };
    pivot_batch_inner(high, low, close, open, sweep, kernel)
}

#[derive(Clone, Debug)]
pub struct PivotBatchOutput {
    pub levels: Vec<[Vec<f64>; 9]>,
    pub combos: Vec<PivotParams>,
    pub rows: usize,
    pub cols: usize,
}
fn expand_grid(r: &PivotBatchRange) -> Vec<PivotParams> {
    let (start, end, step) = r.mode;
    let mut v = Vec::new();
    let mut m = start;
    while m <= end { v.push(PivotParams { mode: Some(m) }); m += step; }
    v
}
fn pivot_batch_inner(
    high: &[f64], low: &[f64], close: &[f64], open: &[f64], sweep: &PivotBatchRange, kernel: Kernel
) -> Result<PivotBatchOutput, PivotError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() { return Err(PivotError::EmptyData); }
    let len = high.len();
    let mut levels = Vec::with_capacity(combos.len());
    for p in &combos {
        let mut r4 = vec![f64::NAN; len]; let mut r3 = vec![f64::NAN; len];
        let mut r2 = vec![f64::NAN; len]; let mut r1 = vec![f64::NAN; len];
        let mut pp = vec![f64::NAN; len];
        let mut s1 = vec![f64::NAN; len]; let mut s2 = vec![f64::NAN; len];
        let mut s3 = vec![f64::NAN; len]; let mut s4 = vec![f64::NAN; len];
        let mode = p.mode.unwrap_or(3);
        let mut first = None;
        for i in 0..len {
            if !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan()) { first = Some(i); break; }
        }
        let first = first.unwrap_or(len);
        unsafe {
            match kernel {
                Kernel::Scalar | Kernel::ScalarBatch => pivot_row_scalar(
                    high, low, close, open, mode, first, &mut r4, &mut r3, &mut r2, &mut r1, &mut pp, &mut s1, &mut s2, &mut s3, &mut s4),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => pivot_row_avx2(
                    high, low, close, open, mode, first, &mut r4, &mut r3, &mut r2, &mut r1, &mut pp, &mut s1, &mut s2, &mut s3, &mut s4),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => pivot_row_avx512(
                    high, low, close, open, mode, first, &mut r4, &mut r3, &mut r2, &mut r1, &mut pp, &mut s1, &mut s2, &mut s3, &mut s4),
                _ => unreachable!(),
            }
        }
        levels.push([r4, r3, r2, r1, pp, s1, s2, s3, s4]);
    }
    let rows = combos.len();
    let cols = high.len();
    Ok(PivotBatchOutput { levels, combos: combos.clone(), rows: combos.len(), cols: high.len() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;
    use paste::paste;

    fn check_pivot_default_mode_camarilla(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = PivotParams { mode: None };
        let input = PivotInput::from_candles(&candles, params);
        let result = pivot_with_kernel(&input, kernel)?;

        assert_eq!(result.r4.len(), candles.close.len());
        assert_eq!(result.r3.len(), candles.close.len());
        assert_eq!(result.r2.len(), candles.close.len());
        assert_eq!(result.r1.len(), candles.close.len());
        assert_eq!(result.pp.len(), candles.close.len());
        assert_eq!(result.s1.len(), candles.close.len());
        assert_eq!(result.s2.len(), candles.close.len());
        assert_eq!(result.s3.len(), candles.close.len());
        assert_eq!(result.s4.len(), candles.close.len());

        // Spot-check Camarilla outputs for a few points
        let last_five_r4 = &result.r4[result.r4.len().saturating_sub(5)..];
        let expected_r4 = [59466.5, 59357.55, 59243.6, 59334.85, 59170.35];
        for (i, &val) in last_five_r4.iter().enumerate() {
            let exp = expected_r4[i];
            assert!(
                (val - exp).abs() < 1e-1,
                "Camarilla r4 mismatch at index {}, expected {}, got {}",
                i, exp, val
            );
        }
        Ok(())
    }

    fn check_pivot_nan_values(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, f64::NAN, 30.0];
        let low = [9.0, 8.5, f64::NAN];
        let close = [9.5, 9.0, 29.0];
        let open = [9.1, 8.8, 28.5];

        let params = PivotParams { mode: Some(3) };
        let input = PivotInput::from_slices(&high, &low, &close, &open, params);
        let result = pivot_with_kernel(&input, kernel)?;
        assert_eq!(result.pp.len(), high.len());
        Ok(())
    }

    fn check_pivot_no_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let close: [f64; 0] = [];
        let open: [f64; 0] = [];
        let params = PivotParams { mode: Some(3) };
        let input = PivotInput::from_slices(&high, &low, &close, &open, params);
        let result = pivot_with_kernel(&input, kernel);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("One or more required fields"),
                "Expected 'EmptyData' error, got: {}",
                e
            );
        }
        Ok(())
    }

    fn check_pivot_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let close = [f64::NAN, f64::NAN];
        let open = [f64::NAN, f64::NAN];
        let params = PivotParams { mode: Some(3) };
        let input = PivotInput::from_slices(&high, &low, &close, &open, params);
        let result = pivot_with_kernel(&input, kernel);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("All values are NaN"),
                "Expected 'AllValuesNaN' error, got: {}",
                e
            );
        }
        Ok(())
    }

    fn check_pivot_fibonacci_mode(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let params = PivotParams { mode: Some(1) };
        let input = PivotInput::from_candles(&candles, params);
        let output = pivot_with_kernel(&input, kernel)?;
        assert_eq!(output.r3.len(), candles.close.len());
        Ok(())
    }

    fn check_pivot_standard_mode(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let params = PivotParams { mode: Some(0) };
        let input = PivotInput::from_candles(&candles, params);
        let output = pivot_with_kernel(&input, kernel)?;
        assert_eq!(output.r2.len(), candles.close.len());
        Ok(())
    }

    fn check_pivot_demark_mode(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let params = PivotParams { mode: Some(2) };
        let input = PivotInput::from_candles(&candles, params);
        let output = pivot_with_kernel(&input, kernel)?;
        assert_eq!(output.r1.len(), candles.close.len());
        Ok(())
    }

    fn check_pivot_woodie_mode(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let params = PivotParams { mode: Some(4) };
        let input = PivotInput::from_candles(&candles, params);
        let output = pivot_with_kernel(&input, kernel)?;
        assert_eq!(output.r4.len(), candles.close.len());
        Ok(())
    }

    fn check_pivot_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;
        let output = PivotBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles)?;
        let default = PivotParams::default();
        let def_idx = output.combos.iter().position(|p| p.mode == default.mode).expect("default row missing");
        for arr in &output.levels[def_idx] {
            assert_eq!(arr.len(), candles.close.len());
        }
        Ok(())
    }

    // Macro for all kernel variants
    macro_rules! generate_all_pivot_tests {
        ($($test_fn:ident),*) => {
            paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar>]() { let _ = $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar); }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2>]() { let _ = $test_fn(stringify!([<$test_fn _avx2>]), Kernel::Avx2); }
                    #[test]
                    fn [<$test_fn _avx512>]() { let _ = $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512); }
                )*
                $(
                    #[test]
                    fn [<$test_fn _auto_detect>]() { let _ = $test_fn(stringify!([<$test_fn _auto_detect>]), Kernel::Auto); }
                )*
            }
        }
    }

    generate_all_pivot_tests!(
        check_pivot_default_mode_camarilla,
        check_pivot_nan_values,
        check_pivot_no_data,
        check_pivot_all_nan,
        check_pivot_fibonacci_mode,
        check_pivot_standard_mode,
        check_pivot_demark_mode,
        check_pivot_woodie_mode,
        check_pivot_batch_default_row
    );
        fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;

        let output = PivotBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&candles)?;

        let def = PivotParams::default();
        let row = output.combos.iter().position(|p| p.mode == def.mode).expect("default row missing");
        let levels = &output.levels[row];

        // Spot check: each level should be the right length
        for arr in levels.iter() {
            assert_eq!(arr.len(), candles.close.len());
        }

        // Optionally, spot-check some values (e.g. Camarilla r4)
        let expected_r4 = [59466.5, 59357.55, 59243.6, 59334.85, 59170.35];
        let r4 = &levels[0];
        let last_five_r4 = &r4[r4.len().saturating_sub(5)..];
        for (i, &val) in last_five_r4.iter().enumerate() {
            let exp = expected_r4[i];
            assert!(
                (val - exp).abs() < 1e-1,
                "[{test}] Camarilla r4 mismatch at idx {i}: {val} vs {exp:?}"
            );
        }
        Ok(())
    }

    // Kernel variant macro expansion (as in alma.rs)
    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste! {
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
