//! # Aroon Indicator
//!
//! A trend-following indicator that measures the strength and potential direction of a market trend
//! based on the recent highs and lows over a specified window. Provides two outputs:
//! - **aroon_up**: How close the most recent highest high is to the current bar (percentage).
//! - **aroon_down**: How close the most recent lowest low is to the current bar (percentage).
//!
//! ## Parameters
//! - **length**: Lookback window (default: 14)
//!
//! ## Errors
//! - **AllValuesNaN**: aroon: All input data values are `NaN`.
//! - **InvalidLength**: aroon: `length` is zero or exceeds the data length.
//! - **NotEnoughValidData**: aroon: Not enough valid data points for the requested `length`.
//! - **MismatchSliceLength**: aroon: `high` and `low` slices differ in length.
//!
//! ## Returns
//! - **`Ok(AroonOutput)`** on success, containing vectors for aroon_up and aroon_down.
//! - **`Err(AroonError)`** otherwise.

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
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum AroonData<'a> {
    Candles { candles: &'a Candles },
    SlicesHL { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct AroonParams {
    pub length: Option<usize>,
}

impl Default for AroonParams {
    fn default() -> Self {
        Self { length: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AroonInput<'a> {
    pub data: AroonData<'a>,
    pub params: AroonParams,
}

impl<'a> AroonInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, p: AroonParams) -> Self {
        Self {
            data: AroonData::Candles { candles: c },
            params: p,
        }
    }
    #[inline]
    pub fn from_slices_hl(high: &'a [f64], low: &'a [f64], p: AroonParams) -> Self {
        Self {
            data: AroonData::SlicesHL { high, low },
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, AroonParams::default())
    }
    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(14)
    }
}

#[derive(Debug, Clone)]
pub struct AroonOutput {
    pub aroon_up: Vec<f64>,
    pub aroon_down: Vec<f64>,
}

#[derive(Copy, Clone, Debug)]
pub struct AroonBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for AroonBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}
impl AroonBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<AroonOutput, AroonError> {
        let p = AroonParams {
            length: self.length,
        };
        let i = AroonInput::from_candles(c, p);
        aroon_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<AroonOutput, AroonError> {
        let p = AroonParams {
            length: self.length,
        };
        let i = AroonInput::from_slices_hl(high, low, p);
        aroon_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<AroonStream, AroonError> {
        let p = AroonParams {
            length: self.length,
        };
        AroonStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AroonError {
    #[error("aroon: All values are NaN.")]
    AllValuesNaN,
    #[error("aroon: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("aroon: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("aroon: Mismatch in high/low slice length: high_len={high_len}, low_len={low_len}")]
    MismatchSliceLength { high_len: usize, low_len: usize },
}

#[inline]
pub fn aroon(input: &AroonInput) -> Result<AroonOutput, AroonError> {
    aroon_with_kernel(input, Kernel::Auto)
}

pub fn aroon_with_kernel(input: &AroonInput, kernel: Kernel) -> Result<AroonOutput, AroonError> {
    let (high, low): (&[f64], &[f64]) = match &input.data {
        AroonData::Candles { candles } => {
            (source_type(candles, "high"), source_type(candles, "low"))
        }
        AroonData::SlicesHL { high, low } => (*high, *low),
    };
    if high.len() != low.len() {
        return Err(AroonError::MismatchSliceLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    let len = high.len();
    let length = input.get_length();

    if length == 0 || length > len {
        return Err(AroonError::InvalidLength {
            length,
            data_len: len,
        });
    }
    if len < length {
        return Err(AroonError::NotEnoughValidData {
            needed: length,
            valid: len,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut up = vec![f64::NAN; len];
    let mut down = vec![f64::NAN; len];

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                aroon_scalar(high, low, length, &mut up, &mut down)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => aroon_avx2(high, low, length, &mut up, &mut down),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                aroon_avx512(high, low, length, &mut up, &mut down)
            }
            _ => unreachable!(),
        }
    }
    Ok(AroonOutput {
        aroon_up: up,
        aroon_down: down,
    })
}

#[inline]
pub fn aroon_scalar(high: &[f64], low: &[f64], length: usize, up: &mut [f64], down: &mut [f64]) {
    let len = high.len();
    assert!(
        length >= 1 && length <= len,
        "Invalid length: {} for data of size {}",
        length,
        len
    );
    assert!(
        low.len() == len && up.len() == len && down.len() == len,
        "Slice lengths must match"
    );

    let inv_length = 1.0 / (length as f64);

    // 1) Fill first `length` entries with NaN
    for i in 0..length {
        up[i] = f64::NAN;
        down[i] = f64::NAN;
    }

    // 2) For each bar i from `length` up to `len - 1`, scan a window of size `length + 1`.
    for i in length..len {
        let start = i - length;
        // Initialize with the first bar in [start..=i]
        let mut max_val = high[start];
        let mut min_val = low[start];
        let mut max_idx = start;
        let mut min_idx = start;

        // Find indices of highest high / lowest low in [start..=i]
        for j in (start + 1)..=i {
            let h = high[j];
            if h > max_val {
                max_val = h;
                max_idx = j;
            }
            let l = low[j];
            if l < min_val {
                min_val = l;
                min_idx = j;
            }
        }

        // periods_hi = how many bars ago the highest high was (0..=length)
        let periods_hi = i - max_idx;
        let periods_lo = i - min_idx;

        // Aroon up/down = (length - periods)/length * 100
        up[i] = (length as f64 - periods_hi as f64) * inv_length * 100.0;
        down[i] = (length as f64 - periods_lo as f64) * inv_length * 100.0;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn aroon_avx512(high: &[f64], low: &[f64], length: usize, up: &mut [f64], down: &mut [f64]) {
    unsafe {
        aroon_scalar(high, low, length, up, down);
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn aroon_avx2(high: &[f64], low: &[f64], length: usize, up: &mut [f64], down: &mut [f64]) {
    unsafe {
        aroon_scalar(high, low, length, up, down);
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn aroon_avx512_short(
    high: &[f64],
    low: &[f64],
    length: usize,
    up: &mut [f64],
    down: &mut [f64],
) {
    aroon_avx512(high, low, length, up, down)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn aroon_avx512_long(
    high: &[f64],
    low: &[f64],
    length: usize,
    up: &mut [f64],
    down: &mut [f64],
) {
    aroon_avx512(high, low, length, up, down)
}

#[derive(Debug)]
pub struct AroonStream {
    length: usize,
    buf_size: usize, // = length + 1
    buffer_high: Vec<f64>,
    buffer_low: Vec<f64>,
    head: usize,  // next write position in [0..buf_size)
    count: usize, // how many total bars have been pushed
}

impl AroonStream {
    /// Create a new streaming Aroon from `params`.  Extracts `length = params.length.unwrap_or(14)`.
    /// Fails if `length == 0`.  Allocates two Vecs of size `length + 1`, each pre‐filled with NaN.
    pub fn try_new(params: AroonParams) -> Result<Self, AroonError> {
        let length = params.length.unwrap_or(14);
        if length == 0 {
            return Err(AroonError::InvalidLength {
                length: 0,
                data_len: 0,
            });
        }
        let buf_size = length + 1;
        Ok(AroonStream {
            length,
            buf_size,
            buffer_high: vec![f64::NAN; buf_size],
            buffer_low: vec![f64::NAN; buf_size],
            head: 0,
            count: 0,
        })
    }

    /// Push a new (high, low).  Until we have seen at least `length+1` bars, this returns `None`.
    /// Once `count >= length+1`, each call returns `Some((aroon_up, aroon_down))`.
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64) -> Option<(f64, f64)> {
        // 1) Overwrite the “head” slot
        self.buffer_high[self.head] = high;
        self.buffer_low[self.head] = low;

        // 2) Advance head mod buf_size
        self.head = (self.head + 1) % self.buf_size;

        // 3) Increment count until we reach buf_size
        if self.count < self.buf_size {
            self.count += 1;
        }

        // 4) If we haven’t yet filled `length+1` bars, return None
        if self.count < self.buf_size {
            return None;
        }

        // 5) Compute “current index” = the slot we just wrote was (head + buf_size − 1) % buf_size
        let cur_idx = (self.head + self.buf_size - 1) % self.buf_size;

        // 6) Scan exactly the last (length+1) bars in chronological order:
        //    - “oldest_idx” is (cur_idx - length) mod buf_size  ≡  (cur_idx + 1) % buf_size
        let oldest_idx = (cur_idx + 1) % self.buf_size;
        // Initialize to the oldest bar in the window:
        let mut max_idx = oldest_idx;
        let mut min_idx = oldest_idx;
        let mut max_h = self.buffer_high[oldest_idx];
        let mut min_l = self.buffer_low[oldest_idx];
        // Walk forward k = 1..=length (so that:
        //    (oldest_idx + length) % buf_size == cur_idx,
        // covering every bar from oldest → current in order)
        for k in 1..=self.length {
            let idx = (oldest_idx + k) % self.buf_size;
            let hv = self.buffer_high[idx];
            if hv > max_h {
                max_h = hv;
                max_idx = idx;
            }
            let lv = self.buffer_low[idx];
            if lv < min_l {
                min_l = lv;
                min_idx = idx;
            }
        }

        // 7) “Bars ago” for that max:  dist_hi = (cur_idx − max_idx) mod buf_size
        let dist_hi =
            ((cur_idx as isize - max_idx as isize).rem_euclid(self.buf_size as isize)) as usize;
        let dist_lo =
            ((cur_idx as isize - min_idx as isize).rem_euclid(self.buf_size as isize)) as usize;

        // 8) Aroon formula: up = (length − dist_hi)/length * 100
        let inv_len = 1.0 / (self.length as f64);
        let up = (self.length as f64 - dist_hi as f64) * inv_len * 100.0;
        let down = (self.length as f64 - dist_lo as f64) * inv_len * 100.0;

        Some((up, down))
    }
}

#[derive(Clone, Debug)]
pub struct AroonBatchRange {
    pub length: (usize, usize, usize),
}
impl Default for AroonBatchRange {
    fn default() -> Self {
        Self {
            length: (14, 50, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AroonBatchBuilder {
    range: AroonBatchRange,
    kernel: Kernel,
}
impl AroonBatchBuilder {
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
    pub fn length_static(mut self, x: usize) -> Self {
        self.range.length = (x, x, 0);
        self
    }
    pub fn apply_slices(self, high: &[f64], low: &[f64]) -> Result<AroonBatchOutput, AroonError> {
        aroon_batch_with_kernel(high, low, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        k: Kernel,
    ) -> Result<AroonBatchOutput, AroonError> {
        AroonBatchBuilder::new().kernel(k).apply_slices(high, low)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<AroonBatchOutput, AroonError> {
        self.apply_slices(source_type(c, "high"), source_type(c, "low"))
    }
    pub fn with_default_candles(c: &Candles) -> Result<AroonBatchOutput, AroonError> {
        AroonBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

pub struct AroonBatchOutput {
    pub up: Vec<f64>,
    pub down: Vec<f64>,
    pub combos: Vec<AroonParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AroonBatchOutput {
    pub fn row_for_params(&self, p: &AroonParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.length.unwrap_or(14) == p.length.unwrap_or(14))
    }
    pub fn up_for(&self, p: &AroonParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.up[start..start + self.cols]
        })
    }
    pub fn down_for(&self, p: &AroonParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.down[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &AroonBatchRange) -> Vec<AroonParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let lengths = axis_usize(r.length);
    let mut out = Vec::with_capacity(lengths.len());
    for &l in &lengths {
        out.push(AroonParams { length: Some(l) });
    }
    out
}

pub fn aroon_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    sweep: &AroonBatchRange,
    k: Kernel,
) -> Result<AroonBatchOutput, AroonError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(AroonError::InvalidLength {
                length: 0,
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
    aroon_batch_par_slice(high, low, sweep, simd)
}

#[inline(always)]
pub fn aroon_batch_slice(
    high: &[f64],
    low: &[f64],
    sweep: &AroonBatchRange,
    kern: Kernel,
) -> Result<AroonBatchOutput, AroonError> {
    aroon_batch_inner(high, low, sweep, kern, false)
}
#[inline(always)]
pub fn aroon_batch_par_slice(
    high: &[f64],
    low: &[f64],
    sweep: &AroonBatchRange,
    kern: Kernel,
) -> Result<AroonBatchOutput, AroonError> {
    aroon_batch_inner(high, low, sweep, kern, true)
}
#[inline(always)]
fn aroon_batch_inner(
    high: &[f64],
    low: &[f64],
    sweep: &AroonBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AroonBatchOutput, AroonError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(AroonError::InvalidLength {
            length: 0,
            data_len: 0,
        });
    }
    if high.len() != low.len() {
        return Err(AroonError::MismatchSliceLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    let len = high.len();
    let max_l = combos.iter().map(|c| c.length.unwrap()).max().unwrap();
    if len < max_l {
        return Err(AroonError::NotEnoughValidData {
            needed: max_l,
            valid: len,
        });
    }
    let rows = combos.len();
    let cols = len;
    let mut up = vec![f64::NAN; rows * cols];
    let mut down = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_up: &mut [f64], out_down: &mut [f64]| unsafe {
        let length = combos[row].length.unwrap();
        match kern {
            Kernel::Scalar => aroon_row_scalar(high, low, length, out_up, out_down),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => aroon_row_avx2(high, low, length, out_up, out_down),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => aroon_row_avx512(high, low, length, out_up, out_down),
            _ => unreachable!(),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            up.par_chunks_mut(cols)
                .zip(down.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (u, d))| do_row(row, u, d));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (u, d)) in up.chunks_mut(cols).zip(down.chunks_mut(cols)).enumerate() {
                do_row(row, u, d);
            }
        }
    } else {
        for (row, (u, d)) in up.chunks_mut(cols).zip(down.chunks_mut(cols)).enumerate() {
            do_row(row, u, d);
        }
    }
    Ok(AroonBatchOutput {
        up,
        down,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub unsafe fn aroon_row_scalar(
    high: &[f64],
    low: &[f64],
    length: usize,
    out_up: &mut [f64],
    out_down: &mut [f64],
) {
    aroon_scalar(high, low, length, out_up, out_down)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_row_avx2(
    high: &[f64],
    low: &[f64],
    length: usize,
    out_up: &mut [f64],
    out_down: &mut [f64],
) {
    aroon_row_scalar(high, low, length, out_up, out_down)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_row_avx512(
    high: &[f64],
    low: &[f64],
    length: usize,
    out_up: &mut [f64],
    out_down: &mut [f64],
) {
    if length <= 32 {
        aroon_row_avx512_short(high, low, length, out_up, out_down)
    } else {
        aroon_row_avx512_long(high, low, length, out_up, out_down)
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_row_avx512_short(
    high: &[f64],
    low: &[f64],
    length: usize,
    out_up: &mut [f64],
    out_down: &mut [f64],
) {
    aroon_row_scalar(high, low, length, out_up, out_down)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn aroon_row_avx512_long(
    high: &[f64],
    low: &[f64],
    length: usize,
    out_up: &mut [f64],
    out_down: &mut [f64],
) {
    aroon_row_scalar(high, low, length, out_up, out_down)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_aroon_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = AroonParams { length: None };
        let input = AroonInput::from_candles(&candles, partial_params);
        let result = aroon_with_kernel(&input, kernel)?;
        assert_eq!(result.aroon_up.len(), candles.close.len());
        assert_eq!(result.aroon_down.len(), candles.close.len());
        Ok(())
    }

    fn check_aroon_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AroonInput::with_default_candles(&candles);
        let result = aroon_with_kernel(&input, kernel)?;

        let expected_up_last_five = [21.43, 14.29, 7.14, 0.0, 0.0];
        let expected_down_last_five = [71.43, 64.29, 57.14, 50.0, 42.86];

        assert!(
            result.aroon_up.len() >= 5 && result.aroon_down.len() >= 5,
            "Not enough Aroon values"
        );

        let start_index = result.aroon_up.len().saturating_sub(5);

        let up_last_five = &result.aroon_up[start_index..];
        let down_last_five = &result.aroon_down[start_index..];

        for (i, &value) in up_last_five.iter().enumerate() {
            assert!(
                (value - expected_up_last_five[i]).abs() < 1e-2,
                "Aroon Up mismatch at index {}: expected {}, got {}",
                i,
                expected_up_last_five[i],
                value
            );
        }

        for (i, &value) in down_last_five.iter().enumerate() {
            assert!(
                (value - expected_down_last_five[i]).abs() < 1e-2,
                "Aroon Down mismatch at index {}: expected {}, got {}",
                i,
                expected_down_last_five[i],
                value
            );
        }

        Ok(())
    }

    fn check_aroon_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AroonInput::with_default_candles(&candles);
        match input.data {
            AroonData::Candles { .. } => {}
            _ => panic!("Expected AroonData::Candles variant"),
        }
        let result = aroon_with_kernel(&input, kernel)?;
        assert_eq!(result.aroon_up.len(), candles.close.len());
        assert_eq!(result.aroon_down.len(), candles.close.len());
        Ok(())
    }

    fn check_aroon_zero_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 10.0, 11.0];
        let params = AroonParams { length: Some(0) };
        let input = AroonInput::from_slices_hl(&high, &low, params);
        let result = aroon_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for zero length");
        Ok(())
    }

    fn check_aroon_length_exceeds_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 10.0, 11.0];
        let params = AroonParams { length: Some(14) };
        let input = AroonInput::from_slices_hl(&high, &low, params);
        let result = aroon_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for length > data.len()");
        Ok(())
    }

    fn check_aroon_very_small_data_set(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [100.0];
        let low = [99.5];
        let params = AroonParams { length: Some(14) };
        let input = AroonInput::from_slices_hl(&high, &low, params);
        let result = aroon_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "Expected error for data smaller than length"
        );
        Ok(())
    }

    fn check_aroon_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = AroonParams { length: Some(14) };
        let first_input = AroonInput::from_candles(&candles, first_params);
        let first_result = aroon_with_kernel(&first_input, kernel)?;
        assert_eq!(first_result.aroon_up.len(), candles.close.len());
        assert_eq!(first_result.aroon_down.len(), candles.close.len());
        let second_params = AroonParams { length: Some(5) };
        let second_input = AroonInput::from_slices_hl(&candles.high, &candles.low, second_params);
        let second_result = aroon_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.aroon_up.len(), candles.close.len());
        assert_eq!(second_result.aroon_down.len(), candles.close.len());
        Ok(())
    }

    fn check_aroon_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = AroonParams { length: Some(14) };
        let input = AroonInput::from_candles(&candles, params);
        let result = aroon_with_kernel(&input, kernel)?;
        assert_eq!(result.aroon_up.len(), candles.close.len());
        assert_eq!(result.aroon_down.len(), candles.close.len());
        if result.aroon_up.len() > 240 {
            for i in 240..result.aroon_up.len() {
                assert!(
                    !result.aroon_up[i].is_nan(),
                    "Found NaN in aroon_up at {}",
                    i
                );
                assert!(
                    !result.aroon_down[i].is_nan(),
                    "Found NaN in aroon_down at {}",
                    i
                );
            }
        }
        Ok(())
    }

    fn check_aroon_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let length = 14;

        let input = AroonInput::from_candles(
            &candles,
            AroonParams {
                length: Some(length),
            },
        );
        let batch_output = aroon_with_kernel(&input, kernel)?;

        let mut stream = AroonStream::try_new(AroonParams {
            length: Some(length),
        })?;
        let mut stream_up = Vec::with_capacity(candles.close.len());
        let mut stream_down = Vec::with_capacity(candles.close.len());
        for (&h, &l) in candles.high.iter().zip(&candles.low) {
            match stream.update(h, l) {
                Some((up, down)) => {
                    stream_up.push(up);
                    stream_down.push(down);
                }
                None => {
                    stream_up.push(f64::NAN);
                    stream_down.push(f64::NAN);
                }
            }
        }
        assert_eq!(batch_output.aroon_up.len(), stream_up.len());
        assert_eq!(batch_output.aroon_down.len(), stream_down.len());
        for (i, (&b, &s)) in batch_output.aroon_up.iter().zip(&stream_up).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-8,
                "[{}] Aroon streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        for (i, (&b, &s)) in batch_output.aroon_down.iter().zip(&stream_down).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-8,
                "[{}] Aroon streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_aroon_tests {
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

    generate_all_aroon_tests!(
        check_aroon_partial_params,
        check_aroon_accuracy,
        check_aroon_default_candles,
        check_aroon_zero_length,
        check_aroon_length_exceeds_data,
        check_aroon_very_small_data_set,
        check_aroon_reinput,
        check_aroon_nan_handling,
        check_aroon_streaming
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = AroonBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = AroonParams::default();
        let row = output.up_for(&def).expect("default up row missing");
        assert_eq!(row.len(), c.close.len());

        let expected = [21.43, 14.29, 7.14, 0.0, 0.0];
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
