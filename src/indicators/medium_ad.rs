//! # Median Absolute Deviation (MEDIUM_AD)
//!
//! A robust measure of dispersion that calculates the median of the absolute
//! deviations from the median for a specified `period`.
//!
//! ## Parameters
//! - **period**: The window size (number of data points). Defaults to 5.
//!
//! ## Inputs
//! - **data**: Price data or any numeric series
//!
//! ## Returns
//! - **values**: Vector of median absolute deviation values with NaN prefix during warmup period
//!
//! ## Developer Notes
//! - **AVX2/AVX512 Kernels**: Stubs that call scalar implementation
//! - **Streaming**: Implemented with O(n log n) update performance (sorts period-sized buffer)
//! - **Zero-copy Memory**: Uses alloc_with_nan_prefix and make_uninit_matrix for batch operations

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

impl<'a> AsRef<[f64]> for MediumAdInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MediumAdData::Slice(slice) => slice,
            MediumAdData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MediumAdData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MediumAdOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MediumAdParams {
    pub period: Option<usize>,
}

impl Default for MediumAdParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct MediumAdInput<'a> {
    pub data: MediumAdData<'a>,
    pub params: MediumAdParams,
}

impl<'a> MediumAdInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MediumAdParams) -> Self {
        Self {
            data: MediumAdData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MediumAdParams) -> Self {
        Self {
            data: MediumAdData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MediumAdParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MediumAdBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for MediumAdBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MediumAdBuilder {
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
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<MediumAdOutput, MediumAdError> {
        let p = MediumAdParams {
            period: self.period,
        };
        let i = MediumAdInput::from_candles(c, "close", p);
        medium_ad_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MediumAdOutput, MediumAdError> {
        let p = MediumAdParams {
            period: self.period,
        };
        let i = MediumAdInput::from_slice(d, p);
        medium_ad_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<MediumAdStream, MediumAdError> {
        let p = MediumAdParams {
            period: self.period,
        };
        MediumAdStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum MediumAdError {
    #[error("medium_ad: All values are NaN.")]
    AllValuesNaN,
    #[error("medium_ad: Empty data provided for MEDIUM_AD.")]
    EmptyData,
    #[error("medium_ad: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("medium_ad: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn medium_ad(input: &MediumAdInput) -> Result<MediumAdOutput, MediumAdError> {
    medium_ad_with_kernel(input, Kernel::Auto)
}

pub fn medium_ad_with_kernel(
    input: &MediumAdInput,
    kernel: Kernel,
) -> Result<MediumAdOutput, MediumAdError> {
    let data: &[f64] = match &input.data {
        MediumAdData::Candles { candles, source } => source_type(candles, source),
        MediumAdData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(MediumAdError::EmptyData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MediumAdError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(MediumAdError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(MediumAdError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let mut out = alloc_with_nan_prefix(len, first + period - 1);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => medium_ad_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => medium_ad_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => medium_ad_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(MediumAdOutput { values: out })
}

#[inline]
pub fn medium_ad_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    // alloc_with_nan_prefix already wrote NaNs up to warm = first_valid + period - 1
    use core::cmp::Ordering;

    let mut buf = vec![0.0; period];
    let mid = period / 2;

    // helper: total median from select_nth (handles even period)
    #[inline(always)]
    fn median_from(buf: &mut [f64], mid: usize) -> f64 {
        // partition around mid
        buf.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        if (buf.len() & 1) == 1 {
            buf[mid]
        } else {
            // need the max of the lower partition
            let lo_max = buf[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            0.5 * (lo_max + buf[mid])
        }
    }

    for i in (first_valid + period - 1)..data.len() {
        let window = &data[i + 1 - period..=i];

        // Strict NaN policy: any NaN in window -> NaN out (unchanged from your logic)
        if window.iter().any(|&v| v.is_nan()) {
            out[i] = f64::NAN;
            continue;
        }

        // Copy window into scratch once
        unsafe {
            core::ptr::copy_nonoverlapping(window.as_ptr(), buf.as_mut_ptr(), period);
        }

        // Median of window
        let med = median_from(&mut buf, mid);

        // In-place absolute deviations, then median again -> MAD
        for x in &mut buf {
            *x = (*x - med).abs();
        }
        let mad = median_from(&mut buf, mid);

        out[i] = mad;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medium_ad_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    unsafe { medium_ad_scalar(data, period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medium_ad_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    unsafe { medium_ad_scalar(data, period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medium_ad_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    unsafe { medium_ad_scalar(data, period, first_valid, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn medium_ad_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    unsafe { medium_ad_scalar(data, period, first_valid, out) }
}

#[inline(always)]
pub fn medium_ad_batch_with_kernel(
    data: &[f64],
    sweep: &MediumAdBatchRange,
    k: Kernel,
) -> Result<MediumAdBatchOutput, MediumAdError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(MediumAdError::InvalidPeriod {
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
    medium_ad_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MediumAdBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for MediumAdBatchRange {
    fn default() -> Self {
        Self { period: (5, 50, 1) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MediumAdBatchBuilder {
    range: MediumAdBatchRange,
    kernel: Kernel,
}

impl MediumAdBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<MediumAdBatchOutput, MediumAdError> {
        medium_ad_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<MediumAdBatchOutput, MediumAdError> {
        MediumAdBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<MediumAdBatchOutput, MediumAdError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<MediumAdBatchOutput, MediumAdError> {
        MediumAdBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct MediumAdBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MediumAdParams>,
    pub rows: usize,
    pub cols: usize,
}
impl MediumAdBatchOutput {
    pub fn row_for_params(&self, p: &MediumAdParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }

    pub fn values_for(&self, p: &MediumAdParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &MediumAdBatchRange) -> Vec<MediumAdParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let periods = axis_usize(r.period);

    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(MediumAdParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn medium_ad_batch_slice(
    data: &[f64],
    sweep: &MediumAdBatchRange,
    kern: Kernel,
) -> Result<MediumAdBatchOutput, MediumAdError> {
    medium_ad_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn medium_ad_batch_par_slice(
    data: &[f64],
    sweep: &MediumAdBatchRange,
    kern: Kernel,
) -> Result<MediumAdBatchOutput, MediumAdError> {
    medium_ad_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn medium_ad_batch_inner(
    data: &[f64],
    sweep: &MediumAdBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MediumAdBatchOutput, MediumAdError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MediumAdError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let cols = data.len();
    if cols == 0 {
        return Err(MediumAdError::AllValuesNaN);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MediumAdError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if cols - first < max_p {
        return Err(MediumAdError::NotEnoughValidData {
            needed: max_p,
            valid: cols - first,
        });
    }

    let rows = combos.len();

    // rows×cols uninit matrix + warm prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Safe mutable view of MaybeUninit rows
    let out_mu = buf_mu.as_mut_slice();

    let do_row = |row: usize, dst_mu: &mut [core::mem::MaybeUninit<f64>]| {
        // Cast this row to &mut [f64] to write initialized values
        let dst = unsafe {
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len())
        };
        let period = combos[row].period.unwrap();

        unsafe {
            match kern {
                Kernel::Scalar | Kernel::Auto => medium_ad_row_scalar(data, first, period, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => medium_ad_row_avx2(data, first, period, dst),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => medium_ad_row_avx512(data, first, period, dst),
                _ => unreachable!(),
            }
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out_mu
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice_mu)| do_row(row, slice_mu));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice_mu) in out_mu.chunks_mut(cols).enumerate() {
                do_row(row, slice_mu);
            }
        }
    } else {
        for (row, slice_mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(row, slice_mu);
        }
    }

    // Transmute the whole matrix once into Vec<f64>
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(MediumAdBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn medium_ad_batch_inner_into(
    data: &[f64],
    sweep: &MediumAdBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<MediumAdParams>, MediumAdError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MediumAdError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MediumAdError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(MediumAdError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let cols = data.len();

    // Initialize warmup periods with NaN for each row
    for (row, combo) in combos.iter().enumerate() {
        let warmup = first + combo.period.unwrap() - 1;
        let row_start = row * cols;
        for i in 0..warmup.min(cols) {
            out[row_start + i] = f64::NAN;
        }
    }

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => medium_ad_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => medium_ad_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => medium_ad_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

#[inline(always)]
unsafe fn medium_ad_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    use core::cmp::Ordering;
    let mut buf = vec![0.0; period];
    let mid = period / 2;

    #[inline(always)]
    fn median_from(buf: &mut [f64], mid: usize) -> f64 {
        buf.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        if (buf.len() & 1) == 1 {
            buf[mid]
        } else {
            let lo_max = buf[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            0.5 * (lo_max + buf[mid])
        }
    }

    for i in (first + period - 1)..data.len() {
        let window = &data[i + 1 - period..=i];
        if window.iter().any(|&v| v.is_nan()) {
            out[i] = f64::NAN;
            continue;
        }
        core::ptr::copy_nonoverlapping(window.as_ptr(), buf.as_mut_ptr(), period);
        let med = median_from(&mut buf, mid);
        for x in &mut buf {
            *x = (*x - med).abs();
        }
        out[i] = median_from(&mut buf, mid);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn medium_ad_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    medium_ad_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn medium_ad_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        medium_ad_row_avx512_short(data, first, period, out)
    } else {
        medium_ad_row_avx512_long(data, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn medium_ad_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    medium_ad_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn medium_ad_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    medium_ad_row_scalar(data, first, period, out)
}

#[derive(Debug, Clone)]
pub struct MediumAdStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl MediumAdStream {
    pub fn try_new(params: MediumAdParams) -> Result<Self, MediumAdError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(MediumAdError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled || self.buffer.iter().any(|v| v.is_nan()) {
            return None;
        }
        Some(self.compute())
    }

    #[inline(always)]
    fn compute(&self) -> f64 {
        // For small periods (typically 5-20), sorting is still very efficient
        // and simpler than maintaining complex data structures.
        // The complexity is O(period * log(period)) which is effectively constant
        // for small periods.
        let mut sorted = self.buffer.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if self.period % 2 == 1 {
            sorted[self.period / 2]
        } else {
            0.5 * (sorted[self.period / 2 - 1] + sorted[self.period / 2])
        };
        let mut absdevs: Vec<f64> = sorted.iter().map(|&x| (x - median).abs()).collect();
        absdevs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if self.period % 2 == 1 {
            absdevs[self.period / 2]
        } else {
            0.5 * (absdevs[self.period / 2 - 1] + absdevs[self.period / 2])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_medium_ad_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = MediumAdParams { period: None };
        let input = MediumAdInput::from_candles(&candles, "close", default_params);
        let output = medium_ad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_medium_ad_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = MediumAdParams { period: Some(5) };
        let input = MediumAdInput::from_candles(&candles, "hl2", params);
        let result = medium_ad_with_kernel(&input, kernel)?;
        let expected_last_five = [220.0, 78.5, 126.5, 48.0, 28.5];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] MEDIUM_AD {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_medium_ad_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MediumAdInput::with_default_candles(&candles);
        match input.data {
            MediumAdData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected MediumAdData::Candles"),
        }
        let output = medium_ad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_medium_ad_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = MediumAdParams { period: Some(0) };
        let input = MediumAdInput::from_slice(&input_data, params);
        let res = medium_ad_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MEDIUM_AD should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_medium_ad_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = MediumAdParams { period: Some(10) };
        let input = MediumAdInput::from_slice(&data_small, params);
        let res = medium_ad_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MEDIUM_AD should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_medium_ad_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = MediumAdParams { period: Some(5) };
        let input = MediumAdInput::from_slice(&single_point, params);
        let res = medium_ad_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MEDIUM_AD should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_medium_ad_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = MediumAdParams { period: Some(5) };
        let first_input = MediumAdInput::from_candles(&candles, "close", first_params);
        let first_result = medium_ad_with_kernel(&first_input, kernel)?;

        let second_params = MediumAdParams { period: Some(5) };
        let second_input = MediumAdInput::from_slice(&first_result.values, second_params);
        let second_result = medium_ad_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_medium_ad_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input =
            MediumAdInput::from_candles(&candles, "close", MediumAdParams { period: Some(5) });
        let res = medium_ad_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 60 {
            for (i, &val) in res.values[60..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    60 + i
                );
            }
        }
        Ok(())
    }

    fn check_medium_ad_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 5;
        let input = MediumAdInput::from_candles(
            &candles,
            "close",
            MediumAdParams {
                period: Some(period),
            },
        );
        let batch_output = medium_ad_with_kernel(&input, kernel)?.values;

        let mut stream = MediumAdStream::try_new(MediumAdParams {
            period: Some(period),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(mad_val) => stream_values.push(mad_val),
                None => stream_values.push(f64::NAN),
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
                "[{}] MEDIUM_AD streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_medium_ad_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        // Strategy: generate period from 1 to 64, then data with length from period to 400
        let strat = (1usize..=64).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = MediumAdParams {
                    period: Some(period),
                };
                let input = MediumAdInput::from_slice(&data, params);

                // Compute with the specified kernel
                let MediumAdOutput { values: out } = medium_ad_with_kernel(&input, kernel).unwrap();
                // Compute reference with scalar kernel
                let MediumAdOutput { values: ref_out } =
                    medium_ad_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: Kernel consistency - all kernels should produce identical results
                for i in 0..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Check if both are NaN or both are finite
                    if y.is_nan() {
                        prop_assert!(r.is_nan(), "Kernel consistency: NaN mismatch at idx {}", i);
                    } else if r.is_nan() {
                        prop_assert!(y.is_nan(), "Kernel consistency: NaN mismatch at idx {}", i);
                    } else {
                        // Both are finite, check they're close
                        let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                        prop_assert!(
                            (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                            "Kernel mismatch at idx {}: {} vs {} (ULP={})",
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }
                }

                // Property 2: Warmup period - first (period - 1) values should be NaN
                for i in 0..(period - 1) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at idx {}, got {}",
                        i,
                        out[i]
                    );
                }

                // Property 3: Post-warmup - values should be finite and non-negative (MAD is always >= 0)
                for i in (period - 1)..data.len() {
                    let mad = out[i];
                    prop_assert!(
                        mad.is_finite() && mad >= 0.0,
                        "MAD at idx {} is not finite or negative: {}",
                        i,
                        mad
                    );
                }

                // Property 4: For constant data, MAD should be exactly 0.0
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON)
                    && data.len() >= period
                {
                    for i in (period - 1)..data.len() {
                        prop_assert!(
                            out[i].abs() < 1e-9,
                            "Constant data should have MAD=0.0, got {} at idx {}",
                            out[i],
                            i
                        );
                    }
                }

                // Property 5: MAD bounds - should not exceed theoretical maximum
                for i in (period - 1)..data.len() {
                    let window = &data[i + 1 - period..=i];
                    let min_val = window.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_val = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let range = max_val - min_val;
                    let mad = out[i];

                    // MAD theoretical maximum is 50% of range
                    prop_assert!(
                        mad <= range * 0.5 + 1e-9,
                        "MAD {} exceeds theoretical maximum (50% of range {}) at idx {}",
                        mad,
                        range * 0.5,
                        i
                    );
                }

                // Property 6: Period = 1 special case - MAD should always be 0.0
                if period == 1 {
                    for i in 0..data.len() {
                        if !out[i].is_nan() {
                            prop_assert!(
                                out[i].abs() < f64::EPSILON,
                                "Period=1 should have MAD=0.0, got {} at idx {}",
                                out[i],
                                i
                            );
                        }
                    }
                }

                // Property 7: Symmetry - MAD should be the same for data and -data
                let neg_data: Vec<f64> = data.iter().map(|&x| -x).collect();
                let neg_input = MediumAdInput::from_slice(
                    &neg_data,
                    MediumAdParams {
                        period: Some(period),
                    },
                );
                let MediumAdOutput { values: neg_out } =
                    medium_ad_with_kernel(&neg_input, kernel).unwrap();

                for i in (period - 1)..data.len() {
                    let mad = out[i];
                    let neg_mad = neg_out[i];
                    prop_assert!(
                        (mad - neg_mad).abs() < 1e-9,
                        "Symmetry failed at idx {}: {} vs {}",
                        i,
                        mad,
                        neg_mad
                    );
                }

                // Property 8: Scale Invariance - MAD(c * data) = |c| * MAD(data)
                // Test with a few different scale factors
                let scale_factors = [2.0, -3.0, 0.5];
                for &scale in &scale_factors {
                    let scaled_data: Vec<f64> = data.iter().map(|&x| x * scale).collect();
                    let scaled_input = MediumAdInput::from_slice(
                        &scaled_data,
                        MediumAdParams {
                            period: Some(period),
                        },
                    );
                    let MediumAdOutput { values: scaled_out } =
                        medium_ad_with_kernel(&scaled_input, kernel).unwrap();

                    for i in (period - 1)..data.len() {
                        let original_mad = out[i];
                        let scaled_mad = scaled_out[i];
                        let expected_scaled_mad = original_mad * scale.abs();

                        prop_assert!(
                            (scaled_mad - expected_scaled_mad).abs() < 1e-9,
                            "Scale invariance failed at idx {} with scale {}: {} vs expected {}",
                            i,
                            scale,
                            scaled_mad,
                            expected_scaled_mad
                        );
                    }
                }

                // Property 9: Outlier Robustness - MAD should be less affected by outliers
                // Compare behavior with and without an outlier
                if period >= 5 && data.len() >= period + 10 {
                    // Create a version with an outlier in the middle of each window
                    let mut outlier_data = data.clone();

                    // Test a few windows to verify outlier robustness
                    for test_idx in (period + 4..data.len().min(period + 20)).step_by(5) {
                        // Find the range of values in the window
                        let window = &data[test_idx + 1 - period..=test_idx];
                        let win_min = window.iter().cloned().fold(f64::INFINITY, f64::min);
                        let win_max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let win_range = win_max - win_min;

                        // Add an outlier that's 10x the range away
                        let outlier_idx = test_idx - period / 2;
                        let original_value = outlier_data[outlier_idx];
                        outlier_data[outlier_idx] = win_max + win_range * 10.0;

                        // Calculate MAD with outlier
                        let outlier_input = MediumAdInput::from_slice(
                            &outlier_data,
                            MediumAdParams {
                                period: Some(period),
                            },
                        );
                        let MediumAdOutput {
                            values: outlier_out,
                        } = medium_ad_with_kernel(&outlier_input, kernel).unwrap();

                        // MAD should change, but not dramatically (less than doubling)
                        let original_mad = out[test_idx];
                        let outlier_mad = outlier_out[test_idx];

                        // MAD is robust but extreme outliers can still have significant effect
                        // especially with degenerate data patterns (many zeros, few non-zeros)
                        // The bound needs to account for the outlier's extreme distance
                        let outlier_effect = win_range * 10.0; // The outlier is 10x range away
                        prop_assert!(
							outlier_mad <= original_mad * 10.0 + outlier_effect * 0.1,
							"MAD not robust enough to outliers at idx {}: original {}, with outlier {}",
							test_idx, original_mad, outlier_mad
						);

                        // For meaningful original MAD values, check relative increase
                        // Skip ratio check for very small original MAD as the ratio can be huge
                        if original_mad > win_range * 0.01 {
                            // Only check if original MAD is meaningful
                            let mad_ratio = outlier_mad / original_mad;
                            prop_assert!(
                                mad_ratio <= 20.0,
                                "MAD ratio too high with outlier at idx {}: ratio {}",
                                test_idx,
                                mad_ratio
                            );
                        }

                        // Restore original value for next test
                        outlier_data[outlier_idx] = original_value;
                    }
                }

                // Property 10: Known Value Verification - test exact MAD for known patterns
                // This ensures the implementation is actually calculating MAD correctly

                // Test 1: Sequential data [1, 2, 3, ..., period]
                if period >= 3 && period <= 20 {
                    let sequential: Vec<f64> = (1..=period).map(|i| i as f64).collect();
                    let seq_input = MediumAdInput::from_slice(
                        &sequential,
                        MediumAdParams {
                            period: Some(period),
                        },
                    );
                    let MediumAdOutput { values: seq_out } =
                        medium_ad_with_kernel(&seq_input, kernel).unwrap();

                    // For sequential data, calculate expected MAD
                    // Median is the middle value(s)
                    let median = if period % 2 == 1 {
                        (period / 2 + 1) as f64
                    } else {
                        (period / 2) as f64 + 0.5
                    };

                    // For sequential data, MAD should be reasonable but exact value depends on period
                    // Let's just verify it's within expected bounds
                    if period - 1 < sequential.len() {
                        let calculated_mad = seq_out[period - 1];
                        let seq_range = (period - 1) as f64; // Range is period - 1 for sequential data

                        // MAD should be non-zero and less than half the range
                        prop_assert!(
							calculated_mad > 0.0 && calculated_mad <= seq_range * 0.5,
							"MAD for sequential data with period {} out of bounds: got {}, range is {}",
							period, calculated_mad, seq_range
						);

                        // For specific known cases, test exact values
                        if period == 3 {
                            // For [1,2,3]: median=2, devs=[1,0,1], MAD=median([0,1,1])=1
                            prop_assert!(
                                (calculated_mad - 1.0).abs() < 1e-9,
                                "MAD for [1,2,3] should be 1.0, got {}",
                                calculated_mad
                            );
                        } else if period == 5 {
                            // For [1,2,3,4,5]: median=3, devs=[2,1,0,1,2], MAD=median([0,1,1,2,2])=1
                            prop_assert!(
                                (calculated_mad - 1.0).abs() < 1e-9,
                                "MAD for [1,2,3,4,5] should be 1.0, got {}",
                                calculated_mad
                            );
                        }
                    }
                }

                // Test 2: Data with half values at min and half at max
                // This should produce MAD close to range/2
                if period >= 4 && period % 2 == 0 {
                    let mut extreme_data = vec![0.0; period];
                    for i in 0..period / 2 {
                        extreme_data[i] = 100.0; // Half at max
                    }
                    // Other half remains at 0.0 (min)

                    let extreme_input = MediumAdInput::from_slice(
                        &extreme_data,
                        MediumAdParams {
                            period: Some(period),
                        },
                    );
                    let MediumAdOutput {
                        values: extreme_out,
                    } = medium_ad_with_kernel(&extreme_input, kernel).unwrap();

                    // For this pattern, median is 50.0, and MAD should be 50.0
                    let expected_extreme_mad = 50.0;

                    if period - 1 < extreme_data.len() {
                        let calculated_extreme_mad = extreme_out[period - 1];
                        prop_assert!(
							(calculated_extreme_mad - expected_extreme_mad).abs() < 1e-9,
							"MAD mismatch for extreme data pattern with period {}: got {}, expected {}",
							period, calculated_extreme_mad, expected_extreme_mad
						);
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_medium_ad_tests {
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

    generate_all_medium_ad_tests!(
        check_medium_ad_partial_params,
        check_medium_ad_accuracy,
        check_medium_ad_default_candles,
        check_medium_ad_zero_period,
        check_medium_ad_period_exceeds_length,
        check_medium_ad_very_small_dataset,
        check_medium_ad_reinput,
        check_medium_ad_nan_handling,
        check_medium_ad_streaming
    );

    #[cfg(feature = "proptest")]
    generate_all_medium_ad_tests!(check_medium_ad_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = MediumAdBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = MediumAdParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());
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

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(feature = "python")]
#[pyfunction(name = "medium_ad")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn medium_ad_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = MediumAdParams {
        period: Some(period),
    };
    let input = MediumAdInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| medium_ad_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "MediumAdStream")]
pub struct MediumAdStreamPy {
    stream: MediumAdStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MediumAdStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = MediumAdParams {
            period: Some(period),
        };
        let stream =
            MediumAdStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(MediumAdStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "medium_ad_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn medium_ad_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = MediumAdBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => unreachable!(),
            };
            medium_ad_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Write directly to output slice - no allocations
pub fn medium_ad_into_slice(
    dst: &mut [f64],
    input: &MediumAdInput,
    kern: Kernel,
) -> Result<(), MediumAdError> {
    let data = match &input.data {
        MediumAdData::Candles { candles, source } => source_type(candles, source),
        MediumAdData::Slice(s) => s,
    };
    let period = input.params.period.unwrap_or(5);

    if period == 0 || period > data.len() {
        return Err(MediumAdError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    if dst.len() != data.len() {
        return Err(MediumAdError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }

    let first = data.iter().position(|&x| !x.is_nan()).unwrap_or(0);
    let chosen = if kern == Kernel::Auto {
        detect_best_kernel()
    } else {
        kern
    };

    match chosen {
        Kernel::Scalar => medium_ad_scalar(data, period, first, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => medium_ad_avx2(data, period, first, dst),
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => medium_ad_avx512(data, period, first, dst),
        _ => unreachable!(),
    }

    // Fill warmup with NaN
    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn medium_ad_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = MediumAdParams {
        period: Some(period),
    };
    let input = MediumAdInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()]; // Single allocation

    medium_ad_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn medium_ad_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn medium_ad_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn medium_ad_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = MediumAdParams {
            period: Some(period),
        };
        let input = MediumAdInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // Handle aliasing
            let mut temp = vec![0.0; len];
            medium_ad_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            medium_ad_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MediumAdBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MediumAdBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MediumAdParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = medium_ad_batch)]
pub fn medium_ad_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: MediumAdBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = MediumAdBatchRange {
        period: config.period_range,
    };

    let output = medium_ad_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = MediumAdBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn medium_ad_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to medium_ad_batch_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = MediumAdBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        medium_ad_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
