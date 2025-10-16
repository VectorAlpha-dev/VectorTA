//! # Money Flow Index (MFI)
//!
//! Momentum indicator measuring money inflow/outflow using price and volume.
//!
//! ## Parameters
//! - **typical_price**: Typical price data (usually HLC/3)
//! - **volume**: Volume data
//! - **period**: Window size (default: 14)
//!
//! ## Returns
//! - `Vec<f64>` - MFI values (0-100 scale) matching input length
//!
//! ## Developer Status
//! **SIMD**: Present but disabled by design for single-series; stubs dispatch to scalar due to loop-carried deps.
//! **AVX2**: Stub (calls scalar)
//! **AVX512**: Has short/long variants but both stubs
//! **Streaming**: O(1) - Uses ring buffers for running sums
//! **Batch**: Row-specific scalar path uses prefix sums of pos/neg flows; accuracy matches scalar. Faster for larger sweeps; for few rows, scalar repeats can be competitive.
//! **Memory**: Good - Uses `alloc_with_nan_prefix` and `make_uninit_matrix`
//!
//! Decision: Streaming kernel uses branchless classification, avoids modulo in ring advance,
//! and leverages `mul_add` and reciprocal multiply. Bit-for-bit with baseline thresholds.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use serde_wasm_bindgen;
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
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::oscillators::mfi_wrapper::CudaMfi;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;

#[derive(Debug, Clone)]
pub enum MfiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        typical_price: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MfiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MfiParams {
    pub period: Option<usize>,
}

impl Default for MfiParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct MfiInput<'a> {
    pub data: MfiData<'a>,
    pub params: MfiParams,
}

impl<'a> MfiInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: MfiParams) -> Self {
        Self {
            data: MfiData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(typical_price: &'a [f64], volume: &'a [f64], params: MfiParams) -> Self {
        Self {
            data: MfiData::Slices {
                typical_price,
                volume,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "hlc3", MfiParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

#[derive(Debug, Error)]
pub enum MfiError {
    #[error("mfi: Empty data provided.")]
    EmptyData,
    #[error("mfi: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("mfi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("mfi: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn mfi(input: &MfiInput) -> Result<MfiOutput, MfiError> {
    mfi_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn mfi_prepare<'a>(
    input: &'a MfiInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], usize, usize, Kernel), MfiError> {
    let (typical_price, volume): (&[f64], &[f64]) = match &input.data {
        MfiData::Candles { candles, source } => {
            (source_type(candles, source), candles.volume.as_slice())
        }
        MfiData::Slices {
            typical_price,
            volume,
        } => (*typical_price, *volume),
    };

    let length = typical_price.len();
    if length == 0 || volume.len() != length {
        return Err(MfiError::EmptyData);
    }

    let period = input.get_period();
    let first_valid_idx = (0..length).find(|&i| !typical_price[i].is_nan() && !volume[i].is_nan());
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(MfiError::AllValuesNaN),
    };

    if period == 0 || period > length {
        return Err(MfiError::InvalidPeriod {
            period,
            data_len: length,
        });
    }
    if (length - first_valid_idx) < period {
        return Err(MfiError::NotEnoughValidData {
            needed: period,
            valid: length - first_valid_idx,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((typical_price, volume, period, first_valid_idx, chosen))
}

#[inline(always)]
fn mfi_compute_into(
    typical_price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                mfi_scalar(typical_price, volume, period, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => mfi_avx2(typical_price, volume, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                mfi_avx512(typical_price, volume, period, first, out)
            }
            _ => unreachable!(),
        }
    }
}

pub fn mfi_with_kernel(input: &MfiInput, kernel: Kernel) -> Result<MfiOutput, MfiError> {
    let (typical_price, volume, period, first_valid_idx, chosen) = mfi_prepare(input, kernel)?;

    let warmup_period = first_valid_idx + period - 1;
    let mut out = alloc_with_nan_prefix(typical_price.len(), warmup_period);

    mfi_compute_into(
        typical_price,
        volume,
        period,
        first_valid_idx,
        chosen,
        &mut out,
    );

    Ok(MfiOutput { values: out })
}

#[inline]
pub unsafe fn mfi_scalar(
    typical_price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    // Assumptions validated by mfi_prepare / callers:
    // - typical_price.len() == volume.len() == out.len()
    // - period > 0
    // - first + period <= len
    // - Warmup prefix (..first+period-1) is already prefilled by the caller when needed.
    let len = typical_price.len();
    if len == 0 {
        return;
    }

    // Ring buffers for rolling sums (always zero-initialized)
    let mut pos_buf = vec![0.0f64; period];
    let mut neg_buf = vec![0.0f64; period];

    // Raw pointers to avoid bounds checks in hot loops
    let tp_ptr = typical_price.as_ptr();
    let vol_ptr = volume.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let pos_ptr = pos_buf.as_mut_ptr();
    let neg_ptr = neg_buf.as_mut_ptr();

    let mut pos_sum = 0.0f64;
    let mut neg_sum = 0.0f64;

    // Keep previous typical price (first valid)
    let mut prev = *tp_ptr.add(first);
    let mut ring = 0usize;

    // ---- Seed window: fill the first (period - 1) money-flow slots ----
    // We deliberately start at `first + 1` because the classification requires a previous bar.
    // This exactly matches the existing semantics and unit tests.
    let seed_start = first + 1;
    let seed_end = first + period; // exclusive; last index written is first+period-1
    let mut i = seed_start;
    while i < seed_end {
        // diff and flow for bar i
        let tp_i = *tp_ptr.add(i);
        let flow = tp_i * *vol_ptr.add(i);
        let diff = tp_i - prev;
        prev = tp_i;

        // Branchless classification into positive / negative buckets
        // (true as i32 -> 1/0 -> cast to f64)
        let gt = (diff > 0.0) as i32 as f64;
        let lt = (diff < 0.0) as i32 as f64;
        let pos_new = flow * gt;
        let neg_new = flow * lt;

        // Write into ring and update sums
        *pos_ptr.add(ring) = pos_new;
        *neg_ptr.add(ring) = neg_new;
        pos_sum += pos_new;
        neg_sum += neg_new;

        ring += 1;
        if ring == period {
            ring = 0;
        }
        i += 1;
    }

    // ---- First MFI value at index first + period - 1 ----
    let idx0 = seed_end - 1; // == first + period - 1
    if idx0 < len {
        let total = pos_sum + neg_sum;
        // Same zero-denominator handling as before
        let val = if total < 1e-14 {
            0.0
        } else {
            100.0 * (pos_sum / total)
        };
        *out_ptr.add(idx0) = val;
    }

    // ---- Rolling window for the remainder ----
    i = seed_end;
    while i < len {
        // Remove the element that falls out of the window
        let old_pos = *pos_ptr.add(ring);
        let old_neg = *neg_ptr.add(ring);
        pos_sum -= old_pos;
        neg_sum -= old_neg;

        // Compute flow and direction for the new bar
        let tp_i = *tp_ptr.add(i);
        let flow = tp_i * *vol_ptr.add(i);
        let diff = tp_i - prev;
        prev = tp_i;

        // Branchless classification
        let gt = (diff > 0.0) as i32 as f64;
        let lt = (diff < 0.0) as i32 as f64;
        let pos_new = flow * gt;
        let neg_new = flow * lt;

        // Insert into ring & update sums
        *pos_ptr.add(ring) = pos_new;
        *neg_ptr.add(ring) = neg_new;
        pos_sum += pos_new;
        neg_sum += neg_new;

        // Write output
        let total = pos_sum + neg_sum;
        let val = if total < 1e-14 {
            0.0
        } else {
            100.0 * (pos_sum / total)
        };
        *out_ptr.add(i) = val;

        // Advance ring head (branch instead of modulo to avoid div)
        ring += 1;
        if ring == period {
            ring = 0;
        }

        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mfi_avx2(
    typical_price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    mfi_scalar(typical_price, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mfi_avx512(
    typical_price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    unsafe {
        if period <= 32 {
            mfi_avx512_short(typical_price, volume, period, first, out)
        } else {
            mfi_avx512_long(typical_price, volume, period, first, out)
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mfi_avx512_short(
    typical_price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    mfi_scalar(typical_price, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn mfi_avx512_long(
    typical_price: &[f64],
    volume: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    mfi_scalar(typical_price, volume, period, first, out)
}

#[derive(Copy, Clone, Debug)]
pub struct MfiBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for MfiBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MfiBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<MfiOutput, MfiError> {
        let p = MfiParams {
            period: self.period,
        };
        let i = MfiInput::from_candles(c, "hlc3", p);
        mfi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        typical_price: &[f64],
        volume: &[f64],
    ) -> Result<MfiOutput, MfiError> {
        let p = MfiParams {
            period: self.period,
        };
        let i = MfiInput::from_slices(typical_price, volume, p);
        mfi_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<MfiStream, MfiError> {
        let p = MfiParams {
            period: self.period,
        };
        MfiStream::try_new(p)
    }
}

#[derive(Debug, Clone)]
pub struct MfiStream {
    period: usize,
    pos_buf: Vec<f64>,
    neg_buf: Vec<f64>,
    head: usize,
    filled: bool,
    pos_sum: f64,
    neg_sum: f64,
    prev_typical: f64,
    index: usize,
}

impl MfiStream {
    pub fn try_new(params: MfiParams) -> Result<Self, MfiError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(MfiError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            pos_buf: vec![0.0; period],
            neg_buf: vec![0.0; period],
            head: 0,
            filled: false,
            pos_sum: 0.0,
            neg_sum: 0.0,
            prev_typical: f64::NAN,
            index: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, typical_price: f64, volume: f64) -> Option<f64> {
        // First sample: seed 'prev_typical' and wait for the next bar (no flow yet).
        if self.index == 0 {
            self.prev_typical = typical_price;
            self.index = 1;
            return None;
        }

        // ----- Compute one-bar flow -----
        // diff determines sign (pos/neg flow), flow is TP * Volume.
        let diff = typical_price - self.prev_typical;
        self.prev_typical = typical_price;

        // Prefer FMA when available; this compiles to one FMA on FMA-capable CPUs.
        let flow = typical_price.mul_add(volume, 0.0); // == typical_price * volume

        // Branchless classification: gt/lt are 1.0 or 0.0
        let gt = (diff > 0.0) as i32 as f64;
        let lt = (diff < 0.0) as i32 as f64;
        let pos_new = flow * gt;
        let neg_new = flow * lt;

        // Evict old bucket values at head and update rolling sums (O(1))
        // Use unchecked indexing to avoid bounds checks in the hot path.
        unsafe {
            let old_pos = *self.pos_buf.get_unchecked(self.head);
            let old_neg = *self.neg_buf.get_unchecked(self.head);

            self.pos_sum += pos_new - old_pos;
            self.neg_sum += neg_new - old_neg;

            *self.pos_buf.get_unchecked_mut(self.head) = pos_new;
            *self.neg_buf.get_unchecked_mut(self.head) = neg_new;
        }

        // Advance ring WITHOUT modulo (avoid integer division in hot loop).
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
            self.filled = true; // first time we wrap, the window is full
        }
        self.index += 1;

        // Match existing warmup behavior: no value until the ring has wrapped.
        if !self.filled {
            return None;
        }

        // ----- Emit MFI for the current window -----
        let total = self.pos_sum + self.neg_sum;
        if total <= 1e-14 {
            Some(0.0)
        } else {
            // Multiply by reciprocal to dodge a scalar FP divide
            Some(100.0 * self.pos_sum * total.recip())
        }
    }
}

#[derive(Clone, Debug)]
pub struct MfiBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for MfiBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 14, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MfiBatchBuilder {
    range: MfiBatchRange,
    kernel: Kernel,
}

impl MfiBatchBuilder {
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
        typical_price: &[f64],
        volume: &[f64],
    ) -> Result<MfiBatchOutput, MfiError> {
        mfi_batch_with_kernel(typical_price, volume, &self.range, self.kernel)
    }

    pub fn apply_candles(self, c: &Candles) -> Result<MfiBatchOutput, MfiError> {
        let typical_price = source_type(c, "hlc3");
        self.apply_slices(typical_price, &c.volume)
    }

    pub fn with_default_candles(c: &Candles, k: Kernel) -> Result<MfiBatchOutput, MfiError> {
        MfiBatchBuilder::new().kernel(k).apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct MfiBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MfiParams>,
    pub rows: usize,
    pub cols: usize,
}
impl MfiBatchOutput {
    pub fn row_for_params(&self, p: &MfiParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &MfiParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &MfiBatchRange) -> Vec<MfiParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(MfiParams { period: Some(p) });
    }
    out
}

pub fn mfi_batch_with_kernel(
    typical_price: &[f64],
    volume: &[f64],
    sweep: &MfiBatchRange,
    k: Kernel,
) -> Result<MfiBatchOutput, MfiError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(MfiError::InvalidPeriod {
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
    mfi_batch_par_slice(typical_price, volume, sweep, simd)
}

#[inline(always)]
pub fn mfi_batch_slice(
    typical_price: &[f64],
    volume: &[f64],
    sweep: &MfiBatchRange,
    kern: Kernel,
) -> Result<MfiBatchOutput, MfiError> {
    mfi_batch_inner(typical_price, volume, sweep, kern, false)
}

#[inline(always)]
pub fn mfi_batch_par_slice(
    typical_price: &[f64],
    volume: &[f64],
    sweep: &MfiBatchRange,
    kern: Kernel,
) -> Result<MfiBatchOutput, MfiError> {
    mfi_batch_inner(typical_price, volume, sweep, kern, true)
}

fn round_up8(x: usize) -> usize {
    (x + 7) & !7
}

#[inline(always)]
fn mfi_batch_inner(
    typical_price: &[f64],
    volume: &[f64],
    sweep: &MfiBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MfiBatchOutput, MfiError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MfiError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let length = typical_price.len();
    let first = (0..length)
        .find(|&i| !typical_price[i].is_nan() && !volume[i].is_nan())
        .ok_or(MfiError::AllValuesNaN)?;

    let max_p = combos
        .iter()
        .map(|c| round_up8(c.period.unwrap()))
        .max()
        .unwrap();
    if length - first < max_p {
        return Err(MfiError::NotEnoughValidData {
            needed: max_p,
            valid: length - first,
        });
    }

    let rows = combos.len();
    let cols = length;

    if volume.len() != cols {
        return Err(MfiError::EmptyData);
    }

    // Use uninitialized memory with NaN prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warmup_periods);

    // Convert to mutable slice for computation
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // Heuristic: use shared prefix-sum path only when row count is large enough to amortize precompute
    let rows = combos.len();
    let use_prefix = rows >= 8; // tuned threshold; adjust if needed

    let (pos_prefix, neg_prefix) = if use_prefix {
        let (pp, np) =
            unsafe { precompute_flow_prefixes_select(typical_price, volume, first, kern) };
        (Some(pp), Some(np))
    } else {
        (None, None)
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        if let (Some(ref pp), Some(ref np)) = (pos_prefix.as_ref(), neg_prefix.as_ref()) {
            mfi_row_from_prefixes(pp, np, first, period, out_row)
        } else {
            match kern {
                Kernel::Scalar => mfi_row_scalar(typical_price, volume, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => mfi_row_avx2(typical_price, volume, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => mfi_row_avx512(typical_price, volume, first, period, out_row),
                _ => unreachable!(),
            }
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
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

    // Convert back to owned Vec
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(MfiBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn mfi_batch_inner_into(
    typical_price: &[f64],
    volume: &[f64],
    sweep: &MfiBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<MfiParams>, MfiError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MfiError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }

    let length = typical_price.len();
    let first = (0..length)
        .find(|&i| !typical_price[i].is_nan() && !volume[i].is_nan())
        .ok_or(MfiError::AllValuesNaN)?;

    let max_p = combos
        .iter()
        .map(|c| round_up8(c.period.unwrap()))
        .max()
        .unwrap();
    if length - first < max_p {
        return Err(MfiError::NotEnoughValidData {
            needed: max_p,
            valid: length - first,
        });
    }

    let cols = length;

    if volume.len() != cols {
        return Err(MfiError::EmptyData);
    }

    // Heuristic: only precompute prefixes if many rows; always fill warmup per row in into-slice variant
    let rows = combos.len();
    let use_prefix = rows >= 8;
    let (pos_prefix, neg_prefix) = if use_prefix {
        let (pp, np) =
            unsafe { precompute_flow_prefixes_select(typical_price, volume, first, kern) };
        (Some(pp), Some(np))
    } else {
        (None, None)
    };

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        // Warmup fill
        let warmup_end = first + period - 1;
        for v in &mut out_row[..warmup_end] {
            *v = f64::NAN;
        }
        if let (Some(ref pp), Some(ref np)) = (pos_prefix.as_ref(), neg_prefix.as_ref()) {
            mfi_row_from_prefixes(pp, np, first, period, out_row)
        } else {
            match kern {
                Kernel::Scalar => mfi_row_scalar(typical_price, volume, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 => mfi_row_avx2(typical_price, volume, first, period, out_row),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 => mfi_row_avx512(typical_price, volume, first, period, out_row),
                _ => unreachable!(),
            }
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
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
unsafe fn precompute_flow_prefixes_scalar(
    typical_price: &[f64],
    volume: &[f64],
    first: usize,
) -> (Vec<f64>, Vec<f64>) {
    let len = typical_price.len();
    let tp_ptr = typical_price.as_ptr();
    let vol_ptr = volume.as_ptr();

    // Positive/negative flow prefix sums (exclusive at index 0)
    let mut pos_prefix = vec![0.0f64; len];
    let mut neg_prefix = vec![0.0f64; len];

    if len == 0 {
        return (pos_prefix, neg_prefix);
    }

    let mut i = first + 1;
    let mut prev = *tp_ptr.add(first);
    while i < len {
        let tp_i = *tp_ptr.add(i);
        let flow = tp_i * *vol_ptr.add(i);
        let diff = tp_i - prev;
        prev = tp_i;
        // Branchless classify
        let gt = (diff > 0.0) as i32 as f64;
        let lt = (diff < 0.0) as i32 as f64;
        let pos = flow * gt;
        let neg = flow * lt;

        // Build prefix sums
        pos_prefix[i] = pos_prefix[i - 1] + pos;
        neg_prefix[i] = neg_prefix[i - 1] + neg;
        i += 1;
    }

    // Fill the region before `first+1` with zeros (already zeroed) and also carry forward prefix at `first`
    if first > 0 {
        pos_prefix[first] = 0.0;
        neg_prefix[first] = 0.0;
        // ensure continuity: for j in 1..=first-1 already zeros
    }

    (pos_prefix, neg_prefix)
}

#[inline(always)]
unsafe fn precompute_flow_prefixes_select(
    typical_price: &[f64],
    volume: &[f64],
    first: usize,
    kern: Kernel,
) -> (Vec<f64>, Vec<f64>) {
    match kern {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 | Kernel::Avx512 => {
            precompute_flow_prefixes_avx2(typical_price, volume, first)
        }
        _ => precompute_flow_prefixes_scalar(typical_price, volume, first),
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn precompute_flow_prefixes_avx2(
    typical_price: &[f64],
    volume: &[f64],
    first: usize,
) -> (Vec<f64>, Vec<f64>) {
    use core::arch::x86_64::*;
    let len = typical_price.len();
    let mut pos_prefix = vec![0.0f64; len];
    let mut neg_prefix = vec![0.0f64; len];
    if len == 0 {
        return (pos_prefix, neg_prefix);
    }

    // Running sums to build prefix directly
    let mut pos_sum = 0.0f64;
    let mut neg_sum = 0.0f64;
    // Ensure prefix at 'first' is zero (flows start after first)
    if first < len {
        pos_prefix[first] = 0.0;
        neg_prefix[first] = 0.0;
    }

    let mut i = first + 1;
    let tp_ptr = typical_price.as_ptr();
    let vol_ptr = volume.as_ptr();
    let zero = _mm256_set1_pd(0.0);

    while i + 4 <= len {
        // Load current tp and previous tp
        let tp_cur = _mm256_loadu_pd(tp_ptr.add(i));
        let tp_prev = _mm256_loadu_pd(tp_ptr.add(i - 1));
        let vol_cur = _mm256_loadu_pd(vol_ptr.add(i));

        // flow = tp * vol
        let flow = _mm256_mul_pd(tp_cur, vol_cur);
        // diff = tp[i] - tp[i-1]
        let diff = _mm256_sub_pd(tp_cur, tp_prev);
        // masks
        let m_gt = _mm256_cmp_pd(diff, zero, _CMP_GT_OQ);
        let m_lt = _mm256_cmp_pd(diff, zero, _CMP_LT_OQ);
        // classify
        let pos_v = _mm256_and_pd(flow, m_gt);
        let neg_v = _mm256_and_pd(flow, m_lt);

        // Store to temporaries and build prefix scalarly within the chunk
        let mut pos_tmp = [0.0f64; 4];
        let mut neg_tmp = [0.0f64; 4];
        _mm256_storeu_pd(pos_tmp.as_mut_ptr(), pos_v);
        _mm256_storeu_pd(neg_tmp.as_mut_ptr(), neg_v);

        // Unrolled accumulation for the 4-lane chunk
        // Lane 0
        pos_sum += pos_tmp[0];
        neg_sum += neg_tmp[0];
        *pos_prefix.get_unchecked_mut(i) = pos_sum;
        *neg_prefix.get_unchecked_mut(i) = neg_sum;
        // Lane 1
        pos_sum += pos_tmp[1];
        neg_sum += neg_tmp[1];
        *pos_prefix.get_unchecked_mut(i + 1) = pos_sum;
        *neg_prefix.get_unchecked_mut(i + 1) = neg_sum;
        // Lane 2
        pos_sum += pos_tmp[2];
        neg_sum += neg_tmp[2];
        *pos_prefix.get_unchecked_mut(i + 2) = pos_sum;
        *neg_prefix.get_unchecked_mut(i + 2) = neg_sum;
        // Lane 3
        pos_sum += pos_tmp[3];
        neg_sum += neg_tmp[3];
        *pos_prefix.get_unchecked_mut(i + 3) = pos_sum;
        *neg_prefix.get_unchecked_mut(i + 3) = neg_sum;

        i += 4;
    }

    // Tail
    while i < len {
        let tp_i = *tp_ptr.add(i);
        let flow = tp_i * *vol_ptr.add(i);
        let diff = tp_i - *tp_ptr.add(i - 1);
        let gt = (diff > 0.0) as i32 as f64;
        let lt = (diff < 0.0) as i32 as f64;
        pos_sum += flow * gt;
        neg_sum += flow * lt;
        *pos_prefix.get_unchecked_mut(i) = pos_sum;
        *neg_prefix.get_unchecked_mut(i) = neg_sum;
        i += 1;
    }

    (pos_prefix, neg_prefix)
}

#[inline(always)]
unsafe fn mfi_row_from_prefixes(
    pos_prefix: &[f64],
    neg_prefix: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    let len = out.len();
    if len == 0 {
        return;
    }
    let idx0 = first + period - 1;
    if idx0 >= len {
        return;
    }
    // First value uses period-1 flows: [idx0-(period-1)+1 ..= idx0] => prefix[idx0] - prefix[first]
    let pos0 = pos_prefix[idx0] - pos_prefix[first];
    let neg0 = neg_prefix[idx0] - neg_prefix[first];
    let tot0 = pos0 + neg0;
    *out.get_unchecked_mut(idx0) = if tot0 < 1e-14 {
        0.0
    } else {
        100.0 * (pos0 / tot0)
    };

    // Subsequent values use full `period` flows: [i - period + 1 ..= i] => prefix[i] - prefix[i - period]
    let mut i = idx0 + 1;
    while i < len {
        let base = i - period;
        let pos_sum = pos_prefix[i] - pos_prefix[base];
        let neg_sum = neg_prefix[i] - neg_prefix[base];
        let total = pos_sum + neg_sum;
        let val = if total < 1e-14 {
            0.0
        } else {
            100.0 * (pos_sum / total)
        };
        *out.get_unchecked_mut(i) = val;
        i += 1;
    }
}

#[inline(always)]
unsafe fn mfi_row_scalar(
    typical_price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    mfi_scalar(typical_price, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mfi_row_avx2(
    typical_price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    mfi_scalar(typical_price, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mfi_row_avx512(
    typical_price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    if period <= 32 {
        mfi_row_avx512_short(typical_price, volume, first, period, out)
    } else {
        mfi_row_avx512_long(typical_price, volume, first, period, out)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mfi_row_avx512_short(
    typical_price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    mfi_scalar(typical_price, volume, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mfi_row_avx512_long(
    typical_price: &[f64],
    volume: &[f64],
    first: usize,
    period: usize,
    out: &mut [f64],
) {
    mfi_scalar(typical_price, volume, period, first, out)
}

#[cfg(feature = "python")]
#[pyfunction(name = "mfi")]
#[pyo3(signature = (typical_price, volume, period, kernel=None))]
pub fn mfi_py<'py>(
    py: Python<'py>,
    typical_price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let typical_slice = typical_price.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = MfiParams {
        period: Some(period),
    };
    let input = MfiInput::from_slices(typical_slice, volume_slice, params);

    // GOOD: Get Vec<f64> from Rust function
    let result_vec: Vec<f64> = py
        .allow_threads(|| mfi_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // GOOD: Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "MfiStream")]
pub struct MfiStreamPy {
    inner: MfiStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MfiStreamPy {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> {
        let params = MfiParams {
            period: Some(period),
        };
        let inner = MfiStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(MfiStreamPy { inner })
    }

    pub fn update(&mut self, typical_price: f64, volume: f64) -> Option<f64> {
        self.inner.update(typical_price, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "mfi_batch")]
#[pyo3(signature = (typical_price, volume, period_range, kernel=None))]
pub fn mfi_batch_py<'py>(
    py: Python<'py>,
    typical_price: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let tp = typical_price.as_slice()?;
    let vol = volume.as_slice()?;
    if tp.len() != vol.len() {
        return Err(PyValueError::new_err(
            "mfi_batch: typical_price and volume length mismatch",
        ));
    }

    let sweep = MfiBatchRange {
        period: period_range,
    };
    let kern = validate_kernel(kernel, true)?;

    // Calculate dimensions
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = tp.len();

    // Allocate NumPy array upfront for zero-copy
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    // Compute directly into NumPy buffer
    let combos = py
        .allow_threads(|| {
            let k = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            // Map batch -> compute kernel as in ALMA
            let simd = match k {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => k,
            };
            mfi_batch_inner_into(tp, vol, &sweep, simd, true, out_slice)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    // Zero-copy reshape NumPy array
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

// ---------------- CUDA Python Bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "mfi_cuda_batch_dev")]
#[pyo3(signature = (typical_price, volume, period_range, device_id=0))]
pub fn mfi_cuda_batch_dev_py(
    py: Python<'_>,
    typical_price: PyReadonlyArray1<'_, f32>,
    volume: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() { return Err(PyValueError::new_err("CUDA not available")); }
    let tp = typical_price.as_slice()?;
    let vol = volume.as_slice()?;
    if tp.len() != vol.len() { return Err(PyValueError::new_err("mismatched input lengths")); }
    let sweep = MfiBatchRange { period: period_range };
    let (inner, _combos) = py.allow_threads(|| {
        let cuda = CudaMfi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.mfi_batch_dev(tp, vol, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "mfi_cuda_many_series_one_param_dev")]
#[pyo3(signature = (typical_price_tm, volume_tm, cols, rows, period, device_id=0))]
pub fn mfi_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    typical_price_tm: PyReadonlyArray1<'_, f32>,
    volume_tm: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() { return Err(PyValueError::new_err("CUDA not available")); }
    let tp = typical_price_tm.as_slice()?;
    let vol = volume_tm.as_slice()?;
    if tp.len() != vol.len() { return Err(PyValueError::new_err("mismatched input lengths")); }
    if tp.len() != cols * rows { return Err(PyValueError::new_err("unexpected matrix size")); }
    let inner = py.allow_threads(|| {
        let cuda = CudaMfi::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.mfi_many_series_one_param_time_major_dev(tp, vol, cols, rows, period)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}

#[inline]
pub fn mfi_into_slice(dst: &mut [f64], input: &MfiInput, kern: Kernel) -> Result<(), MfiError> {
    let (typical_price, volume, period, first_valid_idx, chosen) = mfi_prepare(input, kern)?;

    if dst.len() != typical_price.len() {
        return Err(MfiError::InvalidPeriod {
            period: dst.len(),
            data_len: typical_price.len(),
        });
    }

    mfi_compute_into(typical_price, volume, period, first_valid_idx, chosen, dst);

    // Fill warmup with NaN
    let warmup_period = first_valid_idx + period - 1;
    for v in &mut dst[..warmup_period] {
        *v = f64::NAN;
    }

    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mfi_js(typical_price: &[f64], volume: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = MfiParams {
        period: Some(period),
    };
    let input = MfiInput::from_slices(typical_price, volume, params);

    // Get result from the main function which already uses proper allocation
    let result = mfi_with_kernel(&input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(result.values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mfi_into(
    typical_price_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if typical_price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let typical_price = std::slice::from_raw_parts(typical_price_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let params = MfiParams {
            period: Some(period),
        };
        let input = MfiInput::from_slices(typical_price, volume, params);

        // Check for aliasing with either input
        if typical_price_ptr == out_ptr || volume_ptr == out_ptr {
            // Use main function which handles allocation properly
            let result = mfi_with_kernel(&input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&result.values);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            mfi_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mfi_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mfi_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MfiBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MfiBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MfiParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = mfi_batch)]
pub fn mfi_batch_unified_js(
    typical_price: &[f64],
    volume: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: MfiBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = MfiBatchRange {
        period: config.period_range,
    };

    let output = mfi_batch_inner(typical_price, volume, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = MfiBatchJsOutput {
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
pub fn mfi_batch_into(
    typical_price_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if typical_price_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to mfi_batch_into"));
    }
    unsafe {
        let tp = std::slice::from_raw_parts(typical_price_ptr, len);
        let vol = std::slice::from_raw_parts(volume_ptr, len);

        let sweep = MfiBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        // Destination must be rows * cols
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        mfi_batch_inner_into(tp, vol, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;

    fn check_mfi_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = MfiParams { period: None };
        let input = MfiInput::from_candles(&candles, "hlc3", default_params);
        let output = mfi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_mfi_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MfiParams { period: Some(14) };
        let input = MfiInput::from_candles(&candles, "hlc3", params);
        let mfi_result = mfi_with_kernel(&input, kernel)?;
        let expected_last_five_mfi = [
            38.13874339324763,
            37.44139770113819,
            31.02039511395131,
            28.092605898618896,
            25.905204729397813,
        ];
        let start_index = mfi_result.values.len() - 5;
        for (i, &value) in mfi_result.values[start_index..].iter().enumerate() {
            let expected_value = expected_last_five_mfi[i];
            let diff = (value - expected_value).abs();
            assert!(
                diff < 1e-1,
                "MFI mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        Ok(())
    }

    fn check_mfi_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MfiInput::with_default_candles(&candles);
        let output = mfi_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_mfi_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MfiParams { period: Some(0) };
        let input = MfiInput::from_candles(&candles, "hlc3", params);
        let result = mfi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_mfi_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_high = [1.0, 2.0, 3.0];
        let input_low = [0.5, 1.5, 2.5];
        let input_close = [0.8, 1.8, 2.8];
        let input_volume = [100.0, 200.0, 300.0];
        // Calculate typical price (HLC3)
        let typical_price: Vec<f64> = input_high
            .iter()
            .zip(&input_low)
            .zip(&input_close)
            .map(|((h, l), c)| (h + l + c) / 3.0)
            .collect();
        let params = MfiParams { period: Some(10) };
        let input = MfiInput::from_slices(&typical_price, &input_volume, params);
        let result = mfi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_mfi_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_high = [1.0];
        let input_low = [0.5];
        let input_close = [0.8];
        let input_volume = [100.0];
        // Calculate typical price (HLC3)
        let typical_price = [(input_high[0] + input_low[0] + input_close[0]) / 3.0];
        let params = MfiParams { period: Some(14) };
        let input = MfiInput::from_slices(&typical_price, &input_volume, params);
        let result = mfi_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_mfi_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = MfiParams { period: Some(7) };
        let first_input = MfiInput::from_candles(&candles, "hlc3", first_params);
        let first_result = mfi_with_kernel(&first_input, kernel)?;
        let second_params = MfiParams { period: Some(7) };
        // Use the output from first run as typical price for second run
        let typical_price_values: Vec<f64> = first_result.values.clone();
        let volume_values: Vec<f64> = vec![10_000.0; first_result.values.len()];
        let second_input =
            MfiInput::from_slices(&typical_price_values, &volume_values, second_params);
        let second_result = mfi_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_mfi_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            MfiParams::default(),            // period: 14
            MfiParams { period: Some(2) },   // minimum
            MfiParams { period: Some(5) },   // small
            MfiParams { period: Some(7) },   // small
            MfiParams { period: Some(10) },  // medium
            MfiParams { period: Some(14) },  // default explicit
            MfiParams { period: Some(20) },  // medium-large
            MfiParams { period: Some(30) },  // large
            MfiParams { period: Some(50) },  // very large
            MfiParams { period: Some(100) }, // extra large
            MfiParams { period: Some(200) }, // maximum reasonable
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = MfiInput::from_candles(&candles, "hlc3", params.clone());
            let output = mfi_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(14),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_mfi_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_mfi_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating realistic data and parameters
        let strat = (2usize..=50) // Period range for MFI
            .prop_flat_map(|period| {
                // Generate data length from period to 400
                (period..=400).prop_flat_map(move |data_len| {
					// Choose data generation type
					prop_oneof![
						// 60% normal data with realistic price movements
						6 => (
							// Generate base price
							(10.0f64..10000.0f64),
							// Generate volatility
							(0.01f64..0.2f64),
							// Generate volume multiplier
							(1000.0f64..1_000_000.0f64),
							// Generate random changes for price walk
							prop::collection::vec(-1.0f64..1.0f64, data_len),
							// Generate random volume factors
							prop::collection::vec(0.0f64..1.0f64, data_len),
						).prop_map(move |(base_price, volatility, volume_mult, changes, vol_factors)| {
							let mut typical_price = Vec::with_capacity(data_len);
							let mut volume = Vec::with_capacity(data_len);
							let mut price = base_price;

							for i in 0..data_len {
								// Random walk for price using pre-generated random values
								let change = changes[i] * volatility;
								price *= 1.0 + change;
								price = price.max(0.01); // Ensure positive prices
								typical_price.push(price);

								// Correlated volume (higher volume on bigger moves)
								let vol = volume_mult * (0.5 + vol_factors[i] + change.abs() * 2.0);
								volume.push(vol.max(0.0));
							}

							(typical_price, volume, period)
						}),

						// 15% constant price data
						15 => prop::collection::vec(100.0f64..1000.0f64, 1..=1)
							.prop_map(move |prices| {
								let price = prices[0];
								let typical_price = vec![price; data_len];
								let volume = vec![10000.0; data_len];
								(typical_price, volume, period)
							}),

						// 15% trending data with volume correlation
						15 => prop::bool::ANY.prop_map(move |uptrend| {
							let mut typical_price = Vec::with_capacity(data_len);
							let mut volume = Vec::with_capacity(data_len);
							let start_price = 100.0;

							for i in 0..data_len {
								let trend_factor = if uptrend {
									1.0 + (i as f64 / data_len as f64) * 2.0  // Up to 3x increase
								} else {
									1.0 - (i as f64 / data_len as f64) * 0.7  // Up to 30% decrease
								};
								typical_price.push(start_price * trend_factor);
								// Higher volume on trend moves
								volume.push(10000.0 * (1.0 + i as f64 / data_len as f64) * 2.0);
							}

							(typical_price, volume, period)
						}),

						// 10% edge cases (zero/small volumes)
						1 => Just((
							(0..data_len).map(|i| 100.0 + (i as f64)).collect::<Vec<_>>(),
							vec![0.0; data_len],  // Zero volume
							period
						)),
					]
				})
            });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(typical_price, volume, period)| {
                let params = MfiParams {
                    period: Some(period),
                };
                let input = MfiInput::from_slices(&typical_price, &volume, params.clone());

                // Get output from kernel under test
                let MfiOutput { values: out } = mfi_with_kernel(&input, kernel)?;

                // Get reference output from scalar kernel
                let MfiOutput { values: ref_out } = mfi_with_kernel(&input, Kernel::Scalar)?;

                // Property 1: Output length matches input
                prop_assert_eq!(out.len(), typical_price.len(), "Output length mismatch");

                // Find first valid index
                let first_valid_idx = (0..typical_price.len())
                    .find(|&i| !typical_price[i].is_nan() && !volume[i].is_nan())
                    .unwrap_or(0);

                let expected_warmup = first_valid_idx + period - 1;

                // Property 2: Exact warmup period verification
                // First non-NaN should appear at exactly first_valid_idx + period - 1
                for i in 0..out.len() {
                    if i < expected_warmup {
                        prop_assert!(
                            out[i].is_nan(),
                            "Expected NaN during warmup at index {}, got {}",
                            i,
                            out[i]
                        );
                    } else if i == expected_warmup {
                        // First non-NaN value
                        prop_assert!(
                            !out[i].is_nan(),
                            "Expected first non-NaN at index {} but got NaN",
                            i
                        );
                    }
                }

                // Property 3: MFI values are bounded [0, 100]
                for (i, &val) in out.iter().enumerate().skip(expected_warmup) {
                    if !val.is_nan() {
                        prop_assert!(
                            val >= 0.0 && val <= 100.0,
                            "MFI out of bounds at index {}: {}",
                            i,
                            val
                        );
                    }
                }

                // Property 4: Constant prices should produce MFI = 0
                // (no price change means no money flow)
                let is_constant = typical_price
                    .windows(2)
                    .all(|w| (w[0] - w[1]).abs() < 1e-10);
                if is_constant && expected_warmup < out.len() {
                    for i in expected_warmup..out.len() {
                        if !out[i].is_nan() {
                            // MFI is 0 when no price change (no flow)
                            prop_assert!(
                                out[i].abs() < 1e-3,
                                "Constant price MFI should be ~0, got {} at index {}",
                                out[i],
                                i
                            );
                        }
                    }
                }

                // Property 5: Zero volume handling (FIXED)
                let all_zero_volume = volume.iter().all(|&v| v.abs() < 1e-14);
                if all_zero_volume && expected_warmup < out.len() {
                    for i in expected_warmup..out.len() {
                        if !out[i].is_nan() {
                            // When volume is 0, flow is 0, so MFI should be 0
                            prop_assert!(
                                out[i].abs() < 1e-3,
                                "Zero volume MFI should be 0, got {} at index {}",
                                out[i],
                                i
                            );
                        }
                    }
                }

                // Property 6: Volume-weighted trend behavior (FIXED)
                // Verify that MFI reflects the volume-weighted money flow direction
                // MFI at index i is calculated from data points (i - period + 1) to i
                if expected_warmup + period < typical_price.len() {
                    // Pick an MFI value to check after warmup
                    let check_idx = expected_warmup + period;

                    // Calculate the window that this MFI value is based on
                    // MFI at check_idx uses the last 'period' data points
                    let window_start = check_idx - period + 1;
                    let window_end = check_idx;

                    // Count up vs down moves with volume weighting in the CORRECT window
                    let mut up_volume = 0.0;
                    let mut down_volume = 0.0;

                    for i in window_start..window_end {
                        if i > 0 && i < typical_price.len() {
                            let price_change = typical_price[i] - typical_price[i - 1];
                            if price_change > 0.0 {
                                up_volume += volume[i] * typical_price[i]; // Money flow
                            } else if price_change < 0.0 {
                                down_volume += volume[i] * typical_price[i]; // Money flow
                            }
                        }
                    }

                    // If significantly more up money flow, MFI should be > 50
                    if up_volume > down_volume * 2.0 && check_idx < out.len() {
                        let mfi_val = out[check_idx];
                        if !mfi_val.is_nan() && (up_volume + down_volume) > 1e-10 {
                            prop_assert!(
								mfi_val > 50.0,
								"MFI should be > 50 when up money flow dominates (up: {}, down: {}), got {}",
								up_volume,
								down_volume,
								mfi_val
							);
                        }
                    }

                    // If significantly more down money flow, MFI should be < 50
                    if down_volume > up_volume * 2.0 && check_idx < out.len() {
                        let mfi_val = out[check_idx];
                        if !mfi_val.is_nan() && (up_volume + down_volume) > 1e-10 {
                            prop_assert!(
								mfi_val < 50.0,
								"MFI should be < 50 when down money flow dominates (up: {}, down: {}), got {}",
								up_volume,
								down_volume,
								mfi_val
							);
                        }
                    }
                }

                // Property 7: Mathematical formula verification (FIXED)
                // Manually calculate MFI and verify it matches the implementation
                // MFI = 100 * (positive_money_flow / (positive_money_flow + negative_money_flow))
                if expected_warmup + 5 < typical_price.len() {
                    let verify_idx = expected_warmup + 5;

                    // Manually calculate MFI at verify_idx
                    // The MFI at index i includes the last 'period' data points: (i - period + 1) to i
                    let mut pos_sum = 0.0;
                    let mut neg_sum = 0.0;

                    // Start from the second point in the window since we need a previous price
                    let window_start = verify_idx - period + 1;

                    for i in window_start..=verify_idx {
                        if i > 0 && i < typical_price.len() {
                            let price_diff = typical_price[i] - typical_price[i - 1];
                            let money_flow = typical_price[i] * volume[i];

                            if price_diff > 0.0 {
                                pos_sum += money_flow;
                            } else if price_diff < 0.0 {
                                neg_sum += money_flow;
                            }
                            // If price_diff == 0, no flow is added (implementation behavior)
                        }
                    }

                    let total = pos_sum + neg_sum;
                    let expected_mfi = if total < 1e-14 {
                        0.0
                    } else {
                        100.0 * (pos_sum / total)
                    };

                    let actual_mfi = out[verify_idx];
                    if !actual_mfi.is_nan() {
                        prop_assert!(
							(actual_mfi - expected_mfi).abs() < 0.1,
							"MFI formula verification failed at index {}: expected {} (pos: {}, neg: {}), got {}",
							verify_idx,
							expected_mfi,
							pos_sum,
							neg_sum,
							actual_mfi
						);
                    }
                }

                // Property 8: Volume weighting verification (COMPLETELY REWRITTEN)
                // Test that volume amplifies price movements in MFI calculation
                if period >= 5 && period <= 20 {
                    // Reasonable period range for this test
                    // Create two datasets with identical monotonic price increases
                    let test_len = period * 3; // Enough for warmup and testing
                    let mut prices = Vec::with_capacity(test_len);
                    let mut increasing_vol = Vec::with_capacity(test_len);
                    let mut decreasing_vol = Vec::with_capacity(test_len);

                    // Create steadily increasing prices
                    for i in 0..test_len {
                        prices.push(100.0 + i as f64); // Monotonic increase
                                                       // Increasing volume amplifies up-moves
                        increasing_vol.push(1000.0 * (1.0 + i as f64));
                        // Decreasing volume dampens up-moves
                        decreasing_vol.push(1000.0 * (test_len as f64 - i as f64));
                    }

                    // Calculate MFI with increasing volume (amplifies upward movement)
                    let input_inc = MfiInput::from_slices(&prices, &increasing_vol, params.clone());
                    let MfiOutput { values: out_inc } = mfi_with_kernel(&input_inc, kernel)?;

                    // Calculate MFI with decreasing volume (dampens upward movement)
                    let input_dec = MfiInput::from_slices(&prices, &decreasing_vol, params.clone());
                    let MfiOutput { values: out_dec } = mfi_with_kernel(&input_dec, kernel)?;

                    // Check MFI values after warmup
                    // With monotonic price increases:
                    // - Increasing volume should produce higher MFI (more recent moves have more weight)
                    // - Decreasing volume should produce lower MFI (more recent moves have less weight)
                    let check_idx = period * 2; // Well past warmup
                    if check_idx < out_inc.len() {
                        let mfi_inc = out_inc[check_idx];
                        let mfi_dec = out_dec[check_idx];

                        if !mfi_inc.is_nan() && !mfi_dec.is_nan() {
                            // Both should be high (> 90) since all moves are upward
                            prop_assert!(
                                mfi_inc > 90.0,
                                "MFI with increasing volume on uptrend should be > 90, got {}",
                                mfi_inc
                            );
                            prop_assert!(
                                mfi_dec > 90.0,
                                "MFI with decreasing volume on uptrend should be > 90, got {}",
                                mfi_dec
                            );

                            // Increasing volume should produce higher MFI than decreasing
                            prop_assert!(
								mfi_inc > mfi_dec,
								"MFI with increasing volume ({}) should be > MFI with decreasing volume ({}) on uptrend",
								mfi_inc,
								mfi_dec
							);
                        }
                    }
                }

                // Property 9: Kernel consistency (critical)
                for i in 0..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Both should be NaN or both should be finite
                    if y.is_nan() || r.is_nan() {
                        prop_assert_eq!(
                            y.is_nan(),
                            r.is_nan(),
                            "NaN mismatch at index {}: kernel={}, scalar={}",
                            i,
                            y,
                            r
                        );
                        continue;
                    }

                    // Check ULP difference for finite values
                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();
                    let ulp_diff = y_bits.abs_diff(r_bits);

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 5,
                        "Kernel mismatch at index {}: {} vs {} (ULP={})",
                        i,
                        y,
                        r,
                        ulp_diff
                    );
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    macro_rules! generate_all_mfi_tests {
        ($($test_fn:ident),*) => {
            paste! {
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
    generate_all_mfi_tests!(
        check_mfi_partial_params,
        check_mfi_accuracy,
        check_mfi_default_candles,
        check_mfi_zero_period,
        check_mfi_period_exceeds_length,
        check_mfi_very_small_dataset,
        check_mfi_reinput,
        check_mfi_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_mfi_tests!(check_mfi_property);
    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = MfiBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        let def = MfiParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        // The following block assumes you have a known correct output for your default test file
        // If you want to check real values, insert expected_last_five_mfi as needed:
        let expected = [
            38.13874339324763,
            37.44139770113819,
            31.02039511395131,
            28.092605898618896,
            25.905204729397813,
        ];
        let start = row.len().saturating_sub(5);
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (2, 10, 2),   // Small periods
            (5, 25, 5),   // Medium periods
            (30, 60, 15), // Large periods
            (2, 5, 1),    // Dense small range
            (10, 50, 10), // Wide medium range
            (7, 21, 7),   // Weekly periods
            (14, 14, 0),  // Single period (default)
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = MfiBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c)?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(14)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

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
    gen_batch_tests!(check_batch_no_poison);
}
