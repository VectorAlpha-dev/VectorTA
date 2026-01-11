//! # Awesome Oscillator (AO)
//!
//! A momentum indicator by Bill Williams, showing market momentum by comparing short and long period SMAs of median price (hl2).
//!
//! ## Parameters
//! - **short_period**: Window for short SMA (default: 5)
//! - **long_period**: Window for long SMA (default: 34)
//!
//! ## Returns
//! - **`Ok(AoOutput)`** with Vec<f64> of same length as input, leading values are NaN
//! - **`Err(AoError)`** otherwise
//!
//! ## Developer Status
//! - **SIMD**: Implemented (prefix-sum engine), but disabled at runtime for this indicator due to strict ULP tolerance in property tests. AVX2/AVX512 entry points currently delegate to scalar for correctness.
//! - **Scalar**: Optimized streaming loop with preloaded rolling sums and `mul_add` for FMA.
//! - **Streaming update**: O(1) - Efficient rolling sums for both SMAs.
//! - **CUDA**: Enabled with VRAM-backed handles; host builds DS prefixes and checks launch/memory limits for safety.
//! - **Memory**: Uses zero-copy helpers for outputs; SIMD path builds a per-call prefix array.
//! - **Note**: SIMD shows benefits at 10k–1M sizes; scalar kept as reference path.
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
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
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::CudaAo;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

impl<'a> AsRef<[f64]> for AoInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AoData::Slice(slice) => slice,
            AoData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AoParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}

impl Default for AoParams {
    fn default() -> Self {
        Self {
            short_period: Some(5),
            long_period: Some(34),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AoInput<'a> {
    pub data: AoData<'a>,
    pub params: AoParams,
}

impl<'a> AoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: AoParams) -> Self {
        Self {
            data: AoData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: AoParams) -> Self {
        Self {
            data: AoData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "hl2", AoParams::default())
    }
    #[inline]
    pub fn get_short(&self) -> usize {
        self.params.short_period.unwrap_or(5)
    }
    #[inline]
    pub fn get_long(&self) -> usize {
        self.params.long_period.unwrap_or(34)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AoBuilder {
    short_period: Option<usize>,
    long_period: Option<usize>,
    kernel: Kernel,
}

impl Default for AoBuilder {
    fn default() -> Self {
        Self {
            short_period: None,
            long_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AoBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn short_period(mut self, n: usize) -> Self {
        self.short_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn long_period(mut self, n: usize) -> Self {
        self.long_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AoOutput, AoError> {
        let p = AoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = AoInput::from_candles(c, "hl2", p);
        ao_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<AoOutput, AoError> {
        let p = AoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = AoInput::from_slice(d, p);
        ao_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<AoStream, AoError> {
        let p = AoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        AoStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AoError {
    #[error("ao: Input data slice is empty.")]
    EmptyInputData,
    #[error("ao: All values are NaN.")]
    AllValuesNaN,
    #[error("ao: Invalid periods: short={short}, long={long}")]
    InvalidPeriods { short: usize, long: usize },
    #[error("ao: Short period must be less than long period: short={short}, long={long}")]
    ShortPeriodNotLess { short: usize, long: usize },
    #[error("ao: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ao: High and low arrays must have same length: high={high_len}, low={low_len}")]
    MismatchedArrayLengths { high_len: usize, low_len: usize },
    #[error("ao: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ao: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("ao: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("ao: invalid input: {0}")]
    InvalidInput(String),
}

#[inline]
pub fn ao(input: &AoInput) -> Result<AoOutput, AoError> {
    ao_with_kernel(input, Kernel::Auto)
}

/// Compute Awesome Oscillator (AO) into a caller-provided buffer without allocating.
///
/// - Preserves NaN warmups exactly like `ao()` (quiet NaN prefix using the same pattern as
///   `alloc_with_nan_prefix`).
/// - The output slice length must equal the input length; returns `AoError::OutputLenMismatch`
///   on mismatch.
#[cfg(not(feature = "wasm"))]
pub fn ao_into(input: &AoInput, out: &mut [f64]) -> Result<(), AoError> {
    let (data, short, long, first, len) = ao_prepare(input)?;
    if out.len() != len {
        return Err(AoError::OutputLengthMismatch {
            expected: len,
            got: out.len(),
        });
    }

    
    let warmup_end = first + long - 1;
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut out[..warmup_end.min(len)] {
        *v = qnan;
    }

    
    let chosen = detect_best_kernel();
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => ao_scalar(data, short, long, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ao_avx2(data, short, long, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => ao_avx512(data, short, long, first, out),
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[inline]
pub fn ao_into_slice(dst: &mut [f64], input: &AoInput, kern: Kernel) -> Result<(), AoError> {
    let (data, short, long, first, len) = ao_prepare(input)?;
    if dst.len() != len {
        return Err(AoError::OutputLengthMismatch {
            expected: len,
            got: dst.len(),
        });
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => ao_scalar(data, short, long, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ao_avx2(data, short, long, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => ao_avx512(data, short, long, first, dst),
            _ => unreachable!(),
        }
    }

    let warmup_end = first + long - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }
    Ok(())
}

#[inline(always)]
fn ao_prepare<'a>(input: &'a AoInput) -> Result<(&'a [f64], usize, usize, usize, usize), AoError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(AoError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AoError::AllValuesNaN)?;
    let len = data.len();
    let short = input.get_short();
    let long = input.get_long();

    if short == 0 || long == 0 {
        return Err(AoError::InvalidPeriods { short, long });
    }
    if short >= long {
        return Err(AoError::ShortPeriodNotLess { short, long });
    }
    if len - first < long {
        return Err(AoError::NotEnoughValidData {
            needed: long,
            valid: len - first,
        });
    }
    Ok((data, short, long, first, len))
}

pub fn ao_with_kernel(input: &AoInput, kernel: Kernel) -> Result<AoOutput, AoError> {
    let (data, short, long, first, len) = ao_prepare(input)?;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // Calculate warmup period
    let warmup_period = first + long - 1;

    // Use zero-copy allocation helper
    let mut out = alloc_with_nan_prefix(len, warmup_period);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => ao_scalar(data, short, long, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ao_avx2(data, short, long, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => ao_avx512(data, short, long, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(AoOutput { values: out })
}

// Helper function to compute hl2 from high/low arrays with zero-copy allocation
#[inline]
pub fn compute_hl2(high: &[f64], low: &[f64]) -> Result<Vec<f64>, AoError> {
    if high.len() != low.len() {
        return Err(AoError::MismatchedArrayLengths {
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    if high.is_empty() {
        return Err(AoError::EmptyInputData);
    }

    let mut out = alloc_with_nan_prefix(high.len(), 0);
    // Safe: we write every element before any read.
    for i in 0..high.len() {
        // unchecked to avoid bounds checks in tight loop
        unsafe {
            *out.get_unchecked_mut(i) = (*high.get_unchecked(i) + *low.get_unchecked(i)) * 0.5;
        }
    }
    Ok(out)
}

// Scalar implementation — optimized single streaming loop with preloaded sums
#[inline(always)]
pub fn ao_scalar(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    let len = data.len();
    if len == 0 {
        return;
    }
    let warm = first + long - 1;
    if warm >= len {
        return;
    }

    let inv_s = 1.0 / (short as f64);
    let inv_l = 1.0 / (long as f64);

    unsafe {
        let base = data.as_ptr();

        // Preload rolling sums right up to (but not including) the first write index `warm`.
        // long_sum covers [first .. warm-1] (long-1 elements)
        let mut long_sum = 0.0f64;
        let mut p = base.add(first);
        for _ in 0..(long - 1) {
            long_sum += *p;
            p = p.add(1);
        }

        // short_sum covers the last (short-1) elements of that long window prefix
        let mut short_sum = 0.0f64;
        let mut ps = base.add(first + long - short);
        for _ in 0..(short - 1) {
            short_sum += *ps;
            ps = ps.add(1);
        }

        // Set up pointers for a tight streaming loop
        let mut head = base.add(warm);
        let mut tail_long = base.add(first);
        let mut tail_short = base.add(first + long - short);
        let mut outp = out.as_mut_ptr().add(warm);

        let mut i = warm;
        // Unroll by 2 for reduced loop overhead
        while i + 1 < len {
            // iteration i
            let v0 = *head;
            long_sum += v0;
            short_sum += v0;
            *outp = short_sum.mul_add(inv_s, -long_sum * inv_l);
            long_sum -= *tail_long;
            short_sum -= *tail_short;
            head = head.add(1);
            tail_long = tail_long.add(1);
            tail_short = tail_short.add(1);
            outp = outp.add(1);
            i += 1;

            // iteration i+1
            let v1 = *head;
            long_sum += v1;
            short_sum += v1;
            *outp = short_sum.mul_add(inv_s, -long_sum * inv_l);
            long_sum -= *tail_long;
            short_sum -= *tail_short;
            head = head.add(1);
            tail_long = tail_long.add(1);
            tail_short = tail_short.add(1);
            outp = outp.add(1);
            i += 1;
        }

        // tail for odd count
        while i < len {
            let v = *head;
            long_sum += v;
            short_sum += v;
            *outp = short_sum.mul_add(inv_s, -long_sum * inv_l);
            long_sum -= *tail_long;
            short_sum -= *tail_short;
            head = head.add(1);
            tail_long = tail_long.add(1);
            tail_short = tail_short.add(1);
            outp = outp.add(1);
            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ao_avx512(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    if long <= 32 {
        unsafe { ao_avx512_short(data, short, long, first, out) }
    } else {
        unsafe { ao_avx512_long(data, short, long, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ao_avx2(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    // Disabled due to strict ULP tolerance in property tests; delegate to scalar
    unsafe { ao_scalar(data, short, long, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ao_avx512_short(
    data: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) {
    // Disabled due to strict ULP tolerance in property tests; delegate to scalar
    ao_scalar(data, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ao_avx512_long(
    data: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) {
    // Disabled due to strict ULP tolerance in property tests; delegate to scalar
    ao_scalar(data, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn ao_avx2_prefixsum(
    data: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let len = data.len();
    if len == 0 {
        return;
    }
    let warm = first + long - 1;
    if warm >= len {
        return;
    }

    // Build prefix sums over data[first..]
    let suffix_len = len - first;
    let mut pref = Vec::<f64>::with_capacity(suffix_len + 1);
    pref.push(0.0);
    let mut acc = 0.0f64;
    let p = data.as_ptr().add(first);
    for k in 0..suffix_len {
        acc += *p.add(k);
        pref.push(acc);
    }
    let pref_ptr = pref.as_ptr();

    let n_out = len - warm; // number of outputs written
    let out_base = out.as_mut_ptr().add(warm);

    let inv_s = _mm256_set1_pd(1.0 / (short as f64));
    let inv_l = _mm256_set1_pd(1.0 / (long as f64));

    // For output index warm (= first + long - 1), pref index is i_cur = long
    let cur0 = pref_ptr.add(long);
    let mut cur = cur0;
    let mut prev_s = cur0.sub(short);
    let mut prev_l = cur0.sub(long);
    let mut dst = out_base;

    let vec_chunks = n_out / 4;
    for _ in 0..vec_chunks {
        let pc = _mm256_loadu_pd(cur);
        let ps = _mm256_loadu_pd(prev_s);
        let pl = _mm256_loadu_pd(prev_l);

        let short_sum = _mm256_sub_pd(pc, ps);
        let long_sum = _mm256_sub_pd(pc, pl);

        let z = _mm256_mul_pd(long_sum, inv_l);
        let ao = _mm256_fmsub_pd(short_sum, inv_s, z);

        _mm256_storeu_pd(dst, ao);

        cur = cur.add(4);
        prev_s = prev_s.add(4);
        prev_l = prev_l.add(4);
        dst = dst.add(4);
    }

    // Tail (0..3)
    let tail = n_out & 3;
    for t in 0..tail {
        let i_cur = long + (vec_chunks * 4 + t);
        let pc = *pref_ptr.add(i_cur);
        let ps = *pref_ptr.add(i_cur - short);
        let pl = *pref_ptr.add(i_cur - long);
        let y = pc - ps;
        let z = pc - pl;
        *dst.add(t) = y.mul_add(1.0 / (short as f64), -(z * (1.0 / (long as f64))));
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn ao_avx512_prefixsum(
    data: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) {
    use core::arch::x86_64::*;

    let len = data.len();
    if len == 0 {
        return;
    }
    let warm = first + long - 1;
    if warm >= len {
        return;
    }

    // Build prefix over data[first..]
    let suffix_len = len - first;
    let mut pref = Vec::<f64>::with_capacity(suffix_len + 1);
    pref.push(0.0);
    let mut acc = 0.0f64;
    let p = data.as_ptr().add(first);
    for k in 0..suffix_len {
        acc += *p.add(k);
        pref.push(acc);
    }
    let pref_ptr = pref.as_ptr();

    let n_out = len - warm;
    let out_base = out.as_mut_ptr().add(warm);

    let inv_s = _mm512_set1_pd(1.0 / (short as f64));
    let inv_l = _mm512_set1_pd(1.0 / (long as f64));

    let cur0 = pref_ptr.add(long);
    let mut cur = cur0;
    let mut prev_s = cur0.sub(short);
    let mut prev_l = cur0.sub(long);
    let mut dst = out_base;

    let vec_chunks = n_out / 8;
    for _ in 0..vec_chunks {
        let pc = _mm512_loadu_pd(cur);
        let ps = _mm512_loadu_pd(prev_s);
        let pl = _mm512_loadu_pd(prev_l);

        let short_sum = _mm512_sub_pd(pc, ps);
        let long_sum = _mm512_sub_pd(pc, pl);

        let z = _mm512_mul_pd(long_sum, inv_l);
        let ao = _mm512_fmsub_pd(short_sum, inv_s, z);

        _mm512_storeu_pd(dst, ao);

        cur = cur.add(8);
        prev_s = prev_s.add(8);
        prev_l = prev_l.add(8);
        dst = dst.add(8);
    }

    let tail = n_out & 7;
    for t in 0..tail {
        let i_cur = long + (vec_chunks * 8 + t);
        let pc = *pref_ptr.add(i_cur);
        let ps = *pref_ptr.add(i_cur - short);
        let pl = *pref_ptr.add(i_cur - long);
        let y = pc - ps;
        let z = pc - pl;
        *dst.add(t) = y.mul_add(1.0 / (short as f64), -(z * (1.0 / (long as f64))));
    }
}
// Streaming AO — single-ring, modulo-free O(1) updates.
// Decision: Switched to one circular buffer (size = long), no modulo or NaN sentinels; identical outputs, faster hot path.
#[derive(Debug, Clone)]
pub struct AoStream {
    short: usize,
    long: usize,
    inv_short: f64,
    inv_long: f64,
    // Circular buffer of last `long` values
    buf: Vec<f64>,
    // Next write position in `buf`
    head: usize,
    // Number of valid elements currently in the buffer (<= long)
    filled: usize,
    // Running sums
    short_sum: f64,
    long_sum: f64,
}

impl AoStream {
    pub fn try_new(params: AoParams) -> Result<Self, AoError> {
        let short = params.short_period.unwrap_or(5);
        let long = params.long_period.unwrap_or(34);
        if short == 0 || long == 0 {
            return Err(AoError::InvalidPeriods { short, long });
        }
        if short >= long {
            return Err(AoError::ShortPeriodNotLess { short, long });
        }
        Ok(Self {
            short,
            long,
            inv_short: 1.0 / (short as f64),
            inv_long: 1.0 / (long as f64),
            buf: vec![0.0; long],
            head: 0,
            filled: 0,
            short_sum: 0.0,
            long_sum: 0.0,
        })
    }

    /// Push one hl2 value. Returns AO after warmup (when `filled == long`), else `None`.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Value leaving the long window (valid once buffer is full)
        let old_long = if self.filled == self.long {
            self.buf[self.head]
        } else {
            0.0
        };

        // Value leaving the short window (valid once we have at least `short` samples)
        let old_short = if self.filled >= self.short {
            let idx = if self.head >= self.short {
                self.head - self.short
            } else {
                self.head + self.long - self.short
            };
            self.buf[idx]
        } else {
            0.0
        };

        // Write the new sample
        self.buf[self.head] = value;

        // Advance head with predictable wrap (avoid `%`)
        self.head += 1;
        if self.head == self.long {
            self.head = 0;
        }

        // Update counts
        if self.filled < self.long {
            self.filled += 1;
        }

        // Update rolling sums in O(1)
        self.long_sum += value - old_long;
        self.short_sum += value - old_short;

        // Emit AO when long window is ready
        if self.filled == self.long {
            Some(
                self.short_sum
                    .mul_add(self.inv_short, -(self.long_sum * self.inv_long)),
            )
        } else {
            None
        }
    }

    /// Optional: reset internal state without reallocating the buffer
    #[inline(always)]
    pub fn reset(&mut self) {
        self.head = 0;
        self.filled = 0;
        self.short_sum = 0.0;
        self.long_sum = 0.0;
        for x in &mut self.buf {
            *x = 0.0;
        }
    }
}

// Batch/grid support
#[derive(Clone, Debug)]
pub struct AoBatchRange {
    pub short_period: (usize, usize, usize),
    pub long_period: (usize, usize, usize),
}

impl Default for AoBatchRange {
    fn default() -> Self {
        Self {
            short_period: (5, 5, 0),
            long_period: (34, 283, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AoBatchBuilder {
    range: AoBatchRange,
    kernel: Kernel,
}

impl AoBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.short_period = (start, end, step);
        self
    }
    pub fn short_static(mut self, v: usize) -> Self {
        self.range.short_period = (v, v, 0);
        self
    }
    pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.long_period = (start, end, step);
        self
    }
    pub fn long_static(mut self, v: usize) -> Self {
        self.range.long_period = (v, v, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<AoBatchOutput, AoError> {
        ao_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<AoBatchOutput, AoError> {
        AoBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<AoBatchOutput, AoError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<AoBatchOutput, AoError> {
        AoBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "hl2")
    }
}

pub fn ao_batch_with_kernel(
    data: &[f64],
    sweep: &AoBatchRange,
    k: Kernel,
) -> Result<AoBatchOutput, AoError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(AoError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    ao_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct AoBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AoParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AoBatchOutput {
    pub fn row_for_params(&self, p: &AoParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.short_period.unwrap_or(5) == p.short_period.unwrap_or(5)
                && c.long_period.unwrap_or(34) == p.long_period.unwrap_or(34)
        })
    }
    pub fn values_for(&self, p: &AoParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid_checked(r: &AoBatchRange) -> Result<Vec<AoParams>, AoError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, AoError> {
        if step == 0 {
            return Ok(vec![start]);
        }
        if start == end {
            return Ok(vec![start]);
        }
        let mut out = Vec::new();
        if start < end {
            let mut v = start;
            while v <= end {
                out.push(v);
                match v.checked_add(step) {
                    Some(next) => {
                        if next == v { break; }
                        v = next;
                    }
                    None => break,
                }
            }
        } else {
            let mut v = start;
            while v >= end {
                out.push(v);
                if v < end + step { break; }
                v -= step;
                if v == 0 { break; }
            }
        }
        if out.is_empty() {
            return Err(AoError::InvalidRange { start, end, step });
        }
        Ok(out)
    }
    let shorts = axis_usize(r.short_period)?;
    let longs = axis_usize(r.long_period)?;

    let cap = shorts
        .len()
        .checked_mul(longs.len())
        .ok_or_else(|| AoError::InvalidInput("rows*cols overflow".into()))?;
    let mut out = Vec::with_capacity(cap);
    for &s in &shorts {
        for &l in &longs {
            if s < l && s > 0 && l > 0 {
                out.push(AoParams {
                    short_period: Some(s),
                    long_period: Some(l),
                });
            }
        }
    }
    if out.is_empty() {
        return Err(AoError::InvalidInput("no valid parameter combinations".into()));
    }
    Ok(out)
}

#[inline(always)]
pub fn ao_batch_slice(
    data: &[f64],
    sweep: &AoBatchRange,
    kern: Kernel,
) -> Result<AoBatchOutput, AoError> {
    ao_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn ao_batch_par_slice(
    data: &[f64],
    sweep: &AoBatchRange,
    kern: Kernel,
) -> Result<AoBatchOutput, AoError> {
    ao_batch_inner(data, sweep, kern, true)
}
#[inline(always)]
fn ao_batch_inner_into(
    data: &[f64],
    sweep: &AoBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AoParams>, AoError> {
    let combos = expand_grid_checked(sweep)?;
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AoError::AllValuesNaN)?;
    let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
    if data.len() - first < max_long {
        return Err(AoError::NotEnoughValidData {
            needed: max_long,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let short = combos[row].short_period.unwrap();
        let long = combos[row].long_period.unwrap();
        match kern {
            Kernel::Scalar => ao_row_scalar(data, first, short, long, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ao_row_avx2(data, first, short, long, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ao_row_avx512(data, first, short, long, out_row),
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

// Original batch function that allocates its own storage
#[inline(always)]
fn ao_batch_inner(
    data: &[f64],
    sweep: &AoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AoBatchOutput, AoError> {
    let combos = expand_grid_checked(sweep)?;
    let rows = combos.len();
    let cols = data.len();

    if cols == 0 {
        return Err(AoError::EmptyInputData);
    }

    // Validate BEFORE allocation to prevent memory leaks
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AoError::AllValuesNaN)?;
    let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
    if data.len() - first < max_long {
        return Err(AoError::NotEnoughValidData {
            needed: max_long,
            valid: data.len() - first,
        });
    }

    // Guard rows*cols overflow before allocation
    let _total = rows
        .checked_mul(cols)
        .ok_or_else(|| AoError::InvalidInput("rows*cols overflow".into()))?;

    // Now safe to allocate - we've validated all error conditions
    let mut buf_mu = make_uninit_matrix(rows, cols);

    
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.long_period.unwrap() - 1)
        .collect();

    
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    
    let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
    let values_ptr = buf_guard.as_mut_ptr() as *mut f64;
    let values_len = buf_guard.len();
    let values_cap = buf_guard.capacity();

    let values = unsafe {
        
        let slice = std::slice::from_raw_parts_mut(values_ptr, values_len);

        
        
        ao_batch_inner_into(data, sweep, kern, parallel, slice)?;

        
        Vec::from_raw_parts(values_ptr, values_len, values_cap)
    };

    Ok(AoBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}
#[inline(always)]
unsafe fn ao_row_scalar(data: &[f64], first: usize, short: usize, long: usize, out: &mut [f64]) {
    ao_scalar(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ao_row_avx2(data: &[f64], first: usize, short: usize, long: usize, out: &mut [f64]) {
    ao_scalar(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ao_row_avx512(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    if long <= 32 {
        ao_row_avx512_short(data, first, short, long, out);
    } else {
        ao_row_avx512_long(data, first, short, long, out);
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ao_row_avx512_short(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    ao_scalar(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ao_row_avx512_long(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    ao_scalar(data, short, long, first, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;

    #[test]
    fn test_ao_into_matches_api() {
        
        let len = 256;
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            
            let x = i as f64;
            data.push((x * 0.01).mul_add(1.0, (x * 0.0314159).sin()));
        }

        let input = AoInput::from_slice(&data, AoParams::default());

        
        let base = ao(&input).expect("ao() should succeed");

        
        let mut out = vec![0.0f64; len];
        ao_into(&input, &mut out).expect("ao_into() should succeed");

        assert_eq!(base.values.len(), out.len());

        
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..len {
            assert!(
                eq_or_both_nan(base.values[i], out[i]),
                "Mismatch at index {}: got {}, expected {}",
                i,
                out[i],
                base.values[i]
            );
        }
    }

    fn check_ao_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = AoParams {
            short_period: Some(3),
            long_period: None,
        };
        let input = AoInput::from_candles(&candles, "hl2", partial_params);
        let result = ao_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }
    fn check_ao_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AoInput::with_default_candles(&candles);
        let result = ao_with_kernel(&input, kernel)?;
        let expected_last_five = [-1671.3, -1401.6706, -1262.3559, -1178.4941, -1157.4118];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] AO {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_ao_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AoInput::with_default_candles(&candles);
        match input.data {
            AoData::Candles { source, .. } => assert_eq!(source, "hl2"),
            _ => panic!("Expected AoData::Candles"),
        }
        let output = ao_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_ao_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = AoParams {
            short_period: Some(0),
            long_period: Some(34),
        };
        let input = AoInput::from_slice(&input_data, params);
        let res = ao_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] AO should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_ao_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = AoParams {
            short_period: Some(5),
            long_period: Some(10),
        };
        let input = AoInput::from_slice(&data_small, params);
        let res = ao_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] AO should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_ao_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = AoParams {
            short_period: Some(5),
            long_period: Some(34),
        };
        let input = AoInput::from_slice(&single_point, params);
        let res = ao_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] AO should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_ao_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = AoParams {
            short_period: Some(5),
            long_period: Some(34),
        };
        let first_input = AoInput::from_candles(&candles, "hl2", first_params);
        let first_result = ao_with_kernel(&first_input, kernel)?;
        let second_params = AoParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let second_input = AoInput::from_slice(&first_result.values, second_params);
        let second_result = ao_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }
    fn check_ao_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AoInput::from_candles(
            &candles,
            "hl2",
            AoParams {
                short_period: Some(5),
                long_period: Some(34),
            },
        );
        let res = ao_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }
    
    #[cfg(debug_assertions)]
    fn check_ao_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        
        let test_params = vec![
            AoParams::default(), 
            AoParams {
                short_period: Some(2),
                long_period: Some(10),
            }, 
            AoParams {
                short_period: Some(3),
                long_period: Some(20),
            }, 
            AoParams {
                short_period: Some(10),
                long_period: Some(50),
            }, 
            AoParams {
                short_period: Some(20),
                long_period: Some(100),
            }, 
            AoParams {
                short_period: Some(5),
                long_period: Some(200),
            }, 
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = AoInput::from_candles(&candles, "hl2", params.clone());
            let result = ao_with_kernel(&input, kernel)?;

            
            for (i, &val) in result.values.iter().enumerate() {
                if val.is_nan() {
                    continue; 
                }

                let bits = val.to_bits();

                
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: short={}, long={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_period.unwrap_or(5),
                        params.long_period.unwrap_or(34),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: short={}, long={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_period.unwrap_or(5),
                        params.long_period.unwrap_or(34),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: short={}, long={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.short_period.unwrap_or(5),
                        params.long_period.unwrap_or(34),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_ao_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) 
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_ao_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        
        
        
        
        let strat = (1usize..=50).prop_flat_map(|short_period| {
            ((short_period + 1)..=100).prop_flat_map(move |long_period| {
                (
                    prop::collection::vec(
                        (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                        long_period..400,
                    ),
                    Just(short_period),
                    Just(long_period),
                )
            })
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, short_period, long_period)| {
                let params = AoParams {
                    short_period: Some(short_period),
                    long_period: Some(long_period),
                };
                let input = AoInput::from_slice(&data, params);

                
                let AoOutput { values: out } = ao_with_kernel(&input, kernel).unwrap();
                
                let AoOutput { values: ref_out } = ao_with_kernel(&input, Kernel::Scalar).unwrap();

                
                for i in 0..(long_period - 1) {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN during warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                
                for i in (long_period - 1)..data.len() {
                    prop_assert!(
                        out[i].is_finite(),
                        "Expected finite value after warmup at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) && data.len() >= long_period
                {
                    
                    for i in (long_period - 1)..data.len() {
                        prop_assert!(
                            out[i].abs() < 1e-9,
                            "For constant data, AO should be 0 at index {}, got {}",
                            i,
                            out[i]
                        );
                    }
                }

                
                
                
                if data.len() >= long_period {
                    for i in (long_period - 1)..data.len() {
                        
                        let short_start = i + 1 - short_period;
                        let long_start = i + 1 - long_period;

                        
                        let long_window = &data[long_start..=i];
                        let window_min = long_window.iter().cloned().fold(f64::INFINITY, f64::min);
                        let window_max = long_window
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);

                        
                        
                        let theoretical_max = window_max - window_min;

                        prop_assert!(
                            out[i].abs() <= theoretical_max + 1e-9,
                            "AO value {} at index {} exceeds theoretical max {}",
                            out[i],
                            i,
                            theoretical_max
                        );
                    }
                }

                
                if short_period == 1 && long_period == 2 && data.len() >= 2 {
                    
                    
                    
                    for i in 1..data.len() {
                        let expected = data[i] - (data[i] + data[i - 1]) / 2.0;
                        let actual = out[i];
                        prop_assert!(
							(actual - expected).abs() < 1e-9,
							"Special case (short=1, long=2) mismatch at index {}: expected {}, got {}",
							i,
							expected,
							actual
						);
                    }
                }

                
                
                let is_increasing = data.windows(2).all(|w| w[1] > w[0] + 1e-10);
                let is_decreasing = data.windows(2).all(|w| w[1] < w[0] - 1e-10);

                if is_increasing && data.len() >= long_period {
                    
                    for i in (long_period - 1)..data.len() {
                        prop_assert!(
							out[i] > -1e-9,
							"For strictly increasing data, AO should be positive at index {}, got {}",
							i,
							out[i]
						);
                    }
                }

                if is_decreasing && data.len() >= long_period {
                    
                    for i in (long_period - 1)..data.len() {
                        prop_assert!(
							out[i] < 1e-9,
							"For strictly decreasing data, AO should be negative at index {}, got {}",
							i,
							out[i]
						);
                    }
                }

                
                
                if data.len() >= 3 {
                    let diffs: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
                    let is_linear = diffs.windows(2).all(|w| (w[1] - w[0]).abs() < 1e-10);

                    if is_linear && data.len() >= long_period + 10 {
                        
                        
                        let stable_start = long_period + 5;
                        if stable_start < data.len() - 1 {
                            let stable_values = &out[stable_start..];
                            if stable_values.len() >= 2 {
                                let first_stable = stable_values[0];
                                for (idx, &val) in stable_values.iter().enumerate() {
                                    prop_assert!(
										(val - first_stable).abs() < 1e-8,
										"For linear data, AO should stabilize. Value at {} differs from stable value: {} vs {}",
										stable_start + idx,
										val,
										first_stable
									);
                                }
                            }
                        }
                    }
                }

                
                for i in 0..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    
                    if y.is_nan() || r.is_nan() {
                        prop_assert!(
                            y.is_nan() && r.is_nan(),
                            "NaN mismatch at index {}: kernel={:?} is_nan={}, scalar is_nan={}",
                            i,
                            kernel,
                            y.is_nan(),
                            r.is_nan()
                        );
                        continue;
                    }

                    
                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();
                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "Kernel mismatch at index {}: kernel={:?} value={}, scalar value={}, \
						 diff={}, ULP diff={}",
                        i,
                        kernel,
                        y,
                        r,
                        (y - r).abs(),
                        ulp_diff
                    );
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_ao_tests {
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
    generate_all_ao_tests!(
        check_ao_partial_params,
        check_ao_accuracy,
        check_ao_default_candles,
        check_ao_zero_period,
        check_ao_period_exceeds_length,
        check_ao_very_small_dataset,
        check_ao_reinput,
        check_ao_nan_handling,
        check_ao_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_ao_tests!(check_ao_property);

    #[test]
    fn test_output_len_mismatch_error() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let params = AoParams {
            short_period: Some(2),
            long_period: Some(3),
        };
        let input = AoInput::from_slice(&data, params);

        
        let mut wrong_sized_buf = vec![0.0; 10]; 

        let result = ao_into_slice(&mut wrong_sized_buf, &input, Kernel::Auto);
        assert!(result.is_err());

        if let Err(AoError::OutputLengthMismatch { expected, got }) = result {
            assert_eq!(expected, 5);
            assert_eq!(got, 10);
        } else {
            panic!("Expected OutputLenMismatch error");
        }
    }

    #[test]
    fn test_invalid_kernel_error() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let sweep = AoBatchRange::default();

        
        let result = ao_batch_with_kernel(&data, &sweep, Kernel::Scalar);
        assert!(result.is_err());

        if let Err(AoError::InvalidKernelForBatch(kernel)) = result {
            assert!(matches!(kernel, Kernel::Scalar));
        } else {
            panic!("Expected InvalidKernel error");
        }
    }

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = AoBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "hl2")?;
        let def = AoParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [-1671.3, -1401.6706, -1262.3559, -1178.4941, -1157.4118];
        let start = row.len() - 5;
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
            
            (2, 10, 2, 15, 40, 5),     
            (5, 20, 5, 30, 60, 10),    
            (10, 30, 10, 40, 100, 20), 
            (3, 3, 0, 10, 50, 10),     
            (5, 15, 5, 34, 34, 0),     
        ];

        for (cfg_idx, &(short_start, short_end, short_step, long_start, long_end, long_step)) in
            test_configs.iter().enumerate()
        {
            let sweep = AoBatchRange {
                short_period: (short_start, short_end, short_step),
                long_period: (long_start, long_end, long_step),
            };

            let output = ao_batch_with_kernel(source_type(&c, "hl2"), &sweep, kernel)?;

            
            for (row, combo) in output.combos.iter().enumerate() {
                let row_start = row * output.cols;
                let row_end = row_start + output.cols;
                let row_values = &output.values[row_start..row_end];

                for (col, &val) in row_values.iter().enumerate() {
                    if val.is_nan() {
                        continue; 
                    }

                    let bits = val.to_bits();
                    let idx = row * output.cols + col;

                    
                    if bits == 0x11111111_11111111 {
                        panic!(
							"[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) with params: short={}, long={}",
							test,
							cfg_idx,
							val,
							bits,
							row,
							col,
							idx,
							combo.short_period.unwrap_or(5),
							combo.long_period.unwrap_or(34)
						);
                    }

                    if bits == 0x22222222_22222222 {
                        panic!(
							"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) with params: short={}, long={}",
							test,
							cfg_idx,
							val,
							bits,
							row,
							col,
							idx,
							combo.short_period.unwrap_or(5),
							combo.long_period.unwrap_or(34)
						);
                    }

                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
							 at row {} col {} (flat index {}) with params: short={}, long={}",
                            test,
                            cfg_idx,
                            val,
                            bits,
                            row,
                            col,
                            idx,
                            combo.short_period.unwrap_or(5),
                            combo.long_period.unwrap_or(34)
                        );
                    }
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

#[cfg(feature = "python")]
#[pyfunction(name = "ao")]
#[pyo3(signature = (high, low, short_period, long_period, kernel=None))]
pub fn ao_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    short_period: usize,
    long_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;

    // Use kernel validation for safety
    let kern = validate_kernel(kernel, false)?;

    // Heavy lifting without the GIL
    let result_vec: Vec<f64> = py
        .allow_threads(|| -> Result<Vec<f64>, AoError> {
            // Compute hl2 using zero-copy allocation
            let hl2 = compute_hl2(high_slice, low_slice)?;

            // Build input struct
            let params = AoParams {
                short_period: Some(short_period),
                long_period: Some(long_period),
            };
            let ao_in = AoInput::from_slice(&hl2, params);

            // Compute AO
            ao_with_kernel(&ao_in, kern).map(|o| o.values)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "AoStream")]
pub struct AoStreamPy {
    stream: AoStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AoStreamPy {
    #[new]
    fn new(short_period: usize, long_period: usize) -> PyResult<Self> {
        let params = AoParams {
            short_period: Some(short_period),
            long_period: Some(long_period),
        };
        let stream = AoStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AoStreamPy { stream })
    }

    /// Updates the stream with a new high and low value and returns the calculated AO value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        let hl2 = (high + low) / 2.0;
        self.stream.update(hl2)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ao_batch")]
#[pyo3(signature = (high, low, short_period_range, long_period_range, kernel=None))]
pub fn ao_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    short_period_range: (usize, usize, usize),
    long_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;

    
    let kern = validate_kernel(kernel, true)?;

    let sweep = AoBatchRange {
        short_period: short_period_range,
        long_period: long_period_range,
    };

    
    let combos = expand_grid_checked(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = high_slice.len();

    
    
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let out_arr = unsafe { PyArray1::<f64>::new(py, [expected], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    
    let combos = py
        .allow_threads(|| -> Result<Vec<AoParams>, AoError> {
            
            let hl2 = compute_hl2(high_slice, low_slice)?;

            
            let first = hl2.iter().position(|x| !x.is_nan()).unwrap_or(0);
            let warm: Vec<usize> = combos
                .iter()
                .map(|c| first + c.long_period.unwrap() - 1)
                .collect();

            
            let slice_mu = unsafe {
                std::slice::from_raw_parts_mut(
                    slice_out.as_mut_ptr() as *mut MaybeUninit<f64>,
                    slice_out.len(),
                )
            };

            
            init_matrix_prefixes(slice_mu, cols, &warm);

            
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

            
            ao_batch_inner_into(&hl2, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "short_periods",
        combos
            .iter()
            .map(|p| p.short_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "long_periods",
        combos
            .iter()
            .map(|p| p.long_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}


#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct DeviceArrayF32AoPy {
    pub(crate) buf: Option<DeviceBuffer<f32>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32AoPy {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", (self.rows, self.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let ptr = self
            .buf
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?
            .as_device_ptr()
            .as_raw() as usize;
        d.set_item("data", (ptr, false))?;
        // Stream omitted: producing stream is synchronized before return
        d.set_item("version", 3)?;
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        // 2 == kDLCUDA per DLPack
        (2, self.device_id as i32)
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        // Determine allocation device and validate `dl_device` hint if provided.
        let (kdl, alloc_dev) = self.__dlpack_device__(); // (2, device_id)
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }

        // Producer synchronizes its CUDA stream before returning; reject explicit stream==0 for CUDA.
        if let Some(obj) = stream.as_ref() {
            if let Ok(i) = obj.extract::<i64>(py) {
                if i == 0 {
                    return Err(PyValueError::new_err(
                        "__dlpack__: stream 0 is disallowed for CUDA",
                    ));
                }
            }
        }

        // Move DeviceBuffer into shared DLPack helper so the consumer owns it.
        let buf = self
            .buf
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        let rows = self.rows;
        let cols = self.cols;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ao_cuda_batch_dev")]
#[pyo3(signature = (high, low, short_period_range, long_period_range, device_id=0))]
pub fn ao_cuda_batch_dev_py(
    py: Python<'_>,
    high: numpy::PyReadonlyArray1<'_, f32>,
    low: numpy::PyReadonlyArray1<'_, f32>,
    short_period_range: (usize, usize, usize),
    long_period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32AoPy> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    if high_slice.len() != low_slice.len() {
        return Err(PyValueError::new_err("high/low length mismatch"));
    }
    let mut hl2_f32 = vec![0f32; high_slice.len()];
    for i in 0..high_slice.len() {
        let h = high_slice[i];
        let l = low_slice[i];
        hl2_f32[i] = (h + l) * 0.5;
    }
    let sweep = AoBatchRange {
        short_period: short_period_range,
        long_period: long_period_range,
    };
    let inner = py.allow_threads(|| {
        let cuda = CudaAo::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.ao_batch_dev(&hl2_f32, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let crate::cuda::oscillators::ao_wrapper::DeviceArrayF32Ao { buf, rows, cols, ctx, device_id: dev_id } = inner;
    Ok(DeviceArrayF32AoPy {
        buf: Some(buf),
        rows,
        cols,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ao_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm, low_tm, cols, rows, short_period, long_period, device_id=0))]
pub fn ao_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm: numpy::PyReadonlyArray1<'_, f32>,
    low_tm: numpy::PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    short_period: usize,
    long_period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32AoPy> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let high_slice = high_tm.as_slice()?;
    let low_slice = low_tm.as_slice()?;
    let expected = cols
        .checked_mul(rows)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    if high_slice.len() != expected || low_slice.len() != expected {
        return Err(PyValueError::new_err("time-major input length mismatch"));
    }
    let mut hl2_f32 = vec![0f32; expected];
    for i in 0..expected {
        hl2_f32[i] = (high_slice[i] + low_slice[i]) * 0.5;
    }
    let inner = py.allow_threads(|| {
        let cuda = CudaAo::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.ao_many_series_one_param_time_major_dev(
            &hl2_f32,
            cols,
            rows,
            short_period,
            long_period,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let crate::cuda::oscillators::ao_wrapper::DeviceArrayF32Ao { buf, rows, cols, ctx, device_id: dev_id } = inner;
    Ok(DeviceArrayF32AoPy {
        buf: Some(buf),
        rows,
        cols,
        _ctx: ctx,
        device_id: dev_id,
    })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ao_js(
    high: &[f64],
    low: &[f64],
    short_period: usize,
    long_period: usize,
) -> Result<Vec<f64>, JsValue> {
    // Compute hl2 using zero-copy allocation
    let hl2 = compute_hl2(high, low).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let params = AoParams {
        short_period: Some(short_period),
        long_period: Some(long_period),
    };
    let input = AoInput::from_slice(&hl2, params);

    // Compute AO using the optimal kernel
    ao_with_kernel(&input, Kernel::Auto)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ao_batch_js(
    high: &[f64],
    low: &[f64],
    short_start: usize,
    short_end: usize,
    short_step: usize,
    long_start: usize,
    long_end: usize,
    long_step: usize,
) -> Result<Vec<f64>, JsValue> {
    // Compute hl2 using zero-copy allocation
    let hl2 = compute_hl2(high, low).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let sweep = AoBatchRange {
        short_period: (short_start, short_end, short_step),
        long_period: (long_start, long_end, long_step),
    };

    // Use the existing batch function with parallel=false for WASM
    ao_batch_inner(&hl2, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ao_batch_metadata_js(
    short_start: usize,
    short_end: usize,
    short_step: usize,
    long_start: usize,
    long_end: usize,
    long_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AoBatchRange {
        short_period: (short_start, short_end, short_step),
        long_period: (long_start, long_end, long_step),
    };

    let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut metadata = Vec::with_capacity(combos.len() * 2);

    for combo in combos {
        metadata.push(combo.short_period.unwrap() as f64);
        metadata.push(combo.long_period.unwrap() as f64);
    }

    Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AoBatchConfig {
    pub short_period_range: (usize, usize, usize),
    pub long_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AoBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AoParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ao_batch)]
pub fn ao_batch_unified_js(high: &[f64], low: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    // 1. Deserialize the configuration object from JavaScript
    let config: AoBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    // Compute hl2 using zero-copy allocation
    let hl2 = compute_hl2(high, low).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let sweep = AoBatchRange {
        short_period: config.short_period_range,
        long_period: config.long_period_range,
    };

    // 2. Run the existing core logic
    let output = ao_batch_inner(&hl2, &sweep, Kernel::Scalar, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // 3. Create the structured output
    let js_output = AoBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    // 4. Serialize the output struct into a JavaScript object
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ao_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ao_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ao_into(
    in_high_ptr: *const f64,
    in_low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    short_period: usize,
    long_period: usize,
) -> Result<(), JsValue> {
    if in_high_ptr.is_null() || in_low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    #[inline(always)]
    unsafe fn overlaps(a: *const f64, b: *const f64, n: usize) -> bool {
        let as_ = a as usize;
        let ae = as_.wrapping_add(n * core::mem::size_of::<f64>());
        let bs_ = b as usize;
        let be = bs_.wrapping_add(n * core::mem::size_of::<f64>());
        !(ae <= bs_ || be <= as_)
    }

    unsafe {
        let high = std::slice::from_raw_parts(in_high_ptr, len);
        let low = std::slice::from_raw_parts(in_low_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);

        let alias = overlaps(in_high_ptr, out_ptr as *const f64, len)
            || overlaps(in_low_ptr, out_ptr as *const f64, len);

        let hl2 = compute_hl2(high, low).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let params = AoParams {
            short_period: Some(short_period),
            long_period: Some(long_period),
        };
        let input = AoInput::from_slice(&hl2, params);

        if alias {
            let mut tmp = vec![0.0; len];
            ao_into_slice(&mut tmp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            out.copy_from_slice(&tmp);
        } else {
            ao_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ao_batch_into(
    in_high_ptr: *const f64,
    in_low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    short_period_start: usize,
    short_period_end: usize,
    short_period_step: usize,
    long_period_start: usize,
    long_period_end: usize,
    long_period_step: usize,
) -> Result<usize, JsValue> {
    if in_high_ptr.is_null() || in_low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(in_high_ptr, len);
        let low = std::slice::from_raw_parts(in_low_ptr, len);

        let sweep = AoBatchRange {
            short_period: (short_period_start, short_period_end, short_period_step),
            long_period: (long_period_start, long_period_end, long_period_step),
        };

        let combos = expand_grid_checked(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*len overflow"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        // Compute hl2 using zero-copy allocation
        let hl2 = compute_hl2(high, low).map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Compute batch directly into output
        ao_batch_inner_into(&hl2, &sweep, Kernel::Scalar, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
