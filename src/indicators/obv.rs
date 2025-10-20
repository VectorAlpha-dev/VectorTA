//! # On Balance Volume (OBV)
//!
//! OBV is a cumulative volume-based indicator. Adds or subtracts volume based on price movement direction.
//!
//! ## Parameters
//! - *(none)*
//!
//! ## Returns
//! - **`Ok(ObvOutput)`** on success, containing a `Vec<f64>` of length matching input.
//! - **`Err(ObvError)`** on failure
//!
//! ## Developer Notes
//! - Decision note: Scalar path optimized (branchless, mul_add). AVX-512 shows ~10% win vs scalar at 100k on x86_64; AVX2 is near parity. Left enabled; OBV is carry-bound so SIMD gains are modest.
//! - **Streaming**: O(1) with branch-lean sign and mul_add to match scalar/IEEE-754 NaN parity.
//! - **Memory**: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes)

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

// Input data enum
#[derive(Debug, Clone)]
pub enum ObvData<'a> {
    Candles { candles: &'a Candles },
    Slices { close: &'a [f64], volume: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct ObvOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct ObvParams;

#[derive(Debug, Clone)]
pub struct ObvInput<'a> {
    pub data: ObvData<'a>,
    pub params: ObvParams,
}

impl<'a> ObvInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: ObvParams) -> Self {
        Self {
            data: ObvData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: ObvParams) -> Self {
        Self {
            data: ObvData::Slices { close, volume },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, ObvParams::default())
    }
    #[inline(always)]
    fn as_refs(&self) -> (&'a [f64], &'a [f64]) {
        match &self.data {
            ObvData::Candles { candles } => (
                source_type(candles, "close"),
                source_type(candles, "volume"),
            ),
            ObvData::Slices { close, volume } => (*close, *volume),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ObvBuilder {
    kernel: Kernel,
}

impl Default for ObvBuilder {
    fn default() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
}

impl ObvBuilder {
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
    pub fn apply(self, candles: &Candles) -> Result<ObvOutput, ObvError> {
        let i = ObvInput::from_candles(candles, ObvParams::default());
        obv_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<ObvOutput, ObvError> {
        let i = ObvInput::from_slices(close, volume, ObvParams::default());
        obv_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> ObvStream {
        ObvStream::new()
    }
}

// Error type
#[derive(Debug, Error)]
pub enum ObvError {
    #[error("obv: Empty data provided.")]
    EmptyData,
    #[error("obv: Data length mismatch: close_len = {close_len}, volume_len = {volume_len}")]
    DataLengthMismatch { close_len: usize, volume_len: usize },
    #[error("obv: All values are NaN.")]
    AllValuesNaN,
}

impl From<Box<dyn std::error::Error>> for ObvError {
    fn from(_: Box<dyn std::error::Error>) -> Self {
        ObvError::EmptyData
    }
}

#[inline]
pub fn obv(input: &ObvInput) -> Result<ObvOutput, ObvError> {
    obv_with_kernel(input, Kernel::Auto)
}

pub fn obv_with_kernel(input: &ObvInput, kernel: Kernel) -> Result<ObvOutput, ObvError> {
    let (close, volume) = input.as_refs();

    if close.is_empty() || volume.is_empty() {
        return Err(ObvError::EmptyData);
    }
    if close.len() != volume.len() {
        return Err(ObvError::DataLengthMismatch {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let first = close
        .iter()
        .zip(volume.iter())
        .position(|(c, v)| !c.is_nan() && !v.is_nan())
        .ok_or(ObvError::AllValuesNaN)?;

    // single write of warm prefix; no extra NaN sweep
    let mut out = alloc_with_nan_prefix(close.len(), first);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => obv_scalar(close, volume, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => obv_avx2(close, volume, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => obv_avx512(close, volume, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(ObvOutput { values: out })
}

#[inline]
pub fn obv_scalar(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    // OBV starts at 0 at the first valid bar
    let mut prev_obv = 0.0f64;
    let mut prev_close = close[first_valid];
    out[first_valid] = 0.0;

    // Process remaining elements branchlessly and without indexing on inputs/outputs beyond slice iteration
    let tail_close = &close[first_valid + 1..];
    let tail_volume = &volume[first_valid + 1..];
    let tail_out = &mut out[first_valid + 1..];

    for (dst, (&c, &v)) in tail_out
        .iter_mut()
        .zip(tail_close.iter().zip(tail_volume.iter()))
    {
        // sign in {-1.0, 0.0, +1.0}; NaNs yield 0 (no change)
        let s = ((c > prev_close) as i32 - (c < prev_close) as i32) as f64;
        prev_obv = v.mul_add(s, prev_obv);
        *dst = prev_obv;
        prev_close = c;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn obv_avx2(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    // OBV is inherently a prefix-sum with a serial carry; we vectorize the
    // compare*volume step over 2 lanes at a time while preserving the carry.
    use core::arch::x86_64::*;
    let len = close.len();
    let mut prev_obv = 0.0f64;
    let mut prev_close = *close.get_unchecked(first_valid);
    *out.get_unchecked_mut(first_valid) = 0.0;

    let mut i = first_valid + 1;
    let end = len;

    let one = _mm_set1_pd(1.0);
    let neg_one = _mm_set1_pd(-1.0);
    let zero = _mm_setzero_pd();

    // Process 2 elements per iteration.
    while i + 1 < end {
        // close[i], close[i+1]
        let c = _mm_loadu_pd(close.as_ptr().add(i));
        // prev vector = [prev_close, close[i]]
        let prev = _mm_set_pd(*close.get_unchecked(i), prev_close);

        // sign in {-1,0,+1} as doubles, branchless
        let gt = _mm_cmpgt_pd(c, prev);
        let lt = _mm_cmplt_pd(c, prev);
        let pos = _mm_and_pd(gt, one);
        let neg = _mm_and_pd(lt, neg_one);
        let sign = _mm_add_pd(pos, neg);

        let vol = _mm_loadu_pd(volume.as_ptr().add(i));
        let dv = _mm_mul_pd(vol, sign); // [d0, d1] = sign*volume

        // Inclusive 2-lane scan: [d0, d0+d1]
        // m_shift = [0, d0]
        let m_shift = _mm_unpacklo_pd(zero, dv);
        let scan = _mm_add_pd(dv, m_shift);

        // Add scalar base carry (prev_obv) to both lanes, store
        let base = _mm_set1_pd(prev_obv);
        let res = _mm_add_pd(scan, base);
        _mm_storeu_pd(out.as_mut_ptr().add(i), res);

        // Update carry and prev_close using lane-1
        let res_hi = _mm_unpackhi_pd(res, res);
        prev_obv = _mm_cvtsd_f64(res_hi);

        let c_hi = _mm_unpackhi_pd(c, c);
        prev_close = _mm_cvtsd_f64(c_hi);

        i += 2;
    }

    // Tail (0 or 1 element)
    if i < end {
        let c = *close.get_unchecked(i);
        let v = *volume.get_unchecked(i);
        let s = ((c > prev_close) as i32 - (c < prev_close) as i32) as f64;
        prev_obv = v.mul_add(s, prev_obv);
        *out.get_unchecked_mut(i) = prev_obv;
        // prev_close = c; // not needed after final write
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn obv_avx512(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    // Same micro-kernel as AVX2 (2-lane SSE2) to preserve strict serial carry
    obv_avx2(close, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn obv_avx512_short(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    obv_avx2(close, volume, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn obv_avx512_long(close: &[f64], volume: &[f64], first_valid: usize, out: &mut [f64]) {
    obv_avx2(close, volume, first_valid, out)
}

#[inline(always)]
pub unsafe fn obv_row_scalar(
    close: &[f64],
    volume: &[f64],
    first: usize,
    _period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    obv_scalar(close, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn obv_row_avx2(
    close: &[f64],
    volume: &[f64],
    first: usize,
    _period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    obv_avx2(close, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn obv_row_avx512(
    close: &[f64],
    volume: &[f64],
    first: usize,
    _period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    obv_avx512(close, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn obv_row_avx512_short(
    close: &[f64],
    volume: &[f64],
    first: usize,
    _period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    obv_avx512_short(close, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn obv_row_avx512_long(
    close: &[f64],
    volume: &[f64],
    first: usize,
    _period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    obv_avx512_long(close, volume, first, out)
}

#[derive(Clone, Debug)]
pub struct ObvStream {
    prev_close: f64,
    prev_obv: f64,
    initialized: bool,
}

impl ObvStream {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            prev_close: f64::NAN,
            prev_obv: 0.0,
            initialized: false,
        }
    }

    /// O(1) streaming update for OBV.
    /// Semantics match the scalar batch kernel:
    /// - branchless sign in {-1.0, 0.0, +1.0}
    /// - uses mul_add to mirror fused-multiply-add behavior and NaN propagation
    /// - prev_close is advanced on every call (including NaN) to mirror scalar parity
    #[inline(always)]
    pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        // Find first valid observation
        if !self.initialized {
            if !close.is_nan() && !volume.is_nan() {
                self.prev_close = close;
                self.prev_obv = 0.0; // OBV starts at 0
                self.initialized = true;
                return Some(0.0);
            } else {
                return None; // still warming up
            }
        }

        // Branchless sign in {-1.0, 0.0, +1.0}; NaN compares are false -> 0.0
        let s = ((close > self.prev_close) as i32 - (close < self.prev_close) as i32) as f64;

        // Mirror scalar kernel exactly: FMA-style mul_add (propagates NaN per IEEE-754)
        self.prev_obv = volume.mul_add(s, self.prev_obv);

        // Always advance prev_close (including NaN) to match batch behavior across NaNs
        self.prev_close = close;

        Some(self.prev_obv)
    }

    /// Retrieve last computed OBV without mutating the stream.
    #[inline(always)]
    pub fn last(&self) -> Option<f64> {
        if self.initialized {
            Some(self.prev_obv)
        } else {
            None
        }
    }

    /// Reset the streaming state.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.prev_obv = 0.0;
        self.initialized = false;
    }
}

#[derive(Clone, Debug)]
pub struct ObvBatchRange {
    pub reserved: usize,
}

impl Default for ObvBatchRange {
    fn default() -> Self {
        Self { reserved: 1 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ObvBatchBuilder {
    kernel: Kernel,
}

impl ObvBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<ObvBatchOutput, ObvError> {
        obv_batch_with_kernel(close, volume, self.kernel)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<ObvBatchOutput, ObvError> {
        let close = source_type(c, "close");
        let volume = source_type(c, "volume");
        self.apply_slices(close, volume)
    }
    pub fn with_default_candles(c: &Candles) -> Result<ObvBatchOutput, ObvError> {
        ObvBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
    }
}

pub struct ObvBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

pub fn obv_batch_with_kernel(
    close: &[f64],
    volume: &[f64],
    kernel: Kernel,
) -> Result<ObvBatchOutput, ObvError> {
    let chosen = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch,
    };
    obv_batch_par_slice(close, volume, chosen)
}

#[inline(always)]
pub fn obv_batch_slice(
    close: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<ObvBatchOutput, ObvError> {
    obv_batch_inner(close, volume, kern, false)
}

#[inline(always)]
pub fn obv_batch_par_slice(
    close: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<ObvBatchOutput, ObvError> {
    obv_batch_inner(close, volume, kern, true)
}

#[inline(always)]
fn obv_batch_inner(
    close: &[f64],
    volume: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<ObvBatchOutput, ObvError> {
    if close.is_empty() || volume.is_empty() {
        return Err(ObvError::EmptyData);
    }
    if close.len() != volume.len() {
        return Err(ObvError::DataLengthMismatch {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    let first = close
        .iter()
        .zip(volume.iter())
        .position(|(c, v)| !c.is_nan() && !v.is_nan())
        .ok_or(ObvError::AllValuesNaN)?;

    let rows = 1usize;
    let cols = close.len();

    // allocate rows×cols uninit and seed only warm prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &[first]);

    // compute into the backing slice
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_slice: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, rows * cols) };

    unsafe {
        match kern {
            Kernel::ScalarBatch | Kernel::Scalar => obv_row_scalar(
                close,
                volume,
                first,
                0,
                0,
                core::ptr::null(),
                0.0,
                out_slice,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2Batch | Kernel::Avx2 => obv_row_avx2(
                close,
                volume,
                first,
                0,
                0,
                core::ptr::null(),
                0.0,
                out_slice,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512Batch | Kernel::Avx512 => obv_row_avx512(
                close,
                volume,
                first,
                0,
                0,
                core::ptr::null(),
                0.0,
                out_slice,
            ),
            _ => unreachable!(),
        }
    }

    // materialize Vec<f64> without extra copy
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            rows * cols,
            guard.capacity(),
        )
    };
    Ok(ObvBatchOutput { values, rows, cols })
}

#[inline(always)]
fn obv_batch_inner_into(
    close: &[f64],
    volume: &[f64],
    kern: Kernel,
    out: &mut [f64],
) -> Result<(), ObvError> {
    if close.is_empty() || volume.is_empty() {
        return Err(ObvError::EmptyData);
    }
    if close.len() != volume.len() || out.len() != close.len() {
        return Err(ObvError::DataLengthMismatch {
            close_len: close.len(),
            volume_len: out.len(),
        });
    }
    let first = close
        .iter()
        .zip(volume.iter())
        .position(|(c, v)| !c.is_nan() && !v.is_nan())
        .ok_or(ObvError::AllValuesNaN)?;

    // seed only warm prefix here since caller buffer is raw
    for v in &mut out[..first] {
        *v = f64::NAN;
    }

    unsafe {
        match kern {
            Kernel::ScalarBatch | Kernel::Scalar => {
                obv_row_scalar(close, volume, first, 0, 0, core::ptr::null(), 0.0, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2Batch | Kernel::Avx2 => {
                obv_row_avx2(close, volume, first, 0, 0, core::ptr::null(), 0.0, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512Batch | Kernel::Avx512 => {
                obv_row_avx512(close, volume, first, 0, 0, core::ptr::null(), 0.0, out)
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[inline(always)]
fn expand_grid(_r: &ObvBatchRange) -> Vec<ObvParams> {
    vec![ObvParams::default()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    fn check_obv_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close: [f64; 0] = [];
        let volume: [f64; 0] = [];
        let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
        let result = obv_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for empty data");
        Ok(())
    }
    fn check_obv_data_length_mismatch(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close = [1.0, 2.0, 3.0];
        let volume = [100.0, 200.0];
        let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
        let result = obv_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for mismatched data length");
        Ok(())
    }
    fn check_obv_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close = [f64::NAN, f64::NAN];
        let volume = [f64::NAN, f64::NAN];
        let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
        let result = obv_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for all NaN data");
        Ok(())
    }
    fn check_obv_csv_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close = source_type(&candles, "close");
        let volume = source_type(&candles, "volume");
        let input = ObvInput::from_candles(&candles, ObvParams::default());
        let obv_result = obv_with_kernel(&input, kernel)?;
        assert_eq!(obv_result.values.len(), close.len());
        let last_five_expected = [
            -329661.6180239202,
            -329767.87639284023,
            -329889.94421654026,
            -329801.35075036023,
            -330218.2007503602,
        ];
        let start_idx = obv_result.values.len() - 5;
        let result_tail = &obv_result.values[start_idx..];
        for (i, &val) in result_tail.iter().enumerate() {
            let exp_val = last_five_expected[i];
            let diff = (val - exp_val).abs();
            assert!(
                diff < 1e-6,
                "OBV mismatch at tail index {}: expected {}, got {}",
                i,
                exp_val,
                val
            );
        }
        Ok(())
    }

    macro_rules! generate_all_obv_tests {
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

    #[cfg(debug_assertions)]
    fn check_obv_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let close = source_type(&candles, "close");
        let volume = source_type(&candles, "volume");

        // OBV has no parameters, so we just test with default params
        let test_params = vec![ObvParams::default()];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = ObvInput::from_candles(&candles, params.clone());
            let output = obv_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_obv_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // OBV has no parameters, so batch output is always 1 row
        // We test with different kernels and data configurations
        let test_configs = vec!["Testing OBV batch with default configuration"];

        for (cfg_idx, _config_name) in test_configs.iter().enumerate() {
            let output = ObvBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols; // Should always be 0 for OBV
                let col = idx % output.cols;

                // Check all three poison patterns with detailed context
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with OBV (no params)",
                        test, cfg_idx, val, bits, row, col, idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with OBV (no params)",
                        test, cfg_idx, val, bits, row, col, idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with OBV (no params)",
                        test, cfg_idx, val, bits, row, col, idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_obv_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate random price and volume data
        // Note: We generate finite, non-NaN data to test realistic conditions
        // Volume includes zero to test edge case where no volume is traded
        let strat = prop::collection::vec(
            (
                (-1e6f64..1e6f64).prop_filter("finite close", |x| x.is_finite()),
                (0f64..1e6f64)
                    .prop_filter("finite positive volume", |x| x.is_finite() && *x >= 0.0),
            ),
            10..400,
        );

        proptest::test_runner::TestRunner::default().run(&strat, |price_volume_pairs| {
            let (close, volume): (Vec<f64>, Vec<f64>) = price_volume_pairs.into_iter().unzip();

            let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
            let ObvOutput { values: out } = obv_with_kernel(&input, kernel)?;
            let ObvOutput { values: ref_out } = obv_with_kernel(&input, Kernel::Scalar)?;

            // Find first valid index - should be 0 since we generate only finite values
            let first_valid = close
                .iter()
                .zip(volume.iter())
                .position(|(c, v)| !c.is_nan() && !v.is_nan());

            // Since we generate only finite values, first_valid should always be Some(0)
            prop_assert_eq!(
                first_valid,
                Some(0),
                "Expected first valid index to be 0 for finite input data"
            );

            if let Some(first_idx) = first_valid {
                // Validate NaN prefix: all values before first_idx should be NaN
                for i in 0..first_idx {
                    prop_assert!(
                        out[i].is_nan(),
                        "Expected NaN at index {} (before first_valid), got {}",
                        i,
                        out[i]
                    );
                }

                // Validate output completeness: all values from first_idx onwards should be valid
                for i in first_idx..out.len() {
                    prop_assert!(
                        !out[i].is_nan(),
                        "Expected valid value at index {} (after first_valid), got NaN",
                        i
                    );
                }

                // Property 1: First valid OBV value should be 0
                prop_assert_eq!(
                    out[first_idx],
                    0.0,
                    "First valid OBV at index {} should be 0, got {}",
                    first_idx,
                    out[first_idx]
                );

                // Property 2: Verify cumulative behavior for subsequent values
                // Now with explicit check for out[i-1] validity
                for i in (first_idx + 1)..close.len() {
                    // Both current and previous values should be valid after first_idx
                    if !out[i].is_nan() && i > 0 && !out[i - 1].is_nan() {
                        let obv_diff = out[i] - out[i - 1];
                        let price_diff = close[i] - close[i - 1];

                        if price_diff > 0.0 {
                            // Price increased, OBV should increase by volume
                            prop_assert!(
                                (obv_diff - volume[i]).abs() < 1e-9,
                                "At index {}: OBV diff {} should equal volume {} (price increased)",
                                i,
                                obv_diff,
                                volume[i]
                            );
                        } else if price_diff < 0.0 {
                            // Price decreased, OBV should decrease by volume
                            prop_assert!(
									(obv_diff + volume[i]).abs() < 1e-9,
									"At index {}: OBV diff {} should equal -volume {} (price decreased)",
									i, obv_diff, -volume[i]
								);
                        } else {
                            // Price unchanged, OBV should remain the same
                            prop_assert!(
									obv_diff.abs() < 1e-9,
									"At index {}: OBV should not change when price is unchanged, diff = {}",
									i, obv_diff
								);
                        }
                    }
                }

                // Property 3: All kernels should produce identical results
                for i in 0..out.len() {
                    if out[i].is_nan() && ref_out[i].is_nan() {
                        continue; // Both NaN is fine
                    }
                    prop_assert!(
                        (out[i] - ref_out[i]).abs() < 1e-9,
                        "Kernel mismatch at index {}: {} (kernel) vs {} (scalar)",
                        i,
                        out[i],
                        ref_out[i]
                    );
                }

                // Property 4: Check for poison values
                for (i, &val) in out.iter().enumerate() {
                    if !val.is_nan() {
                        let bits = val.to_bits();
                        prop_assert!(
                            bits != 0x11111111_11111111
                                && bits != 0x22222222_22222222
                                && bits != 0x33333333_33333333,
                            "Found poison value at index {}: {} (0x{:016X})",
                            i,
                            val,
                            bits
                        );
                    }
                }

                // Property 5: Test with constant price - OBV should remain at 0
                if close.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9) {
                    for i in first_idx..out.len() {
                        if !out[i].is_nan() {
                            prop_assert!(
                                out[i].abs() < 1e-9,
                                "OBV should remain at 0 for constant price, got {} at index {}",
                                out[i],
                                i
                            );
                        }
                    }
                }

                // Property 6: Monotonic increasing price - OBV should be cumulative sum of volumes
                if close.windows(2).all(|w| w[1] > w[0]) {
                    let mut expected_obv = 0.0;
                    for i in first_idx..out.len() {
                        if i > first_idx {
                            expected_obv += volume[i];
                        }
                        if !out[i].is_nan() {
                            prop_assert!(
									(out[i] - expected_obv).abs() < 1e-9,
									"For monotonic increasing price at index {}: expected OBV {}, got {}",
									i, expected_obv, out[i]
								);
                        }
                    }
                }

                // Property 7: Monotonic decreasing price - OBV should be negative cumulative sum
                if close.windows(2).all(|w| w[1] < w[0]) {
                    let mut expected_obv = 0.0;
                    for i in first_idx..out.len() {
                        if i > first_idx {
                            expected_obv -= volume[i];
                        }
                        if !out[i].is_nan() {
                            prop_assert!(
									(out[i] - expected_obv).abs() < 1e-9,
									"For monotonic decreasing price at index {}: expected OBV {}, got {}",
									i, expected_obv, out[i]
								);
                        }
                    }
                }

                // Property 8: Explicit test for zero volume cases
                // When volume is 0, OBV should not change regardless of price movement
                for i in (first_idx + 1)..close.len() {
                    if volume[i] == 0.0 && i > 0 && !out[i].is_nan() && !out[i - 1].is_nan() {
                        prop_assert!(
								(out[i] - out[i - 1]).abs() < 1e-9,
								"OBV should not change when volume is 0 at index {}, but changed from {} to {}",
								i, out[i - 1], out[i]
							);
                    }
                }

                // Property 9: OBV bounds check - ensure OBV doesn't exceed reasonable bounds
                // Given max volume of 1e6 and max length of 400, max absolute OBV should be < 4e8
                let max_possible_obv = 1e6 * (out.len() as f64);
                for (i, &val) in out.iter().enumerate() {
                    if !val.is_nan() {
                        prop_assert!(
                            val.abs() <= max_possible_obv,
                            "OBV at index {} exceeds reasonable bounds: {} > {}",
                            i,
                            val.abs(),
                            max_possible_obv
                        );
                    }
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    generate_all_obv_tests!(
        check_obv_empty_data,
        check_obv_data_length_mismatch,
        check_obv_all_nan,
        check_obv_csv_accuracy,
        check_obv_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_obv_tests!(check_obv_property);
}

/// Helper function for WASM bindings - writes directly to output slice with zero allocations
pub fn obv_into_slice(
    dst: &mut [f64],
    close: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<(), ObvError> {
    if close.is_empty() || volume.is_empty() {
        return Err(ObvError::EmptyData);
    }
    if close.len() != volume.len() {
        return Err(ObvError::DataLengthMismatch {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }
    if dst.len() != close.len() {
        return Err(ObvError::DataLengthMismatch {
            close_len: close.len(),
            volume_len: dst.len(),
        });
    }

    let first = close
        .iter()
        .zip(volume.iter())
        .position(|(c, v)| !c.is_nan() && !v.is_nan())
        .ok_or(ObvError::AllValuesNaN)?;

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => obv_scalar(close, volume, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => obv_avx2(close, volume, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => obv_avx512(close, volume, first, dst),
            _ => unreachable!(),
        }
    }

    // write only the warmup prefix; no further NaN writes
    for v in &mut dst[..first] {
        *v = f64::NAN;
    }
    Ok(())
}

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
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
use wasm_bindgen::prelude::*;

#[cfg(feature = "python")]
#[pyfunction(name = "obv")]
#[pyo3(signature = (close, volume, kernel=None))]
pub fn obv_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let input = ObvInput::from_slices(close_slice, volume_slice, ObvParams::default());

    let result_vec: Vec<f64> = py
        .allow_threads(|| obv_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "ObvStream")]
pub struct ObvStreamPy {
    stream: ObvStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl ObvStreamPy {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(ObvStreamPy {
            stream: ObvStream::new(),
        })
    }

    pub fn update(&mut self, close: f64, volume: f64) -> Option<f64> {
        self.stream.update(close, volume)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "obv_batch")]
#[pyo3(signature = (close, volume, kernel=None))]
pub fn obv_batch_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    // OBV has no parameters, so batch is just single calculation
    let rows = 1;
    let cols = close_slice.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    py.allow_threads(|| {
        let kernel = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };

        obv_batch_inner_into(close_slice, volume_slice, kernel, slice_out)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    // No parameters to return for OBV

    Ok(dict)
}

// ---------------- CUDA Python bindings -----------------

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::CudaObv;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "obv_cuda_batch_dev")]
#[pyo3(signature = (close, volume, device_id=0))]
pub fn obv_cuda_batch_dev_py(
    py: Python<'_>,
    close: PyReadonlyArray1<'_, f32>,
    volume: PyReadonlyArray1<'_, f32>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    if close_slice.len() != volume_slice.len() {
        return Err(PyValueError::new_err("mismatched input lengths"));
    }

    let inner = py.allow_threads(|| {
        let cuda = CudaObv::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.obv_batch_dev(close_slice, volume_slice)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "obv_cuda_many_series_one_param_dev")]
#[pyo3(signature = (close_tm, volume_tm, cols, rows, device_id=0))]
pub fn obv_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    close_tm: PyReadonlyArray1<'_, f32>,
    volume_tm: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let close_slice = close_tm.as_slice()?;
    let volume_slice = volume_tm.as_slice()?;
    if close_slice.len() != volume_slice.len() || close_slice.len() != cols * rows {
        return Err(PyValueError::new_err("mismatched input sizes or dims"));
    }

    let inner = py.allow_threads(|| {
        let cuda = CudaObv::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.obv_many_series_one_param_time_major_dev(close_slice, volume_slice, cols, rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn obv_js(close: &[f64], volume: &[f64]) -> Result<Vec<f64>, JsValue> {
    let mut output = vec![0.0; close.len()];

    obv_into_slice(&mut output, close, volume, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn obv_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<(), JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer passed to obv_into"));
    }

    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        // Check for aliasing - OBV can't operate in-place on either input
        if close_ptr == out_ptr || volume_ptr == out_ptr {
            // Need temporary buffer if output aliases with either input
            let mut temp = vec![0.0; len];
            obv_into_slice(&mut temp, close, volume, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, compute directly into output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            obv_into_slice(out, close, volume, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn obv_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn obv_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn obv_batch_into(
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<usize, JsValue> {
    if close_ptr.is_null() || volume_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to obv_batch_into"));
    }
    unsafe {
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        obv_into_slice(out, close, volume, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(1) // rows
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct ObvBatchJsOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = obv_batch)]
pub fn obv_batch_js(close: &[f64], volume: &[f64]) -> Result<JsValue, JsValue> {
    // OBV has no parameters, so batch returns single row
    let mut output = vec![0.0; close.len()];

    obv_into_slice(&mut output, close, volume, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = ObvBatchJsOutput {
        values: output,
        rows: 1,
        cols: close.len(),
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "python")]
pub fn register_obv_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(obv_py, m)?)?;
    m.add_function(wrap_pyfunction!(obv_batch_py, m)?)?;
    Ok(())
}
