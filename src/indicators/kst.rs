//! # Know Sure Thing (KST)
//!
//! KST is a momentum oscillator based on the smoothed rate-of-change (ROC) values of four different time frames.
//! It combines multiple ROC calculations with different periods and applies weighted smoothing to generate
//! momentum signals.
//!
//! ## Parameters
//! - **sma_period1**: Smoothing period for the first ROC (default: 10)
//! - **sma_period2**: Smoothing period for the second ROC (default: 10)
//! - **sma_period3**: Smoothing period for the third ROC (default: 10)
//! - **sma_period4**: Smoothing period for the fourth ROC (default: 15)
//! - **roc_period1**: Period for the first ROC calculation (default: 10)
//! - **roc_period2**: Period for the second ROC calculation (default: 15)
//! - **roc_period3**: Period for the third ROC calculation (default: 20)
//! - **roc_period4**: Period for the fourth ROC calculation (default: 30)
//! - **signal_period**: Smoothing period for the signal line (default: 9)
//!
//! ## Returns
//! - **`Ok(KstOutput)`** containing:
//!   - `line`: The KST oscillator values
//!   - `signal`: The smoothed signal line
//!
//! ## Developer Notes
//! ### Implementation Status
//! - SIMD (per-row): Disabled by design — loop-carried ring-buffer deps make time-lane SIMD ineffective; runtime selection short-circuits to scalar.
//! - AVX2/AVX512 stubs: Present for parity but delegate to the scalar path.
//! - Scalar kernel: Single-pass, builds line and signal together; avoids modulo in hot paths; uses mul_add; stack-first rings (WASM-safe).
//! - Memory: Uses `alloc_with_nan_prefix` and batch helpers; no O(N) temporaries beyond outputs and small rings.
//! - Batch: Parallel-by-row; row-specific SIMD/caching not yet implemented (see TODO below).
//!
//! ### TODO - Performance Improvements
//! - [ ] Row-specific batch optimization: cache ROC streams shared across rows (by identical r1..r4), then update only SMA rings per row.
//! - [ ] If revisiting SIMD, focus on batch lane-packing across rows; per-row SIMD over 4 ROCs is unlikely to beat scalar consistently.

use crate::indicators::moving_averages::sma::{
    sma, SmaData, SmaError, SmaInput, SmaOutput, SmaParams,
};
use crate::indicators::roc::{roc, RocData, RocError, RocInput, RocOutput, RocParams};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
// aligned_vec imports removed - not used in KST
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[cfg(feature = "wasm")]
use js_sys;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum KstData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct KstOutput {
    pub line: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct KstParams {
    pub sma_period1: Option<usize>,
    pub sma_period2: Option<usize>,
    pub sma_period3: Option<usize>,
    pub sma_period4: Option<usize>,
    pub roc_period1: Option<usize>,
    pub roc_period2: Option<usize>,
    pub roc_period3: Option<usize>,
    pub roc_period4: Option<usize>,
    pub signal_period: Option<usize>,
}

impl Default for KstParams {
    fn default() -> Self {
        Self {
            sma_period1: Some(10),
            sma_period2: Some(10),
            sma_period3: Some(10),
            sma_period4: Some(15),
            roc_period1: Some(10),
            roc_period2: Some(15),
            roc_period3: Some(20),
            roc_period4: Some(30),
            signal_period: Some(9),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KstInput<'a> {
    pub data: KstData<'a>,
    pub params: KstParams,
}

impl<'a> AsRef<[f64]> for KstInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            KstData::Slice(slice) => slice,
            KstData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

impl<'a> KstInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: KstParams) -> Self {
        Self {
            data: KstData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: KstParams) -> Self {
        Self {
            data: KstData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", KstParams::default())
    }
    #[inline]
    pub fn get_sma_period1(&self) -> usize {
        self.params.sma_period1.unwrap_or(10)
    }
    #[inline]
    pub fn get_sma_period2(&self) -> usize {
        self.params.sma_period2.unwrap_or(10)
    }
    #[inline]
    pub fn get_sma_period3(&self) -> usize {
        self.params.sma_period3.unwrap_or(10)
    }
    #[inline]
    pub fn get_sma_period4(&self) -> usize {
        self.params.sma_period4.unwrap_or(15)
    }
    #[inline]
    pub fn get_roc_period1(&self) -> usize {
        self.params.roc_period1.unwrap_or(10)
    }
    #[inline]
    pub fn get_roc_period2(&self) -> usize {
        self.params.roc_period2.unwrap_or(15)
    }
    #[inline]
    pub fn get_roc_period3(&self) -> usize {
        self.params.roc_period3.unwrap_or(20)
    }
    #[inline]
    pub fn get_roc_period4(&self) -> usize {
        self.params.roc_period4.unwrap_or(30)
    }
    #[inline]
    pub fn get_signal_period(&self) -> usize {
        self.params.signal_period.unwrap_or(9)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct KstBuilder {
    sma_period1: Option<usize>,
    sma_period2: Option<usize>,
    sma_period3: Option<usize>,
    sma_period4: Option<usize>,
    roc_period1: Option<usize>,
    roc_period2: Option<usize>,
    roc_period3: Option<usize>,
    roc_period4: Option<usize>,
    signal_period: Option<usize>,
    kernel: Kernel,
}

impl Default for KstBuilder {
    fn default() -> Self {
        Self {
            sma_period1: None,
            sma_period2: None,
            sma_period3: None,
            sma_period4: None,
            roc_period1: None,
            roc_period2: None,
            roc_period3: None,
            roc_period4: None,
            signal_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl KstBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn sma_period1(mut self, n: usize) -> Self {
        self.sma_period1 = Some(n);
        self
    }
    #[inline(always)]
    pub fn sma_period2(mut self, n: usize) -> Self {
        self.sma_period2 = Some(n);
        self
    }
    #[inline(always)]
    pub fn sma_period3(mut self, n: usize) -> Self {
        self.sma_period3 = Some(n);
        self
    }
    #[inline(always)]
    pub fn sma_period4(mut self, n: usize) -> Self {
        self.sma_period4 = Some(n);
        self
    }
    #[inline(always)]
    pub fn roc_period1(mut self, n: usize) -> Self {
        self.roc_period1 = Some(n);
        self
    }
    #[inline(always)]
    pub fn roc_period2(mut self, n: usize) -> Self {
        self.roc_period2 = Some(n);
        self
    }
    #[inline(always)]
    pub fn roc_period3(mut self, n: usize) -> Self {
        self.roc_period3 = Some(n);
        self
    }
    #[inline(always)]
    pub fn roc_period4(mut self, n: usize) -> Self {
        self.roc_period4 = Some(n);
        self
    }
    #[inline(always)]
    pub fn signal_period(mut self, n: usize) -> Self {
        self.signal_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<KstOutput, KstError> {
        let p = KstParams {
            sma_period1: self.sma_period1,
            sma_period2: self.sma_period2,
            sma_period3: self.sma_period3,
            sma_period4: self.sma_period4,
            roc_period1: self.roc_period1,
            roc_period2: self.roc_period2,
            roc_period3: self.roc_period3,
            roc_period4: self.roc_period4,
            signal_period: self.signal_period,
        };
        let i = KstInput::from_candles(c, "close", p);
        kst_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<KstOutput, KstError> {
        let p = KstParams {
            sma_period1: self.sma_period1,
            sma_period2: self.sma_period2,
            sma_period3: self.sma_period3,
            sma_period4: self.sma_period4,
            roc_period1: self.roc_period1,
            roc_period2: self.roc_period2,
            roc_period3: self.roc_period3,
            roc_period4: self.roc_period4,
            signal_period: self.signal_period,
        };
        let i = KstInput::from_slice(d, p);
        kst_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<KstStream, KstError> {
        let p = KstParams {
            sma_period1: self.sma_period1,
            sma_period2: self.sma_period2,
            sma_period3: self.sma_period3,
            sma_period4: self.sma_period4,
            roc_period1: self.roc_period1,
            roc_period2: self.roc_period2,
            roc_period3: self.roc_period3,
            roc_period4: self.roc_period4,
            signal_period: self.signal_period,
        };
        KstStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum KstError {
    #[error("kst: {0}")]
    Roc(#[from] RocError),
    #[error("kst: {0}")]
    Sma(#[from] SmaError),
    #[error("kst: Input data slice is empty.")]
    EmptyInputData,
    #[error("kst: All values are NaN.")]
    AllValuesNaN,
    #[error("kst: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("kst: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn kst(input: &KstInput) -> Result<KstOutput, KstError> {
    kst_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn kst_prepare<'a>(
    input: &'a KstInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        // periods
        (usize, usize, usize, usize), // s1..s4
        (usize, usize, usize, usize), // r1..r4
        usize,                        // sig
        usize,                        // first
        usize,                        // warm_line = max(ri+si-1)
        usize,                        // warm_sig  = warm_line + sig - 1
        Kernel,                       // chosen
    ),
    KstError,
> {
    let data = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(KstError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(KstError::AllValuesNaN)?;

    let s1 = input.get_sma_period1();
    let s2 = input.get_sma_period2();
    let s3 = input.get_sma_period3();
    let s4 = input.get_sma_period4();
    let r1 = input.get_roc_period1();
    let r2 = input.get_roc_period2();
    let r3 = input.get_roc_period3();
    let r4 = input.get_roc_period4();
    let sig = input.get_signal_period();

    // explicit period checks
    for &p in [s1, s2, s3, s4, r1, r2, r3, r4, sig].iter() {
        if p == 0 || p > len {
            return Err(KstError::InvalidPeriod {
                period: p,
                data_len: len,
            });
        }
    }

    let warm1 = r1 + s1 - 1;
    let warm2 = r2 + s2 - 1;
    let warm3 = r3 + s3 - 1;
    let warm4 = r4 + s4 - 1;
    let warm_line = warm1.max(warm2).max(warm3).max(warm4);
    if len - first < warm_line {
        return Err(KstError::NotEnoughValidData {
            needed: warm_line,
            valid: len - first,
        });
    }
    let warm_sig = warm_line + sig - 1;
    if warm_sig > len {
        return Err(KstError::NotEnoughValidData {
            needed: warm_sig,
            valid: len,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    Ok((
        data,
        (s1, s2, s3, s4),
        (r1, r2, r3, r4),
        sig,
        first,
        warm_line,
        warm_sig,
        chosen,
    ))
}

#[inline(always)]
fn kst_compute_into(
    data: &[f64],
    s: (usize, usize, usize, usize),
    r: (usize, usize, usize, usize),
    sig: usize,
    first: usize,
    warm_line: usize,
    warm_sig: usize,
    out_line: &mut [f64],
    out_sig: &mut [f64],
) {
    let len = data.len();
    let (s1, s2, s3, s4) = s;
    let (r1, r2, r3, r4) = r;

    // ---- Small stack buffers to avoid wasm memory.grow detaching JS views
    const STACK: usize = 256;
    let mut sb1 = [0.0f64; STACK];
    let mut sb2 = [0.0f64; STACK];
    let mut sb3 = [0.0f64; STACK];
    let mut sb4 = [0.0f64; STACK];
    let mut sbs = [0.0f64; STACK]; // signal ring

    let mut v1_heap;
    let mut v2_heap;
    let mut v3_heap;
    let mut v4_heap;
    let mut vs_heap;

    // Select ring buffers (stack or heap) for SMA(ROC) and signal
    let (b1, b2, b3, b4, sbuf): (&mut [f64], &mut [f64], &mut [f64], &mut [f64], &mut [f64]) = {
        v1_heap = if s1 > STACK {
            vec![0.0; s1]
        } else {
            Vec::new()
        };
        v2_heap = if s2 > STACK {
            vec![0.0; s2]
        } else {
            Vec::new()
        };
        v3_heap = if s3 > STACK {
            vec![0.0; s3]
        } else {
            Vec::new()
        };
        v4_heap = if s4 > STACK {
            vec![0.0; s4]
        } else {
            Vec::new()
        };
        vs_heap = if sig > STACK {
            vec![0.0; sig]
        } else {
            Vec::new()
        };

        let b1 = if s1 <= STACK {
            &mut sb1[..s1]
        } else {
            v1_heap.as_mut_slice()
        };
        let b2 = if s2 <= STACK {
            &mut sb2[..s2]
        } else {
            v2_heap.as_mut_slice()
        };
        let b3 = if s3 <= STACK {
            &mut sb3[..s3]
        } else {
            v3_heap.as_mut_slice()
        };
        let b4 = if s4 <= STACK {
            &mut sb4[..s4]
        } else {
            v4_heap.as_mut_slice()
        };
        let sbuf = if sig <= STACK {
            &mut sbs[..sig]
        } else {
            vs_heap.as_mut_slice()
        };
        (b1, b2, b3, b4, sbuf)
    };

    // State for four SMA(ROC) rings
    let mut i1 = 0usize;
    let mut i2 = 0usize;
    let mut i3 = 0usize;
    let mut i4 = 0usize;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;
    let mut sum4 = 0.0f64;

    // Precompute reciprocals and weights (use mul_add later)
    let inv1 = 1.0 / (s1 as f64);
    let inv2 = 1.0 / (s2 as f64);
    let inv3 = 1.0 / (s3 as f64);
    let inv4 = 1.0 / (s4 as f64);
    let w2 = inv2 + inv2; // 2*inv2
    let w3 = inv3 + inv3 + inv3; // 3*inv3
    let w4 = (4.0f64) * inv4; // 4*inv4

    // Absolute indices where each ROC becomes available
    let start1 = first + r1;
    let start2 = first + r2;
    let start3 = first + r3;
    let start4 = first + r4;

    // Absolute warmups for line/signal
    let start_line = first + warm_line; // first index allowed to write KST line
    let warm_sig_abs = first + warm_sig; // first index allowed to write signal

    // Signal ring state (built up as soon as line starts)
    let mut sidx = 0usize;
    let mut ssum = 0.0f64;
    let mut sbuilt = 0usize; // how many valid entries currently in signal ring [0..=sig]

    // Fast helpers
    #[inline(always)]
    fn safe_roc(curr: f64, prev: f64) -> f64 {
        // Matches prior semantics: zeros when not computable; ignores non-finite inputs.
        if prev != 0.0 && curr.is_finite() && prev.is_finite() {
            ((curr / prev) - 1.0) * 100.0
        } else {
            0.0
        }
    }

    // Hot loop
    unsafe {
        let out_line_ptr = out_line.as_mut_ptr();
        let out_sig_ptr = out_sig.as_mut_ptr();
        let data_ptr = data.as_ptr();

        let b1_ptr = b1.as_mut_ptr();
        let b2_ptr = b2.as_mut_ptr();
        let b3_ptr = b3.as_mut_ptr();
        let b4_ptr = b4.as_mut_ptr();
        let sb_ptr = sbuf.as_mut_ptr();

        // Small inlined "ring update": sum += v - old; buf[idx] = v; idx=(idx+1) with branchless wrap
        #[inline(always)]
        unsafe fn ring_update(buf: *mut f64, idx: &mut usize, cap: usize, sum: &mut f64, v: f64) {
            let old = *buf.add(*idx);
            *sum = (*sum) + (v - old);
            *buf.add(*idx) = v;
            *idx += 1;
            if *idx == cap {
                *idx = 0;
            }
        }

        for i in first..len {
            // Current price (may be NaN; safe_roc handles it)
            let x = *data_ptr.add(i);

            // Four ROC updates become active at their own starts.
            if i >= start1 {
                let p = *data_ptr.add(i - r1);
                let v = safe_roc(x, p);
                ring_update(b1_ptr, &mut i1, s1, &mut sum1, v);
            }
            if i >= start2 {
                let p = *data_ptr.add(i - r2);
                let v = safe_roc(x, p);
                ring_update(b2_ptr, &mut i2, s2, &mut sum2, v);
            }
            if i >= start3 {
                let p = *data_ptr.add(i - r3);
                let v = safe_roc(x, p);
                ring_update(b3_ptr, &mut i3, s3, &mut sum3, v);
            }
            if i >= start4 {
                let p = *data_ptr.add(i - r4);
                let v = safe_roc(x, p);
                ring_update(b4_ptr, &mut i4, s4, &mut sum4, v);
            }

            // Gate writes until all components are available
            if i < start_line {
                continue;
            }

            // KST line using FMAs: sum1*inv1 + 2*sum2*inv2 + 3*sum3*inv3 + 4*sum4*inv4
            // Evaluate as nested mul_add to reduce rounding and latency.
            let kst = sum1.mul_add(inv1, sum2.mul_add(w2, sum3.mul_add(w3, sum4 * w4)));
            *out_line_ptr.add(i) = kst;

            // Build signal in the same pass (simple SMA over last `sig` KST values)
            // Note: sbuf was zero-initialized; we update ssum += kst - old_value
            if sbuilt < sig {
                let old = *sb_ptr.add(sidx); // initially 0.0
                ssum += kst - old;
                *sb_ptr.add(sidx) = kst;
                sidx += 1;
                if sidx == sig {
                    sidx = 0;
                }
                sbuilt += 1;

                if i >= warm_sig_abs {
                    *out_sig_ptr.add(i) = ssum / (sig as f64);
                }
            } else {
                let old = *sb_ptr.add(sidx);
                ssum += kst - old;
                *sb_ptr.add(sidx) = kst;
                sidx += 1;
                if sidx == sig {
                    sidx = 0;
                }
                *out_sig_ptr.add(i) = ssum / (sig as f64);
            }
        }
    }
}

pub fn kst_with_kernel(input: &KstInput, kernel: Kernel) -> Result<KstOutput, KstError> {
    let (data, s, r, sig, first, warm_line, warm_sig, chosen) = kst_prepare(input, kernel)?;
    let len = data.len();

    // Adjust warmup to account for NaN prefix
    let actual_warm_line = first + warm_line;
    let actual_warm_sig = first + warm_sig;
    let mut line = alloc_with_nan_prefix(len, actual_warm_line);
    let mut signal = alloc_with_nan_prefix(len, actual_warm_sig);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                kst_compute_into(
                    data,
                    s,
                    r,
                    sig,
                    first,
                    warm_line,
                    warm_sig,
                    &mut line,
                    &mut signal,
                );
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                // keep stubs, call scalar body
                kst_compute_into(
                    data,
                    s,
                    r,
                    sig,
                    first,
                    warm_line,
                    warm_sig,
                    &mut line,
                    &mut signal,
                );
            }
            _ => unreachable!(),
        }
    }
    Ok(KstOutput { line, signal })
}

#[inline]
pub fn kst_into_slice(
    out_line: &mut [f64],
    out_signal: &mut [f64],
    input: &KstInput,
    kernel: Kernel,
) -> Result<(), KstError> {
    let (data, s, r, sig, first, warm_line, warm_sig, _chosen) = kst_prepare(input, kernel)?;
    if out_line.len() != data.len() || out_signal.len() != data.len() {
        return Err(KstError::InvalidPeriod {
            period: out_line.len(),
            data_len: data.len(),
        });
    }

    kst_compute_into(
        data, s, r, sig, first, warm_line, warm_sig, out_line, out_signal,
    );

    let prefix_line = (first + warm_line).min(out_line.len());
    let prefix_sig = (first + warm_sig).min(out_signal.len());
    for v in &mut out_line[..prefix_line] {
        *v = f64::NAN;
    }
    for v in &mut out_signal[..prefix_sig] {
        *v = f64::NAN;
    }
    Ok(())
}

// The old kst_scalar function has been removed as it's no longer needed
// The functionality is now handled by kst_compute_into

// SIMD stubs - these remain for API parity but use the scalar implementation
// Future optimization: implement actual SIMD versions

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub(crate) unsafe fn kst_avx2(
    _input: &KstInput,
    _first: usize,
    _len: usize,
) -> Result<KstOutput, KstError> {
    // Stub: would implement AVX2 optimization
    unreachable!("AVX2 stub should not be called directly")
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub(crate) unsafe fn kst_avx512(
    _input: &KstInput,
    _first: usize,
    len: usize,
) -> Result<KstOutput, KstError> {
    // Dispatch to long/short stub
    if len <= 32 {
        kst_avx512_short(_input, _first, len)
    } else {
        kst_avx512_long(_input, _first, len)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub(crate) unsafe fn kst_avx512_short(
    _input: &KstInput,
    _first: usize,
    _len: usize,
) -> Result<KstOutput, KstError> {
    // Stub: would implement AVX512 short optimization
    unreachable!("AVX512 short stub should not be called directly")
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub(crate) unsafe fn kst_avx512_long(
    _input: &KstInput,
    _first: usize,
    _len: usize,
) -> Result<KstOutput, KstError> {
    // Stub: would implement AVX512 long optimization
    unreachable!("AVX512 long stub should not be called directly")
}

#[inline]
pub fn kst_batch_with_kernel(
    data: &[f64],
    sweep: &KstBatchRange,
    k: Kernel,
) -> Result<KstBatchOutput, KstError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(KstError::InvalidPeriod {
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
    kst_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct KstBatchRange {
    pub sma_period1: (usize, usize, usize),
    pub sma_period2: (usize, usize, usize),
    pub sma_period3: (usize, usize, usize),
    pub sma_period4: (usize, usize, usize),
    pub roc_period1: (usize, usize, usize),
    pub roc_period2: (usize, usize, usize),
    pub roc_period3: (usize, usize, usize),
    pub roc_period4: (usize, usize, usize),
    pub signal_period: (usize, usize, usize),
}

impl Default for KstBatchRange {
    fn default() -> Self {
        Self {
            sma_period1: (10, 10, 0),
            sma_period2: (10, 10, 0),
            sma_period3: (10, 10, 0),
            sma_period4: (15, 15, 0),
            roc_period1: (10, 10, 0),
            roc_period2: (15, 15, 0),
            roc_period3: (20, 20, 0),
            roc_period4: (30, 30, 0),
            signal_period: (9, 9, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct KstBatchBuilder {
    range: KstBatchRange,
    kernel: Kernel,
}

impl KstBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn sma_period1_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.sma_period1 = (start, end, step);
        self
    }
    pub fn sma_period2_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.sma_period2 = (start, end, step);
        self
    }
    pub fn sma_period3_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.sma_period3 = (start, end, step);
        self
    }
    pub fn sma_period4_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.sma_period4 = (start, end, step);
        self
    }
    pub fn roc_period1_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.roc_period1 = (start, end, step);
        self
    }
    pub fn roc_period2_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.roc_period2 = (start, end, step);
        self
    }
    pub fn roc_period3_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.roc_period3 = (start, end, step);
        self
    }
    pub fn roc_period4_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.roc_period4 = (start, end, step);
        self
    }
    pub fn signal_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.signal_period = (start, end, step);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<KstBatchOutput, KstError> {
        kst_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<KstBatchOutput, KstError> {
        KstBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<KstBatchOutput, KstError> {
        self.apply_slice(source_type(c, src))
    }
    pub fn with_default_candles(c: &Candles) -> Result<KstBatchOutput, KstError> {
        KstBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct KstBatchOutput {
    pub lines: Vec<f64>,
    pub signals: Vec<f64>,
    pub combos: Vec<KstParams>,
    pub rows: usize,
    pub cols: usize,
}
impl KstBatchOutput {
    pub fn row_for_params(&self, p: &KstParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.sma_period1.unwrap_or(10) == p.sma_period1.unwrap_or(10)
                && c.sma_period2.unwrap_or(10) == p.sma_period2.unwrap_or(10)
                && c.sma_period3.unwrap_or(10) == p.sma_period3.unwrap_or(10)
                && c.sma_period4.unwrap_or(15) == p.sma_period4.unwrap_or(15)
                && c.roc_period1.unwrap_or(10) == p.roc_period1.unwrap_or(10)
                && c.roc_period2.unwrap_or(15) == p.roc_period2.unwrap_or(15)
                && c.roc_period3.unwrap_or(20) == p.roc_period3.unwrap_or(20)
                && c.roc_period4.unwrap_or(30) == p.roc_period4.unwrap_or(30)
                && c.signal_period.unwrap_or(9) == p.signal_period.unwrap_or(9)
        })
    }
    pub fn lines_for(&self, p: &KstParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.lines[start..start + self.cols]
        })
    }
    pub fn signals_for(&self, p: &KstParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.signals[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &KstBatchRange) -> Vec<KstParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let s1 = axis(r.sma_period1);
    let s2 = axis(r.sma_period2);
    let s3 = axis(r.sma_period3);
    let s4 = axis(r.sma_period4);
    let r1 = axis(r.roc_period1);
    let r2 = axis(r.roc_period2);
    let r3 = axis(r.roc_period3);
    let r4 = axis(r.roc_period4);
    let sig = axis(r.signal_period);

    let mut out = Vec::with_capacity(
        s1.len()
            * s2.len()
            * s3.len()
            * s4.len()
            * r1.len()
            * r2.len()
            * r3.len()
            * r4.len()
            * sig.len(),
    );
    for &s1v in &s1 {
        for &s2v in &s2 {
            for &s3v in &s3 {
                for &s4v in &s4 {
                    for &r1v in &r1 {
                        for &r2v in &r2 {
                            for &r3v in &r3 {
                                for &r4v in &r4 {
                                    for &sigv in &sig {
                                        out.push(KstParams {
                                            sma_period1: Some(s1v),
                                            sma_period2: Some(s2v),
                                            sma_period3: Some(s3v),
                                            sma_period4: Some(s4v),
                                            roc_period1: Some(r1v),
                                            roc_period2: Some(r2v),
                                            roc_period3: Some(r3v),
                                            roc_period4: Some(r4v),
                                            signal_period: Some(sigv),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

#[inline(always)]
pub fn kst_batch_slice(
    data: &[f64],
    sweep: &KstBatchRange,
    kern: Kernel,
) -> Result<KstBatchOutput, KstError> {
    kst_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn kst_batch_par_slice(
    data: &[f64],
    sweep: &KstBatchRange,
    kern: Kernel,
) -> Result<KstBatchOutput, KstError> {
    kst_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn kst_batch_inner(
    data: &[f64],
    sweep: &KstBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<KstBatchOutput, KstError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(KstError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let cols = data.len();
    if cols == 0 {
        return Err(KstError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(KstError::AllValuesNaN)?;

    // per-row warms
    let mut warm_line = Vec::with_capacity(combos.len());
    let mut warm_sig = Vec::with_capacity(combos.len());
    for c in &combos {
        let s1 = c.sma_period1.unwrap();
        let s2 = c.sma_period2.unwrap();
        let s3 = c.sma_period3.unwrap();
        let s4 = c.sma_period4.unwrap();
        let r1 = c.roc_period1.unwrap();
        let r2 = c.roc_period2.unwrap();
        let r3 = c.roc_period3.unwrap();
        let r4 = c.roc_period4.unwrap();
        let sig = c.signal_period.unwrap();
        let wl = (r1 + s1 - 1)
            .max(r2 + s2 - 1)
            .max(r3 + s3 - 1)
            .max(r4 + s4 - 1);
        warm_line.push(wl);
        warm_sig.push(wl + sig - 1);
    }

    // allocate two uninit matrices then prefix NaNs
    let rows = combos.len();
    let mut line_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);
    let warm_line_abs: Vec<usize> = warm_line.iter().map(|&wl| first + wl).collect();
    let warm_sig_abs: Vec<usize> = warm_sig.iter().map(|&ws| first + ws).collect();
    init_matrix_prefixes(&mut line_mu, cols, &warm_line_abs);
    init_matrix_prefixes(&mut signal_mu, cols, &warm_sig_abs);

    // cast to &mut [f64] safely via ManuallyDrop
    let mut line_guard = core::mem::ManuallyDrop::new(line_mu);
    let mut signal_guard = core::mem::ManuallyDrop::new(signal_mu);
    let line_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(line_guard.as_mut_ptr() as *mut f64, line_guard.len())
    };
    let signal_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };

    // resolve scalar for parity (stubs OK)
    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd = match actual {
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2Batch => Kernel::Scalar, // Use scalar if AVX2 not available
        Kernel::Avx512Batch => Kernel::Scalar, // Use scalar if AVX512 not available
        _ => Kernel::Scalar,
    };

    // --- Row-specific batch optimization: ROC caching across rows with identical (r1..r4)
    use std::collections::HashMap;
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    struct R4(usize, usize, usize, usize);

    // Group rows by ROC tuple
    let mut groups: HashMap<R4, Vec<usize>> = HashMap::new();
    for (idx, prm) in combos.iter().enumerate() {
        groups
            .entry(R4(
                prm.roc_period1.unwrap(),
                prm.roc_period2.unwrap(),
                prm.roc_period3.unwrap(),
                prm.roc_period4.unwrap(),
            ))
            .or_default()
            .push(idx);
    }

    // Precompute ROC streams per group
    fn compute_roc_streams(
        data: &[f64],
        first: usize,
        r: (usize, usize, usize, usize),
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let len = data.len();
        let (r1, r2, r3, r4) = r;
        let mut v1 = vec![0.0f64; len];
        let mut v2 = vec![0.0f64; len];
        let mut v3 = vec![0.0f64; len];
        let mut v4 = vec![0.0f64; len];
        for i in first..len {
            let x = data[i];
            if i >= first + r1 {
                let p = data[i - r1];
                if x.is_finite() && p.is_finite() && p != 0.0 {
                    v1[i] = ((x / p) - 1.0) * 100.0;
                }
            }
            if i >= first + r2 {
                let p = data[i - r2];
                if x.is_finite() && p.is_finite() && p != 0.0 {
                    v2[i] = ((x / p) - 1.0) * 100.0;
                }
            }
            if i >= first + r3 {
                let p = data[i - r3];
                if x.is_finite() && p.is_finite() && p != 0.0 {
                    v3[i] = ((x / p) - 1.0) * 100.0;
                }
            }
            if i >= first + r4 {
                let p = data[i - r4];
                if x.is_finite() && p.is_finite() && p != 0.0 {
                    v4[i] = ((x / p) - 1.0) * 100.0;
                }
            }
        }
        (v1, v2, v3, v4)
    }

    struct Streams {
        v1: Vec<f64>,
        v2: Vec<f64>,
        v3: Vec<f64>,
        v4: Vec<f64>,
    }

    let mut streams_map: HashMap<R4, Streams> = HashMap::with_capacity(groups.len());
    for (key @ R4(r1, r2, r3, r4), _rows) in groups.iter() {
        let (v1, v2, v3, v4) = compute_roc_streams(data, first, (*r1, *r2, *r3, *r4));
        streams_map.insert(*key, Streams { v1, v2, v3, v4 });
    }

    // Per-row compute using shared ROC streams
    let do_row = |row: usize, ldst: &mut [f64], sdst: &mut [f64]| {
        let prm = &combos[row];
        let s = (
            prm.sma_period1.unwrap(),
            prm.sma_period2.unwrap(),
            prm.sma_period3.unwrap(),
            prm.sma_period4.unwrap(),
        );
        let r = (
            prm.roc_period1.unwrap(),
            prm.roc_period2.unwrap(),
            prm.roc_period3.unwrap(),
            prm.roc_period4.unwrap(),
        );
        let sig = prm.signal_period.unwrap();
        let wl = (r.0 + s.0 - 1)
            .max(r.1 + s.1 - 1)
            .max(r.2 + s.2 - 1)
            .max(r.3 + s.3 - 1);
        let ws = wl + sig - 1;

        // Fetch streams
        let key = R4(r.0, r.1, r.2, r.3);
        let st = streams_map.get(&key).unwrap();

        // Compute from streams: SMA(roc) rings + weighted sum + signal ring
        let len = ldst.len();
        let (s1, s2, s3, s4) = s;
        let inv1 = 1.0 / (s1 as f64);
        let inv2 = 1.0 / (s2 as f64);
        let inv3 = 1.0 / (s3 as f64);
        let inv4 = 1.0 / (s4 as f64);
        let w2 = inv2 + inv2;
        let w3 = inv3 + inv3 + inv3;
        let w4 = 4.0f64 * inv4;

        // rings
        let mut b1 = vec![0.0f64; s1];
        let mut b2 = vec![0.0f64; s2];
        let mut b3 = vec![0.0f64; s3];
        let mut b4 = vec![0.0f64; s4];
        let mut i1 = 0usize;
        let mut i2 = 0usize;
        let mut i3 = 0usize;
        let mut i4 = 0usize;
        let mut sum1 = 0.0f64;
        let mut sum2 = 0.0f64;
        let mut sum3 = 0.0f64;
        let mut sum4 = 0.0f64;

        let start_line = first + wl;
        let warm_sig_abs = first + ws;

        // signal ring
        let mut sbuf = vec![0.0f64; sig];
        let mut sidx = 0usize;
        let mut ssum = 0.0f64;
        let mut sbuilt = 0usize;

        unsafe {
            let b1p = b1.as_mut_ptr();
            let b2p = b2.as_mut_ptr();
            let b3p = b3.as_mut_ptr();
            let b4p = b4.as_mut_ptr();
            let sbp = sbuf.as_mut_ptr();
            let lptr = ldst.as_mut_ptr();
            let sptr = sdst.as_mut_ptr();
            let v1p = st.v1.as_ptr();
            let v2p = st.v2.as_ptr();
            let v3p = st.v3.as_ptr();
            let v4p = st.v4.as_ptr();

            #[inline(always)]
            unsafe fn ring_update(
                buf: *mut f64,
                idx: &mut usize,
                cap: usize,
                sum: &mut f64,
                v: f64,
            ) {
                let old = *buf.add(*idx);
                *sum = (*sum) + (v - old);
                *buf.add(*idx) = v;
                *idx += 1;
                if *idx == cap {
                    *idx = 0;
                }
            }

            for i in first..len {
                let v1 = *v1p.add(i);
                let v2 = *v2p.add(i);
                let v3 = *v3p.add(i);
                let v4 = *v4p.add(i);

                ring_update(b1p, &mut i1, s1, &mut sum1, v1);
                ring_update(b2p, &mut i2, s2, &mut sum2, v2);
                ring_update(b3p, &mut i3, s3, &mut sum3, v3);
                ring_update(b4p, &mut i4, s4, &mut sum4, v4);

                if i < start_line {
                    continue;
                }

                let kst = sum1.mul_add(inv1, sum2.mul_add(w2, sum3.mul_add(w3, sum4 * w4)));
                *lptr.add(i) = kst;

                if sbuilt < sig {
                    let old = *sbp.add(sidx);
                    ssum += kst - old;
                    *sbp.add(sidx) = kst;
                    sidx += 1;
                    if sidx == sig {
                        sidx = 0;
                    }
                    sbuilt += 1;
                    if i >= warm_sig_abs {
                        *sptr.add(i) = ssum / (sig as f64);
                    }
                } else {
                    let old = *sbp.add(sidx);
                    ssum += kst - old;
                    *sbp.add(sidx) = kst;
                    sidx += 1;
                    if sidx == sig {
                        sidx = 0;
                    }
                    *sptr.add(i) = ssum / (sig as f64);
                }
            }
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        line_out
            .par_chunks_mut(cols)
            .zip(signal_out.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (l, s))| do_row(row, l, s));
        #[cfg(target_arch = "wasm32")]
        for (row, (l, s)) in line_out
            .chunks_mut(cols)
            .zip(signal_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, l, s);
        }
    } else {
        for (row, (l, s)) in line_out
            .chunks_mut(cols)
            .zip(signal_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, l, s);
        }
    }

    // reconstitute Vecs without copy
    let lines = unsafe {
        Vec::from_raw_parts(
            line_guard.as_mut_ptr() as *mut f64,
            line_guard.len(),
            line_guard.capacity(),
        )
    };
    let signals = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };

    Ok(KstBatchOutput {
        lines,
        signals,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn kst_batch_inner_into(
    data: &[f64],
    sweep: &KstBatchRange,
    kern: Kernel,
    parallel: bool,
    lines_out: &mut [f64],
    signals_out: &mut [f64],
) -> Result<Vec<KstParams>, KstError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(KstError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let cols = data.len();
    let rows = combos.len();
    if lines_out.len() != rows * cols || signals_out.len() != rows * cols {
        return Err(KstError::InvalidPeriod {
            period: lines_out.len(),
            data_len: rows * cols,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(KstError::AllValuesNaN)?;

    // warmups per row
    let mut warm_line = vec![0usize; rows];
    let mut warm_sig = vec![0usize; rows];
    for (row, c) in combos.iter().enumerate() {
        let wl = (c.roc_period1.unwrap() + c.sma_period1.unwrap() - 1)
            .max(c.roc_period2.unwrap() + c.sma_period2.unwrap() - 1)
            .max(c.roc_period3.unwrap() + c.sma_period3.unwrap() - 1)
            .max(c.roc_period4.unwrap() + c.sma_period4.unwrap() - 1);
        warm_line[row] = wl;
        warm_sig[row] = wl + c.signal_period.unwrap() - 1;
        // prefix NaNs directly with first offset
        let abs_wl = first + warm_line[row];
        let abs_ws = first + warm_sig[row];
        for v in &mut lines_out[row * cols..row * cols + abs_wl.min(cols)] {
            *v = f64::NAN;
        }
        for v in &mut signals_out[row * cols..row * cols + abs_ws.min(cols)] {
            *v = f64::NAN;
        }
    }

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd = match actual {
        Kernel::ScalarBatch => Kernel::Scalar,
        Kernel::Avx2Batch => Kernel::Scalar, // Use scalar if AVX2 not available
        Kernel::Avx512Batch => Kernel::Scalar, // Use scalar if AVX512 not available
        _ => Kernel::Scalar,
    };

    // --- Row-specific batch optimization: ROC caching across rows with identical (r1..r4)
    use std::collections::HashMap;
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    struct R4(usize, usize, usize, usize);

    // Group rows by ROC tuple
    let mut groups: HashMap<R4, Vec<usize>> = HashMap::new();
    for (idx, prm) in combos.iter().enumerate() {
        groups
            .entry(R4(
                prm.roc_period1.unwrap(),
                prm.roc_period2.unwrap(),
                prm.roc_period3.unwrap(),
                prm.roc_period4.unwrap(),
            ))
            .or_default()
            .push(idx);
    }

    fn compute_roc_streams(
        data: &[f64],
        first: usize,
        r: (usize, usize, usize, usize),
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let len = data.len();
        let (r1, r2, r3, r4) = r;
        let mut v1 = vec![0.0f64; len];
        let mut v2 = vec![0.0f64; len];
        let mut v3 = vec![0.0f64; len];
        let mut v4 = vec![0.0f64; len];
        for i in first..len {
            let x = data[i];
            if i >= first + r1 {
                let p = data[i - r1];
                if x.is_finite() && p.is_finite() && p != 0.0 {
                    v1[i] = ((x / p) - 1.0) * 100.0;
                }
            }
            if i >= first + r2 {
                let p = data[i - r2];
                if x.is_finite() && p.is_finite() && p != 0.0 {
                    v2[i] = ((x / p) - 1.0) * 100.0;
                }
            }
            if i >= first + r3 {
                let p = data[i - r3];
                if x.is_finite() && p.is_finite() && p != 0.0 {
                    v3[i] = ((x / p) - 1.0) * 100.0;
                }
            }
            if i >= first + r4 {
                let p = data[i - r4];
                if x.is_finite() && p.is_finite() && p != 0.0 {
                    v4[i] = ((x / p) - 1.0) * 100.0;
                }
            }
        }
        (v1, v2, v3, v4)
    }

    struct Streams {
        v1: Vec<f64>,
        v2: Vec<f64>,
        v3: Vec<f64>,
        v4: Vec<f64>,
    }

    let mut streams_map: HashMap<R4, Streams> = HashMap::with_capacity(groups.len());
    for (key @ R4(r1, r2, r3, r4), _rows) in groups.iter() {
        let (v1, v2, v3, v4) = compute_roc_streams(data, first, (*r1, *r2, *r3, *r4));
        streams_map.insert(*key, Streams { v1, v2, v3, v4 });
    }

    let do_row = |row: usize, ldst: &mut [f64], sdst: &mut [f64]| {
        let prm = &combos[row];
        let s = (
            prm.sma_period1.unwrap(),
            prm.sma_period2.unwrap(),
            prm.sma_period3.unwrap(),
            prm.sma_period4.unwrap(),
        );
        let r = (
            prm.roc_period1.unwrap(),
            prm.roc_period2.unwrap(),
            prm.roc_period3.unwrap(),
            prm.roc_period4.unwrap(),
        );
        let sig = prm.signal_period.unwrap();
        let wl = (r.0 + s.0 - 1)
            .max(r.1 + s.1 - 1)
            .max(r.2 + s.2 - 1)
            .max(r.3 + s.3 - 1);
        let ws = wl + sig - 1;

        let key = R4(r.0, r.1, r.2, r.3);
        let st = streams_map.get(&key).unwrap();

        let len = ldst.len();
        let (s1, s2, s3, s4) = s;
        let inv1 = 1.0 / (s1 as f64);
        let inv2 = 1.0 / (s2 as f64);
        let inv3 = 1.0 / (s3 as f64);
        let inv4 = 1.0 / (s4 as f64);
        let w2 = inv2 + inv2;
        let w3 = inv3 + inv3 + inv3;
        let w4 = 4.0f64 * inv4;

        let mut b1 = vec![0.0f64; s1];
        let mut b2 = vec![0.0f64; s2];
        let mut b3 = vec![0.0f64; s3];
        let mut b4 = vec![0.0f64; s4];
        let mut i1 = 0usize;
        let mut i2 = 0usize;
        let mut i3 = 0usize;
        let mut i4 = 0usize;
        let mut sum1 = 0.0f64;
        let mut sum2 = 0.0f64;
        let mut sum3 = 0.0f64;
        let mut sum4 = 0.0f64;

        let start_line = first + wl;
        let warm_sig_abs = first + ws;

        let mut sbuf = vec![0.0f64; sig];
        let mut sidx = 0usize;
        let mut ssum = 0.0f64;
        let mut sbuilt = 0usize;

        unsafe {
            let b1p = b1.as_mut_ptr();
            let b2p = b2.as_mut_ptr();
            let b3p = b3.as_mut_ptr();
            let b4p = b4.as_mut_ptr();
            let sbp = sbuf.as_mut_ptr();
            let lptr = ldst.as_mut_ptr();
            let sptr = sdst.as_mut_ptr();
            let v1p = st.v1.as_ptr();
            let v2p = st.v2.as_ptr();
            let v3p = st.v3.as_ptr();
            let v4p = st.v4.as_ptr();

            #[inline(always)]
            unsafe fn ring_update(
                buf: *mut f64,
                idx: &mut usize,
                cap: usize,
                sum: &mut f64,
                v: f64,
            ) {
                let old = *buf.add(*idx);
                *sum = (*sum) + (v - old);
                *buf.add(*idx) = v;
                *idx += 1;
                if *idx == cap {
                    *idx = 0;
                }
            }

            for i in first..len {
                let v1 = *v1p.add(i);
                let v2 = *v2p.add(i);
                let v3 = *v3p.add(i);
                let v4 = *v4p.add(i);

                ring_update(b1p, &mut i1, s1, &mut sum1, v1);
                ring_update(b2p, &mut i2, s2, &mut sum2, v2);
                ring_update(b3p, &mut i3, s3, &mut sum3, v3);
                ring_update(b4p, &mut i4, s4, &mut sum4, v4);

                if i < start_line {
                    continue;
                }

                let kst = sum1.mul_add(inv1, sum2.mul_add(w2, sum3.mul_add(w3, sum4 * w4)));
                *lptr.add(i) = kst;

                if sbuilt < sig {
                    let old = *sbp.add(sidx);
                    ssum += kst - old;
                    *sbp.add(sidx) = kst;
                    sidx += 1;
                    if sidx == sig {
                        sidx = 0;
                    }
                    sbuilt += 1;
                    if i >= warm_sig_abs {
                        *sptr.add(i) = ssum / (sig as f64);
                    }
                } else {
                    let old = *sbp.add(sidx);
                    ssum += kst - old;
                    *sbp.add(sidx) = kst;
                    sidx += 1;
                    if sidx == sig {
                        sidx = 0;
                    }
                    *sptr.add(i) = ssum / (sig as f64);
                }
            }
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        lines_out
            .par_chunks_mut(cols)
            .zip(signals_out.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(r, (l, s))| do_row(r, l, s));
        #[cfg(target_arch = "wasm32")]
        for (r, (l, s)) in lines_out
            .chunks_mut(cols)
            .zip(signals_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(r, l, s);
        }
    } else {
        for (r, (l, s)) in lines_out
            .chunks_mut(cols)
            .zip(signals_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(r, l, s);
        }
    }

    Ok(combos)
}

// Row helper functions removed - no longer needed with direct compute_into approach

/// Streaming KST – O(1) per tick.
/// Decision: uses cached reciprocals and branchy wrap; reduces divisions/modulo and uses an FMA chain.
#[derive(Debug, Clone)]
pub struct KstStream {
    // periods
    s: (usize, usize, usize, usize),
    r: (usize, usize, usize, usize),
    sig: usize,

    // SMA(ROC) rings
    b1: Vec<f64>,
    b2: Vec<f64>,
    b3: Vec<f64>,
    b4: Vec<f64>,
    i1: usize,
    i2: usize,
    i3: usize,
    i4: usize,
    sum1: f64,
    sum2: f64,
    sum3: f64,
    sum4: f64,

    // precomputed weights for the KST line (FMA chain)
    inv1: f64, // 1/s1
    w2: f64,   // 2/s2
    w3: f64,   // 3/s3
    w4: f64,   // 4/s4

    // signal SMA ring
    sig_buf: Vec<f64>,
    sig_idx: usize,
    sig_sum: f64,

    // price + reciprocal rings
    price_ring: Vec<f64>,
    recip_ring: Vec<f64>,
    head: usize, // next write position

    // warmup tracking
    t: usize,         // ticks seen so far
    warm_line: usize, // max(r_i + s_i - 1)
    warm_sig: usize,  // warm_line + sig - 1

    // optional for external inspection
    last_line: f64,
}
impl KstStream {
    #[inline]
    pub fn try_new(params: KstParams) -> Result<Self, KstError> {
        let s1 = params.sma_period1.unwrap_or(10);
        let s2 = params.sma_period2.unwrap_or(10);
        let s3 = params.sma_period3.unwrap_or(10);
        let s4 = params.sma_period4.unwrap_or(15);

        let r1 = params.roc_period1.unwrap_or(10);
        let r2 = params.roc_period2.unwrap_or(15);
        let r3 = params.roc_period3.unwrap_or(20);
        let r4 = params.roc_period4.unwrap_or(30);

        let sig = params.signal_period.unwrap_or(9);

        for &p in [s1, s2, s3, s4, r1, r2, r3, r4, sig].iter() {
            if p == 0 {
                return Err(KstError::InvalidPeriod {
                    period: p,
                    data_len: 0,
                });
            }
        }

        let warm_line = (r1 + s1 - 1)
            .max(r2 + s2 - 1)
            .max(r3 + s3 - 1)
            .max(r4 + s4 - 1);
        let warm_sig = warm_line + sig - 1;

        let max_roc = r1.max(r2).max(r3).max(r4);
        let price_cap = max_roc + 1; // head is next write

        Ok(Self {
            s: (s1, s2, s3, s4),
            r: (r1, r2, r3, r4),
            sig,

            b1: vec![0.0; s1],
            b2: vec![0.0; s2],
            b3: vec![0.0; s3],
            b4: vec![0.0; s4],
            i1: 0,
            i2: 0,
            i3: 0,
            i4: 0,
            sum1: 0.0,
            sum2: 0.0,
            sum3: 0.0,
            sum4: 0.0,

            inv1: 1.0 / (s1 as f64),
            w2: (2.0f64) / (s2 as f64),
            w3: (3.0f64) / (s3 as f64),
            w4: (4.0f64) / (s4 as f64),

            sig_buf: vec![0.0; sig],
            sig_idx: 0,
            sig_sum: 0.0,

            price_ring: vec![f64::NAN; price_cap],
            recip_ring: vec![f64::NAN; price_cap],
            head: 0,

            t: 0,
            warm_line,
            warm_sig,

            last_line: f64::NAN,
        })
    }

    /// O(1) tick update. Returns (line, signal) or None until fully warmed.
    #[inline(always)]
    pub fn update(&mut self, price: f64) -> Option<(f64, f64)> {
        // Write price and its reciprocal (1 division per tick)
        self.price_ring[self.head] = price;
        self.recip_ring[self.head] = if price.is_finite() && price != 0.0 {
            1.0 / price
        } else {
            f64::NAN
        };

        // Advance head without modulo
        Self::wrap_inc(&mut self.head, self.price_ring.len());

        let cap = self.price_ring.len();
        let (s1, s2, s3, s4) = self.s;
        let (r1, r2, r3, r4) = self.r;

        // Compute 4 ROC terms using cached reciprocals
        let mut v1 = 0.0;
        if self.t >= r1 {
            let idx = Self::back_from_next(self.head, cap, r1 + 1);
            let pinv = self.recip_ring[idx];
            if price.is_finite() && pinv.is_finite() {
                v1 = (price * pinv - 1.0) * 100.0;
            }
        }
        let mut v2 = 0.0;
        if self.t >= r2 {
            let idx = Self::back_from_next(self.head, cap, r2 + 1);
            let pinv = self.recip_ring[idx];
            if price.is_finite() && pinv.is_finite() {
                v2 = (price * pinv - 1.0) * 100.0;
            }
        }
        let mut v3 = 0.0;
        if self.t >= r3 {
            let idx = Self::back_from_next(self.head, cap, r3 + 1);
            let pinv = self.recip_ring[idx];
            if price.is_finite() && pinv.is_finite() {
                v3 = (price * pinv - 1.0) * 100.0;
            }
        }
        let mut v4 = 0.0;
        if self.t >= r4 {
            let idx = Self::back_from_next(self.head, cap, r4 + 1);
            let pinv = self.recip_ring[idx];
            if price.is_finite() && pinv.is_finite() {
                v4 = (price * pinv - 1.0) * 100.0;
            }
        }

        // Update four SMA rings (no modulo in hot path)
        if self.t >= r1 {
            self.sum1 -= self.b1[self.i1];
            self.b1[self.i1] = v1;
            self.sum1 += v1;
            Self::wrap_inc(&mut self.i1, s1);
        }
        if self.t >= r2 {
            self.sum2 -= self.b2[self.i2];
            self.b2[self.i2] = v2;
            self.sum2 += v2;
            Self::wrap_inc(&mut self.i2, s2);
        }
        if self.t >= r3 {
            self.sum3 -= self.b3[self.i3];
            self.b3[self.i3] = v3;
            self.sum3 += v3;
            Self::wrap_inc(&mut self.i3, s3);
        }
        if self.t >= r4 {
            self.sum4 -= self.b4[self.i4];
            self.b4[self.i4] = v4;
            self.sum4 += v4;
            Self::wrap_inc(&mut self.i4, s4);
        }

        // Tick complete
        self.t += 1;

        // Not warmed for KST line yet
        if self.t <= self.warm_line {
            return None;
        }

        // KST line via FMA chain: sum1*(1/s1) + sum2*(2/s2) + sum3*(3/s3) + sum4*(4/s4)
        let line = self.sum1.mul_add(
            self.inv1,
            self.sum2
                .mul_add(self.w2, self.sum3.mul_add(self.w3, self.sum4 * self.w4)),
        );

        self.last_line = line;

        // Signal SMA ring update; return finite only after full warmup
        let old = self.sig_buf[self.sig_idx];
        self.sig_sum += line - old;
        self.sig_buf[self.sig_idx] = line;
        Self::wrap_inc(&mut self.sig_idx, self.sig);

        let signal = if self.t >= self.warm_sig {
            self.sig_sum / (self.sig as f64)
        } else {
            f64::NAN
        };

        Some((line, signal))
    }

    // ========= helpers =========

    #[inline(always)]
    fn wrap_inc(idx: &mut usize, cap: usize) {
        *idx += 1;
        if *idx == cap {
            *idx = 0;
        }
    }

    /// Compute (next - k) mod cap with a single branch, where k ∈ [0, cap]
    #[inline(always)]
    fn back_from_next(next: usize, cap: usize, k: usize) -> usize {
        debug_assert!(k <= cap);
        let mut idx = next;
        if idx < k {
            idx += cap;
        }
        idx - k
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_kst_default_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KstInput::with_default_candles(&candles);
        let result = kst_with_kernel(&input, kernel)?;
        assert_eq!(result.line.len(), candles.close.len());
        assert_eq!(result.signal.len(), candles.close.len());
        Ok(())
    }

    fn check_kst_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = KstInput::with_default_candles(&candles);
        let result = kst_with_kernel(&input, kernel)?;
        let expected_last_five_line = [
            -47.38570195278667,
            -44.42926180347176,
            -42.185693049429034,
            -40.10697793942024,
            -40.17466795905724,
        ];
        let expected_last_five_signal = [
            -52.66743277411538,
            -51.559775662725556,
            -50.113844191238954,
            -48.58923772989874,
            -47.01112630514571,
        ];
        let l = result.line.len();
        let s = result.signal.len();
        for (i, &v) in result.line[l - 5..].iter().enumerate() {
            assert!(
                (v - expected_last_five_line[i]).abs() < 1e-1,
                "KST line mismatch {}: {} vs {}",
                i,
                v,
                expected_last_five_line[i]
            );
        }
        for (i, &v) in result.signal[s - 5..].iter().enumerate() {
            assert!(
                (v - expected_last_five_signal[i]).abs() < 1e-1,
                "KST signal mismatch {}: {} vs {}",
                i,
                v,
                expected_last_five_signal[i]
            );
        }
        Ok(())
    }

    fn check_kst_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let input = KstInput::from_slice(&nan_data, KstParams::default());
        let result = kst_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] Should error with all NaN", test_name);
        Ok(())
    }

    macro_rules! generate_all_kst_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() { let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar); }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2); }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() { let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512); }
                )*
            }
        }
    }

    #[cfg(debug_assertions)]
    fn check_kst_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            KstParams::default(),
            // Minimum viable periods
            KstParams {
                sma_period1: Some(2),
                sma_period2: Some(2),
                sma_period3: Some(2),
                sma_period4: Some(2),
                roc_period1: Some(2),
                roc_period2: Some(2),
                roc_period3: Some(2),
                roc_period4: Some(2),
                signal_period: Some(2),
            },
            // Small periods
            KstParams {
                sma_period1: Some(5),
                sma_period2: Some(5),
                sma_period3: Some(5),
                sma_period4: Some(7),
                roc_period1: Some(5),
                roc_period2: Some(7),
                roc_period3: Some(10),
                roc_period4: Some(15),
                signal_period: Some(5),
            },
            // Mixed small/medium periods
            KstParams {
                sma_period1: Some(7),
                sma_period2: Some(10),
                sma_period3: Some(12),
                sma_period4: Some(15),
                roc_period1: Some(8),
                roc_period2: Some(12),
                roc_period3: Some(16),
                roc_period4: Some(20),
                signal_period: Some(7),
            },
            // Default periods (10, 10, 10, 15, 10, 15, 20, 30, 9)
            KstParams {
                sma_period1: Some(10),
                sma_period2: Some(10),
                sma_period3: Some(10),
                sma_period4: Some(15),
                roc_period1: Some(10),
                roc_period2: Some(15),
                roc_period3: Some(20),
                roc_period4: Some(30),
                signal_period: Some(9),
            },
            // Large periods
            KstParams {
                sma_period1: Some(20),
                sma_period2: Some(25),
                sma_period3: Some(30),
                sma_period4: Some(35),
                roc_period1: Some(25),
                roc_period2: Some(35),
                roc_period3: Some(45),
                roc_period4: Some(60),
                signal_period: Some(15),
            },
            // Very large periods
            KstParams {
                sma_period1: Some(30),
                sma_period2: Some(40),
                sma_period3: Some(50),
                sma_period4: Some(60),
                roc_period1: Some(40),
                roc_period2: Some(60),
                roc_period3: Some(80),
                roc_period4: Some(100),
                signal_period: Some(21),
            },
            // Asymmetric periods
            KstParams {
                sma_period1: Some(5),
                sma_period2: Some(10),
                sma_period3: Some(20),
                sma_period4: Some(50),
                roc_period1: Some(7),
                roc_period2: Some(14),
                roc_period3: Some(28),
                roc_period4: Some(56),
                signal_period: Some(12),
            },
            // Edge case: very small signal period
            KstParams {
                sma_period1: Some(10),
                sma_period2: Some(10),
                sma_period3: Some(10),
                sma_period4: Some(15),
                roc_period1: Some(10),
                roc_period2: Some(15),
                roc_period3: Some(20),
                roc_period4: Some(30),
                signal_period: Some(2),
            },
            // Absolute minimum (all periods = 1)
            KstParams {
                sma_period1: Some(1),
                sma_period2: Some(1),
                sma_period3: Some(1),
                sma_period4: Some(1),
                roc_period1: Some(1),
                roc_period2: Some(1),
                roc_period3: Some(1),
                roc_period4: Some(1),
                signal_period: Some(1),
            },
            // Maximum reasonable periods
            KstParams {
                sma_period1: Some(100),
                sma_period2: Some(120),
                sma_period3: Some(140),
                sma_period4: Some(160),
                roc_period1: Some(100),
                roc_period2: Some(150),
                roc_period3: Some(200),
                roc_period4: Some(250),
                signal_period: Some(50),
            },
            // Common real-world configuration
            KstParams {
                sma_period1: Some(10),
                sma_period2: Some(15),
                sma_period3: Some(20),
                sma_period4: Some(30),
                roc_period1: Some(10),
                roc_period2: Some(15),
                roc_period3: Some(20),
                roc_period4: Some(30),
                signal_period: Some(10),
            },
            // Another asymmetric case
            KstParams {
                sma_period1: Some(3),
                sma_period2: Some(6),
                sma_period3: Some(12),
                sma_period4: Some(24),
                roc_period1: Some(5),
                roc_period2: Some(10),
                roc_period3: Some(20),
                roc_period4: Some(40),
                signal_period: Some(8),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = KstInput::from_candles(&candles, "close", params.clone());
            let output = kst_with_kernel(&input, kernel)?;

            // Check line values
            for (i, &val) in output.line.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in KST line with params: sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), \
						 signal_period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.sma_period1.unwrap_or(10),
                        params.sma_period2.unwrap_or(10),
                        params.sma_period3.unwrap_or(10),
                        params.sma_period4.unwrap_or(15),
                        params.roc_period1.unwrap_or(10),
                        params.roc_period2.unwrap_or(15),
                        params.roc_period3.unwrap_or(20),
                        params.roc_period4.unwrap_or(30),
                        params.signal_period.unwrap_or(9),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in KST line with params: sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), \
						 signal_period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.sma_period1.unwrap_or(10),
                        params.sma_period2.unwrap_or(10),
                        params.sma_period3.unwrap_or(10),
                        params.sma_period4.unwrap_or(15),
                        params.roc_period1.unwrap_or(10),
                        params.roc_period2.unwrap_or(15),
                        params.roc_period3.unwrap_or(20),
                        params.roc_period4.unwrap_or(30),
                        params.signal_period.unwrap_or(9),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in KST line with params: sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), \
						 signal_period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.sma_period1.unwrap_or(10),
                        params.sma_period2.unwrap_or(10),
                        params.sma_period3.unwrap_or(10),
                        params.sma_period4.unwrap_or(15),
                        params.roc_period1.unwrap_or(10),
                        params.roc_period2.unwrap_or(15),
                        params.roc_period3.unwrap_or(20),
                        params.roc_period4.unwrap_or(30),
                        params.signal_period.unwrap_or(9),
                        param_idx
                    );
                }
            }

            // Check signal values
            for (i, &val) in output.signal.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in KST signal with params: sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), \
						 signal_period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.sma_period1.unwrap_or(10),
                        params.sma_period2.unwrap_or(10),
                        params.sma_period3.unwrap_or(10),
                        params.sma_period4.unwrap_or(15),
                        params.roc_period1.unwrap_or(10),
                        params.roc_period2.unwrap_or(15),
                        params.roc_period3.unwrap_or(20),
                        params.roc_period4.unwrap_or(30),
                        params.signal_period.unwrap_or(9),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in KST signal with params: sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), \
						 signal_period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.sma_period1.unwrap_or(10),
                        params.sma_period2.unwrap_or(10),
                        params.sma_period3.unwrap_or(10),
                        params.sma_period4.unwrap_or(15),
                        params.roc_period1.unwrap_or(10),
                        params.roc_period2.unwrap_or(15),
                        params.roc_period3.unwrap_or(20),
                        params.roc_period4.unwrap_or(30),
                        params.signal_period.unwrap_or(9),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in KST signal with params: sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), \
						 signal_period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.sma_period1.unwrap_or(10),
                        params.sma_period2.unwrap_or(10),
                        params.sma_period3.unwrap_or(10),
                        params.sma_period4.unwrap_or(15),
                        params.roc_period1.unwrap_or(10),
                        params.roc_period2.unwrap_or(15),
                        params.roc_period3.unwrap_or(20),
                        params.roc_period4.unwrap_or(30),
                        params.signal_period.unwrap_or(9),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_kst_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_kst_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate realistic parameter combinations for KST
        let strat = (
            // SMA periods (reasonable ranges)
            (3usize..=20), // sma_period1
            (3usize..=20), // sma_period2
            (3usize..=20), // sma_period3
            (5usize..=25), // sma_period4
            // ROC periods (reasonable ranges)
            (5usize..=15),  // roc_period1
            (10usize..=20), // roc_period2
            (15usize..=25), // roc_period3
            (20usize..=35), // roc_period4
            // Signal period
            (3usize..=15), // signal_period
            // Data scenario selector
            (0usize..=3), // scenario type
        )
            .prop_flat_map(|(s1, s2, s3, s4, r1, r2, r3, r4, sig, scenario)| {
                // Calculate minimum data length needed
                let warmup1 = r1 + s1 - 1;
                let warmup2 = r2 + s2 - 1;
                let warmup3 = r3 + s3 - 1;
                let warmup4 = r4 + s4 - 1;
                let warmup = warmup1.max(warmup2).max(warmup3).max(warmup4);
                let min_data_len = warmup + sig + 20; // Extra buffer for testing

                // Different data generation strategies based on scenario
                let data_strategy = match scenario {
                    0 => {
                        // Normal price range (most common)
                        prop::collection::vec(
                            (10.0f64..10000.0f64).prop_filter("finite", |x| x.is_finite()),
                            min_data_len..400,
                        )
                        .boxed()
                    }
                    1 => {
                        // Small values (penny stocks)
                        prop::collection::vec(
                            (0.01f64..5.0f64).prop_filter("finite", |x| x.is_finite()),
                            min_data_len..400,
                        )
                        .boxed()
                    }
                    2 => {
                        // Data with plateaus (same value for extended periods)
                        prop::collection::vec((10.0f64..1000.0f64), min_data_len..400)
                            .prop_map(|mut v| {
                                // Create plateaus by repeating some values
                                for i in 0..v.len() / 4 {
                                    let plateau_start = i * 4;
                                    let plateau_end = (plateau_start + 3).min(v.len() - 1);
                                    let plateau_value = v[plateau_start];
                                    for j in plateau_start..=plateau_end {
                                        v[j] = plateau_value;
                                    }
                                }
                                v
                            })
                            .boxed()
                    }
                    _ => {
                        // Data with large jumps
                        prop::collection::vec((10.0f64..1000.0f64), min_data_len..400)
                            .prop_map(|mut v| {
                                // Introduce some large jumps
                                for i in (5..v.len()).step_by(20) {
                                    v[i] = v[i - 1] * (1.5 + (i % 3) as f64 * 0.5);
                                    // 1.5x to 2.5x jump
                                }
                                v
                            })
                            .boxed()
                    }
                };

                (
                    data_strategy,
                    Just(s1),
                    Just(s2),
                    Just(s3),
                    Just(s4),
                    Just(r1),
                    Just(r2),
                    Just(r3),
                    Just(r4),
                    Just(sig),
                )
            });

        proptest::test_runner::TestRunner::default()
			.run(&strat, |(data, s1, s2, s3, s4, r1, r2, r3, r4, sig)| {
				let params = KstParams {
					sma_period1: Some(s1),
					sma_period2: Some(s2),
					sma_period3: Some(s3),
					sma_period4: Some(s4),
					roc_period1: Some(r1),
					roc_period2: Some(r2),
					roc_period3: Some(r3),
					roc_period4: Some(r4),
					signal_period: Some(sig),
				};
				let input = KstInput::from_slice(&data, params);

				// Calculate warmup period
				let warmup1 = r1 + s1 - 1;
				let warmup2 = r2 + s2 - 1;
				let warmup3 = r3 + s3 - 1;
				let warmup4 = r4 + s4 - 1;
				let warmup = warmup1.max(warmup2).max(warmup3).max(warmup4);
				let signal_warmup = warmup + sig - 1;

				// Get outputs from both kernels
				let KstOutput { line, signal } = kst_with_kernel(&input, kernel).unwrap();
				let KstOutput { line: ref_line, signal: ref_signal } = kst_with_kernel(&input, Kernel::Scalar).unwrap();

				// Property 1: Warmup period validation
				for i in 0..warmup.min(data.len()) {
					prop_assert!(
						line[i].is_nan(),
						"KST line should be NaN during warmup at index {i}"
					);
				}
				for i in 0..signal_warmup.min(data.len()) {
					prop_assert!(
						signal[i].is_nan(),
						"Signal should be NaN during warmup at index {i}"
					);
				}

				// Property 2: Kernel consistency - all kernels should produce identical results
				for i in warmup..data.len() {
					let y = line[i];
					let r = ref_line[i];

					if !y.is_finite() || !r.is_finite() {
						prop_assert!(y.to_bits() == r.to_bits(), "NaN/Inf mismatch at idx {i}: {y} vs {r}");
						continue;
					}

					let ulp_diff = y.to_bits().abs_diff(r.to_bits());
					prop_assert!(
						(y - r).abs() <= 1e-9 || ulp_diff <= 4,
						"KST line kernel mismatch idx {i}: {y} vs {r} (ULP={ulp_diff})"
					);
				}

				for i in signal_warmup..data.len() {
					let y = signal[i];
					let r = ref_signal[i];

					if !y.is_finite() || !r.is_finite() {
						prop_assert!(y.to_bits() == r.to_bits(), "Signal NaN/Inf mismatch at idx {i}: {y} vs {r}");
						continue;
					}

					let ulp_diff = y.to_bits().abs_diff(r.to_bits());
					prop_assert!(
						(y - r).abs() <= 1e-9 || ulp_diff <= 4,
						"Signal kernel mismatch idx {i}: {y} vs {r} (ULP={ulp_diff})"
					);
				}

				// Property 3: Constant data - KST should converge to 0
				if data.windows(2).all(|w| (w[0] - w[1]).abs() <= f64::EPSILON) {
					for i in warmup..data.len() {
						prop_assert!(
							line[i].abs() <= 1e-9,
							"KST should be ~0 for constant data at idx {i}: {}",
							line[i]
						);
					}
				}

				// Property 4: Monotonic increasing data - KST should be positive
				if data.windows(2).all(|w| w[1] > w[0] + f64::EPSILON) {
					let check_start = warmup + 10; // Allow some stabilization
					for i in check_start..data.len() {
						if line[i].is_finite() {
							prop_assert!(
								line[i] > -1e-6,
								"KST should be positive for increasing data at idx {i}: {}",
								line[i]
							);
						}
					}
				}

				// Property 5: Monotonic decreasing data - KST should be negative
				if data.windows(2).all(|w| w[0] > w[1] + f64::EPSILON) {
					let check_start = warmup + 10; // Allow some stabilization
					for i in check_start..data.len() {
						if line[i].is_finite() {
							prop_assert!(
								line[i] < 1e-6,
								"KST should be negative for decreasing data at idx {i}: {}",
								line[i]
							);
						}
					}
				}

				// Property 6: Signal smoothing - signal should be smoother than KST line
				// Check variance of differences for stable portions of data
				if signal_warmup + sig * 2 < data.len() && data.len() > signal_warmup + 20 {
					// Skip initial volatile period after warmup
					let stable_start = signal_warmup + sig;
					let line_diffs: Vec<f64> = line[stable_start..data.len()-1]
						.windows(2)
						.map(|w| (w[1] - w[0]).abs())
						.filter(|x| x.is_finite())
						.collect();

					let signal_diffs: Vec<f64> = signal[stable_start..data.len()-1]
						.windows(2)
						.map(|w| (w[1] - w[0]).abs())
						.filter(|x| x.is_finite())
						.collect();

					if line_diffs.len() > 15 && signal_diffs.len() > 15 {
						// Check mean variance for smoothness
						let line_variance: f64 = line_diffs.iter().sum::<f64>() / line_diffs.len() as f64;
						let signal_variance: f64 = signal_diffs.iter().sum::<f64>() / signal_diffs.len() as f64;

						// Signal should generally be smoother than the line
						// But with edge case data (plateaus, jumps), allow more tolerance
						if line_variance > 1e-10 && signal_variance > 1e-10 {
							// Allow up to 2.5x variance for edge cases
							prop_assert!(
								signal_variance <= line_variance * 2.5,
								"Signal variance too high relative to KST line: signal_var={signal_variance}, line_var={line_variance}"
							);
						}
					}
				}

				// Property 7: Reasonable bounds - KST values should be within reasonable range
				// KST is weighted sum of ROCs: 1*ROC1 + 2*ROC2 + 3*ROC3 + 4*ROC4
				// With extreme price movements, this can reach large values
				// Calculate max possible ROC based on data range
				let min_price = data.iter().cloned().fold(f64::INFINITY, f64::min);
				let max_price = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
				let max_roc = if min_price > 0.0 {
					((max_price / min_price) - 1.0) * 100.0
				} else {
					10000.0 // Default upper bound
				};
				// KST max is approximately 10 * max_roc (sum of weights 1+2+3+4)
				let kst_bound = max_roc * 10.0;

				for i in warmup..data.len() {
					if line[i].is_finite() {
						prop_assert!(
							line[i] >= -kst_bound && line[i] <= kst_bound,
							"KST line out of reasonable bounds at idx {i}: {} (bound: ±{})",
							line[i], kst_bound
						);
					}
				}
				for i in signal_warmup..data.len() {
					if signal[i].is_finite() {
						prop_assert!(
							signal[i] >= -kst_bound && signal[i] <= kst_bound,
							"Signal out of reasonable bounds at idx {i}: {} (bound: ±{})",
							signal[i], kst_bound
						);
					}
				}

				// Property 8: No infinite values in valid output range
				for i in warmup..data.len() {
					prop_assert!(
						line[i].is_nan() || line[i].is_finite(),
						"KST line has infinite value at idx {i}: {}",
						line[i]
					);
				}
				for i in signal_warmup..data.len() {
					prop_assert!(
						signal[i].is_nan() || signal[i].is_finite(),
						"Signal has infinite value at idx {i}: {}",
						signal[i]
					);
				}

				// Property 9: Signal follows KST line trend
				// The signal is a smoothed version of the KST line, so it should follow the general trend
				if signal_warmup + sig + 5 < data.len() {
					// Check individual points with appropriate tolerance
					for i in (signal_warmup + sig)..data.len() {
						// Calculate average KST line over the signal period
						let line_window = &line[i.saturating_sub(sig-1)..=i.min(data.len()-1)];
						let valid_values: Vec<f64> = line_window.iter()
							.filter(|x| x.is_finite())
							.cloned()
							.collect();

						if !valid_values.is_empty() && signal[i].is_finite() {
							let line_avg = valid_values.iter().sum::<f64>() / valid_values.len() as f64;

							// Tighter tolerance: 0.5% for large values, absolute tolerance for small values
							let tolerance = if line_avg.abs() > 100.0 {
								0.005  // 0.5% for large values
							} else if line_avg.abs() > 10.0 {
								0.007  // 0.7% for medium values
							} else {
								0.01   // 1% for small values (original)
							};

							prop_assert!(
								(signal[i] - line_avg).abs() <= 1e-6 ||
								(signal[i] - line_avg).abs() / line_avg.abs().max(1.0) <= tolerance,
								"Signal deviates from KST trend at idx {i}: signal={}, line_avg={}, tolerance={}%",
								signal[i], line_avg, tolerance * 100.0
							);
						}
					}
				}

				Ok(())
			})
			.unwrap();

        Ok(())
    }

    #[cfg(feature = "proptest")]
    generate_all_kst_tests!(check_kst_property);

    generate_all_kst_tests!(
        check_kst_default_params,
        check_kst_accuracy,
        check_kst_nan_handling,
        check_kst_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = KstBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = KstParams::default();
        let row = output.lines_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        Ok(())
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      { let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        { let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch); }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      { let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch); }
                #[test] fn [<$fn_name _auto_detect>]() { let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto); }
            }
        };
    }
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations (kept minimal to avoid OOM)
        let test_configs = vec![
            // Small periods - single combination
            (
                3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 3, 3, 0, 4, 4, 0, 5, 5, 0, 6, 6, 0, 3, 3, 0,
            ),
            // Medium periods - 2x2x2x2 combinations
            (
                5, 10, 5, 5, 10, 5, 5, 10, 5, 8, 13, 5, 5, 10, 5, 8, 13, 5, 10, 15, 5, 15, 20, 5,
                5, 7, 2,
            ),
            // Default values
            (
                10, 10, 0, 10, 10, 0, 10, 10, 0, 15, 15, 0, 10, 10, 0, 15, 15, 0, 20, 20, 0, 30,
                30, 0, 9, 9, 0,
            ),
            // Dense small range - limited
            (
                2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 4, 1, 4, 5, 1, 5, 6, 1, 2, 3, 1,
            ),
        ];

        for (
            cfg_idx,
            &(
                s1_start,
                s1_end,
                s1_step,
                s2_start,
                s2_end,
                s2_step,
                s3_start,
                s3_end,
                s3_step,
                s4_start,
                s4_end,
                s4_step,
                r1_start,
                r1_end,
                r1_step,
                r2_start,
                r2_end,
                r2_step,
                r3_start,
                r3_end,
                r3_step,
                r4_start,
                r4_end,
                r4_step,
                sig_start,
                sig_end,
                sig_step,
            ),
        ) in test_configs.iter().enumerate()
        {
            let output = KstBatchBuilder::new()
                .kernel(kernel)
                .sma_period1_range(s1_start, s1_end, s1_step)
                .sma_period2_range(s2_start, s2_end, s2_step)
                .sma_period3_range(s3_start, s3_end, s3_step)
                .sma_period4_range(s4_start, s4_end, s4_step)
                .roc_period1_range(r1_start, r1_end, r1_step)
                .roc_period2_range(r2_start, r2_end, r2_step)
                .roc_period3_range(r3_start, r3_end, r3_step)
                .roc_period4_range(r4_start, r4_end, r4_step)
                .signal_period_range(sig_start, sig_end, sig_step)
                .apply_candles(&c, "close")?;

            // Check line values
            for (idx, &val) in output.lines.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                // Check all three poison patterns with detailed context
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in KST lines with params: \
						 sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), signal_period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.sma_period1.unwrap_or(10),
                        combo.sma_period2.unwrap_or(10),
                        combo.sma_period3.unwrap_or(10),
                        combo.sma_period4.unwrap_or(15),
                        combo.roc_period1.unwrap_or(10),
                        combo.roc_period2.unwrap_or(15),
                        combo.roc_period3.unwrap_or(20),
                        combo.roc_period4.unwrap_or(30),
                        combo.signal_period.unwrap_or(9)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in KST lines with params: \
						 sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), signal_period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.sma_period1.unwrap_or(10),
                        combo.sma_period2.unwrap_or(10),
                        combo.sma_period3.unwrap_or(10),
                        combo.sma_period4.unwrap_or(15),
                        combo.roc_period1.unwrap_or(10),
                        combo.roc_period2.unwrap_or(15),
                        combo.roc_period3.unwrap_or(20),
                        combo.roc_period4.unwrap_or(30),
                        combo.signal_period.unwrap_or(9)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in KST lines with params: \
						 sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), signal_period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.sma_period1.unwrap_or(10),
                        combo.sma_period2.unwrap_or(10),
                        combo.sma_period3.unwrap_or(10),
                        combo.sma_period4.unwrap_or(15),
                        combo.roc_period1.unwrap_or(10),
                        combo.roc_period2.unwrap_or(15),
                        combo.roc_period3.unwrap_or(20),
                        combo.roc_period4.unwrap_or(30),
                        combo.signal_period.unwrap_or(9)
                    );
                }
            }

            // Check signal values
            for (idx, &val) in output.signals.iter().enumerate() {
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
						 at row {} col {} (flat index {}) in KST signals with params: \
						 sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), signal_period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.sma_period1.unwrap_or(10),
                        combo.sma_period2.unwrap_or(10),
                        combo.sma_period3.unwrap_or(10),
                        combo.sma_period4.unwrap_or(15),
                        combo.roc_period1.unwrap_or(10),
                        combo.roc_period2.unwrap_or(15),
                        combo.roc_period3.unwrap_or(20),
                        combo.roc_period4.unwrap_or(30),
                        combo.signal_period.unwrap_or(9)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in KST signals with params: \
						 sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), signal_period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.sma_period1.unwrap_or(10),
                        combo.sma_period2.unwrap_or(10),
                        combo.sma_period3.unwrap_or(10),
                        combo.sma_period4.unwrap_or(15),
                        combo.roc_period1.unwrap_or(10),
                        combo.roc_period2.unwrap_or(15),
                        combo.roc_period3.unwrap_or(20),
                        combo.roc_period4.unwrap_or(30),
                        combo.signal_period.unwrap_or(9)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) in KST signals with params: \
						 sma_periods=({},{},{},{}), roc_periods=({},{},{},{}), signal_period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.sma_period1.unwrap_or(10),
                        combo.sma_period2.unwrap_or(10),
                        combo.sma_period3.unwrap_or(10),
                        combo.sma_period4.unwrap_or(15),
                        combo.roc_period1.unwrap_or(10),
                        combo.roc_period2.unwrap_or(15),
                        combo.roc_period3.unwrap_or(20),
                        combo.roc_period4.unwrap_or(30),
                        combo.signal_period.unwrap_or(9)
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

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn check_empty_input_error() {
        // Test that empty input returns EmptyInputData, not AllValuesNaN
        let empty_data: Vec<f64> = vec![];
        let params = KstParams::default();
        let input = KstInput::from_slice(&empty_data, params);

        match kst(&input) {
            Err(KstError::EmptyInputData) => {} // Expected
            Err(e) => panic!("Expected EmptyInputData, got: {:?}", e),
            Ok(_) => panic!("Empty input should have failed"),
        }

        // Test that all NaN still returns AllValuesNaN
        let nan_data = vec![f64::NAN; 10];
        let input2 = KstInput::from_slice(&nan_data, params);

        match kst(&input2) {
            Err(KstError::AllValuesNaN) => {} // Expected
            Err(e) => panic!("Expected AllValuesNaN, got: {:?}", e),
            Ok(_) => panic!("All NaN should have failed"),
        }
    }
}

// =============================================================================
// PYTHON BINDINGS
// =============================================================================

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "python")]
#[pyfunction(name = "kst")]
#[pyo3(signature=(data,
    sma_period1=None, sma_period2=None, sma_period3=None, sma_period4=None,
    roc_period1=None, roc_period2=None, roc_period3=None, roc_period4=None,
    signal_period=None, kernel=None))]
pub fn kst_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    sma_period1: Option<usize>,
    sma_period2: Option<usize>,
    sma_period3: Option<usize>,
    sma_period4: Option<usize>,
    roc_period1: Option<usize>,
    roc_period2: Option<usize>,
    roc_period3: Option<usize>,
    roc_period4: Option<usize>,
    signal_period: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let slice = data.as_slice()?;
    let prm = KstParams {
        sma_period1: Some(sma_period1.unwrap_or(10)),
        sma_period2: Some(sma_period2.unwrap_or(10)),
        sma_period3: Some(sma_period3.unwrap_or(10)),
        sma_period4: Some(sma_period4.unwrap_or(15)),
        roc_period1: Some(roc_period1.unwrap_or(10)),
        roc_period2: Some(roc_period2.unwrap_or(15)),
        roc_period3: Some(roc_period3.unwrap_or(20)),
        roc_period4: Some(roc_period4.unwrap_or(30)),
        signal_period: Some(signal_period.unwrap_or(9)),
    };
    let input = KstInput::from_slice(slice, prm);
    let kern = validate_kernel(kernel, false)?;
    let (line, signal) = py
        .allow_threads(|| kst_with_kernel(&input, kern).map(|o| (o.line, o.signal)))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((line.into_pyarray(py), signal.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "KstStream")]
pub struct KstStreamPy {
    stream: KstStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl KstStreamPy {
    #[new]
    fn new(
        sma_period1: Option<usize>,
        sma_period2: Option<usize>,
        sma_period3: Option<usize>,
        sma_period4: Option<usize>,
        roc_period1: Option<usize>,
        roc_period2: Option<usize>,
        roc_period3: Option<usize>,
        roc_period4: Option<usize>,
        signal_period: Option<usize>,
    ) -> PyResult<Self> {
        let params = KstParams {
            sma_period1,
            sma_period2,
            sma_period3,
            sma_period4,
            roc_period1,
            roc_period2,
            roc_period3,
            roc_period4,
            signal_period,
        };
        let stream =
            KstStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(KstStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "kst_batch")]
#[pyo3(signature=(data,
    sma1_range, sma2_range, sma3_range, sma4_range,
    roc1_range, roc2_range, roc3_range, roc4_range,
    sig_range, kernel=None))]
pub fn kst_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    sma1_range: (usize, usize, usize),
    sma2_range: (usize, usize, usize),
    sma3_range: (usize, usize, usize),
    sma4_range: (usize, usize, usize),
    roc1_range: (usize, usize, usize),
    roc2_range: (usize, usize, usize),
    roc3_range: (usize, usize, usize),
    roc4_range: (usize, usize, usize),
    sig_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let slice = data.as_slice()?;
    let sweep = KstBatchRange {
        sma_period1: sma1_range,
        sma_period2: sma2_range,
        sma_period3: sma3_range,
        sma_period4: sma4_range,
        roc_period1: roc1_range,
        roc_period2: roc2_range,
        roc_period3: roc3_range,
        roc_period4: roc4_range,
        signal_period: sig_range,
    };
    let kern = validate_kernel(kernel, true)?;
    let combos;
    let rows;
    let cols = slice.len();
    let (line_arr, sig_arr) = {
        let tmp_combos = expand_grid(&sweep);
        rows = tmp_combos.len();
        combos = tmp_combos; // keep for metadata
        let out_line = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
        let out_sig = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
        let lo = unsafe { out_line.as_slice_mut()? };
        let so = unsafe { out_sig.as_slice_mut()? };
        py.allow_threads(|| {
            let k = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                x => x,
            };
            let simd = match k {
                Kernel::ScalarBatch => Kernel::Scalar,
                Kernel::Avx2Batch => Kernel::Scalar, // Use scalar if AVX2 not available
                Kernel::Avx512Batch => Kernel::Scalar, // Use scalar if AVX512 not available
                _ => Kernel::Scalar,
            };
            kst_batch_inner_into(slice, &sweep, simd, true, lo, so)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        (out_line, out_sig)
    };

    let d = pyo3::types::PyDict::new(py);
    d.set_item("line", line_arr.reshape((rows, cols))?)?;
    d.set_item("signal", sig_arr.reshape((rows, cols))?)?;
    // metadata vectors per combo
    d.set_item(
        "sma1",
        combos
            .iter()
            .map(|c| c.sma_period1.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "sma2",
        combos
            .iter()
            .map(|c| c.sma_period2.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "sma3",
        combos
            .iter()
            .map(|c| c.sma_period3.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "sma4",
        combos
            .iter()
            .map(|c| c.sma_period4.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "roc1",
        combos
            .iter()
            .map(|c| c.roc_period1.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "roc2",
        combos
            .iter()
            .map(|c| c.roc_period2.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "roc3",
        combos
            .iter()
            .map(|c| c.roc_period3.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "roc4",
        combos
            .iter()
            .map(|c| c.roc_period4.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "sig",
        combos
            .iter()
            .map(|c| c.signal_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(d)
}

#[cfg(feature = "python")]
pub fn register_kst_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kst_py, m)?)?;
    m.add_function(wrap_pyfunction!(kst_batch_py, m)?)?;
    Ok(())
}

// =========================== WASM Bindings ===========================

// WASM kst_into_slice has been removed - kst_into_slice is now in the main module

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KstJsResult {
    pub values: Vec<f64>, // [line..., signal...]
    pub rows: usize,      // 2
    pub cols: usize,      // len
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "kst")]
pub fn kst_js(
    data: &[f64],
    sma1: usize,
    sma2: usize,
    sma3: usize,
    sma4: usize,
    roc1: usize,
    roc2: usize,
    roc3: usize,
    roc4: usize,
    sig: usize,
) -> Result<JsValue, JsValue> {
    let prm = KstParams {
        sma_period1: Some(sma1),
        sma_period2: Some(sma2),
        sma_period3: Some(sma3),
        sma_period4: Some(sma4),
        roc_period1: Some(roc1),
        roc_period2: Some(roc2),
        roc_period3: Some(roc3),
        roc_period4: Some(roc4),
        signal_period: Some(sig),
    };
    let input = KstInput::from_slice(data, prm);

    let mut line = vec![0.0; data.len()];
    let mut signal = vec![0.0; data.len()];
    kst_into_slice(&mut line, &mut signal, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut values = line;
    values.extend_from_slice(&signal);
    let result = KstJsResult {
        values,
        rows: 2,
        cols: data.len(),
    };
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

// optional fast pointers API
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kst_into(
    in_ptr: *const f64,
    out_line_ptr: *mut f64,
    out_signal_ptr: *mut f64,
    len: usize,
    sma1: usize,
    sma2: usize,
    sma3: usize,
    sma4: usize,
    roc1: usize,
    roc2: usize,
    roc3: usize,
    roc4: usize,
    sig: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_line_ptr.is_null() || out_signal_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        // detect aliasing (same or overlapping regions)
        let in_beg = in_ptr as usize;
        let in_end = in_beg + len * 8;
        let lo_beg = out_line_ptr as usize;
        let lo_end = lo_beg + len * 8;
        let so_beg = out_signal_ptr as usize;
        let so_end = so_beg + len * 8;
        let overlap = |a0: usize, a1: usize, b0: usize, b1: usize| a0 < b1 && b0 < a1;

        let data_slice = std::slice::from_raw_parts(in_ptr, len);
        let shadow; // optional owned copy if aliasing
        let data =
            if overlap(in_beg, in_end, lo_beg, lo_end) || overlap(in_beg, in_end, so_beg, so_end) {
                // alias-safe fallback
                shadow = data_slice.to_vec();
                &shadow[..]
            } else {
                data_slice
            };

        let prm = KstParams {
            sma_period1: Some(sma1),
            sma_period2: Some(sma2),
            sma_period3: Some(sma3),
            sma_period4: Some(sma4),
            roc_period1: Some(roc1),
            roc_period2: Some(roc2),
            roc_period3: Some(roc3),
            roc_period4: Some(roc4),
            signal_period: Some(sig),
        };
        let input = KstInput::from_slice(data, prm);

        let ldst = std::slice::from_raw_parts_mut(out_line_ptr, len);
        let sdst = std::slice::from_raw_parts_mut(out_signal_ptr, len);

        // compute with stack-backed buffers (see kst_compute_into above) to avoid memory.grow during write
        kst_into_slice(ldst, sdst, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Allocate memory for output arrays
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kst_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

/// Free allocated memory
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kst_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

/// Configuration for batch KST calculation
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KstBatchConfig {
    pub sma_period1_range: (usize, usize, usize),
    pub sma_period2_range: (usize, usize, usize),
    pub sma_period3_range: (usize, usize, usize),
    pub sma_period4_range: (usize, usize, usize),
    pub roc_period1_range: (usize, usize, usize),
    pub roc_period2_range: (usize, usize, usize),
    pub roc_period3_range: (usize, usize, usize),
    pub roc_period4_range: (usize, usize, usize),
    pub signal_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct KstBatchJsOutput {
    pub values: Vec<f64>, // rows = combos*2
    pub combos: Vec<KstParams>,
    pub rows: usize, // combos*2
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "kst_batch")]
pub fn kst_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let sweep: KstBatchRange = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = data.len();

    // allocate flat buffers rows*cols for each output and fill
    let mut lines = vec![0.0; rows * cols];
    let mut sigs = vec![0.0; rows * cols];
    kst_batch_inner_into(
        data,
        &sweep,
        detect_best_kernel(),
        false,
        &mut lines,
        &mut sigs,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // interleave by outputs -> concatenate [line rows..., signal rows...]
    let mut values = lines;
    values.extend_from_slice(&sigs);

    // Note: rows = combos*2 since KST has two outputs (line and signal)
    // This differs from ALMA which reports rows = combos with single output
    let out = KstBatchJsOutput {
        values,
        combos,
        rows: rows * 2,
        cols,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

/// Fast batch calculation into pre-allocated memory
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn kst_batch_into(
    in_ptr: *const f64,
    line_out_ptr: *mut f64,
    signal_out_ptr: *mut f64,
    len: usize,
    sma_period1_start: usize,
    sma_period1_end: usize,
    sma_period1_step: usize,
    sma_period2_start: usize,
    sma_period2_end: usize,
    sma_period2_step: usize,
    sma_period3_start: usize,
    sma_period3_end: usize,
    sma_period3_step: usize,
    sma_period4_start: usize,
    sma_period4_end: usize,
    sma_period4_step: usize,
    roc_period1_start: usize,
    roc_period1_end: usize,
    roc_period1_step: usize,
    roc_period2_start: usize,
    roc_period2_end: usize,
    roc_period2_step: usize,
    roc_period3_start: usize,
    roc_period3_end: usize,
    roc_period3_step: usize,
    roc_period4_start: usize,
    roc_period4_end: usize,
    roc_period4_step: usize,
    signal_period_start: usize,
    signal_period_end: usize,
    signal_period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || line_out_ptr.is_null() || signal_out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to kst_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = KstBatchRange {
            sma_period1: (sma_period1_start, sma_period1_end, sma_period1_step),
            sma_period2: (sma_period2_start, sma_period2_end, sma_period2_step),
            sma_period3: (sma_period3_start, sma_period3_end, sma_period3_step),
            sma_period4: (sma_period4_start, sma_period4_end, sma_period4_step),
            roc_period1: (roc_period1_start, roc_period1_end, roc_period1_step),
            roc_period2: (roc_period2_start, roc_period2_end, roc_period2_step),
            roc_period3: (roc_period3_start, roc_period3_end, roc_period3_step),
            roc_period4: (roc_period4_start, roc_period4_end, roc_period4_step),
            signal_period: (signal_period_start, signal_period_end, signal_period_step),
        };

        // Calculate number of parameter combinations
        let count_range = |r: &(usize, usize, usize)| {
            if r.2 == 0 {
                0
            } else {
                ((r.1.saturating_sub(r.0)) / r.2) + 1
            }
        };

        let rows = count_range(&sweep.sma_period1).max(1)
            * count_range(&sweep.sma_period2).max(1)
            * count_range(&sweep.sma_period3).max(1)
            * count_range(&sweep.sma_period4).max(1)
            * count_range(&sweep.roc_period1).max(1)
            * count_range(&sweep.roc_period2).max(1)
            * count_range(&sweep.roc_period3).max(1)
            * count_range(&sweep.roc_period4).max(1)
            * count_range(&sweep.signal_period).max(1);
        let cols = len;

        let line_out = std::slice::from_raw_parts_mut(line_out_ptr, rows * cols);
        let signal_out = std::slice::from_raw_parts_mut(signal_out_ptr, rows * cols);

        kst_batch_inner_into(data, &sweep, Kernel::Auto, false, line_out, signal_out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
