//! # Predictive Moving Average (PMA)
//!
//! Ehlers' Predictive Moving Average using weighted MAs for responsive prediction.
//!
//! ## Parameters
//! - **data**: Input price data
//!
//! ## Returns
//! - `predict`: Vec<f64> - Predictive values matching input length
//! - `trigger`: Vec<f64> - Signal line values matching input length
//!
//! ## Developer Status
//! - SIMD: AVX2/AVX512 stubs delegate to scalar. Single-series PMA is a tight
//!   recurrence (A,S,A1,S1,A2,T) with loop-carried deps; SIMD brings ≤0–5% in practice.
//!   We keep runtime selection short-circuited to scalar for correctness/simplicity.
//! - Scalar: Optimized O(1) rolling updates for both WMAs and trigger; warmup preserved.
//! - Batch: Single-row “unified” compute writes directly into caller buffers; no
//!   row-specific SIMD due to lack of cross-row reuse.
//! - CUDA: FP32 kernels via `CudaPma` wrapper validated against scalar within tolerance;
//!   VRAM handles reuse shared ALMA CAI v3 + DLPack v1.x helpers for interop.

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
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
use std::convert::AsRef;
use thiserror::Error;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for PmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PmaData::Slice(slice) => slice,
            PmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PmaOutput {
    pub predict: Vec<f64>,
    pub trigger: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PmaParams;

impl Default for PmaParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct PmaInput<'a> {
    pub data: PmaData<'a>,
    pub params: PmaParams,
}

impl<'a> PmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: PmaParams) -> Self {
        Self {
            data: PmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: PmaParams) -> Self {
        Self {
            data: PmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", PmaParams::default())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PmaBuilder {
    kernel: Kernel,
}

impl Default for PmaBuilder {
    fn default() -> Self {
        Self {
            kernel: Kernel::Auto,
        }
    }
}

impl PmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<PmaOutput, PmaError> {
        let i = PmaInput::from_candles(c, "close", PmaParams::default());
        pma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<PmaOutput, PmaError> {
        let i = PmaInput::from_slice(d, PmaParams::default());
        pma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<PmaStream, PmaError> {
        PmaStream::try_new(PmaParams::default())
    }
}

#[derive(Debug, Error)]
pub enum PmaError {
    #[error("pma: Empty data provided.")]
    EmptyInputData,
    #[error("pma: All values are NaN.")]
    AllValuesNaN,
    #[error("pma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("pma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("pma: Output slice length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("pma: Invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("pma: invalid kernel for batch API: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("pma: size overflow computing rows*cols: rows = {rows}, cols = {cols}")]
    SizeOverflow { rows: usize, cols: usize },
}

#[inline(always)]
fn pma_first_valid_idx(data: &[f64]) -> Result<usize, PmaError> {
    if data.is_empty() {
        return Err(PmaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(PmaError::AllValuesNaN)?;
    let valid = data.len() - first;
    if valid < 7 {
        return Err(PmaError::NotEnoughValidData { needed: 7, valid });
    }
    Ok(first)
}

#[inline]
pub fn pma(input: &PmaInput) -> Result<PmaOutput, PmaError> {
    pma_with_kernel(input, Kernel::Auto)
}

pub fn pma_with_kernel(input: &PmaInput, kernel: Kernel) -> Result<PmaOutput, PmaError> {
    let data: &[f64] = match &input.data {
        PmaData::Candles { candles, source } => source_type(candles, source),
        PmaData::Slice(sl) => sl,
    };

    let first = pma_first_valid_idx(data)?;

    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => pma_scalar(data, first),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => pma_avx2(data, first),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => pma_avx512(data, first),
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn pma_scalar(data: &[f64], first_valid_idx: usize) -> Result<PmaOutput, PmaError> {
    let n = data.len();
    let warmup_period = first_valid_idx + 7;
    let mut predict = alloc_with_nan_prefix(n, warmup_period);
    let mut trigger = alloc_with_nan_prefix(n, warmup_period);

    if n <= first_valid_idx + 6 {
        return Ok(PmaOutput { predict, trigger });
    }

    const INV_28: f64 = 1.0 / 28.0;
    const INV_10: f64 = 1.0 / 10.0;

    let mut x_ring = [0.0_f64; 7];
    let mut w_ring = [0.0_f64; 7];
    let mut p_ring = [0.0_f64; 4];
    let mut x_head = 0usize;
    let mut w_head = 0usize;
    let mut p_head = 0usize;

    let mut A = 0.0_f64; // sum of last 7 prices
    let mut S = 0.0_f64; // weighted sum of last 7 prices
    let mut A1 = 0.0_f64; // sum of last 7 wma1
    let mut S1 = 0.0_f64; // weighted sum of last 7 wma1
    let mut A2 = 0.0_f64; // sum of last 4 predicts
    let mut T = 0.0_f64; // weighted sum (4,3,2,1) of predicts

    let j0 = first_valid_idx + 6;

    unsafe {
        let dp = data.as_ptr();

        // Seed A,S and x_ring at j0
        let x0 = *dp.add(j0 - 6);
        let x1 = *dp.add(j0 - 5);
        let x2 = *dp.add(j0 - 4);
        let x3 = *dp.add(j0 - 3);
        let x4 = *dp.add(j0 - 2);
        let x5 = *dp.add(j0 - 1);
        let x6 = *dp.add(j0 - 0);

        x_ring[0] = x0;
        x_ring[1] = x1;
        x_ring[2] = x2;
        x_ring[3] = x3;
        x_ring[4] = x4;
        x_ring[5] = x5;
        x_ring[6] = x6;

        A = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + x6);

        let s01 = x0.mul_add(1.0, 2.0 * x1);
        let s23 = (3.0 * x2) + (4.0 * x3);
        let s45 = (5.0 * x4) + (6.0 * x5);
        S = (s01 + s23) + s45 + 7.0 * x6;

        let mut w1 = S * INV_28;

        // Update WMA2 accumulators
        let old_A1 = A1;
        let old_w = w_ring[w_head];
        S1 = (7.0_f64).mul_add(w1, S1) - old_A1;
        A1 = A1 + w1 - old_w;
        w_ring[w_head] = w1;
        w_head += 1;
        if w_head == 7 {
            w_head = 0;
        }

        let mut w2 = S1 * INV_28;
        let mut pr = (2.0_f64).mul_add(w1, -w2);
        *predict.get_unchecked_mut(j0) = pr;

        // Seed trigger accumulators with first predict
        let old_A2 = A2;
        let old_p = p_ring[p_head];
        T = (4.0_f64).mul_add(pr, T) - old_A2;
        A2 = A2 + pr - old_p;
        p_ring[p_head] = pr;
        p_head += 1;
        if p_head == 4 {
            p_head = 0;
        }
        *trigger.get_unchecked_mut(j0) = f64::NAN;

        // Main loop
        let mut j = j0 + 1;
        while j < n {
            let x_new = *dp.add(j);
            let x_old = x_ring[x_head];
            let old_A = A;

            A = A + x_new - x_old;
            S = (7.0_f64).mul_add(x_new, S) - old_A;

            x_ring[x_head] = x_new;
            x_head += 1;
            if x_head == 7 {
                x_head = 0;
            }

            w1 = S * INV_28;

            let old_A1 = A1;
            let w_old = w_ring[w_head];
            S1 = (7.0_f64).mul_add(w1, S1) - old_A1;
            A1 = A1 + w1 - w_old;

            w_ring[w_head] = w1;
            w_head += 1;
            if w_head == 7 {
                w_head = 0;
            }

            w2 = S1 * INV_28;

            pr = (2.0_f64).mul_add(w1, -w2);
            *predict.get_unchecked_mut(j) = pr;

            let old_A2 = A2;
            let p_old = p_ring[p_head];
            T = (4.0_f64).mul_add(pr, T) - old_A2;
            A2 = A2 + pr - p_old;

            p_ring[p_head] = pr;
            p_head += 1;
            if p_head == 4 {
                p_head = 0;
            }

            if j >= first_valid_idx + 9 {
                *trigger.get_unchecked_mut(j) = T * INV_10;
            } else {
                *trigger.get_unchecked_mut(j) = f64::NAN;
            }

            j += 1;
        }
    }

    Ok(PmaOutput { predict, trigger })
}

#[inline(always)]
fn pma_compute_into(
    data: &[f64],
    first_valid_idx: usize,
    _kernel: Kernel,
    predict_out: &mut [f64],
    trigger_out: &mut [f64],
) {
    let n = data.len();
    if n <= first_valid_idx + 6 {
        return;
    }

    const INV_28: f64 = 1.0 / 28.0;
    const INV_10: f64 = 1.0 / 10.0;

    let mut x_ring = [0.0_f64; 7];
    let mut w_ring = [0.0_f64; 7];
    let mut p_ring = [0.0_f64; 4];
    let mut x_head = 0usize;
    let mut w_head = 0usize;
    let mut p_head = 0usize;

    let mut A = 0.0_f64; // price sum
    let mut S = 0.0_f64; // price weighted sum
    let mut A1 = 0.0_f64; // wma1 sum
    let mut S1 = 0.0_f64; // wma1 weighted sum
    let mut A2 = 0.0_f64; // predict sum
    let mut T = 0.0_f64; // predict weighted (4,3,2,1)

    let j0 = first_valid_idx + 6;

    unsafe {
        let dp = data.as_ptr();

        // Seed A,S and x_ring at j0
        let x0 = *dp.add(j0 - 6);
        let x1 = *dp.add(j0 - 5);
        let x2 = *dp.add(j0 - 4);
        let x3 = *dp.add(j0 - 3);
        let x4 = *dp.add(j0 - 2);
        let x5 = *dp.add(j0 - 1);
        let x6 = *dp.add(j0 - 0);

        x_ring[0] = x0;
        x_ring[1] = x1;
        x_ring[2] = x2;
        x_ring[3] = x3;
        x_ring[4] = x4;
        x_ring[5] = x5;
        x_ring[6] = x6;

        A = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + x6);

        let s01 = x0.mul_add(1.0, 2.0 * x1);
        let s23 = (3.0 * x2) + (4.0 * x3);
        let s45 = (5.0 * x4) + (6.0 * x5);
        S = (s01 + s23) + s45 + 7.0 * x6;

        let mut w1 = S * INV_28;

        let old_A1 = A1;
        let old_w = w_ring[w_head];
        S1 = (7.0_f64).mul_add(w1, S1) - old_A1;
        A1 = A1 + w1 - old_w;
        w_ring[w_head] = w1;
        w_head += 1;
        if w_head == 7 {
            w_head = 0;
        }

        let mut w2 = S1 * INV_28;
        let mut pr = (2.0_f64).mul_add(w1, -w2);
        *predict_out.get_unchecked_mut(j0) = pr;

        let old_A2 = A2;
        let old_p = p_ring[p_head];
        T = (4.0_f64).mul_add(pr, T) - old_A2;
        A2 = A2 + pr - old_p;
        p_ring[p_head] = pr;
        p_head += 1;
        if p_head == 4 {
            p_head = 0;
        }

        *trigger_out.get_unchecked_mut(j0) = f64::NAN;

        // Main loop
        let mut j = j0 + 1;
        while j < n {
            let x_new = *dp.add(j);
            let x_old = x_ring[x_head];
            let old_A = A;

            A = A + x_new - x_old;
            S = (7.0_f64).mul_add(x_new, S) - old_A;

            x_ring[x_head] = x_new;
            x_head += 1;
            if x_head == 7 {
                x_head = 0;
            }

            w1 = S * INV_28;

            let old_A1 = A1;
            let w_old = w_ring[w_head];
            S1 = (7.0_f64).mul_add(w1, S1) - old_A1;
            A1 = A1 + w1 - w_old;

            w_ring[w_head] = w1;
            w_head += 1;
            if w_head == 7 {
                w_head = 0;
            }

            w2 = S1 * INV_28;
            pr = (2.0_f64).mul_add(w1, -w2);

            *predict_out.get_unchecked_mut(j) = pr;

            let old_A2 = A2;
            let p_old = p_ring[p_head];
            T = (4.0_f64).mul_add(pr, T) - old_A2;
            A2 = A2 + pr - p_old;

            p_ring[p_head] = pr;
            p_head += 1;
            if p_head == 4 {
                p_head = 0;
            }

            if j >= first_valid_idx + 9 {
                *trigger_out.get_unchecked_mut(j) = T * INV_10;
            } else {
                *trigger_out.get_unchecked_mut(j) = f64::NAN;
            }

            j += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pma_avx512(data: &[f64], first_valid_idx: usize) -> Result<PmaOutput, PmaError> {
    pma_scalar(data, first_valid_idx)
}

#[inline]
pub fn pma_avx2(data: &[f64], first_valid_idx: usize) -> Result<PmaOutput, PmaError> {
    pma_scalar(data, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pma_avx512_short(data: &[f64], first_valid_idx: usize) -> Result<PmaOutput, PmaError> {
    pma_scalar(data, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn pma_avx512_long(data: &[f64], first_valid_idx: usize) -> Result<PmaOutput, PmaError> {
    pma_scalar(data, first_valid_idx)
}

#[inline]
pub fn pma_batch_with_kernel(
    data: &[f64],
    sweep: &PmaBatchRange,
    k: Kernel,
) -> Result<PmaBatchOutput, PmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(PmaError::InvalidKernelForBatch(other)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    pma_batch_par_slice(data, sweep, simd)
}

#[inline]
pub fn pma_batch_unified_with_kernel(
    data: &[f64],
    k: Kernel,
) -> Result<PmaBatchOutputUnified, PmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch,
    };
    pma_batch_unified_inner(data, kernel)
}

#[inline]
fn pma_batch_unified_inner(data: &[f64], kern: Kernel) -> Result<PmaBatchOutputUnified, PmaError> {
    let first = pma_first_valid_idx(data)?;

    let rows = 2usize;
    let cols = data.len();
    let _ = rows
        .checked_mul(cols)
        .ok_or(PmaError::SizeOverflow { rows, cols })?;

    // Allocate rows×cols uninitialized and write NaN prefixes exactly
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm = [first + 7 - 1; 2];
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Write into the matrix without extra allocations/copies
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let outf: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let (row0, row1) = outf.split_at_mut(cols);
    pma_compute_into(
        data,
        first,
        match kern {
            Kernel::ScalarBatch => Kernel::Scalar,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::Avx512Batch => Kernel::Avx512,
            _ => Kernel::Scalar,
        },
        row0,
        row1,
    );

    // Turn the backing buffer into Vec<f64> without copy
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    Ok(PmaBatchOutputUnified { values, rows, cols })
}

#[derive(Debug, Clone)]
pub struct PmaStream {
    buffer: [f64; 7],
    wma1: [f64; 7],
    idx: usize,
    filled7: bool,
    // Keep last 4 predictions for trigger calculation
    pred4: [f64; 4],
    pred_idx: usize,
    pred_filled: bool,
}

impl PmaStream {
    pub fn try_new(_params: PmaParams) -> Result<Self, PmaError> {
        Ok(Self {
            buffer: [f64::NAN; 7],
            wma1: [0.0; 7],
            idx: 0,
            filled7: false,
            pred4: [f64::NAN; 4],
            pred_idx: 0,
            pred_filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.buffer[self.idx] = value;
        self.idx = (self.idx + 1) % 7;
        if !self.filled7 && self.idx == 0 {
            self.filled7 = true;
        }
        if !self.filled7 {
            return None;
        }

        let s = |k: usize| self.buffer[(self.idx + k) % 7];
        let wma1_j =
            (7.0 * s(6) + 6.0 * s(5) + 5.0 * s(4) + 4.0 * s(3) + 3.0 * s(2) + 2.0 * s(1) + s(0))
                / 28.0;
        self.wma1[self.idx] = wma1_j;

        let w = |k: usize| self.wma1[(self.idx + k) % 7];
        let wma2 =
            (7.0 * w(6) + 6.0 * w(5) + 5.0 * w(4) + 4.0 * w(3) + 3.0 * w(2) + 2.0 * w(1) + w(0))
                / 28.0;

        let predict = 2.0 * wma1_j - wma2;

        // Ring of the last 4 predicts
        self.pred4[self.pred_idx] = predict;
        self.pred_idx = (self.pred_idx + 1) % 4;
        if !self.pred_filled && self.pred_idx == 0 {
            self.pred_filled = true;
        }

        let trigger = if self.pred_filled {
            // Order: [t3,t2,t1,t0] == [current, -1, -2, -3]
            let t3 = self.pred4[(self.pred_idx + 3) % 4];
            let t2 = self.pred4[(self.pred_idx + 2) % 4];
            let t1 = self.pred4[(self.pred_idx + 1) % 4];
            let t0 = self.pred4[(self.pred_idx + 0) % 4];
            (4.0 * t3 + 3.0 * t2 + 2.0 * t1 + t0) / 10.0
        } else {
            f64::NAN
        };

        Some((predict, trigger))
    }
}

#[derive(Clone, Debug)]
pub struct PmaBatchRange {
    // Only dummy for now, PMA has no true sweepable params, but we keep for API parity
    pub dummy: (usize, usize, usize),
}

impl Default for PmaBatchRange {
    fn default() -> Self {
        Self { dummy: (0, 0, 0) }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PmaBatchBuilder {
    range: PmaBatchRange,
    kernel: Kernel,
}

impl PmaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn apply_slice(self, data: &[f64]) -> Result<PmaBatchOutput, PmaError> {
        pma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<PmaBatchOutput, PmaError> {
        PmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<PmaBatchOutput, PmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<PmaBatchOutput, PmaError> {
        PmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct PmaBatchOutput {
    pub predict: Vec<f64>,
    pub trigger: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}
impl PmaBatchOutput {
    pub fn values_for(&self, _dummy: &PmaParams) -> Option<(&[f64], &[f64])> {
        Some((&self.predict[..], &self.trigger[..]))
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct PmaBatchOutputUnified {
    pub values: Vec<f64>, // row-major: row0=predict, row1=trigger
    pub rows: usize,      // 2
    pub cols: usize,      // data.len()
}

#[inline(always)]
pub fn expand_grid(_r: &PmaBatchRange) -> Vec<PmaParams> {
    vec![PmaParams {}]
}

#[inline(always)]
pub fn pma_batch_slice(
    data: &[f64],
    sweep: &PmaBatchRange,
    kern: Kernel,
) -> Result<PmaBatchOutput, PmaError> {
    pma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn pma_batch_par_slice(
    data: &[f64],
    sweep: &PmaBatchRange,
    kern: Kernel,
) -> Result<PmaBatchOutput, PmaError> {
    pma_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn pma_batch_inner(
    data: &[f64],
    _sweep: &PmaBatchRange,
    kern: Kernel,
    _parallel: bool,
) -> Result<PmaBatchOutput, PmaError> {
    let first = pma_first_valid_idx(data)?;
    let out = match kern {
        Kernel::Scalar => pma_scalar(data, first)?,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => pma_avx2(data, first)?,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => pma_avx512(data, first)?,
        _ => unreachable!(),
    };
    Ok(PmaBatchOutput {
        predict: out.predict,
        trigger: out.trigger,
        rows: 1,
        cols: data.len(),
    })
}

// Row functions - stubs for AVX2/AVX512
#[inline(always)]
pub unsafe fn pma_row_scalar(
    data: &[f64],
    first: usize,
    _stride: usize,
    _dummy: *const f64,
    _inv_n: f64,
    out_predict: &mut [f64],
    out_trigger: &mut [f64],
) {
    // Compute directly into output slices - no allocation!
    pma_compute_into(data, first, Kernel::Scalar, out_predict, out_trigger);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pma_row_avx2(
    data: &[f64],
    first: usize,
    stride: usize,
    dummy: *const f64,
    inv_n: f64,
    out_predict: &mut [f64],
    out_trigger: &mut [f64],
) {
    pma_row_scalar(data, first, stride, dummy, inv_n, out_predict, out_trigger);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pma_row_avx512(
    data: &[f64],
    first: usize,
    stride: usize,
    dummy: *const f64,
    inv_n: f64,
    out_predict: &mut [f64],
    out_trigger: &mut [f64],
) {
    pma_row_scalar(data, first, stride, dummy, inv_n, out_predict, out_trigger);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pma_row_avx512_short(
    data: &[f64],
    first: usize,
    stride: usize,
    dummy: *const f64,
    inv_n: f64,
    out_predict: &mut [f64],
    out_trigger: &mut [f64],
) {
    pma_row_scalar(data, first, stride, dummy, inv_n, out_predict, out_trigger);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn pma_row_avx512_long(
    data: &[f64],
    first: usize,
    stride: usize,
    dummy: *const f64,
    inv_n: f64,
    out_predict: &mut [f64],
    out_trigger: &mut [f64],
) {
    pma_row_scalar(data, first, stride, dummy, inv_n, out_predict, out_trigger);
}

// Move pma_into_slice outside of #[cfg(feature = "wasm")] so it's always available
#[inline]
pub fn pma_into_slice(
    predict_dst: &mut [f64],
    trigger_dst: &mut [f64],
    input: &PmaInput,
    kern: Kernel,
) -> Result<(), PmaError> {
    let data = input.as_ref();

    if predict_dst.len() != data.len() || trigger_dst.len() != data.len() {
        return Err(PmaError::OutputLengthMismatch {
            expected: data.len(),
            got: predict_dst.len().min(trigger_dst.len()),
        });
    }

    let first = pma_first_valid_idx(data)?;

    let chosen = match kern {
        Kernel::Auto => Kernel::Scalar,
        k => k,
    };

    // Compute into provided buffers
    pma_compute_into(data, first, chosen, predict_dst, trigger_dst);

    // Write warmup NaNs (prefix only)
    let warm_end = first + 7 - 1;
    for v in &mut predict_dst[..warm_end] {
        *v = f64::NAN;
    }
    for v in &mut trigger_dst[..warm_end] {
        *v = f64::NAN;
    }

    Ok(())
}

/// Compute PMA into caller-provided output buffers without allocating.
///
/// - Preserves the module's NaN warmups exactly (prefix NaNs for both series).
/// - `predict_out.len()` and `trigger_out.len()` must equal `input` length.
/// - Uses `Kernel::Auto` for dispatch and writes results directly into the slices.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn pma_into(
    input: &PmaInput,
    predict_out: &mut [f64],
    trigger_out: &mut [f64],
) -> Result<(), PmaError> {
    pma_into_slice(predict_out, trigger_out, input, Kernel::Auto)
}

//--------------------------
// WASM Bindings
//--------------------------

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pma_js(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    let input = PmaInput::from_slice(data, PmaParams {});
    let rows = 2usize;
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str(&PmaError::SizeOverflow { rows, cols }.to_string()))?;
    let mut values = vec![0.0; total];
    {
        let (pred, trig) = values.split_at_mut(cols);
        pma_into_slice(pred, trig, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    // Return flat array [predict..., trigger...]
    Ok(values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pma_into(
    in_ptr: *const f64,
    predict_ptr: *mut f64,
    trigger_ptr: *mut f64,
    len: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || predict_ptr.is_null() || trigger_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = PmaParams {};
        let input = PmaInput::from_slice(data, params);

        // Check for aliasing - need to handle all possible combinations
        let need_temp =
            in_ptr == predict_ptr || in_ptr == trigger_ptr || predict_ptr == trigger_ptr;

        if need_temp {
            let mut temp_predict = vec![0.0; len];
            let mut temp_trigger = vec![0.0; len];

            pma_into_slice(&mut temp_predict, &mut temp_trigger, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let predict_out = std::slice::from_raw_parts_mut(predict_ptr, len);
            let trigger_out = std::slice::from_raw_parts_mut(trigger_ptr, len);

            predict_out.copy_from_slice(&temp_predict);
            trigger_out.copy_from_slice(&temp_trigger);
        } else {
            let predict_out = std::slice::from_raw_parts_mut(predict_ptr, len);
            let trigger_out = std::slice::from_raw_parts_mut(trigger_ptr, len);

            pma_into_slice(predict_out, trigger_out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pma_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pma_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PmaBatchConfig {
    // PMA has no parameters, but we keep this for API consistency
    pub dummy: Option<usize>,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PmaJsOutput {
    pub values: Vec<f64>, // [predict..., trigger...]
    pub rows: usize,      // 2
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct PmaBatchJsOutput {
    pub predict: Vec<f64>,
    pub trigger: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pma_batch(data: &[f64]) -> Result<JsValue, JsValue> {
    // PMA has no parameters, so batch just returns single run
    let input = PmaInput::from_slice(data, PmaParams {});
    let mut predict = vec![0.0; data.len()];
    let mut trigger = vec![0.0; data.len()];

    pma_into_slice(&mut predict, &mut trigger, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let output = PmaBatchJsOutput {
        predict,
        trigger,
        rows: 1, // No parameter sweep
        cols: data.len(),
    };

    serde_wasm_bindgen::to_value(&output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Optional pointer-based unified INTO for flat output
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pma_unified_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    let rows = 2usize;
    let cols = len;
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str(&PmaError::SizeOverflow { rows, cols }.to_string()))?;
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        let input = PmaInput::from_slice(data, PmaParams {});
        let (pred, trig) = out.split_at_mut(cols);
        pma_into_slice(pred, trig, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows) // rows
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pma_batch_into(
    in_ptr: *const f64,
    predict_ptr: *mut f64,
    trigger_ptr: *mut f64,
    len: usize,
) -> Result<usize, JsValue> {
    // Since PMA has no parameters, this is the same as pma_into but returns row count
    pma_into(in_ptr, predict_ptr, trigger_ptr, len)?;
    Ok(1) // Always returns 1 row since no parameter sweep
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct PmaStreamWasm {
    stream: PmaStream,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl PmaStreamWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<PmaStreamWasm, JsValue> {
        let params = PmaParams {};
        let stream = PmaStream::try_new(params).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(PmaStreamWasm { stream })
    }

    pub fn update(&mut self, value: f64) -> Result<Vec<f64>, JsValue> {
        match self.stream.update(value) {
            Some((predict, trigger)) => Ok(vec![predict, trigger]),
            None => Ok(vec![f64::NAN, f64::NAN]),
        }
    }
}

//--------------------------
// Python Bindings
//--------------------------
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

// ---------------- CUDA Python bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, moving_averages::CudaPma};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::{DeviceArrayF32Py, make_device_array_py};

#[cfg(feature = "python")]
#[pyfunction(name = "pma")]
#[pyo3(signature = (data, kernel=None))]
pub fn pma_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let input = PmaInput::from_slice(slice_in, PmaParams {});

    // Zero-copy: Vec<f64> -> PyArray1 via into_pyarray
    let out = py
        .allow_threads(|| pma_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((out.predict.into_pyarray(py), out.trigger.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "PmaStream")]
pub struct PmaStreamPy {
    stream: PmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PmaStreamPy {
    #[new]
    fn new() -> PyResult<Self> {
        let params = PmaParams {};
        let stream =
            PmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PmaStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "pma_batch")]
#[pyo3(signature = (data, kernel=None))]
pub fn pma_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let (rows, cols) = (2usize, slice_in.len());
    let size = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err(PmaError::SizeOverflow { rows, cols }.to_string()))?;

    // Preallocate one flat array; fill directly with our unified inner
    let values_arr = unsafe { PyArray1::<f64>::new(py, [size], false) };
    let values_slice = unsafe { values_arr.as_slice_mut()? };

    py.allow_threads(|| -> PyResult<()> {
        // Compute directly into values_slice without intermediate copies
        let first = pma_first_valid_idx(slice_in)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // NaN warmup for both rows (ALMA-style prefix init)
        let warm = first + 7 - 1;
        let warm_prefixes = [warm; 2];
        let values_mu: &mut [core::mem::MaybeUninit<f64>] = unsafe {
            core::slice::from_raw_parts_mut(
                values_slice.as_mut_ptr() as *mut core::mem::MaybeUninit<f64>,
                values_slice.len(),
            )
        };
        init_matrix_prefixes(values_mu, cols, &warm_prefixes);

        let (row0, row1) = values_slice.split_at_mut(cols);
        pma_compute_into(
            slice_in,
            first,
            match kern {
                Kernel::Auto => Kernel::Scalar,
                Kernel::ScalarBatch => Kernel::Scalar,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::Avx512Batch => Kernel::Avx512,
                _ => Kernel::Scalar,
            },
            row0,
            row1,
        );
        Ok(())
    })?;

    let dict = PyDict::new(py);
    dict.set_item("values", values_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "pma_cuda_batch_dev")]
#[pyo3(signature = (data_f32, device_id=0))]
pub fn pma_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py)> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_f32.as_slice()?;
    let sweep = PmaBatchRange::default();
    let pair = py.allow_threads(|| {
        let cuda = CudaPma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.pma_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let predict = make_device_array_py(device_id, pair.predict)?;
    let trigger = make_device_array_py(device_id, pair.trigger)?;
    Ok((predict, trigger))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "pma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, device_id=0))]
pub fn pma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py)> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let shape = data_tm_f32.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected time-major 2D array"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let flat = data_tm_f32.as_slice()?;
    let pair = py.allow_threads(|| {
        let cuda = CudaPma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.pma_many_series_one_param_time_major_dev(flat, cols, rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let predict = make_device_array_py(device_id, pair.predict)?;
    let trigger = make_device_array_py(device_id, pair.trigger)?;
    Ok((predict, trigger))
}

//--------------------------
// Tests
//--------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_pma_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PmaInput::with_default_candles(&candles);
        let output = pma_with_kernel(&input, kernel)?;
        assert_eq!(output.predict.len(), candles.close.len());
        assert_eq!(output.trigger.len(), candles.close.len());
        Ok(())
    }

    fn check_pma_with_slice(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let input = PmaInput::from_slice(&data, PmaParams {});
        let output = pma_with_kernel(&input, kernel)?;
        assert_eq!(output.predict.len(), data.len());
        assert_eq!(output.trigger.len(), data.len());
        Ok(())
    }

    fn check_pma_not_enough_data(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let input = PmaInput::from_slice(&data, PmaParams {});
        let result = pma_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for not enough data");
        Ok(())
    }

    fn check_pma_all_values_nan(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let input = PmaInput::from_slice(&data, PmaParams {});
        let result = pma_with_kernel(&input, kernel);
        assert!(result.is_err(), "Expected error for all values NaN");
        Ok(())
    }

    fn check_pma_expected_values(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PmaInput::from_candles(&candles, "hl2", PmaParams {});
        let result = pma_with_kernel(&input, kernel)?;

        assert_eq!(
            result.predict.len(),
            candles.close.len(),
            "Predict length mismatch"
        );
        assert_eq!(
            result.trigger.len(),
            candles.close.len(),
            "Trigger length mismatch"
        );

        let expected_predict = [
            59208.18749999999,
            59233.83609693878,
            59213.19132653061,
            59199.002551020414,
            58993.318877551,
        ];
        let expected_trigger = [
            59157.70790816327,
            59208.60076530612,
            59218.6763392857,
            59211.1443877551,
            59123.05019132652,
        ];

        assert!(
            result.predict.len() >= 5,
            "Output length too short for checking"
        );
        let start_idx = result.predict.len() - 5;
        for i in 0..5 {
            let calc_val = result.predict[start_idx + i];
            let exp_val = expected_predict[i];
            assert!(
                (calc_val - exp_val).abs() < 1e-1,
                "Mismatch in predict at index {}: expected {}, got {}",
                start_idx + i,
                exp_val,
                calc_val
            );
        }
        for i in 0..5 {
            let calc_val = result.trigger[start_idx + i];
            let exp_val = expected_trigger[i];
            assert!(
                (calc_val - exp_val).abs() < 1e-1,
                "Mismatch in trigger at index {}: expected {}, got {}",
                start_idx + i,
                exp_val,
                calc_val
            );
        }
        Ok(())
    }

    macro_rules! generate_all_pma_tests {
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
    fn check_pma_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with different candle sources since PMA has no other parameters
        let test_sources = vec![
            "close", "open", "high", "low", "hl2", "hlc3", "ohlc4", "volume",
        ];

        for (source_idx, source) in test_sources.iter().enumerate() {
            let input = PmaInput::from_candles(&candles, source, PmaParams {});
            let output = pma_with_kernel(&input, kernel)?;

            // Check predict values
            for (i, &val) in output.predict.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in predict array with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in predict array with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in predict array with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }
            }

            // Check trigger values
            for (i, &val) in output.trigger.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 in trigger array with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 in trigger array with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 in trigger array with source: {} (source set {})",
                        test_name, val, bits, i, source, source_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_pma_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_pma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate data with at least 7 valid points after the first valid index
        let strat = prop::collection::vec(
            (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
            7..400, // Ensure at least 7 points for PMA to work
        );

        proptest::test_runner::TestRunner::default().run(&strat, |data| {
            let input = PmaInput::from_slice(&data, PmaParams {});

            // Test kernel consistency - all kernels should produce identical results
            let result = pma_with_kernel(&input, kernel)?;
            let ref_result = pma_with_kernel(&input, Kernel::Scalar)?;

            // Check that both outputs have the correct length
            prop_assert_eq!(result.predict.len(), data.len());
            prop_assert_eq!(result.trigger.len(), data.len());
            prop_assert_eq!(ref_result.predict.len(), data.len());
            prop_assert_eq!(ref_result.trigger.len(), data.len());

            // PMA has a warmup period of 7 (first_valid_idx + 7)
            let warmup_period = 7;

            // Verify NaN pattern in warmup period
            for i in 0..warmup_period {
                prop_assert!(
                    result.predict[i].is_nan(),
                    "Expected NaN in predict warmup at index {}",
                    i
                );
                prop_assert!(
                    result.trigger[i].is_nan(),
                    "Expected NaN in trigger warmup at index {}",
                    i
                );
            }

            // Special case: Test with constant data
            if data.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON)
                && data.len() >= warmup_period
            {
                // For constant data, PMA should output close to that constant value
                for i in warmup_period..data.len() {
                    if result.predict[i].is_finite() {
                        prop_assert!(
                            (result.predict[i] - data[0]).abs() < 1e-9,
                            "Constant data test failed: predict[{}] = {} should be close to {}",
                            i,
                            result.predict[i],
                            data[0]
                        );
                    }
                }
            }

            // Compare results between kernels and verify mathematical properties
            for i in warmup_period..data.len() {
                // Compare predict values between kernels
                if result.predict[i].is_finite() && ref_result.predict[i].is_finite() {
                    let diff_predict = (result.predict[i] - ref_result.predict[i]).abs();
                    prop_assert!(
                        diff_predict < 1e-10,
                        "Predict mismatch at index {}: kernel={}, scalar={}, diff={}",
                        i,
                        result.predict[i],
                        ref_result.predict[i],
                        diff_predict
                    );
                } else {
                    prop_assert_eq!(
                        result.predict[i].is_nan(),
                        ref_result.predict[i].is_nan(),
                        "NaN mismatch in predict at index {}",
                        i
                    );
                }

                // Compare trigger values between kernels
                if result.trigger[i].is_finite() && ref_result.trigger[i].is_finite() {
                    let diff_trigger = (result.trigger[i] - ref_result.trigger[i]).abs();
                    prop_assert!(
                        diff_trigger < 1e-10,
                        "Trigger mismatch at index {}: kernel={}, scalar={}, diff={}",
                        i,
                        result.trigger[i],
                        ref_result.trigger[i],
                        diff_trigger
                    );
                } else {
                    prop_assert_eq!(
                        result.trigger[i].is_nan(),
                        ref_result.trigger[i].is_nan(),
                        "NaN mismatch in trigger at index {}",
                        i
                    );
                }

                // Verify predict values are within tight bounds of the input window
                if i >= warmup_period && result.predict[i].is_finite() {
                    let window_start = i.saturating_sub(6);
                    let window_data = &data[window_start..=i];
                    let min_val = window_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_val = window_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                    // Since PMA is predictive (2*wma1 - wma2), it can extrapolate slightly
                    // But should still be reasonably bounded
                    let tolerance = (max_val - min_val).abs() * 0.1 + 1e-9;
                    prop_assert!(
                        result.predict[i] >= min_val - tolerance
                            && result.predict[i] <= max_val + tolerance,
                        "Predict value {} at index {} outside bounds [{}, {}] with tolerance {}",
                        result.predict[i],
                        i,
                        min_val - tolerance,
                        max_val + tolerance,
                        tolerance
                    );
                }

                // Verify WMA1 calculation for specific positions
                if i == warmup_period && i >= 6 {
                    // Calculate WMA1 manually for the first valid position
                    let wma1_expected = (7.0 * data[i]
                        + 6.0 * data[i - 1]
                        + 5.0 * data[i - 2]
                        + 4.0 * data[i - 3]
                        + 3.0 * data[i - 4]
                        + 2.0 * data[i - 5]
                        + data[i - 6])
                        / 28.0;

                    // At position warmup_period, WMA2 calculation requires 7 WMA1 values
                    // Since we're at the first position, we can't verify the full predict formula
                    // But we can at least check the predict is related to wma1_expected
                    if result.predict[i].is_finite() {
                        // The predict value should be influenced by wma1_expected
                        let window_start = i.saturating_sub(6);
                        let window = &data[window_start..=i];
                        let window_avg = window.iter().sum::<f64>() / window.len() as f64;
                        let min = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let max = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        prop_assert!(
                            (result.predict[i] - window_avg).abs() < (max - min).abs() + 1e-9,
                            "Predict value {} at index {} seems unrelated to window average {}",
                            result.predict[i],
                            i,
                            window_avg
                        );
                    }
                }

                // Verify trigger calculation when we have enough predict values
                if i >= warmup_period + 3
                    && result.trigger[i].is_finite()
                    && result.predict[i].is_finite()
                {
                    // Trigger formula: (4*predict[i] + 3*predict[i-1] + 2*predict[i-2] + predict[i-3]) / 10
                    if result.predict[i - 1].is_finite()
                        && result.predict[i - 2].is_finite()
                        && result.predict[i - 3].is_finite()
                    {
                        let expected_trigger = (4.0 * result.predict[i]
                            + 3.0 * result.predict[i - 1]
                            + 2.0 * result.predict[i - 2]
                            + result.predict[i - 3])
                            / 10.0;
                        let trigger_diff = (result.trigger[i] - expected_trigger).abs();
                        prop_assert!(
                            trigger_diff < 1e-10,
                            "Trigger calculation error at index {}: expected {}, got {}, diff={}",
                            i,
                            expected_trigger,
                            result.trigger[i],
                            trigger_diff
                        );
                    }
                }
            }

            // Test edge case: exactly 7 data points
            if data.len() == 7 {
                // Should have at least one valid output at position 6
                prop_assert!(
                    result.predict[6].is_finite(),
                    "With exactly 7 points, predict[6] should be finite but got NaN"
                );
            }

            Ok(())
        })?;

        Ok(())
    }

    generate_all_pma_tests!(
        check_pma_default_candles,
        check_pma_with_slice,
        check_pma_not_enough_data,
        check_pma_all_values_nan,
        check_pma_expected_values,
        check_pma_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_pma_tests!(check_pma_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = PmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        // Should always be a single row, for PMA's batch API
        assert_eq!(output.rows, 1, "Expected exactly 1 row");
        assert_eq!(output.cols, c.close.len());
        assert_eq!(output.predict.len(), c.close.len());
        assert_eq!(output.trigger.len(), c.close.len());

        // Spot check output against direct calculation
        let input = PmaInput::from_candles(&c, "close", PmaParams::default());
        let expected = pma_with_kernel(&input, kernel)?;

        for (i, (&a, &b)) in output
            .predict
            .iter()
            .zip(expected.predict.iter())
            .enumerate()
        {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() < 1e-12,
                "[{test}] predict mismatch at idx {i}: batch={}, direct={}",
                a,
                b
            );
        }
        for (i, (&a, &b)) in output
            .trigger
            .iter()
            .zip(expected.trigger.iter())
            .enumerate()
        {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() < 1e-12,
                "[{test}] trigger mismatch at idx {i}: batch={}, direct={}",
                a,
                b
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
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test with different candle sources since PMA has no parameters to sweep
        let test_sources = vec!["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"];

        for (source_idx, source) in test_sources.iter().enumerate() {
            let output = PmaBatchBuilder::new()
                .kernel(kernel)
                .apply_candles(&c, source)?;

            // Check predict values
            for (idx, &val) in output.predict.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Source {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at index {} in predict array with source: {}",
                        test, source_idx, val, bits, idx, source
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Source {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at index {} in predict array with source: {}",
                        test, source_idx, val, bits, idx, source
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Source {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at index {} in predict array with source: {}",
                        test, source_idx, val, bits, idx, source
                    );
                }
            }

            // Check trigger values
            for (idx, &val) in output.trigger.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Source {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at index {} in trigger array with source: {}",
                        test, source_idx, val, bits, idx, source
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Source {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at index {} in trigger array with source: {}",
                        test, source_idx, val, bits, idx, source
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Source {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at index {} in trigger array with source: {}",
                        test, source_idx, val, bits, idx, source
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_pma_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Use existing CSV candles to match repository testing conventions
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = PmaInput::with_default_candles(&candles);

        // Baseline via existing Vec-returning API
        let base = pma_with_kernel(&input, Kernel::Auto)?;

        // Preallocate outputs and compute via into API
        let n = candles.close.len();
        let mut out_predict = vec![0.0; n];
        let mut out_trigger = vec![0.0; n];

        #[cfg(not(feature = "wasm"))]
        {
            pma_into(&input, &mut out_predict, &mut out_trigger)?;
        }

        // Length parity
        assert_eq!(base.predict.len(), out_predict.len());
        assert_eq!(base.trigger.len(), out_trigger.len());

        // Helper: equality with NaN == NaN and tight epsilon for finite values
        fn eq_or_both_nan_eps(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for i in 0..n {
            assert!(
                eq_or_both_nan_eps(base.predict[i], out_predict[i]),
                "predict mismatch at {i}: api={}, into={}",
                base.predict[i],
                out_predict[i]
            );
            assert!(
                eq_or_both_nan_eps(base.trigger[i], out_trigger[i]),
                "trigger mismatch at {i}: api={}, into={}",
                base.trigger[i],
                out_trigger[i]
            );
        }

        Ok(())
    }
}
