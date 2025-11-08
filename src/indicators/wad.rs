//! # Williams Accumulation/Distribution (WAD)
//!
//! A cumulative measure of buying and selling pressure based on the relationship between
//! current close, previous close, and the high-low price range.
//!
//! ## Parameters
//! - None (WAD is a cumulative indicator without period)
//!
//! ## Inputs
//! - High, low, and close price series (or candles)
//! - All series must have the same length
//!
//! ## Returns
//! - **values**: Cumulative WAD values as `Vec<f64>` (length matches input)
//!
//! ## Developer Notes
//! - SIMD status: AVX2/AVX512 variants are implemented as unrolled, pointer-based kernels.
//!   WAD is inherently loop-carried (depends on previous close and cumulative sum),
//!   so SIMD here focuses on reducing branch mispredictions and improving ILP via unrolling
//!   and FMA-friendly expressions rather than true wide data-parallelism.
//! - Scalar path remains the reference; it is safe and branchless to minimize mispredictions.
//! - Decision: streaming update uses a branchless, FMA‑friendly kernel (cold start only) matching scalar semantics.
//! - Streaming update: O(1) with minimal state.
//! - Memory: Uses the crate’s zero-copy/uninitialized helpers where applicable.

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::CudaWad;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum WadData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct WadOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WadParams;

#[derive(Debug, Clone)]
pub struct WadInput<'a> {
    pub data: WadData<'a>,
    pub params: WadParams,
}

impl<'a> WadInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: WadData::Candles { candles },
            params: WadParams::default(),
        }
    }
    #[inline]
    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> Self {
        Self {
            data: WadData::Slices { high, low, close },
            params: WadParams::default(),
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct WadBuilder {
    kernel: Kernel,
}
impl WadBuilder {
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
    pub fn apply(self, candles: &Candles) -> Result<WadOutput, WadError> {
        let i = WadInput::from_candles(candles);
        wad_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<WadOutput, WadError> {
        let i = WadInput::from_slices(high, low, close);
        wad_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<WadStream, WadError> {
        WadStream::try_new()
    }
}

#[derive(Debug, Error)]
pub enum WadError {
    #[error("wad: Empty data provided.")]
    EmptyData,
    #[error("wad: All values are NaN.")]
    AllValuesNaN,
    #[error("wad: Invalid batch kernel.")]
    InvalidKernel,
}

#[inline]
pub fn wad(input: &WadInput) -> Result<WadOutput, WadError> {
    wad_with_kernel(input, Kernel::Auto)
}

pub fn wad_with_kernel(input: &WadInput, kernel: Kernel) -> Result<WadOutput, WadError> {
    let (high, low, close): (&[f64], &[f64], &[f64]) = match &input.data {
        WadData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        WadData::Slices { high, low, close } => (*high, *low, *close),
    };
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WadError::EmptyData);
    }
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(WadError::EmptyData);
    }
    if high.iter().all(|x| x.is_nan())
        || low.iter().all(|x| x.is_nan())
        || close.iter().all(|x| x.is_nan())
    {
        return Err(WadError::AllValuesNaN);
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    let mut out = alloc_with_nan_prefix(len, 0);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => wad_scalar(high, low, close, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => wad_avx2(high, low, close, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => wad_avx512(high, low, close, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(WadOutput { values: out })
}

#[inline(always)]
pub fn wad_scalar(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    let n = close.len();
    if n == 0 {
        return;
    }

    out[0] = 0.0;
    let mut acc = 0.0f64;
    let mut pc = close[0];

    // Branchless accumulation to reduce mispredictions; keep scalar path safe.
    for i in 1..n {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let trh = pc.max(h);
        let trl = pc.min(l);
        // booleans → {0.0,1.0} masks
        let gt = (c > pc) as i32 as f64;
        let lt = (c < pc) as i32 as f64;
        // FMA-friendly expression; identical semantics to branched logic
        let ad = gt.mul_add(c - trl, lt * (c - trh));
        acc += ad;
        out[i] = acc;
        pc = c;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_avx2(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    // Unrolled pointer-based kernel with branchless accumulation and modest prefetching.
    // WAD isn't truly vectorizable; the goal is reducing front-end stalls and branch misses.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn inner(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
        let n = close.len();
        if n == 0 {
            return;
        }
        *out.get_unchecked_mut(0) = 0.0;

        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let cp = close.as_ptr();
        let op = out.as_mut_ptr();

        let mut acc = 0.0f64;
        let mut pc = *cp;
        let mut i = 1usize;

        while i + 7 < n {
            // Prefetch ahead to help L1/L2
            use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            if i + 40 < n {
                _mm_prefetch(cp.add(i + 32) as *const i8, _MM_HINT_T0);
                _mm_prefetch(hp.add(i + 32) as *const i8, _MM_HINT_T0);
                _mm_prefetch(lp.add(i + 32) as *const i8, _MM_HINT_T0);
            }

            // ---- step i+0 ----
            let c0 = *cp.add(i);
            let h0 = *hp.add(i);
            let l0 = *lp.add(i);
            let trh0 = if pc > h0 { pc } else { h0 };
            let trl0 = if pc < l0 { pc } else { l0 };
            let gt0 = (c0 > pc) as i32 as f64;
            let lt0 = (c0 < pc) as i32 as f64;
            let ad0 = gt0.mul_add(c0 - trl0, lt0 * (c0 - trh0));
            acc += ad0;
            *op.add(i) = acc;

            // ---- step i+1 ----
            let c1 = *cp.add(i + 1);
            let h1 = *hp.add(i + 1);
            let l1 = *lp.add(i + 1);
            let trh1 = if c0 > h1 { c0 } else { h1 };
            let trl1 = if c0 < l1 { c0 } else { l1 };
            let gt1 = (c1 > c0) as i32 as f64;
            let lt1 = (c1 < c0) as i32 as f64;
            let ad1 = gt1.mul_add(c1 - trl1, lt1 * (c1 - trh1));
            acc += ad1;
            *op.add(i + 1) = acc;

            // ---- step i+2 ----
            let c2 = *cp.add(i + 2);
            let h2 = *hp.add(i + 2);
            let l2 = *lp.add(i + 2);
            let trh2 = if c1 > h2 { c1 } else { h2 };
            let trl2 = if c1 < l2 { c1 } else { l2 };
            let gt2 = (c2 > c1) as i32 as f64;
            let lt2 = (c2 < c1) as i32 as f64;
            let ad2 = gt2.mul_add(c2 - trl2, lt2 * (c2 - trh2));
            acc += ad2;
            *op.add(i + 2) = acc;

            // ---- step i+3 ----
            let c3 = *cp.add(i + 3);
            let h3 = *hp.add(i + 3);
            let l3 = *lp.add(i + 3);
            let trh3 = if c2 > h3 { c2 } else { h3 };
            let trl3 = if c2 < l3 { c2 } else { l3 };
            let gt3 = (c3 > c2) as i32 as f64;
            let lt3 = (c3 < c2) as i32 as f64;
            let ad3 = gt3.mul_add(c3 - trl3, lt3 * (c3 - trh3));
            acc += ad3;
            *op.add(i + 3) = acc;

            // ---- step i+4 ----
            let c4 = *cp.add(i + 4);
            let h4 = *hp.add(i + 4);
            let l4 = *lp.add(i + 4);
            let trh4 = if c3 > h4 { c3 } else { h4 };
            let trl4 = if c3 < l4 { c3 } else { l4 };
            let gt4 = (c4 > c3) as i32 as f64;
            let lt4 = (c4 < c3) as i32 as f64;
            let ad4 = gt4.mul_add(c4 - trl4, lt4 * (c4 - trh4));
            acc += ad4;
            *op.add(i + 4) = acc;

            // ---- step i+5 ----
            let c5 = *cp.add(i + 5);
            let h5 = *hp.add(i + 5);
            let l5 = *lp.add(i + 5);
            let trh5 = if c4 > h5 { c4 } else { h5 };
            let trl5 = if c4 < l5 { c4 } else { l5 };
            let gt5 = (c5 > c4) as i32 as f64;
            let lt5 = (c5 < c4) as i32 as f64;
            let ad5 = gt5.mul_add(c5 - trl5, lt5 * (c5 - trh5));
            acc += ad5;
            *op.add(i + 5) = acc;

            // ---- step i+6 ----
            let c6 = *cp.add(i + 6);
            let h6 = *hp.add(i + 6);
            let l6 = *lp.add(i + 6);
            let trh6 = if c5 > h6 { c5 } else { h6 };
            let trl6 = if c5 < l6 { c5 } else { l6 };
            let gt6 = (c6 > c5) as i32 as f64;
            let lt6 = (c6 < c5) as i32 as f64;
            let ad6 = gt6.mul_add(c6 - trl6, lt6 * (c6 - trh6));
            acc += ad6;
            *op.add(i + 6) = acc;

            // ---- step i+7 ----
            let c7 = *cp.add(i + 7);
            let h7 = *hp.add(i + 7);
            let l7 = *lp.add(i + 7);
            let trh7 = if c6 > h7 { c6 } else { h7 };
            let trl7 = if c6 < l7 { c6 } else { l7 };
            let gt7 = (c7 > c6) as i32 as f64;
            let lt7 = (c7 < c6) as i32 as f64;
            let ad7 = gt7.mul_add(c7 - trl7, lt7 * (c7 - trh7));
            acc += ad7;
            *op.add(i + 7) = acc;

            pc = c7;
            i += 8;
        }

        while i < n {
            let c = *cp.add(i);
            let h = *hp.add(i);
            let l = *lp.add(i);
            let trh = if pc > h { pc } else { h };
            let trl = if pc < l { pc } else { l };
            let gt = (c > pc) as i32 as f64;
            let lt = (c < pc) as i32 as f64;
            let ad = gt.mul_add(c - trl, lt * (c - trh));
            acc += ad;
            *op.add(i) = acc;
            pc = c;
            i += 1;
        }
    }

    inner(high, low, close, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_avx512(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    // Route by length to allow slightly deeper unroll for long sequences
    if high.len() <= 64 {
        wad_avx512_short(high, low, close, out);
    } else {
        wad_avx512_long(high, low, close, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_avx512_short(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    // 8x unroll (short)
    #[target_feature(enable = "avx512f,fma")]
    unsafe fn inner(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
        let n = close.len();
        if n == 0 {
            return;
        }
        *out.get_unchecked_mut(0) = 0.0;

        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let cp = close.as_ptr();
        let op = out.as_mut_ptr();

        let mut acc = 0.0f64;
        let mut pc = *cp;
        let mut i = 1usize;

        while i + 7 < n {
            let c0 = *cp.add(i);
            let h0 = *hp.add(i);
            let l0 = *lp.add(i);
            let trh0 = if pc > h0 { pc } else { h0 };
            let trl0 = if pc < l0 { pc } else { l0 };
            let gt0 = (c0 > pc) as i32 as f64;
            let lt0 = (c0 < pc) as i32 as f64;
            let ad0 = gt0.mul_add(c0 - trl0, lt0 * (c0 - trh0));
            acc += ad0;
            *op.add(i) = acc;

            let c1 = *cp.add(i + 1);
            let h1 = *hp.add(i + 1);
            let l1 = *lp.add(i + 1);
            let trh1 = if c0 > h1 { c0 } else { h1 };
            let trl1 = if c0 < l1 { c0 } else { l1 };
            let gt1 = (c1 > c0) as i32 as f64;
            let lt1 = (c1 < c0) as i32 as f64;
            let ad1 = gt1.mul_add(c1 - trl1, lt1 * (c1 - trh1));
            acc += ad1;
            *op.add(i + 1) = acc;

            let c2 = *cp.add(i + 2);
            let h2 = *hp.add(i + 2);
            let l2 = *lp.add(i + 2);
            let trh2 = if c1 > h2 { c1 } else { h2 };
            let trl2 = if c1 < l2 { c1 } else { l2 };
            let gt2 = (c2 > c1) as i32 as f64;
            let lt2 = (c2 < c1) as i32 as f64;
            let ad2 = gt2.mul_add(c2 - trl2, lt2 * (c2 - trh2));
            acc += ad2;
            *op.add(i + 2) = acc;

            let c3 = *cp.add(i + 3);
            let h3 = *hp.add(i + 3);
            let l3 = *lp.add(i + 3);
            let trh3 = if c2 > h3 { c2 } else { h3 };
            let trl3 = if c2 < l3 { c2 } else { l3 };
            let gt3 = (c3 > c2) as i32 as f64;
            let lt3 = (c3 < c2) as i32 as f64;
            let ad3 = gt3.mul_add(c3 - trl3, lt3 * (c3 - trh3));
            acc += ad3;
            *op.add(i + 3) = acc;

            let c4 = *cp.add(i + 4);
            let h4 = *hp.add(i + 4);
            let l4 = *lp.add(i + 4);
            let trh4 = if c3 > h4 { c3 } else { h4 };
            let trl4 = if c3 < l4 { c3 } else { l4 };
            let gt4 = (c4 > c3) as i32 as f64;
            let lt4 = (c4 < c3) as i32 as f64;
            let ad4 = gt4.mul_add(c4 - trl4, lt4 * (c4 - trh4));
            acc += ad4;
            *op.add(i + 4) = acc;

            let c5 = *cp.add(i + 5);
            let h5 = *hp.add(i + 5);
            let l5 = *lp.add(i + 5);
            let trh5 = if c4 > h5 { c4 } else { h5 };
            let trl5 = if c4 < l5 { c4 } else { l5 };
            let gt5 = (c5 > c4) as i32 as f64;
            let lt5 = (c5 < c4) as i32 as f64;
            let ad5 = gt5.mul_add(c5 - trl5, lt5 * (c5 - trh5));
            acc += ad5;
            *op.add(i + 5) = acc;

            let c6 = *cp.add(i + 6);
            let h6 = *hp.add(i + 6);
            let l6 = *lp.add(i + 6);
            let trh6 = if c5 > h6 { c5 } else { h6 };
            let trl6 = if c5 < l6 { c5 } else { l6 };
            let gt6 = (c6 > c5) as i32 as f64;
            let lt6 = (c6 < c5) as i32 as f64;
            let ad6 = gt6.mul_add(c6 - trl6, lt6 * (c6 - trh6));
            acc += ad6;
            *op.add(i + 6) = acc;

            let c7 = *cp.add(i + 7);
            let h7 = *hp.add(i + 7);
            let l7 = *lp.add(i + 7);
            let trh7 = if c6 > h7 { c6 } else { h7 };
            let trl7 = if c6 < l7 { c6 } else { l7 };
            let gt7 = (c7 > c6) as i32 as f64;
            let lt7 = (c7 < c6) as i32 as f64;
            let ad7 = gt7.mul_add(c7 - trl7, lt7 * (c7 - trh7));
            acc += ad7;
            *op.add(i + 7) = acc;

            pc = c7;
            i += 8;
        }

        while i < n {
            let c = *cp.add(i);
            let h = *hp.add(i);
            let l = *lp.add(i);
            let trh = if pc > h { pc } else { h };
            let trl = if pc < l { pc } else { l };
            let gt = (c > pc) as i32 as f64;
            let lt = (c < pc) as i32 as f64;
            let ad = gt.mul_add(c - trl, lt * (c - trh));
            acc += ad;
            *op.add(i) = acc;
            pc = c;
            i += 1;
        }
    }

    inner(high, low, close, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_avx512_long(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    // 16x unroll (long)
    #[target_feature(enable = "avx512f,fma")]
    unsafe fn inner(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
        let n = close.len();
        if n == 0 {
            return;
        }
        *out.get_unchecked_mut(0) = 0.0;

        let hp = high.as_ptr();
        let lp = low.as_ptr();
        let cp = close.as_ptr();
        let op = out.as_mut_ptr();

        let mut acc = 0.0f64;
        let mut pc = *cp;
        let mut i = 1usize;
        while i + 15 < n {
            use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            if i + 96 < n {
                _mm_prefetch(cp.add(i + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(hp.add(i + 64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(lp.add(i + 64) as *const i8, _MM_HINT_T0);
            }

            macro_rules! step {
                ($off:expr, $pc:expr) => {{
                    let c = *cp.add(i + $off);
                    let h = *hp.add(i + $off);
                    let l = *lp.add(i + $off);
                    let trh = if $pc > h { $pc } else { h };
                    let trl = if $pc < l { $pc } else { l };
                    let gt = (c > $pc) as i32 as f64;
                    let lt = (c < $pc) as i32 as f64;
                    let ad = gt.mul_add(c - trl, lt * (c - trh));
                    acc += ad;
                    *op.add(i + $off) = acc;
                    c
                }};
            }

            let c0 = step!(0, pc);
            let c1 = step!(1, c0);
            let c2 = step!(2, c1);
            let c3 = step!(3, c2);
            let c4 = step!(4, c3);
            let c5 = step!(5, c4);
            let c6 = step!(6, c5);
            let c7 = step!(7, c6);
            let c8 = step!(8, c7);
            let c9 = step!(9, c8);
            let c10 = step!(10, c9);
            let c11 = step!(11, c10);
            let c12 = step!(12, c11);
            let c13 = step!(13, c12);
            let c14 = step!(14, c13);
            let c15 = step!(15, c14);

            pc = c15;
            i += 16;
        }
        while i < n {
            let c = *cp.add(i);
            let h = *hp.add(i);
            let l = *lp.add(i);
            let trh = if pc > h { pc } else { h };
            let trl = if pc < l { pc } else { l };
            let gt = (c > pc) as i32 as f64;
            let lt = (c < pc) as i32 as f64;
            let ad = gt.mul_add(c - trl, lt * (c - trh));
            acc += ad;
            *op.add(i) = acc;
            pc = c;
            i += 1;
        }
    }

    inner(high, low, close, out)
}

// Per-row APIs for batch processing
#[inline(always)]
pub unsafe fn wad_row_scalar(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    wad_scalar(high, low, close, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_row_avx2(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    wad_avx2(high, low, close, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_row_avx512(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    wad_avx512(high, low, close, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_row_avx512_short(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    wad_avx512_short(high, low, close, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn wad_row_avx512_long(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    wad_avx512_long(high, low, close, out)
}

#[derive(Debug, Clone)]
pub struct WadStream {
    sum: f64,
    prev_close: Option<f64>,
}
impl WadStream {
    pub fn try_new() -> Result<Self, WadError> {
        Ok(Self {
            sum: 0.0,
            prev_close: None,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        // Branchless hot path; first tick initializes prev_close and returns 0.0
        let pc = match self.prev_close {
            Some(pc) => pc,
            None => {
                self.prev_close = Some(close);
                return self.sum;
            }
        };

        // TRH = max(high, prev_close); TRL = min(low, prev_close)
        let trh = pc.max(high);
        let trl = pc.min(low);

        // Masks as {0.0, 1.0}; FMA-friendly accumulation
        let gt = (close > pc) as i32 as f64;
        let lt = (close < pc) as i32 as f64;
        let ad = gt.mul_add(close - trl, lt * (close - trh));

        self.sum += ad;
        self.prev_close = Some(close);
        self.sum
    }
}

#[derive(Clone, Debug)]
pub struct WadBatchRange {
    pub dummy: (usize, usize, usize),
}
impl Default for WadBatchRange {
    fn default() -> Self {
        Self { dummy: (0, 0, 0) }
    }
}
#[derive(Clone, Debug, Default)]
pub struct WadBatchBuilder {
    range: WadBatchRange,
    kernel: Kernel,
}
impl WadBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<WadBatchOutput, WadError> {
        wad_batch_with_kernel(high, low, close, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        k: Kernel,
    ) -> Result<WadBatchOutput, WadError> {
        WadBatchBuilder::new()
            .kernel(k)
            .apply_slices(high, low, close)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<WadBatchOutput, WadError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
    pub fn with_default_candles(c: &Candles) -> Result<WadBatchOutput, WadError> {
        WadBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
    }
}

pub fn wad_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    k: Kernel,
) -> Result<WadBatchOutput, WadError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(WadError::InvalidKernel),
    };
    wad_batch_par_slice(high, low, close, kernel)
}

#[derive(Clone, Debug)]
pub struct WadBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}
impl WadBatchOutput {
    pub fn row_for_params(&self, _: &WadParams) -> Option<usize> {
        Some(0)
    }
    pub fn values_for(&self, _: &WadParams) -> Option<&[f64]> {
        Some(&self.values)
    }
}

#[inline(always)]
pub fn expand_grid(_r: &WadBatchRange) -> Vec<WadParams> {
    // WAD has no parameters, so always return single element
    // Using with_capacity to avoid reallocation
    let mut result = Vec::with_capacity(1);
    result.push(WadParams);
    result
}

#[inline(always)]
pub fn wad_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
) -> Result<WadBatchOutput, WadError> {
    wad_batch_inner(high, low, close, kern, false)
}
#[inline(always)]
pub fn wad_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
) -> Result<WadBatchOutput, WadError> {
    wad_batch_inner(high, low, close, kern, true)
}

#[inline(always)]
fn wad_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
    _parallel: bool,
) -> Result<WadBatchOutput, WadError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WadError::EmptyData);
    }
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(WadError::EmptyData);
    }
    if high.iter().all(|x| x.is_nan())
        || low.iter().all(|x| x.is_nan())
        || close.iter().all(|x| x.is_nan())
    {
        return Err(WadError::AllValuesNaN);
    }

    // Use uninit matrix + prefix init, parity with ALMA
    let mut buf_mu = make_uninit_matrix(1, len);
    init_matrix_prefixes(&mut buf_mu, len, &[0]); // WAD warmup = 0

    let mut guard = ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // Write in place
    wad_batch_inner_into(high, low, close, kern, false, out)?;

    // Turn matrix back into Vec without copy
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(WadBatchOutput {
        values,
        rows: 1,
        cols: len,
    })
}

#[inline(always)]
fn wad_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    kern: Kernel,
    _parallel: bool,
    out: &mut [f64],
) -> Result<(), WadError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WadError::EmptyData);
    }
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(WadError::EmptyData);
    }
    if high.iter().all(|x| x.is_nan())
        || low.iter().all(|x| x.is_nan())
        || close.iter().all(|x| x.is_nan())
    {
        return Err(WadError::AllValuesNaN);
    }
    if out.len() != len {
        return Err(WadError::EmptyData); // length guard
    }

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    unsafe {
        match actual {
            Kernel::Scalar | Kernel::ScalarBatch => wad_row_scalar(high, low, close, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => wad_row_avx2(high, low, close, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => wad_row_avx512(high, low, close, out),
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    fn check_wad_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = WadInput::from_candles(&candles);
        let output = wad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_wad_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = WadInput::from_candles(&candles);
        let output = wad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        let expected_last_five_wad = [
            158503.46790000016,
            158279.46790000016,
            158014.46790000016,
            158186.46790000016,
            157605.46790000016,
        ];
        let start = output.values.len().saturating_sub(5);
        for (i, &val) in output.values[start..].iter().enumerate() {
            let exp = expected_last_five_wad[i];
            assert!(
                (val - exp).abs() < 1e-4,
                "[{}] WAD {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                exp
            );
        }
        Ok(())
    }

    fn check_wad_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input = WadInput::from_slices(&[], &[], &[]);
        let result = wad_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_wad_all_values_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_slice = [f64::NAN, f64::NAN, f64::NAN];
        let input = WadInput::from_slices(&nan_slice, &nan_slice, &nan_slice);
        let result = wad_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_wad_basic_slice(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 11.0, 12.0];
        let low = [9.0, 9.0, 10.0, 10.0];
        let close = [9.5, 10.5, 10.5, 11.5];
        let input = WadInput::from_slices(&high, &low, &close);
        let output = wad_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), 4);
        assert!((output.values[0] - 0.0).abs() < 1e-10);
        assert!((output.values[1] - 1.5).abs() < 1e-10);
        assert!((output.values[2] - 1.5).abs() < 1e-10);
        assert!((output.values[3] - 3.0).abs() < 1e-10);
        Ok(())
    }

    fn check_wad_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let high = source_type(&candles, "high");
        let low = source_type(&candles, "low");
        let close = source_type(&candles, "close");
        let batch_output =
            wad_with_kernel(&WadInput::from_slices(high, low, close), kernel)?.values;
        let mut stream = WadStream::try_new()?;
        let mut stream_values = Vec::with_capacity(close.len());
        for ((&h, &l), &c) in high.iter().zip(low).zip(close) {
            stream_values.push(stream.update(h, l, c));
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] WAD streaming mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    // New: small fixed‐input test for wad_scalar via wad_with_kernel
    fn check_wad_small_example(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        // 5‐bar example from documentation:
        let high = [10.0, 11.0, 12.0, 11.5, 12.5];
        let low = [9.0, 9.5, 11.0, 10.5, 11.0];
        let close = [9.5, 10.5, 11.5, 11.0, 12.0];
        let expected = [0.0, 1.0, 2.0, 1.5, 2.5];

        let input = WadInput::from_slices(&high, &low, &close);
        let output = wad_with_kernel(&input, kernel)?;
        // output.values should be length 5
        assert_eq!(output.values.len(), 5);

        for i in 0..5 {
            let got = output.values[i];
            let exp = expected[i];
            assert!(
                (got - exp).abs() < 1e-10,
                "[{}] WAD {:?} small example mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                got,
                exp
            );
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_wad_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // WAD has no parameters, so we test with default configuration
        let test_configs = vec![
            WadParams::default(),
            // Since WAD has no parameters, we only have one configuration
        ];

        for (param_idx, params) in test_configs.iter().enumerate() {
            let input = WadInput {
                data: WadData::Candles { candles: &candles },
                params: params.clone(),
            };
            let output = wad_with_kernel(&input, kernel)?;

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
    fn check_wad_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Since WAD has no parameters, batch processing is simplified
        // We test with different input data configurations to ensure proper initialization
        let test_configs = vec!["high", "low", "close"];

        for (cfg_idx, &source) in test_configs.iter().enumerate() {
            let output = wad_batch_with_kernel(
                source_type(&candles, "high"),
                source_type(&candles, "low"),
                source_type(&candles, "close"),
                kernel,
            )?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;

                // Check all three poison patterns with detailed context
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) - source: {}",
                        test_name, cfg_idx, val, bits, row, col, idx, source
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) - source: {}",
                        test_name, cfg_idx, val, bits, row, col, idx, source
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) - source: {}",
                        test_name, cfg_idx, val, bits, row, col, idx, source
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    macro_rules! generate_all_wad_tests {
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

    generate_all_wad_tests!(
        check_wad_partial_params,
        check_wad_accuracy,
        check_wad_empty_data,
        check_wad_all_values_nan,
        check_wad_basic_slice,
        check_wad_streaming,
        check_wad_small_example,
        check_wad_no_poison
    );

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
            }
        };
    }

    gen_batch_tests!(check_batch_no_poison);

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_wad_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy: Generate realistic OHLC data with proper constraints
        let strat = (1usize..=200).prop_flat_map(|len| {
            prop::collection::vec(
                (1.0f64..1000.0f64).prop_flat_map(|base_price| {
                    // Generate proper OHLC where low <= close <= high
                    let range = base_price * 0.1; // 10% daily range
                    let low = base_price - range;
                    let high = base_price + range;

                    // Generate close within [low, high]
                    (low..=high).prop_map(move |close| {
                        // Ensure low <= close <= high
                        let actual_low = low.min(close);
                        let actual_high = high.max(close);
                        (actual_high, actual_low, close)
                    })
                }),
                len,
            )
        });

        proptest::test_runner::TestRunner::default().run(&strat, |ohlc_data| {
            let (highs, lows, closes): (Vec<f64>, Vec<f64>, Vec<f64>) =
                ohlc_data.into_iter().map(|(h, l, c)| (h, l, c)).unzip3();

            // Create input
            let input = WadInput::from_slices(&highs, &lows, &closes);

            // Calculate WAD with specified kernel and scalar reference
            let WadOutput { values: out } = wad_with_kernel(&input, kernel).unwrap();
            let WadOutput { values: ref_out } = wad_with_kernel(&input, Kernel::Scalar).unwrap();

            // Property 1: First value should always be 0.0
            prop_assert_eq!(out[0], 0.0, "First WAD value must be 0.0");
            prop_assert_eq!(ref_out[0], 0.0, "First reference WAD value must be 0.0");

            // Property 2: WAD is cumulative - verify accumulation logic
            let mut expected_sum = 0.0;
            let mut prev_close = closes[0];

            for i in 1..closes.len() {
                let trh = if prev_close > highs[i] {
                    prev_close
                } else {
                    highs[i]
                };
                let trl = if prev_close < lows[i] {
                    prev_close
                } else {
                    lows[i]
                };

                let ad = if closes[i] > prev_close {
                    closes[i] - trl
                } else if closes[i] < prev_close {
                    closes[i] - trh
                } else {
                    0.0
                };

                expected_sum += ad;

                // Check that the calculated value matches expected
                prop_assert!(
                    (out[i] - expected_sum).abs() <= 1e-9,
                    "WAD mismatch at idx {}: got {}, expected {}",
                    i,
                    out[i],
                    expected_sum
                );

                prev_close = closes[i];
            }

            // Property 3: Different kernels should produce identical results
            for i in 0..out.len() {
                let y = out[i];
                let r = ref_out[i];

                // Check for exact bit-level equality for special values
                if !y.is_finite() || !r.is_finite() {
                    prop_assert_eq!(
                        y.to_bits(),
                        r.to_bits(),
                        "NaN/Inf mismatch at idx {}: {} vs {}",
                        i,
                        y,
                        r
                    );
                    continue;
                }

                // For finite values, allow small numerical tolerance
                let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                prop_assert!(
                    (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                    "Kernel mismatch at idx {}: {} vs {} (diff: {}, ulp: {})",
                    i,
                    y,
                    r,
                    (y - r).abs(),
                    ulp_diff
                );
            }

            // Property 4: When consecutive closes are equal, AD should be 0
            for i in 1..closes.len() {
                if (closes[i] - closes[i - 1]).abs() < f64::EPSILON {
                    let ad_change = if i == 1 {
                        out[i] - 0.0
                    } else {
                        out[i] - out[i - 1]
                    };
                    prop_assert!(
                        ad_change.abs() < 1e-9,
                        "WAD should not change when close[{}] == close[{}], but changed by {}",
                        i,
                        i - 1,
                        ad_change
                    );
                }
            }

            // Edge case 1: Single element should return [0.0]
            if closes.len() == 1 {
                prop_assert_eq!(out.len(), 1);
                prop_assert_eq!(out[0], 0.0);
            }

            // Edge case 2: All constant prices should accumulate to 0
            if closes
                .windows(2)
                .all(|w| (w[0] - w[1]).abs() < f64::EPSILON)
            {
                // All WAD values should remain 0 after the first
                for i in 0..out.len() {
                    prop_assert!(
                        out[i].abs() < 1e-9,
                        "WAD should be 0 for constant prices, but got {} at index {}",
                        out[i],
                        i
                    );
                }
            }

            // Edge case 3: Strictly increasing closes should have positive accumulation
            let strictly_increasing = closes.windows(2).all(|w| w[1] > w[0]);
            if strictly_increasing && closes.len() > 1 {
                // Each WAD value should be >= previous (monotonic increase)
                for i in 1..out.len() {
                    prop_assert!(
							out[i] >= out[i-1] - 1e-9,
							"WAD should increase monotonically for strictly increasing prices, but {} < {} at index {}",
							out[i], out[i-1], i
						);
                }
            }

            // Edge case 4: Strictly decreasing closes should have negative accumulation
            let strictly_decreasing = closes.windows(2).all(|w| w[1] < w[0]);
            if strictly_decreasing && closes.len() > 1 {
                // Each WAD value should be <= previous (monotonic decrease)
                for i in 1..out.len() {
                    prop_assert!(
							out[i] <= out[i-1] + 1e-9,
							"WAD should decrease monotonically for strictly decreasing prices, but {} > {} at index {}",
							out[i], out[i-1], i
						);
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    // Helper for unzipping tuples
    trait Unzip3<A, B, C> {
        fn unzip3(self) -> (Vec<A>, Vec<B>, Vec<C>);
    }

    impl<A, B, C, I> Unzip3<A, B, C> for I
    where
        I: Iterator<Item = (A, B, C)>,
    {
        fn unzip3(self) -> (Vec<A>, Vec<B>, Vec<C>) {
            let (mut a_vec, mut b_vec, mut c_vec) = (Vec::new(), Vec::new(), Vec::new());
            for (a, b, c) in self {
                a_vec.push(a);
                b_vec.push(b);
                c_vec.push(c);
            }
            (a_vec, b_vec, c_vec)
        }
    }

    #[cfg(feature = "proptest")]
    generate_all_wad_tests!(check_wad_property);

    #[test]
    fn test_wad_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Use repository CSV fixture to match other tests
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = WadInput::from_candles(&candles);

        // Baseline via existing Vec-returning API
        let baseline = wad(&input)?.values;

        // New API: preallocated output buffer
        let mut out = vec![0.0; baseline.len()];
        #[allow(unused_variables)]
        {
            #[cfg(not(feature = "wasm"))]
            {
                wad_into(&input, &mut out)?;
            }
            #[cfg(feature = "wasm")]
            {
                wad_into_slice(&mut out, &input, Kernel::Auto)?;
            }
        }

        assert_eq!(baseline.len(), out.len());

        // Treat NaN == NaN as equal; otherwise exact equality
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for i in 0..baseline.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "Mismatch at index {}: baseline={}, into={}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }
}

// Helper functions for WASM zero-copy optimization
#[inline(always)]
fn wad_prepare<'a>(
    input: &'a WadInput,
    _kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, Kernel), WadError> {
    let (high, low, close): (&[f64], &[f64], &[f64]) = match &input.data {
        WadData::Candles { candles } => (
            source_type(candles, "high"),
            source_type(candles, "low"),
            source_type(candles, "close"),
        ),
        WadData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WadError::EmptyData);
    }
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(WadError::EmptyData);
    }
    if high.iter().all(|x| x.is_nan())
        || low.iter().all(|x| x.is_nan())
        || close.iter().all(|x| x.is_nan())
    {
        return Err(WadError::AllValuesNaN);
    }

    let chosen = match _kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((high, low, close, len, chosen))
}

#[inline]
pub fn wad_into_slice(dst: &mut [f64], input: &WadInput, kern: Kernel) -> Result<(), WadError> {
    let (high, low, close, len, chosen) = wad_prepare(input, kern)?;

    if dst.len() != len {
        return Err(WadError::EmptyData);
    }

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => wad_scalar(high, low, close, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => wad_avx2(high, low, close, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => wad_avx512(high, low, close, dst),
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[cfg(not(feature = "wasm"))]
/// Write Williams Accumulation/Distribution (WAD) values into a caller-provided buffer.
///
/// - Preserves the module's warmup behavior (WAD has no NaN warmup; first value is 0.0).
/// - `out.len()` must equal the input series length; returns the module's length error on mismatch.
/// - Uses `Kernel::Auto` for runtime kernel selection and performs no internal allocations.
pub fn wad_into(input: &WadInput, out: &mut [f64]) -> Result<(), WadError> {
    wad_into_slice(out, input, Kernel::Auto)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "wad_cuda_dev")]
#[pyo3(signature = (high_f32, low_f32, close_f32, device_id=0))]
pub fn wad_cuda_dev_py(
    py: Python<'_>,
    high_f32: PyReadonlyArray1<'_, f32>,
    low_f32: PyReadonlyArray1<'_, f32>,
    close_f32: PyReadonlyArray1<'_, f32>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let high = high_f32.as_slice()?;
    let low = low_f32.as_slice()?;
    let close = close_f32.as_slice()?;

    let inner = py.allow_threads(|| {
        let cuda = CudaWad::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.wad_series_dev(high, low, close)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "wad_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, close_f32, device_id=0))]
pub fn wad_cuda_batch_dev_py(
    py: Python<'_>,
    high_f32: PyReadonlyArray1<'_, f32>,
    low_f32: PyReadonlyArray1<'_, f32>,
    close_f32: PyReadonlyArray1<'_, f32>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let high = high_f32.as_slice()?;
    let low = low_f32.as_slice()?;
    let close = close_f32.as_slice()?;
    let inner = py.allow_threads(|| {
        let cuda = CudaWad::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.wad_batch_dev(high, low, close)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "wad_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, close_tm_f32, device_id=0))]
pub fn wad_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    low_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    close_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let rows = high_tm_f32.shape()[0];
    let cols = high_tm_f32.shape()[1];
    if low_tm_f32.shape() != [rows, cols] || close_tm_f32.shape() != [rows, cols] {
        return Err(PyValueError::new_err("high/low/close shapes must match"));
    }
    let high = high_tm_f32.as_slice()?;
    let low = low_tm_f32.as_slice()?;
    let close = close_tm_f32.as_slice()?;
    let inner = py.allow_threads(|| {
        let cuda = CudaWad::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.wad_many_series_one_param_time_major_dev(high, low, close, cols, rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "python")]
#[pyfunction(name = "wad")]
#[pyo3(signature = (high, low, close, kernel=None))]
pub fn wad_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let input = WadInput::from_slices(high_slice, low_slice, close_slice);

    let result_vec: Vec<f64> = py
        .allow_threads(|| wad_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "WadStream")]
pub struct WadStreamPy {
    stream: WadStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl WadStreamPy {
    #[new]
    fn new() -> PyResult<Self> {
        let stream = WadStream::try_new().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(WadStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "wad_batch")]
#[pyo3(signature = (high, low, close, kernel=None))]
pub fn wad_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    let cols = high_slice.len();
    let rows = 1usize;

    let out_arr = unsafe { numpy::PyArray1::<f64>::new(py, [rows * cols], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        wad_batch_inner_into(high_slice, low_slice, close_slice, kern, true, out_slice)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wad_js(high: &[f64], low: &[f64], close: &[f64]) -> Result<Vec<f64>, JsValue> {
    let input = WadInput::from_slices(high, low, close);

    let mut output = vec![0.0; high.len()];

    wad_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wad_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wad_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wad_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to wad_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        let input = WadInput::from_slices(high, low, close);

        // Check for aliasing - if any input pointer equals output pointer
        if high_ptr as *const f64 == out_ptr as *const f64
            || low_ptr as *const f64 == out_ptr as *const f64
            || close_ptr as *const f64 == out_ptr as *const f64
        {
            // Handle aliasing by using temp buffer
            let mut temp = vec![0.0; len];
            wad_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            wad_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn wad_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || close_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to wad_batch_into"));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        wad_batch_inner_into(high, low, close, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(1) // rows
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WadBatchConfig {
    // WAD has no parameters, but we keep the structure for consistency
    pub dummy: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct WadBatchJsOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = wad_batch)]
pub fn wad_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    _config: JsValue, // accept and ignore, WAD has no params
) -> Result<JsValue, JsValue> {
    let out = wad_batch_inner(high, low, close, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let js = WadBatchJsOutput {
        values: out.values,
        rows: out.rows,
        cols: out.cols,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
