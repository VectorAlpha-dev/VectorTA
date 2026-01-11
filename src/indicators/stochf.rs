//! # Stochastic Fast (StochF)
//!
//! A fast variant of the stochastic oscillator with minimal smoothing for more responsive signals.
//! Compares closing price to high-low range over a shorter period than standard stochastic.
//!
//! ## Parameters
//! - **fastk_period**: Lookback period for highest high/lowest low (default: 5)
//! - **fastd_period**: Period for moving average of %K to get %D (default: 3)
//! - **fastd_matype**: MA type - currently only SMA (0) supported (default: 0)
//!
//! ## Inputs
//! - High, low, and close price series (or candles)
//! - All series must have the same length
//!
//! ## Returns
//! - **k**: Fast %K line as `Vec<f64>` (length matches input, range 0-100)
//! - **d**: Fast %D line as `Vec<f64>` (length matches input, range 0-100)
//!
//! ## Developer Notes
//! - Scalar: optimized O(n) via ring-indexed monotonic deques; small-window path uses tight direct scans with unrolling; %K and %D (SMA) computed in one pass.
//! - SIMD: AVX2/AVX512 implemented only for small windows; benches show underperformance vs scalar at 100k (default params). Runtime `Auto` short-circuits to `Scalar`.
//! - Batch: per-row kernel mirrors scalar optimizations; no cross-row sharing (window-dependent extrema).
//! - Streaming: O(1) amortized via ring-indexed monotonic deques for %K and a running-sum ring for %D (SMA).
//! - Memory: follows ALMA patterns (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes). %D supports SMA (matype=0).
//! - Decision log: SIMD remains short-circuited to the scalar path for performance; CUDA wrapper includes typed errors + VRAM checks and feeds shared `DeviceArrayF32Py` (CAI v3 + DLPack v1.x) without changing numerical outputs.
//!
//! Binding test status (Oct 29, 2025):
//! - WASM: all stochf binding tests pass (accuracy last-5 match Rust refs at abs tol 1e-4).
//! - Python: one failure in streaming API test:
//!   - tests/python/test_stochf.py::TestStochFStream::test_stochf_stream_basic
//!     expects first value at the 5th update (fastk=5) to yield a finite %K.
//!     Observed: %K is NaN. Likely cause: StochfStream deque state treats a full
//!     window as empty when `head == tail` (no size tracking), making HH/LL fall
//!     back to ±INF and producing NaN. This is an implementation issue, not a
//!     bindings test setup problem. No changes made to kernels pending review.

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
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
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum StochfData<'a> {
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
pub struct StochfOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct StochfParams {
    pub fastk_period: Option<usize>,
    pub fastd_period: Option<usize>,
    pub fastd_matype: Option<usize>,
}

impl Default for StochfParams {
    fn default() -> Self {
        Self {
            fastk_period: Some(5),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StochfInput<'a> {
    pub data: StochfData<'a>,
    pub params: StochfParams,
}

impl<'a> StochfInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: StochfParams) -> Self {
        Self {
            data: StochfData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: StochfParams,
    ) -> Self {
        Self {
            data: StochfData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, StochfParams::default())
    }
    #[inline]
    pub fn get_fastk_period(&self) -> usize {
        self.params.fastk_period.unwrap_or(5)
    }
    #[inline]
    pub fn get_fastd_period(&self) -> usize {
        self.params.fastd_period.unwrap_or(3)
    }
    #[inline]
    pub fn get_fastd_matype(&self) -> usize {
        self.params.fastd_matype.unwrap_or(0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct StochfBuilder {
    fastk_period: Option<usize>,
    fastd_period: Option<usize>,
    fastd_matype: Option<usize>,
    kernel: Kernel,
}

impl Default for StochfBuilder {
    fn default() -> Self {
        Self {
            fastk_period: None,
            fastd_period: None,
            fastd_matype: None,
            kernel: Kernel::Auto,
        }
    }
}

impl StochfBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline]
    pub fn fastk_period(mut self, n: usize) -> Self {
        self.fastk_period = Some(n);
        self
    }
    #[inline]
    pub fn fastd_period(mut self, n: usize) -> Self {
        self.fastd_period = Some(n);
        self
    }
    #[inline]
    pub fn fastd_matype(mut self, t: usize) -> Self {
        self.fastd_matype = Some(t);
        self
    }
    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn apply(self, candles: &Candles) -> Result<StochfOutput, StochfError> {
        let p = StochfParams {
            fastk_period: self.fastk_period,
            fastd_period: self.fastd_period,
            fastd_matype: self.fastd_matype,
        };
        let i = StochfInput::from_candles(candles, p);
        stochf_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<StochfOutput, StochfError> {
        let p = StochfParams {
            fastk_period: self.fastk_period,
            fastd_period: self.fastd_period,
            fastd_matype: self.fastd_matype,
        };
        let i = StochfInput::from_slices(high, low, close, p);
        stochf_with_kernel(&i, self.kernel)
    }
    #[inline]
    pub fn into_stream(self) -> Result<StochfStream, StochfError> {
        let p = StochfParams {
            fastk_period: self.fastk_period,
            fastd_period: self.fastd_period,
            fastd_matype: self.fastd_matype,
        };
        StochfStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum StochfError {
    #[error("stochf: Empty data provided.")]
    EmptyInputData,
    #[error("stochf: Invalid period (fastk={fastk}, fastd={fastd}), data length={data_len}.")]
    InvalidPeriod {
        fastk: usize,
        fastd: usize,
        data_len: usize,
    },
    #[error("stochf: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "stochf: Not enough valid data after first valid index (needed={needed}, valid={valid})."
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("stochf: Invalid output size (expected={expected}, k_got={k_got}, d_got={d_got}).")]
    OutputLengthMismatch {
        expected: usize,
        k_got: usize,
        d_got: usize,
    },
    #[error("stochf: invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("stochf: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline]
pub fn stochf(input: &StochfInput) -> Result<StochfOutput, StochfError> {
    stochf_with_kernel(input, Kernel::Auto)
}

/// Write StochF outputs into caller-provided buffers without allocations.
///
/// Preserves NaN warmups exactly like the Vec-returning API: the first
/// `first_valid + fastk - 1` values of %K and the first
/// `first_valid + fastk + fastd - 2` values of %D are prefilled with
/// quiet-NaNs. All slice lengths must equal the input length.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn stochf_into(
    input: &StochfInput,
    out_k: &mut [f64],
    out_d: &mut [f64],
    
) -> Result<(), StochfError> {
    
    let (high, low, close) = match &input.data {
        StochfData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| StochfError::EmptyInputData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| StochfError::EmptyInputData)?;
            let close = candles
                .select_candle_field("close")
                .map_err(|_| StochfError::EmptyInputData)?;
            (high, low, close)
        }
        StochfData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(StochfError::EmptyInputData);
    }
    let len = high.len();
    if low.len() != len || close.len() != len {
        return Err(StochfError::EmptyInputData);
    }
    if out_k.len() != len || out_d.len() != len {
        return Err(StochfError::OutputLengthMismatch {
            expected: len,
            k_got: out_k.len(),
            d_got: out_d.len(),
        });
    }

    let fastk = input.get_fastk_period();
    let fastd = input.get_fastd_period();
    let _matype = input.get_fastd_matype();
    if fastk == 0 || fastd == 0 || fastk > len || fastd > len {
        return Err(StochfError::InvalidPeriod {
            fastk,
            fastd,
            data_len: len,
        });
    }

    
    let first = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(StochfError::AllValuesNaN)?;
    if (len - first) < fastk {
        return Err(StochfError::NotEnoughValidData {
            needed: fastk,
            valid: len - first,
        });
    }

    
    let k_warm = first + fastk - 1;
    let d_warm = first + fastk + fastd - 2;
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    for v in &mut out_k[..k_warm.min(len)] {
        *v = qnan;
    }
    for v in &mut out_d[..d_warm.min(len)] {
        *v = qnan;
    }

    
    stochf_into_slice(out_k, out_d, input, Kernel::Auto)
}

#[inline]
pub fn stochf_into_slice(
    dst_k: &mut [f64],
    dst_d: &mut [f64],
    input: &StochfInput,
    kernel: Kernel,
) -> Result<(), StochfError> {
    let (high, low, close) = match &input.data {
        StochfData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| StochfError::EmptyInputData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| StochfError::EmptyInputData)?;
            let close = candles
                .select_candle_field("close")
                .map_err(|_| StochfError::EmptyInputData)?;
            (high, low, close)
        }
        StochfData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(StochfError::EmptyInputData);
    }
    let len = high.len();
    if low.len() != len || close.len() != len {
        return Err(StochfError::EmptyInputData);
    }
    if dst_k.len() != len || dst_d.len() != len {
        return Err(StochfError::OutputLengthMismatch {
            expected: len,
            k_got: dst_k.len(),
            d_got: dst_d.len(),
        });
    }

    let fastk_period = input.get_fastk_period();
    let fastd_period = input.get_fastd_period();
    let matype = input.get_fastd_matype();

    if fastk_period == 0 || fastd_period == 0 || fastk_period > len || fastd_period > len {
        return Err(StochfError::InvalidPeriod {
            fastk: fastk_period,
            fastd: fastd_period,
            data_len: len,
        });
    }
    let first_valid_idx = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(StochfError::AllValuesNaN)?;
    if (len - first_valid_idx) < fastk_period {
        return Err(StochfError::NotEnoughValidData {
            needed: fastk_period,
            valid: len - first_valid_idx,
        });
    }

    
    
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => stochf_scalar(
                high,
                low,
                close,
                fastk_period,
                fastd_period,
                matype,
                first_valid_idx,
                dst_k,
                dst_d,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => stochf_avx2(
                high,
                low,
                close,
                fastk_period,
                fastd_period,
                matype,
                first_valid_idx,
                dst_k,
                dst_d,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => stochf_avx512(
                high,
                low,
                close,
                fastk_period,
                fastd_period,
                matype,
                first_valid_idx,
                dst_k,
                dst_d,
            ),
            _ => unreachable!(),
        }
    }

    
    let k_warmup = (first_valid_idx + fastk_period - 1).min(len);
    let d_warmup = (first_valid_idx + fastk_period + fastd_period - 2).min(len);
    for v in &mut dst_k[..k_warmup] {
        *v = f64::NAN;
    }
    for v in &mut dst_d[..d_warmup] {
        *v = f64::NAN;
    }

    Ok(())
}

pub fn stochf_with_kernel(
    input: &StochfInput,
    kernel: Kernel,
) -> Result<StochfOutput, StochfError> {
    let (high, low, close) = match &input.data {
        StochfData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| StochfError::EmptyInputData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| StochfError::EmptyInputData)?;
            let close = candles
                .select_candle_field("close")
                .map_err(|_| StochfError::EmptyInputData)?;
            (high, low, close)
        }
        StochfData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(StochfError::EmptyInputData);
    }
    let len = high.len();
    if low.len() != len || close.len() != len {
        return Err(StochfError::EmptyInputData);
    }

    let fastk_period = input.get_fastk_period();
    let fastd_period = input.get_fastd_period();
    let matype = input.get_fastd_matype();

    if fastk_period == 0 || fastd_period == 0 || fastk_period > len || fastd_period > len {
        return Err(StochfError::InvalidPeriod {
            fastk: fastk_period,
            fastd: fastd_period,
            data_len: len,
        });
    }
    let first_valid_idx = (0..len)
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(StochfError::AllValuesNaN)?;
    if (len - first_valid_idx) < fastk_period {
        return Err(StochfError::NotEnoughValidData {
            needed: fastk_period,
            valid: len - first_valid_idx,
        });
    }

    let k_warmup = first_valid_idx + fastk_period - 1;
    let d_warmup = first_valid_idx + fastk_period + fastd_period - 2;
    let mut k_vals = alloc_with_nan_prefix(len, k_warmup.min(len));
    let mut d_vals = alloc_with_nan_prefix(len, d_warmup.min(len));

    
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => stochf_scalar(
                high,
                low,
                close,
                fastk_period,
                fastd_period,
                matype,
                first_valid_idx,
                &mut k_vals,
                &mut d_vals,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => stochf_avx2(
                high,
                low,
                close,
                fastk_period,
                fastd_period,
                matype,
                first_valid_idx,
                &mut k_vals,
                &mut d_vals,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => stochf_avx512(
                high,
                low,
                close,
                fastk_period,
                fastd_period,
                matype,
                first_valid_idx,
                &mut k_vals,
                &mut d_vals,
            ),
            _ => unreachable!(),
        }
    }

    Ok(StochfOutput {
        k: k_vals,
        d: d_vals,
    })
}

#[inline]
pub unsafe fn stochf_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    fastd_period: usize,
    matype: usize,
    first_valid_idx: usize,
    k_vals: &mut [f64],
    d_vals: &mut [f64],
) {
    debug_assert_eq!(high.len(), low.len());
    debug_assert_eq!(high.len(), close.len());
    debug_assert_eq!(high.len(), k_vals.len());
    debug_assert_eq!(k_vals.len(), d_vals.len());

    let len = high.len();
    if len == 0 {
        return;
    }

    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();

    let k_start = first_valid_idx + fastk_period - 1;

    
    if fastk_period <= 16 {
        let use_sma_d = matype == 0;
        let mut d_sum: f64 = 0.0;
        let mut d_cnt: usize = 0;

        let mut i = k_start;
        while i < len {
            let start = i + 1 - fastk_period;
            let end = i + 1;

            let mut hh = f64::NEG_INFINITY;
            let mut ll = f64::INFINITY;

            let mut j = start;
            let unroll_end = end - ((end - j) & 3);
            while j < unroll_end {
                let h0 = *hp.add(j);
                let l0 = *lp.add(j);
                if h0 > hh {
                    hh = h0;
                }
                if l0 < ll {
                    ll = l0;
                }

                let h1 = *hp.add(j + 1);
                let l1 = *lp.add(j + 1);
                if h1 > hh {
                    hh = h1;
                }
                if l1 < ll {
                    ll = l1;
                }

                let h2 = *hp.add(j + 2);
                let l2 = *lp.add(j + 2);
                if h2 > hh {
                    hh = h2;
                }
                if l2 < ll {
                    ll = l2;
                }

                let h3 = *hp.add(j + 3);
                let l3 = *lp.add(j + 3);
                if h3 > hh {
                    hh = h3;
                }
                if l3 < ll {
                    ll = l3;
                }

                j += 4;
            }
            while j < end {
                let h = *hp.add(j);
                let l = *lp.add(j);
                if h > hh {
                    hh = h;
                }
                if l < ll {
                    ll = l;
                }
                j += 1;
            }

            let c = *cp.add(i);
            let denom = hh - ll;
            let kv = if denom == 0.0 {
                if c == hh {
                    100.0
                } else {
                    0.0
                }
            } else {
                let inv = 100.0 / denom;
                c.mul_add(inv, (-ll) * inv)
            };
            *k_vals.get_unchecked_mut(i) = kv;

            if use_sma_d {
                if kv.is_nan() {
                    *d_vals.get_unchecked_mut(i) = f64::NAN;
                } else if d_cnt < fastd_period {
                    d_sum += kv;
                    d_cnt += 1;
                    if d_cnt == fastd_period {
                        *d_vals.get_unchecked_mut(i) = d_sum / (fastd_period as f64);
                    } else {
                        *d_vals.get_unchecked_mut(i) = f64::NAN;
                    }
                } else {
                    d_sum += kv - *k_vals.get_unchecked(i - fastd_period);
                    *d_vals.get_unchecked_mut(i) = d_sum / (fastd_period as f64);
                }
            }

            i += 1;
        }

        if matype != 0 {
            d_vals.fill(f64::NAN);
        }
        return;
    }

    
    let cap = fastk_period;
    let mut qh = vec![0usize; cap];
    let mut ql = vec![0usize; cap];
    let mut qh_head = 0usize;
    let mut qh_tail = 0usize;
    let mut ql_head = 0usize;
    let mut ql_tail = 0usize;

    let use_sma_d = matype == 0;
    let mut d_sum: f64 = 0.0;
    let mut d_cnt: usize = 0;

    let mut i = first_valid_idx;
    while i < len {
        if i + 1 >= fastk_period {
            let win_start = i + 1 - fastk_period;
            while qh_head != qh_tail {
                let idx = *qh.get_unchecked(qh_head);
                if idx >= win_start {
                    break;
                }
                qh_head += 1;
                if qh_head == cap {
                    qh_head = 0;
                }
            }
            while ql_head != ql_tail {
                let idx = *ql.get_unchecked(ql_head);
                if idx >= win_start {
                    break;
                }
                ql_head += 1;
                if ql_head == cap {
                    ql_head = 0;
                }
            }
        }

        let h_i = *hp.add(i);
        if h_i == h_i {
            while qh_head != qh_tail {
                let back = if qh_tail == 0 { cap - 1 } else { qh_tail - 1 };
                let back_idx = *qh.get_unchecked(back);
                if *hp.add(back_idx) <= h_i {
                    qh_tail = back;
                } else {
                    break;
                }
            }
            *qh.get_unchecked_mut(qh_tail) = i;
            qh_tail += 1;
            if qh_tail == cap {
                qh_tail = 0;
            }
        }

        let l_i = *lp.add(i);
        if l_i == l_i {
            while ql_head != ql_tail {
                let back = if ql_tail == 0 { cap - 1 } else { ql_tail - 1 };
                let back_idx = *ql.get_unchecked(back);
                if *lp.add(back_idx) >= l_i {
                    ql_tail = back;
                } else {
                    break;
                }
            }
            *ql.get_unchecked_mut(ql_tail) = i;
            ql_tail += 1;
            if ql_tail == cap {
                ql_tail = 0;
            }
        }

        if i >= k_start {
            let hh = if qh_head != qh_tail {
                *hp.add(*qh.get_unchecked(qh_head))
            } else {
                f64::NEG_INFINITY
            };
            let ll = if ql_head != ql_tail {
                *lp.add(*ql.get_unchecked(ql_head))
            } else {
                f64::INFINITY
            };
            let c = *cp.add(i);
            let denom = hh - ll;
            let kv = if denom == 0.0 {
                if c == hh {
                    100.0
                } else {
                    0.0
                }
            } else {
                let inv = 100.0 / denom;
                c.mul_add(inv, (-ll) * inv)
            };
            *k_vals.get_unchecked_mut(i) = kv;

            if use_sma_d {
                if kv.is_nan() {
                    *d_vals.get_unchecked_mut(i) = f64::NAN;
                } else if d_cnt < fastd_period {
                    d_sum += kv;
                    d_cnt += 1;
                    if d_cnt == fastd_period {
                        *d_vals.get_unchecked_mut(i) = d_sum / (fastd_period as f64);
                    } else {
                        *d_vals.get_unchecked_mut(i) = f64::NAN;
                    }
                } else {
                    d_sum += kv - *k_vals.get_unchecked(i - fastd_period);
                    *d_vals.get_unchecked_mut(i) = d_sum / (fastd_period as f64);
                }
            }
        }

        i += 1;
    }

    if !use_sma_d {
        d_vals.fill(f64::NAN);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    fastd_period: usize,
    matype: usize,
    first_valid_idx: usize,
    k_vals: &mut [f64],
    d_vals: &mut [f64],
) {
    
    if fastk_period <= 32 {
        let len = high.len();
        let start_i = first_valid_idx + fastk_period - 1;
        let neg_inf = _mm256_set1_pd(f64::NEG_INFINITY);
        let pos_inf = _mm256_set1_pd(f64::INFINITY);

        let use_sma_d = matype == 0;
        let mut d_sum = 0.0f64;
        let mut d_cnt: usize = 0;

        for i in start_i..len {
            let start = i + 1 - fastk_period;
            let end = i + 1;

            let mut vmax = neg_inf;
            let mut vmin = pos_inf;
            let mut j = start;
            while j + 4 <= end {
                let vh = _mm256_loadu_pd(high.as_ptr().add(j));
                let vl = _mm256_loadu_pd(low.as_ptr().add(j));

                let mask_h = _mm256_cmp_pd(vh, vh, _CMP_ORD_Q);
                let mask_l = _mm256_cmp_pd(vl, vl, _CMP_ORD_Q);
                let vh_nnan = _mm256_blendv_pd(neg_inf, vh, mask_h);
                let vl_nnan = _mm256_blendv_pd(pos_inf, vl, mask_l);

                vmax = _mm256_max_pd(vmax, vh_nnan);
                vmin = _mm256_min_pd(vmin, vl_nnan);
                j += 4;
            }

            
            let vmax_lo = _mm256_castpd256_pd128(vmax);
            let vmax_hi = _mm256_extractf128_pd(vmax, 1);
            let vmax_128 = _mm_max_pd(vmax_lo, vmax_hi);
            let vmax_hi64 = _mm_unpackhi_pd(vmax_128, vmax_128);
            let mut hh = f64::max(_mm_cvtsd_f64(vmax_128), _mm_cvtsd_f64(vmax_hi64));

            let vmin_lo = _mm256_castpd256_pd128(vmin);
            let vmin_hi = _mm256_extractf128_pd(vmin, 1);
            let vmin_128 = _mm_min_pd(vmin_lo, vmin_hi);
            let vmin_hi64 = _mm_unpackhi_pd(vmin_128, vmin_128);
            let mut ll = f64::min(_mm_cvtsd_f64(vmin_128), _mm_cvtsd_f64(vmin_hi64));

            while j < end {
                let h = *high.get_unchecked(j);
                let l = *low.get_unchecked(j);
                if h == h && h > hh {
                    hh = h;
                }
                if l == l && l < ll {
                    ll = l;
                }
                j += 1;
            }

            
            let c = *close.get_unchecked(i);
            let denom = hh - ll;
            let kv = if denom == 0.0 {
                if c == hh {
                    100.0
                } else {
                    0.0
                }
            } else {
                let inv = 100.0 / denom;
                c.mul_add(inv, (-ll) * inv)
            };
            *k_vals.get_unchecked_mut(i) = kv;

            
            if use_sma_d {
                if kv.is_nan() {
                    *d_vals.get_unchecked_mut(i) = f64::NAN;
                } else if d_cnt < fastd_period {
                    d_sum += kv;
                    d_cnt += 1;
                    if d_cnt == fastd_period {
                        *d_vals.get_unchecked_mut(i) = d_sum / (fastd_period as f64);
                    } else {
                        *d_vals.get_unchecked_mut(i) = f64::NAN;
                    }
                } else {
                    d_sum += kv - *k_vals.get_unchecked(i - fastd_period);
                    *d_vals.get_unchecked_mut(i) = d_sum / (fastd_period as f64);
                }
            }
        }

        if !use_sma_d {
            d_vals.fill(f64::NAN);
        }
    } else {
        stochf_scalar(
            high,
            low,
            close,
            fastk_period,
            fastd_period,
            matype,
            first_valid_idx,
            k_vals,
            d_vals,
        );
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    fastd_period: usize,
    matype: usize,
    first_valid_idx: usize,
    k_vals: &mut [f64],
    d_vals: &mut [f64],
) {
    if fastk_period <= 32 {
        stochf_avx512_short(
            high,
            low,
            close,
            fastk_period,
            fastd_period,
            matype,
            first_valid_idx,
            k_vals,
            d_vals,
        );
    } else {
        stochf_avx512_long(
            high,
            low,
            close,
            fastk_period,
            fastd_period,
            matype,
            first_valid_idx,
            k_vals,
            d_vals,
        );
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    fastd_period: usize,
    matype: usize,
    first_valid_idx: usize,
    k_vals: &mut [f64],
    d_vals: &mut [f64],
) {
    stochf_scalar(
        high,
        low,
        close,
        fastk_period,
        fastd_period,
        matype,
        first_valid_idx,
        k_vals,
        d_vals,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    fastd_period: usize,
    matype: usize,
    first_valid_idx: usize,
    k_vals: &mut [f64],
    d_vals: &mut [f64],
) {
    stochf_scalar(
        high,
        low,
        close,
        fastk_period,
        fastd_period,
        matype,
        first_valid_idx,
        k_vals,
        d_vals,
    );
}

/// Decision: Streaming uses O(1) amortized deques for %K and a running-sum ring for %D (SMA);
/// matches offline scalar numerics post-warmup.
#[derive(Debug, Clone)]
pub struct StochfStream {
    
    fastk_period: usize,
    fastd_period: usize,
    fastd_matype: usize,

    
    
    qh_idx: Vec<usize>,
    qh_val: Vec<f64>,
    qh_head: usize,
    qh_tail: usize,

    
    ql_idx: Vec<usize>,
    ql_val: Vec<f64>,
    ql_head: usize,
    ql_tail: usize,

    
    
    
    
    cap_k: usize,

    
    qh_full: bool,
    ql_full: bool,

    
    t: usize,

    
    k_ring: Vec<f64>,
    k_head: usize,
    k_count: usize,
    d_sma_sum: f64,
}

impl StochfStream {
    pub fn try_new(params: StochfParams) -> Result<Self, StochfError> {
        let fastk_period = params.fastk_period.unwrap_or(5);
        let fastd_period = params.fastd_period.unwrap_or(3);
        let fastd_matype = params.fastd_matype.unwrap_or(0);

        if fastk_period == 0 || fastd_period == 0 {
            return Err(StochfError::InvalidPeriod {
                fastk: fastk_period,
                fastd: fastd_period,
                data_len: 0,
            });
        }

        
        
        let cap_k = fastk_period + 1;

        Ok(Self {
            fastk_period,
            fastd_period,
            fastd_matype,

            qh_idx: vec![0; cap_k],
            qh_val: vec![0.0; cap_k],
            qh_head: 0,
            qh_tail: 0,
            qh_full: false,

            ql_idx: vec![0; cap_k],
            ql_val: vec![0.0; cap_k],
            ql_head: 0,
            ql_tail: 0,
            ql_full: false,

            cap_k,

            t: 0,

            k_ring: vec![0.0; fastd_period],
            k_head: 0,
            k_count: 0,
            d_sma_sum: 0.0,
        })
    }

    #[inline(always)]
    fn inc(idx: &mut usize, cap: usize) {
        *idx += 1;
        if *idx == cap {
            *idx = 0;
        }
    }

    #[inline(always)]
    fn dec(idx: &mut usize, cap: usize) {
        if *idx == 0 {
            *idx = cap - 1;
        } else {
            *idx -= 1;
        }
    }

    #[inline(always)]
    fn qh_expire(&mut self, win_start: usize) {
        while (self.qh_head != self.qh_tail || self.qh_full)
            && self.qh_idx[self.qh_head] < win_start
        {
            Self::inc(&mut self.qh_head, self.cap_k);
            
            self.qh_full = false;
        }
    }
    #[inline(always)]
    fn ql_expire(&mut self, win_start: usize) {
        while (self.ql_head != self.ql_tail || self.ql_full)
            && self.ql_idx[self.ql_head] < win_start
        {
            Self::inc(&mut self.ql_head, self.cap_k);
            self.ql_full = false;
        }
    }

    #[inline(always)]
    fn qh_push(&mut self, idx: usize, val: f64) {
        
        while self.qh_head != self.qh_tail || self.qh_full {
            let mut back = self.qh_tail;
            Self::dec(&mut back, self.cap_k);
            if self.qh_val[back] <= val {
                self.qh_tail = back;
                
                self.qh_full = false;
            } else {
                break;
            }
        }
        self.qh_idx[self.qh_tail] = idx;
        self.qh_val[self.qh_tail] = val;
        Self::inc(&mut self.qh_tail, self.cap_k);
        
        if self.qh_tail == self.qh_head {
            self.qh_full = true;
        }
    }

    #[inline(always)]
    fn ql_push(&mut self, idx: usize, val: f64) {
        
        while self.ql_head != self.ql_tail || self.ql_full {
            let mut back = self.ql_tail;
            Self::dec(&mut back, self.cap_k);
            if self.ql_val[back] >= val {
                self.ql_tail = back;
                self.ql_full = false;
            } else {
                break;
            }
        }
        self.ql_idx[self.ql_tail] = idx;
        self.ql_val[self.ql_tail] = val;
        Self::inc(&mut self.ql_tail, self.cap_k);
        if self.ql_tail == self.ql_head {
            self.ql_full = true;
        }
    }

    /// O(1) amortized per tick for both %K and %D (SMA).
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        let i = self.t;
        
        self.t = self.t.wrapping_add(1);

        
        if high == high {
            self.qh_push(i, high);
        }
        if low == low {
            self.ql_push(i, low);
        }

        
        let have_k_window = (i + 1) >= self.fastk_period;
        if have_k_window {
            let win_start = i + 1 - self.fastk_period;
            self.qh_expire(win_start);
            self.ql_expire(win_start);
        } else {
            
            return None;
        }

        
        let hh = if self.qh_head != self.qh_tail || self.qh_full {
            self.qh_val[self.qh_head]
        } else {
            f64::NEG_INFINITY
        };
        let ll = if self.ql_head != self.ql_tail || self.ql_full {
            self.ql_val[self.ql_head]
        } else {
            f64::INFINITY
        };

        

        
        let denom = hh - ll;
        let k = if denom == 0.0 {
            if close == hh {
                100.0
            } else {
                0.0
            }
        } else {
            
            let scale = 100.0 / denom;
            close.mul_add(scale, (-ll) * scale)
        };

        
        let d = if self.fastd_matype != 0 {
            f64::NAN 
        } else if self.k_count < self.fastd_period {
            
            self.k_ring[self.k_head] = k;
            self.d_sma_sum += k;
            self.k_count += 1;
            StochfStream::inc(&mut self.k_head, self.fastd_period);

            if self.k_count == self.fastd_period {
                self.d_sma_sum / (self.fastd_period as f64)
            } else {
                f64::NAN
            }
        } else {
            
            let old = self.k_ring[self.k_head];
            self.k_ring[self.k_head] = k;
            StochfStream::inc(&mut self.k_head, self.fastd_period);

            self.d_sma_sum += k - old;
            self.d_sma_sum / (self.fastd_period as f64)
        };

        Some((k, d))
    }
}

#[derive(Clone, Debug)]
pub struct StochfBatchRange {
    pub fastk_period: (usize, usize, usize),
    pub fastd_period: (usize, usize, usize),
}

impl Default for StochfBatchRange {
    fn default() -> Self {
        Self {
            fastk_period: (5, 254, 1),
            fastd_period: (3, 3, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StochfBatchBuilder {
    range: StochfBatchRange,
    kernel: Kernel,
}

impl StochfBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn fastk_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fastk_period = (start, end, step);
        self
    }
    pub fn fastk_static(mut self, p: usize) -> Self {
        self.range.fastk_period = (p, p, 0);
        self
    }
    pub fn fastd_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fastd_period = (start, end, step);
        self
    }
    pub fn fastd_static(mut self, p: usize) -> Self {
        self.range.fastd_period = (p, p, 0);
        self
    }
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<StochfBatchOutput, StochfError> {
        stochf_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        k: Kernel,
    ) -> Result<StochfBatchOutput, StochfError> {
        StochfBatchBuilder::new()
            .kernel(k)
            .apply_slices(high, low, close)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<StochfBatchOutput, StochfError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        self.apply_slices(high, low, close)
    }
    pub fn with_default_candles(c: &Candles) -> Result<StochfBatchOutput, StochfError> {
        StochfBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

pub fn stochf_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochfBatchRange,
    k: Kernel,
) -> Result<StochfBatchOutput, StochfError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(StochfError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    stochf_batch_par_slice(high, low, close, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct StochfBatchOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    pub combos: Vec<StochfParams>,
    pub rows: usize,
    pub cols: usize,
}
impl StochfBatchOutput {
    pub fn row_for_params(&self, p: &StochfParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.fastk_period.unwrap_or(5) == p.fastk_period.unwrap_or(5)
                && c.fastd_period.unwrap_or(3) == p.fastd_period.unwrap_or(3)
                && c.fastd_matype.unwrap_or(0) == p.fastd_matype.unwrap_or(0)
        })
    }
    pub fn k_for(&self, p: &StochfParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.k[start..start + self.cols]
        })
    }
    pub fn d_for(&self, p: &StochfParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.d[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &StochfBatchRange) -> Vec<StochfParams> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, StochfError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let mut v = Vec::new();
            let st = step.max(1);
            let mut x = start;
            while x <= end {
                v.push(x);
                x = match x.checked_add(st) {
                    Some(next) => next,
                    None => break,
                };
            }
            if v.is_empty() {
                return Err(StochfError::InvalidRange { start, end, step });
            }
            return Ok(v);
        }
        
        let mut v = Vec::new();
        let st = step.max(1) as isize;
        let mut x = start as isize;
        let end_i = end as isize;
        while x >= end_i {
            v.push(x as usize);
            x -= st;
        }
        if v.is_empty() {
            return Err(StochfError::InvalidRange { start, end, step });
        }
        Ok(v)
    }
    let fastk = axis_usize(r.fastk_period).unwrap_or_else(|_| Vec::new());
    let fastd = axis_usize(r.fastd_period).unwrap_or_else(|_| Vec::new());
    let mut out = Vec::with_capacity(fastk.len().saturating_mul(fastd.len()));
    for &k in &fastk {
        for &d in &fastd {
            out.push(StochfParams {
                fastk_period: Some(k),
                fastd_period: Some(d),
                fastd_matype: Some(0),
            });
        }
    }
    out
}

#[inline(always)]
pub fn stochf_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochfBatchRange,
    kern: Kernel,
) -> Result<StochfBatchOutput, StochfError> {
    stochf_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn stochf_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochfBatchRange,
    kern: Kernel,
) -> Result<StochfBatchOutput, StochfError> {
    stochf_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
pub fn stochf_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochfBatchRange,
    kern: Kernel,
    parallel: bool,
    k_out: &mut [f64],
    d_out: &mut [f64],
) -> Result<Vec<StochfParams>, StochfError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(StochfError::InvalidRange {
            start: sweep.fastk_period.0,
            end: sweep.fastk_period.1,
            step: sweep.fastk_period.2,
        });
    }
    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(StochfError::AllValuesNaN)?;
    let max_k = combos
        .iter()
        .map(|c| c.fastk_period.unwrap())
        .max()
        .unwrap();
    if high.len() - first < max_k {
        return Err(StochfError::NotEnoughValidData {
            needed: max_k,
            valid: high.len() - first,
        });
    }
    let rows = combos.len();
    let cols = high.len();

    
    let expected_size = rows
        .checked_mul(cols)
        .ok_or(StochfError::InvalidRange {
            start: sweep.fastk_period.0,
            end: sweep.fastk_period.1,
            step: sweep.fastk_period.2,
        })?;
    if k_out.len() != expected_size || d_out.len() != expected_size {
        return Err(StochfError::OutputLengthMismatch {
            expected: expected_size,
            k_got: k_out.len(),
            d_got: d_out.len(),
        });
    }

    
    for (row, combo) in combos.iter().enumerate() {
        let k_warmup = (first + combo.fastk_period.unwrap() - 1).min(cols);
        let d_warmup =
            (first + combo.fastk_period.unwrap() + combo.fastd_period.unwrap() - 2).min(cols);
        let row_start = row * cols;

        
        for i in 0..k_warmup {
            k_out[row_start + i] = f64::NAN;
        }

        
        for i in 0..d_warmup {
            d_out[row_start + i] = f64::NAN;
        }
    }

    let do_row = |row: usize, kout: &mut [f64], dout: &mut [f64]| unsafe {
        let fastk_period = combos[row].fastk_period.unwrap();
        let fastd_period = combos[row].fastd_period.unwrap();
        let matype = combos[row].fastd_matype.unwrap();
        match kern {
            Kernel::Scalar => stochf_row_scalar(
                high,
                low,
                close,
                first,
                fastk_period,
                fastd_period,
                matype,
                kout,
                dout,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => stochf_row_avx2(
                high,
                low,
                close,
                first,
                fastk_period,
                fastd_period,
                matype,
                kout,
                dout,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => stochf_row_avx512(
                high,
                low,
                close,
                first,
                fastk_period,
                fastd_period,
                matype,
                kout,
                dout,
            ),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            k_out
                .par_chunks_mut(cols)
                .zip(d_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (k, d))| do_row(row, k, d));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (k, d)) in k_out
                .chunks_mut(cols)
                .zip(d_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, k, d);
            }
        }
    } else {
        for (row, (k, d)) in k_out
            .chunks_mut(cols)
            .zip(d_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, k, d);
        }
    }

    Ok(combos)
}

#[inline(always)]
fn stochf_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &StochfBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<StochfBatchOutput, StochfError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(StochfError::InvalidRange {
            start: sweep.fastk_period.0,
            end: sweep.fastk_period.1,
            step: sweep.fastk_period.2,
        });
    }
    let first = (0..high.len())
        .find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
        .ok_or(StochfError::AllValuesNaN)?;
    let max_k = combos
        .iter()
        .map(|c| c.fastk_period.unwrap())
        .max()
        .unwrap();
    if high.len() - first < max_k {
        return Err(StochfError::NotEnoughValidData {
            needed: max_k,
            valid: high.len() - first,
        });
    }
    let rows = combos.len();
    let cols = high.len();

    
    let _total = rows
        .checked_mul(cols)
        .ok_or(StochfError::InvalidRange {
            start: sweep.fastk_period.0,
            end: sweep.fastk_period.1,
            step: sweep.fastk_period.2,
        })?;

    
    let mut k_buf = make_uninit_matrix(rows, cols);
    let mut d_buf = make_uninit_matrix(rows, cols);

    
    let k_warmups: Vec<usize> = combos
        .iter()
        .map(|c| (first + c.fastk_period.unwrap() - 1).min(cols))
        .collect();
    let d_warmups: Vec<usize> = combos
        .iter()
        .map(|c| (first + c.fastk_period.unwrap() + c.fastd_period.unwrap() - 2).min(cols))
        .collect();

    
    init_matrix_prefixes(&mut k_buf, cols, &k_warmups);
    init_matrix_prefixes(&mut d_buf, cols, &d_warmups);

    
    let k_buf_len = k_buf.len();
    let d_buf_len = d_buf.len();
    let k_buf_cap = k_buf.capacity();
    let d_buf_cap = d_buf.capacity();
    let k_ptr = k_buf.as_mut_ptr();
    let d_ptr = d_buf.as_mut_ptr();
    std::mem::forget(k_buf);
    std::mem::forget(d_buf);
    let k_out = unsafe { std::slice::from_raw_parts_mut(k_ptr as *mut f64, k_buf_len) };
    let d_out = unsafe { std::slice::from_raw_parts_mut(d_ptr as *mut f64, d_buf_len) };

    let do_row = |row: usize, kout: &mut [f64], dout: &mut [f64]| unsafe {
        let fastk_period = combos[row].fastk_period.unwrap();
        let fastd_period = combos[row].fastd_period.unwrap();
        let matype = combos[row].fastd_matype.unwrap();
        match kern {
            Kernel::Scalar => stochf_row_scalar(
                high,
                low,
                close,
                first,
                fastk_period,
                fastd_period,
                matype,
                kout,
                dout,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => stochf_row_avx2(
                high,
                low,
                close,
                first,
                fastk_period,
                fastd_period,
                matype,
                kout,
                dout,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => stochf_row_avx512(
                high,
                low,
                close,
                first,
                fastk_period,
                fastd_period,
                matype,
                kout,
                dout,
            ),
            _ => unreachable!(),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            k_out
                .par_chunks_mut(cols)
                .zip(d_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(row, (k, d))| do_row(row, k, d));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (k, d)) in k_out
                .chunks_mut(cols)
                .zip(d_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, k, d);
            }
        }
    } else {
        for (row, (k, d)) in k_out
            .chunks_mut(cols)
            .zip(d_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, k, d);
        }
    }

    
    let k_vec = unsafe { Vec::from_raw_parts(k_ptr as *mut f64, k_buf_len, k_buf_cap) };
    let d_vec = unsafe { Vec::from_raw_parts(d_ptr as *mut f64, d_buf_len, d_buf_cap) };

    Ok(StochfBatchOutput {
        k: k_vec,
        d: d_vec,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn stochf_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    fastk_period: usize,
    fastd_period: usize,
    matype: usize,
    k_out: &mut [f64],
    d_out: &mut [f64],
) {
    let len = high.len();

    let hp = high.as_ptr();
    let lp = low.as_ptr();
    let cp = close.as_ptr();

    let k_start = first + fastk_period - 1;

    if fastk_period <= 16 {
        let use_sma_d = matype == 0;
        let mut d_sum: f64 = 0.0;
        let mut d_cnt: usize = 0;

        let mut i = k_start;
        while i < len {
            let start = i + 1 - fastk_period;
            let end = i + 1;

            let mut hh = f64::NEG_INFINITY;
            let mut ll = f64::INFINITY;

            let mut j = start;
            let unroll_end = end - ((end - j) & 3);
            while j < unroll_end {
                let h0 = *hp.add(j);
                let l0 = *lp.add(j);
                if h0 > hh {
                    hh = h0;
                }
                if l0 < ll {
                    ll = l0;
                }

                let h1 = *hp.add(j + 1);
                let l1 = *lp.add(j + 1);
                if h1 > hh {
                    hh = h1;
                }
                if l1 < ll {
                    ll = l1;
                }

                let h2 = *hp.add(j + 2);
                let l2 = *lp.add(j + 2);
                if h2 > hh {
                    hh = h2;
                }
                if l2 < ll {
                    ll = l2;
                }

                let h3 = *hp.add(j + 3);
                let l3 = *lp.add(j + 3);
                if h3 > hh {
                    hh = h3;
                }
                if l3 < ll {
                    ll = l3;
                }

                j += 4;
            }
            while j < end {
                let h = *hp.add(j);
                let l = *lp.add(j);
                if h > hh {
                    hh = h;
                }
                if l < ll {
                    ll = l;
                }
                j += 1;
            }

            let c = *cp.add(i);
            let denom = hh - ll;
            let kv = if denom == 0.0 {
                if c == hh {
                    100.0
                } else {
                    0.0
                }
            } else {
                let inv = 100.0 / denom;
                c.mul_add(inv, (-ll) * inv)
            };
            *k_out.get_unchecked_mut(i) = kv;

            if use_sma_d {
                if kv.is_nan() {
                    *d_out.get_unchecked_mut(i) = f64::NAN;
                } else if d_cnt < fastd_period {
                    d_sum += kv;
                    d_cnt += 1;
                    if d_cnt == fastd_period {
                        *d_out.get_unchecked_mut(i) = d_sum / (fastd_period as f64);
                    } else {
                        *d_out.get_unchecked_mut(i) = f64::NAN;
                    }
                } else {
                    d_sum += kv - *k_out.get_unchecked(i - fastd_period);
                    *d_out.get_unchecked_mut(i) = d_sum / (fastd_period as f64);
                }
            }

            i += 1;
        }

        if matype != 0 {
            d_out.fill(f64::NAN);
        }
        return;
    }

    
    let cap = fastk_period;
    let mut qh = vec![0usize; cap];
    let mut ql = vec![0usize; cap];
    let mut qh_head = 0usize;
    let mut qh_tail = 0usize;
    let mut ql_head = 0usize;
    let mut ql_tail = 0usize;

    let use_sma_d = matype == 0;
    let mut d_sum: f64 = 0.0;
    let mut d_cnt: usize = 0;

    let mut i = first;
    while i < len {
        if i + 1 >= fastk_period {
            let win_start = i + 1 - fastk_period;
            while qh_head != qh_tail {
                let idx = *qh.get_unchecked(qh_head);
                if idx >= win_start {
                    break;
                }
                qh_head += 1;
                if qh_head == cap {
                    qh_head = 0;
                }
            }
            while ql_head != ql_tail {
                let idx = *ql.get_unchecked(ql_head);
                if idx >= win_start {
                    break;
                }
                ql_head += 1;
                if ql_head == cap {
                    ql_head = 0;
                }
            }
        }

        let h_i = *hp.add(i);
        if h_i == h_i {
            while qh_head != qh_tail {
                let back = if qh_tail == 0 { cap - 1 } else { qh_tail - 1 };
                let back_idx = *qh.get_unchecked(back);
                if *hp.add(back_idx) <= h_i {
                    qh_tail = back;
                } else {
                    break;
                }
            }
            *qh.get_unchecked_mut(qh_tail) = i;
            qh_tail += 1;
            if qh_tail == cap {
                qh_tail = 0;
            }
        }

        let l_i = *lp.add(i);
        if l_i == l_i {
            while ql_head != ql_tail {
                let back = if ql_tail == 0 { cap - 1 } else { ql_tail - 1 };
                let back_idx = *ql.get_unchecked(back);
                if *lp.add(back_idx) >= l_i {
                    ql_tail = back;
                } else {
                    break;
                }
            }
            *ql.get_unchecked_mut(ql_tail) = i;
            ql_tail += 1;
            if ql_tail == cap {
                ql_tail = 0;
            }
        }

        if i >= k_start {
            let hh = if qh_head != qh_tail {
                *hp.add(*qh.get_unchecked(qh_head))
            } else {
                f64::NEG_INFINITY
            };
            let ll = if ql_head != ql_tail {
                *lp.add(*ql.get_unchecked(ql_head))
            } else {
                f64::INFINITY
            };
            let c = *cp.add(i);
            let denom = hh - ll;
            let kv = if denom == 0.0 {
                if c == hh {
                    100.0
                } else {
                    0.0
                }
            } else {
                let inv = 100.0 / denom;
                c.mul_add(inv, (-ll) * inv)
            };
            *k_out.get_unchecked_mut(i) = kv;

            if use_sma_d {
                if kv.is_nan() {
                    *d_out.get_unchecked_mut(i) = f64::NAN;
                } else if d_cnt < fastd_period {
                    d_sum += kv;
                    d_cnt += 1;
                    if d_cnt == fastd_period {
                        *d_out.get_unchecked_mut(i) = d_sum / (fastd_period as f64);
                    } else {
                        *d_out.get_unchecked_mut(i) = f64::NAN;
                    }
                } else {
                    d_sum += kv - *k_out.get_unchecked(i - fastd_period);
                    *d_out.get_unchecked_mut(i) = d_sum / (fastd_period as f64);
                }
            }
        }

        i += 1;
    }

    if !use_sma_d {
        d_out.fill(f64::NAN);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
unsafe fn stochf_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    fastk_period: usize,
    fastd_period: usize,
    matype: usize,
    k_out: &mut [f64],
    d_out: &mut [f64],
) {
    stochf_row_scalar(
        high,
        low,
        close,
        first,
        fastk_period,
        fastd_period,
        matype,
        k_out,
        d_out,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    fastk_period: usize,
    fastd_period: usize,
    matype: usize,
    k_out: &mut [f64],
    d_out: &mut [f64],
) {
    if fastk_period <= 32 {
        stochf_row_avx512_short(
            high,
            low,
            close,
            first,
            fastk_period,
            fastd_period,
            matype,
            k_out,
            d_out,
        );
    } else {
        stochf_row_avx512_long(
            high,
            low,
            close,
            first,
            fastk_period,
            fastd_period,
            matype,
            k_out,
            d_out,
        );
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    fastk_period: usize,
    fastd_period: usize,
    matype: usize,
    k_out: &mut [f64],
    d_out: &mut [f64],
) {
    stochf_row_scalar(
        high,
        low,
        close,
        first,
        fastk_period,
        fastd_period,
        matype,
        k_out,
        d_out,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn stochf_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    first: usize,
    fastk_period: usize,
    fastd_period: usize,
    matype: usize,
    k_out: &mut [f64],
    d_out: &mut [f64],
) {
    stochf_row_scalar(
        high,
        low,
        close,
        first,
        fastk_period,
        fastd_period,
        matype,
        k_out,
        d_out,
    );
}

#[cfg(feature = "python")]
#[pyfunction(name = "stochf")]
#[pyo3(signature = (high, low, close, fastk_period=None, fastd_period=None, fastd_matype=None, kernel=None))]
pub fn stochf_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    fastk_period: Option<usize>,
    fastd_period: Option<usize>,
    fastd_matype: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    // Validate lengths match
    if high_slice.len() != low_slice.len() || high_slice.len() != close_slice.len() {
        return Err(PyValueError::new_err(
            "Input arrays must have the same length",
        ));
    }

    let params = StochfParams {
        fastk_period,
        fastd_period,
        fastd_matype,
    };
    let input = StochfInput::from_slices(high_slice, low_slice, close_slice, params);

    let (k_vec, d_vec) = py
        .allow_threads(|| stochf_with_kernel(&input, kern).map(|o| (o.k, o.d)))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((k_vec.into_pyarray(py), d_vec.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "StochfStream")]
pub struct StochfStreamPy {
    stream: StochfStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl StochfStreamPy {
    #[new]
    fn new(fastk_period: usize, fastd_period: usize, fastd_matype: usize) -> PyResult<Self> {
        let params = StochfParams {
            fastk_period: Some(fastk_period),
            fastd_period: Some(fastd_period),
            fastd_matype: Some(fastd_matype),
        };
        let stream =
            StochfStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(StochfStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        self.stream.update(high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "stochf_batch")]
#[pyo3(signature = (high, low, close, fastk_range, fastd_range, kernel=None))]
pub fn stochf_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    close: numpy::PyReadonlyArray1<'py, f64>,
    fastk_range: (usize, usize, usize),
    fastd_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    // Validate lengths match
    if high_slice.len() != low_slice.len() || high_slice.len() != close_slice.len() {
        return Err(PyValueError::new_err(
            "Input arrays must have the same length",
        ));
    }

    let sweep = StochfBatchRange {
        fastk_period: fastk_range,
        fastd_period: fastd_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    if rows == 0 {
        return Err(PyValueError::new_err(
            "stochf: invalid range (empty expansion)",
        ));
    }
    let cols = high_slice.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("stochf: rows*cols overflow"))?;

    // Pre-allocate output arrays for batch
    let k_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let d_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let k_slice = unsafe { k_arr.as_slice_mut()? };
    let d_slice = unsafe { d_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

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
            stochf_batch_inner_into(
                high_slice,
                low_slice,
                close_slice,
                &sweep,
                simd,
                true,
                k_slice,
                d_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("k_values", k_arr.reshape((rows, cols))?)?;
    dict.set_item("d_values", d_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "fastk_periods",
        combos
            .iter()
            .map(|p| p.fastk_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "fastd_periods",
        combos
            .iter()
            .map(|p| p.fastd_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

// ---- CUDA Python bindings (DeviceArrayF32Py handles) ----
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaStochf};
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray1;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::exceptions::PyValueError as PyErrValue;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::PyErr;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "stochf_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, close_f32, fastk_range, fastd_range, device_id=0))]
pub fn stochf_cuda_batch_dev_py(
    py: Python<'_>,
    high_f32: PyReadonlyArray1<'_, f32>,
    low_f32: PyReadonlyArray1<'_, f32>,
    close_f32: PyReadonlyArray1<'_, f32>,
    fastk_range: (usize, usize, usize),
    fastd_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py)> {
    if !cuda_available() {
        return Err(PyErrValue::new_err("CUDA not available"));
    }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    let c = close_f32.as_slice()?;
    if h.len() != l.len() || h.len() != c.len() {
        return Err(PyErrValue::new_err("mismatched input lengths"));
    }
    let sweep = StochfBatchRange {
        fastk_period: fastk_range,
        fastd_period: fastd_range,
    };
    let (pair, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaStochf::new(device_id).map_err(|e| PyErrValue::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        let (pair, _combos) = cuda
            .stochf_batch_dev(h, l, c, &sweep)
            .map_err(|e| PyErrValue::new_err(e.to_string()))?;
        Ok::<_, PyErr>((pair, ctx, dev_id))
    })?;
    Ok((
        DeviceArrayF32Py { inner: pair.a, _ctx: Some(ctx.clone()), device_id: Some(dev_id) },
        DeviceArrayF32Py { inner: pair.b, _ctx: Some(ctx), device_id: Some(dev_id) },
    ))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "stochf_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, close_tm_f32, cols, rows, fastk, fastd, fastd_matype=0, device_id=0))]
pub fn stochf_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: PyReadonlyArray1<'_, f32>,
    low_tm_f32: PyReadonlyArray1<'_, f32>,
    close_tm_f32: PyReadonlyArray1<'_, f32>,
    cols: usize,
    rows: usize,
    fastk: usize,
    fastd: usize,
    fastd_matype: usize,
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, DeviceArrayF32Py)> {
    if !cuda_available() {
        return Err(PyErrValue::new_err("CUDA not available"));
    }
    let htm = high_tm_f32.as_slice()?;
    let ltm = low_tm_f32.as_slice()?;
    let ctm = close_tm_f32.as_slice()?;
    let params = StochfParams {
        fastk_period: Some(fastk),
        fastd_period: Some(fastd),
        fastd_matype: Some(fastd_matype),
    };
    let (k, d, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaStochf::new(device_id).map_err(|e| PyErrValue::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = cuda.device_id();
        let (k, d) = cuda
            .stochf_many_series_one_param_time_major_dev(htm, ltm, ctm, cols, rows, &params)
            .map_err(|e| PyErrValue::new_err(e.to_string()))?;
        Ok::<_, PyErr>((k, d, ctx, dev_id))
    })?;
    Ok((
        DeviceArrayF32Py { inner: k, _ctx: Some(ctx.clone()), device_id: Some(dev_id) },
        DeviceArrayF32Py { inner: d, _ctx: Some(ctx), device_id: Some(dev_id) },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_stochf_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = StochfParams {
            fastk_period: None,
            fastd_period: None,
            fastd_matype: None,
        };
        let input = StochfInput::from_candles(&candles, params);
        let output = stochf_with_kernel(&input, kernel)?;
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
        Ok(())
    }

    fn check_stochf_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = StochfParams {
            fastk_period: Some(5),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        };
        let input = StochfInput::from_candles(&candles, params);
        let output = stochf_with_kernel(&input, kernel)?;
        let expected_k = [
            80.6987399770905,
            40.88471849865952,
            15.507246376811594,
            36.920529801324506,
            32.1880650994575,
        ];
        let expected_d = [
            70.99960994145033,
            61.44725644908976,
            45.696901617520815,
            31.104164892265487,
            28.205280425864817,
        ];
        let k_slice = &output.k[output.k.len() - 5..];
        let d_slice = &output.d[output.d.len() - 5..];
        for i in 0..5 {
            assert!(
                (k_slice[i] - expected_k[i]).abs() < 1e-4,
                "K mismatch at idx {}",
                i
            );
            assert!(
                (d_slice[i] - expected_d[i]).abs() < 1e-4,
                "D mismatch at idx {}",
                i
            );
        }
        Ok(())
    }

    fn check_stochf_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = StochfInput::with_default_candles(&candles);
        let output = stochf_with_kernel(&input, kernel)?;
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
        Ok(())
    }

    fn check_stochf_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let params = StochfParams {
            fastk_period: Some(0),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        };
        let input = StochfInput::from_slices(&data, &data, &data, params);
        let res = stochf_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_stochf_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = StochfParams {
            fastk_period: Some(10),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        };
        let input = StochfInput::from_slices(&data, &data, &data, params);
        let res = stochf_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_stochf_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let params = StochfParams {
            fastk_period: Some(9),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        };
        let input = StochfInput::from_slices(&data, &data, &data, params);
        let res = stochf_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }

    fn check_stochf_slice_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = StochfParams {
            fastk_period: Some(5),
            fastd_period: Some(3),
            fastd_matype: Some(0),
        };
        let input1 = StochfInput::from_candles(&candles, params.clone());
        let res1 = stochf_with_kernel(&input1, kernel)?;
        let input2 = StochfInput::from_slices(&res1.k, &res1.k, &res1.k, params);
        let res2 = stochf_with_kernel(&input2, kernel)?;
        assert_eq!(res2.k.len(), res1.k.len());
        assert_eq!(res2.d.len(), res1.d.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_stochf_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Define comprehensive parameter combinations
        let test_params = vec![
            StochfParams::default(), // fastk: 5, fastd: 3, matype: 0
            StochfParams {
                fastk_period: Some(2), // minimum viable
                fastd_period: Some(1),
                fastd_matype: Some(0),
            },
            StochfParams {
                fastk_period: Some(3), // small
                fastd_period: Some(2),
                fastd_matype: Some(0),
            },
            StochfParams {
                fastk_period: Some(5), // default fastk with different fastd
                fastd_period: Some(5),
                fastd_matype: Some(0),
            },
            StochfParams {
                fastk_period: Some(10), // medium
                fastd_period: Some(3),
                fastd_matype: Some(0),
            },
            StochfParams {
                fastk_period: Some(14), // standard RSI-like period
                fastd_period: Some(3),
                fastd_matype: Some(0),
            },
            StochfParams {
                fastk_period: Some(20), // large
                fastd_period: Some(5),
                fastd_matype: Some(0),
            },
            StochfParams {
                fastk_period: Some(50), // very large
                fastd_period: Some(10),
                fastd_matype: Some(0),
            },
            StochfParams {
                fastk_period: Some(100), // extreme
                fastd_period: Some(20),
                fastd_matype: Some(0),
            },
            StochfParams {
                fastk_period: Some(8), // edge case: fastd > fastk/2
                fastd_period: Some(7),
                fastd_matype: Some(0),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = StochfInput::from_candles(&candles, params.clone());
            let output = stochf_with_kernel(&input, kernel)?;

            // Check K values
            for (i, &val) in output.k.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at K index {} \
						 with params: fastk_period={}, fastd_period={}, fastd_matype={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(5),
						params.fastd_period.unwrap_or(3),
						params.fastd_matype.unwrap_or(0),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at K index {} \
						 with params: fastk_period={}, fastd_period={}, fastd_matype={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(5),
						params.fastd_period.unwrap_or(3),
						params.fastd_matype.unwrap_or(0),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at K index {} \
						 with params: fastk_period={}, fastd_period={}, fastd_matype={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.fastk_period.unwrap_or(5),
                        params.fastd_period.unwrap_or(3),
                        params.fastd_matype.unwrap_or(0),
                        param_idx
                    );
                }
            }

            // Check D values
            for (i, &val) in output.d.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at D index {} \
						 with params: fastk_period={}, fastd_period={}, fastd_matype={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(5),
						params.fastd_period.unwrap_or(3),
						params.fastd_matype.unwrap_or(0),
						param_idx
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at D index {} \
						 with params: fastk_period={}, fastd_period={}, fastd_matype={} (param set {})",
						test_name, val, bits, i,
						params.fastk_period.unwrap_or(5),
						params.fastd_period.unwrap_or(3),
						params.fastd_matype.unwrap_or(0),
						param_idx
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at D index {} \
						 with params: fastk_period={}, fastd_period={}, fastd_matype={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.fastk_period.unwrap_or(5),
                        params.fastd_period.unwrap_or(3),
                        params.fastd_matype.unwrap_or(0),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_stochf_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(()) // No-op in release builds
    }

    macro_rules! generate_all_stochf_tests {
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

    generate_all_stochf_tests!(
        check_stochf_partial_params,
        check_stochf_accuracy,
        check_stochf_default_candles,
        check_stochf_zero_period,
        check_stochf_period_exceeds_length,
        check_stochf_very_small_dataset,
        check_stochf_slice_reinput,
        check_stochf_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_stochf_tests!(check_stochf_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = StochfBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        let def = StochfParams::default();
        let krow = output.k_for(&def).expect("default row missing");
        let drow = output.d_for(&def).expect("default row missing");
        assert_eq!(krow.len(), c.close.len());
        assert_eq!(drow.len(), c.close.len());
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test various parameter sweep configurations
        let test_configs = vec![
            // (fastk_start, fastk_end, fastk_step, fastd_start, fastd_end, fastd_step)
            (2, 10, 2, 1, 5, 1),       // Small periods
            (5, 25, 5, 3, 3, 0),       // Medium periods with static fastd
            (30, 60, 15, 5, 15, 5),    // Large periods
            (2, 5, 1, 1, 3, 1),        // Dense small range
            (10, 20, 2, 3, 9, 3),      // Medium range with varying fastd
            (14, 14, 0, 1, 7, 2),      // Static fastk with varying fastd
            (3, 12, 3, 2, 2, 0),       // Small to medium with static fastd
            (50, 100, 25, 10, 20, 10), // Very large periods
        ];

        for (cfg_idx, &(fk_start, fk_end, fk_step, fd_start, fd_end, fd_step)) in
            test_configs.iter().enumerate()
        {
            let output = StochfBatchBuilder::new()
                .kernel(kernel)
                .fastk_range(fk_start, fk_end, fk_step)
                .fastd_range(fd_start, fd_end, fd_step)
                .apply_candles(&c)?;

            // Check K values
            for (idx, &val) in output.k.iter().enumerate() {
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
						 at K row {} col {} (flat index {}) with params: fastk={}, fastd={}, matype={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fastk_period.unwrap_or(5),
                        combo.fastd_period.unwrap_or(3),
                        combo.fastd_matype.unwrap_or(0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at K row {} col {} (flat index {}) with params: fastk={}, fastd={}, matype={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fastk_period.unwrap_or(5),
                        combo.fastd_period.unwrap_or(3),
                        combo.fastd_matype.unwrap_or(0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at K row {} col {} (flat index {}) with params: fastk={}, fastd={}, matype={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fastk_period.unwrap_or(5),
                        combo.fastd_period.unwrap_or(3),
                        combo.fastd_matype.unwrap_or(0)
                    );
                }
            }

            // Check D values
            for (idx, &val) in output.d.iter().enumerate() {
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
						 at D row {} col {} (flat index {}) with params: fastk={}, fastd={}, matype={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fastk_period.unwrap_or(5),
                        combo.fastd_period.unwrap_or(3),
                        combo.fastd_matype.unwrap_or(0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at D row {} col {} (flat index {}) with params: fastk={}, fastd={}, matype={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fastk_period.unwrap_or(5),
                        combo.fastd_period.unwrap_or(3),
                        combo.fastd_matype.unwrap_or(0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at D row {} col {} (flat index {}) with params: fastk={}, fastd={}, matype={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.fastk_period.unwrap_or(5),
                        combo.fastd_period.unwrap_or(3),
                        combo.fastd_matype.unwrap_or(0)
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
    fn check_stochf_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy to generate valid OHLC data where high >= close >= low
        let strat = (2usize..=50) // fastk_period range
            .prop_flat_map(|fastk_period| {
                (
                    // Generate base prices and close positions
                    prop::collection::vec(
                        (1.0f64..1000.0f64).prop_filter("finite", |x| x.is_finite()),
                        fastk_period + 50..400, // Ensure enough data
                    ),
                    // Generate random close position within [low, high] for each price
                    prop::collection::vec(
                        0.0f64..=1.0f64, // Position within [low, high] range
                        fastk_period + 50..400,
                    ),
                    Just(fastk_period),
                    1usize..=10,      // fastd_period range
                    0.001f64..0.1f64, // volatility factor for generating high/low spread
                )
            });

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |(base_prices, close_positions, fastk_period, fastd_period, volatility)| {
                    // Generate realistic OHLC data from base prices
                    let len = base_prices.len().min(close_positions.len());
                    let mut high = Vec::with_capacity(len);
                    let mut low = Vec::with_capacity(len);
                    let mut close = Vec::with_capacity(len);

                    for i in 0..len {
                        let base = base_prices[i];
                        let spread = base * volatility;
                        let h = base + spread * 0.5;
                        let l = base - spread * 0.5;
                        // Close is randomly positioned between low and high
                        let c = l + (h - l) * close_positions[i]; // Random position within range

                        high.push(h);
                        low.push(l);
                        close.push(c);
                    }

                    let params = StochfParams {
                        fastk_period: Some(fastk_period),
                        fastd_period: Some(fastd_period),
                        fastd_matype: Some(0),
                    };
                    let input = StochfInput::from_slices(&high, &low, &close, params.clone());

                    let output = stochf_with_kernel(&input, kernel).unwrap();
                    let ref_output = stochf_with_kernel(&input, Kernel::Scalar).unwrap();

                    // Property 1: K values must be in [0, 100] range
                    for (i, &k_val) in output.k.iter().enumerate() {
                        if !k_val.is_nan() {
                            prop_assert!(
                                k_val >= -1e-9 && k_val <= 100.0 + 1e-9,
                                "K value out of range at idx {}: {} (should be in [0, 100])",
                                i,
                                k_val
                            );
                        }
                    }

                    // Property 2: D values must be in [0, 100] range
                    for (i, &d_val) in output.d.iter().enumerate() {
                        if !d_val.is_nan() {
                            prop_assert!(
                                d_val >= -1e-9 && d_val <= 100.0 + 1e-9,
                                "D value out of range at idx {}: {} (should be in [0, 100])",
                                i,
                                d_val
                            );
                        }
                    }

                    // Property 3: Warmup period handling
                    let k_warmup = fastk_period - 1;
                    let d_warmup = fastk_period - 1 + fastd_period - 1;

                    for i in 0..k_warmup.min(len) {
                        prop_assert!(
                            output.k[i].is_nan(),
                            "K value should be NaN during warmup at idx {}: {}",
                            i,
                            output.k[i]
                        );
                    }

                    for i in 0..d_warmup.min(len) {
                        prop_assert!(
                            output.d[i].is_nan(),
                            "D value should be NaN during warmup at idx {}: {}",
                            i,
                            output.d[i]
                        );
                    }

                    // Property 4: Kernel consistency - all kernels should produce same results
                    for i in 0..len {
                        let k_val = output.k[i];
                        let k_ref = ref_output.k[i];
                        let d_val = output.d[i];
                        let d_ref = ref_output.d[i];

                        if !k_val.is_nan() && !k_ref.is_nan() {
                            prop_assert!(
                                (k_val - k_ref).abs() <= 1e-9,
                                "K kernel mismatch at idx {}: {} vs {} (diff: {})",
                                i,
                                k_val,
                                k_ref,
                                (k_val - k_ref).abs()
                            );
                        }

                        if !d_val.is_nan() && !d_ref.is_nan() {
                            prop_assert!(
                                (d_val - d_ref).abs() <= 1e-9,
                                "D kernel mismatch at idx {}: {} vs {} (diff: {})",
                                i,
                                d_val,
                                d_ref,
                                (d_val - d_ref).abs()
                            );
                        }
                    }

                    // Property 5: K value formula verification
                    // K = 100 * (close - lowest_low) / (highest_high - lowest_low)
                    for i in k_warmup..len {
                        let start = i + 1 - fastk_period;
                        let window_high = &high[start..=i];
                        let window_low = &low[start..=i];

                        let hh = window_high
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);
                        let ll = window_low.iter().cloned().fold(f64::INFINITY, f64::min);

                        let expected_k = if hh == ll {
                            if close[i] == hh {
                                100.0
                            } else {
                                0.0
                            }
                        } else {
                            100.0 * (close[i] - ll) / (hh - ll)
                        };

                        let actual_k = output.k[i];
                        prop_assert!(
                            (actual_k - expected_k).abs() <= 1e-9,
                            "K formula mismatch at idx {}: actual {} vs expected {} (diff: {})",
                            i,
                            actual_k,
                            expected_k,
                            (actual_k - expected_k).abs()
                        );
                    }

                    // Property 6: D is simple moving average of K
                    for i in d_warmup..len {
                        let start = i + 1 - fastd_period;
                        let k_window = &output.k[start..=i];
                        let expected_d = k_window.iter().sum::<f64>() / (fastd_period as f64);
                        let actual_d = output.d[i];

                        prop_assert!(
                            (actual_d - expected_d).abs() <= 1e-9,
                            "D SMA mismatch at idx {}: actual {} vs expected {} (diff: {})",
                            i,
                            actual_d,
                            expected_d,
                            (actual_d - expected_d).abs()
                        );
                    }

                    // Property 7: Special case - when high == low == close (constant price)
                    // Test with a small constant price window
                    // Ensure we have enough data for both fastk and fastd periods
                    let const_len = (fastk_period + fastd_period) * 2;
                    if len > const_len {
                        let const_price = 100.0;
                        let const_high = vec![const_price; const_len];
                        let const_low = vec![const_price; const_len];
                        let const_close = vec![const_price; const_len];

                        let const_input = StochfInput::from_slices(
                            &const_high,
                            &const_low,
                            &const_close,
                            params.clone(),
                        );
                        let const_output = stochf_with_kernel(&const_input, kernel).unwrap();

                        // When all prices are equal, K should be 100 (since close == high)
                        for i in k_warmup..const_high.len() {
                            prop_assert!(
                                (const_output.k[i] - 100.0).abs() <= 1e-9,
                                "Constant price K should be 100 at idx {}: {}",
                                i,
                                const_output.k[i]
                            );
                        }
                    }

                    // Property 8: Extreme cases - close at boundaries
                    // Test when close is consistently at low (K should be near 0) or high (K should be near 100)
                    let extreme_len = (fastk_period + fastd_period) * 2;
                    if len > extreme_len {
                        // Test 1: Close at low boundary with constant prices
                        let low_close_high = vec![100.0; extreme_len]; // Constant high
                        let low_close_low = vec![90.0; extreme_len]; // Constant low
                        let low_close_close = vec![90.0; extreme_len]; // Close equals low

                        let low_input = StochfInput::from_slices(
                            &low_close_high,
                            &low_close_low,
                            &low_close_close,
                            params.clone(),
                        );
                        let low_output = stochf_with_kernel(&low_input, kernel).unwrap();

                        // When close == low consistently, K should be 0
                        for i in k_warmup..low_close_high.len() {
                            prop_assert!(
                                low_output.k[i].abs() <= 1e-9,
                                "When close == low, K should be 0 at idx {}: {}",
                                i,
                                low_output.k[i]
                            );
                        }

                        // Test 2: Close at high boundary with constant prices
                        let high_close_high = vec![100.0; extreme_len]; // Constant high
                        let high_close_low = vec![90.0; extreme_len]; // Constant low
                        let high_close_close = vec![100.0; extreme_len]; // Close equals high

                        let high_input = StochfInput::from_slices(
                            &high_close_high,
                            &high_close_low,
                            &high_close_close,
                            params.clone(),
                        );
                        let high_output = stochf_with_kernel(&high_input, kernel).unwrap();

                        // When close == high consistently, K should be 100
                        for i in k_warmup..high_close_high.len() {
                            prop_assert!(
                                (high_output.k[i] - 100.0).abs() <= 1e-9,
                                "When close == high, K should be 100 at idx {}: {}",
                                i,
                                high_output.k[i]
                            );
                        }
                    }

                    // Property 9: No poison values in debug mode
                    #[cfg(debug_assertions)]
                    {
                        for (i, &val) in output.k.iter().enumerate() {
                            if !val.is_nan() {
                                let bits = val.to_bits();
                                prop_assert!(
                                    bits != 0x11111111_11111111
                                        && bits != 0x22222222_22222222
                                        && bits != 0x33333333_33333333,
                                    "Found poison value in K at idx {}: {} (0x{:016X})",
                                    i,
                                    val,
                                    bits
                                );
                            }
                        }

                        for (i, &val) in output.d.iter().enumerate() {
                            if !val.is_nan() {
                                let bits = val.to_bits();
                                prop_assert!(
                                    bits != 0x11111111_11111111
                                        && bits != 0x22222222_22222222
                                        && bits != 0x33333333_33333333,
                                    "Found poison value in D at idx {}: {} (0x{:016X})",
                                    i,
                                    val,
                                    bits
                                );
                            }
                        }
                    }

                    Ok(())
                },
            )
            .unwrap();

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
    #[cfg(feature = "wasm")]
    #[test]
    fn test_wasm_batch_warmup_initialization() {
        // Test that WASM batch function correctly initializes warmup periods
        // even in the aliasing case where we removed redundant initialization
        let high = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let low = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
        let close = vec![0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8];

        // Create output buffers
        let mut k_out = vec![999.0; 10]; // Fill with non-NaN to verify warmup
        let mut d_out = vec![999.0; 10];

        // Call the function (single parameter set for simplicity)
        let result = unsafe {
            stochf_batch_into(
                high.as_ptr(),
                low.as_ptr(),
                close.as_ptr(),
                k_out.as_mut_ptr(),
                d_out.as_mut_ptr(),
                10,
                3,
                3,
                0, // fastk: single value 3
                2,
                2,
                0, // fastd: single value 2
                0, // matype
            )
        };

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1); // 1 combination

        // Verify warmup periods are NaN
        // With first_valid_idx = 0, fastk = 3, fastd = 2:
        // K warmup should be 0 + 3 - 1 = 2 (indices 0, 1)
        // D warmup should be 0 + 3 + 2 - 2 = 3 (indices 0, 1, 2)
        assert!(k_out[0].is_nan(), "K[0] should be NaN");
        assert!(k_out[1].is_nan(), "K[1] should be NaN");
        assert!(!k_out[2].is_nan(), "K[2] should have a value");

        assert!(d_out[0].is_nan(), "D[0] should be NaN");
        assert!(d_out[1].is_nan(), "D[1] should be NaN");
        assert!(d_out[2].is_nan(), "D[2] should be NaN");
        assert!(!d_out[3].is_nan(), "D[3] should have a value");
    }

    #[test]
    fn test_batch_invalid_output_size() {
        // Test that OutputLengthMismatch error is properly returned
        let high = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let low = vec![5.0, 15.0, 25.0, 35.0, 45.0];
        let close = vec![7.0, 17.0, 27.0, 37.0, 47.0];

        let sweep = StochfBatchRange {
            fastk_period: (3, 4, 1), // Will create 2 combinations
            fastd_period: (2, 2, 0),
        };

        // Create output buffers with wrong sizes
        let mut k_out = vec![0.0; 5]; // Should be 2 * 5 = 10
        let mut d_out = vec![0.0; 5]; // Should be 2 * 5 = 10

        let result = stochf_batch_inner_into(
            &high,
            &low,
            &close,
            &sweep,
            Kernel::Scalar,
            false,
            &mut k_out,
            &mut d_out,
        );

        // Should return OutputLengthMismatch error
        assert!(matches!(
            result,
            Err(StochfError::OutputLengthMismatch {
                expected: 10,
                k_got: 5,
                d_got: 5
            })
        ));

        // Test with correct k_out but wrong d_out
        let mut k_out = vec![0.0; 10]; // Correct size
        let mut d_out = vec![0.0; 8]; // Wrong size

        let result = stochf_batch_inner_into(
            &high,
            &low,
            &close,
            &sweep,
            Kernel::Scalar,
            false,
            &mut k_out,
            &mut d_out,
        );

        // Should return OutputLengthMismatch error
        assert!(matches!(
            result,
            Err(StochfError::OutputLengthMismatch {
                expected: 10,
                k_got: 10,
                d_got: 8
            })
        ));

        // Test with correct sizes - should succeed
        let mut k_out = vec![0.0; 10]; // Correct size
        let mut d_out = vec![0.0; 10]; // Correct size

        let result = stochf_batch_inner_into(
            &high,
            &low,
            &close,
            &sweep,
            Kernel::Scalar,
            false,
            &mut k_out,
            &mut d_out,
        );

        // Should succeed
        assert!(result.is_ok());
    }

    // Parity test: into() vs Vec API
    #[test]
    fn test_stochf_into_matches_api() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("failed to read csv");
        let input = StochfInput::with_default_candles(&candles);

        // Baseline via Vec-returning API
        let base = stochf(&input).expect("baseline stochf failed");

        // Into API with preallocated outputs
        let len = candles.close.len();
        let mut k_out = vec![0.0f64; len];
        let mut d_out = vec![0.0f64; len];
        stochf_into(&input, &mut k_out, &mut d_out).expect("stochf_into failed");

        assert_eq!(base.k.len(), k_out.len());
        assert_eq!(base.d.len(), d_out.len());

        fn eq_or_nan(a: f64, b: f64) -> bool { (a.is_nan() && b.is_nan()) || (a == b) }

        for i in 0..len {
            assert!(
                eq_or_nan(base.k[i], k_out[i]),
                "K mismatch at {}: base={:?} into={:?}",
                i,
                base.k[i],
                k_out[i]
            );
            assert!(
                eq_or_nan(base.d[i], d_out[i]),
                "D mismatch at {}: base={:?} into={:?}",
                i,
                base.d[i],
                d_out[i]
            );
        }
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stochf_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    fastk_period: usize,
    fastd_period: usize,
    fastd_matype: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = StochfParams {
        fastk_period: Some(fastk_period),
        fastd_period: Some(fastd_period),
        fastd_matype: Some(fastd_matype),
    };
    let input = StochfInput::from_slices(high, low, close, params);
    let out =
        stochf_with_kernel(&input, Kernel::Auto).map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Return [K..., D...] concatenated
    let mut values = Vec::with_capacity(2 * out.k.len());
    values.extend_from_slice(&out.k);
    values.extend_from_slice(&out.d);

    Ok(values)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stochf_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stochf_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stochf_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    k_out_ptr: *mut f64,
    d_out_ptr: *mut f64,
    len: usize,
    fastk_period: usize,
    fastd_period: usize,
    fastd_matype: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || k_out_ptr.is_null()
        || d_out_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer passed to stochf_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let mut k_out = std::slice::from_raw_parts_mut(k_out_ptr, len);
        let mut d_out = std::slice::from_raw_parts_mut(d_out_ptr, len);

        let params = StochfParams {
            fastk_period: Some(fastk_period),
            fastd_period: Some(fastd_period),
            fastd_matype: Some(fastd_matype),
        };
        let input = StochfInput::from_slices(high, low, close, params);
        stochf_into_slice(&mut k_out, &mut d_out, &input, detect_best_kernel())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct StochfBatchConfig {
    pub fastk_range: (usize, usize, usize),
    pub fastd_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct StochfBatchJsOutput {
    pub k_values: Vec<f64>, // K values for all combinations
    pub d_values: Vec<f64>, // D values for all combinations
    pub rows: usize,        // number of combinations
    pub cols: usize,        // data length
    pub combos: Vec<StochfParams>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = stochf_batch)]
pub fn stochf_batch_unified_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let cfg: StochfBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = StochfBatchRange {
        fastk_period: cfg.fastk_range,
        fastd_period: cfg.fastd_range,
    };

    let out = stochf_batch_inner(high, low, close, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Keep K and D values separate as expected by tests
    let js = StochfBatchJsOutput {
        k_values: out.k,
        d_values: out.d,
        rows: out.rows, // Number of combinations, not doubled
        cols: out.cols,
        combos: out.combos,
    };
    serde_wasm_bindgen::to_value(&js)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn stochf_batch_into(
    in_high_ptr: *const f64,
    in_low_ptr: *const f64,
    in_close_ptr: *const f64,
    out_k_ptr: *mut f64,
    out_d_ptr: *mut f64,
    len: usize,
    fastk_start: usize,
    fastk_end: usize,
    fastk_step: usize,
    fastd_start: usize,
    fastd_end: usize,
    fastd_step: usize,
    fastd_matype: usize,
) -> Result<usize, JsValue> {
    if in_high_ptr.is_null()
        || in_low_ptr.is_null()
        || in_close_ptr.is_null()
        || out_k_ptr.is_null()
        || out_d_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(in_high_ptr, len);
        let low = std::slice::from_raw_parts(in_low_ptr, len);
        let close = std::slice::from_raw_parts(in_close_ptr, len);

        let sweep = StochfBatchRange {
            fastk_period: (fastk_start, fastk_end, fastk_step),
            fastd_period: (fastd_start, fastd_end, fastd_step),
        };

        // Get combos to calculate output size
        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        // Check for aliasing
        let aliasing = in_high_ptr == out_k_ptr
            || in_high_ptr == out_d_ptr
            || in_low_ptr == out_k_ptr
            || in_low_ptr == out_d_ptr
            || in_close_ptr == out_k_ptr
            || in_close_ptr == out_d_ptr;

        if aliasing {
            // Use temporary buffers
            let mut temp_k = vec![0.0; rows * cols];
            let mut temp_d = vec![0.0; rows * cols];

            // Compute - stochf_batch_inner_into will handle NaN initialization with first_valid_idx
            let kernel = detect_best_batch_kernel();
            // Convert batch kernel to regular kernel
            let simd_kernel = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => Kernel::Scalar,
            };

            stochf_batch_inner_into(
                high,
                low,
                close,
                &sweep,
                simd_kernel,
                false,
                &mut temp_k,
                &mut temp_d,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy to output
            let out_k_slice = std::slice::from_raw_parts_mut(out_k_ptr, rows * cols);
            let out_d_slice = std::slice::from_raw_parts_mut(out_d_ptr, rows * cols);

            out_k_slice.copy_from_slice(&temp_k);
            out_d_slice.copy_from_slice(&temp_d);
        } else {
            // Direct write to output
            let out_k_slice = std::slice::from_raw_parts_mut(out_k_ptr, rows * cols);
            let out_d_slice = std::slice::from_raw_parts_mut(out_d_ptr, rows * cols);

            // Compute - stochf_batch_inner_into now handles NaN initialization
            let kernel = detect_best_batch_kernel();
            // Convert batch kernel to regular kernel
            let simd_kernel = match kernel {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => Kernel::Scalar,
            };
            stochf_batch_inner_into(
                high,
                low,
                close,
                &sweep,
                simd_kernel,
                false,
                out_k_slice,
                out_d_slice,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(rows)
    }
}
