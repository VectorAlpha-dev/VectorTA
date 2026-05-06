#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_uninit_f64, alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel,
    init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use std::arch::is_x86_feature_detected;
use std::mem::ManuallyDrop;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum YangZhangVolatilityData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct YangZhangVolatilityOutput {
    pub yz: Vec<f64>,
    pub rs: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct YangZhangVolatilityParams {
    pub lookback: Option<usize>,
    pub k_override: Option<bool>,
    pub k: Option<f64>,
}

impl Default for YangZhangVolatilityParams {
    fn default() -> Self {
        Self {
            lookback: Some(14),
            k_override: Some(false),
            k: Some(0.34),
        }
    }
}

#[derive(Debug, Clone)]
pub struct YangZhangVolatilityInput<'a> {
    pub data: YangZhangVolatilityData<'a>,
    pub params: YangZhangVolatilityParams,
}

impl<'a> YangZhangVolatilityInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: YangZhangVolatilityParams) -> Self {
        Self {
            data: YangZhangVolatilityData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: YangZhangVolatilityParams,
    ) -> Self {
        Self {
            data: YangZhangVolatilityData::Slices {
                open,
                high,
                low,
                close,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, YangZhangVolatilityParams::default())
    }

    #[inline]
    pub fn get_lookback(&self) -> usize {
        self.params.lookback.unwrap_or(14)
    }

    #[inline]
    pub fn get_k_override(&self) -> bool {
        self.params.k_override.unwrap_or(false)
    }

    #[inline]
    pub fn get_k(&self) -> f64 {
        self.params.k.unwrap_or(0.34)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct YangZhangVolatilityBuilder {
    lookback: Option<usize>,
    k_override: Option<bool>,
    k: Option<f64>,
    kernel: Kernel,
}

impl Default for YangZhangVolatilityBuilder {
    fn default() -> Self {
        Self {
            lookback: None,
            k_override: None,
            k: None,
            kernel: Kernel::Auto,
        }
    }
}

impl YangZhangVolatilityBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn lookback(mut self, n: usize) -> Self {
        self.lookback = Some(n);
        self
    }

    #[inline(always)]
    pub fn k_override(mut self, v: bool) -> Self {
        self.k_override = Some(v);
        self
    }

    #[inline(always)]
    pub fn k(mut self, v: f64) -> Self {
        self.k = Some(v);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<YangZhangVolatilityOutput, YangZhangVolatilityError> {
        let p = YangZhangVolatilityParams {
            lookback: self.lookback,
            k_override: self.k_override,
            k: self.k,
        };
        let i = YangZhangVolatilityInput::from_candles(c, p);
        yang_zhang_volatility_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<YangZhangVolatilityOutput, YangZhangVolatilityError> {
        let p = YangZhangVolatilityParams {
            lookback: self.lookback,
            k_override: self.k_override,
            k: self.k,
        };
        let i = YangZhangVolatilityInput::from_slices(open, high, low, close, p);
        yang_zhang_volatility_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<YangZhangVolatilityStream, YangZhangVolatilityError> {
        let p = YangZhangVolatilityParams {
            lookback: self.lookback,
            k_override: self.k_override,
            k: self.k,
        };
        YangZhangVolatilityStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum YangZhangVolatilityError {
    #[error("yang_zhang_volatility: Input data slice is empty.")]
    EmptyInputData,
    #[error("yang_zhang_volatility: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "yang_zhang_volatility: Invalid lookback: lookback = {lookback}, data length = {data_len}"
    )]
    InvalidLookback { lookback: usize, data_len: usize },
    #[error("yang_zhang_volatility: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("yang_zhang_volatility: Inconsistent slice lengths: open={open_len}, high={high_len}, low={low_len}, close={close_len}")]
    InconsistentSliceLengths {
        open_len: usize,
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error(
        "yang_zhang_volatility: Invalid k override value: {k}. Must be finite and within [0, 1]."
    )]
    InvalidK { k: f64 },
    #[error("yang_zhang_volatility: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("yang_zhang_volatility: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("yang_zhang_volatility: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone)]
pub struct YangZhangVolatilityStream {
    lookback: usize,
    k: f64,
    prev_close: f64,
    o: Vec<f64>,
    c: Vec<f64>,
    rs: Vec<f64>,
    sum_o: f64,
    sumsq_o: f64,
    sum_c: f64,
    sumsq_c: f64,
    sum_rs: f64,
    idx: usize,
    cnt: usize,
}

impl YangZhangVolatilityStream {
    #[inline(always)]
    pub fn try_new(params: YangZhangVolatilityParams) -> Result<Self, YangZhangVolatilityError> {
        let lookback = params.lookback.unwrap_or(14);
        if lookback == 0 {
            return Err(YangZhangVolatilityError::InvalidLookback {
                lookback,
                data_len: 0,
            });
        }

        let k = if params.k_override.unwrap_or(false) {
            let k = params.k.unwrap_or(0.34);
            if !k.is_finite() || !(0.0..=1.0).contains(&k) {
                return Err(YangZhangVolatilityError::InvalidK { k });
            }
            k
        } else {
            k_default(lookback)
        };

        Ok(Self {
            lookback,
            k,
            prev_close: f64::NAN,
            o: vec![0.0; lookback],
            c: vec![0.0; lookback],
            rs: vec![0.0; lookback],
            sum_o: 0.0,
            sumsq_o: 0.0,
            sum_c: 0.0,
            sumsq_c: 0.0,
            sum_rs: 0.0,
            idx: 0,
            cnt: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        if !open.is_finite() || !high.is_finite() || !low.is_finite() || !close.is_finite() {
            self.prev_close = close;
            return None;
        }
        if open <= 0.0 || high <= 0.0 || low <= 0.0 || close <= 0.0 {
            self.prev_close = close;
            return None;
        }

        if self.prev_close.is_nan() {
            self.prev_close = close;
            return None;
        }

        let oret = (open / self.prev_close).ln();
        let cret = (close / open).ln();
        let rsc = rs_component(high, low, open, close);

        self.prev_close = close;

        let i = self.idx;
        if self.cnt < self.lookback {
            self.o[i] = oret;
            self.c[i] = cret;
            self.rs[i] = rsc;

            self.sum_o += oret;
            self.sumsq_o += oret * oret;
            self.sum_c += cret;
            self.sumsq_c += cret * cret;
            self.sum_rs += rsc;

            self.cnt += 1;
        } else {
            let old_o = self.o[i];
            let old_c = self.c[i];
            let old_rs = self.rs[i];

            self.sum_o -= old_o;
            self.sumsq_o -= old_o * old_o;
            self.sum_c -= old_c;
            self.sumsq_c -= old_c * old_c;
            self.sum_rs -= old_rs;

            self.o[i] = oret;
            self.c[i] = cret;
            self.rs[i] = rsc;

            self.sum_o += oret;
            self.sumsq_o += oret * oret;
            self.sum_c += cret;
            self.sumsq_c += cret * cret;
            self.sum_rs += rsc;
        }

        self.idx += 1;
        if self.idx == self.lookback {
            self.idx = 0;
        }

        if self.cnt < self.lookback {
            return None;
        }

        let lb_f = self.lookback as f64;
        let mut rs_var = self.sum_rs / lb_f;
        if rs_var < 0.0 {
            rs_var = 0.0;
        }
        let rs_out = rs_var.sqrt();

        let o_var = sample_var(self.sum_o, self.sumsq_o, self.lookback);
        let c_var = sample_var(self.sum_c, self.sumsq_c, self.lookback);
        let mut yz_var = o_var + self.k * c_var + (1.0 - self.k) * rs_var;
        if yz_var < 0.0 {
            yz_var = 0.0;
        }
        let yz_out = yz_var.sqrt();

        Some((yz_out, rs_out))
    }

    #[inline(always)]
    pub fn get_warmup_period(&self) -> usize {
        self.lookback
    }
}

#[inline]
pub fn yang_zhang_volatility(
    input: &YangZhangVolatilityInput,
) -> Result<YangZhangVolatilityOutput, YangZhangVolatilityError> {
    yang_zhang_volatility_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn k_default(lookback: usize) -> f64 {
    if lookback <= 1 {
        0.0
    } else {
        0.34 / (1.34 + ((lookback + 1) as f64) / ((lookback - 1) as f64))
    }
}

#[inline(always)]
fn first_valid_ohlc(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let len = close.len();
    let mut i = 0;
    while i < len {
        if !open[i].is_nan() && !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
            break;
        }
        i += 1;
    }
    i.min(len)
}

#[inline(always)]
fn yang_zhang_prepare<'a>(
    input: &'a YangZhangVolatilityInput,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        usize,
        usize,
        Kernel,
    ),
    YangZhangVolatilityError,
> {
    let (open, high, low, close): (&[f64], &[f64], &[f64], &[f64]) = match &input.data {
        YangZhangVolatilityData::Candles { candles } => {
            (&candles.open, &candles.high, &candles.low, &candles.close)
        }
        YangZhangVolatilityData::Slices {
            open,
            high,
            low,
            close,
        } => (open, high, low, close),
    };

    let len = close.len();
    if len == 0 {
        return Err(YangZhangVolatilityError::EmptyInputData);
    }
    if open.len() != len || high.len() != len || low.len() != len {
        return Err(YangZhangVolatilityError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let first = first_valid_ohlc(open, high, low, close);
    if first >= len {
        return Err(YangZhangVolatilityError::AllValuesNaN);
    }

    let lookback = input.get_lookback();
    if lookback == 0 || lookback > len {
        return Err(YangZhangVolatilityError::InvalidLookback {
            lookback,
            data_len: len,
        });
    }
    if len - first < lookback + 1 {
        return Err(YangZhangVolatilityError::NotEnoughValidData {
            needed: lookback + 1,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Auto => {
            if len >= 262_144 {
                Kernel::Scalar
            } else {
                detect_best_kernel()
            }
        }
        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        Kernel::Auto => detect_best_kernel(),
        k => k.to_non_batch(),
    };

    Ok((open, high, low, close, lookback, first, chosen))
}

#[inline(always)]
fn rs_component(high: f64, low: f64, open: f64, close: f64) -> f64 {
    (high / close).ln() * (high / open).ln() + (low / close).ln() * (low / open).ln()
}

#[inline(always)]
fn sample_var(sum: f64, sumsq: f64, n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let nf = n as f64;
    let denom = (n - 1) as f64;
    let mut v = (sumsq - (sum * sum) / nf) / denom;
    if v < 0.0 {
        v = 0.0;
    }
    v
}

#[inline]
fn yang_zhang_precompute_ln_diff_scalar(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    oret: &mut [f64],
    cret: &mut [f64],
    rs_val: &mut [f64],
) {
    let len = close.len();
    if len == 0 {
        return;
    }

    let mut prev_ln_close = close[0].ln();
    oret[0] = 0.0;

    for i in 0..len {
        let ln_open = open[i].ln();
        let ln_high = high[i].ln();
        let ln_low = low[i].ln();
        let ln_close = close[i].ln();

        let h_c = ln_high - ln_close;
        let h_o = ln_high - ln_open;
        let l_c = ln_low - ln_close;
        let l_o = ln_low - ln_open;
        rs_val[i] = h_c * h_o + l_c * l_o;

        cret[i] = ln_close - ln_open;
        if i > 0 {
            oret[i] = ln_open - prev_ln_close;
        }
        prev_ln_close = ln_close;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _mm256_abs_pd(a: __m256d) -> __m256d {
    let sign_mask = _mm256_set1_pd(-0.0);
    _mm256_andnot_pd(sign_mask, a)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn _mm512_abs_pd(a: __m512d) -> __m512d {
    let sign_mask = _mm512_set1_pd(-0.0);
    _mm512_andnot_pd(sign_mask, a)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn ln1p_taylor_14_avx2(y: __m256d) -> __m256d {
    let c0 = _mm256_set1_pd(1.0);
    let c1 = _mm256_set1_pd(-1.0 / 2.0);
    let c2 = _mm256_set1_pd(1.0 / 3.0);
    let c3 = _mm256_set1_pd(-1.0 / 4.0);
    let c4 = _mm256_set1_pd(1.0 / 5.0);
    let c5 = _mm256_set1_pd(-1.0 / 6.0);
    let c6 = _mm256_set1_pd(1.0 / 7.0);
    let c7 = _mm256_set1_pd(-1.0 / 8.0);
    let c8 = _mm256_set1_pd(1.0 / 9.0);
    let c9 = _mm256_set1_pd(-1.0 / 10.0);
    let c10 = _mm256_set1_pd(1.0 / 11.0);
    let c11 = _mm256_set1_pd(-1.0 / 12.0);
    let c12 = _mm256_set1_pd(1.0 / 13.0);
    let c13 = _mm256_set1_pd(-1.0 / 14.0);

    let mut p = c13;
    p = _mm256_fmadd_pd(y, p, c12);
    p = _mm256_fmadd_pd(y, p, c11);
    p = _mm256_fmadd_pd(y, p, c10);
    p = _mm256_fmadd_pd(y, p, c9);
    p = _mm256_fmadd_pd(y, p, c8);
    p = _mm256_fmadd_pd(y, p, c7);
    p = _mm256_fmadd_pd(y, p, c6);
    p = _mm256_fmadd_pd(y, p, c5);
    p = _mm256_fmadd_pd(y, p, c4);
    p = _mm256_fmadd_pd(y, p, c3);
    p = _mm256_fmadd_pd(y, p, c2);
    p = _mm256_fmadd_pd(y, p, c1);
    p = _mm256_fmadd_pd(y, p, c0);

    _mm256_mul_pd(y, p)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
unsafe fn ln1p_taylor_14_avx512(y: __m512d) -> __m512d {
    let c0 = _mm512_set1_pd(1.0);
    let c1 = _mm512_set1_pd(-1.0 / 2.0);
    let c2 = _mm512_set1_pd(1.0 / 3.0);
    let c3 = _mm512_set1_pd(-1.0 / 4.0);
    let c4 = _mm512_set1_pd(1.0 / 5.0);
    let c5 = _mm512_set1_pd(-1.0 / 6.0);
    let c6 = _mm512_set1_pd(1.0 / 7.0);
    let c7 = _mm512_set1_pd(-1.0 / 8.0);
    let c8 = _mm512_set1_pd(1.0 / 9.0);
    let c9 = _mm512_set1_pd(-1.0 / 10.0);
    let c10 = _mm512_set1_pd(1.0 / 11.0);
    let c11 = _mm512_set1_pd(-1.0 / 12.0);
    let c12 = _mm512_set1_pd(1.0 / 13.0);
    let c13 = _mm512_set1_pd(-1.0 / 14.0);

    let mut p = c13;
    p = _mm512_fmadd_pd(y, p, c12);
    p = _mm512_fmadd_pd(y, p, c11);
    p = _mm512_fmadd_pd(y, p, c10);
    p = _mm512_fmadd_pd(y, p, c9);
    p = _mm512_fmadd_pd(y, p, c8);
    p = _mm512_fmadd_pd(y, p, c7);
    p = _mm512_fmadd_pd(y, p, c6);
    p = _mm512_fmadd_pd(y, p, c5);
    p = _mm512_fmadd_pd(y, p, c4);
    p = _mm512_fmadd_pd(y, p, c3);
    p = _mm512_fmadd_pd(y, p, c2);
    p = _mm512_fmadd_pd(y, p, c1);
    p = _mm512_fmadd_pd(y, p, c0);

    _mm512_mul_pd(y, p)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn yang_zhang_precompute_ln_diff_avx2(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    oret: &mut [f64],
    cret: &mut [f64],
    rs_val: &mut [f64],
) {
    let len = close.len();
    if len == 0 {
        return;
    }

    let ln_open0 = open[0].ln();
    let ln_high0 = high[0].ln();
    let ln_low0 = low[0].ln();
    let ln_close0 = close[0].ln();

    oret[0] = 0.0;
    cret[0] = ln_close0 - ln_open0;
    let h_c0 = ln_high0 - ln_close0;
    let h_o0 = ln_high0 - ln_open0;
    let l_c0 = ln_low0 - ln_close0;
    let l_o0 = ln_low0 - ln_open0;
    rs_val[0] = h_c0 * h_o0 + l_c0 * l_o0;

    let one = _mm256_set1_pd(1.0);
    let threshold = _mm256_set1_pd(0.2);

    let mut i = 1usize;
    while i + 4 <= len {
        let o = _mm256_loadu_pd(open.as_ptr().add(i));
        let h = _mm256_loadu_pd(high.as_ptr().add(i));
        let l = _mm256_loadu_pd(low.as_ptr().add(i));
        let c = _mm256_loadu_pd(close.as_ptr().add(i));
        let c_prev = _mm256_loadu_pd(close.as_ptr().add(i - 1));

        let r_oret = _mm256_div_pd(o, c_prev);
        let r_cret = _mm256_div_pd(c, o);
        let r_hc = _mm256_div_pd(h, c);
        let r_ho = _mm256_div_pd(h, o);
        let r_lc = _mm256_div_pd(l, c);
        let r_lo = _mm256_div_pd(l, o);

        let y_oret = _mm256_sub_pd(r_oret, one);
        let y_cret = _mm256_sub_pd(r_cret, one);
        let y_hc = _mm256_sub_pd(r_hc, one);
        let y_ho = _mm256_sub_pd(r_ho, one);
        let y_lc = _mm256_sub_pd(r_lc, one);
        let y_lo = _mm256_sub_pd(r_lo, one);

        let m_oret =
            _mm256_movemask_pd(_mm256_cmp_pd(_mm256_abs_pd(y_oret), threshold, _CMP_LT_OQ)) as u8;
        let m_cret =
            _mm256_movemask_pd(_mm256_cmp_pd(_mm256_abs_pd(y_cret), threshold, _CMP_LT_OQ)) as u8;
        let m_hc =
            _mm256_movemask_pd(_mm256_cmp_pd(_mm256_abs_pd(y_hc), threshold, _CMP_LT_OQ)) as u8;
        let m_ho =
            _mm256_movemask_pd(_mm256_cmp_pd(_mm256_abs_pd(y_ho), threshold, _CMP_LT_OQ)) as u8;
        let m_lc =
            _mm256_movemask_pd(_mm256_cmp_pd(_mm256_abs_pd(y_lc), threshold, _CMP_LT_OQ)) as u8;
        let m_lo =
            _mm256_movemask_pd(_mm256_cmp_pd(_mm256_abs_pd(y_lo), threshold, _CMP_LT_OQ)) as u8;

        let ln_oret = ln1p_taylor_14_avx2(y_oret);
        let ln_cret = ln1p_taylor_14_avx2(y_cret);
        let ln_hc = ln1p_taylor_14_avx2(y_hc);
        let ln_ho = ln1p_taylor_14_avx2(y_ho);
        let ln_lc = ln1p_taylor_14_avx2(y_lc);
        let ln_lo = ln1p_taylor_14_avx2(y_lo);

        _mm256_storeu_pd(oret.as_mut_ptr().add(i), ln_oret);
        _mm256_storeu_pd(cret.as_mut_ptr().add(i), ln_cret);
        let rs_v = _mm256_add_pd(_mm256_mul_pd(ln_hc, ln_ho), _mm256_mul_pd(ln_lc, ln_lo));
        _mm256_storeu_pd(rs_val.as_mut_ptr().add(i), rs_v);

        let m_rs = m_hc & m_ho & m_lc & m_lo;
        if m_oret == 0b1111 && m_cret == 0b1111 && m_rs == 0b1111 {
            i += 4;
            continue;
        }

        let mut oret_lanes = [0.0f64; 4];
        let mut cret_lanes = [0.0f64; 4];
        let mut hc_lanes = [0.0f64; 4];
        let mut ho_lanes = [0.0f64; 4];
        let mut lc_lanes = [0.0f64; 4];
        let mut lo_lanes = [0.0f64; 4];
        _mm256_storeu_pd(oret_lanes.as_mut_ptr(), ln_oret);
        _mm256_storeu_pd(cret_lanes.as_mut_ptr(), ln_cret);
        _mm256_storeu_pd(hc_lanes.as_mut_ptr(), ln_hc);
        _mm256_storeu_pd(ho_lanes.as_mut_ptr(), ln_ho);
        _mm256_storeu_pd(lc_lanes.as_mut_ptr(), ln_lc);
        _mm256_storeu_pd(lo_lanes.as_mut_ptr(), ln_lo);

        for lane in 0..4 {
            let idx = i + lane;
            let bit = 1u8 << lane;
            if (m_oret & bit) == 0 {
                oret_lanes[lane] = (open[idx] / close[idx - 1]).ln();
            }
            if (m_cret & bit) == 0 {
                cret_lanes[lane] = (close[idx] / open[idx]).ln();
            }
            if (m_hc & bit) == 0 {
                hc_lanes[lane] = (high[idx] / close[idx]).ln();
            }
            if (m_ho & bit) == 0 {
                ho_lanes[lane] = (high[idx] / open[idx]).ln();
            }
            if (m_lc & bit) == 0 {
                lc_lanes[lane] = (low[idx] / close[idx]).ln();
            }
            if (m_lo & bit) == 0 {
                lo_lanes[lane] = (low[idx] / open[idx]).ln();
            }

            oret[idx] = oret_lanes[lane];
            cret[idx] = cret_lanes[lane];
            rs_val[idx] = hc_lanes[lane] * ho_lanes[lane] + lc_lanes[lane] * lo_lanes[lane];
        }

        i += 4;
    }

    while i < len {
        let ln_open = open[i].ln();
        let ln_high = high[i].ln();
        let ln_low = low[i].ln();
        let ln_close = close[i].ln();

        oret[i] = (open[i] / close[i - 1]).ln();
        cret[i] = ln_close - ln_open;
        let h_c = ln_high - ln_close;
        let h_o = ln_high - ln_open;
        let l_c = ln_low - ln_close;
        let l_o = ln_low - ln_open;
        rs_val[i] = h_c * h_o + l_c * l_o;

        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
unsafe fn yang_zhang_precompute_ln_diff_avx512_fast(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    oret: &mut [f64],
    cret: &mut [f64],
    rs_val: &mut [f64],
) {
    let len = close.len();
    if len == 0 {
        return;
    }

    oret[0] = 0.0;
    cret[0] = (close[0] / open[0]).ln();
    rs_val[0] = rs_component(high[0], low[0], open[0], close[0]);

    let one = _mm512_set1_pd(1.0);
    let threshold = _mm512_set1_pd(0.1);

    let mut i = 1usize;
    while i + 8 <= len {
        let o = _mm512_loadu_pd(open.as_ptr().add(i));
        let h = _mm512_loadu_pd(high.as_ptr().add(i));
        let l = _mm512_loadu_pd(low.as_ptr().add(i));
        let c = _mm512_loadu_pd(close.as_ptr().add(i));
        let c_prev = _mm512_loadu_pd(close.as_ptr().add(i - 1));

        let r_oret = _mm512_div_pd(o, c_prev);
        let r_cret = _mm512_div_pd(c, o);

        let r_hc = _mm512_div_pd(h, c);
        let r_ho = _mm512_div_pd(h, o);
        let r_lc = _mm512_div_pd(l, c);
        let r_lo = _mm512_div_pd(l, o);

        let y_oret = _mm512_sub_pd(r_oret, one);
        let y_cret = _mm512_sub_pd(r_cret, one);
        let y_hc = _mm512_sub_pd(r_hc, one);
        let y_ho = _mm512_sub_pd(r_ho, one);
        let y_lc = _mm512_sub_pd(r_lc, one);
        let y_lo = _mm512_sub_pd(r_lo, one);

        let m_oret = _mm512_cmp_pd_mask(_mm512_abs_pd(y_oret), threshold, _CMP_LT_OQ);
        let m_cret = _mm512_cmp_pd_mask(_mm512_abs_pd(y_cret), threshold, _CMP_LT_OQ);
        let m_hc = _mm512_cmp_pd_mask(_mm512_abs_pd(y_hc), threshold, _CMP_LT_OQ);
        let m_ho = _mm512_cmp_pd_mask(_mm512_abs_pd(y_ho), threshold, _CMP_LT_OQ);
        let m_lc = _mm512_cmp_pd_mask(_mm512_abs_pd(y_lc), threshold, _CMP_LT_OQ);
        let m_lo = _mm512_cmp_pd_mask(_mm512_abs_pd(y_lo), threshold, _CMP_LT_OQ);

        let ln_oret = ln1p_taylor_14_avx512(y_oret);
        let ln_cret = ln1p_taylor_14_avx512(y_cret);

        let ln_hc = ln1p_taylor_14_avx512(y_hc);
        let ln_ho = ln1p_taylor_14_avx512(y_ho);
        let ln_lc = ln1p_taylor_14_avx512(y_lc);
        let ln_lo = ln1p_taylor_14_avx512(y_lo);

        let rs_v = _mm512_fmadd_pd(ln_hc, ln_ho, _mm512_mul_pd(ln_lc, ln_lo));
        _mm512_storeu_pd(oret.as_mut_ptr().add(i), ln_oret);
        _mm512_storeu_pd(cret.as_mut_ptr().add(i), ln_cret);
        _mm512_storeu_pd(rs_val.as_mut_ptr().add(i), rs_v);
        let m_rs = m_hc & m_ho & m_lc & m_lo;
        if m_oret == 0xFF && m_cret == 0xFF && m_rs == 0xFF {
            i += 8;
            continue;
        }

        if m_oret != 0xFF {
            let mut missing = !m_oret;
            while missing != 0 {
                let lane = missing.trailing_zeros() as usize;
                let bit = 1u8 << lane;
                let idx = i + lane;
                oret[idx] = (open[idx] / close[idx - 1]).ln();
                missing &= !bit;
            }
        }

        if m_cret != 0xFF {
            let mut missing = !m_cret;
            while missing != 0 {
                let lane = missing.trailing_zeros() as usize;
                let bit = 1u8 << lane;
                let idx = i + lane;
                cret[idx] = (close[idx] / open[idx]).ln();
                missing &= !bit;
            }
        }

        if m_rs != 0xFF {
            let mut hc_lanes = [0.0f64; 8];
            let mut ho_lanes = [0.0f64; 8];
            let mut lc_lanes = [0.0f64; 8];
            let mut lo_lanes = [0.0f64; 8];
            _mm512_storeu_pd(hc_lanes.as_mut_ptr(), ln_hc);
            _mm512_storeu_pd(ho_lanes.as_mut_ptr(), ln_ho);
            _mm512_storeu_pd(lc_lanes.as_mut_ptr(), ln_lc);
            _mm512_storeu_pd(lo_lanes.as_mut_ptr(), ln_lo);

            let mut missing = !m_rs;
            while missing != 0 {
                let lane = missing.trailing_zeros() as usize;
                let bit = 1u8 << lane;
                let idx = i + lane;

                if (m_hc & bit) == 0 {
                    hc_lanes[lane] = (high[idx] / close[idx]).ln();
                }
                if (m_ho & bit) == 0 {
                    ho_lanes[lane] = (high[idx] / open[idx]).ln();
                }
                if (m_lc & bit) == 0 {
                    lc_lanes[lane] = (low[idx] / close[idx]).ln();
                }
                if (m_lo & bit) == 0 {
                    lo_lanes[lane] = (low[idx] / open[idx]).ln();
                }
                rs_val[idx] = hc_lanes[lane] * ho_lanes[lane] + lc_lanes[lane] * lo_lanes[lane];
                missing &= !bit;
            }
        }

        i += 8;
    }

    while i < len {
        oret[i] = (open[i] / close[i - 1]).ln();
        cret[i] = (close[i] / open[i]).ln();
        rs_val[i] = rs_component(high[i], low[i], open[i], close[i]);
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn yang_zhang_precompute_ln_diff_avx512(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    oret: &mut [f64],
    cret: &mut [f64],
    rs_val: &mut [f64],
) {
    let len = close.len();
    if len == 0 {
        return;
    }

    if is_x86_feature_detected!("fma") {
        yang_zhang_precompute_ln_diff_avx512_fast(open, high, low, close, oret, cret, rs_val);
        return;
    }

    let ln_open0 = open[0].ln();
    let ln_high0 = high[0].ln();
    let ln_low0 = low[0].ln();
    let ln_close0 = close[0].ln();

    oret[0] = 0.0;
    cret[0] = ln_close0 - ln_open0;
    let h_c0 = ln_high0 - ln_close0;
    let h_o0 = ln_high0 - ln_open0;
    let l_c0 = ln_low0 - ln_close0;
    let l_o0 = ln_low0 - ln_open0;
    rs_val[0] = h_c0 * h_o0 + l_c0 * l_o0;

    let mut prev_ln_close = ln_close0;

    let mut i = 1usize;
    while i + 8 <= len {
        let ln_open_lanes = [
            open[i].ln(),
            open[i + 1].ln(),
            open[i + 2].ln(),
            open[i + 3].ln(),
            open[i + 4].ln(),
            open[i + 5].ln(),
            open[i + 6].ln(),
            open[i + 7].ln(),
        ];
        let ln_high_lanes = [
            high[i].ln(),
            high[i + 1].ln(),
            high[i + 2].ln(),
            high[i + 3].ln(),
            high[i + 4].ln(),
            high[i + 5].ln(),
            high[i + 6].ln(),
            high[i + 7].ln(),
        ];
        let ln_low_lanes = [
            low[i].ln(),
            low[i + 1].ln(),
            low[i + 2].ln(),
            low[i + 3].ln(),
            low[i + 4].ln(),
            low[i + 5].ln(),
            low[i + 6].ln(),
            low[i + 7].ln(),
        ];
        let ln_close_lanes = [
            close[i].ln(),
            close[i + 1].ln(),
            close[i + 2].ln(),
            close[i + 3].ln(),
            close[i + 4].ln(),
            close[i + 5].ln(),
            close[i + 6].ln(),
            close[i + 7].ln(),
        ];

        let lo = _mm512_loadu_pd(ln_open_lanes.as_ptr());
        let lh = _mm512_loadu_pd(ln_high_lanes.as_ptr());
        let ll = _mm512_loadu_pd(ln_low_lanes.as_ptr());
        let lc = _mm512_loadu_pd(ln_close_lanes.as_ptr());

        let h_c = _mm512_sub_pd(lh, lc);
        let h_o = _mm512_sub_pd(lh, lo);
        let l_c = _mm512_sub_pd(ll, lc);
        let l_o = _mm512_sub_pd(ll, lo);

        let rs_v = _mm512_add_pd(_mm512_mul_pd(h_c, h_o), _mm512_mul_pd(l_c, l_o));
        _mm512_storeu_pd(rs_val.as_mut_ptr().add(i), rs_v);

        let cr = _mm512_sub_pd(lc, lo);
        _mm512_storeu_pd(cret.as_mut_ptr().add(i), cr);

        for lane in 0..8 {
            let idx = i + lane;
            oret[idx] = ln_open_lanes[lane] - prev_ln_close;
            prev_ln_close = ln_close_lanes[lane];
        }

        i += 8;
    }

    while i < len {
        let ln_open = open[i].ln();
        let ln_high = high[i].ln();
        let ln_low = low[i].ln();
        let ln_close = close[i].ln();

        oret[i] = ln_open - prev_ln_close;
        prev_ln_close = ln_close;

        cret[i] = ln_close - ln_open;
        let h_c = ln_high - ln_close;
        let h_o = ln_high - ln_open;
        let l_c = ln_low - ln_close;
        let l_o = ln_low - ln_open;
        rs_val[i] = h_c * h_o + l_c * l_o;

        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn yang_zhang_compute_into_avx2(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    first: usize,
    k: f64,
    out_yz: &mut [f64],
    out_rs: &mut [f64],
) {
    let len = close.len();
    let mut rs_val = alloc_uninit_f64(len);
    let mut oret = alloc_uninit_f64(len);
    let mut cret = alloc_uninit_f64(len);
    yang_zhang_precompute_ln_diff_avx2(open, high, low, close, &mut oret, &mut cret, &mut rs_val);
    yang_zhang_row_precomputed_into(&oret, &cret, &rs_val, lookback, first, k, out_yz, out_rs);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn yang_zhang_compute_into_avx512(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    first: usize,
    k: f64,
    out_yz: &mut [f64],
    out_rs: &mut [f64],
) {
    let len = close.len();
    let mut rs_val = alloc_uninit_f64(len);
    let mut oret = alloc_uninit_f64(len);
    let mut cret = alloc_uninit_f64(len);
    yang_zhang_precompute_ln_diff_avx512(open, high, low, close, &mut oret, &mut cret, &mut rs_val);
    yang_zhang_row_precomputed_into(&oret, &cret, &rs_val, lookback, first, k, out_yz, out_rs);
}

#[inline(always)]
fn yang_zhang_compute_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    first: usize,
    k: f64,
    out_yz: &mut [f64],
    out_rs: &mut [f64],
) {
    let len = close.len();
    let warmup = first + lookback;
    if warmup >= len {
        return;
    }

    let mut rs_val = alloc_uninit_f64(len);
    let mut oret = alloc_uninit_f64(len);
    let mut cret = alloc_uninit_f64(len);
    yang_zhang_precompute_ln_diff_scalar(open, high, low, close, &mut oret, &mut cret, &mut rs_val);
    yang_zhang_row_precomputed_into(&oret, &cret, &rs_val, lookback, first, k, out_yz, out_rs);
}

#[inline]
pub fn yang_zhang_volatility_with_kernel(
    input: &YangZhangVolatilityInput,
    kernel: Kernel,
) -> Result<YangZhangVolatilityOutput, YangZhangVolatilityError> {
    let (open, high, low, close, lookback, first, chosen) = yang_zhang_prepare(input, kernel)?;

    let k = if input.get_k_override() {
        let k = input.get_k();
        if !k.is_finite() || !(0.0..=1.0).contains(&k) {
            return Err(YangZhangVolatilityError::InvalidK { k });
        }
        k
    } else {
        k_default(lookback)
    };

    let len = close.len();
    let warmup = first + lookback;
    let mut yz = alloc_with_nan_prefix(len, warmup);
    let mut rs = alloc_with_nan_prefix(len, warmup);

    unsafe {
        match chosen {
            Kernel::Scalar => yang_zhang_compute_into(
                open, high, low, close, lookback, first, k, &mut yz, &mut rs,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => yang_zhang_compute_into_avx2(
                open, high, low, close, lookback, first, k, &mut yz, &mut rs,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => yang_zhang_compute_into_avx512(
                open, high, low, close, lookback, first, k, &mut yz, &mut rs,
            ),
            #[allow(unreachable_patterns)]
            _ => yang_zhang_compute_into(
                open, high, low, close, lookback, first, k, &mut yz, &mut rs,
            ),
        }
    }

    Ok(YangZhangVolatilityOutput { yz, rs })
}

#[inline]
pub fn yang_zhang_volatility_into_slice(
    dst_yz: &mut [f64],
    dst_rs: &mut [f64],
    input: &YangZhangVolatilityInput,
    kern: Kernel,
) -> Result<(), YangZhangVolatilityError> {
    let (open, high, low, close, lookback, first, chosen) = yang_zhang_prepare(input, kern)?;
    let expected = close.len();
    if dst_yz.len() != expected || dst_rs.len() != expected {
        let got = dst_yz.len().max(dst_rs.len());
        return Err(YangZhangVolatilityError::OutputLengthMismatch { expected, got });
    }

    let k = if input.get_k_override() {
        let k = input.get_k();
        if !k.is_finite() || !(0.0..=1.0).contains(&k) {
            return Err(YangZhangVolatilityError::InvalidK { k });
        }
        k
    } else {
        k_default(lookback)
    };

    let warmup = first + lookback;
    let warm = warmup.min(expected);
    for v in &mut dst_yz[..warm] {
        *v = f64::NAN;
    }
    for v in &mut dst_rs[..warm] {
        *v = f64::NAN;
    }

    unsafe {
        match chosen {
            Kernel::Scalar => {
                yang_zhang_compute_into(open, high, low, close, lookback, first, k, dst_yz, dst_rs)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => yang_zhang_compute_into_avx2(
                open, high, low, close, lookback, first, k, dst_yz, dst_rs,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => yang_zhang_compute_into_avx512(
                open, high, low, close, lookback, first, k, dst_yz, dst_rs,
            ),
            #[allow(unreachable_patterns)]
            _ => {
                yang_zhang_compute_into(open, high, low, close, lookback, first, k, dst_yz, dst_rs)
            }
        }
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn yang_zhang_volatility_into(
    input: &YangZhangVolatilityInput,
    out_yz: &mut [f64],
    out_rs: &mut [f64],
) -> Result<(), YangZhangVolatilityError> {
    yang_zhang_volatility_into_slice(out_yz, out_rs, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct YangZhangVolatilityBatchRange {
    pub lookback: (usize, usize, usize),
    pub k_override: bool,
    pub k: (f64, f64, f64),
}

impl Default for YangZhangVolatilityBatchRange {
    fn default() -> Self {
        Self {
            lookback: (14, 252, 1),
            k_override: false,
            k: (0.34, 0.34, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct YangZhangVolatilityBatchBuilder {
    range: YangZhangVolatilityBatchRange,
    kernel: Kernel,
}

impl YangZhangVolatilityBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn lookback_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lookback = (start, end, step);
        self
    }

    #[inline]
    pub fn lookback_static(mut self, n: usize) -> Self {
        self.range.lookback = (n, n, 0);
        self
    }

    #[inline]
    pub fn k_override(mut self, v: bool) -> Self {
        self.range.k_override = v;
        self
    }

    #[inline]
    pub fn k_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.k = (start, end, step);
        self
    }

    #[inline]
    pub fn k_static(mut self, k: f64) -> Self {
        self.range.k = (k, k, 0.0);
        self
    }

    pub fn apply_slices(
        self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<YangZhangVolatilityBatchOutput, YangZhangVolatilityError> {
        yang_zhang_volatility_batch_with_kernel(open, high, low, close, &self.range, self.kernel)
    }

    pub fn with_default_slices(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        k: Kernel,
    ) -> Result<YangZhangVolatilityBatchOutput, YangZhangVolatilityError> {
        YangZhangVolatilityBatchBuilder::new()
            .kernel(k)
            .apply_slices(open, high, low, close)
    }

    pub fn apply_candles(
        self,
        c: &Candles,
    ) -> Result<YangZhangVolatilityBatchOutput, YangZhangVolatilityError> {
        self.apply_slices(&c.open, &c.high, &c.low, &c.close)
    }

    pub fn with_default_candles(
        c: &Candles,
    ) -> Result<YangZhangVolatilityBatchOutput, YangZhangVolatilityError> {
        YangZhangVolatilityBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct YangZhangVolatilityBatchOutput {
    pub yz: Vec<f64>,
    pub rs: Vec<f64>,
    pub combos: Vec<YangZhangVolatilityParams>,
    pub rows: usize,
    pub cols: usize,
}

impl YangZhangVolatilityBatchOutput {
    pub fn row_for_params(&self, p: &YangZhangVolatilityParams) -> Option<usize> {
        let lb = p.lookback.unwrap_or(14);
        let ko = p.k_override.unwrap_or(false);
        let k = p.k.unwrap_or(0.34);
        self.combos.iter().position(|c| {
            let clb = c.lookback.unwrap_or(14);
            let cko = c.k_override.unwrap_or(false);
            if clb != lb || cko != ko {
                return false;
            }
            if ko {
                (c.k.unwrap_or(0.34) - k).abs() < 1e-12
            } else {
                true
            }
        })
    }

    pub fn yz_for(&self, p: &YangZhangVolatilityParams) -> Option<&[f64]> {
        self.row_for_params(p).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.yz.get(start..start + self.cols))
        })
    }

    pub fn rs_for(&self, p: &YangZhangVolatilityParams) -> Option<&[f64]> {
        self.row_for_params(p).and_then(|row| {
            row.checked_mul(self.cols)
                .and_then(|start| self.rs.get(start..start + self.cols))
        })
    }
}

#[inline(always)]
fn expand_grid_yang_zhang(
    r: &YangZhangVolatilityBatchRange,
) -> Result<Vec<YangZhangVolatilityParams>, YangZhangVolatilityError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, YangZhangVolatilityError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        let st = step.max(1);
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            while x <= end {
                v.push(x);
                match x.checked_add(st) {
                    Some(next) => {
                        if next == x {
                            break;
                        }
                        x = next;
                    }
                    None => break,
                }
            }
            if v.is_empty() {
                return Err(YangZhangVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(v)
        } else {
            let mut v = Vec::new();
            let mut x = start;
            loop {
                v.push(x);
                if x == end {
                    break;
                }
                let next = x.saturating_sub(st);
                if next == x {
                    break;
                }
                x = next;
                if x < end {
                    break;
                }
            }
            if v.is_empty() {
                return Err(YangZhangVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(v)
        }
    }

    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, YangZhangVolatilityError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let st = step.abs();
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            while x <= end + 1e-12 {
                v.push(x);
                x += st;
            }
            if v.is_empty() {
                return Err(YangZhangVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(v)
        } else {
            let mut v = Vec::new();
            let mut x = start;
            while x >= end - 1e-12 {
                v.push(x);
                x -= st;
            }
            if v.is_empty() {
                return Err(YangZhangVolatilityError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(v)
        }
    }

    let lookbacks = axis_usize(r.lookback)?;
    let ks = if r.k_override {
        axis_f64(r.k)?
    } else {
        vec![r.k.0]
    };

    let mut out = Vec::with_capacity(lookbacks.len().saturating_mul(ks.len()));
    for &lb in &lookbacks {
        for &k in &ks {
            out.push(YangZhangVolatilityParams {
                lookback: Some(lb),
                k_override: Some(r.k_override),
                k: Some(k),
            });
        }
    }
    Ok(out)
}

pub fn yang_zhang_volatility_batch_with_kernel(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &YangZhangVolatilityBatchRange,
    k: Kernel,
) -> Result<YangZhangVolatilityBatchOutput, YangZhangVolatilityError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(YangZhangVolatilityError::InvalidKernelForBatch(k)),
    };
    yang_zhang_volatility_batch_par_slice(open, high, low, close, sweep, kernel.to_non_batch())
}

#[inline(always)]
pub fn yang_zhang_volatility_batch_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &YangZhangVolatilityBatchRange,
    kern: Kernel,
) -> Result<YangZhangVolatilityBatchOutput, YangZhangVolatilityError> {
    yang_zhang_volatility_batch_inner(open, high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn yang_zhang_volatility_batch_par_slice(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &YangZhangVolatilityBatchRange,
    kern: Kernel,
) -> Result<YangZhangVolatilityBatchOutput, YangZhangVolatilityError> {
    yang_zhang_volatility_batch_inner(open, high, low, close, sweep, kern, true)
}

#[inline(always)]
fn yang_zhang_volatility_batch_inner(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &YangZhangVolatilityBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<YangZhangVolatilityBatchOutput, YangZhangVolatilityError> {
    let combos = expand_grid_yang_zhang(sweep)?;
    let len = close.len();
    if len == 0 {
        return Err(YangZhangVolatilityError::EmptyInputData);
    }
    if open.len() != len || high.len() != len || low.len() != len {
        return Err(YangZhangVolatilityError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let first = first_valid_ohlc(open, high, low, close);
    if first >= len {
        return Err(YangZhangVolatilityError::AllValuesNaN);
    }

    let max_lb = combos
        .iter()
        .map(|c| c.lookback.unwrap_or(14))
        .max()
        .unwrap_or(0);

    if max_lb == 0 || len - first < max_lb + 1 {
        return Err(YangZhangVolatilityError::NotEnoughValidData {
            needed: max_lb + 1,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;

    let mut buf_yz_mu = make_uninit_matrix(rows, cols);
    let mut buf_rs_mu = make_uninit_matrix(rows, cols);

    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first.saturating_add(c.lookback.unwrap_or(14)))
        .collect();

    init_matrix_prefixes(&mut buf_yz_mu, cols, &warmup_periods);
    init_matrix_prefixes(&mut buf_rs_mu, cols, &warmup_periods);

    let mut buf_yz_guard = ManuallyDrop::new(buf_yz_mu);
    let mut buf_rs_guard = ManuallyDrop::new(buf_rs_mu);

    let yz: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_yz_guard.as_mut_ptr() as *mut f64, buf_yz_guard.len())
    };
    let rs: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_rs_guard.as_mut_ptr() as *mut f64, buf_rs_guard.len())
    };

    yang_zhang_volatility_batch_inner_into(open, high, low, close, sweep, kern, parallel, yz, rs)?;

    for (row, &warmup) in warmup_periods.iter().enumerate() {
        let row_start = row * cols;
        let warm_end = (row_start + warmup).min(row_start + cols);
        for i in row_start..warm_end {
            yz[i] = f64::NAN;
            rs[i] = f64::NAN;
        }
    }

    let yz_values = unsafe {
        Vec::from_raw_parts(
            buf_yz_guard.as_mut_ptr() as *mut f64,
            buf_yz_guard.len(),
            buf_yz_guard.capacity(),
        )
    };
    let rs_values = unsafe {
        Vec::from_raw_parts(
            buf_rs_guard.as_mut_ptr() as *mut f64,
            buf_rs_guard.len(),
            buf_rs_guard.capacity(),
        )
    };

    Ok(YangZhangVolatilityBatchOutput {
        yz: yz_values,
        rs: rs_values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn yang_zhang_row_precomputed_into(
    oret: &[f64],
    cret: &[f64],
    rs_val: &[f64],
    lookback: usize,
    first: usize,
    k: f64,
    out_yz: &mut [f64],
    out_rs: &mut [f64],
) {
    let len = out_yz.len();
    let warmup = first + lookback;
    if warmup >= len {
        return;
    }

    let lb_f = lookback as f64;
    let start = warmup;
    let win_start = start + 1 - lookback;

    let mut sum_rs = 0.0;
    let mut sum_o = 0.0;
    let mut sumsq_o = 0.0;
    let mut sum_c = 0.0;
    let mut sumsq_c = 0.0;

    for j in win_start..=start {
        let rsc = rs_val[j];
        sum_rs += rsc;

        let o = oret[j];
        sum_o += o;
        sumsq_o += o * o;

        let c = cret[j];
        sum_c += c;
        sumsq_c += c * c;
    }

    for t in start..len {
        let mut rs_var = sum_rs / lb_f;
        if rs_var < 0.0 {
            rs_var = 0.0;
        }
        out_rs[t] = rs_var.sqrt();

        let o_var = sample_var(sum_o, sumsq_o, lookback);
        let c_var = sample_var(sum_c, sumsq_c, lookback);

        let mut yz_var = o_var + k * c_var + (1.0 - k) * rs_var;
        if yz_var < 0.0 {
            yz_var = 0.0;
        }
        out_yz[t] = yz_var.sqrt();

        if t + 1 < len {
            let add_idx = t + 1;
            let sub_idx = add_idx - lookback;

            sum_rs += rs_val[add_idx] - rs_val[sub_idx];

            let ao = oret[add_idx];
            let so = oret[sub_idx];
            sum_o += ao - so;
            sumsq_o += ao * ao - so * so;

            let ac = cret[add_idx];
            let sc = cret[sub_idx];
            sum_c += ac - sc;
            sumsq_c += ac * ac - sc * sc;
        }
    }
}

#[inline(always)]
fn yang_zhang_volatility_batch_inner_into(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &YangZhangVolatilityBatchRange,
    kern: Kernel,
    parallel: bool,
    out_yz: &mut [f64],
    out_rs: &mut [f64],
) -> Result<Vec<YangZhangVolatilityParams>, YangZhangVolatilityError> {
    let combos = expand_grid_yang_zhang(sweep)?;

    let len = close.len();
    if len == 0 {
        return Err(YangZhangVolatilityError::EmptyInputData);
    }
    if open.len() != len || high.len() != len || low.len() != len {
        return Err(YangZhangVolatilityError::InconsistentSliceLengths {
            open_len: open.len(),
            high_len: high.len(),
            low_len: low.len(),
            close_len: close.len(),
        });
    }

    let first = first_valid_ohlc(open, high, low, close);
    if first >= len {
        return Err(YangZhangVolatilityError::AllValuesNaN);
    }

    let max_lb = combos
        .iter()
        .map(|c| c.lookback.unwrap_or(14))
        .max()
        .unwrap_or(0);

    if max_lb == 0 || len - first < max_lb + 1 {
        return Err(YangZhangVolatilityError::NotEnoughValidData {
            needed: max_lb + 1,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;
    let expected =
        rows.checked_mul(cols)
            .ok_or_else(|| YangZhangVolatilityError::InvalidRange {
                start: rows.to_string(),
                end: cols.to_string(),
                step: "rows*cols".into(),
            })?;

    if out_yz.len() != expected || out_rs.len() != expected {
        return Err(YangZhangVolatilityError::OutputLengthMismatch {
            expected,
            got: out_yz.len().max(out_rs.len()),
        });
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    let mut oret = alloc_uninit_f64(len);
    let mut cret = alloc_uninit_f64(len);
    let mut rs_val = alloc_uninit_f64(len);
    match chosen {
        Kernel::Scalar => {
            yang_zhang_precompute_ln_diff_scalar(
                open,
                high,
                low,
                close,
                &mut oret,
                &mut cret,
                &mut rs_val,
            );
        }
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2 => unsafe {
            yang_zhang_precompute_ln_diff_avx2(
                open,
                high,
                low,
                close,
                &mut oret,
                &mut cret,
                &mut rs_val,
            );
        },
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512 => unsafe {
            yang_zhang_precompute_ln_diff_avx512(
                open,
                high,
                low,
                close,
                &mut oret,
                &mut cret,
                &mut rs_val,
            );
        },
        #[allow(unreachable_patterns)]
        _ => {
            yang_zhang_precompute_ln_diff_scalar(
                open,
                high,
                low,
                close,
                &mut oret,
                &mut cret,
                &mut rs_val,
            );
        }
    }

    let warmup_periods: Vec<usize> = combos
        .iter()
        .map(|c| first.saturating_add(c.lookback.unwrap_or(14)))
        .collect();

    for (row, &warmup) in warmup_periods.iter().enumerate() {
        let row_start = row * cols;
        let warm = warmup.min(cols);
        out_yz[row_start..row_start + warm].fill(f64::NAN);
        out_rs[row_start..row_start + warm].fill(f64::NAN);
    }

    let do_row = |row: usize,
                  dst_yz: &mut [f64],
                  dst_rs: &mut [f64]|
     -> Result<(), YangZhangVolatilityError> {
        let lookback = combos[row].lookback.unwrap_or(14);
        if lookback == 0 {
            return Err(YangZhangVolatilityError::InvalidLookback {
                lookback,
                data_len: len,
            });
        }

        let ko = combos[row].k_override.unwrap_or(false);
        let k = if ko {
            let k = combos[row].k.unwrap_or(0.34);
            if !k.is_finite() || !(0.0..=1.0).contains(&k) {
                return Err(YangZhangVolatilityError::InvalidK { k });
            }
            k
        } else {
            k_default(lookback)
        };

        yang_zhang_row_precomputed_into(&oret, &cret, &rs_val, lookback, first, k, dst_yz, dst_rs);
        Ok(())
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_yz
                .par_chunks_mut(cols)
                .zip(out_rs.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(|(row, (y, r))| do_row(row, y, r))?;
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, (y, r)) in out_yz
                .chunks_mut(cols)
                .zip(out_rs.chunks_mut(cols))
                .enumerate()
            {
                do_row(row, y, r)?;
            }
        }
    } else {
        for (row, (y, r)) in out_yz
            .chunks_mut(cols)
            .zip(out_rs.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, y, r)?;
        }
    }

    for (row, &warmup) in warmup_periods.iter().enumerate() {
        let row_start = row * cols;
        let warm = warmup.min(cols);
        out_yz[row_start..row_start + warm].fill(f64::NAN);
        out_rs[row_start..row_start + warm].fill(f64::NAN);
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "yang_zhang_volatility")]
#[pyo3(signature = (open, high, low, close, lookback, k_override=false, k=0.34, kernel=None))]
pub fn yang_zhang_volatility_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    lookback: usize,
    k_override: bool,
    k: f64,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    if o.len() != h.len() || o.len() != l.len() || o.len() != c.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }

    let kern = validate_kernel(kernel, false)?;
    let params = YangZhangVolatilityParams {
        lookback: Some(lookback),
        k_override: Some(k_override),
        k: Some(k),
    };
    let input = YangZhangVolatilityInput::from_slices(o, h, l, c, params);

    let out = py
        .allow_threads(|| yang_zhang_volatility_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((out.yz.into_pyarray(py), out.rs.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "YangZhangVolatilityStream")]
pub struct YangZhangVolatilityStreamPy {
    stream: YangZhangVolatilityStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl YangZhangVolatilityStreamPy {
    #[new]
    fn new(lookback: usize, k_override: bool, k: f64) -> PyResult<Self> {
        let params = YangZhangVolatilityParams {
            lookback: Some(lookback),
            k_override: Some(k_override),
            k: Some(k),
        };
        let stream = YangZhangVolatilityStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        self.stream.update(open, high, low, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "yang_zhang_volatility_batch")]
#[pyo3(signature = (open, high, low, close, lookback_range, k_override=false, k_range=(0.34,0.34,0.0), kernel=None))]
pub fn yang_zhang_volatility_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    lookback_range: (usize, usize, usize),
    k_override: bool,
    k_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    if o.len() != h.len() || o.len() != l.len() || o.len() != c.len() {
        return Err(PyValueError::new_err("OHLC slice length mismatch"));
    }

    let sweep = YangZhangVolatilityBatchRange {
        lookback: lookback_range,
        k_override,
        k: k_range,
    };

    let combos =
        expand_grid_yang_zhang(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = c.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let yz_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let rs_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let yz_out = unsafe { yz_arr.as_slice_mut()? };
    let rs_out = unsafe { rs_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    py.allow_threads(|| {
        let batch = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        yang_zhang_volatility_batch_inner_into(
            o,
            h,
            l,
            c,
            &sweep,
            batch.to_non_batch(),
            true,
            yz_out,
            rs_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("yz", yz_arr.reshape((rows, cols))?)?;
    dict.set_item("rs", rs_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lookbacks",
        combos
            .iter()
            .map(|p| p.lookback.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "k_overrides",
        combos
            .iter()
            .map(|p| p.k_override.unwrap_or(false))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "ks",
        combos
            .iter()
            .map(|p| p.k.unwrap_or(0.34))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_yang_zhang_volatility_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(yang_zhang_volatility_py, m)?)?;
    m.add_function(wrap_pyfunction!(yang_zhang_volatility_batch_py, m)?)?;
    m.add_class::<YangZhangVolatilityStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "yang_zhang_volatility_js")]
pub fn yang_zhang_volatility_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    k_override: bool,
    k: f64,
) -> Result<JsValue, JsValue> {
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(JsValue::from_str("OHLC slice length mismatch"));
    }

    let params = YangZhangVolatilityParams {
        lookback: Some(lookback),
        k_override: Some(k_override),
        k: Some(k),
    };
    let input = YangZhangVolatilityInput::from_slices(open, high, low, close, params);

    let mut yz = vec![0.0; close.len()];
    let mut rs = vec![0.0; close.len()];
    yang_zhang_volatility_into_slice(&mut yz, &mut rs, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("yz"),
        &serde_wasm_bindgen::to_value(&yz).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rs"),
        &serde_wasm_bindgen::to_value(&rs).unwrap(),
    )?;

    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct YangZhangVolatilityBatchConfig {
    pub lookback_range: Vec<usize>,
    pub k_override: bool,
    pub k_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "yang_zhang_volatility_batch_js")]
pub fn yang_zhang_volatility_batch_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(JsValue::from_str("OHLC slice length mismatch"));
    }

    let config: YangZhangVolatilityBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;

    if config.lookback_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: lookback_range must have exactly 3 elements [start, end, step]",
        ));
    }
    if config.k_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: k_range must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = YangZhangVolatilityBatchRange {
        lookback: (
            config.lookback_range[0],
            config.lookback_range[1],
            config.lookback_range[2],
        ),
        k_override: config.k_override,
        k: (config.k_range[0], config.k_range[1], config.k_range[2]),
    };

    let combos = expand_grid_yang_zhang(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

    let mut yz = vec![0.0; total];
    let mut rs = vec![0.0; total];

    yang_zhang_volatility_batch_inner_into(
        open,
        high,
        low,
        close,
        &sweep,
        detect_best_kernel(),
        false,
        &mut yz,
        &mut rs,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("yz"),
        &serde_wasm_bindgen::to_value(&yz).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rs"),
        &serde_wasm_bindgen::to_value(&rs).unwrap(),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(rows as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(cols as f64),
    )?;
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("combos"),
        &serde_wasm_bindgen::to_value(&combos).unwrap(),
    )?;
    Ok(obj.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn yang_zhang_volatility_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(2 * len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn yang_zhang_volatility_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, 2 * len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn yang_zhang_volatility_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    lookback: usize,
    k_override: bool,
    k: f64,
) -> Result<(), JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to yang_zhang_volatility_into",
        ));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, 2 * len);

        let params = YangZhangVolatilityParams {
            lookback: Some(lookback),
            k_override: Some(k_override),
            k: Some(k),
        };
        let input = YangZhangVolatilityInput::from_slices(open, high, low, close, params);

        let (yz, rs) = out.split_at_mut(len);
        yang_zhang_volatility_into_slice(yz, rs, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn yang_zhang_volatility_batch_into(
    open_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    yz_ptr: *mut f64,
    rs_ptr: *mut f64,
    len: usize,
    lookback_start: usize,
    lookback_end: usize,
    lookback_step: usize,
    k_override: bool,
    k_start: f64,
    k_end: f64,
    k_step: f64,
) -> Result<usize, JsValue> {
    if open_ptr.is_null()
        || high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || yz_ptr.is_null()
        || rs_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer"));
    }

    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);

        let sweep = YangZhangVolatilityBatchRange {
            lookback: (lookback_start, lookback_end, lookback_step),
            k_override,
            k: (k_start, k_end, k_step),
        };

        let combos =
            expand_grid_yang_zhang(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;

        let yz_out = std::slice::from_raw_parts_mut(yz_ptr, rows * cols);
        let rs_out = std::slice::from_raw_parts_mut(rs_ptr, rows * cols);

        yang_zhang_volatility_batch_inner_into(
            open,
            high,
            low,
            close,
            &sweep,
            detect_best_kernel(),
            false,
            yz_out,
            rs_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn yang_zhang_volatility_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    lookback: usize,
    k_override: bool,
    k: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = yang_zhang_volatility_js(open, high, low, close, lookback, k_override, k)?;
    crate::write_wasm_object_f64_outputs("yang_zhang_volatility_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn yang_zhang_volatility_batch_output_into_js(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = yang_zhang_volatility_batch_js(open, high, low, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "yang_zhang_volatility_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    const TEST_FILE: &str = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";

    #[inline]
    fn eq_or_both_nan_eps(a: f64, b: f64, eps: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() <= eps
    }

    fn assert_series_close(test: &str, lhs: &[f64], rhs: &[f64], eps: f64, label: &str) {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "[{test}] {label} length mismatch: {} vs {}",
            lhs.len(),
            rhs.len()
        );
        for i in 0..lhs.len() {
            assert!(
                eq_or_both_nan_eps(lhs[i], rhs[i], eps),
                "[{test}] {label} mismatch at index {i}: {} vs {}",
                lhs[i],
                rhs[i]
            );
        }
    }

    fn check_yang_zhang_partial_params(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let candles = read_candles_from_csv(TEST_FILE)?;
        let params = YangZhangVolatilityParams {
            lookback: None,
            k_override: None,
            k: None,
        };
        let input = YangZhangVolatilityInput::from_candles(&candles, params);
        let out = yang_zhang_volatility_with_kernel(&input, kernel)?;
        assert_eq!(out.yz.len(), candles.close.len());
        assert_eq!(out.rs.len(), candles.close.len());
        Ok(())
    }

    fn check_yang_zhang_default_candles(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let candles = read_candles_from_csv(TEST_FILE)?;
        let input = YangZhangVolatilityInput::with_default_candles(&candles);
        match input.data {
            YangZhangVolatilityData::Candles { .. } => {}
            _ => panic!("Expected YangZhangVolatilityData::Candles"),
        }
        let out = yang_zhang_volatility_with_kernel(&input, kernel)?;
        assert_eq!(out.yz.len(), candles.close.len());
        assert_eq!(out.rs.len(), candles.close.len());
        Ok(())
    }

    fn check_yang_zhang_empty_input(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let empty: [f64; 0] = [];
        let input = YangZhangVolatilityInput::from_slices(
            &empty,
            &empty,
            &empty,
            &empty,
            YangZhangVolatilityParams::default(),
        );
        let res = yang_zhang_volatility_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(YangZhangVolatilityError::EmptyInputData)),
            "[{test}] expected EmptyInputData"
        );
        Ok(())
    }

    fn check_yang_zhang_inconsistent_slices(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let open = [10.0, 11.0, 12.0, 13.0];
        let high = [10.5, 11.5, 12.5];
        let low = [9.5, 10.5, 11.5, 12.5];
        let close = [10.2, 11.1, 12.3, 13.4];
        let input = YangZhangVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            YangZhangVolatilityParams::default(),
        );
        let res = yang_zhang_volatility_with_kernel(&input, kernel);
        assert!(
            matches!(
                res,
                Err(YangZhangVolatilityError::InconsistentSliceLengths { .. })
            ),
            "[{test}] expected InconsistentSliceLengths"
        );
        Ok(())
    }

    fn check_yang_zhang_invalid_lookback_zero(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let open = [10.0, 11.0, 12.0, 13.0];
        let high = [10.5, 11.5, 12.5, 13.5];
        let low = [9.5, 10.5, 11.5, 12.5];
        let close = [10.2, 11.1, 12.3, 13.4];
        let input = YangZhangVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            YangZhangVolatilityParams {
                lookback: Some(0),
                k_override: Some(false),
                k: Some(0.34),
            },
        );
        let res = yang_zhang_volatility_with_kernel(&input, kernel);
        assert!(
            matches!(
                res,
                Err(YangZhangVolatilityError::InvalidLookback { lookback: 0, .. })
            ),
            "[{test}] expected InvalidLookback(0)"
        );
        Ok(())
    }

    fn check_yang_zhang_invalid_lookback_exceeds_len(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let open = [10.0, 11.0, 12.0, 13.0];
        let high = [10.5, 11.5, 12.5, 13.5];
        let low = [9.5, 10.5, 11.5, 12.5];
        let close = [10.2, 11.1, 12.3, 13.4];
        let input = YangZhangVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            YangZhangVolatilityParams {
                lookback: Some(16),
                k_override: Some(false),
                k: Some(0.34),
            },
        );
        let res = yang_zhang_volatility_with_kernel(&input, kernel);
        assert!(
            matches!(
                res,
                Err(YangZhangVolatilityError::InvalidLookback { lookback: 16, .. })
            ),
            "[{test}] expected InvalidLookback(16)"
        );
        Ok(())
    }

    fn check_yang_zhang_invalid_k_override(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let open = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let high = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5];
        let low = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5];
        let close = [10.2, 11.1, 12.3, 13.4, 14.2, 15.1];
        let input = YangZhangVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            YangZhangVolatilityParams {
                lookback: Some(3),
                k_override: Some(true),
                k: Some(1.25),
            },
        );
        let res = yang_zhang_volatility_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(YangZhangVolatilityError::InvalidK { .. })),
            "[{test}] expected InvalidK"
        );
        Ok(())
    }

    fn check_yang_zhang_nan_handling(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let candles = read_candles_from_csv(TEST_FILE)?;
        let input = YangZhangVolatilityInput::with_default_candles(&candles);
        let out = yang_zhang_volatility_with_kernel(&input, kernel)?;
        let first = first_valid_ohlc(&candles.open, &candles.high, &candles.low, &candles.close);
        let warmup = first + input.get_lookback();
        for i in warmup..out.yz.len() {
            assert!(!out.yz[i].is_nan(), "[{test}] yz NaN at {i}");
            assert!(!out.rs[i].is_nan(), "[{test}] rs NaN at {i}");
        }
        Ok(())
    }

    fn check_yang_zhang_into_slice_matches_api(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let candles = read_candles_from_csv(TEST_FILE)?;
        let params = YangZhangVolatilityParams {
            lookback: Some(20),
            k_override: Some(true),
            k: Some(0.42),
        };
        let input = YangZhangVolatilityInput::from_candles(&candles, params);

        let baseline = yang_zhang_volatility_with_kernel(&input, kernel)?;

        let mut yz = vec![0.0; candles.close.len()];
        let mut rs = vec![0.0; candles.close.len()];
        yang_zhang_volatility_into_slice(&mut yz, &mut rs, &input, kernel)?;

        assert_series_close(test, &baseline.yz, &yz, 1e-12, "into_slice yz");
        assert_series_close(test, &baseline.rs, &rs, 1e-12, "into_slice rs");

        let mut yz_short = vec![0.0; candles.close.len().saturating_sub(1)];
        let mut rs_ok = vec![0.0; candles.close.len()];
        let err = yang_zhang_volatility_into_slice(&mut yz_short, &mut rs_ok, &input, kernel)
            .expect_err("expected OutputLengthMismatch");
        assert!(matches!(
            err,
            YangZhangVolatilityError::OutputLengthMismatch { .. }
        ));
        Ok(())
    }

    fn check_yang_zhang_streaming(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let candles = read_candles_from_csv(TEST_FILE)?;
        let params = YangZhangVolatilityParams {
            lookback: Some(20),
            k_override: Some(true),
            k: Some(0.42),
        };
        let input = YangZhangVolatilityInput::from_candles(&candles, params.clone());
        let batch = yang_zhang_volatility_with_kernel(&input, kernel)?;

        let mut stream = YangZhangVolatilityStream::try_new(params)?;
        let mut yz_stream = Vec::with_capacity(candles.close.len());
        let mut rs_stream = Vec::with_capacity(candles.close.len());
        for i in 0..candles.close.len() {
            match stream.update(
                candles.open[i],
                candles.high[i],
                candles.low[i],
                candles.close[i],
            ) {
                Some((yz, rs)) => {
                    yz_stream.push(yz);
                    rs_stream.push(rs);
                }
                None => {
                    yz_stream.push(f64::NAN);
                    rs_stream.push(f64::NAN);
                }
            }
        }

        let eps = match kernel {
            Kernel::Avx512 => 1e-10,
            _ => 1e-10,
        };
        assert_series_close(test, &batch.yz, &yz_stream, eps, "stream yz");
        assert_series_close(test, &batch.rs, &rs_stream, eps, "stream rs");
        Ok(())
    }

    fn check_yang_zhang_matches_naive(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let candles = read_candles_from_csv(TEST_FILE)?;
        let input = YangZhangVolatilityInput::with_default_candles(&candles);
        let out = yang_zhang_volatility_with_kernel(&input, kernel)?;

        let len = candles.close.len();
        assert_eq!(out.yz.len(), len);
        assert_eq!(out.rs.len(), len);

        let lookback = input.get_lookback();
        let k = k_default(lookback);

        let open = &candles.open;
        let high = &candles.high;
        let low = &candles.low;
        let close = &candles.close;

        let first = first_valid_ohlc(open, high, low, close);
        let warmup = first + lookback;

        let mut expected_yz = vec![f64::NAN; len];
        let mut expected_rs = vec![f64::NAN; len];

        for t in warmup..len {
            let start = t + 1 - lookback;

            let mut sum_rs = 0.0;
            let mut sum_o = 0.0;
            let mut sumsq_o = 0.0;
            let mut sum_c = 0.0;
            let mut sumsq_c = 0.0;

            for j in start..=t {
                let rsc = rs_component(high[j], low[j], open[j], close[j]);
                sum_rs += rsc;

                let oret = (open[j] / close[j - 1]).ln();
                sum_o += oret;
                sumsq_o += oret * oret;

                let cret = (close[j] / open[j]).ln();
                sum_c += cret;
                sumsq_c += cret * cret;
            }

            let mut rs_var = sum_rs / (lookback as f64);
            if rs_var < 0.0 {
                rs_var = 0.0;
            }
            expected_rs[t] = rs_var.sqrt();

            let o_var = sample_var(sum_o, sumsq_o, lookback);
            let c_var = sample_var(sum_c, sumsq_c, lookback);

            let mut yz_var = o_var + k * c_var + (1.0 - k) * rs_var;
            if yz_var < 0.0 {
                yz_var = 0.0;
            }
            expected_yz[t] = yz_var.sqrt();
        }

        for i in 0..len {
            let a = out.yz[i];
            let e = expected_yz[i];
            if a.is_nan() && e.is_nan() {
                continue;
            }
            let eps = match kernel {
                Kernel::Avx512 => 1e-10,
                _ => 1e-12,
            };
            assert!(
                (a - e).abs() <= eps,
                "[{test}] YZ mismatch at index {}: expected {}, got {}",
                i,
                e,
                a
            );

            let a = out.rs[i];
            let e = expected_rs[i];
            if a.is_nan() && e.is_nan() {
                continue;
            }
            assert!(
                (a - e).abs() <= eps,
                "[{test}] RS mismatch at index {}: expected {}, got {}",
                i,
                e,
                a
            );
        }
        Ok(())
    }

    fn check_yang_zhang_near_one_accuracy(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let len = 256usize;
        let gap_cases = [
            1e-12, -1e-12, 1e-9, -1e-9, 1e-6, -1e-6, 1e-4, -1e-4, 1e-2, -1e-2, 5e-2, -5e-2, 0.19,
            -0.19, 0.24, -0.24,
        ];
        let body_cases = [
            -1e-12, 1e-12, -1e-8, 1e-8, -1e-5, 1e-5, -1e-3, 1e-3, -2e-2, 2e-2, -0.08, 0.08, -0.19,
            0.19, -0.24, 0.24,
        ];
        let wick_cases = [1e-12, 1e-9, 1e-6, 1e-4, 1e-2, 3e-2, 0.15];

        let mut open: Vec<f64> = vec![0.0; len];
        let mut high: Vec<f64> = vec![0.0; len];
        let mut low: Vec<f64> = vec![0.0; len];
        let mut close: Vec<f64> = vec![0.0; len];

        open[0] = 100.0;
        close[0] = 100.05;
        high[0] = 100.10;
        low[0] = 99.95;

        for i in 1..len {
            let gap = gap_cases[i % gap_cases.len()];
            let body = body_cases[(i * 3) % body_cases.len()];
            let upper_wick = wick_cases[(i * 5) % wick_cases.len()];
            let lower_wick = wick_cases[(i * 7) % wick_cases.len()];

            open[i] = close[i - 1] * (1.0 + gap);
            close[i] = open[i] * (1.0 + body);

            let top = open[i].max(close[i]);
            let bottom = open[i].min(close[i]);
            high[i] = top * (1.0 + upper_wick);
            low[i] = bottom * (1.0 - lower_wick);
        }

        let params = YangZhangVolatilityParams {
            lookback: Some(14),
            k_override: Some(true),
            k: Some(0.42),
        };
        let input = YangZhangVolatilityInput::from_slices(&open, &high, &low, &close, params);
        let scalar = yang_zhang_volatility_with_kernel(&input, Kernel::Scalar)?;
        let simd = yang_zhang_volatility_with_kernel(&input, kernel)?;

        assert_series_close(test, &scalar.yz, &simd.yz, 1e-8, "near-one yz");
        assert_series_close(test, &scalar.rs, &simd.rs, 1e-8, "near-one rs");
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_yang_zhang_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let candles = read_candles_from_csv(TEST_FILE)?;
        let configs = [
            YangZhangVolatilityParams::default(),
            YangZhangVolatilityParams {
                lookback: Some(2),
                k_override: Some(true),
                k: Some(0.0),
            },
            YangZhangVolatilityParams {
                lookback: Some(5),
                k_override: Some(true),
                k: Some(1.0),
            },
            YangZhangVolatilityParams {
                lookback: Some(30),
                k_override: Some(false),
                k: Some(0.34),
            },
            YangZhangVolatilityParams {
                lookback: Some(80),
                k_override: Some(true),
                k: Some(0.5),
            },
        ];

        for params in configs {
            let input = YangZhangVolatilityInput::from_candles(&candles, params);
            let out = yang_zhang_volatility_with_kernel(&input, kernel)?;
            for &v in out.yz.iter().chain(out.rs.iter()) {
                if v.is_nan() {
                    continue;
                }
                let bits = v.to_bits();
                assert_ne!(
                    bits, 0x11111111_11111111,
                    "[{test}] poison 0x11 encountered"
                );
                assert_ne!(
                    bits, 0x22222222_22222222,
                    "[{test}] poison 0x22 encountered"
                );
                assert_ne!(
                    bits, 0x33333333_33333333,
                    "[{test}] poison 0x33 encountered"
                );
            }
        }
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_yang_zhang_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_yang_zhang_tests {
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

    generate_all_yang_zhang_tests!(
        check_yang_zhang_partial_params,
        check_yang_zhang_default_candles,
        check_yang_zhang_empty_input,
        check_yang_zhang_inconsistent_slices,
        check_yang_zhang_invalid_lookback_zero,
        check_yang_zhang_invalid_lookback_exceeds_len,
        check_yang_zhang_invalid_k_override,
        check_yang_zhang_nan_handling,
        check_yang_zhang_into_slice_matches_api,
        check_yang_zhang_streaming,
        check_yang_zhang_matches_naive,
        check_yang_zhang_near_one_accuracy,
        check_yang_zhang_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let candles = read_candles_from_csv(TEST_FILE)?;
        let batch = YangZhangVolatilityBatchBuilder::new()
            .kernel(kernel)
            .lookback_range(14, 16, 1)
            .k_override(false)
            .k_static(0.34)
            .apply_candles(&candles)?;

        let def = YangZhangVolatilityParams::default();
        let yz = batch.yz_for(&def).expect("default yz row missing");
        let rs = batch.rs_for(&def).expect("default rs row missing");

        let input = YangZhangVolatilityInput::from_candles(&candles, def);
        let single = yang_zhang_volatility_with_kernel(&input, Kernel::Scalar)?;

        assert_series_close(test, yz, &single.yz, 1e-12, "batch default yz");
        assert_series_close(test, rs, &single.rs, 1e-12, "batch default rs");
        Ok(())
    }

    fn check_batch_sweep_vs_single(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let candles = read_candles_from_csv(TEST_FILE)?;
        let batch = YangZhangVolatilityBatchBuilder::new()
            .kernel(kernel)
            .lookback_range(8, 12, 2)
            .k_override(true)
            .k_range(0.2, 0.6, 0.2)
            .apply_slices(&candles.open, &candles.high, &candles.low, &candles.close)?;

        assert_eq!(batch.rows, batch.combos.len());
        assert_eq!(batch.cols, candles.close.len());

        for params in &batch.combos {
            let row = batch.row_for_params(params).expect("row missing");
            let start = row * batch.cols;
            let end = start + batch.cols;

            let input = YangZhangVolatilityInput::from_candles(&candles, params.clone());
            let single = yang_zhang_volatility_with_kernel(&input, Kernel::Scalar)?;

            assert_series_close(
                test,
                &batch.yz[start..end],
                &single.yz,
                1e-10,
                "batch sweep yz",
            );
            assert_series_close(
                test,
                &batch.rs[start..end],
                &single.rs,
                1e-10,
                "batch sweep rs",
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let candles = read_candles_from_csv(TEST_FILE)?;
        let configs = [
            (2, 10, 2, true, 0.0, 1.0, 0.2),
            (5, 25, 5, false, 0.34, 0.34, 0.0),
            (10, 40, 10, true, 0.2, 0.8, 0.3),
            (8, 12, 1, true, 0.4, 0.4, 0.0),
        ];

        for (lb_s, lb_e, lb_st, ko, k_s, k_e, k_st) in configs {
            let out = YangZhangVolatilityBatchBuilder::new()
                .kernel(kernel)
                .lookback_range(lb_s, lb_e, lb_st)
                .k_override(ko)
                .k_range(k_s, k_e, k_st)
                .apply_candles(&candles)?;
            for &v in out.yz.iter().chain(out.rs.iter()) {
                if v.is_nan() {
                    continue;
                }
                let bits = v.to_bits();
                assert_ne!(
                    bits, 0x11111111_11111111,
                    "[{test}] poison 0x11 encountered"
                );
                assert_ne!(
                    bits, 0x22222222_22222222,
                    "[{test}] poison 0x22 encountered"
                );
                assert_ne!(
                    bits, 0x33333333_33333333,
                    "[{test}] poison 0x33 encountered"
                );
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
            paste::paste! {
                #[test]
                fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test]
                fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep_vs_single);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_batch_invalid_kernel_for_batch() -> Result<(), Box<dyn Error>> {
        let candles = read_candles_from_csv(TEST_FILE)?;
        let sweep = YangZhangVolatilityBatchRange {
            lookback: (8, 12, 2),
            k_override: true,
            k: (0.2, 0.6, 0.2),
        };
        let err = yang_zhang_volatility_batch_with_kernel(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
            &sweep,
            Kernel::Scalar,
        )
        .expect_err("expected InvalidKernelForBatch");
        assert!(matches!(
            err,
            YangZhangVolatilityError::InvalidKernelForBatch(Kernel::Scalar)
        ));
        Ok(())
    }

    #[test]
    fn test_batch_inner_into_output_length_mismatch() -> Result<(), Box<dyn Error>> {
        let candles = read_candles_from_csv(TEST_FILE)?;
        let sweep = YangZhangVolatilityBatchRange {
            lookback: (8, 12, 2),
            k_override: true,
            k: (0.2, 0.6, 0.2),
        };
        let combos = expand_grid_yang_zhang(&sweep)?;
        let rows = combos.len();
        let cols = candles.close.len();
        let expected = rows * cols;
        let mut yz = vec![0.0; expected.saturating_sub(1)];
        let mut rs = vec![0.0; expected];
        let err = yang_zhang_volatility_batch_inner_into(
            &candles.open,
            &candles.high,
            &candles.low,
            &candles.close,
            &sweep,
            Kernel::Scalar,
            false,
            &mut yz,
            &mut rs,
        )
        .expect_err("expected OutputLengthMismatch");
        assert!(matches!(
            err,
            YangZhangVolatilityError::OutputLengthMismatch { .. }
        ));
        Ok(())
    }

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    #[test]
    fn test_yang_zhang_into_matches_api() -> Result<(), Box<dyn Error>> {
        let candles = read_candles_from_csv(TEST_FILE)?;
        let params = YangZhangVolatilityParams {
            lookback: Some(14),
            k_override: Some(true),
            k: Some(0.34),
        };
        let input = YangZhangVolatilityInput::from_candles(&candles, params);
        let baseline = yang_zhang_volatility(&input)?;

        let mut yz = vec![0.0; candles.close.len()];
        let mut rs = vec![0.0; candles.close.len()];
        yang_zhang_volatility_into(&input, &mut yz, &mut rs)?;

        assert_series_close(
            "test_yang_zhang_into_matches_api",
            &baseline.yz,
            &yz,
            1e-12,
            "into yz",
        );
        assert_series_close(
            "test_yang_zhang_into_matches_api",
            &baseline.rs,
            &rs,
            1e-12,
            "into rs",
        );
        Ok(())
    }

    #[test]
    fn test_yang_zhang_builder_apply_matches_slices() -> Result<(), Box<dyn Error>> {
        let candles = read_candles_from_csv(TEST_FILE)?;
        let b = YangZhangVolatilityBuilder::new()
            .lookback(20)
            .k_override(true)
            .k(0.42)
            .kernel(Kernel::Scalar);
        let by_candles = b.apply(&candles)?;
        let by_slices =
            b.apply_slices(&candles.open, &candles.high, &candles.low, &candles.close)?;
        assert_series_close(
            "test_yang_zhang_builder_apply_matches_slices",
            &by_candles.yz,
            &by_slices.yz,
            1e-12,
            "builder yz",
        );
        assert_series_close(
            "test_yang_zhang_builder_apply_matches_slices",
            &by_candles.rs,
            &by_slices.rs,
            1e-12,
            "builder rs",
        );
        Ok(())
    }
}
