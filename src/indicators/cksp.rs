//! # Chande Kroll Stop (CKSP)
//!
//! Computes two stop lines (long and short) using ATR and rolling maxima/minima.
//! Its parameters (`p`, `x`, `q`) control the ATR period, ATR multiplier, and rolling window size.
//!
//! ## Parameters
//! - **p**: ATR period (default: 10)
//! - **x**: ATR multiplier (default: 1.0)
//! - **q**: Rolling window (default: 9)
//!
//! ## Errors
//! - **NoData**: cksp: Data is empty or all values NaN
//! - **NotEnoughData**: cksp: Not enough data for the provided parameters
//! - **InconsistentLengths**: cksp: Input slices have different lengths
//! - **InvalidParam**: cksp: Parameter(s) invalid (e.g., zero/NaN/negative)
//!
//! ## Returns
//! - **`Ok(CkspOutput)`** on success, containing two `Vec<f64>` of length matching the input
//! - **`Err(CkspError)`** otherwise
//!
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

// ========================= Input Structs, AsRef =========================

#[derive(Debug, Clone)]
pub enum CkspData<'a> {
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
pub struct CkspParams {
    pub p: Option<usize>,
    pub x: Option<f64>,
    pub q: Option<usize>,
}

impl Default for CkspParams {
    fn default() -> Self {
        Self {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CkspInput<'a> {
    pub data: CkspData<'a>,
    pub params: CkspParams,
}

impl<'a> CkspInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: CkspParams) -> Self {
        Self {
            data: CkspData::Candles { candles },
            params,
        }
    }
    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: CkspParams,
    ) -> Self {
        Self {
            data: CkspData::Slices { high, low, close },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, CkspParams::default())
    }
    #[inline]
    pub fn get_p(&self) -> usize {
        self.params.p.unwrap_or(10)
    }
    #[inline]
    pub fn get_x(&self) -> f64 {
        self.params.x.unwrap_or(1.0)
    }
    #[inline]
    pub fn get_q(&self) -> usize {
        self.params.q.unwrap_or(9)
    }
}

impl<'a> AsRef<[f64]> for CkspInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            CkspData::Candles { candles } => &candles.close,
            CkspData::Slices { close, .. } => close,
        }
    }
}

// ========================= Output Struct =========================

#[derive(Debug, Clone)]
pub struct CkspOutput {
    pub long_values: Vec<f64>,
    pub short_values: Vec<f64>,
}

// ========================= Error Type =========================

#[derive(Debug, Error)]
pub enum CkspError {
    #[error("cksp: Data is empty or all values are NaN.")]
    NoData,
    #[error("cksp: Not enough data for p={p}, q={q}, data_len={data_len}.")]
    NotEnoughData { p: usize, q: usize, data_len: usize },
    #[error("cksp: Inconsistent input lengths.")]
    InconsistentLengths,
    #[error("cksp: Invalid param value: {param}")]
    InvalidParam { param: &'static str },
    #[error("cksp: Candle field error: {0}")]
    CandleFieldError(String),
}

// ========================= Builder Struct =========================

#[derive(Copy, Clone, Debug)]
pub struct CkspBuilder {
    p: Option<usize>,
    x: Option<f64>,
    q: Option<usize>,
    kernel: Kernel,
}

impl Default for CkspBuilder {
    fn default() -> Self {
        Self {
            p: None,
            x: None,
            q: None,
            kernel: Kernel::Auto,
        }
    }
}

impl CkspBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn p(mut self, n: usize) -> Self {
        self.p = Some(n);
        self
    }
    #[inline(always)]
    pub fn x(mut self, v: f64) -> Self {
        self.x = Some(v);
        self
    }
    #[inline(always)]
    pub fn q(mut self, n: usize) -> Self {
        self.q = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<CkspOutput, CkspError> {
        let params = CkspParams {
            p: self.p,
            x: self.x,
            q: self.q,
        };
        let input = CkspInput::from_candles(candles, params);
        cksp_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<CkspOutput, CkspError> {
        let params = CkspParams {
            p: self.p,
            x: self.x,
            q: self.q,
        };
        let input = CkspInput::from_slices(high, low, close, params);
        cksp_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<CkspStream, CkspError> {
        let params = CkspParams {
            p: self.p,
            x: self.x,
            q: self.q,
        };
        CkspStream::try_new(params)
    }
}

// ========================= Main Indicator Functions =========================

#[inline]
pub fn cksp(input: &CkspInput) -> Result<CkspOutput, CkspError> {
    cksp_with_kernel(input, Kernel::Auto)
}

pub fn cksp_with_kernel(input: &CkspInput, kernel: Kernel) -> Result<CkspOutput, CkspError> {
    let (high, low, close) = match &input.data {
        CkspData::Candles { candles } => {
            let h = candles
                .select_candle_field("high")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
            let l = candles
                .select_candle_field("low")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
            let c = candles
                .select_candle_field("close")
                .map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
            (h, l, c)
        }
        CkspData::Slices { high, low, close } => {
            if high.len() != low.len() || low.len() != close.len() {
                return Err(CkspError::InconsistentLengths);
            }
            (*high, *low, *close)
        }
    };
    let p = input.get_p();
    let x = input.get_x();
    let q = input.get_q();
    let size = close.len();

    if size == 0 {
        return Err(CkspError::NoData);
    }
    if p == 0 || q == 0 {
        return Err(CkspError::InvalidParam { param: "p/q" });
    }
    if p > size || q > size {
        return Err(CkspError::NotEnoughData { p, q, data_len: size });
    }
    if !(x.is_finite()) || x.is_nan() {
        return Err(CkspError::InvalidParam { param: "x" });
    }

    let first_valid_idx = match close.iter().position(|&v| !v.is_nan()) {
        Some(idx) => idx,
        None => return Err(CkspError::NoData),
    };

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                cksp_scalar(high, low, close, p, x, q, first_valid_idx)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                cksp_avx2(high, low, close, p, x, q, first_valid_idx)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                cksp_avx512(high, low, close, p, x, q, first_valid_idx)
            }
            _ => unreachable!(),
        }
    }
}

// ========================= Scalar Logic =========================

#[inline]
pub unsafe fn cksp_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
) -> Result<CkspOutput, CkspError> {
    let size = close.len();
    let mut long_values = vec![f64::NAN; size];
    let mut short_values = vec![f64::NAN; size];
    let mut atr = vec![0.0; size];
    let mut sum_tr = 0.0;
    let mut rma = 0.0;
    let alpha = 1.0 / (p as f64);
    let mut dq_h = std::collections::VecDeque::<(usize, f64)>::new();
    let mut dq_ls0 = std::collections::VecDeque::<(usize, f64)>::new();
    let mut dq_l = std::collections::VecDeque::<(usize, f64)>::new();
    let mut dq_ss0 = std::collections::VecDeque::<(usize, f64)>::new();

    for i in 0..size {
        if i < first_valid_idx {
            continue;
        }
        let tr = if i == first_valid_idx {
            high[i] - low[i]
        
            } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };
        if i - first_valid_idx < p {
            sum_tr += tr;
            if i - first_valid_idx == p - 1 {
                rma = sum_tr / p as f64;
                atr[i] = rma;
} else {
            rma += alpha * (tr - rma);
            atr[i] = rma;
        }
        while let Some((_, v)) = dq_h.back() {
            if *v <= high[i] {
                dq_h.pop_back();
            
                } else {
                break;
            }
        }
        dq_h.push_back((i, high[i]));
        let start_h = i.saturating_sub(q - 1);
        while let Some(&(idx, _)) = dq_h.front() {
            if idx < start_h {
                dq_h.pop_front();
            
                } else {
                break;
            }
        }
        while let Some((_, v)) = dq_l.back() {
            if *v >= low[i] {
                dq_l.pop_back();
            
                } else {
                break;
            }
        }
        dq_l.push_back((i, low[i]));
        let start_l = i.saturating_sub(q - 1);
        while let Some(&(idx, _)) = dq_l.front() {
            if idx < start_l {
                dq_l.pop_front();
            
                } else {
                break;
            }
        }
        if atr[i] != 0.0 && i >= first_valid_idx + p - 1 {
            if let (Some(&(_, mh)), Some(&(_, ml))) = (dq_h.front(), dq_l.front()) {
                let ls0_val = mh - x * atr[i];
                let ss0_val = ml + x * atr[i];
                while let Some((_, val)) = dq_ls0.back() {
                    if *val <= ls0_val {
                        dq_ls0.pop_back();
                    
                        }
                         else {
                        break;
                    }
                }
                dq_ls0.push_back((i, ls0_val));
                let start_ls0 = i.saturating_sub(q - 1);
                while let Some(&(idx, _)) = dq_ls0.front() {
                    if idx < start_ls0 {
                        dq_ls0.pop_front();
                    
                        } else {
                        break;
                    }
                }
                if let Some(&(_, mx)) = dq_ls0.front() {
                    long_values[i] = mx;
                }
                while let Some((_, val)) = dq_ss0.back() {
                    if *val >= ss0_val {
                        dq_ss0.pop_back();
                    
                        } else {
                        break;
                    }
                }
                dq_ss0.push_back((i, ss0_val));
                let start_ss0 = i.saturating_sub(q - 1);
                while let Some(&(idx, _)) = dq_ss0.front() {
                    if idx < start_ss0 {
                        dq_ss0.pop_front();
                    
                        } else {
                        break;
                    }
                }
                if let Some(&(_, mn)) = dq_ss0.front() {
                    short_values[i] = mn;
                }
            }
        }
    }
    Ok(CkspOutput {
        long_values,
        short_values,
    })
}

// ========================= AVX2/AVX512 Stubs =========================

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cksp_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
) -> Result<CkspOutput, CkspError> {
    // For API parity, fallback to scalar
    cksp_scalar(high, low, close, p, x, q, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cksp_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
) -> Result<CkspOutput, CkspError> {
    // For API parity, fallback to scalar
    cksp_scalar(high, low, close, p, x, q, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cksp_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
) -> Result<CkspOutput, CkspError> {
    cksp_avx512(high, low, close, p, x, q, first_valid_idx)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn cksp_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
) -> Result<CkspOutput, CkspError> {
    cksp_avx512(high, low, close, p, x, q, first_valid_idx)
}

// ========================= Row/Batched API =========================

#[inline(always)]
pub unsafe fn cksp_row_scalar(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
    out_long: &mut [f64],
    out_short: &mut [f64],
) {
    let out = cksp_scalar(high, low, close, p, x, q, first_valid_idx).unwrap();
    out_long.copy_from_slice(&out.long_values);
    out_short.copy_from_slice(&out.short_values);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cksp_row_avx2(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
    out_long: &mut [f64],
    out_short: &mut [f64],
) {
    cksp_row_scalar(high, low, close, p, x, q, first_valid_idx, out_long, out_short)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cksp_row_avx512(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
    out_long: &mut [f64],
    out_short: &mut [f64],
) {
    cksp_row_scalar(high, low, close, p, x, q, first_valid_idx, out_long, out_short)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cksp_row_avx512_short(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
    out_long: &mut [f64],
    out_short: &mut [f64],
) {
    cksp_row_avx512(high, low, close, p, x, q, first_valid_idx, out_long, out_short)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn cksp_row_avx512_long(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    p: usize,
    x: f64,
    q: usize,
    first_valid_idx: usize,
    out_long: &mut [f64],
    out_short: &mut [f64],
) {
    cksp_row_avx512(high, low, close, p, x, q, first_valid_idx, out_long, out_short)
}

// ========================= Stream API =========================

use std::collections::VecDeque;
use crate::indicators::atr::{AtrParams, AtrStream};

#[derive(Debug, Clone)]
pub struct CkspStream {
    p: usize,
    x: f64,
    q: usize,
    alpha: f64,
    sum_tr: f64,
    rma: f64,
    prev_close: f64,
    dq_h: VecDeque<(usize, f64)>,
    dq_l: VecDeque<(usize, f64)>,
    dq_ls0: VecDeque<(usize, f64)>,
    dq_ss0: VecDeque<(usize, f64)>,
    i: usize,
}

impl CkspStream {
    pub fn try_new(params: CkspParams) -> Result<Self, CkspError> {
        let p = params.p.unwrap_or(10);
        let x = params.x.unwrap_or(1.0);
        let q = params.q.unwrap_or(9);
        if p == 0 || q == 0 {
            return Err(CkspError::InvalidParam { param: "p/q" });
        }
        if !x.is_finite() {
            return Err(CkspError::InvalidParam { param: "x" });
        }
        Ok(Self {
            p,
            x,
            q,
            alpha: 1.0 / p as f64,
            sum_tr: 0.0,
            rma: 0.0,
            prev_close: f64::NAN,
            dq_h: VecDeque::new(),
            dq_l: VecDeque::new(),
            dq_ls0: VecDeque::new(),
            dq_ss0: VecDeque::new(),
            i: 0,
        })
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<(f64, f64)> {
        let tr = if self.prev_close.is_nan() {
            high - low
        
            } else {
            let hl = high - low;
            let hc = (high - self.prev_close).abs();
            let lc = (low - self.prev_close).abs();
            hl.max(hc).max(lc)
        };
        self.prev_close = close;
        let atr_opt = if self.i < self.p {
            self.sum_tr += tr;
            if self.i == self.p - 1 {
                self.rma = self.sum_tr / self.p as f64;
                Some(self.rma)
            } else {
                None
            }
        } else {
            self.rma += self.alpha * (tr - self.rma);
            Some(self.rma)
        };

        while let Some((_, v)) = self.dq_h.back() {
            if *v <= high {
                self.dq_h.pop_back();
            
                } else {
                break;
            }
        }
        self.dq_h.push_back((self.i, high));
        let start_h = self.i.saturating_sub(self.q - 1);
        while let Some(&(idx, _)) = self.dq_h.front() {
            if idx < start_h {
                self.dq_h.pop_front();
            
                } else {
                break;
            }
        }

        while let Some((_, v)) = self.dq_l.back() {
            if *v >= low {
                self.dq_l.pop_back();
            
                } else {
                break;
            }
        }
        self.dq_l.push_back((self.i, low));
        let start_l = self.i.saturating_sub(self.q - 1);
        while let Some(&(idx, _)) = self.dq_l.front() {
            if idx < start_l {
                self.dq_l.pop_front();
            
                } else {
                break;
            }
        }
        let atr = match atr_opt {
            Some(v) => v,
            None => { self.i += 1; return None; }
        };

        let (mh, ml) = match (self.dq_h.front(), self.dq_l.front()) {
            (Some(&(_, mh)), Some(&(_, ml))) => (mh, ml),
            _ => {
                self.i += 1;
                return None;
            }
        };
        let ls0_val = mh - self.x * atr;
        let ss0_val = ml + self.x * atr;

        while let Some((_, val)) = self.dq_ls0.back() {
            if *val <= ls0_val {
                self.dq_ls0.pop_back();
            
                } else {
                break;
            }
        }
        self.dq_ls0.push_back((self.i, ls0_val));
        let start_ls0 = self.i.saturating_sub(self.q - 1);
        while let Some(&(idx, _)) = self.dq_ls0.front() {
            if idx < start_ls0 {
                self.dq_ls0.pop_front();
            
                } else {
                break;
            }
        }
        let long = self.dq_ls0.front().map(|&(_, v)| v).unwrap_or(f64::NAN);

        while let Some((_, val)) = self.dq_ss0.back() {
            if *val >= ss0_val {
                self.dq_ss0.pop_back();
            
                } else {
                break;
            }
        }
        self.dq_ss0.push_back((self.i, ss0_val));
        let start_ss0 = self.i.saturating_sub(self.q - 1);
        while let Some(&(idx, _)) = self.dq_ss0.front() {
            if idx < start_ss0 {
                self.dq_ss0.pop_front();
            
                } else {
                break;
            }
        }
        let short = self.dq_ss0.front().map(|&(_, v)| v).unwrap_or(f64::NAN);

        self.i += 1;
        Some((long, short))
    }
}

// ========================= Batch/Range Builder & Output =========================

#[derive(Clone, Debug)]
pub struct CkspBatchRange {
    pub p: (usize, usize, usize),
    pub x: (f64, f64, f64),
    pub q: (usize, usize, usize),
}

impl Default for CkspBatchRange {
    fn default() -> Self {
        Self {
            p: (10, 40, 1),
            x: (1.0, 1.0, 0.0),
            q: (9, 24, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CkspBatchBuilder {
    range: CkspBatchRange,
    kernel: Kernel,
}

impl CkspBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn p_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.p = (start, end, step);
        self
    }
    #[inline]
    pub fn p_static(mut self, p: usize) -> Self {
        self.range.p = (p, p, 0);
        self
    }
    #[inline]
    pub fn x_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.x = (start, end, step);
        self
    }
    #[inline]
    pub fn x_static(mut self, x: f64) -> Self {
        self.range.x = (x, x, 0.0);
        self
    }
    #[inline]
    pub fn q_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.q = (start, end, step);
        self
    }
    #[inline]
    pub fn q_static(mut self, q: usize) -> Self {
        self.range.q = (q, q, 0);
        self
    }
    pub fn apply_slices(self, high: &[f64], low: &[f64], close: &[f64]) -> Result<CkspBatchOutput, CkspError> {
        cksp_batch_with_kernel(high, low, close, &self.range, self.kernel)
    }
    pub fn with_default_slices(high: &[f64], low: &[f64], close: &[f64], k: Kernel) -> Result<CkspBatchOutput, CkspError> {
        CkspBatchBuilder::new().kernel(k).apply_slices(high, low, close)
    }
    pub fn apply_candles(self, c: &Candles) -> Result<CkspBatchOutput, CkspError> {
        let h = c.select_candle_field("high").map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
        let l = c.select_candle_field("low").map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
        let cl = c.select_candle_field("close").map_err(|e| CkspError::CandleFieldError(e.to_string()))?;
        self.apply_slices(h, l, cl)
    }
    pub fn with_default_candles(c: &Candles) -> Result<CkspBatchOutput, CkspError> {
        CkspBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c)
    }
}

#[derive(Clone, Debug)]
pub struct CkspBatchOutput {
    pub long_values: Vec<f64>,
    pub short_values: Vec<f64>,
    pub combos: Vec<CkspParams>,
    pub rows: usize,
    pub cols: usize,
}
impl CkspBatchOutput {
    pub fn row_for_params(&self, p: &CkspParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.p.unwrap_or(10) == p.p.unwrap_or(10)
                && (c.x.unwrap_or(1.0) - p.x.unwrap_or(1.0)).abs() < 1e-12
                && c.q.unwrap_or(9) == p.q.unwrap_or(9)
        })
    }
    pub fn values_for(&self, p: &CkspParams) -> Option<(&[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.long_values[start..start + self.cols],
                &self.short_values[start..start + self.cols],
            )
        })
    }
}

#[inline(always)]
fn expand_grid(r: &CkspBatchRange) -> Vec<CkspParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }

    let ps = axis_usize(r.p);
    let xs = axis_f64(r.x);
    let qs = axis_usize(r.q);

    let mut out = Vec::with_capacity(ps.len() * xs.len() * qs.len());
    for &p in &ps {
        for &x in &xs {
            for &q in &qs {
                out.push(CkspParams {
                    p: Some(p),
                    x: Some(x),
                    q: Some(q),
                });
            }
        }
    }
    out
}

pub fn cksp_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CkspBatchRange,
    k: Kernel,
) -> Result<CkspBatchOutput, CkspError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(CkspError::InvalidParam { param: "kernel" }),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    cksp_batch_par_slice(high, low, close, sweep, simd)
}

#[inline(always)]
pub fn cksp_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CkspBatchRange,
    kern: Kernel,
) -> Result<CkspBatchOutput, CkspError> {
    cksp_batch_inner(high, low, close, sweep, kern, false)
}

#[inline(always)]
pub fn cksp_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CkspBatchRange,
    kern: Kernel,
) -> Result<CkspBatchOutput, CkspError> {
    cksp_batch_inner(high, low, close, sweep, kern, true)
}

#[inline(always)]
fn cksp_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &CkspBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<CkspBatchOutput, CkspError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(CkspError::InvalidParam { param: "combos" });
    }
    let size = close.len();
    if high.len() != low.len() || low.len() != close.len() {
        return Err(CkspError::InconsistentLengths);
    }
    let first_valid = close.iter().position(|x| !x.is_nan()).ok_or(CkspError::NoData)?;

    let rows = combos.len();
    let cols = size;

    let mut long_values = vec![f64::NAN; rows * cols];
    let mut short_values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_long: &mut [f64], out_short: &mut [f64]| unsafe {
        let prm = &combos[row];
        let (p, x, q) = (prm.p.unwrap(), prm.x.unwrap(), prm.q.unwrap());
        match kern {
            Kernel::Scalar => cksp_row_scalar(high, low, close, p, x, q, first_valid, out_long, out_short),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => cksp_row_avx2(high, low, close, p, x, q, first_valid, out_long, out_short),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => cksp_row_avx512(high, low, close, p, x, q, first_valid, out_long, out_short),
            _ => unreachable!(),
        }
    };

    if parallel {


        #[cfg(not(target_arch = "wasm32"))] {


        long_values


                    .par_chunks_mut(cols)


                    .zip(short_values.par_chunks_mut(cols))


                    .enumerate()


                    .for_each(|(row, (lv, sv))| do_row(row, lv, sv));


        }


        #[cfg(target_arch = "wasm32")] {


        for (row, (lv, sv)) in long_values.chunks_mut(cols).zip(short_values.chunks_mut(cols)).enumerate() {


                    do_row(row, lv, sv);


        }

    }
    } else {
        for (row, (lv, sv)) in long_values.chunks_mut(cols).zip(short_values.chunks_mut(cols)).enumerate() {
            do_row(row, lv, sv);
        }
    }

    Ok(CkspBatchOutput {
        long_values,
        short_values,
        combos,
        rows,
        cols,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_cksp_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = CkspParams {
            p: None,
            x: None,
            q: None,
        };
        let input = CkspInput::from_candles(&candles, default_params);
        let output = cksp_with_kernel(&input, kernel)?;
        assert_eq!(output.long_values.len(), candles.close.len());
        assert_eq!(output.short_values.len(), candles.close.len());
        Ok(())
    }

    fn check_cksp_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let input = CkspInput::from_candles(&candles, params);
        let output = cksp_with_kernel(&input, kernel)?;

        let expected_long_last_5 = [
            60306.66197802568,
            60306.66197802568,
            60306.66197802568,
            60203.29578022311,
            60201.57958198072,
        ];
        let l_start = output.long_values.len() - 5;
        let long_slice = &output.long_values[l_start..];
        for (i, &val) in long_slice.iter().enumerate() {
            let exp_val = expected_long_last_5[i];
            assert!(
                (val - exp_val).abs() < 1e-5,
                "[{}] CKSP long mismatch at idx {}: expected {}, got {}",
                test_name, i, exp_val, val
            );
        }

        let expected_short_last_5 = [
            58757.826484736055,
            58701.74383626245,
            58656.36945263621,
            58611.03250737258,
            58611.03250737258,
        ];
        let s_start = output.short_values.len() - 5;
        let short_slice = &output.short_values[s_start..];
        for (i, &val) in short_slice.iter().enumerate() {
            let exp_val = expected_short_last_5[i];
            assert!(
                (val - exp_val).abs() < 1e-5,
                "[{}] CKSP short mismatch at idx {}: expected {}, got {}",
                test_name, i, exp_val, val
            );
        }
        Ok(())
    }

    fn check_cksp_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = CkspInput::with_default_candles(&candles);
        match input.data {
            CkspData::Candles { .. } => {},
            _ => panic!("Expected CkspData::Candles"),
        }
        let output = cksp_with_kernel(&input, kernel)?;
        assert_eq!(output.long_values.len(), candles.close.len());
        assert_eq!(output.short_values.len(), candles.close.len());
        Ok(())
    }

    fn check_cksp_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 10.0, 10.5];
        let close = [9.5, 10.5, 11.0];
        let params = CkspParams {
            p: Some(0),
            x: Some(1.0),
            q: Some(9),
        };
        let input = CkspInput::from_slices(&high, &low, &close, params);
        let res = cksp_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CKSP should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_cksp_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 10.0, 10.5];
        let close = [9.5, 10.5, 11.0];
        let params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let input = CkspInput::from_slices(&high, &low, &close, params);
        let res = cksp_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CKSP should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_cksp_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [42.0];
        let low = [41.0];
        let close = [41.5];
        let params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let input = CkspInput::from_slices(&high, &low, &close, params);
        let res = cksp_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] CKSP should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_cksp_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        };
        let first_input = CkspInput::from_candles(&candles, first_params.clone());
        let first_result = cksp_with_kernel(&first_input, kernel)?;

        let dummy_close = vec![0.0; first_result.long_values.len()];
        let second_input = CkspInput::from_slices(
            &first_result.long_values,
            &first_result.short_values,
            &dummy_close,
            first_params,
        );
        let second_result = cksp_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.long_values.len(), dummy_close.len());
        assert_eq!(second_result.short_values.len(), dummy_close.len());
        Ok(())
    }

    fn check_cksp_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = CkspInput::from_candles(&candles, CkspParams {
            p: Some(10),
            x: Some(1.0),
            q: Some(9),
        });
        let res = cksp_with_kernel(&input, kernel)?;
        assert_eq!(res.long_values.len(), candles.close.len());
        assert_eq!(res.short_values.len(), candles.close.len());
        if res.long_values.len() > 240 {
            for i in 240..res.long_values.len() {
                assert!(
                    !res.long_values[i].is_nan(),
                    "[{}] Found unexpected NaN in long_values at out-index {}",
                    test_name, i
                );
                assert!(
                    !res.short_values[i].is_nan(),
                    "[{}] Found unexpected NaN in short_values at out-index {}",
                    test_name, i
                );
            }
        }
        Ok(())
    }

    fn check_cksp_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let p = 10;
        let x = 1.0;
        let q = 9;

        let input = CkspInput::from_candles(
            &candles,
            CkspParams {
                p: Some(p),
                x: Some(x),
                q: Some(q),
            },
        );
        let batch_output = cksp_with_kernel(&input, kernel)?;
        let mut stream = CkspStream::try_new(CkspParams {
            p: Some(p),
            x: Some(x),
            q: Some(q),
        })?;

        let mut stream_long = Vec::with_capacity(candles.close.len());
        let mut stream_short = Vec::with_capacity(candles.close.len());
        for i in 0..candles.close.len() {
            let h = candles.high[i];
            let l = candles.low[i];
            let c = candles.close[i];
            match stream.update(h, l, c) {
                Some((long, short)) => {
                    stream_long.push(long);
                    stream_short.push(short);
                }
                None => {
                    stream_long.push(f64::NAN);
                    stream_short.push(f64::NAN);
                }
            }
        }
        assert_eq!(batch_output.long_values.len(), stream_long.len());
        assert_eq!(batch_output.short_values.len(), stream_short.len());
        for i in 0..stream_long.len() {
            let b_long = batch_output.long_values[i];
            let b_short = batch_output.short_values[i];
            let s_long = stream_long[i];
            let s_short = stream_short[i];
            let diff_long = (b_long - s_long).abs();
            let diff_short = (b_short - s_short).abs();
            if b_long.is_nan() && s_long.is_nan() && b_short.is_nan() && s_short.is_nan() {
                continue;
            }
            assert!(
                diff_long < 1e-8,
                "[{}] CKSP streaming long f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, b_long, s_long, diff_long
            );
            assert!(
                diff_short < 1e-8,
                "[{}] CKSP streaming short f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name, i, b_short, s_short, diff_short
            );
        }
        Ok(())
    }

    macro_rules! generate_all_cksp_tests {
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

    generate_all_cksp_tests!(
        check_cksp_partial_params,
        check_cksp_accuracy,
        check_cksp_default_candles,
        check_cksp_zero_period,
        check_cksp_period_exceeds_length,
        check_cksp_very_small_dataset,
        check_cksp_reinput,
        check_cksp_nan_handling,
        check_cksp_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = CkspBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c)?;

        let def = CkspParams::default();
        let (long_row, short_row) = output.values_for(&def).expect("default row missing");

        assert_eq!(long_row.len(), c.close.len());
        assert_eq!(short_row.len(), c.close.len());

        let expected_long = [
            60306.66197802568,
            60306.66197802568,
            60306.66197802568,
            60203.29578022311,
            60201.57958198072,
        ];
        let start = long_row.len() - 5;
        for (i, &v) in long_row[start..].iter().enumerate() {
            assert!(
                (v - expected_long[i]).abs() < 1e-5,
                "[{test}] default-row long mismatch at idx {i}: {v} vs {expected_long:?}"
            );
        }

        let expected_short = [
            58757.826484736055,
            58701.74383626245,
            58656.36945263621,
            58611.03250737258,
            58611.03250737258,
        ];
        for (i, &v) in short_row[start..].iter().enumerate() {
            assert!(
                (v - expected_short[i]).abs() < 1e-5,
                "[{test}] default-row short mismatch at idx {i}: {v} vs {expected_short:?}"
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
