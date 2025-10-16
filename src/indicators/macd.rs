//! # Moving Average Convergence/Divergence (MACD)
//!
//! A trend-following momentum indicator that shows the relationship between two moving averages of a data series.
//! Calculates the MACD line, signal line, and histogram.
//!
//! ## Parameters
//! - **fast_period**: Shorter moving average period (default: 12)
//! - **slow_period**: Longer moving average period (default: 26)
//! - **signal_period**: Signal line moving average period (default: 9)
//! - **ma_type**: Moving average type for all components (default: "ema")
//!
//! ## Returns
//! - `Ok(MacdOutput)` on success, containing MACD, signal, and histogram vectors
//! - `Err(MacdError)` on failure
//!
//! ## Developer Notes
//! - SIMD: AVX2/AVX512 accelerate only the SMA seeding of EMAs; the EMA recurrences
//!   are sequential by definition. Expect modest speedups on large windows; scalar
//!   remains the reference path and is single-pass and FMA-enabled.
//! - Streaming: O(1) for EMA type, O(n) fallback for other MA types. Non-EMA stream
//!   remains a simple fallback; further optimization is out of scope here.
//! - Memory: Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix, init_matrix_prefixes)
//!
//! Decision: Streaming path updated to O(1) per tick for EMA/RMA/SMA/WMA.
//! Seed EMAs by SMA; signal seeded by SMA of MACD. Matches batch warmups.
//! Unknown MA types fall back to Unsupported in the stream path.

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
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MacdData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MacdOutput {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub hist: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MacdParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub signal_period: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for MacdParams {
    fn default() -> Self {
        Self {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MacdInput<'a> {
    pub data: MacdData<'a>,
    pub params: MacdParams,
}

impl<'a> MacdInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MacdParams) -> Self {
        Self {
            data: MacdData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MacdParams) -> Self {
        Self {
            data: MacdData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MacdParams::default())
    }
    #[inline]
    pub fn get_fast_period(&self) -> usize {
        self.params.fast_period.unwrap_or(12)
    }
    #[inline]
    pub fn get_slow_period(&self) -> usize {
        self.params.slow_period.unwrap_or(26)
    }
    #[inline]
    pub fn get_signal_period(&self) -> usize {
        self.params.signal_period.unwrap_or(9)
    }
    #[inline]
    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .clone()
            .unwrap_or_else(|| "ema".to_string())
    }
}

impl<'a> AsRef<[f64]> for MacdInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MacdData::Slice(s) => s,
            MacdData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MacdBuilder {
    fast_period: Option<usize>,
    slow_period: Option<usize>,
    signal_period: Option<usize>,
    ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for MacdBuilder {
    fn default() -> Self {
        Self {
            fast_period: None,
            slow_period: None,
            signal_period: None,
            ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MacdBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline]
    pub fn fast_period(mut self, n: usize) -> Self {
        self.fast_period = Some(n);
        self
    }
    #[inline]
    pub fn slow_period(mut self, n: usize) -> Self {
        self.slow_period = Some(n);
        self
    }
    #[inline]
    pub fn signal_period(mut self, n: usize) -> Self {
        self.signal_period = Some(n);
        self
    }
    #[inline]
    pub fn ma_type<S: Into<String>>(mut self, s: S) -> Self {
        self.ma_type = Some(s.into());
        self
    }
    #[inline]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn apply(self, c: &Candles) -> Result<MacdOutput, MacdError> {
        let p = MacdParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            signal_period: self.signal_period,
            ma_type: self.ma_type,
        };
        let i = MacdInput::from_candles(c, "close", p);
        macd_with_kernel(&i, self.kernel)
    }

    #[inline]
    pub fn apply_slice(self, d: &[f64]) -> Result<MacdOutput, MacdError> {
        let p = MacdParams {
            fast_period: self.fast_period,
            slow_period: self.slow_period,
            signal_period: self.signal_period,
            ma_type: self.ma_type,
        };
        let i = MacdInput::from_slice(d, p);
        macd_with_kernel(&i, self.kernel)
    }
}

#[derive(Debug, Error)]
pub enum MacdError {
    #[error("macd: All values are NaN.")]
    AllValuesNaN,
    #[error("macd: Invalid period: fast = {fast}, slow = {slow}, signal = {signal}, data length = {data_len}")]
    InvalidPeriod {
        fast: usize,
        slow: usize,
        signal: usize,
        data_len: usize,
    },
    #[error("macd: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("macd: Unknown MA type: {0}")]
    UnknownMA(String),
    #[error("macd: Invalid kernel for batch operation: {0}")]
    InvalidKernel(String),
}

#[derive(Debug, Clone)]
pub struct MacdStream {
    fast: usize,
    slow: usize,
    signal: usize,
    kind: MaKind,
    inner: StreamImpl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaKind {
    Ema, // α = 2/(n+1); seeded with SMA
    Rma, // Wilders: α = 1/n; seeded with SMA
    Sma,
    Wma,     // Linear weights 1..n (aka LWMA)
    Unknown, // Fallback
}

impl MaKind {
    #[inline]
    fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "ema" => MaKind::Ema,
            "rma" | "wilders" | "smma" => MaKind::Rma,
            "sma" => MaKind::Sma,
            "wma" | "lwma" => MaKind::Wma,
            _ => MaKind::Unknown,
        }
    }
}

#[derive(Debug, Clone)]
enum StreamImpl {
    // EMA / RMA (Wilders) seeded by SMA, then classic recurrence
    Ema(EmaState),
    // Pure SMA MACD + SMA signal via rolling sums
    Sma(SmaState),
    // Linear WMA(1..n) MACD + WMA signal via O(1) weighted-sum update
    Wma(WmaState),
    // If MA type not supported in stream path
    Unsupported,
}

#[derive(Debug, Clone)]
struct EmaState {
    // constants
    af: f64,  // 2/(fast+1) or 1/fast (RMA)
    omf: f64, // 1 - af
    aslow: f64,
    oms: f64,
    asig: f64,
    omsi: f64,
    inv_fast: f64,
    inv_slow: f64,
    inv_sig: f64,

    // seeding (SMA) accumulators
    fsum: f64,
    ssum: f64,
    fcnt: usize,
    scnt: usize,

    // EMA state once seeded
    fast_ema: Option<f64>,
    slow_ema: Option<f64>,

    // signal seeding (SMA of MACD), then EMA
    sig_accum: f64,
    sig_cnt: usize,
    sig_ema: Option<f64>,
}

#[derive(Debug, Clone)]
struct SmaState {
    fast: RollingSma,
    slow: RollingSma,
    sig: RollingSma,
}

#[derive(Debug, Clone)]
struct WmaState {
    fast: RollingWma,
    slow: RollingWma,
    sig: RollingWma,
}

// ---------- Rolling helpers (alloc once, O(1) updates) ----------

#[derive(Debug, Clone)]
struct RollingSma {
    n: usize,
    inv_n: f64,
    buf: Vec<f64>,
    sum: f64,
    idx: usize,
    cnt: usize,
}

impl RollingSma {
    #[inline]
    fn new(n: usize) -> Self {
        Self {
            n,
            inv_n: 1.0 / n as f64,
            buf: vec![0.0; n],
            sum: 0.0,
            idx: 0,
            cnt: 0,
        }
    }
    #[inline]
    fn push(&mut self, x: f64) -> Option<f64> {
        if !x.is_finite() {
            return None;
        } // ignore NaNs/Infs in stream
        if self.cnt < self.n {
            self.sum += x;
            self.buf[self.idx] = x;
            self.idx = (self.idx + 1) % self.n;
            self.cnt += 1;
            if self.cnt == self.n {
                Some(self.sum * self.inv_n)
            } else {
                None
            }
        } else {
            let old = self.buf[self.idx];
            self.buf[self.idx] = x;
            self.idx = (self.idx + 1) % self.n;
            self.sum += x - old;
            Some(self.sum * self.inv_n)
        }
    }
}

#[derive(Debug, Clone)]
struct RollingWma {
    n: usize,
    inv_denom: f64, // 2 / (n*(n+1))
    buf: Vec<f64>,
    idx: usize,
    cnt: usize,
    sum: f64,  // S = sum of window
    wsum: f64, // W = sum_{k=1..n} k*x
}

impl RollingWma {
    #[inline]
    fn new(n: usize) -> Self {
        let denom = (n as f64) * (n as f64 + 1.0) * 0.5;
        Self {
            n,
            inv_denom: 1.0 / denom,
            buf: vec![0.0; n],
            idx: 0,
            cnt: 0,
            sum: 0.0,
            wsum: 0.0,
        }
    }
    #[inline]
    fn push(&mut self, x: f64) -> Option<f64> {
        if !x.is_finite() {
            return None;
        }
        if self.cnt < self.n {
            // Build initial window: W += (cnt+1)*x
            self.cnt += 1;
            self.sum += x;
            self.wsum += (self.cnt as f64) * x;
            self.buf[self.idx] = x;
            self.idx = (self.idx + 1) % self.n;
            if self.cnt == self.n {
                Some(self.wsum * self.inv_denom)
            } else {
                None
            }
        } else {
            // Use O(1) recurrence: W' = W + n*x_new - S_prev, then update S
            let s_prev = self.sum;
            let old = self.buf[self.idx];
            self.buf[self.idx] = x;
            self.idx = (self.idx + 1) % self.n;

            self.wsum = self.wsum + (self.n as f64) * x - s_prev;
            self.sum = s_prev + x - old;
            Some(self.wsum * self.inv_denom)
        }
    }
}

// ----------------- MacdStream public API -----------------

impl MacdStream {
    pub fn new(fast: usize, slow: usize, signal: usize, ma_type: &str) -> Self {
        let kind = MaKind::from_str(ma_type);
        let inner = match kind {
            MaKind::Ema | MaKind::Rma => {
                let (af, aslow) = match kind {
                    MaKind::Ema => (2.0 / (fast as f64 + 1.0), 2.0 / (slow as f64 + 1.0)),
                    MaKind::Rma => (
                        1.0 / fast as f64, // Wilders
                        1.0 / slow as f64,
                    ),
                    _ => unreachable!(),
                };
                let asig = match kind {
                    MaKind::Ema => 2.0 / (signal as f64 + 1.0),
                    MaKind::Rma => 1.0 / signal as f64,
                    _ => unreachable!(),
                };
                StreamImpl::Ema(EmaState {
                    af,
                    omf: 1.0 - af,
                    aslow,
                    oms: 1.0 - aslow,
                    asig,
                    omsi: 1.0 - asig,
                    inv_fast: 1.0 / fast as f64,
                    inv_slow: 1.0 / slow as f64,
                    inv_sig: 1.0 / signal as f64,
                    fsum: 0.0,
                    ssum: 0.0,
                    fcnt: 0,
                    scnt: 0,
                    fast_ema: None,
                    slow_ema: None,
                    sig_accum: 0.0,
                    sig_cnt: 0,
                    sig_ema: None,
                })
            }
            MaKind::Sma => StreamImpl::Sma(SmaState {
                fast: RollingSma::new(fast),
                slow: RollingSma::new(slow),
                sig: RollingSma::new(signal),
            }),
            MaKind::Wma => StreamImpl::Wma(WmaState {
                fast: RollingWma::new(fast),
                slow: RollingWma::new(slow),
                sig: RollingWma::new(signal),
            }),
            MaKind::Unknown => StreamImpl::Unsupported,
        };

        Self {
            fast,
            slow,
            signal,
            kind,
            inner,
        }
    }

    /// Update with the next value. Returns (macd, signal, hist) once warmed up; otherwise None.
    /// NaN/Inf inputs are ignored (no state change; returns None).
    pub fn update(&mut self, x: f64) -> Option<(f64, f64, f64)> {
        if !x.is_finite() {
            return None;
        }

        match &mut self.inner {
            StreamImpl::Ema(st) => {
                // --- seed or advance FAST ---
                if st.fcnt < self.fast {
                    st.fcnt += 1;
                    st.fsum += x;
                    if st.fcnt == self.fast {
                        st.fast_ema = Some(st.fsum * st.inv_fast);
                    }
                } else {
                    // classic EMA with FMA
                    let fe = st.fast_ema.unwrap();
                    st.fast_ema = Some(x.mul_add(st.af, st.omf * fe));
                }

                // --- seed or advance SLOW ---
                if st.scnt < self.slow {
                    st.scnt += 1;
                    st.ssum += x;
                    if st.scnt == self.slow {
                        st.slow_ema = Some(st.ssum * st.inv_slow);
                    }
                } else {
                    let se = st.slow_ema.unwrap();
                    st.slow_ema = Some(x.mul_add(st.aslow, st.oms * se));
                }

                // MACD becomes available when slow is seeded
                if st.scnt >= self.slow {
                    let m = st.fast_ema.unwrap() - st.slow_ema.unwrap();

                    // Seed signal with SMA of the first `signal` MACD values, then EMA
                    if st.sig_ema.is_none() {
                        st.sig_cnt += 1;
                        st.sig_accum += m;
                        if st.sig_cnt == self.signal {
                            let se = st.sig_accum * st.inv_sig;
                            st.sig_ema = Some(se);
                            let hist = m - se;
                            return Some((m, se, hist));
                        }
                        return None;
                    } else {
                        let prev = st.sig_ema.unwrap();
                        let se = m.mul_add(st.asig, st.omsi * prev);
                        st.sig_ema = Some(se);
                        let hist = m - se;
                        return Some((m, se, hist));
                    }
                }
                None
            }

            StreamImpl::Sma(st) => {
                let f = st.fast.push(x)?;
                let s = st.slow.push(x)?;
                let m = f - s;
                if let Some(se) = st.sig.push(m) {
                    Some((m, se, m - se))
                } else {
                    None
                }
            }

            StreamImpl::Wma(st) => {
                let f = st.fast.push(x)?;
                let s = st.slow.push(x)?;
                let m = f - s;
                if let Some(se) = st.sig.push(m) {
                    Some((m, se, m - se))
                } else {
                    None
                }
            }

            StreamImpl::Unsupported => {
                // keep prior behavior: return None; caller may fall back
                None
            }
        }
    }
}

#[inline]
pub fn macd(input: &MacdInput) -> Result<MacdOutput, MacdError> {
    macd_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn macd_prepare<'a>(
    input: &'a MacdInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, String, usize, usize, Kernel), MacdError> {
    let data = input.as_ref();
    let len = data.len();
    let fast = input.get_fast_period();
    let slow = input.get_slow_period();
    let signal = input.get_signal_period();
    let ma_type = input.get_ma_type();

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MacdError::AllValuesNaN)?;
    if fast == 0 || slow == 0 || signal == 0 || fast > len || slow > len || signal > len {
        return Err(MacdError::InvalidPeriod {
            fast,
            slow,
            signal,
            data_len: len,
        });
    }
    if len - first < slow {
        return Err(MacdError::NotEnoughValidData {
            needed: slow,
            valid: len - first,
        });
    }

    let macd_warmup = first + slow - 1;
    let signal_warmup = first + slow + signal - 2;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    Ok((
        data,
        fast,
        slow,
        signal,
        ma_type,
        macd_warmup,
        signal_warmup,
        chosen,
    ))
}

#[inline(always)]
/// Classic EMA kernel – single-pass, SMA-seeded, and warmup-accurate.
/// Seeds fast/slow with SMA over their first windows, advances fast EMA up to
/// the slow warmup, then computes MACD, signal (SMA-seeded), and histogram in
/// a single forward pass. Uses `mul_add` where beneficial.
fn macd_compute_into_classic_ema(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    first: usize,
    macd_out: &mut [f64],
    signal_out: &mut [f64],
    hist_out: &mut [f64],
) -> Result<(), MacdError> {
    let len = data.len();
    let macd_warmup = first + slow - 1;
    let signal_warmup = first + slow + signal - 2;

    // EMA coefficients
    let af = 2.0 / (fast as f64 + 1.0);
    let aslow = 2.0 / (slow as f64 + 1.0);
    let asig = 2.0 / (signal as f64 + 1.0);
    let omf = 1.0 - af;
    let oms = 1.0 - aslow;
    let omsi = 1.0 - asig;

    // Seed fast EMA with SMA over [first .. first+fast)
    let mut fsum = 0.0;
    let mut i = 0usize;
    // small unroll – cheap on short windows
    while i + 4 <= fast {
        fsum += data[first + i + 0];
        fsum += data[first + i + 1];
        fsum += data[first + i + 2];
        fsum += data[first + i + 3];
        i += 4;
    }
    while i < fast {
        fsum += data[first + i];
        i += 1;
    }
    let mut fast_ema = fsum / fast as f64;

    // Seed slow EMA with SMA over [first .. first+slow)
    let mut ssum = 0.0;
    let mut j = 0usize;
    while j + 4 <= slow {
        ssum += data[first + j + 0];
        ssum += data[first + j + 1];
        ssum += data[first + j + 2];
        ssum += data[first + j + 3];
        j += 4;
    }
    while j < slow {
        ssum += data[first + j];
        j += 1;
    }
    let mut slow_ema = ssum / slow as f64;

    // Advance FAST EMA from (first+fast) ..= macd_warmup to align with SLOW at macd_warmup
    let mut t = first + fast;
    while t <= macd_warmup {
        let x = data[t];
        fast_ema = x.mul_add(af, omf * fast_ema);
        t += 1;
    }

    // First valid MACD at macd_warmup
    if macd_warmup < len {
        let m0 = fast_ema - slow_ema;
        macd_out[macd_warmup] = m0;
    }

    // Signal seeding state
    let mut have_seed = false;
    let mut se = 0.0_f64;
    if signal == 1 {
        // seed immediately with m0
        if signal_warmup < len {
            se = macd_out[signal_warmup];
            have_seed = true;
            signal_out[signal_warmup] = se;
            hist_out[signal_warmup] = macd_out[signal_warmup] - se;
        }
    }
    let mut sig_accum = if signal > 1 && macd_warmup < len {
        macd_out[macd_warmup]
    } else {
        0.0
    };

    // Single forward pass from macd_warmup+1 .. len
    let mut k = macd_warmup.saturating_add(1);
    while k < len {
        let x = data[k];
        fast_ema = x.mul_add(af, omf * fast_ema);
        slow_ema = x.mul_add(aslow, oms * slow_ema);
        let m = fast_ema - slow_ema;
        macd_out[k] = m;

        if !have_seed {
            if signal > 1 && k <= signal_warmup {
                sig_accum += m;
                if k == signal_warmup {
                    se = sig_accum / (signal as f64);
                    have_seed = true;
                    signal_out[k] = se;
                    hist_out[k] = m - se;
                }
            }
        } else {
            se = m.mul_add(asig, omsi * se);
            if k >= signal_warmup {
                signal_out[k] = se;
                hist_out[k] = m - se;
            }
        }

        k += 1;
    }

    Ok(())
}

fn macd_compute_into(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    ma_type: &str,
    first: usize,
    macd_out: &mut [f64],
    signal_out: &mut [f64],
    hist_out: &mut [f64],
) -> Result<(), MacdError> {
    // Use classic kernel for EMA
    if ma_type.eq_ignore_ascii_case("ema") {
        return macd_compute_into_classic_ema(
            data, fast, slow, signal, first, macd_out, signal_out, hist_out,
        );
    }

    use crate::indicators::moving_averages::ma::{ma, MaData};

    debug_assert_eq!(macd_out.len(), data.len());
    debug_assert_eq!(signal_out.len(), data.len());
    debug_assert_eq!(hist_out.len(), data.len());

    // 1) Fast/slow MA (Vecs returned by ma.rs)
    let fast_ma = ma(&ma_type, MaData::Slice(data), fast).map_err(|e| {
        // Check if it's actually an unknown MA type or another error
        if e.to_string().contains("Unknown moving average type")
            || e.to_string().contains("Unsupported")
        {
            MacdError::UnknownMA(ma_type.to_string())
        } else if e.to_string().contains("All values are NaN") {
            MacdError::AllValuesNaN
        } else {
            // For other errors like invalid period, propagate the original message
            MacdError::UnknownMA(format!("{}: {}", ma_type, e))
        }
    })?;
    let slow_ma = ma(&ma_type, MaData::Slice(data), slow).map_err(|e| {
        if e.to_string().contains("Unknown moving average type")
            || e.to_string().contains("Unsupported")
        {
            MacdError::UnknownMA(ma_type.to_string())
        } else if e.to_string().contains("All values are NaN") {
            MacdError::AllValuesNaN
        } else {
            MacdError::UnknownMA(format!("{}: {}", ma_type, e))
        }
    })?;

    // 2) macd = fast - slow written directly to macd_out
    let macd_warmup = first + slow - 1;
    for i in macd_warmup..data.len() {
        let f = fast_ma[i];
        let s = slow_ma[i];
        if f.is_nan() || s.is_nan() {
            continue;
        }
        macd_out[i] = f - s;
    }

    // 3) signal and hist with minimal copies
    let signal_warmup = first + slow + signal - 2;
    if ma_type.eq_ignore_ascii_case("ema") {
        // In-place EMA of macd_out without allocating signal Vec
        // Standard EMA: alpha = 2/(n+1)
        let alpha = 2.0 / (signal as f64 + 1.0);
        // Start EMA after we have enough MACD values (signal period)
        let signal_start = macd_warmup + signal - 1;
        if signal_start < data.len() {
            // Find first non-NaN MACD value to seed with
            let mut seed_idx = signal_start;
            while seed_idx < data.len() && macd_out[seed_idx].is_nan() {
                seed_idx += 1;
            }

            // If we found a valid seed value, start EMA from there
            if seed_idx < data.len() {
                let mut prev = macd_out[seed_idx];
                signal_out[seed_idx] = prev;

                for i in (seed_idx + 1)..data.len() {
                    let x = macd_out[i];
                    if !x.is_nan() {
                        prev = alpha * x + (1.0 - alpha) * prev;
                        signal_out[i] = prev;
                    }
                }
            }
        }
    } else {
        // Fallback: one temporary vector, then copy into signal_out segment
        let sig_tmp = ma(&ma_type, MaData::Slice(macd_out), signal).map_err(|e| {
            if e.to_string().contains("Unknown moving average type")
                || e.to_string().contains("Unsupported")
            {
                MacdError::UnknownMA(ma_type.to_string())
            } else if e.to_string().contains("All values are NaN") {
                MacdError::AllValuesNaN
            } else {
                MacdError::UnknownMA(format!("{}: {}", ma_type, e))
            }
        })?;
        // Copy only the valid suffix; prefixes are already NaN via alloc_with_nan_prefix
        signal_out[signal_warmup..].copy_from_slice(&sig_tmp[signal_warmup..]);
    }

    // 4) hist = macd - signal directly
    for i in signal_warmup..data.len() {
        let m = macd_out[i];
        let s = signal_out[i];
        if m.is_nan() || s.is_nan() {
            continue;
        }
        hist_out[i] = m - s;
    }
    Ok(())
}

pub fn macd_with_kernel(input: &MacdInput, kernel: Kernel) -> Result<MacdOutput, MacdError> {
    let (data, fast, slow, signal, ma_type, macd_warmup, signal_warmup, chosen) =
        macd_prepare(input, kernel)?;
    let len = data.len();

    // EMA path can leverage kernel selection; others use scalar fallback
    if ma_type.eq_ignore_ascii_case("ema") {
        let first = macd_warmup + 1 - slow;
        // Dispatch like ALMA: Scalar → AVX2 → AVX512 (feature-gated)
        unsafe {
            match chosen {
                Kernel::Scalar | Kernel::ScalarBatch => {
                    return macd_scalar(data, fast, slow, signal, &ma_type, first);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => {
                    return macd_avx2(data, fast, slow, signal, &ma_type, first);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => {
                    return macd_avx512(data, fast, slow, signal, &ma_type, first);
                }
                #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
                Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                    return macd_scalar(data, fast, slow, signal, &ma_type, first);
                }
                _ => unreachable!(),
            }
        }
    }

    // Fallback for non-EMA: compute into preallocated buffers (scalar)
    let mut macd = alloc_with_nan_prefix(len, macd_warmup);
    let mut signal_vec = alloc_with_nan_prefix(len, signal_warmup);
    let mut hist = alloc_with_nan_prefix(len, signal_warmup);
    let first = macd_warmup + 1 - slow;
    macd_compute_into(
        data,
        fast,
        slow,
        signal,
        &ma_type,
        first,
        &mut macd,
        &mut signal_vec,
        &mut hist,
    )?;
    Ok(MacdOutput {
        macd,
        signal: signal_vec,
        hist,
    })
}

#[inline(always)]
/// Classic kernel - optimized scalar implementation for EMA
pub unsafe fn macd_scalar_classic_ema(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    first: usize,
) -> Result<MacdOutput, MacdError> {
    let len = data.len();
    let macd_warmup = first + slow - 1;
    let signal_warmup = first + slow + signal - 2;

    let mut macd = alloc_with_nan_prefix(len, macd_warmup);
    let mut signal_vec = alloc_with_nan_prefix(len, signal_warmup);
    let mut hist = alloc_with_nan_prefix(len, signal_warmup);

    macd_compute_into_classic_ema(
        data,
        fast,
        slow,
        signal,
        first,
        &mut macd,
        &mut signal_vec,
        &mut hist,
    )?;

    Ok(MacdOutput {
        macd,
        signal: signal_vec,
        hist,
    })
}

pub unsafe fn macd_scalar(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    ma_type: &str,
    first: usize,
) -> Result<MacdOutput, MacdError> {
    // Use classic kernel for EMA
    if ma_type.eq_ignore_ascii_case("ema") {
        return macd_scalar_classic_ema(data, fast, slow, signal, first);
    }

    use crate::indicators::moving_averages::ma::{ma, MaData};
    let len = data.len();
    let fast_ma = ma(ma_type, MaData::Slice(data), fast).map_err(|_| MacdError::AllValuesNaN)?;
    let slow_ma = ma(ma_type, MaData::Slice(data), slow).map_err(|_| MacdError::AllValuesNaN)?;

    // MACD warmup is when we have enough data for the slow MA
    let warmup = first + slow - 1;
    let mut macd = alloc_with_nan_prefix(len, warmup);
    for i in warmup..len {
        if fast_ma[i].is_nan() || slow_ma[i].is_nan() {
            continue;
        }
        macd[i] = fast_ma[i] - slow_ma[i];
    }
    let signal_ma =
        ma(ma_type, MaData::Slice(&macd), signal).map_err(|_| MacdError::AllValuesNaN)?;
    // Signal warmup is MACD warmup + signal period - 1
    let signal_warmup = warmup + signal - 1;
    let mut signal_vec = alloc_with_nan_prefix(len, signal_warmup);
    let mut hist = alloc_with_nan_prefix(len, signal_warmup);
    for i in first..len {
        if macd[i].is_nan() || signal_ma[i].is_nan() {
            continue;
        }
        signal_vec[i] = signal_ma[i];
        hist[i] = macd[i] - signal_ma[i];
    }
    Ok(MacdOutput {
        macd,
        signal: signal_vec,
        hist,
    })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn macd_avx2(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    ma_type: &str,
    first: usize,
) -> Result<MacdOutput, MacdError> {
    if !ma_type.eq_ignore_ascii_case("ema") {
        return macd_scalar(data, fast, slow, signal, ma_type, first);
    }

    // Helpers: horizontal sum and AVX2 reduction
    #[inline(always)]
    unsafe fn hsum256_pd(v: __m256d) -> f64 {
        let hi = _mm256_extractf128_pd(v, 1);
        let lo = _mm256_castpd256_pd128(v);
        let sum2 = _mm_add_pd(lo, hi);
        let shuf = _mm_unpackhi_pd(sum2, sum2);
        let sum = _mm_add_sd(sum2, shuf);
        _mm_cvtsd_f64(sum)
    }
    #[inline(always)]
    unsafe fn avx2_sum(ptr: *const f64, n: usize) -> f64 {
        let mut i = 0usize;
        let mut a0 = _mm256_setzero_pd();
        let mut a1 = _mm256_setzero_pd();
        while i + 8 <= n {
            let v0 = _mm256_loadu_pd(ptr.add(i));
            let v1 = _mm256_loadu_pd(ptr.add(i + 4));
            a0 = _mm256_add_pd(a0, v0);
            a1 = _mm256_add_pd(a1, v1);
            i += 8;
        }
        let mut acc = _mm256_add_pd(a0, a1);
        if i + 4 <= n {
            let v = _mm256_loadu_pd(ptr.add(i));
            acc = _mm256_add_pd(acc, v);
            i += 4;
        }
        let mut sum = hsum256_pd(acc);
        while i < n {
            sum += *ptr.add(i);
            i += 1;
        }
        sum
    }

    let len = data.len();
    let macd_warmup = first + slow - 1;
    let signal_warmup = first + slow + signal - 2;

    let mut macd = alloc_with_nan_prefix(len, macd_warmup);
    let mut signal_vec = alloc_with_nan_prefix(len, signal_warmup);
    let mut hist = alloc_with_nan_prefix(len, signal_warmup);

    // EMA coefficients
    let af = 2.0 / (fast as f64 + 1.0);
    let aslow = 2.0 / (slow as f64 + 1.0);
    let asig = 2.0 / (signal as f64 + 1.0);
    let omf = 1.0 - af;
    let oms = 1.0 - aslow;
    let omsi = 1.0 - asig;

    // SIMD-accelerated SMA seeds
    let base = data.as_ptr().add(first);
    let mut fast_ema = avx2_sum(base, fast) / fast as f64;
    let mut slow_ema = avx2_sum(base, slow) / slow as f64;

    // Advance fast EMA to macd_warmup
    let mut t = first + fast;
    while t <= macd_warmup {
        let x = *data.get_unchecked(t);
        fast_ema = x.mul_add(af, omf * fast_ema);
        t += 1;
    }

    let m0 = fast_ema - slow_ema;
    *macd.get_unchecked_mut(macd_warmup) = m0;

    let mut se = 0.0f64;
    let mut have_seed = false;
    if signal == 1 {
        se = m0;
        have_seed = true;
        if signal_warmup < len {
            *signal_vec.get_unchecked_mut(signal_warmup) = se;
            *hist.get_unchecked_mut(signal_warmup) = m0 - se;
        }
    }
    let mut sig_accum = if signal > 1 { m0 } else { 0.0 };

    let mut i = macd_warmup + 1;
    while i < len {
        let x = *data.get_unchecked(i);
        fast_ema = x.mul_add(af, omf * fast_ema);
        slow_ema = x.mul_add(aslow, oms * slow_ema);
        let m = fast_ema - slow_ema;
        *macd.get_unchecked_mut(i) = m;

        if !have_seed {
            if signal > 1 && i <= signal_warmup {
                sig_accum += m;
                if i == signal_warmup {
                    se = sig_accum / (signal as f64);
                    have_seed = true;
                    *signal_vec.get_unchecked_mut(i) = se;
                    *hist.get_unchecked_mut(i) = m - se;
                }
            }
        } else {
            se = m.mul_add(asig, omsi * se);
            if i >= signal_warmup {
                *signal_vec.get_unchecked_mut(i) = se;
                *hist.get_unchecked_mut(i) = m - se;
            }
        }
        i += 1;
    }

    Ok(MacdOutput {
        macd,
        signal: signal_vec,
        hist,
    })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn macd_avx512(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    ma_type: &str,
    first: usize,
) -> Result<MacdOutput, MacdError> {
    if !ma_type.eq_ignore_ascii_case("ema") {
        return macd_scalar(data, fast, slow, signal, ma_type, first);
    }

    #[inline(always)]
    unsafe fn avx512_sum(ptr: *const f64, n: usize) -> f64 {
        let mut i = 0usize;
        let mut a0 = _mm512_setzero_pd();
        let mut a1 = _mm512_setzero_pd();
        while i + 16 <= n {
            let v0 = _mm512_loadu_pd(ptr.add(i));
            let v1 = _mm512_loadu_pd(ptr.add(i + 8));
            a0 = _mm512_add_pd(a0, v0);
            a1 = _mm512_add_pd(a1, v1);
            i += 16;
        }
        let mut acc = _mm512_add_pd(a0, a1);
        if i + 8 <= n {
            let v = _mm512_loadu_pd(ptr.add(i));
            acc = _mm512_add_pd(acc, v);
            i += 8;
        }
        let mut sum = _mm512_reduce_add_pd(acc);
        while i < n {
            sum += *ptr.add(i);
            i += 1;
        }
        sum
    }

    let len = data.len();
    let macd_warmup = first + slow - 1;
    let signal_warmup = first + slow + signal - 2;

    let mut macd = alloc_with_nan_prefix(len, macd_warmup);
    let mut signal_vec = alloc_with_nan_prefix(len, signal_warmup);
    let mut hist = alloc_with_nan_prefix(len, signal_warmup);

    // EMA coefficients
    let af = 2.0 / (fast as f64 + 1.0);
    let aslow = 2.0 / (slow as f64 + 1.0);
    let asig = 2.0 / (signal as f64 + 1.0);
    let omf = 1.0 - af;
    let oms = 1.0 - aslow;
    let omsi = 1.0 - asig;

    // SIMD-accelerated SMA seeds
    let base = data.as_ptr().add(first);
    let mut fast_ema = avx512_sum(base, fast) / fast as f64;
    let mut slow_ema = avx512_sum(base, slow) / slow as f64;

    // Advance fast EMA to macd_warmup
    let mut t = first + fast;
    while t <= macd_warmup {
        let x = *data.get_unchecked(t);
        fast_ema = x.mul_add(af, omf * fast_ema);
        t += 1;
    }

    let m0 = fast_ema - slow_ema;
    *macd.get_unchecked_mut(macd_warmup) = m0;

    let mut se = 0.0f64;
    let mut have_seed = false;
    if signal == 1 {
        se = m0;
        have_seed = true;
        if signal_warmup < len {
            *signal_vec.get_unchecked_mut(signal_warmup) = se;
            *hist.get_unchecked_mut(signal_warmup) = m0 - se;
        }
    }
    let mut sig_accum = if signal > 1 { m0 } else { 0.0 };

    let mut i = macd_warmup + 1;
    while i < len {
        let x = *data.get_unchecked(i);
        fast_ema = x.mul_add(af, omf * fast_ema);
        slow_ema = x.mul_add(aslow, oms * slow_ema);
        let m = fast_ema - slow_ema;
        *macd.get_unchecked_mut(i) = m;

        if !have_seed {
            if signal > 1 && i <= signal_warmup {
                sig_accum += m;
                if i == signal_warmup {
                    se = sig_accum / (signal as f64);
                    have_seed = true;
                    *signal_vec.get_unchecked_mut(i) = se;
                    *hist.get_unchecked_mut(i) = m - se;
                }
            }
        } else {
            se = m.mul_add(asig, omsi * se);
            if i >= signal_warmup {
                *signal_vec.get_unchecked_mut(i) = se;
                *hist.get_unchecked_mut(i) = m - se;
            }
        }
        i += 1;
    }

    Ok(MacdOutput {
        macd,
        signal: signal_vec,
        hist,
    })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn macd_avx512_short(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    ma_type: &str,
    first: usize,
) -> Result<MacdOutput, MacdError> {
    macd_avx512(data, fast, slow, signal, ma_type, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn macd_avx512_long(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    ma_type: &str,
    first: usize,
) -> Result<MacdOutput, MacdError> {
    macd_avx512(data, fast, slow, signal, ma_type, first)
}

#[inline(always)]
pub fn macd_row_scalar(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    ma_type: &str,
    first: usize,
) -> Result<MacdOutput, MacdError> {
    unsafe { macd_scalar(data, fast, slow, signal, ma_type, first) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn macd_row_avx2(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    ma_type: &str,
    first: usize,
) -> Result<MacdOutput, MacdError> {
    unsafe { macd_avx2(data, fast, slow, signal, ma_type, first) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn macd_row_avx512(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    ma_type: &str,
    first: usize,
) -> Result<MacdOutput, MacdError> {
    unsafe { macd_avx512(data, fast, slow, signal, ma_type, first) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn macd_row_avx512_short(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    ma_type: &str,
    first: usize,
) -> Result<MacdOutput, MacdError> {
    unsafe { macd_avx512_short(data, fast, slow, signal, ma_type, first) }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn macd_row_avx512_long(
    data: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    ma_type: &str,
    first: usize,
) -> Result<MacdOutput, MacdError> {
    unsafe { macd_avx512_long(data, fast, slow, signal, ma_type, first) }
}

#[derive(Clone, Debug)]
pub struct MacdBatchRange {
    pub fast_period: (usize, usize, usize),
    pub slow_period: (usize, usize, usize),
    pub signal_period: (usize, usize, usize),
    pub ma_type: (String, String, String),
}

impl Default for MacdBatchRange {
    fn default() -> Self {
        Self {
            fast_period: (12, 12, 0),
            slow_period: (26, 26, 0),
            signal_period: (9, 9, 0),
            ma_type: ("ema".to_string(), "ema".to_string(), "".to_string()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MacdBatchBuilder {
    range: MacdBatchRange,
    kernel: Kernel,
}

impl MacdBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn fast_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast_period = (start, end, step);
        self
    }
    #[inline]
    pub fn slow_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow_period = (start, end, step);
        self
    }
    #[inline]
    pub fn signal_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.signal_period = (start, end, step);
        self
    }
    #[inline]
    pub fn ma_type_static(mut self, s: &str) -> Self {
        self.range.ma_type = (s.to_string(), s.to_string(), "".to_string());
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<MacdBatchOutput, MacdError> {
        macd_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MacdBatchOutput, MacdError> {
        MacdBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MacdBatchOutput, MacdError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MacdBatchOutput, MacdError> {
        MacdBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn macd_batch_with_kernel(
    data: &[f64],
    sweep: &MacdBatchRange,
    k: Kernel,
) -> Result<MacdBatchOutput, MacdError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(MacdError::InvalidKernel(format!(
                "{:?} is not a valid batch kernel",
                k
            )));
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    macd_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MacdBatchOutput {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub hist: Vec<f64>,
    pub combos: Vec<MacdParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
pub fn expand_grid(r: &MacdBatchRange) -> Vec<MacdParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let fasts = axis_usize(r.fast_period);
    let slows = axis_usize(r.slow_period);
    let signals = axis_usize(r.signal_period);
    let ma_types = vec![r.ma_type.0.clone()]; // For now, static MA type

    let mut combos = vec![];
    for &f in &fasts {
        for &s in &slows {
            for &g in &signals {
                for t in &ma_types {
                    combos.push(MacdParams {
                        fast_period: Some(f),
                        slow_period: Some(s),
                        signal_period: Some(g),
                        ma_type: Some(t.clone()),
                    });
                }
            }
        }
    }
    combos
}

pub fn macd_batch_par_slice(
    data: &[f64],
    sweep: &MacdBatchRange,
    _simd: Kernel,
) -> Result<MacdBatchOutput, MacdError> {
    let combos = expand_grid(sweep);
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(MacdError::AllValuesNaN);
    }

    // allocate 3 matrices uninitialized
    let mut macd_mu = make_uninit_matrix(rows, cols);
    let mut sig_mu = make_uninit_matrix(rows, cols);
    let mut hist_mu = make_uninit_matrix(rows, cols);

    // warmups per row
    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let macd_warm: Vec<usize> = combos
        .iter()
        .map(|p| {
            let slow = p.slow_period.unwrap_or(26);
            first + slow - 1
        })
        .collect();
    let warm: Vec<usize> = combos
        .iter()
        .map(|p| {
            let slow = p.slow_period.unwrap_or(26);
            let signal = p.signal_period.unwrap_or(9);
            first + slow + signal - 2
        })
        .collect();

    init_matrix_prefixes(&mut macd_mu, cols, &macd_warm);
    init_matrix_prefixes(&mut sig_mu, cols, &warm);
    init_matrix_prefixes(&mut hist_mu, cols, &warm);

    // cast to &mut [f64]
    let mut macd_guard = core::mem::ManuallyDrop::new(macd_mu);
    let mut sig_guard = core::mem::ManuallyDrop::new(sig_mu);
    let mut hist_guard = core::mem::ManuallyDrop::new(hist_mu);

    let macd_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(macd_guard.as_mut_ptr() as *mut f64, macd_guard.len())
    };
    let sig_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(sig_guard.as_mut_ptr() as *mut f64, sig_guard.len())
    };
    let hist_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(hist_guard.as_mut_ptr() as *mut f64, hist_guard.len())
    };

    // row-wise direct compute, no extra Vecs copied back
    for (row, prm) in combos.iter().enumerate() {
        let fast = prm.fast_period.unwrap_or(12);
        let slow = prm.slow_period.unwrap_or(26);
        let signal = prm.signal_period.unwrap_or(9);
        let ma_t = prm.ma_type.as_deref().unwrap_or("ema");

        let r0 = row * cols;
        let r1 = r0 + cols;

        // Ignore errors for individual rows - they're already NaN-filled
        let _ = macd_compute_into(
            data,
            fast,
            slow,
            signal,
            ma_t,
            first,
            &mut macd_out[r0..r1],
            &mut sig_out[r0..r1],
            &mut hist_out[r0..r1],
        );
    }

    // turn matrices into Vecs
    let macd = unsafe {
        Vec::from_raw_parts(
            macd_guard.as_mut_ptr() as *mut f64,
            macd_guard.len(),
            macd_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            sig_guard.as_mut_ptr() as *mut f64,
            sig_guard.len(),
            sig_guard.capacity(),
        )
    };
    let hist = unsafe {
        Vec::from_raw_parts(
            hist_guard.as_mut_ptr() as *mut f64,
            hist_guard.len(),
            hist_guard.capacity(),
        )
    };

    Ok(MacdBatchOutput {
        macd,
        signal,
        hist,
        combos,
        rows,
        cols,
    })
}

#[cfg(any(feature = "python", feature = "wasm"))]
pub fn macd_batch_inner_into(
    data: &[f64],
    sweep: &MacdBatchRange,
    _simd: Kernel,
    _fill_invalid: bool,
    macd_out: &mut [f64],
    signal_out: &mut [f64],
    hist_out: &mut [f64],
) -> Result<Vec<MacdParams>, MacdError> {
    let combos = expand_grid(sweep);
    let rows = combos.len();
    let cols = data.len();
    if macd_out.len() != rows * cols
        || signal_out.len() != rows * cols
        || hist_out.len() != rows * cols
    {
        return Err(MacdError::InvalidPeriod {
            fast: 0,
            slow: 0,
            signal: 0,
            data_len: data.len(),
        });
    }
    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

    for (row, prm) in combos.iter().enumerate() {
        let r0 = row * cols;
        let r1 = r0 + cols;

        // Initialize the row with NaN first
        let fast_period = prm.fast_period.unwrap_or(12);
        let slow_period = prm.slow_period.unwrap_or(26);
        let signal_period = prm.signal_period.unwrap_or(9);
        let macd_warmup = first + slow_period - 1;
        let signal_warmup = first + slow_period + signal_period - 2;

        // Fill with NaN up to warmup periods
        for i in 0..macd_warmup.min(cols) {
            macd_out[r0 + i] = f64::NAN;
        }
        for i in 0..signal_warmup.min(cols) {
            signal_out[r0 + i] = f64::NAN;
            hist_out[r0 + i] = f64::NAN;
        }

        let _ = macd_compute_into(
            data,
            fast_period,
            slow_period,
            signal_period,
            prm.ma_type.as_deref().unwrap_or("ema"),
            first,
            &mut macd_out[r0..r1],
            &mut signal_out[r0..r1],
            &mut hist_out[r0..r1],
        );
    }
    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "macd")]
#[pyo3(signature = (data, fast_period, slow_period, signal_period, ma_type, kernel=None))]
pub fn macd_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    use numpy::PyArray1;

    let slice_in = data.as_slice()?;
    let len = slice_in.len();

    // Prepare warmups exactly once
    let first = slice_in.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let macd_warmup = first + slow_period - 1;
    let signal_warmup = first + slow_period + signal_period - 2;

    // Allocate Py arrays and mark warm prefixes via our helper outputs
    let macd_arr = unsafe { PyArray1::<f64>::new(py, [len], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [len], false) };
    let hist_arr = unsafe { PyArray1::<f64>::new(py, [len], false) };

    let macd_slice = unsafe { macd_arr.as_slice_mut()? };
    let signal_slice = unsafe { signal_arr.as_slice_mut()? };
    let hist_slice = unsafe { hist_arr.as_slice_mut()? };

    // Initialize prefixes with our allocator semantics (check bounds)
    if macd_warmup <= len {
        macd_slice[..macd_warmup].fill(f64::from_bits(0x7ff8_0000_0000_0000));
    } else {
        macd_slice.fill(f64::from_bits(0x7ff8_0000_0000_0000));
    }
    if signal_warmup <= len {
        signal_slice[..signal_warmup].fill(f64::from_bits(0x7ff8_0000_0000_0000));
        hist_slice[..signal_warmup].fill(f64::from_bits(0x7ff8_0000_0000_0000));
    } else {
        signal_slice.fill(f64::from_bits(0x7ff8_0000_0000_0000));
        hist_slice.fill(f64::from_bits(0x7ff8_0000_0000_0000));
    }

    let kern = validate_kernel(kernel, false)?;

    // Create MacdInput and call macd_with_kernel to respect kernel parameter
    let params = MacdParams {
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
        signal_period: Some(signal_period),
        ma_type: Some(ma_type.to_string()),
    };
    let input = MacdInput::from_slice(slice_in, params);

    let result = py
        .allow_threads(|| macd_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Copy results to pre-allocated arrays
    macd_slice.copy_from_slice(&result.macd);
    signal_slice.copy_from_slice(&result.signal);
    hist_slice.copy_from_slice(&result.hist);

    Ok((macd_arr, signal_arr, hist_arr))
}

#[cfg(feature = "python")]
#[pyclass(name = "MacdStream")]
pub struct MacdStreamPy {
    stream: MacdStream,
    data_buffer: Vec<f64>, // Keep buffer for non-EMA fallback
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    ma_type: String,
}

#[cfg(feature = "python")]
#[pymethods]
impl MacdStreamPy {
    #[new]
    fn new(
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
        ma_type: &str,
    ) -> PyResult<Self> {
        Ok(MacdStreamPy {
            stream: MacdStream::new(fast_period, slow_period, signal_period, ma_type),
            data_buffer: Vec::new(),
            fast_period,
            slow_period,
            signal_period,
            ma_type: ma_type.to_string(),
        })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        // Try optimized EMA stream first
        if let Some(result) = self.stream.update(value) {
            return Some(result);
        }

        // Fallback for non-EMA MA types
        if !self.ma_type.eq_ignore_ascii_case("ema") {
            self.data_buffer.push(value);

            // Need at least slow_period + signal_period - 1 values
            let min_needed = self.slow_period + self.signal_period - 1;
            if self.data_buffer.len() < min_needed {
                return None;
            }

            // Calculate MACD on the buffer
            let params = MacdParams {
                fast_period: Some(self.fast_period),
                slow_period: Some(self.slow_period),
                signal_period: Some(self.signal_period),
                ma_type: Some(self.ma_type.clone()),
            };
            let input = MacdInput::from_slice(&self.data_buffer, params);

            match macd(&input) {
                Ok(output) => {
                    let last_idx = output.macd.len() - 1;
                    Some((
                        output.macd[last_idx],
                        output.signal[last_idx],
                        output.hist[last_idx],
                    ))
                }
                Err(_) => None,
            }
        } else {
            None
        }
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "macd_batch")]
#[pyo3(signature = (data, fast_period_range, slow_period_range, signal_period_range, ma_type, kernel=None))]
pub fn macd_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    fast_period_range: (usize, usize, usize),
    slow_period_range: (usize, usize, usize),
    signal_period_range: (usize, usize, usize),
    ma_type: &str,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    // Check for empty input
    if slice_in.is_empty() {
        return Err(PyValueError::new_err("macd: Input data slice is empty"));
    }

    // Check if all values are NaN
    if slice_in.iter().all(|x| x.is_nan()) {
        return Err(PyValueError::new_err("macd: All values are NaN"));
    }

    let kern = validate_kernel(kernel, true)?;

    let sweep = MacdBatchRange {
        fast_period: fast_period_range,
        slow_period: slow_period_range,
        signal_period: signal_period_range,
        ma_type: (ma_type.to_string(), ma_type.to_string(), String::new()),
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate output arrays
    let macd_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let hist_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };

    let macd_slice = unsafe { macd_arr.as_slice_mut()? };
    let signal_slice = unsafe { signal_arr.as_slice_mut()? };
    let hist_slice = unsafe { hist_arr.as_slice_mut()? };

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
            macd_batch_inner_into(
                slice_in,
                &sweep,
                simd,
                true,
                macd_slice,
                signal_slice,
                hist_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("macd", macd_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item("hist", hist_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "fast_periods",
        combos
            .iter()
            .map(|p| p.fast_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "slow_periods",
        combos
            .iter()
            .map(|p| p.slow_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "signal_periods",
        combos
            .iter()
            .map(|p| p.signal_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_macd_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(macd_py, m)?)?;
    m.add_function(wrap_pyfunction!(macd_batch_py, m)?)?;
    Ok(())
}

// ==================== PYTHON: CUDA BINDINGS (EMA only) ====================
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "macd_cuda_batch_dev")]
#[pyo3(signature = (data_f32, fast_range, slow_range, signal_range, ma_type="ema", device_id=0))]
pub fn macd_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    data_f32: numpy::PyReadonlyArray1<'py, f32>,
    fast_range: (usize, usize, usize),
    slow_range: (usize, usize, usize),
    signal_range: (usize, usize, usize),
    ma_type: &str,
    device_id: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use crate::cuda::cuda_available;
    use crate::cuda::oscillators::CudaMacd;
    use crate::cuda::oscillators::macd_wrapper::DeviceMacdTriplet;
    use numpy::{IntoPyArray, PyList};

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    if !ma_type.eq_ignore_ascii_case("ema") {
        return Err(PyValueError::new_err("macd_cuda: only ma_type=\"ema\" is supported on CUDA"));
    }
    let slice = data_f32.as_slice()?;
    let sweep = MacdBatchRange {
        fast_period: fast_range,
        slow_period: slow_range,
        signal_period: signal_range,
        ma_type: (ma_type.to_string(), ma_type.to_string(), String::new()),
    };

    let (outputs, combos) = py.allow_threads(|| {
        let cuda = CudaMacd::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.macd_batch_dev(slice, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    let DeviceMacdTriplet { macd, signal, hist } = outputs;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("macd", Py::new(py, super::moving_averages::alma::DeviceArrayF32Py { inner: macd })?)?;
    dict.set_item("signal", Py::new(py, super::moving_averages::alma::DeviceArrayF32Py { inner: signal })?)?;
    dict.set_item("hist", Py::new(py, super::moving_averages::alma::DeviceArrayF32Py { inner: hist })?)?;

    let fasts: Vec<u64> = combos.iter().map(|p| p.fast_period.unwrap() as u64).collect();
    let slows: Vec<u64> = combos.iter().map(|p| p.slow_period.unwrap() as u64).collect();
    let signals: Vec<u64> = combos.iter().map(|p| p.signal_period.unwrap() as u64).collect();
    let ma_types = PyList::new(py, vec![ma_type; combos.len()])?;
    dict.set_item("fast_periods", fasts.into_pyarray(py))?;
    dict.set_item("slow_periods", slows.into_pyarray(py))?;
    dict.set_item("signal_periods", signals.into_pyarray(py))?;
    dict.set_item("ma_types", ma_types)?;
    dict.set_item("rows", combos.len())?;
    dict.set_item("cols", slice.len())?;
    Ok(dict)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "macd_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, fast_period, slow_period, signal_period, ma_type="ema", device_id=0))]
pub fn macd_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    ma_type: &str,
    device_id: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use crate::cuda::cuda_available;
    use crate::cuda::oscillators::CudaMacd;
    use crate::cuda::oscillators::macd_wrapper::DeviceMacdTriplet;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    if !ma_type.eq_ignore_ascii_case("ema") {
        return Err(PyValueError::new_err("macd_cuda: only ma_type=\"ema\" is supported on CUDA"));
    }
    let shape = data_tm_f32.shape();
    if shape.len() != 2 { return Err(PyValueError::new_err("expected 2D array")); }
    let rows = shape[0];
    let cols = shape[1];
    let flat = data_tm_f32.as_slice()?;
    let params = MacdParams {
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
        signal_period: Some(signal_period),
        ma_type: Some(ma_type.to_string()),
    };
    let DeviceMacdTriplet { macd, signal, hist } = py.allow_threads(|| {
        let cuda = CudaMacd::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.macd_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("macd", Py::new(py, super::moving_averages::alma::DeviceArrayF32Py { inner: macd })?)?;
    dict.set_item("signal", Py::new(py, super::moving_averages::alma::DeviceArrayF32Py { inner: signal })?)?;
    dict.set_item("hist", Py::new(py, super::moving_averages::alma::DeviceArrayF32Py { inner: hist })?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("fast_period", fast_period)?;
    dict.set_item("slow_period", slow_period)?;
    dict.set_item("signal_period", signal_period)?;
    dict.set_item("ma_type", ma_type)?;
    Ok(dict)
}

// =============================================================================
// WASM BINDINGS
// =============================================================================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct MacdResult {
    values: Vec<f64>, // [macd..., signal..., hist...]
    rows: usize,      // 3 for MACD
    cols: usize,      // data length
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl MacdResult {
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f64> {
        self.values.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.cols
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn macd_js(
    data: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    ma_type: &str,
) -> Result<MacdResult, JsValue> {
    let len = data.len();
    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let macd_warmup = first + slow_period - 1;
    let signal_warmup = first + slow_period + signal_period - 2;

    let mut macd = alloc_with_nan_prefix(len, macd_warmup);
    let mut signal = alloc_with_nan_prefix(len, signal_warmup);
    let mut hist = alloc_with_nan_prefix(len, signal_warmup);

    macd_compute_into(
        data,
        fast_period,
        slow_period,
        signal_period,
        ma_type,
        first,
        &mut macd,
        &mut signal,
        &mut hist,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // concatenate without extra work
    let mut values = Vec::with_capacity(3 * len);
    values.extend_from_slice(&macd);
    values.extend_from_slice(&signal);
    values.extend_from_slice(&hist);

    Ok(MacdResult {
        values,
        rows: 3,
        cols: len,
    })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn macd_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn macd_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn macd_into(
    in_ptr: *const f64,
    macd_ptr: *mut f64,
    signal_ptr: *mut f64,
    hist_ptr: *mut f64,
    len: usize,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    ma_type: &str,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || macd_ptr.is_null() || signal_ptr.is_null() || hist_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = MacdParams {
            fast_period: Some(fast_period),
            slow_period: Some(slow_period),
            signal_period: Some(signal_period),
            ma_type: Some(ma_type.to_string()),
        };
        let input = MacdInput::from_slice(data, params);

        // Check for aliasing with ALL output pointers
        let needs_temp = in_ptr == macd_ptr as *const f64
            || in_ptr == signal_ptr as *const f64
            || in_ptr == hist_ptr as *const f64;

        if needs_temp {
            // Calculate warmup periods
            let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
            let macd_warmup = first + slow_period - 1;
            let signal_warmup = first + slow_period + signal_period - 2;

            // Use temporary buffers with proper NaN prefixes
            let mut temp_macd = alloc_with_nan_prefix(len, macd_warmup);
            let mut temp_signal = alloc_with_nan_prefix(len, signal_warmup);
            let mut temp_hist = alloc_with_nan_prefix(len, signal_warmup);

            macd_compute_into(
                data,
                fast_period,
                slow_period,
                signal_period,
                ma_type,
                first,
                &mut temp_macd,
                &mut temp_signal,
                &mut temp_hist,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results to output pointers
            let macd_out = std::slice::from_raw_parts_mut(macd_ptr, len);
            let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);
            let hist_out = std::slice::from_raw_parts_mut(hist_ptr, len);

            macd_out.copy_from_slice(&temp_macd);
            signal_out.copy_from_slice(&temp_signal);
            hist_out.copy_from_slice(&temp_hist);
        } else {
            // Non-aliasing case: call macd() and copy results directly
            // This is simpler and avoids issues with macd_compute_into expectations
            let result = macd(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results to output pointers
            let macd_out = std::slice::from_raw_parts_mut(macd_ptr, len);
            let signal_out = std::slice::from_raw_parts_mut(signal_ptr, len);
            let hist_out = std::slice::from_raw_parts_mut(hist_ptr, len);

            macd_out.copy_from_slice(&result.macd);
            signal_out.copy_from_slice(&result.signal);
            hist_out.copy_from_slice(&result.hist);
        }

        Ok(())
    }
}

// Batch processing structures
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MacdBatchConfig {
    pub fast_period_range: (usize, usize, usize),
    pub slow_period_range: (usize, usize, usize),
    pub signal_period_range: (usize, usize, usize),
    pub ma_type: String,
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MacdBatchJsOutput {
    pub values: Vec<f64>, // [macd block | signal block | hist block], each length rows*cols
    pub rows: usize,
    pub cols: usize,
    pub fast_periods: Vec<usize>,
    pub slow_periods: Vec<usize>,
    pub signal_periods: Vec<usize>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = macd_batch)]
pub fn macd_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    // Check for empty input
    if data.is_empty() {
        return Err(JsValue::from_str("macd: Input data slice is empty"));
    }

    // Check if all values are NaN
    if data.iter().all(|x| x.is_nan()) {
        return Err(JsValue::from_str("macd: All values are NaN"));
    }

    let config: MacdBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = MacdBatchRange {
        fast_period: config.fast_period_range,
        slow_period: config.slow_period_range,
        signal_period: config.signal_period_range,
        ma_type: (
            config.ma_type.clone(),
            config.ma_type.clone(),
            String::new(),
        ),
    };
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = data.len();

    // pre-alloc three matrices via helpers (+ NaN prefixes in caller if desired)
    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let macd_warm: Vec<usize> = combos
        .iter()
        .map(|p| first + p.slow_period.unwrap_or(26) - 1)
        .collect();
    let sig_warm: Vec<usize> = combos
        .iter()
        .map(|p| first + p.slow_period.unwrap_or(26) + p.signal_period.unwrap_or(9) - 2)
        .collect();

    let mut macd_mu = make_uninit_matrix(rows, cols);
    let mut sig_mu = make_uninit_matrix(rows, cols);
    let mut hist_mu = make_uninit_matrix(rows, cols);

    init_matrix_prefixes(&mut macd_mu, cols, &macd_warm);
    init_matrix_prefixes(&mut sig_mu, cols, &sig_warm);
    init_matrix_prefixes(&mut hist_mu, cols, &sig_warm);

    let mut macd_guard = core::mem::ManuallyDrop::new(macd_mu);
    let mut sig_guard = core::mem::ManuallyDrop::new(sig_mu);
    let mut hist_guard = core::mem::ManuallyDrop::new(hist_mu);

    let macd_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(macd_guard.as_mut_ptr() as *mut f64, macd_guard.len())
    };
    let sig_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(sig_guard.as_mut_ptr() as *mut f64, sig_guard.len())
    };
    let hist_out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(hist_guard.as_mut_ptr() as *mut f64, hist_guard.len())
    };

    macd_batch_inner_into(
        data,
        &sweep,
        detect_best_kernel(),
        true,
        macd_out,
        sig_out,
        hist_out,
    )
    .map_err(|e| JsValue::from_str(&format!("Batch computation error: {}", e)))?;

    let macd = unsafe {
        Vec::from_raw_parts(
            macd_guard.as_mut_ptr() as *mut f64,
            macd_guard.len(),
            macd_guard.capacity(),
        )
    };
    let sig = unsafe {
        Vec::from_raw_parts(
            sig_guard.as_mut_ptr() as *mut f64,
            sig_guard.len(),
            sig_guard.capacity(),
        )
    };
    let hist = unsafe {
        Vec::from_raw_parts(
            hist_guard.as_mut_ptr() as *mut f64,
            hist_guard.len(),
            hist_guard.capacity(),
        )
    };

    let mut values = Vec::with_capacity(3 * rows * cols);
    values.extend_from_slice(&macd);
    values.extend_from_slice(&sig);
    values.extend_from_slice(&hist);

    let out = MacdBatchJsOutput {
        values,
        rows,
        cols,
        fast_periods: combos.iter().map(|p| p.fast_period.unwrap()).collect(),
        slow_periods: combos.iter().map(|p| p.slow_period.unwrap()).collect(),
        signal_periods: combos.iter().map(|p| p.signal_period.unwrap()).collect(),
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::utilities::enums::Kernel;

    fn check_macd_partial_params(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;

        let default_params = MacdParams {
            fast_period: None,
            slow_period: None,
            signal_period: None,
            ma_type: None,
        };
        let input = MacdInput::from_candles(&candles, "close", default_params);
        let output = macd_with_kernel(&input, kernel)?;
        assert_eq!(output.macd.len(), candles.close.len());
        assert_eq!(output.signal.len(), candles.close.len());
        assert_eq!(output.hist.len(), candles.close.len());
        Ok(())
    }

    fn check_macd_accuracy(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;

        let params = MacdParams::default();
        let input = MacdInput::from_candles(&candles, "close", params);
        let result = macd_with_kernel(&input, kernel)?;

        let expected_macd = [
            -629.8674025082801,
            -600.2986584356258,
            -581.6188884820076,
            -551.1020443476082,
            -560.798510688488,
        ];
        let expected_signal = [
            -721.9744591891067,
            -697.6392990384105,
            -674.4352169271299,
            -649.7685824112256,
            -631.9745680666781,
        ];
        let expected_hist = [
            92.10705668082664,
            97.34064060278467,
            92.81632844512228,
            98.6665380636174,
            71.17605737819008,
        ];
        let len = result.macd.len();
        let start = len - 5;
        for i in 0..5 {
            assert!((result.macd[start + i] - expected_macd[i]).abs() < 1e-1);
            assert!((result.signal[start + i] - expected_signal[i]).abs() < 1e-1);
            assert!((result.hist[start + i] - expected_hist[i]).abs() < 1e-1);
        }
        Ok(())
    }

    fn check_macd_zero_period(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let input_data = [10.0, 20.0, 30.0];
        let params = MacdParams {
            fast_period: Some(0),
            slow_period: Some(26),
            signal_period: Some(9),
            ma_type: Some("ema".to_string()),
        };
        let input = MacdInput::from_slice(&input_data, params);
        let res = macd_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MACD should fail with zero fast period",
            test
        );
        Ok(())
    }

    fn check_macd_period_exceeds_length(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [10.0, 20.0, 30.0];
        let params = MacdParams {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            ma_type: Some("ema".to_string()),
        };
        let input = MacdInput::from_slice(&data, params);
        let res = macd_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MACD should fail with period exceeding length",
            test
        );
        Ok(())
    }

    fn check_macd_very_small_dataset(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let data = [42.0];
        let params = MacdParams {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            ma_type: Some("ema".to_string()),
        };
        let input = MacdInput::from_slice(&data, params);
        let res = macd_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MACD should fail with insufficient data",
            test
        );
        Ok(())
    }

    fn check_macd_reinput(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;

        let params = MacdParams::default();
        let input = MacdInput::from_candles(&candles, "close", params.clone());
        let first_result = macd_with_kernel(&input, kernel)?;

        let reinput = MacdInput::from_slice(&first_result.macd, params);
        let re_result = macd_with_kernel(&reinput, kernel)?;

        assert_eq!(re_result.macd.len(), first_result.macd.len());
        for i in 52..re_result.macd.len() {
            assert!(!re_result.macd[i].is_nan());
        }
        Ok(())
    }

    fn check_macd_nan_handling(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file)?;

        let params = MacdParams::default();
        let input = MacdInput::from_candles(&candles, "close", params);
        let res = macd_with_kernel(&input, kernel)?;
        let n = res.macd.len();
        if n > 240 {
            for i in 240..n {
                assert!(!res.macd[i].is_nan());
                assert!(!res.signal[i].is_nan());
                assert!(!res.hist[i].is_nan());
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_macd_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            MacdParams::default(),
            MacdParams {
                fast_period: Some(2),
                slow_period: Some(3),
                signal_period: Some(2),
                ma_type: Some("ema".to_string()),
            },
            MacdParams {
                fast_period: Some(5),
                slow_period: Some(10),
                signal_period: Some(5),
                ma_type: Some("ema".to_string()),
            },
            MacdParams {
                fast_period: Some(8),
                slow_period: Some(21),
                signal_period: Some(8),
                ma_type: Some("sma".to_string()),
            },
            MacdParams {
                fast_period: Some(20),
                slow_period: Some(50),
                signal_period: Some(15),
                ma_type: Some("ema".to_string()),
            },
            MacdParams {
                fast_period: Some(50),
                slow_period: Some(100),
                signal_period: Some(20),
                ma_type: Some("wma".to_string()),
            },
            MacdParams {
                fast_period: Some(3),
                slow_period: Some(6),
                signal_period: Some(3),
                ma_type: Some("sma".to_string()),
            },
            MacdParams {
                fast_period: Some(10),
                slow_period: Some(30),
                signal_period: Some(10),
                ma_type: Some("wma".to_string()),
            },
            MacdParams {
                fast_period: Some(15),
                slow_period: Some(35),
                signal_period: Some(12),
                ma_type: Some("ema".to_string()),
            },
            MacdParams {
                fast_period: Some(25),
                slow_period: Some(75),
                signal_period: Some(18),
                ma_type: Some("sma".to_string()),
            },
            MacdParams {
                fast_period: Some(6),
                slow_period: Some(13),
                signal_period: Some(4),
                ma_type: Some("ema".to_string()),
            },
            MacdParams {
                fast_period: Some(9),
                slow_period: Some(18),
                signal_period: Some(7),
                ma_type: Some("wma".to_string()),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = MacdInput::from_candles(&candles, "close", params.clone());
            let output = macd_with_kernel(&input, kernel)?;

            // Check MACD values
            for (i, &val) in output.macd.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						in MACD output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						in MACD output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						in MACD output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
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
						in signal output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						in signal output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						in signal output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }
            }

            // Check histogram values
            for (i, &val) in output.hist.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						in histogram output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						in histogram output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						in histogram output with params: {:?} (param set {})",
                        test_name, val, bits, i, params, param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_macd_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_macd_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy for generating realistic data and parameters
        let strat = (2usize..=20) // Fast period from 2-20
            .prop_flat_map(|fast_period| {
                // Slow period must be greater than fast period
                (fast_period + 1..=50).prop_flat_map(move |slow_period| {
                    // Signal period from 2-20
                    (2usize..=20).prop_flat_map(move |signal_period| {
                        // Generate base price and volatility
                        (100f64..10000f64, 0.0001f64..0.1f64).prop_flat_map(
                            move |(base_price, volatility)| {
                                // Data length: needs to be at least slow + signal + some extra for testing
                                let min_len = slow_period + signal_period + 10;
                                (min_len..400).prop_flat_map(move |data_len| {
                                    // Generate price changes with different scenarios
                                    let price_changes = prop::collection::vec(
                                        prop_oneof![
                                            // 60% normal market moves
                                            6 => (-volatility..volatility),
                                            // 10% constant (no change)
                                            1 => Just(0.0),
                                            // 15% uptrend
                                            15 => (0.0..volatility * 2.0),
                                            // 15% downtrend
                                            15 => (-volatility * 2.0..0.0),
                                        ],
                                        data_len,
                                    );

                                    price_changes.prop_map(move |changes| {
                                        // Build the price series
                                        let mut data = Vec::with_capacity(data_len);
                                        data.push(base_price);

                                        for i in 1..data_len {
                                            let prev = data[i - 1];
                                            let change = changes[i];
                                            let new_price = prev * (1.0 + change);
                                            // Ensure prices stay positive and reasonable
                                            data.push(new_price.max(1.0).min(1e6));
                                        }

                                        (data, fast_period, slow_period, signal_period)
                                    })
                                })
                            },
                        )
                    })
                })
            });

        proptest::test_runner::TestRunner::default().run(
            &strat,
            |(data, fast_period, slow_period, signal_period)| {
                // Create MACD input
                let params = MacdParams {
                    fast_period: Some(fast_period),
                    slow_period: Some(slow_period),
                    signal_period: Some(signal_period),
                    ma_type: Some("ema".to_string()),
                };
                let input = MacdInput::from_slice(&data, params.clone());

                // Compute MACD with the specified kernel
                let result = macd_with_kernel(&input, kernel)?;

                // Also compute with scalar kernel for comparison
                let reference = macd_with_kernel(&input, Kernel::Scalar)?;

                let len = data.len();

                // Property 1: Output lengths match input length
                prop_assert_eq!(result.macd.len(), len, "MACD output length mismatch");
                prop_assert_eq!(result.signal.len(), len, "Signal output length mismatch");
                prop_assert_eq!(result.hist.len(), len, "Histogram output length mismatch");

                // Calculate expected warmup periods (assuming clean data, first = 0)
                let macd_warmup = slow_period - 1;
                let signal_warmup = slow_period + signal_period - 2;

                // Property 2: Warmup period values are NaN
                for i in 0..macd_warmup.min(len) {
                    prop_assert!(
                        result.macd[i].is_nan(),
                        "MACD[{}] should be NaN during warmup (< {})",
                        i,
                        macd_warmup
                    );
                }

                for i in 0..signal_warmup.min(len) {
                    prop_assert!(
                        result.signal[i].is_nan(),
                        "Signal[{}] should be NaN during warmup (< {})",
                        i,
                        signal_warmup
                    );
                    prop_assert!(
                        result.hist[i].is_nan(),
                        "Histogram[{}] should be NaN during warmup (< {})",
                        i,
                        signal_warmup
                    );
                }

                // Property 3: Mathematical correctness - histogram = macd - signal
                for i in signal_warmup..len {
                    if !result.macd[i].is_nan() && !result.signal[i].is_nan() {
                        let expected_hist = result.macd[i] - result.signal[i];
                        prop_assert!(
                            (result.hist[i] - expected_hist).abs() < 1e-10,
                            "Histogram[{}] = {} != MACD - Signal = {} - {} = {}",
                            i,
                            result.hist[i],
                            result.macd[i],
                            result.signal[i],
                            expected_hist
                        );
                    }
                }

                // Property 4: Constant data produces MACD near zero
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
                    // For constant data, MACD should converge to 0
                    for i in signal_warmup..len {
                        if !result.macd[i].is_nan() {
                            prop_assert!(
                                result.macd[i].abs() < 1e-3,
                                "MACD[{}] = {} should be near 0 for constant data",
                                i,
                                result.macd[i]
                            );
                        }
                    }
                }

                // Property 5: MACD bounded by reasonable range
                // MACD is difference between two MAs, so it should be much smaller than the data values
                let data_range = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    - data.iter().cloned().fold(f64::INFINITY, f64::min);
                for i in macd_warmup..len {
                    if !result.macd[i].is_nan() {
                        prop_assert!(
                            result.macd[i].abs() <= data_range,
                            "MACD[{}] = {} exceeds data range {}",
                            i,
                            result.macd[i],
                            data_range
                        );
                    }
                }

                // Property 6: Monotonic data behavior
                // Check if data is monotonic increasing or decreasing
                let is_monotonic_inc = data.windows(2).all(|w| w[1] >= w[0] - 1e-10);
                let is_monotonic_dec = data.windows(2).all(|w| w[1] <= w[0] + 1e-10);

                if is_monotonic_inc && !data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
                    // For monotonic increasing data, MACD should eventually be positive
                    let stable_start = (signal_warmup + 10).min(len - 1);
                    if stable_start < len {
                        let stable_macd = &result.macd[stable_start..len];
                        let positive_count = stable_macd
                            .iter()
                            .filter(|&&v| !v.is_nan() && v > -1e-10)
                            .count();
                        let total_valid = stable_macd.iter().filter(|&&v| !v.is_nan()).count();
                        if total_valid > 0 {
                            prop_assert!(
                                positive_count as f64 / total_valid as f64 > 0.9,
                                "MACD should be mostly positive for monotonic increasing data"
                            );
                        }
                    }
                } else if is_monotonic_dec && !data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                {
                    // For monotonic decreasing data, MACD should eventually be negative
                    let stable_start = (signal_warmup + 10).min(len - 1);
                    if stable_start < len {
                        let stable_macd = &result.macd[stable_start..len];
                        let negative_count = stable_macd
                            .iter()
                            .filter(|&&v| !v.is_nan() && v < 1e-10)
                            .count();
                        let total_valid = stable_macd.iter().filter(|&&v| !v.is_nan()).count();
                        if total_valid > 0 {
                            prop_assert!(
                                negative_count as f64 / total_valid as f64 > 0.9,
                                "MACD should be mostly negative for monotonic decreasing data"
                            );
                        }
                    }
                }

                // Property 7: Edge case - minimum period difference
                if fast_period == slow_period - 1 {
                    // With minimum difference, MACD should be very small relative to data
                    let data_scale = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max).abs();
                    for i in signal_warmup..len {
                        if !result.macd[i].is_nan() && data_scale > 1e-10 {
                            let relative_macd = result.macd[i].abs() / data_scale;
                            prop_assert!(
								relative_macd < 0.1,
								"MACD[{}] relative magnitude {} too large for minimum period difference",
								i, relative_macd
							);
                        }
                    }
                }

                // Property 8: Kernel consistency - different kernels produce similar results
                for i in 0..len {
                    let macd_y = result.macd[i];
                    let macd_r = reference.macd[i];
                    let signal_y = result.signal[i];
                    let signal_r = reference.signal[i];
                    let hist_y = result.hist[i];
                    let hist_r = reference.hist[i];

                    // Check MACD consistency
                    if !macd_y.is_finite() || !macd_r.is_finite() {
                        prop_assert_eq!(
                            macd_y.to_bits(),
                            macd_r.to_bits(),
                            "MACD NaN/finite mismatch at index {}",
                            i
                        );
                    } else {
                        let ulp_diff = macd_y.to_bits().abs_diff(macd_r.to_bits());
                        prop_assert!(
                            (macd_y - macd_r).abs() <= 1e-9 || ulp_diff <= 5,
                            "MACD mismatch at index {}: {} vs {} (ULP={})",
                            i,
                            macd_y,
                            macd_r,
                            ulp_diff
                        );
                    }

                    // Check signal consistency
                    if !signal_y.is_finite() || !signal_r.is_finite() {
                        prop_assert_eq!(
                            signal_y.to_bits(),
                            signal_r.to_bits(),
                            "Signal NaN/finite mismatch at index {}",
                            i
                        );
                    } else {
                        let ulp_diff = signal_y.to_bits().abs_diff(signal_r.to_bits());
                        prop_assert!(
                            (signal_y - signal_r).abs() <= 1e-9 || ulp_diff <= 5,
                            "Signal mismatch at index {}: {} vs {} (ULP={})",
                            i,
                            signal_y,
                            signal_r,
                            ulp_diff
                        );
                    }

                    // Check histogram consistency
                    if !hist_y.is_finite() || !hist_r.is_finite() {
                        prop_assert_eq!(
                            hist_y.to_bits(),
                            hist_r.to_bits(),
                            "Histogram NaN/finite mismatch at index {}",
                            i
                        );
                    } else {
                        let ulp_diff = hist_y.to_bits().abs_diff(hist_r.to_bits());
                        prop_assert!(
                            (hist_y - hist_r).abs() <= 1e-9 || ulp_diff <= 5,
                            "Histogram mismatch at index {}: {} vs {} (ULP={})",
                            i,
                            hist_y,
                            hist_r,
                            ulp_diff
                        );
                    }
                }

                Ok(())
            },
        )?;

        Ok(())
    }

    macro_rules! generate_all_macd_tests {
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
    generate_all_macd_tests!(
        check_macd_partial_params,
        check_macd_accuracy,
        check_macd_zero_period,
        check_macd_period_exceeds_length,
        check_macd_very_small_dataset,
        check_macd_reinput,
        check_macd_nan_handling,
        check_macd_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_macd_tests!(check_macd_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = MacdBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = MacdParams::default();
        let row = output
            .combos
            .iter()
            .position(|prm| {
                prm.fast_period == def.fast_period
                    && prm.slow_period == def.slow_period
                    && prm.signal_period == def.signal_period
                    && prm.ma_type == def.ma_type
            })
            .expect("default row missing");
        let start = row * output.cols;
        let macd = &output.macd[start..start + output.cols];
        let signal = &output.signal[start..start + output.cols];
        let hist = &output.hist[start..start + output.cols];
        let expected_macd = [
            -629.8674025082801,
            -600.2986584356258,
            -581.6188884820076,
            -551.1020443476082,
            -560.798510688488,
        ];
        let len = macd.len();
        let s = len - 5;
        for i in 0..5 {
            assert!((macd[s + i] - expected_macd[i]).abs() < 1e-1);
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            // Small fast periods, moderate slow and signal
            (2, 10, 2, 10, 20, 2, 2, 6, 2),
            // Medium periods across all parameters
            (5, 25, 5, 26, 50, 5, 5, 15, 5),
            // Large periods
            (20, 40, 10, 50, 100, 10, 10, 20, 5),
            // Edge case: minimal ranges
            (2, 5, 1, 6, 10, 1, 2, 4, 1),
            // Standard MACD variations
            (10, 15, 1, 20, 30, 2, 8, 12, 1),
            // Wide period sweep
            (3, 30, 3, 26, 52, 13, 5, 20, 5),
            // Dense small range
            (2, 6, 1, 8, 12, 1, 3, 5, 1),
        ];

        for (cfg_idx, &(f_start, f_end, f_step, s_start, s_end, s_step, g_start, g_end, g_step)) in
            test_configs.iter().enumerate()
        {
            let output = MacdBatchBuilder::new()
                .kernel(kernel)
                .fast_period_range(f_start, f_end, f_step)
                .slow_period_range(s_start, s_end, s_step)
                .signal_period_range(g_start, g_end, g_step)
                .ma_type_static("ema")
                .apply_candles(&c, "close")?;

            // Check MACD values
            for (idx, &val) in output.macd.iter().enumerate() {
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
						at row {} col {} (flat index {}) in MACD output with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) in MACD output with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) in MACD output with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }
            }

            // Check signal values
            for (idx, &val) in output.signal.iter().enumerate() {
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
						at row {} col {} (flat index {}) in signal output with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) in signal output with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) in signal output with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }
            }

            // Check histogram values
            for (idx, &val) in output.hist.iter().enumerate() {
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
						at row {} col {} (flat index {}) in histogram output with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) in histogram output with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at row {} col {} (flat index {}) in histogram output with params: {:?}",
                        test, cfg_idx, val, bits, row, col, idx, combo
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
    gen_batch_tests!(check_batch_no_poison);
}
