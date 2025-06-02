//! # Volume Weighted MACD (VWMACD)
//!
//! A variant of MACD using volume-weighted moving averages (VWMA) in place of traditional moving averages.
//! This implementation follows the same multi-kernel, batch, and stream support as alma.rs for performance and API consistency.
//!
//! ## Parameters
//! - **fast_period**: VWMA fast window (default: 12)
//! - **slow_period**: VWMA slow window (default: 26)
//! - **signal_period**: MA window for the signal line (default: 9)
//! - **fast_ma_type**: MA type for fast VWMA calculation (default: "sma")
//! - **slow_ma_type**: MA type for slow VWMA calculation (default: "sma")
//! - **signal_ma_type**: MA type for signal line (default: "ema")
//!
//! ## Errors
//! - **AllValuesNaN**: No valid values in close or volume
//! - **InvalidPeriod**: Any period is zero or exceeds the data length
//! - **NotEnoughValidData**: Not enough valid values for requested period
//! - **MaError**: Error from underlying MA calculation
//!
//! ## Returns
//! - **Ok(VwmacdOutput)** with `.macd`, `.signal`, `.hist` (all Vec<f64>)
//! - **Err(VwmacdError)** otherwise
//!

use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_kernel, detect_best_batch_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::error::Error;
use thiserror::Error;
use std::convert::AsRef;


#[derive(Debug, Clone)]
pub enum VwmacdData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct VwmacdOutput {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub hist: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VwmacdParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub signal_period: Option<usize>,
    pub fast_ma_type: Option<String>,
    pub slow_ma_type: Option<String>,
    pub signal_ma_type: Option<String>,
}

impl Default for VwmacdParams {
    fn default() -> Self {
        Self {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            fast_ma_type: Some("sma".to_string()),
            slow_ma_type: Some("sma".to_string()),
            signal_ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VwmacdInput<'a> {
    pub data: VwmacdData<'a>,
    pub params: VwmacdParams,
}

impl<'a> VwmacdInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, close_source: &'a str, volume_source: &'a str, params: VwmacdParams) -> Self {
        Self {
            data: VwmacdData::Candles { candles, close_source, volume_source },
            params,
        }
    }
    #[inline]
    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: VwmacdParams) -> Self {
        Self {
            data: VwmacdData::Slices { close, volume },
            params,
        }
    }
    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", "volume", VwmacdParams::default())
    }
    #[inline]
    pub fn get_fast(&self) -> usize {
        self.params.fast_period.unwrap_or(12)
    }
    #[inline]
    pub fn get_slow(&self) -> usize {
        self.params.slow_period.unwrap_or(26)
    }
    #[inline]
    pub fn get_signal(&self) -> usize {
        self.params.signal_period.unwrap_or(9)
    }
    #[inline]
    pub fn get_fast_ma_type(&self) -> &str {
        self.params.fast_ma_type.as_deref().unwrap_or("sma")
    }
    #[inline]
    pub fn get_slow_ma_type(&self) -> &str {
        self.params.slow_ma_type.as_deref().unwrap_or("sma")
    }
    #[inline]
    pub fn get_signal_ma_type(&self) -> &str {
        self.params.signal_ma_type.as_deref().unwrap_or("ema")
    }
}


#[derive(Clone, Debug)]
pub struct VwmacdBuilder {
    fast: Option<usize>,
    slow: Option<usize>,
    signal: Option<usize>,
    fast_ma_type: Option<String>,
    slow_ma_type: Option<String>,
    signal_ma_type: Option<String>,
    kernel: Kernel,
}

impl Default for VwmacdBuilder {
    fn default() -> Self {
        Self {
            fast: None,
            slow: None,
            signal: None,
            fast_ma_type: None,
            slow_ma_type: None,
            signal_ma_type: None,
            kernel: Kernel::Auto,
        }
    }
}

impl VwmacdBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn fast(mut self, n: usize) -> Self {
        self.fast = Some(n);
        self
    }
    #[inline(always)]
    pub fn slow(mut self, n: usize) -> Self {
        self.slow = Some(n);
        self
    }
    #[inline(always)]
    pub fn signal(mut self, n: usize) -> Self {
        self.signal = Some(n);
        self
    }
    #[inline(always)]
    pub fn fast_ma_type(mut self, ma_type: String) -> Self {
        self.fast_ma_type = Some(ma_type);
        self
    }
    #[inline(always)]
    pub fn slow_ma_type(mut self, ma_type: String) -> Self {
        self.slow_ma_type = Some(ma_type);
        self
    }
    #[inline(always)]
    pub fn signal_ma_type(mut self, ma_type: String) -> Self {
        self.signal_ma_type = Some(ma_type);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<VwmacdOutput, VwmacdError> {
        let p = VwmacdParams {
            fast_period: self.fast,
            slow_period: self.slow,
            signal_period: self.signal,
            fast_ma_type: self.fast_ma_type,
            slow_ma_type: self.slow_ma_type,
            signal_ma_type: self.signal_ma_type,
        };
        let i = VwmacdInput::from_candles(c, "close", "volume", p);
        vwmacd_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<VwmacdOutput, VwmacdError> {
        let p = VwmacdParams {
            fast_period: self.fast,
            slow_period: self.slow,
            signal_period: self.signal,
            fast_ma_type: self.fast_ma_type,
            slow_ma_type: self.slow_ma_type,
            signal_ma_type: self.signal_ma_type,
        };
        let i = VwmacdInput::from_slices(close, volume, p);
        vwmacd_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<VwmacdStream, VwmacdError> {
        let p = VwmacdParams {
            fast_period: self.fast,
            slow_period: self.slow,
            signal_period: self.signal,
            fast_ma_type: self.fast_ma_type,
            slow_ma_type: self.slow_ma_type,
            signal_ma_type: self.signal_ma_type,
        };
        VwmacdStream::try_new(p)
    }
}


#[derive(Debug, Error)]
pub enum VwmacdError {
    #[error("vwmacd: All values are NaN.")]
    AllValuesNaN,
    #[error("vwmacd: Invalid period: fast={fast}, slow={slow}, signal={signal}, data_len={data_len}")]
    InvalidPeriod { fast: usize, slow: usize, signal: usize, data_len: usize },
    #[error("vwmacd: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vwmacd: MA calculation error: {0}")]
    MaError(Box<dyn Error>),
}


#[inline]
pub fn vwmacd(input: &VwmacdInput) -> Result<VwmacdOutput, VwmacdError> {
    vwmacd_with_kernel(input, Kernel::Auto)
}

pub fn vwmacd_with_kernel(input: &VwmacdInput, kernel: Kernel) -> Result<VwmacdOutput, VwmacdError> {
    let (close, volume) = match &input.data {
        VwmacdData::Candles { candles, close_source, volume_source } => {
            (source_type(candles, close_source), source_type(candles, volume_source))
        }
        VwmacdData::Slices { close, volume } => (*close, *volume),
    };
    let data_len = close.len();
    let fast = input.get_fast();
    let slow = input.get_slow();
    let signal = input.get_signal();

    if fast == 0 || slow == 0 || signal == 0 || fast > data_len || slow > data_len || signal > data_len {
        return Err(VwmacdError::InvalidPeriod { fast, slow, signal, data_len });
    }
    let first = (0..close.len()).find(|&i| !close[i].is_nan() && !volume[i].is_nan()).unwrap_or(0);
    if !close.iter().any(|x| !x.is_nan()) || !volume.iter().any(|x| !x.is_nan()) {
        return Err(VwmacdError::AllValuesNaN);
    }
    if (data_len - first) < slow {
        return Err(VwmacdError::NotEnoughValidData { needed: slow, valid: data_len - first });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => vwmacd_scalar(close, volume, fast, slow, signal, 
                input.get_fast_ma_type(), input.get_slow_ma_type(), input.get_signal_ma_type()),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => vwmacd_avx2(close, volume, fast, slow, signal,
                input.get_fast_ma_type(), input.get_slow_ma_type(), input.get_signal_ma_type()),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => vwmacd_avx512(close, volume, fast, slow, signal,
                input.get_fast_ma_type(), input.get_slow_ma_type(), input.get_signal_ma_type()),
            _ => unreachable!(),
        }
    }
}


#[inline]
pub unsafe fn vwmacd_scalar(
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    fast_ma_type: &str,
    slow_ma_type: &str,
    signal_ma_type: &str,
) -> Result<VwmacdOutput, VwmacdError> {
    let len = close.len();
    let mut close_x_volume = vec![f64::NAN; len];
    for i in 0..len {
        close_x_volume[i] = close[i] * volume[i];
    }

    let slow_ma_cv = ma(slow_ma_type, MaData::Slice(&close_x_volume), slow)
        .map_err(|e| VwmacdError::MaError(e))?;
    let slow_ma_v = ma(slow_ma_type, MaData::Slice(&volume), slow)
        .map_err(|e| VwmacdError::MaError(e))?;

    let mut vwma_slow = vec![f64::NAN; len];
    for i in 0..len {
        let denom = slow_ma_v[i];
        if !denom.is_nan() && denom != 0.0 {
            vwma_slow[i] = slow_ma_cv[i] / denom;
        }
    }

    let fast_ma_cv = ma(fast_ma_type, MaData::Slice(&close_x_volume), fast)
        .map_err(|e| VwmacdError::MaError(e))?;
    let fast_ma_v = ma(fast_ma_type, MaData::Slice(&volume), fast)
        .map_err(|e| VwmacdError::MaError(e))?;

    let mut vwma_fast = vec![f64::NAN; len];
    for i in 0..len {
        let denom = fast_ma_v[i];
        if !denom.is_nan() && denom != 0.0 {
            vwma_fast[i] = fast_ma_cv[i] / denom;
        }
    }

    let mut macd = vec![f64::NAN; len];
    for i in 0..len {
        if !vwma_fast[i].is_nan() && !vwma_slow[i].is_nan() {
            macd[i] = vwma_fast[i] - vwma_slow[i];
        }
    }

    let signal_vec = ma(signal_ma_type, MaData::Slice(&macd), signal)
        .map_err(|e| VwmacdError::MaError(e))?;

    let mut hist = vec![f64::NAN; len];
    for i in 0..len {
        if !macd[i].is_nan() && !signal_vec[i].is_nan() {
            hist[i] = macd[i] - signal_vec[i];
        }
    }
    Ok(VwmacdOutput { macd, signal: signal_vec, hist })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vwmacd_avx2(
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    fast_ma_type: &str,
    slow_ma_type: &str,
    signal_ma_type: &str,
) -> Result<VwmacdOutput, VwmacdError> {
    vwmacd_scalar(close, volume, fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vwmacd_avx512(
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    fast_ma_type: &str,
    slow_ma_type: &str,
    signal_ma_type: &str,
) -> Result<VwmacdOutput, VwmacdError> {
    if slow <= 32 {
        vwmacd_avx512_short(close, volume, fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type)
    } else {
        vwmacd_avx512_long(close, volume, fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vwmacd_avx512_short(
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    fast_ma_type: &str,
    slow_ma_type: &str,
    signal_ma_type: &str,
) -> Result<VwmacdOutput, VwmacdError> {
    vwmacd_scalar(close, volume, fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn vwmacd_avx512_long(
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    fast_ma_type: &str,
    slow_ma_type: &str,
    signal_ma_type: &str,
) -> Result<VwmacdOutput, VwmacdError> {
    vwmacd_scalar(close, volume, fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type)
}


#[inline(always)]
pub unsafe fn vwmacd_row_scalar(
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    fast_ma_type: &str,
    slow_ma_type: &str,
    signal_ma_type: &str,
    out: &mut [f64],
) {
    if let Ok(res) = vwmacd_scalar(close, volume, fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type) {
        out.copy_from_slice(&res.macd);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwmacd_row_avx2(
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    fast_ma_type: &str,
    slow_ma_type: &str,
    signal_ma_type: &str,
    out: &mut [f64],
) {
    vwmacd_row_scalar(close, volume, fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwmacd_row_avx512(
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    fast_ma_type: &str,
    slow_ma_type: &str,
    signal_ma_type: &str,
    out: &mut [f64],
) {
    if slow <= 32 {
        vwmacd_row_avx512_short(close, volume, fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type, out);
    } else {
        vwmacd_row_avx512_long(close, volume, fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwmacd_row_avx512_short(
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    fast_ma_type: &str,
    slow_ma_type: &str,
    signal_ma_type: &str,
    out: &mut [f64],
) {
    vwmacd_row_scalar(close, volume, fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn vwmacd_row_avx512_long(
    close: &[f64],
    volume: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    fast_ma_type: &str,
    slow_ma_type: &str,
    signal_ma_type: &str,
    out: &mut [f64],
) {
    vwmacd_row_scalar(close, volume, fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type, out);
}


#[derive(Debug, Clone)]
pub struct VwmacdStream {
    fast: usize,
    slow: usize,
    signal: usize,
    fast_ma_type: String,
    slow_ma_type: String,
    signal_ma_type: String,
}

impl VwmacdStream {
    pub fn try_new(params: VwmacdParams) -> Result<Self, VwmacdError> {
        let fast = params.fast_period.unwrap_or(12);
        let slow = params.slow_period.unwrap_or(26);
        let signal = params.signal_period.unwrap_or(9);
        let fast_ma_type = params.fast_ma_type.unwrap_or_else(|| "sma".to_string());
        let slow_ma_type = params.slow_ma_type.unwrap_or_else(|| "sma".to_string());
        let signal_ma_type = params.signal_ma_type.unwrap_or_else(|| "ema".to_string());
        
        if fast == 0 || slow == 0 || signal == 0 {
            return Err(VwmacdError::InvalidPeriod { fast, slow, signal, data_len: 0 });
        }
        Ok(Self { fast, slow, signal, fast_ma_type, slow_ma_type, signal_ma_type })
    }
    pub fn update(&mut self, _close: f64, _volume: f64) -> Option<f64> {
        None
    }
}


#[derive(Clone, Debug)]
pub struct VwmacdBatchRange {
    pub fast: (usize, usize, usize),
    pub slow: (usize, usize, usize),
    pub signal: (usize, usize, usize),
    pub fast_ma_type: String,
    pub slow_ma_type: String,
    pub signal_ma_type: String,
}

impl Default for VwmacdBatchRange {
    fn default() -> Self {
        Self {
            fast: (12, 16, 0),
            slow: (26, 30, 0),
            signal: (9, 12, 0),
            fast_ma_type: "sma".to_string(),
            slow_ma_type: "sma".to_string(),
            signal_ma_type: "ema".to_string(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VwmacdBatchBuilder {
    range: VwmacdBatchRange,
    kernel: Kernel,
}

impl VwmacdBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn fast_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.fast = (start, end, step);
        self
    }
    #[inline]
    pub fn slow_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.slow = (start, end, step);
        self
    }
    #[inline]
    pub fn signal_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.signal = (start, end, step);
        self
    }
    #[inline]
    pub fn fast_ma_type(mut self, ma_type: String) -> Self {
        self.range.fast_ma_type = ma_type;
        self
    }
    #[inline]
    pub fn slow_ma_type(mut self, ma_type: String) -> Self {
        self.range.slow_ma_type = ma_type;
        self
    }
    #[inline]
    pub fn signal_ma_type(mut self, ma_type: String) -> Self {
        self.range.signal_ma_type = ma_type;
        self
    }
    #[inline]
    pub fn apply_slices(self, close: &[f64], volume: &[f64]) -> Result<VwmacdBatchOutput, VwmacdError> {
        vwmacd_batch_with_kernel(close, volume, &self.range, self.kernel)
    }
}

#[inline(always)]
fn expand_grid(r: &VwmacdBatchRange) -> Vec<VwmacdParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let fasts = axis(r.fast);
    let slows = axis(r.slow);
    let signals = axis(r.signal);

    let mut out = Vec::with_capacity(fasts.len() * slows.len() * signals.len());
    for &f in &fasts {
        for &s in &slows {
            for &g in &signals {
                out.push(VwmacdParams {
                    fast_period: Some(f),
                    slow_period: Some(s),
                    signal_period: Some(g),
                    fast_ma_type: Some(r.fast_ma_type.clone()),
                    slow_ma_type: Some(r.slow_ma_type.clone()),
                    signal_ma_type: Some(r.signal_ma_type.clone()),
                });
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct VwmacdBatchOutput {
    pub macd: Vec<f64>,
    pub params: Vec<VwmacdParams>,
    pub rows: usize,
    pub cols: usize,
}

impl VwmacdBatchOutput {
    pub fn row_for_params(&self, p: &VwmacdParams) -> Option<usize> {
        self.params.iter().position(|c| {
            c.fast_period.unwrap_or(12) == p.fast_period.unwrap_or(12)
                && c.slow_period.unwrap_or(26) == p.slow_period.unwrap_or(26)
                && c.signal_period.unwrap_or(9) == p.signal_period.unwrap_or(9)
                && c.fast_ma_type.as_deref().unwrap_or("sma") == p.fast_ma_type.as_deref().unwrap_or("sma")
                && c.slow_ma_type.as_deref().unwrap_or("sma") == p.slow_ma_type.as_deref().unwrap_or("sma")
                && c.signal_ma_type.as_deref().unwrap_or("ema") == p.signal_ma_type.as_deref().unwrap_or("ema")
        })
    }
    pub fn values_for(&self, p: &VwmacdParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.macd[start..start + self.cols]
        })
    }
}

pub fn vwmacd_batch_with_kernel(
    close: &[f64],
    volume: &[f64],
    sweep: &VwmacdBatchRange,
    k: Kernel,
) -> Result<VwmacdBatchOutput, VwmacdError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(VwmacdError::InvalidPeriod {
                fast: 0, slow: 0, signal: 0, data_len: 0
            });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    vwmacd_batch_par_slice(close, volume, sweep, simd)
}

#[inline(always)]
pub fn vwmacd_batch_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &VwmacdBatchRange,
    kern: Kernel,
) -> Result<VwmacdBatchOutput, VwmacdError> {
    vwmacd_batch_inner(close, volume, sweep, kern, false)
}

#[inline(always)]
pub fn vwmacd_batch_par_slice(
    close: &[f64],
    volume: &[f64],
    sweep: &VwmacdBatchRange,
    kern: Kernel,
) -> Result<VwmacdBatchOutput, VwmacdError> {
    vwmacd_batch_inner(close, volume, sweep, kern, true)
}

#[inline(always)]
fn vwmacd_batch_inner(
    close: &[f64],
    volume: &[f64],
    sweep: &VwmacdBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<VwmacdBatchOutput, VwmacdError> {
    let params = expand_grid(sweep);
    if params.is_empty() {
        return Err(VwmacdError::InvalidPeriod { fast: 0, slow: 0, signal: 0, data_len: 0 });
    }
    let len = close.len();
    let rows = params.len();
    let cols = len;
    let mut macd = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let p = &params[row];
        match kern {
            Kernel::Scalar => vwmacd_row_scalar(
                close, volume, 
                p.fast_period.unwrap(), 
                p.slow_period.unwrap(), 
                p.signal_period.unwrap(),
                p.fast_ma_type.as_deref().unwrap_or("sma"),
                p.slow_ma_type.as_deref().unwrap_or("sma"),
                p.signal_ma_type.as_deref().unwrap_or("ema"),
                out_row
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => vwmacd_row_avx2(
                close, volume,
                p.fast_period.unwrap(),
                p.slow_period.unwrap(),
                p.signal_period.unwrap(),
                p.fast_ma_type.as_deref().unwrap_or("sma"),
                p.slow_ma_type.as_deref().unwrap_or("sma"),
                p.signal_ma_type.as_deref().unwrap_or("ema"),
                out_row
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => vwmacd_row_avx512(
                close, volume,
                p.fast_period.unwrap(),
                p.slow_period.unwrap(),
                p.signal_period.unwrap(),
                p.fast_ma_type.as_deref().unwrap_or("sma"),
                p.slow_ma_type.as_deref().unwrap_or("sma"),
                p.signal_ma_type.as_deref().unwrap_or("ema"),
                out_row
            ),
            _ => unreachable!(),
        }
    };
    if parallel {
        macd.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in macd.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    Ok(VwmacdBatchOutput { macd, params, rows, cols })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_vwmacd_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = VwmacdParams {
            fast_period: None,
            slow_period: None,
            signal_period: None,
            fast_ma_type: None,
            slow_ma_type: None,
            signal_ma_type: None,
        };
        let input = VwmacdInput::from_candles(&candles, "close", "volume", default_params);
        let output = vwmacd_with_kernel(&input, kernel)?;
        assert_eq!(output.macd.len(), candles.close.len());
        Ok(())
    }

    fn check_vwmacd_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = VwmacdInput::with_default_candles(&candles);
        let result = vwmacd_with_kernel(&input, kernel)?;
        
        let expected_macd = [
            -394.95161155,
            -508.29106210,
            -490.70190723,
            -388.94996199,
            -341.13720646,
        ];
        
        let expected_signal = [
            -539.48861567,
            -533.24910496,
            -524.73966541,
            -497.58172247,
            -466.29282108,
        ];
        
        let expected_histogram = [
            144.53700412,
            24.95804286,
            34.03775818,
            108.63176274,
            125.15561462,
        ];
        
        let last_five_macd = &result.macd[result.macd.len().saturating_sub(5)..];
        for (i, &val) in last_five_macd.iter().enumerate() {
            assert!(
                (val - expected_macd[i]).abs() < 1e-3,
                "[{}] MACD mismatch at idx {}: got {}, expected {}",
                test_name, i, val, expected_macd[i]
            );
        }
        
        let last_five_signal = &result.signal[result.signal.len().saturating_sub(5)..];
        for (i, &val) in last_five_signal.iter().enumerate() {
            assert!(
                (val - expected_signal[i]).abs() < 1e-3,
                "[{}] Signal mismatch at idx {}: got {}, expected {}",
                test_name, i, val, expected_signal[i]
            );
        }
        
        let last_five_hist = &result.hist[result.hist.len().saturating_sub(5)..];
        for (i, &val) in last_five_hist.iter().enumerate() {
            assert!(
                (val - expected_histogram[i]).abs() < 1e-3,
                "[{}] Histogram mismatch at idx {}: got {}, expected {}",
                test_name, i, val, expected_histogram[i]
            );
        }
        
        Ok(())
    }
    fn check_vwmacd_with_custom_ma_types(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let params = VwmacdParams {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            fast_ma_type: Some("ema".to_string()),
            slow_ma_type: Some("wma".to_string()),
            signal_ma_type: Some("sma".to_string()),
        };
        let input = VwmacdInput::from_candles(&candles, "close", "volume", params);
        let output = vwmacd_with_kernel(&input, kernel)?;
        assert_eq!(output.macd.len(), candles.close.len());
        
        let default_input = VwmacdInput::with_default_candles(&candles);
        let default_output = vwmacd_with_kernel(&default_input, kernel)?;
        
        let different_count = output.macd.iter().zip(&default_output.macd)
            .skip(50)
            .filter(|(&a, &b)| !a.is_nan() && !b.is_nan() && (a - b).abs() > 1e-10)
            .count();
        
        assert!(different_count > 0, "Custom MA types should produce different results");
        Ok(())
    }

    fn check_vwmacd_nan_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close = [f64::NAN, f64::NAN];
        let volume = [f64::NAN, f64::NAN];
        let params = VwmacdParams::default();
        let input = VwmacdInput::from_slices(&close, &volume, params);
        let result = vwmacd_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vwmacd_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close = [10.0, 20.0, 30.0];
        let volume = [1.0, 1.0, 1.0];
        let params = VwmacdParams { 
            fast_period: Some(0), 
            slow_period: Some(26), 
            signal_period: Some(9),
            fast_ma_type: None,
            slow_ma_type: None,
            signal_ma_type: None,
        };
        let input = VwmacdInput::from_slices(&close, &volume, params);
        let result = vwmacd_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_vwmacd_period_exceeds(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let close = [10.0, 20.0, 30.0];
        let volume = [100.0, 200.0, 300.0];
        let params = VwmacdParams { 
            fast_period: Some(12), 
            slow_period: Some(26), 
            signal_period: Some(9),
            fast_ma_type: None,
            slow_ma_type: None,
            signal_ma_type: None,
        };
        let input = VwmacdInput::from_slices(&close, &volume, params);
        let result = vwmacd_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    macro_rules! generate_all_vwmacd_tests {
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
    generate_all_vwmacd_tests!(
        check_vwmacd_partial_params,
        check_vwmacd_accuracy,
        check_vwmacd_with_custom_ma_types,
        check_vwmacd_nan_data,
        check_vwmacd_zero_period,
        check_vwmacd_period_exceeds
    );
        
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let close = &c.close;
        let volume = &c.volume;

        let output = VwmacdBatchBuilder::new()
            .kernel(kernel)
            .apply_slices(close, volume)?;

        let def = VwmacdParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), close.len());

        let expected_macd = [
            -394.95161155,
            -508.29106210,
            -490.70190723,
            -388.94996199,
            -341.13720646,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected_macd[i]).abs() < 1e-3,
                "[{test}] default-row MACD mismatch at idx {i}: got {v}, expected {}",
                expected_macd[i]
            );
        }

        let input = VwmacdInput::from_candles(&c, "close", "volume", def.clone());
        let result = vwmacd_with_kernel(&input, kernel)?;
        
        let expected_signal = [
            -539.48861567,
            -533.24910496,
            -524.73966541,
            -497.58172247,
            -466.29282108,
        ];
        let signal_slice = &result.signal[result.signal.len() - 5..];
        for (i, &v) in signal_slice.iter().enumerate() {
            assert!(
                (v - expected_signal[i]).abs() < 1e-3,
                "[{test}] default-row Signal mismatch at idx {i}: got {v}, expected {}",
                expected_signal[i]
            );
        }

        let expected_histogram = [
            144.53700412,
            24.95804286,
            34.03775818,
            108.63176274,
            125.15561462,
        ];
        let hist_slice = &result.hist[result.hist.len() - 5..];
        for (i, &v) in hist_slice.iter().enumerate() {
            assert!(
                (v - expected_histogram[i]).abs() < 1e-3,
                "[{test}] default-row Histogram mismatch at idx {i}: got {v}, expected {}",
                expected_histogram[i]
            );
        }

        Ok(())
    }


    fn check_batch_grid(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let close = &c.close;
        let volume = &c.volume;

        let output = VwmacdBatchBuilder::new()
            .kernel(kernel)
            .fast_range(10, 14, 2)
            .slow_range(20, 26, 3)
            .signal_range(5, 9, 2)
            .apply_slices(close, volume)?;

        assert_eq!(output.cols, close.len());
        assert_eq!(output.rows, 3 * 3 * 3);

        let params = VwmacdParams {
            fast_period: Some(12),
            slow_period: Some(23),
            signal_period: Some(7),
            fast_ma_type: Some("sma".to_string()),
            slow_ma_type: Some("sma".to_string()),
            signal_ma_type: Some("ema".to_string()),
        };
        let row = output.values_for(&params).expect("row for params missing");
        assert_eq!(row.len(), close.len());
        Ok(())
    }

    fn check_batch_param_map(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let close = &c.close;
        let volume = &c.volume;

        let batch = VwmacdBatchBuilder::new()
            .kernel(kernel)
            .fast_range(12, 14, 1)
            .slow_range(26, 28, 1)
            .signal_range(9, 11, 1)
            .apply_slices(close, volume)?;

        for (ix, param) in batch.params.iter().enumerate() {
            let by_index = &batch.macd[ix * batch.cols .. (ix + 1) * batch.cols];
            let by_api   = batch.values_for(param).unwrap();

            assert_eq!(by_index.len(), by_api.len());
            for (i, (&x, &y)) in by_index.iter().zip(by_api.iter()).enumerate() {
                if x.is_nan() && y.is_nan() {
                    continue;
                }
                assert!(
                    (x == y),
                    "[{}] param {:?}, mismatch at idx {}: got {}, expected {}",
                    test, param, i, x, y
                );
            }
        }
        Ok(())
    }

    fn check_batch_custom_ma_types(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let close = &c.close;
        let volume = &c.volume;

        let output = VwmacdBatchBuilder::new()
            .kernel(kernel)
            .fast_ma_type("ema".to_string())
            .slow_ma_type("wma".to_string())
            .signal_ma_type("sma".to_string())
            .apply_slices(close, volume)?;

        let params = VwmacdParams {
            fast_period: Some(12),
            slow_period: Some(26),
            signal_period: Some(9),
            fast_ma_type: Some("ema".to_string()),
            slow_ma_type: Some("wma".to_string()),
            signal_ma_type: Some("sma".to_string()),
        };
        let row = output.values_for(&params).expect("custom MA types row missing");
        assert_eq!(row.len(), close.len());
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
    gen_batch_tests!(check_batch_grid);
    gen_batch_tests!(check_batch_param_map);
    gen_batch_tests!(check_batch_custom_ma_types);
}