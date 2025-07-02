//! # WaveTrend Indicator
//!
//! A technical oscillator indicator (WT1, WT2, WT_DIFF = WT2 - WT1) using a combination
//! of EMA, absolute deviations, and SMA. Parameters are: `channel_length`, `average_length`,
//! `ma_length`, and `factor`. Full AVX2/AVX512 stub-parity with alma.rs for performance extension
//! and batch/grid API. Scalar logic untouched for accuracy.
//!
//! ## Parameters
//! - **channel_length**: EMA period (default: 9)
//! - **average_length**: EMA for transformed CI (default: 12)
//! - **ma_length**: SMA period on WT1 (default: 3)
//! - **factor**: Multiplier for transformation (default: 0.015)
//!
//! ## Errors
//! - **EmptyData**, **AllValuesNaN**
//! - **InvalidChannelLen**, **InvalidAverageLen**, **InvalidMaLen**
//! - **NotEnoughValidData**
//!
//! ## Output
//! - `WavetrendOutput` (fields: wt1, wt2, wt_diff)
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
use thiserror::Error;
use crate::indicators::moving_averages::ema::{ema, EmaError, EmaInput, EmaParams};
use crate::indicators::moving_averages::sma::{sma, SmaError, SmaInput, SmaParams};

impl<'a> AsRef<[f64]> for WavetrendInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            WavetrendData::Slice(slice) => slice,
            WavetrendData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum WavetrendData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct WavetrendOutput {
    pub wt1: Vec<f64>,
    pub wt2: Vec<f64>,
    pub wt_diff: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WavetrendParams {
    pub channel_length: Option<usize>,
    pub average_length: Option<usize>,
    pub ma_length: Option<usize>,
    pub factor: Option<f64>,
}

impl Default for WavetrendParams {
    fn default() -> Self {
        Self {
            channel_length: Some(9),
            average_length: Some(12),
            ma_length: Some(3),
            factor: Some(0.015),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WavetrendInput<'a> {
    pub data: WavetrendData<'a>,
    pub params: WavetrendParams,
}

impl<'a> WavetrendInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: WavetrendParams) -> Self {
        Self {
            data: WavetrendData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: WavetrendParams) -> Self {
        Self { data: WavetrendData::Slice(sl), params: p }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "hlc3", WavetrendParams::default())
    }
    #[inline]
    pub fn get_channel_length(&self) -> usize {
        self.params.channel_length.unwrap_or(9)
    }
    #[inline]
    pub fn get_average_length(&self) -> usize {
        self.params.average_length.unwrap_or(12)
    }
    #[inline]
    pub fn get_ma_length(&self) -> usize {
        self.params.ma_length.unwrap_or(3)
    }
    #[inline]
    pub fn get_factor(&self) -> f64 {
        self.params.factor.unwrap_or(0.015)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct WavetrendBuilder {
    channel_length: Option<usize>,
    average_length: Option<usize>,
    ma_length: Option<usize>,
    factor: Option<f64>,
    kernel: Kernel,
}

impl Default for WavetrendBuilder {
    fn default() -> Self {
        Self {
            channel_length: None,
            average_length: None,
            ma_length: None,
            factor: None,
            kernel: Kernel::Auto,
        }
    }
}

impl WavetrendBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn channel_length(mut self, n: usize) -> Self {
        self.channel_length = Some(n); self
    }
    #[inline(always)]
    pub fn average_length(mut self, n: usize) -> Self {
        self.average_length = Some(n); self
    }
    #[inline(always)]
    pub fn ma_length(mut self, n: usize) -> Self {
        self.ma_length = Some(n); self
    }
    #[inline(always)]
    pub fn factor(mut self, f: f64) -> Self {
        self.factor = Some(f); self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k; self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<WavetrendOutput, WavetrendError> {
        let p = WavetrendParams {
            channel_length: self.channel_length,
            average_length: self.average_length,
            ma_length: self.ma_length,
            factor: self.factor,
        };
        let i = WavetrendInput::from_candles(c, "hlc3", p);
        wavetrend_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<WavetrendOutput, WavetrendError> {
        let p = WavetrendParams {
            channel_length: self.channel_length,
            average_length: self.average_length,
            ma_length: self.ma_length,
            factor: self.factor,
        };
        let i = WavetrendInput::from_slice(d, p);
        wavetrend_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<WavetrendStream, WavetrendError> {
        let p = WavetrendParams {
            channel_length: self.channel_length,
            average_length: self.average_length,
            ma_length: self.ma_length,
            factor: self.factor,
        };
        WavetrendStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum WavetrendError {
    #[error("wavetrend: Empty data provided.")]
    EmptyData,
    #[error("wavetrend: All values are NaN.")]
    AllValuesNaN,
    #[error("wavetrend: Invalid channel_length = {channel_length}, data length = {data_len}")]
    InvalidChannelLen { channel_length: usize, data_len: usize },
    #[error("wavetrend: Invalid average_length = {average_length}, data length = {data_len}")]
    InvalidAverageLen { average_length: usize, data_len: usize },
    #[error("wavetrend: Invalid ma_length = {ma_length}, data length = {data_len}")]
    InvalidMaLen { ma_length: usize, data_len: usize },
    #[error("wavetrend: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("wavetrend: EMA error {0}")]
    EmaError(#[from] EmaError),
    #[error("wavetrend: SMA error {0}")]
    SmaError(#[from] SmaError),
}

#[inline]
pub fn wavetrend(input: &WavetrendInput) -> Result<WavetrendOutput, WavetrendError> {
    wavetrend_with_kernel(input, Kernel::Auto)
}

pub fn wavetrend_with_kernel(input: &WavetrendInput, kernel: Kernel) -> Result<WavetrendOutput, WavetrendError> {
    let data: &[f64] = input.as_ref();
    if data.is_empty() {
        return Err(WavetrendError::EmptyData);
    }
    let channel_len = input.get_channel_length();
    let average_len = input.get_average_length();
    let ma_len = input.get_ma_length();
    let factor = input.get_factor();

    let first = data.iter().position(|x| !x.is_nan()).ok_or(WavetrendError::AllValuesNaN)?;
    let needed = *[channel_len, average_len, ma_len].iter().max().unwrap();
    let valid = data.len() - first;

    if channel_len == 0 || channel_len > data.len() {
        return Err(WavetrendError::InvalidChannelLen { channel_length: channel_len, data_len: data.len() });
    }
    if average_len == 0 || average_len > data.len() {
        return Err(WavetrendError::InvalidAverageLen { average_length: average_len, data_len: data.len() });
    }
    if ma_len == 0 || ma_len > data.len() {
        return Err(WavetrendError::InvalidMaLen { ma_length: ma_len, data_len: data.len() });
    }
    if valid < needed {
        return Err(WavetrendError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => wavetrend_scalar(data, channel_len, average_len, ma_len, factor, first),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => wavetrend_avx2(data, channel_len, average_len, ma_len, factor, first),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => wavetrend_avx512(data, channel_len, average_len, ma_len, factor, first),
            _ => unreachable!(),
        }
    }
}

pub fn wavetrend_scalar(
    data: &[f64], channel_len: usize, average_len: usize, ma_len: usize, factor: f64, first: usize
) -> Result<WavetrendOutput, WavetrendError> {
    let data_valid = &data[first..];

    let esa_input = EmaInput::from_slice(
        data_valid,
        EmaParams { period: Some(channel_len) },
    );
    let esa_output = ema(&esa_input)?;
    let esa_values = &esa_output.values;

    let mut diff_esa = vec![f64::NAN; data_valid.len()];
    for i in 0..data_valid.len() {
        if !data_valid[i].is_nan() && !esa_values[i].is_nan() {
            diff_esa[i] = (data_valid[i] - esa_values[i]).abs();
        }
    }

    let de_input = EmaInput::from_slice(
        &diff_esa,
        EmaParams { period: Some(channel_len) },
    );
    let de_output = ema(&de_input)?;
    let de_values = &de_output.values;

    let mut ci = vec![f64::NAN; data_valid.len()];
    for i in 0..data_valid.len() {
        if !data_valid[i].is_nan() && !esa_values[i].is_nan() && !de_values[i].is_nan() {
            let den = factor * de_values[i];
            if den != 0.0 {
                ci[i] = (data_valid[i] - esa_values[i]) / den;
            }
        }
    }

    let wt1_input = EmaInput::from_slice(
        &ci,
        EmaParams { period: Some(average_len) },
    );
    let wt1_output = ema(&wt1_input)?;
    let wt1_values = &wt1_output.values;

    let wt2_input = SmaInput::from_slice(
        wt1_values,
        SmaParams { period: Some(ma_len) },
    );
    let wt2_output = sma(&wt2_input)?;
    let wt2_values = &wt2_output.values;

    let mut wt1_final = vec![f64::NAN; data.len()];
    let mut wt2_final = vec![f64::NAN; data.len()];
    let mut diff_final = vec![f64::NAN; data.len()];

    for i in 0..data_valid.len() {
        wt1_final[i + first] = wt1_values[i];
        wt2_final[i + first] = wt2_values[i];
        if !wt1_values[i].is_nan() && !wt2_values[i].is_nan() {
            diff_final[i + first] = wt2_values[i] - wt1_values[i];
        }
    }

    Ok(WavetrendOutput {
        wt1: wt1_final,
        wt2: wt2_final,
        wt_diff: diff_final,
    })
}

use std::collections::VecDeque;
// AVX2 stub - points to scalar
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wavetrend_avx2(
    data: &[f64], channel_len: usize, average_len: usize, ma_len: usize, factor: f64, first: usize
) -> Result<WavetrendOutput, WavetrendError> {
    wavetrend_scalar(data, channel_len, average_len, ma_len, factor, first)
}

// AVX512 stub logic for short/long
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wavetrend_avx512(
    data: &[f64], channel_len: usize, average_len: usize, ma_len: usize, factor: f64, first: usize
) -> Result<WavetrendOutput, WavetrendError> {
    if channel_len <= 32 {
        wavetrend_avx512_short(data, channel_len, average_len, ma_len, factor, first)
    
        } else {
        wavetrend_avx512_long(data, channel_len, average_len, ma_len, factor, first)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wavetrend_avx512_short(
    data: &[f64], channel_len: usize, average_len: usize, ma_len: usize, factor: f64, first: usize
) -> Result<WavetrendOutput, WavetrendError> {
    wavetrend_scalar(data, channel_len, average_len, ma_len, factor, first)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn wavetrend_avx512_long(
    data: &[f64], channel_len: usize, average_len: usize, ma_len: usize, factor: f64, first: usize
) -> Result<WavetrendOutput, WavetrendError> {
    wavetrend_scalar(data, channel_len, average_len, ma_len, factor, first)
}



#[derive(Clone, Debug)]
pub struct WavetrendStream {
    // ─────────── user-provided parameters (never change after construction) ───────────
    pub channel_length: usize,
    pub average_length: usize,
    pub ma_length:      usize,
    pub factor:         f64,

    // ─────────────────────────────────────────────────────────────────────────────────
    // Stage 1: “ESA” = EMA(channel_length) on price
    //   We seed immediately: last_esa = first finite price. α_ch = 2/(channel_length+1).
    esa_buf:    VecDeque<f64>,   // not used for seeding, but we keep it for capacity hint
    last_esa:   Option<f64>,
    alpha_ch:   f64,

    // ─────────────────────────────────────────────────────────────────────────────────
    // Stage 2: “DE” = EMA(channel_length) on |price − ESA|
    //   Seed immediately: last_de = first |price₀ − esa₀|. Then recurse with α_ch again.
    de_buf:     VecDeque<f64>,   // not used for seeding, but kept for symmetry
    last_de:    Option<f64>,

    // ─────────────────────────────────────────────────────────────────────────────────
    // Stage 3: “WT1” = EMA(average_length) on CI = (price − ESA)/(factor·DE)
    //   Seed immediately as soon as we get the very first valid CI. α_avg = 2/(average_length+1).
    ci_buf:     VecDeque<f64>,   // not used for seeding, but kept for capacity hint
    last_wt1:   Option<f64>,
    alpha_avg:  f64,

    // ─────────────────────────────────────────────────────────────────────────────────
    // Stage 4: “WT2” = SMA(ma_length) on the most recent WT1 values
    //   We keep a sliding window of length ma_length in wt1_buf and a running_sum.
    wt1_buf:    VecDeque<f64>,
    running_sum: f64,

    // ─────────────────────────────────────────────────────────────────────────────────
    // history: so that streaming index = batch index. Every time update(...) is called,
    // we push the raw `price` so that the test harness can compare indexes directly.
    pub history: Vec<f64>,
}

impl WavetrendStream {
    pub fn try_new(p: WavetrendParams) -> Result<Self, WavetrendError> {
        let channel_length = p.channel_length.unwrap_or(9);
        let average_length = p.average_length.unwrap_or(12);
        let ma_length      = p.ma_length.unwrap_or(3);
        let factor         = p.factor.unwrap_or(0.015);

        // Exactly the same error categories as the batch version if any period is zero.
        if channel_length == 0 || average_length == 0 || ma_length == 0 {
            return Err(WavetrendError::InvalidChannelLen { channel_length, data_len: 0 });
        }

        // Precompute smoothing constants:
        //   α_ch  = 2 / (channel_length + 1)
        //   α_avg = 2 / (average_length + 1)
        let alpha_ch  = 2.0 / (channel_length as f64 + 1.0);
        let alpha_avg = 2.0 / (average_length as f64 + 1.0);

        Ok(Self {
            channel_length,
            average_length,
            ma_length,
            factor,

            esa_buf:    VecDeque::with_capacity(channel_length),
            last_esa:   None,
            alpha_ch,

            de_buf:     VecDeque::with_capacity(channel_length),
            last_de:    None,

            ci_buf:     VecDeque::with_capacity(average_length),
            last_wt1:   None,
            alpha_avg,

            wt1_buf:    VecDeque::with_capacity(ma_length),
            running_sum: 0.0,

            history:    Vec::new(),
        })
    }

    /// Push one new `price`.  Returns `None` (→ “NaN” in the test harness) until
    /// all four stages (ESA, DE, WT1, WT2) have produced a finite number.  Once
    /// all four are valid, returns `Some((wt1, wt2, wt2 − wt1))`.
    #[inline(always)]
    pub fn update(&mut self, price: f64) -> Option<(f64, f64, f64)> {
        // 1) Record raw price in history so streaming index = batch index.
        self.history.push(price);

        // 2) If price is not a finite f64, the scalar EMA would have produced NaN,
        //    so we return None here (but history was recorded).
        if !price.is_finite() {
            return None;
        }

        // ─── Stage 1: ESA = EMA(channel_length) on price ────────────────────────────
        let esa = if let Some(prev_esa) = self.last_esa {
            // Already seeded: do EMA recurrence:
            let new_esa = self.alpha_ch * price + (1.0 - self.alpha_ch) * prev_esa;
            self.last_esa = Some(new_esa);
            new_esa
        
            } else {
            // First-ever finite price: seed ESA = price₀ (exactly what ema_scalar does).
            self.last_esa = Some(price);
            price
        };

        // ─── Stage 2: DE = EMA(channel_length) on |price − ESA| ─────────────────────
        let abs_diff = (price - esa).abs();
        let de = if let Some(prev_de) = self.last_de {
            // Already seeded: do EMA recurrence on abs_diff:
            let new_de = self.alpha_ch * abs_diff + (1.0 - self.alpha_ch) * prev_de;
            self.last_de = Some(new_de);
            new_de
        
            } else {
            // First-ever |price₀ − esa₀|: seed DE = abs_diff₀ (likely 0 at i=0).
            self.last_de = Some(abs_diff);
            abs_diff
        };

        // If DE == 0.0, then scalar would have produced CI = NaN here → return None.
        if de == 0.0 {
            return None;
        }

        // ─── Stage 3: WT1 = EMA(average_length) on CI = (price − ESA)/(factor·DE) ────
        let ci = (price - esa) / (self.factor * de);

        let wt1 = if let Some(prev_wt1) = self.last_wt1 {
            // Already seeded: do EMA recurrence on CI
            let new_wt1 = self.alpha_avg * ci + (1.0 - self.alpha_avg) * prev_wt1;
            self.last_wt1 = Some(new_wt1);
            new_wt1
        
            } else {
            // First-ever valid CI: seed WT1 = ci₁ (just like ema_scalar seeds)
            self.last_wt1 = Some(ci);
            ci
        };

        // ─── Stage 4: WT2 = SMA(ma_length) on the most recent ma_length WT1s ─────────
        // Push new WT1 into the fifo buffer and maintain running_sum:
        self.wt1_buf.push_back(wt1);
        self.running_sum += wt1;

        // If we exceed ma_length, pop the oldest and subtract from running_sum:
        if self.wt1_buf.len() > self.ma_length {
            let oldest = self.wt1_buf.pop_front().unwrap();
            self.running_sum -= oldest;
        }

        // Only once we have at least ma_length many WT1s do we form a valid WT2:
        if self.wt1_buf.len() < self.ma_length {
            return None;
        }

        let wt2 = self.running_sum / (self.ma_length as f64);
        let diff = wt2 - wt1;
        Some((wt1, wt2, diff))
    }
}





#[derive(Clone, Debug)]
pub struct WavetrendBatchRange {
    pub channel_length: (usize, usize, usize),
    pub average_length: (usize, usize, usize),
    pub ma_length: (usize, usize, usize),
    pub factor: (f64, f64, f64),
}

impl Default for WavetrendBatchRange {
    fn default() -> Self {
        Self {
            channel_length: (9, 9, 1),
            average_length: (12, 12, 1),
            ma_length: (3, 3, 1),
            factor: (0.015, 0.015, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WavetrendBatchBuilder {
    range: WavetrendBatchRange,
    kernel: Kernel,
}

impl WavetrendBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn channel_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.channel_length = (start, end, step); self
    }
    pub fn channel_static(mut self, x: usize) -> Self {
        self.range.channel_length = (x, x, 0); self
    }
    pub fn avg_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.average_length = (start, end, step); self
    }
    pub fn avg_static(mut self, x: usize) -> Self {
        self.range.average_length = (x, x, 0); self
    }
    pub fn ma_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.ma_length = (start, end, step); self
    }
    pub fn ma_static(mut self, x: usize) -> Self {
        self.range.ma_length = (x, x, 0); self
    }
    pub fn factor_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.factor = (start, end, step); self
    }
    pub fn factor_static(mut self, x: f64) -> Self {
        self.range.factor = (x, x, 0.0); self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<WavetrendBatchOutput, WavetrendError> {
        wavetrend_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<WavetrendBatchOutput, WavetrendError> {
        WavetrendBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<WavetrendBatchOutput, WavetrendError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<WavetrendBatchOutput, WavetrendError> {
        WavetrendBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "hlc3")
    }
}

pub fn wavetrend_batch_with_kernel(
    data: &[f64], sweep: &WavetrendBatchRange, k: Kernel,
) -> Result<WavetrendBatchOutput, WavetrendError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(WavetrendError::InvalidChannelLen { channel_length: 0, data_len: 0 })
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    wavetrend_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct WavetrendBatchOutput {
    pub wt1: Vec<f64>,
    pub wt2: Vec<f64>,
    pub wt_diff: Vec<f64>,
    pub combos: Vec<WavetrendParams>,
    pub rows: usize,
    pub cols: usize,
}
impl WavetrendBatchOutput {
    pub fn row_for_params(&self, p: &WavetrendParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.channel_length.unwrap_or(9) == p.channel_length.unwrap_or(9) &&
            c.average_length.unwrap_or(12) == p.average_length.unwrap_or(12) &&
            c.ma_length.unwrap_or(3) == p.ma_length.unwrap_or(3) &&
            (c.factor.unwrap_or(0.015) - p.factor.unwrap_or(0.015)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &WavetrendParams) -> Option<(&[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (&self.wt1[start..start + self.cols], &self.wt2[start..start + self.cols], &self.wt_diff[start..start + self.cols])
        })
    }
}

#[inline(always)]
fn expand_grid(r: &WavetrendBatchRange) -> Vec<WavetrendParams> {
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
    let chs = axis_usize(r.channel_length);
    let avgs = axis_usize(r.average_length);
    let mas = axis_usize(r.ma_length);
    let factors = axis_f64(r.factor);
    let mut out = Vec::with_capacity(chs.len() * avgs.len() * mas.len() * factors.len());
    for &c in &chs {
        for &a in &avgs {
            for &m in &mas {
                for &f in &factors {
                    out.push(WavetrendParams {
                        channel_length: Some(c),
                        average_length: Some(a),
                        ma_length: Some(m),
                        factor: Some(f),
                    });
                }
            }
        }
    }
    out
}

#[inline(always)]
pub fn wavetrend_batch_slice(
    data: &[f64], sweep: &WavetrendBatchRange, kern: Kernel
) -> Result<WavetrendBatchOutput, WavetrendError> {
    wavetrend_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn wavetrend_batch_par_slice(
    data: &[f64], sweep: &WavetrendBatchRange, kern: Kernel
) -> Result<WavetrendBatchOutput, WavetrendError> {
    wavetrend_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn wavetrend_batch_inner(
    data: &[f64], sweep: &WavetrendBatchRange, kern: Kernel, parallel: bool
) -> Result<WavetrendBatchOutput, WavetrendError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(WavetrendError::InvalidChannelLen { channel_length: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(WavetrendError::AllValuesNaN)?;
    let max_ch = combos.iter().map(|c| c.channel_length.unwrap()).max().unwrap();
    let max_avg = combos.iter().map(|c| c.average_length.unwrap()).max().unwrap();
    let max_ma = combos.iter().map(|c| c.ma_length.unwrap()).max().unwrap();
    let max_p = *[max_ch, max_avg, max_ma].iter().max().unwrap();
    if data.len() - first < max_p {
        return Err(WavetrendError::NotEnoughValidData { needed: max_p, valid: data.len() - first });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut wt1 = vec![f64::NAN; rows * cols];
    let mut wt2 = vec![f64::NAN; rows * cols];
    let mut wt_diff = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, w1: &mut [f64], w2: &mut [f64], wd: &mut [f64]| unsafe {
        let p = &combos[row];
        let r = wavetrend_row_scalar(
            data, first,
            p.channel_length.unwrap(),
            p.average_length.unwrap(),
            p.ma_length.unwrap(),
            p.factor.unwrap_or(0.015),
            w1, w2, wd
        );
        if let Err(e) = r { panic!("wavetrend row error: {:?}", e); }
    };

    if parallel {


        #[cfg(not(target_arch = "wasm32"))] {


        wt1.par_chunks_mut(cols).zip(wt2.par_chunks_mut(cols)).zip(wt_diff.par_chunks_mut(cols))


                    .enumerate()


                    .for_each(|(row, ((w1, w2), wd))| do_row(row, w1, w2, wd));


        }


        #[cfg(target_arch = "wasm32")] {


        for (row, (((w1, w2), wd))) in wt1.chunks_mut(cols).zip(wt2.chunks_mut(cols)).zip(wt_diff.chunks_mut(cols)).enumerate() {


                    do_row(row, w1, w2, wd);


        }


    } else {
        for (row, (((w1, w2), wd))) in wt1.chunks_mut(cols).zip(wt2.chunks_mut(cols)).zip(wt_diff.chunks_mut(cols)).enumerate() {
            do_row(row, w1, w2, wd);
        }
    }
    Ok(WavetrendBatchOutput {
        wt1, wt2, wt_diff, combos, rows, cols
    })
}

#[inline(always)]
unsafe fn wavetrend_row_scalar(
    data: &[f64], first: usize, channel_len: usize, average_len: usize, ma_len: usize, factor: f64,
    wt1: &mut [f64], wt2: &mut [f64], wd: &mut [f64]
) -> Result<(), WavetrendError> {
    let out = wavetrend_scalar(data, channel_len, average_len, ma_len, factor, first)?;
    wt1.copy_from_slice(&out.wt1);
    wt2.copy_from_slice(&out.wt2);
    wd.copy_from_slice(&out.wt_diff);
    Ok(())
}

// AVX2/AVX512 batch row stubs - always point to scalar
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wavetrend_row_avx2(
    data: &[f64], first: usize, channel_len: usize, average_len: usize, ma_len: usize, factor: f64,
    wt1: &mut [f64], wt2: &mut [f64], wd: &mut [f64]
) -> Result<(), WavetrendError> {
    wavetrend_row_scalar(data, first, channel_len, average_len, ma_len, factor, wt1, wt2, wd)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wavetrend_row_avx512(
    data: &[f64], first: usize, channel_len: usize, average_len: usize, ma_len: usize, factor: f64,
    wt1: &mut [f64], wt2: &mut [f64], wd: &mut [f64]
) -> Result<(), WavetrendError> {
    wavetrend_row_scalar(data, first, channel_len, average_len, ma_len, factor, wt1, wt2, wd)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wavetrend_row_avx512_short(
    data: &[f64], first: usize, channel_len: usize, average_len: usize, ma_len: usize, factor: f64,
    wt1: &mut [f64], wt2: &mut [f64], wd: &mut [f64]
) -> Result<(), WavetrendError> {
    wavetrend_row_scalar(data, first, channel_len, average_len, ma_len, factor, wt1, wt2, wd)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn wavetrend_row_avx512_long(
    data: &[f64], first: usize, channel_len: usize, average_len: usize, ma_len: usize, factor: f64,
    wt1: &mut [f64], wt2: &mut [f64], wd: &mut [f64]
) -> Result<(), WavetrendError> {
    wavetrend_row_scalar(data, first, channel_len, average_len, ma_len, factor, wt1, wt2, wd)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use crate::utilities::enums::Kernel;

    fn check_wavetrend_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = WavetrendParams { channel_length: None, average_length: None, ma_length: None, factor: None };
        let input = WavetrendInput::from_candles(&candles, "hlc3", default_params);
        let output = wavetrend_with_kernel(&input, kernel)?;
        assert_eq!(output.wt1.len(), candles.close.len());
        Ok(())
    }

    fn check_wavetrend_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = WavetrendInput::from_candles(&candles, "hlc3", WavetrendParams::default());
        let result = wavetrend_with_kernel(&input, kernel)?;
        let len = result.wt1.len();
        let expected_wt1 = [
            -29.02058232514538,
            -28.207769813591664,
            -31.991808642927193,
            -31.9218051759519,
            -44.956245952893866,
        ];
        let expected_wt2 = [
            -30.651043230696555,
            -28.686329669808583,
            -29.740053593887932,
            -30.707127877490105,
            -36.2899532572575,
        ];
        for (i, &val) in result.wt1[len - 5..].iter().enumerate() {
            let diff = (val - expected_wt1[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Wavetrend {:?} WT1 mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_wt1[i]
            );
        }
        for (i, &val) in result.wt2[len - 5..].iter().enumerate() {
            let diff = (val - expected_wt2[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Wavetrend {:?} WT2 mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_wt2[i]
            );
        }
        let last_five_diff = &result.wt_diff[len - 5..];
        for i in 0..5 {
            let expected = expected_wt2[i] - expected_wt1[i];
            let diff = (last_five_diff[i] - expected).abs();
            assert!(
                diff < 1e-6,
                "[{}] Wavetrend {:?} WT_DIFF mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                last_five_diff[i],
                expected
            );
        }
        Ok(())
    }

    fn check_wavetrend_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = WavetrendInput::with_default_candles(&candles);
        match input.data {
            WavetrendData::Candles { source, .. } => assert_eq!(source, "hlc3"),
            _ => panic!("Expected WavetrendData::Candles"),
        }
        let output = wavetrend_with_kernel(&input, kernel)?;
        assert_eq!(output.wt1.len(), candles.close.len());
        Ok(())
    }

    fn check_wavetrend_zero_channel(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = WavetrendParams {
            channel_length: Some(0),
            average_length: Some(12),
            ma_length: Some(3),
            factor: Some(0.015),
        };
        let input = WavetrendInput::from_slice(&input_data, params);
        let res = wavetrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Wavetrend should fail with zero channel_length",
            test_name
        );
        Ok(())
    }

    fn check_wavetrend_channel_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = WavetrendParams {
            channel_length: Some(10),
            average_length: Some(12),
            ma_length: Some(3),
            factor: Some(0.015),
        };
        let input = WavetrendInput::from_slice(&data_small, params);
        let res = wavetrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Wavetrend should fail with channel_length exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_wavetrend_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = WavetrendParams::default();
        let input = WavetrendInput::from_slice(&single_point, params);
        let res = wavetrend_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Wavetrend should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_wavetrend_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = WavetrendInput::from_candles(
            &candles,
            "hlc3",
            WavetrendParams {
                channel_length: Some(9),
                average_length: Some(12),
                ma_length: Some(3),
                factor: Some(0.015),
            },
        );
        let res = wavetrend_with_kernel(&input, kernel)?;
        assert_eq!(res.wt1.len(), candles.close.len());
        if res.wt1.len() > 240 {
            for (i, &val) in res.wt1[240..].iter().enumerate() {
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

    fn check_wavetrend_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let channel_length = 9;
        let average_length = 12;
        let ma_length = 3;
        let factor = 0.015;

        let input = WavetrendInput::from_candles(
            &candles,
            "hlc3",
            WavetrendParams {
                channel_length: Some(channel_length),
                average_length: Some(average_length),
                ma_length: Some(ma_length),
                factor: Some(factor),
            },
        );
        let full_output = wavetrend_with_kernel(&input, kernel)?;

        let mut stream = WavetrendStream::try_new(WavetrendParams {
            channel_length: Some(channel_length),
            average_length: Some(average_length),
            ma_length: Some(ma_length),
            factor: Some(factor),
        })?;

        let mut wt1_stream = Vec::with_capacity(candles.hlc3.len());
        let mut wt2_stream = Vec::with_capacity(candles.hlc3.len());
        let mut diff_stream = Vec::with_capacity(candles.hlc3.len());
        for &price in &candles.hlc3 {
            match stream.update(price) {
                Some((wt1, wt2, diff)) => {
                    wt1_stream.push(wt1);
                    wt2_stream.push(wt2);
                    diff_stream.push(diff);
                }
                None => {
                    wt1_stream.push(f64::NAN);
                    wt2_stream.push(f64::NAN);
                    diff_stream.push(f64::NAN);
                }
            }
        }

        let mut first_non_nan = None;
        for (i, &b) in full_output.wt1.iter().enumerate() {
            if !b.is_nan() {
                first_non_nan = Some(i);
                break;
            }
        }
        let start = first_non_nan.unwrap_or(0);
        assert_eq!(full_output.wt1.len(), wt1_stream.len());
        for (i, (&b, &s)) in full_output.wt1.iter().zip(wt1_stream.iter()).enumerate().skip(start) {
            if b.is_nan() || s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Wavetrend streaming wt1 f64 mismatch at idx {}: full={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        for (i, (&b, &s)) in full_output.wt2.iter().zip(wt2_stream.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Wavetrend streaming wt2 f64 mismatch at idx {}: full={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        for (i, (&b, &s)) in full_output.wt_diff.iter().zip(diff_stream.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] Wavetrend streaming wt_diff f64 mismatch at idx {}: full={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_wavetrend_tests {
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

    generate_all_wavetrend_tests!(
        check_wavetrend_partial_params,
        check_wavetrend_accuracy,
        check_wavetrend_default_candles,
        check_wavetrend_zero_channel,
        check_wavetrend_channel_exceeds_length,
        check_wavetrend_very_small_dataset,
        check_wavetrend_nan_handling,
        check_wavetrend_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = WavetrendBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "hlc3")?;

        let def = WavetrendParams::default();
        let (wt1_row, wt2_row, diff_row) = output.values_for(&def).expect("default row missing");

        assert_eq!(wt1_row.len(), c.close.len());
        assert_eq!(wt2_row.len(), c.close.len());
        assert_eq!(diff_row.len(), c.close.len());

        let expected_wt1 = [
            -29.02058232514538,
            -28.207769813591664,
            -31.991808642927193,
            -31.9218051759519,
            -44.956245952893866,
        ];
        let expected_wt2 = [
            -30.651043230696555,
            -28.686329669808583,
            -29.740053593887932,
            -30.707127877490105,
            -36.2899532572575,
        ];

        let start = wt1_row.len().saturating_sub(5);
        for (i, &v) in wt1_row[start..].iter().enumerate() {
            assert!(
                (v - expected_wt1[i]).abs() < 1e-8,
                "[{test}] default-row WT1 mismatch at idx {i}: {v} vs {expected}",
                test = test,
                i = i,
                v = v,
                expected = expected_wt1[i]
            );
        }
        for (i, &v) in wt2_row[start..].iter().enumerate() {
            assert!(
                (v - expected_wt2[i]).abs() < 1e-6,
                "[{test}] default-row WT2 mismatch at idx {i}: {v} vs {expected}",
                test = test,
                i = i,
                v = v,
                expected = expected_wt2[i]
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
