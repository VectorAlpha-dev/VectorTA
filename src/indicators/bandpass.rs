//! # Band-Pass Filter
//!
//! A frequency-based filter (Ehlers-inspired) that isolates a band of interest by removing both high- and low-frequency components from a time series. Parameters `period` and `bandwidth` control the central window and width. Batch, stream, and SIMD kernels are supported.
//!
//! ## Parameters
//! - **period**: Central lookback period (>=2).
//! - **bandwidth**: Passband width in [0,1] (default: 0.3).
//!
//! ## Errors
//! - **NotEnoughData**: Data length < period.
//! - **InvalidPeriod**: period < 2.
//! - **HpPeriodTooSmall**: hp_period after rounding < 2.
//! - **TriggerPeriodTooSmall**: trigger_period after rounding < 2.
//! - **HighPassError**: errors from underlying highpass filter.
//!
//! ## Returns
//! - **`Ok(BandPassOutput)`** on success.  
//! - **`Err(BandPassError)`** otherwise.
//!
use crate::indicators::highpass::{highpass, HighPassError, HighPassInput, HighPassParams};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;
use std::f64::consts::PI;
use std::convert::AsRef;

impl<'a> AsRef<[f64]> for BandPassInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            BandPassData::Slice(slice) => slice,
            BandPassData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum BandPassData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct BandPassParams {
    pub period: Option<usize>,
    pub bandwidth: Option<f64>,
}

impl Default for BandPassParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            bandwidth: Some(0.3),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BandPassInput<'a> {
    pub data: BandPassData<'a>,
    pub params: BandPassParams,
}

impl<'a> BandPassInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: BandPassParams) -> Self {
        Self {
            data: BandPassData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: BandPassParams) -> Self {
        Self {
            data: BandPassData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", BandPassParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(20)
    }
    #[inline]
    pub fn get_bandwidth(&self) -> f64 {
        self.params.bandwidth.unwrap_or(0.3)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BandPassBuilder {
    period: Option<usize>,
    bandwidth: Option<f64>,
    kernel: Kernel,
}

impl Default for BandPassBuilder {
    fn default() -> Self {
        Self {
            period: None,
            bandwidth: None,
            kernel: Kernel::Auto,
        }
    }
}

impl BandPassBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline(always)]
    pub fn bandwidth(mut self, b: f64) -> Self {
        self.bandwidth = Some(b);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<BandPassOutput, BandPassError> {
        let p = BandPassParams {
            period: self.period,
            bandwidth: self.bandwidth,
        };
        let i = BandPassInput::from_candles(c, "close", p);
        bandpass_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<BandPassOutput, BandPassError> {
        let p = BandPassParams {
            period: self.period,
            bandwidth: self.bandwidth,
        };
        let i = BandPassInput::from_slice(d, p);
        bandpass_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<BandPassStream, BandPassError> {
        let p = BandPassParams {
            period: self.period,
            bandwidth: self.bandwidth,
        };
        BandPassStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum BandPassError {
    #[error("bandpass: Not enough data, data_len={data_len}, period={period}")]
    NotEnoughData { data_len: usize, period: usize },
    #[error("bandpass: Invalid period={period}")]
    InvalidPeriod { period: usize },
    #[error("bandpass: hp_period too small ({hp_period})")]
    HpPeriodTooSmall { hp_period: usize },
    #[error("bandpass: trigger_period too small ({trigger_period})")]
    TriggerPeriodTooSmall { trigger_period: usize },
    #[error(transparent)]
    HighPassError(#[from] HighPassError),
}

#[derive(Debug, Clone)]
pub struct BandPassOutput {
    pub bp: Vec<f64>,
    pub bp_normalized: Vec<f64>,
    pub signal: Vec<f64>,
    pub trigger: Vec<f64>,
}

#[inline]
pub fn bandpass(input: &BandPassInput) -> Result<BandPassOutput, BandPassError> {
    bandpass_with_kernel(input, Kernel::Auto)
}

pub fn bandpass_with_kernel(input: &BandPassInput, kernel: Kernel) -> Result<BandPassOutput, BandPassError> {
    let data: &[f64] = match &input.data {
        BandPassData::Candles { candles, source } => source_type(candles, source),
        BandPassData::Slice(sl) => sl,
    };

    let len = data.len();
    let period = input.get_period();
    let bandwidth = input.get_bandwidth();

    if len == 0 || len < period {
        return Err(BandPassError::NotEnoughData { data_len: len, period });
    }
    if period < 2 {
        return Err(BandPassError::InvalidPeriod { period });
    }
    if !(0.0..=1.0).contains(&bandwidth) || bandwidth.is_nan() || bandwidth.is_infinite() {
        return Err(BandPassError::InvalidPeriod { period });
    }

    let hp_period_f = 4.0 * (period as f64) / bandwidth;
    let hp_period = hp_period_f.round() as usize;
    if hp_period < 2 {
        return Err(BandPassError::HpPeriodTooSmall { hp_period });
    }

    let mut hp_params = HighPassParams::default();
    hp_params.period = Some(hp_period);

    let hp_input = HighPassInput::from_slice(data, hp_params);
    let hp_result = highpass(&hp_input)?;
    let hp = hp_result.values;

    let beta = (2.0 * PI / period as f64).cos();
    let gamma = (2.0 * PI * bandwidth / period as f64).cos();
    let alpha = 1.0 / gamma - ((1.0 / (gamma * gamma)) - 1.0).sqrt();

    let mut bp = hp.clone();
    let mut bp_normalized = vec![0.0; len];

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                bandpass_scalar(&hp, period, alpha, beta, &mut bp)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                bandpass_avx2(&hp, period, alpha, beta, &mut bp)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                bandpass_avx512(&hp, period, alpha, beta, &mut bp)
            }
            _ => unreachable!(),
        }
    }

    let k = 0.991;
    let mut peak_prev = 0.0;
    for i in 0..len {
        peak_prev *= k;
        let abs_bp = bp[i].abs();
        if abs_bp > peak_prev {
            peak_prev = abs_bp;
        }
        bp_normalized[i] = if peak_prev != 0.0 {
            bp[i] / peak_prev
        } else {
            0.0
        };
    }

    let trigger_period_f = (period as f64 / bandwidth) / 1.5;
    let trigger_period = trigger_period_f.round() as usize;
    if trigger_period < 2 {
        return Err(BandPassError::TriggerPeriodTooSmall { trigger_period });
    }
    let mut trigger_params = HighPassParams::default();
    trigger_params.period = Some(trigger_period);
    let trigger_input = HighPassInput::from_slice(&bp_normalized, trigger_params);
    let trigger_result = highpass(&trigger_input)?;
    let trigger = trigger_result.values;

    let mut signal = vec![0.0; len];
    for i in 0..len {
        let bn = bp_normalized[i];
        let tr = trigger[i];
        if bn < tr {
            signal[i] = 1.0;
        } else if bn > tr {
            signal[i] = -1.0;
        } else {
            signal[i] = 0.0;
        }
    }

    Ok(BandPassOutput { bp, bp_normalized, signal, trigger })
}

#[inline(always)]
pub fn bandpass_scalar(
    hp: &[f64],
    period: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    let len = hp.len();
    if len >= 2 {
        for i in 2..len {
            out[i] = 0.5 * (1.0 - alpha) * hp[i] - 0.5 * (1.0 - alpha) * hp[i - 2]
                + beta * (1.0 + alpha) * out[i - 1]
                - alpha * out[i - 2];
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_avx2(
    hp: &[f64],
    period: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_avx512(
    hp: &[f64],
    period: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_avx512_short(
    hp: &[f64],
    period: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_avx512_long(
    hp: &[f64],
    period: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    bandpass_scalar(hp, period, alpha, beta, out)
}

#[derive(Debug, Clone)]
pub struct BandPassStream {
    period: usize,
    alpha: f64,
    beta: f64,
    hp_stream: crate::indicators::highpass::HighPassStream,
    buf: Vec<f64>,
    idx: usize,
    len: usize,
    last_hp: [f64; 2],
    last_out: [f64; 2],
}

impl BandPassStream {
    pub fn try_new(params: BandPassParams) -> Result<Self, BandPassError> {
        let period = params.period.unwrap_or(20);
        if period < 2 {
            return Err(BandPassError::InvalidPeriod { period });
        }
        let bandwidth = params.bandwidth.unwrap_or(0.3);
        if !(0.0..=1.0).contains(&bandwidth) || bandwidth.is_nan() || bandwidth.is_infinite() {
            return Err(BandPassError::InvalidPeriod { period });
        }
        let hp_period = (4.0 * period as f64 / bandwidth).round() as usize;
        if hp_period < 2 {
            return Err(BandPassError::HpPeriodTooSmall { hp_period });
        }
        let mut hp_params = HighPassParams::default();
        hp_params.period = Some(hp_period);

        let hp_stream = crate::indicators::highpass::HighPassStream::try_new(hp_params)?;
        let beta = (2.0 * PI / period as f64).cos();
        let gamma = (2.0 * PI * bandwidth / period as f64).cos();
        let alpha = 1.0 / gamma - ((1.0 / (gamma * gamma)) - 1.0).sqrt();

        Ok(Self {
            period,
            alpha,
            beta,
            hp_stream,
            buf: vec![0.0; 2],
            idx: 0,
            len: 0,
            last_hp: [0.0; 2],
            last_out: [0.0; 2],
        })
    }

    pub fn update(&mut self, value: f64) -> f64 {
        let hp_val = self.hp_stream.update(value);
        // rotate buffers for hp and output
        let prev_hp2 = self.last_hp[0];
        let prev_hp1 = self.last_hp[1];
        let prev_out2 = self.last_out[0];
        let prev_out1 = self.last_out[1];

        let out_val = if self.len < 2 {
            self.len += 1;
            self.last_hp[0] = prev_hp1;
            self.last_hp[1] = hp_val;
            self.last_out[0] = prev_out1;
            self.last_out[1] = hp_val;
            hp_val
        } else {
            let res = 0.5 * (1.0 - self.alpha) * hp_val
                - 0.5 * (1.0 - self.alpha) * prev_hp2
                + self.beta * (1.0 + self.alpha) * prev_out1
                - self.alpha * prev_out2;
            self.last_hp[0] = prev_hp1;
            self.last_hp[1] = hp_val;
            self.last_out[0] = prev_out1;
            self.last_out[1] = res;
            res
        };
        out_val
    }
}

#[derive(Clone, Debug)]
pub struct BandPassBatchRange {
    pub period: (usize, usize, usize),
    pub bandwidth: (f64, f64, f64),
}

impl Default for BandPassBatchRange {
    fn default() -> Self {
        Self {
            period: (20, 60, 1),
            bandwidth: (0.3, 0.3, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BandPassBatchBuilder {
    range: BandPassBatchRange,
    kernel: Kernel,
}

impl BandPassBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    #[inline]
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    #[inline]
    pub fn bandwidth_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.bandwidth = (start, end, step);
        self
    }
    #[inline]
    pub fn bandwidth_static(mut self, b: f64) -> Self {
        self.range.bandwidth = (b, b, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<BandPassBatchOutput, BandPassError> {
        bandpass_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<BandPassBatchOutput, BandPassError> {
        BandPassBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<BandPassBatchOutput, BandPassError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<BandPassBatchOutput, BandPassError> {
        BandPassBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct BandPassBatchOutput {
    pub values: Vec<BandPassOutput>,
    pub combos: Vec<BandPassParams>,
    pub rows: usize,
    pub cols: usize,
}

impl BandPassBatchOutput {
    pub fn row_for_params(&self, p: &BandPassParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(20) == p.period.unwrap_or(20)
                && (c.bandwidth.unwrap_or(0.3) - p.bandwidth.unwrap_or(0.3)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &BandPassParams) -> Option<&BandPassOutput> {
        self.row_for_params(p).map(|row| &self.values[row])
    }
}

#[inline(always)]
fn expand_grid(r: &BandPassBatchRange) -> Vec<BandPassParams> {
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
    let periods = axis_usize(r.period);
    let bandwidths = axis_f64(r.bandwidth);
    let mut out = Vec::with_capacity(periods.len() * bandwidths.len());
    for &p in &periods {
        for &b in &bandwidths {
            out.push(BandPassParams {
                period: Some(p),
                bandwidth: Some(b),
            });
        }
    }
    out
}

pub fn bandpass_batch_with_kernel(
    data: &[f64],
    sweep: &BandPassBatchRange,
    k: Kernel,
) -> Result<BandPassBatchOutput, BandPassError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(BandPassError::InvalidPeriod { period: 0 });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    bandpass_batch_par_slice(data, sweep, simd)
}

pub fn bandpass_batch_slice(
    data: &[f64],
    sweep: &BandPassBatchRange,
    kern: Kernel,
) -> Result<BandPassBatchOutput, BandPassError> {
    bandpass_batch_inner(data, sweep, kern, false)
}

pub fn bandpass_batch_par_slice(
    data: &[f64],
    sweep: &BandPassBatchRange,
    kern: Kernel,
) -> Result<BandPassBatchOutput, BandPassError> {
    bandpass_batch_inner(data, sweep, kern, true)
}

fn bandpass_batch_inner(
    data: &[f64],
    sweep: &BandPassBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<BandPassBatchOutput, BandPassError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(BandPassError::InvalidPeriod { period: 0 });
    }
    let len = data.len();
    let mut outputs: Vec<Option<BandPassOutput>> = vec![None; combos.len()];
    let do_row = |row: usize| -> Result<BandPassOutput, BandPassError> {
        let p = combos[row].clone();
        let input = BandPassInput::from_slice(data, p);
        bandpass_with_kernel(&input, kern)
    };
    if parallel {
        outputs.par_iter_mut().enumerate().for_each(|(row, slot)| {
            *slot = do_row(row).ok();
        });
    } else {
        for (row, slot) in outputs.iter_mut().enumerate() {
            *slot = do_row(row).ok();
        }
    }
    let mut values = Vec::with_capacity(combos.len());
    for v in outputs {
        values.push(v.ok_or_else(|| BandPassError::InvalidPeriod { period: 0 })?);
    }
    let rows = combos.len();
    let combos_clone = combos.clone();
    Ok(BandPassBatchOutput {
        values,
        combos: combos_clone,
        rows,
        cols: len,
    })
}

#[inline(always)]
pub fn bandpass_row_scalar(
    hp: &[f64],
    period: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_row_avx2(
    hp: &[f64],
    period: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_row_avx512(
    hp: &[f64],
    period: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_row_avx512_short(
    hp: &[f64],
    period: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    bandpass_scalar(hp, period, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn bandpass_row_avx512_long(
    hp: &[f64],
    period: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    bandpass_scalar(hp, period, alpha, beta, out)
}

#[inline(always)]
fn expand_grid_for_bandpass(r: &BandPassBatchRange) -> Vec<BandPassParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_bandpass_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = BandPassParams::default();
        let input = BandPassInput::from_candles(&candles, "close", default_params);
        let output = bandpass_with_kernel(&input, kernel)?;
        assert_eq!(output.bp.len(), candles.close.len());
        Ok(())
    }
    fn check_bandpass_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = BandPassInput::with_default_candles(&candles);
        let result = bandpass_with_kernel(&input, kernel)?;
        let expected_bp_last_five = [
            -236.23678021132827,
            -247.4846395608195,
            -242.3788746078502,
            -212.89589193350128,
            -179.97293838509464,
        ];
        let expected_bp_normalized_last_five = [
            -0.4399672555578846,
            -0.4651011734720517,
            -0.4596426251402882,
            -0.40739824942488945,
            -0.3475245023284841,
        ];
        let expected_signal_last_five = [-1.0, 1.0, 1.0, 1.0, 1.0];
        let expected_trigger_last_five = [
            -0.4746908356434579,
            -0.4353877348116954,
            -0.3727126131420441,
            -0.2746336628365846,
            -0.18240018384226137,
        ];
        let start = result.bp.len().saturating_sub(5);
        assert!(result.bp.len() >= 5);
        assert!(result.bp_normalized.len() >= 5);
        assert!(result.signal.len() >= 5);
        assert!(result.trigger.len() >= 5);
        for (i, &value) in result.bp[start..].iter().enumerate() {
            assert!(
                (value - expected_bp_last_five[i]).abs() < 1e-1,
                "BP value mismatch at index {}: expected {}, got {}",
                i, expected_bp_last_five[i], value
            );
        }
        for (i, &value) in result.bp_normalized[start..].iter().enumerate() {
            assert!(
                (value - expected_bp_normalized_last_five[i]).abs() < 1e-1,
                "BP Normalized value mismatch at index {}: expected {}, got {}",
                i, expected_bp_normalized_last_five[i], value
            );
        }
        for (i, &value) in result.signal[start..].iter().enumerate() {
            assert!(
                (value - expected_signal_last_five[i]).abs() < 1e-1,
                "Signal value mismatch at index {}: expected {}, got {}",
                i, expected_signal_last_five[i], value
            );
        }
        for (i, &value) in result.trigger[start..].iter().enumerate() {
            assert!(
                (value - expected_trigger_last_five[i]).abs() < 1e-1,
                "Trigger value mismatch at index {}: expected {}, got {}",
                i, expected_trigger_last_five[i], value
            );
        }
        for val in &result.bp {
            assert!(val.is_finite());
        }
        for val in &result.bp_normalized {
            assert!(val.is_finite());
        }
        for val in &result.signal {
            assert!(val.is_finite());
        }
        for val in &result.trigger {
            assert!(val.is_finite());
        }
        Ok(())
    }
    fn check_bandpass_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = BandPassInput::with_default_candles(&candles);
        match input.data {
            BandPassData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected BandPassData::Candles"),
        }
        let output = bandpass_with_kernel(&input, kernel)?;
        assert_eq!(output.bp.len(), candles.close.len());
        Ok(())
    }
    fn check_bandpass_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = BandPassParams {
            period: Some(0),
            bandwidth: Some(0.3),
        };
        let input = BandPassInput::from_slice(&input_data, params);
        let res = bandpass_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }
    fn check_bandpass_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = BandPassParams {
            period: Some(10),
            bandwidth: Some(0.3),
        };
        let input = BandPassInput::from_slice(&data_small, params);
        let res = bandpass_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }
    fn check_bandpass_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = BandPassParams {
            period: Some(20),
            bandwidth: Some(0.3),
        };
        let input = BandPassInput::from_slice(&single_point, params);
        let res = bandpass_with_kernel(&input, kernel);
        assert!(res.is_err());
        Ok(())
    }
    fn check_bandpass_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = BandPassParams {
            period: Some(20),
            bandwidth: Some(0.3),
        };
        let first_input = BandPassInput::from_candles(&candles, "close", first_params);
        let first_result = bandpass_with_kernel(&first_input, kernel)?;
        let second_params = BandPassParams {
            period: Some(30),
            bandwidth: Some(0.5),
        };
        let second_input = BandPassInput::from_slice(&first_result.bp, second_params);
        let second_result = bandpass_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.bp.len(), first_result.bp.len());
        Ok(())
    }
    fn check_bandpass_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = BandPassInput::from_candles(
            &candles,
            "close",
            BandPassParams {
                period: Some(20),
                bandwidth: Some(0.3),
            },
        );
        let res = bandpass_with_kernel(&input, kernel)?;
        assert_eq!(res.bp.len(), candles.close.len());
        if res.bp.len() > 30 {
            for i in 30..res.bp.len() {
                assert!(!res.bp[i].is_nan());
                assert!(!res.bp_normalized[i].is_nan());
                assert!(!res.signal[i].is_nan());
                assert!(!res.trigger[i].is_nan());
            }
        }
        Ok(())
    }

    macro_rules! generate_all_bandpass_tests {
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

    generate_all_bandpass_tests!(
        check_bandpass_partial_params,
        check_bandpass_accuracy,
        check_bandpass_default_candles,
        check_bandpass_zero_period,
        check_bandpass_period_exceeds_length,
        check_bandpass_very_small_dataset,
        check_bandpass_reinput,
        check_bandpass_nan_handling
    );
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
    skip_if_unsupported!(kernel, test);

    let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let c = read_candles_from_csv(file)?;

    let output = BandPassBatchBuilder::new()
        .kernel(kernel)
        .apply_candles(&c, "close")?;

    let def = BandPassParams::default();
    let row = output.values_for(&def).expect("default row missing");

    assert_eq!(row.bp.len(), c.close.len());

    // Optional: Test known last 5 values for one column (bp)
    let expected = [
        -236.23678021132827,
        -247.4846395608195,
        -242.3788746078502,
        -212.89589193350128,
        -179.97293838509464,
    ];
    let start = row.bp.len() - 5;
    for (i, &v) in row.bp[start..].iter().enumerate() {
        assert!(
            (v - expected[i]).abs() < 1e-1,
            "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
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
