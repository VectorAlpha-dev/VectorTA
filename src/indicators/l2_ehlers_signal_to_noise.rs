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

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "hl2";
const DEFAULT_SMOOTH_PERIOD: usize = 10;
const MIN_WARMUP_BARS: usize = 6;

#[derive(Debug, Clone)]
pub enum L2EhlersSignalToNoiseData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        source: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct L2EhlersSignalToNoiseOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct L2EhlersSignalToNoiseParams {
    pub smooth_period: Option<usize>,
}

impl Default for L2EhlersSignalToNoiseParams {
    fn default() -> Self {
        Self {
            smooth_period: Some(DEFAULT_SMOOTH_PERIOD),
        }
    }
}

#[derive(Debug, Clone)]
pub struct L2EhlersSignalToNoiseInput<'a> {
    pub data: L2EhlersSignalToNoiseData<'a>,
    pub params: L2EhlersSignalToNoiseParams,
}

impl<'a> L2EhlersSignalToNoiseInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: L2EhlersSignalToNoiseParams,
    ) -> Self {
        Self {
            data: L2EhlersSignalToNoiseData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        source: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        params: L2EhlersSignalToNoiseParams,
    ) -> Self {
        Self {
            data: L2EhlersSignalToNoiseData::Slices { source, high, low },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            L2EhlersSignalToNoiseParams::default(),
        )
    }

    #[inline]
    pub fn get_smooth_period(&self) -> usize {
        self.params.smooth_period.unwrap_or(DEFAULT_SMOOTH_PERIOD)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct L2EhlersSignalToNoiseBuilder {
    source: Option<&'static str>,
    smooth_period: Option<usize>,
    kernel: Kernel,
}

impl Default for L2EhlersSignalToNoiseBuilder {
    fn default() -> Self {
        Self {
            source: None,
            smooth_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl L2EhlersSignalToNoiseBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: &'static str) -> Self {
        self.source = Some(value);
        self
    }

    #[inline(always)]
    pub fn smooth_period(mut self, value: usize) -> Self {
        self.smooth_period = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<L2EhlersSignalToNoiseOutput, L2EhlersSignalToNoiseError> {
        let input = L2EhlersSignalToNoiseInput::from_candles(
            candles,
            self.source.unwrap_or(DEFAULT_SOURCE),
            L2EhlersSignalToNoiseParams {
                smooth_period: self.smooth_period,
            },
        );
        l2_ehlers_signal_to_noise_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        high: &[f64],
        low: &[f64],
    ) -> Result<L2EhlersSignalToNoiseOutput, L2EhlersSignalToNoiseError> {
        let input = L2EhlersSignalToNoiseInput::from_slices(
            source,
            high,
            low,
            L2EhlersSignalToNoiseParams {
                smooth_period: self.smooth_period,
            },
        );
        l2_ehlers_signal_to_noise_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<L2EhlersSignalToNoiseStream, L2EhlersSignalToNoiseError> {
        L2EhlersSignalToNoiseStream::try_new(L2EhlersSignalToNoiseParams {
            smooth_period: self.smooth_period,
        })
    }
}

#[derive(Debug, Error)]
pub enum L2EhlersSignalToNoiseError {
    #[error("l2_ehlers_signal_to_noise: Input data slice is empty.")]
    EmptyInputData,
    #[error("l2_ehlers_signal_to_noise: All values are NaN.")]
    AllValuesNaN,
    #[error(
        "l2_ehlers_signal_to_noise: Inconsistent slice lengths: source={source_len}, high={high_len}, low={low_len}"
    )]
    InconsistentSliceLengths {
        source_len: usize,
        high_len: usize,
        low_len: usize,
    },
    #[error("l2_ehlers_signal_to_noise: Invalid smooth_period: {smooth_period}")]
    InvalidSmoothPeriod { smooth_period: usize },
    #[error("l2_ehlers_signal_to_noise: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("l2_ehlers_signal_to_noise: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("l2_ehlers_signal_to_noise: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("l2_ehlers_signal_to_noise: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn extract_triples<'a>(
    input: &'a L2EhlersSignalToNoiseInput<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64]), L2EhlersSignalToNoiseError> {
    let (source, high, low) = match &input.data {
        L2EhlersSignalToNoiseData::Candles { candles, source } => (
            l2_source(candles, source),
            candles.high.as_slice(),
            candles.low.as_slice(),
        ),
        L2EhlersSignalToNoiseData::Slices { source, high, low } => (*source, *high, *low),
    };
    if source.is_empty() || high.is_empty() || low.is_empty() {
        return Err(L2EhlersSignalToNoiseError::EmptyInputData);
    }
    if source.len() != high.len() || source.len() != low.len() {
        return Err(L2EhlersSignalToNoiseError::InconsistentSliceLengths {
            source_len: source.len(),
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    Ok((source, high, low))
}

#[inline(always)]
fn l2_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "hl2" => &candles.hl2,
        "close" => &candles.close,
        "open" => &candles.open,
        "high" => &candles.high,
        "low" => &candles.low,
        "volume" => &candles.volume,
        _ => source_type(candles, source),
    }
}

#[inline(always)]
fn first_valid_triple(source: &[f64], high: &[f64], low: &[f64]) -> Option<usize> {
    (0..source.len()).find(|&i| source[i].is_finite() && high[i].is_finite() && low[i].is_finite())
}

#[inline(always)]
fn validate_input<'a>(
    input: &'a L2EhlersSignalToNoiseInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], usize, usize, Kernel), L2EhlersSignalToNoiseError> {
    let (source, high, low) = extract_triples(input)?;
    let smooth_period = input.get_smooth_period();
    if smooth_period == 0 {
        return Err(L2EhlersSignalToNoiseError::InvalidSmoothPeriod { smooth_period });
    }
    let first =
        first_valid_triple(source, high, low).ok_or(L2EhlersSignalToNoiseError::AllValuesNaN)?;
    let valid = source.len().saturating_sub(first);
    let needed = MIN_WARMUP_BARS + 1;
    if valid < needed {
        return Err(L2EhlersSignalToNoiseError::NotEnoughValidData { needed, valid });
    }
    Ok((
        source,
        high,
        low,
        smooth_period,
        first,
        kernel.to_non_batch(),
    ))
}

#[inline(always)]
fn ring_get<const N: usize>(buf: &[f64; N], center: usize, off: usize) -> f64 {
    let mut idx = center + N - (off % N);
    if idx >= N {
        idx -= N;
    }
    buf[idx]
}

#[derive(Clone, Debug)]
struct SignalToNoiseCore {
    period_mult: f64,
    source_ring: [f64; 4],
    source_idx: usize,
    smooth_ring: [f64; 7],
    smooth_idx: usize,
    detrender_ring: [f64; 7],
    detrender_idx: usize,
    q1_ring: [f64; 7],
    q1_idx: usize,
    i1_ring: [f64; 7],
    i1_idx: usize,
    range_1: f64,
    i2: f64,
    q2: f64,
    re: f64,
    im: f64,
    period: f64,
    snr: f64,
    valid_count: usize,
}

impl SignalToNoiseCore {
    #[inline(always)]
    fn new(smooth_period: usize) -> Self {
        Self {
            period_mult: 0.075 * smooth_period as f64 + 0.54,
            source_ring: [0.0; 4],
            source_idx: 0,
            smooth_ring: [0.0; 7],
            smooth_idx: 0,
            detrender_ring: [0.0; 7],
            detrender_idx: 0,
            q1_ring: [0.0; 7],
            q1_idx: 0,
            i1_ring: [0.0; 7],
            i1_idx: 0,
            range_1: 0.0,
            i2: 0.0,
            q2: 0.0,
            re: 0.0,
            im: 0.0,
            period: 0.0,
            snr: 0.0,
            valid_count: 0,
        }
    }

    #[inline(always)]
    fn update(&mut self, source: f64, high: f64, low: f64) -> f64 {
        if !(source.is_finite() && high.is_finite() && low.is_finite()) {
            return f64::NAN;
        }

        self.update_clean(source, high, low)
    }

    #[inline(always)]
    fn update_clean(&mut self, source: f64, high: f64, low: f64) -> f64 {
        self.range_1 = 0.1 * (high - low) + 0.9 * self.range_1;
        self.source_ring[self.source_idx] = source;

        let mut smooth = 0.0;
        let mut detrender = 0.0;
        let mut i1 = 0.0;
        let mut q1 = 0.0;

        if self.valid_count > 5 {
            let x0 = ring_get(&self.source_ring, self.source_idx, 0);
            let x1 = ring_get(&self.source_ring, self.source_idx, 1);
            let x2 = ring_get(&self.source_ring, self.source_idx, 2);
            let x3 = ring_get(&self.source_ring, self.source_idx, 3);
            smooth = (4.0 * x0 + 3.0 * x1 + 2.0 * x2 + x3) / 10.0;

            self.smooth_ring[self.smooth_idx] = smooth;
            let s0 = ring_get(&self.smooth_ring, self.smooth_idx, 0);
            let s2 = ring_get(&self.smooth_ring, self.smooth_idx, 2);
            let s4 = ring_get(&self.smooth_ring, self.smooth_idx, 4);
            let s6 = ring_get(&self.smooth_ring, self.smooth_idx, 6);
            detrender = (0.0962 * s0 + 0.5769 * s2 - 0.5769 * s4 - 0.0962 * s6) * self.period_mult;

            self.detrender_ring[self.detrender_idx] = detrender;
            i1 = ring_get(&self.detrender_ring, self.detrender_idx, 3);
            self.i1_ring[self.i1_idx] = i1;

            let d0 = ring_get(&self.detrender_ring, self.detrender_idx, 0);
            let d2 = ring_get(&self.detrender_ring, self.detrender_idx, 2);
            let d4 = ring_get(&self.detrender_ring, self.detrender_idx, 4);
            let d6 = ring_get(&self.detrender_ring, self.detrender_idx, 6);
            q1 = (0.0962 * d0 + 0.5769 * d2 - 0.5769 * d4 - 0.0962 * d6) * self.period_mult;
            self.q1_ring[self.q1_idx] = q1;

            let i0 = ring_get(&self.i1_ring, self.i1_idx, 0);
            let i2 = ring_get(&self.i1_ring, self.i1_idx, 2);
            let i4 = ring_get(&self.i1_ring, self.i1_idx, 4);
            let i6 = ring_get(&self.i1_ring, self.i1_idx, 6);
            let ji = (0.0962 * i0 + 0.5769 * i2 - 0.5769 * i4 - 0.0962 * i6) * self.period_mult;

            let q0 = ring_get(&self.q1_ring, self.q1_idx, 0);
            let q2_hist = ring_get(&self.q1_ring, self.q1_idx, 2);
            let q4 = ring_get(&self.q1_ring, self.q1_idx, 4);
            let q6 = ring_get(&self.q1_ring, self.q1_idx, 6);
            let jq =
                (0.0962 * q0 + 0.5769 * q2_hist - 0.5769 * q4 - 0.0962 * q6) * self.period_mult;

            let prev_i2 = self.i2;
            let prev_q2 = self.q2;
            let prev_re = self.re;
            let prev_im = self.im;
            let prev_period = self.period;
            let prev_snr = self.snr;

            self.i2 = 0.2 * (i1 - jq) + 0.8 * prev_i2;
            self.q2 = 0.2 * (q1 + ji) + 0.8 * prev_q2;

            let re_raw = self.i2 * prev_i2 + self.q2 * prev_q2;
            let im_raw = self.i2 * prev_q2 - self.q2 * prev_i2;
            self.re = 0.2 * re_raw + 0.8 * prev_re;
            self.im = 0.2 * im_raw + 0.8 * prev_im;

            let mut period = prev_period;
            if self.re != 0.0 && self.im != 0.0 {
                let angle = self.im.atan2(self.re);
                if angle != 0.0 {
                    period = (2.0 * std::f64::consts::PI) / angle.abs();
                }
            }
            if prev_period != 0.0 {
                let upper = 1.5 * prev_period;
                let lower = 0.67 * prev_period;
                if period > upper {
                    period = upper;
                }
                if period < lower {
                    period = lower;
                }
            }
            period = period.clamp(6.0, 50.0);
            self.period = 0.2 * period + 0.8 * prev_period;

            let power = i1 * i1 + q1 * q1;
            let noise = self.range_1 * self.range_1;
            if power > 0.0 && noise > 0.0 {
                let snr_raw = 10.0 * (power / noise).ln() / std::f64::consts::LN_10 + 6.0;
                self.snr = 0.25 * snr_raw + 0.75 * prev_snr;
            } else {
                self.snr = prev_snr;
            }
        } else {
            self.smooth_ring[self.smooth_idx] = smooth;
            self.detrender_ring[self.detrender_idx] = detrender;
            self.i1_ring[self.i1_idx] = i1;
            self.q1_ring[self.q1_idx] = q1;
        }

        self.valid_count += 1;
        self.source_idx += 1;
        if self.source_idx == 4 {
            self.source_idx = 0;
        }
        self.smooth_idx += 1;
        if self.smooth_idx == 7 {
            self.smooth_idx = 0;
        }
        self.detrender_idx += 1;
        if self.detrender_idx == 7 {
            self.detrender_idx = 0;
        }
        self.i1_idx += 1;
        if self.i1_idx == 7 {
            self.i1_idx = 0;
        }
        self.q1_idx += 1;
        if self.q1_idx == 7 {
            self.q1_idx = 0;
        }

        if self.valid_count <= MIN_WARMUP_BARS {
            f64::NAN
        } else {
            self.snr
        }
    }
}

#[inline(always)]
fn compute_l2_ehlers_signal_to_noise_into(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    smooth_period: usize,
    first: usize,
    out: &mut [f64],
) -> Result<(), L2EhlersSignalToNoiseError> {
    let n = source.len();
    if out.len() != n {
        return Err(L2EhlersSignalToNoiseError::OutputLengthMismatch {
            expected: n,
            got: out.len(),
        });
    }
    if compute_l2_ehlers_signal_to_noise_clean(source, high, low, smooth_period, first, out) {
        return Ok(());
    }
    let mut core = SignalToNoiseCore::new(smooth_period);
    for i in 0..n {
        out[i] = core.update(source[i], high[i], low[i]);
    }
    Ok(())
}

#[inline(always)]
fn compute_l2_ehlers_signal_to_noise_clean(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    smooth_period: usize,
    first: usize,
    out: &mut [f64],
) -> bool {
    let mut core = SignalToNoiseCore::new(smooth_period);
    for value in &mut out[..first] {
        *value = f64::NAN;
    }
    for i in first..source.len() {
        let source_value = source[i];
        let high_value = high[i];
        let low_value = low[i];
        if !(source_value.is_finite() && high_value.is_finite() && low_value.is_finite()) {
            return false;
        }
        out[i] = core.update_clean(source_value, high_value, low_value);
    }
    true
}

#[inline(always)]
fn alloc_l2_output(len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    unsafe {
        out.set_len(len);
    }
    out
}

#[inline]
pub fn l2_ehlers_signal_to_noise(
    input: &L2EhlersSignalToNoiseInput,
) -> Result<L2EhlersSignalToNoiseOutput, L2EhlersSignalToNoiseError> {
    l2_ehlers_signal_to_noise_with_kernel(input, Kernel::Auto)
}

pub fn l2_ehlers_signal_to_noise_with_kernel(
    input: &L2EhlersSignalToNoiseInput,
    kernel: Kernel,
) -> Result<L2EhlersSignalToNoiseOutput, L2EhlersSignalToNoiseError> {
    let (source, high, low, smooth_period, first, _kernel) = validate_input(input, kernel)?;
    let mut out = alloc_l2_output(source.len());
    compute_l2_ehlers_signal_to_noise_into(source, high, low, smooth_period, first, &mut out)?;
    Ok(L2EhlersSignalToNoiseOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn l2_ehlers_signal_to_noise_into(
    out: &mut [f64],
    input: &L2EhlersSignalToNoiseInput,
    kernel: Kernel,
) -> Result<(), L2EhlersSignalToNoiseError> {
    l2_ehlers_signal_to_noise_into_slice(out, input, kernel)
}

pub fn l2_ehlers_signal_to_noise_into_slice(
    out: &mut [f64],
    input: &L2EhlersSignalToNoiseInput,
    kernel: Kernel,
) -> Result<(), L2EhlersSignalToNoiseError> {
    let (source, high, low, smooth_period, first, _kernel) = validate_input(input, kernel)?;
    compute_l2_ehlers_signal_to_noise_into(source, high, low, smooth_period, first, out)
}

#[derive(Clone, Debug)]
pub struct L2EhlersSignalToNoiseStream {
    core: SignalToNoiseCore,
}

impl L2EhlersSignalToNoiseStream {
    pub fn try_new(
        params: L2EhlersSignalToNoiseParams,
    ) -> Result<Self, L2EhlersSignalToNoiseError> {
        let smooth_period = params.smooth_period.unwrap_or(DEFAULT_SMOOTH_PERIOD);
        if smooth_period == 0 {
            return Err(L2EhlersSignalToNoiseError::InvalidSmoothPeriod { smooth_period });
        }
        Ok(Self {
            core: SignalToNoiseCore::new(smooth_period),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, source: f64, high: f64, low: f64) -> f64 {
        self.core.update(source, high, low)
    }
}

#[derive(Clone, Debug)]
pub struct L2EhlersSignalToNoiseBatchRange {
    pub smooth_period: (usize, usize, usize),
}

#[derive(Clone, Debug)]
pub struct L2EhlersSignalToNoiseBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<L2EhlersSignalToNoiseParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct L2EhlersSignalToNoiseBatchBuilder {
    source: Option<&'static str>,
    smooth_period: (usize, usize, usize),
    kernel: Kernel,
}

impl Default for L2EhlersSignalToNoiseBatchBuilder {
    fn default() -> Self {
        Self {
            source: None,
            smooth_period: (DEFAULT_SMOOTH_PERIOD, DEFAULT_SMOOTH_PERIOD, 0),
            kernel: Kernel::Auto,
        }
    }
}

impl L2EhlersSignalToNoiseBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn source(mut self, value: &'static str) -> Self {
        self.source = Some(value);
        self
    }

    #[inline(always)]
    pub fn smooth_period_range(mut self, value: (usize, usize, usize)) -> Self {
        self.smooth_period = value;
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<L2EhlersSignalToNoiseBatchOutput, L2EhlersSignalToNoiseError> {
        let source = source_type(candles, self.source.unwrap_or(DEFAULT_SOURCE));
        l2_ehlers_signal_to_noise_batch_with_kernel(
            source,
            candles.high.as_slice(),
            candles.low.as_slice(),
            &L2EhlersSignalToNoiseBatchRange {
                smooth_period: self.smooth_period,
            },
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        source: &[f64],
        high: &[f64],
        low: &[f64],
    ) -> Result<L2EhlersSignalToNoiseBatchOutput, L2EhlersSignalToNoiseError> {
        l2_ehlers_signal_to_noise_batch_with_kernel(
            source,
            high,
            low,
            &L2EhlersSignalToNoiseBatchRange {
                smooth_period: self.smooth_period,
            },
            self.kernel,
        )
    }
}

pub fn expand_grid(
    sweep: &L2EhlersSignalToNoiseBatchRange,
) -> Result<Vec<L2EhlersSignalToNoiseParams>, L2EhlersSignalToNoiseError> {
    let (start, end, step) = sweep.smooth_period;
    if start == 0 {
        return Err(L2EhlersSignalToNoiseError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut values = Vec::new();
    if step == 0 {
        if start != end {
            return Err(L2EhlersSignalToNoiseError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        values.push(start);
    } else {
        if start > end {
            return Err(L2EhlersSignalToNoiseError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        let mut current = start;
        while current <= end {
            values.push(current);
            current = match current.checked_add(step) {
                Some(next) => next,
                None => break,
            };
        }
    }
    Ok(values
        .into_iter()
        .map(|smooth_period| L2EhlersSignalToNoiseParams {
            smooth_period: Some(smooth_period),
        })
        .collect())
}

fn validate_raw_slices(
    source: &[f64],
    high: &[f64],
    low: &[f64],
) -> Result<usize, L2EhlersSignalToNoiseError> {
    if source.is_empty() || high.is_empty() || low.is_empty() {
        return Err(L2EhlersSignalToNoiseError::EmptyInputData);
    }
    if source.len() != high.len() || source.len() != low.len() {
        return Err(L2EhlersSignalToNoiseError::InconsistentSliceLengths {
            source_len: source.len(),
            high_len: high.len(),
            low_len: low.len(),
        });
    }
    first_valid_triple(source, high, low).ok_or(L2EhlersSignalToNoiseError::AllValuesNaN)
}

pub fn l2_ehlers_signal_to_noise_batch_with_kernel(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    sweep: &L2EhlersSignalToNoiseBatchRange,
    kernel: Kernel,
) -> Result<L2EhlersSignalToNoiseBatchOutput, L2EhlersSignalToNoiseError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(L2EhlersSignalToNoiseError::InvalidKernelForBatch(kernel)),
    };
    l2_ehlers_signal_to_noise_batch_par_slice(source, high, low, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn l2_ehlers_signal_to_noise_batch_slice(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    sweep: &L2EhlersSignalToNoiseBatchRange,
    kernel: Kernel,
) -> Result<L2EhlersSignalToNoiseBatchOutput, L2EhlersSignalToNoiseError> {
    l2_ehlers_signal_to_noise_batch_inner(source, high, low, sweep, kernel, false)
}

#[inline(always)]
pub fn l2_ehlers_signal_to_noise_batch_par_slice(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    sweep: &L2EhlersSignalToNoiseBatchRange,
    kernel: Kernel,
) -> Result<L2EhlersSignalToNoiseBatchOutput, L2EhlersSignalToNoiseError> {
    l2_ehlers_signal_to_noise_batch_inner(source, high, low, sweep, kernel, true)
}

fn l2_ehlers_signal_to_noise_batch_inner(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    sweep: &L2EhlersSignalToNoiseBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<L2EhlersSignalToNoiseBatchOutput, L2EhlersSignalToNoiseError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(source, high, low)?;
    let rows = combos.len();
    let cols = source.len();
    let warmups = vec![(first + MIN_WARMUP_BARS).min(cols); rows];

    let mut buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf, cols, &warmups);
    let mut guard = ManuallyDrop::new(buf);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    l2_ehlers_signal_to_noise_batch_inner_into(source, high, low, sweep, kernel, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(L2EhlersSignalToNoiseBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn l2_ehlers_signal_to_noise_batch_into_slice(
    out: &mut [f64],
    source: &[f64],
    high: &[f64],
    low: &[f64],
    sweep: &L2EhlersSignalToNoiseBatchRange,
    kernel: Kernel,
) -> Result<(), L2EhlersSignalToNoiseError> {
    l2_ehlers_signal_to_noise_batch_inner_into(source, high, low, sweep, kernel, false, out)?;
    Ok(())
}

fn l2_ehlers_signal_to_noise_batch_inner_into(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    sweep: &L2EhlersSignalToNoiseBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<L2EhlersSignalToNoiseParams>, L2EhlersSignalToNoiseError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(source, high, low)?;
    let rows = combos.len();
    let cols = source.len();
    let expected =
        rows.checked_mul(cols)
            .ok_or_else(|| L2EhlersSignalToNoiseError::InvalidRange {
                start: rows.to_string(),
                end: cols.to_string(),
                step: "rows*cols".to_string(),
            })?;
    if out.len() != expected {
        return Err(L2EhlersSignalToNoiseError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    let valid = cols.saturating_sub(first);
    if valid < MIN_WARMUP_BARS + 1 {
        return Err(L2EhlersSignalToNoiseError::NotEnoughValidData {
            needed: MIN_WARMUP_BARS + 1,
            valid,
        });
    }

    let do_row = |row: usize, dst: &mut [f64]| -> Result<(), L2EhlersSignalToNoiseError> {
        let smooth_period = combos[row].smooth_period.unwrap_or(DEFAULT_SMOOTH_PERIOD);
        compute_l2_ehlers_signal_to_noise_into(source, high, low, smooth_period, first, dst)
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .try_for_each(|(row, dst)| do_row(row, dst))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, dst) in out.chunks_mut(cols).enumerate() {
                do_row(row, dst)?;
            }
        }
    } else {
        for (row, dst) in out.chunks_mut(cols).enumerate() {
            do_row(row, dst)?;
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "l2_ehlers_signal_to_noise")]
#[pyo3(signature = (source, high, low, smooth_period=10, kernel=None))]
pub fn l2_ehlers_signal_to_noise_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    smooth_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let source = source.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = L2EhlersSignalToNoiseInput::from_slices(
        source,
        high,
        low,
        L2EhlersSignalToNoiseParams {
            smooth_period: Some(smooth_period),
        },
    );
    let out = py
        .allow_threads(|| l2_ehlers_signal_to_noise_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "L2EhlersSignalToNoiseStream")]
pub struct L2EhlersSignalToNoiseStreamPy {
    stream: L2EhlersSignalToNoiseStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl L2EhlersSignalToNoiseStreamPy {
    #[new]
    #[pyo3(signature = (smooth_period=10))]
    fn new(smooth_period: usize) -> PyResult<Self> {
        let stream = L2EhlersSignalToNoiseStream::try_new(L2EhlersSignalToNoiseParams {
            smooth_period: Some(smooth_period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, source: f64, high: f64, low: f64) -> f64 {
        self.stream.update(source, high, low)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "l2_ehlers_signal_to_noise_batch")]
#[pyo3(signature = (source, high, low, smooth_period_range=(10,10,0), kernel=None))]
pub fn l2_ehlers_signal_to_noise_batch_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    smooth_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = source.as_slice()?;
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = L2EhlersSignalToNoiseBatchRange {
        smooth_period: smooth_period_range,
    };
    let out = py
        .allow_threads(|| {
            l2_ehlers_signal_to_noise_batch_with_kernel(source, high, low, &sweep, kernel)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item(
        "values",
        out.values
            .clone()
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "smooth_periods",
        out.combos
            .iter()
            .map(|combo| combo.smooth_period.unwrap_or(DEFAULT_SMOOTH_PERIOD) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_l2_ehlers_signal_to_noise_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(l2_ehlers_signal_to_noise_py, m)?)?;
    m.add_function(wrap_pyfunction!(l2_ehlers_signal_to_noise_batch_py, m)?)?;
    m.add_class::<L2EhlersSignalToNoiseStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "l2_ehlers_signal_to_noise_js")]
pub fn l2_ehlers_signal_to_noise_js(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    smooth_period: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = L2EhlersSignalToNoiseInput::from_slices(
        source,
        high,
        low,
        L2EhlersSignalToNoiseParams {
            smooth_period: Some(smooth_period),
        },
    );
    let mut out = vec![0.0; source.len()];
    l2_ehlers_signal_to_noise_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct L2EhlersSignalToNoiseBatchConfig {
    pub smooth_period_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct L2EhlersSignalToNoiseBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<L2EhlersSignalToNoiseParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "l2_ehlers_signal_to_noise_batch_js")]
pub fn l2_ehlers_signal_to_noise_batch_js(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: L2EhlersSignalToNoiseBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.smooth_period_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: smooth_period_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let out = l2_ehlers_signal_to_noise_batch_with_kernel(
        source,
        high,
        low,
        &L2EhlersSignalToNoiseBatchRange {
            smooth_period: (
                config.smooth_period_range[0],
                config.smooth_period_range[1],
                config.smooth_period_range[2],
            ),
        },
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&L2EhlersSignalToNoiseBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn l2_ehlers_signal_to_noise_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn l2_ehlers_signal_to_noise_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "l2_ehlers_signal_to_noise_into")]
pub fn l2_ehlers_signal_to_noise_into_wasm(
    source_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    smooth_period: usize,
) -> Result<(), JsValue> {
    if source_ptr.is_null() || high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }
    unsafe {
        let source = std::slice::from_raw_parts(source_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = L2EhlersSignalToNoiseInput::from_slices(
            source,
            high,
            low,
            L2EhlersSignalToNoiseParams {
                smooth_period: Some(smooth_period),
            },
        );
        l2_ehlers_signal_to_noise_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "l2_ehlers_signal_to_noise_batch_into")]
pub fn l2_ehlers_signal_to_noise_batch_into(
    source_ptr: *const f64,
    high_ptr: *const f64,
    low_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    smooth_period_start: usize,
    smooth_period_end: usize,
    smooth_period_step: usize,
) -> Result<usize, JsValue> {
    if source_ptr.is_null() || high_ptr.is_null() || low_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to l2_ehlers_signal_to_noise_batch_into",
        ));
    }
    unsafe {
        let source = std::slice::from_raw_parts(source_ptr, len);
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let sweep = L2EhlersSignalToNoiseBatchRange {
            smooth_period: (smooth_period_start, smooth_period_end, smooth_period_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows.checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in l2_ehlers_signal_to_noise_batch_into")
        })?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        l2_ehlers_signal_to_noise_batch_into_slice(out, source, high, low, &sweep, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn l2_ehlers_signal_to_noise_output_into_js(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    smooth_period: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = l2_ehlers_signal_to_noise_js(source, high, low, smooth_period)?;
    crate::write_wasm_f64_output("l2_ehlers_signal_to_noise_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn l2_ehlers_signal_to_noise_batch_output_into_js(
    source: &[f64],
    high: &[f64],
    low: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = l2_ehlers_signal_to_noise_batch_js(source, high, low, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "l2_ehlers_signal_to_noise_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut source = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + (i as f64) * 0.25 + ((i as f64) * 0.17).sin();
            source.push(base + ((i as f64) * 0.11).cos() * 0.4);
            high.push(base + 0.8 + ((i as f64) * 0.07).sin().abs());
            low.push(base - 0.8 - ((i as f64) * 0.05).cos().abs());
        }
        (source, high, low)
    }

    fn assert_close_series(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for i in 0..lhs.len() {
            let a = lhs[i];
            let b = rhs[i];
            assert!(
                (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol,
                "mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn l2_ehlers_signal_to_noise_stream_matches_batch() {
        let (source, high, low) = build_ohlc(128);
        let batch = L2EhlersSignalToNoiseBuilder::new()
            .smooth_period(10)
            .apply_slices(&source, &high, &low)
            .unwrap();
        let mut stream = L2EhlersSignalToNoiseBuilder::new()
            .smooth_period(10)
            .into_stream()
            .unwrap();
        let streamed: Vec<f64> = (0..source.len())
            .map(|i| stream.update(source[i], high[i], low[i]))
            .collect();
        assert_close_series(&batch.values, &streamed, 1e-12);
    }

    #[test]
    fn l2_ehlers_signal_to_noise_batch_rows_match_single() {
        let (source, high, low) = build_ohlc(160);
        let sweep = L2EhlersSignalToNoiseBatchRange {
            smooth_period: (8, 10, 2),
        };
        let batch =
            l2_ehlers_signal_to_noise_batch_with_kernel(&source, &high, &low, &sweep, Kernel::Auto)
                .unwrap();
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, source.len());

        for (row, smooth_period) in [8usize, 10usize].iter().enumerate() {
            let single = L2EhlersSignalToNoiseBuilder::new()
                .smooth_period(*smooth_period)
                .apply_slices(&source, &high, &low)
                .unwrap();
            let start = row * source.len();
            assert_close_series(
                &batch.values[start..start + source.len()],
                &single.values,
                1e-12,
            );
        }
    }

    #[test]
    fn l2_ehlers_signal_to_noise_into_slice_matches_single() {
        let (source, high, low) = build_ohlc(96);
        let input = L2EhlersSignalToNoiseInput::from_slices(
            &source,
            &high,
            &low,
            L2EhlersSignalToNoiseParams {
                smooth_period: Some(10),
            },
        );
        let single = l2_ehlers_signal_to_noise(&input).unwrap();
        let mut out = vec![f64::NAN; source.len()];
        l2_ehlers_signal_to_noise_into_slice(&mut out, &input, Kernel::Auto).unwrap();
        assert_close_series(&single.values, &out, 1e-12);
    }

    #[test]
    fn l2_ehlers_signal_to_noise_rejects_invalid_period() {
        let (source, high, low) = build_ohlc(32);
        let input = L2EhlersSignalToNoiseInput::from_slices(
            &source,
            &high,
            &low,
            L2EhlersSignalToNoiseParams {
                smooth_period: Some(0),
            },
        );
        let err = l2_ehlers_signal_to_noise(&input).unwrap_err();
        assert!(matches!(
            err,
            L2EhlersSignalToNoiseError::InvalidSmoothPeriod { .. }
        ));
    }
}
