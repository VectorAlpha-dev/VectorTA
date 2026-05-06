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

use crate::indicators::atr::{AtrParams, AtrStream};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_uninit_f64, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::HashMap;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_LENGTH: usize = 10;
const DEFAULT_MULT: f64 = 2.0;
const DEFAULT_SMOOTH: usize = 72;
const OUTPUT_SCALE: f64 = 100.0;

#[derive(Debug, Clone)]
pub enum SuperTrendOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        source: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct SuperTrendOscillatorOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
    pub histogram: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct SuperTrendOscillatorParams {
    pub length: Option<usize>,
    pub mult: Option<f64>,
    pub smooth: Option<usize>,
}

impl Default for SuperTrendOscillatorParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            mult: Some(DEFAULT_MULT),
            smooth: Some(DEFAULT_SMOOTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SuperTrendOscillatorInput<'a> {
    pub data: SuperTrendOscillatorData<'a>,
    pub params: SuperTrendOscillatorParams,
}

impl<'a> SuperTrendOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: SuperTrendOscillatorParams,
    ) -> Self {
        Self {
            data: SuperTrendOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        source: &'a [f64],
        params: SuperTrendOscillatorParams,
    ) -> Self {
        Self {
            data: SuperTrendOscillatorData::Slices { high, low, source },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", SuperTrendOscillatorParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline]
    pub fn get_mult(&self) -> f64 {
        self.params.mult.unwrap_or(DEFAULT_MULT)
    }

    #[inline]
    pub fn get_smooth(&self) -> usize {
        self.params.smooth.unwrap_or(DEFAULT_SMOOTH)
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            SuperTrendOscillatorData::Candles { candles, source } => {
                let source = match *source {
                    "open" => candles.open.as_slice(),
                    "high" => candles.high.as_slice(),
                    "low" => candles.low.as_slice(),
                    "close" => candles.close.as_slice(),
                    "volume" => candles.volume.as_slice(),
                    "hl2" => candles.hl2.as_slice(),
                    "hlc3" => candles.hlc3.as_slice(),
                    "ohlc4" => candles.ohlc4.as_slice(),
                    "hlcc4" | "hlcc" => candles.hlcc4.as_slice(),
                    _ => source_type(candles, source),
                };
                (candles.high.as_slice(), candles.low.as_slice(), source)
            }
            SuperTrendOscillatorData::Slices { high, low, source } => (*high, *low, *source),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SuperTrendOscillatorBuilder {
    length: Option<usize>,
    mult: Option<f64>,
    smooth: Option<usize>,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for SuperTrendOscillatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            mult: None,
            smooth: None,
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SuperTrendOscillatorBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn length(mut self, value: usize) -> Self {
        self.length = Some(value);
        self
    }

    #[inline]
    pub fn mult(mut self, value: f64) -> Self {
        self.mult = Some(value);
        self
    }

    #[inline]
    pub fn smooth(mut self, value: usize) -> Self {
        self.smooth = Some(value);
        self
    }

    #[inline]
    pub fn source<S: Into<String>>(mut self, value: S) -> Self {
        self.source = Some(value.into());
        self
    }

    #[inline]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<SuperTrendOscillatorOutput, SuperTrendOscillatorError> {
        let input = SuperTrendOscillatorInput::from_candles(
            candles,
            self.source.as_deref().unwrap_or("close"),
            SuperTrendOscillatorParams {
                length: self.length,
                mult: self.mult,
                smooth: self.smooth,
            },
        );
        supertrend_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        source: &[f64],
    ) -> Result<SuperTrendOscillatorOutput, SuperTrendOscillatorError> {
        let input = SuperTrendOscillatorInput::from_slices(
            high,
            low,
            source,
            SuperTrendOscillatorParams {
                length: self.length,
                mult: self.mult,
                smooth: self.smooth,
            },
        );
        supertrend_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<SuperTrendOscillatorStream, SuperTrendOscillatorError> {
        SuperTrendOscillatorStream::try_new(SuperTrendOscillatorParams {
            length: self.length,
            mult: self.mult,
            smooth: self.smooth,
        })
    }
}

#[derive(Debug, Error)]
pub enum SuperTrendOscillatorError {
    #[error("supertrend_oscillator: Empty input data.")]
    EmptyInputData,
    #[error(
        "supertrend_oscillator: Input length mismatch: high={high}, low={low}, source={source_len}"
    )]
    DataLengthMismatch {
        high: usize,
        low: usize,
        source_len: usize,
    },
    #[error("supertrend_oscillator: All input values are invalid.")]
    AllValuesNaN,
    #[error("supertrend_oscillator: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("supertrend_oscillator: Invalid multiplier: {mult}")]
    InvalidMultiplier { mult: f64 },
    #[error("supertrend_oscillator: Invalid smooth: {smooth}")]
    InvalidSmooth { smooth: usize },
    #[error("supertrend_oscillator: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("supertrend_oscillator: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("supertrend_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("supertrend_oscillator: Invalid float range: start={start}, end={end}, step={step}")]
    InvalidFloatRange { start: f64, end: f64, step: f64 },
    #[error("supertrend_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn valid_bar(high: f64, low: f64, source: f64) -> bool {
    high.is_finite() && low.is_finite() && source.is_finite() && high >= low
}

#[inline(always)]
fn first_valid_bar(high: &[f64], low: &[f64], source: &[f64]) -> Option<usize> {
    (0..source.len()).find(|&i| valid_bar(high[i], low[i], source[i]))
}

#[inline(always)]
fn max_valid_run(high: &[f64], low: &[f64], source: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for i in 0..source.len() {
        if valid_bar(high[i], low[i], source[i]) {
            cur += 1;
            if cur > best {
                best = cur;
            }
        } else {
            cur = 0;
        }
    }
    best
}

#[inline(always)]
fn valid_bar_stats(high: &[f64], low: &[f64], source: &[f64]) -> (Option<usize>, usize) {
    let mut first = None;
    let mut best = 0usize;
    let mut cur = 0usize;
    for i in 0..source.len() {
        if valid_bar(high[i], low[i], source[i]) {
            if first.is_none() {
                first = Some(i);
            }
            cur += 1;
            if cur > best {
                best = cur;
            }
        } else {
            cur = 0;
        }
    }
    (first, best)
}

#[inline(always)]
fn normalized_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    return Kernel::Avx2;
                }
                if std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    return Kernel::Avx512;
                }
                Kernel::Scalar
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            {
                Kernel::Scalar
            }
        }
        other if other.is_batch() => other.to_non_batch(),
        other => other,
    }
}

#[inline(always)]
fn write_nan3(oscillator: &mut [f64], signal: &mut [f64], histogram: &mut [f64], index: usize) {
    oscillator[index] = f64::NAN;
    signal[index] = f64::NAN;
    histogram[index] = f64::NAN;
}

#[inline(always)]
fn write_oscillator_values(
    oscillator: &mut [f64],
    signal: &mut [f64],
    histogram: &mut [f64],
    index: usize,
    osc: f64,
    ama: f64,
    hist: f64,
) {
    oscillator[index] = osc * OUTPUT_SCALE;
    signal[index] = ama * OUTPUT_SCALE;
    histogram[index] = hist * OUTPUT_SCALE;
}

#[inline(always)]
fn clamp_unit(value: f64) -> f64 {
    value.clamp(-1.0, 1.0)
}

#[inline(always)]
fn warmup_end(first_valid: usize, length: usize) -> usize {
    first_valid.saturating_add(length.saturating_sub(1))
}

#[inline(always)]
fn supertrend_oscillator_compute_into(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    length: usize,
    mult: f64,
    smooth: usize,
    all_valid: bool,
    kernel: Kernel,
    out_oscillator: &mut [f64],
    out_signal: &mut [f64],
    out_histogram: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => supertrend_oscillator_row_fused(
                high,
                low,
                source,
                length,
                mult,
                smooth,
                all_valid,
                out_oscillator,
                out_signal,
                out_histogram,
            ),
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                supertrend_oscillator_row_fused(
                    high,
                    low,
                    source,
                    length,
                    mult,
                    smooth,
                    all_valid,
                    out_oscillator,
                    out_signal,
                    out_histogram,
                )
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => supertrend_oscillator_row_fused_avx2(
                high,
                low,
                source,
                length,
                mult,
                smooth,
                all_valid,
                out_oscillator,
                out_signal,
                out_histogram,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => supertrend_oscillator_row_fused_avx512(
                high,
                low,
                source,
                length,
                mult,
                smooth,
                all_valid,
                out_oscillator,
                out_signal,
                out_histogram,
            ),
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
unsafe fn supertrend_oscillator_row_fused(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    length: usize,
    mult: f64,
    smooth: usize,
    all_valid: bool,
    out_oscillator: &mut [f64],
    out_signal: &mut [f64],
    out_histogram: &mut [f64],
) {
    if all_valid {
        supertrend_oscillator_row_fused_all_valid(
            high,
            low,
            source,
            length,
            mult,
            smooth,
            out_oscillator,
            out_signal,
            out_histogram,
        );
    } else {
        supertrend_oscillator_row_fused_checked(
            high,
            low,
            source,
            length,
            mult,
            smooth,
            out_oscillator,
            out_signal,
            out_histogram,
        );
    }
}

#[inline(always)]
unsafe fn supertrend_oscillator_row_fused_all_valid(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    length: usize,
    mult: f64,
    smooth: usize,
    out_oscillator: &mut [f64],
    out_signal: &mut [f64],
    out_histogram: &mut [f64],
) {
    let len = source.len();
    let atr_alpha = 1.0 / (length as f64);
    let hist_alpha = 2.0 / (smooth as f64 + 1.0);
    let length_f64 = length as f64;

    let mut prev_close = f64::NAN;
    let mut atr = f64::NAN;
    let mut warm_sum = 0.0;
    let mut warm_count = 0usize;
    let mut seeded = false;

    let mut prev_source = f64::NAN;
    let mut prev_upper = f64::NAN;
    let mut prev_lower = f64::NAN;
    let mut prev_trend = 0.0;
    let mut ama_seeded = false;
    let mut hist_seeded = false;
    let mut ama_prev = 0.0;
    let mut hist_prev = 0.0;

    let h_ptr = high.as_ptr();
    let l_ptr = low.as_ptr();
    let s_ptr = source.as_ptr();

    let mut i = 0usize;
    while i < len {
        let hi = *h_ptr.add(i);
        let lo = *l_ptr.add(i);
        let src = *s_ptr.add(i);
        let true_range = if prev_close.is_nan() {
            hi - lo
        } else {
            let up = if hi > prev_close { hi } else { prev_close };
            let dn = if lo < prev_close { lo } else { prev_close };
            up - dn
        };
        prev_close = src;

        if !seeded {
            warm_sum += true_range;
            warm_count += 1;
            if warm_count == length {
                atr = warm_sum * atr_alpha;
                seeded = true;
            } else {
                write_nan3(out_oscillator, out_signal, out_histogram, i);
                prev_source = src;
                i += 1;
                continue;
            }
        } else {
            atr = atr_alpha.mul_add(true_range - atr, atr);
        }

        let mid = 0.5 * (hi + lo);
        let band = atr * mult;
        let up = mid + band;
        let dn = mid - band;

        let upper = if prev_source.is_finite() && prev_upper.is_finite() && prev_source < prev_upper
        {
            up.min(prev_upper)
        } else {
            up
        };
        let lower = if prev_source.is_finite() && prev_lower.is_finite() && prev_source > prev_lower
        {
            dn.max(prev_lower)
        } else {
            dn
        };

        let trend = if prev_upper.is_finite() && src > prev_upper {
            1.0
        } else if prev_lower.is_finite() && src < prev_lower {
            0.0
        } else {
            prev_trend
        };

        let supertrend = trend * lower + (1.0 - trend) * upper;
        let width = upper - lower;
        let osc = if width.is_finite() && width != 0.0 {
            clamp_unit((src - supertrend) / width)
        } else {
            0.0
        };
        let alpha = (osc * osc) / length_f64;
        let ama = if ama_seeded {
            ama_prev + alpha * (osc - ama_prev)
        } else {
            ama_seeded = true;
            osc
        };
        let diff = osc - ama;
        let hist = if hist_seeded {
            hist_prev + hist_alpha * (diff - hist_prev)
        } else {
            hist_seeded = true;
            diff
        };

        write_oscillator_values(out_oscillator, out_signal, out_histogram, i, osc, ama, hist);

        prev_source = src;
        prev_upper = upper;
        prev_lower = lower;
        prev_trend = trend;
        ama_prev = ama;
        hist_prev = hist;
        i += 1;
    }
}

#[inline(always)]
unsafe fn supertrend_oscillator_row_fused_checked(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    length: usize,
    mult: f64,
    smooth: usize,
    out_oscillator: &mut [f64],
    out_signal: &mut [f64],
    out_histogram: &mut [f64],
) {
    let len = source.len();
    let atr_alpha = 1.0 / (length as f64);
    let hist_alpha = 2.0 / (smooth as f64 + 1.0);
    let length_f64 = length as f64;

    let mut prev_close = f64::NAN;
    let mut atr = f64::NAN;
    let mut warm_sum = 0.0;
    let mut warm_count = 0usize;
    let mut seeded = false;

    let mut prev_source = f64::NAN;
    let mut prev_upper = f64::NAN;
    let mut prev_lower = f64::NAN;
    let mut prev_trend = 0.0;
    let mut ama_seeded = false;
    let mut hist_seeded = false;
    let mut ama_prev = 0.0;
    let mut hist_prev = 0.0;

    let h_ptr = high.as_ptr();
    let l_ptr = low.as_ptr();
    let s_ptr = source.as_ptr();

    let mut i = 0usize;
    while i < len {
        let hi = *h_ptr.add(i);
        let lo = *l_ptr.add(i);
        let src = *s_ptr.add(i);
        if !valid_bar(hi, lo, src) {
            write_nan3(out_oscillator, out_signal, out_histogram, i);
            prev_close = f64::NAN;
            atr = f64::NAN;
            warm_sum = 0.0;
            warm_count = 0;
            seeded = false;
            prev_source = f64::NAN;
            prev_upper = f64::NAN;
            prev_lower = f64::NAN;
            prev_trend = 0.0;
            ama_seeded = false;
            hist_seeded = false;
            ama_prev = 0.0;
            hist_prev = 0.0;
            i += 1;
            continue;
        }

        let true_range = if prev_close.is_nan() {
            hi - lo
        } else {
            let up = if hi > prev_close { hi } else { prev_close };
            let dn = if lo < prev_close { lo } else { prev_close };
            up - dn
        };
        prev_close = src;

        if !seeded {
            warm_sum += true_range;
            warm_count += 1;
            if warm_count == length {
                atr = warm_sum * atr_alpha;
                seeded = true;
            } else {
                write_nan3(out_oscillator, out_signal, out_histogram, i);
                prev_source = src;
                i += 1;
                continue;
            }
        } else {
            atr = atr_alpha.mul_add(true_range - atr, atr);
        }

        let mid = 0.5 * (hi + lo);
        let band = atr * mult;
        let up = mid + band;
        let dn = mid - band;

        let upper = if prev_source.is_finite() && prev_upper.is_finite() && prev_source < prev_upper
        {
            up.min(prev_upper)
        } else {
            up
        };
        let lower = if prev_source.is_finite() && prev_lower.is_finite() && prev_source > prev_lower
        {
            dn.max(prev_lower)
        } else {
            dn
        };

        let trend = if prev_upper.is_finite() && src > prev_upper {
            1.0
        } else if prev_lower.is_finite() && src < prev_lower {
            0.0
        } else {
            prev_trend
        };

        let supertrend = trend * lower + (1.0 - trend) * upper;
        let width = upper - lower;
        let osc = if width.is_finite() && width != 0.0 {
            clamp_unit((src - supertrend) / width)
        } else {
            0.0
        };
        let alpha = (osc * osc) / length_f64;
        let ama = if ama_seeded {
            ama_prev + alpha * (osc - ama_prev)
        } else {
            ama_seeded = true;
            osc
        };
        let diff = osc - ama;
        let hist = if hist_seeded {
            hist_prev + hist_alpha * (diff - hist_prev)
        } else {
            hist_seeded = true;
            diff
        };

        write_oscillator_values(out_oscillator, out_signal, out_histogram, i, osc, ama, hist);

        prev_source = src;
        prev_upper = upper;
        prev_lower = lower;
        prev_trend = trend;
        ama_prev = ama;
        hist_prev = hist;
        i += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn supertrend_oscillator_row_fused_avx2(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    length: usize,
    mult: f64,
    smooth: usize,
    all_valid: bool,
    out_oscillator: &mut [f64],
    out_signal: &mut [f64],
    out_histogram: &mut [f64],
) {
    supertrend_oscillator_row_fused(
        high,
        low,
        source,
        length,
        mult,
        smooth,
        all_valid,
        out_oscillator,
        out_signal,
        out_histogram,
    );
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn supertrend_oscillator_row_fused_avx512(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    length: usize,
    mult: f64,
    smooth: usize,
    all_valid: bool,
    out_oscillator: &mut [f64],
    out_signal: &mut [f64],
    out_histogram: &mut [f64],
) {
    supertrend_oscillator_row_fused(
        high,
        low,
        source,
        length,
        mult,
        smooth,
        all_valid,
        out_oscillator,
        out_signal,
        out_histogram,
    );
}

#[inline(always)]
fn validate_lengths(
    high: &[f64],
    low: &[f64],
    source: &[f64],
) -> Result<(), SuperTrendOscillatorError> {
    if high.is_empty() || low.is_empty() || source.is_empty() {
        return Err(SuperTrendOscillatorError::EmptyInputData);
    }
    if high.len() != low.len() || low.len() != source.len() {
        return Err(SuperTrendOscillatorError::DataLengthMismatch {
            high: high.len(),
            low: low.len(),
            source_len: source.len(),
        });
    }
    Ok(())
}

#[inline(always)]
fn validate_params(
    length: usize,
    mult: f64,
    smooth: usize,
    data_len: usize,
) -> Result<(), SuperTrendOscillatorError> {
    if length == 0 || length > data_len {
        return Err(SuperTrendOscillatorError::InvalidLength { length, data_len });
    }
    if !mult.is_finite() || mult <= 0.0 {
        return Err(SuperTrendOscillatorError::InvalidMultiplier { mult });
    }
    if smooth == 0 {
        return Err(SuperTrendOscillatorError::InvalidSmooth { smooth });
    }
    Ok(())
}

fn compute_atr_series(high: &[f64], low: &[f64], source: &[f64], length: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; source.len()];
    let mut stream = AtrStream::try_new(AtrParams {
        length: Some(length),
    })
    .expect("validated length");

    for i in 0..source.len() {
        if !valid_bar(high[i], low[i], source[i]) {
            stream = AtrStream::try_new(AtrParams {
                length: Some(length),
            })
            .expect("validated length");
            continue;
        }
        if let Some(atr) = stream.update(high[i], low[i], source[i]) {
            out[i] = atr;
        }
    }

    out
}

#[inline(always)]
fn supertrend_oscillator_row_scalar(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    length: usize,
    mult: f64,
    smooth: usize,
    atr_values: &[f64],
    out_oscillator: &mut [f64],
    out_signal: &mut [f64],
    out_histogram: &mut [f64],
) {
    let hist_alpha = 2.0 / (smooth as f64 + 1.0);
    let mut prev_source = f64::NAN;
    let mut prev_upper = f64::NAN;
    let mut prev_lower = f64::NAN;
    let mut prev_trend = 0.0;
    let mut ama_prev: Option<f64> = None;
    let mut hist_prev: Option<f64> = None;
    let length_f64 = length as f64;

    for i in 0..source.len() {
        let src = source[i];
        if !valid_bar(high[i], low[i], src) {
            out_oscillator[i] = f64::NAN;
            out_signal[i] = f64::NAN;
            out_histogram[i] = f64::NAN;
            prev_source = f64::NAN;
            prev_upper = f64::NAN;
            prev_lower = f64::NAN;
            prev_trend = 0.0;
            ama_prev = None;
            hist_prev = None;
            continue;
        }

        if !atr_values[i].is_finite() {
            out_oscillator[i] = f64::NAN;
            out_signal[i] = f64::NAN;
            out_histogram[i] = f64::NAN;
            prev_source = src;
            continue;
        }

        let mid = 0.5 * (high[i] + low[i]);
        let band = atr_values[i] * mult;
        let up = mid + band;
        let dn = mid - band;

        let upper = if prev_source.is_finite() && prev_upper.is_finite() && prev_source < prev_upper
        {
            up.min(prev_upper)
        } else {
            up
        };
        let lower = if prev_source.is_finite() && prev_lower.is_finite() && prev_source > prev_lower
        {
            dn.max(prev_lower)
        } else {
            dn
        };

        let trend = if prev_upper.is_finite() && src > prev_upper {
            1.0
        } else if prev_lower.is_finite() && src < prev_lower {
            0.0
        } else {
            prev_trend
        };

        let supertrend = trend * lower + (1.0 - trend) * upper;
        let width = upper - lower;
        let osc = if width.is_finite() && width != 0.0 {
            clamp_unit((src - supertrend) / width)
        } else {
            0.0
        };
        let alpha = (osc * osc) / length_f64;
        let ama = match ama_prev {
            Some(prev) => prev + alpha * (osc - prev),
            None => osc,
        };
        let diff = osc - ama;
        let hist = match hist_prev {
            Some(prev) => prev + hist_alpha * (diff - prev),
            None => diff,
        };

        out_oscillator[i] = osc * OUTPUT_SCALE;
        out_signal[i] = ama * OUTPUT_SCALE;
        out_histogram[i] = hist * OUTPUT_SCALE;

        prev_source = src;
        prev_upper = upper;
        prev_lower = lower;
        prev_trend = trend;
        ama_prev = Some(ama);
        hist_prev = Some(hist);
    }
}

fn supertrend_oscillator_prepare<'a>(
    input: &'a SuperTrendOscillatorInput<'a>,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        usize,
        f64,
        usize,
        usize,
        bool,
        Kernel,
    ),
    SuperTrendOscillatorError,
> {
    let (high, low, source) = input.as_refs();
    validate_lengths(high, low, source)?;

    let length = input.get_length();
    let mult = input.get_mult();
    let smooth = input.get_smooth();
    validate_params(length, mult, smooth, source.len())?;

    let (first_valid, max_run) = valid_bar_stats(high, low, source);
    let first_valid = first_valid.ok_or(SuperTrendOscillatorError::AllValuesNaN)?;
    if max_run < length {
        return Err(SuperTrendOscillatorError::NotEnoughValidData {
            needed: length,
            valid: max_run,
        });
    }

    let all_valid = max_run == source.len();

    Ok((
        high,
        low,
        source,
        length,
        mult,
        smooth,
        first_valid,
        all_valid,
        normalized_kernel(kernel),
    ))
}

#[inline]
pub fn supertrend_oscillator(
    input: &SuperTrendOscillatorInput,
) -> Result<SuperTrendOscillatorOutput, SuperTrendOscillatorError> {
    supertrend_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn supertrend_oscillator_with_kernel(
    input: &SuperTrendOscillatorInput,
    kernel: Kernel,
) -> Result<SuperTrendOscillatorOutput, SuperTrendOscillatorError> {
    let (high, low, source, length, mult, smooth, _first_valid, all_valid, chosen) =
        supertrend_oscillator_prepare(input, kernel)?;

    let len = source.len();
    let mut oscillator = alloc_uninit_f64(len);
    let mut signal = alloc_uninit_f64(len);
    let mut histogram = alloc_uninit_f64(len);

    supertrend_oscillator_compute_into(
        high,
        low,
        source,
        length,
        mult,
        smooth,
        all_valid,
        chosen,
        &mut oscillator,
        &mut signal,
        &mut histogram,
    );

    Ok(SuperTrendOscillatorOutput {
        oscillator,
        signal,
        histogram,
    })
}

#[inline]
pub fn supertrend_oscillator_into_slice(
    out_oscillator: &mut [f64],
    out_signal: &mut [f64],
    out_histogram: &mut [f64],
    input: &SuperTrendOscillatorInput,
    kernel: Kernel,
) -> Result<(), SuperTrendOscillatorError> {
    let (high, low, source, length, mult, smooth, _first_valid, all_valid, chosen) =
        supertrend_oscillator_prepare(input, kernel)?;
    let len = source.len();
    if out_oscillator.len() != len || out_signal.len() != len || out_histogram.len() != len {
        return Err(SuperTrendOscillatorError::OutputLengthMismatch {
            expected: len,
            got: out_oscillator
                .len()
                .max(out_signal.len())
                .max(out_histogram.len()),
        });
    }

    supertrend_oscillator_compute_into(
        high,
        low,
        source,
        length,
        mult,
        smooth,
        all_valid,
        chosen,
        out_oscillator,
        out_signal,
        out_histogram,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn supertrend_oscillator_into(
    input: &SuperTrendOscillatorInput,
    out_oscillator: &mut [f64],
    out_signal: &mut [f64],
    out_histogram: &mut [f64],
) -> Result<(), SuperTrendOscillatorError> {
    supertrend_oscillator_into_slice(
        out_oscillator,
        out_signal,
        out_histogram,
        input,
        Kernel::Auto,
    )
}

#[derive(Clone, Debug)]
pub struct SuperTrendOscillatorStream {
    length: usize,
    mult: f64,
    hist_alpha: f64,
    atr_stream: AtrStream,
    prev_source: f64,
    prev_upper: f64,
    prev_lower: f64,
    prev_trend: f64,
    ama_prev: Option<f64>,
    hist_prev: Option<f64>,
}

impl SuperTrendOscillatorStream {
    #[inline]
    pub fn try_new(params: SuperTrendOscillatorParams) -> Result<Self, SuperTrendOscillatorError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let mult = params.mult.unwrap_or(DEFAULT_MULT);
        let smooth = params.smooth.unwrap_or(DEFAULT_SMOOTH);
        validate_params(length, mult, smooth, length)?;
        Ok(Self {
            length,
            mult,
            hist_alpha: 2.0 / (smooth as f64 + 1.0),
            atr_stream: AtrStream::try_new(AtrParams {
                length: Some(length),
            })
            .expect("validated length"),
            prev_source: f64::NAN,
            prev_upper: f64::NAN,
            prev_lower: f64::NAN,
            prev_trend: 0.0,
            ama_prev: None,
            hist_prev: None,
        })
    }

    #[inline]
    fn reset(&mut self) {
        self.atr_stream = AtrStream::try_new(AtrParams {
            length: Some(self.length),
        })
        .expect("validated length");
        self.prev_source = f64::NAN;
        self.prev_upper = f64::NAN;
        self.prev_lower = f64::NAN;
        self.prev_trend = 0.0;
        self.ama_prev = None;
        self.hist_prev = None;
    }

    #[inline]
    pub fn update(&mut self, high: f64, low: f64, source: f64) -> Option<(f64, f64, f64)> {
        if !valid_bar(high, low, source) {
            self.reset();
            return None;
        }

        let atr = match self.atr_stream.update(high, low, source) {
            Some(value) => value,
            None => {
                self.prev_source = source;
                return None;
            }
        };

        let mid = 0.5 * (high + low);
        let up = mid + atr * self.mult;
        let dn = mid - atr * self.mult;

        let upper = if self.prev_source.is_finite()
            && self.prev_upper.is_finite()
            && self.prev_source < self.prev_upper
        {
            up.min(self.prev_upper)
        } else {
            up
        };
        let lower = if self.prev_source.is_finite()
            && self.prev_lower.is_finite()
            && self.prev_source > self.prev_lower
        {
            dn.max(self.prev_lower)
        } else {
            dn
        };

        let trend = if self.prev_upper.is_finite() && source > self.prev_upper {
            1.0
        } else if self.prev_lower.is_finite() && source < self.prev_lower {
            0.0
        } else {
            self.prev_trend
        };

        let supertrend = trend * lower + (1.0 - trend) * upper;
        let width = upper - lower;
        let osc = if width.is_finite() && width != 0.0 {
            clamp_unit((source - supertrend) / width)
        } else {
            0.0
        };
        let alpha = (osc * osc) / self.length as f64;
        let ama = match self.ama_prev {
            Some(prev) => prev + alpha * (osc - prev),
            None => osc,
        };
        let diff = osc - ama;
        let hist = match self.hist_prev {
            Some(prev) => prev + self.hist_alpha * (diff - prev),
            None => diff,
        };

        self.prev_source = source;
        self.prev_upper = upper;
        self.prev_lower = lower;
        self.prev_trend = trend;
        self.ama_prev = Some(ama);
        self.hist_prev = Some(hist);

        Some((osc * OUTPUT_SCALE, ama * OUTPUT_SCALE, hist * OUTPUT_SCALE))
    }
}

#[derive(Debug, Clone)]
pub struct SuperTrendOscillatorBatchOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
    pub histogram: Vec<f64>,
    pub combos: Vec<SuperTrendOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl SuperTrendOscillatorBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &SuperTrendOscillatorParams) -> Option<usize> {
        self.combos.iter().position(|p| {
            p.length == params.length && p.mult == params.mult && p.smooth == params.smooth
        })
    }
}

#[derive(Debug, Clone)]
pub struct SuperTrendOscillatorBatchRange {
    pub length: (usize, usize, usize),
    pub mult: (f64, f64, f64),
    pub smooth: (usize, usize, usize),
}

impl Default for SuperTrendOscillatorBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            mult: (DEFAULT_MULT, DEFAULT_MULT, 0.0),
            smooth: (DEFAULT_SMOOTH, DEFAULT_SMOOTH, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SuperTrendOscillatorBatchBuilder {
    range: SuperTrendOscillatorBatchRange,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for SuperTrendOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: SuperTrendOscillatorBatchRange::default(),
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SuperTrendOscillatorBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline]
    pub fn source<S: Into<String>>(mut self, value: S) -> Self {
        self.source = Some(value.into());
        self
    }

    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn length_static(mut self, value: usize) -> Self {
        self.range.length = (value, value, 0);
        self
    }

    #[inline]
    pub fn mult_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.mult = (start, end, step);
        self
    }

    #[inline]
    pub fn mult_static(mut self, value: f64) -> Self {
        self.range.mult = (value, value, 0.0);
        self
    }

    #[inline]
    pub fn smooth_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smooth = (start, end, step);
        self
    }

    #[inline]
    pub fn smooth_static(mut self, value: usize) -> Self {
        self.range.smooth = (value, value, 0);
        self
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
    ) -> Result<SuperTrendOscillatorBatchOutput, SuperTrendOscillatorError> {
        let source = source_type(candles, self.source.as_deref().unwrap_or("close"));
        supertrend_oscillator_batch_with_kernel(
            candles.high.as_slice(),
            candles.low.as_slice(),
            source,
            &self.range,
            self.kernel,
        )
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        source: &[f64],
    ) -> Result<SuperTrendOscillatorBatchOutput, SuperTrendOscillatorError> {
        supertrend_oscillator_batch_with_kernel(high, low, source, &self.range, self.kernel)
    }
}

#[inline]
pub fn expand_grid_supertrend_oscillator(
    range: &SuperTrendOscillatorBatchRange,
) -> Result<Vec<SuperTrendOscillatorParams>, SuperTrendOscillatorError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, SuperTrendOscillatorError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start <= end {
            let mut out = Vec::new();
            let mut x = start;
            while x <= end {
                out.push(x);
                match x.checked_add(step.max(1)) {
                    Some(next) if next > x => x = next,
                    _ => break,
                }
            }
            if out.is_empty() {
                return Err(SuperTrendOscillatorError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            return Ok(out);
        }

        let mut out = Vec::new();
        let mut x = start;
        let step = step.max(1);
        while x >= end {
            out.push(x);
            if x == end {
                break;
            }
            let next = x.saturating_sub(step);
            if next == x || next < end {
                break;
            }
            x = next;
        }
        if out.is_empty() {
            return Err(SuperTrendOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        Ok(out)
    }

    fn axis_f64(
        (start, end, step): (f64, f64, f64),
    ) -> Result<Vec<f64>, SuperTrendOscillatorError> {
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(SuperTrendOscillatorError::InvalidFloatRange { start, end, step });
        }
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return Ok(vec![start]);
        }
        let step = step.abs();
        let mut out = Vec::new();
        if start <= end {
            let mut x = start;
            while x <= end + 1e-12 {
                out.push(x);
                x += step;
            }
        } else {
            let mut x = start;
            while x + 1e-12 >= end {
                out.push(x);
                x -= step;
            }
        }
        if out.is_empty() {
            return Err(SuperTrendOscillatorError::InvalidFloatRange { start, end, step });
        }
        Ok(out)
    }

    let lengths = axis_usize(range.length)?;
    let mults = axis_f64(range.mult)?;
    let smooths = axis_usize(range.smooth)?;

    let cap = lengths
        .len()
        .checked_mul(mults.len())
        .and_then(|value| value.checked_mul(smooths.len()))
        .ok_or(SuperTrendOscillatorError::InvalidRange {
            start: range.length.0.to_string(),
            end: range.length.1.to_string(),
            step: range.length.2.to_string(),
        })?;

    let mut out = Vec::with_capacity(cap);
    for &length in &lengths {
        for &mult in &mults {
            for &smooth in &smooths {
                out.push(SuperTrendOscillatorParams {
                    length: Some(length),
                    mult: Some(mult),
                    smooth: Some(smooth),
                });
            }
        }
    }
    Ok(out)
}

#[inline]
pub fn supertrend_oscillator_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &SuperTrendOscillatorBatchRange,
    kernel: Kernel,
) -> Result<SuperTrendOscillatorBatchOutput, SuperTrendOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(SuperTrendOscillatorError::InvalidKernelForBatch(other)),
    };
    supertrend_oscillator_batch_par_slice(high, low, source, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn supertrend_oscillator_batch_slice(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &SuperTrendOscillatorBatchRange,
    kernel: Kernel,
) -> Result<SuperTrendOscillatorBatchOutput, SuperTrendOscillatorError> {
    supertrend_oscillator_batch_inner(high, low, source, sweep, kernel, false)
}

#[inline]
pub fn supertrend_oscillator_batch_par_slice(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &SuperTrendOscillatorBatchRange,
    kernel: Kernel,
) -> Result<SuperTrendOscillatorBatchOutput, SuperTrendOscillatorError> {
    supertrend_oscillator_batch_inner(high, low, source, sweep, kernel, true)
}

fn supertrend_oscillator_batch_inner(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &SuperTrendOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<SuperTrendOscillatorBatchOutput, SuperTrendOscillatorError> {
    validate_lengths(high, low, source)?;
    let combos = expand_grid_supertrend_oscillator(sweep)?;
    let first_valid =
        first_valid_bar(high, low, source).ok_or(SuperTrendOscillatorError::AllValuesNaN)?;
    let max_run = max_valid_run(high, low, source);
    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(DEFAULT_LENGTH))
        .max()
        .unwrap_or(DEFAULT_LENGTH);
    if max_run < max_length {
        return Err(SuperTrendOscillatorError::NotEnoughValidData {
            needed: max_length,
            valid: max_run,
        });
    }
    for params in &combos {
        validate_params(
            params.length.unwrap_or(DEFAULT_LENGTH),
            params.mult.unwrap_or(DEFAULT_MULT),
            params.smooth.unwrap_or(DEFAULT_SMOOTH),
            source.len(),
        )?;
    }

    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(SuperTrendOscillatorError::OutputLengthMismatch {
            expected: usize::MAX,
            got: 0,
        })?;

    let mut oscillator_matrix = make_uninit_matrix(rows, cols);
    let mut signal_matrix = make_uninit_matrix(rows, cols);
    let mut histogram_matrix = make_uninit_matrix(rows, cols);

    let warmups: Vec<usize> = combos
        .iter()
        .map(|params| warmup_end(first_valid, params.length.unwrap_or(DEFAULT_LENGTH)))
        .collect();
    init_matrix_prefixes(&mut oscillator_matrix, cols, &warmups);
    init_matrix_prefixes(&mut signal_matrix, cols, &warmups);
    init_matrix_prefixes(&mut histogram_matrix, cols, &warmups);

    let mut oscillator_guard = ManuallyDrop::new(oscillator_matrix);
    let mut signal_guard = ManuallyDrop::new(signal_matrix);
    let mut histogram_guard = ManuallyDrop::new(histogram_matrix);

    let oscillator_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(oscillator_guard.as_mut_ptr(), oscillator_guard.len())
    };
    let signal_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(signal_guard.as_mut_ptr(), signal_guard.len()) };
    let histogram_mu: &mut [MaybeUninit<f64>] = unsafe {
        std::slice::from_raw_parts_mut(histogram_guard.as_mut_ptr(), histogram_guard.len())
    };

    let mut atr_cache: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut lengths: Vec<usize> = combos
        .iter()
        .map(|params| params.length.unwrap_or(DEFAULT_LENGTH))
        .collect();
    lengths.sort_unstable();
    lengths.dedup();
    for length in lengths {
        atr_cache.insert(length, compute_atr_series(high, low, source, length));
    }

    let do_row = |row: usize,
                  row_oscillator: &mut [MaybeUninit<f64>],
                  row_signal: &mut [MaybeUninit<f64>],
                  row_histogram: &mut [MaybeUninit<f64>]| {
        let params = &combos[row];
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let mult = params.mult.unwrap_or(DEFAULT_MULT);
        let smooth = params.smooth.unwrap_or(DEFAULT_SMOOTH);
        let atr_values = atr_cache.get(&length).expect("cached atr");

        let dst_oscillator = unsafe {
            std::slice::from_raw_parts_mut(row_oscillator.as_mut_ptr() as *mut f64, cols)
        };
        let dst_signal =
            unsafe { std::slice::from_raw_parts_mut(row_signal.as_mut_ptr() as *mut f64, cols) };
        let dst_histogram =
            unsafe { std::slice::from_raw_parts_mut(row_histogram.as_mut_ptr() as *mut f64, cols) };

        supertrend_oscillator_row_scalar(
            high,
            low,
            source,
            length,
            mult,
            smooth,
            atr_values,
            dst_oscillator,
            dst_signal,
            dst_histogram,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        oscillator_mu
            .par_chunks_mut(cols)
            .zip(signal_mu.par_chunks_mut(cols))
            .zip(histogram_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, ((row_oscillator, row_signal), row_histogram))| {
                do_row(row, row_oscillator, row_signal, row_histogram)
            });

        #[cfg(target_arch = "wasm32")]
        for (row, ((row_oscillator, row_signal), row_histogram)) in oscillator_mu
            .chunks_mut(cols)
            .zip(signal_mu.chunks_mut(cols))
            .zip(histogram_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_oscillator, row_signal, row_histogram);
        }
    } else {
        for (row, ((row_oscillator, row_signal), row_histogram)) in oscillator_mu
            .chunks_mut(cols)
            .zip(signal_mu.chunks_mut(cols))
            .zip(histogram_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_oscillator, row_signal, row_histogram);
        }
    }

    let oscillator = unsafe {
        Vec::from_raw_parts(
            oscillator_guard.as_mut_ptr() as *mut f64,
            total,
            oscillator_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            total,
            signal_guard.capacity(),
        )
    };
    let histogram = unsafe {
        Vec::from_raw_parts(
            histogram_guard.as_mut_ptr() as *mut f64,
            total,
            histogram_guard.capacity(),
        )
    };

    Ok(SuperTrendOscillatorBatchOutput {
        oscillator,
        signal,
        histogram,
        combos,
        rows,
        cols,
    })
}

fn supertrend_oscillator_batch_inner_into(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    sweep: &SuperTrendOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_oscillator: &mut [f64],
    out_signal: &mut [f64],
    out_histogram: &mut [f64],
) -> Result<Vec<SuperTrendOscillatorParams>, SuperTrendOscillatorError> {
    validate_lengths(high, low, source)?;
    let combos = expand_grid_supertrend_oscillator(sweep)?;
    let max_run = max_valid_run(high, low, source);
    let max_length = combos
        .iter()
        .map(|params| params.length.unwrap_or(DEFAULT_LENGTH))
        .max()
        .unwrap_or(DEFAULT_LENGTH);
    if max_run < max_length {
        return Err(SuperTrendOscillatorError::NotEnoughValidData {
            needed: max_length,
            valid: max_run,
        });
    }

    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(SuperTrendOscillatorError::OutputLengthMismatch {
            expected: usize::MAX,
            got: 0,
        })?;
    if out_oscillator.len() != total || out_signal.len() != total || out_histogram.len() != total {
        return Err(SuperTrendOscillatorError::OutputLengthMismatch {
            expected: total,
            got: out_oscillator
                .len()
                .max(out_signal.len())
                .max(out_histogram.len()),
        });
    }

    let mut atr_cache: HashMap<usize, Vec<f64>> = HashMap::new();
    for params in &combos {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        validate_params(
            length,
            params.mult.unwrap_or(DEFAULT_MULT),
            params.smooth.unwrap_or(DEFAULT_SMOOTH),
            cols,
        )?;
        atr_cache
            .entry(length)
            .or_insert_with(|| compute_atr_series(high, low, source, length));
    }

    let _ = kernel;
    let do_row = |row: usize,
                  dst_oscillator: &mut [f64],
                  dst_signal: &mut [f64],
                  dst_histogram: &mut [f64]| {
        let params = &combos[row];
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let mult = params.mult.unwrap_or(DEFAULT_MULT);
        let smooth = params.smooth.unwrap_or(DEFAULT_SMOOTH);
        let atr_values = atr_cache.get(&length).expect("cached atr");

        supertrend_oscillator_row_scalar(
            high,
            low,
            source,
            length,
            mult,
            smooth,
            atr_values,
            dst_oscillator,
            dst_signal,
            dst_histogram,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_oscillator
            .par_chunks_mut(cols)
            .zip(out_signal.par_chunks_mut(cols))
            .zip(out_histogram.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, ((dst_oscillator, dst_signal), dst_histogram))| {
                do_row(row, dst_oscillator, dst_signal, dst_histogram)
            });

        #[cfg(target_arch = "wasm32")]
        for (row, ((dst_oscillator, dst_signal), dst_histogram)) in out_oscillator
            .chunks_mut(cols)
            .zip(out_signal.chunks_mut(cols))
            .zip(out_histogram.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_oscillator, dst_signal, dst_histogram);
        }
    } else {
        for (row, ((dst_oscillator, dst_signal), dst_histogram)) in out_oscillator
            .chunks_mut(cols)
            .zip(out_signal.chunks_mut(cols))
            .zip(out_histogram.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_oscillator, dst_signal, dst_histogram);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "supertrend_oscillator")]
#[pyo3(signature = (high, low, source, length=DEFAULT_LENGTH, mult=DEFAULT_MULT, smooth=DEFAULT_SMOOTH, kernel=None))]
pub fn supertrend_oscillator_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    source: PyReadonlyArray1<'py, f64>,
    length: usize,
    mult: f64,
    smooth: usize,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let source = source.as_slice()?;
    let input = SuperTrendOscillatorInput::from_slices(
        high,
        low,
        source,
        SuperTrendOscillatorParams {
            length: Some(length),
            mult: Some(mult),
            smooth: Some(smooth),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| supertrend_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.oscillator.into_pyarray(py),
        out.signal.into_pyarray(py),
        out.histogram.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "SuperTrendOscillatorStream")]
pub struct SuperTrendOscillatorStreamPy {
    stream: SuperTrendOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SuperTrendOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, mult=DEFAULT_MULT, smooth=DEFAULT_SMOOTH))]
    fn new(length: usize, mult: f64, smooth: usize) -> PyResult<Self> {
        let stream = SuperTrendOscillatorStream::try_new(SuperTrendOscillatorParams {
            length: Some(length),
            mult: Some(mult),
            smooth: Some(smooth),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, source: f64) -> Option<(f64, f64, f64)> {
        self.stream.update(high, low, source)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "supertrend_oscillator_batch")]
#[pyo3(signature = (high, low, source, length_range, mult_range, smooth_range, kernel=None))]
pub fn supertrend_oscillator_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    source: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    mult_range: (f64, f64, f64),
    smooth_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let source = source.as_slice()?;
    let sweep = SuperTrendOscillatorBatchRange {
        length: length_range,
        mult: mult_range,
        smooth: smooth_range,
    };
    let combos = expand_grid_supertrend_oscillator(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let oscillator_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let histogram_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_oscillator = unsafe { oscillator_arr.as_slice_mut()? };
    let out_signal = unsafe { signal_arr.as_slice_mut()? };
    let out_histogram = unsafe { histogram_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        supertrend_oscillator_batch_inner_into(
            high,
            low,
            source,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_oscillator,
            out_signal,
            out_histogram,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let lengths: Vec<usize> = combos
        .iter()
        .map(|params| params.length.unwrap_or(DEFAULT_LENGTH))
        .collect();
    let mults: Vec<f64> = combos
        .iter()
        .map(|params| params.mult.unwrap_or(DEFAULT_MULT))
        .collect();
    let smooths: Vec<usize> = combos
        .iter()
        .map(|params| params.smooth.unwrap_or(DEFAULT_SMOOTH))
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("oscillator", oscillator_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item("histogram", histogram_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("lengths", lengths.into_pyarray(py))?;
    dict.set_item("mults", mults.into_pyarray(py))?;
    dict.set_item("smooths", smooths.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_supertrend_oscillator_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(supertrend_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(supertrend_oscillator_batch_py, m)?)?;
    m.add_class::<SuperTrendOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SuperTrendOscillatorJsOutput {
    oscillator: Vec<f64>,
    signal: Vec<f64>,
    histogram: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SuperTrendOscillatorBatchConfig {
    length_range: Vec<usize>,
    mult_range: Vec<f64>,
    smooth_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SuperTrendOscillatorBatchJsOutput {
    oscillator: Vec<f64>,
    signal: Vec<f64>,
    histogram: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<SuperTrendOscillatorParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "supertrend_oscillator")]
pub fn supertrend_oscillator_js(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    length: usize,
    mult: f64,
    smooth: usize,
) -> Result<JsValue, JsValue> {
    let input = SuperTrendOscillatorInput::from_slices(
        high,
        low,
        source,
        SuperTrendOscillatorParams {
            length: Some(length),
            mult: Some(mult),
            smooth: Some(smooth),
        },
    );
    let out = supertrend_oscillator(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&SuperTrendOscillatorJsOutput {
        oscillator: out.oscillator,
        signal: out.signal,
        histogram: out.histogram,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn supertrend_oscillator_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    source_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    mult: f64,
    smooth: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null() || low_ptr.is_null() || source_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to supertrend_oscillator_into",
        ));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let source = std::slice::from_raw_parts(source_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 3);
        let (out_oscillator, rest) = out.split_at_mut(len);
        let (out_signal, out_histogram) = rest.split_at_mut(len);
        let input = SuperTrendOscillatorInput::from_slices(
            high,
            low,
            source,
            SuperTrendOscillatorParams {
                length: Some(length),
                mult: Some(mult),
                smooth: Some(smooth),
            },
        );
        supertrend_oscillator_into_slice(
            out_oscillator,
            out_signal,
            out_histogram,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "supertrend_oscillator_into_host")]
pub fn supertrend_oscillator_into_host(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    out_ptr: *mut f64,
    length: usize,
    mult: f64,
    smooth: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to supertrend_oscillator_into_host",
        ));
    }

    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, source.len() * 3);
        let (out_oscillator, rest) = out.split_at_mut(source.len());
        let (out_signal, out_histogram) = rest.split_at_mut(source.len());
        let input = SuperTrendOscillatorInput::from_slices(
            high,
            low,
            source,
            SuperTrendOscillatorParams {
                length: Some(length),
                mult: Some(mult),
                smooth: Some(smooth),
            },
        );
        supertrend_oscillator_into_slice(
            out_oscillator,
            out_signal,
            out_histogram,
            &input,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn supertrend_oscillator_alloc(len: usize) -> *mut f64 {
    let mut buf = vec![0.0_f64; len * 3];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn supertrend_oscillator_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 3);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "supertrend_oscillator_batch")]
pub fn supertrend_oscillator_batch_js(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: SuperTrendOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3
        || config.mult_range.len() != 3
        || config.smooth_range.len() != 3
    {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }

    let sweep = SuperTrendOscillatorBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
        mult: (
            config.mult_range[0],
            config.mult_range[1],
            config.mult_range[2],
        ),
        smooth: (
            config.smooth_range[0],
            config.smooth_range[1],
            config.smooth_range[2],
        ),
    };
    let batch = supertrend_oscillator_batch_slice(high, low, source, &sweep, Kernel::Scalar)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&SuperTrendOscillatorBatchJsOutput {
        oscillator: batch.oscillator,
        signal: batch.signal,
        histogram: batch.histogram,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn supertrend_oscillator_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    source_ptr: *const f64,
    oscillator_ptr: *mut f64,
    signal_ptr: *mut f64,
    histogram_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    mult_start: f64,
    mult_end: f64,
    mult_step: f64,
    smooth_start: usize,
    smooth_end: usize,
    smooth_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || source_ptr.is_null()
        || oscillator_ptr.is_null()
        || signal_ptr.is_null()
        || histogram_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to supertrend_oscillator_batch_into",
        ));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let source = std::slice::from_raw_parts(source_ptr, len);
        let sweep = SuperTrendOscillatorBatchRange {
            length: (length_start, length_end, length_step),
            mult: (mult_start, mult_end, mult_step),
            smooth: (smooth_start, smooth_end, smooth_step),
        };
        let combos = expand_grid_supertrend_oscillator(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let oscillator = std::slice::from_raw_parts_mut(oscillator_ptr, total);
        let signal = std::slice::from_raw_parts_mut(signal_ptr, total);
        let histogram = std::slice::from_raw_parts_mut(histogram_ptr, total);
        supertrend_oscillator_batch_inner_into(
            high,
            low,
            source,
            &sweep,
            Kernel::Scalar,
            false,
            oscillator,
            signal,
            histogram,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn supertrend_oscillator_output_into_js(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    length: usize,
    mult: f64,
    smooth: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = supertrend_oscillator_js(high, low, source, length, mult, smooth)?;
    crate::write_wasm_object_f64_outputs("supertrend_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn supertrend_oscillator_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    source: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = supertrend_oscillator_batch_js(high, low, source, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "supertrend_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::dispatch::{
        compute_cpu_batch, IndicatorBatchRequest, IndicatorDataRef, IndicatorParamSet, ParamKV,
        ParamValue,
    };

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            let lhs = a[i];
            let rhs = b[i];
            if lhs.is_nan() || rhs.is_nan() {
                assert!(
                    lhs.is_nan() && rhs.is_nan(),
                    "nan mismatch at {i}: {lhs} vs {rhs}"
                );
            } else {
                assert!(
                    (lhs - rhs).abs() <= tol,
                    "mismatch at {i}: {lhs} vs {rhs} with tol {tol}"
                );
            }
        }
    }

    fn sample_hls(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut source = Vec::with_capacity(len);

        for i in 0..len {
            let base = 100.0 + i as f64 * 0.21 + (i as f64 * 0.17).sin() * 2.0;
            let spread = 1.5 + (i as f64 * 0.11).cos().abs() * 1.25;
            let src = base + (i as f64 * 0.07).cos() * 0.6;
            high.push(base + spread);
            low.push(base - spread);
            source.push(src);
        }

        (high, low, source)
    }

    fn check_output_contract(kernel: Kernel) {
        let (high, low, source) = sample_hls(192);
        let input = SuperTrendOscillatorInput::from_slices(
            &high,
            &low,
            &source,
            SuperTrendOscillatorParams {
                length: Some(10),
                mult: Some(2.0),
                smooth: Some(72),
            },
        );
        let out = supertrend_oscillator_with_kernel(&input, kernel).expect("indicator");
        assert_eq!(out.oscillator.len(), source.len());
        assert_eq!(out.signal.len(), source.len());
        assert_eq!(out.histogram.len(), source.len());
        assert!(out.oscillator[..9].iter().all(|v| v.is_nan()));
        assert!(out.signal[..9].iter().all(|v| v.is_nan()));
        assert!(out.histogram[..9].iter().all(|v| v.is_nan()));
        assert!(out.oscillator[9..].iter().any(|v| v.is_finite()));
        assert!(out.signal[9..].iter().any(|v| v.is_finite()));
        assert!(out.histogram[9..].iter().any(|v| v.is_finite()));
    }

    fn check_into_matches_api(kernel: Kernel) {
        let (high, low, source) = sample_hls(224);
        let input = SuperTrendOscillatorInput::from_slices(
            &high,
            &low,
            &source,
            SuperTrendOscillatorParams {
                length: Some(11),
                mult: Some(2.5),
                smooth: Some(20),
            },
        );
        let baseline = supertrend_oscillator_with_kernel(&input, kernel).expect("baseline");
        let mut oscillator = vec![0.0; source.len()];
        let mut signal = vec![0.0; source.len()];
        let mut histogram = vec![0.0; source.len()];
        supertrend_oscillator_into_slice(
            &mut oscillator,
            &mut signal,
            &mut histogram,
            &input,
            kernel,
        )
        .expect("into");

        assert_close(&baseline.oscillator, &oscillator, 1e-12);
        assert_close(&baseline.signal, &signal, 1e-12);
        assert_close(&baseline.histogram, &histogram, 1e-12);
    }

    fn check_stream_matches_batch() {
        let (high, low, source) = sample_hls(200);
        let input = SuperTrendOscillatorInput::from_slices(
            &high,
            &low,
            &source,
            SuperTrendOscillatorParams {
                length: Some(12),
                mult: Some(1.75),
                smooth: Some(18),
            },
        );
        let batch = supertrend_oscillator(&input).expect("batch");
        let mut stream = SuperTrendOscillatorStream::try_new(SuperTrendOscillatorParams {
            length: Some(12),
            mult: Some(1.75),
            smooth: Some(18),
        })
        .expect("stream");

        let mut oscillator = vec![f64::NAN; source.len()];
        let mut signal = vec![f64::NAN; source.len()];
        let mut histogram = vec![f64::NAN; source.len()];
        for i in 0..source.len() {
            if let Some((osc, sig, hist)) = stream.update(high[i], low[i], source[i]) {
                oscillator[i] = osc;
                signal[i] = sig;
                histogram[i] = hist;
            }
        }

        assert_close(&batch.oscillator, &oscillator, 1e-12);
        assert_close(&batch.signal, &signal, 1e-12);
        assert_close(&batch.histogram, &histogram, 1e-12);
    }

    fn check_batch_single_matches_single(kernel: Kernel) {
        let (high, low, source) = sample_hls(180);
        let batch = supertrend_oscillator_batch_with_kernel(
            &high,
            &low,
            &source,
            &SuperTrendOscillatorBatchRange {
                length: (12, 12, 0),
                mult: (2.5, 2.5, 0.0),
                smooth: (18, 18, 0),
            },
            kernel,
        )
        .expect("batch");
        let single = supertrend_oscillator(&SuperTrendOscillatorInput::from_slices(
            &high,
            &low,
            &source,
            SuperTrendOscillatorParams {
                length: Some(12),
                mult: Some(2.5),
                smooth: Some(18),
            },
        ))
        .expect("single");

        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, source.len());
        assert_close(&batch.oscillator[..source.len()], &single.oscillator, 1e-12);
        assert_close(&batch.signal[..source.len()], &single.signal, 1e-12);
        assert_close(&batch.histogram[..source.len()], &single.histogram, 1e-12);
    }

    #[test]
    fn supertrend_oscillator_invalid_params() {
        let (high, low, source) = sample_hls(64);

        let err = supertrend_oscillator(&SuperTrendOscillatorInput::from_slices(
            &high,
            &low,
            &source,
            SuperTrendOscillatorParams {
                length: Some(0),
                mult: Some(2.0),
                smooth: Some(10),
            },
        ))
        .expect_err("invalid length");
        assert!(matches!(
            err,
            SuperTrendOscillatorError::InvalidLength { .. }
        ));

        let err = supertrend_oscillator(&SuperTrendOscillatorInput::from_slices(
            &high,
            &low,
            &source,
            SuperTrendOscillatorParams {
                length: Some(10),
                mult: Some(0.0),
                smooth: Some(10),
            },
        ))
        .expect_err("invalid mult");
        assert!(matches!(
            err,
            SuperTrendOscillatorError::InvalidMultiplier { .. }
        ));

        let err = supertrend_oscillator(&SuperTrendOscillatorInput::from_slices(
            &high,
            &low,
            &source,
            SuperTrendOscillatorParams {
                length: Some(10),
                mult: Some(2.0),
                smooth: Some(0),
            },
        ))
        .expect_err("invalid smooth");
        assert!(matches!(
            err,
            SuperTrendOscillatorError::InvalidSmooth { .. }
        ));
    }

    #[test]
    fn supertrend_oscillator_dispatch_matches_direct() {
        let (high, low, source) = sample_hls(160);
        let combo = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(12),
            },
            ParamKV {
                key: "mult",
                value: ParamValue::Float(2.5),
            },
            ParamKV {
                key: "smooth",
                value: ParamValue::Int(18),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "supertrend_oscillator",
            output_id: Some("oscillator"),
            data: IndicatorDataRef::Ohlc {
                open: &source,
                high: &high,
                low: &low,
                close: &source,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).expect("dispatch");
        let direct = supertrend_oscillator(&SuperTrendOscillatorInput::from_slices(
            &high,
            &low,
            &source,
            SuperTrendOscillatorParams {
                length: Some(12),
                mult: Some(2.5),
                smooth: Some(18),
            },
        ))
        .expect("direct");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, source.len());
        assert_close(&out.values_f64.expect("values"), &direct.oscillator, 1e-12);
    }

    macro_rules! gen_kernel_tests {
        ($module:ident, $kernel:expr, $batch_kernel:expr) => {
            mod $module {
                use super::*;

                #[test]
                fn output_contract() {
                    check_output_contract($kernel);
                }

                #[test]
                fn into_matches_api() {
                    check_into_matches_api($kernel);
                }

                #[test]
                fn batch_single_matches_single() {
                    check_batch_single_matches_single($batch_kernel);
                }
            }
        };
    }

    gen_kernel_tests!(scalar_kernel, Kernel::Scalar, Kernel::ScalarBatch);
    gen_kernel_tests!(auto_kernel, Kernel::Auto, Kernel::Auto);

    #[test]
    fn supertrend_oscillator_stream_matches_batch() {
        check_stream_matches_batch();
    }
}
