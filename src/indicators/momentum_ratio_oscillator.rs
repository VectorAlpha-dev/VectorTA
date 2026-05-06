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
    alloc_uninit_f64, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_PERIOD: usize = 50;

impl<'a> AsRef<[f64]> for MomentumRatioOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MomentumRatioOscillatorData::Slice(slice) => slice,
            MomentumRatioOscillatorData::Candles { candles, source } => match *source {
                "open" => &candles.open,
                "high" => &candles.high,
                "low" => &candles.low,
                "close" => &candles.close,
                "volume" => &candles.volume,
                "hl2" => &candles.hl2,
                "hlc3" => &candles.hlc3,
                "ohlc4" => &candles.ohlc4,
                "hlcc4" | "hlcc" => &candles.hlcc4,
                _ => source_type(candles, source),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum MomentumRatioOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MomentumRatioOscillatorOutput {
    pub line: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct MomentumRatioOscillatorParams {
    pub period: Option<usize>,
}

impl Default for MomentumRatioOscillatorParams {
    fn default() -> Self {
        Self {
            period: Some(DEFAULT_PERIOD),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MomentumRatioOscillatorInput<'a> {
    pub data: MomentumRatioOscillatorData<'a>,
    pub params: MomentumRatioOscillatorParams,
}

impl<'a> MomentumRatioOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: MomentumRatioOscillatorParams,
    ) -> Self {
        Self {
            data: MomentumRatioOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: MomentumRatioOscillatorParams) -> Self {
        Self {
            data: MomentumRatioOscillatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", MomentumRatioOscillatorParams::default())
    }

    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(DEFAULT_PERIOD)
    }
}

#[derive(Clone, Debug)]
pub struct MomentumRatioOscillatorBuilder {
    period: Option<usize>,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for MomentumRatioOscillatorBuilder {
    fn default() -> Self {
        Self {
            period: None,
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MomentumRatioOscillatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn period(mut self, value: usize) -> Self {
        self.period = Some(value);
        self
    }

    #[inline(always)]
    pub fn source<S: Into<String>>(mut self, value: S) -> Self {
        self.source = Some(value.into());
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
    ) -> Result<MomentumRatioOscillatorOutput, MomentumRatioOscillatorError> {
        let input = MomentumRatioOscillatorInput::from_candles(
            candles,
            self.source.as_deref().unwrap_or("close"),
            MomentumRatioOscillatorParams {
                period: self.period,
            },
        );
        momentum_ratio_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<MomentumRatioOscillatorOutput, MomentumRatioOscillatorError> {
        let input = MomentumRatioOscillatorInput::from_slice(
            data,
            MomentumRatioOscillatorParams {
                period: self.period,
            },
        );
        momentum_ratio_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<MomentumRatioOscillatorStream, MomentumRatioOscillatorError> {
        MomentumRatioOscillatorStream::try_new(MomentumRatioOscillatorParams {
            period: self.period,
        })
    }
}

#[derive(Debug, Error)]
pub enum MomentumRatioOscillatorError {
    #[error("momentum_ratio_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("momentum_ratio_oscillator: All source values are invalid.")]
    AllValuesNaN,
    #[error(
        "momentum_ratio_oscillator: Invalid period: period = {period}, data length = {data_len}"
    )]
    InvalidPeriod { period: usize, data_len: usize },
    #[error(
        "momentum_ratio_oscillator: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "momentum_ratio_oscillator: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("momentum_ratio_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("momentum_ratio_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn first_valid_source(data: &[f64]) -> Option<usize> {
    data.iter().position(|x| x.is_finite())
}

#[inline(always)]
fn count_valid_from(data: &[f64], start: usize) -> usize {
    data[start..].iter().filter(|x| x.is_finite()).count()
}

#[inline(always)]
fn first_valid_with_second(data: &[f64]) -> Option<(usize, bool)> {
    let mut first = 0usize;
    let mut found = 0usize;
    for (i, value) in data.iter().enumerate() {
        if value.is_finite() {
            if found == 0 {
                first = i;
            }
            found += 1;
            if found == 2 {
                return Some((first, true));
            }
        }
    }
    if found == 0 {
        None
    } else {
        Some((first, false))
    }
}

#[inline(always)]
fn line_warmup(first: usize) -> usize {
    first.saturating_add(1)
}

#[inline(always)]
fn signal_warmup(first: usize) -> usize {
    first.saturating_add(2)
}

#[inline(always)]
fn normalized_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => detect_best_kernel(),
        other if other.is_batch() => other.to_non_batch(),
        other => other,
    }
}

#[inline(always)]
fn safe_ratio(num: f64, den: f64) -> f64 {
    if num.is_finite() && den.is_finite() && den != 0.0 {
        num / den
    } else {
        f64::NAN
    }
}

#[inline(always)]
fn momentum_ratio_oscillator_compute_into(
    data: &[f64],
    period: usize,
    _kernel: Kernel,
    out_line: &mut [f64],
    out_signal: &mut [f64],
) {
    let alpha = 2.0 / period as f64;
    let mut ema_prev = 0.0;
    let mut emaa_prev = 0.0;
    let mut emab_prev = 0.0;
    let mut has_ema = false;
    let mut has_emaa = false;
    let mut has_emab = false;
    let mut val_prev = f64::NAN;

    for i in 0..data.len() {
        let value = data[i];
        if !value.is_finite() {
            out_line[i] = f64::NAN;
            out_signal[i] = f64::NAN;
            ema_prev = 0.0;
            emaa_prev = 0.0;
            emab_prev = 0.0;
            has_ema = false;
            has_emaa = false;
            has_emab = false;
            val_prev = f64::NAN;
            continue;
        }

        let prev_ema_nz = if has_ema { ema_prev } else { 0.0 };
        let ema = prev_ema_nz + alpha * (value - prev_ema_nz);
        let ratioa = if has_ema {
            safe_ratio(ema, ema_prev)
        } else {
            f64::NAN
        };
        let prev_emaa_nz = if has_emaa { emaa_prev } else { 0.0 };
        let prev_emab_nz = if has_emab { emab_prev } else { 0.0 };
        let emaa_input = if ratioa.is_finite() && ratioa < 1.0 {
            ratioa
        } else {
            0.0
        };
        let emab_input = if ratioa.is_finite() && ratioa > 1.0 {
            ratioa
        } else {
            0.0
        };
        let emaa = prev_emaa_nz + alpha * (emaa_input - prev_emaa_nz);
        let emab = prev_emab_nz + alpha * (emab_input - prev_emab_nz);
        let ratiob = safe_ratio(ratioa, ratioa + emab);
        let val = {
            let denom = ratioa + ratiob * emaa;
            if ratioa.is_finite()
                && ratiob.is_finite()
                && emaa.is_finite()
                && denom.is_finite()
                && denom != 0.0
            {
                2.0 * ratioa / denom - 1.0
            } else {
                f64::NAN
            }
        };

        out_line[i] = val;
        out_signal[i] = val_prev;

        ema_prev = ema;
        emaa_prev = emaa;
        emab_prev = emab;
        has_ema = true;
        has_emaa = true;
        has_emab = true;
        val_prev = val;
    }
}

#[inline]
pub fn momentum_ratio_oscillator(
    input: &MomentumRatioOscillatorInput,
) -> Result<MomentumRatioOscillatorOutput, MomentumRatioOscillatorError> {
    momentum_ratio_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn momentum_ratio_oscillator_with_kernel(
    input: &MomentumRatioOscillatorInput,
    kernel: Kernel,
) -> Result<MomentumRatioOscillatorOutput, MomentumRatioOscillatorError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(MomentumRatioOscillatorError::EmptyInputData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(MomentumRatioOscillatorError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let (first, has_second) =
        first_valid_with_second(data).ok_or(MomentumRatioOscillatorError::AllValuesNaN)?;
    if !has_second {
        return Err(MomentumRatioOscillatorError::NotEnoughValidData {
            needed: 2,
            valid: 1,
        });
    }

    let mut line = alloc_uninit_f64(data.len());
    let mut signal = alloc_uninit_f64(data.len());
    momentum_ratio_oscillator_compute_into(
        data,
        period,
        normalized_kernel(kernel),
        &mut line,
        &mut signal,
    );
    Ok(MomentumRatioOscillatorOutput { line, signal })
}

#[inline]
pub fn momentum_ratio_oscillator_into_slice(
    out_line: &mut [f64],
    out_signal: &mut [f64],
    input: &MomentumRatioOscillatorInput,
    kernel: Kernel,
) -> Result<(), MomentumRatioOscillatorError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(MomentumRatioOscillatorError::EmptyInputData);
    }
    if out_line.len() != data.len() || out_signal.len() != data.len() {
        return Err(MomentumRatioOscillatorError::OutputLengthMismatch {
            expected: data.len(),
            got: out_line.len().max(out_signal.len()),
        });
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(MomentumRatioOscillatorError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }
    let (_first, has_second) =
        first_valid_with_second(data).ok_or(MomentumRatioOscillatorError::AllValuesNaN)?;
    if !has_second {
        return Err(MomentumRatioOscillatorError::NotEnoughValidData {
            needed: 2,
            valid: 1,
        });
    }

    momentum_ratio_oscillator_compute_into(
        data,
        period,
        normalized_kernel(kernel),
        out_line,
        out_signal,
    );
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn momentum_ratio_oscillator_into(
    input: &MomentumRatioOscillatorInput,
    out_line: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), MomentumRatioOscillatorError> {
    momentum_ratio_oscillator_into_slice(out_line, out_signal, input, Kernel::Auto)
}

#[derive(Clone, Debug)]
pub struct MomentumRatioOscillatorStream {
    alpha: f64,
    ema_prev: Option<f64>,
    emaa_prev: Option<f64>,
    emab_prev: Option<f64>,
    val_prev: f64,
}

impl MomentumRatioOscillatorStream {
    #[inline]
    pub fn try_new(
        params: MomentumRatioOscillatorParams,
    ) -> Result<Self, MomentumRatioOscillatorError> {
        let period = params.period.unwrap_or(DEFAULT_PERIOD);
        if period == 0 {
            return Err(MomentumRatioOscillatorError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            alpha: 2.0 / period as f64,
            ema_prev: None,
            emaa_prev: None,
            emab_prev: None,
            val_prev: f64::NAN,
        })
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.ema_prev = None;
        self.emaa_prev = None;
        self.emab_prev = None;
        self.val_prev = f64::NAN;
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.update_reset_on_nan(value)
    }

    #[inline]
    pub fn update_reset_on_nan(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let prev_ema_nz = self.ema_prev.unwrap_or(0.0);
        let ema = prev_ema_nz + self.alpha * (value - prev_ema_nz);
        let ratioa = match self.ema_prev {
            Some(prev_ema) => safe_ratio(ema, prev_ema),
            None => f64::NAN,
        };
        let prev_emaa_nz = self.emaa_prev.unwrap_or(0.0);
        let prev_emab_nz = self.emab_prev.unwrap_or(0.0);
        let emaa_input = if ratioa.is_finite() && ratioa < 1.0 {
            ratioa
        } else {
            0.0
        };
        let emab_input = if ratioa.is_finite() && ratioa > 1.0 {
            ratioa
        } else {
            0.0
        };
        let emaa = prev_emaa_nz + self.alpha * (emaa_input - prev_emaa_nz);
        let emab = prev_emab_nz + self.alpha * (emab_input - prev_emab_nz);
        let ratiob = safe_ratio(ratioa, ratioa + emab);
        let val = {
            let denom = ratioa + ratiob * emaa;
            if ratioa.is_finite()
                && ratiob.is_finite()
                && emaa.is_finite()
                && denom.is_finite()
                && denom != 0.0
            {
                2.0 * ratioa / denom - 1.0
            } else {
                f64::NAN
            }
        };
        let signal = self.val_prev;

        self.ema_prev = Some(ema);
        self.emaa_prev = Some(emaa);
        self.emab_prev = Some(emab);
        self.val_prev = val;

        if val.is_finite() {
            Some((val, signal))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct MomentumRatioOscillatorBatchOutput {
    pub line: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<MomentumRatioOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MomentumRatioOscillatorBatchOutput {
    #[inline]
    pub fn row_for_params(&self, params: &MomentumRatioOscillatorParams) -> Option<usize> {
        self.combos.iter().position(|p| p.period == params.period)
    }
}

#[derive(Debug, Clone)]
pub struct MomentumRatioOscillatorBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for MomentumRatioOscillatorBatchRange {
    fn default() -> Self {
        Self {
            period: (DEFAULT_PERIOD, DEFAULT_PERIOD, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MomentumRatioOscillatorBatchBuilder {
    range: MomentumRatioOscillatorBatchRange,
    kernel: Kernel,
}

impl Default for MomentumRatioOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: MomentumRatioOscillatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl MomentumRatioOscillatorBatchBuilder {
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
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }

    #[inline]
    pub fn period_static(mut self, value: usize) -> Self {
        self.range.period = (value, value, 0);
        self
    }

    #[inline]
    pub fn run_on_slice(
        self,
        data: &[f64],
    ) -> Result<MomentumRatioOscillatorBatchOutput, MomentumRatioOscillatorError> {
        momentum_ratio_oscillator_batch_with_kernel(data, &self.range, self.kernel)
    }
}

#[inline]
pub fn expand_grid_momentum_ratio_oscillator(
    range: &MomentumRatioOscillatorBatchRange,
) -> Result<Vec<MomentumRatioOscillatorParams>, MomentumRatioOscillatorError> {
    fn axis_usize(
        (start, end, step): (usize, usize, usize),
    ) -> Result<Vec<usize>, MomentumRatioOscillatorError> {
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
                return Err(MomentumRatioOscillatorError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(out)
        } else {
            let mut out = Vec::new();
            let mut x = start;
            while x >= end {
                out.push(x);
                if x == end {
                    break;
                }
                let next = x.saturating_sub(step.max(1));
                if next == x || next < end {
                    break;
                }
                x = next;
            }
            if out.is_empty() {
                return Err(MomentumRatioOscillatorError::InvalidRange {
                    start: start.to_string(),
                    end: end.to_string(),
                    step: step.to_string(),
                });
            }
            Ok(out)
        }
    }

    let periods = axis_usize(range.period)?;
    Ok(periods
        .into_iter()
        .map(|period| MomentumRatioOscillatorParams {
            period: Some(period),
        })
        .collect())
}

#[inline]
pub fn momentum_ratio_oscillator_batch_with_kernel(
    data: &[f64],
    sweep: &MomentumRatioOscillatorBatchRange,
    kernel: Kernel,
) -> Result<MomentumRatioOscillatorBatchOutput, MomentumRatioOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(MomentumRatioOscillatorError::InvalidKernelForBatch(other)),
    };
    momentum_ratio_oscillator_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn momentum_ratio_oscillator_batch_slice(
    data: &[f64],
    sweep: &MomentumRatioOscillatorBatchRange,
    kernel: Kernel,
) -> Result<MomentumRatioOscillatorBatchOutput, MomentumRatioOscillatorError> {
    momentum_ratio_oscillator_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn momentum_ratio_oscillator_batch_par_slice(
    data: &[f64],
    sweep: &MomentumRatioOscillatorBatchRange,
    kernel: Kernel,
) -> Result<MomentumRatioOscillatorBatchOutput, MomentumRatioOscillatorError> {
    momentum_ratio_oscillator_batch_inner(data, sweep, kernel, true)
}

fn momentum_ratio_oscillator_batch_inner(
    data: &[f64],
    sweep: &MomentumRatioOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<MomentumRatioOscillatorBatchOutput, MomentumRatioOscillatorError> {
    let combos = expand_grid_momentum_ratio_oscillator(sweep)?;
    if data.is_empty() {
        return Err(MomentumRatioOscillatorError::EmptyInputData);
    }
    let first = first_valid_source(data).ok_or(MomentumRatioOscillatorError::AllValuesNaN)?;
    let valid = count_valid_from(data, first);
    if valid < 2 {
        return Err(MomentumRatioOscillatorError::NotEnoughValidData { needed: 2, valid });
    }
    let max_period = combos
        .iter()
        .map(|p| p.period.unwrap_or(DEFAULT_PERIOD))
        .max()
        .unwrap_or(DEFAULT_PERIOD);
    if max_period == 0 || max_period > data.len() {
        return Err(MomentumRatioOscillatorError::InvalidPeriod {
            period: max_period,
            data_len: data.len(),
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut line_matrix = make_uninit_matrix(rows, cols);
    let mut signal_matrix = make_uninit_matrix(rows, cols);
    let line_warmups: Vec<usize> = (0..rows).map(|_| line_warmup(first)).collect();
    let signal_warmups: Vec<usize> = (0..rows).map(|_| signal_warmup(first)).collect();
    init_matrix_prefixes(&mut line_matrix, cols, &line_warmups);
    init_matrix_prefixes(&mut signal_matrix, cols, &signal_warmups);

    let mut line_guard = ManuallyDrop::new(line_matrix);
    let mut signal_guard = ManuallyDrop::new(signal_matrix);
    let line_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(line_guard.as_mut_ptr(), line_guard.len()) };
    let signal_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(signal_guard.as_mut_ptr(), signal_guard.len()) };

    let do_row = |row: usize,
                  row_line_mu: &mut [MaybeUninit<f64>],
                  row_signal_mu: &mut [MaybeUninit<f64>]| {
        let params = &combos[row];
        let period = params.period.unwrap_or(DEFAULT_PERIOD);
        let dst_line =
            unsafe { std::slice::from_raw_parts_mut(row_line_mu.as_mut_ptr() as *mut f64, cols) };
        let dst_signal =
            unsafe { std::slice::from_raw_parts_mut(row_signal_mu.as_mut_ptr() as *mut f64, cols) };
        momentum_ratio_oscillator_compute_into(data, period, kernel, dst_line, dst_signal);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        line_mu
            .par_chunks_mut(cols)
            .zip(signal_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (row_line, row_signal))| do_row(row, row_line, row_signal));
        #[cfg(target_arch = "wasm32")]
        for (row, (row_line, row_signal)) in line_mu
            .chunks_mut(cols)
            .zip(signal_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_line, row_signal);
        }
    } else {
        for (row, (row_line, row_signal)) in line_mu
            .chunks_mut(cols)
            .zip(signal_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_line, row_signal);
        }
    }

    let line = unsafe {
        Vec::from_raw_parts(
            line_guard.as_mut_ptr() as *mut f64,
            line_guard.len(),
            line_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };

    Ok(MomentumRatioOscillatorBatchOutput {
        line,
        signal,
        combos,
        rows,
        cols,
    })
}

fn momentum_ratio_oscillator_batch_inner_into(
    data: &[f64],
    sweep: &MomentumRatioOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_line: &mut [f64],
    out_signal: &mut [f64],
) -> Result<Vec<MomentumRatioOscillatorParams>, MomentumRatioOscillatorError> {
    let combos = expand_grid_momentum_ratio_oscillator(sweep)?;
    if data.is_empty() {
        return Err(MomentumRatioOscillatorError::EmptyInputData);
    }
    let first = first_valid_source(data).ok_or(MomentumRatioOscillatorError::AllValuesNaN)?;
    let valid = count_valid_from(data, first);
    if valid < 2 {
        return Err(MomentumRatioOscillatorError::NotEnoughValidData { needed: 2, valid });
    }

    let rows = combos.len();
    let cols = data.len();
    let total =
        rows.checked_mul(cols)
            .ok_or(MomentumRatioOscillatorError::OutputLengthMismatch {
                expected: usize::MAX,
                got: 0,
            })?;
    if out_line.len() != total || out_signal.len() != total {
        return Err(MomentumRatioOscillatorError::OutputLengthMismatch {
            expected: total,
            got: out_line.len().max(out_signal.len()),
        });
    }

    let do_row = |row: usize, dst_line: &mut [f64], dst_signal: &mut [f64]| {
        let params = &combos[row];
        let period = params.period.unwrap_or(DEFAULT_PERIOD);
        momentum_ratio_oscillator_compute_into(data, period, kernel, dst_line, dst_signal);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_line
            .par_chunks_mut(cols)
            .zip(out_signal.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (dst_line, dst_signal))| do_row(row, dst_line, dst_signal));
        #[cfg(target_arch = "wasm32")]
        for (row, (dst_line, dst_signal)) in out_line
            .chunks_mut(cols)
            .zip(out_signal.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_line, dst_signal);
        }
    } else {
        for (row, (dst_line, dst_signal)) in out_line
            .chunks_mut(cols)
            .zip(out_signal.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, dst_line, dst_signal);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "momentum_ratio_oscillator")]
#[pyo3(signature = (data, period=DEFAULT_PERIOD, kernel=None))]
pub fn momentum_ratio_oscillator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = MomentumRatioOscillatorInput::from_slice(
        data,
        MomentumRatioOscillatorParams {
            period: Some(period),
        },
    );
    let output = py
        .allow_threads(|| momentum_ratio_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((output.line.into_pyarray(py), output.signal.into_pyarray(py)))
}

#[cfg(feature = "python")]
#[pyclass(name = "MomentumRatioOscillatorStream")]
pub struct MomentumRatioOscillatorStreamPy {
    stream: MomentumRatioOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MomentumRatioOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (period=DEFAULT_PERIOD))]
    fn new(period: usize) -> PyResult<Self> {
        let stream = MomentumRatioOscillatorStream::try_new(MomentumRatioOscillatorParams {
            period: Some(period),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "momentum_ratio_oscillator_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn momentum_ratio_oscillator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let sweep = MomentumRatioOscillatorBatchRange {
        period: period_range,
    };
    let combos = expand_grid_momentum_ratio_oscillator(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let line_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let line_out = unsafe { line_arr.as_slice_mut()? };
    let signal_out = unsafe { signal_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        momentum_ratio_oscillator_batch_inner_into(
            data,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            line_out,
            signal_out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let periods: Vec<usize> = combos
        .iter()
        .map(|c| c.period.unwrap_or(DEFAULT_PERIOD))
        .collect();
    let dict = PyDict::new(py);
    dict.set_item("line", line_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("periods", periods.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_momentum_ratio_oscillator_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(momentum_ratio_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(momentum_ratio_oscillator_batch_py, m)?)?;
    m.add_class::<MomentumRatioOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MomentumRatioOscillatorJsOutput {
    line: Vec<f64>,
    signal: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MomentumRatioOscillatorBatchConfig {
    period_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MomentumRatioOscillatorBatchJsOutput {
    line: Vec<f64>,
    signal: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<MomentumRatioOscillatorParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "momentum_ratio_oscillator_js")]
pub fn momentum_ratio_oscillator_js(data: &[f64], period: usize) -> Result<JsValue, JsValue> {
    let input = MomentumRatioOscillatorInput::from_slice(
        data,
        MomentumRatioOscillatorParams {
            period: Some(period),
        },
    );
    let output =
        momentum_ratio_oscillator(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MomentumRatioOscillatorJsOutput {
        line: output.line,
        signal: output.signal,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "momentum_ratio_oscillator_batch_js")]
pub fn momentum_ratio_oscillator_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: MomentumRatioOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.period_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: ranges must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = MomentumRatioOscillatorBatchRange {
        period: (
            config.period_range[0],
            config.period_range[1],
            config.period_range[2],
        ),
    };
    let batch = momentum_ratio_oscillator_batch_slice(data, &sweep, Kernel::Scalar)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MomentumRatioOscillatorBatchJsOutput {
        line: batch.line,
        signal: batch.signal,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn momentum_ratio_oscillator_alloc(len: usize) -> *mut f64 {
    let mut buf = vec![0.0_f64; len * 2];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn momentum_ratio_oscillator_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 2);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn momentum_ratio_oscillator_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to momentum_ratio_oscillator_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 2);
        let (out_line, out_signal) = out.split_at_mut(len);
        let input = MomentumRatioOscillatorInput::from_slice(
            data,
            MomentumRatioOscillatorParams {
                period: Some(period),
            },
        );
        momentum_ratio_oscillator_into_slice(out_line, out_signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "momentum_ratio_oscillator_into_host")]
pub fn momentum_ratio_oscillator_into_host(
    data: &[f64],
    out_ptr: *mut f64,
    period: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to momentum_ratio_oscillator_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, data.len() * 2);
        let (out_line, out_signal) = out.split_at_mut(data.len());
        let input = MomentumRatioOscillatorInput::from_slice(
            data,
            MomentumRatioOscillatorParams {
                period: Some(period),
            },
        );
        momentum_ratio_oscillator_into_slice(out_line, out_signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn momentum_ratio_oscillator_batch_into(
    data_ptr: *const f64,
    line_ptr: *mut f64,
    signal_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || line_ptr.is_null() || signal_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to momentum_ratio_oscillator_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let sweep = MomentumRatioOscillatorBatchRange {
            period: (period_start, period_end, period_step),
        };
        let combos = expand_grid_momentum_ratio_oscillator(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows
            .checked_mul(len)
            .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
        let line = std::slice::from_raw_parts_mut(line_ptr, total);
        let signal = std::slice::from_raw_parts_mut(signal_ptr, total);
        momentum_ratio_oscillator_batch_inner_into(
            data,
            &sweep,
            Kernel::Scalar,
            false,
            line,
            signal,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn momentum_ratio_oscillator_output_into_js(
    data: &[f64],
    period: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = momentum_ratio_oscillator_js(data, period)?;
    crate::write_wasm_object_f64_outputs("momentum_ratio_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn momentum_ratio_oscillator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = momentum_ratio_oscillator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "momentum_ratio_oscillator_batch_output_into_js",
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
            let x = a[i];
            let y = b[i];
            if x.is_nan() || y.is_nan() {
                assert!(x.is_nan() && y.is_nan(), "nan mismatch at {i}: {x} vs {y}");
            } else {
                assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y}");
            }
        }
    }

    fn sample_data(len: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            out.push(100.0 + i as f64 * 0.12 + (i as f64 * 0.17).sin() * 1.8);
        }
        out
    }

    #[test]
    fn momentum_ratio_oscillator_output_contract() {
        let data = sample_data(128);
        let input = MomentumRatioOscillatorInput::from_slice(
            &data,
            MomentumRatioOscillatorParams { period: Some(50) },
        );
        let out = momentum_ratio_oscillator(&input).expect("indicator");
        assert_eq!(out.line.len(), data.len());
        assert_eq!(out.signal.len(), data.len());
        assert!(out.line[0].is_nan());
        assert!(out.signal[0].is_nan());
        assert!(out.signal[1].is_nan());
        assert!(out.line[1..].iter().any(|x| x.is_finite()));
        assert!(out.signal[2..].iter().any(|x| x.is_finite()));
    }

    #[test]
    fn momentum_ratio_oscillator_into_matches_api() {
        let data = sample_data(160);
        let input = MomentumRatioOscillatorInput::from_slice(
            &data,
            MomentumRatioOscillatorParams { period: Some(25) },
        );
        let baseline = momentum_ratio_oscillator(&input).expect("baseline");
        let mut line = vec![0.0; data.len()];
        let mut signal = vec![0.0; data.len()];
        momentum_ratio_oscillator_into_slice(&mut line, &mut signal, &input, Kernel::Auto)
            .expect("into");
        assert_close(&baseline.line, &line, 1e-12);
        assert_close(&baseline.signal, &signal, 1e-12);
    }

    #[test]
    fn momentum_ratio_oscillator_stream_matches_batch() {
        let data = sample_data(160);
        let input = MomentumRatioOscillatorInput::from_slice(
            &data,
            MomentumRatioOscillatorParams { period: Some(30) },
        );
        let batch = momentum_ratio_oscillator(&input).expect("batch");
        let mut stream = MomentumRatioOscillatorStream::try_new(MomentumRatioOscillatorParams {
            period: Some(30),
        })
        .expect("stream");
        let mut line = vec![f64::NAN; data.len()];
        let mut signal = vec![f64::NAN; data.len()];
        for (i, value) in data.iter().copied().enumerate() {
            if let Some((l, s)) = stream.update(value) {
                line[i] = l;
                signal[i] = s;
            }
        }
        assert_close(&batch.line, &line, 1e-12);
        assert_close(&batch.signal, &signal, 1e-12);
    }

    #[test]
    fn momentum_ratio_oscillator_batch_single_param_matches_single() {
        let data = sample_data(160);
        let batch = momentum_ratio_oscillator_batch_with_kernel(
            &data,
            &MomentumRatioOscillatorBatchRange {
                period: (30, 30, 0),
            },
            Kernel::Auto,
        )
        .expect("batch");
        let single = momentum_ratio_oscillator(&MomentumRatioOscillatorInput::from_slice(
            &data,
            MomentumRatioOscillatorParams { period: Some(30) },
        ))
        .expect("single");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_close(&batch.line[..data.len()], &single.line, 1e-12);
        assert_close(&batch.signal[..data.len()], &single.signal, 1e-12);
    }

    #[test]
    fn momentum_ratio_oscillator_dispatch_matches_direct() {
        let data = sample_data(160);
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(30),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "momentum_ratio_oscillator",
            output_id: Some("line"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).expect("dispatch");
        let direct = momentum_ratio_oscillator(&MomentumRatioOscillatorInput::from_slice(
            &data,
            MomentumRatioOscillatorParams { period: Some(30) },
        ))
        .expect("direct");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, data.len());
        assert_close(&out.values_f64.expect("values"), &direct.line, 1e-12);
    }
}
