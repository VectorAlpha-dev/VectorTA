#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

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
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_EMA_LENGTH: usize = 21;
const DEFAULT_SIGNAL_LENGTH: usize = 34;
const OSL_DIVISOR: f64 = 10.0;

impl<'a> AsRef<[f64]> for AbsoluteStrengthIndexOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AbsoluteStrengthIndexOscillatorData::Slice(slice) => slice,
            AbsoluteStrengthIndexOscillatorData::Candles { candles, source } => {
                source_type(candles, source)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum AbsoluteStrengthIndexOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AbsoluteStrengthIndexOscillatorOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
    pub histogram: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsoluteStrengthIndexOscillatorOutputField {
    Oscillator,
    Signal,
    Histogram,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AbsoluteStrengthIndexOscillatorParams {
    pub ema_length: Option<usize>,
    pub signal_length: Option<usize>,
}

impl Default for AbsoluteStrengthIndexOscillatorParams {
    fn default() -> Self {
        Self {
            ema_length: Some(DEFAULT_EMA_LENGTH),
            signal_length: Some(DEFAULT_SIGNAL_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AbsoluteStrengthIndexOscillatorInput<'a> {
    pub data: AbsoluteStrengthIndexOscillatorData<'a>,
    pub params: AbsoluteStrengthIndexOscillatorParams,
}

impl<'a> AbsoluteStrengthIndexOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: AbsoluteStrengthIndexOscillatorParams,
    ) -> Self {
        Self {
            data: AbsoluteStrengthIndexOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: AbsoluteStrengthIndexOscillatorParams) -> Self {
        Self {
            data: AbsoluteStrengthIndexOscillatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "close",
            AbsoluteStrengthIndexOscillatorParams::default(),
        )
    }

    #[inline]
    pub fn get_ema_length(&self) -> usize {
        self.params.ema_length.unwrap_or(DEFAULT_EMA_LENGTH)
    }

    #[inline]
    pub fn get_signal_length(&self) -> usize {
        self.params.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AbsoluteStrengthIndexOscillatorBuilder {
    ema_length: Option<usize>,
    signal_length: Option<usize>,
    kernel: Kernel,
}

impl Default for AbsoluteStrengthIndexOscillatorBuilder {
    fn default() -> Self {
        Self {
            ema_length: None,
            signal_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AbsoluteStrengthIndexOscillatorBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn ema_length(mut self, ema_length: usize) -> Self {
        self.ema_length = Some(ema_length);
        self
    }

    #[inline]
    pub fn signal_length(mut self, signal_length: usize) -> Self {
        self.signal_length = Some(signal_length);
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<AbsoluteStrengthIndexOscillatorOutput, AbsoluteStrengthIndexOscillatorError> {
        let input = AbsoluteStrengthIndexOscillatorInput::from_candles(
            candles,
            source,
            AbsoluteStrengthIndexOscillatorParams {
                ema_length: self.ema_length,
                signal_length: self.signal_length,
            },
        );
        absolute_strength_index_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<AbsoluteStrengthIndexOscillatorOutput, AbsoluteStrengthIndexOscillatorError> {
        let input = AbsoluteStrengthIndexOscillatorInput::from_slice(
            data,
            AbsoluteStrengthIndexOscillatorParams {
                ema_length: self.ema_length,
                signal_length: self.signal_length,
            },
        );
        absolute_strength_index_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<AbsoluteStrengthIndexOscillatorStream, AbsoluteStrengthIndexOscillatorError> {
        AbsoluteStrengthIndexOscillatorStream::try_new(AbsoluteStrengthIndexOscillatorParams {
            ema_length: self.ema_length,
            signal_length: self.signal_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum AbsoluteStrengthIndexOscillatorError {
    #[error("absolute_strength_index_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("absolute_strength_index_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error("absolute_strength_index_oscillator: Invalid ema_length: {ema_length}")]
    InvalidEmaLength { ema_length: usize },
    #[error("absolute_strength_index_oscillator: Invalid signal_length: {signal_length}")]
    InvalidSignalLength { signal_length: usize },
    #[error("absolute_strength_index_oscillator: Output length mismatch: expected = {expected}, oscillator = {oscillator_got}, signal = {signal_got}, histogram = {histogram_got}")]
    OutputLengthMismatch {
        expected: usize,
        oscillator_got: usize,
        signal_got: usize,
        histogram_got: usize,
    },
    #[error(
        "absolute_strength_index_oscillator: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("absolute_strength_index_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    ema_length: usize,
    signal_length: usize,
    ema_alpha: f64,
    signal_alpha: f64,
}

#[derive(Debug, Clone)]
pub struct AbsoluteStrengthIndexOscillatorStream {
    params: ResolvedParams,
    prev_close: Option<f64>,
    a: f64,
    m: f64,
    d: f64,
    ema_abssi: Option<f64>,
    mt: f64,
    ut: f64,
}

impl AbsoluteStrengthIndexOscillatorStream {
    pub fn try_new(
        params: AbsoluteStrengthIndexOscillatorParams,
    ) -> Result<Self, AbsoluteStrengthIndexOscillatorError> {
        let params = resolve_params(&params)?;
        Ok(Self::new_resolved(params))
    }

    #[inline]
    fn new_resolved(params: ResolvedParams) -> Self {
        Self {
            params,
            prev_close: None,
            a: 0.0,
            m: 0.0,
            d: 0.0,
            ema_abssi: None,
            mt: 0.0,
            ut: 0.0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new_resolved(self.params);
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let abssi = if let Some(prev) = self.prev_close {
            if value > prev {
                if prev != 0.0 {
                    self.a += value / prev - 1.0;
                }
            } else if value < prev {
                if value != 0.0 {
                    self.d += prev / value - 1.0;
                }
            } else {
                self.m += 1.0 / OSL_DIVISOR;
            }
            self.prev_close = Some(value);
            let denom = self.d + self.m * 0.5;
            if denom == 0.0 {
                1.0
            } else {
                1.0 - 1.0 / (1.0 + (self.a + self.m * 0.5) / denom)
            }
        } else {
            self.prev_close = Some(value);
            1.0
        };

        let ema_abssi = match self.ema_abssi {
            Some(prev) => self.params.ema_alpha * abssi + (1.0 - self.params.ema_alpha) * prev,
            None => abssi,
        };
        self.ema_abssi = Some(ema_abssi);

        let oscillator = abssi - ema_abssi;
        self.mt =
            self.params.signal_alpha * oscillator + (1.0 - self.params.signal_alpha) * self.mt;
        self.ut = self.params.signal_alpha * self.mt + (1.0 - self.params.signal_alpha) * self.ut;

        let signal = ((2.0 - self.params.signal_alpha) * self.mt - self.ut)
            / (1.0 - self.params.signal_alpha);
        let histogram = oscillator - signal;
        Some((oscillator, signal, histogram))
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AbsoluteStrengthIndexOscillatorBatchRange {
    pub ema_length: (usize, usize, usize),
    pub signal_length: (usize, usize, usize),
}

impl Default for AbsoluteStrengthIndexOscillatorBatchRange {
    fn default() -> Self {
        Self {
            ema_length: (DEFAULT_EMA_LENGTH, DEFAULT_EMA_LENGTH, 0),
            signal_length: (DEFAULT_SIGNAL_LENGTH, DEFAULT_SIGNAL_LENGTH, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AbsoluteStrengthIndexOscillatorBatchOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
    pub histogram: Vec<f64>,
    pub combos: Vec<AbsoluteStrengthIndexOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub struct AbsoluteStrengthIndexOscillatorBatchBuilder {
    range: AbsoluteStrengthIndexOscillatorBatchRange,
    kernel: Kernel,
}

impl Default for AbsoluteStrengthIndexOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: AbsoluteStrengthIndexOscillatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl AbsoluteStrengthIndexOscillatorBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn ema_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.ema_length = (start, end, step);
        self
    }

    #[inline]
    pub fn signal_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.signal_length = (start, end, step);
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<AbsoluteStrengthIndexOscillatorBatchOutput, AbsoluteStrengthIndexOscillatorError>
    {
        absolute_strength_index_oscillator_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<AbsoluteStrengthIndexOscillatorBatchOutput, AbsoluteStrengthIndexOscillatorError>
    {
        self.apply_slice(source_type(candles, source))
    }

    #[inline]
    pub fn with_default_candles(
        candles: &Candles,
    ) -> Result<AbsoluteStrengthIndexOscillatorBatchOutput, AbsoluteStrengthIndexOscillatorError>
    {
        Self::new().apply_candles(candles, "close")
    }
}

#[inline]
fn first_valid_value(data: &[f64]) -> usize {
    data.iter()
        .position(|v| v.is_finite())
        .unwrap_or(data.len())
}

fn resolve_params(
    params: &AbsoluteStrengthIndexOscillatorParams,
) -> Result<ResolvedParams, AbsoluteStrengthIndexOscillatorError> {
    let ema_length = params.ema_length.unwrap_or(DEFAULT_EMA_LENGTH);
    if ema_length == 0 {
        return Err(AbsoluteStrengthIndexOscillatorError::InvalidEmaLength { ema_length });
    }

    let signal_length = params.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH);
    if signal_length <= 1 {
        return Err(AbsoluteStrengthIndexOscillatorError::InvalidSignalLength { signal_length });
    }

    Ok(ResolvedParams {
        ema_length,
        signal_length,
        ema_alpha: 2.0 / (ema_length as f64 + 1.0),
        signal_alpha: 2.0 / (signal_length as f64 + 1.0),
    })
}

#[inline]
fn prepare<'a>(
    input: &'a AbsoluteStrengthIndexOscillatorInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], ResolvedParams, usize, Kernel), AbsoluteStrengthIndexOscillatorError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(AbsoluteStrengthIndexOscillatorError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(AbsoluteStrengthIndexOscillatorError::AllValuesNaN);
    }
    let params = resolve_params(&input.params)?;
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    Ok((data, params, first, chosen))
}

#[inline]
fn absolute_strength_index_oscillator_row_from_slice(
    data: &[f64],
    params: ResolvedParams,
    oscillator: &mut [f64],
    signal: &mut [f64],
    histogram: &mut [f64],
) {
    let mut stream = AbsoluteStrengthIndexOscillatorStream::new_resolved(params);
    for i in 0..data.len() {
        match stream.update(data[i]) {
            Some((osc, sig, hist)) => {
                oscillator[i] = osc;
                signal[i] = sig;
                histogram[i] = hist;
            }
            None => {
                oscillator[i] = f64::NAN;
                signal[i] = f64::NAN;
                histogram[i] = f64::NAN;
            }
        }
    }
}

#[inline]
fn absolute_strength_index_oscillator_row_field_from_slice(
    data: &[f64],
    params: ResolvedParams,
    out: &mut [f64],
    field: AbsoluteStrengthIndexOscillatorOutputField,
) {
    let mut stream = AbsoluteStrengthIndexOscillatorStream::new_resolved(params);
    for i in 0..data.len() {
        out[i] = match stream.update(data[i]) {
            Some((osc, sig, hist)) => match field {
                AbsoluteStrengthIndexOscillatorOutputField::Oscillator => osc,
                AbsoluteStrengthIndexOscillatorOutputField::Signal => sig,
                AbsoluteStrengthIndexOscillatorOutputField::Histogram => hist,
            },
            None => f64::NAN,
        };
    }
}

pub fn absolute_strength_index_oscillator(
    input: &AbsoluteStrengthIndexOscillatorInput,
) -> Result<AbsoluteStrengthIndexOscillatorOutput, AbsoluteStrengthIndexOscillatorError> {
    absolute_strength_index_oscillator_with_kernel(input, Kernel::Auto)
}

pub fn absolute_strength_index_oscillator_with_kernel(
    input: &AbsoluteStrengthIndexOscillatorInput,
    kernel: Kernel,
) -> Result<AbsoluteStrengthIndexOscillatorOutput, AbsoluteStrengthIndexOscillatorError> {
    let (data, params, _first, _chosen) = prepare(input, kernel)?;
    let mut oscillator = alloc_uninit_f64(data.len());
    let mut signal = alloc_uninit_f64(data.len());
    let mut histogram = alloc_uninit_f64(data.len());

    absolute_strength_index_oscillator_row_from_slice(
        data,
        params,
        &mut oscillator,
        &mut signal,
        &mut histogram,
    );

    Ok(AbsoluteStrengthIndexOscillatorOutput {
        oscillator,
        signal,
        histogram,
    })
}

pub fn absolute_strength_index_oscillator_into_slices(
    oscillator_out: &mut [f64],
    signal_out: &mut [f64],
    histogram_out: &mut [f64],
    input: &AbsoluteStrengthIndexOscillatorInput,
    kernel: Kernel,
) -> Result<(), AbsoluteStrengthIndexOscillatorError> {
    let (data, params, _first, _chosen) = prepare(input, kernel)?;
    let expected = data.len();
    if oscillator_out.len() != expected
        || signal_out.len() != expected
        || histogram_out.len() != expected
    {
        return Err(AbsoluteStrengthIndexOscillatorError::OutputLengthMismatch {
            expected,
            oscillator_got: oscillator_out.len(),
            signal_got: signal_out.len(),
            histogram_got: histogram_out.len(),
        });
    }

    absolute_strength_index_oscillator_row_from_slice(
        data,
        params,
        oscillator_out,
        signal_out,
        histogram_out,
    );
    Ok(())
}

pub fn absolute_strength_index_oscillator_output_into_slice(
    out: &mut [f64],
    input: &AbsoluteStrengthIndexOscillatorInput,
    kernel: Kernel,
    field: AbsoluteStrengthIndexOscillatorOutputField,
) -> Result<(), AbsoluteStrengthIndexOscillatorError> {
    let (data, params, _first, _chosen) = prepare(input, kernel)?;
    if out.len() != data.len() {
        return Err(AbsoluteStrengthIndexOscillatorError::OutputLengthMismatch {
            expected: data.len(),
            oscillator_got: out.len(),
            signal_got: data.len(),
            histogram_got: data.len(),
        });
    }

    absolute_strength_index_oscillator_row_field_from_slice(data, params, out, field);
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn absolute_strength_index_oscillator_into(
    input: &AbsoluteStrengthIndexOscillatorInput,
    oscillator_out: &mut [f64],
    signal_out: &mut [f64],
    histogram_out: &mut [f64],
) -> Result<(), AbsoluteStrengthIndexOscillatorError> {
    absolute_strength_index_oscillator_into_slices(
        oscillator_out,
        signal_out,
        histogram_out,
        input,
        Kernel::Auto,
    )
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, AbsoluteStrengthIndexOscillatorError> {
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(AbsoluteStrengthIndexOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }

    let mut out = Vec::new();
    if start < end {
        let mut x = start;
        while x <= end {
            out.push(x);
            let next = x.saturating_add(step);
            if next == x {
                break;
            }
            x = next;
        }
    } else {
        let mut x = start;
        while x >= end {
            out.push(x);
            let next = x.saturating_sub(step);
            if next == x {
                break;
            }
            x = next;
        }
    }

    if out.is_empty() {
        return Err(AbsoluteStrengthIndexOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid_absolute_strength_index_oscillator(
    range: &AbsoluteStrengthIndexOscillatorBatchRange,
) -> Result<Vec<AbsoluteStrengthIndexOscillatorParams>, AbsoluteStrengthIndexOscillatorError> {
    let ema_lengths =
        expand_axis_usize(range.ema_length.0, range.ema_length.1, range.ema_length.2)?;
    let signal_lengths = expand_axis_usize(
        range.signal_length.0,
        range.signal_length.1,
        range.signal_length.2,
    )?;

    let mut combos = Vec::with_capacity(ema_lengths.len().saturating_mul(signal_lengths.len()));
    for &ema_length in &ema_lengths {
        if ema_length == 0 {
            return Err(AbsoluteStrengthIndexOscillatorError::InvalidEmaLength { ema_length });
        }
        for &signal_length in &signal_lengths {
            if signal_length <= 1 {
                return Err(AbsoluteStrengthIndexOscillatorError::InvalidSignalLength {
                    signal_length,
                });
            }
            combos.push(AbsoluteStrengthIndexOscillatorParams {
                ema_length: Some(ema_length),
                signal_length: Some(signal_length),
            });
        }
    }
    Ok(combos)
}

#[inline]
pub fn absolute_strength_index_oscillator_batch_with_kernel(
    data: &[f64],
    sweep: &AbsoluteStrengthIndexOscillatorBatchRange,
    kernel: Kernel,
) -> Result<AbsoluteStrengthIndexOscillatorBatchOutput, AbsoluteStrengthIndexOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => {
            return Err(AbsoluteStrengthIndexOscillatorError::InvalidKernelForBatch(
                other,
            ))
        }
    };
    absolute_strength_index_oscillator_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn absolute_strength_index_oscillator_batch_slice(
    data: &[f64],
    sweep: &AbsoluteStrengthIndexOscillatorBatchRange,
    kernel: Kernel,
) -> Result<AbsoluteStrengthIndexOscillatorBatchOutput, AbsoluteStrengthIndexOscillatorError> {
    absolute_strength_index_oscillator_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn absolute_strength_index_oscillator_batch_par_slice(
    data: &[f64],
    sweep: &AbsoluteStrengthIndexOscillatorBatchRange,
    kernel: Kernel,
) -> Result<AbsoluteStrengthIndexOscillatorBatchOutput, AbsoluteStrengthIndexOscillatorError> {
    absolute_strength_index_oscillator_batch_inner(data, sweep, kernel, true)
}

fn absolute_strength_index_oscillator_batch_inner(
    data: &[f64],
    sweep: &AbsoluteStrengthIndexOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<AbsoluteStrengthIndexOscillatorBatchOutput, AbsoluteStrengthIndexOscillatorError> {
    let combos = expand_grid_absolute_strength_index_oscillator(sweep)?;
    if data.is_empty() {
        return Err(AbsoluteStrengthIndexOscillatorError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(AbsoluteStrengthIndexOscillatorError::AllValuesNaN);
    }

    let rows = combos.len();
    let cols = data.len();
    let total = rows.checked_mul(cols).ok_or_else(|| {
        AbsoluteStrengthIndexOscillatorError::OutputLengthMismatch {
            expected: usize::MAX,
            oscillator_got: 0,
            signal_got: 0,
            histogram_got: 0,
        }
    })?;
    let warms = vec![first; rows];

    let mut oscillator_mu = make_uninit_matrix(rows, cols);
    let mut signal_mu = make_uninit_matrix(rows, cols);
    let mut histogram_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut oscillator_mu, cols, &warms);
    init_matrix_prefixes(&mut signal_mu, cols, &warms);
    init_matrix_prefixes(&mut histogram_mu, cols, &warms);

    let mut oscillator_guard = ManuallyDrop::new(oscillator_mu);
    let mut signal_guard = ManuallyDrop::new(signal_mu);
    let mut histogram_guard = ManuallyDrop::new(histogram_mu);

    let oscillator_out =
        unsafe { std::slice::from_raw_parts_mut(oscillator_guard.as_mut_ptr() as *mut f64, total) };
    let signal_out =
        unsafe { std::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, total) };
    let histogram_out =
        unsafe { std::slice::from_raw_parts_mut(histogram_guard.as_mut_ptr() as *mut f64, total) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            oscillator_out
                .par_chunks_mut(cols)
                .zip(signal_out.par_chunks_mut(cols))
                .zip(histogram_out.par_chunks_mut(cols))
                .zip(combos.par_iter())
                .for_each(|(((osc_row, sig_row), hist_row), combo)| {
                    let params = resolve_params(combo).unwrap();
                    absolute_strength_index_oscillator_row_from_slice(
                        data, params, osc_row, sig_row, hist_row,
                    );
                });
        }

        #[cfg(target_arch = "wasm32")]
        for (row, combo) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            let params = resolve_params(combo).unwrap();
            absolute_strength_index_oscillator_row_from_slice(
                data,
                params,
                &mut oscillator_out[start..end],
                &mut signal_out[start..end],
                &mut histogram_out[start..end],
            );
        }
    } else {
        for (row, combo) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            let params = resolve_params(combo).unwrap();
            absolute_strength_index_oscillator_row_from_slice(
                data,
                params,
                &mut oscillator_out[start..end],
                &mut signal_out[start..end],
                &mut histogram_out[start..end],
            );
        }
    }

    let oscillator = unsafe {
        Vec::from_raw_parts(
            oscillator_guard.as_mut_ptr() as *mut f64,
            oscillator_guard.len(),
            oscillator_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };
    let histogram = unsafe {
        Vec::from_raw_parts(
            histogram_guard.as_mut_ptr() as *mut f64,
            histogram_guard.len(),
            histogram_guard.capacity(),
        )
    };
    core::mem::forget(oscillator_guard);
    core::mem::forget(signal_guard);
    core::mem::forget(histogram_guard);

    Ok(AbsoluteStrengthIndexOscillatorBatchOutput {
        oscillator,
        signal,
        histogram,
        combos,
        rows,
        cols,
    })
}

pub fn absolute_strength_index_oscillator_batch_inner_into(
    data: &[f64],
    sweep: &AbsoluteStrengthIndexOscillatorBatchRange,
    kernel: Kernel,
    oscillator_out: &mut [f64],
    signal_out: &mut [f64],
    histogram_out: &mut [f64],
) -> Result<Vec<AbsoluteStrengthIndexOscillatorParams>, AbsoluteStrengthIndexOscillatorError> {
    let out = absolute_strength_index_oscillator_batch_inner(data, sweep, kernel, false)?;
    let total = out.rows * out.cols;
    if oscillator_out.len() != total || signal_out.len() != total || histogram_out.len() != total {
        return Err(AbsoluteStrengthIndexOscillatorError::OutputLengthMismatch {
            expected: total,
            oscillator_got: oscillator_out.len(),
            signal_got: signal_out.len(),
            histogram_got: histogram_out.len(),
        });
    }
    oscillator_out.copy_from_slice(&out.oscillator);
    signal_out.copy_from_slice(&out.signal);
    histogram_out.copy_from_slice(&out.histogram);
    Ok(out.combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "absolute_strength_index_oscillator")]
#[pyo3(signature = (data, ema_length=None, signal_length=None, kernel=None))]
pub fn absolute_strength_index_oscillator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    ema_length: Option<usize>,
    signal_length: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = AbsoluteStrengthIndexOscillatorInput::from_slice(
        data,
        AbsoluteStrengthIndexOscillatorParams {
            ema_length,
            signal_length,
        },
    );
    let out = py
        .allow_threads(|| absolute_strength_index_oscillator_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        out.oscillator.into_pyarray(py),
        out.signal.into_pyarray(py),
        out.histogram.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "AbsoluteStrengthIndexOscillatorStream")]
pub struct AbsoluteStrengthIndexOscillatorStreamPy {
    inner: AbsoluteStrengthIndexOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AbsoluteStrengthIndexOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (ema_length=DEFAULT_EMA_LENGTH, signal_length=DEFAULT_SIGNAL_LENGTH))]
    fn new(ema_length: usize, signal_length: usize) -> PyResult<Self> {
        let inner =
            AbsoluteStrengthIndexOscillatorStream::try_new(AbsoluteStrengthIndexOscillatorParams {
                ema_length: Some(ema_length),
                signal_length: Some(signal_length),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        self.inner.update(value)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.inner.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "absolute_strength_index_oscillator_batch")]
#[pyo3(signature = (data, ema_length_range=(DEFAULT_EMA_LENGTH, DEFAULT_EMA_LENGTH, 0), signal_length_range=(DEFAULT_SIGNAL_LENGTH, DEFAULT_SIGNAL_LENGTH, 0), kernel=None))]
pub fn absolute_strength_index_oscillator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    ema_length_range: (usize, usize, usize),
    signal_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = AbsoluteStrengthIndexOscillatorBatchRange {
        ema_length: ema_length_range,
        signal_length: signal_length_range,
    };
    let combos = expand_grid_absolute_strength_index_oscillator(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let oscillator_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let histogram_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let oscillator_slice = unsafe { oscillator_arr.as_slice_mut()? };
    let signal_slice = unsafe { signal_arr.as_slice_mut()? };
    let histogram_slice = unsafe { histogram_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            absolute_strength_index_oscillator_batch_inner_into(
                data,
                &sweep,
                batch.to_non_batch(),
                oscillator_slice,
                signal_slice,
                histogram_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("oscillator", oscillator_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item("histogram", histogram_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "ema_lengths",
        combos
            .iter()
            .map(|p| p.ema_length.unwrap_or(DEFAULT_EMA_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "signal_lengths",
        combos
            .iter()
            .map(|p| p.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_absolute_strength_index_oscillator_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        absolute_strength_index_oscillator_py,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        absolute_strength_index_oscillator_batch_py,
        module
    )?)?;
    module.add_class::<AbsoluteStrengthIndexOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "absolute_strength_index_oscillator_js")]
pub fn absolute_strength_index_oscillator_js(
    data: &[f64],
    ema_length: usize,
    signal_length: usize,
) -> Result<JsValue, JsValue> {
    let input = AbsoluteStrengthIndexOscillatorInput::from_slice(
        data,
        AbsoluteStrengthIndexOscillatorParams {
            ema_length: Some(ema_length),
            signal_length: Some(signal_length),
        },
    );
    let out = absolute_strength_index_oscillator(&input)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = js_sys::Object::new();

    let oscillator = js_sys::Float64Array::new_with_length(out.oscillator.len() as u32);
    oscillator.copy_from(&out.oscillator);
    js_sys::Reflect::set(&result, &JsValue::from_str("oscillator"), &oscillator)?;

    let signal = js_sys::Float64Array::new_with_length(out.signal.len() as u32);
    signal.copy_from(&out.signal);
    js_sys::Reflect::set(&result, &JsValue::from_str("signal"), &signal)?;

    let histogram = js_sys::Float64Array::new_with_length(out.histogram.len() as u32);
    histogram.copy_from(&out.histogram);
    js_sys::Reflect::set(&result, &JsValue::from_str("histogram"), &histogram)?;

    Ok(result.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn absolute_strength_index_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn absolute_strength_index_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn absolute_strength_index_oscillator_into(
    data_ptr: *const f64,
    oscillator_ptr: *mut f64,
    signal_ptr: *mut f64,
    histogram_ptr: *mut f64,
    len: usize,
    ema_length: usize,
    signal_length: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null()
        || oscillator_ptr.is_null()
        || signal_ptr.is_null()
        || histogram_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let input = AbsoluteStrengthIndexOscillatorInput::from_slice(
            data,
            AbsoluteStrengthIndexOscillatorParams {
                ema_length: Some(ema_length),
                signal_length: Some(signal_length),
            },
        );
        let alias =
            data_ptr == oscillator_ptr || data_ptr == signal_ptr || data_ptr == histogram_ptr;
        if alias {
            let mut oscillator_tmp = vec![0.0; len];
            let mut signal_tmp = vec![0.0; len];
            let mut histogram_tmp = vec![0.0; len];
            absolute_strength_index_oscillator_into_slices(
                &mut oscillator_tmp,
                &mut signal_tmp,
                &mut histogram_tmp,
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(oscillator_ptr, len).copy_from_slice(&oscillator_tmp);
            std::slice::from_raw_parts_mut(signal_ptr, len).copy_from_slice(&signal_tmp);
            std::slice::from_raw_parts_mut(histogram_ptr, len).copy_from_slice(&histogram_tmp);
        } else {
            absolute_strength_index_oscillator_into_slices(
                std::slice::from_raw_parts_mut(oscillator_ptr, len),
                std::slice::from_raw_parts_mut(signal_ptr, len),
                std::slice::from_raw_parts_mut(histogram_ptr, len),
                &input,
                Kernel::Auto,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AbsoluteStrengthIndexOscillatorBatchConfig {
    pub ema_length_range: (usize, usize, usize),
    pub signal_length_range: Option<(usize, usize, usize)>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AbsoluteStrengthIndexOscillatorBatchJsOutput {
    pub oscillator: Vec<f64>,
    pub signal: Vec<f64>,
    pub histogram: Vec<f64>,
    pub combos: Vec<AbsoluteStrengthIndexOscillatorParams>,
    pub ema_lengths: Vec<usize>,
    pub signal_lengths: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "absolute_strength_index_oscillator_batch_js")]
pub fn absolute_strength_index_oscillator_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: AbsoluteStrengthIndexOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = AbsoluteStrengthIndexOscillatorBatchRange {
        ema_length: config.ema_length_range,
        signal_length: config.signal_length_range.unwrap_or((
            DEFAULT_SIGNAL_LENGTH,
            DEFAULT_SIGNAL_LENGTH,
            0,
        )),
    };
    let out = absolute_strength_index_oscillator_batch_inner(
        data,
        &sweep,
        detect_best_batch_kernel().to_non_batch(),
        false,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&AbsoluteStrengthIndexOscillatorBatchJsOutput {
        ema_lengths: out
            .combos
            .iter()
            .map(|p| p.ema_length.unwrap_or(DEFAULT_EMA_LENGTH))
            .collect(),
        signal_lengths: out
            .combos
            .iter()
            .map(|p| p.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH))
            .collect(),
        oscillator: out.oscillator,
        signal: out.signal,
        histogram: out.histogram,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn absolute_strength_index_oscillator_batch_into(
    data_ptr: *const f64,
    oscillator_ptr: *mut f64,
    signal_ptr: *mut f64,
    histogram_ptr: *mut f64,
    len: usize,
    ema_start: usize,
    ema_end: usize,
    ema_step: usize,
    signal_start: usize,
    signal_end: usize,
    signal_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null()
        || oscillator_ptr.is_null()
        || signal_ptr.is_null()
        || histogram_ptr.is_null()
    {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = AbsoluteStrengthIndexOscillatorBatchRange {
        ema_length: (ema_start, ema_end, ema_step),
        signal_length: (signal_start, signal_end, signal_step),
    };
    let combos = expand_grid_absolute_strength_index_oscillator(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let oscillator_out = std::slice::from_raw_parts_mut(oscillator_ptr, total);
        let signal_out = std::slice::from_raw_parts_mut(signal_ptr, total);
        let histogram_out = std::slice::from_raw_parts_mut(histogram_ptr, total);
        absolute_strength_index_oscillator_batch_inner_into(
            data,
            &sweep,
            detect_best_batch_kernel().to_non_batch(),
            oscillator_out,
            signal_out,
            histogram_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn absolute_strength_index_oscillator_output_into_js(
    data: &[f64],
    ema_length: usize,
    signal_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = absolute_strength_index_oscillator_js(data, ema_length, signal_length)?;
    crate::write_wasm_object_f64_outputs(
        "absolute_strength_index_oscillator_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn absolute_strength_index_oscillator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = absolute_strength_index_oscillator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "absolute_strength_index_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn absolute_strength_index_oscillator_manual_contract() -> Result<(), Box<dyn Error>> {
        let data = [10.0, 10.0];
        let input = AbsoluteStrengthIndexOscillatorInput::from_slice(
            &data,
            AbsoluteStrengthIndexOscillatorParams {
                ema_length: Some(2),
                signal_length: Some(2),
            },
        );
        let out = absolute_strength_index_oscillator(&input)?;

        assert_eq!(out.oscillator.len(), 2);
        assert!((out.oscillator[0] - 0.0).abs() <= 1e-12);
        assert!((out.signal[0] - 0.0).abs() <= 1e-12);
        assert!((out.histogram[0] - 0.0).abs() <= 1e-12);
        assert!((out.oscillator[1] + 1.0 / 6.0).abs() <= 1e-12);
        assert!((out.signal[1] + 2.0 / 9.0).abs() <= 1e-12);
        assert!((out.histogram[1] - 1.0 / 18.0).abs() <= 1e-12);
        Ok(())
    }

    #[test]
    fn absolute_strength_index_oscillator_rejects_invalid_signal_length() {
        let data = [1.0, 2.0, 3.0];
        let input = AbsoluteStrengthIndexOscillatorInput::from_slice(
            &data,
            AbsoluteStrengthIndexOscillatorParams {
                ema_length: Some(2),
                signal_length: Some(1),
            },
        );
        let err = absolute_strength_index_oscillator(&input).unwrap_err();
        assert!(matches!(
            err,
            AbsoluteStrengthIndexOscillatorError::InvalidSignalLength { signal_length: 1 }
        ));
    }

    #[test]
    fn absolute_strength_index_oscillator_stream_matches_batch_with_reset(
    ) -> Result<(), Box<dyn Error>> {
        let data = [10.0, 10.0, 9.0, f64::NAN, 10.0, 10.0];
        let input = AbsoluteStrengthIndexOscillatorInput::from_slice(
            &data,
            AbsoluteStrengthIndexOscillatorParams {
                ema_length: Some(2),
                signal_length: Some(2),
            },
        );
        let out = absolute_strength_index_oscillator(&input)?;
        let mut stream = AbsoluteStrengthIndexOscillatorStream::try_new(
            AbsoluteStrengthIndexOscillatorParams {
                ema_length: Some(2),
                signal_length: Some(2),
            },
        )?;

        let mut osc = Vec::new();
        let mut sig = Vec::new();
        let mut hist = Vec::new();
        for &value in &data {
            match stream.update(value) {
                Some((o, s, h)) => {
                    osc.push(o);
                    sig.push(s);
                    hist.push(h);
                }
                None => {
                    osc.push(f64::NAN);
                    sig.push(f64::NAN);
                    hist.push(f64::NAN);
                }
            }
        }

        for (lhs, rhs) in out.oscillator.iter().zip(osc.iter()) {
            if lhs.is_nan() || rhs.is_nan() {
                assert!(lhs.is_nan() && rhs.is_nan());
            } else {
                assert!((lhs - rhs).abs() <= 1e-12);
            }
        }
        for (lhs, rhs) in out.signal.iter().zip(sig.iter()) {
            if lhs.is_nan() || rhs.is_nan() {
                assert!(lhs.is_nan() && rhs.is_nan());
            } else {
                assert!((lhs - rhs).abs() <= 1e-12);
            }
        }
        for (lhs, rhs) in out.histogram.iter().zip(hist.iter()) {
            if lhs.is_nan() || rhs.is_nan() {
                assert!(lhs.is_nan() && rhs.is_nan());
            } else {
                assert!((lhs - rhs).abs() <= 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn absolute_strength_index_oscillator_into_matches_api() -> Result<(), Box<dyn Error>> {
        let data = [10.0, 10.0, 9.0, 9.0, 10.0];
        let input = AbsoluteStrengthIndexOscillatorInput::from_slice(
            &data,
            AbsoluteStrengthIndexOscillatorParams {
                ema_length: Some(3),
                signal_length: Some(4),
            },
        );
        let out = absolute_strength_index_oscillator(&input)?;
        let mut oscillator = vec![f64::NAN; data.len()];
        let mut signal = vec![f64::NAN; data.len()];
        let mut histogram = vec![f64::NAN; data.len()];

        absolute_strength_index_oscillator_into(
            &input,
            &mut oscillator,
            &mut signal,
            &mut histogram,
        )?;

        assert_eq!(oscillator, out.oscillator);
        assert_eq!(signal, out.signal);
        assert_eq!(histogram, out.histogram);
        Ok(())
    }
}
