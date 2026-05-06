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

use crate::indicators::rsi::{RsiParams, RsiStream};
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::convert::AsRef;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_RSI_LENGTH: usize = 14;
const DEFAULT_STOCH_LENGTH: usize = 8;
const DEFAULT_SMOOTH_LENGTH: usize = 25;
const FLOAT_TOL: f64 = 1e-12;

impl<'a> AsRef<[f64]> for PremierRsiOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PremierRsiOscillatorData::Slice(slice) => slice,
            PremierRsiOscillatorData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PremierRsiOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PremierRsiOscillatorOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct PremierRsiOscillatorParams {
    pub rsi_length: Option<usize>,
    pub stoch_length: Option<usize>,
    pub smooth_length: Option<usize>,
}

impl Default for PremierRsiOscillatorParams {
    fn default() -> Self {
        Self {
            rsi_length: Some(DEFAULT_RSI_LENGTH),
            stoch_length: Some(DEFAULT_STOCH_LENGTH),
            smooth_length: Some(DEFAULT_SMOOTH_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PremierRsiOscillatorInput<'a> {
    pub data: PremierRsiOscillatorData<'a>,
    pub params: PremierRsiOscillatorParams,
}

impl<'a> PremierRsiOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: PremierRsiOscillatorParams,
    ) -> Self {
        Self {
            data: PremierRsiOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: PremierRsiOscillatorParams) -> Self {
        Self {
            data: PremierRsiOscillatorData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", PremierRsiOscillatorParams::default())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PremierRsiOscillatorBuilder {
    rsi_length: Option<usize>,
    stoch_length: Option<usize>,
    smooth_length: Option<usize>,
    kernel: Kernel,
}

impl Default for PremierRsiOscillatorBuilder {
    fn default() -> Self {
        Self {
            rsi_length: None,
            stoch_length: None,
            smooth_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PremierRsiOscillatorBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn rsi_length(mut self, rsi_length: usize) -> Self {
        self.rsi_length = Some(rsi_length);
        self
    }

    #[inline]
    pub fn stoch_length(mut self, stoch_length: usize) -> Self {
        self.stoch_length = Some(stoch_length);
        self
    }

    #[inline]
    pub fn smooth_length(mut self, smooth_length: usize) -> Self {
        self.smooth_length = Some(smooth_length);
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
    ) -> Result<PremierRsiOscillatorOutput, PremierRsiOscillatorError> {
        let input = PremierRsiOscillatorInput::from_candles(
            candles,
            source,
            PremierRsiOscillatorParams {
                rsi_length: self.rsi_length,
                stoch_length: self.stoch_length,
                smooth_length: self.smooth_length,
            },
        );
        premier_rsi_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<PremierRsiOscillatorOutput, PremierRsiOscillatorError> {
        let input = PremierRsiOscillatorInput::from_slice(
            data,
            PremierRsiOscillatorParams {
                rsi_length: self.rsi_length,
                stoch_length: self.stoch_length,
                smooth_length: self.smooth_length,
            },
        );
        premier_rsi_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(self) -> Result<PremierRsiOscillatorStream, PremierRsiOscillatorError> {
        PremierRsiOscillatorStream::try_new(PremierRsiOscillatorParams {
            rsi_length: self.rsi_length,
            stoch_length: self.stoch_length,
            smooth_length: self.smooth_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum PremierRsiOscillatorError {
    #[error("premier_rsi_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error("premier_rsi_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error("premier_rsi_oscillator: Invalid rsi_length: {rsi_length}")]
    InvalidRsiLength { rsi_length: usize },
    #[error("premier_rsi_oscillator: Invalid stoch_length: {stoch_length}")]
    InvalidStochLength { stoch_length: usize },
    #[error("premier_rsi_oscillator: Invalid smooth_length: {smooth_length}")]
    InvalidSmoothLength { smooth_length: usize },
    #[error("premier_rsi_oscillator: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("premier_rsi_oscillator: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("premier_rsi_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("premier_rsi_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    rsi_length: usize,
    stoch_length: usize,
    smooth_length: usize,
    ema_length: usize,
    ema_alpha: f64,
}

#[derive(Debug, Clone)]
pub struct PremierRsiOscillatorStream {
    params: ResolvedParams,
    rsi_stream: RsiStream,
    rsi_index: usize,
    max_q: VecDeque<(usize, f64)>,
    min_q: VecDeque<(usize, f64)>,
    ema1: Option<f64>,
    ema2: Option<f64>,
}

impl PremierRsiOscillatorStream {
    pub fn try_new(params: PremierRsiOscillatorParams) -> Result<Self, PremierRsiOscillatorError> {
        let params = resolve_params(&params, 0)?;
        Ok(Self::new_resolved(params))
    }

    #[inline]
    fn new_resolved(params: ResolvedParams) -> Self {
        Self {
            params,
            rsi_stream: RsiStream::try_new(RsiParams {
                period: Some(params.rsi_length),
            })
            .expect("resolved RSI params must be valid"),
            rsi_index: 0,
            max_q: VecDeque::with_capacity(params.stoch_length),
            min_q: VecDeque::with_capacity(params.stoch_length),
            ema1: None,
            ema2: None,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new_resolved(self.params);
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        self.params.rsi_length + self.params.stoch_length - 1
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }

        let rsi = self.rsi_stream.update(value)?;
        if !rsi.is_finite() {
            self.reset();
            return None;
        }

        while let Some(&(_, back)) = self.max_q.back() {
            if back <= rsi {
                self.max_q.pop_back();
            } else {
                break;
            }
        }
        self.max_q.push_back((self.rsi_index, rsi));

        while let Some(&(_, back)) = self.min_q.back() {
            if back >= rsi {
                self.min_q.pop_back();
            } else {
                break;
            }
        }
        self.min_q.push_back((self.rsi_index, rsi));

        let window_start = self
            .rsi_index
            .saturating_add(1)
            .saturating_sub(self.params.stoch_length);

        while let Some(&(idx, _)) = self.max_q.front() {
            if idx < window_start {
                self.max_q.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(idx, _)) = self.min_q.front() {
            if idx < window_start {
                self.min_q.pop_front();
            } else {
                break;
            }
        }

        self.rsi_index += 1;
        if self.rsi_index < self.params.stoch_length {
            return None;
        }

        let highest = self.max_q.front().map(|(_, v)| *v).unwrap_or(rsi);
        let lowest = self.min_q.front().map(|(_, v)| *v).unwrap_or(rsi);
        let denom = highest - lowest;
        let sk = if denom.abs() <= FLOAT_TOL {
            50.0
        } else {
            (rsi - lowest).mul_add(100.0 / denom, 0.0)
        };
        let nsk = 0.1 * (sk - 50.0);

        let ema1 = match self.ema1 {
            Some(prev) => self.params.ema_alpha * nsk + (1.0 - self.params.ema_alpha) * prev,
            None => nsk,
        };
        self.ema1 = Some(ema1);

        let ema2 = match self.ema2 {
            Some(prev) => self.params.ema_alpha * ema1 + (1.0 - self.params.ema_alpha) * prev,
            None => ema1,
        };
        self.ema2 = Some(ema2);

        Some((ema2 * 0.5).tanh())
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct PremierRsiOscillatorBatchRange {
    pub rsi_length: (usize, usize, usize),
    pub stoch_length: (usize, usize, usize),
    pub smooth_length: (usize, usize, usize),
}

impl Default for PremierRsiOscillatorBatchRange {
    fn default() -> Self {
        Self {
            rsi_length: (DEFAULT_RSI_LENGTH, DEFAULT_RSI_LENGTH, 0),
            stoch_length: (DEFAULT_STOCH_LENGTH, DEFAULT_STOCH_LENGTH, 0),
            smooth_length: (DEFAULT_SMOOTH_LENGTH, DEFAULT_SMOOTH_LENGTH, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PremierRsiOscillatorBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PremierRsiOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug, Default)]
pub struct PremierRsiOscillatorBatchBuilder {
    range: PremierRsiOscillatorBatchRange,
    kernel: Kernel,
}

impl PremierRsiOscillatorBatchBuilder {
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
    pub fn rsi_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.rsi_length = (start, end, step);
        self
    }

    #[inline]
    pub fn stoch_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.stoch_length = (start, end, step);
        self
    }

    #[inline]
    pub fn smooth_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.smooth_length = (start, end, step);
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<PremierRsiOscillatorBatchOutput, PremierRsiOscillatorError> {
        premier_rsi_oscillator_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<PremierRsiOscillatorBatchOutput, PremierRsiOscillatorError> {
        self.apply_slice(source_type(candles, source))
    }
}

#[inline]
pub fn premier_rsi_oscillator(
    input: &PremierRsiOscillatorInput,
) -> Result<PremierRsiOscillatorOutput, PremierRsiOscillatorError> {
    premier_rsi_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline]
fn first_valid_value(data: &[f64]) -> usize {
    data.iter()
        .position(|v| v.is_finite())
        .unwrap_or(data.len())
}

#[inline]
fn count_valid_values(data: &[f64]) -> usize {
    data.iter().filter(|&&v| v.is_finite()).count()
}

#[inline]
fn resolve_params(
    params: &PremierRsiOscillatorParams,
    _data_len: usize,
) -> Result<ResolvedParams, PremierRsiOscillatorError> {
    let rsi_length = params.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH);
    if rsi_length == 0 {
        return Err(PremierRsiOscillatorError::InvalidRsiLength { rsi_length });
    }

    let stoch_length = params.stoch_length.unwrap_or(DEFAULT_STOCH_LENGTH);
    if stoch_length == 0 {
        return Err(PremierRsiOscillatorError::InvalidStochLength { stoch_length });
    }

    let smooth_length = params.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH);
    if smooth_length == 0 {
        return Err(PremierRsiOscillatorError::InvalidSmoothLength { smooth_length });
    }

    let ema_length = ((smooth_length as f64).sqrt().round() as usize).max(1);
    Ok(ResolvedParams {
        rsi_length,
        stoch_length,
        smooth_length,
        ema_length,
        ema_alpha: 2.0 / (ema_length as f64 + 1.0),
    })
}

#[inline]
fn prepare<'a>(
    input: &'a PremierRsiOscillatorInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], ResolvedParams, usize, bool, Kernel), PremierRsiOscillatorError> {
    let data = input.as_ref();
    if data.is_empty() {
        return Err(PremierRsiOscillatorError::EmptyInputData);
    }

    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(PremierRsiOscillatorError::AllValuesNaN);
    }

    let params = resolve_params(&input.params, data.len())?;
    let needed = params.rsi_length + params.stoch_length - 1;
    let valid = count_valid_values(data);
    if valid < needed {
        return Err(PremierRsiOscillatorError::NotEnoughValidData { needed, valid });
    }
    let all_finite_after_first = valid == data.len() - first;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };
    Ok((data, params, first, all_finite_after_first, chosen))
}

#[inline]
fn row_from_slice(data: &[f64], params: ResolvedParams, out: &mut [f64]) {
    out.fill(f64::NAN);
    row_from_slice_prefilled(data, params, out);
}

#[inline]
fn row_from_slice_prefilled(data: &[f64], params: ResolvedParams, out: &mut [f64]) {
    let mut stream = PremierRsiOscillatorStream::new_resolved(params);
    for (i, &value) in data.iter().enumerate() {
        if let Some(out_value) = stream.update(value) {
            out[i] = out_value;
        }
    }
}

#[inline]
pub fn premier_rsi_oscillator_with_kernel(
    input: &PremierRsiOscillatorInput,
    kernel: Kernel,
) -> Result<PremierRsiOscillatorOutput, PremierRsiOscillatorError> {
    let (data, params, first, all_finite_after_first, _chosen) = prepare(input, kernel)?;
    let warmup = first + params.rsi_length + params.stoch_length - 1;
    let mut values = alloc_with_nan_prefix(data.len(), warmup);
    if all_finite_after_first {
        row_from_slice_prefilled(data, params, &mut values);
    } else {
        row_from_slice(data, params, &mut values);
    }
    Ok(PremierRsiOscillatorOutput { values })
}

#[inline]
pub fn premier_rsi_oscillator_into_slice(
    dst: &mut [f64],
    input: &PremierRsiOscillatorInput,
    kernel: Kernel,
) -> Result<(), PremierRsiOscillatorError> {
    let (data, params, first, all_finite_after_first, _chosen) = prepare(input, kernel)?;
    if dst.len() != data.len() {
        return Err(PremierRsiOscillatorError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }
    if all_finite_after_first {
        let warmup = first + params.rsi_length + params.stoch_length - 1;
        for value in &mut dst[..warmup] {
            *value = f64::NAN;
        }
        row_from_slice_prefilled(data, params, dst);
    } else {
        row_from_slice(data, params, dst);
    }
    Ok(())
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn premier_rsi_oscillator_into(
    input: &PremierRsiOscillatorInput,
    out: &mut [f64],
) -> Result<(), PremierRsiOscillatorError> {
    premier_rsi_oscillator_into_slice(out, input, Kernel::Auto)
}

fn expand_axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, PremierRsiOscillatorError> {
    if start == end {
        return Ok(vec![start]);
    }
    if step == 0 {
        return Err(PremierRsiOscillatorError::InvalidRange {
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
        return Err(PremierRsiOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

fn expand_grid_premier_rsi_oscillator(
    range: &PremierRsiOscillatorBatchRange,
) -> Result<Vec<PremierRsiOscillatorParams>, PremierRsiOscillatorError> {
    let rsi_lengths =
        expand_axis_usize(range.rsi_length.0, range.rsi_length.1, range.rsi_length.2)?;
    let stoch_lengths = expand_axis_usize(
        range.stoch_length.0,
        range.stoch_length.1,
        range.stoch_length.2,
    )?;
    let smooth_lengths = expand_axis_usize(
        range.smooth_length.0,
        range.smooth_length.1,
        range.smooth_length.2,
    )?;

    let mut combos = Vec::with_capacity(
        rsi_lengths
            .len()
            .saturating_mul(stoch_lengths.len())
            .saturating_mul(smooth_lengths.len()),
    );
    for &rsi_length in &rsi_lengths {
        if rsi_length == 0 {
            return Err(PremierRsiOscillatorError::InvalidRsiLength { rsi_length });
        }
        for &stoch_length in &stoch_lengths {
            if stoch_length == 0 {
                return Err(PremierRsiOscillatorError::InvalidStochLength { stoch_length });
            }
            for &smooth_length in &smooth_lengths {
                if smooth_length == 0 {
                    return Err(PremierRsiOscillatorError::InvalidSmoothLength { smooth_length });
                }
                combos.push(PremierRsiOscillatorParams {
                    rsi_length: Some(rsi_length),
                    stoch_length: Some(stoch_length),
                    smooth_length: Some(smooth_length),
                });
            }
        }
    }
    Ok(combos)
}

#[inline]
pub fn premier_rsi_oscillator_batch_with_kernel(
    data: &[f64],
    sweep: &PremierRsiOscillatorBatchRange,
    kernel: Kernel,
) -> Result<PremierRsiOscillatorBatchOutput, PremierRsiOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(PremierRsiOscillatorError::InvalidKernelForBatch(other)),
    };
    premier_rsi_oscillator_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline]
pub fn premier_rsi_oscillator_batch_slice(
    data: &[f64],
    sweep: &PremierRsiOscillatorBatchRange,
    kernel: Kernel,
) -> Result<PremierRsiOscillatorBatchOutput, PremierRsiOscillatorError> {
    premier_rsi_oscillator_batch_inner(data, sweep, kernel, false)
}

#[inline]
pub fn premier_rsi_oscillator_batch_par_slice(
    data: &[f64],
    sweep: &PremierRsiOscillatorBatchRange,
    kernel: Kernel,
) -> Result<PremierRsiOscillatorBatchOutput, PremierRsiOscillatorError> {
    premier_rsi_oscillator_batch_inner(data, sweep, kernel, true)
}

fn premier_rsi_oscillator_batch_inner(
    data: &[f64],
    sweep: &PremierRsiOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
) -> Result<PremierRsiOscillatorBatchOutput, PremierRsiOscillatorError> {
    let combos = expand_grid_premier_rsi_oscillator(sweep)?;
    if data.is_empty() {
        return Err(PremierRsiOscillatorError::EmptyInputData);
    }
    let first = first_valid_value(data);
    if first >= data.len() {
        return Err(PremierRsiOscillatorError::AllValuesNaN);
    }

    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(PremierRsiOscillatorError::OutputLengthMismatch {
            expected: usize::MAX,
            got: 0,
        })?;

    let warmups = combos
        .iter()
        .map(|combo| {
            let params = resolve_params(combo, cols)?;
            Ok(first + params.rsi_length + params.stoch_length - 1)
        })
        .collect::<Result<Vec<_>, PremierRsiOscillatorError>>()?;

    let mut values_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut values_mu, cols, &warmups);
    let mut values_guard = ManuallyDrop::new(values_mu);
    let values_out =
        unsafe { std::slice::from_raw_parts_mut(values_guard.as_mut_ptr() as *mut f64, total) };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            values_out
                .par_chunks_mut(cols)
                .zip(combos.par_iter())
                .for_each(|(row, combo)| {
                    let params = resolve_params(combo, cols).unwrap();
                    row_from_slice(data, params, row);
                });
        }

        #[cfg(target_arch = "wasm32")]
        for (row, combo) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            let params = resolve_params(combo, cols).unwrap();
            row_from_slice(data, params, &mut values_out[start..end]);
        }
    } else {
        for (row, combo) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            let params = resolve_params(combo, cols).unwrap();
            row_from_slice(data, params, &mut values_out[start..end]);
        }
    }

    let values = unsafe {
        Vec::from_raw_parts(
            values_guard.as_mut_ptr() as *mut f64,
            values_guard.len(),
            values_guard.capacity(),
        )
    };
    core::mem::forget(values_guard);

    Ok(PremierRsiOscillatorBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn premier_rsi_oscillator_batch_inner_into(
    data: &[f64],
    sweep: &PremierRsiOscillatorBatchRange,
    kernel: Kernel,
    values_out: &mut [f64],
) -> Result<Vec<PremierRsiOscillatorParams>, PremierRsiOscillatorError> {
    let out = premier_rsi_oscillator_batch_inner(data, sweep, kernel, false)?;
    let total = out.rows * out.cols;
    if values_out.len() != total {
        return Err(PremierRsiOscillatorError::OutputLengthMismatch {
            expected: total,
            got: values_out.len(),
        });
    }
    values_out.copy_from_slice(&out.values);
    Ok(out.combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "premier_rsi_oscillator")]
#[pyo3(signature = (data, rsi_length=None, stoch_length=None, smooth_length=None, kernel=None))]
pub fn premier_rsi_oscillator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_length: Option<usize>,
    stoch_length: Option<usize>,
    smooth_length: Option<usize>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let input = PremierRsiOscillatorInput::from_slice(
        data,
        PremierRsiOscillatorParams {
            rsi_length,
            stoch_length,
            smooth_length,
        },
    );
    let out = py
        .allow_threads(|| premier_rsi_oscillator_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "PremierRsiOscillatorStream")]
pub struct PremierRsiOscillatorStreamPy {
    inner: PremierRsiOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PremierRsiOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (rsi_length=DEFAULT_RSI_LENGTH, stoch_length=DEFAULT_STOCH_LENGTH, smooth_length=DEFAULT_SMOOTH_LENGTH))]
    fn new(rsi_length: usize, stoch_length: usize, smooth_length: usize) -> PyResult<Self> {
        let inner = PremierRsiOscillatorStream::try_new(PremierRsiOscillatorParams {
            rsi_length: Some(rsi_length),
            stoch_length: Some(stoch_length),
            smooth_length: Some(smooth_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }

    #[getter]
    fn warmup_period(&self) -> usize {
        self.inner.get_warmup_period()
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "premier_rsi_oscillator_batch")]
#[pyo3(signature = (data, rsi_length_range=(DEFAULT_RSI_LENGTH, DEFAULT_RSI_LENGTH, 0), stoch_length_range=(DEFAULT_STOCH_LENGTH, DEFAULT_STOCH_LENGTH, 0), smooth_length_range=(DEFAULT_SMOOTH_LENGTH, DEFAULT_SMOOTH_LENGTH, 0), kernel=None))]
pub fn premier_rsi_oscillator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    rsi_length_range: (usize, usize, usize),
    stoch_length_range: (usize, usize, usize),
    smooth_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;
    let sweep = PremierRsiOscillatorBatchRange {
        rsi_length: rsi_length_range,
        stoch_length: stoch_length_range,
        smooth_length: smooth_length_range,
    };
    let combos = expand_grid_premier_rsi_oscillator(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let values_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let values_slice = unsafe { values_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let batch = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                other => other,
            };
            premier_rsi_oscillator_batch_inner_into(
                data,
                &sweep,
                batch.to_non_batch(),
                values_slice,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", values_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "rsi_lengths",
        combos
            .iter()
            .map(|p| p.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "stoch_lengths",
        combos
            .iter()
            .map(|p| p.stoch_length.unwrap_or(DEFAULT_STOCH_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "smooth_lengths",
        combos
            .iter()
            .map(|p| p.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_premier_rsi_oscillator_module(
    module: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(premier_rsi_oscillator_py, module)?)?;
    module.add_function(wrap_pyfunction!(premier_rsi_oscillator_batch_py, module)?)?;
    module.add_class::<PremierRsiOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "premier_rsi_oscillator_js")]
pub fn premier_rsi_oscillator_js(
    data: &[f64],
    rsi_length: usize,
    stoch_length: usize,
    smooth_length: usize,
) -> Result<JsValue, JsValue> {
    let input = PremierRsiOscillatorInput::from_slice(
        data,
        PremierRsiOscillatorParams {
            rsi_length: Some(rsi_length),
            stoch_length: Some(stoch_length),
            smooth_length: Some(smooth_length),
        },
    );
    let out = premier_rsi_oscillator(&input).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = js_sys::Object::new();
    let values = js_sys::Float64Array::new_with_length(out.values.len() as u32);
    values.copy_from(&out.values);
    js_sys::Reflect::set(&result, &JsValue::from_str("values"), &values)?;
    Ok(result.into())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn premier_rsi_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn premier_rsi_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn premier_rsi_oscillator_into(
    data_ptr: *const f64,
    values_ptr: *mut f64,
    len: usize,
    rsi_length: usize,
    stoch_length: usize,
    smooth_length: usize,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || values_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let input = PremierRsiOscillatorInput::from_slice(
            data,
            PremierRsiOscillatorParams {
                rsi_length: Some(rsi_length),
                stoch_length: Some(stoch_length),
                smooth_length: Some(smooth_length),
            },
        );
        if data_ptr == values_ptr {
            let mut tmp = vec![0.0; len];
            premier_rsi_oscillator_into_slice(&mut tmp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            std::slice::from_raw_parts_mut(values_ptr, len).copy_from_slice(&tmp);
        } else {
            premier_rsi_oscillator_into_slice(
                std::slice::from_raw_parts_mut(values_ptr, len),
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
pub struct PremierRsiOscillatorBatchConfig {
    pub rsi_length_range: (usize, usize, usize),
    pub stoch_length_range: Option<(usize, usize, usize)>,
    pub smooth_length_range: Option<(usize, usize, usize)>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct PremierRsiOscillatorBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<PremierRsiOscillatorParams>,
    pub rsi_lengths: Vec<usize>,
    pub stoch_lengths: Vec<usize>,
    pub smooth_lengths: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "premier_rsi_oscillator_batch_js")]
pub fn premier_rsi_oscillator_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: PremierRsiOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = PremierRsiOscillatorBatchRange {
        rsi_length: config.rsi_length_range,
        stoch_length: config.stoch_length_range.unwrap_or((
            DEFAULT_STOCH_LENGTH,
            DEFAULT_STOCH_LENGTH,
            0,
        )),
        smooth_length: config.smooth_length_range.unwrap_or((
            DEFAULT_SMOOTH_LENGTH,
            DEFAULT_SMOOTH_LENGTH,
            0,
        )),
    };
    let out = premier_rsi_oscillator_batch_inner(
        data,
        &sweep,
        detect_best_batch_kernel().to_non_batch(),
        false,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&PremierRsiOscillatorBatchJsOutput {
        rsi_lengths: out
            .combos
            .iter()
            .map(|p| p.rsi_length.unwrap_or(DEFAULT_RSI_LENGTH))
            .collect(),
        stoch_lengths: out
            .combos
            .iter()
            .map(|p| p.stoch_length.unwrap_or(DEFAULT_STOCH_LENGTH))
            .collect(),
        smooth_lengths: out
            .combos
            .iter()
            .map(|p| p.smooth_length.unwrap_or(DEFAULT_SMOOTH_LENGTH))
            .collect(),
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn premier_rsi_oscillator_batch_into(
    data_ptr: *const f64,
    values_ptr: *mut f64,
    len: usize,
    rsi_start: usize,
    rsi_end: usize,
    rsi_step: usize,
    stoch_start: usize,
    stoch_end: usize,
    stoch_step: usize,
    smooth_start: usize,
    smooth_end: usize,
    smooth_step: usize,
) -> Result<usize, JsValue> {
    if data_ptr.is_null() || values_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    let sweep = PremierRsiOscillatorBatchRange {
        rsi_length: (rsi_start, rsi_end, rsi_step),
        stoch_length: (stoch_start, stoch_end, stoch_step),
        smooth_length: (smooth_start, smooth_end, smooth_step),
    };
    let combos = expand_grid_premier_rsi_oscillator(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows
        .checked_mul(len)
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let values_out = std::slice::from_raw_parts_mut(values_ptr, total);
        premier_rsi_oscillator_batch_inner_into(
            data,
            &sweep,
            detect_best_batch_kernel().to_non_batch(),
            values_out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn premier_rsi_oscillator_output_into_js(
    data: &[f64],
    rsi_length: usize,
    stoch_length: usize,
    smooth_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = premier_rsi_oscillator_js(data, rsi_length, stoch_length, smooth_length)?;
    crate::write_wasm_object_f64_outputs("premier_rsi_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn premier_rsi_oscillator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = premier_rsi_oscillator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "premier_rsi_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn premier_rsi_oscillator_constant_contract() -> Result<(), Box<dyn Error>> {
        let data = [10.0; 8];
        let input = PremierRsiOscillatorInput::from_slice(
            &data,
            PremierRsiOscillatorParams {
                rsi_length: Some(2),
                stoch_length: Some(2),
                smooth_length: Some(4),
            },
        );
        let out = premier_rsi_oscillator(&input)?;

        assert_eq!(out.values.len(), data.len());
        assert!(out.values[..3].iter().all(|v| v.is_nan()));
        for value in &out.values[3..] {
            assert!(value.abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn premier_rsi_oscillator_rejects_invalid_smooth_length() {
        let data = [1.0, 2.0, 3.0];
        let input = PremierRsiOscillatorInput::from_slice(
            &data,
            PremierRsiOscillatorParams {
                rsi_length: Some(2),
                stoch_length: Some(2),
                smooth_length: Some(0),
            },
        );
        let err = premier_rsi_oscillator(&input).unwrap_err();
        assert!(matches!(
            err,
            PremierRsiOscillatorError::InvalidSmoothLength { smooth_length: 0 }
        ));
    }

    #[test]
    fn premier_rsi_oscillator_stream_matches_batch_with_reset() -> Result<(), Box<dyn Error>> {
        let data = [
            10.0,
            11.0,
            10.5,
            10.0,
            10.2,
            f64::NAN,
            10.0,
            10.0,
            10.1,
            10.2,
        ];
        let params = PremierRsiOscillatorParams {
            rsi_length: Some(2),
            stoch_length: Some(2),
            smooth_length: Some(4),
        };
        let batch = premier_rsi_oscillator(&PremierRsiOscillatorInput::from_slice(
            &data,
            params.clone(),
        ))?;
        let mut stream = PremierRsiOscillatorStream::try_new(params)?;
        let mut streamed = Vec::with_capacity(data.len());

        for &value in &data {
            streamed.push(stream.update(value).unwrap_or(f64::NAN));
        }

        assert_eq!(stream.get_warmup_period(), 3);
        for (lhs, rhs) in batch.values.iter().zip(streamed.iter()) {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!((lhs - rhs).abs() <= 1e-12);
        }
        Ok(())
    }
}
