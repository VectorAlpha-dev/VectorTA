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
use crate::indicators::moving_averages::sma::{SmaParams, SmaStream};
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
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

impl<'a> AsRef<[f64]> for PrettyGoodOscillatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            PrettyGoodOscillatorData::Candles { candles, source } => source_type(candles, source),
            PrettyGoodOscillatorData::Slices { source, .. } => source,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PrettyGoodOscillatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        source: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct PrettyGoodOscillatorOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct PrettyGoodOscillatorParams {
    pub length: Option<usize>,
}

impl Default for PrettyGoodOscillatorParams {
    fn default() -> Self {
        Self { length: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct PrettyGoodOscillatorInput<'a> {
    pub data: PrettyGoodOscillatorData<'a>,
    pub params: PrettyGoodOscillatorParams,
}

impl<'a> PrettyGoodOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: PrettyGoodOscillatorParams,
    ) -> Self {
        Self {
            data: PrettyGoodOscillatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        source: &'a [f64],
        params: PrettyGoodOscillatorParams,
    ) -> Self {
        Self {
            data: PrettyGoodOscillatorData::Slices {
                high,
                low,
                close,
                source,
            },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", PrettyGoodOscillatorParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(14)
    }

    #[inline]
    pub fn as_refs(&'a self) -> (&'a [f64], &'a [f64], &'a [f64], &'a [f64]) {
        match &self.data {
            PrettyGoodOscillatorData::Candles { candles, source } => (
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
                source_type(candles, source),
            ),
            PrettyGoodOscillatorData::Slices {
                high,
                low,
                close,
                source,
            } => (*high, *low, *close, *source),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PrettyGoodOscillatorBuilder {
    length: Option<usize>,
    kernel: Kernel,
}

impl Default for PrettyGoodOscillatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl PrettyGoodOscillatorBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, value: usize) -> Self {
        self.length = Some(value);
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
    ) -> Result<PrettyGoodOscillatorOutput, PrettyGoodOscillatorError> {
        let input = PrettyGoodOscillatorInput::from_candles(
            candles,
            "close",
            PrettyGoodOscillatorParams {
                length: self.length,
            },
        );
        pretty_good_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        source: &[f64],
    ) -> Result<PrettyGoodOscillatorOutput, PrettyGoodOscillatorError> {
        let input = PrettyGoodOscillatorInput::from_slices(
            high,
            low,
            close,
            source,
            PrettyGoodOscillatorParams {
                length: self.length,
            },
        );
        pretty_good_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<PrettyGoodOscillatorStream, PrettyGoodOscillatorError> {
        PrettyGoodOscillatorStream::try_new(PrettyGoodOscillatorParams {
            length: self.length,
        })
    }
}

#[derive(Debug, Error)]
pub enum PrettyGoodOscillatorError {
    #[error("pretty_good_oscillator: Empty input data.")]
    EmptyInputData,
    #[error("pretty_good_oscillator: Data length mismatch across high, low, close, and source.")]
    DataLengthMismatch,
    #[error("pretty_good_oscillator: All OHLC/source values are invalid.")]
    AllValuesNaN,
    #[error("pretty_good_oscillator: Invalid length: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("pretty_good_oscillator: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("pretty_good_oscillator: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("pretty_good_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: usize,
        end: usize,
        step: usize,
    },
    #[error("pretty_good_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn is_valid_bar(high: f64, low: f64, close: f64, source: f64) -> bool {
    high.is_finite() && low.is_finite() && close.is_finite() && source.is_finite() && high >= low
}

#[inline(always)]
fn first_valid_bar(high: &[f64], low: &[f64], close: &[f64], source: &[f64]) -> Option<usize> {
    (0..source.len()).find(|&i| is_valid_bar(high[i], low[i], close[i], source[i]))
}

#[inline(always)]
fn true_range(high: &[f64], low: &[f64], close: &[f64], first: usize, i: usize) -> f64 {
    if i == first {
        high[i] - low[i]
    } else {
        let prev_close = close[i - 1];
        let up = if high[i] > prev_close {
            high[i]
        } else {
            prev_close
        };
        let dn = if low[i] < prev_close {
            low[i]
        } else {
            prev_close
        };
        up - dn
    }
}

#[inline(always)]
fn is_fast_path_clean(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    first: usize,
) -> bool {
    for i in first..source.len() {
        if !is_valid_bar(high[i], low[i], close[i], source[i]) {
            return false;
        }
    }
    true
}

#[inline(always)]
fn pgo_compute_fast(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    length: usize,
    first: usize,
    out: &mut [f64],
) {
    let warmup = first + length - 1;
    let alpha = 1.0 / length as f64;
    let mut sum_source = 0.0;
    let mut sum_tr = 0.0;

    for i in first..=warmup {
        sum_source += source[i];
        sum_tr += true_range(high, low, close, first, i);
    }

    let inv = 1.0 / length as f64;
    let mut sma = sum_source * inv;
    let mut atr = sum_tr * inv;
    out[warmup] = if atr != 0.0 {
        (source[warmup] - sma) / atr
    } else {
        f64::NAN
    };

    for i in (warmup + 1)..source.len() {
        sum_source += source[i] - source[i - length];
        sma = sum_source * inv;
        let tr = true_range(high, low, close, first, i);
        atr = alpha.mul_add(tr - atr, atr);
        out[i] = if atr != 0.0 {
            (source[i] - sma) / atr
        } else {
            f64::NAN
        };
    }
}

#[inline]
fn pgo_prepare<'a>(
    input: &'a PrettyGoodOscillatorInput<'a>,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        &'a [f64],
        usize,
        usize,
        Kernel,
    ),
    PrettyGoodOscillatorError,
> {
    let (high, low, close, source) = input.as_refs();
    if high.is_empty() {
        return Err(PrettyGoodOscillatorError::EmptyInputData);
    }
    if high.len() != low.len() || low.len() != close.len() || close.len() != source.len() {
        return Err(PrettyGoodOscillatorError::DataLengthMismatch);
    }
    let length = input.get_length();
    if length == 0 || length > source.len() {
        return Err(PrettyGoodOscillatorError::InvalidLength {
            length,
            data_len: source.len(),
        });
    }
    let first =
        first_valid_bar(high, low, close, source).ok_or(PrettyGoodOscillatorError::AllValuesNaN)?;
    let valid = source.len().saturating_sub(first);
    if valid < length {
        return Err(PrettyGoodOscillatorError::NotEnoughValidData {
            needed: length,
            valid,
        });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other.to_non_batch(),
    };
    Ok((high, low, close, source, length, first, chosen))
}

#[inline]
pub fn pretty_good_oscillator(
    input: &PrettyGoodOscillatorInput,
) -> Result<PrettyGoodOscillatorOutput, PrettyGoodOscillatorError> {
    pretty_good_oscillator_with_kernel(input, Kernel::Auto)
}

pub fn pretty_good_oscillator_with_kernel(
    input: &PrettyGoodOscillatorInput,
    kernel: Kernel,
) -> Result<PrettyGoodOscillatorOutput, PrettyGoodOscillatorError> {
    let (_, _, _, source, length, first, chosen) = pgo_prepare(input, kernel)?;
    let mut out = alloc_with_nan_prefix(source.len(), first + length - 1);
    pretty_good_oscillator_into_slice(&mut out, input, chosen)?;
    Ok(PrettyGoodOscillatorOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub fn pretty_good_oscillator_into(
    input: &PrettyGoodOscillatorInput,
    out: &mut [f64],
) -> Result<(), PrettyGoodOscillatorError> {
    pretty_good_oscillator_into_slice(out, input, Kernel::Auto)
}

pub fn pretty_good_oscillator_into_slice(
    dst: &mut [f64],
    input: &PrettyGoodOscillatorInput,
    kern: Kernel,
) -> Result<(), PrettyGoodOscillatorError> {
    let (high, low, close, source, length, first, _chosen) = pgo_prepare(input, kern)?;
    if dst.len() != source.len() {
        return Err(PrettyGoodOscillatorError::OutputLengthMismatch {
            expected: source.len(),
            got: dst.len(),
        });
    }

    let warmup = first + length - 1;
    let prefix = warmup.min(dst.len());
    for v in &mut dst[..prefix] {
        *v = f64::NAN;
    }

    if is_fast_path_clean(high, low, close, source, first) {
        pgo_compute_fast(high, low, close, source, length, first, dst);
        return Ok(());
    }

    let mut sma_stream = SmaStream::try_new(SmaParams {
        period: Some(length),
    })
    .map_err(|_| PrettyGoodOscillatorError::InvalidLength {
        length,
        data_len: source.len(),
    })?;
    let mut atr_stream = AtrStream::try_new(AtrParams {
        length: Some(length),
    })
    .map_err(|_| PrettyGoodOscillatorError::InvalidLength {
        length,
        data_len: source.len(),
    })?;

    for i in 0..source.len() {
        if !is_valid_bar(high[i], low[i], close[i], source[i]) {
            dst[i] = f64::NAN;
            continue;
        }
        let sma = sma_stream.update(source[i]);
        let atr = atr_stream.update(high[i], low[i], close[i]);
        dst[i] = match (sma, atr) {
            (Some(sma), Some(atr)) if atr != 0.0 => (source[i] - sma) / atr,
            _ => f64::NAN,
        };
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub struct PrettyGoodOscillatorStream {
    sma: SmaStream,
    atr: AtrStream,
}

impl PrettyGoodOscillatorStream {
    #[inline(always)]
    pub fn try_new(params: PrettyGoodOscillatorParams) -> Result<Self, PrettyGoodOscillatorError> {
        let length = params.length.unwrap_or(14);
        if length == 0 {
            return Err(PrettyGoodOscillatorError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        let sma = SmaStream::try_new(SmaParams {
            period: Some(length),
        })
        .map_err(|_| PrettyGoodOscillatorError::InvalidLength {
            length,
            data_len: 0,
        })?;
        let atr = AtrStream::try_new(AtrParams {
            length: Some(length),
        })
        .map_err(|_| PrettyGoodOscillatorError::InvalidLength {
            length,
            data_len: 0,
        })?;
        Ok(Self { sma, atr })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, close: f64, source: f64) -> Option<f64> {
        if !is_valid_bar(high, low, close, source) {
            return None;
        }
        let sma = self.sma.update(source);
        let atr = self.atr.update(high, low, close);
        match (sma, atr) {
            (Some(sma), Some(atr)) if atr != 0.0 => Some((source - sma) / atr),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PrettyGoodOscillatorBatchRange {
    pub length: (usize, usize, usize),
}

impl Default for PrettyGoodOscillatorBatchRange {
    fn default() -> Self {
        Self {
            length: (14, 14, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PrettyGoodOscillatorBatchBuilder {
    range: PrettyGoodOscillatorBatchRange,
    kernel: Kernel,
}

impl PrettyGoodOscillatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline(always)]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline(always)]
    pub fn length_static(mut self, length: usize) -> Self {
        self.range.length = (length, length, 0);
        self
    }

    #[inline(always)]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<PrettyGoodOscillatorBatchOutput, PrettyGoodOscillatorError> {
        pretty_good_oscillator_batch_with_kernel(
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            source_type(candles, source),
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        source: &[f64],
    ) -> Result<PrettyGoodOscillatorBatchOutput, PrettyGoodOscillatorError> {
        pretty_good_oscillator_batch_with_kernel(high, low, close, source, &self.range, self.kernel)
    }
}

#[derive(Debug, Clone)]
pub struct PrettyGoodOscillatorBatchOutput {
    pub values: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
    pub combos: Vec<PrettyGoodOscillatorParams>,
}

impl PrettyGoodOscillatorBatchOutput {
    #[inline(always)]
    pub fn row_for_params(&self, params: &PrettyGoodOscillatorParams) -> Option<usize> {
        self.combos.iter().position(|p| p.length == params.length)
    }

    #[inline(always)]
    pub fn values_for(&self, params: &PrettyGoodOscillatorParams) -> Option<&[f64]> {
        let row = self.row_for_params(params)?;
        let start = row * self.cols;
        Some(&self.values[start..start + self.cols])
    }
}

#[inline]
pub fn expand_grid_pretty_good_oscillator(
    sweep: &PrettyGoodOscillatorBatchRange,
) -> Result<Vec<PrettyGoodOscillatorParams>, PrettyGoodOscillatorError> {
    let (start, end, step) = sweep.length;
    if start == 0 || end == 0 || start > end || (start != end && step == 0) {
        return Err(PrettyGoodOscillatorError::InvalidRange { start, end, step });
    }
    let mut combos = Vec::new();
    let mut value = start;
    loop {
        combos.push(PrettyGoodOscillatorParams {
            length: Some(value),
        });
        if value == end {
            break;
        }
        value = value.saturating_add(step);
        if value > end {
            break;
        }
    }
    Ok(combos)
}

pub fn pretty_good_oscillator_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &PrettyGoodOscillatorBatchRange,
    kernel: Kernel,
) -> Result<PrettyGoodOscillatorBatchOutput, PrettyGoodOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(PrettyGoodOscillatorError::InvalidKernelForBatch(other)),
    };
    pretty_good_oscillator_batch_impl(high, low, close, source, sweep, batch_kernel, true)
}

pub fn pretty_good_oscillator_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &PrettyGoodOscillatorBatchRange,
) -> Result<PrettyGoodOscillatorBatchOutput, PrettyGoodOscillatorError> {
    pretty_good_oscillator_batch_impl(high, low, close, source, sweep, Kernel::ScalarBatch, false)
}

pub fn pretty_good_oscillator_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &PrettyGoodOscillatorBatchRange,
) -> Result<PrettyGoodOscillatorBatchOutput, PrettyGoodOscillatorError> {
    pretty_good_oscillator_batch_impl(high, low, close, source, sweep, Kernel::ScalarBatch, true)
}

fn pretty_good_oscillator_batch_impl(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &PrettyGoodOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<PrettyGoodOscillatorBatchOutput, PrettyGoodOscillatorError> {
    if high.is_empty() {
        return Err(PrettyGoodOscillatorError::EmptyInputData);
    }
    if high.len() != low.len() || low.len() != close.len() || close.len() != source.len() {
        return Err(PrettyGoodOscillatorError::DataLengthMismatch);
    }
    let combos = expand_grid_pretty_good_oscillator(sweep)?;
    let rows = combos.len();
    let cols = source.len();
    let mut out_mu = make_uninit_matrix(rows, cols);
    let warmups: Vec<usize> = combos
        .iter()
        .map(|p| p.length.unwrap_or(14).saturating_sub(1))
        .collect();
    init_matrix_prefixes(&mut out_mu, cols, &warmups);
    let mut guard = ManuallyDrop::new(out_mu);
    let out =
        unsafe { std::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    pretty_good_oscillator_batch_inner_into(
        high,
        low,
        close,
        source,
        sweep,
        kernel.to_non_batch(),
        parallel,
        out,
    )?;
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };
    Ok(PrettyGoodOscillatorBatchOutput {
        values,
        rows,
        cols,
        combos,
    })
}

fn pretty_good_oscillator_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    sweep: &PrettyGoodOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<PrettyGoodOscillatorParams>, PrettyGoodOscillatorError> {
    if high.is_empty() {
        return Err(PrettyGoodOscillatorError::EmptyInputData);
    }
    if high.len() != low.len() || low.len() != close.len() || close.len() != source.len() {
        return Err(PrettyGoodOscillatorError::DataLengthMismatch);
    }
    let combos = expand_grid_pretty_good_oscillator(sweep)?;
    let rows = combos.len();
    let cols = source.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or(PrettyGoodOscillatorError::InvalidRange {
            start: sweep.length.0,
            end: sweep.length.1,
            step: sweep.length.2,
        })?;
    if out.len() != expected {
        return Err(PrettyGoodOscillatorError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }

    unsafe {
        let out_mu =
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, expected);
        let warmups: Vec<usize> = combos
            .iter()
            .map(|p| p.length.unwrap_or(14).saturating_sub(1))
            .collect();
        init_matrix_prefixes(out_mu, cols, &warmups);
    }

    let do_row = |row: usize, dst: &mut [f64]| {
        let input =
            PrettyGoodOscillatorInput::from_slices(high, low, close, source, combos[row].clone());
        let _ = pretty_good_oscillator_into_slice(dst, &input, kernel);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, dst)| do_row(row, dst));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, dst) in out.chunks_mut(cols).enumerate() {
                do_row(row, dst);
            }
        }
    } else {
        for (row, dst) in out.chunks_mut(cols).enumerate() {
            do_row(row, dst);
        }
    }

    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "pretty_good_oscillator")]
#[pyo3(signature = (high, low, close, source, length=14, kernel=None))]
pub fn pretty_good_oscillator_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    source: PyReadonlyArray1<'py, f64>,
    length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let source = source.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = PrettyGoodOscillatorInput::from_slices(
        high,
        low,
        close,
        source,
        PrettyGoodOscillatorParams {
            length: Some(length),
        },
    );
    let output = py
        .allow_threads(|| pretty_good_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(output.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "PrettyGoodOscillatorStream")]
pub struct PrettyGoodOscillatorStreamPy {
    stream: PrettyGoodOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl PrettyGoodOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (length=14))]
    fn new(length: usize) -> PyResult<Self> {
        let stream = PrettyGoodOscillatorStream::try_new(PrettyGoodOscillatorParams {
            length: Some(length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64, source: f64) -> Option<f64> {
        self.stream.update(high, low, close, source)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "pretty_good_oscillator_batch")]
#[pyo3(signature = (high, low, close, source, length_range, kernel=None))]
pub fn pretty_good_oscillator_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    source: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let high = high.as_slice()?;
    let low = low.as_slice()?;
    let close = close.as_slice()?;
    let source = source.as_slice()?;
    let sweep = PrettyGoodOscillatorBatchRange {
        length: length_range,
    };
    let combos = expand_grid_pretty_good_oscillator(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out = unsafe { arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        pretty_good_oscillator_batch_inner_into(
            high,
            low,
            close,
            source,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_pretty_good_oscillator_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pretty_good_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(pretty_good_oscillator_batch_py, m)?)?;
    m.add_class::<PrettyGoodOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrettyGoodOscillatorBatchConfig {
    length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrettyGoodOscillatorBatchJsOutput {
    values: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<PrettyGoodOscillatorParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "pretty_good_oscillator_js")]
pub fn pretty_good_oscillator_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    length: usize,
) -> Result<Vec<f64>, JsValue> {
    let input = PrettyGoodOscillatorInput::from_slices(
        high,
        low,
        close,
        source,
        PrettyGoodOscillatorParams {
            length: Some(length),
        },
    );
    let mut out = vec![0.0; source.len()];
    pretty_good_oscillator_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "pretty_good_oscillator_batch_js")]
pub fn pretty_good_oscillator_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: PrettyGoodOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: length_range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = PrettyGoodOscillatorBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
    };
    let batch = pretty_good_oscillator_batch_slice(high, low, close, source, &sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&PrettyGoodOscillatorBatchJsOutput {
        values: batch.values,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pretty_good_oscillator_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pretty_good_oscillator_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pretty_good_oscillator_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    source_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || source_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to pretty_good_oscillator_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let source = std::slice::from_raw_parts(source_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = PrettyGoodOscillatorInput::from_slices(
            high,
            low,
            close,
            source,
            PrettyGoodOscillatorParams {
                length: Some(length),
            },
        );
        pretty_good_oscillator_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "pretty_good_oscillator_into_host")]
pub fn pretty_good_oscillator_into_host(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    out_ptr: *mut f64,
    length: usize,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to pretty_good_oscillator_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, source.len());
        let input = PrettyGoodOscillatorInput::from_slices(
            high,
            low,
            close,
            source,
            PrettyGoodOscillatorParams {
                length: Some(length),
            },
        );
        pretty_good_oscillator_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pretty_good_oscillator_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    source_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || source_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to pretty_good_oscillator_batch_into",
        ));
    }
    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let source = std::slice::from_raw_parts(source_ptr, len);
        let sweep = PrettyGoodOscillatorBatchRange {
            length: (length_start, length_end, length_step),
        };
        let combos = expand_grid_pretty_good_oscillator(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * len);
        pretty_good_oscillator_batch_inner_into(
            high,
            low,
            close,
            source,
            &sweep,
            Kernel::Scalar,
            false,
            out,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pretty_good_oscillator_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    length: usize,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = pretty_good_oscillator_js(high, low, close, source, length)?;
    crate::write_wasm_f64_output("pretty_good_oscillator_output_into_js", &values, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn pretty_good_oscillator_batch_output_into_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    source: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = pretty_good_oscillator_batch_js(high, low, close, source, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "pretty_good_oscillator_batch_output_into_js",
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

    fn sample_ohlcs(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut source = Vec::with_capacity(len);
        for i in 0..len {
            let base = 100.0 + (i as f64 * 0.17).sin() * 2.0 + i as f64 * 0.03;
            let c = base + (i as f64 * 0.11).cos() * 0.4;
            let h = c + 1.2 + (i as f64 * 0.07).sin().abs() * 0.3;
            let l = c - 1.1 - (i as f64 * 0.05).cos().abs() * 0.25;
            high.push(h);
            low.push(l);
            close.push(c);
            source.push((h + l) * 0.5);
        }
        (high, low, close, source)
    }

    fn naive_pgo(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        source: &[f64],
        length: usize,
    ) -> Vec<f64> {
        let len = source.len();
        let mut out = vec![f64::NAN; len];
        if len < length {
            return out;
        }
        let mut sum_source = 0.0;
        let mut sum_tr = 0.0;
        for i in 0..length {
            sum_source += source[i];
            let tr = if i == 0 {
                high[i] - low[i]
            } else {
                let up = high[i].max(close[i - 1]);
                let dn = low[i].min(close[i - 1]);
                up - dn
            };
            sum_tr += tr;
        }
        let mut atr = sum_tr / length as f64;
        out[length - 1] = (source[length - 1] - (sum_source / length as f64)) / atr;
        for i in length..len {
            sum_source += source[i] - source[i - length];
            let tr = {
                let up = high[i].max(close[i - 1]);
                let dn = low[i].min(close[i - 1]);
                up - dn
            };
            atr = atr + (tr - atr) / length as f64;
            out[i] = (source[i] - (sum_source / length as f64)) / atr;
        }
        out
    }

    fn assert_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            if a[i].is_nan() || b[i].is_nan() {
                assert!(
                    a[i].is_nan() && b[i].is_nan(),
                    "nan mismatch at {i}: {} vs {}",
                    a[i],
                    b[i]
                );
            } else {
                assert!(
                    (a[i] - b[i]).abs() <= 1e-10,
                    "mismatch at {i}: {} vs {}",
                    a[i],
                    b[i]
                );
            }
        }
    }

    #[test]
    fn pretty_good_oscillator_matches_naive() {
        let (high, low, close, source) = sample_ohlcs(256);
        let input = PrettyGoodOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            PrettyGoodOscillatorParams { length: Some(14) },
        );
        let out = pretty_good_oscillator(&input).expect("indicator");
        let reference = naive_pgo(&high, &low, &close, &source, 14);
        assert_close(&out.values, &reference);
    }

    #[test]
    fn pretty_good_oscillator_into_matches_api() {
        let (high, low, close, source) = sample_ohlcs(192);
        let input = PrettyGoodOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            PrettyGoodOscillatorParams { length: Some(10) },
        );
        let baseline = pretty_good_oscillator(&input).expect("baseline");
        let mut out = vec![0.0; source.len()];
        pretty_good_oscillator_into(&input, &mut out).expect("into");
        assert_close(&baseline.values, &out);
    }

    #[test]
    fn pretty_good_oscillator_stream_matches_batch() {
        let (high, low, close, source) = sample_ohlcs(192);
        let input = PrettyGoodOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            PrettyGoodOscillatorParams { length: Some(14) },
        );
        let batch = pretty_good_oscillator(&input).expect("batch");
        let mut stream =
            PrettyGoodOscillatorStream::try_new(PrettyGoodOscillatorParams { length: Some(14) })
                .expect("stream");
        let mut values = Vec::with_capacity(source.len());
        for i in 0..source.len() {
            values.push(
                stream
                    .update(high[i], low[i], close[i], source[i])
                    .unwrap_or(f64::NAN),
            );
        }
        assert_close(&batch.values, &values);
    }

    #[test]
    fn pretty_good_oscillator_batch_single_param_matches_single() {
        let (high, low, close, source) = sample_ohlcs(192);
        let sweep = PrettyGoodOscillatorBatchRange {
            length: (14, 14, 0),
        };
        let batch = pretty_good_oscillator_batch_with_kernel(
            &high,
            &low,
            &close,
            &source,
            &sweep,
            Kernel::ScalarBatch,
        )
        .expect("batch");
        let single = pretty_good_oscillator(&PrettyGoodOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            PrettyGoodOscillatorParams { length: Some(14) },
        ))
        .expect("single");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, source.len());
        assert_close(&batch.values, &single.values);
    }

    #[test]
    fn pretty_good_oscillator_rejects_invalid_length() {
        let (high, low, close, source) = sample_ohlcs(32);
        let err = pretty_good_oscillator(&PrettyGoodOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &source,
            PrettyGoodOscillatorParams { length: Some(0) },
        ))
        .expect_err("invalid length");
        assert!(matches!(
            err,
            PrettyGoodOscillatorError::InvalidLength { .. }
        ));
    }

    #[test]
    fn pretty_good_oscillator_dispatch_matches_direct() {
        let (high, low, close, _source) = sample_ohlcs(192);
        let params = [ParamKV {
            key: "length",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &params }];
        let out = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "pretty_good_oscillator",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &close,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::ScalarBatch,
        })
        .expect("dispatch");
        let direct = pretty_good_oscillator(&PrettyGoodOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &close,
            PrettyGoodOscillatorParams { length: Some(14) },
        ))
        .expect("direct");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, close.len());
        assert_close(out.values_f64.as_ref().expect("values"), &direct.values);
    }
}
