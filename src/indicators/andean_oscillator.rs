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

use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_LENGTH: usize = 50;
const DEFAULT_SIGNAL_LENGTH: usize = 9;

#[derive(Debug, Clone)]
pub enum AndeanOscillatorData<'a> {
    Candles { candles: &'a Candles },
    Slices { open: &'a [f64], close: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct AndeanOscillatorOutput {
    pub bull: Vec<f64>,
    pub bear: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AndeanOscillatorOutputField {
    Bull,
    Bear,
    Signal,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct AndeanOscillatorParams {
    pub length: Option<usize>,
    pub signal_length: Option<usize>,
}

impl Default for AndeanOscillatorParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            signal_length: Some(DEFAULT_SIGNAL_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AndeanOscillatorInput<'a> {
    pub data: AndeanOscillatorData<'a>,
    pub params: AndeanOscillatorParams,
}

impl<'a> AndeanOscillatorInput<'a> {
    #[inline]
    pub fn from_candles(candles: &'a Candles, params: AndeanOscillatorParams) -> Self {
        Self {
            data: AndeanOscillatorData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(open: &'a [f64], close: &'a [f64], params: AndeanOscillatorParams) -> Self {
        Self {
            data: AndeanOscillatorData::Slices { open, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, AndeanOscillatorParams::default())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AndeanOscillatorBuilder {
    length: Option<usize>,
    signal_length: Option<usize>,
    kernel: Kernel,
}

impl Default for AndeanOscillatorBuilder {
    fn default() -> Self {
        Self {
            length: None,
            signal_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AndeanOscillatorBuilder {
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
    pub fn signal_length(mut self, value: usize) -> Self {
        self.signal_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    pub fn apply(self, candles: &Candles) -> Result<AndeanOscillatorOutput, AndeanOscillatorError> {
        let input = AndeanOscillatorInput::from_candles(
            candles,
            AndeanOscillatorParams {
                length: self.length,
                signal_length: self.signal_length,
            },
        );
        andean_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        close: &[f64],
    ) -> Result<AndeanOscillatorOutput, AndeanOscillatorError> {
        let input = AndeanOscillatorInput::from_slices(
            open,
            close,
            AndeanOscillatorParams {
                length: self.length,
                signal_length: self.signal_length,
            },
        );
        andean_oscillator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AndeanOscillatorStream, AndeanOscillatorError> {
        AndeanOscillatorStream::try_new(AndeanOscillatorParams {
            length: self.length,
            signal_length: self.signal_length,
        })
    }
}

#[derive(Debug, Error)]
pub enum AndeanOscillatorError {
    #[error("andean_oscillator: Input data slice is empty.")]
    EmptyInputData,
    #[error(
        "andean_oscillator: Input slices must have the same length: open={open_len}, close={close_len}"
    )]
    LengthMismatch { open_len: usize, close_len: usize },
    #[error("andean_oscillator: All values are NaN.")]
    AllValuesNaN,
    #[error("andean_oscillator: Invalid length: {length}")]
    InvalidLength { length: usize },
    #[error("andean_oscillator: Invalid signal_length: {signal_length}")]
    InvalidSignalLength { signal_length: usize },
    #[error("andean_oscillator: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("andean_oscillator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("andean_oscillator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    alpha: f64,
    signal_alpha: f64,
}

#[inline(always)]
fn extract_slices<'a>(
    input: &'a AndeanOscillatorInput<'a>,
) -> Result<(&'a [f64], &'a [f64]), AndeanOscillatorError> {
    let (open, close) = match &input.data {
        AndeanOscillatorData::Candles { candles } => {
            (candles.open.as_slice(), candles.close.as_slice())
        }
        AndeanOscillatorData::Slices { open, close } => (*open, *close),
    };
    validate_raw_slices(open, close)?;
    Ok((open, close))
}

#[inline(always)]
fn resolve_params(
    params: &AndeanOscillatorParams,
) -> Result<ResolvedParams, AndeanOscillatorError> {
    let length = params.length.unwrap_or(DEFAULT_LENGTH);
    if length == 0 {
        return Err(AndeanOscillatorError::InvalidLength { length });
    }
    let signal_length = params.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH);
    if signal_length == 0 {
        return Err(AndeanOscillatorError::InvalidSignalLength { signal_length });
    }
    Ok(ResolvedParams {
        alpha: 2.0 / (length as f64 + 1.0),
        signal_alpha: 2.0 / (signal_length as f64 + 1.0),
    })
}

#[inline(always)]
fn first_valid_pair(open: &[f64], close: &[f64]) -> Option<usize> {
    (0..open.len()).find(|&i| open[i].is_finite() && close[i].is_finite())
}

#[inline(always)]
fn validate_raw_slices(open: &[f64], close: &[f64]) -> Result<usize, AndeanOscillatorError> {
    if open.is_empty() || close.is_empty() {
        return Err(AndeanOscillatorError::EmptyInputData);
    }
    if open.len() != close.len() {
        return Err(AndeanOscillatorError::LengthMismatch {
            open_len: open.len(),
            close_len: close.len(),
        });
    }
    first_valid_pair(open, close).ok_or(AndeanOscillatorError::AllValuesNaN)
}

#[inline(always)]
fn validate_input<'a>(
    input: &'a AndeanOscillatorInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], &'a [f64], ResolvedParams, usize, Kernel), AndeanOscillatorError> {
    let (open, close) = extract_slices(input)?;
    let params = resolve_params(&input.params)?;
    let first = first_valid_pair(open, close).ok_or(AndeanOscillatorError::AllValuesNaN)?;
    Ok((open, close, params, first, kernel.to_non_batch()))
}

#[derive(Clone, Debug)]
struct AndeanCore {
    params: ResolvedParams,
    initialized: bool,
    up1: f64,
    up2: f64,
    dn1: f64,
    dn2: f64,
    signal: f64,
}

impl AndeanCore {
    #[inline(always)]
    fn new(params: ResolvedParams) -> Self {
        Self {
            params,
            initialized: false,
            up1: f64::NAN,
            up2: f64::NAN,
            dn1: f64::NAN,
            dn2: f64::NAN,
            signal: f64::NAN,
        }
    }

    #[inline(always)]
    fn update(&mut self, open: f64, close: f64) -> (f64, f64, f64) {
        if !open.is_finite() || !close.is_finite() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        let close_sq = close * close;
        let open_sq = open * open;

        if !self.initialized {
            self.up1 = close;
            self.up2 = close_sq;
            self.dn1 = close;
            self.dn2 = close_sq;
            self.signal = 0.0;
            self.initialized = true;
            return (0.0, 0.0, 0.0);
        }

        let alpha = self.params.alpha;
        let up1 = self.up1 - (self.up1 - close) * alpha;
        let up2 = self.up2 - (self.up2 - close_sq) * alpha;
        let dn1 = self.dn1 + (close - self.dn1) * alpha;
        let dn2 = self.dn2 + (close_sq - self.dn2) * alpha;

        self.up1 = close.max(open.max(up1));
        self.up2 = close_sq.max(open_sq.max(up2));
        self.dn1 = close.min(open.min(dn1));
        self.dn2 = close_sq.min(open_sq.min(dn2));

        let bull = (self.dn2 - self.dn1 * self.dn1).max(0.0).sqrt();
        let bear = (self.up2 - self.up1 * self.up1).max(0.0).sqrt();
        let signal_input = bull.max(bear);
        self.signal = if self.signal.is_finite() {
            self.params.signal_alpha * signal_input + (1.0 - self.params.signal_alpha) * self.signal
        } else {
            signal_input
        };
        (bull, bear, self.signal)
    }
}

#[inline(always)]
fn compute_andean_oscillator_into(
    open: &[f64],
    close: &[f64],
    params: ResolvedParams,
    out_bull: &mut [f64],
    out_bear: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), AndeanOscillatorError> {
    let len = open.len();
    if out_bull.len() != len {
        return Err(AndeanOscillatorError::OutputLengthMismatch {
            expected: len,
            got: out_bull.len(),
        });
    }
    if out_bear.len() != len {
        return Err(AndeanOscillatorError::OutputLengthMismatch {
            expected: len,
            got: out_bear.len(),
        });
    }
    if out_signal.len() != len {
        return Err(AndeanOscillatorError::OutputLengthMismatch {
            expected: len,
            got: out_signal.len(),
        });
    }

    let mut core = AndeanCore::new(params);
    for i in 0..len {
        let (bull, bear, signal) = core.update(open[i], close[i]);
        out_bull[i] = bull;
        out_bear[i] = bear;
        out_signal[i] = signal;
    }
    Ok(())
}

#[inline]
pub fn andean_oscillator(
    input: &AndeanOscillatorInput,
) -> Result<AndeanOscillatorOutput, AndeanOscillatorError> {
    andean_oscillator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn andean_oscillator_with_kernel(
    input: &AndeanOscillatorInput,
    kernel: Kernel,
) -> Result<AndeanOscillatorOutput, AndeanOscillatorError> {
    let (open, close, params, first, _kernel) = validate_input(input, kernel)?;
    let mut bull = alloc_with_nan_prefix(open.len(), first.min(open.len()));
    let mut bear = alloc_with_nan_prefix(open.len(), first.min(open.len()));
    let mut signal = alloc_with_nan_prefix(open.len(), first.min(open.len()));
    compute_andean_oscillator_into(open, close, params, &mut bull, &mut bear, &mut signal)?;
    Ok(AndeanOscillatorOutput { bull, bear, signal })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn andean_oscillator_into(
    out_bull: &mut [f64],
    out_bear: &mut [f64],
    out_signal: &mut [f64],
    input: &AndeanOscillatorInput,
    kernel: Kernel,
) -> Result<(), AndeanOscillatorError> {
    andean_oscillator_into_slice(out_bull, out_bear, out_signal, input, kernel)
}

#[inline]
pub fn andean_oscillator_into_slice(
    out_bull: &mut [f64],
    out_bear: &mut [f64],
    out_signal: &mut [f64],
    input: &AndeanOscillatorInput,
    kernel: Kernel,
) -> Result<(), AndeanOscillatorError> {
    let (open, close, params, _first, _kernel) = validate_input(input, kernel)?;
    out_bull.fill(f64::NAN);
    out_bear.fill(f64::NAN);
    out_signal.fill(f64::NAN);
    compute_andean_oscillator_into(open, close, params, out_bull, out_bear, out_signal)
}

#[inline]
pub fn andean_oscillator_output_into_slice(
    dst: &mut [f64],
    input: &AndeanOscillatorInput,
    kernel: Kernel,
    field: AndeanOscillatorOutputField,
) -> Result<(), AndeanOscillatorError> {
    let (open, close, params, _first, _kernel) = validate_input(input, kernel)?;
    if dst.len() != open.len() {
        return Err(AndeanOscillatorError::OutputLengthMismatch {
            expected: open.len(),
            got: dst.len(),
        });
    }
    dst.fill(f64::NAN);
    let mut core = AndeanCore::new(params);
    for i in 0..open.len() {
        let (bull, bear, signal) = core.update(open[i], close[i]);
        dst[i] = match field {
            AndeanOscillatorOutputField::Bull => bull,
            AndeanOscillatorOutputField::Bear => bear,
            AndeanOscillatorOutputField::Signal => signal,
        };
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct AndeanOscillatorStream {
    core: AndeanCore,
}

impl AndeanOscillatorStream {
    pub fn try_new(params: AndeanOscillatorParams) -> Result<Self, AndeanOscillatorError> {
        Ok(Self {
            core: AndeanCore::new(resolve_params(&params)?),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, open: f64, close: f64) -> (f64, f64, f64) {
        self.core.update(open, close)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AndeanOscillatorBatchRange {
    pub length: (usize, usize, usize),
    pub signal_length: (usize, usize, usize),
}

impl Default for AndeanOscillatorBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            signal_length: (DEFAULT_SIGNAL_LENGTH, DEFAULT_SIGNAL_LENGTH, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AndeanOscillatorBatchOutput {
    pub bull: Vec<f64>,
    pub bear: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<AndeanOscillatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct AndeanOscillatorBatchBuilder {
    range: AndeanOscillatorBatchRange,
    kernel: Kernel,
}

impl Default for AndeanOscillatorBatchBuilder {
    fn default() -> Self {
        Self {
            range: AndeanOscillatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl AndeanOscillatorBatchBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.length = value;
        self
    }

    #[inline(always)]
    pub fn signal_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.range.signal_length = value;
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
    ) -> Result<AndeanOscillatorBatchOutput, AndeanOscillatorError> {
        andean_oscillator_batch_with_kernel(
            candles.open.as_slice(),
            candles.close.as_slice(),
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slices(
        self,
        open: &[f64],
        close: &[f64],
    ) -> Result<AndeanOscillatorBatchOutput, AndeanOscillatorError> {
        andean_oscillator_batch_with_kernel(open, close, &self.range, self.kernel)
    }
}

#[inline(always)]
fn expand_usize_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, AndeanOscillatorError> {
    if step == 0 {
        if start != end {
            return Err(AndeanOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if start > end {
        return Err(AndeanOscillatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    while current <= end {
        out.push(current);
        current = current
            .checked_add(step)
            .ok_or_else(|| AndeanOscillatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            })?;
    }
    Ok(out)
}

pub fn expand_grid(
    sweep: &AndeanOscillatorBatchRange,
) -> Result<Vec<AndeanOscillatorParams>, AndeanOscillatorError> {
    let lengths = expand_usize_range(sweep.length.0, sweep.length.1, sweep.length.2)?;
    let signal_lengths = expand_usize_range(
        sweep.signal_length.0,
        sweep.signal_length.1,
        sweep.signal_length.2,
    )?;
    let mut combos = Vec::with_capacity(lengths.len() * signal_lengths.len());
    for length in lengths {
        for signal_length in signal_lengths.iter().copied() {
            combos.push(AndeanOscillatorParams {
                length: Some(length),
                signal_length: Some(signal_length),
            });
        }
    }
    Ok(combos)
}

#[inline(always)]
fn batch_shape(rows: usize, cols: usize) -> Result<usize, AndeanOscillatorError> {
    rows.checked_mul(cols)
        .ok_or_else(|| AndeanOscillatorError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })
}

pub fn andean_oscillator_batch_with_kernel(
    open: &[f64],
    close: &[f64],
    sweep: &AndeanOscillatorBatchRange,
    kernel: Kernel,
) -> Result<AndeanOscillatorBatchOutput, AndeanOscillatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(AndeanOscillatorError::InvalidKernelForBatch(kernel)),
    };
    andean_oscillator_batch_par_slice(open, close, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn andean_oscillator_batch_slice(
    open: &[f64],
    close: &[f64],
    sweep: &AndeanOscillatorBatchRange,
    kernel: Kernel,
) -> Result<AndeanOscillatorBatchOutput, AndeanOscillatorError> {
    andean_oscillator_batch_inner(open, close, sweep, kernel, false)
}

#[inline(always)]
pub fn andean_oscillator_batch_par_slice(
    open: &[f64],
    close: &[f64],
    sweep: &AndeanOscillatorBatchRange,
    kernel: Kernel,
) -> Result<AndeanOscillatorBatchOutput, AndeanOscillatorError> {
    andean_oscillator_batch_inner(open, close, sweep, kernel, true)
}

fn andean_oscillator_batch_inner(
    open: &[f64],
    close: &[f64],
    sweep: &AndeanOscillatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<AndeanOscillatorBatchOutput, AndeanOscillatorError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(open, close)?;
    let rows = combos.len();
    let cols = open.len();
    let total = batch_shape(rows, cols)?;
    let warmups = vec![first.min(cols); rows];

    let mut bull_buf = make_uninit_matrix(rows, cols);
    let mut bear_buf = make_uninit_matrix(rows, cols);
    let mut signal_buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut bull_buf, cols, &warmups);
    init_matrix_prefixes(&mut bear_buf, cols, &warmups);
    init_matrix_prefixes(&mut signal_buf, cols, &warmups);

    let mut bull_guard = ManuallyDrop::new(bull_buf);
    let mut bear_guard = ManuallyDrop::new(bear_buf);
    let mut signal_guard = ManuallyDrop::new(signal_buf);
    let out_bull: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(bull_guard.as_mut_ptr() as *mut f64, bull_guard.len())
    };
    let out_bear: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(bear_guard.as_mut_ptr() as *mut f64, bear_guard.len())
    };
    let out_signal: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(signal_guard.as_mut_ptr() as *mut f64, signal_guard.len())
    };

    andean_oscillator_batch_inner_into(
        open, close, sweep, kernel, parallel, out_bull, out_bear, out_signal,
    )?;

    let bull = unsafe {
        Vec::from_raw_parts(
            bull_guard.as_mut_ptr() as *mut f64,
            total,
            bull_guard.capacity(),
        )
    };
    let bear = unsafe {
        Vec::from_raw_parts(
            bear_guard.as_mut_ptr() as *mut f64,
            total,
            bear_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            total,
            signal_guard.capacity(),
        )
    };

    Ok(AndeanOscillatorBatchOutput {
        bull,
        bear,
        signal,
        combos,
        rows,
        cols,
    })
}

pub fn andean_oscillator_batch_into_slice(
    out_bull: &mut [f64],
    out_bear: &mut [f64],
    out_signal: &mut [f64],
    open: &[f64],
    close: &[f64],
    sweep: &AndeanOscillatorBatchRange,
    kernel: Kernel,
) -> Result<(), AndeanOscillatorError> {
    andean_oscillator_batch_inner_into(
        open, close, sweep, kernel, false, out_bull, out_bear, out_signal,
    )?;
    Ok(())
}

fn andean_oscillator_batch_inner_into(
    open: &[f64],
    close: &[f64],
    sweep: &AndeanOscillatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_bull: &mut [f64],
    out_bear: &mut [f64],
    out_signal: &mut [f64],
) -> Result<Vec<AndeanOscillatorParams>, AndeanOscillatorError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slices(open, close)?;
    let rows = combos.len();
    let cols = open.len();
    let total = batch_shape(rows, cols)?;
    if out_bull.len() != total {
        return Err(AndeanOscillatorError::OutputLengthMismatch {
            expected: total,
            got: out_bull.len(),
        });
    }
    if out_bear.len() != total {
        return Err(AndeanOscillatorError::OutputLengthMismatch {
            expected: total,
            got: out_bear.len(),
        });
    }
    if out_signal.len() != total {
        return Err(AndeanOscillatorError::OutputLengthMismatch {
            expected: total,
            got: out_signal.len(),
        });
    }

    let compute_row = |row: usize,
                       bull_row: &mut [f64],
                       bear_row: &mut [f64],
                       signal_row: &mut [f64]|
     -> Result<(), AndeanOscillatorError> {
        bull_row[..first].fill(f64::NAN);
        bear_row[..first].fill(f64::NAN);
        signal_row[..first].fill(f64::NAN);
        let params = resolve_params(&combos[row])?;
        compute_andean_oscillator_into(open, close, params, bull_row, bear_row, signal_row)
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_bull
                .par_chunks_mut(cols)
                .zip(out_bear.par_chunks_mut(cols))
                .zip(out_signal.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(|(row, ((bull_row, bear_row), signal_row))| {
                    compute_row(row, bull_row, bear_row, signal_row)
                })?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for row in 0..rows {
                let start = row * cols;
                let end = start + cols;
                compute_row(
                    row,
                    &mut out_bull[start..end],
                    &mut out_bear[start..end],
                    &mut out_signal[start..end],
                )?;
            }
        }
    } else {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            compute_row(
                row,
                &mut out_bull[start..end],
                &mut out_bear[start..end],
                &mut out_signal[start..end],
            )?;
        }
    }
    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "andean_oscillator")]
#[pyo3(signature = (open, close, length=50, signal_length=9, kernel=None))]
pub fn andean_oscillator_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length: usize,
    signal_length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open_slice = open.as_slice()?;
    let close_slice = close.as_slice()?;
    let input = AndeanOscillatorInput::from_slices(
        open_slice,
        close_slice,
        AndeanOscillatorParams {
            length: Some(length),
            signal_length: Some(signal_length),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| andean_oscillator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("bull", out.bull.into_pyarray(py))?;
    dict.set_item("bear", out.bear.into_pyarray(py))?;
    dict.set_item("signal", out.signal.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "AndeanOscillatorStream")]
pub struct AndeanOscillatorStreamPy {
    stream: AndeanOscillatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AndeanOscillatorStreamPy {
    #[new]
    #[pyo3(signature = (length=50, signal_length=9))]
    fn new(length: usize, signal_length: usize) -> PyResult<Self> {
        let stream = AndeanOscillatorStream::try_new(AndeanOscillatorParams {
            length: Some(length),
            signal_length: Some(signal_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, open: f64, close: f64) -> (f64, f64, f64) {
        self.stream.update(open, close)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "andean_oscillator_batch")]
#[pyo3(signature = (open, close, length_range=(50,50,0), signal_length_range=(9,9,0), kernel=None))]
pub fn andean_oscillator_batch_py<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    signal_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let open_slice = open.as_slice()?;
    let close_slice = close.as_slice()?;
    let sweep = AndeanOscillatorBatchRange {
        length: length_range,
        signal_length: signal_length_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = open_slice.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_bull = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_bear = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_signal = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let bull_slice = unsafe { out_bull.as_slice_mut()? };
    let bear_slice = unsafe { out_bear.as_slice_mut()? };
    let signal_slice = unsafe { out_signal.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        andean_oscillator_batch_inner_into(
            open_slice,
            close_slice,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            bull_slice,
            bear_slice,
            signal_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("bull", out_bull.reshape((rows, cols))?)?;
    dict.set_item("bear", out_bear.reshape((rows, cols))?)?;
    dict.set_item("signal", out_signal.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "signal_lengths",
        combos
            .iter()
            .map(|combo| combo.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_andean_oscillator_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(andean_oscillator_py, m)?)?;
    m.add_function(wrap_pyfunction!(andean_oscillator_batch_py, m)?)?;
    m.add_class::<AndeanOscillatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AndeanOscillatorJsOutput {
    pub bull: Vec<f64>,
    pub bear: Vec<f64>,
    pub signal: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "andean_oscillator_js")]
pub fn andean_oscillator_js(
    open: &[f64],
    close: &[f64],
    length: usize,
    signal_length: usize,
) -> Result<JsValue, JsValue> {
    let input = AndeanOscillatorInput::from_slices(
        open,
        close,
        AndeanOscillatorParams {
            length: Some(length),
            signal_length: Some(signal_length),
        },
    );
    let out = andean_oscillator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&AndeanOscillatorJsOutput {
        bull: out.bull,
        bear: out.bear,
        signal: out.signal,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AndeanOscillatorBatchConfig {
    pub length_range: Vec<usize>,
    pub signal_length_range: Vec<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct AndeanOscillatorBatchJsOutput {
    pub bull: Vec<f64>,
    pub bear: Vec<f64>,
    pub signal: Vec<f64>,
    pub lengths: Vec<usize>,
    pub signal_lengths: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_usize(name: &str, values: &[usize]) -> Result<(usize, usize, usize), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    Ok((values[0], values[1], values[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "andean_oscillator_batch_js")]
pub fn andean_oscillator_batch_js(
    open: &[f64],
    close: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: AndeanOscillatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = AndeanOscillatorBatchRange {
        length: js_vec3_to_usize("length_range", &config.length_range)?,
        signal_length: js_vec3_to_usize("signal_length_range", &config.signal_length_range)?,
    };
    let out = andean_oscillator_batch_with_kernel(open, close, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&AndeanOscillatorBatchJsOutput {
        bull: out.bull,
        bear: out.bear,
        signal: out.signal,
        lengths: out
            .combos
            .iter()
            .map(|combo| combo.length.unwrap_or(DEFAULT_LENGTH))
            .collect(),
        signal_lengths: out
            .combos
            .iter()
            .map(|combo| combo.signal_length.unwrap_or(DEFAULT_SIGNAL_LENGTH))
            .collect(),
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn andean_oscillator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn andean_oscillator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn andean_oscillator_into(
    open_ptr: *const f64,
    close_ptr: *const f64,
    out_bull_ptr: *mut f64,
    out_bear_ptr: *mut f64,
    out_signal_ptr: *mut f64,
    len: usize,
    length: usize,
    signal_length: usize,
) -> Result<(), JsValue> {
    if [open_ptr, close_ptr].iter().any(|ptr| ptr.is_null())
        || [out_bull_ptr, out_bear_ptr, out_signal_ptr]
            .iter()
            .any(|ptr| ptr.is_null())
    {
        return Err(JsValue::from_str(
            "null pointer passed to andean_oscillator_into",
        ));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let out_bull = std::slice::from_raw_parts_mut(out_bull_ptr, len);
        let out_bear = std::slice::from_raw_parts_mut(out_bear_ptr, len);
        let out_signal = std::slice::from_raw_parts_mut(out_signal_ptr, len);
        let input = AndeanOscillatorInput::from_slices(
            open,
            close,
            AndeanOscillatorParams {
                length: Some(length),
                signal_length: Some(signal_length),
            },
        );
        andean_oscillator_into_slice(out_bull, out_bear, out_signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn andean_oscillator_batch_into(
    open_ptr: *const f64,
    close_ptr: *const f64,
    out_bull_ptr: *mut f64,
    out_bear_ptr: *mut f64,
    out_signal_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    signal_length_start: usize,
    signal_length_end: usize,
    signal_length_step: usize,
) -> Result<usize, JsValue> {
    if [open_ptr, close_ptr].iter().any(|ptr| ptr.is_null())
        || [out_bull_ptr, out_bear_ptr, out_signal_ptr]
            .iter()
            .any(|ptr| ptr.is_null())
    {
        return Err(JsValue::from_str(
            "null pointer passed to andean_oscillator_batch_into",
        ));
    }
    unsafe {
        let open = std::slice::from_raw_parts(open_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let sweep = AndeanOscillatorBatchRange {
            length: (length_start, length_end, length_step),
            signal_length: (signal_length_start, signal_length_end, signal_length_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows.checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in andean_oscillator_batch_into")
        })?;
        let out_bull = std::slice::from_raw_parts_mut(out_bull_ptr, total);
        let out_bear = std::slice::from_raw_parts_mut(out_bear_ptr, total);
        let out_signal = std::slice::from_raw_parts_mut(out_signal_ptr, total);
        andean_oscillator_batch_into_slice(
            out_bull,
            out_bear,
            out_signal,
            open,
            close,
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn andean_oscillator_output_into_js(
    open: &[f64],
    close: &[f64],
    length: usize,
    signal_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = andean_oscillator_js(open, close, length, signal_length)?;
    crate::write_wasm_object_f64_outputs("andean_oscillator_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn andean_oscillator_batch_output_into_js(
    open: &[f64],
    close: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = andean_oscillator_batch_js(open, close, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "andean_oscillator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_open_close(len: usize) -> (Vec<f64>, Vec<f64>) {
        let mut open = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        for i in 0..len {
            let x = i as f64;
            let base = 100.0 + (x * 0.13).sin() * 2.0 + (x * 0.05).cos() * 0.9 + x * 0.02;
            open.push(base - 0.35 + (x * 0.07).sin() * 0.2);
            close.push(base + 0.28 + (x * 0.11).cos() * 0.25);
        }
        (open, close)
    }

    fn assert_series_eq(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            if a[i].is_nan() || b[i].is_nan() {
                assert!(a[i].is_nan() && b[i].is_nan(), "NaN mismatch at {i}");
            } else {
                assert!(
                    (a[i] - b[i]).abs() <= 1e-12,
                    "mismatch at {i}: {} vs {}",
                    a[i],
                    b[i]
                );
            }
        }
    }

    #[test]
    fn manual_reference_matches_api() {
        let (open, close) = sample_open_close(160);
        let input = AndeanOscillatorInput::from_slices(
            &open,
            &close,
            AndeanOscillatorParams {
                length: Some(50),
                signal_length: Some(9),
            },
        );
        let out = andean_oscillator(&input).unwrap();
        let params = resolve_params(&AndeanOscillatorParams {
            length: Some(50),
            signal_length: Some(9),
        })
        .unwrap();
        let mut core = AndeanCore::new(params);
        let mut bull = vec![f64::NAN; open.len()];
        let mut bear = vec![f64::NAN; open.len()];
        let mut signal = vec![f64::NAN; open.len()];
        for i in 0..open.len() {
            let (b1, b2, s) = core.update(open[i], close[i]);
            bull[i] = b1;
            bear[i] = b2;
            signal[i] = s;
        }
        assert_series_eq(&out.bull, &bull);
        assert_series_eq(&out.bear, &bear);
        assert_series_eq(&out.signal, &signal);
    }

    #[test]
    fn stream_matches_batch() {
        let (open, close) = sample_open_close(144);
        let input = AndeanOscillatorInput::from_slices(
            &open,
            &close,
            AndeanOscillatorParams {
                length: Some(50),
                signal_length: Some(9),
            },
        );
        let batch = andean_oscillator(&input).unwrap();
        let mut stream = AndeanOscillatorStream::try_new(AndeanOscillatorParams {
            length: Some(50),
            signal_length: Some(9),
        })
        .unwrap();
        let mut bull = vec![f64::NAN; open.len()];
        let mut bear = vec![f64::NAN; open.len()];
        let mut signal = vec![f64::NAN; open.len()];
        for i in 0..open.len() {
            let (b1, b2, s) = stream.update(open[i], close[i]);
            bull[i] = b1;
            bear[i] = b2;
            signal[i] = s;
        }
        assert_series_eq(&batch.bull, &bull);
        assert_series_eq(&batch.bear, &bear);
        assert_series_eq(&batch.signal, &signal);
    }

    #[test]
    fn batch_first_row_matches_single() {
        let (open, close) = sample_open_close(128);
        let sweep = AndeanOscillatorBatchRange {
            length: (50, 52, 2),
            signal_length: (9, 11, 2),
        };
        let batch = andean_oscillator_batch_with_kernel(&open, &close, &sweep, Kernel::ScalarBatch)
            .unwrap();
        let input = AndeanOscillatorInput::from_slices(
            &open,
            &close,
            AndeanOscillatorParams {
                length: Some(50),
                signal_length: Some(9),
            },
        );
        let single = andean_oscillator_with_kernel(&input, Kernel::Scalar).unwrap();
        assert_eq!(batch.rows, 4);
        assert_eq!(batch.cols, open.len());
        assert_series_eq(&batch.bull[..open.len()], &single.bull);
        assert_series_eq(&batch.bear[..open.len()], &single.bear);
        assert_series_eq(&batch.signal[..open.len()], &single.signal);
    }

    #[test]
    fn into_slice_matches_single() {
        let (open, close) = sample_open_close(96);
        let input = AndeanOscillatorInput::from_slices(
            &open,
            &close,
            AndeanOscillatorParams {
                length: Some(50),
                signal_length: Some(9),
            },
        );
        let single = andean_oscillator(&input).unwrap();
        let mut bull = vec![f64::NAN; open.len()];
        let mut bear = vec![f64::NAN; open.len()];
        let mut signal = vec![f64::NAN; open.len()];
        andean_oscillator_into_slice(&mut bull, &mut bear, &mut signal, &input, Kernel::Auto)
            .unwrap();
        assert_series_eq(&single.bull, &bull);
        assert_series_eq(&single.bear, &bear);
        assert_series_eq(&single.signal, &signal);
    }

    #[test]
    fn invalid_lengths_are_rejected() {
        let (open, close) = sample_open_close(32);
        let err = andean_oscillator(&AndeanOscillatorInput::from_slices(
            &open,
            &close,
            AndeanOscillatorParams {
                length: Some(0),
                signal_length: Some(9),
            },
        ))
        .unwrap_err();
        assert!(matches!(err, AndeanOscillatorError::InvalidLength { .. }));
        let err = andean_oscillator(&AndeanOscillatorInput::from_slices(
            &open,
            &close,
            AndeanOscillatorParams {
                length: Some(50),
                signal_length: Some(0),
            },
        ))
        .unwrap_err();
        assert!(matches!(
            err,
            AndeanOscillatorError::InvalidSignalLength { .. }
        ));
    }
}
