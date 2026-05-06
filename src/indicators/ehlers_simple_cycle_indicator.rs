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
use crate::utilities::helpers::{alloc_uninit_f64, detect_best_batch_kernel, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "hl2";
const DEFAULT_ALPHA: f64 = 0.07;
const REQUIRED_VALID_SAMPLES: usize = 3;

#[derive(Debug, Clone)]
pub enum EhlersSimpleCycleIndicatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersSimpleCycleIndicatorOutput {
    pub cycle: Vec<f64>,
    pub trigger: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EhlersSimpleCycleIndicatorParams {
    pub alpha: Option<f64>,
}

impl Default for EhlersSimpleCycleIndicatorParams {
    fn default() -> Self {
        Self {
            alpha: Some(DEFAULT_ALPHA),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersSimpleCycleIndicatorInput<'a> {
    pub data: EhlersSimpleCycleIndicatorData<'a>,
    pub params: EhlersSimpleCycleIndicatorParams,
}

impl<'a> EhlersSimpleCycleIndicatorInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: EhlersSimpleCycleIndicatorParams,
    ) -> Self {
        Self {
            data: EhlersSimpleCycleIndicatorData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: EhlersSimpleCycleIndicatorParams) -> Self {
        Self {
            data: EhlersSimpleCycleIndicatorData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            EhlersSimpleCycleIndicatorParams::default(),
        )
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersSimpleCycleIndicatorBuilder {
    source: Option<&'static str>,
    alpha: Option<f64>,
    kernel: Kernel,
}

impl Default for EhlersSimpleCycleIndicatorBuilder {
    fn default() -> Self {
        Self {
            source: None,
            alpha: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersSimpleCycleIndicatorBuilder {
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
    pub fn alpha(mut self, value: f64) -> Self {
        self.alpha = Some(value);
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
    ) -> Result<EhlersSimpleCycleIndicatorOutput, EhlersSimpleCycleIndicatorError> {
        let input = EhlersSimpleCycleIndicatorInput::from_candles(
            candles,
            self.source.unwrap_or(DEFAULT_SOURCE),
            EhlersSimpleCycleIndicatorParams { alpha: self.alpha },
        );
        ehlers_simple_cycle_indicator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EhlersSimpleCycleIndicatorOutput, EhlersSimpleCycleIndicatorError> {
        let input = EhlersSimpleCycleIndicatorInput::from_slice(
            data,
            EhlersSimpleCycleIndicatorParams { alpha: self.alpha },
        );
        ehlers_simple_cycle_indicator_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<EhlersSimpleCycleIndicatorStream, EhlersSimpleCycleIndicatorError> {
        EhlersSimpleCycleIndicatorStream::try_new(EhlersSimpleCycleIndicatorParams {
            alpha: self.alpha,
        })
    }
}

#[derive(Debug, Error)]
pub enum EhlersSimpleCycleIndicatorError {
    #[error("ehlers_simple_cycle_indicator: Input data slice is empty.")]
    EmptyInputData,
    #[error("ehlers_simple_cycle_indicator: All values are NaN.")]
    AllValuesNaN,
    #[error("ehlers_simple_cycle_indicator: Invalid alpha: {alpha}")]
    InvalidAlpha { alpha: f64 },
    #[error(
        "ehlers_simple_cycle_indicator: Not enough valid data: needed={needed}, valid={valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "ehlers_simple_cycle_indicator: Output length mismatch: expected={expected}, got={got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ehlers_simple_cycle_indicator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("ehlers_simple_cycle_indicator: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    coef_cycle: f64,
    coef_prev1: f64,
    coef_prev2: f64,
}

#[inline(always)]
fn extract_slice<'a>(
    input: &'a EhlersSimpleCycleIndicatorInput<'a>,
) -> Result<&'a [f64], EhlersSimpleCycleIndicatorError> {
    let data = match &input.data {
        EhlersSimpleCycleIndicatorData::Candles { candles, source } => source_type(candles, source),
        EhlersSimpleCycleIndicatorData::Slice(values) => *values,
    };
    if data.is_empty() {
        return Err(EhlersSimpleCycleIndicatorError::EmptyInputData);
    }
    Ok(data)
}

#[inline(always)]
fn first_valid(data: &[f64]) -> Option<usize> {
    data.iter().position(|v| v.is_finite())
}

#[inline(always)]
fn resolve_params(
    params: &EhlersSimpleCycleIndicatorParams,
) -> Result<ResolvedParams, EhlersSimpleCycleIndicatorError> {
    let alpha = params.alpha.unwrap_or(DEFAULT_ALPHA);
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(EhlersSimpleCycleIndicatorError::InvalidAlpha { alpha });
    }
    let coef_cycle = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
    let one_minus_alpha = 1.0 - alpha;
    Ok(ResolvedParams {
        coef_cycle,
        coef_prev1: 2.0 * one_minus_alpha,
        coef_prev2: one_minus_alpha * one_minus_alpha,
    })
}

#[inline(always)]
fn validate_input<'a>(
    input: &'a EhlersSimpleCycleIndicatorInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], ResolvedParams, usize, Kernel), EhlersSimpleCycleIndicatorError> {
    let data = extract_slice(input)?;
    let params = resolve_params(&input.params)?;
    let first = first_valid(data).ok_or(EhlersSimpleCycleIndicatorError::AllValuesNaN)?;
    let valid = data.len().saturating_sub(first);
    if valid < REQUIRED_VALID_SAMPLES {
        return Err(EhlersSimpleCycleIndicatorError::NotEnoughValidData {
            needed: REQUIRED_VALID_SAMPLES,
            valid,
        });
    }
    Ok((data, params, first, kernel.to_non_batch()))
}

#[derive(Clone, Debug)]
struct EsciCore {
    params: ResolvedParams,
    src_ring: [f64; 4],
    src_idx: usize,
    smooth_ring: [f64; 3],
    smooth_idx: usize,
    cycle_hist: [f64; 2],
    valid_count: usize,
}

impl EsciCore {
    #[inline(always)]
    fn new(params: ResolvedParams) -> Self {
        Self {
            params,
            src_ring: [f64::NAN; 4],
            src_idx: 0,
            smooth_ring: [f64::NAN; 3],
            smooth_idx: 0,
            cycle_hist: [f64::NAN; 2],
            valid_count: 0,
        }
    }

    #[inline(always)]
    fn update(&mut self, source: f64) -> (f64, f64) {
        if !source.is_finite() {
            return (f64::NAN, f64::NAN);
        }

        self.src_ring[self.src_idx] = source;
        let src_idx = self.src_idx;
        let src0 = source;
        let src1 = self.src_ring[(src_idx + 3) & 3];
        let src2 = self.src_ring[(src_idx + 2) & 3];
        let src3 = self.src_ring[(src_idx + 1) & 3];

        let smooth = if src0.is_finite() && src1.is_finite() && src2.is_finite() && src3.is_finite()
        {
            (src0 + 2.0 * src1 + 2.0 * src2 + src3) / 6.0
        } else {
            f64::NAN
        };
        self.smooth_ring[self.smooth_idx] = smooth;

        let smooth_idx = self.smooth_idx;
        let smooth1 = self.smooth_ring[if smooth_idx == 0 { 2 } else { smooth_idx - 1 }];
        let smooth2 = self.smooth_ring[if smooth_idx >= 2 {
            smooth_idx - 2
        } else {
            smooth_idx + 1
        }];
        let prev_cycle1 = self.cycle_hist[0];
        let prev_cycle2 = self.cycle_hist[1];

        let cycle_main = if smooth.is_finite()
            && smooth1.is_finite()
            && smooth2.is_finite()
            && prev_cycle1.is_finite()
            && prev_cycle2.is_finite()
        {
            self.params.coef_cycle * (smooth - 2.0 * smooth1 + smooth2)
                + self.params.coef_prev1 * prev_cycle1
                - self.params.coef_prev2 * prev_cycle2
        } else {
            f64::NAN
        };

        let cycle_fallback = if src0.is_finite() && src1.is_finite() && src2.is_finite() {
            (src0 - 2.0 * src1 + src2) / 4.0
        } else {
            f64::NAN
        };

        let cycle = if self.valid_count < 7 {
            cycle_fallback
        } else {
            cycle_main
        };
        let trigger = prev_cycle1;

        self.cycle_hist[1] = self.cycle_hist[0];
        self.cycle_hist[0] = cycle;
        self.valid_count += 1;
        self.src_idx = (src_idx + 1) & 3;
        self.smooth_idx = if smooth_idx == 2 { 0 } else { smooth_idx + 1 };

        (cycle, trigger)
    }
}

#[inline(always)]
fn compute_esci_into(
    data: &[f64],
    params: ResolvedParams,
    out_cycle: &mut [f64],
    out_trigger: &mut [f64],
) -> Result<(), EhlersSimpleCycleIndicatorError> {
    if out_cycle.len() != data.len() {
        return Err(EhlersSimpleCycleIndicatorError::OutputLengthMismatch {
            expected: data.len(),
            got: out_cycle.len(),
        });
    }
    if out_trigger.len() != data.len() {
        return Err(EhlersSimpleCycleIndicatorError::OutputLengthMismatch {
            expected: data.len(),
            got: out_trigger.len(),
        });
    }
    let mut core = EsciCore::new(params);
    for i in 0..data.len() {
        let (cycle, trigger) = core.update(data[i]);
        out_cycle[i] = cycle;
        out_trigger[i] = trigger;
    }
    Ok(())
}

#[inline]
pub fn ehlers_simple_cycle_indicator(
    input: &EhlersSimpleCycleIndicatorInput,
) -> Result<EhlersSimpleCycleIndicatorOutput, EhlersSimpleCycleIndicatorError> {
    ehlers_simple_cycle_indicator_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn ehlers_simple_cycle_indicator_with_kernel(
    input: &EhlersSimpleCycleIndicatorInput,
    kernel: Kernel,
) -> Result<EhlersSimpleCycleIndicatorOutput, EhlersSimpleCycleIndicatorError> {
    let (data, params, _first, _kernel) = validate_input(input, kernel)?;
    let mut cycle = alloc_uninit_f64(data.len());
    let mut trigger = alloc_uninit_f64(data.len());
    compute_esci_into(data, params, &mut cycle, &mut trigger)?;
    Ok(EhlersSimpleCycleIndicatorOutput { cycle, trigger })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn ehlers_simple_cycle_indicator_into(
    out_cycle: &mut [f64],
    out_trigger: &mut [f64],
    input: &EhlersSimpleCycleIndicatorInput,
    kernel: Kernel,
) -> Result<(), EhlersSimpleCycleIndicatorError> {
    ehlers_simple_cycle_indicator_into_slice(out_cycle, out_trigger, input, kernel)
}

#[inline]
pub fn ehlers_simple_cycle_indicator_into_slice(
    out_cycle: &mut [f64],
    out_trigger: &mut [f64],
    input: &EhlersSimpleCycleIndicatorInput,
    kernel: Kernel,
) -> Result<(), EhlersSimpleCycleIndicatorError> {
    let (data, params, _first, _kernel) = validate_input(input, kernel)?;
    compute_esci_into(data, params, out_cycle, out_trigger)
}

#[derive(Clone, Debug)]
pub struct EhlersSimpleCycleIndicatorStream {
    core: EsciCore,
}

impl EhlersSimpleCycleIndicatorStream {
    pub fn try_new(
        params: EhlersSimpleCycleIndicatorParams,
    ) -> Result<Self, EhlersSimpleCycleIndicatorError> {
        Ok(Self {
            core: EsciCore::new(resolve_params(&params)?),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> (f64, f64) {
        self.core.update(value)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EhlersSimpleCycleIndicatorBatchRange {
    pub alpha: (f64, f64, f64),
}

impl Default for EhlersSimpleCycleIndicatorBatchRange {
    fn default() -> Self {
        Self {
            alpha: (DEFAULT_ALPHA, DEFAULT_ALPHA, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EhlersSimpleCycleIndicatorBatchOutput {
    pub cycle: Vec<f64>,
    pub trigger: Vec<f64>,
    pub combos: Vec<EhlersSimpleCycleIndicatorParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct EhlersSimpleCycleIndicatorBatchBuilder {
    source: Option<&'static str>,
    range: EhlersSimpleCycleIndicatorBatchRange,
    kernel: Kernel,
}

impl Default for EhlersSimpleCycleIndicatorBatchBuilder {
    fn default() -> Self {
        Self {
            source: None,
            range: EhlersSimpleCycleIndicatorBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersSimpleCycleIndicatorBatchBuilder {
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
    pub fn alpha_range(mut self, value: (f64, f64, f64)) -> Self {
        self.range.alpha = value;
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
    ) -> Result<EhlersSimpleCycleIndicatorBatchOutput, EhlersSimpleCycleIndicatorError> {
        ehlers_simple_cycle_indicator_batch_with_kernel(
            source_type(candles, self.source.unwrap_or(DEFAULT_SOURCE)),
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EhlersSimpleCycleIndicatorBatchOutput, EhlersSimpleCycleIndicatorError> {
        ehlers_simple_cycle_indicator_batch_with_kernel(data, &self.range, self.kernel)
    }
}

#[inline(always)]
fn expand_float_range(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, EhlersSimpleCycleIndicatorError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(EhlersSimpleCycleIndicatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 {
        if (start - end).abs() > 1e-12 {
            return Err(EhlersSimpleCycleIndicatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if start > end || step < 0.0 {
        return Err(EhlersSimpleCycleIndicatorError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut current = start;
    while current <= end + 1e-12 {
        out.push(current);
        if out.len() > 1_000_000 {
            return Err(EhlersSimpleCycleIndicatorError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        current += step;
    }
    Ok(out)
}

pub fn expand_grid(
    sweep: &EhlersSimpleCycleIndicatorBatchRange,
) -> Result<Vec<EhlersSimpleCycleIndicatorParams>, EhlersSimpleCycleIndicatorError> {
    let alphas = expand_float_range(sweep.alpha.0, sweep.alpha.1, sweep.alpha.2)?;
    let mut combos = Vec::with_capacity(alphas.len());
    for alpha in alphas {
        combos.push(EhlersSimpleCycleIndicatorParams { alpha: Some(alpha) });
    }
    Ok(combos)
}

#[inline(always)]
fn validate_raw_slice(data: &[f64]) -> Result<usize, EhlersSimpleCycleIndicatorError> {
    if data.is_empty() {
        return Err(EhlersSimpleCycleIndicatorError::EmptyInputData);
    }
    first_valid(data).ok_or(EhlersSimpleCycleIndicatorError::AllValuesNaN)
}

#[inline(always)]
fn batch_shape(rows: usize, cols: usize) -> Result<usize, EhlersSimpleCycleIndicatorError> {
    rows.checked_mul(cols)
        .ok_or_else(|| EhlersSimpleCycleIndicatorError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })
}

pub fn ehlers_simple_cycle_indicator_batch_with_kernel(
    data: &[f64],
    sweep: &EhlersSimpleCycleIndicatorBatchRange,
    kernel: Kernel,
) -> Result<EhlersSimpleCycleIndicatorBatchOutput, EhlersSimpleCycleIndicatorError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(EhlersSimpleCycleIndicatorError::InvalidKernelForBatch(
                kernel,
            ))
        }
    };
    ehlers_simple_cycle_indicator_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn ehlers_simple_cycle_indicator_batch_slice(
    data: &[f64],
    sweep: &EhlersSimpleCycleIndicatorBatchRange,
    kernel: Kernel,
) -> Result<EhlersSimpleCycleIndicatorBatchOutput, EhlersSimpleCycleIndicatorError> {
    ehlers_simple_cycle_indicator_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn ehlers_simple_cycle_indicator_batch_par_slice(
    data: &[f64],
    sweep: &EhlersSimpleCycleIndicatorBatchRange,
    kernel: Kernel,
) -> Result<EhlersSimpleCycleIndicatorBatchOutput, EhlersSimpleCycleIndicatorError> {
    ehlers_simple_cycle_indicator_batch_inner(data, sweep, kernel, true)
}

fn ehlers_simple_cycle_indicator_batch_inner(
    data: &[f64],
    sweep: &EhlersSimpleCycleIndicatorBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<EhlersSimpleCycleIndicatorBatchOutput, EhlersSimpleCycleIndicatorError> {
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let total = batch_shape(rows, cols)?;

    let mut cycle_buf = make_uninit_matrix(rows, cols);
    let mut trigger_buf = make_uninit_matrix(rows, cols);

    let mut cycle_guard = ManuallyDrop::new(cycle_buf);
    let mut trigger_guard = ManuallyDrop::new(trigger_buf);
    let out_cycle: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(cycle_guard.as_mut_ptr() as *mut f64, cycle_guard.len())
    };
    let out_trigger: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(trigger_guard.as_mut_ptr() as *mut f64, trigger_guard.len())
    };
    ehlers_simple_cycle_indicator_batch_inner_into(
        data,
        sweep,
        kernel,
        parallel,
        out_cycle,
        out_trigger,
    )?;

    let cycle = unsafe {
        Vec::from_raw_parts(
            cycle_guard.as_mut_ptr() as *mut f64,
            total,
            cycle_guard.capacity(),
        )
    };
    let trigger = unsafe {
        Vec::from_raw_parts(
            trigger_guard.as_mut_ptr() as *mut f64,
            total,
            trigger_guard.capacity(),
        )
    };

    Ok(EhlersSimpleCycleIndicatorBatchOutput {
        cycle,
        trigger,
        combos,
        rows,
        cols,
    })
}

pub fn ehlers_simple_cycle_indicator_batch_into_slice(
    out_cycle: &mut [f64],
    out_trigger: &mut [f64],
    data: &[f64],
    sweep: &EhlersSimpleCycleIndicatorBatchRange,
    kernel: Kernel,
) -> Result<(), EhlersSimpleCycleIndicatorError> {
    ehlers_simple_cycle_indicator_batch_inner_into(
        data,
        sweep,
        kernel,
        false,
        out_cycle,
        out_trigger,
    )?;
    Ok(())
}

fn ehlers_simple_cycle_indicator_batch_inner_into(
    data: &[f64],
    sweep: &EhlersSimpleCycleIndicatorBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_cycle: &mut [f64],
    out_trigger: &mut [f64],
) -> Result<Vec<EhlersSimpleCycleIndicatorParams>, EhlersSimpleCycleIndicatorError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slice(data)?;
    let rows = combos.len();
    let cols = data.len();
    let total = batch_shape(rows, cols)?;
    if out_cycle.len() != total {
        return Err(EhlersSimpleCycleIndicatorError::OutputLengthMismatch {
            expected: total,
            got: out_cycle.len(),
        });
    }
    if out_trigger.len() != total {
        return Err(EhlersSimpleCycleIndicatorError::OutputLengthMismatch {
            expected: total,
            got: out_trigger.len(),
        });
    }
    let valid = cols.saturating_sub(first);
    if valid < REQUIRED_VALID_SAMPLES {
        return Err(EhlersSimpleCycleIndicatorError::NotEnoughValidData {
            needed: REQUIRED_VALID_SAMPLES,
            valid,
        });
    }

    let compute_row = |row: usize,
                       cycle_dst: &mut [f64],
                       trigger_dst: &mut [f64]|
     -> Result<(), EhlersSimpleCycleIndicatorError> {
        let params = resolve_params(&combos[row])?;
        compute_esci_into(data, params, cycle_dst, trigger_dst)
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_cycle
                .par_chunks_mut(cols)
                .zip(out_trigger.par_chunks_mut(cols))
                .enumerate()
                .try_for_each(|(row, (cycle_dst, trigger_dst))| {
                    compute_row(row, cycle_dst, trigger_dst)
                })?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for row in 0..rows {
                let start = row * cols;
                let end = start + cols;
                compute_row(
                    row,
                    &mut out_cycle[start..end],
                    &mut out_trigger[start..end],
                )?;
            }
        }
    } else {
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            compute_row(
                row,
                &mut out_cycle[start..end],
                &mut out_trigger[start..end],
            )?;
        }
    }
    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_simple_cycle_indicator")]
#[pyo3(signature = (data, alpha=0.07, kernel=None))]
pub fn ehlers_simple_cycle_indicator_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let input = EhlersSimpleCycleIndicatorInput::from_slice(
        slice,
        EhlersSimpleCycleIndicatorParams { alpha: Some(alpha) },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| ehlers_simple_cycle_indicator_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("cycle", out.cycle.into_pyarray(py))?;
    dict.set_item("trigger", out.trigger.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "EhlersSimpleCycleIndicatorStream")]
pub struct EhlersSimpleCycleIndicatorStreamPy {
    stream: EhlersSimpleCycleIndicatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EhlersSimpleCycleIndicatorStreamPy {
    #[new]
    #[pyo3(signature = (alpha=0.07))]
    fn new(alpha: f64) -> PyResult<Self> {
        let stream = EhlersSimpleCycleIndicatorStream::try_new(EhlersSimpleCycleIndicatorParams {
            alpha: Some(alpha),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> (f64, f64) {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_simple_cycle_indicator_batch")]
#[pyo3(signature = (data, alpha_range=(0.07,0.07,0.0), kernel=None))]
pub fn ehlers_simple_cycle_indicator_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    alpha_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let sweep = EhlersSimpleCycleIndicatorBatchRange { alpha: alpha_range };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;

    let out_cycle = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_trigger = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let cycle_slice = unsafe { out_cycle.as_slice_mut()? };
    let trigger_slice = unsafe { out_trigger.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        ehlers_simple_cycle_indicator_batch_inner_into(
            slice,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            cycle_slice,
            trigger_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("cycle", out_cycle.reshape((rows, cols))?)?;
    dict.set_item("trigger", out_trigger.reshape((rows, cols))?)?;
    dict.set_item(
        "alphas",
        combos
            .iter()
            .map(|combo| combo.alpha.unwrap_or(DEFAULT_ALPHA))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_ehlers_simple_cycle_indicator_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ehlers_simple_cycle_indicator_py, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers_simple_cycle_indicator_batch_py, m)?)?;
    m.add_class::<EhlersSimpleCycleIndicatorStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersSimpleCycleIndicatorJsOutput {
    pub cycle: Vec<f64>,
    pub trigger: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_simple_cycle_indicator_js")]
pub fn ehlers_simple_cycle_indicator_js(data: &[f64], alpha: f64) -> Result<JsValue, JsValue> {
    let input = EhlersSimpleCycleIndicatorInput::from_slice(
        data,
        EhlersSimpleCycleIndicatorParams { alpha: Some(alpha) },
    );
    let out = ehlers_simple_cycle_indicator_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&EhlersSimpleCycleIndicatorJsOutput {
        cycle: out.cycle,
        trigger: out.trigger,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersSimpleCycleIndicatorBatchConfig {
    pub alpha_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersSimpleCycleIndicatorBatchJsOutput {
    pub cycle: Vec<f64>,
    pub trigger: Vec<f64>,
    pub alphas: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_f64(name: &str, values: &[f64]) -> Result<(f64, f64, f64), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} values must be finite numbers"
        )));
    }
    Ok((values[0], values[1], values[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_simple_cycle_indicator_batch_js")]
pub fn ehlers_simple_cycle_indicator_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: EhlersSimpleCycleIndicatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = EhlersSimpleCycleIndicatorBatchRange {
        alpha: js_vec3_to_f64("alpha_range", &config.alpha_range)?,
    };
    let out = ehlers_simple_cycle_indicator_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&EhlersSimpleCycleIndicatorBatchJsOutput {
        cycle: out.cycle,
        trigger: out.trigger,
        alphas: out
            .combos
            .iter()
            .map(|combo| combo.alpha.unwrap_or(DEFAULT_ALPHA))
            .collect(),
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_simple_cycle_indicator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_simple_cycle_indicator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_simple_cycle_indicator_into(
    in_ptr: *const f64,
    out_cycle_ptr: *mut f64,
    out_trigger_ptr: *mut f64,
    len: usize,
    alpha: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_cycle_ptr.is_null() || out_trigger_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_simple_cycle_indicator_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out_cycle = std::slice::from_raw_parts_mut(out_cycle_ptr, len);
        let out_trigger = std::slice::from_raw_parts_mut(out_trigger_ptr, len);
        let input = EhlersSimpleCycleIndicatorInput::from_slice(
            data,
            EhlersSimpleCycleIndicatorParams { alpha: Some(alpha) },
        );
        ehlers_simple_cycle_indicator_into_slice(out_cycle, out_trigger, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_simple_cycle_indicator_batch_into(
    in_ptr: *const f64,
    out_cycle_ptr: *mut f64,
    out_trigger_ptr: *mut f64,
    len: usize,
    alpha_start: f64,
    alpha_end: f64,
    alpha_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_cycle_ptr.is_null() || out_trigger_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_simple_cycle_indicator_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = EhlersSimpleCycleIndicatorBatchRange {
            alpha: (alpha_start, alpha_end, alpha_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows.checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in ehlers_simple_cycle_indicator_batch_into")
        })?;
        let out_cycle = std::slice::from_raw_parts_mut(out_cycle_ptr, total);
        let out_trigger = std::slice::from_raw_parts_mut(out_trigger_ptr, total);
        ehlers_simple_cycle_indicator_batch_into_slice(
            out_cycle,
            out_trigger,
            data,
            &sweep,
            Kernel::Auto,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_simple_cycle_indicator_output_into_js(
    data: &[f64],
    alpha: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_simple_cycle_indicator_js(data, alpha)?;
    crate::write_wasm_object_f64_outputs(
        "ehlers_simple_cycle_indicator_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_simple_cycle_indicator_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_simple_cycle_indicator_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "ehlers_simple_cycle_indicator_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let x = i as f64;
                100.0 + (x * 0.17).sin() * 2.5 + (x * 0.04).cos() * 0.8 + x * 0.03
            })
            .collect()
    }

    fn assert_pair_eq(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            if a[i].is_nan() || b[i].is_nan() {
                assert!(a[i].is_nan() && b[i].is_nan());
            } else {
                assert!((a[i] - b[i]).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn manual_reference_matches_api() {
        let data = sample_data(160);
        let input = EhlersSimpleCycleIndicatorInput::from_slice(
            &data,
            EhlersSimpleCycleIndicatorParams { alpha: Some(0.07) },
        );
        let out = ehlers_simple_cycle_indicator(&input).unwrap();
        let params =
            resolve_params(&EhlersSimpleCycleIndicatorParams { alpha: Some(0.07) }).unwrap();
        let mut core = EsciCore::new(params);
        let mut cycle = vec![f64::NAN; data.len()];
        let mut trigger = vec![f64::NAN; data.len()];
        for i in 0..data.len() {
            let (c, t) = core.update(data[i]);
            cycle[i] = c;
            trigger[i] = t;
        }
        assert_pair_eq(&out.cycle, &cycle);
        assert_pair_eq(&out.trigger, &trigger);
    }

    #[test]
    fn stream_matches_batch() {
        let data = sample_data(144);
        let input = EhlersSimpleCycleIndicatorInput::from_slice(
            &data,
            EhlersSimpleCycleIndicatorParams { alpha: Some(0.07) },
        );
        let batch = ehlers_simple_cycle_indicator(&input).unwrap();
        let mut stream =
            EhlersSimpleCycleIndicatorStream::try_new(EhlersSimpleCycleIndicatorParams {
                alpha: Some(0.07),
            })
            .unwrap();
        let mut cycle = vec![f64::NAN; data.len()];
        let mut trigger = vec![f64::NAN; data.len()];
        for i in 0..data.len() {
            let (c, t) = stream.update(data[i]);
            cycle[i] = c;
            trigger[i] = t;
        }
        assert_pair_eq(&batch.cycle, &cycle);
        assert_pair_eq(&batch.trigger, &trigger);
    }

    #[test]
    fn batch_first_row_matches_single() {
        let data = sample_data(128);
        let sweep = EhlersSimpleCycleIndicatorBatchRange {
            alpha: (0.07, 0.09, 0.02),
        };
        let batch =
            ehlers_simple_cycle_indicator_batch_with_kernel(&data, &sweep, Kernel::Auto).unwrap();
        let single_input = EhlersSimpleCycleIndicatorInput::from_slice(
            &data,
            EhlersSimpleCycleIndicatorParams { alpha: Some(0.07) },
        );
        let single = ehlers_simple_cycle_indicator(&single_input).unwrap();
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, data.len());
        assert_pair_eq(&batch.cycle[0..data.len()], &single.cycle);
        assert_pair_eq(&batch.trigger[0..data.len()], &single.trigger);
    }

    #[test]
    fn into_slice_matches_single() {
        let data = sample_data(100);
        let input = EhlersSimpleCycleIndicatorInput::from_slice(
            &data,
            EhlersSimpleCycleIndicatorParams { alpha: Some(0.07) },
        );
        let single = ehlers_simple_cycle_indicator(&input).unwrap();
        let mut cycle = vec![f64::NAN; data.len()];
        let mut trigger = vec![f64::NAN; data.len()];
        ehlers_simple_cycle_indicator_into_slice(&mut cycle, &mut trigger, &input, Kernel::Auto)
            .unwrap();
        assert_pair_eq(&cycle, &single.cycle);
        assert_pair_eq(&trigger, &single.trigger);
    }

    #[test]
    fn invalid_alpha_is_rejected() {
        let data = sample_data(32);
        let input = EhlersSimpleCycleIndicatorInput::from_slice(
            &data,
            EhlersSimpleCycleIndicatorParams { alpha: Some(1.5) },
        );
        let err = ehlers_simple_cycle_indicator(&input).unwrap_err();
        assert!(matches!(
            err,
            EhlersSimpleCycleIndicatorError::InvalidAlpha { .. }
        ));
    }
}
