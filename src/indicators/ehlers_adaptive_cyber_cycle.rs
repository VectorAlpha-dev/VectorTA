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
    alloc_with_nan_prefix, detect_best_batch_kernel, init_matrix_prefixes, make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "hl2";
const DEFAULT_ALPHA: f64 = 0.07;
const REQUIRED_VALID_SAMPLES: usize = 3;
const WARMUP_CYCLE: usize = 2;
const WARMUP_TRIGGER: usize = 3;

#[derive(Debug, Clone)]
pub enum EhlersAdaptiveCyberCycleData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersAdaptiveCyberCycleOutput {
    pub cycle: Vec<f64>,
    pub trigger: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EhlersAdaptiveCyberCycleParams {
    pub alpha: Option<f64>,
}

impl Default for EhlersAdaptiveCyberCycleParams {
    fn default() -> Self {
        Self {
            alpha: Some(DEFAULT_ALPHA),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersAdaptiveCyberCycleInput<'a> {
    pub data: EhlersAdaptiveCyberCycleData<'a>,
    pub params: EhlersAdaptiveCyberCycleParams,
}

impl<'a> EhlersAdaptiveCyberCycleInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: EhlersAdaptiveCyberCycleParams,
    ) -> Self {
        Self {
            data: EhlersAdaptiveCyberCycleData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: EhlersAdaptiveCyberCycleParams) -> Self {
        Self {
            data: EhlersAdaptiveCyberCycleData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            EhlersAdaptiveCyberCycleParams::default(),
        )
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersAdaptiveCyberCycleBuilder {
    source: Option<&'static str>,
    alpha: Option<f64>,
    kernel: Kernel,
}

impl Default for EhlersAdaptiveCyberCycleBuilder {
    fn default() -> Self {
        Self {
            source: None,
            alpha: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersAdaptiveCyberCycleBuilder {
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
    ) -> Result<EhlersAdaptiveCyberCycleOutput, EhlersAdaptiveCyberCycleError> {
        let input = EhlersAdaptiveCyberCycleInput::from_candles(
            candles,
            self.source.unwrap_or(DEFAULT_SOURCE),
            EhlersAdaptiveCyberCycleParams { alpha: self.alpha },
        );
        ehlers_adaptive_cyber_cycle_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EhlersAdaptiveCyberCycleOutput, EhlersAdaptiveCyberCycleError> {
        let input = EhlersAdaptiveCyberCycleInput::from_slice(
            data,
            EhlersAdaptiveCyberCycleParams { alpha: self.alpha },
        );
        ehlers_adaptive_cyber_cycle_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<EhlersAdaptiveCyberCycleStream, EhlersAdaptiveCyberCycleError> {
        EhlersAdaptiveCyberCycleStream::try_new(EhlersAdaptiveCyberCycleParams {
            alpha: self.alpha,
        })
    }
}

#[derive(Debug, Error)]
pub enum EhlersAdaptiveCyberCycleError {
    #[error("ehlers_adaptive_cyber_cycle: Input data slice is empty.")]
    EmptyInputData,
    #[error("ehlers_adaptive_cyber_cycle: All values are NaN.")]
    AllValuesNaN,
    #[error("ehlers_adaptive_cyber_cycle: Invalid alpha: {alpha}")]
    InvalidAlpha { alpha: f64 },
    #[error("ehlers_adaptive_cyber_cycle: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ehlers_adaptive_cyber_cycle: Output length mismatch: expected={expected}, got={got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ehlers_adaptive_cyber_cycle: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("ehlers_adaptive_cyber_cycle: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Clone, Copy, Debug)]
struct ResolvedParams {
    alpha: f64,
    cycle_coef: f64,
    cycle_prev1: f64,
    cycle_prev2: f64,
}

#[inline(always)]
fn nz(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[inline(always)]
fn median3(a: f64, b: f64, c: f64) -> f64 {
    if !(a.is_finite() && b.is_finite() && c.is_finite()) {
        return f64::NAN;
    }
    (a + b + c) - a.min(b.min(c)) - a.max(b.max(c))
}

#[inline(always)]
fn extract_slice<'a>(
    input: &'a EhlersAdaptiveCyberCycleInput<'a>,
) -> Result<&'a [f64], EhlersAdaptiveCyberCycleError> {
    let data = match &input.data {
        EhlersAdaptiveCyberCycleData::Candles { candles, source } => source_type(candles, source),
        EhlersAdaptiveCyberCycleData::Slice(values) => *values,
    };
    if data.is_empty() {
        return Err(EhlersAdaptiveCyberCycleError::EmptyInputData);
    }
    Ok(data)
}

#[inline(always)]
fn first_valid(data: &[f64]) -> Option<usize> {
    data.iter().position(|v| v.is_finite())
}

#[inline(always)]
fn resolve_params(
    params: &EhlersAdaptiveCyberCycleParams,
) -> Result<ResolvedParams, EhlersAdaptiveCyberCycleError> {
    let alpha = params.alpha.unwrap_or(DEFAULT_ALPHA);
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(EhlersAdaptiveCyberCycleError::InvalidAlpha { alpha });
    }
    let one_minus_alpha = 1.0 - alpha;
    Ok(ResolvedParams {
        alpha,
        cycle_coef: (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha),
        cycle_prev1: 2.0 * one_minus_alpha,
        cycle_prev2: one_minus_alpha * one_minus_alpha,
    })
}

#[inline(always)]
fn validate_input<'a>(
    input: &'a EhlersAdaptiveCyberCycleInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], ResolvedParams, usize, Kernel), EhlersAdaptiveCyberCycleError> {
    let data = extract_slice(input)?;
    let params = resolve_params(&input.params)?;
    let first = first_valid(data).ok_or(EhlersAdaptiveCyberCycleError::AllValuesNaN)?;
    let valid = data.len().saturating_sub(first);
    if valid < REQUIRED_VALID_SAMPLES {
        return Err(EhlersAdaptiveCyberCycleError::NotEnoughValidData {
            needed: REQUIRED_VALID_SAMPLES,
            valid,
        });
    }
    Ok((data, params, first, kernel.to_non_batch()))
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
struct EaccCore {
    params: ResolvedParams,
    src_ring: [f64; 4],
    src_idx: usize,
    smooth_ring: [f64; 3],
    smooth_idx: usize,
    cycle_ring: [f64; 7],
    cycle_idx: usize,
    dp_ring: [f64; 5],
    dp_idx: usize,
    adaptive_hist: [f64; 2],
    prev_ip: f64,
    prev_p: f64,
    prev_q1: f64,
    prev_i1: f64,
    valid_count: usize,
}

impl EaccCore {
    #[inline(always)]
    fn new(params: ResolvedParams) -> Self {
        let _ = params.alpha;
        Self {
            params,
            src_ring: [f64::NAN; 4],
            src_idx: 0,
            smooth_ring: [f64::NAN; 3],
            smooth_idx: 0,
            cycle_ring: [f64::NAN; 7],
            cycle_idx: 0,
            dp_ring: [f64::NAN; 5],
            dp_idx: 0,
            adaptive_hist: [f64::NAN; 2],
            prev_ip: f64::NAN,
            prev_p: f64::NAN,
            prev_q1: f64::NAN,
            prev_i1: f64::NAN,
            valid_count: 0,
        }
    }

    #[inline(always)]
    fn update(&mut self, source: f64) -> (f64, f64) {
        if !source.is_finite() {
            return (f64::NAN, f64::NAN);
        }

        let bar = self.valid_count;
        self.src_ring[self.src_idx] = source;

        let src0 = ring_get(&self.src_ring, self.src_idx, 0);
        let src1 = ring_get(&self.src_ring, self.src_idx, 1);
        let src2 = ring_get(&self.src_ring, self.src_idx, 2);
        let src3 = ring_get(&self.src_ring, self.src_idx, 3);

        let smooth = if src0.is_finite() && src1.is_finite() && src2.is_finite() && src3.is_finite()
        {
            (src0 + 2.0 * src1 + 2.0 * src2 + src3) / 6.0
        } else {
            f64::NAN
        };
        self.smooth_ring[self.smooth_idx] = smooth;

        let smooth1 = ring_get(&self.smooth_ring, self.smooth_idx, 1);
        let smooth2 = ring_get(&self.smooth_ring, self.smooth_idx, 2);
        let cycle_prev1 = ring_get(&self.cycle_ring, self.cycle_idx, 1);
        let cycle_prev2 = ring_get(&self.cycle_ring, self.cycle_idx, 2);

        let cycle_main = if smooth.is_finite()
            && smooth1.is_finite()
            && smooth2.is_finite()
            && cycle_prev1.is_finite()
            && cycle_prev2.is_finite()
        {
            self.params.cycle_coef * (smooth - 2.0 * smooth1 + smooth2)
                + self.params.cycle_prev1 * cycle_prev1
                - self.params.cycle_prev2 * cycle_prev2
        } else {
            f64::NAN
        };

        let cycle_fallback = if src0.is_finite() && src1.is_finite() && src2.is_finite() {
            (src0 - 2.0 * src1 + src2) / 4.0
        } else {
            f64::NAN
        };

        let cycle = if bar < 7 { cycle_fallback } else { cycle_main };
        self.cycle_ring[self.cycle_idx] = cycle;

        let q1 = if cycle.is_finite() {
            (0.0962 * cycle + 0.5769 * nz(ring_get(&self.cycle_ring, self.cycle_idx, 2))
                - 0.5769 * nz(ring_get(&self.cycle_ring, self.cycle_idx, 4))
                - 0.0962 * nz(ring_get(&self.cycle_ring, self.cycle_idx, 6)))
                * (0.5 + 0.08 * nz(self.prev_ip))
        } else {
            f64::NAN
        };
        let i1 = nz(ring_get(&self.cycle_ring, self.cycle_idx, 3));

        let dp_raw =
            if q1.is_finite() && self.prev_q1.is_finite() && q1 != 0.0 && self.prev_q1 != 0.0 {
                let numer = (i1 / q1) - (nz(self.prev_i1) / nz(self.prev_q1));
                let denom = 1.0 + i1 * nz(self.prev_i1) / (q1 * nz(self.prev_q1));
                numer / denom
            } else {
                0.0
            };
        let dp = if dp_raw < 0.1 {
            0.1
        } else if dp_raw > 1.1 {
            1.1
        } else {
            dp_raw
        };
        self.dp_ring[self.dp_idx] = dp;

        let md_inner = median3(
            ring_get(&self.dp_ring, self.dp_idx, 2),
            ring_get(&self.dp_ring, self.dp_idx, 3),
            ring_get(&self.dp_ring, self.dp_idx, 4),
        );
        let md = median3(dp, ring_get(&self.dp_ring, self.dp_idx, 1), md_inner);
        let dc = if md == 0.0 {
            15.0
        } else {
            (2.0 * std::f64::consts::PI / md) + 0.5
        };
        let ip = 0.33 * dc + 0.67 * nz(self.prev_ip);
        let p = 0.15 * ip + 0.85 * nz(self.prev_p);
        let a1 = 2.0 / (p + 1.0);

        let adaptive_main = if smooth.is_finite()
            && smooth1.is_finite()
            && smooth2.is_finite()
            && self.adaptive_hist[0].is_finite()
            && self.adaptive_hist[1].is_finite()
            && a1.is_finite()
        {
            let adapt_coef = (1.0 - 0.5 * a1) * (1.0 - 0.5 * a1);
            let one_minus_a1 = 1.0 - a1;
            adapt_coef * (smooth - 2.0 * smooth1 + smooth2)
                + 2.0 * one_minus_a1 * self.adaptive_hist[0]
                - one_minus_a1 * one_minus_a1 * self.adaptive_hist[1]
        } else {
            f64::NAN
        };

        let adaptive_cycle = if bar < 7 {
            cycle_fallback
        } else if adaptive_main.is_finite() {
            adaptive_main
        } else {
            cycle_fallback
        };
        let trigger = self.adaptive_hist[0];

        self.prev_q1 = q1;
        self.prev_i1 = i1;
        self.prev_ip = ip;
        self.prev_p = p;
        self.adaptive_hist[1] = self.adaptive_hist[0];
        self.adaptive_hist[0] = adaptive_cycle;

        self.valid_count += 1;
        self.src_idx = (self.src_idx + 1) % self.src_ring.len();
        self.smooth_idx = (self.smooth_idx + 1) % self.smooth_ring.len();
        self.cycle_idx = (self.cycle_idx + 1) % self.cycle_ring.len();
        self.dp_idx = (self.dp_idx + 1) % self.dp_ring.len();

        (adaptive_cycle, trigger)
    }
}

#[inline(always)]
fn compute_eacc_into(
    data: &[f64],
    params: ResolvedParams,
    out_cycle: &mut [f64],
    out_trigger: &mut [f64],
) -> Result<(), EhlersAdaptiveCyberCycleError> {
    if out_cycle.len() != data.len() {
        return Err(EhlersAdaptiveCyberCycleError::OutputLengthMismatch {
            expected: data.len(),
            got: out_cycle.len(),
        });
    }
    if out_trigger.len() != data.len() {
        return Err(EhlersAdaptiveCyberCycleError::OutputLengthMismatch {
            expected: data.len(),
            got: out_trigger.len(),
        });
    }

    let mut core = EaccCore::new(params);
    for i in 0..data.len() {
        let (cycle, trigger) = core.update(data[i]);
        out_cycle[i] = cycle;
        out_trigger[i] = trigger;
    }
    Ok(())
}

#[inline]
pub fn ehlers_adaptive_cyber_cycle(
    input: &EhlersAdaptiveCyberCycleInput,
) -> Result<EhlersAdaptiveCyberCycleOutput, EhlersAdaptiveCyberCycleError> {
    ehlers_adaptive_cyber_cycle_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn ehlers_adaptive_cyber_cycle_with_kernel(
    input: &EhlersAdaptiveCyberCycleInput,
    kernel: Kernel,
) -> Result<EhlersAdaptiveCyberCycleOutput, EhlersAdaptiveCyberCycleError> {
    let (data, params, first, _kernel) = validate_input(input, kernel)?;
    let mut cycle = alloc_with_nan_prefix(data.len(), (first + WARMUP_CYCLE).min(data.len()));
    let mut trigger = alloc_with_nan_prefix(data.len(), (first + WARMUP_TRIGGER).min(data.len()));
    compute_eacc_into(data, params, &mut cycle, &mut trigger)?;
    Ok(EhlersAdaptiveCyberCycleOutput { cycle, trigger })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn ehlers_adaptive_cyber_cycle_into(
    out_cycle: &mut [f64],
    out_trigger: &mut [f64],
    input: &EhlersAdaptiveCyberCycleInput,
    kernel: Kernel,
) -> Result<(), EhlersAdaptiveCyberCycleError> {
    ehlers_adaptive_cyber_cycle_into_slice(out_cycle, out_trigger, input, kernel)
}

#[inline]
pub fn ehlers_adaptive_cyber_cycle_into_slice(
    out_cycle: &mut [f64],
    out_trigger: &mut [f64],
    input: &EhlersAdaptiveCyberCycleInput,
    kernel: Kernel,
) -> Result<(), EhlersAdaptiveCyberCycleError> {
    let (data, params, _first, _kernel) = validate_input(input, kernel)?;
    compute_eacc_into(data, params, out_cycle, out_trigger)
}

#[derive(Clone, Debug)]
pub struct EhlersAdaptiveCyberCycleStream {
    core: EaccCore,
}

impl EhlersAdaptiveCyberCycleStream {
    pub fn try_new(
        params: EhlersAdaptiveCyberCycleParams,
    ) -> Result<Self, EhlersAdaptiveCyberCycleError> {
        Ok(Self {
            core: EaccCore::new(resolve_params(&params)?),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> (f64, f64) {
        self.core.update(value)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EhlersAdaptiveCyberCycleBatchRange {
    pub alpha: (f64, f64, f64),
}

impl Default for EhlersAdaptiveCyberCycleBatchRange {
    fn default() -> Self {
        Self {
            alpha: (DEFAULT_ALPHA, DEFAULT_ALPHA, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EhlersAdaptiveCyberCycleBatchOutput {
    pub cycle: Vec<f64>,
    pub trigger: Vec<f64>,
    pub combos: Vec<EhlersAdaptiveCyberCycleParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct EhlersAdaptiveCyberCycleBatchBuilder {
    source: Option<&'static str>,
    range: EhlersAdaptiveCyberCycleBatchRange,
    kernel: Kernel,
}

impl Default for EhlersAdaptiveCyberCycleBatchBuilder {
    fn default() -> Self {
        Self {
            source: None,
            range: EhlersAdaptiveCyberCycleBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersAdaptiveCyberCycleBatchBuilder {
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
    ) -> Result<EhlersAdaptiveCyberCycleBatchOutput, EhlersAdaptiveCyberCycleError> {
        ehlers_adaptive_cyber_cycle_batch_with_kernel(
            source_type(candles, self.source.unwrap_or(DEFAULT_SOURCE)),
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EhlersAdaptiveCyberCycleBatchOutput, EhlersAdaptiveCyberCycleError> {
        ehlers_adaptive_cyber_cycle_batch_with_kernel(data, &self.range, self.kernel)
    }
}

#[inline(always)]
fn expand_float_range(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, EhlersAdaptiveCyberCycleError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(EhlersAdaptiveCyberCycleError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 {
        if (start - end).abs() > 1e-12 {
            return Err(EhlersAdaptiveCyberCycleError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if start > end || step < 0.0 {
        return Err(EhlersAdaptiveCyberCycleError::InvalidRange {
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
            return Err(EhlersAdaptiveCyberCycleError::InvalidRange {
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
    sweep: &EhlersAdaptiveCyberCycleBatchRange,
) -> Result<Vec<EhlersAdaptiveCyberCycleParams>, EhlersAdaptiveCyberCycleError> {
    let alphas = expand_float_range(sweep.alpha.0, sweep.alpha.1, sweep.alpha.2)?;
    let mut combos = Vec::with_capacity(alphas.len());
    for alpha in alphas {
        combos.push(EhlersAdaptiveCyberCycleParams { alpha: Some(alpha) });
    }
    Ok(combos)
}

#[inline(always)]
fn validate_raw_slice(data: &[f64]) -> Result<usize, EhlersAdaptiveCyberCycleError> {
    if data.is_empty() {
        return Err(EhlersAdaptiveCyberCycleError::EmptyInputData);
    }
    first_valid(data).ok_or(EhlersAdaptiveCyberCycleError::AllValuesNaN)
}

#[inline(always)]
fn batch_shape(rows: usize, cols: usize) -> Result<usize, EhlersAdaptiveCyberCycleError> {
    rows.checked_mul(cols)
        .ok_or_else(|| EhlersAdaptiveCyberCycleError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        })
}

pub fn ehlers_adaptive_cyber_cycle_batch_with_kernel(
    data: &[f64],
    sweep: &EhlersAdaptiveCyberCycleBatchRange,
    kernel: Kernel,
) -> Result<EhlersAdaptiveCyberCycleBatchOutput, EhlersAdaptiveCyberCycleError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(EhlersAdaptiveCyberCycleError::InvalidKernelForBatch(kernel));
        }
    };
    ehlers_adaptive_cyber_cycle_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn ehlers_adaptive_cyber_cycle_batch_slice(
    data: &[f64],
    sweep: &EhlersAdaptiveCyberCycleBatchRange,
    kernel: Kernel,
) -> Result<EhlersAdaptiveCyberCycleBatchOutput, EhlersAdaptiveCyberCycleError> {
    ehlers_adaptive_cyber_cycle_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn ehlers_adaptive_cyber_cycle_batch_par_slice(
    data: &[f64],
    sweep: &EhlersAdaptiveCyberCycleBatchRange,
    kernel: Kernel,
) -> Result<EhlersAdaptiveCyberCycleBatchOutput, EhlersAdaptiveCyberCycleError> {
    ehlers_adaptive_cyber_cycle_batch_inner(data, sweep, kernel, true)
}

fn ehlers_adaptive_cyber_cycle_batch_inner(
    data: &[f64],
    sweep: &EhlersAdaptiveCyberCycleBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<EhlersAdaptiveCyberCycleBatchOutput, EhlersAdaptiveCyberCycleError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slice(data)?;
    let rows = combos.len();
    let cols = data.len();
    let total = batch_shape(rows, cols)?;
    let cycle_warmups = vec![(first + WARMUP_CYCLE).min(cols); rows];
    let trigger_warmups = vec![(first + WARMUP_TRIGGER).min(cols); rows];

    let mut cycle_buf = make_uninit_matrix(rows, cols);
    let mut trigger_buf = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut cycle_buf, cols, &cycle_warmups);
    init_matrix_prefixes(&mut trigger_buf, cols, &trigger_warmups);

    let mut cycle_guard = ManuallyDrop::new(cycle_buf);
    let mut trigger_guard = ManuallyDrop::new(trigger_buf);
    let out_cycle: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(cycle_guard.as_mut_ptr() as *mut f64, cycle_guard.len())
    };
    let out_trigger: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(trigger_guard.as_mut_ptr() as *mut f64, trigger_guard.len())
    };

    ehlers_adaptive_cyber_cycle_batch_inner_into(
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

    Ok(EhlersAdaptiveCyberCycleBatchOutput {
        cycle,
        trigger,
        combos,
        rows,
        cols,
    })
}

pub fn ehlers_adaptive_cyber_cycle_batch_into_slice(
    out_cycle: &mut [f64],
    out_trigger: &mut [f64],
    data: &[f64],
    sweep: &EhlersAdaptiveCyberCycleBatchRange,
    kernel: Kernel,
) -> Result<(), EhlersAdaptiveCyberCycleError> {
    ehlers_adaptive_cyber_cycle_batch_inner_into(
        data,
        sweep,
        kernel,
        false,
        out_cycle,
        out_trigger,
    )?;
    Ok(())
}

fn ehlers_adaptive_cyber_cycle_batch_inner_into(
    data: &[f64],
    sweep: &EhlersAdaptiveCyberCycleBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out_cycle: &mut [f64],
    out_trigger: &mut [f64],
) -> Result<Vec<EhlersAdaptiveCyberCycleParams>, EhlersAdaptiveCyberCycleError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slice(data)?;
    let rows = combos.len();
    let cols = data.len();
    let total = batch_shape(rows, cols)?;
    if out_cycle.len() != total {
        return Err(EhlersAdaptiveCyberCycleError::OutputLengthMismatch {
            expected: total,
            got: out_cycle.len(),
        });
    }
    if out_trigger.len() != total {
        return Err(EhlersAdaptiveCyberCycleError::OutputLengthMismatch {
            expected: total,
            got: out_trigger.len(),
        });
    }
    let valid = cols.saturating_sub(first);
    if valid < REQUIRED_VALID_SAMPLES {
        return Err(EhlersAdaptiveCyberCycleError::NotEnoughValidData {
            needed: REQUIRED_VALID_SAMPLES,
            valid,
        });
    }

    let compute_row = |row: usize,
                       cycle_dst: &mut [f64],
                       trigger_dst: &mut [f64]|
     -> Result<(), EhlersAdaptiveCyberCycleError> {
        let params = resolve_params(&combos[row])?;
        compute_eacc_into(data, params, cycle_dst, trigger_dst)
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
#[pyfunction(name = "ehlers_adaptive_cyber_cycle")]
#[pyo3(signature = (data, alpha=0.07, kernel=None))]
pub fn ehlers_adaptive_cyber_cycle_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let input = EhlersAdaptiveCyberCycleInput::from_slice(
        slice,
        EhlersAdaptiveCyberCycleParams { alpha: Some(alpha) },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| ehlers_adaptive_cyber_cycle_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("cycle", out.cycle.into_pyarray(py))?;
    dict.set_item("trigger", out.trigger.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "EhlersAdaptiveCyberCycleStream")]
pub struct EhlersAdaptiveCyberCycleStreamPy {
    stream: EhlersAdaptiveCyberCycleStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EhlersAdaptiveCyberCycleStreamPy {
    #[new]
    #[pyo3(signature = (alpha=0.07))]
    fn new(alpha: f64) -> PyResult<Self> {
        let stream = EhlersAdaptiveCyberCycleStream::try_new(EhlersAdaptiveCyberCycleParams {
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
#[pyfunction(name = "ehlers_adaptive_cyber_cycle_batch")]
#[pyo3(signature = (data, alpha_range=(0.07,0.07,0.0), kernel=None))]
pub fn ehlers_adaptive_cyber_cycle_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    alpha_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let sweep = EhlersAdaptiveCyberCycleBatchRange { alpha: alpha_range };
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
        ehlers_adaptive_cyber_cycle_batch_inner_into(
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
pub fn register_ehlers_adaptive_cyber_cycle_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ehlers_adaptive_cyber_cycle_py, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers_adaptive_cyber_cycle_batch_py, m)?)?;
    m.add_class::<EhlersAdaptiveCyberCycleStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersAdaptiveCyberCycleJsOutput {
    pub cycle: Vec<f64>,
    pub trigger: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_adaptive_cyber_cycle_js")]
pub fn ehlers_adaptive_cyber_cycle_js(data: &[f64], alpha: f64) -> Result<JsValue, JsValue> {
    let input = EhlersAdaptiveCyberCycleInput::from_slice(
        data,
        EhlersAdaptiveCyberCycleParams { alpha: Some(alpha) },
    );
    let out = ehlers_adaptive_cyber_cycle_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&EhlersAdaptiveCyberCycleJsOutput {
        cycle: out.cycle,
        trigger: out.trigger,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersAdaptiveCyberCycleBatchConfig {
    pub alpha_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersAdaptiveCyberCycleBatchJsOutput {
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
#[wasm_bindgen(js_name = "ehlers_adaptive_cyber_cycle_batch_js")]
pub fn ehlers_adaptive_cyber_cycle_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: EhlersAdaptiveCyberCycleBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = EhlersAdaptiveCyberCycleBatchRange {
        alpha: js_vec3_to_f64("alpha_range", &config.alpha_range)?,
    };
    let out = ehlers_adaptive_cyber_cycle_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&EhlersAdaptiveCyberCycleBatchJsOutput {
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
pub fn ehlers_adaptive_cyber_cycle_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_adaptive_cyber_cycle_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_adaptive_cyber_cycle_into(
    in_ptr: *const f64,
    out_cycle_ptr: *mut f64,
    out_trigger_ptr: *mut f64,
    len: usize,
    alpha: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_cycle_ptr.is_null() || out_trigger_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_adaptive_cyber_cycle_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out_cycle = std::slice::from_raw_parts_mut(out_cycle_ptr, len);
        let out_trigger = std::slice::from_raw_parts_mut(out_trigger_ptr, len);
        let input = EhlersAdaptiveCyberCycleInput::from_slice(
            data,
            EhlersAdaptiveCyberCycleParams { alpha: Some(alpha) },
        );
        ehlers_adaptive_cyber_cycle_into_slice(out_cycle, out_trigger, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_adaptive_cyber_cycle_batch_into(
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
            "null pointer passed to ehlers_adaptive_cyber_cycle_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = EhlersAdaptiveCyberCycleBatchRange {
            alpha: (alpha_start, alpha_end, alpha_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows.checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in ehlers_adaptive_cyber_cycle_batch_into")
        })?;
        let out_cycle = std::slice::from_raw_parts_mut(out_cycle_ptr, total);
        let out_trigger = std::slice::from_raw_parts_mut(out_trigger_ptr, total);
        ehlers_adaptive_cyber_cycle_batch_into_slice(
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
pub fn ehlers_adaptive_cyber_cycle_output_into_js(
    data: &[f64],
    alpha: f64,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_adaptive_cyber_cycle_js(data, alpha)?;
    crate::write_wasm_object_f64_outputs("ehlers_adaptive_cyber_cycle_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_adaptive_cyber_cycle_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_adaptive_cyber_cycle_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "ehlers_adaptive_cyber_cycle_batch_output_into_js",
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

    fn manual_reference(data: &[f64], alpha: f64) -> (Vec<f64>, Vec<f64>) {
        let len = data.len();
        let mut smooth = vec![f64::NAN; len];
        let mut cycle = vec![f64::NAN; len];
        let mut q1 = vec![f64::NAN; len];
        let mut i1 = vec![0.0; len];
        let mut dp = vec![f64::NAN; len];
        let mut ip = vec![f64::NAN; len];
        let mut p = vec![f64::NAN; len];
        let mut adaptive = vec![f64::NAN; len];
        let mut trigger = vec![f64::NAN; len];

        let cycle_coef = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
        let cycle_prev1 = 2.0 * (1.0 - alpha);
        let cycle_prev2 = (1.0 - alpha) * (1.0 - alpha);

        for idx in 0..len {
            if !data[idx].is_finite() {
                continue;
            }
            if idx >= 3
                && data[idx - 1].is_finite()
                && data[idx - 2].is_finite()
                && data[idx - 3].is_finite()
            {
                smooth[idx] =
                    (data[idx] + 2.0 * data[idx - 1] + 2.0 * data[idx - 2] + data[idx - 3]) / 6.0;
            }

            let cycle_fallback =
                if idx >= 2 && data[idx - 1].is_finite() && data[idx - 2].is_finite() {
                    (data[idx] - 2.0 * data[idx - 1] + data[idx - 2]) / 4.0
                } else {
                    f64::NAN
                };

            let cycle_main = if idx >= 2
                && smooth[idx].is_finite()
                && smooth[idx - 1].is_finite()
                && smooth[idx - 2].is_finite()
                && cycle[idx - 1].is_finite()
                && cycle[idx - 2].is_finite()
            {
                cycle_coef * (smooth[idx] - 2.0 * smooth[idx - 1] + smooth[idx - 2])
                    + cycle_prev1 * cycle[idx - 1]
                    - cycle_prev2 * cycle[idx - 2]
            } else {
                f64::NAN
            };

            cycle[idx] = if idx < 7 { cycle_fallback } else { cycle_main };

            q1[idx] = if cycle[idx].is_finite() {
                let c2 = if idx >= 2 && cycle[idx - 2].is_finite() {
                    cycle[idx - 2]
                } else {
                    0.0
                };
                let c4 = if idx >= 4 && cycle[idx - 4].is_finite() {
                    cycle[idx - 4]
                } else {
                    0.0
                };
                let c6 = if idx >= 6 && cycle[idx - 6].is_finite() {
                    cycle[idx - 6]
                } else {
                    0.0
                };
                let ip1 = if idx >= 1 && ip[idx - 1].is_finite() {
                    ip[idx - 1]
                } else {
                    0.0
                };
                (0.0962 * cycle[idx] + 0.5769 * c2 - 0.5769 * c4 - 0.0962 * c6) * (0.5 + 0.08 * ip1)
            } else {
                f64::NAN
            };

            i1[idx] = if idx >= 3 && cycle[idx - 3].is_finite() {
                cycle[idx - 3]
            } else {
                0.0
            };

            let prev_q1 = if idx >= 1 { q1[idx - 1] } else { f64::NAN };
            let prev_i1 = if idx >= 1 { i1[idx - 1] } else { 0.0 };
            let dp_raw =
                if q1[idx].is_finite() && prev_q1.is_finite() && q1[idx] != 0.0 && prev_q1 != 0.0 {
                    let numer = (i1[idx] / q1[idx]) - (prev_i1 / prev_q1);
                    let denom = 1.0 + i1[idx] * prev_i1 / (q1[idx] * prev_q1);
                    numer / denom
                } else {
                    0.0
                };
            dp[idx] = if dp_raw < 0.1 {
                0.1
            } else if dp_raw > 1.1 {
                1.1
            } else {
                dp_raw
            };

            let md_inner = if idx >= 4 {
                median3(dp[idx - 2], dp[idx - 3], dp[idx - 4])
            } else {
                f64::NAN
            };
            let md = if idx >= 1 {
                median3(dp[idx], dp[idx - 1], md_inner)
            } else {
                f64::NAN
            };
            let dc = if md == 0.0 {
                15.0
            } else {
                (2.0 * std::f64::consts::PI / md) + 0.5
            };
            ip[idx] = 0.33 * dc
                + 0.67
                    * if idx >= 1 && ip[idx - 1].is_finite() {
                        ip[idx - 1]
                    } else {
                        0.0
                    };
            p[idx] = 0.15 * ip[idx]
                + 0.85
                    * if idx >= 1 && p[idx - 1].is_finite() {
                        p[idx - 1]
                    } else {
                        0.0
                    };

            let adaptive_main = if idx >= 2
                && smooth[idx].is_finite()
                && smooth[idx - 1].is_finite()
                && smooth[idx - 2].is_finite()
                && adaptive[idx - 1].is_finite()
                && adaptive[idx - 2].is_finite()
            {
                let a1 = 2.0 / (p[idx] + 1.0);
                let adapt_coef = (1.0 - 0.5 * a1) * (1.0 - 0.5 * a1);
                let one_minus_a1 = 1.0 - a1;
                adapt_coef * (smooth[idx] - 2.0 * smooth[idx - 1] + smooth[idx - 2])
                    + 2.0 * one_minus_a1 * adaptive[idx - 1]
                    - one_minus_a1 * one_minus_a1 * adaptive[idx - 2]
            } else {
                f64::NAN
            };

            adaptive[idx] = if idx < 7 {
                cycle_fallback
            } else if adaptive_main.is_finite() {
                adaptive_main
            } else {
                cycle_fallback
            };

            if idx >= 1 && adaptive[idx - 1].is_finite() {
                trigger[idx] = adaptive[idx - 1];
            }
        }

        (adaptive, trigger)
    }

    fn assert_pair_eq(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            if a[i].is_nan() || b[i].is_nan() {
                assert!(
                    a[i].is_nan() && b[i].is_nan(),
                    "mismatch at {i}: {} vs {}",
                    a[i],
                    b[i]
                );
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
        let data = sample_data(192);
        let input = EhlersAdaptiveCyberCycleInput::from_slice(
            &data,
            EhlersAdaptiveCyberCycleParams { alpha: Some(0.07) },
        );
        let out = ehlers_adaptive_cyber_cycle(&input).unwrap();
        let (cycle, trigger) = manual_reference(&data, 0.07);
        assert_pair_eq(&out.cycle, &cycle);
        assert_pair_eq(&out.trigger, &trigger);
    }

    #[test]
    fn stream_matches_batch() {
        let data = sample_data(160);
        let input = EhlersAdaptiveCyberCycleInput::from_slice(
            &data,
            EhlersAdaptiveCyberCycleParams { alpha: Some(0.07) },
        );
        let batch = ehlers_adaptive_cyber_cycle(&input).unwrap();
        let mut stream = EhlersAdaptiveCyberCycleStream::try_new(EhlersAdaptiveCyberCycleParams {
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
        let data = sample_data(144);
        let sweep = EhlersAdaptiveCyberCycleBatchRange {
            alpha: (0.07, 0.09, 0.02),
        };
        let batch =
            ehlers_adaptive_cyber_cycle_batch_with_kernel(&data, &sweep, Kernel::Auto).unwrap();
        let single_input = EhlersAdaptiveCyberCycleInput::from_slice(
            &data,
            EhlersAdaptiveCyberCycleParams { alpha: Some(0.07) },
        );
        let single = ehlers_adaptive_cyber_cycle(&single_input).unwrap();
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, data.len());
        assert_pair_eq(&batch.cycle[0..data.len()], &single.cycle);
        assert_pair_eq(&batch.trigger[0..data.len()], &single.trigger);
    }

    #[test]
    fn into_slice_matches_single() {
        let data = sample_data(128);
        let input = EhlersAdaptiveCyberCycleInput::from_slice(
            &data,
            EhlersAdaptiveCyberCycleParams { alpha: Some(0.07) },
        );
        let single = ehlers_adaptive_cyber_cycle(&input).unwrap();
        let mut cycle = vec![f64::NAN; data.len()];
        let mut trigger = vec![f64::NAN; data.len()];
        ehlers_adaptive_cyber_cycle_into_slice(&mut cycle, &mut trigger, &input, Kernel::Auto)
            .unwrap();
        assert_pair_eq(&cycle, &single.cycle);
        assert_pair_eq(&trigger, &single.trigger);
    }

    #[test]
    fn invalid_alpha_is_rejected() {
        let data = sample_data(64);
        let input = EhlersAdaptiveCyberCycleInput::from_slice(
            &data,
            EhlersAdaptiveCyberCycleParams { alpha: Some(1.5) },
        );
        let err = ehlers_adaptive_cyber_cycle(&input).unwrap_err();
        assert!(matches!(
            err,
            EhlersAdaptiveCyberCycleError::InvalidAlpha { .. }
        ));
    }
}
