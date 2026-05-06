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
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use std::arch::is_x86_feature_detected;
use std::mem::ManuallyDrop;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use std::sync::OnceLock;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "hl2";
const DEFAULT_ALPHA: f64 = 0.07;
const DEFAULT_CUTOFF: f64 = 8.0;
const MAX_ADAPTIVE_LOOKBACK: usize = 75;
const REQUIRED_VALID_SAMPLES: usize = MAX_ADAPTIVE_LOOKBACK + 1;
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
static ESAM_AUTO_KERNEL: OnceLock<Kernel> = OnceLock::new();

#[derive(Debug, Clone)]
pub enum EhlersSmoothedAdaptiveMomentumData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EhlersSmoothedAdaptiveMomentumOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct EhlersSmoothedAdaptiveMomentumParams {
    pub alpha: Option<f64>,
    pub cutoff: Option<f64>,
}

impl Default for EhlersSmoothedAdaptiveMomentumParams {
    fn default() -> Self {
        Self {
            alpha: Some(DEFAULT_ALPHA),
            cutoff: Some(DEFAULT_CUTOFF),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EhlersSmoothedAdaptiveMomentumInput<'a> {
    pub data: EhlersSmoothedAdaptiveMomentumData<'a>,
    pub params: EhlersSmoothedAdaptiveMomentumParams,
}

impl<'a> EhlersSmoothedAdaptiveMomentumInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: EhlersSmoothedAdaptiveMomentumParams,
    ) -> Self {
        Self {
            data: EhlersSmoothedAdaptiveMomentumData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a [f64], params: EhlersSmoothedAdaptiveMomentumParams) -> Self {
        Self {
            data: EhlersSmoothedAdaptiveMomentumData::Slice(data),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            EhlersSmoothedAdaptiveMomentumParams::default(),
        )
    }

    #[inline]
    pub fn get_alpha(&self) -> f64 {
        self.params.alpha.unwrap_or(DEFAULT_ALPHA)
    }

    #[inline]
    pub fn get_cutoff(&self) -> f64 {
        self.params.cutoff.unwrap_or(DEFAULT_CUTOFF)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EhlersSmoothedAdaptiveMomentumBuilder {
    source: Option<&'static str>,
    alpha: Option<f64>,
    cutoff: Option<f64>,
    kernel: Kernel,
}

impl Default for EhlersSmoothedAdaptiveMomentumBuilder {
    fn default() -> Self {
        Self {
            source: None,
            alpha: None,
            cutoff: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersSmoothedAdaptiveMomentumBuilder {
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
    pub fn cutoff(mut self, value: f64) -> Self {
        self.cutoff = Some(value);
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
    ) -> Result<EhlersSmoothedAdaptiveMomentumOutput, EhlersSmoothedAdaptiveMomentumError> {
        let input = EhlersSmoothedAdaptiveMomentumInput::from_candles(
            candles,
            self.source.unwrap_or(DEFAULT_SOURCE),
            EhlersSmoothedAdaptiveMomentumParams {
                alpha: self.alpha,
                cutoff: self.cutoff,
            },
        );
        ehlers_smoothed_adaptive_momentum_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EhlersSmoothedAdaptiveMomentumOutput, EhlersSmoothedAdaptiveMomentumError> {
        let input = EhlersSmoothedAdaptiveMomentumInput::from_slice(
            data,
            EhlersSmoothedAdaptiveMomentumParams {
                alpha: self.alpha,
                cutoff: self.cutoff,
            },
        );
        ehlers_smoothed_adaptive_momentum_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<EhlersSmoothedAdaptiveMomentumStream, EhlersSmoothedAdaptiveMomentumError> {
        EhlersSmoothedAdaptiveMomentumStream::try_new(EhlersSmoothedAdaptiveMomentumParams {
            alpha: self.alpha,
            cutoff: self.cutoff,
        })
    }
}

#[derive(Debug, Error)]
pub enum EhlersSmoothedAdaptiveMomentumError {
    #[error("ehlers_smoothed_adaptive_momentum: Input data slice is empty.")]
    EmptyInputData,
    #[error("ehlers_smoothed_adaptive_momentum: All values are NaN.")]
    AllValuesNaN,
    #[error("ehlers_smoothed_adaptive_momentum: Invalid alpha: {alpha}")]
    InvalidAlpha { alpha: f64 },
    #[error("ehlers_smoothed_adaptive_momentum: Invalid cutoff: {cutoff}")]
    InvalidCutoff { cutoff: f64 },
    #[error(
        "ehlers_smoothed_adaptive_momentum: Not enough valid data: needed={needed}, valid={valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "ehlers_smoothed_adaptive_momentum: Output length mismatch: expected={expected}, got={got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "ehlers_smoothed_adaptive_momentum: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("ehlers_smoothed_adaptive_momentum: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParams {
    coef_c: f64,
    coef_prev1: f64,
    coef_prev2: f64,
    coef1: f64,
    coef2: f64,
    coef3: f64,
    coef4: f64,
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
    input: &'a EhlersSmoothedAdaptiveMomentumInput<'a>,
) -> Result<&'a [f64], EhlersSmoothedAdaptiveMomentumError> {
    let data = match &input.data {
        EhlersSmoothedAdaptiveMomentumData::Candles { candles, source } => {
            source_type(candles, source)
        }
        EhlersSmoothedAdaptiveMomentumData::Slice(values) => *values,
    };
    if data.is_empty() {
        return Err(EhlersSmoothedAdaptiveMomentumError::EmptyInputData);
    }
    Ok(data)
}

#[inline(always)]
fn first_valid(data: &[f64]) -> Option<usize> {
    data.iter().position(|v| v.is_finite())
}

#[inline(always)]
fn resolve_params(
    params: &EhlersSmoothedAdaptiveMomentumParams,
) -> Result<ResolvedParams, EhlersSmoothedAdaptiveMomentumError> {
    let alpha = params.alpha.unwrap_or(DEFAULT_ALPHA);
    let cutoff = params.cutoff.unwrap_or(DEFAULT_CUTOFF);
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(EhlersSmoothedAdaptiveMomentumError::InvalidAlpha { alpha });
    }
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(EhlersSmoothedAdaptiveMomentumError::InvalidCutoff { cutoff });
    }
    let coef_c = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha);
    let one_minus_alpha = 1.0 - alpha;
    let a1 = (-std::f64::consts::PI / cutoff).exp();
    let b1 = 2.0 * a1 * (1.738 * std::f64::consts::PI / cutoff).cos();
    let c1 = a1 * a1;
    let coef2 = b1 + c1;
    let coef3 = -(c1 + b1 * c1);
    let coef4 = c1 * c1;
    let coef1 = 1.0 - coef2 - coef3 - coef4;
    Ok(ResolvedParams {
        coef_c,
        coef_prev1: 2.0 * one_minus_alpha,
        coef_prev2: one_minus_alpha * one_minus_alpha,
        coef1,
        coef2,
        coef3,
        coef4,
    })
}

#[inline(always)]
fn validate_input<'a>(
    input: &'a EhlersSmoothedAdaptiveMomentumInput<'a>,
    kernel: Kernel,
) -> Result<(&'a [f64], ResolvedParams, usize, Kernel), EhlersSmoothedAdaptiveMomentumError> {
    let data = extract_slice(input)?;
    let resolved = resolve_params(&input.params)?;
    let first = first_valid(data).ok_or(EhlersSmoothedAdaptiveMomentumError::AllValuesNaN)?;
    let valid = data.len().saturating_sub(first);
    if valid < REQUIRED_VALID_SAMPLES {
        return Err(EhlersSmoothedAdaptiveMomentumError::NotEnoughValidData {
            needed: REQUIRED_VALID_SAMPLES,
            valid,
        });
    }
    Ok((data, resolved, first, kernel.to_non_batch()))
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
struct EsamCore {
    params: ResolvedParams,
    src_ring: [f64; MAX_ADAPTIVE_LOOKBACK + 1],
    src_idx: usize,
    smooth_ring: [f64; 3],
    smooth_idx: usize,
    c_ring: [f64; 7],
    c_idx: usize,
    dp_ring: [f64; 5],
    dp_idx: usize,
    prev_ip: f64,
    prev_p: f64,
    prev_q1: f64,
    prev_i1: f64,
    f3_hist: [f64; 3],
    valid_count: usize,
}

impl EsamCore {
    #[inline(always)]
    fn new(params: ResolvedParams) -> Self {
        Self {
            params,
            src_ring: [f64::NAN; MAX_ADAPTIVE_LOOKBACK + 1],
            src_idx: 0,
            smooth_ring: [f64::NAN; 3],
            smooth_idx: 0,
            c_ring: [f64::NAN; 7],
            c_idx: 0,
            dp_ring: [f64::NAN; 5],
            dp_idx: 0,
            prev_ip: f64::NAN,
            prev_p: f64::NAN,
            prev_q1: f64::NAN,
            prev_i1: f64::NAN,
            f3_hist: [f64::NAN; 3],
            valid_count: 0,
        }
    }

    #[inline(always)]
    fn update(&mut self, source: f64) -> f64 {
        if !source.is_finite() {
            return f64::NAN;
        }

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

        let smooth1 = nz(ring_get(&self.smooth_ring, self.smooth_idx, 1));
        let smooth2 = nz(ring_get(&self.smooth_ring, self.smooth_idx, 2));
        let c_prev1 = nz(ring_get(&self.c_ring, self.c_idx, 1));
        let c_prev2 = nz(ring_get(&self.c_ring, self.c_idx, 2));
        let c_main = if smooth.is_finite() {
            self.params.coef_c * (smooth - 2.0 * smooth1 + smooth2)
                + self.params.coef_prev1 * c_prev1
                - self.params.coef_prev2 * c_prev2
        } else {
            f64::NAN
        };
        let c_fallback = if src0.is_finite() && src1.is_finite() && src2.is_finite() {
            (src0 - 2.0 * src1 + src2) / 4.0
        } else {
            f64::NAN
        };
        let c = if c_main.is_finite() {
            c_main
        } else {
            c_fallback
        };
        self.c_ring[self.c_idx] = c;

        let q1 = if c.is_finite() {
            let factor = 0.5 + 0.08 * nz(self.prev_ip);
            (0.0962 * c + 0.5769 * nz(ring_get(&self.c_ring, self.c_idx, 2))
                - 0.5769 * nz(ring_get(&self.c_ring, self.c_idx, 4))
                - 0.0962 * nz(ring_get(&self.c_ring, self.c_idx, 6)))
                * factor
        } else {
            f64::NAN
        };
        let i1 = nz(ring_get(&self.c_ring, self.c_idx, 3));

        let dp_raw =
            if q1.is_finite() && self.prev_q1.is_finite() && q1 != 0.0 && self.prev_q1 != 0.0 {
                let prev_i1 = nz(self.prev_i1);
                let prev_q1 = nz(self.prev_q1);
                let numer = (i1 / q1) - (prev_i1 / prev_q1);
                let denom = 1.0 + i1 * prev_i1 / (q1 * prev_q1);
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

        let pr = if p.is_finite() {
            (p - 1.0).abs().round()
        } else {
            f64::NAN
        };
        let v1 = if pr.is_finite() {
            let lookback = pr as usize;
            if (1..=MAX_ADAPTIVE_LOOKBACK).contains(&lookback) {
                let past = ring_get(&self.src_ring, self.src_idx, lookback);
                if past.is_finite() {
                    source - past
                } else {
                    f64::NAN
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        let raw_f3 = if v1.is_finite() {
            self.params.coef1 * v1
                + self.params.coef2 * nz(self.f3_hist[0])
                + self.params.coef3 * nz(self.f3_hist[1])
                + self.params.coef4 * nz(self.f3_hist[2])
        } else {
            f64::NAN
        };
        let f3 = if raw_f3.is_finite() { raw_f3 } else { v1 };

        self.prev_q1 = q1;
        self.prev_i1 = i1;
        self.prev_ip = ip;
        self.prev_p = p;
        self.f3_hist[2] = self.f3_hist[1];
        self.f3_hist[1] = self.f3_hist[0];
        self.f3_hist[0] = f3;

        self.valid_count += 1;
        self.src_idx = (self.src_idx + 1) % self.src_ring.len();
        self.smooth_idx = (self.smooth_idx + 1) % self.smooth_ring.len();
        self.c_idx = (self.c_idx + 1) % self.c_ring.len();
        self.dp_idx = (self.dp_idx + 1) % self.dp_ring.len();

        if self.valid_count <= MAX_ADAPTIVE_LOOKBACK {
            f64::NAN
        } else {
            f3
        }
    }
}

#[inline(always)]
fn compute_esam_into(
    data: &[f64],
    params: ResolvedParams,
    out: &mut [f64],
) -> Result<(), EhlersSmoothedAdaptiveMomentumError> {
    if out.len() != data.len() {
        return Err(EhlersSmoothedAdaptiveMomentumError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }
    let mut core = EsamCore::new(params);
    for (dst, &value) in out.iter_mut().zip(data.iter()) {
        *dst = core.update(value);
    }
    Ok(())
}

#[inline(always)]
fn compute_esam_with_kernel(
    data: &[f64],
    params: ResolvedParams,
    out: &mut [f64],
    kernel: Kernel,
) -> Result<(), EhlersSmoothedAdaptiveMomentumError> {
    let chosen = match kernel {
        Kernel::Auto => detect_esam_kernel(),
        other if other.is_batch() => other.to_non_batch(),
        other => other,
    };

    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    unsafe {
        match chosen {
            Kernel::Avx2 => compute_esam_avx2(data, params, out),
            _ => compute_esam_into(data, params, out),
        }
    }

    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    {
        let _ = chosen;
        compute_esam_into(data, params, out)
    }
}

#[inline(always)]
fn detect_esam_kernel() -> Kernel {
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        return *ESAM_AUTO_KERNEL.get_or_init(|| {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                Kernel::Avx2
            } else {
                Kernel::Scalar
            }
        });
    }

    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    {
        Kernel::Scalar
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_esam_avx2(
    data: &[f64],
    params: ResolvedParams,
    out: &mut [f64],
) -> Result<(), EhlersSmoothedAdaptiveMomentumError> {
    compute_esam_into(data, params, out)
}

#[inline]
pub fn ehlers_smoothed_adaptive_momentum(
    input: &EhlersSmoothedAdaptiveMomentumInput,
) -> Result<EhlersSmoothedAdaptiveMomentumOutput, EhlersSmoothedAdaptiveMomentumError> {
    ehlers_smoothed_adaptive_momentum_with_kernel(input, Kernel::Auto)
}

#[inline]
pub fn ehlers_smoothed_adaptive_momentum_with_kernel(
    input: &EhlersSmoothedAdaptiveMomentumInput,
    kernel: Kernel,
) -> Result<EhlersSmoothedAdaptiveMomentumOutput, EhlersSmoothedAdaptiveMomentumError> {
    let (data, params, _first, kernel) = validate_input(input, kernel)?;
    let mut out = alloc_uninit_f64(data.len());
    compute_esam_with_kernel(data, params, &mut out, kernel)?;
    Ok(EhlersSmoothedAdaptiveMomentumOutput { values: out })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn ehlers_smoothed_adaptive_momentum_into(
    out: &mut [f64],
    input: &EhlersSmoothedAdaptiveMomentumInput,
    kernel: Kernel,
) -> Result<(), EhlersSmoothedAdaptiveMomentumError> {
    ehlers_smoothed_adaptive_momentum_into_slice(out, input, kernel)
}

#[inline]
pub fn ehlers_smoothed_adaptive_momentum_into_slice(
    out: &mut [f64],
    input: &EhlersSmoothedAdaptiveMomentumInput,
    kernel: Kernel,
) -> Result<(), EhlersSmoothedAdaptiveMomentumError> {
    let (data, params, _first, kernel) = validate_input(input, kernel)?;
    compute_esam_with_kernel(data, params, out, kernel)
}

#[derive(Clone, Debug)]
pub struct EhlersSmoothedAdaptiveMomentumStream {
    core: EsamCore,
}

impl EhlersSmoothedAdaptiveMomentumStream {
    pub fn try_new(
        params: EhlersSmoothedAdaptiveMomentumParams,
    ) -> Result<Self, EhlersSmoothedAdaptiveMomentumError> {
        Ok(Self {
            core: EsamCore::new(resolve_params(&params)?),
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> f64 {
        self.core.update(value)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EhlersSmoothedAdaptiveMomentumBatchRange {
    pub alpha: (f64, f64, f64),
    pub cutoff: (f64, f64, f64),
}

impl Default for EhlersSmoothedAdaptiveMomentumBatchRange {
    fn default() -> Self {
        Self {
            alpha: (DEFAULT_ALPHA, DEFAULT_ALPHA, 0.0),
            cutoff: (DEFAULT_CUTOFF, DEFAULT_CUTOFF, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EhlersSmoothedAdaptiveMomentumBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EhlersSmoothedAdaptiveMomentumParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct EhlersSmoothedAdaptiveMomentumBatchBuilder {
    source: Option<&'static str>,
    range: EhlersSmoothedAdaptiveMomentumBatchRange,
    kernel: Kernel,
}

impl Default for EhlersSmoothedAdaptiveMomentumBatchBuilder {
    fn default() -> Self {
        Self {
            source: None,
            range: EhlersSmoothedAdaptiveMomentumBatchRange::default(),
            kernel: Kernel::Auto,
        }
    }
}

impl EhlersSmoothedAdaptiveMomentumBatchBuilder {
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
    pub fn cutoff_range(mut self, value: (f64, f64, f64)) -> Self {
        self.range.cutoff = value;
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
    ) -> Result<EhlersSmoothedAdaptiveMomentumBatchOutput, EhlersSmoothedAdaptiveMomentumError>
    {
        ehlers_smoothed_adaptive_momentum_batch_with_kernel(
            source_type(candles, self.source.unwrap_or(DEFAULT_SOURCE)),
            &self.range,
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<EhlersSmoothedAdaptiveMomentumBatchOutput, EhlersSmoothedAdaptiveMomentumError>
    {
        ehlers_smoothed_adaptive_momentum_batch_with_kernel(data, &self.range, self.kernel)
    }
}

#[inline(always)]
fn expand_float_range(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, EhlersSmoothedAdaptiveMomentumError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(EhlersSmoothedAdaptiveMomentumError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 {
        if (start - end).abs() > 1e-12 {
            return Err(EhlersSmoothedAdaptiveMomentumError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if start > end || step < 0.0 {
        return Err(EhlersSmoothedAdaptiveMomentumError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut current = start;
    let mut out = Vec::new();
    let limit = 1_000_000usize;
    while current <= end + 1e-12 {
        out.push(current);
        if out.len() > limit {
            return Err(EhlersSmoothedAdaptiveMomentumError::InvalidRange {
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
    sweep: &EhlersSmoothedAdaptiveMomentumBatchRange,
) -> Result<Vec<EhlersSmoothedAdaptiveMomentumParams>, EhlersSmoothedAdaptiveMomentumError> {
    let alphas = expand_float_range(sweep.alpha.0, sweep.alpha.1, sweep.alpha.2)?;
    let cutoffs = expand_float_range(sweep.cutoff.0, sweep.cutoff.1, sweep.cutoff.2)?;
    let mut combos = Vec::with_capacity(alphas.len().saturating_mul(cutoffs.len()));
    for &alpha in &alphas {
        for &cutoff in &cutoffs {
            combos.push(EhlersSmoothedAdaptiveMomentumParams {
                alpha: Some(alpha),
                cutoff: Some(cutoff),
            });
        }
    }
    Ok(combos)
}

#[inline(always)]
fn validate_raw_slice(data: &[f64]) -> Result<usize, EhlersSmoothedAdaptiveMomentumError> {
    if data.is_empty() {
        return Err(EhlersSmoothedAdaptiveMomentumError::EmptyInputData);
    }
    first_valid(data).ok_or(EhlersSmoothedAdaptiveMomentumError::AllValuesNaN)
}

pub fn ehlers_smoothed_adaptive_momentum_batch_with_kernel(
    data: &[f64],
    sweep: &EhlersSmoothedAdaptiveMomentumBatchRange,
    kernel: Kernel,
) -> Result<EhlersSmoothedAdaptiveMomentumBatchOutput, EhlersSmoothedAdaptiveMomentumError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(EhlersSmoothedAdaptiveMomentumError::InvalidKernelForBatch(
                kernel,
            ));
        }
    };
    ehlers_smoothed_adaptive_momentum_batch_par_slice(data, sweep, batch_kernel.to_non_batch())
}

#[inline(always)]
pub fn ehlers_smoothed_adaptive_momentum_batch_slice(
    data: &[f64],
    sweep: &EhlersSmoothedAdaptiveMomentumBatchRange,
    kernel: Kernel,
) -> Result<EhlersSmoothedAdaptiveMomentumBatchOutput, EhlersSmoothedAdaptiveMomentumError> {
    ehlers_smoothed_adaptive_momentum_batch_inner(data, sweep, kernel, false)
}

#[inline(always)]
pub fn ehlers_smoothed_adaptive_momentum_batch_par_slice(
    data: &[f64],
    sweep: &EhlersSmoothedAdaptiveMomentumBatchRange,
    kernel: Kernel,
) -> Result<EhlersSmoothedAdaptiveMomentumBatchOutput, EhlersSmoothedAdaptiveMomentumError> {
    ehlers_smoothed_adaptive_momentum_batch_inner(data, sweep, kernel, true)
}

fn ehlers_smoothed_adaptive_momentum_batch_inner(
    data: &[f64],
    sweep: &EhlersSmoothedAdaptiveMomentumBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<EhlersSmoothedAdaptiveMomentumBatchOutput, EhlersSmoothedAdaptiveMomentumError> {
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        EhlersSmoothedAdaptiveMomentumError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        }
    })?;
    let mut buf = make_uninit_matrix(rows, cols);
    let mut guard = ManuallyDrop::new(buf);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };
    ehlers_smoothed_adaptive_momentum_batch_inner_into(data, sweep, kernel, parallel, out)?;
    let values =
        unsafe { Vec::from_raw_parts(guard.as_mut_ptr() as *mut f64, expected, guard.capacity()) };
    Ok(EhlersSmoothedAdaptiveMomentumBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

pub fn ehlers_smoothed_adaptive_momentum_batch_into_slice(
    out: &mut [f64],
    data: &[f64],
    sweep: &EhlersSmoothedAdaptiveMomentumBatchRange,
    kernel: Kernel,
) -> Result<(), EhlersSmoothedAdaptiveMomentumError> {
    ehlers_smoothed_adaptive_momentum_batch_inner_into(data, sweep, kernel, false, out)?;
    Ok(())
}

fn ehlers_smoothed_adaptive_momentum_batch_inner_into(
    data: &[f64],
    sweep: &EhlersSmoothedAdaptiveMomentumBatchRange,
    _kernel: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<EhlersSmoothedAdaptiveMomentumParams>, EhlersSmoothedAdaptiveMomentumError> {
    let combos = expand_grid(sweep)?;
    let first = validate_raw_slice(data)?;
    let rows = combos.len();
    let cols = data.len();
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        EhlersSmoothedAdaptiveMomentumError::InvalidRange {
            start: rows.to_string(),
            end: cols.to_string(),
            step: "rows*cols".to_string(),
        }
    })?;
    if out.len() != expected {
        return Err(EhlersSmoothedAdaptiveMomentumError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }
    let valid = cols.saturating_sub(first);
    if valid < REQUIRED_VALID_SAMPLES {
        return Err(EhlersSmoothedAdaptiveMomentumError::NotEnoughValidData {
            needed: REQUIRED_VALID_SAMPLES,
            valid,
        });
    }

    let compute_row =
        |row: usize, dst: &mut [f64]| -> Result<(), EhlersSmoothedAdaptiveMomentumError> {
            let params = resolve_params(&combos[row])?;
            compute_esam_into(data, params, dst)
        };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out.par_chunks_mut(cols)
                .enumerate()
                .try_for_each(|(row, dst)| compute_row(row, dst))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, dst) in out.chunks_mut(cols).enumerate() {
                compute_row(row, dst)?;
            }
        }
    } else {
        for (row, dst) in out.chunks_mut(cols).enumerate() {
            compute_row(row, dst)?;
        }
    }
    Ok(combos)
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_smoothed_adaptive_momentum")]
#[pyo3(signature = (data, alpha=0.07, cutoff=8.0, kernel=None))]
pub fn ehlers_smoothed_adaptive_momentum_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    cutoff: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = data.as_slice()?;
    let input = EhlersSmoothedAdaptiveMomentumInput::from_slice(
        slice,
        EhlersSmoothedAdaptiveMomentumParams {
            alpha: Some(alpha),
            cutoff: Some(cutoff),
        },
    );
    let kernel = validate_kernel(kernel, false)?;
    let out = py
        .allow_threads(|| ehlers_smoothed_adaptive_momentum_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.values.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "EhlersSmoothedAdaptiveMomentumStream")]
pub struct EhlersSmoothedAdaptiveMomentumStreamPy {
    stream: EhlersSmoothedAdaptiveMomentumStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EhlersSmoothedAdaptiveMomentumStreamPy {
    #[new]
    #[pyo3(signature = (alpha=0.07, cutoff=8.0))]
    fn new(alpha: f64, cutoff: f64) -> PyResult<Self> {
        let stream =
            EhlersSmoothedAdaptiveMomentumStream::try_new(EhlersSmoothedAdaptiveMomentumParams {
                alpha: Some(alpha),
                cutoff: Some(cutoff),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> f64 {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ehlers_smoothed_adaptive_momentum_batch")]
#[pyo3(signature = (data, alpha_range=(0.07,0.07,0.0), cutoff_range=(8.0,8.0,0.0), kernel=None))]
pub fn ehlers_smoothed_adaptive_momentum_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    alpha_range: (f64, f64, f64),
    cutoff_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let sweep = EhlersSmoothedAdaptiveMomentumBatchRange {
        alpha: alpha_range,
        cutoff: cutoff_range,
    };
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let out = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_slice = unsafe { out.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        ehlers_smoothed_adaptive_momentum_batch_inner_into(
            slice,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_slice,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out.reshape((rows, cols))?)?;
    dict.set_item(
        "alphas",
        combos
            .iter()
            .map(|combo| combo.alpha.unwrap_or(DEFAULT_ALPHA))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "cutoffs",
        combos
            .iter()
            .map(|combo| combo.cutoff.unwrap_or(DEFAULT_CUTOFF))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_ehlers_smoothed_adaptive_momentum_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ehlers_smoothed_adaptive_momentum_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        ehlers_smoothed_adaptive_momentum_batch_py,
        m
    )?)?;
    m.add_class::<EhlersSmoothedAdaptiveMomentumStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_smoothed_adaptive_momentum_js")]
pub fn ehlers_smoothed_adaptive_momentum_js(
    data: &[f64],
    alpha: f64,
    cutoff: f64,
) -> Result<Vec<f64>, JsValue> {
    let input = EhlersSmoothedAdaptiveMomentumInput::from_slice(
        data,
        EhlersSmoothedAdaptiveMomentumParams {
            alpha: Some(alpha),
            cutoff: Some(cutoff),
        },
    );
    let mut out = vec![f64::NAN; data.len()];
    ehlers_smoothed_adaptive_momentum_into_slice(&mut out, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersSmoothedAdaptiveMomentumBatchConfig {
    pub alpha_range: Vec<f64>,
    pub cutoff_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct EhlersSmoothedAdaptiveMomentumBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EhlersSmoothedAdaptiveMomentumParams>,
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
#[wasm_bindgen(js_name = "ehlers_smoothed_adaptive_momentum_batch_js")]
pub fn ehlers_smoothed_adaptive_momentum_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: EhlersSmoothedAdaptiveMomentumBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    let sweep = EhlersSmoothedAdaptiveMomentumBatchRange {
        alpha: js_vec3_to_f64("alpha_range", &config.alpha_range)?,
        cutoff: js_vec3_to_f64("cutoff_range", &config.cutoff_range)?,
    };
    let out = ehlers_smoothed_adaptive_momentum_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&EhlersSmoothedAdaptiveMomentumBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_smoothed_adaptive_momentum_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_smoothed_adaptive_momentum_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_smoothed_adaptive_momentum_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    alpha: f64,
    cutoff: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_smoothed_adaptive_momentum_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        let input = EhlersSmoothedAdaptiveMomentumInput::from_slice(
            data,
            EhlersSmoothedAdaptiveMomentumParams {
                alpha: Some(alpha),
                cutoff: Some(cutoff),
            },
        );
        ehlers_smoothed_adaptive_momentum_into_slice(out, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "ehlers_smoothed_adaptive_momentum_batch_into")]
pub fn ehlers_smoothed_adaptive_momentum_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    alpha_start: f64,
    alpha_end: f64,
    alpha_step: f64,
    cutoff_start: f64,
    cutoff_end: f64,
    cutoff_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to ehlers_smoothed_adaptive_momentum_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = EhlersSmoothedAdaptiveMomentumBatchRange {
            alpha: (alpha_start, alpha_end, alpha_step),
            cutoff: (cutoff_start, cutoff_end, cutoff_step),
        };
        let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let total = rows.checked_mul(len).ok_or_else(|| {
            JsValue::from_str("rows*cols overflow in ehlers_smoothed_adaptive_momentum_batch_into")
        })?;
        let out = std::slice::from_raw_parts_mut(out_ptr, total);
        ehlers_smoothed_adaptive_momentum_batch_into_slice(out, data, &sweep, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_smoothed_adaptive_momentum_output_into_js(
    data: &[f64],
    alpha: f64,
    cutoff: f64,
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let values = ehlers_smoothed_adaptive_momentum_js(data, alpha, cutoff)?;
    crate::write_wasm_f64_output(
        "ehlers_smoothed_adaptive_momentum_output_into_js",
        &values,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn ehlers_smoothed_adaptive_momentum_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = ehlers_smoothed_adaptive_momentum_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "ehlers_smoothed_adaptive_momentum_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_series(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let x = i as f64;
                100.0 + (x * 0.13).sin() * 2.1 + (x * 0.047).cos() * 0.8 + x * 0.02
            })
            .collect()
    }

    fn manual_esam(data: &[f64], alpha: f64, cutoff: f64) -> Vec<f64> {
        let params = resolve_params(&EhlersSmoothedAdaptiveMomentumParams {
            alpha: Some(alpha),
            cutoff: Some(cutoff),
        })
        .unwrap();
        let mut out = vec![f64::NAN; data.len()];
        let mut core = EsamCore::new(params);
        for (i, &value) in data.iter().enumerate() {
            out[i] = core.update(value);
        }
        out
    }

    fn assert_close(lhs: &[f64], rhs: &[f64], tol: f64) {
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
    fn manual_reference_matches_api() {
        let data = sample_series(192);
        let input = EhlersSmoothedAdaptiveMomentumInput::from_slice(
            &data,
            EhlersSmoothedAdaptiveMomentumParams::default(),
        );
        let out = ehlers_smoothed_adaptive_momentum(&input).unwrap();
        let expected = manual_esam(&data, DEFAULT_ALPHA, DEFAULT_CUTOFF);
        assert_close(&out.values, &expected, 1e-12);
    }

    #[test]
    fn stream_matches_batch() {
        let data = sample_series(192);
        let input = EhlersSmoothedAdaptiveMomentumInput::from_slice(
            &data,
            EhlersSmoothedAdaptiveMomentumParams::default(),
        );
        let batch = ehlers_smoothed_adaptive_momentum(&input).unwrap();
        let mut stream = EhlersSmoothedAdaptiveMomentumStream::try_new(
            EhlersSmoothedAdaptiveMomentumParams::default(),
        )
        .unwrap();
        let mut streamed = vec![f64::NAN; data.len()];
        for (i, &value) in data.iter().enumerate() {
            streamed[i] = stream.update(value);
        }
        assert_close(&streamed, &batch.values, 1e-12);
    }

    #[test]
    fn batch_first_row_matches_single() {
        let data = sample_series(192);
        let single =
            ehlers_smoothed_adaptive_momentum(&EhlersSmoothedAdaptiveMomentumInput::from_slice(
                &data,
                EhlersSmoothedAdaptiveMomentumParams::default(),
            ))
            .unwrap();
        let batch = ehlers_smoothed_adaptive_momentum_batch_with_kernel(
            &data,
            &EhlersSmoothedAdaptiveMomentumBatchRange {
                alpha: (0.07, 0.09, 0.02),
                cutoff: (8.0, 8.0, 0.0),
            },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(batch.rows, 2);
        assert_eq!(batch.cols, data.len());
        assert_close(&batch.values[..data.len()], &single.values, 1e-12);
    }

    #[test]
    fn into_slice_matches_single() {
        let data = sample_series(192);
        let input = EhlersSmoothedAdaptiveMomentumInput::from_slice(
            &data,
            EhlersSmoothedAdaptiveMomentumParams::default(),
        );
        let single = ehlers_smoothed_adaptive_momentum(&input).unwrap();
        let mut out = vec![0.0; data.len()];
        ehlers_smoothed_adaptive_momentum_into_slice(&mut out, &input, Kernel::Auto).unwrap();
        assert_close(&out, &single.values, 1e-12);
    }

    #[test]
    fn invalid_cutoff_is_rejected() {
        let data = sample_series(128);
        let input = EhlersSmoothedAdaptiveMomentumInput::from_slice(
            &data,
            EhlersSmoothedAdaptiveMomentumParams {
                alpha: Some(DEFAULT_ALPHA),
                cutoff: Some(0.0),
            },
        );
        match ehlers_smoothed_adaptive_momentum(&input) {
            Err(EhlersSmoothedAdaptiveMomentumError::InvalidCutoff { cutoff }) => {
                assert_eq!(cutoff, 0.0)
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }
}
