#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
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
use crate::utilities::helpers::{alloc_uninit_f64, detect_best_batch_kernel};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

const DEFAULT_SOURCE: &str = "close";
const DEFAULT_LENGTH_1: usize = 48;
const DEFAULT_LENGTH_2: usize = 21;
const DEFAULT_LENGTH_3: usize = 9;
const DEFAULT_LENGTH_4: usize = 6;
const DEFAULT_TRIGGER_LENGTH: usize = 2;
const PI: f64 = 3.14;

#[derive(Debug, Clone)]
pub enum MesaStochasticMultiLengthData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice {
        source: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MesaStochasticMultiLengthOutput {
    pub mesa_1: Vec<f64>,
    pub mesa_2: Vec<f64>,
    pub mesa_3: Vec<f64>,
    pub mesa_4: Vec<f64>,
    pub trigger_1: Vec<f64>,
    pub trigger_2: Vec<f64>,
    pub trigger_3: Vec<f64>,
    pub trigger_4: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct MesaStochasticMultiLengthParams {
    pub length_1: Option<usize>,
    pub length_2: Option<usize>,
    pub length_3: Option<usize>,
    pub length_4: Option<usize>,
    pub trigger_length: Option<usize>,
}

impl Default for MesaStochasticMultiLengthParams {
    fn default() -> Self {
        Self {
            length_1: Some(DEFAULT_LENGTH_1),
            length_2: Some(DEFAULT_LENGTH_2),
            length_3: Some(DEFAULT_LENGTH_3),
            length_4: Some(DEFAULT_LENGTH_4),
            trigger_length: Some(DEFAULT_TRIGGER_LENGTH),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MesaStochasticMultiLengthInput<'a> {
    pub data: MesaStochasticMultiLengthData<'a>,
    pub params: MesaStochasticMultiLengthParams,
}

impl<'a> MesaStochasticMultiLengthInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: MesaStochasticMultiLengthParams,
    ) -> Self {
        Self {
            data: MesaStochasticMultiLengthData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(source: &'a [f64], params: MesaStochasticMultiLengthParams) -> Self {
        Self {
            data: MesaStochasticMultiLengthData::Slice { source },
            params,
        }
    }

    #[inline]
    pub fn from_slices(source: &'a [f64], params: MesaStochasticMultiLengthParams) -> Self {
        Self::from_slice(source, params)
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            DEFAULT_SOURCE,
            MesaStochasticMultiLengthParams::default(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct ValidatedParams {
    length_1: usize,
    length_2: usize,
    length_3: usize,
    length_4: usize,
    trigger_length: usize,
}

impl ValidatedParams {
    fn from_params(
        params: &MesaStochasticMultiLengthParams,
    ) -> Result<Self, MesaStochasticMultiLengthError> {
        let out = Self {
            length_1: params.length_1.unwrap_or(DEFAULT_LENGTH_1),
            length_2: params.length_2.unwrap_or(DEFAULT_LENGTH_2),
            length_3: params.length_3.unwrap_or(DEFAULT_LENGTH_3),
            length_4: params.length_4.unwrap_or(DEFAULT_LENGTH_4),
            trigger_length: params.trigger_length.unwrap_or(DEFAULT_TRIGGER_LENGTH),
        };
        for (name, value) in [
            ("length_1", out.length_1),
            ("length_2", out.length_2),
            ("length_3", out.length_3),
            ("length_4", out.length_4),
            ("trigger_length", out.trigger_length),
        ] {
            if value == 0 {
                return Err(MesaStochasticMultiLengthError::InvalidPeriod {
                    name: name.to_string(),
                    value,
                });
            }
        }
        Ok(out)
    }

    fn into_params(self) -> MesaStochasticMultiLengthParams {
        MesaStochasticMultiLengthParams {
            length_1: Some(self.length_1),
            length_2: Some(self.length_2),
            length_3: Some(self.length_3),
            length_4: Some(self.length_4),
            trigger_length: Some(self.trigger_length),
        }
    }
}

#[derive(Debug, Error)]
pub enum MesaStochasticMultiLengthError {
    #[error("mesa_stochastic_multi_length: Input data slice is empty.")]
    EmptyInputData,
    #[error("mesa_stochastic_multi_length: All values are NaN.")]
    AllValuesNaN,
    #[error("mesa_stochastic_multi_length: Invalid period `{name}`: {value}")]
    InvalidPeriod { name: String, value: usize },
    #[error(
        "mesa_stochastic_multi_length: Output length mismatch: expected={expected}, got={got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("mesa_stochastic_multi_length: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("mesa_stochastic_multi_length: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[inline(always)]
fn extract_source<'a>(
    input: &'a MesaStochasticMultiLengthInput<'a>,
) -> Result<&'a [f64], MesaStochasticMultiLengthError> {
    let source = match &input.data {
        MesaStochasticMultiLengthData::Candles { candles, source } => {
            mesa_stochastic_multi_length_source(candles, source)
        }
        MesaStochasticMultiLengthData::Slice { source } => *source,
    };
    if source.is_empty() {
        return Err(MesaStochasticMultiLengthError::EmptyInputData);
    }
    Ok(source)
}

#[inline(always)]
fn mesa_stochastic_multi_length_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
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
    }
}

#[inline(always)]
fn first_valid(source: &[f64]) -> Option<usize> {
    source.iter().position(|value| value.is_finite())
}

#[derive(Debug, Clone, Copy)]
pub struct MesaStochasticMultiLengthBuilder {
    source: Option<&'static str>,
    length_1: Option<usize>,
    length_2: Option<usize>,
    length_3: Option<usize>,
    length_4: Option<usize>,
    trigger_length: Option<usize>,
    kernel: Kernel,
}

impl Default for MesaStochasticMultiLengthBuilder {
    fn default() -> Self {
        Self {
            source: None,
            length_1: None,
            length_2: None,
            length_3: None,
            length_4: None,
            trigger_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MesaStochasticMultiLengthBuilder {
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
    pub fn length_1(mut self, value: usize) -> Self {
        self.length_1 = Some(value);
        self
    }

    #[inline(always)]
    pub fn length_2(mut self, value: usize) -> Self {
        self.length_2 = Some(value);
        self
    }

    #[inline(always)]
    pub fn length_3(mut self, value: usize) -> Self {
        self.length_3 = Some(value);
        self
    }

    #[inline(always)]
    pub fn length_4(mut self, value: usize) -> Self {
        self.length_4 = Some(value);
        self
    }

    #[inline(always)]
    pub fn trigger_length(mut self, value: usize) -> Self {
        self.trigger_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
        self
    }

    #[inline(always)]
    fn params(self) -> MesaStochasticMultiLengthParams {
        MesaStochasticMultiLengthParams {
            length_1: self.length_1,
            length_2: self.length_2,
            length_3: self.length_3,
            length_4: self.length_4,
            trigger_length: self.trigger_length,
        }
    }

    #[inline(always)]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<MesaStochasticMultiLengthOutput, MesaStochasticMultiLengthError> {
        let input = MesaStochasticMultiLengthInput::from_candles(
            candles,
            self.source.unwrap_or(DEFAULT_SOURCE),
            self.params(),
        );
        mesa_stochastic_multi_length_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        source: &[f64],
    ) -> Result<MesaStochasticMultiLengthOutput, MesaStochasticMultiLengthError> {
        let input = MesaStochasticMultiLengthInput::from_slice(source, self.params());
        mesa_stochastic_multi_length_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<MesaStochasticMultiLengthStream, MesaStochasticMultiLengthError> {
        MesaStochasticMultiLengthStream::try_new(self.params())
    }
}

#[inline(always)]
fn nz(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[derive(Clone, Debug)]
struct RollingSmaState {
    length: usize,
    window: Vec<f64>,
    index: usize,
    count: usize,
    finite_sum: f64,
    finite_count: usize,
}

impl RollingSmaState {
    fn new(length: usize) -> Self {
        Self {
            length,
            window: vec![0.0; length],
            index: 0,
            count: 0,
            finite_sum: 0.0,
            finite_count: 0,
        }
    }

    fn update(&mut self, value: f64) -> f64 {
        let old = if self.count == self.length {
            Some(self.window[self.index])
        } else {
            None
        };

        self.window[self.index] = value;
        if self.count != self.length {
            self.count += 1;
        }
        if value.is_finite() {
            self.finite_sum += value;
            self.finite_count += 1;
        }
        if let Some(old) = old {
            if old.is_finite() {
                self.finite_sum -= old;
                self.finite_count -= 1;
            }
        }
        self.index += 1;
        if self.index == self.length {
            self.index = 0;
        }

        if self.count == self.length && self.finite_count == self.length {
            self.finite_sum / self.length as f64
        } else {
            f64::NAN
        }
    }
}

#[derive(Clone, Debug)]
struct MesaLineState {
    length: usize,
    filt_window: Vec<f64>,
    index: usize,
    count: usize,
    prev_1: f64,
    prev_2: f64,
}

impl MesaLineState {
    fn new(length: usize) -> Self {
        Self {
            length,
            filt_window: vec![0.0; length],
            index: 0,
            count: 0,
            prev_1: f64::NAN,
            prev_2: f64::NAN,
        }
    }

    fn update(&mut self, filt: f64, c1: f64, c2: f64, c3: f64) -> f64 {
        let filt_nz = nz(filt);
        self.filt_window[self.index] = filt_nz;
        self.index += 1;
        if self.index == self.length {
            self.index = 0;
        }
        if self.count < self.length {
            self.count += 1;
        }

        let out = if filt.is_finite() {
            let mut highest = filt;
            let mut lowest = filt;
            for &value in &self.filt_window[..self.count] {
                if value > highest {
                    highest = value;
                }
                if value < lowest {
                    lowest = value;
                }
            }
            if self.count < self.length {
                if 0.0 > highest {
                    highest = 0.0;
                }
                if 0.0 < lowest {
                    lowest = 0.0;
                }
            }
            let denom = highest - lowest;
            if denom != 0.0 && denom.is_finite() {
                let stoc = (filt - lowest) / denom;
                if stoc.is_finite() {
                    c1.mul_add(stoc, c2.mul_add(nz(self.prev_1), c3 * nz(self.prev_2)))
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };

        self.prev_2 = self.prev_1;
        self.prev_1 = out;
        out
    }
}

#[derive(Clone, Debug)]
struct SharedFilterState {
    c1: f64,
    c2: f64,
    c3: f64,
    hp_coef: f64,
    hp_feedback_1: f64,
    hp_feedback_2: f64,
    prev_src_1: f64,
    prev_src_2: f64,
    prev_hp_1: f64,
    prev_hp_2: f64,
    prev_filt_1: f64,
    prev_filt_2: f64,
}

impl SharedFilterState {
    fn new() -> Self {
        let alpha1 = ((0.707 * 2.0 * PI / 48.0).cos() + (0.707 * 2.0 * PI / 48.0).sin() - 1.0)
            / (0.707 * 2.0 * PI / 48.0).cos();
        let one_minus_alpha = 1.0 - alpha1;
        let hp_coef = (1.0 - alpha1 * 0.5) * (1.0 - alpha1 * 0.5);
        let a1 = (-1.414 * PI / 10.0).exp();
        let b1 = 2.0 * a1 * (1.414 * PI / 10.0).cos();
        let c2 = b1;
        let c3 = -(a1 * a1);
        let c1 = 1.0 - c2 - c3;
        Self {
            c1,
            c2,
            c3,
            hp_coef,
            hp_feedback_1: 2.0 * one_minus_alpha,
            hp_feedback_2: -(one_minus_alpha * one_minus_alpha),
            prev_src_1: f64::NAN,
            prev_src_2: f64::NAN,
            prev_hp_1: f64::NAN,
            prev_hp_2: f64::NAN,
            prev_filt_1: f64::NAN,
            prev_filt_2: f64::NAN,
        }
    }

    fn update(&mut self, source: f64) -> f64 {
        let hp = if source.is_finite() {
            self.hp_coef.mul_add(
                source - 2.0 * nz(self.prev_src_1) + nz(self.prev_src_2),
                self.hp_feedback_1
                    .mul_add(nz(self.prev_hp_1), self.hp_feedback_2 * nz(self.prev_hp_2)),
            )
        } else {
            f64::NAN
        };
        let filt = if hp.is_finite() {
            self.c1.mul_add(
                hp,
                self.c2
                    .mul_add(nz(self.prev_filt_1), self.c3 * nz(self.prev_filt_2)),
            )
        } else {
            f64::NAN
        };

        self.prev_src_2 = self.prev_src_1;
        self.prev_src_1 = source;
        self.prev_hp_2 = self.prev_hp_1;
        self.prev_hp_1 = hp;
        self.prev_filt_2 = self.prev_filt_1;
        self.prev_filt_1 = filt;
        filt
    }
}

#[derive(Clone, Debug)]
pub struct MesaStochasticMultiLengthStream {
    filter_state: SharedFilterState,
    mesa_1_state: MesaLineState,
    mesa_2_state: MesaLineState,
    mesa_3_state: MesaLineState,
    mesa_4_state: MesaLineState,
    trigger_1_state: RollingSmaState,
    trigger_2_state: RollingSmaState,
    trigger_3_state: RollingSmaState,
    trigger_4_state: RollingSmaState,
}

impl MesaStochasticMultiLengthStream {
    pub fn try_new(
        params: MesaStochasticMultiLengthParams,
    ) -> Result<Self, MesaStochasticMultiLengthError> {
        let params = ValidatedParams::from_params(&params)?;
        Ok(Self {
            filter_state: SharedFilterState::new(),
            mesa_1_state: MesaLineState::new(params.length_1),
            mesa_2_state: MesaLineState::new(params.length_2),
            mesa_3_state: MesaLineState::new(params.length_3),
            mesa_4_state: MesaLineState::new(params.length_4),
            trigger_1_state: RollingSmaState::new(params.trigger_length),
            trigger_2_state: RollingSmaState::new(params.trigger_length),
            trigger_3_state: RollingSmaState::new(params.trigger_length),
            trigger_4_state: RollingSmaState::new(params.trigger_length),
        })
    }

    pub fn update(&mut self, source: f64) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
        let filt = self.filter_state.update(source);
        let c1 = self.filter_state.c1;
        let c2 = self.filter_state.c2;
        let c3 = self.filter_state.c3;

        let mesa_1 = self.mesa_1_state.update(filt, c1, c2, c3);
        let mesa_2 = self.mesa_2_state.update(filt, c1, c2, c3);
        let mesa_3 = self.mesa_3_state.update(filt, c1, c2, c3);
        let mesa_4 = self.mesa_4_state.update(filt, c1, c2, c3);

        let trigger_1 = self.trigger_1_state.update(mesa_1);
        let trigger_2 = self.trigger_2_state.update(mesa_2);
        let trigger_3 = self.trigger_3_state.update(mesa_3);
        let trigger_4 = self.trigger_4_state.update(mesa_4);

        (
            mesa_1, mesa_2, mesa_3, mesa_4, trigger_1, trigger_2, trigger_3, trigger_4,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_mesa_stochastic_multi_length_into(
    source: &[f64],
    params: ValidatedParams,
    mesa_1_out: &mut [f64],
    mesa_2_out: &mut [f64],
    mesa_3_out: &mut [f64],
    mesa_4_out: &mut [f64],
    trigger_1_out: &mut [f64],
    trigger_2_out: &mut [f64],
    trigger_3_out: &mut [f64],
    trigger_4_out: &mut [f64],
) -> Result<(), MesaStochasticMultiLengthError> {
    let n = source.len();
    if mesa_1_out.len() != n
        || mesa_2_out.len() != n
        || mesa_3_out.len() != n
        || mesa_4_out.len() != n
        || trigger_1_out.len() != n
        || trigger_2_out.len() != n
        || trigger_3_out.len() != n
        || trigger_4_out.len() != n
    {
        let got = [
            mesa_1_out.len(),
            mesa_2_out.len(),
            mesa_3_out.len(),
            mesa_4_out.len(),
            trigger_1_out.len(),
            trigger_2_out.len(),
            trigger_3_out.len(),
            trigger_4_out.len(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        return Err(MesaStochasticMultiLengthError::OutputLengthMismatch { expected: n, got });
    }

    let mut stream = MesaStochasticMultiLengthStream::try_new(params.into_params())?;
    for (i, value) in source.iter().copied().enumerate() {
        let (mesa_1, mesa_2, mesa_3, mesa_4, trigger_1, trigger_2, trigger_3, trigger_4) =
            stream.update(value);
        mesa_1_out[i] = mesa_1;
        mesa_2_out[i] = mesa_2;
        mesa_3_out[i] = mesa_3;
        mesa_4_out[i] = mesa_4;
        trigger_1_out[i] = trigger_1;
        trigger_2_out[i] = trigger_2;
        trigger_3_out[i] = trigger_3;
        trigger_4_out[i] = trigger_4;
    }

    Ok(())
}

pub fn mesa_stochastic_multi_length(
    input: &MesaStochasticMultiLengthInput,
) -> Result<MesaStochasticMultiLengthOutput, MesaStochasticMultiLengthError> {
    mesa_stochastic_multi_length_with_kernel(input, Kernel::Auto)
}

pub fn mesa_stochastic_multi_length_with_kernel(
    input: &MesaStochasticMultiLengthInput,
    _kernel: Kernel,
) -> Result<MesaStochasticMultiLengthOutput, MesaStochasticMultiLengthError> {
    let source = extract_source(input)?;
    let _ = first_valid(source).ok_or(MesaStochasticMultiLengthError::AllValuesNaN)?;
    let params = ValidatedParams::from_params(&input.params)?;
    let n = source.len();
    let mut out = MesaStochasticMultiLengthOutput {
        mesa_1: alloc_uninit_f64(n),
        mesa_2: alloc_uninit_f64(n),
        mesa_3: alloc_uninit_f64(n),
        mesa_4: alloc_uninit_f64(n),
        trigger_1: alloc_uninit_f64(n),
        trigger_2: alloc_uninit_f64(n),
        trigger_3: alloc_uninit_f64(n),
        trigger_4: alloc_uninit_f64(n),
    };
    compute_mesa_stochastic_multi_length_into(
        source,
        params,
        &mut out.mesa_1,
        &mut out.mesa_2,
        &mut out.mesa_3,
        &mut out.mesa_4,
        &mut out.trigger_1,
        &mut out.trigger_2,
        &mut out.trigger_3,
        &mut out.trigger_4,
    )?;
    Ok(out)
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[allow(clippy::too_many_arguments)]
pub fn mesa_stochastic_multi_length_into(
    mesa_1_out: &mut [f64],
    mesa_2_out: &mut [f64],
    mesa_3_out: &mut [f64],
    mesa_4_out: &mut [f64],
    trigger_1_out: &mut [f64],
    trigger_2_out: &mut [f64],
    trigger_3_out: &mut [f64],
    trigger_4_out: &mut [f64],
    input: &MesaStochasticMultiLengthInput,
    kernel: Kernel,
) -> Result<(), MesaStochasticMultiLengthError> {
    mesa_stochastic_multi_length_into_slice(
        mesa_1_out,
        mesa_2_out,
        mesa_3_out,
        mesa_4_out,
        trigger_1_out,
        trigger_2_out,
        trigger_3_out,
        trigger_4_out,
        input,
        kernel,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn mesa_stochastic_multi_length_into_slice(
    mesa_1_out: &mut [f64],
    mesa_2_out: &mut [f64],
    mesa_3_out: &mut [f64],
    mesa_4_out: &mut [f64],
    trigger_1_out: &mut [f64],
    trigger_2_out: &mut [f64],
    trigger_3_out: &mut [f64],
    trigger_4_out: &mut [f64],
    input: &MesaStochasticMultiLengthInput,
    _kernel: Kernel,
) -> Result<(), MesaStochasticMultiLengthError> {
    let source = extract_source(input)?;
    let _ = first_valid(source).ok_or(MesaStochasticMultiLengthError::AllValuesNaN)?;
    let params = ValidatedParams::from_params(&input.params)?;
    compute_mesa_stochastic_multi_length_into(
        source,
        params,
        mesa_1_out,
        mesa_2_out,
        mesa_3_out,
        mesa_4_out,
        trigger_1_out,
        trigger_2_out,
        trigger_3_out,
        trigger_4_out,
    )
}

#[derive(Clone, Debug)]
pub struct MesaStochasticMultiLengthBatchRange {
    pub length_1: (usize, usize, usize),
    pub length_2: (usize, usize, usize),
    pub length_3: (usize, usize, usize),
    pub length_4: (usize, usize, usize),
    pub trigger_length: (usize, usize, usize),
}

#[derive(Clone, Debug)]
pub struct MesaStochasticMultiLengthBatchOutput {
    pub mesa_1: Vec<f64>,
    pub mesa_2: Vec<f64>,
    pub mesa_3: Vec<f64>,
    pub mesa_4: Vec<f64>,
    pub trigger_1: Vec<f64>,
    pub trigger_2: Vec<f64>,
    pub trigger_3: Vec<f64>,
    pub trigger_4: Vec<f64>,
    pub combos: Vec<MesaStochasticMultiLengthParams>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct MesaStochasticMultiLengthBatchBuilder {
    source: Option<&'static str>,
    length_1: (usize, usize, usize),
    length_2: (usize, usize, usize),
    length_3: (usize, usize, usize),
    length_4: (usize, usize, usize),
    trigger_length: (usize, usize, usize),
    kernel: Kernel,
}

impl Default for MesaStochasticMultiLengthBatchBuilder {
    fn default() -> Self {
        Self {
            source: None,
            length_1: (DEFAULT_LENGTH_1, DEFAULT_LENGTH_1, 0),
            length_2: (DEFAULT_LENGTH_2, DEFAULT_LENGTH_2, 0),
            length_3: (DEFAULT_LENGTH_3, DEFAULT_LENGTH_3, 0),
            length_4: (DEFAULT_LENGTH_4, DEFAULT_LENGTH_4, 0),
            trigger_length: (DEFAULT_TRIGGER_LENGTH, DEFAULT_TRIGGER_LENGTH, 0),
            kernel: Kernel::Auto,
        }
    }
}

impl MesaStochasticMultiLengthBatchBuilder {
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
    pub fn length_1_range(mut self, value: (usize, usize, usize)) -> Self {
        self.length_1 = value;
        self
    }

    #[inline(always)]
    pub fn length_2_range(mut self, value: (usize, usize, usize)) -> Self {
        self.length_2 = value;
        self
    }

    #[inline(always)]
    pub fn length_3_range(mut self, value: (usize, usize, usize)) -> Self {
        self.length_3 = value;
        self
    }

    #[inline(always)]
    pub fn length_4_range(mut self, value: (usize, usize, usize)) -> Self {
        self.length_4 = value;
        self
    }

    #[inline(always)]
    pub fn trigger_length_range(mut self, value: (usize, usize, usize)) -> Self {
        self.trigger_length = value;
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
    ) -> Result<MesaStochasticMultiLengthBatchOutput, MesaStochasticMultiLengthError> {
        let source =
            mesa_stochastic_multi_length_source(candles, self.source.unwrap_or(DEFAULT_SOURCE));
        mesa_stochastic_multi_length_batch_with_kernel(
            source,
            &MesaStochasticMultiLengthBatchRange {
                length_1: self.length_1,
                length_2: self.length_2,
                length_3: self.length_3,
                length_4: self.length_4,
                trigger_length: self.trigger_length,
            },
            self.kernel,
        )
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        source: &[f64],
    ) -> Result<MesaStochasticMultiLengthBatchOutput, MesaStochasticMultiLengthError> {
        mesa_stochastic_multi_length_batch_with_kernel(
            source,
            &MesaStochasticMultiLengthBatchRange {
                length_1: self.length_1,
                length_2: self.length_2,
                length_3: self.length_3,
                length_4: self.length_4,
                trigger_length: self.trigger_length,
            },
            self.kernel,
        )
    }
}

fn expand_one_range(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, MesaStochasticMultiLengthError> {
    if start == 0 {
        return Err(MesaStochasticMultiLengthError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0 {
        if start != end {
            return Err(MesaStochasticMultiLengthError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if start > end {
        return Err(MesaStochasticMultiLengthError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut values = Vec::new();
    let mut current = start;
    while current <= end {
        values.push(current);
        current = match current.checked_add(step) {
            Some(next) => next,
            None => break,
        };
    }
    Ok(values)
}

pub fn expand_grid(
    sweep: &MesaStochasticMultiLengthBatchRange,
) -> Result<Vec<MesaStochasticMultiLengthParams>, MesaStochasticMultiLengthError> {
    let lengths_1 = expand_one_range(sweep.length_1.0, sweep.length_1.1, sweep.length_1.2)?;
    let lengths_2 = expand_one_range(sweep.length_2.0, sweep.length_2.1, sweep.length_2.2)?;
    let lengths_3 = expand_one_range(sweep.length_3.0, sweep.length_3.1, sweep.length_3.2)?;
    let lengths_4 = expand_one_range(sweep.length_4.0, sweep.length_4.1, sweep.length_4.2)?;
    let trigger_lengths = expand_one_range(
        sweep.trigger_length.0,
        sweep.trigger_length.1,
        sweep.trigger_length.2,
    )?;

    let mut out = Vec::new();
    for length_1 in lengths_1 {
        for &length_2 in &lengths_2 {
            for &length_3 in &lengths_3 {
                for &length_4 in &lengths_4 {
                    for &trigger_length in &trigger_lengths {
                        out.push(MesaStochasticMultiLengthParams {
                            length_1: Some(length_1),
                            length_2: Some(length_2),
                            length_3: Some(length_3),
                            length_4: Some(length_4),
                            trigger_length: Some(trigger_length),
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

fn batch_compute_rows(
    source: &[f64],
    sweep: &MesaStochasticMultiLengthBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<
    (
        Vec<MesaStochasticMultiLengthParams>,
        Vec<MesaStochasticMultiLengthOutput>,
    ),
    MesaStochasticMultiLengthError,
> {
    let combos = expand_grid(sweep)?;
    let kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel().to_non_batch(),
        other if other.is_batch() => other.to_non_batch(),
        other => other.to_non_batch(),
    };
    let compute = |params: &MesaStochasticMultiLengthParams| {
        let input = MesaStochasticMultiLengthInput::from_slice(source, params.clone());
        mesa_stochastic_multi_length_with_kernel(&input, kernel)
    };
    let rows = if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            combos
                .par_iter()
                .map(compute)
                .collect::<Result<Vec<_>, _>>()?
        }
        #[cfg(target_arch = "wasm32")]
        {
            combos.iter().map(compute).collect::<Result<Vec<_>, _>>()?
        }
    } else {
        combos.iter().map(compute).collect::<Result<Vec<_>, _>>()?
    };
    Ok((combos, rows))
}

fn flatten_rows(
    rows: &[MesaStochasticMultiLengthOutput],
    cols: usize,
) -> MesaStochasticMultiLengthOutput {
    let total = rows.len() * cols;
    let mut out = MesaStochasticMultiLengthOutput {
        mesa_1: vec![f64::NAN; total],
        mesa_2: vec![f64::NAN; total],
        mesa_3: vec![f64::NAN; total],
        mesa_4: vec![f64::NAN; total],
        trigger_1: vec![f64::NAN; total],
        trigger_2: vec![f64::NAN; total],
        trigger_3: vec![f64::NAN; total],
        trigger_4: vec![f64::NAN; total],
    };
    for (row_idx, row) in rows.iter().enumerate() {
        let start = row_idx * cols;
        let end = start + cols;
        out.mesa_1[start..end].copy_from_slice(&row.mesa_1);
        out.mesa_2[start..end].copy_from_slice(&row.mesa_2);
        out.mesa_3[start..end].copy_from_slice(&row.mesa_3);
        out.mesa_4[start..end].copy_from_slice(&row.mesa_4);
        out.trigger_1[start..end].copy_from_slice(&row.trigger_1);
        out.trigger_2[start..end].copy_from_slice(&row.trigger_2);
        out.trigger_3[start..end].copy_from_slice(&row.trigger_3);
        out.trigger_4[start..end].copy_from_slice(&row.trigger_4);
    }
    out
}

pub fn mesa_stochastic_multi_length_batch_with_kernel(
    source: &[f64],
    sweep: &MesaStochasticMultiLengthBatchRange,
    kernel: Kernel,
) -> Result<MesaStochasticMultiLengthBatchOutput, MesaStochasticMultiLengthError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(MesaStochasticMultiLengthError::InvalidKernelForBatch(
                kernel,
            ))
        }
    };
    mesa_stochastic_multi_length_batch_par_slice(source, sweep, batch_kernel.to_non_batch())
}

pub fn mesa_stochastic_multi_length_batch_slice(
    source: &[f64],
    sweep: &MesaStochasticMultiLengthBatchRange,
    kernel: Kernel,
) -> Result<MesaStochasticMultiLengthBatchOutput, MesaStochasticMultiLengthError> {
    if source.is_empty() {
        return Err(MesaStochasticMultiLengthError::EmptyInputData);
    }
    let _ = first_valid(source).ok_or(MesaStochasticMultiLengthError::AllValuesNaN)?;
    let (combos, rows) = batch_compute_rows(source, sweep, kernel, false)?;
    let flat = flatten_rows(&rows, source.len());
    Ok(MesaStochasticMultiLengthBatchOutput {
        mesa_1: flat.mesa_1,
        mesa_2: flat.mesa_2,
        mesa_3: flat.mesa_3,
        mesa_4: flat.mesa_4,
        trigger_1: flat.trigger_1,
        trigger_2: flat.trigger_2,
        trigger_3: flat.trigger_3,
        trigger_4: flat.trigger_4,
        rows: combos.len(),
        cols: source.len(),
        combos,
    })
}

pub fn mesa_stochastic_multi_length_batch_par_slice(
    source: &[f64],
    sweep: &MesaStochasticMultiLengthBatchRange,
    kernel: Kernel,
) -> Result<MesaStochasticMultiLengthBatchOutput, MesaStochasticMultiLengthError> {
    if source.is_empty() {
        return Err(MesaStochasticMultiLengthError::EmptyInputData);
    }
    let _ = first_valid(source).ok_or(MesaStochasticMultiLengthError::AllValuesNaN)?;
    let (combos, rows) = batch_compute_rows(source, sweep, kernel, true)?;
    let flat = flatten_rows(&rows, source.len());
    Ok(MesaStochasticMultiLengthBatchOutput {
        mesa_1: flat.mesa_1,
        mesa_2: flat.mesa_2,
        mesa_3: flat.mesa_3,
        mesa_4: flat.mesa_4,
        trigger_1: flat.trigger_1,
        trigger_2: flat.trigger_2,
        trigger_3: flat.trigger_3,
        trigger_4: flat.trigger_4,
        rows: combos.len(),
        cols: source.len(),
        combos,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn mesa_stochastic_multi_length_batch_into_slice(
    mesa_1_out: &mut [f64],
    mesa_2_out: &mut [f64],
    mesa_3_out: &mut [f64],
    mesa_4_out: &mut [f64],
    trigger_1_out: &mut [f64],
    trigger_2_out: &mut [f64],
    trigger_3_out: &mut [f64],
    trigger_4_out: &mut [f64],
    source: &[f64],
    sweep: &MesaStochasticMultiLengthBatchRange,
    kernel: Kernel,
) -> Result<(), MesaStochasticMultiLengthError> {
    if source.is_empty() {
        return Err(MesaStochasticMultiLengthError::EmptyInputData);
    }
    let _ = first_valid(source).ok_or(MesaStochasticMultiLengthError::AllValuesNaN)?;
    let combos = expand_grid(sweep)?;
    let expected = combos.len().checked_mul(source.len()).ok_or_else(|| {
        MesaStochasticMultiLengthError::InvalidRange {
            start: combos.len().to_string(),
            end: source.len().to_string(),
            step: "rows*cols".to_string(),
        }
    })?;
    let got = [
        mesa_1_out.len(),
        mesa_2_out.len(),
        mesa_3_out.len(),
        mesa_4_out.len(),
        trigger_1_out.len(),
        trigger_2_out.len(),
        trigger_3_out.len(),
        trigger_4_out.len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    if mesa_1_out.len() != expected
        || mesa_2_out.len() != expected
        || mesa_3_out.len() != expected
        || mesa_4_out.len() != expected
        || trigger_1_out.len() != expected
        || trigger_2_out.len() != expected
        || trigger_3_out.len() != expected
        || trigger_4_out.len() != expected
    {
        return Err(MesaStochasticMultiLengthError::OutputLengthMismatch { expected, got });
    }
    let (combos, rows) = batch_compute_rows(source, sweep, kernel, false)?;
    let cols = source.len();
    for (row_idx, row) in rows.iter().enumerate() {
        let start = row_idx * cols;
        let end = start + cols;
        mesa_1_out[start..end].copy_from_slice(&row.mesa_1);
        mesa_2_out[start..end].copy_from_slice(&row.mesa_2);
        mesa_3_out[start..end].copy_from_slice(&row.mesa_3);
        mesa_4_out[start..end].copy_from_slice(&row.mesa_4);
        trigger_1_out[start..end].copy_from_slice(&row.trigger_1);
        trigger_2_out[start..end].copy_from_slice(&row.trigger_2);
        trigger_3_out[start..end].copy_from_slice(&row.trigger_3);
        trigger_4_out[start..end].copy_from_slice(&row.trigger_4);
    }
    debug_assert_eq!(combos.len() * cols, expected);
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "mesa_stochastic_multi_length")]
#[pyo3(signature = (
    source,
    length_1=48,
    length_2=21,
    length_3=9,
    length_4=6,
    trigger_length=2,
    kernel=None
))]
pub fn mesa_stochastic_multi_length_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    length_1: usize,
    length_2: usize,
    length_3: usize,
    length_4: usize,
    trigger_length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = source.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = MesaStochasticMultiLengthInput::from_slice(
        source,
        MesaStochasticMultiLengthParams {
            length_1: Some(length_1),
            length_2: Some(length_2),
            length_3: Some(length_3),
            length_4: Some(length_4),
            trigger_length: Some(trigger_length),
        },
    );
    let out = py
        .allow_threads(|| mesa_stochastic_multi_length_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("mesa_1", out.mesa_1.into_pyarray(py))?;
    dict.set_item("mesa_2", out.mesa_2.into_pyarray(py))?;
    dict.set_item("mesa_3", out.mesa_3.into_pyarray(py))?;
    dict.set_item("mesa_4", out.mesa_4.into_pyarray(py))?;
    dict.set_item("trigger_1", out.trigger_1.into_pyarray(py))?;
    dict.set_item("trigger_2", out.trigger_2.into_pyarray(py))?;
    dict.set_item("trigger_3", out.trigger_3.into_pyarray(py))?;
    dict.set_item("trigger_4", out.trigger_4.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "MesaStochasticMultiLengthStream")]
pub struct MesaStochasticMultiLengthStreamPy {
    stream: MesaStochasticMultiLengthStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MesaStochasticMultiLengthStreamPy {
    #[new]
    #[pyo3(signature = (
        length_1=48,
        length_2=21,
        length_3=9,
        length_4=6,
        trigger_length=2
    ))]
    fn new(
        length_1: usize,
        length_2: usize,
        length_3: usize,
        length_4: usize,
        trigger_length: usize,
    ) -> PyResult<Self> {
        let stream = MesaStochasticMultiLengthStream::try_new(MesaStochasticMultiLengthParams {
            length_1: Some(length_1),
            length_2: Some(length_2),
            length_3: Some(length_3),
            length_4: Some(length_4),
            trigger_length: Some(trigger_length),
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update<'py>(&mut self, py: Python<'py>, source: f64) -> PyResult<Bound<'py, PyDict>> {
        let values = self.stream.update(source);
        let dict = PyDict::new(py);
        dict.set_item("mesa_1", values.0)?;
        dict.set_item("mesa_2", values.1)?;
        dict.set_item("mesa_3", values.2)?;
        dict.set_item("mesa_4", values.3)?;
        dict.set_item("trigger_1", values.4)?;
        dict.set_item("trigger_2", values.5)?;
        dict.set_item("trigger_3", values.6)?;
        dict.set_item("trigger_4", values.7)?;
        Ok(dict)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "mesa_stochastic_multi_length_batch")]
#[pyo3(signature = (
    source,
    length_1_range=(48,48,0),
    length_2_range=(21,21,0),
    length_3_range=(9,9,0),
    length_4_range=(6,6,0),
    trigger_length_range=(2,2,0),
    kernel=None
))]
pub fn mesa_stochastic_multi_length_batch_py<'py>(
    py: Python<'py>,
    source: PyReadonlyArray1<'py, f64>,
    length_1_range: (usize, usize, usize),
    length_2_range: (usize, usize, usize),
    length_3_range: (usize, usize, usize),
    length_4_range: (usize, usize, usize),
    trigger_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = source.as_slice()?;
    let kernel = validate_kernel(kernel, true)?;
    let sweep = MesaStochasticMultiLengthBatchRange {
        length_1: length_1_range,
        length_2: length_2_range,
        length_3: length_3_range,
        length_4: length_4_range,
        trigger_length: trigger_length_range,
    };
    let out = py
        .allow_threads(|| mesa_stochastic_multi_length_batch_with_kernel(source, &sweep, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item(
        "mesa_1",
        out.mesa_1.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "mesa_2",
        out.mesa_2.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "mesa_3",
        out.mesa_3.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "mesa_4",
        out.mesa_4.into_pyarray(py).reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "trigger_1",
        out.trigger_1
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "trigger_2",
        out.trigger_2
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "trigger_3",
        out.trigger_3
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "trigger_4",
        out.trigger_4
            .into_pyarray(py)
            .reshape((out.rows, out.cols))?,
    )?;
    dict.set_item(
        "length_1",
        out.combos
            .iter()
            .map(|p| p.length_1.unwrap())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "length_2",
        out.combos
            .iter()
            .map(|p| p.length_2.unwrap())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "length_3",
        out.combos
            .iter()
            .map(|p| p.length_3.unwrap())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "length_4",
        out.combos
            .iter()
            .map(|p| p.length_4.unwrap())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "trigger_length",
        out.combos
            .iter()
            .map(|p| p.trigger_length.unwrap())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("rows", out.rows)?;
    dict.set_item("cols", out.cols)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_mesa_stochastic_multi_length_module(
    m: &Bound<'_, pyo3::types::PyModule>,
) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mesa_stochastic_multi_length_py, m)?)?;
    m.add_function(wrap_pyfunction!(mesa_stochastic_multi_length_batch_py, m)?)?;
    m.add_class::<MesaStochasticMultiLengthStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MesaStochasticMultiLengthJsOutput {
    pub mesa_1: Vec<f64>,
    pub mesa_2: Vec<f64>,
    pub mesa_3: Vec<f64>,
    pub mesa_4: Vec<f64>,
    pub trigger_1: Vec<f64>,
    pub trigger_2: Vec<f64>,
    pub trigger_3: Vec<f64>,
    pub trigger_4: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MesaStochasticMultiLengthBatchConfig {
    pub length_1_range: Vec<f64>,
    pub length_2_range: Vec<f64>,
    pub length_3_range: Vec<f64>,
    pub length_4_range: Vec<f64>,
    pub trigger_length_range: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct MesaStochasticMultiLengthBatchJsOutput {
    pub mesa_1: Vec<f64>,
    pub mesa_2: Vec<f64>,
    pub mesa_3: Vec<f64>,
    pub mesa_4: Vec<f64>,
    pub trigger_1: Vec<f64>,
    pub trigger_2: Vec<f64>,
    pub trigger_3: Vec<f64>,
    pub trigger_4: Vec<f64>,
    pub length_1: Vec<usize>,
    pub length_2: Vec<usize>,
    pub length_3: Vec<usize>,
    pub length_4: Vec<usize>,
    pub trigger_length: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn js_vec3_to_usize(name: &str, values: &[f64]) -> Result<(usize, usize, usize), JsValue> {
    if values.len() != 3 {
        return Err(JsValue::from_str(&format!(
            "Invalid config: {name} must have exactly 3 elements [start, end, step]"
        )));
    }
    let mut out = [0usize; 3];
    for (i, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name}[{i}] must be a finite non-negative whole number"
            )));
        }
        let rounded = value.round();
        if (value - rounded).abs() > 1e-9 {
            return Err(JsValue::from_str(&format!(
                "Invalid config: {name}[{i}] must be a whole number"
            )));
        }
        out[i] = rounded as usize;
    }
    Ok((out[0], out[1], out[2]))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "mesa_stochastic_multi_length_js")]
pub fn mesa_stochastic_multi_length_js(
    source: &[f64],
    length_1: usize,
    length_2: usize,
    length_3: usize,
    length_4: usize,
    trigger_length: usize,
) -> Result<JsValue, JsValue> {
    let input = MesaStochasticMultiLengthInput::from_slice(
        source,
        MesaStochasticMultiLengthParams {
            length_1: Some(length_1),
            length_2: Some(length_2),
            length_3: Some(length_3),
            length_4: Some(length_4),
            trigger_length: Some(trigger_length),
        },
    );
    let out = mesa_stochastic_multi_length_with_kernel(&input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MesaStochasticMultiLengthJsOutput {
        mesa_1: out.mesa_1,
        mesa_2: out.mesa_2,
        mesa_3: out.mesa_3,
        mesa_4: out.mesa_4,
        trigger_1: out.trigger_1,
        trigger_2: out.trigger_2,
        trigger_3: out.trigger_3,
        trigger_4: out.trigger_4,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "mesa_stochastic_multi_length_batch_js")]
pub fn mesa_stochastic_multi_length_batch_js(
    source: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: MesaStochasticMultiLengthBatchConfig =
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let sweep = MesaStochasticMultiLengthBatchRange {
        length_1: js_vec3_to_usize("length_1_range", &config.length_1_range)?,
        length_2: js_vec3_to_usize("length_2_range", &config.length_2_range)?,
        length_3: js_vec3_to_usize("length_3_range", &config.length_3_range)?,
        length_4: js_vec3_to_usize("length_4_range", &config.length_4_range)?,
        trigger_length: js_vec3_to_usize("trigger_length_range", &config.trigger_length_range)?,
    };
    let out = mesa_stochastic_multi_length_batch_with_kernel(source, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&MesaStochasticMultiLengthBatchJsOutput {
        mesa_1: out.mesa_1,
        mesa_2: out.mesa_2,
        mesa_3: out.mesa_3,
        mesa_4: out.mesa_4,
        trigger_1: out.trigger_1,
        trigger_2: out.trigger_2,
        trigger_3: out.trigger_3,
        trigger_4: out.trigger_4,
        length_1: out.combos.iter().map(|p| p.length_1.unwrap()).collect(),
        length_2: out.combos.iter().map(|p| p.length_2.unwrap()).collect(),
        length_3: out.combos.iter().map(|p| p.length_3.unwrap()).collect(),
        length_4: out.combos.iter().map(|p| p.length_4.unwrap()).collect(),
        trigger_length: out
            .combos
            .iter()
            .map(|p| p.trigger_length.unwrap())
            .collect(),
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mesa_stochastic_multi_length_alloc(len: usize) -> *mut f64 {
    let mut buf = vec![0.0; len];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mesa_stochastic_multi_length_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(ptr, 0, len));
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "mesa_stochastic_multi_length_into")]
#[allow(clippy::too_many_arguments)]
pub fn mesa_stochastic_multi_length_into(
    source_ptr: *const f64,
    mesa_1_ptr: *mut f64,
    mesa_2_ptr: *mut f64,
    mesa_3_ptr: *mut f64,
    mesa_4_ptr: *mut f64,
    trigger_1_ptr: *mut f64,
    trigger_2_ptr: *mut f64,
    trigger_3_ptr: *mut f64,
    trigger_4_ptr: *mut f64,
    len: usize,
    length_1: usize,
    length_2: usize,
    length_3: usize,
    length_4: usize,
    trigger_length: usize,
) -> Result<(), JsValue> {
    if source_ptr.is_null()
        || mesa_1_ptr.is_null()
        || mesa_2_ptr.is_null()
        || mesa_3_ptr.is_null()
        || mesa_4_ptr.is_null()
        || trigger_1_ptr.is_null()
        || trigger_2_ptr.is_null()
        || trigger_3_ptr.is_null()
        || trigger_4_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to mesa_stochastic_multi_length_into",
        ));
    }
    let source = unsafe { std::slice::from_raw_parts(source_ptr, len) };
    let mesa_1_out = unsafe { std::slice::from_raw_parts_mut(mesa_1_ptr, len) };
    let mesa_2_out = unsafe { std::slice::from_raw_parts_mut(mesa_2_ptr, len) };
    let mesa_3_out = unsafe { std::slice::from_raw_parts_mut(mesa_3_ptr, len) };
    let mesa_4_out = unsafe { std::slice::from_raw_parts_mut(mesa_4_ptr, len) };
    let trigger_1_out = unsafe { std::slice::from_raw_parts_mut(trigger_1_ptr, len) };
    let trigger_2_out = unsafe { std::slice::from_raw_parts_mut(trigger_2_ptr, len) };
    let trigger_3_out = unsafe { std::slice::from_raw_parts_mut(trigger_3_ptr, len) };
    let trigger_4_out = unsafe { std::slice::from_raw_parts_mut(trigger_4_ptr, len) };
    let input = MesaStochasticMultiLengthInput::from_slice(
        source,
        MesaStochasticMultiLengthParams {
            length_1: Some(length_1),
            length_2: Some(length_2),
            length_3: Some(length_3),
            length_4: Some(length_4),
            trigger_length: Some(trigger_length),
        },
    );
    mesa_stochastic_multi_length_into_slice(
        mesa_1_out,
        mesa_2_out,
        mesa_3_out,
        mesa_4_out,
        trigger_1_out,
        trigger_2_out,
        trigger_3_out,
        trigger_4_out,
        &input,
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "mesa_stochastic_multi_length_batch_into")]
#[allow(clippy::too_many_arguments)]
pub fn mesa_stochastic_multi_length_batch_into(
    source_ptr: *const f64,
    mesa_1_ptr: *mut f64,
    mesa_2_ptr: *mut f64,
    mesa_3_ptr: *mut f64,
    mesa_4_ptr: *mut f64,
    trigger_1_ptr: *mut f64,
    trigger_2_ptr: *mut f64,
    trigger_3_ptr: *mut f64,
    trigger_4_ptr: *mut f64,
    len: usize,
    length_1_start: usize,
    length_1_end: usize,
    length_1_step: usize,
    length_2_start: usize,
    length_2_end: usize,
    length_2_step: usize,
    length_3_start: usize,
    length_3_end: usize,
    length_3_step: usize,
    length_4_start: usize,
    length_4_end: usize,
    length_4_step: usize,
    trigger_length_start: usize,
    trigger_length_end: usize,
    trigger_length_step: usize,
) -> Result<usize, JsValue> {
    if source_ptr.is_null()
        || mesa_1_ptr.is_null()
        || mesa_2_ptr.is_null()
        || mesa_3_ptr.is_null()
        || mesa_4_ptr.is_null()
        || trigger_1_ptr.is_null()
        || trigger_2_ptr.is_null()
        || trigger_3_ptr.is_null()
        || trigger_4_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to mesa_stochastic_multi_length_batch_into",
        ));
    }
    let source = unsafe { std::slice::from_raw_parts(source_ptr, len) };
    let sweep = MesaStochasticMultiLengthBatchRange {
        length_1: (length_1_start, length_1_end, length_1_step),
        length_2: (length_2_start, length_2_end, length_2_step),
        length_3: (length_3_start, length_3_end, length_3_step),
        length_4: (length_4_start, length_4_end, length_4_step),
        trigger_length: (
            trigger_length_start,
            trigger_length_end,
            trigger_length_step,
        ),
    };
    let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let total = rows.checked_mul(len).ok_or_else(|| {
        JsValue::from_str("rows*cols overflow in mesa_stochastic_multi_length_batch_into")
    })?;
    let mesa_1_out = unsafe { std::slice::from_raw_parts_mut(mesa_1_ptr, total) };
    let mesa_2_out = unsafe { std::slice::from_raw_parts_mut(mesa_2_ptr, total) };
    let mesa_3_out = unsafe { std::slice::from_raw_parts_mut(mesa_3_ptr, total) };
    let mesa_4_out = unsafe { std::slice::from_raw_parts_mut(mesa_4_ptr, total) };
    let trigger_1_out = unsafe { std::slice::from_raw_parts_mut(trigger_1_ptr, total) };
    let trigger_2_out = unsafe { std::slice::from_raw_parts_mut(trigger_2_ptr, total) };
    let trigger_3_out = unsafe { std::slice::from_raw_parts_mut(trigger_3_ptr, total) };
    let trigger_4_out = unsafe { std::slice::from_raw_parts_mut(trigger_4_ptr, total) };
    mesa_stochastic_multi_length_batch_into_slice(
        mesa_1_out,
        mesa_2_out,
        mesa_3_out,
        mesa_4_out,
        trigger_1_out,
        trigger_2_out,
        trigger_3_out,
        trigger_4_out,
        source,
        &sweep,
        Kernel::Auto,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(rows)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mesa_stochastic_multi_length_output_into_js(
    source: &[f64],
    length_1: usize,
    length_2: usize,
    length_3: usize,
    length_4: usize,
    trigger_length: usize,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = mesa_stochastic_multi_length_js(
        source,
        length_1,
        length_2,
        length_3,
        length_4,
        trigger_length,
    )?;
    crate::write_wasm_object_f64_outputs("mesa_stochastic_multi_length_output_into_js", &value, out)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn mesa_stochastic_multi_length_batch_output_into_js(
    source: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = mesa_stochastic_multi_length_batch_js(source, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "mesa_stochastic_multi_length_batch_output_into_js",
        &value,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn manual_sma(values: &[f64], length: usize) -> Vec<f64> {
        let mut out = vec![f64::NAN; values.len()];
        if length == 0 || values.len() < length {
            return out;
        }
        for i in (length - 1)..values.len() {
            let window = &values[i + 1 - length..=i];
            if window.iter().all(|v| v.is_finite()) {
                out[i] = window.iter().sum::<f64>() / length as f64;
            }
        }
        out
    }

    fn manual_reference(
        source: &[f64],
        params: ValidatedParams,
    ) -> MesaStochasticMultiLengthOutput {
        let n = source.len();
        let mut out = MesaStochasticMultiLengthOutput {
            mesa_1: vec![f64::NAN; n],
            mesa_2: vec![f64::NAN; n],
            mesa_3: vec![f64::NAN; n],
            mesa_4: vec![f64::NAN; n],
            trigger_1: vec![f64::NAN; n],
            trigger_2: vec![f64::NAN; n],
            trigger_3: vec![f64::NAN; n],
            trigger_4: vec![f64::NAN; n],
        };

        let alpha1 = ((0.707 * 2.0 * PI / 48.0).cos() + (0.707 * 2.0 * PI / 48.0).sin() - 1.0)
            / (0.707 * 2.0 * PI / 48.0).cos();
        let hp_coef = (1.0 - alpha1 * 0.5) * (1.0 - alpha1 * 0.5);
        let one_minus_alpha = 1.0 - alpha1;
        let hp_feedback_1 = 2.0 * one_minus_alpha;
        let hp_feedback_2 = -(one_minus_alpha * one_minus_alpha);
        let a1 = (-1.414 * PI / 10.0).exp();
        let b1 = 2.0 * a1 * (1.414 * PI / 10.0).cos();
        let c2 = b1;
        let c3 = -(a1 * a1);
        let c1 = 1.0 - c2 - c3;

        let mut hp = vec![f64::NAN; n];
        let mut filt = vec![f64::NAN; n];
        for i in 0..n {
            if source[i].is_finite() {
                let src1 = if i >= 1 { nz(source[i - 1]) } else { 0.0 };
                let src2 = if i >= 2 { nz(source[i - 2]) } else { 0.0 };
                let hp1 = if i >= 1 { nz(hp[i - 1]) } else { 0.0 };
                let hp2 = if i >= 2 { nz(hp[i - 2]) } else { 0.0 };
                hp[i] = hp_coef.mul_add(
                    source[i] - 2.0 * src1 + src2,
                    hp_feedback_1.mul_add(hp1, hp_feedback_2 * hp2),
                );
                let filt1 = if i >= 1 { nz(filt[i - 1]) } else { 0.0 };
                let filt2 = if i >= 2 { nz(filt[i - 2]) } else { 0.0 };
                filt[i] = c1.mul_add(hp[i], c2.mul_add(filt1, c3 * filt2));
            }
        }

        fn mesa_from_filt(filt: &[f64], length: usize, c1: f64, c2: f64, c3: f64) -> Vec<f64> {
            let n = filt.len();
            let mut out = vec![f64::NAN; n];
            for i in 0..n {
                if !filt[i].is_finite() {
                    continue;
                }
                let mut highest = filt[i];
                let mut lowest = filt[i];
                for count in 0..length {
                    let value = if i >= count { nz(filt[i - count]) } else { 0.0 };
                    if value > highest {
                        highest = value;
                    }
                    if value < lowest {
                        lowest = value;
                    }
                }
                let denom = highest - lowest;
                if denom == 0.0 || !denom.is_finite() {
                    continue;
                }
                let stoc = (filt[i] - lowest) / denom;
                if !stoc.is_finite() {
                    continue;
                }
                let prev1 = if i >= 1 { nz(out[i - 1]) } else { 0.0 };
                let prev2 = if i >= 2 { nz(out[i - 2]) } else { 0.0 };
                out[i] = c1.mul_add(stoc, c2.mul_add(prev1, c3 * prev2));
            }
            out
        }

        out.mesa_1 = mesa_from_filt(&filt, params.length_1, c1, c2, c3);
        out.mesa_2 = mesa_from_filt(&filt, params.length_2, c1, c2, c3);
        out.mesa_3 = mesa_from_filt(&filt, params.length_3, c1, c2, c3);
        out.mesa_4 = mesa_from_filt(&filt, params.length_4, c1, c2, c3);
        out.trigger_1 = manual_sma(&out.mesa_1, params.trigger_length);
        out.trigger_2 = manual_sma(&out.mesa_2, params.trigger_length);
        out.trigger_3 = manual_sma(&out.mesa_3, params.trigger_length);
        out.trigger_4 = manual_sma(&out.mesa_4, params.trigger_length);
        out
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() <= 1e-12,
                "mismatch at {i}: actual={a:?} expected={b:?}"
            );
        }
    }

    #[test]
    fn manual_reference_matches_core() {
        let candles =
            read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv").unwrap();
        let source = &candles.close[..160];
        let params = MesaStochasticMultiLengthParams::default();
        let input = MesaStochasticMultiLengthInput::from_slice(source, params.clone());
        let got = mesa_stochastic_multi_length(&input).unwrap();
        let want = manual_reference(source, ValidatedParams::from_params(&params).unwrap());
        assert_close(&got.mesa_1, &want.mesa_1);
        assert_close(&got.mesa_2, &want.mesa_2);
        assert_close(&got.mesa_3, &want.mesa_3);
        assert_close(&got.mesa_4, &want.mesa_4);
        assert_close(&got.trigger_1, &want.trigger_1);
        assert_close(&got.trigger_2, &want.trigger_2);
        assert_close(&got.trigger_3, &want.trigger_3);
        assert_close(&got.trigger_4, &want.trigger_4);
    }

    #[test]
    fn stream_matches_batch() {
        let candles =
            read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv").unwrap();
        let source = &candles.close[..160];
        let input = MesaStochasticMultiLengthInput::from_slice(
            source,
            MesaStochasticMultiLengthParams::default(),
        );
        let batch = mesa_stochastic_multi_length(&input).unwrap();
        let mut stream =
            MesaStochasticMultiLengthStream::try_new(MesaStochasticMultiLengthParams::default())
                .unwrap();
        let mut last = (
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        );
        for &value in source {
            last = stream.update(value);
        }
        assert!(
            (last.0 - batch.mesa_1[source.len() - 1]).abs() <= 1e-12
                || (last.0.is_nan() && batch.mesa_1[source.len() - 1].is_nan())
        );
        assert!(
            (last.1 - batch.mesa_2[source.len() - 1]).abs() <= 1e-12
                || (last.1.is_nan() && batch.mesa_2[source.len() - 1].is_nan())
        );
        assert!(
            (last.2 - batch.mesa_3[source.len() - 1]).abs() <= 1e-12
                || (last.2.is_nan() && batch.mesa_3[source.len() - 1].is_nan())
        );
        assert!(
            (last.3 - batch.mesa_4[source.len() - 1]).abs() <= 1e-12
                || (last.3.is_nan() && batch.mesa_4[source.len() - 1].is_nan())
        );
        assert!(
            (last.4 - batch.trigger_1[source.len() - 1]).abs() <= 1e-12
                || (last.4.is_nan() && batch.trigger_1[source.len() - 1].is_nan())
        );
        assert!(
            (last.5 - batch.trigger_2[source.len() - 1]).abs() <= 1e-12
                || (last.5.is_nan() && batch.trigger_2[source.len() - 1].is_nan())
        );
        assert!(
            (last.6 - batch.trigger_3[source.len() - 1]).abs() <= 1e-12
                || (last.6.is_nan() && batch.trigger_3[source.len() - 1].is_nan())
        );
        assert!(
            (last.7 - batch.trigger_4[source.len() - 1]).abs() <= 1e-12
                || (last.7.is_nan() && batch.trigger_4[source.len() - 1].is_nan())
        );
    }

    #[test]
    fn batch_first_row_matches_single() {
        let candles =
            read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv").unwrap();
        let source = &candles.close[..128];
        let single = mesa_stochastic_multi_length(&MesaStochasticMultiLengthInput::from_slice(
            source,
            MesaStochasticMultiLengthParams::default(),
        ))
        .unwrap();
        let batch = mesa_stochastic_multi_length_batch_with_kernel(
            source,
            &MesaStochasticMultiLengthBatchRange {
                length_1: (48, 50, 2),
                length_2: (21, 21, 0),
                length_3: (9, 9, 0),
                length_4: (6, 6, 0),
                trigger_length: (2, 2, 0),
            },
            Kernel::ScalarBatch,
        )
        .unwrap();
        let cols = source.len();
        assert_eq!(batch.rows, 2);
        assert_close(&batch.mesa_1[..cols], &single.mesa_1);
        assert_close(&batch.trigger_1[..cols], &single.trigger_1);
    }

    #[test]
    fn into_slice_matches_owned_output() {
        let candles =
            read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv").unwrap();
        let source = &candles.close[..128];
        let input = MesaStochasticMultiLengthInput::from_slice(
            source,
            MesaStochasticMultiLengthParams::default(),
        );
        let single = mesa_stochastic_multi_length(&input).unwrap();
        let mut mesa_1 = vec![f64::NAN; source.len()];
        let mut mesa_2 = vec![f64::NAN; source.len()];
        let mut mesa_3 = vec![f64::NAN; source.len()];
        let mut mesa_4 = vec![f64::NAN; source.len()];
        let mut trigger_1 = vec![f64::NAN; source.len()];
        let mut trigger_2 = vec![f64::NAN; source.len()];
        let mut trigger_3 = vec![f64::NAN; source.len()];
        let mut trigger_4 = vec![f64::NAN; source.len()];
        mesa_stochastic_multi_length_into_slice(
            &mut mesa_1,
            &mut mesa_2,
            &mut mesa_3,
            &mut mesa_4,
            &mut trigger_1,
            &mut trigger_2,
            &mut trigger_3,
            &mut trigger_4,
            &input,
            Kernel::Auto,
        )
        .unwrap();
        assert_close(&mesa_1, &single.mesa_1);
        assert_close(&mesa_2, &single.mesa_2);
        assert_close(&mesa_3, &single.mesa_3);
        assert_close(&mesa_4, &single.mesa_4);
        assert_close(&trigger_1, &single.trigger_1);
        assert_close(&trigger_2, &single.trigger_2);
        assert_close(&trigger_3, &single.trigger_3);
        assert_close(&trigger_4, &single.trigger_4);
    }

    #[test]
    fn rejects_invalid_period() {
        let candles =
            read_candles_from_csv("src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv").unwrap();
        let input = MesaStochasticMultiLengthInput::from_slice(
            &candles.close[..64],
            MesaStochasticMultiLengthParams {
                length_1: Some(0),
                ..MesaStochasticMultiLengthParams::default()
            },
        );
        let err = mesa_stochastic_multi_length(&input).unwrap_err();
        assert!(err.to_string().contains("Invalid period"));
    }
}
