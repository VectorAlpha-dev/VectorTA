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
use crate::utilities::helpers::{detect_best_batch_kernel, make_uninit_matrix};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

const DEFAULT_LENGTH: usize = 70;
const DEFAULT_NORM_LENGTH: usize = 150;

impl<'a> AsRef<[f64]> for LeavittConvolutionAccelerationInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            LeavittConvolutionAccelerationData::Slice(slice) => slice,
            LeavittConvolutionAccelerationData::Candles { candles, source } => {
                leavitt_source(candles, source)
            }
        }
    }
}

#[inline(always)]
fn leavitt_source<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    match source {
        "hlcc4" | "hlcc" => &candles.hlcc4,
        "hl2" => &candles.hl2,
        "hlc3" => &candles.hlc3,
        "ohlc4" => &candles.ohlc4,
        "close" => &candles.close,
        "open" => &candles.open,
        "high" => &candles.high,
        "low" => &candles.low,
        "volume" => &candles.volume,
        _ => source_type(candles, source),
    }
}

#[derive(Debug, Clone)]
pub enum LeavittConvolutionAccelerationData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct LeavittConvolutionAccelerationOutput {
    pub conv_acceleration: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(
    all(target_arch = "wasm32", feature = "wasm"),
    derive(Serialize, Deserialize)
)]
pub struct LeavittConvolutionAccelerationParams {
    pub length: Option<usize>,
    pub norm_length: Option<usize>,
    pub use_norm_hyperbolic: Option<bool>,
}

impl Default for LeavittConvolutionAccelerationParams {
    fn default() -> Self {
        Self {
            length: Some(DEFAULT_LENGTH),
            norm_length: Some(DEFAULT_NORM_LENGTH),
            use_norm_hyperbolic: Some(true),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LeavittConvolutionAccelerationInput<'a> {
    pub data: LeavittConvolutionAccelerationData<'a>,
    pub params: LeavittConvolutionAccelerationParams,
}

impl<'a> LeavittConvolutionAccelerationInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: LeavittConvolutionAccelerationParams,
    ) -> Self {
        Self {
            data: LeavittConvolutionAccelerationData::Candles { candles, source },
            params,
        }
    }

    #[inline]
    pub fn from_slice(slice: &'a [f64], params: LeavittConvolutionAccelerationParams) -> Self {
        Self {
            data: LeavittConvolutionAccelerationData::Slice(slice),
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(
            candles,
            "hlcc4",
            LeavittConvolutionAccelerationParams::default(),
        )
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(DEFAULT_LENGTH)
    }

    #[inline]
    pub fn get_norm_length(&self) -> usize {
        self.params.norm_length.unwrap_or(DEFAULT_NORM_LENGTH)
    }

    #[inline]
    pub fn get_use_norm_hyperbolic(&self) -> bool {
        self.params.use_norm_hyperbolic.unwrap_or(true)
    }
}

#[derive(Clone, Debug)]
pub struct LeavittConvolutionAccelerationBuilder {
    length: Option<usize>,
    norm_length: Option<usize>,
    use_norm_hyperbolic: Option<bool>,
    source: Option<String>,
    kernel: Kernel,
}

impl Default for LeavittConvolutionAccelerationBuilder {
    fn default() -> Self {
        Self {
            length: None,
            norm_length: None,
            use_norm_hyperbolic: None,
            source: None,
            kernel: Kernel::Auto,
        }
    }
}

impl LeavittConvolutionAccelerationBuilder {
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
    pub fn norm_length(mut self, value: usize) -> Self {
        self.norm_length = Some(value);
        self
    }

    #[inline(always)]
    pub fn use_norm_hyperbolic(mut self, value: bool) -> Self {
        self.use_norm_hyperbolic = Some(value);
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
    ) -> Result<LeavittConvolutionAccelerationOutput, LeavittConvolutionAccelerationError> {
        let input = LeavittConvolutionAccelerationInput::from_candles(
            candles,
            self.source.as_deref().unwrap_or("hlcc4"),
            LeavittConvolutionAccelerationParams {
                length: self.length,
                norm_length: self.norm_length,
                use_norm_hyperbolic: self.use_norm_hyperbolic,
            },
        );
        leavitt_convolution_acceleration_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<LeavittConvolutionAccelerationOutput, LeavittConvolutionAccelerationError> {
        let input = LeavittConvolutionAccelerationInput::from_slice(
            data,
            LeavittConvolutionAccelerationParams {
                length: self.length,
                norm_length: self.norm_length,
                use_norm_hyperbolic: self.use_norm_hyperbolic,
            },
        );
        leavitt_convolution_acceleration_with_kernel(&input, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(
        self,
    ) -> Result<LeavittConvolutionAccelerationStream, LeavittConvolutionAccelerationError> {
        LeavittConvolutionAccelerationStream::try_new(LeavittConvolutionAccelerationParams {
            length: self.length,
            norm_length: self.norm_length,
            use_norm_hyperbolic: self.use_norm_hyperbolic,
        })
    }
}

#[derive(Debug, Error)]
pub enum LeavittConvolutionAccelerationError {
    #[error("leavitt_convolution_acceleration: Input data slice is empty.")]
    EmptyInputData,
    #[error("leavitt_convolution_acceleration: All source values are invalid.")]
    AllValuesNaN,
    #[error(
        "leavitt_convolution_acceleration: Invalid length: length = {length}, data length = {data_len}"
    )]
    InvalidLength { length: usize, data_len: usize },
    #[error(
        "leavitt_convolution_acceleration: Invalid norm_length: norm_length = {norm_length}, data length = {data_len}"
    )]
    InvalidNormLength { norm_length: usize, data_len: usize },
    #[error(
        "leavitt_convolution_acceleration: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "leavitt_convolution_acceleration: Output length mismatch: expected = {expected}, got = {got}"
    )]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error(
        "leavitt_convolution_acceleration: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("leavitt_convolution_acceleration: Invalid kernel for batch: {0:?}")]
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
fn sqrt_length(length: usize) -> usize {
    ((length as f64).sqrt().floor() as usize).max(1)
}

#[inline(always)]
fn required_valid_bars(length: usize, norm_length: usize) -> usize {
    length + sqrt_length(length) + norm_length - 2
}

#[inline(always)]
fn normalized_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => Kernel::Scalar,
        other if other.is_batch() => other.to_non_batch(),
        other => other,
    }
}

#[derive(Clone, Debug)]
struct RollingLinRegState {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    count: usize,
    filled: bool,
    n: f64,
    sum_x: f64,
    inv_n: f64,
    inv_denom: f64,
    mean_x: f64,
    forecast_x: f64,
    sum_y: f64,
    sum_xy: f64,
}

impl RollingLinRegState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        let n = period as f64;
        let m = period.saturating_sub(1) as f64;
        let sum_x = 0.5 * m * n;
        let sum_x2 = (m * n) * (2.0 * m + 1.0) / 6.0;
        let denom = n * sum_x2 - sum_x * sum_x;
        Self {
            period,
            buffer: vec![0.0; period.max(1)],
            head: 0,
            count: 0,
            filled: false,
            n: n.max(1.0),
            sum_x,
            inv_n: if n > 0.0 { 1.0 / n } else { 0.0 },
            inv_denom: if denom.abs() > f64::EPSILON {
                1.0 / denom
            } else {
                0.0
            },
            mean_x: if n > 0.0 { sum_x / n } else { 0.0 },
            forecast_x: period as f64,
            sum_y: 0.0,
            sum_xy: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.head = 0;
        self.count = 0;
        self.filled = false;
        self.sum_y = 0.0;
        self.sum_xy = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        self.update_clean(value)
    }

    #[inline(always)]
    fn update_clean(&mut self, value: f64) -> Option<(f64, f64)> {
        if self.period == 1 {
            self.buffer[0] = value;
            self.count = 1;
            self.filled = true;
            return Some((value, 0.0));
        }

        if !self.filled {
            let j = self.count as f64;
            self.buffer[self.head] = value;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
            self.sum_y += value;
            self.sum_xy += j * value;
            self.count += 1;
            if self.count < self.period {
                return None;
            }
            self.filled = true;
            return Some((self.forecast_next(), self.slope()));
        }

        let y_old = self.buffer[self.head];
        self.buffer[self.head] = value;
        let new_sum_y = self.sum_y + value - y_old;
        let new_sum_xy = self.n * value + self.sum_xy - new_sum_y;
        self.sum_y = new_sum_y;
        self.sum_xy = new_sum_xy;
        self.head += 1;
        if self.head == self.period {
            self.head = 0;
        }
        Some((self.forecast_next(), self.slope()))
    }

    #[inline(always)]
    fn slope(&self) -> f64 {
        if self.period <= 1 {
            return 0.0;
        }
        (self.n.mul_add(self.sum_xy, -self.sum_x * self.sum_y)) * self.inv_denom
    }

    #[inline(always)]
    fn forecast_next(&self) -> f64 {
        if self.period == 1 {
            return self.buffer[(self.head + self.period - 1) % self.period];
        }
        let slope = self.slope();
        let mean_y = self.sum_y * self.inv_n;
        mean_y + slope * (self.forecast_x - self.mean_x)
    }
}

#[derive(Clone, Debug)]
struct RollingMeanStdState {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    count: usize,
    filled: bool,
    sum: f64,
    sum_sq: f64,
}

impl RollingMeanStdState {
    #[inline(always)]
    fn new(period: usize) -> Self {
        Self {
            period,
            buffer: vec![0.0; period.max(1)],
            head: 0,
            count: 0,
            filled: false,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.head = 0;
        self.count = 0;
        self.filled = false;
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }

    #[inline(always)]
    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        self.update_clean(value)
    }

    #[inline(always)]
    fn update_clean(&mut self, value: f64) -> Option<(f64, f64)> {
        if !self.filled {
            self.buffer[self.head] = value;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
            self.count += 1;
            self.sum += value;
            self.sum_sq += value * value;
            if self.count < self.period {
                return None;
            }
            self.filled = true;
        } else {
            let old = self.buffer[self.head];
            self.buffer[self.head] = value;
            self.head += 1;
            if self.head == self.period {
                self.head = 0;
            }
            self.sum += value - old;
            self.sum_sq += value * value - old * old;
        }

        let n = self.period as f64;
        let mean = self.sum / n;
        let variance = (self.sum_sq / n - mean * mean).max(0.0);
        Some((mean, variance.sqrt()))
    }
}

#[derive(Clone, Debug)]
pub struct LeavittConvolutionAccelerationStream {
    source_projection: RollingLinRegState,
    projection_slope: RollingLinRegState,
    norm: RollingMeanStdState,
    use_norm_hyperbolic: bool,
    prev_scaled: f64,
    prev_conv_acceleration: f64,
    prev_slo: f64,
    prev_src1: f64,
    prev_src2: f64,
    have_src1: bool,
    have_src2: bool,
}

impl LeavittConvolutionAccelerationStream {
    pub fn try_new(
        params: LeavittConvolutionAccelerationParams,
    ) -> Result<Self, LeavittConvolutionAccelerationError> {
        let length = params.length.unwrap_or(DEFAULT_LENGTH);
        let norm_length = params.norm_length.unwrap_or(DEFAULT_NORM_LENGTH);
        if length == 0 {
            return Err(LeavittConvolutionAccelerationError::InvalidLength {
                length,
                data_len: 0,
            });
        }
        if norm_length == 0 {
            return Err(LeavittConvolutionAccelerationError::InvalidNormLength {
                norm_length,
                data_len: 0,
            });
        }
        Ok(Self {
            source_projection: RollingLinRegState::new(length),
            projection_slope: RollingLinRegState::new(sqrt_length(length)),
            norm: RollingMeanStdState::new(norm_length),
            use_norm_hyperbolic: params.use_norm_hyperbolic.unwrap_or(true),
            prev_scaled: 0.0,
            prev_conv_acceleration: 0.0,
            prev_slo: 0.0,
            prev_src1: 0.0,
            prev_src2: 0.0,
            have_src1: false,
            have_src2: false,
        })
    }

    #[inline(always)]
    fn logistic(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    #[inline(always)]
    fn hyperbolic(z: f64) -> f64 {
        let e = (-z).exp();
        (1.0 - e) / (1.0 + e)
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.source_projection.reset();
        self.projection_slope.reset();
        self.norm.reset();
        self.prev_scaled = 0.0;
        self.prev_conv_acceleration = 0.0;
        self.prev_slo = 0.0;
        self.prev_src1 = 0.0;
        self.prev_src2 = 0.0;
        self.have_src1 = false;
        self.have_src2 = false;
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        self.update_clean(value)
    }

    #[inline(always)]
    pub fn update_clean(&mut self, value: f64) -> Option<(f64, f64)> {
        let src1 = if self.have_src1 { self.prev_src1 } else { 0.0 };
        let src2 = if self.have_src2 { self.prev_src2 } else { 0.0 };
        let is_accelerated = src2 - 2.0 * src1 + value > 0.0;

        let projection = match self.source_projection.update_clean(value) {
            Some((forecast, _)) if forecast.is_finite() => forecast,
            Some(_) => {
                self.projection_slope.reset();
                self.norm.reset();
                self.prev_scaled = 0.0;
                self.prev_conv_acceleration = 0.0;
                self.prev_slo = 0.0;
                self.bump_source_history(value);
                return None;
            }
            None => {
                self.bump_source_history(value);
                return None;
            }
        };

        let conv_slope = match self.projection_slope.update_clean(projection) {
            Some((_, slope)) if slope.is_finite() => slope,
            Some(_) => {
                self.norm.reset();
                self.prev_scaled = 0.0;
                self.prev_conv_acceleration = 0.0;
                self.prev_slo = 0.0;
                self.bump_source_history(value);
                return None;
            }
            None => {
                self.bump_source_history(value);
                return None;
            }
        };

        let scaled = match self.norm.update_clean(conv_slope) {
            Some((mean, dev)) => {
                let z = if dev != 0.0 {
                    (conv_slope - mean) / dev
                } else {
                    0.0
                };
                if self.use_norm_hyperbolic {
                    Self::hyperbolic(z)
                } else {
                    Self::logistic(z)
                }
            }
            None => {
                self.bump_source_history(value);
                return None;
            }
        };

        let conv_acceleration = scaled - self.prev_scaled;
        let slo = if self.use_norm_hyperbolic {
            conv_acceleration
        } else {
            conv_acceleration - self.prev_conv_acceleration
        };
        let signal = if slo > 0.0 && is_accelerated {
            if slo > self.prev_slo {
                2.0
            } else {
                1.0
            }
        } else if slo < 0.0 && !is_accelerated {
            if slo < self.prev_slo {
                -2.0
            } else {
                -1.0
            }
        } else {
            0.0
        };

        self.prev_scaled = scaled;
        self.prev_conv_acceleration = conv_acceleration;
        self.prev_slo = slo;
        self.bump_source_history(value);
        Some((conv_acceleration, signal))
    }

    #[inline(always)]
    pub fn update_reset_on_nan(&mut self, value: f64) -> Option<(f64, f64)> {
        self.update(value)
    }

    #[inline(always)]
    fn bump_source_history(&mut self, value: f64) {
        if self.have_src1 {
            self.prev_src2 = self.prev_src1;
            self.have_src2 = true;
        }
        self.prev_src1 = value;
        self.have_src1 = true;
    }
}

#[inline(always)]
fn leavitt_convolution_acceleration_prepare<'a>(
    input: &'a LeavittConvolutionAccelerationInput,
) -> Result<(&'a [f64], usize, usize, bool, usize), LeavittConvolutionAccelerationError> {
    let data = input.as_ref();
    let data_len = data.len();
    if data_len == 0 {
        return Err(LeavittConvolutionAccelerationError::EmptyInputData);
    }
    let first =
        first_valid_source(data).ok_or(LeavittConvolutionAccelerationError::AllValuesNaN)?;
    let length = input.get_length();
    if length == 0 || length > data_len {
        return Err(LeavittConvolutionAccelerationError::InvalidLength { length, data_len });
    }
    let norm_length = input.get_norm_length();
    if norm_length == 0 || norm_length > data_len {
        return Err(LeavittConvolutionAccelerationError::InvalidNormLength {
            norm_length,
            data_len,
        });
    }
    let needed = required_valid_bars(length, norm_length);
    let valid = count_valid_from(data, first);
    if valid < needed {
        return Err(LeavittConvolutionAccelerationError::NotEnoughValidData { needed, valid });
    }
    Ok((
        data,
        length,
        norm_length,
        input.get_use_norm_hyperbolic(),
        first,
    ))
}

#[inline(always)]
fn leavitt_convolution_acceleration_compute_into(
    data: &[f64],
    length: usize,
    norm_length: usize,
    use_norm_hyperbolic: bool,
    first: usize,
    _kernel: Kernel,
    out_conv_acceleration: &mut [f64],
    out_signal: &mut [f64],
) {
    if leavitt_convolution_acceleration_compute_clean(
        data,
        length,
        norm_length,
        use_norm_hyperbolic,
        first,
        out_conv_acceleration,
        out_signal,
    ) {
        return;
    }

    let mut stream =
        LeavittConvolutionAccelerationStream::try_new(LeavittConvolutionAccelerationParams {
            length: Some(length),
            norm_length: Some(norm_length),
            use_norm_hyperbolic: Some(use_norm_hyperbolic),
        })
        .expect("validated stream params");

    for i in 0..data.len() {
        if let Some((conv_acceleration, signal)) = stream.update_reset_on_nan(data[i]) {
            out_conv_acceleration[i] = conv_acceleration;
            out_signal[i] = signal;
        } else {
            out_conv_acceleration[i] = f64::NAN;
            out_signal[i] = f64::NAN;
        }
    }
}

#[inline(always)]
fn leavitt_convolution_acceleration_compute_clean(
    data: &[f64],
    length: usize,
    norm_length: usize,
    use_norm_hyperbolic: bool,
    first: usize,
    out_conv_acceleration: &mut [f64],
    out_signal: &mut [f64],
) -> bool {
    for value in &mut out_conv_acceleration[..first] {
        *value = f64::NAN;
    }
    for value in &mut out_signal[..first] {
        *value = f64::NAN;
    }

    let mut stream =
        LeavittConvolutionAccelerationStream::try_new(LeavittConvolutionAccelerationParams {
            length: Some(length),
            norm_length: Some(norm_length),
            use_norm_hyperbolic: Some(use_norm_hyperbolic),
        })
        .expect("validated stream params");

    for i in first..data.len() {
        let value = data[i];
        if !value.is_finite() {
            return false;
        }
        if let Some((conv_acceleration, signal)) = stream.update_clean(value) {
            out_conv_acceleration[i] = conv_acceleration;
            out_signal[i] = signal;
        } else {
            out_conv_acceleration[i] = f64::NAN;
            out_signal[i] = f64::NAN;
        }
    }
    true
}

#[inline(always)]
fn alloc_leavitt_output(len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    unsafe {
        out.set_len(len);
    }
    out
}

#[inline]
pub fn leavitt_convolution_acceleration(
    input: &LeavittConvolutionAccelerationInput,
) -> Result<LeavittConvolutionAccelerationOutput, LeavittConvolutionAccelerationError> {
    leavitt_convolution_acceleration_with_kernel(input, Kernel::Auto)
}

pub fn leavitt_convolution_acceleration_with_kernel(
    input: &LeavittConvolutionAccelerationInput,
    kernel: Kernel,
) -> Result<LeavittConvolutionAccelerationOutput, LeavittConvolutionAccelerationError> {
    let (data, length, norm_length, use_norm_hyperbolic, first) =
        leavitt_convolution_acceleration_prepare(input)?;
    let mut conv_acceleration = alloc_leavitt_output(data.len());
    let mut signal = alloc_leavitt_output(data.len());
    leavitt_convolution_acceleration_compute_into(
        data,
        length,
        norm_length,
        use_norm_hyperbolic,
        first,
        normalized_kernel(kernel),
        &mut conv_acceleration,
        &mut signal,
    );
    Ok(LeavittConvolutionAccelerationOutput {
        conv_acceleration,
        signal,
    })
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[inline]
pub fn leavitt_convolution_acceleration_into(
    input: &LeavittConvolutionAccelerationInput,
    out_conv_acceleration: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), LeavittConvolutionAccelerationError> {
    leavitt_convolution_acceleration_into_slice(
        out_conv_acceleration,
        out_signal,
        input,
        Kernel::Auto,
    )
}

pub fn leavitt_convolution_acceleration_into_slice(
    out_conv_acceleration: &mut [f64],
    out_signal: &mut [f64],
    input: &LeavittConvolutionAccelerationInput,
    kernel: Kernel,
) -> Result<(), LeavittConvolutionAccelerationError> {
    let (data, length, norm_length, use_norm_hyperbolic, _first) =
        leavitt_convolution_acceleration_prepare(input)?;
    if out_conv_acceleration.len() != data.len() || out_signal.len() != data.len() {
        return Err(LeavittConvolutionAccelerationError::OutputLengthMismatch {
            expected: data.len(),
            got: out_conv_acceleration.len().max(out_signal.len()),
        });
    }
    leavitt_convolution_acceleration_compute_into(
        data,
        length,
        norm_length,
        use_norm_hyperbolic,
        _first,
        normalized_kernel(kernel),
        out_conv_acceleration,
        out_signal,
    );
    Ok(())
}

#[derive(Clone, Debug)]
pub struct LeavittConvolutionAccelerationBatchRange {
    pub length: (usize, usize, usize),
    pub norm_length: (usize, usize, usize),
    pub use_norm_hyperbolic: Option<bool>,
}

impl Default for LeavittConvolutionAccelerationBatchRange {
    fn default() -> Self {
        Self {
            length: (DEFAULT_LENGTH, DEFAULT_LENGTH, 0),
            norm_length: (DEFAULT_NORM_LENGTH, DEFAULT_NORM_LENGTH, 0),
            use_norm_hyperbolic: Some(true),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LeavittConvolutionAccelerationBatchBuilder {
    range: LeavittConvolutionAccelerationBatchRange,
    kernel: Kernel,
}

impl LeavittConvolutionAccelerationBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, value: Kernel) -> Self {
        self.kernel = value;
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
    pub fn norm_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.norm_length = (start, end, step);
        self
    }

    #[inline]
    pub fn norm_length_static(mut self, value: usize) -> Self {
        self.range.norm_length = (value, value, 0);
        self
    }

    #[inline]
    pub fn use_norm_hyperbolic(mut self, value: bool) -> Self {
        self.range.use_norm_hyperbolic = Some(value);
        self
    }

    #[inline]
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<LeavittConvolutionAccelerationBatchOutput, LeavittConvolutionAccelerationError>
    {
        leavitt_convolution_acceleration_batch_with_kernel(data, &self.range, self.kernel)
    }

    #[inline]
    pub fn apply_candles(
        self,
        candles: &Candles,
        source: &str,
    ) -> Result<LeavittConvolutionAccelerationBatchOutput, LeavittConvolutionAccelerationError>
    {
        self.apply_slice(source_type(candles, source))
    }
}

#[derive(Debug, Clone)]
pub struct LeavittConvolutionAccelerationBatchOutput {
    pub conv_acceleration: Vec<f64>,
    pub signal: Vec<f64>,
    pub combos: Vec<LeavittConvolutionAccelerationParams>,
    pub rows: usize,
    pub cols: usize,
}

#[inline(always)]
fn expand_usize_axis(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, LeavittConvolutionAccelerationError> {
    if start == 0 || end == 0 {
        return Err(LeavittConvolutionAccelerationError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0 {
        if start != end {
            return Err(LeavittConvolutionAccelerationError::InvalidRange {
                start: start.to_string(),
                end: end.to_string(),
                step: step.to_string(),
            });
        }
        return Ok(vec![start]);
    }
    if start > end {
        return Err(LeavittConvolutionAccelerationError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut value = start;
    while value <= end {
        out.push(value);
        match value.checked_add(step) {
            Some(next) => value = next,
            None => break,
        }
    }
    if out.is_empty() {
        return Err(LeavittConvolutionAccelerationError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    Ok(out)
}

pub fn expand_grid_leavitt_convolution_acceleration(
    sweep: &LeavittConvolutionAccelerationBatchRange,
) -> Result<Vec<LeavittConvolutionAccelerationParams>, LeavittConvolutionAccelerationError> {
    let lengths = expand_usize_axis(sweep.length.0, sweep.length.1, sweep.length.2)?;
    let norm_lengths = expand_usize_axis(
        sweep.norm_length.0,
        sweep.norm_length.1,
        sweep.norm_length.2,
    )?;
    let use_norm_hyperbolic = sweep.use_norm_hyperbolic.unwrap_or(true);

    let total = lengths
        .len()
        .checked_mul(norm_lengths.len())
        .ok_or_else(|| LeavittConvolutionAccelerationError::InvalidRange {
            start: sweep.length.0.to_string(),
            end: sweep.norm_length.1.to_string(),
            step: "overflow".to_string(),
        })?;
    let mut combos = Vec::with_capacity(total);
    for &length in &lengths {
        for &norm_length in &norm_lengths {
            combos.push(LeavittConvolutionAccelerationParams {
                length: Some(length),
                norm_length: Some(norm_length),
                use_norm_hyperbolic: Some(use_norm_hyperbolic),
            });
        }
    }
    Ok(combos)
}

pub fn leavitt_convolution_acceleration_batch_with_kernel(
    data: &[f64],
    sweep: &LeavittConvolutionAccelerationBatchRange,
    kernel: Kernel,
) -> Result<LeavittConvolutionAccelerationBatchOutput, LeavittConvolutionAccelerationError> {
    let chosen = match kernel {
        Kernel::Auto => Kernel::ScalarBatch,
        other => other,
    };
    match chosen {
        Kernel::Scalar | Kernel::ScalarBatch => {
            leavitt_convolution_acceleration_batch_par_slice(data, sweep)
        }
        other => Err(LeavittConvolutionAccelerationError::InvalidKernelForBatch(
            other,
        )),
    }
}

pub fn leavitt_convolution_acceleration_batch_slice(
    data: &[f64],
    sweep: &LeavittConvolutionAccelerationBatchRange,
) -> Result<LeavittConvolutionAccelerationBatchOutput, LeavittConvolutionAccelerationError> {
    leavitt_convolution_acceleration_batch_impl(data, sweep, Kernel::Scalar, false)
}

pub fn leavitt_convolution_acceleration_batch_par_slice(
    data: &[f64],
    sweep: &LeavittConvolutionAccelerationBatchRange,
) -> Result<LeavittConvolutionAccelerationBatchOutput, LeavittConvolutionAccelerationError> {
    leavitt_convolution_acceleration_batch_impl(data, sweep, Kernel::Scalar, true)
}

fn leavitt_convolution_acceleration_batch_impl(
    data: &[f64],
    sweep: &LeavittConvolutionAccelerationBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<LeavittConvolutionAccelerationBatchOutput, LeavittConvolutionAccelerationError> {
    let combos = expand_grid_leavitt_convolution_acceleration(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if cols == 0 {
        return Err(LeavittConvolutionAccelerationError::EmptyInputData);
    }
    let first =
        first_valid_source(data).ok_or(LeavittConvolutionAccelerationError::AllValuesNaN)?;
    let valid = count_valid_from(data, first);
    for params in &combos {
        let needed = required_valid_bars(
            params.length.unwrap_or(DEFAULT_LENGTH),
            params.norm_length.unwrap_or(DEFAULT_NORM_LENGTH),
        );
        if valid < needed {
            return Err(LeavittConvolutionAccelerationError::NotEnoughValidData { needed, valid });
        }
    }

    let mut conv_matrix = make_uninit_matrix(rows, cols);
    let mut signal_matrix = make_uninit_matrix(rows, cols);

    let mut conv_guard = ManuallyDrop::new(conv_matrix);
    let mut signal_guard = ManuallyDrop::new(signal_matrix);
    let conv_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(conv_guard.as_mut_ptr(), conv_guard.len()) };
    let signal_mu: &mut [MaybeUninit<f64>] =
        unsafe { std::slice::from_raw_parts_mut(signal_guard.as_mut_ptr(), signal_guard.len()) };

    let do_row = |row: usize,
                  row_conv_mu: &mut [MaybeUninit<f64>],
                  row_signal_mu: &mut [MaybeUninit<f64>]| {
        let params = &combos[row];
        let out_conv = unsafe {
            std::slice::from_raw_parts_mut(row_conv_mu.as_mut_ptr() as *mut f64, row_conv_mu.len())
        };
        let out_signal = unsafe {
            std::slice::from_raw_parts_mut(
                row_signal_mu.as_mut_ptr() as *mut f64,
                row_signal_mu.len(),
            )
        };
        leavitt_convolution_acceleration_compute_into(
            data,
            params.length.unwrap_or(DEFAULT_LENGTH),
            params.norm_length.unwrap_or(DEFAULT_NORM_LENGTH),
            params.use_norm_hyperbolic.unwrap_or(true),
            first,
            kernel,
            out_conv,
            out_signal,
        );
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        conv_mu
            .par_chunks_mut(cols)
            .zip(signal_mu.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (row_conv, row_signal))| do_row(row, row_conv, row_signal));
        #[cfg(target_arch = "wasm32")]
        for (row, (row_conv, row_signal)) in conv_mu
            .chunks_mut(cols)
            .zip(signal_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_conv, row_signal);
        }
    } else {
        for (row, (row_conv, row_signal)) in conv_mu
            .chunks_mut(cols)
            .zip(signal_mu.chunks_mut(cols))
            .enumerate()
        {
            do_row(row, row_conv, row_signal);
        }
    }

    let conv_acceleration = unsafe {
        Vec::from_raw_parts(
            conv_guard.as_mut_ptr() as *mut f64,
            conv_guard.len(),
            conv_guard.capacity(),
        )
    };
    let signal = unsafe {
        Vec::from_raw_parts(
            signal_guard.as_mut_ptr() as *mut f64,
            signal_guard.len(),
            signal_guard.capacity(),
        )
    };

    Ok(LeavittConvolutionAccelerationBatchOutput {
        conv_acceleration,
        signal,
        combos,
        rows,
        cols,
    })
}

fn leavitt_convolution_acceleration_batch_inner_into(
    data: &[f64],
    sweep: &LeavittConvolutionAccelerationBatchRange,
    kernel: Kernel,
    parallel: bool,
    out_conv_acceleration: &mut [f64],
    out_signal: &mut [f64],
) -> Result<(), LeavittConvolutionAccelerationError> {
    let combos = expand_grid_leavitt_convolution_acceleration(sweep)?;
    let rows = combos.len();
    let cols = data.len();
    if out_conv_acceleration.len() != rows * cols || out_signal.len() != rows * cols {
        return Err(LeavittConvolutionAccelerationError::OutputLengthMismatch {
            expected: rows * cols,
            got: out_conv_acceleration.len().max(out_signal.len()),
        });
    }
    let first = first_valid_source(data).unwrap_or(0);

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let ptr_conv = out_conv_acceleration.as_mut_ptr() as usize;
            let ptr_signal = out_signal.as_mut_ptr() as usize;
            combos.par_iter().enumerate().for_each(|(row, params)| {
                let start = row * cols;
                let out_conv = unsafe {
                    std::slice::from_raw_parts_mut((ptr_conv as *mut f64).add(start), cols)
                };
                let out_sig = unsafe {
                    std::slice::from_raw_parts_mut((ptr_signal as *mut f64).add(start), cols)
                };
                leavitt_convolution_acceleration_compute_into(
                    data,
                    params.length.unwrap_or(DEFAULT_LENGTH),
                    params.norm_length.unwrap_or(DEFAULT_NORM_LENGTH),
                    params.use_norm_hyperbolic.unwrap_or(true),
                    first,
                    kernel,
                    out_conv,
                    out_sig,
                );
            });
        }
        #[cfg(target_arch = "wasm32")]
        for (row, params) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            leavitt_convolution_acceleration_compute_into(
                data,
                params.length.unwrap_or(DEFAULT_LENGTH),
                params.norm_length.unwrap_or(DEFAULT_NORM_LENGTH),
                params.use_norm_hyperbolic.unwrap_or(true),
                first,
                kernel,
                &mut out_conv_acceleration[start..end],
                &mut out_signal[start..end],
            );
        }
    } else {
        for (row, params) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            leavitt_convolution_acceleration_compute_into(
                data,
                params.length.unwrap_or(DEFAULT_LENGTH),
                params.norm_length.unwrap_or(DEFAULT_NORM_LENGTH),
                params.use_norm_hyperbolic.unwrap_or(true),
                first,
                kernel,
                &mut out_conv_acceleration[start..end],
                &mut out_signal[start..end],
            );
        }
    }

    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "leavitt_convolution_acceleration")]
#[pyo3(signature = (data, length=DEFAULT_LENGTH, norm_length=DEFAULT_NORM_LENGTH, use_norm_hyperbolic=true, kernel=None))]
pub fn leavitt_convolution_acceleration_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    norm_length: usize,
    use_norm_hyperbolic: bool,
    kernel: Option<&str>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let data = data.as_slice()?;
    let kernel = validate_kernel(kernel, false)?;
    let input = LeavittConvolutionAccelerationInput::from_slice(
        data,
        LeavittConvolutionAccelerationParams {
            length: Some(length),
            norm_length: Some(norm_length),
            use_norm_hyperbolic: Some(use_norm_hyperbolic),
        },
    );
    let output = py
        .allow_threads(|| leavitt_convolution_acceleration_with_kernel(&input, kernel))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        output.conv_acceleration.into_pyarray(py),
        output.signal.into_pyarray(py),
    ))
}

#[cfg(feature = "python")]
#[pyclass(name = "LeavittConvolutionAccelerationStream")]
pub struct LeavittConvolutionAccelerationStreamPy {
    stream: LeavittConvolutionAccelerationStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl LeavittConvolutionAccelerationStreamPy {
    #[new]
    #[pyo3(signature = (length=DEFAULT_LENGTH, norm_length=DEFAULT_NORM_LENGTH, use_norm_hyperbolic=true))]
    fn new(length: usize, norm_length: usize, use_norm_hyperbolic: bool) -> PyResult<Self> {
        let stream =
            LeavittConvolutionAccelerationStream::try_new(LeavittConvolutionAccelerationParams {
                length: Some(length),
                norm_length: Some(norm_length),
                use_norm_hyperbolic: Some(use_norm_hyperbolic),
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { stream })
    }

    fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "leavitt_convolution_acceleration_batch")]
#[pyo3(signature = (data, length_range, norm_length_range, use_norm_hyperbolic=true, kernel=None))]
pub fn leavitt_convolution_acceleration_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    norm_length_range: (usize, usize, usize),
    use_norm_hyperbolic: bool,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let data = data.as_slice()?;
    let sweep = LeavittConvolutionAccelerationBatchRange {
        length: length_range,
        norm_length: norm_length_range,
        use_norm_hyperbolic: Some(use_norm_hyperbolic),
    };
    let combos = expand_grid_leavitt_convolution_acceleration(&sweep)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow"))?;
    let conv_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let signal_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let out_conv = unsafe { conv_arr.as_slice_mut()? };
    let out_signal = unsafe { signal_arr.as_slice_mut()? };
    let kernel = validate_kernel(kernel, true)?;

    py.allow_threads(|| {
        let batch_kernel = match kernel {
            Kernel::Auto => detect_best_batch_kernel(),
            other => other,
        };
        leavitt_convolution_acceleration_batch_inner_into(
            data,
            &sweep,
            batch_kernel.to_non_batch(),
            true,
            out_conv,
            out_signal,
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let lengths: Vec<usize> = combos
        .iter()
        .map(|p| p.length.unwrap_or(DEFAULT_LENGTH))
        .collect();
    let norm_lengths: Vec<usize> = combos
        .iter()
        .map(|p| p.norm_length.unwrap_or(DEFAULT_NORM_LENGTH))
        .collect();
    let dict = PyDict::new(py);
    dict.set_item("conv_acceleration", conv_arr.reshape((rows, cols))?)?;
    dict.set_item("signal", signal_arr.reshape((rows, cols))?)?;
    dict.set_item("rows", rows)?;
    dict.set_item("cols", cols)?;
    dict.set_item("lengths", lengths.into_pyarray(py))?;
    dict.set_item("norm_lengths", norm_lengths.into_pyarray(py))?;
    dict.set_item("use_norm_hyperbolic", use_norm_hyperbolic)?;
    Ok(dict)
}

#[cfg(feature = "python")]
pub fn register_leavitt_convolution_acceleration_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(leavitt_convolution_acceleration_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        leavitt_convolution_acceleration_batch_py,
        m
    )?)?;
    m.add_class::<LeavittConvolutionAccelerationStreamPy>()?;
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LeavittConvolutionAccelerationJsOutput {
    conv_acceleration: Vec<f64>,
    signal: Vec<f64>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LeavittConvolutionAccelerationBatchConfig {
    length_range: Vec<usize>,
    norm_length_range: Vec<usize>,
    use_norm_hyperbolic: Option<bool>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LeavittConvolutionAccelerationBatchJsOutput {
    conv_acceleration: Vec<f64>,
    signal: Vec<f64>,
    rows: usize,
    cols: usize,
    combos: Vec<LeavittConvolutionAccelerationParams>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn leavitt_convolution_acceleration_js(
    data: &[f64],
    length: usize,
    norm_length: usize,
    use_norm_hyperbolic: bool,
) -> Result<JsValue, JsValue> {
    let input = LeavittConvolutionAccelerationInput::from_slice(
        data,
        LeavittConvolutionAccelerationParams {
            length: Some(length),
            norm_length: Some(norm_length),
            use_norm_hyperbolic: Some(use_norm_hyperbolic),
        },
    );
    let output = leavitt_convolution_acceleration_with_kernel(&input, Kernel::Scalar)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&LeavittConvolutionAccelerationJsOutput {
        conv_acceleration: output.conv_acceleration,
        signal: output.signal,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "leavitt_convolution_acceleration_batch_js")]
pub fn leavitt_convolution_acceleration_batch_js(
    data: &[f64],
    config: JsValue,
) -> Result<JsValue, JsValue> {
    let config: LeavittConvolutionAccelerationBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 || config.norm_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: each range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = LeavittConvolutionAccelerationBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
        norm_length: (
            config.norm_length_range[0],
            config.norm_length_range[1],
            config.norm_length_range[2],
        ),
        use_norm_hyperbolic: config.use_norm_hyperbolic,
    };
    let batch = leavitt_convolution_acceleration_batch_slice(data, &sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&LeavittConvolutionAccelerationBatchJsOutput {
        conv_acceleration: batch.conv_acceleration,
        signal: batch.signal,
        rows: batch.rows,
        cols: batch.cols,
        combos: batch.combos,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn leavitt_convolution_acceleration_alloc(len: usize) -> *mut f64 {
    let mut buf = vec![0.0_f64; len * 2];
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn leavitt_convolution_acceleration_free(ptr: *mut f64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len * 2);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn leavitt_convolution_acceleration_into(
    data_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    norm_length: usize,
    use_norm_hyperbolic: bool,
) -> Result<(), JsValue> {
    if data_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to leavitt_convolution_acceleration_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(data_ptr, len);
        let out = std::slice::from_raw_parts_mut(out_ptr, len * 2);
        let (out_conv, out_signal) = out.split_at_mut(len);
        let input = LeavittConvolutionAccelerationInput::from_slice(
            data,
            LeavittConvolutionAccelerationParams {
                length: Some(length),
                norm_length: Some(norm_length),
                use_norm_hyperbolic: Some(use_norm_hyperbolic),
            },
        );
        leavitt_convolution_acceleration_into_slice(out_conv, out_signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = "leavitt_convolution_acceleration_into_host")]
pub fn leavitt_convolution_acceleration_into_host(
    data: &[f64],
    out_ptr: *mut f64,
    length: usize,
    norm_length: usize,
    use_norm_hyperbolic: bool,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to leavitt_convolution_acceleration_into_host",
        ));
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut(out_ptr, data.len() * 2);
        let (out_conv, out_signal) = out.split_at_mut(data.len());
        let input = LeavittConvolutionAccelerationInput::from_slice(
            data,
            LeavittConvolutionAccelerationParams {
                length: Some(length),
                norm_length: Some(norm_length),
                use_norm_hyperbolic: Some(use_norm_hyperbolic),
            },
        );
        leavitt_convolution_acceleration_into_slice(out_conv, out_signal, &input, Kernel::Auto)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn leavitt_convolution_acceleration_batch_into(
    data: &[f64],
    out_ptr: *mut f64,
    config: JsValue,
) -> Result<(), JsValue> {
    if out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to leavitt_convolution_acceleration_batch_into",
        ));
    }
    let config: LeavittConvolutionAccelerationBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {e}")))?;
    if config.length_range.len() != 3 || config.norm_length_range.len() != 3 {
        return Err(JsValue::from_str(
            "Invalid config: each range must have exactly 3 elements [start, end, step]",
        ));
    }
    let sweep = LeavittConvolutionAccelerationBatchRange {
        length: (
            config.length_range[0],
            config.length_range[1],
            config.length_range[2],
        ),
        norm_length: (
            config.norm_length_range[0],
            config.norm_length_range[1],
            config.norm_length_range[2],
        ),
        use_norm_hyperbolic: config.use_norm_hyperbolic,
    };
    let combos = expand_grid_leavitt_convolution_acceleration(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, expected) };
    let (out_conv, out_signal) = out.split_at_mut(rows * cols);
    leavitt_convolution_acceleration_batch_inner_into(
        data,
        &sweep,
        Kernel::Scalar,
        false,
        out_conv,
        out_signal,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn leavitt_convolution_acceleration_output_into_js(
    data: &[f64],
    length: usize,
    norm_length: usize,
    use_norm_hyperbolic: bool,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value =
        leavitt_convolution_acceleration_js(data, length, norm_length, use_norm_hyperbolic)?;
    crate::write_wasm_object_f64_outputs(
        "leavitt_convolution_acceleration_output_into_js",
        &value,
        out,
    )
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn leavitt_convolution_acceleration_batch_output_into_js(
    data: &[f64],
    config: JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let value = leavitt_convolution_acceleration_batch_js(data, config)?;
    crate::write_wasm_selected_object_f64_outputs(
        "leavitt_convolution_acceleration_batch_output_into_js",
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

    fn sample_data(len: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let x = i as f64;
            out.push(100.0 + x * 0.04 + (x * 0.13).sin() * 2.7 + (x * 0.07).cos() * 1.1);
        }
        out
    }

    fn assert_close_nan(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (idx, (&lhs, &rhs)) in a.iter().zip(b.iter()).enumerate() {
            if lhs.is_nan() || rhs.is_nan() {
                assert!(
                    lhs.is_nan() && rhs.is_nan(),
                    "nan mismatch at {idx}: {lhs} vs {rhs}"
                );
            } else {
                assert!(
                    (lhs - rhs).abs() <= 1e-10,
                    "value mismatch at {idx}: {lhs} vs {rhs}"
                );
            }
        }
    }

    fn linreg_value(window: &[f64], offset: isize) -> f64 {
        let n = window.len();
        if n == 1 {
            return window[0];
        }
        let nf = n as f64;
        let mean_x = (n - 1) as f64 / 2.0;
        let mean_y = window.iter().sum::<f64>() / nf;
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &y) in window.iter().enumerate() {
            let x = i as f64;
            num += (x - mean_x) * (y - mean_y);
            den += (x - mean_x) * (x - mean_x);
        }
        let slope = if den != 0.0 { num / den } else { 0.0 };
        let intercept = mean_y - slope * mean_x;
        intercept + slope * ((n - 1) as isize - offset) as f64
    }

    fn naive_expected(
        data: &[f64],
        length: usize,
        norm_length: usize,
        use_norm_hyperbolic: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        let sqrt_len = sqrt_length(length);
        let mut projection = vec![f64::NAN; n];
        let mut conv_slope = vec![f64::NAN; n];
        let mut conv_acceleration = vec![f64::NAN; n];
        let mut signal = vec![f64::NAN; n];

        for i in 0..n {
            if i + 1 >= length {
                let start = i + 1 - length;
                projection[i] = linreg_value(&data[start..=i], -1);
            }
        }

        for i in 0..n {
            if i + 1 >= sqrt_len {
                let start = i + 1 - sqrt_len;
                let window = &projection[start..=i];
                if window.iter().all(|v| v.is_finite()) {
                    let curr = linreg_value(window, 0);
                    let prev = linreg_value(window, 1);
                    conv_slope[i] = curr - prev;
                }
            }
        }

        let mut prev_scaled = 0.0;
        let mut prev_conv = 0.0;
        let mut prev_slo = 0.0;
        for i in 0..n {
            if i + 1 >= norm_length {
                let start = i + 1 - norm_length;
                let window = &conv_slope[start..=i];
                if window.iter().all(|v| v.is_finite()) {
                    let mean = window.iter().sum::<f64>() / norm_length as f64;
                    let variance = window
                        .iter()
                        .map(|v| {
                            let d = *v - mean;
                            d * d
                        })
                        .sum::<f64>()
                        / norm_length as f64;
                    let dev = variance.sqrt();
                    let z = if dev != 0.0 {
                        (conv_slope[i] - mean) / dev
                    } else {
                        0.0
                    };
                    let scaled = if use_norm_hyperbolic {
                        let e = (-z).exp();
                        (1.0 - e) / (1.0 + e)
                    } else {
                        1.0 / (1.0 + (-z).exp())
                    };
                    let accel = scaled - prev_scaled;
                    conv_acceleration[i] = accel;
                    let slo = if use_norm_hyperbolic {
                        accel
                    } else {
                        accel - prev_conv
                    };
                    let src1 = if i >= 1 { data[i - 1] } else { 0.0 };
                    let src2 = if i >= 2 { data[i - 2] } else { 0.0 };
                    let is_accelerated = src2 - 2.0 * src1 + data[i] > 0.0;
                    signal[i] = if slo > 0.0 && is_accelerated {
                        if slo > prev_slo {
                            2.0
                        } else {
                            1.0
                        }
                    } else if slo < 0.0 && !is_accelerated {
                        if slo < prev_slo {
                            -2.0
                        } else {
                            -1.0
                        }
                    } else {
                        0.0
                    };
                    prev_scaled = scaled;
                    prev_conv = accel;
                    prev_slo = slo;
                }
            }
        }

        (conv_acceleration, signal)
    }

    #[test]
    fn leavitt_convolution_acceleration_matches_naive() {
        let data = sample_data(320);
        let input = LeavittConvolutionAccelerationInput::from_slice(
            &data,
            LeavittConvolutionAccelerationParams {
                length: Some(21),
                norm_length: Some(34),
                use_norm_hyperbolic: Some(true),
            },
        );
        let out = leavitt_convolution_acceleration(&input).expect("indicator");
        let (expected_conv, expected_signal) = naive_expected(&data, 21, 34, true);
        assert_close_nan(&out.conv_acceleration, &expected_conv);
        assert_close_nan(&out.signal, &expected_signal);
    }

    #[test]
    fn leavitt_convolution_acceleration_into_matches_api() {
        let data = sample_data(240);
        let input = LeavittConvolutionAccelerationInput::from_slice(
            &data,
            LeavittConvolutionAccelerationParams {
                length: Some(14),
                norm_length: Some(28),
                use_norm_hyperbolic: Some(false),
            },
        );
        let baseline = leavitt_convolution_acceleration(&input).expect("baseline");
        let mut conv = vec![0.0; data.len()];
        let mut signal = vec![0.0; data.len()];
        leavitt_convolution_acceleration_into(&input, &mut conv, &mut signal).expect("into");
        assert_close_nan(&conv, &baseline.conv_acceleration);
        assert_close_nan(&signal, &baseline.signal);
    }

    #[test]
    fn leavitt_convolution_acceleration_stream_matches_batch() {
        let data = sample_data(256);
        let params = LeavittConvolutionAccelerationParams {
            length: Some(20),
            norm_length: Some(25),
            use_norm_hyperbolic: Some(true),
        };
        let batch = leavitt_convolution_acceleration(
            &LeavittConvolutionAccelerationInput::from_slice(&data, params.clone()),
        )
        .expect("batch");
        let mut stream = LeavittConvolutionAccelerationStream::try_new(params).expect("stream");
        let mut conv = vec![f64::NAN; data.len()];
        let mut sig = vec![f64::NAN; data.len()];
        for (i, &value) in data.iter().enumerate() {
            if let Some((a, b)) = stream.update_reset_on_nan(value) {
                conv[i] = a;
                sig[i] = b;
            }
        }
        assert_close_nan(&conv, &batch.conv_acceleration);
        assert_close_nan(&sig, &batch.signal);
    }

    #[test]
    fn leavitt_convolution_acceleration_batch_single_param_matches_single() {
        let data = sample_data(220);
        let batch = leavitt_convolution_acceleration_batch_with_kernel(
            &data,
            &LeavittConvolutionAccelerationBatchRange {
                length: (21, 21, 0),
                norm_length: (34, 34, 0),
                use_norm_hyperbolic: Some(false),
            },
            Kernel::ScalarBatch,
        )
        .expect("batch");
        let direct =
            leavitt_convolution_acceleration(&LeavittConvolutionAccelerationInput::from_slice(
                &data,
                LeavittConvolutionAccelerationParams {
                    length: Some(21),
                    norm_length: Some(34),
                    use_norm_hyperbolic: Some(false),
                },
            ))
            .expect("direct");
        assert_eq!(batch.rows, 1);
        assert_eq!(batch.cols, data.len());
        assert_close_nan(
            &batch.conv_acceleration[..data.len()],
            &direct.conv_acceleration,
        );
        assert_close_nan(&batch.signal[..data.len()], &direct.signal);
    }

    #[test]
    fn leavitt_convolution_acceleration_rejects_invalid_norm_length() {
        let data = sample_data(32);
        let input = LeavittConvolutionAccelerationInput::from_slice(
            &data,
            LeavittConvolutionAccelerationParams {
                length: Some(10),
                norm_length: Some(0),
                use_norm_hyperbolic: Some(true),
            },
        );
        let err = leavitt_convolution_acceleration(&input).expect_err("invalid");
        assert!(matches!(
            err,
            LeavittConvolutionAccelerationError::InvalidNormLength { .. }
        ));
    }

    #[test]
    fn leavitt_convolution_acceleration_dispatch_matches_direct() {
        let data = sample_data(280);
        let combo = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(21),
            },
            ParamKV {
                key: "norm_length",
                value: ParamValue::Int(34),
            },
            ParamKV {
                key: "use_norm_hyperbolic",
                value: ParamValue::Bool(true),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "leavitt_convolution_acceleration",
            output_id: Some("signal"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::ScalarBatch,
        };
        let out = compute_cpu_batch(req).expect("dispatch");
        let direct =
            leavitt_convolution_acceleration(&LeavittConvolutionAccelerationInput::from_slice(
                &data,
                LeavittConvolutionAccelerationParams {
                    length: Some(21),
                    norm_length: Some(34),
                    use_norm_hyperbolic: Some(true),
                },
            ))
            .expect("direct");
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, data.len());
        assert_close_nan(out.values_f64.as_ref().expect("values"), &direct.signal);
    }
}
