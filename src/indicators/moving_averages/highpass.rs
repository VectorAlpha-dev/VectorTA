//! # High-Pass Filter (HP)
//!
//! A digital filter that attenuates low-frequency components of the input data,
//! allowing higher-frequency fluctuations to pass through. This helps to remove
//! or reduce slow-moving trends or bias.
//!
//! ## Parameters
//! - **period**: The size of the window (number of data points). Defaults to 48.
//!
//! ## Errors
//! - **AllValuesNaN**: highpass: All input data values are `NaN`.
//! - **InvalidPeriod**: highpass: `period` is zero, exceeds data length, or data length is insufficient.
//! - **InvalidAlpha**: highpass: `cos_val` is too close to zero, preventing valid alpha computation.
//!
//! ## Returns
//! - **`Ok(HighPassOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(HighPassError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "python")]
use numpy::ndarray::{Array1, Array2};
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyArrayMethods};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

impl<'a> AsRef<[f64]> for HighPassInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            HighPassData::Slice(slice) => slice,
            HighPassData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HighPassData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone, Copy)]
pub struct HighPassParams {
    pub period: Option<usize>,
}
impl Default for HighPassParams {
    fn default() -> Self {
        Self { period: Some(48) }
    }
}

#[derive(Debug, Clone)]
pub struct HighPassOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HighPassInput<'a> {
    pub data: HighPassData<'a>,
    pub params: HighPassParams,
}

impl<'a> HighPassInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: HighPassParams) -> Self {
        Self {
            data: HighPassData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: HighPassParams) -> Self {
        Self {
            data: HighPassData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", HighPassParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(48)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HighPassBuilder {
    period: Option<usize>,
    kernel: Kernel,
}
impl Default for HighPassBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}
impl HighPassBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn period(mut self, n: usize) -> Self {
        self.period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<HighPassOutput, HighPassError> {
        let p = HighPassParams {
            period: self.period,
        };
        let i = HighPassInput::from_candles(c, "close", p);
        highpass_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<HighPassOutput, HighPassError> {
        let p = HighPassParams {
            period: self.period,
        };
        let i = HighPassInput::from_slice(d, p);
        highpass_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<HighPassStream, HighPassError> {
        let p = HighPassParams {
            period: self.period,
        };
        HighPassStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum HighPassError {
    #[error("highpass: Input data slice is empty.")]
    EmptyInputData,
    #[error("highpass: All values are NaN.")]
    AllValuesNaN,
    #[error("highpass: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("highpass: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("highpass: Invalid output buffer size: expected = {expected}, actual = {actual}")]
    InvalidOutputBuffer { expected: usize, actual: usize },
    #[error(
        "highpass: Invalid alpha calculation. cos_val is too close to zero: cos_val = {cos_val}"
    )]
    InvalidAlpha { cos_val: f64 },
}

#[inline]
pub fn highpass(input: &HighPassInput) -> Result<HighPassOutput, HighPassError> {
    highpass_with_kernel(input, Kernel::Auto)
}

#[inline]
fn highpass_into(
    input: &HighPassInput,
    out: &mut [f64],
) -> Result<(), HighPassError> {
    highpass_with_kernel_into(input, Kernel::Auto, out)
}

pub fn highpass_with_kernel(
    input: &HighPassInput,
    kernel: Kernel,
) -> Result<HighPassOutput, HighPassError> {
    let data: &[f64] = match &input.data {
        HighPassData::Candles { candles, source } => source_type(candles, source),
        HighPassData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(HighPassError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HighPassError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    if len <= 2 || period == 0 || period > len {
        return Err(HighPassError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(HighPassError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let k = 1.0;
    let two_pi_k_div = 2.0 * std::f64::consts::PI * k / (period as f64);
    let cos_val = two_pi_k_div.cos();
    if cos_val.abs() < 1e-15 {
        return Err(HighPassError::InvalidAlpha { cos_val });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm = first + period;
    let mut out = alloc_with_nan_prefix(len, warm);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => highpass_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => highpass_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => highpass_avx512(data, period, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(HighPassOutput { values: out })
}

fn highpass_with_kernel_into(
    input: &HighPassInput,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), HighPassError> {
    let data: &[f64] = match &input.data {
        HighPassData::Candles { candles, source } => source_type(candles, source),
        HighPassData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(HighPassError::EmptyInputData);
    }
    
    // Ensure output buffer is the correct size
    if out.len() != data.len() {
        return Err(HighPassError::InvalidOutputBuffer {
            expected: data.len(),
            actual: out.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HighPassError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    if len <= 2 || period == 0 || period > len {
        return Err(HighPassError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(HighPassError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let k = 1.0;
    let two_pi_k_div = 2.0 * std::f64::consts::PI * k / (period as f64);
    let cos_val = two_pi_k_div.cos();
    if cos_val.abs() < 1e-15 {
        return Err(HighPassError::InvalidAlpha { cos_val });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm = first + period;
    // Initialize NaN prefix
    out[..warm].fill(f64::NAN);
    
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => highpass_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => highpass_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => highpass_avx512(data, period, first, out),
            _ => unreachable!(),
        }
    }

    Ok(())
}

// Scalar implementation
#[inline]
pub unsafe fn highpass_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let len = data.len();
    let k = 1.0;
    let two_pi_k_div = 2.0 * std::f64::consts::PI * k / (period as f64);
    let sin_val = two_pi_k_div.sin();
    let cos_val = two_pi_k_div.cos();

    let alpha = 1.0 + (sin_val - 1.0) / cos_val;
    let one_minus_half_alpha = 1.0 - alpha / 2.0;
    let one_minus_alpha = 1.0 - alpha;

    out[0] = data[0];
    for i in 1..len {
        let val = one_minus_half_alpha * data[i] - one_minus_half_alpha * data[i - 1]
            + one_minus_alpha * out[i - 1];
        out[i] = val;
    }
}

// AVX2 stub
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn highpass_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    use core::f64::consts::PI;

    let n = data.len();
    if n == 0 {
        return;
    }

    /* --- pre-compute coefficients --------------------------------------- */
    let k = 1.0;
    let theta = 2.0 * PI * k / period as f64;
    let alpha = 1.0 + ((theta.sin() - 1.0) / theta.cos());
    let c = 1.0 - 0.5 * alpha; // (1-α/2)
    let oma = 1.0 - alpha; // (1-α)

    /* --- seed ----------------------------------------------------------- */
    out[0] = data[0];
    if n == 1 {
        return;
    }

    /* --- pointer loop, 2× unrolled ------------------------------------- */
    let mut src = data.as_ptr().add(1);
    let mut dst = out.as_mut_ptr().add(1);
    let mut y_im1 = out[0];
    let mut x_im1 = data[0];
    let mut rem = n - 1;

    while rem >= 2 {
        // y[i]
        let x_i = *src;
        let y_i = oma.mul_add(y_im1, c * (x_i - x_im1));
        *dst = y_i;

        // y[i+1]
        let x_ip1 = *src.add(1);
        let y_ip1 = oma.mul_add(y_i, c * (x_ip1 - x_i));
        *dst.add(1) = y_ip1;

        /* rotate state */
        x_im1 = x_ip1;
        y_im1 = y_ip1;
        src = src.add(2);
        dst = dst.add(2);
        rem -= 2;
    }

    if rem == 1 {
        let x_i = *src;
        *dst = oma.mul_add(y_im1, c * (x_i - x_im1));
    }
}

// AVX512 stub and long/short variants
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn highpass_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    highpass_avx2(data, period, first, out)
}

// Batch/Range types and functions
#[derive(Clone, Debug)]
pub struct HighPassBatchRange {
    pub period: (usize, usize, usize),
}
impl Default for HighPassBatchRange {
    fn default() -> Self {
        Self {
            period: (48, 48, 0),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct HighPassBatchBuilder {
    range: HighPassBatchRange,
    kernel: Kernel,
}
impl HighPassBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<HighPassBatchOutput, HighPassError> {
        highpass_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<HighPassBatchOutput, HighPassError> {
        HighPassBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<HighPassBatchOutput, HighPassError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<HighPassBatchOutput, HighPassError> {
        HighPassBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct HighPassBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HighPassParams>,
    pub rows: usize,
    pub cols: usize,
}
impl HighPassBatchOutput {
    pub fn row_for_params(&self, p: &HighPassParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(48) == p.period.unwrap_or(48))
    }
    pub fn values_for(&self, p: &HighPassParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &HighPassBatchRange) -> Vec<HighPassParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(HighPassParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn highpass_batch_with_kernel(
    data: &[f64],
    sweep: &HighPassBatchRange,
    k: Kernel,
) -> Result<HighPassBatchOutput, HighPassError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(HighPassError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    highpass_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn highpass_batch_slice(
    data: &[f64],
    sweep: &HighPassBatchRange,
    kern: Kernel,
) -> Result<HighPassBatchOutput, HighPassError> {
    highpass_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn highpass_batch_par_slice(
    data: &[f64],
    sweep: &HighPassBatchRange,
    kern: Kernel,
) -> Result<HighPassBatchOutput, HighPassError> {
    highpass_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn highpass_batch_inner(
    data: &[f64],
    sweep: &HighPassBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<HighPassBatchOutput, HighPassError> {
    // Get combos to calculate dimensions
    let combos = expand_grid(sweep);
    let rows = combos.len();
    let cols = data.len();
    
    // Allocate output buffer
    let mut values = vec![0.0; rows * cols];
    
    // Delegate to the _into version which contains all the logic
    highpass_batch_inner_into(data, sweep, kern, parallel, &mut values)?;
    
    Ok(HighPassBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

// Row functions, all variants just call scalar
#[inline(always)]
pub unsafe fn highpass_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    highpass_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn highpass_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    highpass_avx2(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn highpass_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    highpass_row_avx2(data, first, period, out)
}

// Streaming
#[derive(Debug, Clone)]
pub struct HighPassStream {
    period: usize,
    alpha: f64,
    one_minus_half_alpha: f64,
    one_minus_alpha: f64,
    prev_data: f64,
    prev_output: f64,
    initialized: bool,
}
impl HighPassStream {
    pub fn try_new(params: HighPassParams) -> Result<Self, HighPassError> {
        let period = params.period.unwrap_or(48);
        if period == 0 {
            return Err(HighPassError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let k = 1.0;
        let two_pi_k_div = 2.0 * std::f64::consts::PI * k / (period as f64);
        let sin_val = two_pi_k_div.sin();
        let cos_val = two_pi_k_div.cos();
        if cos_val.abs() < 1e-15 {
            return Err(HighPassError::InvalidAlpha { cos_val });
        }
        let alpha = 1.0 + (sin_val - 1.0) / cos_val;
        Ok(Self {
            period,
            alpha,
            one_minus_half_alpha: 1.0 - alpha / 2.0,
            one_minus_alpha: 1.0 - alpha,
            prev_data: f64::NAN,
            prev_output: f64::NAN,
            initialized: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> f64 {
        if !self.initialized {
            self.prev_data = value;
            self.prev_output = value;
            self.initialized = true;
            return value;
        }
        let out = self.one_minus_half_alpha * value - self.one_minus_half_alpha * self.prev_data
            + self.one_minus_alpha * self.prev_output;
        self.prev_data = value;
        self.prev_output = out;
        out
    }
}

// Tests: macro structure matches ALMA
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use proptest::prelude::*;
    use std::error::Error;

    fn check_highpass_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = HighPassParams { period: None };
        let input_default = HighPassInput::from_candles(&candles, "close", default_params);
        let output_default = highpass_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        let params_period = HighPassParams { period: Some(36) };
        let input_period = HighPassInput::from_candles(&candles, "hl2", params_period);
        let output_period = highpass_with_kernel(&input_period, kernel)?;
        assert_eq!(output_period.values.len(), candles.close.len());
        Ok(())
    }
    fn check_highpass_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HighPassInput::with_default_candles(&candles);
        let result = highpass_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -265.1027020005024,
            -330.0916060058495,
            -422.7478979710918,
            -261.87532144673423,
            -698.9026088956363,
        ];
        let start = result.values.len().saturating_sub(5);
        let last_five = &result.values[start..];
        for (i, &val) in last_five.iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Highpass mismatch at {}: expected {}, got {}",
                test_name,
                i,
                expected_last_five[i],
                val
            );
        }
        Ok(())
    }
    fn check_highpass_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HighPassInput::with_default_candles(&candles);
        match input.data {
            HighPassData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Unexpected data variant"),
        }
        let output = highpass_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_highpass_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = HighPassParams { period: Some(0) };
        let input = HighPassInput::from_slice(&input_data, params);
        let result = highpass_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Highpass should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_highpass_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = HighPassParams { period: Some(48) };
        let input = HighPassInput::from_slice(&input_data, params);
        let result = highpass_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Highpass should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_highpass_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [42.0, 43.0];
        let params = HighPassParams { period: Some(2) };
        let input = HighPassInput::from_slice(&input_data, params);
        let result = highpass_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Highpass should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_highpass_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = HighPassParams { period: Some(36) };
        let first_input = HighPassInput::from_candles(&candles, "close", first_params);
        let first_result = highpass_with_kernel(&first_input, kernel)?;
        let second_params = HighPassParams { period: Some(24) };
        let second_input = HighPassInput::from_slice(&first_result.values, second_params);
        let second_result = highpass_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for val in &second_result.values[240..] {
            assert!(!val.is_nan());
        }
        Ok(())
    }
    fn check_highpass_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = HighPassParams { period: Some(48) };
        let input = HighPassInput::from_candles(&candles, "close", params);
        let result = highpass_with_kernel(&input, kernel)?;
        for val in &result.values {
            assert!(!val.is_nan());
        }
        Ok(())
    }
    fn check_highpass_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 48;
        let input = HighPassInput::from_candles(
            &candles,
            "close",
            HighPassParams {
                period: Some(period),
            },
        );
        let batch_output = highpass_with_kernel(&input, kernel)?.values;
        let mut stream = HighPassStream::try_new(HighPassParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            let hp_val = stream.update(price);
            stream_values.push(hp_val);
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-8,
                "[{}] Highpass streaming mismatch at idx {}: batch={}, stream={}",
                test_name,
                i,
                b,
                s
            );
        }
        Ok(())
    }

    fn check_highpass_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = HighPassInput::from_slice(&empty, HighPassParams::default());
        let res = highpass_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(HighPassError::EmptyInputData)),
            "[{}] expected EmptyInputData",
            test_name
        );
        Ok(())
    }

    fn check_highpass_invalid_alpha(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = HighPassParams { period: Some(4) }; // cos(pi/2) ~ 0
        let input = HighPassInput::from_slice(&data, params);
        let res = highpass_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(HighPassError::InvalidAlpha { .. })),
            "[{}] expected InvalidAlpha",
            test_name
        );
        Ok(())
    }

    fn check_highpass_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let strat = (
            (-1e6f64..1e6).prop_filter("finite", |x| x.is_finite()),
            30usize..200,
            3usize..30,
        );
        proptest::test_runner::TestRunner::default()
            .run(&strat, |(val, len, period)| {
                let cos_val = (2.0 * std::f64::consts::PI / (period as f64)).cos();
                prop_assume!(cos_val.abs() >= 1e-15);
                let data = vec![val; len];
                let params = HighPassParams {
                    period: Some(period),
                };
                let input = HighPassInput::from_slice(&data, params);
                let HighPassOutput { values: out } = highpass_with_kernel(&input, kernel).unwrap();
                let last = *out.last().unwrap();
                prop_assert!(last.abs() <= val.abs() * 0.01);
                Ok(())
            })
            .unwrap();
        Ok(())
    }

    macro_rules! generate_all_highpass_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test] fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[test] fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }
    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_highpass_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>>      
    {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations
        let test_cases = vec![
            HighPassParams { period: Some(48) },  // default
            HighPassParams { period: Some(10) },  // small period
            HighPassParams { period: Some(100) }, // large period
            HighPassParams { period: Some(3) },   // very small period
            HighPassParams { period: Some(20) },  // medium period
            HighPassParams { period: Some(60) },  // larger period
            HighPassParams { period: Some(5) },   // another small period
            HighPassParams { period: Some(80) },  // another large period
            HighPassParams { period: None },      // None value (use default)
        ];

        for params in test_cases {
            // Skip period=4 as it causes InvalidAlpha error (cos(pi/2) ~ 0)
            if params.period == Some(4) {
                continue;
            }
            
            let input = HighPassInput::from_candles(&candles, "close", params);
            let output = highpass_with_kernel(&input, kernel)?;

            // Check every value for poison patterns
            for (i, &val) in output.values.iter().enumerate() {
                // Skip NaN values as they're expected in the warmup period
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",        
                        test_name, val, bits, i, params.period
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_highpass_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    generate_all_highpass_tests!(
        check_highpass_partial_params,
        check_highpass_accuracy,
        check_highpass_default_candles,
        check_highpass_zero_period,
        check_highpass_period_exceeds_length,
        check_highpass_very_small_dataset,
        check_highpass_reinput,
        check_highpass_nan_handling,
        check_highpass_streaming,
        check_highpass_empty_input,
        check_highpass_invalid_alpha,
        check_highpass_property,
        check_highpass_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = HighPassBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = HighPassParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -265.1027020005024,
            -330.0916060058495,
            -422.7478979710918,
            -261.87532144673423,
            -698.9026088956363,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-6,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }
    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test] fn [<$fn_name _scalar>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx2>]()        {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test] fn [<$fn_name _avx512>]()      {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test] fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple batch configurations with different parameter ranges
        let batch_configs = vec![
            // Original test case
            (10, 30, 10),
            // Edge cases
            (48, 48, 0),       // Single parameter (default)
            (3, 15, 3),        // Small periods
            (50, 100, 25),     // Large periods
            (5, 25, 5),        // Different step
            (20, 80, 20),      // Medium to large
            (7, 21, 7),        // Different small range
            (100, 120, 10),    // Very large periods
        ];

        for (p_start, p_end, p_step) in batch_configs {
            // Skip any period that would be exactly 4 (causes InvalidAlpha)
            let periods: Vec<usize> = if p_step == 0 || p_start == p_end {
                vec![p_start]
            } else {
                (p_start..=p_end).step_by(p_step).filter(|&p| p != 4).collect()
            };
            
            // If all periods would be 4, skip this config
            if periods.is_empty() || (periods.len() == 1 && periods[0] == 4) {
                continue;
            }
            
            let output = HighPassBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c, "close")?;

            // Check every value in the entire batch matrix for poison patterns
            for (idx, &val) in output.values.iter().enumerate() {
                // Skip NaN values as they're expected in warmup periods
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
                        test, val, bits, row, col, idx, combo.period
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
                        test, val, bits, row, col, idx, combo.period
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
                        test, val, bits, row, col, idx, combo.period
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

/// highpass_batch_inner_into writes directly to the output buffer
#[inline(always)]
fn highpass_batch_inner_into(
    data: &[f64],
    sweep: &HighPassBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<HighPassParams>, HighPassError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(HighPassError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    if data.is_empty() {
        return Err(HighPassError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HighPassError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(HighPassError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    for c in &combos {
        let period = c.period.unwrap();
        let k = 1.0;
        let cos_val = (2.0 * std::f64::consts::PI * k / period as f64).cos();
        if cos_val.abs() < 1e-15 {
            return Err(HighPassError::InvalidAlpha { cos_val });
        }
    }

    let rows = combos.len();
    let cols = data.len();

    // ---------- per-row warm-up lengths ----------
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();

    // Reinterpret output slice as MaybeUninit
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut MaybeUninit<f64>,
            out.len()
        )
    };

    // Initialize NaN prefixes
    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    // ---------- 2. worker that fills one row ----------
    let do_row = |row: usize, dst_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // Re-interpret this row as &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => highpass_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => highpass_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => highpass_row_avx512(data, first, period, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))] {
            out_uninit.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")] {
            for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out_uninit.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(combos)
}

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "highpass", signature = (arr_in, period=48))]
pub fn highpass_py<'py>(
    py: Python<'py>,
    arr_in: numpy::PyReadonlyArray1<'py, f64>,
    period: Option<usize>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};
    
    let slice_in = arr_in.as_slice()?; // zero-copy, read-only view
    
    // Build input struct with optional period (defaults to 48)
    let params = HighPassParams {
        period: period,
    };
    let hp_in = HighPassInput::from_slice(slice_in, params);
    
    // Allocate NumPy output buffer
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? }; // safe: we own the array
    
    // Heavy lifting without the GIL
    py.allow_threads(|| -> Result<(), HighPassError> {
        highpass_into(&hp_in, slice_out)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?; // unify error type
    
    Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyfunction(name = "highpass_batch")]
pub fn highpass_batch_py<'py>(
    py: Python<'py>,
    arr_in: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{PyArray1, PyArrayMethods, IntoPyArray};
    use pyo3::types::PyDict;
    
    let slice_in = arr_in.as_slice()?;
    
    let sweep = HighPassBatchRange {
        period: period_range,
    };
    
    // Expand grid once to know rows*cols
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();
    
    // Pre-allocate NumPy array (1-D, will reshape later)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };
    
    // Heavy work without the GIL
    let combos = py.allow_threads(|| {
        // Resolve Kernel::Auto to a specific kernel
        let kernel = match Kernel::Auto {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let simd = match kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };
        // Use the _into variant that writes directly to our pre-allocated buffer
        highpass_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    // Build dict with the GIL
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "periods",
        combos
            .iter()
            .map(|p| p.period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "HighPassStream")]
pub struct HighPassStreamPy {
    inner: HighPassStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl HighPassStreamPy {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> {
        let params = HighPassParams {
            period: Some(period),
        };
        match HighPassStream::try_new(params) {
            Ok(stream) => Ok(Self { inner: stream }),
            Err(e) => Err(PyValueError::new_err(format!("HighPassStream error: {}", e))),
        }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        Some(self.inner.update(value))
    }
}

// WASM bindings
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn highpass_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = HighPassParams {
        period: Some(period),
    };
    let input = HighPassInput::from_slice(data, params);
    match highpass(&input) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&format!("HighPass error: {}", e))),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn highpass_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = HighPassBatchRange {
        period: (period_start, period_end, period_step),
    };
    match highpass_batch_with_kernel(data, &sweep, Kernel::Auto) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&format!("HighPass batch error: {}", e))),
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn highpass_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Vec<f64> {
    let periods: Vec<usize> = if period_step == 0 || period_start == period_end {
        vec![period_start]
    } else {
        (period_start..=period_end).step_by(period_step).collect()
    };
    
    let mut result = Vec::new();
    for &period in &periods {
        result.push(period as f64);
    }
    result
}
