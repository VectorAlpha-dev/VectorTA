//! # Midway Weighted Exponential (MWDX)
//!
//! A custom exponential smoothing approach using a user-defined `factor` to
//! determine weighting for new vs historical data.
//!
//! ## Parameters
//! - **factor**: Controls balance between new and historical data (must be > 0).
//!
//! ## Errors
//! - **EmptyData**: No input data provided.
//! - **InvalidFactor**: `factor` is ≤ 0.0, NaN, or infinite.
//! - **InvalidDenominator**: Factor leads to zero/negative denominator.
//!
//! ## Returns
//! - **Ok(MwdxOutput)**: Contains Vec<f64> with result, matching input length.
//! - **Err(MwdxError)**: On error.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

impl<'a> AsRef<[f64]> for MwdxInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MwdxData::Slice(slice) => slice,
            MwdxData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MwdxData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MwdxOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MwdxParams {
    pub factor: Option<f64>,
}

impl Default for MwdxParams {
    fn default() -> Self {
        Self { factor: Some(0.2) }
    }
}

#[derive(Debug, Clone)]
pub struct MwdxInput<'a> {
    pub data: MwdxData<'a>,
    pub params: MwdxParams,
}

impl<'a> MwdxInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MwdxParams) -> Self {
        Self {
            data: MwdxData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MwdxParams) -> Self {
        Self {
            data: MwdxData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MwdxParams::default())
    }
    #[inline]
    pub fn get_factor(&self) -> f64 {
        self.params.factor.unwrap_or(0.2)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MwdxBuilder {
    factor: Option<f64>,
    kernel: Kernel,
}

impl Default for MwdxBuilder {
    fn default() -> Self {
        Self {
            factor: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MwdxBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn factor(mut self, x: f64) -> Self {
        self.factor = Some(x);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<MwdxOutput, MwdxError> {
        let p = MwdxParams { factor: self.factor };
        let i = MwdxInput::from_candles(c, "close", p);
        mwdx_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MwdxOutput, MwdxError> {
        let p = MwdxParams { factor: self.factor };
        let i = MwdxInput::from_slice(d, p);
        mwdx_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<MwdxStream, MwdxError> {
        let p = MwdxParams { factor: self.factor };
        MwdxStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum MwdxError {
    #[error("mwdx: No input data was provided.")]
    EmptyData,
    #[error("mwdx: Factor must be greater than 0, got {factor}")]
    InvalidFactor { factor: f64 },
    #[error("mwdx: Factor leads to invalid denominator, factor: {factor}")]
    InvalidDenominator { factor: f64 },
}

#[inline]
pub fn mwdx(input: &MwdxInput) -> Result<MwdxOutput, MwdxError> {
    mwdx_with_kernel(input, Kernel::Auto)
}

pub fn mwdx_with_kernel(input: &MwdxInput, kernel: Kernel) -> Result<MwdxOutput, MwdxError> {
    let data: &[f64] = match &input.data {
        MwdxData::Candles { candles, source } => source_type(candles, source),
        MwdxData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(MwdxError::EmptyData);
    }
    let factor = input.get_factor();
    if factor <= 0.0 || factor.is_nan() || factor.is_infinite() {
        return Err(MwdxError::InvalidFactor { factor });
    }

    let val2 = (2.0 / factor) - 1.0;
    if val2 + 1.0 <= 0.0 {
        return Err(MwdxError::InvalidDenominator { factor });
    }
    let fac = 2.0 / (val2 + 1.0);

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let mut out = vec![f64::NAN; len];

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                mwdx_scalar(data, fac, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                mwdx_avx2(data, fac, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                mwdx_avx512(data, fac, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(MwdxOutput { values: out })
}

#[inline]
pub fn mwdx_scalar(
    data: &[f64],
    fac: f64,
    out: &mut [f64],
) {
    let n = data.len();
    if n == 0 { return; }
    out[0] = data[0];
    for i in 1..n {
        out[i] = fac * data[i] + (1.0 - fac) * out[i - 1];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mwdx_avx2(
    data: &[f64],
    fac: f64,
    out: &mut [f64],
) {
    // API parity: always call scalar for now
    mwdx_scalar(data, fac, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mwdx_avx512(
    data: &[f64],
    fac: f64,
    out: &mut [f64],
) {
    // API parity: always call scalar for now
    mwdx_scalar(data, fac, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mwdx_avx512_short(
    data: &[f64],
    fac: f64,
    out: &mut [f64],
) {
    mwdx_scalar(data, fac, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn mwdx_avx512_long(
    data: &[f64],
    fac: f64,
    out: &mut [f64],
) {
    mwdx_scalar(data, fac, out);
}

#[inline]
pub fn mwdx_batch_with_kernel(
    data: &[f64],
    sweep: &MwdxBatchRange,
    k: Kernel,
) -> Result<MwdxBatchOutput, MwdxError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(MwdxError::EmptyData)
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    mwdx_batch_par_slice(data, sweep, simd)
}

#[derive(Debug, Clone)]
pub struct MwdxStream {
    factor: f64,
    fac: f64,
    prev: Option<f64>,
}

impl MwdxStream {
    pub fn try_new(params: MwdxParams) -> Result<Self, MwdxError> {
        let factor = params.factor.unwrap_or(0.2);
        if factor <= 0.0 || factor.is_nan() || factor.is_infinite() {
            return Err(MwdxError::InvalidFactor { factor });
        }
        let val2 = (2.0 / factor) - 1.0;
        if val2 + 1.0 <= 0.0 {
            return Err(MwdxError::InvalidDenominator { factor });
        }
        let fac = 2.0 / (val2 + 1.0);
        Ok(Self { factor, fac, prev: None })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> f64 {
        let out = match self.prev {
            None => value,
            Some(prev) => self.fac * value + (1.0 - self.fac) * prev,
        };
        self.prev = Some(out);
        out
    }
}

#[derive(Clone, Debug)]
pub struct MwdxBatchRange {
    pub factor: (f64, f64, f64),
}

impl Default for MwdxBatchRange {
    fn default() -> Self {
        Self {
            factor: (0.2, 0.2, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MwdxBatchBuilder {
    range: MwdxBatchRange,
    kernel: Kernel,
}

impl MwdxBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn factor_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.factor = (start, end, step);
        self
    }
    #[inline]
    pub fn factor_static(mut self, x: f64) -> Self {
        self.range.factor = (x, x, 0.0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<MwdxBatchOutput, MwdxError> {
        mwdx_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MwdxBatchOutput, MwdxError> {
        MwdxBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MwdxBatchOutput, MwdxError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<MwdxBatchOutput, MwdxError> {
        MwdxBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct MwdxBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MwdxParams>,
    pub rows: usize,
    pub cols: usize,
}
impl MwdxBatchOutput {
    pub fn row_for_params(&self, p: &MwdxParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            (c.factor.unwrap_or(0.2) - p.factor.unwrap_or(0.2)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &MwdxParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &MwdxBatchRange) -> Vec<MwdxParams> {
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Vec<f64> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut v = Vec::new();
        let mut x = start;
        while x <= end + 1e-12 {
            v.push(x);
            x += step;
        }
        v
    }
    let factors = axis_f64(r.factor);
    let mut out = Vec::with_capacity(factors.len());
    for &f in &factors {
        out.push(MwdxParams {
            factor: Some(f),
        });
    }
    out
}

#[inline(always)]
pub fn mwdx_batch_slice(
    data: &[f64],
    sweep: &MwdxBatchRange,
    kern: Kernel,
) -> Result<MwdxBatchOutput, MwdxError> {
    mwdx_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn mwdx_batch_par_slice(
    data: &[f64],
    sweep: &MwdxBatchRange,
    kern: Kernel,
    // note: API expects "par" to mean "parallel"
) -> Result<MwdxBatchOutput, MwdxError> {
    mwdx_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn mwdx_batch_inner(
    data: &[f64],
    sweep: &MwdxBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MwdxBatchOutput, MwdxError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MwdxError::EmptyData);
    }
    if data.is_empty() {
        return Err(MwdxError::EmptyData);
    }

    let rows = combos.len();
    let cols = data.len();
    let mut values = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let prm = &combos[row];
        let factor = prm.factor.unwrap();
        let val2 = (2.0 / factor) - 1.0;
        if factor <= 0.0 || factor.is_nan() || factor.is_infinite() {
            return;
        }
        if val2 + 1.0 <= 0.0 {
            return;
        }
        let fac = 2.0 / (val2 + 1.0);
        match kern {
            Kernel::Scalar => mwdx_row_scalar(data, fac, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => mwdx_row_avx2(data, fac, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => mwdx_row_avx512(data, fac, out_row),
            _ => unreachable!(),
        }
    };

    if parallel {
        values
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in values.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    Ok(MwdxBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn mwdx_row_scalar(
    data: &[f64],
    fac: f64,
    out: &mut [f64],
) {
    let n = data.len();
    if n == 0 { return; }
    out[0] = data[0];
    for i in 1..n {
        out[i] = fac * data[i] + (1.0 - fac) * out[i - 1];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mwdx_row_avx2(
    data: &[f64],
    fac: f64,
    out: &mut [f64],
) {
    mwdx_row_scalar(data, fac, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn mwdx_row_avx512(
    data: &[f64],
    fac: f64,
    out: &mut [f64],
) {
    if data.len() <= 32 {
        mwdx_row_avx512_short(data, fac, out);
    } else {
        mwdx_row_avx512_long(data, fac, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mwdx_row_avx512_short(
    data: &[f64],
    fac: f64,
    out: &mut [f64],
) {
    mwdx_row_scalar(data, fac, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mwdx_row_avx512_long(
    data: &[f64],
    fac: f64,
    out: &mut [f64],
) {
    mwdx_row_scalar(data, fac, out);
}

#[inline(always)]
pub fn expand_grid_mwdx(r: &MwdxBatchRange) -> Vec<MwdxParams> {
    expand_grid(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_mwdx_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = MwdxParams { factor: None };
        let input = MwdxInput::from_candles(&candles, "close", default_params);
        let output = mwdx_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        let params_factor_05 = MwdxParams { factor: Some(0.5) };
        let input2 = MwdxInput::from_candles(&candles, "hl2", params_factor_05);
        let output2 = mwdx_with_kernel(&input2, kernel)?;
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = MwdxParams { factor: Some(0.7) };
        let input3 = MwdxInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = mwdx_with_kernel(&input3, kernel)?;
        assert_eq!(output3.values.len(), candles.close.len());
        Ok(())
    }

    fn check_mwdx_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let expected_last_five = [
            59302.181566190935,
            59277.94525295275,
            59230.1562023622,
            59215.124961889764,
            59103.099969511815,
        ];
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MwdxParams { factor: Some(0.2) };
        let input = MwdxInput::from_candles(&candles, "close", params);
        let result = mwdx_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        assert!(result.values.len() >= 5);
        let start_idx = result.values.len() - 5;
        let actual_last_five = &result.values[start_idx..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp_val = expected_last_five[i];
            assert!(
                (val - exp_val).abs() < 1e-5,
                "[{}] MWDX mismatch at index {}, expected {}, got {}",
                test_name,
                i,
                exp_val,
                val
            );
        }
        Ok(())
    }

    fn check_mwdx_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MwdxInput::with_default_candles(&candles);
        match input.data {
            MwdxData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected MwdxData::Candles"),
        }
        let output = mwdx_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_mwdx_zero_factor(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = MwdxParams { factor: Some(0.0) };
        let input = MwdxInput::from_slice(&input_data, params);
        let res = mwdx_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MWDX should fail with zero factor",
            test_name
        );
        Ok(())
    }

    fn check_mwdx_negative_factor(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = MwdxParams { factor: Some(-0.5) };
        let input = MwdxInput::from_slice(&data, params);
        let result = mwdx_with_kernel(&input, kernel);
        assert!(result.is_err(), "[{}] MWDX should fail with negative factor", test_name);
        Ok(())
    }

    fn check_mwdx_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [42.0];
        let params = MwdxParams { factor: Some(0.2) };
        let input = MwdxInput::from_slice(&data, params);
        let result = mwdx_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), data.len());
        assert_eq!(result.values[0], 42.0);
        Ok(())
    }

    fn check_mwdx_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_input = MwdxInput::from_candles(&candles, "close", MwdxParams { factor: Some(0.2) });
        let first_result = mwdx_with_kernel(&first_input, kernel)?;
        let second_input = MwdxInput::from_slice(&first_result.values, MwdxParams { factor: Some(0.3) });
        let second_result = mwdx_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 0..second_result.values.len() {
            assert!(
                second_result.values[i].is_finite(),
                "[{}] NaN found at index {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    fn check_mwdx_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MwdxInput::from_candles(&candles, "close", MwdxParams { factor: Some(0.2) });
        let result = mwdx_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        for (i, &val) in result.values.iter().enumerate() {
            assert!(val.is_finite(), "[{}] NaN found at index {}", test_name, i);
        }
        Ok(())
    }

    macro_rules! generate_all_mwdx_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                )*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }

    generate_all_mwdx_tests!(
        check_mwdx_partial_params,
        check_mwdx_accuracy,
        check_mwdx_default_candles,
        check_mwdx_zero_factor,
        check_mwdx_negative_factor,
        check_mwdx_very_small_dataset,
        check_mwdx_reinput,
        check_mwdx_nan_handling
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = MwdxBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = MwdxParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59302.181566190935,
            59277.94525295275,
            59230.1562023622,
            59215.124961889764,
            59103.099969511815,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-5,
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
    gen_batch_tests!(check_batch_default_row);
}
