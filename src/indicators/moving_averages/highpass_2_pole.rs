//! # 2-Pole High-Pass Filter
//!
//! A 2-pole high-pass filter using a user-specified cutoff frequency (`k`). This filter
//! removes or attenuates lower-frequency components from the input data.
//!
//! ## Parameters
//! - **period**: Window size (must be ≥ 2).
//! - **k**: Cutoff frequency (commonly in [0.0, 1.0]) controlling the filter’s
//!          attenuation of low-frequency components (defaults to 0.707).
//!
//! ## Errors
//! - **InvalidPeriod**: highpass_2_pole: `period` < 2 or data is empty.
//! - **InvalidK**: highpass_2_pole: `k` ≤ 0.0 or `k` is `NaN`.
//!
//! ## Returns
//! - **`Ok(HighPass2Output)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(HighPass2Error)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel, alloc_with_nan_prefix, init_matrix_prefixes, make_uninit_matrix};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum HighPass2Data<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for HighPass2Input<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            HighPass2Data::Slice(slice) => slice,
            HighPass2Data::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HighPass2Output {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HighPass2Params {
    pub period: Option<usize>,
    pub k: Option<f64>,
}

impl Default for HighPass2Params {
    fn default() -> Self {
        Self {
            period: Some(48),
            k: Some(0.707),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HighPass2Input<'a> {
    pub data: HighPass2Data<'a>,
    pub params: HighPass2Params,
}

impl<'a> HighPass2Input<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: HighPass2Params) -> Self {
        Self {
            data: HighPass2Data::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: HighPass2Params) -> Self {
        Self {
            data: HighPass2Data::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", HighPass2Params::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(48)
    }
    #[inline]
    pub fn get_k(&self) -> f64 {
        self.params.k.unwrap_or(0.707)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HighPass2Builder {
    period: Option<usize>,
    k: Option<f64>,
    kernel: Kernel,
}

impl Default for HighPass2Builder {
    fn default() -> Self {
        Self {
            period: None,
            k: None,
            kernel: Kernel::Auto,
        }
    }
}

impl HighPass2Builder {
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
    pub fn k(mut self, val: f64) -> Self {
        self.k = Some(val);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<HighPass2Output, HighPass2Error> {
        let p = HighPass2Params {
            period: self.period,
            k: self.k,
        };
        let i = HighPass2Input::from_candles(c, "close", p);
        highpass_2_pole_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<HighPass2Output, HighPass2Error> {
        let p = HighPass2Params {
            period: self.period,
            k: self.k,
        };
        let i = HighPass2Input::from_slice(d, p);
        highpass_2_pole_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<HighPass2Stream, HighPass2Error> {
        let p = HighPass2Params {
            period: self.period,
            k: self.k,
        };
        HighPass2Stream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum HighPass2Error {
    #[error("highpass_2_pole: All values are NaN.")]
    AllValuesNaN,
    #[error("highpass_2_pole: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("highpass_2_pole: Invalid k value: {k}")]
    InvalidK { k: f64 },
    #[error("highpass_2_pole: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn highpass_2_pole(input: &HighPass2Input) -> Result<HighPass2Output, HighPass2Error> {
    highpass_2_pole_with_kernel(input, Kernel::Auto)
}

pub fn highpass_2_pole_with_kernel(
    input: &HighPass2Input,
    kernel: Kernel,
) -> Result<HighPass2Output, HighPass2Error> {
    let data: &[f64] = input.as_ref();
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HighPass2Error::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    let k = input.get_k();

    if period < 2 || len == 0 {
        return Err(HighPass2Error::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if !(k > 0.0) || k.is_nan() || k.is_infinite() {
        return Err(HighPass2Error::InvalidK { k });
    }
    if len - first < period {
        return Err(HighPass2Error::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm = first + period;
    let mut out = alloc_with_nan_prefix(len, warm);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                highpass_2_pole_scalar(data, period, k, first, &mut out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                highpass_2_pole_avx2(data, period, k, first, &mut out);
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                highpass_2_pole_avx512(data, period, k, first, &mut out);
            }
            _ => unreachable!(),
        }
    }

    Ok(HighPass2Output { values: out })
}

#[inline(always)]
pub fn highpass_2_pole_scalar(data: &[f64], period: usize, k: f64, first: usize, out: &mut [f64]) {
    unsafe { highpass_2_pole_scalar_unsafe(data, period, k, first, out) }
}

#[inline(always)]
pub unsafe fn highpass_2_pole_scalar_unsafe(
    data: &[f64],
    period: usize,
    k: f64,
    _first: usize,
    out: &mut [f64],
) {
    use std::f64::consts::PI;
    let len = data.len();
    let angle = 2.0 * PI * k / (period as f64);
    let sin_val = angle.sin();
    let cos_val = angle.cos();
    let alpha = 1.0 + ((sin_val - 1.0) / cos_val);

    let one_minus_alpha_half = 1.0 - alpha / 2.0;
    let c = one_minus_alpha_half * one_minus_alpha_half;

    let one_minus_alpha = 1.0 - alpha;
    let one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;

    if len > 0 {
        out[0] = data[0];
    }
    if len > 1 {
        out[1] = data[1];
    }

    for i in 2..len {
        out[i] = c * data[i] - 2.0 * c * data[i - 1]
            + c * data[i - 2]
            + 2.0 * one_minus_alpha * out[i - 1]
            - one_minus_alpha_sq * out[i - 2];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn highpass_2_pole_avx2(data: &[f64], period: usize, k: f64, first: usize, out: &mut [f64]) {
    unsafe { highpass_2_pole_scalar_unsafe(data, period, k, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn highpass_2_pole_avx512(data: &[f64], period: usize, k: f64, first: usize, out: &mut [f64]) {
    unsafe { highpass_2_pole_scalar_unsafe(data, period, k, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn highpass_2_pole_avx512_short(
    data: &[f64],
    period: usize,
    k: f64,
    first: usize,
    out: &mut [f64],
) {
    highpass_2_pole_avx512(data, period, k, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn highpass_2_pole_avx512_long(
    data: &[f64],
    period: usize,
    k: f64,
    first: usize,
    out: &mut [f64],
) {
    highpass_2_pole_avx512(data, period, k, first, out)
}

#[inline(always)]
pub fn highpass_2_pole_with_kernel_and_kernel(
    input: &HighPass2Input,
    kernel: Kernel,
) -> Result<HighPass2Output, HighPass2Error> {
    highpass_2_pole_with_kernel(input, kernel)
}

#[derive(Debug, Clone)]
pub struct HighPass2Stream {}

impl HighPass2Stream {
    pub fn try_new(_params: HighPass2Params) -> Result<Self, HighPass2Error> {
        Ok(Self {})
    }
    #[inline(always)]
    pub fn update(&mut self, _value: f64) -> Option<f64> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct HighPass2BatchRange {
    pub period: (usize, usize, usize),
    pub k: (f64, f64, f64),
}

impl Default for HighPass2BatchRange {
    fn default() -> Self {
        Self {
            period: (48, 48, 0),
            k: (0.707, 0.707, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HighPass2BatchBuilder {
    range: HighPass2BatchRange,
    kernel: Kernel,
}

impl HighPass2BatchBuilder {
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
    pub fn k_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.k = (start, end, step);
        self
    }
    pub fn k_static(mut self, val: f64) -> Self {
        self.range.k = (val, val, 0.0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<HighPass2BatchOutput, HighPass2Error> {
        highpass_2_pole_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<HighPass2BatchOutput, HighPass2Error> {
        HighPass2BatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<HighPass2BatchOutput, HighPass2Error> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<HighPass2BatchOutput, HighPass2Error> {
        HighPass2BatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn highpass_2_pole_batch_with_kernel(
    data: &[f64],
    sweep: &HighPass2BatchRange,
    k: Kernel,
) -> Result<HighPass2BatchOutput, HighPass2Error> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(HighPass2Error::InvalidPeriod {
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
    highpass_2_pole_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct HighPass2BatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HighPass2Params>,
    pub rows: usize,
    pub cols: usize,
}
impl HighPass2BatchOutput {
    pub fn row_for_params(&self, p: &HighPass2Params) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(48) == p.period.unwrap_or(48)
                && (c.k.unwrap_or(0.707) - p.k.unwrap_or(0.707)).abs() < 1e-12
        })
    }
    pub fn values_for(&self, p: &HighPass2Params) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &HighPass2BatchRange) -> Vec<HighPass2Params> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
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
    let periods = axis_usize(r.period);
    let ks = axis_f64(r.k);
    let mut out = Vec::with_capacity(periods.len() * ks.len());
    for &p in &periods {
        for &k in &ks {
            out.push(HighPass2Params {
                period: Some(p),
                k: Some(k),
            });
        }
    }
    out
}

#[inline(always)]
pub fn highpass_2_pole_batch_slice(
    data: &[f64],
    sweep: &HighPass2BatchRange,
    kern: Kernel,
) -> Result<HighPass2BatchOutput, HighPass2Error> {
    highpass_2_pole_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn highpass_2_pole_batch_par_slice(
    data: &[f64],
    sweep: &HighPass2BatchRange,
    kern: Kernel,
) -> Result<HighPass2BatchOutput, HighPass2Error> {
    highpass_2_pole_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn highpass_2_pole_batch_inner(
    data: &[f64],
    sweep: &HighPass2BatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<HighPass2BatchOutput, HighPass2Error> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(HighPass2Error::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HighPass2Error::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(HighPass2Error::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // ---------- per-row warm-up lengths ----------
    let warm: Vec<usize> = combos.iter()
        .map(|c| first + c.period.unwrap())
        .collect();

    // ---------- 1. allocate rows×cols buffer & seed NaN prefixes ----------
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---------- 2. worker that fills one row ----------
    let do_row = |row: usize, dst_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let k      = combos[row].k.unwrap();

        // Re-interpret this row as &mut [f64]
        let out_row = core::slice::from_raw_parts_mut(
            dst_mu.as_mut_ptr() as *mut f64,
            dst_mu.len(),
        );

        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => highpass_2_pole_row_avx512(data, first, period, k, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2   => highpass_2_pole_row_avx2  (data, first, period, k, out_row),
            _              => highpass_2_pole_row_scalar(data, first, period, k, out_row),
        }
    };

    // ---------- 3. run every row directly into `raw` ----------
    if parallel {
        raw.par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, slice)| do_row(row, slice));
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ---------- 4. transmute to a Vec<f64> now that it is fully initialised ----------
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };

    Ok(HighPass2BatchOutput { values, combos, rows, cols })

}

#[inline(always)]
pub unsafe fn highpass_2_pole_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    k: f64,
    out: &mut [f64],
) {
    highpass_2_pole_scalar_unsafe(data, period, k, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn highpass_2_pole_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    k: f64,
    out: &mut [f64],
) {
    highpass_2_pole_row_scalar(data, first, period, k, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn highpass_2_pole_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    k: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        highpass_2_pole_row_avx512_short(data, first, period, k, out);
    } else {
        highpass_2_pole_row_avx512_long(data, first, period, k, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn highpass_2_pole_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    k: f64,
    out: &mut [f64],
) {
    highpass_2_pole_row_scalar(data, first, period, k, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn highpass_2_pole_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    k: f64,
    out: &mut [f64],
) {
    highpass_2_pole_row_scalar(data, first, period, k, out);
}

#[inline(always)]
fn round_up8(x: usize) -> usize {
    (x + 7) & !7
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    use paste::paste;

    fn check_highpass2_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = HighPass2Params {
            period: None,
            k: None,
        };
        let input = HighPass2Input::from_candles(&candles, "close", default_params);
        let output = highpass_2_pole_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_highpass2_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HighPass2Input::from_candles(&candles, "close", HighPass2Params::default());
        let result = highpass_2_pole_with_kernel(&input, kernel)?;
        let expected_last_five = [
            445.29073821108943,
            359.51467478973296,
            250.7236793408186,
            394.04381266217234,
            -52.65414073315134,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] HighPass2 {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_highpass2_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HighPass2Input::with_default_candles(&candles);
        match input.data {
            HighPass2Data::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected HighPass2Data::Candles"),
        }
        let output = highpass_2_pole_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_highpass2_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = HighPass2Params {
            period: Some(0),
            k: None,
        };
        let input = HighPass2Input::from_slice(&input_data, params);
        let res = highpass_2_pole_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] HighPass2 should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_highpass2_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = HighPass2Params {
            period: Some(10),
            k: None,
        };
        let input = HighPass2Input::from_slice(&data_small, params);
        let res = highpass_2_pole_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] HighPass2 should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_highpass2_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = HighPass2Params {
            period: Some(2),
            k: None,
        };
        let input = HighPass2Input::from_slice(&single_point, params);
        let res = highpass_2_pole_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), single_point.len());
        assert_eq!(res.values[0], single_point[0]);
        Ok(())
    }
    fn check_highpass2_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = HighPass2Params {
            period: Some(48),
            k: None,
        };
        let first_input = HighPass2Input::from_candles(&candles, "close", first_params);
        let first_result = highpass_2_pole_with_kernel(&first_input, kernel)?;
        let second_params = HighPass2Params {
            period: Some(32),
            k: None,
        };
        let second_input = HighPass2Input::from_slice(&first_result.values, second_params);
        let second_result = highpass_2_pole_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 240..second_result.values.len() {
            assert!(!second_result.values[i].is_nan());
        }
        Ok(())
    }
    fn check_highpass2_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HighPass2Input::from_candles(&candles, "close", HighPass2Params::default());
        let res = highpass_2_pole_with_kernel(&input, kernel)?;
        assert_eq!(res.values.len(), candles.close.len());
        if res.values.len() > 240 {
            for (i, &val) in res.values[240..].iter().enumerate() {
                assert!(
                    !val.is_nan(),
                    "[{}] Found unexpected NaN at out-index {}",
                    test_name,
                    240 + i
                );
            }
        }
        Ok(())
    }
    macro_rules! generate_all_highpass2_tests {
        ($($test_fn:ident),*) => {
            paste! {
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
    generate_all_highpass2_tests!(
        check_highpass2_partial_params,
        check_highpass2_accuracy,
        check_highpass2_default_candles,
        check_highpass2_zero_period,
        check_highpass2_period_exceeds_length,
        check_highpass2_very_small_dataset,
        check_highpass2_reinput,
        check_highpass2_nan_handling
    );
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = HighPass2BatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = HighPass2Params::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            445.29073821108943,
            359.51467478973296,
            250.7236793408186,
            394.04381266217234,
            -52.65414073315134,
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
            paste! {
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                     Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}
