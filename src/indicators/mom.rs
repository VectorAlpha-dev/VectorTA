//! # Momentum (MOM)
//!
//! MOM measures the amount that a security's price has changed over a given time span.
//! It is calculated by subtracting the previous price (from the chosen period) from the
//! current price, i.e., `momentum[i] = data[i] - data[i - period]`.
//!
//! ## Parameters
//! - **period**: The lookback window size (number of data points). Defaults to 10.
//!
//! ## Inputs
//! - **data**: Price data or any numeric series
//!
//! ## Returns
//! - **values**: Vector of momentum values with NaN prefix during warmup period
//!
//! ## Developer Notes
//! - **AVX2/AVX512 Kernels**: Stubs that call scalar implementation
//! - **Streaming**: Implemented with O(1) update performance
//! - **Zero-copy Memory**: Uses alloc_with_nan_prefix and make_uninit_matrix for batch operations

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

impl<'a> AsRef<[f64]> for MomInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MomData::Slice(slice) => slice,
            MomData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MomData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MomOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct MomParams {
    pub period: Option<usize>,
}

impl Default for MomParams {
    fn default() -> Self {
        Self { period: Some(10) }
    }
}

#[derive(Debug, Clone)]
pub struct MomInput<'a> {
    pub data: MomData<'a>,
    pub params: MomParams,
}

impl<'a> MomInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MomParams) -> Self {
        Self {
            data: MomData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MomParams) -> Self {
        Self {
            data: MomData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MomParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(10)
    }
}

#[derive(Clone, Debug)]
pub struct MomBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for MomBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MomBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<MomOutput, MomError> {
        let p = MomParams {
            period: self.period,
        };
        let i = MomInput::from_candles(c, "close", p);
        mom_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MomOutput, MomError> {
        let p = MomParams {
            period: self.period,
        };
        let i = MomInput::from_slice(d, p);
        mom_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<MomStream, MomError> {
        let p = MomParams {
            period: self.period,
        };
        MomStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum MomError {
    #[error("mom: Input data slice is empty.")]
    EmptyInputData,
    #[error("mom: All values are NaN.")]
    AllValuesNaN,

    #[error("mom: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("mom: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn mom(input: &MomInput) -> Result<MomOutput, MomError> {
    mom_with_kernel(input, Kernel::Auto)
}

// New helper function (mirror ALMA pattern)
#[inline(always)]
fn mom_prepare<'a>(
    input: &'a MomInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, Kernel), MomError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(MomError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MomError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(MomError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(MomError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };
    Ok((data, period, first, chosen))
}

#[inline(always)]
fn mom_compute_into(data: &[f64], period: usize, first: usize, kernel: Kernel, out: &mut [f64]) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => mom_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => mom_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => mom_avx512(data, period, first, out),
            _ => unreachable!(),
        }
    }
}

// Refactored mom_with_kernel to use the helpers
pub fn mom_with_kernel(input: &MomInput, kernel: Kernel) -> Result<MomOutput, MomError> {
    let (data, period, first, chosen) = mom_prepare(input, kernel)?;
    let warm = first + period;
    let mut out = alloc_with_nan_prefix(data.len(), warm);
    mom_compute_into(data, period, first, chosen, &mut out);
    Ok(MomOutput { values: out })
}

#[inline(always)]
pub fn mom_scalar(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    for i in (first_valid + period)..data.len() {
        out[i] = data[i] - data[i - period];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn mom_avx512(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    unsafe {
        if period <= 32 {
            mom_avx512_short(data, period, first_valid, out);
        } else {
            mom_avx512_long(data, period, first_valid, out);
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn mom_avx512_short(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    // For API parity; fallback to scalar logic
    mom_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn mom_avx512_long(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    // For API parity; fallback to scalar logic
    mom_scalar(data, period, first_valid, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mom_avx2(data: &[f64], period: usize, first_valid: usize, out: &mut [f64]) {
    // For API parity; fallback to scalar logic
    mom_scalar(data, period, first_valid, out)
}

#[derive(Debug, Clone)]
pub struct MomStream {
    period: usize,
    buffer: Vec<f64>,
    head: usize,
    filled: bool,
}

impl MomStream {
    pub fn try_new(params: MomParams) -> Result<Self, MomError> {
        let period = params.period.unwrap_or(10);
        if period == 0 {
            return Err(MomError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            head: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let prev = self.buffer[self.head];
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;

        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        Some(value - prev)
    }
}

#[derive(Clone, Debug)]
pub struct MomBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for MomBatchRange {
    fn default() -> Self {
        Self {
            period: (10, 10, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MomBatchBuilder {
    range: MomBatchRange,
    kernel: Kernel,
}

impl MomBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline]
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    #[inline]
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<MomBatchOutput, MomError> {
        mom_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MomBatchOutput, MomError> {
        MomBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MomBatchOutput, MomError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MomBatchOutput, MomError> {
        MomBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn mom_batch_with_kernel(
    data: &[f64],
    sweep: &MomBatchRange,
    k: Kernel,
) -> Result<MomBatchOutput, MomError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(MomError::InvalidPeriod {
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
    mom_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MomBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MomParams>,
    pub rows: usize,
    pub cols: usize,
}
impl MomBatchOutput {
    pub fn row_for_params(&self, p: &MomParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(10) == p.period.unwrap_or(10))
    }

    pub fn values_for(&self, p: &MomParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &MomBatchRange) -> Vec<MomParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(MomParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn mom_batch_slice(
    data: &[f64],
    sweep: &MomBatchRange,
    kern: Kernel,
) -> Result<MomBatchOutput, MomError> {
    mom_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn mom_batch_par_slice(
    data: &[f64],
    sweep: &MomBatchRange,
    kern: Kernel,
) -> Result<MomBatchOutput, MomError> {
    mom_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn mom_batch_inner(
    data: &[f64],
    sweep: &MomBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MomBatchOutput, MomError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MomError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let len = data.len();
    if len == 0 {
        return Err(MomError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MomError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(MomError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;

    // allocate uninit and write NaN warm prefixes
    let mut buf_mu = make_uninit_matrix(rows, cols);
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // keep MaybeUninit for compute to avoid any UB window
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out_mu: &mut [std::mem::MaybeUninit<f64>] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr(), guard.len()) };

    let simd = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        k if !k.is_batch() => k,
        _ => unreachable!(),
    };

    let do_row = |row: usize, row_mu: &mut [std::mem::MaybeUninit<f64>]| {
        let period = combos[row].period.unwrap();
        let dst = unsafe {
            core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };
        mom_compute_into(data, period, first, simd, dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_mu
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, mu)| do_row(r, mu));
        #[cfg(target_arch = "wasm32")]
        for (r, mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, mu);
        }
    } else {
        for (r, mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, mu);
        }
    }

    // materialize Vec<f64> without copy
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(MomBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn mom_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    for i in (first + period)..data.len() {
        out[i] = data[i] - data[i - period];
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mom_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    mom_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mom_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        mom_row_avx512_short(data, first, period, out);
    } else {
        mom_row_avx512_long(data, first, period, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mom_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    mom_row_scalar(data, first, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn mom_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    mom_row_scalar(data, first, period, out)
}

// Helper function for batch operations writing directly to output
#[inline(always)]
pub fn mom_batch_inner_into(
    data: &[f64],
    sweep: &MomBatchRange,
    kern: Kernel,
    parallel: bool,
    output: &mut [f64],
) -> Result<Vec<MomParams>, MomError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MomError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let len = data.len();
    if len == 0 {
        return Err(MomError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MomError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if len - first < max_p {
        return Err(MomError::NotEnoughValidData {
            needed: max_p,
            valid: len - first,
        });
    }

    let rows = combos.len();
    let cols = len;
    assert_eq!(output.len(), rows * cols, "output length mismatch");

    // view caller buffer as MaybeUninit and write NaN warm prefixes via helper
    let out_mu: &mut [std::mem::MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(
            output.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            output.len(),
        )
    };
    let warm: Vec<usize> = combos.iter().map(|c| first + c.period.unwrap()).collect();
    init_matrix_prefixes(out_mu, cols, &warm);

    let simd = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        k if !k.is_batch() => k,
        _ => unreachable!(),
    };

    let do_row = |row: usize, row_mu: &mut [std::mem::MaybeUninit<f64>]| {
        let period = combos[row].period.unwrap();
        let dst = unsafe {
            core::slice::from_raw_parts_mut(row_mu.as_mut_ptr() as *mut f64, row_mu.len())
        };
        mom_compute_into(data, period, first, simd, dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        out_mu
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, mu)| do_row(r, mu));
        #[cfg(target_arch = "wasm32")]
        for (r, mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, mu);
        }
    } else {
        for (r, mu) in out_mu.chunks_mut(cols).enumerate() {
            do_row(r, mu);
        }
    }

    Ok(combos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_mom_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = MomParams { period: None };
        let input = MomInput::from_candles(&candles, "close", default_params);
        let output = mom_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_mom_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = MomInput::from_candles(&candles, "close", MomParams::default());
        let result = mom_with_kernel(&input, kernel)?;
        let expected_last_five = [-134.0, -331.0, -194.0, -294.0, -896.0];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] MOM {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_mom_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = MomInput::with_default_candles(&candles);
        match input.data {
            MomData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected MomData::Candles"),
        }
        let output = mom_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_mom_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = MomParams { period: Some(0) };
        let input = MomInput::from_slice(&input_data, params);
        let res = mom_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MOM should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_mom_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = MomParams { period: Some(10) };
        let input = MomInput::from_slice(&data_small, params);
        let res = mom_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MOM should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_mom_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = MomParams { period: Some(9) };
        let input = MomInput::from_slice(&single_point, params);
        let res = mom_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MOM should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_mom_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = MomParams { period: Some(14) };
        let first_input = MomInput::from_candles(&candles, "close", first_params);
        let first_result = mom_with_kernel(&first_input, kernel)?;

        let second_params = MomParams { period: Some(14) };
        let second_input = MomInput::from_slice(&first_result.values, second_params);
        let second_result = mom_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "[{}] MOM Slice Reinput {:?} unexpected NaN at idx {}",
                test_name,
                kernel,
                i
            );
        }
        Ok(())
    }

    fn check_mom_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = MomInput::from_candles(&candles, "close", MomParams { period: Some(10) });
        let res = mom_with_kernel(&input, kernel)?;
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

    fn check_mom_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 10;
        let input = MomInput::from_candles(
            &candles,
            "close",
            MomParams {
                period: Some(period),
            },
        );
        let batch_output = mom_with_kernel(&input, kernel)?.values;

        let mut stream = MomStream::try_new(MomParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] MOM streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_mom_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            MomParams::default(),            // period: 10
            MomParams { period: Some(2) },   // minimum viable
            MomParams { period: Some(5) },   // small
            MomParams { period: Some(7) },   // small
            MomParams { period: Some(14) },  // medium
            MomParams { period: Some(20) },  // medium
            MomParams { period: Some(50) },  // large
            MomParams { period: Some(100) }, // very large
            MomParams { period: Some(200) }, // edge case large
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = MomInput::from_candles(&candles, "close", params.clone());
            let output = mom_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(10),
                        param_idx
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(10),
                        param_idx
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 with params: period={} (param set {})",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(10),
                        param_idx
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_mom_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_mom_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        // Strategy: generate period from 1 to 64, then data with length from period to 400
        let strat = (1usize..=64).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = MomParams {
                    period: Some(period),
                };
                let input = MomInput::from_slice(&data, params.clone());

                // Get output from the kernel under test
                let MomOutput { values: out } = mom_with_kernel(&input, kernel).unwrap();

                // Get reference output from scalar kernel for consistency check
                let MomOutput { values: ref_out } =
                    mom_with_kernel(&input, Kernel::Scalar).unwrap();

                // Find first valid index (first non-NaN in data)
                let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
                let warmup_period = first_valid + period;

                // Property 1: Warmup period - first 'warmup_period' values should be NaN
                for i in 0..warmup_period.min(out.len()) {
                    prop_assert!(
                        out[i].is_nan(),
                        "[{}] Expected NaN during warmup at index {}, got {}",
                        test_name,
                        i,
                        out[i]
                    );
                }

                // Property 2: Exact formula - momentum[i] = data[i] - data[i - period]
                for i in warmup_period..data.len() {
                    let expected = data[i] - data[i - period];
                    let actual = out[i];

                    if expected.is_finite() && actual.is_finite() {
                        prop_assert!(
                            (actual - expected).abs() < 1e-10,
                            "[{}] MOM formula mismatch at index {}: expected {}, got {}",
                            test_name,
                            i,
                            expected,
                            actual
                        );
                    }
                }

                // Property 3: Kernel consistency - all kernels should produce identical results
                for i in 0..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Check bit-exact equality for NaN/infinite values
                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "[{}] NaN/Inf mismatch at index {}: {} vs {}",
                            test_name,
                            i,
                            y,
                            r
                        );
                    } else {
                        // For finite values, allow small numerical difference or ULP difference
                        let ulp_diff = y.to_bits().abs_diff(r.to_bits());
                        prop_assert!(
                            (y - r).abs() <= 1e-10 || ulp_diff <= 4,
                            "[{}] Kernel mismatch at index {}: {} vs {} (ULP diff: {})",
                            test_name,
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }
                }

                // Property 4: Constant data produces zero momentum
                let all_same = data.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
                if all_same && data.len() > period {
                    for i in warmup_period..out.len() {
                        prop_assert!(
                            out[i].abs() < 1e-10,
                            "[{}] Constant data should produce zero momentum at index {}, got {}",
                            test_name,
                            i,
                            out[i]
                        );
                    }
                }

                // Property 5: Linear data produces constant momentum
                // Check if data is approximately linear (consecutive differences are constant)
                if data.len() > period + 2 {
                    let diffs: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
                    let is_linear = diffs.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9);

                    if is_linear && diffs.len() > 0 {
                        let expected_momentum = diffs[0] * period as f64;
                        for i in warmup_period..out.len().min(warmup_period + 10) {
                            if out[i].is_finite() {
                                prop_assert!(
									(out[i] - expected_momentum).abs() < 1e-9,
									"[{}] Linear data should produce constant momentum {} at index {}, got {}",
									test_name, expected_momentum, i, out[i]
								);
                            }
                        }
                    }
                }

                // Property 6: Symmetry - negating input negates output
                if data.len() < 100 {
                    // Only test on smaller datasets for efficiency
                    let neg_data: Vec<f64> = data.iter().map(|&x| -x).collect();
                    let neg_input = MomInput::from_slice(&neg_data, params);
                    let MomOutput { values: neg_out } =
                        mom_with_kernel(&neg_input, kernel).unwrap();

                    for i in warmup_period..out.len() {
                        if out[i].is_finite() && neg_out[i].is_finite() {
                            prop_assert!(
                                (out[i] + neg_out[i]).abs() < 1e-10,
                                "[{}] Symmetry violated at index {}: {} vs {}",
                                test_name,
                                i,
                                out[i],
                                -neg_out[i]
                            );
                        }
                    }
                }

                // Property 7: Period=1 gives adjacent differences
                if period == 1 && data.len() > 1 {
                    for i in 1..data.len() {
                        let expected = data[i] - data[i - 1];
                        if out[i].is_finite() && expected.is_finite() {
                            prop_assert!(
								(out[i] - expected).abs() < 1e-10,
								"[{}] Period=1 should give adjacent differences at index {}: expected {}, got {}",
								test_name, i, expected, out[i]
							);
                        }
                    }
                }

                // Property 8: Output magnitude bounded by max difference in data
                if data.len() > period {
                    let min_val = data
                        .iter()
                        .filter(|x| x.is_finite())
                        .fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_val = data
                        .iter()
                        .filter(|x| x.is_finite())
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let max_diff = max_val - min_val;

                    for i in warmup_period..out.len() {
                        if out[i].is_finite() {
                            prop_assert!(
								out[i].abs() <= max_diff + 1e-9,
								"[{}] Output magnitude {} exceeds max possible difference {} at index {}",
								test_name, out[i].abs(), max_diff, i
							);
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_mom_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test]
                fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $( #[test]
                fn [<$test_fn _avx2_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                }
                #[test]
                fn [<$test_fn _avx512_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                })*
            }
        }
    }

    generate_all_mom_tests!(
        check_mom_partial_params,
        check_mom_accuracy,
        check_mom_default_candles,
        check_mom_zero_period,
        check_mom_period_exceeds_length,
        check_mom_very_small_dataset,
        check_mom_reinput,
        check_mom_nan_handling,
        check_mom_streaming,
        check_mom_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_mom_tests!(check_mom_property);

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = MomBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = MomParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [-134.0, -331.0, -194.0, -294.0, -896.0];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (2, 10, 2),     // Small periods
            (5, 25, 5),     // Medium periods
            (30, 60, 15),   // Large periods
            (2, 5, 1),      // Dense small range
            (10, 50, 10),   // Wide range with medium step
            (100, 200, 50), // Very large periods
        ];

        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = MomBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                // Check all three poison patterns with detailed context
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						 at row {} col {} (flat index {}) with params: period={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.period.unwrap_or(10)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
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
    gen_batch_tests!(check_batch_no_poison);
}

/// Core helper for zero-allocation WASM API
/// Writes directly to output slice - no intermediate allocations
#[inline(always)]
pub fn mom_into_slice(dst: &mut [f64], input: &MomInput, kernel: Kernel) -> Result<(), MomError> {
    let (data, period, first, chosen) = mom_prepare(input, kernel)?;
    if dst.len() != data.len() {
        return Err(MomError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }
    let warm = first + period;
    for v in &mut dst[..warm] {
        *v = f64::NAN;
    }
    mom_compute_into(data, period, first, chosen, dst);
    Ok(())
}

// WASM bindings
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// WASM Safe API
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mom_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = MomParams {
        period: Some(period),
    };
    let input = MomInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()]; // Single allocation
    mom_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

// WASM Fast API with aliasing detection
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mom_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = MomParams {
            period: Some(period),
        };
        let input = MomInput::from_slice(data, params);

        if in_ptr == out_ptr {
            // CRITICAL: Aliasing check
            let mut temp = vec![0.0; len];
            mom_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            mom_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        Ok(())
    }
}

// WASM Memory Management
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mom_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mom_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

// WASM Batch API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MomBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct MomBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<MomParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = mom_batch)]
pub fn mom_batch_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: MomBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = MomBatchRange {
        period: config.period_range,
    };

    let output = mom_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = MomBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mom_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to mom_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = MomBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        // Use mom_batch_inner_into for direct output
        mom_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

// Python bindings
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction(name = "mom")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn mom_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = MomParams {
        period: Some(period),
    };
    let input = MomInput::from_slice(slice_in, params);

    let result_vec: Vec<f64> = py
        .allow_threads(|| mom_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "mom_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn mom_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = MomBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
            let kernel = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };

            // For mom, we use mom_batch_inner_into
            mom_batch_inner_into(slice_in, &sweep, kernel, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

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
#[pyclass(name = "MomStream")]
pub struct MomStreamPy {
    inner: MomStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl MomStreamPy {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> {
        let params = MomParams {
            period: Some(period),
        };
        let inner = MomStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(MomStreamPy { inner })
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.inner.update(value)
    }
}

#[cfg(feature = "python")]
pub fn register_mom_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mom_py, m)?)?;
    m.add_function(wrap_pyfunction!(mom_batch_py, m)?)?;
    Ok(())
}
