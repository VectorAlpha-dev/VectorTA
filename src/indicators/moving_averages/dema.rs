//! # Double Exponential Moving Average (DEMA)
//!
//! A moving average technique that seeks to reduce lag by combining two
//! exponential moving averages (EMA). First, an EMA is calculated on the input
//! data. Then, a second EMA is computed on the first EMA. Finally, the DEMA is
//! determined by subtracting the second EMA from twice the first EMA.
//!
//! ## Parameters
//! - **period**: Lookback period for the EMA calculations (must be ≥ 1).
//!
//! ## Errors
//! - **AllValuesNaN**: DEMA: All input data values are `NaN`.
//! - **InvalidPeriod**: DEMA: `period` is less than 1 or exceeds the data length.
//! - **NotEnoughData**: DEMA: Not enough data points (needs `2 * (period - 1)`).
//!
//! ## Returns
//! - **`Ok(DemaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(DemaError)`** otherwise.

//! Scalar Only

use crate::utilities::aligned_vector::AlignedVec;
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DemaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for DemaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            DemaData::Slice(slice) => slice,
            DemaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DemaParams {
    pub period: Option<usize>,
}

impl Default for DemaParams {
    fn default() -> Self {
        Self { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct DemaInput<'a> {
    pub data: DemaData<'a>,
    pub params: DemaParams,
}

impl<'a> DemaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: DemaParams) -> Self {
        Self {
            data: DemaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: DemaParams) -> Self {
        Self {
            data: DemaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", DemaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(30)
    }
}

#[derive(Debug, Clone)]
pub struct DemaOutput {
    pub values: Vec<f64>,
}

#[derive(Copy, Clone, Debug)]
pub struct DemaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for DemaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl DemaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<DemaOutput, DemaError> {
        let p = DemaParams {
            period: self.period,
        };
        let i = DemaInput::from_candles(c, "close", p);
        dema_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<DemaOutput, DemaError> {
        let p = DemaParams {
            period: self.period,
        };
        let i = DemaInput::from_slice(d, p);
        dema_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<DemaStream, DemaError> {
        let p = DemaParams {
            period: self.period,
        };
        DemaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum DemaError {
    #[error("dema: Input data slice is empty.")]
    EmptyInputData,
    #[error("dema: All values are NaN.")]
    AllValuesNaN,

    #[error("dema: Invalid period: period = {period}")]
    InvalidPeriod { period: usize },

    #[error("dema: Not enough data: needed = {needed}, valid = {valid}")]
    NotEnoughData { needed: usize, valid: usize },

    #[error("dema: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn dema(input: &DemaInput) -> Result<DemaOutput, DemaError> {
    dema_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn dema_prepare<'a>(
    input: &'a DemaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, Kernel), DemaError> {
    let data: &[f64] = match &input.data {
        DemaData::Candles { candles, source } => source_type(candles, source),
        DemaData::Slice(sl) => sl,
    };

    let len = data.len();
    if len == 0 {
        return Err(DemaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DemaError::AllValuesNaN)?;

    let period = input.get_period();

    if period < 1 || period > len {
        return Err(DemaError::InvalidPeriod { period });
    }
    let needed = 2 * (period - 1);
    if len < needed {
        return Err(DemaError::NotEnoughData { needed, valid: len });
    }
    let valid = len - first;
    if valid < needed {
        return Err(DemaError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm = first + period - 1;

    Ok((data, period, first, warm, chosen))
}

#[inline(always)]
fn dema_compute_into(
    data: &[f64],
    period: usize,
    first: usize,
    chosen: Kernel,
    out: &mut [f64],
) {
    unsafe {
        match chosen {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => dema_avx512(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => dema_avx2(data, period, first, out),
            _ => dema_scalar(data, period, first, out),
        }
    }
}

pub fn dema_with_kernel(input: &DemaInput, kernel: Kernel) -> Result<DemaOutput, DemaError> {
    let (data, period, first, warm, chosen) = dema_prepare(input, kernel)?;
    let len = data.len();
    let mut out = alloc_with_nan_prefix(len, warm);
    dema_compute_into(data, period, first, chosen, &mut out);
    Ok(DemaOutput { values: out })
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
#[inline]
pub unsafe fn dema_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    debug_assert!(period >= 1 && data.len() == out.len());
    if first >= data.len() {
        return;
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let alpha_1 = 1.0 - alpha;
    let n = data.len();

    let mut p = data.as_ptr().add(first);
    let mut q = out.as_mut_ptr().add(first);

    let mut ema = *p;
    let mut ema2 = ema;
    *q = ema;

    for i in (first + 1)..n {
        p = p.add(1);
        q = q.add(1);
        if i + 8 < n {
            core::arch::x86_64::_mm_prefetch(
                p.add(8) as *const i8,
                core::arch::x86_64::_MM_HINT_T0,
            );
        }
        let price = *p;
        ema = ema.mul_add(alpha_1, price * alpha);
        ema2 = ema2.mul_add(alpha_1, ema * alpha);

        *q = (2.0 * ema) - ema2;
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
pub unsafe fn dema_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    debug_assert!(period >= 1 && data.len() == out.len());
    if first >= data.len() {
        return;
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let alpha_1 = 1.0 - alpha;
    let n = data.len();

    let mut p = data.as_ptr().add(first);
    let mut q = out.as_mut_ptr().add(first);

    let mut ema = *p;
    let mut ema2 = ema;
    *q = ema;

    for i in (first + 1)..n {
        p = p.add(1);
        q = q.add(1);
        
        let price = *p;
        // Note: This uses regular multiplication instead of mul_add
        // which may be less accurate but works on all architectures
        ema = ema * alpha_1 + price * alpha;
        ema2 = ema2 * alpha_1 + ema * alpha;

        *q = (2.0 * ema) - ema2;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn dema_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    dema_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512dq,fma")]
pub unsafe fn dema_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    dema_scalar(data, period, first, out);
}

#[derive(Debug, Clone)]
pub struct DemaStream {
    period: usize,
    alpha: f64,
    alpha_1: f64,
    ema: f64,
    ema2: f64,
    filled: usize,
    nan_fill: usize,
}

impl DemaStream {
    pub fn try_new(params: DemaParams) -> Result<Self, DemaError> {
        let period = params.period.unwrap_or(30);
        if period < 1 {
            return Err(DemaError::InvalidPeriod { period });
        }
        Ok(Self {
            period,
            alpha: 2.0 / (period as f64 + 1.0),
            alpha_1: 1.0 - 2.0 / (period as f64 + 1.0),
            ema: f64::NAN,
            ema2: f64::NAN,
            filled: 0,
            nan_fill: period - 1,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        if self.filled == 0 {
            self.ema  = value;
            self.ema2 = value;
        
            } else {
            self.ema  = self.ema * self.alpha_1 + value * self.alpha;
            self.ema2 = self.ema2 * self.alpha_1 + self.ema  * self.alpha;
        }
        let out = if self.filled >= self.nan_fill {
            (2.0 * self.ema) - self.ema2
        
            } else {
            f64::NAN
        };

        self.filled += 1;
        Some(out)
    }
}

#[derive(Clone, Debug)]
pub struct DemaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for DemaBatchRange {
    fn default() -> Self {
        Self {
            period: (30, 240, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DemaBatchBuilder {
    range: DemaBatchRange,
    kernel: Kernel,
}

impl DemaBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<DemaBatchOutput, DemaError> {
        dema_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<DemaBatchOutput, DemaError> {
        DemaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<DemaBatchOutput, DemaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<DemaBatchOutput, DemaError> {
        DemaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub struct DemaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<DemaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl DemaBatchOutput {
    pub fn row_for_params(&self, p: &DemaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(30) == p.period.unwrap_or(30))
    }
    pub fn values_for(&self, p: &DemaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &DemaBatchRange) -> Vec<DemaParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(DemaParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn dema_batch_slice(
    data: &[f64],
    sweep: &DemaBatchRange,
    kern: Kernel,
) -> Result<DemaBatchOutput, DemaError> {
    dema_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn dema_batch_par_slice(
    data: &[f64],
    sweep: &DemaBatchRange,
    kern: Kernel,
) -> Result<DemaBatchOutput, DemaError> {
    dema_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn dema_batch_with_kernel(
    data: &[f64],
    sweep: &DemaBatchRange,
    k: Kernel,
) -> Result<DemaBatchOutput, DemaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(DemaError::InvalidPeriod { period: 0 }),
    };
    let simd = match kernel {
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512Batch => Kernel::Avx512,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    dema_batch_par_slice(data, sweep, simd)
}


#[inline(always)]
fn dema_batch_inner(
    data: &[f64],
    sweep: &DemaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<DemaBatchOutput, DemaError> {
    let combos = expand_grid(sweep);
    let cols = data.len();
    let rows = combos.len();
    let mut values = vec![0.0; rows * cols];
    
    dema_batch_inner_into(data, sweep, kern, parallel, &mut values)?;
    
    Ok(DemaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn dema_batch_inner_into(
    data: &[f64],
    sweep: &DemaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<DemaParams>, DemaError> {
    // ── 1. validation ──────────────────────────────────────────────────────
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(DemaError::InvalidPeriod { period: 0 });
    }

    if data.is_empty() {
        return Err(DemaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(DemaError::AllValuesNaN)?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    let needed = 2 * (max_p - 1);
    if data.len() < needed {
        return Err(DemaError::NotEnoughData {
            needed,
            valid: data.len(),
        });
    }
    let valid = data.len() - first;
    if valid < needed {
        return Err(DemaError::NotEnoughValidData { needed, valid });
    }

    let rows = combos.len();
    let cols = data.len();
    
    // Verify output buffer size
    if out.len() != rows * cols {
        return Err(DemaError::InvalidPeriod { period: 0 }); // TODO: better error
    }

    // ── 2. Initialize NaN warm-ups using efficient method ────────────────────
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + 2 * (c.period.unwrap() - 1))  // DEMA needs 2*(period-1) warmup
        .collect();
    
    // SAFETY: We're reinterpreting the output slice as MaybeUninit to use the efficient
    // init_matrix_prefixes function. This is safe because:
    // 1. MaybeUninit<T> has the same layout as T
    // 2. We ensure all values are written before the slice is used again
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut MaybeUninit<f64>,
            out.len()
        )
    };

    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    // ── 3. per-row kernel closure; *dst_mu* is &mut [MaybeUninit<f64>] ─────
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let p = combos[row].period.unwrap();

        // Cast just this slice to &mut [f64]
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => dema_row_avx512(data, first, p, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => dema_row_avx2(data, first, p, dst),
            _ => dema_row_scalar(data, first, p, dst),
        }
    };

    // ── 4. run every row kernel, parallel or sequential ────────────────────
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
#[inline(always)]
unsafe fn dema_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dema_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dema_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dema_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn dema_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    dema_scalar(data, period, first, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use proptest::prelude::*;

    fn check_dema_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = DemaParams { period: None };
        let input_default = DemaInput::from_candles(&candles, "close", default_params);
        let output_default = dema_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = DemaParams { period: Some(14) };
        let input_period_14 = DemaInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 = dema_with_kernel(&input_period_14, kernel)?;
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = DemaParams { period: Some(20) };
        let input_custom = DemaInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = dema_with_kernel(&input_custom, kernel)?;
        assert_eq!(output_custom.values.len(), candles.close.len());
        Ok(())
    }

    fn check_dema_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = DemaInput::with_default_candles(&candles);
        let result = dema_with_kernel(&input, kernel)?;

        let expected_last_five = [
            59189.73193987478,
            59129.24920772847,
            59058.80282420511,
            59011.5555611042,
            58908.370159946775,
        ];
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-6,
                "DEMA mismatch at index {}: expected {}, got {}",
                start_index + i,
                exp,
                val
            );
        }
        Ok(())
    }

    fn check_dema_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = DemaInput::with_default_candles(&candles);
        match input.data {
            DemaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected DemaData::Candles"),
        }
        assert_eq!(input.params.period, Some(30));
        let output = dema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_dema_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = DemaParams { period: Some(0) };
        let input = DemaInput::from_slice(&input_data, params);
        let result = dema_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_dema_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = DemaParams { period: Some(10) };
        let input = DemaInput::from_slice(&data_small, params);
        let result = dema_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_dema_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = DemaParams { period: Some(9) };
        let input = DemaInput::from_slice(&single_point, params);
        let result = dema_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_dema_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = DemaParams { period: Some(80) };
        let first_input = DemaInput::from_candles(&candles, "close", first_params);
        let first_result = dema_with_kernel(&first_input, kernel)?;

        let second_params = DemaParams { period: Some(60) };
        let second_input = DemaInput::from_slice(&first_result.values, second_params);
        let second_result = dema_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(!second_result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_dema_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = DemaParams { period: Some(30) };
        let input = DemaInput::from_candles(&candles, "close", params);
        let result = dema_with_kernel(&input, kernel)?;
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
        Ok(())
    }

    fn check_dema_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = DemaInput::from_slice(&empty, DemaParams::default());
        let res = dema_with_kernel(&input, kernel);
        assert!(matches!(res, Err(DemaError::EmptyInputData)));
        Ok(())
    }

    fn check_dema_not_enough_valid(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [f64::NAN, f64::NAN, 1.0, 2.0];
        let params = DemaParams { period: Some(3) };
        let input = DemaInput::from_slice(&data, params);
        let res = dema_with_kernel(&input, kernel);
        assert!(matches!(res, Err(DemaError::NotEnoughValidData { .. })));
        Ok(())
    }

    #[allow(clippy::float_cmp)]
    fn check_dema_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        use float_cmp::approx_eq;

        skip_if_unsupported!(kernel, test_name);

        /* 1 ─ Strategy: choose period first, then generate a ≥-warm-up finite vector,
            plus random affine parameters (a ≠ 0, b). */
        let strat = (1usize..=32).prop_flat_map(|period| {
            let min_len = 2 * period.max(2);            // enough for DEMA warm-up
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64)
                        .prop_filter("finite", |x| x.is_finite()),
                    min_len..400,
                ),
                Just(period),
                (-1e3f64..1e3f64)
                    .prop_filter("non-zero scale", |a| a.is_finite() && *a != 0.0),
                -1e3f64..1e3f64,                        // b may be zero
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, a, b)| {
                let params = DemaParams { period: Some(period) };
                let input  = DemaInput::from_slice(&data, params.clone());

                /* --- run both kernels (fast & scalar) --------------------------- */
                let fast = dema_with_kernel(&input, kernel);
                let slow = dema_with_kernel(&input, Kernel::Scalar);

                match (fast, slow) {
                    /* ➊ Same error kind ⇒ property holds. */
                    (Err(e1), Err(e2))
                        if std::mem::discriminant(&e1) == std::mem::discriminant(&e2) =>
                        return Ok(()),
                    /* ➊′ Different error kinds → fail. */
                    (Err(e1), Err(e2)) =>
                        prop_assert!(false, "different errors: fast={:?} slow={:?}", e1, e2),
                    /* ➋ Kernels disagree on success / error. */
                    (Err(e1), Ok(_))   =>
                        prop_assert!(false, "fast errored {e1:?} but scalar succeeded"),
                    (Ok(_),   Err(e2)) =>
                        prop_assert!(false, "scalar errored {e2:?} but fast succeeded"),

                    /* ➌ Both succeeded – run invariant suite. */
                    (Ok(fast), Ok(reference)) => {
                        let DemaOutput { values: out  } = fast;
                        let DemaOutput { values: rref } = reference;

                        /* Streaming version (for parity check) */
                        let mut stream = DemaStream::try_new(params.clone()).unwrap();
                        let mut s_out  = Vec::with_capacity(data.len());
                        for &v in &data {
                            s_out.push(stream.update(v).unwrap_or(f64::NAN));
                        }

                        /* Affine-transformed run */
                        let transformed: Vec<f64> =
                            data.iter().map(|x| a * *x + b).collect();
                        let t_out = dema(
                            &DemaInput::from_slice(&transformed, params.clone())
                        )?.values;

                        /* -------- core invariants -------------------------------- */
                        let nan_fill = period -1 ;        // streaming warm-up
                        for i in 0..data.len() {
                            let y  = out[i];
                            let yr = rref[i];
                            let ys = s_out[i];
                            let yt = t_out[i];

                            /* 1️⃣ Period-1 identity */
                            if period == 1 && y.is_finite() {
                                prop_assert!(approx_eq!(f64, y, data[i], ulps = 2));
                            }

                            /* 2️⃣ Constant-series invariance (when the window is flat) */
                            let window = &data[i.saturating_sub(period - 1)..=i];
                            if window.iter().all(|v| *v == window[0]) {
                                prop_assert!(approx_eq!(f64, y, window[0], epsilon = 1e-9));
                            }

                            /* 3️⃣ Affine equivariance */
                            if i >= nan_fill {             // compare only after warm-ups
                                if y.is_finite() {
                                    let expected = a * y + b;
                                    let diff     = (yt - expected).abs();
                                    let tol      = 1e-9_f64.max(expected.abs() * 1e-9);
                                    let ulp      = yt.to_bits().abs_diff(expected.to_bits());
                                    prop_assert!(
                                        diff <= tol || ulp <= 8,
                                        "idx {i}: affine mismatch diff={diff:e}  ULP={ulp}"
                                    );
                                } else {
                                    prop_assert_eq!(
                                        y.to_bits(),
                                        yt.to_bits(),
                                        "idx {}: special-value mismatch under affine map",
                                        i
                                    );
                                }
                            }

                            /* 4️⃣ Scalar ≡ fast (ULP ≤ 4 or abs ≤ 1e-9) */
                            let ulp = y.to_bits().abs_diff(yr.to_bits());
                            prop_assert!(
                                (y - yr).abs() <= 1e-9 || ulp <= 4,
                                "idx {i}: fast={y} ref={yr} ULP={ulp}"
                            );

                            /* 5️⃣ Streaming parity (after nan_fill) */
                            if i >= nan_fill {
                                prop_assert!(
                                    (y - ys).abs() <= 1e-9 || (y.is_nan() && ys.is_nan()),
                                    "idx {i}: stream mismatch"
                                );
                            }
                        }
                    }
                }

                Ok(())
            })
            .unwrap();

        /* 🔟  Error-path smoke tests (keep suite uniform) ------------------------- */
        assert!(dema(&DemaInput::from_slice(&[], DemaParams::default())).is_err());
        assert!(dema(&DemaInput::from_slice(&[f64::NAN; 12], DemaParams::default())).is_err());
        assert!(dema(&DemaInput::from_slice(&[1.0; 5], DemaParams { period: Some(12) })).is_err());
        assert!(dema(&DemaInput::from_slice(&[1.0; 5], DemaParams { period: Some(0)  })).is_err());

        Ok(())
    }


    fn check_dema_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 30;
        let input = DemaInput::from_candles(
            &candles,
            "close",
            DemaParams {
                period: Some(period),
            },
        );
        let batch_output = dema_with_kernel(&input, kernel)?.values;

        let mut stream = DemaStream::try_new(DemaParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            stream_values.push(stream.update(price).unwrap());
        }

        assert_eq!(batch_output.len(), stream_values.len());

        for (i, (&b, &s)) in batch_output
            .iter()
            .zip(&stream_values)
            .enumerate()
            .skip(period)
        {
            if b.is_nan() && s.is_nan() {
                continue;
            }

            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] DEMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_dema_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        // Test multiple parameter combinations to better catch uninitialized memory bugs
        let test_params = vec![
            // Default parameters
            DemaParams::default(),
            // Small periods
            DemaParams { period: Some(2) },
            DemaParams { period: Some(3) },
            DemaParams { period: Some(5) },
            // Medium periods
            DemaParams { period: Some(7) },
            DemaParams { period: Some(10) },
            DemaParams { period: Some(12) },
            DemaParams { period: Some(20) },
            DemaParams { period: Some(30) },
            // Large periods
            DemaParams { period: Some(50) },
            DemaParams { period: Some(100) },
            DemaParams { period: Some(200) },
            // Edge cases
            DemaParams { period: Some(1) },
            DemaParams { period: Some(250) },
        ];
        
        for (param_idx, params) in test_params.iter().enumerate() {
            let input = DemaInput::from_candles(&candles, "close", params.clone());
            let output = dema_with_kernel(&input, kernel)?;
            
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
                        with params: period={}",
                        test_name, val, bits, i,
                        params.period.unwrap_or(30)
                    );
                }
                
                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name, val, bits, i,
                        params.period.unwrap_or(30)
                    );
                }
                
                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: period={}",
                        test_name, val, bits, i,
                        params.period.unwrap_or(30)
                    );
                }
            }
        }
        
        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_dema_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    macro_rules! generate_all_dema_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                    #[test]
                    fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }

    generate_all_dema_tests!(
        check_dema_partial_params,
        check_dema_accuracy,
        check_dema_default_candles,
        check_dema_zero_period,
        check_dema_period_exceeds_length,
        check_dema_very_small_dataset,
        check_dema_empty_input,
        check_dema_not_enough_valid,
        check_dema_reinput,
        check_dema_nan_handling,
        check_dema_streaming,
        check_dema_property,
        check_dema_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = DemaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = DemaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59189.73193987478,
            59129.24920772847,
            59058.80282420511,
            59011.5555611042,
            58908.370159946775,
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

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        
        // Test multiple batch configurations to better catch uninitialized memory bugs
        let test_configs = vec![
            // Small range
            (2, 5, 1),      // periods: 2, 3, 4, 5
            // Medium range with gaps
            (5, 25, 5),     // periods: 5, 10, 15, 20, 25
            // Large range
            (10, 50, 10),   // periods: 10, 20, 30, 40, 50
            // Edge case: very small periods
            (1, 3, 1),      // periods: 1, 2, 3
            // Edge case: large periods
            (50, 150, 25),  // periods: 50, 75, 100, 125, 150
            // Dense range
            (10, 30, 2),    // periods: 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
            // Original configuration
            (10, 30, 10),   // periods: 10, 20, 30
            // Very large periods
            (100, 300, 50), // periods: 100, 150, 200, 250, 300
        ];
        
        for (cfg_idx, &(p_start, p_end, p_step)) in test_configs.iter().enumerate() {
            let output = DemaBatchBuilder::new()
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
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test, cfg_idx, val, bits, row, col, idx,
                        combo.period.unwrap_or(30)
                    );
                }
                
                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test, cfg_idx, val, bits, row, col, idx,
                        combo.period.unwrap_or(30)
                    );
                }
                
                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}",
                        test, cfg_idx, val, bits, row, col, idx,
                        combo.period.unwrap_or(30)
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

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test]
                fn [<$fn_name _scalar>]() {
                    let _ = $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch);
                }
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() {
                    let _ = $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch);
                }
                #[test]
                fn [<$fn_name _auto_detect>]() {
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "dema")]
pub fn dema_py<'py>(
    py: Python<'py>,
    arr_in: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};
    
    let slice_in = arr_in.as_slice()?;
    
    let params = DemaParams {
        period: Some(period),
    };
    let dema_in = DemaInput::from_slice(slice_in, params);
    
    let out_arr = unsafe { PyArray1::<f64>::new(py, [slice_in.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };
    
    py.allow_threads(|| -> Result<(), DemaError> {
        let (data, period, first, warm, chosen) = dema_prepare(&dema_in, Kernel::Auto)?;
        slice_out[..warm].fill(f64::NAN);
        dema_compute_into(data, period, first, chosen, slice_out);
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyclass(name = "DemaStream")]
pub struct DemaStreamPy {
    stream: DemaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl DemaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = DemaParams {
            period: Some(period),
        };
        let stream = DemaStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(DemaStreamPy { stream })
    }
    
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "dema_batch")]
pub fn dema_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{PyArray1, PyArrayMethods, IntoPyArray};
    use pyo3::types::PyDict;
    
    let slice_in = data.as_slice()?;
    
    let sweep = DemaBatchRange {
        period: period_range,
    };
    
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();
    
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };
    
    let combos = py.allow_threads(|| {
        let kernel = match Kernel::Auto {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        dema_batch_inner_into(slice_in, &sweep, kernel, true, slice_out)
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

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dema_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = DemaParams {
        period: Some(period),
    };
    let input = DemaInput::from_slice(data, params);
    
    dema_with_kernel(&input, Kernel::Scalar)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dema_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = DemaBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    dema_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dema_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = DemaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let metadata: Vec<f64> = combos
        .iter()
        .map(|combo| combo.period.unwrap() as f64)
        .collect();
    
    Ok(metadata)
}
