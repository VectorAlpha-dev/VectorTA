//! # Awesome Oscillator (AO)
//!
//! A momentum indicator by Bill Williams, showing market momentum by comparing short and long period SMAs of median price (hl2).
//!
//! ## Parameters
//! - **short_period**: Window for short SMA (default: 5)
//! - **long_period**: Window for long SMA (default: 34)
//!
//! ## Errors
//! - **AllValuesNaN**: ao: All input values are NaN
//! - **InvalidPeriods**: ao: short=0 or long=0
//! - **ShortPeriodNotLess**: ao: short >= long
//! - **NoData**: ao: Input slice is empty
//! - **NotEnoughData**: ao: Not enough valid data for requested long_period
//!
//! ## Returns
//! - **`Ok(AoOutput)`** with Vec<f64> of same length as input, leading values are NaN
//! - **`Err(AoError)`** otherwise
//!
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use std::mem::{ManuallyDrop, MaybeUninit};
use thiserror::Error;

impl<'a> AsRef<[f64]> for AoInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AoData::Slice(slice) => slice,
            AoData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AoParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}

impl Default for AoParams {
    fn default() -> Self {
        Self {
            short_period: Some(5),
            long_period: Some(34),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AoInput<'a> {
    pub data: AoData<'a>,
    pub params: AoParams,
}

impl<'a> AoInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: AoParams) -> Self {
        Self {
            data: AoData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: AoParams) -> Self {
        Self {
            data: AoData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "hl2", AoParams::default())
    }
    #[inline]
    pub fn get_short(&self) -> usize {
        self.params.short_period.unwrap_or(5)
    }
    #[inline]
    pub fn get_long(&self) -> usize {
        self.params.long_period.unwrap_or(34)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AoBuilder {
    short_period: Option<usize>,
    long_period: Option<usize>,
    kernel: Kernel,
}

impl Default for AoBuilder {
    fn default() -> Self {
        Self {
            short_period: None,
            long_period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AoBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn short_period(mut self, n: usize) -> Self {
        self.short_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn long_period(mut self, n: usize) -> Self {
        self.long_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AoOutput, AoError> {
        let p = AoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = AoInput::from_candles(c, "hl2", p);
        ao_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<AoOutput, AoError> {
        let p = AoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        let i = AoInput::from_slice(d, p);
        ao_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<AoStream, AoError> {
        let p = AoParams {
            short_period: self.short_period,
            long_period: self.long_period,
        };
        AoStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AoError {
    #[error("ao: All values are NaN.")]
    AllValuesNaN,
    #[error("ao: Invalid periods: short={short}, long={long}")]
    InvalidPeriods { short: usize, long: usize },
    #[error("ao: Short period must be less than long period: short={short}, long={long}")]
    ShortPeriodNotLess { short: usize, long: usize },
    #[error("ao: No data provided.")]
    NoData,
    #[error("ao: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn ao(input: &AoInput) -> Result<AoOutput, AoError> {
    ao_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn ao_prepare<'a>(
    input: &'a AoInput,
) -> Result<(
    /*data*/ &'a [f64],
    /*short*/ usize,
    /*long*/ usize,
    /*first*/ usize,
    /*len*/ usize,
), AoError> {
    let data: &[f64] = match &input.data {
        AoData::Candles { candles, source } => source_type(candles, source),
        AoData::Slice(sl) => sl,
    };

    // Early empty slice detection
    if data.is_empty() {
        return Err(AoError::NoData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AoError::AllValuesNaN)?;

    let len = data.len();
    let short = input.get_short();
    let long = input.get_long();

    if short == 0 || long == 0 {
        return Err(AoError::InvalidPeriods { short, long });
    }
    if short >= long {
        return Err(AoError::ShortPeriodNotLess { short, long });
    }
    if len == 0 {
        return Err(AoError::NoData);
    }
    if (len - first) < long {
        return Err(AoError::NotEnoughValidData {
            needed: long,
            valid: len - first,
        });
    }

    Ok((data, short, long, first, len))
}

pub fn ao_with_kernel(input: &AoInput, kernel: Kernel) -> Result<AoOutput, AoError> {
    let (data, short, long, first, len) = ao_prepare(input)?;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    
    // Calculate warmup period
    let warmup_period = first + long - 1;
    
    // Use zero-copy allocation helper
    let mut out = alloc_with_nan_prefix(len, warmup_period);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => ao_scalar(data, short, long, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => ao_avx2(data, short, long, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => ao_avx512(data, short, long, first, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(AoOutput { values: out })
}

// Scalar implementation - ensures all elements are written
#[inline]
pub fn ao_scalar(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    let len = data.len();
    let mut short_sum = 0.0;
    let mut long_sum = 0.0;

    for i in 0..len {
        let val = data[i];
        short_sum += val;
        long_sum += val;

        if i >= short {
            short_sum -= data[i - short];
        }
        if i >= long {
            long_sum -= data[i - long];
        }
        
        if i >= first + long - 1 {
            // Only write values after warmup period
            // NaN values in warmup are already set by alloc_with_nan_prefix
            let short_sma = short_sum / (short as f64);
            let long_sma = long_sum / (long as f64);
            out[i] = short_sma - long_sma;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ao_avx512(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    if long <= 32 {
        unsafe { ao_avx512_short(data, short, long, first, out) }
    } else {
        unsafe { ao_avx512_long(data, short, long, first, out) }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn ao_avx2(data: &[f64], short: usize, long: usize, first: usize, out: &mut [f64]) {
    unsafe { ao_scalar(data, short, long, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ao_avx512_short(
    data: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) {
    ao_scalar(data, short, long, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ao_avx512_long(
    data: &[f64],
    short: usize,
    long: usize,
    first: usize,
    out: &mut [f64],
) {
    ao_scalar(data, short, long, first, out)
}

// Streaming AO
#[derive(Debug, Clone)]
pub struct AoStream {
    short: usize,
    long: usize,
    short_sum: f64,
    long_sum: f64,
    short_buf: Vec<f64>,
    long_buf: Vec<f64>,
    head: usize,
    filled: bool,
}

impl AoStream {
    pub fn try_new(params: AoParams) -> Result<Self, AoError> {
        let short = params.short_period.unwrap_or(5);
        let long = params.long_period.unwrap_or(34);
        if short == 0 || long == 0 {
            return Err(AoError::InvalidPeriods { short, long });
        }
        if short >= long {
            return Err(AoError::ShortPeriodNotLess { short, long });
        }
        Ok(Self {
            short,
            long,
            short_sum: 0.0,
            long_sum: 0.0,
            short_buf: vec![f64::NAN; short],
            long_buf: vec![f64::NAN; long],
            head: 0,
            filled: false,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let idx_short = self.head % self.short;
        let idx_long = self.head % self.long;

        let old_short = self.short_buf[idx_short];
        let old_long = self.long_buf[idx_long];

        self.short_buf[idx_short] = value;
        self.long_buf[idx_long] = value;

        if !old_short.is_nan() {
            self.short_sum -= old_short;
        }
        if !old_long.is_nan() {
            self.long_sum -= old_long;
        }
        self.short_sum += value;
        self.long_sum += value;

        self.head += 1;
        if self.head >= self.long {
            let short_sma = self.short_sum / (self.short as f64);
            let long_sma = self.long_sum / (self.long as f64);
            Some(short_sma - long_sma)
        } else {
            None
        }
    }
}

// Batch/grid support
#[derive(Clone, Debug)]
pub struct AoBatchRange {
    pub short_period: (usize, usize, usize),
    pub long_period: (usize, usize, usize),
}

impl Default for AoBatchRange {
    fn default() -> Self {
        Self {
            short_period: (5, 15, 1),
            long_period: (34, 60, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AoBatchBuilder {
    range: AoBatchRange,
    kernel: Kernel,
}

impl AoBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn short_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.short_period = (start, end, step);
        self
    }
    pub fn short_static(mut self, v: usize) -> Self {
        self.range.short_period = (v, v, 0);
        self
    }
    pub fn long_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.long_period = (start, end, step);
        self
    }
    pub fn long_static(mut self, v: usize) -> Self {
        self.range.long_period = (v, v, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<AoBatchOutput, AoError> {
        ao_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<AoBatchOutput, AoError> {
        AoBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<AoBatchOutput, AoError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<AoBatchOutput, AoError> {
        AoBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "hl2")
    }
}

pub fn ao_batch_with_kernel(
    data: &[f64],
    sweep: &AoBatchRange,
    k: Kernel,
) -> Result<AoBatchOutput, AoError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(AoError::InvalidPeriods { short: 0, long: 0 });
        }
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    ao_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct AoBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AoParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AoBatchOutput {
    pub fn row_for_params(&self, p: &AoParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.short_period.unwrap_or(5) == p.short_period.unwrap_or(5)
                && c.long_period.unwrap_or(34) == p.long_period.unwrap_or(34)
        })
    }
    pub fn values_for(&self, p: &AoParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &AoBatchRange) -> Vec<AoParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let shorts = axis(r.short_period);
    let longs = axis(r.long_period);

    let mut out = Vec::with_capacity(shorts.len() * longs.len());
    for &s in &shorts {
        for &l in &longs {
            if s < l && s > 0 && l > 0 {
                out.push(AoParams {
                    short_period: Some(s),
                    long_period: Some(l),
                });
            }
        }
    }
    out
}

#[inline(always)]
pub fn ao_batch_slice(
    data: &[f64],
    sweep: &AoBatchRange,
    kern: Kernel,
) -> Result<AoBatchOutput, AoError> {
    ao_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn ao_batch_par_slice(
    data: &[f64],
    sweep: &AoBatchRange,
    kern: Kernel,
) -> Result<AoBatchOutput, AoError> {
    ao_batch_inner(data, sweep, kern, true)
}
#[inline(always)]
fn ao_batch_inner_into(
    data: &[f64],
    sweep: &AoBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AoParams>, AoError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(AoError::InvalidPeriods { short: 0, long: 0 });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AoError::AllValuesNaN)?;
    let max_long = combos.iter().map(|c| c.long_period.unwrap()).max().unwrap();
    if data.len() - first < max_long {
        return Err(AoError::NotEnoughValidData {
            needed: max_long,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let do_row = |row: usize, out_row: &mut [f64]| unsafe {
        let short = combos[row].short_period.unwrap();
        let long = combos[row].long_period.unwrap();
        match kern {
            Kernel::Scalar => ao_row_scalar(data, first, short, long, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ao_row_avx2(data, first, short, long, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ao_row_avx512(data, first, short, long, out_row),
            _ => unreachable!(),
        }
    };
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in out.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in out.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }
    Ok(combos)
}

// Original batch function that allocates its own storage
#[inline(always)]
fn ao_batch_inner(
    data: &[f64],
    sweep: &AoBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AoBatchOutput, AoError> {
    let combos = expand_grid(sweep);
    let rows = combos.len();
    let cols = data.len();
    
    if cols == 0 {
        return Err(AoError::NoData);
    }
    
    // Use zero-copy matrix allocation
    let mut buf_mu = make_uninit_matrix(rows, cols);
    
    // Calculate warmup periods for each combination
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .unwrap_or(0);
    
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.long_period.unwrap() - 1)
        .collect();
    
    // Initialize NaN prefixes
    init_matrix_prefixes(&mut buf_mu, cols, &warm);
    
    // Convert to mutable slice for computation
    let mut buf_guard = std::mem::ManuallyDrop::new(buf_mu);
    let values_ptr = buf_guard.as_mut_ptr() as *mut f64;
    let values_len = buf_guard.len();
    let values_cap = buf_guard.capacity();
    
    let values = unsafe {
        // Create slice for ao_batch_inner_into
        let slice = std::slice::from_raw_parts_mut(values_ptr, values_len);
        
        // Do the computation
        ao_batch_inner_into(data, sweep, kern, parallel, slice)?;
        
        // Reclaim as Vec<f64>
        Vec::from_raw_parts(values_ptr, values_len, values_cap)
    };
    
    Ok(AoBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}
#[inline(always)]
unsafe fn ao_row_scalar(data: &[f64], first: usize, short: usize, long: usize, out: &mut [f64]) {
    ao_scalar(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ao_row_avx2(data: &[f64], first: usize, short: usize, long: usize, out: &mut [f64]) {
    ao_scalar(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ao_row_avx512(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    if long <= 32 {
        ao_row_avx512_short(data, first, short, long, out);
    } else {
        ao_row_avx512_long(data, first, short, long, out);
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ao_row_avx512_short(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    ao_scalar(data, short, long, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ao_row_avx512_long(
    data: &[f64],
    first: usize,
    short: usize,
    long: usize,
    out: &mut [f64],
) {
    ao_scalar(data, short, long, first, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use paste::paste;

    fn check_ao_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = AoParams {
            short_period: Some(3),
            long_period: None,
        };
        let input = AoInput::from_candles(&candles, "hl2", partial_params);
        let result = ao_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        Ok(())
    }
    fn check_ao_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AoInput::with_default_candles(&candles);
        let result = ao_with_kernel(&input, kernel)?;
        let expected_last_five = [-1671.3, -1401.6706, -1262.3559, -1178.4941, -1157.4118];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] AO {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_ao_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AoInput::with_default_candles(&candles);
        match input.data {
            AoData::Candles { source, .. } => assert_eq!(source, "hl2"),
            _ => panic!("Expected AoData::Candles"),
        }
        let output = ao_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_ao_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = AoParams {
            short_period: Some(0),
            long_period: Some(34),
        };
        let input = AoInput::from_slice(&input_data, params);
        let res = ao_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] AO should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_ao_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = AoParams {
            short_period: Some(5),
            long_period: Some(10),
        };
        let input = AoInput::from_slice(&data_small, params);
        let res = ao_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] AO should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_ao_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = AoParams {
            short_period: Some(5),
            long_period: Some(34),
        };
        let input = AoInput::from_slice(&single_point, params);
        let res = ao_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] AO should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_ao_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = AoParams {
            short_period: Some(5),
            long_period: Some(34),
        };
        let first_input = AoInput::from_candles(&candles, "hl2", first_params);
        let first_result = ao_with_kernel(&first_input, kernel)?;
        let second_params = AoParams {
            short_period: Some(3),
            long_period: Some(10),
        };
        let second_input = AoInput::from_slice(&first_result.values, second_params);
        let second_result = ao_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }
    fn check_ao_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AoInput::from_candles(
            &candles,
            "hl2",
            AoParams {
                short_period: Some(5),
                long_period: Some(34),
            },
        );
        let res = ao_with_kernel(&input, kernel)?;
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
    macro_rules! generate_all_ao_tests {
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
    generate_all_ao_tests!(
        check_ao_partial_params,
        check_ao_accuracy,
        check_ao_default_candles,
        check_ao_zero_period,
        check_ao_period_exceeds_length,
        check_ao_very_small_dataset,
        check_ao_reinput,
        check_ao_nan_handling
    );
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = AoBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "hl2")?;
        let def = AoParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [-1671.3, -1401.6706, -1262.3559, -1178.4941, -1157.4118];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto);
                }
            }
        };
    }
    gen_batch_tests!(check_batch_default_row);
}

#[cfg(feature = "python")]
#[pyfunction(name = "ao")]
#[pyo3(signature = (high, low, short_period, long_period, kernel=None))]
pub fn ao_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    short_period: usize,
    long_period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{PyArray1, PyArrayMethods};

    let high_slice = high.as_slice()?; // zero-copy, read-only view
    let low_slice = low.as_slice()?;   // zero-copy, read-only view

    // Verify slices have same length
    if high_slice.len() != low_slice.len() {
        return Err(PyValueError::new_err("High and low arrays must have same length"));
    }

    // Use kernel validation for safety
    let kern = validate_kernel(kernel, false)?;

    // Compute hl2 (median price)
    let hl2: Vec<f64> = high_slice.iter()
        .zip(low_slice.iter())
        .map(|(&h, &l)| (h + l) / 2.0)
        .collect();

    // Build input struct
    let params = AoParams {
        short_period: Some(short_period),
        long_period: Some(long_period),
    };
    let ao_in = AoInput::from_slice(&hl2, params);

    // Pre-allocate uninitialized NumPy output buffer
    // NOTE: PyArray1::new() creates uninitialized memory, not zero-initialized
    let out_arr = unsafe { PyArray1::<f64>::new(py, [hl2.len()], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Heavy lifting without the GIL
    py.allow_threads(|| -> Result<(), AoError> {
        let (data, short, long, first, _len) = ao_prepare(&ao_in)?;
        
        // Calculate warmup period and initialize NaN prefix
        let warmup_period = first + long - 1;
        if warmup_period > 0 {
            slice_out[..warmup_period].fill(f64::NAN);
        }
        
        // Resolve Kernel::Auto to a specific kernel
        let chosen = match kern {
            Kernel::Auto => detect_best_kernel(),
            other => other,
        };

        // SAFETY: We must write to ALL elements before returning to Python
        // ao_scalar now only writes values after warmup period
        unsafe {
            match chosen {
                Kernel::Scalar | Kernel::ScalarBatch => ao_scalar(data, short, long, first, slice_out),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx2 | Kernel::Avx2Batch => ao_avx2(data, short, long, first, slice_out),
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                Kernel::Avx512 | Kernel::Avx512Batch => ao_avx512(data, short, long, first, slice_out),
                _ => unreachable!(),
            }
        }
        
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(out_arr)
}

#[cfg(feature = "python")]
#[pyclass(name = "AoStream")]
pub struct AoStreamPy {
    stream: AoStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AoStreamPy {
    #[new]
    fn new(short_period: usize, long_period: usize) -> PyResult<Self> {
        let params = AoParams {
            short_period: Some(short_period),
            long_period: Some(long_period),
        };
        let stream =
            AoStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AoStreamPy { stream })
    }

    /// Updates the stream with a new high and low value and returns the calculated AO value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, high: f64, low: f64) -> Option<f64> {
        let hl2 = (high + low) / 2.0;
        self.stream.update(hl2)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ao_batch")]
#[pyo3(signature = (high, low, short_period_range, long_period_range, kernel=None))]
pub fn ao_batch_py<'py>(
    py: Python<'py>,
    high: numpy::PyReadonlyArray1<'py, f64>,
    low: numpy::PyReadonlyArray1<'py, f64>,
    short_period_range: (usize, usize, usize),
    long_period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;

    // Verify slices have same length
    if high_slice.len() != low_slice.len() {
        return Err(PyValueError::new_err("High and low arrays must have same length"));
    }

    // Compute hl2 (median price)
    let hl2: Vec<f64> = high_slice.iter()
        .zip(low_slice.iter())
        .map(|(&h, &l)| (h + l) / 2.0)
        .collect();

    let sweep = AoBatchRange {
        short_period: short_period_range,
        long_period: long_period_range,
    };

    // 1. Expand grid once to know rows*cols
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = hl2.len();

    // 2. Pre-allocate uninitialized NumPy array (1-D, will reshape later)
    // NOTE: PyArray1::new() creates uninitialized memory, not zero-initialized
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Use kernel validation for safety
    let kern = validate_kernel(kernel, true)?;

    // 3. Heavy work without the GIL
    let combos = py.allow_threads(|| -> Result<Vec<AoParams>, AoError> {
        // Initialize NaN prefixes for each row
        let first = hl2.iter().position(|x| !x.is_nan()).unwrap_or(0);
        let warm: Vec<usize> = combos
            .iter()
            .map(|c| first + c.long_period.unwrap() - 1)
            .collect();
        
        // Convert slice to MaybeUninit for init_matrix_prefixes
        let slice_mu = unsafe {
            std::slice::from_raw_parts_mut(
                slice_out.as_mut_ptr() as *mut MaybeUninit<f64>,
                slice_out.len()
            )
        };
        
        // Initialize NaN prefixes
        init_matrix_prefixes(slice_mu, cols, &warm);
        
        // Resolve Kernel::Auto to a specific kernel
        let kernel = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let simd = match kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => unreachable!(),
        };

        // Compute batch directly into pre-allocated buffer - no copy needed!
        ao_batch_inner_into(&hl2, &sweep, simd, true, slice_out)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // 4. Build dict with the GIL
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "short_periods",
        combos
            .iter()
            .map(|p| p.short_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "long_periods",
        combos
            .iter()
            .map(|p| p.long_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ao_js(high: &[f64], low: &[f64], short_period: usize, long_period: usize) -> Result<Vec<f64>, JsValue> {
    // Verify arrays have same length
    if high.len() != low.len() {
        return Err(JsValue::from_str("High and low arrays must have same length"));
    }

    // Compute hl2
    let hl2: Vec<f64> = high.iter()
        .zip(low.iter())
        .map(|(&h, &l)| (h + l) / 2.0)
        .collect();

    let params = AoParams {
        short_period: Some(short_period),
        long_period: Some(long_period),
    };
    let input = AoInput::from_slice(&hl2, params);

    ao_with_kernel(&input, Kernel::Scalar)
        .map(|o| o.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ao_batch_js(
    high: &[f64],
    low: &[f64],
    short_start: usize,
    short_end: usize,
    short_step: usize,
    long_start: usize,
    long_end: usize,
    long_step: usize,
) -> Result<Vec<f64>, JsValue> {
    // Verify arrays have same length
    if high.len() != low.len() {
        return Err(JsValue::from_str("High and low arrays must have same length"));
    }

    // Compute hl2
    let hl2: Vec<f64> = high.iter()
        .zip(low.iter())
        .map(|(&h, &l)| (h + l) / 2.0)
        .collect();

    let sweep = AoBatchRange {
        short_period: (short_start, short_end, short_step),
        long_period: (long_start, long_end, long_step),
    };

    // Use the existing batch function with parallel=false for WASM
    ao_batch_inner(&hl2, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ao_batch_metadata_js(
    short_start: usize,
    short_end: usize,
    short_step: usize,
    long_start: usize,
    long_end: usize,
    long_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AoBatchRange {
        short_period: (short_start, short_end, short_step),
        long_period: (long_start, long_end, long_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len() * 2);

    for combo in combos {
        metadata.push(combo.short_period.unwrap() as f64);
        metadata.push(combo.long_period.unwrap() as f64);
    }

    Ok(metadata)
}

// New ergonomic WASM API
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AoBatchConfig {
    pub short_period_range: (usize, usize, usize),
    pub long_period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AoBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AoParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ao_batch)]
pub fn ao_batch_unified_js(high: &[f64], low: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    // 1. Deserialize the configuration object from JavaScript
    let config: AoBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    // Verify arrays have same length
    if high.len() != low.len() {
        return Err(JsValue::from_str("High and low arrays must have same length"));
    }

    // Compute hl2
    let hl2: Vec<f64> = high.iter()
        .zip(low.iter())
        .map(|(&h, &l)| (h + l) / 2.0)
        .collect();

    let sweep = AoBatchRange {
        short_period: config.short_period_range,
        long_period: config.long_period_range,
    };

    // 2. Run the existing core logic
    let output = ao_batch_inner(&hl2, &sweep, Kernel::Scalar, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // 3. Create the structured output
    let js_output = AoBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    // 4. Serialize the output struct into a JavaScript object
    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
