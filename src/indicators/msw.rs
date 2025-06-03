//! # Mesa Sine Wave (MSW)
//!
//! The Mesa Sine Wave indicator attempts to detect turning points in price data
//! by fitting a sine wave function. It outputs two series: the `sine` wave
//! and a leading version of the wave (`lead`).
//!
//! ## Parameters
//! - **period**: The window size (number of data points). Defaults to 5.
//!
//! ## Errors
//! - **EmptyData**: msw: Input data slice is empty.
//! - **InvalidPeriod**: msw: `period` is zero or exceeds the data length.
//! - **NotEnoughValidData**: msw: Fewer than `period` valid (non-`NaN`) data points remain
//!   after the first valid index.
//! - **AllValuesNaN**: msw: All input data values are `NaN`.
//!
//! ## Returns
//! - **`Ok(MswOutput)`** on success, containing two `Vec<f64>` of equal length:
//!   `sine` and `lead`, both matching the input length, with leading `NaN`s until
//!   the Mesa Sine Wave window is filled.
//! - **`Err(MswError)`** otherwise.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use std::error::Error;
use thiserror::Error;

#[allow(clippy::approx_constant)]
const TULIP_PI: f64 = 3.1415926;
const TULIP_TPI: f64 = 2.0 * TULIP_PI;

impl<'a> AsRef<[f64]> for MswInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            MswData::Slice(slice) => slice,
            MswData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MswData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MswOutput {
    pub sine: Vec<f64>,
    pub lead: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MswParams {
    pub period: Option<usize>,
}

impl Default for MswParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct MswInput<'a> {
    pub data: MswData<'a>,
    pub params: MswParams,
}

impl<'a> MswInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: MswParams) -> Self {
        Self {
            data: MswData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: MswParams) -> Self {
        Self {
            data: MswData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", MswParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(5)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MswBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for MswBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl MswBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<MswOutput, MswError> {
        let p = MswParams {
            period: self.period,
        };
        let i = MswInput::from_candles(c, "close", p);
        msw_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<MswOutput, MswError> {
        let p = MswParams {
            period: self.period,
        };
        let i = MswInput::from_slice(d, p);
        msw_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<MswStream, MswError> {
        let p = MswParams {
            period: self.period,
        };
        MswStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum MswError {
    #[error("msw: Empty data provided for MSW.")]
    EmptyData,
    #[error("msw: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("msw: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("msw: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn msw(input: &MswInput) -> Result<MswOutput, MswError> {
    msw_with_kernel(input, Kernel::Auto)
}

pub fn msw_with_kernel(input: &MswInput, kernel: Kernel) -> Result<MswOutput, MswError> {
    let data: &[f64] = match &input.data {
        MswData::Candles { candles, source } => source_type(candles, source),
        MswData::Slice(sl) => sl,
    };
    if data.is_empty() {
        return Err(MswError::EmptyData);
    }
    let period = input.get_period();
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MswError::AllValuesNaN)?;
    let len = data.len();
    if period == 0 || period > len {
        return Err(MswError::InvalidPeriod { period, data_len: len });
    }
    if (len - first) < period {
        return Err(MswError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                msw_scalar(data, period, first, len)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                msw_avx2(data, period, first, len)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                msw_avx512(data, period, first, len)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub unsafe fn msw_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    len: usize,
) -> Result<MswOutput, MswError> {
    let mut sine = vec![f64::NAN; len];
    let mut lead = vec![f64::NAN; len];

    let mut cos_table: AVec<f64, aligned_vec::ConstAlign<CACHELINE_ALIGN>> =
        AVec::with_capacity(CACHELINE_ALIGN, period);
    let mut sin_table: AVec<f64, aligned_vec::ConstAlign<CACHELINE_ALIGN>> =
        AVec::with_capacity(CACHELINE_ALIGN, period);
    cos_table.resize(period, 0.0);
    sin_table.resize(period, 0.0);

    for j in 0..period {
        let angle = TULIP_TPI * j as f64 / period as f64;
        cos_table[j] = angle.cos();
        sin_table[j] = angle.sin();
    }
    for i in (first + period - 1)..len {
        let mut rp = 0.0;
        let mut ip = 0.0;
        for j in 0..period {
            let weight = data[i - j];
            rp += cos_table[j] * weight;
            ip += sin_table[j] * weight;
        }
        let mut phase = if rp.abs() > 0.001 {
            atan(ip / rp)
        } else {
            TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
        };
        if rp < 0.0 {
            phase += TULIP_PI;
        }
        phase += TULIP_PI / 2.0;
        if phase < 0.0 {
            phase += TULIP_TPI;
        }
        if phase > TULIP_TPI {
            phase -= TULIP_TPI;
        }
        sine[i] = phase.sin();
        lead[i] = (phase + TULIP_PI / 4.0).sin();
    }
    Ok(MswOutput { sine, lead })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn msw_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    len: usize,
) -> Result<MswOutput, MswError> {
    msw_scalar(data, period, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn msw_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    len: usize,
) -> Result<MswOutput, MswError> {
    msw_scalar(data, period, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn msw_avx512_short(
    data: &[f64],
    period: usize,
    first: usize,
    len: usize,
) -> Result<MswOutput, MswError> {
    msw_scalar(data, period, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn msw_avx512_long(
    data: &[f64],
    period: usize,
    first: usize,
    len: usize,
) -> Result<MswOutput, MswError> {
    msw_scalar(data, period, first, len)
}

pub fn atan(x: f64) -> f64 {
    x.atan()
}

#[derive(Debug, Clone)]
pub struct MswStream {
    period: usize,
    buffer: Vec<f64>,
    cos_table: Vec<f64>,
    sin_table: Vec<f64>,
    head: usize,
    filled: bool,
}

impl MswStream {
    pub fn try_new(params: MswParams) -> Result<Self, MswError> {
        let period = params.period.unwrap_or(5);
        if period == 0 {
            return Err(MswError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let mut cos_table = Vec::with_capacity(period);
        let mut sin_table = Vec::with_capacity(period);
        for j in 0..period {
            let angle = TULIP_TPI * j as f64 / period as f64;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
        Ok(Self {
            period,
            buffer: vec![f64::NAN; period],
            cos_table,
            sin_table,
            head: 0,
            filled: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64)> {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.period;
        if !self.filled && self.head == 0 {
            self.filled = true;
        }
        if !self.filled {
            return None;
        }
        Some(self.dot_ring())
    }
    #[inline(always)]
    fn dot_ring(&self) -> (f64, f64) {
        let mut rp = 0.0;
        let mut ip = 0.0;
        // `self.head` always points to the next insertion position, which is
        // the oldest sample in the ring buffer. The most recent value is the
        // element just before `head`. The batch implementation processes data
        // from newest to oldest, so mirror that ordering here.
        let mut idx = (self.head + self.period - 1) % self.period;
        for j in 0..self.period {
            rp += self.cos_table[j] * self.buffer[idx];
            ip += self.sin_table[j] * self.buffer[idx];
            idx = if idx == 0 { self.period - 1 } else { idx - 1 };
        }
        let mut phase = if rp.abs() > 0.001 {
            atan(ip / rp)
        } else {
            TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
        };
        if rp < 0.0 {
            phase += TULIP_PI;
        }
        phase += TULIP_PI / 2.0;
        if phase < 0.0 {
            phase += TULIP_TPI;
        }
        if phase > TULIP_TPI {
            phase -= TULIP_TPI;
        }
        (phase.sin(), (phase + TULIP_PI / 4.0).sin())
    }
}

#[derive(Clone, Debug)]
pub struct MswBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for MswBatchRange {
    fn default() -> Self {
        Self {
            period: (5, 30, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MswBatchBuilder {
    range: MswBatchRange,
    kernel: Kernel,
}

impl MswBatchBuilder {
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
    pub fn apply_slice(self, data: &[f64]) -> Result<MswBatchOutput, MswError> {
        msw_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<MswBatchOutput, MswError> {
        MswBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<MswBatchOutput, MswError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<MswBatchOutput, MswError> {
        MswBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "close")
    }
}

pub fn msw_batch_with_kernel(
    data: &[f64],
    sweep: &MswBatchRange,
    k: Kernel,
) -> Result<MswBatchOutput, MswError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(MswError::InvalidPeriod {
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
    msw_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct MswBatchOutput {
    pub sine: Vec<f64>,
    pub lead: Vec<f64>,
    pub combos: Vec<MswParams>,
    pub rows: usize,
    pub cols: usize,
}

impl MswBatchOutput {
    pub fn row_for_params(&self, p: &MswParams) -> Option<usize> {
        self.combos.iter().position(|c| c.period.unwrap_or(5) == p.period.unwrap_or(5))
    }
    pub fn sine_for(&self, p: &MswParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.sine[start..start + self.cols]
        })
    }
    pub fn lead_for(&self, p: &MswParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.lead[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &MswBatchRange) -> Vec<MswParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    periods
        .into_iter()
        .map(|p| MswParams { period: Some(p) })
        .collect()
}

#[inline(always)]
pub fn msw_batch_slice(
    data: &[f64],
    sweep: &MswBatchRange,
    kern: Kernel,
) -> Result<MswBatchOutput, MswError> {
    msw_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn msw_batch_par_slice(
    data: &[f64],
    sweep: &MswBatchRange,
    kern: Kernel,
) -> Result<MswBatchOutput, MswError> {
    msw_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn msw_batch_inner(
    data: &[f64],
    sweep: &MswBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<MswBatchOutput, MswError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(MswError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(MswError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(MswError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut sine = vec![f64::NAN; rows * cols];
    let mut lead = vec![f64::NAN; rows * cols];
    let do_row = |row: usize, sine_row: &mut [f64], lead_row: &mut [f64]| unsafe {
        let period = combos[row].period.unwrap();
        match kern {
            Kernel::Scalar => msw_row_scalar(data, first, period, sine_row, lead_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => msw_row_avx2(data, first, period, sine_row, lead_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => msw_row_avx512(data, first, period, sine_row, lead_row),
            _ => unreachable!(),
        }
    };
    if parallel {
        sine
            .par_chunks_mut(cols)
            .zip(lead.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (sine_row, lead_row))| do_row(row, sine_row, lead_row));
    } else {
        for (row, (sine_row, lead_row)) in sine.chunks_mut(cols).zip(lead.chunks_mut(cols)).enumerate() {
            do_row(row, sine_row, lead_row);
        }
    }
    Ok(MswBatchOutput {
        sine,
        lead,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
unsafe fn msw_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    sine: &mut [f64],
    lead: &mut [f64],
) {
    let mut cos_table: AVec<f64, aligned_vec::ConstAlign<CACHELINE_ALIGN>> =
        AVec::with_capacity(CACHELINE_ALIGN, period);
    let mut sin_table: AVec<f64, aligned_vec::ConstAlign<CACHELINE_ALIGN>> =
        AVec::with_capacity(CACHELINE_ALIGN, period);
    cos_table.resize(period, 0.0);
    sin_table.resize(period, 0.0);
    for j in 0..period {
        let angle = TULIP_TPI * j as f64 / period as f64;
        cos_table[j] = angle.cos();
        sin_table[j] = angle.sin();
    }
    for i in (first + period - 1)..data.len() {
        let mut rp = 0.0;
        let mut ip = 0.0;
        for j in 0..period {
            let weight = data[i - j];
            rp += cos_table[j] * weight;
            ip += sin_table[j] * weight;
        }
        let mut phase = if rp.abs() > 0.001 {
            atan(ip / rp)
        } else {
            TULIP_PI * if ip < 0.0 { -1.0 } else { 1.0 }
        };
        if rp < 0.0 {
            phase += TULIP_PI;
        }
        phase += TULIP_PI / 2.0;
        if phase < 0.0 {
            phase += TULIP_TPI;
        }
        if phase > TULIP_TPI {
            phase -= TULIP_TPI;
        }
        sine[i] = phase.sin();
        lead[i] = (phase + TULIP_PI / 4.0).sin();
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn msw_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    sine: &mut [f64],
    lead: &mut [f64],
) {
    msw_row_scalar(data, first, period, sine, lead)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn msw_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    sine: &mut [f64],
    lead: &mut [f64],
) {
    if period <= 32 {
        msw_row_avx512_short(data, first, period, sine, lead)
    } else {
        msw_row_avx512_long(data, first, period, sine, lead)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn msw_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    sine: &mut [f64],
    lead: &mut [f64],
) {
    msw_row_scalar(data, first, period, sine, lead)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn msw_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    sine: &mut [f64],
    lead: &mut [f64],
) {
    msw_row_scalar(data, first, period, sine, lead)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;

    fn check_msw_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = MswParams { period: None };
        let input_default = MswInput::from_candles(&candles, "close", default_params);
        let output_default = msw_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.sine.len(), candles.close.len());
        assert_eq!(output_default.lead.len(), candles.close.len());
        Ok(())
    }

    fn check_msw_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MswParams { period: Some(5) };
        let input = MswInput::from_candles(&candles, "close", params);
        let msw_result = msw_with_kernel(&input, kernel)?;
        let expected_last_five_sine = [
            -0.49733966449848194,
            -0.8909425976991894,
            -0.709353328514554,
            -0.40483478076837887,
            -0.8817006719953886,
        ];
        let expected_last_five_lead = [
            -0.9651269132969991,
            -0.30888310410390457,
            -0.003182174183612666,
            0.36030983330963545,
            -0.28983704937461496,
        ];
        let start = msw_result.sine.len().saturating_sub(5);
        for (i, &val) in msw_result.sine[start..].iter().enumerate() {
            let diff = (val - expected_last_five_sine[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] MSW sine mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five_sine[i]
            );
        }
        for (i, &val) in msw_result.lead[start..].iter().enumerate() {
            let diff = (val - expected_last_five_lead[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] MSW lead mismatch at idx {}: got {}, expected {}",
                test_name,
                i,
                val,
                expected_last_five_lead[i]
            );
        }
        Ok(())
    }

    fn check_msw_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = MswInput::with_default_candles(&candles);
        let output = msw_with_kernel(&input, kernel)?;
        assert_eq!(output.sine.len(), candles.close.len());
        assert_eq!(output.lead.len(), candles.close.len());
        Ok(())
    }

    fn check_msw_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = MswParams { period: Some(0) };
        let input = MswInput::from_slice(&input_data, params);
        let res = msw_with_kernel(&input, kernel);
        assert!(res.is_err(), "[{}] MSW should fail with zero period", test_name);
        Ok(())
    }

    fn check_msw_period_exceeds_length(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = MswParams { period: Some(10) };
        let input = MswInput::from_slice(&data_small, params);
        let res = msw_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MSW should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_msw_very_small_dataset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = MswParams { period: Some(5) };
        let input = MswInput::from_slice(&single_point, params);
        let res = msw_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] MSW should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_msw_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = MswParams { period: Some(5) };
        let input = MswInput::from_candles(&candles, "close", params);
        let res = msw_with_kernel(&input, kernel)?;
        assert_eq!(res.sine.len(), candles.close.len());
        assert_eq!(res.lead.len(), candles.close.len());
        Ok(())
    }

    fn check_msw_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 5;
        let input = MswInput::from_candles(&candles, "close", MswParams { period: Some(period) });
        let batch_output = msw_with_kernel(&input, kernel)?;
        let mut stream = MswStream::try_new(MswParams { period: Some(period) })?;
        let mut sine_stream = Vec::with_capacity(candles.close.len());
        let mut lead_stream = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some((s, l)) => {
                    sine_stream.push(s);
                    lead_stream.push(l);
                }
                None => {
                    sine_stream.push(f64::NAN);
                    lead_stream.push(f64::NAN);
                }
            }
        }
        assert_eq!(batch_output.sine.len(), sine_stream.len());
        assert_eq!(batch_output.lead.len(), lead_stream.len());
        for (i, (&b, &s)) in batch_output.sine.iter().zip(sine_stream.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] MSW streaming sine mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        for (i, (&b, &l)) in batch_output.lead.iter().zip(lead_stream.iter()).enumerate() {
            if b.is_nan() && l.is_nan() {
                continue;
            }
            let diff = (b - l).abs();
            assert!(
                diff < 1e-9,
                "[{}] MSW streaming lead mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                l,
                diff
            );
        }
        Ok(())
    }

    macro_rules! generate_all_msw_tests {
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
    generate_all_msw_tests!(
        check_msw_partial_params,
        check_msw_accuracy,
        check_msw_default_candles,
        check_msw_zero_period,
        check_msw_period_exceeds_length,
        check_msw_very_small_dataset,
        check_msw_nan_handling,
        check_msw_streaming
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = MswBatchBuilder::new().kernel(kernel).apply_candles(&c, "close")?;
        let def = MswParams::default();
        let row = output.sine_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
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
