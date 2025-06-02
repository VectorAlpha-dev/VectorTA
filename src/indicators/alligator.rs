//! # Bill Williams Alligator Indicator
//!
//! A trend-following indicator composed of three smoothed moving averages
//! ("jaw", "teeth", "lips") with configurable periods and forward shifts.
//! API/feature/test parity with alma.rs, including AVX2/AVX512 batch stubs,
//! grid expansion, parameter structs/builders, and full kernel selection.
//!
//! ## Parameters
//! - **jaw_period** (default = 13): period for the "jaw" SMMA
//! - **jaw_offset** (default = 8): forward shift for jaw
//! - **teeth_period** (default = 8): period for "teeth"
//! - **teeth_offset** (default = 5): forward shift for teeth
//! - **lips_period** (default = 5): period for "lips"
//! - **lips_offset** (default = 3): forward shift for lips
//!
//! ## Errors
//! - **AllValuesNaN**: all data is NaN
//! - **InvalidPeriod**: period is zero or too large
//! - **InvalidOffset**: offset is too large
//!
//! ## Returns
//! - **Ok(AlligatorOutput)** on success, with jaw/teeth/lips vectors (shifted)
//! - **Err(AlligatorError)** otherwise
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{detect_best_batch_kernel, detect_best_kernel};
use aligned_vec::{AVec, CACHELINE_ALIGN};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;
use paste::paste;

impl<'a> AsRef<[f64]> for AlligatorInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AlligatorData::Slice(slice) => slice,
            AlligatorData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AlligatorData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AlligatorOutput {
    pub jaw: Vec<f64>,
    pub teeth: Vec<f64>,
    pub lips: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct AlligatorParams {
    pub jaw_period: Option<usize>,
    pub jaw_offset: Option<usize>,
    pub teeth_period: Option<usize>,
    pub teeth_offset: Option<usize>,
    pub lips_period: Option<usize>,
    pub lips_offset: Option<usize>,
}
impl Default for AlligatorParams {
    fn default() -> Self {
        Self {
            jaw_period: Some(13),
            jaw_offset: Some(8),
            teeth_period: Some(8),
            teeth_offset: Some(5),
            lips_period: Some(5),
            lips_offset: Some(3),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlligatorInput<'a> {
    pub data: AlligatorData<'a>,
    pub params: AlligatorParams,
}
impl<'a> AlligatorInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: AlligatorParams) -> Self {
        Self {
            data: AlligatorData::Candles { candles: c, source: s },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: AlligatorParams) -> Self {
        Self {
            data: AlligatorData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "hl2", AlligatorParams::default())
    }
    #[inline]
    pub fn get_jaw_period(&self) -> usize { self.params.jaw_period.unwrap_or(13) }
    #[inline]
    pub fn get_jaw_offset(&self) -> usize { self.params.jaw_offset.unwrap_or(8) }
    #[inline]
    pub fn get_teeth_period(&self) -> usize { self.params.teeth_period.unwrap_or(8) }
    #[inline]
    pub fn get_teeth_offset(&self) -> usize { self.params.teeth_offset.unwrap_or(5) }
    #[inline]
    pub fn get_lips_period(&self) -> usize { self.params.lips_period.unwrap_or(5) }
    #[inline]
    pub fn get_lips_offset(&self) -> usize { self.params.lips_offset.unwrap_or(3) }
}

#[derive(Copy, Clone, Debug)]
pub struct AlligatorBuilder {
    jaw_period: Option<usize>,
    jaw_offset: Option<usize>,
    teeth_period: Option<usize>,
    teeth_offset: Option<usize>,
    lips_period: Option<usize>,
    lips_offset: Option<usize>,
    kernel: Kernel,
}
impl Default for AlligatorBuilder {
    fn default() -> Self {
        Self {
            jaw_period: None,
            jaw_offset: None,
            teeth_period: None,
            teeth_offset: None,
            lips_period: None,
            lips_offset: None,
            kernel: Kernel::Auto,
        }
    }
}
impl AlligatorBuilder {
    #[inline(always)]
    pub fn new() -> Self { Self::default() }
    #[inline(always)]
    pub fn jaw_period(mut self, n: usize) -> Self { self.jaw_period = Some(n); self }
    #[inline(always)]
    pub fn jaw_offset(mut self, n: usize) -> Self { self.jaw_offset = Some(n); self }
    #[inline(always)]
    pub fn teeth_period(mut self, n: usize) -> Self { self.teeth_period = Some(n); self }
    #[inline(always)]
    pub fn teeth_offset(mut self, n: usize) -> Self { self.teeth_offset = Some(n); self }
    #[inline(always)]
    pub fn lips_period(mut self, n: usize) -> Self { self.lips_period = Some(n); self }
    #[inline(always)]
    pub fn lips_offset(mut self, n: usize) -> Self { self.lips_offset = Some(n); self }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AlligatorOutput, AlligatorError> {
        let p = AlligatorParams {
            jaw_period: self.jaw_period,
            jaw_offset: self.jaw_offset,
            teeth_period: self.teeth_period,
            teeth_offset: self.teeth_offset,
            lips_period: self.lips_period,
            lips_offset: self.lips_offset,
        };
        let i = AlligatorInput::from_candles(c, "hl2", p);
        alligator_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<AlligatorOutput, AlligatorError> {
        let p = AlligatorParams {
            jaw_period: self.jaw_period,
            jaw_offset: self.jaw_offset,
            teeth_period: self.teeth_period,
            teeth_offset: self.teeth_offset,
            lips_period: self.lips_period,
            lips_offset: self.lips_offset,
        };
        let i = AlligatorInput::from_slice(d, p);
        alligator_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<AlligatorStream, AlligatorError> {
        let p = AlligatorParams {
            jaw_period: self.jaw_period,
            jaw_offset: self.jaw_offset,
            teeth_period: self.teeth_period,
            teeth_offset: self.teeth_offset,
            lips_period: self.lips_period,
            lips_offset: self.lips_offset,
        };
        AlligatorStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AlligatorError {
    #[error("alligator: All values are NaN.")]
    AllValuesNaN,
    #[error("alligator: Invalid jaw period: period = {period}, data length = {data_len}")]
    InvalidJawPeriod { period: usize, data_len: usize },
    #[error("alligator: Invalid jaw offset: offset = {offset}, data_len = {data_len}")]
    InvalidJawOffset { offset: usize, data_len: usize },
    #[error("alligator: Invalid teeth period: period = {period}, data length = {data_len}")]
    InvalidTeethPeriod { period: usize, data_len: usize },
    #[error("alligator: Invalid teeth offset: offset = {offset}, data_len = {data_len}")]
    InvalidTeethOffset { offset: usize, data_len: usize },
    #[error("alligator: Invalid lips period: period = {period}, data length = {data_len}")]
    InvalidLipsPeriod { period: usize, data_len: usize },
    #[error("alligator: Invalid lips offset: offset = {offset}, data_len = {data_len}")]
    InvalidLipsOffset { offset: usize, data_len: usize },
}

#[inline]
pub fn alligator(input: &AlligatorInput) -> Result<AlligatorOutput, AlligatorError> {
    alligator_with_kernel(input, Kernel::Auto)
}
pub fn alligator_with_kernel(input: &AlligatorInput, kernel: Kernel) -> Result<AlligatorOutput, AlligatorError> {
    let data: &[f64] = match &input.data {
        AlligatorData::Candles { candles, source } => source_type(candles, source),
        AlligatorData::Slice(sl) => sl,
    };
    let first = data.iter().position(|x| !x.is_nan()).ok_or(AlligatorError::AllValuesNaN)?;
    let len = data.len();
    let jaw_period = input.get_jaw_period();
    let jaw_offset = input.get_jaw_offset();
    let teeth_period = input.get_teeth_period();
    let teeth_offset = input.get_teeth_offset();
    let lips_period = input.get_lips_period();
    let lips_offset = input.get_lips_offset();
    if jaw_period == 0 || jaw_period > len {
        return Err(AlligatorError::InvalidJawPeriod { period: jaw_period, data_len: len });
    }
    if jaw_offset > len {
        return Err(AlligatorError::InvalidJawOffset { offset: jaw_offset, data_len: len });
    }
    if teeth_period == 0 || teeth_period > len {
        return Err(AlligatorError::InvalidTeethPeriod { period: teeth_period, data_len: len });
    }
    if teeth_offset > len {
        return Err(AlligatorError::InvalidTeethOffset { offset: teeth_offset, data_len: len });
    }
    if lips_period == 0 || lips_period > len {
        return Err(AlligatorError::InvalidLipsPeriod { period: lips_period, data_len: len });
    }
    if lips_offset > len {
        return Err(AlligatorError::InvalidLipsOffset { offset: lips_offset, data_len: len });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                alligator_scalar(data, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, first, len)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                alligator_avx2(data, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, first, len)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                alligator_avx512(data, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, first, len)
            }
            _ => unreachable!(),
        }
    }
}
#[inline]
pub unsafe fn alligator_scalar(
    data: &[f64], jaw_period: usize, jaw_offset: usize, teeth_period: usize, teeth_offset: usize, lips_period: usize, lips_offset: usize, first: usize, len: usize
) -> Result<AlligatorOutput, AlligatorError> {
    let mut jaw = vec![f64::NAN; len];
    let mut teeth = vec![f64::NAN; len];
    let mut lips = vec![f64::NAN; len];

    let (jaw_val, teeth_val, lips_val) =
        alligator_smma_scalar(data, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, first, len, &mut jaw, &mut teeth, &mut lips);
    Ok(AlligatorOutput { jaw, teeth, lips })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn alligator_avx2(
    data: &[f64], jaw_period: usize, jaw_offset: usize, teeth_period: usize, teeth_offset: usize, lips_period: usize, lips_offset: usize, first: usize, len: usize
) -> Result<AlligatorOutput, AlligatorError> {
    // API parity only; forward to scalar
    alligator_scalar(data, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, first, len)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn alligator_avx512(
    data: &[f64], jaw_period: usize, jaw_offset: usize, teeth_period: usize, teeth_offset: usize, lips_period: usize, lips_offset: usize, first: usize, len: usize
) -> Result<AlligatorOutput, AlligatorError> {
    if jaw_period <= 32 && teeth_period <= 32 && lips_period <= 32 {
        alligator_avx512_short(data, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, first, len)
    } else {
        alligator_avx512_long(data, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, first, len)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn alligator_avx512_short(
    data: &[f64], jaw_period: usize, jaw_offset: usize, teeth_period: usize, teeth_offset: usize, lips_period: usize, lips_offset: usize, first: usize, len: usize
) -> Result<AlligatorOutput, AlligatorError> {
    // API stub: forwards to scalar
    alligator_scalar(data, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, first, len)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn alligator_avx512_long(
    data: &[f64], jaw_period: usize, jaw_offset: usize, teeth_period: usize, teeth_offset: usize, lips_period: usize, lips_offset: usize, first: usize, len: usize
) -> Result<AlligatorOutput, AlligatorError> {
    alligator_scalar(data, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, first, len)
}

#[inline(always)]
pub unsafe fn alligator_smma_scalar(
    data: &[f64], jaw_period: usize, jaw_offset: usize, teeth_period: usize, teeth_offset: usize, lips_period: usize, lips_offset: usize, _first: usize, len: usize,
    jaw: &mut [f64], teeth: &mut [f64], lips: &mut [f64]
) -> (f64, f64, f64) {
    let mut jaw_sum = 0.0;
    let mut teeth_sum = 0.0;
    let mut lips_sum = 0.0;

    let mut jaw_smma_val = 0.0;
    let mut teeth_smma_val = 0.0;
    let mut lips_smma_val = 0.0;

    let mut jaw_ready = false;
    let mut teeth_ready = false;
    let mut lips_ready = false;

    let jaw_scale = (jaw_period - 1) as f64;
    let jaw_inv_period = 1.0 / jaw_period as f64;

    let teeth_scale = (teeth_period - 1) as f64;
    let teeth_inv_period = 1.0 / teeth_period as f64;

    let lips_scale = (lips_period - 1) as f64;
    let lips_inv_period = 1.0 / lips_period as f64;

    for i in 0..len {
        let data_point = data[i];
        if !jaw_ready {
            if i < jaw_period {
                jaw_sum += data_point;
                if i == jaw_period - 1 {
                    jaw_smma_val = jaw_sum / (jaw_period as f64);
                    jaw_ready = true;
                    let shifted_index = i + jaw_offset;
                    if shifted_index < len {
                        jaw[shifted_index] = jaw_smma_val;
                    }
                }
            }
        } else {
            jaw_smma_val = (jaw_smma_val * jaw_scale + data_point) * jaw_inv_period;
            let shifted_index = i + jaw_offset;
            if shifted_index < len {
                jaw[shifted_index] = jaw_smma_val;
            }
        }

        if !teeth_ready {
            if i < teeth_period {
                teeth_sum += data_point;
                if i == teeth_period - 1 {
                    teeth_smma_val = teeth_sum / (teeth_period as f64);
                    teeth_ready = true;
                    let shifted_index = i + teeth_offset;
                    if shifted_index < len {
                        teeth[shifted_index] = teeth_smma_val;
                    }
                }
            }
        } else {
            teeth_smma_val = (teeth_smma_val * teeth_scale + data_point) * teeth_inv_period;
            let shifted_index = i + teeth_offset;
            if shifted_index < len {
                teeth[shifted_index] = teeth_smma_val;
            }
        }

        if !lips_ready {
            if i < lips_period {
                lips_sum += data_point;
                if i == lips_period - 1 {
                    lips_smma_val = lips_sum / (lips_period as f64);
                    lips_ready = true;
                    let shifted_index = i + lips_offset;
                    if shifted_index < len {
                        lips[shifted_index] = lips_smma_val;
                    }
                }
            }
        } else {
            lips_smma_val = (lips_smma_val * lips_scale + data_point) * lips_inv_period;
            let shifted_index = i + lips_offset;
            if shifted_index < len {
                lips[shifted_index] = lips_smma_val;
            }
        }
    }
    (jaw_smma_val, teeth_smma_val, lips_smma_val)
}

// Streaming variant for tick-by-tick mode (parity with AlmaStream)
#[derive(Debug, Clone)]
pub struct AlligatorStream {
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    jaw_buf: Vec<f64>,
    teeth_buf: Vec<f64>,
    lips_buf: Vec<f64>,
    jaw_head: usize,
    teeth_head: usize,
    lips_head: usize,
    jaw_filled: bool,
    teeth_filled: bool,
    lips_filled: bool,
    jaw_val: f64,
    teeth_val: f64,
    lips_val: f64,
    idx: usize,
}
impl AlligatorStream {
    pub fn try_new(params: AlligatorParams) -> Result<Self, AlligatorError> {
        let jaw_period = params.jaw_period.unwrap_or(13);
        let jaw_offset = params.jaw_offset.unwrap_or(8);
        let teeth_period = params.teeth_period.unwrap_or(8);
        let teeth_offset = params.teeth_offset.unwrap_or(5);
        let lips_period = params.lips_period.unwrap_or(5);
        let lips_offset = params.lips_offset.unwrap_or(3);

        if jaw_period == 0 {
            return Err(AlligatorError::InvalidJawPeriod { period: jaw_period, data_len: 0 });
        }
        if teeth_period == 0 {
            return Err(AlligatorError::InvalidTeethPeriod { period: teeth_period, data_len: 0 });
        }
        if lips_period == 0 {
            return Err(AlligatorError::InvalidLipsPeriod { period: lips_period, data_len: 0 });
        }

        Ok(Self {
            jaw_period,
            jaw_offset,
            teeth_period,
            teeth_offset,
            lips_period,
            lips_offset,
            jaw_buf: vec![f64::NAN; jaw_period],
            teeth_buf: vec![f64::NAN; teeth_period],
            lips_buf: vec![f64::NAN; lips_period],
            jaw_head: 0,
            teeth_head: 0,
            lips_head: 0,
            jaw_filled: false,
            teeth_filled: false,
            lips_filled: false,
            jaw_val: f64::NAN,
            teeth_val: f64::NAN,
            lips_val: f64::NAN,
            idx: 0,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        self.idx += 1;

        // Jaw
        self.jaw_buf[self.jaw_head] = value;
        self.jaw_head = (self.jaw_head + 1) % self.jaw_period;
        if !self.jaw_filled && self.jaw_head == 0 {
            self.jaw_filled = true;
            self.jaw_val = self.jaw_buf.iter().copied().sum::<f64>() / self.jaw_period as f64;
        } else if self.jaw_filled {
            self.jaw_val = (self.jaw_val * (self.jaw_period as f64 - 1.0) + value) / self.jaw_period as f64;
        }

        // Teeth
        self.teeth_buf[self.teeth_head] = value;
        self.teeth_head = (self.teeth_head + 1) % self.teeth_period;
        if !self.teeth_filled && self.teeth_head == 0 {
            self.teeth_filled = true;
            self.teeth_val = self.teeth_buf.iter().copied().sum::<f64>() / self.teeth_period as f64;
        } else if self.teeth_filled {
            self.teeth_val = (self.teeth_val * (self.teeth_period as f64 - 1.0) + value) / self.teeth_period as f64;
        }

        // Lips
        self.lips_buf[self.lips_head] = value;
        self.lips_head = (self.lips_head + 1) % self.lips_period;
        if !self.lips_filled && self.lips_head == 0 {
            self.lips_filled = true;
            self.lips_val = self.lips_buf.iter().copied().sum::<f64>() / self.lips_period as f64;
        } else if self.lips_filled {
            self.lips_val = (self.lips_val * (self.lips_period as f64 - 1.0) + value) / self.lips_period as f64;
        }

        if self.idx < self.jaw_period.max(self.teeth_period).max(self.lips_period) {
            return None;
        }
        Some((self.jaw_val, self.teeth_val, self.lips_val))
    }
}

// Batch parameter grid
#[derive(Clone, Debug)]
pub struct AlligatorBatchRange {
    pub jaw_period: (usize, usize, usize),
    pub jaw_offset: (usize, usize, usize),
    pub teeth_period: (usize, usize, usize),
    pub teeth_offset: (usize, usize, usize),
    pub lips_period: (usize, usize, usize),
    pub lips_offset: (usize, usize, usize),
}
impl Default for AlligatorBatchRange {
    fn default() -> Self {
        Self {
            jaw_period: (13, 13, 0),
            jaw_offset: (8, 8, 0),
            teeth_period: (8, 8, 0),
            teeth_offset: (5, 5, 0),
            lips_period: (5, 5, 0),
            lips_offset: (3, 3, 0),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct AlligatorBatchBuilder {
    range: AlligatorBatchRange,
    kernel: Kernel,
}
impl AlligatorBatchBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn kernel(mut self, k: Kernel) -> Self { self.kernel = k; self }
    pub fn jaw_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.jaw_period = (start, end, step); self
    }
    pub fn jaw_offset_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.jaw_offset = (start, end, step); self
    }
    pub fn teeth_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.teeth_period = (start, end, step); self
    }
    pub fn teeth_offset_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.teeth_offset = (start, end, step); self
    }
    pub fn lips_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lips_period = (start, end, step); self
    }
    pub fn lips_offset_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lips_offset = (start, end, step); self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<AlligatorBatchOutput, AlligatorError> {
        alligator_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<AlligatorBatchOutput, AlligatorError> {
        AlligatorBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<AlligatorBatchOutput, AlligatorError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<AlligatorBatchOutput, AlligatorError> {
        AlligatorBatchBuilder::new().kernel(Kernel::Auto).apply_candles(c, "hl2")
    }
}

pub fn alligator_batch_with_kernel(
    data: &[f64], sweep: &AlligatorBatchRange, k: Kernel,
) -> Result<AlligatorBatchOutput, AlligatorError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(AlligatorError::InvalidJawPeriod { period: 0, data_len: 0 }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    alligator_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct AlligatorBatchOutput {
    pub jaw: Vec<f64>,
    pub teeth: Vec<f64>,
    pub lips: Vec<f64>,
    pub combos: Vec<AlligatorParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AlligatorBatchOutput {
    pub fn row_for_params(&self, p: &AlligatorParams) -> Option<usize> {
        self.combos.iter().position(|c|
            c.jaw_period.unwrap_or(13) == p.jaw_period.unwrap_or(13)
            && c.jaw_offset.unwrap_or(8) == p.jaw_offset.unwrap_or(8)
            && c.teeth_period.unwrap_or(8) == p.teeth_period.unwrap_or(8)
            && c.teeth_offset.unwrap_or(5) == p.teeth_offset.unwrap_or(5)
            && c.lips_period.unwrap_or(5) == p.lips_period.unwrap_or(5)
            && c.lips_offset.unwrap_or(3) == p.lips_offset.unwrap_or(3)
        )
    }
    pub fn values_for(&self, p: &AlligatorParams) -> Option<(&[f64], &[f64], &[f64])> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            (
                &self.jaw[start..start + self.cols],
                &self.teeth[start..start + self.cols],
                &self.lips[start..start + self.cols],
            )
        })
    }
}

#[inline(always)]
fn expand_grid(r: &AlligatorBatchRange) -> Vec<AlligatorParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end { return vec![start]; }
        (start..=end).step_by(step).collect()
    }
    let jaw_periods = axis(r.jaw_period);
    let jaw_offsets = axis(r.jaw_offset);
    let teeth_periods = axis(r.teeth_period);
    let teeth_offsets = axis(r.teeth_offset);
    let lips_periods = axis(r.lips_period);
    let lips_offsets = axis(r.lips_offset);

    let mut out = Vec::with_capacity(jaw_periods.len() * jaw_offsets.len() * teeth_periods.len() * teeth_offsets.len() * lips_periods.len() * lips_offsets.len());
    for &jp in &jaw_periods {
        for &jo in &jaw_offsets {
            for &tp in &teeth_periods {
                for &to in &teeth_offsets {
                    for &lp in &lips_periods {
                        for &lo in &lips_offsets {
                            out.push(AlligatorParams {
                                jaw_period: Some(jp),
                                jaw_offset: Some(jo),
                                teeth_period: Some(tp),
                                teeth_offset: Some(to),
                                lips_period: Some(lp),
                                lips_offset: Some(lo),
                            });
                        }
                    }
                }
            }
        }
    }
    out
}

#[inline(always)]
pub fn alligator_batch_slice(
    data: &[f64], sweep: &AlligatorBatchRange, kern: Kernel,
) -> Result<AlligatorBatchOutput, AlligatorError> {
    alligator_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn alligator_batch_par_slice(
    data: &[f64], sweep: &AlligatorBatchRange, kern: Kernel,
) -> Result<AlligatorBatchOutput, AlligatorError> {
    alligator_batch_inner(data, sweep, kern, true)
}
#[inline(always)]
fn alligator_batch_inner(
    data: &[f64], sweep: &AlligatorBatchRange, kern: Kernel, parallel: bool
) -> Result<AlligatorBatchOutput, AlligatorError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(AlligatorError::InvalidJawPeriod { period: 0, data_len: 0 });
    }
    let first = data.iter().position(|x| !x.is_nan()).ok_or(AlligatorError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.jaw_period.unwrap().max(c.teeth_period.unwrap()).max(c.lips_period.unwrap())).max().unwrap();
    if data.len() - first < max_p {
        return Err(AlligatorError::InvalidJawPeriod { period: max_p, data_len: data.len() });
    }
    let rows = combos.len();
    let cols = data.len();
    let mut jaw = vec![f64::NAN; rows * cols];
    let mut teeth = vec![f64::NAN; rows * cols];
    let mut lips = vec![f64::NAN; rows * cols];

    let do_row = |row: usize, jaw_out: &mut [f64], teeth_out: &mut [f64], lips_out: &mut [f64]| unsafe {
        let prm = &combos[row];
        let (jaw_val, teeth_val, lips_val) = match kern {
            Kernel::Scalar => alligator_row_scalar(data, first,
                prm.jaw_period.unwrap(), prm.jaw_offset.unwrap(),
                prm.teeth_period.unwrap(), prm.teeth_offset.unwrap(),
                prm.lips_period.unwrap(), prm.lips_offset.unwrap(),
                cols, jaw_out, teeth_out, lips_out
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => alligator_row_avx2(data, first,
                prm.jaw_period.unwrap(), prm.jaw_offset.unwrap(),
                prm.teeth_period.unwrap(), prm.teeth_offset.unwrap(),
                prm.lips_period.unwrap(), prm.lips_offset.unwrap(),
                cols, jaw_out, teeth_out, lips_out
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => alligator_row_avx512(data, first,
                prm.jaw_period.unwrap(), prm.jaw_offset.unwrap(),
                prm.teeth_period.unwrap(), prm.teeth_offset.unwrap(),
                prm.lips_period.unwrap(), prm.lips_offset.unwrap(),
                cols, jaw_out, teeth_out, lips_out
            ),
            _ => unreachable!(),
        };
    };
    if parallel {
        jaw.par_chunks_mut(cols).zip(teeth.par_chunks_mut(cols)).zip(lips.par_chunks_mut(cols)).enumerate()
            .for_each(|(row, ((j, t), l))| do_row(row, j, t, l));
    } else {
        for (row, ((j, t), l)) in jaw.chunks_mut(cols).zip(teeth.chunks_mut(cols)).zip(lips.chunks_mut(cols)).enumerate() {
            do_row(row, j, t, l);
        }
    }
    Ok(AlligatorBatchOutput { jaw, teeth, lips, combos, rows, cols })
}

#[inline(always)]
pub unsafe fn alligator_row_scalar(
    data: &[f64], first: usize,
    jaw_period: usize, jaw_offset: usize, teeth_period: usize, teeth_offset: usize, lips_period: usize, lips_offset: usize,
    cols: usize, jaw: &mut [f64], teeth: &mut [f64], lips: &mut [f64]
) -> (f64, f64, f64) {
    alligator_smma_scalar(data, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, first, cols, jaw, teeth, lips)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn alligator_row_avx2(
    data: &[f64], first: usize,
    jaw_period: usize, jaw_offset: usize, teeth_period: usize, teeth_offset: usize, lips_period: usize, lips_offset: usize,
    cols: usize, jaw: &mut [f64], teeth: &mut [f64], lips: &mut [f64]
) -> (f64, f64, f64) {
    alligator_row_scalar(data, first, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, cols, jaw, teeth, lips)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn alligator_row_avx512(
    data: &[f64], first: usize,
    jaw_period: usize, jaw_offset: usize, teeth_period: usize, teeth_offset: usize, lips_period: usize, lips_offset: usize,
    cols: usize, jaw: &mut [f64], teeth: &mut [f64], lips: &mut [f64]
) -> (f64, f64, f64) {
    alligator_row_scalar(data, first, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, cols, jaw, teeth, lips)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn alligator_row_avx512_short(
    data: &[f64], first: usize,
    jaw_period: usize, jaw_offset: usize, teeth_period: usize, teeth_offset: usize, lips_period: usize, lips_offset: usize,
    cols: usize, jaw: &mut [f64], teeth: &mut [f64], lips: &mut [f64]
) -> (f64, f64, f64) {
    alligator_row_scalar(data, first, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, cols, jaw, teeth, lips)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn alligator_row_avx512_long(
    data: &[f64], first: usize,
    jaw_period: usize, jaw_offset: usize, teeth_period: usize, teeth_offset: usize, lips_period: usize, lips_offset: usize,
    cols: usize, jaw: &mut [f64], teeth: &mut [f64], lips: &mut [f64]
) -> (f64, f64, f64) {
    alligator_row_scalar(data, first, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, cols, jaw, teeth, lips)
}

#[inline(always)]
fn expand_grid_len(r: &AlligatorBatchRange) -> usize {
    fn axis((start, end, step): (usize, usize, usize)) -> usize {
        if step == 0 || start == end { 1 } else { ((end - start) / step + 1) }
    }
    axis(r.jaw_period) * axis(r.jaw_offset) * axis(r.teeth_period) * axis(r.teeth_offset) * axis(r.lips_period) * axis(r.lips_offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;
    use crate::skip_if_unsupported;
    fn check_alligator_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = AlligatorParams {
            jaw_period: Some(14), jaw_offset: None,
            teeth_period: None, teeth_offset: None,
            lips_period: None, lips_offset: Some(2),
        };
        let input = AlligatorInput::from_candles(&candles, "hl2", partial_params);
        let result = alligator_with_kernel(&input, kernel)?;
        assert_eq!(result.jaw.len(), candles.close.len());
        assert_eq!(result.teeth.len(), candles.close.len());
        assert_eq!(result.lips.len(), candles.close.len());
        Ok(())
    }
    fn check_alligator_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let hl2_prices = candles.get_calculated_field("hl2").expect("hl2 fail");
        let input = AlligatorInput::with_default_candles(&candles);
        let result = alligator_with_kernel(&input, kernel)?;
        let expected_last_five_jaw_result = [60742.4, 60632.6, 60555.1, 60442.7, 60308.7];
        let expected_last_five_teeth_result = [59908.0, 59757.2, 59684.3, 59653.5, 59621.1];
        let expected_last_five_lips_result = [59355.2, 59371.7, 59376.2, 59334.1, 59316.2];
        let start_index: usize = result.jaw.len() - 5;
        let result_last_five_jaws = &result.jaw[start_index..];
        let result_last_five_teeth = &result.teeth[start_index..];
        let result_last_five_lips = &result.lips[start_index..];
        for (i, &value) in result_last_five_jaws.iter().enumerate() {
            let expected_value = expected_last_five_jaw_result[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "alligator jaw value mismatch at index {}: expected {}, got {}",
                i, expected_value, value
            );
        }
        for (i, &value) in result_last_five_teeth.iter().enumerate() {
            let expected_value = expected_last_five_teeth_result[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "alligator teeth value mismatch at index {}: expected {}, got {}",
                i, expected_value, value
            );
        }
        for (i, &value) in result_last_five_lips.iter().enumerate() {
            let expected_value = expected_last_five_lips_result[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "alligator lips value mismatch at index {}: expected {}, got {}",
                i, expected_value, value
            );
        }
        Ok(())
    }
    fn check_alligator_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AlligatorInput::with_default_candles(&candles);
        match input.data {
            AlligatorData::Candles { source, .. } => assert_eq!(source, "hl2"),
            _ => panic!("Expected AlligatorData::Candles"),
        }
        let output = alligator_with_kernel(&input, kernel)?;
        assert_eq!(output.jaw.len(), candles.close.len());
        Ok(())
    }
    fn check_alligator_with_slice_data_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_input = AlligatorInput::with_default_candles(&candles);
        let first_result = alligator_with_kernel(&first_input, kernel)?;
        let second_input = AlligatorInput::from_slice(&first_result.jaw, AlligatorParams::default());
        let second_result = alligator_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.jaw.len(), first_result.jaw.len());
        assert_eq!(second_result.teeth.len(), first_result.teeth.len());
        assert_eq!(second_result.lips.len(), first_result.lips.len());
        Ok(())
    }
    fn check_alligator_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = AlligatorInput::with_default_candles(&candles);
        let result = alligator_with_kernel(&input, kernel)?;
        if result.jaw.len() > 50 {
            for i in 50..result.jaw.len() {
                assert!(!result.jaw[i].is_nan());
                assert!(!result.teeth[i].is_nan());
                assert!(!result.lips[i].is_nan());
            }
        }
        Ok(())
    }
    fn check_alligator_zero_jaw_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
    skip_if_unsupported!(kernel, test_name);
    let data = vec![10.0, 20.0, 30.0];
    let params = AlligatorParams { jaw_period: Some(0), ..AlligatorParams::default() };
    let input = AlligatorInput::from_slice(&data, params);
    let res = alligator_with_kernel(&input, kernel);
    assert!(res.is_err(), "[{}] Alligator should fail with zero jaw period", test_name);
    Ok(())
    }
    macro_rules! generate_all_alligator_tests {
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
    generate_all_alligator_tests!(
        check_alligator_partial_params,
        check_alligator_accuracy,
        check_alligator_default_candles,
        check_alligator_with_slice_data_reinput,
        check_alligator_nan_handling,
        check_alligator_zero_jaw_period
    );
    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = AlligatorBatchBuilder::new().kernel(kernel).apply_candles(&c, "hl2")?;
        let def = AlligatorParams::default();
        let (row_jaw, row_teeth, row_lips) = output.values_for(&def).expect("default row missing");
        assert_eq!(row_jaw.len(), c.close.len());
        let expected = [
            60742.4, 60632.6, 60555.1, 60442.7, 60308.7
        ];
        let start = row_jaw.len() - 5;
        for (i, &v) in row_jaw[start..].iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}");
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
