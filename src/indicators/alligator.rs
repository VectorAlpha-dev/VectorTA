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
//!
//! ## Decision Log
//! - SIMD: AVX2/AVX512 present but currently stubbed to scalar due to limited gains; selection short-circuits to scalar where applicable.
//! - CUDA: Wrapper present; returns VRAM handles used by Python via CAI v3 + DLPack. Numerical outputs unchanged.
//! - Performance: Scalar path optimized (O(1) SMMA recurrences). Batch/streaming preserve warmup/NaN semantics.
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

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
use paste::paste;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use thiserror::Error;

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
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
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
            data: AlligatorData::Candles {
                candles: c,
                source: s,
            },
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
    pub fn get_jaw_period(&self) -> usize {
        self.params.jaw_period.unwrap_or(13)
    }
    #[inline]
    pub fn get_jaw_offset(&self) -> usize {
        self.params.jaw_offset.unwrap_or(8)
    }
    #[inline]
    pub fn get_teeth_period(&self) -> usize {
        self.params.teeth_period.unwrap_or(8)
    }
    #[inline]
    pub fn get_teeth_offset(&self) -> usize {
        self.params.teeth_offset.unwrap_or(5)
    }
    #[inline]
    pub fn get_lips_period(&self) -> usize {
        self.params.lips_period.unwrap_or(5)
    }
    #[inline]
    pub fn get_lips_offset(&self) -> usize {
        self.params.lips_offset.unwrap_or(3)
    }
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
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn jaw_period(mut self, n: usize) -> Self {
        self.jaw_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn jaw_offset(mut self, n: usize) -> Self {
        self.jaw_offset = Some(n);
        self
    }
    #[inline(always)]
    pub fn teeth_period(mut self, n: usize) -> Self {
        self.teeth_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn teeth_offset(mut self, n: usize) -> Self {
        self.teeth_offset = Some(n);
        self
    }
    #[inline(always)]
    pub fn lips_period(mut self, n: usize) -> Self {
        self.lips_period = Some(n);
        self
    }
    #[inline(always)]
    pub fn lips_offset(mut self, n: usize) -> Self {
        self.lips_offset = Some(n);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

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
    #[error("alligator: Input data slice is empty.")]
    EmptyInputData,
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
    #[error(
        "alligator: Invalid kernel for batch operation. Expected batch kernel, got: {kernel:?}"
    )]
    InvalidKernel { kernel: Kernel },
    #[error("alligator: invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("alligator: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("alligator: output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("alligator: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: i64, end: i64, step: i64 },
}

#[inline]
pub fn alligator(input: &AlligatorInput) -> Result<AlligatorOutput, AlligatorError> {
    alligator_with_kernel(input, Kernel::Auto)
}
pub fn alligator_with_kernel(
    input: &AlligatorInput,
    kernel: Kernel,
) -> Result<AlligatorOutput, AlligatorError> {
    let data: &[f64] = match &input.data {
        AlligatorData::Candles { candles, source } => source_type(candles, source),
        AlligatorData::Slice(sl) => sl,
    };
    if data.is_empty() {
        return Err(AlligatorError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlligatorError::AllValuesNaN)?;
    let len = data.len();
    let jaw_period = input.get_jaw_period();
    let jaw_offset = input.get_jaw_offset();
    let teeth_period = input.get_teeth_period();
    let teeth_offset = input.get_teeth_offset();
    let lips_period = input.get_lips_period();
    let lips_offset = input.get_lips_offset();
    if jaw_period == 0 || jaw_period > len {
        return Err(AlligatorError::InvalidJawPeriod {
            period: jaw_period,
            data_len: len,
        });
    }
    if jaw_offset > len {
        return Err(AlligatorError::InvalidJawOffset {
            offset: jaw_offset,
            data_len: len,
        });
    }
    if teeth_period == 0 || teeth_period > len {
        return Err(AlligatorError::InvalidTeethPeriod {
            period: teeth_period,
            data_len: len,
        });
    }
    if teeth_offset > len {
        return Err(AlligatorError::InvalidTeethOffset {
            offset: teeth_offset,
            data_len: len,
        });
    }
    if lips_period == 0 || lips_period > len {
        return Err(AlligatorError::InvalidLipsPeriod {
            period: lips_period,
            data_len: len,
        });
    }
    if lips_offset > len {
        return Err(AlligatorError::InvalidLipsOffset {
            offset: lips_offset,
            data_len: len,
        });
    }

    // Ensure enough valid data after first non-NaN to seed the largest period
    let needed = jaw_period.max(teeth_period).max(lips_period);
    let valid = len - first;
    if valid < needed {
        return Err(AlligatorError::NotEnoughValidData { needed, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => alligator_scalar(
                data,
                jaw_period,
                jaw_offset,
                teeth_period,
                teeth_offset,
                lips_period,
                lips_offset,
                first,
                len,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => alligator_avx2(
                data,
                jaw_period,
                jaw_offset,
                teeth_period,
                teeth_offset,
                lips_period,
                lips_offset,
                first,
                len,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => alligator_avx512(
                data,
                jaw_period,
                jaw_offset,
                teeth_period,
                teeth_offset,
                lips_period,
                lips_offset,
                first,
                len,
            ),
            _ => unreachable!(),
        }
    }
}
#[inline]
pub unsafe fn alligator_scalar(
    data: &[f64],
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    first: usize,
    len: usize,
) -> Result<AlligatorOutput, AlligatorError> {
    // Calculate warmup periods for each line (including offset)
    let jaw_warmup = first + jaw_period - 1 + jaw_offset;
    let teeth_warmup = first + teeth_period - 1 + teeth_offset;
    let lips_warmup = first + lips_period - 1 + lips_offset;

    // Use zero-copy memory allocation
    let mut jaw = alloc_with_nan_prefix(len, jaw_warmup);
    let mut teeth = alloc_with_nan_prefix(len, teeth_warmup);
    let mut lips = alloc_with_nan_prefix(len, lips_warmup);

    let _ = alligator_smma_scalar(
        data,
        jaw_period,
        jaw_offset,
        teeth_period,
        teeth_offset,
        lips_period,
        lips_offset,
        first,
        len,
        &mut jaw,
        &mut teeth,
        &mut lips,
    );
    Ok(AlligatorOutput { jaw, teeth, lips })
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn alligator_avx2(
    data: &[f64],
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    first: usize,
    len: usize,
) -> Result<AlligatorOutput, AlligatorError> {
    // API parity only; forward to scalar
    alligator_scalar(
        data,
        jaw_period,
        jaw_offset,
        teeth_period,
        teeth_offset,
        lips_period,
        lips_offset,
        first,
        len,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn alligator_avx512(
    data: &[f64],
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    first: usize,
    len: usize,
) -> Result<AlligatorOutput, AlligatorError> {
    if jaw_period <= 32 && teeth_period <= 32 && lips_period <= 32 {
        alligator_avx512_short(
            data,
            jaw_period,
            jaw_offset,
            teeth_period,
            teeth_offset,
            lips_period,
            lips_offset,
            first,
            len,
        )
    } else {
        alligator_avx512_long(
            data,
            jaw_period,
            jaw_offset,
            teeth_period,
            teeth_offset,
            lips_period,
            lips_offset,
            first,
            len,
        )
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn alligator_avx512_short(
    data: &[f64],
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    first: usize,
    len: usize,
) -> Result<AlligatorOutput, AlligatorError> {
    // API stub: forwards to scalar
    alligator_scalar(
        data,
        jaw_period,
        jaw_offset,
        teeth_period,
        teeth_offset,
        lips_period,
        lips_offset,
        first,
        len,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn alligator_avx512_long(
    data: &[f64],
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    first: usize,
    len: usize,
) -> Result<AlligatorOutput, AlligatorError> {
    alligator_scalar(
        data,
        jaw_period,
        jaw_offset,
        teeth_period,
        teeth_offset,
        lips_period,
        lips_offset,
        first,
        len,
    )
}

#[inline(always)]
pub unsafe fn alligator_smma_scalar(
    data: &[f64],
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    first: usize,
    len: usize,
    jaw: &mut [f64],
    teeth: &mut [f64],
    lips: &mut [f64],
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

    for i in first..len {
        let data_point = data[i];
        if !jaw_ready {
            if i < first + jaw_period {
                jaw_sum += data_point;
                if i == first + jaw_period - 1 {
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
            if i < first + teeth_period {
                teeth_sum += data_point;
                if i == first + teeth_period - 1 {
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
            if i < first + lips_period {
                lips_sum += data_point;
                if i == first + lips_period - 1 {
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

// Streaming variant for tick-by-tick mode (drop-in; preserves API)
// Internal helper for one SMMA line (period p, forward offset k).
#[derive(Debug, Clone)]
struct Smmaline {
    // Params
    period: usize,
    offset: usize,
    inv: f64, // 1.0 / period

    // State
    seeded: bool, // true once the first SMA seed is complete
    count: usize, // samples seen during seeding
    sum: f64,     // running sum used only until seeded
    value: f64,   // last SMMA value

    // Forward shift (k bars into the future): ring buffer of size = offset
    off_head: usize,
    off_filled: bool,
    off_buf: Vec<f64>,
}

impl Smmaline {
    #[inline(always)]
    fn new(period: usize, offset: usize) -> Self {
        debug_assert!(period > 0);
        // Allocate only what's needed for the forward shift.
        let off_buf = if offset > 0 {
            // The contents don't matter; we gate on off_filled.
            vec![0.0_f64; offset]
        } else {
            Vec::new()
        };
        Self {
            period,
            offset,
            inv: 1.0 / period as f64,
            seeded: false,
            count: 0,
            sum: 0.0,
            value: f64::NAN,
            off_head: 0,
            off_filled: false,
            off_buf,
        }
    }

    // O(1) update that returns the *unshifted* SMMA once the line is seeded.
    #[inline(always)]
    fn update_unshifted(&mut self, x: f64) -> Option<f64> {
        if !self.seeded {
            // Build the initial SMA seed in O(1) per tick with a running sum.
            self.sum += x;
            self.count += 1;
            if self.count == self.period {
                self.value = self.sum * self.inv; // SMA seed
                self.seeded = true;
                Some(self.value)
            } else {
                None
            }
        } else {
            // SMMA recurrence: v += (x - v) / period
            // Use FMA if available via mul_add.
            let delta = x - self.value;
            // self.value = self.value + delta * self.inv;
            self.value = delta.mul_add(self.inv, self.value);
            Some(self.value)
        }
    }

    // O(1) update that returns the *shifted* (forward) output.
    // First returns Some after both the seed is ready and the forward queue is "full".
    #[inline(always)]
    fn update_shifted(&mut self, x: f64) -> Option<f64> {
        let y = self.update_unshifted(x)?;
        if self.offset == 0 {
            return Some(y);
        }
        // Pop from ring only after it has wrapped once.
        let out = if self.off_filled {
            Some(self.off_buf[self.off_head])
        } else {
            None
        };
        self.off_buf[self.off_head] = y;
        self.off_head += 1;
        if self.off_head == self.offset {
            self.off_head = 0;
            self.off_filled = true;
        }
        out
    }

    #[inline(always)]
    fn is_seeded(&self) -> bool {
        self.seeded
    }
}

#[derive(Debug, Clone)]
pub struct AlligatorStream {
    jaw: Smmaline,
    teeth: Smmaline,
    lips: Smmaline,
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
            return Err(AlligatorError::InvalidJawPeriod {
                period: jaw_period,
                data_len: 0,
            });
        }
        if teeth_period == 0 {
            return Err(AlligatorError::InvalidTeethPeriod {
                period: teeth_period,
                data_len: 0,
            });
        }
        if lips_period == 0 {
            return Err(AlligatorError::InvalidLipsPeriod {
                period: lips_period,
                data_len: 0,
            });
        }

        Ok(Self {
            jaw: Smmaline::new(jaw_period, jaw_offset),
            teeth: Smmaline::new(teeth_period, teeth_offset),
            lips: Smmaline::new(lips_period, lips_offset),
        })
    }

    /// O(1) tick update.
    /// Returns unshifted (no forward shift applied) SMMA values once *all three* lines are seeded,
    /// preserving the original stream behavior/semantics.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        let j = self.jaw.update_unshifted(value);
        let t = self.teeth.update_unshifted(value);
        let l = self.lips.update_unshifted(value);
        match (j, t, l) {
            (Some(jv), Some(tv), Some(lv)) => Some((jv, tv, lv)),
            _ => None,
        }
    }

    /// Optional helper: returns *forward-shifted* outputs matching batch/array semantics.
    /// First returns Some after each line is seeded *and* its forward offset queue is filled.
    #[inline(always)]
    pub fn update_shifted(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        let j = self.jaw.update_shifted(value);
        let t = self.teeth.update_shifted(value);
        let l = self.lips.update_shifted(value);
        match (j, t, l) {
            (Some(jv), Some(tv), Some(lv)) => Some((jv, tv, lv)),
            _ => None,
        }
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
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn jaw_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.jaw_period = (start, end, step);
        self
    }
    pub fn jaw_offset_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.jaw_offset = (start, end, step);
        self
    }
    pub fn teeth_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.teeth_period = (start, end, step);
        self
    }
    pub fn teeth_offset_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.teeth_offset = (start, end, step);
        self
    }
    pub fn lips_period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lips_period = (start, end, step);
        self
    }
    pub fn lips_offset_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.lips_offset = (start, end, step);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<AlligatorBatchOutput, AlligatorError> {
        alligator_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<AlligatorBatchOutput, AlligatorError> {
        AlligatorBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<AlligatorBatchOutput, AlligatorError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<AlligatorBatchOutput, AlligatorError> {
        AlligatorBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "hl2")
    }
}

pub fn alligator_batch_with_kernel(
    data: &[f64],
    sweep: &AlligatorBatchRange,
    k: Kernel,
) -> Result<AlligatorBatchOutput, AlligatorError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        non_batch => return Err(AlligatorError::InvalidKernelForBatch(non_batch)),
    };
    // Pass the batch kernel through without conversion
    alligator_batch_par_slice(data, sweep, kernel)
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
        self.combos.iter().position(|c| {
            c.jaw_period.unwrap_or(13) == p.jaw_period.unwrap_or(13)
                && c.jaw_offset.unwrap_or(8) == p.jaw_offset.unwrap_or(8)
                && c.teeth_period.unwrap_or(8) == p.teeth_period.unwrap_or(8)
                && c.teeth_offset.unwrap_or(5) == p.teeth_offset.unwrap_or(5)
                && c.lips_period.unwrap_or(5) == p.lips_period.unwrap_or(5)
                && c.lips_offset.unwrap_or(3) == p.lips_offset.unwrap_or(3)
        })
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
fn expand_grid(r: &AlligatorBatchRange) -> Result<Vec<AlligatorParams>, AlligatorError> {
    fn axis((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, AlligatorError> {
        if step == 0 || start == end {
            return Ok(vec![start]);
        }
        if start < end {
            let v: Vec<usize> = (start..=end).step_by(step).collect();
            if v.is_empty() {
                return Err(AlligatorError::InvalidRange {
                    start: start as i64,
                    end: end as i64,
                    step: step as i64,
                });
            }
            Ok(v)
        } else {
            // reversed bounds
            let mut v = Vec::new();
            let mut cur = start;
            while cur >= end {
                v.push(cur);
                if cur - end < step {
                    break;
                }
                cur -= step;
            }
            if v.is_empty() {
                return Err(AlligatorError::InvalidRange {
                    start: start as i64,
                    end: end as i64,
                    step: step as i64,
                });
            }
            Ok(v)
        }
    }
    let jaw_periods = axis(r.jaw_period)?;
    let jaw_offsets = axis(r.jaw_offset)?;
    let teeth_periods = axis(r.teeth_period)?;
    let teeth_offsets = axis(r.teeth_offset)?;
    let lips_periods = axis(r.lips_period)?;
    let lips_offsets = axis(r.lips_offset)?;

    let cap = jaw_periods
        .len()
        .checked_mul(jaw_offsets.len())
        .and_then(|v| v.checked_mul(teeth_periods.len()))
        .and_then(|v| v.checked_mul(teeth_offsets.len()))
        .and_then(|v| v.checked_mul(lips_periods.len()))
        .and_then(|v| v.checked_mul(lips_offsets.len()))
        .unwrap_or(0);
    let mut out = Vec::with_capacity(cap);
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
    if out.is_empty() {
        return Err(AlligatorError::InvalidRange {
            start: 0,
            end: 0,
            step: 0,
        });
    }
    Ok(out)
}

#[inline(always)]
pub fn alligator_batch_slice(
    data: &[f64],
    sweep: &AlligatorBatchRange,
    kern: Kernel,
) -> Result<AlligatorBatchOutput, AlligatorError> {
    alligator_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn alligator_batch_par_slice(
    data: &[f64],
    sweep: &AlligatorBatchRange,
    kern: Kernel,
) -> Result<AlligatorBatchOutput, AlligatorError> {
    alligator_batch_inner(data, sweep, kern, true)
}
#[inline(always)]
fn alligator_batch_inner(
    data: &[f64],
    sweep: &AlligatorBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AlligatorBatchOutput, AlligatorError> {
    let combos = expand_grid(sweep)?;
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlligatorError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| {
            c.jaw_period
                .unwrap()
                .max(c.teeth_period.unwrap())
                .max(c.lips_period.unwrap())
        })
        .max()
        .unwrap();
    if data.len() - first < max_p {
        return Err(AlligatorError::InvalidJawPeriod {
            period: max_p,
            data_len: data.len(),
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // Allocate uninitialized matrices for zero-copy memory
    let _rc = rows.checked_mul(cols).ok_or(AlligatorError::InvalidRange {
        start: rows as i64,
        end: cols as i64,
        step: 1,
    })?;
    let mut jaw_mu = make_uninit_matrix(rows, cols);
    let mut teeth_mu = make_uninit_matrix(rows, cols);
    let mut lips_mu = make_uninit_matrix(rows, cols);

    // Calculate warmup periods for each combination (including offset)
    let jaw_warmups: Vec<usize> = combos
        .iter()
        .map(|c| first + c.jaw_period.unwrap() - 1 + c.jaw_offset.unwrap())
        .collect();
    let teeth_warmups: Vec<usize> = combos
        .iter()
        .map(|c| first + c.teeth_period.unwrap() - 1 + c.teeth_offset.unwrap())
        .collect();
    let lips_warmups: Vec<usize> = combos
        .iter()
        .map(|c| first + c.lips_period.unwrap() - 1 + c.lips_offset.unwrap())
        .collect();

    // Initialize NaN prefixes
    init_matrix_prefixes(&mut jaw_mu, cols, &jaw_warmups);
    init_matrix_prefixes(&mut teeth_mu, cols, &teeth_warmups);
    init_matrix_prefixes(&mut lips_mu, cols, &lips_warmups);

    // Convert to mutable slices
    let mut jaw_guard = std::mem::ManuallyDrop::new(jaw_mu);
    let mut teeth_guard = std::mem::ManuallyDrop::new(teeth_mu);
    let mut lips_guard = std::mem::ManuallyDrop::new(lips_mu);

    let jaw: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(jaw_guard.as_mut_ptr() as *mut f64, jaw_guard.len())
    };
    let teeth: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(teeth_guard.as_mut_ptr() as *mut f64, teeth_guard.len())
    };
    let lips: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(lips_guard.as_mut_ptr() as *mut f64, lips_guard.len())
    };

    // Use the new alligator_batch_inner_into function
    let combos = alligator_batch_inner_into(data, sweep, kern, parallel, jaw, teeth, lips)?;
    // Reclaim as Vec<f64> (takes ownership from ManuallyDrop)
    let jaw_vec = unsafe {
        Vec::from_raw_parts(
            jaw_guard.as_mut_ptr() as *mut f64,
            jaw_guard.len(),
            jaw_guard.capacity(),
        )
    };
    let teeth_vec = unsafe {
        Vec::from_raw_parts(
            teeth_guard.as_mut_ptr() as *mut f64,
            teeth_guard.len(),
            teeth_guard.capacity(),
        )
    };
    let lips_vec = unsafe {
        Vec::from_raw_parts(
            lips_guard.as_mut_ptr() as *mut f64,
            lips_guard.len(),
            lips_guard.capacity(),
        )
    };

    Ok(AlligatorBatchOutput {
        jaw: jaw_vec,
        teeth: teeth_vec,
        lips: lips_vec,
        combos,
        rows,
        cols,
    })
}

#[inline]
fn alligator_batch_inner_into(
    data: &[f64],
    sweep: &AlligatorBatchRange,
    kern: Kernel,
    parallel: bool,
    jaw_out: &mut [f64],
    teeth_out: &mut [f64],
    lips_out: &mut [f64],

) -> Result<Vec<AlligatorParams>, AlligatorError> {
    let combos = expand_grid(sweep)?;

    let cols = data.len();
    let rows = combos.len();
    let total = rows
        .checked_mul(cols)
        .ok_or(AlligatorError::InvalidRange { start: rows as i64, end: cols as i64, step: 1 })?;
    if jaw_out.len() != total {
        return Err(AlligatorError::OutputLengthMismatch { expected: total, got: jaw_out.len() });
    }
    if teeth_out.len() != total {
        return Err(AlligatorError::OutputLengthMismatch { expected: total, got: teeth_out.len() });
    }
    if lips_out.len() != total {
        return Err(AlligatorError::OutputLengthMismatch { expected: total, got: lips_out.len() });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlligatorError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| {
            c.jaw_period
                .unwrap()
                .max(c.teeth_period.unwrap())
                .max(c.lips_period.unwrap())
        })
        .max()
        .unwrap();

    if data.len() - first < max_p {
        return Err(AlligatorError::InvalidJawPeriod {
            period: max_p,
            data_len: data.len(),
        });
    }

    let actual = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };
    let simd = match actual {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    let do_row = |row: usize, jdst: &mut [f64], tdst: &mut [f64], ldst: &mut [f64]| unsafe {
        let p = &combos[row];
        match simd {
            Kernel::Scalar => {
                let _ = alligator_row_scalar(
                    data,
                    first,
                    p.jaw_period.unwrap(),
                    p.jaw_offset.unwrap(),
                    p.teeth_period.unwrap(),
                    p.teeth_offset.unwrap(),
                    p.lips_period.unwrap(),
                    p.lips_offset.unwrap(),
                    cols,
                    jdst,
                    tdst,
                    ldst,
                );
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => {
                let _ = alligator_row_avx2(
                    data,
                    first,
                    p.jaw_period.unwrap(),
                    p.jaw_offset.unwrap(),
                    p.teeth_period.unwrap(),
                    p.teeth_offset.unwrap(),
                    p.lips_period.unwrap(),
                    p.lips_offset.unwrap(),
                    cols,
                    jdst,
                    tdst,
                    ldst,
                );
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => {
                let _ = alligator_row_avx512(
                    data,
                    first,
                    p.jaw_period.unwrap(),
                    p.jaw_offset.unwrap(),
                    p.teeth_period.unwrap(),
                    p.teeth_offset.unwrap(),
                    p.lips_period.unwrap(),
                    p.lips_offset.unwrap(),
                    cols,
                    jdst,
                    tdst,
                    ldst,
                );
            }
            _ => unreachable!(),
        }
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            jaw_out
                .par_chunks_mut(cols)
                .zip(teeth_out.par_chunks_mut(cols))
                .zip(lips_out.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(r, ((j, t), l))| do_row(r, j, t, l));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (r, ((j, t), l)) in jaw_out
                .chunks_mut(cols)
                .zip(teeth_out.chunks_mut(cols))
                .zip(lips_out.chunks_mut(cols))
                .enumerate()
            {
                do_row(r, j, t, l);
            }
        }
    } else {
        for (r, ((j, t), l)) in jaw_out
            .chunks_mut(cols)
            .zip(teeth_out.chunks_mut(cols))
            .zip(lips_out.chunks_mut(cols))
            .enumerate()
        {
            do_row(r, j, t, l);
        }
    }

    Ok(combos)
}

#[inline(always)]
pub unsafe fn alligator_row_scalar(
    data: &[f64],
    first: usize,
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    cols: usize,
    jaw: &mut [f64],
    teeth: &mut [f64],
    lips: &mut [f64],
) -> (f64, f64, f64) {
    alligator_smma_scalar(
        data,
        jaw_period,
        jaw_offset,
        teeth_period,
        teeth_offset,
        lips_period,
        lips_offset,
        first,
        cols,
        jaw,
        teeth,
        lips,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn alligator_row_avx2(
    data: &[f64],
    first: usize,
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    cols: usize,
    jaw: &mut [f64],
    teeth: &mut [f64],
    lips: &mut [f64],
) -> (f64, f64, f64) {
    alligator_row_scalar(
        data,
        first,
        jaw_period,
        jaw_offset,
        teeth_period,
        teeth_offset,
        lips_period,
        lips_offset,
        cols,
        jaw,
        teeth,
        lips,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn alligator_row_avx512(
    data: &[f64],
    first: usize,
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    cols: usize,
    jaw: &mut [f64],
    teeth: &mut [f64],
    lips: &mut [f64],
) -> (f64, f64, f64) {
    alligator_row_scalar(
        data,
        first,
        jaw_period,
        jaw_offset,
        teeth_period,
        teeth_offset,
        lips_period,
        lips_offset,
        cols,
        jaw,
        teeth,
        lips,
    )
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn alligator_row_avx512_short(
    data: &[f64],
    first: usize,
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    cols: usize,
    jaw: &mut [f64],
    teeth: &mut [f64],
    lips: &mut [f64],
) -> (f64, f64, f64) {
    alligator_row_scalar(
        data,
        first,
        jaw_period,
        jaw_offset,
        teeth_period,
        teeth_offset,
        lips_period,
        lips_offset,
        cols,
        jaw,
        teeth,
        lips,
    )
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn alligator_row_avx512_long(
    data: &[f64],
    first: usize,
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    cols: usize,
    jaw: &mut [f64],
    teeth: &mut [f64],
    lips: &mut [f64],
) -> (f64, f64, f64) {
    alligator_row_scalar(
        data,
        first,
        jaw_period,
        jaw_offset,
        teeth_period,
        teeth_offset,
        lips_period,
        lips_offset,
        cols,
        jaw,
        teeth,
        lips,
    )
}

#[inline(always)]
fn expand_grid_len(r: &AlligatorBatchRange) -> usize {
    fn axis((start, end, step): (usize, usize, usize)) -> usize {
        if step == 0 || start == end {
            1
        } else {
            ((end - start) / step + 1)
        }
    }
    axis(r.jaw_period)
        * axis(r.jaw_offset)
        * axis(r.teeth_period)
        * axis(r.teeth_offset)
        * axis(r.lips_period)
        * axis(r.lips_offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    
    #[test]
    fn test_alligator_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Build a small but non-trivial input with a NaN prefix
        let mut data = Vec::with_capacity(256);
        for _ in 0..7 { data.push(f64::NAN); }
        for i in 0..249 {
            let x = i as f64;
            data.push((x * 0.01) + (x.sin() * 0.1));
        }

        let input = AlligatorInput::from_slice(&data, AlligatorParams::default());

        // Baseline via Vec-returning API
        let AlligatorOutput { jaw: bj, teeth: bt, lips: bl } = alligator(&input)?;

        // Preallocate outputs and compute via into API
        let mut oj = vec![0.0; data.len()];
        let mut ot = vec![0.0; data.len()];
        let mut ol = vec![0.0; data.len()];
        alligator_into(&input, &mut oj, &mut ot, &mut ol)?;

        assert_eq!(oj.len(), bj.len());
        assert_eq!(ot.len(), bt.len());
        assert_eq!(ol.len(), bl.len());

        // Helper: NaN equals NaN; finite within tight epsilon
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12
        }

        for i in 0..data.len() {
            assert!(eq_or_both_nan(oj[i], bj[i]), "jaw mismatch at {}: {} vs {}", i, oj[i], bj[i]);
            assert!(eq_or_both_nan(ot[i], bt[i]), "teeth mismatch at {}: {} vs {}", i, ot[i], bt[i]);
            assert!(eq_or_both_nan(ol[i], bl[i]), "lips mismatch at {}: {} vs {}", i, ol[i], bl[i]);
        }
        Ok(())
    }
    fn check_alligator_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let partial_params = AlligatorParams {
            jaw_period: Some(14),
            jaw_offset: None,
            teeth_period: None,
            teeth_offset: None,
            lips_period: None,
            lips_offset: Some(2),
        };
        let input = AlligatorInput::from_candles(&candles, "hl2", partial_params);
        let result = alligator_with_kernel(&input, kernel)?;
        assert_eq!(result.jaw.len(), candles.close.len());
        assert_eq!(result.teeth.len(), candles.close.len());
        assert_eq!(result.lips.len(), candles.close.len());
        Ok(())
    }
    fn check_alligator_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
                i,
                expected_value,
                value
            );
        }
        for (i, &value) in result_last_five_teeth.iter().enumerate() {
            let expected_value = expected_last_five_teeth_result[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "alligator teeth value mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        for (i, &value) in result_last_five_lips.iter().enumerate() {
            let expected_value = expected_last_five_lips_result[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "alligator lips value mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        Ok(())
    }
    fn check_alligator_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
    fn check_alligator_with_slice_data_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_input = AlligatorInput::with_default_candles(&candles);
        let first_result = alligator_with_kernel(&first_input, kernel)?;
        let second_input =
            AlligatorInput::from_slice(&first_result.jaw, AlligatorParams::default());
        let second_result = alligator_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.jaw.len(), first_result.jaw.len());
        assert_eq!(second_result.teeth.len(), first_result.teeth.len());
        assert_eq!(second_result.lips.len(), first_result.lips.len());
        Ok(())
    }
    fn check_alligator_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
    fn check_alligator_zero_jaw_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = vec![10.0, 20.0, 30.0];
        let params = AlligatorParams {
            jaw_period: Some(0),
            ..AlligatorParams::default()
        };
        let input = AlligatorInput::from_slice(&data, params);
        let res = alligator_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] Alligator should fail with zero jaw period",
            test_name
        );
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_alligator_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate realistic parameter ranges for the three SMMA lines
        let strat = (6usize..=50).prop_flat_map(|max_period| {
            (
                // Data vector with length sufficient for the largest period
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    max_period..400,
                ),
                // Jaw parameters (typically the slowest/longest)
                ((max_period / 2).max(2)..=max_period), // jaw_period
                (0usize..=10),                          // jaw_offset
                // Teeth parameters (typically medium)
                ((max_period / 3).max(2)..=(max_period * 2 / 3).max(2)), // teeth_period
                (0usize..=8),                                            // teeth_offset
                // Lips parameters (typically the fastest/shortest)
                (2usize..=(max_period / 3).max(2)), // lips_period
                (0usize..=5),                       // lips_offset
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(
                &strat,
                |(
                    data,
                    jaw_period,
                    jaw_offset,
                    teeth_period,
                    teeth_offset,
                    lips_period,
                    lips_offset,
                )| {
                    let params = AlligatorParams {
                        jaw_period: Some(jaw_period),
                        jaw_offset: Some(jaw_offset),
                        teeth_period: Some(teeth_period),
                        teeth_offset: Some(teeth_offset),
                        lips_period: Some(lips_period),
                        lips_offset: Some(lips_offset),
                    };
                    let input = AlligatorInput::from_slice(&data, params);

                    let AlligatorOutput {
                        jaw: out_jaw,
                        teeth: out_teeth,
                        lips: out_lips,
                    } = alligator_with_kernel(&input, kernel).unwrap();
                    let AlligatorOutput {
                        jaw: ref_jaw,
                        teeth: ref_teeth,
                        lips: ref_lips,
                    } = alligator_with_kernel(&input, Kernel::Scalar).unwrap();

                    // Find first valid index in data
                    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

                    // Calculate warmup periods for each line
                    let jaw_warmup = first + jaw_period - 1 + jaw_offset;
                    let teeth_warmup = first + teeth_period - 1 + teeth_offset;
                    let lips_warmup = first + lips_period - 1 + lips_offset;

                    // Verify warmup periods have NaN values
                    for i in 0..jaw_warmup.min(out_jaw.len()) {
                        prop_assert!(
                            out_jaw[i].is_nan(),
                            "Expected NaN in jaw warmup at index {}",
                            i
                        );
                    }
                    for i in 0..teeth_warmup.min(out_teeth.len()) {
                        prop_assert!(
                            out_teeth[i].is_nan(),
                            "Expected NaN in teeth warmup at index {}",
                            i
                        );
                    }
                    for i in 0..lips_warmup.min(out_lips.len()) {
                        prop_assert!(
                            out_lips[i].is_nan(),
                            "Expected NaN in lips warmup at index {}",
                            i
                        );
                    }

                    // Verify that offset shifts output correctly
                    // The SMMA value calculated at position i should appear at output position i + offset
                    if jaw_warmup > 0 && jaw_warmup < data.len() {
                        // Check the first non-NaN value appears at the correct position
                        prop_assert!(
                            out_jaw[jaw_warmup].is_finite(),
                            "Expected first jaw value at index {} after warmup",
                            jaw_warmup
                        );
                        if jaw_warmup > 0 {
                            prop_assert!(
                                out_jaw[jaw_warmup - 1].is_nan(),
                                "Expected NaN before jaw warmup at index {}",
                                jaw_warmup - 1
                            );
                        }
                    }
                    if teeth_warmup > 0 && teeth_warmup < data.len() {
                        prop_assert!(
                            out_teeth[teeth_warmup].is_finite(),
                            "Expected first teeth value at index {} after warmup",
                            teeth_warmup
                        );
                        if teeth_warmup > 0 {
                            prop_assert!(
                                out_teeth[teeth_warmup - 1].is_nan(),
                                "Expected NaN before teeth warmup at index {}",
                                teeth_warmup - 1
                            );
                        }
                    }
                    if lips_warmup > 0 && lips_warmup < data.len() {
                        prop_assert!(
                            out_lips[lips_warmup].is_finite(),
                            "Expected first lips value at index {} after warmup",
                            lips_warmup
                        );
                        if lips_warmup > 0 {
                            prop_assert!(
                                out_lips[lips_warmup - 1].is_nan(),
                                "Expected NaN before lips warmup at index {}",
                                lips_warmup - 1
                            );
                        }
                    }

                    // Check consistency between kernels for all values
                    for i in 0..data.len() {
                        // Check jaw kernel consistency
                        let y_jaw = out_jaw[i];
                        let r_jaw = ref_jaw[i];
                        if !y_jaw.is_finite() || !r_jaw.is_finite() {
                            prop_assert!(
                                y_jaw.to_bits() == r_jaw.to_bits(),
                                "jaw finite/NaN mismatch idx {}: {} vs {}",
                                i,
                                y_jaw,
                                r_jaw
                            );
                        } else {
                            let ulp_diff: u64 = y_jaw.to_bits().abs_diff(r_jaw.to_bits());
                            prop_assert!(
                                (y_jaw - r_jaw).abs() <= 1e-8 || ulp_diff <= 16,
                                "jaw mismatch idx {}: {} vs {} (ULP={})",
                                i,
                                y_jaw,
                                r_jaw,
                                ulp_diff
                            );
                        }

                        // Check teeth kernel consistency
                        let y_teeth = out_teeth[i];
                        let r_teeth = ref_teeth[i];
                        if !y_teeth.is_finite() || !r_teeth.is_finite() {
                            prop_assert!(
                                y_teeth.to_bits() == r_teeth.to_bits(),
                                "teeth finite/NaN mismatch idx {}: {} vs {}",
                                i,
                                y_teeth,
                                r_teeth
                            );
                        } else {
                            let ulp_diff: u64 = y_teeth.to_bits().abs_diff(r_teeth.to_bits());
                            prop_assert!(
                                (y_teeth - r_teeth).abs() <= 1e-8 || ulp_diff <= 16,
                                "teeth mismatch idx {}: {} vs {} (ULP={})",
                                i,
                                y_teeth,
                                r_teeth,
                                ulp_diff
                            );
                        }

                        // Check lips kernel consistency
                        let y_lips = out_lips[i];
                        let r_lips = ref_lips[i];
                        if !y_lips.is_finite() || !r_lips.is_finite() {
                            prop_assert!(
                                y_lips.to_bits() == r_lips.to_bits(),
                                "lips finite/NaN mismatch idx {}: {} vs {}",
                                i,
                                y_lips,
                                r_lips
                            );
                        } else {
                            let ulp_diff: u64 = y_lips.to_bits().abs_diff(r_lips.to_bits());
                            prop_assert!(
                                (y_lips - r_lips).abs() <= 1e-8 || ulp_diff <= 16,
                                "lips mismatch idx {}: {} vs {} (ULP={})",
                                i,
                                y_lips,
                                r_lips,
                                ulp_diff
                            );
                        }
                    }

                    // SMMA-specific property: smoothness check
                    // SMMA output should be smoother (less volatile) than input
                    if data.len() > jaw_warmup + 10 {
                        // Calculate variance of a segment of input data
                        let segment_start = jaw_warmup;
                        let segment_end = (jaw_warmup + 20).min(data.len());

                        let input_variance = if segment_end > segment_start + 1 {
                            let input_segment = &data[segment_start..segment_end];
                            let input_mean: f64 =
                                input_segment.iter().sum::<f64>() / input_segment.len() as f64;
                            let var: f64 = input_segment
                                .iter()
                                .map(|x| (x - input_mean).powi(2))
                                .sum::<f64>()
                                / input_segment.len() as f64;
                            var
                        } else {
                            0.0
                        };

                        let output_variance = if segment_end > segment_start + 1 {
                            let output_segment = &out_jaw[segment_start..segment_end];
                            let valid_outputs: Vec<f64> = output_segment
                                .iter()
                                .filter(|x| x.is_finite())
                                .cloned()
                                .collect();
                            if valid_outputs.len() > 1 {
                                let output_mean: f64 =
                                    valid_outputs.iter().sum::<f64>() / valid_outputs.len() as f64;
                                let var: f64 = valid_outputs
                                    .iter()
                                    .map(|x| (x - output_mean).powi(2))
                                    .sum::<f64>()
                                    / valid_outputs.len() as f64;
                                var
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };

                        // SMMA should reduce variance (be smoother) for non-constant data
                        if input_variance > 1e-10 && output_variance > 1e-10 {
                            prop_assert!(
							output_variance <= input_variance * 1.1, // Allow 10% tolerance for numerical errors
							"SMMA should smooth the data: output variance {} > input variance {}",
							output_variance, input_variance
						);
                        }
                    }

                    // Special case: when period is 1 and offset is 0, output should match input
                    if jaw_period == 1 && jaw_offset == 0 {
                        for i in first..data.len() {
                            prop_assert!(
                                (out_jaw[i] - data[i]).abs() <= f64::EPSILON,
                                "jaw with period=1, offset=0 should match input at idx {}",
                                i
                            );
                        }
                    }
                    if teeth_period == 1 && teeth_offset == 0 {
                        for i in first..data.len() {
                            prop_assert!(
                                (out_teeth[i] - data[i]).abs() <= f64::EPSILON,
                                "teeth with period=1, offset=0 should match input at idx {}",
                                i
                            );
                        }
                    }
                    if lips_period == 1 && lips_offset == 0 {
                        for i in first..data.len() {
                            prop_assert!(
                                (out_lips[i] - data[i]).abs() <= f64::EPSILON,
                                "lips with period=1, offset=0 should match input at idx {}",
                                i
                            );
                        }
                    }

                    // If all input values are the same constant, SMMA should converge to that constant
                    if data.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON)
                        && !data.is_empty()
                    {
                        let constant = data[first];
                        // For SMMA convergence, we need many iterations relative to the period
                        // Check convergence only if we have enough data (at least 5x the period)
                        if data.len() >= jaw_warmup + jaw_period * 5 {
                            // Check the last few values for convergence
                            let check_start = data.len().saturating_sub(5);
                            for i in check_start..data.len() {
                                if i >= jaw_warmup && i < out_jaw.len() {
                                    // SMMA converges asymptotically, so use a more relaxed tolerance
                                    prop_assert!(
                                        (out_jaw[i] - constant).abs() <= 1e-4,
                                        "jaw should converge to constant {} at idx {}, got {}",
                                        constant,
                                        i,
                                        out_jaw[i]
                                    );
                                }
                            }
                        }
                        if data.len() >= teeth_warmup + teeth_period * 5 {
                            let check_start = data.len().saturating_sub(5);
                            for i in check_start..data.len() {
                                if i >= teeth_warmup && i < out_teeth.len() {
                                    prop_assert!(
                                        (out_teeth[i] - constant).abs() <= 1e-4,
                                        "teeth should converge to constant {} at idx {}, got {}",
                                        constant,
                                        i,
                                        out_teeth[i]
                                    );
                                }
                            }
                        }
                        if data.len() >= lips_warmup + lips_period * 5 {
                            let check_start = data.len().saturating_sub(5);
                            for i in check_start..data.len() {
                                if i >= lips_warmup && i < out_lips.len() {
                                    prop_assert!(
                                        (out_lips[i] - constant).abs() <= 1e-4,
                                        "lips should converge to constant {} at idx {}, got {}",
                                        constant,
                                        i,
                                        out_lips[i]
                                    );
                                }
                            }
                        }
                    }

                    Ok(())
                },
            )
            .unwrap();

        Ok(())
    }

    #[cfg(not(feature = "proptest"))]
    fn check_alligator_property(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_alligator_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            AlligatorParams::default(),
            AlligatorParams {
                jaw_period: Some(5),
                jaw_offset: Some(3),
                teeth_period: Some(3),
                teeth_offset: Some(2),
                lips_period: Some(2),
                lips_offset: Some(1),
            },
            AlligatorParams {
                jaw_period: Some(21),
                jaw_offset: Some(13),
                teeth_period: Some(13),
                teeth_offset: Some(8),
                lips_period: Some(8),
                lips_offset: Some(5),
            },
            AlligatorParams {
                jaw_period: Some(30),
                jaw_offset: Some(15),
                teeth_period: Some(20),
                teeth_offset: Some(10),
                lips_period: Some(10),
                lips_offset: Some(5),
            },
            AlligatorParams {
                jaw_period: Some(50),
                jaw_offset: Some(25),
                teeth_period: Some(30),
                teeth_offset: Some(15),
                lips_period: Some(20),
                lips_offset: Some(10),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = AlligatorInput::from_candles(&candles, "hl2", params.clone());
            let output = alligator_with_kernel(&input, kernel)?;

            // Check jaw values
            for (i, &val) in output.jaw.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at jaw index {} \
						with params: jaw_period={}, jaw_offset={}, teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
						test_name,
						val,
						bits,
						i,
						params.jaw_period.unwrap_or(13),
						params.jaw_offset.unwrap_or(8),
						params.teeth_period.unwrap_or(8),
						params.teeth_offset.unwrap_or(5),
						params.lips_period.unwrap_or(5),
						params.lips_offset.unwrap_or(3),
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at jaw index {} \
						with params: jaw_period={}, jaw_offset={}, teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
						test_name,
						val,
						bits,
						i,
						params.jaw_period.unwrap_or(13),
						params.jaw_offset.unwrap_or(8),
						params.teeth_period.unwrap_or(8),
						params.teeth_offset.unwrap_or(5),
						params.lips_period.unwrap_or(5),
						params.lips_offset.unwrap_or(3),
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at jaw index {} \
						with params: jaw_period={}, jaw_offset={}, teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
						test_name,
						val,
						bits,
						i,
						params.jaw_period.unwrap_or(13),
						params.jaw_offset.unwrap_or(8),
						params.teeth_period.unwrap_or(8),
						params.teeth_offset.unwrap_or(5),
						params.lips_period.unwrap_or(5),
						params.lips_offset.unwrap_or(3),
					);
                }
            }

            // Check teeth values
            for (i, &val) in output.teeth.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at teeth index {} \
						with params: jaw_period={}, jaw_offset={}, teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
						test_name,
						val,
						bits,
						i,
						params.jaw_period.unwrap_or(13),
						params.jaw_offset.unwrap_or(8),
						params.teeth_period.unwrap_or(8),
						params.teeth_offset.unwrap_or(5),
						params.lips_period.unwrap_or(5),
						params.lips_offset.unwrap_or(3),
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at teeth index {} \
						with params: jaw_period={}, jaw_offset={}, teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
						test_name,
						val,
						bits,
						i,
						params.jaw_period.unwrap_or(13),
						params.jaw_offset.unwrap_or(8),
						params.teeth_period.unwrap_or(8),
						params.teeth_offset.unwrap_or(5),
						params.lips_period.unwrap_or(5),
						params.lips_offset.unwrap_or(3),
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at teeth index {} \
						with params: jaw_period={}, jaw_offset={}, teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
						test_name,
						val,
						bits,
						i,
						params.jaw_period.unwrap_or(13),
						params.jaw_offset.unwrap_or(8),
						params.teeth_period.unwrap_or(8),
						params.teeth_offset.unwrap_or(5),
						params.lips_period.unwrap_or(5),
						params.lips_offset.unwrap_or(3),
					);
                }
            }

            // Check lips values
            for (i, &val) in output.lips.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at lips index {} \
						with params: jaw_period={}, jaw_offset={}, teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
						test_name,
						val,
						bits,
						i,
						params.jaw_period.unwrap_or(13),
						params.jaw_offset.unwrap_or(8),
						params.teeth_period.unwrap_or(8),
						params.teeth_offset.unwrap_or(5),
						params.lips_period.unwrap_or(5),
						params.lips_offset.unwrap_or(3),
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at lips index {} \
						with params: jaw_period={}, jaw_offset={}, teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
						test_name,
						val,
						bits,
						i,
						params.jaw_period.unwrap_or(13),
						params.jaw_offset.unwrap_or(8),
						params.teeth_period.unwrap_or(8),
						params.teeth_offset.unwrap_or(5),
						params.lips_period.unwrap_or(5),
						params.lips_offset.unwrap_or(3),
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at lips index {} \
						with params: jaw_period={}, jaw_offset={}, teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
						test_name,
						val,
						bits,
						i,
						params.jaw_period.unwrap_or(13),
						params.jaw_offset.unwrap_or(8),
						params.teeth_period.unwrap_or(8),
						params.teeth_offset.unwrap_or(5),
						params.lips_period.unwrap_or(5),
						params.lips_offset.unwrap_or(3),
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_alligator_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
        check_alligator_zero_jaw_period,
        check_alligator_property,
        check_alligator_no_poison
    );
    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = AlligatorBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "hl2")?;
        let def = AlligatorParams::default();
        let (row_jaw, row_teeth, row_lips) = output.values_for(&def).expect("default row missing");
        assert_eq!(row_jaw.len(), c.close.len());
        let expected = [60742.4, 60632.6, 60555.1, 60442.7, 60308.7];
        let start = row_jaw.len() - 5;
        for (i, &v) in row_jaw[start..].iter().enumerate() {
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
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (5, 15, 2, 3, 10, 2, 3, 10, 2, 2, 8, 2, 2, 8, 2, 1, 5, 2),
            (10, 20, 5, 5, 10, 5, 8, 15, 5, 3, 8, 5, 3, 8, 3, 1, 5, 2),
            (13, 13, 0, 8, 8, 0, 8, 8, 0, 5, 5, 0, 5, 5, 0, 3, 3, 0),
            (
                20, 30, 10, 10, 15, 5, 15, 20, 5, 8, 10, 2, 10, 12, 2, 5, 6, 1,
            ),
        ];

        for (
            cfg_idx,
            &(
                jp_start,
                jp_end,
                jp_step,
                jo_start,
                jo_end,
                jo_step,
                tp_start,
                tp_end,
                tp_step,
                to_start,
                to_end,
                to_step,
                lp_start,
                lp_end,
                lp_step,
                lo_start,
                lo_end,
                lo_step,
            ),
        ) in test_configs.iter().enumerate()
        {
            let output = AlligatorBatchBuilder::new()
                .kernel(kernel)
                .jaw_period_range(jp_start, jp_end, jp_step)
                .jaw_offset_range(jo_start, jo_end, jo_step)
                .teeth_period_range(tp_start, tp_end, tp_step)
                .teeth_offset_range(to_start, to_end, to_step)
                .lips_period_range(lp_start, lp_end, lp_step)
                .lips_offset_range(lo_start, lo_end, lo_step)
                .apply_candles(&c, "hl2")?;

            // Check jaw values
            for (idx, &val) in output.jaw.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						at jaw row {} col {} (flat index {}) with params: jaw_period={}, jaw_offset={}, \
						teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.jaw_period.unwrap_or(13),
                        combo.jaw_offset.unwrap_or(8),
                        combo.teeth_period.unwrap_or(8),
                        combo.teeth_offset.unwrap_or(5),
                        combo.lips_period.unwrap_or(5),
                        combo.lips_offset.unwrap_or(3),
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at jaw row {} col {} (flat index {}) with params: jaw_period={}, jaw_offset={}, \
						teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.jaw_period.unwrap_or(13),
                        combo.jaw_offset.unwrap_or(8),
                        combo.teeth_period.unwrap_or(8),
                        combo.teeth_offset.unwrap_or(5),
                        combo.lips_period.unwrap_or(5),
                        combo.lips_offset.unwrap_or(3),
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at jaw row {} col {} (flat index {}) with params: jaw_period={}, jaw_offset={}, \
						teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.jaw_period.unwrap_or(13),
                        combo.jaw_offset.unwrap_or(8),
                        combo.teeth_period.unwrap_or(8),
                        combo.teeth_offset.unwrap_or(5),
                        combo.lips_period.unwrap_or(5),
                        combo.lips_offset.unwrap_or(3),
                    );
                }
            }

            // Check teeth values
            for (idx, &val) in output.teeth.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						at teeth row {} col {} (flat index {}) with params: jaw_period={}, jaw_offset={}, \
						teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.jaw_period.unwrap_or(13),
                        combo.jaw_offset.unwrap_or(8),
                        combo.teeth_period.unwrap_or(8),
                        combo.teeth_offset.unwrap_or(5),
                        combo.lips_period.unwrap_or(5),
                        combo.lips_offset.unwrap_or(3),
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at teeth row {} col {} (flat index {}) with params: jaw_period={}, jaw_offset={}, \
						teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.jaw_period.unwrap_or(13),
                        combo.jaw_offset.unwrap_or(8),
                        combo.teeth_period.unwrap_or(8),
                        combo.teeth_offset.unwrap_or(5),
                        combo.lips_period.unwrap_or(5),
                        combo.lips_offset.unwrap_or(3),
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at teeth row {} col {} (flat index {}) with params: jaw_period={}, jaw_offset={}, \
						teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.jaw_period.unwrap_or(13),
                        combo.jaw_offset.unwrap_or(8),
                        combo.teeth_period.unwrap_or(8),
                        combo.teeth_offset.unwrap_or(5),
                        combo.lips_period.unwrap_or(5),
                        combo.lips_offset.unwrap_or(3),
                    );
                }
            }

            // Check lips values
            for (idx, &val) in output.lips.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;
                let combo = &output.combos[row];

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Config {}: Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
						at lips row {} col {} (flat index {}) with params: jaw_period={}, jaw_offset={}, \
						teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.jaw_period.unwrap_or(13),
                        combo.jaw_offset.unwrap_or(8),
                        combo.teeth_period.unwrap_or(8),
                        combo.teeth_offset.unwrap_or(5),
                        combo.lips_period.unwrap_or(5),
                        combo.lips_offset.unwrap_or(3),
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
						at lips row {} col {} (flat index {}) with params: jaw_period={}, jaw_offset={}, \
						teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.jaw_period.unwrap_or(13),
                        combo.jaw_offset.unwrap_or(8),
                        combo.teeth_period.unwrap_or(8),
                        combo.teeth_offset.unwrap_or(5),
                        combo.lips_period.unwrap_or(5),
                        combo.lips_offset.unwrap_or(3),
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
						at lips row {} col {} (flat index {}) with params: jaw_period={}, jaw_offset={}, \
						teeth_period={}, teeth_offset={}, lips_period={}, lips_offset={}",
                        test,
                        cfg_idx,
                        val,
                        bits,
                        row,
                        col,
                        idx,
                        combo.jaw_period.unwrap_or(13),
                        combo.jaw_offset.unwrap_or(8),
                        combo.teeth_period.unwrap_or(8),
                        combo.teeth_offset.unwrap_or(5),
                        combo.lips_period.unwrap_or(5),
                        combo.lips_offset.unwrap_or(3),
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
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_no_poison);

    #[test]
    fn test_invalid_kernel_error() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sweep = AlligatorBatchRange {
            jaw_period: (5, 5, 0),
            jaw_offset: (1, 1, 0),
            teeth_period: (3, 3, 0),
            teeth_offset: (1, 1, 0),
            lips_period: (2, 2, 0),
            lips_offset: (1, 1, 0),
        };

        // Test that non-batch kernels are rejected with proper error
        let result = alligator_batch_with_kernel(&data, &sweep, Kernel::Scalar);
        assert!(matches!(result, Err(AlligatorError::InvalidKernelForBatch(Kernel::Scalar))));

        let result = alligator_batch_with_kernel(&data, &sweep, Kernel::Avx2);
        assert!(matches!(result, Err(AlligatorError::InvalidKernelForBatch(Kernel::Avx2))));

        // Test that batch kernels work
        let result = alligator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch);
        assert!(result.is_ok());

        // Test that Auto works
        let result = alligator_batch_with_kernel(&data, &sweep, Kernel::Auto);
        assert!(result.is_ok());
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "alligator")]
#[pyo3(signature = (data, jaw_period=13, jaw_offset=8, teeth_period=8, teeth_offset=5, lips_period=5, lips_offset=3, kernel=None))]
pub fn alligator_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let params = AlligatorParams {
        jaw_period: Some(jaw_period),
        jaw_offset: Some(jaw_offset),
        teeth_period: Some(teeth_period),
        teeth_offset: Some(teeth_offset),
        lips_period: Some(lips_period),
        lips_offset: Some(lips_offset),
    };
    let input = AlligatorInput::from_slice(slice_in, params);
    let kern = validate_kernel(kernel, false)?;

    let out = py
        .allow_threads(|| alligator_with_kernel(&input, kern))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("jaw", out.jaw.into_pyarray(py))?;
    dict.set_item("teeth", out.teeth.into_pyarray(py))?;
    dict.set_item("lips", out.lips.into_pyarray(py))?;
    Ok(dict)
}

#[cfg(feature = "python")]
#[pyclass(name = "AlligatorStream")]
pub struct AlligatorStreamPy {
    stream: AlligatorStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AlligatorStreamPy {
    #[new]
    #[pyo3(signature = (jaw_period=13, jaw_offset=8, teeth_period=8, teeth_offset=5, lips_period=5, lips_offset=3))]
    fn new(
        jaw_period: usize,
        jaw_offset: usize,
        teeth_period: usize,
        teeth_offset: usize,
        lips_period: usize,
        lips_offset: usize,
    ) -> PyResult<Self> {
        let params = AlligatorParams {
            jaw_period: Some(jaw_period),
            jaw_offset: Some(jaw_offset),
            teeth_period: Some(teeth_period),
            teeth_offset: Some(teeth_offset),
            lips_period: Some(lips_period),
            lips_offset: Some(lips_offset),
        };
        let stream =
            AlligatorStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AlligatorStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated Alligator values.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "alligator_batch")]
#[pyo3(signature = (data, jaw_period_range, jaw_offset_range, teeth_period_range, teeth_offset_range, lips_period_range, lips_offset_range, kernel=None))]
pub fn alligator_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    jaw_period_range: (usize, usize, usize),
    jaw_offset_range: (usize, usize, usize),
    teeth_period_range: (usize, usize, usize),
    teeth_offset_range: (usize, usize, usize),
    lips_period_range: (usize, usize, usize),
    lips_offset_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    let slice_in = data.as_slice()?;
    let sweep = AlligatorBatchRange {
        jaw_period: jaw_period_range,
        jaw_offset: jaw_offset_range,
        teeth_period: teeth_period_range,
        teeth_offset: teeth_offset_range,
        lips_period: lips_period_range,
        lips_offset: lips_offset_range,
    };

    // Map domain errors to Python exceptions to satisfy PyResult
    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    let jaw_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let teeth_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let lips_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let jaw_out = unsafe { jaw_arr.as_slice_mut()? };
    let teeth_out = unsafe { teeth_arr.as_slice_mut()? };
    let lips_out = unsafe { lips_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;
    let combos = py
        .allow_threads(|| {
            let batch_k = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            alligator_batch_inner_into(
                slice_in, &sweep, batch_k, true, jaw_out, teeth_out, lips_out,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("jaw", jaw_arr.reshape((rows, cols))?)?;
    dict.set_item("teeth", teeth_arr.reshape((rows, cols))?)?;
    dict.set_item("lips", lips_arr.reshape((rows, cols))?)?;

    dict.set_item(
        "jaw_periods",
        combos
            .iter()
            .map(|p| p.jaw_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "jaw_offsets",
        combos
            .iter()
            .map(|p| p.jaw_offset.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "teeth_periods",
        combos
            .iter()
            .map(|p| p.teeth_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "teeth_offsets",
        combos
            .iter()
            .map(|p| p.teeth_offset.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "lips_periods",
        combos
            .iter()
            .map(|p| p.lips_period.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "lips_offsets",
        combos
            .iter()
            .map(|p| p.lips_offset.unwrap() as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

/// Write directly to output slices - no allocations (WASM optimization helper)
pub fn alligator_into_slice(
    jaw_dst: &mut [f64],
    teeth_dst: &mut [f64],
    lips_dst: &mut [f64],
    input: &AlligatorInput,
    kern: Kernel,
) -> Result<(), AlligatorError> {
    // Extract data and validate parameters
    let data: &[f64] = match &input.data {
        AlligatorData::Candles { candles, source } => source_type(candles, source),
        AlligatorData::Slice(sl) => sl,
    };

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlligatorError::AllValuesNaN)?;

    let len = data.len();

    // Validate destination slice lengths
    if jaw_dst.len() != len || teeth_dst.len() != len || lips_dst.len() != len {
        return Err(AlligatorError::InvalidJawPeriod {
            period: jaw_dst.len(),
            data_len: len,
        });
    }

    // Get parameters
    let jaw_period = input.get_jaw_period();
    let jaw_offset = input.get_jaw_offset();
    let teeth_period = input.get_teeth_period();
    let teeth_offset = input.get_teeth_offset();
    let lips_period = input.get_lips_period();
    let lips_offset = input.get_lips_offset();

    // Validate parameters
    if jaw_period == 0 || jaw_period > len {
        return Err(AlligatorError::InvalidJawPeriod {
            period: jaw_period,
            data_len: len,
        });
    }
    if jaw_offset > len {
        return Err(AlligatorError::InvalidJawOffset {
            offset: jaw_offset,
            data_len: len,
        });
    }
    if teeth_period == 0 || teeth_period > len {
        return Err(AlligatorError::InvalidTeethPeriod {
            period: teeth_period,
            data_len: len,
        });
    }
    if teeth_offset > len {
        return Err(AlligatorError::InvalidTeethOffset {
            offset: teeth_offset,
            data_len: len,
        });
    }
    if lips_period == 0 || lips_period > len {
        return Err(AlligatorError::InvalidLipsPeriod {
            period: lips_period,
            data_len: len,
        });
    }
    if lips_offset > len {
        return Err(AlligatorError::InvalidLipsOffset {
            offset: lips_offset,
            data_len: len,
        });
    }

    // Calculate warmup periods for each line (including offset)
    let jaw_warmup = first + jaw_period - 1 + jaw_offset;
    let teeth_warmup = first + teeth_period - 1 + teeth_offset;
    let lips_warmup = first + lips_period - 1 + lips_offset;

    // Fill warmup periods with NaN
    for v in &mut jaw_dst[..jaw_warmup] {
        *v = f64::NAN;
    }
    for v in &mut teeth_dst[..teeth_warmup] {
        *v = f64::NAN;
    }
    for v in &mut lips_dst[..lips_warmup] {
        *v = f64::NAN;
    }

    // Compute SMMA values directly into the output slices
    unsafe {
        alligator_smma_scalar(
            data,
            jaw_period,
            jaw_offset,
            teeth_period,
            teeth_offset,
            lips_period,
            lips_offset,
            first,
            len,
            jaw_dst,
            teeth_dst,
            lips_dst,
        );
    }

    Ok(())
}

// ---------------- Python CUDA bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, CudaAlligator};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", name = "CudaContextGuard", unsendable)]
struct CudaContextGuardPy {
    #[pyo3(get)]
    device_id: u32,
    _ctx: Arc<Context>,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "alligator_cuda_batch_dev")]
#[pyo3(signature = (prices_f32, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, device_id=0))]
pub fn alligator_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    prices_f32: numpy::PyReadonlyArray1<'py, f32>,
    jaw_period: (usize, usize, usize),
    jaw_offset: (usize, usize, usize),
    teeth_period: (usize, usize, usize),
    teeth_offset: (usize, usize, usize),
    lips_period: (usize, usize, usize),
    lips_offset: (usize, usize, usize),
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice = prices_f32.as_slice()?;
    let sweep = AlligatorBatchRange {
        jaw_period,
        jaw_offset,
        teeth_period,
        teeth_offset,
        lips_period,
        lips_offset,
    };
    let (jaw, teeth, lips, rows, cols, jp, jo, tp, to, lp, lo, guard_dev, guard_ctx) =
        py.allow_threads(|| {
            let cuda =
                CudaAlligator::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let res = cuda
                .alligator_batch_dev(slice, &sweep)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let rows = res.outputs.rows();
            let cols = res.outputs.cols();
            let jp: Vec<usize> = res.combos.iter().map(|c| c.jaw_period.unwrap()).collect();
            let jo: Vec<usize> = res.combos.iter().map(|c| c.jaw_offset.unwrap()).collect();
            let tp: Vec<usize> = res.combos.iter().map(|c| c.teeth_period.unwrap()).collect();
            let to: Vec<usize> = res.combos.iter().map(|c| c.teeth_offset.unwrap()).collect();
            let lp: Vec<usize> = res.combos.iter().map(|c| c.lips_period.unwrap()).collect();
            let lo: Vec<usize> = res.combos.iter().map(|c| c.lips_offset.unwrap()).collect();
            Ok::<_, PyErr>(
                (
                    res.outputs.jaw,
                    res.outputs.teeth,
                    res.outputs.lips,
                    rows,
                    cols,
                    jp,
                    jo,
                    tp,
                    to,
                    lp,
                    lo,
                    res.outputs.device_id,
                    // retain context while Python holds buffers
                    res.outputs._ctx.clone(),
                ),
            )
        })?;
    use numpy::IntoPyArray;
    let d = PyDict::new(py);
    d.set_item("jaw", Py::new(py, DeviceArrayF32Py { inner: jaw })?)?;
    d.set_item("teeth", Py::new(py, DeviceArrayF32Py { inner: teeth })?)?;
    d.set_item("lips", Py::new(py, DeviceArrayF32Py { inner: lips })?)?;
    d.set_item("rows", rows)?;
    d.set_item("cols", cols)?;
    d.set_item("jaw_periods", jp.into_pyarray(py))?;
    d.set_item("jaw_offsets", jo.into_pyarray(py))?;
    d.set_item("teeth_periods", tp.into_pyarray(py))?;
    d.set_item("teeth_offsets", to.into_pyarray(py))?;
    d.set_item("lips_periods", lp.into_pyarray(py))?;
    d.set_item("lips_offsets", lo.into_pyarray(py))?;
    d.set_item(
        "context_guard",
        Py::new(
            py,
            CudaContextGuardPy {
                device_id: guard_dev,
                _ctx: guard_ctx,
            },
        )?,
    )?;
    Ok(d)
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "alligator_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset, device_id=0))]
pub fn alligator_cuda_many_series_one_param_dev_py<'py>(
    py: Python<'py>,
    data_tm_f32: numpy::PyReadonlyArray2<'py, f32>,
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
    device_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let shape = data_tm_f32.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("expected 2D array"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let flat = data_tm_f32.as_slice()?;
    let params = AlligatorParams {
        jaw_period: Some(jaw_period),
        jaw_offset: Some(jaw_offset),
        teeth_period: Some(teeth_period),
        teeth_offset: Some(teeth_offset),
        lips_period: Some(lips_period),
        lips_offset: Some(lips_offset),
    };
    let (jaw, teeth, lips, guard_dev, guard_ctx) = py.allow_threads(|| {
        let cuda =
            CudaAlligator::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let out = cuda
            .alligator_many_series_one_param_time_major_dev(flat, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok::<_, PyErr>((out.jaw, out.teeth, out.lips, cuda.device_id(), cuda.context_arc()))
    })?;
    let d = PyDict::new(py);
    d.set_item("jaw", Py::new(py, DeviceArrayF32Py { inner: jaw })?)?;
    d.set_item("teeth", Py::new(py, DeviceArrayF32Py { inner: teeth })?)?;
    d.set_item("lips", Py::new(py, DeviceArrayF32Py { inner: lips })?)?;
    d.set_item("rows", rows)?;
    d.set_item("cols", cols)?;
    d.set_item(
        "context_guard",
        Py::new(
            py,
            CudaContextGuardPy {
                device_id: guard_dev,
                _ctx: guard_ctx,
            },
        )?,
    )?;
    Ok(d)
}

#[inline]
pub fn alligator_into_slices(
    jaw_out: &mut [f64],
    teeth_out: &mut [f64],
    lips_out: &mut [f64],
    input: &AlligatorInput,
    kern: Kernel,
) -> Result<(), AlligatorError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(AlligatorError::EmptyInputData);
    }
    if jaw_out.len() != len || teeth_out.len() != len || lips_out.len() != len {
        return Err(AlligatorError::InvalidJawPeriod {
            period: len,
            data_len: len,
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlligatorError::AllValuesNaN)?;
    let jp = input.get_jaw_period();
    let jo = input.get_jaw_offset();
    let tp = input.get_teeth_period();
    let to = input.get_teeth_offset();
    let lp = input.get_lips_period();
    let lo = input.get_lips_offset();

    if jp == 0 || jp > len {
        return Err(AlligatorError::InvalidJawPeriod {
            period: jp,
            data_len: len,
        });
    }
    if tp == 0 || tp > len {
        return Err(AlligatorError::InvalidTeethPeriod {
            period: tp,
            data_len: len,
        });
    }
    if lp == 0 || lp > len {
        return Err(AlligatorError::InvalidLipsPeriod {
            period: lp,
            data_len: len,
        });
    }
    if jo > len {
        return Err(AlligatorError::InvalidJawOffset {
            offset: jo,
            data_len: len,
        });
    }
    if to > len {
        return Err(AlligatorError::InvalidTeethOffset {
            offset: to,
            data_len: len,
        });
    }
    if lo > len {
        return Err(AlligatorError::InvalidLipsOffset {
            offset: lo,
            data_len: len,
        });
    }

    // Warmup NaN prefixes only, no bulk fill
    let jw = first + jp - 1 + jo;
    let tw = first + tp - 1 + to;
    let lw = first + lp - 1 + lo;
    for v in &mut jaw_out[..jw.min(len)] {
        *v = f64::NAN;
    }
    for v in &mut teeth_out[..tw.min(len)] {
        *v = f64::NAN;
    }
    for v in &mut lips_out[..lw.min(len)] {
        *v = f64::NAN;
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                let _ = alligator_smma_scalar(
                    data, jp, jo, tp, to, lp, lo, first, len, jaw_out, teeth_out, lips_out,
                );
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                let _ = alligator_row_avx2(
                    data, first, jp, jo, tp, to, lp, lo, len, jaw_out, teeth_out, lips_out,
                );
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                let _ = alligator_row_avx512(
                    data, first, jp, jo, tp, to, lp, lo, len, jaw_out, teeth_out, lips_out,
                );
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

/// Writes Alligator outputs into caller-provided buffers without allocating.
///
/// - Preserves NaN warmup prefixes exactly like the Vec-returning API.
/// - The `jaw_out`, `teeth_out`, and `lips_out` slice lengths must equal the input length.
/// - Uses kernel auto-detection (`Kernel::Auto`) to match the standard API behavior.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn alligator_into(
    input: &AlligatorInput,
    jaw_out: &mut [f64],
    teeth_out: &mut [f64],
    lips_out: &mut [f64],
) -> Result<(), AlligatorError> {
    alligator_into_slices(jaw_out, teeth_out, lips_out, input, Kernel::Auto)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alligator_js(
    data: &[f64],
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = AlligatorParams {
        jaw_period: Some(jaw_period),
        jaw_offset: Some(jaw_offset),
        teeth_period: Some(teeth_period),
        teeth_offset: Some(teeth_offset),
        lips_period: Some(lips_period),
        lips_offset: Some(lips_offset),
    };
    let input = AlligatorInput::from_slice(data, params);
    let out = alligator_with_kernel(&input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Return flattened array: [jaw, teeth, lips]
    let mut result = Vec::with_capacity(data.len() * 3);
    result.extend_from_slice(&out.jaw);
    result.extend_from_slice(&out.teeth);
    result.extend_from_slice(&out.lips);
    Ok(result)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alligator_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alligator_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alligator_into(
    in_ptr: *const f64,
    jaw_ptr: *mut f64,
    teeth_ptr: *mut f64,
    lips_ptr: *mut f64,
    len: usize,
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || jaw_ptr.is_null() || teeth_ptr.is_null() || lips_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = AlligatorParams {
            jaw_period: Some(jaw_period),
            jaw_offset: Some(jaw_offset),
            teeth_period: Some(teeth_period),
            teeth_offset: Some(teeth_offset),
            lips_period: Some(lips_period),
            lips_offset: Some(lips_offset),
        };
        let input = AlligatorInput::from_slice(data, params);

        // Check for aliasing - need to check all combinations
        let aliased = in_ptr == jaw_ptr as *const f64
            || in_ptr == teeth_ptr as *const f64
            || in_ptr == lips_ptr as *const f64
            || jaw_ptr == teeth_ptr
            || jaw_ptr == lips_ptr
            || teeth_ptr == lips_ptr;

        if aliased {
            // Use temporary buffers when aliasing detected
            let mut temp_jaw = vec![0.0; len];
            let mut temp_teeth = vec![0.0; len];
            let mut temp_lips = vec![0.0; len];

            alligator_into_slices(
                &mut temp_jaw,
                &mut temp_teeth,
                &mut temp_lips,
                &input,
                detect_best_kernel(),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results to output pointers
            let jaw_out = std::slice::from_raw_parts_mut(jaw_ptr, len);
            let teeth_out = std::slice::from_raw_parts_mut(teeth_ptr, len);
            let lips_out = std::slice::from_raw_parts_mut(lips_ptr, len);

            jaw_out.copy_from_slice(&temp_jaw);
            teeth_out.copy_from_slice(&temp_teeth);
            lips_out.copy_from_slice(&temp_lips);
        } else {
            // Direct write when no aliasing
            let jaw_out = std::slice::from_raw_parts_mut(jaw_ptr, len);
            let teeth_out = std::slice::from_raw_parts_mut(teeth_ptr, len);
            let lips_out = std::slice::from_raw_parts_mut(lips_ptr, len);

            alligator_into_slices(jaw_out, teeth_out, lips_out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alligator_batch_js(
    data: &[f64],
    jaw_period_start: usize,
    jaw_period_end: usize,
    jaw_period_step: usize,
    jaw_offset_start: usize,
    jaw_offset_end: usize,
    jaw_offset_step: usize,
    teeth_period_start: usize,
    teeth_period_end: usize,
    teeth_period_step: usize,
    teeth_offset_start: usize,
    teeth_offset_end: usize,
    teeth_offset_step: usize,
    lips_period_start: usize,
    lips_period_end: usize,
    lips_period_step: usize,
    lips_offset_start: usize,
    lips_offset_end: usize,
    lips_offset_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AlligatorBatchRange {
        jaw_period: (jaw_period_start, jaw_period_end, jaw_period_step),
        jaw_offset: (jaw_offset_start, jaw_offset_end, jaw_offset_step),
        teeth_period: (teeth_period_start, teeth_period_end, teeth_period_step),
        teeth_offset: (teeth_offset_start, teeth_offset_end, teeth_offset_step),
        lips_period: (lips_period_start, lips_period_end, lips_period_step),
        lips_offset: (lips_offset_start, lips_offset_end, lips_offset_step),
    };

    // Use the existing batch function with parallel=false for WASM
    alligator_batch_inner(data, &sweep, Kernel::ScalarBatch, false)
        .map(|output| {
            // Flatten all three arrays into one for JS compatibility
            let mut result =
                Vec::with_capacity((output.jaw.len() + output.teeth.len() + output.lips.len()));
            result.extend_from_slice(&output.jaw);
            result.extend_from_slice(&output.teeth);
            result.extend_from_slice(&output.lips);
            result
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alligator_batch_metadata_js(
    jaw_period_start: usize,
    jaw_period_end: usize,
    jaw_period_step: usize,
    jaw_offset_start: usize,
    jaw_offset_end: usize,
    jaw_offset_step: usize,
    teeth_period_start: usize,
    teeth_period_end: usize,
    teeth_period_step: usize,
    teeth_offset_start: usize,
    teeth_offset_end: usize,
    teeth_offset_step: usize,
    lips_period_start: usize,
    lips_period_end: usize,
    lips_period_step: usize,
    lips_offset_start: usize,
    lips_offset_end: usize,
    lips_offset_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = AlligatorBatchRange {
        jaw_period: (jaw_period_start, jaw_period_end, jaw_period_step),
        jaw_offset: (jaw_offset_start, jaw_offset_end, jaw_offset_step),
        teeth_period: (teeth_period_start, teeth_period_end, teeth_period_step),
        teeth_offset: (teeth_offset_start, teeth_offset_end, teeth_offset_step),
        lips_period: (lips_period_start, lips_period_end, lips_period_step),
        lips_offset: (lips_offset_start, lips_offset_end, lips_offset_step),
    };

    let combos = expand_grid(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut metadata = Vec::with_capacity(combos.len() * 6);

    for combo in combos {
        metadata.push(combo.jaw_period.unwrap() as f64);
        metadata.push(combo.jaw_offset.unwrap() as f64);
        metadata.push(combo.teeth_period.unwrap() as f64);
        metadata.push(combo.teeth_offset.unwrap() as f64);
        metadata.push(combo.lips_period.unwrap() as f64);
        metadata.push(combo.lips_offset.unwrap() as f64);
    }

    Ok(metadata)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AlligatorBatchJsOutput {
    pub jaw: Vec<f64>,
    pub teeth: Vec<f64>,
    pub lips: Vec<f64>,
    pub combos: Vec<AlligatorParams>,
    pub rows: usize, // number of parameter combinations
    pub cols: usize, // data length
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AlligatorBatchConfig {
    pub jaw_period_range: (usize, usize, usize),
    pub jaw_offset_range: (usize, usize, usize),
    pub teeth_period_range: (usize, usize, usize),
    pub teeth_offset_range: (usize, usize, usize),
    pub lips_period_range: (usize, usize, usize),
    pub lips_offset_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = alligator_batch)]
pub fn alligator_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: AlligatorBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = AlligatorBatchRange {
        jaw_period: config.jaw_period_range,
        jaw_offset: config.jaw_offset_range,
        teeth_period: config.teeth_period_range,
        teeth_offset: config.teeth_offset_range,
        lips_period: config.lips_period_range,
        lips_offset: config.lips_offset_range,
    };
    let rows = expand_grid(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .len();
    let cols = data.len();
    let mut jaw = vec![f64::NAN; rows * cols];
    let mut teeth = vec![f64::NAN; rows * cols];
    let mut lips = vec![f64::NAN; rows * cols];

    let combos = alligator_batch_inner_into(
        data,
        &sweep,
        Kernel::ScalarBatch,
        false,
        &mut jaw,
        &mut teeth,
        &mut lips,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js = AlligatorBatchJsOutput {
        jaw,
        teeth,
        lips,
        combos,
        rows,
        cols,
    };
    serde_wasm_bindgen::to_value(&js).map_err(|e| JsValue::from_str(&format!("serde: {e}")))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alligator_batch_into(
    in_ptr: *const f64,
    jaw_out_ptr: *mut f64,
    teeth_out_ptr: *mut f64,
    lips_out_ptr: *mut f64,
    len: usize,
    // ranges
    jp_s: usize,
    jp_e: usize,
    jp_step: usize,
    jo_s: usize,
    jo_e: usize,
    jo_step: usize,
    tp_s: usize,
    tp_e: usize,
    tp_step: usize,
    to_s: usize,
    to_e: usize,
    to_step: usize,
    lp_s: usize,
    lp_e: usize,
    lp_step: usize,
    lo_s: usize,
    lo_e: usize,
    lo_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null()
        || jaw_out_ptr.is_null()
        || teeth_out_ptr.is_null()
        || lips_out_ptr.is_null()
    {
        return Err(JsValue::from_str(
            "null pointer passed to alligator_batch_into",
        ));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = AlligatorBatchRange {
            jaw_period: (jp_s, jp_e, jp_step),
            jaw_offset: (jo_s, jo_e, jo_step),
            teeth_period: (tp_s, tp_e, tp_step),
            teeth_offset: (to_s, to_e, to_step),
            lips_period: (lp_s, lp_e, lp_step),
            lips_offset: (lo_s, lo_e, lo_step),
        };
    let combos = expand_grid(&sweep).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;

        let jaw = std::slice::from_raw_parts_mut(jaw_out_ptr, rows * cols);
        let teeth = std::slice::from_raw_parts_mut(teeth_out_ptr, rows * cols);
        let lips = std::slice::from_raw_parts_mut(lips_out_ptr, rows * cols);

        alligator_batch_inner_into(data, &sweep, Kernel::ScalarBatch, false, jaw, teeth, lips)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}
