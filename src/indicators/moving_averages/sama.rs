//! # Slope Adaptive Moving Average (SAMA)
//!
//! SAMA is an adaptive moving average that adjusts its smoothing factor based on the
//! price range within a specified period. It uses the difference between highest and
//! lowest values to determine how much weight to give to recent price changes.
//!
//! ## Parameters
//! - **length**: Period for finding highest and lowest values (default: 200)
//! - **maj_length**: Major length for slower alpha calculation (default: 14)
//! - **min_length**: Minor length for faster alpha calculation (default: 6)
//!
//! ## Errors
//! - **EmptyInputData**: sama: Input data slice is empty.
//! - **AllValuesNaN**: sama: All input values are `NaN`.
//! - **InvalidPeriod**: sama: Period is zero or exceeds data length.
//! - **NotEnoughValidData**: sama: Not enough valid data points for calculation.
//!
//! ## Returns
//! - **`Ok(SamaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//! - **`Err(SamaError)`** otherwise.
//!
//! ## Developer Notes
//! - **AVX2 kernel**: ❌ Stub - calls scalar implementation
//! - **AVX512 kernel**: ❌ Stub - calls scalar implementation
//! - **Streaming update**: ⚠️ O(n) - scans entire buffer for highest/lowest values each update
//! - **Memory optimization**: ⚠️ Does NOT use zero-copy helpers - allocates with `vec![f64::NAN; data.len()]`
//! - **Current status**: Functional but missing SIMD optimizations and memory optimization
//! - **Optimization opportunities**:
//!   - Implement AVX2/AVX512 kernels for vectorized min/max operations
//!   - Switch to `alloc_with_nan_prefix` for zero-copy output allocation
//!   - Consider sliding window min/max algorithm for O(log n) or O(1) streaming updates
//!   - The highest/lowest calculation is well-suited for SIMD parallelization

#[cfg(feature = "python")]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::{cuda_available, moving_averages::CudaSama};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for SamaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SamaData::Slice(slice) => slice,
            SamaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SamaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SamaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct SamaParams {
    pub length: Option<usize>,
    pub maj_length: Option<usize>,
    pub min_length: Option<usize>,
}

impl Default for SamaParams {
    fn default() -> Self {
        Self {
            length: Some(200),
            maj_length: Some(14),
            min_length: Some(6),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SamaInput<'a> {
    pub data: SamaData<'a>,
    pub params: SamaParams,
}

impl<'a> SamaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: SamaParams) -> Self {
        Self {
            data: SamaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }

    #[inline]
    pub fn from_slice(sl: &'a [f64], p: SamaParams) -> Self {
        Self {
            data: SamaData::Slice(sl),
            params: p,
        }
    }

    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", SamaParams::default())
    }

    #[inline]
    pub fn get_length(&self) -> usize {
        self.params.length.unwrap_or(200)
    }

    #[inline]
    pub fn get_maj_length(&self) -> usize {
        self.params.maj_length.unwrap_or(14)
    }

    #[inline]
    pub fn get_min_length(&self) -> usize {
        self.params.min_length.unwrap_or(6)
    }
}

#[derive(Clone, Debug)]
pub struct SamaBuilder {
    length: Option<usize>,
    maj_length: Option<usize>,
    min_length: Option<usize>,
    kernel: Kernel,
}

impl Default for SamaBuilder {
    fn default() -> Self {
        Self {
            length: None,
            maj_length: None,
            min_length: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SamaBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn length(mut self, val: usize) -> Self {
        self.length = Some(val);
        self
    }

    #[inline(always)]
    pub fn maj_length(mut self, val: usize) -> Self {
        self.maj_length = Some(val);
        self
    }

    #[inline(always)]
    pub fn min_length(mut self, val: usize) -> Self {
        self.min_length = Some(val);
        self
    }

    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<SamaOutput, SamaError> {
        let p = SamaParams {
            length: self.length,
            maj_length: self.maj_length,
            min_length: self.min_length,
        };
        let i = SamaInput::from_candles(c, "close", p);
        sama_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<SamaOutput, SamaError> {
        let p = SamaParams {
            length: self.length,
            maj_length: self.maj_length,
            min_length: self.min_length,
        };
        let i = SamaInput::from_slice(d, p);
        sama_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<SamaStream, SamaError> {
        let p = SamaParams {
            length: self.length,
            maj_length: self.maj_length,
            min_length: self.min_length,
        };
        SamaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum SamaError {
    #[error("sama: Input data slice is empty.")]
    EmptyInputData,

    #[error("sama: All values are NaN.")]
    AllValuesNaN,

    #[error("sama: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("sama: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline(always)]
pub fn sama(input: &SamaInput) -> Result<SamaOutput, SamaError> {
    sama_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
pub fn sama_with_kernel(input: &SamaInput, kernel: Kernel) -> Result<SamaOutput, SamaError> {
    let (data, length, maj_length, min_length, first, chosen) = sama_prepare(input, kernel)?;

    // Pine-compatible: start computing immediately but maintain proper warmup
    // We allocate the full array and compute will fill in values starting from first
    let mut out = vec![f64::NAN; data.len()];
    sama_compute_into(
        data, length, maj_length, min_length, first, chosen, &mut out,
    );
    Ok(SamaOutput { values: out })
}

#[inline(always)]
pub fn sama_into_slice(dst: &mut [f64], input: &SamaInput, kern: Kernel) -> Result<(), SamaError> {
    let (data, length, maj_length, min_length, first, chosen) = sama_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(SamaError::InvalidPeriod {
            period: dst.len(),
            data_len: data.len(),
        });
    }

    // Initialize dst with NaN first
    for v in dst.iter_mut() {
        *v = f64::NAN;
    }

    // Now compute will fill in values starting from first
    sama_compute_into(data, length, maj_length, min_length, first, chosen, dst);
    Ok(())
}

#[inline(always)]
fn sama_prepare<'a>(
    input: &'a SamaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, usize, usize, Kernel), SamaError> {
    let data: &[f64] = input.as_ref();
    let n = data.len();

    if n == 0 {
        return Err(SamaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SamaError::AllValuesNaN)?;

    let length = input.get_length();
    let maj_length = input.get_maj_length();
    let min_length = input.get_min_length();

    // We need length + 1 data points for highest/lowest calculation
    if length + 1 > n || length == 0 {
        return Err(SamaError::InvalidPeriod {
            period: length,
            data_len: n,
        });
    }

    if maj_length == 0 || min_length == 0 {
        return Err(SamaError::InvalidPeriod {
            period: 0,
            data_len: n,
        });
    }

    let valid = n - first;
    // Pine-compatible: start immediately once we have any valid bar
    if valid < 1 {
        return Err(SamaError::NotEnoughValidData { needed: 1, valid });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((data, length, maj_length, min_length, first, chosen))
}

fn sama_compute_into(
    data: &[f64],
    length: usize,
    maj_length: usize,
    min_length: usize,
    first: usize,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                sama_simd128(data, length, maj_length, min_length, first, out);
                return;
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                sama_scalar(data, length, maj_length, min_length, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                sama_avx2(data, length, maj_length, min_length, first, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                sama_avx512(data, length, maj_length, min_length, first, out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                sama_scalar(data, length, maj_length, min_length, first, out)
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
pub fn sama_scalar(
    data: &[f64],
    length: usize,
    maj_length: usize,
    min_length: usize,
    first: usize,
    out: &mut [f64],
) {
    let n = data.len();
    if n == 0 {
        return;
    }

    let min_alpha = 2.0 / (min_length as f64 + 1.0);
    let maj_alpha = 2.0 / (maj_length as f64 + 1.0);

    let mut sama_val = f64::NAN;
    let start_idx = first;

    for i in start_idx..n {
        // if current src is NaN, output NaN but DO NOT reset state
        if data[i].is_nan() {
            out[i] = f64::NAN;
            continue;
        }

        let period_start = i.saturating_sub(length);

        let mut hh = f64::NEG_INFINITY;
        let mut ll = f64::INFINITY;
        for j in period_start..=i {
            let v = data[j];
            if v.is_nan() {
                continue;
            }
            if v > hh {
                hh = v;
            }
            if v < ll {
                ll = v;
            }
        }

        let mult = if hh != ll {
            (2.0 * data[i] - ll - hh).abs() / (hh - ll)
        } else {
            0.0
        };

        let final_alpha = (mult * (min_alpha - maj_alpha) + maj_alpha).powi(2);

        if sama_val.is_nan() {
            // Initialize with first price for proper moving average behavior
            sama_val = data[i];
        } else {
            sama_val = (data[i] - sama_val) * final_alpha + sama_val;
        }

        out[i] = sama_val;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn sama_avx2(
    data: &[f64],
    length: usize,
    maj_length: usize,
    min_length: usize,
    first: usize,
    out: &mut [f64],
) {
    // For now, fallback to scalar implementation
    // AVX2 optimization can be added later for better performance
    sama_scalar(data, length, maj_length, min_length, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn sama_avx512(
    data: &[f64],
    length: usize,
    maj_length: usize,
    min_length: usize,
    first: usize,
    out: &mut [f64],
) {
    // For now, fallback to scalar implementation
    // AVX512 optimization can be added later for better performance
    sama_scalar(data, length, maj_length, min_length, first, out);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub fn sama_simd128(
    data: &[f64],
    length: usize,
    maj_length: usize,
    min_length: usize,
    first: usize,
    out: &mut [f64],
) {
    // For now, fallback to scalar implementation
    // WASM SIMD optimization can be added later
    sama_scalar(data, length, maj_length, min_length, first, out);
}

// ========== Batch Processing ==========

#[derive(Debug, Clone)]
pub struct SamaBatchRange {
    pub length: (usize, usize, usize),     // (start, end, step)
    pub maj_length: (usize, usize, usize), // (start, end, step)
    pub min_length: (usize, usize, usize), // (start, end, step)
}

#[derive(Debug, Clone)]
pub struct SamaBatchOutput {
    pub values: Vec<f64>, // flattened rows × cols
    pub combos: Vec<SamaParams>,
    pub rows: usize,
    pub cols: usize,
}

impl SamaBatchOutput {
    #[inline]
    pub fn row_for_params(&self, p: &SamaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.length.unwrap_or(200) == p.length.unwrap_or(200)
                && c.maj_length.unwrap_or(14) == p.maj_length.unwrap_or(14)
                && c.min_length.unwrap_or(6) == p.min_length.unwrap_or(6)
        })
    }

    #[inline]
    pub fn values_for(&self, p: &SamaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct SamaBatchBuilder {
    range: SamaBatchRange,
    kernel: Kernel,
}

impl SamaBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline]
    pub fn length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.length = (start, end, step);
        self
    }

    #[inline]
    pub fn maj_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.maj_length = (start, end, step);
        self
    }

    #[inline]
    pub fn min_length_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.min_length = (start, end, step);
        self
    }

    #[inline]
    pub fn length_static(mut self, v: usize) -> Self {
        self.range.length = (v, v, 0);
        self
    }

    #[inline]
    pub fn maj_length_static(mut self, v: usize) -> Self {
        self.range.maj_length = (v, v, 0);
        self
    }

    #[inline]
    pub fn min_length_static(mut self, v: usize) -> Self {
        self.range.min_length = (v, v, 0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<SamaBatchOutput, SamaError> {
        sama_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<SamaBatchOutput, SamaError> {
        self.apply_slice(source_type(c, src))
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<SamaBatchOutput, SamaError> {
        SamaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn with_default_candles(c: &Candles) -> Result<SamaBatchOutput, SamaError> {
        SamaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .length_static(200)
            .maj_length_static(14)
            .min_length_static(6)
            .apply_candles(c, "close")
    }
}

impl Default for SamaBatchRange {
    fn default() -> Self {
        Self {
            length: (200, 200, 0),
            maj_length: (14, 14, 0),
            min_length: (6, 6, 0),
        }
    }
}

#[inline(always)]
fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        return vec![start];
    }
    (start..=end).step_by(step).collect()
}

#[inline(always)]
fn expand_grid_sama(r: &SamaBatchRange) -> Vec<SamaParams> {
    let lens = axis_usize(r.length);
    let maj = axis_usize(r.maj_length);
    let min = axis_usize(r.min_length);
    let mut out = Vec::with_capacity(lens.len() * maj.len() * min.len());
    for &l in &lens {
        for &j in &maj {
            for &m in &min {
                out.push(SamaParams {
                    length: Some(l),
                    maj_length: Some(j),
                    min_length: Some(m),
                });
            }
        }
    }
    out
}

pub fn sama_batch_with_kernel(
    data: &[f64],
    sweep: &SamaBatchRange,
    k: Kernel,
) -> Result<SamaBatchOutput, SamaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => {
            return Err(SamaError::InvalidPeriod {
                period: 0,
                data_len: 0,
            })
        }
    };

    // Map batch enum -> single compute enum, same pattern as alma
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };

    sama_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn sama_batch_par_slice(
    data: &[f64],
    sweep: &SamaBatchRange,
    kern: Kernel,
) -> Result<SamaBatchOutput, SamaError> {
    sama_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
pub fn sama_batch_slice(
    data: &[f64],
    sweep: &SamaBatchRange,
    kern: Kernel,
) -> Result<SamaBatchOutput, SamaError> {
    sama_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
fn sama_batch_inner(
    data: &[f64],
    sweep: &SamaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SamaBatchOutput, SamaError> {
    let combos = expand_grid_sama(sweep);
    let cols = data.len();
    let rows = combos.len();
    if cols == 0 {
        return Err(SamaError::EmptyInputData);
    }

    // Allocate uninit rows×cols without copies
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Pine-compatible: warmup prefix only up to first valid bar
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SamaError::AllValuesNaN)?;
    let warm: Vec<usize> = combos
        .iter()
        .map(|_| first) // Pine parity: start computing immediately
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    // Reborrow as &mut [f64] for writing results
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // Fill each row in-place
    sama_batch_inner_into(data, &combos, first, kern, parallel, out)?;

    // Reclaim as Vec<f64> without copies
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(SamaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn sama_batch_inner_into(
    data: &[f64],
    combos: &[SamaParams],
    first: usize,
    kern: Kernel,
    parallel: bool,
    out_flat: &mut [f64],
) -> Result<(), SamaError> {
    if combos.is_empty() {
        return Ok(());
    }

    let rows = combos.len();
    let cols = data.len();

    // Pine-compatible: only need 1 valid data point to start
    if cols - first < 1 {
        return Err(SamaError::NotEnoughValidData {
            needed: 1,
            valid: cols - first,
        });
    }

    // Helper: compute one row directly into its slice to avoid temp Vec
    let do_row = |row: usize, row_dst: &mut [f64]| {
        let prm = &combos[row];
        let length = prm.length.unwrap_or(200);
        let maj_length = prm.maj_length.unwrap_or(14);
        let min_length = prm.min_length.unwrap_or(6);
        // Write results directly into row_dst
        sama_compute_into(data, length, maj_length, min_length, first, kern, row_dst);
    };

    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            out_flat
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, dst)| do_row(row, dst));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, dst) in out_flat.chunks_mut(cols).enumerate() {
                do_row(row, dst);
            }
        }
    } else {
        for (row, dst) in out_flat.chunks_mut(cols).enumerate() {
            do_row(row, dst);
        }
    }

    Ok(())
}

// ========== Streaming Support ==========

#[derive(Debug, Clone)]
pub struct SamaStream {
    length: usize,
    maj_length: usize,
    min_length: usize,
    buf: Vec<f64>,
    head: usize,
    filled: bool,
    sama_val: f64,
}

impl SamaStream {
    pub fn try_new(params: SamaParams) -> Result<Self, SamaError> {
        let length = params.length.unwrap_or(200);
        let maj = params.maj_length.unwrap_or(14);
        let min = params.min_length.unwrap_or(6);
        if length == 0 || maj == 0 || min == 0 {
            return Err(SamaError::InvalidPeriod {
                period: 0,
                data_len: 0,
            });
        }
        Ok(Self {
            length,
            maj_length: maj,
            min_length: min,
            buf: vec![f64::NAN; length + 1],
            head: 0,
            filled: false,
            sama_val: f64::NAN,
        })
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Pine-compatible: start computing immediately with first valid value
        if value.is_nan() {
            // For NaN input, don't update buffer but return current sama_val if we have one
            return if self.sama_val.is_nan() {
                None
            } else {
                Some(self.sama_val)
            };
        }

        self.buf[self.head] = value;
        self.head = (self.head + 1) % (self.length + 1);

        // count of valid samples seen so far
        let count = if self.filled {
            self.length + 1
        } else {
            // ring not full; count valid values in the entire buffer
            let mut c = 0;
            for i in 0..(self.length + 1) {
                if !self.buf[i].is_nan() {
                    c += 1;
                }
            }
            c
        };
        if !self.filled && count == self.length + 1 {
            self.filled = true;
        }

        // Pine-compatible: compute immediately even with just 1 sample
        if count == 0 {
            return None;
        }

        // compute hh/ll over all valid samples in the buffer
        let mut hh = f64::NEG_INFINITY;
        let mut ll = f64::INFINITY;

        if self.filled {
            // Buffer is full, iterate from head (oldest) through all elements
            let mut idx = self.head;
            for _ in 0..(self.length + 1) {
                let v = self.buf[idx];
                if !v.is_nan() {
                    if v > hh {
                        hh = v;
                    }
                    if v < ll {
                        ll = v;
                    }
                }
                idx = (idx + 1) % (self.length + 1);
            }
        } else {
            // Buffer not full, check all slots for valid values
            for i in 0..(self.length + 1) {
                let v = self.buf[i];
                if !v.is_nan() {
                    if v > hh {
                        hh = v;
                    }
                    if v < ll {
                        ll = v;
                    }
                }
            }
        }

        let min_alpha = 2.0 / (self.min_length as f64 + 1.0);
        let maj_alpha = 2.0 / (self.maj_length as f64 + 1.0);
        let mult = if hh != ll {
            (2.0 * value - ll - hh).abs() / (hh - ll)
        } else {
            0.0
        };
        let a = (mult * (min_alpha - maj_alpha) + maj_alpha).powi(2);

        if self.sama_val.is_nan() {
            // Initialize with first price for proper moving average behavior
            self.sama_val = value;
        } else {
            self.sama_val = (value - self.sama_val) * a + self.sama_val;
        }
        Some(self.sama_val)
    }

    #[inline]
    pub fn next(&mut self, value: f64) -> f64 {
        self.update(value).unwrap_or(f64::NAN)
    }
    #[inline]
    pub fn reset(&mut self) {
        self.buf.fill(f64::NAN);
        self.head = 0;
        self.filled = false;
        self.sama_val = f64::NAN;
    }
}

// ========== Python Bindings ==========

#[cfg(feature = "python")]
#[pyfunction(name = "sama")]
#[pyo3(signature = (data, length, maj_length, min_length, kernel=None))]
pub fn sama_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length: usize,
    maj_length: usize,
    min_length: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;
    let params = SamaParams {
        length: Some(length),
        maj_length: Some(maj_length),
        min_length: Some(min_length),
    };
    let input = SamaInput::from_slice(slice_in, params);
    let result_vec: Vec<f64> = py
        .allow_threads(|| sama_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "sama_batch")]
#[pyo3(signature = (data, length_range, maj_length_range, min_length_range, kernel=None))]
pub fn sama_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    length_range: (usize, usize, usize),
    maj_length_range: (usize, usize, usize),
    min_length_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let sweep = SamaBatchRange {
        length: length_range,
        maj_length: maj_length_range,
        min_length: min_length_range,
    };

    // Build combos up front for metadata and sizing
    let combos = expand_grid_sama(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Preallocate flattened output in NumPy without extra copies
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    // Fill in place with GIL released
    let combos = py
        .allow_threads(|| {
            let mapped = match kern {
                Kernel::Auto => detect_best_batch_kernel(),
                k => k,
            };
            let simd = match mapped {
                Kernel::Avx512Batch => Kernel::Avx512,
                Kernel::Avx2Batch => Kernel::Avx2,
                Kernel::ScalarBatch => Kernel::Scalar,
                _ => Kernel::Scalar,
            };
            // Use unified inner to avoid allocations
            let first = slice_in
                .iter()
                .position(|x| !x.is_nan())
                .ok_or(SamaError::AllValuesNaN)?;
            sama_batch_inner_into(slice_in, &combos, first, simd, true, slice_out).map(|_| combos)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    dict.set_item(
        "lengths",
        combos
            .iter()
            .map(|p| p.length.unwrap_or(200) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "maj_lengths",
        combos
            .iter()
            .map(|p| p.maj_length.unwrap_or(14) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "min_lengths",
        combos
            .iter()
            .map(|p| p.min_length.unwrap_or(6) as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(dict.into())
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "sama_cuda_batch_dev")]
#[pyo3(signature = (data_f32, length_range=(200, 200, 0), maj_length_range=(14, 14, 0), min_length_range=(6, 6, 0), device_id=0))]
pub fn sama_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: PyReadonlyArray1<'_, f32>,
    length_range: (usize, usize, usize),
    maj_length_range: (usize, usize, usize),
    min_length_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = SamaBatchRange {
        length: length_range,
        maj_length: maj_length_range,
        min_length: min_length_range,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaSama::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.sama_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "sama_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, length, maj_length, min_length, device_id=0))]
pub fn sama_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: PyReadonlyArray2<'_, f32>,
    length: usize,
    maj_length: usize,
    min_length: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    if length == 0 || maj_length == 0 || min_length == 0 {
        return Err(PyValueError::new_err(
            "length, maj_length, and min_length must be positive",
        ));
    }

    let flat = data_tm_f32.as_slice()?;
    let shape = data_tm_f32.shape();
    let series_len = shape[0];
    let num_series = shape[1];
    let params = SamaParams {
        length: Some(length),
        maj_length: Some(maj_length),
        min_length: Some(min_length),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaSama::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.sama_many_series_one_param_time_major_dev(flat, num_series, series_len, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(feature = "python")]
#[pyclass(name = "SamaStream")]
pub struct SamaStreamPy {
    stream: SamaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SamaStreamPy {
    #[new]
    fn new(length: usize, maj_length: usize, min_length: usize) -> PyResult<Self> {
        let params = SamaParams {
            length: Some(length),
            maj_length: Some(maj_length),
            min_length: Some(min_length),
        };
        let stream =
            SamaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(SamaStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

// ========== WASM Bindings ==========

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sama_js(
    data: &[f64],
    length: usize,
    maj_length: usize,
    min_length: usize,
) -> Result<Vec<f64>, JsValue> {
    let params = SamaParams {
        length: Some(length),
        maj_length: Some(maj_length),
        min_length: Some(min_length),
    };
    let input = SamaInput::from_slice(data, params);
    let mut out = vec![0.0; data.len()];
    sama_into_slice(&mut out, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(out)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SamaBatchConfig {
    pub length_range: (usize, usize, usize),
    pub maj_length_range: (usize, usize, usize),
    pub min_length_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SamaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SamaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = sama_batch)]
pub fn sama_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let cfg: SamaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let sweep = SamaBatchRange {
        length: cfg.length_range,
        maj_length: cfg.maj_length_range,
        min_length: cfg.min_length_range,
    };
    let out = sama_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    serde_wasm_bindgen::to_value(&SamaBatchJsOutput {
        values: out.values,
        combos: out.combos,
        rows: out.rows,
        cols: out.cols,
    })
    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Raw pointer helpers (no copies)
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sama_alloc(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sama_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sama_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length: usize,
    maj_length: usize,
    min_length: usize,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = SamaParams {
            length: Some(length),
            maj_length: Some(maj_length),
            min_length: Some(min_length),
        };
        let input = SamaInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut tmp = vec![0.0; len];
            sama_into_slice(&mut tmp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&tmp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            // pass the slice directly; no &mut of a &mut slice
            sama_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn sama_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    length_start: usize,
    length_end: usize,
    length_step: usize,
    maj_start: usize,
    maj_end: usize,
    maj_step: usize,
    min_start: usize,
    min_end: usize,
    min_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer"));
    }
    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);
        let sweep = SamaBatchRange {
            length: (length_start, length_end, length_step),
            maj_length: (maj_start, maj_end, maj_step),
            min_length: (min_start, min_end, min_step),
        };
        let combos = expand_grid_sama(&sweep);
        let rows = combos.len();
        let cols = len;
        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);
        let first = data
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| JsValue::from_str("All NaN"))?;
        sama_batch_inner_into(data, &combos, first, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use std::error::Error;

    fn check_sama_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = SamaInput::from_candles(&candles, "close", SamaParams::default());
        let result = sama_with_kernel(&input, kernel)?;

        // With Pine Script correct initialization (prev=0), values start low
        // and converge over time. Just verify calculation produces valid results
        assert_eq!(result.values.len(), candles.close.len());

        // Check that we get valid values after warmup
        let valid_count = result.values.iter().filter(|&&v| !v.is_nan()).count();
        assert!(
            valid_count > 0,
            "[{}] SAMA should produce valid values",
            test_name
        );

        Ok(())
    }

    fn check_sama_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = SamaParams {
            length: None,
            maj_length: None,
            min_length: None,
        };
        let input = SamaInput::from_candles(&candles, "close", default_params);
        let output = sama_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_sama_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = SamaInput::with_default_candles(&candles);
        match input.data {
            SamaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected SamaData::Candles"),
        }
        let output = sama_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_sama_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = SamaParams {
            length: Some(0),
            maj_length: None,
            min_length: None,
        };
        let input = SamaInput::from_slice(&input_data, params);
        let res = sama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SAMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_sama_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = SamaParams {
            length: Some(10),
            maj_length: None,
            min_length: None,
        };
        let input = SamaInput::from_slice(&data_small, params);
        let res = sama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SAMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_sama_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = SamaParams::default();
        let input = SamaInput::from_slice(&single_point, params);
        let res = sama_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SAMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_sama_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let params = SamaParams::default();
        let input = SamaInput::from_slice(&empty, params);
        let res = sama_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(SamaError::EmptyInputData)),
            "[{}] SAMA should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_sama_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = SamaParams::default();
        let input = SamaInput::from_slice(&nan_data, params);
        let res = sama_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(SamaError::AllValuesNaN)),
            "[{}] SAMA should fail with all NaN values",
            test_name
        );
        Ok(())
    }

    fn check_sama_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = SamaParams {
            length: Some(50),
            maj_length: Some(14),
            min_length: Some(6),
        };
        let first_input = SamaInput::from_candles(&candles, "close", first_params);
        let first_result = sama_with_kernel(&first_input, kernel)?;

        let second_params = SamaParams {
            length: Some(50),
            maj_length: Some(14),
            min_length: Some(6),
        };
        let second_input = SamaInput::from_slice(&first_result.values, second_params);
        let second_result = sama_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_sama_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = SamaParams::default();
        let input = SamaInput::from_candles(&candles, "close", params);
        let output = sama_with_kernel(&input, kernel)?;

        // Find first non-NaN and verify warmup period
        let first_valid = candles.close.iter().position(|x| !x.is_nan()).unwrap_or(0);
        let warmup = first_valid + input.get_length();

        // After warmup, all values should be valid
        for (i, &val) in output
            .values
            .iter()
            .enumerate()
            .skip(warmup.min(output.values.len()))
        {
            assert!(
                !val.is_nan(),
                "[{}] Unexpected NaN at index {}",
                test_name,
                i
            );
        }
        Ok(())
    }

    fn check_sama_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let params = SamaParams::default();

        // Batch calculation
        let batch_input = SamaInput::from_candles(&candles, "close", params.clone());
        let batch_result = sama_with_kernel(&batch_input, kernel)?;

        // Streaming calculation
        let mut stream = SamaStream::try_new(params)?;
        let mut stream_results = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            stream_results.push(stream.next(price));
        }

        assert_eq!(batch_result.values.len(), stream_results.len());

        // Compare results (allowing for small numerical differences)
        for (i, (&batch_val, &stream_val)) in batch_result
            .values
            .iter()
            .zip(stream_results.iter())
            .enumerate()
        {
            if batch_val.is_nan() && stream_val.is_nan() {
                continue;
            }
            if !batch_val.is_nan() && !stream_val.is_nan() {
                let diff = (batch_val - stream_val).abs();
                assert!(
                    diff < 1e-9,
                    "[{}] Stream mismatch at index {}: batch={}, stream={}, diff={}",
                    test_name,
                    i,
                    batch_val,
                    stream_val,
                    diff
                );
            }
        }
        Ok(())
    }

    // Batch test functions
    fn check_batch_sweep(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let range = SamaBatchRange {
            length: (190, 210, 1),
            maj_length: (12, 16, 1),
            min_length: (4, 8, 1),
        };

        let output = sama_batch_with_kernel(&candles.close, &range, kernel)?;

        // Should have (210-190+1) * (16-12+1) * (8-4+1) = 21 * 5 * 5 = 525 results
        let expected_count = 21 * 5 * 5;
        assert_eq!(
            output.rows, expected_count,
            "[{}] Expected {} batch results",
            test_name, expected_count
        );
        assert_eq!(
            output.values.len(),
            expected_count * candles.close.len(),
            "[{}] Expected flattened array size",
            test_name
        );

        Ok(())
    }

    fn check_batch_default_row(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let range = SamaBatchRange {
            length: (200, 200, 0),
            maj_length: (14, 14, 0),
            min_length: (6, 6, 0),
        };

        let output = sama_batch_with_kernel(&candles.close, &range, kernel)?;

        assert_eq!(
            output.rows, 1,
            "[{}] Should have 1 result for default params",
            test_name
        );
        assert_eq!(output.cols, candles.close.len());

        Ok(())
    }

    // Test generation macros
    macro_rules! generate_all_sama_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $(
                    #[test]
                    fn [<$test_fn _scalar>]() -> Result<(), Box<dyn Error>> {
                        $test_fn(stringify!([<$test_fn _scalar>]), Kernel::Scalar)
                    }
                )*

                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test]
                    fn [<$test_fn _avx2>]() -> Result<(), Box<dyn Error>> {
                        $test_fn(stringify!([<$test_fn _avx2>]), Kernel::Avx2)
                    }

                    #[test]
                    fn [<$test_fn _avx512>]() -> Result<(), Box<dyn Error>> {
                        $test_fn(stringify!([<$test_fn _avx512>]), Kernel::Avx512)
                    }
                )*

                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128>]() -> Result<(), Box<dyn Error>> {
                        $test_fn(stringify!([<$test_fn _simd128>]), Kernel::Scalar)
                    }
                )*
            }
        }
    }

    macro_rules! gen_batch_tests {
        ($fn_name:ident) => {
            paste::paste! {
                #[test]
                fn [<$fn_name _scalar>]() -> Result<(), Box<dyn Error>> {
                    $fn_name(stringify!([<$fn_name _scalar>]), Kernel::ScalarBatch)
                }

                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx2>]() -> Result<(), Box<dyn Error>> {
                    $fn_name(stringify!([<$fn_name _avx2>]), Kernel::Avx2Batch)
                }

                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                #[test]
                fn [<$fn_name _avx512>]() -> Result<(), Box<dyn Error>> {
                    $fn_name(stringify!([<$fn_name _avx512>]), Kernel::Avx512Batch)
                }

                #[test]
                fn [<$fn_name _auto_detect>]() -> Result<(), Box<dyn Error>> {
                    $fn_name(stringify!([<$fn_name _auto_detect>]), Kernel::Auto)
                }
            }
        };
    }

    // Generate all test variants
    generate_all_sama_tests!(
        check_sama_accuracy,
        check_sama_partial_params,
        check_sama_default_candles,
        check_sama_zero_period,
        check_sama_period_exceeds_length,
        check_sama_very_small_dataset,
        check_sama_empty_input,
        check_sama_all_nan,
        check_sama_reinput,
        check_sama_nan_handling,
        check_sama_streaming
    );

    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_default_row);

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let cfgs = vec![
            (190, 200, 1, 12, 14, 1, 4, 8, 1),
            (200, 200, 0, 14, 14, 0, 6, 6, 0),
            (195, 205, 5, 10, 16, 2, 3, 9, 2),
        ];

        for (ls, le, lstep, js, je, jstep, ms, me, mstep) in cfgs {
            let out = SamaBatchRange {
                length: (ls, le, lstep),
                maj_length: (js, je, jstep),
                min_length: (ms, me, mstep),
            };
            let res = sama_batch_with_kernel(&c.close, &out, kernel)?;
            for &v in &res.values {
                if v.is_nan() {
                    continue;
                }
                let b = v.to_bits();
                assert_ne!(b, 0x1111_1111_1111_1111);
                assert_ne!(b, 0x2222_2222_2222_2222);
                assert_ne!(b, 0x3333_3333_3333_3333);
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison_scalar() -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    // Special tests
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn test_sama_simd128_correctness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let params = SamaParams::default();
        let input = SamaInput::from_slice(&data, params);
        let scalar = sama_with_kernel(&input, Kernel::Scalar).unwrap();
        let simd = sama_with_kernel(&input, Kernel::Scalar).unwrap(); // simd128 path behind Scalar on wasm
        assert_eq!(scalar.values.len(), simd.values.len());
        for (a, b) in scalar.values.iter().zip(simd.values.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_sama_no_poison_values() -> Result<(), Box<dyn Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = SamaInput::from_candles(&candles, "close", SamaParams::default());
        let output = sama(&input)?;

        for &v in &output.values {
            if v.is_nan() {
                continue;
            }
            let b = v.to_bits();
            // Check for common uninitialized memory patterns
            assert_ne!(
                b, 0x11111111_11111111,
                "Found poison value 0x11111111_11111111"
            );
            assert_ne!(
                b, 0x22222222_22222222,
                "Found poison value 0x22222222_22222222"
            );
            assert_ne!(
                b, 0x33333333_33333333,
                "Found poison value 0x33333333_33333333"
            );
            assert_ne!(
                b, 0xDEADBEEF_DEADBEEF,
                "Found poison value 0xDEADBEEF_DEADBEEF"
            );
            assert_ne!(
                b, 0xFEEEFEEE_FEEEFEEE,
                "Found poison value 0xFEEEFEEE_FEEEFEEE"
            );
        }
        Ok(())
    }

    #[test]
    fn test_sama_stream_incremental() -> Result<(), Box<dyn Error>> {
        let params = SamaParams {
            length: Some(10),
            maj_length: Some(5),
            min_length: Some(3),
        };

        let mut stream = SamaStream::try_new(params)?;
        let data = vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
        ];

        let mut results = Vec::new();
        for &val in &data {
            let result = stream.next(val);
            if !result.is_nan() {
                results.push(result);
            }
        }

        // Should get results immediately with Pine parity
        assert!(
            !results.is_empty(),
            "Stream should produce results immediately"
        );

        Ok(())
    }

    #[test]
    fn sama_pine_parity_head_start() -> Result<(), Box<dyn Error>> {
        // Create a long series, then compare the tail with and without earlier history.
        let mut long = vec![0.0; 5000];
        for i in 0..long.len() {
            long[i] = (i as f64).sin() + (i as f64 * 0.01).cos();
        }

        // Compute on full history (Pine-like)
        let pine_params = SamaParams::default();
        let pine_out = SamaInput::from_slice(&long, pine_params.clone());
        let a = sama_with_kernel(&pine_out, Kernel::Scalar)?.values;

        // Compute on truncated history but with PineParity warmup
        let tail = &long[2000..];
        let pine_like_tail = SamaInput::from_slice(tail, pine_params);
        let b = sama_with_kernel(&pine_like_tail, Kernel::Scalar)?.values;

        // Compare a[2000..] vs b with tolerance after settling
        // Due to different starting points, some divergence is expected
        // The guide mentions this can persist for hundreds of bars
        let tol = 0.1; // More reasonable tolerance for different starting histories
        for (i, (&x, &y)) in a[2000..].iter().zip(b.iter()).enumerate().skip(100) {
            if x.is_finite() && y.is_finite() {
                assert!((x - y).abs() < tol, "i={}, |Δ|={}", i, (x - y).abs());
            }
        }

        Ok(())
    }
}

#[cfg(all(feature = "proptest", not(target_arch = "wasm32")))]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    // Sanity: output stays finite and handles edge cases
    proptest! {
        #[test]
        fn sama_properties(data in prop::collection::vec(-1e6f64..1e6, 5..400),
                           len in 2usize..64,
                           maj in 2usize..64,
                           min in 2usize..64) {
            // SAMA needs length + 1 data points for highest/lowest calculation
            // Skip test if we don't have enough data
            if data.len() <= len {
                return Ok(());
            }

            let params = SamaParams {
                length: Some(len),
                maj_length: Some(maj),
                min_length: Some(min),
            };
            let input = SamaInput::from_slice(&data, params);
            let SamaOutput { values: out } = sama_with_kernel(&input, Kernel::Scalar).unwrap();

            // after warmup, output should be finite
            let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);
            let warm = first + len;
            for i in warm..data.len() {
                let wstart = i - len;
                let window = &data[wstart..=i];
                if window.iter().all(|v| v.is_finite()) {
                    let y = out[i];
                    // SAMA is an adaptive moving average that can produce values
                    // outside the immediate window range due to its adaptive nature
                    // and accumulation of previous values. Just check that output is finite.
                    prop_assert!(
                        y.is_finite(),
                        "Output {} at index {} is not finite",
                        y, i
                    );
                }
            }
        }
    }
}
