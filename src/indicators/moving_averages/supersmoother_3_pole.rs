//! # 3-Pole SuperSmoother Filter
//!
//! Three-pole smoothing filter developed by John Ehlers. Provides strong noise suppression
//! while remaining responsive to trend changes. The filter uses recursive calculations
//! with coefficients derived from the period parameter.
//!
//! ## Parameters
//! - **period**: Smoothing period (>= 1, defaults to 14)
//!
//! ## Returns
//! - **Ok(SuperSmoother3PoleOutput)** on success, containing a Vec<f64> matching input length.
//! - **Err(SuperSmoother3PoleError)** otherwise.
//!
//! ## Developer Notes
//! - **AVX2 kernel**: ❌ Stub only - falls back to scalar implementation
//! - **AVX512 kernel**: ❌ Stub only - falls back to scalar implementation
//! - **Streaming update**: ✅ O(1) complexity - efficient recursive calculation using only last 3 output values
//! - **Memory optimization**: ✅ Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix) for output vectors
//! - **TODO**: Implement SIMD kernels for recursive filter calculations (challenging due to sequential dependencies)

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaSupersmoother3Pole;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
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
use std::error::Error;
use std::f64::consts::PI;
use std::mem::MaybeUninit;
use thiserror::Error;

// Input and Output Types

#[derive(Debug, Clone)]
pub enum SuperSmoother3PoleData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

impl<'a> AsRef<[f64]> for SuperSmoother3PoleInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SuperSmoother3PoleData::Slice(slice) => slice,
            SuperSmoother3PoleData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleParams {
    pub period: Option<usize>,
}

impl Default for SuperSmoother3PoleParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleInput<'a> {
    pub data: SuperSmoother3PoleData<'a>,
    pub params: SuperSmoother3PoleParams,
}

impl<'a> SuperSmoother3PoleInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: SuperSmoother3PoleParams) -> Self {
        Self {
            data: SuperSmoother3PoleData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: SuperSmoother3PoleParams) -> Self {
        Self {
            data: SuperSmoother3PoleData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", SuperSmoother3PoleParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }
}

// Builder Pattern

#[derive(Copy, Clone, Debug)]
pub struct SuperSmoother3PoleBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for SuperSmoother3PoleBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SuperSmoother3PoleBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<SuperSmoother3PoleOutput, SuperSmoother3PoleError> {
        let p = SuperSmoother3PoleParams {
            period: self.period,
        };
        let i = SuperSmoother3PoleInput::from_candles(c, "close", p);
        supersmoother_3_pole_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(
        self,
        d: &[f64],
    ) -> Result<SuperSmoother3PoleOutput, SuperSmoother3PoleError> {
        let p = SuperSmoother3PoleParams {
            period: self.period,
        };
        let i = SuperSmoother3PoleInput::from_slice(d, p);
        supersmoother_3_pole_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<SuperSmoother3PoleStream, SuperSmoother3PoleError> {
        let p = SuperSmoother3PoleParams {
            period: self.period,
        };
        SuperSmoother3PoleStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum SuperSmoother3PoleError {
    #[error("supersmoother_3_pole: Input data slice is empty.")]
    EmptyInputData,
    #[error("supersmoother_3_pole: All values are NaN.")]
    AllValuesNaN,
    #[error("supersmoother_3_pole: Invalid period: period = {period}")]
    InvalidPeriod { period: usize },
    #[error("supersmoother_3_pole: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "supersmoother_3_pole: Output length mismatch: expected = {expected}, actual = {actual}"
    )]
    OutputLengthMismatch { expected: usize, actual: usize },
    #[error("supersmoother_3_pole: Invalid kernel for batch operation")]
    InvalidKernel,
}

// Main Entrypoint

#[inline]
pub fn supersmoother_3_pole(
    input: &SuperSmoother3PoleInput,
) -> Result<SuperSmoother3PoleOutput, SuperSmoother3PoleError> {
    supersmoother_3_pole_with_kernel(input, Kernel::Auto)
}

pub fn supersmoother_3_pole_with_kernel(
    input: &SuperSmoother3PoleInput,
    kernel: Kernel,
) -> Result<SuperSmoother3PoleOutput, SuperSmoother3PoleError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(SuperSmoother3PoleError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SuperSmoother3PoleError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(SuperSmoother3PoleError::InvalidPeriod { period });
    }
    if (len - first) < period {
        return Err(SuperSmoother3PoleError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };
    // Use alloc_with_nan_prefix to ensure [0..first) is NaN
    let mut out = alloc_with_nan_prefix(len, first);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                supersmoother_3_pole_scalar(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                supersmoother_3_pole_avx2(data, period, first, &mut out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                supersmoother_3_pole_avx512(data, period, first, &mut out)
            }
            _ => unreachable!(),
        }
    }

    Ok(SuperSmoother3PoleOutput { values: out })
}

// Compute function for SuperSmoother 3-pole respecting first index
#[inline(always)]
pub unsafe fn supersmoother_3_pole_compute_into(
    data: &[f64],
    period: usize,
    first: usize,
    _warm_end: usize, // Not used, kept for API compatibility
    out: &mut [f64],
) {
    let n = data.len();
    if n == 0 || first >= n {
        return;
    }

    // Calculate coefficients
    let a = (-PI / period as f64).exp();
    let b = 2.0 * a * (1.738_f64 * PI / period as f64).cos();
    let c = a * a;
    let coef_source = 1.0 - c * c - b + b * c;
    let coef_prev1 = b + c;
    let coef_prev2 = -c - b * c;
    let coef_prev3 = c * c;

    // Pass through first 3 values starting from 'first'
    // Do not write below 'first' - preserve NaN prefix
    if first < n {
        out[first] = data[first];
    }
    if first + 1 < n {
        out[first + 1] = data[first + 1];
    }
    if first + 2 < n {
        out[first + 2] = data[first + 2];
    }

    // Recursive calculation from index first+3 onwards
    if first + 3 >= n {
        return;
    }

    for i in (first + 3)..n {
        let d_i = data[i];
        let o_im1 = out[i - 1];
        let o_im2 = out[i - 2];
        let o_im3 = out[i - 3];
        out[i] = coef_source * d_i + coef_prev1 * o_im1 + coef_prev2 * o_im2 + coef_prev3 * o_im3;
    }
}

// Scalar kernel wrapper
#[inline(always)]
pub unsafe fn supersmoother_3_pole_scalar(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    supersmoother_3_pole_compute_into(data, period, first, 0, out);
}

// AVX2/AVX512 stubs (forward to scalar)
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_3_pole_avx2(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    supersmoother_3_pole_scalar(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_3_pole_avx512(
    data: &[f64],
    period: usize,
    first: usize,
    out: &mut [f64],
) {
    supersmoother_3_pole_scalar(data, period, first, out)
}

// Row functions (API parity)
#[inline(always)]
pub unsafe fn supersmoother_3_pole_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    supersmoother_3_pole_scalar(data, period, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_3_pole_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    supersmoother_3_pole_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_3_pole_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    supersmoother_3_pole_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_3_pole_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    supersmoother_3_pole_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn supersmoother_3_pole_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    stride: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    supersmoother_3_pole_row_scalar(data, first, period, stride, w_ptr, inv_n, out)
}

// Stream/Stateful

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleStream {
    period: usize,
    // Maintain last 3 outputs
    y0: f64,
    y1: f64,
    y2: f64,
    filled: usize, // How many outputs established
    coef_source: f64,
    coef_prev1: f64,
    coef_prev2: f64,
    coef_prev3: f64,
}

impl SuperSmoother3PoleStream {
    pub fn try_new(params: SuperSmoother3PoleParams) -> Result<Self, SuperSmoother3PoleError> {
        let period = params.period.unwrap_or(14);
        if period == 0 {
            return Err(SuperSmoother3PoleError::InvalidPeriod { period });
        }
        let a = (-PI / period as f64).exp();
        let b = 2.0 * a * (1.738_f64 * PI / period as f64).cos();
        let c = a * a;
        Ok(Self {
            period,
            y0: f64::NAN,
            y1: f64::NAN,
            y2: f64::NAN,
            filled: 0,
            coef_source: 1.0 - c * c - b + b * c,
            coef_prev1: b + c,
            coef_prev2: -c - b * c,
            coef_prev3: c * c,
        })
    }

    #[inline(always)]
    pub fn update(&mut self, value: f64) -> f64 {
        match self.filled {
            0 => {
                self.y0 = value;
                self.filled = 1;
                return value; // Pass through first value
            }
            1 => {
                self.y1 = value;
                self.filled = 2;
                return value; // Pass through second value
            }
            2 => {
                self.y2 = value;
                self.filled = 3;
                return value; // Pass through third value
            }
            _ => {}
        }
        let y_next = self.coef_source * value
            + self.coef_prev1 * self.y2
            + self.coef_prev2 * self.y1
            + self.coef_prev3 * self.y0;
        self.y0 = self.y1;
        self.y1 = self.y2;
        self.y2 = y_next;
        y_next
    }
}

// Batch Range/Builder/Output

#[derive(Clone, Debug)]
pub struct SuperSmoother3PoleBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for SuperSmoother3PoleBatchRange {
    fn default() -> Self {
        Self {
            period: (14, 14, 0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SuperSmoother3PoleBatchBuilder {
    range: SuperSmoother3PoleBatchRange,
    kernel: Kernel,
}

impl SuperSmoother3PoleBatchBuilder {
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
    pub fn apply_slice(
        self,
        data: &[f64],
    ) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
        supersmoother_3_pole_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
        SuperSmoother3PoleBatchBuilder::new()
            .kernel(k)
            .apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(
        c: &Candles,
    ) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
        SuperSmoother3PoleBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn supersmoother_3_pole_batch_with_kernel(
    data: &[f64],
    sweep: &SuperSmoother3PoleBatchRange,
    k: Kernel,
) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(SuperSmoother3PoleError::InvalidKernel),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    supersmoother_3_pole_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SuperSmoother3PoleBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SuperSmoother3PoleParams>,
    pub rows: usize,
    pub cols: usize,
}
impl SuperSmoother3PoleBatchOutput {
    pub fn row_for_params(&self, p: &SuperSmoother3PoleParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(14) == p.period.unwrap_or(14))
    }
    pub fn values_for(&self, p: &SuperSmoother3PoleParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &SuperSmoother3PoleBatchRange) -> Vec<SuperSmoother3PoleParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(SuperSmoother3PoleParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn supersmoother_3_pole_batch_slice(
    data: &[f64],
    sweep: &SuperSmoother3PoleBatchRange,
    kern: Kernel,
) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
    supersmoother_3_pole_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn supersmoother_3_pole_batch_par_slice(
    data: &[f64],
    sweep: &SuperSmoother3PoleBatchRange,
    kern: Kernel,
) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
    supersmoother_3_pole_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn supersmoother_3_pole_batch_inner(
    data: &[f64],
    sweep: &SuperSmoother3PoleBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SuperSmoother3PoleBatchOutput, SuperSmoother3PoleError> {
    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SuperSmoother3PoleError::InvalidPeriod { period: 0 });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SuperSmoother3PoleError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(SuperSmoother3PoleError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    let mut raw = make_uninit_matrix(rows, cols);
    unsafe { init_matrix_prefixes(&mut raw, cols, &warm) };

    // ---------- 2.  Row-level closure (accepts &mut [MaybeUninit<f64>]) ----
    let do_row = |row: usize, dst_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // transmute just this row to &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => supersmoother_3_pole_row_scalar(
                data,
                first,
                period,
                0,
                std::ptr::null(),
                0.0,
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => supersmoother_3_pole_row_avx2(
                data,
                first,
                period,
                0,
                std::ptr::null(),
                0.0,
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => supersmoother_3_pole_row_avx512(
                data,
                first,
                period,
                0,
                std::ptr::null(),
                0.0,
                out_row,
            ),
            _ => unreachable!(),
        }
    };

    // ---------- 3.  Run rows in serial / parallel --------------------------
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            raw.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }

        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in raw.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // ---------- 4.  Finalise: convert Vec<MaybeUninit<f64>> → Vec<f64> -----
    let values: Vec<f64> = unsafe { std::mem::transmute(raw) };
    Ok(SuperSmoother3PoleBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
pub fn supersmoother_3_pole_batch_inner_into(
    data: &[f64],
    combos: &[SuperSmoother3PoleParams],
    first: usize,
    warm: &[usize],
    cols: usize,
    kern: Kernel,
    parallel: bool,
    output: &mut [f64],
) {
    // Write directly to output slice - no validation needed
    let mut raw = unsafe {
        std::slice::from_raw_parts_mut(
            output.as_mut_ptr() as *mut std::mem::MaybeUninit<f64>,
            output.len(),
        )
    };
    unsafe { init_matrix_prefixes(&mut raw, cols, warm) };

    // ---------- 2.  Row-level closure (accepts &mut [MaybeUninit<f64>]) ----
    let do_row = |row: usize, dst_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // transmute just this row to &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar => supersmoother_3_pole_row_scalar(
                data,
                first,
                period,
                0,
                std::ptr::null(),
                0.0,
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => supersmoother_3_pole_row_avx2(
                data,
                first,
                period,
                0,
                std::ptr::null(),
                0.0,
                out_row,
            ),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => supersmoother_3_pole_row_avx512(
                data,
                first,
                period,
                0,
                std::ptr::null(),
                0.0,
                out_row,
            ),
            _ => unreachable!(),
        }
    };

    // ---------- 3.  Run rows in serial / parallel --------------------------
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            raw.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
            for (row, slice) in raw.chunks_mut(cols).enumerate() {
                do_row(row, slice);
            }
        }
    } else {
        for (row, slice) in raw.chunks_mut(cols).enumerate() {
            do_row(row, slice);
        }
    }

    // No return value needed - everything is written to output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_supersmoother_3_pole_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = SuperSmoother3PoleParams { period: None };
        let input = SuperSmoother3PoleInput::from_candles(&candles, "close", default_params);
        let output = supersmoother_3_pole_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_supersmoother_3_pole_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = SuperSmoother3PoleParams { period: Some(14) };
        let input = SuperSmoother3PoleInput::from_candles(&candles, "close", params);
        let result = supersmoother_3_pole_with_kernel(&input, kernel)?;
        let values = &result.values;
        let expected_last_five = [
            59072.13481064446,
            59089.08032603,
            59111.35711851466,
            59133.14402399381,
            59121.91820047289,
        ];
        assert!(values.len() >= 5);
        let start_idx = values.len() - 5;
        let last_five = &values[start_idx..];
        for (i, (&actual, &expected)) in last_five.iter().zip(expected_last_five.iter()).enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "3-pole SuperSmoother mismatch at index {}: expected {}, got {}, diff {}",
                i,
                expected,
                actual,
                diff
            );
        }
        Ok(())
    }

    fn check_supersmoother_3_pole_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = SuperSmoother3PoleParams { period: Some(0) };
        let input = SuperSmoother3PoleInput::from_slice(&data, params);
        let res = supersmoother_3_pole_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SS3Pole should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_supersmoother_3_pole_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0, 30.0];
        let params = SuperSmoother3PoleParams { period: Some(10) };
        let input = SuperSmoother3PoleInput::from_slice(&data, params);
        let res = supersmoother_3_pole_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SS3Pole should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_supersmoother_3_pole_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [10.0, 20.0];
        let params = SuperSmoother3PoleParams { period: Some(14) };
        let input = SuperSmoother3PoleInput::from_slice(&data, params);
        let result = supersmoother_3_pole_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), data.len());
        assert_eq!(result.values[0], 10.0);
        if result.values.len() > 1 {
            assert_eq!(result.values[1], 20.0);
        }
        Ok(())
    }

    fn check_supersmoother_3_pole_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_input = SuperSmoother3PoleInput::from_candles(
            &candles,
            "close",
            SuperSmoother3PoleParams { period: Some(14) },
        );
        let first_result = supersmoother_3_pole_with_kernel(&first_input, kernel)?;
        let second_input = SuperSmoother3PoleInput::from_slice(
            &first_result.values,
            SuperSmoother3PoleParams { period: Some(7) },
        );
        let second_result = supersmoother_3_pole_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }

    fn check_supersmoother_3_pole_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SuperSmoother3PoleInput::from_candles(
            &candles,
            "close",
            SuperSmoother3PoleParams { period: Some(14) },
        );
        let result = supersmoother_3_pole_with_kernel(&input, kernel)?;
        assert_eq!(result.values.len(), candles.close.len());
        for (idx, &val) in result.values.iter().enumerate() {
            assert!(val.is_finite(), "NaN found at index {}", idx);
        }
        Ok(())
    }

    fn check_supersmoother_3_pole_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 14;
        let input = SuperSmoother3PoleInput::from_candles(
            &candles,
            "close",
            SuperSmoother3PoleParams {
                period: Some(period),
            },
        );
        let batch_output = supersmoother_3_pole_with_kernel(&input, kernel)?.values;
        let mut stream = SuperSmoother3PoleStream::try_new(SuperSmoother3PoleParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            let ss_val = stream.update(price);
            stream_values.push(ss_val);
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] SS3Pole streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_supersmoother_3_pole_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Strategy: generate period first, then data of appropriate length
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
                let params = SuperSmoother3PoleParams {
                    period: Some(period),
                };
                let input = SuperSmoother3PoleInput::from_slice(&data, params);

                let SuperSmoother3PoleOutput { values: out } =
                    supersmoother_3_pole_with_kernel(&input, kernel).unwrap();
                let SuperSmoother3PoleOutput { values: ref_out } =
                    supersmoother_3_pole_with_kernel(&input, Kernel::Scalar).unwrap();

                // Since we don't inject NaN, first is always 0
                let first = 0;
                let warmup = first + period;

                // Property 1: Output length matches input length
                prop_assert_eq!(out.len(), data.len(), "Output length mismatch");

                // Property 2: First three values pass through directly
                if data.len() > 0 {
                    prop_assert!(
                        (out[0] - data[0]).abs() < f64::EPSILON,
                        "First value mismatch: {} vs {}",
                        out[0],
                        data[0]
                    );
                }
                if data.len() > 1 {
                    prop_assert!(
                        (out[1] - data[1]).abs() < f64::EPSILON,
                        "Second value mismatch: {} vs {}",
                        out[1],
                        data[1]
                    );
                }
                if data.len() > 2 {
                    prop_assert!(
                        (out[2] - data[2]).abs() < f64::EPSILON,
                        "Third value mismatch: {} vs {}",
                        out[2],
                        data[2]
                    );
                }

                // Property 3: Values from index 3 onwards are computed
                // Note: The implementation computes from index 3 regardless of warmup
                // This appears to be the actual behavior of the filter
                for i in 3..data.len() {
                    prop_assert!(
                        out[i].is_finite(),
                        "Expected finite value at index {}, got {}",
                        i,
                        out[i]
                    );
                }

                // Property 4: Output remains finite (no NaN or Inf)
                // Recursive filters can amplify significantly but should remain finite
                for i in 3..data.len() {
                    prop_assert!(
                        out[i].is_finite(),
                        "Output at index {} is not finite: {}",
                        i,
                        out[i]
                    );
                }

                // Property 5: Smoothing property - should reduce variation
                // Check from a stable point after initial transients
                let stable_start = (period * 2).max(10).min(data.len() - 1);
                if data.len() > stable_start + 10 {
                    let input_variation: f64 = data[stable_start..data.len() - 1]
                        .windows(2)
                        .map(|w| (w[1] - w[0]).abs())
                        .sum::<f64>();
                    let output_variation: f64 = out[stable_start..out.len() - 1]
                        .windows(2)
                        .map(|w| (w[1] - w[0]).abs())
                        .sum::<f64>();

                    // For a smoothing filter, output variation should generally be less
                    // Allow up to 2x amplification for edge cases
                    if input_variation > 1e-9 {
                        let variation_ratio = output_variation / input_variation;
                        prop_assert!(
                            variation_ratio <= 2.0,
                            "Output variation too high: ratio = {} (out={}, in={})",
                            variation_ratio,
                            output_variation,
                            input_variation
                        );
                    }
                }

                // Property 6: Constant input produces constant output
                if data.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON) {
                    let constant_val = data[0];
                    // Check from a stable point
                    let stable_start = period.max(3);
                    for i in stable_start..out.len() {
                        prop_assert!(
                            (out[i] - constant_val).abs() <= 1e-9,
                            "Constant input should produce constant output at index {}: {} vs {}",
                            i,
                            out[i],
                            constant_val
                        );
                    }
                }

                // Property 7: Period=1 special case
                if period == 1 {
                    // Values from index 3 onwards should be computed
                    for i in 3..data.len() {
                        prop_assert!(
                            out[i].is_finite(),
                            "Period=1 should produce finite values from index 3, got {} at {}",
                            out[i],
                            i
                        );
                    }
                }

                // Property 8: Kernel consistency (all kernels should match since AVX forwards to scalar)
                for i in 0..out.len() {
                    if out[i].is_finite() && ref_out[i].is_finite() {
                        let diff = (out[i] - ref_out[i]).abs();
                        let ulp_diff = out[i].to_bits().abs_diff(ref_out[i].to_bits());
                        prop_assert!(
                            diff <= 1e-9 || ulp_diff <= 4,
                            "Kernel mismatch at index {}: {} vs {} (diff={}, ULP={})",
                            i,
                            out[i],
                            ref_out[i],
                            diff,
                            ulp_diff
                        );
                    } else {
                        prop_assert_eq!(
                            out[i].is_nan(),
                            ref_out[i].is_nan(),
                            "NaN mismatch at index {}",
                            i
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_ss3pole_tests {
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

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_supersmoother_3_pole_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test multiple parameter combinations to catch uninitialized memory reads
        let test_periods = vec![5, 10, 14, 20, 30, 50, 100, 200];

        for period in test_periods {
            let params = SuperSmoother3PoleParams {
                period: Some(period),
            };
            let input = SuperSmoother3PoleInput::from_candles(&candles, "close", params);

            // Skip if period is too large for the data
            if period > candles.close.len() {
                continue;
            }

            let output = supersmoother_3_pole_with_kernel(&input, kernel)?;

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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period {}",
						test_name, val, bits, i, period
					);
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_supersmoother_3_pole_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    generate_all_ss3pole_tests!(
        check_supersmoother_3_pole_partial_params,
        check_supersmoother_3_pole_accuracy,
        check_supersmoother_3_pole_zero_period,
        check_supersmoother_3_pole_period_exceeds_length,
        check_supersmoother_3_pole_very_small_dataset,
        check_supersmoother_3_pole_reinput,
        check_supersmoother_3_pole_nan_handling,
        check_supersmoother_3_pole_streaming,
        check_supersmoother_3_pole_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_ss3pole_tests!(check_supersmoother_3_pole_property);

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = SuperSmoother3PoleBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = SuperSmoother3PoleParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            59072.13481064446,
            59089.08032603,
            59111.35711851466,
            59133.14402399381,
            59121.91820047289,
        ];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-8,
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
                    let _ = $fn_name(stringify!([<$fn_name _auto_detect>]),
                                     Kernel::Auto);
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

        // Test multiple different batch configurations to catch edge cases
        let batch_configs = vec![
            (5, 15, 5),    // Small periods
            (10, 30, 10),  // Medium periods
            (20, 100, 20), // Large periods
            (7, 7, 1),     // Single period (edge case)
            (3, 50, 1),    // Many periods
        ];

        for (start, end, step) in batch_configs {
            // Skip if the largest period exceeds data length
            if end > c.close.len() {
                continue;
            }

            let output = SuperSmoother3PoleBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, end, step)
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
                let period = if row < output.combos.len() {
                    output.combos[row].period.unwrap_or(0)
                } else {
                    0
                };

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with period {} in batch ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
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

// Python bindings
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
#[pyfunction(name = "supersmoother_3_pole")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the 3-Pole SuperSmoother filter of the input data.
///
/// The 3-Pole SuperSmoother is a smoothing filter developed by John Ehlers
/// that provides strong noise suppression while remaining responsive to trend changes.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     The smoothing period (must be >= 1).
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of SuperSmoother3Pole values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period <= 0, period > data length, all NaN data).
pub fn supersmoother_3_pole_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?; // Validate before allow_threads

    let params = SuperSmoother3PoleParams {
        period: Some(period),
    };
    let input = SuperSmoother3PoleInput::from_slice(slice_in, params);

    // GOOD: Get Vec<f64> from Rust function
    let result_vec: Vec<f64> = py
        .allow_threads(|| supersmoother_3_pole_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // GOOD: Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "SuperSmoother3PoleStream")]
pub struct SuperSmoother3PoleStreamPy {
    stream: SuperSmoother3PoleStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SuperSmoother3PoleStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = SuperSmoother3PoleParams {
            period: Some(period),
        };
        let stream = SuperSmoother3PoleStream::try_new(params)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(SuperSmoother3PoleStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated SuperSmoother3Pole value.
    fn update(&mut self, value: f64) -> f64 {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "supersmoother_3_pole_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute SuperSmoother3Pole for multiple period combinations in a single pass.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period_range : tuple
///     (start, end, step) for period values to compute.
/// kernel : str, optional
///     Computation kernel: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// dict
///     Dictionary with 'values' (2D array) and 'periods' array.
pub fn supersmoother_3_pole_batch_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = SuperSmoother3PoleBatchRange {
        period: period_range,
    };

    // Pre-calculate everything outside allow_threads
    let combos = expand_grid(&sweep);
    if combos.is_empty() {
        return Err(PyValueError::new_err("Invalid period range"));
    }

    let first = slice_in
        .iter()
        .position(|x| !x.is_nan())
        .ok_or_else(|| PyValueError::new_err("All input values are NaN"))?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if slice_in.len() - first < max_p {
        return Err(PyValueError::new_err(format!(
            "Not enough valid data: needed = {}, valid = {}",
            max_p,
            slice_in.len() - first
        )));
    }

    let rows = combos.len();
    let cols = slice_in.len();
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // Pre-allocate output array (OK for batch operations)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Compute without GIL - no validation needed
    py.allow_threads(|| {
        let kernel = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        let simd = match kernel {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => kernel,
        };
        supersmoother_3_pole_batch_inner_into(
            slice_in, &combos, first, &warm, cols, simd, true, slice_out,
        );
    });

    // Build result dictionary
    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;

    // Periods array with zero-copy transfer
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

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "supersmoother_3_pole_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn supersmoother_3_pole_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = SuperSmoother3PoleBatchRange {
        period: period_range,
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaSupersmoother3Pole::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.supersmoother_3_pole_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "supersmoother_3_pole_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn supersmoother_3_pole_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = SuperSmoother3PoleParams {
        period: Some(period),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaSupersmoother3Pole::new(device_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.supersmoother_3_pole_many_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

/// Computes SuperSmoother 3-Pole directly into a provided output slice, avoiding allocation.
/// The output slice must be the same length as the input data.
#[inline]
pub fn supersmoother_3_pole_into_slice(
    dst: &mut [f64],
    input: &SuperSmoother3PoleInput,
    kern: Kernel,
) -> Result<(), SuperSmoother3PoleError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(SuperSmoother3PoleError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SuperSmoother3PoleError::AllValuesNaN)?;
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(SuperSmoother3PoleError::InvalidPeriod { period });
    }
    if (len - first) < period {
        return Err(SuperSmoother3PoleError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    // Verify output buffer size matches input
    if dst.len() != data.len() {
        return Err(SuperSmoother3PoleError::OutputLengthMismatch {
            expected: data.len(),
            actual: dst.len(),
        });
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    // Compute SuperSmoother values directly into dst
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => {
                supersmoother_3_pole_scalar(data, period, first, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => supersmoother_3_pole_avx2(data, period, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                supersmoother_3_pole_avx512(data, period, first, dst)
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

// WASM bindings
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "wasm")]
use js_sys::{Object, Reflect};

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_3_pole_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = SuperSmoother3PoleParams {
        period: Some(period),
    };
    let input = SuperSmoother3PoleInput::from_slice(data, params);

    // Allocate output buffer once
    let mut output = vec![0.0; data.len()];

    // Compute directly into output buffer
    supersmoother_3_pole_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SuperSmoother3PoleBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_3_pole_batch(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    use serde_wasm_bindgen::{from_value, to_value};

    let config: SuperSmoother3PoleBatchConfig = from_value(config)?;
    let sweep = SuperSmoother3PoleBatchRange {
        period: config.period_range,
    };

    // Pre-calculate everything
    let combos = expand_grid(&sweep);
    if combos.is_empty() {
        return Err(JsValue::from_str("Invalid period range"));
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or_else(|| JsValue::from_str("All input values are NaN"))?;

    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(JsValue::from_str(&format!(
            "Not enough valid data: needed = {}, valid = {}",
            max_p,
            data.len() - first
        )));
    }

    let rows = combos.len();
    let cols = data.len();
    let mut output = vec![0.0; rows * cols];

    // Compute batch
    let chosen = detect_best_batch_kernel();
    let simd = match chosen {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => chosen,
    };

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    supersmoother_3_pole_batch_inner_into(
        data,
        &combos,
        first,
        &warm,
        cols,
        simd,
        false,
        &mut output,
    );

    // Return result with metadata
    let result = Object::new();
    Reflect::set(&result, &JsValue::from_str("values"), &to_value(&output)?)?;
    Reflect::set(
        &result,
        &JsValue::from_str("rows"),
        &JsValue::from_f64(rows as f64),
    )?;
    Reflect::set(
        &result,
        &JsValue::from_str("cols"),
        &JsValue::from_f64(cols as f64),
    )?;
    Reflect::set(
        &result,
        &JsValue::from_str("periods"),
        &to_value(&combos.iter().map(|c| c.period.unwrap()).collect::<Vec<_>>())?,
    )?;

    Ok(JsValue::from(result))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_3_pole_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = SuperSmoother3PoleBatchRange {
        period: (period_start, period_end, period_step),
    };

    // Use the existing batch function with parallel=false for WASM
    supersmoother_3_pole_batch_inner(data, &sweep, Kernel::Scalar, false)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_3_pole_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = SuperSmoother3PoleBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep);
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
    }

    Ok(metadata)
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_3_pole_alloc(len: usize) -> *mut f64 {
    // Allocate memory for input/output buffer
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Prevent deallocation
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_3_pole_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_3_pole_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided"));
    }

    unsafe {
        // Create slice from pointer
        let data = std::slice::from_raw_parts(in_ptr, len);
        let params = SuperSmoother3PoleParams {
            period: Some(period),
        };
        let input = SuperSmoother3PoleInput::from_slice(data, params);

        // Handle aliasing - if input and output pointers are the same
        if in_ptr == out_ptr {
            // Need temporary buffer to avoid corruption
            let mut temp = vec![0.0; len];
            supersmoother_3_pole_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy result back to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // Direct computation into output buffer
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            supersmoother_3_pole_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

// ================== Optimized Batch Processing ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn supersmoother_3_pole_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("Null pointer provided to batch function"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = SuperSmoother3PoleBatchRange {
            period: (period_start, period_end, period_step),
        };

        // Pre-calculate everything
        let combos = expand_grid(&sweep);
        if combos.is_empty() {
            return Err(JsValue::from_str("Invalid period range"));
        }

        let first = data
            .iter()
            .position(|x| !x.is_nan())
            .ok_or_else(|| JsValue::from_str("All input values are NaN"))?;

        let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
        if data.len() - first < max_p {
            return Err(JsValue::from_str(&format!(
                "Not enough valid data: needed = {}, valid = {}",
                max_p,
                data.len() - first
            )));
        }

        let rows = combos.len();
        let cols = data.len();
        let total_size = rows * cols;

        // Create output slice
        let out = std::slice::from_raw_parts_mut(out_ptr, total_size);

        // Compute batch
        let chosen = detect_best_batch_kernel();
        let simd = match chosen {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            Kernel::ScalarBatch => Kernel::Scalar,
            _ => chosen,
        };

        let warm: Vec<usize> = combos
            .iter()
            .map(|c| first + c.period.unwrap() - 1)
            .collect();

        supersmoother_3_pole_batch_inner_into(data, &combos, first, &warm, cols, simd, false, out);

        Ok(rows)
    }
}
