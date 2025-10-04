//! # Smoothed Moving Average (SMMA)
//!
//! The Smoothed Moving Average (SMMA) uses a recursive smoothing formula where the first value
//! is the mean of the first `period` points and subsequent values use a blend of the prior SMMA
//! and the new point. API matches ALMA in structure and extensibility, supporting streaming,
//! parameter sweeps, and SIMD feature stubs for AVX2/AVX512 kernels.
//!
//! ## Parameters
//! - **period**: Window size (number of data points).
//!
//! ## Returns
//! - **`Ok(SmmaOutput)`** on success, containing a `Vec<f64>` matching the input.
//! - **`Err(SmmaError)`** otherwise.
//!
//! ## Developer Status
//! - **AVX2 kernel**: Enabled (relaxed FMA). Uses FMA and reciprocal multiply for speed;
//!   rounding may differ very slightly vs scalar but stays within 1e-7 in tests.
//! - **AVX512 kernel**: Routes to AVX2 implementation.
//! - **Streaming update**: O(1) – Drop-in state machine matching batch semantics; no ring buffer
//! - **Memory optimization**: GOOD - Uses zero-copy helpers (alloc_with_nan_prefix, make_uninit_matrix)
//! - **SIMD note**: The SMMA recurrence is sequential; SIMD provides little gain without
//!   changing the algorithm and may change rounding. Stubs keep scalar for bit‑consistency.

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
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaSmma;
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
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for SmmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            SmmaData::Slice(slice) => slice,
            SmmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SmmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SmmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct SmmaParams {
    pub period: Option<usize>,
}

impl Default for SmmaParams {
    fn default() -> Self {
        Self { period: Some(7) }
    }
}

#[derive(Debug, Clone)]
pub struct SmmaInput<'a> {
    pub data: SmmaData<'a>,
    pub params: SmmaParams,
}

impl<'a> SmmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: SmmaParams) -> Self {
        Self {
            data: SmmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: SmmaParams) -> Self {
        Self {
            data: SmmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", SmmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(7)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SmmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for SmmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl SmmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<SmmaOutput, SmmaError> {
        let p = SmmaParams {
            period: self.period,
        };
        let i = SmmaInput::from_candles(c, "close", p);
        smma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<SmmaOutput, SmmaError> {
        let p = SmmaParams {
            period: self.period,
        };
        let i = SmmaInput::from_slice(d, p);
        smma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<SmmaStream, SmmaError> {
        let p = SmmaParams {
            period: self.period,
        };
        SmmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum SmmaError {
    #[error("smma: Input data slice is empty.")]
    EmptyInputData,
    #[error("smma: All values are NaN.")]
    AllValuesNaN,
    #[error("smma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("smma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("smma: Invalid kernel for batch operation: {kernel:?} (expected batch kernel)")]
    InvalidKernel { kernel: Kernel },
    #[error("smma: Output buffer length mismatch: expected = {expected}, actual = {actual}")]
    OutputLenMismatch { expected: usize, actual: usize },
}

#[inline]
pub fn smma(input: &SmmaInput) -> Result<SmmaOutput, SmmaError> {
    smma_with_kernel(input, Kernel::Auto)
}

pub fn smma_with_kernel(input: &SmmaInput, kernel: Kernel) -> Result<SmmaOutput, SmmaError> {
    let data: &[f64] = match &input.data {
        SmmaData::Candles { candles, source } => source_type(candles, source),
        SmmaData::Slice(sl) => sl,
    };

    // Check for empty input first
    if data.is_empty() {
        return Err(SmmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SmmaError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(SmmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(SmmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    let warm = first + period - 1;
    let mut out = alloc_with_nan_prefix(len, warm);

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => smma_scalar(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => smma_avx2(data, period, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => smma_avx512(data, period, first, &mut out),
            _ => smma_scalar(data, period, first, &mut out), // Default to scalar
        }
    }
    Ok(SmmaOutput { values: out })
}

#[inline]
pub fn smma_scalar(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let len = data.len();
    let end = first + period;

    // Fast path: period == 1 → output equals input from `first` onward
    if period == 1 {
        // Write element-wise to avoid bulk copies per repo conventions
        out[first] = data[first];
        let mut i = first + 1;
        while i < len {
            out[i] = data[i];
            i += 1;
        }
        return;
    }

    // Initial SMA over [first .. end)
    let mut sum = 0.0f64;
    for i in first..end {
        sum += data[i];
    }

    let pf64 = period as f64;
    let pm1 = pf64 - 1.0;

    // First valid SMMA value at index end-1
    let mut prev = sum / pf64;
    out[end - 1] = prev;

    // Recursive smoothing: keep exact mul + add + div order (avoid FMA/1/pf64 mul)
    for i in end..len {
        prev = (prev * pm1 + data[i]) / pf64;
        out[i] = prev;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn smma_simd128(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // Calculate initial average
    let end = first + period;
    let sum: f64 = data[first..end].iter().sum();
    let mut prev = sum / period as f64;
    out[end - 1] = prev;

    // SIMD constants
    let period_f64 = period as f64;
    let period_minus_1 = period_f64 - 1.0;
    let inv_period = 1.0 / period_f64;

    // Process scalar loop - SMMA is inherently sequential
    // SIMD doesn't provide much benefit due to the dependency chain
    // Each value depends on the previous one
    for i in end..data.len() {
        prev = (prev * period_minus_1 + data[i]) * inv_period;
        out[i] = prev;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn smma_avx512(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    unsafe { smma_avx512_long(data, period, first, out) }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn smma_avx2(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    let len = data.len();
    let end = first + period;

    if period == 1 {
        out[first] = data[first];
        let mut i = first + 1;
        while i < len {
            out[i] = data[i];
            i += 1;
        }
        return;
    }

    // Initial SMA – preserve left-to-right accumulation order
    let mut sum = 0.0f64;
    let mut i = first;
    while i < end {
        sum += data[i];
        i += 1;
    }

    let pf64 = period as f64;
    let pm1 = pf64 - 1.0;
    let inv_p = 1.0 / pf64; // reciprocal multiply in the hot loop

    let mut prev = sum * inv_p;
    out[end - 1] = prev;

    // Hot loop with fused multiply-add; relaxed rounding
    let mut t = end;
    while t + 4 <= len {
        let x0 = data[t];
        prev = f64::mul_add(prev, pm1, x0) * inv_p;
        out[t] = prev;

        let x1 = data[t + 1];
        prev = f64::mul_add(prev, pm1, x1) * inv_p;
        out[t + 1] = prev;

        let x2 = data[t + 2];
        prev = f64::mul_add(prev, pm1, x2) * inv_p;
        out[t + 2] = prev;

        let x3 = data[t + 3];
        prev = f64::mul_add(prev, pm1, x3) * inv_p;
        out[t + 3] = prev;

        t += 4;
    }
    while t < len {
        let x = data[t];
        prev = f64::mul_add(prev, pm1, x) * inv_p;
        out[t] = prev;
        t += 1;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn smma_avx512_short(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // Route AVX512 to AVX2 implementation (same semantics here)
    smma_avx2(data, period, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn smma_avx512_long(data: &[f64], period: usize, first: usize, out: &mut [f64]) {
    // Route AVX512 to AVX2 implementation (same semantics here)
    smma_avx2(data, period, first, out)
}

// smma_avx2_relaxed_fma logic is now inlined in smma_avx2

#[inline(always)]
fn smma_prepare<'a>(
    input: &'a SmmaInput,
    kernel: Kernel,
) -> Result<
    (
        // data
        &'a [f64],
        // period
        usize,
        // first
        usize,
        // chosen
        Kernel,
    ),
    SmmaError,
> {
    let data: &[f64] = input.as_ref();

    // Check for empty input first
    if data.is_empty() {
        return Err(SmmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SmmaError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > len {
        return Err(SmmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first) < period {
        return Err(SmmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    Ok((data, period, first, chosen))
}

#[inline(always)]
fn smma_compute_into(data: &[f64], period: usize, first: usize, kernel: Kernel, out: &mut [f64]) {
    unsafe {
        // For WASM, use SIMD128 when available instead of scalar
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                smma_simd128(data, period, first, out);
                return;
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => smma_scalar(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => smma_avx2(data, period, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => smma_avx512(data, period, first, out),
            _ => smma_scalar(data, period, first, out), // Default to scalar
        }
    }
}

#[inline]
pub fn expand_grid(r: &SmmaBatchRange) -> Vec<SmmaParams> {
    let axis_usize = |(start, end, step): (usize, usize, usize)| {
        if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        }
    };
    axis_usize(r.period)
        .into_iter()
        .map(|p| SmmaParams { period: Some(p) })
        .collect()
}

/// Decision: Streaming uses an O(1) state machine (no ring buffer) that
/// exactly matches batch warm-up and NaN handling.
#[derive(Debug, Clone)]
pub struct SmmaStream {
    period: usize,
    // Cached constants
    pf64: f64,
    pm1: f64,
    inv_p: f64,
    // State machine mirrors batch semantics
    state: SmmaStreamState,
}

#[derive(Debug, Clone)]
enum SmmaStreamState {
    SeekingFirst,
    Warming { sum: f64, count: usize },
    Ready { value: f64 },
}

impl SmmaStream {
    #[inline]
    pub fn try_new(params: SmmaParams) -> Result<Self, SmmaError> {
        let period = params.period.unwrap_or(7);
        if period == 0 {
            return Err(SmmaError::InvalidPeriod { period, data_len: 0 });
        }
        let pf64 = period as f64;
        Ok(Self {
            period,
            pf64,
            pm1: pf64 - 1.0,
            inv_p: 1.0 / pf64,
            state: SmmaStreamState::SeekingFirst,
        })
    }

    /// O(1) streaming update. Returns None until the first `period` finite
    /// values after the first finite have been seen; Some(smma) thereafter.
    #[inline(always)]
    pub fn update(&mut self, v: f64) -> Option<f64> {
        use SmmaStreamState::*;
        match &mut self.state {
            SeekingFirst => {
                if v.is_finite() {
                    if self.period == 1 {
                        self.state = Ready { value: v };
                        return Some(v);
                    }
                    self.state = Warming { sum: v, count: 1 };
                }
                None
            }
            Warming { sum, count } => {
                // Accumulate the first `period` samples starting at first finite
                *sum += v;
                *count += 1;
                if *count == self.period {
                    let first_val = *sum / self.pf64;
                    self.state = Ready { value: first_val };
                    Some(first_val)
                } else {
                    None
                }
            }
            Ready { value } => {
                let next = (*value * self.pm1 + v) / self.pf64;
                *value = next;
                Some(next)
            }
        }
    }

    #[inline(always)]
    pub fn is_ready(&self) -> bool {
        matches!(self.state, SmmaStreamState::Ready { .. })
    }
    #[inline(always)]
    pub fn current(&self) -> Option<f64> {
        match self.state {
            SmmaStreamState::Ready { value } => Some(value),
            _ => None,
        }
    }
    #[inline]
    pub fn reset(&mut self) {
        self.state = SmmaStreamState::SeekingFirst;
    }
}

#[derive(Clone, Debug)]
pub struct SmmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for SmmaBatchRange {
    fn default() -> Self {
        Self {
            period: (7, 100, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SmmaBatchBuilder {
    range: SmmaBatchRange,
    kernel: Kernel,
}

impl SmmaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<SmmaBatchOutput, SmmaError> {
        smma_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<SmmaBatchOutput, SmmaError> {
        SmmaBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<SmmaBatchOutput, SmmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<SmmaBatchOutput, SmmaError> {
        SmmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn smma_batch_with_kernel(
    data: &[f64],
    sweep: &SmmaBatchRange,
    k: Kernel,
) -> Result<SmmaBatchOutput, SmmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        other => return Err(SmmaError::InvalidKernel { kernel: other }),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => Kernel::Scalar, // Default to scalar for any other case
    };
    smma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct SmmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SmmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl SmmaBatchOutput {
    pub fn row_for_params(&self, p: &SmmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(7) == p.period.unwrap_or(7))
    }
    pub fn values_for(&self, p: &SmmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
pub fn smma_batch_slice(
    data: &[f64],
    sweep: &SmmaBatchRange,
    kern: Kernel,
) -> Result<SmmaBatchOutput, SmmaError> {
    smma_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn smma_batch_par_slice(
    data: &[f64],
    sweep: &SmmaBatchRange,
    kern: Kernel,
) -> Result<SmmaBatchOutput, SmmaError> {
    smma_batch_inner(data, sweep, kern, true)
}
#[inline(always)]
fn smma_batch_inner(
    data: &[f64],
    sweep: &SmmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<SmmaBatchOutput, SmmaError> {
    // Check for empty input first
    if data.is_empty() {
        return Err(SmmaError::EmptyInputData);
    }

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SmmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SmmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(SmmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }
    let rows = combos.len();
    let cols = data.len();

    // ------------------------------------------------------------------
    // 1.  Figure out how long each row’s NaN prefix should be
    // ------------------------------------------------------------------
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // ------------------------------------------------------------------
    // 2.  Allocate rows × cols uninitialised and write the NaN prefixes
    // ------------------------------------------------------------------
    let mut raw = make_uninit_matrix(rows, cols);
    unsafe {
        init_matrix_prefixes(&mut raw, cols, &warm);
    }

    // ------------------------------------------------------------------
    // 3.  Closure that computes ONE row in-place
    //     - receives the row as &mut [MaybeUninit<f64>]
    //     - casts it to &mut [f64] after the prefix
    // ------------------------------------------------------------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast this row to plain f64s
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => smma_row_scalar(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => smma_row_avx2(data, first, period, out_row),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => smma_row_avx512(data, first, period, out_row),
            _ => smma_row_scalar(data, first, period, out_row), // Default to scalar
        }
    };

    // ------------------------------------------------------------------
    // 4.  Run every row (optionally in parallel) straight into `raw`
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // 5.  All elements are now initialised – materialize Vec<f64> without copies
    // ------------------------------------------------------------------
    let mut guard = core::mem::ManuallyDrop::new(raw);
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(SmmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn smma_batch_inner_into(
    data: &[f64],
    sweep: &SmmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<SmmaParams>, SmmaError> {
    // Check for empty input first
    if data.is_empty() {
        return Err(SmmaError::EmptyInputData);
    }

    let combos = expand_grid(sweep);
    if combos.is_empty() {
        return Err(SmmaError::InvalidPeriod {
            period: 0,
            data_len: 0,
        });
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(SmmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(SmmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // Collect warm-up lengths per row
    let warm: Vec<usize> = combos
        .iter()
        .map(|c| first + c.period.unwrap() - 1)
        .collect();

    // SAFETY: We're reinterpreting the output slice as MaybeUninit to use the efficient
    // init_matrix_prefixes function. This is safe because:
    // 1. MaybeUninit<T> has the same layout as T
    // 2. We ensure all values are written before the slice is used again
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    unsafe { init_matrix_prefixes(out_uninit, cols, &warm) };

    // Closure that writes one row
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // Cast the row slice to f64
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => smma_row_scalar(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => smma_row_avx2(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => smma_row_avx512(data, first, period, dst),
            _ => smma_row_scalar(data, first, period, dst), // Default to scalar
        }
    };

    // Run every row kernel
    if parallel {
        #[cfg(not(target_arch = "wasm32"))]
        {
            out_uninit
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row, slice)| do_row(row, slice));
        }
        #[cfg(target_arch = "wasm32")]
        {
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
pub unsafe fn smma_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    smma_scalar(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn smma_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    // Reuse the AVX2 kernel directly
    smma_avx2(data, period, first, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn smma_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    if period <= 32 {
        smma_row_avx512_short(data, first, period, out);
    } else {
        smma_row_avx512_long(data, first, period, out);
    }
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn smma_row_avx512_short(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    smma_row_avx2(data, first, period, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn smma_row_avx512_long(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    smma_row_avx2(data, first, period, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_smma_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SmmaInput::from_candles(&candles, "close", SmmaParams { period: None });
        let output = smma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_smma_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SmmaInput::from_candles(&candles, "close", SmmaParams::default());
        let result = smma_with_kernel(&input, kernel)?;
        let expected_last_five = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] SMMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }
    fn check_smma_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SmmaInput::with_default_candles(&candles);
        match input.data {
            SmmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected SmmaData::Candles"),
        }
        let output = smma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_smma_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = SmmaParams { period: Some(0) };
        let input = SmmaInput::from_slice(&input_data, params);
        let res = smma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SMMA should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_smma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = SmmaParams { period: Some(10) };
        let input = SmmaInput::from_slice(&data_small, params);
        let res = smma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SMMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_smma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = SmmaParams { period: Some(9) };
        let input = SmmaInput::from_slice(&single_point, params);
        let res = smma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SMMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_smma_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: Vec<f64> = vec![];
        let params = SmmaParams { period: Some(7) };
        let input = SmmaInput::from_slice(&empty, params);
        let res = smma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] SMMA should fail with empty input",
            test_name
        );
        if let Err(SmmaError::EmptyInputData) = res {
            // Good, expected error type
        } else {
            panic!(
                "[{}] Expected EmptyInputData error, got {:?}",
                test_name, res
            );
        }
        Ok(())
    }
    fn check_smma_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = SmmaParams { period: Some(7) };
        let first_input = SmmaInput::from_candles(&candles, "close", first_params);
        let first_result = smma_with_kernel(&first_input, kernel)?;
        let second_params = SmmaParams { period: Some(5) };
        let second_input = SmmaInput::from_slice(&first_result.values, second_params);
        let second_result = smma_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        Ok(())
    }
    fn check_smma_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = SmmaInput::from_candles(&candles, "close", SmmaParams { period: Some(7) });
        let res = smma_with_kernel(&input, kernel)?;
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
    fn check_smma_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 7;
        let input = SmmaInput::from_candles(
            &candles,
            "close",
            SmmaParams {
                period: Some(period),
            },
        );
        let batch_output = smma_with_kernel(&input, kernel)?.values;
        let mut stream = SmmaStream::try_new(SmmaParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(smma_val) => stream_values.push(smma_val),
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
                diff < 1e-7,
                "[{}] SMMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_smma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Property test strategy: generate period and matching data length
        let strat = (1usize..=100) // period (include 1 for edge case testing)
            .prop_flat_map(|period| {
                (
                    prop::collection::vec(
                        (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                        period.max(2)..400, // ensure we have at least 2 data points
                    ),
                    Just(period),
                )
            });

        proptest::test_runner::TestRunner::default().run(&strat, |(data, period)| {
            let params = SmmaParams {
                period: Some(period),
            };
            let input = SmmaInput::from_slice(&data, params);

            // Get output for the kernel being tested
            let output = smma_with_kernel(&input, kernel)?;

            // Get scalar reference for cross-kernel validation
            let reference = smma_with_kernel(&input, Kernel::Scalar)?;

            // Property 1: Output length equals input length
            prop_assert_eq!(output.values.len(), data.len());
            prop_assert_eq!(reference.values.len(), data.len());

            // Property 2: First period-1 values are NaN (warmup period)
            // Skip for period=1 since there's no warmup
            if period > 1 {
                for i in 0..period - 1 {
                    prop_assert!(
                        output.values[i].is_nan(),
                        "Expected NaN at index {} but got {}",
                        i,
                        output.values[i]
                    );
                }
            }

            // Property 3: First valid SMMA value should be simple average of first period values
            // For period=1, first valid is at index 0; otherwise at period-1
            let first_smma_idx = if period == 1 { 0 } else { period - 1 };
            let first_sum: f64 = data[0..period].iter().sum();
            let expected_first = first_sum / period as f64;
            let actual_first = output.values[first_smma_idx];
            prop_assert!(
                (actual_first - expected_first).abs() < 1e-7,
                "First SMMA value mismatch: expected {}, got {} (diff: {})",
                expected_first,
                actual_first,
                (actual_first - expected_first).abs()
            );

            // Property 4: Verify recursive formula for subsequent values
            // SMMA[i] = (SMMA[i-1] * (period - 1) + data[i]) / period
            if data.len() > period {
                for i in period..data.len().min(period + 10) {
                    let prev_smma = output.values[i - 1];
                    let expected = (prev_smma * (period as f64 - 1.0) + data[i]) / period as f64;
                    let actual = output.values[i];

                    // Allow small tolerance for floating-point arithmetic
                    prop_assert!(
                        (actual - expected).abs() < 1e-7,
                        "Recursive formula mismatch at index {}: expected {}, got {} (diff: {})",
                        i,
                        expected,
                        actual,
                        (actual - expected).abs()
                    );
                }
            }

            // Property 5: Cross-kernel validation - compare against scalar reference
            for i in 0..output.values.len() {
                let test_val = output.values[i];
                let ref_val = reference.values[i];

                // Both should be NaN or both should be finite
                if test_val.is_nan() && ref_val.is_nan() {
                    continue;
                }

                prop_assert!(
                    test_val.is_finite() == ref_val.is_finite(),
                    "Finite/NaN mismatch at index {}: test={}, ref={}",
                    i,
                    test_val,
                    ref_val
                );

                if test_val.is_finite() && ref_val.is_finite() {
                    // Check ULP difference for floating-point precision
                    let test_bits = test_val.to_bits();
                    let ref_bits = ref_val.to_bits();
                    let ulp_diff = test_bits.abs_diff(ref_bits);

                    // SMMA uses simple arithmetic; allow up to 10/20 ULPs or abs < 1e-7
                    let max_ulps = if matches!(kernel, Kernel::Avx512 | Kernel::Avx512Batch) { 20 } else { 10 };

                    prop_assert!(
						ulp_diff <= max_ulps || (test_val - ref_val).abs() < 1e-7,
						"Cross-kernel mismatch at index {}: test={}, ref={}, ULP diff={}, abs diff={}",
						i,
						test_val,
						ref_val,
						ulp_diff,
						(test_val - ref_val).abs()
					);
                }
            }

            // Property 6: SMMA should be bounded by min/max of all data seen so far
            // (not just the window, since SMMA has infinite memory)
            let start_idx = if period == 1 { 0 } else { period - 1 };
            for i in start_idx..output.values.len() {
                let val = output.values[i];
                if val.is_finite() {
                    let data_up_to_i = &data[0..=i];
                    let min = data_up_to_i.iter().copied().fold(f64::INFINITY, f64::min);
                    let max = data_up_to_i
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);

                    prop_assert!(
                        val >= min - 1e-9 && val <= max + 1e-9,
                        "SMMA value {} at index {} outside historical bounds [{}, {}]",
                        val,
                        i,
                        min,
                        max
                    );
                }
            }

            // Property 7: For constant data, SMMA should converge to that constant
            if data.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON) {
                // All data points are essentially the same
                let constant_val = data[0];
                let check_start = if period == 1 { 0 } else { period - 1 };
                for i in check_start..output.values.len() {
                    let val = output.values[i];
                    prop_assert!(
                        (val - constant_val).abs() < 1e-7,
                        "SMMA should converge to {} for constant data, but got {} at index {}",
                        constant_val,
                        val,
                        i
                    );
                }
            }

            // Property 8: Period = 1 edge case - output should equal input after warmup
            if period == 1 {
                prop_assert_eq!(
                    output.values[0],
                    data[0],
                    "Period=1: first value should equal input"
                );
                for i in 1..data.len() {
                    prop_assert!(
                        (output.values[i] - data[i]).abs() < 1e-7,
                        "Period=1: output should equal input at index {}: {} != {}",
                        i,
                        output.values[i],
                        data[i]
                    );
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_smma_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to increase chance of catching bugs
        let test_periods = vec![
            5,   // Small period
            7,   // Default period
            14,  // Medium period
            50,  // Large period
            100, // Very large period
            200, // Extra large period
        ];

        for &period in &test_periods {
            // Skip if period would be too large for the data
            if period > candles.close.len() {
                continue;
            }

            let input = SmmaInput::from_candles(
                &candles,
                "close",
                SmmaParams {
                    period: Some(period),
                },
            );
            let output = smma_with_kernel(&input, kernel)?;

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
    fn check_smma_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    macro_rules! generate_all_smma_tests {
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
    generate_all_smma_tests!(
        check_smma_partial_params,
        check_smma_accuracy,
        check_smma_default_candles,
        check_smma_zero_period,
        check_smma_period_exceeds_length,
        check_smma_very_small_dataset,
        check_smma_empty_input,
        check_smma_reinput,
        check_smma_nan_handling,
        check_smma_streaming,
        check_smma_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_smma_tests!(check_smma_property);
    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = SmmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = SmmaParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5];
        let start = row.len() - 5;
        for (i, &v) in row[start..].iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-1,
                "[{test}] default-row mismatch at idx {i}: {v} vs {expected:?}"
            );
        }
        Ok(())
    }

    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple batch configurations to increase detection coverage
        let batch_configs = vec![
            (3, 7, 1),     // Small range with step 1
            (10, 30, 10),  // Medium range with larger step
            (5, 100, 5),   // Large range with step 5
            (50, 200, 50), // Very large periods
            (1, 10, 1),    // Edge case: starting from 1
        ];

        for (start, end, step) in batch_configs {
            // Skip configurations that would exceed data length
            if start > c.close.len() {
                continue;
            }

            let output = SmmaBatchBuilder::new()
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
                let period = output.combos[row].period.unwrap_or(0);

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) for period {} in range ({}, {}, {})",
                        test, val, bits, row, col, idx, period, start, end, step
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(
        _test: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
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
    gen_batch_tests!(check_batch_no_poison);
}

#[cfg(feature = "python")]
#[pyfunction(name = "smma")]
#[pyo3(signature = (data, period, kernel=None))]
/// Compute the Smoothed Moving Average (SMMA) of the input data.
///
/// SMMA uses a recursive smoothing formula where the first value is the mean
/// of the first `period` points and subsequent values use a blend of the prior
/// SMMA and the new point.
///
/// Parameters:
/// -----------
/// data : np.ndarray
///     Input data array (float64).
/// period : int
///     Number of data points in the moving average window.
/// kernel : str, optional
///     Computation kernel to use: 'auto', 'scalar', 'avx2', 'avx512'.
///     Default is 'auto' which auto-detects the best available.
///
/// Returns:
/// --------
/// np.ndarray
///     Array of SMMA values, same length as input.
///
/// Raises:
/// -------
/// ValueError
///     If inputs are invalid (period <= 0, insufficient data, etc).
pub fn smma_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?; // Validate before allow_threads

    let params = SmmaParams {
        period: Some(period),
    };
    let smma_in = SmmaInput::from_slice(slice_in, params);

    // Get Vec<f64> from Rust function
    let result_vec: Vec<f64> = py
        .allow_threads(|| smma_with_kernel(&smma_in, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "SmmaStream")]
pub struct SmmaStreamPy {
    stream: SmmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl SmmaStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = SmmaParams {
            period: Some(period),
        };
        let stream =
            SmmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(SmmaStreamPy { stream })
    }

    /// Updates the stream with a new value and returns the calculated SMMA value.
    /// Returns `None` if the buffer is not yet full.
    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "smma_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
/// Compute SMMA for multiple period values in a single pass.
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
///     Dictionary with 'values' (2D array) and 'periods' arrays.
pub fn smma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?; // true for batch operations
    let sweep = SmmaBatchRange {
        period: period_range,
    };

    // Calculate dimensions
    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    // Pre-allocate output array (OK for batch operations)
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    // Compute without GIL
    let combos = py
        .allow_threads(|| {
            // Handle kernel selection for batch operations
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

            smma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Build result dictionary
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

    Ok(dict.into())
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "smma_cuda_batch_dev")]
#[pyo3(signature = (data, period_range, device_id=0))]
pub fn smma_cuda_batch_dev_py(
    py: Python<'_>,
    data: numpy::PyReadonlyArray1<'_, f64>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data.as_slice()?;
    let sweep = SmmaBatchRange {
        period: period_range,
    };
    let data_f32: Vec<f32> = slice_in.iter().map(|&v| v as f32).collect();

    let inner = py.allow_threads(|| {
        let cuda = CudaSmma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.smma_batch_dev(&data_f32, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "smma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn smma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
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
    let params = SmmaParams {
        period: Some(period),
    };

    let inner = py.allow_threads(|| {
        let cuda = CudaSmma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.smma_multi_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner })
}

/// Write SMMA values directly to output slice - no allocations
#[inline]
pub fn smma_into_slice(dst: &mut [f64], input: &SmmaInput, kern: Kernel) -> Result<(), SmmaError> {
    let (data, period, first, chosen) = smma_prepare(input, kern)?;

    // Verify output buffer size matches input
    if dst.len() != data.len() {
        return Err(SmmaError::OutputLenMismatch {
            expected: data.len(),
            actual: dst.len(),
        });
    }

    // Reinterpret dst as MaybeUninit to use init_matrix_prefixes for NaN filling
    let warmup_end = first + period - 1;
    unsafe {
        let dst_uninit =
            std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut MaybeUninit<f64>, dst.len());
        init_matrix_prefixes(dst_uninit, dst.len(), &[warmup_end]);
    }

    // Compute SMMA values directly into dst
    smma_compute_into(data, period, first, chosen, dst);

    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "smma")]
pub fn smma_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = SmmaParams {
        period: Some(period),
    };
    let input = SmmaInput::from_slice(data, params);

    // Allocate output buffer once
    let mut output = vec![0.0; data.len()];

    // Compute directly into output buffer
    smma_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SmmaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct SmmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<SmmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "smma_batch")]
pub fn smma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: SmmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = SmmaBatchRange {
        period: config.period_range,
    };

    let output = smma_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = SmmaBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Legacy wrapper for backward compatibility
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "smma_batch_legacy")]
pub fn smma_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = SmmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    smma_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map(|output| output.values)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "smma_batch_metadata")]
/// Get metadata about the batch computation (periods used)
pub fn smma_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Vec<usize> {
    let range = SmmaBatchRange {
        period: (period_start, period_end, period_step),
    };
    let combos = expand_grid(&range);
    combos.iter().map(|c| c.period.unwrap_or(7)).collect()
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "smma_batch_rows_cols")]
/// Get the dimensions of the batch output
pub fn smma_batch_rows_cols_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
    data_len: usize,
) -> Vec<usize> {
    let range = SmmaBatchRange {
        period: (period_start, period_end, period_step),
    };
    let combos = expand_grid(&range);
    vec![combos.len(), data_len]
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn smma_alloc(len: usize) -> *mut f64 {
    // Allocate memory for input/output buffer
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Prevent deallocation
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn smma_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn smma_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to smma_into"));
    }

    unsafe {
        // Create slice from pointer
        let data = std::slice::from_raw_parts(in_ptr, len);

        // Validate inputs
        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        // Calculate SMMA
        let params = SmmaParams {
            period: Some(period),
        };
        let input = SmmaInput::from_slice(data, params);

        // Check for aliasing (input and output buffers are the same)
        if in_ptr == out_ptr as *const f64 {
            // Aliasing detected - use temporary buffer
            let mut temp = vec![0.0; len];
            smma_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results back to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, compute directly into output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            smma_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn smma_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to smma_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = SmmaBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        // Resolve Auto kernel to concrete batch kernel, then map to non-batch
        let batch = detect_best_batch_kernel();
        let simd = match batch {
            Kernel::Avx512Batch => Kernel::Avx512,
            Kernel::Avx2Batch => Kernel::Avx2,
            _ => Kernel::Scalar,
        };

        // Use optimized batch processing
        smma_batch_inner_into(data, &sweep, simd, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
