//! # Ease of Movement (EMV)
//!
//! Measures how easily price moves given volume. Calculates the ratio of price movement to volume.
//!
//! ## Parameters
//! - None (requires high, low, close, and volume data)
//!
//! ## Returns
//! - **`Ok(EmvOutput)`** on success (`values: Vec<f64>` of length matching input)
//! - **`Err(EmvError)`** on failure
//!
//! ## Developer Status
//! - **SIMD Kernels**: AVX2/AVX512 present as stubs delegating to scalar. EMV has a strict
//!   loop-carried dependency on the prior valid midpoint and NaN/zero-range semantics, making
//!   wide SIMD non-viable without complex masked prefix-scan logic and risking parity. Runtime
//!   selection keeps kernels for parity but they currently short-circuit to the scalar path.
//! - **Row-specific batch**: Not applicable — EMV has no parameters; batch is a single row.
//! - **Streaming**: O(1) performance with exact parity vs batch. An optional
//!   fast update helper is provided (multiply-by-reciprocal + Newton refine),
//!   but default streaming keeps exact arithmetic order for test parity.
//! - **Memory**: Good zero-copy usage (alloc_with_nan_prefix, make_uninit_matrix)
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
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::DeviceArrayF32Py;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::CudaEmv;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub enum EmvData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct EmvOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct EmvParams;

#[derive(Debug, Clone)]
pub struct EmvInput<'a> {
    pub data: EmvData<'a>,
    pub params: EmvParams,
}

impl<'a> EmvInput<'a> {
    #[inline(always)]
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: EmvData::Candles { candles },
            params: EmvParams,
        }
    }

    #[inline(always)]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    ) -> Self {
        Self {
            data: EmvData::Slices {
                high,
                low,
                close,
                volume,
            },
            params: EmvParams,
        }
    }

    #[inline(always)]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct EmvBuilder {
    kernel: Kernel,
}

impl EmvBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<EmvOutput, EmvError> {
        let input = EmvInput::from_candles(c);
        emv_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<EmvOutput, EmvError> {
        let input = EmvInput::from_slices(high, low, close, volume);
        emv_with_kernel(&input, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<EmvStream, EmvError> {
        EmvStream::try_new()
    }
}

#[derive(Debug, Error)]
pub enum EmvError {
    #[error("emv: Empty data provided.")]
    EmptyData,
    #[error("emv: Not enough data: needed at least 2 valid points, found {valid}.")]
    NotEnoughData { valid: usize },
    #[error("emv: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn emv(input: &EmvInput) -> Result<EmvOutput, EmvError> {
    emv_with_kernel(input, Kernel::Auto)
}

pub fn emv_with_kernel(input: &EmvInput, kernel: Kernel) -> Result<EmvOutput, EmvError> {
    let (high, low, close, volume) = match &input.data {
        EmvData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            let volume = source_type(candles, "volume");
            (high, low, close, volume)
        }
        EmvData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    if high.is_empty() || low.is_empty() || volume.is_empty() {
        return Err(EmvError::EmptyData);
    }
    let len = high.len().min(low.len()).min(volume.len());
    if len == 0 {
        return Err(EmvError::EmptyData);
    }

    let first = (0..len).find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()));
    let first = match first {
        Some(idx) => idx,
        None => return Err(EmvError::AllValuesNaN),
    };

    let mut valid_count = 0_usize;
    for i in first..len {
        if !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()) {
            valid_count += 1;
        }
    }
    if valid_count < 2 {
        return Err(EmvError::NotEnoughData { valid: valid_count });
    }

    let mut out = alloc_with_nan_prefix(len, first + 1);
    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => emv_scalar(high, low, volume, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => emv_avx2(high, low, volume, first, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => emv_avx512(high, low, volume, first, &mut out),
            _ => unreachable!(),
        }
    }
    Ok(EmvOutput { values: out })
}

#[inline]
pub fn emv_into_slice(dst: &mut [f64], input: &EmvInput, kern: Kernel) -> Result<(), EmvError> {
    let (high, low, close, volume) = match &input.data {
        EmvData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            let volume = source_type(candles, "volume");
            (high, low, close, volume)
        }
        EmvData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    if high.is_empty() || low.is_empty() || volume.is_empty() {
        return Err(EmvError::EmptyData);
    }
    let len = high.len().min(low.len()).min(volume.len());
    if len == 0 {
        return Err(EmvError::EmptyData);
    }

    if dst.len() != len {
        return Err(EmvError::NotEnoughData { valid: dst.len() });
    }

    let first = (0..len).find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()));
    let first = match first {
        Some(idx) => idx,
        None => return Err(EmvError::AllValuesNaN),
    };

    let mut valid_count = 0_usize;
    for i in first..len {
        if !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()) {
            valid_count += 1;
        }
    }
    if valid_count < 2 {
        return Err(EmvError::NotEnoughData { valid: valid_count });
    }

    // Fill warmup period with NaN
    for v in &mut dst[..(first + 1)] {
        *v = f64::NAN;
    }

    let chosen = match kern {
        Kernel::Auto => detect_best_kernel(),
        other => other,
    };

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => emv_scalar(high, low, volume, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => emv_avx2(high, low, volume, first, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => emv_avx512(high, low, volume, first, dst),
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[inline]
pub fn emv_scalar(high: &[f64], low: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    let len = high.len().min(low.len()).min(volume.len());
    let mut last_mid = 0.5 * (high[first] + low[first]);

    // Tight, safe scalar loop with hoisted loads to minimize bounds checks
    for i in (first + 1)..len {
        let h = high[i];
        let l = low[i];
        let v = volume[i];

        if h.is_nan() || l.is_nan() || v.is_nan() {
            out[i] = f64::NAN;
            continue;
        }

        // Keep arithmetic order identical to streaming path
        let current_mid = 0.5 * (h + l);
        let range = h - l;
        if range == 0.0 {
            out[i] = f64::NAN;
            last_mid = current_mid; // advance last_mid on zero-range
            continue;
        }

        let br = v / 10000.0 / range;
        out[i] = (current_mid - last_mid) / br;
        last_mid = current_mid;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn emv_avx512(high: &[f64], low: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    // Delegate to AVX2 stub which uses an unsafe pointer-walk scalar kernel.
    // Rationale: EMV is dependency-bound; SIMD is not beneficial without complex
    // masked scans. Keep parity by reusing the same kernel.
    emv_avx2(high, low, volume, first, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub fn emv_avx2(high: &[f64], low: &[f64], volume: &[f64], first: usize, out: &mut [f64]) {
    // Unsafe pointer-walk variant of the scalar kernel to minimize bound checks.
    // Keeps arithmetic order identical to streaming for parity.
    let len = high.len().min(low.len()).min(volume.len());
    let mut last_mid = 0.5 * (high[first] + low[first]);
    unsafe {
        let h_ptr = high.as_ptr();
        let l_ptr = low.as_ptr();
        let v_ptr = volume.as_ptr();
        let o_ptr = out.as_mut_ptr();

        let mut i = first + 1;
        while i < len {
            let h = *h_ptr.add(i);
            let l = *l_ptr.add(i);
            let v = *v_ptr.add(i);

            if !(h.is_nan() || l.is_nan() || v.is_nan()) {
                let range = h - l;
                let current_mid = 0.5 * (h + l);

                if range == 0.0 {
                    *o_ptr.add(i) = f64::NAN;
                    last_mid = current_mid;
                } else {
                    let br = (v / 10000.0) / range;
                    let dmid = current_mid - last_mid;
                    *o_ptr.add(i) = dmid / br;
                    last_mid = current_mid;
                }
            } else {
                // Any NaN in inputs -> NaN output; do not advance last_mid.
                *o_ptr.add(i) = f64::NAN;
            }

            i += 1;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn emv_avx512_short(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    // Route to the AVX2 stub to keep a single parity-preserving kernel.
    emv_avx2(high, low, volume, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn emv_avx512_long(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    out: &mut [f64],
) {
    // Route to the AVX2 stub to keep a single parity-preserving kernel.
    emv_avx2(high, low, volume, first, out);
}

#[derive(Debug, Clone)]
pub struct EmvStream {
    last_mid: Option<f64>,
}

impl EmvStream {
    pub fn try_new() -> Result<Self, EmvError> {
        Ok(Self { last_mid: None })
    }

    #[inline(always)]
    pub fn update(&mut self, high: f64, low: f64, volume: f64) -> Option<f64> {
        if high.is_nan() || low.is_nan() || volume.is_nan() {
            return None;
        }
        let current_mid = 0.5 * (high + low);
        if self.last_mid.is_none() {
            self.last_mid = Some(current_mid);
            return None;
        }
        let last_mid = self.last_mid.unwrap();
        let range = high - low;
        if range == 0.0 {
            self.last_mid = Some(current_mid);
            return None;
        }
        let br = volume / 10000.0 / range;
        let out = (current_mid - last_mid) / br;
        self.last_mid = Some(current_mid);
        Some(out)
    }

    /// Optional faster update using reciprocal + Newton refinement.
    /// Not bit-for-bit identical to batch (re-association); keep `update` as default.
    #[inline(always)]
    pub fn update_fast(&mut self, high: f64, low: f64, volume: f64) -> Option<f64> {
        if high.is_nan() || low.is_nan() || volume.is_nan() {
            return None;
        }
        let current_mid = 0.5 * (high + low);
        if self.last_mid.is_none() {
            self.last_mid = Some(current_mid);
            return None;
        }
        let last_mid = self.last_mid.unwrap();
        let range = high - low;
        if range == 0.0 {
            self.last_mid = Some(current_mid);
            return None;
        }
        // Re-associated form with fast reciprocal: (Δmid * range * 10000) * (1/volume)
        let inv_v = fast_recip_f64(volume);
        let out = (current_mid - last_mid) * range * 10_000.0 * inv_v;
        self.last_mid = Some(current_mid);
        Some(out)
    }
}

// --- fast math helpers -------------------------------------------------------
#[inline(always)]
fn newton_refine_recip(y0: f64, x: f64) -> f64 {
    // One Newton-Raphson step for 1/x: y_{n+1} = y_n * (2 - x*y_n)
    let t = 2.0_f64 - x.mul_add(y0, 0.0);
    y0 * t
}

#[inline(always)]
fn fast_recip_f64(x: f64) -> f64 {
    #[cfg(all(
        feature = "nightly-avx",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let vx = _mm512_set1_pd(x);
        let rcp = _mm512_rcp14_pd(vx);
        let lo = _mm512_castpd512_pd128(rcp);
        let y0 = _mm_cvtsd_f64(lo);
        let y1 = newton_refine_recip(y0, x);
        let y2 = newton_refine_recip(y1, x);
        return y2;
    }
    1.0 / x
}

#[derive(Clone, Debug)]
pub struct EmvBatchRange {}

impl Default for EmvBatchRange {
    fn default() -> Self {
        Self {}
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmvBatchBuilder {
    kernel: Kernel,
    _range: EmvBatchRange,
}

impl EmvBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Result<EmvBatchOutput, EmvError> {
        emv_batch_with_kernel(high, low, close, volume, self.kernel)
    }

    pub fn with_default_slices(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
        k: Kernel,
    ) -> Result<EmvBatchOutput, EmvError> {
        EmvBatchBuilder::new()
            .kernel(k)
            .apply_slices(high, low, close, volume)
    }

    pub fn apply_candles(self, c: &Candles) -> Result<EmvBatchOutput, EmvError> {
        let high = source_type(c, "high");
        let low = source_type(c, "low");
        let close = source_type(c, "close");
        let volume = source_type(c, "volume");
        self.apply_slices(high, low, close, volume)
    }

    pub fn with_default_candles(c: &Candles, k: Kernel) -> Result<EmvBatchOutput, EmvError> {
        EmvBatchBuilder::new().kernel(k).apply_candles(c)
    }
}

pub fn emv_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    _close: &[f64],
    volume: &[f64],
    kernel: Kernel,
) -> Result<EmvBatchOutput, EmvError> {
    let simd = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => Kernel::ScalarBatch,
    };
    emv_batch_par_slice(high, low, volume, simd)
}

#[derive(Clone, Debug)]
pub struct EmvBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EmvParams>, // parity with alma.rs
    pub rows: usize,
    pub cols: usize,
}

impl EmvBatchOutput {
    #[inline]
    pub fn single_row(&self) -> &[f64] {
        debug_assert_eq!(self.rows, 1);
        &self.values[..self.cols]
    }
}

#[inline(always)]
fn expand_grid(_r: &EmvBatchRange) -> Vec<()> {
    vec![()]
}

#[inline(always)]
pub fn emv_batch_slice(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<EmvBatchOutput, EmvError> {
    emv_batch_inner(high, low, volume, kern, false)
}

#[inline(always)]
pub fn emv_batch_par_slice(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    kern: Kernel,
) -> Result<EmvBatchOutput, EmvError> {
    emv_batch_inner(high, low, volume, kern, true)
}

fn emv_batch_inner(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    kern: Kernel,
    _parallel: bool, // no-op for EMV
) -> Result<EmvBatchOutput, EmvError> {
    let len = high.len().min(low.len()).min(volume.len());
    if len == 0 {
        return Err(EmvError::EmptyData);
    }

    let first = (0..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()))
        .ok_or(EmvError::AllValuesNaN)?;

    // Validate we have at least two valid points
    let valid = (first..len)
        .filter(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()))
        .count();
    if valid < 2 {
        return Err(EmvError::NotEnoughData { valid });
    }

    // 1 row x len cols
    let rows = 1usize;
    let cols = len;

    // Uninitialized matrix + warmup NaNs only for the needed prefix
    let mut buf_mu = make_uninit_matrix(rows, cols);
    init_matrix_prefixes(&mut buf_mu, cols, &[first + 1]);

    // Safely reinterpret to &mut [f64] without extra allocs or copies
    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    // Compute into output
    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => emv_scalar(high, low, volume, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => emv_avx2(high, low, volume, first, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => emv_avx512(high, low, volume, first, out),
            _ => emv_scalar(high, low, volume, first, out),
        }
    }

    // Move buffer out as Vec<f64> with zero copy
    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(EmvBatchOutput {
        values,
        combos: vec![EmvParams], // single combination
        rows,
        cols,
    })
}

#[inline(always)]
pub fn emv_row_scalar(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    emv_scalar(high, low, volume, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emv_row_avx2(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    emv_scalar(high, low, volume, first, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub fn emv_row_avx512(
    high: &[f64],
    low: &[f64],
    volume: &[f64],
    first: usize,
    _stride: usize,
    _w_ptr: *const f64,
    _inv_n: f64,
    out: &mut [f64],
) {
    emv_avx512(high, low, volume, first, out);
}

#[inline(always)]
fn expand_grid_emv(_r: &EmvBatchRange) -> Vec<()> {
    vec![()]
}

#[cfg(feature = "python")]
#[pyfunction(name = "emv")]
#[pyo3(signature = (high, low, close, volume, kernel=None))]
pub fn emv_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let data = EmvData::Slices {
        high: high_slice,
        low: low_slice,
        close: close_slice,
        volume: volume_slice,
    };
    let input = EmvInput {
        data,
        params: EmvParams,
    };

    let result_vec: Vec<f64> = py
        .allow_threads(|| emv_with_kernel(&input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "EmvStream")]
pub struct EmvStreamPy {
    stream: EmvStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EmvStreamPy {
    #[new]
    fn new() -> PyResult<Self> {
        let stream = EmvStream::try_new().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(EmvStreamPy { stream })
    }

    fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> Option<f64> {
        self.stream.update(high, low, volume)
    }
}

#[cfg(feature = "python")]
fn emv_batch_inner_into(
    high: &[f64],
    low: &[f64],
    _close: &[f64],
    volume: &[f64],
    _range: &EmvBatchRange,
    kern: Kernel,
    _parallel: bool,
    out: &mut [f64],
) -> Vec<EmvParams> {
    let len = high.len().min(low.len()).min(volume.len());
    // Treat NumPy-owned memory as MaybeUninit to set only warm prefixes
    let out_mu: &mut [MaybeUninit<f64>] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    let first = (0..len)
        .find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()))
        .unwrap_or(0);

    // Warm prefix NaNs without extra passes
    init_matrix_prefixes(out_mu, len, &[first + 1]);

    // Reinterpret back to f64 for compute
    let out_f: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(out_mu.as_mut_ptr() as *mut f64, out_mu.len()) };

    unsafe {
        match kern {
            Kernel::Scalar | Kernel::ScalarBatch => emv_scalar(high, low, volume, first, out_f),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => emv_avx2(high, low, volume, first, out_f),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => emv_avx512(high, low, volume, first, out_f),
            _ => emv_scalar(high, low, volume, first, out_f),
        }
    }

    vec![EmvParams]
}

#[cfg(feature = "python")]
#[pyfunction(name = "emv_batch")]
#[pyo3(signature = (high, low, close, volume, kernel=None))]
pub fn emv_batch_py<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;
    let volume_slice = volume.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    let sweep = EmvBatchRange {};
    let combos = expand_grid(&sweep);
    let rows = combos.len(); // Always 1 for EMV
    let cols = high_slice.len();

    // Pre-allocate output array
    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    py.allow_threads(|| {
        let kernel = match kern {
            Kernel::Auto => detect_best_batch_kernel(),
            k => k,
        };
        emv_batch_inner_into(
            high_slice,
            low_slice,
            close_slice,
            volume_slice,
            &sweep,
            kernel,
            true,
            slice_out,
        );
    });

    let dict = PyDict::new(py);
    dict.set_item("values", out_arr.reshape((rows, cols))?)?;
    // No parameter arrays for EMV since it has no parameters

    Ok(dict)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emv_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
) -> Result<Vec<f64>, JsValue> {
    let input = EmvInput::from_slices(high, low, close, volume);

    let mut output = vec![0.0; high.len().min(low.len()).min(close.len()).min(volume.len())];

    let kernel = detect_best_kernel();

    emv_into_slice(&mut output, &input, kernel).map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emv_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<(), JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer passed to emv_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        let input = EmvInput::from_slices(high, low, close, volume);

        let kernel = detect_best_kernel();

        // Check if output pointer aliases with any input pointer
        if out_ptr == high_ptr as *mut f64
            || out_ptr == low_ptr as *mut f64
            || out_ptr == close_ptr as *mut f64
            || out_ptr == volume_ptr as *mut f64
        {
            // Use temp buffer for aliased operation
            let mut temp = vec![0.0; len];
            emv_into_slice(&mut temp, &input, kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // Direct write to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            emv_into_slice(out, &input, kernel).map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emv_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emv_free(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EmvBatchConfig {
    // EMV has no parameters to sweep
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EmvBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EmvParams>, // add combos for parity
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = emv_batch)]
pub fn emv_batch_js(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    _config: JsValue,
) -> Result<JsValue, JsValue> {
    let input = EmvInput::from_slices(high, low, close, volume);
    let len = high.len().min(low.len()).min(close.len()).min(volume.len());

    let mut output = vec![0.0; len];

    let kernel = detect_best_kernel();

    emv_into_slice(&mut output, &input, kernel).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = EmvBatchJsOutput {
        values: output,
        combos: vec![EmvParams], // single combo, EMV has no params
        rows: 1,
        cols: len,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn emv_batch_into(
    high_ptr: *const f64,
    low_ptr: *const f64,
    close_ptr: *const f64,
    volume_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> Result<usize, JsValue> {
    if high_ptr.is_null()
        || low_ptr.is_null()
        || close_ptr.is_null()
        || volume_ptr.is_null()
        || out_ptr.is_null()
    {
        return Err(JsValue::from_str("null pointer passed to emv_batch_into"));
    }

    unsafe {
        let high = std::slice::from_raw_parts(high_ptr, len);
        let low = std::slice::from_raw_parts(low_ptr, len);
        let close = std::slice::from_raw_parts(close_ptr, len);
        let volume = std::slice::from_raw_parts(volume_ptr, len);

        let input = EmvInput::from_slices(high, low, close, volume);

        let kernel = detect_best_kernel();

        let out = std::slice::from_raw_parts_mut(out_ptr, len);
        emv_into_slice(out, &input, kernel).map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(1) // Always 1 row for EMV (no parameter sweep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;

    fn check_emv_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EmvInput::from_candles(&candles);
        let output = emv_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        let expected_last_five_emv = [
            -6488905.579799851,
            2371436.7401001123,
            -3855069.958128531,
            1051939.877943717,
            -8519287.22257077,
        ];
        let start = output.values.len().saturating_sub(5);
        for (i, &val) in output.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five_emv[i]).abs();
            let tol = expected_last_five_emv[i].abs() * 0.0001;
            assert!(
                diff <= tol,
                "[{}] EMV {:?} mismatch at idx {}: got {}, expected {}, diff={}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five_emv[i],
                diff
            );
        }
        Ok(())
    }

    fn check_emv_with_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = EmvInput::with_default_candles(&candles);
        let output = emv_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_emv_empty_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = EmvInput::from_slices(&empty, &empty, &empty, &empty);
        let result = emv_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_emv_all_nan(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let nan_arr = [f64::NAN, f64::NAN];
        let input = EmvInput::from_slices(&nan_arr, &nan_arr, &nan_arr, &nan_arr);
        let result = emv_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_emv_not_enough_data(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10000.0, f64::NAN];
        let low = [9990.0, f64::NAN];
        let close = [9995.0, f64::NAN];
        let volume = [1_000_000.0, f64::NAN];
        let input = EmvInput::from_slices(&high, &low, &close, &volume);
        let result = emv_with_kernel(&input, kernel);
        assert!(result.is_err());
        Ok(())
    }

    fn check_emv_basic_calculation(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let high = [10.0, 12.0, 13.0, 15.0];
        let low = [5.0, 7.0, 8.0, 10.0];
        let close = [7.5, 9.0, 10.5, 12.5];
        let volume = [10000.0, 20000.0, 25000.0, 30000.0];
        let input = EmvInput::from_slices(&high, &low, &close, &volume);
        let output = emv_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), 4);
        assert!(output.values[0].is_nan());
        for &val in &output.values[1..] {
            assert!(!val.is_nan());
        }
        Ok(())
    }

    fn check_emv_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let high = source_type(&candles, "high");
        let low = source_type(&candles, "low");
        let volume = source_type(&candles, "volume");

        let output = emv_with_kernel(&EmvInput::from_candles(&candles), kernel)?.values;

        let mut stream = EmvStream::try_new()?;
        let mut stream_values = Vec::with_capacity(high.len());
        for i in 0..high.len() {
            match stream.update(high[i], low[i], volume[i]) {
                Some(val) => stream_values.push(val),
                None => stream_values.push(f64::NAN),
            }
        }
        assert_eq!(output.len(), stream_values.len());
        for (b, s) in output.iter().zip(stream_values.iter()) {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] EMV streaming f64 mismatch: batch={}, stream={}, diff={}",
                test_name,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_emv_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Since EMV has no parameters, we test different ways of calling it
        // Test 1: From candles
        let input1 = EmvInput::from_candles(&candles);
        let output1 = emv_with_kernel(&input1, kernel)?;

        // Test 2: From slices
        let high = source_type(&candles, "high");
        let low = source_type(&candles, "low");
        let close = source_type(&candles, "close");
        let volume = source_type(&candles, "volume");
        let input2 = EmvInput::from_slices(high, low, close, volume);
        let output2 = emv_with_kernel(&input2, kernel)?;

        // Test 3: With default candles
        let input3 = EmvInput::with_default_candles(&candles);
        let output3 = emv_with_kernel(&input3, kernel)?;

        // Check all outputs for poison values
        let outputs = [
            ("from_candles", &output1.values),
            ("from_slices", &output2.values),
            ("with_default_candles", &output3.values),
        ];

        for (method_name, values) in &outputs {
            for (i, &val) in values.iter().enumerate() {
                if val.is_nan() {
                    continue; // NaN values are expected during warmup
                }

                let bits = val.to_bits();

                // Check all three poison patterns
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
						 using method: {}",
                        test_name, val, bits, i, method_name
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
						 using method: {}",
                        test_name, val, bits, i, method_name
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
						 using method: {}",
                        test_name, val, bits, i, method_name
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_emv_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(()) // No-op in release builds
    }

    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_emv_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Generate realistic market data
        let strat = prop::collection::vec(
            (
                // High price (10 to 100000)
                10.0f64..100000.0f64,
                // Low as percentage of high (50% to 99.9% of high)
                0.5f64..0.999f64,
                // Volume (1000 to 1e9)
                1000.0f64..1e9f64,
            ),
            2..400, // Need at least 2 points for EMV
        )
        .prop_map(|data| {
            // Convert percentages to actual low values
            let high: Vec<f64> = data.iter().map(|(h, _, _)| *h).collect();
            let low: Vec<f64> = data
                .iter()
                .zip(&high)
                .map(|((_, l_pct, _), h)| h * l_pct)
                .collect();
            let volume: Vec<f64> = data.iter().map(|(_, _, v)| *v).collect();
            // Close values are not used in EMV calculation, but needed for API
            let close = high.clone();
            (high, low, close, volume)
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(high, low, close, volume)| {
                let input = EmvInput::from_slices(&high, &low, &close, &volume);

                // Test with specified kernel
                let EmvOutput { values: out } = emv_with_kernel(&input, kernel).unwrap();

                // Test with scalar reference
                let EmvOutput { values: ref_out } =
                    emv_with_kernel(&input, Kernel::Scalar).unwrap();

                // Property 1: Warmup period - first value should always be NaN
                prop_assert!(
                    out[0].is_nan(),
                    "First EMV value should always be NaN (warmup period)"
                );

                // Property 2: Output finiteness - when inputs are finite, outputs should be finite (except warmup and zero range)
                for i in 1..out.len() {
                    if high[i].is_finite() && low[i].is_finite() && volume[i].is_finite() {
                        let range = high[i] - low[i];
                        if range != 0.0 {
                            prop_assert!(
								out[i].is_finite(),
								"EMV at index {} should be finite when inputs are finite and range != 0",
								i
							);
                        }
                    }
                }

                // Property 3: Kernel consistency - different kernels should produce nearly identical results
                for i in 0..out.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    if !y.is_finite() || !r.is_finite() {
                        // Both should be NaN or infinite in the same way
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "Non-finite mismatch at index {}: {} vs {}",
                            i,
                            y,
                            r
                        );
                    } else {
                        // Check ULP difference for finite values
                        let y_bits = y.to_bits();
                        let r_bits = r.to_bits();
                        let ulp_diff = y_bits.abs_diff(r_bits);

                        prop_assert!(
                            ulp_diff <= 3,
                            "ULP difference too large at index {}: {} vs {} (ULP={})",
                            i,
                            y,
                            r,
                            ulp_diff
                        );
                    }
                }

                // Property 4: EMV formula verification
                // EMV = (current_mid - last_mid) / (volume / 10000 / range)
                let mut last_mid = 0.5 * (high[0] + low[0]);
                for i in 1..out.len() {
                    let current_mid = 0.5 * (high[i] + low[i]);
                    let range = high[i] - low[i];

                    if range == 0.0 {
                        // Property 5: Zero range handling - output should be NaN
                        prop_assert!(
                            out[i].is_nan(),
                            "EMV at index {} should be NaN when range is zero",
                            i
                        );
                    } else {
                        let expected_emv = (current_mid - last_mid) / (volume[i] / 10000.0 / range);

                        // Allow for small numerical errors
                        if out[i].is_finite() && expected_emv.is_finite() {
                            let diff = (out[i] - expected_emv).abs();
                            let tolerance = 1e-9;
                            prop_assert!(
                                diff <= tolerance,
                                "EMV formula mismatch at index {}: got {}, expected {}, diff={}",
                                i,
                                out[i],
                                expected_emv,
                                diff
                            );
                        }
                    }

                    last_mid = current_mid;
                }

                // Property 6: Bounded output - EMV should be reasonable relative to price movement
                for i in 1..out.len() {
                    if out[i].is_finite() {
                        let price_change =
                            (high[i] + low[i]) / 2.0 - (high[i - 1] + low[i - 1]) / 2.0;
                        let max_reasonable = price_change.abs() * 1e8; // Very generous bound

                        prop_assert!(
                            out[i].abs() <= max_reasonable,
                            "EMV at index {} seems unreasonably large: {} (price change: {})",
                            i,
                            out[i],
                            price_change
                        );
                    }
                }

                // Property 7: Constant data with non-zero range produces zero EMV
                if high.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && low.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
                    && high.iter().zip(&low).all(|(h, l)| h > l)
                {
                    for i in 1..out.len() {
                        if out[i].is_finite() {
                            prop_assert!(
                                out[i].abs() < 1e-9,
                                "EMV should be ~0 for constant prices, got {} at index {}",
                                out[i],
                                i
                            );
                        }
                    }
                }

                // Property 8: No poison values
                for (i, &val) in out.iter().enumerate() {
                    if !val.is_nan() {
                        let bits = val.to_bits();
                        prop_assert!(
                            bits != 0x11111111_11111111
                                && bits != 0x22222222_22222222
                                && bits != 0x33333333_33333333,
                            "Found poison value at index {}: {} (0x{:016X})",
                            i,
                            val,
                            bits
                        );
                    }
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_emv_tests {
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

    generate_all_emv_tests!(
        check_emv_accuracy,
        check_emv_with_default_candles,
        check_emv_empty_data,
        check_emv_all_nan,
        check_emv_not_enough_data,
        check_emv_basic_calculation,
        check_emv_streaming,
        check_emv_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_emv_tests!(check_emv_property);

    fn check_batch_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = EmvBatchBuilder::new().kernel(kernel).apply_candles(&c)?;
        assert_eq!(output.values.len(), c.close.len());
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
    gen_batch_tests!(check_batch_row);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // EMV batch has no parameters to sweep, so we just test the default case
        let output = EmvBatchBuilder::new().kernel(kernel).apply_candles(&c)?;

        // Check values for poison patterns
        for (idx, &val) in output.values.iter().enumerate() {
            if val.is_nan() {
                continue;
            }

            let bits = val.to_bits();
            let row = idx / output.cols;
            let col = idx % output.cols;

            // Check all three poison patterns with detailed context
            if bits == 0x11111111_11111111 {
                panic!(
                    "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) \
					 at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }

            if bits == 0x22222222_22222222 {
                panic!(
                    "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) \
					 at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
            }

            if bits == 0x33333333_33333333 {
                panic!(
                    "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) \
					 at row {} col {} (flat index {})",
                    test, val, bits, row, col, idx
                );
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
}

// ---------------- Python CUDA bindings ----------------
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "emv_cuda_batch_dev")]
#[pyo3(signature = (high_f32, low_f32, volume_f32, device_id=0))]
pub fn emv_cuda_batch_dev_py<'py>(
    py: Python<'py>,
    high_f32: numpy::PyReadonlyArray1<'py, f32>,
    low_f32: numpy::PyReadonlyArray1<'py, f32>,
    volume_f32: numpy::PyReadonlyArray1<'py, f32>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let h = high_f32.as_slice()?;
    let l = low_f32.as_slice()?;
    let v = volume_f32.as_slice()?;
    let inner = py.allow_threads(|| {
        let cuda = CudaEmv::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.emv_batch_dev(h, l, v)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "emv_cuda_many_series_one_param_dev")]
#[pyo3(signature = (high_tm_f32, low_tm_f32, volume_tm_f32, device_id=0))]
pub fn emv_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    high_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    low_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    volume_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    use numpy::PyUntypedArrayMethods;
    let h_flat = high_tm_f32.as_slice()?;
    let l_flat = low_tm_f32.as_slice()?;
    let v_flat = volume_tm_f32.as_slice()?;
    let rows = high_tm_f32.shape()[0];
    let cols = high_tm_f32.shape()[1];
    if low_tm_f32.shape() != [rows, cols] || volume_tm_f32.shape() != [rows, cols] {
        return Err(PyValueError::new_err("high/low/volume shapes mismatch"));
    }
    let inner = py.allow_threads(|| {
        let cuda = CudaEmv::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.emv_many_series_one_param_time_major_dev(h_flat, l_flat, v_flat, cols, rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(DeviceArrayF32Py { inner })
}
