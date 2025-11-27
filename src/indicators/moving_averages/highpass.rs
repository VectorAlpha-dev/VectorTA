//! # High-Pass Filter (HP)
//!
//! Decision log: SIMD kept scalar-first due to IIR dependency; AVX512 routes to AVX2; CUDA provided via wrappers; interop exposes CAI v3 and DLPack v1.x with version negotiation. Perf: scalar is baseline; GPU paths target large sweeps.
//!
//! A digital filter that attenuates low-frequency components of the input data,
//! allowing higher-frequency fluctuations to pass through. This helps to remove
//! or reduce slow-moving trends or bias.
//!
//! ## Parameters
//! - **period**: The size of the window (number of data points). Defaults to 48.
//!
//! ## Returns
//! - **`Ok(HighPassOutput)`** on success, containing a `Vec<f64>` matching the input length.
//! - **`Err(HighPassError)`** otherwise.
//!
//! ## Developer Status
//! - SIMD: single-series uses deep unrolling + FMA; true lane SIMD not applicable due to IIR dependence.
//! - AVX2/AVX512: serial kernels with 16× ILP and prefetch; AVX512 routes to AVX2.
//! - Batch: row-specific path precomputes Δx once and reuses across rows.
//! - Streaming update: O(1) with prior state.
//! - Allocation: uses zero-copy helpers; no O(N) output prefill.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, make_uninit_matrix,
};
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
use core::arch::x86_64::*;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaHighpass;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::highpass_wrapper::DeviceArrayF32Highpass;
// For highpass CUDA Python returns, provide a local PyClass with CAI v3 + DLPack
#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy::ndarray::{Array1, Array2};
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray2;
#[cfg(feature = "python")]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct HighPassDeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32Highpass,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl HighPassDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("shape", (self.inner.rows, self.inner.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        d.set_item("data", (self.inner.device_ptr() as usize, false))?;
        // Stream omitted: producing stream is synchronized before returning the handle
        d.set_item("version", 3)?;
        Ok(d)
    }
    fn __dlpack_device__(&self) -> (i32, i32) { (2, self.inner.device_id as i32) }

    // DLPack v1.x with legacy fallback and versioned capsules via shared helper.
    #[pyo3(signature=(stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        use crate::utilities::dlpack_cuda::export_f32_cuda_dlpack_2d;

        // Compute target device id and validate `dl_device` hint if provided.
        let (kdl, alloc_dev) = self.__dlpack_device__(); // (2, device_id)
        if let Some(dev_obj) = dl_device.as_ref() {
            if let Ok((dev_ty, dev_id)) = dev_obj.extract::<(i32, i32)>(py) {
                if dev_ty != kdl || dev_id != alloc_dev {
                    let wants_copy = copy
                        .as_ref()
                        .and_then(|c| c.extract::<bool>(py).ok())
                        .unwrap_or(false);
                    if wants_copy {
                        return Err(PyValueError::new_err(
                            "device copy not implemented for __dlpack__",
                        ));
                    } else {
                        return Err(PyValueError::new_err("dl_device mismatch for __dlpack__"));
                    }
                }
            }
        }
        let _ = stream;

        // Move VRAM handle out of this wrapper; the DLPack capsule owns it afterwards.
        let dummy = DeviceBuffer::from_slice(&[])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Preserve context and device id for the placeholder handle.
        let ctx = self.inner.ctx.clone();
        let device_id = self.inner.device_id;

        let inner = std::mem::replace(
            &mut self.inner,
            DeviceArrayF32Highpass {
                buf: dummy,
                rows: 0,
                cols: 0,
                ctx,
                device_id,
            },
        );

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

impl<'a> AsRef<[f64]> for HighPassInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            HighPassData::Slice(slice) => slice,
            HighPassData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HighPassData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct HighPassParams {
    pub period: Option<usize>,
}
impl Default for HighPassParams {
    fn default() -> Self {
        Self { period: Some(48) }
    }
}

#[derive(Debug, Clone)]
pub struct HighPassOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HighPassInput<'a> {
    pub data: HighPassData<'a>,
    pub params: HighPassParams,
}

impl<'a> HighPassInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: HighPassParams) -> Self {
        Self {
            data: HighPassData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: HighPassParams) -> Self {
        Self {
            data: HighPassData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", HighPassParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(48)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HighPassBuilder {
    period: Option<usize>,
    kernel: Kernel,
}
impl Default for HighPassBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}
impl HighPassBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<HighPassOutput, HighPassError> {
        let p = HighPassParams {
            period: self.period,
        };
        let i = HighPassInput::from_candles(c, "close", p);
        highpass_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<HighPassOutput, HighPassError> {
        let p = HighPassParams {
            period: self.period,
        };
        let i = HighPassInput::from_slice(d, p);
        highpass_with_kernel(&i, self.kernel)
    }
    #[inline(always)]
    pub fn into_stream(self) -> Result<HighPassStream, HighPassError> {
        let p = HighPassParams {
            period: self.period,
        };
        HighPassStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum HighPassError {
    #[error("highpass: Input data slice is empty.")]
    EmptyInputData,
    #[error("highpass: All values are NaN.")]
    AllValuesNaN,
    #[error("highpass: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("highpass: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "highpass: Invalid alpha calculation. cos_val is too close to zero: cos_val = {cos_val}"
    )]
    InvalidAlpha { cos_val: f64 },
    #[error("highpass: Output slice length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("highpass: Invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("highpass: Invalid kernel type for batch operation: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("highpass: dimensions too large to allocate: rows = {rows}, cols = {cols}")]
    DimensionsTooLarge { rows: usize, cols: usize },
}

#[inline]
pub fn highpass(input: &HighPassInput) -> Result<HighPassOutput, HighPassError> {
    highpass_with_kernel(input, Kernel::Auto)
}

#[inline]
fn highpass_into_internal(input: &HighPassInput, out: &mut [f64]) -> Result<(), HighPassError> {
    highpass_with_kernel_into(input, Kernel::Auto, out)
}

/// Write High-Pass outputs into a caller-provided buffer without allocating.
///
/// - Preserves the indicator’s warmup semantics (High-Pass computes from the first
///   valid sample; no NaN warmup prefix is used).
/// - The `out` slice length must exactly match the input length.
/// - Uses `Kernel::Auto` for runtime kernel selection.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn highpass_into(input: &HighPassInput, out: &mut [f64]) -> Result<(), HighPassError> {
    highpass_with_kernel_into(input, Kernel::Auto, out)
}

#[inline(always)]
pub fn highpass_with_kernel(
    input: &HighPassInput,
    kernel: Kernel,
) -> Result<HighPassOutput, HighPassError> {
    let data: &[f64] = match &input.data {
        HighPassData::Candles { candles, source } => source_type(candles, source),
        HighPassData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(HighPassError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HighPassError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    if len <= 2 || period == 0 || period > len {
        return Err(HighPassError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(HighPassError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let k = 1.0;
    let two_pi_k_div = 2.0 * std::f64::consts::PI * k / (period as f64);
    let cos_val = two_pi_k_div.cos();
    if cos_val.abs() < 1e-15 {
        return Err(HighPassError::InvalidAlpha { cos_val });
    }

    // SIMD underperforms for this IIR; prefer scalar on Auto
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    // Highpass writes all values; allocate without warmup prefix using zero-copy helper
    let mut out = alloc_with_nan_prefix(len, 0);
    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => highpass_scalar(data, period, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => highpass_avx2(data, period, &mut out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => highpass_avx512(data, period, &mut out),
            _ => unreachable!(),
        }
    }

    Ok(HighPassOutput { values: out })
}

#[inline(always)]
fn highpass_with_kernel_into(
    input: &HighPassInput,
    kernel: Kernel,
    out: &mut [f64],
) -> Result<(), HighPassError> {
    let data: &[f64] = match &input.data {
        HighPassData::Candles { candles, source } => source_type(candles, source),
        HighPassData::Slice(sl) => sl,
    };

    if data.is_empty() {
        return Err(HighPassError::EmptyInputData);
    }

    // Ensure output buffer is the correct size
    if out.len() != data.len() {
        return Err(HighPassError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HighPassError::AllValuesNaN)?;
    let len = data.len();
    let period = input.get_period();
    if len <= 2 || period == 0 || period > len {
        return Err(HighPassError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(HighPassError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let k = 1.0;
    let two_pi_k_div = 2.0 * std::f64::consts::PI * k / (period as f64);
    let cos_val = two_pi_k_div.cos();
    if cos_val.abs() < 1e-15 {
        return Err(HighPassError::InvalidAlpha { cos_val });
    }

    // SIMD underperforms for this IIR; prefer scalar on Auto
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other,
    };

    // The caller is responsible for initializing the output buffer
    // No need to fill with NaN here as the buffer should be pre-initialized

    unsafe {
        match chosen {
            Kernel::Scalar | Kernel::ScalarBatch => highpass_scalar(data, period, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => highpass_avx2(data, period, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => highpass_avx512(data, period, out),
            _ => unreachable!(),
        }
    }

    Ok(())
}

// Optimized scalar implementation with pointer arithmetic (2× unrolled)
#[inline(always)]
pub unsafe fn highpass_scalar(data: &[f64], period: usize, out: &mut [f64]) {
    use core::f64::consts::PI;

    let n = data.len();
    if n == 0 {
        return;
    }

    let theta = 2.0 * PI / period as f64; // k=1 baked in
    let alpha = 1.0 + ((theta.sin() - 1.0) / theta.cos());
    let c = 1.0 - 0.5 * alpha; // (1 - α/2)
    let oma = 1.0 - alpha; // (1 - α)

    let mut src = data.as_ptr();
    let mut dst = out.as_mut_ptr();

    // seed
    *dst = *src;
    if n == 1 {
        return;
    }

    let mut x_im1 = *src;
    let mut y_im1 = *dst;

    src = src.add(1);
    dst = dst.add(1);

    let mut rem = n - 1;
    while rem >= 2 {
        let x_i = *src;
        let y_i = oma.mul_add(y_im1, c * (x_i - x_im1));
        *dst = y_i;

        let x_ip1 = *src.add(1);
        let y_ip1 = oma.mul_add(y_i, c * (x_ip1 - x_i));
        *dst.add(1) = y_ip1;

        x_im1 = x_ip1;
        y_im1 = y_ip1;
        src = src.add(2);
        dst = dst.add(2);
        rem -= 2;
    }
    if rem == 1 {
        let x_i = *src;
        *dst = oma.mul_add(y_im1, c * (x_i - x_im1));
    }
}

// AVX2 ILP kernel (serial due to recurrence)
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn highpass_avx2(data: &[f64], period: usize, out: &mut [f64]) {
    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
    use core::f64::consts::PI;

    let n = data.len();
    if n == 0 {
        return;
    }

    let theta = 2.0 * PI / period as f64;
    let sin_t = theta.sin();
    let cos_t = theta.cos();
    let alpha = 1.0 + (sin_t - 1.0) / cos_t;

    let c = 1.0 - 0.5 * alpha; // (1-α/2)
    let oma = 1.0 - alpha; // (1-α)

    // seed
    let mut src = data.as_ptr();
    let mut dst = out.as_mut_ptr();
    let mut x_prev = *src;
    let mut y_prev = x_prev;
    *dst = y_prev;

    if n == 1 {
        return;
    }

    src = src.add(1);
    dst = dst.add(1);
    let mut rem = n - 1;

    // 16× unrolled main loop with simple L1 prefetch
    while rem >= 16 {
        _mm_prefetch(src.add(64) as *const i8, _MM_HINT_T0);

        let x0 = *src;
        let y0 = oma.mul_add(y_prev, c * (x0 - x_prev));
        *dst = y0;
        let x1 = *src.add(1);
        let y1 = oma.mul_add(y0, c * (x1 - x0));
        *dst.add(1) = y1;
        let x2 = *src.add(2);
        let y2 = oma.mul_add(y1, c * (x2 - x1));
        *dst.add(2) = y2;
        let x3 = *src.add(3);
        let y3 = oma.mul_add(y2, c * (x3 - x2));
        *dst.add(3) = y3;
        let x4 = *src.add(4);
        let y4 = oma.mul_add(y3, c * (x4 - x3));
        *dst.add(4) = y4;
        let x5 = *src.add(5);
        let y5 = oma.mul_add(y4, c * (x5 - x4));
        *dst.add(5) = y5;
        let x6 = *src.add(6);
        let y6 = oma.mul_add(y5, c * (x6 - x5));
        *dst.add(6) = y6;
        let x7 = *src.add(7);
        let y7 = oma.mul_add(y6, c * (x7 - x6));
        *dst.add(7) = y7;
        let x8 = *src.add(8);
        let y8 = oma.mul_add(y7, c * (x8 - x7));
        *dst.add(8) = y8;
        let x9 = *src.add(9);
        let y9 = oma.mul_add(y8, c * (x9 - x8));
        *dst.add(9) = y9;
        let x10 = *src.add(10);
        let y10 = oma.mul_add(y9, c * (x10 - x9));
        *dst.add(10) = y10;
        let x11 = *src.add(11);
        let y11 = oma.mul_add(y10, c * (x11 - x10));
        *dst.add(11) = y11;
        let x12 = *src.add(12);
        let y12 = oma.mul_add(y11, c * (x12 - x11));
        *dst.add(12) = y12;
        let x13 = *src.add(13);
        let y13 = oma.mul_add(y12, c * (x13 - x12));
        *dst.add(13) = y13;
        let x14 = *src.add(14);
        let y14 = oma.mul_add(y13, c * (x14 - x13));
        *dst.add(14) = y14;
        let x15 = *src.add(15);
        let y15 = oma.mul_add(y14, c * (x15 - x14));
        *dst.add(15) = y15;

        x_prev = x15;
        y_prev = y15;
        src = src.add(16);
        dst = dst.add(16);
        rem -= 16;
    }

    while rem >= 8 {
        let x0 = *src;
        let y0 = oma.mul_add(y_prev, c * (x0 - x_prev));
        *dst = y0;
        let x1 = *src.add(1);
        let y1 = oma.mul_add(y0, c * (x1 - x0));
        *dst.add(1) = y1;
        let x2 = *src.add(2);
        let y2 = oma.mul_add(y1, c * (x2 - x1));
        *dst.add(2) = y2;
        let x3 = *src.add(3);
        let y3 = oma.mul_add(y2, c * (x3 - x2));
        *dst.add(3) = y3;
        let x4 = *src.add(4);
        let y4 = oma.mul_add(y3, c * (x4 - x3));
        *dst.add(4) = y4;
        let x5 = *src.add(5);
        let y5 = oma.mul_add(y4, c * (x5 - x4));
        *dst.add(5) = y5;
        let x6 = *src.add(6);
        let y6 = oma.mul_add(y5, c * (x6 - x5));
        *dst.add(6) = y6;
        let x7 = *src.add(7);
        let y7 = oma.mul_add(y6, c * (x7 - x6));
        *dst.add(7) = y7;

        x_prev = x7;
        y_prev = y7;
        src = src.add(8);
        dst = dst.add(8);
        rem -= 8;
    }

    while rem >= 2 {
        let x0 = *src;
        let y0 = oma.mul_add(y_prev, c * (x0 - x_prev));
        *dst = y0;
        let x1 = *src.add(1);
        let y1 = oma.mul_add(y0, c * (x1 - x0));
        *dst.add(1) = y1;
        x_prev = x1;
        y_prev = y1;
        src = src.add(2);
        dst = dst.add(2);
        rem -= 2;
    }

    if rem == 1 {
        let x0 = *src;
        *dst = oma.mul_add(y_prev, c * (x0 - x_prev));
    }
}

// AVX512 stub and long/short variants
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
pub unsafe fn highpass_avx512(data: &[f64], period: usize, out: &mut [f64]) {
    highpass_avx2(data, period, out)
}

// Batch/Range types and functions
#[derive(Clone, Debug)]
pub struct HighPassBatchRange {
    pub period: (usize, usize, usize),
}
impl Default for HighPassBatchRange {
    fn default() -> Self {
        Self {
            period: (48, 48, 0),
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct HighPassBatchBuilder {
    range: HighPassBatchRange,
    kernel: Kernel,
}
impl HighPassBatchBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }
    pub fn period_range(mut self, start: usize, end: usize, step: usize) -> Self {
        self.range.period = (start, end, step);
        self
    }
    pub fn period_static(mut self, p: usize) -> Self {
        self.range.period = (p, p, 0);
        self
    }
    pub fn apply_slice(self, data: &[f64]) -> Result<HighPassBatchOutput, HighPassError> {
        highpass_batch_with_kernel(data, &self.range, self.kernel)
    }
    pub fn with_default_slice(
        data: &[f64],
        k: Kernel,
    ) -> Result<HighPassBatchOutput, HighPassError> {
        HighPassBatchBuilder::new().kernel(k).apply_slice(data)
    }
    pub fn apply_candles(
        self,
        c: &Candles,
        src: &str,
    ) -> Result<HighPassBatchOutput, HighPassError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }
    pub fn with_default_candles(c: &Candles) -> Result<HighPassBatchOutput, HighPassError> {
        HighPassBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

#[derive(Clone, Debug)]
pub struct HighPassBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HighPassParams>,
    pub rows: usize,
    pub cols: usize,
}
impl HighPassBatchOutput {
    pub fn row_for_params(&self, p: &HighPassParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(48) == p.period.unwrap_or(48))
    }
    pub fn values_for(&self, p: &HighPassParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &HighPassBatchRange) -> Vec<HighPassParams> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            return vec![start];
        }
        if start < end {
            (start..=end).step_by(step).collect()
        } else {
            let mut v: Vec<usize> = (end..=start).step_by(step).collect();
            v.reverse();
            v
        }
    }
    let periods = axis_usize(r.period);
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(HighPassParams { period: Some(p) });
    }
    out
}

#[inline(always)]
pub fn highpass_batch_with_kernel(
    data: &[f64],
    sweep: &HighPassBatchRange,
    k: Kernel,
) -> Result<HighPassBatchOutput, HighPassError> {
    // Row-specific path is serial per row; prefer scalar on Auto
    let kernel = match k {
        Kernel::Auto => Kernel::ScalarBatch,
        other if other.is_batch() => other,
        _ => return Err(HighPassError::InvalidKernelForBatch(k)),
    };
    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    highpass_batch_par_slice(data, sweep, simd)
}

#[inline(always)]
pub fn highpass_batch_slice(
    data: &[f64],
    sweep: &HighPassBatchRange,
    kern: Kernel,
) -> Result<HighPassBatchOutput, HighPassError> {
    highpass_batch_inner(data, sweep, kern, false)
}
#[inline(always)]
pub fn highpass_batch_par_slice(
    data: &[f64],
    sweep: &HighPassBatchRange,
    kern: Kernel,
) -> Result<HighPassBatchOutput, HighPassError> {
    highpass_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn highpass_batch_inner(
    data: &[f64],
    sweep: &HighPassBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<HighPassBatchOutput, HighPassError> {
    // Get combos to calculate dimensions
    let combos = expand_grid(sweep);
    let rows = combos.len();
    let cols = data.len();

    if combos.is_empty() {
        return Err(HighPassError::InvalidRange {
            start: sweep.period.0,
            end: sweep.period.1,
            step: sweep.period.2,
        });
    }
    if data.is_empty() {
        return Err(HighPassError::EmptyInputData);
    }

    // checked rows*cols to prevent overflow before allocations
    let _total = rows
        .checked_mul(cols)
        .ok_or(HighPassError::DimensionsTooLarge { rows, cols })?;

    // Find first valid value
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(HighPassError::AllValuesNaN)?;

    // Allocate uninitialized matrix - no NaN prefixes needed for highpass
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // Use ManuallyDrop to keep capacity when converting
    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    // Delegate to the _into version which contains all the computation logic
    highpass_batch_inner_into(data, sweep, kern, parallel, out)?;

    // Reclaim ownership as Vec<f64>
    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(HighPassBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

// Row functions, all variants just call scalar
#[inline(always)]
pub unsafe fn highpass_row_scalar(data: &[f64], period: usize, out: &mut [f64]) {
    highpass_scalar(data, period, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn highpass_row_avx2(data: &[f64], period: usize, out: &mut [f64]) {
    highpass_avx2(data, period, out)
}
#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn highpass_row_avx512(data: &[f64], period: usize, out: &mut [f64]) {
    highpass_row_avx2(data, period, out)
}

// Streaming
#[derive(Debug, Clone)]
pub struct HighPassStream {
    period: usize,
    alpha: f64,
    one_minus_half_alpha: f64,
    one_minus_alpha: f64,
    prev_data: f64,
    prev_output: f64,
    initialized: bool,
}
impl HighPassStream {
    pub fn try_new(params: HighPassParams) -> Result<Self, HighPassError> {
        let period = params.period.unwrap_or(48);
        if period == 0 {
            return Err(HighPassError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        // θ = 2π/period (k fixed at 1)
        let theta = (2.0 * core::f64::consts::PI) / (period as f64);
        let (sin_val, cos_val) = theta.sin_cos();
        if cos_val.abs() < 1e-15 {
            return Err(HighPassError::InvalidAlpha { cos_val });
        }
        let alpha = 1.0 + (sin_val - 1.0) / cos_val;
        Ok(Self {
            period,
            alpha,
            one_minus_half_alpha: 1.0 - 0.5 * alpha,
            one_minus_alpha: 1.0 - alpha,
            prev_data: f64::NAN,
            prev_output: f64::NAN,
            initialized: false,
        })
    }
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> f64 {
        // Cold seed path kept out of the hot block to help the optimizer.
        #[cold]
        #[inline(never)]
        fn seed(this: &mut HighPassStream, v: f64) -> f64 {
            this.prev_data = v;
            this.prev_output = v;
            this.initialized = true;
            v
        }

        if self.initialized {
            // Hot path: 1 sub + 1 mul + 1 FMA
            // out = (1 - α) * prev_output + (1 - α/2) * (value - prev_data)
            let dx = value - self.prev_data;
            let y = self
                .one_minus_alpha
                .mul_add(self.prev_output, self.one_minus_half_alpha * dx);
            self.prev_data = value;
            self.prev_output = y;
            y
        } else {
            seed(self, value)
        }
    }
}

// Tests: macro structure matches ALMA
#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use proptest::prelude::*;
    use std::error::Error;

    #[test]
    fn test_highpass_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Prepare non-trivial synthetic data (length 512)
        let n = 512usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64;
            let v = (t * 0.07).sin() + (t * 0.013).cos() + 0.001 * t; // mix of signals
            data.push(v);
        }

        // Default params (period=48)
        let input = HighPassInput::from_slice(&data, HighPassParams::default());

        // Baseline via Vec-returning API
        let base = highpass(&input)?.values;

        // Preallocate output and compute via into API
        let mut out = vec![0.0f64; n];
        super::highpass_into(&input, &mut out)?;

        assert_eq!(base.len(), out.len());

        // Equality check: exact for finite values; treat NaN==NaN if any
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b)
        }

        for (i, (&a, &b)) in base.iter().zip(out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(a, b),
                "mismatch at {}: api={}, into={}",
                i,
                a,
                b
            );
        }

        Ok(())
    }

    fn check_highpass_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let default_params = HighPassParams { period: None };
        let input_default = HighPassInput::from_candles(&candles, "close", default_params);
        let output_default = highpass_with_kernel(&input_default, kernel)?;
        assert_eq!(output_default.values.len(), candles.close.len());
        let params_period = HighPassParams { period: Some(36) };
        let input_period = HighPassInput::from_candles(&candles, "hl2", params_period);
        let output_period = highpass_with_kernel(&input_period, kernel)?;
        assert_eq!(output_period.values.len(), candles.close.len());
        Ok(())
    }
    fn check_highpass_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HighPassInput::with_default_candles(&candles);
        let result = highpass_with_kernel(&input, kernel)?;
        let expected_last_five = [
            -265.1027020005024,
            -330.0916060058495,
            -422.7478979710918,
            -261.87532144673423,
            -698.9026088956363,
        ];
        let start = result.values.len().saturating_sub(5);
        let last_five = &result.values[start..];
        for (i, &val) in last_five.iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-6,
                "[{}] Highpass mismatch at {}: expected {}, got {}",
                test_name,
                i,
                expected_last_five[i],
                val
            );
        }
        Ok(())
    }
    fn check_highpass_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let input = HighPassInput::with_default_candles(&candles);
        match input.data {
            HighPassData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Unexpected data variant"),
        }
        let output = highpass_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }
    fn check_highpass_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = HighPassParams { period: Some(0) };
        let input = HighPassInput::from_slice(&input_data, params);
        let result = highpass_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Highpass should fail with zero period",
            test_name
        );
        Ok(())
    }
    fn check_highpass_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = HighPassParams { period: Some(48) };
        let input = HighPassInput::from_slice(&input_data, params);
        let result = highpass_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Highpass should fail with period exceeding length",
            test_name
        );
        Ok(())
    }
    fn check_highpass_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [42.0, 43.0];
        let params = HighPassParams { period: Some(2) };
        let input = HighPassInput::from_slice(&input_data, params);
        let result = highpass_with_kernel(&input, kernel);
        assert!(
            result.is_err(),
            "[{}] Highpass should fail with insufficient data",
            test_name
        );
        Ok(())
    }
    fn check_highpass_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let first_params = HighPassParams { period: Some(36) };
        let first_input = HighPassInput::from_candles(&candles, "close", first_params);
        let first_result = highpass_with_kernel(&first_input, kernel)?;
        let second_params = HighPassParams { period: Some(24) };
        let second_input = HighPassInput::from_slice(&first_result.values, second_params);
        let second_result = highpass_with_kernel(&second_input, kernel)?;
        assert_eq!(second_result.values.len(), first_result.values.len());
        for val in &second_result.values[240..] {
            assert!(!val.is_nan());
        }
        Ok(())
    }
    fn check_highpass_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let params = HighPassParams { period: Some(48) };
        let input = HighPassInput::from_candles(&candles, "close", params);
        let result = highpass_with_kernel(&input, kernel)?;
        for val in &result.values {
            assert!(!val.is_nan());
        }
        Ok(())
    }
    fn check_highpass_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        let period = 48;
        let input = HighPassInput::from_candles(
            &candles,
            "close",
            HighPassParams {
                period: Some(period),
            },
        );
        let batch_output = highpass_with_kernel(&input, kernel)?.values;
        let mut stream = HighPassStream::try_new(HighPassParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            let hp_val = stream.update(price);
            stream_values.push(hp_val);
        }
        assert_eq!(batch_output.len(), stream_values.len());
        for (i, (&b, &s)) in batch_output.iter().zip(stream_values.iter()).enumerate() {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-8,
                "[{}] Highpass streaming mismatch at idx {}: batch={}, stream={}",
                test_name,
                i,
                b,
                s
            );
        }
        Ok(())
    }

    fn check_highpass_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = HighPassInput::from_slice(&empty, HighPassParams::default());
        let res = highpass_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(HighPassError::EmptyInputData)),
            "[{}] expected EmptyInputData",
            test_name
        );
        Ok(())
    }

    fn check_highpass_invalid_alpha(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = HighPassParams { period: Some(4) }; // cos(pi/2) ~ 0
        let input = HighPassInput::from_slice(&data, params);
        let res = highpass_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(HighPassError::InvalidAlpha { .. })),
            "[{}] expected InvalidAlpha",
            test_name
        );
        Ok(())
    }

    // Helper function for comparing floating-point values with ULP tolerance
    fn ulps_diff(a: f64, b: f64) -> u64 {
        if a.is_nan() && b.is_nan() {
            return 0;
        }
        if a.is_nan() || b.is_nan() {
            return u64::MAX;
        }
        if a == b {
            return 0;
        }
        if a.is_infinite() || b.is_infinite() {
            return if a == b { 0 } else { u64::MAX };
        }
        let a_bits = a.to_bits() as i64;
        let b_bits = b.to_bits() as i64;
        (a_bits.wrapping_sub(b_bits)).unsigned_abs()
    }

    fn check_highpass_property(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Enhanced property testing strategy
        let strat = (3usize..=100)
            .prop_filter("avoid invalid alpha", |&p| {
                // Avoid periods that cause cos_val ≈ 0 (InvalidAlpha error)
                let cos_val = (2.0 * std::f64::consts::PI / (p as f64)).cos();
                cos_val.abs() >= 1e-14
            })
            .prop_flat_map(|period| {
                (
                    prop::collection::vec(
                        (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                        (period + 20)..500,
                    ),
                    Just(period),
                )
            });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                // Create input and compute highpass
                let params = HighPassParams {
                    period: Some(period),
                };
                let input = HighPassInput::from_slice(&data, params);
                let HighPassOutput { values: result } =
                    highpass_with_kernel(&input, kernel).unwrap();

                // Property 1: Output length matches input length
                prop_assert_eq!(
                    result.len(),
                    data.len(),
                    "[{}] Output length {} should match input length {}",
                    test_name,
                    result.len(),
                    data.len()
                );

                // Property 2: No NaN values in output (highpass computes from start)
                for (i, &val) in result.iter().enumerate() {
                    prop_assert!(
                        !val.is_nan(),
                        "[{}] Unexpected NaN at index {}",
                        test_name,
                        i
                    );
                }

                // Property 3: All output values are finite
                for (i, &val) in result.iter().enumerate() {
                    prop_assert!(
                        val.is_finite(),
                        "[{}] Expected finite value at index {}, got {}",
                        test_name,
                        i,
                        val
                    );
                }

                // Property 4: DC removal - constant data should converge to near zero
                // (Highpass filters remove DC components)
                // Use non-zero constant to avoid numerical edge cases
                let constant_val = 42.0;
                let constant_data = vec![constant_val; data.len()];
                let constant_input = HighPassInput::from_slice(&constant_data, params);
                let HighPassOutput {
                    values: constant_result,
                } = highpass_with_kernel(&constant_input, kernel).unwrap();

                // After sufficient samples for stabilization, output should be near zero for constant input
                let check_start = (period * 3).min(constant_result.len());
                if check_start < constant_result.len() {
                    for i in check_start..constant_result.len() {
                        let abs_val = constant_result[i].abs();
                        // More lenient tolerance for edge cases
                        prop_assert!(abs_val < 1e-3,
							"[{}] Highpass should remove DC component at index {}, got {} (should be near 0)",
							test_name, i, constant_result[i]);
                    }
                }

                // Property 5: Kernel consistency - all kernels should produce similar results
                if cfg!(all(feature = "nightly-avx", target_arch = "x86_64")) {
                    let scalar_result =
                        highpass_with_kernel(&input, Kernel::Scalar).unwrap().values;
                    for i in 0..result.len() {
                        let diff = (result[i] - scalar_result[i]).abs();
                        let ulps = ulps_diff(result[i], scalar_result[i]);
                        prop_assert!(
                            ulps <= 10 || diff < 1e-9,
                            "[{}] Kernel mismatch at index {}: {} vs {} (diff={}, ulps={})",
                            test_name,
                            i,
                            result[i],
                            scalar_result[i],
                            diff,
                            ulps
                        );
                    }
                }

                // Property 6: IIR formula verification (spot check)
                // out[i] = (1 - α/2) * (data[i] - data[i-1]) + (1 - α) * out[i-1]
                if result.len() >= 10 {
                    let k = 1.0;
                    let two_pi_k_div = 2.0 * std::f64::consts::PI * k / (period as f64);
                    let sin_val = two_pi_k_div.sin();
                    let cos_val = two_pi_k_div.cos();
                    let alpha = 1.0 + (sin_val - 1.0) / cos_val;
                    let one_minus_half_alpha = 1.0 - alpha / 2.0;
                    let one_minus_alpha = 1.0 - alpha;

                    // Check a few points after initial value
                    for i in 5..10.min(result.len()) {
                        let expected = one_minus_half_alpha * data[i]
                            - one_minus_half_alpha * data[i - 1]
                            + one_minus_alpha * result[i - 1];
                        let diff = (result[i] - expected).abs();
                        prop_assert!(
                            diff < 1e-8,
                            "[{}] IIR formula mismatch at index {}: expected {}, got {} (diff={})",
                            test_name,
                            i,
                            expected,
                            result[i],
                            diff
                        );
                    }
                }

                // Property 7: Stability - bounded input produces bounded output
                let data_max = data.iter().fold(f64::NEG_INFINITY, |a, &b| {
                    if b.is_finite() {
                        a.max(b.abs())
                    } else {
                        a
                    }
                });
                if data_max.is_finite() && data_max > 0.0 {
                    for (i, &val) in result.iter().enumerate() {
                        // Highpass filter output magnitude should not exceed a reasonable multiple of input
                        prop_assert!(
                            val.abs() <= data_max * 10.0,
                            "[{}] Output {} at index {} exceeds reasonable bounds for input max {}",
                            test_name,
                            val,
                            i,
                            data_max
                        );
                    }
                }

                // Property 8: Basic signal processing - output changes with input changes
                // Verify the filter responds to input variations (not stuck at zero or constant)
                if data.len() >= 10 {
                    // Check that output varies when input varies
                    let input_variance = {
                        let mean = data.iter().sum::<f64>() / data.len() as f64;
                        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
                    };

                    if input_variance > 1e-10 {
                        // Only check if input has meaningful variance
                        let output_variance = {
                            let mean = result.iter().sum::<f64>() / result.len() as f64;
                            result.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                                / result.len() as f64
                        };

                        // Output should have some variance if input does
                        prop_assert!(
                            output_variance > 0.0,
                            "[{}] Output variance {} should be non-zero when input variance is {}",
                            test_name,
                            output_variance,
                            input_variance
                        );
                    }
                }

                Ok(())
            })
            .unwrap();
        Ok(())
    }

    macro_rules! generate_all_highpass_tests {
        ($($test_fn:ident),*) => {
            paste::paste! {
                $( #[test] fn [<$test_fn _scalar_f64>]() {
                    let _ = $test_fn(stringify!([<$test_fn _scalar_f64>]), Kernel::Scalar);
                })*
                #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
                $(
                    #[test] fn [<$test_fn _avx2_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx2_f64>]), Kernel::Avx2);
                    }
                    #[test] fn [<$test_fn _avx512_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _avx512_f64>]), Kernel::Avx512);
                    }
                )*
            }
        }
    }
    // Check for poison values in single output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_highpass_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations
        let test_cases = vec![
            HighPassParams { period: Some(48) },  // default
            HighPassParams { period: Some(10) },  // small period
            HighPassParams { period: Some(100) }, // large period
            HighPassParams { period: Some(3) },   // very small period
            HighPassParams { period: Some(20) },  // medium period
            HighPassParams { period: Some(60) },  // larger period
            HighPassParams { period: Some(5) },   // another small period
            HighPassParams { period: Some(80) },  // another large period
            HighPassParams { period: None },      // None value (use default)
        ];

        for params in test_cases {
            // Skip period=4 as it causes InvalidAlpha error (cos(pi/2) ~ 0)
            if params.period == Some(4) {
                continue;
            }

            let input = HighPassInput::from_candles(&candles, "close", params);
            let output = highpass_with_kernel(&input, kernel)?;

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
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                         with params period={:?}",
                        test_name, val, bits, i, params.period
                    );
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_highpass_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    generate_all_highpass_tests!(
        check_highpass_partial_params,
        check_highpass_accuracy,
        check_highpass_default_candles,
        check_highpass_zero_period,
        check_highpass_period_exceeds_length,
        check_highpass_very_small_dataset,
        check_highpass_reinput,
        check_highpass_nan_handling,
        check_highpass_streaming,
        check_highpass_empty_input,
        check_highpass_invalid_alpha,
        check_highpass_property,
        check_highpass_no_poison
    );

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);
        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;
        let output = HighPassBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;
        let def = HighPassParams::default();
        let row = output.values_for(&def).expect("default row missing");
        assert_eq!(row.len(), c.close.len());
        let expected = [
            -265.1027020005024,
            -330.0916060058495,
            -422.7478979710918,
            -261.87532144673423,
            -698.9026088956363,
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
    // Check for poison values in batch output - only runs in debug mode
    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        // Test multiple batch configurations with different parameter ranges
        let batch_configs = vec![
            // Original test case
            (10, 30, 10),
            // Edge cases
            (48, 48, 0),    // Single parameter (default)
            (3, 15, 3),     // Small periods
            (50, 100, 25),  // Large periods
            (5, 25, 5),     // Different step
            (20, 80, 20),   // Medium to large
            (7, 21, 7),     // Different small range
            (100, 120, 10), // Very large periods
        ];

        for (p_start, p_end, p_step) in batch_configs {
            // Skip any period that would be exactly 4 (causes InvalidAlpha)
            let periods: Vec<usize> = if p_step == 0 || p_start == p_end {
                vec![p_start]
            } else {
                (p_start..=p_end)
                    .step_by(p_step)
                    .filter(|&p| p != 4)
                    .collect()
            };

            // If all periods would be 4, skip this config
            if periods.is_empty() || (periods.len() == 1 && periods[0] == 4) {
                continue;
            }

            let output = HighPassBatchBuilder::new()
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
						"[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
					);
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
					);
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} \
                         (flat index {}) with params period={:?}",
						test, val, bits, row, col, idx, combo.period
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

/// highpass_batch_inner_into writes directly to the output buffer
#[inline(always)]
fn highpass_batch_inner_into(
    data: &[f64],
    sweep: &HighPassBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<HighPassParams>, HighPassError> {
    let combos = expand_grid(sweep);
    let rows = combos.len();
    let cols = data.len();
    let expected = rows
        .checked_mul(cols)
        .ok_or(HighPassError::DimensionsTooLarge { rows, cols })?;
    if out.len() != expected {
        return Err(HighPassError::OutputLengthMismatch {
            expected,
            got: out.len(),
        });
    }
    let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

    // Validate alpha for all parameter combinations
    for c in &combos {
        let period = c.period.unwrap();
        let k = 1.0;
        let cos_val = (2.0 * std::f64::consts::PI * k / period as f64).cos();
        if cos_val.abs() < 1e-15 {
            return Err(HighPassError::InvalidAlpha { cos_val });
        }
    }

    let rows = combos.len();
    let cols = data.len();

    // Reinterpret output slice as MaybeUninit for row processing
    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };
    // No NaN initialization needed - highpass writes all values

    // Precompute Δx once per series: dx[0]=0, dx[i]=x[i]-x[i-1]
    let mut dx: Vec<f64> = Vec::with_capacity(cols);
    if cols > 0 {
        dx.push(data[0]); // store x0 in slot 0 for convenience; we'll treat dx[0] as x0
        for i in 1..cols {
            dx.push(data[i] - data[i - 1]);
        }
    }

    // ---------- worker that fills one row ----------
    let do_row = |row: usize, dst_mu: &mut [std::mem::MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // Re-interpret this row as &mut [f64]
        let out_row =
            core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        // Compute coefficients as in streaming path
        let theta = 2.0 * std::f64::consts::PI / period as f64;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let alpha = 1.0 + (sin_t - 1.0) / cos_t;
        let c = 1.0 - 0.5 * alpha;
        let oma = 1.0 - alpha;

        // y[0] = x0
        let mut y_prev = dx[0];
        out_row[0] = y_prev;

        // Process i=1..n-1 using shared Δx
        let mut i = 1usize;
        let n = cols;
        // 8× unrolled loop
        while i + 7 < n {
            let d1 = dx[i];
            let y1 = oma.mul_add(y_prev, c * d1);
            out_row[i] = y1;

            let d2 = dx[i + 1];
            let y2 = oma.mul_add(y1, c * d2);
            out_row[i + 1] = y2;

            let d3 = dx[i + 2];
            let y3 = oma.mul_add(y2, c * d3);
            out_row[i + 2] = y3;

            let d4 = dx[i + 3];
            let y4 = oma.mul_add(y3, c * d4);
            out_row[i + 3] = y4;

            let d5 = dx[i + 4];
            let y5 = oma.mul_add(y4, c * d5);
            out_row[i + 4] = y5;

            let d6 = dx[i + 5];
            let y6 = oma.mul_add(y5, c * d6);
            out_row[i + 5] = y6;

            let d7 = dx[i + 6];
            let y7 = oma.mul_add(y6, c * d7);
            out_row[i + 6] = y7;

            let d8 = dx[i + 7];
            let y8 = oma.mul_add(y7, c * d8);
            out_row[i + 7] = y8;

            y_prev = y8;
            i += 8;
        }

        while i + 1 < n {
            let d1 = dx[i];
            let y1 = oma.mul_add(y_prev, c * d1);
            out_row[i] = y1;
            let d2 = dx[i + 1];
            let y2 = oma.mul_add(y1, c * d2);
            out_row[i + 1] = y2;
            y_prev = y2;
            i += 2;
        }

        if i < n {
            let d = dx[i];
            out_row[i] = oma.mul_add(y_prev, c * d);
        }
    };

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

// Python bindings
#[cfg(feature = "python")]
#[pyfunction(name = "highpass")]
#[pyo3(signature = (data, period=48, kernel=None))]
pub fn highpass_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArrayMethods};

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, false)?;

    let params = HighPassParams {
        period: Some(period),
    };
    let hp_input = HighPassInput::from_slice(slice_in, params);

    // Get Vec<f64> from Rust function
    let result_vec: Vec<f64> = py
        .allow_threads(|| highpass_with_kernel(&hp_input, kern).map(|o| o.values))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Zero-copy transfer to NumPy
    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "highpass_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn highpass_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;
    let kern = validate_kernel(kernel, true)?;

    // Validate input like the Rust path does
    if slice_in.is_empty() {
        return Err(PyValueError::new_err(
            "highpass: Input data slice is empty.",
        ));
    }
    if slice_in.iter().all(|x| x.is_nan()) {
        return Err(PyValueError::new_err("highpass: All values are NaN."));
    }

    let sweep = HighPassBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep);
    let rows = combos.len();
    let cols = slice_in.len();

    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("highpass: dimensions too large to allocate"))?;
    let out_arr = unsafe { PyArray1::<f64>::new(py, [total], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let combos = py
        .allow_threads(|| {
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
            highpass_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "highpass_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn highpass_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<HighPassDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = HighPassBatchRange {
        period: period_range,
    };

    let inner = py.allow_threads(|| {
        let cuda =
            CudaHighpass::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.highpass_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(HighPassDeviceArrayF32Py { inner })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "highpass_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn highpass_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<HighPassDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = HighPassParams {
        period: Some(period),
    };

    let inner = py.allow_threads(|| {
        let cuda =
            CudaHighpass::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.highpass_many_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(HighPassDeviceArrayF32Py { inner })
}

#[cfg(feature = "python")]
#[pyclass(name = "HighPassStream")]
pub struct HighPassStreamPy {
    stream: HighPassStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl HighPassStreamPy {
    #[new]
    fn new(period: usize) -> PyResult<Self> {
        let params = HighPassParams {
            period: Some(period),
        };
        let stream =
            HighPassStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(HighPassStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        Some(self.stream.update(value))
    }
}

// ================== WASM Helper Functions ==================

/// Helper function to write directly to output slice - no allocations
#[inline]
pub fn highpass_into_slice(
    dst: &mut [f64],
    input: &HighPassInput,
    kern: Kernel,
) -> Result<(), HighPassError> {
    // Validate input
    let data = input.as_ref();

    if data.is_empty() {
        return Err(HighPassError::EmptyInputData);
    }

    if dst.len() != data.len() {
        return Err(HighPassError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    // Use the existing _into function
    // Note: highpass doesn't have a traditional warmup period - it starts computing from index 0
    highpass_with_kernel_into(input, kern, dst)
}

// ================== WASM Bindings ==================

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use serde::{Deserialize, Serialize};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn highpass_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = HighPassParams {
        period: Some(period),
    };
    let input = HighPassInput::from_slice(data, params);

    // Allocate output buffer once
    let mut output = vec![0.0; data.len()];

    // Compute directly into output buffer
    highpass_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn highpass_alloc(len: usize) -> *mut f64 {
    // Allocate memory for input/output buffer
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Prevent deallocation
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn highpass_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn highpass_into(
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

        // Validate inputs
        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        // Create input
        let params = HighPassParams {
            period: Some(period),
        };
        let input = HighPassInput::from_slice(data, params);

        // Check for aliasing (input and output buffers are the same)
        if in_ptr == out_ptr {
            // Use temporary buffer to avoid corruption during computation
            let mut temp = vec![0.0; len];
            highpass_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results back to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, compute directly into output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            highpass_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

// ================== Batch Processing ==================

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HighPassBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[derive(Serialize, Deserialize)]
pub struct HighPassBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<HighPassParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen(js_name = highpass_batch)]
pub fn highpass_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: HighPassBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = HighPassBatchRange {
        period: config.period_range,
    };

    let output = highpass_batch_with_kernel(data, &sweep, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = HighPassBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// Keep old batch API for compatibility
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn highpass_batch_js(
    data: &[f64],
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = HighPassBatchRange {
        period: (period_start, period_end, period_step),
    };
    match highpass_batch_with_kernel(data, &sweep, Kernel::Auto) {
        Ok(output) => Ok(output.values),
        Err(e) => Err(JsValue::from_str(&format!("HighPass batch error: {}", e))),
    }
}

// ================== Optimized Batch Processing ==================

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn highpass_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str(
            "null pointer passed to highpass_batch_into",
        ));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = HighPassBatchRange {
            period: (period_start, period_end, period_step),
        };

        let combos = expand_grid(&sweep);
        let rows = combos.len();
        let cols = len;

        let out = std::slice::from_raw_parts_mut(out_ptr, rows * cols);

        // Resolve Auto kernel before calling inner_into
        let kernel = detect_best_kernel();
        highpass_batch_inner_into(data, &sweep, kernel, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn highpass_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Vec<f64> {
    let periods: Vec<usize> = if period_step == 0 || period_start == period_end {
        vec![period_start]
    } else {
        (period_start..=period_end).step_by(period_step).collect()
    };

    let mut result = Vec::new();
    for &period in &periods {
        result.push(period as f64);
    }
    result
}
