//! # Exponential Moving Average (EMA)
//!
//! Decision log: SIMD disabled (sequential recurrence; stubs fallback to scalar). CUDA wrappers enabled with
//! typed errors and VRAM headroom checks; Python exposes CAI v3 and DLPack with a context guard. No reference
//! output changes; performance unchanged vs. prior scalar semantics.
//!
//! The Exponential Moving Average (EMA) provides a moving average that
//! places a greater weight and significance on the most recent data points.
//! The EMA reacts faster to recent price changes than the simple moving average (SMA).
//!
//! ## Parameters
//! - **period**: Window size (number of data points, default: 9).
//!
//! ## Inputs
//! - **data**: Time series data as a slice of f64 values or Candles with source selection.
//!
//! ## Returns
//! - **`Ok(EmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
//!   Leading values up to period-1 are NaN during the warmup period.
//! - **`Err(EmaError)`** on invalid input or parameters.
//!
//! ## Developer Notes
//! - **AVX2 kernel**: ❌ Stub only - falls back to scalar implementation
//! - **AVX512 kernel**: ❌ Stub only - falls back to scalar implementation
//! - **Streaming update**: ✅ O(1) complexity – uses precomputed reciprocals + FMA in warmup for fewer divisions
//! - **Memory optimization**: ✅ Uses zero-copy helpers (alloc_with_nan_prefix) for output vectors
//! - **Note**: EMA is inherently sequential (each value depends on the previous), making SIMD parallelization
//!   less beneficial than for window-based indicators. The scalar implementation is already optimal.
//! - **Batch (rows) SIMD**: ❌ Not implemented; EMA row kernels retain a scalar per-row update with unchecked indexing
//!   and fast finite checks. Vertical SIMD across rows may be revisited if typical row-counts justify the complexity.
//!
//! Decision: Streaming kernel uses FMA and precomputed 1/n for warmup; outputs match batch within existing tolerances.

use crate::utilities::data_loader::{source_type, Candles};
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{
    alloc_with_nan_prefix, detect_best_batch_kernel, detect_best_kernel, init_matrix_prefixes,
    make_uninit_matrix,
};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::convert::AsRef;
use std::mem::MaybeUninit;
use thiserror::Error;

#[cfg(feature = "python")]
use crate::utilities::kernel_validation::validate_kernel;
#[cfg(feature = "python")]
use numpy;
#[cfg(feature = "python")]
use numpy::PyUntypedArrayMethods;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaEma;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::context::Context;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::ffi::c_void;
#[cfg(all(feature = "python", feature = "cuda"))]
use std::sync::Arc;

// EMA-specific VRAM-backed Python handle with CAI v3 + DLPack
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct EmaDeviceArrayF32Py {
    pub(crate) buf: Option<DeviceBuffer<f32>>, // moved into DLPack once exported
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) _ctx: Arc<Context>,
    pub(crate) device_id: u32,
}
#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl EmaDeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("shape", (self.rows, self.cols))?;
        d.set_item("typestr", "<f4")?;
        d.set_item(
            "strides",
            (
                self.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let ptr = self
            .buf
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("buffer already exported via __dlpack__"))?
            .as_device_ptr()
            .as_raw() as usize;
        d.set_item("data", (ptr, false))?;
        d.set_item("version", 3)?; // producer sync in wrapper; omit 'stream'
        Ok(d)
    }

    fn __dlpack_device__(&self) -> (i32, i32) { (2, self.device_id as i32) }

    #[pyo3(signature=(stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        _stream: Option<&pyo3::types::PyAny>,
        max_version: Option<&pyo3::types::PyAny>,
        _dl_device: Option<&pyo3::types::PyAny>,
        _copy: Option<&pyo3::types::PyAny>,
    ) -> PyResult<PyObject> {
        use std::os::raw::c_char;

        #[repr(C)]
        struct DLDevice { device_type: i32, device_id: i32 }
        #[repr(C)]
        struct DLDataType { code: u8, bits: u8, lanes: u16 }
        #[repr(C)]
        struct DLTensor {
            data: *mut c_void,
            device: DLDevice,
            ndim: i32,
            dtype: DLDataType,
            shape: *mut i64,
            strides: *mut i64,
            byte_offset: u64,
        }
        #[repr(C)]
        struct DLManagedTensor { dl_tensor: DLTensor, manager_ctx: *mut c_void, deleter: Option<extern "C" fn(*mut DLManagedTensor)> }
        #[repr(C)]
        struct DLVersion { major: i32, minor: i32 }
        #[repr(C)]
        struct DLManagedTensorVersioned { dl_managed_tensor: DLManagedTensor, version: DLVersion }

        struct HolderLegacy {
            managed: DLManagedTensor,
            shape: [i64; 2],
            strides: [i64; 2],
            buf: DeviceBuffer<f32>,
            retained: cust::sys::CUcontext,
            device_id: i32,
        }
        struct HolderV1 {
            managed: DLManagedTensorVersioned,
            shape: [i64; 2],
            strides: [i64; 2],
            buf: DeviceBuffer<f32>,
            retained: cust::sys::CUcontext,
            device_id: i32,
        }

        unsafe extern "C" fn deleter_legacy(p: *mut DLManagedTensor) {
            if p.is_null() { return; }
            let holder = (*p).manager_ctx as *mut HolderLegacy;
            if !holder.is_null() {
                let ctx = (*holder).retained;
                if !ctx.is_null() {
                    let _ = cust::sys::cuCtxPushCurrent(ctx);
                    let dev = (*holder).device_id;
                    drop(Box::from_raw(holder));
                    let mut _out: cust::sys::CUcontext = std::ptr::null_mut();
                    let _ = cust::sys::cuCtxPopCurrent(&mut _out);
                    let _ = cust::sys::cuDevicePrimaryCtxRelease(dev);
                }
            }
            drop(Box::from_raw(p));
        }

        unsafe extern "C" fn deleter_v1(p: *mut DLManagedTensorVersioned) {
            if p.is_null() { return; }
            let holder = (*p).dl_managed_tensor.manager_ctx as *mut HolderV1;
            if !holder.is_null() {
                let ctx = (*holder).retained;
                if !ctx.is_null() {
                    let _ = cust::sys::cuCtxPushCurrent(ctx);
                    let dev = (*holder).device_id;
                    drop(Box::from_raw(holder));
                    let mut _out: cust::sys::CUcontext = std::ptr::null_mut();
                    let _ = cust::sys::cuCtxPopCurrent(&mut _out);
                    let _ = cust::sys::cuDevicePrimaryCtxRelease(dev);
                }
            }
            drop(Box::from_raw(p));
        }

        unsafe extern "C" fn cap_destructor_legacy(capsule: *mut pyo3::ffi::PyObject) {
            let name = b"dltensor\0";
            let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr() as *const c_char) as *mut DLManagedTensor;
            if !ptr.is_null() {
                if let Some(del) = (*ptr).deleter { del(ptr); }
                let used = b"used_dltensor\0";
                pyo3::ffi::PyCapsule_SetName(capsule, used.as_ptr() as *const _);
            }
        }
        unsafe extern "C" fn cap_destructor_v1(capsule: *mut pyo3::ffi::PyObject) {
            let name = b"dltensor_versioned\0";
            let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr() as *const c_char) as *mut DLManagedTensorVersioned;
            if !ptr.is_null() {
                let mt = &mut (*ptr).dl_managed_tensor;
                if let Some(del) = mt.deleter { del(mt); }
                let used = b"used_dltensor_versioned\0";
                pyo3::ffi::PyCapsule_SetName(capsule, used.as_ptr() as *const _);
            }
        }

        // Move buffer into capsule-managed holder
        let buf = self
            .buf
            .take()
            .ok_or_else(|| PyValueError::new_err("__dlpack__ may only be called once"))?;

        let alloc_dev = self.device_id as i32;
        let mut retained: cust::sys::CUcontext = std::ptr::null_mut();
        unsafe { let _ = cust::sys::cuDevicePrimaryCtxRetain(&mut retained, alloc_dev); }

        let rows = self.rows as i64;
        let cols = self.cols as i64;
        let data_ptr: *mut c_void = if self.rows == 0 || self.cols == 0 { std::ptr::null_mut() } else { buf.as_device_ptr().as_raw() as *mut c_void };

        let want_v1 = if let Some(v) = max_version {
            v.getattr("__iter").ok().and_then(|_| v.extract::<(i32,i32)>().ok()).map(|(maj, _)| maj >= 1).unwrap_or(false)
        } else { false };

        if want_v1 {
            let mut holder = Box::new(HolderV1 {
                managed: DLManagedTensorVersioned {
                    dl_managed_tensor: DLManagedTensor {
                        dl_tensor: DLTensor {
                            data: data_ptr,
                            device: DLDevice { device_type: 2, device_id: alloc_dev },
                            ndim: 2,
                            dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
                            shape: std::ptr::null_mut(),
                            strides: std::ptr::null_mut(),
                            byte_offset: 0,
                        },
                        manager_ctx: std::ptr::null_mut(),
                        deleter: Some(|mt| {
                            if !mt.is_null() {
                                let outer = (mt as *mut u8).offset(-(std::mem::size_of::<DLVersion>() as isize)) as *mut DLManagedTensorVersioned;
                                deleter_v1(outer);
                            }
                        }),
                    },
                    version: DLVersion { major: 1, minor: 0 },
                },
                shape: [rows, cols],
                strides: [cols, 1],
                buf,
                retained,
                device_id: alloc_dev,
            });
            holder.managed.dl_managed_tensor.dl_tensor.shape = holder.shape.as_mut_ptr();
            holder.managed.dl_managed_tensor.dl_tensor.strides = holder.strides.as_mut_ptr();
            holder.managed.dl_managed_tensor.manager_ctx = &mut *holder as *mut HolderV1 as *mut c_void;
            let mt_ptr: *mut DLManagedTensorVersioned = &mut holder.managed;
            let _leak = Box::into_raw(holder);
            let name = b"dltensor_versioned\0";
            let cap = unsafe { pyo3::ffi::PyCapsule_New(mt_ptr as *mut c_void, name.as_ptr() as *const c_char, Some(cap_destructor_v1)) };
            if cap.is_null() { return Err(PyValueError::new_err("failed to create DLPack capsule")); }
            Ok(unsafe { PyObject::from_owned_ptr(py, cap) })
        } else {
            let mut holder = Box::new(HolderLegacy {
                managed: DLManagedTensor {
                    dl_tensor: DLTensor {
                        data: data_ptr,
                        device: DLDevice { device_type: 2, device_id: alloc_dev },
                        ndim: 2,
                        dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
                        shape: std::ptr::null_mut(),
                        strides: std::ptr::null_mut(),
                        byte_offset: 0,
                    },
                    manager_ctx: std::ptr::null_mut(),
                    deleter: Some(deleter_legacy),
                },
                shape: [rows, cols],
                strides: [cols, 1],
                buf,
                retained,
                device_id: alloc_dev,
            });
            holder.managed.dl_tensor.shape = holder.shape.as_mut_ptr();
            holder.managed.dl_tensor.strides = holder.strides.as_mut_ptr();
            holder.managed.manager_ctx = &mut *holder as *mut HolderLegacy as *mut c_void;
            let mt_ptr: *mut DLManagedTensor = &mut holder.managed;
            let _leak = Box::into_raw(holder);
            let name = b"dltensor\0";
            let cap = unsafe { pyo3::ffi::PyCapsule_New(mt_ptr as *mut c_void, name.as_ptr() as *const c_char, Some(cap_destructor_legacy)) };
            if cap.is_null() { return Err(PyValueError::new_err("failed to create DLPack capsule")); }
            Ok(unsafe { PyObject::from_owned_ptr(py, cap) })
        }
    }
}
impl<'a> AsRef<[f64]> for EmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            EmaData::Slice(slice) => slice,
            EmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct EmaParams {
    pub period: Option<usize>,
}

impl Default for EmaParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct EmaInput<'a> {
    pub data: EmaData<'a>,
    pub params: EmaParams,
}

impl<'a> EmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: EmaParams) -> Self {
        Self {
            data: EmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: EmaParams) -> Self {
        Self {
            data: EmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", EmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(9)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EmaBuilder {
    period: Option<usize>,
    kernel: Kernel,
}

impl Default for EmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            kernel: Kernel::Auto,
        }
    }
}

impl EmaBuilder {
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
    pub fn apply(self, c: &Candles) -> Result<EmaOutput, EmaError> {
        let p = EmaParams {
            period: self.period,
        };
        let i = EmaInput::from_candles(c, "close", p);
        ema_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<EmaOutput, EmaError> {
        let p = EmaParams {
            period: self.period,
        };
        let i = EmaInput::from_slice(d, p);
        ema_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<EmaStream, EmaError> {
        let p = EmaParams {
            period: self.period,
        };
        EmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum EmaError {
    #[error("ema: Input data slice is empty.")]
    EmptyInputData,
    #[error("ema: All values are NaN.")]
    AllValuesNaN,
    #[error("ema: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("ema: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ema: Output length mismatch: expected = {expected}, got = {got}")]
    OutputLengthMismatch { expected: usize, got: usize },
    #[error("ema: Invalid range: start = {start}, end = {end}, step = {step}")]
    InvalidRange { start: usize, end: usize, step: usize },
    #[error("ema: Invalid kernel for batch API: {0:?}")]
    InvalidKernelForBatch(Kernel),
    #[error("ema: arithmetic overflow while computing {context}")]
    ArithmeticOverflow { context: &'static str },
}

#[inline]
pub fn ema(input: &EmaInput) -> Result<EmaOutput, EmaError> {
    ema_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn ema_prepare<'a>(
    input: &'a EmaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], usize, usize, f64, f64, Kernel), EmaError> {
    let data: &[f64] = input.as_ref();

    let len = data.len();
    if len == 0 {
        return Err(EmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EmaError::AllValuesNaN)?;
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(EmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(EmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    let beta = 1.0 - alpha;
    let chosen = if matches!(kernel, Kernel::Auto) {
        detect_best_kernel()
    } else {
        kernel
    };
    Ok((data, period, first, alpha, beta, chosen))
}

#[inline(always)]
fn ema_compute_into(
    data: &[f64],
    period: usize,
    first: usize,
    alpha: f64,
    beta: f64,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                ema_scalar_into(data, period, first, alpha, beta, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                ema_avx2_into(data, period, first, alpha, beta, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                ema_avx512_into(data, period, first, alpha, beta, out)
            }

            // Fallback to scalar when AVX* is requested but not built/available
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                ema_scalar_into(data, period, first, alpha, beta, out)
            }
            _ => unreachable!(),
        }
    }
}

pub fn ema_with_kernel(input: &EmaInput, kernel: Kernel) -> Result<EmaOutput, EmaError> {
    let (data, period, first, alpha, beta, chosen) = ema_prepare(input, kernel)?;

    let mut out = alloc_with_nan_prefix(data.len(), first);
    ema_compute_into(data, period, first, alpha, beta, chosen, &mut out);

    Ok(EmaOutput { values: out })
}

/// Compute EMA directly into a caller-provided buffer without allocations.
///
/// - Preserves the module's warmup semantics: fills the NaN prefix up to the
///   first finite input value using the same quiet-NaN pattern as `alloc_with_nan_prefix`.
/// - Writes results in-place for the remaining entries via the selected kernel (Auto).
/// - `out.len()` must equal the input length; returns the existing length/period error on mismatch.
#[cfg(not(feature = "wasm"))]
pub fn ema_into(input: &EmaInput, out: &mut [f64]) -> Result<(), EmaError> {
    let (data, period, first, alpha, beta, chosen) = ema_prepare(input, Kernel::Auto)?;

    // Enforce output length parity with input
    if out.len() != data.len() {
        return Err(EmaError::OutputLengthMismatch {
            expected: data.len(),
            got: out.len(),
        });
    }

    // Prefill warmup prefix with the same quiet-NaN pattern used by Vec API
    // alloc_with_nan_prefix writes 0x7ff8_0000_0000_0000 for warmups.
    let warm = first.min(out.len());
    for i in 0..warm {
        out[i] = f64::from_bits(0x7ff8_0000_0000_0000);
    }

    // Compute EMA values into the provided buffer
    ema_compute_into(data, period, first, alpha, beta, chosen, out);

    Ok(())
}

#[inline(always)]
fn is_finite_fast(x: f64) -> bool {
    // True for finite values; false for ±Inf/NaN
    const EXP_MASK: u64 = 0x7ff0_0000_0000_0000;
    (x.to_bits() & EXP_MASK) != EXP_MASK
}

/// Computes EMA directly into a provided output slice, avoiding allocation.
/// The output slice must be the same length as the input data.
#[inline]
pub fn ema_into_slice(dst: &mut [f64], input: &EmaInput, kern: Kernel) -> Result<(), EmaError> {
    let (data, period, first, alpha, beta, chosen) = ema_prepare(input, kern)?;

    // Verify output buffer size matches input
    if dst.len() != data.len() {
        return Err(EmaError::OutputLengthMismatch {
            expected: data.len(),
            got: dst.len(),
        });
    }

    // Compute EMA values directly into dst
    ema_compute_into(data, period, first, alpha, beta, chosen, dst);

    // Fill warmup period with NaN
    for v in &mut dst[..first] {
        *v = f64::NAN;
    }

    Ok(())
}

#[inline(always)]
pub unsafe fn ema_scalar(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut Vec<f64>,
) -> Result<EmaOutput, EmaError> {
    let alpha = 2.0 / (period as f64 + 1.0);
    let beta = 1.0 - alpha;
    ema_scalar_into(data, period, first_val, alpha, beta, out);
    let values = std::mem::take(out);
    Ok(EmaOutput { values })
}

#[inline(always)]
unsafe fn ema_scalar_into(
    data: &[f64],
    period: usize,
    first_val: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    let len = data.len();
    debug_assert_eq!(out.len(), len);

    // Use running mean for the first period samples, like the stream does
    let mut mean = *data.get_unchecked(first_val);
    *out.get_unchecked_mut(first_val) = mean;
    let mut valid_count = 1usize;

    // Running mean phase (indices first_val+1 to first_val+period-1)
    let warmup_end = (first_val + period).min(len);
    for i in (first_val + 1)..warmup_end {
        let x = *data.get_unchecked(i);
        if is_finite_fast(x) {
            valid_count += 1;
            mean = ((valid_count as f64 - 1.0) * mean + x) / valid_count as f64;
            *out.get_unchecked_mut(i) = mean;
        } else {
            // During warmup, skip NaN values and carry forward
            *out.get_unchecked_mut(i) = mean;
        }
    }

    // EMA phase (from first_val+period onwards)
    if warmup_end < len {
        let mut prev = mean;
        for i in warmup_end..len {
            let x = *data.get_unchecked(i);
            if is_finite_fast(x) {
                prev = beta.mul_add(prev, alpha * x);
                *out.get_unchecked_mut(i) = prev;
            } else {
                // Skip NaN values - carry forward previous value
                *out.get_unchecked_mut(i) = prev;
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ema_avx2(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut Vec<f64>,
) -> Result<EmaOutput, EmaError> {
    ema_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ema_avx2_into(
    data: &[f64],
    period: usize,
    first_val: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    ema_scalar_into(data, period, first_val, alpha, beta, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn ema_avx512(
    data: &[f64],
    period: usize,
    first_val: usize,
    out: &mut Vec<f64>,
) -> Result<EmaOutput, EmaError> {
    ema_scalar(data, period, first_val, out)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ema_avx512_into(
    data: &[f64],
    period: usize,
    first_val: usize,
    alpha: f64,
    beta: f64,
    out: &mut [f64],
) {
    ema_scalar_into(data, period, first_val, alpha, beta, out)
}

#[derive(Debug, Clone)]
pub struct EmaStream {
    period: usize,
    alpha: f64,
    beta: f64,
    count: usize,
    mean: f64,
    filled: bool,
    // Precomputed reciprocals 1..=period to remove divisions during warm-up.
    inv: Box<[f64]>,
}

impl EmaStream {
    #[inline]
    pub fn try_new(params: EmaParams) -> Result<Self, EmaError> {
        let period = params.period.unwrap_or(9);
        if period == 0 {
            return Err(EmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        // α = 2/(period+1), β = 1 - α
        let alpha = 2.0 / (period as f64 + 1.0);
        let beta = 1.0 - alpha;

        // Precompute exact reciprocals 1/n for n = 1..=period (warm-up only).
        let mut inv = Vec::with_capacity(period);
        for n in 1..=period {
            inv.push(1.0 / n as f64);
        }

        Ok(Self {
            period,
            alpha,
            beta,
            count: 0,
            mean: f64::NAN,
            filled: false,
            inv: inv.into_boxed_slice(),
        })
    }

    /// O(1) update. Returns None until the stream has seen `period` finite values.
    #[inline(always)]
    pub fn update(&mut self, x: f64) -> Option<f64> {
        // Ignore NaN/±Inf; carry previous state if filled, else remain None
        if !is_finite_fast(x) {
            return if self.filled { Some(self.mean) } else { None };
        }

        self.count += 1;
        let c = self.count;

        if c == 1 {
            // Initialize with first finite sample
            self.mean = x;
        } else if c <= self.period {
            // Warm-up as running mean: mean += (x - mean) / c
            // Use precomputed 1/c and FMA to reduce divisions and rounding.
            let inv = self.inv[c - 1];
            self.mean = (x - self.mean).mul_add(inv, self.mean);
        } else {
            // EMA phase: ema = β*ema + α*x
            self.mean = self.beta.mul_add(self.mean, self.alpha * x);
        }

        if !self.filled && c >= self.period {
            self.filled = true;
        }
        if self.filled {
            Some(self.mean)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct EmaBatchRange {
    pub period: (usize, usize, usize),
}

impl Default for EmaBatchRange {
    fn default() -> Self {
        Self {
            period: (9, 240, 1),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmaBatchBuilder {
    range: EmaBatchRange,
    kernel: Kernel,
}

impl EmaBatchBuilder {
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

    pub fn apply_slice(self, data: &[f64]) -> Result<EmaBatchOutput, EmaError> {
        ema_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<EmaBatchOutput, EmaError> {
        EmaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<EmaBatchOutput, EmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<EmaBatchOutput, EmaError> {
        EmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn ema_batch_with_kernel(
    data: &[f64],
    sweep: &EmaBatchRange,
    k: Kernel,
) -> Result<EmaBatchOutput, EmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(EmaError::InvalidKernelForBatch(k)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    ema_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct EmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl EmaBatchOutput {
    pub fn row_for_params(&self, p: &EmaParams) -> Option<usize> {
        self.combos
            .iter()
            .position(|c| c.period.unwrap_or(9) == p.period.unwrap_or(9))
    }

    pub fn values_for(&self, p: &EmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &EmaBatchRange) -> Result<Vec<EmaParams>, EmaError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        // Zero step or equal bounds => singleton (stable semantics for existing tests)
        if step == 0 || start == end {
            return vec![start];
        }
        let (lo, hi) = if start <= end { (start, end) } else { (end, start) };
        (lo..=hi).step_by(step).collect()
    }

    let periods = axis_usize(r.period);
    if periods.is_empty() {
        return Err(EmaError::InvalidRange { start: r.period.0, end: r.period.1, step: r.period.2 });
    }
    let mut out = Vec::with_capacity(periods.len());
    for &p in &periods {
        out.push(EmaParams { period: Some(p) });
    }
    Ok(out)
}

#[inline(always)]
pub fn ema_batch_slice(
    data: &[f64],
    sweep: &EmaBatchRange,
    kern: Kernel,
) -> Result<EmaBatchOutput, EmaError> {
    ema_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn ema_batch_par_slice(
    data: &[f64],
    sweep: &EmaBatchRange,
    kern: Kernel,
) -> Result<EmaBatchOutput, EmaError> {
    ema_batch_inner(data, sweep, kern, true)
}

#[inline(always)]
fn ema_batch_inner(
    data: &[f64],
    sweep: &EmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<EmaBatchOutput, EmaError> {
    let combos = expand_grid(sweep)?;
    let rows = combos.len();
    let cols = data.len();

    if cols == 0 {
        return Err(EmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EmaError::AllValuesNaN)?;

    // checked rows * cols prior to allocating
    let _total = rows
        .checked_mul(cols)
        .ok_or(EmaError::ArithmeticOverflow { context: "rows*cols" })?;
    let mut buf_mu = make_uninit_matrix(rows, cols);

    // warm prefix per row = first (EMA defines from first onward)
    let warm: Vec<usize> = std::iter::repeat(first).take(rows).collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] =
        unsafe { core::slice::from_raw_parts_mut(guard.as_mut_ptr() as *mut f64, guard.len()) };

    let returned_combos = ema_batch_inner_into(data, sweep, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            guard.as_mut_ptr() as *mut f64,
            guard.len(),
            guard.capacity(),
        )
    };

    Ok(EmaBatchOutput {
        values,
        combos: returned_combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn ema_batch_inner_into(
    data: &[f64],
    sweep: &EmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<EmaParams>, EmaError> {
    // ------------ boiler-plate unchanged -----------------------------------
    let combos = expand_grid(sweep)?;

    if data.is_empty() {
        return Err(EmaError::EmptyInputData);
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(EmaError::AllValuesNaN)?;
    let max_p = combos.iter().map(|c| c.period.unwrap()).max().unwrap();
    if data.len() - first < max_p {
        return Err(EmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();

    // ------------ 1. Convert slice to MaybeUninit for row-kernel operations ----
    let raw = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    // ------------ 2. row-kernel closure on MaybeUninit rows ---------------
    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();

        // cast *this* row slice to &mut [f64] for the kernel
        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match kern {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => ema_row_avx512(data, first, period, dst),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => ema_row_avx2(data, first, period, dst),
            _ => ema_row_scalar(data, first, period, dst),
        }
    };

    // ------------ 3. run rows in parallel or serial ------------------------
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

    Ok(combos)
}

#[inline(always)]
unsafe fn ema_row_scalar(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    let alpha = 2.0 / (period as f64 + 1.0);
    let beta = 1.0 - alpha;

    let len = data.len();

    // Use running mean for the first period samples, like the stream does
    let mut mean = unsafe { *data.get_unchecked(first) };
    unsafe { *out.get_unchecked_mut(first) = mean };
    let mut valid_count = 1usize;

    // Running mean phase (indices first+1 to first+period-1)
    let warmup_end = (first + period).min(len);
    for i in (first + 1)..warmup_end {
        let x = unsafe { *data.get_unchecked(i) };
        if is_finite_fast(x) {
            valid_count += 1;
            mean = ((valid_count as f64 - 1.0) * mean + x) / valid_count as f64;
            unsafe { *out.get_unchecked_mut(i) = mean };
        } else {
            // Skip NaN values like stream does - carry forward previous value
            unsafe { *out.get_unchecked_mut(i) = mean };
        }
    }

    // EMA phase (from first+period onwards)
    if warmup_end < len {
        let mut prev = mean;
        for i in warmup_end..len {
            let x = unsafe { *data.get_unchecked(i) };
            if is_finite_fast(x) {
                prev = beta.mul_add(prev, alpha * x);
                unsafe { *out.get_unchecked_mut(i) = prev };
            } else {
                // Skip NaN values - carry forward previous value
                unsafe { *out.get_unchecked_mut(i) = prev };
            }
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ema_row_avx2(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    ema_row_scalar(data, first, period, out);
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn ema_row_avx512(data: &[f64], first: usize, period: usize, out: &mut [f64]) {
    ema_row_scalar(data, first, period, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    use proptest::prelude::*;
    
    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_ema_into_matches_api() -> Result<(), Box<dyn std::error::Error>> {
        // Build a non-trivial input with a small NaN prefix and varying values
        let mut data = Vec::with_capacity(256);
        for _ in 0..5 {
            data.push(f64::NAN);
        }
        for i in 0..251 {
            let x = (i as f64).sin() * 3.14159 + 100.0 + ((i % 7) as f64) * 0.01;
            data.push(x);
        }

        let input = EmaInput::from_slice(&data, EmaParams::default());
        let baseline = ema(&input)?.values;

        let mut out = vec![0.0; data.len()];
        ema_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());
        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || (a - b).abs() <= 1e-12
        }
        for (i, (&a, &b)) in baseline.iter().zip(out.iter()).enumerate() {
            assert!(
                eq_or_both_nan(a, b),
                "mismatch at index {}: api={} into={}",
                i,
                a,
                b
            );
        }
        Ok(())
    }

    fn check_ema_partial_params(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = EmaParams { period: None };
        let input = EmaInput::from_candles(&candles, "close", default_params);
        let output = ema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ema_accuracy(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EmaInput::from_candles(&candles, "close", EmaParams::default());
        let result = ema_with_kernel(&input, kernel)?;
        let expected_last_five = [59302.2, 59277.9, 59230.2, 59215.1, 59103.1];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-1,
                "[{}] EMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_ema_default_candles(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EmaInput::with_default_candles(&candles);
        match input.data {
            EmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected EmaData::Candles"),
        }
        let output = ema_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());
        Ok(())
    }

    fn check_ema_zero_period(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = EmaParams { period: Some(0) };
        let input = EmaInput::from_slice(&input_data, params);
        let res = ema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_ema_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = EmaParams { period: Some(10) };
        let input = EmaInput::from_slice(&data_small, params);
        let res = ema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_ema_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = EmaParams { period: Some(9) };
        let input = EmaInput::from_slice(&single_point, params);
        let res = ema_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] EMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_ema_empty_input(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = EmaInput::from_slice(&empty, EmaParams::default());
        let res = ema_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(EmaError::EmptyInputData)),
            "[{}] EMA should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_ema_reinput(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = EmaParams { period: Some(9) };
        let first_input = EmaInput::from_candles(&candles, "close", first_params);
        let first_result = ema_with_kernel(&first_input, kernel)?;

        let second_params = EmaParams { period: Some(5) };
        let second_input = EmaInput::from_slice(&first_result.values, second_params);
        let second_result = ema_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for (i, &val) in second_result.values[240..].iter().enumerate() {
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

    fn check_ema_nan_handling(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = EmaInput::from_candles(&candles, "close", EmaParams { period: Some(9) });
        let res = ema_with_kernel(&input, kernel)?;
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

    fn check_ema_streaming(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 9;
        let warm_up = 240;

        let input = EmaInput::from_candles(
            &candles,
            "close",
            EmaParams {
                period: Some(period),
            },
        );
        let batch_output = ema_with_kernel(&input, kernel)?.values;

        let mut stream = EmaStream::try_new(EmaParams {
            period: Some(period),
        })?;
        let mut stream_values = Vec::with_capacity(candles.close.len());

        // Stream now returns None until warmup is filled, so we need to track this differently
        for (i, &price) in candles.close.iter().enumerate() {
            let stream_val = stream.update(price);
            // Before warmup period, stream returns None, batch has actual values
            if i < period - 1 {
                assert!(
                    stream_val.is_none(),
                    "[{}] Stream should return None during warmup at idx {}",
                    test_name,
                    i
                );
                stream_values.push(f64::NAN);
            } else {
                stream_values.push(stream_val.unwrap_or(f64::NAN));
            }
        }

        assert_eq!(batch_output.len(), stream_values.len());

        // Compare after warmup period when both have valid values
        for (i, (&b, &s)) in batch_output
            .iter()
            .zip(&stream_values)
            .enumerate()
            .skip(warm_up)
        {
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let diff = (b - s).abs();
            assert!(
                diff < 1e-9,
                "[{}] EMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
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
    fn check_ema_no_poison(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        // Test with multiple parameter combinations to better catch uninitialized memory bugs
        let test_periods = vec![2, 5, 9, 14, 20, 50, 100, 200];
        let test_sources = vec!["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"];

        for period in &test_periods {
            for source in &test_sources {
                let input = EmaInput::from_candles(
                    &candles,
                    source,
                    EmaParams {
                        period: Some(*period),
                    },
                );
                let output = ema_with_kernel(&input, kernel)?;

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
                            "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} with period={}, source={}",
                            test_name, val, bits, i, period, source
                        );
                    }

                    // Check for init_matrix_prefixes poison (0x22222222_22222222)
                    if bits == 0x22222222_22222222 {
                        panic!(
                            "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} with period={}, source={}",
                            test_name, val, bits, i, period, source
                        );
                    }

                    // Check for make_uninit_matrix poison (0x33333333_33333333)
                    if bits == 0x33333333_33333333 {
                        panic!(
                            "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} with period={}, source={}",
                            test_name, val, bits, i, period, source
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // Release mode stub - does nothing
    #[cfg(not(debug_assertions))]
    fn check_ema_no_poison(
        _test_name: &str,
        _kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn check_ema_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        // Enhanced property testing strategy
        let strat = (1usize..=100).prop_flat_map(|period| {
            (
                // Generate data with length >= period + warmup buffer
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period + 10..400,
                ),
                Just(period),
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period)| {
                let params = EmaParams {
                    period: Some(period),
                };
                let input = EmaInput::from_slice(&data, params);

                // Get output from the kernel being tested
                let EmaOutput { values: out } = ema_with_kernel(&input, kernel).unwrap();

                // Get reference output from scalar kernel for comparison
                let EmaOutput { values: ref_out } =
                    ema_with_kernel(&input, Kernel::Scalar).unwrap();

                // EMA specific alpha/beta for validation
                let alpha = 2.0 / (period as f64 + 1.0);
                let beta = 1.0 - alpha;

                // Find first non-NaN value position
                let first_valid = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

                for i in 0..data.len() {
                    let y = out[i];
                    let r = ref_out[i];

                    // Test 1: Warmup period should be NaN
                    if i < first_valid {
                        prop_assert!(
                            y.is_nan(),
                            "[{}] Expected NaN during warmup at idx {}, got {}",
                            test_name,
                            i,
                            y
                        );
                        continue;
                    }

                    // Test 2: Values should be within min/max bounds of all seen data
                    if i >= first_valid {
                        let window = &data[first_valid..=i];
                        let lo = window
                            .iter()
                            .cloned()
                            .filter(|x| x.is_finite())
                            .fold(f64::INFINITY, f64::min);
                        let hi = window
                            .iter()
                            .cloned()
                            .filter(|x| x.is_finite())
                            .fold(f64::NEG_INFINITY, f64::max);

                        if !y.is_nan() && lo.is_finite() && hi.is_finite() {
                            prop_assert!(
                                y >= lo - 1e-9 && y <= hi + 1e-9,
                                "[{}] idx {}: {} not in [{}, {}]",
                                test_name,
                                i,
                                y,
                                lo,
                                hi
                            );
                        }
                    }

                    // Test 3: Special case - period=1 should equal input
                    if period == 1 && i >= first_valid && data[i].is_finite() {
                        // For period=1, alpha=2/2=1, so EMA = current value
                        prop_assert!(
                            (y - data[i]).abs() <= 1e-10,
                            "[{}] Period=1 mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            y,
                            data[i]
                        );
                    }

                    // Test 4: Constant data should converge to that constant
                    if i >= first_valid + period {
                        let window_start = i.saturating_sub(period);
                        let window = &data[window_start..=i];
                        if window
                            .iter()
                            .all(|&x| (x - data[window_start]).abs() < 1e-10)
                        {
                            let expected = data[window_start];
                            prop_assert!(
                                (y - expected).abs() <= 1e-6,
                                "[{}] Constant data convergence failed at idx {}: {} vs {}",
                                test_name,
                                i,
                                y,
                                expected
                            );
                        }
                    }

                    // Test 5: Kernel consistency - compare with scalar reference
                    if !y.is_finite() || !r.is_finite() {
                        // Both should be NaN or infinite in the same way
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "[{}] NaN/infinite mismatch at idx {}: {} vs {}",
                            test_name,
                            i,
                            y,
                            r
                        );
                    } else {
                        // For finite values, check they are close
                        let abs_diff = (y - r).abs();
                        let rel_diff = if r.abs() > 1e-10 {
                            abs_diff / r.abs()
                        } else {
                            abs_diff
                        };

                        prop_assert!(
                            abs_diff <= 1e-9 || rel_diff <= 1e-9,
                            "[{}] Kernel mismatch at idx {}: {} vs {} (abs_diff={}, rel_diff={})",
                            test_name,
                            i,
                            y,
                            r,
                            abs_diff,
                            rel_diff
                        );
                    }

                    // Test 6: EMA recursive property (only after warmup period)
                    // During warmup (first_valid to first_valid+period-1), we use running mean
                    // After warmup (first_valid+period onwards), we use EMA formula
                    if i >= first_valid + period
                        && y.is_finite()
                        && out[i - 1].is_finite()
                        && data[i].is_finite()
                    {
                        let expected_ema = alpha * data[i] + beta * out[i - 1];
                        let diff = (y - expected_ema).abs();

                        // Allow small numerical error accumulation
                        prop_assert!(
                            diff <= 1e-9 * ((i - first_valid) as f64).max(1.0),
                            "[{}] EMA recursive property failed at idx {}: {} vs {} (diff={})",
                            test_name,
                            i,
                            y,
                            expected_ema,
                            diff
                        );
                    }

                    // Test 7: EMA value range after warmup
                    // After sufficient warmup, EMA should be within historical bounds
                    if i >= first_valid + period * 2 {
                        // Look at all historical data from first_valid to current
                        let historical = &data[first_valid..=i];
                        let hist_min = historical
                            .iter()
                            .filter(|x| x.is_finite())
                            .fold(f64::INFINITY, |a, &b| a.min(b));
                        let hist_max = historical
                            .iter()
                            .filter(|x| x.is_finite())
                            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                        if hist_min.is_finite() && hist_max.is_finite() && y.is_finite() {
                            prop_assert!(
                                y >= hist_min - 1e-6 && y <= hist_max + 1e-6,
                                "[{}] EMA outside historical bounds at idx {}: {} not in [{}, {}]",
                                test_name,
                                i,
                                y,
                                hist_min,
                                hist_max
                            );
                        }
                    }
                }

                // Test 8: Verify first valid output matches first valid input
                if first_valid < data.len() && out[first_valid].is_finite() {
                    prop_assert!(
                        (out[first_valid] - data[first_valid]).abs() <= 1e-10,
                        "[{}] First valid output should equal first valid input: {} vs {}",
                        test_name,
                        out[first_valid],
                        data[first_valid]
                    );
                }

                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_ema_tests {
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

    generate_all_ema_tests!(
        check_ema_partial_params,
        check_ema_accuracy,
        check_ema_default_candles,
        check_ema_zero_period,
        check_ema_period_exceeds_length,
        check_ema_very_small_dataset,
        check_ema_empty_input,
        check_ema_reinput,
        check_ema_nan_handling,
        check_ema_streaming,
        check_ema_property,
        check_ema_no_poison
    );

    fn check_batch_default_row(
        test: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = EmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = EmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [59302.2, 59277.9, 59230.2, 59215.1, 59103.1];
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

        // Test batch with multiple parameter combinations to better catch uninitialized memory bugs
        let test_sources = vec!["open", "high", "low", "close", "hl2", "hlc3", "ohlc4"];

        for source in &test_sources {
            // Test with comprehensive period ranges
            let output = EmaBatchBuilder::new()
                .kernel(kernel)
                .period_range(2, 200, 3) // Wide range: 2 to 200 with step 3
                .apply_candles(&c, source)?;

            // Check every value in the entire batch matrix for poison patterns
            for (idx, &val) in output.values.iter().enumerate() {
                // Skip NaN values as they're expected in warmup periods
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;

                // Check for alloc_with_nan_prefix poison (0x11111111_11111111)
                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }

                // Check for init_matrix_prefixes poison (0x22222222_22222222)
                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }

                // Check for make_uninit_matrix poison (0x33333333_33333333)
                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at row {} col {} (flat index {}) with source={}",
                        test, val, bits, row, col, idx, source
                    );
                }
            }
        }

        // Also test edge cases with very small and very large periods
        let edge_case_ranges = vec![(2, 5, 1), (190, 200, 2), (50, 100, 10)];
        for (start, end, step) in edge_case_ranges {
            let output = EmaBatchBuilder::new()
                .kernel(kernel)
                .period_range(start, end, step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();
                let row = idx / output.cols;
                let col = idx % output.cols;

                if bits == 0x11111111_11111111
                    || bits == 0x22222222_22222222
                    || bits == 0x33333333_33333333
                {
                    panic!(
						"[{}] Found poison value {} (0x{:016X}) at row {} col {} with range ({},{},{})",
						test, val, bits, row, col, start, end, step
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

    // Test that batch and stream produce identical results
    #[test]
    fn test_batch_stream_consistency() -> Result<(), Box<dyn std::error::Error>> {
        // Test data with NaN values mid-series
        let test_data = vec![
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            f64::NAN, // Test NaN handling
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            f64::NAN,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
        ];

        let period = 5;

        // Get batch output
        let params = EmaParams {
            period: Some(period),
        };
        let input = EmaInput::from_slice(&test_data, params.clone());
        let batch_output = ema(&input)?;

        // Get stream output
        let mut stream = EmaStream::try_new(params)?;
        let mut stream_output = Vec::new();
        for &val in &test_data {
            let result = stream.update(val);
            // Stream returns None during warmup, batch has values
            // But after warmup they should match
            stream_output.push(result.unwrap_or(f64::NAN));
        }

        // Check consistency after warmup period
        for i in period..test_data.len() {
            let batch_val = batch_output.values[i];
            let stream_val = stream_output[i];

            // Both should handle NaN the same way
            if batch_val.is_finite() && stream_val.is_finite() {
                let diff = (batch_val - stream_val).abs();
                assert!(
                    diff < 1e-10,
                    "Batch/Stream mismatch at index {}: batch={}, stream={}, diff={}",
                    i,
                    batch_val,
                    stream_val,
                    diff
                );
            } else {
                assert_eq!(
                    batch_val.is_nan(),
                    stream_val.is_nan(),
                    "Batch/Stream NaN mismatch at index {}: batch={}, stream={}",
                    i,
                    batch_val,
                    stream_val
                );
            }
        }

        // Also test early warmup consistency - both should use running mean
        for i in 0..period.min(test_data.len()) {
            if test_data[i].is_finite() {
                // During warmup, stream returns NaN but internally tracks running mean
                // Batch should also be computing running mean during this phase
                // We can't directly compare since stream returns None, but we can check
                // that batch values are reasonable (not just the first value repeated)
                if i > 0 && batch_output.values[i].is_finite() {
                    // Should not just be the first value
                    assert!(
                        (batch_output.values[i] - test_data[0]).abs() > 1e-10 || i == 0,
                        "Batch should use running mean during warmup, not just first value"
                    );
                }
            }
        }

        Ok(())
    }
}

// ====== PYTHON BINDINGS ======

#[cfg(feature = "python")]
#[pyfunction(name = "ema")]
#[pyo3(signature = (data, period, kernel=None))]
pub fn ema_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let kern = validate_kernel(kernel, false)?;

    let params = EmaParams {
        period: Some(period),
    };
    // Zero-copy for contiguous inputs; minimal copy for non-contiguous views (e.g., column slices)
    let result_vec: Vec<f64> = if let Ok(slice_in) = data.as_slice() {
        let ema_in = EmaInput::from_slice(slice_in, params);
        py.allow_threads(|| ema_with_kernel(&ema_in, kern).map(|o| o.values))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    } else {
        let owned = data.as_array().to_owned();
        let slice_in = owned
            .as_slice()
            .expect("owned numpy array should be contiguous");
        let ema_in = EmaInput::from_slice(slice_in, params);
        py.allow_threads(|| ema_with_kernel(&ema_in, kern).map(|o| o.values))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    };

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyfunction(name = "ema_batch")]
#[pyo3(signature = (data, period_range, kernel=None))]
pub fn ema_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = EmaBatchRange {
        period: period_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

    // Initialize NaN prefixes before computation
    let first = slice_in.iter().position(|x| !x.is_nan()).unwrap_or(0);
    for r in 0..rows {
        let row_start = r * cols;
        for i in 0..first {
            slice_out[row_start + i] = f64::NAN;
        }
    }

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
                _ => unreachable!(),
            };
            ema_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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

    Ok(dict.into())
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ema_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range=(9, 9, 0), device_id=0))]
pub fn ema_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<EmaDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in = data_f32.as_slice()?;
    let sweep = EmaBatchRange {
        period: period_range,
    };

    let (buf, rows, cols, ctx, dev) = py
        .allow_threads(|| {
            let cuda = CudaEma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let handle = cuda
                .ema_batch_dev(slice_in, &sweep)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let ctx = cuda.context_arc();
            let dev = cuda.device_id();
            Ok::<_, PyErr>((handle.buf, handle.rows, handle.cols, ctx, dev))
        })?;

    Ok(EmaDeviceArrayF32Py { buf: Some(buf), rows, cols, _ctx: ctx, device_id: dev })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "ema_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn ema_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<EmaDeviceArrayF32Py> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    if period == 0 {
        return Err(PyValueError::new_err("period must be positive"));
    }

    let flat = data_tm_f32.as_slice()?;
    let shape = data_tm_f32.shape();
    let series_len = shape[0];
    let num_series = shape[1];
    let params = EmaParams {
        period: Some(period),
    };

    let (buf, rows, cols, ctx, dev) = py
        .allow_threads(|| {
            let cuda = CudaEma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let handle = cuda
                .ema_many_series_one_param_time_major_dev(flat, num_series, series_len, &params)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let ctx = cuda.context_arc();
            let dev = cuda.device_id();
            Ok::<_, PyErr>((handle.buf, handle.rows, handle.cols, ctx, dev))
        })?;

    Ok(EmaDeviceArrayF32Py { buf: Some(buf), rows, cols, _ctx: ctx, device_id: dev })
}

#[cfg(feature = "python")]
#[pyclass(name = "EmaStream")]
pub struct EmaStreamPy {
    inner: EmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl EmaStreamPy {
    #[new]
    pub fn new(period: usize) -> PyResult<Self> {
        let params = EmaParams {
            period: Some(period),
        };
        let inner = EmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.inner.update(value).unwrap_or(f64::NAN)
    }
}

// ====== WASM BINDINGS ======

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ema_js(data: &[f64], period: usize) -> Result<Vec<f64>, JsValue> {
    let params = EmaParams {
        period: Some(period),
    };
    let input = EmaInput::from_slice(data, params);

    // Allocate output buffer once
    let mut output = vec![0.0; data.len()];

    // Compute directly into output buffer
    ema_into_slice(&mut output, &input, Kernel::Auto)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EmaBatchConfig {
    pub period_range: (usize, usize, usize),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct EmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<EmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = ema_batch)]
pub fn ema_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: EmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = EmaBatchRange {
        period: config.period_range,
    };

    let output = ema_batch_inner(data, &sweep, Kernel::Auto, false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = EmaBatchJsOutput {
        values: output.values,
        combos: output.combos,
        rows: output.rows,
        cols: output.cols,
    };

    serde_wasm_bindgen::to_value(&js_output)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ema_batch_metadata_js(
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<Vec<f64>, JsValue> {
    let sweep = EmaBatchRange {
        period: (period_start, period_end, period_step),
    };

    let combos = expand_grid(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut metadata = Vec::with_capacity(combos.len());

    for combo in combos {
        metadata.push(combo.period.unwrap() as f64);
    }

    Ok(metadata)
}

// ================== Zero-Copy WASM Functions ==================

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ema_alloc(len: usize) -> *mut f64 {
    // Allocate memory for input/output buffer
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Prevent deallocation
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ema_free(ptr: *mut f64, len: usize) {
    // Free allocated memory
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ema_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
) -> Result<(), JsValue> {
    // Check for null pointers
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to ema_into"));
    }

    unsafe {
        // Create slice from pointer
        let data = std::slice::from_raw_parts(in_ptr, len);

        // Validate inputs
        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        // Calculate EMA
        let params = EmaParams {
            period: Some(period),
        };
        let input = EmaInput::from_slice(data, params);

        // Check for aliasing (input and output buffers are the same)
        if in_ptr == out_ptr {
            // Use temporary buffer to avoid corruption during sliding window computation
            let mut temp = vec![0.0; len];
            ema_into_slice(&mut temp, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Copy results back to output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            // No aliasing, compute directly into output
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            ema_into_slice(out, &input, Kernel::Auto)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn ema_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to ema_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = EmaBatchRange {
            period: (period_start, period_end, period_step),
        };

    let combos = expand_grid(&sweep)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rows = combos.len();
    let cols = len;
        let elems = rows
            .checked_mul(cols)
            .ok_or(JsValue::from_str("overflow rows*cols"))?;
        let out = std::slice::from_raw_parts_mut(out_ptr, elems);

        // Initialize NaN prefixes before computation
        let first = data
            .iter()
            .position(|x| !x.is_nan())
            .ok_or(JsValue::from_str("All NaN"))?;
        for r in 0..rows {
            let s = r * cols;
            out[s..s + first].fill(f64::NAN);
        }

        // Use optimized batch processing
        ema_batch_inner_into(data, &sweep, Kernel::Auto, false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
