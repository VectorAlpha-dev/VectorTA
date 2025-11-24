#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;
#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
/// # ALMA - Arnaud Legoux Moving Average
///
/// A Gaussian-weighted moving average designed to reduce lag while maintaining smoothness.
/// Uses a shifted Gaussian distribution to weight historical data points.
///
/// ## Parameters
/// - `period`: Window size for the moving average (default: 9)
/// - `offset`: Controls the position of the Gaussian peak, 0.0-1.0 (default: 0.85)
/// - `sigma`: Controls the width of the Gaussian distribution (default: 6.0)
///
/// ## Returns
/// Vector of ALMA values with NaN prefix for warmup period (period - 1 values)
///
/// ## Developer Notes
/// - **AVX2 kernel**: Fully implemented - processes 4 values per iteration
/// - **AVX512 kernel**: Fully implemented with optimized short (<= 32) and long (> 32) period variants
/// - **Streaming update**: Exact O(period) dot-product per update. Optimized with a mirrored ring
///   buffer to make the active window contiguous and enable SIMD dot products.
/// - **Decision**: Streaming uses a mirrored buffer + AVX2/AVX512 contiguous dot for speed; scalar
///   path remains safe and exact. Outputs match batch ALMA (tests unchanged).
/// - **Decision log**: SIMD enabled (AVX2/AVX512) and fastest by >5% at 100k; CUDA wrapper present
///   and returns VRAM handles; Python interop provides CAI v3 + DLPack. Numerical outputs unchanged.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray1};
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyList};

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyclass(module = "ta_indicators.cuda", unsendable)]
pub struct DeviceArrayF32Py {
    pub(crate) inner: DeviceArrayF32,
    // Optional context + device id to keep primary context alive for VRAM frees
    pub(crate) _ctx: Option<std::sync::Arc<cust::context::Context>>, // kept for lifetime
    pub(crate) device_id: Option<u32>,
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pymethods]
impl DeviceArrayF32Py {
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = &self.inner;
        let d = PyDict::new(py);
        // shape: (rows, cols)
        d.set_item("shape", (inner.rows, inner.cols))?;
        // typestr: little-endian float32
        d.set_item("typestr", "<f4")?;
        // Explicit strides for row-major FP32: (row stride in bytes, item stride in bytes)
        d.set_item(
            "strides",
            (
                inner.cols * std::mem::size_of::<f32>(),
                std::mem::size_of::<f32>(),
            ),
        )?;
        let size = inner.rows.saturating_mul(inner.cols);
        let ptr = if size == 0 { 0usize } else { inner.device_ptr() as usize };
        d.set_item("data", (ptr, false))?;
        // Stream is omitted because producing kernels synchronize before returning
        // the VRAM handle; consumers need no additional synchronization per CAI v3.
        d.set_item("version", 3)?;
        Ok(d)
    }
    fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        // Return allocation device id (kDLCUDA = 2)
        unsafe {
            use cust::sys::{cuPointerGetAttribute, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL};
            let mut dev_ordinal: i32 = -1;
            let mut ptr = self.inner.device_ptr() as *mut std::ffi::c_void;
            let res = cuPointerGetAttribute(
                &mut dev_ordinal as *mut _ as *mut std::ffi::c_void,
                CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                ptr,
            );
            if res == cust::sys::CUresult::CUDA_SUCCESS {
                return Ok((2, dev_ordinal));
            }
            // Fallback to current context if attribute query fails
            let mut dev: i32 = 0;
            let _ = cust::sys::cuCtxGetDevice(&mut dev);
            Ok((2, dev))
        }
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<(i32, i32)>,
        _dl_device: Option<(i32, i32)>,
        _copy: Option<bool>,
    ) -> PyResult<PyObject> {
        use std::os::raw::c_char;
        use std::ptr::{null_mut};

        #[repr(C)]
        struct DLDataType { code: u8, bits: u8, lanes: u16 }
        #[repr(C)]
        struct DLDevice { device_type: i32, device_id: i32 }
        #[repr(C)]
        struct DLTensor {
            data: *mut std::ffi::c_void,
            device: DLDevice,
            ndim: i32,
            dtype: DLDataType,
            shape: *mut i64,
            strides: *mut i64,
            byte_offset: u64,
        }
        #[repr(C)]
        struct DLPackVersion { major: u32, minor: u32 }
        #[repr(C)]
        struct DLManagedTensorVersioned {
            version: DLPackVersion,
            manager_ctx: *mut std::ffi::c_void,
            deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
            flags: u64,
            dl_tensor: DLTensor,
        }
        #[repr(C)]
        struct DLManagedTensor {
            dl_tensor: DLTensor,
            manager_ctx: *mut std::ffi::c_void,
            deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
        }

        // Make capsule name and struct choice based on negotiation.
        let use_versioned = max_version.map(|(maj, _)| maj >= 1).unwrap_or(false);

        // Ignore stream parameter: producing side is synchronized (no pending work).
        let _ = stream;

        // Move the VRAM buffer into the capsule manager so consumer owns it.
        let dummy = DeviceBuffer::from_slice(&[])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = std::mem::replace(
            &mut self.inner,
            DeviceArrayF32 { buf: dummy, rows: 0, cols: 0 },
        );

        // Determine allocation device and retain its primary context for safe free.
        let (dev_type, dev_id) = self.__dlpack_device__()?;
        debug_assert_eq!(dev_type, 2); // kDLCUDA
        let mut retained: cust::sys::CUcontext = std::ptr::null_mut();
        unsafe {
            let dev = dev_id as i32;
            let _ = cust::sys::cuDevicePrimaryCtxRetain(&mut retained as *mut _, dev);
        }

        // Build common tensor state
        let rows = inner.rows as i64;
        let cols = inner.cols as i64;
        let size = (rows as usize).saturating_mul(cols as usize);
        let data_ptr = if size == 0 { std::ptr::null_mut() } else { inner.buf.as_device_ptr().as_raw() as *mut std::ffi::c_void };
        let mut shape = Box::new([rows, cols]);
        let mut strides = Box::new([cols, 1]); // elements

        unsafe extern "C" fn deleter_v1(p: *mut DLManagedTensorVersioned) {
            if p.is_null() { return; }
            // Recover manager
            let mgr = (*p).manager_ctx as *mut Manager;
            if !mgr.is_null() {
                // Ensure frees occur under the retained primary context
                let mut old: cust::sys::CUcontext = std::ptr::null_mut();
                let _ = cust::sys::cuCtxPushCurrent((*mgr).ctx);
                // drop mgr (drops DeviceBuffer)
                let _boxed: Box<Manager> = Box::from_raw(mgr);
                let _ = cust::sys::cuCtxPopCurrent(&mut old as *mut _);
                // Release primary context
                let _ = cust::sys::cuDevicePrimaryCtxRelease((*p).dl_tensor.device.device_id as i32);
            }
            // Drop the DLManagedTensorVersioned itself
            let _ = Box::from_raw(p);
        }
        unsafe extern "C" fn deleter_legacy(p: *mut DLManagedTensor) {
            if p.is_null() { return; }
            let mgr = (*p).manager_ctx as *mut Manager;
            if !mgr.is_null() {
                let mut old: cust::sys::CUcontext = std::ptr::null_mut();
                let _ = cust::sys::cuCtxPushCurrent((*mgr).ctx);
                let _boxed: Box<Manager> = Box::from_raw(mgr);
                let _ = cust::sys::cuCtxPopCurrent(&mut old as *mut _);
                // Legacy capsule lacks versioned device; we still release by device id we stored
                let _ = cust::sys::cuDevicePrimaryCtxRelease((*_boxed).device_id);
            }
            let _ = Box::from_raw(p);
        }

        struct Manager {
            ctx: cust::sys::CUcontext,
            device_id: i32,
            _buf: DeviceBuffer<f32>,
            _shape: Box<[i64; 2]>,
            _strides: Box<[i64; 2]>,
        }

        unsafe extern "C" fn capsule_dtor(capsule: *mut pyo3::ffi::PyObject) {
            // Try versioned first
            let vname = std::ffi::CString::new("dltensor_versioned").unwrap();
            let legacy = std::ffi::CString::new("dltensor").unwrap();
            let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, vname.as_ptr());
            if !ptr.is_null() {
                let mt = ptr as *mut DLManagedTensorVersioned;
                if let Some(del) = (*mt).deleter { del(mt) }
                return;
            }
            let ptr2 = pyo3::ffi::PyCapsule_GetPointer(capsule, legacy.as_ptr());
            if !ptr2.is_null() {
                let mt = ptr2 as *mut DLManagedTensor;
                if let Some(del) = (*mt).deleter { del(mt) }
            }
        }

        // Prepare manager and capsule contents
        let mgr = Box::new(Manager { ctx: retained, device_id: dev_id, _buf: inner.buf, _shape: shape, _strides: strides });
        let mgr_ptr = Box::into_raw(mgr);

        if use_versioned {
            let mut mt = Box::new(DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 2 },
                manager_ctx: mgr_ptr as *mut _,
                deleter: Some(deleter_v1),
                flags: 0,
                dl_tensor: DLTensor {
                    data: data_ptr,
                    device: DLDevice { device_type: dev_type, device_id: dev_id },
                    ndim: 2,
                    dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
                    shape: unsafe { (*mgr_ptr)._shape.as_mut_ptr() },
                    strides: unsafe { (*mgr_ptr)._strides.as_mut_ptr() },
                    byte_offset: 0,
                }
            });
            // shape/strides pointers set from manager buffers
            let raw = Box::into_raw(mt);
            let name = std::ffi::CString::new("dltensor_versioned").unwrap();
            let cap = unsafe { pyo3::ffi::PyCapsule_New(raw as *mut std::ffi::c_void, name.as_ptr(), Some(capsule_dtor)) };
            if cap.is_null() { unsafe { deleter_v1(raw) }; return Err(PyValueError::new_err("failed to create DLPack capsule")); }
            Ok(unsafe { PyObject::from_owned_ptr(py, cap) })
        } else {
            let mut mt = Box::new(DLManagedTensor {
                dl_tensor: DLTensor {
                    data: data_ptr,
                    device: DLDevice { device_type: dev_type, device_id: dev_id },
                    ndim: 2,
                    dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
                    shape: unsafe { (*mgr_ptr)._shape.as_mut_ptr() },
                    strides: unsafe { (*mgr_ptr)._strides.as_mut_ptr() },
                    byte_offset: 0,
                },
                manager_ctx: mgr_ptr as *mut _,
                deleter: Some(deleter_legacy),
            });
            let raw = Box::into_raw(mt);
            let name = std::ffi::CString::new("dltensor").unwrap();
            let cap = unsafe { pyo3::ffi::PyCapsule_New(raw as *mut std::ffi::c_void, name.as_ptr(), Some(capsule_dtor)) };
            if cap.is_null() { unsafe { deleter_legacy(raw) }; return Err(PyValueError::new_err("failed to create DLPack capsule")); }
            Ok(unsafe { PyObject::from_owned_ptr(py, cap) })
        }
    }
}

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
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::alloc::{alloc, dealloc, Layout};
use std::convert::AsRef;
use std::error::Error;
use std::mem::MaybeUninit;
use thiserror::Error;

impl<'a> AsRef<[f64]> for AlmaInput<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f64] {
        match &self.data {
            AlmaData::Slice(slice) => slice,
            AlmaData::Candles { candles, source } => source_type(candles, source),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AlmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AlmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", derive(Serialize, Deserialize))]
pub struct AlmaParams {
    pub period: Option<usize>,
    pub offset: Option<f64>,
    pub sigma: Option<f64>,
}

impl Default for AlmaParams {
    fn default() -> Self {
        Self {
            period: Some(9),
            offset: Some(0.85),
            sigma: Some(6.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlmaInput<'a> {
    pub data: AlmaData<'a>,
    pub params: AlmaParams,
}

impl<'a> AlmaInput<'a> {
    #[inline]
    pub fn from_candles(c: &'a Candles, s: &'a str, p: AlmaParams) -> Self {
        Self {
            data: AlmaData::Candles {
                candles: c,
                source: s,
            },
            params: p,
        }
    }
    #[inline]
    pub fn from_slice(sl: &'a [f64], p: AlmaParams) -> Self {
        Self {
            data: AlmaData::Slice(sl),
            params: p,
        }
    }
    #[inline]
    pub fn with_default_candles(c: &'a Candles) -> Self {
        Self::from_candles(c, "close", AlmaParams::default())
    }
    #[inline]
    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(9)
    }
    #[inline]
    pub fn get_offset(&self) -> f64 {
        self.params.offset.unwrap_or(0.85)
    }
    #[inline]
    pub fn get_sigma(&self) -> f64 {
        self.params.sigma.unwrap_or(6.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AlmaBuilder {
    period: Option<usize>,
    offset: Option<f64>,
    sigma: Option<f64>,
    kernel: Kernel,
}

impl Default for AlmaBuilder {
    fn default() -> Self {
        Self {
            period: None,
            offset: None,
            sigma: None,
            kernel: Kernel::Auto,
        }
    }
}

impl AlmaBuilder {
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
    pub fn offset(mut self, x: f64) -> Self {
        self.offset = Some(x);
        self
    }
    #[inline(always)]
    pub fn sigma(mut self, s: f64) -> Self {
        self.sigma = Some(s);
        self
    }
    #[inline(always)]
    pub fn kernel(mut self, k: Kernel) -> Self {
        self.kernel = k;
        self
    }

    #[inline(always)]
    pub fn apply(self, c: &Candles) -> Result<AlmaOutput, AlmaError> {
        let p = AlmaParams {
            period: self.period,
            offset: self.offset,
            sigma: self.sigma,
        };
        let i = AlmaInput::from_candles(c, "close", p);
        alma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn apply_slice(self, d: &[f64]) -> Result<AlmaOutput, AlmaError> {
        let p = AlmaParams {
            period: self.period,
            offset: self.offset,
            sigma: self.sigma,
        };
        let i = AlmaInput::from_slice(d, p);
        alma_with_kernel(&i, self.kernel)
    }

    #[inline(always)]
    pub fn into_stream(self) -> Result<AlmaStream, AlmaError> {
        let p = AlmaParams {
            period: self.period,
            offset: self.offset,
            sigma: self.sigma,
        };
        AlmaStream::try_new(p)
    }
}

#[derive(Debug, Error)]
pub enum AlmaError {
    #[error("alma: Input data slice is empty.")]
    EmptyInputData,
    #[error("alma: All values are NaN.")]
    AllValuesNaN,

    #[error("alma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("alma: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },

    #[error("alma: Invalid sigma: {sigma}")]
    InvalidSigma { sigma: f64 },

    #[error("alma: Invalid offset: {offset}")]
    InvalidOffset { offset: f64 },

    #[error("alma: Output length mismatch: expected {expected}, got {got}")]
    OutputLengthMismatch { expected: usize, got: usize },

    #[error("alma: Invalid range: start={start}, end={end}, step={step}")]
    InvalidRange { start: String, end: String, step: String },

    #[error("alma: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(crate::utilities::enums::Kernel),
}

#[inline]
pub fn alma(input: &AlmaInput) -> Result<AlmaOutput, AlmaError> {
    alma_with_kernel(input, Kernel::Auto)
}

#[inline(always)]
fn alma_compute_into(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first: usize,
    inv_n: f64,
    kernel: Kernel,
    out: &mut [f64],
) {
    unsafe {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if matches!(kernel, Kernel::Scalar | Kernel::ScalarBatch) {
                alma_simd128(data, weights, period, first, inv_n, out);
                return;
            }
        }

        match kernel {
            Kernel::Scalar | Kernel::ScalarBatch => {
                alma_scalar(data, weights, period, first, inv_n, out)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => alma_avx2(data, weights, period, first, inv_n, out),
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                alma_avx512(data, weights, period, first, inv_n, out)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                alma_scalar(data, weights, period, first, inv_n, out)
            }
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
fn alma_prepare<'a>(
    input: &'a AlmaInput,
    kernel: Kernel,
) -> Result<(&'a [f64], AVec<f64>, usize, usize, f64, Kernel), AlmaError> {
    let data: &[f64] = input.as_ref();
    let len = data.len();
    if len == 0 {
        return Err(AlmaError::EmptyInputData);
    }
    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlmaError::AllValuesNaN)?;
    let period = input.get_period();
    let offset = input.get_offset();
    let sigma = input.get_sigma();

    if period == 0 || period > len {
        return Err(AlmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if len - first < period {
        return Err(AlmaError::NotEnoughValidData {
            needed: period,
            valid: len - first,
        });
    }
    if sigma <= 0.0 {
        return Err(AlmaError::InvalidSigma { sigma });
    }
    if !(0.0..=1.0).contains(&offset) || offset.is_nan() || offset.is_infinite() {
        return Err(AlmaError::InvalidOffset { offset });
    }

    let m = offset * (period - 1) as f64;
    let s = period as f64 / sigma;
    let s2 = 2.0 * s * s;

    // Allocate aligned weights with padding for SIMD operations
    let aligned_period = ((period + 7) / 8) * 8; // Round up to nearest multiple of 8
    let mut weights: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, aligned_period);
    weights.resize(aligned_period, 0.0);

    // Pre-compute reciprocal for multiplication instead of division
    let inv_s2 = 1.0 / s2;
    let mut norm = 0.0;

    // Vectorized weight computation where possible
    for i in 0..period {
        let diff = i as f64 - m;
        let w = (-diff * diff * inv_s2).exp();
        weights[i] = w;
        norm += w;
    }
    let inv_norm = 1.0 / norm;

    let chosen = match kernel {
        Kernel::Auto => detect_best_kernel(),
        k => k,
    };

    Ok((data, weights, period, first, inv_norm, chosen))
}

pub fn alma_with_kernel(input: &AlmaInput, kernel: Kernel) -> Result<AlmaOutput, AlmaError> {
    let (data, weights, period, first, inv_n, chosen) = alma_prepare(input, kernel)?;

    let mut out = alloc_with_nan_prefix(data.len(), first + period - 1);

    alma_compute_into(data, &weights, period, first, inv_n, chosen, &mut out);

    Ok(AlmaOutput { values: out })
}

#[inline]
pub fn alma_into_slice(dst: &mut [f64], input: &AlmaInput, kern: Kernel) -> Result<(), AlmaError> {
    let (data, weights, period, first, inv_n, chosen) = alma_prepare(input, kern)?;

    if dst.len() != data.len() {
        return Err(AlmaError::OutputLengthMismatch { expected: data.len(), got: dst.len() });
    }

    alma_compute_into(data, &weights, period, first, inv_n, chosen, dst);

    let warmup_end = first + period - 1;
    for v in &mut dst[..warmup_end] {
        *v = f64::NAN;
    }

    Ok(())
}

/// Writes ALMA results into the provided output slice without allocating.
///
/// - Preserves NaN warmups exactly like the Vec-returning API (quiet-NaN prefix).
/// - The output slice length must equal the input length.
#[cfg(not(feature = "wasm"))]
#[inline]
pub fn alma_into(input: &AlmaInput, out: &mut [f64]) -> Result<(), AlmaError> {
    let (data, weights, period, first, inv_n, chosen) = alma_prepare(input, Kernel::Auto)?;

    if out.len() != data.len() {
        return Err(AlmaError::OutputLengthMismatch { expected: data.len(), got: out.len() });
    }

    // Prefill warmup prefix with the same quiet-NaN pattern used by alloc_with_nan_prefix
    let warmup_end = first + period - 1;
    let qnan = f64::from_bits(0x7ff8_0000_0000_0000);
    let warm = warmup_end.min(out.len());
    for v in &mut out[..warm] {
        *v = qnan;
    }

    // Compute into the destination buffer for the valid range
    alma_compute_into(data, &weights, period, first, inv_n, chosen, out);

    Ok(())
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn hsum_pd_zmm(v: __m512d) -> f64 {
    #[allow(unused_unsafe)]
    {
        _mm512_reduce_add_pd(v)
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f")]
pub fn alma_avx512(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        unsafe { alma_avx512_short(data, weights, period, first_valid, inv_norm, out) }
    } else {
        unsafe { alma_avx512_long(data, weights, period, first_valid, inv_norm, out) }
    }
}

#[inline(always)]
pub fn alma_scalar(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_val: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    assert!(
        weights.len() >= period,
        "weights.len() must be at least `period`"
    );
    assert!(
        out.len() >= data.len(),
        "`out` must be at least as long as `data`"
    );

    let p4 = period & !3;

    for i in (first_val + period - 1)..data.len() {
        let start = i + 1 - period;
        let window = &data[start..start + period];

        let mut sum = 0.0;
        for (d4, w4) in window[..p4]
            .chunks_exact(4)
            .zip(weights[..p4].chunks_exact(4))
        {
            sum += d4[0] * w4[0] + d4[1] * w4[1] + d4[2] * w4[2] + d4[3] * w4[3];
        }

        for (d, w) in window[p4..].iter().zip(&weights[p4..]) {
            sum += d * w;
        }

        out[i] = sum * inv_norm;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn alma_simd128(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_val: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    use core::arch::wasm32::*;

    assert!(
        weights.len() >= period,
        "weights.len() must be at least `period`"
    );
    assert!(
        out.len() >= data.len(),
        "`out` must be at least as long as `data`"
    );

    const STEP: usize = 2;
    let chunks = period / STEP;
    let tail = period % STEP;

    for i in (first_val + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut acc = f64x2_splat(0.0);

        for blk in 0..chunks {
            let idx = blk * STEP;
            let w = v128_load(weights.as_ptr().add(idx) as *const v128);
            let d = v128_load(data.as_ptr().add(start + idx) as *const v128);
            acc = f64x2_add(acc, f64x2_mul(d, w));
        }

        let mut sum = f64x2_extract_lane::<0>(acc) + f64x2_extract_lane::<1>(acc);

        if tail != 0 {
            sum += data[start + chunks * STEP] * weights[chunks * STEP];
        }

        out[i] = sum * inv_norm;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn alma_avx2(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    const STEP: usize = 4;
    let chunks = period / STEP;
    let tail = period % STEP;

    // Use 2-way unrolling for better instruction-level parallelism
    let paired_chunks = chunks / 2;
    let odd_chunk = chunks % 2;

    let tail_mask = match tail {
        0 => _mm256_setzero_si256(),
        1 => _mm256_setr_epi64x(-1, 0, 0, 0),
        2 => _mm256_setr_epi64x(-1, -1, 0, 0),
        3 => _mm256_setr_epi64x(-1, -1, -1, 0),
        _ => unreachable!(),
    };

    for i in (first_valid + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();

        // Process chunks in pairs for better pipelining
        for blk in 0..paired_chunks {
            let idx0 = (blk * 2) * STEP;
            let idx1 = (blk * 2 + 1) * STEP;

            let w0 = _mm256_loadu_pd(weights.as_ptr().add(idx0));
            let w1 = _mm256_loadu_pd(weights.as_ptr().add(idx1));
            let d0 = _mm256_loadu_pd(data.as_ptr().add(start + idx0));
            let d1 = _mm256_loadu_pd(data.as_ptr().add(start + idx1));

            acc0 = _mm256_fmadd_pd(d0, w0, acc0);
            acc1 = _mm256_fmadd_pd(d1, w1, acc1);
        }

        // Handle odd chunk if present
        if odd_chunk != 0 {
            let idx = (paired_chunks * 2) * STEP;
            let w = _mm256_loadu_pd(weights.as_ptr().add(idx));
            let d = _mm256_loadu_pd(data.as_ptr().add(start + idx));
            acc0 = _mm256_fmadd_pd(d, w, acc0);
        }

        // Combine accumulators
        let acc = _mm256_add_pd(acc0, acc1);

        // Handle tail elements
        let final_acc = if tail != 0 {
            let w_tail = _mm256_maskload_pd(weights.as_ptr().add(chunks * STEP), tail_mask);
            let d_tail = _mm256_maskload_pd(data.as_ptr().add(start + chunks * STEP), tail_mask);
            _mm256_fmadd_pd(d_tail, w_tail, acc)
        } else {
            acc
        };

        // Horizontal sum with hadd for better throughput
        let sum128 = _mm_add_pd(
            _mm256_castpd256_pd128(final_acc),
            _mm256_extractf128_pd(final_acc, 1),
        );
        let sum = _mm_cvtsd_f64(_mm_hadd_pd(sum128, sum128));

        *out.get_unchecked_mut(i) = sum * inv_norm;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
unsafe fn alma_avx512_short(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    debug_assert!(period >= 1);
    debug_assert!(data.len() == out.len());
    debug_assert!(weights.len() >= period);

    const STEP: usize = 8;
    let chunks = period / STEP;
    let tail_len = period % STEP;
    let tail_mask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

    if chunks == 0 {
        let w_vec = _mm512_maskz_loadu_pd(tail_mask, weights.as_ptr());
        for i in (first_valid + period - 1)..data.len() {
            let start = i + 1 - period;
            let d_vec = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start));
            let sum = hsum_pd_zmm(_mm512_mul_pd(d_vec, w_vec)) * inv_norm;
            *out.get_unchecked_mut(i) = sum;
        }
        return;
    }

    for i in (first_valid + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut acc = _mm512_setzero_pd();

        for blk in 0..chunks {
            let w = _mm512_load_pd(weights.as_ptr().add(blk * STEP));
            let d = _mm512_loadu_pd(data.as_ptr().add(start + blk * STEP));
            acc = _mm512_fmadd_pd(d, w, acc);
        }

        if tail_len != 0 {
            let w_tail = _mm512_maskz_loadu_pd(tail_mask, weights.as_ptr().add(chunks * STEP));
            let d_tail = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + chunks * STEP));
            acc = _mm512_fmadd_pd(d_tail, w_tail, acc);
        }

        *out.get_unchecked_mut(i) = hsum_pd_zmm(acc) * inv_norm;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx512f,fma")]
unsafe fn alma_avx512_long(
    data: &[f64],
    weights: &[f64],
    period: usize,
    first_valid: usize,
    inv_norm: f64,
    out: &mut [f64],
) {
    const STEP: usize = 8;
    let n_chunks = period / STEP;
    let tail_len = period % STEP;
    // Use 8-way unrolling for maximum ILP
    let unroll8 = n_chunks & !7;
    let tail_mask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

    debug_assert!(period >= 1 && n_chunks > 0);
    debug_assert_eq!(data.len(), out.len());
    debug_assert!(weights.len() >= period);

    const MAX_STACK_CHUNKS: usize = 256;
    let mut stack_storage = MaybeUninit::<[__m512d; MAX_STACK_CHUNKS]>::uninit();
    let mut heap_storage: Option<Vec<__m512d>> = None;

    let wregs: &[__m512d] = if n_chunks <= MAX_STACK_CHUNKS {
        let base = stack_storage.as_mut_ptr().cast::<__m512d>();
        for blk in 0..n_chunks {
            unsafe {
                base.add(blk)
                    .write(_mm512_load_pd(weights.as_ptr().add(blk * STEP)));
            }
        }
        unsafe { core::slice::from_raw_parts(base, n_chunks) }
    } else {
        let mut regs = Vec::with_capacity(n_chunks);
        for blk in 0..n_chunks {
            regs.push(_mm512_load_pd(weights.as_ptr().add(blk * STEP)));
        }
        heap_storage = Some(regs);
        heap_storage.as_ref().unwrap().as_slice()
    };
    let w_tail = if tail_len != 0 {
        Some(_mm512_maskz_loadu_pd(
            tail_mask,
            weights.as_ptr().add(n_chunks * STEP),
        ))
    } else {
        None
    };

    let mut data_ptr = data.as_ptr().add(first_valid);
    let stop_ptr = data.as_ptr().add(data.len());

    let mut dst_ptr = out.as_mut_ptr().add(first_valid + period - 1);

    if tail_len == 0 {
        while data_ptr.add(period) <= stop_ptr {
            let mut s0 = _mm512_setzero_pd();
            let mut s1 = _mm512_setzero_pd();
            let mut s2 = _mm512_setzero_pd();
            let mut s3 = _mm512_setzero_pd();
            let mut s4 = _mm512_setzero_pd();
            let mut s5 = _mm512_setzero_pd();
            let mut s6 = _mm512_setzero_pd();
            let mut s7 = _mm512_setzero_pd();

            // 8-way unrolled loop for maximum throughput
            for blk in (0..unroll8).step_by(8) {
                let d0 = _mm512_loadu_pd(data_ptr.add((blk + 0) * STEP));
                let d1 = _mm512_loadu_pd(data_ptr.add((blk + 1) * STEP));
                let d2 = _mm512_loadu_pd(data_ptr.add((blk + 2) * STEP));
                let d3 = _mm512_loadu_pd(data_ptr.add((blk + 3) * STEP));
                let d4 = _mm512_loadu_pd(data_ptr.add((blk + 4) * STEP));
                let d5 = _mm512_loadu_pd(data_ptr.add((blk + 5) * STEP));
                let d6 = _mm512_loadu_pd(data_ptr.add((blk + 6) * STEP));
                let d7 = _mm512_loadu_pd(data_ptr.add((blk + 7) * STEP));

                s0 = _mm512_fmadd_pd(d0, *wregs.get_unchecked(blk + 0), s0);
                s1 = _mm512_fmadd_pd(d1, *wregs.get_unchecked(blk + 1), s1);
                s2 = _mm512_fmadd_pd(d2, *wregs.get_unchecked(blk + 2), s2);
                s3 = _mm512_fmadd_pd(d3, *wregs.get_unchecked(blk + 3), s3);
                s4 = _mm512_fmadd_pd(d4, *wregs.get_unchecked(blk + 4), s4);
                s5 = _mm512_fmadd_pd(d5, *wregs.get_unchecked(blk + 5), s5);
                s6 = _mm512_fmadd_pd(d6, *wregs.get_unchecked(blk + 6), s6);
                s7 = _mm512_fmadd_pd(d7, *wregs.get_unchecked(blk + 7), s7);
            }

            // Handle remaining chunks
            for blk in unroll8..n_chunks {
                let d = _mm512_loadu_pd(data_ptr.add(blk * STEP));
                s0 = _mm512_fmadd_pd(d, *wregs.get_unchecked(blk), s0);
            }

            // Tree reduction for better parallelism
            let sum01 = _mm512_add_pd(s0, s1);
            let sum23 = _mm512_add_pd(s2, s3);
            let sum45 = _mm512_add_pd(s4, s5);
            let sum67 = _mm512_add_pd(s6, s7);
            let sum0123 = _mm512_add_pd(sum01, sum23);
            let sum4567 = _mm512_add_pd(sum45, sum67);
            let tot = _mm512_add_pd(sum0123, sum4567);

            *dst_ptr = hsum_pd_zmm(tot) * inv_norm;

            data_ptr = data_ptr.add(1);
            dst_ptr = dst_ptr.add(1);
        }
    } else {
        let wt = w_tail.expect("tail_len != 0 but w_tail missing");

        while data_ptr.add(period) <= stop_ptr {
            let mut s0 = _mm512_setzero_pd();
            let mut s1 = _mm512_setzero_pd();
            let mut s2 = _mm512_setzero_pd();
            let mut s3 = _mm512_setzero_pd();
            let mut s4 = _mm512_setzero_pd();
            let mut s5 = _mm512_setzero_pd();
            let mut s6 = _mm512_setzero_pd();
            let mut s7 = _mm512_setzero_pd();

            // 8-way unrolled loop
            for blk in (0..unroll8).step_by(8) {
                let d0 = _mm512_loadu_pd(data_ptr.add((blk + 0) * STEP));
                let d1 = _mm512_loadu_pd(data_ptr.add((blk + 1) * STEP));
                let d2 = _mm512_loadu_pd(data_ptr.add((blk + 2) * STEP));
                let d3 = _mm512_loadu_pd(data_ptr.add((blk + 3) * STEP));
                let d4 = _mm512_loadu_pd(data_ptr.add((blk + 4) * STEP));
                let d5 = _mm512_loadu_pd(data_ptr.add((blk + 5) * STEP));
                let d6 = _mm512_loadu_pd(data_ptr.add((blk + 6) * STEP));
                let d7 = _mm512_loadu_pd(data_ptr.add((blk + 7) * STEP));

                s0 = _mm512_fmadd_pd(d0, *wregs.get_unchecked(blk + 0), s0);
                s1 = _mm512_fmadd_pd(d1, *wregs.get_unchecked(blk + 1), s1);
                s2 = _mm512_fmadd_pd(d2, *wregs.get_unchecked(blk + 2), s2);
                s3 = _mm512_fmadd_pd(d3, *wregs.get_unchecked(blk + 3), s3);
                s4 = _mm512_fmadd_pd(d4, *wregs.get_unchecked(blk + 4), s4);
                s5 = _mm512_fmadd_pd(d5, *wregs.get_unchecked(blk + 5), s5);
                s6 = _mm512_fmadd_pd(d6, *wregs.get_unchecked(blk + 6), s6);
                s7 = _mm512_fmadd_pd(d7, *wregs.get_unchecked(blk + 7), s7);
            }

            // Handle remaining chunks before tail
            for blk in unroll8..n_chunks {
                let d = _mm512_loadu_pd(data_ptr.add(blk * STEP));
                s0 = _mm512_fmadd_pd(d, *wregs.get_unchecked(blk), s0);
            }

            // Handle tail elements
            let d_tail = _mm512_maskz_loadu_pd(tail_mask, data_ptr.add(n_chunks * STEP));
            s0 = _mm512_fmadd_pd(d_tail, wt, s0);

            // Tree reduction
            let sum01 = _mm512_add_pd(s0, s1);
            let sum23 = _mm512_add_pd(s2, s3);
            let sum45 = _mm512_add_pd(s4, s5);
            let sum67 = _mm512_add_pd(s6, s7);
            let sum0123 = _mm512_add_pd(sum01, sum23);
            let sum4567 = _mm512_add_pd(sum45, sum67);
            let tot = _mm512_add_pd(sum0123, sum4567);

            *dst_ptr = hsum_pd_zmm(tot) * inv_norm;

            data_ptr = data_ptr.add(1);
            dst_ptr = dst_ptr.add(1);
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlmaStream {
    period: usize,
    // keep weights aligned for fast SIMD loads
    weights: AVec<f64>,
    inv_norm: f64,

    // canonical ring (for inspection/compat)
    buffer: Vec<f64>,

    // mirrored ring — active window is contiguous at &buf2[head .. head+period]
    buf2: Vec<f64>, // length = 2*period

    head: usize,   // index of oldest element (next overwrite)
    filled: usize, // number of values seen so far (<= period)
    kernel: Kernel,
}

impl AlmaStream {
    pub fn try_new(params: AlmaParams) -> Result<Self, AlmaError> {
        let period = params.period.unwrap_or(9);
        if period == 0 {
            return Err(AlmaError::InvalidPeriod {
                period,
                data_len: 0,
            });
        }
        let offset = params.offset.unwrap_or(0.85);
        if !(0.0..=1.0).contains(&offset) || offset.is_nan() || offset.is_infinite() {
            return Err(AlmaError::InvalidOffset { offset });
        }
        let sigma = params.sigma.unwrap_or(6.0);
        if sigma <= 0.0 {
            return Err(AlmaError::InvalidSigma { sigma });
        }

        // build weights (exact)
        let m = offset * (period - 1) as f64;
        let s = period as f64 / sigma;
        let s2 = 2.0 * s * s;

        let mut weights = AVec::<f64>::with_capacity(CACHELINE_ALIGN, period);
        weights.resize(period, 0.0);

        let mut norm = 0.0;
        for i in 0..period {
            let diff = i as f64 - m;
            let w = (-(diff * diff) / s2).exp();
            weights[i] = w;
            norm += w;
        }
        let inv_norm = 1.0 / norm;

        // mirrored ring buffer
        let buffer = vec![f64::NAN; period];
        let buf2 = vec![f64::NAN; period * 2];
        let kernel = detect_best_kernel();

        Ok(Self {
            period,
            weights,
            inv_norm,
            buffer,
            buf2,
            head: 0,
            filled: 0,
            kernel,
        })
    }

    /// Push one value; returns None until warm, then exact ALMA value thereafter.
    #[inline(always)]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        let h = self.head;

        // write into canonical ring
        self.buffer[h] = value;
        // mirror writes so our active window is contiguous
        self.buf2[h] = value;
        self.buf2[h + self.period] = value;

        // advance head (oldest = next slot to overwrite)
        let mut new_h = h + 1;
        if new_h == self.period {
            new_h = 0;
        }
        self.head = new_h;

        // warmup tracking
        if self.filled < self.period {
            self.filled += 1;
            if self.filled < self.period {
                return None;
            }
        }

        Some(self.dot_at_head())
    }

    #[inline(always)]
    fn dot_at_head(&self) -> f64 {
        let start = self.head;
        let end = start + self.period;
        let x = &self.buf2[start..end];
        let w = &self.weights[..self.period];
        let acc = dot_contiguous(self.kernel, x, w);
        acc * self.inv_norm
    }
}

// ---------- contiguous dot helpers (SIMD where available; safe scalar fallback) ----------

#[inline(always)]
fn dot_scalar_unrolled_safe(x: &[f64], w: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), w.len());
    let n = x.len();
    let mut i = 0usize;
    let n4 = n & !3;
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;

    while i < n4 {
        s0 += x[i] * w[i];
        s1 += x[i + 1] * w[i + 1];
        s2 += x[i + 2] * w[i + 2];
        s3 += x[i + 3] * w[i + 3];
        i += 4;
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    while i < n {
        sum += x[i] * w[i];
        i += 1;
    }
    sum
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hsum256(v: __m256d) -> f64 {
    let hi = _mm256_extractf128_pd(v, 1);
    let lo = _mm256_castpd256_pd128(v);
    let s = _mm_add_pd(hi, lo);
    let s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
    _mm_cvtsd_f64(s)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hsum512(v: __m512d) -> f64 {
    _mm512_reduce_add_pd(v)
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dot_avx2(x: *const f64, w: *const f64, n: usize) -> f64 {
    let mut i = 0usize;
    let n4 = n & !3;
    let mut acc = _mm256_setzero_pd();
    while i < n4 {
        let xv = _mm256_loadu_pd(x.add(i));
        let wv = _mm256_loadu_pd(w.add(i));
        acc = _mm256_fmadd_pd(xv, wv, acc);
        i += 4;
    }
    let mut sum = hsum256(acc);
    while i < n {
        sum += *x.add(i) * *w.add(i);
        i += 1;
    }
    sum
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dot_avx512(x: *const f64, w: *const f64, n: usize) -> f64 {
    let mut i = 0usize;
    let n8 = n & !7;
    let mut acc = _mm512_setzero_pd();
    while i < n8 {
        let xv = _mm512_loadu_pd(x.add(i));
        let wv = _mm512_loadu_pd(w.add(i));
        acc = _mm512_fmadd_pd(xv, wv, acc);
        i += 8;
    }
    let mut sum = hsum512(acc);
    while i < n {
        sum += *x.add(i) * *w.add(i);
        i += 1;
    }
    sum
}

#[inline(always)]
fn dot_contiguous(kernel: Kernel, x: &[f64], w: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), w.len());
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        match kernel {
            Kernel::Avx512 | Kernel::Avx512Batch => unsafe {
                return dot_avx512(x.as_ptr(), w.as_ptr(), x.len());
            },
            Kernel::Avx2 | Kernel::Avx2Batch => unsafe {
                return dot_avx2(x.as_ptr(), w.as_ptr(), x.len());
            },
            _ => {}
        }
    }
    // safe scalar fallback
    dot_scalar_unrolled_safe(x, w)
}

#[derive(Clone, Debug)]
pub struct AlmaBatchRange {
    pub period: (usize, usize, usize),
    pub offset: (f64, f64, f64),
    pub sigma: (f64, f64, f64),
}

impl Default for AlmaBatchRange {
    fn default() -> Self {
        Self {
            period: (9, 240, 1),
            offset: (0.85, 0.85, 0.0),
            sigma: (6.0, 6.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AlmaBatchBuilder {
    range: AlmaBatchRange,
    kernel: Kernel,
}

impl AlmaBatchBuilder {
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

    #[inline]
    pub fn offset_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.offset = (start, end, step);
        self
    }
    #[inline]
    pub fn offset_static(mut self, x: f64) -> Self {
        self.range.offset = (x, x, 0.0);
        self
    }

    #[inline]
    pub fn sigma_range(mut self, start: f64, end: f64, step: f64) -> Self {
        self.range.sigma = (start, end, step);
        self
    }
    #[inline]
    pub fn sigma_static(mut self, s: f64) -> Self {
        self.range.sigma = (s, s, 0.0);
        self
    }

    pub fn apply_slice(self, data: &[f64]) -> Result<AlmaBatchOutput, AlmaError> {
        alma_batch_with_kernel(data, &self.range, self.kernel)
    }

    pub fn with_default_slice(data: &[f64], k: Kernel) -> Result<AlmaBatchOutput, AlmaError> {
        AlmaBatchBuilder::new().kernel(k).apply_slice(data)
    }

    pub fn apply_candles(self, c: &Candles, src: &str) -> Result<AlmaBatchOutput, AlmaError> {
        let slice = source_type(c, src);
        self.apply_slice(slice)
    }

    pub fn with_default_candles(c: &Candles) -> Result<AlmaBatchOutput, AlmaError> {
        AlmaBatchBuilder::new()
            .kernel(Kernel::Auto)
            .apply_candles(c, "close")
    }
}

pub fn alma_batch_with_kernel(
    data: &[f64],
    sweep: &AlmaBatchRange,
    k: Kernel,
) -> Result<AlmaBatchOutput, AlmaError> {
    let kernel = match k {
        Kernel::Auto => detect_best_batch_kernel(),
        other if other.is_batch() => other,
        _ => return Err(AlmaError::InvalidKernelForBatch(k)),
    };

    let simd = match kernel {
        Kernel::Avx512Batch => Kernel::Avx512,
        Kernel::Avx2Batch => Kernel::Avx2,
        Kernel::ScalarBatch => Kernel::Scalar,
        _ => unreachable!(),
    };
    alma_batch_par_slice(data, sweep, simd)
}

#[derive(Clone, Debug)]
pub struct AlmaBatchOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AlmaParams>,
    pub rows: usize,
    pub cols: usize,
}
impl AlmaBatchOutput {
    pub fn row_for_params(&self, p: &AlmaParams) -> Option<usize> {
        self.combos.iter().position(|c| {
            c.period.unwrap_or(9) == p.period.unwrap_or(9)
                && (c.offset.unwrap_or(0.85) - p.offset.unwrap_or(0.85)).abs() < 1e-12
                && (c.sigma.unwrap_or(6.0) - p.sigma.unwrap_or(6.0)).abs() < 1e-12
        })
    }

    pub fn values_for(&self, p: &AlmaParams) -> Option<&[f64]> {
        self.row_for_params(p).map(|row| {
            let start = row * self.cols;
            &self.values[start..start + self.cols]
        })
    }
}

#[inline(always)]
fn expand_grid(r: &AlmaBatchRange) -> Result<Vec<AlmaParams>, AlmaError> {
    fn axis_usize((start, end, step): (usize, usize, usize)) -> Result<Vec<usize>, AlmaError> {
        if step == 0 || start == end { return Ok(vec![start]); }
        if start < end {
            return Ok((start..=end).step_by(step.max(1)).collect());
        }
        // reversed bounds
        let mut v = Vec::new();
        let mut x = start as isize;
        let end_i = end as isize;
        let st = (step as isize).max(1);
        while x >= end_i { v.push(x as usize); x -= st; }
        if v.is_empty() {
            return Err(AlmaError::InvalidRange { start: start.to_string(), end: end.to_string(), step: step.to_string() });
        }
        Ok(v)
    }
    fn axis_f64((start, end, step): (f64, f64, f64)) -> Result<Vec<f64>, AlmaError> {
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 { return Ok(vec![start]); }
        if start < end {
            let mut v = Vec::new();
            let mut x = start;
            let st = step.abs();
            while x <= end + 1e-12 { v.push(x); x += st; }
            if v.is_empty() { return Err(AlmaError::InvalidRange { start: start.to_string(), end: end.to_string(), step: step.to_string() }); }
            return Ok(v);
        }
        let mut v = Vec::new();
        let mut x = start;
        let st = step.abs();
        while x + 1e-12 >= end { v.push(x); x -= st; }
        if v.is_empty() { return Err(AlmaError::InvalidRange { start: start.to_string(), end: end.to_string(), step: step.to_string() }); }
        Ok(v)
    }

    let periods = axis_usize(r.period)?;
    let offsets = axis_f64(r.offset)?;
    let sigmas = axis_f64(r.sigma)?;

    let cap = periods.len()
        .checked_mul(offsets.len())
        .and_then(|x| x.checked_mul(sigmas.len()))
        .ok_or_else(|| AlmaError::InvalidRange { start: "cap".into(), end: "overflow".into(), step: "mul".into() })?;

    let mut out = Vec::with_capacity(cap);
    for &p in &periods {
        for &o in &offsets {
            for &s in &sigmas {
                out.push(AlmaParams { period: Some(p), offset: Some(o), sigma: Some(s) });
            }
        }
    }
    Ok(out)
}

#[inline(always)]
pub fn alma_batch_slice(
    data: &[f64],
    sweep: &AlmaBatchRange,
    kern: Kernel,
) -> Result<AlmaBatchOutput, AlmaError> {
    alma_batch_inner(data, sweep, kern, false)
}

#[inline(always)]
pub fn alma_batch_par_slice(
    data: &[f64],
    sweep: &AlmaBatchRange,
    kern: Kernel,
) -> Result<AlmaBatchOutput, AlmaError> {
    alma_batch_inner(data, sweep, kern, true)
}

#[inline]
fn round_up8(x: usize) -> usize {
    (x + 7) & !7
}

#[inline(always)]
fn alma_batch_inner(
    data: &[f64],
    sweep: &AlmaBatchRange,
    kern: Kernel,
    parallel: bool,
) -> Result<AlmaBatchOutput, AlmaError> {
    let combos = expand_grid(sweep)?;
    let cols = data.len();
    let rows = combos.len();

    if cols == 0 {
        return Err(AlmaError::AllValuesNaN);
    }

    // Guard rows*cols overflow
    let _ = rows.checked_mul(cols).ok_or_else(|| AlmaError::InvalidRange {
        start: rows.to_string(), end: cols.to_string(), step: "rows*cols".into()
    })?;
    let mut buf_mu = make_uninit_matrix(rows, cols);

    let warm: Vec<usize> = combos
        .iter()
        .map(|c| data.iter().position(|x| !x.is_nan()).unwrap_or(0) + c.period.unwrap() - 1)
        .collect();
    init_matrix_prefixes(&mut buf_mu, cols, &warm);

    let mut buf_guard = core::mem::ManuallyDrop::new(buf_mu);
    let out: &mut [f64] = unsafe {
        core::slice::from_raw_parts_mut(buf_guard.as_mut_ptr() as *mut f64, buf_guard.len())
    };

    alma_batch_inner_into(data, sweep, kern, parallel, out)?;

    let values = unsafe {
        Vec::from_raw_parts(
            buf_guard.as_mut_ptr() as *mut f64,
            buf_guard.len(),
            buf_guard.capacity(),
        )
    };

    Ok(AlmaBatchOutput {
        values,
        combos,
        rows,
        cols,
    })
}

#[inline(always)]
fn alma_batch_inner_into(
    data: &[f64],
    sweep: &AlmaBatchRange,
    kern: Kernel,
    parallel: bool,
    out: &mut [f64],
) -> Result<Vec<AlmaParams>, AlmaError> {
    let combos = expand_grid(sweep)?;
    if combos.is_empty() {
        return Err(AlmaError::InvalidRange { start: "range".into(), end: "range".into(), step: "empty".into() });
    }

    let first = data
        .iter()
        .position(|x| !x.is_nan())
        .ok_or(AlmaError::AllValuesNaN)?;
    let max_p = combos
        .iter()
        .map(|c| round_up8(c.period.unwrap()))
        .max()
        .unwrap();
    if data.len() - first < max_p {
        return Err(AlmaError::NotEnoughValidData {
            needed: max_p,
            valid: data.len() - first,
        });
    }

    let rows = combos.len();
    let cols = data.len();
    let mut inv_norms = vec![0.0; rows];

    let cap = rows * max_p;
    let mut flat_w = AVec::<f64>::with_capacity(CACHELINE_ALIGN, cap);
    flat_w.resize(cap, 0.0);

    for (row, prm) in combos.iter().enumerate() {
        let period = prm.period.unwrap();
        let offset = prm.offset.unwrap();
        let sigma = prm.sigma.unwrap();

        if sigma <= 0.0 {
            return Err(AlmaError::InvalidSigma { sigma });
        }
        if !(0.0..=1.0).contains(&offset) || offset.is_nan() || offset.is_infinite() {
            return Err(AlmaError::InvalidOffset { offset });
        }

        let m = offset * (period - 1) as f64;
        let s = period as f64 / sigma;
        let s2 = 2.0 * s * s;

        let mut norm = 0.0;
        for i in 0..period {
            let w = (-(i as f64 - m).powi(2) / s2).exp();
            flat_w[row * max_p + i] = w;
            norm += w;
        }
        inv_norms[row] = 1.0 / norm;
    }

    let out_uninit = unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<f64>, out.len())
    };

    let actual_kern = match kern {
        Kernel::Auto => detect_best_batch_kernel(),
        k => k,
    };

    let do_row = |row: usize, dst_mu: &mut [MaybeUninit<f64>]| unsafe {
        let period = combos[row].period.unwrap();
        let w_ptr = flat_w.as_ptr().add(row * max_p);
        let inv_n = *inv_norms.get_unchecked(row);

        let dst = core::slice::from_raw_parts_mut(dst_mu.as_mut_ptr() as *mut f64, dst_mu.len());

        match actual_kern {
            Kernel::Scalar | Kernel::ScalarBatch => {
                alma_row_scalar(data, first, period, w_ptr, inv_n, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 | Kernel::Avx2Batch => {
                alma_row_avx2(data, first, period, w_ptr, inv_n, dst)
            }
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 | Kernel::Avx512Batch => {
                alma_row_avx512(data, first, period, w_ptr, inv_n, dst)
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Kernel::Avx2 | Kernel::Avx2Batch | Kernel::Avx512 | Kernel::Avx512Batch => {
                alma_row_scalar(data, first, period, w_ptr, inv_n, dst)
            }
            Kernel::Auto => unreachable!("Auto kernel should have been resolved"),
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

#[inline(always)]
unsafe fn alma_row_scalar(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    let p4 = period & !3;
    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut sum = 0.0;
        for k in (0..p4).step_by(4) {
            let w = std::slice::from_raw_parts(w_ptr.add(k), 4);
            let d = &data[start + k..start + k + 4];
            sum += d[0] * w[0] + d[1] * w[1] + d[2] * w[2] + d[3] * w[3];
        }
        for k in p4..period {
            sum += *data.get_unchecked(start + k) * *w_ptr.add(k);
        }
        out[i] = sum * inv_n;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn alma_row_avx2(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    const STEP: usize = 4;
    let vec_blocks = period / STEP;
    let tail = period % STEP;
    let tail_mask = match tail {
        0 => _mm256_setzero_si256(),
        1 => _mm256_setr_epi64x(-1, 0, 0, 0),
        2 => _mm256_setr_epi64x(-1, -1, 0, 0),
        3 => _mm256_setr_epi64x(-1, -1, -1, 0),
        _ => unreachable!(),
    };

    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut acc = _mm256_setzero_pd();

        for blk in 0..vec_blocks {
            let d = _mm256_loadu_pd(data.as_ptr().add(start + blk * STEP));
            let w = _mm256_loadu_pd(w_ptr.add(blk * STEP));
            acc = _mm256_fmadd_pd(d, w, acc);
        }

        if tail != 0 {
            let d = _mm256_maskload_pd(data.as_ptr().add(start + vec_blocks * STEP), tail_mask);
            let w = _mm256_maskload_pd(w_ptr.add(vec_blocks * STEP), tail_mask);
            acc = _mm256_fmadd_pd(d, w, acc);
        }

        let hi = _mm256_extractf128_pd(acc, 1);
        let lo = _mm256_castpd256_pd128(acc);
        let s2 = _mm_add_pd(hi, lo);
        let s1 = _mm_add_pd(s2, _mm_unpackhi_pd(s2, s2));
        out[i] = _mm_cvtsd_f64(s1) * inv_n;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn alma_row_avx512(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    if period <= 32 {
        alma_row_avx512_short(data, first, period, w_ptr, inv_n, out);
    } else {
        alma_row_avx512_long(data, first, period, w_ptr, inv_n, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn alma_row_avx512_short(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    debug_assert!(period <= 32);
    const STEP: usize = 8;

    let chunks = period / STEP;
    let tail_len = period % STEP;
    let tail_mask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

    if chunks == 0 {
        let w_tail = _mm512_maskz_loadu_pd(tail_mask, w_ptr);
        for i in (first + period - 1)..data.len() {
            let start = i + 1 - period;
            let d_tail = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start));
            let res = hsum_pd_zmm(_mm512_mul_pd(d_tail, w_tail)) * inv_n;
            *out.get_unchecked_mut(i) = res;
        }
        return;
    }

    for i in (first + period - 1)..data.len() {
        let start = i + 1 - period;
        let mut acc = _mm512_setzero_pd();

        for blk in 0..chunks {
            let w = _mm512_load_pd(w_ptr.add(blk * STEP));
            let d = _mm512_loadu_pd(data.as_ptr().add(start + blk * STEP));
            acc = _mm512_fmadd_pd(d, w, acc);
        }

        if tail_len != 0 {
            let w_tail = _mm512_maskz_loadu_pd(tail_mask, w_ptr.add(chunks * STEP));
            let d_tail = _mm512_maskz_loadu_pd(tail_mask, data.as_ptr().add(start + chunks * STEP));
            acc = _mm512_fmadd_pd(d_tail, w_tail, acc);
        }

        let res = hsum_pd_zmm(acc) * inv_n;
        *out.get_unchecked_mut(i) = res;
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn alma_row_avx512_long(
    data: &[f64],
    first: usize,
    period: usize,
    w_ptr: *const f64,
    inv_n: f64,
    out: &mut [f64],
) {
    const STEP: usize = 8;
    let n_chunks = period / STEP;
    let tail_len = period % STEP;
    let tmask: __mmask8 = (1u8 << tail_len).wrapping_sub(1);

    const MAX_CHUNKS: usize = 512;
    debug_assert!(n_chunks + (tail_len != 0) as usize <= MAX_CHUNKS);

    let mut wregs: [core::mem::MaybeUninit<__m512d>; MAX_CHUNKS] =
        core::mem::MaybeUninit::uninit().assume_init();

    for blk in 0..n_chunks {
        wregs[blk]
            .as_mut_ptr()
            .write(_mm512_load_pd(w_ptr.add(blk * STEP)));
    }
    if tail_len != 0 {
        wregs[n_chunks]
            .as_mut_ptr()
            .write(_mm512_maskz_loadu_pd(tmask, w_ptr.add(n_chunks * STEP)));
    }

    let wregs: &[__m512d] = core::slice::from_raw_parts(
        wregs.as_ptr() as *const __m512d,
        n_chunks + (tail_len != 0) as usize,
    );

    if tail_len == 0 {
        long_kernel_no_tail(data, first, n_chunks, wregs, inv_n, out);
    } else {
        long_kernel_with_tail(data, first, n_chunks, tail_len, tmask, wregs, inv_n, out);
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn long_kernel_no_tail(
    data: &[f64],
    first: usize,
    n_chunks: usize,
    wregs: &[__m512d],
    inv_n: f64,
    out: &mut [f64],
) {
    const STEP: usize = 8;
    let paired = n_chunks & !3;

    let mut data_ptr = data.as_ptr().add(first);
    let stop_ptr = data.as_ptr().add(data.len());
    let mut dst_ptr = out.as_mut_ptr().add(first + n_chunks * STEP - 1);

    while data_ptr < stop_ptr {
        let mut s0 = _mm512_setzero_pd();
        let mut s1 = _mm512_setzero_pd();
        let mut s2 = _mm512_setzero_pd();
        let mut s3 = _mm512_setzero_pd();

        let mut blk = 0;
        while blk < paired {
            let d0 = _mm512_loadu_pd(data_ptr.add((blk + 0) * STEP));
            let d1 = _mm512_loadu_pd(data_ptr.add((blk + 1) * STEP));
            let d2 = _mm512_loadu_pd(data_ptr.add((blk + 2) * STEP));
            let d3 = _mm512_loadu_pd(data_ptr.add((blk + 3) * STEP));

            s0 = _mm512_fmadd_pd(d0, *wregs.get_unchecked(blk + 0), s0);
            s1 = _mm512_fmadd_pd(d1, *wregs.get_unchecked(blk + 1), s1);
            s2 = _mm512_fmadd_pd(d2, *wregs.get_unchecked(blk + 2), s2);
            s3 = _mm512_fmadd_pd(d3, *wregs.get_unchecked(blk + 3), s3);

            blk += 4;
        }

        for r in blk..n_chunks {
            let d = _mm512_loadu_pd(data_ptr.add(r * STEP));
            s0 = _mm512_fmadd_pd(d, *wregs.get_unchecked(r), s0);
        }

        let sum = _mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3));
        let res = hsum_pd_zmm(sum) * inv_n;

        *dst_ptr = res;

        data_ptr = data_ptr.add(1);
        dst_ptr = dst_ptr.add(1);
        if data_ptr.add(n_chunks * STEP) > stop_ptr {
            break;
        }
    }
}

#[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,fma")]
unsafe fn long_kernel_with_tail(
    data: &[f64],
    first: usize,
    n_chunks: usize,
    tail_len: usize,
    tmask: __mmask8,
    wregs: &[__m512d],
    inv_n: f64,
    out: &mut [f64],
) {
    const STEP: usize = 8;
    let paired = n_chunks & !3;

    let w_tail = *wregs.get_unchecked(n_chunks);

    let mut data_ptr = data.as_ptr().add(first);
    let stop_ptr = data.as_ptr().add(data.len());
    let mut dst_ptr = out.as_mut_ptr().add(first + n_chunks * STEP + tail_len - 1);

    while data_ptr < stop_ptr {
        let mut s0 = _mm512_setzero_pd();
        let mut s1 = _mm512_setzero_pd();
        let mut s2 = _mm512_setzero_pd();
        let mut s3 = _mm512_setzero_pd();

        let mut blk = 0;
        while blk < paired {
            let d0 = _mm512_loadu_pd(data_ptr.add((blk + 0) * STEP));
            let d1 = _mm512_loadu_pd(data_ptr.add((blk + 1) * STEP));
            let d2 = _mm512_loadu_pd(data_ptr.add((blk + 2) * STEP));
            let d3 = _mm512_loadu_pd(data_ptr.add((blk + 3) * STEP));

            s0 = _mm512_fmadd_pd(d0, *wregs.get_unchecked(blk + 0), s0);
            s1 = _mm512_fmadd_pd(d1, *wregs.get_unchecked(blk + 1), s1);
            s2 = _mm512_fmadd_pd(d2, *wregs.get_unchecked(blk + 2), s2);
            s3 = _mm512_fmadd_pd(d3, *wregs.get_unchecked(blk + 3), s3);

            blk += 4;
        }

        for r in blk..n_chunks {
            let d = _mm512_loadu_pd(data_ptr.add(r * STEP));
            s0 = _mm512_fmadd_pd(d, *wregs.get_unchecked(r), s0);
        }

        let d_tail = _mm512_maskz_loadu_pd(tmask, data_ptr.add(n_chunks * STEP));
        s0 = _mm512_fmadd_pd(d_tail, w_tail, s0);

        let sum = _mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3));
        let res = hsum_pd_zmm(sum) * inv_n;

        *dst_ptr = res;

        data_ptr = data_ptr.add(1);
        dst_ptr = dst_ptr.add(1);
        if data_ptr.add(n_chunks * STEP + tail_len) > stop_ptr {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skip_if_unsupported;
    use crate::utilities::data_loader::read_candles_from_csv;
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    fn check_alma_partial_params(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let default_params = AlmaParams {
            period: None,
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_candles(&candles, "close", default_params);
        let output = alma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_alma_accuracy(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AlmaInput::from_candles(&candles, "close", AlmaParams::default());
        let result = alma_with_kernel(&input, kernel)?;
        let expected_last_five = [
            59286.72216704,
            59273.53428138,
            59204.37290721,
            59155.93381742,
            59026.92526112,
        ];
        let start = result.values.len().saturating_sub(5);
        for (i, &val) in result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] ALMA {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_alma_default_candles(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AlmaInput::with_default_candles(&candles);
        match input.data {
            AlmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected AlmaData::Candles"),
        }
        let output = alma_with_kernel(&input, kernel)?;
        assert_eq!(output.values.len(), candles.close.len());

        Ok(())
    }

    fn check_alma_zero_period(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let input_data = [10.0, 20.0, 30.0];
        let params = AlmaParams {
            period: Some(0),
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_slice(&input_data, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA should fail with zero period",
            test_name
        );
        Ok(())
    }

    fn check_alma_period_exceeds_length(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data_small = [10.0, 20.0, 30.0];
        let params = AlmaParams {
            period: Some(10),
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_slice(&data_small, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA should fail with period exceeding length",
            test_name
        );
        Ok(())
    }

    fn check_alma_very_small_dataset(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let single_point = [42.0];
        let params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_slice(&single_point, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            res.is_err(),
            "[{}] ALMA should fail with insufficient data",
            test_name
        );
        Ok(())
    }

    fn check_alma_empty_input(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let empty: [f64; 0] = [];
        let input = AlmaInput::from_slice(&empty, AlmaParams::default());
        let res = alma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(AlmaError::EmptyInputData)),
            "[{}] ALMA should fail with empty input",
            test_name
        );
        Ok(())
    }

    fn check_alma_invalid_sigma(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0];
        let params = AlmaParams {
            period: Some(2),
            offset: None,
            sigma: Some(0.0),
        };
        let input = AlmaInput::from_slice(&data, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(AlmaError::InvalidSigma { .. })),
            "[{}] ALMA should fail with invalid sigma",
            test_name
        );
        Ok(())
    }

    fn check_alma_invalid_offset(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let data = [1.0, 2.0, 3.0];
        let params = AlmaParams {
            period: Some(2),
            offset: Some(f64::NAN),
            sigma: None,
        };
        let input = AlmaInput::from_slice(&data, params);
        let res = alma_with_kernel(&input, kernel);
        assert!(
            matches!(res, Err(AlmaError::InvalidOffset { .. })),
            "[{}] ALMA should fail with invalid offset",
            test_name
        );
        Ok(())
    }

    fn check_alma_reinput(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let first_params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let first_input = AlmaInput::from_candles(&candles, "close", first_params);
        let first_result = alma_with_kernel(&first_input, kernel)?;

        let second_params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let second_input = AlmaInput::from_slice(&first_result.values, second_params);
        let second_result = alma_with_kernel(&second_input, kernel)?;

        assert_eq!(second_result.values.len(), first_result.values.len());
        let expected_last_five = [
            59140.73195170,
            59211.58090986,
            59238.16030697,
            59222.63528822,
            59165.14427332,
        ];
        let start = second_result.values.len().saturating_sub(5);
        for (i, &val) in second_result.values[start..].iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-8,
                "[{}] ALMA Slice Reinput {:?} mismatch at idx {}: got {}, expected {}",
                test_name,
                kernel,
                i,
                val,
                expected_last_five[i]
            );
        }
        Ok(())
    }

    fn check_alma_nan_handling(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let input = AlmaInput::from_candles(
            &candles,
            "close",
            AlmaParams {
                period: Some(9),
                offset: None,
                sigma: None,
            },
        );
        let res = alma_with_kernel(&input, kernel)?;
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

    fn check_alma_streaming(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let period = 9;
        let offset = 0.85;
        let sigma = 6.0;

        let input = AlmaInput::from_candles(
            &candles,
            "close",
            AlmaParams {
                period: Some(period),
                offset: Some(offset),
                sigma: Some(sigma),
            },
        );
        let batch_output = alma_with_kernel(&input, kernel)?.values;

        let mut stream = AlmaStream::try_new(AlmaParams {
            period: Some(period),
            offset: Some(offset),
            sigma: Some(sigma),
        })?;

        let mut stream_values = Vec::with_capacity(candles.close.len());
        for &price in &candles.close {
            match stream.update(price) {
                Some(alma_val) => stream_values.push(alma_val),
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
                diff < 1e-9,
                "[{}] ALMA streaming f64 mismatch at idx {}: batch={}, stream={}, diff={}",
                test_name,
                i,
                b,
                s,
                diff
            );
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_alma_no_poison(test_name: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test_name);

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;

        let test_params = vec![
            AlmaParams::default(),
            AlmaParams {
                period: Some(5),
                offset: Some(0.5),
                sigma: Some(3.0),
            },
            AlmaParams {
                period: Some(5),
                offset: Some(0.85),
                sigma: Some(6.0),
            },
            AlmaParams {
                period: Some(5),
                offset: Some(1.0),
                sigma: Some(10.0),
            },
            AlmaParams {
                period: Some(9),
                offset: Some(0.2),
                sigma: Some(4.0),
            },
            AlmaParams {
                period: Some(9),
                offset: Some(0.85),
                sigma: Some(6.0),
            },
            AlmaParams {
                period: Some(9),
                offset: Some(0.95),
                sigma: Some(8.0),
            },
            AlmaParams {
                period: Some(20),
                offset: Some(0.0),
                sigma: Some(2.0),
            },
            AlmaParams {
                period: Some(20),
                offset: Some(0.5),
                sigma: Some(5.0),
            },
            AlmaParams {
                period: Some(20),
                offset: Some(0.85),
                sigma: Some(6.0),
            },
            AlmaParams {
                period: Some(20),
                offset: Some(1.0),
                sigma: Some(10.0),
            },
            AlmaParams {
                period: Some(2),
                offset: Some(0.0),
                sigma: Some(0.1),
            },
            AlmaParams {
                period: Some(50),
                offset: Some(0.5),
                sigma: Some(15.0),
            },
            AlmaParams {
                period: Some(100),
                offset: Some(0.85),
                sigma: Some(20.0),
            },
        ];

        for (param_idx, params) in test_params.iter().enumerate() {
            let input = AlmaInput::from_candles(&candles, "close", params.clone());
            let output = alma_with_kernel(&input, kernel)?;

            for (i, &val) in output.values.iter().enumerate() {
                if val.is_nan() {
                    continue;
                }

                let bits = val.to_bits();

                if bits == 0x11111111_11111111 {
                    panic!(
                        "[{}] Found alloc_with_nan_prefix poison value {} (0x{:016X}) at index {} \
                        with params: period={}, offset={}, sigma={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(9),
                        params.offset.unwrap_or(0.85),
                        params.sigma.unwrap_or(6.0)
                    );
                }

                if bits == 0x22222222_22222222 {
                    panic!(
                        "[{}] Found init_matrix_prefixes poison value {} (0x{:016X}) at index {} \
                        with params: period={}, offset={}, sigma={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(9),
                        params.offset.unwrap_or(0.85),
                        params.sigma.unwrap_or(6.0)
                    );
                }

                if bits == 0x33333333_33333333 {
                    panic!(
                        "[{}] Found make_uninit_matrix poison value {} (0x{:016X}) at index {} \
                        with params: period={}, offset={}, sigma={}",
                        test_name,
                        val,
                        bits,
                        i,
                        params.period.unwrap_or(9),
                        params.offset.unwrap_or(0.85),
                        params.sigma.unwrap_or(6.0)
                    );
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_alma_no_poison(_test_name: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    #[cfg(feature = "proptest")]
    #[allow(clippy::float_cmp)]
    fn check_alma_property(
        test_name: &str,
        kernel: Kernel,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use proptest::prelude::*;
        skip_if_unsupported!(kernel, test_name);

        let strat = (1usize..=64).prop_flat_map(|period| {
            (
                prop::collection::vec(
                    (-1e6f64..1e6f64).prop_filter("finite", |x| x.is_finite()),
                    period..400,
                ),
                Just(period),
                0f64..1f64,
                0.1f64..10.0f64,
            )
        });

        proptest::test_runner::TestRunner::default()
            .run(&strat, |(data, period, offset, sigma)| {
                let params = AlmaParams {
                    period: Some(period),
                    offset: Some(offset),
                    sigma: Some(sigma),
                };
                let input = AlmaInput::from_slice(&data, params);

                let AlmaOutput { values: out } = alma_with_kernel(&input, kernel).unwrap();
                let AlmaOutput { values: ref_out } =
                    alma_with_kernel(&input, Kernel::Scalar).unwrap();

                for i in (period - 1)..data.len() {
                    let window = &data[i + 1 - period..=i];
                    let lo = window.iter().cloned().fold(f64::INFINITY, f64::min);
                    let hi = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let y = out[i];
                    let r = ref_out[i];

                    prop_assert!(
                        y.is_nan() || (y >= lo - 1e-9 && y <= hi + 1e-9),
                        "idx {i}: {y} ∉ [{lo}, {hi}]"
                    );

                    if period == 1 {
                        prop_assert!((y - data[i]).abs() <= f64::EPSILON);
                    }

                    if data.windows(2).all(|w| w[0] == w[1]) {
                        prop_assert!((y - data[0]).abs() <= 1e-9);
                    }

                    let y_bits = y.to_bits();
                    let r_bits = r.to_bits();

                    if !y.is_finite() || !r.is_finite() {
                        prop_assert!(
                            y.to_bits() == r.to_bits(),
                            "finite/NaN mismatch idx {i}: {y} vs {r}"
                        );
                        continue;
                    }

                    let ulp_diff: u64 = y_bits.abs_diff(r_bits);

                    prop_assert!(
                        (y - r).abs() <= 1e-9 || ulp_diff <= 4,
                        "mismatch idx {i}: {y} vs {r} (ULP={ulp_diff})"
                    );
                }
                Ok(())
            })
            .unwrap();

        Ok(())
    }

    macro_rules! generate_all_alma_tests {
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
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                $(
                    #[test]
                    fn [<$test_fn _simd128_f64>]() {
                        let _ = $test_fn(stringify!([<$test_fn _simd128_f64>]), Kernel::Scalar);
                    }
                )*
            }
        }
    }

    generate_all_alma_tests!(
        check_alma_partial_params,
        check_alma_accuracy,
        check_alma_default_candles,
        check_alma_zero_period,
        check_alma_period_exceeds_length,
        check_alma_very_small_dataset,
        check_alma_empty_input,
        check_alma_invalid_sigma,
        check_alma_invalid_offset,
        check_alma_reinput,
        check_alma_nan_handling,
        check_alma_streaming,
        check_alma_no_poison
    );

    #[cfg(feature = "proptest")]
    generate_all_alma_tests!(check_alma_property);

    #[cfg(not(feature = "wasm"))]
    #[test]
    fn test_alma_into_matches_api() -> Result<(), Box<dyn Error>> {
        // Build a small but non-trivial input with an initial NaN prefix
        let mut data = vec![f64::NAN; 3];
        data.extend((0..256).map(|i| (i as f64).sin() * 100.0 + (i as f64) * 0.1));

        let input = AlmaInput::from_slice(&data, AlmaParams::default());

        // Baseline via Vec API
        let baseline = alma_with_kernel(&input, Kernel::Auto)?.values;

        // Preallocate output and compute via into API
        let mut out = vec![0.0; data.len()];
        alma_into(&input, &mut out)?;

        assert_eq!(baseline.len(), out.len());

        fn eq_or_both_nan(a: f64, b: f64) -> bool {
            (a.is_nan() && b.is_nan()) || (a == b) || ((a - b).abs() <= 1e-12)
        }

        for i in 0..out.len() {
            assert!(
                eq_or_both_nan(baseline[i], out[i]),
                "mismatch at {}: baseline={} out={}",
                i,
                baseline[i],
                out[i]
            );
        }

        Ok(())
    }

    fn check_batch_default_row(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = AlmaBatchBuilder::new()
            .kernel(kernel)
            .apply_candles(&c, "close")?;

        let def = AlmaParams::default();
        let row = output.values_for(&def).expect("default row missing");

        assert_eq!(row.len(), c.close.len());

        let expected = [
            59286.72216704,
            59273.53428138,
            59204.37290721,
            59155.93381742,
            59026.92526112,
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

    fn check_batch_sweep(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let output = AlmaBatchBuilder::new()
            .kernel(kernel)
            .period_range(9, 20, 1)
            .offset_range(0.5, 1.0, 0.1)
            .sigma_range(3.0, 9.0, 1.0)
            .apply_candles(&c, "close")?;

        let expected_combos = 12 * 6 * 7;
        assert_eq!(output.combos.len(), expected_combos);
        assert_eq!(output.rows, expected_combos);
        assert_eq!(output.cols, c.close.len());

        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_batch_no_poison(test: &str, kernel: Kernel) -> Result<(), Box<dyn Error>> {
        skip_if_unsupported!(kernel, test);

        let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let c = read_candles_from_csv(file)?;

        let test_configs = vec![
            (2, 10, 2, 0.0, 1.0, 0.2, 1.0, 10.0, 3.0),
            (5, 25, 5, 0.85, 0.85, 0.0, 6.0, 6.0, 0.0),
            (10, 10, 0, 0.0, 1.0, 0.1, 5.0, 5.0, 0.0),
            (2, 5, 1, 0.5, 0.5, 0.0, 3.0, 8.0, 1.0),
            (30, 60, 15, 0.85, 0.85, 0.0, 6.0, 6.0, 0.0),
            (9, 15, 3, 0.8, 0.9, 0.1, 6.0, 8.0, 2.0),
            (8, 12, 1, 0.7, 0.9, 0.05, 4.0, 8.0, 0.5),
        ];

        for (cfg_idx, &(p_start, p_end, p_step, o_start, o_end, o_step, s_start, s_end, s_step)) in
            test_configs.iter().enumerate()
        {
            let output = AlmaBatchBuilder::new()
                .kernel(kernel)
                .period_range(p_start, p_end, p_step)
                .offset_range(o_start, o_end, o_step)
                .sigma_range(s_start, s_end, s_step)
                .apply_candles(&c, "close")?;

            for (idx, &val) in output.values.iter().enumerate() {
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
                        at row {} col {} (flat index {}) with params: period={}, offset={}, sigma={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(9),
						combo.offset.unwrap_or(0.85),
						combo.sigma.unwrap_or(6.0)
					);
                }

                if bits == 0x22222222_22222222 {
                    panic!(
						"[{}] Config {}: Found init_matrix_prefixes poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}, offset={}, sigma={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(9),
						combo.offset.unwrap_or(0.85),
						combo.sigma.unwrap_or(6.0)
					);
                }

                if bits == 0x33333333_33333333 {
                    panic!(
						"[{}] Config {}: Found make_uninit_matrix poison value {} (0x{:016X}) \
                        at row {} col {} (flat index {}) with params: period={}, offset={}, sigma={}",
						test,
						cfg_idx,
						val,
						bits,
						row,
						col,
						idx,
						combo.period.unwrap_or(9),
						combo.offset.unwrap_or(0.85),
						combo.sigma.unwrap_or(6.0)
					);
                }
            }
        }

        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn check_batch_no_poison(_test: &str, _kernel: Kernel) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    gen_batch_tests!(check_batch_default_row);
    gen_batch_tests!(check_batch_sweep);
    gen_batch_tests!(check_batch_no_poison);

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn test_alma_simd128_correctness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let period = 5;
        let offset = 0.85;
        let sigma = 6.0;

        // Compute with scalar version
        let params = AlmaParams {
            period: Some(period),
            offset: Some(offset),
            sigma: Some(sigma),
        };
        let input = AlmaInput::from_slice(&data, params);
        let scalar_output = alma_with_kernel(&input, Kernel::Scalar).unwrap();

        // Compute with SIMD128 (via Scalar kernel on WASM)
        let simd128_output = alma_with_kernel(&input, Kernel::Scalar).unwrap();

        // Compare results
        assert_eq!(scalar_output.values.len(), simd128_output.values.len());
        for (i, (scalar_val, simd_val)) in scalar_output
            .values
            .iter()
            .zip(simd128_output.values.iter())
            .enumerate()
        {
            assert!(
                (scalar_val - simd_val).abs() < 1e-10,
                "SIMD128 mismatch at index {}: scalar={}, simd128={}",
                i,
                scalar_val,
                simd_val
            );
        }
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "alma")]
#[pyo3(signature = (data, period, offset, sigma, kernel=None))]

pub fn alma_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period: usize,
    offset: f64,
    sigma: f64,
    kernel: Option<&str>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};

    let kern = validate_kernel(kernel, false)?;
    let params = AlmaParams {
        period: Some(period),
        offset: Some(offset),
        sigma: Some(sigma),
    };
    // Prefer zero-copy for contiguous input; fallback to a minimal copy for non-contiguous views.
    let result_vec: Vec<f64> = if let Ok(slice_in) = data.as_slice() {
        let alma_in = AlmaInput::from_slice(slice_in, params);
        py.allow_threads(|| alma_with_kernel(&alma_in, kern).map(|o| o.values))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    } else {
        // Non-contiguous (e.g., a column view) — copy just once.
        let owned = data.as_array().to_owned();
        let slice_in = owned.as_slice().expect("owned array should be contiguous");
        let alma_in = AlmaInput::from_slice(slice_in, params);
        let out = py
            .allow_threads(|| alma_with_kernel(&alma_in, kern).map(|o| o.values))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        out
    };

    Ok(result_vec.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pyclass(name = "AlmaStream")]
pub struct AlmaStreamPy {
    stream: AlmaStream,
}

#[cfg(feature = "python")]
#[pymethods]
impl AlmaStreamPy {
    #[new]
    fn new(period: usize, offset: f64, sigma: f64) -> PyResult<Self> {
        let params = AlmaParams {
            period: Some(period),
            offset: Some(offset),
            sigma: Some(sigma),
        };
        let stream =
            AlmaStream::try_new(params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(AlmaStreamPy { stream })
    }

    fn update(&mut self, value: f64) -> Option<f64> {
        self.stream.update(value)
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "alma_batch")]
#[pyo3(signature = (data, period_range, offset_range, sigma_range, kernel=None))]

pub fn alma_batch_py<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray1<'py, f64>,
    period_range: (usize, usize, usize),
    offset_range: (f64, f64, f64),
    sigma_range: (f64, f64, f64),
    kernel: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
    use pyo3::types::PyDict;

    let slice_in = data.as_slice()?;

    let sweep = AlmaBatchRange {
        period: period_range,
        offset: offset_range,
        sigma: sigma_range,
    };

    let combos = expand_grid(&sweep).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = combos.len();
    let cols = slice_in.len();

    let out_arr = unsafe { PyArray1::<f64>::new(py, [rows * cols], false) };
    let slice_out = unsafe { out_arr.as_slice_mut()? };

    let kern = validate_kernel(kernel, true)?;

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
            alma_batch_inner_into(slice_in, &sweep, simd, true, slice_out)
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
    dict.set_item(
        "offsets",
        combos
            .iter()
            .map(|p| p.offset.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    dict.set_item(
        "sigmas",
        combos
            .iter()
            .map(|p| p.sigma.unwrap())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    Ok(dict)
}

// ==================== PYTHON: CUDA BINDINGS (zero-copy) ====================
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "alma_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, offset_range, sigma_range, device_id=0))]
pub fn alma_cuda_batch_dev_py(
    py: Python<'_>,
    data_f32: numpy::PyReadonlyArray1<'_, f32>,
    period_range: (usize, usize, usize),
    offset_range: (f64, f64, f64),
    sigma_range: (f64, f64, f64),
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::moving_averages::CudaAlma;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let slice_in: &[f32] = data_f32.as_slice()?;
    let sweep = AlmaBatchRange {
        period: period_range,
        offset: offset_range,
        sigma: sigma_range,
    };

    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaAlma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = device_id as u32;
        cuda
            .alma_batch_dev(slice_in, &sweep)
            .map(|inner| (inner, ctx, dev_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner, _ctx: Some(ctx), device_id: Some(dev_id) })
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "alma_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, offset, sigma, device_id=0))]
pub fn alma_cuda_many_series_one_param_dev_py(
    py: Python<'_>,
    data_tm_f32: numpy::PyReadonlyArray2<'_, f32>,
    period: usize,
    offset: f64,
    sigma: f64,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use crate::cuda::cuda_available;
    use crate::cuda::moving_averages::CudaAlma;
    use numpy::PyUntypedArrayMethods;

    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }

    let flat_in: &[f32] = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let params = AlmaParams {
        period: Some(period),
        offset: Some(offset),
        sigma: Some(sigma),
    };

    let (inner, ctx, dev_id) = py.allow_threads(|| {
        let cuda = CudaAlma::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = cuda.context_arc();
        let dev_id = device_id as u32;
        cuda
            .alma_multi_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map(|inner| (inner, ctx, dev_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;

    Ok(DeviceArrayF32Py { inner, _ctx: Some(ctx), device_id: Some(dev_id) })
}

// FP32-only, user-supplied output (zero extra copies)

#[cfg(feature = "python")]
pub fn register_alma_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    // CPU
    m.add_function(wrap_pyfunction!(alma_py, m)?)?;
    m.add_function(wrap_pyfunction!(alma_batch_py, m)?)?;
    m.add_class::<AlmaStreamPy>()?;

    // CUDA (feature-gated at compile time)
    #[cfg(feature = "cuda")]
    {
        m.add_class::<DeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(alma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(alma_cuda_many_series_one_param_dev_py, m)?)?;
    }
    Ok(())
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alma_js(data: &[f64], period: usize, offset: f64, sigma: f64) -> Result<Vec<f64>, JsValue> {
    let params = AlmaParams {
        period: Some(period),
        offset: Some(offset),
        sigma: Some(sigma),
    };
    let input = AlmaInput::from_slice(data, params);

    let mut output = vec![0.0; data.len()];

    alma_into_slice(&mut output, &input, detect_best_kernel())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(output)
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AlmaBatchConfig {
    pub period_range: (usize, usize, usize),
    pub offset_range: (f64, f64, f64),
    pub sigma_range: (f64, f64, f64),
}

#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct AlmaBatchJsOutput {
    pub values: Vec<f64>,
    pub combos: Vec<AlmaParams>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = alma_batch)]
pub fn alma_batch_unified_js(data: &[f64], config: JsValue) -> Result<JsValue, JsValue> {
    let config: AlmaBatchConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let sweep = AlmaBatchRange {
        period: config.period_range,
        offset: config.offset_range,
        sigma: config.sigma_range,
    };

    let output = alma_batch_inner(data, &sweep, detect_best_kernel(), false)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let js_output = AlmaBatchJsOutput {
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
pub fn alma_alloc(len: usize) -> *mut f64 {
    let mut vec = Vec::<f64>::with_capacity(len);
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alma_free(ptr: *mut f64, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alma_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period: usize,
    offset: f64,
    sigma: f64,
) -> Result<(), JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to alma_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        if period == 0 || period > len {
            return Err(JsValue::from_str("Invalid period"));
        }

        let params = AlmaParams {
            period: Some(period),
            offset: Some(offset),
            sigma: Some(sigma),
        };
        let input = AlmaInput::from_slice(data, params);

        if in_ptr == out_ptr {
            let mut temp = vec![0.0; len];
            alma_into_slice(&mut temp, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            out.copy_from_slice(&temp);
        } else {
            let out = std::slice::from_raw_parts_mut(out_ptr, len);
            alma_into_slice(out, &input, detect_best_kernel())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[deprecated(
    since = "1.0.0",
    note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
)]
pub struct AlmaContext {
    weights: AVec<f64>,
    inv_norm: f64,
    period: usize,
    first: usize,
    kernel: Kernel,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(deprecated)]
impl AlmaContext {
    #[wasm_bindgen(constructor)]
    #[deprecated(
        since = "1.0.0",
        note = "For weight reuse patterns, use the fast/unsafe API with persistent buffers"
    )]
    pub fn new(period: usize, offset: f64, sigma: f64) -> Result<AlmaContext, JsValue> {
        if period == 0 {
            return Err(JsValue::from_str("Invalid period: 0"));
        }
        if !(0.0..=1.0).contains(&offset) || offset.is_nan() || offset.is_infinite() {
            return Err(JsValue::from_str(&format!("Invalid offset: {}", offset)));
        }
        if sigma <= 0.0 {
            return Err(JsValue::from_str(&format!("Invalid sigma: {}", sigma)));
        }

        let m = offset * (period - 1) as f64;
        let s = period as f64 / sigma;
        let s2 = 2.0 * s * s;

        let mut weights: AVec<f64> = AVec::with_capacity(CACHELINE_ALIGN, period);
        weights.resize(period, 0.0);
        let mut norm = 0.0;

        for i in 0..period {
            let w = (-(i as f64 - m).powi(2) / s2).exp();
            weights[i] = w;
            norm += w;
        }

        let inv_norm = 1.0 / norm;

        Ok(AlmaContext {
            weights,
            inv_norm,
            period,
            first: 0,
            kernel: detect_best_kernel(),
        })
    }

    pub fn update_into(
        &self,
        in_ptr: *const f64,
        out_ptr: *mut f64,
        len: usize,
    ) -> Result<(), JsValue> {
        if len < self.period {
            return Err(JsValue::from_str("Data length less than period"));
        }

        unsafe {
            let data = std::slice::from_raw_parts(in_ptr, len);
            let out = std::slice::from_raw_parts_mut(out_ptr, len);

            let first = data.iter().position(|x| !x.is_nan()).unwrap_or(0);

            if in_ptr == out_ptr {
                let mut temp = vec![0.0; len];
                alma_compute_into(
                    data,
                    self.weights.as_slice(),
                    self.period,
                    first,
                    self.inv_norm,
                    self.kernel,
                    &mut temp,
                );

                out.copy_from_slice(&temp);
            } else {
                alma_compute_into(
                    data,
                    self.weights.as_slice(),
                    self.period,
                    first,
                    self.inv_norm,
                    self.kernel,
                    out,
                );
            }

            for i in 0..(first + self.period - 1) {
                out[i] = f64::NAN;
            }
        }

        Ok(())
    }

    pub fn get_warmup_period(&self) -> usize {
        self.period - 1
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn alma_batch_into(
    in_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    period_start: usize,
    period_end: usize,
    period_step: usize,
    offset_start: f64,
    offset_end: f64,
    offset_step: f64,
    sigma_start: f64,
    sigma_end: f64,
    sigma_step: f64,
) -> Result<usize, JsValue> {
    if in_ptr.is_null() || out_ptr.is_null() {
        return Err(JsValue::from_str("null pointer passed to alma_batch_into"));
    }

    unsafe {
        let data = std::slice::from_raw_parts(in_ptr, len);

        let sweep = AlmaBatchRange {
            period: (period_start, period_end, period_step),
            offset: (offset_start, offset_end, offset_step),
            sigma: (sigma_start, sigma_end, sigma_step),
        };

        let combos = expand_grid(&sweep)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let rows = combos.len();
        let cols = len;
        let total = rows.checked_mul(cols).ok_or_else(|| JsValue::from_str("rows*cols overflow"))?;

        let out = std::slice::from_raw_parts_mut(out_ptr, total);

        alma_batch_inner_into(data, &sweep, detect_best_kernel(), false, out)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(rows)
    }
}
