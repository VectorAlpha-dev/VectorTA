#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::exceptions::PyValueError;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::prelude::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::types::{PyAny, PyDict};
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::Bound;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::DeviceArrayF32;

/// Shared helper to export a 2D `DeviceBuffer<f32>` as a CUDA DLPack capsule.
///
/// - `rows`, `cols`: logical 2D shape (row-major, contiguous).
/// - `device_id`: CUDA device ordinal where the buffer is allocated.
/// - `max_version`: optional `(major, minor)` version hint per Array API.
#[cfg(all(feature = "python", feature = "cuda"))]
pub fn export_f32_cuda_dlpack_2d<'py>(
    py: Python<'py>,
    buf: DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
    device_id: i32,
    max_version: Option<Bound<'py, PyAny>>,
) -> PyResult<PyObject> {
    use std::ffi::c_void;
    use std::os::raw::c_char;
    use std::ptr::null_mut;

    #[repr(C)]
    struct DLDataType {
        code: u8,
        bits: u8,
        lanes: u16,
    }
    #[repr(C)]
    struct DLDevice {
        device_type: i32,
        device_id: i32,
    }
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
    struct DLPackVersion {
        major: u32,
        minor: u32,
    }
    #[repr(C)]
    struct DLManagedTensorVersioned {
        version: DLPackVersion,
        manager_ctx: *mut c_void,
        deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
        flags: u64,
        dl_tensor: DLTensor,
    }
    #[repr(C)]
    struct DLManagedTensor {
        dl_tensor: DLTensor,
        manager_ctx: *mut c_void,
        deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
    }

    // Decide whether to use versioned capsules based on negotiation.
    let use_versioned = max_version
        .as_ref()
        .and_then(|t| t.extract::<(i32, i32)>().ok())
        .map(|(maj, _)| maj >= 1)
        .unwrap_or(false);

    // Retain primary context for allocation device so frees are safe even if the
    // original Rust-side context has been dropped.
    let mut retained: cust::sys::CUcontext = null_mut();

    // Build common tensor state (2-D, contiguous, row-major).
    let rows_i64 = rows as i64;
    let cols_i64 = cols as i64;
    let size = rows
        .checked_mul(cols)
        .ok_or_else(|| PyValueError::new_err("rows*cols overflow in DLPack export"))?;

    let data_ptr = if size == 0 {
        std::ptr::null_mut()
    } else {
        buf.as_device_ptr().as_raw() as *mut c_void
    };
    let mut shape = Box::new([rows_i64, cols_i64]);
    let mut strides = Box::new([cols_i64, 1_i64]);

    struct Manager {
        ctx: cust::sys::CUcontext,
        device_id: i32,
        _buf: DeviceBuffer<f32>,
        _shape: Box<[i64; 2]>,
        _strides: Box<[i64; 2]>,
    }

    unsafe extern "C" fn deleter_v1(p: *mut DLManagedTensorVersioned) {
        if p.is_null() {
            return;
        }
        let mgr = (*p).manager_ctx as *mut Manager;
        if !mgr.is_null() {
            let ctx = (*mgr).ctx;
            let dev_id = (*mgr).device_id as cust::sys::CUdevice;
            if !ctx.is_null() {
                let mut old: cust::sys::CUcontext = null_mut();
                let _ = cust::sys::cuCtxPushCurrent_v2(ctx);
                let _boxed: Box<Manager> = Box::from_raw(mgr);
                let _ = cust::sys::cuCtxPopCurrent_v2(&mut old as *mut _);
                let _ = cust::sys::cuDevicePrimaryCtxRelease_v2(dev_id);
            } else {
                let _boxed: Box<Manager> = Box::from_raw(mgr);
            }
        }
        let _ = Box::from_raw(p);
    }

    unsafe extern "C" fn deleter_legacy(p: *mut DLManagedTensor) {
        if p.is_null() {
            return;
        }
        let mgr = (*p).manager_ctx as *mut Manager;
        if !mgr.is_null() {
            let ctx = (*mgr).ctx;
            let dev_id = (*mgr).device_id as cust::sys::CUdevice;
            if !ctx.is_null() {
                let mut old: cust::sys::CUcontext = null_mut();
                let _ = cust::sys::cuCtxPushCurrent_v2(ctx);
                let _boxed: Box<Manager> = Box::from_raw(mgr);
                let _ = cust::sys::cuCtxPopCurrent_v2(&mut old as *mut _);
                let _ = cust::sys::cuDevicePrimaryCtxRelease_v2(dev_id);
            } else {
                let _boxed: Box<Manager> = Box::from_raw(mgr);
            }
        }
        let _ = Box::from_raw(p);
    }

    // Capsule destructor following Python DLPack specification:
    // - If capsule has been consumed and renamed to "used_dltensor[_versioned]",
    //   do nothing.
    // - Otherwise, get pointer for "dltensor_versioned" or legacy "dltensor"
    //   and call the DLManagedTensor* deleter.
    unsafe extern "C" fn capsule_dtor(capsule: *mut pyo3::ffi::PyObject) {
        use pyo3::ffi;

        if capsule.is_null() {
            return;
        }

        static DLTENSOR_NAME: &[u8] = b"dltensor\0";
        static USED_DLTENSOR_NAME: &[u8] = b"used_dltensor\0";
        static DLTENSOR_VERSIONED_NAME: &[u8] = b"dltensor_versioned\0";
        static USED_DLTENSOR_VERSIONED_NAME: &[u8] = b"used_dltensor_versioned\0";

        // If capsule was renamed to "used_*", the consumer owns the tensor
        // and will call the DLManagedTensor deleter. Do nothing in that case.
        if ffi::PyCapsule_IsValid(
            capsule,
            USED_DLTENSOR_VERSIONED_NAME.as_ptr() as *const c_char,
        ) == 1
            || ffi::PyCapsule_IsValid(
                capsule,
                USED_DLTENSOR_NAME.as_ptr() as *const c_char,
            ) == 1
        {
            return;
        }

        // Prefer versioned, fall back to legacy.
        let ptr_v = ffi::PyCapsule_GetPointer(
            capsule,
            DLTENSOR_VERSIONED_NAME.as_ptr() as *const c_char,
        );
        if !ptr_v.is_null() {
            let mt = ptr_v as *mut DLManagedTensorVersioned;
            if let Some(del) = (*mt).deleter {
                del(mt);
            }
            return;
        }

        let ptr_l =
            ffi::PyCapsule_GetPointer(capsule, DLTENSOR_NAME.as_ptr() as *const c_char);
        if !ptr_l.is_null() {
            let mt = ptr_l as *mut DLManagedTensor;
            if let Some(del) = (*mt).deleter {
                del(mt);
            }
            return;
        }
    }

    // Prepare manager and capsule contents.
    let mgr = Box::new(Manager {
        ctx: retained,
        device_id,
        _buf: buf,
        _shape: shape,
        _strides: strides,
    });
    let mgr_ptr = Box::into_raw(mgr);

    if use_versioned {
        let mt = Box::new(DLManagedTensorVersioned {
            version: DLPackVersion { major: 1, minor: 2 },
            manager_ctx: mgr_ptr as *mut _,
            deleter: Some(deleter_v1),
            flags: 0,
            dl_tensor: DLTensor {
                data: data_ptr,
                device: DLDevice {
                    device_type: 2,
                    device_id,
                },
                ndim: 2,
                dtype: DLDataType {
                    code: 2,
                    bits: 32,
                    lanes: 1,
                },
                shape: unsafe { (*mgr_ptr)._shape.as_mut_ptr() },
                strides: unsafe { (*mgr_ptr)._strides.as_mut_ptr() },
                byte_offset: 0,
            },
        });
        let raw = Box::into_raw(mt);
        static DLTENSOR_VERSIONED_NAME: &[u8] = b"dltensor_versioned\0";
        let cap = unsafe {
            pyo3::ffi::PyCapsule_New(
                raw as *mut c_void,
                DLTENSOR_VERSIONED_NAME.as_ptr() as *const c_char,
                Some(capsule_dtor),
            )
        };
        if cap.is_null() {
            unsafe { deleter_v1(raw) };
            return Err(PyValueError::new_err("failed to create DLPack capsule"));
        }
        Ok(unsafe { PyObject::from_owned_ptr(py, cap) })
    } else {
        let mt = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr,
                device: DLDevice {
                    device_type: 2,
                    device_id,
                },
                ndim: 2,
                dtype: DLDataType {
                    code: 2,
                    bits: 32,
                    lanes: 1,
                },
                shape: unsafe { (*mgr_ptr)._shape.as_mut_ptr() },
                strides: unsafe { (*mgr_ptr)._strides.as_mut_ptr() },
                byte_offset: 0,
            },
            manager_ctx: mgr_ptr as *mut _,
            deleter: Some(deleter_legacy),
        });
        let raw = Box::into_raw(mt);
        static DLTENSOR_NAME: &[u8] = b"dltensor\0";
        let cap = unsafe {
            pyo3::ffi::PyCapsule_New(
                raw as *mut c_void,
                DLTENSOR_NAME.as_ptr() as *const c_char,
                Some(capsule_dtor),
            )
        };
        if cap.is_null() {
            unsafe { deleter_legacy(raw) };
            return Err(PyValueError::new_err("failed to create DLPack capsule"));
        }
        Ok(unsafe { PyObject::from_owned_ptr(py, cap) })
    }
}

/// Shared CUDA VRAM handle for a single `DeviceArrayF32` matrix, exposing
/// `__cuda_array_interface__` v3 and `__dlpack__` backed by
/// `export_f32_cuda_dlpack_2d`.
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
    pub fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
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

    pub fn __dlpack_device__(&self) -> PyResult<(i32, i32)> {
        // Prefer the explicit device id tracked by the wrapper. This
        // avoids relying on pointer attribute queries that have proven
        // brittle across driver/toolkit combinations.
        if let Some(dev) = self.device_id {
            Ok((2, dev as i32))
        } else {
            // Fallback: query current device if no explicit id is stored.
            let mut device_ordinal: i32 = 0;
            unsafe {
                let _ = cust::sys::cuCtxGetDevice(&mut device_ordinal);
            }
            Ok((2, device_ordinal))
        }
    }

    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    pub fn __dlpack__<'py>(
        &mut self,
        py: Python<'py>,
        stream: Option<pyo3::PyObject>,
        max_version: Option<pyo3::PyObject>,
        dl_device: Option<pyo3::PyObject>,
        copy: Option<pyo3::PyObject>,
    ) -> PyResult<PyObject> {
        // Compute target device id and validate `dl_device` hint if provided.
        let (kdl, alloc_dev) = self.__dlpack_device__()?; // (2, device_id)
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
        let inner = std::mem::replace(
            &mut self.inner,
            DeviceArrayF32 { buf: dummy, rows: 0, cols: 0 },
        );

        let rows = inner.rows;
        let cols = inner.cols;
        let buf = inner.buf;

        let max_version_bound = max_version.map(|obj| obj.into_bound(py));

        export_f32_cuda_dlpack_2d(py, buf, rows, cols, alloc_dev, max_version_bound)
    }
}

/// Helper to wrap a generic `DeviceArrayF32` into the shared Python handle.
/// Context is not retained here; callers that need a primary-context guard
/// should construct `DeviceArrayF32Py` directly.
#[cfg(all(feature = "python", feature = "cuda"))]
pub fn make_device_array_py(
    device_id: usize,
    inner: DeviceArrayF32,
) -> PyResult<DeviceArrayF32Py> {
    Ok(DeviceArrayF32Py {
        inner,
        _ctx: None,
        device_id: Some(device_id as u32),
    })
}
