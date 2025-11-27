#[cfg(all(feature = "python", feature = "cuda"))]
use cust::memory::DeviceBuffer;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::exceptions::PyValueError;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::prelude::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::types::PyAny;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::Bound;

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
    use std::ffi::{c_void, CString};
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
    unsafe {
        let _ = cust::sys::cuDevicePrimaryCtxRetain(&mut retained as *mut _, device_id);
    }

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
            let mut old: cust::sys::CUcontext = null_mut();
            let _ = cust::sys::cuCtxPushCurrent_v2((*mgr).ctx);
            let boxed: Box<Manager> = Box::from_raw(mgr);
            let dev_id = boxed.device_id as cust::sys::CUdevice;
            let _ = cust::sys::cuCtxPopCurrent_v2(&mut old as *mut _);
            let _ = cust::sys::cuDevicePrimaryCtxRelease_v2(dev_id);
        }
        let _ = Box::from_raw(p);
    }

    unsafe extern "C" fn deleter_legacy(p: *mut DLManagedTensor) {
        if p.is_null() {
            return;
        }
        let mgr = (*p).manager_ctx as *mut Manager;
        if !mgr.is_null() {
            let mut old: cust::sys::CUcontext = null_mut();
            let _ = cust::sys::cuCtxPushCurrent_v2((*mgr).ctx);
            let boxed: Box<Manager> = Box::from_raw(mgr);
            let dev_id = boxed.device_id as cust::sys::CUdevice;
            let _ = cust::sys::cuCtxPopCurrent_v2(&mut old as *mut _);
            let _ = cust::sys::cuDevicePrimaryCtxRelease_v2(dev_id);
        }
        let _ = Box::from_raw(p);
    }

    unsafe extern "C" fn capsule_dtor(capsule: *mut pyo3::ffi::PyObject) {
        if capsule.is_null() {
            return;
        }
        let vname = CString::new("dltensor_versioned").unwrap();
        let legacy = CString::new("dltensor").unwrap();

        let ptr_v = pyo3::ffi::PyCapsule_GetPointer(capsule, vname.as_ptr());
        if !ptr_v.is_null() {
            let mt = ptr_v as *mut DLManagedTensorVersioned;
            if let Some(del) = (*mt).deleter {
                del(mt);
            }
            return;
        }
        let ptr_l = pyo3::ffi::PyCapsule_GetPointer(capsule, legacy.as_ptr());
        if !ptr_l.is_null() {
            let mt = ptr_l as *mut DLManagedTensor;
            if let Some(del) = (*mt).deleter {
                del(mt);
            }
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
        let name = CString::new("dltensor_versioned").unwrap();
        let cap = unsafe {
            pyo3::ffi::PyCapsule_New(
                raw as *mut c_void,
                name.as_ptr(),
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
        let name = CString::new("dltensor").unwrap();
        let cap = unsafe {
            pyo3::ffi::PyCapsule_New(
                raw as *mut c_void,
                name.as_ptr(),
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
