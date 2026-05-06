#![cfg_attr(
    all(feature = "nightly-avx", rustc_is_nightly),
    feature(stdarch_x86_avx512)
)]
#![cfg_attr(
    all(feature = "nightly-avx", rustc_is_nightly),
    feature(avx512_target_feature)
)]
#![cfg_attr(all(feature = "nightly-avx", rustc_is_nightly), feature(portable_simd))]
#![cfg_attr(
    all(feature = "nightly-avx", rustc_is_nightly),
    feature(likely_unlikely)
)]
#![allow(warnings)]
#![allow(clippy::needless_range_loop)]

pub mod indicators;
pub mod utilities;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod _rayon_one_big_stack {
    use ctor::ctor;
    use rayon::ThreadPoolBuilder;

    #[ctor]
    fn init_rayon_pool() {
        let _ = ThreadPoolBuilder::new()
            .num_threads(1)
            .stack_size(8 * 1024 * 1024)
            .build_global();
    }
}

pub mod bindings {
    #[cfg(feature = "python")]
    pub mod python;

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub mod wasm;
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use std::{cell::RefCell, collections::HashMap};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
thread_local! {
    static WASM_F64_ALLOCATIONS: RefCell<HashMap<usize, usize>> = RefCell::new(HashMap::new());
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn register_f64_allocation(ptr: *mut f64, cap: usize) -> *mut f64 {
    WASM_F64_ALLOCATIONS.with(|allocations| {
        allocations.borrow_mut().insert(ptr as usize, cap);
    });
    ptr
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
fn take_f64_allocation(ptr: *mut f64) -> Option<usize> {
    WASM_F64_ALLOCATIONS.with(|allocations| allocations.borrow_mut().remove(&(ptr as usize)))
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn allocate_f64_array(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let ptr = v.as_mut_ptr();
    let cap = v.capacity();
    std::mem::forget(v);
    register_f64_allocation(ptr, cap)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn copy_f64_array(values: &[f64]) -> *mut f64 {
    let mut v = values.to_vec();
    let ptr = v.as_mut_ptr();
    let cap = v.capacity();
    std::mem::forget(v);
    register_f64_allocation(ptr, cap)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn deallocate_f64_array(ptr: *mut f64) {
    if ptr.is_null() {
        return;
    }
    if let Some(cap) = take_f64_allocation(ptr) {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, cap);
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn read_f64_array(ptr: *const f64, len: usize) -> Vec<f64> {
    if len == 0 {
        return Vec::new();
    }
    assert!(!ptr.is_null(), "read_f64_array: null pointer");
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn write_f64_array(ptr: *mut f64, data: &[f64]) {
    if data.is_empty() {
        return;
    }
    assert!(!ptr.is_null(), "write_f64_array: null pointer");
    unsafe {
        std::slice::from_raw_parts_mut(ptr, data.len()).copy_from_slice(data);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn view_f64_array(ptr: *const f64, len: usize) -> Result<js_sys::Float64Array, JsValue> {
    if len == 0 {
        return Ok(js_sys::Float64Array::new_with_length(0));
    }
    if ptr.is_null() {
        return Err(JsValue::from_str("view_f64_array: null pointer"));
    }
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(unsafe { js_sys::Float64Array::view(slice) })
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn read_f64_array_into(
    ptr: *const f64,
    len: usize,
    out: &js_sys::Float64Array,
) -> Result<(), JsValue> {
    if len == 0 {
        return Ok(());
    }
    if ptr.is_null() {
        return Err(JsValue::from_str("read_f64_array_into: null pointer"));
    }
    let out_len = out.length() as usize;
    if out_len < len {
        return Err(JsValue::from_str(
            "read_f64_array_into: output is too small",
        ));
    }
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    let view = unsafe { js_sys::Float64Array::view(slice) };
    out.set(&view, 0);
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub(crate) fn write_wasm_f64_output(
    context: &str,
    values: &[f64],
    out: &js_sys::Float64Array,
) -> Result<usize, JsValue> {
    let len = values.len();
    if (out.length() as usize) < len {
        return Err(JsValue::from_str(&format!(
            "{}: output is too small: expected at least {}, got {}",
            context,
            len,
            out.length()
        )));
    }
    if len == 0 {
        return Ok(0);
    }
    let view = unsafe { js_sys::Float64Array::view(values) };
    out.set(&view, 0);
    Ok(len)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub(crate) fn write_wasm_object_f64_outputs(
    context: &str,
    value: &JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let object = value
        .dyn_ref::<js_sys::Object>()
        .ok_or_else(|| JsValue::from_str(&format!("{}: result is not an object", context)))?;
    let keys = js_sys::Object::keys(object);
    let mut copied = 0usize;

    for i in 0..keys.length() {
        let key = keys.get(i);
        let Some(key_name) = key.as_string() else {
            continue;
        };
        let source = js_sys::Reflect::get(value, &key)?;
        let source_array = if let Some(array) = source.dyn_ref::<js_sys::Float64Array>() {
            Some(array.clone())
        } else if js_sys::Array::is_array(&source) {
            let array = js_sys::Array::from(&source);
            let mut numeric = true;
            for j in 0..array.length() {
                if array.get(j).as_f64().is_none() {
                    numeric = false;
                    break;
                }
            }
            if numeric {
                Some(js_sys::Float64Array::new(&source))
            } else {
                None
            }
        } else {
            None
        };

        let Some(source_array) = source_array else {
            continue;
        };
        let dest = js_sys::Reflect::get(out, &key)?;
        let dest_array = dest.dyn_ref::<js_sys::Float64Array>().ok_or_else(|| {
            JsValue::from_str(&format!(
                "{}: missing Float64Array output for field {}",
                context, key_name
            ))
        })?;
        if dest_array.length() < source_array.length() {
            return Err(JsValue::from_str(&format!(
                "{}: output field {} is too small: expected at least {}, got {}",
                context,
                key_name,
                source_array.length(),
                dest_array.length()
            )));
        }
        if source_array.length() > 0 {
            dest_array.set(&source_array, 0);
        }
        copied += source_array.length() as usize;
    }

    if copied == 0 {
        return Err(JsValue::from_str(&format!(
            "{}: no f64 output fields copied",
            context
        )));
    }

    Ok(copied)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub(crate) fn write_wasm_selected_object_f64_outputs(
    context: &str,
    value: &JsValue,
    out: &js_sys::Object,
) -> Result<usize, JsValue> {
    let keys = js_sys::Object::keys(out);
    let mut copied = 0usize;

    for i in 0..keys.length() {
        let key = keys.get(i);
        let key_name = key.as_string().ok_or_else(|| {
            JsValue::from_str(&format!("{}: output key is not a string", context))
        })?;
        let source = js_sys::Reflect::get(value, &key)?;
        if source.is_undefined() || source.is_null() {
            return Err(JsValue::from_str(&format!(
                "{}: missing result field {}",
                context, key_name
            )));
        }
        let source_array = if let Some(array) = source.dyn_ref::<js_sys::Float64Array>() {
            array.clone()
        } else if js_sys::Array::is_array(&source) {
            let array = js_sys::Array::from(&source);
            for j in 0..array.length() {
                if array.get(j).as_f64().is_none() {
                    return Err(JsValue::from_str(&format!(
                        "{}: result field {} is not numeric",
                        context, key_name
                    )));
                }
            }
            js_sys::Float64Array::new(&source)
        } else {
            return Err(JsValue::from_str(&format!(
                "{}: result field {} is not a numeric array",
                context, key_name
            )));
        };
        let dest = js_sys::Reflect::get(out, &key)?;
        let dest_array = dest.dyn_ref::<js_sys::Float64Array>().ok_or_else(|| {
            JsValue::from_str(&format!(
                "{}: output field {} is not a Float64Array",
                context, key_name
            ))
        })?;
        if dest_array.length() < source_array.length() {
            return Err(JsValue::from_str(&format!(
                "{}: output field {} is too small: expected at least {}, got {}",
                context,
                key_name,
                source_array.length(),
                dest_array.length()
            )));
        }
        if source_array.length() > 0 {
            dest_array.set(&source_array, 0);
        }
        copied += source_array.length() as usize;
    }

    if copied == 0 {
        return Err(JsValue::from_str(&format!(
            "{}: no f64 output fields copied",
            context
        )));
    }

    Ok(copied)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn allocate_f64_matrix(rows: usize, cols: usize) -> *mut f64 {
    let Some(total) = rows.checked_mul(cols) else {
        return std::ptr::null_mut();
    };
    let mut v = Vec::<f64>::with_capacity(total);
    let ptr = v.as_mut_ptr();
    let cap = v.capacity();
    std::mem::forget(v);
    register_f64_allocation(ptr, cap)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn deallocate_f64_matrix(ptr: *mut f64) {
    deallocate_f64_array(ptr);
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn read_f64_matrix(ptr: *const f64, rows: usize, cols: usize) -> js_sys::Array {
    unsafe {
        let Some(total) = rows.checked_mul(cols) else {
            return js_sys::Array::new();
        };
        let flat = std::slice::from_raw_parts(ptr, total);
        let result = js_sys::Array::new_with_length(rows as u32);
        for i in 0..rows {
            let row = js_sys::Float64Array::from(&flat[i * cols..(i + 1) * cols][..]);
            result.set(i as u32, row.into());
        }
        result
    }
}
