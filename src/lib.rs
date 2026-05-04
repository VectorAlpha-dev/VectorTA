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
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub fn write_f64_array(ptr: *mut f64, data: &[f64]) {
    unsafe {
        std::slice::from_raw_parts_mut(ptr, data.len()).copy_from_slice(data);
    }
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
