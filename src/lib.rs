#![cfg_attr(all(feature = "nightly-avx", rustc_is_nightly), feature(stdarch_x86_avx512))]
#![cfg_attr(all(feature = "nightly-avx", rustc_is_nightly), feature(avx512_target_feature))]
#![cfg_attr(all(feature = "nightly-avx", rustc_is_nightly), feature(portable_simd))]
#![cfg_attr(all(feature = "nightly-avx", rustc_is_nightly), feature(likely_unlikely))]
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

    #[cfg(feature = "wasm")]
    pub mod wasm;
}

// Global WASM memory management functions
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn allocate_f64_array(len: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn deallocate_f64_array(ptr: *mut f64) {
    // The JavaScript side is responsible for tracking the length
    // This is a no-op as memory is managed by the WASM runtime
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn read_f64_array(ptr: *const f64, len: usize) -> Vec<f64> {
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn allocate_f64_matrix(rows: usize, cols: usize) -> *mut f64 {
    let mut v = Vec::<f64>::with_capacity(rows * cols);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn deallocate_f64_matrix(ptr: *mut f64) {
    // The JavaScript side is responsible for tracking the size
    // This is a no-op as memory is managed by the WASM runtime
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn read_f64_matrix(ptr: *const f64, rows: usize, cols: usize) -> js_sys::Array {
    unsafe {
        let flat = std::slice::from_raw_parts(ptr, rows * cols);
        let result = js_sys::Array::new_with_length(rows as u32);
        for i in 0..rows {
            let row = js_sys::Float64Array::from(&flat[i * cols..(i + 1) * cols][..]);
            result.set(i as u32, row.into());
        }
        result
    }
}
