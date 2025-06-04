#![cfg_attr(feature = "nightly-avx", feature(stdarch_x86_avx512))]
#![cfg_attr(feature = "nightly-avx", feature(avx512_target_feature))]
#![cfg_attr(feature = "nightly-avx", feature(portable_simd))]
#![cfg_attr(feature = "nightly-avx", feature(likely_unlikely))] 
#![allow(warnings)]
#![allow(clippy::needless_range_loop)]


pub mod indicators;
pub mod utilities;

#[cfg(test)]
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

#[cfg(feature = "wasm")]
pub mod bindings {
    pub mod wasm;
}

#[cfg(feature = "python")]
pub mod bindings {
    pub mod python;
}