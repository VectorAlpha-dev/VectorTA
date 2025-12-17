#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![cfg_attr(all(feature = "nightly-avx", rustc_is_nightly), feature(stdarch_x86_avx512))]
#![cfg_attr(all(feature = "nightly-avx", rustc_is_nightly), feature(portable_simd))]
#![cfg_attr(all(feature = "nightly-avx", rustc_is_nightly), feature(avx512_target_feature))]
#![cfg_attr(all(feature = "nightly-avx", rustc_is_nightly), feature(likely_unlikely))]
#![allow(warnings)]
#![allow(non_snake_case)]

#[cfg(feature = "cuda")]
mod cuda;
mod indicators;
mod utilities;
use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    Ok(())
}
