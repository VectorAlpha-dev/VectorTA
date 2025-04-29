#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![feature(stdarch_x86_avx512)]
#![feature(portable_simd)]
#![feature(avx512_target_feature)]
#[allow(non_snake_case)]


mod indicators;
mod utilities;
use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    Ok(())
}
