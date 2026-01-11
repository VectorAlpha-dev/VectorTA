#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::os::raw::{c_double, c_int};


pub mod rust_ffi;


pub mod report;


pub mod ffi_overhead;


pub mod benchmark_methodology;


pub mod unified_benchmark;


pub mod json_export;


pub mod benchmark_collector;


include!(concat!(env!("OUT_DIR"), "/tulip_bindings.rs"));


#[cfg(feature = "talib")]
include!(concat!(env!("OUT_DIR"), "/talib_bindings.rs"));


pub const TI_OKAY: i32 = 0;


pub type TulipReal = c_double;
pub type TulipInt = c_int;


pub mod tulip {
    use super::*;
    use std::slice;

    /// Find a Tulip indicator by name and return its metadata pointer.
    ///
    /// This is useful for hot loops (benchmarks) that want to avoid allocating
    /// per-call pointer arrays in `call_indicator`.
    pub unsafe fn find_indicator(name: &str) -> Result<*const ti_indicator_info, String> {
        let c_name = std::ffi::CString::new(name).unwrap();
        let indicator = ti_find_indicator(c_name.as_ptr());

        if indicator.is_null() {
            return Err(format!("Indicator '{}' not found", name));
        }

        Ok(indicator)
    }

    /// Low-overhead Tulip call that uses precomputed input/output pointer arrays.
    ///
    /// This avoids per-call allocations present in `call_indicator` and is intended for
    /// benchmark hot loops. Caller must ensure pointer arrays have the correct lengths.
    pub unsafe fn call_indicator_ptrs(
        indicator: *const ti_indicator_info,
        size: usize,
        input_ptrs: &[*const TulipReal],
        options: &[f64],
        output_ptrs: &mut [*mut TulipReal],
    ) -> Result<(), String> {
        if indicator.is_null() {
            return Err("Null indicator pointer".to_string());
        }

        let info = &*indicator;

        
        if input_ptrs.len() != info.inputs as usize {
            return Err(format!(
                "Expected {} inputs, got {}",
                info.inputs,
                input_ptrs.len()
            ));
        }

        if output_ptrs.len() != info.outputs as usize {
            return Err(format!(
                "Expected {} outputs, got {}",
                info.outputs,
                output_ptrs.len()
            ));
        }

        
        let indicator_fn = info.indicator.expect("Indicator function not found");
        let result = indicator_fn(
            size as c_int,
            input_ptrs.as_ptr(),
            options.as_ptr() as *const TulipReal,
            output_ptrs.as_mut_ptr(),
        );

        if result != TI_OKAY as i32 {
            return Err(format!("Indicator failed with code {}", result));
        }

        Ok(())
    }

    /// Safe wrapper for Tulip indicator calls
    pub unsafe fn call_indicator(
        name: &str,
        size: usize,
        inputs: &[&[f64]],
        options: &[f64],
        outputs: &mut [&mut [f64]],
    ) -> Result<(), String> {
        let c_name = std::ffi::CString::new(name).unwrap();
        let indicator = ti_find_indicator(c_name.as_ptr());

        if indicator.is_null() {
            return Err(format!("Indicator '{}' not found", name));
        }

        let info = &*indicator;

        
        if inputs.len() != info.inputs as usize {
            return Err(format!(
                "Expected {} inputs, got {}",
                info.inputs, inputs.len()
            ));
        }

        if outputs.len() != info.outputs as usize {
            return Err(format!(
                "Expected {} outputs, got {}",
                info.outputs, outputs.len()
            ));
        }

        
        let input_ptrs: Vec<*const TulipReal> = inputs
            .iter()
            .map(|slice| slice.as_ptr() as *const TulipReal)
            .collect();

        let mut output_ptrs: Vec<*mut TulipReal> = outputs
            .iter_mut()
            .map(|slice| slice.as_mut_ptr() as *mut TulipReal)
            .collect();

        
        let indicator_fn = info.indicator.expect("Indicator function not found");
        let result = indicator_fn(
            size as c_int,
            input_ptrs.as_ptr(),
            options.as_ptr() as *const TulipReal,
            output_ptrs.as_mut_ptr(),
        );

        if result != TI_OKAY as i32 {
            return Err(format!("Indicator failed with code {}", result));
        }

        Ok(())
    }

    /// Get the start index for an indicator
    pub unsafe fn get_start_index(name: &str, options: &[f64]) -> Result<usize, String> {
        let c_name = std::ffi::CString::new(name).unwrap();
        let indicator = ti_find_indicator(c_name.as_ptr());

        if indicator.is_null() {
            return Err(format!("Indicator '{}' not found", name));
        }

        let info = &*indicator;
        let start_fn = info.start.expect("Start function not found");
        let start = start_fn(options.as_ptr() as *const TulipReal);

        Ok(start as usize)
    }

    /// Query Tulip for input/output arity for a given indicator name.
    pub unsafe fn get_io_counts(name: &str) -> Result<(usize, usize), String> {
        let c_name = std::ffi::CString::new(name).unwrap();
        let indicator = ti_find_indicator(c_name.as_ptr());
        if indicator.is_null() {
            return Err(format!("Indicator '{}' not found", name));
        }
        let info = &*indicator;
        Ok((info.inputs as usize, info.outputs as usize))
    }
}


#[cfg(feature = "talib")]
pub mod talib_wrapper;


pub mod utils {
    use std::path::Path;
    use csv::Reader;
    use std::error::Error;

    #[derive(Debug)]
    pub struct CandleData {
        pub timestamps: Vec<i64>,
        pub open: Vec<f64>,
        pub high: Vec<f64>,
        pub low: Vec<f64>,
        pub close: Vec<f64>,
        pub volume: Vec<f64>,
    }

    impl CandleData {
        pub fn from_csv(path: &Path) -> Result<Self, Box<dyn Error>> {
            let mut reader = Reader::from_path(path)?;
            let mut data = CandleData {
                timestamps: Vec::new(),
                open: Vec::new(),
                high: Vec::new(),
                low: Vec::new(),
                close: Vec::new(),
                volume: Vec::new(),
            };

            for result in reader.records() {
                let record = result?;
                data.timestamps.push(record[0].parse()?);
                data.open.push(record[1].parse()?);
                data.high.push(record[2].parse()?);
                data.low.push(record[3].parse()?);
                data.close.push(record[4].parse()?);
                data.volume.push(record[5].parse()?);
            }

            Ok(data)
        }

        pub fn len(&self) -> usize {
            self.close.len()
        }
    }
}
