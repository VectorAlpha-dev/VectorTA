#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::os::raw::{c_double, c_int};

// Include our Rust FFI exports
pub mod rust_ffi;

// Report generation module
pub mod report;

// Include generated bindings
include!(concat!(env!("OUT_DIR"), "/tulip_bindings.rs"));

#[cfg(not(no_talib))]
include!(concat!(env!("OUT_DIR"), "/talib_bindings.rs"));

// Re-export useful types
pub type TulipReal = c_double;
pub type TulipInt = c_int;

// Wrapper module for Tulip indicators
pub mod tulip {
    use super::*;
    use std::slice;

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
        
        // Verify input/output counts
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

        // Convert to raw pointers
        let input_ptrs: Vec<*const TulipReal> = inputs
            .iter()
            .map(|slice| slice.as_ptr() as *const TulipReal)
            .collect();
        
        let mut output_ptrs: Vec<*mut TulipReal> = outputs
            .iter_mut()
            .map(|slice| slice.as_mut_ptr() as *mut TulipReal)
            .collect();

        // Call the indicator
        let result = (info.indicator)(
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
        let start = (info.start)(options.as_ptr() as *const TulipReal);
        
        Ok(start as usize)
    }
}

// Module for TA-Lib wrappers (if available)
#[cfg(not(no_talib))]
pub mod talib {
    use super::*;
    
    // TA-Lib wrapper implementations will go here
    pub unsafe fn call_sma(
        _input: &[f64],
        _period: i32,
        _output: &mut [f64],
    ) -> Result<(), String> {
        // Placeholder - actual implementation depends on TA-Lib bindings
        Err("TA-Lib not implemented yet".to_string())
    }
}

// Common benchmark utilities
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