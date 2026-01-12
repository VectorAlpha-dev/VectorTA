use std::os::raw::{c_double, c_int};
use std::time::{Duration, Instant};
use std::slice;


#[no_mangle]
pub unsafe extern "C" fn noop() -> c_int {
    0
}


#[no_mangle]
pub unsafe extern "C" fn copy_array(
    size: c_int,
    input: *const c_double,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let output_slice = slice::from_raw_parts_mut(output, size as usize);
    output_slice.copy_from_slice(input_slice);
    0
}


#[no_mangle]
pub unsafe extern "C" fn validated_sum(
    size: c_int,
    input: *const c_double,
    output: *mut c_double,
) -> c_int {
    if input.is_null() || output.is_null() || size <= 0 {
        return -1;
    }

    let input_slice = slice::from_raw_parts(input, size as usize);
    let sum: f64 = input_slice.iter().sum();
    *output = sum;
    0
}


pub fn measure_ffi_overhead(iterations: usize) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            noop();
        }
    }
    start.elapsed()
}


pub fn measure_marshalling_overhead(data: &[f64], iterations: usize) -> Duration {
    let mut output = vec![0.0; data.len()];
    let start = Instant::now();

    for _ in 0..iterations {
        unsafe {
            copy_array(
                data.len() as c_int,
                data.as_ptr(),
                output.as_mut_ptr(),
            );
        }
    }

    start.elapsed()
}


pub fn measure_validation_overhead(data: &[f64], iterations: usize) -> Duration {
    let mut result = 0.0;
    let start = Instant::now();

    for _ in 0..iterations {
        unsafe {
            validated_sum(
                data.len() as c_int,
                data.as_ptr(),
                &mut result as *mut f64,
            );
        }
    }

    start.elapsed()
}


#[derive(Debug, Clone)]
pub struct FfiOverheadProfile {
    pub call_overhead_ns: f64,
    pub marshalling_overhead_ns_per_kb: f64,
    pub validation_overhead_ns: f64,
}

impl FfiOverheadProfile {

    pub fn profile(data_size: usize, iterations: usize) -> Self {

        let data: Vec<f64> = (0..data_size).map(|i| i as f64).collect();


        let call_duration = measure_ffi_overhead(iterations);
        let call_overhead_ns = call_duration.as_nanos() as f64 / iterations as f64;


        let marshalling_duration = measure_marshalling_overhead(&data, iterations);
        let marshalling_overhead_ns = marshalling_duration.as_nanos() as f64 / iterations as f64;
        let marshalling_overhead_ns_per_kb = marshalling_overhead_ns / (data_size as f64 * 8.0 / 1024.0);


        let validation_duration = measure_validation_overhead(&data, iterations);
        let validation_total_ns = validation_duration.as_nanos() as f64 / iterations as f64;
        let validation_overhead_ns = (validation_total_ns - call_overhead_ns).max(0.0);

        FfiOverheadProfile {
            call_overhead_ns,
            marshalling_overhead_ns_per_kb,
            validation_overhead_ns,
        }
    }


    pub fn estimate_overhead(&self, data_size_bytes: usize) -> Duration {
        let data_size_kb = data_size_bytes as f64 / 1024.0;
        let total_ns = self.call_overhead_ns +
                      (self.marshalling_overhead_ns_per_kb * data_size_kb) +
                      self.validation_overhead_ns;
        Duration::from_nanos(total_ns as u64)
    }


    pub fn compensate(&self, measured: Duration, data_size_bytes: usize) -> Duration {
        let overhead = self.estimate_overhead(data_size_bytes);
        measured.saturating_sub(overhead)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_overhead_measurement() {
        let profile = FfiOverheadProfile::profile(1000, 10000);


        assert!(profile.call_overhead_ns > 0.0);
        assert!(profile.marshalling_overhead_ns_per_kb > 0.0);
        assert!(profile.validation_overhead_ns >= 0.0);


        let overhead = profile.estimate_overhead(8000);
        assert!(overhead.as_nanos() > 0);
    }

    #[test]
    fn test_compensation() {
        let profile = FfiOverheadProfile::profile(1000, 1000);
        let measured = Duration::from_micros(100);
        let compensated = profile.compensate(measured, 8000);


        assert!(compensated <= measured);
    }
}