pub mod moving_averages;

use cudarc::driver::CudaDevice;
use std::panic;


pub fn cuda_available() -> bool {

    if let Ok(count) = CudaDevice::count() {
        return count > 0;
    }



    panic::catch_unwind(|| {
        CudaDevice::new(0).is_ok()
    }).unwrap_or(false)
}


pub fn cuda_device_count() -> usize {
    panic::catch_unwind(|| {
        CudaDevice::count().unwrap_or(0) as usize
    }).unwrap_or(0)
}