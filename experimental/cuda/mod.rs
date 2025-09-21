pub mod moving_averages;

use cudarc::driver::CudaDevice;
use std::panic;

/// Check if CUDA is available on this system
pub fn cuda_available() -> bool {
    // First try the simple approach
    if let Ok(count) = CudaDevice::count() {
        return count > 0;
    }

    // If that fails, try creating a device directly
    // This works better in WSL2 environments
    panic::catch_unwind(|| {
        CudaDevice::new(0).is_ok()
    }).unwrap_or(false)
}

/// Get the number of CUDA devices available
pub fn cuda_device_count() -> usize {
    panic::catch_unwind(|| {
        CudaDevice::count().unwrap_or(0) as usize
    }).unwrap_or(0)
}