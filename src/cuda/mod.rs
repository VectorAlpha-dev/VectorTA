//! CUDA integration scaffolding (cust-based)
//!
//! This module is built only when the `cuda` feature is enabled. It provides
//! runtime detection helpers and submodules for GPU-accelerated indicators.

#[cfg(feature = "cuda")]
pub mod moving_averages;
#[cfg(feature = "cuda")]
pub mod wad_wrapper;
#[cfg(feature = "cuda")]
pub mod zscore_wrapper;

#[cfg(feature = "cuda")]
pub use moving_averages::{
    CudaAlma, CudaBuffAverages, CudaBuffAveragesError, CudaFrama, CudaFramaError, CudaHma,
    CudaHmaError, CudaLinreg, CudaLinregError, CudaNma, CudaNmaError, CudaSma, CudaSmaError,
    CudaSuperSmoother, CudaSuperSmootherError, CudaTrendflex, CudaTrendflexError, CudaVpwma,
    CudaVpwmaError, CudaZlema, CudaZlemaError, DeviceArrayF32,
};
#[cfg(feature = "cuda")]
pub use wad_wrapper::{CudaWad, CudaWadError};
#[cfg(feature = "cuda")]
pub use zscore_wrapper::{CudaZscore, CudaZscoreError};

/// Returns true if a CUDA device is available and the driver API can be initialized.
#[inline]
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        use cust::{device::Device, prelude::CudaFlags};
        // Initialize the CUDA driver and query devices. Keep this defensive so
        // it never panics when CUDA is missing.
        if cust::init(CudaFlags::empty()).is_err() {
            return false;
        }
        match Device::num_devices() {
            Ok(n) => n > 0,
            Err(_) => false,
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Returns the number of CUDA devices available (0 on error or when disabled).
#[inline]
pub fn cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        use cust::{device::Device, prelude::CudaFlags};
        if cust::init(CudaFlags::empty()).is_err() {
            return 0;
        }
        match Device::num_devices() {
            Ok(n) => n as usize,
            Err(_) => 0,
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}
