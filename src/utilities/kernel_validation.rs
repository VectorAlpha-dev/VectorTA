//! Kernel validation for Python bindings
//!
//! This module provides validation for kernel selection in Python bindings
//! to prevent segfaults when unsupported SIMD instructions are requested.

use crate::utilities::enums::Kernel;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::PyResult;

#[cfg(feature = "python")]
pub fn validate_kernel(kernel: Option<&str>, batch: bool) -> PyResult<Kernel> {
    match kernel {
        None | Some("auto") => Ok(if batch { Kernel::Auto } else { Kernel::Auto }),
        Some("scalar") => Ok(if batch {
            Kernel::ScalarBatch
        } else {
            Kernel::Scalar
        }),
        Some("avx2") => {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    Ok(if batch {
                        Kernel::Avx2Batch
                    } else {
                        Kernel::Avx2
                    })
                } else {
                    Err(PyValueError::new_err(
                        "AVX2 kernel requested but not available on this CPU",
                    ))
                }
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Err(PyValueError::new_err(
                "AVX2 kernel not compiled in this build",
            ))
        }
        Some("avx512") => {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            {
                if std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    Ok(if batch {
                        Kernel::Avx512Batch
                    } else {
                        Kernel::Avx512
                    })
                } else {
                    Err(PyValueError::new_err(
                        "AVX512 kernel requested but not available on this CPU",
                    ))
                }
            }
            #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
            Err(PyValueError::new_err(
                "AVX512 kernel not compiled in this build",
            ))
        }
        Some(k) => Err(PyValueError::new_err(format!("Unknown kernel: {}", k))),
    }
}
