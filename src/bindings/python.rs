use pyo3::prelude::*;

// Re-export all Python functions and classes from indicators
// Add module initialization here

#[cfg(feature = "python")]
use crate::indicators::moving_averages::alma::{alma_py, alma_batch_py, AlmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::cwma::{cwma_py, cwma_batch_py, CwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::dema::{dema_py, dema_batch_py, DemaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::edcf::{edcf_py, edcf_batch_py, EdcfStreamPy};

#[pymodule]
fn my_project(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register ALMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(alma_py, m)?)?;
    m.add_function(wrap_pyfunction!(alma_batch_py, m)?)?;
    m.add_class::<AlmaStreamPy>()?;
    
    // Register CWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(cwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(cwma_batch_py, m)?)?;
    m.add_class::<CwmaStreamPy>()?;
    
    // Register DEMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(dema_py, m)?)?;
    m.add_function(wrap_pyfunction!(dema_batch_py, m)?)?;
    m.add_class::<DemaStreamPy>()?;
    
    // Register EDCF functions with their user-facing names
    m.add_function(wrap_pyfunction!(edcf_py, m)?)?;
    m.add_function(wrap_pyfunction!(edcf_batch_py, m)?)?;
    m.add_class::<EdcfStreamPy>()?;
    
    // Add other indicators here as you implement their Python bindings
    
    Ok(())
}