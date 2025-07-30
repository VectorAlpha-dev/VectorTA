//! Python module registration for technical indicators

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn ta_indicators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register indicator modules
    crate::indicators::zscore::register_zscore_module(m)?;
    crate::indicators::moving_averages::alma::register_alma_module(m)?;
    
    // Add other indicators here as they are updated
    
    Ok(())
}