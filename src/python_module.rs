//! Python module registration for technical indicators

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn ta_indicators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register indicator modules
    crate::indicators::zscore::register_zscore_module(m)?;
    crate::indicators::moving_averages::alma::register_alma_module(m)?;
    crate::indicators::other_indicators::volume_adjusted_ma::register_volume_adjusted_ma_module(m)?;
    crate::indicators::other_indicators::nadaraya_watson_envelope::register_nadaraya_watson_envelope_module(m)?;
    
    // Add other indicators here as they are updated
    
    Ok(())
}