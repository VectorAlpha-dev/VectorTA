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
#[cfg(feature = "python")]
use crate::indicators::moving_averages::ema::{ema_py, ema_batch_py, EmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::ehlers_itrend::{ehlers_itrend_py, ehlers_itrend_batch_py, EhlersITrendStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::epma::{epma_py, epma_batch_py, EpmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::frama::{frama_py, frama_batch_py, FramaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::fwma::{fwma_py, fwma_batch_py, FwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::gaussian::{gaussian_py, gaussian_batch_py, GaussianStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::highpass_2_pole::{highpass_2_pole_py, highpass_2_pole_batch_py, HighPass2StreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::highpass::{highpass_py, highpass_batch_py, HighPassStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::hma::{hma_py, hma_batch_py, HmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::hwma::{hwma_py, hwma_batch_py, HwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::jma::{jma_py, jma_batch_py, JmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::jsa::{jsa_py, jsa_batch_py, jsa_batch_with_metadata_py, jsa_batch_2d_py, JsaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::kama::{kama_py, kama_batch_py, kama_batch_with_metadata_py, kama_batch_2d_py, KamaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::linreg::{linreg_py, linreg_batch_py, linreg_batch_with_metadata_py, linreg_batch_2d_py, LinRegStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::maaq::{maaq_py, maaq_batch_py, maaq_batch_with_metadata_py, maaq_batch_2d_py, MaaqStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::mama::{mama_py, mama_batch_py, mama_batch_with_metadata_py, mama_batch_2d_py, MamaStreamPy};

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
    
    // Register EMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(ema_py, m)?)?;
    m.add_function(wrap_pyfunction!(ema_batch_py, m)?)?;
    m.add_class::<EmaStreamPy>()?;
    
    // Register Ehlers ITrend functions with their user-facing names
    m.add_function(wrap_pyfunction!(ehlers_itrend_py, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers_itrend_batch_py, m)?)?;
    m.add_class::<EhlersITrendStreamPy>()?;
    
    // Register EPMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(epma_py, m)?)?;
    m.add_function(wrap_pyfunction!(epma_batch_py, m)?)?;
    m.add_class::<EpmaStreamPy>()?;
    
    // Register FRAMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(frama_py, m)?)?;
    m.add_function(wrap_pyfunction!(frama_batch_py, m)?)?;
    m.add_class::<FramaStreamPy>()?;
    
    // Register FWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(fwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(fwma_batch_py, m)?)?;
    m.add_class::<FwmaStreamPy>()?;
    
    // Register Gaussian functions with their user-facing names
    m.add_function(wrap_pyfunction!(gaussian_py, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_batch_py, m)?)?;
    m.add_class::<GaussianStreamPy>()?;
    
    // Register HighPass2 functions with their user-facing names
    m.add_function(wrap_pyfunction!(highpass_2_pole_py, m)?)?;
    m.add_function(wrap_pyfunction!(highpass_2_pole_batch_py, m)?)?;
    m.add_class::<HighPass2StreamPy>()?;
    
    // Register HighPass functions with their user-facing names
    m.add_function(wrap_pyfunction!(highpass_py, m)?)?;
    m.add_function(wrap_pyfunction!(highpass_batch_py, m)?)?;
    m.add_class::<HighPassStreamPy>()?;
    
    // Register HMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(hma_py, m)?)?;
    m.add_function(wrap_pyfunction!(hma_batch_py, m)?)?;
    m.add_class::<HmaStreamPy>()?;
    
    // Register HWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(hwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(hwma_batch_py, m)?)?;
    m.add_class::<HwmaStreamPy>()?;
    
    // Register JMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(jma_py, m)?)?;
    m.add_function(wrap_pyfunction!(jma_batch_py, m)?)?;
    m.add_class::<JmaStreamPy>()?;
    
    // Register JSA functions with their user-facing names
    m.add_function(wrap_pyfunction!(jsa_py, m)?)?;
    m.add_function(wrap_pyfunction!(jsa_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(jsa_batch_with_metadata_py, m)?)?;
    m.add_function(wrap_pyfunction!(jsa_batch_2d_py, m)?)?;
    m.add_class::<JsaStreamPy>()?;
    
    // Register KAMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(kama_py, m)?)?;
    m.add_function(wrap_pyfunction!(kama_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(kama_batch_with_metadata_py, m)?)?;
    m.add_function(wrap_pyfunction!(kama_batch_2d_py, m)?)?;
    m.add_class::<KamaStreamPy>()?;
    
    // Register LinReg functions with their user-facing names
    m.add_function(wrap_pyfunction!(linreg_py, m)?)?;
    m.add_function(wrap_pyfunction!(linreg_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(linreg_batch_with_metadata_py, m)?)?;
    m.add_function(wrap_pyfunction!(linreg_batch_2d_py, m)?)?;
    m.add_class::<LinRegStreamPy>()?;
    
    // Register MAAQ functions with their user-facing names
    m.add_function(wrap_pyfunction!(maaq_py, m)?)?;
    m.add_function(wrap_pyfunction!(maaq_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(maaq_batch_with_metadata_py, m)?)?;
    m.add_function(wrap_pyfunction!(maaq_batch_2d_py, m)?)?;
    m.add_class::<MaaqStreamPy>()?;
    
    // Register MAMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(mama_py, m)?)?;
    m.add_function(wrap_pyfunction!(mama_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(mama_batch_with_metadata_py, m)?)?;
    m.add_function(wrap_pyfunction!(mama_batch_2d_py, m)?)?;
    m.add_class::<MamaStreamPy>()?;
    
    // Add other indicators here as you implement their Python bindings
    
    Ok(())
}