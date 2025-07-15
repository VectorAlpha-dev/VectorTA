use pyo3::prelude::*;

// Re-export all Python functions and classes from indicators
// Add module initialization here

#[cfg(feature = "python")]
use crate::indicators::acosc::{acosc_py, acosc_batch_py, AcoscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::apo::{apo_py, apo_batch_py, ApoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::bandpass::{bandpass_py, bandpass_batch_py, BandPassStreamPy};
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
use crate::indicators::moving_averages::jsa::{jsa_py, jsa_batch_py, JsaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::kama::{kama_py, kama_batch_py, KamaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::linreg::{linreg_py, linreg_batch_py, LinRegStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::maaq::{maaq_py, maaq_batch_py, MaaqStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::mama::{mama_py, mama_batch_py, MamaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::mwdx::{mwdx_py, mwdx_batch_py, MwdxStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::nma::{nma_py, nma_batch_py, NmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::pwma::{pwma_py, pwma_batch_py, PwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::reflex::{reflex_py, reflex_batch_py, ReflexStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::sinwma::{sinwma_py, sinwma_batch_py, SinWmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::sma::{sma_py, sma_batch_py, SmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::smma::{smma_py, smma_batch_py, SmmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::sqwma::{sqwma_py, sqwma_batch_py, SqwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::srwma::{srwma_py, srwma_batch_py, SrwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::supersmoother_3_pole::{supersmoother_3_pole_py, supersmoother_3_pole_batch_py, SuperSmoother3PoleStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::supersmoother::{supersmoother_py, supersmoother_batch_py, SuperSmootherStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::swma::{swma_py, swma_batch_py, SwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::tema::{tema_py, tema_batch_py, TemaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::tilson::{tilson_py, tilson_batch_py, TilsonStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::trima::{trima_py, trima_batch_py, TrimaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::trendflex::{trendflex_py, trendflex_batch_py, TrendFlexStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::wilders::{wilders_py, wilders_batch_py, WildersStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ad::{ad_py, ad_batch_py, AdStreamPy};
#[cfg(feature = "python")]
use crate::indicators::alligator::{alligator_py, alligator_batch_py, AlligatorStreamPy};
#[cfg(feature = "python")]
use crate::indicators::aroonosc::{aroon_osc_py, aroon_osc_batch_py, AroonOscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::bollinger_bands::{bollinger_bands_py, bollinger_bands_batch_py, BollingerBandsStreamPy};
#[cfg(feature = "python")]
use crate::indicators::adx::{adx_py, adx_batch_py, AdxStreamPy};
#[cfg(feature = "python")]
use crate::indicators::adosc::{adosc_py, adosc_batch_py, AdoscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::adxr::{adxr_py, adxr_batch_py, AdxrStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::vwma::{vwma_py, vwma_batch_py, VwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::vwap::{vwap_py, vwap_batch_py, VwapStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::zlema::{zlema_py, zlema_batch_py, ZlemaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::vpwma::{vpwma_py, vpwma_batch_py, VpwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::wma::{wma_py, wma_batch_py, WmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::ma::ma_py;
#[cfg(feature = "python")]
use crate::indicators::aroon::{aroon_py, aroon_batch_py, AroonStreamPy};
#[cfg(feature = "python")]
use crate::indicators::bollinger_bands_width::{bollinger_bands_width_py, bollinger_bands_width_batch_py, BollingerBandsWidthStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ao::{ao_py, ao_batch_py, AoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::atr::{atr_py, atr_batch_py, AtrStreamPy};
#[cfg(feature = "python")]
use crate::indicators::cci::{cci_py, cci_batch_py, CciStreamPy};

#[pymodule]
fn my_project(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register AD functions with their user-facing names
    m.add_function(wrap_pyfunction!(ad_py, m)?)?;
    m.add_function(wrap_pyfunction!(ad_batch_py, m)?)?;
    m.add_class::<AdStreamPy>()?;
    
    // Register ADX functions with their user-facing names
    m.add_function(wrap_pyfunction!(adx_py, m)?)?;
    m.add_function(wrap_pyfunction!(adx_batch_py, m)?)?;
    m.add_class::<AdxStreamPy>()?;
    
    // Register ADOSC functions with their user-facing names
    m.add_function(wrap_pyfunction!(adosc_py, m)?)?;
    m.add_function(wrap_pyfunction!(adosc_batch_py, m)?)?;
    m.add_class::<AdoscStreamPy>()?;
    
    // Register ADXR functions with their user-facing names
    m.add_function(wrap_pyfunction!(adxr_py, m)?)?;
    m.add_function(wrap_pyfunction!(adxr_batch_py, m)?)?;
    m.add_class::<AdxrStreamPy>()?;
    
    // Register ACOSC functions with their user-facing names
    m.add_function(wrap_pyfunction!(acosc_py, m)?)?;
    m.add_function(wrap_pyfunction!(acosc_batch_py, m)?)?;
    m.add_class::<AcoscStreamPy>()?;
    
    // Register APO functions with their user-facing names
    m.add_function(wrap_pyfunction!(apo_py, m)?)?;
    m.add_function(wrap_pyfunction!(apo_batch_py, m)?)?;
    m.add_class::<ApoStreamPy>()?;
    
    // Register Band-Pass functions with their user-facing names
    m.add_function(wrap_pyfunction!(bandpass_py, m)?)?;
    m.add_function(wrap_pyfunction!(bandpass_batch_py, m)?)?;
    m.add_class::<BandPassStreamPy>()?;
    
    // Register Alligator functions with their user-facing names
    m.add_function(wrap_pyfunction!(alligator_py, m)?)?;
    m.add_function(wrap_pyfunction!(alligator_batch_py, m)?)?;
    m.add_class::<AlligatorStreamPy>()?;
    
    // Register ALMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(alma_py, m)?)?;
    m.add_function(wrap_pyfunction!(alma_batch_py, m)?)?;
    m.add_class::<AlmaStreamPy>()?;
    
    // Register AroonOsc functions with their user-facing names
    m.add_function(wrap_pyfunction!(aroon_osc_py, m)?)?;
    m.add_function(wrap_pyfunction!(aroon_osc_batch_py, m)?)?;
    m.add_class::<AroonOscStreamPy>()?;
    
    // Register Bollinger Bands functions with their user-facing names
    m.add_function(wrap_pyfunction!(bollinger_bands_py, m)?)?;
    m.add_function(wrap_pyfunction!(bollinger_bands_batch_py, m)?)?;
    m.add_class::<BollingerBandsStreamPy>()?;
    
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
    m.add_class::<JsaStreamPy>()?;
    
    // Register KAMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(kama_py, m)?)?;
    m.add_function(wrap_pyfunction!(kama_batch_py, m)?)?;
    m.add_class::<KamaStreamPy>()?;
    
    // Register LinReg functions with their user-facing names
    m.add_function(wrap_pyfunction!(linreg_py, m)?)?;
    m.add_function(wrap_pyfunction!(linreg_batch_py, m)?)?;
    m.add_class::<LinRegStreamPy>()?;
    
    // Register MAAQ functions with their user-facing names
    m.add_function(wrap_pyfunction!(maaq_py, m)?)?;
    m.add_function(wrap_pyfunction!(maaq_batch_py, m)?)?;
    m.add_class::<MaaqStreamPy>()?;
    
    // Register MAMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(mama_py, m)?)?;
    m.add_function(wrap_pyfunction!(mama_batch_py, m)?)?;
    m.add_class::<MamaStreamPy>()?;
    
    // Register MWDX functions with their user-facing names
    m.add_function(wrap_pyfunction!(mwdx_py, m)?)?;
    m.add_function(wrap_pyfunction!(mwdx_batch_py, m)?)?;
    m.add_class::<MwdxStreamPy>()?;
    
    // Register NMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(nma_py, m)?)?;
    m.add_function(wrap_pyfunction!(nma_batch_py, m)?)?;
    m.add_class::<NmaStreamPy>()?;
    
    // Register PWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(pwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(pwma_batch_py, m)?)?;
    m.add_class::<PwmaStreamPy>()?;
    
    // Register Reflex functions with their user-facing names
    m.add_function(wrap_pyfunction!(reflex_py, m)?)?;
    m.add_function(wrap_pyfunction!(reflex_batch_py, m)?)?;
    m.add_class::<ReflexStreamPy>()?;
    
    // Register SINWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(sinwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(sinwma_batch_py, m)?)?;
    m.add_class::<SinWmaStreamPy>()?;
    
    // Register SMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(sma_py, m)?)?;
    m.add_function(wrap_pyfunction!(sma_batch_py, m)?)?;
    m.add_class::<SmaStreamPy>()?;
    
    // Register SMMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(smma_py, m)?)?;
    m.add_function(wrap_pyfunction!(smma_batch_py, m)?)?;
    m.add_class::<SmmaStreamPy>()?;
    
    // Register SQWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(sqwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(sqwma_batch_py, m)?)?;
    m.add_class::<SqwmaStreamPy>()?;
    
    // Register SRWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(srwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(srwma_batch_py, m)?)?;
    m.add_class::<SrwmaStreamPy>()?;
    
    // Register SuperSmoother3Pole functions with their user-facing names
    m.add_function(wrap_pyfunction!(supersmoother_3_pole_py, m)?)?;
    m.add_function(wrap_pyfunction!(supersmoother_3_pole_batch_py, m)?)?;
    m.add_class::<SuperSmoother3PoleStreamPy>()?;
    
    // Register SuperSmoother functions with their user-facing names
    m.add_function(wrap_pyfunction!(supersmoother_py, m)?)?;
    m.add_function(wrap_pyfunction!(supersmoother_batch_py, m)?)?;
    m.add_class::<SuperSmootherStreamPy>()?;
    
    // Register SWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(swma_py, m)?)?;
    m.add_function(wrap_pyfunction!(swma_batch_py, m)?)?;
    m.add_class::<SwmaStreamPy>()?;
    
    // Register TEMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(tema_py, m)?)?;
    m.add_function(wrap_pyfunction!(tema_batch_py, m)?)?;
    
    // Register TRIMA functions
    m.add_function(wrap_pyfunction!(trima_py, m)?)?;
    m.add_function(wrap_pyfunction!(trima_batch_py, m)?)?;
    m.add_class::<TrimaStreamPy>()?;
    m.add_class::<TemaStreamPy>()?;
    
    // Register Tilson functions with their user-facing names
    m.add_function(wrap_pyfunction!(tilson_py, m)?)?;
    m.add_function(wrap_pyfunction!(tilson_batch_py, m)?)?;
    m.add_class::<TilsonStreamPy>()?;
    
    // Register TrendFlex functions with their user-facing names
    m.add_function(wrap_pyfunction!(trendflex_py, m)?)?;
    m.add_function(wrap_pyfunction!(trendflex_batch_py, m)?)?;
    m.add_class::<TrendFlexStreamPy>()?;
    
    // Register Wilders functions with their user-facing names
    m.add_function(wrap_pyfunction!(wilders_py, m)?)?;
    m.add_function(wrap_pyfunction!(wilders_batch_py, m)?)?;
    m.add_class::<WildersStreamPy>()?;
    
    // Register VWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(vwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(vwma_batch_py, m)?)?;
    m.add_class::<VwmaStreamPy>()?;
    
    // Register VWAP functions with their user-facing names
    m.add_function(wrap_pyfunction!(vwap_py, m)?)?;
    m.add_function(wrap_pyfunction!(vwap_batch_py, m)?)?;
    m.add_class::<VwapStreamPy>()?;
    
    // Register ZLEMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(zlema_py, m)?)?;
    m.add_function(wrap_pyfunction!(zlema_batch_py, m)?)?;
    m.add_class::<ZlemaStreamPy>()?;
    
    // Register VPWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(vpwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(vpwma_batch_py, m)?)?;
    m.add_class::<VpwmaStreamPy>()?;
    
    // Register WMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(wma_py, m)?)?;
    m.add_function(wrap_pyfunction!(wma_batch_py, m)?)?;
    m.add_class::<WmaStreamPy>()?;
    
    // Register MA dispatcher function
    m.add_function(wrap_pyfunction!(ma_py, m)?)?;
    
    // Register Aroon functions with their user-facing names
    m.add_function(wrap_pyfunction!(aroon_py, m)?)?;
    m.add_function(wrap_pyfunction!(aroon_batch_py, m)?)?;
    m.add_class::<AroonStreamPy>()?;
    
    // Register Bollinger Bands Width functions with their user-facing names
    m.add_function(wrap_pyfunction!(bollinger_bands_width_py, m)?)?;
    m.add_function(wrap_pyfunction!(bollinger_bands_width_batch_py, m)?)?;
    m.add_class::<BollingerBandsWidthStreamPy>()?;
    
    // Register AO functions with their user-facing names
    m.add_function(wrap_pyfunction!(ao_py, m)?)?;
    m.add_function(wrap_pyfunction!(ao_batch_py, m)?)?;
    m.add_class::<AoStreamPy>()?;
    
    // Register ATR functions with their user-facing names
    m.add_function(wrap_pyfunction!(atr_py, m)?)?;
    m.add_function(wrap_pyfunction!(atr_batch_py, m)?)?;
    m.add_class::<AtrStreamPy>()?;
    
    // Register CCI functions with their user-facing names
    m.add_function(wrap_pyfunction!(cci_py, m)?)?;
    m.add_function(wrap_pyfunction!(cci_batch_py, m)?)?;
    m.add_class::<CciStreamPy>()?;
    
    // Add other indicators here as you implement their Python bindings
    
    Ok(())
}