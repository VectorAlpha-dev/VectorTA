use pyo3::prelude::*;

// Re-export all Python functions and classes from indicators
// Add module initialization here

#[cfg(feature = "python")]
use crate::indicators::acosc::{acosc_batch_py, acosc_py, AcoscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ad::{ad_batch_py, ad_py, AdStreamPy};
#[cfg(feature = "python")]
use crate::indicators::adosc::{adosc_batch_py, adosc_py, AdoscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::adx::{adx_batch_py, adx_py, AdxStreamPy};
#[cfg(feature = "python")]
use crate::indicators::adxr::{adxr_batch_py, adxr_py, AdxrStreamPy};
#[cfg(feature = "python")]
use crate::indicators::alligator::{alligator_batch_py, alligator_py, AlligatorStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ao::{ao_batch_py, ao_py, AoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::apo::{apo_batch_py, apo_py, ApoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::aroon::{aroon_batch_py, aroon_py, AroonStreamPy};
#[cfg(feature = "python")]
use crate::indicators::aroonosc::{aroon_osc_batch_py, aroon_osc_py, AroonOscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::atr::{atr_batch_py, atr_py, AtrStreamPy};
#[cfg(feature = "python")]
use crate::indicators::bandpass::{bandpass_batch_py, bandpass_py, BandPassStreamPy};
#[cfg(feature = "python")]
use crate::indicators::bollinger_bands::{bollinger_bands_batch_py, bollinger_bands_py, BollingerBandsStreamPy};
#[cfg(feature = "python")]
use crate::indicators::bollinger_bands_width::{
	bollinger_bands_width_batch_py, bollinger_bands_width_py, BollingerBandsWidthStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::bop::{bop_batch_py, bop_py, BopStreamPy};
#[cfg(feature = "python")]
use crate::indicators::cci::{cci_batch_py, cci_py, CciStreamPy};
#[cfg(feature = "python")]
use crate::indicators::cfo::{cfo_batch_py, cfo_py, CfoBatchResult, CfoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::cg::{cg_batch_py, cg_py, CgStreamPy};
#[cfg(feature = "python")]
use crate::indicators::chande::{chande_batch_py, chande_py, ChandeStreamPy};
#[cfg(feature = "python")]
use crate::indicators::correlation_cycle::{
	correlation_cycle_batch_py, correlation_cycle_py, CorrelationCycleStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::dti::{dti_batch_py, dti_py, DtiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::dx::{dx_batch_py, dx_py, DxStreamPy};
#[cfg(feature = "python")]
use crate::indicators::fisher::{fisher_batch_py, fisher_py, FisherStreamPy};
#[cfg(feature = "python")]
use crate::indicators::keltner::{keltner_batch_py, keltner_py, KeltnerStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::alma::{alma_batch_py, alma_py, AlmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::cwma::{cwma_batch_py, cwma_py, CwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::dema::{dema_batch_py, dema_py, DemaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::edcf::{edcf_batch_py, edcf_py, EdcfStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::ehlers_itrend::{
	ehlers_itrend_batch_py, ehlers_itrend_py, EhlersITrendStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::ema::{ema_batch_py, ema_py, EmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::epma::{epma_batch_py, epma_py, EpmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::frama::{frama_batch_py, frama_py, FramaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::fwma::{fwma_batch_py, fwma_py, FwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::gaussian::{gaussian_batch_py, gaussian_py, GaussianStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::highpass::{highpass_batch_py, highpass_py, HighPassStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::highpass_2_pole::{
	highpass_2_pole_batch_py, highpass_2_pole_py, HighPass2StreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::hma::{hma_batch_py, hma_py, HmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::hwma::{hwma_batch_py, hwma_py, HwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::jma::{jma_batch_py, jma_py, JmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::jsa::{jsa_batch_py, jsa_py, JsaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::kama::{kama_batch_py, kama_py, KamaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::linreg::{linreg_batch_py, linreg_py, LinRegStreamPy};
#[cfg(feature = "python")]
use crate::indicators::linearreg_slope::{linearreg_slope_batch_py, linearreg_slope_py, LinearRegSlopeStreamPy};
#[cfg(feature = "python")]
use crate::indicators::medium_ad::{medium_ad_batch_py, medium_ad_py, MediumAdStreamPy};
#[cfg(feature = "python")]
use crate::indicators::minmax::{minmax_batch_py, minmax_py, MinmaxStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::ma::ma_py;
#[cfg(feature = "python")]
use crate::indicators::moving_averages::maaq::{maaq_batch_py, maaq_py, MaaqStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::mama::{mama_batch_py, mama_py, MamaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::mwdx::{mwdx_batch_py, mwdx_py, MwdxStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::nma::{nma_batch_py, nma_py, NmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::pwma::{pwma_batch_py, pwma_py, PwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::reflex::{reflex_batch_py, reflex_py, ReflexStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::sinwma::{sinwma_batch_py, sinwma_py, SinWmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::sma::{sma_batch_py, sma_py, SmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::smma::{smma_batch_py, smma_py, SmmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::sqwma::{sqwma_batch_py, sqwma_py, SqwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::srwma::{srwma_batch_py, srwma_py, SrwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::stddev::{stddev_batch_py, stddev_py, StdDevStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::supersmoother::{
	supersmoother_batch_py, supersmoother_py, SuperSmootherStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::supersmoother_3_pole::{
	supersmoother_3_pole_batch_py, supersmoother_3_pole_py, SuperSmoother3PoleStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::swma::{swma_batch_py, swma_py, SwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::tema::{tema_batch_py, tema_py, TemaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::tilson::{tilson_batch_py, tilson_py, TilsonStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::trendflex::{trendflex_batch_py, trendflex_py, TrendFlexStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ttm_trend::{ttm_trend_batch_py, ttm_trend_py, TtmTrendStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::trima::{trima_batch_py, trima_py, TrimaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::vpwma::{vpwma_batch_py, vpwma_py, VpwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::vwap::{vwap_batch_py, vwap_py, VwapStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::vwma::{vwma_batch_py, vwma_py, VwmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::vwmacd::{vwmacd_batch_py, vwmacd_py, VwmacdStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::wilders::{wilders_batch_py, wilders_py, WildersStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::wma::{wma_batch_py, wma_py, WmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::zlema::{zlema_batch_py, zlema_py, ZlemaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::pfe::{pfe_batch_py, pfe_py, PfeStreamPy};
#[cfg(feature = "python")]
use crate::indicators::roc::{roc_batch_py, roc_py, RocStreamPy};
#[cfg(feature = "python")]
use crate::indicators::rvi::{rvi_batch_py, rvi_py, RviStreamPy};
#[cfg(feature = "python")]
use crate::indicators::vlma::{vlma_batch_py, vlma_py, VlmaStreamPy};

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
	
	// Register LinearRegSlope functions with their user-facing names
	m.add_function(wrap_pyfunction!(linearreg_slope_py, m)?)?;
	m.add_function(wrap_pyfunction!(linearreg_slope_batch_py, m)?)?;
	m.add_class::<LinearRegSlopeStreamPy>()?;

	// Register MediumAd functions with their user-facing names
	m.add_function(wrap_pyfunction!(medium_ad_py, m)?)?;
	m.add_function(wrap_pyfunction!(medium_ad_batch_py, m)?)?;
	m.add_class::<MediumAdStreamPy>()?;

	// Register MinMax functions with their user-facing names
	m.add_function(wrap_pyfunction!(minmax_py, m)?)?;
	m.add_function(wrap_pyfunction!(minmax_batch_py, m)?)?;
	m.add_class::<MinmaxStreamPy>()?;

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

	// Register PFE functions with their user-facing names
	m.add_function(wrap_pyfunction!(pfe_py, m)?)?;
	m.add_function(wrap_pyfunction!(pfe_batch_py, m)?)?;
	m.add_class::<PfeStreamPy>()?;

	// Register ROC functions with their user-facing names
	m.add_function(wrap_pyfunction!(roc_py, m)?)?;
	m.add_function(wrap_pyfunction!(roc_batch_py, m)?)?;
	m.add_class::<RocStreamPy>()?;

	// Register RVI functions with their user-facing names
	m.add_function(wrap_pyfunction!(rvi_py, m)?)?;
	m.add_function(wrap_pyfunction!(rvi_batch_py, m)?)?;
	m.add_class::<RviStreamPy>()?;

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

	// Register StdDev functions with their user-facing names
	m.add_function(wrap_pyfunction!(stddev_py, m)?)?;
	m.add_function(wrap_pyfunction!(stddev_batch_py, m)?)?;
	m.add_class::<StdDevStreamPy>()?;

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

	// Register TTM Trend functions with their user-facing names
	m.add_function(wrap_pyfunction!(ttm_trend_py, m)?)?;
	m.add_function(wrap_pyfunction!(ttm_trend_batch_py, m)?)?;
	m.add_class::<TtmTrendStreamPy>()?;

	// Register VLMA functions with their user-facing names
	m.add_function(wrap_pyfunction!(vlma_py, m)?)?;
	m.add_function(wrap_pyfunction!(vlma_batch_py, m)?)?;
	m.add_class::<VlmaStreamPy>()?;

	// Register Wilders functions with their user-facing names
	m.add_function(wrap_pyfunction!(wilders_py, m)?)?;
	m.add_function(wrap_pyfunction!(wilders_batch_py, m)?)?;
	m.add_class::<WildersStreamPy>()?;

	// Register VWMA functions with their user-facing names
	m.add_function(wrap_pyfunction!(vwma_py, m)?)?;
	m.add_function(wrap_pyfunction!(vwma_batch_py, m)?)?;
	m.add_class::<VwmaStreamPy>()?;

	// Register VWMACD functions with their user-facing names
	m.add_function(wrap_pyfunction!(vwmacd_py, m)?)?;
	m.add_function(wrap_pyfunction!(vwmacd_batch_py, m)?)?;
	m.add_class::<VwmacdStreamPy>()?;

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

	// Register CG functions with their user-facing names
	m.add_function(wrap_pyfunction!(cg_py, m)?)?;
	m.add_function(wrap_pyfunction!(cg_batch_py, m)?)?;
	m.add_class::<CgStreamPy>()?;

	// Register Correlation Cycle functions with their user-facing names
	m.add_function(wrap_pyfunction!(correlation_cycle_py, m)?)?;
	m.add_function(wrap_pyfunction!(correlation_cycle_batch_py, m)?)?;
	m.add_class::<CorrelationCycleStreamPy>()?;

	// Register DTI functions with their user-facing names
	m.add_function(wrap_pyfunction!(dti_py, m)?)?;
	m.add_function(wrap_pyfunction!(dti_batch_py, m)?)?;
	m.add_class::<DtiStreamPy>()?;

	// Register DX functions with their user-facing names
	m.add_function(wrap_pyfunction!(dx_py, m)?)?;
	m.add_function(wrap_pyfunction!(dx_batch_py, m)?)?;
	m.add_class::<DxStreamPy>()?;

	// Register Fisher functions
	m.add_function(wrap_pyfunction!(fisher_py, m)?)?;
	m.add_function(wrap_pyfunction!(fisher_batch_py, m)?)?;
	m.add_class::<FisherStreamPy>()?;

	// Register Keltner functions
	m.add_function(wrap_pyfunction!(keltner_py, m)?)?;
	m.add_function(wrap_pyfunction!(keltner_batch_py, m)?)?;
	m.add_class::<KeltnerStreamPy>()?;

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

	// Register CFO functions with their user-facing names
	m.add_function(wrap_pyfunction!(cfo_py, m)?)?;
	m.add_function(wrap_pyfunction!(cfo_batch_py, m)?)?;
	m.add_class::<CfoStreamPy>()?;
	m.add_class::<CfoBatchResult>()?;

	// Register BOP functions with their user-facing names
	m.add_function(wrap_pyfunction!(bop_py, m)?)?;
	m.add_function(wrap_pyfunction!(bop_batch_py, m)?)?;
	m.add_class::<BopStreamPy>()?;

	// Register Chande functions with their user-facing names
	m.add_function(wrap_pyfunction!(chande_py, m)?)?;
	m.add_function(wrap_pyfunction!(chande_batch_py, m)?)?;
	m.add_class::<ChandeStreamPy>()?;

	// Add other indicators here as you implement their Python bindings

	Ok(())
}
