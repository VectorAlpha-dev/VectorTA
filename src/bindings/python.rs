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
use crate::indicators::moving_averages::buff_averages::{buff_averages_py, buff_averages_batch_py, BuffAveragesStreamPy};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::volume_adjusted_ma::{volume_adjusted_ma_py, volume_adjusted_ma_batch_py, VolumeAdjustedMaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::qqe::{qqe_py, qqe_batch_py, QqeStreamPy};
#[cfg(feature = "python")]
use crate::indicators::nadaraya_watson_envelope::{nadaraya_watson_envelope_py, nadaraya_watson_envelope_batch_py, NweStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ttm_squeeze::{ttm_squeeze_py, ttm_squeeze_batch_py, TtmSqueezeStreamPy};
#[cfg(feature = "python")]
use crate::indicators::mod_god_mode::{mod_god_mode_py, mod_god_mode_batch_py, ModGodModeStreamPy};
#[cfg(feature = "python")]
use crate::indicators::cci::{cci_batch_py, cci_py, CciStreamPy};
#[cfg(feature = "python")]
use crate::indicators::cfo::{cfo_batch_py, cfo_py, CfoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::cg::{cg_batch_py, cg_py, CgStreamPy};
#[cfg(feature = "python")]
use crate::indicators::coppock::{coppock_batch_py, coppock_py, CoppockStreamPy};
#[cfg(feature = "python")]
use crate::indicators::cmo::{cmo_batch_py, cmo_py, CmoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::cksp::{cksp_batch_py, cksp_py, CkspStreamPy};
#[cfg(feature = "python")]
use crate::indicators::chop::{chop_batch_py, chop_py, ChopStreamPy};
#[cfg(feature = "python")]
use crate::indicators::chande::{chande_batch_py, chande_py, ChandeStreamPy};
#[cfg(feature = "python")]
use crate::indicators::correlation_cycle::{
	correlation_cycle_batch_py, correlation_cycle_py, CorrelationCycleStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::correl_hl::{correl_hl_batch_py, correl_hl_py, CorrelHlStreamPy};
#[cfg(feature = "python")]
use crate::indicators::deviation::{deviation_batch_py, deviation_py, DeviationStreamPy};
#[cfg(feature = "python")]
use crate::indicators::dti::{dti_batch_py, dti_py, DtiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::eri::{eri_batch_py, eri_py, EriStreamPy};
#[cfg(feature = "python")]
use crate::indicators::kdj::{kdj_batch_py, kdj_py, KdjStreamPy};
#[cfg(feature = "python")]
use crate::indicators::decycler::{decycler_batch_py, decycler_py, DecyclerStreamPy};
#[cfg(feature = "python")]
use crate::indicators::devstop::{devstop_batch_py, devstop_py};
#[cfg(feature = "python")]
use crate::indicators::dpo::{dpo_batch_py, dpo_py, DpoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::er::{er_batch_py, er_py, ErStreamPy};
#[cfg(feature = "python")]
use crate::indicators::kaufmanstop::{kaufmanstop_batch_py, kaufmanstop_py, KaufmanstopStreamPy};
#[cfg(feature = "python")]
use crate::indicators::linearreg_angle::{linearreg_angle_batch_py, linearreg_angle_py, Linearreg_angleStreamPy};
#[cfg(feature = "python")]
use crate::indicators::marketefi::{marketefi_batch_py, marketefi_py, MarketefiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::midpoint::{midpoint_batch_py, midpoint_py, MidpointStreamPy};
#[cfg(feature = "python")]
use crate::indicators::dec_osc::{dec_osc_batch_py, dec_osc_py, DecOscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::donchian::{donchian_batch_py, donchian_py, DonchianStreamPy};
#[cfg(feature = "python")]
use crate::indicators::emv::{emv_batch_py, emv_py, EmvStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ift_rsi::{ift_rsi_batch_py, ift_rsi_py, IftRsiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::kvo::{kvo_batch_py, kvo_py, KvoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::macd::{macd_batch_py, macd_py, MacdStreamPy};
#[cfg(feature = "python")]
use crate::indicators::mfi::{mfi_batch_py, mfi_py, MfiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::natr::{natr_batch_py, natr_py, NatrStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ppo::{ppo_batch_py, ppo_py, PpoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::rsi::{rsi_batch_py, rsi_py, RsiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::rsx::{rsx_batch_py, rsx_py, RsxStreamPy};
#[cfg(feature = "python")]
use crate::indicators::squeeze_momentum::{squeeze_momentum_batch_py, squeeze_momentum_py, SqueezeMomentumStreamPy};
#[cfg(feature = "python")]
use crate::indicators::trix::{trix_batch_py, trix_py, TrixStreamPy};
#[cfg(feature = "python")]
use crate::indicators::var::{var_batch_py, var_py, VarStreamPy};
#[cfg(feature = "python")]
use crate::indicators::vpci::{vpci_batch_py, vpci_py};
#[cfg(feature = "python")]
use crate::indicators::wclprice::{wclprice_batch_py, wclprice_py, WclpriceStreamPy};
#[cfg(feature = "python")]
use crate::indicators::damiani_volatmeter::{
	damiani_batch_py, damiani_py, DamianiVolatmeterStreamPy, DamianiVolatmeterFeedStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::emd::{emd_batch_py, emd_py, EmdStreamPy};
#[cfg(feature = "python")]
use crate::indicators::gatorosc::{gatorosc_batch_py, gatorosc_py, GatorOscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::kurtosis::{kurtosis_batch_py, kurtosis_py, KurtosisStreamPy};
#[cfg(feature = "python")]
use crate::indicators::mab::{mab_batch_py, mab_py, MabStreamPy};
#[cfg(feature = "python")]
use crate::indicators::medprice::{medprice_batch_py, medprice_py, MedpriceStreamPy};
#[cfg(feature = "python")]
use crate::indicators::msw::{msw_batch_py, msw_py, MswStreamPy};
#[cfg(feature = "python")]
use crate::indicators::pma::{pma_batch_py, pma_py, PmaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::rocr::{rocr_batch_py, rocr_py, RocrStreamPy};
#[cfg(feature = "python")]
use crate::indicators::sar::{sar_batch_py, sar_py, SarStreamPy};
#[cfg(feature = "python")]
use crate::indicators::supertrend::{supertrend_batch_py, supertrend_py, SuperTrendStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ultosc::{ultosc_batch_py, ultosc_py, UltOscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::voss::{voss_batch_py, voss_py, VossStreamPy};
#[cfg(feature = "python")]
use crate::indicators::wavetrend::{wavetrend_batch_py, wavetrend_py, WavetrendStreamPy};
#[cfg(feature = "python")]
use crate::indicators::cvi::{cvi_batch_py, cvi_py, CviStreamPy};
#[cfg(feature = "python")]
use crate::indicators::di::{di_batch_py, di_py, DiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::dm::{dm_batch_py, dm_py, DmStreamPy};
#[cfg(feature = "python")]
use crate::indicators::efi::{efi_batch_py, efi_py, EfiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::fosc::{fosc_batch_py, fosc_py, FoscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::kst::{kst_batch_py, kst_py, KstStreamPy};
#[cfg(feature = "python")]
use crate::indicators::lrsi::{lrsi_batch_py, lrsi_py, LrsiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::mean_ad::{mean_ad_batch_py, mean_ad_py, MeanAdStreamPy};
#[cfg(feature = "python")]
use crate::indicators::mom::{mom_batch_py, mom_py, MomStreamPy};
#[cfg(feature = "python")]
use crate::indicators::pivot::{pivot_batch_py, pivot_py, PivotStreamPy};
#[cfg(feature = "python")]
use crate::indicators::rocp::{rocp_batch_py, rocp_py, RocpStreamPy};
#[cfg(feature = "python")]
use crate::indicators::safezonestop::{safezonestop_batch_py, safezonestop_py, SafeZoneStopStreamPy};
#[cfg(feature = "python")]
use crate::indicators::stoch::{stoch_batch_py, stoch_py, StochStreamPy};
#[cfg(feature = "python")]
use crate::indicators::stochf::{stochf_batch_py, stochf_py, StochfStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ui::{ui_batch_py, ui_py, UiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::vosc::{vosc_batch_py, vosc_py, VoscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::wad::{wad_batch_py, wad_py, WadStreamPy};
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
use crate::indicators::moving_averages::ehlers_kama::{ehlers_kama_py, ehlers_kama_batch_py, EhlersKamaStreamPy};
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
use crate::indicators::linearreg_intercept::{
	linearreg_intercept_batch_py, linearreg_intercept_py, LinearRegInterceptStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::mass::{mass_batch_py, mass_py, MassStreamPy};
#[cfg(feature = "python")]
use crate::indicators::midprice::{midprice_batch_py, midprice_py, MidpriceStreamPy};
#[cfg(feature = "python")]
use crate::indicators::obv::{obv_batch_py, obv_py, ObvStreamPy};
#[cfg(feature = "python")]
use crate::indicators::qstick::{qstick_batch_py, qstick_py, QstickStreamPy};
#[cfg(feature = "python")]
use crate::indicators::stc::{stc_batch_py, stc_py, StcStreamPy};
#[cfg(feature = "python")]
use crate::indicators::tsi::{tsi_batch_py, tsi_py, TsiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::vidya::{vidya_batch_py, vidya_py, VidyaStreamPy};
#[cfg(feature = "python")]
use crate::indicators::willr::{willr_batch_py, willr_py, WillrStreamPy};
#[cfg(feature = "python")]
use crate::indicators::nvi::{nvi_py, nvi_batch_py, NviStreamPy};
#[cfg(feature = "python")]
use crate::indicators::pvi::{pvi_batch_py, pvi_py, PviStreamPy};
#[cfg(feature = "python")]
use crate::indicators::rsmk::{rsmk_batch_py, rsmk_py, RsmkStreamPy};
#[cfg(feature = "python")]
use crate::indicators::srsi::{srsi_batch_py, srsi_py, SrsiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::tsf::{tsf_batch_py, tsf_py, TsfStreamPy};
#[cfg(feature = "python")]
use crate::indicators::vi::{vi_batch_py, vi_py, ViStreamPy};
#[cfg(feature = "python")]
use crate::indicators::vpt::{vpt_batch_py, vpt_py, VptStreamPy};
#[cfg(feature = "python")]
use crate::indicators::zscore::{zscore_batch_py, zscore_py, ZscoreStreamPy};
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

	// Register Ehlers KAMA functions with their user-facing names
	m.add_function(wrap_pyfunction!(ehlers_kama_py, m)?)?;
	m.add_function(wrap_pyfunction!(ehlers_kama_batch_py, m)?)?;
	m.add_class::<EhlersKamaStreamPy>()?;

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

	// Register NVI functions with their user-facing names
	m.add_function(wrap_pyfunction!(nvi_py, m)?)?;
	m.add_function(wrap_pyfunction!(nvi_batch_py, m)?)?;
	m.add_class::<NviStreamPy>()?;

	// Register PVI functions with their user-facing names
	m.add_function(wrap_pyfunction!(pvi_py, m)?)?;
	m.add_function(wrap_pyfunction!(pvi_batch_py, m)?)?;
	m.add_class::<PviStreamPy>()?;

	// Register RSMK functions with their user-facing names
	m.add_function(wrap_pyfunction!(rsmk_py, m)?)?;
	m.add_function(wrap_pyfunction!(rsmk_batch_py, m)?)?;
	m.add_class::<RsmkStreamPy>()?;

	// Register SRSI functions with their user-facing names
	m.add_function(wrap_pyfunction!(srsi_py, m)?)?;
	m.add_function(wrap_pyfunction!(srsi_batch_py, m)?)?;
	m.add_class::<SrsiStreamPy>()?;

	// Register TSF functions with their user-facing names
	m.add_function(wrap_pyfunction!(tsf_py, m)?)?;
	m.add_function(wrap_pyfunction!(tsf_batch_py, m)?)?;
	m.add_class::<TsfStreamPy>()?;

	// Register VI functions with their user-facing names
	m.add_function(wrap_pyfunction!(vi_py, m)?)?;
	m.add_function(wrap_pyfunction!(vi_batch_py, m)?)?;
	m.add_class::<ViStreamPy>()?;

	// Register VPT functions with their user-facing names
	m.add_function(wrap_pyfunction!(vpt_py, m)?)?;
	m.add_function(wrap_pyfunction!(vpt_batch_py, m)?)?;
	m.add_class::<VptStreamPy>()?;

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

	// Register Coppock functions with their user-facing names
	m.add_function(wrap_pyfunction!(coppock_py, m)?)?;
	m.add_function(wrap_pyfunction!(coppock_batch_py, m)?)?;
	m.add_class::<CoppockStreamPy>()?;

	// Register CMO functions with their user-facing names
	m.add_function(wrap_pyfunction!(cmo_py, m)?)?;
	m.add_function(wrap_pyfunction!(cmo_batch_py, m)?)?;
	m.add_class::<CmoStreamPy>()?;

	// Register CKSP functions with their user-facing names
	m.add_function(wrap_pyfunction!(cksp_py, m)?)?;
	m.add_function(wrap_pyfunction!(cksp_batch_py, m)?)?;
	m.add_class::<CkspStreamPy>()?;

	// Register CHOP functions with their user-facing names
	m.add_function(wrap_pyfunction!(chop_py, m)?)?;
	m.add_function(wrap_pyfunction!(chop_batch_py, m)?)?;
	m.add_class::<ChopStreamPy>()?;

	// Register Correlation Cycle functions with their user-facing names
	m.add_function(wrap_pyfunction!(correlation_cycle_py, m)?)?;
	m.add_function(wrap_pyfunction!(correlation_cycle_batch_py, m)?)?;
	m.add_class::<CorrelationCycleStreamPy>()?;

	// Register Correl HL functions with their user-facing names
	m.add_function(wrap_pyfunction!(correl_hl_py, m)?)?;
	m.add_function(wrap_pyfunction!(correl_hl_batch_py, m)?)?;
	m.add_class::<CorrelHlStreamPy>()?;

	// Register Deviation functions with their user-facing names
	m.add_function(wrap_pyfunction!(deviation_py, m)?)?;
	m.add_function(wrap_pyfunction!(deviation_batch_py, m)?)?;
	m.add_class::<DeviationStreamPy>()?;

	// Register DTI functions with their user-facing names
	m.add_function(wrap_pyfunction!(dti_py, m)?)?;
	m.add_function(wrap_pyfunction!(dti_batch_py, m)?)?;
	m.add_class::<DtiStreamPy>()?;

	// Register ERI functions with their user-facing names
	m.add_function(wrap_pyfunction!(eri_py, m)?)?;
	m.add_function(wrap_pyfunction!(eri_batch_py, m)?)?;
	m.add_class::<EriStreamPy>()?;

	// Register KDJ functions with their user-facing names
	m.add_function(wrap_pyfunction!(kdj_py, m)?)?;
	m.add_function(wrap_pyfunction!(kdj_batch_py, m)?)?;
	m.add_class::<KdjStreamPy>()?;

	// Register Decycler functions with their user-facing names
	m.add_function(wrap_pyfunction!(decycler_py, m)?)?;
	m.add_function(wrap_pyfunction!(decycler_batch_py, m)?)?;
	m.add_class::<DecyclerStreamPy>()?;

	// Register DevStop functions with their user-facing names
	m.add_function(wrap_pyfunction!(devstop_py, m)?)?;
	m.add_function(wrap_pyfunction!(devstop_batch_py, m)?)?;

	// Register DPO functions with their user-facing names
	m.add_function(wrap_pyfunction!(dpo_py, m)?)?;
	m.add_function(wrap_pyfunction!(dpo_batch_py, m)?)?;
	m.add_class::<DpoStreamPy>()?;

	// Register ER functions with their user-facing names
	m.add_function(wrap_pyfunction!(er_py, m)?)?;
	m.add_function(wrap_pyfunction!(er_batch_py, m)?)?;
	m.add_class::<ErStreamPy>()?;

	// Register Kaufmanstop functions with their user-facing names
	m.add_function(wrap_pyfunction!(kaufmanstop_py, m)?)?;
	m.add_function(wrap_pyfunction!(kaufmanstop_batch_py, m)?)?;
	m.add_class::<KaufmanstopStreamPy>()?;

	// Register Linear Regression Angle functions with their user-facing names
	m.add_function(wrap_pyfunction!(linearreg_angle_py, m)?)?;
	m.add_function(wrap_pyfunction!(linearreg_angle_batch_py, m)?)?;
	m.add_class::<Linearreg_angleStreamPy>()?;

	// Register MarketEFI functions with their user-facing names
	m.add_function(wrap_pyfunction!(marketefi_py, m)?)?;
	m.add_function(wrap_pyfunction!(marketefi_batch_py, m)?)?;
	m.add_class::<MarketefiStreamPy>()?;

	// Register Midpoint functions with their user-facing names
	m.add_function(wrap_pyfunction!(midpoint_py, m)?)?;
	m.add_function(wrap_pyfunction!(midpoint_batch_py, m)?)?;
	m.add_class::<MidpointStreamPy>()?;

	// Register Decycler Oscillator functions with their user-facing names
	m.add_function(wrap_pyfunction!(dec_osc_py, m)?)?;
	m.add_function(wrap_pyfunction!(dec_osc_batch_py, m)?)?;
	m.add_class::<DecOscStreamPy>()?;

	// Register Donchian Channel functions with their user-facing names
	m.add_function(wrap_pyfunction!(donchian_py, m)?)?;
	m.add_function(wrap_pyfunction!(donchian_batch_py, m)?)?;
	m.add_class::<DonchianStreamPy>()?;

	// Register EMV functions with their user-facing names
	m.add_function(wrap_pyfunction!(emv_py, m)?)?;
	m.add_function(wrap_pyfunction!(emv_batch_py, m)?)?;
	m.add_class::<EmvStreamPy>()?;

	// Register IFT RSI functions with their user-facing names
	m.add_function(wrap_pyfunction!(ift_rsi_py, m)?)?;
	m.add_function(wrap_pyfunction!(ift_rsi_batch_py, m)?)?;
	m.add_class::<IftRsiStreamPy>()?;

	// Register KVO functions with their user-facing names
	m.add_function(wrap_pyfunction!(kvo_py, m)?)?;
	m.add_function(wrap_pyfunction!(kvo_batch_py, m)?)?;
	m.add_class::<KvoStreamPy>()?;

	// Register MACD functions with their user-facing names
	m.add_function(wrap_pyfunction!(macd_py, m)?)?;
	m.add_function(wrap_pyfunction!(macd_batch_py, m)?)?;
	m.add_class::<MacdStreamPy>()?;

	// Register MFI functions with their user-facing names
	m.add_function(wrap_pyfunction!(mfi_py, m)?)?;
	m.add_function(wrap_pyfunction!(mfi_batch_py, m)?)?;
	m.add_class::<MfiStreamPy>()?;

	// Register NATR functions with their user-facing names
	m.add_function(wrap_pyfunction!(natr_py, m)?)?;
	m.add_function(wrap_pyfunction!(natr_batch_py, m)?)?;
	m.add_class::<NatrStreamPy>()?;

	// Register PPO functions with their user-facing names
	m.add_function(wrap_pyfunction!(ppo_py, m)?)?;
	m.add_function(wrap_pyfunction!(ppo_batch_py, m)?)?;
	m.add_class::<PpoStreamPy>()?;

	// Register RSI functions with their user-facing names
	m.add_function(wrap_pyfunction!(rsi_py, m)?)?;
	m.add_function(wrap_pyfunction!(rsi_batch_py, m)?)?;
	m.add_class::<RsiStreamPy>()?;

	// Register RSX functions with their user-facing names
	m.add_function(wrap_pyfunction!(rsx_py, m)?)?;
	m.add_function(wrap_pyfunction!(rsx_batch_py, m)?)?;
	m.add_class::<RsxStreamPy>()?;

	// Register Squeeze Momentum functions with their user-facing names
	m.add_function(wrap_pyfunction!(squeeze_momentum_py, m)?)?;
	m.add_function(wrap_pyfunction!(squeeze_momentum_batch_py, m)?)?;
	m.add_class::<SqueezeMomentumStreamPy>()?;

	// Register TRIX functions with their user-facing names
	m.add_function(wrap_pyfunction!(trix_py, m)?)?;
	m.add_function(wrap_pyfunction!(trix_batch_py, m)?)?;
	m.add_class::<TrixStreamPy>()?;

	// Register VAR functions with their user-facing names
	m.add_function(wrap_pyfunction!(var_py, m)?)?;
	m.add_function(wrap_pyfunction!(var_batch_py, m)?)?;
	m.add_class::<VarStreamPy>()?;

	// Register VPCI functions with their user-facing names
	m.add_function(wrap_pyfunction!(vpci_py, m)?)?;
	m.add_function(wrap_pyfunction!(vpci_batch_py, m)?)?;

	// Register WCLPRICE functions with their user-facing names
	m.add_function(wrap_pyfunction!(wclprice_py, m)?)?;
	m.add_function(wrap_pyfunction!(wclprice_batch_py, m)?)?;
	m.add_class::<WclpriceStreamPy>()?;

	// Register Damiani Volatmeter functions with their user-facing names
	m.add_function(wrap_pyfunction!(damiani_py, m)?)?;
	m.add_function(wrap_pyfunction!(damiani_batch_py, m)?)?;
	m.add_class::<DamianiVolatmeterStreamPy>()?;
	m.add_class::<DamianiVolatmeterFeedStreamPy>()?;

	// Register EMD functions with their user-facing names
	m.add_function(wrap_pyfunction!(emd_py, m)?)?;
	m.add_function(wrap_pyfunction!(emd_batch_py, m)?)?;
	m.add_class::<EmdStreamPy>()?;

	// Register CVI functions with their user-facing names
	m.add_function(wrap_pyfunction!(cvi_py, m)?)?;
	m.add_function(wrap_pyfunction!(cvi_batch_py, m)?)?;
	m.add_class::<CviStreamPy>()?;

	// Register DI functions with their user-facing names
	m.add_function(wrap_pyfunction!(di_py, m)?)?;
	m.add_function(wrap_pyfunction!(di_batch_py, m)?)?;
	m.add_class::<DiStreamPy>()?;

	// Register DM functions with their user-facing names
	m.add_function(wrap_pyfunction!(dm_py, m)?)?;
	m.add_function(wrap_pyfunction!(dm_batch_py, m)?)?;
	m.add_class::<DmStreamPy>()?;

	// Register EFI functions with their user-facing names
	m.add_function(wrap_pyfunction!(efi_py, m)?)?;
	m.add_function(wrap_pyfunction!(efi_batch_py, m)?)?;
	m.add_class::<EfiStreamPy>()?;

	// Register FOSC functions with their user-facing names
	m.add_function(wrap_pyfunction!(fosc_py, m)?)?;
	m.add_function(wrap_pyfunction!(fosc_batch_py, m)?)?;
	m.add_class::<FoscStreamPy>()?;

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

	// Register BOP functions with their user-facing names
	m.add_function(wrap_pyfunction!(bop_py, m)?)?;
	m.add_function(wrap_pyfunction!(bop_batch_py, m)?)?;
	m.add_class::<BopStreamPy>()?;

	// Buff Averages
	m.add_function(wrap_pyfunction!(buff_averages_py, m)?)?;
	m.add_function(wrap_pyfunction!(buff_averages_batch_py, m)?)?;
	m.add_class::<BuffAveragesStreamPy>()?;

	// QQE
	m.add_function(wrap_pyfunction!(qqe_py, m)?)?;
	m.add_function(wrap_pyfunction!(qqe_batch_py, m)?)?;
	m.add_class::<QqeStreamPy>()?;

	// Volume Adjusted MA
	m.add_function(wrap_pyfunction!(volume_adjusted_ma_py, m)?)?;
	m.add_function(wrap_pyfunction!(volume_adjusted_ma_batch_py, m)?)?;
	m.add_class::<VolumeAdjustedMaStreamPy>()?;

	// Nadaraya-Watson Envelope
	m.add_function(wrap_pyfunction!(nadaraya_watson_envelope_py, m)?)?;
	m.add_function(wrap_pyfunction!(nadaraya_watson_envelope_batch_py, m)?)?;
	m.add_class::<NweStreamPy>()?;
	
	// TTM Squeeze
	m.add_function(wrap_pyfunction!(ttm_squeeze_py, m)?)?;
	m.add_function(wrap_pyfunction!(ttm_squeeze_batch_py, m)?)?;
	m.add_class::<TtmSqueezeStreamPy>()?;
	
	// Modified God Mode
	m.add_function(wrap_pyfunction!(mod_god_mode_py, m)?)?;
	m.add_function(wrap_pyfunction!(mod_god_mode_batch_py, m)?)?;
	m.add_class::<ModGodModeStreamPy>()?;

	// Register Linear Regression Intercept functions with their user-facing names
	m.add_function(wrap_pyfunction!(linearreg_intercept_py, m)?)?;
	m.add_function(wrap_pyfunction!(linearreg_intercept_batch_py, m)?)?;
	m.add_class::<LinearRegInterceptStreamPy>()?;

	// Register Mass Index functions with their user-facing names
	m.add_function(wrap_pyfunction!(mass_py, m)?)?;
	m.add_function(wrap_pyfunction!(mass_batch_py, m)?)?;
	m.add_class::<MassStreamPy>()?;

	// Register Midprice functions with their user-facing names
	m.add_function(wrap_pyfunction!(midprice_py, m)?)?;
	m.add_function(wrap_pyfunction!(midprice_batch_py, m)?)?;
	m.add_class::<MidpriceStreamPy>()?;

	// Register OBV functions with their user-facing names
	m.add_function(wrap_pyfunction!(obv_py, m)?)?;
	m.add_function(wrap_pyfunction!(obv_batch_py, m)?)?;
	m.add_class::<ObvStreamPy>()?;

	// Register Qstick functions with their user-facing names
	m.add_function(wrap_pyfunction!(qstick_py, m)?)?;
	m.add_function(wrap_pyfunction!(qstick_batch_py, m)?)?;
	m.add_class::<QstickStreamPy>()?;

	// Register RSX functions with their user-facing names
	m.add_function(wrap_pyfunction!(rsx_py, m)?)?;
	m.add_function(wrap_pyfunction!(rsx_batch_py, m)?)?;
	m.add_class::<RsxStreamPy>()?;

	// Register STC functions with their user-facing names
	m.add_function(wrap_pyfunction!(stc_py, m)?)?;
	m.add_function(wrap_pyfunction!(stc_batch_py, m)?)?;
	m.add_class::<StcStreamPy>()?;

	// Register TSI functions with their user-facing names
	m.add_function(wrap_pyfunction!(tsi_py, m)?)?;
	m.add_function(wrap_pyfunction!(tsi_batch_py, m)?)?;
	m.add_class::<TsiStreamPy>()?;

	// Register VIDYA functions with their user-facing names
	m.add_function(wrap_pyfunction!(vidya_py, m)?)?;
	m.add_function(wrap_pyfunction!(vidya_batch_py, m)?)?;
	m.add_class::<VidyaStreamPy>()?;
	
	// Register WILLR functions with their user-facing names
	m.add_function(wrap_pyfunction!(willr_py, m)?)?;
	m.add_function(wrap_pyfunction!(willr_batch_py, m)?)?;
	m.add_class::<WillrStreamPy>()?;

	// Register ZSCORE functions with their user-facing names
	m.add_function(wrap_pyfunction!(zscore_py, m)?)?;
	m.add_function(wrap_pyfunction!(zscore_batch_py, m)?)?;
	m.add_class::<ZscoreStreamPy>()?;

	// Register GatorOsc functions with their user-facing names
	m.add_function(wrap_pyfunction!(gatorosc_py, m)?)?;
	m.add_function(wrap_pyfunction!(gatorosc_batch_py, m)?)?;
	m.add_class::<GatorOscStreamPy>()?;

	// Register Kurtosis functions with their user-facing names
	m.add_function(wrap_pyfunction!(kurtosis_py, m)?)?;
	m.add_function(wrap_pyfunction!(kurtosis_batch_py, m)?)?;
	m.add_class::<KurtosisStreamPy>()?;

	// Register MAB functions with their user-facing names
	m.add_function(wrap_pyfunction!(mab_py, m)?)?;
	m.add_function(wrap_pyfunction!(mab_batch_py, m)?)?;
	m.add_class::<MabStreamPy>()?;

	// Register medprice functions
	m.add_function(wrap_pyfunction!(medprice_py, m)?)?;
	m.add_function(wrap_pyfunction!(medprice_batch_py, m)?)?;
	m.add_class::<MedpriceStreamPy>()?;

	// Register MSW functions with their user-facing names
	m.add_function(wrap_pyfunction!(msw_py, m)?)?;
	m.add_function(wrap_pyfunction!(msw_batch_py, m)?)?;
	m.add_class::<MswStreamPy>()?;

	// Register PMA functions with their user-facing names
	m.add_function(wrap_pyfunction!(pma_py, m)?)?;
	m.add_function(wrap_pyfunction!(pma_batch_py, m)?)?;
	m.add_class::<PmaStreamPy>()?;

	// Register ROCR functions with their user-facing names
	m.add_function(wrap_pyfunction!(rocr_py, m)?)?;
	m.add_function(wrap_pyfunction!(rocr_batch_py, m)?)?;
	m.add_class::<RocrStreamPy>()?;

	// Register SAR functions with their user-facing names
	m.add_function(wrap_pyfunction!(sar_py, m)?)?;
	m.add_function(wrap_pyfunction!(sar_batch_py, m)?)?;
	m.add_class::<SarStreamPy>()?;

	// Register SuperTrend functions with their user-facing names
	m.add_function(wrap_pyfunction!(supertrend_py, m)?)?;
	m.add_function(wrap_pyfunction!(supertrend_batch_py, m)?)?;
	m.add_class::<SuperTrendStreamPy>()?;

	// Register UltOsc functions with their user-facing names
	m.add_function(wrap_pyfunction!(ultosc_py, m)?)?;
	m.add_function(wrap_pyfunction!(ultosc_batch_py, m)?)?;
	m.add_class::<UltOscStreamPy>()?;

	// Register Voss functions with their user-facing names
	m.add_function(wrap_pyfunction!(voss_py, m)?)?;
	m.add_function(wrap_pyfunction!(voss_batch_py, m)?)?;
	m.add_class::<VossStreamPy>()?;

	// Register Wavetrend functions with their user-facing names
	m.add_function(wrap_pyfunction!(wavetrend_py, m)?)?;
	m.add_function(wrap_pyfunction!(wavetrend_batch_py, m)?)?;
	m.add_class::<WavetrendStreamPy>()?;

	// Register KST functions with their user-facing names
	m.add_function(wrap_pyfunction!(kst_py, m)?)?;
	m.add_function(wrap_pyfunction!(kst_batch_py, m)?)?;
	m.add_class::<KstStreamPy>()?;

	// Register LRSI functions with their user-facing names
	m.add_function(wrap_pyfunction!(lrsi_py, m)?)?;
	m.add_function(wrap_pyfunction!(lrsi_batch_py, m)?)?;
	m.add_class::<LrsiStreamPy>()?;

	// Register Mean AD functions with their user-facing names
	m.add_function(wrap_pyfunction!(mean_ad_py, m)?)?;
	m.add_function(wrap_pyfunction!(mean_ad_batch_py, m)?)?;
	m.add_class::<MeanAdStreamPy>()?;

	// Register MOM functions with their user-facing names
	m.add_function(wrap_pyfunction!(mom_py, m)?)?;
	m.add_function(wrap_pyfunction!(mom_batch_py, m)?)?;
	m.add_class::<MomStreamPy>()?;

	// Register Pivot functions with their user-facing names
	m.add_function(wrap_pyfunction!(pivot_py, m)?)?;
	m.add_function(wrap_pyfunction!(pivot_batch_py, m)?)?;
	m.add_class::<PivotStreamPy>()?;

	// Register ROCP functions with their user-facing names
	m.add_function(wrap_pyfunction!(rocp_py, m)?)?;
	m.add_function(wrap_pyfunction!(rocp_batch_py, m)?)?;
	m.add_class::<RocpStreamPy>()?;

	// Register SafeZoneStop functions with their user-facing names
	m.add_function(wrap_pyfunction!(safezonestop_py, m)?)?;
	m.add_function(wrap_pyfunction!(safezonestop_batch_py, m)?)?;
	m.add_class::<SafeZoneStopStreamPy>()?;

	// Register Stoch functions with their user-facing names
	m.add_function(wrap_pyfunction!(stoch_py, m)?)?;
	m.add_function(wrap_pyfunction!(stoch_batch_py, m)?)?;
	m.add_class::<StochStreamPy>()?;

	// Register StochF functions with their user-facing names
	m.add_function(wrap_pyfunction!(stochf_py, m)?)?;
	m.add_function(wrap_pyfunction!(stochf_batch_py, m)?)?;
	m.add_class::<StochfStreamPy>()?;

	// Register UI functions with their user-facing names
	m.add_function(wrap_pyfunction!(ui_py, m)?)?;
	m.add_function(wrap_pyfunction!(ui_batch_py, m)?)?;
	m.add_class::<UiStreamPy>()?;

	// Register VOSC functions with their user-facing names
	m.add_function(wrap_pyfunction!(vosc_py, m)?)?;
	m.add_function(wrap_pyfunction!(vosc_batch_py, m)?)?;
	m.add_class::<VoscStreamPy>()?;

	// Register WAD functions with their user-facing names
	m.add_function(wrap_pyfunction!(wad_py, m)?)?;
	m.add_function(wrap_pyfunction!(wad_batch_py, m)?)?;
	m.add_class::<WadStreamPy>()?;

	// Register Chande functions with their user-facing names
	m.add_function(wrap_pyfunction!(chande_py, m)?)?;
	m.add_function(wrap_pyfunction!(chande_batch_py, m)?)?;
	m.add_class::<ChandeStreamPy>()?;

	// Add other indicators here as you implement their Python bindings

	Ok(())
}
