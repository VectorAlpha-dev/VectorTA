use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};

// Re-export all Python functions and classes from indicators
// Add module initialization here

#[cfg(feature = "python")]
use crate::indicators::acosc::{acosc_batch_py, acosc_py, AcoscStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::acosc::{acosc_cuda_batch_dev_py, acosc_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::ad::{ad_batch_py, ad_py, AdStreamPy};
#[cfg(feature = "python")]
use crate::indicators::adosc::{adosc_batch_py, adosc_py, AdoscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::adx::{adx_batch_py, adx_py, AdxStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::adx::{adx_cuda_batch_dev_py, adx_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::adxr::{adxr_batch_py, adxr_py, AdxrStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::adxr::{adxr_cuda_batch_dev_py, adxr_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::alligator::{alligator_batch_py, alligator_py, AlligatorStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::alligator::{
    alligator_cuda_batch_dev_py, alligator_cuda_many_series_one_param_dev_py,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::alphatrend::{
    alphatrend_cuda_batch_dev_py, alphatrend_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::alphatrend::{alphatrend_py, AlphaTrendStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ao::{ao_batch_py, ao_py, AoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::apo::{apo_batch_py, apo_py, ApoStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::apo::{apo_cuda_batch_dev_py, apo_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::aroon::{aroon_batch_py, aroon_py, AroonStreamPy};
#[cfg(feature = "python")]
use crate::indicators::aroonosc::{aroon_osc_batch_py, aroon_osc_py, AroonOscStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::aroonosc::{
    aroonosc_cuda_batch_dev_py, aroonosc_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::aso::{aso_batch_py, aso_py, AsoStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::aso::{aso_cuda_batch_dev_py, aso_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::atr::{atr_batch_py, atr_py, AtrStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::atr::{atr_cuda_batch_dev_py, atr_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::avsl::{avsl_batch_py, avsl_py, AvslStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::avsl::{avsl_cuda_batch_dev_py, avsl_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::bandpass::{bandpass_batch_py, bandpass_py, BandPassStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::bandpass::{
    bandpass_cuda_batch_dev_py, bandpass_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::bollinger_bands::{
    bollinger_bands_batch_py, bollinger_bands_py, BollingerBandsStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::bollinger_bands_width::{
    bollinger_bands_width_batch_py, bollinger_bands_width_py, BollingerBandsWidthStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::bollinger_bands_width::{
    bollinger_bands_width_cuda_batch_dev_py,
    bollinger_bands_width_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::bop::{bop_batch_py, bop_py, BopStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::bop::{bop_cuda_batch_dev_py, bop_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::cci::{cci_batch_py, cci_py, CciStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::cci::{
    cci_cuda_batch_dev_py, cci_cuda_many_series_one_param_dev_py, CciDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::cci_cycle::{cci_cycle_batch_py, cci_cycle_py, CciCycleStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::cci_cycle::{
    cci_cycle_cuda_batch_dev_py, cci_cycle_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::cfo::{cfo_batch_py, cfo_py, CfoStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::cfo::{cfo_cuda_batch_dev_py, cfo_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::cg::{cg_batch_py, cg_py, CgStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::cg::{cg_cuda_batch_dev_py, cg_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::chande::{chande_batch_py, chande_py, ChandeStreamPy};
#[cfg(feature = "python")]
use crate::indicators::chandelier_exit::{
    chandelier_exit_batch_py, chandelier_exit_py, ChandelierExitStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::chop::{chop_batch_py, chop_py, ChopStreamPy};
#[cfg(feature = "python")]
use crate::indicators::cksp::{cksp_batch_py, cksp_py, CkspStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::cksp::{cksp_cuda_batch_dev_py, cksp_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::cmo::{cmo_batch_py, cmo_py, CmoStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::cmo::{cmo_cuda_batch_dev_py, cmo_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::coppock::{coppock_batch_py, coppock_py, CoppockStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::coppock::{
    coppock_cuda_batch_dev_py, coppock_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::cora_wave::{cora_wave_batch_py, cora_wave_py, CoraWaveStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::cora_wave::{
    cora_wave_cuda_batch_dev_py, cora_wave_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::correl_hl::{correl_hl_batch_py, correl_hl_py, CorrelHlStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::correl_hl::{
    correl_hl_cuda_batch_dev_py, correl_hl_cuda_many_series_one_param_dev_py,
    CorrelHlDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::correlation_cycle::{
    correlation_cycle_batch_py, correlation_cycle_py, CorrelationCycleStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::correlation_cycle::{
    correlation_cycle_cuda_batch_dev_py, correlation_cycle_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::cvi::{cvi_batch_py, cvi_py, CviStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::cvi::{cvi_cuda_batch_dev_py, cvi_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::damiani_volatmeter::{
    damiani_batch_py, damiani_py, DamianiVolatmeterFeedStreamPy, DamianiVolatmeterStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::dec_osc::{dec_osc_batch_py, dec_osc_py, DecOscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::decycler::{decycler_batch_py, decycler_py, DecyclerStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::decycler::{
    decycler_cuda_batch_dev_py, decycler_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::deviation::{deviation_batch_py, deviation_py, DeviationStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::deviation::{
    deviation_cuda_batch_dev_py, deviation_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::devstop::{devstop_batch_py, devstop_py};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::devstop::{
    devstop_cuda_batch_dev_py, devstop_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::di::{di_batch_py, di_py, DiStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::di::{di_cuda_batch_dev_py, di_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::dm::{dm_batch_py, dm_py, DmStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::dm::{dm_cuda_batch_dev_py, dm_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::donchian::{donchian_batch_py, donchian_py, DonchianStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::donchian::{
    donchian_cuda_batch_dev_py, donchian_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::dpo::{dpo_batch_py, dpo_py, DpoStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::dpo::{dpo_cuda_batch_dev_py, dpo_cuda_many_series_one_param_dev_py};
#[cfg(feature = "cuda")]
use crate::indicators::dpo::DpoDeviceArrayF32Py;
#[cfg(feature = "python")]
use crate::indicators::dti::{dti_batch_py, dti_py, DtiStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::dti::{dti_cuda_batch_dev_py, dti_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::dvdiqqe::{dvdiqqe_batch_py, dvdiqqe_py};
#[cfg(feature = "python")]
use crate::indicators::dx::{dx_batch_py, dx_py, DxStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::dx::{dx_cuda_batch_dev_py, dx_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::efi::{efi_batch_py, efi_py, EfiStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::efi::{
    efi_cuda_batch_dev_py, efi_cuda_many_series_one_param_dev_py, EfiDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::emd::{emd_batch_py, emd_py, EmdStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::emd::{emd_cuda_batch_dev_py, emd_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::emv::{emv_batch_py, emv_py, EmvStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::emv::{emv_cuda_batch_dev_py, emv_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::er::{er_batch_py, er_py, ErStreamPy};
#[cfg(feature = "python")]
use crate::indicators::eri::{eri_batch_py, eri_py, EriStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::eri::{eri_cuda_batch_dev_py, eri_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::fisher::{fisher_batch_py, fisher_py, FisherStreamPy};
#[cfg(feature = "python")]
use crate::indicators::fosc::{fosc_batch_py, fosc_py, FoscStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::fosc::{fosc_cuda_batch_dev_py, fosc_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::fvg_trailing_stop::{
    fvg_trailing_stop_batch_py, fvg_trailing_stop_py, FvgTrailingStopStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::fvg_trailing_stop::{
    fvg_trailing_stop_cuda_batch_dev_py, fvg_trailing_stop_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::gatorosc::{gatorosc_batch_py, gatorosc_py, GatorOscStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::gatorosc::{
    gatorosc_cuda_batch_dev_py, gatorosc_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::halftrend::{
    halftrend_batch_py, halftrend_py, halftrend_tuple_py, HalfTrendStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::halftrend::{
    halftrend_cuda_batch_dev_py, halftrend_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::ift_rsi::{ift_rsi_batch_py, ift_rsi_py, IftRsiStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::ift_rsi::{
    ift_rsi_cuda_batch_dev_py, ift_rsi_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::kaufmanstop::{kaufmanstop_batch_py, kaufmanstop_py, KaufmanstopStreamPy};
#[cfg(feature = "python")]
use crate::indicators::kdj::{kdj_batch_py, kdj_py, KdjStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::kdj::{kdj_cuda_batch_dev_py, kdj_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::keltner::{keltner_batch_py, keltner_py, KeltnerStreamPy};
#[cfg(feature = "python")]
use crate::indicators::kst::{kst_batch_py, kst_py, KstStreamPy};
#[cfg(feature = "python")]
use crate::indicators::kurtosis::{kurtosis_batch_py, kurtosis_py, KurtosisStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::kurtosis::{
    kurtosis_cuda_batch_dev_py, kurtosis_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::kvo::{kvo_batch_py, kvo_py, KvoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::linearreg_angle::{
    linearreg_angle_batch_py, linearreg_angle_py, Linearreg_angleStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::linearreg_intercept::{
    linearreg_intercept_batch_py, linearreg_intercept_py, LinearRegInterceptStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::linearreg_slope::{
    linearreg_slope_batch_py, linearreg_slope_py, LinearRegSlopeStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::linearreg_slope::{
    linearreg_slope_cuda_batch_dev_py, linearreg_slope_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::lpc::{lpc_batch_py, lpc_py, LpcStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::lpc::{lpc_cuda_batch_dev_py, lpc_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::lrsi::{lrsi_batch_py, lrsi_py, LrsiStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::lrsi::{lrsi_cuda_batch_dev_py, lrsi_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::mab::{mab_batch_py, mab_py, MabStreamPy};
#[cfg(feature = "cuda")]
use crate::indicators::mab::{mab_cuda_batch_dev_py, mab_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::macd::{macd_batch_py, macd_py, MacdStreamPy};
#[cfg(feature = "python")]
use crate::indicators::macz::{macz_batch_py, macz_py, MaczStreamPy};
#[cfg(feature = "cuda")]
use crate::indicators::macz::{macz_cuda_batch_dev_py, macz_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::marketefi::{marketefi_batch_py, marketefi_py, MarketefiStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::marketefi::{
    marketefi_cuda_batch_dev_py, marketefi_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::mass::{mass_batch_py, mass_py, MassStreamPy};
#[cfg(feature = "python")]
use crate::indicators::mean_ad::{mean_ad_batch_py, mean_ad_py, MeanAdStreamPy};
#[cfg(feature = "cuda")]
use crate::indicators::mean_ad::{
    mean_ad_cuda_batch_dev_py, mean_ad_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::medium_ad::{medium_ad_batch_py, medium_ad_py, MediumAdStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::medium_ad::{
    medium_ad_cuda_batch_dev_py, medium_ad_cuda_many_series_one_param_dev_py,
    MediumAdDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::medprice::{medprice_batch_py, medprice_py, MedpriceStreamPy};
#[cfg(feature = "python")]
use crate::indicators::mfi::{mfi_batch_py, mfi_py, MfiStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::mfi::{
    mfi_cuda_batch_dev_py, mfi_cuda_many_series_one_param_dev_py, MfiDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::midpoint::{midpoint_batch_py, midpoint_py, MidpointStreamPy};
#[cfg(feature = "python")]
use crate::indicators::midprice::{midprice_batch_py, midprice_py, MidpriceStreamPy};
#[cfg(feature = "python")]
use crate::indicators::minmax::{minmax_batch_py, minmax_py, MinmaxStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::minmax::{
    minmax_cuda_batch_dev_py, minmax_cuda_many_series_one_param_dev_py, MinmaxDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::mod_god_mode::{mod_god_mode_batch_py, mod_god_mode_py, ModGodModeStreamPy};
#[cfg(feature = "python")]
use crate::indicators::mom::{mom_batch_py, mom_py, MomStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::mom::{mom_cuda_batch_dev_py, mom_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::alma::{alma_batch_py, alma_py, AlmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::alma::{
    alma_cuda_batch_dev_py, alma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::buff_averages::{
    buff_averages_batch_py, buff_averages_py, BuffAveragesStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::buff_averages::{
    buff_averages_cuda_batch_dev_py, buff_averages_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::cwma::{cwma_batch_py, cwma_py, CwmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::cwma::{
    cwma_cuda_batch_dev_py, cwma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::dema::{dema_batch_py, dema_py, DemaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::dema::{
    dema_cuda_batch_dev_py, dema_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::dma::{dma_batch_py, dma_py, DmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::dma::{
    dma_cuda_batch_dev_py, dma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::edcf::{edcf_batch_py, edcf_py, EdcfStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::edcf::{
    edcf_cuda_batch_dev_py, edcf_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::ehlers_ecema::{
    ehlers_ecema_batch_py, ehlers_ecema_py, EhlersEcemaStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::ehlers_ecema::{
    ehlers_ecema_cuda_batch_dev_py, ehlers_ecema_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::ehlers_itrend::{
    ehlers_itrend_batch_py, ehlers_itrend_py, EhlersITrendStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::ehlers_itrend::{
    ehlers_itrend_cuda_batch_dev_py, ehlers_itrend_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::ehlers_kama::{
    ehlers_kama_batch_py, ehlers_kama_py, EhlersKamaStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::ehlers_kama::{
    ehlers_kama_cuda_batch_dev_py, ehlers_kama_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::ehlers_pma::{
    ehlers_pma_batch_py, ehlers_pma_flat_py, ehlers_pma_py, EhlersPmaStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::ehlers_pma::{
    ehlers_pma_cuda_batch_dev_py, ehlers_pma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::ehma::{ehma_batch_py, ehma_py, EhmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::ehma::{
    ehma_cuda_batch_dev_py, ehma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::ema::{ema_batch_py, ema_py, EmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::ema::{
    ema_cuda_batch_dev_py, ema_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::epma::{epma_batch_py, epma_py, EpmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::epma::{
    epma_cuda_batch_dev_py, epma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::frama::{frama_batch_py, frama_py, FramaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::frama::{
    frama_cuda_batch_dev_py, frama_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::fwma::{fwma_batch_py, fwma_py, FwmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::fwma::{
    fwma_cuda_batch_dev_py, fwma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::gaussian::{
    gaussian_batch_py, gaussian_py, GaussianStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::gaussian::{
    gaussian_cuda_batch_dev_py, gaussian_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::highpass::{
    highpass_batch_py, highpass_py, HighPassStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::highpass::{
    highpass_cuda_batch_dev_py, highpass_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::highpass_2_pole::{
    highpass_2_pole_batch_py, highpass_2_pole_py, HighPass2StreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::highpass_2_pole::{
    highpass_2_pole_cuda_batch_dev_py, highpass_2_pole_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::hma::{hma_batch_py, hma_py, HmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::hma::{
    hma_cuda_batch_dev_py, hma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::hwma::{hwma_batch_py, hwma_py, HwmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::hwma::{
    hwma_cuda_batch_dev_py, hwma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::jma::{jma_batch_py, jma_py, JmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::jma::{
    jma_cuda_batch_dev_py, jma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::jsa::{jsa_batch_py, jsa_py, JsaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::jsa::{
    jsa_cuda_batch_dev_py, jsa_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::kama::{kama_batch_py, kama_py, KamaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::kama::{
    kama_cuda_batch_dev_py, kama_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::linreg::{linreg_batch_py, linreg_py, LinRegStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::linreg::{
    linreg_cuda_batch_dev_py, linreg_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::ma::ma_py;
#[cfg(feature = "python")]
use crate::indicators::moving_averages::maaq::{maaq_batch_py, maaq_py, MaaqStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::maaq::{
    maaq_cuda_batch_dev_py, maaq_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::mama::{mama_batch_py, mama_py, MamaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::mama::{
    mama_cuda_batch_dev_py, mama_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::mwdx::{mwdx_batch_py, mwdx_py, MwdxStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::mwdx::{
    mwdx_cuda_batch_dev_py, mwdx_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::nama::{nama_batch_py, nama_py, NamaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::nama::{
    nama_cuda_batch_dev_py, nama_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::nma::{nma_batch_py, nma_py, NmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::nma::{
    nma_cuda_batch_dev_py, nma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::pwma::{pwma_batch_py, pwma_py, PwmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::pwma::{
    pwma_cuda_batch_dev_py, pwma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::reflex::{reflex_batch_py, reflex_py, ReflexStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::reflex::{
    reflex_cuda_batch_dev_py, reflex_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::sama::{sama_batch_py, sama_py, SamaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::sama::{
    sama_cuda_batch_dev_py, sama_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::sinwma::{sinwma_batch_py, sinwma_py, SinWmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::sinwma::{
    sinwma_cuda_batch_dev_py, sinwma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::sma::{sma_batch_py, sma_py, SmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::sma::{
    sma_cuda_batch_dev_py, sma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::smma::{smma_batch_py, smma_py, SmmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::smma::{
    smma_cuda_batch_dev_py, smma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::sqwma::{sqwma_batch_py, sqwma_py, SqwmaStreamPy};
#[cfg(feature = "cuda")]
use crate::indicators::moving_averages::sqwma::{
    sqwma_cuda_batch_dev_py, sqwma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::srwma::{srwma_batch_py, srwma_py, SrwmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::srwma::{
    srwma_cuda_batch_dev_py, srwma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::supersmoother::{
    supersmoother_batch_py, supersmoother_py, SuperSmootherStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::supersmoother::{
    supersmoother_cuda_batch_dev_py, supersmoother_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::supersmoother_3_pole::{
    supersmoother_3_pole_batch_py, supersmoother_3_pole_py, SuperSmoother3PoleStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::supersmoother_3_pole::{
    supersmoother_3_pole_cuda_batch_dev_py, supersmoother_3_pole_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::swma::{swma_batch_py, swma_py, SwmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::swma::{
    swma_cuda_batch_dev_py, swma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::tema::{tema_batch_py, tema_py, TemaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::tema::{
    tema_cuda_batch_dev_py, tema_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::tilson::{tilson_batch_py, tilson_py, TilsonStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::tilson::{
    tilson_cuda_batch_dev_py, tilson_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::tradjema::{
    tradjema_batch_py, tradjema_py, TradjemaStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::tradjema::{
    tradjema_cuda_batch_dev_py, tradjema_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::trendflex::{
    trendflex_batch_py, trendflex_py, TrendFlexStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::trima::{trima_batch_py, trima_py, TrimaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::trima::{
    trima_cuda_batch_dev_py, trima_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::uma::{uma_batch_py, uma_py, UmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::uma::{
    uma_cuda_batch_dev_py, uma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::volatility_adjusted_ma as vama_vol;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::volatility_adjusted_ma::{
    vama_cuda_batch_dev_py, vama_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::volume_adjusted_ma as vama_volu;

// -----------------------------
// Unified VAMA Python functions
// -----------------------------
#[cfg(feature = "python")]
#[pyfunction(name = "vama")]
#[pyo3(signature = (*args, **kwargs))]
fn vama_unified_py<'py>(
    py: Python<'py>,
    args: &'py Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&'py Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    // Decide variant: volume-adjusted if a second ndarray is present or 'volume' kw is provided
    let is_volume_variant = || -> PyResult<bool> {
        if args.len() >= 2 {
            if args.get_item(1)?.downcast::<PyArray1<f64>>().is_ok() {
                return Ok(true);
            }
        }
        if let Some(kw) = &kwargs {
            if let Ok(Some(_v)) = kw.get_item("volume") {
                if _v.downcast::<PyArray1<f64>>().is_ok() {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }()?;

    if is_volume_variant {
        // Expect: (data, volume, [length, vi_factor, strict, sample_period, kernel]) with kw overrides
        let data: PyReadonlyArray1<'_, f64> = args.get_item(0)?.extract()?;
        // 2nd arg may be positional or kw
        let volume: PyReadonlyArray1<'_, f64> = if args.len() >= 2 {
            args.get_item(1)?.extract()?
        } else {
            kwargs
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "vama: missing volume array",
                ))?
                .get_item("volume")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "vama: missing volume array",
                ))?
                .extract()?
        };

        // helpers to read either kw or positional with defaults
        let get_kw = |name: &str| -> Option<Bound<'_, pyo3::types::PyAny>> {
            kwargs.and_then(|k| k.get_item(name).ok().flatten())
        };
        let mut idx = 2usize; // after data, volume
        let length: usize = if let Some(v) = get_kw("length") {
            v.extract()?
        } else if args.len() > idx {
            let out: usize = args.get_item(idx)?.extract()?;
            idx += 1;
            out
        } else {
            13
        };
        let vi_factor: f64 = if let Some(v) = get_kw("vi_factor") {
            v.extract()?
        } else if args.len() > idx {
            let out: f64 = args.get_item(idx)?.extract()?;
            idx += 1;
            out
        } else {
            0.67
        };
        let strict: bool = if let Some(v) = get_kw("strict") {
            v.extract()?
        } else if args.len() > idx {
            let out: bool = args.get_item(idx)?.extract()?;
            idx += 1;
            out
        } else {
            true
        };
        let sample_period: usize = if let Some(v) = get_kw("sample_period") {
            v.extract()?
        } else if args.len() > idx {
            let out: usize = args.get_item(idx)?.extract()?;
            idx += 1;
            out
        } else {
            0
        };
        let kernel_s: Option<String> = get_kw("kernel").map(|v| v.extract()).transpose()?;
        let kernel = kernel_s.as_deref();

        let arr = vama_volu::volume_adjusted_ma_py(
            py,
            data,
            volume,
            length,
            vi_factor,
            strict,
            sample_period,
            kernel,
        )?;
        return Ok(arr.into_any());
    }

    // Volatility-adjusted variant: (data, [base_period, vol_period, smoothing, smooth_type, smooth_period, kernel])
    let data: PyReadonlyArray1<'_, f64> = args.get_item(0)?.extract()?;
    let get_kw = |name: &str| -> Option<Bound<'_, pyo3::types::PyAny>> {
        kwargs.and_then(|k| k.get_item(name).ok().flatten())
    };
    let mut idx = 1usize;
    let base_period: usize = if let Some(v) = get_kw("base_period") {
        v.extract()?
    } else if let Some(v) = get_kw("length") {
        // allow alias 'length' for base_period
        v.extract()?
    } else if args.len() > idx && args.get_item(idx)?.extract::<usize>().is_ok() {
        let out: usize = args.get_item(idx)?.extract()?;
        idx += 1;
        out
    } else {
        113
    };
    let vol_period: usize = if let Some(v) = get_kw("vol_period") {
        v.extract()?
    } else if args.len() > idx {
        let out: usize = args.get_item(idx)?.extract()?;
        idx += 1;
        out
    } else {
        51
    };
    let smoothing: bool = if let Some(v) = get_kw("smoothing") {
        v.extract()?
    } else if args.len() > idx {
        let out: bool = args.get_item(idx)?.extract()?;
        idx += 1;
        out
    } else {
        true
    };
    let smooth_type: usize = if let Some(v) = get_kw("smooth_type") {
        v.extract()?
    } else if args.len() > idx {
        let out: usize = args.get_item(idx)?.extract()?;
        idx += 1;
        out
    } else {
        3
    };
    let smooth_period: usize = if let Some(v) = get_kw("smooth_period") {
        v.extract()?
    } else if args.len() > idx {
        let out: usize = args.get_item(idx)?.extract()?;
        idx += 1;
        out
    } else {
        5
    };
    let kernel_s: Option<String> = get_kw("kernel").map(|v| v.extract()).transpose()?;
    let kernel = kernel_s.as_deref();
    let arr = vama_vol::vama_py(
        py, data, Some(base_period), vol_period, smoothing, smooth_type, smooth_period, kernel, None,
    )?;
    Ok(arr.into_any())
}

#[cfg(feature = "python")]
#[pyfunction(name = "vama_batch")]
#[pyo3(signature = (*args, **kwargs))]
fn vama_batch_unified_py<'py>(
    py: Python<'py>,
    args: &'py Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&'py Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let is_volume_variant = || -> PyResult<bool> {
        if args.len() >= 2 {
            if args.get_item(1)?.downcast::<PyArray1<f64>>().is_ok() {
                return Ok(true);
            }
        }
        if let Some(kw) = &kwargs {
            if kw.get_item("volume").ok().flatten().is_some() {
                return Ok(true);
            }
        }
        Ok(false)
    }()?;

    if is_volume_variant {
        // (data, volume, length_range=(...), vi_factor_range=(...), sample_period_range=(...), strict=None, kernel=None)
        let data: PyReadonlyArray1<'_, f64> = args.get_item(0)?.extract()?;
        let volume: PyReadonlyArray1<'_, f64> = if args.len() >= 2 {
            args.get_item(1)?.extract()?
        } else {
            kwargs
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "vama_batch: missing volume array",
                ))?
                .get_item("volume")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "vama_batch: missing volume array",
                ))?
                .extract()?
        };
        let get_kw = |name: &str| -> Option<Bound<'_, pyo3::types::PyAny>> {
            kwargs.and_then(|k| k.get_item(name).ok().flatten())
        };
        let length_range: (usize, usize, usize) = get_kw("length_range")
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or((13, 13, 0));
        let vi_factor_range: (f64, f64, f64) = get_kw("vi_factor_range")
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or((0.67, 0.67, 0.0));
        let sample_period_range: (usize, usize, usize) = get_kw("sample_period_range")
            .map(|v| v.extract())
            .transpose()?
            .unwrap_or((0, 0, 0));
        let strict: Option<bool> = get_kw("strict").map(|v| v.extract()).transpose()?;
        let kernel_s: Option<String> = get_kw("kernel").map(|v| v.extract()).transpose()?;
        let kernel = kernel_s.as_deref();
        return vama_volu::volume_adjusted_ma_batch_py(
            py,
            data,
            volume,
            length_range,
            vi_factor_range,
            sample_period_range,
            strict,
            kernel,
        );
    }

    // Volatility-adjusted batch: (data, base_period_range=(...), vol_period_range=(...), kernel=None)
    let data: PyReadonlyArray1<'_, f64> = args.get_item(0)?.extract()?;
    let get_kw = |name: &str| -> Option<Bound<'_, pyo3::types::PyAny>> {
        kwargs.and_then(|k| k.get_item(name).ok().flatten())
    };
    let base_period_range: (usize, usize, usize) = get_kw("base_period_range")
        .map(|v| v.extract())
        .transpose()?
        .or_else(|| get_kw("length_range").map(|v| v.extract()).transpose().ok().flatten())
        .unwrap_or((100, 130, 10));
    let vol_period_range: (usize, usize, usize) = get_kw("vol_period_range")
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or((40, 60, 10));
    let kernel_s: Option<String> = get_kw("kernel").map(|v| v.extract()).transpose()?;
    let kernel = kernel_s.as_deref();
    vama_vol::vama_batch_py(py, data, Some(base_period_range), vol_period_range, kernel, None)
}

// Unified stream with dual signatures
#[cfg(feature = "python")]
#[pyclass(name = "VamaStream")]
pub struct VamaStreamUnifiedPy {
    inner: VamaStreamKind,
}

#[cfg(feature = "python")]
enum VamaStreamKind {
    Volatility(vama_vol::VamaStream),
    Volume(vama_volu::VolumeAdjustedMaStream),
}

#[cfg(feature = "python")]
#[pymethods]
impl VamaStreamUnifiedPy {
    #[new]
    #[pyo3(signature = (length=None, vi_factor=None, strict=None, sample_period=None, base_period=None, vol_period=None, smoothing=None, smooth_type=None, smooth_period=None))]
    fn new(
        length: Option<usize>,
        vi_factor: Option<f64>,
        strict: Option<bool>,
        sample_period: Option<usize>,
        base_period: Option<usize>,
        vol_period: Option<usize>,
        smoothing: Option<bool>,
        smooth_type: Option<usize>,
        smooth_period: Option<usize>,
    ) -> PyResult<Self> {
        // Prefer volume-adjusted when any of its params are specified
        if length.is_some() || vi_factor.is_some() || strict.is_some() || sample_period.is_some() {
            let s = vama_volu::VolumeAdjustedMaStream::try_new(vama_volu::VolumeAdjustedMaParams {
                length: Some(length.unwrap_or(13)),
                vi_factor: Some(vi_factor.unwrap_or(0.67)),
                strict: Some(strict.unwrap_or(true)),
                sample_period: Some(sample_period.unwrap_or(0)),
            })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            return Ok(Self { inner: VamaStreamKind::Volume(s) });
        }
        // Otherwise assume volatility-adjusted
        let s = vama_vol::VamaStream::try_new(vama_vol::VamaParams {
            base_period: Some(base_period.unwrap_or(113)),
            vol_period: Some(vol_period.unwrap_or(51)),
            smoothing: Some(smoothing.unwrap_or(true)),
            smooth_type: Some(smooth_type.unwrap_or(3)),
            smooth_period: Some(smooth_period.unwrap_or(5)),
        })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: VamaStreamKind::Volatility(s) })
    }

    /// Update with one (price) or two (price, volume) arguments.
    #[pyo3(signature = (price, volume=None))]
    fn update(&mut self, price: f64, volume: Option<f64>) -> Option<f64> {
        match &mut self.inner {
            VamaStreamKind::Volatility(s) => s.update(price),
            VamaStreamKind::Volume(s) => s.update(price, volume.unwrap_or(f64::NAN)),
        }
    }
}
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::volume_adjusted_ma::{
    volume_adjusted_ma_cuda_batch_dev_py, volume_adjusted_ma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::vpwma::{vpwma_batch_py, vpwma_py, VpwmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::vpwma::{
    vpwma_cuda_batch_dev_py, vpwma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::vwap::{vwap_batch_py, vwap_py, VwapStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::vwap::{
    vwap_cuda_batch_dev_py, vwap_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::vwma::{vwma_batch_py, vwma_py, VwmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::vwma::{
    vwma_cuda_batch_dev_py, vwma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::wilders::{wilders_batch_py, wilders_py, WildersStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::wilders::{
    wilders_cuda_batch_dev_py, wilders_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::wma::{wma_batch_py, wma_py, WmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::wma::{
    wma_cuda_batch_dev_py, wma_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::moving_averages::zlema::{zlema_batch_py, zlema_py, ZlemaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::moving_averages::zlema::{
    zlema_cuda_batch_dev_py, zlema_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::msw::{msw_batch_py, msw_py, MswStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::msw::{msw_cuda_batch_dev_py, msw_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::nadaraya_watson_envelope::{
    nadaraya_watson_envelope_batch_py, nadaraya_watson_envelope_py, NweStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::natr::{natr_batch_py, natr_py, NatrStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::natr::{natr_cuda_batch_dev_py, natr_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::net_myrsi::{net_myrsi_batch_py, net_myrsi_py, NetMyrsiStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::net_myrsi::{
    net_myrsi_cuda_batch_dev_py, net_myrsi_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::nvi::{nvi_batch_py, nvi_py, NviStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::nvi::{nvi_cuda_batch_dev_py, nvi_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::obv::{obv_batch_py, obv_py, ObvStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::obv::{obv_cuda_batch_dev_py, obv_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::ott::{ott_batch_py, ott_py, OttStreamPy};
#[cfg(feature = "python")]
use crate::indicators::otto::{otto_batch_py, otto_py, OttoStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::otto::{otto_cuda_batch_dev_py, otto_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::percentile_nearest_rank::{
    percentile_nearest_rank_batch_py, percentile_nearest_rank_py, PercentileNearestRankStreamPy,
};
#[cfg(feature = "python")]
use crate::indicators::pfe::{pfe_batch_py, pfe_py, PfeStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::pfe::{pfe_cuda_batch_dev_py, pfe_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::pivot::{pivot_batch_py, pivot_py, PivotStreamPy};
#[cfg(feature = "python")]
use crate::indicators::pma::{pma_batch_py, pma_py, PmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::pma::{pma_cuda_batch_dev_py, pma_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::ppo::{ppo_batch_py, ppo_py, PpoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::prb::{prb_batch_py, prb_py, PrbStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::prb::{prb_cuda_batch_dev_py, prb_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::pvi::{pvi_batch_py, pvi_py, PviStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::pvi::{
    pvi_cuda_batch_dev_py, pvi_cuda_many_series_one_param_dev_py, PviDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::qqe::{qqe_batch_py, qqe_py, QqeStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::qqe::{qqe_cuda_batch_dev_py, qqe_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::qstick::{qstick_batch_py, qstick_py, QstickStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::qstick::{
    qstick_cuda_batch_dev_py, qstick_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::range_filter::{
    range_filter_batch_py, range_filter_py, RangeFilterStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::range_filter::{
    range_filter_cuda_batch_dev_py, range_filter_cuda_many_series_one_param_dev_py,
    RangeFilterDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::reverse_rsi::{reverse_rsi_batch_py, reverse_rsi_py, ReverseRsiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::roc::{roc_batch_py, roc_py, RocStreamPy};
#[cfg(feature = "python")]
use crate::indicators::rocp::{rocp_batch_py, rocp_py, RocpStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::rocp::{rocp_cuda_batch_dev_py, rocp_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::rocr::{rocr_batch_py, rocr_py, RocrStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::rocr::{
    rocr_cuda_batch_dev_py, rocr_cuda_many_series_one_param_dev_py, RocrDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::rsi::{rsi_batch_py, rsi_py, RsiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::rsmk::{rsmk_batch_py, rsmk_py, RsmkStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::rsmk::{rsmk_cuda_batch_dev_py, rsmk_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::rsx::{rsx_batch_py, rsx_py, RsxStreamPy};
#[cfg(feature = "python")]
use crate::indicators::rvi::{rvi_batch_py, rvi_py, RviStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::rvi::{rvi_cuda_batch_dev_py, rvi_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::safezonestop::{
    safezonestop_batch_py, safezonestop_py, SafeZoneStopStreamPy,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::safezonestop::{
    safezonestop_cuda_batch_dev_py, safezonestop_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::sar::{sar_batch_py, sar_py, SarStreamPy};
#[cfg(feature = "python")]
use crate::indicators::squeeze_momentum::{
    squeeze_momentum_batch_py, squeeze_momentum_py, SqueezeMomentumStreamPy,
};
#[cfg(feature = "cuda")]
use crate::indicators::squeeze_momentum::{
    squeeze_momentum_cuda_batch_dev_py, squeeze_momentum_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::srsi::{srsi_batch_py, srsi_py, SrsiStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::srsi::SrsiDeviceArrayF32Py;
#[cfg(feature = "python")]
use crate::indicators::stc::{stc_batch_py, stc_py, StcStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::stc::{stc_cuda_batch_dev_py, stc_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::stddev::{stddev_batch_py, stddev_py, StdDevStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::stddev::{
    stddev_cuda_batch_dev_py, stddev_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::stoch::{stoch_batch_py, stoch_py, StochStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::stoch::{
    stoch_cuda_batch_dev_py, stoch_cuda_many_series_one_param_dev_py, StochDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::stochf::{stochf_batch_py, stochf_py, StochfStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::stochf::{
    stochf_cuda_batch_dev_py, stochf_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::supertrend::{supertrend_batch_py, supertrend_py, SuperTrendStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::supertrend::{
    supertrend_cuda_batch_dev_py, supertrend_cuda_many_series_one_param_dev_py,
    SupertrendDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::trix::{trix_batch_py, trix_py, TrixStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::trix::{trix_cuda_batch_dev_py, trix_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::tsf::{tsf_batch_py, tsf_py, TsfStreamPy};
// CUDA TSF pyfunctions (explicit import so wrap_pyfunction! can resolve reliably)
#[cfg(feature = "python")]
use crate::indicators::tsi::{tsi_batch_py, tsi_py, TsiStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::tsi::{
    tsi_cuda_batch_dev_py, tsi_cuda_many_series_one_param_dev_py, TsiDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::ttm_squeeze::{ttm_squeeze_batch_py, ttm_squeeze_py, TtmSqueezeStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ttm_trend::{ttm_trend_batch_py, ttm_trend_py, TtmTrendStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ui::{ui_batch_py, ui_py, UiStreamPy};
#[cfg(feature = "python")]
use crate::indicators::ultosc::{ultosc_batch_py, ultosc_py, UltOscStreamPy};
#[cfg(feature = "python")]
use crate::indicators::var::{var_batch_py, var_py, VarStreamPy};
#[cfg(feature = "python")]
use crate::indicators::vi::{vi_batch_py, vi_py, ViStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::vi::{vi_cuda_batch_dev_py, vi_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::vidya::{vidya_batch_py, vidya_py, VidyaStreamPy};
// (duplicate removed)
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::ad::{ad_cuda_dev_py, ad_cuda_many_series_one_param_dev_py};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::medprice::{
    medprice_cuda_batch_dev_py, medprice_cuda_dev_py, medprice_cuda_many_series_one_param_dev_py,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::sar::{
    sar_cuda_batch_dev_py, sar_cuda_many_series_one_param_dev_py, SarDeviceArrayF32Py,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::vidya::{
    vidya_cuda_batch_dev_py, vidya_cuda_many_series_one_param_dev_py, VidyaDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::vlma::{vlma_batch_py, vlma_py, VlmaStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::vlma::{vlma_cuda_batch_dev_py, vlma_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::vosc::{vosc_batch_py, vosc_py, VoscStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::vosc::{vosc_cuda_batch_dev_py, vosc_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::voss::{voss_batch_py, voss_py, VossStreamPy};
#[cfg(feature = "python")]
use crate::indicators::vpci::{vpci_batch_py, vpci_py};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::vpci::{vpci_cuda_batch_dev_py, vpci_cuda_many_series_one_param_dev_py};
#[cfg(feature = "python")]
use crate::indicators::vpt::{vpt_batch_py, vpt_py, VptStreamPy};
#[cfg(feature = "python")]
use crate::indicators::vwmacd::{vwmacd_batch_py, vwmacd_py, VwmacdStreamPy};
#[cfg(feature = "python")]
use crate::indicators::wad::{wad_batch_py, wad_py, WadStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::wad::{
    wad_cuda_batch_dev_py, wad_cuda_dev_py, wad_cuda_many_series_one_param_dev_py,
};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::wavetrend::wavetrend_cuda_batch_dev_py;
#[cfg(feature = "python")]
use crate::indicators::wavetrend::{wavetrend_batch_py, wavetrend_py, WavetrendStreamPy};
#[cfg(feature = "python")]
use crate::indicators::wclprice::{wclprice_batch_py, wclprice_py, WclpriceStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::wclprice::{
    wclprice_cuda_batch_dev_py, wclprice_cuda_dev_py, wclprice_cuda_many_series_one_param_dev_py,
};
#[cfg(feature = "python")]
use crate::indicators::willr::{willr_batch_py, willr_py, WillrStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::willr::{
    willr_cuda_batch_dev_py, willr_cuda_many_series_one_param_dev_py, WillrDeviceArrayF32Py,
};
#[cfg(feature = "python")]
use crate::indicators::wto::{wto_batch_py, wto_py, WtoStreamPy};
#[cfg(feature = "python")]
use crate::indicators::zscore::{zscore_batch_py, zscore_py, ZscoreStreamPy};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::zscore::{
    zscore_cuda_batch_dev_py, zscore_cuda_many_series_one_param_dev_py,
};

// Local Python+CUDA helpers. Most indicator CUDA bindings are registered directly
// from their modules; a few shared utilities (like DeviceArrayF32Py) live in dlpack_cuda.
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::cuda_available;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::cuda::moving_averages::CudaTsf;
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::indicators::tsf::{TsfBatchRange, TsfParams};
#[cfg(all(feature = "python", feature = "cuda"))]
use crate::utilities::dlpack_cuda::{make_device_array_py, DeviceArrayF32Py};
#[cfg(all(feature = "python", feature = "cuda"))]
use numpy::PyReadonlyArray2;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::exceptions::PyValueError;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::prelude::*;
#[cfg(all(feature = "python", feature = "cuda"))]
use pyo3::types::PyDict;

// TSF CUDA pyfunction shims
#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "tsf_cuda_batch_dev")]
#[pyo3(signature = (data_f32, period_range, device_id=0))]
pub fn tsf_cuda_batch_dev_py_bindings<'py>(
    py: Python<'py>,
    data_f32: PyReadonlyArray1<'py, f32>,
    period_range: (usize, usize, usize),
    device_id: usize,
) -> PyResult<(DeviceArrayF32Py, Bound<'py, PyDict>)> {
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let slice_in = data_f32.as_slice()?;
    let sweep = TsfBatchRange { period: period_range };
    let (inner, combos) = py.allow_threads(|| {
        let cuda = CudaTsf::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.tsf_batch_dev(slice_in, &sweep)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    let dict = PyDict::new(py);
    let periods: Vec<u64> = combos.iter().map(|c| c.period.unwrap() as u64).collect();
    dict.set_item("periods", periods.into_pyarray(py))?;
    Ok((make_device_array_py(device_id, inner)?, dict))
}

#[cfg(all(feature = "python", feature = "cuda"))]
#[pyfunction(name = "tsf_cuda_many_series_one_param_dev")]
#[pyo3(signature = (data_tm_f32, period, device_id=0))]
pub fn tsf_cuda_many_series_one_param_dev_py_bindings(
    py: Python<'_>,
    data_tm_f32: PyReadonlyArray2<'_, f32>,
    period: usize,
    device_id: usize,
) -> PyResult<DeviceArrayF32Py> {
    use numpy::PyUntypedArrayMethods;
    if !cuda_available() {
        return Err(PyValueError::new_err("CUDA not available"));
    }
    let flat_in = data_tm_f32.as_slice()?;
    let rows = data_tm_f32.shape()[0];
    let cols = data_tm_f32.shape()[1];
    let expected = cols
        .checked_mul(rows)
        .ok_or_else(|| PyValueError::new_err("tsf_cuda_many_series_one_param_dev: rows*cols overflow"))?;
    if flat_in.len() != expected {
        return Err(PyValueError::new_err(
            "tsf_cuda_many_series_one_param_dev: time-major input length mismatch",
        ));
    }
    let params = TsfParams { period: Some(period) };
    let inner = py.allow_threads(|| {
        let cuda = CudaTsf::new(device_id).map_err(|e| PyValueError::new_err(e.to_string()))?;
        cuda.tsf_multi_series_one_param_time_major_dev(flat_in, cols, rows, &params)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })?;
    Ok(make_device_array_py(device_id, inner)?)
}

// Older ROC CUDA shim functions have been superseded by the
// indicator-local `roc_cuda_*` pyfunctions.
#[pymodule]
fn vector_ta(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register AD functions with their user-facing names
    m.add_function(wrap_pyfunction!(ad_py, m)?)?;
    m.add_function(wrap_pyfunction!(ad_batch_py, m)?)?;
    m.add_class::<AdStreamPy>()?;
#[cfg(feature = "cuda")]
{
        use crate::cuda::moving_averages::ma_selector::{
            ma_selector_cuda_sweep_to_device_py, ma_selector_cuda_to_device_py,
        };
        m.add_function(wrap_pyfunction!(ad_cuda_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(ad_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register ADX functions with their user-facing names
    m.add_function(wrap_pyfunction!(adx_py, m)?)?;
    m.add_function(wrap_pyfunction!(adx_batch_py, m)?)?;
    m.add_class::<AdxStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(adx_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(adx_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register DM CUDA (CPU registration appears later in this file)
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(dm_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(dm_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register ADOSC functions with their user-facing names
    m.add_function(wrap_pyfunction!(adosc_py, m)?)?;
    m.add_function(wrap_pyfunction!(adosc_batch_py, m)?)?;
    m.add_class::<AdoscStreamPy>()?;

    // Register ADXR functions with their user-facing names
    m.add_function(wrap_pyfunction!(adxr_py, m)?)?;
    m.add_function(wrap_pyfunction!(adxr_batch_py, m)?)?;
    m.add_class::<AdxrStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(adxr_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(adxr_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register ACOSC functions with their user-facing names
    m.add_function(wrap_pyfunction!(acosc_py, m)?)?;
    m.add_function(wrap_pyfunction!(acosc_batch_py, m)?)?;
    m.add_class::<AcoscStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(acosc_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            acosc_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        // Ensure ACOSC device array class is exported (CAI v3 + __dlpack_device__)
        use crate::indicators::acosc::AcoscDeviceArrayF32Py;
        m.add_class::<AcoscDeviceArrayF32Py>()?;
    }

    // Register APO functions with their user-facing names
    m.add_function(wrap_pyfunction!(apo_py, m)?)?;
    m.add_function(wrap_pyfunction!(apo_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(apo_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(apo_cuda_many_series_one_param_dev_py, m)?)?;
    }
    m.add_class::<ApoStreamPy>()?;

    // Register Band-Pass functions with their user-facing names
    m.add_function(wrap_pyfunction!(bandpass_py, m)?)?;
    m.add_function(wrap_pyfunction!(bandpass_batch_py, m)?)?;
    m.add_class::<BandPassStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(bandpass_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            bandpass_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Alligator functions with their user-facing names
    m.add_function(wrap_pyfunction!(alligator_py, m)?)?;
    m.add_function(wrap_pyfunction!(alligator_batch_py, m)?)?;
    m.add_class::<AlligatorStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(alligator_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            alligator_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register ALMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(alma_py, m)?)?;
    m.add_function(wrap_pyfunction!(alma_batch_py, m)?)?;
    m.add_class::<AlmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::moving_averages::supersmoother::{
            supersmoother_cuda_batch_dev_py, supersmoother_cuda_many_series_one_param_dev_py,
        };
        use crate::indicators::moving_averages::trendflex::{
            trendflex_cuda_batch_dev_py, trendflex_cuda_many_series_one_param_dev_py,
        };
        use crate::cuda::moving_averages::ma_selector::{
            ma_selector_cuda_sweep_to_device_py, ma_selector_cuda_to_device_py,
        };
        m.add_function(wrap_pyfunction!(alma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(alma_cuda_many_series_one_param_dev_py, m)?)?;
        // MA selector (price-only dispatcher) helpers
        m.add_function(wrap_pyfunction!(ma_selector_cuda_to_device_py, m)?)?;
        m.add_function(wrap_pyfunction!(ma_selector_cuda_sweep_to_device_py, m)?)?;
        m.add_function(wrap_pyfunction!(linreg_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            linreg_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(sma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(sma_cuda_many_series_one_param_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(nma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(nma_cuda_many_series_one_param_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(frama_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            frama_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(hma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(hma_cuda_many_series_one_param_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(wad_cuda_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(wad_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(wad_cuda_many_series_one_param_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(zscore_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            zscore_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(zlema_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            zlema_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        // MEDIUM_AD CUDA
        m.add_function(wrap_pyfunction!(medium_ad_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            medium_ad_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_class::<MediumAdDeviceArrayF32Py>()?;
        // ADOSC CUDA
        use crate::indicators::adosc::{
            adosc_cuda_batch_dev_py, adosc_cuda_many_series_one_param_dev_py,
            DeviceArrayF32AdoscPy,
        };
        // Export the device handle class so consumers can isinstance-check or import it
        m.add_class::<DeviceArrayF32AdoscPy>()?;
        m.add_function(wrap_pyfunction!(adosc_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            adosc_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(trendflex_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            trendflex_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(vpwma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            vpwma_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(supersmoother_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            supersmoother_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register AlphaTrend CUDA functions
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(alphatrend_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            alphatrend_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register AroonOsc functions with their user-facing names
    m.add_function(wrap_pyfunction!(aroon_osc_py, m)?)?;
    m.add_function(wrap_pyfunction!(aroon_osc_batch_py, m)?)?;
    m.add_class::<AroonOscStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(aroonosc_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            aroonosc_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        // Export AroonOsc device array class (CAI v3 + DLPack)
        use crate::indicators::aroonosc::AroonOscDeviceArrayF32Py;
        m.add_class::<AroonOscDeviceArrayF32Py>()?;
    }

    // Register Bollinger Bands functions with their user-facing names
    m.add_function(wrap_pyfunction!(bollinger_bands_py, m)?)?;
    m.add_function(wrap_pyfunction!(bollinger_bands_batch_py, m)?)?;
    m.add_class::<BollingerBandsStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::bollinger_bands::BollingerDeviceArrayF32Py;
        m.add_function(wrap_pyfunction!(
            crate::indicators::bollinger_bands::bollinger_bands_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::bollinger_bands::bollinger_bands_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_class::<BollingerDeviceArrayF32Py>()?;
    }

    // Register CWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(cwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(cwma_batch_py, m)?)?;
    m.add_class::<CwmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(cwma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(cwma_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register DEMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(dema_py, m)?)?;
    m.add_function(wrap_pyfunction!(dema_batch_py, m)?)?;
    m.add_class::<DemaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(dema_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(dema_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register EDCF functions with their user-facing names
    m.add_function(wrap_pyfunction!(edcf_py, m)?)?;
    m.add_function(wrap_pyfunction!(edcf_batch_py, m)?)?;
    m.add_class::<EdcfStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(edcf_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(edcf_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register EMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(ema_py, m)?)?;
    m.add_function(wrap_pyfunction!(ema_batch_py, m)?)?;
    m.add_class::<EmaStreamPy>()?;

    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(ema_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(ema_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Ehlers ITrend functions with their user-facing names
    m.add_function(wrap_pyfunction!(ehlers_itrend_py, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers_itrend_batch_py, m)?)?;
    m.add_class::<EhlersITrendStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(ehlers_itrend_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            ehlers_itrend_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register EPMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(epma_py, m)?)?;
    m.add_function(wrap_pyfunction!(epma_batch_py, m)?)?;
    m.add_class::<EpmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(epma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(epma_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register FRAMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(frama_py, m)?)?;
    m.add_function(wrap_pyfunction!(frama_batch_py, m)?)?;
    m.add_class::<FramaStreamPy>()?;

    // Register FWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(fwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(fwma_batch_py, m)?)?;
    m.add_class::<FwmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(fwma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(fwma_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Gaussian functions with their user-facing names
    m.add_function(wrap_pyfunction!(gaussian_py, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_batch_py, m)?)?;
    m.add_class::<GaussianStreamPy>()?;

    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(gaussian_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            gaussian_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register HighPass2 functions with their user-facing names
    m.add_function(wrap_pyfunction!(highpass_2_pole_py, m)?)?;
    m.add_function(wrap_pyfunction!(highpass_2_pole_batch_py, m)?)?;
    m.add_class::<HighPass2StreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(highpass_2_pole_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            highpass_2_pole_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register HighPass functions with their user-facing names
    m.add_function(wrap_pyfunction!(highpass_py, m)?)?;
    m.add_function(wrap_pyfunction!(highpass_batch_py, m)?)?;
    m.add_class::<HighPassStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(highpass_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            highpass_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register HMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(hma_py, m)?)?;
    m.add_function(wrap_pyfunction!(hma_batch_py, m)?)?;
    m.add_class::<HmaStreamPy>()?;

    // Register HWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(hwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(hwma_batch_py, m)?)?;
    m.add_class::<HwmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(hwma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(hwma_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register JMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(jma_py, m)?)?;
    m.add_function(wrap_pyfunction!(jma_batch_py, m)?)?;
    m.add_class::<JmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(jma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(jma_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register JSA functions with their user-facing names
    m.add_function(wrap_pyfunction!(jsa_py, m)?)?;
    m.add_function(wrap_pyfunction!(jsa_batch_py, m)?)?;
    m.add_class::<JsaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(jsa_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(jsa_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register KAMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(kama_py, m)?)?;
    m.add_function(wrap_pyfunction!(kama_batch_py, m)?)?;
    m.add_class::<KamaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::moving_averages::kama::KamaDeviceArrayF32Py;
        m.add_class::<KamaDeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(kama_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(kama_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Ehlers KAMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(ehlers_kama_py, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers_kama_batch_py, m)?)?;
    m.add_class::<EhlersKamaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(ehlers_kama_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            ehlers_kama_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register LinReg functions with their user-facing names
    m.add_function(wrap_pyfunction!(linreg_py, m)?)?;
    m.add_function(wrap_pyfunction!(linreg_batch_py, m)?)?;
    m.add_class::<LinRegStreamPy>()?;

    // Register LinearRegSlope functions with their user-facing names
    m.add_function(wrap_pyfunction!(linearreg_slope_py, m)?)?;
    m.add_function(wrap_pyfunction!(linearreg_slope_batch_py, m)?)?;
    m.add_class::<LinearRegSlopeStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(linearreg_slope_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            linearreg_slope_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register MediumAd functions with their user-facing names
    m.add_function(wrap_pyfunction!(medium_ad_py, m)?)?;
    m.add_function(wrap_pyfunction!(medium_ad_batch_py, m)?)?;
    m.add_class::<MediumAdStreamPy>()?;

    // Register MinMax functions with their user-facing names
    m.add_function(wrap_pyfunction!(minmax_py, m)?)?;
    m.add_function(wrap_pyfunction!(minmax_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(minmax_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            minmax_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_class::<MinmaxDeviceArrayF32Py>()?;
    }
    m.add_class::<MinmaxStreamPy>()?;

    // Register MAAQ functions with their user-facing names
    m.add_function(wrap_pyfunction!(maaq_py, m)?)?;
    m.add_function(wrap_pyfunction!(maaq_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(maaq_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(maaq_cuda_many_series_one_param_dev_py, m)?)?;
    }
    m.add_class::<MaaqStreamPy>()?;

    // Register MAMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(mama_py, m)?)?;
    m.add_function(wrap_pyfunction!(mama_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(mama_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(mama_cuda_many_series_one_param_dev_py, m)?)?;
    }
    m.add_class::<MamaStreamPy>()?;

    // Register MWDX functions with their user-facing names
    m.add_function(wrap_pyfunction!(mwdx_py, m)?)?;
    m.add_function(wrap_pyfunction!(mwdx_batch_py, m)?)?;
    m.add_class::<MwdxStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(mwdx_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(mwdx_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register NMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(nma_py, m)?)?;
    m.add_function(wrap_pyfunction!(nma_batch_py, m)?)?;
    m.add_class::<NmaStreamPy>()?;

    // Register NVI functions with their user-facing names
    m.add_function(wrap_pyfunction!(nvi_py, m)?)?;
    m.add_function(wrap_pyfunction!(nvi_batch_py, m)?)?;
    m.add_class::<NviStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(nvi_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(nvi_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register PVI functions with their user-facing names
    m.add_function(wrap_pyfunction!(pvi_py, m)?)?;
    m.add_function(wrap_pyfunction!(pvi_batch_py, m)?)?;
    m.add_class::<PviStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(pvi_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(pvi_cuda_many_series_one_param_dev_py, m)?)?;
        m.add_class::<PviDeviceArrayF32Py>()?;
    }

    // Register RSMK functions with their user-facing names
    m.add_function(wrap_pyfunction!(rsmk_py, m)?)?;
    m.add_function(wrap_pyfunction!(rsmk_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(rsmk_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(rsmk_cuda_many_series_one_param_dev_py, m)?)?;
    }
    m.add_class::<RsmkStreamPy>()?;

    // Register SRSI functions with their user-facing names
    m.add_function(wrap_pyfunction!(srsi_py, m)?)?;
    m.add_function(wrap_pyfunction!(srsi_batch_py, m)?)?;
    m.add_class::<SrsiStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::srsi::srsi_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::srsi::srsi_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_class::<SrsiDeviceArrayF32Py>()?;
    }

    // Register TSF functions with their user-facing names
    m.add_function(wrap_pyfunction!(tsf_py, m)?)?;
    m.add_function(wrap_pyfunction!(tsf_batch_py, m)?)?;
    m.add_class::<TsfStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(tsf_cuda_batch_dev_py_bindings, m)?)?;
        m.add_function(wrap_pyfunction!(
            tsf_cuda_many_series_one_param_dev_py_bindings,
            m
        )?)?;
    }

    // Register VI functions with their user-facing names
    m.add_function(wrap_pyfunction!(vi_py, m)?)?;
    m.add_function(wrap_pyfunction!(vi_batch_py, m)?)?;
    m.add_class::<ViStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(vi_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(vi_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register VPT functions with their user-facing names
    m.add_function(wrap_pyfunction!(vpt_py, m)?)?;
    m.add_function(wrap_pyfunction!(vpt_batch_py, m)?)?;
    m.add_class::<VptStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::vpt::{
            vpt_cuda_batch_dev_py, vpt_cuda_many_series_one_param_dev_py,
        };
        m.add_function(wrap_pyfunction!(vpt_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(vpt_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register PWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(pwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(pwma_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(pwma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(pwma_cuda_many_series_one_param_dev_py, m)?)?;
    }
    m.add_class::<PwmaStreamPy>()?;

    // Register PFE functions with their user-facing names
    m.add_function(wrap_pyfunction!(pfe_py, m)?)?;
    m.add_function(wrap_pyfunction!(pfe_batch_py, m)?)?;
    m.add_class::<PfeStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(pfe_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(pfe_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register PMA CUDA functions (non-lag PMA variant)
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(pma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(pma_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register ROC functions with their user-facing names
    m.add_function(wrap_pyfunction!(roc_py, m)?)?;
    m.add_function(wrap_pyfunction!(roc_batch_py, m)?)?;
    m.add_class::<RocStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::roc::roc_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::roc::roc_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register RVI functions with their user-facing names
    m.add_function(wrap_pyfunction!(rvi_py, m)?)?;
    m.add_function(wrap_pyfunction!(rvi_batch_py, m)?)?;
    m.add_class::<RviStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(rvi_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(rvi_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Reflex functions with their user-facing names
    m.add_function(wrap_pyfunction!(reflex_py, m)?)?;
    m.add_function(wrap_pyfunction!(reflex_batch_py, m)?)?;
    m.add_class::<ReflexStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(reflex_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            reflex_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register SINWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(sinwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(sinwma_batch_py, m)?)?;
    m.add_class::<SinWmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(sinwma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            sinwma_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register SMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(sma_py, m)?)?;
    m.add_function(wrap_pyfunction!(sma_batch_py, m)?)?;
    m.add_class::<SmaStreamPy>()?;

    // Register SMMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(smma_py, m)?)?;
    m.add_function(wrap_pyfunction!(smma_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(smma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(smma_cuda_many_series_one_param_dev_py, m)?)?;
    }
    m.add_class::<SmmaStreamPy>()?;

    // Register SQWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(sqwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(sqwma_batch_py, m)?)?;
    m.add_class::<SqwmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(sqwma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            sqwma_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register SRWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(srwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(srwma_batch_py, m)?)?;
    m.add_class::<SrwmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(srwma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            srwma_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register StdDev functions with their user-facing names
    m.add_function(wrap_pyfunction!(stddev_py, m)?)?;
    m.add_function(wrap_pyfunction!(stddev_batch_py, m)?)?;
    m.add_class::<StdDevStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(stddev_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            stddev_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register SuperSmoother3Pole functions with their user-facing names
    m.add_function(wrap_pyfunction!(supersmoother_3_pole_py, m)?)?;
    m.add_function(wrap_pyfunction!(supersmoother_3_pole_batch_py, m)?)?;
    m.add_class::<SuperSmoother3PoleStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(supersmoother_3_pole_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            supersmoother_3_pole_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register SuperSmoother functions with their user-facing names
    m.add_function(wrap_pyfunction!(supersmoother_py, m)?)?;
    m.add_function(wrap_pyfunction!(supersmoother_batch_py, m)?)?;
    m.add_class::<SuperSmootherStreamPy>()?;

    // Register SWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(swma_py, m)?)?;
    m.add_function(wrap_pyfunction!(swma_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(swma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(swma_cuda_many_series_one_param_dev_py, m)?)?;
    }
    m.add_class::<SwmaStreamPy>()?;

    // Register TEMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(tema_py, m)?)?;
    m.add_function(wrap_pyfunction!(tema_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(tema_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(tema_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register TRIMA functions
    m.add_function(wrap_pyfunction!(trima_py, m)?)?;
    m.add_function(wrap_pyfunction!(trima_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(trima_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            trima_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }
    m.add_class::<TrimaStreamPy>()?;
    m.add_class::<TemaStreamPy>()?;

    // Register Tilson functions with their user-facing names
    m.add_function(wrap_pyfunction!(tilson_py, m)?)?;
    m.add_function(wrap_pyfunction!(tilson_batch_py, m)?)?;
    m.add_class::<TilsonStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(tilson_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            tilson_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register TrendFlex functions with their user-facing names
    m.add_function(wrap_pyfunction!(trendflex_py, m)?)?;
    m.add_function(wrap_pyfunction!(trendflex_batch_py, m)?)?;
    m.add_class::<TrendFlexStreamPy>()?;

    // Register TTM Trend functions with their user-facing names
    m.add_function(wrap_pyfunction!(ttm_trend_py, m)?)?;
    m.add_function(wrap_pyfunction!(ttm_trend_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::ttm_trend::{
            ttm_trend_cuda_batch_dev_py, ttm_trend_cuda_many_series_one_param_dev_py,
            TtmTrendDeviceArrayF32Py,
        };
        m.add_class::<TtmTrendDeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(ttm_trend_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            ttm_trend_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }
    m.add_class::<TtmTrendStreamPy>()?;

    // Register VLMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(vlma_py, m)?)?;
    m.add_function(wrap_pyfunction!(vlma_batch_py, m)?)?;
    m.add_class::<VlmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(vlma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(vlma_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Wilders functions with their user-facing names
    m.add_function(wrap_pyfunction!(wilders_py, m)?)?;
    m.add_function(wrap_pyfunction!(wilders_batch_py, m)?)?;
    m.add_class::<WildersStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(wilders_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            wilders_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register VWMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(vwma_py, m)?)?;
    m.add_function(wrap_pyfunction!(vwma_batch_py, m)?)?;
    m.add_class::<VwmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(vwma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(vwma_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register VWMACD functions with their user-facing names
    m.add_function(wrap_pyfunction!(vwmacd_py, m)?)?;
    m.add_function(wrap_pyfunction!(vwmacd_batch_py, m)?)?;
    m.add_class::<VwmacdStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        use crate::indicators::vwmacd::{
            vwmacd_cuda_batch_dev_py, vwmacd_cuda_many_series_one_param_dev_py,
        };
        m.add_function(wrap_pyfunction!(vwmacd_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            vwmacd_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register VWAP functions with their user-facing names
    m.add_function(wrap_pyfunction!(vwap_py, m)?)?;
    m.add_function(wrap_pyfunction!(vwap_batch_py, m)?)?;
    m.add_class::<VwapStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(vwap_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(vwap_cuda_many_series_one_param_dev_py, m)?)?;
    }

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
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(wma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(wma_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register MA dispatcher function
    m.add_function(wrap_pyfunction!(ma_py, m)?)?;

    // Register CoRa Wave functions with their user-facing names
    m.add_function(wrap_pyfunction!(cora_wave_py, m)?)?;
    m.add_function(wrap_pyfunction!(cora_wave_batch_py, m)?)?;
    m.add_class::<CoraWaveStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(cora_wave_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            cora_wave_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Ehlers PMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(ehlers_pma_py, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers_pma_flat_py, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers_pma_batch_py, m)?)?;
    m.add_class::<EhlersPmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(ehlers_pma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            ehlers_pma_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Chandelier Exit functions with their user-facing names
    m.add_function(wrap_pyfunction!(chandelier_exit_py, m)?)?;
    m.add_function(wrap_pyfunction!(chandelier_exit_batch_py, m)?)?;
    m.add_class::<ChandelierExitStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::chandelier_exit::chandelier_exit_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::chandelier_exit::chandelier_exit_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Percentile Nearest Rank functions with their user-facing names
    m.add_function(wrap_pyfunction!(percentile_nearest_rank_py, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_nearest_rank_batch_py, m)?)?;
    m.add_class::<PercentileNearestRankStreamPy>()?;

    // Register UMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(uma_py, m)?)?;
    m.add_function(wrap_pyfunction!(uma_batch_py, m)?)?;
    m.add_class::<UmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(uma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(uma_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register OTTO functions with their user-facing names
    m.add_function(wrap_pyfunction!(otto_py, m)?)?;
    m.add_function(wrap_pyfunction!(otto_batch_py, m)?)?;
    m.add_class::<OttoStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::otto::otto_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::otto::otto_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Aroon functions with their user-facing names
    m.add_function(wrap_pyfunction!(aroon_py, m)?)?;
    m.add_function(wrap_pyfunction!(aroon_batch_py, m)?)?;
    m.add_class::<AroonStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::aroon::{
            aroon_cuda_batch_dev_py, aroon_cuda_many_series_one_param_dev_py,
        };
        m.add_function(wrap_pyfunction!(aroon_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            aroon_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register TRADJEMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(tradjema_py, m)?)?;
    m.add_function(wrap_pyfunction!(tradjema_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(tradjema_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            tradjema_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }
    m.add_class::<TradjemaStreamPy>()?;

    // Register ASO functions with their user-facing names
    m.add_function(wrap_pyfunction!(aso_py, m)?)?;
    m.add_function(wrap_pyfunction!(aso_batch_py, m)?)?;
    m.add_class::<AsoStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(aso_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(aso_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register MAC-Z functions with their user-facing names
    m.add_function(wrap_pyfunction!(macz_py, m)?)?;
    m.add_function(wrap_pyfunction!(macz_batch_py, m)?)?;
    m.add_class::<MaczStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::macz::macz_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::macz::macz_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register OTT functions with their user-facing names
    m.add_function(wrap_pyfunction!(ott_py, m)?)?;
    m.add_function(wrap_pyfunction!(ott_batch_py, m)?)?;
    m.add_class::<OttStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::ott::ott_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::ott::ott_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register DVDIQQE function
    m.add_function(wrap_pyfunction!(dvdiqqe_py, m)?)?;
    m.add_function(wrap_pyfunction!(dvdiqqe_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::dvdiqqe::dvdiqqe_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::dvdiqqe::dvdiqqe_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register PRB functions
    m.add_function(wrap_pyfunction!(prb_py, m)?)?;
    m.add_function(wrap_pyfunction!(prb_batch_py, m)?)?;
    m.add_class::<PrbStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(prb_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(prb_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register LPC functions
    m.add_function(wrap_pyfunction!(lpc_py, m)?)?;
    m.add_function(wrap_pyfunction!(lpc_batch_py, m)?)?;
    m.add_class::<LpcStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(lpc_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(lpc_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Bollinger Bands Width functions with their user-facing names
    m.add_function(wrap_pyfunction!(bollinger_bands_width_py, m)?)?;
    m.add_function(wrap_pyfunction!(bollinger_bands_width_batch_py, m)?)?;
    m.add_class::<BollingerBandsWidthStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(
            bollinger_bands_width_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            bollinger_bands_width_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register CG functions with their user-facing names
    m.add_function(wrap_pyfunction!(cg_py, m)?)?;
    m.add_function(wrap_pyfunction!(cg_batch_py, m)?)?;
    m.add_class::<CgStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(cg_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(cg_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Coppock functions with their user-facing names
    m.add_function(wrap_pyfunction!(coppock_py, m)?)?;
    m.add_function(wrap_pyfunction!(coppock_batch_py, m)?)?;
    m.add_class::<CoppockStreamPy>()?;

    // Register CMO functions with their user-facing names
    m.add_function(wrap_pyfunction!(cmo_py, m)?)?;
    m.add_function(wrap_pyfunction!(cmo_batch_py, m)?)?;
    m.add_class::<CmoStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(cmo_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(cmo_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register CKSP functions with their user-facing names
    m.add_function(wrap_pyfunction!(cksp_py, m)?)?;
    m.add_function(wrap_pyfunction!(cksp_batch_py, m)?)?;
    m.add_class::<CkspStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(cksp_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(cksp_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register CHOP functions with their user-facing names
    m.add_function(wrap_pyfunction!(chop_py, m)?)?;
    m.add_function(wrap_pyfunction!(chop_batch_py, m)?)?;
    m.add_class::<ChopStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        use crate::indicators::chop::{
            chop_cuda_batch_dev_py, chop_cuda_many_series_one_param_dev_py,
        };
        m.add_function(wrap_pyfunction!(chop_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(chop_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Correlation Cycle functions with their user-facing names
    m.add_function(wrap_pyfunction!(correlation_cycle_py, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_cycle_batch_py, m)?)?;
    m.add_class::<CorrelationCycleStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(correlation_cycle_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            correlation_cycle_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Correl HL functions with their user-facing names
    m.add_function(wrap_pyfunction!(correl_hl_py, m)?)?;
    m.add_function(wrap_pyfunction!(correl_hl_batch_py, m)?)?;
    m.add_class::<CorrelHlStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_class::<CorrelHlDeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(correl_hl_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            correl_hl_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Deviation functions with their user-facing names
    m.add_function(wrap_pyfunction!(deviation_py, m)?)?;
    m.add_function(wrap_pyfunction!(deviation_batch_py, m)?)?;
    m.add_class::<DeviationStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(deviation_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            deviation_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register DevStop functions with their user-facing names
    m.add_function(wrap_pyfunction!(devstop_py, m)?)?;
    m.add_function(wrap_pyfunction!(devstop_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(devstop_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            devstop_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register DTI functions with their user-facing names
    m.add_function(wrap_pyfunction!(dti_py, m)?)?;
    m.add_function(wrap_pyfunction!(dti_batch_py, m)?)?;
    m.add_class::<DtiStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(dti_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(dti_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register ERI functions with their user-facing names
    m.add_function(wrap_pyfunction!(eri_py, m)?)?;
    m.add_function(wrap_pyfunction!(eri_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(eri_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(eri_cuda_many_series_one_param_dev_py, m)?)?;
    }
    m.add_class::<EriStreamPy>()?;

    // Register KDJ functions with their user-facing names
    m.add_function(wrap_pyfunction!(kdj_py, m)?)?;
    m.add_function(wrap_pyfunction!(kdj_batch_py, m)?)?;
    m.add_class::<KdjStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(kdj_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(kdj_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Decycler functions with their user-facing names
    m.add_function(wrap_pyfunction!(decycler_py, m)?)?;
    m.add_function(wrap_pyfunction!(decycler_batch_py, m)?)?;
    m.add_class::<DecyclerStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(decycler_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            decycler_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register DevStop functions with their user-facing names
    m.add_function(wrap_pyfunction!(devstop_py, m)?)?;
    m.add_function(wrap_pyfunction!(devstop_batch_py, m)?)?;

    // Register DPO functions with their user-facing names
    m.add_function(wrap_pyfunction!(dpo_py, m)?)?;
    m.add_function(wrap_pyfunction!(dpo_batch_py, m)?)?;
    m.add_class::<DpoStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(dpo_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(dpo_cuda_many_series_one_param_dev_py, m)?)?;
        m.add_class::<DpoDeviceArrayF32Py>()?;
    }

    // Register ER functions with their user-facing names
    m.add_function(wrap_pyfunction!(er_py, m)?)?;
    m.add_function(wrap_pyfunction!(er_batch_py, m)?)?;
    m.add_class::<ErStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::er::{
            er_cuda_batch_dev_py,
            er_cuda_many_series_one_param_dev_py,
            DeviceArrayF32ErPy,
        };
        m.add_class::<DeviceArrayF32ErPy>()?;
        m.add_function(wrap_pyfunction!(er_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(er_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Kaufmanstop functions with their user-facing names
    m.add_function(wrap_pyfunction!(kaufmanstop_py, m)?)?;
    m.add_function(wrap_pyfunction!(kaufmanstop_batch_py, m)?)?;
    m.add_class::<KaufmanstopStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        use crate::indicators::kaufmanstop::{
            kaufmanstop_cuda_batch_dev_py, kaufmanstop_cuda_many_series_one_param_dev_py,
        };
        m.add_function(wrap_pyfunction!(kaufmanstop_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            kaufmanstop_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Linear Regression Angle functions with their user-facing names
    m.add_function(wrap_pyfunction!(linearreg_angle_py, m)?)?;
    m.add_function(wrap_pyfunction!(linearreg_angle_batch_py, m)?)?;
    m.add_class::<Linearreg_angleStreamPy>()?;

    // Register MarketEFI functions with their user-facing names
    m.add_function(wrap_pyfunction!(marketefi_py, m)?)?;
    m.add_function(wrap_pyfunction!(marketefi_batch_py, m)?)?;
    m.add_class::<MarketefiStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        use crate::indicators::marketefi::MarketefiDeviceArrayF32Py;
        m.add_function(wrap_pyfunction!(marketefi_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            marketefi_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_class::<MarketefiDeviceArrayF32Py>()?;
    }

    // Register Midpoint functions with their user-facing names
    m.add_function(wrap_pyfunction!(midpoint_py, m)?)?;
    m.add_function(wrap_pyfunction!(midpoint_batch_py, m)?)?;
    m.add_class::<MidpointStreamPy>()?;

    // Register Decycler Oscillator functions with their user-facing names
    m.add_function(wrap_pyfunction!(dec_osc_py, m)?)?;
    m.add_function(wrap_pyfunction!(dec_osc_batch_py, m)?)?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        use crate::indicators::dec_osc::{
            dec_osc_cuda_batch_dev_py, dec_osc_cuda_many_series_one_param_dev_py,
        };
        m.add_function(wrap_pyfunction!(dec_osc_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            dec_osc_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }
    m.add_class::<DecOscStreamPy>()?;

    // Register Donchian Channel functions with their user-facing names
    m.add_function(wrap_pyfunction!(donchian_py, m)?)?;
    m.add_function(wrap_pyfunction!(donchian_batch_py, m)?)?;
    m.add_class::<DonchianStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(donchian_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            donchian_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register EMV functions with their user-facing names
    m.add_function(wrap_pyfunction!(emv_py, m)?)?;
    m.add_function(wrap_pyfunction!(emv_batch_py, m)?)?;
    m.add_class::<EmvStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(emv_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(emv_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register IFT RSI functions with their user-facing names
    m.add_function(wrap_pyfunction!(ift_rsi_py, m)?)?;
    m.add_function(wrap_pyfunction!(ift_rsi_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(ift_rsi_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            ift_rsi_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }
    m.add_class::<IftRsiStreamPy>()?;

    // Register KVO functions with their user-facing names
    m.add_function(wrap_pyfunction!(kvo_py, m)?)?;
    m.add_function(wrap_pyfunction!(kvo_batch_py, m)?)?;
    m.add_class::<KvoStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        use crate::indicators::kvo::{
            kvo_cuda_batch_dev_py, kvo_cuda_many_series_one_param_dev_py,
        };
        m.add_function(wrap_pyfunction!(kvo_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(kvo_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register MACD functions with their user-facing names
    m.add_function(wrap_pyfunction!(macd_py, m)?)?;
    m.add_function(wrap_pyfunction!(macd_batch_py, m)?)?;
    m.add_class::<MacdStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        use crate::indicators::macd::{
            macd_cuda_batch_dev_py, macd_cuda_many_series_one_param_dev_py,
            DeviceArrayF32MacdPy,
        };
        m.add_class::<DeviceArrayF32MacdPy>()?;
        m.add_function(wrap_pyfunction!(macd_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(macd_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register MFI functions with their user-facing names
    m.add_function(wrap_pyfunction!(mfi_py, m)?)?;
    m.add_function(wrap_pyfunction!(mfi_batch_py, m)?)?;
    m.add_class::<MfiStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(mfi_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(mfi_cuda_many_series_one_param_dev_py, m)?)?;
        m.add_class::<MfiDeviceArrayF32Py>()?;
    }

    // Register NATR functions with their user-facing names
    m.add_function(wrap_pyfunction!(natr_py, m)?)?;
    m.add_function(wrap_pyfunction!(natr_batch_py, m)?)?;
    m.add_class::<NatrStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(natr_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(natr_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register PPO functions with their user-facing names
    m.add_function(wrap_pyfunction!(ppo_py, m)?)?;
    m.add_function(wrap_pyfunction!(ppo_batch_py, m)?)?;
    m.add_class::<PpoStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        use crate::indicators::ppo::{
            ppo_cuda_batch_dev_py, ppo_cuda_many_series_one_param_dev_py,
        };
        m.add_function(wrap_pyfunction!(ppo_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(ppo_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register RSI functions with their user-facing names
    m.add_function(wrap_pyfunction!(rsi_py, m)?)?;
    m.add_function(wrap_pyfunction!(rsi_batch_py, m)?)?;
    m.add_class::<RsiStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::rsi::{
            rsi_cuda_batch_dev_py, rsi_cuda_many_series_one_param_dev_py,
        };
        // DeviceArrayF32Py is already exported in other CUDA sections (e.g., RSX/ALMA)
        m.add_function(wrap_pyfunction!(rsi_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(rsi_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register RSX functions with their user-facing names
    m.add_function(wrap_pyfunction!(rsx_py, m)?)?;
    m.add_function(wrap_pyfunction!(rsx_batch_py, m)?)?;
    m.add_class::<RsxStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::rsx::{
            rsx_cuda_batch_dev_py, rsx_cuda_many_series_one_param_dev_py, RsxDeviceArrayF32Py,
        };
        m.add_class::<RsxDeviceArrayF32Py>()?;
        #[cfg(all(feature = "python", feature = "cuda"))]
        {
            use crate::indicators::moving_averages::cwma::DeviceArrayF32CwmaPy;
            m.add_class::<DeviceArrayF32CwmaPy>()?;
        }
        m.add_function(wrap_pyfunction!(rsx_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(rsx_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Squeeze Momentum functions with their user-facing names
    m.add_function(wrap_pyfunction!(squeeze_momentum_py, m)?)?;
    m.add_function(wrap_pyfunction!(squeeze_momentum_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(squeeze_momentum_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            squeeze_momentum_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }
    m.add_class::<SqueezeMomentumStreamPy>()?;

    // Register TRIX functions with their user-facing names
    m.add_function(wrap_pyfunction!(trix_py, m)?)?;
    m.add_function(wrap_pyfunction!(trix_batch_py, m)?)?;
    m.add_class::<TrixStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(trix_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(trix_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register VAR functions with their user-facing names
    m.add_function(wrap_pyfunction!(var_py, m)?)?;
    m.add_function(wrap_pyfunction!(var_batch_py, m)?)?;
    m.add_class::<VarStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::var::{
            var_cuda_batch_dev_py, var_cuda_many_series_one_param_dev_py,
            VarDeviceArrayF32Py,
        };
        m.add_class::<VarDeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(var_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(var_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register VPCI functions with their user-facing names
    m.add_function(wrap_pyfunction!(vpci_py, m)?)?;
    m.add_function(wrap_pyfunction!(vpci_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(vpci_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(vpci_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register WCLPRICE functions with their user-facing names
    m.add_function(wrap_pyfunction!(wclprice_py, m)?)?;
    m.add_function(wrap_pyfunction!(wclprice_batch_py, m)?)?;
    m.add_class::<WclpriceStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(wclprice_cuda_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(wclprice_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            wclprice_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Damiani Volatmeter functions with their user-facing names
    m.add_function(wrap_pyfunction!(damiani_py, m)?)?;
    m.add_function(wrap_pyfunction!(damiani_batch_py, m)?)?;
    m.add_class::<DamianiVolatmeterStreamPy>()?;
    m.add_class::<DamianiVolatmeterFeedStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::damiani_volatmeter::{
            damiani_cuda_batch_dev_py,
            damiani_cuda_many_series_one_param_dev_py,
            DeviceArrayF32DamianiPy,
        };
        m.add_class::<DeviceArrayF32DamianiPy>()?;
        m.add_function(wrap_pyfunction!(damiani_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            damiani_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register EMD functions with their user-facing names
    m.add_function(wrap_pyfunction!(emd_py, m)?)?;
    m.add_function(wrap_pyfunction!(emd_batch_py, m)?)?;
    m.add_class::<EmdStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(emd_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(emd_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register CVI functions with their user-facing names
    m.add_function(wrap_pyfunction!(cvi_py, m)?)?;
    m.add_function(wrap_pyfunction!(cvi_batch_py, m)?)?;
    m.add_class::<CviStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(cvi_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(cvi_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register DI functions with their user-facing names
    m.add_function(wrap_pyfunction!(di_py, m)?)?;
    m.add_function(wrap_pyfunction!(di_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(di_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(di_cuda_many_series_one_param_dev_py, m)?)?;
    }
    m.add_class::<DiStreamPy>()?;

    // Register DM functions with their user-facing names
    m.add_function(wrap_pyfunction!(dm_py, m)?)?;
    m.add_function(wrap_pyfunction!(dm_batch_py, m)?)?;
    m.add_class::<DmStreamPy>()?;

    // Register EFI functions with their user-facing names
    m.add_function(wrap_pyfunction!(efi_py, m)?)?;
    m.add_function(wrap_pyfunction!(efi_batch_py, m)?)?;
    m.add_class::<EfiStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_class::<EfiDeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(efi_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(efi_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register FOSC functions with their user-facing names
    m.add_function(wrap_pyfunction!(fosc_py, m)?)?;
    m.add_function(wrap_pyfunction!(fosc_batch_py, m)?)?;
    m.add_class::<FoscStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(fosc_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(fosc_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register DX functions with their user-facing names
    m.add_function(wrap_pyfunction!(dx_py, m)?)?;
    m.add_function(wrap_pyfunction!(dx_batch_py, m)?)?;
    m.add_class::<DxStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(dx_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(dx_cuda_many_series_one_param_dev_py, m)?)?;
        // Export DX device array class (with CAI v3 + DLPack + context guard)
        use crate::indicators::dx::DxDeviceArrayF32Py;
        m.add_class::<DxDeviceArrayF32Py>()?;
    }

    // Register Fisher functions
    m.add_function(wrap_pyfunction!(fisher_py, m)?)?;
    m.add_function(wrap_pyfunction!(fisher_batch_py, m)?)?;
    m.add_class::<FisherStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        use crate::indicators::fisher::*;
        m.add_function(wrap_pyfunction!(fisher_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            fisher_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Keltner functions
    m.add_function(wrap_pyfunction!(keltner_py, m)?)?;
    m.add_function(wrap_pyfunction!(keltner_batch_py, m)?)?;
    m.add_class::<KeltnerStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        use crate::indicators::keltner::*;
        m.add_function(wrap_pyfunction!(keltner_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            keltner_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_class::<KeltnerDeviceArrayF32Py>()?;
    }

    // Register AO functions with their user-facing names
    m.add_function(wrap_pyfunction!(ao_py, m)?)?;
    m.add_function(wrap_pyfunction!(ao_batch_py, m)?)?;
    m.add_class::<AoStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        use crate::indicators::ao::*;
        use crate::indicators::coppock::*;
        m.add_function(wrap_pyfunction!(ao_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(ao_cuda_many_series_one_param_dev_py, m)?)?;
        // Coppock CUDA entrypoints
        m.add_function(wrap_pyfunction!(coppock_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            coppock_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        // Export Coppock CUDA device handle (CAI v3 + DLPack v1.x)
        m.add_class::<CoppockDeviceArrayF32Py>()?;
    }

    // Register ATR functions with their user-facing names
    m.add_function(wrap_pyfunction!(atr_py, m)?)?;
    m.add_function(wrap_pyfunction!(atr_batch_py, m)?)?;
    m.add_class::<AtrStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(atr_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(atr_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register AVSL functions with their user-facing names
    m.add_function(wrap_pyfunction!(avsl_py, m)?)?;
    m.add_function(wrap_pyfunction!(avsl_batch_py, m)?)?;
    m.add_class::<AvslStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(avsl_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(avsl_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register DMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(dma_py, m)?)?;
    m.add_function(wrap_pyfunction!(dma_batch_py, m)?)?;
    m.add_class::<DmaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(dma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(dma_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Range Filter functions with their user-facing names
    m.add_function(wrap_pyfunction!(range_filter_py, m)?)?;
    m.add_function(wrap_pyfunction!(range_filter_batch_py, m)?)?;
    m.add_class::<RangeFilterStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_class::<RangeFilterDeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(range_filter_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            range_filter_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register SAMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(sama_py, m)?)?;
    m.add_function(wrap_pyfunction!(sama_batch_py, m)?)?;
    m.add_class::<SamaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(sama_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(sama_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register WTO functions with their user-facing names
    m.add_function(wrap_pyfunction!(wto_py, m)?)?;
    m.add_function(wrap_pyfunction!(wto_batch_py, m)?)?;
    m.add_class::<WtoStreamPy>()?;

    // Register EHMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(ehma_py, m)?)?;
    m.add_function(wrap_pyfunction!(ehma_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(ehma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(ehma_cuda_many_series_one_param_dev_py, m)?)?;
    }
    m.add_class::<EhmaStreamPy>()?;

    // Register NAMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(nama_py, m)?)?;
    m.add_function(wrap_pyfunction!(nama_batch_py, m)?)?;
    m.add_class::<NamaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(nama_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(nama_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register CCI functions with their user-facing names
    m.add_function(wrap_pyfunction!(cci_py, m)?)?;
    m.add_function(wrap_pyfunction!(cci_batch_py, m)?)?;
    m.add_class::<CciStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_class::<CciDeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(cci_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(cci_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register CCI Cycle functions
    m.add_function(wrap_pyfunction!(cci_cycle_py, m)?)?;
    m.add_function(wrap_pyfunction!(cci_cycle_batch_py, m)?)?;
    m.add_class::<CciCycleStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(cci_cycle_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            cci_cycle_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register HalfTrend functions
    m.add_function(wrap_pyfunction!(halftrend_py, m)?)?;
    m.add_function(wrap_pyfunction!(halftrend_tuple_py, m)?)?; // Compatibility function for tuple return
    m.add_function(wrap_pyfunction!(halftrend_batch_py, m)?)?;
    m.add_class::<HalfTrendStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(halftrend_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            halftrend_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register unified VAMA (Volatility/Volume Adjusted MA combined under `vama`)
    m.add_function(wrap_pyfunction!(vama_unified_py, m)?)?;
    m.add_function(wrap_pyfunction!(vama_batch_unified_py, m)?)?;
    m.add_class::<VamaStreamUnifiedPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(vama_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(vama_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register FVG Trailing Stop functions
    m.add_function(wrap_pyfunction!(fvg_trailing_stop_py, m)?)?;
    m.add_function(wrap_pyfunction!(fvg_trailing_stop_batch_py, m)?)?;
    m.add_class::<FvgTrailingStopStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(fvg_trailing_stop_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            fvg_trailing_stop_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register NET MyRSI functions
    m.add_function(wrap_pyfunction!(net_myrsi_py, m)?)?;
    m.add_function(wrap_pyfunction!(net_myrsi_batch_py, m)?)?;
    m.add_class::<NetMyrsiStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(net_myrsi_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            net_myrsi_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Reverse RSI functions
    m.add_function(wrap_pyfunction!(reverse_rsi_py, m)?)?;
    m.add_function(wrap_pyfunction!(reverse_rsi_batch_py, m)?)?;
    m.add_class::<ReverseRsiStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::reverse_rsi::reverse_rsi_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::reverse_rsi::reverse_rsi_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Ehlers Error Correcting EMA functions
    m.add_function(wrap_pyfunction!(ehlers_ecema_py, m)?)?;
    m.add_function(wrap_pyfunction!(ehlers_ecema_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(ehlers_ecema_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            ehlers_ecema_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }
    m.add_class::<EhlersEcemaStreamPy>()?;

    // Register CFO functions with their user-facing names
    m.add_function(wrap_pyfunction!(cfo_py, m)?)?;
    m.add_function(wrap_pyfunction!(cfo_batch_py, m)?)?;
    m.add_class::<CfoStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(cfo_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(cfo_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register BOP functions with their user-facing names
    m.add_function(wrap_pyfunction!(bop_py, m)?)?;
    m.add_function(wrap_pyfunction!(bop_batch_py, m)?)?;
    m.add_class::<BopStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(bop_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(bop_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Buff Averages
    m.add_function(wrap_pyfunction!(buff_averages_py, m)?)?;
    m.add_function(wrap_pyfunction!(buff_averages_batch_py, m)?)?;
    m.add_class::<BuffAveragesStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(buff_averages_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            buff_averages_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // QQE
    m.add_function(wrap_pyfunction!(qqe_py, m)?)?;
    m.add_function(wrap_pyfunction!(qqe_batch_py, m)?)?;
    m.add_class::<QqeStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(qqe_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(qqe_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Volume Adjusted MA (explicit API)
    m.add_function(wrap_pyfunction!(vama_volu::volume_adjusted_ma_py, m)?)?;
    m.add_function(wrap_pyfunction!(vama_volu::volume_adjusted_ma_batch_py, m)?)?;
    m.add_class::<vama_volu::VolumeAdjustedMaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(volume_adjusted_ma_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            volume_adjusted_ma_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Nadaraya-Watson Envelope
    m.add_function(wrap_pyfunction!(nadaraya_watson_envelope_py, m)?)?;
    m.add_function(wrap_pyfunction!(nadaraya_watson_envelope_batch_py, m)?)?;
    m.add_class::<NweStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::nadaraya_watson_envelope::nadaraya_watson_envelope_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(crate::indicators::nadaraya_watson_envelope::nadaraya_watson_envelope_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // TTM Squeeze
    m.add_function(wrap_pyfunction!(ttm_squeeze_py, m)?)?;
    m.add_function(wrap_pyfunction!(ttm_squeeze_batch_py, m)?)?;
    m.add_class::<TtmSqueezeStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::ttm_squeeze::ttm_squeeze_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::ttm_squeeze::ttm_squeeze_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Modified God Mode
    m.add_function(wrap_pyfunction!(mod_god_mode_py, m)?)?;
    m.add_function(wrap_pyfunction!(mod_god_mode_batch_py, m)?)?;
    m.add_class::<ModGodModeStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::mod_god_mode::mod_god_mode_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::mod_god_mode::mod_god_mode_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Linear Regression Intercept functions with their user-facing names
    m.add_function(wrap_pyfunction!(linearreg_intercept_py, m)?)?;
    m.add_function(wrap_pyfunction!(linearreg_intercept_batch_py, m)?)?;
    m.add_class::<LinearRegInterceptStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        use crate::indicators::linearreg_intercept::LinearRegInterceptDeviceArrayF32Py;
        m.add_function(wrap_pyfunction!(
            crate::indicators::linearreg_intercept::linearreg_intercept_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(crate::indicators::linearreg_intercept::linearreg_intercept_cuda_many_series_one_param_dev_py, m)?)?;
        m.add_class::<LinearRegInterceptDeviceArrayF32Py>()?;
    }

    // Register Mass Index functions with their user-facing names
    m.add_function(wrap_pyfunction!(mass_py, m)?)?;
    m.add_function(wrap_pyfunction!(mass_batch_py, m)?)?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::mass::mass_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::mass::mass_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }
    m.add_class::<MassStreamPy>()?;

    // Register Midprice functions with their user-facing names
    m.add_function(wrap_pyfunction!(midprice_py, m)?)?;
    m.add_function(wrap_pyfunction!(midprice_batch_py, m)?)?;
    m.add_class::<MidpriceStreamPy>()?;

    // Register OBV functions with their user-facing names
    m.add_function(wrap_pyfunction!(obv_py, m)?)?;
    m.add_function(wrap_pyfunction!(obv_batch_py, m)?)?;
    m.add_class::<ObvStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(obv_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(obv_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Qstick functions with their user-facing names
    m.add_function(wrap_pyfunction!(qstick_py, m)?)?;
    m.add_function(wrap_pyfunction!(qstick_batch_py, m)?)?;
    m.add_class::<QstickStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::qstick::{
            qstick_cuda_batch_dev_py, qstick_cuda_many_series_one_param_dev_py,
        };
        m.add_function(wrap_pyfunction!(qstick_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            qstick_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register RSX functions with their user-facing names
    m.add_function(wrap_pyfunction!(rsx_py, m)?)?;
    m.add_function(wrap_pyfunction!(rsx_batch_py, m)?)?;
    m.add_class::<RsxStreamPy>()?;

    // Register STC functions with their user-facing names
    m.add_function(wrap_pyfunction!(stc_py, m)?)?;
    m.add_function(wrap_pyfunction!(stc_batch_py, m)?)?;
    m.add_class::<StcStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::stc::{
            stc_cuda_batch_dev_py, stc_cuda_many_series_one_param_dev_py,
        };
        m.add_function(wrap_pyfunction!(stc_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(stc_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register TSI functions with their user-facing names
    m.add_function(wrap_pyfunction!(tsi_py, m)?)?;
    m.add_function(wrap_pyfunction!(tsi_batch_py, m)?)?;
    m.add_class::<TsiStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_class::<TsiDeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(tsi_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(tsi_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register VIDYA functions with their user-facing names
    m.add_function(wrap_pyfunction!(vidya_py, m)?)?;
    m.add_function(wrap_pyfunction!(vidya_batch_py, m)?)?;
    m.add_class::<VidyaStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(vidya_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            vidya_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_class::<VidyaDeviceArrayF32Py>()?;
    }

    // Register WILLR functions with their user-facing names
    m.add_function(wrap_pyfunction!(willr_py, m)?)?;
    m.add_function(wrap_pyfunction!(willr_batch_py, m)?)?;
    m.add_class::<WillrStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(willr_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            willr_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_class::<WillrDeviceArrayF32Py>()?;
    }

    // Register ZSCORE functions with their user-facing names
    m.add_function(wrap_pyfunction!(zscore_py, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_batch_py, m)?)?;
    m.add_class::<ZscoreStreamPy>()?;

    // Register AlphaTrend functions with their user-facing names
    m.add_function(wrap_pyfunction!(alphatrend_py, m)?)?;
    m.add_class::<AlphaTrendStreamPy>()?;

    // Register GatorOsc functions with their user-facing names
    m.add_function(wrap_pyfunction!(gatorosc_py, m)?)?;
    m.add_function(wrap_pyfunction!(gatorosc_batch_py, m)?)?;
    m.add_class::<GatorOscStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        use crate::indicators::gatorosc::{
            gatorosc_cuda_batch_dev_py, gatorosc_cuda_many_series_one_param_dev_py,
            DeviceArrayF32GatorPy,
        };
        m.add_class::<DeviceArrayF32GatorPy>()?;
        m.add_function(wrap_pyfunction!(gatorosc_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            gatorosc_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Kurtosis functions with their user-facing names
    m.add_function(wrap_pyfunction!(kurtosis_py, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis_batch_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(kurtosis_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            kurtosis_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }
    m.add_class::<KurtosisStreamPy>()?;

    // Register MAB functions with their user-facing names
    m.add_function(wrap_pyfunction!(mab_py, m)?)?;
    m.add_function(wrap_pyfunction!(mab_batch_py, m)?)?;
    m.add_class::<MabStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(mab_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(mab_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register medprice functions
    m.add_function(wrap_pyfunction!(medprice_py, m)?)?;
    m.add_function(wrap_pyfunction!(medprice_batch_py, m)?)?;
    m.add_class::<MedpriceStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(medprice_cuda_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(medprice_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            medprice_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register MSW functions with their user-facing names
    m.add_function(wrap_pyfunction!(msw_py, m)?)?;
    m.add_function(wrap_pyfunction!(msw_batch_py, m)?)?;
    m.add_class::<MswStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(msw_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(msw_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register PMA functions with their user-facing names
    m.add_function(wrap_pyfunction!(pma_py, m)?)?;
    m.add_function(wrap_pyfunction!(pma_batch_py, m)?)?;
    m.add_class::<PmaStreamPy>()?;

    // Register ROCR functions with their user-facing names
    m.add_function(wrap_pyfunction!(rocr_py, m)?)?;
    m.add_function(wrap_pyfunction!(rocr_batch_py, m)?)?;
    m.add_class::<RocrStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(rocr_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(rocr_cuda_many_series_one_param_dev_py, m)?)?;
        m.add_class::<RocrDeviceArrayF32Py>()?;
    }

    // Register SAR functions with their user-facing names
    m.add_function(wrap_pyfunction!(sar_py, m)?)?;
    m.add_function(wrap_pyfunction!(sar_batch_py, m)?)?;
    m.add_class::<SarStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_class::<SarDeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(sar_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(sar_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register SuperTrend functions with their user-facing names
    m.add_function(wrap_pyfunction!(supertrend_py, m)?)?;
    m.add_function(wrap_pyfunction!(supertrend_batch_py, m)?)?;
    m.add_class::<SuperTrendStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(supertrend_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            supertrend_cuda_many_series_one_param_dev_py,
            m
        )?)?;
        m.add_class::<SupertrendDeviceArrayF32Py>()?;
    }

    // Register UltOsc functions with their user-facing names
    m.add_function(wrap_pyfunction!(ultosc_py, m)?)?;
    m.add_function(wrap_pyfunction!(ultosc_batch_py, m)?)?;
    m.add_class::<UltOscStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::ultosc::ultosc_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::ultosc::ultosc_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Voss functions with their user-facing names
    m.add_function(wrap_pyfunction!(voss_py, m)?)?;
    m.add_function(wrap_pyfunction!(voss_batch_py, m)?)?;
    m.add_class::<VossStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::voss::voss_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::voss::voss_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Wavetrend functions with their user-facing names
    m.add_function(wrap_pyfunction!(wavetrend_py, m)?)?;
    m.add_function(wrap_pyfunction!(wavetrend_batch_py, m)?)?;
    m.add_class::<WavetrendStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::wavetrend::wavetrend_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::wavetrend::wavetrend_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register KST functions with their user-facing names
    m.add_function(wrap_pyfunction!(kst_py, m)?)?;
    m.add_function(wrap_pyfunction!(kst_batch_py, m)?)?;
    m.add_class::<KstStreamPy>()?;

    // Register LRSI functions with their user-facing names
    m.add_function(wrap_pyfunction!(lrsi_py, m)?)?;
    m.add_function(wrap_pyfunction!(lrsi_batch_py, m)?)?;
    m.add_class::<LrsiStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(lrsi_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(lrsi_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Mean AD functions with their user-facing names
    m.add_function(wrap_pyfunction!(mean_ad_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_ad_batch_py, m)?)?;
    m.add_class::<MeanAdStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(mean_ad_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            mean_ad_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register MOM functions with their user-facing names
    m.add_function(wrap_pyfunction!(mom_py, m)?)?;
    m.add_function(wrap_pyfunction!(mom_batch_py, m)?)?;
    m.add_class::<MomStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(mom_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(mom_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register Pivot functions with their user-facing names
    m.add_function(wrap_pyfunction!(pivot_py, m)?)?;
    m.add_function(wrap_pyfunction!(pivot_batch_py, m)?)?;
    m.add_class::<PivotStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::pivot::pivot_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::pivot::pivot_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register ROCP functions with their user-facing names
    m.add_function(wrap_pyfunction!(rocp_py, m)?)?;
    m.add_function(wrap_pyfunction!(rocp_batch_py, m)?)?;
    m.add_class::<RocpStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(rocp_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(rocp_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register SafeZoneStop functions with their user-facing names
    m.add_function(wrap_pyfunction!(safezonestop_py, m)?)?;
    m.add_function(wrap_pyfunction!(safezonestop_batch_py, m)?)?;
    m.add_class::<SafeZoneStopStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(safezonestop_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            safezonestop_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register Stoch functions with their user-facing names
    m.add_function(wrap_pyfunction!(stoch_py, m)?)?;
    m.add_function(wrap_pyfunction!(stoch_batch_py, m)?)?;
    m.add_class::<StochStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_class::<StochDeviceArrayF32Py>()?;
        m.add_function(wrap_pyfunction!(stoch_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            stoch_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register StochF functions with their user-facing names
    m.add_function(wrap_pyfunction!(stochf_py, m)?)?;
    m.add_function(wrap_pyfunction!(stochf_batch_py, m)?)?;
    m.add_class::<StochfStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(stochf_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(
            stochf_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register UI functions with their user-facing names
    m.add_function(wrap_pyfunction!(ui_py, m)?)?;
    m.add_function(wrap_pyfunction!(ui_batch_py, m)?)?;
    m.add_class::<UiStreamPy>()?;
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        m.add_function(wrap_pyfunction!(
            crate::indicators::ui::ui_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::ui::ui_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Register VOSC functions with their user-facing names
    m.add_function(wrap_pyfunction!(vosc_py, m)?)?;
    m.add_function(wrap_pyfunction!(vosc_batch_py, m)?)?;
    m.add_class::<VoscStreamPy>()?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(vosc_cuda_batch_dev_py, m)?)?;
        m.add_function(wrap_pyfunction!(vosc_cuda_many_series_one_param_dev_py, m)?)?;
    }

    // Register WAD functions with their user-facing names
    m.add_function(wrap_pyfunction!(wad_py, m)?)?;
    m.add_function(wrap_pyfunction!(wad_batch_py, m)?)?;
    m.add_class::<WadStreamPy>()?;

    // Register Chande functions with their user-facing names
    m.add_function(wrap_pyfunction!(chande_py, m)?)?;
    m.add_function(wrap_pyfunction!(chande_batch_py, m)?)?;
    m.add_class::<ChandeStreamPy>()?;
    // CUDA (feature-gated)
    #[cfg(all(feature = "python", feature = "cuda"))]
    {
        use crate::indicators::chande::DeviceArrayF32ChandePy;
        m.add_class::<DeviceArrayF32ChandePy>()?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::chande::chande_cuda_batch_dev_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            crate::indicators::chande::chande_cuda_many_series_one_param_dev_py,
            m
        )?)?;
    }

    // Add other indicators here as you implement their Python bindings

    Ok(())
}
