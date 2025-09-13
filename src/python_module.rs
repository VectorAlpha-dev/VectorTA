//! Python module registration for technical indicators

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn ta_indicators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register indicator modules
    crate::indicators::zscore::register_zscore_module(m)?;
    crate::indicators::moving_averages::alma::register_alma_module(m)?;
    crate::indicators::moving_averages::cwma::register_cwma_module(m)?;
    crate::indicators::moving_averages::edcf::register_edcf_module(m)?;
    crate::indicators::moving_averages::ehlers_kama::register_ehlers_kama_module(m)?;
    crate::indicators::moving_averages::hma::register_hma_module(m)?;
    crate::indicators::moving_averages::jsa::register_jsa_module(m)?;
    crate::indicators::moving_averages::volume_adjusted_ma::register_VolumeAdjustedMa_module(m)?;
    crate::indicators::nadaraya_watson_envelope::register_nadaraya_watson_envelope_module(m)?;
    crate::indicators::wto::register_wto_module(m)?;
    crate::indicators::voss::register_voss_module(m)?;
    crate::indicators::vi::register_vi_module(m)?;
    crate::indicators::ttm_trend::register_ttm_trend_module(m)?;
    crate::indicators::trix::register_trix_module(m)?;
    crate::indicators::stc::register_stc_module(m)?;
    crate::indicators::rvi::register_rvi_module(m)?;
    crate::indicators::rocp::register_rocp_module(m)?;
    crate::indicators::qstick::register_qstick_module(m)?;
    crate::indicators::ppo::register_ppo_module(m)?;
    crate::indicators::percentile_nearest_rank::register_percentile_nearest_rank_module(m)?;
    crate::indicators::obv::register_obv_module(m)?;
    crate::indicators::natr::register_natr_module(m)?;
    crate::indicators::mom::register_mom_module(m)?;
    crate::indicators::midprice::register_midprice_module(m)?;
    crate::indicators::medprice::register_medprice_module(m)?;
    crate::indicators::mass::register_mass_module(m)?;
    crate::indicators::macd::register_macd_module(m)?;
    crate::indicators::lpc::register_lpc_module(m)?;
    crate::indicators::linearreg_angle::register_linearreg_angle_module(m)?;
    crate::indicators::kst::register_kst_module(m)?;
    
    // Add other indicators here as they are updated
    
    Ok(())
}