use super::{
    IndicatorBatchOutput, IndicatorBatchRequest, IndicatorDataRef, IndicatorDispatchError,
    IndicatorParamSet, ParamKV, ParamValue,
};
use crate::indicators::acosc::{acosc_with_kernel, AcoscInput, AcoscParams};
use crate::indicators::ad::{ad_with_kernel, AdInput, AdParams};
use crate::indicators::adosc::{adosc_with_kernel, AdoscInput, AdoscParams};
use crate::indicators::adx::{adx_with_kernel, AdxInput, AdxParams};
use crate::indicators::adxr::{adxr_with_kernel, AdxrInput, AdxrParams};
use crate::indicators::alligator::{alligator_with_kernel, AlligatorInput, AlligatorParams};
use crate::indicators::alphatrend::{alphatrend_with_kernel, AlphaTrendInput, AlphaTrendParams};
use crate::indicators::ao::{ao_into_slice, AoInput, AoParams};
use crate::indicators::apo::{apo_with_kernel, ApoInput, ApoParams};
use crate::indicators::aroon::{aroon_with_kernel, AroonInput, AroonParams};
use crate::indicators::aroonosc::{aroon_osc_with_kernel, AroonOscInput, AroonOscParams};
use crate::indicators::aso::{aso_with_kernel, AsoInput, AsoParams};
use crate::indicators::atr::{atr_with_kernel, AtrInput, AtrParams};
use crate::indicators::autocorrelation_indicator::{
    autocorrelation_indicator_with_kernel, AutocorrelationIndicatorInput,
    AutocorrelationIndicatorParams,
};
use crate::indicators::avsl::{avsl_with_kernel, AvslInput, AvslParams};
use crate::indicators::bandpass::{bandpass_with_kernel, BandPassInput, BandPassParams};
use crate::indicators::bollinger_bands::{
    bollinger_bands_with_kernel, BollingerBandsInput, BollingerBandsParams,
};
use crate::indicators::bollinger_bands_width::{
    bollinger_bands_width_with_kernel, BollingerBandsWidthInput, BollingerBandsWidthParams,
};
use crate::indicators::bop::{bop_with_kernel, BopInput, BopParams};
use crate::indicators::candle_strength_oscillator::{
    candle_strength_oscillator_with_kernel, CandleStrengthOscillatorInput,
    CandleStrengthOscillatorParams,
};
use crate::indicators::cci::{cci_with_kernel, CciInput, CciParams};
use crate::indicators::cci_cycle::{cci_cycle_with_kernel, CciCycleInput, CciCycleParams};
use crate::indicators::cfo::{cfo_with_kernel, CfoInput, CfoParams};
use crate::indicators::chande::{chande_with_kernel, ChandeInput, ChandeParams};
use crate::indicators::chandelier_exit::{
    chandelier_exit_with_kernel, ChandelierExitInput, ChandelierExitParams,
};
use crate::indicators::chop::{chop_with_kernel, ChopInput, ChopParams};
use crate::indicators::cksp::{cksp_with_kernel, CkspInput, CkspParams};
use crate::indicators::cmo::{cmo_with_kernel, CmoInput, CmoParams};
use crate::indicators::coppock::{coppock_with_kernel, CoppockInput, CoppockParams};
use crate::indicators::correl_hl::{correl_hl_with_kernel, CorrelHlInput, CorrelHlParams};
use crate::indicators::correlation_cycle::{
    correlation_cycle_with_kernel, CorrelationCycleInput, CorrelationCycleParams,
};
use crate::indicators::cyberpunk_value_trend_analyzer::{
    cyberpunk_value_trend_analyzer_with_kernel, CyberpunkValueTrendAnalyzerInput,
    CyberpunkValueTrendAnalyzerParams,
};
use crate::indicators::damiani_volatmeter::{
    damiani_volatmeter_with_kernel, DamianiVolatmeterInput, DamianiVolatmeterParams,
};
use crate::indicators::deviation::{deviation_with_kernel, DeviationInput, DeviationParams};
use crate::indicators::devstop::{devstop_with_kernel, DevStopInput, DevStopParams};
use crate::indicators::di::{di_with_kernel, DiInput, DiParams};
use crate::indicators::directional_imbalance_index::{
    directional_imbalance_index_with_kernel, DirectionalImbalanceIndexInput,
    DirectionalImbalanceIndexParams,
};
use crate::indicators::disparity_index::{
    disparity_index_into_slice, DisparityIndexInput, DisparityIndexParams,
};
use crate::indicators::dm::{dm_with_kernel, DmInput, DmParams};
use crate::indicators::donchian::{donchian_with_kernel, DonchianInput, DonchianParams};
use crate::indicators::donchian_channel_width::{
    donchian_channel_width_into_slice, DonchianChannelWidthInput, DonchianChannelWidthParams,
};
use crate::indicators::dpo::{dpo_with_kernel, DpoInput, DpoParams};
use crate::indicators::dti::{dti_into_slice, DtiInput, DtiParams};
use crate::indicators::dual_ulcer_index::{
    dual_ulcer_index_with_kernel, DualUlcerIndexInput, DualUlcerIndexParams,
};
use crate::indicators::dvdiqqe::{dvdiqqe_with_kernel, DvdiqqeInput, DvdiqqeParams};
use crate::indicators::dx::{dx_batch_with_kernel, dx_into_slice, DxBatchRange, DxInput, DxParams};
use crate::indicators::dynamic_momentum_index::{
    dynamic_momentum_index_into_slice, dynamic_momentum_index_with_kernel,
    DynamicMomentumIndexInput, DynamicMomentumIndexParams,
};
use crate::indicators::efi::{efi_with_kernel, EfiInput, EfiParams};
use crate::indicators::ehlers_data_sampling_relative_strength_indicator::{
    ehlers_data_sampling_relative_strength_indicator_with_kernel,
    EhlersDataSamplingRelativeStrengthIndicatorInput,
    EhlersDataSamplingRelativeStrengthIndicatorParams,
};
use crate::indicators::emd::{emd_with_kernel, EmdInput, EmdParams};
use crate::indicators::emd_trend::{emd_trend_with_kernel, EmdTrendInput, EmdTrendParams};
use crate::indicators::emv::{emv_with_kernel, EmvInput};
use crate::indicators::er::{er_with_kernel, ErInput, ErParams};
use crate::indicators::eri::{eri_with_kernel, EriInput, EriParams};
use crate::indicators::evasive_supertrend::{
    evasive_supertrend_with_kernel, EvasiveSuperTrendInput, EvasiveSuperTrendParams,
};
use crate::indicators::reversal_signals::{
    reversal_signals_with_kernel, ReversalSignalsInput, ReversalSignalsParams,
};
use crate::indicators::fisher::{fisher_with_kernel, FisherInput, FisherParams};
use crate::indicators::fosc::{fosc_with_kernel, FoscInput, FoscParams};
use crate::indicators::fractal_dimension_index::{
    fractal_dimension_index_with_kernel, FractalDimensionIndexInput, FractalDimensionIndexParams,
};
use crate::indicators::fvg_positioning_average::{
    fvg_positioning_average_with_kernel, FvgPositioningAverageInput, FvgPositioningAverageParams,
};
use crate::indicators::fvg_trailing_stop::{
    fvg_trailing_stop_with_kernel, FvgTrailingStopInput, FvgTrailingStopParams,
};
use crate::indicators::gatorosc::{gatorosc_with_kernel, GatorOscInput, GatorOscParams};
use crate::indicators::gmma_oscillator::{
    gmma_oscillator_with_kernel, GmmaOscillatorInput, GmmaOscillatorParams,
};
use crate::indicators::goertzel_cycle_composite_wave::{
    goertzel_cycle_composite_wave_into_slice, GoertzelCycleCompositeWaveInput,
    GoertzelCycleCompositeWaveParams, GoertzelDetrendMode,
};
use crate::indicators::halftrend::{halftrend_with_kernel, HalfTrendInput, HalfTrendParams};
use crate::indicators::historical_volatility_rank::{
    historical_volatility_rank_with_kernel, HistoricalVolatilityRankInput,
    HistoricalVolatilityRankParams,
};
use crate::indicators::ift_rsi::{ift_rsi_with_kernel, IftRsiInput, IftRsiParams};
use crate::indicators::kairi_relative_index::{
    kairi_relative_index_into_slice, KairiRelativeIndexInput, KairiRelativeIndexParams,
};
use crate::indicators::kaufmanstop::{
    kaufmanstop_with_kernel, KaufmanstopInput, KaufmanstopParams,
};
use crate::indicators::kdj::{kdj_with_kernel, KdjInput, KdjParams};
use crate::indicators::keltner::{keltner_with_kernel, KeltnerInput, KeltnerParams};
use crate::indicators::kst::{kst_with_kernel, KstInput, KstParams};
use crate::indicators::kurtosis::{kurtosis_with_kernel, KurtosisInput, KurtosisParams};
use crate::indicators::kvo::{kvo_with_kernel, KvoInput, KvoParams};
use crate::indicators::linearreg_angle::{
    linearreg_angle_with_kernel, Linearreg_angleInput, Linearreg_angleParams,
};
use crate::indicators::linearreg_intercept::{
    linearreg_intercept_with_kernel, LinearRegInterceptInput, LinearRegInterceptParams,
};
use crate::indicators::linearreg_slope::{
    linearreg_slope_with_kernel, LinearRegSlopeInput, LinearRegSlopeParams,
};
use crate::indicators::lpc::{lpc_with_kernel, LpcInput, LpcParams};
use crate::indicators::lrsi::{lrsi_with_kernel, LrsiInput, LrsiParams};
use crate::indicators::mab::{mab_with_kernel, MabInput, MabParams};
use crate::indicators::macd::{macd_with_kernel, MacdInput, MacdParams};
use crate::indicators::macz::{macz_with_kernel, MaczInput, MaczParams};
use crate::indicators::market_structure_trailing_stop::{
    market_structure_trailing_stop_with_kernel, MarketStructureTrailingStopInput,
    MarketStructureTrailingStopParams,
};
use crate::indicators::mass::{mass_with_kernel, MassInput, MassParams};
use crate::indicators::mean_ad::{mean_ad_with_kernel, MeanAdInput, MeanAdParams};
use crate::indicators::medium_ad::{medium_ad_with_kernel, MediumAdInput, MediumAdParams};
use crate::indicators::medprice::{medprice_with_kernel, MedpriceInput, MedpriceParams};
use crate::indicators::mfi::{
    mfi_batch_with_kernel, mfi_into_slice, MfiBatchRange, MfiInput, MfiParams,
};
use crate::indicators::midpoint::{midpoint_with_kernel, MidpointInput, MidpointParams};
use crate::indicators::midprice::{midprice_with_kernel, MidpriceInput, MidpriceParams};
use crate::indicators::minmax::{minmax_with_kernel, MinmaxInput, MinmaxParams};
use crate::indicators::mod_god_mode::{
    mod_god_mode, ModGodModeData, ModGodModeInput, ModGodModeMode, ModGodModeParams,
};
use crate::indicators::mom::{mom_with_kernel, MomInput, MomParams};
use crate::indicators::moving_averages::ma::MaData;
use crate::indicators::moving_averages::ma_batch::{
    ma_batch_with_kernel_and_typed_params, MaBatchParamKV, MaBatchParamValue,
};
use crate::indicators::moving_averages::registry::list_moving_averages;
use crate::indicators::msw::{msw_with_kernel, MswInput, MswParams};
use crate::indicators::nadaraya_watson_envelope::{
    nadaraya_watson_envelope_with_kernel, NweInput, NweParams,
};
use crate::indicators::natr::{natr_with_kernel, NatrInput, NatrParams};
use crate::indicators::net_myrsi::{net_myrsi_with_kernel, NetMyrsiInput, NetMyrsiParams};
use crate::indicators::nonlinear_regression_zero_lag_moving_average::{
    nonlinear_regression_zero_lag_moving_average_with_kernel,
    NonlinearRegressionZeroLagMovingAverageInput, NonlinearRegressionZeroLagMovingAverageParams,
};
use crate::indicators::nvi::{nvi_with_kernel, NviInput, NviParams};
use crate::indicators::obv::{obv_with_kernel, ObvInput, ObvParams};
use crate::indicators::otto::{otto_with_kernel, OttoInput, OttoParams};
use crate::indicators::percentile_nearest_rank::{
    percentile_nearest_rank_with_kernel, PercentileNearestRankInput, PercentileNearestRankParams,
};
use crate::indicators::pfe::{pfe_with_kernel, PfeInput, PfeParams};
use crate::indicators::pivot::{pivot_with_kernel, PivotInput, PivotParams};
use crate::indicators::pma::{pma_with_kernel, PmaInput, PmaParams};
use crate::indicators::possible_rsi::{
    possible_rsi_with_kernel, PossibleRsiInput, PossibleRsiParams,
};
use crate::indicators::ppo::{ppo_with_kernel, PpoInput, PpoParams};
use crate::indicators::prb::{prb_with_kernel, PrbInput, PrbParams};
use crate::indicators::projection_oscillator::{
    projection_oscillator_with_kernel, ProjectionOscillatorInput, ProjectionOscillatorParams,
};
use crate::indicators::pvi::{pvi_with_kernel, PviInput, PviParams};
use crate::indicators::qqe::{qqe_with_kernel, QqeInput, QqeParams};
use crate::indicators::qstick::{qstick_with_kernel, QstickInput, QstickParams};
use crate::indicators::range_filter::{
    range_filter_with_kernel, RangeFilterInput, RangeFilterParams,
};
use crate::indicators::registry::{
    get_indicator, IndicatorInfo, IndicatorInputKind, ParamValueStatic,
};
use crate::indicators::reverse_rsi::{reverse_rsi_with_kernel, ReverseRsiInput, ReverseRsiParams};
use crate::indicators::roc::{roc_with_kernel, RocInput, RocParams};
use crate::indicators::rocp::{rocp_with_kernel, RocpInput, RocpParams};
use crate::indicators::rocr::{rocr_with_kernel, RocrInput, RocrParams};
use crate::indicators::rogers_satchell_volatility::{
    rogers_satchell_volatility_with_kernel, RogersSatchellVolatilityInput,
    RogersSatchellVolatilityParams,
};
use crate::indicators::rolling_skewness_kurtosis::{
    rolling_skewness_kurtosis_with_kernel, RollingSkewnessKurtosisInput,
    RollingSkewnessKurtosisParams,
};
use crate::indicators::rolling_z_score_trend::{
    rolling_z_score_trend_with_kernel, RollingZScoreTrendInput, RollingZScoreTrendParams,
};
use crate::indicators::rsi::{rsi_with_kernel, RsiInput, RsiParams};
use crate::indicators::rsmk::{rsmk_with_kernel, RsmkInput, RsmkParams};
use crate::indicators::rvi::{rvi_with_kernel, RviInput, RviParams};
use crate::indicators::safezonestop::{
    safezonestop_with_kernel, SafeZoneStopInput, SafeZoneStopParams,
};
use crate::indicators::squeeze_momentum::{
    squeeze_momentum_with_kernel, SqueezeMomentumInput, SqueezeMomentumParams,
};
use crate::indicators::srsi::{srsi_with_kernel, SrsiInput, SrsiParams};
use crate::indicators::stc::{stc_with_kernel, StcInput, StcParams};
use crate::indicators::stddev::{stddev_with_kernel, StdDevInput, StdDevParams};
use crate::indicators::stoch::{stoch_with_kernel, StochInput, StochParams};
use crate::indicators::stochastic_money_flow_index::{
    stochastic_money_flow_index_with_kernel, StochasticMoneyFlowIndexInput,
    StochasticMoneyFlowIndexParams,
};
use crate::indicators::stochf::{stochf_with_kernel, StochfInput, StochfParams};
use crate::indicators::supertrend::{supertrend_with_kernel, SuperTrendInput, SuperTrendParams};
use crate::indicators::trend_direction_force_index::{
    trend_direction_force_index_into_slice, TrendDirectionForceIndexInput,
    TrendDirectionForceIndexParams,
};
use crate::indicators::trix::{
    trix_batch_with_kernel, trix_into_slice, trix_with_kernel, TrixBatchRange, TrixInput,
    TrixParams,
};
use crate::indicators::tsf::{tsf_with_kernel, TsfInput, TsfParams};
use crate::indicators::tsi::{tsi_with_kernel, TsiInput, TsiParams};
use crate::indicators::ttm_squeeze::{ttm_squeeze_with_kernel, TtmSqueezeInput, TtmSqueezeParams};
use crate::indicators::ttm_trend::{ttm_trend_with_kernel, TtmTrendInput, TtmTrendParams};
use crate::indicators::ui::{ui_with_kernel, UiInput, UiParams};
use crate::indicators::ultosc::{ultosc_with_kernel, UltOscInput, UltOscParams};
use crate::indicators::var::{var_with_kernel, VarInput, VarParams};
use crate::indicators::velocity_acceleration_convergence_divergence_indicator::{
    velocity_acceleration_convergence_divergence_indicator_with_kernel,
    VelocityAccelerationConvergenceDivergenceIndicatorInput,
    VelocityAccelerationConvergenceDivergenceIndicatorParams,
};
use crate::indicators::vi::{vi_with_kernel, ViInput, ViParams};
use crate::indicators::vidya::{vidya_with_kernel, VidyaInput, VidyaParams};
use crate::indicators::vlma::{vlma_with_kernel, VlmaInput, VlmaParams};
use crate::indicators::volume_weighted_rsi::{
    volume_weighted_rsi_batch_with_kernel, volume_weighted_rsi_into_slice,
    VolumeWeightedRsiBatchRange, VolumeWeightedRsiInput, VolumeWeightedRsiParams,
};
use crate::indicators::vosc::{vosc_with_kernel, VoscInput, VoscParams};
use crate::indicators::voss::{voss_with_kernel, VossInput, VossParams};
use crate::indicators::vpci::{vpci_with_kernel, VpciInput, VpciParams};
use crate::indicators::vpt::{vpt_with_kernel, VptInput};
use crate::indicators::vwmacd::{vwmacd_with_kernel, VwmacdInput, VwmacdParams};
use crate::indicators::wad::{wad_with_kernel, WadInput};
use crate::indicators::wavetrend::{wavetrend_with_kernel, WavetrendInput, WavetrendParams};
use crate::indicators::wclprice::{wclprice_with_kernel, WclpriceInput};
use crate::indicators::willr::{willr_with_kernel, WillrInput, WillrParams};
use crate::indicators::wto::{wto_with_kernel, WtoInput, WtoParams};
use crate::indicators::yang_zhang_volatility::{
    yang_zhang_volatility_with_kernel, YangZhangVolatilityInput, YangZhangVolatilityParams,
};
use crate::indicators::zig_zag_channels::{
    zig_zag_channels_with_kernel, ZigZagChannelsInput, ZigZagChannelsParams,
};
use crate::indicators::zscore::{zscore_with_kernel, ZscoreInput, ZscoreParams};
use crate::indicators::{cg::cg_with_kernel, cg::CgInput, cg::CgParams};
use crate::utilities::data_loader::source_type;
use crate::utilities::enums::Kernel;
use std::collections::HashMap;

pub fn compute_cpu_batch(
    req: IndicatorBatchRequest<'_>,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    compute_cpu_batch_internal(req, false)
}

pub fn compute_cpu_batch_strict(
    req: IndicatorBatchRequest<'_>,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    compute_cpu_batch_internal(req, true)
}

fn compute_cpu_batch_internal(
    req: IndicatorBatchRequest<'_>,
    strict_inputs: bool,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    if !strict_inputs {
        if let Some(out) = try_fast_dispatch_non_strict(req) {
            return out;
        }
    }

    let info = get_indicator(req.indicator_id).ok_or_else(|| {
        IndicatorDispatchError::UnknownIndicator {
            id: req.indicator_id.to_string(),
        }
    })?;

    if !info.capabilities.supports_cpu_batch {
        return Err(IndicatorDispatchError::UnsupportedCapability {
            indicator: info.id.to_string(),
            capability: "cpu_batch",
        });
    }

    if strict_inputs {
        validate_input_kind_strict(info.id, info.input_kind, req.data)?;
    }

    let output_id = resolve_output_id(info, req.output_id)?;

    dispatch_cpu_batch_by_indicator(req, info, output_id)
}

fn try_fast_dispatch_non_strict(
    req: IndicatorBatchRequest<'_>,
) -> Option<Result<IndicatorBatchOutput, IndicatorDispatchError>> {
    let id = req.indicator_id;
    let output_id = req.output_id;

    if !id.as_bytes().iter().any(|b| b.is_ascii_uppercase()) {
        return match id {
            "bop" => Some(compute_bop_batch(req, output_id.unwrap_or("value"))),
            "dpo" => Some(compute_dpo_batch(req, output_id.unwrap_or("value"))),
            "cmo" => Some(compute_cmo_batch(req, output_id.unwrap_or("value"))),
            "fosc" => Some(compute_fosc_batch(req, output_id.unwrap_or("value"))),
            "emv" => Some(compute_emv_batch(req, output_id.unwrap_or("value"))),
            "cci_cycle" => Some(compute_cci_cycle_batch(req, output_id.unwrap_or("value"))),
            "cfo" => Some(compute_cfo_batch(req, output_id.unwrap_or("value"))),
            "lrsi" => Some(compute_lrsi_batch(req, output_id.unwrap_or("value"))),
            "nvi" => Some(compute_nvi_batch(req, output_id.unwrap_or("value"))),
            "mom" => Some(compute_mom_batch(req, output_id.unwrap_or("value"))),
            "vi" => {
                if let Some(out) = output_id {
                    Some(compute_vi_batch(req, out))
                } else {
                    None
                }
            }
            "wto" => {
                if let Some(out) = output_id {
                    Some(compute_wto_batch(req, out))
                } else {
                    None
                }
            }
            "rogers_satchell_volatility" => {
                if let Some(out) = output_id {
                    Some(compute_rogers_satchell_volatility_batch(req, out))
                } else {
                    None
                }
            }
            "historical_volatility_rank" => {
                if let Some(out) = output_id {
                    Some(compute_historical_volatility_rank_batch(req, out))
                } else {
                    None
                }
            }
            "dual_ulcer_index" => {
                if let Some(out) = output_id {
                    Some(compute_dual_ulcer_index_batch(req, out))
                } else {
                    None
                }
            }
            "fractal_dimension_index" => {
                if let Some(out) = output_id {
                    Some(compute_fractal_dimension_index_batch(req, out))
                } else {
                    None
                }
            }
            "volume_weighted_rsi" => {
                if let Some(out) = output_id {
                    Some(compute_volume_weighted_rsi_batch(req, out))
                } else {
                    None
                }
            }
            "dynamic_momentum_index" => {
                if let Some(out) = output_id {
                    Some(compute_dynamic_momentum_index_batch(req, out))
                } else {
                    None
                }
            }
            "disparity_index" => {
                if let Some(out) = output_id {
                    Some(compute_disparity_index_batch(req, out))
                } else {
                    None
                }
            }
            "donchian_channel_width" => {
                if let Some(out) = output_id {
                    Some(compute_donchian_channel_width_batch(req, out))
                } else {
                    None
                }
            }
            "kairi_relative_index" => {
                if let Some(out) = output_id {
                    Some(compute_kairi_relative_index_batch(req, out))
                } else {
                    None
                }
            }
            "projection_oscillator" => {
                if let Some(out) = output_id {
                    Some(compute_projection_oscillator_batch(req, out))
                } else {
                    None
                }
            }
            "market_structure_trailing_stop" => {
                if let Some(out) = output_id {
                    Some(compute_market_structure_trailing_stop_batch(req, out))
                } else {
                    None
                }
            }
            "emd_trend" => {
                if let Some(out) = output_id {
                    Some(compute_emd_trend_batch(req, out))
                } else {
                    None
                }
            }
            "cyberpunk_value_trend_analyzer" => {
                if let Some(out) = output_id {
                    Some(compute_cyberpunk_value_trend_analyzer_batch(req, out))
                } else {
                    None
                }
            }
            "evasive_supertrend" => {
                if let Some(out) = output_id {
                    Some(compute_evasive_supertrend_batch(req, out))
                } else {
                    None
                }
            }
            "reversal_signals" => {
                if let Some(out) = output_id {
                    Some(compute_reversal_signals_batch(req, out))
                } else {
                    None
                }
            }
            "zig_zag_channels" => {
                if let Some(out) = output_id {
                    Some(compute_zig_zag_channels_batch(req, out))
                } else {
                    None
                }
            }
            "directional_imbalance_index" => {
                if let Some(out) = output_id {
                    Some(compute_directional_imbalance_index_batch(req, out))
                } else {
                    None
                }
            }
            "candle_strength_oscillator" => {
                if let Some(out) = output_id {
                    Some(compute_candle_strength_oscillator_batch(req, out))
                } else {
                    None
                }
            }
            "gmma_oscillator" => {
                if let Some(out) = output_id {
                    Some(compute_gmma_oscillator_batch(req, out))
                } else {
                    None
                }
            }
            "nonlinear_regression_zero_lag_moving_average" => {
                if let Some(out) = output_id {
                    Some(compute_nonlinear_regression_zero_lag_moving_average_batch(
                        req, out,
                    ))
                } else {
                    None
                }
            }
            "possible_rsi" => {
                if let Some(out) = output_id {
                    Some(compute_possible_rsi_batch(req, out))
                } else {
                    None
                }
            }
            "autocorrelation_indicator" => {
                if let Some(out) = output_id {
                    Some(compute_autocorrelation_indicator_batch(req, out))
                } else {
                    None
                }
            }
            "goertzel_cycle_composite_wave" => {
                if let Some(out) = output_id {
                    Some(compute_goertzel_cycle_composite_wave_batch(req, out))
                } else {
                    None
                }
            }
            "rolling_skewness_kurtosis" => {
                if let Some(out) = output_id {
                    Some(compute_rolling_skewness_kurtosis_batch(req, out))
                } else {
                    None
                }
            }
            "rolling_z_score_trend" => {
                if let Some(out) = output_id {
                    Some(compute_rolling_z_score_trend_batch(req, out))
                } else {
                    None
                }
            }
            "ehlers_data_sampling_relative_strength_indicator" => {
                if let Some(out) = output_id {
                    Some(compute_ehlers_data_sampling_relative_strength_indicator_batch(req, out))
                } else {
                    None
                }
            }
            "velocity_acceleration_convergence_divergence_indicator" => {
                if let Some(out) = output_id {
                    Some(
                        compute_velocity_acceleration_convergence_divergence_indicator_batch(
                            req, out,
                        ),
                    )
                } else {
                    None
                }
            }
            "trend_direction_force_index" => {
                if let Some(out) = output_id {
                    Some(compute_trend_direction_force_index_batch(req, out))
                } else {
                    None
                }
            }
            "yang_zhang_volatility" => {
                if let Some(out) = output_id {
                    Some(compute_yang_zhang_volatility_batch(req, out))
                } else {
                    None
                }
            }
            "voss" => {
                if let Some(out) = output_id {
                    Some(compute_voss_batch(req, out))
                } else {
                    None
                }
            }
            "acosc" => {
                if let Some(out) = output_id {
                    Some(compute_acosc_batch(req, out))
                } else {
                    None
                }
            }
            _ => None,
        };
    }

    if id.eq_ignore_ascii_case("bop") {
        return Some(compute_bop_batch(req, output_id.unwrap_or("value")));
    }
    if id.eq_ignore_ascii_case("dpo") {
        return Some(compute_dpo_batch(req, output_id.unwrap_or("value")));
    }
    if id.eq_ignore_ascii_case("cmo") {
        return Some(compute_cmo_batch(req, output_id.unwrap_or("value")));
    }
    if id.eq_ignore_ascii_case("fosc") {
        return Some(compute_fosc_batch(req, output_id.unwrap_or("value")));
    }
    if id.eq_ignore_ascii_case("emv") {
        return Some(compute_emv_batch(req, output_id.unwrap_or("value")));
    }
    if id.eq_ignore_ascii_case("cfo") {
        return Some(compute_cfo_batch(req, output_id.unwrap_or("value")));
    }
    if id.eq_ignore_ascii_case("lrsi") {
        return Some(compute_lrsi_batch(req, output_id.unwrap_or("value")));
    }
    if id.eq_ignore_ascii_case("nvi") {
        return Some(compute_nvi_batch(req, output_id.unwrap_or("value")));
    }
    if id.eq_ignore_ascii_case("mom") {
        return Some(compute_mom_batch(req, output_id.unwrap_or("value")));
    }
    if id.eq_ignore_ascii_case("vi") {
        if let Some(out) = output_id {
            return Some(compute_vi_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("wto") {
        if let Some(out) = output_id {
            return Some(compute_wto_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("rogers_satchell_volatility") {
        if let Some(out) = output_id {
            return Some(compute_rogers_satchell_volatility_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("historical_volatility_rank") {
        if let Some(out) = output_id {
            return Some(compute_historical_volatility_rank_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("dual_ulcer_index") {
        if let Some(out) = output_id {
            return Some(compute_dual_ulcer_index_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("fractal_dimension_index") {
        if let Some(out) = output_id {
            return Some(compute_fractal_dimension_index_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("volume_weighted_rsi") {
        if let Some(out) = output_id {
            return Some(compute_volume_weighted_rsi_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("dynamic_momentum_index") {
        if let Some(out) = output_id {
            return Some(compute_dynamic_momentum_index_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("disparity_index") {
        if let Some(out) = output_id {
            return Some(compute_disparity_index_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("donchian_channel_width") {
        if let Some(out) = output_id {
            return Some(compute_donchian_channel_width_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("kairi_relative_index") {
        if let Some(out) = output_id {
            return Some(compute_kairi_relative_index_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("projection_oscillator") {
        if let Some(out) = output_id {
            return Some(compute_projection_oscillator_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("market_structure_trailing_stop") {
        if let Some(out) = output_id {
            return Some(compute_market_structure_trailing_stop_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("emd_trend") {
        if let Some(out) = output_id {
            return Some(compute_emd_trend_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("cyberpunk_value_trend_analyzer") {
        if let Some(out) = output_id {
            return Some(compute_cyberpunk_value_trend_analyzer_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("evasive_supertrend") {
        if let Some(out) = output_id {
            return Some(compute_evasive_supertrend_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("reversal_signals") {
        if let Some(out) = output_id {
            return Some(compute_reversal_signals_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("zig_zag_channels") {
        if let Some(out) = output_id {
            return Some(compute_zig_zag_channels_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("directional_imbalance_index") {
        if let Some(out) = output_id {
            return Some(compute_directional_imbalance_index_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("candle_strength_oscillator") {
        if let Some(out) = output_id {
            return Some(compute_candle_strength_oscillator_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("gmma_oscillator") {
        if let Some(out) = output_id {
            return Some(compute_gmma_oscillator_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("nonlinear_regression_zero_lag_moving_average") {
        if let Some(out) = output_id {
            return Some(compute_nonlinear_regression_zero_lag_moving_average_batch(
                req, out,
            ));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("autocorrelation_indicator") {
        if let Some(out) = output_id {
            return Some(compute_autocorrelation_indicator_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("goertzel_cycle_composite_wave") {
        if let Some(out) = output_id {
            return Some(compute_goertzel_cycle_composite_wave_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("rolling_skewness_kurtosis") {
        if let Some(out) = output_id {
            return Some(compute_rolling_skewness_kurtosis_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("rolling_z_score_trend") {
        if let Some(out) = output_id {
            return Some(compute_rolling_z_score_trend_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("ehlers_data_sampling_relative_strength_indicator") {
        if let Some(out) = output_id {
            return Some(compute_ehlers_data_sampling_relative_strength_indicator_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("velocity_acceleration_convergence_divergence_indicator") {
        if let Some(out) = output_id {
            return Some(
                compute_velocity_acceleration_convergence_divergence_indicator_batch(req, out),
            );
        }
        return None;
    }
    if id.eq_ignore_ascii_case("trend_direction_force_index") {
        if let Some(out) = output_id {
            return Some(compute_trend_direction_force_index_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("yang_zhang_volatility") {
        if let Some(out) = output_id {
            return Some(compute_yang_zhang_volatility_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("voss") {
        if let Some(out) = output_id {
            return Some(compute_voss_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("acosc") {
        if let Some(out) = output_id {
            return Some(compute_acosc_batch(req, out));
        }
        return None;
    }

    None
}

fn dispatch_cpu_batch_by_indicator(
    req: IndicatorBatchRequest<'_>,
    info: &IndicatorInfo,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    if is_moving_average(info.id) {
        return compute_ma_batch(req, info, output_id);
    }

    match info.id {
        "ad" => compute_ad_batch(req, output_id),
        "adosc" => compute_adosc_batch(req, output_id),
        "ao" => compute_ao_batch(req, output_id),
        "emv" => compute_emv_batch(req, output_id),
        "efi" => compute_efi_batch(req, output_id),
        "mfi" => compute_mfi_batch(req, output_id),
        "mass" => compute_mass_batch(req, output_id),
        "kvo" => compute_kvo_batch(req, output_id),
        "vosc" => compute_vosc_batch(req, output_id),
        "wad" => compute_wad_batch(req, output_id),
        "dx" => compute_dx_batch(req, output_id),
        "fosc" => compute_fosc_batch(req, output_id),
        "ift_rsi" => compute_ift_rsi_batch(req, output_id),
        "linearreg_angle" => compute_linearreg_angle_batch(req, output_id),
        "linearreg_intercept" => compute_linearreg_intercept_batch(req, output_id),
        "linearreg_slope" => compute_linearreg_slope_batch(req, output_id),
        "cg" => compute_cg_batch(req, output_id),
        "rsi" => compute_rsi_batch(req, output_id),
        "roc" => compute_roc_batch(req, output_id),
        "apo" => compute_apo_batch(req, output_id),
        "bop" => compute_bop_batch(req, output_id),
        "cci" => compute_cci_batch(req, output_id),
        "cci_cycle" => compute_cci_cycle_batch(req, output_id),
        "cfo" => compute_cfo_batch(req, output_id),
        "lrsi" => compute_lrsi_batch(req, output_id),
        "er" => compute_er_batch(req, output_id),
        "kurtosis" => compute_kurtosis_batch(req, output_id),
        "natr" => compute_natr_batch(req, output_id),
        "net_myrsi" => compute_net_myrsi_batch(req, output_id),
        "mean_ad" => compute_mean_ad_batch(req, output_id),
        "medium_ad" => compute_medium_ad_batch(req, output_id),
        "deviation" => compute_deviation_batch(req, output_id),
        "dpo" => compute_dpo_batch(req, output_id),
        "pfe" => compute_pfe_batch(req, output_id),
        "qstick" => compute_qstick_batch(req, output_id),
        "reverse_rsi" => compute_reverse_rsi_batch(req, output_id),
        "percentile_nearest_rank" => compute_percentile_nearest_rank_batch(req, output_id),
        "obv" => compute_obv_batch(req, output_id),
        "vpt" => compute_vpt_batch(req, output_id),
        "nvi" => compute_nvi_batch(req, output_id),
        "pvi" => compute_pvi_batch(req, output_id),
        "wclprice" => compute_wclprice_batch(req, output_id),
        "ui" => compute_ui_batch(req, output_id),
        "zscore" => compute_zscore_batch(req, output_id),
        "medprice" => compute_medprice_batch(req, output_id),
        "midpoint" => compute_midpoint_batch(req, output_id),
        "midprice" => compute_midprice_batch(req, output_id),
        "mom" => compute_mom_batch(req, output_id),
        "cmo" => compute_cmo_batch(req, output_id),
        "rocp" => compute_rocp_batch(req, output_id),
        "rocr" => compute_rocr_batch(req, output_id),
        "ppo" => compute_ppo_batch(req, output_id),
        "tsf" => compute_tsf_batch(req, output_id),
        "trix" => compute_trix_batch(req, output_id),
        "tsi" => compute_tsi_batch(req, output_id),
        "var" => compute_var_batch(req, output_id),
        "stddev" => compute_stddev_batch(req, output_id),
        "willr" => compute_willr_batch(req, output_id),
        "ultosc" => compute_ultosc_batch(req, output_id),
        "adx" => compute_adx_batch(req, output_id),
        "adxr" => compute_adxr_batch(req, output_id),
        "atr" => compute_atr_batch(req, output_id),
        "macd" => compute_macd_batch(req, output_id),
        "bollinger_bands" => compute_bollinger_batch(req, output_id),
        "bollinger_bands_width" => compute_bbw_batch(req, output_id),
        "stoch" => compute_stoch_batch(req, output_id),
        "stochf" => compute_stochf_batch(req, output_id),
        "stochastic_money_flow_index" => compute_stochastic_money_flow_index_batch(req, output_id),
        "vwmacd" => compute_vwmacd_batch(req, output_id),
        "vpci" => compute_vpci_batch(req, output_id),
        "ttm_trend" => compute_ttm_trend_batch(req, output_id),
        "ttm_squeeze" => compute_ttm_squeeze_batch(req, output_id),
        "aroon" => compute_aroon_batch(req, output_id),
        "aroonosc" => compute_aroonosc_batch(req, output_id),
        "di" => compute_di_batch(req, output_id),
        "dm" => compute_dm_batch(req, output_id),
        "dti" => compute_dti_batch(req, output_id),
        "donchian" => compute_donchian_batch(req, output_id),
        "kdj" => compute_kdj_batch(req, output_id),
        "keltner" => compute_keltner_batch(req, output_id),
        "squeeze_momentum" => compute_squeeze_momentum_batch(req, output_id),
        "srsi" => compute_srsi_batch(req, output_id),
        "supertrend" => compute_supertrend_batch(req, output_id),
        "vi" => compute_vi_batch(req, output_id),
        "wavetrend" => compute_wavetrend_batch(req, output_id),
        "wto" => compute_wto_batch(req, output_id),
        "rogers_satchell_volatility" => compute_rogers_satchell_volatility_batch(req, output_id),
        "historical_volatility_rank" => compute_historical_volatility_rank_batch(req, output_id),
        "dual_ulcer_index" => compute_dual_ulcer_index_batch(req, output_id),
        "fractal_dimension_index" => compute_fractal_dimension_index_batch(req, output_id),
        "volume_weighted_rsi" => compute_volume_weighted_rsi_batch(req, output_id),
        "dynamic_momentum_index" => compute_dynamic_momentum_index_batch(req, output_id),
        "disparity_index" => compute_disparity_index_batch(req, output_id),
        "donchian_channel_width" => compute_donchian_channel_width_batch(req, output_id),
        "kairi_relative_index" => compute_kairi_relative_index_batch(req, output_id),
        "projection_oscillator" => compute_projection_oscillator_batch(req, output_id),
        "market_structure_trailing_stop" => {
            compute_market_structure_trailing_stop_batch(req, output_id)
        }
        "emd_trend" => compute_emd_trend_batch(req, output_id),
        "cyberpunk_value_trend_analyzer" => {
            compute_cyberpunk_value_trend_analyzer_batch(req, output_id)
        }
        "evasive_supertrend" => compute_evasive_supertrend_batch(req, output_id),
        "reversal_signals" => compute_reversal_signals_batch(req, output_id),
        "zig_zag_channels" => compute_zig_zag_channels_batch(req, output_id),
        "directional_imbalance_index" => compute_directional_imbalance_index_batch(req, output_id),
        "candle_strength_oscillator" => compute_candle_strength_oscillator_batch(req, output_id),
        "gmma_oscillator" => compute_gmma_oscillator_batch(req, output_id),
        "nonlinear_regression_zero_lag_moving_average" => {
            compute_nonlinear_regression_zero_lag_moving_average_batch(req, output_id)
        }
        "possible_rsi" => compute_possible_rsi_batch(req, output_id),
        "autocorrelation_indicator" => compute_autocorrelation_indicator_batch(req, output_id),
        "goertzel_cycle_composite_wave" => {
            compute_goertzel_cycle_composite_wave_batch(req, output_id)
        }
        "rolling_skewness_kurtosis" => compute_rolling_skewness_kurtosis_batch(req, output_id),
        "rolling_z_score_trend" => compute_rolling_z_score_trend_batch(req, output_id),
        "ehlers_data_sampling_relative_strength_indicator" => {
            compute_ehlers_data_sampling_relative_strength_indicator_batch(req, output_id)
        }
        "velocity_acceleration_convergence_divergence_indicator" => {
            compute_velocity_acceleration_convergence_divergence_indicator_batch(req, output_id)
        }
        "trend_direction_force_index" => compute_trend_direction_force_index_batch(req, output_id),
        "yang_zhang_volatility" => compute_yang_zhang_volatility_batch(req, output_id),
        "acosc" => compute_acosc_batch(req, output_id),
        "alligator" => compute_alligator_batch(req, output_id),
        "alphatrend" => compute_alphatrend_batch(req, output_id),
        "aso" => compute_aso_batch(req, output_id),
        "avsl" => compute_avsl_batch(req, output_id),
        "bandpass" => compute_bandpass_batch(req, output_id),
        "chande" => compute_chande_batch(req, output_id),
        "chandelier_exit" => compute_chandelier_exit_batch(req, output_id),
        "cksp" => compute_cksp_batch(req, output_id),
        "coppock" => compute_coppock_batch(req, output_id),
        "correl_hl" => compute_correl_hl_batch(req, output_id),
        "correlation_cycle" => compute_correlation_cycle_batch(req, output_id),
        "damiani_volatmeter" => compute_damiani_volatmeter_batch(req, output_id),
        "dvdiqqe" => compute_dvdiqqe_batch(req, output_id),
        "emd" => compute_emd_batch(req, output_id),
        "eri" => compute_eri_batch(req, output_id),
        "fisher" => compute_fisher_batch(req, output_id),
        "fvg_positioning_average" => compute_fvg_positioning_average_batch(req, output_id),
        "fvg_trailing_stop" => compute_fvg_trailing_stop_batch(req, output_id),
        "gatorosc" => compute_gatorosc_batch(req, output_id),
        "halftrend" => compute_halftrend_batch(req, output_id),
        "kaufmanstop" => compute_kaufmanstop_batch(req, output_id),
        "kst" => compute_kst_batch(req, output_id),
        "lpc" => compute_lpc_batch(req, output_id),
        "mab" => compute_mab_batch(req, output_id),
        "macz" => compute_macz_batch(req, output_id),
        "minmax" => compute_minmax_batch(req, output_id),
        "mod_god_mode" => compute_mod_god_mode_batch(req, output_id),
        "msw" => compute_msw_batch(req, output_id),
        "nadaraya_watson_envelope" => compute_nadaraya_watson_envelope_batch(req, output_id),
        "otto" => compute_otto_batch(req, output_id),
        "vidya" => compute_vidya_batch(req, output_id),
        "vlma" => compute_vlma_batch(req, output_id),
        "pma" => compute_pma_batch(req, output_id),
        "prb" => compute_prb_batch(req, output_id),
        "qqe" => compute_qqe_batch(req, output_id),
        "range_filter" => compute_range_filter_batch(req, output_id),
        "rsmk" => compute_rsmk_batch(req, output_id),
        "voss" => compute_voss_batch(req, output_id),
        "stc" => compute_stc_batch(req, output_id),
        "rvi" => compute_rvi_batch(req, output_id),
        "safezonestop" => compute_safezonestop_batch(req, output_id),
        "devstop" => compute_devstop_batch(req, output_id),
        "chop" => compute_chop_batch(req, output_id),
        "pivot" => compute_pivot_batch(req, output_id),
        _ => Err(IndicatorDispatchError::UnsupportedCapability {
            indicator: info.id.to_string(),
            capability: "cpu_batch",
        }),
    }
}

fn validate_input_kind_strict(
    indicator: &str,
    expected: IndicatorInputKind,
    data: IndicatorDataRef<'_>,
) -> Result<(), IndicatorDispatchError> {
    let expected = strict_expected_input_kind(indicator, expected);
    if indicator.eq_ignore_ascii_case("mod_god_mode") {
        let matches = matches!(
            data,
            IndicatorDataRef::Candles { .. }
                | IndicatorDataRef::Ohlc { .. }
                | IndicatorDataRef::Ohlcv { .. }
        );
        if matches {
            return Ok(());
        }
    }
    let matches = matches!(
        (expected, data),
        (IndicatorInputKind::Slice, IndicatorDataRef::Slice { .. })
            | (
                IndicatorInputKind::Candles,
                IndicatorDataRef::Candles { .. }
            )
            | (IndicatorInputKind::Ohlc, IndicatorDataRef::Ohlc { .. })
            | (IndicatorInputKind::Ohlcv, IndicatorDataRef::Ohlcv { .. })
            | (
                IndicatorInputKind::HighLow,
                IndicatorDataRef::HighLow { .. }
            )
            | (
                IndicatorInputKind::CloseVolume,
                IndicatorDataRef::CloseVolume { .. }
            )
    );

    if matches {
        Ok(())
    } else {
        Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: expected,
        })
    }
}

fn strict_expected_input_kind(indicator: &str, fallback: IndicatorInputKind) -> IndicatorInputKind {
    if indicator.eq_ignore_ascii_case("ao") {
        return IndicatorInputKind::Slice;
    }
    if indicator.eq_ignore_ascii_case("ttm_trend") {
        return IndicatorInputKind::Candles;
    }
    fallback
}

fn resolve_output_id<'a>(
    info: &'a IndicatorInfo,
    requested: Option<&str>,
) -> Result<&'a str, IndicatorDispatchError> {
    if info.outputs.is_empty() {
        return Err(IndicatorDispatchError::ComputeFailed {
            indicator: info.id.to_string(),
            details: "indicator has no registered outputs".to_string(),
        });
    }

    if info.outputs.len() == 1 {
        let only = info.outputs[0].id;
        if let Some(req) = requested {
            if req == only {
                return Ok(only);
            }
            if !req.eq_ignore_ascii_case(only) {
                return Err(IndicatorDispatchError::UnknownOutput {
                    indicator: info.id.to_string(),
                    output: req.to_string(),
                });
            }
        }
        return Ok(only);
    }

    let req = requested.ok_or_else(|| IndicatorDispatchError::InvalidParam {
        indicator: info.id.to_string(),
        key: "output_id".to_string(),
        reason: "output_id is required for multi-output indicators".to_string(),
    })?;

    if let Some(out) = info.outputs.iter().find(|o| o.id == req) {
        return Ok(out.id);
    }
    info.outputs
        .iter()
        .find(|o| o.id.eq_ignore_ascii_case(req))
        .map(|o| o.id)
        .ok_or_else(|| IndicatorDispatchError::UnknownOutput {
            indicator: info.id.to_string(),
            output: req.to_string(),
        })
}

fn is_moving_average(id: &str) -> bool {
    list_moving_averages()
        .iter()
        .any(|ma| ma.id.eq_ignore_ascii_case(id))
}

fn ma_is_period_based(info: &IndicatorInfo) -> bool {
    info.params
        .iter()
        .any(|p| p.key.eq_ignore_ascii_case("period"))
}

fn compute_ma_batch(
    req: IndicatorBatchRequest<'_>,
    info: &IndicatorInfo,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = ma_data_from_req(info.id, req.data)?;
    let cols = ma_len_from_req(info.id, req.data)?;
    let period_based = ma_is_period_based(info);
    if period_based {
        if let Some(out) = try_compute_ma_batch_fast(req, info, output_id, data.clone(), cols)? {
            return Ok(out);
        }
    }
    let rows = req.combos.len();
    let mut matrix = Vec::with_capacity(rows.saturating_mul(cols));

    for combo in req.combos {
        let period = ma_period_for_combo(info, combo.params)?;
        let mut params = convert_ma_params(combo.params, info.id, output_id)?;
        if info.outputs.len() > 1 && !has_key(combo.params, "output") {
            params.push(MaBatchParamKV {
                key: "output",
                value: MaBatchParamValue::EnumString(output_id),
            });
        }
        let out = ma_batch_with_kernel_and_typed_params(
            info.id,
            data.clone(),
            (period, period, 0),
            req.kernel,
            &params,
        )
        .map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: info.id.to_string(),
            details: e.to_string(),
        })?;
        ensure_len(info.id, cols, out.cols)?;
        let row_values = if out.rows == 1 {
            out.values
        } else {
            reorder_or_take_f64_matrix_by_period(
                info.id,
                &[period],
                &out.periods,
                out.cols,
                out.values,
            )?
        };
        ensure_len(info.id, cols, row_values.len())?;
        matrix.extend_from_slice(&row_values);
    }

    Ok(f64_output(output_id, rows, cols, matrix))
}

fn try_compute_ma_batch_fast(
    req: IndicatorBatchRequest<'_>,
    info: &IndicatorInfo,
    output_id: &str,
    data: MaData<'_>,
    cols: usize,
) -> Result<Option<IndicatorBatchOutput>, IndicatorDispatchError> {
    if req.combos.is_empty() {
        return Ok(Some(f64_output(output_id, 0, cols, Vec::new())));
    }
    if !ma_is_period_based(info) {
        return Ok(None);
    }

    let mut periods = Vec::with_capacity(req.combos.len());
    let mut shared_params: Option<Vec<MaBatchParamKV<'_>>> = None;

    for combo in req.combos {
        periods.push(ma_period_for_combo(info, combo.params)?);
        let mut params = convert_ma_params(combo.params, info.id, output_id)?;
        if info.outputs.len() > 1 && !has_key(combo.params, "output") {
            params.push(MaBatchParamKV {
                key: "output",
                value: MaBatchParamValue::EnumString(output_id),
            });
        }
        match &shared_params {
            None => shared_params = Some(params),
            Some(existing) => {
                if !ma_params_equal(existing, &params) {
                    return Ok(None);
                }
            }
        }
    }

    let Some((start, end, step)) = derive_period_sweep(&periods) else {
        return Ok(None);
    };

    let out = ma_batch_with_kernel_and_typed_params(
        info.id,
        data,
        (start, end, step),
        req.kernel,
        shared_params.as_deref().unwrap_or(&[]),
    )
    .map_err(|e| IndicatorDispatchError::ComputeFailed {
        indicator: info.id.to_string(),
        details: e.to_string(),
    })?;
    ensure_len(info.id, cols, out.cols)?;

    let values = reorder_or_take_f64_matrix_by_period(
        info.id,
        &periods,
        &out.periods,
        out.cols,
        out.values,
    )?;
    Ok(Some(f64_output(output_id, periods.len(), cols, values)))
}

fn ma_params_equal(a: &[MaBatchParamKV<'_>], b: &[MaBatchParamKV<'_>]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (lhs, rhs) in a.iter().zip(b.iter()) {
        if !lhs.key.eq_ignore_ascii_case(rhs.key) {
            return false;
        }
        let same = match (&lhs.value, &rhs.value) {
            (MaBatchParamValue::Int(x), MaBatchParamValue::Int(y)) => x == y,
            (MaBatchParamValue::Float(x), MaBatchParamValue::Float(y)) => x == y,
            (MaBatchParamValue::Bool(x), MaBatchParamValue::Bool(y)) => x == y,
            (MaBatchParamValue::EnumString(x), MaBatchParamValue::EnumString(y)) => {
                x.eq_ignore_ascii_case(y)
            }
            _ => false,
        };
        if !same {
            return false;
        }
    }
    true
}

fn collect_f64(
    indicator: &str,
    output_id: &str,
    combos: &[IndicatorParamSet<'_>],
    cols: usize,
    mut eval: impl FnMut(&[ParamKV<'_>]) -> Result<Vec<f64>, IndicatorDispatchError>,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let rows = combos.len();
    let mut matrix = Vec::with_capacity(rows.saturating_mul(cols));
    for combo in combos {
        let series = eval(combo.params)?;
        ensure_len(indicator, cols, series.len())?;
        matrix.extend_from_slice(&series);
    }
    Ok(f64_output(output_id, rows, cols, matrix))
}

fn collect_bool(
    indicator: &str,
    output_id: &str,
    combos: &[IndicatorParamSet<'_>],
    cols: usize,
    mut eval: impl FnMut(&[ParamKV<'_>]) -> Result<Vec<bool>, IndicatorDispatchError>,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let rows = combos.len();
    let mut matrix = Vec::with_capacity(rows.saturating_mul(cols));
    for combo in combos {
        let series = eval(combo.params)?;
        ensure_len(indicator, cols, series.len())?;
        matrix.extend_from_slice(&series);
    }
    Ok(bool_output(output_id, rows, cols, matrix))
}

fn collect_f64_into_rows(
    indicator: &str,
    output_id: &str,
    combos: &[IndicatorParamSet<'_>],
    cols: usize,
    mut eval_into: impl FnMut(&[ParamKV<'_>], &mut [f64]) -> Result<(), IndicatorDispatchError>,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let rows = combos.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: indicator.to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = vec![f64::NAN; total];
    for (row, combo) in combos.iter().enumerate() {
        let start = row * cols;
        let end = start + cols;
        eval_into(combo.params, &mut matrix[start..end])?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
}

fn to_batch_kernel(kernel: Kernel) -> Kernel {
    match kernel {
        Kernel::Auto => Kernel::Auto,
        Kernel::Scalar => Kernel::ScalarBatch,
        Kernel::Avx2 => Kernel::Avx2Batch,
        Kernel::Avx512 => Kernel::Avx512Batch,
        other => other,
    }
}

fn combo_periods(
    indicator: &str,
    combos: &[IndicatorParamSet<'_>],
    key: &str,
    default: usize,
) -> Result<Vec<usize>, IndicatorDispatchError> {
    let mut out = Vec::with_capacity(combos.len());
    for combo in combos {
        out.push(get_usize_param(indicator, combo.params, key, default)?);
    }
    Ok(out)
}

fn derive_period_sweep(periods: &[usize]) -> Option<(usize, usize, usize)> {
    if periods.is_empty() {
        return None;
    }
    if periods.len() == 1 {
        return Some((periods[0], periods[0], 0));
    }
    if periods.windows(2).all(|w| w[0] == w[1]) {
        return Some((periods[0], periods[0], 0));
    }

    let diff = periods[1] as isize - periods[0] as isize;
    if diff == 0 {
        return None;
    }
    if !periods
        .windows(2)
        .all(|w| (w[1] as isize - w[0] as isize) == diff)
    {
        return None;
    }

    Some((
        periods[0],
        *periods.last().unwrap_or(&periods[0]),
        diff.unsigned_abs(),
    ))
}

fn reorder_or_take_f64_matrix_by_period(
    indicator: &str,
    requested_periods: &[usize],
    produced_periods: &[usize],
    cols: usize,
    values: Vec<f64>,
) -> Result<Vec<f64>, IndicatorDispatchError> {
    ensure_len(
        indicator,
        produced_periods.len().saturating_mul(cols),
        values.len(),
    )?;

    if requested_periods.len() == produced_periods.len() && requested_periods == produced_periods {
        return Ok(values);
    }

    let period_to_row: HashMap<usize, usize> = produced_periods
        .iter()
        .copied()
        .enumerate()
        .map(|(row, period)| (period, row))
        .collect();

    let mut out = Vec::with_capacity(requested_periods.len().saturating_mul(cols));
    for period in requested_periods {
        let row = period_to_row.get(period).copied().ok_or_else(|| {
            IndicatorDispatchError::ComputeFailed {
                indicator: indicator.to_string(),
                details: format!("batch output did not contain requested period {period}"),
            }
        })?;
        let start = row * cols;
        let end = start + cols;
        out.extend_from_slice(&values[start..end]);
    }
    Ok(out)
}

fn compute_ad_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("ad", output_id)?;
    let (high, low, close, volume) = extract_hlcv_input("ad", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("ad", output_id, req.combos, close.len(), |_params| {
        let input = AdInput::from_slices(high, low, close, volume, AdParams::default());
        let out =
            ad_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "ad".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_adosc_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("adosc", output_id)?;
    let (high, low, close, volume) = extract_hlcv_input("adosc", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("adosc", output_id, req.combos, close.len(), |params| {
        let short_period = get_usize_param("adosc", params, "short_period", 3)?;
        let long_period = get_usize_param("adosc", params, "long_period", 10)?;
        let input = AdoscInput::from_slices(
            high,
            low,
            close,
            volume,
            AdoscParams {
                short_period: Some(short_period),
                long_period: Some(long_period),
            },
        );
        let out = adosc_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "adosc".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_ao_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("ao", output_id)?;
    let mut derived_source: Option<Vec<f64>> = None;
    let source: &[f64] = match req.data {
        IndicatorDataRef::Slice { values } => values,
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("hl2"))
        }
        IndicatorDataRef::HighLow { high, low } => {
            ensure_same_len_2("ao", high.len(), low.len())?;
            derived_source = Some(high.iter().zip(low).map(|(h, l)| 0.5 * (h + l)).collect());
            derived_source.as_deref().unwrap_or(high)
        }
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => {
            ensure_same_len_4("ao", open.len(), high.len(), low.len(), close.len())?;
            derived_source = Some(high.iter().zip(low).map(|(h, l)| 0.5 * (h + l)).collect());
            derived_source.as_deref().unwrap_or(close)
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "ao",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            derived_source = Some(high.iter().zip(low).map(|(h, l)| 0.5 * (h + l)).collect());
            derived_source.as_deref().unwrap_or(close)
        }
        IndicatorDataRef::CloseVolume { .. } => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "ao".to_string(),
                input: IndicatorInputKind::HighLow,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows("ao", output_id, req.combos, source.len(), |params, row| {
        let short_period = get_usize_param("ao", params, "short_period", 5)?;
        let long_period = get_usize_param("ao", params, "long_period", 34)?;
        let input = AoInput::from_slice(
            source,
            AoParams {
                short_period: Some(short_period),
                long_period: Some(long_period),
            },
        );
        ao_into_slice(row, &input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "ao".to_string(),
            details: e.to_string(),
        })
    })
}

fn compute_bop_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("bop", output_id)?;
    let (open, high, low, close): (&[f64], &[f64], &[f64], &[f64]) = match req.data {
        IndicatorDataRef::Candles { candles, .. } => (
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => {
            ensure_same_len_4("bop", open.len(), high.len(), low.len(), close.len())?;
            (open, high, low, close)
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "bop",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            (open, high, low, close)
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "bop".to_string(),
                input: IndicatorInputKind::Ohlc,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64("bop", output_id, req.combos, close.len(), |_params| {
        let input = BopInput::from_slices(open, high, low, close, BopParams::default());
        let out =
            bop_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "bop".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_emv_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("emv", output_id)?;
    let (high, low, close, volume) = extract_hlcv_input("emv", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("emv", output_id, req.combos, close.len(), |_params| {
        let input = EmvInput::from_slices(high, low, close, volume);
        let out =
            emv_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "emv".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_efi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("efi", output_id)?;
    let (price, volume) = extract_close_volume_input("efi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("efi", output_id, req.combos, price.len(), |params| {
        let period = get_usize_param("efi", params, "period", 13)?;
        let input = EfiInput::from_slices(
            price,
            volume,
            EfiParams {
                period: Some(period),
            },
        );
        let out =
            efi_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "efi".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_mfi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("mfi", output_id)?;
    let mut derived_typical_price: Option<Vec<f64>> = None;
    let (typical_price, volume): (&[f64], &[f64]) = match req.data {
        IndicatorDataRef::Candles { candles, source } => (
            source_type(candles, source.unwrap_or("hlc3")),
            candles.volume.as_slice(),
        ),
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "mfi",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            derived_typical_price = Some(
                high.iter()
                    .zip(low)
                    .zip(close)
                    .map(|((h, l), c)| (h + l + c) / 3.0)
                    .collect(),
            );
            (derived_typical_price.as_deref().unwrap_or(close), volume)
        }
        IndicatorDataRef::CloseVolume { close, volume } => {
            ensure_same_len_2("mfi", close.len(), volume.len())?;
            (close, volume)
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "mfi".to_string(),
                input: IndicatorInputKind::CloseVolume,
            })
        }
    };

    let periods = combo_periods("mfi", req.combos, "period", 14)?;
    if let Some((start, end, step)) = derive_period_sweep(&periods) {
        let out = mfi_batch_with_kernel(
            typical_price,
            volume,
            &MfiBatchRange {
                period: (start, end, step),
            },
            to_batch_kernel(req.kernel),
        )
        .map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "mfi".to_string(),
            details: e.to_string(),
        })?;
        ensure_len("mfi", typical_price.len(), out.cols)?;
        let produced_periods: Vec<usize> = out
            .combos
            .iter()
            .map(|combo| combo.period.unwrap_or(14))
            .collect();
        let values = reorder_or_take_f64_matrix_by_period(
            "mfi",
            &periods,
            &produced_periods,
            out.cols,
            out.values,
        )?;
        return Ok(f64_output(output_id, periods.len(), out.cols, values));
    }

    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "mfi",
        output_id,
        req.combos,
        typical_price.len(),
        |params, row| {
            let period = get_usize_param("mfi", params, "period", 14)?;
            let input = MfiInput::from_slices(
                typical_price,
                volume,
                MfiParams {
                    period: Some(period),
                },
            );
            mfi_into_slice(row, &input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "mfi".to_string(),
                details: e.to_string(),
            })
        },
    )
}

fn compute_mass_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("mass", output_id)?;
    let (high, low) = extract_high_low_input("mass", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("mass", output_id, req.combos, high.len(), |params| {
        let period = get_usize_param("mass", params, "period", 5)?;
        let input = MassInput::from_slices(
            high,
            low,
            MassParams {
                period: Some(period),
            },
        );
        let out = mass_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "mass".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_kvo_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("kvo", output_id)?;
    let (high, low, close, volume) = extract_hlcv_input("kvo", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("kvo", output_id, req.combos, close.len(), |params| {
        let short_period = get_usize_param("kvo", params, "short_period", 2)?;
        let long_period = get_usize_param("kvo", params, "long_period", 5)?;
        let input = KvoInput::from_slices(
            high,
            low,
            close,
            volume,
            KvoParams {
                short_period: Some(short_period),
                long_period: Some(long_period),
            },
        );
        let out =
            kvo_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "kvo".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_vosc_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("vosc", output_id)?;
    let volume = extract_volume_input("vosc", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("vosc", output_id, req.combos, volume.len(), |params| {
        let short_period = get_usize_param("vosc", params, "short_period", 2)?;
        let long_period = get_usize_param("vosc", params, "long_period", 5)?;
        let input = VoscInput::from_slice(
            volume,
            VoscParams {
                short_period: Some(short_period),
                long_period: Some(long_period),
            },
        );
        let out = vosc_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "vosc".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_dx_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("dx", output_id)?;
    let (high, low, close) = extract_ohlc_input("dx", req.data)?;

    let periods = combo_periods("dx", req.combos, "period", 14)?;
    if let Some((start, end, step)) = derive_period_sweep(&periods) {
        let out = dx_batch_with_kernel(
            high,
            low,
            close,
            &DxBatchRange {
                period: (start, end, step),
            },
            to_batch_kernel(req.kernel),
        )
        .map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "dx".to_string(),
            details: e.to_string(),
        })?;
        ensure_len("dx", close.len(), out.cols)?;
        let produced_periods: Vec<usize> = out
            .combos
            .iter()
            .map(|combo| combo.period.unwrap_or(14))
            .collect();
        let values = reorder_or_take_f64_matrix_by_period(
            "dx",
            &periods,
            &produced_periods,
            out.cols,
            out.values,
        )?;
        return Ok(f64_output(output_id, periods.len(), out.cols, values));
    }

    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows("dx", output_id, req.combos, close.len(), |params, row| {
        let period = get_usize_param("dx", params, "period", 14)?;
        let input = DxInput::from_hlc_slices(
            high,
            low,
            close,
            DxParams {
                period: Some(period),
            },
        );
        dx_into_slice(row, &input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "dx".to_string(),
            details: e.to_string(),
        })
    })
}

fn compute_fosc_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("fosc", output_id)?;
    let data = extract_slice_input("fosc", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("fosc", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("fosc", params, "period", 5)?;
        let input = FoscInput::from_slice(
            data,
            FoscParams {
                period: Some(period),
            },
        );
        let out = fosc_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "fosc".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_ift_rsi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("ift_rsi", output_id)?;
    let data = extract_slice_input("ift_rsi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("ift_rsi", output_id, req.combos, data.len(), |params| {
        let rsi_period = get_usize_param("ift_rsi", params, "rsi_period", 5)?;
        let wma_period = get_usize_param("ift_rsi", params, "wma_period", 9)?;
        let input = IftRsiInput::from_slice(
            data,
            IftRsiParams {
                rsi_period: Some(rsi_period),
                wma_period: Some(wma_period),
            },
        );
        let out = ift_rsi_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "ift_rsi".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_linearreg_angle_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("linearreg_angle", output_id)?;
    let data = extract_slice_input("linearreg_angle", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "linearreg_angle",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let period = get_usize_param("linearreg_angle", params, "period", 14)?;
            let input = Linearreg_angleInput::from_slice(
                data,
                Linearreg_angleParams {
                    period: Some(period),
                },
            );
            let out = linearreg_angle_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "linearreg_angle".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_linearreg_intercept_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("linearreg_intercept", output_id)?;
    let data = extract_slice_input("linearreg_intercept", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "linearreg_intercept",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let period = get_usize_param("linearreg_intercept", params, "period", 14)?;
            let input = LinearRegInterceptInput::from_slice(
                data,
                LinearRegInterceptParams {
                    period: Some(period),
                },
            );
            let out = linearreg_intercept_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "linearreg_intercept".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_linearreg_slope_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("linearreg_slope", output_id)?;
    let data = extract_slice_input("linearreg_slope", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "linearreg_slope",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let period = get_usize_param("linearreg_slope", params, "period", 14)?;
            let input = LinearRegSlopeInput::from_slice(
                data,
                LinearRegSlopeParams {
                    period: Some(period),
                },
            );
            let out = linearreg_slope_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "linearreg_slope".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_cg_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("cg", output_id)?;
    let data = extract_slice_input("cg", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("cg", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("cg", params, "period", 10)?;
        let input = CgInput::from_slice(
            data,
            CgParams {
                period: Some(period),
            },
        );
        let out =
            cg_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "cg".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_rsi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("rsi", output_id)?;
    let data = extract_slice_input("rsi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("rsi", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("rsi", params, "period", 14)?;
        let input = RsiInput::from_slice(
            data,
            RsiParams {
                period: Some(period),
            },
        );
        let out =
            rsi_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "rsi".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_roc_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("roc", output_id)?;
    let data = extract_slice_input("roc", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("roc", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("roc", params, "period", 9)?;
        let input = RocInput::from_slice(
            data,
            RocParams {
                period: Some(period),
            },
        );
        let out =
            roc_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "roc".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_apo_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("apo", output_id)?;
    let data = extract_slice_input("apo", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("apo", output_id, req.combos, data.len(), |params| {
        let short_period = get_usize_param("apo", params, "short_period", 10)?;
        let long_period = get_usize_param("apo", params, "long_period", 20)?;
        let input = ApoInput::from_slice(
            data,
            ApoParams {
                short_period: Some(short_period),
                long_period: Some(long_period),
            },
        );
        let out =
            apo_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "apo".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_cci_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("cci", output_id)?;
    let data = extract_slice_input("cci", req.data, "hlc3")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("cci", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("cci", params, "period", 14)?;
        let input = CciInput::from_slice(
            data,
            CciParams {
                period: Some(period),
            },
        );
        let out =
            cci_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "cci".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_cfo_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("cfo", output_id)?;
    let data = extract_slice_input("cfo", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("cfo", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("cfo", params, "period", 14)?;
        let scalar = get_f64_param("cfo", params, "scalar", 100.0)?;
        let input = CfoInput::from_slice(
            data,
            CfoParams {
                period: Some(period),
                scalar: Some(scalar),
            },
        );
        let out =
            cfo_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "cfo".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_cci_cycle_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("cci_cycle", output_id)?;
    let data = extract_slice_input("cci_cycle", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("cci_cycle", output_id, req.combos, data.len(), |params| {
        let length = get_usize_param("cci_cycle", params, "length", 10)?;
        let factor = get_f64_param("cci_cycle", params, "factor", 0.5)?;
        let input = CciCycleInput::from_slice(
            data,
            CciCycleParams {
                length: Some(length),
                factor: Some(factor),
            },
        );
        let out = cci_cycle_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "cci_cycle".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_lrsi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("lrsi", output_id)?;
    let (high, low) = extract_high_low_input("lrsi", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("lrsi", output_id, req.combos, high.len(), |params| {
        let alpha = get_f64_param("lrsi", params, "alpha", 0.2)?;
        let input = LrsiInput::from_slices(high, low, LrsiParams { alpha: Some(alpha) });
        let out = lrsi_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "lrsi".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_er_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("er", output_id)?;
    let data = extract_slice_input("er", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("er", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("er", params, "period", 5)?;
        let input = ErInput::from_slice(
            data,
            ErParams {
                period: Some(period),
            },
        );
        let out =
            er_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "er".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_kurtosis_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("kurtosis", output_id)?;
    let data = extract_slice_input("kurtosis", req.data, "hl2")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("kurtosis", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("kurtosis", params, "period", 5)?;
        let input = KurtosisInput::from_slice(
            data,
            KurtosisParams {
                period: Some(period),
            },
        );
        let out = kurtosis_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "kurtosis".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_natr_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("natr", output_id)?;
    let (high, low, close) = extract_ohlc_input("natr", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("natr", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("natr", params, "period", 14)?;
        let input = NatrInput::from_slices(
            high,
            low,
            close,
            NatrParams {
                period: Some(period),
            },
        );
        let out = natr_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "natr".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_mean_ad_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("mean_ad", output_id)?;
    let data = extract_slice_input("mean_ad", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("mean_ad", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("mean_ad", params, "period", 5)?;
        let input = MeanAdInput::from_slice(
            data,
            MeanAdParams {
                period: Some(period),
            },
        );
        let out = mean_ad_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "mean_ad".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_medium_ad_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("medium_ad", output_id)?;
    let data = extract_slice_input("medium_ad", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("medium_ad", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("medium_ad", params, "period", 5)?;
        let input = MediumAdInput::from_slice(
            data,
            MediumAdParams {
                period: Some(period),
            },
        );
        let out = medium_ad_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "medium_ad".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_deviation_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("deviation", output_id)?;
    let data = extract_slice_input("deviation", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("deviation", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("deviation", params, "period", 9)?;
        let devtype = get_usize_param("deviation", params, "devtype", 0)?;
        let input = DeviationInput::from_slice(
            data,
            DeviationParams {
                period: Some(period),
                devtype: Some(devtype),
            },
        );
        let out = deviation_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "deviation".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_dpo_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("dpo", output_id)?;
    let data = extract_slice_input("dpo", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("dpo", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("dpo", params, "period", 5)?;
        let input = DpoInput::from_slice(
            data,
            DpoParams {
                period: Some(period),
            },
        );
        let out =
            dpo_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "dpo".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_pfe_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("pfe", output_id)?;
    let data = extract_slice_input("pfe", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("pfe", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("pfe", params, "period", 10)?;
        let smoothing = get_usize_param("pfe", params, "smoothing", 5)?;
        let input = PfeInput::from_slice(
            data,
            PfeParams {
                period: Some(period),
                smoothing: Some(smoothing),
            },
        );
        let out =
            pfe_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "pfe".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_qstick_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("qstick", output_id)?;
    let (open, close) = match req.data {
        IndicatorDataRef::Candles { candles, .. } => {
            (candles.open.as_slice(), candles.close.as_slice())
        }
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => {
            ensure_same_len_4("qstick", open.len(), high.len(), low.len(), close.len())?;
            (open, close)
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "qstick",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            (open, close)
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "qstick".to_string(),
                input: IndicatorInputKind::Ohlc,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64("qstick", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("qstick", params, "period", 5)?;
        let input = QstickInput::from_slices(
            open,
            close,
            QstickParams {
                period: Some(period),
            },
        );
        let out = qstick_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "qstick".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_reverse_rsi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("reverse_rsi", output_id)?;
    let data = extract_slice_input("reverse_rsi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("reverse_rsi", output_id, req.combos, data.len(), |params| {
        let rsi_length = get_usize_param("reverse_rsi", params, "rsi_length", 14)?;
        let rsi_level = get_f64_param("reverse_rsi", params, "rsi_level", 50.0)?;
        let input = ReverseRsiInput::from_slice(
            data,
            ReverseRsiParams {
                rsi_length: Some(rsi_length),
                rsi_level: Some(rsi_level),
            },
        );
        let out = reverse_rsi_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "reverse_rsi".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_percentile_nearest_rank_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("percentile_nearest_rank", output_id)?;
    let data = extract_slice_input("percentile_nearest_rank", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "percentile_nearest_rank",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let length = get_usize_param("percentile_nearest_rank", params, "length", 15)?;
            let percentage = get_f64_param("percentile_nearest_rank", params, "percentage", 50.0)?;
            let input = PercentileNearestRankInput::from_slice(
                data,
                PercentileNearestRankParams {
                    length: Some(length),
                    percentage: Some(percentage),
                },
            );
            let out = percentile_nearest_rank_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "percentile_nearest_rank".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_obv_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("obv", output_id)?;
    let (close, volume) = extract_close_volume_input("obv", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("obv", output_id, req.combos, close.len(), |_params| {
        let input = ObvInput::from_slices(close, volume, ObvParams::default());
        let out =
            obv_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "obv".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_vpt_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("vpt", output_id)?;
    let (close, volume) = extract_close_volume_input("vpt", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("vpt", output_id, req.combos, close.len(), |_params| {
        let input = VptInput::from_slices(close, volume);
        let out =
            vpt_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "vpt".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_nvi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("nvi", output_id)?;
    let (close, volume) = extract_close_volume_input("nvi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("nvi", output_id, req.combos, close.len(), |_params| {
        let input = NviInput::from_slices(close, volume, NviParams::default());
        let out =
            nvi_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "nvi".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_pvi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("pvi", output_id)?;
    let (close, volume) = extract_close_volume_input("pvi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("pvi", output_id, req.combos, close.len(), |params| {
        let initial_value = get_f64_param("pvi", params, "initial_value", 1000.0)?;
        let input = PviInput::from_slices(
            close,
            volume,
            PviParams {
                initial_value: Some(initial_value),
            },
        );
        let out =
            pvi_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "pvi".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_wclprice_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("wclprice", output_id)?;
    let (high, low, close) = extract_ohlc_input("wclprice", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("wclprice", output_id, req.combos, close.len(), |_params| {
        let input = WclpriceInput::from_slices(high, low, close);
        let out = wclprice_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "wclprice".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_ui_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("ui", output_id)?;
    let data = extract_slice_input("ui", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("ui", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("ui", params, "period", 14)?;
        let scalar = get_f64_param("ui", params, "scalar", 100.0)?;
        let input = UiInput::from_slice(
            data,
            UiParams {
                period: Some(period),
                scalar: Some(scalar),
            },
        );
        let out =
            ui_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "ui".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_zscore_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("zscore", output_id)?;
    let data = extract_slice_input("zscore", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("zscore", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("zscore", params, "period", 14)?;
        let ma_type = get_enum_param("zscore", params, "ma_type", "sma")?;
        let nbdev = get_f64_param("zscore", params, "nbdev", 1.0)?;
        let devtype = get_usize_param("zscore", params, "devtype", 0)?;
        let input = ZscoreInput::from_slice(
            data,
            ZscoreParams {
                period: Some(period),
                ma_type: Some(ma_type),
                nbdev: Some(nbdev),
                devtype: Some(devtype),
            },
        );
        let out = zscore_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "zscore".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_medprice_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("medprice", output_id)?;
    let (high, low) = extract_high_low_input("medprice", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("medprice", output_id, req.combos, high.len(), |_params| {
        let input = MedpriceInput::from_slices(high, low, MedpriceParams::default());
        let out = medprice_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "medprice".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_midpoint_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("midpoint", output_id)?;
    let data = extract_slice_input("midpoint", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("midpoint", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("midpoint", params, "period", 14)?;
        let input = MidpointInput::from_slice(
            data,
            MidpointParams {
                period: Some(period),
            },
        );
        let out = midpoint_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "midpoint".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_midprice_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("midprice", output_id)?;
    let (high, low) = extract_high_low_input("midprice", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("midprice", output_id, req.combos, high.len(), |params| {
        let period = get_usize_param("midprice", params, "period", 14)?;
        let input = MidpriceInput::from_slices(
            high,
            low,
            MidpriceParams {
                period: Some(period),
            },
        );
        let out = midprice_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "midprice".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_mom_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("mom", output_id)?;
    let data = extract_slice_input("mom", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("mom", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("mom", params, "period", 10)?;
        let input = MomInput::from_slice(
            data,
            MomParams {
                period: Some(period),
            },
        );
        let out =
            mom_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "mom".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_cmo_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("cmo", output_id)?;
    let data = extract_slice_input("cmo", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("cmo", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("cmo", params, "period", 14)?;
        let input = CmoInput::from_slice(
            data,
            CmoParams {
                period: Some(period),
            },
        );
        let out =
            cmo_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "cmo".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_rocp_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("rocp", output_id)?;
    let data = extract_slice_input("rocp", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("rocp", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("rocp", params, "period", 10)?;
        let input = RocpInput::from_slice(
            data,
            RocpParams {
                period: Some(period),
            },
        );
        let out = rocp_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "rocp".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_rocr_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("rocr", output_id)?;
    let data = extract_slice_input("rocr", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("rocr", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("rocr", params, "period", 10)?;
        let input = RocrInput::from_slice(
            data,
            RocrParams {
                period: Some(period),
            },
        );
        let out = rocr_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "rocr".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_ppo_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("ppo", output_id)?;
    let data = extract_slice_input("ppo", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("ppo", output_id, req.combos, data.len(), |params| {
        let fast_period = get_usize_param("ppo", params, "fast_period", 12)?;
        let slow_period = get_usize_param("ppo", params, "slow_period", 26)?;
        let ma_type = get_enum_param("ppo", params, "ma_type", "sma")?;
        let input = PpoInput::from_slice(
            data,
            PpoParams {
                fast_period: Some(fast_period),
                slow_period: Some(slow_period),
                ma_type: Some(ma_type),
            },
        );
        let out =
            ppo_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "ppo".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_trix_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("trix", output_id)?;
    let data = extract_slice_input("trix", req.data, "close")?;
    let periods = combo_periods("trix", req.combos, "period", 18)?;
    if let Some((start, end, step)) = derive_period_sweep(&periods) {
        let out = trix_batch_with_kernel(
            data,
            &TrixBatchRange {
                period: (start, end, step),
            },
            to_batch_kernel(req.kernel),
        )
        .map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "trix".to_string(),
            details: e.to_string(),
        })?;
        ensure_len("trix", data.len(), out.cols)?;
        let produced_periods: Vec<usize> = out
            .combos
            .iter()
            .map(|combo| combo.period.unwrap_or(18))
            .collect();
        let values = reorder_or_take_f64_matrix_by_period(
            "trix",
            &periods,
            &produced_periods,
            out.cols,
            out.values,
        )?;
        return Ok(f64_output(output_id, periods.len(), out.cols, values));
    }

    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows("trix", output_id, req.combos, data.len(), |params, row| {
        let period = get_usize_param("trix", params, "period", 18)?;
        let input = TrixInput::from_slice(
            data,
            TrixParams {
                period: Some(period),
            },
        );
        trix_into_slice(row, &input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "trix".to_string(),
            details: e.to_string(),
        })
    })
}

fn compute_tsi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("tsi", output_id)?;
    let data = extract_slice_input("tsi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("tsi", output_id, req.combos, data.len(), |params| {
        let long_period = get_usize_param("tsi", params, "long_period", 25)?;
        let short_period = get_usize_param("tsi", params, "short_period", 13)?;
        let input = TsiInput::from_slice(
            data,
            TsiParams {
                long_period: Some(long_period),
                short_period: Some(short_period),
            },
        );
        let out =
            tsi_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "tsi".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_tsf_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("tsf", output_id)?;
    let data = extract_slice_input("tsf", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("tsf", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("tsf", params, "period", 14)?;
        let input = TsfInput::from_slice(
            data,
            TsfParams {
                period: Some(period),
            },
        );
        let out =
            tsf_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "tsf".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_stddev_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("stddev", output_id)?;
    let data = extract_slice_input("stddev", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("stddev", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("stddev", params, "period", 5)?;
        let nbdev = get_f64_param("stddev", params, "nbdev", 1.0)?;
        let input = StdDevInput::from_slice(
            data,
            StdDevParams {
                period: Some(period),
                nbdev: Some(nbdev),
            },
        );
        let out = stddev_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "stddev".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_var_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("var", output_id)?;
    let data = extract_slice_input("var", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("var", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("var", params, "period", 14)?;
        let nbdev = get_f64_param("var", params, "nbdev", 1.0)?;
        let input = VarInput::from_slice(
            data,
            VarParams {
                period: Some(period),
                nbdev: Some(nbdev),
            },
        );
        let out =
            var_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "var".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_willr_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("willr", output_id)?;
    let (high, low, close) = extract_ohlc_input("willr", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("willr", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("willr", params, "period", 14)?;
        let input = WillrInput::from_slices(
            high,
            low,
            close,
            WillrParams {
                period: Some(period),
            },
        );
        let out = willr_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "willr".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_ultosc_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("ultosc", output_id)?;
    let (high, low, close) = extract_ohlc_input("ultosc", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("ultosc", output_id, req.combos, close.len(), |params| {
        let timeperiod1 = get_usize_param("ultosc", params, "timeperiod1", 7)?;
        let timeperiod2 = get_usize_param("ultosc", params, "timeperiod2", 14)?;
        let timeperiod3 = get_usize_param("ultosc", params, "timeperiod3", 28)?;
        let input = UltOscInput::from_slices(
            high,
            low,
            close,
            UltOscParams {
                timeperiod1: Some(timeperiod1),
                timeperiod2: Some(timeperiod2),
                timeperiod3: Some(timeperiod3),
            },
        );
        let out = ultosc_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "ultosc".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_adx_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("adx", output_id)?;
    let (high, low, close) = extract_ohlc_input("adx", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("adx", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("adx", params, "period", 14)?;
        let input = AdxInput::from_slices(
            high,
            low,
            close,
            AdxParams {
                period: Some(period),
            },
        );
        let out =
            adx_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "adx".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_adxr_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("adxr", output_id)?;
    let (high, low, close) = extract_ohlc_input("adxr", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("adxr", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("adxr", params, "period", 14)?;
        let input = AdxrInput::from_slices(
            high,
            low,
            close,
            AdxrParams {
                period: Some(period),
            },
        );
        let out = adxr_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "adxr".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_atr_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("atr", output_id)?;
    let (high, low, close) = extract_ohlc_input("atr", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("atr", output_id, req.combos, close.len(), |params| {
        let length = get_usize_param("atr", params, "length", 14)?;
        let input = AtrInput::from_slices(
            high,
            low,
            close,
            AtrParams {
                length: Some(length),
            },
        );
        let out =
            atr_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "atr".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn compute_macd_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("macd", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("macd", output_id, req.combos, data.len(), |params| {
        let fast_period = get_usize_param("macd", params, "fast_period", 12)?;
        let slow_period = get_usize_param("macd", params, "slow_period", 26)?;
        let signal_period = get_usize_param("macd", params, "signal_period", 9)?;
        let ma_type = get_enum_param("macd", params, "ma_type", "ema")?;
        let input = MacdInput::from_slice(
            data,
            MacdParams {
                fast_period: Some(fast_period),
                slow_period: Some(slow_period),
                signal_period: Some(signal_period),
                ma_type: Some(ma_type),
            },
        );
        let out = macd_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "macd".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("macd") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.macd);
        }
        if output_id.eq_ignore_ascii_case("signal") {
            return Ok(out.signal);
        }
        if output_id.eq_ignore_ascii_case("hist") {
            return Ok(out.hist);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "macd".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_bollinger_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("bollinger_bands", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "bollinger_bands",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let period = get_usize_param("bollinger_bands", params, "period", 20)?;
            let devup = get_f64_param("bollinger_bands", params, "devup", 2.0)?;
            let devdn = get_f64_param("bollinger_bands", params, "devdn", 2.0)?;
            let matype = get_enum_param("bollinger_bands", params, "matype", "sma")?;
            let devtype = get_usize_param("bollinger_bands", params, "devtype", 0)?;
            let input = BollingerBandsInput::from_slice(
                data,
                BollingerBandsParams {
                    period: Some(period),
                    devup: Some(devup),
                    devdn: Some(devdn),
                    matype: Some(matype),
                    devtype: Some(devtype),
                },
            );
            let out = bollinger_bands_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "bollinger_bands".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("upper") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.upper_band);
            }
            if output_id.eq_ignore_ascii_case("middle") {
                return Ok(out.middle_band);
            }
            if output_id.eq_ignore_ascii_case("lower") {
                return Ok(out.lower_band);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "bollinger_bands".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_bbw_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("bollinger_bands_width", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "bollinger_bands_width",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let period = get_usize_param("bollinger_bands_width", params, "period", 20)?;
            let devup = get_f64_param("bollinger_bands_width", params, "devup", 2.0)?;
            let devdn = get_f64_param("bollinger_bands_width", params, "devdn", 2.0)?;
            let matype = get_enum_param("bollinger_bands_width", params, "matype", "sma")?;
            let devtype = get_usize_param("bollinger_bands_width", params, "devtype", 0)?;
            let input = BollingerBandsWidthInput::from_slice(
                data,
                BollingerBandsWidthParams {
                    period: Some(period),
                    devup: Some(devup),
                    devdn: Some(devdn),
                    matype: Some(matype),
                    devtype: Some(devtype),
                },
            );
            let out = bollinger_bands_width_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "bollinger_bands_width".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") || output_id.eq_ignore_ascii_case("values") {
                return Ok(out.values);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "bollinger_bands_width".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_stoch_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("stoch", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("stoch", output_id, req.combos, close.len(), |params| {
        let fastk_period = get_usize_param("stoch", params, "fastk_period", 14)?;
        let slowk_period = get_usize_param("stoch", params, "slowk_period", 3)?;
        let slowd_period = get_usize_param("stoch", params, "slowd_period", 3)?;
        let slowk_ma_type = get_enum_param("stoch", params, "slowk_ma_type", "sma")?;
        let slowd_ma_type = get_enum_param("stoch", params, "slowd_ma_type", "sma")?;
        let input = StochInput::from_slices(
            high,
            low,
            close,
            StochParams {
                fastk_period: Some(fastk_period),
                slowk_period: Some(slowk_period),
                slowk_ma_type: Some(slowk_ma_type),
                slowd_period: Some(slowd_period),
                slowd_ma_type: Some(slowd_ma_type),
            },
        );
        let out = stoch_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "stoch".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("k") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.k);
        }
        if output_id.eq_ignore_ascii_case("d") {
            return Ok(out.d);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "stoch".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_stochf_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("stochf", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("stochf", output_id, req.combos, close.len(), |params| {
        let fastk_period = get_usize_param("stochf", params, "fastk_period", 5)?;
        let fastd_period = get_usize_param("stochf", params, "fastd_period", 3)?;
        let fastd_matype = get_usize_param("stochf", params, "fastd_matype", 0)?;
        let input = StochfInput::from_slices(
            high,
            low,
            close,
            StochfParams {
                fastk_period: Some(fastk_period),
                fastd_period: Some(fastd_period),
                fastd_matype: Some(fastd_matype),
            },
        );
        let out = stochf_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "stochf".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("k") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.k);
        }
        if output_id.eq_ignore_ascii_case("d") {
            return Ok(out.d);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "stochf".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_stochastic_money_flow_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (source, volume) =
        extract_close_volume_input("stochastic_money_flow_index", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "stochastic_money_flow_index",
        output_id,
        req.combos,
        source.len(),
        |params| {
            let stoch_k_length =
                get_usize_param("stochastic_money_flow_index", params, "stoch_k_length", 14)?;
            let stoch_k_smooth =
                get_usize_param("stochastic_money_flow_index", params, "stoch_k_smooth", 3)?;
            let stoch_d_smooth =
                get_usize_param("stochastic_money_flow_index", params, "stoch_d_smooth", 3)?;
            let mfi_length =
                get_usize_param("stochastic_money_flow_index", params, "mfi_length", 14)?;
            let input = StochasticMoneyFlowIndexInput::from_slices(
                source,
                volume,
                StochasticMoneyFlowIndexParams {
                    stoch_k_length: Some(stoch_k_length),
                    stoch_k_smooth: Some(stoch_k_smooth),
                    stoch_d_smooth: Some(stoch_d_smooth),
                    mfi_length: Some(mfi_length),
                },
            );
            let out = stochastic_money_flow_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "stochastic_money_flow_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("k") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.k);
            }
            if output_id.eq_ignore_ascii_case("d") {
                return Ok(out.d);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "stochastic_money_flow_index".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_vwmacd_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (close, volume) = extract_close_volume_input("vwmacd", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("vwmacd", output_id, req.combos, close.len(), |params| {
        let fast_period =
            get_usize_param_with_aliases("vwmacd", params, &["fast", "fast_period"], 12)?;
        let slow_period =
            get_usize_param_with_aliases("vwmacd", params, &["slow", "slow_period"], 26)?;
        let signal_period =
            get_usize_param_with_aliases("vwmacd", params, &["signal", "signal_period"], 9)?;
        let fast_ma_type = get_enum_param("vwmacd", params, "fast_ma_type", "sma")?;
        let slow_ma_type = get_enum_param("vwmacd", params, "slow_ma_type", "sma")?;
        let signal_ma_type = get_enum_param("vwmacd", params, "signal_ma_type", "ema")?;
        let input = VwmacdInput::from_slices(
            close,
            volume,
            VwmacdParams {
                fast_period: Some(fast_period),
                slow_period: Some(slow_period),
                signal_period: Some(signal_period),
                fast_ma_type: Some(fast_ma_type),
                slow_ma_type: Some(slow_ma_type),
                signal_ma_type: Some(signal_ma_type),
            },
        );
        let out = vwmacd_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "vwmacd".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("macd") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.macd);
        }
        if output_id.eq_ignore_ascii_case("signal") {
            return Ok(out.signal);
        }
        if output_id.eq_ignore_ascii_case("hist") {
            return Ok(out.hist);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "vwmacd".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_vpci_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (close, volume) = extract_close_volume_input("vpci", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("vpci", output_id, req.combos, close.len(), |params| {
        let short_range = get_usize_param("vpci", params, "short_range", 5)?;
        let long_range = get_usize_param("vpci", params, "long_range", 25)?;
        let input = VpciInput::from_slices(
            close,
            volume,
            VpciParams {
                short_range: Some(short_range),
                long_range: Some(long_range),
            },
        );
        let out = vpci_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "vpci".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("vpci") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.vpci);
        }
        if output_id.eq_ignore_ascii_case("vpcis") {
            return Ok(out.vpcis);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "vpci".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_ttm_trend_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("ttm_trend", output_id)?;
    let mut derived_source: Option<Vec<f64>> = None;
    let (source, close): (&[f64], &[f64]) = match req.data {
        IndicatorDataRef::Candles { candles, source } => (
            source_type(candles, source.unwrap_or("hl2")),
            candles.close.as_slice(),
        ),
        IndicatorDataRef::Ohlc {
            high, low, close, ..
        } => {
            ensure_same_len_3("ttm_trend", high.len(), low.len(), close.len())?;
            derived_source = Some(high.iter().zip(low).map(|(h, l)| 0.5 * (h + l)).collect());
            (derived_source.as_deref().unwrap_or(close), close)
        }
        IndicatorDataRef::Ohlcv {
            high, low, close, ..
        } => {
            ensure_same_len_3("ttm_trend", high.len(), low.len(), close.len())?;
            derived_source = Some(high.iter().zip(low).map(|(h, l)| 0.5 * (h + l)).collect());
            (derived_source.as_deref().unwrap_or(close), close)
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "ttm_trend".to_string(),
                input: IndicatorInputKind::Ohlc,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_bool("ttm_trend", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("ttm_trend", params, "period", 5)?;
        let input = TtmTrendInput::from_slices(
            source,
            close,
            TtmTrendParams {
                period: Some(period),
            },
        );
        let out = ttm_trend_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "ttm_trend".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_ttm_squeeze_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("ttm_squeeze", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ttm_squeeze",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param("ttm_squeeze", params, "length", 20)?;
            let bb_mult = get_f64_param("ttm_squeeze", params, "bb_mult", 2.0)?;
            let kc_mult_high = get_f64_param_with_aliases(
                "ttm_squeeze",
                params,
                &["kc_high", "kc_mult_high"],
                1.0,
            )?;
            let kc_mult_mid =
                get_f64_param_with_aliases("ttm_squeeze", params, &["kc_mid", "kc_mult_mid"], 1.5)?;
            let kc_mult_low =
                get_f64_param_with_aliases("ttm_squeeze", params, &["kc_low", "kc_mult_low"], 2.0)?;
            let input = TtmSqueezeInput::from_slices(
                high,
                low,
                close,
                TtmSqueezeParams {
                    length: Some(length),
                    bb_mult: Some(bb_mult),
                    kc_mult_high: Some(kc_mult_high),
                    kc_mult_mid: Some(kc_mult_mid),
                    kc_mult_low: Some(kc_mult_low),
                },
            );
            let out = ttm_squeeze_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "ttm_squeeze".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("momentum") || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.momentum);
            }
            if output_id.eq_ignore_ascii_case("squeeze") {
                return Ok(out.squeeze);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "ttm_squeeze".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_aroon_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low) = extract_high_low_input("aroon", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("aroon", output_id, req.combos, high.len(), |params| {
        let length = get_usize_param("aroon", params, "length", 14)?;
        let input = AroonInput::from_slices_hl(
            high,
            low,
            AroonParams {
                length: Some(length),
            },
        );
        let out = aroon_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "aroon".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("up")
            || output_id.eq_ignore_ascii_case("aroon_up")
            || output_id.eq_ignore_ascii_case("value")
        {
            return Ok(out.aroon_up);
        }
        if output_id.eq_ignore_ascii_case("down") || output_id.eq_ignore_ascii_case("aroon_down") {
            return Ok(out.aroon_down);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "aroon".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_aroonosc_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low) = extract_high_low_input("aroonosc", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("aroonosc", output_id, req.combos, high.len(), |params| {
        let length = get_usize_param("aroonosc", params, "length", 14)?;
        let input = AroonOscInput::from_slices_hl(
            high,
            low,
            AroonOscParams {
                length: Some(length),
            },
        );
        let out = aroon_osc_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "aroonosc".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("value") {
            return Ok(out.values);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "aroonosc".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_di_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("di", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("di", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("di", params, "period", 14)?;
        let input = DiInput::from_slices(
            high,
            low,
            close,
            DiParams {
                period: Some(period),
            },
        );
        let out =
            di_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "di".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("plus") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.plus);
        }
        if output_id.eq_ignore_ascii_case("minus") {
            return Ok(out.minus);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "di".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_dm_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low) = extract_high_low_input("dm", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("dm", output_id, req.combos, high.len(), |params| {
        let period = get_usize_param("dm", params, "period", 14)?;
        let input = DmInput::from_slices(
            high,
            low,
            DmParams {
                period: Some(period),
            },
        );
        let out =
            dm_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "dm".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("plus") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.plus);
        }
        if output_id.eq_ignore_ascii_case("minus") {
            return Ok(out.minus);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "dm".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_dti_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("dti", output_id)?;
    let (high, low) = extract_high_low_input("dti", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows("dti", output_id, req.combos, high.len(), |params, row| {
        let r = get_usize_param("dti", params, "r", 14)?;
        let s = get_usize_param("dti", params, "s", 10)?;
        let u = get_usize_param("dti", params, "u", 5)?;
        let input = DtiInput::from_slices(
            high,
            low,
            DtiParams {
                r: Some(r),
                s: Some(s),
                u: Some(u),
            },
        );
        dti_into_slice(row, &input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "dti".to_string(),
            details: e.to_string(),
        })
    })
}

fn compute_donchian_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low) = extract_high_low_input("donchian", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("donchian", output_id, req.combos, high.len(), |params| {
        let period = get_usize_param("donchian", params, "period", 20)?;
        let input = DonchianInput::from_slices(
            high,
            low,
            DonchianParams {
                period: Some(period),
            },
        );
        let out = donchian_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "donchian".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("upper") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.upperband);
        }
        if output_id.eq_ignore_ascii_case("middle") {
            return Ok(out.middleband);
        }
        if output_id.eq_ignore_ascii_case("lower") {
            return Ok(out.lowerband);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "donchian".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_kdj_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("kdj", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("kdj", output_id, req.combos, close.len(), |params| {
        let fast_k_period = get_usize_param("kdj", params, "fast_k_period", 9)?;
        let slow_k_period = get_usize_param("kdj", params, "slow_k_period", 3)?;
        let slow_k_ma_type = get_enum_param("kdj", params, "slow_k_ma_type", "sma")?;
        let slow_d_period = get_usize_param("kdj", params, "slow_d_period", 3)?;
        let slow_d_ma_type = get_enum_param("kdj", params, "slow_d_ma_type", "sma")?;
        let input = KdjInput::from_slices(
            high,
            low,
            close,
            KdjParams {
                fast_k_period: Some(fast_k_period),
                slow_k_period: Some(slow_k_period),
                slow_k_ma_type: Some(slow_k_ma_type),
                slow_d_period: Some(slow_d_period),
                slow_d_ma_type: Some(slow_d_ma_type),
            },
        );
        let out =
            kdj_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "kdj".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("k") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.k);
        }
        if output_id.eq_ignore_ascii_case("d") {
            return Ok(out.d);
        }
        if output_id.eq_ignore_ascii_case("j") {
            return Ok(out.j);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "kdj".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_keltner_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("keltner", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("keltner", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("keltner", params, "period", 20)?;
        let multiplier = get_f64_param("keltner", params, "multiplier", 2.0)?;
        let ma_type = get_enum_param("keltner", params, "ma_type", "ema")?;
        let input = KeltnerInput::from_slice(
            high,
            low,
            close,
            close,
            KeltnerParams {
                period: Some(period),
                multiplier: Some(multiplier),
                ma_type: Some(ma_type),
            },
        );
        let out = keltner_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "keltner".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("upper") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.upper_band);
        }
        if output_id.eq_ignore_ascii_case("middle") {
            return Ok(out.middle_band);
        }
        if output_id.eq_ignore_ascii_case("lower") {
            return Ok(out.lower_band);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "keltner".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_squeeze_momentum_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("squeeze_momentum", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "squeeze_momentum",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length_bb = get_usize_param("squeeze_momentum", params, "length_bb", 20)?;
            let mult_bb = get_f64_param("squeeze_momentum", params, "mult_bb", 2.0)?;
            let length_kc = get_usize_param("squeeze_momentum", params, "length_kc", 20)?;
            let mult_kc = get_f64_param("squeeze_momentum", params, "mult_kc", 1.5)?;
            let input = SqueezeMomentumInput::from_slices(
                high,
                low,
                close,
                SqueezeMomentumParams {
                    length_bb: Some(length_bb),
                    mult_bb: Some(mult_bb),
                    length_kc: Some(length_kc),
                    mult_kc: Some(mult_kc),
                },
            );
            let out = squeeze_momentum_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "squeeze_momentum".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("momentum") || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.momentum);
            }
            if output_id.eq_ignore_ascii_case("squeeze") {
                return Ok(out.squeeze);
            }
            if output_id.eq_ignore_ascii_case("signal")
                || output_id.eq_ignore_ascii_case("momentum_signal")
            {
                return Ok(out.momentum_signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "squeeze_momentum".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_srsi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("srsi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("srsi", output_id, req.combos, data.len(), |params| {
        let rsi_period = get_usize_param("srsi", params, "rsi_period", 14)?;
        let stoch_period = get_usize_param("srsi", params, "stoch_period", 14)?;
        let k = get_usize_param("srsi", params, "k", 3)?;
        let d = get_usize_param("srsi", params, "d", 3)?;
        let source = get_enum_param("srsi", params, "source", "close")?;
        let input = SrsiInput::from_slice(
            data,
            SrsiParams {
                rsi_period: Some(rsi_period),
                stoch_period: Some(stoch_period),
                k: Some(k),
                d: Some(d),
                source: Some(source),
            },
        );
        let out = srsi_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "srsi".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("k") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.k);
        }
        if output_id.eq_ignore_ascii_case("d") {
            return Ok(out.d);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "srsi".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_supertrend_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("supertrend", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("supertrend", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("supertrend", params, "period", 10)?;
        let factor = get_f64_param("supertrend", params, "factor", 3.0)?;
        let input = SuperTrendInput::from_slices(
            high,
            low,
            close,
            SuperTrendParams {
                period: Some(period),
                factor: Some(factor),
            },
        );
        let out = supertrend_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "supertrend".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("trend") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.trend);
        }
        if output_id.eq_ignore_ascii_case("changed") {
            return Ok(out.changed);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "supertrend".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_vi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("vi", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("vi", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("vi", params, "period", 14)?;
        let input = ViInput::from_slices(
            high,
            low,
            close,
            ViParams {
                period: Some(period),
            },
        );
        let out =
            vi_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "vi".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("plus") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.plus);
        }
        if output_id.eq_ignore_ascii_case("minus") {
            return Ok(out.minus);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "vi".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_wavetrend_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("wavetrend", req.data, "hlc3")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("wavetrend", output_id, req.combos, data.len(), |params| {
        let channel_length = get_usize_param("wavetrend", params, "channel_length", 9)?;
        let average_length = get_usize_param("wavetrend", params, "average_length", 12)?;
        let ma_length = get_usize_param("wavetrend", params, "ma_length", 3)?;
        let factor = get_f64_param("wavetrend", params, "factor", 0.015)?;
        let input = WavetrendInput::from_slice(
            data,
            WavetrendParams {
                channel_length: Some(channel_length),
                average_length: Some(average_length),
                ma_length: Some(ma_length),
                factor: Some(factor),
            },
        );
        let out = wavetrend_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "wavetrend".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("wt1") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.wt1);
        }
        if output_id.eq_ignore_ascii_case("wt2") {
            return Ok(out.wt2);
        }
        if output_id.eq_ignore_ascii_case("wt_diff") {
            return Ok(out.wt_diff);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "wavetrend".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_wto_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("wto", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("wto", output_id, req.combos, data.len(), |params| {
        let channel_length = get_usize_param("wto", params, "channel_length", 10)?;
        let average_length = get_usize_param("wto", params, "average_length", 21)?;
        let input = WtoInput::from_slice(
            data,
            WtoParams {
                channel_length: Some(channel_length),
                average_length: Some(average_length),
            },
        );
        let out =
            wto_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "wto".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("wavetrend1")
            || output_id.eq_ignore_ascii_case("wt1")
            || output_id.eq_ignore_ascii_case("value")
        {
            return Ok(out.wavetrend1);
        }
        if output_id.eq_ignore_ascii_case("wavetrend2") || output_id.eq_ignore_ascii_case("wt2") {
            return Ok(out.wavetrend2);
        }
        if output_id.eq_ignore_ascii_case("histogram") || output_id.eq_ignore_ascii_case("hist") {
            return Ok(out.histogram);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "wto".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_rogers_satchell_volatility_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("rogers_satchell_volatility", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "rogers_satchell_volatility",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let lookback = get_usize_param("rogers_satchell_volatility", params, "lookback", 8)?;
            let signal_length =
                get_usize_param("rogers_satchell_volatility", params, "signal_length", 8)?;
            let input = RogersSatchellVolatilityInput::from_slices(
                open,
                high,
                low,
                close,
                RogersSatchellVolatilityParams {
                    lookback: Some(lookback),
                    signal_length: Some(signal_length),
                },
            );
            let out = rogers_satchell_volatility_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "rogers_satchell_volatility".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("rs") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.rs);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "rogers_satchell_volatility".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_historical_volatility_rank_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("historical_volatility_rank", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "historical_volatility_rank",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let hv_length = get_usize_param("historical_volatility_rank", params, "hv_length", 10)?;
            let rank_length =
                get_usize_param("historical_volatility_rank", params, "rank_length", 52 * 7)?;
            let annualization_days = get_f64_param(
                "historical_volatility_rank",
                params,
                "annualization_days",
                365.0,
            )?;
            let bar_days = get_f64_param("historical_volatility_rank", params, "bar_days", 1.0)?;
            let input = HistoricalVolatilityRankInput::from_slice(
                data,
                HistoricalVolatilityRankParams {
                    hv_length: Some(hv_length),
                    rank_length: Some(rank_length),
                    annualization_days: Some(annualization_days),
                    bar_days: Some(bar_days),
                },
            );
            let out = historical_volatility_rank_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "historical_volatility_rank".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("hvr") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.hvr);
            }
            if output_id.eq_ignore_ascii_case("hv") {
                return Ok(out.hv);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "historical_volatility_rank".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_dual_ulcer_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("dual_ulcer_index", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "dual_ulcer_index",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let period = get_usize_param("dual_ulcer_index", params, "period", 5)?;
            let auto_threshold =
                get_bool_param("dual_ulcer_index", params, "auto_threshold", true)?;
            let threshold = get_f64_param("dual_ulcer_index", params, "threshold", 0.1)?;
            let input = DualUlcerIndexInput::from_slice(
                data,
                DualUlcerIndexParams {
                    period: Some(period),
                    auto_threshold: Some(auto_threshold),
                    threshold: Some(threshold),
                },
            );
            let out = dual_ulcer_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "dual_ulcer_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("long_ulcer")
                || output_id.eq_ignore_ascii_case("uulcer")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.long_ulcer);
            }
            if output_id.eq_ignore_ascii_case("short_ulcer")
                || output_id.eq_ignore_ascii_case("dulcer")
            {
                return Ok(out.short_ulcer);
            }
            if output_id.eq_ignore_ascii_case("threshold") {
                return Ok(out.threshold);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "dual_ulcer_index".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_fractal_dimension_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("fractal_dimension_index", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "fractal_dimension_index",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let length = get_usize_param("fractal_dimension_index", params, "length", 30)?;
            let input = FractalDimensionIndexInput::from_slice(
                data,
                FractalDimensionIndexParams {
                    length: Some(length),
                },
            );
            let out = fractal_dimension_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "fractal_dimension_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.values);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "fractal_dimension_index".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_volume_weighted_rsi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("volume_weighted_rsi", output_id)?;
    let (close, volume) = extract_close_volume_input("volume_weighted_rsi", req.data, "close")?;
    let periods = combo_periods("volume_weighted_rsi", req.combos, "period", 14)?;
    if let Some((start, end, step)) = derive_period_sweep(&periods) {
        let out = volume_weighted_rsi_batch_with_kernel(
            close,
            volume,
            &VolumeWeightedRsiBatchRange {
                period: (start, end, step),
            },
            to_batch_kernel(req.kernel),
        )
        .map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "volume_weighted_rsi".to_string(),
            details: e.to_string(),
        })?;
        ensure_len("volume_weighted_rsi", close.len(), out.cols)?;
        let produced_periods: Vec<usize> = out
            .combos
            .iter()
            .map(|combo| combo.period.unwrap_or(14))
            .collect();
        let values = reorder_or_take_f64_matrix_by_period(
            "volume_weighted_rsi",
            &periods,
            &produced_periods,
            out.cols,
            out.values,
        )?;
        return Ok(f64_output(output_id, periods.len(), out.cols, values));
    }

    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "volume_weighted_rsi",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
            let period = get_usize_param("volume_weighted_rsi", params, "period", 14)?;
            let input = VolumeWeightedRsiInput::from_slices(
                close,
                volume,
                VolumeWeightedRsiParams {
                    period: Some(period),
                },
            );
            volume_weighted_rsi_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "volume_weighted_rsi".to_string(),
                    details: e.to_string(),
                }
            })
        },
    )
}

fn compute_dynamic_momentum_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("dynamic_momentum_index", output_id)?;
    let data = extract_slice_input("dynamic_momentum_index", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "dynamic_momentum_index",
        output_id,
        req.combos,
        data.len(),
        |params, row| {
            let rsi_period = get_usize_param("dynamic_momentum_index", params, "rsi_period", 14)?;
            let volatility_period =
                get_usize_param("dynamic_momentum_index", params, "volatility_period", 5)?;
            let volatility_sma_period = get_usize_param(
                "dynamic_momentum_index",
                params,
                "volatility_sma_period",
                10,
            )?;
            let upper_limit = get_usize_param("dynamic_momentum_index", params, "upper_limit", 30)?;
            let lower_limit = get_usize_param("dynamic_momentum_index", params, "lower_limit", 5)?;
            let input = DynamicMomentumIndexInput::from_slice(
                data,
                DynamicMomentumIndexParams {
                    rsi_period: Some(rsi_period),
                    volatility_period: Some(volatility_period),
                    volatility_sma_period: Some(volatility_sma_period),
                    upper_limit: Some(upper_limit),
                    lower_limit: Some(lower_limit),
                },
            );
            dynamic_momentum_index_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "dynamic_momentum_index".to_string(),
                    details: e.to_string(),
                }
            })
        },
    )
}

fn compute_disparity_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("disparity_index", output_id)?;
    let data = extract_slice_input("disparity_index", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "disparity_index",
        output_id,
        req.combos,
        data.len(),
        |params, row| {
            let ema_period = get_usize_param("disparity_index", params, "ema_period", 14)?;
            let lookback_period =
                get_usize_param("disparity_index", params, "lookback_period", 14)?;
            let smoothing_period =
                get_usize_param("disparity_index", params, "smoothing_period", 9)?;
            let smoothing_type =
                get_enum_param("disparity_index", params, "smoothing_type", "ema")?;
            let input = DisparityIndexInput::from_slice(
                data,
                DisparityIndexParams {
                    ema_period: Some(ema_period),
                    lookback_period: Some(lookback_period),
                    smoothing_period: Some(smoothing_period),
                    smoothing_type: Some(smoothing_type),
                },
            );
            disparity_index_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "disparity_index".to_string(),
                    details: e.to_string(),
                }
            })
        },
    )
}

fn compute_donchian_channel_width_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("donchian_channel_width", output_id)?;
    let (high, low) = extract_high_low_input("donchian_channel_width", req.data)?;

    collect_f64_into_rows(
        "donchian_channel_width",
        output_id,
        req.combos,
        high.len(),
        |params, row| {
            let period = get_usize_param("donchian_channel_width", params, "period", 20)?;
            let kernel = req.kernel;
            let input = DonchianChannelWidthInput::from_slices(
                high,
                low,
                DonchianChannelWidthParams {
                    period: Some(period),
                },
            );
            donchian_channel_width_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "donchian_channel_width".to_string(),
                    details: e.to_string(),
                }
            })
        },
    )
}

fn compute_kairi_relative_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("kairi_relative_index", output_id)?;
    let kernel = req.kernel.to_non_batch();
    let len = match req.data {
        IndicatorDataRef::Slice { values } => values.len(),
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        IndicatorDataRef::CloseVolume { close, volume } => {
            ensure_same_len_2("kairi_relative_index", close.len(), volume.len())?;
            close.len()
        }
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => {
            ensure_same_len_4(
                "kairi_relative_index",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
            )?;
            close.len()
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "kairi_relative_index",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            close.len()
        }
        IndicatorDataRef::HighLow { .. } => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "kairi_relative_index".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };

    collect_f64_into_rows(
        "kairi_relative_index",
        output_id,
        req.combos,
        len,
        |params, row| {
            let length = get_usize_param("kairi_relative_index", params, "length", 50)?;
            let ma_type = get_enum_param("kairi_relative_index", params, "ma_type", "SMA")?;
            if ma_type.eq_ignore_ascii_case("VWMA") {
                match req.data {
                    IndicatorDataRef::Slice { .. } | IndicatorDataRef::Ohlc { .. } => {
                        return Err(IndicatorDispatchError::MissingRequiredInput {
                            indicator: "kairi_relative_index".to_string(),
                            input: IndicatorInputKind::CloseVolume,
                        });
                    }
                    _ => {}
                }
            }

            let input = match req.data {
                IndicatorDataRef::Slice { values } => KairiRelativeIndexInput::from_slices(
                    values,
                    values,
                    KairiRelativeIndexParams {
                        length: Some(length),
                        ma_type: Some(ma_type.to_string()),
                    },
                ),
                IndicatorDataRef::Candles { candles, source } => {
                    KairiRelativeIndexInput::from_candles(
                        candles,
                        source.unwrap_or("close"),
                        KairiRelativeIndexParams {
                            length: Some(length),
                            ma_type: Some(ma_type.to_string()),
                        },
                    )
                }
                IndicatorDataRef::CloseVolume { close, volume } => {
                    KairiRelativeIndexInput::from_slices(
                        close,
                        volume,
                        KairiRelativeIndexParams {
                            length: Some(length),
                            ma_type: Some(ma_type.to_string()),
                        },
                    )
                }
                IndicatorDataRef::Ohlc { close, .. } => KairiRelativeIndexInput::from_slices(
                    close,
                    close,
                    KairiRelativeIndexParams {
                        length: Some(length),
                        ma_type: Some(ma_type.to_string()),
                    },
                ),
                IndicatorDataRef::Ohlcv { close, volume, .. } => {
                    KairiRelativeIndexInput::from_slices(
                        close,
                        volume,
                        KairiRelativeIndexParams {
                            length: Some(length),
                            ma_type: Some(ma_type.to_string()),
                        },
                    )
                }
                IndicatorDataRef::HighLow { .. } => unreachable!(),
            };

            kairi_relative_index_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "kairi_relative_index".to_string(),
                    details: e.to_string(),
                }
            })
        },
    )
}

fn compute_projection_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("projection_oscillator", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "projection_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param("projection_oscillator", params, "length", 14)?;
            let smooth_length =
                get_usize_param("projection_oscillator", params, "smooth_length", 4)?;
            let input = ProjectionOscillatorInput::from_slices(
                high,
                low,
                close,
                ProjectionOscillatorParams {
                    length: Some(length),
                    smooth_length: Some(smooth_length),
                },
            );
            let out = projection_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "projection_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("pbo") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.pbo);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "projection_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_market_structure_trailing_stop_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) =
        extract_ohlc_full_input("market_structure_trailing_stop", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "market_structure_trailing_stop",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param("market_structure_trailing_stop", params, "length", 14)?;
            let increment_factor = get_f64_param(
                "market_structure_trailing_stop",
                params,
                "increment_factor",
                100.0,
            )?;
            let reset_on = get_enum_param(
                "market_structure_trailing_stop",
                params,
                "reset_on",
                "CHoCH",
            )?;
            let input = MarketStructureTrailingStopInput::from_slices(
                open,
                high,
                low,
                close,
                MarketStructureTrailingStopParams {
                    length: Some(length),
                    increment_factor: Some(increment_factor),
                    reset_on: Some(reset_on),
                },
            );
            let out = market_structure_trailing_stop_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "market_structure_trailing_stop".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("trailing_stop")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.trailing_stop);
            }
            if output_id.eq_ignore_ascii_case("state") {
                return Ok(out.state);
            }
            if output_id.eq_ignore_ascii_case("structure") {
                return Ok(out.structure);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "market_structure_trailing_stop".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_evasive_supertrend_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("evasive_supertrend", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "evasive_supertrend",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let atr_length = get_usize_param("evasive_supertrend", params, "atr_length", 10)?;
            let base_multiplier =
                get_f64_param("evasive_supertrend", params, "base_multiplier", 3.0)?;
            let noise_threshold =
                get_f64_param("evasive_supertrend", params, "noise_threshold", 1.0)?;
            let expansion_alpha =
                get_f64_param("evasive_supertrend", params, "expansion_alpha", 0.5)?;
            let input = EvasiveSuperTrendInput::from_slices(
                open,
                high,
                low,
                close,
                EvasiveSuperTrendParams {
                    atr_length: Some(atr_length),
                    base_multiplier: Some(base_multiplier),
                    noise_threshold: Some(noise_threshold),
                    expansion_alpha: Some(expansion_alpha),
                },
            );
            let out = evasive_supertrend_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "evasive_supertrend".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("band") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.band);
            }
            if output_id.eq_ignore_ascii_case("state") {
                return Ok(out.state);
            }
            if output_id.eq_ignore_ascii_case("noisy") {
                return Ok(out.noisy);
            }
            if output_id.eq_ignore_ascii_case("changed") {
                return Ok(out.changed);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "evasive_supertrend".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_reversal_signals_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close, volume) =
        extract_ohlcv_full_input("reversal_signals", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("reversal_signals", output_id, req.combos, close.len(), |params| {
        let lookback_period =
            get_usize_param("reversal_signals", params, "lookback_period", 12)?;
        let confirmation_period =
            get_usize_param("reversal_signals", params, "confirmation_period", 3)?;
        let use_volume_confirmation = get_bool_param(
            "reversal_signals",
            params,
            "use_volume_confirmation",
            true,
        )?;
        let trend_ma_period =
            get_usize_param("reversal_signals", params, "trend_ma_period", 50)?;
        let trend_ma_type =
            get_enum_param("reversal_signals", params, "trend_ma_type", "EMA")?;
        let ma_step_period =
            get_usize_param("reversal_signals", params, "ma_step_period", 33)?;
        let input = ReversalSignalsInput::from_slices(
            open,
            high,
            low,
            close,
            volume,
            ReversalSignalsParams {
                lookback_period: Some(lookback_period),
                confirmation_period: Some(confirmation_period),
                use_volume_confirmation: Some(use_volume_confirmation),
                trend_ma_period: Some(trend_ma_period),
                trend_ma_type: Some(trend_ma_type.to_string()),
                ma_step_period: Some(ma_step_period),
            },
        );
        let out = reversal_signals_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "reversal_signals".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("buy_signal") {
            return Ok(out.buy_signal);
        }
        if output_id.eq_ignore_ascii_case("sell_signal") {
            return Ok(out.sell_signal);
        }
        if output_id.eq_ignore_ascii_case("stepped_ma") || output_id.eq_ignore_ascii_case("value")
        {
            return Ok(out.stepped_ma);
        }
        if output_id.eq_ignore_ascii_case("state") {
            return Ok(out.state);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "reversal_signals".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_zig_zag_channels_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("zig_zag_channels", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "zig_zag_channels",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param("zig_zag_channels", params, "length", 100)?;
            let extend = get_bool_param("zig_zag_channels", params, "extend", true)?;
            let input = ZigZagChannelsInput::from_slices(
                open,
                high,
                low,
                close,
                ZigZagChannelsParams {
                    length: Some(length),
                    extend: Some(extend),
                },
            );
            let out = zig_zag_channels_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "zig_zag_channels".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("middle") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.middle);
            }
            if output_id.eq_ignore_ascii_case("upper") {
                return Ok(out.upper);
            }
            if output_id.eq_ignore_ascii_case("lower") {
                return Ok(out.lower);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "zig_zag_channels".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_directional_imbalance_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low) = match req.data {
        IndicatorDataRef::Candles { candles, .. } => {
            (candles.high.as_slice(), candles.low.as_slice())
        }
        IndicatorDataRef::HighLow { high, low } => (high, low),
        IndicatorDataRef::Ohlc { high, low, .. } => (high, low),
        IndicatorDataRef::Ohlcv { high, low, .. } => (high, low),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "directional_imbalance_index".to_string(),
                input: IndicatorInputKind::HighLow,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "directional_imbalance_index",
        output_id,
        req.combos,
        high.len(),
        |params| {
            let length = get_usize_param("directional_imbalance_index", params, "length", 10)?;
            let period = get_usize_param("directional_imbalance_index", params, "period", 70)?;
            let input = DirectionalImbalanceIndexInput::from_slices(
                high,
                low,
                DirectionalImbalanceIndexParams {
                    length: Some(length),
                    period: Some(period),
                },
            );
            let out = directional_imbalance_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "directional_imbalance_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("up") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.up);
            }
            if output_id.eq_ignore_ascii_case("down") {
                return Ok(out.down);
            }
            if output_id.eq_ignore_ascii_case("bulls") {
                return Ok(out.bulls);
            }
            if output_id.eq_ignore_ascii_case("bears") {
                return Ok(out.bears);
            }
            if output_id.eq_ignore_ascii_case("upper") {
                return Ok(out.upper);
            }
            if output_id.eq_ignore_ascii_case("lower") {
                return Ok(out.lower);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "directional_imbalance_index".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_candle_strength_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = match req.data {
        IndicatorDataRef::Candles { candles, .. } => (
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => (open, high, low, close),
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            ..
        } => (open, high, low, close),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "candle_strength_oscillator".to_string(),
                input: IndicatorInputKind::Ohlc,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "candle_strength_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let period = get_usize_param("candle_strength_oscillator", params, "period", 50)?;
            let atr_enabled =
                get_bool_param("candle_strength_oscillator", params, "atr_enabled", false)?;
            let atr_length =
                get_usize_param("candle_strength_oscillator", params, "atr_length", 50)?;
            let mode = get_enum_param("candle_strength_oscillator", params, "mode", "bollinger")?;
            let input = CandleStrengthOscillatorInput::from_slices(
                open,
                high,
                low,
                close,
                CandleStrengthOscillatorParams {
                    period: Some(period),
                    atr_enabled: Some(atr_enabled),
                    atr_length: Some(atr_length),
                    mode: Some(mode.to_string()),
                },
            );
            let out = candle_strength_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "candle_strength_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("strength") || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.strength);
            }
            if output_id.eq_ignore_ascii_case("highs") {
                return Ok(out.highs);
            }
            if output_id.eq_ignore_ascii_case("lows") {
                return Ok(out.lows);
            }
            if output_id.eq_ignore_ascii_case("mid") {
                return Ok(out.mid);
            }
            if output_id.eq_ignore_ascii_case("long_signal") {
                return Ok(out.long_signal);
            }
            if output_id.eq_ignore_ascii_case("short_signal") {
                return Ok(out.short_signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "candle_strength_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_gmma_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let kernel = req.kernel.to_non_batch();
    let owned_source;
    let data = match req.data {
        IndicatorDataRef::Slice { values } => values,
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close"))
        }
        IndicatorDataRef::Ohlc { close, .. } => close,
        IndicatorDataRef::Ohlcv { close, .. } => close,
        IndicatorDataRef::CloseVolume { close, volume } => {
            ensure_same_len_2("gmma_oscillator", close.len(), volume.len())?;
            close
        }
        IndicatorDataRef::HighLow { high, low } => {
            ensure_same_len_2("gmma_oscillator", high.len(), low.len())?;
            owned_source = high
                .iter()
                .zip(low.iter())
                .map(|(&h, &l)| (h + l) * 0.5)
                .collect::<Vec<_>>();
            owned_source.as_slice()
        }
    };

    collect_f64(
        "gmma_oscillator",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let gmma_type = get_enum_param("gmma_oscillator", params, "gmma_type", "guppy")?;
            let smooth_length = get_usize_param("gmma_oscillator", params, "smooth_length", 1)?;
            let signal_length = get_usize_param("gmma_oscillator", params, "signal_length", 13)?;
            let anchor_minutes = get_usize_param("gmma_oscillator", params, "anchor_minutes", 0)?;
            let interval_minutes = if params
                .iter()
                .any(|param| param.key.eq_ignore_ascii_case("interval_minutes"))
            {
                Some(get_usize_param(
                    "gmma_oscillator",
                    params,
                    "interval_minutes",
                    1,
                )?)
            } else {
                None
            };
            let input = GmmaOscillatorInput::from_slice(
                data,
                GmmaOscillatorParams {
                    gmma_type: Some(gmma_type.to_string()),
                    smooth_length: Some(smooth_length),
                    signal_length: Some(signal_length),
                    anchor_minutes: Some(anchor_minutes),
                    interval_minutes,
                },
            );
            let out = gmma_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "gmma_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("oscillator")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.oscillator);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "gmma_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_nonlinear_regression_zero_lag_moving_average_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input(
        "nonlinear_regression_zero_lag_moving_average",
        req.data,
        "close",
    )?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "nonlinear_regression_zero_lag_moving_average",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let zlma_period = get_usize_param(
                "nonlinear_regression_zero_lag_moving_average",
                params,
                "zlma_period",
                15,
            )?;
            let regression_period = get_usize_param(
                "nonlinear_regression_zero_lag_moving_average",
                params,
                "regression_period",
                15,
            )?;
            let input = NonlinearRegressionZeroLagMovingAverageInput::from_slice(
                data,
                NonlinearRegressionZeroLagMovingAverageParams {
                    zlma_period: Some(zlma_period),
                    regression_period: Some(regression_period),
                },
            );
            let out = nonlinear_regression_zero_lag_moving_average_with_kernel(&input, kernel)
                .map_err(|e| IndicatorDispatchError::ComputeFailed {
                    indicator: "nonlinear_regression_zero_lag_moving_average".to_string(),
                    details: e.to_string(),
                })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.value);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            if output_id.eq_ignore_ascii_case("long_signal") {
                return Ok(out.long_signal);
            }
            if output_id.eq_ignore_ascii_case("short_signal") {
                return Ok(out.short_signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "nonlinear_regression_zero_lag_moving_average".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_possible_rsi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("possible_rsi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "possible_rsi",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let period = get_usize_param("possible_rsi", params, "period", 32)?;
            let rsi_mode = get_enum_param("possible_rsi", params, "rsi_mode", "regular")?;
            let norm_period = get_usize_param("possible_rsi", params, "norm_period", 100)?;
            let normalization_mode = get_enum_param(
                "possible_rsi",
                params,
                "normalization_mode",
                "gaussian_fisher",
            )?;
            let normalization_length =
                get_usize_param("possible_rsi", params, "normalization_length", 15)?;
            let nonlag_period = get_usize_param("possible_rsi", params, "nonlag_period", 15)?;
            let dynamic_zone_period =
                get_usize_param("possible_rsi", params, "dynamic_zone_period", 20)?;
            let buy_probability = get_f64_param("possible_rsi", params, "buy_probability", 0.2)?;
            let sell_probability = get_f64_param("possible_rsi", params, "sell_probability", 0.2)?;
            let signal_type =
                get_enum_param("possible_rsi", params, "signal_type", "zeroline_crossover")?;
            let run_highpass = get_bool_param("possible_rsi", params, "run_highpass", false)?;
            let highpass_period = get_usize_param("possible_rsi", params, "highpass_period", 15)?;
            let input = PossibleRsiInput::from_slice(
                data,
                PossibleRsiParams {
                    period: Some(period),
                    rsi_mode: Some(rsi_mode.to_string()),
                    norm_period: Some(norm_period),
                    normalization_mode: Some(normalization_mode.to_string()),
                    normalization_length: Some(normalization_length),
                    nonlag_period: Some(nonlag_period),
                    dynamic_zone_period: Some(dynamic_zone_period),
                    buy_probability: Some(buy_probability),
                    sell_probability: Some(sell_probability),
                    signal_type: Some(signal_type.to_string()),
                    run_highpass: Some(run_highpass),
                    highpass_period: Some(highpass_period),
                },
            );
            let out = possible_rsi_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "possible_rsi".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.value);
            }
            if output_id.eq_ignore_ascii_case("buy_level") {
                return Ok(out.buy_level);
            }
            if output_id.eq_ignore_ascii_case("sell_level") {
                return Ok(out.sell_level);
            }
            if output_id.eq_ignore_ascii_case("middle")
                || output_id.eq_ignore_ascii_case("middle_level")
            {
                return Ok(out.middle_level);
            }
            if output_id.eq_ignore_ascii_case("trend") || output_id.eq_ignore_ascii_case("state") {
                return Ok(out.state);
            }
            if output_id.eq_ignore_ascii_case("long_signal") {
                return Ok(out.long_signal);
            }
            if output_id.eq_ignore_ascii_case("short_signal") {
                return Ok(out.short_signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "possible_rsi".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_autocorrelation_indicator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("autocorrelation_indicator", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "autocorrelation_indicator",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let length = get_usize_param("autocorrelation_indicator", params, "length", 20)?;
            let lag = get_usize_param("autocorrelation_indicator", params, "lag", 1)?;
            let use_test_signal = get_bool_param(
                "autocorrelation_indicator",
                params,
                "use_test_signal",
                false,
            )?;
            let max_lag = if output_id.eq_ignore_ascii_case("correlation") {
                lag
            } else {
                1
            };
            let input = AutocorrelationIndicatorInput::from_slice(
                data,
                AutocorrelationIndicatorParams {
                    length: Some(length),
                    max_lag: Some(max_lag),
                    use_test_signal: Some(use_test_signal),
                },
            );
            let out = autocorrelation_indicator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "autocorrelation_indicator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("filtered") || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.filtered);
            }
            if output_id.eq_ignore_ascii_case("correlation") {
                let start = (lag - 1).checked_mul(data.len()).ok_or_else(|| {
                    IndicatorDispatchError::ComputeFailed {
                        indicator: "autocorrelation_indicator".to_string(),
                        details: "lag * cols overflow".to_string(),
                    }
                })?;
                let end = start + data.len();
                return Ok(out.correlations[start..end].to_vec());
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "autocorrelation_indicator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_goertzel_cycle_composite_wave_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    if !output_id.eq_ignore_ascii_case("value") && !output_id.eq_ignore_ascii_case("wave") {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "goertzel_cycle_composite_wave".to_string(),
            output: output_id.to_string(),
        });
    }
    let data = extract_slice_input("goertzel_cycle_composite_wave", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "goertzel_cycle_composite_wave",
        output_id,
        req.combos,
        data.len(),
        |params, row| {
            let max_period =
                get_usize_param("goertzel_cycle_composite_wave", params, "max_period", 120)?;
            let start_at_cycle =
                get_usize_param("goertzel_cycle_composite_wave", params, "start_at_cycle", 1)?;
            let use_top_cycles =
                get_usize_param("goertzel_cycle_composite_wave", params, "use_top_cycles", 2)?;
            let bar_to_calculate = get_usize_param(
                "goertzel_cycle_composite_wave",
                params,
                "bar_to_calculate",
                1,
            )?;
            let detrend_mode = get_enum_string_param(
                "goertzel_cycle_composite_wave",
                params,
                "detrend_mode",
                "hodrick_prescott_detrending",
            )?;
            let detrend_mode = GoertzelDetrendMode::parse(detrend_mode).ok_or_else(|| {
                IndicatorDispatchError::InvalidParam {
                    indicator: "goertzel_cycle_composite_wave".to_string(),
                    key: "detrend_mode".to_string(),
                    reason: format!("unknown mode: {detrend_mode}"),
                }
            })?;
            let dt_zl_per1 =
                get_usize_param("goertzel_cycle_composite_wave", params, "dt_zl_per1", 10)?;
            let dt_zl_per2 =
                get_usize_param("goertzel_cycle_composite_wave", params, "dt_zl_per2", 40)?;
            let dt_hp_per1 =
                get_usize_param("goertzel_cycle_composite_wave", params, "dt_hp_per1", 20)?;
            let dt_hp_per2 =
                get_usize_param("goertzel_cycle_composite_wave", params, "dt_hp_per2", 80)?;
            let dt_reg_zl_smooth_per = get_usize_param(
                "goertzel_cycle_composite_wave",
                params,
                "dt_reg_zl_smooth_per",
                5,
            )?;
            let hp_smooth_per =
                get_usize_param("goertzel_cycle_composite_wave", params, "hp_smooth_per", 20)?;
            let zlma_smooth_per = get_usize_param(
                "goertzel_cycle_composite_wave",
                params,
                "zlma_smooth_per",
                10,
            )?;
            let filter_bartels = get_bool_param(
                "goertzel_cycle_composite_wave",
                params,
                "filter_bartels",
                false,
            )?;
            let bart_no_cycles =
                get_usize_param("goertzel_cycle_composite_wave", params, "bart_no_cycles", 5)?;
            let bart_smooth_per = get_usize_param(
                "goertzel_cycle_composite_wave",
                params,
                "bart_smooth_per",
                2,
            )?;
            let bart_sig_limit = get_usize_param(
                "goertzel_cycle_composite_wave",
                params,
                "bart_sig_limit",
                50,
            )?;
            let sort_bartels = get_bool_param(
                "goertzel_cycle_composite_wave",
                params,
                "sort_bartels",
                false,
            )?;
            let squared_amp =
                get_bool_param("goertzel_cycle_composite_wave", params, "squared_amp", true)?;
            let use_cosine =
                get_bool_param("goertzel_cycle_composite_wave", params, "use_cosine", true)?;
            let subtract_noise = get_bool_param(
                "goertzel_cycle_composite_wave",
                params,
                "subtract_noise",
                false,
            )?;
            let use_cycle_strength = get_bool_param(
                "goertzel_cycle_composite_wave",
                params,
                "use_cycle_strength",
                true,
            )?;

            let input = GoertzelCycleCompositeWaveInput::from_slice(
                data,
                GoertzelCycleCompositeWaveParams {
                    max_period: Some(max_period),
                    start_at_cycle: Some(start_at_cycle),
                    use_top_cycles: Some(use_top_cycles),
                    bar_to_calculate: Some(bar_to_calculate),
                    detrend_mode: Some(detrend_mode),
                    dt_zl_per1: Some(dt_zl_per1),
                    dt_zl_per2: Some(dt_zl_per2),
                    dt_hp_per1: Some(dt_hp_per1),
                    dt_hp_per2: Some(dt_hp_per2),
                    dt_reg_zl_smooth_per: Some(dt_reg_zl_smooth_per),
                    hp_smooth_per: Some(hp_smooth_per),
                    zlma_smooth_per: Some(zlma_smooth_per),
                    filter_bartels: Some(filter_bartels),
                    bart_no_cycles: Some(bart_no_cycles),
                    bart_smooth_per: Some(bart_smooth_per),
                    bart_sig_limit: Some(bart_sig_limit),
                    sort_bartels: Some(sort_bartels),
                    squared_amp: Some(squared_amp),
                    use_cosine: Some(use_cosine),
                    subtract_noise: Some(subtract_noise),
                    use_cycle_strength: Some(use_cycle_strength),
                },
            );
            goertzel_cycle_composite_wave_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "goertzel_cycle_composite_wave".to_string(),
                    details: e.to_string(),
                }
            })
        },
    )
}

fn compute_rolling_skewness_kurtosis_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("rolling_skewness_kurtosis", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "rolling_skewness_kurtosis",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let length = get_usize_param("rolling_skewness_kurtosis", params, "length", 50)?;
            let smooth_length =
                get_usize_param("rolling_skewness_kurtosis", params, "smooth_length", 3)?;
            let input = RollingSkewnessKurtosisInput::from_slice(
                data,
                RollingSkewnessKurtosisParams {
                    length: Some(length),
                    smooth_length: Some(smooth_length),
                },
            );
            let out = rolling_skewness_kurtosis_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "rolling_skewness_kurtosis".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("skewness") {
                return Ok(out.skewness);
            }
            if output_id.eq_ignore_ascii_case("kurtosis") {
                return Ok(out.kurtosis);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "rolling_skewness_kurtosis".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_rolling_z_score_trend_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("rolling_z_score_trend", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "rolling_z_score_trend",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let lookback_period =
                get_usize_param("rolling_z_score_trend", params, "lookback_period", 20)?;
            let input = RollingZScoreTrendInput::from_slice(
                data,
                RollingZScoreTrendParams {
                    lookback_period: Some(lookback_period),
                },
            );
            let out = rolling_z_score_trend_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "rolling_z_score_trend".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("zscore") {
                return Ok(out.zscore);
            }
            if output_id.eq_ignore_ascii_case("momentum") {
                return Ok(out.momentum);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "rolling_z_score_trend".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_ehlers_data_sampling_relative_strength_indicator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, close) = match req.data {
        IndicatorDataRef::Candles { candles, .. } => {
            (candles.open.as_slice(), candles.close.as_slice())
        }
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => {
            ensure_same_len_4(
                "ehlers_data_sampling_relative_strength_indicator",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
            )?;
            (open, close)
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "ehlers_data_sampling_relative_strength_indicator",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            (open, close)
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "ehlers_data_sampling_relative_strength_indicator".to_string(),
                input: IndicatorInputKind::Ohlc,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ehlers_data_sampling_relative_strength_indicator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param(
                "ehlers_data_sampling_relative_strength_indicator",
                params,
                "length",
                14,
            )?;
            let input = EhlersDataSamplingRelativeStrengthIndicatorInput::from_slices(
                open,
                close,
                EhlersDataSamplingRelativeStrengthIndicatorParams {
                    length: Some(length),
                },
            );
            let out = ehlers_data_sampling_relative_strength_indicator_with_kernel(&input, kernel)
                .map_err(|e| IndicatorDispatchError::ComputeFailed {
                    indicator: "ehlers_data_sampling_relative_strength_indicator".to_string(),
                    details: e.to_string(),
                })?;
            if output_id.eq_ignore_ascii_case("ds_rsi")
                || output_id.eq_ignore_ascii_case("data_sampling_rsi")
            {
                return Ok(out.ds_rsi);
            }
            if output_id.eq_ignore_ascii_case("original_rsi")
                || output_id.eq_ignore_ascii_case("orig_rsi")
            {
                return Ok(out.original_rsi);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "ehlers_data_sampling_relative_strength_indicator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_velocity_acceleration_convergence_divergence_indicator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let owned_source;
    let data = match req.data {
        IndicatorDataRef::Slice { values } => values,
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("hlcc4"))
        }
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => {
            ensure_same_len_4(
                "velocity_acceleration_convergence_divergence_indicator",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
            )?;
            owned_source = high
                .iter()
                .zip(low.iter())
                .zip(close.iter())
                .map(|((&h, &l), &c)| (h + l + 2.0 * c) * 0.25)
                .collect::<Vec<_>>();
            owned_source.as_slice()
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "velocity_acceleration_convergence_divergence_indicator",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            owned_source = high
                .iter()
                .zip(low.iter())
                .zip(close.iter())
                .map(|((&h, &l), &c)| (h + l + 2.0 * c) * 0.25)
                .collect::<Vec<_>>();
            owned_source.as_slice()
        }
        IndicatorDataRef::CloseVolume { close, volume } => {
            ensure_same_len_2(
                "velocity_acceleration_convergence_divergence_indicator",
                close.len(),
                volume.len(),
            )?;
            close
        }
        IndicatorDataRef::HighLow { .. } => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "velocity_acceleration_convergence_divergence_indicator".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "velocity_acceleration_convergence_divergence_indicator",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let length = get_usize_param(
                "velocity_acceleration_convergence_divergence_indicator",
                params,
                "length",
                21,
            )?;
            let smooth_length = get_usize_param(
                "velocity_acceleration_convergence_divergence_indicator",
                params,
                "smooth_length",
                5,
            )?;
            let input = VelocityAccelerationConvergenceDivergenceIndicatorInput::from_slice(
                data,
                VelocityAccelerationConvergenceDivergenceIndicatorParams {
                    length: Some(length),
                    smooth_length: Some(smooth_length),
                },
            );
            let out =
                velocity_acceleration_convergence_divergence_indicator_with_kernel(&input, kernel)
                    .map_err(|e| IndicatorDispatchError::ComputeFailed {
                        indicator: "velocity_acceleration_convergence_divergence_indicator"
                            .to_string(),
                        details: e.to_string(),
                    })?;
            if output_id.eq_ignore_ascii_case("vacd") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.vacd);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "velocity_acceleration_convergence_divergence_indicator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_trend_direction_force_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("trend_direction_force_index", output_id)?;
    let data = extract_slice_input("trend_direction_force_index", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "trend_direction_force_index",
        output_id,
        req.combos,
        data.len(),
        |params, row| {
            let length = get_usize_param("trend_direction_force_index", params, "length", 10)?;
            let input = TrendDirectionForceIndexInput::from_slice(
                data,
                TrendDirectionForceIndexParams {
                    length: Some(length),
                },
            );
            trend_direction_force_index_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "trend_direction_force_index".to_string(),
                    details: e.to_string(),
                }
            })
        },
    )
}

fn compute_yang_zhang_volatility_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("yang_zhang_volatility", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "yang_zhang_volatility",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let lookback = get_usize_param("yang_zhang_volatility", params, "lookback", 14)?;
            let k_override = get_bool_param("yang_zhang_volatility", params, "k_override", false)?;
            let k = get_f64_param("yang_zhang_volatility", params, "k", 0.34)?;
            let input = YangZhangVolatilityInput::from_slices(
                open,
                high,
                low,
                close,
                YangZhangVolatilityParams {
                    lookback: Some(lookback),
                    k_override: Some(k_override),
                    k: Some(k),
                },
            );
            let out = yang_zhang_volatility_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "yang_zhang_volatility".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("yz") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.yz);
            }
            if output_id.eq_ignore_ascii_case("rs") {
                return Ok(out.rs);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "yang_zhang_volatility".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_acosc_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low) = extract_high_low_input("acosc", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("acosc", output_id, req.combos, high.len(), |_params| {
        let input = AcoscInput::from_slices(high, low, AcoscParams::default());
        let out = acosc_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "acosc".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("osc") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.osc);
        }
        if output_id.eq_ignore_ascii_case("change") {
            return Ok(out.change);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "acosc".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_alligator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("alligator", req.data, "hl2")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("alligator", output_id, req.combos, data.len(), |params| {
        let jaw_period = get_usize_param("alligator", params, "jaw_period", 13)?;
        let jaw_offset = get_usize_param("alligator", params, "jaw_offset", 8)?;
        let teeth_period = get_usize_param("alligator", params, "teeth_period", 8)?;
        let teeth_offset = get_usize_param("alligator", params, "teeth_offset", 5)?;
        let lips_period = get_usize_param("alligator", params, "lips_period", 5)?;
        let lips_offset = get_usize_param("alligator", params, "lips_offset", 3)?;
        let input = AlligatorInput::from_slice(
            data,
            AlligatorParams {
                jaw_period: Some(jaw_period),
                jaw_offset: Some(jaw_offset),
                teeth_period: Some(teeth_period),
                teeth_offset: Some(teeth_offset),
                lips_period: Some(lips_period),
                lips_offset: Some(lips_offset),
            },
        );
        let out = alligator_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "alligator".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("jaw") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.jaw);
        }
        if output_id.eq_ignore_ascii_case("teeth") {
            return Ok(out.teeth);
        }
        if output_id.eq_ignore_ascii_case("lips") {
            return Ok(out.lips);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "alligator".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_alphatrend_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close, volume) = extract_ohlcv_full_input("alphatrend", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("alphatrend", output_id, req.combos, close.len(), |params| {
        let coeff = get_f64_param("alphatrend", params, "coeff", 1.0)?;
        let period = get_usize_param("alphatrend", params, "period", 14)?;
        let no_volume = get_bool_param("alphatrend", params, "no_volume", false)?;
        let input = AlphaTrendInput::from_slices(
            open,
            high,
            low,
            close,
            volume,
            AlphaTrendParams {
                coeff: Some(coeff),
                period: Some(period),
                no_volume: Some(no_volume),
            },
        );
        let out = alphatrend_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "alphatrend".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("k1") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.k1);
        }
        if output_id.eq_ignore_ascii_case("k2") {
            return Ok(out.k2);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "alphatrend".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_aso_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = match req.data {
        IndicatorDataRef::Candles { candles, source } => (
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            source_type(candles, source.unwrap_or("close")),
        ),
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => {
            ensure_same_len_4("aso", open.len(), high.len(), low.len(), close.len())?;
            (open, high, low, close)
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "aso",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            (open, high, low, close)
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "aso".to_string(),
                input: IndicatorInputKind::Ohlc,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64("aso", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("aso", params, "period", 10)?;
        let mode = get_usize_param("aso", params, "mode", 0)?;
        let input = AsoInput::from_slices(
            open,
            high,
            low,
            close,
            AsoParams {
                period: Some(period),
                mode: Some(mode),
            },
        );
        let out =
            aso_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "aso".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("bulls") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.bulls);
        }
        if output_id.eq_ignore_ascii_case("bears") {
            return Ok(out.bears);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "aso".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_avsl_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("avsl", output_id)?;
    let (_high, low, close, volume) = extract_hlcv_input("avsl", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("avsl", output_id, req.combos, close.len(), |params| {
        let fast_period = get_usize_param("avsl", params, "fast_period", 12)?;
        let slow_period = get_usize_param("avsl", params, "slow_period", 26)?;
        let multiplier = get_f64_param("avsl", params, "multiplier", 2.0)?;
        let input = AvslInput::from_slices(
            close,
            low,
            volume,
            AvslParams {
                fast_period: Some(fast_period),
                slow_period: Some(slow_period),
                multiplier: Some(multiplier),
            },
        );
        let out = avsl_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "avsl".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_bandpass_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("bandpass", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("bandpass", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("bandpass", params, "period", 20)?;
        let bandwidth = get_f64_param("bandpass", params, "bandwidth", 0.3)?;
        let input = BandPassInput::from_slice(
            data,
            BandPassParams {
                period: Some(period),
                bandwidth: Some(bandwidth),
            },
        );
        let out = bandpass_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "bandpass".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("bp") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.bp);
        }
        if output_id.eq_ignore_ascii_case("bp_normalized")
            || output_id.eq_ignore_ascii_case("normalized")
        {
            return Ok(out.bp_normalized);
        }
        if output_id.eq_ignore_ascii_case("signal") {
            return Ok(out.signal);
        }
        if output_id.eq_ignore_ascii_case("trigger") {
            return Ok(out.trigger);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "bandpass".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_chande_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("chande", output_id)?;
    let (high, low, close) = extract_ohlc_input("chande", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("chande", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("chande", params, "period", 22)?;
        let mult = get_f64_param("chande", params, "mult", 3.0)?;
        let direction = get_enum_param("chande", params, "direction", "long")?;
        let input = ChandeInput::from_slices(
            high,
            low,
            close,
            ChandeParams {
                period: Some(period),
                mult: Some(mult),
                direction: Some(direction.to_string()),
            },
        );
        let out = chande_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "chande".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_chandelier_exit_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("chandelier_exit", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "chandelier_exit",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let period = get_usize_param("chandelier_exit", params, "period", 22)?;
            let mult = get_f64_param("chandelier_exit", params, "mult", 3.0)?;
            let use_close = get_bool_param("chandelier_exit", params, "use_close", true)?;
            let input = ChandelierExitInput::from_slices(
                high,
                low,
                close,
                ChandelierExitParams {
                    period: Some(period),
                    mult: Some(mult),
                    use_close: Some(use_close),
                },
            );
            let out = chandelier_exit_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "chandelier_exit".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("long_stop")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.long_stop);
            }
            if output_id.eq_ignore_ascii_case("short_stop") {
                return Ok(out.short_stop);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "chandelier_exit".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_cksp_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("cksp", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("cksp", output_id, req.combos, close.len(), |params| {
        let p = get_usize_param("cksp", params, "p", 10)?;
        let x = get_f64_param("cksp", params, "x", 1.0)?;
        let q = get_usize_param("cksp", params, "q", 9)?;
        let input = CkspInput::from_slices(
            high,
            low,
            close,
            CkspParams {
                p: Some(p),
                x: Some(x),
                q: Some(q),
            },
        );
        let out = cksp_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "cksp".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("long_values")
            || output_id.eq_ignore_ascii_case("long")
            || output_id.eq_ignore_ascii_case("value")
        {
            return Ok(out.long_values);
        }
        if output_id.eq_ignore_ascii_case("short_values") || output_id.eq_ignore_ascii_case("short")
        {
            return Ok(out.short_values);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "cksp".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_correlation_cycle_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("correlation_cycle", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "correlation_cycle",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let period = get_usize_param("correlation_cycle", params, "period", 20)?;
            let threshold = get_f64_param("correlation_cycle", params, "threshold", 9.0)?;
            let input = CorrelationCycleInput::from_slice(
                data,
                CorrelationCycleParams {
                    period: Some(period),
                    threshold: Some(threshold),
                },
            );
            let out = correlation_cycle_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "correlation_cycle".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("real") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.real);
            }
            if output_id.eq_ignore_ascii_case("imag") {
                return Ok(out.imag);
            }
            if output_id.eq_ignore_ascii_case("angle") {
                return Ok(out.angle);
            }
            if output_id.eq_ignore_ascii_case("state") {
                return Ok(out.state);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "correlation_cycle".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_damiani_volatmeter_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("damiani_volatmeter", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "damiani_volatmeter",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let vis_atr = get_usize_param("damiani_volatmeter", params, "vis_atr", 13)?;
            let vis_std = get_usize_param("damiani_volatmeter", params, "vis_std", 20)?;
            let sed_atr = get_usize_param("damiani_volatmeter", params, "sed_atr", 40)?;
            let sed_std = get_usize_param("damiani_volatmeter", params, "sed_std", 100)?;
            let threshold = get_f64_param("damiani_volatmeter", params, "threshold", 1.4)?;
            let input = DamianiVolatmeterInput::from_slice(
                data,
                DamianiVolatmeterParams {
                    vis_atr: Some(vis_atr),
                    vis_std: Some(vis_std),
                    sed_atr: Some(sed_atr),
                    sed_std: Some(sed_std),
                    threshold: Some(threshold),
                },
            );
            let out = damiani_volatmeter_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "damiani_volatmeter".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("vol") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.vol);
            }
            if output_id.eq_ignore_ascii_case("anti") {
                return Ok(out.anti);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "damiani_volatmeter".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_dvdiqqe_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close, volume) = match req.data {
        IndicatorDataRef::Candles { candles, .. } => (
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            Some(candles.volume.as_slice()),
        ),
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "dvdiqqe",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            (open, high, low, close, Some(volume))
        }
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => {
            ensure_same_len_4("dvdiqqe", open.len(), high.len(), low.len(), close.len())?;
            (open, high, low, close, None)
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "dvdiqqe".to_string(),
                input: IndicatorInputKind::Ohlc,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64("dvdiqqe", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("dvdiqqe", params, "period", 13)?;
        let smoothing_period = get_usize_param("dvdiqqe", params, "smoothing_period", 6)?;
        let fast_multiplier = get_f64_param("dvdiqqe", params, "fast_multiplier", 2.618)?;
        let slow_multiplier = get_f64_param("dvdiqqe", params, "slow_multiplier", 4.236)?;
        let volume_type = get_enum_param("dvdiqqe", params, "volume_type", "default")?;
        let center_type = get_enum_param("dvdiqqe", params, "center_type", "dynamic")?;
        let tick_size = get_f64_param("dvdiqqe", params, "tick_size", 0.01)?;
        let input = DvdiqqeInput::from_slices(
            open,
            high,
            low,
            close,
            volume,
            DvdiqqeParams {
                period: Some(period),
                smoothing_period: Some(smoothing_period),
                fast_multiplier: Some(fast_multiplier),
                slow_multiplier: Some(slow_multiplier),
                volume_type: Some(volume_type),
                center_type: Some(center_type),
                tick_size: Some(tick_size),
            },
        );
        let out = dvdiqqe_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "dvdiqqe".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("dvdi") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.dvdi);
        }
        if output_id.eq_ignore_ascii_case("fast_tl") || output_id.eq_ignore_ascii_case("fast") {
            return Ok(out.fast_tl);
        }
        if output_id.eq_ignore_ascii_case("slow_tl") || output_id.eq_ignore_ascii_case("slow") {
            return Ok(out.slow_tl);
        }
        if output_id.eq_ignore_ascii_case("center_line") || output_id.eq_ignore_ascii_case("center")
        {
            return Ok(out.center_line);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "dvdiqqe".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_emd_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close, volume) = extract_hlcv_input("emd", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("emd", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("emd", params, "period", 20)?;
        let delta = get_f64_param("emd", params, "delta", 0.5)?;
        let fraction = get_f64_param("emd", params, "fraction", 0.1)?;
        let input = EmdInput::from_slices(
            high,
            low,
            close,
            volume,
            EmdParams {
                period: Some(period),
                delta: Some(delta),
                fraction: Some(fraction),
            },
        );
        let out =
            emd_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "emd".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("upperband")
            || output_id.eq_ignore_ascii_case("upper")
            || output_id.eq_ignore_ascii_case("value")
        {
            return Ok(out.upperband);
        }
        if output_id.eq_ignore_ascii_case("middleband") || output_id.eq_ignore_ascii_case("middle")
        {
            return Ok(out.middleband);
        }
        if output_id.eq_ignore_ascii_case("lowerband") || output_id.eq_ignore_ascii_case("lower") {
            return Ok(out.lowerband);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "emd".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_emd_trend_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("emd_trend", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("emd_trend", output_id, req.combos, close.len(), |params| {
        let source = get_enum_param("emd_trend", params, "source", "close")?;
        let avg_type = get_enum_param("emd_trend", params, "avg_type", "SMA")?;
        let length = get_usize_param("emd_trend", params, "length", 28)?;
        let mult = get_f64_param("emd_trend", params, "mult", 1.0)?;
        let input = EmdTrendInput::from_slices(
            open,
            high,
            low,
            close,
            EmdTrendParams {
                source: Some(source),
                avg_type: Some(avg_type),
                length: Some(length),
                mult: Some(mult),
            },
        );
        let out = emd_trend_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "emd_trend".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("direction") {
            return Ok(out.direction);
        }
        if output_id.eq_ignore_ascii_case("average") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.average);
        }
        if output_id.eq_ignore_ascii_case("upper") {
            return Ok(out.upper);
        }
        if output_id.eq_ignore_ascii_case("lower") {
            return Ok(out.lower);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "emd_trend".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_cyberpunk_value_trend_analyzer_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) =
        extract_ohlc_full_input("cyberpunk_value_trend_analyzer", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "cyberpunk_value_trend_analyzer",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let entry_level =
                get_usize_param("cyberpunk_value_trend_analyzer", params, "entry_level", 30)?;
            let exit_level =
                get_usize_param("cyberpunk_value_trend_analyzer", params, "exit_level", 75)?;
            let input = CyberpunkValueTrendAnalyzerInput::from_slices(
                open,
                high,
                low,
                close,
                CyberpunkValueTrendAnalyzerParams {
                    entry_level: Some(entry_level),
                    exit_level: Some(exit_level),
                },
            );
            let out = cyberpunk_value_trend_analyzer_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "cyberpunk_value_trend_analyzer".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value_trend")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.value_trend);
            }
            if output_id.eq_ignore_ascii_case("value_trend_lag")
                || output_id.eq_ignore_ascii_case("lag")
            {
                return Ok(out.value_trend_lag);
            }
            if output_id.eq_ignore_ascii_case("deviation_index") {
                return Ok(out.deviation_index);
            }
            if output_id.eq_ignore_ascii_case("overbought_signal")
                || output_id.eq_ignore_ascii_case("overbought")
            {
                return Ok(out.overbought_signal);
            }
            if output_id.eq_ignore_ascii_case("buy_signal") {
                return Ok(out.buy_signal);
            }
            if output_id.eq_ignore_ascii_case("sell_signal") {
                return Ok(out.sell_signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "cyberpunk_value_trend_analyzer".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_eri_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, source) = match req.data {
        IndicatorDataRef::Candles { candles, source } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            source_type(candles, source.unwrap_or("close")),
        ),
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => {
            ensure_same_len_4("eri", open.len(), high.len(), low.len(), close.len())?;
            (high, low, close)
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "eri",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            (high, low, close)
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "eri".to_string(),
                input: IndicatorInputKind::Ohlc,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64("eri", output_id, req.combos, source.len(), |params| {
        let period = get_usize_param("eri", params, "period", 13)?;
        let ma_type = get_enum_param("eri", params, "ma_type", "ema")?;
        let input = EriInput::from_slices(
            high,
            low,
            source,
            EriParams {
                period: Some(period),
                ma_type: Some(ma_type),
            },
        );
        let out =
            eri_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "eri".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("bull") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.bull);
        }
        if output_id.eq_ignore_ascii_case("bear") {
            return Ok(out.bear);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "eri".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_fisher_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low) = extract_high_low_input("fisher", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("fisher", output_id, req.combos, high.len(), |params| {
        let period = get_usize_param("fisher", params, "period", 9)?;
        let input = FisherInput::from_slices(
            high,
            low,
            FisherParams {
                period: Some(period),
            },
        );
        let out = fisher_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "fisher".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("fisher") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.fisher);
        }
        if output_id.eq_ignore_ascii_case("signal") {
            return Ok(out.signal);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "fisher".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_fvg_positioning_average_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("fvg_positioning_average", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "fvg_positioning_average",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let lookback = get_usize_param("fvg_positioning_average", params, "lookback", 30)?;
            let lookback_type = get_enum_param(
                "fvg_positioning_average",
                params,
                "lookback_type",
                "Bar Count",
            )?;
            let atr_multiplier =
                get_f64_param("fvg_positioning_average", params, "atr_multiplier", 0.25)?;
            let input = FvgPositioningAverageInput::from_slices(
                open,
                high,
                low,
                close,
                FvgPositioningAverageParams {
                    lookback: Some(lookback),
                    lookback_type: Some(lookback_type),
                    atr_multiplier: Some(atr_multiplier),
                },
            );
            let out = fvg_positioning_average_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "fvg_positioning_average".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("bull_average")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.bull_average);
            }
            if output_id.eq_ignore_ascii_case("bear_average") {
                return Ok(out.bear_average);
            }
            if output_id.eq_ignore_ascii_case("bull_mid") {
                return Ok(out.bull_mid);
            }
            if output_id.eq_ignore_ascii_case("bear_mid") {
                return Ok(out.bear_mid);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "fvg_positioning_average".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_fvg_trailing_stop_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("fvg_trailing_stop", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "fvg_trailing_stop",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let lookback =
                get_usize_param("fvg_trailing_stop", params, "unmitigated_fvg_lookback", 5)?;
            let smoothing_length =
                get_usize_param("fvg_trailing_stop", params, "smoothing_length", 9)?;
            let reset_on_cross =
                get_bool_param("fvg_trailing_stop", params, "reset_on_cross", false)?;
            let input = FvgTrailingStopInput::from_slices(
                high,
                low,
                close,
                FvgTrailingStopParams {
                    unmitigated_fvg_lookback: Some(lookback),
                    smoothing_length: Some(smoothing_length),
                    reset_on_cross: Some(reset_on_cross),
                },
            );
            let out = fvg_trailing_stop_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "fvg_trailing_stop".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("upper") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.upper);
            }
            if output_id.eq_ignore_ascii_case("lower") {
                return Ok(out.lower);
            }
            if output_id.eq_ignore_ascii_case("upper_ts") {
                return Ok(out.upper_ts);
            }
            if output_id.eq_ignore_ascii_case("lower_ts") {
                return Ok(out.lower_ts);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "fvg_trailing_stop".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_gatorosc_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("gatorosc", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("gatorosc", output_id, req.combos, data.len(), |params| {
        let jaws_length = get_usize_param("gatorosc", params, "jaws_length", 13)?;
        let jaws_shift = get_usize_param("gatorosc", params, "jaws_shift", 8)?;
        let teeth_length = get_usize_param("gatorosc", params, "teeth_length", 8)?;
        let teeth_shift = get_usize_param("gatorosc", params, "teeth_shift", 5)?;
        let lips_length = get_usize_param("gatorosc", params, "lips_length", 5)?;
        let lips_shift = get_usize_param("gatorosc", params, "lips_shift", 3)?;
        let input = GatorOscInput::from_slice(
            data,
            GatorOscParams {
                jaws_length: Some(jaws_length),
                jaws_shift: Some(jaws_shift),
                teeth_length: Some(teeth_length),
                teeth_shift: Some(teeth_shift),
                lips_length: Some(lips_length),
                lips_shift: Some(lips_shift),
            },
        );
        let out = gatorosc_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "gatorosc".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("upper") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.upper);
        }
        if output_id.eq_ignore_ascii_case("lower") {
            return Ok(out.lower);
        }
        if output_id.eq_ignore_ascii_case("upper_change") {
            return Ok(out.upper_change);
        }
        if output_id.eq_ignore_ascii_case("lower_change") {
            return Ok(out.lower_change);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "gatorosc".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_halftrend_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("halftrend", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("halftrend", output_id, req.combos, close.len(), |params| {
        let amplitude = get_usize_param("halftrend", params, "amplitude", 2)?;
        let channel_deviation = get_f64_param("halftrend", params, "channel_deviation", 2.0)?;
        let atr_period = get_usize_param("halftrend", params, "atr_period", 100)?;
        let input = HalfTrendInput::from_slices(
            high,
            low,
            close,
            HalfTrendParams {
                amplitude: Some(amplitude),
                channel_deviation: Some(channel_deviation),
                atr_period: Some(atr_period),
            },
        );
        let out = halftrend_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "halftrend".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("halftrend") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.halftrend);
        }
        if output_id.eq_ignore_ascii_case("trend") {
            return Ok(out.trend);
        }
        if output_id.eq_ignore_ascii_case("atr_high") {
            return Ok(out.atr_high);
        }
        if output_id.eq_ignore_ascii_case("atr_low") {
            return Ok(out.atr_low);
        }
        if output_id.eq_ignore_ascii_case("buy_signal") || output_id.eq_ignore_ascii_case("buy") {
            return Ok(out.buy_signal);
        }
        if output_id.eq_ignore_ascii_case("sell_signal") || output_id.eq_ignore_ascii_case("sell") {
            return Ok(out.sell_signal);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "halftrend".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_safezonestop_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low) = extract_high_low_input("safezonestop", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "safezonestop",
        output_id,
        req.combos,
        high.len(),
        |params| {
            let period = get_usize_param("safezonestop", params, "period", 22)?;
            let mult = get_f64_param("safezonestop", params, "mult", 2.5)?;
            let max_lookback = get_usize_param("safezonestop", params, "max_lookback", 3)?;
            let direction = get_enum_param("safezonestop", params, "direction", "long")?;
            let input = SafeZoneStopInput::from_slices(
                high,
                low,
                direction.as_str(),
                SafeZoneStopParams {
                    period: Some(period),
                    mult: Some(mult),
                    max_lookback: Some(max_lookback),
                },
            );
            let out = safezonestop_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "safezonestop".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.values);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "safezonestop".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_devstop_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low) = extract_high_low_input("devstop", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("devstop", output_id, req.combos, high.len(), |params| {
        let period = get_usize_param("devstop", params, "period", 20)?;
        let mult = get_f64_param("devstop", params, "mult", 0.0)?;
        let devtype = get_usize_param("devstop", params, "devtype", 0)?;
        let direction = get_enum_param("devstop", params, "direction", "long")?;
        let ma_type = get_enum_param("devstop", params, "ma_type", "sma")?;
        let input = DevStopInput::from_slices(
            high,
            low,
            DevStopParams {
                period: Some(period),
                mult: Some(mult),
                devtype: Some(devtype),
                direction: Some(direction),
                ma_type: Some(ma_type),
            },
        );
        let out = devstop_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "devstop".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("value") {
            return Ok(out.values);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "devstop".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_chop_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("chop", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("chop", output_id, req.combos, close.len(), |params| {
        let period = get_usize_param("chop", params, "period", 14)?;
        let scalar = get_f64_param("chop", params, "scalar", 100.0)?;
        let drift = get_usize_param("chop", params, "drift", 1)?;
        let input = ChopInput::from_slices(
            high,
            low,
            close,
            ChopParams {
                period: Some(period),
                scalar: Some(scalar),
                drift: Some(drift),
            },
        );
        let out = chop_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "chop".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("value") {
            return Ok(out.values);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "chop".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_kst_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("kst", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("kst", output_id, req.combos, data.len(), |params| {
        let sma_period1 = get_usize_param("kst", params, "sma_period1", 10)?;
        let sma_period2 = get_usize_param("kst", params, "sma_period2", 10)?;
        let sma_period3 = get_usize_param("kst", params, "sma_period3", 10)?;
        let sma_period4 = get_usize_param("kst", params, "sma_period4", 15)?;
        let roc_period1 = get_usize_param("kst", params, "roc_period1", 10)?;
        let roc_period2 = get_usize_param("kst", params, "roc_period2", 15)?;
        let roc_period3 = get_usize_param("kst", params, "roc_period3", 20)?;
        let roc_period4 = get_usize_param("kst", params, "roc_period4", 30)?;
        let signal_period = get_usize_param("kst", params, "signal_period", 9)?;
        let input = KstInput::from_slice(
            data,
            KstParams {
                sma_period1: Some(sma_period1),
                sma_period2: Some(sma_period2),
                sma_period3: Some(sma_period3),
                sma_period4: Some(sma_period4),
                roc_period1: Some(roc_period1),
                roc_period2: Some(roc_period2),
                roc_period3: Some(roc_period3),
                roc_period4: Some(roc_period4),
                signal_period: Some(signal_period),
            },
        );
        let out =
            kst_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "kst".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("line") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.line);
        }
        if output_id.eq_ignore_ascii_case("signal") {
            return Ok(out.signal);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "kst".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_kaufmanstop_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("kaufmanstop", output_id)?;
    let (high, low) = extract_high_low_input("kaufmanstop", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("kaufmanstop", output_id, req.combos, high.len(), |params| {
        let period = get_usize_param("kaufmanstop", params, "period", 22)?;
        let mult = get_f64_param("kaufmanstop", params, "mult", 2.0)?;
        let direction = get_enum_param("kaufmanstop", params, "direction", "long")?;
        let ma_type = get_enum_param("kaufmanstop", params, "ma_type", "sma")?;
        let input = KaufmanstopInput::from_slices(
            high,
            low,
            KaufmanstopParams {
                period: Some(period),
                mult: Some(mult),
                direction: Some(direction),
                ma_type: Some(ma_type),
            },
        );
        let out = kaufmanstop_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "kaufmanstop".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_lpc_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close, src) = match req.data {
        IndicatorDataRef::Candles { candles, source } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            source_type(candles, source.unwrap_or("close")),
        ),
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => {
            ensure_same_len_4("lpc", open.len(), high.len(), low.len(), close.len())?;
            (high, low, close, close)
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "lpc",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            (high, low, close, close)
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "lpc".to_string(),
                input: IndicatorInputKind::Ohlc,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64("lpc", output_id, req.combos, src.len(), |params| {
        let cutoff_type = get_enum_param("lpc", params, "cutoff_type", "adaptive")?;
        let fixed_period = get_usize_param("lpc", params, "fixed_period", 20)?;
        let max_cycle_limit = get_usize_param("lpc", params, "max_cycle_limit", 60)?;
        let cycle_mult = get_f64_param("lpc", params, "cycle_mult", 1.0)?;
        let tr_mult = get_f64_param("lpc", params, "tr_mult", 1.0)?;
        let input = LpcInput::from_slices(
            high,
            low,
            close,
            src,
            LpcParams {
                cutoff_type: Some(cutoff_type),
                fixed_period: Some(fixed_period),
                max_cycle_limit: Some(max_cycle_limit),
                cycle_mult: Some(cycle_mult),
                tr_mult: Some(tr_mult),
            },
        );
        let out =
            lpc_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "lpc".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("filter") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.filter);
        }
        if output_id.eq_ignore_ascii_case("high_band") || output_id.eq_ignore_ascii_case("high") {
            return Ok(out.high_band);
        }
        if output_id.eq_ignore_ascii_case("low_band") || output_id.eq_ignore_ascii_case("low") {
            return Ok(out.low_band);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "lpc".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_mab_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("mab", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("mab", output_id, req.combos, data.len(), |params| {
        let fast_period = get_usize_param("mab", params, "fast_period", 10)?;
        let slow_period = get_usize_param("mab", params, "slow_period", 50)?;
        let devup = get_f64_param("mab", params, "devup", 1.0)?;
        let devdn = get_f64_param("mab", params, "devdn", 1.0)?;
        let fast_ma_type = get_enum_param("mab", params, "fast_ma_type", "sma")?;
        let slow_ma_type = get_enum_param("mab", params, "slow_ma_type", "sma")?;
        let input = MabInput::from_slice(
            data,
            MabParams {
                fast_period: Some(fast_period),
                slow_period: Some(slow_period),
                devup: Some(devup),
                devdn: Some(devdn),
                fast_ma_type: Some(fast_ma_type),
                slow_ma_type: Some(slow_ma_type),
            },
        );
        let out =
            mab_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "mab".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("upperband")
            || output_id.eq_ignore_ascii_case("upper")
            || output_id.eq_ignore_ascii_case("value")
        {
            return Ok(out.upperband);
        }
        if output_id.eq_ignore_ascii_case("middleband") || output_id.eq_ignore_ascii_case("middle")
        {
            return Ok(out.middleband);
        }
        if output_id.eq_ignore_ascii_case("lowerband") || output_id.eq_ignore_ascii_case("lower") {
            return Ok(out.lowerband);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "mab".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_macz_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (data, volume) = match req.data {
        IndicatorDataRef::Slice { values } => (values, None),
        IndicatorDataRef::Candles { candles, source } => (
            source_type(candles, source.unwrap_or("close")),
            Some(candles.volume.as_slice()),
        ),
        IndicatorDataRef::CloseVolume { close, volume } => {
            ensure_same_len_2("macz", close.len(), volume.len())?;
            (close, Some(volume))
        }
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => {
            ensure_same_len_4("macz", open.len(), high.len(), low.len(), close.len())?;
            (close, None)
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "macz",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            (close, Some(volume))
        }
        IndicatorDataRef::HighLow { .. } => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "macz".to_string(),
                input: IndicatorInputKind::Slice,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64("macz", output_id, req.combos, data.len(), |params| {
        let fast_length = get_usize_param("macz", params, "fast_length", 12)?;
        let slow_length = get_usize_param("macz", params, "slow_length", 25)?;
        let signal_length = get_usize_param("macz", params, "signal_length", 9)?;
        let lengthz = get_usize_param("macz", params, "lengthz", 20)?;
        let length_stdev = get_usize_param("macz", params, "length_stdev", 25)?;
        let a = get_f64_param("macz", params, "a", 1.0)?;
        let b = get_f64_param("macz", params, "b", 1.0)?;
        let use_lag = get_bool_param("macz", params, "use_lag", false)?;
        let gamma = get_f64_param("macz", params, "gamma", 0.02)?;
        let macz_params = MaczParams {
            fast_length: Some(fast_length),
            slow_length: Some(slow_length),
            signal_length: Some(signal_length),
            lengthz: Some(lengthz),
            length_stdev: Some(length_stdev),
            a: Some(a),
            b: Some(b),
            use_lag: Some(use_lag),
            gamma: Some(gamma),
        };
        let input = if let Some(vol) = volume {
            MaczInput::from_slice_with_volume(data, vol, macz_params)
        } else {
            MaczInput::from_slice(data, macz_params)
        };
        let out = macz_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "macz".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("value") || output_id.eq_ignore_ascii_case("values") {
            return Ok(out.values);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "macz".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_minmax_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low) = extract_high_low_input("minmax", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("minmax", output_id, req.combos, high.len(), |params| {
        let order = get_usize_param("minmax", params, "order", 3)?;
        let input = MinmaxInput::from_slices(high, low, MinmaxParams { order: Some(order) });
        let out = minmax_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "minmax".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("is_min") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.is_min);
        }
        if output_id.eq_ignore_ascii_case("is_max") {
            return Ok(out.is_max);
        }
        if output_id.eq_ignore_ascii_case("last_min") {
            return Ok(out.last_min);
        }
        if output_id.eq_ignore_ascii_case("last_max") {
            return Ok(out.last_max);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "minmax".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_mod_god_mode_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close, volume) = match req.data {
        IndicatorDataRef::Candles { candles, .. } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            Some(candles.volume.as_slice()),
        ),
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => {
            ensure_same_len_4(
                "mod_god_mode",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
            )?;
            (high, low, close, None)
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "mod_god_mode",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            (high, low, close, Some(volume))
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "mod_god_mode".to_string(),
                input: IndicatorInputKind::Ohlc,
            });
        }
    };

    collect_f64(
        "mod_god_mode",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let n1 = get_usize_param("mod_god_mode", params, "n1", 17)?;
            let n2 = get_usize_param("mod_god_mode", params, "n2", 6)?;
            let n3 = get_usize_param("mod_god_mode", params, "n3", 4)?;
            let mode = get_enum_param("mod_god_mode", params, "mode", "tradition_mg")?;
            let use_volume = get_bool_param("mod_god_mode", params, "use_volume", true)?;
            let mode = match mode.as_str() {
                "godmode" => ModGodModeMode::Godmode,
                "tradition" => ModGodModeMode::Tradition,
                "godmode_mg" => ModGodModeMode::GodmodeMg,
                "tradition_mg" => ModGodModeMode::TraditionMg,
                other => {
                    return Err(IndicatorDispatchError::InvalidParam {
                        indicator: "mod_god_mode".to_string(),
                        key: "mode".to_string(),
                        reason: format!("unknown mode: {other}"),
                    });
                }
            };
            let input = ModGodModeInput {
                data: ModGodModeData::Slices {
                    high,
                    low,
                    close,
                    volume: if use_volume { volume } else { None },
                },
                params: ModGodModeParams {
                    n1: Some(n1),
                    n2: Some(n2),
                    n3: Some(n3),
                    mode: Some(mode),
                    use_volume: Some(use_volume),
                },
            };
            let out = mod_god_mode(&input).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "mod_god_mode".to_string(),
                details: e.to_string(),
            })?;
            if output_id.eq_ignore_ascii_case("wavetrend")
                || output_id.eq_ignore_ascii_case("wt1")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.wavetrend);
            }
            if output_id.eq_ignore_ascii_case("signal") || output_id.eq_ignore_ascii_case("wt2") {
                return Ok(out.signal);
            }
            if output_id.eq_ignore_ascii_case("histogram") || output_id.eq_ignore_ascii_case("hist")
            {
                return Ok(out.histogram);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "mod_god_mode".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_msw_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("msw", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("msw", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("msw", params, "period", 5)?;
        let input = MswInput::from_slice(
            data,
            MswParams {
                period: Some(period),
            },
        );
        let out =
            msw_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "msw".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("sine") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.sine);
        }
        if output_id.eq_ignore_ascii_case("lead") {
            return Ok(out.lead);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "msw".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_nadaraya_watson_envelope_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("nadaraya_watson_envelope", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "nadaraya_watson_envelope",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let bandwidth = get_f64_param("nadaraya_watson_envelope", params, "bandwidth", 8.0)?;
            let multiplier = get_f64_param("nadaraya_watson_envelope", params, "multiplier", 3.0)?;
            let lookback = get_usize_param("nadaraya_watson_envelope", params, "lookback", 500)?;
            let input = NweInput::from_slice(
                data,
                NweParams {
                    bandwidth: Some(bandwidth),
                    multiplier: Some(multiplier),
                    lookback: Some(lookback),
                },
            );
            let out = nadaraya_watson_envelope_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "nadaraya_watson_envelope".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("upper") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.upper);
            }
            if output_id.eq_ignore_ascii_case("lower") {
                return Ok(out.lower);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "nadaraya_watson_envelope".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_otto_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("otto", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("otto", output_id, req.combos, data.len(), |params| {
        let ott_period = get_usize_param("otto", params, "ott_period", 2)?;
        let ott_percent = get_f64_param("otto", params, "ott_percent", 0.6)?;
        let fast_vidya_length = get_usize_param("otto", params, "fast_vidya_length", 10)?;
        let slow_vidya_length = get_usize_param("otto", params, "slow_vidya_length", 25)?;
        let correcting_constant = get_f64_param("otto", params, "correcting_constant", 100000.0)?;
        let ma_type = get_enum_param("otto", params, "ma_type", "VAR")?;
        let input = OttoInput::from_slice(
            data,
            OttoParams {
                ott_period: Some(ott_period),
                ott_percent: Some(ott_percent),
                fast_vidya_length: Some(fast_vidya_length),
                slow_vidya_length: Some(slow_vidya_length),
                correcting_constant: Some(correcting_constant),
                ma_type: Some(ma_type),
            },
        );
        let out = otto_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "otto".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("hott") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.hott);
        }
        if output_id.eq_ignore_ascii_case("lott") {
            return Ok(out.lott);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "otto".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_vidya_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("vidya", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("vidya", output_id, req.combos, data.len(), |params| {
        let short_period = get_usize_param("vidya", params, "short_period", 2)?;
        let long_period = get_usize_param("vidya", params, "long_period", 5)?;
        let alpha = get_f64_param("vidya", params, "alpha", 0.2)?;
        let input = VidyaInput::from_slice(
            data,
            VidyaParams {
                short_period: Some(short_period),
                long_period: Some(long_period),
                alpha: Some(alpha),
            },
        );
        let out = vidya_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "vidya".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("value") || output_id.eq_ignore_ascii_case("values") {
            return Ok(out.values);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "vidya".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_vlma_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("vlma", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("vlma", output_id, req.combos, data.len(), |params| {
        let min_period = get_usize_param("vlma", params, "min_period", 5)?;
        let max_period = get_usize_param("vlma", params, "max_period", 50)?;
        let matype = get_enum_param("vlma", params, "matype", "sma")?;
        let devtype = get_usize_param("vlma", params, "devtype", 0)?;
        let input = VlmaInput::from_slice(
            data,
            VlmaParams {
                min_period: Some(min_period),
                max_period: Some(max_period),
                matype: Some(matype),
                devtype: Some(devtype),
            },
        );
        let out = vlma_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "vlma".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("value") || output_id.eq_ignore_ascii_case("values") {
            return Ok(out.values);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "vlma".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_pma_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("pma", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("pma", output_id, req.combos, data.len(), |_params| {
        let input = PmaInput::from_slice(data, PmaParams::default());
        let out =
            pma_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "pma".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("predict") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.predict);
        }
        if output_id.eq_ignore_ascii_case("trigger") {
            return Ok(out.trigger);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "pma".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_prb_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("prb", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("prb", output_id, req.combos, data.len(), |params| {
        let smooth_data = get_bool_param("prb", params, "smooth_data", true)?;
        let smooth_period = get_usize_param("prb", params, "smooth_period", 10)?;
        let regression_period = get_usize_param("prb", params, "regression_period", 100)?;
        let polynomial_order = get_usize_param("prb", params, "polynomial_order", 2)?;
        let regression_offset = get_i32_param("prb", params, "regression_offset", 0)?;
        let ndev = get_f64_param("prb", params, "ndev", 2.0)?;
        let equ_from = get_usize_param("prb", params, "equ_from", 0)?;
        let input = PrbInput::from_slice(
            data,
            PrbParams {
                smooth_data: Some(smooth_data),
                smooth_period: Some(smooth_period),
                regression_period: Some(regression_period),
                polynomial_order: Some(polynomial_order),
                regression_offset: Some(regression_offset),
                ndev: Some(ndev),
                equ_from: Some(equ_from),
            },
        );
        let out =
            prb_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "prb".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("values") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.values);
        }
        if output_id.eq_ignore_ascii_case("upper_band") || output_id.eq_ignore_ascii_case("upper") {
            return Ok(out.upper_band);
        }
        if output_id.eq_ignore_ascii_case("lower_band") || output_id.eq_ignore_ascii_case("lower") {
            return Ok(out.lower_band);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "prb".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_qqe_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("qqe", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("qqe", output_id, req.combos, data.len(), |params| {
        let rsi_period = get_usize_param("qqe", params, "rsi_period", 14)?;
        let smoothing_factor = get_usize_param("qqe", params, "smoothing_factor", 5)?;
        let fast_factor = get_f64_param("qqe", params, "fast_factor", 4.236)?;
        let input = QqeInput::from_slice(
            data,
            QqeParams {
                rsi_period: Some(rsi_period),
                smoothing_factor: Some(smoothing_factor),
                fast_factor: Some(fast_factor),
            },
        );
        let out =
            qqe_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "qqe".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("fast") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.fast);
        }
        if output_id.eq_ignore_ascii_case("slow") {
            return Ok(out.slow);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "qqe".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_range_filter_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("range_filter", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "range_filter",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let range_size = get_f64_param("range_filter", params, "range_size", 2.618)?;
            let range_period = get_usize_param("range_filter", params, "range_period", 14)?;
            let smooth_range = get_bool_param("range_filter", params, "smooth_range", true)?;
            let smooth_period = get_usize_param("range_filter", params, "smooth_period", 27)?;
            let input = RangeFilterInput::from_slice(
                data,
                RangeFilterParams {
                    range_size: Some(range_size),
                    range_period: Some(range_period),
                    smooth_range: Some(smooth_range),
                    smooth_period: Some(smooth_period),
                },
            );
            let out = range_filter_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "range_filter".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("filter") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.filter);
            }
            if output_id.eq_ignore_ascii_case("high_band") || output_id.eq_ignore_ascii_case("high")
            {
                return Ok(out.high_band);
            }
            if output_id.eq_ignore_ascii_case("low_band") || output_id.eq_ignore_ascii_case("low") {
                return Ok(out.low_band);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "range_filter".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_rsmk_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (main, compare) = match req.data {
        IndicatorDataRef::CloseVolume { close, volume } => {
            ensure_same_len_2("rsmk", close.len(), volume.len())?;
            (close, volume)
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                "rsmk",
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            (close, volume)
        }
        IndicatorDataRef::Candles { candles, source } => (
            source_type(candles, source.unwrap_or("close")),
            candles.volume.as_slice(),
        ),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "rsmk".to_string(),
                input: IndicatorInputKind::CloseVolume,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64("rsmk", output_id, req.combos, main.len(), |params| {
        let lookback = get_usize_param("rsmk", params, "lookback", 90)?;
        let period = get_usize_param("rsmk", params, "period", 3)?;
        let signal_period = get_usize_param("rsmk", params, "signal_period", 20)?;
        let matype = get_enum_param("rsmk", params, "matype", "ema")?;
        let signal_matype = get_enum_param("rsmk", params, "signal_matype", "ema")?;
        let input = RsmkInput::from_slices(
            main,
            compare,
            RsmkParams {
                lookback: Some(lookback),
                period: Some(period),
                signal_period: Some(signal_period),
                matype: Some(matype),
                signal_matype: Some(signal_matype),
            },
        );
        let out = rsmk_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "rsmk".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("indicator") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.indicator);
        }
        if output_id.eq_ignore_ascii_case("signal") {
            return Ok(out.signal);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "rsmk".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_voss_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("voss", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("voss", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("voss", params, "period", 20)?;
        let predict = get_usize_param("voss", params, "predict", 3)?;
        let bandwidth = get_f64_param("voss", params, "bandwidth", 0.25)?;
        let input = VossInput::from_slice(
            data,
            VossParams {
                period: Some(period),
                predict: Some(predict),
                bandwidth: Some(bandwidth),
            },
        );
        let out = voss_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "voss".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("voss") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.voss);
        }
        if output_id.eq_ignore_ascii_case("filt") || output_id.eq_ignore_ascii_case("filter") {
            return Ok(out.filt);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "voss".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_stc_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("stc", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("stc", output_id, req.combos, data.len(), |params| {
        let fast_period = get_usize_param("stc", params, "fast_period", 23)?;
        let slow_period = get_usize_param("stc", params, "slow_period", 50)?;
        let k_period = get_usize_param("stc", params, "k_period", 10)?;
        let d_period = get_usize_param("stc", params, "d_period", 3)?;
        let input = StcInput::from_slice(
            data,
            StcParams {
                fast_period: Some(fast_period),
                slow_period: Some(slow_period),
                k_period: Some(k_period),
                d_period: Some(d_period),
                fast_ma_type: Some("ema".to_string()),
                slow_ma_type: Some("ema".to_string()),
            },
        );
        let out =
            stc_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "stc".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("value") || output_id.eq_ignore_ascii_case("values") {
            return Ok(out.values);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "stc".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_rvi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("rvi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("rvi", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("rvi", params, "period", 10)?;
        let ma_len = get_usize_param("rvi", params, "ma_len", 14)?;
        let matype = get_usize_param("rvi", params, "matype", 1)?;
        let devtype = get_usize_param("rvi", params, "devtype", 0)?;
        let input = RviInput::from_slice(
            data,
            RviParams {
                period: Some(period),
                ma_len: Some(ma_len),
                matype: Some(matype),
                devtype: Some(devtype),
            },
        );
        let out =
            rvi_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "rvi".to_string(),
                details: e.to_string(),
            })?;
        if output_id.eq_ignore_ascii_case("value") || output_id.eq_ignore_ascii_case("values") {
            return Ok(out.values);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "rvi".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_coppock_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("coppock", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("coppock", output_id, req.combos, data.len(), |params| {
        let short_roc_period = get_usize_param("coppock", params, "short_roc_period", 11)?;
        let long_roc_period = get_usize_param("coppock", params, "long_roc_period", 14)?;
        let ma_period = get_usize_param("coppock", params, "ma_period", 10)?;
        let input = CoppockInput::from_slice(
            data,
            CoppockParams {
                short_roc_period: Some(short_roc_period),
                long_roc_period: Some(long_roc_period),
                ma_period: Some(ma_period),
                ma_type: Some("wma".to_string()),
            },
        );
        let out = coppock_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "coppock".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("value") || output_id.eq_ignore_ascii_case("values") {
            return Ok(out.values);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "coppock".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_correl_hl_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("correl_hl", output_id)?;
    let (high, low) = extract_high_low_input("correl_hl", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("correl_hl", output_id, req.combos, high.len(), |params| {
        let period = get_usize_param("correl_hl", params, "period", 9)?;
        let input = CorrelHlInput::from_slices(
            high,
            low,
            CorrelHlParams {
                period: Some(period),
            },
        );
        let out = correl_hl_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "correl_hl".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_net_myrsi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("net_myrsi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("net_myrsi", output_id, req.combos, data.len(), |params| {
        let period = get_usize_param("net_myrsi", params, "period", 14)?;
        let input = NetMyrsiInput::from_slice(
            data,
            NetMyrsiParams {
                period: Some(period),
            },
        );
        let out = net_myrsi_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "net_myrsi".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("value") || output_id.eq_ignore_ascii_case("values") {
            return Ok(out.values);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "net_myrsi".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_pivot_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("pivot", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("pivot", output_id, req.combos, close.len(), |params| {
        let mode = get_usize_param("pivot", params, "mode", 3)?;
        let input =
            PivotInput::from_slices(high, low, close, open, PivotParams { mode: Some(mode) });
        let out = pivot_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "pivot".to_string(),
                details: e.to_string(),
            }
        })?;
        if output_id.eq_ignore_ascii_case("pp") || output_id.eq_ignore_ascii_case("value") {
            return Ok(out.pp);
        }
        if output_id.eq_ignore_ascii_case("r1") {
            return Ok(out.r1);
        }
        if output_id.eq_ignore_ascii_case("r2") {
            return Ok(out.r2);
        }
        if output_id.eq_ignore_ascii_case("r3") {
            return Ok(out.r3);
        }
        if output_id.eq_ignore_ascii_case("r4") {
            return Ok(out.r4);
        }
        if output_id.eq_ignore_ascii_case("s1") {
            return Ok(out.s1);
        }
        if output_id.eq_ignore_ascii_case("s2") {
            return Ok(out.s2);
        }
        if output_id.eq_ignore_ascii_case("s3") {
            return Ok(out.s3);
        }
        if output_id.eq_ignore_ascii_case("s4") {
            return Ok(out.s4);
        }
        Err(IndicatorDispatchError::UnknownOutput {
            indicator: "pivot".to_string(),
            output: output_id.to_string(),
        })
    })
}

fn compute_wad_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("wad", output_id)?;
    let (_open, high, low, close) = extract_ohlc_full_input("wad", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("wad", output_id, req.combos, close.len(), |_params| {
        let input = WadInput::from_slices(high, low, close);
        let out =
            wad_with_kernel(&input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "wad".to_string(),
                details: e.to_string(),
            })?;
        Ok(out.values)
    })
}

fn ma_data_from_req<'a>(
    indicator: &str,
    data: IndicatorDataRef<'a>,
) -> Result<MaData<'a>, IndicatorDispatchError> {
    match data {
        IndicatorDataRef::Slice { values } => Ok(MaData::Slice(values)),
        IndicatorDataRef::Candles { candles, source } => Ok(MaData::Candles {
            candles,
            source: source.unwrap_or("close"),
        }),
        IndicatorDataRef::Ohlc { close, .. } => Ok(MaData::Slice(close)),
        IndicatorDataRef::Ohlcv { close, .. } => Ok(MaData::Slice(close)),
        IndicatorDataRef::CloseVolume { close, .. } => Ok(MaData::Slice(close)),
        IndicatorDataRef::HighLow { .. } => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::Slice,
        }),
    }
}

fn ma_len_from_req(
    indicator: &str,
    data: IndicatorDataRef<'_>,
) -> Result<usize, IndicatorDispatchError> {
    match data {
        IndicatorDataRef::Slice { values } => Ok(values.len()),
        IndicatorDataRef::Candles { candles, source } => {
            Ok(source_type(candles, source.unwrap_or("close")).len())
        }
        IndicatorDataRef::Ohlc { close, .. } => Ok(close.len()),
        IndicatorDataRef::Ohlcv { close, .. } => Ok(close.len()),
        IndicatorDataRef::CloseVolume { close, .. } => Ok(close.len()),
        IndicatorDataRef::HighLow { .. } => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::Slice,
        }),
    }
}

fn ma_period_for_combo(
    info: &IndicatorInfo,
    params: &[ParamKV<'_>],
) -> Result<usize, IndicatorDispatchError> {
    if let Some(v) = find_param(params, "period") {
        return parse_usize_param_value(info.id, "period", v);
    }
    if let Some(default) = info
        .params
        .iter()
        .find(|p| p.key.eq_ignore_ascii_case("period"))
        .and_then(|p| p.default.as_ref())
    {
        if let ParamValueStatic::Int(v) = default {
            if *v >= 0 {
                return Ok(*v as usize);
            }
        }
    }
    Ok(14)
}

fn convert_ma_params<'a>(
    params: &'a [ParamKV<'a>],
    indicator: &str,
    output_id: &str,
) -> Result<Vec<MaBatchParamKV<'a>>, IndicatorDispatchError> {
    let mut out = Vec::with_capacity(params.len());
    for p in params {
        if p.key.eq_ignore_ascii_case("period") {
            continue;
        }
        if p.key.eq_ignore_ascii_case("output") {
            let selected = match p.value {
                ParamValue::EnumString(v) => v,
                _ => {
                    return Err(IndicatorDispatchError::InvalidParam {
                        indicator: indicator.to_string(),
                        key: "output".to_string(),
                        reason: "expected EnumString".to_string(),
                    })
                }
            };
            if !selected.eq_ignore_ascii_case(output_id) {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: "output".to_string(),
                    reason: format!(
                        "param output '{}' does not match requested output_id '{}'",
                        selected, output_id
                    ),
                });
            }
        }
        let value = match p.value {
            ParamValue::Int(v) => MaBatchParamValue::Int(v),
            ParamValue::Float(v) => {
                if !v.is_finite() {
                    return Err(IndicatorDispatchError::InvalidParam {
                        indicator: indicator.to_string(),
                        key: p.key.to_string(),
                        reason: "expected finite float".to_string(),
                    });
                }
                MaBatchParamValue::Float(v)
            }
            ParamValue::Bool(v) => MaBatchParamValue::Bool(v),
            ParamValue::EnumString(v) => MaBatchParamValue::EnumString(v),
        };
        out.push(MaBatchParamKV { key: p.key, value });
    }
    Ok(out)
}

fn extract_slice_input<'a>(
    indicator: &str,
    data: IndicatorDataRef<'a>,
    default_source: &'a str,
) -> Result<&'a [f64], IndicatorDispatchError> {
    match data {
        IndicatorDataRef::Slice { values } => Ok(values),
        IndicatorDataRef::Candles { candles, source } => {
            Ok(source_type(candles, source.unwrap_or(default_source)))
        }
        IndicatorDataRef::Ohlc { close, .. } => Ok(close),
        IndicatorDataRef::Ohlcv { close, .. } => Ok(close),
        IndicatorDataRef::CloseVolume { close, .. } => Ok(close),
        IndicatorDataRef::HighLow { .. } => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::Slice,
        }),
    }
}

fn extract_ohlc_input<'a>(
    indicator: &str,
    data: IndicatorDataRef<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64]), IndicatorDispatchError> {
    match data {
        IndicatorDataRef::Candles { candles, .. } => Ok((
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )),
        IndicatorDataRef::Ohlc {
            high,
            low,
            close,
            open,
        } => {
            ensure_same_len_4(indicator, open.len(), high.len(), low.len(), close.len())?;
            Ok((high, low, close))
        }
        IndicatorDataRef::Ohlcv {
            high,
            low,
            close,
            open,
            volume,
        } => {
            ensure_same_len_5(
                indicator,
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            Ok((high, low, close))
        }
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::Ohlc,
        }),
    }
}

fn extract_ohlc_full_input<'a>(
    indicator: &str,
    data: IndicatorDataRef<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), IndicatorDispatchError> {
    match data {
        IndicatorDataRef::Candles { candles, .. } => Ok((
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        )),
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => {
            ensure_same_len_4(indicator, open.len(), high.len(), low.len(), close.len())?;
            Ok((open, high, low, close))
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                indicator,
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            Ok((open, high, low, close))
        }
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::Ohlc,
        }),
    }
}

fn extract_ohlcv_full_input<'a>(
    indicator: &str,
    data: IndicatorDataRef<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64], &'a [f64]), IndicatorDispatchError> {
    match data {
        IndicatorDataRef::Candles { candles, .. } => Ok((
            candles.open.as_slice(),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
        )),
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                indicator,
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            Ok((open, high, low, close, volume))
        }
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::Ohlcv,
        }),
    }
}

fn extract_high_low_input<'a>(
    indicator: &str,
    data: IndicatorDataRef<'a>,
) -> Result<(&'a [f64], &'a [f64]), IndicatorDispatchError> {
    match data {
        IndicatorDataRef::Candles { candles, .. } => {
            Ok((candles.high.as_slice(), candles.low.as_slice()))
        }
        IndicatorDataRef::Ohlc {
            high,
            low,
            open,
            close,
        } => {
            ensure_same_len_4(indicator, open.len(), high.len(), low.len(), close.len())?;
            Ok((high, low))
        }
        IndicatorDataRef::Ohlcv {
            high,
            low,
            open,
            close,
            volume,
        } => {
            ensure_same_len_5(
                indicator,
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            Ok((high, low))
        }
        IndicatorDataRef::HighLow { high, low } => {
            ensure_same_len_2(indicator, high.len(), low.len())?;
            Ok((high, low))
        }
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::HighLow,
        }),
    }
}

fn extract_hlcv_input<'a>(
    indicator: &str,
    data: IndicatorDataRef<'a>,
) -> Result<(&'a [f64], &'a [f64], &'a [f64], &'a [f64]), IndicatorDispatchError> {
    match data {
        IndicatorDataRef::Candles { candles, .. } => Ok((
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            candles.volume.as_slice(),
        )),
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                indicator,
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            Ok((high, low, close, volume))
        }
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::Ohlcv,
        }),
    }
}

fn extract_volume_input<'a>(
    indicator: &str,
    data: IndicatorDataRef<'a>,
) -> Result<&'a [f64], IndicatorDispatchError> {
    match data {
        IndicatorDataRef::Slice { values } => Ok(values),
        IndicatorDataRef::Candles { candles, source } => {
            Ok(source_type(candles, source.unwrap_or("volume")))
        }
        IndicatorDataRef::CloseVolume { close, volume } => {
            ensure_same_len_2(indicator, close.len(), volume.len())?;
            Ok(volume)
        }
        IndicatorDataRef::Ohlcv {
            open,
            high,
            low,
            close,
            volume,
        } => {
            ensure_same_len_5(
                indicator,
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            Ok(volume)
        }
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::Slice,
        }),
    }
}

fn extract_close_volume_input<'a>(
    indicator: &str,
    data: IndicatorDataRef<'a>,
    default_close_source: &'a str,
) -> Result<(&'a [f64], &'a [f64]), IndicatorDispatchError> {
    match data {
        IndicatorDataRef::CloseVolume { close, volume } => {
            ensure_same_len_2(indicator, close.len(), volume.len())?;
            Ok((close, volume))
        }
        IndicatorDataRef::Ohlcv {
            close,
            volume,
            open,
            high,
            low,
        } => {
            ensure_same_len_5(
                indicator,
                open.len(),
                high.len(),
                low.len(),
                close.len(),
                volume.len(),
            )?;
            Ok((close, volume))
        }
        IndicatorDataRef::Candles { candles, source } => {
            let close = source_type(candles, source.unwrap_or(default_close_source));
            let volume = candles.volume.as_slice();
            ensure_same_len_2(indicator, close.len(), volume.len())?;
            Ok((close, volume))
        }
        _ => Err(IndicatorDispatchError::MissingRequiredInput {
            indicator: indicator.to_string(),
            input: IndicatorInputKind::CloseVolume,
        }),
    }
}

fn f64_output(output_id: &str, rows: usize, cols: usize, values: Vec<f64>) -> IndicatorBatchOutput {
    IndicatorBatchOutput {
        output_id: output_id.to_string(),
        rows,
        cols,
        values_f64: Some(values),
        values_i32: None,
        values_bool: None,
    }
}

fn bool_output(
    output_id: &str,
    rows: usize,
    cols: usize,
    values: Vec<bool>,
) -> IndicatorBatchOutput {
    IndicatorBatchOutput {
        output_id: output_id.to_string(),
        rows,
        cols,
        values_f64: None,
        values_i32: None,
        values_bool: Some(values),
    }
}

fn expect_value_output(indicator: &str, output_id: &str) -> Result<(), IndicatorDispatchError> {
    if output_id.eq_ignore_ascii_case("value") {
        return Ok(());
    }
    Err(IndicatorDispatchError::UnknownOutput {
        indicator: indicator.to_string(),
        output: output_id.to_string(),
    })
}

fn ensure_len(indicator: &str, expected: usize, got: usize) -> Result<(), IndicatorDispatchError> {
    if expected == got {
        return Ok(());
    }
    Err(IndicatorDispatchError::DataLengthMismatch {
        details: format!("{indicator}: expected output length {expected}, got {got}"),
    })
}

fn ensure_same_len_2(indicator: &str, a: usize, b: usize) -> Result<(), IndicatorDispatchError> {
    if a == b {
        return Ok(());
    }
    Err(IndicatorDispatchError::DataLengthMismatch {
        details: format!("{indicator}: expected equal lengths, got {a} and {b}"),
    })
}

fn ensure_same_len_3(
    indicator: &str,
    a: usize,
    b: usize,
    c: usize,
) -> Result<(), IndicatorDispatchError> {
    if a == b && b == c {
        return Ok(());
    }
    Err(IndicatorDispatchError::DataLengthMismatch {
        details: format!("{indicator}: expected equal lengths, got {a}, {b}, {c}"),
    })
}

fn ensure_same_len_4(
    indicator: &str,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) -> Result<(), IndicatorDispatchError> {
    if a == b && b == c && c == d {
        return Ok(());
    }
    Err(IndicatorDispatchError::DataLengthMismatch {
        details: format!("{indicator}: expected equal lengths, got {a}, {b}, {c}, {d}"),
    })
}

fn ensure_same_len_5(
    indicator: &str,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    e: usize,
) -> Result<(), IndicatorDispatchError> {
    if a == b && b == c && c == d && d == e {
        return Ok(());
    }
    Err(IndicatorDispatchError::DataLengthMismatch {
        details: format!("{indicator}: expected equal lengths, got {a}, {b}, {c}, {d}, {e}"),
    })
}

fn has_key(params: &[ParamKV<'_>], key: &str) -> bool {
    params.iter().any(|kv| kv.key.eq_ignore_ascii_case(key))
}

fn find_param<'a>(params: &'a [ParamKV<'a>], key: &str) -> Option<&'a ParamValue<'a>> {
    params
        .iter()
        .rev()
        .find(|kv| kv.key.eq_ignore_ascii_case(key))
        .map(|kv| &kv.value)
}

fn get_usize_param(
    indicator: &str,
    params: &[ParamKV<'_>],
    key: &str,
    default: usize,
) -> Result<usize, IndicatorDispatchError> {
    match find_param(params, key) {
        Some(v) => parse_usize_param_value(indicator, key, v),
        None => Ok(default),
    }
}

fn get_usize_param_with_aliases(
    indicator: &str,
    params: &[ParamKV<'_>],
    keys: &[&str],
    default: usize,
) -> Result<usize, IndicatorDispatchError> {
    for key in keys {
        if let Some(v) = find_param(params, key) {
            return parse_usize_param_value(indicator, key, v);
        }
    }
    Ok(default)
}

fn get_f64_param_with_aliases(
    indicator: &str,
    params: &[ParamKV<'_>],
    keys: &[&str],
    default: f64,
) -> Result<f64, IndicatorDispatchError> {
    for key in keys {
        match find_param(params, key) {
            Some(ParamValue::Int(v)) => return Ok(*v as f64),
            Some(ParamValue::Float(v)) => {
                if v.is_finite() {
                    return Ok(*v);
                }
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected finite float".to_string(),
                });
            }
            Some(_) => {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected Int or Float".to_string(),
                });
            }
            None => continue,
        }
    }
    Ok(default)
}

fn parse_usize_param_value(
    indicator: &str,
    key: &str,
    value: &ParamValue<'_>,
) -> Result<usize, IndicatorDispatchError> {
    match value {
        ParamValue::Int(v) => {
            if *v < 0 {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected integer >= 0".to_string(),
                });
            }
            Ok(*v as usize)
        }
        ParamValue::Float(v) => {
            if !v.is_finite() {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected finite number".to_string(),
                });
            }
            if *v < 0.0 {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected number >= 0".to_string(),
                });
            }
            let r = v.round();
            if (*v - r).abs() > 1e-9 {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected integer value".to_string(),
                });
            }
            Ok(r as usize)
        }
        _ => Err(IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: key.to_string(),
            reason: "expected Int or Float".to_string(),
        }),
    }
}

fn get_f64_param(
    indicator: &str,
    params: &[ParamKV<'_>],
    key: &str,
    default: f64,
) -> Result<f64, IndicatorDispatchError> {
    match find_param(params, key) {
        Some(ParamValue::Int(v)) => Ok(*v as f64),
        Some(ParamValue::Float(v)) => {
            if v.is_finite() {
                Ok(*v)
            } else {
                Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected finite float".to_string(),
                })
            }
        }
        Some(_) => Err(IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: key.to_string(),
            reason: "expected Int or Float".to_string(),
        }),
        None => Ok(default),
    }
}

fn get_bool_param(
    indicator: &str,
    params: &[ParamKV<'_>],
    key: &str,
    default: bool,
) -> Result<bool, IndicatorDispatchError> {
    match find_param(params, key) {
        Some(ParamValue::Bool(v)) => Ok(*v),
        Some(ParamValue::Int(v)) => match *v {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(IndicatorDispatchError::InvalidParam {
                indicator: indicator.to_string(),
                key: key.to_string(),
                reason: "expected Bool or Int(0/1)".to_string(),
            }),
        },
        Some(_) => Err(IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: key.to_string(),
            reason: "expected Bool".to_string(),
        }),
        None => Ok(default),
    }
}

fn get_enum_string_param<'a>(
    indicator: &str,
    params: &'a [ParamKV<'a>],
    key: &str,
    default: &'a str,
) -> Result<&'a str, IndicatorDispatchError> {
    match find_param(params, key) {
        Some(ParamValue::EnumString(v)) => Ok(v),
        Some(_) => Err(IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: key.to_string(),
            reason: "expected EnumString".to_string(),
        }),
        None => Ok(default),
    }
}

fn get_i32_param(
    indicator: &str,
    params: &[ParamKV<'_>],
    key: &str,
    default: i32,
) -> Result<i32, IndicatorDispatchError> {
    match find_param(params, key) {
        Some(ParamValue::Int(v)) => {
            if *v < i32::MIN as i64 || *v > i32::MAX as i64 {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "integer out of i32 range".to_string(),
                });
            }
            Ok(*v as i32)
        }
        Some(ParamValue::Float(v)) => {
            if !v.is_finite() {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected finite number".to_string(),
                });
            }
            let r = v.round();
            if (*v - r).abs() > 1e-9 || r < i32::MIN as f64 || r > i32::MAX as f64 {
                return Err(IndicatorDispatchError::InvalidParam {
                    indicator: indicator.to_string(),
                    key: key.to_string(),
                    reason: "expected i32-compatible whole number".to_string(),
                });
            }
            Ok(r as i32)
        }
        Some(_) => Err(IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: key.to_string(),
            reason: "expected Int or Float".to_string(),
        }),
        None => Ok(default),
    }
}

fn get_enum_param(
    indicator: &str,
    params: &[ParamKV<'_>],
    key: &str,
    default: &str,
) -> Result<String, IndicatorDispatchError> {
    match find_param(params, key) {
        Some(ParamValue::EnumString(v)) => Ok((*v).to_string()),
        Some(_) => Err(IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: key.to_string(),
            reason: "expected EnumString".to_string(),
        }),
        None => Ok(default.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::ad::{ad_with_kernel, AdInput, AdParams};
    use crate::indicators::adx::{adx_with_kernel, AdxInput, AdxParams};
    use crate::indicators::ao::{ao_with_kernel, AoInput, AoParams};
    use crate::indicators::apo::{apo_with_kernel, ApoInput, ApoParams};
    use crate::indicators::cg::{cg_with_kernel, CgInput, CgParams};
    use crate::indicators::cmo::{cmo_with_kernel, CmoInput, CmoParams};
    use crate::indicators::deviation::{deviation_with_kernel, DeviationInput, DeviationParams};
    use crate::indicators::dx::{
        dx_batch_with_kernel, dx_with_kernel, DxBatchRange, DxInput, DxParams,
    };
    use crate::indicators::efi::{efi_with_kernel, EfiInput, EfiParams};
    use crate::indicators::fosc::{fosc_with_kernel, FoscInput, FoscParams};
    use crate::indicators::ift_rsi::{ift_rsi_with_kernel, IftRsiInput, IftRsiParams};
    use crate::indicators::kvo::{kvo_with_kernel, KvoInput, KvoParams};
    use crate::indicators::linearreg_angle::{
        linearreg_angle_with_kernel, Linearreg_angleInput, Linearreg_angleParams,
    };
    use crate::indicators::linearreg_intercept::{
        linearreg_intercept_with_kernel, LinearRegInterceptInput, LinearRegInterceptParams,
    };
    use crate::indicators::linearreg_slope::{
        linearreg_slope_with_kernel, LinearRegSlopeInput, LinearRegSlopeParams,
    };
    use crate::indicators::macd::{macd_with_kernel, MacdInput, MacdParams};
    use crate::indicators::mean_ad::{mean_ad_with_kernel, MeanAdInput, MeanAdParams};
    use crate::indicators::medprice::{medprice_with_kernel, MedpriceInput, MedpriceParams};
    use crate::indicators::mfi::{
        mfi_batch_with_kernel, mfi_with_kernel, MfiBatchRange, MfiInput, MfiParams,
    };
    use crate::indicators::moving_averages::ma::MaData;
    use crate::indicators::moving_averages::ma_batch::{
        ma_batch_with_kernel_and_typed_params, MaBatchParamKV, MaBatchParamValue,
    };
    use crate::indicators::natr::{natr_with_kernel, NatrInput, NatrParams};
    use crate::indicators::percentile_nearest_rank::{
        percentile_nearest_rank_with_kernel, PercentileNearestRankInput,
        PercentileNearestRankParams,
    };
    use crate::indicators::ppo::{ppo_with_kernel, PpoInput, PpoParams};
    use crate::indicators::pvi::{pvi_with_kernel, PviInput, PviParams};
    use crate::indicators::registry::{list_indicators, IndicatorParamKind};
    use crate::indicators::trix::{
        trix_batch_with_kernel, trix_with_kernel, TrixBatchRange, TrixInput, TrixParams,
    };
    use crate::indicators::ttm_trend::{ttm_trend_with_kernel, TtmTrendInput, TtmTrendParams};
    use crate::indicators::vpci::{vpci_with_kernel, VpciInput, VpciParams};
    use crate::indicators::yang_zhang_volatility::{
        yang_zhang_volatility_with_kernel, YangZhangVolatilityInput, YangZhangVolatilityParams,
    };
    use crate::indicators::zscore::{zscore_with_kernel, ZscoreInput, ZscoreParams};
    use crate::utilities::enums::Kernel;
    use std::time::Instant;

    fn sample_series() -> Vec<f64> {
        (1..=64).map(|v| v as f64).collect()
    }

    fn sample_ohlc() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let open: Vec<f64> = (0..128).map(|i| 100.0 + (i as f64 * 0.1)).collect();
        let high: Vec<f64> = open.iter().map(|v| v + 1.25).collect();
        let low: Vec<f64> = open.iter().map(|v| v - 1.1).collect();
        let close: Vec<f64> = open.iter().map(|v| v + 0.3).collect();
        (open, high, low, close)
    }

    fn sample_candles() -> crate::utilities::data_loader::Candles {
        let (open, high, low, close) = sample_ohlc();
        let volume: Vec<f64> = (0..close.len()).map(|i| 1000.0 + (i as f64)).collect();
        let timestamp: Vec<i64> = (0..close.len()).map(|i| i as i64).collect();
        crate::utilities::data_loader::Candles::new(timestamp, open, high, low, close, volume)
    }

    fn assert_series_eq(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for i in 0..actual.len() {
            let a = actual[i];
            let b = expected[i];
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert!(
                (a - b).abs() <= tol,
                "mismatch at index {i}: actual={a}, expected={b}, tol={tol}"
            );
        }
    }

    #[test]
    fn unknown_indicator_is_rejected() {
        let data = sample_series();
        let req = IndicatorBatchRequest {
            indicator_id: "not_real",
            output_id: None,
            data: IndicatorDataRef::Slice { values: &data },
            combos: &[],
            kernel: Kernel::Auto,
        };
        let err = compute_cpu_batch(req).unwrap_err();
        assert!(matches!(
            err,
            IndicatorDispatchError::UnknownIndicator { .. }
        ));
    }

    #[test]
    fn bucket_b_ma_indicator_is_supported() {
        let data = sample_series();
        let combos = [IndicatorParamSet { params: &[] }];
        let req = IndicatorBatchRequest {
            indicator_id: "mama",
            output_id: Some("mama"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, data.len());
        assert!(out.values_f64.is_some());
    }

    #[test]
    fn strict_mode_rejects_convenience_mfi_ohlcv() {
        let (open, high, low, close) = sample_ohlc();
        let volume: Vec<f64> = (0..close.len()).map(|i| 1200.0 + (i as f64)).collect();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "mfi",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlcv {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let err = compute_cpu_batch_strict(req).unwrap_err();
        match err {
            IndicatorDispatchError::MissingRequiredInput { indicator, input } => {
                assert_eq!(indicator, "mfi");
                assert_eq!(input, IndicatorInputKind::CloseVolume);
            }
            other => panic!("expected MissingRequiredInput, got {other:?}"),
        }
    }

    #[test]
    fn strict_mode_accepts_precomputed_mfi_close_volume() {
        let (_open, high, low, close) = sample_ohlc();
        let volume: Vec<f64> = (0..close.len())
            .map(|i| 1000.0 + (i as f64 * 2.0))
            .collect();
        let typical: Vec<f64> = high
            .iter()
            .zip(&low)
            .zip(&close)
            .map(|((h, l), c)| (h + l + c) / 3.0)
            .collect();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "mfi",
            output_id: Some("value"),
            data: IndicatorDataRef::CloseVolume {
                close: &typical,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let strict = compute_cpu_batch_strict(req).unwrap();
        let input = MfiInput::from_slices(&typical, &volume, MfiParams { period: Some(14) });
        let direct = mfi_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        assert_series_eq(strict.values_f64.as_ref().unwrap(), &direct, 1e-12);
    }

    #[test]
    fn strict_mode_rejects_ao_high_low_and_requires_slice() {
        let (_open, high, low, _close) = sample_ohlc();
        let combo = [
            ParamKV {
                key: "short_period",
                value: ParamValue::Int(5),
            },
            ParamKV {
                key: "long_period",
                value: ParamValue::Int(34),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "ao",
            output_id: Some("value"),
            data: IndicatorDataRef::HighLow {
                high: &high,
                low: &low,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let err = compute_cpu_batch_strict(req).unwrap_err();
        match err {
            IndicatorDispatchError::MissingRequiredInput { indicator, input } => {
                assert_eq!(indicator, "ao");
                assert_eq!(input, IndicatorInputKind::Slice);
            }
            other => panic!("expected MissingRequiredInput, got {other:?}"),
        }
    }

    #[test]
    fn strict_mode_rejects_ttm_trend_ohlc_and_requires_candles() {
        let (open, high, low, close) = sample_ohlc();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(5),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "ttm_trend",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let err = compute_cpu_batch_strict(req).unwrap_err();
        match err {
            IndicatorDispatchError::MissingRequiredInput { indicator, input } => {
                assert_eq!(indicator, "ttm_trend");
                assert_eq!(input, IndicatorInputKind::Candles);
            }
            other => panic!("expected MissingRequiredInput, got {other:?}"),
        }
    }

    #[test]
    fn strict_mode_accepts_ttm_trend_candles() {
        let candles = sample_candles();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(5),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "ttm_trend",
            output_id: Some("value"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("hl2"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let strict = compute_cpu_batch_strict(req).unwrap();
        let input = TtmTrendInput::from_slices(
            candles.hl2.as_slice(),
            candles.close.as_slice(),
            TtmTrendParams { period: Some(5) },
        );
        let direct = ttm_trend_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = strict.values_bool.unwrap();
        assert_eq!(got, direct);
    }

    #[test]
    fn rsi_cpu_batch_smoke() {
        let data = sample_series();
        let combo_1 = [ParamKV {
            key: "period",
            value: ParamValue::Int(7),
        }];
        let combo_2 = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [
            IndicatorParamSet { params: &combo_1 },
            IndicatorParamSet { params: &combo_2 },
        ];
        let req = IndicatorBatchRequest {
            indicator_id: "rsi",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        assert_eq!(out.output_id, "value");
        assert_eq!(out.rows, 2);
        assert_eq!(out.cols, data.len());
        assert_eq!(out.values_f64.as_ref().map(Vec::len), Some(2 * data.len()));
    }

    #[test]
    fn ma_dispatch_regression_sma_matches_existing_ma_batch_api() {
        let data = sample_series();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let dispatch = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "sma",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = ma_batch_with_kernel_and_typed_params(
            "sma",
            MaData::Slice(&data),
            (14, 14, 0),
            Kernel::Auto,
            &[],
        )
        .unwrap();
        assert_eq!(dispatch.rows, direct.rows);
        assert_eq!(dispatch.cols, direct.cols);
        assert_series_eq(dispatch.values_f64.as_ref().unwrap(), &direct.values, 1e-12);
    }

    #[test]
    fn ma_dispatch_sma_period_sweep_matches_direct_batch() {
        let data = sample_series();
        let combo_1 = [ParamKV {
            key: "period",
            value: ParamValue::Int(5),
        }];
        let combo_2 = [ParamKV {
            key: "period",
            value: ParamValue::Int(7),
        }];
        let combo_3 = [ParamKV {
            key: "period",
            value: ParamValue::Int(9),
        }];
        let combos = [
            IndicatorParamSet { params: &combo_1 },
            IndicatorParamSet { params: &combo_2 },
            IndicatorParamSet { params: &combo_3 },
        ];
        let dispatch = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "sma",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = ma_batch_with_kernel_and_typed_params(
            "sma",
            MaData::Slice(&data),
            (5, 9, 2),
            Kernel::Auto,
            &[],
        )
        .unwrap();
        assert_eq!(dispatch.rows, direct.rows);
        assert_eq!(dispatch.cols, direct.cols);
        assert_series_eq(dispatch.values_f64.as_ref().unwrap(), &direct.values, 1e-12);
    }

    #[test]
    fn mfi_dispatch_period_sweep_matches_direct_batch() {
        let (_open, high, low, close) = sample_ohlc();
        let volume: Vec<f64> = (0..close.len())
            .map(|i| 1000.0 + (i as f64 * 2.0))
            .collect();
        let typical: Vec<f64> = high
            .iter()
            .zip(&low)
            .zip(&close)
            .map(|((h, l), c)| (h + l + c) / 3.0)
            .collect();
        let combo_1 = [ParamKV {
            key: "period",
            value: ParamValue::Int(5),
        }];
        let combo_2 = [ParamKV {
            key: "period",
            value: ParamValue::Int(7),
        }];
        let combo_3 = [ParamKV {
            key: "period",
            value: ParamValue::Int(9),
        }];
        let combos = [
            IndicatorParamSet { params: &combo_1 },
            IndicatorParamSet { params: &combo_2 },
            IndicatorParamSet { params: &combo_3 },
        ];
        let dispatch = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "mfi",
            output_id: Some("value"),
            data: IndicatorDataRef::CloseVolume {
                close: &typical,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();
        let direct = mfi_batch_with_kernel(
            &typical,
            &volume,
            &MfiBatchRange { period: (5, 9, 2) },
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(dispatch.rows, direct.rows);
        assert_eq!(dispatch.cols, direct.cols);
        assert_series_eq(dispatch.values_f64.as_ref().unwrap(), &direct.values, 1e-12);
    }

    #[test]
    fn dx_dispatch_period_sweep_keeps_requested_row_order() {
        let (open, high, low, close) = sample_ohlc();
        let combo_1 = [ParamKV {
            key: "period",
            value: ParamValue::Int(9),
        }];
        let combo_2 = [ParamKV {
            key: "period",
            value: ParamValue::Int(7),
        }];
        let combo_3 = [ParamKV {
            key: "period",
            value: ParamValue::Int(5),
        }];
        let combos = [
            IndicatorParamSet { params: &combo_1 },
            IndicatorParamSet { params: &combo_2 },
            IndicatorParamSet { params: &combo_3 },
        ];
        let dispatch = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "dx",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();
        let direct = dx_batch_with_kernel(
            &high,
            &low,
            &close,
            &DxBatchRange { period: (9, 5, 2) },
            Kernel::Auto,
        )
        .unwrap();
        let direct_periods: Vec<usize> = direct
            .combos
            .iter()
            .map(|combo| combo.period.unwrap_or(14))
            .collect();
        let period_to_row: std::collections::HashMap<usize, usize> = direct_periods
            .iter()
            .copied()
            .enumerate()
            .map(|(row, period)| (period, row))
            .collect();
        let requested = [9usize, 7usize, 5usize];
        let mut expected = Vec::with_capacity(requested.len() * direct.cols);
        for period in requested {
            let row = period_to_row[&period];
            let start = row * direct.cols;
            let end = start + direct.cols;
            expected.extend_from_slice(&direct.values[start..end]);
        }
        assert_eq!(dispatch.rows, requested.len());
        assert_eq!(dispatch.cols, direct.cols);
        assert_series_eq(dispatch.values_f64.as_ref().unwrap(), &expected, 1e-12);
    }

    #[test]
    fn ma_dispatch_regression_alma_typed_params_match_existing_ma_batch_api() {
        let data = sample_series();
        let combo = [
            ParamKV {
                key: "period",
                value: ParamValue::Int(14),
            },
            ParamKV {
                key: "offset",
                value: ParamValue::Float(0.87),
            },
            ParamKV {
                key: "sigma",
                value: ParamValue::Float(5.5),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let dispatch = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "alma",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let typed = [
            MaBatchParamKV {
                key: "offset",
                value: MaBatchParamValue::Float(0.87),
            },
            MaBatchParamKV {
                key: "sigma",
                value: MaBatchParamValue::Float(5.5),
            },
        ];
        let direct = ma_batch_with_kernel_and_typed_params(
            "alma",
            MaData::Slice(&data),
            (14, 14, 0),
            Kernel::Auto,
            &typed,
        )
        .unwrap();
        assert_eq!(dispatch.rows, direct.rows);
        assert_eq!(dispatch.cols, direct.cols);
        assert_series_eq(dispatch.values_f64.as_ref().unwrap(), &direct.values, 1e-12);
    }

    #[test]
    fn macd_signal_output_matches_direct() {
        let data = sample_series();
        let combo_1 = [
            ParamKV {
                key: "fast_period",
                value: ParamValue::Int(8),
            },
            ParamKV {
                key: "slow_period",
                value: ParamValue::Int(21),
            },
            ParamKV {
                key: "signal_period",
                value: ParamValue::Int(5),
            },
        ];
        let combo_2 = [
            ParamKV {
                key: "fast_period",
                value: ParamValue::Int(12),
            },
            ParamKV {
                key: "slow_period",
                value: ParamValue::Int(26),
            },
            ParamKV {
                key: "signal_period",
                value: ParamValue::Int(9),
            },
        ];
        let combos = [
            IndicatorParamSet { params: &combo_1 },
            IndicatorParamSet { params: &combo_2 },
        ];
        let req = IndicatorBatchRequest {
            indicator_id: "macd",
            output_id: Some("signal"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let matrix = out.values_f64.unwrap();
        for (row, combo) in combos.iter().enumerate() {
            let fast = match combo.params[0].value {
                ParamValue::Int(v) => v as usize,
                _ => unreachable!(),
            };
            let slow = match combo.params[1].value {
                ParamValue::Int(v) => v as usize,
                _ => unreachable!(),
            };
            let signal = match combo.params[2].value {
                ParamValue::Int(v) => v as usize,
                _ => unreachable!(),
            };
            let input = MacdInput::from_slice(
                &data,
                MacdParams {
                    fast_period: Some(fast),
                    slow_period: Some(slow),
                    signal_period: Some(signal),
                    ma_type: Some("ema".to_string()),
                },
            );
            let direct = macd_with_kernel(&input, Kernel::Auto.to_non_batch())
                .unwrap()
                .signal;
            let start = row * out.cols;
            let end = start + out.cols;
            assert_series_eq(&matrix[start..end], direct.as_slice(), 1e-12);
        }
    }

    #[test]
    fn adx_output_matches_direct() {
        let (open, high, low, close) = sample_ohlc();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "adx",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let matrix = out.values_f64.unwrap();
        let input = AdxInput::from_slices(&high, &low, &close, AdxParams { period: Some(14) });
        let direct = adx_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        assert_series_eq(&matrix, &direct, 1e-12);
    }

    #[test]
    fn cmo_output_matches_direct() {
        let data = sample_series();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "cmo",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = CmoInput::from_slice(&data, CmoParams { period: Some(14) });
        let direct = cmo_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn ppo_output_matches_direct() {
        let data = sample_series();
        let combo = [
            ParamKV {
                key: "fast_period",
                value: ParamValue::Int(12),
            },
            ParamKV {
                key: "slow_period",
                value: ParamValue::Int(26),
            },
            ParamKV {
                key: "ma_type",
                value: ParamValue::EnumString("sma"),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "ppo",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = PpoInput::from_slice(
            &data,
            PpoParams {
                fast_period: Some(12),
                slow_period: Some(26),
                ma_type: Some("sma".to_string()),
            },
        );
        let direct = ppo_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn apo_output_matches_direct() {
        let data = sample_series();
        let combo = [
            ParamKV {
                key: "short_period",
                value: ParamValue::Int(10),
            },
            ParamKV {
                key: "long_period",
                value: ParamValue::Int(20),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "apo",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = ApoInput::from_slice(
            &data,
            ApoParams {
                short_period: Some(10),
                long_period: Some(20),
            },
        );
        let direct = apo_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn natr_output_matches_direct() {
        let (open, high, low, close) = sample_ohlc();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "natr",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = NatrInput::from_slices(&high, &low, &close, NatrParams { period: Some(14) });
        let direct = natr_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn ad_output_matches_direct() {
        let (open, high, low, close) = sample_ohlc();
        let volume: Vec<f64> = (0..close.len())
            .map(|i| 1000.0 + (i as f64 * 3.0))
            .collect();
        let combos = [IndicatorParamSet { params: &[] }];
        let req = IndicatorBatchRequest {
            indicator_id: "ad",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlcv {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = AdInput::from_slices(&high, &low, &close, &volume, AdParams::default());
        let direct = ad_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn ao_output_matches_direct() {
        let (open, high, low, close) = sample_ohlc();
        let combo = [
            ParamKV {
                key: "short_period",
                value: ParamValue::Int(5),
            },
            ParamKV {
                key: "long_period",
                value: ParamValue::Int(34),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "ao",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let source: Vec<f64> = high.iter().zip(&low).map(|(h, l)| 0.5 * (h + l)).collect();
        let input = AoInput::from_slice(
            &source,
            AoParams {
                short_period: Some(5),
                long_period: Some(34),
            },
        );
        let direct = ao_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn pvi_output_matches_direct() {
        let data = sample_series();
        let volume: Vec<f64> = (0..data.len()).map(|i| 900.0 + (i as f64 * 5.0)).collect();
        let combo = [ParamKV {
            key: "initial_value",
            value: ParamValue::Float(1000.0),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "pvi",
            output_id: Some("value"),
            data: IndicatorDataRef::CloseVolume {
                close: &data,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = PviInput::from_slices(
            &data,
            &volume,
            PviParams {
                initial_value: Some(1000.0),
            },
        );
        let direct = pvi_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn efi_output_matches_direct() {
        let data = sample_series();
        let volume: Vec<f64> = (0..data.len()).map(|i| 1000.0 + (i as f64 * 4.0)).collect();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(13),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "efi",
            output_id: Some("value"),
            data: IndicatorDataRef::CloseVolume {
                close: &data,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = EfiInput::from_slices(&data, &volume, EfiParams { period: Some(13) });
        let direct = efi_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn mfi_output_matches_direct() {
        let (open, high, low, close) = sample_ohlc();
        let volume: Vec<f64> = (0..close.len()).map(|i| 900.0 + (i as f64 * 6.0)).collect();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "mfi",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlcv {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let typical_price: Vec<f64> = high
            .iter()
            .zip(&low)
            .zip(&close)
            .map(|((h, l), c)| (h + l + c) / 3.0)
            .collect();
        let input = MfiInput::from_slices(&typical_price, &volume, MfiParams { period: Some(14) });
        let direct = mfi_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn mfi_non_sweep_fallback_rows_match_direct() {
        let (open, high, low, close) = sample_ohlc();
        let volume: Vec<f64> = (0..close.len()).map(|i| 950.0 + (i as f64 * 5.0)).collect();
        let combo_1 = [ParamKV {
            key: "period",
            value: ParamValue::Int(5),
        }];
        let combo_2 = [ParamKV {
            key: "period",
            value: ParamValue::Int(9),
        }];
        let combo_3 = [ParamKV {
            key: "period",
            value: ParamValue::Int(8),
        }];
        let combos = [
            IndicatorParamSet { params: &combo_1 },
            IndicatorParamSet { params: &combo_2 },
            IndicatorParamSet { params: &combo_3 },
        ];
        let req = IndicatorBatchRequest {
            indicator_id: "mfi",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlcv {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let matrix = out.values_f64.unwrap();
        let typical_price: Vec<f64> = high
            .iter()
            .zip(&low)
            .zip(&close)
            .map(|((h, l), c)| (h + l + c) / 3.0)
            .collect();
        for (row, period) in [5usize, 9usize, 8usize].iter().enumerate() {
            let input = MfiInput::from_slices(
                &typical_price,
                &volume,
                MfiParams {
                    period: Some(*period),
                },
            );
            let direct = mfi_with_kernel(&input, Kernel::Auto.to_non_batch())
                .unwrap()
                .values;
            let start = row * close.len();
            let end = start + close.len();
            assert_series_eq(&matrix[start..end], &direct, 1e-12);
        }
    }

    #[test]
    fn kvo_output_matches_direct() {
        let (open, high, low, close) = sample_ohlc();
        let volume: Vec<f64> = (0..close.len())
            .map(|i| 1200.0 + (i as f64 * 5.0))
            .collect();
        let combo = [
            ParamKV {
                key: "short_period",
                value: ParamValue::Int(2),
            },
            ParamKV {
                key: "long_period",
                value: ParamValue::Int(5),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "kvo",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlcv {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = KvoInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            KvoParams {
                short_period: Some(2),
                long_period: Some(5),
            },
        );
        let direct = kvo_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn dx_output_matches_direct() {
        let (open, high, low, close) = sample_ohlc();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "dx",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = DxInput::from_hlc_slices(&high, &low, &close, DxParams { period: Some(14) });
        let direct = dx_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn dx_non_sweep_fallback_rows_match_direct() {
        let (open, high, low, close) = sample_ohlc();
        let combo_1 = [ParamKV {
            key: "period",
            value: ParamValue::Int(9),
        }];
        let combo_2 = [ParamKV {
            key: "period",
            value: ParamValue::Int(5),
        }];
        let combo_3 = [ParamKV {
            key: "period",
            value: ParamValue::Int(8),
        }];
        let combos = [
            IndicatorParamSet { params: &combo_1 },
            IndicatorParamSet { params: &combo_2 },
            IndicatorParamSet { params: &combo_3 },
        ];
        let req = IndicatorBatchRequest {
            indicator_id: "dx",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let matrix = out.values_f64.unwrap();
        for (row, period) in [9usize, 5usize, 8usize].iter().enumerate() {
            let input = DxInput::from_hlc_slices(
                &high,
                &low,
                &close,
                DxParams {
                    period: Some(*period),
                },
            );
            let direct = dx_with_kernel(&input, Kernel::Auto.to_non_batch())
                .unwrap()
                .values;
            let start = row * close.len();
            let end = start + close.len();
            assert_series_eq(&matrix[start..end], &direct, 1e-12);
        }
    }

    #[test]
    fn trix_dispatch_period_sweep_keeps_requested_row_order() {
        let data = sample_series();
        let combo_1 = [ParamKV {
            key: "period",
            value: ParamValue::Int(9),
        }];
        let combo_2 = [ParamKV {
            key: "period",
            value: ParamValue::Int(7),
        }];
        let combo_3 = [ParamKV {
            key: "period",
            value: ParamValue::Int(5),
        }];
        let combos = [
            IndicatorParamSet { params: &combo_1 },
            IndicatorParamSet { params: &combo_2 },
            IndicatorParamSet { params: &combo_3 },
        ];
        let dispatch = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "trix",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct =
            trix_batch_with_kernel(&data, &TrixBatchRange { period: (9, 5, 2) }, Kernel::Auto)
                .unwrap();
        let direct_periods: Vec<usize> = direct
            .combos
            .iter()
            .map(|combo| combo.period.unwrap_or(18))
            .collect();
        let period_to_row: std::collections::HashMap<usize, usize> = direct_periods
            .iter()
            .copied()
            .enumerate()
            .map(|(row, period)| (period, row))
            .collect();
        let requested = [9usize, 7usize, 5usize];
        let mut expected = Vec::with_capacity(requested.len() * direct.cols);
        for period in requested {
            let row = period_to_row[&period];
            let start = row * direct.cols;
            let end = start + direct.cols;
            expected.extend_from_slice(&direct.values[start..end]);
        }
        assert_eq!(dispatch.rows, requested.len());
        assert_eq!(dispatch.cols, direct.cols);
        assert_series_eq(dispatch.values_f64.as_ref().unwrap(), &expected, 1e-12);
    }

    #[test]
    fn trix_non_sweep_fallback_rows_match_direct() {
        let data = sample_series();
        let combo_1 = [ParamKV {
            key: "period",
            value: ParamValue::Int(9),
        }];
        let combo_2 = [ParamKV {
            key: "period",
            value: ParamValue::Int(5),
        }];
        let combo_3 = [ParamKV {
            key: "period",
            value: ParamValue::Int(8),
        }];
        let combos = [
            IndicatorParamSet { params: &combo_1 },
            IndicatorParamSet { params: &combo_2 },
            IndicatorParamSet { params: &combo_3 },
        ];
        let out = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "trix",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();
        let matrix = out.values_f64.unwrap();
        for (row, period) in [9usize, 5usize, 8usize].iter().enumerate() {
            let input = TrixInput::from_slice(
                &data,
                TrixParams {
                    period: Some(*period),
                },
            );
            let direct = trix_with_kernel(&input, Kernel::Auto.to_non_batch())
                .unwrap()
                .values;
            let start = row * data.len();
            let end = start + data.len();
            assert_series_eq(&matrix[start..end], &direct, 1e-12);
        }
    }

    #[test]
    fn ift_rsi_output_matches_direct() {
        let data = sample_series();
        let combo = [
            ParamKV {
                key: "rsi_period",
                value: ParamValue::Int(6),
            },
            ParamKV {
                key: "wma_period",
                value: ParamValue::Int(10),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "ift_rsi",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = IftRsiInput::from_slice(
            &data,
            IftRsiParams {
                rsi_period: Some(6),
                wma_period: Some(10),
            },
        );
        let direct = ift_rsi_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn fosc_output_matches_direct() {
        let data = sample_series();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(8),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "fosc",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = FoscInput::from_slice(&data, FoscParams { period: Some(8) });
        let direct = fosc_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn linearreg_angle_output_matches_direct() {
        let data = sample_series();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "linearreg_angle",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input =
            Linearreg_angleInput::from_slice(&data, Linearreg_angleParams { period: Some(14) });
        let direct = linearreg_angle_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn linearreg_intercept_output_matches_direct() {
        let data = sample_series();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "linearreg_intercept",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = LinearRegInterceptInput::from_slice(
            &data,
            LinearRegInterceptParams { period: Some(14) },
        );
        let direct = linearreg_intercept_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn cg_output_matches_direct() {
        let data = sample_series();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(10),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "cg",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = CgInput::from_slice(&data, CgParams { period: Some(10) });
        let direct = cg_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn linearreg_slope_output_matches_direct() {
        let data = sample_series();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "linearreg_slope",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input =
            LinearRegSlopeInput::from_slice(&data, LinearRegSlopeParams { period: Some(14) });
        let direct = linearreg_slope_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn mean_ad_output_matches_direct() {
        let data = sample_series();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(7),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "mean_ad",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = MeanAdInput::from_slice(&data, MeanAdParams { period: Some(7) });
        let direct = mean_ad_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn deviation_output_matches_direct() {
        let data = sample_series();
        let combo = [
            ParamKV {
                key: "period",
                value: ParamValue::Int(9),
            },
            ParamKV {
                key: "devtype",
                value: ParamValue::Int(2),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "deviation",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = DeviationInput::from_slice(
            &data,
            DeviationParams {
                period: Some(9),
                devtype: Some(2),
            },
        );
        let direct = deviation_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn medprice_output_matches_direct() {
        let (_open, high, low, _close) = sample_ohlc();
        let combos = [IndicatorParamSet { params: &[] }];
        let req = IndicatorBatchRequest {
            indicator_id: "medprice",
            output_id: Some("value"),
            data: IndicatorDataRef::HighLow {
                high: &high,
                low: &low,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams::default());
        let direct = medprice_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn percentile_nearest_rank_output_matches_direct() {
        let data = sample_series();
        let combo = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(12),
            },
            ParamKV {
                key: "percentage",
                value: ParamValue::Float(70.0),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "percentile_nearest_rank",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = PercentileNearestRankInput::from_slice(
            &data,
            PercentileNearestRankParams {
                length: Some(12),
                percentage: Some(70.0),
            },
        );
        let direct = percentile_nearest_rank_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn zscore_output_matches_direct() {
        let data = sample_series();
        let combo = [
            ParamKV {
                key: "period",
                value: ParamValue::Int(14),
            },
            ParamKV {
                key: "ma_type",
                value: ParamValue::EnumString("ema"),
            },
            ParamKV {
                key: "nbdev",
                value: ParamValue::Float(1.25),
            },
            ParamKV {
                key: "devtype",
                value: ParamValue::Int(1),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "zscore",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = ZscoreInput::from_slice(
            &data,
            ZscoreParams {
                period: Some(14),
                ma_type: Some("ema".to_string()),
                nbdev: Some(1.25),
                devtype: Some(1),
            },
        );
        let direct = zscore_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn vpci_secondary_output_matches_direct() {
        let close = sample_series();
        let volume: Vec<f64> = (0..close.len())
            .map(|i| 1000.0 + (i as f64 * 7.0))
            .collect();
        let combo = [
            ParamKV {
                key: "short_range",
                value: ParamValue::Int(5),
            },
            ParamKV {
                key: "long_range",
                value: ParamValue::Int(25),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "vpci",
            output_id: Some("vpcis"),
            data: IndicatorDataRef::CloseVolume {
                close: &close,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = VpciInput::from_slices(
            &close,
            &volume,
            VpciParams {
                short_range: Some(5),
                long_range: Some(25),
            },
        );
        let direct = vpci_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .vpcis;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn yang_zhang_secondary_output_matches_direct() {
        let (open, high, low, close) = sample_ohlc();
        let combo = [
            ParamKV {
                key: "lookback",
                value: ParamValue::Int(21),
            },
            ParamKV {
                key: "k_override",
                value: ParamValue::Bool(true),
            },
            ParamKV {
                key: "k",
                value: ParamValue::Float(0.28),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "yang_zhang_volatility",
            output_id: Some("rs"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = YangZhangVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            YangZhangVolatilityParams {
                lookback: Some(21),
                k_override: Some(true),
                k: Some(0.28),
            },
        );
        let direct = yang_zhang_volatility_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .rs;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn ttm_trend_bool_output_matches_direct() {
        let (open, high, low, close) = sample_ohlc();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(5),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "ttm_trend",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let source: Vec<f64> = high.iter().zip(&low).map(|(h, l)| 0.5 * (h + l)).collect();
        let input = TtmTrendInput::from_slices(&source, &close, TtmTrendParams { period: Some(5) });
        let direct = ttm_trend_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        assert_eq!(out.values_bool.unwrap(), direct);
    }

    fn build_default_params_for_indicator(
        info: &crate::indicators::registry::IndicatorInfo,
    ) -> Option<Vec<ParamKV<'static>>> {
        let mut params: Vec<ParamKV<'static>> = Vec::new();
        for p in &info.params {
            if p.key.eq_ignore_ascii_case("output") {
                continue;
            }
            let value = if let Some(default) = p.default {
                match default {
                    crate::indicators::registry::ParamValueStatic::Int(v) => {
                        Some(ParamValue::Int(v))
                    }
                    crate::indicators::registry::ParamValueStatic::Float(v) => {
                        Some(ParamValue::Float(v))
                    }
                    crate::indicators::registry::ParamValueStatic::Bool(v) => {
                        Some(ParamValue::Bool(v))
                    }
                    crate::indicators::registry::ParamValueStatic::EnumString(v) => {
                        Some(ParamValue::EnumString(v))
                    }
                }
            } else {
                match p.kind {
                    IndicatorParamKind::Int => {
                        let mut v = p.min.unwrap_or(14.0).round() as i64;
                        if v < 0 {
                            v = 0;
                        }
                        if let Some(max) = p.max {
                            v = v.min(max.round() as i64);
                        }
                        Some(ParamValue::Int(v))
                    }
                    IndicatorParamKind::Float => {
                        let mut v = p.min.unwrap_or(1.0);
                        if !v.is_finite() {
                            v = 1.0;
                        }
                        if let Some(max) = p.max {
                            v = v.min(max);
                        }
                        Some(ParamValue::Float(v))
                    }
                    IndicatorParamKind::Bool => Some(ParamValue::Bool(false)),
                    IndicatorParamKind::EnumString => {
                        p.enum_values.first().copied().map(ParamValue::EnumString)
                    }
                }
            };

            match value {
                Some(v) => params.push(ParamKV {
                    key: p.key,
                    value: v,
                }),
                None => {
                    if p.required {
                        return None;
                    }
                }
            }
        }
        Some(params)
    }

    fn median_ns(mut samples: Vec<u128>) -> u128 {
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    #[test]
    #[ignore]
    fn full_cpu_dispatch_perf_sweep_vs_direct_route() {
        const LEN: usize = 10_000;
        const REPS: usize = 5;

        let open: Vec<f64> = (0..LEN).map(|i| 100.0 + (i as f64 * 0.01)).collect();
        let high: Vec<f64> = open.iter().map(|v| v + 1.0).collect();
        let low: Vec<f64> = open.iter().map(|v| v - 1.0).collect();
        let close: Vec<f64> = open.iter().map(|v| v + 0.25).collect();
        let volume: Vec<f64> = (0..LEN).map(|i| 1000.0 + (i as f64 * 0.5)).collect();
        let timestamp: Vec<i64> = (0..LEN).map(|i| i as i64).collect();
        let candles = crate::utilities::data_loader::Candles::new(
            timestamp,
            open.clone(),
            high.clone(),
            low.clone(),
            close.clone(),
            volume.clone(),
        );

        let infos: Vec<_> = list_indicators()
            .iter()
            .filter(|i| i.capabilities.supports_cpu_batch)
            .collect();
        let mut rows: Vec<(String, f64, f64, f64)> = Vec::new();
        let mut failures: Vec<String> = Vec::new();

        for info in infos {
            let Some(output) = info.outputs.first() else {
                failures.push(format!("{}: no outputs", info.id));
                continue;
            };
            let output_id = output.id;
            let Some(params_vec) = build_default_params_for_indicator(info) else {
                failures.push(format!("{}: missing required param defaults", info.id));
                continue;
            };
            let combos = [IndicatorParamSet {
                params: params_vec.as_slice(),
            }];
            let data = match info.input_kind {
                IndicatorInputKind::Slice => IndicatorDataRef::Slice {
                    values: close.as_slice(),
                },
                IndicatorInputKind::Candles => IndicatorDataRef::Candles {
                    candles: &candles,
                    source: None,
                },
                IndicatorInputKind::Ohlc => IndicatorDataRef::Ohlc {
                    open: open.as_slice(),
                    high: high.as_slice(),
                    low: low.as_slice(),
                    close: close.as_slice(),
                },
                IndicatorInputKind::Ohlcv => IndicatorDataRef::Ohlcv {
                    open: open.as_slice(),
                    high: high.as_slice(),
                    low: low.as_slice(),
                    close: close.as_slice(),
                    volume: volume.as_slice(),
                },
                IndicatorInputKind::HighLow => IndicatorDataRef::HighLow {
                    high: high.as_slice(),
                    low: low.as_slice(),
                },
                IndicatorInputKind::CloseVolume => IndicatorDataRef::CloseVolume {
                    close: close.as_slice(),
                    volume: volume.as_slice(),
                },
            };

            let req = IndicatorBatchRequest {
                indicator_id: info.id,
                output_id: Some(output_id),
                data,
                combos: &combos,
                kernel: Kernel::Auto,
            };

            let dispatch_once = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                compute_cpu_batch(req)
            })) {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => {
                    failures.push(format!("{}: dispatch error: {}", info.id, e));
                    continue;
                }
                Err(_) => {
                    failures.push(format!("{}: dispatch panic", info.id));
                    continue;
                }
            };
            let direct_once = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                dispatch_cpu_batch_by_indicator(req, info, output_id)
            })) {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => {
                    failures.push(format!("{}: direct-route error: {}", info.id, e));
                    continue;
                }
                Err(_) => {
                    failures.push(format!("{}: direct-route panic", info.id));
                    continue;
                }
            };

            if dispatch_once.rows != direct_once.rows || dispatch_once.cols != direct_once.cols {
                failures.push(format!(
                    "{}: shape mismatch dispatch=({},{}) direct=({},{})",
                    info.id,
                    dispatch_once.rows,
                    dispatch_once.cols,
                    direct_once.rows,
                    direct_once.cols
                ));
                continue;
            }

            let mut dispatch_samples = Vec::with_capacity(REPS);
            let mut direct_samples = Vec::with_capacity(REPS);
            let mut panicked = false;
            for _ in 0..REPS {
                let t0 = Instant::now();
                let dispatch_iter = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    compute_cpu_batch(req)
                }));
                if !matches!(dispatch_iter, Ok(Ok(_))) {
                    failures.push(format!("{}: dispatch panic/error during sample", info.id));
                    panicked = true;
                    break;
                }
                dispatch_samples.push(t0.elapsed().as_nanos());

                let t1 = Instant::now();
                let direct_iter = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    dispatch_cpu_batch_by_indicator(req, info, output_id)
                }));
                if !matches!(direct_iter, Ok(Ok(_))) {
                    failures.push(format!(
                        "{}: direct-route panic/error during sample",
                        info.id
                    ));
                    panicked = true;
                    break;
                }
                direct_samples.push(t1.elapsed().as_nanos());
            }
            if panicked {
                continue;
            }

            let dispatch_median = median_ns(dispatch_samples) as f64 / 1_000_000.0;
            let direct_median = median_ns(direct_samples) as f64 / 1_000_000.0;
            let delta_pct = if direct_median > 0.0 {
                ((dispatch_median - direct_median) / direct_median) * 100.0
            } else {
                0.0
            };
            rows.push((
                info.id.to_string(),
                direct_median,
                dispatch_median,
                delta_pct,
            ));
        }

        rows.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        println!("id,direct_ms,dispatch_ms,delta_pct");
        for (id, direct_ms, dispatch_ms, delta_pct) in &rows {
            println!("{id},{direct_ms:.6},{dispatch_ms:.6},{delta_pct:.2}");
        }
        println!("total_indicators={}", rows.len());

        assert!(
            failures.is_empty(),
            "perf sweep failures: {}",
            failures.join(" | ")
        );
        assert!(!rows.is_empty(), "no indicators were swept");
    }

    #[test]
    fn multi_output_requires_output_id() {
        let data = sample_series();
        let combos: [IndicatorParamSet<'_>; 0] = [];
        let req = IndicatorBatchRequest {
            indicator_id: "macd",
            output_id: None,
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let err = compute_cpu_batch(req).unwrap_err();
        assert!(matches!(err, IndicatorDispatchError::InvalidParam { .. }));
    }

    #[test]
    fn multi_output_unknown_output_is_rejected_globally() {
        let (open, high, low, close) = sample_ohlc();
        let volume: Vec<f64> = (0..close.len())
            .map(|i| 1000.0 + (i as f64 * 0.5))
            .collect();
        let timestamp: Vec<i64> = (0..close.len()).map(|i| i as i64).collect();
        let candles = crate::utilities::data_loader::Candles::new(
            timestamp,
            open.clone(),
            high.clone(),
            low.clone(),
            close.clone(),
            volume.clone(),
        );

        for info in list_indicators()
            .iter()
            .filter(|i| i.capabilities.supports_cpu_batch && i.outputs.len() > 1)
        {
            let Some(params_vec) = build_default_params_for_indicator(info) else {
                continue;
            };
            let combos = [IndicatorParamSet {
                params: params_vec.as_slice(),
            }];
            let data = match info.input_kind {
                IndicatorInputKind::Slice => IndicatorDataRef::Slice {
                    values: close.as_slice(),
                },
                IndicatorInputKind::Candles => IndicatorDataRef::Candles {
                    candles: &candles,
                    source: None,
                },
                IndicatorInputKind::Ohlc => IndicatorDataRef::Ohlc {
                    open: open.as_slice(),
                    high: high.as_slice(),
                    low: low.as_slice(),
                    close: close.as_slice(),
                },
                IndicatorInputKind::Ohlcv => IndicatorDataRef::Ohlcv {
                    open: open.as_slice(),
                    high: high.as_slice(),
                    low: low.as_slice(),
                    close: close.as_slice(),
                    volume: volume.as_slice(),
                },
                IndicatorInputKind::HighLow => IndicatorDataRef::HighLow {
                    high: high.as_slice(),
                    low: low.as_slice(),
                },
                IndicatorInputKind::CloseVolume => IndicatorDataRef::CloseVolume {
                    close: close.as_slice(),
                    volume: volume.as_slice(),
                },
            };
            let req = IndicatorBatchRequest {
                indicator_id: info.id,
                output_id: Some("__unknown_output__"),
                data,
                combos: &combos,
                kernel: Kernel::Auto,
            };
            let err = compute_cpu_batch(req).unwrap_err();
            assert!(
                matches!(err, IndicatorDispatchError::UnknownOutput { .. }),
                "indicator {} returned unexpected error for unknown output: {:?}",
                info.id,
                err
            );
        }
    }

    #[test]
    fn strict_mode_rejects_mismatched_input_kind_globally() {
        let data = sample_series();
        let candles = sample_candles();

        for info in list_indicators()
            .iter()
            .filter(|i| i.capabilities.supports_cpu_batch)
        {
            let Some(output) = info.outputs.first() else {
                continue;
            };
            let Some(params_vec) = build_default_params_for_indicator(info) else {
                continue;
            };
            let combos = [IndicatorParamSet {
                params: params_vec.as_slice(),
            }];
            let expected = strict_expected_input_kind(info.id, info.input_kind);
            let mismatched = match expected {
                IndicatorInputKind::Slice => IndicatorDataRef::Candles {
                    candles: &candles,
                    source: None,
                },
                IndicatorInputKind::Candles => IndicatorDataRef::Slice { values: &data },
                IndicatorInputKind::Ohlc
                | IndicatorInputKind::Ohlcv
                | IndicatorInputKind::HighLow
                | IndicatorInputKind::CloseVolume => IndicatorDataRef::Slice { values: &data },
            };
            let req = IndicatorBatchRequest {
                indicator_id: info.id,
                output_id: Some(output.id),
                data: mismatched,
                combos: &combos,
                kernel: Kernel::Auto,
            };
            let err = compute_cpu_batch_strict(req).unwrap_err();
            assert!(
                matches!(err, IndicatorDispatchError::MissingRequiredInput { .. }),
                "indicator {} did not reject strict mismatched input: {:?}",
                info.id,
                err
            );
        }
    }

    #[test]
    fn full_cpu_dispatch_parity_vs_direct_route_for_all_outputs() {
        const LEN: usize = 4096;
        let open: Vec<f64> = (0..LEN).map(|i| 100.0 + (i as f64 * 0.01)).collect();
        let high: Vec<f64> = open.iter().map(|v| v + 1.0).collect();
        let low: Vec<f64> = open.iter().map(|v| v - 1.0).collect();
        let close: Vec<f64> = open.iter().map(|v| v + 0.25).collect();
        let volume: Vec<f64> = (0..LEN).map(|i| 1000.0 + (i as f64 * 0.5)).collect();
        let timestamp: Vec<i64> = (0..LEN).map(|i| i as i64).collect();
        let candles = crate::utilities::data_loader::Candles::new(
            timestamp,
            open.clone(),
            high.clone(),
            low.clone(),
            close.clone(),
            volume.clone(),
        );

        for info in list_indicators()
            .iter()
            .filter(|i| i.capabilities.supports_cpu_batch)
        {
            let Some(params_vec) = build_default_params_for_indicator(info) else {
                continue;
            };
            let combos = [IndicatorParamSet {
                params: params_vec.as_slice(),
            }];
            let data = match info.input_kind {
                IndicatorInputKind::Slice => IndicatorDataRef::Slice {
                    values: close.as_slice(),
                },
                IndicatorInputKind::Candles => IndicatorDataRef::Candles {
                    candles: &candles,
                    source: None,
                },
                IndicatorInputKind::Ohlc => IndicatorDataRef::Ohlc {
                    open: open.as_slice(),
                    high: high.as_slice(),
                    low: low.as_slice(),
                    close: close.as_slice(),
                },
                IndicatorInputKind::Ohlcv => IndicatorDataRef::Ohlcv {
                    open: open.as_slice(),
                    high: high.as_slice(),
                    low: low.as_slice(),
                    close: close.as_slice(),
                    volume: volume.as_slice(),
                },
                IndicatorInputKind::HighLow => IndicatorDataRef::HighLow {
                    high: high.as_slice(),
                    low: low.as_slice(),
                },
                IndicatorInputKind::CloseVolume => IndicatorDataRef::CloseVolume {
                    close: close.as_slice(),
                    volume: volume.as_slice(),
                },
            };

            for output in info.outputs.iter() {
                let req = IndicatorBatchRequest {
                    indicator_id: info.id,
                    output_id: Some(output.id),
                    data,
                    combos: &combos,
                    kernel: Kernel::Auto,
                };
                let generic = compute_cpu_batch(req).unwrap_or_else(|e| {
                    panic!(
                        "generic dispatch failed for {}:{}: {}",
                        info.id, output.id, e
                    )
                });
                let direct =
                    dispatch_cpu_batch_by_indicator(req, info, output.id).unwrap_or_else(|e| {
                        panic!("direct route failed for {}:{}: {}", info.id, output.id, e)
                    });

                assert_eq!(
                    generic.rows, direct.rows,
                    "rows mismatch for {}:{}",
                    info.id, output.id
                );
                assert_eq!(
                    generic.cols, direct.cols,
                    "cols mismatch for {}:{}",
                    info.id, output.id
                );
                assert_eq!(
                    generic.output_id, direct.output_id,
                    "output id mismatch for {}:{}",
                    info.id, output.id
                );

                match (
                    generic.values_f64.as_ref(),
                    direct.values_f64.as_ref(),
                    generic.values_i32.as_ref(),
                    direct.values_i32.as_ref(),
                    generic.values_bool.as_ref(),
                    direct.values_bool.as_ref(),
                ) {
                    (Some(g), Some(d), None, None, None, None) => assert_series_eq(g, d, 1e-9),
                    (None, None, Some(g), Some(d), None, None) => assert_eq!(g, d),
                    (None, None, None, None, Some(g), Some(d)) => assert_eq!(g, d),
                    _ => panic!("value type mismatch for {}:{}", info.id, output.id),
                }
            }
        }
    }
}
