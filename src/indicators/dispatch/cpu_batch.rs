use super::{
    IndicatorBatchOutput, IndicatorBatchRequest, IndicatorDataRef, IndicatorDispatchError,
    IndicatorParamSet, ParamKV, ParamValue,
};
use crate::indicators::accumulation_swing_index::{
    accumulation_swing_index_with_kernel, AccumulationSwingIndexInput, AccumulationSwingIndexParams,
};
use crate::indicators::acosc::{acosc_with_kernel, AcoscInput, AcoscParams};
use crate::indicators::ad::{ad_with_kernel, AdInput, AdParams};
use crate::indicators::adaptive_bounds_rsi::{
    adaptive_bounds_rsi_with_kernel, AdaptiveBoundsRsiInput, AdaptiveBoundsRsiParams,
};
use crate::indicators::adjustable_ma_alternating_extremities::{
    adjustable_ma_alternating_extremities_with_kernel, AdjustableMaAlternatingExtremitiesInput,
    AdjustableMaAlternatingExtremitiesParams,
};
use crate::indicators::adaptive_macd::{
    adaptive_macd_with_kernel, AdaptiveMacdInput, AdaptiveMacdParams,
};
use crate::indicators::adaptive_momentum_oscillator::{
    adaptive_momentum_oscillator_with_kernel, AdaptiveMomentumOscillatorInput,
    AdaptiveMomentumOscillatorParams,
};
use crate::indicators::adosc::{adosc_with_kernel, AdoscInput, AdoscParams};
use crate::indicators::adx::{adx_with_kernel, AdxInput, AdxParams};
use crate::indicators::adxr::{adxr_with_kernel, AdxrInput, AdxrParams};
use crate::indicators::alligator::{alligator_with_kernel, AlligatorInput, AlligatorParams};
use crate::indicators::alphatrend::{alphatrend_with_kernel, AlphaTrendInput, AlphaTrendParams};
use crate::indicators::andean_oscillator::{
    andean_oscillator_with_kernel, AndeanOscillatorInput, AndeanOscillatorParams,
};
use crate::indicators::ao::{ao_into_slice, AoInput, AoParams};
use crate::indicators::apo::{apo_with_kernel, ApoInput, ApoParams};
use crate::indicators::aroon::{aroon_with_kernel, AroonInput, AroonParams};
use crate::indicators::aroonosc::{aroon_osc_with_kernel, AroonOscInput, AroonOscParams};
use crate::indicators::aso::{aso_with_kernel, AsoInput, AsoParams};
use crate::indicators::atr::{atr_with_kernel, AtrInput, AtrParams};
use crate::indicators::avsl::{avsl_with_kernel, AvslInput, AvslParams};
use crate::indicators::bandpass::{bandpass_with_kernel, BandPassInput, BandPassParams};
use crate::indicators::bollinger_bands::{
    bollinger_bands_with_kernel, BollingerBandsInput, BollingerBandsParams,
};
use crate::indicators::bollinger_bands_width::{
    bollinger_bands_width_with_kernel, BollingerBandsWidthInput, BollingerBandsWidthParams,
};
use crate::indicators::bop::{bop_with_kernel, BopInput, BopParams};
use crate::indicators::bulls_v_bears::{
    bulls_v_bears_with_kernel, BullsVBearsCalculationMethod, BullsVBearsInput, BullsVBearsMaType,
    BullsVBearsParams,
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
use crate::indicators::cycle_channel_oscillator::{
    cycle_channel_oscillator_with_kernel, CycleChannelOscillatorInput, CycleChannelOscillatorParams,
};
use crate::indicators::daily_factor::{
    daily_factor_with_kernel, DailyFactorInput, DailyFactorParams,
};
use crate::indicators::damiani_volatmeter::{
    damiani_volatmeter_with_kernel, DamianiVolatmeterInput, DamianiVolatmeterParams,
};
use crate::indicators::deviation::{deviation_with_kernel, DeviationInput, DeviationParams};
use crate::indicators::devstop::{devstop_with_kernel, DevStopInput, DevStopParams};
use crate::indicators::di::{di_with_kernel, DiInput, DiParams};
use crate::indicators::dm::{dm_with_kernel, DmInput, DmParams};
use crate::indicators::donchian::{donchian_with_kernel, DonchianInput, DonchianParams};
use crate::indicators::dpo::{dpo_with_kernel, DpoInput, DpoParams};
use crate::indicators::dti::{dti_into_slice, DtiInput, DtiParams};
use crate::indicators::dvdiqqe::{dvdiqqe_with_kernel, DvdiqqeInput, DvdiqqeParams};
use crate::indicators::dx::{dx_batch_with_kernel, dx_into_slice, DxBatchRange, DxInput, DxParams};
use crate::indicators::efi::{efi_with_kernel, EfiInput, EfiParams};
use crate::indicators::ehlers_adaptive_cyber_cycle::{
    ehlers_adaptive_cyber_cycle_with_kernel, EhlersAdaptiveCyberCycleInput,
    EhlersAdaptiveCyberCycleParams,
};
use crate::indicators::ehlers_simple_cycle_indicator::{
    ehlers_simple_cycle_indicator_with_kernel, EhlersSimpleCycleIndicatorInput,
    EhlersSimpleCycleIndicatorParams,
};
use crate::indicators::ehlers_smoothed_adaptive_momentum::{
    ehlers_smoothed_adaptive_momentum_with_kernel, EhlersSmoothedAdaptiveMomentumInput,
    EhlersSmoothedAdaptiveMomentumParams,
};
use crate::indicators::ehlers_adaptive_cg::{
    ehlers_adaptive_cg_with_kernel, EhlersAdaptiveCgInput, EhlersAdaptiveCgParams,
};
use crate::indicators::ehlers_fm_demodulator::{
    ehlers_fm_demodulator_with_kernel, EhlersFmDemodulatorInput, EhlersFmDemodulatorParams,
};
use crate::indicators::emd::{emd_with_kernel, EmdInput, EmdParams};
use crate::indicators::emv::{emv_with_kernel, EmvInput};
use crate::indicators::er::{er_with_kernel, ErInput, ErParams};
use crate::indicators::eri::{eri_with_kernel, EriInput, EriParams};
use crate::indicators::ewma_volatility::{
    ewma_volatility_with_kernel, EwmaVolatilityInput, EwmaVolatilityParams,
};
use crate::indicators::exponential_trend::{
    exponential_trend_with_kernel, ExponentialTrendInput, ExponentialTrendParams,
};
use crate::indicators::fisher::{fisher_with_kernel, FisherInput, FisherParams};
use crate::indicators::forward_backward_exponential_oscillator::{
    forward_backward_exponential_oscillator_with_kernel, ForwardBackwardExponentialOscillatorInput,
    ForwardBackwardExponentialOscillatorParams,
};
use crate::indicators::fosc::{fosc_with_kernel, FoscInput, FoscParams};
use crate::indicators::fvg_trailing_stop::{
    fvg_trailing_stop_with_kernel, FvgTrailingStopInput, FvgTrailingStopParams,
};
use crate::indicators::gatorosc::{gatorosc_with_kernel, GatorOscInput, GatorOscParams};
use crate::indicators::geometric_bias_oscillator::{
    geometric_bias_oscillator_with_kernel, GeometricBiasOscillatorInput,
    GeometricBiasOscillatorParams,
};
use crate::indicators::halftrend::{halftrend_with_kernel, HalfTrendInput, HalfTrendParams};
use crate::indicators::ichimoku_oscillator::{
    ichimoku_oscillator_with_kernel, IchimokuOscillatorInput, IchimokuOscillatorNormalizeMode,
    IchimokuOscillatorParams,
};
use crate::indicators::ift_rsi::{ift_rsi_with_kernel, IftRsiInput, IftRsiParams};
use crate::indicators::kaufmanstop::{
    kaufmanstop_with_kernel, KaufmanstopInput, KaufmanstopParams,
};
use crate::indicators::kdj::{kdj_with_kernel, KdjInput, KdjParams};
use crate::indicators::keltner::{keltner_with_kernel, KeltnerInput, KeltnerParams};
use crate::indicators::kst::{kst_with_kernel, KstInput, KstParams};
use crate::indicators::kurtosis::{kurtosis_with_kernel, KurtosisInput, KurtosisParams};
use crate::indicators::kvo::{kvo_with_kernel, KvoInput, KvoParams};
use crate::indicators::l1_ehlers_phasor::{
    l1_ehlers_phasor_with_kernel, L1EhlersPhasorInput, L1EhlersPhasorParams,
};
use crate::indicators::l2_ehlers_signal_to_noise::{
    l2_ehlers_signal_to_noise_with_kernel, L2EhlersSignalToNoiseInput, L2EhlersSignalToNoiseParams,
};
use crate::indicators::linear_correlation_oscillator::{
    linear_correlation_oscillator_with_kernel, LinearCorrelationOscillatorInput,
    LinearCorrelationOscillatorParams,
};
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
use crate::indicators::mass::{mass_with_kernel, MassInput, MassParams};
use crate::indicators::mean_ad::{mean_ad_with_kernel, MeanAdInput, MeanAdParams};
use crate::indicators::medium_ad::{medium_ad_with_kernel, MediumAdInput, MediumAdParams};
use crate::indicators::medprice::{medprice_with_kernel, MedpriceInput, MedpriceParams};
use crate::indicators::mesa_stochastic_multi_length::{
    mesa_stochastic_multi_length_with_kernel, MesaStochasticMultiLengthInput,
    MesaStochasticMultiLengthParams,
};
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
use crate::indicators::moving_average_cross_probability::{
    moving_average_cross_probability_with_kernel, MovingAverageCrossProbabilityInput,
    MovingAverageCrossProbabilityMaType, MovingAverageCrossProbabilityParams,
};
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
use crate::indicators::normalized_volume_true_range::{
    normalized_volume_true_range_with_kernel, NormalizedVolumeTrueRangeInput,
    NormalizedVolumeTrueRangeParams, NormalizedVolumeTrueRangeStyle,
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
use crate::indicators::polynomial_regression_extrapolation::{
    polynomial_regression_extrapolation_with_kernel, PolynomialRegressionExtrapolationInput,
    PolynomialRegressionExtrapolationParams,
};
use crate::indicators::ppo::{ppo_with_kernel, PpoInput, PpoParams};
use crate::indicators::prb::{prb_with_kernel, PrbInput, PrbParams};
use crate::indicators::price_moving_average_ratio_percentile::{
    price_moving_average_ratio_percentile_with_kernel, PriceMovingAverageRatioPercentileInput,
    PriceMovingAverageRatioPercentileLineMode, PriceMovingAverageRatioPercentileMaType,
    PriceMovingAverageRatioPercentileParams,
};
use crate::indicators::pvi::{pvi_with_kernel, PviInput, PviParams};
use crate::indicators::qqe::{qqe_with_kernel, QqeInput, QqeParams};
use crate::indicators::qqe_weighted_oscillator::{
    qqe_weighted_oscillator_with_kernel, QqeWeightedOscillatorInput, QqeWeightedOscillatorParams,
};
use crate::indicators::qstick::{qstick_with_kernel, QstickInput, QstickParams};
use crate::indicators::random_walk_index::{
    random_walk_index_with_kernel, RandomWalkIndexInput, RandomWalkIndexParams,
};
use crate::indicators::range_breakout_signals::{
    range_breakout_signals_with_kernel, RangeBreakoutSignalsInput, RangeBreakoutSignalsParams,
};
use crate::indicators::range_filter::{
    range_filter_with_kernel, RangeFilterInput, RangeFilterParams,
};
use crate::indicators::market_structure_confluence::{
    market_structure_confluence_with_kernel, MarketStructureConfluenceInput,
    MarketStructureConfluenceParams,
};
use crate::indicators::range_filtered_trend_signals::{
    range_filtered_trend_signals_with_kernel, RangeFilteredTrendSignalsInput,
    RangeFilteredTrendSignalsParams,
};
use crate::indicators::range_oscillator::{
    range_oscillator_with_kernel, RangeOscillatorInput, RangeOscillatorParams,
};
use crate::indicators::registry::{
    get_indicator, IndicatorInfo, IndicatorInputKind, ParamValueStatic,
};
use crate::indicators::regression_slope_oscillator::{
    regression_slope_oscillator_with_kernel, RegressionSlopeOscillatorInput,
    RegressionSlopeOscillatorParams,
};
use crate::indicators::relative_strength_index_wave_indicator::{
    relative_strength_index_wave_indicator_with_kernel, RelativeStrengthIndexWaveIndicatorInput,
    RelativeStrengthIndexWaveIndicatorParams,
};
use crate::indicators::reverse_rsi::{reverse_rsi_with_kernel, ReverseRsiInput, ReverseRsiParams};
use crate::indicators::roc::{roc_with_kernel, RocInput, RocParams};
use crate::indicators::rocp::{rocp_with_kernel, RocpInput, RocpParams};
use crate::indicators::rocr::{rocr_with_kernel, RocrInput, RocrParams};
use crate::indicators::rsi::{rsi_with_kernel, RsiInput, RsiParams};
use crate::indicators::rsmk::{rsmk_with_kernel, RsmkInput, RsmkParams};
use crate::indicators::rvi::{rvi_with_kernel, RviInput, RviParams};
use crate::indicators::safezonestop::{
    safezonestop_with_kernel, SafeZoneStopInput, SafeZoneStopParams,
};
use crate::indicators::smooth_theil_sen::{
    smooth_theil_sen_with_kernel, SmoothTheilSenDeviationType, SmoothTheilSenInput,
    SmoothTheilSenParams, SmoothTheilSenStatStyle,
};
use crate::indicators::spearman_correlation::{
    spearman_correlation_with_kernel, SpearmanCorrelationInput, SpearmanCorrelationParams,
};
use crate::indicators::squeeze_momentum::{
    squeeze_momentum_with_kernel, SqueezeMomentumInput, SqueezeMomentumParams,
};
use crate::indicators::srsi::{srsi_with_kernel, SrsiInput, SrsiParams};
use crate::indicators::standardized_psar_oscillator::{
    standardized_psar_oscillator_with_kernel, StandardizedPsarOscillatorInput,
    StandardizedPsarOscillatorParams,
};
use crate::indicators::statistical_trailing_stop::{
    statistical_trailing_stop_with_kernel, StatisticalTrailingStopInput,
    StatisticalTrailingStopParams,
};
use crate::indicators::stc::{stc_with_kernel, StcInput, StcParams};
use crate::indicators::stddev::{stddev_with_kernel, StdDevInput, StdDevParams};
use crate::indicators::stoch::{stoch_with_kernel, StochInput, StochParams};
use crate::indicators::stochf::{stochf_with_kernel, StochfInput, StochfParams};
use crate::indicators::supertrend::{supertrend_with_kernel, SuperTrendInput, SuperTrendParams};
use crate::indicators::trend_trigger_factor::{
    trend_trigger_factor_with_kernel, TrendTriggerFactorInput, TrendTriggerFactorParams,
};
use crate::indicators::supertrend_recovery::{
    supertrend_recovery_with_kernel, SuperTrendRecoveryInput, SuperTrendRecoveryParams,
};
use crate::indicators::trend_flow_trail::{
    trend_flow_trail_with_kernel, TrendFlowTrailInput, TrendFlowTrailParams,
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
use crate::indicators::vdubus_divergence_wave_pattern_generator::{
    vdubus_divergence_wave_pattern_generator_with_kernel,
    VdubusDivergenceWavePatternGeneratorInput, VdubusDivergenceWavePatternGeneratorParams,
};
use crate::indicators::velocity::{velocity_with_kernel, VelocityInput, VelocityParams};
use crate::indicators::vi::{vi_with_kernel, ViInput, ViParams};
use crate::indicators::vidya::{vidya_with_kernel, VidyaInput, VidyaParams};
use crate::indicators::vlma::{vlma_with_kernel, VlmaInput, VlmaParams};
use crate::indicators::volatility_quality_index::{
    volatility_quality_index_with_kernel, VolatilityQualityIndexInput, VolatilityQualityIndexParams,
};
use crate::indicators::volume_weighted_relative_strength_index::{
    volume_weighted_relative_strength_index_with_kernel, VolumeWeightedRelativeStrengthIndexInput,
    VolumeWeightedRelativeStrengthIndexParams,
};
use crate::indicators::volume_zone_oscillator::{
    volume_zone_oscillator_with_kernel, VolumeZoneOscillatorInput, VolumeZoneOscillatorParams,
};
use crate::indicators::vosc::{vosc_with_kernel, VoscInput, VoscParams};
use crate::indicators::voss::{voss_with_kernel, VossInput, VossParams};
use crate::indicators::vpci::{vpci_with_kernel, VpciInput, VpciParams};
use crate::indicators::vpt::{vpt_with_kernel, VptInput};
use crate::indicators::vwap_deviation_oscillator::{
    vwap_deviation_oscillator_with_kernel, VwapDeviationMode, VwapDeviationOscillatorInput,
    VwapDeviationOscillatorParams, VwapDeviationSessionMode,
};
use crate::indicators::vwmacd::{vwmacd_with_kernel, VwmacdInput, VwmacdParams};
use crate::indicators::wad::{wad_with_kernel, WadInput};
use crate::indicators::wavetrend::{wavetrend_with_kernel, WavetrendInput, WavetrendParams};
use crate::indicators::wclprice::{wclprice_with_kernel, WclpriceInput};
use crate::indicators::willr::{willr_with_kernel, WillrInput, WillrParams};
use crate::indicators::wto::{wto_with_kernel, WtoInput, WtoParams};
use crate::indicators::yang_zhang_volatility::{
    yang_zhang_volatility_with_kernel, YangZhangVolatilityInput, YangZhangVolatilityParams,
};
use crate::indicators::zscore::{zscore_with_kernel, ZscoreInput, ZscoreParams};
use crate::indicators::{cg::cg_with_kernel, cg::CgInput, cg::CgParams};
use crate::utilities::data_loader::source_type;
use crate::utilities::enums::Kernel;
use std::collections::HashMap;
use std::str::FromStr;

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
            "ehlers_adaptive_cg" => Some(compute_ehlers_adaptive_cg_batch(
                req,
                output_id.unwrap_or("cg"),
            )),
            "adaptive_momentum_oscillator" => Some(compute_adaptive_momentum_oscillator_batch(
                req,
                output_id.unwrap_or("amo"),
            )),
            "lrsi" => Some(compute_lrsi_batch(req, output_id.unwrap_or("value"))),
            "nvi" => Some(compute_nvi_batch(req, output_id.unwrap_or("value"))),
            "mom" => Some(compute_mom_batch(req, output_id.unwrap_or("value"))),
            "velocity" => Some(compute_velocity_batch(req, output_id.unwrap_or("value"))),
            "normalized_volume_true_range" => Some(compute_normalized_volume_true_range_batch(
                req,
                output_id.unwrap_or("normalized_volume"),
            )),
            "exponential_trend" => Some(compute_exponential_trend_batch(
                req,
                output_id.unwrap_or("uptrend_base"),
            )),
            "trend_flow_trail" => Some(compute_trend_flow_trail_batch(
                req,
                output_id.unwrap_or("alpha_trail"),
            )),
            "range_breakout_signals" => Some(compute_range_breakout_signals_batch(
                req,
                output_id.unwrap_or("range_top"),
            )),
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
            "accumulation_swing_index" => Some(compute_accumulation_swing_index_batch(
                req,
                output_id.unwrap_or("value"),
            )),
            "andean_oscillator" => {
                if let Some(out) = output_id {
                    Some(compute_andean_oscillator_batch(req, out))
                } else {
                    None
                }
            }
            "daily_factor" => {
                if let Some(out) = output_id {
                    Some(compute_daily_factor_batch(req, out))
                } else {
                    None
                }
            }
            "moving_average_cross_probability" => {
                if let Some(out) = output_id {
                    Some(compute_moving_average_cross_probability_batch(req, out))
                } else {
                    None
                }
            }
            "bulls_v_bears" => {
                if let Some(out) = output_id {
                    Some(compute_bulls_v_bears_batch(req, out))
                } else {
                    None
                }
            }
            "regression_slope_oscillator" => {
                if let Some(out) = output_id {
                    Some(compute_regression_slope_oscillator_batch(req, out))
                } else {
                    None
                }
            }
            "smooth_theil_sen" => {
                if let Some(out) = output_id {
                    Some(compute_smooth_theil_sen_batch(req, out))
                } else {
                    None
                }
            }
            "l2_ehlers_signal_to_noise" => Some(compute_l2_ehlers_signal_to_noise_batch(
                req,
                output_id.unwrap_or("value"),
            )),
            "ehlers_smoothed_adaptive_momentum" => Some(
                compute_ehlers_smoothed_adaptive_momentum_batch(req, output_id.unwrap_or("value")),
            ),
            "ehlers_adaptive_cyber_cycle" => {
                if let Some(out) = output_id {
                    Some(compute_ehlers_adaptive_cyber_cycle_batch(req, out))
                } else {
                    None
                }
            }
            "ehlers_simple_cycle_indicator" => {
                if let Some(out) = output_id {
                    Some(compute_ehlers_simple_cycle_indicator_batch(req, out))
                } else {
                    None
                }
            }
            "l1_ehlers_phasor" => Some(compute_l1_ehlers_phasor_batch(
                req,
                output_id.unwrap_or("value"),
            )),
            "cycle_channel_oscillator" => {
                if let Some(out) = output_id {
                    Some(compute_cycle_channel_oscillator_batch(req, out))
                } else {
                    None
                }
            }
            "ewma_volatility" => Some(compute_ewma_volatility_batch(
                req,
                output_id.unwrap_or("value"),
            )),
            "ichimoku_oscillator" => {
                if let Some(out) = output_id {
                    Some(compute_ichimoku_oscillator_batch(req, out))
                } else {
                    None
                }
            }
            "mesa_stochastic_multi_length" => {
                if let Some(out) = output_id {
                    Some(compute_mesa_stochastic_multi_length_batch(req, out))
                } else {
                    None
                }
            }
            "spearman_correlation" => {
                if let Some(out) = output_id {
                    Some(compute_spearman_correlation_batch(req, out))
                } else {
                    None
                }
            }
            "random_walk_index" => {
                if let Some(out) = output_id {
                    Some(compute_random_walk_index_batch(req, out))
                } else {
                    None
                }
            }
            "price_moving_average_ratio_percentile" => {
                if let Some(out) = output_id {
                    Some(compute_price_moving_average_ratio_percentile_batch(
                        req, out,
                    ))
                } else {
                    None
                }
            }
            "relative_strength_index_wave_indicator" => {
                if let Some(out) = output_id {
                    Some(compute_relative_strength_index_wave_indicator_batch(
                        req, out,
                    ))
                } else {
                    None
                }
            }
            "trend_trigger_factor" => Some(compute_trend_trigger_factor_batch(
                req,
                output_id.unwrap_or("value"),
            )),
            "vwap_deviation_oscillator" => {
                if let Some(out) = output_id {
                    Some(compute_vwap_deviation_oscillator_batch(req, out))
                } else {
                    None
                }
            }
            "volume_zone_oscillator" => Some(compute_volume_zone_oscillator_batch(
                req,
                output_id.unwrap_or("value"),
            )),
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
    if id.eq_ignore_ascii_case("ehlers_adaptive_cg") {
        return Some(compute_ehlers_adaptive_cg_batch(
            req,
            output_id.unwrap_or("cg"),
        ));
    }
    if id.eq_ignore_ascii_case("adaptive_momentum_oscillator") {
        return Some(compute_adaptive_momentum_oscillator_batch(
            req,
            output_id.unwrap_or("amo"),
        ));
    }
    if id.eq_ignore_ascii_case("adaptive_macd") {
        return Some(compute_adaptive_macd_batch(
            req,
            output_id.unwrap_or("macd"),
        ));
    }
    if id.eq_ignore_ascii_case("linear_correlation_oscillator") {
        return Some(compute_linear_correlation_oscillator_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("polynomial_regression_extrapolation") {
        return Some(compute_polynomial_regression_extrapolation_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("statistical_trailing_stop") {
        return Some(compute_statistical_trailing_stop_batch(
            req,
            output_id.unwrap_or("level"),
        ));
    }
    if id.eq_ignore_ascii_case("supertrend_recovery") {
        return Some(compute_supertrend_recovery_batch(
            req,
            output_id.unwrap_or("band"),
        ));
    }
    if id.eq_ignore_ascii_case("standardized_psar_oscillator") {
        return Some(compute_standardized_psar_oscillator_batch(
            req,
            output_id.unwrap_or("oscillator"),
        ));
    }
    if id.eq_ignore_ascii_case("geometric_bias_oscillator") {
        return Some(compute_geometric_bias_oscillator_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("lrsi") {
        return Some(compute_lrsi_batch(req, output_id.unwrap_or("value")));
    }
    if id.eq_ignore_ascii_case("nvi") {
        return Some(compute_nvi_batch(req, output_id.unwrap_or("value")));
    }
    if id.eq_ignore_ascii_case("ichimoku_oscillator") {
        if let Some(out) = output_id {
            return Some(compute_ichimoku_oscillator_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("mesa_stochastic_multi_length") {
        if let Some(out) = output_id {
            return Some(compute_mesa_stochastic_multi_length_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("spearman_correlation") {
        if let Some(out) = output_id {
            return Some(compute_spearman_correlation_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("mom") {
        return Some(compute_mom_batch(req, output_id.unwrap_or("value")));
    }
    if id.eq_ignore_ascii_case("velocity") {
        return Some(compute_velocity_batch(req, output_id.unwrap_or("value")));
    }
    if id.eq_ignore_ascii_case("normalized_volume_true_range") {
        return Some(compute_normalized_volume_true_range_batch(
            req,
            output_id.unwrap_or("normalized_volume"),
        ));
    }
    if id.eq_ignore_ascii_case("exponential_trend") {
        return Some(compute_exponential_trend_batch(
            req,
            output_id.unwrap_or("uptrend_base"),
        ));
    }
    if id.eq_ignore_ascii_case("trend_flow_trail") {
        return Some(compute_trend_flow_trail_batch(
            req,
            output_id.unwrap_or("alpha_trail"),
        ));
    }
    if id.eq_ignore_ascii_case("range_breakout_signals") {
        return Some(compute_range_breakout_signals_batch(
            req,
            output_id.unwrap_or("range_top"),
        ));
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
    if id.eq_ignore_ascii_case("accumulation_swing_index") {
        return Some(compute_accumulation_swing_index_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("andean_oscillator") {
        if let Some(out) = output_id {
            return Some(compute_andean_oscillator_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("daily_factor") {
        if let Some(out) = output_id {
            return Some(compute_daily_factor_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("moving_average_cross_probability") {
        if let Some(out) = output_id {
            return Some(compute_moving_average_cross_probability_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("bulls_v_bears") {
        if let Some(out) = output_id {
            return Some(compute_bulls_v_bears_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("regression_slope_oscillator") {
        if let Some(out) = output_id {
            return Some(compute_regression_slope_oscillator_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("smooth_theil_sen") {
        if let Some(out) = output_id {
            return Some(compute_smooth_theil_sen_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("l2_ehlers_signal_to_noise") {
        return Some(compute_l2_ehlers_signal_to_noise_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("ehlers_smoothed_adaptive_momentum") {
        return Some(compute_ehlers_smoothed_adaptive_momentum_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("ehlers_adaptive_cyber_cycle") {
        if let Some(out) = output_id {
            return Some(compute_ehlers_adaptive_cyber_cycle_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("ehlers_simple_cycle_indicator") {
        if let Some(out) = output_id {
            return Some(compute_ehlers_simple_cycle_indicator_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("cycle_channel_oscillator") {
        if let Some(out) = output_id {
            return Some(compute_cycle_channel_oscillator_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("ewma_volatility") {
        return Some(compute_ewma_volatility_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("random_walk_index") {
        if let Some(out) = output_id {
            return Some(compute_random_walk_index_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("price_moving_average_ratio_percentile") {
        if let Some(out) = output_id {
            return Some(compute_price_moving_average_ratio_percentile_batch(
                req, out,
            ));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("relative_strength_index_wave_indicator") {
        if let Some(out) = output_id {
            return Some(compute_relative_strength_index_wave_indicator_batch(
                req, out,
            ));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("l1_ehlers_phasor") {
        return Some(compute_l1_ehlers_phasor_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("trend_trigger_factor") {
        return Some(compute_trend_trigger_factor_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("vwap_deviation_oscillator") {
        if let Some(out) = output_id {
            return Some(compute_vwap_deviation_oscillator_batch(req, out));
        }
        return None;
    }
    if id.eq_ignore_ascii_case("volume_zone_oscillator") {
        return Some(compute_volume_zone_oscillator_batch(
            req,
            output_id.unwrap_or("value"),
        ));
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
        "ehlers_adaptive_cg" => compute_ehlers_adaptive_cg_batch(req, output_id),
        "adaptive_momentum_oscillator" => {
            compute_adaptive_momentum_oscillator_batch(req, output_id)
        }
        "adaptive_macd" => compute_adaptive_macd_batch(req, output_id),
        "linear_correlation_oscillator" => {
            compute_linear_correlation_oscillator_batch(req, output_id)
        }
        "polynomial_regression_extrapolation" => {
            compute_polynomial_regression_extrapolation_batch(req, output_id)
        }
        "statistical_trailing_stop" => compute_statistical_trailing_stop_batch(req, output_id),
        "supertrend_recovery" => compute_supertrend_recovery_batch(req, output_id),
        "standardized_psar_oscillator" => {
            compute_standardized_psar_oscillator_batch(req, output_id)
        }
        "geometric_bias_oscillator" => compute_geometric_bias_oscillator_batch(req, output_id),
        "vdubus_divergence_wave_pattern_generator" => {
            compute_vdubus_divergence_wave_pattern_generator_batch(req, output_id)
        }
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
        "ehlers_fm_demodulator" => compute_ehlers_fm_demodulator_batch(req, output_id),
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
        "velocity" => compute_velocity_batch(req, output_id),
        "normalized_volume_true_range" => {
            compute_normalized_volume_true_range_batch(req, output_id)
        }
        "exponential_trend" => compute_exponential_trend_batch(req, output_id),
        "trend_flow_trail" => compute_trend_flow_trail_batch(req, output_id),
        "range_breakout_signals" => compute_range_breakout_signals_batch(req, output_id),
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
        "adjustable_ma_alternating_extremities" => {
            compute_adjustable_ma_alternating_extremities_batch(req, output_id)
        }
        "vi" => compute_vi_batch(req, output_id),
        "wavetrend" => compute_wavetrend_batch(req, output_id),
        "wto" => compute_wto_batch(req, output_id),
        "accumulation_swing_index" => compute_accumulation_swing_index_batch(req, output_id),
        "andean_oscillator" => compute_andean_oscillator_batch(req, output_id),
        "daily_factor" => compute_daily_factor_batch(req, output_id),
        "moving_average_cross_probability" => {
            compute_moving_average_cross_probability_batch(req, output_id)
        }
        "bulls_v_bears" => compute_bulls_v_bears_batch(req, output_id),
        "regression_slope_oscillator" => compute_regression_slope_oscillator_batch(req, output_id),
        "smooth_theil_sen" => compute_smooth_theil_sen_batch(req, output_id),
        "l2_ehlers_signal_to_noise" => compute_l2_ehlers_signal_to_noise_batch(req, output_id),
        "ehlers_smoothed_adaptive_momentum" => {
            compute_ehlers_smoothed_adaptive_momentum_batch(req, output_id)
        }
        "ehlers_adaptive_cyber_cycle" => compute_ehlers_adaptive_cyber_cycle_batch(req, output_id),
        "ehlers_simple_cycle_indicator" => {
            compute_ehlers_simple_cycle_indicator_batch(req, output_id)
        }
        "l1_ehlers_phasor" => compute_l1_ehlers_phasor_batch(req, output_id),
        "cycle_channel_oscillator" => compute_cycle_channel_oscillator_batch(req, output_id),
        "ewma_volatility" => compute_ewma_volatility_batch(req, output_id),
        "ichimoku_oscillator" => compute_ichimoku_oscillator_batch(req, output_id),
        "mesa_stochastic_multi_length" => {
            compute_mesa_stochastic_multi_length_batch(req, output_id)
        }
        "spearman_correlation" => compute_spearman_correlation_batch(req, output_id),
        "random_walk_index" => compute_random_walk_index_batch(req, output_id),
        "price_moving_average_ratio_percentile" => {
            compute_price_moving_average_ratio_percentile_batch(req, output_id)
        }
        "relative_strength_index_wave_indicator" => {
            compute_relative_strength_index_wave_indicator_batch(req, output_id)
        }
        "trend_trigger_factor" => compute_trend_trigger_factor_batch(req, output_id),
        "volatility_quality_index" => compute_volatility_quality_index_batch(req, output_id),
        "vwap_deviation_oscillator" => compute_vwap_deviation_oscillator_batch(req, output_id),
        "volume_zone_oscillator" => compute_volume_zone_oscillator_batch(req, output_id),
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
        "adaptive_bounds_rsi" => compute_adaptive_bounds_rsi_batch(req, output_id),
        "forward_backward_exponential_oscillator" => {
            compute_forward_backward_exponential_oscillator_batch(req, output_id)
        }
        "qqe_weighted_oscillator" => compute_qqe_weighted_oscillator_batch(req, output_id),
        "market_structure_confluence" => compute_market_structure_confluence_batch(req, output_id),
        "range_filtered_trend_signals" => {
            compute_range_filtered_trend_signals_batch(req, output_id)
        }
        "range_oscillator" => compute_range_oscillator_batch(req, output_id),
        "volume_weighted_relative_strength_index" => {
            compute_volume_weighted_relative_strength_index_batch(req, output_id)
        }
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

fn compute_linear_correlation_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("linear_correlation_oscillator", output_id)?;
    let data = extract_slice_input("linear_correlation_oscillator", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "linear_correlation_oscillator",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let period = get_usize_param("linear_correlation_oscillator", params, "period", 14)?;
            let input = LinearCorrelationOscillatorInput::from_slice(
                data,
                LinearCorrelationOscillatorParams {
                    period: Some(period),
                },
            );
            let out = linear_correlation_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "linear_correlation_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
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

fn compute_ehlers_fm_demodulator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("ehlers_fm_demodulator", output_id)?;
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
                "ehlers_fm_demodulator",
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
                "ehlers_fm_demodulator",
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
                indicator: "ehlers_fm_demodulator".to_string(),
                input: IndicatorInputKind::Ohlc,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ehlers_fm_demodulator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let period = get_usize_param("ehlers_fm_demodulator", params, "period", 30)?;
            let input = EhlersFmDemodulatorInput::from_slices(
                open,
                close,
                EhlersFmDemodulatorParams {
                    period: Some(period),
                },
            );
            let out = ehlers_fm_demodulator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "ehlers_fm_demodulator".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
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

fn compute_velocity_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("velocity", output_id)?;
    let data = extract_slice_input("velocity", req.data, "hlcc4")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64("velocity", output_id, req.combos, data.len(), |params| {
        let length = get_usize_param("velocity", params, "length", 21)?;
        let smooth_length = get_usize_param("velocity", params, "smooth_length", 5)?;
        let input = VelocityInput::from_slice(
            data,
            VelocityParams {
                length: Some(length),
                smooth_length: Some(smooth_length),
            },
        );
        let out = velocity_with_kernel(&input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "velocity".to_string(),
                details: e.to_string(),
            }
        })?;
        Ok(out.values)
    })
}

fn compute_adaptive_momentum_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("adaptive_momentum_oscillator", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "adaptive_momentum_oscillator",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let length = get_usize_param("adaptive_momentum_oscillator", params, "length", 14)?;
            let smoothing_length = get_usize_param(
                "adaptive_momentum_oscillator",
                params,
                "smoothing_length",
                9,
            )?;
            let input = AdaptiveMomentumOscillatorInput::from_slice(
                data,
                AdaptiveMomentumOscillatorParams {
                    length: Some(length),
                    smoothing_length: Some(smoothing_length),
                },
            );
            let out = adaptive_momentum_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "adaptive_momentum_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            match output_id {
                "amo" | "value" => Ok(out.amo),
                "ama" => Ok(out.ama),
                other => Err(IndicatorDispatchError::UnknownOutput {
                    indicator: "adaptive_momentum_oscillator".to_string(),
                    output: other.to_string(),
                }),
            }
        },
    )
}

fn compute_normalized_volume_true_range_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close, volume) =
        extract_ohlcv_full_input("normalized_volume_true_range", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "normalized_volume_true_range",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let true_range_style = match find_param(params, "true_range_style") {
                Some(ParamValue::EnumString(value)) => Some(
                    value
                        .parse::<NormalizedVolumeTrueRangeStyle>()
                        .map_err(|e| IndicatorDispatchError::InvalidParam {
                            indicator: "normalized_volume_true_range".to_string(),
                            key: "true_range_style".to_string(),
                            reason: e,
                        })?,
                ),
                Some(_) => {
                    return Err(IndicatorDispatchError::InvalidParam {
                        indicator: "normalized_volume_true_range".to_string(),
                        key: "true_range_style".to_string(),
                        reason: "expected enum string".to_string(),
                    });
                }
                None => Some(NormalizedVolumeTrueRangeStyle::Body),
            };
            let outlier_range =
                get_f64_param("normalized_volume_true_range", params, "outlier_range", 5.0)?;
            let atr_length =
                get_usize_param("normalized_volume_true_range", params, "atr_length", 14)?;
            let volume_length =
                get_usize_param("normalized_volume_true_range", params, "volume_length", 14)?;

            let input = NormalizedVolumeTrueRangeInput::from_slices(
                open,
                high,
                low,
                close,
                volume,
                NormalizedVolumeTrueRangeParams {
                    true_range_style,
                    outlier_range: Some(outlier_range),
                    atr_length: Some(atr_length),
                    volume_length: Some(volume_length),
                },
            );
            let out = normalized_volume_true_range_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "normalized_volume_true_range".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("normalized_volume")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.normalized_volume);
            }
            if output_id.eq_ignore_ascii_case("normalized_true_range") {
                return Ok(out.normalized_true_range);
            }
            if output_id.eq_ignore_ascii_case("baseline") {
                return Ok(out.baseline);
            }
            if output_id.eq_ignore_ascii_case("atr") {
                return Ok(out.atr);
            }
            if output_id.eq_ignore_ascii_case("average_volume") {
                return Ok(out.average_volume);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "normalized_volume_true_range".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_range_breakout_signals_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close, volume) =
        extract_ohlcv_full_input("range_breakout_signals", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "range_breakout_signals",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let range_length =
                get_usize_param("range_breakout_signals", params, "range_length", 20)?;
            let confirmation_length =
                get_usize_param("range_breakout_signals", params, "confirmation_length", 5)?;
            let input = RangeBreakoutSignalsInput::from_slices(
                open,
                high,
                low,
                close,
                volume,
                RangeBreakoutSignalsParams {
                    range_length: Some(range_length),
                    confirmation_length: Some(confirmation_length),
                },
            );
            let out = range_breakout_signals_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "range_breakout_signals".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("range_top")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.range_top);
            }
            if output_id.eq_ignore_ascii_case("range_bottom") {
                return Ok(out.range_bottom);
            }
            if output_id.eq_ignore_ascii_case("bullish") {
                return Ok(out.bullish);
            }
            if output_id.eq_ignore_ascii_case("extra_bullish") {
                return Ok(out.extra_bullish);
            }
            if output_id.eq_ignore_ascii_case("bearish") {
                return Ok(out.bearish);
            }
            if output_id.eq_ignore_ascii_case("extra_bearish") {
                return Ok(out.extra_bearish);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "range_breakout_signals".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_exponential_trend_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("exponential_trend", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "exponential_trend",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let exp_rate = get_f64_param("exponential_trend", params, "exp_rate", 0.00003)?;
            let initial_distance =
                get_f64_param("exponential_trend", params, "initial_distance", 4.0)?;
            let width_multiplier =
                get_f64_param("exponential_trend", params, "width_multiplier", 1.0)?;
            let input = ExponentialTrendInput::from_slices(
                high,
                low,
                close,
                ExponentialTrendParams {
                    exp_rate: Some(exp_rate),
                    initial_distance: Some(initial_distance),
                    width_multiplier: Some(width_multiplier),
                },
            );
            let out = exponential_trend_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "exponential_trend".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("uptrend_base")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.uptrend_base);
            }
            if output_id.eq_ignore_ascii_case("downtrend_base") {
                return Ok(out.downtrend_base);
            }
            if output_id.eq_ignore_ascii_case("uptrend_extension") {
                return Ok(out.uptrend_extension);
            }
            if output_id.eq_ignore_ascii_case("downtrend_extension") {
                return Ok(out.downtrend_extension);
            }
            if output_id.eq_ignore_ascii_case("bullish_change") {
                return Ok(out.bullish_change);
            }
            if output_id.eq_ignore_ascii_case("bearish_change") {
                return Ok(out.bearish_change);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "exponential_trend".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_trend_flow_trail_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close, volume) = extract_ohlcv_full_input("trend_flow_trail", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "trend_flow_trail",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let alpha_length = get_usize_param("trend_flow_trail", params, "alpha_length", 33)?;
            let alpha_multiplier =
                get_f64_param("trend_flow_trail", params, "alpha_multiplier", 3.3)?;
            let mfi_length = get_usize_param("trend_flow_trail", params, "mfi_length", 14)?;
            let input = TrendFlowTrailInput::from_slices(
                open,
                high,
                low,
                close,
                volume,
                TrendFlowTrailParams {
                    alpha_length: Some(alpha_length),
                    alpha_multiplier: Some(alpha_multiplier),
                    mfi_length: Some(mfi_length),
                },
            );
            let out = trend_flow_trail_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "trend_flow_trail".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("alpha_trail")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.alpha_trail);
            }
            if output_id.eq_ignore_ascii_case("alpha_trail_bullish") {
                return Ok(out.alpha_trail_bullish);
            }
            if output_id.eq_ignore_ascii_case("alpha_trail_bearish") {
                return Ok(out.alpha_trail_bearish);
            }
            if output_id.eq_ignore_ascii_case("alpha_dir") {
                return Ok(out.alpha_dir);
            }
            if output_id.eq_ignore_ascii_case("mfi") {
                return Ok(out.mfi);
            }
            if output_id.eq_ignore_ascii_case("tp_upper") {
                return Ok(out.tp_upper);
            }
            if output_id.eq_ignore_ascii_case("tp_lower") {
                return Ok(out.tp_lower);
            }
            if output_id.eq_ignore_ascii_case("alpha_trail_bullish_switch") {
                return Ok(out.alpha_trail_bullish_switch);
            }
            if output_id.eq_ignore_ascii_case("alpha_trail_bearish_switch") {
                return Ok(out.alpha_trail_bearish_switch);
            }
            if output_id.eq_ignore_ascii_case("mfi_overbought") {
                return Ok(out.mfi_overbought);
            }
            if output_id.eq_ignore_ascii_case("mfi_oversold") {
                return Ok(out.mfi_oversold);
            }
            if output_id.eq_ignore_ascii_case("mfi_cross_up_mid") {
                return Ok(out.mfi_cross_up_mid);
            }
            if output_id.eq_ignore_ascii_case("mfi_cross_down_mid") {
                return Ok(out.mfi_cross_down_mid);
            }
            if output_id.eq_ignore_ascii_case("price_cross_alpha_trail_up") {
                return Ok(out.price_cross_alpha_trail_up);
            }
            if output_id.eq_ignore_ascii_case("price_cross_alpha_trail_down") {
                return Ok(out.price_cross_alpha_trail_down);
            }
            if output_id.eq_ignore_ascii_case("mfi_above_90") {
                return Ok(out.mfi_above_90);
            }
            if output_id.eq_ignore_ascii_case("mfi_below_10") {
                return Ok(out.mfi_below_10);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "trend_flow_trail".to_string(),
                output: output_id.to_string(),
            })
        },
    )
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

fn compute_polynomial_regression_extrapolation_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("polynomial_regression_extrapolation", output_id)?;
    let data = extract_slice_input("polynomial_regression_extrapolation", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "polynomial_regression_extrapolation",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let length =
                get_usize_param("polynomial_regression_extrapolation", params, "length", 100)?;
            let extrapolate = get_usize_param(
                "polynomial_regression_extrapolation",
                params,
                "extrapolate",
                10,
            )?;
            let degree =
                get_usize_param("polynomial_regression_extrapolation", params, "degree", 3)?;
            let input = PolynomialRegressionExtrapolationInput::from_slice(
                data,
                PolynomialRegressionExtrapolationParams {
                    length: Some(length),
                    extrapolate: Some(extrapolate),
                    degree: Some(degree),
                },
            );
            let out =
                polynomial_regression_extrapolation_with_kernel(&input, kernel).map_err(|e| {
                    IndicatorDispatchError::ComputeFailed {
                        indicator: "polynomial_regression_extrapolation".to_string(),
                        details: e.to_string(),
                    }
                })?;
            Ok(out.values)
        },
    )
}

fn compute_adaptive_macd_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("adaptive_macd", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "adaptive_macd",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let length = get_usize_param("adaptive_macd", params, "length", 20)?;
            let fast_period = get_usize_param("adaptive_macd", params, "fast_period", 10)?;
            let slow_period = get_usize_param("adaptive_macd", params, "slow_period", 20)?;
            let signal_period = get_usize_param("adaptive_macd", params, "signal_period", 9)?;
            let input = AdaptiveMacdInput::from_slice(
                data,
                AdaptiveMacdParams {
                    length: Some(length),
                    fast_period: Some(fast_period),
                    slow_period: Some(slow_period),
                    signal_period: Some(signal_period),
                },
            );
            let out = adaptive_macd_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "adaptive_macd".to_string(),
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
                indicator: "adaptive_macd".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_statistical_trailing_stop_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("statistical_trailing_stop", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "statistical_trailing_stop",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let data_length =
                get_usize_param("statistical_trailing_stop", params, "data_length", 10)?;
            let normalization_length = get_usize_param(
                "statistical_trailing_stop",
                params,
                "normalization_length",
                100,
            )?;
            let base_level =
                get_enum_param("statistical_trailing_stop", params, "base_level", "level2")?;
            let input = StatisticalTrailingStopInput::from_slices(
                high,
                low,
                close,
                StatisticalTrailingStopParams {
                    data_length: Some(data_length),
                    normalization_length: Some(normalization_length),
                    base_level: Some(base_level),
                },
            );
            let out = statistical_trailing_stop_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "statistical_trailing_stop".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("level") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.level);
            }
            if output_id.eq_ignore_ascii_case("anchor") {
                return Ok(out.anchor);
            }
            if output_id.eq_ignore_ascii_case("bias") {
                return Ok(out.bias);
            }
            if output_id.eq_ignore_ascii_case("changed") {
                return Ok(out.changed);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "statistical_trailing_stop".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_supertrend_recovery_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("supertrend_recovery", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "supertrend_recovery",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let atr_length = get_usize_param("supertrend_recovery", params, "atr_length", 10)?;
            let multiplier = get_f64_param("supertrend_recovery", params, "multiplier", 3.0)?;
            let alpha_percent = get_f64_param("supertrend_recovery", params, "alpha_percent", 5.0)?;
            let threshold_atr = get_f64_param("supertrend_recovery", params, "threshold_atr", 1.0)?;
            let input = SuperTrendRecoveryInput::from_slices(
                high,
                low,
                close,
                SuperTrendRecoveryParams {
                    atr_length: Some(atr_length),
                    multiplier: Some(multiplier),
                    alpha_percent: Some(alpha_percent),
                    threshold_atr: Some(threshold_atr),
                },
            );
            let out = supertrend_recovery_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "supertrend_recovery".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("band") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.band);
            }
            if output_id.eq_ignore_ascii_case("switch_price") {
                return Ok(out.switch_price);
            }
            if output_id.eq_ignore_ascii_case("trend") {
                return Ok(out.trend);
            }
            if output_id.eq_ignore_ascii_case("changed") {
                return Ok(out.changed);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "supertrend_recovery".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_standardized_psar_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("standardized_psar_oscillator", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "standardized_psar_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let start = get_f64_param("standardized_psar_oscillator", params, "start", 0.02)?;
            let increment =
                get_f64_param("standardized_psar_oscillator", params, "increment", 0.0005)?;
            let maximum = get_f64_param("standardized_psar_oscillator", params, "maximum", 0.2)?;
            let standardization_length = get_usize_param(
                "standardized_psar_oscillator",
                params,
                "standardization_length",
                21,
            )?;
            let wma_length =
                get_usize_param("standardized_psar_oscillator", params, "wma_length", 40)?;
            let wma_lag = get_usize_param("standardized_psar_oscillator", params, "wma_lag", 3)?;
            let pivot_left =
                get_usize_param("standardized_psar_oscillator", params, "pivot_left", 15)?;
            let pivot_right =
                get_usize_param("standardized_psar_oscillator", params, "pivot_right", 1)?;
            let plot_bullish =
                get_bool_param("standardized_psar_oscillator", params, "plot_bullish", true)?;
            let plot_bearish =
                get_bool_param("standardized_psar_oscillator", params, "plot_bearish", true)?;
            let input = StandardizedPsarOscillatorInput::from_slices(
                high,
                low,
                close,
                StandardizedPsarOscillatorParams {
                    start: Some(start),
                    increment: Some(increment),
                    maximum: Some(maximum),
                    standardization_length: Some(standardization_length),
                    wma_length: Some(wma_length),
                    wma_lag: Some(wma_lag),
                    pivot_left: Some(pivot_left),
                    pivot_right: Some(pivot_right),
                    plot_bullish: Some(plot_bullish),
                    plot_bearish: Some(plot_bearish),
                },
            );
            let out = standardized_psar_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "standardized_psar_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            match output_id {
                "oscillator" | "value" => Ok(out.oscillator),
                "ma" => Ok(out.ma),
                "bullish_reversal" => Ok(out.bullish_reversal),
                "bearish_reversal" => Ok(out.bearish_reversal),
                "regular_bullish" => Ok(out.regular_bullish),
                "regular_bearish" => Ok(out.regular_bearish),
                "bullish_weakening" => Ok(out.bullish_weakening),
                "bearish_weakening" => Ok(out.bearish_weakening),
                _ => Err(IndicatorDispatchError::UnknownOutput {
                    indicator: "standardized_psar_oscillator".to_string(),
                    output: output_id.to_string(),
                }),
            }
        },
    )
}

fn compute_geometric_bias_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("geometric_bias_oscillator", output_id)?;
    let (high, low, close) = extract_ohlc_input("geometric_bias_oscillator", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "geometric_bias_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param("geometric_bias_oscillator", params, "length", 100)?;
            let multiplier = get_f64_param("geometric_bias_oscillator", params, "multiplier", 2.0)?;
            let atr_length =
                get_usize_param("geometric_bias_oscillator", params, "atr_length", 14)?;
            let smooth = get_usize_param("geometric_bias_oscillator", params, "smooth", 1)?;
            let input = GeometricBiasOscillatorInput::from_slices(
                high,
                low,
                close,
                GeometricBiasOscillatorParams {
                    length: Some(length),
                    multiplier: Some(multiplier),
                    atr_length: Some(atr_length),
                    smooth: Some(smooth),
                },
            );
            let out = geometric_bias_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "geometric_bias_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
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

fn compute_vdubus_divergence_wave_pattern_generator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("vdubus_divergence_wave_pattern_generator", output_id)?;
    let (high, low, close) =
        extract_ohlc_input("vdubus_divergence_wave_pattern_generator", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "vdubus_divergence_wave_pattern_generator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let fast_depth = get_usize_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "fast_depth",
                9,
            )?;
            let slow_depth = get_usize_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "slow_depth",
                24,
            )?;
            let fast_length = get_usize_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "fast_length",
                21,
            )?;
            let slow_length = get_usize_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "slow_length",
                34,
            )?;
            let signal_length = get_usize_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "signal_length",
                5,
            )?;
            let lookback = get_usize_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "lookback",
                3,
            )?;
            let err_tol = get_f64_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "err_tol",
                0.15,
            )?;
            let show_standard = get_bool_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "show_standard",
                true,
            )?;
            let show_climax = get_bool_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "show_climax",
                true,
            )?;
            let show_rounded = get_bool_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "show_rounded",
                true,
            )?;
            let show_predator = get_bool_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "show_predator",
                true,
            )?;
            let show_gartley = get_bool_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "show_gartley",
                false,
            )?;
            let show_bat = get_bool_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "show_bat",
                false,
            )?;
            let show_butterfly = get_bool_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "show_butterfly",
                false,
            )?;
            let show_crab = get_bool_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "show_crab",
                false,
            )?;
            let show_deep = get_bool_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "show_deep",
                false,
            )?;
            let show_hs = get_bool_param(
                "vdubus_divergence_wave_pattern_generator",
                params,
                "show_hs",
                true,
            )?;
            let input = VdubusDivergenceWavePatternGeneratorInput::from_slices(
                high,
                low,
                close,
                VdubusDivergenceWavePatternGeneratorParams {
                    fast_depth: Some(fast_depth),
                    slow_depth: Some(slow_depth),
                    fast_length: Some(fast_length),
                    slow_length: Some(slow_length),
                    signal_length: Some(signal_length),
                    lookback: Some(lookback),
                    err_tol: Some(err_tol),
                    show_standard: Some(show_standard),
                    show_climax: Some(show_climax),
                    show_rounded: Some(show_rounded),
                    show_predator: Some(show_predator),
                    show_gartley: Some(show_gartley),
                    show_bat: Some(show_bat),
                    show_butterfly: Some(show_butterfly),
                    show_crab: Some(show_crab),
                    show_deep: Some(show_deep),
                    show_hs: Some(show_hs),
                },
            );
            let out = vdubus_divergence_wave_pattern_generator_with_kernel(&input, kernel)
                .map_err(|e| IndicatorDispatchError::ComputeFailed {
                    indicator: "vdubus_divergence_wave_pattern_generator".to_string(),
                    details: e.to_string(),
                })?;
            match output_id {
                "fast_standard" => Ok(out.fast_standard),
                "fast_climax" => Ok(out.fast_climax),
                "fast_rounded" => Ok(out.fast_rounded),
                "fast_predator" => Ok(out.fast_predator),
                "slow_standard" => Ok(out.slow_standard),
                "slow_climax" => Ok(out.slow_climax),
                "slow_rounded" => Ok(out.slow_rounded),
                "slow_predator" => Ok(out.slow_predator),
                "opposing_force" => Ok(out.opposing_force),
                "macd" => Ok(out.macd),
                "signal" => Ok(out.signal),
                "hist" => Ok(out.hist),
                _ => Err(IndicatorDispatchError::UnknownOutput {
                    indicator: "vdubus_divergence_wave_pattern_generator".to_string(),
                    output: output_id.to_string(),
                }),
            }
        },
    )
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

fn compute_adjustable_ma_alternating_extremities_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("adjustable_ma_alternating_extremities", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "adjustable_ma_alternating_extremities",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param(
                "adjustable_ma_alternating_extremities",
                params,
                "length",
                50,
            )?;
            let mult = get_f64_param("adjustable_ma_alternating_extremities", params, "mult", 2.0)?;
            let alpha = get_f64_param(
                "adjustable_ma_alternating_extremities",
                params,
                "alpha",
                1.0,
            )?;
            let beta = get_f64_param("adjustable_ma_alternating_extremities", params, "beta", 0.5)?;
            let input = AdjustableMaAlternatingExtremitiesInput::from_slices(
                high,
                low,
                close,
                AdjustableMaAlternatingExtremitiesParams {
                    length: Some(length),
                    mult: Some(mult),
                    alpha: Some(alpha),
                    beta: Some(beta),
                },
            );
            let out =
                adjustable_ma_alternating_extremities_with_kernel(&input, kernel).map_err(|e| {
                    IndicatorDispatchError::ComputeFailed {
                        indicator: "adjustable_ma_alternating_extremities".to_string(),
                        details: e.to_string(),
                    }
                })?;
            if output_id.eq_ignore_ascii_case("ma") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.ma);
            }
            if output_id.eq_ignore_ascii_case("upper") {
                return Ok(out.upper);
            }
            if output_id.eq_ignore_ascii_case("lower") {
                return Ok(out.lower);
            }
            if output_id.eq_ignore_ascii_case("extremity") {
                return Ok(out.extremity);
            }
            if output_id.eq_ignore_ascii_case("state") {
                return Ok(out.state);
            }
            if output_id.eq_ignore_ascii_case("changed") {
                return Ok(out.changed);
            }
            if output_id.eq_ignore_ascii_case("smoothed_open") {
                return Ok(out.smoothed_open);
            }
            if output_id.eq_ignore_ascii_case("smoothed_high") {
                return Ok(out.smoothed_high);
            }
            if output_id.eq_ignore_ascii_case("smoothed_low") {
                return Ok(out.smoothed_low);
            }
            if output_id.eq_ignore_ascii_case("smoothed_close") {
                return Ok(out.smoothed_close);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "adjustable_ma_alternating_extremities".to_string(),
                output: output_id.to_string(),
            })
        },
    )
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

fn compute_l2_ehlers_signal_to_noise_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (candles, default_source) = match req.data {
        IndicatorDataRef::Candles { candles, source } => (candles, source.unwrap_or("hl2")),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "l2_ehlers_signal_to_noise".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "l2_ehlers_signal_to_noise",
        output_id,
        req.combos,
        candles.close.len(),
        |params| {
            let source = get_enum_param(
                "l2_ehlers_signal_to_noise",
                params,
                "source",
                default_source,
            )?;
            let smooth_period =
                get_usize_param("l2_ehlers_signal_to_noise", params, "smooth_period", 10)?;
            let input = L2EhlersSignalToNoiseInput::from_slices(
                source_type(candles, &source),
                candles.high.as_slice(),
                candles.low.as_slice(),
                L2EhlersSignalToNoiseParams {
                    smooth_period: Some(smooth_period),
                },
            );
            let out = l2_ehlers_signal_to_noise_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "l2_ehlers_signal_to_noise".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.values);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "l2_ehlers_signal_to_noise".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_ehlers_smoothed_adaptive_momentum_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (candles, default_source) = match req.data {
        IndicatorDataRef::Candles { candles, source } => (candles, source.unwrap_or("hl2")),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "ehlers_smoothed_adaptive_momentum".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ehlers_smoothed_adaptive_momentum",
        output_id,
        req.combos,
        candles.close.len(),
        |params| {
            let source = get_enum_param(
                "ehlers_smoothed_adaptive_momentum",
                params,
                "source",
                default_source,
            )?;
            let alpha = get_f64_param("ehlers_smoothed_adaptive_momentum", params, "alpha", 0.07)?;
            let cutoff = get_f64_param("ehlers_smoothed_adaptive_momentum", params, "cutoff", 8.0)?;
            let input = EhlersSmoothedAdaptiveMomentumInput::from_slice(
                source_type(candles, &source),
                EhlersSmoothedAdaptiveMomentumParams {
                    alpha: Some(alpha),
                    cutoff: Some(cutoff),
                },
            );
            let out =
                ehlers_smoothed_adaptive_momentum_with_kernel(&input, kernel).map_err(|e| {
                    IndicatorDispatchError::ComputeFailed {
                        indicator: "ehlers_smoothed_adaptive_momentum".to_string(),
                        details: e.to_string(),
                    }
                })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.values);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "ehlers_smoothed_adaptive_momentum".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_ehlers_adaptive_cyber_cycle_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (candles, default_source) = match req.data {
        IndicatorDataRef::Candles { candles, source } => (candles, source.unwrap_or("hl2")),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "ehlers_adaptive_cyber_cycle".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ehlers_adaptive_cyber_cycle",
        output_id,
        req.combos,
        candles.close.len(),
        |params| {
            let source = get_enum_param(
                "ehlers_adaptive_cyber_cycle",
                params,
                "source",
                default_source,
            )?;
            let alpha = get_f64_param("ehlers_adaptive_cyber_cycle", params, "alpha", 0.07)?;
            let input = EhlersAdaptiveCyberCycleInput::from_slice(
                source_type(candles, &source),
                EhlersAdaptiveCyberCycleParams { alpha: Some(alpha) },
            );
            let out = ehlers_adaptive_cyber_cycle_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "ehlers_adaptive_cyber_cycle".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("cycle") {
                return Ok(out.cycle);
            }
            if output_id.eq_ignore_ascii_case("trigger") {
                return Ok(out.trigger);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "ehlers_adaptive_cyber_cycle".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_ehlers_simple_cycle_indicator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (candles, default_source) = match req.data {
        IndicatorDataRef::Candles { candles, source } => (candles, source.unwrap_or("hl2")),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "ehlers_simple_cycle_indicator".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ehlers_simple_cycle_indicator",
        output_id,
        req.combos,
        candles.close.len(),
        |params| {
            let source = get_enum_param(
                "ehlers_simple_cycle_indicator",
                params,
                "source",
                default_source,
            )?;
            let alpha = get_f64_param("ehlers_simple_cycle_indicator", params, "alpha", 0.07)?;
            let input = EhlersSimpleCycleIndicatorInput::from_slice(
                source_type(candles, &source),
                EhlersSimpleCycleIndicatorParams { alpha: Some(alpha) },
            );
            let out = ehlers_simple_cycle_indicator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "ehlers_simple_cycle_indicator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("cycle") {
                return Ok(out.cycle);
            }
            if output_id.eq_ignore_ascii_case("trigger") {
                return Ok(out.trigger);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "ehlers_simple_cycle_indicator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_l1_ehlers_phasor_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let candles = match req.data {
        IndicatorDataRef::Candles { candles, .. } => candles,
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "l1_ehlers_phasor".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "l1_ehlers_phasor",
        output_id,
        req.combos,
        candles.close.len(),
        |params| {
            let domestic_cycle_length =
                get_usize_param("l1_ehlers_phasor", params, "domestic_cycle_length", 15)?;
            let input = L1EhlersPhasorInput::from_slice(
                candles.close.as_slice(),
                L1EhlersPhasorParams {
                    domestic_cycle_length: Some(domestic_cycle_length),
                },
            );
            let out = l1_ehlers_phasor_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "l1_ehlers_phasor".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.values);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "l1_ehlers_phasor".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_andean_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, close): (&[f64], &[f64]) = match req.data {
        IndicatorDataRef::Candles { candles, .. } => {
            (candles.open.as_slice(), candles.close.as_slice())
        }
        IndicatorDataRef::Ohlc { open, close, .. } => (open, close),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "andean_oscillator".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "andean_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param("andean_oscillator", params, "length", 50)?;
            let signal_length = get_usize_param("andean_oscillator", params, "signal_length", 9)?;
            let input = AndeanOscillatorInput::from_slices(
                open,
                close,
                AndeanOscillatorParams {
                    length: Some(length),
                    signal_length: Some(signal_length),
                },
            );
            let out = andean_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "andean_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("bull") {
                return Ok(out.bull);
            }
            if output_id.eq_ignore_ascii_case("bear") {
                return Ok(out.bear);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "andean_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_daily_factor_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("daily_factor", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "daily_factor",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let threshold_level = get_f64_param("daily_factor", params, "threshold_level", 0.35)?;
            let input = DailyFactorInput::from_slices(
                open,
                high,
                low,
                close,
                DailyFactorParams {
                    threshold_level: Some(threshold_level),
                },
            );
            let out = daily_factor_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "daily_factor".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.value);
            }
            if output_id.eq_ignore_ascii_case("ema") {
                return Ok(out.ema);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "daily_factor".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn parse_moving_average_cross_probability_ma_type(
    indicator: &str,
    key: &str,
    value: &str,
) -> Result<MovingAverageCrossProbabilityMaType, IndicatorDispatchError> {
    MovingAverageCrossProbabilityMaType::from_str(value).map_err(|_| {
        IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: key.to_string(),
            reason: format!("invalid enum value: {value}"),
        }
    })
}

fn compute_moving_average_cross_probability_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("moving_average_cross_probability", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "moving_average_cross_probability",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let ma_type = parse_moving_average_cross_probability_ma_type(
                "moving_average_cross_probability",
                "ma_type",
                &get_enum_param("moving_average_cross_probability", params, "ma_type", "ema")?,
            )?;
            let input = MovingAverageCrossProbabilityInput::from_slice(
                data,
                MovingAverageCrossProbabilityParams {
                    ma_type: Some(ma_type),
                    smoothing_window: Some(get_usize_param(
                        "moving_average_cross_probability",
                        params,
                        "smoothing_window",
                        7,
                    )?),
                    slow_length: Some(get_usize_param(
                        "moving_average_cross_probability",
                        params,
                        "slow_length",
                        30,
                    )?),
                    fast_length: Some(get_usize_param(
                        "moving_average_cross_probability",
                        params,
                        "fast_length",
                        14,
                    )?),
                    resolution: Some(get_usize_param(
                        "moving_average_cross_probability",
                        params,
                        "resolution",
                        50,
                    )?),
                },
            );
            let out =
                moving_average_cross_probability_with_kernel(&input, kernel).map_err(|e| {
                    IndicatorDispatchError::ComputeFailed {
                        indicator: "moving_average_cross_probability".to_string(),
                        details: e.to_string(),
                    }
                })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.value);
            }
            if output_id.eq_ignore_ascii_case("slow_ma") {
                return Ok(out.slow_ma);
            }
            if output_id.eq_ignore_ascii_case("fast_ma") {
                return Ok(out.fast_ma);
            }
            if output_id.eq_ignore_ascii_case("forecast") {
                return Ok(out.forecast);
            }
            if output_id.eq_ignore_ascii_case("upper") {
                return Ok(out.upper);
            }
            if output_id.eq_ignore_ascii_case("lower") {
                return Ok(out.lower);
            }
            if output_id.eq_ignore_ascii_case("direction") {
                return Ok(out.direction);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "moving_average_cross_probability".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn parse_bulls_v_bears_ma_type(
    indicator: &str,
    key: &str,
    value: &str,
) -> Result<BullsVBearsMaType, IndicatorDispatchError> {
    BullsVBearsMaType::from_str(value).map_err(|_| IndicatorDispatchError::InvalidParam {
        indicator: indicator.to_string(),
        key: key.to_string(),
        reason: format!("invalid enum value: {value}"),
    })
}

fn parse_bulls_v_bears_calculation_method(
    indicator: &str,
    key: &str,
    value: &str,
) -> Result<BullsVBearsCalculationMethod, IndicatorDispatchError> {
    BullsVBearsCalculationMethod::from_str(value).map_err(|_| {
        IndicatorDispatchError::InvalidParam {
            indicator: indicator.to_string(),
            key: key.to_string(),
            reason: format!("invalid enum value: {value}"),
        }
    })
}

fn compute_bulls_v_bears_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("bulls_v_bears", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "bulls_v_bears",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let ma_type = parse_bulls_v_bears_ma_type(
                "bulls_v_bears",
                "ma_type",
                &get_enum_param("bulls_v_bears", params, "ma_type", "ema")?,
            )?;
            let calculation_method = parse_bulls_v_bears_calculation_method(
                "bulls_v_bears",
                "calculation_method",
                &get_enum_param("bulls_v_bears", params, "calculation_method", "normalized")?,
            )?;
            let input = BullsVBearsInput::from_slices(
                high,
                low,
                close,
                BullsVBearsParams {
                    period: Some(get_usize_param("bulls_v_bears", params, "period", 14)?),
                    ma_type: Some(ma_type),
                    calculation_method: Some(calculation_method),
                    normalized_bars_back: Some(get_usize_param(
                        "bulls_v_bears",
                        params,
                        "normalized_bars_back",
                        120,
                    )?),
                    raw_rolling_period: Some(get_usize_param(
                        "bulls_v_bears",
                        params,
                        "raw_rolling_period",
                        50,
                    )?),
                    raw_threshold_percentile: Some(get_f64_param(
                        "bulls_v_bears",
                        params,
                        "raw_threshold_percentile",
                        95.0,
                    )?),
                    threshold_level: Some(get_f64_param(
                        "bulls_v_bears",
                        params,
                        "threshold_level",
                        80.0,
                    )?),
                },
            );
            let out = bulls_v_bears_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "bulls_v_bears".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.value);
            }
            if output_id.eq_ignore_ascii_case("bull") {
                return Ok(out.bull);
            }
            if output_id.eq_ignore_ascii_case("bear") {
                return Ok(out.bear);
            }
            if output_id.eq_ignore_ascii_case("ma") {
                return Ok(out.ma);
            }
            if output_id.eq_ignore_ascii_case("upper") {
                return Ok(out.upper);
            }
            if output_id.eq_ignore_ascii_case("lower") {
                return Ok(out.lower);
            }
            if output_id.eq_ignore_ascii_case("bullish_signal") {
                return Ok(out.bullish_signal);
            }
            if output_id.eq_ignore_ascii_case("bearish_signal") {
                return Ok(out.bearish_signal);
            }
            if output_id.eq_ignore_ascii_case("zero_cross_up") {
                return Ok(out.zero_cross_up);
            }
            if output_id.eq_ignore_ascii_case("zero_cross_down") {
                return Ok(out.zero_cross_down);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "bulls_v_bears".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_regression_slope_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("regression_slope_oscillator", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "regression_slope_oscillator",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let input = RegressionSlopeOscillatorInput::from_slice(
                data,
                RegressionSlopeOscillatorParams {
                    min_range: Some(get_usize_param(
                        "regression_slope_oscillator",
                        params,
                        "min_range",
                        10,
                    )?),
                    max_range: Some(get_usize_param(
                        "regression_slope_oscillator",
                        params,
                        "max_range",
                        100,
                    )?),
                    step: Some(get_usize_param(
                        "regression_slope_oscillator",
                        params,
                        "step",
                        5,
                    )?),
                    signal_line: Some(get_usize_param(
                        "regression_slope_oscillator",
                        params,
                        "signal_line",
                        7,
                    )?),
                },
            );
            let out = regression_slope_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "regression_slope_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.value);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            if output_id.eq_ignore_ascii_case("bullish_reversal") {
                return Ok(out.bullish_reversal);
            }
            if output_id.eq_ignore_ascii_case("bearish_reversal") {
                return Ok(out.bearish_reversal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "regression_slope_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn parse_smooth_theil_sen_stat_style(
    indicator: &str,
    key: &str,
    value: &str,
) -> Result<SmoothTheilSenStatStyle, IndicatorDispatchError> {
    SmoothTheilSenStatStyle::from_str(value).map_err(|_| IndicatorDispatchError::InvalidParam {
        indicator: indicator.to_string(),
        key: key.to_string(),
        reason: format!("invalid enum value: {value}"),
    })
}

fn parse_smooth_theil_sen_deviation_style(
    indicator: &str,
    key: &str,
    value: &str,
) -> Result<SmoothTheilSenDeviationType, IndicatorDispatchError> {
    SmoothTheilSenDeviationType::from_str(value).map_err(|_| IndicatorDispatchError::InvalidParam {
        indicator: indicator.to_string(),
        key: key.to_string(),
        reason: format!("invalid enum value: {value}"),
    })
}

fn compute_smooth_theil_sen_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let kernel = req.kernel.to_non_batch();
    let (data, default_source): (&[f64], &str) = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            let default_source = source.unwrap_or("close");
            (source_type(candles, default_source), default_source)
        }
        IndicatorDataRef::Slice { values } => (values, "close"),
        IndicatorDataRef::Ohlc { close, .. } => (close, "close"),
        IndicatorDataRef::Ohlcv { close, .. } => (close, "close"),
        IndicatorDataRef::CloseVolume { close, .. } => (close, "close"),
        IndicatorDataRef::HighLow { .. } => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "smooth_theil_sen".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    collect_f64(
        "smooth_theil_sen",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let series = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    let source =
                        get_enum_param("smooth_theil_sen", params, "source", default_source)?;
                    source_type(candles, &source)
                }
                _ => data,
            };
            let slope_style = parse_smooth_theil_sen_stat_style(
                "smooth_theil_sen",
                "slope_style",
                &get_enum_param("smooth_theil_sen", params, "slope_style", "smooth_median")?,
            )?;
            let residual_style = parse_smooth_theil_sen_stat_style(
                "smooth_theil_sen",
                "residual_style",
                &get_enum_param(
                    "smooth_theil_sen",
                    params,
                    "residual_style",
                    "smooth_median",
                )?,
            )?;
            let deviation_style = parse_smooth_theil_sen_deviation_style(
                "smooth_theil_sen",
                "deviation_style",
                &get_enum_param("smooth_theil_sen", params, "deviation_style", "mad")?,
            )?;
            let mad_style = parse_smooth_theil_sen_stat_style(
                "smooth_theil_sen",
                "mad_style",
                &get_enum_param("smooth_theil_sen", params, "mad_style", "smooth_median")?,
            )?;
            let input = SmoothTheilSenInput::from_slice(
                series,
                SmoothTheilSenParams {
                    length: Some(get_usize_param("smooth_theil_sen", params, "length", 25)?),
                    offset: Some(get_usize_param("smooth_theil_sen", params, "offset", 0)?),
                    multiplier: Some(get_f64_param(
                        "smooth_theil_sen",
                        params,
                        "multiplier",
                        2.0,
                    )?),
                    slope_style: Some(slope_style),
                    residual_style: Some(residual_style),
                    deviation_style: Some(deviation_style),
                    mad_style: Some(mad_style),
                    include_prediction_in_deviation: Some(get_bool_param(
                        "smooth_theil_sen",
                        params,
                        "include_prediction_in_deviation",
                        false,
                    )?),
                },
            );
            let out = smooth_theil_sen_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "smooth_theil_sen".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.value);
            }
            if output_id.eq_ignore_ascii_case("upper") {
                return Ok(out.upper);
            }
            if output_id.eq_ignore_ascii_case("lower") {
                return Ok(out.lower);
            }
            if output_id.eq_ignore_ascii_case("slope") {
                return Ok(out.slope);
            }
            if output_id.eq_ignore_ascii_case("intercept") {
                return Ok(out.intercept);
            }
            if output_id.eq_ignore_ascii_case("deviation") {
                return Ok(out.deviation);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "smooth_theil_sen".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_cycle_channel_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close, source, default_source): (&[f64], &[f64], &[f64], &[f64], &str) =
        match req.data {
            IndicatorDataRef::Candles { candles, source } => {
                let default_source = source.unwrap_or("close");
                (
                    candles.high.as_slice(),
                    candles.low.as_slice(),
                    candles.close.as_slice(),
                    source_type(candles, default_source),
                    default_source,
                )
            }
            IndicatorDataRef::Ohlc {
                high, low, close, ..
            } => (high, low, close, close, "close"),
            _ => {
                return Err(IndicatorDispatchError::MissingRequiredInput {
                    indicator: "cycle_channel_oscillator".to_string(),
                    input: IndicatorInputKind::Candles,
                });
            }
        };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "cycle_channel_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let selected_source =
                get_enum_param("cycle_channel_oscillator", params, "source", default_source)?;
            let source_values = match req.data {
                IndicatorDataRef::Candles { candles, .. } => source_type(candles, &selected_source),
                _ => source,
            };
            let input = CycleChannelOscillatorInput::from_slices(
                source_values,
                high,
                low,
                close,
                CycleChannelOscillatorParams {
                    short_cycle_length: Some(get_usize_param(
                        "cycle_channel_oscillator",
                        params,
                        "short_cycle_length",
                        10,
                    )?),
                    medium_cycle_length: Some(get_usize_param(
                        "cycle_channel_oscillator",
                        params,
                        "medium_cycle_length",
                        30,
                    )?),
                    short_multiplier: Some(get_f64_param(
                        "cycle_channel_oscillator",
                        params,
                        "short_multiplier",
                        1.0,
                    )?),
                    medium_multiplier: Some(get_f64_param(
                        "cycle_channel_oscillator",
                        params,
                        "medium_multiplier",
                        3.0,
                    )?),
                },
            );
            let out = cycle_channel_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "cycle_channel_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("fast") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.fast);
            }
            if output_id.eq_ignore_ascii_case("slow") {
                return Ok(out.slow);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "cycle_channel_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_ewma_volatility_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("ewma_volatility", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ewma_volatility",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let lambda = get_f64_param("ewma_volatility", params, "lambda", 0.94)?;
            let input = EwmaVolatilityInput::from_slice(
                data,
                EwmaVolatilityParams {
                    lambda: Some(lambda),
                },
            );
            let out = ewma_volatility_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "ewma_volatility".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.values);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "ewma_volatility".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_accumulation_swing_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("accumulation_swing_index", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "accumulation_swing_index",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let daily_limit =
                get_f64_param("accumulation_swing_index", params, "daily_limit", 10_000.0)?;
            let input = AccumulationSwingIndexInput::from_slices(
                open,
                high,
                low,
                close,
                AccumulationSwingIndexParams {
                    daily_limit: Some(daily_limit),
                },
            );
            let out = accumulation_swing_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "accumulation_swing_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.values);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "accumulation_swing_index".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_ichimoku_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close, source): (&[f64], &[f64], &[f64], &[f64]) = match req.data {
        IndicatorDataRef::Candles { candles, source } => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            source_type(candles, source.unwrap_or("close")),
        ),
        IndicatorDataRef::Ohlc {
            high, low, close, ..
        } => (high, low, close, close),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "ichimoku_oscillator".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ichimoku_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let normalize = get_enum_param("ichimoku_oscillator", params, "normalize", "window")?
                .parse::<IchimokuOscillatorNormalizeMode>()
                .map_err(|reason| IndicatorDispatchError::InvalidParam {
                    indicator: "ichimoku_oscillator".to_string(),
                    key: "normalize".to_string(),
                    reason,
                })?;
            let input = IchimokuOscillatorInput::from_slices(
                high,
                low,
                close,
                source,
                IchimokuOscillatorParams {
                    conversion_periods: Some(get_usize_param(
                        "ichimoku_oscillator",
                        params,
                        "conversion_periods",
                        9,
                    )?),
                    base_periods: Some(get_usize_param(
                        "ichimoku_oscillator",
                        params,
                        "base_periods",
                        26,
                    )?),
                    lagging_span_periods: Some(get_usize_param(
                        "ichimoku_oscillator",
                        params,
                        "lagging_span_periods",
                        52,
                    )?),
                    displacement: Some(get_usize_param(
                        "ichimoku_oscillator",
                        params,
                        "displacement",
                        26,
                    )?),
                    ma_length: Some(get_usize_param(
                        "ichimoku_oscillator",
                        params,
                        "ma_length",
                        12,
                    )?),
                    smoothing_length: Some(get_usize_param(
                        "ichimoku_oscillator",
                        params,
                        "smoothing_length",
                        3,
                    )?),
                    extra_smoothing: Some(get_bool_param(
                        "ichimoku_oscillator",
                        params,
                        "extra_smoothing",
                        true,
                    )?),
                    normalize: Some(normalize),
                    window_size: Some(get_usize_param(
                        "ichimoku_oscillator",
                        params,
                        "window_size",
                        20,
                    )?),
                    clamp: Some(get_bool_param(
                        "ichimoku_oscillator",
                        params,
                        "clamp",
                        true,
                    )?),
                    top_band: Some(get_f64_param(
                        "ichimoku_oscillator",
                        params,
                        "top_band",
                        2.0,
                    )?),
                    mid_band: Some(get_f64_param(
                        "ichimoku_oscillator",
                        params,
                        "mid_band",
                        1.5,
                    )?),
                },
            );
            let out = ichimoku_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "ichimoku_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            let value = if output_id.eq_ignore_ascii_case("signal")
                || output_id.eq_ignore_ascii_case("value")
            {
                out.signal
            } else if output_id.eq_ignore_ascii_case("ma") {
                out.ma
            } else if output_id.eq_ignore_ascii_case("conversion") {
                out.conversion
            } else if output_id.eq_ignore_ascii_case("base") {
                out.base
            } else if output_id.eq_ignore_ascii_case("chikou") {
                out.chikou
            } else if output_id.eq_ignore_ascii_case("current_kumo_a") {
                out.current_kumo_a
            } else if output_id.eq_ignore_ascii_case("current_kumo_b") {
                out.current_kumo_b
            } else if output_id.eq_ignore_ascii_case("future_kumo_a") {
                out.future_kumo_a
            } else if output_id.eq_ignore_ascii_case("future_kumo_b") {
                out.future_kumo_b
            } else if output_id.eq_ignore_ascii_case("max_level") {
                out.max_level
            } else if output_id.eq_ignore_ascii_case("high_level") {
                out.high_level
            } else if output_id.eq_ignore_ascii_case("low_level") {
                out.low_level
            } else if output_id.eq_ignore_ascii_case("min_level") {
                out.min_level
            } else {
                return Err(IndicatorDispatchError::UnknownOutput {
                    indicator: "ichimoku_oscillator".to_string(),
                    output: output_id.to_string(),
                });
            };
            Ok(value)
        },
    )
}

fn compute_mesa_stochastic_multi_length_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (candles, default_source) = match req.data {
        IndicatorDataRef::Candles { candles, source } => (candles, source.unwrap_or("close")),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "mesa_stochastic_multi_length".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "mesa_stochastic_multi_length",
        output_id,
        req.combos,
        candles.close.len(),
        |params| {
            let source = get_enum_param(
                "mesa_stochastic_multi_length",
                params,
                "source",
                default_source,
            )?;
            let input = MesaStochasticMultiLengthInput::from_slices(
                source_type(candles, &source),
                MesaStochasticMultiLengthParams {
                    length_1: Some(get_usize_param(
                        "mesa_stochastic_multi_length",
                        params,
                        "length_1",
                        48,
                    )?),
                    length_2: Some(get_usize_param(
                        "mesa_stochastic_multi_length",
                        params,
                        "length_2",
                        21,
                    )?),
                    length_3: Some(get_usize_param(
                        "mesa_stochastic_multi_length",
                        params,
                        "length_3",
                        9,
                    )?),
                    length_4: Some(get_usize_param(
                        "mesa_stochastic_multi_length",
                        params,
                        "length_4",
                        6,
                    )?),
                    trigger_length: Some(get_usize_param(
                        "mesa_stochastic_multi_length",
                        params,
                        "trigger_length",
                        2,
                    )?),
                },
            );
            let out = mesa_stochastic_multi_length_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "mesa_stochastic_multi_length".to_string(),
                    details: e.to_string(),
                }
            })?;
            let value = if output_id.eq_ignore_ascii_case("mesa_1")
                || output_id.eq_ignore_ascii_case("value")
            {
                out.mesa_1
            } else if output_id.eq_ignore_ascii_case("mesa_2") {
                out.mesa_2
            } else if output_id.eq_ignore_ascii_case("mesa_3") {
                out.mesa_3
            } else if output_id.eq_ignore_ascii_case("mesa_4") {
                out.mesa_4
            } else if output_id.eq_ignore_ascii_case("trigger_1") {
                out.trigger_1
            } else if output_id.eq_ignore_ascii_case("trigger_2") {
                out.trigger_2
            } else if output_id.eq_ignore_ascii_case("trigger_3") {
                out.trigger_3
            } else if output_id.eq_ignore_ascii_case("trigger_4") {
                out.trigger_4
            } else {
                return Err(IndicatorDispatchError::UnknownOutput {
                    indicator: "mesa_stochastic_multi_length".to_string(),
                    output: output_id.to_string(),
                });
            };
            Ok(value)
        },
    )
}

fn compute_spearman_correlation_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (candles, default_source) = match req.data {
        IndicatorDataRef::Candles { candles, source } => (candles, source.unwrap_or("close")),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "spearman_correlation".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "spearman_correlation",
        output_id,
        req.combos,
        candles.close.len(),
        |params| {
            let source = get_enum_param("spearman_correlation", params, "source", default_source)?;
            let comparison_source =
                get_enum_param("spearman_correlation", params, "comparison_source", "open")?;
            let input = SpearmanCorrelationInput::from_slices(
                source_type(candles, &source),
                source_type(candles, &comparison_source),
                SpearmanCorrelationParams {
                    lookback: Some(get_usize_param(
                        "spearman_correlation",
                        params,
                        "lookback",
                        30,
                    )?),
                    smoothing_length: Some(get_usize_param(
                        "spearman_correlation",
                        params,
                        "smoothing_length",
                        3,
                    )?),
                },
            );
            let out = spearman_correlation_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "spearman_correlation".to_string(),
                    details: e.to_string(),
                }
            })?;
            let value = if output_id.eq_ignore_ascii_case("smoothed")
                || output_id.eq_ignore_ascii_case("value")
            {
                out.smoothed
            } else if output_id.eq_ignore_ascii_case("raw") {
                out.raw
            } else {
                return Err(IndicatorDispatchError::UnknownOutput {
                    indicator: "spearman_correlation".to_string(),
                    output: output_id.to_string(),
                });
            };
            Ok(value)
        },
    )
}

fn compute_random_walk_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("random_walk_index", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "random_walk_index",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param("random_walk_index", params, "length", 14)?;
            let input = RandomWalkIndexInput::from_slices(
                high,
                low,
                close,
                RandomWalkIndexParams {
                    length: Some(length),
                },
            );
            let out = random_walk_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "random_walk_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("high") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.high);
            }
            if output_id.eq_ignore_ascii_case("low") {
                return Ok(out.low);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "random_walk_index".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_price_moving_average_ratio_percentile_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (candles, default_source) = match req.data {
        IndicatorDataRef::Candles { candles, source } => (candles, source.unwrap_or("close")),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "price_moving_average_ratio_percentile".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "price_moving_average_ratio_percentile",
        output_id,
        req.combos,
        candles.close.len(),
        |params| {
            let source = get_enum_param(
                "price_moving_average_ratio_percentile",
                params,
                "source",
                default_source,
            )?;
            let ma_type = get_enum_param(
                "price_moving_average_ratio_percentile",
                params,
                "ma_type",
                "vwma",
            )?
            .parse::<PriceMovingAverageRatioPercentileMaType>()
            .map_err(|reason| IndicatorDispatchError::InvalidParam {
                indicator: "price_moving_average_ratio_percentile".to_string(),
                key: "ma_type".to_string(),
                reason,
            })?;
            let signal_ma_type = get_enum_param(
                "price_moving_average_ratio_percentile",
                params,
                "signal_ma_type",
                "sma",
            )?
            .parse::<PriceMovingAverageRatioPercentileMaType>()
            .map_err(|reason| IndicatorDispatchError::InvalidParam {
                indicator: "price_moving_average_ratio_percentile".to_string(),
                key: "signal_ma_type".to_string(),
                reason,
            })?;
            let line_mode = get_enum_param(
                "price_moving_average_ratio_percentile",
                params,
                "line_mode",
                "pmarp",
            )?
            .parse::<PriceMovingAverageRatioPercentileLineMode>()
            .map_err(|reason| IndicatorDispatchError::InvalidParam {
                indicator: "price_moving_average_ratio_percentile".to_string(),
                key: "line_mode".to_string(),
                reason,
            })?;
            let price = source_type(candles, &source);
            let input = PriceMovingAverageRatioPercentileInput::from_slices(
                price,
                candles.volume.as_slice(),
                PriceMovingAverageRatioPercentileParams {
                    ma_length: Some(get_usize_param(
                        "price_moving_average_ratio_percentile",
                        params,
                        "ma_length",
                        20,
                    )?),
                    ma_type: Some(ma_type),
                    pmarp_lookback: Some(get_usize_param(
                        "price_moving_average_ratio_percentile",
                        params,
                        "pmarp_lookback",
                        350,
                    )?),
                    signal_ma_length: Some(get_usize_param(
                        "price_moving_average_ratio_percentile",
                        params,
                        "signal_ma_length",
                        20,
                    )?),
                    signal_ma_type: Some(signal_ma_type),
                    line_mode: Some(line_mode),
                },
            );
            let out =
                price_moving_average_ratio_percentile_with_kernel(&input, kernel).map_err(|e| {
                    IndicatorDispatchError::ComputeFailed {
                        indicator: "price_moving_average_ratio_percentile".to_string(),
                        details: e.to_string(),
                    }
                })?;
            if output_id.eq_ignore_ascii_case("value") || output_id.eq_ignore_ascii_case("plotline")
            {
                return Ok(out.plotline);
            }
            if output_id.eq_ignore_ascii_case("pmar") {
                return Ok(out.pmar);
            }
            if output_id.eq_ignore_ascii_case("pmarp") {
                return Ok(out.pmarp);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            if output_id.eq_ignore_ascii_case("pmar_high") {
                return Ok(out.pmar_high);
            }
            if output_id.eq_ignore_ascii_case("pmar_low") {
                return Ok(out.pmar_low);
            }
            if output_id.eq_ignore_ascii_case("scaled_pmar") {
                return Ok(out.scaled_pmar);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "price_moving_average_ratio_percentile".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_relative_strength_index_wave_indicator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    enum Inputs<'a> {
        Candles {
            candles: &'a crate::utilities::data_loader::Candles,
            default_source: &'a str,
        },
        Ohlc {
            open: &'a [f64],
            high: &'a [f64],
            low: &'a [f64],
            close: &'a [f64],
        },
    }

    let inputs = match req.data {
        IndicatorDataRef::Candles { candles, source } => Inputs::Candles {
            candles,
            default_source: source.unwrap_or("hlcc4"),
        },
        IndicatorDataRef::Ohlc {
            open,
            high,
            low,
            close,
        } => Inputs::Ohlc {
            open,
            high,
            low,
            close,
        },
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "relative_strength_index_wave_indicator".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };

    let len = match &inputs {
        Inputs::Candles { candles, .. } => candles.close.len(),
        Inputs::Ohlc { close, .. } => close.len(),
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "relative_strength_index_wave_indicator",
        output_id,
        req.combos,
        len,
        |params| {
            let default_source = match &inputs {
                Inputs::Candles { default_source, .. } => *default_source,
                Inputs::Ohlc { .. } => "hlcc4",
            };
            let source_name = get_enum_param(
                "relative_strength_index_wave_indicator",
                params,
                "source",
                default_source,
            )?;
            let rsi_length = get_usize_param(
                "relative_strength_index_wave_indicator",
                params,
                "rsi_length",
                14,
            )?;
            let length1 = get_usize_param(
                "relative_strength_index_wave_indicator",
                params,
                "length1",
                2,
            )?;
            let length2 = get_usize_param(
                "relative_strength_index_wave_indicator",
                params,
                "length2",
                5,
            )?;
            let length3 = get_usize_param(
                "relative_strength_index_wave_indicator",
                params,
                "length3",
                9,
            )?;
            let length4 = get_usize_param(
                "relative_strength_index_wave_indicator",
                params,
                "length4",
                13,
            )?;

            let owned_source;
            let (source, high, low) = match &inputs {
                Inputs::Candles { candles, .. } => (
                    source_type(candles, &source_name),
                    candles.high.as_slice(),
                    candles.low.as_slice(),
                ),
                Inputs::Ohlc {
                    open,
                    high,
                    low,
                    close,
                } => {
                    owned_source = match source_name.as_str() {
                        "open" => open.to_vec(),
                        "high" => high.to_vec(),
                        "low" => low.to_vec(),
                        "close" => close.to_vec(),
                        "hl2" => high
                            .iter()
                            .zip(low.iter())
                            .map(|(h, l)| (h + l) * 0.5)
                            .collect::<Vec<_>>(),
                        "hlc3" => high
                            .iter()
                            .zip(low.iter())
                            .zip(close.iter())
                            .map(|((h, l), c)| (h + l + c) / 3.0)
                            .collect::<Vec<_>>(),
                        "ohlc4" => open
                            .iter()
                            .zip(high.iter())
                            .zip(low.iter())
                            .zip(close.iter())
                            .map(|(((o, h), l), c)| (o + h + l + c) * 0.25)
                            .collect::<Vec<_>>(),
                        "hlcc4" | "hlcc" => high
                            .iter()
                            .zip(low.iter())
                            .zip(close.iter())
                            .map(|((h, l), c)| (h + l + 2.0 * c) * 0.25)
                            .collect::<Vec<_>>(),
                        other => {
                            return Err(IndicatorDispatchError::InvalidParam {
                                indicator: "relative_strength_index_wave_indicator".to_string(),
                                key: "source".to_string(),
                                reason: format!("unsupported OHLC source '{other}'"),
                            });
                        }
                    };
                    (owned_source.as_slice(), *high, *low)
                }
            };

            let input = RelativeStrengthIndexWaveIndicatorInput::from_slices(
                source,
                high,
                low,
                RelativeStrengthIndexWaveIndicatorParams {
                    rsi_length: Some(rsi_length),
                    length1: Some(length1),
                    length2: Some(length2),
                    length3: Some(length3),
                    length4: Some(length4),
                },
            );
            let out = relative_strength_index_wave_indicator_with_kernel(&input, kernel).map_err(
                |e| IndicatorDispatchError::ComputeFailed {
                    indicator: "relative_strength_index_wave_indicator".to_string(),
                    details: e.to_string(),
                },
            )?;
            if output_id.eq_ignore_ascii_case("value") || output_id.eq_ignore_ascii_case("rsi_ma1")
            {
                return Ok(out.rsi_ma1);
            }
            if output_id.eq_ignore_ascii_case("rsi_ma2") {
                return Ok(out.rsi_ma2);
            }
            if output_id.eq_ignore_ascii_case("rsi_ma3") {
                return Ok(out.rsi_ma3);
            }
            if output_id.eq_ignore_ascii_case("rsi_ma4") {
                return Ok(out.rsi_ma4);
            }
            if output_id.eq_ignore_ascii_case("state") {
                return Ok(out.state);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "relative_strength_index_wave_indicator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_trend_trigger_factor_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low) = extract_high_low_input("trend_trigger_factor", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "trend_trigger_factor",
        output_id,
        req.combos,
        high.len(),
        |params| {
            let length = get_usize_param("trend_trigger_factor", params, "length", 15)?;
            let input = TrendTriggerFactorInput::from_slices(
                high,
                low,
                TrendTriggerFactorParams {
                    length: Some(length),
                },
            );
            let out = trend_trigger_factor_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "trend_trigger_factor".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.values);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "trend_trigger_factor".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_volatility_quality_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("volatility_quality_index", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "volatility_quality_index",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let fast_length =
                get_usize_param("volatility_quality_index", params, "fast_length", 9)?;
            let slow_length =
                get_usize_param("volatility_quality_index", params, "slow_length", 200)?;
            let input = VolatilityQualityIndexInput::from_slices(
                open,
                high,
                low,
                close,
                VolatilityQualityIndexParams {
                    fast_length: Some(fast_length),
                    slow_length: Some(slow_length),
                },
            );
            let out = volatility_quality_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "volatility_quality_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("vqi_sum") || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.vqi_sum);
            }
            if output_id.eq_ignore_ascii_case("fast_sma") {
                return Ok(out.fast_sma);
            }
            if output_id.eq_ignore_ascii_case("slow_sma") {
                return Ok(out.slow_sma);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "volatility_quality_index".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_vwap_deviation_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let candles = match req.data {
        IndicatorDataRef::Candles { candles, .. } => candles,
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "vwap_deviation_oscillator".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "vwap_deviation_oscillator",
        output_id,
        req.combos,
        candles.close.len(),
        |params| {
            let session_mode = get_enum_param(
                "vwap_deviation_oscillator",
                params,
                "session_mode",
                "rolling_bars",
            )?
            .parse::<VwapDeviationSessionMode>()
            .map_err(|reason| IndicatorDispatchError::InvalidParam {
                indicator: "vwap_deviation_oscillator".to_string(),
                key: "session_mode".to_string(),
                reason,
            })?;
            let deviation_mode = get_enum_param(
                "vwap_deviation_oscillator",
                params,
                "deviation_mode",
                "absolute",
            )?
            .parse::<VwapDeviationMode>()
            .map_err(|reason| IndicatorDispatchError::InvalidParam {
                indicator: "vwap_deviation_oscillator".to_string(),
                key: "deviation_mode".to_string(),
                reason,
            })?;
            let input = VwapDeviationOscillatorInput::from_candles(
                candles,
                VwapDeviationOscillatorParams {
                    session_mode: Some(session_mode),
                    rolling_period: Some(get_usize_param(
                        "vwap_deviation_oscillator",
                        params,
                        "rolling_period",
                        20,
                    )?),
                    rolling_days: Some(get_usize_param(
                        "vwap_deviation_oscillator",
                        params,
                        "rolling_days",
                        30,
                    )?),
                    use_close: Some(get_bool_param(
                        "vwap_deviation_oscillator",
                        params,
                        "use_close",
                        false,
                    )?),
                    deviation_mode: Some(deviation_mode),
                    z_window: Some(get_usize_param(
                        "vwap_deviation_oscillator",
                        params,
                        "z_window",
                        50,
                    )?),
                    pct_vol_lookback: Some(get_usize_param(
                        "vwap_deviation_oscillator",
                        params,
                        "pct_vol_lookback",
                        100,
                    )?),
                    pct_min_sigma: Some(get_f64_param(
                        "vwap_deviation_oscillator",
                        params,
                        "pct_min_sigma",
                        0.1,
                    )?),
                    abs_vol_lookback: Some(get_usize_param(
                        "vwap_deviation_oscillator",
                        params,
                        "abs_vol_lookback",
                        100,
                    )?),
                },
            );
            let out = vwap_deviation_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "vwap_deviation_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("osc") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.osc);
            }
            if output_id.eq_ignore_ascii_case("std1") {
                return Ok(out.std1);
            }
            if output_id.eq_ignore_ascii_case("std2") {
                return Ok(out.std2);
            }
            if output_id.eq_ignore_ascii_case("std3") {
                return Ok(out.std3);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "vwap_deviation_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_volume_zone_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (close, volume) = extract_close_volume_input("volume_zone_oscillator", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "volume_zone_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param("volume_zone_oscillator", params, "length", 14)?;
            let intraday_smoothing =
                get_bool_param("volume_zone_oscillator", params, "intraday_smoothing", true)?;
            let noise_filter =
                get_usize_param("volume_zone_oscillator", params, "noise_filter", 4)?;
            let input = VolumeZoneOscillatorInput::from_slices(
                close,
                volume,
                VolumeZoneOscillatorParams {
                    length: Some(length),
                    intraday_smoothing: Some(intraday_smoothing),
                    noise_filter: Some(noise_filter),
                },
            );
            let out = volume_zone_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "volume_zone_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.values);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "volume_zone_oscillator".to_string(),
                output: output_id.to_string(),
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

fn compute_ehlers_adaptive_cg_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("ehlers_adaptive_cg", req.data, "hl2")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ehlers_adaptive_cg",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let alpha = get_f64_param("ehlers_adaptive_cg", params, "alpha", 0.07)?;
            let input = EhlersAdaptiveCgInput::from_slice(
                data,
                EhlersAdaptiveCgParams { alpha: Some(alpha) },
            );
            let out = ehlers_adaptive_cg_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "ehlers_adaptive_cg".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("cg") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.cg);
            }
            if output_id.eq_ignore_ascii_case("trigger") {
                return Ok(out.trigger);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "ehlers_adaptive_cg".to_string(),
                output: output_id.to_string(),
            })
        },
    )
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

fn compute_qqe_weighted_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("qqe_weighted_oscillator", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "qqe_weighted_oscillator",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let length = get_usize_param("qqe_weighted_oscillator", params, "length", 14)?;
            let factor = get_f64_param("qqe_weighted_oscillator", params, "factor", 4.236)?;
            let smooth = get_usize_param("qqe_weighted_oscillator", params, "smooth", 5)?;
            let weight = get_f64_param("qqe_weighted_oscillator", params, "weight", 2.0)?;
            let input = QqeWeightedOscillatorInput::from_slice(
                data,
                QqeWeightedOscillatorParams {
                    length: Some(length),
                    factor: Some(factor),
                    smooth: Some(smooth),
                    weight: Some(weight),
                },
            );
            let out = qqe_weighted_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "qqe_weighted_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("rsi") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.rsi);
            }
            if output_id.eq_ignore_ascii_case("trailing_stop")
                || output_id.eq_ignore_ascii_case("ts")
            {
                return Ok(out.trailing_stop);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "qqe_weighted_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_forward_backward_exponential_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("forward_backward_exponential_oscillator", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "forward_backward_exponential_oscillator",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let length = get_usize_param(
                "forward_backward_exponential_oscillator",
                params,
                "length",
                20,
            )?;
            let smooth = get_usize_param(
                "forward_backward_exponential_oscillator",
                params,
                "smooth",
                10,
            )?;
            let input = ForwardBackwardExponentialOscillatorInput::from_slice(
                data,
                ForwardBackwardExponentialOscillatorParams {
                    length: Some(length),
                    smooth: Some(smooth),
                },
            );
            let out = forward_backward_exponential_oscillator_with_kernel(&input, kernel).map_err(
                |e| IndicatorDispatchError::ComputeFailed {
                    indicator: "forward_backward_exponential_oscillator".to_string(),
                    details: e.to_string(),
                },
            )?;
            if output_id.eq_ignore_ascii_case("forward_backward")
                || output_id.eq_ignore_ascii_case("value")
                || output_id.eq_ignore_ascii_case("fb")
            {
                return Ok(out.forward_backward);
            }
            if output_id.eq_ignore_ascii_case("backward")
                || output_id.eq_ignore_ascii_case("bwrd")
                || output_id.eq_ignore_ascii_case("bw")
            {
                return Ok(out.backward);
            }
            if output_id.eq_ignore_ascii_case("histogram") || output_id.eq_ignore_ascii_case("hist")
            {
                return Ok(out.histogram);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "forward_backward_exponential_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_adaptive_bounds_rsi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("adaptive_bounds_rsi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "adaptive_bounds_rsi",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let rsi_length = get_usize_param("adaptive_bounds_rsi", params, "rsi_length", 14)?;
            let alpha = get_f64_param("adaptive_bounds_rsi", params, "alpha", 0.1)?;
            let input = AdaptiveBoundsRsiInput::from_slice(
                data,
                AdaptiveBoundsRsiParams {
                    rsi_length: Some(rsi_length),
                    alpha: Some(alpha),
                },
            );
            let out = adaptive_bounds_rsi_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "adaptive_bounds_rsi".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("rsi") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.rsi);
            }
            if output_id.eq_ignore_ascii_case("lower_bound") || output_id.eq_ignore_ascii_case("c1")
            {
                return Ok(out.lower_bound);
            }
            if output_id.eq_ignore_ascii_case("lower_mid") || output_id.eq_ignore_ascii_case("c2") {
                return Ok(out.lower_mid);
            }
            if output_id.eq_ignore_ascii_case("mid") || output_id.eq_ignore_ascii_case("c3") {
                return Ok(out.mid);
            }
            if output_id.eq_ignore_ascii_case("upper_mid") || output_id.eq_ignore_ascii_case("c4") {
                return Ok(out.upper_mid);
            }
            if output_id.eq_ignore_ascii_case("upper_bound") || output_id.eq_ignore_ascii_case("c5")
            {
                return Ok(out.upper_bound);
            }
            if output_id.eq_ignore_ascii_case("regime") {
                return Ok(out.regime);
            }
            if output_id.eq_ignore_ascii_case("regime_flip")
                || output_id.eq_ignore_ascii_case("flip")
            {
                return Ok(out.regime_flip);
            }
            if output_id.eq_ignore_ascii_case("lower_signal") {
                return Ok(out.lower_signal);
            }
            if output_id.eq_ignore_ascii_case("upper_signal") {
                return Ok(out.upper_signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "adaptive_bounds_rsi".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_range_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("range_oscillator", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "range_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param("range_oscillator", params, "length", 50)?;
            let mult = get_f64_param("range_oscillator", params, "mult", 2.0)?;
            let input = RangeOscillatorInput::from_slices(
                high,
                low,
                close,
                RangeOscillatorParams {
                    length: Some(length),
                    mult: Some(mult),
                },
            );
            let out = range_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "range_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("oscillator")
                || output_id.eq_ignore_ascii_case("osc")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.oscillator);
            }
            if output_id.eq_ignore_ascii_case("ma") {
                return Ok(out.ma);
            }
            if output_id.eq_ignore_ascii_case("upper_band")
                || output_id.eq_ignore_ascii_case("upper")
            {
                return Ok(out.upper_band);
            }
            if output_id.eq_ignore_ascii_case("lower_band")
                || output_id.eq_ignore_ascii_case("lower")
            {
                return Ok(out.lower_band);
            }
            if output_id.eq_ignore_ascii_case("range_width")
                || output_id.eq_ignore_ascii_case("width")
            {
                return Ok(out.range_width);
            }
            if output_id.eq_ignore_ascii_case("in_range") {
                return Ok(out.in_range);
            }
            if output_id.eq_ignore_ascii_case("trend") {
                return Ok(out.trend);
            }
            if output_id.eq_ignore_ascii_case("break_up") {
                return Ok(out.break_up);
            }
            if output_id.eq_ignore_ascii_case("break_down") {
                return Ok(out.break_down);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "range_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_market_structure_confluence_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("market_structure_confluence", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "market_structure_confluence",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let swing_size =
                get_usize_param("market_structure_confluence", params, "swing_size", 10)?;
            let bos_confirmation = get_enum_param(
                "market_structure_confluence",
                params,
                "bos_confirmation",
                "Candle Close",
            )?;
            let basis_length =
                get_usize_param("market_structure_confluence", params, "basis_length", 100)?;
            let atr_length =
                get_usize_param("market_structure_confluence", params, "atr_length", 14)?;
            let atr_smooth =
                get_usize_param("market_structure_confluence", params, "atr_smooth", 21)?;
            let vol_mult =
                get_f64_param("market_structure_confluence", params, "vol_mult", 2.0)?;
            let input = MarketStructureConfluenceInput::from_slices(
                high,
                low,
                close,
                MarketStructureConfluenceParams {
                    swing_size: Some(swing_size),
                    bos_confirmation: Some(bos_confirmation),
                    basis_length: Some(basis_length),
                    atr_length: Some(atr_length),
                    atr_smooth: Some(atr_smooth),
                    vol_mult: Some(vol_mult),
                },
            );
            let out = market_structure_confluence_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "market_structure_confluence".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("basis") {
                return Ok(out.basis);
            }
            if output_id.eq_ignore_ascii_case("upper_band")
                || output_id.eq_ignore_ascii_case("upper")
            {
                return Ok(out.upper_band);
            }
            if output_id.eq_ignore_ascii_case("lower_band")
                || output_id.eq_ignore_ascii_case("lower")
            {
                return Ok(out.lower_band);
            }
            if output_id.eq_ignore_ascii_case("structure_direction")
                || output_id.eq_ignore_ascii_case("direction")
                || output_id.eq_ignore_ascii_case("trend")
            {
                return Ok(out.structure_direction);
            }
            if output_id.eq_ignore_ascii_case("bullish_arrow") {
                return Ok(out.bullish_arrow);
            }
            if output_id.eq_ignore_ascii_case("bearish_arrow") {
                return Ok(out.bearish_arrow);
            }
            if output_id.eq_ignore_ascii_case("bullish_change") {
                return Ok(out.bullish_change);
            }
            if output_id.eq_ignore_ascii_case("bearish_change") {
                return Ok(out.bearish_change);
            }
            if output_id.eq_ignore_ascii_case("hh") {
                return Ok(out.hh);
            }
            if output_id.eq_ignore_ascii_case("lh") {
                return Ok(out.lh);
            }
            if output_id.eq_ignore_ascii_case("hl") {
                return Ok(out.hl);
            }
            if output_id.eq_ignore_ascii_case("ll") {
                return Ok(out.ll);
            }
            if output_id.eq_ignore_ascii_case("bullish_bos") {
                return Ok(out.bullish_bos);
            }
            if output_id.eq_ignore_ascii_case("bullish_choch") {
                return Ok(out.bullish_choch);
            }
            if output_id.eq_ignore_ascii_case("bearish_bos") {
                return Ok(out.bearish_bos);
            }
            if output_id.eq_ignore_ascii_case("bearish_choch") {
                return Ok(out.bearish_choch);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "market_structure_confluence".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_range_filtered_trend_signals_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("range_filtered_trend_signals", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "range_filtered_trend_signals",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let kalman_alpha =
                get_f64_param("range_filtered_trend_signals", params, "kalman_alpha", 0.01)?;
            let kalman_beta =
                get_f64_param("range_filtered_trend_signals", params, "kalman_beta", 0.1)?;
            let kalman_period =
                get_usize_param("range_filtered_trend_signals", params, "kalman_period", 77)?;
            let dev = get_f64_param("range_filtered_trend_signals", params, "dev", 1.2)?;
            let supertrend_factor = get_f64_param(
                "range_filtered_trend_signals",
                params,
                "supertrend_factor",
                0.7,
            )?;
            let supertrend_atr_period = get_usize_param(
                "range_filtered_trend_signals",
                params,
                "supertrend_atr_period",
                7,
            )?;
            let input = RangeFilteredTrendSignalsInput::from_slices(
                high,
                low,
                close,
                RangeFilteredTrendSignalsParams {
                    kalman_alpha: Some(kalman_alpha),
                    kalman_beta: Some(kalman_beta),
                    kalman_period: Some(kalman_period),
                    dev: Some(dev),
                    supertrend_factor: Some(supertrend_factor),
                    supertrend_atr_period: Some(supertrend_atr_period),
                },
            );
            let out = range_filtered_trend_signals_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "range_filtered_trend_signals".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("kalman") {
                return Ok(out.kalman);
            }
            if output_id.eq_ignore_ascii_case("supertrend") {
                return Ok(out.supertrend);
            }
            if output_id.eq_ignore_ascii_case("upper_band")
                || output_id.eq_ignore_ascii_case("upper")
            {
                return Ok(out.upper_band);
            }
            if output_id.eq_ignore_ascii_case("lower_band")
                || output_id.eq_ignore_ascii_case("lower")
            {
                return Ok(out.lower_band);
            }
            if output_id.eq_ignore_ascii_case("trend") {
                return Ok(out.trend);
            }
            if output_id.eq_ignore_ascii_case("kalman_trend")
                || output_id.eq_ignore_ascii_case("long_trend")
            {
                return Ok(out.kalman_trend);
            }
            if output_id.eq_ignore_ascii_case("state") {
                return Ok(out.state);
            }
            if output_id.eq_ignore_ascii_case("market_trending") {
                return Ok(out.market_trending);
            }
            if output_id.eq_ignore_ascii_case("market_ranging") {
                return Ok(out.market_ranging);
            }
            if output_id.eq_ignore_ascii_case("short_term_bullish") {
                return Ok(out.short_term_bullish);
            }
            if output_id.eq_ignore_ascii_case("short_term_bearish") {
                return Ok(out.short_term_bearish);
            }
            if output_id.eq_ignore_ascii_case("long_term_bullish") {
                return Ok(out.long_term_bullish);
            }
            if output_id.eq_ignore_ascii_case("long_term_bearish") {
                return Ok(out.long_term_bearish);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "range_filtered_trend_signals".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_volume_weighted_relative_strength_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (source, volume) =
        extract_close_volume_input("volume_weighted_relative_strength_index", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "volume_weighted_relative_strength_index",
        output_id,
        req.combos,
        source.len(),
        |params| {
            let rsi_length = get_usize_param(
                "volume_weighted_relative_strength_index",
                params,
                "rsi_length",
                14,
            )?;
            let range_length = get_usize_param(
                "volume_weighted_relative_strength_index",
                params,
                "range_length",
                10,
            )?;
            let ma_length = get_usize_param(
                "volume_weighted_relative_strength_index",
                params,
                "ma_length",
                14,
            )?;
            let ma_type = get_enum_param(
                "volume_weighted_relative_strength_index",
                params,
                "ma_type",
                "EMA",
            )?;
            let input = VolumeWeightedRelativeStrengthIndexInput::from_slices(
                source,
                volume,
                VolumeWeightedRelativeStrengthIndexParams {
                    rsi_length: Some(rsi_length),
                    range_length: Some(range_length),
                    ma_length: Some(ma_length),
                    ma_type: Some(ma_type),
                },
            );
            let out = volume_weighted_relative_strength_index_with_kernel(&input, kernel).map_err(
                |e| IndicatorDispatchError::ComputeFailed {
                    indicator: "volume_weighted_relative_strength_index".to_string(),
                    details: e.to_string(),
                },
            )?;
            if output_id.eq_ignore_ascii_case("rsi") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.rsi);
            }
            if output_id.eq_ignore_ascii_case("consolidation_strength")
                || output_id.eq_ignore_ascii_case("consolidation")
            {
                return Ok(out.consolidation_strength);
            }
            if output_id.eq_ignore_ascii_case("rsi_ma") || output_id.eq_ignore_ascii_case("ma") {
                return Ok(out.rsi_ma);
            }
            if output_id.eq_ignore_ascii_case("bearish_tp") {
                return Ok(out.bearish_tp);
            }
            if output_id.eq_ignore_ascii_case("bullish_tp") {
                return Ok(out.bullish_tp);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "volume_weighted_relative_strength_index".to_string(),
                output: output_id.to_string(),
            })
        },
    )
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
    use crate::indicators::accumulation_swing_index::{
        accumulation_swing_index_with_kernel, AccumulationSwingIndexInput,
        AccumulationSwingIndexParams,
    };
    use crate::indicators::ad::{ad_with_kernel, AdInput, AdParams};
    use crate::indicators::adx::{adx_with_kernel, AdxInput, AdxParams};
    use crate::indicators::ao::{ao_with_kernel, AoInput, AoParams};
    use crate::indicators::apo::{apo_with_kernel, ApoInput, ApoParams};
    use crate::indicators::cg::{cg_with_kernel, CgInput, CgParams};
    use crate::indicators::cmo::{cmo_with_kernel, CmoInput, CmoParams};
    use crate::indicators::cycle_channel_oscillator::{
        cycle_channel_oscillator_with_kernel, CycleChannelOscillatorInput,
        CycleChannelOscillatorParams,
    };
    use crate::indicators::daily_factor::{
        daily_factor_with_kernel, DailyFactorInput, DailyFactorParams,
    };
    use crate::indicators::deviation::{deviation_with_kernel, DeviationInput, DeviationParams};
    use crate::indicators::dx::{
        dx_batch_with_kernel, dx_with_kernel, DxBatchRange, DxInput, DxParams,
    };
    use crate::indicators::efi::{efi_with_kernel, EfiInput, EfiParams};
    use crate::indicators::ehlers_adaptive_cyber_cycle::{
        ehlers_adaptive_cyber_cycle_with_kernel, EhlersAdaptiveCyberCycleInput,
        EhlersAdaptiveCyberCycleParams,
    };
    use crate::indicators::ehlers_simple_cycle_indicator::{
        ehlers_simple_cycle_indicator_with_kernel, EhlersSimpleCycleIndicatorInput,
        EhlersSimpleCycleIndicatorParams,
    };
    use crate::indicators::ehlers_smoothed_adaptive_momentum::{
        ehlers_smoothed_adaptive_momentum_with_kernel, EhlersSmoothedAdaptiveMomentumInput,
        EhlersSmoothedAdaptiveMomentumParams,
    };
    use crate::indicators::ewma_volatility::{
        ewma_volatility_with_kernel, EwmaVolatilityInput, EwmaVolatilityParams,
    };
    use crate::indicators::fosc::{fosc_with_kernel, FoscInput, FoscParams};
    use crate::indicators::ift_rsi::{ift_rsi_with_kernel, IftRsiInput, IftRsiParams};
    use crate::indicators::kvo::{kvo_with_kernel, KvoInput, KvoParams};
    use crate::indicators::l2_ehlers_signal_to_noise::{
        l2_ehlers_signal_to_noise_with_kernel, L2EhlersSignalToNoiseInput,
        L2EhlersSignalToNoiseParams,
    };
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
    use crate::indicators::mesa_stochastic_multi_length::{
        mesa_stochastic_multi_length_with_kernel, MesaStochasticMultiLengthInput,
        MesaStochasticMultiLengthParams,
    };
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
    use crate::indicators::price_moving_average_ratio_percentile::{
        price_moving_average_ratio_percentile_with_kernel, PriceMovingAverageRatioPercentileInput,
        PriceMovingAverageRatioPercentileLineMode, PriceMovingAverageRatioPercentileMaType,
        PriceMovingAverageRatioPercentileParams,
    };
    use crate::indicators::pvi::{pvi_with_kernel, PviInput, PviParams};
    use crate::indicators::random_walk_index::{
        random_walk_index_with_kernel, RandomWalkIndexInput, RandomWalkIndexParams,
    };
    use crate::indicators::registry::{list_indicators, IndicatorParamKind};
    use crate::indicators::relative_strength_index_wave_indicator::{
        relative_strength_index_wave_indicator_with_kernel,
        RelativeStrengthIndexWaveIndicatorInput, RelativeStrengthIndexWaveIndicatorParams,
    };
    use crate::indicators::spearman_correlation::{
        spearman_correlation_with_kernel, SpearmanCorrelationInput, SpearmanCorrelationParams,
    };
    use crate::indicators::trend_trigger_factor::{
        trend_trigger_factor_with_kernel, TrendTriggerFactorInput, TrendTriggerFactorParams,
    };
    use crate::indicators::trix::{
        trix_batch_with_kernel, trix_with_kernel, TrixBatchRange, TrixInput, TrixParams,
    };
    use crate::indicators::ttm_trend::{ttm_trend_with_kernel, TtmTrendInput, TtmTrendParams};
    use crate::indicators::volatility_quality_index::{
        volatility_quality_index_with_kernel, VolatilityQualityIndexInput,
        VolatilityQualityIndexParams,
    };
    use crate::indicators::volume_zone_oscillator::{
        volume_zone_oscillator_with_kernel, VolumeZoneOscillatorInput, VolumeZoneOscillatorParams,
    };
    use crate::indicators::vpci::{vpci_with_kernel, VpciInput, VpciParams};
    use crate::indicators::vwap_deviation_oscillator::{
        vwap_deviation_oscillator_with_kernel, VwapDeviationMode, VwapDeviationOscillatorInput,
        VwapDeviationOscillatorParams, VwapDeviationSessionMode,
    };
    use crate::indicators::yang_zhang_volatility::{
        yang_zhang_volatility_with_kernel, YangZhangVolatilityInput, YangZhangVolatilityParams,
    };
    use crate::indicators::zscore::{zscore_with_kernel, ZscoreInput, ZscoreParams};
    use crate::utilities::data_loader::Candles;
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
    fn l2_ehlers_signal_to_noise_output_matches_direct() {
        let candles = sample_candles();
        let combo = [
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("hl2"),
            },
            ParamKV {
                key: "smooth_period",
                value: ParamValue::Int(10),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "l2_ehlers_signal_to_noise",
            output_id: Some("value"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("hl2"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = L2EhlersSignalToNoiseInput::from_slices(
            crate::utilities::data_loader::source_type(&candles, "hl2"),
            candles.high.as_slice(),
            candles.low.as_slice(),
            L2EhlersSignalToNoiseParams {
                smooth_period: Some(10),
            },
        );
        let direct = l2_ehlers_signal_to_noise_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn cycle_channel_oscillator_output_matches_direct() {
        let candles = sample_candles();
        let combo = [
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("close"),
            },
            ParamKV {
                key: "short_cycle_length",
                value: ParamValue::Int(10),
            },
            ParamKV {
                key: "medium_cycle_length",
                value: ParamValue::Int(30),
            },
            ParamKV {
                key: "short_multiplier",
                value: ParamValue::Float(1.0),
            },
            ParamKV {
                key: "medium_multiplier",
                value: ParamValue::Float(3.0),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "cycle_channel_oscillator",
            output_id: Some("fast"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("close"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = CycleChannelOscillatorInput::from_slices(
            crate::utilities::data_loader::source_type(&candles, "close"),
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
            CycleChannelOscillatorParams {
                short_cycle_length: Some(10),
                medium_cycle_length: Some(30),
                short_multiplier: Some(1.0),
                medium_multiplier: Some(3.0),
            },
        );
        let direct = cycle_channel_oscillator_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .fast;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn andean_oscillator_output_matches_direct() {
        let candles = sample_candles();
        let combo = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(50),
            },
            ParamKV {
                key: "signal_length",
                value: ParamValue::Int(9),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "andean_oscillator",
            output_id: Some("bull"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: None,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = AndeanOscillatorInput::from_slices(
            candles.open.as_slice(),
            candles.close.as_slice(),
            AndeanOscillatorParams {
                length: Some(50),
                signal_length: Some(9),
            },
        );
        let direct = andean_oscillator_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .bull;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn daily_factor_output_matches_direct() {
        let (open, high, low, close) = sample_ohlc();
        let combo = [ParamKV {
            key: "threshold_level",
            value: ParamValue::Float(0.35),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "daily_factor",
            output_id: Some("signal"),
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
        let input = DailyFactorInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            DailyFactorParams {
                threshold_level: Some(0.35),
            },
        );
        let direct = daily_factor_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .signal;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn ehlers_adaptive_cyber_cycle_output_matches_direct() {
        let candles = sample_candles();
        let combo = [
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("hl2"),
            },
            ParamKV {
                key: "alpha",
                value: ParamValue::Float(0.07),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "ehlers_adaptive_cyber_cycle",
            output_id: Some("cycle"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("hl2"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = EhlersAdaptiveCyberCycleInput::from_slice(
            crate::utilities::data_loader::source_type(&candles, "hl2"),
            EhlersAdaptiveCyberCycleParams { alpha: Some(0.07) },
        );
        let direct = ehlers_adaptive_cyber_cycle_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .cycle;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn ehlers_simple_cycle_indicator_output_matches_direct() {
        let candles = sample_candles();
        let combo = [
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("hl2"),
            },
            ParamKV {
                key: "alpha",
                value: ParamValue::Float(0.07),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "ehlers_simple_cycle_indicator",
            output_id: Some("cycle"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("hl2"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = EhlersSimpleCycleIndicatorInput::from_slice(
            crate::utilities::data_loader::source_type(&candles, "hl2"),
            EhlersSimpleCycleIndicatorParams { alpha: Some(0.07) },
        );
        let direct = ehlers_simple_cycle_indicator_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .cycle;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn l1_ehlers_phasor_output_matches_direct() {
        let candles = sample_candles();
        let combo = [ParamKV {
            key: "domestic_cycle_length",
            value: ParamValue::Int(15),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "l1_ehlers_phasor",
            output_id: Some("value"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("close"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = L1EhlersPhasorInput::from_slice(
            candles.close.as_slice(),
            L1EhlersPhasorParams {
                domestic_cycle_length: Some(15),
            },
        );
        let direct = l1_ehlers_phasor_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn ehlers_smoothed_adaptive_momentum_output_matches_direct() {
        let candles = sample_candles();
        let combo = [
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("hl2"),
            },
            ParamKV {
                key: "alpha",
                value: ParamValue::Float(0.07),
            },
            ParamKV {
                key: "cutoff",
                value: ParamValue::Float(8.0),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "ehlers_smoothed_adaptive_momentum",
            output_id: Some("value"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("hl2"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = EhlersSmoothedAdaptiveMomentumInput::from_slice(
            crate::utilities::data_loader::source_type(&candles, "hl2"),
            EhlersSmoothedAdaptiveMomentumParams {
                alpha: Some(0.07),
                cutoff: Some(8.0),
            },
        );
        let direct =
            ehlers_smoothed_adaptive_momentum_with_kernel(&input, Kernel::Auto.to_non_batch())
                .unwrap()
                .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn ewma_volatility_output_matches_direct() {
        let close = sample_series();
        let combo = [ParamKV {
            key: "lambda",
            value: ParamValue::Float(0.94),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "ewma_volatility",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &close },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input =
            EwmaVolatilityInput::from_slice(&close, EwmaVolatilityParams { lambda: Some(0.94) });
        let direct = ewma_volatility_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn random_walk_index_output_matches_direct() {
        let open = sample_series();
        let high: Vec<f64> = open.iter().map(|v| v + 1.0).collect();
        let low: Vec<f64> = open.iter().map(|v| v - 1.0).collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.1 * (i as f64 + 1.0))
            .collect();
        let combo = [ParamKV {
            key: "length",
            value: ParamValue::Int(14),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "random_walk_index",
            output_id: Some("high"),
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
        let input = RandomWalkIndexInput::from_slices(
            &high,
            &low,
            &close,
            RandomWalkIndexParams { length: Some(14) },
        );
        let direct = random_walk_index_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .high;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn price_moving_average_ratio_percentile_output_matches_direct() {
        let open = sample_series();
        let high: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v + 1.0 + (i as f64 * 0.03).sin() * 0.15)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v - 1.0 - (i as f64 * 0.05).cos() * 0.12)
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.12 * (i as f64 + 1.0))
            .collect();
        let volume: Vec<f64> = (0..open.len())
            .map(|i| 1_000.0 + i as f64 * 2.0 + (i as f64 * 0.09).sin() * 40.0)
            .collect();
        let combo = [
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("close"),
            },
            ParamKV {
                key: "ma_length",
                value: ParamValue::Int(20),
            },
            ParamKV {
                key: "ma_type",
                value: ParamValue::EnumString("vwma"),
            },
            ParamKV {
                key: "pmarp_lookback",
                value: ParamValue::Int(30),
            },
            ParamKV {
                key: "signal_ma_length",
                value: ParamValue::Int(10),
            },
            ParamKV {
                key: "signal_ma_type",
                value: ParamValue::EnumString("sma"),
            },
            ParamKV {
                key: "line_mode",
                value: ParamValue::EnumString("pmarp"),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "price_moving_average_ratio_percentile",
            output_id: Some("plotline"),
            data: IndicatorDataRef::Candles {
                candles: &crate::utilities::data_loader::Candles {
                    timestamp: vec![0; open.len()],
                    open: open.clone(),
                    high: high.clone(),
                    low: low.clone(),
                    close: close.clone(),
                    volume: volume.clone(),
                    fields: crate::utilities::data_loader::CandleFieldFlags {
                        open: true,
                        high: true,
                        low: true,
                        close: true,
                        volume: true,
                    },
                    hl2: high
                        .iter()
                        .zip(low.iter())
                        .map(|(h, l)| (h + l) * 0.5)
                        .collect(),
                    hlc3: high
                        .iter()
                        .zip(low.iter())
                        .zip(close.iter())
                        .map(|((h, l), c)| (h + l + c) / 3.0)
                        .collect(),
                    ohlc4: open
                        .iter()
                        .zip(high.iter())
                        .zip(low.iter())
                        .zip(close.iter())
                        .map(|(((o, h), l), c)| (o + h + l + c) * 0.25)
                        .collect(),
                    hlcc4: high
                        .iter()
                        .zip(low.iter())
                        .zip(close.iter())
                        .map(|((h, l), c)| (h + l + c + c) * 0.25)
                        .collect(),
                },
                source: Some("close"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = PriceMovingAverageRatioPercentileInput::from_slices(
            &close,
            &volume,
            PriceMovingAverageRatioPercentileParams {
                ma_length: Some(20),
                ma_type: Some(PriceMovingAverageRatioPercentileMaType::Vwma),
                pmarp_lookback: Some(30),
                signal_ma_length: Some(10),
                signal_ma_type: Some(PriceMovingAverageRatioPercentileMaType::Sma),
                line_mode: Some(PriceMovingAverageRatioPercentileLineMode::Pmarp),
            },
        );
        let direct =
            price_moving_average_ratio_percentile_with_kernel(&input, Kernel::Auto.to_non_batch())
                .unwrap()
                .plotline;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn trend_trigger_factor_output_matches_direct() {
        let base = sample_series();
        let high: Vec<f64> = base
            .iter()
            .enumerate()
            .map(|(i, v)| v + 1.0 + (i as f64 * 0.03).sin() * 0.15)
            .collect();
        let low: Vec<f64> = base
            .iter()
            .enumerate()
            .map(|(i, v)| v - 1.0 - (i as f64 * 0.05).cos() * 0.12)
            .collect();
        let combo = [ParamKV {
            key: "length",
            value: ParamValue::Int(15),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "trend_trigger_factor",
            output_id: Some("value"),
            data: IndicatorDataRef::HighLow {
                high: &high,
                low: &low,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = TrendTriggerFactorInput::from_slices(
            &high,
            &low,
            TrendTriggerFactorParams { length: Some(15) },
        );
        let direct = trend_trigger_factor_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn mesa_stochastic_multi_length_output_matches_direct() {
        let source: Vec<f64> = (0..180)
            .map(|i| 100.0 + (i as f64 * 0.09).sin() * 2.0 + i as f64 * 0.015)
            .collect();
        let high: Vec<f64> = source.iter().map(|v| v + 1.0).collect();
        let low: Vec<f64> = source.iter().map(|v| v - 1.0).collect();
        let open = source.clone();
        let volume: Vec<f64> = (0..180).map(|i| 1000.0 + i as f64).collect();
        let combo = [
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("close"),
            },
            ParamKV {
                key: "length_1",
                value: ParamValue::Int(48),
            },
            ParamKV {
                key: "length_2",
                value: ParamValue::Int(21),
            },
            ParamKV {
                key: "length_3",
                value: ParamValue::Int(9),
            },
            ParamKV {
                key: "length_4",
                value: ParamValue::Int(6),
            },
            ParamKV {
                key: "trigger_length",
                value: ParamValue::Int(2),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "mesa_stochastic_multi_length",
            output_id: Some("mesa_1"),
            data: IndicatorDataRef::Candles {
                candles: &crate::utilities::data_loader::Candles {
                    timestamp: vec![0; source.len()],
                    open: open.clone(),
                    high: high.clone(),
                    low: low.clone(),
                    close: source.clone(),
                    volume,
                    fields: crate::utilities::data_loader::CandleFieldFlags {
                        open: true,
                        high: true,
                        low: true,
                        close: true,
                        volume: true,
                    },
                    hl2: high
                        .iter()
                        .zip(low.iter())
                        .map(|(h, l)| (h + l) * 0.5)
                        .collect(),
                    hlc3: high
                        .iter()
                        .zip(low.iter())
                        .zip(source.iter())
                        .map(|((h, l), c)| (h + l + c) / 3.0)
                        .collect(),
                    ohlc4: open
                        .iter()
                        .zip(high.iter())
                        .zip(low.iter())
                        .zip(source.iter())
                        .map(|(((o, h), l), c)| (o + h + l + c) * 0.25)
                        .collect(),
                    hlcc4: high
                        .iter()
                        .zip(low.iter())
                        .zip(source.iter())
                        .map(|((h, l), c)| (h + l + c + c) * 0.25)
                        .collect(),
                },
                source: Some("close"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = MesaStochasticMultiLengthInput::from_slices(
            &source,
            MesaStochasticMultiLengthParams::default(),
        );
        let direct = mesa_stochastic_multi_length_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .mesa_1;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn spearman_correlation_output_matches_direct() {
        let close: Vec<f64> = (0..180)
            .map(|i| 100.0 + (i as f64 * 0.13).sin() * 2.0 + i as f64 * 0.02)
            .collect();
        let open: Vec<f64> = (0..180)
            .map(|i| 98.0 + (i as f64 * 0.07).cos() * 1.6 + i as f64 * 0.015)
            .collect();
        let high: Vec<f64> = close.iter().map(|v| v + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|v| v - 1.0).collect();
        let volume: Vec<f64> = (0..180).map(|i| 1000.0 + i as f64).collect();
        let combo = [
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("close"),
            },
            ParamKV {
                key: "comparison_source",
                value: ParamValue::EnumString("open"),
            },
            ParamKV {
                key: "lookback",
                value: ParamValue::Int(30),
            },
            ParamKV {
                key: "smoothing_length",
                value: ParamValue::Int(3),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "spearman_correlation",
            output_id: Some("smoothed"),
            data: IndicatorDataRef::Candles {
                candles: &crate::utilities::data_loader::Candles {
                    timestamp: vec![0; close.len()],
                    open: open.clone(),
                    high: high.clone(),
                    low: low.clone(),
                    close: close.clone(),
                    volume,
                    fields: crate::utilities::data_loader::CandleFieldFlags {
                        open: true,
                        high: true,
                        low: true,
                        close: true,
                        volume: true,
                    },
                    hl2: high
                        .iter()
                        .zip(low.iter())
                        .map(|(h, l)| (h + l) * 0.5)
                        .collect(),
                    hlc3: high
                        .iter()
                        .zip(low.iter())
                        .zip(close.iter())
                        .map(|((h, l), c)| (h + l + c) / 3.0)
                        .collect(),
                    ohlc4: open
                        .iter()
                        .zip(high.iter())
                        .zip(low.iter())
                        .zip(close.iter())
                        .map(|(((o, h), l), c)| (o + h + l + c) * 0.25)
                        .collect(),
                    hlcc4: high
                        .iter()
                        .zip(low.iter())
                        .zip(close.iter())
                        .map(|((h, l), c)| (h + l + c + c) * 0.25)
                        .collect(),
                },
                source: Some("close"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = SpearmanCorrelationInput::from_slices(
            &close,
            &open,
            SpearmanCorrelationParams {
                lookback: Some(30),
                smoothing_length: Some(3),
            },
        );
        let direct = spearman_correlation_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .smoothed;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn relative_strength_index_wave_indicator_output_matches_direct() {
        let open = sample_series();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.2 * (i as f64 * 0.1).sin())
            .collect();
        let high: Vec<f64> = close.iter().map(|v| v + 0.9).collect();
        let low: Vec<f64> = close.iter().map(|v| v - 0.8).collect();
        let volume: Vec<f64> = (0..close.len()).map(|i| 1_000.0 + i as f64).collect();
        let candles = crate::utilities::data_loader::Candles {
            timestamp: vec![0; close.len()],
            open: open.clone(),
            high: high.clone(),
            low: low.clone(),
            close: close.clone(),
            volume,
            fields: crate::utilities::data_loader::CandleFieldFlags {
                open: true,
                high: true,
                low: true,
                close: true,
                volume: true,
            },
            hl2: high
                .iter()
                .zip(low.iter())
                .map(|(h, l)| (h + l) * 0.5)
                .collect(),
            hlc3: high
                .iter()
                .zip(low.iter())
                .zip(close.iter())
                .map(|((h, l), c)| (h + l + c) / 3.0)
                .collect(),
            ohlc4: open
                .iter()
                .zip(high.iter())
                .zip(low.iter())
                .zip(close.iter())
                .map(|(((o, h), l), c)| (o + h + l + c) * 0.25)
                .collect(),
            hlcc4: high
                .iter()
                .zip(low.iter())
                .zip(close.iter())
                .map(|((h, l), c)| (h + l + 2.0 * c) * 0.25)
                .collect(),
        };
        let combo = [
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("hlcc4"),
            },
            ParamKV {
                key: "rsi_length",
                value: ParamValue::Int(14),
            },
            ParamKV {
                key: "length1",
                value: ParamValue::Int(2),
            },
            ParamKV {
                key: "length2",
                value: ParamValue::Int(5),
            },
            ParamKV {
                key: "length3",
                value: ParamValue::Int(9),
            },
            ParamKV {
                key: "length4",
                value: ParamValue::Int(13),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "relative_strength_index_wave_indicator",
            output_id: Some("rsi_ma1"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("hlcc4"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = RelativeStrengthIndexWaveIndicatorInput::from_slices(
            &candles.hlcc4,
            &high,
            &low,
            RelativeStrengthIndexWaveIndicatorParams {
                rsi_length: Some(14),
                length1: Some(2),
                length2: Some(5),
                length3: Some(9),
                length4: Some(13),
            },
        );
        let direct =
            relative_strength_index_wave_indicator_with_kernel(&input, Kernel::Auto.to_non_batch())
                .unwrap()
                .rsi_ma1;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn accumulation_swing_index_output_matches_direct() {
        let open = sample_series();
        let high: Vec<f64> = open.iter().map(|v| v + 1.0).collect();
        let low: Vec<f64> = open.iter().map(|v| v - 1.0).collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.1 * (i as f64 + 1.0))
            .collect();
        let combo = [ParamKV {
            key: "daily_limit",
            value: ParamValue::Float(10_000.0),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "accumulation_swing_index",
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
        let input = AccumulationSwingIndexInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            AccumulationSwingIndexParams {
                daily_limit: Some(10_000.0),
            },
        );
        let direct = accumulation_swing_index_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn ichimoku_oscillator_output_matches_direct() {
        let open: Vec<f64> = (0..160)
            .map(|i| 100.0 + (i as f64 * 0.07).sin() * 3.0 + i as f64 * 0.02)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v + 1.2 + (i as f64 * 0.03).sin() * 0.25)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v - 1.1 - (i as f64 * 0.05).cos() * 0.2)
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.12 * (i as f64 + 1.0))
            .collect();
        let combo = [
            ParamKV {
                key: "conversion_periods",
                value: ParamValue::Int(9),
            },
            ParamKV {
                key: "base_periods",
                value: ParamValue::Int(26),
            },
            ParamKV {
                key: "lagging_span_periods",
                value: ParamValue::Int(52),
            },
            ParamKV {
                key: "displacement",
                value: ParamValue::Int(26),
            },
            ParamKV {
                key: "ma_length",
                value: ParamValue::Int(12),
            },
            ParamKV {
                key: "smoothing_length",
                value: ParamValue::Int(3),
            },
            ParamKV {
                key: "extra_smoothing",
                value: ParamValue::Bool(true),
            },
            ParamKV {
                key: "normalize",
                value: ParamValue::EnumString("window"),
            },
            ParamKV {
                key: "window_size",
                value: ParamValue::Int(20),
            },
            ParamKV {
                key: "clamp",
                value: ParamValue::Bool(true),
            },
            ParamKV {
                key: "top_band",
                value: ParamValue::Float(2.0),
            },
            ParamKV {
                key: "mid_band",
                value: ParamValue::Float(1.5),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "ichimoku_oscillator",
            output_id: Some("signal"),
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
        let input = IchimokuOscillatorInput::from_slices(
            &high,
            &low,
            &close,
            &close,
            IchimokuOscillatorParams {
                conversion_periods: Some(9),
                base_periods: Some(26),
                lagging_span_periods: Some(52),
                displacement: Some(26),
                ma_length: Some(12),
                smoothing_length: Some(3),
                extra_smoothing: Some(true),
                normalize: Some(IchimokuOscillatorNormalizeMode::Window),
                window_size: Some(20),
                clamp: Some(true),
                top_band: Some(2.0),
                mid_band: Some(1.5),
            },
        );
        let direct = ichimoku_oscillator_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .signal;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn volatility_quality_index_output_matches_direct() {
        let open = sample_series();
        let high: Vec<f64> = open.iter().map(|v| v + 1.0).collect();
        let low: Vec<f64> = open.iter().map(|v| v - 1.0).collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.2 * (i as f64 + 1.0))
            .collect();
        let combo = [
            ParamKV {
                key: "fast_length",
                value: ParamValue::Int(9),
            },
            ParamKV {
                key: "slow_length",
                value: ParamValue::Int(21),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "volatility_quality_index",
            output_id: Some("fast_sma"),
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
        let input = VolatilityQualityIndexInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            VolatilityQualityIndexParams {
                fast_length: Some(9),
                slow_length: Some(21),
            },
        );
        let direct = volatility_quality_index_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .fast_sma;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn vwap_deviation_oscillator_output_matches_direct() {
        let open = sample_series();
        let high: Vec<f64> = open.iter().map(|v| v + 1.0).collect();
        let low: Vec<f64> = open.iter().map(|v| v - 1.0).collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.15 * (i as f64 + 1.0))
            .collect();
        let volume: Vec<f64> = (0..close.len())
            .map(|i| 1000.0 + (i as f64 * 11.0))
            .collect();
        let timestamps: Vec<i64> = (0..close.len())
            .map(|i| 1_700_000_000_000i64 + (i as i64) * 14_400_000)
            .collect();
        let candles = Candles::new(
            timestamps.clone(),
            open.clone(),
            high.clone(),
            low.clone(),
            close.clone(),
            volume.clone(),
        );
        let combo = [
            ParamKV {
                key: "session_mode",
                value: ParamValue::EnumString("rolling_bars"),
            },
            ParamKV {
                key: "rolling_period",
                value: ParamValue::Int(20),
            },
            ParamKV {
                key: "rolling_days",
                value: ParamValue::Int(30),
            },
            ParamKV {
                key: "use_close",
                value: ParamValue::Bool(false),
            },
            ParamKV {
                key: "deviation_mode",
                value: ParamValue::EnumString("zscore"),
            },
            ParamKV {
                key: "z_window",
                value: ParamValue::Int(25),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "vwap_deviation_oscillator",
            output_id: Some("osc"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: None,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = VwapDeviationOscillatorInput::from_slices(
            &timestamps,
            &high,
            &low,
            &close,
            &volume,
            VwapDeviationOscillatorParams {
                session_mode: Some(VwapDeviationSessionMode::RollingBars),
                rolling_period: Some(20),
                rolling_days: Some(30),
                use_close: Some(false),
                deviation_mode: Some(VwapDeviationMode::ZScore),
                z_window: Some(25),
                pct_vol_lookback: Some(100),
                pct_min_sigma: Some(0.1),
                abs_vol_lookback: Some(100),
            },
        );
        let direct = vwap_deviation_oscillator_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .osc;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn volume_zone_oscillator_output_matches_direct() {
        let close = sample_series();
        let volume: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, _)| 1000.0 + (i as f64 * 17.0))
            .collect();
        let combo = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(14),
            },
            ParamKV {
                key: "intraday_smoothing",
                value: ParamValue::Bool(true),
            },
            ParamKV {
                key: "noise_filter",
                value: ParamValue::Int(4),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "volume_zone_oscillator",
            output_id: Some("value"),
            data: IndicatorDataRef::CloseVolume {
                close: &close,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = VolumeZoneOscillatorInput::from_slices(
            &close,
            &volume,
            VolumeZoneOscillatorParams {
                length: Some(14),
                intraday_smoothing: Some(true),
                noise_filter: Some(4),
            },
        );
        let direct = volume_zone_oscillator_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
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
