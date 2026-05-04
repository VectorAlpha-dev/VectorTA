use super::{
    IndicatorBatchOutput, IndicatorBatchRequest, IndicatorDataRef, IndicatorDispatchError,
    IndicatorParamSet, ParamKV, ParamValue,
};
use crate::indicators::absolute_strength_index_oscillator::{
    absolute_strength_index_oscillator_output_into_slice,
    absolute_strength_index_oscillator_with_kernel, AbsoluteStrengthIndexOscillatorInput,
    AbsoluteStrengthIndexOscillatorOutputField, AbsoluteStrengthIndexOscillatorParams,
};
use crate::indicators::accumulation_swing_index::{
    accumulation_swing_index_into_slice, accumulation_swing_index_with_kernel,
    AccumulationSwingIndexInput, AccumulationSwingIndexParams,
};
use crate::indicators::acosc::{
    acosc_output_into_slice, AcoscInput, AcoscOutputField, AcoscParams,
};
use crate::indicators::ad::{ad_with_kernel, AdInput, AdParams};
use crate::indicators::adaptive_bandpass_trigger_oscillator::{
    adaptive_bandpass_trigger_oscillator_with_kernel, AdaptiveBandpassTriggerOscillatorInput,
    AdaptiveBandpassTriggerOscillatorParams,
};
use crate::indicators::adaptive_bounds_rsi::{
    adaptive_bounds_rsi_output_into_slice, AdaptiveBoundsRsiInput, AdaptiveBoundsRsiOutputField,
    AdaptiveBoundsRsiParams,
};
use crate::indicators::adaptive_macd::{
    adaptive_macd_output_into_slice, adaptive_macd_with_kernel, AdaptiveMacdInput,
    AdaptiveMacdOutputField, AdaptiveMacdParams,
};
use crate::indicators::adaptive_momentum_oscillator::{
    adaptive_momentum_oscillator_output_into_slice, adaptive_momentum_oscillator_with_kernel,
    AdaptiveMomentumOscillatorInput, AdaptiveMomentumOscillatorOutputField,
    AdaptiveMomentumOscillatorParams,
};
use crate::indicators::adaptive_schaff_trend_cycle::{
    adaptive_schaff_trend_cycle_output_into_slice, adaptive_schaff_trend_cycle_with_kernel,
    AdaptiveSchaffTrendCycleInput, AdaptiveSchaffTrendCycleOutputField,
    AdaptiveSchaffTrendCycleParams,
};
use crate::indicators::adjustable_ma_alternating_extremities::{
    adjustable_ma_alternating_extremities_output_into_slice,
    adjustable_ma_alternating_extremities_with_kernel, AdjustableMaAlternatingExtremitiesInput,
    AdjustableMaAlternatingExtremitiesOutputField, AdjustableMaAlternatingExtremitiesParams,
};
use crate::indicators::adosc::{adosc_with_kernel, AdoscInput, AdoscParams};
use crate::indicators::advance_decline_line::{
    advance_decline_line_into_slice, advance_decline_line_with_kernel, AdvanceDeclineLineInput,
    AdvanceDeclineLineParams,
};
use crate::indicators::adx::{adx_with_kernel, AdxInput, AdxParams};
use crate::indicators::adxr::{adxr_with_kernel, AdxrInput, AdxrParams};
use crate::indicators::alligator::{
    alligator_output_into_slice, AlligatorInput, AlligatorOutputField, AlligatorParams,
};
use crate::indicators::alphatrend::{
    alphatrend_output_into_slice, AlphaTrendInput, AlphaTrendOutputField, AlphaTrendParams,
};
use crate::indicators::andean_oscillator::{
    andean_oscillator_output_into_slice, andean_oscillator_with_kernel, AndeanOscillatorInput,
    AndeanOscillatorOutputField, AndeanOscillatorParams,
};
use crate::indicators::ao::{ao_into_slice, AoInput, AoParams};
use crate::indicators::apo::{apo_into_slice, ApoInput, ApoParams};
use crate::indicators::aroon::{
    aroon_output_into_slice, AroonInput, AroonOutputField, AroonParams,
};
use crate::indicators::aroonosc::{
    aroon_osc_into_slice, aroon_osc_with_kernel, AroonOscInput, AroonOscParams,
};
use crate::indicators::aso::{aso_output_into_slice, AsoInput, AsoOutputField, AsoParams};
use crate::indicators::atr::{atr_with_kernel, AtrInput, AtrParams};
use crate::indicators::atr_percentile::{
    atr_percentile_into_slice, atr_percentile_with_kernel, AtrPercentileInput, AtrPercentileParams,
};
use crate::indicators::autocorrelation_indicator::{
    autocorrelation_indicator_output_into_slice, autocorrelation_indicator_with_kernel,
    AutocorrelationIndicatorInput, AutocorrelationIndicatorOutputField,
    AutocorrelationIndicatorParams,
};
use crate::indicators::avsl::{avsl_into_slice, avsl_with_kernel, AvslInput, AvslParams};
use crate::indicators::bandpass::{
    bandpass_output_into_slice, BandPassInput, BandPassOutputField, BandPassParams,
};
use crate::indicators::bollinger_bands::{
    bollinger_bands_with_kernel, BollingerBandsInput, BollingerBandsParams,
};
use crate::indicators::bollinger_bands_width::{
    bollinger_bands_width_with_kernel, BollingerBandsWidthInput, BollingerBandsWidthParams,
};
use crate::indicators::bop::{bop_with_kernel, BopInput, BopParams};
use crate::indicators::bull_power_vs_bear_power::{
    bull_power_vs_bear_power_into_slice, bull_power_vs_bear_power_with_kernel,
    BullPowerVsBearPowerInput, BullPowerVsBearPowerParams,
};
use crate::indicators::bulls_v_bears::{
    bulls_v_bears_output_into_slice, bulls_v_bears_with_kernel, BullsVBearsCalculationMethod,
    BullsVBearsInput, BullsVBearsMaType, BullsVBearsOutputField, BullsVBearsParams,
};
use crate::indicators::candle_strength_oscillator::{
    candle_strength_oscillator_output_into_slice, candle_strength_oscillator_with_kernel,
    CandleStrengthOscillatorInput, CandleStrengthOscillatorOutputField,
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
    cyberpunk_value_trend_analyzer_output_into_slice, cyberpunk_value_trend_analyzer_with_kernel,
    CyberpunkValueTrendAnalyzerInput, CyberpunkValueTrendAnalyzerOutputField,
    CyberpunkValueTrendAnalyzerParams,
};
use crate::indicators::cycle_channel_oscillator::{
    cycle_channel_oscillator_output_into_slice, cycle_channel_oscillator_with_kernel,
    CycleChannelOscillatorInput, CycleChannelOscillatorOutputField, CycleChannelOscillatorParams,
};
use crate::indicators::daily_factor::{
    daily_factor_output_into_slice, daily_factor_with_kernel, DailyFactorInput,
    DailyFactorOutputField, DailyFactorParams,
};
use crate::indicators::damiani_volatmeter::{
    damiani_volatmeter_with_kernel, DamianiVolatmeterInput, DamianiVolatmeterParams,
};
use crate::indicators::decisionpoint_breadth_swenlin_trading_oscillator::{
    decisionpoint_breadth_swenlin_trading_oscillator_into_slice,
    decisionpoint_breadth_swenlin_trading_oscillator_with_kernel,
    DecisionPointBreadthSwenlinTradingOscillatorInput,
    DecisionPointBreadthSwenlinTradingOscillatorParams,
};
use crate::indicators::demand_index::{
    demand_index_with_kernel, DemandIndexInput, DemandIndexParams,
};
use crate::indicators::deviation::{deviation_with_kernel, DeviationInput, DeviationParams};
use crate::indicators::devstop::{devstop_with_kernel, DevStopInput, DevStopParams};
use crate::indicators::di::{di_minus_with_kernel, di_plus_with_kernel, DiInput, DiParams};
use crate::indicators::didi_index::{
    didi_index_output_into_slice, didi_index_with_kernel, DidiIndexInput, DidiIndexOutputField,
    DidiIndexParams,
};
use crate::indicators::directional_imbalance_index::{
    directional_imbalance_index_output_into_slice, directional_imbalance_index_with_kernel,
    DirectionalImbalanceIndexInput, DirectionalImbalanceIndexOutputField,
    DirectionalImbalanceIndexParams,
};
use crate::indicators::disparity_index::{
    disparity_index_into_slice, DisparityIndexInput, DisparityIndexParams,
};
use crate::indicators::dm::{dm_minus_with_kernel, dm_plus_with_kernel, DmInput, DmParams};
use crate::indicators::donchian::{
    donchian_lower_with_kernel, donchian_middle_with_kernel, donchian_upper_with_kernel,
    DonchianInput, DonchianParams,
};
use crate::indicators::donchian_channel_width::{
    donchian_channel_width_into_slice, DonchianChannelWidthInput, DonchianChannelWidthParams,
};
use crate::indicators::dpo::{dpo_into_slice, DpoInput, DpoParams};
use crate::indicators::dti::{dti_into_slice, DtiInput, DtiParams};
use crate::indicators::dual_ulcer_index::{
    dual_ulcer_index_output_into_slice, dual_ulcer_index_with_kernel, DualUlcerIndexInput,
    DualUlcerIndexOutputField, DualUlcerIndexParams,
};
use crate::indicators::dvdiqqe::{
    dvdiqqe_output_into_slice, DvdiqqeInput, DvdiqqeOutputField, DvdiqqeParams,
};
use crate::indicators::dx::{dx_batch_with_kernel, dx_into_slice, DxBatchRange, DxInput, DxParams};
use crate::indicators::dynamic_momentum_index::{
    dynamic_momentum_index_into_slice, dynamic_momentum_index_with_kernel,
    DynamicMomentumIndexInput, DynamicMomentumIndexParams,
};
use crate::indicators::efi::{efi_into_slice, EfiInput, EfiParams};
use crate::indicators::ehlers_adaptive_cg::{
    ehlers_adaptive_cg_with_kernel, EhlersAdaptiveCgInput, EhlersAdaptiveCgParams,
};
use crate::indicators::ehlers_adaptive_cyber_cycle::{
    ehlers_adaptive_cyber_cycle_with_kernel, EhlersAdaptiveCyberCycleInput,
    EhlersAdaptiveCyberCycleParams,
};
use crate::indicators::ehlers_autocorrelation_periodogram::{
    ehlers_autocorrelation_periodogram_with_kernel, EhlersAutocorrelationPeriodogramInput,
    EhlersAutocorrelationPeriodogramParams,
};
use crate::indicators::ehlers_data_sampling_relative_strength_indicator::{
    ehlers_data_sampling_relative_strength_indicator_with_kernel,
    EhlersDataSamplingRelativeStrengthIndicatorInput,
    EhlersDataSamplingRelativeStrengthIndicatorParams,
};
use crate::indicators::ehlers_detrending_filter::{
    ehlers_detrending_filter_with_kernel, EhlersDetrendingFilterInput, EhlersDetrendingFilterParams,
};
use crate::indicators::ehlers_fm_demodulator::{
    ehlers_fm_demodulator_with_kernel, EhlersFmDemodulatorInput, EhlersFmDemodulatorParams,
};
use crate::indicators::ehlers_linear_extrapolation_predictor::{
    ehlers_linear_extrapolation_predictor_with_kernel, EhlersLinearExtrapolationPredictorInput,
    EhlersLinearExtrapolationPredictorParams,
};
use crate::indicators::ehlers_simple_cycle_indicator::{
    ehlers_simple_cycle_indicator_with_kernel, EhlersSimpleCycleIndicatorInput,
    EhlersSimpleCycleIndicatorParams,
};
use crate::indicators::ehlers_smoothed_adaptive_momentum::{
    ehlers_smoothed_adaptive_momentum_with_kernel, EhlersSmoothedAdaptiveMomentumInput,
    EhlersSmoothedAdaptiveMomentumParams,
};
use crate::indicators::emd::{emd_with_kernel, EmdInput, EmdParams};
use crate::indicators::emd_trend::{emd_trend_with_kernel, EmdTrendInput, EmdTrendParams};
use crate::indicators::emv::{emv_with_kernel, EmvInput};
use crate::indicators::er::{er_with_kernel, ErInput, ErParams};
use crate::indicators::eri::{eri_with_kernel, EriInput, EriParams};
use crate::indicators::evasive_supertrend::{
    evasive_supertrend_with_kernel, EvasiveSuperTrendInput, EvasiveSuperTrendParams,
};
use crate::indicators::ewma_volatility::{
    ewma_volatility_with_kernel, EwmaVolatilityInput, EwmaVolatilityParams,
};
use crate::indicators::exponential_trend::{
    exponential_trend_with_kernel, ExponentialTrendInput, ExponentialTrendParams,
};
use crate::indicators::fibonacci_entry_bands::{
    fibonacci_entry_bands_with_kernel, FibonacciEntryBandsInput, FibonacciEntryBandsParams,
};
use crate::indicators::fibonacci_trailing_stop::{
    fibonacci_trailing_stop_with_kernel, FibonacciTrailingStopInput, FibonacciTrailingStopParams,
};
use crate::indicators::fisher::{fisher_with_kernel, FisherInput, FisherParams};
use crate::indicators::forward_backward_exponential_oscillator::{
    forward_backward_exponential_oscillator_with_kernel, ForwardBackwardExponentialOscillatorInput,
    ForwardBackwardExponentialOscillatorParams,
};
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
use crate::indicators::garman_klass_volatility::{
    garman_klass_volatility_with_kernel, GarmanKlassVolatilityInput, GarmanKlassVolatilityParams,
};
use crate::indicators::gatorosc::{gatorosc_with_kernel, GatorOscInput, GatorOscParams};
use crate::indicators::geometric_bias_oscillator::{
    geometric_bias_oscillator_with_kernel, GeometricBiasOscillatorInput,
    GeometricBiasOscillatorParams,
};
use crate::indicators::gmma_oscillator::{
    gmma_oscillator_with_kernel, GmmaOscillatorInput, GmmaOscillatorParams,
};
use crate::indicators::goertzel_cycle_composite_wave::{
    goertzel_cycle_composite_wave_into_slice, GoertzelCycleCompositeWaveInput,
    GoertzelCycleCompositeWaveParams, GoertzelDetrendMode,
};
use crate::indicators::gopalakrishnan_range_index::{
    gopalakrishnan_range_index_with_kernel, GopalakrishnanRangeIndexInput,
    GopalakrishnanRangeIndexParams,
};
use crate::indicators::grover_llorens_cycle_oscillator::{
    grover_llorens_cycle_oscillator_with_kernel, GroverLlorensCycleOscillatorInput,
    GroverLlorensCycleOscillatorParams,
};
use crate::indicators::half_causal_estimator::{
    half_causal_estimator_with_kernel, HalfCausalEstimatorConfidenceAdjust,
    HalfCausalEstimatorInput, HalfCausalEstimatorKernelType, HalfCausalEstimatorParams,
};
use crate::indicators::halftrend::{halftrend_with_kernel, HalfTrendInput, HalfTrendParams};
use crate::indicators::hema_trend_levels::{
    hema_trend_levels_output_into_slice, HemaTrendLevelsInput, HemaTrendLevelsOutputField,
    HemaTrendLevelsParams,
};
use crate::indicators::historical_volatility::{
    historical_volatility_into_slice, historical_volatility_with_kernel, HistoricalVolatilityInput,
    HistoricalVolatilityParams,
};
use crate::indicators::historical_volatility_percentile::{
    historical_volatility_percentile_with_kernel, HistoricalVolatilityPercentileInput,
    HistoricalVolatilityPercentileParams,
};
use crate::indicators::historical_volatility_rank::{
    historical_volatility_rank_output_into_slice, historical_volatility_rank_with_kernel,
    HistoricalVolatilityRankInput, HistoricalVolatilityRankOutputField,
    HistoricalVolatilityRankParams,
};
use crate::indicators::hull_butterfly_oscillator::{
    hull_butterfly_oscillator_output_into_slice, HullButterflyOscillatorInput,
    HullButterflyOscillatorOutputField, HullButterflyOscillatorParams,
};
use crate::indicators::hypertrend::{
    hypertrend_output_into_slice, HyperTrendInput, HyperTrendOutputField, HyperTrendParams,
};
use crate::indicators::ichimoku_oscillator::{
    ichimoku_oscillator_with_kernel, IchimokuOscillatorInput, IchimokuOscillatorNormalizeMode,
    IchimokuOscillatorParams,
};
use crate::indicators::ict_propulsion_block::{
    ict_propulsion_block_into_slice, IctPropulsionBlockInput, IctPropulsionBlockMitigationPrice,
    IctPropulsionBlockParams,
};
use crate::indicators::ift_rsi::{ift_rsi_with_kernel, IftRsiInput, IftRsiParams};
use crate::indicators::impulse_macd::{
    impulse_macd_with_kernel, ImpulseMacdInput, ImpulseMacdParams,
};
use crate::indicators::intraday_momentum_index::{
    intraday_momentum_index_with_kernel, IntradayMomentumIndexInput, IntradayMomentumIndexParams,
};
use crate::indicators::kairi_relative_index::{
    kairi_relative_index_into_slice, KairiRelativeIndexInput, KairiRelativeIndexParams,
};
use crate::indicators::kase_peak_oscillator_with_divergences::{
    kase_peak_oscillator_with_divergences_with_kernel, KasePeakOscillatorWithDivergencesInput,
    KasePeakOscillatorWithDivergencesParams,
};
use crate::indicators::kaufmanstop::{
    kaufmanstop_with_kernel, KaufmanstopInput, KaufmanstopParams,
};
use crate::indicators::kdj::{kdj_with_kernel, KdjInput, KdjParams};
use crate::indicators::keltner::{keltner_with_kernel, KeltnerInput, KeltnerParams};
use crate::indicators::keltner_channel_width_oscillator::{
    keltner_channel_width_oscillator_with_kernel, KeltnerChannelWidthOscillatorInput,
    KeltnerChannelWidthOscillatorParams,
};
use crate::indicators::kst::{kst_with_kernel, KstInput, KstParams};
use crate::indicators::kurtosis::{kurtosis_with_kernel, KurtosisInput, KurtosisParams};
use crate::indicators::kvo::{kvo_with_kernel, KvoInput, KvoParams};
use crate::indicators::l1_ehlers_phasor::{
    l1_ehlers_phasor_with_kernel, L1EhlersPhasorInput, L1EhlersPhasorParams,
};
use crate::indicators::l2_ehlers_signal_to_noise::{
    l2_ehlers_signal_to_noise_with_kernel, L2EhlersSignalToNoiseInput, L2EhlersSignalToNoiseParams,
};
use crate::indicators::leavitt_convolution_acceleration::{
    leavitt_convolution_acceleration_with_kernel, LeavittConvolutionAccelerationInput,
    LeavittConvolutionAccelerationParams,
};
use crate::indicators::linear_correlation_oscillator::{
    linear_correlation_oscillator_with_kernel, LinearCorrelationOscillatorInput,
    LinearCorrelationOscillatorParams,
};
use crate::indicators::linear_regression_intensity::{
    linear_regression_intensity_with_kernel, LinearRegressionIntensityInput,
    LinearRegressionIntensityParams,
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
use crate::indicators::macd_wave_signal_pro::{
    macd_wave_signal_pro_with_kernel, MacdWaveSignalProInput,
};
use crate::indicators::macz::{macz_with_kernel, MaczInput, MaczParams};
use crate::indicators::market_meanness_index::{
    market_meanness_index_with_kernel, MarketMeannessIndexInput, MarketMeannessIndexParams,
};
use crate::indicators::market_structure_confluence::{
    market_structure_confluence_with_kernel, MarketStructureConfluenceInput,
    MarketStructureConfluenceParams,
};
use crate::indicators::market_structure_trailing_stop::{
    market_structure_trailing_stop_with_kernel, MarketStructureTrailingStopInput,
    MarketStructureTrailingStopParams,
};
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
use crate::indicators::momentum_ratio_oscillator::{
    momentum_ratio_oscillator_with_kernel, MomentumRatioOscillatorInput,
    MomentumRatioOscillatorParams,
};
use crate::indicators::monotonicity_index::{
    monotonicity_index_with_kernel, MonotonicityIndexInput, MonotonicityIndexMode,
    MonotonicityIndexParams,
};
use crate::indicators::moving_average_cross_probability::{
    moving_average_cross_probability_with_kernel, MovingAverageCrossProbabilityInput,
    MovingAverageCrossProbabilityMaType, MovingAverageCrossProbabilityParams,
};
use crate::indicators::moving_averages::edcf::{edcf_into_slice, EdcfInput, EdcfParams};
use crate::indicators::moving_averages::logarithmic_moving_average::{
    logarithmic_moving_average_with_kernel, LogarithmicMovingAverageInput,
    LogarithmicMovingAverageParams,
};
use crate::indicators::moving_averages::ma::MaData;
use crate::indicators::moving_averages::ma_batch::{
    ma_batch_with_kernel_and_typed_params, MaBatchParamKV, MaBatchParamValue,
};
use crate::indicators::moving_averages::registry::list_moving_averages;
use crate::indicators::moving_averages::wilders::{
    wilders_into_slice, WildersInput, WildersParams,
};
use crate::indicators::moving_averages::zlema::{zlema_into_slice, ZlemaInput, ZlemaParams};
use crate::indicators::msw::{msw_with_kernel, MswInput, MswParams};
use crate::indicators::multi_length_stochastic_average::{
    multi_length_stochastic_average_with_kernel, MultiLengthStochasticAverageInput,
    MultiLengthStochasticAverageParams,
};
use crate::indicators::nadaraya_watson_envelope::{
    nadaraya_watson_envelope_with_kernel, NweInput, NweParams,
};
use crate::indicators::natr::{natr_with_kernel, NatrInput, NatrParams};
use crate::indicators::neighboring_trailing_stop::{
    neighboring_trailing_stop_with_kernel, NeighboringTrailingStopInput,
    NeighboringTrailingStopParams,
};
use crate::indicators::net_myrsi::{net_myrsi_with_kernel, NetMyrsiInput, NetMyrsiParams};
use crate::indicators::nonlinear_regression_zero_lag_moving_average::{
    nonlinear_regression_zero_lag_moving_average_with_kernel,
    NonlinearRegressionZeroLagMovingAverageInput, NonlinearRegressionZeroLagMovingAverageParams,
};
use crate::indicators::normalized_resonator::{
    normalized_resonator_with_kernel, NormalizedResonatorInput, NormalizedResonatorParams,
};
use crate::indicators::normalized_volume_true_range::{
    normalized_volume_true_range_with_kernel, NormalizedVolumeTrueRangeInput,
    NormalizedVolumeTrueRangeParams, NormalizedVolumeTrueRangeStyle,
};
use crate::indicators::nvi::{nvi_with_kernel, NviInput, NviParams};
use crate::indicators::obv::{obv_with_kernel, ObvInput, ObvParams};
use crate::indicators::on_balance_volume_oscillator::{
    on_balance_volume_oscillator_with_kernel, OnBalanceVolumeOscillatorInput,
    OnBalanceVolumeOscillatorParams,
};
use crate::indicators::otto::{otto_with_kernel, OttoInput, OttoParams};
use crate::indicators::parkinson_volatility::{
    parkinson_volatility_with_kernel, ParkinsonVolatilityInput, ParkinsonVolatilityParams,
};
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
use crate::indicators::possible_rsi::{
    possible_rsi_with_kernel, PossibleRsiInput, PossibleRsiParams,
};
use crate::indicators::ppo::{ppo_with_kernel, PpoInput, PpoParams};
use crate::indicators::prb::{prb_with_kernel, PrbInput, PrbParams};
use crate::indicators::premier_rsi_oscillator::{
    premier_rsi_oscillator_with_kernel, PremierRsiOscillatorInput, PremierRsiOscillatorParams,
};
use crate::indicators::pretty_good_oscillator::{
    pretty_good_oscillator_with_kernel, PrettyGoodOscillatorInput, PrettyGoodOscillatorParams,
};
use crate::indicators::price_density_market_noise::{
    price_density_market_noise_with_kernel, PriceDensityMarketNoiseInput,
    PriceDensityMarketNoiseParams,
};
use crate::indicators::price_moving_average_ratio_percentile::{
    price_moving_average_ratio_percentile_with_kernel, PriceMovingAverageRatioPercentileInput,
    PriceMovingAverageRatioPercentileLineMode, PriceMovingAverageRatioPercentileMaType,
    PriceMovingAverageRatioPercentileParams,
};
use crate::indicators::projection_oscillator::{
    projection_oscillator_with_kernel, ProjectionOscillatorInput, ProjectionOscillatorParams,
};
use crate::indicators::psychological_line::{
    psychological_line_with_kernel, PsychologicalLineInput, PsychologicalLineParams,
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
use crate::indicators::range_filtered_trend_signals::{
    range_filtered_trend_signals_with_kernel, RangeFilteredTrendSignalsInput,
    RangeFilteredTrendSignalsParams,
};
use crate::indicators::range_oscillator::{
    range_oscillator_with_kernel, RangeOscillatorInput, RangeOscillatorParams,
};
use crate::indicators::rank_correlation_index::{
    rank_correlation_index_with_kernel, RankCorrelationIndexInput, RankCorrelationIndexParams,
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
use crate::indicators::reversal_signals::{
    reversal_signals_with_kernel, ReversalSignalsInput, ReversalSignalsParams,
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
use crate::indicators::smooth_theil_sen::{
    smooth_theil_sen_with_kernel, SmoothTheilSenDeviationType, SmoothTheilSenInput,
    SmoothTheilSenParams, SmoothTheilSenStatStyle,
};
use crate::indicators::smoothed_gaussian_trend_filter::{
    smoothed_gaussian_trend_filter_filter_with_kernel, smoothed_gaussian_trend_filter_with_kernel,
    SmoothedGaussianTrendFilterInput, SmoothedGaussianTrendFilterParams,
};
use crate::indicators::spearman_correlation::{
    spearman_correlation_with_kernel, SpearmanCorrelationInput, SpearmanCorrelationParams,
};
use crate::indicators::squeeze_index::{
    squeeze_index_with_kernel, SqueezeIndexInput, SqueezeIndexParams,
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
use crate::indicators::stochastic_adaptive_d::{
    stochastic_adaptive_d_with_kernel, StochasticAdaptiveDInput, StochasticAdaptiveDParams,
};
use crate::indicators::stochastic_connors_rsi::{
    stochastic_connors_rsi_with_kernel, StochasticConnorsRsiInput, StochasticConnorsRsiParams,
};
use crate::indicators::stochastic_distance::{
    stochastic_distance_with_kernel, StochasticDistanceInput, StochasticDistanceParams,
};
use crate::indicators::stochastic_money_flow_index::{
    stochastic_money_flow_index_with_kernel, StochasticMoneyFlowIndexInput,
    StochasticMoneyFlowIndexParams,
};
use crate::indicators::stochf::{stochf_with_kernel, StochfInput, StochfParams};
use crate::indicators::supertrend::{supertrend_with_kernel, SuperTrendInput, SuperTrendParams};
use crate::indicators::supertrend_oscillator::{
    supertrend_oscillator_with_kernel, SuperTrendOscillatorInput, SuperTrendOscillatorParams,
};
use crate::indicators::supertrend_recovery::{
    supertrend_recovery_with_kernel, SuperTrendRecoveryInput, SuperTrendRecoveryParams,
};
use crate::indicators::trend_continuation_factor::{
    trend_continuation_factor_with_kernel, TrendContinuationFactorInput,
    TrendContinuationFactorParams,
};
use crate::indicators::trend_direction_force_index::{
    trend_direction_force_index_into_slice, TrendDirectionForceIndexInput,
    TrendDirectionForceIndexParams,
};
use crate::indicators::trend_flow_trail::{
    trend_flow_trail_with_kernel, TrendFlowTrailInput, TrendFlowTrailParams,
};
use crate::indicators::trend_trigger_factor::{
    trend_trigger_factor_with_kernel, TrendTriggerFactorInput, TrendTriggerFactorParams,
};
use crate::indicators::trix::{
    trix_batch_with_kernel, trix_into_slice, trix_with_kernel, TrixBatchRange, TrixInput,
    TrixParams,
};
use crate::indicators::tsf::{tsf_with_kernel, TsfInput, TsfParams};
use crate::indicators::tsi::{tsi_with_kernel, TsiInput, TsiParams};
use crate::indicators::ttm_squeeze::{ttm_squeeze_with_kernel, TtmSqueezeInput, TtmSqueezeParams};
use crate::indicators::ttm_trend::{ttm_trend_with_kernel, TtmTrendInput, TtmTrendParams};
use crate::indicators::twiggs_money_flow::{
    twiggs_money_flow_with_kernel, TwiggsMoneyFlowInput, TwiggsMoneyFlowParams,
};
use crate::indicators::ui::{ui_with_kernel, UiInput, UiParams};
use crate::indicators::ultosc::{ultosc_with_kernel, UltOscInput, UltOscParams};
use crate::indicators::var::{var_with_kernel, VarInput, VarParams};
use crate::indicators::vdubus_divergence_wave_pattern_generator::{
    vdubus_divergence_wave_pattern_generator_with_kernel,
    VdubusDivergenceWavePatternGeneratorInput, VdubusDivergenceWavePatternGeneratorParams,
};
use crate::indicators::velocity::{velocity_with_kernel, VelocityInput, VelocityParams};
use crate::indicators::velocity_acceleration_convergence_divergence_indicator::{
    velocity_acceleration_convergence_divergence_indicator_with_kernel,
    VelocityAccelerationConvergenceDivergenceIndicatorInput,
    VelocityAccelerationConvergenceDivergenceIndicatorParams,
};
use crate::indicators::velocity_acceleration_indicator::{
    velocity_acceleration_indicator_with_kernel, VelocityAccelerationIndicatorInput,
    VelocityAccelerationIndicatorParams,
};
use crate::indicators::vertical_horizontal_filter::{
    vertical_horizontal_filter_with_kernel, VerticalHorizontalFilterInput,
    VerticalHorizontalFilterParams,
};
use crate::indicators::vi::{vi_with_kernel, ViInput, ViParams};
use crate::indicators::vidya::{vidya_with_kernel, VidyaInput, VidyaParams};
use crate::indicators::vlma::{vlma_with_kernel, VlmaInput, VlmaParams};
use crate::indicators::volatility_quality_index::{
    volatility_quality_index_with_kernel, VolatilityQualityIndexInput, VolatilityQualityIndexParams,
};
use crate::indicators::volatility_ratio_adaptive_rsx::{
    volatility_ratio_adaptive_rsx_with_kernel, VolatilityRatioAdaptiveRsxInput,
    VolatilityRatioAdaptiveRsxParams,
};
use crate::indicators::volume_energy_reservoirs::{
    volume_energy_reservoirs_output_into_slice, VolumeEnergyReservoirsInput,
    VolumeEnergyReservoirsOutputField, VolumeEnergyReservoirsParams,
};
use crate::indicators::volume_weighted_relative_strength_index::{
    volume_weighted_relative_strength_index_output_into_slice,
    VolumeWeightedRelativeStrengthIndexInput, VolumeWeightedRelativeStrengthIndexOutputField,
    VolumeWeightedRelativeStrengthIndexParams,
};
use crate::indicators::volume_weighted_rsi::{
    volume_weighted_rsi_batch_with_kernel, volume_weighted_rsi_into_slice,
    VolumeWeightedRsiBatchRange, VolumeWeightedRsiInput, VolumeWeightedRsiParams,
};
use crate::indicators::volume_weighted_stochastic_rsi::{
    volume_weighted_stochastic_rsi_output_into_slice, VolumeWeightedStochasticRsiInput,
    VolumeWeightedStochasticRsiOutputField, VolumeWeightedStochasticRsiParams,
};
use crate::indicators::volume_zone_oscillator::{
    volume_zone_oscillator_into_slice, VolumeZoneOscillatorInput, VolumeZoneOscillatorParams,
};
use crate::indicators::vosc::{vosc_into_slice, VoscInput, VoscParams};
use crate::indicators::voss::{voss_output_into_slice, VossInput, VossOutputField, VossParams};
use crate::indicators::vpci::{vpci_output_into_slice, VpciInput, VpciOutputField, VpciParams};
use crate::indicators::vpt::vpt_into_slice;
use crate::indicators::vwap_deviation_oscillator::{
    vwap_deviation_oscillator_output_into_slice, VwapDeviationMode, VwapDeviationOscillatorInput,
    VwapDeviationOscillatorOutputField, VwapDeviationOscillatorParams, VwapDeviationSessionMode,
};
use crate::indicators::vwap_zscore_with_signals::{
    vwap_zscore_with_signals_output_into_slice, VwapZscoreWithSignalsInput,
    VwapZscoreWithSignalsOutputField, VwapZscoreWithSignalsParams,
};
use crate::indicators::vwmacd::{
    vwmacd_output_into_slice, VwmacdInput, VwmacdOutputField, VwmacdParams,
};
use crate::indicators::wad::{wad_into_slice, WadInput};
use crate::indicators::wavetrend::{
    wavetrend_output_into_slice, WavetrendInput, WavetrendOutputField, WavetrendParams,
};
use crate::indicators::wclprice::{wclprice_into_slice, WclpriceInput};
use crate::indicators::willr::{willr_into_slice, WillrInput, WillrParams};
use crate::indicators::wto::{wto_output_into_slice, WtoInput, WtoOutputField, WtoParams};
use crate::indicators::yang_zhang_volatility::{
    yang_zhang_volatility_with_kernel, YangZhangVolatilityInput, YangZhangVolatilityParams,
};
use crate::indicators::zig_zag_channels::{
    zig_zag_channels_into_slice, ZigZagChannelsInput, ZigZagChannelsParams,
};
use crate::indicators::zscore::{zscore_into_slice, ZscoreInput, ZscoreParams};
use crate::indicators::{cg::cg_with_kernel, cg::CgInput, cg::CgParams};
use crate::utilities::data_loader::source_type;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::alloc_uninit_f64;
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

    let info = get_indicator(req.indicator_id);

    if let Some(info) = info {
        if strict_inputs {
            validate_input_kind_strict(info.id, info.input_kind, req.data)?;
        }

        let output_id = resolve_output_id(info, req.output_id)?;

        if info.id.eq_ignore_ascii_case("logarithmic_moving_average") {
            return compute_logarithmic_moving_average_batch(req, output_id);
        }

        if is_moving_average(info.id) {
            return compute_ma_batch(req, info, output_id);
        }

        return dispatch_cpu_batch_by_indicator(req, info.id, output_id);
    }

    let output_id = req.output_id.unwrap_or("value");
    match dispatch_cpu_batch_by_indicator(req, req.indicator_id, output_id) {
        Err(IndicatorDispatchError::UnsupportedCapability { .. }) => {
            Err(IndicatorDispatchError::UnknownIndicator {
                id: req.indicator_id.to_string(),
            })
        }
        other => other,
    }
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
            "adaptive_bounds_rsi" => Some(compute_adaptive_bounds_rsi_batch(
                req,
                output_id.unwrap_or("rsi"),
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
            "garman_klass_volatility" => Some(compute_garman_klass_volatility_batch(
                req,
                output_id.unwrap_or("value"),
            )),
            "advance_decline_line" => Some(compute_advance_decline_line_batch(
                req,
                output_id.unwrap_or("value"),
            )),
            "decisionpoint_breadth_swenlin_trading_oscillator" => Some(
                compute_decisionpoint_breadth_swenlin_trading_oscillator_batch(
                    req,
                    output_id.unwrap_or("value"),
                ),
            ),
            "velocity_acceleration_indicator" => Some(
                compute_velocity_acceleration_indicator_batch(req, output_id.unwrap_or("value")),
            ),
            "normalized_resonator" => Some(compute_normalized_resonator_batch(
                req,
                output_id.unwrap_or("oscillator"),
            )),
            "monotonicity_index" => Some(compute_monotonicity_index_batch(
                req,
                output_id.unwrap_or("index"),
            )),
            "half_causal_estimator" => Some(compute_half_causal_estimator_batch(
                req,
                output_id.unwrap_or("estimate"),
            )),
            "atr_percentile" => Some(compute_atr_percentile_batch(
                req,
                output_id.unwrap_or("value"),
            )),
            "bull_power_vs_bear_power" => Some(compute_bull_power_vs_bear_power_batch(
                req,
                output_id.unwrap_or("value"),
            )),
            "didi_index" => Some(compute_didi_index_batch(req, output_id.unwrap_or("short"))),
            "ehlers_autocorrelation_periodogram" => {
                Some(compute_ehlers_autocorrelation_periodogram_batch(
                    req,
                    output_id.unwrap_or("dominant_cycle"),
                ))
            }
            "ehlers_linear_extrapolation_predictor" => {
                Some(compute_ehlers_linear_extrapolation_predictor_batch(
                    req,
                    output_id.unwrap_or("prediction"),
                ))
            }
            "kase_peak_oscillator_with_divergences" => {
                Some(compute_kase_peak_oscillator_with_divergences_batch(
                    req,
                    output_id.unwrap_or("oscillator"),
                ))
            }
            "absolute_strength_index_oscillator" => {
                Some(compute_absolute_strength_index_oscillator_batch(
                    req,
                    output_id.unwrap_or("oscillator"),
                ))
            }
            "adaptive_bandpass_trigger_oscillator" => {
                Some(compute_adaptive_bandpass_trigger_oscillator_batch(
                    req,
                    output_id.unwrap_or("in_phase"),
                ))
            }
            "premier_rsi_oscillator" => Some(compute_premier_rsi_oscillator_batch(
                req,
                output_id.unwrap_or("value"),
            )),
            "multi_length_stochastic_average" => Some(
                compute_multi_length_stochastic_average_batch(req, output_id.unwrap_or("value")),
            ),
            "hull_butterfly_oscillator" => Some(compute_hull_butterfly_oscillator_batch(
                req,
                output_id.unwrap_or("oscillator"),
            )),
            "fibonacci_trailing_stop" => Some(compute_fibonacci_trailing_stop_batch(
                req,
                output_id.unwrap_or("trailing_stop"),
            )),
            "fibonacci_entry_bands" => Some(compute_fibonacci_entry_bands_batch(
                req,
                output_id.unwrap_or("middle"),
            )),
            "volume_energy_reservoirs" => Some(compute_volume_energy_reservoirs_batch(
                req,
                output_id.unwrap_or("momentum"),
            )),
            "neighboring_trailing_stop" => Some(compute_neighboring_trailing_stop_batch(
                req,
                output_id.unwrap_or("trailing_stop"),
            )),
            "grover_llorens_cycle_oscillator" => Some(
                compute_grover_llorens_cycle_oscillator_batch(req, output_id.unwrap_or("value")),
            ),
            "historical_volatility" => Some(compute_historical_volatility_batch(
                req,
                output_id.unwrap_or("value"),
            )),
            "squeeze_index" => Some(compute_squeeze_index_batch(
                req,
                output_id.unwrap_or("value"),
            )),
            "stochastic_distance" => Some(compute_stochastic_distance_batch(
                req,
                output_id.unwrap_or("oscillator"),
            )),
            "vertical_horizontal_filter" => Some(compute_vertical_horizontal_filter_batch(
                req,
                output_id.unwrap_or("value"),
            )),
            "intraday_momentum_index" => {
                if let Some(out) = output_id {
                    Some(compute_intraday_momentum_index_batch(req, out))
                } else {
                    None
                }
            }
            "vwap_zscore_with_signals" => {
                if let Some(out) = output_id {
                    Some(compute_vwap_zscore_with_signals_batch(req, out))
                } else {
                    None
                }
            }
            "macd_wave_signal_pro" => {
                if let Some(out) = output_id {
                    Some(compute_macd_wave_signal_pro_batch(req, out))
                } else {
                    None
                }
            }
            "hema_trend_levels" => {
                if let Some(out) = output_id {
                    Some(compute_hema_trend_levels_batch(req, out))
                } else {
                    None
                }
            }
            "demand_index" => {
                if let Some(out) = output_id {
                    Some(compute_demand_index_batch(req, out))
                } else {
                    None
                }
            }
            "gopalakrishnan_range_index" => Some(compute_gopalakrishnan_range_index_batch(
                req,
                output_id.unwrap_or("value"),
            )),
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
    if id.eq_ignore_ascii_case("adaptive_bounds_rsi") {
        return Some(compute_adaptive_bounds_rsi_batch(
            req,
            output_id.unwrap_or("rsi"),
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
    if id.eq_ignore_ascii_case("garman_klass_volatility") {
        return Some(compute_garman_klass_volatility_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("advance_decline_line") {
        return Some(compute_advance_decline_line_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("decisionpoint_breadth_swenlin_trading_oscillator") {
        return Some(
            compute_decisionpoint_breadth_swenlin_trading_oscillator_batch(
                req,
                output_id.unwrap_or("value"),
            ),
        );
    }
    if id.eq_ignore_ascii_case("velocity_acceleration_indicator") {
        return Some(compute_velocity_acceleration_indicator_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("normalized_resonator") {
        return Some(compute_normalized_resonator_batch(
            req,
            output_id.unwrap_or("oscillator"),
        ));
    }
    if id.eq_ignore_ascii_case("monotonicity_index") {
        return Some(compute_monotonicity_index_batch(
            req,
            output_id.unwrap_or("index"),
        ));
    }
    if id.eq_ignore_ascii_case("half_causal_estimator") {
        return Some(compute_half_causal_estimator_batch(
            req,
            output_id.unwrap_or("estimate"),
        ));
    }
    if id.eq_ignore_ascii_case("atr_percentile") {
        return Some(compute_atr_percentile_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("bull_power_vs_bear_power") {
        return Some(compute_bull_power_vs_bear_power_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("didi_index") {
        return Some(compute_didi_index_batch(req, output_id.unwrap_or("short")));
    }
    if id.eq_ignore_ascii_case("ehlers_autocorrelation_periodogram") {
        return Some(compute_ehlers_autocorrelation_periodogram_batch(
            req,
            output_id.unwrap_or("dominant_cycle"),
        ));
    }
    if id.eq_ignore_ascii_case("ehlers_linear_extrapolation_predictor") {
        return Some(compute_ehlers_linear_extrapolation_predictor_batch(
            req,
            output_id.unwrap_or("prediction"),
        ));
    }
    if id.eq_ignore_ascii_case("kase_peak_oscillator_with_divergences") {
        return Some(compute_kase_peak_oscillator_with_divergences_batch(
            req,
            output_id.unwrap_or("oscillator"),
        ));
    }
    if id.eq_ignore_ascii_case("absolute_strength_index_oscillator") {
        return Some(compute_absolute_strength_index_oscillator_batch(
            req,
            output_id.unwrap_or("oscillator"),
        ));
    }
    if id.eq_ignore_ascii_case("adaptive_bandpass_trigger_oscillator") {
        return Some(compute_adaptive_bandpass_trigger_oscillator_batch(
            req,
            output_id.unwrap_or("in_phase"),
        ));
    }
    if id.eq_ignore_ascii_case("premier_rsi_oscillator") {
        return Some(compute_premier_rsi_oscillator_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("multi_length_stochastic_average") {
        return Some(compute_multi_length_stochastic_average_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("hull_butterfly_oscillator") {
        return Some(compute_hull_butterfly_oscillator_batch(
            req,
            output_id.unwrap_or("oscillator"),
        ));
    }
    if id.eq_ignore_ascii_case("fibonacci_trailing_stop") {
        return Some(compute_fibonacci_trailing_stop_batch(
            req,
            output_id.unwrap_or("trailing_stop"),
        ));
    }
    if id.eq_ignore_ascii_case("fibonacci_entry_bands") {
        return Some(compute_fibonacci_entry_bands_batch(
            req,
            output_id.unwrap_or("middle"),
        ));
    }
    if id.eq_ignore_ascii_case("volume_energy_reservoirs") {
        return Some(compute_volume_energy_reservoirs_batch(
            req,
            output_id.unwrap_or("momentum"),
        ));
    }
    if id.eq_ignore_ascii_case("neighboring_trailing_stop") {
        return Some(compute_neighboring_trailing_stop_batch(
            req,
            output_id.unwrap_or("trailing_stop"),
        ));
    }
    if id.eq_ignore_ascii_case("grover_llorens_cycle_oscillator") {
        return Some(compute_grover_llorens_cycle_oscillator_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("historical_volatility") {
        return Some(compute_historical_volatility_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("squeeze_index") {
        return Some(compute_squeeze_index_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("stochastic_distance") {
        return Some(compute_stochastic_distance_batch(
            req,
            output_id.unwrap_or("oscillator"),
        ));
    }
    if id.eq_ignore_ascii_case("vertical_horizontal_filter") {
        return Some(compute_vertical_horizontal_filter_batch(
            req,
            output_id.unwrap_or("value"),
        ));
    }
    if id.eq_ignore_ascii_case("intraday_momentum_index") {
        if let Some(out) = output_id {
            return Some(compute_intraday_momentum_index_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("vwap_zscore_with_signals") {
        if let Some(out) = output_id {
            return Some(compute_vwap_zscore_with_signals_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("macd_wave_signal_pro") {
        if let Some(out) = output_id {
            return Some(compute_macd_wave_signal_pro_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("hema_trend_levels") {
        if let Some(out) = output_id {
            return Some(compute_hema_trend_levels_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("demand_index") {
        if let Some(out) = output_id {
            return Some(compute_demand_index_batch(req, out));
        }
    }
    if id.eq_ignore_ascii_case("gopalakrishnan_range_index") {
        return Some(compute_gopalakrishnan_range_index_batch(
            req,
            output_id.unwrap_or("value"),
        ));
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
    indicator_id: &str,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    if indicator_id.eq_ignore_ascii_case("logarithmic_moving_average") {
        return compute_logarithmic_moving_average_batch(req, output_id);
    }
    if is_moving_average(indicator_id) {
        if let Some(info) = get_indicator(indicator_id) {
            return compute_ma_batch(req, info, output_id);
        }
    }
    match indicator_id {
        "accumulation_swing_index" => compute_accumulation_swing_index_batch(req, output_id),
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
        "bulls_v_bears" => compute_bulls_v_bears_batch(req, output_id),
        "cci" => compute_cci_batch(req, output_id),
        "cci_cycle" => compute_cci_cycle_batch(req, output_id),
        "cfo" => compute_cfo_batch(req, output_id),
        "cycle_channel_oscillator" => compute_cycle_channel_oscillator_batch(req, output_id),
        "daily_factor" => compute_daily_factor_batch(req, output_id),
        "ehlers_adaptive_cg" => compute_ehlers_adaptive_cg_batch(req, output_id),
        "ehlers_adaptive_cyber_cycle" => compute_ehlers_adaptive_cyber_cycle_batch(req, output_id),
        "adaptive_schaff_trend_cycle" => compute_adaptive_schaff_trend_cycle_batch(req, output_id),
        "adaptive_momentum_oscillator" => {
            compute_adaptive_momentum_oscillator_batch(req, output_id)
        }
        "adaptive_bounds_rsi" => compute_adaptive_bounds_rsi_batch(req, output_id),
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
        "ehlers_detrending_filter" => compute_ehlers_detrending_filter_batch(req, output_id),
        "ehlers_fm_demodulator" => compute_ehlers_fm_demodulator_batch(req, output_id),
        "ehlers_simple_cycle_indicator" => {
            compute_ehlers_simple_cycle_indicator_batch(req, output_id)
        }
        "ehlers_smoothed_adaptive_momentum" => {
            compute_ehlers_smoothed_adaptive_momentum_batch(req, output_id)
        }
        "ewma_volatility" => compute_ewma_volatility_batch(req, output_id),
        "qstick" => compute_qstick_batch(req, output_id),
        "reverse_rsi" => compute_reverse_rsi_batch(req, output_id),
        "percentile_nearest_rank" => compute_percentile_nearest_rank_batch(req, output_id),
        "obv" => compute_obv_batch(req, output_id),
        "on_balance_volume_oscillator" => {
            compute_on_balance_volume_oscillator_batch(req, output_id)
        }
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
        "adjustable_ma_alternating_extremities" => {
            compute_adjustable_ma_alternating_extremities_batch(req, output_id)
        }
        "vi" => compute_vi_batch(req, output_id),
        "wavetrend" => compute_wavetrend_batch(req, output_id),
        "wto" => compute_wto_batch(req, output_id),
        "rogers_satchell_volatility" => compute_rogers_satchell_volatility_batch(req, output_id),
        "historical_volatility_percentile" => {
            compute_historical_volatility_percentile_batch(req, output_id)
        }
        "historical_volatility_rank" => compute_historical_volatility_rank_batch(req, output_id),
        "dual_ulcer_index" => compute_dual_ulcer_index_batch(req, output_id),
        "fractal_dimension_index" => compute_fractal_dimension_index_batch(req, output_id),
        "ichimoku_oscillator" => compute_ichimoku_oscillator_batch(req, output_id),
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
        "garman_klass_volatility" => compute_garman_klass_volatility_batch(req, output_id),
        "advance_decline_line" => compute_advance_decline_line_batch(req, output_id),
        "decisionpoint_breadth_swenlin_trading_oscillator" => {
            compute_decisionpoint_breadth_swenlin_trading_oscillator_batch(req, output_id)
        }
        "velocity_acceleration_indicator" => {
            compute_velocity_acceleration_indicator_batch(req, output_id)
        }
        "normalized_resonator" => compute_normalized_resonator_batch(req, output_id),
        "monotonicity_index" => compute_monotonicity_index_batch(req, output_id),
        "half_causal_estimator" => compute_half_causal_estimator_batch(req, output_id),
        "atr_percentile" => compute_atr_percentile_batch(req, output_id),
        "andean_oscillator" => compute_andean_oscillator_batch(req, output_id),
        "bull_power_vs_bear_power" => compute_bull_power_vs_bear_power_batch(req, output_id),
        "didi_index" => compute_didi_index_batch(req, output_id),
        "ehlers_autocorrelation_periodogram" => {
            compute_ehlers_autocorrelation_periodogram_batch(req, output_id)
        }
        "ehlers_linear_extrapolation_predictor" => {
            compute_ehlers_linear_extrapolation_predictor_batch(req, output_id)
        }
        "absolute_strength_index_oscillator" => {
            compute_absolute_strength_index_oscillator_batch(req, output_id)
        }
        "adaptive_bandpass_trigger_oscillator" => {
            compute_adaptive_bandpass_trigger_oscillator_batch(req, output_id)
        }
        "premier_rsi_oscillator" => compute_premier_rsi_oscillator_batch(req, output_id),
        "multi_length_stochastic_average" => {
            compute_multi_length_stochastic_average_batch(req, output_id)
        }
        "hull_butterfly_oscillator" => compute_hull_butterfly_oscillator_batch(req, output_id),
        "fibonacci_trailing_stop" => compute_fibonacci_trailing_stop_batch(req, output_id),
        "fibonacci_entry_bands" => compute_fibonacci_entry_bands_batch(req, output_id),
        "volume_energy_reservoirs" => compute_volume_energy_reservoirs_batch(req, output_id),
        "neighboring_trailing_stop" => compute_neighboring_trailing_stop_batch(req, output_id),
        "grover_llorens_cycle_oscillator" => {
            compute_grover_llorens_cycle_oscillator_batch(req, output_id)
        }
        "historical_volatility" => compute_historical_volatility_batch(req, output_id),
        "hypertrend" => compute_hypertrend_batch(req, output_id),
        "ict_propulsion_block" => compute_ict_propulsion_block_batch(req, output_id),
        "impulse_macd" => compute_impulse_macd_batch(req, output_id),
        "l1_ehlers_phasor" => compute_l1_ehlers_phasor_batch(req, output_id),
        "l2_ehlers_signal_to_noise" => compute_l2_ehlers_signal_to_noise_batch(req, output_id),
        "keltner_channel_width_oscillator" => {
            compute_keltner_channel_width_oscillator_batch(req, output_id)
        }
        "leavitt_convolution_acceleration" => {
            compute_leavitt_convolution_acceleration_batch(req, output_id)
        }
        "linear_regression_intensity" => compute_linear_regression_intensity_batch(req, output_id),
        "market_meanness_index" => compute_market_meanness_index_batch(req, output_id),
        "mesa_stochastic_multi_length" => {
            compute_mesa_stochastic_multi_length_batch(req, output_id)
        }
        "moving_average_cross_probability" => {
            compute_moving_average_cross_probability_batch(req, output_id)
        }
        "momentum_ratio_oscillator" => compute_momentum_ratio_oscillator_batch(req, output_id),
        "parkinson_volatility" => compute_parkinson_volatility_batch(req, output_id),
        "price_moving_average_ratio_percentile" => {
            compute_price_moving_average_ratio_percentile_batch(req, output_id)
        }
        "pretty_good_oscillator" => compute_pretty_good_oscillator_batch(req, output_id),
        "price_density_market_noise" => compute_price_density_market_noise_batch(req, output_id),
        "psychological_line" => compute_psychological_line_batch(req, output_id),
        "random_walk_index" => compute_random_walk_index_batch(req, output_id),
        "rank_correlation_index" => compute_rank_correlation_index_batch(req, output_id),
        "relative_strength_index_wave_indicator" => {
            compute_relative_strength_index_wave_indicator_batch(req, output_id)
        }
        "regression_slope_oscillator" => compute_regression_slope_oscillator_batch(req, output_id),
        "squeeze_index" => compute_squeeze_index_batch(req, output_id),
        "smoothed_gaussian_trend_filter" => {
            compute_smoothed_gaussian_trend_filter_batch(req, output_id)
        }
        "smooth_theil_sen" => compute_smooth_theil_sen_batch(req, output_id),
        "spearman_correlation" => compute_spearman_correlation_batch(req, output_id),
        "stochastic_adaptive_d" => compute_stochastic_adaptive_d_batch(req, output_id),
        "stochastic_connors_rsi" => compute_stochastic_connors_rsi_batch(req, output_id),
        "stochastic_distance" => compute_stochastic_distance_batch(req, output_id),
        "supertrend_oscillator" => compute_supertrend_oscillator_batch(req, output_id),
        "trend_trigger_factor" => compute_trend_trigger_factor_batch(req, output_id),
        "trend_continuation_factor" => compute_trend_continuation_factor_batch(req, output_id),
        "twiggs_money_flow" => compute_twiggs_money_flow_batch(req, output_id),
        "vertical_horizontal_filter" => compute_vertical_horizontal_filter_batch(req, output_id),
        "intraday_momentum_index" => compute_intraday_momentum_index_batch(req, output_id),
        "volatility_quality_index" => compute_volatility_quality_index_batch(req, output_id),
        "volatility_ratio_adaptive_rsx" => {
            compute_volatility_ratio_adaptive_rsx_batch(req, output_id)
        }
        "volume_weighted_stochastic_rsi" => {
            compute_volume_weighted_stochastic_rsi_batch(req, output_id)
        }
        "volume_zone_oscillator" => compute_volume_zone_oscillator_batch(req, output_id),
        "vwap_deviation_oscillator" => compute_vwap_deviation_oscillator_batch(req, output_id),
        "vwap_zscore_with_signals" => compute_vwap_zscore_with_signals_batch(req, output_id),
        "macd_wave_signal_pro" => compute_macd_wave_signal_pro_batch(req, output_id),
        "hema_trend_levels" => compute_hema_trend_levels_batch(req, output_id),
        "demand_index" => compute_demand_index_batch(req, output_id),
        "kase_peak_oscillator_with_divergences" => {
            compute_kase_peak_oscillator_with_divergences_batch(req, output_id)
        }
        "gopalakrishnan_range_index" => compute_gopalakrishnan_range_index_batch(req, output_id),
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
            indicator: indicator_id.to_string(),
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

fn normalize_output_token(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
        }
    }
    if normalized == "values" {
        "value".to_string()
    } else {
        normalized
    }
}

fn output_id_matches(candidate: &str, requested: &str) -> bool {
    candidate.eq_ignore_ascii_case(requested)
        || normalize_output_token(candidate) == normalize_output_token(requested)
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
            if !output_id_matches(only, req) {
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
        .find(|o| output_id_matches(o.id, req))
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
    if info.id.eq_ignore_ascii_case("wilders") {
        return compute_wilders_ma_batch_direct(req, info, output_id, data, cols);
    }
    if info.id.eq_ignore_ascii_case("edcf")
        && matches!(req.kernel.to_non_batch(), Kernel::Avx2 | Kernel::Avx512)
    {
        return compute_edcf_ma_batch_direct(req, info, output_id, data, cols);
    }
    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    if info.id.eq_ignore_ascii_case("zlema") {
        return compute_zlema_ma_batch_direct(req, info, output_id, data, cols);
    }
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

fn compute_wilders_ma_batch_direct(
    req: IndicatorBatchRequest<'_>,
    info: &IndicatorInfo,
    output_id: &str,
    data: MaData<'_>,
    cols: usize,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let prices = match data {
        MaData::Slice(values) => values,
        MaData::Candles { candles, source } => source_type(candles, source),
    };
    let rows = req.combos.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: info.id.to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    let kernel = req.kernel.to_non_batch();
    for (row, combo) in req.combos.iter().enumerate() {
        let period = ma_period_for_combo(info, combo.params)?;
        let input = WildersInput::from_slice(
            prices,
            WildersParams {
                period: Some(period),
            },
        );
        let start = row * cols;
        let end = start + cols;
        wilders_into_slice(&mut matrix[start..end], &input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: info.id.to_string(),
                details: e.to_string(),
            }
        })?;
    }

    Ok(f64_output(output_id, rows, cols, matrix))
}

fn compute_zlema_ma_batch_direct(
    req: IndicatorBatchRequest<'_>,
    info: &IndicatorInfo,
    output_id: &str,
    data: MaData<'_>,
    cols: usize,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output(info.id, output_id)?;
    let prices = match data {
        MaData::Slice(values) => values,
        MaData::Candles { candles, source } => source_type(candles, source),
    };
    let rows = req.combos.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: info.id.to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    let kernel = req.kernel.to_non_batch();
    for (row, combo) in req.combos.iter().enumerate() {
        let period = ma_period_for_combo(info, combo.params)?;
        let input = ZlemaInput::from_slice(
            prices,
            ZlemaParams {
                period: Some(period),
            },
        );
        let start = row * cols;
        let end = start + cols;
        zlema_into_slice(&mut matrix[start..end], &input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: info.id.to_string(),
                details: e.to_string(),
            }
        })?;
    }

    Ok(f64_output(output_id, rows, cols, matrix))
}

fn compute_edcf_ma_batch_direct(
    req: IndicatorBatchRequest<'_>,
    info: &IndicatorInfo,
    output_id: &str,
    data: MaData<'_>,
    cols: usize,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output(info.id, output_id)?;
    let prices = match data {
        MaData::Slice(values) => values,
        MaData::Candles { candles, source } => source_type(candles, source),
    };
    let rows = req.combos.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: info.id.to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    let kernel = req.kernel.to_non_batch();
    for (row, combo) in req.combos.iter().enumerate() {
        let period = ma_period_for_combo(info, combo.params)?;
        let input = EdcfInput::from_slice(
            prices,
            EdcfParams {
                period: Some(period),
            },
        );
        let start = row * cols;
        let end = start + cols;
        edcf_into_slice(&mut matrix[start..end], &input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: info.id.to_string(),
                details: e.to_string(),
            }
        })?;
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
    collect_f64_into_rows("efi", output_id, req.combos, price.len(), |params, row| {
        let period = get_usize_param("efi", params, "period", 13)?;
        let input = EfiInput::from_slices(
            price,
            volume,
            EfiParams {
                period: Some(period),
            },
        );
        efi_into_slice(row, &input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "efi".to_string(),
            details: e.to_string(),
        })
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
    let rows = req.combos.len();
    let cols = volume.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "vosc".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
        let short_period = get_usize_param("vosc", params, "short_period", 2)?;
        let long_period = get_usize_param("vosc", params, "long_period", 5)?;
        let input = VoscInput::from_slice(
            volume,
            VoscParams {
                short_period: Some(short_period),
                long_period: Some(long_period),
            },
        );
        let start = row * cols;
        let end = start + cols;
        vosc_into_slice(&mut matrix[start..end], &input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "vosc".to_string(),
                details: e.to_string(),
            }
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
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
    let rows = req.combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "apo".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
        let short_period = get_usize_param("apo", params, "short_period", 10)?;
        let long_period = get_usize_param("apo", params, "long_period", 20)?;
        let input = ApoInput::from_slice(
            data,
            ApoParams {
                short_period: Some(short_period),
                long_period: Some(long_period),
            },
        );
        let start = row * cols;
        let end = start + cols;
        apo_into_slice(&mut matrix[start..end], &input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "apo".to_string(),
                details: e.to_string(),
            }
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
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
    collect_f64_into_rows("dpo", output_id, req.combos, data.len(), |params, row| {
        let period = get_usize_param("dpo", params, "period", 5)?;
        let input = DpoInput::from_slice(
            data,
            DpoParams {
                period: Some(period),
            },
        );
        dpo_into_slice(row, &input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "dpo".to_string(),
            details: e.to_string(),
        })
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
    let rows = req.combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "vpt".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        vpt_into_slice(&mut matrix[start..end], close, volume, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "vpt".to_string(),
                details: e.to_string(),
            }
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
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
    let rows = req.combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "wclprice".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for row in 0..rows {
        let input = WclpriceInput::from_slices(high, low, close);
        let start = row * cols;
        let end = start + cols;
        wclprice_into_slice(&mut matrix[start..end], &input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "wclprice".to_string(),
                details: e.to_string(),
            }
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
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
    let rows = req.combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "zscore".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
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
        let start = row * cols;
        let end = start + cols;
        zscore_into_slice(&mut matrix[start..end], &input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "zscore".to_string(),
                details: e.to_string(),
            }
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
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
    let field = match output_id {
        "amo" | "value" => AdaptiveMomentumOscillatorOutputField::Amo,
        "ama" => AdaptiveMomentumOscillatorOutputField::Ama,
        other => {
            return Err(IndicatorDispatchError::UnknownOutput {
                indicator: "adaptive_momentum_oscillator".to_string(),
                output: other.to_string(),
            })
        }
    };
    collect_f64_into_rows(
        "adaptive_momentum_oscillator",
        output_id,
        req.combos,
        data.len(),
        |params, row| {
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
            adaptive_momentum_oscillator_output_into_slice(row, &input, kernel, field).map_err(
                |e| IndicatorDispatchError::ComputeFailed {
                    indicator: "adaptive_momentum_oscillator".to_string(),
                    details: e.to_string(),
                },
            )?;
            Ok(())
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
    let field = if output_id.eq_ignore_ascii_case("macd") || output_id.eq_ignore_ascii_case("value")
    {
        AdaptiveMacdOutputField::Macd
    } else if output_id.eq_ignore_ascii_case("signal") {
        AdaptiveMacdOutputField::Signal
    } else if output_id.eq_ignore_ascii_case("hist") {
        AdaptiveMacdOutputField::Hist
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "adaptive_macd".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "adaptive_macd",
        output_id,
        req.combos,
        data.len(),
        |params, row| {
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
            adaptive_macd_output_into_slice(row, &input, kernel, field).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "adaptive_macd".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(())
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
    let rows = req.combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "willr".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
        let period = get_usize_param("willr", params, "period", 14)?;
        let input = WillrInput::from_slices(
            high,
            low,
            close,
            WillrParams {
                period: Some(period),
            },
        );
        let start = row * cols;
        let end = start + cols;
        willr_into_slice(&mut matrix[start..end], &input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "willr".to_string(),
                details: e.to_string(),
            }
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
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
    let field = if output_id.eq_ignore_ascii_case("macd") || output_id.eq_ignore_ascii_case("value")
    {
        VwmacdOutputField::Macd
    } else if output_id.eq_ignore_ascii_case("signal") {
        VwmacdOutputField::Signal
    } else if output_id.eq_ignore_ascii_case("hist") {
        VwmacdOutputField::Hist
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "vwmacd".to_string(),
            output: output_id.to_string(),
        });
    };
    let rows = req.combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "vwmacd".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
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
        let start = row * cols;
        let end = start + cols;
        vwmacd_output_into_slice(&mut matrix[start..end], &input, kernel, field).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "vwmacd".to_string(),
                details: e.to_string(),
            }
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
}

fn compute_vpci_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (close, volume) = extract_close_volume_input("vpci", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("vpci") || output_id.eq_ignore_ascii_case("value")
    {
        VpciOutputField::Vpci
    } else if output_id.eq_ignore_ascii_case("vpcis") {
        VpciOutputField::Vpcis
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "vpci".to_string(),
            output: output_id.to_string(),
        });
    };
    let rows = req.combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "vpci".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
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
        let start = row * cols;
        let end = start + cols;
        vpci_output_into_slice(&mut matrix[start..end], &input, kernel, field).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "vpci".to_string(),
                details: e.to_string(),
            }
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
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
    let field = if output_id.eq_ignore_ascii_case("up")
        || output_id.eq_ignore_ascii_case("aroon_up")
        || output_id.eq_ignore_ascii_case("value")
    {
        AroonOutputField::Up
    } else if output_id.eq_ignore_ascii_case("down") || output_id.eq_ignore_ascii_case("aroon_down")
    {
        AroonOutputField::Down
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "aroon".to_string(),
            output: output_id.to_string(),
        });
    };

    let rows = req.combos.len();
    let cols = high.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "aroon".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
        let length = get_usize_param("aroon", params, "length", 14)?;
        let input = AroonInput::from_slices_hl(
            high,
            low,
            AroonParams {
                length: Some(length),
            },
        );
        let start = row * cols;
        let end = start + cols;
        aroon_output_into_slice(&mut matrix[start..end], &input, kernel, field).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "aroon".to_string(),
                details: e.to_string(),
            }
        })?;
    }

    Ok(f64_output(output_id, rows, cols, matrix))
}

fn compute_aroonosc_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("aroonosc", output_id)?;
    let (high, low) = extract_high_low_input("aroonosc", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "aroonosc",
        output_id,
        req.combos,
        high.len(),
        |params, row| {
            let length = get_usize_param("aroonosc", params, "length", 14)?;
            let input = AroonOscInput::from_slices_hl(
                high,
                low,
                AroonOscParams {
                    length: Some(length),
                },
            );
            aroon_osc_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "aroonosc".to_string(),
                    details: e.to_string(),
                }
            })
        },
    )
}

fn compute_di_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("di", req.data)?;
    let kernel = req.kernel.to_non_batch();
    let selected =
        if output_id.eq_ignore_ascii_case("plus") || output_id.eq_ignore_ascii_case("value") {
            1u8
        } else if output_id.eq_ignore_ascii_case("minus") {
            2u8
        } else {
            return Err(IndicatorDispatchError::UnknownOutput {
                indicator: "di".to_string(),
                output: output_id.to_string(),
            });
        };
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
        let out = if selected == 1 {
            di_plus_with_kernel(&input, kernel)
        } else {
            di_minus_with_kernel(&input, kernel)
        }
        .map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "di".to_string(),
            details: e.to_string(),
        })?;
        Ok(out)
    })
}

fn compute_dm_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let selected =
        if output_id.eq_ignore_ascii_case("plus") || output_id.eq_ignore_ascii_case("value") {
            1
        } else if output_id.eq_ignore_ascii_case("minus") {
            2
        } else {
            return Err(IndicatorDispatchError::UnknownOutput {
                indicator: "dm".to_string(),
                output: output_id.to_string(),
            });
        };
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
        let out = if selected == 1 {
            dm_plus_with_kernel(&input, kernel)
        } else {
            dm_minus_with_kernel(&input, kernel)
        }
        .map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "dm".to_string(),
            details: e.to_string(),
        })?;
        Ok(out)
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
    let selected =
        if output_id.eq_ignore_ascii_case("upper") || output_id.eq_ignore_ascii_case("value") {
            0
        } else if output_id.eq_ignore_ascii_case("middle") {
            1
        } else if output_id.eq_ignore_ascii_case("lower") {
            2
        } else {
            return Err(IndicatorDispatchError::UnknownOutput {
                indicator: "donchian".to_string(),
                output: output_id.to_string(),
            });
        };
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
        let out = match selected {
            0 => donchian_upper_with_kernel(&input, kernel),
            1 => donchian_middle_with_kernel(&input, kernel),
            2 => donchian_lower_with_kernel(&input, kernel),
            _ => unreachable!(),
        }
        .map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "donchian".to_string(),
            details: e.to_string(),
        })?;
        Ok(out)
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
    let field = if output_id.eq_ignore_ascii_case("ma") || output_id.eq_ignore_ascii_case("value") {
        AdjustableMaAlternatingExtremitiesOutputField::Ma
    } else if output_id.eq_ignore_ascii_case("upper") {
        AdjustableMaAlternatingExtremitiesOutputField::Upper
    } else if output_id.eq_ignore_ascii_case("lower") {
        AdjustableMaAlternatingExtremitiesOutputField::Lower
    } else if output_id.eq_ignore_ascii_case("extremity") {
        AdjustableMaAlternatingExtremitiesOutputField::Extremity
    } else if output_id.eq_ignore_ascii_case("state") {
        AdjustableMaAlternatingExtremitiesOutputField::State
    } else if output_id.eq_ignore_ascii_case("changed") {
        AdjustableMaAlternatingExtremitiesOutputField::Changed
    } else if output_id.eq_ignore_ascii_case("smoothed_open") {
        AdjustableMaAlternatingExtremitiesOutputField::SmoothedOpen
    } else if output_id.eq_ignore_ascii_case("smoothed_high") {
        AdjustableMaAlternatingExtremitiesOutputField::SmoothedHigh
    } else if output_id.eq_ignore_ascii_case("smoothed_low") {
        AdjustableMaAlternatingExtremitiesOutputField::SmoothedLow
    } else if output_id.eq_ignore_ascii_case("smoothed_close") {
        AdjustableMaAlternatingExtremitiesOutputField::SmoothedClose
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "adjustable_ma_alternating_extremities".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "adjustable_ma_alternating_extremities",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
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
            adjustable_ma_alternating_extremities_output_into_slice(row, &input, kernel, field)
                .map_err(|e| IndicatorDispatchError::ComputeFailed {
                    indicator: "adjustable_ma_alternating_extremities".to_string(),
                    details: e.to_string(),
                })?;
            Ok(())
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
    let field = if output_id.eq_ignore_ascii_case("wt1") || output_id.eq_ignore_ascii_case("value")
    {
        WavetrendOutputField::Wt1
    } else if output_id.eq_ignore_ascii_case("wt2") {
        WavetrendOutputField::Wt2
    } else if output_id.eq_ignore_ascii_case("wt_diff") {
        WavetrendOutputField::WtDiff
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "wavetrend".to_string(),
            output: output_id.to_string(),
        });
    };
    let rows = req.combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "wavetrend".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
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
        let start = row * cols;
        let end = start + cols;
        wavetrend_output_into_slice(&mut matrix[start..end], &input, kernel, field).map_err(
            |e| IndicatorDispatchError::ComputeFailed {
                indicator: "wavetrend".to_string(),
                details: e.to_string(),
            },
        )?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
}

fn compute_wto_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("wto", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("wavetrend1")
        || output_id.eq_ignore_ascii_case("wt1")
        || output_id.eq_ignore_ascii_case("value")
    {
        WtoOutputField::Wavetrend1
    } else if output_id.eq_ignore_ascii_case("wavetrend2") || output_id.eq_ignore_ascii_case("wt2")
    {
        WtoOutputField::Wavetrend2
    } else if output_id.eq_ignore_ascii_case("histogram") || output_id.eq_ignore_ascii_case("hist")
    {
        WtoOutputField::Histogram
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "wto".to_string(),
            output: output_id.to_string(),
        });
    };
    let rows = req.combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "wto".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
        let channel_length = get_usize_param("wto", params, "channel_length", 10)?;
        let average_length = get_usize_param("wto", params, "average_length", 21)?;
        let input = WtoInput::from_slice(
            data,
            WtoParams {
                channel_length: Some(channel_length),
                average_length: Some(average_length),
            },
        );
        let start = row * cols;
        let end = start + cols;
        wto_output_into_slice(&mut matrix[start..end], &input, kernel, field).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "wto".to_string(),
                details: e.to_string(),
            }
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
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
    let field = if output_id.eq_ignore_ascii_case("hvr") || output_id.eq_ignore_ascii_case("value")
    {
        HistoricalVolatilityRankOutputField::Hvr
    } else if output_id.eq_ignore_ascii_case("hv") {
        HistoricalVolatilityRankOutputField::Hv
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "historical_volatility_rank".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "historical_volatility_rank",
        output_id,
        req.combos,
        data.len(),
        |params, row| {
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
            historical_volatility_rank_output_into_slice(row, &input, kernel, field).map_err(
                |e| IndicatorDispatchError::ComputeFailed {
                    indicator: "historical_volatility_rank".to_string(),
                    details: e.to_string(),
                },
            )?;
            Ok(())
        },
    )
}

fn compute_dual_ulcer_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("dual_ulcer_index", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("long_ulcer")
        || output_id.eq_ignore_ascii_case("uulcer")
        || output_id.eq_ignore_ascii_case("value")
    {
        DualUlcerIndexOutputField::LongUlcer
    } else if output_id.eq_ignore_ascii_case("short_ulcer")
        || output_id.eq_ignore_ascii_case("dulcer")
    {
        DualUlcerIndexOutputField::ShortUlcer
    } else if output_id.eq_ignore_ascii_case("threshold") {
        DualUlcerIndexOutputField::Threshold
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "dual_ulcer_index".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "dual_ulcer_index",
        output_id,
        req.combos,
        data.len(),
        |params, row| {
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
            dual_ulcer_index_output_into_slice(row, &input, kernel, field).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "dual_ulcer_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(())
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
    if periods.len() == 1 {
        let period = periods[0];
        let input = VolumeWeightedRsiInput::from_slices(
            close,
            volume,
            VolumeWeightedRsiParams {
                period: Some(period),
            },
        );
        let mut values = alloc_uninit_f64(close.len());
        volume_weighted_rsi_into_slice(&mut values, &input, req.kernel.to_non_batch()).map_err(
            |e| IndicatorDispatchError::ComputeFailed {
                indicator: "volume_weighted_rsi".to_string(),
                details: e.to_string(),
            },
        )?;
        return Ok(f64_output(output_id, 1, close.len(), values));
    }
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
    let (open, high, low, close, volume) = extract_ohlcv_full_input("reversal_signals", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "reversal_signals",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let lookback_period =
                get_usize_param("reversal_signals", params, "lookback_period", 12)?;
            let confirmation_period =
                get_usize_param("reversal_signals", params, "confirmation_period", 3)?;
            let use_volume_confirmation =
                get_bool_param("reversal_signals", params, "use_volume_confirmation", true)?;
            let trend_ma_period =
                get_usize_param("reversal_signals", params, "trend_ma_period", 50)?;
            let trend_ma_type = get_enum_param("reversal_signals", params, "trend_ma_type", "EMA")?;
            let ma_step_period = get_usize_param("reversal_signals", params, "ma_step_period", 33)?;
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
            if output_id.eq_ignore_ascii_case("stepped_ma")
                || output_id.eq_ignore_ascii_case("value")
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
        },
    )
}

fn compute_zig_zag_channels_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("zig_zag_channels", req.data)?;
    let kernel = req.kernel.to_non_batch();
    let field =
        if output_id.eq_ignore_ascii_case("middle") || output_id.eq_ignore_ascii_case("value") {
            0
        } else if output_id.eq_ignore_ascii_case("upper") {
            1
        } else if output_id.eq_ignore_ascii_case("lower") {
            2
        } else {
            return Err(IndicatorDispatchError::UnknownOutput {
                indicator: "zig_zag_channels".to_string(),
                output: output_id.to_string(),
            });
        };
    collect_f64_into_rows(
        "zig_zag_channels",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
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
            let mut s0 = alloc_uninit_f64(close.len());
            let mut s1 = alloc_uninit_f64(close.len());
            let result = match field {
                0 => zig_zag_channels_into_slice(row, &mut s0, &mut s1, &input, kernel),
                1 => zig_zag_channels_into_slice(&mut s0, row, &mut s1, &input, kernel),
                2 => zig_zag_channels_into_slice(&mut s0, &mut s1, row, &input, kernel),
                _ => unreachable!(),
            };
            result.map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "zig_zag_channels".to_string(),
                details: e.to_string(),
            })?;
            Ok(())
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
    let field = if output_id.eq_ignore_ascii_case("up") || output_id.eq_ignore_ascii_case("value") {
        DirectionalImbalanceIndexOutputField::Up
    } else if output_id.eq_ignore_ascii_case("down") {
        DirectionalImbalanceIndexOutputField::Down
    } else if output_id.eq_ignore_ascii_case("bulls") {
        DirectionalImbalanceIndexOutputField::Bulls
    } else if output_id.eq_ignore_ascii_case("bears") {
        DirectionalImbalanceIndexOutputField::Bears
    } else if output_id.eq_ignore_ascii_case("upper") {
        DirectionalImbalanceIndexOutputField::Upper
    } else if output_id.eq_ignore_ascii_case("lower") {
        DirectionalImbalanceIndexOutputField::Lower
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "directional_imbalance_index".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "directional_imbalance_index",
        output_id,
        req.combos,
        high.len(),
        |params, row| {
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
            directional_imbalance_index_output_into_slice(row, &input, kernel, field).map_err(
                |e| IndicatorDispatchError::ComputeFailed {
                    indicator: "directional_imbalance_index".to_string(),
                    details: e.to_string(),
                },
            )?;
            Ok(())
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
    let field =
        if output_id.eq_ignore_ascii_case("strength") || output_id.eq_ignore_ascii_case("value") {
            CandleStrengthOscillatorOutputField::Strength
        } else if output_id.eq_ignore_ascii_case("highs") {
            CandleStrengthOscillatorOutputField::Highs
        } else if output_id.eq_ignore_ascii_case("lows") {
            CandleStrengthOscillatorOutputField::Lows
        } else if output_id.eq_ignore_ascii_case("mid") {
            CandleStrengthOscillatorOutputField::Mid
        } else if output_id.eq_ignore_ascii_case("long_signal") {
            CandleStrengthOscillatorOutputField::LongSignal
        } else if output_id.eq_ignore_ascii_case("short_signal") {
            CandleStrengthOscillatorOutputField::ShortSignal
        } else {
            return Err(IndicatorDispatchError::UnknownOutput {
                indicator: "candle_strength_oscillator".to_string(),
                output: output_id.to_string(),
            });
        };
    collect_f64_into_rows(
        "candle_strength_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
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
            candle_strength_oscillator_output_into_slice(row, &input, kernel, field).map_err(
                |e| IndicatorDispatchError::ComputeFailed {
                    indicator: "candle_strength_oscillator".to_string(),
                    details: e.to_string(),
                },
            )?;
            Ok(())
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
    let output_is_filtered =
        output_id.eq_ignore_ascii_case("filtered") || output_id.eq_ignore_ascii_case("value");
    let output_is_correlation = output_id.eq_ignore_ascii_case("correlation");
    if !(output_is_filtered || output_is_correlation) {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "autocorrelation_indicator".to_string(),
            output: output_id.to_string(),
        });
    }
    let field = if output_is_filtered {
        AutocorrelationIndicatorOutputField::Filtered
    } else {
        AutocorrelationIndicatorOutputField::Correlation { lag: 1 }
    };
    collect_f64_into_rows(
        "autocorrelation_indicator",
        output_id,
        req.combos,
        data.len(),
        |params, row| {
            let length = get_usize_param("autocorrelation_indicator", params, "length", 20)?;
            let lag = get_usize_param("autocorrelation_indicator", params, "lag", 1)?;
            let use_test_signal = get_bool_param(
                "autocorrelation_indicator",
                params,
                "use_test_signal",
                false,
            )?;
            let max_lag = lag.max(1);
            let input = AutocorrelationIndicatorInput::from_slice(
                data,
                AutocorrelationIndicatorParams {
                    length: Some(length),
                    max_lag: Some(max_lag),
                    use_test_signal: Some(use_test_signal),
                },
            );
            let field = match field {
                AutocorrelationIndicatorOutputField::Filtered => {
                    AutocorrelationIndicatorOutputField::Filtered
                }
                AutocorrelationIndicatorOutputField::Correlation { .. } => {
                    AutocorrelationIndicatorOutputField::Correlation { lag }
                }
            };
            autocorrelation_indicator_output_into_slice(row, &input, kernel, field).map_err(
                |e| IndicatorDispatchError::ComputeFailed {
                    indicator: "autocorrelation_indicator".to_string(),
                    details: e.to_string(),
                },
            )?;
            Ok(())
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

fn compute_garman_klass_volatility_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("garman_klass_volatility", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "garman_klass_volatility",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let lookback = get_usize_param("garman_klass_volatility", params, "lookback", 14)?;
            let input = GarmanKlassVolatilityInput::from_slices(
                open,
                high,
                low,
                close,
                GarmanKlassVolatilityParams {
                    lookback: Some(lookback),
                },
            );
            let out = garman_klass_volatility_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "garman_klass_volatility".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("value") {
                return Ok(out.values);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "garman_klass_volatility".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_atr_percentile_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("atr_percentile", output_id)?;
    let (high, low, close) = extract_ohlc_input("atr_percentile", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "atr_percentile",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
            let atr_length = get_usize_param("atr_percentile", params, "atr_length", 10)?;
            let percentile_length =
                get_usize_param("atr_percentile", params, "percentile_length", 50)?;
            let input = AtrPercentileInput::from_slices(
                high,
                low,
                close,
                AtrPercentileParams {
                    atr_length: Some(atr_length),
                    percentile_length: Some(percentile_length),
                },
            );
            atr_percentile_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "atr_percentile".to_string(),
                    details: e.to_string(),
                }
            })
        },
    )
}

fn compute_bull_power_vs_bear_power_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("bull_power_vs_bear_power", output_id)?;
    let (open, high, low, close) = extract_ohlc_full_input("bull_power_vs_bear_power", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "bull_power_vs_bear_power",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
            let period = get_usize_param("bull_power_vs_bear_power", params, "period", 5)?;
            let input = BullPowerVsBearPowerInput::from_slices(
                open,
                high,
                low,
                close,
                BullPowerVsBearPowerParams {
                    period: Some(period),
                },
            );
            bull_power_vs_bear_power_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "bull_power_vs_bear_power".to_string(),
                    details: e.to_string(),
                }
            })
        },
    )
}

fn compute_advance_decline_line_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("advance_decline_line", output_id)?;
    let data = extract_slice_input("advance_decline_line", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "advance_decline_line",
        output_id,
        req.combos,
        data.len(),
        |_params, row| {
            let input = AdvanceDeclineLineInput::from_slice(data, AdvanceDeclineLineParams);
            advance_decline_line_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "advance_decline_line".to_string(),
                    details: e.to_string(),
                }
            })
        },
    )
}

fn compute_didi_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("didi_index", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    let field =
        if output_id.eq_ignore_ascii_case("short") || output_id.eq_ignore_ascii_case("value") {
            DidiIndexOutputField::Short
        } else if output_id.eq_ignore_ascii_case("long") {
            DidiIndexOutputField::Long
        } else if output_id.eq_ignore_ascii_case("crossover") {
            DidiIndexOutputField::Crossover
        } else if output_id.eq_ignore_ascii_case("crossunder") {
            DidiIndexOutputField::Crossunder
        } else {
            return Err(IndicatorDispatchError::UnknownOutput {
                indicator: "didi_index".to_string(),
                output: output_id.to_string(),
            });
        };
    collect_f64_into_rows(
        "didi_index",
        output_id,
        req.combos,
        data.len(),
        |params, row| {
            let short_length = get_usize_param("didi_index", params, "short_length", 3)?;
            let medium_length = get_usize_param("didi_index", params, "medium_length", 8)?;
            let long_length = get_usize_param("didi_index", params, "long_length", 20)?;
            let input = DidiIndexInput::from_slice(
                data,
                DidiIndexParams {
                    short_length: Some(short_length),
                    medium_length: Some(medium_length),
                    long_length: Some(long_length),
                },
            );
            didi_index_output_into_slice(row, &input, kernel, field).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "didi_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(())
        },
    )
}

fn compute_absolute_strength_index_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("absolute_strength_index_oscillator", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("oscillator")
        || output_id.eq_ignore_ascii_case("indicator")
        || output_id.eq_ignore_ascii_case("value")
    {
        AbsoluteStrengthIndexOscillatorOutputField::Oscillator
    } else if output_id.eq_ignore_ascii_case("signal") {
        AbsoluteStrengthIndexOscillatorOutputField::Signal
    } else if output_id.eq_ignore_ascii_case("histogram") || output_id.eq_ignore_ascii_case("hist")
    {
        AbsoluteStrengthIndexOscillatorOutputField::Histogram
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "absolute_strength_index_oscillator".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "absolute_strength_index_oscillator",
        output_id,
        req.combos,
        data.len(),
        |params, row| {
            let ema_length = get_usize_param(
                "absolute_strength_index_oscillator",
                params,
                "ema_length",
                21,
            )?;
            let signal_length = get_usize_param(
                "absolute_strength_index_oscillator",
                params,
                "signal_length",
                34,
            )?;
            let input = AbsoluteStrengthIndexOscillatorInput::from_slice(
                data,
                AbsoluteStrengthIndexOscillatorParams {
                    ema_length: Some(ema_length),
                    signal_length: Some(signal_length),
                },
            );
            absolute_strength_index_oscillator_output_into_slice(row, &input, kernel, field)
                .map_err(|e| IndicatorDispatchError::ComputeFailed {
                    indicator: "absolute_strength_index_oscillator".to_string(),
                    details: e.to_string(),
                })?;
            Ok(())
        },
    )
}

fn compute_adaptive_bandpass_trigger_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("adaptive_bandpass_trigger_oscillator", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "adaptive_bandpass_trigger_oscillator",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let delta =
                get_f64_param("adaptive_bandpass_trigger_oscillator", params, "delta", 0.1)?;
            let alpha = get_f64_param(
                "adaptive_bandpass_trigger_oscillator",
                params,
                "alpha",
                0.07,
            )?;
            let input = AdaptiveBandpassTriggerOscillatorInput::from_slice(
                data,
                AdaptiveBandpassTriggerOscillatorParams {
                    delta: Some(delta),
                    alpha: Some(alpha),
                },
            );
            let out =
                adaptive_bandpass_trigger_oscillator_with_kernel(&input, kernel).map_err(|e| {
                    IndicatorDispatchError::ComputeFailed {
                        indicator: "adaptive_bandpass_trigger_oscillator".to_string(),
                        details: e.to_string(),
                    }
                })?;
            match output_id {
                "in_phase" => Ok(out.in_phase),
                "lead" => Ok(out.lead),
                _ => Err(IndicatorDispatchError::UnknownOutput {
                    indicator: "adaptive_bandpass_trigger_oscillator".to_string(),
                    output: output_id.to_string(),
                }),
            }
        },
    )
}

fn compute_premier_rsi_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("premier_rsi_oscillator", output_id)?;
    let data = extract_slice_input("premier_rsi_oscillator", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "premier_rsi_oscillator",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let rsi_length = get_usize_param("premier_rsi_oscillator", params, "rsi_length", 14)?;
            let stoch_length =
                get_usize_param("premier_rsi_oscillator", params, "stoch_length", 8)?;
            let smooth_length =
                get_usize_param("premier_rsi_oscillator", params, "smooth_length", 25)?;
            let input = PremierRsiOscillatorInput::from_slice(
                data,
                PremierRsiOscillatorParams {
                    rsi_length: Some(rsi_length),
                    stoch_length: Some(stoch_length),
                    smooth_length: Some(smooth_length),
                },
            );
            let out = premier_rsi_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "premier_rsi_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_multi_length_stochastic_average_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("multi_length_stochastic_average", output_id)?;
    let data_len = match req.data {
        IndicatorDataRef::Slice { values } => values.len(),
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "multi_length_stochastic_average".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "multi_length_stochastic_average",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source =
                get_enum_param("multi_length_stochastic_average", params, "source", "close")?;
            let length = get_usize_param("multi_length_stochastic_average", params, "length", 14)?;
            let presmooth =
                get_usize_param("multi_length_stochastic_average", params, "presmooth", 10)?;
            let premethod = get_enum_param(
                "multi_length_stochastic_average",
                params,
                "premethod",
                "sma",
            )?;
            let postsmooth =
                get_usize_param("multi_length_stochastic_average", params, "postsmooth", 10)?;
            let postmethod = get_enum_param(
                "multi_length_stochastic_average",
                params,
                "postmethod",
                "sma",
            )?;
            let data = match req.data {
                IndicatorDataRef::Slice { values } => values,
                IndicatorDataRef::Candles { candles, .. } => source_type(candles, &source),
                _ => unreachable!(),
            };
            let input = MultiLengthStochasticAverageInput::from_slice(
                data,
                MultiLengthStochasticAverageParams {
                    length: Some(length),
                    presmooth: Some(presmooth),
                    premethod: Some(premethod),
                    postsmooth: Some(postsmooth),
                    postmethod: Some(postmethod),
                },
            );
            let out = multi_length_stochastic_average_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "multi_length_stochastic_average".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_hull_butterfly_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Slice { values } => values.len(),
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "hull_butterfly_oscillator".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("oscillator") {
        HullButterflyOscillatorOutputField::Oscillator
    } else if output_id.eq_ignore_ascii_case("cumulative_mean") {
        HullButterflyOscillatorOutputField::CumulativeMean
    } else if output_id.eq_ignore_ascii_case("signal") {
        HullButterflyOscillatorOutputField::Signal
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "hull_butterfly_oscillator".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "hull_butterfly_oscillator",
        output_id,
        req.combos,
        data_len,
        |params, row| {
            let source = get_enum_param("hull_butterfly_oscillator", params, "source", "close")?;
            let length = get_usize_param("hull_butterfly_oscillator", params, "length", 14)?;
            let mult = get_f64_param("hull_butterfly_oscillator", params, "mult", 2.0)?;
            let data = match req.data {
                IndicatorDataRef::Slice { values } => values,
                IndicatorDataRef::Candles { candles, .. } => source_type(candles, &source),
                _ => unreachable!(),
            };
            let input = HullButterflyOscillatorInput::from_slice(
                data,
                HullButterflyOscillatorParams {
                    length: Some(length),
                    mult: Some(mult),
                },
            );
            hull_butterfly_oscillator_output_into_slice(row, &input, kernel, field).map_err(
                |e| IndicatorDispatchError::ComputeFailed {
                    indicator: "hull_butterfly_oscillator".to_string(),
                    details: e.to_string(),
                },
            )?;
            Ok(())
        },
    )
}

fn compute_fibonacci_trailing_stop_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("fibonacci_trailing_stop", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "fibonacci_trailing_stop",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let left_bars = get_usize_param("fibonacci_trailing_stop", params, "left_bars", 20)?;
            let right_bars = get_usize_param("fibonacci_trailing_stop", params, "right_bars", 1)?;
            let level = get_f64_param("fibonacci_trailing_stop", params, "level", -0.382)?;
            let trigger = get_enum_param("fibonacci_trailing_stop", params, "trigger", "close")?;
            let input = FibonacciTrailingStopInput::from_slices(
                high,
                low,
                close,
                FibonacciTrailingStopParams {
                    left_bars: Some(left_bars),
                    right_bars: Some(right_bars),
                    level: Some(level),
                    trigger: Some(trigger),
                },
            );
            let out = fibonacci_trailing_stop_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "fibonacci_trailing_stop".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("trailing_stop")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.trailing_stop);
            }
            if output_id.eq_ignore_ascii_case("long_stop") {
                return Ok(out.long_stop);
            }
            if output_id.eq_ignore_ascii_case("short_stop") {
                return Ok(out.short_stop);
            }
            if output_id.eq_ignore_ascii_case("direction") {
                return Ok(out.direction);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "fibonacci_trailing_stop".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_fibonacci_entry_bands_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("fibonacci_entry_bands", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "fibonacci_entry_bands",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let source = get_enum_param("fibonacci_entry_bands", params, "source", "hlc3")?;
            let length = get_usize_param("fibonacci_entry_bands", params, "length", 21)?;
            let atr_length = get_usize_param("fibonacci_entry_bands", params, "atr_length", 14)?;
            let use_atr = get_bool_param("fibonacci_entry_bands", params, "use_atr", true)?;
            let tp_aggressiveness =
                get_enum_param("fibonacci_entry_bands", params, "tp_aggressiveness", "low")?;
            let input = FibonacciEntryBandsInput::from_slices(
                open,
                high,
                low,
                close,
                FibonacciEntryBandsParams {
                    source: Some(source),
                    length: Some(length),
                    atr_length: Some(atr_length),
                    use_atr: Some(use_atr),
                    tp_aggressiveness: Some(tp_aggressiveness),
                },
            );
            let out = fibonacci_entry_bands_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "fibonacci_entry_bands".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("middle") || output_id.eq_ignore_ascii_case("basis") {
                return Ok(out.basis);
            }
            if output_id.eq_ignore_ascii_case("trend") {
                return Ok(out.trend);
            }
            if output_id.eq_ignore_ascii_case("upper_0618") {
                return Ok(out.upper_0618);
            }
            if output_id.eq_ignore_ascii_case("upper_1000") {
                return Ok(out.upper_1000);
            }
            if output_id.eq_ignore_ascii_case("upper_1618") {
                return Ok(out.upper_1618);
            }
            if output_id.eq_ignore_ascii_case("upper_2618") {
                return Ok(out.upper_2618);
            }
            if output_id.eq_ignore_ascii_case("lower_0618") {
                return Ok(out.lower_0618);
            }
            if output_id.eq_ignore_ascii_case("lower_1000") {
                return Ok(out.lower_1000);
            }
            if output_id.eq_ignore_ascii_case("lower_1618") {
                return Ok(out.lower_1618);
            }
            if output_id.eq_ignore_ascii_case("lower_2618") {
                return Ok(out.lower_2618);
            }
            if output_id.eq_ignore_ascii_case("tp_long_band") {
                return Ok(out.tp_long_band);
            }
            if output_id.eq_ignore_ascii_case("tp_short_band") {
                return Ok(out.tp_short_band);
            }
            if output_id.eq_ignore_ascii_case("go_long")
                || output_id.eq_ignore_ascii_case("long_entry")
            {
                return Ok(out.long_entry);
            }
            if output_id.eq_ignore_ascii_case("go_short")
                || output_id.eq_ignore_ascii_case("short_entry")
            {
                return Ok(out.short_entry);
            }
            if output_id.eq_ignore_ascii_case("rejection_long") {
                return Ok(out.rejection_long);
            }
            if output_id.eq_ignore_ascii_case("rejection_short") {
                return Ok(out.rejection_short);
            }
            if output_id.eq_ignore_ascii_case("long_bounce") {
                return Ok(out.long_bounce);
            }
            if output_id.eq_ignore_ascii_case("short_bounce") {
                return Ok(out.short_bounce);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "fibonacci_entry_bands".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_volume_energy_reservoirs_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (_, high, low, close, volume) =
        extract_ohlcv_full_input("volume_energy_reservoirs", req.data)?;
    let kernel = req.kernel.to_non_batch();
    let field = match output_id {
        "momentum" | "value" => VolumeEnergyReservoirsOutputField::Momentum,
        "reservoir" => VolumeEnergyReservoirsOutputField::Reservoir,
        "squeeze_active" => VolumeEnergyReservoirsOutputField::SqueezeActive,
        "squeeze_start" => VolumeEnergyReservoirsOutputField::SqueezeStart,
        "range_high" => VolumeEnergyReservoirsOutputField::RangeHigh,
        "range_low" => VolumeEnergyReservoirsOutputField::RangeLow,
        _ => {
            return Err(IndicatorDispatchError::UnknownOutput {
                indicator: "volume_energy_reservoirs".to_string(),
                output: output_id.to_string(),
            })
        }
    };
    let rows = req.combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "volume_energy_reservoirs".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let length = get_usize_param("volume_energy_reservoirs", combo.params, "length", 20)?;
        let sensitivity =
            get_f64_param("volume_energy_reservoirs", combo.params, "sensitivity", 1.5)?;
        let input = VolumeEnergyReservoirsInput::from_slices(
            high,
            low,
            close,
            volume,
            VolumeEnergyReservoirsParams {
                length: Some(length),
                sensitivity: Some(sensitivity),
            },
        );
        let start = row * cols;
        let end = start + cols;
        volume_energy_reservoirs_output_into_slice(&mut matrix[start..end], &input, kernel, field)
            .map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "volume_energy_reservoirs".to_string(),
                details: e.to_string(),
            })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
}

fn compute_neighboring_trailing_stop_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("neighboring_trailing_stop", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "neighboring_trailing_stop",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let buffer_size =
                get_usize_param("neighboring_trailing_stop", params, "buffer_size", 200)?;
            let k = get_usize_param("neighboring_trailing_stop", params, "k", 50)?;
            let percentile =
                get_f64_param("neighboring_trailing_stop", params, "percentile", 90.0)?;
            let smooth = get_usize_param("neighboring_trailing_stop", params, "smooth", 5)?;
            let input = NeighboringTrailingStopInput::from_slices(
                high,
                low,
                close,
                NeighboringTrailingStopParams {
                    buffer_size: Some(buffer_size),
                    k: Some(k),
                    percentile: Some(percentile),
                    smooth: Some(smooth),
                },
            );
            let out = neighboring_trailing_stop_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "neighboring_trailing_stop".to_string(),
                    details: e.to_string(),
                }
            })?;
            match output_id {
                "trailing_stop" | "value" => Ok(out.trailing_stop),
                "bullish_band" => Ok(out.bullish_band),
                "bearish_band" => Ok(out.bearish_band),
                "direction" => Ok(out.direction),
                "discovery_bull" => Ok(out.discovery_bull),
                "discovery_bear" => Ok(out.discovery_bear),
                _ => Err(IndicatorDispatchError::UnknownOutput {
                    indicator: "neighboring_trailing_stop".to_string(),
                    output: output_id.to_string(),
                }),
            }
        },
    )
}

fn compute_grover_llorens_cycle_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("grover_llorens_cycle_oscillator", output_id)?;
    let (open, high, low, close) =
        extract_ohlc_full_input("grover_llorens_cycle_oscillator", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "grover_llorens_cycle_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param("grover_llorens_cycle_oscillator", params, "length", 100)?;
            let mult = get_f64_param("grover_llorens_cycle_oscillator", params, "mult", 10.0)?;
            let source = match find_param(params, "source") {
                Some(ParamValue::EnumString(v)) => (*v).to_string(),
                Some(_) => {
                    return Err(IndicatorDispatchError::InvalidParam {
                        indicator: "grover_llorens_cycle_oscillator".to_string(),
                        key: "source".to_string(),
                        reason: "expected string".to_string(),
                    });
                }
                None => "close".to_string(),
            };
            let smooth = get_bool_param("grover_llorens_cycle_oscillator", params, "smooth", true)?;
            let rsi_period =
                get_usize_param("grover_llorens_cycle_oscillator", params, "rsi_period", 20)?;
            let input = GroverLlorensCycleOscillatorInput::from_slices(
                open,
                high,
                low,
                close,
                GroverLlorensCycleOscillatorParams {
                    length: Some(length),
                    mult: Some(mult),
                    source: Some(source),
                    smooth: Some(smooth),
                    rsi_period: Some(rsi_period),
                },
            );
            let out = grover_llorens_cycle_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "grover_llorens_cycle_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_ehlers_autocorrelation_periodogram_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("ehlers_autocorrelation_periodogram", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ehlers_autocorrelation_periodogram",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let min_period = get_usize_param(
                "ehlers_autocorrelation_periodogram",
                params,
                "min_period",
                8,
            )?;
            let max_period = get_usize_param(
                "ehlers_autocorrelation_periodogram",
                params,
                "max_period",
                48,
            )?;
            let avg_length = get_usize_param(
                "ehlers_autocorrelation_periodogram",
                params,
                "avg_length",
                3,
            )?;
            let enhance = get_bool_param(
                "ehlers_autocorrelation_periodogram",
                params,
                "enhance",
                true,
            )?;
            let input = EhlersAutocorrelationPeriodogramInput::from_slice(
                data,
                EhlersAutocorrelationPeriodogramParams {
                    min_period: Some(min_period),
                    max_period: Some(max_period),
                    avg_length: Some(avg_length),
                    enhance: Some(enhance),
                },
            );
            let out =
                ehlers_autocorrelation_periodogram_with_kernel(&input, kernel).map_err(|e| {
                    IndicatorDispatchError::ComputeFailed {
                        indicator: "ehlers_autocorrelation_periodogram".to_string(),
                        details: e.to_string(),
                    }
                })?;
            if output_id.eq_ignore_ascii_case("dominant_cycle")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.dominant_cycle);
            }
            if output_id.eq_ignore_ascii_case("normalized_power") {
                return Ok(out.normalized_power);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "ehlers_autocorrelation_periodogram".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_ehlers_linear_extrapolation_predictor_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("ehlers_linear_extrapolation_predictor", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ehlers_linear_extrapolation_predictor",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let high_pass_length = get_usize_param(
                "ehlers_linear_extrapolation_predictor",
                params,
                "high_pass_length",
                125,
            )?;
            let low_pass_length = get_usize_param(
                "ehlers_linear_extrapolation_predictor",
                params,
                "low_pass_length",
                12,
            )?;
            let gain = get_f64_param("ehlers_linear_extrapolation_predictor", params, "gain", 0.7)?;
            let bars_forward = get_usize_param(
                "ehlers_linear_extrapolation_predictor",
                params,
                "bars_forward",
                5,
            )?;
            let signal_mode = get_enum_param(
                "ehlers_linear_extrapolation_predictor",
                params,
                "signal_mode",
                "predict_filter_crosses",
            )?;
            let input = EhlersLinearExtrapolationPredictorInput::from_slice(
                data,
                EhlersLinearExtrapolationPredictorParams {
                    high_pass_length: Some(high_pass_length),
                    low_pass_length: Some(low_pass_length),
                    gain: Some(gain),
                    bars_forward: Some(bars_forward),
                    signal_mode: Some(signal_mode),
                },
            );
            let out =
                ehlers_linear_extrapolation_predictor_with_kernel(&input, kernel).map_err(|e| {
                    IndicatorDispatchError::ComputeFailed {
                        indicator: "ehlers_linear_extrapolation_predictor".to_string(),
                        details: e.to_string(),
                    }
                })?;
            if output_id.eq_ignore_ascii_case("prediction")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.prediction);
            }
            if output_id.eq_ignore_ascii_case("filter") {
                return Ok(out.filter);
            }
            if output_id.eq_ignore_ascii_case("state") {
                return Ok(out.state);
            }
            if output_id.eq_ignore_ascii_case("go_long") {
                return Ok(out.go_long);
            }
            if output_id.eq_ignore_ascii_case("go_short") {
                return Ok(out.go_short);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "ehlers_linear_extrapolation_predictor".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_decisionpoint_breadth_swenlin_trading_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output(
        "decisionpoint_breadth_swenlin_trading_oscillator",
        output_id,
    )?;
    let (advancing, declining) =
        extract_high_low_input("decisionpoint_breadth_swenlin_trading_oscillator", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "decisionpoint_breadth_swenlin_trading_oscillator",
        output_id,
        req.combos,
        advancing.len(),
        |_params, row| {
            let input = DecisionPointBreadthSwenlinTradingOscillatorInput::from_slices(
                advancing,
                declining,
                DecisionPointBreadthSwenlinTradingOscillatorParams,
            );
            decisionpoint_breadth_swenlin_trading_oscillator_into_slice(row, &input, kernel)
                .map_err(|e| IndicatorDispatchError::ComputeFailed {
                    indicator: "decisionpoint_breadth_swenlin_trading_oscillator".to_string(),
                    details: e.to_string(),
                })?;
            Ok(())
        },
    )
}

fn compute_velocity_acceleration_indicator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("velocity_acceleration_indicator", output_id)?;
    let data_len = match req.data {
        IndicatorDataRef::Slice { values } => values.len(),
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("hlcc4")).len()
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "velocity_acceleration_indicator".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "velocity_acceleration_indicator",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source =
                get_enum_param("velocity_acceleration_indicator", params, "source", "hlcc4")?;
            let length = get_usize_param("velocity_acceleration_indicator", params, "length", 21)?;
            let smooth_length = get_usize_param(
                "velocity_acceleration_indicator",
                params,
                "smooth_length",
                5,
            )?;
            let data = match req.data {
                IndicatorDataRef::Slice { values } => values,
                IndicatorDataRef::Candles { candles, .. } => source_type(candles, &source),
                _ => unreachable!(),
            };
            let input = VelocityAccelerationIndicatorInput::from_slice(
                data,
                VelocityAccelerationIndicatorParams {
                    length: Some(length),
                    smooth_length: Some(smooth_length),
                },
            );
            let out = velocity_acceleration_indicator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "velocity_acceleration_indicator".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_normalized_resonator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Slice { values } => values.len(),
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("hl2")).len()
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "normalized_resonator".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "normalized_resonator",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("normalized_resonator", params, "source", "hl2")?;
            let period = get_usize_param("normalized_resonator", params, "period", 100)?;
            let delta = get_f64_param("normalized_resonator", params, "delta", 0.5)?;
            let lookback_mult =
                get_f64_param("normalized_resonator", params, "lookback_mult", 1.0)?;
            let signal_length =
                get_usize_param("normalized_resonator", params, "signal_length", 9)?;
            let data = match req.data {
                IndicatorDataRef::Slice { values } => values,
                IndicatorDataRef::Candles { candles, .. } => source_type(candles, &source),
                _ => unreachable!(),
            };
            let input = NormalizedResonatorInput::from_slice(
                data,
                NormalizedResonatorParams {
                    period: Some(period),
                    delta: Some(delta),
                    lookback_mult: Some(lookback_mult),
                    signal_length: Some(signal_length),
                },
            );
            let out = normalized_resonator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "normalized_resonator".to_string(),
                    details: e.to_string(),
                }
            })?;
            match output_id {
                "oscillator" => Ok(out.oscillator),
                "signal" => Ok(out.signal),
                _ => Err(IndicatorDispatchError::UnknownOutput {
                    indicator: "normalized_resonator".to_string(),
                    output: output_id.to_string(),
                }),
            }
        },
    )
}

fn compute_monotonicity_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Slice { values } => values.len(),
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "monotonicity_index".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "monotonicity_index",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("monotonicity_index", params, "source", "close")?;
            let length = get_usize_param("monotonicity_index", params, "length", 20)?;
            let mode = get_enum_param("monotonicity_index", params, "mode", "efficiency")?;
            let index_smooth = get_usize_param("monotonicity_index", params, "index_smooth", 5)?;
            let mode = MonotonicityIndexMode::parse(&mode).ok_or_else(|| {
                IndicatorDispatchError::InvalidParam {
                    indicator: "monotonicity_index".to_string(),
                    key: "mode".to_string(),
                    reason: format!("invalid mode: {mode}"),
                }
            })?;
            let data = match req.data {
                IndicatorDataRef::Slice { values } => values,
                IndicatorDataRef::Candles { candles, .. } => source_type(candles, &source),
                _ => unreachable!(),
            };
            let input = MonotonicityIndexInput::from_slice(
                data,
                MonotonicityIndexParams {
                    length: Some(length),
                    mode: Some(mode),
                    index_smooth: Some(index_smooth),
                },
            );
            let out = monotonicity_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "monotonicity_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            match output_id {
                "index" => Ok(out.index),
                "cumulative_mean" => Ok(out.cumulative_mean),
                "upper_bound" => Ok(out.upper_bound),
                _ => Err(IndicatorDispatchError::UnknownOutput {
                    indicator: "monotonicity_index".to_string(),
                    output: output_id.to_string(),
                }),
            }
        },
    )
}

fn compute_half_causal_estimator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    if output_id != "estimate" && output_id != "expected_value" {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "half_causal_estimator".to_string(),
            output: output_id.to_string(),
        });
    }

    let data_len = match req.data {
        IndicatorDataRef::Slice { values } => values.len(),
        IndicatorDataRef::Candles { candles, .. } => candles.close.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "half_causal_estimator".to_string(),
                input: IndicatorInputKind::Candles,
            });
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "half_causal_estimator",
        output_id,
        req.combos,
        data_len,
        |params| {
            let slots_per_day = get_usize_param_with_aliases(
                "half_causal_estimator",
                params,
                &["slots_per_day"],
                0,
            )?;
            let data_period = get_usize_param("half_causal_estimator", params, "data_period", 5)?;
            let filter_length =
                get_usize_param("half_causal_estimator", params, "filter_length", 20)?;
            let kernel_width =
                get_f64_param("half_causal_estimator", params, "kernel_width", 20.0)?;
            let maximum_confidence_adjust = get_f64_param(
                "half_causal_estimator",
                params,
                "maximum_confidence_adjust",
                100.0,
            )?;
            let extra_smoothing =
                get_usize_param("half_causal_estimator", params, "extra_smoothing", 0)?;
            let enable_expected_value = get_bool_param(
                "half_causal_estimator",
                params,
                "enable_expected_value",
                false,
            )?;
            let source = get_enum_param("half_causal_estimator", params, "source", "volume")?;
            let kernel_type = match get_enum_param(
                "half_causal_estimator",
                params,
                "kernel_type",
                "epanechnikov",
            )?
            .to_ascii_lowercase()
            .as_str()
            {
                "gaussian" => HalfCausalEstimatorKernelType::Gaussian,
                "epanechnikov" => HalfCausalEstimatorKernelType::Epanechnikov,
                "triangular" => HalfCausalEstimatorKernelType::Triangular,
                "sinc" => HalfCausalEstimatorKernelType::Sinc,
                other => {
                    return Err(IndicatorDispatchError::InvalidParam {
                        indicator: "half_causal_estimator".to_string(),
                        key: "kernel_type".to_string(),
                        reason: format!("unsupported value '{other}'"),
                    })
                }
            };
            let confidence_adjust = match get_enum_param(
                "half_causal_estimator",
                params,
                "confidence_adjust",
                "symmetric",
            )?
            .to_ascii_lowercase()
            .as_str()
            {
                "symmetric" => HalfCausalEstimatorConfidenceAdjust::Symmetric,
                "linear" => HalfCausalEstimatorConfidenceAdjust::Linear,
                "none" => HalfCausalEstimatorConfidenceAdjust::None,
                other => {
                    return Err(IndicatorDispatchError::InvalidParam {
                        indicator: "half_causal_estimator".to_string(),
                        key: "confidence_adjust".to_string(),
                        reason: format!("unsupported value '{other}'"),
                    })
                }
            };

            let indicator_params = HalfCausalEstimatorParams {
                slots_per_day: if slots_per_day == 0 {
                    None
                } else {
                    Some(slots_per_day)
                },
                data_period: Some(data_period),
                filter_length: Some(filter_length),
                kernel_width: Some(kernel_width),
                kernel_type: Some(kernel_type),
                confidence_adjust: Some(confidence_adjust),
                maximum_confidence_adjust: Some(maximum_confidence_adjust),
                enable_expected_value: Some(enable_expected_value),
                extra_smoothing: Some(extra_smoothing),
            };

            let out = match req.data {
                IndicatorDataRef::Slice { values } => {
                    let input = HalfCausalEstimatorInput::from_slice(values, indicator_params);
                    half_causal_estimator_with_kernel(&input, kernel)
                }
                IndicatorDataRef::Candles { candles, .. } => {
                    let input =
                        HalfCausalEstimatorInput::from_candles(candles, &source, indicator_params);
                    half_causal_estimator_with_kernel(&input, kernel)
                }
                _ => unreachable!(),
            }
            .map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "half_causal_estimator".to_string(),
                details: e.to_string(),
            })?;

            Ok(match output_id {
                "estimate" => out.estimate,
                "expected_value" => out.expected_value,
                _ => unreachable!(),
            })
        },
    )
}

fn compute_historical_volatility_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("historical_volatility", output_id)?;
    let data = extract_slice_input("historical_volatility", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "historical_volatility",
        output_id,
        req.combos,
        data.len(),
        |params, row| {
            let lookback = get_usize_param("historical_volatility", params, "lookback", 20)?;
            let annualization_days =
                get_f64_param("historical_volatility", params, "annualization_days", 250.0)?;
            let input = HistoricalVolatilityInput::from_slice(
                data,
                HistoricalVolatilityParams {
                    lookback: Some(lookback),
                    annualization_days: Some(annualization_days),
                },
            );
            historical_volatility_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "historical_volatility".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(())
        },
    )
}

fn compute_historical_volatility_percentile_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("historical_volatility_percentile", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "historical_volatility_percentile",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let length = get_usize_param("historical_volatility_percentile", params, "length", 20)?;
            let annual_length = get_usize_param(
                "historical_volatility_percentile",
                params,
                "annual_length",
                252,
            )?;
            let input = HistoricalVolatilityPercentileInput::from_slice(
                data,
                HistoricalVolatilityPercentileParams {
                    length: Some(length),
                    annual_length: Some(annual_length),
                },
            );
            let out =
                historical_volatility_percentile_with_kernel(&input, kernel).map_err(|e| {
                    IndicatorDispatchError::ComputeFailed {
                        indicator: "historical_volatility_percentile".to_string(),
                        details: e.to_string(),
                    }
                })?;
            if output_id.eq_ignore_ascii_case("hvp") {
                return Ok(out.hvp);
            }
            if output_id.eq_ignore_ascii_case("hvp_sma") {
                return Ok(out.hvp_sma);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "historical_volatility_percentile".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_volatility_ratio_adaptive_rsx_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("volatility_ratio_adaptive_rsx", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "volatility_ratio_adaptive_rsx",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let period = get_usize_param("volatility_ratio_adaptive_rsx", params, "period", 14)?;
            let speed = get_f64_param("volatility_ratio_adaptive_rsx", params, "speed", 0.5)?;
            let input = VolatilityRatioAdaptiveRsxInput::from_slice(
                data,
                VolatilityRatioAdaptiveRsxParams {
                    period: Some(period),
                    speed: Some(speed),
                },
            );
            let out = volatility_ratio_adaptive_rsx_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "volatility_ratio_adaptive_rsx".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("line") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.line);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "volatility_ratio_adaptive_rsx".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_on_balance_volume_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (close, volume) =
        extract_close_volume_input("on_balance_volume_oscillator", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "on_balance_volume_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let obv_length =
                get_usize_param("on_balance_volume_oscillator", params, "obv_length", 20)?;
            let ema_length =
                get_usize_param("on_balance_volume_oscillator", params, "ema_length", 9)?;
            let input = OnBalanceVolumeOscillatorInput::from_slices(
                close,
                volume,
                OnBalanceVolumeOscillatorParams {
                    obv_length: Some(obv_length),
                    ema_length: Some(ema_length),
                },
            );
            let out = on_balance_volume_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "on_balance_volume_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("line") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.line);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "on_balance_volume_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_twiggs_money_flow_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close, volume) = extract_hlcv_input("twiggs_money_flow", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "twiggs_money_flow",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param("twiggs_money_flow", params, "length", 21)?;
            let smoothing_length =
                get_usize_param("twiggs_money_flow", params, "smoothing_length", 4)?;
            let ma_type = get_enum_param("twiggs_money_flow", params, "ma_type", "ema")?;
            let input = TwiggsMoneyFlowInput::from_slices(
                high,
                low,
                close,
                volume,
                TwiggsMoneyFlowParams {
                    length: Some(length),
                    smoothing_length: Some(smoothing_length),
                    ma_type: Some(ma_type),
                },
            );
            let out = twiggs_money_flow_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "twiggs_money_flow".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("tmf") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.tmf);
            }
            if output_id.eq_ignore_ascii_case("smoothed") {
                return Ok(out.smoothed);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "twiggs_money_flow".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_parkinson_volatility_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low) = extract_high_low_input("parkinson_volatility", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "parkinson_volatility",
        output_id,
        req.combos,
        high.len(),
        |params| {
            let period = get_usize_param("parkinson_volatility", params, "period", 10)?;
            let input = ParkinsonVolatilityInput::from_slices(
                high,
                low,
                ParkinsonVolatilityParams {
                    period: Some(period),
                },
            );
            let out = parkinson_volatility_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "parkinson_volatility".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("volatility")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.volatility);
            }
            if output_id.eq_ignore_ascii_case("variance") {
                return Ok(out.variance);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "parkinson_volatility".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_l2_ehlers_signal_to_noise_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("l2_ehlers_signal_to_noise", output_id)?;
    let data_len = match req.data {
        IndicatorDataRef::Slice { values } => values.len(),
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("hl2")).len()
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "l2_ehlers_signal_to_noise".to_string(),
                input: IndicatorInputKind::Candles,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "l2_ehlers_signal_to_noise",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("l2_ehlers_signal_to_noise", params, "source", "hl2")?;
            let smooth_period =
                get_usize_param("l2_ehlers_signal_to_noise", params, "smooth_period", 10)?;
            let src = match req.data {
                IndicatorDataRef::Slice { values } => values,
                IndicatorDataRef::Candles { candles, .. } => source_type(candles, &source),
                _ => unreachable!(),
            };
            let (high, low) = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    (candles.high.as_slice(), candles.low.as_slice())
                }
                IndicatorDataRef::Ohlc { high, low, .. } => (high, low),
                IndicatorDataRef::Ohlcv { high, low, .. } => (high, low),
                _ => {
                    return Err(IndicatorDispatchError::MissingRequiredInput {
                        indicator: "l2_ehlers_signal_to_noise".to_string(),
                        input: IndicatorInputKind::Candles,
                    })
                }
            };
            let input = L2EhlersSignalToNoiseInput::from_slices(
                src,
                high,
                low,
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
            Ok(out.values)
        },
    )
}

fn compute_cycle_channel_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "cycle_channel_oscillator".to_string(),
                input: IndicatorInputKind::Candles,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("fast") || output_id.eq_ignore_ascii_case("value")
    {
        CycleChannelOscillatorOutputField::Fast
    } else if output_id.eq_ignore_ascii_case("slow") {
        CycleChannelOscillatorOutputField::Slow
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "cycle_channel_oscillator".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "cycle_channel_oscillator",
        output_id,
        req.combos,
        data_len,
        |params, row| {
            let source = get_enum_param("cycle_channel_oscillator", params, "source", "close")?;
            let short_cycle_length =
                get_usize_param("cycle_channel_oscillator", params, "short_cycle_length", 10)?;
            let medium_cycle_length = get_usize_param(
                "cycle_channel_oscillator",
                params,
                "medium_cycle_length",
                30,
            )?;
            let short_multiplier =
                get_f64_param("cycle_channel_oscillator", params, "short_multiplier", 1.0)?;
            let medium_multiplier =
                get_f64_param("cycle_channel_oscillator", params, "medium_multiplier", 3.0)?;
            let (src, high, low, close) = match req.data {
                IndicatorDataRef::Candles { candles, .. } => (
                    source_type(candles, &source),
                    candles.high.as_slice(),
                    candles.low.as_slice(),
                    candles.close.as_slice(),
                ),
                _ => unreachable!(),
            };
            let input = CycleChannelOscillatorInput::from_slices(
                src,
                high,
                low,
                close,
                CycleChannelOscillatorParams {
                    short_cycle_length: Some(short_cycle_length),
                    medium_cycle_length: Some(medium_cycle_length),
                    short_multiplier: Some(short_multiplier),
                    medium_multiplier: Some(medium_multiplier),
                },
            );
            cycle_channel_oscillator_output_into_slice(row, &input, kernel, field).map_err(
                |e| IndicatorDispatchError::ComputeFailed {
                    indicator: "cycle_channel_oscillator".to_string(),
                    details: e.to_string(),
                },
            )?;
            Ok(())
        },
    )
}

fn compute_andean_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, _high, _low, close) = extract_ohlc_full_input("andean_oscillator", req.data)?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("bull") {
        AndeanOscillatorOutputField::Bull
    } else if output_id.eq_ignore_ascii_case("bear") {
        AndeanOscillatorOutputField::Bear
    } else if output_id.eq_ignore_ascii_case("signal") {
        AndeanOscillatorOutputField::Signal
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "andean_oscillator".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "andean_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
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
            andean_oscillator_output_into_slice(row, &input, kernel, field).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "andean_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(())
        },
    )
}

fn compute_daily_factor_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("daily_factor", req.data)?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("value") {
        DailyFactorOutputField::Value
    } else if output_id.eq_ignore_ascii_case("ema") {
        DailyFactorOutputField::Ema
    } else if output_id.eq_ignore_ascii_case("signal") {
        DailyFactorOutputField::Signal
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "daily_factor".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "daily_factor",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
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
            daily_factor_output_into_slice(row, &input, kernel, field).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "daily_factor".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(())
        },
    )
}

fn compute_ehlers_adaptive_cyber_cycle_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Slice { values } => values.len(),
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("hl2")).len()
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "ehlers_adaptive_cyber_cycle".to_string(),
                input: IndicatorInputKind::Candles,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ehlers_adaptive_cyber_cycle",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("ehlers_adaptive_cyber_cycle", params, "source", "hl2")?;
            let alpha = get_f64_param("ehlers_adaptive_cyber_cycle", params, "alpha", 0.07)?;
            let data = match req.data {
                IndicatorDataRef::Slice { values } => values,
                IndicatorDataRef::Candles { candles, .. } => source_type(candles, &source),
                _ => unreachable!(),
            };
            let input = EhlersAdaptiveCyberCycleInput::from_slice(
                data,
                EhlersAdaptiveCyberCycleParams { alpha: Some(alpha) },
            );
            let out = ehlers_adaptive_cyber_cycle_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "ehlers_adaptive_cyber_cycle".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("cycle") || output_id.eq_ignore_ascii_case("value") {
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
    let data_len = match req.data {
        IndicatorDataRef::Slice { values } => values.len(),
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("hl2")).len()
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "ehlers_simple_cycle_indicator".to_string(),
                input: IndicatorInputKind::Candles,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ehlers_simple_cycle_indicator",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("ehlers_simple_cycle_indicator", params, "source", "hl2")?;
            let alpha = get_f64_param("ehlers_simple_cycle_indicator", params, "alpha", 0.07)?;
            let data = match req.data {
                IndicatorDataRef::Slice { values } => values,
                IndicatorDataRef::Candles { candles, .. } => source_type(candles, &source),
                _ => unreachable!(),
            };
            let input = EhlersSimpleCycleIndicatorInput::from_slice(
                data,
                EhlersSimpleCycleIndicatorParams { alpha: Some(alpha) },
            );
            let out = ehlers_simple_cycle_indicator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "ehlers_simple_cycle_indicator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("cycle") || output_id.eq_ignore_ascii_case("value") {
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
    expect_value_output("l1_ehlers_phasor", output_id)?;
    let data = extract_slice_input("l1_ehlers_phasor", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "l1_ehlers_phasor",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let domestic_cycle_length =
                get_usize_param("l1_ehlers_phasor", params, "domestic_cycle_length", 15)?;
            let input = L1EhlersPhasorInput::from_slice(
                data,
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
            Ok(out.values)
        },
    )
}

fn compute_ehlers_smoothed_adaptive_momentum_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("ehlers_smoothed_adaptive_momentum", output_id)?;
    let data_len = match req.data {
        IndicatorDataRef::Slice { values } => values.len(),
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("hl2")).len()
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "ehlers_smoothed_adaptive_momentum".to_string(),
                input: IndicatorInputKind::Candles,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ehlers_smoothed_adaptive_momentum",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source =
                get_enum_param("ehlers_smoothed_adaptive_momentum", params, "source", "hl2")?;
            let alpha = get_f64_param("ehlers_smoothed_adaptive_momentum", params, "alpha", 0.07)?;
            let cutoff = get_f64_param("ehlers_smoothed_adaptive_momentum", params, "cutoff", 8.0)?;
            let data = match req.data {
                IndicatorDataRef::Slice { values } => values,
                IndicatorDataRef::Candles { candles, .. } => source_type(candles, &source),
                _ => unreachable!(),
            };
            let input = EhlersSmoothedAdaptiveMomentumInput::from_slice(
                data,
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
            Ok(out.values)
        },
    )
}

fn compute_ewma_volatility_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("ewma_volatility", output_id)?;
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
            Ok(out.values)
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
            if output_id.eq_ignore_ascii_case("high") {
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
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        IndicatorDataRef::CloseVolume { close, .. } => close.len(),
        IndicatorDataRef::Ohlcv { close, .. } => close.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "price_moving_average_ratio_percentile".to_string(),
                input: IndicatorInputKind::CloseVolume,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "price_moving_average_ratio_percentile",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param(
                "price_moving_average_ratio_percentile",
                params,
                "source",
                "close",
            )?;
            let ma_length = get_usize_param(
                "price_moving_average_ratio_percentile",
                params,
                "ma_length",
                20,
            )?;
            let ma_type = get_enum_param(
                "price_moving_average_ratio_percentile",
                params,
                "ma_type",
                "sma",
            )?
            .parse::<PriceMovingAverageRatioPercentileMaType>()
            .map_err(|e| IndicatorDispatchError::InvalidParam {
                indicator: "price_moving_average_ratio_percentile".to_string(),
                key: "ma_type".to_string(),
                reason: e,
            })?;
            let pmarp_lookback = get_usize_param(
                "price_moving_average_ratio_percentile",
                params,
                "pmarp_lookback",
                350,
            )?;
            let signal_ma_length = get_usize_param(
                "price_moving_average_ratio_percentile",
                params,
                "signal_ma_length",
                20,
            )?;
            let signal_ma_type = get_enum_param(
                "price_moving_average_ratio_percentile",
                params,
                "signal_ma_type",
                "sma",
            )?
            .parse::<PriceMovingAverageRatioPercentileMaType>()
            .map_err(|e| IndicatorDispatchError::InvalidParam {
                indicator: "price_moving_average_ratio_percentile".to_string(),
                key: "signal_ma_type".to_string(),
                reason: e,
            })?;
            let line_mode = get_enum_param(
                "price_moving_average_ratio_percentile",
                params,
                "line_mode",
                "pmar",
            )?
            .parse::<PriceMovingAverageRatioPercentileLineMode>()
            .map_err(|e| IndicatorDispatchError::InvalidParam {
                indicator: "price_moving_average_ratio_percentile".to_string(),
                key: "line_mode".to_string(),
                reason: e,
            })?;
            let (price, volume) = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    (source_type(candles, &source), candles.volume.as_slice())
                }
                IndicatorDataRef::CloseVolume { close, volume } => (close, volume),
                IndicatorDataRef::Ohlcv {
                    open,
                    high,
                    low,
                    close,
                    volume,
                } => {
                    let price = match source.to_ascii_lowercase().as_str() {
                        "open" => open,
                        "high" => high,
                        "low" => low,
                        _ => close,
                    };
                    (price, volume)
                }
                _ => unreachable!(),
            };
            let input = PriceMovingAverageRatioPercentileInput::from_slices(
                price,
                volume,
                PriceMovingAverageRatioPercentileParams {
                    ma_length: Some(ma_length),
                    ma_type: Some(ma_type),
                    pmarp_lookback: Some(pmarp_lookback),
                    signal_ma_length: Some(signal_ma_length),
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
            if output_id.eq_ignore_ascii_case("pmar") {
                return Ok(out.pmar);
            }
            if output_id.eq_ignore_ascii_case("pmarp") {
                return Ok(out.pmarp);
            }
            if output_id.eq_ignore_ascii_case("plotline") || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.plotline);
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

fn compute_trend_trigger_factor_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("trend_trigger_factor", output_id)?;
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
            Ok(out.values)
        },
    )
}

fn compute_mesa_stochastic_multi_length_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Slice { values } => values.len(),
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "mesa_stochastic_multi_length".to_string(),
                input: IndicatorInputKind::Candles,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "mesa_stochastic_multi_length",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("mesa_stochastic_multi_length", params, "source", "close")?;
            let length_1 = get_usize_param("mesa_stochastic_multi_length", params, "length_1", 48)?;
            let length_2 = get_usize_param("mesa_stochastic_multi_length", params, "length_2", 21)?;
            let length_3 = get_usize_param("mesa_stochastic_multi_length", params, "length_3", 9)?;
            let length_4 = get_usize_param("mesa_stochastic_multi_length", params, "length_4", 6)?;
            let trigger_length =
                get_usize_param("mesa_stochastic_multi_length", params, "trigger_length", 2)?;
            let data = match req.data {
                IndicatorDataRef::Slice { values } => values,
                IndicatorDataRef::Candles { candles, .. } => source_type(candles, &source),
                _ => unreachable!(),
            };
            let input = MesaStochasticMultiLengthInput::from_slices(
                data,
                MesaStochasticMultiLengthParams {
                    length_1: Some(length_1),
                    length_2: Some(length_2),
                    length_3: Some(length_3),
                    length_4: Some(length_4),
                    trigger_length: Some(trigger_length),
                },
            );
            let out = mesa_stochastic_multi_length_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "mesa_stochastic_multi_length".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("mesa_1") {
                return Ok(out.mesa_1);
            }
            if output_id.eq_ignore_ascii_case("mesa_2") {
                return Ok(out.mesa_2);
            }
            if output_id.eq_ignore_ascii_case("mesa_3") {
                return Ok(out.mesa_3);
            }
            if output_id.eq_ignore_ascii_case("mesa_4") {
                return Ok(out.mesa_4);
            }
            if output_id.eq_ignore_ascii_case("trigger_1") {
                return Ok(out.trigger_1);
            }
            if output_id.eq_ignore_ascii_case("trigger_2") {
                return Ok(out.trigger_2);
            }
            if output_id.eq_ignore_ascii_case("trigger_3") {
                return Ok(out.trigger_3);
            }
            if output_id.eq_ignore_ascii_case("trigger_4") {
                return Ok(out.trigger_4);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "mesa_stochastic_multi_length".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_spearman_correlation_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "spearman_correlation".to_string(),
                input: IndicatorInputKind::Candles,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "spearman_correlation",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("spearman_correlation", params, "source", "close")?;
            let comparison_source =
                get_enum_param("spearman_correlation", params, "comparison_source", "open")?;
            let lookback = get_usize_param("spearman_correlation", params, "lookback", 30)?;
            let smoothing_length =
                get_usize_param("spearman_correlation", params, "smoothing_length", 3)?;
            let (main, compare) = match req.data {
                IndicatorDataRef::Candles { candles, .. } => (
                    source_type(candles, &source),
                    source_type(candles, &comparison_source),
                ),
                _ => unreachable!(),
            };
            let input = SpearmanCorrelationInput::from_slices(
                main,
                compare,
                SpearmanCorrelationParams {
                    lookback: Some(lookback),
                    smoothing_length: Some(smoothing_length),
                },
            );
            let out = spearman_correlation_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "spearman_correlation".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("raw") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.raw);
            }
            if output_id.eq_ignore_ascii_case("smoothed") {
                return Ok(out.smoothed);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "spearman_correlation".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_relative_strength_index_wave_indicator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "relative_strength_index_wave_indicator".to_string(),
                input: IndicatorInputKind::Candles,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "relative_strength_index_wave_indicator",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param(
                "relative_strength_index_wave_indicator",
                params,
                "source",
                "close",
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
            let (src, high, low) = match req.data {
                IndicatorDataRef::Candles { candles, .. } => (
                    source_type(candles, &source),
                    candles.high.as_slice(),
                    candles.low.as_slice(),
                ),
                _ => unreachable!(),
            };
            let input = RelativeStrengthIndexWaveIndicatorInput::from_slices(
                src,
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
            if output_id.eq_ignore_ascii_case("rsi_ma1") || output_id.eq_ignore_ascii_case("value")
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

fn compute_accumulation_swing_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("accumulation_swing_index", output_id)?;
    let (open, high, low, close) = extract_ohlc_full_input("accumulation_swing_index", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "accumulation_swing_index",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
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
            accumulation_swing_index_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "accumulation_swing_index".to_string(),
                    details: e.to_string(),
                }
            })
        },
    )
}

fn compute_ichimoku_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("ichimoku_oscillator", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ichimoku_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let source_name = get_enum_param("ichimoku_oscillator", params, "source", "close")?;
            let conversion_periods =
                get_usize_param("ichimoku_oscillator", params, "conversion_periods", 9)?;
            let base_periods = get_usize_param("ichimoku_oscillator", params, "base_periods", 26)?;
            let lagging_span_periods =
                get_usize_param("ichimoku_oscillator", params, "lagging_span_periods", 52)?;
            let displacement = get_usize_param("ichimoku_oscillator", params, "displacement", 26)?;
            let ma_length = get_usize_param("ichimoku_oscillator", params, "ma_length", 12)?;
            let smoothing_length =
                get_usize_param("ichimoku_oscillator", params, "smoothing_length", 3)?;
            let extra_smoothing =
                get_bool_param("ichimoku_oscillator", params, "extra_smoothing", true)?;
            let normalize = get_enum_param("ichimoku_oscillator", params, "normalize", "window")?
                .parse::<IchimokuOscillatorNormalizeMode>()
                .map_err(|e| IndicatorDispatchError::InvalidParam {
                    indicator: "ichimoku_oscillator".to_string(),
                    key: "normalize".to_string(),
                    reason: e,
                })?;
            let window_size = get_usize_param("ichimoku_oscillator", params, "window_size", 20)?;
            let clamp = get_bool_param("ichimoku_oscillator", params, "clamp", true)?;
            let top_band = get_f64_param("ichimoku_oscillator", params, "top_band", 2.0)?;
            let mid_band = get_f64_param("ichimoku_oscillator", params, "mid_band", 1.5)?;
            let source = match req.data {
                IndicatorDataRef::Candles { candles, .. } => source_type(candles, &source_name),
                _ => close,
            };
            let input = IchimokuOscillatorInput::from_slices(
                high,
                low,
                close,
                source,
                IchimokuOscillatorParams {
                    conversion_periods: Some(conversion_periods),
                    base_periods: Some(base_periods),
                    lagging_span_periods: Some(lagging_span_periods),
                    displacement: Some(displacement),
                    ma_length: Some(ma_length),
                    smoothing_length: Some(smoothing_length),
                    extra_smoothing: Some(extra_smoothing),
                    normalize: Some(normalize),
                    window_size: Some(window_size),
                    clamp: Some(clamp),
                    top_band: Some(top_band),
                    mid_band: Some(mid_band),
                },
            );
            let out = ichimoku_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "ichimoku_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("signal") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.signal);
            }
            if output_id.eq_ignore_ascii_case("ma") {
                return Ok(out.ma);
            }
            if output_id.eq_ignore_ascii_case("conversion") {
                return Ok(out.conversion);
            }
            if output_id.eq_ignore_ascii_case("base") {
                return Ok(out.base);
            }
            if output_id.eq_ignore_ascii_case("chikou") {
                return Ok(out.chikou);
            }
            if output_id.eq_ignore_ascii_case("current_kumo_a") {
                return Ok(out.current_kumo_a);
            }
            if output_id.eq_ignore_ascii_case("current_kumo_b") {
                return Ok(out.current_kumo_b);
            }
            if output_id.eq_ignore_ascii_case("future_kumo_a") {
                return Ok(out.future_kumo_a);
            }
            if output_id.eq_ignore_ascii_case("future_kumo_b") {
                return Ok(out.future_kumo_b);
            }
            if output_id.eq_ignore_ascii_case("max_level") {
                return Ok(out.max_level);
            }
            if output_id.eq_ignore_ascii_case("high_level") {
                return Ok(out.high_level);
            }
            if output_id.eq_ignore_ascii_case("low_level") {
                return Ok(out.low_level);
            }
            if output_id.eq_ignore_ascii_case("min_level") {
                return Ok(out.min_level);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "ichimoku_oscillator".to_string(),
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
    let (timestamps, high, low, close, volume): (&[i64], &[f64], &[f64], &[f64], &[f64]) =
        match req.data {
            IndicatorDataRef::Candles { candles, .. } => (
                candles.timestamp.as_slice(),
                candles.high.as_slice(),
                candles.low.as_slice(),
                candles.close.as_slice(),
                candles.volume.as_slice(),
            ),
            _ => {
                return Err(IndicatorDispatchError::MissingRequiredInput {
                    indicator: "vwap_deviation_oscillator".to_string(),
                    input: IndicatorInputKind::Candles,
                })
            }
        };
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("osc") || output_id.eq_ignore_ascii_case("value")
    {
        VwapDeviationOscillatorOutputField::Osc
    } else if output_id.eq_ignore_ascii_case("std1") {
        VwapDeviationOscillatorOutputField::Std1
    } else if output_id.eq_ignore_ascii_case("std2") {
        VwapDeviationOscillatorOutputField::Std2
    } else if output_id.eq_ignore_ascii_case("std3") {
        VwapDeviationOscillatorOutputField::Std3
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "vwap_deviation_oscillator".to_string(),
            output: output_id.to_string(),
        });
    };
    let rows = req.combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "vwap_deviation_oscillator".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
        let session_mode = get_enum_param(
            "vwap_deviation_oscillator",
            params,
            "session_mode",
            "rolling_bars",
        )?
        .parse::<VwapDeviationSessionMode>()
        .map_err(|e| IndicatorDispatchError::InvalidParam {
            indicator: "vwap_deviation_oscillator".to_string(),
            key: "session_mode".to_string(),
            reason: e,
        })?;
        let rolling_period =
            get_usize_param("vwap_deviation_oscillator", params, "rolling_period", 20)?;
        let rolling_days =
            get_usize_param("vwap_deviation_oscillator", params, "rolling_days", 30)?;
        let use_close = get_bool_param("vwap_deviation_oscillator", params, "use_close", false)?;
        let deviation_mode = get_enum_param(
            "vwap_deviation_oscillator",
            params,
            "deviation_mode",
            "absolute",
        )?
        .parse::<VwapDeviationMode>()
        .map_err(|e| IndicatorDispatchError::InvalidParam {
            indicator: "vwap_deviation_oscillator".to_string(),
            key: "deviation_mode".to_string(),
            reason: e,
        })?;
        let z_window = get_usize_param("vwap_deviation_oscillator", params, "z_window", 50)?;
        let pct_vol_lookback =
            get_usize_param("vwap_deviation_oscillator", params, "pct_vol_lookback", 100)?;
        let pct_min_sigma =
            get_f64_param("vwap_deviation_oscillator", params, "pct_min_sigma", 0.1)?;
        let abs_vol_lookback =
            get_usize_param("vwap_deviation_oscillator", params, "abs_vol_lookback", 100)?;
        let input = VwapDeviationOscillatorInput::from_slices(
            timestamps,
            high,
            low,
            close,
            volume,
            VwapDeviationOscillatorParams {
                session_mode: Some(session_mode),
                rolling_period: Some(rolling_period),
                rolling_days: Some(rolling_days),
                use_close: Some(use_close),
                deviation_mode: Some(deviation_mode),
                z_window: Some(z_window),
                pct_vol_lookback: Some(pct_vol_lookback),
                pct_min_sigma: Some(pct_min_sigma),
                abs_vol_lookback: Some(abs_vol_lookback),
            },
        );
        let start = row * cols;
        let end = start + cols;
        vwap_deviation_oscillator_output_into_slice(&mut matrix[start..end], &input, kernel, field)
            .map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "vwap_deviation_oscillator".to_string(),
                details: e.to_string(),
            })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
}

fn compute_bulls_v_bears_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("bulls_v_bears", req.data)?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("value") {
        BullsVBearsOutputField::Value
    } else if output_id.eq_ignore_ascii_case("bull") {
        BullsVBearsOutputField::Bull
    } else if output_id.eq_ignore_ascii_case("bear") {
        BullsVBearsOutputField::Bear
    } else if output_id.eq_ignore_ascii_case("ma") {
        BullsVBearsOutputField::Ma
    } else if output_id.eq_ignore_ascii_case("upper") {
        BullsVBearsOutputField::Upper
    } else if output_id.eq_ignore_ascii_case("lower") {
        BullsVBearsOutputField::Lower
    } else if output_id.eq_ignore_ascii_case("bullish_signal") {
        BullsVBearsOutputField::BullishSignal
    } else if output_id.eq_ignore_ascii_case("bearish_signal") {
        BullsVBearsOutputField::BearishSignal
    } else if output_id.eq_ignore_ascii_case("zero_cross_up") {
        BullsVBearsOutputField::ZeroCrossUp
    } else if output_id.eq_ignore_ascii_case("zero_cross_down") {
        BullsVBearsOutputField::ZeroCrossDown
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "bulls_v_bears".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "bulls_v_bears",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
            let period = get_usize_param("bulls_v_bears", params, "period", 14)?;
            let ma_type = get_enum_param("bulls_v_bears", params, "ma_type", "ema")?
                .parse::<BullsVBearsMaType>()
                .map_err(|e| IndicatorDispatchError::InvalidParam {
                    indicator: "bulls_v_bears".to_string(),
                    key: "ma_type".to_string(),
                    reason: e,
                })?;
            let calculation_method =
                get_enum_param("bulls_v_bears", params, "calculation_method", "normalized")?
                    .parse::<BullsVBearsCalculationMethod>()
                    .map_err(|e| IndicatorDispatchError::InvalidParam {
                        indicator: "bulls_v_bears".to_string(),
                        key: "calculation_method".to_string(),
                        reason: e,
                    })?;
            let normalized_bars_back =
                get_usize_param("bulls_v_bears", params, "normalized_bars_back", 120)?;
            let raw_rolling_period =
                get_usize_param("bulls_v_bears", params, "raw_rolling_period", 50)?;
            let raw_threshold_percentile =
                get_f64_param("bulls_v_bears", params, "raw_threshold_percentile", 95.0)?;
            let threshold_level = get_f64_param("bulls_v_bears", params, "threshold_level", 80.0)?;
            let input = BullsVBearsInput::from_slices(
                high,
                low,
                close,
                BullsVBearsParams {
                    period: Some(period),
                    ma_type: Some(ma_type),
                    calculation_method: Some(calculation_method),
                    normalized_bars_back: Some(normalized_bars_back),
                    raw_rolling_period: Some(raw_rolling_period),
                    raw_threshold_percentile: Some(raw_threshold_percentile),
                    threshold_level: Some(threshold_level),
                },
            );
            bulls_v_bears_output_into_slice(row, &input, kernel, field).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "bulls_v_bears".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(())
        },
    )
}

fn compute_smooth_theil_sen_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("smooth_theil_sen", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "smooth_theil_sen",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let length = get_usize_param("smooth_theil_sen", params, "length", 25)?;
            let offset = get_usize_param("smooth_theil_sen", params, "offset", 0)?;
            let multiplier = get_f64_param("smooth_theil_sen", params, "multiplier", 2.0)?;
            let slope_style =
                get_enum_param("smooth_theil_sen", params, "slope_style", "smooth_median")?
                    .parse::<SmoothTheilSenStatStyle>()
                    .map_err(|e| IndicatorDispatchError::InvalidParam {
                        indicator: "smooth_theil_sen".to_string(),
                        key: "slope_style".to_string(),
                        reason: e,
                    })?;
            let residual_style = get_enum_param(
                "smooth_theil_sen",
                params,
                "residual_style",
                "smooth_median",
            )?
            .parse::<SmoothTheilSenStatStyle>()
            .map_err(|e| IndicatorDispatchError::InvalidParam {
                indicator: "smooth_theil_sen".to_string(),
                key: "residual_style".to_string(),
                reason: e,
            })?;
            let deviation_style =
                get_enum_param("smooth_theil_sen", params, "deviation_style", "mad")?
                    .parse::<SmoothTheilSenDeviationType>()
                    .map_err(|e| IndicatorDispatchError::InvalidParam {
                        indicator: "smooth_theil_sen".to_string(),
                        key: "deviation_style".to_string(),
                        reason: e,
                    })?;
            let mad_style =
                get_enum_param("smooth_theil_sen", params, "mad_style", "smooth_median")?
                    .parse::<SmoothTheilSenStatStyle>()
                    .map_err(|e| IndicatorDispatchError::InvalidParam {
                        indicator: "smooth_theil_sen".to_string(),
                        key: "mad_style".to_string(),
                        reason: e,
                    })?;
            let include_prediction_in_deviation = get_bool_param(
                "smooth_theil_sen",
                params,
                "include_prediction_in_deviation",
                false,
            )?;
            let input = SmoothTheilSenInput::from_slice(
                data,
                SmoothTheilSenParams {
                    length: Some(length),
                    offset: Some(offset),
                    multiplier: Some(multiplier),
                    slope_style: Some(slope_style),
                    residual_style: Some(residual_style),
                    deviation_style: Some(deviation_style),
                    mad_style: Some(mad_style),
                    include_prediction_in_deviation: Some(include_prediction_in_deviation),
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

fn compute_regression_slope_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, .. } => candles.close.len(),
        IndicatorDataRef::Slice { values } => values.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "regression_slope_oscillator".to_string(),
                input: IndicatorInputKind::Slice,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "regression_slope_oscillator",
        output_id,
        req.combos,
        data_len,
        |params| {
            let min_range =
                get_usize_param("regression_slope_oscillator", params, "min_range", 10)?;
            let max_range =
                get_usize_param("regression_slope_oscillator", params, "max_range", 100)?;
            let step = get_usize_param("regression_slope_oscillator", params, "step", 5)?;
            let signal_line =
                get_usize_param("regression_slope_oscillator", params, "signal_line", 7)?;
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    RegressionSlopeOscillatorInput::from_candles(
                        candles,
                        RegressionSlopeOscillatorParams {
                            min_range: Some(min_range),
                            max_range: Some(max_range),
                            step: Some(step),
                            signal_line: Some(signal_line),
                        },
                    )
                }
                IndicatorDataRef::Slice { values } => RegressionSlopeOscillatorInput::from_slice(
                    values,
                    RegressionSlopeOscillatorParams {
                        min_range: Some(min_range),
                        max_range: Some(max_range),
                        step: Some(step),
                        signal_line: Some(signal_line),
                    },
                ),
                _ => unreachable!(),
            };
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

fn compute_linear_regression_intensity_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("linear_regression_intensity", output_id)?;
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        IndicatorDataRef::Slice { values } => values.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "linear_regression_intensity".to_string(),
                input: IndicatorInputKind::Slice,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "linear_regression_intensity",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("linear_regression_intensity", params, "source", "close")?;
            let lookback_period =
                get_usize_param("linear_regression_intensity", params, "lookback_period", 12)?;
            let range_tolerance = get_f64_param(
                "linear_regression_intensity",
                params,
                "range_tolerance",
                90.0,
            )?;
            let linreg_length =
                get_usize_param("linear_regression_intensity", params, "linreg_length", 90)?;
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    LinearRegressionIntensityInput::from_candles(
                        candles,
                        &source,
                        LinearRegressionIntensityParams {
                            lookback_period: Some(lookback_period),
                            range_tolerance: Some(range_tolerance),
                            linreg_length: Some(linreg_length),
                        },
                    )
                }
                IndicatorDataRef::Slice { values } => LinearRegressionIntensityInput::from_slice(
                    values,
                    LinearRegressionIntensityParams {
                        lookback_period: Some(lookback_period),
                        range_tolerance: Some(range_tolerance),
                        linreg_length: Some(linreg_length),
                    },
                ),
                _ => unreachable!(),
            };
            let out = linear_regression_intensity_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "linear_regression_intensity".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_moving_average_cross_probability_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, .. } => candles.close.len(),
        IndicatorDataRef::Slice { values } => values.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "moving_average_cross_probability".to_string(),
                input: IndicatorInputKind::Slice,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "moving_average_cross_probability",
        output_id,
        req.combos,
        data_len,
        |params| {
            let ma_type =
                get_enum_param("moving_average_cross_probability", params, "ma_type", "ema")?
                    .parse::<MovingAverageCrossProbabilityMaType>()
                    .map_err(|e| IndicatorDispatchError::InvalidParam {
                        indicator: "moving_average_cross_probability".to_string(),
                        key: "ma_type".to_string(),
                        reason: e,
                    })?;
            let smoothing_window = get_usize_param(
                "moving_average_cross_probability",
                params,
                "smoothing_window",
                7,
            )?;
            let slow_length = get_usize_param(
                "moving_average_cross_probability",
                params,
                "slow_length",
                30,
            )?;
            let fast_length = get_usize_param(
                "moving_average_cross_probability",
                params,
                "fast_length",
                14,
            )?;
            let resolution =
                get_usize_param("moving_average_cross_probability", params, "resolution", 50)?;
            let params = MovingAverageCrossProbabilityParams {
                ma_type: Some(ma_type),
                smoothing_window: Some(smoothing_window),
                slow_length: Some(slow_length),
                fast_length: Some(fast_length),
                resolution: Some(resolution),
            };
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    MovingAverageCrossProbabilityInput::from_candles(candles, params)
                }
                IndicatorDataRef::Slice { values } => {
                    MovingAverageCrossProbabilityInput::from_slice(values, params)
                }
                _ => unreachable!(),
            };
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

fn compute_volume_zone_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("volume_zone_oscillator", output_id)?;
    let (close, volume) = extract_close_volume_input("volume_zone_oscillator", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows(
        "volume_zone_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
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
            volume_zone_oscillator_into_slice(row, &input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "volume_zone_oscillator".to_string(),
                    details: e.to_string(),
                }
            })
        },
    )
}

fn compute_market_meanness_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, .. } => candles.close.len(),
        IndicatorDataRef::Ohlc { close, .. } => close.len(),
        IndicatorDataRef::Ohlcv { close, .. } => close.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "market_meanness_index".to_string(),
                input: IndicatorInputKind::Ohlc,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "market_meanness_index",
        output_id,
        req.combos,
        data_len,
        |params| {
            let length = get_usize_param("market_meanness_index", params, "length", 300)?;
            let source_mode =
                get_enum_param("market_meanness_index", params, "source_mode", "Price")?;
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    MarketMeannessIndexInput::from_candles(
                        candles,
                        MarketMeannessIndexParams {
                            length: Some(length),
                            source_mode: Some(source_mode),
                        },
                    )
                }
                IndicatorDataRef::Ohlc { open, close, .. } => {
                    MarketMeannessIndexInput::from_slices(
                        open,
                        close,
                        MarketMeannessIndexParams {
                            length: Some(length),
                            source_mode: Some(source_mode),
                        },
                    )
                }
                IndicatorDataRef::Ohlcv { open, close, .. } => {
                    MarketMeannessIndexInput::from_slices(
                        open,
                        close,
                        MarketMeannessIndexParams {
                            length: Some(length),
                            source_mode: Some(source_mode),
                        },
                    )
                }
                _ => unreachable!(),
            };
            let out = market_meanness_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "market_meanness_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("mmi") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.mmi);
            }
            if output_id.eq_ignore_ascii_case("mmi_smoothed") {
                return Ok(out.mmi_smoothed);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "market_meanness_index".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_momentum_ratio_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        IndicatorDataRef::Slice { values } => values.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "momentum_ratio_oscillator".to_string(),
                input: IndicatorInputKind::Slice,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "momentum_ratio_oscillator",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("momentum_ratio_oscillator", params, "source", "close")?;
            let period = get_usize_param("momentum_ratio_oscillator", params, "period", 50)?;
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    MomentumRatioOscillatorInput::from_candles(
                        candles,
                        &source,
                        MomentumRatioOscillatorParams {
                            period: Some(period),
                        },
                    )
                }
                IndicatorDataRef::Slice { values } => MomentumRatioOscillatorInput::from_slice(
                    values,
                    MomentumRatioOscillatorParams {
                        period: Some(period),
                    },
                ),
                _ => unreachable!(),
            };
            let out = momentum_ratio_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "momentum_ratio_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("line") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.line);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "momentum_ratio_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_pretty_good_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("pretty_good_oscillator", output_id)?;
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        IndicatorDataRef::Ohlc { close, .. } => close.len(),
        IndicatorDataRef::Ohlcv { close, .. } => close.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "pretty_good_oscillator".to_string(),
                input: IndicatorInputKind::Ohlc,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "pretty_good_oscillator",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("pretty_good_oscillator", params, "source", "close")?;
            let length = get_usize_param("pretty_good_oscillator", params, "length", 14)?;
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    PrettyGoodOscillatorInput::from_candles(
                        candles,
                        &source,
                        PrettyGoodOscillatorParams {
                            length: Some(length),
                        },
                    )
                }
                IndicatorDataRef::Ohlc {
                    high,
                    low,
                    close,
                    open,
                } => {
                    ensure_same_len_4(
                        "pretty_good_oscillator",
                        open.len(),
                        high.len(),
                        low.len(),
                        close.len(),
                    )?;
                    let src = match source.to_ascii_lowercase().as_str() {
                        "open" => open,
                        "high" => high,
                        "low" => low,
                        _ => close,
                    };
                    PrettyGoodOscillatorInput::from_slices(
                        high,
                        low,
                        close,
                        src,
                        PrettyGoodOscillatorParams {
                            length: Some(length),
                        },
                    )
                }
                IndicatorDataRef::Ohlcv {
                    high,
                    low,
                    close,
                    open,
                    volume,
                } => {
                    ensure_same_len_5(
                        "pretty_good_oscillator",
                        open.len(),
                        high.len(),
                        low.len(),
                        close.len(),
                        volume.len(),
                    )?;
                    let src = match source.to_ascii_lowercase().as_str() {
                        "open" => open,
                        "high" => high,
                        "low" => low,
                        _ => close,
                    };
                    PrettyGoodOscillatorInput::from_slices(
                        high,
                        low,
                        close,
                        src,
                        PrettyGoodOscillatorParams {
                            length: Some(length),
                        },
                    )
                }
                _ => unreachable!(),
            };
            let out = pretty_good_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "pretty_good_oscillator".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_price_density_market_noise_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("price_density_market_noise", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "price_density_market_noise",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length = get_usize_param("price_density_market_noise", params, "length", 14)?;
            let eval_period =
                get_usize_param("price_density_market_noise", params, "eval_period", 200)?;
            let input = PriceDensityMarketNoiseInput::from_slices(
                high,
                low,
                close,
                PriceDensityMarketNoiseParams {
                    length: Some(length),
                    eval_period: Some(eval_period),
                },
            );
            let out = price_density_market_noise_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "price_density_market_noise".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("price_density")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.price_density);
            }
            if output_id.eq_ignore_ascii_case("price_density_percent") {
                return Ok(out.price_density_percent);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "price_density_market_noise".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_psychological_line_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("psychological_line", output_id)?;
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        IndicatorDataRef::Slice { values } => values.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "psychological_line".to_string(),
                input: IndicatorInputKind::Slice,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "psychological_line",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("psychological_line", params, "source", "close")?;
            let length = get_usize_param("psychological_line", params, "length", 20)?;
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => PsychologicalLineInput::from_candles(
                    candles,
                    &source,
                    PsychologicalLineParams {
                        length: Some(length),
                    },
                ),
                IndicatorDataRef::Slice { values } => PsychologicalLineInput::from_slice(
                    values,
                    PsychologicalLineParams {
                        length: Some(length),
                    },
                ),
                _ => unreachable!(),
            };
            let out = psychological_line_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "psychological_line".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_rank_correlation_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("rank_correlation_index", output_id)?;
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        IndicatorDataRef::Slice { values } => values.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "rank_correlation_index".to_string(),
                input: IndicatorInputKind::Slice,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "rank_correlation_index",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("rank_correlation_index", params, "source", "close")?;
            let length = get_usize_param("rank_correlation_index", params, "length", 12)?;
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    RankCorrelationIndexInput::from_candles(
                        candles,
                        &source,
                        RankCorrelationIndexParams {
                            length: Some(length),
                        },
                    )
                }
                IndicatorDataRef::Slice { values } => RankCorrelationIndexInput::from_slice(
                    values,
                    RankCorrelationIndexParams {
                        length: Some(length),
                    },
                ),
                _ => unreachable!(),
            };
            let out = rank_correlation_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "rank_correlation_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_smoothed_gaussian_trend_filter_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("smoothed_gaussian_trend_filter", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "smoothed_gaussian_trend_filter",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let gaussian_length = get_usize_param(
                "smoothed_gaussian_trend_filter",
                params,
                "gaussian_length",
                15,
            )?;
            let poles = get_usize_param("smoothed_gaussian_trend_filter", params, "poles", 3)?;
            let smoothing_length = get_usize_param(
                "smoothed_gaussian_trend_filter",
                params,
                "smoothing_length",
                22,
            )?;
            let linreg_offset =
                get_usize_param("smoothed_gaussian_trend_filter", params, "linreg_offset", 7)?;
            let input = SmoothedGaussianTrendFilterInput::from_slices(
                high,
                low,
                close,
                SmoothedGaussianTrendFilterParams {
                    gaussian_length: Some(gaussian_length),
                    poles: Some(poles),
                    smoothing_length: Some(smoothing_length),
                    linreg_offset: Some(linreg_offset),
                },
            );
            if output_id.eq_ignore_ascii_case("filter") || output_id.eq_ignore_ascii_case("value") {
                return smoothed_gaussian_trend_filter_filter_with_kernel(&input, kernel).map_err(
                    |e| IndicatorDispatchError::ComputeFailed {
                        indicator: "smoothed_gaussian_trend_filter".to_string(),
                        details: e.to_string(),
                    },
                );
            }
            let out = smoothed_gaussian_trend_filter_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "smoothed_gaussian_trend_filter".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("supertrend") {
                return Ok(out.supertrend);
            }
            if output_id.eq_ignore_ascii_case("trend") {
                return Ok(out.trend);
            }
            if output_id.eq_ignore_ascii_case("ranging") {
                return Ok(out.ranging);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "smoothed_gaussian_trend_filter".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_stochastic_adaptive_d_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("stochastic_adaptive_d", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "stochastic_adaptive_d",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let k_length = get_usize_param("stochastic_adaptive_d", params, "k_length", 20)?;
            let d_smoothing = get_usize_param("stochastic_adaptive_d", params, "d_smoothing", 9)?;
            let pre_smooth = get_usize_param("stochastic_adaptive_d", params, "pre_smooth", 20)?;
            let attenuation = get_f64_param("stochastic_adaptive_d", params, "attenuation", 2.0)?;
            let input = StochasticAdaptiveDInput::from_slices(
                high,
                low,
                close,
                StochasticAdaptiveDParams {
                    k_length: Some(k_length),
                    d_smoothing: Some(d_smoothing),
                    pre_smooth: Some(pre_smooth),
                    attenuation: Some(attenuation),
                },
            );
            let out = stochastic_adaptive_d_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "stochastic_adaptive_d".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("standard_d")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.standard_d);
            }
            if output_id.eq_ignore_ascii_case("adaptive_d") {
                return Ok(out.adaptive_d);
            }
            if output_id.eq_ignore_ascii_case("difference") {
                return Ok(out.difference);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "stochastic_adaptive_d".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_stochastic_connors_rsi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        IndicatorDataRef::Slice { values } => values.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "stochastic_connors_rsi".to_string(),
                input: IndicatorInputKind::Slice,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "stochastic_connors_rsi",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("stochastic_connors_rsi", params, "source", "close")?;
            let stoch_length =
                get_usize_param("stochastic_connors_rsi", params, "stoch_length", 3)?;
            let smooth_k = get_usize_param("stochastic_connors_rsi", params, "smooth_k", 3)?;
            let smooth_d = get_usize_param("stochastic_connors_rsi", params, "smooth_d", 3)?;
            let rsi_length = get_usize_param("stochastic_connors_rsi", params, "rsi_length", 3)?;
            let updown_length =
                get_usize_param("stochastic_connors_rsi", params, "updown_length", 2)?;
            let roc_length = get_usize_param("stochastic_connors_rsi", params, "roc_length", 100)?;
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    StochasticConnorsRsiInput::from_candles(
                        candles,
                        &source,
                        StochasticConnorsRsiParams {
                            stoch_length: Some(stoch_length),
                            smooth_k: Some(smooth_k),
                            smooth_d: Some(smooth_d),
                            rsi_length: Some(rsi_length),
                            updown_length: Some(updown_length),
                            roc_length: Some(roc_length),
                        },
                    )
                }
                IndicatorDataRef::Slice { values } => StochasticConnorsRsiInput::from_slice(
                    values,
                    StochasticConnorsRsiParams {
                        stoch_length: Some(stoch_length),
                        smooth_k: Some(smooth_k),
                        smooth_d: Some(smooth_d),
                        rsi_length: Some(rsi_length),
                        updown_length: Some(updown_length),
                        roc_length: Some(roc_length),
                    },
                ),
                _ => unreachable!(),
            };
            let out = stochastic_connors_rsi_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "stochastic_connors_rsi".to_string(),
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
                indicator: "stochastic_connors_rsi".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_supertrend_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        IndicatorDataRef::Ohlc { close, .. } => close.len(),
        IndicatorDataRef::Ohlcv { close, .. } => close.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "supertrend_oscillator".to_string(),
                input: IndicatorInputKind::Ohlc,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "supertrend_oscillator",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("supertrend_oscillator", params, "source", "close")?;
            let length = get_usize_param("supertrend_oscillator", params, "length", 10)?;
            let mult = get_f64_param("supertrend_oscillator", params, "mult", 2.0)?;
            let smooth = get_usize_param("supertrend_oscillator", params, "smooth", 72)?;
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    SuperTrendOscillatorInput::from_candles(
                        candles,
                        &source,
                        SuperTrendOscillatorParams {
                            length: Some(length),
                            mult: Some(mult),
                            smooth: Some(smooth),
                        },
                    )
                }
                IndicatorDataRef::Ohlc {
                    high,
                    low,
                    close,
                    open,
                } => {
                    ensure_same_len_4(
                        "supertrend_oscillator",
                        open.len(),
                        high.len(),
                        low.len(),
                        close.len(),
                    )?;
                    let src = match source.to_ascii_lowercase().as_str() {
                        "open" => open,
                        "high" => high,
                        "low" => low,
                        _ => close,
                    };
                    SuperTrendOscillatorInput::from_slices(
                        high,
                        low,
                        src,
                        SuperTrendOscillatorParams {
                            length: Some(length),
                            mult: Some(mult),
                            smooth: Some(smooth),
                        },
                    )
                }
                IndicatorDataRef::Ohlcv {
                    high,
                    low,
                    close,
                    open,
                    volume,
                } => {
                    ensure_same_len_5(
                        "supertrend_oscillator",
                        open.len(),
                        high.len(),
                        low.len(),
                        close.len(),
                        volume.len(),
                    )?;
                    let src = match source.to_ascii_lowercase().as_str() {
                        "open" => open,
                        "high" => high,
                        "low" => low,
                        _ => close,
                    };
                    SuperTrendOscillatorInput::from_slices(
                        high,
                        low,
                        src,
                        SuperTrendOscillatorParams {
                            length: Some(length),
                            mult: Some(mult),
                            smooth: Some(smooth),
                        },
                    )
                }
                _ => unreachable!(),
            };
            let out = supertrend_oscillator_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "supertrend_oscillator".to_string(),
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
            if output_id.eq_ignore_ascii_case("histogram") || output_id.eq_ignore_ascii_case("hist")
            {
                return Ok(out.histogram);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "supertrend_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_trend_continuation_factor_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        IndicatorDataRef::Slice { values } => values.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "trend_continuation_factor".to_string(),
                input: IndicatorInputKind::Slice,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "trend_continuation_factor",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("trend_continuation_factor", params, "source", "close")?;
            let length = get_usize_param("trend_continuation_factor", params, "length", 35)?;
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    TrendContinuationFactorInput::from_candles(
                        candles,
                        &source,
                        TrendContinuationFactorParams {
                            length: Some(length),
                        },
                    )
                }
                IndicatorDataRef::Slice { values } => TrendContinuationFactorInput::from_slice(
                    values,
                    TrendContinuationFactorParams {
                        length: Some(length),
                    },
                ),
                _ => unreachable!(),
            };
            let out = trend_continuation_factor_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "trend_continuation_factor".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("plus_tcf") || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.plus_tcf);
            }
            if output_id.eq_ignore_ascii_case("minus_tcf") {
                return Ok(out.minus_tcf);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "trend_continuation_factor".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_volume_weighted_stochastic_rsi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (source, volume) =
        extract_close_volume_input("volume_weighted_stochastic_rsi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("k") || output_id.eq_ignore_ascii_case("value") {
        VolumeWeightedStochasticRsiOutputField::K
    } else if output_id.eq_ignore_ascii_case("d") {
        VolumeWeightedStochasticRsiOutputField::D
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "volume_weighted_stochastic_rsi".to_string(),
            output: output_id.to_string(),
        });
    };
    let rows = req.combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "volume_weighted_stochastic_rsi".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
        let rsi_length =
            get_usize_param("volume_weighted_stochastic_rsi", params, "rsi_length", 14)?;
        let stoch_length =
            get_usize_param("volume_weighted_stochastic_rsi", params, "stoch_length", 14)?;
        let k_length = get_usize_param("volume_weighted_stochastic_rsi", params, "k_length", 3)?;
        let d_length = get_usize_param("volume_weighted_stochastic_rsi", params, "d_length", 3)?;
        let ma_type = get_enum_param("volume_weighted_stochastic_rsi", params, "ma_type", "WSMA")?;
        let input = VolumeWeightedStochasticRsiInput::from_slices(
            source,
            volume,
            VolumeWeightedStochasticRsiParams {
                rsi_length: Some(rsi_length),
                stoch_length: Some(stoch_length),
                k_length: Some(k_length),
                d_length: Some(d_length),
                ma_type: Some(ma_type),
            },
        );
        let start = row * cols;
        let end = start + cols;
        volume_weighted_stochastic_rsi_output_into_slice(
            &mut matrix[start..end],
            &input,
            kernel,
            field,
        )
        .map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "volume_weighted_stochastic_rsi".to_string(),
            details: e.to_string(),
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
}

fn compute_logarithmic_moving_average_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        IndicatorDataRef::Slice { values } => values.len(),
        IndicatorDataRef::CloseVolume { close, .. } => close.len(),
        IndicatorDataRef::Ohlcv { close, .. } => close.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "logarithmic_moving_average".to_string(),
                input: IndicatorInputKind::CloseVolume,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "logarithmic_moving_average",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("logarithmic_moving_average", params, "source", "close")?;
            let period = get_usize_param("logarithmic_moving_average", params, "period", 100)?;
            let steepness = get_f64_param("logarithmic_moving_average", params, "steepness", 2.5)?;
            let ma_type = get_enum_param("logarithmic_moving_average", params, "ma_type", "ema")?;
            let smooth = get_usize_param("logarithmic_moving_average", params, "smooth", 10)?;
            let momentum_weight =
                get_f64_param("logarithmic_moving_average", params, "momentum_weight", 1.2)?;
            let long_threshold =
                get_f64_param("logarithmic_moving_average", params, "long_threshold", 0.5)?;
            let short_threshold = get_f64_param(
                "logarithmic_moving_average",
                params,
                "short_threshold",
                -0.5,
            )?;
            let params = LogarithmicMovingAverageParams {
                period: Some(period),
                steepness: Some(steepness),
                ma_type: Some(ma_type),
                smooth: Some(smooth),
                momentum_weight: Some(momentum_weight),
                long_threshold: Some(long_threshold),
                short_threshold: Some(short_threshold),
            };
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    LogarithmicMovingAverageInput::from_candles(candles, &source, params)
                }
                IndicatorDataRef::Slice { values } => {
                    LogarithmicMovingAverageInput::from_slice(values, params)
                }
                IndicatorDataRef::CloseVolume { close, volume } => {
                    LogarithmicMovingAverageInput::from_slice_with_volume(close, volume, params)
                }
                IndicatorDataRef::Ohlcv {
                    open,
                    high,
                    low,
                    close,
                    volume,
                } => {
                    let price = match source.to_ascii_lowercase().as_str() {
                        "open" => open,
                        "high" => high,
                        "low" => low,
                        _ => close,
                    };
                    LogarithmicMovingAverageInput::from_slice_with_volume(price, volume, params)
                }
                _ => unreachable!(),
            };
            let out = logarithmic_moving_average_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "logarithmic_moving_average".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("lma") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.lma);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            if output_id.eq_ignore_ascii_case("position") {
                return Ok(out.position);
            }
            if output_id.eq_ignore_ascii_case("momentum_confirmed") {
                return Ok(out.momentum_confirmed);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "logarithmic_moving_average".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn adaptive_bounds_rsi_field(
    output_id: &str,
) -> Result<AdaptiveBoundsRsiOutputField, IndicatorDispatchError> {
    if output_id.eq_ignore_ascii_case("rsi") || output_id.eq_ignore_ascii_case("value") {
        return Ok(AdaptiveBoundsRsiOutputField::Rsi);
    }
    if output_id.eq_ignore_ascii_case("lower_bound") {
        return Ok(AdaptiveBoundsRsiOutputField::LowerBound);
    }
    if output_id.eq_ignore_ascii_case("lower_mid") {
        return Ok(AdaptiveBoundsRsiOutputField::LowerMid);
    }
    if output_id.eq_ignore_ascii_case("mid") {
        return Ok(AdaptiveBoundsRsiOutputField::Mid);
    }
    if output_id.eq_ignore_ascii_case("upper_mid") {
        return Ok(AdaptiveBoundsRsiOutputField::UpperMid);
    }
    if output_id.eq_ignore_ascii_case("upper_bound") {
        return Ok(AdaptiveBoundsRsiOutputField::UpperBound);
    }
    if output_id.eq_ignore_ascii_case("regime") {
        return Ok(AdaptiveBoundsRsiOutputField::Regime);
    }
    if output_id.eq_ignore_ascii_case("regime_flip") {
        return Ok(AdaptiveBoundsRsiOutputField::RegimeFlip);
    }
    if output_id.eq_ignore_ascii_case("lower_signal") {
        return Ok(AdaptiveBoundsRsiOutputField::LowerSignal);
    }
    if output_id.eq_ignore_ascii_case("upper_signal") {
        return Ok(AdaptiveBoundsRsiOutputField::UpperSignal);
    }
    Err(IndicatorDispatchError::UnknownOutput {
        indicator: "adaptive_bounds_rsi".to_string(),
        output: output_id.to_string(),
    })
}

fn compute_adaptive_bounds_rsi_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("adaptive_bounds_rsi", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    let field = adaptive_bounds_rsi_field(output_id)?;
    collect_f64_into_rows(
        "adaptive_bounds_rsi",
        output_id,
        req.combos,
        data.len(),
        |params, row| {
            let rsi_length = get_usize_param("adaptive_bounds_rsi", params, "rsi_length", 14)?;
            let alpha = get_f64_param("adaptive_bounds_rsi", params, "alpha", 0.1)?;
            let input = AdaptiveBoundsRsiInput::from_slice(
                data,
                AdaptiveBoundsRsiParams {
                    rsi_length: Some(rsi_length),
                    alpha: Some(alpha),
                },
            );
            adaptive_bounds_rsi_output_into_slice(row, &input, kernel, field).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "adaptive_bounds_rsi".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(())
        },
    )
}

fn compute_adaptive_schaff_trend_cycle_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("adaptive_schaff_trend_cycle", req.data)?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("stc") || output_id.eq_ignore_ascii_case("value")
    {
        AdaptiveSchaffTrendCycleOutputField::Stc
    } else if output_id.eq_ignore_ascii_case("histogram") || output_id.eq_ignore_ascii_case("hist")
    {
        AdaptiveSchaffTrendCycleOutputField::Histogram
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "adaptive_schaff_trend_cycle".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "adaptive_schaff_trend_cycle",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
            let adaptive_length =
                get_usize_param("adaptive_schaff_trend_cycle", params, "adaptive_length", 55)?;
            let stc_length =
                get_usize_param("adaptive_schaff_trend_cycle", params, "stc_length", 12)?;
            let smoothing_factor = get_f64_param(
                "adaptive_schaff_trend_cycle",
                params,
                "smoothing_factor",
                0.45,
            )?;
            let fast_length =
                get_usize_param("adaptive_schaff_trend_cycle", params, "fast_length", 26)?;
            let slow_length =
                get_usize_param("adaptive_schaff_trend_cycle", params, "slow_length", 50)?;
            let input = AdaptiveSchaffTrendCycleInput::from_slices(
                high,
                low,
                close,
                AdaptiveSchaffTrendCycleParams {
                    adaptive_length: Some(adaptive_length),
                    stc_length: Some(stc_length),
                    smoothing_factor: Some(smoothing_factor),
                    fast_length: Some(fast_length),
                    slow_length: Some(slow_length),
                },
            );
            adaptive_schaff_trend_cycle_output_into_slice(row, &input, kernel, field).map_err(
                |e| IndicatorDispatchError::ComputeFailed {
                    indicator: "adaptive_schaff_trend_cycle".to_string(),
                    details: e.to_string(),
                },
            )?;
            Ok(())
        },
    )
}

fn compute_ehlers_detrending_filter_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("hlcc4")).len()
        }
        IndicatorDataRef::Slice { values } => values.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "ehlers_detrending_filter".to_string(),
                input: IndicatorInputKind::Slice,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "ehlers_detrending_filter",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param("ehlers_detrending_filter", params, "source", "hlcc4")?;
            let length = get_usize_param("ehlers_detrending_filter", params, "length", 10)?;
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    EhlersDetrendingFilterInput::from_candles(
                        candles,
                        &source,
                        EhlersDetrendingFilterParams {
                            length: Some(length),
                        },
                    )
                }
                IndicatorDataRef::Slice { values } => EhlersDetrendingFilterInput::from_slice(
                    values,
                    EhlersDetrendingFilterParams {
                        length: Some(length),
                    },
                ),
                _ => unreachable!(),
            };
            let out = ehlers_detrending_filter_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "ehlers_detrending_filter".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("edf") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.edf);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "ehlers_detrending_filter".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_hypertrend_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        IndicatorDataRef::Ohlc { close, .. } => close.len(),
        IndicatorDataRef::Ohlcv { close, .. } => close.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "hypertrend".to_string(),
                input: IndicatorInputKind::Ohlc,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("upper") {
        HyperTrendOutputField::Upper
    } else if output_id.eq_ignore_ascii_case("average") || output_id.eq_ignore_ascii_case("value") {
        HyperTrendOutputField::Average
    } else if output_id.eq_ignore_ascii_case("lower") {
        HyperTrendOutputField::Lower
    } else if output_id.eq_ignore_ascii_case("trend") {
        HyperTrendOutputField::Trend
    } else if output_id.eq_ignore_ascii_case("changed") {
        HyperTrendOutputField::Changed
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "hypertrend".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "hypertrend",
        output_id,
        req.combos,
        data_len,
        |params, row| {
            let source = get_enum_param("hypertrend", params, "source", "close")?;
            let factor = get_f64_param("hypertrend", params, "factor", 5.0)?;
            let slope = get_f64_param("hypertrend", params, "slope", 14.0)?;
            let width_percent = get_f64_param("hypertrend", params, "width_percent", 80.0)?;
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => HyperTrendInput::from_candles(
                    candles,
                    &source,
                    HyperTrendParams {
                        factor: Some(factor),
                        slope: Some(slope),
                        width_percent: Some(width_percent),
                    },
                ),
                IndicatorDataRef::Ohlc {
                    high,
                    low,
                    close,
                    open,
                } => {
                    ensure_same_len_4(
                        "hypertrend",
                        open.len(),
                        high.len(),
                        low.len(),
                        close.len(),
                    )?;
                    let src = match source.to_ascii_lowercase().as_str() {
                        "open" => open,
                        "high" => high,
                        "low" => low,
                        _ => close,
                    };
                    HyperTrendInput::from_slices(
                        high,
                        low,
                        src,
                        HyperTrendParams {
                            factor: Some(factor),
                            slope: Some(slope),
                            width_percent: Some(width_percent),
                        },
                    )
                }
                IndicatorDataRef::Ohlcv {
                    high,
                    low,
                    close,
                    open,
                    volume,
                } => {
                    ensure_same_len_5(
                        "hypertrend",
                        open.len(),
                        high.len(),
                        low.len(),
                        close.len(),
                        volume.len(),
                    )?;
                    let src = match source.to_ascii_lowercase().as_str() {
                        "open" => open,
                        "high" => high,
                        "low" => low,
                        _ => close,
                    };
                    HyperTrendInput::from_slices(
                        high,
                        low,
                        src,
                        HyperTrendParams {
                            factor: Some(factor),
                            slope: Some(slope),
                            width_percent: Some(width_percent),
                        },
                    )
                }
                _ => unreachable!(),
            };
            hypertrend_output_into_slice(row, &input, kernel, field).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "hypertrend".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(())
        },
    )
}

fn compute_ict_propulsion_block_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("ict_propulsion_block", req.data)?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("bullish_high") {
        0
    } else if output_id.eq_ignore_ascii_case("bullish_low") {
        1
    } else if output_id.eq_ignore_ascii_case("bullish_kind") {
        2
    } else if output_id.eq_ignore_ascii_case("bullish_active") {
        3
    } else if output_id.eq_ignore_ascii_case("bullish_mitigated") {
        4
    } else if output_id.eq_ignore_ascii_case("bullish_new") {
        5
    } else if output_id.eq_ignore_ascii_case("bearish_high") {
        6
    } else if output_id.eq_ignore_ascii_case("bearish_low") {
        7
    } else if output_id.eq_ignore_ascii_case("bearish_kind") {
        8
    } else if output_id.eq_ignore_ascii_case("bearish_active") {
        9
    } else if output_id.eq_ignore_ascii_case("bearish_mitigated") {
        10
    } else if output_id.eq_ignore_ascii_case("bearish_new") {
        11
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "ict_propulsion_block".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "ict_propulsion_block",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
            let swing_length = get_usize_param("ict_propulsion_block", params, "swing_length", 3)?;
            let mitigation_price =
                match get_enum_param("ict_propulsion_block", params, "mitigation_price", "close")?
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "close" => IctPropulsionBlockMitigationPrice::Close,
                    "wick" => IctPropulsionBlockMitigationPrice::Wick,
                    other => {
                        return Err(IndicatorDispatchError::InvalidParam {
                            indicator: "ict_propulsion_block".to_string(),
                            key: "mitigation_price".to_string(),
                            reason: format!("unsupported value '{other}'"),
                        })
                    }
                };
            let input = IctPropulsionBlockInput::from_slices(
                open,
                high,
                low,
                close,
                IctPropulsionBlockParams {
                    swing_length: Some(swing_length),
                    mitigation_price: Some(mitigation_price),
                },
            );
            let len = close.len();
            let mut s0 = alloc_uninit_f64(len);
            let mut s1 = alloc_uninit_f64(len);
            let mut s2 = alloc_uninit_f64(len);
            let mut s3 = alloc_uninit_f64(len);
            let mut s4 = alloc_uninit_f64(len);
            let mut s5 = alloc_uninit_f64(len);
            let mut s6 = alloc_uninit_f64(len);
            let mut s7 = alloc_uninit_f64(len);
            let mut s8 = alloc_uninit_f64(len);
            let mut s9 = alloc_uninit_f64(len);
            let mut s10 = alloc_uninit_f64(len);
            let result = match field {
                0 => ict_propulsion_block_into_slice(
                    row, &mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, &mut s6, &mut s7,
                    &mut s8, &mut s9, &mut s10, &input, kernel,
                ),
                1 => ict_propulsion_block_into_slice(
                    &mut s0, row, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, &mut s6, &mut s7,
                    &mut s8, &mut s9, &mut s10, &input, kernel,
                ),
                2 => ict_propulsion_block_into_slice(
                    &mut s0, &mut s1, row, &mut s2, &mut s3, &mut s4, &mut s5, &mut s6, &mut s7,
                    &mut s8, &mut s9, &mut s10, &input, kernel,
                ),
                3 => ict_propulsion_block_into_slice(
                    &mut s0, &mut s1, &mut s2, row, &mut s3, &mut s4, &mut s5, &mut s6, &mut s7,
                    &mut s8, &mut s9, &mut s10, &input, kernel,
                ),
                4 => ict_propulsion_block_into_slice(
                    &mut s0, &mut s1, &mut s2, &mut s3, row, &mut s4, &mut s5, &mut s6, &mut s7,
                    &mut s8, &mut s9, &mut s10, &input, kernel,
                ),
                5 => ict_propulsion_block_into_slice(
                    &mut s0, &mut s1, &mut s2, &mut s3, &mut s4, row, &mut s5, &mut s6, &mut s7,
                    &mut s8, &mut s9, &mut s10, &input, kernel,
                ),
                6 => ict_propulsion_block_into_slice(
                    &mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, row, &mut s6, &mut s7,
                    &mut s8, &mut s9, &mut s10, &input, kernel,
                ),
                7 => ict_propulsion_block_into_slice(
                    &mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, &mut s6, row, &mut s7,
                    &mut s8, &mut s9, &mut s10, &input, kernel,
                ),
                8 => ict_propulsion_block_into_slice(
                    &mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, &mut s6, &mut s7, row,
                    &mut s8, &mut s9, &mut s10, &input, kernel,
                ),
                9 => ict_propulsion_block_into_slice(
                    &mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, &mut s6, &mut s7,
                    &mut s8, row, &mut s9, &mut s10, &input, kernel,
                ),
                10 => ict_propulsion_block_into_slice(
                    &mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, &mut s6, &mut s7,
                    &mut s8, &mut s9, row, &mut s10, &input, kernel,
                ),
                11 => ict_propulsion_block_into_slice(
                    &mut s0, &mut s1, &mut s2, &mut s3, &mut s4, &mut s5, &mut s6, &mut s7,
                    &mut s8, &mut s9, &mut s10, row, &input, kernel,
                ),
                _ => unreachable!(),
            };
            result.map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "ict_propulsion_block".to_string(),
                details: e.to_string(),
            })?;
            Ok(())
        },
    )
}

fn compute_impulse_macd_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("impulse_macd", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "impulse_macd",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let length_ma = get_usize_param("impulse_macd", params, "length_ma", 34)?;
            let length_signal = get_usize_param("impulse_macd", params, "length_signal", 9)?;
            let input = ImpulseMacdInput::from_slices(
                high,
                low,
                close,
                ImpulseMacdParams {
                    length_ma: Some(length_ma),
                    length_signal: Some(length_signal),
                },
            );
            let out = impulse_macd_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "impulse_macd".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("impulse_macd")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.impulse_macd);
            }
            if output_id.eq_ignore_ascii_case("impulse_histo")
                || output_id.eq_ignore_ascii_case("histogram")
                || output_id.eq_ignore_ascii_case("hist")
            {
                return Ok(out.impulse_histo);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "impulse_macd".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_keltner_channel_width_oscillator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("keltner_channel_width_oscillator", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "keltner_channel_width_oscillator",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let source = get_enum_param(
                "keltner_channel_width_oscillator",
                params,
                "source",
                "close",
            )?;
            let length = get_usize_param("keltner_channel_width_oscillator", params, "length", 20)?;
            let multiplier = get_f64_param(
                "keltner_channel_width_oscillator",
                params,
                "multiplier",
                2.0,
            )?;
            let use_exponential = get_bool_param(
                "keltner_channel_width_oscillator",
                params,
                "use_exponential",
                true,
            )?;
            let bands_style = get_enum_param(
                "keltner_channel_width_oscillator",
                params,
                "bands_style",
                "Average True Range",
            )?;
            let atr_length =
                get_usize_param("keltner_channel_width_oscillator", params, "atr_length", 10)?;
            let src = match req.data {
                IndicatorDataRef::Candles { candles, .. } => source_type(candles, &source),
                IndicatorDataRef::Ohlc {
                    open,
                    high,
                    low,
                    close,
                } => {
                    ensure_same_len_4(
                        "keltner_channel_width_oscillator",
                        open.len(),
                        high.len(),
                        low.len(),
                        close.len(),
                    )?;
                    match source.to_ascii_lowercase().as_str() {
                        "open" => open,
                        "high" => high,
                        "low" => low,
                        _ => close,
                    }
                }
                IndicatorDataRef::Ohlcv {
                    open,
                    high,
                    low,
                    close,
                    volume,
                } => {
                    ensure_same_len_5(
                        "keltner_channel_width_oscillator",
                        open.len(),
                        high.len(),
                        low.len(),
                        close.len(),
                        volume.len(),
                    )?;
                    match source.to_ascii_lowercase().as_str() {
                        "open" => open,
                        "high" => high,
                        "low" => low,
                        _ => close,
                    }
                }
                _ => close,
            };
            let input = KeltnerChannelWidthOscillatorInput::from_slices(
                high,
                low,
                close,
                src,
                KeltnerChannelWidthOscillatorParams {
                    length: Some(length),
                    multiplier: Some(multiplier),
                    use_exponential: Some(use_exponential),
                    bands_style: Some(bands_style),
                    atr_length: Some(atr_length),
                },
            );
            let out =
                keltner_channel_width_oscillator_with_kernel(&input, kernel).map_err(|e| {
                    IndicatorDispatchError::ComputeFailed {
                        indicator: "keltner_channel_width_oscillator".to_string(),
                        details: e.to_string(),
                    }
                })?;
            if output_id.eq_ignore_ascii_case("kbw") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.kbw);
            }
            if output_id.eq_ignore_ascii_case("kbw_sma") {
                return Ok(out.kbw_sma);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "keltner_channel_width_oscillator".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_leavitt_convolution_acceleration_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data_len = match req.data {
        IndicatorDataRef::Candles { candles, source } => {
            source_type(candles, source.unwrap_or("close")).len()
        }
        IndicatorDataRef::Slice { values } => values.len(),
        _ => {
            return Err(IndicatorDispatchError::MissingRequiredInput {
                indicator: "leavitt_convolution_acceleration".to_string(),
                input: IndicatorInputKind::Slice,
            })
        }
    };
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "leavitt_convolution_acceleration",
        output_id,
        req.combos,
        data_len,
        |params| {
            let source = get_enum_param(
                "leavitt_convolution_acceleration",
                params,
                "source",
                "close",
            )?;
            let length = get_usize_param("leavitt_convolution_acceleration", params, "length", 70)?;
            let norm_length = get_usize_param(
                "leavitt_convolution_acceleration",
                params,
                "norm_length",
                150,
            )?;
            let use_norm_hyperbolic = get_bool_param(
                "leavitt_convolution_acceleration",
                params,
                "use_norm_hyperbolic",
                true,
            )?;
            let input = match req.data {
                IndicatorDataRef::Candles { candles, .. } => {
                    LeavittConvolutionAccelerationInput::from_candles(
                        candles,
                        &source,
                        LeavittConvolutionAccelerationParams {
                            length: Some(length),
                            norm_length: Some(norm_length),
                            use_norm_hyperbolic: Some(use_norm_hyperbolic),
                        },
                    )
                }
                IndicatorDataRef::Slice { values } => {
                    LeavittConvolutionAccelerationInput::from_slice(
                        values,
                        LeavittConvolutionAccelerationParams {
                            length: Some(length),
                            norm_length: Some(norm_length),
                            use_norm_hyperbolic: Some(use_norm_hyperbolic),
                        },
                    )
                }
                _ => unreachable!(),
            };
            let out =
                leavitt_convolution_acceleration_with_kernel(&input, kernel).map_err(|e| {
                    IndicatorDispatchError::ComputeFailed {
                        indicator: "leavitt_convolution_acceleration".to_string(),
                        details: e.to_string(),
                    }
                })?;
            if output_id.eq_ignore_ascii_case("conv_acceleration")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.conv_acceleration);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "leavitt_convolution_acceleration".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_squeeze_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("squeeze_index", output_id)?;
    let data = extract_slice_input("squeeze_index", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "squeeze_index",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let conv = get_f64_param("squeeze_index", params, "conv", 50.0)?;
            let length = get_usize_param("squeeze_index", params, "length", 20)?;
            let input = SqueezeIndexInput::from_slice(
                data,
                SqueezeIndexParams {
                    conv: Some(conv),
                    length: Some(length),
                },
            );
            let out = squeeze_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "squeeze_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_stochastic_distance_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    if !output_id.eq_ignore_ascii_case("oscillator") && !output_id.eq_ignore_ascii_case("signal") {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "stochastic_distance".to_string(),
            output: output_id.to_string(),
        });
    }
    let data = extract_slice_input("stochastic_distance", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "stochastic_distance",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let lookback_length =
                get_usize_param("stochastic_distance", params, "lookback_length", 200)?;
            let length1 = get_usize_param("stochastic_distance", params, "length1", 12)?;
            let length2 = get_usize_param("stochastic_distance", params, "length2", 3)?;
            let ob_level = get_i32_param("stochastic_distance", params, "ob_level", 40)?;
            let os_level = get_i32_param("stochastic_distance", params, "os_level", -40)?;
            let input = StochasticDistanceInput::from_slice(
                data,
                StochasticDistanceParams {
                    lookback_length: Some(lookback_length),
                    length1: Some(length1),
                    length2: Some(length2),
                    ob_level: Some(ob_level),
                    os_level: Some(os_level),
                },
            );
            let out = stochastic_distance_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "stochastic_distance".to_string(),
                    details: e.to_string(),
                }
            })?;
            match output_id {
                "oscillator" => Ok(out.oscillator),
                "signal" => Ok(out.signal),
                _ => Err(IndicatorDispatchError::UnknownOutput {
                    indicator: "stochastic_distance".to_string(),
                    output: output_id.to_string(),
                }),
            }
        },
    )
}

fn compute_vertical_horizontal_filter_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("vertical_horizontal_filter", output_id)?;
    let data = extract_slice_input("vertical_horizontal_filter", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "vertical_horizontal_filter",
        output_id,
        req.combos,
        data.len(),
        |params| {
            let length = get_usize_param("vertical_horizontal_filter", params, "length", 28)?;
            let input = VerticalHorizontalFilterInput::from_slice(
                data,
                VerticalHorizontalFilterParams {
                    length: Some(length),
                },
            );
            let out = vertical_horizontal_filter_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "vertical_horizontal_filter".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_intraday_momentum_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, _high, _low, close) = extract_ohlc_full_input("intraday_momentum_index", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "intraday_momentum_index",
        output_id,
        req.combos,
        open.len(),
        |params| {
            let length = get_usize_param("intraday_momentum_index", params, "length", 14)?;
            let length_ma = get_usize_param("intraday_momentum_index", params, "length_ma", 6)?;
            let mult = get_f64_param("intraday_momentum_index", params, "mult", 2.0)?;
            let length_bb = get_usize_param("intraday_momentum_index", params, "length_bb", 20)?;
            let apply_smoothing =
                get_bool_param("intraday_momentum_index", params, "apply_smoothing", false)?;
            let low_band = get_usize_param("intraday_momentum_index", params, "low_band", 10)?;
            let input = IntradayMomentumIndexInput::from_slices(
                open,
                close,
                IntradayMomentumIndexParams {
                    length: Some(length),
                    length_ma: Some(length_ma),
                    mult: Some(mult),
                    length_bb: Some(length_bb),
                    apply_smoothing: Some(apply_smoothing),
                    low_band: Some(low_band),
                },
            );
            let out = intraday_momentum_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "intraday_momentum_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("imi") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.imi);
            }
            if output_id.eq_ignore_ascii_case("upper_hit") {
                return Ok(out.upper_hit);
            }
            if output_id.eq_ignore_ascii_case("lower_hit") {
                return Ok(out.lower_hit);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "intraday_momentum_index".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_vwap_zscore_with_signals_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (close, volume) =
        extract_close_volume_input("vwap_zscore_with_signals", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    let field =
        if output_id.eq_ignore_ascii_case("zvwap") || output_id.eq_ignore_ascii_case("value") {
            VwapZscoreWithSignalsOutputField::Zvwap
        } else if output_id.eq_ignore_ascii_case("support_signal") {
            VwapZscoreWithSignalsOutputField::SupportSignal
        } else if output_id.eq_ignore_ascii_case("resistance_signal") {
            VwapZscoreWithSignalsOutputField::ResistanceSignal
        } else {
            return Err(IndicatorDispatchError::UnknownOutput {
                indicator: "vwap_zscore_with_signals".to_string(),
                output: output_id.to_string(),
            });
        };
    let rows = req.combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "vwap_zscore_with_signals".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
        let length = get_usize_param("vwap_zscore_with_signals", params, "length", 20)?;
        let upper_bottom = get_f64_param("vwap_zscore_with_signals", params, "upper_bottom", 2.5)?;
        let lower_bottom = get_f64_param("vwap_zscore_with_signals", params, "lower_bottom", -2.5)?;
        let input = VwapZscoreWithSignalsInput::from_slices(
            close,
            volume,
            VwapZscoreWithSignalsParams {
                length: Some(length),
                upper_bottom: Some(upper_bottom),
                lower_bottom: Some(lower_bottom),
            },
        );
        let start = row * cols;
        let end = start + cols;
        vwap_zscore_with_signals_output_into_slice(&mut matrix[start..end], &input, kernel, field)
            .map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: "vwap_zscore_with_signals".to_string(),
                details: e.to_string(),
            })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
}

fn compute_hema_trend_levels_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("hema_trend_levels", req.data)?;
    let kernel = req.kernel.to_non_batch();
    let field =
        if output_id.eq_ignore_ascii_case("fast_hema") || output_id.eq_ignore_ascii_case("value") {
            HemaTrendLevelsOutputField::FastHema
        } else if output_id.eq_ignore_ascii_case("slow_hema") {
            HemaTrendLevelsOutputField::SlowHema
        } else if output_id.eq_ignore_ascii_case("trend_direction")
            || output_id.eq_ignore_ascii_case("trend")
        {
            HemaTrendLevelsOutputField::TrendDirection
        } else if output_id.eq_ignore_ascii_case("bar_state") {
            HemaTrendLevelsOutputField::BarState
        } else if output_id.eq_ignore_ascii_case("bullish_crossover")
            || output_id.eq_ignore_ascii_case("buy_signal")
            || output_id.eq_ignore_ascii_case("buy")
        {
            HemaTrendLevelsOutputField::BullishCrossover
        } else if output_id.eq_ignore_ascii_case("bearish_crossunder")
            || output_id.eq_ignore_ascii_case("sell_signal")
            || output_id.eq_ignore_ascii_case("sell")
        {
            HemaTrendLevelsOutputField::BearishCrossunder
        } else if output_id.eq_ignore_ascii_case("box_offset") {
            HemaTrendLevelsOutputField::BoxOffset
        } else if output_id.eq_ignore_ascii_case("bull_box_top") {
            HemaTrendLevelsOutputField::BullBoxTop
        } else if output_id.eq_ignore_ascii_case("bull_box_bottom") {
            HemaTrendLevelsOutputField::BullBoxBottom
        } else if output_id.eq_ignore_ascii_case("bear_box_top") {
            HemaTrendLevelsOutputField::BearBoxTop
        } else if output_id.eq_ignore_ascii_case("bear_box_bottom") {
            HemaTrendLevelsOutputField::BearBoxBottom
        } else if output_id.eq_ignore_ascii_case("bullish_test") {
            HemaTrendLevelsOutputField::BullishTest
        } else if output_id.eq_ignore_ascii_case("bearish_test") {
            HemaTrendLevelsOutputField::BearishTest
        } else if output_id.eq_ignore_ascii_case("bullish_test_level") {
            HemaTrendLevelsOutputField::BullishTestLevel
        } else if output_id.eq_ignore_ascii_case("bearish_test_level") {
            HemaTrendLevelsOutputField::BearishTestLevel
        } else {
            return Err(IndicatorDispatchError::UnknownOutput {
                indicator: "hema_trend_levels".to_string(),
                output: output_id.to_string(),
            });
        };
    collect_f64_into_rows(
        "hema_trend_levels",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
            let fast_length = get_usize_param("hema_trend_levels", params, "fast_length", 20)?;
            let slow_length = get_usize_param("hema_trend_levels", params, "slow_length", 40)?;
            let input = HemaTrendLevelsInput::from_slices(
                open,
                high,
                low,
                close,
                HemaTrendLevelsParams {
                    fast_length: Some(fast_length),
                    slow_length: Some(slow_length),
                },
            );
            hema_trend_levels_output_into_slice(row, &input, kernel, field).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "hema_trend_levels".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(())
        },
    )
}

fn compute_macd_wave_signal_pro_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close) = extract_ohlc_full_input("macd_wave_signal_pro", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "macd_wave_signal_pro",
        output_id,
        req.combos,
        close.len(),
        |_params| {
            let input =
                MacdWaveSignalProInput::from_slices(open, high, low, close, Default::default());
            let out = macd_wave_signal_pro_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "macd_wave_signal_pro".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("diff") || output_id.eq_ignore_ascii_case("value") {
                return Ok(out.diff);
            }
            if output_id.eq_ignore_ascii_case("dea") {
                return Ok(out.dea);
            }
            if output_id.eq_ignore_ascii_case("macd_histogram")
                || output_id.eq_ignore_ascii_case("macd")
                || output_id.eq_ignore_ascii_case("histogram")
                || output_id.eq_ignore_ascii_case("hist")
            {
                return Ok(out.macd_histogram);
            }
            if output_id.eq_ignore_ascii_case("line_convergence")
                || output_id.eq_ignore_ascii_case("line_conv")
            {
                return Ok(out.line_convergence);
            }
            if output_id.eq_ignore_ascii_case("buy_signal") || output_id.eq_ignore_ascii_case("buy")
            {
                return Ok(out.buy_signal);
            }
            if output_id.eq_ignore_ascii_case("sell_signal")
                || output_id.eq_ignore_ascii_case("sell")
            {
                return Ok(out.sell_signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "macd_wave_signal_pro".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_demand_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close, volume) = extract_hlcv_input("demand_index", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "demand_index",
        output_id,
        req.combos,
        high.len(),
        |params| {
            let len_bs = get_usize_param("demand_index", params, "len_bs", 19)?;
            let len_bs_ma = get_usize_param("demand_index", params, "len_bs_ma", 19)?;
            let len_di_ma = get_usize_param("demand_index", params, "len_di_ma", 19)?;
            let ma_type = get_enum_param("demand_index", params, "ma_type", "ema")?;
            let input = DemandIndexInput::from_slices(
                high,
                low,
                close,
                volume,
                DemandIndexParams {
                    len_bs: Some(len_bs),
                    len_bs_ma: Some(len_bs_ma),
                    len_di_ma: Some(len_di_ma),
                    ma_type: Some(ma_type),
                },
            );
            let out = demand_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "demand_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            if output_id.eq_ignore_ascii_case("demand_index")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.demand_index);
            }
            if output_id.eq_ignore_ascii_case("signal") {
                return Ok(out.signal);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "demand_index".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_kase_peak_oscillator_with_divergences_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low, close) = extract_ohlc_input("kase_peak_oscillator_with_divergences", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "kase_peak_oscillator_with_divergences",
        output_id,
        req.combos,
        close.len(),
        |params| {
            let deviations = get_f64_param(
                "kase_peak_oscillator_with_divergences",
                params,
                "deviations",
                2.0,
            )?;
            let short_cycle = get_usize_param(
                "kase_peak_oscillator_with_divergences",
                params,
                "short_cycle",
                8,
            )?;
            let long_cycle = get_usize_param(
                "kase_peak_oscillator_with_divergences",
                params,
                "long_cycle",
                65,
            )?;
            let sensitivity = get_f64_param(
                "kase_peak_oscillator_with_divergences",
                params,
                "sensitivity",
                40.0,
            )?;
            let all_peaks_mode = get_bool_param(
                "kase_peak_oscillator_with_divergences",
                params,
                "all_peaks_mode",
                true,
            )?;
            let lb_r = get_usize_param("kase_peak_oscillator_with_divergences", params, "lb_r", 5)?;
            let lb_l = get_usize_param("kase_peak_oscillator_with_divergences", params, "lb_l", 5)?;
            let range_upper = get_usize_param(
                "kase_peak_oscillator_with_divergences",
                params,
                "range_upper",
                60,
            )?;
            let range_lower = get_usize_param(
                "kase_peak_oscillator_with_divergences",
                params,
                "range_lower",
                5,
            )?;
            let plot_bull = get_bool_param(
                "kase_peak_oscillator_with_divergences",
                params,
                "plot_bull",
                true,
            )?;
            let plot_hidden_bull = get_bool_param(
                "kase_peak_oscillator_with_divergences",
                params,
                "plot_hidden_bull",
                false,
            )?;
            let plot_bear = get_bool_param(
                "kase_peak_oscillator_with_divergences",
                params,
                "plot_bear",
                true,
            )?;
            let plot_hidden_bear = get_bool_param(
                "kase_peak_oscillator_with_divergences",
                params,
                "plot_hidden_bear",
                false,
            )?;
            let input = KasePeakOscillatorWithDivergencesInput::from_slices(
                high,
                low,
                close,
                KasePeakOscillatorWithDivergencesParams {
                    deviations: Some(deviations),
                    short_cycle: Some(short_cycle),
                    long_cycle: Some(long_cycle),
                    sensitivity: Some(sensitivity),
                    all_peaks_mode: Some(all_peaks_mode),
                    lb_r: Some(lb_r),
                    lb_l: Some(lb_l),
                    range_upper: Some(range_upper),
                    range_lower: Some(range_lower),
                    plot_bull: Some(plot_bull),
                    plot_hidden_bull: Some(plot_hidden_bull),
                    plot_bear: Some(plot_bear),
                    plot_hidden_bear: Some(plot_hidden_bear),
                },
            );
            let out =
                kase_peak_oscillator_with_divergences_with_kernel(&input, kernel).map_err(|e| {
                    IndicatorDispatchError::ComputeFailed {
                        indicator: "kase_peak_oscillator_with_divergences".to_string(),
                        details: e.to_string(),
                    }
                })?;
            if output_id.eq_ignore_ascii_case("oscillator")
                || output_id.eq_ignore_ascii_case("value")
            {
                return Ok(out.oscillator);
            }
            if output_id.eq_ignore_ascii_case("hist") || output_id.eq_ignore_ascii_case("histogram")
            {
                return Ok(out.histogram);
            }
            if output_id.eq_ignore_ascii_case("max_peak_value") {
                return Ok(out.max_peak_value);
            }
            if output_id.eq_ignore_ascii_case("min_peak_value") {
                return Ok(out.min_peak_value);
            }
            if output_id.eq_ignore_ascii_case("market_extreme") {
                return Ok(out.market_extreme);
            }
            if output_id.eq_ignore_ascii_case("regular_bullish") {
                return Ok(out.regular_bullish);
            }
            if output_id.eq_ignore_ascii_case("hidden_bullish") {
                return Ok(out.hidden_bullish);
            }
            if output_id.eq_ignore_ascii_case("regular_bearish") {
                return Ok(out.regular_bearish);
            }
            if output_id.eq_ignore_ascii_case("hidden_bearish") {
                return Ok(out.hidden_bearish);
            }
            if output_id.eq_ignore_ascii_case("go_long") {
                return Ok(out.go_long);
            }
            if output_id.eq_ignore_ascii_case("go_short") {
                return Ok(out.go_short);
            }
            Err(IndicatorDispatchError::UnknownOutput {
                indicator: "kase_peak_oscillator_with_divergences".to_string(),
                output: output_id.to_string(),
            })
        },
    )
}

fn compute_gopalakrishnan_range_index_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("gopalakrishnan_range_index", output_id)?;
    let (high, low) = extract_high_low_input("gopalakrishnan_range_index", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64(
        "gopalakrishnan_range_index",
        output_id,
        req.combos,
        high.len(),
        |params| {
            let length = get_usize_param("gopalakrishnan_range_index", params, "length", 5)?;
            let input = GopalakrishnanRangeIndexInput::from_slices(
                high,
                low,
                GopalakrishnanRangeIndexParams {
                    length: Some(length),
                },
            );
            let out = gopalakrishnan_range_index_with_kernel(&input, kernel).map_err(|e| {
                IndicatorDispatchError::ComputeFailed {
                    indicator: "gopalakrishnan_range_index".to_string(),
                    details: e.to_string(),
                }
            })?;
            Ok(out.values)
        },
    )
}

fn compute_acosc_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (high, low) = extract_high_low_input("acosc", req.data)?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("osc") || output_id.eq_ignore_ascii_case("value")
    {
        AcoscOutputField::Osc
    } else if output_id.eq_ignore_ascii_case("change") {
        AcoscOutputField::Change
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "acosc".to_string(),
            output: output_id.to_string(),
        });
    };
    let rows = req.combos.len();
    let cols = high.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "acosc".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for row in 0..rows {
        let input = AcoscInput::from_slices(high, low, AcoscParams::default());
        let start = row * cols;
        let end = start + cols;
        acosc_output_into_slice(&mut matrix[start..end], &input, kernel, field).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "acosc".to_string(),
                details: e.to_string(),
            }
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
}

fn compute_alligator_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("alligator", req.data, "hl2")?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("jaw") || output_id.eq_ignore_ascii_case("value")
    {
        AlligatorOutputField::Jaw
    } else if output_id.eq_ignore_ascii_case("teeth") {
        AlligatorOutputField::Teeth
    } else if output_id.eq_ignore_ascii_case("lips") {
        AlligatorOutputField::Lips
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "alligator".to_string(),
            output: output_id.to_string(),
        });
    };

    let rows = req.combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "alligator".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
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
        let start = row * cols;
        let end = start + cols;
        alligator_output_into_slice(&mut matrix[start..end], &input, kernel, field).map_err(
            |e| IndicatorDispatchError::ComputeFailed {
                indicator: "alligator".to_string(),
                details: e.to_string(),
            },
        )?;
    }

    Ok(f64_output(output_id, rows, cols, matrix))
}

fn compute_alphatrend_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let (open, high, low, close, volume) = extract_ohlcv_full_input("alphatrend", req.data)?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("k1") || output_id.eq_ignore_ascii_case("value") {
        AlphaTrendOutputField::K1
    } else if output_id.eq_ignore_ascii_case("k2") {
        AlphaTrendOutputField::K2
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "alphatrend".to_string(),
            output: output_id.to_string(),
        });
    };

    let rows = req.combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "alphatrend".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
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
        let start = row * cols;
        let end = start + cols;
        alphatrend_output_into_slice(&mut matrix[start..end], &input, kernel, field).map_err(
            |e| IndicatorDispatchError::ComputeFailed {
                indicator: "alphatrend".to_string(),
                details: e.to_string(),
            },
        )?;
    }

    Ok(f64_output(output_id, rows, cols, matrix))
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
    let field =
        if output_id.eq_ignore_ascii_case("bulls") || output_id.eq_ignore_ascii_case("value") {
            AsoOutputField::Bulls
        } else if output_id.eq_ignore_ascii_case("bears") {
            AsoOutputField::Bears
        } else {
            return Err(IndicatorDispatchError::UnknownOutput {
                indicator: "aso".to_string(),
                output: output_id.to_string(),
            });
        };

    let rows = req.combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "aso".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
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
        let start = row * cols;
        let end = start + cols;
        aso_output_into_slice(&mut matrix[start..end], &input, kernel, field).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "aso".to_string(),
                details: e.to_string(),
            }
        })?;
    }

    Ok(f64_output(output_id, rows, cols, matrix))
}

fn compute_avsl_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    expect_value_output("avsl", output_id)?;
    let (_high, low, close, volume) = extract_hlcv_input("avsl", req.data)?;
    let kernel = req.kernel.to_non_batch();
    collect_f64_into_rows("avsl", output_id, req.combos, close.len(), |params, row| {
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
        avsl_into_slice(row, &input, kernel).map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "avsl".to_string(),
            details: e.to_string(),
        })
    })
}

fn compute_bandpass_batch(
    req: IndicatorBatchRequest<'_>,
    output_id: &str,
) -> Result<IndicatorBatchOutput, IndicatorDispatchError> {
    let data = extract_slice_input("bandpass", req.data, "close")?;
    let kernel = req.kernel.to_non_batch();
    let field = if output_id.eq_ignore_ascii_case("bp") || output_id.eq_ignore_ascii_case("value") {
        BandPassOutputField::Bp
    } else if output_id.eq_ignore_ascii_case("bp_normalized")
        || output_id.eq_ignore_ascii_case("normalized")
    {
        BandPassOutputField::BpNormalized
    } else if output_id.eq_ignore_ascii_case("signal") {
        BandPassOutputField::Signal
    } else if output_id.eq_ignore_ascii_case("trigger") {
        BandPassOutputField::Trigger
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "bandpass".to_string(),
            output: output_id.to_string(),
        });
    };
    let rows = req.combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "bandpass".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
        let period = get_usize_param("bandpass", params, "period", 20)?;
        let bandwidth = get_f64_param("bandpass", params, "bandwidth", 0.3)?;
        let input = BandPassInput::from_slice(
            data,
            BandPassParams {
                period: Some(period),
                bandwidth: Some(bandwidth),
            },
        );
        let start = row * cols;
        let end = start + cols;
        bandpass_output_into_slice(&mut matrix[start..end], &input, kernel, field).map_err(
            |e| IndicatorDispatchError::ComputeFailed {
                indicator: "bandpass".to_string(),
                details: e.to_string(),
            },
        )?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
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
    let field = if output_id.eq_ignore_ascii_case("dvdi") || output_id.eq_ignore_ascii_case("value")
    {
        DvdiqqeOutputField::Dvdi
    } else if output_id.eq_ignore_ascii_case("fast_tl") || output_id.eq_ignore_ascii_case("fast") {
        DvdiqqeOutputField::FastTl
    } else if output_id.eq_ignore_ascii_case("slow_tl") || output_id.eq_ignore_ascii_case("slow") {
        DvdiqqeOutputField::SlowTl
    } else if output_id.eq_ignore_ascii_case("center_line")
        || output_id.eq_ignore_ascii_case("center")
    {
        DvdiqqeOutputField::CenterLine
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "dvdiqqe".to_string(),
            output: output_id.to_string(),
        });
    };

    let rows = req.combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "dvdiqqe".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
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
        let start = row * cols;
        let end = start + cols;
        dvdiqqe_output_into_slice(&mut matrix[start..end], &input, kernel, field).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "dvdiqqe".to_string(),
                details: e.to_string(),
            }
        })?;
    }

    Ok(f64_output(output_id, rows, cols, matrix))
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
    let field = if output_id.eq_ignore_ascii_case("value_trend")
        || output_id.eq_ignore_ascii_case("value")
    {
        CyberpunkValueTrendAnalyzerOutputField::ValueTrend
    } else if output_id.eq_ignore_ascii_case("value_trend_lag")
        || output_id.eq_ignore_ascii_case("lag")
    {
        CyberpunkValueTrendAnalyzerOutputField::ValueTrendLag
    } else if output_id.eq_ignore_ascii_case("deviation_index") {
        CyberpunkValueTrendAnalyzerOutputField::DeviationIndex
    } else if output_id.eq_ignore_ascii_case("overbought_signal")
        || output_id.eq_ignore_ascii_case("overbought")
    {
        CyberpunkValueTrendAnalyzerOutputField::OverboughtSignal
    } else if output_id.eq_ignore_ascii_case("buy_signal") {
        CyberpunkValueTrendAnalyzerOutputField::BuySignal
    } else if output_id.eq_ignore_ascii_case("sell_signal") {
        CyberpunkValueTrendAnalyzerOutputField::SellSignal
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "cyberpunk_value_trend_analyzer".to_string(),
            output: output_id.to_string(),
        });
    };
    collect_f64_into_rows(
        "cyberpunk_value_trend_analyzer",
        output_id,
        req.combos,
        close.len(),
        |params, row| {
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
            cyberpunk_value_trend_analyzer_output_into_slice(row, &input, kernel, field).map_err(
                |e| IndicatorDispatchError::ComputeFailed {
                    indicator: "cyberpunk_value_trend_analyzer".to_string(),
                    details: e.to_string(),
                },
            )?;
            Ok(())
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
            let vol_mult = get_f64_param("market_structure_confluence", params, "vol_mult", 2.0)?;
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
    let field = if output_id.eq_ignore_ascii_case("rsi") || output_id.eq_ignore_ascii_case("value")
    {
        VolumeWeightedRelativeStrengthIndexOutputField::Rsi
    } else if output_id.eq_ignore_ascii_case("consolidation_strength")
        || output_id.eq_ignore_ascii_case("consolidation")
    {
        VolumeWeightedRelativeStrengthIndexOutputField::ConsolidationStrength
    } else if output_id.eq_ignore_ascii_case("rsi_ma") || output_id.eq_ignore_ascii_case("ma") {
        VolumeWeightedRelativeStrengthIndexOutputField::RsiMa
    } else if output_id.eq_ignore_ascii_case("bearish_tp") {
        VolumeWeightedRelativeStrengthIndexOutputField::BearishTp
    } else if output_id.eq_ignore_ascii_case("bullish_tp") {
        VolumeWeightedRelativeStrengthIndexOutputField::BullishTp
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "volume_weighted_relative_strength_index".to_string(),
            output: output_id.to_string(),
        });
    };

    let rows = req.combos.len();
    let cols = source.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "volume_weighted_relative_strength_index".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
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
        let start = row * cols;
        let end = start + cols;
        volume_weighted_relative_strength_index_output_into_slice(
            &input,
            kernel,
            field,
            &mut matrix[start..end],
        )
        .map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: "volume_weighted_relative_strength_index".to_string(),
            details: e.to_string(),
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
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
    let field = if output_id.eq_ignore_ascii_case("voss") || output_id.eq_ignore_ascii_case("value")
    {
        VossOutputField::Voss
    } else if output_id.eq_ignore_ascii_case("filt") || output_id.eq_ignore_ascii_case("filter") {
        VossOutputField::Filt
    } else {
        return Err(IndicatorDispatchError::UnknownOutput {
            indicator: "voss".to_string(),
            output: output_id.to_string(),
        });
    };
    let rows = req.combos.len();
    let cols = data.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "voss".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for (row, combo) in req.combos.iter().enumerate() {
        let params = combo.params;
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
        let start = row * cols;
        let end = start + cols;
        voss_output_into_slice(&mut matrix[start..end], &input, kernel, field).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "voss".to_string(),
                details: e.to_string(),
            }
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
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
    let rows = req.combos.len();
    let cols = close.len();
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| IndicatorDispatchError::ComputeFailed {
            indicator: "wad".to_string(),
            details: "rows*cols overflow".to_string(),
        })?;
    let mut matrix = alloc_uninit_f64(total);
    for row in 0..rows {
        let input = WadInput::from_slices(high, low, close);
        let start = row * cols;
        let end = start + cols;
        wad_into_slice(&mut matrix[start..end], &input, kernel).map_err(|e| {
            IndicatorDispatchError::ComputeFailed {
                indicator: "wad".to_string(),
                details: e.to_string(),
            }
        })?;
    }
    Ok(f64_output(output_id, rows, cols, matrix))
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
    use crate::indicators::absolute_strength_index_oscillator::{
        absolute_strength_index_oscillator_with_kernel, AbsoluteStrengthIndexOscillatorInput,
        AbsoluteStrengthIndexOscillatorParams,
    };
    use crate::indicators::accumulation_swing_index::{
        accumulation_swing_index_with_kernel, AccumulationSwingIndexInput,
        AccumulationSwingIndexParams,
    };
    use crate::indicators::ad::{ad_with_kernel, AdInput, AdParams};
    use crate::indicators::adaptive_bandpass_trigger_oscillator::{
        adaptive_bandpass_trigger_oscillator_with_kernel, AdaptiveBandpassTriggerOscillatorInput,
        AdaptiveBandpassTriggerOscillatorParams,
    };
    use crate::indicators::advance_decline_line::{
        advance_decline_line_with_kernel, AdvanceDeclineLineInput, AdvanceDeclineLineParams,
    };
    use crate::indicators::adx::{adx_with_kernel, AdxInput, AdxParams};
    use crate::indicators::ao::{ao_with_kernel, AoInput, AoParams};
    use crate::indicators::apo::{apo_with_kernel, ApoInput, ApoParams};
    use crate::indicators::atr_percentile::{
        atr_percentile_with_kernel, AtrPercentileInput, AtrPercentileParams,
    };
    use crate::indicators::bull_power_vs_bear_power::{
        bull_power_vs_bear_power_with_kernel, BullPowerVsBearPowerInput, BullPowerVsBearPowerParams,
    };
    use crate::indicators::cg::{cg_with_kernel, CgInput, CgParams};
    use crate::indicators::cmo::{cmo_with_kernel, CmoInput, CmoParams};
    use crate::indicators::cycle_channel_oscillator::{
        cycle_channel_oscillator_with_kernel, CycleChannelOscillatorInput,
        CycleChannelOscillatorParams,
    };
    use crate::indicators::daily_factor::{
        daily_factor_with_kernel, DailyFactorInput, DailyFactorParams,
    };
    use crate::indicators::decisionpoint_breadth_swenlin_trading_oscillator::{
        decisionpoint_breadth_swenlin_trading_oscillator_with_kernel,
        DecisionPointBreadthSwenlinTradingOscillatorInput,
        DecisionPointBreadthSwenlinTradingOscillatorParams,
    };
    use crate::indicators::demand_index::{
        demand_index_with_kernel, DemandIndexInput, DemandIndexParams,
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
    use crate::indicators::ehlers_linear_extrapolation_predictor::{
        ehlers_linear_extrapolation_predictor_with_kernel, EhlersLinearExtrapolationPredictorInput,
        EhlersLinearExtrapolationPredictorParams,
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
    use crate::indicators::fibonacci_entry_bands::{
        fibonacci_entry_bands_with_kernel, FibonacciEntryBandsInput, FibonacciEntryBandsParams,
    };
    use crate::indicators::fibonacci_trailing_stop::{
        fibonacci_trailing_stop_with_kernel, FibonacciTrailingStopInput,
        FibonacciTrailingStopParams,
    };
    use crate::indicators::fosc::{fosc_with_kernel, FoscInput, FoscParams};
    use crate::indicators::garman_klass_volatility::{
        garman_klass_volatility_with_kernel, GarmanKlassVolatilityInput,
        GarmanKlassVolatilityParams,
    };
    use crate::indicators::gopalakrishnan_range_index::{
        gopalakrishnan_range_index_with_kernel, GopalakrishnanRangeIndexInput,
        GopalakrishnanRangeIndexParams,
    };
    use crate::indicators::grover_llorens_cycle_oscillator::{
        grover_llorens_cycle_oscillator_with_kernel, GroverLlorensCycleOscillatorInput,
        GroverLlorensCycleOscillatorParams,
    };
    use crate::indicators::hema_trend_levels::{
        hema_trend_levels_with_kernel, HemaTrendLevelsInput, HemaTrendLevelsParams,
    };
    use crate::indicators::historical_volatility::{
        historical_volatility_with_kernel, HistoricalVolatilityInput, HistoricalVolatilityParams,
    };
    use crate::indicators::historical_volatility_percentile::{
        historical_volatility_percentile_with_kernel, HistoricalVolatilityPercentileInput,
        HistoricalVolatilityPercentileParams,
    };
    use crate::indicators::hull_butterfly_oscillator::{
        hull_butterfly_oscillator_with_kernel, HullButterflyOscillatorInput,
        HullButterflyOscillatorParams,
    };
    use crate::indicators::ichimoku_oscillator::{
        ichimoku_oscillator_with_kernel, IchimokuOscillatorInput, IchimokuOscillatorNormalizeMode,
        IchimokuOscillatorParams,
    };
    use crate::indicators::ift_rsi::{ift_rsi_with_kernel, IftRsiInput, IftRsiParams};
    use crate::indicators::intraday_momentum_index::{
        intraday_momentum_index_with_kernel, IntradayMomentumIndexInput,
        IntradayMomentumIndexParams,
    };
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
    use crate::indicators::macd_wave_signal_pro::{
        macd_wave_signal_pro_with_kernel, MacdWaveSignalProInput,
    };
    use crate::indicators::mean_ad::{mean_ad_with_kernel, MeanAdInput, MeanAdParams};
    use crate::indicators::medprice::{medprice_with_kernel, MedpriceInput, MedpriceParams};
    use crate::indicators::mesa_stochastic_multi_length::{
        mesa_stochastic_multi_length_with_kernel, MesaStochasticMultiLengthInput,
        MesaStochasticMultiLengthParams,
    };
    use crate::indicators::mfi::{
        mfi_batch_with_kernel, mfi_with_kernel, MfiBatchRange, MfiInput, MfiParams,
    };
    use crate::indicators::monotonicity_index::{
        monotonicity_index_with_kernel, MonotonicityIndexInput, MonotonicityIndexMode,
        MonotonicityIndexParams,
    };
    use crate::indicators::moving_averages::ma::MaData;
    use crate::indicators::moving_averages::ma_batch::{
        ma_batch_with_kernel_and_typed_params, MaBatchParamKV, MaBatchParamValue,
    };
    use crate::indicators::multi_length_stochastic_average::{
        multi_length_stochastic_average_with_kernel, MultiLengthStochasticAverageInput,
        MultiLengthStochasticAverageParams,
    };
    use crate::indicators::natr::{natr_with_kernel, NatrInput, NatrParams};
    use crate::indicators::neighboring_trailing_stop::{
        neighboring_trailing_stop_with_kernel, NeighboringTrailingStopInput,
        NeighboringTrailingStopParams,
    };
    use crate::indicators::percentile_nearest_rank::{
        percentile_nearest_rank_with_kernel, PercentileNearestRankInput,
        PercentileNearestRankParams,
    };
    use crate::indicators::ppo::{ppo_with_kernel, PpoInput, PpoParams};
    use crate::indicators::premier_rsi_oscillator::{
        premier_rsi_oscillator_with_kernel, PremierRsiOscillatorInput, PremierRsiOscillatorParams,
    };
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
    use crate::indicators::spearman_correlation::{
        spearman_correlation_with_kernel, SpearmanCorrelationInput, SpearmanCorrelationParams,
    };
    use crate::indicators::squeeze_index::{
        squeeze_index_with_kernel, SqueezeIndexInput, SqueezeIndexParams,
    };
    use crate::indicators::stochastic_distance::{
        stochastic_distance_with_kernel, StochasticDistanceInput, StochasticDistanceParams,
    };
    use crate::indicators::trend_trigger_factor::{
        trend_trigger_factor_with_kernel, TrendTriggerFactorInput, TrendTriggerFactorParams,
    };
    use crate::indicators::trix::{
        trix_batch_with_kernel, trix_with_kernel, TrixBatchRange, TrixInput, TrixParams,
    };
    use crate::indicators::ttm_trend::{ttm_trend_with_kernel, TtmTrendInput, TtmTrendParams};
    use crate::indicators::velocity_acceleration_convergence_divergence_indicator::{
        velocity_acceleration_convergence_divergence_indicator_with_kernel,
        VelocityAccelerationConvergenceDivergenceIndicatorInput,
        VelocityAccelerationConvergenceDivergenceIndicatorParams,
    };
    use crate::indicators::velocity_acceleration_indicator::{
        velocity_acceleration_indicator_with_kernel, VelocityAccelerationIndicatorInput,
        VelocityAccelerationIndicatorParams,
    };
    use crate::indicators::volatility_quality_index::{
        volatility_quality_index_with_kernel, VolatilityQualityIndexInput,
        VolatilityQualityIndexParams,
    };
    use crate::indicators::volatility_ratio_adaptive_rsx::{
        volatility_ratio_adaptive_rsx_with_kernel, VolatilityRatioAdaptiveRsxInput,
        VolatilityRatioAdaptiveRsxParams,
    };
    use crate::indicators::volume_energy_reservoirs::{
        volume_energy_reservoirs_with_kernel, VolumeEnergyReservoirsInput,
        VolumeEnergyReservoirsParams,
    };
    use crate::indicators::volume_zone_oscillator::{
        volume_zone_oscillator_with_kernel, VolumeZoneOscillatorInput, VolumeZoneOscillatorParams,
    };
    use crate::indicators::vpci::{vpci_with_kernel, VpciInput, VpciParams};
    use crate::indicators::vwap_deviation_oscillator::{
        vwap_deviation_oscillator_with_kernel, VwapDeviationMode, VwapDeviationOscillatorInput,
        VwapDeviationOscillatorParams, VwapDeviationSessionMode,
    };
    use crate::indicators::vwap_zscore_with_signals::{
        vwap_zscore_with_signals_with_kernel, VwapZscoreWithSignalsInput,
        VwapZscoreWithSignalsParams,
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
    fn garman_klass_output_matches_direct() {
        let (open, high, low, close) = sample_ohlc();
        let combo = [ParamKV {
            key: "lookback",
            value: ParamValue::Int(17),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "garman_klass_volatility",
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
        let got = out.values_f64.unwrap();
        let input = GarmanKlassVolatilityInput::from_slices(
            &open,
            &high,
            &low,
            &close,
            GarmanKlassVolatilityParams { lookback: Some(17) },
        );
        let direct = garman_klass_volatility_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .values;
        assert_series_eq(&got, &direct, 1e-12);
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
    fn historical_volatility_percentile_signal_output_matches_direct() {
        let data = sample_series();
        let combo = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(5),
            },
            ParamKV {
                key: "annual_length",
                value: ParamValue::Int(10),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "historical_volatility_percentile",
            output_id: Some("hvp_sma"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = HistoricalVolatilityPercentileInput::from_slice(
            &data,
            HistoricalVolatilityPercentileParams {
                length: Some(5),
                annual_length: Some(10),
            },
        );
        let direct =
            historical_volatility_percentile_with_kernel(&input, Kernel::Auto.to_non_batch())
                .unwrap()
                .hvp_sma;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn volatility_ratio_adaptive_rsx_signal_output_matches_direct() {
        let data = sample_series();
        let combo = [
            ParamKV {
                key: "period",
                value: ParamValue::Int(6),
            },
            ParamKV {
                key: "speed",
                value: ParamValue::Float(0.5),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "volatility_ratio_adaptive_rsx",
            output_id: Some("signal"),
            data: IndicatorDataRef::Slice { values: &data },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = VolatilityRatioAdaptiveRsxInput::from_slice(
            &data,
            VolatilityRatioAdaptiveRsxParams {
                period: Some(6),
                speed: Some(0.5),
            },
        );
        let direct = volatility_ratio_adaptive_rsx_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .signal;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn on_balance_volume_oscillator_signal_output_matches_direct() {
        let close = sample_series();
        let volume: Vec<f64> = (0..close.len()).map(|i| 1000.0 + i as f64 * 3.0).collect();
        let combo = [
            ParamKV {
                key: "obv_length",
                value: ParamValue::Int(20),
            },
            ParamKV {
                key: "ema_length",
                value: ParamValue::Int(9),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "on_balance_volume_oscillator",
            output_id: Some("signal"),
            data: IndicatorDataRef::CloseVolume {
                close: &close,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = OnBalanceVolumeOscillatorInput::from_slices(
            &close,
            &volume,
            OnBalanceVolumeOscillatorParams {
                obv_length: Some(20),
                ema_length: Some(9),
            },
        );
        let direct = on_balance_volume_oscillator_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .signal;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn twiggs_money_flow_smoothed_output_matches_direct() {
        let open = vec![10.0, 10.2, 10.4, 10.7, 10.9, 11.1, 11.3, 11.5, 11.7, 11.9];
        let high = vec![10.4, 10.7, 10.9, 11.1, 11.4, 11.6, 11.8, 12.0, 12.2, 12.4];
        let low = vec![9.8, 10.0, 10.2, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5, 11.7];
        let close = vec![10.1, 10.5, 10.7, 10.9, 11.2, 11.4, 11.6, 11.8, 12.0, 12.2];
        let volume = vec![
            1000.0, 1015.0, 1030.0, 1045.0, 1060.0, 1075.0, 1090.0, 1105.0, 1120.0, 1135.0,
        ];
        let combo = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(5),
            },
            ParamKV {
                key: "smoothing_length",
                value: ParamValue::Int(4),
            },
            ParamKV {
                key: "ma_type",
                value: ParamValue::EnumString("WMA"),
            },
        ];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "twiggs_money_flow",
            output_id: Some("smoothed"),
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
        let input = TwiggsMoneyFlowInput::from_slices(
            &high,
            &low,
            &close,
            &volume,
            TwiggsMoneyFlowParams {
                length: Some(5),
                smoothing_length: Some(4),
                ma_type: Some("WMA".to_string()),
            },
        );
        let direct = twiggs_money_flow_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .smoothed;
        let got = out.values_f64.unwrap();
        assert_series_eq(&got, &direct, 1e-12);
    }

    #[test]
    fn parkinson_variance_output_matches_direct() {
        let (_open, high, low, _close) = sample_ohlc();
        let combo = [ParamKV {
            key: "period",
            value: ParamValue::Int(9),
        }];
        let combos = [IndicatorParamSet { params: &combo }];
        let req = IndicatorBatchRequest {
            indicator_id: "parkinson_volatility",
            output_id: Some("variance"),
            data: IndicatorDataRef::HighLow {
                high: &high,
                low: &low,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        };
        let out = compute_cpu_batch(req).unwrap();
        let input = ParkinsonVolatilityInput::from_slices(
            &high,
            &low,
            ParkinsonVolatilityParams { period: Some(9) },
        );
        let direct = parkinson_volatility_with_kernel(&input, Kernel::Auto.to_non_batch())
            .unwrap()
            .variance;
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
                dispatch_cpu_batch_by_indicator(req, info.id, output_id)
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
                    dispatch_cpu_batch_by_indicator(req, info.id, output_id)
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
                let direct = dispatch_cpu_batch_by_indicator(req, info.id, output.id)
                    .unwrap_or_else(|e| {
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

    #[test]
    fn compute_cpu_batch_bull_power_vs_bear_power_matches_direct() {
        let open: Vec<f64> = (0..256)
            .map(|i| 100.0 + (i as f64 * 0.03).sin() + i as f64 * 0.02)
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + (i as f64 * 0.025).cos() * 0.8)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.5 + (i as f64 * 0.013).sin().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.4 - (i as f64 * 0.017).cos().abs() * 0.15)
            .collect();
        let params = [ParamKV {
            key: "period",
            value: ParamValue::Int(5),
        }];
        let combos = [IndicatorParamSet { params: &params }];
        let candles = crate::utilities::data_loader::Candles::new(
            vec![0; close.len()],
            open.clone(),
            high.clone(),
            low.clone(),
            close.clone(),
            vec![0.0; close.len()],
        );

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "bull_power_vs_bear_power",
            output_id: Some("value"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: None,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = bull_power_vs_bear_power_with_kernel(
            &BullPowerVsBearPowerInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                BullPowerVsBearPowerParams { period: Some(5) },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.values, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_advance_decline_line_matches_direct() {
        let close: Vec<f64> = (0..256)
            .map(|i| ((i as f64) * 0.05).sin() * 100.0 + ((i as f64) * 0.02).cos() * 25.0)
            .collect();
        let combos = [IndicatorParamSet { params: &[] }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "advance_decline_line",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &close },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = advance_decline_line_with_kernel(
            &AdvanceDeclineLineInput::from_slice(&close, AdvanceDeclineLineParams),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.values, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_decisionpoint_breadth_swenlin_trading_oscillator_matches_direct() {
        let advancing: Vec<f64> = (0..256)
            .map(|i| 1500.0 + i as f64 * 0.8 + (i as f64 * 0.07).sin() * 120.0 + 40.0)
            .collect();
        let declining: Vec<f64> = (0..256)
            .map(|i| 1300.0 + i as f64 * 0.5 + (i as f64 * 0.05).cos() * 95.0 + 30.0)
            .collect();
        let combos = [IndicatorParamSet { params: &[] }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "decisionpoint_breadth_swenlin_trading_oscillator",
            output_id: Some("value"),
            data: IndicatorDataRef::HighLow {
                high: &advancing,
                low: &declining,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = decisionpoint_breadth_swenlin_trading_oscillator_with_kernel(
            &DecisionPointBreadthSwenlinTradingOscillatorInput::from_slices(
                &advancing,
                &declining,
                DecisionPointBreadthSwenlinTradingOscillatorParams,
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), advancing.len());
        assert_series_eq(values, &direct.values, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_velocity_acceleration_indicator_matches_direct() {
        let open: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.04 + (i as f64 * 0.09).sin())
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + (i as f64 * 0.11).cos() * 0.9)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.6 + (i as f64 * 0.03).sin().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.6 - (i as f64 * 0.05).cos().abs() * 0.2)
            .collect();
        let candles = crate::utilities::data_loader::Candles::new(
            (0..256_i64).collect(),
            open,
            high,
            low,
            close,
            vec![1_000.0; 256],
        );
        let params = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(21),
            },
            ParamKV {
                key: "smooth_length",
                value: ParamValue::Int(5),
            },
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("hlcc4"),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "velocity_acceleration_indicator",
            output_id: Some("value"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("hlcc4"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = velocity_acceleration_indicator_with_kernel(
            &VelocityAccelerationIndicatorInput::from_candles(
                &candles,
                "hlcc4",
                VelocityAccelerationIndicatorParams {
                    length: Some(21),
                    smooth_length: Some(5),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), candles.close.len());
        assert_series_eq(values, &direct.values, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_normalized_resonator_matches_direct() {
        let open: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.03 + (i as f64 * 0.07).sin())
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + (i as f64 * 0.11).cos() * 0.8)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.6 + (i as f64 * 0.05).sin().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.6 - (i as f64 * 0.03).cos().abs() * 0.2)
            .collect();
        let candles = crate::utilities::data_loader::Candles::new(
            (0..256_i64).collect(),
            open,
            high,
            low,
            close,
            vec![1_000.0; 256],
        );
        let params = [
            ParamKV {
                key: "period",
                value: ParamValue::Int(48),
            },
            ParamKV {
                key: "delta",
                value: ParamValue::Float(0.4),
            },
            ParamKV {
                key: "lookback_mult",
                value: ParamValue::Float(1.2),
            },
            ParamKV {
                key: "signal_length",
                value: ParamValue::Int(7),
            },
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("hl2"),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "normalized_resonator",
            output_id: Some("oscillator"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("hl2"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = normalized_resonator_with_kernel(
            &NormalizedResonatorInput::from_candles(
                &candles,
                "hl2",
                NormalizedResonatorParams {
                    period: Some(48),
                    delta: Some(0.4),
                    lookback_mult: Some(1.2),
                    signal_length: Some(7),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), candles.close.len());
        assert_series_eq(values, &direct.oscillator, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_monotonicity_index_matches_direct() {
        let open: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.04 + (i as f64 * 0.08).sin())
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + (i as f64 * 0.13).cos() * 0.9)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.6 + (i as f64 * 0.05).sin().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.6 - (i as f64 * 0.03).cos().abs() * 0.2)
            .collect();
        let candles = crate::utilities::data_loader::Candles::new(
            (0..256_i64).collect(),
            open,
            high,
            low,
            close,
            vec![1_000.0; 256],
        );
        let params = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(20),
            },
            ParamKV {
                key: "mode",
                value: ParamValue::EnumString("efficiency"),
            },
            ParamKV {
                key: "index_smooth",
                value: ParamValue::Int(5),
            },
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("close"),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "monotonicity_index",
            output_id: Some("index"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("close"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = monotonicity_index_with_kernel(
            &MonotonicityIndexInput::from_candles(
                &candles,
                "close",
                MonotonicityIndexParams {
                    length: Some(20),
                    mode: Some(MonotonicityIndexMode::Efficiency),
                    index_smooth: Some(5),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), candles.close.len());
        assert_series_eq(values, &direct.index, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_half_causal_estimator_matches_direct() {
        let len = 240usize;
        let slots_per_day = 60usize;
        let close: Vec<f64> = (0..len)
            .map(|i| {
                let slot = (i % slots_per_day) as f64;
                let day = (i / slots_per_day) as f64;
                1000.0
                    + day * 4.0
                    + (slot * 0.13).sin() * 25.0
                    + (slot * 0.04).cos() * 9.0
                    + slot * 0.2
            })
            .collect();
        let params = [
            ParamKV {
                key: "slots_per_day",
                value: ParamValue::Int(slots_per_day as i64),
            },
            ParamKV {
                key: "data_period",
                value: ParamValue::Int(5),
            },
            ParamKV {
                key: "filter_length",
                value: ParamValue::Int(20),
            },
            ParamKV {
                key: "kernel_width",
                value: ParamValue::Float(20.0),
            },
            ParamKV {
                key: "kernel_type",
                value: ParamValue::EnumString("epanechnikov"),
            },
            ParamKV {
                key: "confidence_adjust",
                value: ParamValue::EnumString("symmetric"),
            },
            ParamKV {
                key: "maximum_confidence_adjust",
                value: ParamValue::Float(100.0),
            },
            ParamKV {
                key: "enable_expected_value",
                value: ParamValue::Bool(true),
            },
            ParamKV {
                key: "extra_smoothing",
                value: ParamValue::Int(0),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "half_causal_estimator",
            output_id: Some("estimate"),
            data: IndicatorDataRef::Slice { values: &close },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = crate::indicators::half_causal_estimator::half_causal_estimator_with_kernel(
            &crate::indicators::half_causal_estimator::HalfCausalEstimatorInput::from_slice(
                &close,
                crate::indicators::half_causal_estimator::HalfCausalEstimatorParams {
                    slots_per_day: Some(slots_per_day),
                    data_period: Some(5),
                    filter_length: Some(20),
                    kernel_width: Some(20.0),
                    kernel_type: Some(
                        crate::indicators::half_causal_estimator::HalfCausalEstimatorKernelType::Epanechnikov,
                    ),
                    confidence_adjust: Some(
                        crate::indicators::half_causal_estimator::HalfCausalEstimatorConfidenceAdjust::Symmetric,
                    ),
                    maximum_confidence_adjust: Some(100.0),
                    enable_expected_value: Some(true),
                    extra_smoothing: Some(0),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.estimate, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_didi_index_matches_direct() {
        let close: Vec<f64> = (0..256)
            .map(|i| 100.0 + ((i as f64) * 0.09).sin() * 7.0 + (i as f64) * 0.03)
            .collect();
        let params = [
            ParamKV {
                key: "short_length",
                value: ParamValue::Int(3),
            },
            ParamKV {
                key: "medium_length",
                value: ParamValue::Int(8),
            },
            ParamKV {
                key: "long_length",
                value: ParamValue::Int(20),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "didi_index",
            output_id: Some("short"),
            data: IndicatorDataRef::Slice { values: &close },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = didi_index_with_kernel(
            &DidiIndexInput::from_slice(
                &close,
                DidiIndexParams {
                    short_length: Some(3),
                    medium_length: Some(8),
                    long_length: Some(20),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.short, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_ehlers_autocorrelation_periodogram_matches_direct() {
        let close: Vec<f64> = (0..256)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / 20.0;
                phase.sin() + 0.15 * (phase * 0.5).cos()
            })
            .collect();
        let params = [
            ParamKV {
                key: "min_period",
                value: ParamValue::Int(8),
            },
            ParamKV {
                key: "max_period",
                value: ParamValue::Int(48),
            },
            ParamKV {
                key: "avg_length",
                value: ParamValue::Int(3),
            },
            ParamKV {
                key: "enhance",
                value: ParamValue::Bool(true),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "ehlers_autocorrelation_periodogram",
            output_id: Some("dominant_cycle"),
            data: IndicatorDataRef::Slice { values: &close },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = ehlers_autocorrelation_periodogram_with_kernel(
            &EhlersAutocorrelationPeriodogramInput::from_slice(
                &close,
                EhlersAutocorrelationPeriodogramParams {
                    min_period: Some(8),
                    max_period: Some(48),
                    avg_length: Some(3),
                    enhance: Some(true),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.dominant_cycle, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_ehlers_linear_extrapolation_predictor_matches_direct() {
        let close: Vec<f64> = (0..256)
            .map(|i| 100.0 + ((i as f64) * 0.09).sin() * 2.0 + (i as f64 * 0.03))
            .collect();
        let params = [
            ParamKV {
                key: "high_pass_length",
                value: ParamValue::Int(125),
            },
            ParamKV {
                key: "low_pass_length",
                value: ParamValue::Int(12),
            },
            ParamKV {
                key: "gain",
                value: ParamValue::Float(0.7),
            },
            ParamKV {
                key: "bars_forward",
                value: ParamValue::Int(5),
            },
            ParamKV {
                key: "signal_mode",
                value: ParamValue::EnumString("predict_filter_crosses"),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "ehlers_linear_extrapolation_predictor",
            output_id: Some("prediction"),
            data: IndicatorDataRef::Slice { values: &close },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = ehlers_linear_extrapolation_predictor_with_kernel(
            &EhlersLinearExtrapolationPredictorInput::from_slice(
                &close,
                EhlersLinearExtrapolationPredictorParams {
                    high_pass_length: Some(125),
                    low_pass_length: Some(12),
                    gain: Some(0.7),
                    bars_forward: Some(5),
                    signal_mode: Some("predict_filter_crosses".to_string()),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.prediction, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_grover_llorens_cycle_oscillator_matches_direct() {
        let mut open = Vec::with_capacity(256);
        let mut high = Vec::with_capacity(256);
        let mut low = Vec::with_capacity(256);
        let mut close = Vec::with_capacity(256);
        let mut prev = 100.0;
        for i in 0..256 {
            let x = i as f64;
            let wave = (x * 0.11).sin() * 2.4 + (x * 0.037).cos() * 1.3;
            let o = prev + wave * 0.35;
            let c = o + (x * 0.19).sin() * 1.1 - (x * 0.07).cos() * 0.4;
            let h = o.max(c) + 0.6 + (x * 0.03).sin().abs() * 0.25;
            let l = o.min(c) - 0.6 - (x * 0.02).cos().abs() * 0.25;
            open.push(o);
            high.push(h);
            low.push(l);
            close.push(c);
            prev = c;
        }

        let params = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(60),
            },
            ParamKV {
                key: "mult",
                value: ParamValue::Float(8.0),
            },
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("hlc3"),
            },
            ParamKV {
                key: "smooth",
                value: ParamValue::Bool(true),
            },
            ParamKV {
                key: "rsi_period",
                value: ParamValue::Int(14),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "grover_llorens_cycle_oscillator",
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

        let direct = grover_llorens_cycle_oscillator_with_kernel(
            &GroverLlorensCycleOscillatorInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                GroverLlorensCycleOscillatorParams {
                    length: Some(60),
                    mult: Some(8.0),
                    source: Some("hlc3".to_string()),
                    smooth: Some(true),
                    rsi_period: Some(14),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.values, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_historical_volatility_matches_direct() {
        let close: Vec<f64> = (0..256)
            .map(|i| 100.0 + ((i as f64) * 0.02).sin() + (i as f64 * 0.1))
            .collect();
        let params = [
            ParamKV {
                key: "lookback",
                value: ParamValue::Int(20),
            },
            ParamKV {
                key: "annualization_days",
                value: ParamValue::Float(252.0),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "historical_volatility",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &close },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = historical_volatility_with_kernel(
            &HistoricalVolatilityInput::from_slice(
                &close,
                HistoricalVolatilityParams {
                    lookback: Some(20),
                    annualization_days: Some(252.0),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.values, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_stochastic_distance_matches_direct() {
        let close: Vec<f64> = (0..256)
            .map(|i| 100.0 + (i as f64 * 0.07).sin() * 1.3 + i as f64 * 0.03)
            .collect();
        let params = [
            ParamKV {
                key: "lookback_length",
                value: ParamValue::Int(50),
            },
            ParamKV {
                key: "length1",
                value: ParamValue::Int(8),
            },
            ParamKV {
                key: "length2",
                value: ParamValue::Int(4),
            },
            ParamKV {
                key: "ob_level",
                value: ParamValue::Int(40),
            },
            ParamKV {
                key: "os_level",
                value: ParamValue::Int(-40),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "stochastic_distance",
            output_id: Some("oscillator"),
            data: IndicatorDataRef::Slice { values: &close },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = stochastic_distance_with_kernel(
            &StochasticDistanceInput::from_slice(
                &close,
                StochasticDistanceParams {
                    lookback_length: Some(50),
                    length1: Some(8),
                    length2: Some(4),
                    ob_level: Some(40),
                    os_level: Some(-40),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.oscillator, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_adaptive_bandpass_trigger_oscillator_matches_direct() {
        let close: Vec<f64> = (0..256)
            .map(|i| 100.0 + (i as f64 * 0.07).sin() * 1.3 + (i as f64 * 0.03).cos() * 0.6)
            .collect();
        let params = [
            ParamKV {
                key: "delta",
                value: ParamValue::Float(0.1),
            },
            ParamKV {
                key: "alpha",
                value: ParamValue::Float(0.07),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "adaptive_bandpass_trigger_oscillator",
            output_id: Some("in_phase"),
            data: IndicatorDataRef::Slice { values: &close },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = adaptive_bandpass_trigger_oscillator_with_kernel(
            &AdaptiveBandpassTriggerOscillatorInput::from_slice(
                &close,
                AdaptiveBandpassTriggerOscillatorParams {
                    delta: Some(0.1),
                    alpha: Some(0.07),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.in_phase, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_squeeze_index_matches_direct() {
        let close: Vec<f64> = (0..256)
            .map(|i| 100.0 + ((i as f64) * 0.11).sin() * 1.2 + (i as f64 * 0.02))
            .collect();
        let params = [
            ParamKV {
                key: "conv",
                value: ParamValue::Float(50.0),
            },
            ParamKV {
                key: "length",
                value: ParamValue::Int(20),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "squeeze_index",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &close },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = squeeze_index_with_kernel(
            &SqueezeIndexInput::from_slice(
                &close,
                SqueezeIndexParams {
                    conv: Some(50.0),
                    length: Some(20),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.values, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_absolute_strength_index_oscillator_matches_direct() {
        let close: Vec<f64> = (0..256)
            .map(|i| 100.0 + ((i as f64) * 0.17).sin() * 1.8 + ((i % 7) as f64 - 3.0) * 0.04)
            .collect();
        let params = [
            ParamKV {
                key: "ema_length",
                value: ParamValue::Int(21),
            },
            ParamKV {
                key: "signal_length",
                value: ParamValue::Int(34),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "absolute_strength_index_oscillator",
            output_id: Some("oscillator"),
            data: IndicatorDataRef::Slice { values: &close },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = absolute_strength_index_oscillator_with_kernel(
            &AbsoluteStrengthIndexOscillatorInput::from_slice(
                &close,
                AbsoluteStrengthIndexOscillatorParams {
                    ema_length: Some(21),
                    signal_length: Some(34),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.oscillator, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_premier_rsi_oscillator_matches_direct() {
        let close: Vec<f64> = (0..256)
            .map(|i| 100.0 + ((i as f64) * 0.13).sin() * 1.4 + ((i % 11) as f64 - 5.0) * 0.03)
            .collect();
        let params = [
            ParamKV {
                key: "rsi_length",
                value: ParamValue::Int(14),
            },
            ParamKV {
                key: "stoch_length",
                value: ParamValue::Int(8),
            },
            ParamKV {
                key: "smooth_length",
                value: ParamValue::Int(25),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "premier_rsi_oscillator",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &close },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = premier_rsi_oscillator_with_kernel(
            &PremierRsiOscillatorInput::from_slice(
                &close,
                PremierRsiOscillatorParams {
                    rsi_length: Some(14),
                    stoch_length: Some(8),
                    smooth_length: Some(25),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.values, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_multi_length_stochastic_average_matches_direct() {
        let open: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.03 + (i as f64 * 0.09).sin())
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + (i as f64 * 0.13).cos() * 0.8)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.5 + (i as f64 * 0.05).sin().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.5 - (i as f64 * 0.07).cos().abs() * 0.2)
            .collect();
        let candles = crate::utilities::data_loader::Candles::new(
            (0..256_i64).collect(),
            open,
            high,
            low,
            close,
            vec![1_000.0; 256],
        );
        let params = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(14),
            },
            ParamKV {
                key: "presmooth",
                value: ParamValue::Int(10),
            },
            ParamKV {
                key: "premethod",
                value: ParamValue::EnumString("sma"),
            },
            ParamKV {
                key: "postsmooth",
                value: ParamValue::Int(10),
            },
            ParamKV {
                key: "postmethod",
                value: ParamValue::EnumString("lsma"),
            },
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("hlc3"),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "multi_length_stochastic_average",
            output_id: Some("value"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("hlc3"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = multi_length_stochastic_average_with_kernel(
            &MultiLengthStochasticAverageInput::from_candles(
                &candles,
                "hlc3",
                MultiLengthStochasticAverageParams {
                    length: Some(14),
                    presmooth: Some(10),
                    premethod: Some("sma".to_string()),
                    postsmooth: Some(10),
                    postmethod: Some("lsma".to_string()),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), candles.close.len());
        assert_series_eq(values, &direct.values, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_hull_butterfly_oscillator_matches_direct() {
        let open: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.03 + (i as f64 * 0.09).sin())
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + (i as f64 * 0.13).cos() * 0.8)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.5 + (i as f64 * 0.05).sin().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.5 - (i as f64 * 0.07).cos().abs() * 0.2)
            .collect();
        let candles = crate::utilities::data_loader::Candles::new(
            (0..256_i64).collect(),
            open,
            high,
            low,
            close,
            vec![1_000.0; 256],
        );
        let params = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(14),
            },
            ParamKV {
                key: "mult",
                value: ParamValue::Float(1.75),
            },
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("hlc3"),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "hull_butterfly_oscillator",
            output_id: Some("oscillator"),
            data: IndicatorDataRef::Candles {
                candles: &candles,
                source: Some("hlc3"),
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = hull_butterfly_oscillator_with_kernel(
            &HullButterflyOscillatorInput::from_candles(
                &candles,
                "hlc3",
                HullButterflyOscillatorParams {
                    length: Some(14),
                    mult: Some(1.75),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), candles.close.len());
        assert_series_eq(values, &direct.oscillator, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_fibonacci_trailing_stop_matches_direct() {
        let open: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.03 + (i as f64 * 0.09).sin())
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + (i as f64 * 0.13).cos() * 0.8)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.5 + (i as f64 * 0.05).sin().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.5 - (i as f64 * 0.07).cos().abs() * 0.2)
            .collect();

        let params = [
            ParamKV {
                key: "left_bars",
                value: ParamValue::Int(12),
            },
            ParamKV {
                key: "right_bars",
                value: ParamValue::Int(2),
            },
            ParamKV {
                key: "level",
                value: ParamValue::Float(-0.236),
            },
            ParamKV {
                key: "trigger",
                value: ParamValue::EnumString("wick"),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "fibonacci_trailing_stop",
            output_id: Some("trailing_stop"),
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

        let direct = fibonacci_trailing_stop_with_kernel(
            &FibonacciTrailingStopInput::from_slices(
                &high,
                &low,
                &close,
                FibonacciTrailingStopParams {
                    left_bars: Some(12),
                    right_bars: Some(2),
                    level: Some(-0.236),
                    trigger: Some("wick".to_string()),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.trailing_stop, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_volume_energy_reservoirs_matches_direct() {
        let open: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.03 + (i as f64 * 0.08).sin())
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + (i as f64 * 0.11).cos() * 0.9)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.6 + (i as f64 * 0.03).sin().abs() * 0.25)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.6 - (i as f64 * 0.05).cos().abs() * 0.2)
            .collect();
        let volume: Vec<f64> = (0..256)
            .map(|i| 1_000.0 + i as f64 * 4.0 + (i as f64 * 0.09).sin() * 180.0)
            .collect();

        let params = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(18),
            },
            ParamKV {
                key: "sensitivity",
                value: ParamValue::Float(1.7),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "volume_energy_reservoirs",
            output_id: Some("momentum"),
            data: IndicatorDataRef::Ohlcv {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = volume_energy_reservoirs_with_kernel(
            &VolumeEnergyReservoirsInput::from_slices(
                &high,
                &low,
                &close,
                &volume,
                VolumeEnergyReservoirsParams {
                    length: Some(18),
                    sensitivity: Some(1.7),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.momentum, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_neighboring_trailing_stop_matches_direct() {
        let open: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.04 + (i as f64 * 0.07).sin())
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, &o)| o + (i as f64 * 0.11).cos() * 0.85)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.55 + (i as f64 * 0.03).sin().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.55 - (i as f64 * 0.05).cos().abs() * 0.2)
            .collect();

        let params = [
            ParamKV {
                key: "buffer_size",
                value: ParamValue::Int(180),
            },
            ParamKV {
                key: "k",
                value: ParamValue::Int(30),
            },
            ParamKV {
                key: "percentile",
                value: ParamValue::Float(87.5),
            },
            ParamKV {
                key: "smooth",
                value: ParamValue::Int(4),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "neighboring_trailing_stop",
            output_id: Some("trailing_stop"),
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

        let direct = neighboring_trailing_stop_with_kernel(
            &NeighboringTrailingStopInput::from_slices(
                &high,
                &low,
                &close,
                NeighboringTrailingStopParams {
                    buffer_size: Some(180),
                    k: Some(30),
                    percentile: Some(87.5),
                    smooth: Some(4),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.trailing_stop, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_macd_wave_signal_pro_matches_direct() {
        let open: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.08 + ((i as f64) * 0.05).sin() * 0.7)
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, o)| o + ((i as f64) * 0.09).cos() * 0.9)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.55 + (i as f64 * 0.03).sin().abs() * 0.2)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.55 - (i as f64 * 0.05).cos().abs() * 0.2)
            .collect();
        let combos = [IndicatorParamSet { params: &[] }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "macd_wave_signal_pro",
            output_id: Some("line_convergence"),
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

        let direct = macd_wave_signal_pro_with_kernel(
            &MacdWaveSignalProInput::from_slices(&open, &high, &low, &close, Default::default()),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.line_convergence, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_hema_trend_levels_matches_direct() {
        let open: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.05 + ((i as f64) * 0.09).sin() * 1.3)
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, o)| o + ((i as f64) * 0.07).cos() * 1.1)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.65 + (i as f64 * 0.03).sin().abs() * 0.25)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.65 - (i as f64 * 0.05).cos().abs() * 0.25)
            .collect();
        let params = [
            ParamKV {
                key: "fast_length",
                value: ParamValue::Int(20),
            },
            ParamKV {
                key: "slow_length",
                value: ParamValue::Int(40),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "hema_trend_levels",
            output_id: Some("bullish_test_level"),
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

        let direct = hema_trend_levels_with_kernel(
            &HemaTrendLevelsInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                HemaTrendLevelsParams {
                    fast_length: Some(20),
                    slow_length: Some(40),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.bullish_test_level, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_fibonacci_entry_bands_matches_direct() {
        let open: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.05 + ((i as f64) * 0.09).sin() * 1.3)
            .collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, o)| o + ((i as f64) * 0.07).cos() * 1.1)
            .collect();
        let high: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.max(c) + 0.65 + (i as f64 * 0.03).sin().abs() * 0.25)
            .collect();
        let low: Vec<f64> = open
            .iter()
            .zip(close.iter())
            .enumerate()
            .map(|(i, (&o, &c))| o.min(c) - 0.65 - (i as f64 * 0.05).cos().abs() * 0.25)
            .collect();
        let params = [
            ParamKV {
                key: "source",
                value: ParamValue::EnumString("hlc3"),
            },
            ParamKV {
                key: "length",
                value: ParamValue::Int(20),
            },
            ParamKV {
                key: "atr_length",
                value: ParamValue::Int(11),
            },
            ParamKV {
                key: "use_atr",
                value: ParamValue::Bool(true),
            },
            ParamKV {
                key: "tp_aggressiveness",
                value: ParamValue::EnumString("medium"),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "fibonacci_entry_bands",
            output_id: Some("tp_long_band"),
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

        let direct = fibonacci_entry_bands_with_kernel(
            &FibonacciEntryBandsInput::from_slices(
                &open,
                &high,
                &low,
                &close,
                FibonacciEntryBandsParams {
                    source: Some("hlc3".to_string()),
                    length: Some(20),
                    atr_length: Some(11),
                    use_atr: Some(true),
                    tp_aggressiveness: Some("medium".to_string()),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.tp_long_band, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_vertical_horizontal_filter_matches_direct() {
        let close: Vec<f64> = (0..256)
            .map(|i| 100.0 + ((i as f64) * 0.02).sin() + (i as f64 * 0.1))
            .collect();
        let params = [ParamKV {
            key: "length",
            value: ParamValue::Int(28),
        }];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "vertical_horizontal_filter",
            output_id: Some("value"),
            data: IndicatorDataRef::Slice { values: &close },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = vertical_horizontal_filter_with_kernel(
            &VerticalHorizontalFilterInput::from_slice(
                &close,
                VerticalHorizontalFilterParams { length: Some(28) },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.values, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_intraday_momentum_index_matches_direct() {
        let open: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.1 + ((i as f64) * 0.05).cos() * 0.2)
            .collect();
        let high: Vec<f64> = open.iter().map(|v| v + 0.9).collect();
        let low: Vec<f64> = open.iter().map(|v| v - 0.8).collect();
        let close: Vec<f64> = open
            .iter()
            .enumerate()
            .map(|(i, o)| o + ((i as f64) * 0.09).sin() * 0.6)
            .collect();
        let params = [
            ParamKV {
                key: "length",
                value: ParamValue::Int(14),
            },
            ParamKV {
                key: "length_ma",
                value: ParamValue::Int(6),
            },
            ParamKV {
                key: "mult",
                value: ParamValue::Float(2.0),
            },
            ParamKV {
                key: "length_bb",
                value: ParamValue::Int(20),
            },
            ParamKV {
                key: "apply_smoothing",
                value: ParamValue::Bool(true),
            },
            ParamKV {
                key: "low_band",
                value: ParamValue::Int(10),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "intraday_momentum_index",
            output_id: Some("imi"),
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

        let direct = intraday_momentum_index_with_kernel(
            &IntradayMomentumIndexInput::from_slices(
                &open,
                &close,
                IntradayMomentumIndexParams {
                    length: Some(14),
                    length_ma: Some(6),
                    mult: Some(2.0),
                    length_bb: Some(20),
                    apply_smoothing: Some(true),
                    low_band: Some(10),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.imi, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_atr_percentile_matches_direct() {
        let high: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.1 + ((i as f64) * 0.03).sin().abs())
            .collect();
        let low: Vec<f64> = high
            .iter()
            .enumerate()
            .map(|(i, h)| h - 0.75 - ((i as f64) * 0.02).cos().abs() * 0.2)
            .collect();
        let close: Vec<f64> = low
            .iter()
            .zip(high.iter())
            .enumerate()
            .map(|(i, (l, h))| l + (h - l) * (0.35 + 0.2 * ((i as f64) * 0.05).sin().abs()))
            .collect();
        let params = [
            ParamKV {
                key: "atr_length",
                value: ParamValue::Int(10),
            },
            ParamKV {
                key: "percentile_length",
                value: ParamValue::Int(20),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "atr_percentile",
            output_id: Some("value"),
            data: IndicatorDataRef::Ohlc {
                open: &close,
                high: &high,
                low: &low,
                close: &close,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = atr_percentile_with_kernel(
            &AtrPercentileInput::from_slices(
                &high,
                &low,
                &close,
                AtrPercentileParams {
                    atr_length: Some(10),
                    percentile_length: Some(20),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.values, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_demand_index_matches_direct() {
        let high: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.15 + ((i as f64) * 0.03).sin().abs())
            .collect();
        let low: Vec<f64> = high
            .iter()
            .enumerate()
            .map(|(i, h)| h - 0.9 - ((i as f64) * 0.04).cos().abs() * 0.3)
            .collect();
        let close: Vec<f64> = low
            .iter()
            .zip(high.iter())
            .enumerate()
            .map(|(i, (l, h))| l + (h - l) * (0.25 + 0.5 * ((i as f64) * 0.07).sin().abs()))
            .collect();
        let open: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, c)| c - 0.2 + ((i as f64) * 0.05).cos() * 0.1)
            .collect();
        let volume: Vec<f64> = (0..256)
            .map(|i| 1000.0 + (i as f64) * 3.0 + ((i as f64) * 0.11).sin().abs() * 40.0)
            .collect();
        let params = [
            ParamKV {
                key: "len_bs",
                value: ParamValue::Int(19),
            },
            ParamKV {
                key: "len_bs_ma",
                value: ParamValue::Int(19),
            },
            ParamKV {
                key: "len_di_ma",
                value: ParamValue::Int(19),
            },
            ParamKV {
                key: "ma_type",
                value: ParamValue::EnumString("ema"),
            },
        ];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "demand_index",
            output_id: Some("demand_index"),
            data: IndicatorDataRef::Ohlcv {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                volume: &volume,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = demand_index_with_kernel(
            &DemandIndexInput::from_slices(
                &high,
                &low,
                &close,
                &volume,
                DemandIndexParams {
                    len_bs: Some(19),
                    len_bs_ma: Some(19),
                    len_di_ma: Some(19),
                    ma_type: Some("ema".to_string()),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), close.len());
        assert_series_eq(values, &direct.demand_index, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_vwap_zscore_with_signals_matches_direct() {
        let close: Vec<f64> = (0..192).map(|i| 100.0 + (i as f64 * 0.15)).collect();
        let volume: Vec<f64> = (0..192).map(|i| 1_000.0 + (i as f64 * 2.0)).collect();
        let req = IndicatorBatchRequest {
            indicator_id: "vwap_zscore_with_signals",
            output_id: Some("zvwap"),
            data: IndicatorDataRef::CloseVolume {
                close: &close,
                volume: &volume,
            },
            combos: &[IndicatorParamSet {
                params: &[
                    ParamKV {
                        key: "length",
                        value: ParamValue::Int(20),
                    },
                    ParamKV {
                        key: "upper_bottom",
                        value: ParamValue::Float(2.5),
                    },
                    ParamKV {
                        key: "lower_bottom",
                        value: ParamValue::Float(-2.5),
                    },
                ],
            }],
            kernel: Kernel::Auto,
        };

        let out = compute_cpu_batch(req).unwrap();
        let values = out.values_f64.as_ref().unwrap();
        let direct = vwap_zscore_with_signals_with_kernel(
            &VwapZscoreWithSignalsInput::from_slices(
                &close,
                &volume,
                VwapZscoreWithSignalsParams {
                    length: Some(20),
                    upper_bottom: Some(2.5),
                    lower_bottom: Some(-2.5),
                },
            ),
            Kernel::Auto,
        )
        .unwrap();
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, close.len());
        assert_series_eq(values, &direct.zvwap, 1e-9);
    }

    #[test]
    fn compute_cpu_batch_gopalakrishnan_range_index_matches_direct() {
        let high: Vec<f64> = (0..256)
            .map(|i| 100.0 + i as f64 * 0.1 + ((i as f64) * 0.03).sin().abs())
            .collect();
        let low: Vec<f64> = high
            .iter()
            .enumerate()
            .map(|(i, h)| h - 0.75 - ((i as f64) * 0.02).cos().abs() * 0.2)
            .collect();
        let params = [ParamKV {
            key: "length",
            value: ParamValue::Int(5),
        }];
        let combos = [IndicatorParamSet { params: &params }];

        let dispatched = compute_cpu_batch(IndicatorBatchRequest {
            indicator_id: "gopalakrishnan_range_index",
            output_id: Some("value"),
            data: IndicatorDataRef::HighLow {
                high: &high,
                low: &low,
            },
            combos: &combos,
            kernel: Kernel::Auto,
        })
        .unwrap();

        let direct = gopalakrishnan_range_index_with_kernel(
            &GopalakrishnanRangeIndexInput::from_slices(
                &high,
                &low,
                GopalakrishnanRangeIndexParams { length: Some(5) },
            ),
            Kernel::Auto,
        )
        .unwrap();

        let values = dispatched.values_f64.as_ref().unwrap();
        assert_eq!(values.len(), high.len());
        assert_series_eq(values, &direct.values, 1e-9);
    }
}
