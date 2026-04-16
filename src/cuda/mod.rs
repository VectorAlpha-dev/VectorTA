#[cfg(feature = "cuda")]
use std::sync::OnceLock;
#[cfg(all(feature = "cuda", test))]
use std::{
    cell::Cell,
    sync::{Mutex, MutexGuard},
};

#[cfg(feature = "cuda")]
pub mod absolute_strength_index_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod accumulation_swing_index_wrapper;
#[cfg(feature = "cuda")]
pub mod ad_wrapper;
#[cfg(feature = "cuda")]
pub mod adaptive_bandpass_trigger_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod adaptive_bounds_rsi_wrapper;
#[cfg(feature = "cuda")]
pub mod adaptive_macd_wrapper;
#[cfg(feature = "cuda")]
pub mod adaptive_momentum_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod adaptive_schaff_trend_cycle_wrapper;
#[cfg(feature = "cuda")]
pub mod adjustable_ma_alternating_extremities_wrapper;
#[cfg(feature = "cuda")]
pub mod advance_decline_line_wrapper;
#[cfg(feature = "cuda")]
pub mod adx_wrapper;
#[cfg(feature = "cuda")]
pub mod adxr_wrapper;
#[cfg(feature = "cuda")]
pub mod alligator_wrapper;
#[cfg(feature = "cuda")]
pub mod alphatrend_wrapper;
#[cfg(feature = "cuda")]
pub mod andean_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod aroon_wrapper;
#[cfg(feature = "cuda")]
pub mod atr_percentile_wrapper;
#[cfg(feature = "cuda")]
pub mod atr_wrapper;
#[cfg(feature = "cuda")]
pub mod autocorrelation_indicator_wrapper;
#[cfg(feature = "cuda")]
pub mod avsl_wrapper;
#[cfg(feature = "cuda")]
pub mod bandpass_wrapper;
#[cfg(feature = "cuda")]
pub mod bench;
#[cfg(feature = "cuda")]
pub mod candle_strength_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod chande_wrapper;
#[cfg(feature = "cuda")]
pub mod cvi_wrapper;
#[cfg(feature = "cuda")]
pub mod cyberpunk_value_trend_analyzer_wrapper;
#[cfg(feature = "cuda")]
pub mod cycle_channel_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod daily_factor_wrapper;
#[cfg(feature = "cuda")]
pub mod device_types;
#[cfg(feature = "cuda")]
pub mod di_wrapper;
#[cfg(feature = "cuda")]
pub mod disparity_index_wrapper;
#[cfg(feature = "cuda")]
pub mod dm_wrapper;
#[cfg(feature = "cuda")]
pub mod donchian_channel_width_wrapper;
#[cfg(feature = "cuda")]
pub mod donchian_wrapper;
#[cfg(feature = "cuda")]
pub mod dx_wrapper;
#[cfg(feature = "cuda")]
pub mod dynamic_momentum_index_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_adaptive_cg_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_adaptive_cyber_cycle_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_autocorrelation_periodogram_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_data_sampling_relative_strength_indicator_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_detrending_filter_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_linear_extrapolation_predictor_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_smoothed_adaptive_momentum_wrapper;
#[cfg(feature = "cuda")]
pub mod eri_wrapper;
#[cfg(feature = "cuda")]
pub mod evasive_supertrend_wrapper;
#[cfg(feature = "cuda")]
pub mod ewma_volatility_wrapper;
#[cfg(feature = "cuda")]
pub mod goertzel_cycle_composite_wave_wrapper;
#[cfg(feature = "cuda")]
pub mod hypertrend_wrapper;
#[cfg(feature = "cuda")]
pub mod ichimoku_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod ict_propulsion_block_wrapper;
#[cfg(feature = "cuda")]
pub mod impulse_macd_wrapper;
#[cfg(feature = "cuda")]
pub mod insync_index_wrapper;
#[cfg(feature = "cuda")]
pub mod kairi_relative_index_wrapper;
#[cfg(feature = "cuda")]
pub mod kase_peak_oscillator_with_divergences_wrapper;
#[cfg(feature = "cuda")]
pub mod keltner_channel_width_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod keltner_wrapper;
#[cfg(feature = "cuda")]
pub mod market_meanness_index_wrapper;
#[cfg(feature = "cuda")]
pub mod market_structure_confluence_wrapper;
#[cfg(feature = "cuda")]
pub mod market_structure_trailing_stop_wrapper;
#[cfg(feature = "cuda")]
pub mod marketefi_wrapper;
#[cfg(feature = "cuda")]
pub mod medprice_wrapper;
#[cfg(feature = "cuda")]
pub mod mesa_stochastic_multi_length_wrapper;
#[cfg(feature = "cuda")]
pub mod module_loader;
#[cfg(feature = "cuda")]
pub mod moving_average_cross_probability_wrapper;
#[cfg(feature = "cuda")]
pub mod moving_averages;
#[cfg(feature = "cuda")]
pub mod multi_length_stochastic_average_wrapper;
#[cfg(feature = "cuda")]
pub mod normalized_resonator_wrapper;
#[cfg(feature = "cuda")]
pub mod possible_rsi_wrapper;
#[cfg(feature = "cuda")]
pub mod price_moving_average_ratio_percentile_wrapper;
#[cfg(feature = "cuda")]
pub mod qstick_wrapper;
#[cfg(feature = "cuda")]
pub mod range_breakout_signals_wrapper;
#[cfg(feature = "cuda")]
pub mod range_filtered_trend_signals_wrapper;
#[cfg(feature = "cuda")]
pub mod range_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod rocr_wrapper;
#[cfg(feature = "cuda")]
pub mod runtime;
#[cfg(feature = "cuda")]
pub mod smooth_theil_sen_wrapper;
#[cfg(feature = "cuda")]
pub mod standardized_psar_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod statistical_trailing_stop_wrapper;
#[cfg(feature = "cuda")]
pub mod supertrend_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod supertrend_recovery_wrapper;
#[cfg(feature = "cuda")]
pub mod vdubus_divergence_wave_pattern_generator_wrapper;
#[cfg(feature = "cuda")]
pub mod velocity_acceleration_convergence_divergence_indicator_wrapper;
#[cfg(feature = "cuda")]
pub mod volume_weighted_stochastic_rsi_wrapper;
#[cfg(feature = "cuda")]
pub mod vwap_deviation_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod vwap_zscore_with_signals_wrapper;
#[cfg(feature = "cuda")]
pub mod wavetrend;

#[cfg(feature = "cuda")]
pub use absolute_strength_index_oscillator_wrapper::{
    CudaAbsoluteStrengthIndexOscillator, CudaAbsoluteStrengthIndexOscillatorError,
};
#[cfg(feature = "cuda")]
pub use accumulation_swing_index_wrapper::{
    CudaAccumulationSwingIndex, CudaAccumulationSwingIndexError,
};
#[cfg(feature = "cuda")]
pub use ad_wrapper::{CudaAd, CudaAdError};
#[cfg(feature = "cuda")]
pub use adaptive_bandpass_trigger_oscillator_wrapper::{
    CudaAdaptiveBandpassTriggerOscillator, CudaAdaptiveBandpassTriggerOscillatorError,
};
#[cfg(feature = "cuda")]
pub use adaptive_bounds_rsi_wrapper::{CudaAdaptiveBoundsRsi, CudaAdaptiveBoundsRsiError};
#[cfg(feature = "cuda")]
pub use adaptive_macd_wrapper::{CudaAdaptiveMacd, CudaAdaptiveMacdError};
#[cfg(feature = "cuda")]
pub use adaptive_momentum_oscillator_wrapper::{
    CudaAdaptiveMomentumOscillator, CudaAdaptiveMomentumOscillatorError,
};
#[cfg(feature = "cuda")]
pub use adaptive_schaff_trend_cycle_wrapper::{
    CudaAdaptiveSchaffTrendCycle, CudaAdaptiveSchaffTrendCycleError,
};
#[cfg(feature = "cuda")]
pub use adjustable_ma_alternating_extremities_wrapper::{
    CudaAdjustableMaAlternatingExtremities, CudaAdjustableMaAlternatingExtremitiesError,
};
#[cfg(feature = "cuda")]
pub use advance_decline_line_wrapper::{CudaAdvanceDeclineLine, CudaAdvanceDeclineLineError};
#[cfg(feature = "cuda")]
pub use adx_wrapper::{CudaAdx, CudaAdxError};
#[cfg(feature = "cuda")]
pub use adxr_wrapper::{CudaAdxr, CudaAdxrError};
#[cfg(feature = "cuda")]
pub use alligator_wrapper::{
    CudaAlligator, CudaAlligatorBatchResult, CudaAlligatorError, DeviceArrayF32Trio,
};
#[cfg(feature = "cuda")]
pub use alphatrend_wrapper::{CudaAlphaTrend, CudaAlphaTrendError};
#[cfg(feature = "cuda")]
pub use andean_oscillator_wrapper::{CudaAndeanOscillator, CudaAndeanOscillatorError};
#[cfg(feature = "cuda")]
pub use aroon_wrapper::{CudaAroon, CudaAroonError};
#[cfg(feature = "cuda")]
pub use atr_percentile_wrapper::{CudaAtrPercentile, CudaAtrPercentileError};
#[cfg(feature = "cuda")]
pub use atr_wrapper::CudaAtr;
#[cfg(feature = "cuda")]
pub use autocorrelation_indicator_wrapper::{
    CudaAutocorrelationIndicator, CudaAutocorrelationIndicatorError,
};
#[cfg(feature = "cuda")]
pub use avsl_wrapper::{CudaAvsl, CudaAvslError};
#[cfg(feature = "cuda")]
pub use bandpass_wrapper::{CudaBandpass, CudaBandpassBatchResult, DeviceArrayF32Quad};
#[cfg(feature = "cuda")]
pub use bench::{CudaBenchScenario, CudaBenchState};
#[cfg(feature = "cuda")]
pub use candle_strength_oscillator_wrapper::{
    CudaCandleStrengthOscillator, CudaCandleStrengthOscillatorError,
};
#[cfg(feature = "cuda")]
pub use chande_wrapper::CudaChande;
#[cfg(feature = "cuda")]
pub use cvi_wrapper::{CudaCvi, CudaCviError};
#[cfg(feature = "cuda")]
pub use cyberpunk_value_trend_analyzer_wrapper::{
    CudaCyberpunkValueTrendAnalyzer, CudaCyberpunkValueTrendAnalyzerError,
};
#[cfg(feature = "cuda")]
pub use cycle_channel_oscillator_wrapper::{
    CudaCycleChannelOscillator, CudaCycleChannelOscillatorError,
};
#[cfg(feature = "cuda")]
pub use daily_factor_wrapper::{CudaDailyFactor, CudaDailyFactorError};
#[cfg(feature = "cuda")]
pub use device_types::{
    CudaDeviceCloseVolumeRef, CudaDeviceHighLowRef, CudaDeviceMatrix, CudaDeviceMatrixF32,
    CudaDeviceMatrixF32Ref, CudaDeviceOhlc, CudaDeviceOhlcRef, CudaDeviceOhlcv, CudaDeviceOhlcvRef,
    CudaDeviceSliceF32Ref, CudaDeviceSliceI32Ref, CudaDeviceSliceI64Ref, CudaDeviceVector,
    CudaDeviceVectorF32, CudaDeviceVectorI32, CudaDeviceVectorI64, CudaDeviceViewError,
};
#[cfg(feature = "cuda")]
pub use di_wrapper::{CudaDi, CudaDiError, DeviceArrayF32Pair};
#[cfg(feature = "cuda")]
pub use disparity_index_wrapper::{CudaDisparityIndex, CudaDisparityIndexError};
#[cfg(feature = "cuda")]
pub use dm_wrapper::{CudaDm, CudaDmError};
#[cfg(feature = "cuda")]
pub use donchian_channel_width_wrapper::{CudaDonchianChannelWidth, CudaDonchianChannelWidthError};
#[cfg(feature = "cuda")]
pub use donchian_wrapper::{CudaDonchian, CudaDonchianError};
#[cfg(feature = "cuda")]
pub use dx_wrapper::{CudaDx, CudaDxError};
#[cfg(feature = "cuda")]
pub use dynamic_momentum_index_wrapper::{CudaDynamicMomentumIndex, CudaDynamicMomentumIndexError};
#[cfg(feature = "cuda")]
pub use ehlers_adaptive_cg_wrapper::{CudaEhlersAdaptiveCg, CudaEhlersAdaptiveCgError};
#[cfg(feature = "cuda")]
pub use ehlers_adaptive_cyber_cycle_wrapper::{
    CudaEhlersAdaptiveCyberCycle, CudaEhlersAdaptiveCyberCycleError,
};
#[cfg(feature = "cuda")]
pub use ehlers_autocorrelation_periodogram_wrapper::{
    CudaEhlersAutocorrelationPeriodogram, CudaEhlersAutocorrelationPeriodogramError,
};
#[cfg(feature = "cuda")]
pub use ehlers_data_sampling_relative_strength_indicator_wrapper::{
    CudaEhlersDataSamplingRelativeStrengthIndicator,
    CudaEhlersDataSamplingRelativeStrengthIndicatorError,
};
#[cfg(feature = "cuda")]
pub use ehlers_detrending_filter_wrapper::{
    CudaEhlersDetrendingFilter, CudaEhlersDetrendingFilterError,
};
#[cfg(feature = "cuda")]
pub use ehlers_linear_extrapolation_predictor_wrapper::{
    CudaEhlersLinearExtrapolationPredictor, CudaEhlersLinearExtrapolationPredictorError,
};
#[cfg(feature = "cuda")]
pub use ehlers_smoothed_adaptive_momentum_wrapper::{
    CudaEhlersSmoothedAdaptiveMomentum, CudaEhlersSmoothedAdaptiveMomentumError,
};
#[cfg(feature = "cuda")]
pub use eri_wrapper::{CudaEri, CudaEriError};
#[cfg(feature = "cuda")]
pub use evasive_supertrend_wrapper::{CudaEvasiveSuperTrend, CudaEvasiveSuperTrendError};
#[cfg(feature = "cuda")]
pub use ewma_volatility_wrapper::{CudaEwmaVolatility, CudaEwmaVolatilityError};
#[cfg(feature = "cuda")]
pub use goertzel_cycle_composite_wave_wrapper::{
    CudaGoertzelCycleCompositeWave, CudaGoertzelCycleCompositeWaveError,
};
#[cfg(feature = "cuda")]
pub use hypertrend_wrapper::{CudaHyperTrend, CudaHyperTrendError};
#[cfg(feature = "cuda")]
pub use ichimoku_oscillator_wrapper::{CudaIchimokuOscillator, CudaIchimokuOscillatorError};
#[cfg(feature = "cuda")]
pub use ict_propulsion_block_wrapper::{CudaIctPropulsionBlock, CudaIctPropulsionBlockError};
#[cfg(feature = "cuda")]
pub use impulse_macd_wrapper::{CudaImpulseMacd, CudaImpulseMacdError};
#[cfg(feature = "cuda")]
pub use insync_index_wrapper::{CudaInsyncIndex, CudaInsyncIndexError};
#[cfg(feature = "cuda")]
pub use kairi_relative_index_wrapper::{CudaKairiRelativeIndex, CudaKairiRelativeIndexError};
#[cfg(feature = "cuda")]
pub use kase_peak_oscillator_with_divergences_wrapper::{
    CudaKasePeakOscillatorWithDivergences, CudaKasePeakOscillatorWithDivergencesError,
};
#[cfg(feature = "cuda")]
pub use keltner_channel_width_oscillator_wrapper::{
    CudaKeltnerChannelWidthOscillator, CudaKeltnerChannelWidthOscillatorError,
};
#[cfg(feature = "cuda")]
pub use keltner_wrapper::{
    CudaKeltner, CudaKeltnerBatchResult, CudaKeltnerError, DeviceKeltnerTriplet,
};
#[cfg(feature = "cuda")]
pub use market_meanness_index_wrapper::{CudaMarketMeannessIndex, CudaMarketMeannessIndexError};
#[cfg(feature = "cuda")]
pub use market_structure_confluence_wrapper::{
    CudaMarketStructureConfluence, CudaMarketStructureConfluenceError,
};
#[cfg(feature = "cuda")]
pub use market_structure_trailing_stop_wrapper::{
    CudaMarketStructureTrailingStop, CudaMarketStructureTrailingStopError,
};
#[cfg(feature = "cuda")]
pub use marketefi_wrapper::{CudaMarketefi, CudaMarketefiError};
#[cfg(feature = "cuda")]
pub use medprice_wrapper::CudaMedprice;
#[cfg(feature = "cuda")]
pub use mesa_stochastic_multi_length_wrapper::{
    CudaMesaStochasticMultiLength, CudaMesaStochasticMultiLengthError,
};
#[cfg(feature = "cuda")]
pub use moving_average_cross_probability_wrapper::{
    CudaMovingAverageCrossProbability, CudaMovingAverageCrossProbabilityError,
};
#[cfg(feature = "cuda")]
pub use moving_averages::rsmk_wrapper::{CudaRsmk, CudaRsmkError};
#[cfg(feature = "cuda")]
pub use moving_averages::wclprice_wrapper::CudaWclprice;
#[cfg(feature = "cuda")]
pub use moving_averages::{
    CudaAlma, CudaDma, CudaEhlersPma, CudaGaussian, CudaJma, CudaMaDeviceDataRef, CudaMama,
    CudaReflex, CudaSqwma, CudaTema, CudaVwma, DeviceArrayF32, DeviceEhlersPmaPair, DeviceMamaPair,
};
#[cfg(feature = "cuda")]
pub use multi_length_stochastic_average_wrapper::{
    CudaMultiLengthStochasticAverage, CudaMultiLengthStochasticAverageError,
};
#[cfg(feature = "cuda")]
pub use normalized_resonator_wrapper::{CudaNormalizedResonator, CudaNormalizedResonatorError};
#[cfg(feature = "cuda")]
pub use possible_rsi_wrapper::{CudaPossibleRsi, CudaPossibleRsiError};
#[cfg(feature = "cuda")]
pub use price_moving_average_ratio_percentile_wrapper::{
    CudaPriceMovingAverageRatioPercentile, CudaPriceMovingAverageRatioPercentileError,
};
#[cfg(feature = "cuda")]
pub use qstick_wrapper::{
    BatchKernelPolicy as QsBatchKernelPolicy, CudaQstick, CudaQstickError, CudaQstickPolicy,
    ManySeriesKernelPolicy as QsManySeriesKernelPolicy,
};
#[cfg(feature = "cuda")]
pub use range_breakout_signals_wrapper::{CudaRangeBreakoutSignals, CudaRangeBreakoutSignalsError};
#[cfg(feature = "cuda")]
pub use range_filtered_trend_signals_wrapper::{
    CudaRangeFilteredTrendSignals, CudaRangeFilteredTrendSignalsError,
};
#[cfg(feature = "cuda")]
pub use range_oscillator_wrapper::{CudaRangeOscillator, CudaRangeOscillatorError};
#[cfg(feature = "cuda")]
pub use rocr_wrapper::{CudaRocr, CudaRocrError};
#[cfg(feature = "cuda")]
pub use runtime::{CudaRuntime, CudaRuntimeError};
#[cfg(feature = "cuda")]
pub use smooth_theil_sen_wrapper::{CudaSmoothTheilSen, CudaSmoothTheilSenError};
#[cfg(feature = "cuda")]
pub use standardized_psar_oscillator_wrapper::{
    CudaStandardizedPsarOscillator, CudaStandardizedPsarOscillatorError,
};
#[cfg(feature = "cuda")]
pub use statistical_trailing_stop_wrapper::{
    CudaStatisticalTrailingStop, CudaStatisticalTrailingStopError,
};
#[cfg(feature = "cuda")]
pub use supertrend_oscillator_wrapper::{CudaSupertrendOscillator, CudaSupertrendOscillatorError};
#[cfg(feature = "cuda")]
pub use supertrend_recovery_wrapper::{CudaSuperTrendRecovery, CudaSuperTrendRecoveryError};
#[cfg(feature = "cuda")]
pub use vdubus_divergence_wave_pattern_generator_wrapper::{
    CudaVdubusDivergenceWavePatternGenerator, CudaVdubusDivergenceWavePatternGeneratorError,
};
#[cfg(feature = "cuda")]
pub use velocity_acceleration_convergence_divergence_indicator_wrapper::{
    CudaVelocityAccelerationConvergenceDivergenceIndicator,
    CudaVelocityAccelerationConvergenceDivergenceIndicatorError,
};
#[cfg(feature = "cuda")]
pub use volume_weighted_stochastic_rsi_wrapper::{
    CudaVolumeWeightedStochasticRsi, CudaVolumeWeightedStochasticRsiError,
};
#[cfg(feature = "cuda")]
pub use vwap_deviation_oscillator_wrapper::{
    CudaVwapDeviationOscillator, CudaVwapDeviationOscillatorError,
};
#[cfg(feature = "cuda")]
pub use vwap_zscore_with_signals_wrapper::{
    CudaVwapZscoreWithSignals, CudaVwapZscoreWithSignalsError,
};
#[cfg(feature = "cuda")]
pub mod oscillators;
#[cfg(feature = "cuda")]
pub use oscillators::msw_wrapper::{CudaMsw, CudaMswError};
#[cfg(feature = "cuda")]
pub use oscillators::qqe_wrapper::{CudaQqe, CudaQqeError};
#[cfg(feature = "cuda")]
pub use oscillators::rvi_wrapper::{CudaRvi, CudaRviError};
#[cfg(feature = "cuda")]
pub use oscillators::stc_wrapper::{CudaStc, CudaStcError};
#[cfg(feature = "cuda")]
pub mod bollinger_bands_wrapper;
#[cfg(feature = "cuda")]
pub mod dvdiqqe_wrapper;
#[cfg(feature = "cuda")]
pub mod er_wrapper;
#[cfg(feature = "cuda")]
pub mod nadaraya_watson_envelope_wrapper;
#[cfg(feature = "cuda")]
pub mod nvi_wrapper;
#[cfg(feature = "cuda")]
pub mod pfe_wrapper;
#[cfg(feature = "cuda")]
pub mod pvi_wrapper;
#[cfg(feature = "cuda")]
pub mod supertrend_wrapper;
#[cfg(feature = "cuda")]
pub mod ttm_trend_wrapper;
#[cfg(feature = "cuda")]
pub mod vertical_horizontal_filter_wrapper;
#[cfg(feature = "cuda")]
pub mod vpt_wrapper;
#[cfg(feature = "cuda")]
pub mod vwmacd_wrapper;
#[cfg(feature = "cuda")]
pub mod wto_wrapper;
#[cfg(feature = "cuda")]
pub mod zig_zag_channels_wrapper;

#[cfg(feature = "cuda")]
pub use dvdiqqe_wrapper::{CudaDvdiqqe, CudaDvdiqqeError};
#[cfg(feature = "cuda")]
pub use er_wrapper::{CudaEr, CudaErError};
#[cfg(feature = "cuda")]
pub use moving_averages::cwma_wrapper::CudaCwma;
#[cfg(feature = "cuda")]
pub use moving_averages::ehlers_ecema_wrapper::CudaEhlersEcema;
#[cfg(feature = "cuda")]
pub use moving_averages::epma_wrapper::CudaEpma;
#[cfg(feature = "cuda")]
pub use moving_averages::highpass_wrapper::CudaHighpass;
#[cfg(feature = "cuda")]
pub use moving_averages::kama_wrapper::CudaKama;
#[cfg(feature = "cuda")]
pub use moving_averages::nama_wrapper::CudaNama;
#[cfg(feature = "cuda")]
pub use moving_averages::sinwma_wrapper::CudaSinwma;
#[cfg(feature = "cuda")]
pub use moving_averages::supersmoother_3_pole_wrapper::CudaSupersmoother3Pole;
#[cfg(feature = "cuda")]
pub use moving_averages::tradjema_wrapper::CudaTradjema;
#[cfg(feature = "cuda")]
pub use moving_averages::wma_wrapper::CudaWma;
#[cfg(feature = "cuda")]
pub use nadaraya_watson_envelope_wrapper::{CudaNwe, CudaNweError, DeviceNwePair};
#[cfg(feature = "cuda")]
pub use nvi_wrapper::{CudaNvi, CudaNviError};
#[cfg(feature = "cuda")]
pub use pfe_wrapper::{CudaPfe, CudaPfeError};
#[cfg(feature = "cuda")]
pub use pvi_wrapper::{CudaPvi, CudaPviError};
#[cfg(feature = "cuda")]
pub use supertrend_wrapper::{CudaSupertrend, CudaSupertrendError};
#[cfg(feature = "cuda")]
pub use ttm_trend_wrapper::{CudaTtmTrend, CudaTtmTrendError};
#[cfg(feature = "cuda")]
pub use vertical_horizontal_filter_wrapper::{
    CudaVerticalHorizontalFilter, CudaVerticalHorizontalFilterBatchResult,
    CudaVerticalHorizontalFilterError,
};
#[cfg(feature = "cuda")]
pub use vpt_wrapper::{CudaVpt, CudaVptError};
#[cfg(feature = "cuda")]
pub use vwmacd_wrapper::{CudaVwmacd, CudaVwmacdError};
#[cfg(feature = "cuda")]
pub use wto_wrapper::{CudaWto, CudaWtoBatchResult, DeviceArrayF32Triplet};
#[cfg(feature = "cuda")]
pub use zig_zag_channels_wrapper::{CudaZigZagChannels, CudaZigZagChannelsError};
#[cfg(feature = "cuda")]
pub mod bollinger_bands_width_wrapper;
#[cfg(feature = "cuda")]
pub mod bull_power_vs_bear_power_wrapper;
#[cfg(feature = "cuda")]
pub mod bulls_v_bears_wrapper;
#[cfg(feature = "cuda")]
pub mod chandelier_exit_wrapper;
#[cfg(feature = "cuda")]
pub mod cksp_wrapper;
#[cfg(feature = "cuda")]
pub mod correl_hl_wrapper;
#[cfg(feature = "cuda")]
pub mod damiani_volatmeter_wrapper;
#[cfg(feature = "cuda")]
pub mod decisionpoint_breadth_swenlin_trading_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod demand_index_wrapper;
#[cfg(feature = "cuda")]
pub mod deviation_wrapper;
#[cfg(feature = "cuda")]
pub mod devstop_wrapper;
#[cfg(feature = "cuda")]
pub mod didi_index_wrapper;
#[cfg(feature = "cuda")]
pub mod directional_imbalance_index_wrapper;
#[cfg(feature = "cuda")]
pub mod dual_ulcer_index_wrapper;
#[cfg(feature = "cuda")]
pub mod efi_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_fm_demodulator_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_simple_cycle_indicator_wrapper;
#[cfg(feature = "cuda")]
pub mod emd_trend_wrapper;
#[cfg(feature = "cuda")]
pub mod emd_wrapper;
#[cfg(feature = "cuda")]
pub mod exponential_trend_wrapper;
#[cfg(feature = "cuda")]
pub mod fibonacci_entry_bands_wrapper;
#[cfg(feature = "cuda")]
pub mod fibonacci_trailing_stop_wrapper;
#[cfg(feature = "cuda")]
pub mod forward_backward_exponential_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod fractal_dimension_index_wrapper;
#[cfg(feature = "cuda")]
pub mod fvg_positioning_average_wrapper;
#[cfg(feature = "cuda")]
pub mod fvg_trailing_stop_wrapper;
#[cfg(feature = "cuda")]
pub mod garman_klass_volatility_wrapper;
#[cfg(feature = "cuda")]
pub mod geometric_bias_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod gmma_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod gopalakrishnan_range_index_wrapper;
#[cfg(feature = "cuda")]
pub mod grover_llorens_cycle_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod half_causal_estimator_wrapper;
#[cfg(feature = "cuda")]
pub mod halftrend_wrapper;
#[cfg(feature = "cuda")]
pub mod hema_trend_levels_wrapper;
#[cfg(feature = "cuda")]
pub mod historical_volatility_percentile_wrapper;
#[cfg(feature = "cuda")]
pub mod historical_volatility_rank_wrapper;
#[cfg(feature = "cuda")]
pub mod historical_volatility_wrapper;
#[cfg(feature = "cuda")]
pub mod hull_butterfly_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod intraday_momentum_index_wrapper;
#[cfg(feature = "cuda")]
pub mod kaufmanstop_wrapper;
#[cfg(feature = "cuda")]
pub mod kurtosis_wrapper;
#[cfg(feature = "cuda")]
pub mod l1_ehlers_phasor_wrapper;
#[cfg(feature = "cuda")]
pub mod l2_ehlers_signal_to_noise_wrapper;
#[cfg(feature = "cuda")]
pub mod leavitt_convolution_acceleration_wrapper;
#[cfg(feature = "cuda")]
pub mod linear_correlation_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod linear_regression_intensity_wrapper;
#[cfg(feature = "cuda")]
pub mod lpc_wrapper;
#[cfg(feature = "cuda")]
pub mod macd_wave_signal_pro_wrapper;
#[cfg(feature = "cuda")]
pub mod mass_wrapper;
#[cfg(feature = "cuda")]
pub mod mean_ad_wrapper;
#[cfg(feature = "cuda")]
pub mod medium_ad_wrapper;
#[cfg(feature = "cuda")]
pub mod minmax_wrapper;
#[cfg(feature = "cuda")]
pub mod mod_god_mode_wrapper;
#[cfg(feature = "cuda")]
pub mod momentum_ratio_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod monotonicity_index_wrapper;
#[cfg(feature = "cuda")]
pub mod natr_wrapper;
#[cfg(feature = "cuda")]
pub mod neighboring_trailing_stop_wrapper;
#[cfg(feature = "cuda")]
pub mod net_myrsi_wrapper;
#[cfg(feature = "cuda")]
pub mod nonlinear_regression_zero_lag_moving_average_wrapper;
#[cfg(feature = "cuda")]
pub mod normalized_volume_true_range_wrapper;
#[cfg(feature = "cuda")]
pub mod obv_wrapper;
#[cfg(feature = "cuda")]
pub mod on_balance_volume_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod parkinson_volatility_wrapper;
#[cfg(feature = "cuda")]
pub mod pattern_recognition_wrapper;
#[cfg(feature = "cuda")]
pub mod percentile_nearest_rank_wrapper;
#[cfg(feature = "cuda")]
pub mod pivot_wrapper;
#[cfg(feature = "cuda")]
pub mod polynomial_regression_extrapolation_wrapper;
#[cfg(feature = "cuda")]
pub mod prb_wrapper;
#[cfg(feature = "cuda")]
pub mod premier_rsi_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod pretty_good_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod price_density_market_noise_wrapper;
#[cfg(feature = "cuda")]
pub mod projection_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod psychological_line_wrapper;
#[cfg(feature = "cuda")]
pub mod qqe_weighted_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod random_walk_index_wrapper;
#[cfg(feature = "cuda")]
pub mod range_filter_wrapper;
#[cfg(feature = "cuda")]
pub mod rank_correlation_index_wrapper;
#[cfg(feature = "cuda")]
pub mod regression_slope_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod relative_strength_index_wave_indicator_wrapper;
#[cfg(feature = "cuda")]
pub mod reversal_signals_wrapper;
#[cfg(feature = "cuda")]
pub mod rogers_satchell_volatility_wrapper;
#[cfg(feature = "cuda")]
pub mod rolling_skewness_kurtosis_wrapper;
#[cfg(feature = "cuda")]
pub mod rolling_z_score_trend_wrapper;
#[cfg(feature = "cuda")]
pub mod safezonestop_wrapper;
#[cfg(feature = "cuda")]
pub mod sar_wrapper;
#[cfg(feature = "cuda")]
pub mod smoothed_gaussian_trend_filter_wrapper;
#[cfg(feature = "cuda")]
pub mod spearman_correlation_wrapper;
#[cfg(feature = "cuda")]
pub mod squeeze_index_wrapper;
#[cfg(feature = "cuda")]
pub mod stddev_wrapper;
#[cfg(feature = "cuda")]
pub mod stochastic_adaptive_d_wrapper;
#[cfg(feature = "cuda")]
pub mod stochastic_connors_rsi_wrapper;
#[cfg(feature = "cuda")]
pub mod stochastic_distance_wrapper;
#[cfg(feature = "cuda")]
pub mod stochastic_money_flow_index_wrapper;
#[cfg(feature = "cuda")]
pub mod trend_continuation_factor_wrapper;
#[cfg(feature = "cuda")]
pub mod trend_direction_force_index_wrapper;
#[cfg(feature = "cuda")]
pub mod trend_flow_trail_wrapper;
#[cfg(feature = "cuda")]
pub mod trend_follower_wrapper;
#[cfg(feature = "cuda")]
pub mod trend_trigger_factor_wrapper;
#[cfg(feature = "cuda")]
pub mod twiggs_money_flow_wrapper;
#[cfg(feature = "cuda")]
pub mod ui_wrapper;
#[cfg(feature = "cuda")]
pub mod var_wrapper;
#[cfg(feature = "cuda")]
pub mod velocity_acceleration_indicator_wrapper;
#[cfg(feature = "cuda")]
pub mod velocity_wrapper;
#[cfg(feature = "cuda")]
pub mod vi_wrapper;
#[cfg(feature = "cuda")]
pub mod volatility_quality_index_wrapper;
#[cfg(feature = "cuda")]
pub mod volatility_ratio_adaptive_rsx_wrapper;
#[cfg(feature = "cuda")]
pub mod volume_energy_reservoirs_wrapper;
#[cfg(feature = "cuda")]
pub mod volume_weighted_relative_strength_index_wrapper;
#[cfg(feature = "cuda")]
pub mod volume_weighted_rsi_wrapper;
#[cfg(feature = "cuda")]
pub mod volume_zone_oscillator_wrapper;
#[cfg(feature = "cuda")]
pub mod vosc_wrapper;
#[cfg(feature = "cuda")]
pub mod voss_wrapper;
#[cfg(feature = "cuda")]
pub mod vpci_wrapper;
#[cfg(feature = "cuda")]
pub mod wad_wrapper;
#[cfg(feature = "cuda")]
pub mod yang_zhang_volatility_wrapper;
#[cfg(feature = "cuda")]
pub mod zscore_wrapper;

#[cfg(feature = "cuda")]
pub use oscillators::{CudaDecOsc, CudaDecOscError};
#[cfg(feature = "cuda")]
pub use oscillators::{CudaFisher, CudaFisherError};
#[cfg(feature = "cuda")]
pub use oscillators::{CudaIftRsi, CudaIftRsiError};
#[cfg(feature = "cuda")]
pub use oscillators::{CudaMfi, CudaMfiError};

#[cfg(feature = "cuda")]
pub use bollinger_bands_width_wrapper::{CudaBbw, CudaBbwError};
#[cfg(feature = "cuda")]
pub use bull_power_vs_bear_power_wrapper::{
    CudaBullPowerVsBearPower, CudaBullPowerVsBearPowerError,
};
#[cfg(feature = "cuda")]
pub use bulls_v_bears_wrapper::{CudaBullsVBears, CudaBullsVBearsError};
#[cfg(feature = "cuda")]
pub use chande_wrapper::CudaChandeError;
#[cfg(feature = "cuda")]
pub use cksp_wrapper::{CudaCksp, CudaCkspError};
#[cfg(feature = "cuda")]
pub use decisionpoint_breadth_swenlin_trading_oscillator_wrapper::{
    CudaDecisionPointBreadthSwenlinTradingOscillator,
    CudaDecisionPointBreadthSwenlinTradingOscillatorError,
};
#[cfg(feature = "cuda")]
pub use demand_index_wrapper::{CudaDemandIndex, CudaDemandIndexError};
#[cfg(feature = "cuda")]
pub use deviation_wrapper::{CudaDeviation, CudaDeviationError};
#[cfg(feature = "cuda")]
pub use devstop_wrapper::{CudaDevStop, CudaDevStopError};
#[cfg(feature = "cuda")]
pub use didi_index_wrapper::{CudaDidiIndex, CudaDidiIndexError};
#[cfg(feature = "cuda")]
pub use directional_imbalance_index_wrapper::{
    CudaDirectionalImbalanceIndex, CudaDirectionalImbalanceIndexError,
};
#[cfg(feature = "cuda")]
pub use dual_ulcer_index_wrapper::{CudaDualUlcerIndex, CudaDualUlcerIndexError};
#[cfg(feature = "cuda")]
pub use ehlers_fm_demodulator_wrapper::{CudaEhlersFmDemodulator, CudaEhlersFmDemodulatorError};
#[cfg(feature = "cuda")]
pub use ehlers_simple_cycle_indicator_wrapper::{
    CudaEhlersSimpleCycleIndicator, CudaEhlersSimpleCycleIndicatorError,
};
#[cfg(feature = "cuda")]
pub use emd_trend_wrapper::{CudaEmdTrend, CudaEmdTrendError};
#[cfg(feature = "cuda")]
pub use emd_wrapper::{CudaEmd, CudaEmdBatchResult, CudaEmdError, DeviceArrayF32Triple};
#[cfg(feature = "cuda")]
pub use exponential_trend_wrapper::{CudaExponentialTrend, CudaExponentialTrendError};
#[cfg(feature = "cuda")]
pub use fibonacci_entry_bands_wrapper::{CudaFibonacciEntryBands, CudaFibonacciEntryBandsError};
#[cfg(feature = "cuda")]
pub use fibonacci_trailing_stop_wrapper::{
    CudaFibonacciTrailingStop, CudaFibonacciTrailingStopError,
};
#[cfg(feature = "cuda")]
pub use forward_backward_exponential_oscillator_wrapper::{
    CudaForwardBackwardExponentialOscillator, CudaForwardBackwardExponentialOscillatorError,
};
#[cfg(feature = "cuda")]
pub use fractal_dimension_index_wrapper::{
    CudaFractalDimensionIndex, CudaFractalDimensionIndexError,
};
#[cfg(feature = "cuda")]
pub use fvg_positioning_average_wrapper::{
    CudaFvgPositioningAverage, CudaFvgPositioningAverageError,
};
#[cfg(feature = "cuda")]
pub use fvg_trailing_stop_wrapper::{CudaFvgTs, CudaFvgTsError};
#[cfg(feature = "cuda")]
pub use garman_klass_volatility_wrapper::{
    CudaGarmanKlassBatchResult, CudaGarmanKlassVolatility, CudaGarmanKlassVolatilityError,
};
#[cfg(feature = "cuda")]
pub use geometric_bias_oscillator_wrapper::{
    CudaGeometricBiasOscillator, CudaGeometricBiasOscillatorError,
};
#[cfg(feature = "cuda")]
pub use gmma_oscillator_wrapper::{CudaGmmaOscillator, CudaGmmaOscillatorError};
#[cfg(feature = "cuda")]
pub use gopalakrishnan_range_index_wrapper::{
    CudaGopalakrishnanRangeIndex, CudaGopalakrishnanRangeIndexBatchResult,
    CudaGopalakrishnanRangeIndexError,
};
#[cfg(feature = "cuda")]
pub use grover_llorens_cycle_oscillator_wrapper::{
    CudaGroverLlorensCycleOscillator, CudaGroverLlorensCycleOscillatorError,
};
#[cfg(feature = "cuda")]
pub use historical_volatility_percentile_wrapper::{
    CudaHistoricalVolatilityPercentile, CudaHistoricalVolatilityPercentileError,
};
#[cfg(feature = "cuda")]
pub use historical_volatility_rank_wrapper::{
    CudaHistoricalVolatilityRank, CudaHistoricalVolatilityRankError,
};
#[cfg(feature = "cuda")]
pub use historical_volatility_wrapper::{
    CudaHistoricalVolatility, CudaHistoricalVolatilityBatchResult, CudaHistoricalVolatilityError,
};
#[cfg(feature = "cuda")]
pub use hull_butterfly_oscillator_wrapper::{
    CudaHullButterflyOscillator, CudaHullButterflyOscillatorError,
};
#[cfg(feature = "cuda")]
pub use intraday_momentum_index_wrapper::{
    CudaIntradayMomentumIndex, CudaIntradayMomentumIndexError,
};
#[cfg(feature = "cuda")]
pub use kaufmanstop_wrapper::{CudaKaufmanstop, CudaKaufmanstopError};
#[cfg(feature = "cuda")]
pub use l1_ehlers_phasor_wrapper::{CudaL1EhlersPhasor, CudaL1EhlersPhasorError};
#[cfg(feature = "cuda")]
pub use l2_ehlers_signal_to_noise_wrapper::{
    CudaL2EhlersSignalToNoise, CudaL2EhlersSignalToNoiseError,
};
#[cfg(feature = "cuda")]
pub use leavitt_convolution_acceleration_wrapper::{
    CudaLeavittConvolutionAcceleration, CudaLeavittConvolutionAccelerationError,
};
#[cfg(feature = "cuda")]
pub use linear_correlation_oscillator_wrapper::{
    CudaLinearCorrelationOscillator, CudaLinearCorrelationOscillatorError,
};
#[cfg(feature = "cuda")]
pub use linear_regression_intensity_wrapper::{
    CudaLinearRegressionIntensity, CudaLinearRegressionIntensityError,
};
#[cfg(feature = "cuda")]
pub use macd_wave_signal_pro_wrapper::{CudaMacdWaveSignalPro, CudaMacdWaveSignalProError};
#[cfg(feature = "cuda")]
pub use mass_wrapper::{CudaMass, CudaMassError};
#[cfg(feature = "cuda")]
pub use mean_ad_wrapper::{CudaMeanAd, CudaMeanAdError};
#[cfg(feature = "cuda")]
pub use medium_ad_wrapper::{CudaMediumAd, CudaMediumAdError};
#[cfg(feature = "cuda")]
pub use minmax_wrapper::{CudaMinmax, CudaMinmaxError};
#[cfg(feature = "cuda")]
pub use mod_god_mode_wrapper::{CudaModGodMode, CudaModGodModeBatchResult};
#[cfg(feature = "cuda")]
pub use momentum_ratio_oscillator_wrapper::{
    CudaMomentumRatioOscillator, CudaMomentumRatioOscillatorError,
};
#[cfg(feature = "cuda")]
pub use monotonicity_index_wrapper::{CudaMonotonicityIndex, CudaMonotonicityIndexError};
#[cfg(feature = "cuda")]
pub use moving_averages::{
    CudaApo, CudaBuffAverages, CudaBuffAveragesError, CudaFrama, CudaFramaError, CudaHma,
    CudaHmaError, CudaLinearregSlope, CudaLinearregSlopeError, CudaLinreg, CudaLinregError,
    CudaLinregIntercept, CudaLinregInterceptError, CudaNma, CudaNmaError, CudaSma, CudaSmaError,
    CudaSuperSmoother, CudaSuperSmootherError, CudaTrendflex, CudaTrendflexError, CudaTsf,
    CudaTsfError, CudaVidya, CudaVidyaError, CudaVlma, CudaVolumeAdjustedMa,
    CudaVolumeAdjustedMaError, CudaVpwma, CudaVpwmaError, CudaZlema, CudaZlemaError,
};
#[cfg(feature = "cuda")]
pub use natr_wrapper::{CudaNatr, CudaNatrError};
#[cfg(feature = "cuda")]
pub use neighboring_trailing_stop_wrapper::{
    CudaNeighboringTrailingStop, CudaNeighboringTrailingStopError,
};
#[cfg(feature = "cuda")]
pub use net_myrsi_wrapper::{CudaNetMyrsi, CudaNetMyrsiError};
#[cfg(feature = "cuda")]
pub use nonlinear_regression_zero_lag_moving_average_wrapper::{
    CudaNonlinearRegressionZeroLagMovingAverage, CudaNonlinearRegressionZeroLagMovingAverageError,
};
#[cfg(feature = "cuda")]
pub use normalized_volume_true_range_wrapper::{
    CudaNormalizedVolumeTrueRange, CudaNormalizedVolumeTrueRangeError,
};
#[cfg(feature = "cuda")]
pub use oscillators::adosc_wrapper::{CudaAdosc, CudaAdoscError};
#[cfg(feature = "cuda")]
pub use oscillators::ao_wrapper::{CudaAo, CudaAoError};
#[cfg(feature = "cuda")]
pub use oscillators::cfo_wrapper::{CudaCfo, CudaCfoError};
#[cfg(feature = "cuda")]
pub use oscillators::coppock_wrapper::{CudaCoppock, CudaCoppockError};
#[cfg(feature = "cuda")]
pub use oscillators::dpo_wrapper::{CudaDpo, CudaDpoError};
#[cfg(feature = "cuda")]
pub use oscillators::fosc_wrapper::{CudaFosc, CudaFoscError};
#[cfg(feature = "cuda")]
pub use oscillators::gatorosc_wrapper::{CudaGatorOsc, CudaGatorOscError};
#[cfg(feature = "cuda")]
pub use oscillators::kvo_wrapper::{CudaKvo, CudaKvoError};
#[cfg(feature = "cuda")]
pub use oscillators::macd_wrapper::{CudaMacd, CudaMacdError};
#[cfg(feature = "cuda")]
pub use oscillators::ppo_wrapper::{CudaPpo, CudaPpoError};
#[cfg(feature = "cuda")]
pub use oscillators::tsi_wrapper::{CudaTsi, CudaTsiError};
#[cfg(feature = "cuda")]
pub use parkinson_volatility_wrapper::{
    CudaParkinsonVolatility, CudaParkinsonVolatilityBatchResult, CudaParkinsonVolatilityError,
    ParkinsonDeviceArrayF32Pair,
};
#[cfg(feature = "cuda")]
pub use pattern_recognition_wrapper::{
    CudaPatternRecognition, CudaPatternRecognitionError, DevicePatternFeatures, NativeSubsetRows,
};
#[cfg(feature = "cuda")]
pub use percentile_nearest_rank_wrapper::{CudaPercentileNearestRank, CudaPnrError};
#[cfg(feature = "cuda")]
pub use polynomial_regression_extrapolation_wrapper::{
    CudaPolynomialRegressionExtrapolation, CudaPolynomialRegressionExtrapolationError,
};
#[cfg(feature = "cuda")]
pub use prb_wrapper::{CudaPrb, CudaPrbError};
#[cfg(feature = "cuda")]
pub use premier_rsi_oscillator_wrapper::{CudaPremierRsiOscillator, CudaPremierRsiOscillatorError};
#[cfg(feature = "cuda")]
pub use price_density_market_noise_wrapper::{
    CudaPriceDensityMarketNoise, CudaPriceDensityMarketNoiseError,
};
#[cfg(feature = "cuda")]
pub use projection_oscillator_wrapper::{CudaProjectionOscillator, CudaProjectionOscillatorError};
#[cfg(feature = "cuda")]
pub use psychological_line_wrapper::{CudaPsychologicalLine, CudaPsychologicalLineError};
#[cfg(feature = "cuda")]
pub use qqe_weighted_oscillator_wrapper::{
    CudaQqeWeightedOscillator, CudaQqeWeightedOscillatorError,
};
#[cfg(feature = "cuda")]
pub use random_walk_index_wrapper::{CudaRandomWalkIndex, CudaRandomWalkIndexError};
#[cfg(feature = "cuda")]
pub use range_filter_wrapper::{CudaRangeFilter, CudaRangeFilterError, DeviceRangeFilterTrio};
#[cfg(feature = "cuda")]
pub use rank_correlation_index_wrapper::{CudaRankCorrelationIndex, CudaRankCorrelationIndexError};
#[cfg(feature = "cuda")]
pub use regression_slope_oscillator_wrapper::{
    CudaRegressionSlopeOscillator, CudaRegressionSlopeOscillatorError,
};
#[cfg(feature = "cuda")]
pub use relative_strength_index_wave_indicator_wrapper::{
    CudaRelativeStrengthIndexWaveIndicator, CudaRelativeStrengthIndexWaveIndicatorError,
};
#[cfg(feature = "cuda")]
pub use reversal_signals_wrapper::{CudaReversalSignals, CudaReversalSignalsError};
#[cfg(feature = "cuda")]
pub use rogers_satchell_volatility_wrapper::{
    CudaRogersSatchellBatchResult, CudaRogersSatchellManySeriesResult,
    CudaRogersSatchellVolatility, CudaRogersSatchellVolatilityError,
    DeviceArrayF32Pair as RogersSatchellDeviceArrayF32Pair,
};
#[cfg(feature = "cuda")]
pub use rolling_skewness_kurtosis_wrapper::{
    CudaRollingSkewnessKurtosis, CudaRollingSkewnessKurtosisError,
};
#[cfg(feature = "cuda")]
pub use rolling_z_score_trend_wrapper::{CudaRollingZScoreTrend, CudaRollingZScoreTrendError};
#[cfg(feature = "cuda")]
pub use sar_wrapper::{CudaSar, CudaSarError};
#[cfg(feature = "cuda")]
pub use smoothed_gaussian_trend_filter_wrapper::{
    CudaSmoothedGaussianTrendFilter, CudaSmoothedGaussianTrendFilterError,
};
#[cfg(feature = "cuda")]
pub use spearman_correlation_wrapper::{CudaSpearmanCorrelation, CudaSpearmanCorrelationError};
#[cfg(feature = "cuda")]
pub use squeeze_index_wrapper::{CudaSqueezeIndex, CudaSqueezeIndexError};
#[cfg(feature = "cuda")]
pub use stochastic_adaptive_d_wrapper::{CudaStochasticAdaptiveD, CudaStochasticAdaptiveDError};
#[cfg(feature = "cuda")]
pub use stochastic_connors_rsi_wrapper::{CudaStochasticConnorsRsi, CudaStochasticConnorsRsiError};
#[cfg(feature = "cuda")]
pub use stochastic_distance_wrapper::{CudaStochasticDistance, CudaStochasticDistanceError};
#[cfg(feature = "cuda")]
pub use stochastic_money_flow_index_wrapper::{
    CudaStochasticMoneyFlowIndex, CudaStochasticMoneyFlowIndexError,
};
#[cfg(feature = "cuda")]
pub use trend_continuation_factor_wrapper::{
    CudaTrendContinuationFactor, CudaTrendContinuationFactorError,
};
#[cfg(feature = "cuda")]
pub use trend_direction_force_index_wrapper::{
    CudaTrendDirectionForceIndex, CudaTrendDirectionForceIndexError,
};
#[cfg(feature = "cuda")]
pub use trend_flow_trail_wrapper::{CudaTrendFlowTrail, CudaTrendFlowTrailError};
#[cfg(feature = "cuda")]
pub use trend_follower_wrapper::{CudaTrendFollower, CudaTrendFollowerError};
#[cfg(feature = "cuda")]
pub use trend_trigger_factor_wrapper::{CudaTrendTriggerFactor, CudaTrendTriggerFactorError};
#[cfg(feature = "cuda")]
pub use twiggs_money_flow_wrapper::{CudaTwiggsMoneyFlow, CudaTwiggsMoneyFlowError};
#[cfg(feature = "cuda")]
pub use var_wrapper::{CudaVar, CudaVarError};
#[cfg(feature = "cuda")]
pub use velocity_acceleration_indicator_wrapper::{
    CudaVelocityAccelerationIndicator, CudaVelocityAccelerationIndicatorError,
};
#[cfg(feature = "cuda")]
pub use velocity_wrapper::{CudaVelocity, CudaVelocityError};
#[cfg(feature = "cuda")]
pub use vi_wrapper::{CudaVi, CudaViError};
#[cfg(feature = "cuda")]
pub use volatility_quality_index_wrapper::{
    CudaVolatilityQualityIndex, CudaVolatilityQualityIndexError,
};
#[cfg(feature = "cuda")]
pub use volatility_ratio_adaptive_rsx_wrapper::{
    CudaVolatilityRatioAdaptiveRsx, CudaVolatilityRatioAdaptiveRsxError,
};
#[cfg(feature = "cuda")]
pub use volume_energy_reservoirs_wrapper::{
    CudaVolumeEnergyReservoirs, CudaVolumeEnergyReservoirsError,
};
#[cfg(feature = "cuda")]
pub use volume_weighted_relative_strength_index_wrapper::{
    CudaVolumeWeightedRelativeStrengthIndex, CudaVolumeWeightedRelativeStrengthIndexError,
};
#[cfg(feature = "cuda")]
pub use volume_weighted_rsi_wrapper::{CudaVolumeWeightedRsi, CudaVolumeWeightedRsiError};
#[cfg(feature = "cuda")]
pub use volume_zone_oscillator_wrapper::{CudaVolumeZoneOscillator, CudaVolumeZoneOscillatorError};
#[cfg(feature = "cuda")]
pub use voss_wrapper::{CudaVoss, CudaVossError};
#[cfg(feature = "cuda")]
pub use vpci_wrapper::{CudaVpci, CudaVpciError};
#[cfg(feature = "cuda")]
pub use wad_wrapper::{CudaWad, CudaWadError};
#[cfg(feature = "cuda")]
pub use yang_zhang_volatility_wrapper::{
    CudaYangZhangBatchResult, CudaYangZhangVolatility, CudaYangZhangVolatilityError,
};
#[cfg(feature = "cuda")]
pub use zscore_wrapper::{CudaZscore, CudaZscoreError};
#[cfg(feature = "cuda")]
pub mod linearreg_angle_wrapper;
#[cfg(feature = "cuda")]
pub use bollinger_bands_wrapper::{CudaBollingerBands, CudaBollingerError};
#[cfg(feature = "cuda")]
pub use chandelier_exit_wrapper::{CudaCeError, CudaChandelierExit};
#[cfg(feature = "cuda")]
pub use correl_hl_wrapper::{CudaCorrelHl, CudaCorrelHlError};
#[cfg(feature = "cuda")]
pub use damiani_volatmeter_wrapper::{CudaDamianiError, CudaDamianiVolatmeter};
#[cfg(feature = "cuda")]
pub use efi_wrapper::{CudaEfi, CudaEfiError};
#[cfg(feature = "cuda")]
pub use half_causal_estimator_wrapper::{CudaHalfCausalEstimator, CudaHalfCausalEstimatorError};
#[cfg(feature = "cuda")]
pub use halftrend_wrapper::{CudaHalftrend, CudaHalftrendError};
#[cfg(feature = "cuda")]
pub use hema_trend_levels_wrapper::{CudaHemaTrendLevels, CudaHemaTrendLevelsError};
#[cfg(feature = "cuda")]
pub use kurtosis_wrapper::{CudaKurtosis, CudaKurtosisError};
#[cfg(feature = "cuda")]
pub use linearreg_angle_wrapper::{CudaLinearregAngle, CudaLinearregAngleError};
#[cfg(feature = "cuda")]
pub use lpc_wrapper::{
    BatchKernelPolicy as LpcBatchKernelPolicy, CudaLpc, CudaLpcError, CudaLpcPolicy,
    ManySeriesKernelPolicy as LpcManySeriesKernelPolicy,
};
#[cfg(feature = "cuda")]
pub use obv_wrapper::{CudaObv, CudaObvError};
#[cfg(feature = "cuda")]
pub use on_balance_volume_oscillator_wrapper::{
    CudaOnBalanceVolumeOscillator, CudaOnBalanceVolumeOscillatorError,
};
#[cfg(feature = "cuda")]
pub use oscillators::cg_wrapper::{CudaCg, CudaCgError};
#[cfg(feature = "cuda")]
pub use oscillators::cmo_wrapper::{CudaCmo, CudaCmoError};
#[cfg(feature = "cuda")]
pub use oscillators::dti_wrapper::{CudaDti, CudaDtiError};
#[cfg(feature = "cuda")]
pub use oscillators::emv_wrapper::{CudaEmv, CudaEmvError};
#[cfg(feature = "cuda")]
pub use oscillators::kdj_wrapper::{CudaKdj, CudaKdjError};
#[cfg(feature = "cuda")]
pub use oscillators::reverse_rsi_wrapper::{CudaReverseRsi, CudaReverseRsiError};
#[cfg(feature = "cuda")]
pub use oscillators::squeeze_momentum_wrapper::{CudaSmiError, CudaSqueezeMomentum};
#[cfg(feature = "cuda")]
pub use oscillators::stochf_wrapper::{CudaStochf, CudaStochfError};
#[cfg(feature = "cuda")]
pub use oscillators::ttm_squeeze_wrapper::{CudaTtmSqueeze, CudaTtmSqueezeError};
#[cfg(feature = "cuda")]
pub use pivot_wrapper::{CudaPivot, CudaPivotError};
#[cfg(feature = "cuda")]
pub use pretty_good_oscillator_wrapper::{CudaPrettyGoodOscillator, CudaPrettyGoodOscillatorError};
#[cfg(feature = "cuda")]
pub use safezonestop_wrapper::{CudaSafeZoneStop, CudaSafeZoneStopError};
#[cfg(feature = "cuda")]
pub use stddev_wrapper::{CudaStddev, CudaStddevError};
#[cfg(feature = "cuda")]
pub use ui_wrapper::{CudaUi, CudaUiError};
#[cfg(feature = "cuda")]
pub use vosc_wrapper::{
    BatchKernelPolicy as VoscBatchKernelPolicy, CudaVosc, CudaVoscError, CudaVoscPolicy,
    ManySeriesKernelPolicy as VoscManySeriesKernelPolicy,
};

#[cfg(all(feature = "cuda", test))]
pub(crate) struct CudaTestLock {
    _guard: Option<MutexGuard<'static, ()>>,
}

#[cfg(all(feature = "cuda", test))]
impl Drop for CudaTestLock {
    fn drop(&mut self) {
        CUDA_TEST_LOCK_DEPTH.with(|depth| {
            let current = depth.get();
            debug_assert!(current > 0);
            depth.set(current.saturating_sub(1));
        });
    }
}

#[cfg(all(feature = "cuda", test))]
thread_local! {
    static CUDA_TEST_LOCK_DEPTH: Cell<usize> = const { Cell::new(0) };
}

#[cfg(all(feature = "cuda", test))]
pub(crate) fn cuda_test_lock() -> CudaTestLock {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = CUDA_TEST_LOCK_DEPTH.with(|depth| {
        let current = depth.get();
        depth.set(current + 1);
        if current == 0 {
            Some(
                LOCK.get_or_init(|| Mutex::new(()))
                    .lock()
                    .unwrap_or_else(|err| err.into_inner()),
            )
        } else {
            None
        }
    });
    CudaTestLock { _guard: guard }
}

#[inline]
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        static CUDA_AVAILABLE_CACHED: OnceLock<bool> = OnceLock::new();
        static CUDA_PROBE_CONTEXT_0: OnceLock<Option<cust::context::Context>> = OnceLock::new();

        if std::env::var("CUDA_PLACEHOLDER_ON_FAIL").ok().as_deref() == Some("1")
            || std::env::var("CUDA_FORCE_SKIP").ok().as_deref() == Some("1")
        {
            return false;
        }
        *CUDA_AVAILABLE_CACHED.get_or_init(|| {
            use cust::{
                device::Device,
                function::BlockSize,
                function::GridSize,
                module::Module,
                prelude::CudaFlags,
                stream::{Stream, StreamFlags},
            };

            let debug = std::env::var("CUDA_PROBE_DEBUG").ok().as_deref() == Some("1");

            if let Err(err) = cust::init(CudaFlags::empty()) {
                if debug {
                    eprintln!("cuda_available: cust::init failed: {err:?}");
                }
                return false;
            }

            let ndev = match Device::num_devices() {
                Ok(n) => n,
                Err(err) => {
                    if debug {
                        eprintln!("cuda_available: Device::num_devices failed: {err:?}");
                    }
                    0
                }
            };
            if ndev == 0 {
                if debug {
                    eprintln!("cuda_available: no CUDA devices reported");
                }
                return false;
            }

            const PROBE_PTXS: [&str; 3] = [
                r#"
                    .version 9.0
                    .target sm_52
                    .address_size 64
                    .visible .entry probe() {
                        ret;
                    }
                "#,
                r#"
                    .version 8.0
                    .target sm_52
                    .address_size 64
                    .visible .entry probe() {
                        ret;
                    }
                "#,
                r#"
                    .version 7.0
                    .target compute_52
                    .address_size 64
                    .visible .entry probe() {
                        ret;
                    }
                "#,
            ];

            let device = match Device::get_device(0) {
                Ok(d) => d,
                Err(err) => {
                    if debug {
                        eprintln!("cuda_available: Device::get_device(0) failed: {err:?}");
                    }
                    return false;
                }
            };

            let ctx0 =
                CUDA_PROBE_CONTEXT_0.get_or_init(|| match cust::context::Context::new(device) {
                    Ok(c) => Some(c),
                    Err(err) => {
                        if debug {
                            eprintln!("cuda_available: Context::new failed: {err:?}");
                        }
                        None
                    }
                });
            if ctx0.is_none() {
                return false;
            }

            let mut module_opt = None;
            if debug {
                eprintln!(
                    "cuda_available: trying PTX probe module load ({} candidates)",
                    PROBE_PTXS.len()
                );
            }
            for (idx, &ptx) in PROBE_PTXS.iter().enumerate() {
                match Module::from_ptx(ptx, &[]) {
                    Ok(m) => {
                        module_opt = Some(m);
                        if debug {
                            eprintln!("cuda_available: probe PTX loaded (candidate #{idx})");
                        }
                        break;
                    }
                    Err(err) => {
                        if debug {
                            eprintln!(
                                "cuda_available: probe PTX load failed (candidate #{idx}): {err:?}"
                            );
                        }
                    }
                }
            }
            let module = match module_opt {
                Some(m) => m,
                None => return false,
            };
            let func = match module.get_function("probe") {
                Ok(f) => f,
                Err(err) => {
                    if debug {
                        eprintln!("cuda_available: module.get_function(\"probe\") failed: {err:?}");
                    }
                    return false;
                }
            };
            let stream = match Stream::new(StreamFlags::NON_BLOCKING, None) {
                Ok(s) => s,
                Err(err) => {
                    if debug {
                        eprintln!("cuda_available: Stream::new failed: {err:?}");
                    }
                    return false;
                }
            };
            unsafe {
                let args: &mut [*mut std::ffi::c_void] = &mut [];
                if let Err(err) =
                    stream.launch(&func, GridSize::xy(1, 1), BlockSize::xyz(1, 1, 1), 0, args)
                {
                    if debug {
                        eprintln!("cuda_available: stream.launch failed: {err:?}");
                    }
                    return false;
                }
            }
            if let Err(err) = stream.synchronize() {
                if debug {
                    eprintln!("cuda_available: stream.synchronize failed: {err:?}");
                }
                return false;
            }
            true
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[inline]
pub fn cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        use cust::{device::Device, prelude::CudaFlags};
        if cust::init(CudaFlags::empty()).is_err() {
            return 0;
        }
        match Device::num_devices() {
            Ok(n) => n as usize,
            Err(_) => 0,
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}
