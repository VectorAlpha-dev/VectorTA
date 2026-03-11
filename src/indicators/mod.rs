pub mod accumulation_swing_index;
pub mod acosc;
pub mod ad;
pub mod adaptive_bounds_rsi;
pub mod adjustable_ma_alternating_extremities;
pub mod adaptive_macd;
pub mod adaptive_momentum_oscillator;
pub mod adosc;
pub mod adx;
pub mod adxr;
pub mod alligator;
pub mod alphatrend;
pub mod dispatch;
pub mod standardized_psar_oscillator;
pub mod statistical_trailing_stop;
pub mod supertrend_recovery;
pub use alphatrend::{alphatrend, AlphaTrendInput, AlphaTrendOutput, AlphaTrendParams};
pub mod andean_oscillator;
pub mod ao;
pub mod apo;
pub mod aroon;
pub mod aroonosc;
pub mod aso;
pub use adaptive_bounds_rsi::{
    adaptive_bounds_rsi, adaptive_bounds_rsi_batch_with_kernel, adaptive_bounds_rsi_into,
    adaptive_bounds_rsi_into_slices, adaptive_bounds_rsi_with_kernel,
    AdaptiveBoundsRsiBatchBuilder, AdaptiveBoundsRsiBatchOutput, AdaptiveBoundsRsiBatchRange,
    AdaptiveBoundsRsiBuilder, AdaptiveBoundsRsiData, AdaptiveBoundsRsiError,
    AdaptiveBoundsRsiInput, AdaptiveBoundsRsiOutput, AdaptiveBoundsRsiParams,
    AdaptiveBoundsRsiStream,
};
pub use adjustable_ma_alternating_extremities::{
    adjustable_ma_alternating_extremities, adjustable_ma_alternating_extremities_batch_with_kernel,
    adjustable_ma_alternating_extremities_into, adjustable_ma_alternating_extremities_into_slices,
    adjustable_ma_alternating_extremities_with_kernel,
    AdjustableMaAlternatingExtremitiesBatchBuilder, AdjustableMaAlternatingExtremitiesBatchOutput,
    AdjustableMaAlternatingExtremitiesBatchRange, AdjustableMaAlternatingExtremitiesBuilder,
    AdjustableMaAlternatingExtremitiesData, AdjustableMaAlternatingExtremitiesError,
    AdjustableMaAlternatingExtremitiesInput, AdjustableMaAlternatingExtremitiesOutput,
    AdjustableMaAlternatingExtremitiesParams, AdjustableMaAlternatingExtremitiesStream,
};
pub use aso::{aso, AsoInput, AsoOutput, AsoParams};
pub mod atr;
pub mod avsl;
pub use avsl::{
    avsl, avsl_batch_with_kernel, avsl_into_slice, avsl_with_kernel, AvslBatchBuilder,
    AvslBatchOutput, AvslBatchRange, AvslBuilder, AvslData, AvslError, AvslInput, AvslOutput,
    AvslParams,
};
pub mod bandpass;
pub mod bollinger_bands;
pub mod bollinger_bands_width;
pub mod bop;
pub mod bulls_v_bears;
pub mod cci;
pub mod cci_cycle;
pub use cci_cycle::{cci_cycle, CciCycleInput, CciCycleOutput, CciCycleParams};
pub mod cfo;
pub mod cg;
pub mod chande;
pub mod chandelier_exit;
pub use chandelier_exit::{
    ce_batch_par_slice, ce_batch_slice, ce_batch_with_kernel, chandelier_exit,
    chandelier_exit_into_flat, chandelier_exit_into_slices, chandelier_exit_with_kernel,
    CeBatchBuilder, CeBatchOutput, CeBatchRange, ChandelierExitBuilder, ChandelierExitData,
    ChandelierExitError, ChandelierExitInput, ChandelierExitOutput, ChandelierExitParams,
};
pub mod chop;
pub mod cksp;
pub mod cmo;
pub mod coppock;
pub mod cora_wave;
pub use cora_wave::{cora_wave, CoraWaveInput, CoraWaveOutput, CoraWaveParams};
pub mod correl_hl;
pub mod correlation_cycle;
pub use correlation_cycle::{
    correlation_cycle, CorrelationCycleBatchBuilder, CorrelationCycleBatchOutput,
    CorrelationCycleBatchRange, CorrelationCycleBuilder, CorrelationCycleError,
    CorrelationCycleInput, CorrelationCycleOutput, CorrelationCycleParams, CorrelationCycleStream,
};
pub mod cvi;
pub use cvi::{
    cvi, CviBatchBuilder, CviBatchOutput, CviBatchRange, CviBuilder, CviData, CviError, CviInput,
    CviOutput, CviParams, CviStream,
};
pub mod cycle_channel_oscillator;
pub mod daily_factor;
pub mod damiani_volatmeter;
pub mod dec_osc;
pub mod decycler;
pub mod deviation;
pub use deviation::{deviation, DeviationInput, DeviationOutput, DeviationParams};
pub mod devstop;
pub use devstop::{devstop, DevStopData, DevStopError, DevStopInput, DevStopOutput, DevStopParams};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use devstop::{
    devstop_alloc, devstop_batch_unified_js, devstop_free, devstop_into_js, devstop_js,
};
#[cfg(feature = "python")]
pub use devstop::{devstop_batch_py, devstop_py};
pub mod di;
pub mod dm;
pub mod donchian;
pub mod dpo;
pub mod dti;
pub mod dvdiqqe;
pub use dvdiqqe::{
    dvdiqqe, dvdiqqe_batch_par_slice, dvdiqqe_batch_slice, dvdiqqe_batch_with_kernel,
    dvdiqqe_into_slices, dvdiqqe_with_kernel, DvdiqqeBatchBuilder, DvdiqqeBatchOutput,
    DvdiqqeBatchRange, DvdiqqeBuilder, DvdiqqeInput, DvdiqqeOutput, DvdiqqeParams, DvdiqqeStream,
};
pub mod dx;
pub mod efi;
pub mod ehlers_adaptive_cyber_cycle;
pub mod ehlers_simple_cycle_indicator;
pub mod ehlers_smoothed_adaptive_momentum;
pub mod ehlers_adaptive_cg;
pub mod emd;
pub mod emv;
pub mod er;
pub mod eri;
pub mod ewma_volatility;
pub mod exponential_trend;
pub mod fisher;
pub mod forward_backward_exponential_oscillator;
pub mod fosc;
pub mod fvg_trailing_stop;
pub mod l1_ehlers_phasor;
pub mod trend_flow_trail;
pub use fvg_trailing_stop::{
    fvg_trailing_stop, FvgTrailingStopInput, FvgTrailingStopOutput, FvgTrailingStopParams,
};
pub mod gatorosc;
pub mod geometric_bias_oscillator;
pub mod halftrend;
pub mod vdubus_divergence_wave_pattern_generator;
pub use halftrend::{halftrend, HalfTrendInput, HalfTrendOutput, HalfTrendParams};
pub mod ichimoku_oscillator;
pub mod ehlers_fm_demodulator;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use adaptive_macd::adaptive_macd_into;
pub use adaptive_macd::{
    adaptive_macd, adaptive_macd_batch_par_slice, adaptive_macd_batch_slice,
    adaptive_macd_batch_with_kernel, adaptive_macd_into_slice, adaptive_macd_with_kernel,
    AdaptiveMacdBatchBuilder, AdaptiveMacdBatchOutput, AdaptiveMacdBatchRange, AdaptiveMacdBuilder,
    AdaptiveMacdData, AdaptiveMacdError, AdaptiveMacdInput, AdaptiveMacdOutput, AdaptiveMacdParams,
    AdaptiveMacdStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use adaptive_macd::{
    adaptive_macd_alloc, adaptive_macd_batch_into,
    adaptive_macd_batch_unified_js as adaptive_macd_batch, adaptive_macd_free, adaptive_macd_into,
    adaptive_macd_js,
};
#[cfg(feature = "python")]
pub use adaptive_macd::{adaptive_macd_batch_py, adaptive_macd_py, AdaptiveMacdStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use geometric_bias_oscillator::geometric_bias_oscillator_into;
pub use geometric_bias_oscillator::{
    expand_grid_geometric_bias_oscillator, geometric_bias_oscillator,
    geometric_bias_oscillator_batch_par_slice, geometric_bias_oscillator_batch_slice,
    geometric_bias_oscillator_batch_with_kernel, geometric_bias_oscillator_into_slice,
    geometric_bias_oscillator_with_kernel, GeometricBiasOscillatorBatchBuilder,
    GeometricBiasOscillatorBatchOutput, GeometricBiasOscillatorBatchRange,
    GeometricBiasOscillatorBuilder, GeometricBiasOscillatorData, GeometricBiasOscillatorError,
    GeometricBiasOscillatorInput, GeometricBiasOscillatorOutput, GeometricBiasOscillatorParams,
    GeometricBiasOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use geometric_bias_oscillator::{
    geometric_bias_oscillator_alloc, geometric_bias_oscillator_batch_into,
    geometric_bias_oscillator_batch_unified_js as geometric_bias_oscillator_batch,
    geometric_bias_oscillator_free, geometric_bias_oscillator_into, geometric_bias_oscillator_js,
};
#[cfg(feature = "python")]
pub use geometric_bias_oscillator::{
    geometric_bias_oscillator_batch_py, geometric_bias_oscillator_py,
    GeometricBiasOscillatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use standardized_psar_oscillator::standardized_psar_oscillator_into;
pub use standardized_psar_oscillator::{
    expand_grid_standardized_psar_oscillator, standardized_psar_oscillator,
    standardized_psar_oscillator_batch_par_slice, standardized_psar_oscillator_batch_slice,
    standardized_psar_oscillator_batch_with_kernel, standardized_psar_oscillator_into_slice,
    standardized_psar_oscillator_with_kernel, StandardizedPsarOscillatorBatchBuilder,
    StandardizedPsarOscillatorBatchOutput, StandardizedPsarOscillatorBatchRange,
    StandardizedPsarOscillatorBuilder, StandardizedPsarOscillatorData,
    StandardizedPsarOscillatorError, StandardizedPsarOscillatorInput,
    StandardizedPsarOscillatorOutput, StandardizedPsarOscillatorParams,
    StandardizedPsarOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use standardized_psar_oscillator::{
    standardized_psar_oscillator_alloc, standardized_psar_oscillator_batch_into,
    standardized_psar_oscillator_batch_unified_js as standardized_psar_oscillator_batch,
    standardized_psar_oscillator_free, standardized_psar_oscillator_into,
    standardized_psar_oscillator_js,
};
#[cfg(feature = "python")]
pub use standardized_psar_oscillator::{
    standardized_psar_oscillator_batch_py, standardized_psar_oscillator_py,
    StandardizedPsarOscillatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use statistical_trailing_stop::statistical_trailing_stop_into;
pub use statistical_trailing_stop::{
    statistical_trailing_stop, statistical_trailing_stop_batch_par_slice,
    statistical_trailing_stop_batch_slice, statistical_trailing_stop_batch_with_kernel,
    statistical_trailing_stop_into_slice, statistical_trailing_stop_with_kernel,
    StatisticalTrailingStopBatchBuilder, StatisticalTrailingStopBatchOutput,
    StatisticalTrailingStopBatchRange, StatisticalTrailingStopBuilder, StatisticalTrailingStopData,
    StatisticalTrailingStopError, StatisticalTrailingStopInput, StatisticalTrailingStopOutput,
    StatisticalTrailingStopParams, StatisticalTrailingStopStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use statistical_trailing_stop::{
    statistical_trailing_stop_alloc, statistical_trailing_stop_batch_into,
    statistical_trailing_stop_batch_unified_js as statistical_trailing_stop_batch,
    statistical_trailing_stop_free, statistical_trailing_stop_into, statistical_trailing_stop_js,
};
#[cfg(feature = "python")]
pub use statistical_trailing_stop::{
    statistical_trailing_stop_batch_py, statistical_trailing_stop_py,
    StatisticalTrailingStopStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use supertrend_recovery::supertrend_recovery_into;
pub use supertrend_recovery::{
    supertrend_recovery, supertrend_recovery_batch_par_slice, supertrend_recovery_batch_slice,
    supertrend_recovery_batch_with_kernel, supertrend_recovery_into_slice,
    supertrend_recovery_with_kernel, SuperTrendRecoveryBatchBuilder, SuperTrendRecoveryBatchOutput,
    SuperTrendRecoveryBatchRange, SuperTrendRecoveryBuilder, SuperTrendRecoveryData,
    SuperTrendRecoveryError, SuperTrendRecoveryInput, SuperTrendRecoveryOutput,
    SuperTrendRecoveryParams, SuperTrendRecoveryStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use supertrend_recovery::{
    supertrend_recovery_alloc, supertrend_recovery_batch_into,
    supertrend_recovery_batch_unified_js as supertrend_recovery_batch, supertrend_recovery_free,
    supertrend_recovery_into, supertrend_recovery_js,
};
#[cfg(feature = "python")]
pub use supertrend_recovery::{
    supertrend_recovery_batch_py, supertrend_recovery_py, SuperTrendRecoveryStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use vdubus_divergence_wave_pattern_generator::vdubus_divergence_wave_pattern_generator_into;
pub use vdubus_divergence_wave_pattern_generator::{
    expand_grid_vdubus_divergence_wave_pattern_generator, vdubus_divergence_wave_pattern_generator,
    vdubus_divergence_wave_pattern_generator_batch_par_slice,
    vdubus_divergence_wave_pattern_generator_batch_slice,
    vdubus_divergence_wave_pattern_generator_batch_with_kernel,
    vdubus_divergence_wave_pattern_generator_into_slice,
    vdubus_divergence_wave_pattern_generator_with_kernel,
    VdubusDivergenceWavePatternGeneratorBatchBuilder,
    VdubusDivergenceWavePatternGeneratorBatchOutput,
    VdubusDivergenceWavePatternGeneratorBatchRange, VdubusDivergenceWavePatternGeneratorBuilder,
    VdubusDivergenceWavePatternGeneratorData, VdubusDivergenceWavePatternGeneratorError,
    VdubusDivergenceWavePatternGeneratorInput, VdubusDivergenceWavePatternGeneratorOutput,
    VdubusDivergenceWavePatternGeneratorParams, VdubusDivergenceWavePatternGeneratorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use vdubus_divergence_wave_pattern_generator::{
    vdubus_divergence_wave_pattern_generator_alloc,
    vdubus_divergence_wave_pattern_generator_batch_into,
    vdubus_divergence_wave_pattern_generator_batch_unified_js as vdubus_divergence_wave_pattern_generator_batch,
    vdubus_divergence_wave_pattern_generator_into, vdubus_divergence_wave_pattern_generator_js,
};
#[cfg(feature = "python")]
pub use vdubus_divergence_wave_pattern_generator::{
    vdubus_divergence_wave_pattern_generator_batch_py, vdubus_divergence_wave_pattern_generator_py,
    VdubusDivergenceWavePatternGeneratorStreamPy,
};

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use adaptive_momentum_oscillator::adaptive_momentum_oscillator_into;
pub use adaptive_momentum_oscillator::{
    adaptive_momentum_oscillator, adaptive_momentum_oscillator_batch_par_slice,
    adaptive_momentum_oscillator_batch_slice, adaptive_momentum_oscillator_batch_with_kernel,
    adaptive_momentum_oscillator_into_slice, adaptive_momentum_oscillator_with_kernel,
    expand_grid_adaptive_momentum_oscillator, AdaptiveMomentumOscillatorBatchBuilder,
    AdaptiveMomentumOscillatorBatchOutput, AdaptiveMomentumOscillatorBatchRange,
    AdaptiveMomentumOscillatorBuilder, AdaptiveMomentumOscillatorData,
    AdaptiveMomentumOscillatorError, AdaptiveMomentumOscillatorInput,
    AdaptiveMomentumOscillatorOutput, AdaptiveMomentumOscillatorParams,
    AdaptiveMomentumOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use adaptive_momentum_oscillator::{
    adaptive_momentum_oscillator_alloc, adaptive_momentum_oscillator_batch_into,
    adaptive_momentum_oscillator_batch_unified_js as adaptive_momentum_oscillator_batch,
    adaptive_momentum_oscillator_free,
    adaptive_momentum_oscillator_into_js as adaptive_momentum_oscillator_into,
    adaptive_momentum_oscillator_js, AdaptiveMomentumOscillatorStreamWasm,
};
#[cfg(feature = "python")]
pub use adaptive_momentum_oscillator::{
    adaptive_momentum_oscillator_batch_py, adaptive_momentum_oscillator_py,
    AdaptiveMomentumOscillatorStreamPy,
};

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use ehlers_adaptive_cg::ehlers_adaptive_cg_into;
pub use ehlers_adaptive_cg::{
    ehlers_adaptive_cg, ehlers_adaptive_cg_batch_par_slice, ehlers_adaptive_cg_batch_slice,
    ehlers_adaptive_cg_batch_with_kernel, ehlers_adaptive_cg_into_slice,
    EhlersAdaptiveCgBatchBuilder, EhlersAdaptiveCgBatchOutput, EhlersAdaptiveCgBatchRange,
    EhlersAdaptiveCgBuilder, EhlersAdaptiveCgError, EhlersAdaptiveCgInput, EhlersAdaptiveCgOutput,
    EhlersAdaptiveCgParams, EhlersAdaptiveCgStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use ehlers_adaptive_cg::{
    ehlers_adaptive_cg_alloc, ehlers_adaptive_cg_batch_unified_js as ehlers_adaptive_cg_batch,
    ehlers_adaptive_cg_free, ehlers_adaptive_cg_into, ehlers_adaptive_cg_js,
};
#[cfg(feature = "python")]
pub use ehlers_adaptive_cg::{
    ehlers_adaptive_cg_batch_py, ehlers_adaptive_cg_py, EhlersAdaptiveCgStreamPy,
};
pub mod ift_rsi;
pub mod kaufmanstop;
pub mod kdj;
pub mod keltner;
pub mod kst;
pub mod kurtosis;
pub mod kvo;
pub mod l2_ehlers_signal_to_noise;
pub mod linear_correlation_oscillator;
pub mod linearreg_angle;
pub mod linearreg_intercept;
pub mod linearreg_slope;
pub mod lpc;
pub use l2_ehlers_signal_to_noise::expand_grid as l2_ehlers_signal_to_noise_expand_grid;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use l2_ehlers_signal_to_noise::l2_ehlers_signal_to_noise_into;
pub use l2_ehlers_signal_to_noise::{
    l2_ehlers_signal_to_noise, l2_ehlers_signal_to_noise_batch_into_slice,
    l2_ehlers_signal_to_noise_batch_par_slice, l2_ehlers_signal_to_noise_batch_slice,
    l2_ehlers_signal_to_noise_batch_with_kernel, l2_ehlers_signal_to_noise_into_slice,
    l2_ehlers_signal_to_noise_with_kernel, L2EhlersSignalToNoiseBatchBuilder,
    L2EhlersSignalToNoiseBatchOutput, L2EhlersSignalToNoiseBatchRange,
    L2EhlersSignalToNoiseBuilder, L2EhlersSignalToNoiseData, L2EhlersSignalToNoiseError,
    L2EhlersSignalToNoiseInput, L2EhlersSignalToNoiseOutput, L2EhlersSignalToNoiseParams,
    L2EhlersSignalToNoiseStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use l2_ehlers_signal_to_noise::{
    l2_ehlers_signal_to_noise_alloc, l2_ehlers_signal_to_noise_batch_into,
    l2_ehlers_signal_to_noise_batch_js, l2_ehlers_signal_to_noise_free,
    l2_ehlers_signal_to_noise_into_wasm as l2_ehlers_signal_to_noise_into,
    l2_ehlers_signal_to_noise_js,
};
#[cfg(feature = "python")]
pub use l2_ehlers_signal_to_noise::{
    l2_ehlers_signal_to_noise_batch_py, l2_ehlers_signal_to_noise_py,
    register_l2_ehlers_signal_to_noise_module, L2EhlersSignalToNoiseStreamPy,
};
pub mod polynomial_regression_extrapolation;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use ehlers_fm_demodulator::ehlers_fm_demodulator_into;
pub use ehlers_fm_demodulator::{
    ehlers_fm_demodulator, ehlers_fm_demodulator_batch_par_slice,
    ehlers_fm_demodulator_batch_slice, ehlers_fm_demodulator_batch_with_kernel,
    ehlers_fm_demodulator_into_slice, ehlers_fm_demodulator_with_kernel,
    EhlersFmDemodulatorBatchBuilder, EhlersFmDemodulatorBatchOutput, EhlersFmDemodulatorBatchRange,
    EhlersFmDemodulatorBuilder, EhlersFmDemodulatorError, EhlersFmDemodulatorInput,
    EhlersFmDemodulatorOutput, EhlersFmDemodulatorParams, EhlersFmDemodulatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use ehlers_fm_demodulator::{
    ehlers_fm_demodulator_alloc,
    ehlers_fm_demodulator_batch_unified_js as ehlers_fm_demodulator_batch,
    ehlers_fm_demodulator_free, ehlers_fm_demodulator_into, ehlers_fm_demodulator_js,
};
#[cfg(feature = "python")]
pub use ehlers_fm_demodulator::{
    ehlers_fm_demodulator_batch_py, ehlers_fm_demodulator_py, EhlersFmDemodulatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use linear_correlation_oscillator::linear_correlation_oscillator_into;
pub use linear_correlation_oscillator::{
    linear_correlation_oscillator, linear_correlation_oscillator_batch_par_slice,
    linear_correlation_oscillator_batch_slice, linear_correlation_oscillator_batch_with_kernel,
    linear_correlation_oscillator_into_slice, linear_correlation_oscillator_with_kernel,
    LinearCorrelationOscillatorBatchBuilder, LinearCorrelationOscillatorBatchOutput,
    LinearCorrelationOscillatorBatchRange, LinearCorrelationOscillatorBuilder,
    LinearCorrelationOscillatorError, LinearCorrelationOscillatorInput,
    LinearCorrelationOscillatorOutput, LinearCorrelationOscillatorParams,
    LinearCorrelationOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use linear_correlation_oscillator::{
    linear_correlation_oscillator_alloc, linear_correlation_oscillator_batch,
    linear_correlation_oscillator_free, linear_correlation_oscillator_into,
    linear_correlation_oscillator_js,
};
#[cfg(feature = "python")]
pub use linear_correlation_oscillator::{
    linear_correlation_oscillator_batch_py, linear_correlation_oscillator_py,
    LinearCorrelationOscillatorStreamPy,
};
pub use lpc::{lpc, LpcInput, LpcOutput, LpcParams};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use polynomial_regression_extrapolation::polynomial_regression_extrapolation_into;
pub use polynomial_regression_extrapolation::{
    polynomial_regression_extrapolation, polynomial_regression_extrapolation_batch_par_slice,
    polynomial_regression_extrapolation_batch_slice,
    polynomial_regression_extrapolation_batch_with_kernel,
    polynomial_regression_extrapolation_into_slice,
    polynomial_regression_extrapolation_with_kernel, PolynomialRegressionExtrapolationBatchBuilder,
    PolynomialRegressionExtrapolationBatchOutput, PolynomialRegressionExtrapolationBatchRange,
    PolynomialRegressionExtrapolationBuilder, PolynomialRegressionExtrapolationError,
    PolynomialRegressionExtrapolationInput, PolynomialRegressionExtrapolationOutput,
    PolynomialRegressionExtrapolationParams, PolynomialRegressionExtrapolationStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use polynomial_regression_extrapolation::{
    polynomial_regression_extrapolation_alloc, polynomial_regression_extrapolation_batch_into,
    polynomial_regression_extrapolation_batch_unified_js as polynomial_regression_extrapolation_batch,
    polynomial_regression_extrapolation_free, polynomial_regression_extrapolation_into,
    polynomial_regression_extrapolation_js,
};
#[cfg(feature = "python")]
pub use polynomial_regression_extrapolation::{
    polynomial_regression_extrapolation_batch_py, polynomial_regression_extrapolation_py,
    PolynomialRegressionExtrapolationStreamPy,
};
pub mod lrsi;
pub mod mab;
pub mod macd;
pub mod macz;
pub use macz::{macz, MaczInput, MaczOutput, MaczParams};
pub mod marketefi;
pub mod mass;
pub mod mean_ad;
pub mod medium_ad;
pub mod medprice;
pub mod mesa_stochastic_multi_length;
pub mod mfi;
pub mod midpoint;
pub mod midprice;
pub mod minmax;
pub use minmax::{minmax, MinmaxInput, MinmaxOutput, MinmaxParams};
pub mod mod_god_mode;
pub mod mom;
pub mod moving_averages;
pub use moving_averages::ehlers_kama::{
    ehlers_kama, EhlersKamaInput, EhlersKamaOutput, EhlersKamaParams,
};
pub mod msw;
pub mod nadaraya_watson_envelope;
pub mod natr;
pub mod net_myrsi;
pub mod normalized_volume_true_range;
pub use net_myrsi::{net_myrsi, NetMyrsiInput, NetMyrsiOutput, NetMyrsiParams};
pub mod nvi;
pub mod obv;
pub mod on_balance_volume_oscillator;
pub mod ott;
pub use ott::{
    ott, ott_batch_par_slice, ott_batch_slice, ott_batch_with_kernel, OttInput, OttOutput,
    OttParams,
};
pub mod otto;
pub use otto::{
    otto, OttoBatchBuilder, OttoBatchOutput, OttoBatchRange, OttoBuilder, OttoData, OttoError,
    OttoInput, OttoOutput, OttoParams, OttoStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use otto::{otto_alloc, otto_batch_unified_js, otto_free, otto_into, otto_js};
#[cfg(feature = "python")]
pub use otto::{otto_batch_py, otto_py, OttoStreamPy};
pub mod ehlers_detrending_filter;
pub mod historical_volatility_percentile;
pub mod hypertrend;
pub mod ict_propulsion_block;
pub mod impulse_macd;
pub mod insync_index;
pub mod keltner_channel_width_oscillator;
pub mod leavitt_convolution_acceleration;
pub mod linear_regression_intensity;
pub mod market_meanness_index;
pub mod momentum_ratio_oscillator;
pub mod parkinson_volatility;
pub mod pattern_recognition;
pub mod percentile_nearest_rank;
pub mod pfe;
pub mod pretty_good_oscillator;
pub mod price_density_market_noise;
pub mod psychological_line;
pub mod rank_correlation_index;
pub mod smoothed_gaussian_trend_filter;
pub mod trend_continuation_factor;
pub mod trend_direction_force_index;
pub mod trend_follower;
pub use percentile_nearest_rank::{
    percentile_nearest_rank, percentile_nearest_rank_into_slice,
    percentile_nearest_rank_with_kernel, pnr_batch_par_slice, pnr_batch_slice,
    pnr_batch_with_kernel, PercentileNearestRankBatchBuilder, PercentileNearestRankBatchOutput,
    PercentileNearestRankBatchRange, PercentileNearestRankBuilder, PercentileNearestRankData,
    PercentileNearestRankError, PercentileNearestRankInput, PercentileNearestRankOutput,
    PercentileNearestRankParams, PercentileNearestRankStream,
};
pub mod pivot;
pub mod pma;
pub mod ppo;
pub mod price_moving_average_ratio_percentile;
pub use ppo::{ppo, PpoInput, PpoOutput, PpoParams};
pub mod prb;
pub use prb::{
    prb, prb_batch_par_slice, prb_batch_slice, prb_batch_with_kernel, prb_with_kernel,
    PrbBatchBuilder, PrbBatchOutput, PrbBatchRange, PrbBuilder, PrbInput, PrbOutput, PrbParams,
    PrbStream,
};
pub mod pvi;
pub mod qqe;
pub mod qqe_weighted_oscillator;
pub mod qstick;
pub mod random_walk_index;
pub mod range_breakout_signals;
pub mod range_filter;
pub mod market_structure_confluence;
pub mod range_filtered_trend_signals;
pub mod range_oscillator;
pub mod registry;
pub mod volume_weighted_relative_strength_index;
pub use range_filter::{
    range_filter, range_filter_batch_par_slice, range_filter_batch_slice, range_filter_into_slice,
    range_filter_with_kernel, RangeFilterBatchBuilder, RangeFilterBatchOutput,
    RangeFilterBatchRange, RangeFilterBuilder, RangeFilterData, RangeFilterError, RangeFilterInput,
    RangeFilterOutput, RangeFilterParams, RangeFilterStream,
};
pub use market_structure_confluence::{
    market_structure_confluence, market_structure_confluence_batch_with_kernel,
    market_structure_confluence_into, market_structure_confluence_into_slices,
    market_structure_confluence_with_kernel, MarketStructureConfluenceBatchBuilder,
    MarketStructureConfluenceBatchOutput, MarketStructureConfluenceBatchRange,
    MarketStructureConfluenceBosConfirmation, MarketStructureConfluenceBuilder,
    MarketStructureConfluenceData, MarketStructureConfluenceError,
    MarketStructureConfluenceInput, MarketStructureConfluenceOutput,
    MarketStructureConfluenceParams, MarketStructureConfluenceStream,
};
pub use range_filtered_trend_signals::{
    range_filtered_trend_signals, range_filtered_trend_signals_batch_with_kernel,
    range_filtered_trend_signals_into, range_filtered_trend_signals_into_slices,
    range_filtered_trend_signals_with_kernel, RangeFilteredTrendSignalsBatchBuilder,
    RangeFilteredTrendSignalsBatchOutput, RangeFilteredTrendSignalsBatchRange,
    RangeFilteredTrendSignalsBuilder, RangeFilteredTrendSignalsData,
    RangeFilteredTrendSignalsError, RangeFilteredTrendSignalsInput,
    RangeFilteredTrendSignalsOutput, RangeFilteredTrendSignalsParams,
    RangeFilteredTrendSignalsStream,
};
pub mod roc;
pub use roc::{
    roc, RocBatchBuilder, RocBatchOutput, RocBatchRange, RocBuilder, RocError, RocInput, RocOutput,
    RocParams, RocStream,
};
pub mod reverse_rsi;
pub mod rocp;
pub mod rocr;
pub use forward_backward_exponential_oscillator::{
    forward_backward_exponential_oscillator,
    forward_backward_exponential_oscillator_batch_with_kernel,
    forward_backward_exponential_oscillator_into,
    forward_backward_exponential_oscillator_into_slices,
    forward_backward_exponential_oscillator_with_kernel,
    ForwardBackwardExponentialOscillatorBatchBuilder,
    ForwardBackwardExponentialOscillatorBatchOutput,
    ForwardBackwardExponentialOscillatorBatchRange, ForwardBackwardExponentialOscillatorBuilder,
    ForwardBackwardExponentialOscillatorData, ForwardBackwardExponentialOscillatorError,
    ForwardBackwardExponentialOscillatorInput, ForwardBackwardExponentialOscillatorOutput,
    ForwardBackwardExponentialOscillatorParams, ForwardBackwardExponentialOscillatorStream,
};
pub use qqe_weighted_oscillator::{
    qqe_weighted_oscillator, qqe_weighted_oscillator_batch_with_kernel,
    qqe_weighted_oscillator_into, qqe_weighted_oscillator_into_slices,
    qqe_weighted_oscillator_with_kernel, QqeWeightedOscillatorBatchBuilder,
    QqeWeightedOscillatorBatchOutput, QqeWeightedOscillatorBatchRange,
    QqeWeightedOscillatorBuilder, QqeWeightedOscillatorData, QqeWeightedOscillatorError,
    QqeWeightedOscillatorInput, QqeWeightedOscillatorOutput, QqeWeightedOscillatorParams,
    QqeWeightedOscillatorStream,
};
pub use range_oscillator::{
    range_oscillator, range_oscillator_batch_with_kernel, range_oscillator_into,
    range_oscillator_into_slices, range_oscillator_with_kernel, RangeOscillatorBatchBuilder,
    RangeOscillatorBatchOutput, RangeOscillatorBatchRange, RangeOscillatorBuilder,
    RangeOscillatorData, RangeOscillatorError, RangeOscillatorInput, RangeOscillatorOutput,
    RangeOscillatorParams, RangeOscillatorStream,
};
pub use reverse_rsi::{reverse_rsi, ReverseRsiInput, ReverseRsiOutput, ReverseRsiParams};
pub use volume_weighted_relative_strength_index::{
    volume_weighted_relative_strength_index,
    volume_weighted_relative_strength_index_batch_with_kernel,
    volume_weighted_relative_strength_index_into,
    volume_weighted_relative_strength_index_into_slices,
    volume_weighted_relative_strength_index_with_kernel,
    VolumeWeightedRelativeStrengthIndexBatchBuilder,
    VolumeWeightedRelativeStrengthIndexBatchOutput, VolumeWeightedRelativeStrengthIndexBatchRange,
    VolumeWeightedRelativeStrengthIndexBuilder, VolumeWeightedRelativeStrengthIndexData,
    VolumeWeightedRelativeStrengthIndexError, VolumeWeightedRelativeStrengthIndexInput,
    VolumeWeightedRelativeStrengthIndexOutput, VolumeWeightedRelativeStrengthIndexParams,
    VolumeWeightedRelativeStrengthIndexStream,
};
pub mod moving_average_cross_probability;
pub mod regression_slope_oscillator;
pub mod relative_strength_index_wave_indicator;
pub mod rsi;
pub mod rsmk;
pub mod rsx;
pub mod volatility_ratio_adaptive_rsx;
pub use rsx::{
    rsx, RsxBatchOutput, RsxBatchRange, RsxBuilder, RsxInput, RsxOutput, RsxParams, RsxStream,
};
pub mod adaptive_schaff_trend_cycle;
pub mod rvi;
pub mod safezonestop;
pub mod sar;
pub mod spearman_correlation;
pub mod squeeze_momentum;
pub mod srsi;
pub mod stc;
pub mod stddev;
pub use stddev::{stddev, StdDevInput, StdDevOutput, StdDevParams};
pub mod smooth_theil_sen;
pub mod stoch;
pub mod stochastic_adaptive_d;
pub mod stochastic_connors_rsi;
pub mod stochf;
pub mod supertrend;
pub mod supertrend_oscillator;
pub mod trend_trigger_factor;
pub mod trix;
pub mod tsf;
pub mod tsi;
pub mod ttm_squeeze;
pub mod ttm_trend;
pub mod twiggs_money_flow;
pub mod ui;
pub mod ultosc;
pub mod utility_functions;
pub mod var;
pub mod velocity;
pub mod vi;
pub mod vidya;
pub mod vlma;
pub mod volume_weighted_stochastic_rsi;
pub mod volatility_quality_index;
pub mod volume_zone_oscillator;
pub mod vosc;
pub mod voss;
pub mod vpci;
pub mod vpt;
pub mod vwap_deviation_oscillator;
pub use vpt::{vpt, VptInput, VptOutput, VptParams};
pub mod vwmacd;
pub mod wad;
pub mod wavetrend;
pub mod wclprice;
pub mod willr;
pub mod wto;
pub use wto::{
    wto, wto_batch_candles, wto_batch_slice, wto_into_slices, wto_with_kernel, WtoBatchBuilder,
    WtoBatchOutput, WtoBatchRange, WtoBuilder, WtoData, WtoError, WtoInput, WtoOutput, WtoParams,
    WtoStream,
};
pub mod yang_zhang_volatility;
pub mod zscore;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use on_balance_volume_oscillator::on_balance_volume_oscillator_into;
pub use on_balance_volume_oscillator::{
    expand_grid_on_balance_volume_oscillator, on_balance_volume_oscillator,
    on_balance_volume_oscillator_batch_par_slice, on_balance_volume_oscillator_batch_slice,
    on_balance_volume_oscillator_batch_with_kernel, on_balance_volume_oscillator_into_slice,
    on_balance_volume_oscillator_with_kernel, OnBalanceVolumeOscillatorBatchBuilder,
    OnBalanceVolumeOscillatorBatchOutput, OnBalanceVolumeOscillatorBatchRange,
    OnBalanceVolumeOscillatorBuilder, OnBalanceVolumeOscillatorData,
    OnBalanceVolumeOscillatorError, OnBalanceVolumeOscillatorInput,
    OnBalanceVolumeOscillatorOutput, OnBalanceVolumeOscillatorParams,
    OnBalanceVolumeOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use on_balance_volume_oscillator::{
    on_balance_volume_oscillator_alloc, on_balance_volume_oscillator_batch_into,
    on_balance_volume_oscillator_batch_js, on_balance_volume_oscillator_free,
    on_balance_volume_oscillator_into, on_balance_volume_oscillator_into_host,
    on_balance_volume_oscillator_js,
};
#[cfg(feature = "python")]
pub use on_balance_volume_oscillator::{
    on_balance_volume_oscillator_batch_py, on_balance_volume_oscillator_py,
    OnBalanceVolumeOscillatorStreamPy,
};
pub use vpci::{
    vpci, VpciBatchBuilder, VpciBatchOutput, VpciBatchRange, VpciData, VpciError, VpciInput,
    VpciOutput, VpciParams, VpciStream,
};
#[cfg(feature = "python")]
pub use vpci::{vpci_batch_py, vpci_py, VpciStreamPy};

#[cfg(feature = "python")]
pub use avsl::{avsl_batch_py, avsl_py, AvslStreamPy};

#[cfg(feature = "python")]
pub use range_filter::{range_filter_batch_py, range_filter_py, RangeFilterStreamPy};

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use adaptive_schaff_trend_cycle::adaptive_schaff_trend_cycle_into;
pub use adaptive_schaff_trend_cycle::{
    adaptive_schaff_trend_cycle, adaptive_schaff_trend_cycle_batch_par_slice,
    adaptive_schaff_trend_cycle_batch_slice, adaptive_schaff_trend_cycle_batch_with_kernel,
    adaptive_schaff_trend_cycle_into_slice, adaptive_schaff_trend_cycle_with_kernel,
    expand_grid_adaptive_schaff_trend_cycle, AdaptiveSchaffTrendCycleBatchBuilder,
    AdaptiveSchaffTrendCycleBatchOutput, AdaptiveSchaffTrendCycleBatchRange,
    AdaptiveSchaffTrendCycleBuilder, AdaptiveSchaffTrendCycleData, AdaptiveSchaffTrendCycleError,
    AdaptiveSchaffTrendCycleInput, AdaptiveSchaffTrendCycleOutput, AdaptiveSchaffTrendCycleParams,
    AdaptiveSchaffTrendCycleStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use adaptive_schaff_trend_cycle::{
    adaptive_schaff_trend_cycle_alloc, adaptive_schaff_trend_cycle_batch_into,
    adaptive_schaff_trend_cycle_batch_js, adaptive_schaff_trend_cycle_free,
    adaptive_schaff_trend_cycle_into, adaptive_schaff_trend_cycle_into_host,
    adaptive_schaff_trend_cycle_js,
};
#[cfg(feature = "python")]
pub use adaptive_schaff_trend_cycle::{
    adaptive_schaff_trend_cycle_batch_py, adaptive_schaff_trend_cycle_py,
    AdaptiveSchaffTrendCycleStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use accumulation_swing_index::accumulation_swing_index_into;
pub use accumulation_swing_index::{
    accumulation_swing_index, accumulation_swing_index_batch_into_slice,
    accumulation_swing_index_batch_par_slice, accumulation_swing_index_batch_slice,
    accumulation_swing_index_batch_with_kernel, accumulation_swing_index_into_slice,
    accumulation_swing_index_with_kernel, AccumulationSwingIndexBatchBuilder,
    AccumulationSwingIndexBatchOutput, AccumulationSwingIndexBatchRange,
    AccumulationSwingIndexBuilder, AccumulationSwingIndexData, AccumulationSwingIndexError,
    AccumulationSwingIndexInput, AccumulationSwingIndexOutput, AccumulationSwingIndexParams,
    AccumulationSwingIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use accumulation_swing_index::{
    accumulation_swing_index_alloc, accumulation_swing_index_batch_into,
    accumulation_swing_index_batch_js, accumulation_swing_index_free,
    accumulation_swing_index_into, accumulation_swing_index_js,
};
#[cfg(feature = "python")]
pub use accumulation_swing_index::{
    accumulation_swing_index_batch_py, accumulation_swing_index_py,
    register_accumulation_swing_index_module, AccumulationSwingIndexStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use andean_oscillator::andean_oscillator_into;
pub use andean_oscillator::expand_grid as andean_oscillator_expand_grid;
pub use andean_oscillator::{
    andean_oscillator, andean_oscillator_batch_into_slice, andean_oscillator_batch_par_slice,
    andean_oscillator_batch_slice, andean_oscillator_batch_with_kernel,
    andean_oscillator_into_slice, andean_oscillator_with_kernel, AndeanOscillatorBatchBuilder,
    AndeanOscillatorBatchOutput, AndeanOscillatorBatchRange, AndeanOscillatorBuilder,
    AndeanOscillatorData, AndeanOscillatorError, AndeanOscillatorInput, AndeanOscillatorOutput,
    AndeanOscillatorParams, AndeanOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use andean_oscillator::{
    andean_oscillator_alloc, andean_oscillator_batch_into, andean_oscillator_batch_js,
    andean_oscillator_free, andean_oscillator_into, andean_oscillator_js,
};
#[cfg(feature = "python")]
pub use andean_oscillator::{
    andean_oscillator_batch_py, andean_oscillator_py, register_andean_oscillator_module,
    AndeanOscillatorStreamPy,
};
pub use apo::{apo, ApoInput, ApoOutput, ApoParams};
pub use bulls_v_bears::bulls_v_bears_expand_grid;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use bulls_v_bears::bulls_v_bears_into;
pub use bulls_v_bears::{
    bulls_v_bears, bulls_v_bears_batch_into_slice, bulls_v_bears_batch_par_slice,
    bulls_v_bears_batch_slice, bulls_v_bears_batch_with_kernel, bulls_v_bears_into_slice,
    bulls_v_bears_with_kernel, BullsVBearsBatchBuilder, BullsVBearsBatchOutput,
    BullsVBearsBatchRange, BullsVBearsBuilder, BullsVBearsCalculationMethod, BullsVBearsData,
    BullsVBearsError, BullsVBearsInput, BullsVBearsMaType, BullsVBearsOutput, BullsVBearsParams,
    BullsVBearsStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use bulls_v_bears::{
    bulls_v_bears_alloc, bulls_v_bears_batch_into, bulls_v_bears_batch_js, bulls_v_bears_free,
    bulls_v_bears_into, bulls_v_bears_js,
};
#[cfg(feature = "python")]
pub use bulls_v_bears::{
    bulls_v_bears_batch_py, bulls_v_bears_py, register_bulls_v_bears_module, BullsVBearsStreamPy,
};
pub use cci::{cci, CciInput, CciOutput, CciParams};
pub use cfo::{cfo, CfoInput, CfoOutput, CfoParams};
pub use coppock::{coppock, CoppockInput, CoppockOutput, CoppockParams};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use ehlers_detrending_filter::ehlers_detrending_filter_into;
pub use ehlers_detrending_filter::{
    ehlers_detrending_filter, ehlers_detrending_filter_batch_par_slice,
    ehlers_detrending_filter_batch_slice, ehlers_detrending_filter_batch_with_kernel,
    ehlers_detrending_filter_into_slice, ehlers_detrending_filter_with_kernel,
    expand_grid_ehlers_detrending_filter, EhlersDetrendingFilterBatchBuilder,
    EhlersDetrendingFilterBatchOutput, EhlersDetrendingFilterBatchRange,
    EhlersDetrendingFilterBuilder, EhlersDetrendingFilterData, EhlersDetrendingFilterError,
    EhlersDetrendingFilterInput, EhlersDetrendingFilterOutput, EhlersDetrendingFilterParams,
    EhlersDetrendingFilterStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use ehlers_detrending_filter::{
    ehlers_detrending_filter_alloc, ehlers_detrending_filter_batch_into,
    ehlers_detrending_filter_batch_js, ehlers_detrending_filter_free,
    ehlers_detrending_filter_into, ehlers_detrending_filter_into_host, ehlers_detrending_filter_js,
};
#[cfg(feature = "python")]
pub use ehlers_detrending_filter::{
    ehlers_detrending_filter_batch_py, ehlers_detrending_filter_py, EhlersDetrendingFilterStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use cycle_channel_oscillator::cycle_channel_oscillator_into;
pub use cycle_channel_oscillator::expand_grid as cycle_channel_oscillator_expand_grid;
pub use cycle_channel_oscillator::{
    cycle_channel_oscillator, cycle_channel_oscillator_batch_into_slice,
    cycle_channel_oscillator_batch_par_slice, cycle_channel_oscillator_batch_slice,
    cycle_channel_oscillator_batch_with_kernel, cycle_channel_oscillator_into_slice,
    cycle_channel_oscillator_with_kernel, CycleChannelOscillatorBatchBuilder,
    CycleChannelOscillatorBatchOutput, CycleChannelOscillatorBatchRange,
    CycleChannelOscillatorBuilder, CycleChannelOscillatorData, CycleChannelOscillatorError,
    CycleChannelOscillatorInput, CycleChannelOscillatorOutput, CycleChannelOscillatorParams,
    CycleChannelOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use cycle_channel_oscillator::{
    cycle_channel_oscillator_alloc, cycle_channel_oscillator_batch_into,
    cycle_channel_oscillator_batch_js, cycle_channel_oscillator_free,
    cycle_channel_oscillator_into, cycle_channel_oscillator_js,
};
#[cfg(feature = "python")]
pub use cycle_channel_oscillator::{
    cycle_channel_oscillator_batch_py, cycle_channel_oscillator_py,
    register_cycle_channel_oscillator_module, CycleChannelOscillatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use daily_factor::daily_factor_into;
pub use daily_factor::expand_grid as daily_factor_expand_grid;
pub use daily_factor::{
    daily_factor, daily_factor_batch_into_slice, daily_factor_batch_par_slice,
    daily_factor_batch_slice, daily_factor_batch_with_kernel, daily_factor_into_slice,
    daily_factor_with_kernel, DailyFactorBatchBuilder, DailyFactorBatchOutput,
    DailyFactorBatchRange, DailyFactorBuilder, DailyFactorData, DailyFactorError, DailyFactorInput,
    DailyFactorOutput, DailyFactorParams, DailyFactorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use daily_factor::{
    daily_factor_alloc, daily_factor_batch_into, daily_factor_batch_js, daily_factor_free,
    daily_factor_into, daily_factor_js,
};
#[cfg(feature = "python")]
pub use daily_factor::{
    daily_factor_batch_py, daily_factor_py, register_daily_factor_module, DailyFactorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use ehlers_adaptive_cyber_cycle::ehlers_adaptive_cyber_cycle_into;
pub use ehlers_adaptive_cyber_cycle::expand_grid as ehlers_adaptive_cyber_cycle_expand_grid;
pub use ehlers_adaptive_cyber_cycle::{
    ehlers_adaptive_cyber_cycle, ehlers_adaptive_cyber_cycle_batch_into_slice,
    ehlers_adaptive_cyber_cycle_batch_par_slice, ehlers_adaptive_cyber_cycle_batch_slice,
    ehlers_adaptive_cyber_cycle_batch_with_kernel, ehlers_adaptive_cyber_cycle_into_slice,
    ehlers_adaptive_cyber_cycle_with_kernel, EhlersAdaptiveCyberCycleBatchBuilder,
    EhlersAdaptiveCyberCycleBatchOutput, EhlersAdaptiveCyberCycleBatchRange,
    EhlersAdaptiveCyberCycleBuilder, EhlersAdaptiveCyberCycleData, EhlersAdaptiveCyberCycleError,
    EhlersAdaptiveCyberCycleInput, EhlersAdaptiveCyberCycleOutput, EhlersAdaptiveCyberCycleParams,
    EhlersAdaptiveCyberCycleStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use ehlers_adaptive_cyber_cycle::{
    ehlers_adaptive_cyber_cycle_alloc, ehlers_adaptive_cyber_cycle_batch_into,
    ehlers_adaptive_cyber_cycle_batch_js, ehlers_adaptive_cyber_cycle_free,
    ehlers_adaptive_cyber_cycle_into, ehlers_adaptive_cyber_cycle_js,
};
#[cfg(feature = "python")]
pub use ehlers_adaptive_cyber_cycle::{
    ehlers_adaptive_cyber_cycle_batch_py, ehlers_adaptive_cyber_cycle_py,
    register_ehlers_adaptive_cyber_cycle_module, EhlersAdaptiveCyberCycleStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use ehlers_simple_cycle_indicator::ehlers_simple_cycle_indicator_into;
pub use ehlers_simple_cycle_indicator::expand_grid as ehlers_simple_cycle_indicator_expand_grid;
pub use ehlers_simple_cycle_indicator::{
    ehlers_simple_cycle_indicator, ehlers_simple_cycle_indicator_batch_into_slice,
    ehlers_simple_cycle_indicator_batch_par_slice, ehlers_simple_cycle_indicator_batch_slice,
    ehlers_simple_cycle_indicator_batch_with_kernel, ehlers_simple_cycle_indicator_into_slice,
    ehlers_simple_cycle_indicator_with_kernel, EhlersSimpleCycleIndicatorBatchBuilder,
    EhlersSimpleCycleIndicatorBatchOutput, EhlersSimpleCycleIndicatorBatchRange,
    EhlersSimpleCycleIndicatorBuilder, EhlersSimpleCycleIndicatorData,
    EhlersSimpleCycleIndicatorError, EhlersSimpleCycleIndicatorInput,
    EhlersSimpleCycleIndicatorOutput, EhlersSimpleCycleIndicatorParams,
    EhlersSimpleCycleIndicatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use ehlers_simple_cycle_indicator::{
    ehlers_simple_cycle_indicator_alloc, ehlers_simple_cycle_indicator_batch_into,
    ehlers_simple_cycle_indicator_batch_js, ehlers_simple_cycle_indicator_free,
    ehlers_simple_cycle_indicator_into, ehlers_simple_cycle_indicator_js,
};
#[cfg(feature = "python")]
pub use ehlers_simple_cycle_indicator::{
    ehlers_simple_cycle_indicator_batch_py, ehlers_simple_cycle_indicator_py,
    register_ehlers_simple_cycle_indicator_module, EhlersSimpleCycleIndicatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use ehlers_smoothed_adaptive_momentum::ehlers_smoothed_adaptive_momentum_into;
pub use ehlers_smoothed_adaptive_momentum::expand_grid as ehlers_smoothed_adaptive_momentum_expand_grid;
pub use ehlers_smoothed_adaptive_momentum::{
    ehlers_smoothed_adaptive_momentum, ehlers_smoothed_adaptive_momentum_batch_into_slice,
    ehlers_smoothed_adaptive_momentum_batch_par_slice,
    ehlers_smoothed_adaptive_momentum_batch_slice,
    ehlers_smoothed_adaptive_momentum_batch_with_kernel,
    ehlers_smoothed_adaptive_momentum_into_slice, ehlers_smoothed_adaptive_momentum_with_kernel,
    EhlersSmoothedAdaptiveMomentumBatchBuilder, EhlersSmoothedAdaptiveMomentumBatchOutput,
    EhlersSmoothedAdaptiveMomentumBatchRange, EhlersSmoothedAdaptiveMomentumBuilder,
    EhlersSmoothedAdaptiveMomentumData, EhlersSmoothedAdaptiveMomentumError,
    EhlersSmoothedAdaptiveMomentumInput, EhlersSmoothedAdaptiveMomentumOutput,
    EhlersSmoothedAdaptiveMomentumParams, EhlersSmoothedAdaptiveMomentumStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use ehlers_smoothed_adaptive_momentum::{
    ehlers_smoothed_adaptive_momentum_alloc, ehlers_smoothed_adaptive_momentum_batch_into,
    ehlers_smoothed_adaptive_momentum_batch_js, ehlers_smoothed_adaptive_momentum_free,
    ehlers_smoothed_adaptive_momentum_into, ehlers_smoothed_adaptive_momentum_js,
};
#[cfg(feature = "python")]
pub use ehlers_smoothed_adaptive_momentum::{
    ehlers_smoothed_adaptive_momentum_batch_py, ehlers_smoothed_adaptive_momentum_py,
    register_ehlers_smoothed_adaptive_momentum_module, EhlersSmoothedAdaptiveMomentumStreamPy,
};
pub use er::{er, ErInput, ErOutput, ErParams};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use historical_volatility_percentile::historical_volatility_percentile_into;
pub use historical_volatility_percentile::{
    historical_volatility_percentile, historical_volatility_percentile_batch_par_slice,
    historical_volatility_percentile_batch_slice,
    historical_volatility_percentile_batch_with_kernel,
    historical_volatility_percentile_into_slice, historical_volatility_percentile_with_kernel,
    HistoricalVolatilityPercentileBatchBuilder, HistoricalVolatilityPercentileBatchOutput,
    HistoricalVolatilityPercentileBatchRange, HistoricalVolatilityPercentileBuilder,
    HistoricalVolatilityPercentileData, HistoricalVolatilityPercentileError,
    HistoricalVolatilityPercentileInput, HistoricalVolatilityPercentileOutput,
    HistoricalVolatilityPercentileParams, HistoricalVolatilityPercentileStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use historical_volatility_percentile::{
    historical_volatility_percentile_alloc, historical_volatility_percentile_batch_into,
    historical_volatility_percentile_batch_js, historical_volatility_percentile_free,
    historical_volatility_percentile_into, historical_volatility_percentile_into_host,
    historical_volatility_percentile_js,
};
#[cfg(feature = "python")]
pub use historical_volatility_percentile::{
    historical_volatility_percentile_batch_py, historical_volatility_percentile_py,
    HistoricalVolatilityPercentileStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use hypertrend::hypertrend_into;
pub use hypertrend::{
    expand_grid_hypertrend, hypertrend, hypertrend_batch_par_slice, hypertrend_batch_slice,
    hypertrend_batch_with_kernel, hypertrend_into_slice, hypertrend_with_kernel,
    HyperTrendBatchBuilder, HyperTrendBatchOutput, HyperTrendBatchRange, HyperTrendBuilder,
    HyperTrendData, HyperTrendError, HyperTrendInput, HyperTrendOutput, HyperTrendParams,
    HyperTrendStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use hypertrend::{
    hypertrend_alloc, hypertrend_batch_into, hypertrend_batch_js, hypertrend_free, hypertrend_into,
    hypertrend_into_host, hypertrend_js,
};
#[cfg(feature = "python")]
pub use hypertrend::{hypertrend_batch_py, hypertrend_py, HyperTrendStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use ict_propulsion_block::ict_propulsion_block_into;
pub use ict_propulsion_block::{
    expand_grid_ict_propulsion_block, ict_propulsion_block, ict_propulsion_block_batch_par_slice,
    ict_propulsion_block_batch_slice, ict_propulsion_block_batch_with_kernel,
    ict_propulsion_block_into_slice, ict_propulsion_block_with_kernel,
    IctPropulsionBlockBatchBuilder, IctPropulsionBlockBatchOutput, IctPropulsionBlockBatchRange,
    IctPropulsionBlockBuilder, IctPropulsionBlockData, IctPropulsionBlockError,
    IctPropulsionBlockInput, IctPropulsionBlockMitigationPrice, IctPropulsionBlockOutput,
    IctPropulsionBlockParams, IctPropulsionBlockStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use ict_propulsion_block::{
    ict_propulsion_block_alloc, ict_propulsion_block_batch_into, ict_propulsion_block_batch_js,
    ict_propulsion_block_free, ict_propulsion_block_into, ict_propulsion_block_into_host,
    ict_propulsion_block_js,
};
#[cfg(feature = "python")]
pub use ict_propulsion_block::{
    ict_propulsion_block_batch_py, ict_propulsion_block_py, IctPropulsionBlockStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use ewma_volatility::ewma_volatility_into;
pub use ewma_volatility::{
    ewma_volatility, ewma_volatility_batch_into_slice, ewma_volatility_batch_par_slice,
    ewma_volatility_batch_slice, ewma_volatility_batch_with_kernel, ewma_volatility_into_slice,
    ewma_volatility_with_kernel, EwmaVolatilityBatchBuilder, EwmaVolatilityBatchOutput,
    EwmaVolatilityBatchRange, EwmaVolatilityBuilder, EwmaVolatilityData, EwmaVolatilityError,
    EwmaVolatilityInput, EwmaVolatilityOutput, EwmaVolatilityParams, EwmaVolatilityStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use ewma_volatility::{
    ewma_volatility_alloc, ewma_volatility_batch_into, ewma_volatility_batch_js,
    ewma_volatility_free, ewma_volatility_into, ewma_volatility_js,
};
#[cfg(feature = "python")]
pub use ewma_volatility::{ewma_volatility_batch_py, ewma_volatility_py, EwmaVolatilityStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use ichimoku_oscillator::ichimoku_oscillator_into;
pub use ichimoku_oscillator::{
    ichimoku_oscillator, ichimoku_oscillator_batch_into_slice, ichimoku_oscillator_batch_par_slice,
    ichimoku_oscillator_batch_slice, ichimoku_oscillator_batch_with_kernel,
    ichimoku_oscillator_into_slice, ichimoku_oscillator_with_kernel,
    IchimokuOscillatorBatchBuilder, IchimokuOscillatorBatchOutput, IchimokuOscillatorBatchRange,
    IchimokuOscillatorBuilder, IchimokuOscillatorData, IchimokuOscillatorError,
    IchimokuOscillatorInput, IchimokuOscillatorNormalizeMode, IchimokuOscillatorOutput,
    IchimokuOscillatorParams, IchimokuOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use ichimoku_oscillator::{
    ichimoku_oscillator_alloc, ichimoku_oscillator_batch_into, ichimoku_oscillator_batch_js,
    ichimoku_oscillator_free, ichimoku_oscillator_into, ichimoku_oscillator_js,
};
#[cfg(feature = "python")]
pub use ichimoku_oscillator::{
    ichimoku_oscillator_batch_py, ichimoku_oscillator_py, register_ichimoku_oscillator_module,
    IchimokuOscillatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use exponential_trend::exponential_trend_into;
pub use exponential_trend::{
    expand_grid_exponential_trend, exponential_trend, exponential_trend_batch_par_slice,
    exponential_trend_batch_slice, exponential_trend_batch_with_kernel,
    exponential_trend_into_slice, exponential_trend_with_kernel, ExponentialTrendBatchBuilder,
    ExponentialTrendBatchOutput, ExponentialTrendBatchRange, ExponentialTrendBuilder,
    ExponentialTrendData, ExponentialTrendError, ExponentialTrendInput, ExponentialTrendOutput,
    ExponentialTrendParams, ExponentialTrendStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use exponential_trend::{
    exponential_trend_alloc, exponential_trend_batch_into,
    exponential_trend_batch_unified_js as exponential_trend_batch, exponential_trend_free,
    exponential_trend_into, exponential_trend_js, ExponentialTrendStreamWasm,
};
#[cfg(feature = "python")]
pub use exponential_trend::{
    exponential_trend_batch_py, exponential_trend_py, ExponentialTrendStreamPy,
};
pub use ift_rsi::{
    ift_rsi, IftRsiBatchBuilder, IftRsiBatchOutput, IftRsiBatchRange, IftRsiBuilder, IftRsiError,
    IftRsiInput, IftRsiOutput, IftRsiParams, IftRsiStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use ift_rsi::{
    ift_rsi_alloc, ift_rsi_batch_unified_js, ift_rsi_free, ift_rsi_into, ift_rsi_js,
};
#[cfg(feature = "python")]
pub use ift_rsi::{ift_rsi_batch_py, ift_rsi_py, IftRsiStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use impulse_macd::impulse_macd_into;
pub use impulse_macd::{
    expand_grid_impulse_macd, impulse_macd, impulse_macd_batch_par_slice, impulse_macd_batch_slice,
    impulse_macd_batch_with_kernel, impulse_macd_into_slice, impulse_macd_with_kernel,
    ImpulseMacdBatchBuilder, ImpulseMacdBatchOutput, ImpulseMacdBatchRange, ImpulseMacdBuilder,
    ImpulseMacdData, ImpulseMacdError, ImpulseMacdInput, ImpulseMacdOutput, ImpulseMacdParams,
    ImpulseMacdStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use impulse_macd::{
    impulse_macd_alloc, impulse_macd_batch_into, impulse_macd_batch_js, impulse_macd_free,
    impulse_macd_into, impulse_macd_into_host, impulse_macd_js,
};
#[cfg(feature = "python")]
pub use impulse_macd::{impulse_macd_batch_py, impulse_macd_py, ImpulseMacdStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use insync_index::insync_index_into;
pub use insync_index::{
    expand_grid_insync_index, insync_index, insync_index_batch_par_slice, insync_index_batch_slice,
    insync_index_batch_with_kernel, insync_index_into_slice, insync_index_with_kernel,
    InsyncIndexBatchBuilder, InsyncIndexBatchOutput, InsyncIndexBatchRange, InsyncIndexBuilder,
    InsyncIndexData, InsyncIndexError, InsyncIndexInput, InsyncIndexOutput, InsyncIndexParams,
    InsyncIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use insync_index::{
    insync_index_alloc, insync_index_batch_into, insync_index_batch_js, insync_index_free,
    insync_index_into, insync_index_into_host, insync_index_js,
};
#[cfg(feature = "python")]
pub use insync_index::{insync_index_batch_py, insync_index_py, InsyncIndexStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use keltner_channel_width_oscillator::keltner_channel_width_oscillator_into;
pub use keltner_channel_width_oscillator::{
    expand_grid_keltner_channel_width_oscillator, keltner_channel_width_oscillator,
    keltner_channel_width_oscillator_batch_par_slice, keltner_channel_width_oscillator_batch_slice,
    keltner_channel_width_oscillator_batch_with_kernel,
    keltner_channel_width_oscillator_into_slice, keltner_channel_width_oscillator_with_kernel,
    KeltnerChannelWidthOscillatorBatchBuilder, KeltnerChannelWidthOscillatorBatchOutput,
    KeltnerChannelWidthOscillatorBatchRange, KeltnerChannelWidthOscillatorBuilder,
    KeltnerChannelWidthOscillatorData, KeltnerChannelWidthOscillatorError,
    KeltnerChannelWidthOscillatorInput, KeltnerChannelWidthOscillatorOutput,
    KeltnerChannelWidthOscillatorParams, KeltnerChannelWidthOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use keltner_channel_width_oscillator::{
    keltner_channel_width_oscillator_alloc, keltner_channel_width_oscillator_batch_into,
    keltner_channel_width_oscillator_batch_js, keltner_channel_width_oscillator_free,
    keltner_channel_width_oscillator_into, keltner_channel_width_oscillator_into_host,
    keltner_channel_width_oscillator_js,
};
#[cfg(feature = "python")]
pub use keltner_channel_width_oscillator::{
    keltner_channel_width_oscillator_batch_py, keltner_channel_width_oscillator_py,
    KeltnerChannelWidthOscillatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use leavitt_convolution_acceleration::leavitt_convolution_acceleration_into;
pub use leavitt_convolution_acceleration::{
    expand_grid_leavitt_convolution_acceleration, leavitt_convolution_acceleration,
    leavitt_convolution_acceleration_batch_par_slice, leavitt_convolution_acceleration_batch_slice,
    leavitt_convolution_acceleration_batch_with_kernel,
    leavitt_convolution_acceleration_into_slice, leavitt_convolution_acceleration_with_kernel,
    LeavittConvolutionAccelerationBatchBuilder, LeavittConvolutionAccelerationBatchOutput,
    LeavittConvolutionAccelerationBatchRange, LeavittConvolutionAccelerationBuilder,
    LeavittConvolutionAccelerationData, LeavittConvolutionAccelerationError,
    LeavittConvolutionAccelerationInput, LeavittConvolutionAccelerationOutput,
    LeavittConvolutionAccelerationParams, LeavittConvolutionAccelerationStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use leavitt_convolution_acceleration::{
    leavitt_convolution_acceleration_alloc, leavitt_convolution_acceleration_batch_into,
    leavitt_convolution_acceleration_batch_js, leavitt_convolution_acceleration_free,
    leavitt_convolution_acceleration_into, leavitt_convolution_acceleration_into_host,
    leavitt_convolution_acceleration_js,
};
#[cfg(feature = "python")]
pub use leavitt_convolution_acceleration::{
    leavitt_convolution_acceleration_batch_py, leavitt_convolution_acceleration_py,
    LeavittConvolutionAccelerationStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use linear_regression_intensity::linear_regression_intensity_into;
pub use linear_regression_intensity::{
    expand_grid_linear_regression_intensity, linear_regression_intensity,
    linear_regression_intensity_batch_par_slice, linear_regression_intensity_batch_slice,
    linear_regression_intensity_batch_with_kernel, linear_regression_intensity_into_slice,
    linear_regression_intensity_with_kernel, LinearRegressionIntensityBatchBuilder,
    LinearRegressionIntensityBatchOutput, LinearRegressionIntensityBatchRange,
    LinearRegressionIntensityBuilder, LinearRegressionIntensityData,
    LinearRegressionIntensityError, LinearRegressionIntensityInput,
    LinearRegressionIntensityOutput, LinearRegressionIntensityParams,
    LinearRegressionIntensityStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use linear_regression_intensity::{
    linear_regression_intensity_alloc, linear_regression_intensity_batch_into,
    linear_regression_intensity_batch_js, linear_regression_intensity_free,
    linear_regression_intensity_into, linear_regression_intensity_into_host,
    linear_regression_intensity_js,
};
#[cfg(feature = "python")]
pub use linear_regression_intensity::{
    linear_regression_intensity_batch_py, linear_regression_intensity_py,
    LinearRegressionIntensityStreamPy,
};
pub use l1_ehlers_phasor::expand_grid as l1_ehlers_phasor_expand_grid;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use l1_ehlers_phasor::l1_ehlers_phasor_into;
pub use l1_ehlers_phasor::{
    l1_ehlers_phasor, l1_ehlers_phasor_batch_into_slice, l1_ehlers_phasor_batch_par_slice,
    l1_ehlers_phasor_batch_slice, l1_ehlers_phasor_batch_with_kernel, l1_ehlers_phasor_into_slice,
    l1_ehlers_phasor_with_kernel, L1EhlersPhasorBatchBuilder, L1EhlersPhasorBatchOutput,
    L1EhlersPhasorBatchRange, L1EhlersPhasorBuilder, L1EhlersPhasorData, L1EhlersPhasorError,
    L1EhlersPhasorInput, L1EhlersPhasorOutput, L1EhlersPhasorParams, L1EhlersPhasorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use l1_ehlers_phasor::{
    l1_ehlers_phasor_alloc, l1_ehlers_phasor_batch_into, l1_ehlers_phasor_batch_js,
    l1_ehlers_phasor_free, l1_ehlers_phasor_into_wasm as l1_ehlers_phasor_into,
    l1_ehlers_phasor_js,
};
#[cfg(feature = "python")]
pub use l1_ehlers_phasor::{
    l1_ehlers_phasor_batch_py, l1_ehlers_phasor_py, register_l1_ehlers_phasor_module,
    L1EhlersPhasorStreamPy,
};
pub use linearreg_angle::{
    linearreg_angle, Linearreg_angleInput, Linearreg_angleOutput, Linearreg_angleParams,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use market_meanness_index::market_meanness_index_into;
pub use market_meanness_index::{
    expand_grid_market_meanness_index, market_meanness_index,
    market_meanness_index_batch_par_slice, market_meanness_index_batch_slice,
    market_meanness_index_batch_with_kernel, market_meanness_index_into_slice,
    market_meanness_index_with_kernel, MarketMeannessIndexBatchBuilder,
    MarketMeannessIndexBatchOutput, MarketMeannessIndexBatchRange, MarketMeannessIndexBuilder,
    MarketMeannessIndexData, MarketMeannessIndexError, MarketMeannessIndexInput,
    MarketMeannessIndexOutput, MarketMeannessIndexParams, MarketMeannessIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use market_meanness_index::{
    market_meanness_index_alloc, market_meanness_index_batch_into, market_meanness_index_batch_js,
    market_meanness_index_free, market_meanness_index_into, market_meanness_index_into_host,
    market_meanness_index_js,
};
#[cfg(feature = "python")]
pub use market_meanness_index::{
    market_meanness_index_batch_py, market_meanness_index_py, MarketMeannessIndexStreamPy,
};
pub use mean_ad::{mean_ad, MeanAdInput, MeanAdOutput, MeanAdParams};
pub use mesa_stochastic_multi_length::expand_grid as mesa_stochastic_multi_length_expand_grid;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use mesa_stochastic_multi_length::mesa_stochastic_multi_length_into;
pub use mesa_stochastic_multi_length::{
    mesa_stochastic_multi_length, mesa_stochastic_multi_length_batch_into_slice,
    mesa_stochastic_multi_length_batch_par_slice, mesa_stochastic_multi_length_batch_slice,
    mesa_stochastic_multi_length_batch_with_kernel, mesa_stochastic_multi_length_into_slice,
    mesa_stochastic_multi_length_with_kernel, MesaStochasticMultiLengthBatchBuilder,
    MesaStochasticMultiLengthBatchOutput, MesaStochasticMultiLengthBatchRange,
    MesaStochasticMultiLengthBuilder, MesaStochasticMultiLengthData,
    MesaStochasticMultiLengthError, MesaStochasticMultiLengthInput,
    MesaStochasticMultiLengthOutput, MesaStochasticMultiLengthParams,
    MesaStochasticMultiLengthStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use mesa_stochastic_multi_length::{
    mesa_stochastic_multi_length_alloc, mesa_stochastic_multi_length_batch_into,
    mesa_stochastic_multi_length_batch_js, mesa_stochastic_multi_length_free,
    mesa_stochastic_multi_length_into, mesa_stochastic_multi_length_js,
};
#[cfg(feature = "python")]
pub use mesa_stochastic_multi_length::{
    mesa_stochastic_multi_length_batch_py, mesa_stochastic_multi_length_py,
    register_mesa_stochastic_multi_length_module, MesaStochasticMultiLengthStreamPy,
};
pub use mom::{mom, MomInput, MomOutput, MomParams};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use momentum_ratio_oscillator::momentum_ratio_oscillator_into;
pub use momentum_ratio_oscillator::{
    expand_grid_momentum_ratio_oscillator, momentum_ratio_oscillator,
    momentum_ratio_oscillator_batch_par_slice, momentum_ratio_oscillator_batch_slice,
    momentum_ratio_oscillator_batch_with_kernel, momentum_ratio_oscillator_into_slice,
    momentum_ratio_oscillator_with_kernel, MomentumRatioOscillatorBatchBuilder,
    MomentumRatioOscillatorBatchOutput, MomentumRatioOscillatorBatchRange,
    MomentumRatioOscillatorBuilder, MomentumRatioOscillatorData, MomentumRatioOscillatorError,
    MomentumRatioOscillatorInput, MomentumRatioOscillatorOutput, MomentumRatioOscillatorParams,
    MomentumRatioOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use momentum_ratio_oscillator::{
    momentum_ratio_oscillator_alloc, momentum_ratio_oscillator_batch_into,
    momentum_ratio_oscillator_batch_js, momentum_ratio_oscillator_free,
    momentum_ratio_oscillator_into, momentum_ratio_oscillator_into_host,
    momentum_ratio_oscillator_js,
};
#[cfg(feature = "python")]
pub use momentum_ratio_oscillator::{
    momentum_ratio_oscillator_batch_py, momentum_ratio_oscillator_py,
    MomentumRatioOscillatorStreamPy,
};
pub use moving_average_cross_probability::moving_average_cross_probability_expand_grid;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use moving_average_cross_probability::moving_average_cross_probability_into;
pub use moving_average_cross_probability::{
    moving_average_cross_probability, moving_average_cross_probability_batch_into_slice,
    moving_average_cross_probability_batch_par_slice, moving_average_cross_probability_batch_slice,
    moving_average_cross_probability_batch_with_kernel,
    moving_average_cross_probability_into_slice, moving_average_cross_probability_with_kernel,
    MovingAverageCrossProbabilityBatchBuilder, MovingAverageCrossProbabilityBatchOutput,
    MovingAverageCrossProbabilityBatchRange, MovingAverageCrossProbabilityBuilder,
    MovingAverageCrossProbabilityData, MovingAverageCrossProbabilityError,
    MovingAverageCrossProbabilityInput, MovingAverageCrossProbabilityMaType,
    MovingAverageCrossProbabilityOutput, MovingAverageCrossProbabilityParams,
    MovingAverageCrossProbabilityStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use moving_average_cross_probability::{
    moving_average_cross_probability_alloc, moving_average_cross_probability_batch_into,
    moving_average_cross_probability_batch_js, moving_average_cross_probability_free,
    moving_average_cross_probability_into, moving_average_cross_probability_js,
};
#[cfg(feature = "python")]
pub use moving_average_cross_probability::{
    moving_average_cross_probability_batch_py, moving_average_cross_probability_py,
    register_moving_average_cross_probability_module, MovingAverageCrossProbabilityStreamPy,
};
pub use moving_averages::{
    alma, buff_averages, corrected_moving_average, cwma, dema, edcf, ehlers_itrend, ehlers_pma,
    ema, ema_deviation_corrected_t3, epma, frama, fwma, gaussian, highpass, highpass_2_pole, hma,
    hwma, jma, jsa, kama, linreg, logarithmic_moving_average, maaq, mama, mwdx, nma, pwma, reflex,
    sinwma, sma, smma, sqwma, srwma, supersmoother, supersmoother_3_pole, swma, tema, tilson,
    tradjema, trendflex, trima, uma, volatility_adjusted_ma, volume_adjusted_ma, vpwma, vwap, vwma,
    wave_smoother, wilders, wma, zlema,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use parkinson_volatility::parkinson_volatility_into;
pub use parkinson_volatility::{
    parkinson_volatility, parkinson_volatility_batch_par_slice, parkinson_volatility_batch_slice,
    parkinson_volatility_batch_with_kernel, parkinson_volatility_into_slice,
    parkinson_volatility_with_kernel, ParkinsonVolatilityBatchBuilder,
    ParkinsonVolatilityBatchOutput, ParkinsonVolatilityBatchRange, ParkinsonVolatilityBuilder,
    ParkinsonVolatilityData, ParkinsonVolatilityError, ParkinsonVolatilityInput,
    ParkinsonVolatilityOutput, ParkinsonVolatilityParams, ParkinsonVolatilityStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use parkinson_volatility::{
    parkinson_volatility_alloc, parkinson_volatility_batch_into, parkinson_volatility_batch_js,
    parkinson_volatility_free, parkinson_volatility_into, parkinson_volatility_js,
};
#[cfg(feature = "python")]
pub use parkinson_volatility::{
    parkinson_volatility_batch_py, parkinson_volatility_py, ParkinsonVolatilityStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use pretty_good_oscillator::pretty_good_oscillator_into;
pub use pretty_good_oscillator::{
    pretty_good_oscillator, pretty_good_oscillator_batch_par_slice,
    pretty_good_oscillator_batch_slice, pretty_good_oscillator_batch_with_kernel,
    pretty_good_oscillator_into_slice, pretty_good_oscillator_with_kernel,
    PrettyGoodOscillatorBatchBuilder, PrettyGoodOscillatorBatchOutput,
    PrettyGoodOscillatorBatchRange, PrettyGoodOscillatorBuilder, PrettyGoodOscillatorData,
    PrettyGoodOscillatorError, PrettyGoodOscillatorInput, PrettyGoodOscillatorOutput,
    PrettyGoodOscillatorParams, PrettyGoodOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use pretty_good_oscillator::{
    pretty_good_oscillator_alloc, pretty_good_oscillator_batch_into,
    pretty_good_oscillator_batch_js, pretty_good_oscillator_free, pretty_good_oscillator_into,
    pretty_good_oscillator_into_host, pretty_good_oscillator_js,
};
#[cfg(feature = "python")]
pub use pretty_good_oscillator::{
    pretty_good_oscillator_batch_py, pretty_good_oscillator_py, PrettyGoodOscillatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use price_density_market_noise::price_density_market_noise_into;
pub use price_density_market_noise::{
    expand_grid_price_density_market_noise, price_density_market_noise,
    price_density_market_noise_batch_par_slice, price_density_market_noise_batch_slice,
    price_density_market_noise_batch_with_kernel, price_density_market_noise_into_slice,
    price_density_market_noise_with_kernel, PriceDensityMarketNoiseBatchBuilder,
    PriceDensityMarketNoiseBatchOutput, PriceDensityMarketNoiseBatchRange,
    PriceDensityMarketNoiseBuilder, PriceDensityMarketNoiseData, PriceDensityMarketNoiseError,
    PriceDensityMarketNoiseInput, PriceDensityMarketNoiseOutput, PriceDensityMarketNoiseParams,
    PriceDensityMarketNoiseStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use price_density_market_noise::{
    price_density_market_noise_alloc, price_density_market_noise_batch_into,
    price_density_market_noise_batch_js, price_density_market_noise_free,
    price_density_market_noise_into, price_density_market_noise_into_host,
    price_density_market_noise_js,
};
#[cfg(feature = "python")]
pub use price_density_market_noise::{
    price_density_market_noise_batch_py, price_density_market_noise_py,
    PriceDensityMarketNoiseStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use psychological_line::psychological_line_into;
pub use psychological_line::{
    expand_grid_psychological_line, psychological_line, psychological_line_batch_par_slice,
    psychological_line_batch_slice, psychological_line_batch_with_kernel,
    psychological_line_into_slice, psychological_line_with_kernel, PsychologicalLineBatchBuilder,
    PsychologicalLineBatchOutput, PsychologicalLineBatchRange, PsychologicalLineBuilder,
    PsychologicalLineData, PsychologicalLineError, PsychologicalLineInput, PsychologicalLineOutput,
    PsychologicalLineParams, PsychologicalLineStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use psychological_line::{
    psychological_line_alloc, psychological_line_batch_into, psychological_line_batch_js,
    psychological_line_free, psychological_line_into, psychological_line_into_host,
    psychological_line_js,
};
#[cfg(feature = "python")]
pub use psychological_line::{
    psychological_line_batch_py, psychological_line_py, PsychologicalLineStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use rank_correlation_index::rank_correlation_index_into;
pub use rank_correlation_index::{
    expand_grid_rank_correlation_index, rank_correlation_index,
    rank_correlation_index_batch_par_slice, rank_correlation_index_batch_slice,
    rank_correlation_index_batch_with_kernel, rank_correlation_index_into_slice,
    rank_correlation_index_with_kernel, RankCorrelationIndexBatchBuilder,
    RankCorrelationIndexBatchOutput, RankCorrelationIndexBatchRange, RankCorrelationIndexBuilder,
    RankCorrelationIndexData, RankCorrelationIndexError, RankCorrelationIndexInput,
    RankCorrelationIndexOutput, RankCorrelationIndexParams, RankCorrelationIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use rank_correlation_index::{
    rank_correlation_index_alloc, rank_correlation_index_batch_into,
    rank_correlation_index_batch_js, rank_correlation_index_free, rank_correlation_index_into,
    rank_correlation_index_into_host, rank_correlation_index_js,
};
#[cfg(feature = "python")]
pub use rank_correlation_index::{
    rank_correlation_index_batch_py, rank_correlation_index_py, RankCorrelationIndexStreamPy,
};
pub use rsi::{rsi, RsiBatchOutput, RsiInput, RsiOutput, RsiParams, RsiStream};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use smoothed_gaussian_trend_filter::smoothed_gaussian_trend_filter_into;
pub use smoothed_gaussian_trend_filter::{
    expand_grid_smoothed_gaussian_trend_filter, smoothed_gaussian_trend_filter,
    smoothed_gaussian_trend_filter_batch_par_slice, smoothed_gaussian_trend_filter_batch_slice,
    smoothed_gaussian_trend_filter_batch_with_kernel, smoothed_gaussian_trend_filter_into_slice,
    smoothed_gaussian_trend_filter_with_kernel, SmoothedGaussianTrendFilterBatchBuilder,
    SmoothedGaussianTrendFilterBatchOutput, SmoothedGaussianTrendFilterBatchRange,
    SmoothedGaussianTrendFilterBuilder, SmoothedGaussianTrendFilterData,
    SmoothedGaussianTrendFilterError, SmoothedGaussianTrendFilterInput,
    SmoothedGaussianTrendFilterOutput, SmoothedGaussianTrendFilterParams,
    SmoothedGaussianTrendFilterStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use smoothed_gaussian_trend_filter::{
    smoothed_gaussian_trend_filter_alloc, smoothed_gaussian_trend_filter_batch_into,
    smoothed_gaussian_trend_filter_batch_js, smoothed_gaussian_trend_filter_free,
    smoothed_gaussian_trend_filter_into, smoothed_gaussian_trend_filter_into_host,
    smoothed_gaussian_trend_filter_js,
};
#[cfg(feature = "python")]
pub use smoothed_gaussian_trend_filter::{
    smoothed_gaussian_trend_filter_batch_py, smoothed_gaussian_trend_filter_py,
    SmoothedGaussianTrendFilterStreamPy,
};
pub use smooth_theil_sen::smooth_theil_sen_expand_grid;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use smooth_theil_sen::smooth_theil_sen_into;
#[cfg(feature = "python")]
pub use smooth_theil_sen::{
    register_smooth_theil_sen_module, smooth_theil_sen_batch_py, smooth_theil_sen_py,
    SmoothTheilSenStreamPy,
};
pub use smooth_theil_sen::{
    smooth_theil_sen, smooth_theil_sen_batch_into_slice, smooth_theil_sen_batch_par_slice,
    smooth_theil_sen_batch_slice, smooth_theil_sen_batch_with_kernel, smooth_theil_sen_into_slice,
    smooth_theil_sen_with_kernel, SmoothTheilSenBatchBuilder, SmoothTheilSenBatchOutput,
    SmoothTheilSenBatchRange, SmoothTheilSenBuilder, SmoothTheilSenData,
    SmoothTheilSenDeviationType, SmoothTheilSenError, SmoothTheilSenInput, SmoothTheilSenOutput,
    SmoothTheilSenParams, SmoothTheilSenStatStyle, SmoothTheilSenStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use smooth_theil_sen::{
    smooth_theil_sen_alloc, smooth_theil_sen_batch_into, smooth_theil_sen_batch_js,
    smooth_theil_sen_free, smooth_theil_sen_into, smooth_theil_sen_js,
};
pub use spearman_correlation::expand_grid as spearman_correlation_expand_grid;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use spearman_correlation::spearman_correlation_into;
#[cfg(feature = "python")]
pub use spearman_correlation::{
    register_spearman_correlation_module, spearman_correlation_batch_py, spearman_correlation_py,
    SpearmanCorrelationStreamPy,
};
pub use spearman_correlation::{
    spearman_correlation, spearman_correlation_batch_into_slice,
    spearman_correlation_batch_par_slice, spearman_correlation_batch_slice,
    spearman_correlation_batch_with_kernel, spearman_correlation_into_slice,
    spearman_correlation_with_kernel, SpearmanCorrelationBatchBuilder,
    SpearmanCorrelationBatchOutput, SpearmanCorrelationBatchRange, SpearmanCorrelationBuilder,
    SpearmanCorrelationData, SpearmanCorrelationError, SpearmanCorrelationInput,
    SpearmanCorrelationOutput, SpearmanCorrelationParams, SpearmanCorrelationStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use spearman_correlation::{
    spearman_correlation_alloc, spearman_correlation_batch_into, spearman_correlation_batch_js,
    spearman_correlation_free, spearman_correlation_into, spearman_correlation_js,
};
pub use squeeze_momentum::{
    squeeze_momentum, SqueezeMomentumBatchOutput, SqueezeMomentumBatchParams,
    SqueezeMomentumBuilder, SqueezeMomentumInput, SqueezeMomentumOutput, SqueezeMomentumParams,
    SqueezeMomentumStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use squeeze_momentum::{
    squeeze_momentum_alloc, squeeze_momentum_batch, squeeze_momentum_free, squeeze_momentum_into,
    squeeze_momentum_js, SmiBatchJsOutput, SmiResult,
};
#[cfg(feature = "python")]
pub use squeeze_momentum::{
    squeeze_momentum_batch_py, squeeze_momentum_py, SqueezeMomentumStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use stochastic_adaptive_d::stochastic_adaptive_d_into;
pub use stochastic_adaptive_d::{
    expand_grid_stochastic_adaptive_d, stochastic_adaptive_d,
    stochastic_adaptive_d_batch_par_slice, stochastic_adaptive_d_batch_slice,
    stochastic_adaptive_d_batch_with_kernel, stochastic_adaptive_d_into_slice,
    stochastic_adaptive_d_with_kernel, StochasticAdaptiveDBatchBuilder,
    StochasticAdaptiveDBatchOutput, StochasticAdaptiveDBatchRange, StochasticAdaptiveDBuilder,
    StochasticAdaptiveDData, StochasticAdaptiveDError, StochasticAdaptiveDInput,
    StochasticAdaptiveDOutput, StochasticAdaptiveDParams, StochasticAdaptiveDStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use stochastic_adaptive_d::{
    stochastic_adaptive_d_alloc, stochastic_adaptive_d_batch_into, stochastic_adaptive_d_batch_js,
    stochastic_adaptive_d_free, stochastic_adaptive_d_into, stochastic_adaptive_d_into_host,
    stochastic_adaptive_d_js,
};
#[cfg(feature = "python")]
pub use stochastic_adaptive_d::{
    stochastic_adaptive_d_batch_py, stochastic_adaptive_d_py, StochasticAdaptiveDStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use stochastic_connors_rsi::stochastic_connors_rsi_into;
pub use stochastic_connors_rsi::{
    expand_grid_stochastic_connors_rsi, stochastic_connors_rsi,
    stochastic_connors_rsi_batch_par_slice, stochastic_connors_rsi_batch_slice,
    stochastic_connors_rsi_batch_with_kernel, stochastic_connors_rsi_into_slice,
    stochastic_connors_rsi_with_kernel, StochasticConnorsRsiBatchBuilder,
    StochasticConnorsRsiBatchOutput, StochasticConnorsRsiBatchRange, StochasticConnorsRsiBuilder,
    StochasticConnorsRsiData, StochasticConnorsRsiError, StochasticConnorsRsiInput,
    StochasticConnorsRsiOutput, StochasticConnorsRsiParams, StochasticConnorsRsiStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use stochastic_connors_rsi::{
    stochastic_connors_rsi_alloc, stochastic_connors_rsi_batch_into,
    stochastic_connors_rsi_batch_js, stochastic_connors_rsi_free, stochastic_connors_rsi_into,
    stochastic_connors_rsi_into_host, stochastic_connors_rsi_js,
};
#[cfg(feature = "python")]
pub use stochastic_connors_rsi::{
    stochastic_connors_rsi_batch_py, stochastic_connors_rsi_py, StochasticConnorsRsiStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use supertrend_oscillator::supertrend_oscillator_into;
pub use supertrend_oscillator::{
    expand_grid_supertrend_oscillator, supertrend_oscillator,
    supertrend_oscillator_batch_par_slice, supertrend_oscillator_batch_slice,
    supertrend_oscillator_batch_with_kernel, supertrend_oscillator_into_slice,
    supertrend_oscillator_with_kernel, SuperTrendOscillatorBatchBuilder,
    SuperTrendOscillatorBatchOutput, SuperTrendOscillatorBatchRange, SuperTrendOscillatorBuilder,
    SuperTrendOscillatorData, SuperTrendOscillatorError, SuperTrendOscillatorInput,
    SuperTrendOscillatorOutput, SuperTrendOscillatorParams, SuperTrendOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use supertrend_oscillator::{
    supertrend_oscillator_alloc, supertrend_oscillator_batch_into, supertrend_oscillator_batch_js,
    supertrend_oscillator_free, supertrend_oscillator_into, supertrend_oscillator_into_host,
    supertrend_oscillator_js,
};
#[cfg(feature = "python")]
pub use supertrend_oscillator::{
    supertrend_oscillator_batch_py, supertrend_oscillator_py, SuperTrendOscillatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use trend_continuation_factor::trend_continuation_factor_into;
pub use trend_continuation_factor::{
    expand_grid_trend_continuation_factor, trend_continuation_factor,
    trend_continuation_factor_batch_par_slice, trend_continuation_factor_batch_slice,
    trend_continuation_factor_batch_with_kernel, trend_continuation_factor_into_slice,
    trend_continuation_factor_with_kernel, TrendContinuationFactorBatchBuilder,
    TrendContinuationFactorBatchOutput, TrendContinuationFactorBatchRange,
    TrendContinuationFactorBuilder, TrendContinuationFactorData, TrendContinuationFactorError,
    TrendContinuationFactorInput, TrendContinuationFactorOutput, TrendContinuationFactorParams,
    TrendContinuationFactorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use trend_continuation_factor::{
    trend_continuation_factor_alloc, trend_continuation_factor_batch_into,
    trend_continuation_factor_batch_js, trend_continuation_factor_free,
    trend_continuation_factor_into, trend_continuation_factor_into_host,
    trend_continuation_factor_js,
};
#[cfg(feature = "python")]
pub use trend_continuation_factor::{
    trend_continuation_factor_batch_py, trend_continuation_factor_py,
    TrendContinuationFactorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use trend_direction_force_index::trend_direction_force_index_into;
pub use trend_direction_force_index::{
    expand_grid_trend_direction_force_index, trend_direction_force_index,
    trend_direction_force_index_batch_par_slice, trend_direction_force_index_batch_slice,
    trend_direction_force_index_batch_with_kernel, trend_direction_force_index_into_slice,
    trend_direction_force_index_with_kernel, TrendDirectionForceIndexBatchBuilder,
    TrendDirectionForceIndexBatchOutput, TrendDirectionForceIndexBatchRange,
    TrendDirectionForceIndexBuilder, TrendDirectionForceIndexData, TrendDirectionForceIndexError,
    TrendDirectionForceIndexInput, TrendDirectionForceIndexOutput, TrendDirectionForceIndexParams,
    TrendDirectionForceIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use trend_direction_force_index::{
    trend_direction_force_index_alloc, trend_direction_force_index_batch_into,
    trend_direction_force_index_batch_js, trend_direction_force_index_free,
    trend_direction_force_index_into, trend_direction_force_index_into_host,
    trend_direction_force_index_js,
};
#[cfg(feature = "python")]
pub use trend_direction_force_index::{
    trend_direction_force_index_batch_py, trend_direction_force_index_py,
    TrendDirectionForceIndexStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use trend_follower::trend_follower_into;
pub use trend_follower::{
    expand_grid_trend_follower, trend_follower, trend_follower_batch_par_slice,
    trend_follower_batch_slice, trend_follower_batch_with_kernel, trend_follower_into_slice,
    trend_follower_with_kernel, TrendFollowerBatchBuilder, TrendFollowerBatchOutput,
    TrendFollowerBatchRange, TrendFollowerBuilder, TrendFollowerData, TrendFollowerError,
    TrendFollowerInput, TrendFollowerOutput, TrendFollowerParams, TrendFollowerStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use trend_follower::{
    trend_follower_alloc, trend_follower_batch_into, trend_follower_batch_js, trend_follower_free,
    trend_follower_into, trend_follower_into_host, trend_follower_js,
};
#[cfg(feature = "python")]
pub use trend_follower::{trend_follower_batch_py, trend_follower_py, TrendFollowerStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use trend_trigger_factor::trend_trigger_factor_into;
#[cfg(feature = "python")]
pub use trend_trigger_factor::{
    register_trend_trigger_factor_module, trend_trigger_factor_batch_py, trend_trigger_factor_py,
    TrendTriggerFactorStreamPy,
};
pub use trend_trigger_factor::{
    trend_trigger_factor, trend_trigger_factor_batch_into_slice,
    trend_trigger_factor_batch_par_slice, trend_trigger_factor_batch_slice,
    trend_trigger_factor_batch_with_kernel, trend_trigger_factor_into_slice,
    trend_trigger_factor_with_kernel, TrendTriggerFactorBatchBuilder,
    TrendTriggerFactorBatchOutput, TrendTriggerFactorBatchRange, TrendTriggerFactorBuilder,
    TrendTriggerFactorData, TrendTriggerFactorError, TrendTriggerFactorInput,
    TrendTriggerFactorOutput, TrendTriggerFactorParams, TrendTriggerFactorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use trend_trigger_factor::{
    trend_trigger_factor_alloc, trend_trigger_factor_batch_into, trend_trigger_factor_batch_js,
    trend_trigger_factor_free, trend_trigger_factor_into, trend_trigger_factor_js,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use trend_flow_trail::trend_flow_trail_into;
pub use trend_flow_trail::{
    expand_grid_trend_flow_trail, trend_flow_trail, trend_flow_trail_batch_par_slice,
    trend_flow_trail_batch_slice, trend_flow_trail_batch_with_kernel, trend_flow_trail_into_slice,
    trend_flow_trail_with_kernel, TrendFlowTrailBatchBuilder, TrendFlowTrailBatchOutput,
    TrendFlowTrailBatchRange, TrendFlowTrailBuilder, TrendFlowTrailData, TrendFlowTrailError,
    TrendFlowTrailInput, TrendFlowTrailOutput, TrendFlowTrailParams, TrendFlowTrailStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use trend_flow_trail::{
    trend_flow_trail_alloc, trend_flow_trail_batch_unified_js as trend_flow_trail_batch,
    trend_flow_trail_free, trend_flow_trail_into, trend_flow_trail_js, TrendFlowTrailStreamWasm,
};
#[cfg(feature = "python")]
pub use trend_flow_trail::{
    trend_flow_trail_batch_py, trend_flow_trail_py, TrendFlowTrailStreamPy,
};
pub use trix::{trix, TrixBatchOutput, TrixInput, TrixOutput, TrixParams, TrixStream};
#[cfg(feature = "python")]
pub use trix::{trix_batch_py, trix_py, TrixStreamPy};
pub use tsf::{
    tsf, TsfBatchBuilder, TsfBatchOutput, TsfBatchRange, TsfBuilder, TsfError, TsfInput, TsfOutput,
    TsfParams, TsfStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use tsf::{tsf_alloc, tsf_batch_into, tsf_batch_unified_js, tsf_free, tsf_into, tsf_js};
#[cfg(feature = "python")]
pub use tsf::{tsf_batch_py, tsf_py, TsfStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use twiggs_money_flow::twiggs_money_flow_into;
pub use twiggs_money_flow::{
    twiggs_money_flow, twiggs_money_flow_batch_par_slice, twiggs_money_flow_batch_slice,
    twiggs_money_flow_batch_with_kernel, twiggs_money_flow_into_slice,
    twiggs_money_flow_with_kernel, TwiggsMoneyFlowBatchBuilder, TwiggsMoneyFlowBatchOutput,
    TwiggsMoneyFlowBatchRange, TwiggsMoneyFlowBuilder, TwiggsMoneyFlowData, TwiggsMoneyFlowError,
    TwiggsMoneyFlowInput, TwiggsMoneyFlowOutput, TwiggsMoneyFlowParams, TwiggsMoneyFlowStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use twiggs_money_flow::{
    twiggs_money_flow_alloc, twiggs_money_flow_batch_into, twiggs_money_flow_batch_js,
    twiggs_money_flow_free, twiggs_money_flow_into, twiggs_money_flow_into_host,
    twiggs_money_flow_js,
};
#[cfg(feature = "python")]
pub use twiggs_money_flow::{
    twiggs_money_flow_batch_py, twiggs_money_flow_py, TwiggsMoneyFlowStreamPy,
};
pub use ui::{ui, UiInput, UiOutput, UiParams};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use velocity::velocity_into;
pub use velocity::{
    velocity, velocity_batch_par_slice, velocity_batch_slice, velocity_batch_with_kernel,
    velocity_into_slice, VelocityBatchBuilder, VelocityBatchOutput, VelocityBatchRange,
    VelocityBuilder, VelocityData, VelocityError, VelocityInput, VelocityOutput, VelocityParams,
    VelocityStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use velocity::{
    velocity_alloc, velocity_batch_into, velocity_batch_js, velocity_free, velocity_into,
    velocity_js, VelocityStreamWasm,
};
#[cfg(feature = "python")]
pub use velocity::{velocity_batch_py, velocity_py, VelocityStreamPy};
pub use vidya::{
    vidya, VidyaBatchBuilder, VidyaBatchOutput, VidyaBatchRange, VidyaBuilder, VidyaData,
    VidyaError, VidyaInput, VidyaOutput, VidyaParams, VidyaStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use vidya::{vidya_alloc, vidya_batch_into, vidya_batch_js, vidya_free, vidya_into, vidya_js};
#[cfg(feature = "python")]
pub use vidya::{vidya_batch_py, vidya_py, VidyaStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use volatility_ratio_adaptive_rsx::volatility_ratio_adaptive_rsx_into;
pub use volatility_ratio_adaptive_rsx::{
    volatility_ratio_adaptive_rsx, volatility_ratio_adaptive_rsx_batch_par_slice,
    volatility_ratio_adaptive_rsx_batch_slice, volatility_ratio_adaptive_rsx_batch_with_kernel,
    volatility_ratio_adaptive_rsx_into_slice, volatility_ratio_adaptive_rsx_with_kernel,
    VolatilityRatioAdaptiveRsxBatchBuilder, VolatilityRatioAdaptiveRsxBatchOutput,
    VolatilityRatioAdaptiveRsxBatchRange, VolatilityRatioAdaptiveRsxBuilder,
    VolatilityRatioAdaptiveRsxData, VolatilityRatioAdaptiveRsxError,
    VolatilityRatioAdaptiveRsxInput, VolatilityRatioAdaptiveRsxOutput,
    VolatilityRatioAdaptiveRsxParams, VolatilityRatioAdaptiveRsxStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use volatility_ratio_adaptive_rsx::{
    volatility_ratio_adaptive_rsx_alloc, volatility_ratio_adaptive_rsx_batch_into,
    volatility_ratio_adaptive_rsx_batch_js, volatility_ratio_adaptive_rsx_free,
    volatility_ratio_adaptive_rsx_into, volatility_ratio_adaptive_rsx_into_host,
    volatility_ratio_adaptive_rsx_js,
};
#[cfg(feature = "python")]
pub use volatility_ratio_adaptive_rsx::{
    volatility_ratio_adaptive_rsx_batch_py, volatility_ratio_adaptive_rsx_py,
    VolatilityRatioAdaptiveRsxStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use volume_weighted_stochastic_rsi::volume_weighted_stochastic_rsi_into;
pub use volume_weighted_stochastic_rsi::{
    volume_weighted_stochastic_rsi, volume_weighted_stochastic_rsi_batch_par_slice,
    volume_weighted_stochastic_rsi_batch_slice, volume_weighted_stochastic_rsi_batch_with_kernel,
    volume_weighted_stochastic_rsi_into_slice, volume_weighted_stochastic_rsi_with_kernel,
    VolumeWeightedStochasticRsiBatchBuilder, VolumeWeightedStochasticRsiBatchOutput,
    VolumeWeightedStochasticRsiBatchRange, VolumeWeightedStochasticRsiBuilder,
    VolumeWeightedStochasticRsiData, VolumeWeightedStochasticRsiError,
    VolumeWeightedStochasticRsiInput, VolumeWeightedStochasticRsiOutput,
    VolumeWeightedStochasticRsiParams, VolumeWeightedStochasticRsiStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use volume_weighted_stochastic_rsi::{
    volume_weighted_stochastic_rsi_alloc, volume_weighted_stochastic_rsi_batch_into,
    volume_weighted_stochastic_rsi_batch_js, volume_weighted_stochastic_rsi_free,
    volume_weighted_stochastic_rsi_into, volume_weighted_stochastic_rsi_into_host,
    volume_weighted_stochastic_rsi_js,
};
#[cfg(feature = "python")]
pub use volume_weighted_stochastic_rsi::{
    volume_weighted_stochastic_rsi_batch_py, volume_weighted_stochastic_rsi_py,
    VolumeWeightedStochasticRsiStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use volatility_quality_index::volatility_quality_index_into;
pub use volatility_quality_index::{
    volatility_quality_index, volatility_quality_index_batch_into_slice,
    volatility_quality_index_batch_par_slice, volatility_quality_index_batch_slice,
    volatility_quality_index_batch_with_kernel, volatility_quality_index_into_slice,
    volatility_quality_index_with_kernel, VolatilityQualityIndexBatchBuilder,
    VolatilityQualityIndexBatchOutput, VolatilityQualityIndexBatchRange,
    VolatilityQualityIndexBuilder, VolatilityQualityIndexData, VolatilityQualityIndexError,
    VolatilityQualityIndexInput, VolatilityQualityIndexOutput, VolatilityQualityIndexParams,
    VolatilityQualityIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use volatility_quality_index::{
    volatility_quality_index_alloc, volatility_quality_index_batch_into,
    volatility_quality_index_batch_js, volatility_quality_index_free,
    volatility_quality_index_into, volatility_quality_index_js,
};
#[cfg(feature = "python")]
pub use volatility_quality_index::{
    volatility_quality_index_batch_py, volatility_quality_index_py, VolatilityQualityIndexStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use volume_zone_oscillator::volume_zone_oscillator_into;
pub use volume_zone_oscillator::{
    volume_zone_oscillator, volume_zone_oscillator_batch_into_slice,
    volume_zone_oscillator_batch_par_slice, volume_zone_oscillator_batch_slice,
    volume_zone_oscillator_batch_with_kernel, volume_zone_oscillator_into_slice,
    volume_zone_oscillator_with_kernel, VolumeZoneOscillatorBatchBuilder,
    VolumeZoneOscillatorBatchOutput, VolumeZoneOscillatorBatchRange, VolumeZoneOscillatorBuilder,
    VolumeZoneOscillatorData, VolumeZoneOscillatorError, VolumeZoneOscillatorInput,
    VolumeZoneOscillatorOutput, VolumeZoneOscillatorParams, VolumeZoneOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use volume_zone_oscillator::{
    volume_zone_oscillator_alloc, volume_zone_oscillator_batch_into,
    volume_zone_oscillator_batch_js, volume_zone_oscillator_free, volume_zone_oscillator_into,
    volume_zone_oscillator_js,
};
#[cfg(feature = "python")]
pub use volume_zone_oscillator::{
    volume_zone_oscillator_batch_py, volume_zone_oscillator_py, VolumeZoneOscillatorStreamPy,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use vpci::{
    vpci_alloc, vpci_batch_into, vpci_batch_unified_js, vpci_free, vpci_into, vpci_js, VpciContext,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use vwap_deviation_oscillator::vwap_deviation_oscillator_into;
#[cfg(feature = "python")]
pub use vwap_deviation_oscillator::{
    register_vwap_deviation_oscillator_module, vwap_deviation_oscillator_batch_py,
    vwap_deviation_oscillator_py, VwapDeviationOscillatorStreamPy,
};
pub use vwap_deviation_oscillator::{
    vwap_deviation_oscillator, vwap_deviation_oscillator_batch_into_slice,
    vwap_deviation_oscillator_batch_par_slice, vwap_deviation_oscillator_batch_slice,
    vwap_deviation_oscillator_batch_with_kernel, vwap_deviation_oscillator_into_slice,
    vwap_deviation_oscillator_with_kernel, VwapDeviationMode, VwapDeviationOscillatorBatchBuilder,
    VwapDeviationOscillatorBatchOutput, VwapDeviationOscillatorBatchRange,
    VwapDeviationOscillatorBuilder, VwapDeviationOscillatorData, VwapDeviationOscillatorError,
    VwapDeviationOscillatorInput, VwapDeviationOscillatorOutput, VwapDeviationOscillatorParams,
    VwapDeviationOscillatorStream, VwapDeviationSessionMode,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use vwap_deviation_oscillator::{
    vwap_deviation_oscillator_alloc, vwap_deviation_oscillator_batch_into,
    vwap_deviation_oscillator_batch_js, vwap_deviation_oscillator_free,
    vwap_deviation_oscillator_into, vwap_deviation_oscillator_js,
};
#[cfg(feature = "python")]
pub use wto::{wto_batch_py, wto_py, WtoStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use yang_zhang_volatility::yang_zhang_volatility_into;
pub use yang_zhang_volatility::{
    yang_zhang_volatility, yang_zhang_volatility_batch_par_slice,
    yang_zhang_volatility_batch_slice, yang_zhang_volatility_batch_with_kernel,
    yang_zhang_volatility_into_slice, yang_zhang_volatility_with_kernel,
    YangZhangVolatilityBatchBuilder, YangZhangVolatilityBatchOutput, YangZhangVolatilityBatchRange,
    YangZhangVolatilityBuilder, YangZhangVolatilityData, YangZhangVolatilityError,
    YangZhangVolatilityInput, YangZhangVolatilityOutput, YangZhangVolatilityParams,
    YangZhangVolatilityStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use yang_zhang_volatility::{
    yang_zhang_volatility_alloc, yang_zhang_volatility_batch_into, yang_zhang_volatility_batch_js,
    yang_zhang_volatility_free, yang_zhang_volatility_into, yang_zhang_volatility_js,
};
#[cfg(feature = "python")]
pub use yang_zhang_volatility::{
    yang_zhang_volatility_batch_py, yang_zhang_volatility_py, YangZhangVolatilityStreamPy,
};
