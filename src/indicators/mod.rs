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
pub mod autocorrelation_indicator;
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
pub mod cyberpunk_value_trend_analyzer;
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
pub mod emd_trend;
pub mod emv;
pub mod er;
pub mod eri;
pub mod evasive_supertrend;
pub mod fisher;
pub mod forward_backward_exponential_oscillator;
pub mod fosc;
pub mod fvg_positioning_average;
pub mod fvg_trailing_stop;
pub mod goertzel_cycle_composite_wave;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use cyberpunk_value_trend_analyzer::cyberpunk_value_trend_analyzer_into;
pub use cyberpunk_value_trend_analyzer::{
    cyberpunk_value_trend_analyzer, cyberpunk_value_trend_analyzer_batch_par_slice,
    cyberpunk_value_trend_analyzer_batch_slice, cyberpunk_value_trend_analyzer_batch_with_kernel,
    cyberpunk_value_trend_analyzer_into_slice, cyberpunk_value_trend_analyzer_with_kernel,
    expand_grid_cyberpunk_value_trend_analyzer, CyberpunkValueTrendAnalyzerBatchBuilder,
    CyberpunkValueTrendAnalyzerBatchOutput, CyberpunkValueTrendAnalyzerBatchRange,
    CyberpunkValueTrendAnalyzerBuilder, CyberpunkValueTrendAnalyzerData,
    CyberpunkValueTrendAnalyzerError, CyberpunkValueTrendAnalyzerInput,
    CyberpunkValueTrendAnalyzerOutput, CyberpunkValueTrendAnalyzerParams,
    CyberpunkValueTrendAnalyzerStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use cyberpunk_value_trend_analyzer::{
    cyberpunk_value_trend_analyzer_alloc, cyberpunk_value_trend_analyzer_batch_into,
    cyberpunk_value_trend_analyzer_batch_js, cyberpunk_value_trend_analyzer_free,
    cyberpunk_value_trend_analyzer_into, cyberpunk_value_trend_analyzer_js,
};
#[cfg(feature = "python")]
pub use cyberpunk_value_trend_analyzer::{
    cyberpunk_value_trend_analyzer_batch_py, cyberpunk_value_trend_analyzer_py,
    register_cyberpunk_value_trend_analyzer_module, CyberpunkValueTrendAnalyzerStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use emd_trend::emd_trend_into;
pub use emd_trend::{
    emd_trend, emd_trend_batch_par_slice, emd_trend_batch_slice, emd_trend_batch_with_kernel,
    emd_trend_into_slice, emd_trend_with_kernel, expand_grid_emd_trend, EmdTrendBatchBuilder,
    EmdTrendBatchOutput, EmdTrendBatchRange, EmdTrendBuilder, EmdTrendError, EmdTrendInput,
    EmdTrendOutput, EmdTrendParams, EmdTrendStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use emd_trend::{
    emd_trend_alloc, emd_trend_batch_into, emd_trend_batch_js, emd_trend_free, emd_trend_into,
    emd_trend_js,
};
#[cfg(feature = "python")]
pub use emd_trend::{
    emd_trend_batch_py, emd_trend_py, register_emd_trend_module, EmdTrendStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use reversal_signals::reversal_signals_into;
pub use reversal_signals::{
    expand_grid_reversal_signals, reversal_signals, reversal_signals_batch_par_slice,
    reversal_signals_batch_slice, reversal_signals_batch_with_kernel, reversal_signals_into_slice,
    reversal_signals_with_kernel, ReversalSignalsBatchBuilder, ReversalSignalsBatchOutput,
    ReversalSignalsBatchRange, ReversalSignalsBuilder, ReversalSignalsData,
    ReversalSignalsError, ReversalSignalsInput, ReversalSignalsOutput, ReversalSignalsParams,
    ReversalSignalsStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use reversal_signals::{
    reversal_signals_alloc, reversal_signals_batch_into, reversal_signals_batch_js,
    reversal_signals_free, reversal_signals_into, reversal_signals_js,
};
#[cfg(feature = "python")]
pub use reversal_signals::{
    register_reversal_signals_module, reversal_signals_batch_py, reversal_signals_py,
    ReversalSignalsStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use fvg_positioning_average::fvg_positioning_average_into;
pub use fvg_positioning_average::{
    expand_grid_fvg_positioning_average, fvg_positioning_average,
    fvg_positioning_average_batch_par_slice, fvg_positioning_average_batch_slice,
    fvg_positioning_average_batch_with_kernel, fvg_positioning_average_into_slice,
    fvg_positioning_average_with_kernel, FvgPositioningAverageBatchBuilder,
    FvgPositioningAverageBatchOutput, FvgPositioningAverageBatchRange,
    FvgPositioningAverageBuilder, FvgPositioningAverageError, FvgPositioningAverageInput,
    FvgPositioningAverageOutput, FvgPositioningAverageParams, FvgPositioningAverageStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use fvg_positioning_average::{
    fvg_positioning_average_alloc, fvg_positioning_average_batch_into,
    fvg_positioning_average_batch_js, fvg_positioning_average_free, fvg_positioning_average_into,
    fvg_positioning_average_js,
};
#[cfg(feature = "python")]
pub use fvg_positioning_average::{
    fvg_positioning_average_batch_py, fvg_positioning_average_py, FvgPositioningAverageStreamPy,
};
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
pub mod reversal_signals;
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
pub mod stochastic_money_flow_index;
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
pub mod candle_strength_oscillator;
pub mod directional_imbalance_index;
pub mod disparity_index;
pub mod donchian_channel_width;
pub mod dual_ulcer_index;
pub mod dynamic_momentum_index;
pub mod ehlers_data_sampling_relative_strength_indicator;
pub mod fractal_dimension_index;
pub mod gmma_oscillator;
pub mod historical_volatility_rank;
pub mod kairi_relative_index;
pub mod market_structure_trailing_stop;
pub mod nonlinear_regression_zero_lag_moving_average;
pub mod possible_rsi;
pub mod projection_oscillator;
pub mod rogers_satchell_volatility;
pub mod rolling_skewness_kurtosis;
pub mod rolling_z_score_trend;
pub mod trend_direction_force_index;
pub mod velocity_acceleration_convergence_divergence_indicator;
pub mod volume_weighted_rsi;
pub mod yang_zhang_volatility;
pub mod zig_zag_channels;
pub mod zscore;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use autocorrelation_indicator::autocorrelation_indicator_into;
pub use autocorrelation_indicator::{
    autocorrelation_indicator, autocorrelation_indicator_batch_par_slice,
    autocorrelation_indicator_batch_slice, autocorrelation_indicator_batch_with_kernel,
    autocorrelation_indicator_into_slice, autocorrelation_indicator_with_kernel,
    expand_grid_autocorrelation_indicator, AutocorrelationIndicatorBatchBuilder,
    AutocorrelationIndicatorBatchOutput, AutocorrelationIndicatorBatchRange,
    AutocorrelationIndicatorBuilder, AutocorrelationIndicatorData, AutocorrelationIndicatorError,
    AutocorrelationIndicatorInput, AutocorrelationIndicatorOutput, AutocorrelationIndicatorParams,
    AutocorrelationIndicatorStream, AutocorrelationIndicatorStreamPoint,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use autocorrelation_indicator::{
    autocorrelation_indicator_alloc, autocorrelation_indicator_batch_into,
    autocorrelation_indicator_batch_js, autocorrelation_indicator_free,
    autocorrelation_indicator_into, autocorrelation_indicator_js,
};
#[cfg(feature = "python")]
pub use autocorrelation_indicator::{
    autocorrelation_indicator_batch_py, autocorrelation_indicator_py,
    register_autocorrelation_indicator_module, AutocorrelationIndicatorStreamPy,
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
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use candle_strength_oscillator::candle_strength_oscillator_into;
pub use candle_strength_oscillator::{
    candle_strength_oscillator, candle_strength_oscillator_batch_par_slice,
    candle_strength_oscillator_batch_slice, candle_strength_oscillator_batch_with_kernel,
    candle_strength_oscillator_into_slice, candle_strength_oscillator_with_kernel,
    expand_grid_candle_strength_oscillator, CandleStrengthOscillatorBatchBuilder,
    CandleStrengthOscillatorBatchOutput, CandleStrengthOscillatorBatchRange,
    CandleStrengthOscillatorBuilder, CandleStrengthOscillatorData, CandleStrengthOscillatorError,
    CandleStrengthOscillatorInput, CandleStrengthOscillatorOutput, CandleStrengthOscillatorParams,
    CandleStrengthOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use candle_strength_oscillator::{
    candle_strength_oscillator_alloc, candle_strength_oscillator_batch_into,
    candle_strength_oscillator_batch_js, candle_strength_oscillator_free,
    candle_strength_oscillator_into, candle_strength_oscillator_js,
};
#[cfg(feature = "python")]
pub use candle_strength_oscillator::{
    candle_strength_oscillator_batch_py, candle_strength_oscillator_py,
    register_candle_strength_oscillator_module, CandleStrengthOscillatorStreamPy,
};
pub use cci::{cci, CciInput, CciOutput, CciParams};
pub use cfo::{cfo, CfoInput, CfoOutput, CfoParams};
pub use coppock::{coppock, CoppockInput, CoppockOutput, CoppockParams};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use directional_imbalance_index::directional_imbalance_index_into;
pub use directional_imbalance_index::{
    directional_imbalance_index, directional_imbalance_index_batch_par_slice,
    directional_imbalance_index_batch_slice, directional_imbalance_index_batch_with_kernel,
    directional_imbalance_index_into_slice, directional_imbalance_index_with_kernel,
    expand_grid_directional_imbalance_index, DirectionalImbalanceIndexBatchBuilder,
    DirectionalImbalanceIndexBatchOutput, DirectionalImbalanceIndexBatchRange,
    DirectionalImbalanceIndexBuilder, DirectionalImbalanceIndexData,
    DirectionalImbalanceIndexError, DirectionalImbalanceIndexInput,
    DirectionalImbalanceIndexOutput, DirectionalImbalanceIndexParams,
    DirectionalImbalanceIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use directional_imbalance_index::{
    directional_imbalance_index_alloc, directional_imbalance_index_batch_into,
    directional_imbalance_index_batch_js, directional_imbalance_index_free,
    directional_imbalance_index_into, directional_imbalance_index_js,
};
#[cfg(feature = "python")]
pub use directional_imbalance_index::{
    directional_imbalance_index_batch_py, directional_imbalance_index_py,
    register_directional_imbalance_index_module, DirectionalImbalanceIndexStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use disparity_index::disparity_index_into;
pub use disparity_index::{
    disparity_index, disparity_index_batch_par_slice, disparity_index_batch_slice,
    disparity_index_batch_with_kernel, disparity_index_into_slice, disparity_index_with_kernel,
    expand_grid_disparity_index, DisparityIndexBatchBuilder, DisparityIndexBatchOutput,
    DisparityIndexBatchRange, DisparityIndexBuilder, DisparityIndexData, DisparityIndexError,
    DisparityIndexInput, DisparityIndexOutput, DisparityIndexParams, DisparityIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use disparity_index::{
    disparity_index_alloc, disparity_index_batch_into, disparity_index_batch_js,
    disparity_index_free, disparity_index_into, disparity_index_js,
};
#[cfg(feature = "python")]
pub use disparity_index::{disparity_index_batch_py, disparity_index_py, DisparityIndexStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use donchian_channel_width::donchian_channel_width_into;
pub use donchian_channel_width::{
    donchian_channel_width, donchian_channel_width_batch_par_slice,
    donchian_channel_width_batch_slice, donchian_channel_width_batch_with_kernel,
    donchian_channel_width_into_slice, donchian_channel_width_with_kernel,
    expand_grid_donchian_channel_width, DonchianChannelWidthBatchBuilder,
    DonchianChannelWidthBatchOutput, DonchianChannelWidthBatchRange, DonchianChannelWidthBuilder,
    DonchianChannelWidthData, DonchianChannelWidthError, DonchianChannelWidthInput,
    DonchianChannelWidthOutput, DonchianChannelWidthParams, DonchianChannelWidthStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use donchian_channel_width::{
    donchian_channel_width_alloc, donchian_channel_width_batch_into,
    donchian_channel_width_batch_js, donchian_channel_width_free, donchian_channel_width_into,
    donchian_channel_width_js,
};
#[cfg(feature = "python")]
pub use donchian_channel_width::{
    donchian_channel_width_batch_py, donchian_channel_width_py, DonchianChannelWidthStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use dual_ulcer_index::dual_ulcer_index_into;
pub use dual_ulcer_index::{
    dual_ulcer_index, dual_ulcer_index_batch_par_slice, dual_ulcer_index_batch_slice,
    dual_ulcer_index_batch_with_kernel, dual_ulcer_index_into_slice, dual_ulcer_index_with_kernel,
    expand_grid_dual_ulcer_index, DualUlcerIndexBatchBuilder, DualUlcerIndexBatchOutput,
    DualUlcerIndexBatchRange, DualUlcerIndexBuilder, DualUlcerIndexData, DualUlcerIndexError,
    DualUlcerIndexInput, DualUlcerIndexOutput, DualUlcerIndexParams, DualUlcerIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use dual_ulcer_index::{
    dual_ulcer_index_alloc, dual_ulcer_index_batch_into, dual_ulcer_index_batch_js,
    dual_ulcer_index_free, dual_ulcer_index_into, dual_ulcer_index_js,
};
#[cfg(feature = "python")]
pub use dual_ulcer_index::{
    dual_ulcer_index_batch_py, dual_ulcer_index_py, DualUlcerIndexStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use dynamic_momentum_index::dynamic_momentum_index_into;
pub use dynamic_momentum_index::{
    dynamic_momentum_index, dynamic_momentum_index_batch_par_slice,
    dynamic_momentum_index_batch_slice, dynamic_momentum_index_batch_with_kernel,
    dynamic_momentum_index_into_slice, dynamic_momentum_index_with_kernel,
    expand_grid_dynamic_momentum_index, DynamicMomentumIndexBatchBuilder,
    DynamicMomentumIndexBatchOutput, DynamicMomentumIndexBatchRange, DynamicMomentumIndexBuilder,
    DynamicMomentumIndexData, DynamicMomentumIndexError, DynamicMomentumIndexInput,
    DynamicMomentumIndexOutput, DynamicMomentumIndexParams, DynamicMomentumIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use dynamic_momentum_index::{
    dynamic_momentum_index_alloc, dynamic_momentum_index_batch_into,
    dynamic_momentum_index_batch_js, dynamic_momentum_index_free, dynamic_momentum_index_into,
    dynamic_momentum_index_js,
};
#[cfg(feature = "python")]
pub use dynamic_momentum_index::{
    dynamic_momentum_index_batch_py, dynamic_momentum_index_py, DynamicMomentumIndexStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use ehlers_data_sampling_relative_strength_indicator::ehlers_data_sampling_relative_strength_indicator_into;
pub use ehlers_data_sampling_relative_strength_indicator::{
    ehlers_data_sampling_relative_strength_indicator,
    ehlers_data_sampling_relative_strength_indicator_batch_par_slice,
    ehlers_data_sampling_relative_strength_indicator_batch_slice,
    ehlers_data_sampling_relative_strength_indicator_batch_with_kernel,
    ehlers_data_sampling_relative_strength_indicator_into_slice,
    ehlers_data_sampling_relative_strength_indicator_with_kernel,
    expand_grid_ehlers_data_sampling_relative_strength_indicator,
    EhlersDataSamplingRelativeStrengthIndicatorBatchBuilder,
    EhlersDataSamplingRelativeStrengthIndicatorBatchOutput,
    EhlersDataSamplingRelativeStrengthIndicatorBatchRange,
    EhlersDataSamplingRelativeStrengthIndicatorBuilder,
    EhlersDataSamplingRelativeStrengthIndicatorData,
    EhlersDataSamplingRelativeStrengthIndicatorError,
    EhlersDataSamplingRelativeStrengthIndicatorInput,
    EhlersDataSamplingRelativeStrengthIndicatorOutput,
    EhlersDataSamplingRelativeStrengthIndicatorParams,
    EhlersDataSamplingRelativeStrengthIndicatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use ehlers_data_sampling_relative_strength_indicator::{
    ehlers_data_sampling_relative_strength_indicator_alloc,
    ehlers_data_sampling_relative_strength_indicator_batch_into,
    ehlers_data_sampling_relative_strength_indicator_batch_js,
    ehlers_data_sampling_relative_strength_indicator_free,
    ehlers_data_sampling_relative_strength_indicator_into,
    ehlers_data_sampling_relative_strength_indicator_js,
};
#[cfg(feature = "python")]
pub use ehlers_data_sampling_relative_strength_indicator::{
    ehlers_data_sampling_relative_strength_indicator_batch_py,
    ehlers_data_sampling_relative_strength_indicator_py,
    register_ehlers_data_sampling_relative_strength_indicator_module,
    EhlersDataSamplingRelativeStrengthIndicatorStreamPy,
};
pub use er::{er, ErInput, ErOutput, ErParams};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use evasive_supertrend::evasive_supertrend_into;
pub use evasive_supertrend::{
    evasive_supertrend, evasive_supertrend_batch_par_slice, evasive_supertrend_batch_slice,
    evasive_supertrend_batch_with_kernel, evasive_supertrend_into_slice,
    evasive_supertrend_with_kernel, expand_grid_evasive_supertrend, EvasiveSuperTrendBatchBuilder,
    EvasiveSuperTrendBatchOutput, EvasiveSuperTrendBatchRange, EvasiveSuperTrendBuilder,
    EvasiveSuperTrendData, EvasiveSuperTrendError, EvasiveSuperTrendInput, EvasiveSuperTrendOutput,
    EvasiveSuperTrendParams, EvasiveSuperTrendStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use evasive_supertrend::{
    evasive_supertrend_alloc, evasive_supertrend_batch_into, evasive_supertrend_batch_js,
    evasive_supertrend_free, evasive_supertrend_into, evasive_supertrend_js,
};
#[cfg(feature = "python")]
pub use evasive_supertrend::{
    evasive_supertrend_batch_py, evasive_supertrend_py, register_evasive_supertrend_module,
    EvasiveSuperTrendStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use fractal_dimension_index::fractal_dimension_index_into;
pub use fractal_dimension_index::{
    expand_grid_fractal_dimension_index, fractal_dimension_index,
    fractal_dimension_index_batch_par_slice, fractal_dimension_index_batch_slice,
    fractal_dimension_index_batch_with_kernel, fractal_dimension_index_into_slice,
    fractal_dimension_index_with_kernel, FractalDimensionIndexBatchBuilder,
    FractalDimensionIndexBatchOutput, FractalDimensionIndexBatchRange,
    FractalDimensionIndexBuilder, FractalDimensionIndexData, FractalDimensionIndexError,
    FractalDimensionIndexInput, FractalDimensionIndexOutput, FractalDimensionIndexParams,
    FractalDimensionIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use fractal_dimension_index::{
    fractal_dimension_index_alloc, fractal_dimension_index_batch_into,
    fractal_dimension_index_batch_js, fractal_dimension_index_free, fractal_dimension_index_into,
    fractal_dimension_index_js,
};
#[cfg(feature = "python")]
pub use fractal_dimension_index::{
    fractal_dimension_index_batch_py, fractal_dimension_index_py, FractalDimensionIndexStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use gmma_oscillator::gmma_oscillator_into;
pub use gmma_oscillator::{
    expand_grid_gmma_oscillator, gmma_oscillator, gmma_oscillator_batch_par_slice,
    gmma_oscillator_batch_slice, gmma_oscillator_batch_with_kernel, gmma_oscillator_into_slice,
    gmma_oscillator_with_kernel, GmmaOscillatorBatchBuilder, GmmaOscillatorBatchOutput,
    GmmaOscillatorBatchRange, GmmaOscillatorBuilder, GmmaOscillatorData, GmmaOscillatorError,
    GmmaOscillatorInput, GmmaOscillatorOutput, GmmaOscillatorParams, GmmaOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use gmma_oscillator::{
    gmma_oscillator_alloc, gmma_oscillator_batch_into, gmma_oscillator_batch_js,
    gmma_oscillator_free, gmma_oscillator_into, gmma_oscillator_js,
};
#[cfg(feature = "python")]
pub use gmma_oscillator::{
    gmma_oscillator_batch_py, gmma_oscillator_py, register_gmma_oscillator_module,
    GmmaOscillatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use goertzel_cycle_composite_wave::goertzel_cycle_composite_wave_into;
pub use goertzel_cycle_composite_wave::{
    expand_grid_goertzel_cycle_composite_wave, goertzel_cycle_composite_wave,
    goertzel_cycle_composite_wave_batch_par_slice, goertzel_cycle_composite_wave_batch_slice,
    goertzel_cycle_composite_wave_batch_with_kernel, goertzel_cycle_composite_wave_into_slice,
    goertzel_cycle_composite_wave_with_kernel, GoertzelCycleCompositeWaveBatchBuilder,
    GoertzelCycleCompositeWaveBatchOutput, GoertzelCycleCompositeWaveBatchRange,
    GoertzelCycleCompositeWaveBuilder, GoertzelCycleCompositeWaveData,
    GoertzelCycleCompositeWaveError, GoertzelCycleCompositeWaveInput,
    GoertzelCycleCompositeWaveOutput, GoertzelCycleCompositeWaveParams,
    GoertzelCycleCompositeWaveStream, GoertzelDetrendMode,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use goertzel_cycle_composite_wave::{
    goertzel_cycle_composite_wave_alloc, goertzel_cycle_composite_wave_batch_into,
    goertzel_cycle_composite_wave_batch_js, goertzel_cycle_composite_wave_free,
    goertzel_cycle_composite_wave_into, goertzel_cycle_composite_wave_js,
};
#[cfg(feature = "python")]
pub use goertzel_cycle_composite_wave::{
    goertzel_cycle_composite_wave_batch_py, goertzel_cycle_composite_wave_py,
    register_goertzel_cycle_composite_wave_module, GoertzelCycleCompositeWaveStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use historical_volatility_rank::historical_volatility_rank_into;
pub use historical_volatility_rank::{
    expand_grid_historical_volatility_rank, historical_volatility_rank,
    historical_volatility_rank_batch_par_slice, historical_volatility_rank_batch_slice,
    historical_volatility_rank_batch_with_kernel, historical_volatility_rank_into_slice,
    historical_volatility_rank_with_kernel, HistoricalVolatilityRankBatchBuilder,
    HistoricalVolatilityRankBatchOutput, HistoricalVolatilityRankBatchRange,
    HistoricalVolatilityRankBuilder, HistoricalVolatilityRankData, HistoricalVolatilityRankError,
    HistoricalVolatilityRankInput, HistoricalVolatilityRankOutput, HistoricalVolatilityRankParams,
    HistoricalVolatilityRankStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use historical_volatility_rank::{
    historical_volatility_rank_alloc, historical_volatility_rank_batch_into,
    historical_volatility_rank_batch_js, historical_volatility_rank_free,
    historical_volatility_rank_into, historical_volatility_rank_js,
};
#[cfg(feature = "python")]
pub use historical_volatility_rank::{
    historical_volatility_rank_batch_py, historical_volatility_rank_py,
    HistoricalVolatilityRankStreamPy,
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
pub use kairi_relative_index::kairi_relative_index_into;
pub use kairi_relative_index::{
    expand_grid_kairi_relative_index, kairi_relative_index, kairi_relative_index_batch_par_slice,
    kairi_relative_index_batch_slice, kairi_relative_index_batch_with_kernel,
    kairi_relative_index_into_slice, kairi_relative_index_with_kernel,
    KairiRelativeIndexBatchBuilder, KairiRelativeIndexBatchOutput, KairiRelativeIndexBatchRange,
    KairiRelativeIndexBuilder, KairiRelativeIndexData, KairiRelativeIndexError,
    KairiRelativeIndexInput, KairiRelativeIndexOutput, KairiRelativeIndexParams,
    KairiRelativeIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use kairi_relative_index::{
    kairi_relative_index_alloc, kairi_relative_index_batch_into, kairi_relative_index_batch_js,
    kairi_relative_index_free, kairi_relative_index_into, kairi_relative_index_js,
};
#[cfg(feature = "python")]
pub use kairi_relative_index::{
    kairi_relative_index_batch_py, kairi_relative_index_py, KairiRelativeIndexStreamPy,
};
pub use linearreg_angle::{
    linearreg_angle, Linearreg_angleInput, Linearreg_angleOutput, Linearreg_angleParams,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use market_structure_trailing_stop::market_structure_trailing_stop_into;
pub use market_structure_trailing_stop::{
    expand_grid_market_structure_trailing_stop, market_structure_trailing_stop,
    market_structure_trailing_stop_batch_par_slice, market_structure_trailing_stop_batch_slice,
    market_structure_trailing_stop_batch_with_kernel, market_structure_trailing_stop_into_slice,
    market_structure_trailing_stop_with_kernel, MarketStructureTrailingStopBatchBuilder,
    MarketStructureTrailingStopBatchOutput, MarketStructureTrailingStopBatchRange,
    MarketStructureTrailingStopBuilder, MarketStructureTrailingStopData,
    MarketStructureTrailingStopError, MarketStructureTrailingStopInput,
    MarketStructureTrailingStopOutput, MarketStructureTrailingStopParams,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use market_structure_trailing_stop::{
    market_structure_trailing_stop_alloc, market_structure_trailing_stop_batch_into,
    market_structure_trailing_stop_batch_js, market_structure_trailing_stop_free,
    market_structure_trailing_stop_into, market_structure_trailing_stop_js,
};
#[cfg(feature = "python")]
pub use market_structure_trailing_stop::{
    market_structure_trailing_stop_batch_py, market_structure_trailing_stop_py,
    register_market_structure_trailing_stop_module,
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
    ema, epma, frama, fwma, gaussian, highpass, highpass_2_pole, hma, hwma, jma, jsa, kama, linreg,
    maaq, mama, mwdx, nma, pwma, reflex, sinwma, sma, smma, sqwma, srwma, supersmoother,
    supersmoother_3_pole, swma, tema, tilson, tradjema, trendflex, trima, uma,
    volatility_adjusted_ma, volume_adjusted_ma, vpwma, vwap, vwma, wilders, wma, zlema,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use nonlinear_regression_zero_lag_moving_average::nonlinear_regression_zero_lag_moving_average_into;
pub use nonlinear_regression_zero_lag_moving_average::{
    expand_grid_nonlinear_regression_zero_lag_moving_average,
    nonlinear_regression_zero_lag_moving_average,
    nonlinear_regression_zero_lag_moving_average_batch_par_slice,
    nonlinear_regression_zero_lag_moving_average_batch_slice,
    nonlinear_regression_zero_lag_moving_average_batch_with_kernel,
    nonlinear_regression_zero_lag_moving_average_into_slice,
    nonlinear_regression_zero_lag_moving_average_with_kernel,
    NonlinearRegressionZeroLagMovingAverageBatchBuilder,
    NonlinearRegressionZeroLagMovingAverageBatchOutput,
    NonlinearRegressionZeroLagMovingAverageBatchRange,
    NonlinearRegressionZeroLagMovingAverageBuilder, NonlinearRegressionZeroLagMovingAverageData,
    NonlinearRegressionZeroLagMovingAverageError, NonlinearRegressionZeroLagMovingAverageInput,
    NonlinearRegressionZeroLagMovingAverageOutput, NonlinearRegressionZeroLagMovingAverageParams,
    NonlinearRegressionZeroLagMovingAverageStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use nonlinear_regression_zero_lag_moving_average::{
    nonlinear_regression_zero_lag_moving_average_alloc,
    nonlinear_regression_zero_lag_moving_average_batch_into,
    nonlinear_regression_zero_lag_moving_average_batch_js,
    nonlinear_regression_zero_lag_moving_average_free,
    nonlinear_regression_zero_lag_moving_average_into,
    nonlinear_regression_zero_lag_moving_average_js,
};
#[cfg(feature = "python")]
pub use nonlinear_regression_zero_lag_moving_average::{
    nonlinear_regression_zero_lag_moving_average_batch_py,
    nonlinear_regression_zero_lag_moving_average_py,
    register_nonlinear_regression_zero_lag_moving_average_module,
    NonlinearRegressionZeroLagMovingAverageStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use possible_rsi::possible_rsi_into;
pub use possible_rsi::{
    expand_grid_possible_rsi, possible_rsi, possible_rsi_batch_par_slice, possible_rsi_batch_slice,
    possible_rsi_batch_with_kernel, possible_rsi_into_slice, possible_rsi_with_kernel,
    PossibleRsiBatchBuilder, PossibleRsiBatchOutput, PossibleRsiBatchRange, PossibleRsiBuilder,
    PossibleRsiData, PossibleRsiError, PossibleRsiInput, PossibleRsiOutput, PossibleRsiParams,
    PossibleRsiStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use possible_rsi::{
    possible_rsi_alloc, possible_rsi_batch_into, possible_rsi_batch_js, possible_rsi_free,
    possible_rsi_into, possible_rsi_js,
};
#[cfg(feature = "python")]
pub use possible_rsi::{
    possible_rsi_batch_py, possible_rsi_py, register_possible_rsi_module, PossibleRsiStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use projection_oscillator::projection_oscillator_into;
pub use projection_oscillator::{
    expand_grid_projection_oscillator, projection_oscillator,
    projection_oscillator_batch_par_slice, projection_oscillator_batch_slice,
    projection_oscillator_batch_with_kernel, projection_oscillator_into_slice,
    projection_oscillator_with_kernel, ProjectionOscillatorBatchBuilder,
    ProjectionOscillatorBatchOutput, ProjectionOscillatorBatchRange, ProjectionOscillatorBuilder,
    ProjectionOscillatorData, ProjectionOscillatorError, ProjectionOscillatorInput,
    ProjectionOscillatorOutput, ProjectionOscillatorParams, ProjectionOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use projection_oscillator::{
    projection_oscillator_alloc, projection_oscillator_batch_into, projection_oscillator_batch_js,
    projection_oscillator_free, projection_oscillator_into, projection_oscillator_js,
};
#[cfg(feature = "python")]
pub use projection_oscillator::{
    projection_oscillator_batch_py, projection_oscillator_py, ProjectionOscillatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use rogers_satchell_volatility::rogers_satchell_volatility_into;
pub use rogers_satchell_volatility::{
    rogers_satchell_volatility, rogers_satchell_volatility_batch_par_slice,
    rogers_satchell_volatility_batch_slice, rogers_satchell_volatility_batch_with_kernel,
    rogers_satchell_volatility_into_slice, rogers_satchell_volatility_with_kernel,
    RogersSatchellVolatilityBatchBuilder, RogersSatchellVolatilityBatchOutput,
    RogersSatchellVolatilityBatchRange, RogersSatchellVolatilityBuilder,
    RogersSatchellVolatilityData, RogersSatchellVolatilityError, RogersSatchellVolatilityInput,
    RogersSatchellVolatilityOutput, RogersSatchellVolatilityParams, RogersSatchellVolatilityStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use rogers_satchell_volatility::{
    rogers_satchell_volatility_alloc, rogers_satchell_volatility_batch_into,
    rogers_satchell_volatility_batch_js, rogers_satchell_volatility_free,
    rogers_satchell_volatility_into, rogers_satchell_volatility_js,
};
#[cfg(feature = "python")]
pub use rogers_satchell_volatility::{
    rogers_satchell_volatility_batch_py, rogers_satchell_volatility_py,
    RogersSatchellVolatilityStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use rolling_skewness_kurtosis::rolling_skewness_kurtosis_into;
pub use rolling_skewness_kurtosis::{
    expand_grid_rolling_skewness_kurtosis, rolling_skewness_kurtosis,
    rolling_skewness_kurtosis_batch_par_slice, rolling_skewness_kurtosis_batch_slice,
    rolling_skewness_kurtosis_batch_with_kernel, rolling_skewness_kurtosis_into_slice,
    rolling_skewness_kurtosis_with_kernel, RollingSkewnessKurtosisBatchBuilder,
    RollingSkewnessKurtosisBatchOutput, RollingSkewnessKurtosisBatchRange,
    RollingSkewnessKurtosisBuilder, RollingSkewnessKurtosisData, RollingSkewnessKurtosisError,
    RollingSkewnessKurtosisInput, RollingSkewnessKurtosisOutput, RollingSkewnessKurtosisParams,
    RollingSkewnessKurtosisStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use rolling_skewness_kurtosis::{
    rolling_skewness_kurtosis_alloc, rolling_skewness_kurtosis_batch_into,
    rolling_skewness_kurtosis_batch_js, rolling_skewness_kurtosis_free,
    rolling_skewness_kurtosis_into, rolling_skewness_kurtosis_js,
};
#[cfg(feature = "python")]
pub use rolling_skewness_kurtosis::{
    rolling_skewness_kurtosis_batch_py, rolling_skewness_kurtosis_py,
    RollingSkewnessKurtosisStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use rolling_z_score_trend::rolling_z_score_trend_into;
pub use rolling_z_score_trend::{
    expand_grid_rolling_z_score_trend, rolling_z_score_trend,
    rolling_z_score_trend_batch_par_slice, rolling_z_score_trend_batch_slice,
    rolling_z_score_trend_batch_with_kernel, rolling_z_score_trend_into_slice,
    rolling_z_score_trend_with_kernel, RollingZScoreTrendBatchBuilder,
    RollingZScoreTrendBatchOutput, RollingZScoreTrendBatchRange, RollingZScoreTrendBuilder,
    RollingZScoreTrendData, RollingZScoreTrendError, RollingZScoreTrendInput,
    RollingZScoreTrendOutput, RollingZScoreTrendParams, RollingZScoreTrendStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use rolling_z_score_trend::{
    rolling_z_score_trend_alloc, rolling_z_score_trend_batch_into, rolling_z_score_trend_batch_js,
    rolling_z_score_trend_free, rolling_z_score_trend_into, rolling_z_score_trend_js,
};
#[cfg(feature = "python")]
pub use rolling_z_score_trend::{
    rolling_z_score_trend_batch_py, rolling_z_score_trend_py, RollingZScoreTrendStreamPy,
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
pub use stochastic_money_flow_index::stochastic_money_flow_index_into;
pub use stochastic_money_flow_index::{
    expand_grid_stochastic_money_flow_index, stochastic_money_flow_index,
    stochastic_money_flow_index_batch_par_slice, stochastic_money_flow_index_batch_slice,
    stochastic_money_flow_index_batch_with_kernel, stochastic_money_flow_index_into_slice,
    stochastic_money_flow_index_with_kernel, StochasticMoneyFlowIndexBatchBuilder,
    StochasticMoneyFlowIndexBatchOutput, StochasticMoneyFlowIndexBatchRange,
    StochasticMoneyFlowIndexBuilder, StochasticMoneyFlowIndexData, StochasticMoneyFlowIndexError,
    StochasticMoneyFlowIndexInput, StochasticMoneyFlowIndexOutput, StochasticMoneyFlowIndexParams,
    StochasticMoneyFlowIndexStream,
};
#[cfg(feature = "python")]
pub use stochastic_money_flow_index::{
    register_stochastic_money_flow_index_module, stochastic_money_flow_index_batch_py,
    stochastic_money_flow_index_py, StochasticMoneyFlowIndexStreamPy,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use stochastic_money_flow_index::{
    stochastic_money_flow_index_alloc, stochastic_money_flow_index_batch_into,
    stochastic_money_flow_index_batch_js, stochastic_money_flow_index_free,
    stochastic_money_flow_index_into, stochastic_money_flow_index_js,
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
    trend_direction_force_index_into, trend_direction_force_index_js,
};
#[cfg(feature = "python")]
pub use trend_direction_force_index::{
    trend_direction_force_index_batch_py, trend_direction_force_index_py,
    TrendDirectionForceIndexStreamPy,
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
pub use velocity_acceleration_convergence_divergence_indicator::velocity_acceleration_convergence_divergence_indicator_into;
pub use velocity_acceleration_convergence_divergence_indicator::{
    expand_grid_velocity_acceleration_convergence_divergence_indicator,
    velocity_acceleration_convergence_divergence_indicator,
    velocity_acceleration_convergence_divergence_indicator_batch_par_slice,
    velocity_acceleration_convergence_divergence_indicator_batch_slice,
    velocity_acceleration_convergence_divergence_indicator_batch_with_kernel,
    velocity_acceleration_convergence_divergence_indicator_into_slice,
    velocity_acceleration_convergence_divergence_indicator_with_kernel,
    VelocityAccelerationConvergenceDivergenceIndicatorBatchBuilder,
    VelocityAccelerationConvergenceDivergenceIndicatorBatchOutput,
    VelocityAccelerationConvergenceDivergenceIndicatorBatchRange,
    VelocityAccelerationConvergenceDivergenceIndicatorBuilder,
    VelocityAccelerationConvergenceDivergenceIndicatorData,
    VelocityAccelerationConvergenceDivergenceIndicatorError,
    VelocityAccelerationConvergenceDivergenceIndicatorInput,
    VelocityAccelerationConvergenceDivergenceIndicatorOutput,
    VelocityAccelerationConvergenceDivergenceIndicatorParams,
    VelocityAccelerationConvergenceDivergenceIndicatorStream,
};
#[cfg(feature = "python")]
pub use velocity_acceleration_convergence_divergence_indicator::{
    register_velocity_acceleration_convergence_divergence_indicator_module,
    velocity_acceleration_convergence_divergence_indicator_batch_py,
    velocity_acceleration_convergence_divergence_indicator_py,
    VelocityAccelerationConvergenceDivergenceIndicatorStreamPy,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use velocity_acceleration_convergence_divergence_indicator::{
    velocity_acceleration_convergence_divergence_indicator_alloc,
    velocity_acceleration_convergence_divergence_indicator_batch_into,
    velocity_acceleration_convergence_divergence_indicator_batch_js,
    velocity_acceleration_convergence_divergence_indicator_free,
    velocity_acceleration_convergence_divergence_indicator_into,
    velocity_acceleration_convergence_divergence_indicator_js,
};
pub use vidya::{
    vidya, VidyaBatchBuilder, VidyaBatchOutput, VidyaBatchRange, VidyaBuilder, VidyaData,
    VidyaError, VidyaInput, VidyaOutput, VidyaParams, VidyaStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use vidya::{vidya_alloc, vidya_batch_into, vidya_batch_js, vidya_free, vidya_into, vidya_js};
#[cfg(feature = "python")]
pub use vidya::{vidya_batch_py, vidya_py, VidyaStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use volume_weighted_rsi::volume_weighted_rsi_into;
pub use volume_weighted_rsi::{
    expand_grid_volume_weighted_rsi, volume_weighted_rsi, volume_weighted_rsi_batch_par_slice,
    volume_weighted_rsi_batch_slice, volume_weighted_rsi_batch_with_kernel,
    volume_weighted_rsi_into_slice, volume_weighted_rsi_with_kernel, VolumeWeightedRsiBatchBuilder,
    VolumeWeightedRsiBatchOutput, VolumeWeightedRsiBatchRange, VolumeWeightedRsiBuilder,
    VolumeWeightedRsiData, VolumeWeightedRsiError, VolumeWeightedRsiInput, VolumeWeightedRsiOutput,
    VolumeWeightedRsiParams, VolumeWeightedRsiStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use volume_weighted_rsi::{
    volume_weighted_rsi_alloc, volume_weighted_rsi_batch_into, volume_weighted_rsi_batch_js,
    volume_weighted_rsi_free, volume_weighted_rsi_into, volume_weighted_rsi_js,
};
#[cfg(feature = "python")]
pub use volume_weighted_rsi::{
    volume_weighted_rsi_batch_py, volume_weighted_rsi_py, VolumeWeightedRsiStreamPy,
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
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use zig_zag_channels::zig_zag_channels_into;
pub use zig_zag_channels::{
    expand_grid_zig_zag_channels, zig_zag_channels, zig_zag_channels_batch_par_slice,
    zig_zag_channels_batch_slice, zig_zag_channels_batch_with_kernel, zig_zag_channels_into_slice,
    zig_zag_channels_with_kernel, ZigZagChannelsBatchBuilder, ZigZagChannelsBatchOutput,
    ZigZagChannelsBatchRange, ZigZagChannelsBuilder, ZigZagChannelsData, ZigZagChannelsError,
    ZigZagChannelsInput, ZigZagChannelsOutput, ZigZagChannelsParams,
};
#[cfg(feature = "python")]
pub use zig_zag_channels::{
    register_zig_zag_channels_module, zig_zag_channels_batch_py, zig_zag_channels_py,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use zig_zag_channels::{
    zig_zag_channels_alloc, zig_zag_channels_batch_into, zig_zag_channels_batch_js,
    zig_zag_channels_free, zig_zag_channels_into, zig_zag_channels_js,
};
