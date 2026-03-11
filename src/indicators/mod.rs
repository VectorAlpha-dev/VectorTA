pub mod acosc;
pub mod ad;
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
pub mod ao;
pub mod apo;
pub mod aroon;
pub mod aroonosc;
pub mod aso;
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
pub mod ehlers_adaptive_cg;
pub mod emd;
pub mod emv;
pub mod er;
pub mod eri;
pub mod exponential_trend;
pub mod fisher;
pub mod fosc;
pub mod fvg_trailing_stop;
pub mod trend_flow_trail;
pub use fvg_trailing_stop::{
    fvg_trailing_stop, FvgTrailingStopInput, FvgTrailingStopOutput, FvgTrailingStopParams,
};
pub mod gatorosc;
pub mod geometric_bias_oscillator;
pub mod halftrend;
pub mod vdubus_divergence_wave_pattern_generator;
pub use halftrend::{halftrend, HalfTrendInput, HalfTrendOutput, HalfTrendParams};
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
pub mod linear_correlation_oscillator;
pub mod linearreg_angle;
pub mod linearreg_intercept;
pub mod linearreg_slope;
pub mod lpc;
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
pub mod pattern_recognition;
pub mod percentile_nearest_rank;
pub mod pfe;
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
pub use ppo::{ppo, PpoInput, PpoOutput, PpoParams};
pub mod prb;
pub use prb::{
    prb, prb_batch_par_slice, prb_batch_slice, prb_batch_with_kernel, prb_with_kernel,
    PrbBatchBuilder, PrbBatchOutput, PrbBatchRange, PrbBuilder, PrbInput, PrbOutput, PrbParams,
    PrbStream,
};
pub mod pvi;
pub mod qqe;
pub mod qstick;
pub mod range_breakout_signals;
pub mod range_filter;
pub mod registry;
pub use range_filter::{
    range_filter, range_filter_batch_par_slice, range_filter_batch_slice, range_filter_into_slice,
    range_filter_with_kernel, RangeFilterBatchBuilder, RangeFilterBatchOutput,
    RangeFilterBatchRange, RangeFilterBuilder, RangeFilterData, RangeFilterError, RangeFilterInput,
    RangeFilterOutput, RangeFilterParams, RangeFilterStream,
};
pub mod roc;
pub use roc::{
    roc, RocBatchBuilder, RocBatchOutput, RocBatchRange, RocBuilder, RocError, RocInput, RocOutput,
    RocParams, RocStream,
};
pub mod reverse_rsi;
pub mod rocp;
pub mod rocr;
pub use reverse_rsi::{reverse_rsi, ReverseRsiInput, ReverseRsiOutput, ReverseRsiParams};
pub mod rsi;
pub mod rsmk;
pub mod rsx;
pub use rsx::{
    rsx, RsxBatchOutput, RsxBatchRange, RsxBuilder, RsxInput, RsxOutput, RsxParams, RsxStream,
};
pub mod rvi;
pub mod safezonestop;
pub mod sar;
pub mod squeeze_momentum;
pub mod srsi;
pub mod stc;
pub mod stddev;
pub use stddev::{stddev, StdDevInput, StdDevOutput, StdDevParams};
pub mod stoch;
pub mod stochf;
pub mod supertrend;
pub mod trix;
pub mod tsf;
pub mod tsi;
pub mod ttm_squeeze;
pub mod ttm_trend;
pub mod ui;
pub mod ultosc;
pub mod utility_functions;
pub mod var;
pub mod velocity;
pub mod vi;
pub mod vidya;
pub mod vlma;
pub mod vosc;
pub mod voss;
pub mod vpci;
pub mod vpt;
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

pub use apo::{apo, ApoInput, ApoOutput, ApoParams};
pub use cci::{cci, CciInput, CciOutput, CciParams};
pub use cfo::{cfo, CfoInput, CfoOutput, CfoParams};
pub use coppock::{coppock, CoppockInput, CoppockOutput, CoppockParams};
pub use er::{er, ErInput, ErOutput, ErParams};
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
pub use linearreg_angle::{
    linearreg_angle, Linearreg_angleInput, Linearreg_angleOutput, Linearreg_angleParams,
};
pub use mean_ad::{mean_ad, MeanAdInput, MeanAdOutput, MeanAdParams};
pub use mom::{mom, MomInput, MomOutput, MomParams};
pub use moving_averages::{
    alma, buff_averages, cwma, dema, edcf, ehlers_itrend, ehlers_pma,
    ehlers_undersampled_double_moving_average, elastic_volume_weighted_moving_average, ema, epma,
    frama, fwma, gaussian, highpass, highpass_2_pole, hma, hwma, jma, jsa, kama, linreg, maaq,
    mama, mwdx, nma, pwma, reflex, sinwma, sma, smma, sqwma, srwma, supersmoother,
    supersmoother_3_pole, swma, tema, tilson, tradjema, trendflex, trima, uma,
    volatility_adjusted_ma, volume_adjusted_ma, vpwma, vwap, vwma, wilders, wma, zlema,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use normalized_volume_true_range::normalized_volume_true_range_into;
pub use normalized_volume_true_range::{
    expand_grid_normalized_volume_true_range, normalized_volume_true_range,
    normalized_volume_true_range_batch_par_slice, normalized_volume_true_range_batch_slice,
    normalized_volume_true_range_batch_with_kernel, normalized_volume_true_range_into_slice,
    normalized_volume_true_range_with_kernel, NormalizedVolumeTrueRangeBatchBuilder,
    NormalizedVolumeTrueRangeBatchOutput, NormalizedVolumeTrueRangeBatchRange,
    NormalizedVolumeTrueRangeBuilder, NormalizedVolumeTrueRangeData,
    NormalizedVolumeTrueRangeError, NormalizedVolumeTrueRangeInput,
    NormalizedVolumeTrueRangeOutput, NormalizedVolumeTrueRangeParams,
    NormalizedVolumeTrueRangeStream, NormalizedVolumeTrueRangeStyle,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use normalized_volume_true_range::{
    normalized_volume_true_range_alloc, normalized_volume_true_range_batch_into,
    normalized_volume_true_range_batch_unified_js as normalized_volume_true_range_batch,
    normalized_volume_true_range_free, normalized_volume_true_range_into,
    normalized_volume_true_range_js, NormalizedVolumeTrueRangeStreamWasm,
};
#[cfg(feature = "python")]
pub use normalized_volume_true_range::{
    normalized_volume_true_range_batch_py, normalized_volume_true_range_py,
    NormalizedVolumeTrueRangeStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use range_breakout_signals::range_breakout_signals_into;
pub use range_breakout_signals::{
    expand_grid_range_breakout_signals, range_breakout_signals,
    range_breakout_signals_batch_par_slice, range_breakout_signals_batch_slice,
    range_breakout_signals_batch_with_kernel, range_breakout_signals_into_slice,
    range_breakout_signals_with_kernel, RangeBreakoutSignalsBatchBuilder,
    RangeBreakoutSignalsBatchOutput, RangeBreakoutSignalsBatchRange, RangeBreakoutSignalsBuilder,
    RangeBreakoutSignalsData, RangeBreakoutSignalsError, RangeBreakoutSignalsInput,
    RangeBreakoutSignalsOutput, RangeBreakoutSignalsParams, RangeBreakoutSignalsStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use range_breakout_signals::{
    range_breakout_signals_alloc, range_breakout_signals_batch_into,
    range_breakout_signals_batch_unified_js as range_breakout_signals_batch,
    range_breakout_signals_free, range_breakout_signals_into, range_breakout_signals_js,
    RangeBreakoutSignalsStreamWasm,
};
#[cfg(feature = "python")]
pub use range_breakout_signals::{
    range_breakout_signals_batch_py, range_breakout_signals_py, RangeBreakoutSignalsStreamPy,
};
pub use rsi::{rsi, RsiBatchOutput, RsiInput, RsiOutput, RsiParams, RsiStream};
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
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use vpci::{
    vpci_alloc, vpci_batch_into, vpci_batch_unified_js, vpci_free, vpci_into, vpci_js, VpciContext,
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
