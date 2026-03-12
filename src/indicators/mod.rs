pub mod absolute_strength_index_oscillator;
pub mod accumulation_swing_index;
pub mod acosc;
pub mod ad;
pub mod adaptive_bandpass_trigger_oscillator;
pub mod adaptive_bounds_rsi;
pub mod adaptive_macd;
pub mod adaptive_momentum_oscillator;
pub mod adjustable_ma_alternating_extremities;
pub mod adosc;
pub mod advance_decline_line;
pub mod adx;
pub mod adxr;
pub mod alligator;
pub mod alphatrend;
pub mod dispatch;
pub mod ehlers_fm_demodulator;
pub mod evasive_supertrend;
pub mod ewma_volatility;
pub mod exponential_trend;
pub mod geometric_bias_oscillator;
pub mod goertzel_cycle_composite_wave;
pub mod ichimoku_oscillator;
pub mod l1_ehlers_phasor;
pub mod market_structure_confluence;
pub mod pretty_good_oscillator;
pub mod price_density_market_noise;
pub mod psychological_line;
pub mod range_filtered_trend_signals;
pub mod range_oscillator;
pub mod rank_correlation_index;
pub mod smoothed_gaussian_trend_filter;
pub mod spearman_correlation;
pub mod standardized_psar_oscillator;
pub mod statistical_trailing_stop;
pub mod stochastic_adaptive_d;
pub mod stochastic_connors_rsi;
pub mod stochastic_money_flow_index;
pub mod supertrend_recovery;
pub mod trend_continuation_factor;
pub mod trend_flow_trail;
pub mod trend_follower;
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
pub mod atr_percentile;
pub mod avsl;
pub mod bull_power_vs_bear_power;
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
pub mod decisionpoint_breadth_swenlin_trading_oscillator;
pub mod demand_index;
pub mod devstop;
pub mod didi_index;
pub mod ehlers_autocorrelation_periodogram;
pub mod ehlers_linear_extrapolation_predictor;
pub mod velocity_acceleration_indicator;
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
pub mod ehlers_adaptive_cg;
pub mod ehlers_adaptive_cyber_cycle;
pub mod ehlers_simple_cycle_indicator;
pub mod ehlers_smoothed_adaptive_momentum;
pub mod emd;
pub mod emd_trend;
pub mod emv;
pub mod er;
pub mod eri;
pub mod fibonacci_entry_bands;
pub mod fibonacci_trailing_stop;
pub mod fisher;
pub mod forward_backward_exponential_oscillator;
pub mod fosc;
pub mod fvg_positioning_average;
pub mod fvg_trailing_stop;
pub mod garman_klass_volatility;
pub mod gopalakrishnan_range_index;
pub mod grover_llorens_cycle_oscillator;
pub mod historical_volatility;
pub mod hull_butterfly_oscillator;
pub mod intraday_momentum_index;
pub mod kase_peak_oscillator_with_divergences;
pub mod neighboring_trailing_stop;
pub mod vertical_horizontal_filter;
pub mod volume_energy_reservoirs;
pub mod vwap_zscore_with_signals;
pub use fibonacci_entry_bands::{
    fibonacci_entry_bands, FibonacciEntryBandsInput, FibonacciEntryBandsOutput,
    FibonacciEntryBandsParams,
};
pub use fibonacci_trailing_stop::{
    fibonacci_trailing_stop, FibonacciTrailingStopInput, FibonacciTrailingStopOutput,
    FibonacciTrailingStopParams,
};
pub use fvg_trailing_stop::{
    fvg_trailing_stop, FvgTrailingStopInput, FvgTrailingStopOutput, FvgTrailingStopParams,
};
pub mod gatorosc;
pub mod half_causal_estimator;
pub mod halftrend;
pub mod vdubus_divergence_wave_pattern_generator;
pub use halftrend::{halftrend, HalfTrendInput, HalfTrendOutput, HalfTrendParams};
pub mod hema_trend_levels;
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
pub mod macd_wave_signal_pro;
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
pub mod monotonicity_index;
pub mod moving_averages;
pub use moving_averages::ehlers_kama::{
    ehlers_kama, EhlersKamaInput, EhlersKamaOutput, EhlersKamaParams,
};
pub mod msw;
pub mod multi_length_stochastic_average;
pub mod nadaraya_watson_envelope;
pub mod natr;
pub mod net_myrsi;
pub mod normalized_volume_true_range;
pub use net_myrsi::{net_myrsi, NetMyrsiInput, NetMyrsiOutput, NetMyrsiParams};
pub mod normalized_resonator;
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
pub mod premier_rsi_oscillator;
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
pub mod registry;
pub mod reversal_signals;
pub mod volume_weighted_relative_strength_index;
pub use market_structure_confluence::{
    market_structure_confluence, market_structure_confluence_batch_with_kernel,
    market_structure_confluence_into, market_structure_confluence_into_slices,
    market_structure_confluence_with_kernel, MarketStructureConfluenceBatchBuilder,
    MarketStructureConfluenceBatchOutput, MarketStructureConfluenceBatchRange,
    MarketStructureConfluenceBosConfirmation, MarketStructureConfluenceBuilder,
    MarketStructureConfluenceData, MarketStructureConfluenceError, MarketStructureConfluenceInput,
    MarketStructureConfluenceOutput, MarketStructureConfluenceParams,
    MarketStructureConfluenceStream,
};
pub use range_filter::{
    range_filter, range_filter_batch_par_slice, range_filter_batch_slice, range_filter_into_slice,
    range_filter_with_kernel, RangeFilterBatchBuilder, RangeFilterBatchOutput,
    RangeFilterBatchRange, RangeFilterBuilder, RangeFilterData, RangeFilterError, RangeFilterInput,
    RangeFilterOutput, RangeFilterParams, RangeFilterStream,
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
pub mod squeeze_index;
pub mod squeeze_momentum;
pub mod srsi;
pub mod stc;
pub mod stddev;
pub use stddev::{stddev, StdDevInput, StdDevOutput, StdDevParams};
pub mod smooth_theil_sen;
pub mod stoch;
pub mod stochastic_distance;
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
pub mod volatility_quality_index;
pub mod volume_weighted_stochastic_rsi;
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

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use advance_decline_line::advance_decline_line_into;
pub use advance_decline_line::{
    advance_decline_line, advance_decline_line_batch_inner_into,
    advance_decline_line_batch_par_slice, advance_decline_line_batch_slice,
    advance_decline_line_batch_with_kernel, advance_decline_line_into_slice,
    advance_decline_line_with_kernel, AdvanceDeclineLineBatchBuilder,
    AdvanceDeclineLineBatchOutput, AdvanceDeclineLineBatchRange, AdvanceDeclineLineBuilder,
    AdvanceDeclineLineData, AdvanceDeclineLineError, AdvanceDeclineLineInput,
    AdvanceDeclineLineOutput, AdvanceDeclineLineParams, AdvanceDeclineLineStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use advance_decline_line::{
    advance_decline_line_alloc, advance_decline_line_batch_into, advance_decline_line_batch_js,
    advance_decline_line_free, advance_decline_line_into, advance_decline_line_js,
};
#[cfg(feature = "python")]
pub use advance_decline_line::{
    advance_decline_line_batch_py, advance_decline_line_py, AdvanceDeclineLineStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use atr_percentile::atr_percentile_into;
pub use atr_percentile::{
    atr_percentile, atr_percentile_batch_inner_into, atr_percentile_batch_par_slice,
    atr_percentile_batch_slice, atr_percentile_batch_with_kernel, atr_percentile_into_slice,
    atr_percentile_with_kernel, AtrPercentileBatchBuilder, AtrPercentileBatchOutput,
    AtrPercentileBatchRange, AtrPercentileBuilder, AtrPercentileData, AtrPercentileError,
    AtrPercentileInput, AtrPercentileOutput, AtrPercentileParams, AtrPercentileStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use atr_percentile::{
    atr_percentile_alloc, atr_percentile_batch_into, atr_percentile_batch_js, atr_percentile_free,
    atr_percentile_into, atr_percentile_js,
};
#[cfg(feature = "python")]
pub use atr_percentile::{atr_percentile_batch_py, atr_percentile_py, AtrPercentileStreamPy};
#[cfg(feature = "python")]
pub use avsl::{avsl_batch_py, avsl_py, AvslStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use bull_power_vs_bear_power::bull_power_vs_bear_power_into;
pub use bull_power_vs_bear_power::{
    bull_power_vs_bear_power, bull_power_vs_bear_power_batch_inner_into,
    bull_power_vs_bear_power_batch_par_slice, bull_power_vs_bear_power_batch_slice,
    bull_power_vs_bear_power_batch_with_kernel, bull_power_vs_bear_power_into_slice,
    bull_power_vs_bear_power_with_kernel, BullPowerVsBearPowerBatchBuilder,
    BullPowerVsBearPowerBatchOutput, BullPowerVsBearPowerBatchRange, BullPowerVsBearPowerBuilder,
    BullPowerVsBearPowerData, BullPowerVsBearPowerError, BullPowerVsBearPowerInput,
    BullPowerVsBearPowerOutput, BullPowerVsBearPowerParams, BullPowerVsBearPowerStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use bull_power_vs_bear_power::{
    bull_power_vs_bear_power_alloc, bull_power_vs_bear_power_batch_into,
    bull_power_vs_bear_power_batch_js, bull_power_vs_bear_power_free,
    bull_power_vs_bear_power_into, bull_power_vs_bear_power_js,
};
#[cfg(feature = "python")]
pub use bull_power_vs_bear_power::{
    bull_power_vs_bear_power_batch_py, bull_power_vs_bear_power_py, BullPowerVsBearPowerStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use decisionpoint_breadth_swenlin_trading_oscillator::decisionpoint_breadth_swenlin_trading_oscillator_into;
pub use decisionpoint_breadth_swenlin_trading_oscillator::{
    decisionpoint_breadth_swenlin_trading_oscillator,
    decisionpoint_breadth_swenlin_trading_oscillator_batch_inner,
    decisionpoint_breadth_swenlin_trading_oscillator_batch_inner_into,
    decisionpoint_breadth_swenlin_trading_oscillator_batch_par_slices,
    decisionpoint_breadth_swenlin_trading_oscillator_batch_slice,
    decisionpoint_breadth_swenlin_trading_oscillator_batch_with_kernel,
    decisionpoint_breadth_swenlin_trading_oscillator_into_slice,
    decisionpoint_breadth_swenlin_trading_oscillator_with_kernel,
    DecisionPointBreadthSwenlinTradingOscillatorBatchBuilder,
    DecisionPointBreadthSwenlinTradingOscillatorBatchOutput,
    DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
    DecisionPointBreadthSwenlinTradingOscillatorBuilder,
    DecisionPointBreadthSwenlinTradingOscillatorData,
    DecisionPointBreadthSwenlinTradingOscillatorError,
    DecisionPointBreadthSwenlinTradingOscillatorInput,
    DecisionPointBreadthSwenlinTradingOscillatorOutput,
    DecisionPointBreadthSwenlinTradingOscillatorParams,
    DecisionPointBreadthSwenlinTradingOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use decisionpoint_breadth_swenlin_trading_oscillator::{
    decisionpoint_breadth_swenlin_trading_oscillator_alloc,
    decisionpoint_breadth_swenlin_trading_oscillator_batch_into,
    decisionpoint_breadth_swenlin_trading_oscillator_batch_js,
    decisionpoint_breadth_swenlin_trading_oscillator_free,
    decisionpoint_breadth_swenlin_trading_oscillator_into,
    decisionpoint_breadth_swenlin_trading_oscillator_js,
    DecisionPointBreadthSwenlinTradingOscillatorBatchJsOutput,
};
#[cfg(feature = "python")]
pub use decisionpoint_breadth_swenlin_trading_oscillator::{
    decisionpoint_breadth_swenlin_trading_oscillator_batch_py,
    decisionpoint_breadth_swenlin_trading_oscillator_py,
    DecisionPointBreadthSwenlinTradingOscillatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use demand_index::demand_index_into;
pub use demand_index::{
    demand_index, demand_index_batch_inner_into, demand_index_batch_par_slice,
    demand_index_batch_slice, demand_index_batch_with_kernel, demand_index_into_slices,
    demand_index_with_kernel, DemandIndexBatchBuilder, DemandIndexBatchOutput,
    DemandIndexBatchRange, DemandIndexBuilder, DemandIndexData, DemandIndexError, DemandIndexInput,
    DemandIndexOutput, DemandIndexParams, DemandIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use demand_index::{
    demand_index_alloc, demand_index_batch_into, demand_index_batch_js, demand_index_free,
    demand_index_into, demand_index_js,
};
#[cfg(feature = "python")]
pub use demand_index::{demand_index_batch_py, demand_index_py, DemandIndexStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use didi_index::didi_index_into;
pub use didi_index::{
    didi_index, didi_index_batch_inner_into, didi_index_batch_par_slice, didi_index_batch_slice,
    didi_index_batch_with_kernel, didi_index_into_slices, didi_index_with_kernel,
    DidiIndexBatchBuilder, DidiIndexBatchOutput, DidiIndexBatchRange, DidiIndexBuilder,
    DidiIndexData, DidiIndexError, DidiIndexInput, DidiIndexOutput, DidiIndexParams,
    DidiIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use didi_index::{
    didi_index_alloc, didi_index_batch_into, didi_index_batch_js, didi_index_free, didi_index_into,
    didi_index_js,
};
#[cfg(feature = "python")]
pub use didi_index::{didi_index_batch_py, didi_index_py, DidiIndexStreamPy};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use fibonacci_entry_bands::fibonacci_entry_bands_into;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use fibonacci_entry_bands::{
    fibonacci_entry_bands_alloc, fibonacci_entry_bands_batch_into, fibonacci_entry_bands_batch_js,
    fibonacci_entry_bands_free, fibonacci_entry_bands_into, fibonacci_entry_bands_js,
    FibonacciEntryBandsBatchJsOutput, FibonacciEntryBandsJsOutput,
};
pub use fibonacci_entry_bands::{
    fibonacci_entry_bands_batch_inner_into, fibonacci_entry_bands_batch_with_kernel,
    fibonacci_entry_bands_into_slices, fibonacci_entry_bands_with_kernel,
    FibonacciEntryBandsBatchBuilder, FibonacciEntryBandsBatchOutput, FibonacciEntryBandsBatchRange,
    FibonacciEntryBandsBuilder, FibonacciEntryBandsData, FibonacciEntryBandsError,
    FibonacciEntryBandsStream,
};
#[cfg(feature = "python")]
pub use fibonacci_entry_bands::{
    fibonacci_entry_bands_batch_py, fibonacci_entry_bands_py, FibonacciEntryBandsStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use fibonacci_trailing_stop::fibonacci_trailing_stop_into;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use fibonacci_trailing_stop::{
    fibonacci_trailing_stop_alloc, fibonacci_trailing_stop_batch_into,
    fibonacci_trailing_stop_batch_js, fibonacci_trailing_stop_free, fibonacci_trailing_stop_into,
    fibonacci_trailing_stop_js, FibonacciTrailingStopBatchJsOutput, FibonacciTrailingStopJsOutput,
};
pub use fibonacci_trailing_stop::{
    fibonacci_trailing_stop_batch_inner, fibonacci_trailing_stop_batch_inner_into,
    fibonacci_trailing_stop_batch_par_slices, fibonacci_trailing_stop_batch_slices,
    fibonacci_trailing_stop_batch_with_kernel, fibonacci_trailing_stop_into_slices,
    fibonacci_trailing_stop_with_kernel, FibonacciTrailingStopBatchBuilder,
    FibonacciTrailingStopBatchOutput, FibonacciTrailingStopBatchRange,
    FibonacciTrailingStopBuilder, FibonacciTrailingStopData, FibonacciTrailingStopError,
    FibonacciTrailingStopStream,
};
#[cfg(feature = "python")]
pub use fibonacci_trailing_stop::{
    fibonacci_trailing_stop_batch_py, fibonacci_trailing_stop_py, FibonacciTrailingStopStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use half_causal_estimator::half_causal_estimator_into;
pub use half_causal_estimator::{
    half_causal_estimator, half_causal_estimator_batch_inner,
    half_causal_estimator_batch_inner_into, half_causal_estimator_batch_par_slice,
    half_causal_estimator_batch_slice, half_causal_estimator_batch_with_kernel,
    half_causal_estimator_into_slices, half_causal_estimator_with_kernel,
    HalfCausalEstimatorBatchBuilder, HalfCausalEstimatorBatchOutput, HalfCausalEstimatorBatchRange,
    HalfCausalEstimatorBuilder, HalfCausalEstimatorConfidenceAdjust, HalfCausalEstimatorData,
    HalfCausalEstimatorError, HalfCausalEstimatorInput, HalfCausalEstimatorKernelType,
    HalfCausalEstimatorOutput, HalfCausalEstimatorParams, HalfCausalEstimatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use half_causal_estimator::{
    half_causal_estimator_alloc, half_causal_estimator_batch_into, half_causal_estimator_batch_js,
    half_causal_estimator_free, half_causal_estimator_into, half_causal_estimator_js,
    HalfCausalEstimatorBatchJsOutput, HalfCausalEstimatorJsOutput,
};
#[cfg(feature = "python")]
pub use half_causal_estimator::{
    half_causal_estimator_batch_py, half_causal_estimator_py, HalfCausalEstimatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use hema_trend_levels::hema_trend_levels_into;
pub use hema_trend_levels::{
    hema_trend_levels, hema_trend_levels_batch_inner_into, hema_trend_levels_batch_par_slice,
    hema_trend_levels_batch_slice, hema_trend_levels_batch_with_kernel,
    hema_trend_levels_into_slices, hema_trend_levels_with_kernel, HemaTrendLevelsBatchBuilder,
    HemaTrendLevelsBatchOutput, HemaTrendLevelsBatchRange, HemaTrendLevelsBuilder,
    HemaTrendLevelsData, HemaTrendLevelsError, HemaTrendLevelsInput, HemaTrendLevelsOutput,
    HemaTrendLevelsParams, HemaTrendLevelsStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use hema_trend_levels::{
    hema_trend_levels_alloc, hema_trend_levels_batch_into, hema_trend_levels_batch_js,
    hema_trend_levels_free, hema_trend_levels_into, hema_trend_levels_js,
    HemaTrendLevelsBatchJsOutput, HemaTrendLevelsContext, HemaTrendLevelsJsOutput,
};
#[cfg(feature = "python")]
pub use hema_trend_levels::{
    hema_trend_levels_batch_py, hema_trend_levels_py, HemaTrendLevelsStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use hull_butterfly_oscillator::hull_butterfly_oscillator_into;
pub use hull_butterfly_oscillator::{
    hull_butterfly_oscillator, hull_butterfly_oscillator_batch_inner,
    hull_butterfly_oscillator_batch_inner_into, hull_butterfly_oscillator_batch_par_slice,
    hull_butterfly_oscillator_batch_slice, hull_butterfly_oscillator_batch_with_kernel,
    hull_butterfly_oscillator_into_slices, hull_butterfly_oscillator_with_kernel,
    HullButterflyOscillatorBatchBuilder, HullButterflyOscillatorBatchOutput,
    HullButterflyOscillatorBatchRange, HullButterflyOscillatorBuilder, HullButterflyOscillatorData,
    HullButterflyOscillatorError, HullButterflyOscillatorInput, HullButterflyOscillatorOutput,
    HullButterflyOscillatorParams, HullButterflyOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use hull_butterfly_oscillator::{
    hull_butterfly_oscillator_alloc, hull_butterfly_oscillator_batch_into,
    hull_butterfly_oscillator_batch_js, hull_butterfly_oscillator_free,
    hull_butterfly_oscillator_into, hull_butterfly_oscillator_js,
    HullButterflyOscillatorBatchJsOutput, HullButterflyOscillatorJsOutput,
};
#[cfg(feature = "python")]
pub use hull_butterfly_oscillator::{
    hull_butterfly_oscillator_batch_py, hull_butterfly_oscillator_py,
    HullButterflyOscillatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use kase_peak_oscillator_with_divergences::kase_peak_oscillator_with_divergences_into;
pub use kase_peak_oscillator_with_divergences::{
    kase_peak_oscillator_with_divergences, kase_peak_oscillator_with_divergences_batch_inner_into,
    kase_peak_oscillator_with_divergences_batch_par_slice,
    kase_peak_oscillator_with_divergences_batch_slice,
    kase_peak_oscillator_with_divergences_batch_with_kernel,
    kase_peak_oscillator_with_divergences_into_slices,
    kase_peak_oscillator_with_divergences_with_kernel,
    KasePeakOscillatorWithDivergencesBatchBuilder, KasePeakOscillatorWithDivergencesBatchOutput,
    KasePeakOscillatorWithDivergencesBatchRange, KasePeakOscillatorWithDivergencesBuilder,
    KasePeakOscillatorWithDivergencesData, KasePeakOscillatorWithDivergencesError,
    KasePeakOscillatorWithDivergencesInput, KasePeakOscillatorWithDivergencesOutput,
    KasePeakOscillatorWithDivergencesParams, KasePeakOscillatorWithDivergencesStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use kase_peak_oscillator_with_divergences::{
    kase_peak_oscillator_with_divergences_alloc,
    kase_peak_oscillator_with_divergences_batch_into_js,
    kase_peak_oscillator_with_divergences_batch_js, kase_peak_oscillator_with_divergences_free,
    kase_peak_oscillator_with_divergences_into_js, kase_peak_oscillator_with_divergences_js,
};
#[cfg(feature = "python")]
pub use kase_peak_oscillator_with_divergences::{
    kase_peak_oscillator_with_divergences_batch_py, kase_peak_oscillator_with_divergences_py,
    KasePeakOscillatorWithDivergencesStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use monotonicity_index::monotonicity_index_into;
pub use monotonicity_index::{
    monotonicity_index, monotonicity_index_batch_inner, monotonicity_index_batch_inner_into,
    monotonicity_index_batch_par_slice, monotonicity_index_batch_slice,
    monotonicity_index_batch_with_kernel, monotonicity_index_into_slices,
    monotonicity_index_with_kernel, MonotonicityIndexBatchBuilder, MonotonicityIndexBatchOutput,
    MonotonicityIndexBatchRange, MonotonicityIndexBuilder, MonotonicityIndexData,
    MonotonicityIndexError, MonotonicityIndexInput, MonotonicityIndexMode, MonotonicityIndexOutput,
    MonotonicityIndexParams, MonotonicityIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use monotonicity_index::{
    monotonicity_index_alloc, monotonicity_index_batch_into, monotonicity_index_batch_js,
    monotonicity_index_free, monotonicity_index_into, monotonicity_index_js,
    MonotonicityIndexBatchJsOutput, MonotonicityIndexJsOutput,
};
#[cfg(feature = "python")]
pub use monotonicity_index::{
    monotonicity_index_batch_py, monotonicity_index_py, MonotonicityIndexStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use multi_length_stochastic_average::multi_length_stochastic_average_into;
pub use multi_length_stochastic_average::{
    multi_length_stochastic_average, multi_length_stochastic_average_batch_inner,
    multi_length_stochastic_average_batch_inner_into,
    multi_length_stochastic_average_batch_par_slice, multi_length_stochastic_average_batch_slice,
    multi_length_stochastic_average_batch_with_kernel, multi_length_stochastic_average_into_slice,
    multi_length_stochastic_average_with_kernel, MultiLengthStochasticAverageBatchBuilder,
    MultiLengthStochasticAverageBatchOutput, MultiLengthStochasticAverageBatchRange,
    MultiLengthStochasticAverageBuilder, MultiLengthStochasticAverageData,
    MultiLengthStochasticAverageError, MultiLengthStochasticAverageInput,
    MultiLengthStochasticAverageOutput, MultiLengthStochasticAverageParams,
    MultiLengthStochasticAverageStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use multi_length_stochastic_average::{
    multi_length_stochastic_average_alloc, multi_length_stochastic_average_batch_into,
    multi_length_stochastic_average_batch_js, multi_length_stochastic_average_free,
    multi_length_stochastic_average_into, multi_length_stochastic_average_js,
    MultiLengthStochasticAverageBatchJsOutput, MultiLengthStochasticAverageJsOutput,
};
#[cfg(feature = "python")]
pub use multi_length_stochastic_average::{
    multi_length_stochastic_average_batch_py, multi_length_stochastic_average_py,
    MultiLengthStochasticAverageStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use neighboring_trailing_stop::neighboring_trailing_stop_into;
pub use neighboring_trailing_stop::{
    neighboring_trailing_stop, neighboring_trailing_stop_batch_inner,
    neighboring_trailing_stop_batch_inner_into, neighboring_trailing_stop_batch_par_slices,
    neighboring_trailing_stop_batch_slices, neighboring_trailing_stop_batch_with_kernel,
    neighboring_trailing_stop_into_slices, neighboring_trailing_stop_with_kernel,
    NeighboringTrailingStopBatchBuilder, NeighboringTrailingStopBatchOutput,
    NeighboringTrailingStopBatchRange, NeighboringTrailingStopBuilder, NeighboringTrailingStopData,
    NeighboringTrailingStopError, NeighboringTrailingStopInput, NeighboringTrailingStopOutput,
    NeighboringTrailingStopParams, NeighboringTrailingStopStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use neighboring_trailing_stop::{
    neighboring_trailing_stop_alloc, neighboring_trailing_stop_batch_into,
    neighboring_trailing_stop_batch_js, neighboring_trailing_stop_free,
    neighboring_trailing_stop_into, neighboring_trailing_stop_js,
    NeighboringTrailingStopBatchJsOutput, NeighboringTrailingStopJsOutput,
};
#[cfg(feature = "python")]
pub use neighboring_trailing_stop::{
    neighboring_trailing_stop_batch_py, neighboring_trailing_stop_py,
    NeighboringTrailingStopStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use normalized_resonator::normalized_resonator_into;
pub use normalized_resonator::{
    normalized_resonator, normalized_resonator_batch_inner, normalized_resonator_batch_inner_into,
    normalized_resonator_batch_par_slice, normalized_resonator_batch_slice,
    normalized_resonator_batch_with_kernel, normalized_resonator_into_slices,
    normalized_resonator_with_kernel, NormalizedResonatorBatchBuilder,
    NormalizedResonatorBatchOutput, NormalizedResonatorBatchRange, NormalizedResonatorBuilder,
    NormalizedResonatorData, NormalizedResonatorError, NormalizedResonatorInput,
    NormalizedResonatorOutput, NormalizedResonatorParams, NormalizedResonatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use normalized_resonator::{
    normalized_resonator_alloc, normalized_resonator_batch_into, normalized_resonator_batch_js,
    normalized_resonator_free, normalized_resonator_into, normalized_resonator_js,
    NormalizedResonatorBatchJsOutput, NormalizedResonatorJsOutput,
};
#[cfg(feature = "python")]
pub use normalized_resonator::{
    normalized_resonator_batch_py, normalized_resonator_py, NormalizedResonatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use velocity_acceleration_indicator::velocity_acceleration_indicator_into;
pub use velocity_acceleration_indicator::{
    velocity_acceleration_indicator, velocity_acceleration_indicator_batch_inner,
    velocity_acceleration_indicator_batch_inner_into,
    velocity_acceleration_indicator_batch_par_slice, velocity_acceleration_indicator_batch_slice,
    velocity_acceleration_indicator_batch_with_kernel, velocity_acceleration_indicator_into_slice,
    velocity_acceleration_indicator_with_kernel, VelocityAccelerationIndicatorBatchBuilder,
    VelocityAccelerationIndicatorBatchOutput, VelocityAccelerationIndicatorBatchRange,
    VelocityAccelerationIndicatorBuilder, VelocityAccelerationIndicatorData,
    VelocityAccelerationIndicatorError, VelocityAccelerationIndicatorInput,
    VelocityAccelerationIndicatorOutput, VelocityAccelerationIndicatorParams,
    VelocityAccelerationIndicatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use velocity_acceleration_indicator::{
    velocity_acceleration_indicator_alloc, velocity_acceleration_indicator_batch_into,
    velocity_acceleration_indicator_batch_js, velocity_acceleration_indicator_free,
    velocity_acceleration_indicator_into, velocity_acceleration_indicator_js,
    VelocityAccelerationIndicatorBatchJsOutput,
};
#[cfg(feature = "python")]
pub use velocity_acceleration_indicator::{
    velocity_acceleration_indicator_batch_py, velocity_acceleration_indicator_py,
    VelocityAccelerationIndicatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use volume_energy_reservoirs::volume_energy_reservoirs_into;
pub use volume_energy_reservoirs::{
    volume_energy_reservoirs, volume_energy_reservoirs_batch_inner,
    volume_energy_reservoirs_batch_inner_into, volume_energy_reservoirs_batch_par_slices,
    volume_energy_reservoirs_batch_slices, volume_energy_reservoirs_batch_with_kernel,
    volume_energy_reservoirs_into_slices, volume_energy_reservoirs_with_kernel,
    VolumeEnergyReservoirsBatchBuilder, VolumeEnergyReservoirsBatchOutput,
    VolumeEnergyReservoirsBatchRange, VolumeEnergyReservoirsBuilder, VolumeEnergyReservoirsData,
    VolumeEnergyReservoirsError, VolumeEnergyReservoirsInput, VolumeEnergyReservoirsOutput,
    VolumeEnergyReservoirsParams, VolumeEnergyReservoirsStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use volume_energy_reservoirs::{
    volume_energy_reservoirs_alloc, volume_energy_reservoirs_batch_into,
    volume_energy_reservoirs_batch_js, volume_energy_reservoirs_free,
    volume_energy_reservoirs_into, volume_energy_reservoirs_js,
    VolumeEnergyReservoirsBatchJsOutput, VolumeEnergyReservoirsJsOutput,
};
#[cfg(feature = "python")]
pub use volume_energy_reservoirs::{
    volume_energy_reservoirs_batch_py, volume_energy_reservoirs_py, VolumeEnergyReservoirsStreamPy,
};

#[cfg(feature = "python")]
pub use range_filter::{range_filter_batch_py, range_filter_py, RangeFilterStreamPy};

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use absolute_strength_index_oscillator::absolute_strength_index_oscillator_into;
pub use absolute_strength_index_oscillator::{
    absolute_strength_index_oscillator, absolute_strength_index_oscillator_batch_inner_into,
    absolute_strength_index_oscillator_batch_par_slice,
    absolute_strength_index_oscillator_batch_slice,
    absolute_strength_index_oscillator_batch_with_kernel,
    absolute_strength_index_oscillator_into_slices, absolute_strength_index_oscillator_with_kernel,
    AbsoluteStrengthIndexOscillatorBatchBuilder, AbsoluteStrengthIndexOscillatorBatchOutput,
    AbsoluteStrengthIndexOscillatorBatchRange, AbsoluteStrengthIndexOscillatorBuilder,
    AbsoluteStrengthIndexOscillatorData, AbsoluteStrengthIndexOscillatorError,
    AbsoluteStrengthIndexOscillatorInput, AbsoluteStrengthIndexOscillatorOutput,
    AbsoluteStrengthIndexOscillatorParams, AbsoluteStrengthIndexOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use absolute_strength_index_oscillator::{
    absolute_strength_index_oscillator_alloc, absolute_strength_index_oscillator_batch_into,
    absolute_strength_index_oscillator_batch_js, absolute_strength_index_oscillator_free,
    absolute_strength_index_oscillator_into, absolute_strength_index_oscillator_js,
    AbsoluteStrengthIndexOscillatorBatchJsOutput,
};
#[cfg(feature = "python")]
pub use absolute_strength_index_oscillator::{
    absolute_strength_index_oscillator_batch_py, absolute_strength_index_oscillator_py,
    AbsoluteStrengthIndexOscillatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use adaptive_bandpass_trigger_oscillator::adaptive_bandpass_trigger_oscillator_into;
pub use adaptive_bandpass_trigger_oscillator::{
    adaptive_bandpass_trigger_oscillator, adaptive_bandpass_trigger_oscillator_batch_inner_into,
    adaptive_bandpass_trigger_oscillator_batch_par_slice,
    adaptive_bandpass_trigger_oscillator_batch_slice,
    adaptive_bandpass_trigger_oscillator_batch_with_kernel,
    adaptive_bandpass_trigger_oscillator_into_slices,
    adaptive_bandpass_trigger_oscillator_with_kernel,
    AdaptiveBandpassTriggerOscillatorBatchBuilder, AdaptiveBandpassTriggerOscillatorBatchOutput,
    AdaptiveBandpassTriggerOscillatorBatchRange, AdaptiveBandpassTriggerOscillatorBuilder,
    AdaptiveBandpassTriggerOscillatorData, AdaptiveBandpassTriggerOscillatorError,
    AdaptiveBandpassTriggerOscillatorInput, AdaptiveBandpassTriggerOscillatorOutput,
    AdaptiveBandpassTriggerOscillatorParams, AdaptiveBandpassTriggerOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use adaptive_bandpass_trigger_oscillator::{
    adaptive_bandpass_trigger_oscillator_alloc, adaptive_bandpass_trigger_oscillator_batch_into,
    adaptive_bandpass_trigger_oscillator_batch_js, adaptive_bandpass_trigger_oscillator_free,
    adaptive_bandpass_trigger_oscillator_into, adaptive_bandpass_trigger_oscillator_js,
    AdaptiveBandpassTriggerOscillatorBatchJsOutput,
};
#[cfg(feature = "python")]
pub use adaptive_bandpass_trigger_oscillator::{
    adaptive_bandpass_trigger_oscillator_batch_py, adaptive_bandpass_trigger_oscillator_py,
    AdaptiveBandpassTriggerOscillatorStreamPy,
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
pub use ehlers_linear_extrapolation_predictor::ehlers_linear_extrapolation_predictor_into;
pub use ehlers_linear_extrapolation_predictor::{
    ehlers_linear_extrapolation_predictor, ehlers_linear_extrapolation_predictor_batch_inner_into,
    ehlers_linear_extrapolation_predictor_batch_par_slice,
    ehlers_linear_extrapolation_predictor_batch_slice,
    ehlers_linear_extrapolation_predictor_batch_with_kernel,
    ehlers_linear_extrapolation_predictor_into_slices,
    ehlers_linear_extrapolation_predictor_with_kernel,
    EhlersLinearExtrapolationPredictorBatchBuilder, EhlersLinearExtrapolationPredictorBatchOutput,
    EhlersLinearExtrapolationPredictorBatchRange, EhlersLinearExtrapolationPredictorBuilder,
    EhlersLinearExtrapolationPredictorData, EhlersLinearExtrapolationPredictorError,
    EhlersLinearExtrapolationPredictorInput, EhlersLinearExtrapolationPredictorOutput,
    EhlersLinearExtrapolationPredictorParams, EhlersLinearExtrapolationPredictorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use ehlers_linear_extrapolation_predictor::{
    ehlers_linear_extrapolation_predictor_alloc, ehlers_linear_extrapolation_predictor_batch_into,
    ehlers_linear_extrapolation_predictor_batch_js, ehlers_linear_extrapolation_predictor_free,
    ehlers_linear_extrapolation_predictor_into, ehlers_linear_extrapolation_predictor_js,
    EhlersLinearExtrapolationPredictorBatchJsOutput,
};
#[cfg(feature = "python")]
pub use ehlers_linear_extrapolation_predictor::{
    ehlers_linear_extrapolation_predictor_batch_py, ehlers_linear_extrapolation_predictor_py,
    EhlersLinearExtrapolationPredictorStreamPy,
};
pub use er::{er, ErInput, ErOutput, ErParams};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use garman_klass_volatility::garman_klass_volatility_into;
pub use garman_klass_volatility::{
    garman_klass_volatility, garman_klass_volatility_batch_par_slice,
    garman_klass_volatility_batch_slice, garman_klass_volatility_batch_with_kernel,
    garman_klass_volatility_into_slice, garman_klass_volatility_with_kernel,
    GarmanKlassVolatilityBatchBuilder, GarmanKlassVolatilityBatchOutput,
    GarmanKlassVolatilityBatchRange, GarmanKlassVolatilityBuilder, GarmanKlassVolatilityData,
    GarmanKlassVolatilityError, GarmanKlassVolatilityInput, GarmanKlassVolatilityOutput,
    GarmanKlassVolatilityParams, GarmanKlassVolatilityStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use garman_klass_volatility::{
    garman_klass_volatility_alloc, garman_klass_volatility_batch_into,
    garman_klass_volatility_batch_js, garman_klass_volatility_free, garman_klass_volatility_into,
    garman_klass_volatility_js,
};
#[cfg(feature = "python")]
pub use garman_klass_volatility::{
    garman_klass_volatility_batch_py, garman_klass_volatility_py, GarmanKlassVolatilityStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use gopalakrishnan_range_index::gopalakrishnan_range_index_into;
pub use gopalakrishnan_range_index::{
    gopalakrishnan_range_index, gopalakrishnan_range_index_batch_inner_into,
    gopalakrishnan_range_index_batch_par_slice, gopalakrishnan_range_index_batch_slice,
    gopalakrishnan_range_index_batch_with_kernel, gopalakrishnan_range_index_into_slice,
    gopalakrishnan_range_index_with_kernel, GopalakrishnanRangeIndexBatchBuilder,
    GopalakrishnanRangeIndexBatchOutput, GopalakrishnanRangeIndexBatchRange,
    GopalakrishnanRangeIndexBuilder, GopalakrishnanRangeIndexData, GopalakrishnanRangeIndexError,
    GopalakrishnanRangeIndexInput, GopalakrishnanRangeIndexOutput, GopalakrishnanRangeIndexParams,
    GopalakrishnanRangeIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use gopalakrishnan_range_index::{
    gopalakrishnan_range_index_alloc, gopalakrishnan_range_index_batch_into,
    gopalakrishnan_range_index_batch_js, gopalakrishnan_range_index_free,
    gopalakrishnan_range_index_into, gopalakrishnan_range_index_js,
};
#[cfg(feature = "python")]
pub use gopalakrishnan_range_index::{
    gopalakrishnan_range_index_batch_py, gopalakrishnan_range_index_py,
    GopalakrishnanRangeIndexStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use grover_llorens_cycle_oscillator::grover_llorens_cycle_oscillator_into;
pub use grover_llorens_cycle_oscillator::{
    grover_llorens_cycle_oscillator, grover_llorens_cycle_oscillator_batch_inner_into,
    grover_llorens_cycle_oscillator_batch_par_slice, grover_llorens_cycle_oscillator_batch_slice,
    grover_llorens_cycle_oscillator_batch_with_kernel, grover_llorens_cycle_oscillator_into_slice,
    grover_llorens_cycle_oscillator_with_kernel, GroverLlorensCycleOscillatorBatchBuilder,
    GroverLlorensCycleOscillatorBatchOutput, GroverLlorensCycleOscillatorBatchRange,
    GroverLlorensCycleOscillatorBuilder, GroverLlorensCycleOscillatorData,
    GroverLlorensCycleOscillatorError, GroverLlorensCycleOscillatorInput,
    GroverLlorensCycleOscillatorOutput, GroverLlorensCycleOscillatorParams,
    GroverLlorensCycleOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use grover_llorens_cycle_oscillator::{
    grover_llorens_cycle_oscillator_alloc, grover_llorens_cycle_oscillator_batch_into,
    grover_llorens_cycle_oscillator_batch_js, grover_llorens_cycle_oscillator_free,
    grover_llorens_cycle_oscillator_into, grover_llorens_cycle_oscillator_js,
    GroverLlorensCycleOscillatorBatchJsOutput,
};
#[cfg(feature = "python")]
pub use grover_llorens_cycle_oscillator::{
    grover_llorens_cycle_oscillator_batch_py, grover_llorens_cycle_oscillator_py,
    GroverLlorensCycleOscillatorStreamPy,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use historical_volatility::historical_volatility_into;
pub use historical_volatility::{
    historical_volatility, historical_volatility_batch_inner_into,
    historical_volatility_batch_par_slice, historical_volatility_batch_slice,
    historical_volatility_batch_with_kernel, historical_volatility_into_slice,
    historical_volatility_with_kernel, HistoricalVolatilityBatchBuilder,
    HistoricalVolatilityBatchOutput, HistoricalVolatilityBatchRange, HistoricalVolatilityBuilder,
    HistoricalVolatilityData, HistoricalVolatilityError, HistoricalVolatilityInput,
    HistoricalVolatilityOutput, HistoricalVolatilityParams, HistoricalVolatilityStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use historical_volatility::{
    historical_volatility_alloc, historical_volatility_batch_into, historical_volatility_batch_js,
    historical_volatility_free, historical_volatility_into, historical_volatility_js,
};
#[cfg(feature = "python")]
pub use historical_volatility::{
    historical_volatility_batch_py, historical_volatility_py, HistoricalVolatilityStreamPy,
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
pub use intraday_momentum_index::intraday_momentum_index_into;
pub use intraday_momentum_index::{
    intraday_momentum_index, intraday_momentum_index_batch_inner_into,
    intraday_momentum_index_batch_par_slice, intraday_momentum_index_batch_slice,
    intraday_momentum_index_batch_with_kernel, intraday_momentum_index_into_slices,
    intraday_momentum_index_with_kernel, IntradayMomentumIndexBatchBuilder,
    IntradayMomentumIndexBatchOutput, IntradayMomentumIndexBatchRange,
    IntradayMomentumIndexBuilder, IntradayMomentumIndexData, IntradayMomentumIndexError,
    IntradayMomentumIndexInput, IntradayMomentumIndexOutput, IntradayMomentumIndexParams,
    IntradayMomentumIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use intraday_momentum_index::{
    intraday_momentum_index_alloc, intraday_momentum_index_batch_into,
    intraday_momentum_index_batch_js, intraday_momentum_index_free, intraday_momentum_index_into,
    intraday_momentum_index_js,
};
#[cfg(feature = "python")]
pub use intraday_momentum_index::{
    intraday_momentum_index_batch_py, intraday_momentum_index_py, IntradayMomentumIndexStreamPy,
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
pub use premier_rsi_oscillator::premier_rsi_oscillator_into;
pub use premier_rsi_oscillator::{
    premier_rsi_oscillator, premier_rsi_oscillator_batch_inner_into,
    premier_rsi_oscillator_batch_par_slice, premier_rsi_oscillator_batch_slice,
    premier_rsi_oscillator_batch_with_kernel, premier_rsi_oscillator_into_slice,
    premier_rsi_oscillator_with_kernel, PremierRsiOscillatorBatchBuilder,
    PremierRsiOscillatorBatchOutput, PremierRsiOscillatorBatchRange, PremierRsiOscillatorBuilder,
    PremierRsiOscillatorData, PremierRsiOscillatorError, PremierRsiOscillatorInput,
    PremierRsiOscillatorOutput, PremierRsiOscillatorParams, PremierRsiOscillatorStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use premier_rsi_oscillator::{
    premier_rsi_oscillator_alloc, premier_rsi_oscillator_batch_into,
    premier_rsi_oscillator_batch_js, premier_rsi_oscillator_free, premier_rsi_oscillator_into,
    premier_rsi_oscillator_js, PremierRsiOscillatorBatchJsOutput,
};
#[cfg(feature = "python")]
pub use premier_rsi_oscillator::{
    premier_rsi_oscillator_batch_py, premier_rsi_oscillator_py, PremierRsiOscillatorStreamPy,
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
pub use squeeze_index::squeeze_index_into;
pub use squeeze_index::{
    squeeze_index, squeeze_index_batch_inner_into, squeeze_index_batch_par_slice,
    squeeze_index_batch_slice, squeeze_index_batch_with_kernel, squeeze_index_into_slice,
    squeeze_index_with_kernel, SqueezeIndexBatchBuilder, SqueezeIndexBatchOutput,
    SqueezeIndexBatchRange, SqueezeIndexBuilder, SqueezeIndexData, SqueezeIndexError,
    SqueezeIndexInput, SqueezeIndexOutput, SqueezeIndexParams, SqueezeIndexStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use squeeze_index::{
    squeeze_index_alloc, squeeze_index_batch_into, squeeze_index_batch_js, squeeze_index_free,
    squeeze_index_into, squeeze_index_js, SqueezeIndexBatchJsOutput,
};
#[cfg(feature = "python")]
pub use squeeze_index::{squeeze_index_batch_py, squeeze_index_py, SqueezeIndexStreamPy};
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
pub use stochastic_distance::{
    stochastic_distance, stochastic_distance_batch_inner_into, stochastic_distance_batch_par_slice,
    stochastic_distance_batch_slice, stochastic_distance_batch_with_kernel,
    stochastic_distance_into_slices, stochastic_distance_with_kernel,
    StochasticDistanceBatchBuilder, StochasticDistanceBatchOutput, StochasticDistanceBatchRange,
    StochasticDistanceBuilder, StochasticDistanceData, StochasticDistanceError,
    StochasticDistanceInput, StochasticDistanceOutput, StochasticDistanceParams,
    StochasticDistanceStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use stochastic_distance::{
    stochastic_distance_alloc, stochastic_distance_batch_into, stochastic_distance_batch_js,
    stochastic_distance_free, stochastic_distance_into, stochastic_distance_js,
    StochasticDistanceBatchJsOutput,
};
#[cfg(feature = "python")]
pub use stochastic_distance::{
    stochastic_distance_batch_py, stochastic_distance_py, StochasticDistanceStreamPy,
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
pub use vertical_horizontal_filter::vertical_horizontal_filter_into;
pub use vertical_horizontal_filter::{
    vertical_horizontal_filter, vertical_horizontal_filter_batch_inner_into,
    vertical_horizontal_filter_batch_par_slice, vertical_horizontal_filter_batch_slice,
    vertical_horizontal_filter_batch_with_kernel, vertical_horizontal_filter_into_slice,
    vertical_horizontal_filter_with_kernel, VerticalHorizontalFilterBatchBuilder,
    VerticalHorizontalFilterBatchOutput, VerticalHorizontalFilterBatchRange,
    VerticalHorizontalFilterBuilder, VerticalHorizontalFilterData, VerticalHorizontalFilterError,
    VerticalHorizontalFilterInput, VerticalHorizontalFilterOutput, VerticalHorizontalFilterParams,
    VerticalHorizontalFilterStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use vertical_horizontal_filter::{
    vertical_horizontal_filter_alloc, vertical_horizontal_filter_batch_into,
    vertical_horizontal_filter_batch_js, vertical_horizontal_filter_free,
    vertical_horizontal_filter_into, vertical_horizontal_filter_js,
};
#[cfg(feature = "python")]
pub use vertical_horizontal_filter::{
    vertical_horizontal_filter_batch_py, vertical_horizontal_filter_py,
    VerticalHorizontalFilterStreamPy,
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
pub use vwap_zscore_with_signals::vwap_zscore_with_signals_into;
pub use vwap_zscore_with_signals::{
    vwap_zscore_with_signals, vwap_zscore_with_signals_batch_inner_into,
    vwap_zscore_with_signals_batch_par_slice, vwap_zscore_with_signals_batch_slice,
    vwap_zscore_with_signals_batch_with_kernel, vwap_zscore_with_signals_into_slices,
    vwap_zscore_with_signals_with_kernel, VwapZscoreWithSignalsBatchBuilder,
    VwapZscoreWithSignalsBatchOutput, VwapZscoreWithSignalsBatchRange,
    VwapZscoreWithSignalsBuilder, VwapZscoreWithSignalsData, VwapZscoreWithSignalsError,
    VwapZscoreWithSignalsInput, VwapZscoreWithSignalsOutput, VwapZscoreWithSignalsParams,
    VwapZscoreWithSignalsStream,
};
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use vwap_zscore_with_signals::{
    vwap_zscore_with_signals_alloc, vwap_zscore_with_signals_batch_into,
    vwap_zscore_with_signals_batch_js, vwap_zscore_with_signals_free,
    vwap_zscore_with_signals_into, vwap_zscore_with_signals_js,
};
#[cfg(feature = "python")]
pub use vwap_zscore_with_signals::{
    vwap_zscore_with_signals_batch_py, vwap_zscore_with_signals_py, VwapZscoreWithSignalsStreamPy,
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
