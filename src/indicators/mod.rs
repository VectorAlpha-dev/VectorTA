pub mod acosc;
pub mod ad;
pub mod adosc;
pub mod adx;
pub mod adxr;
pub mod alligator;
pub mod alphatrend;
pub mod dispatch;
pub use alphatrend::{alphatrend, AlphaTrendInput, AlphaTrendOutput, AlphaTrendParams};
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
pub mod emd;
pub mod emd_trend;
pub mod emv;
pub mod er;
pub mod eri;
pub mod evasive_supertrend;
pub mod fisher;
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
pub mod halftrend;
pub use halftrend::{halftrend, HalfTrendInput, HalfTrendOutput, HalfTrendParams};
pub mod ift_rsi;
pub mod kaufmanstop;
pub mod kdj;
pub mod keltner;
pub mod kst;
pub mod kurtosis;
pub mod kvo;
pub mod linearreg_angle;
pub mod linearreg_intercept;
pub mod linearreg_slope;
pub mod lpc;
pub use lpc::{lpc, LpcInput, LpcOutput, LpcParams};
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
pub mod range_filter;
pub mod reversal_signals;
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
pub mod stochastic_money_flow_index;
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
pub use mom::{mom, MomInput, MomOutput, MomParams};
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
