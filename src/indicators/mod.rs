pub mod acosc;
pub mod ad;
pub mod adosc;
pub mod adx;
pub mod adxr;
pub mod alligator;
pub mod alphatrend;
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
pub mod emd;
pub mod emv;
pub mod er;
pub mod eri;
pub mod fisher;
pub mod fosc;
pub mod fvg_trailing_stop;
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
    alma, buff_averages, cwma, dema, edcf, ehlers_itrend, ehlers_pma, ema, epma, frama, fwma,
    gaussian, highpass, highpass_2_pole, hma, hwma, jma, jsa, kama, linreg, maaq, mama, mwdx, nma,
    pwma, reflex, sinwma, sma, smma, sqwma, srwma, supersmoother, supersmoother_3_pole, swma, tema,
    tilson, tradjema, trendflex, trima, uma, volatility_adjusted_ma, volume_adjusted_ma, vpwma,
    vwap, vwma, wilders, wma, zlema,
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
