pub mod alma;
pub mod buff_averages;
pub mod corrected_moving_average;
pub mod cwma;
pub mod dema;
pub mod dma;
pub mod edcf;
pub mod ehlers_ecema;
pub mod ehlers_itrend;
pub mod ehlers_kama;
pub mod ehlers_pma;
pub mod ehlers_undersampled_double_moving_average;
pub mod ehma;
pub mod elastic_volume_weighted_moving_average;
pub mod ema;
pub mod ema_deviation_corrected_t3;
pub mod epma;
pub mod frama;
pub mod fwma;
pub mod gaussian;
pub mod highpass;
pub mod highpass_2_pole;
pub mod hma;
pub mod hwma;
pub mod jma;
pub mod jsa;
pub mod kama;
pub mod linreg;
pub mod logarithmic_moving_average;
pub mod ma;
pub mod ma_batch;
pub mod ma_stream;
pub mod maaq;
pub mod mama;
pub mod mwdx;
pub mod n_order_ema;
pub mod nama;
pub mod nma;
pub mod param_schema;
pub mod pwma;
pub mod reflex;
pub mod registry;
pub mod sama;
pub mod sgf;
pub mod sinwma;
pub mod sma;
pub mod smma;
pub mod sqwma;
pub mod srwma;
pub mod supersmoother;
pub mod supersmoother_3_pole;
pub mod swma;
pub mod tema;
pub mod tilson;
pub mod tradjema;
pub mod trendflex;
pub mod trima;
pub mod uma;
pub mod volatility_adjusted_ma;
pub mod volume_adjusted_ma;
pub mod vpwma;
pub mod vwap;
pub mod vwma;
pub mod wave_smoother;
pub mod wilders;
pub mod wma;
pub mod zlema;

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use corrected_moving_average::corrected_moving_average_into;
pub use corrected_moving_average::{
    corrected_moving_average, corrected_moving_average_batch_par_slice,
    corrected_moving_average_batch_slice, corrected_moving_average_batch_with_kernel,
    corrected_moving_average_into_slice, corrected_moving_average_with_kernel,
    expand_grid_corrected_moving_average, CorrectedMovingAverageBatchBuilder,
    CorrectedMovingAverageBatchOutput, CorrectedMovingAverageBatchRange,
    CorrectedMovingAverageBuilder, CorrectedMovingAverageData, CorrectedMovingAverageError,
    CorrectedMovingAverageInput, CorrectedMovingAverageOutput, CorrectedMovingAverageParams,
    CorrectedMovingAverageStream,
};
pub use cwma::{cwma, CwmaInput, CwmaOutput, CwmaParams};
pub use dma::{
    dma, dma_batch_with_kernel, dma_into_slice, dma_with_kernel, DmaBatchBuilder, DmaBatchOutput,
    DmaBatchRange, DmaBuilder, DmaData, DmaError, DmaInput, DmaOutput, DmaParams, DmaStream,
};

pub use edcf::{edcf, EdcfInput, EdcfOutput, EdcfParams};
pub use ehlers_kama::{ehlers_kama, EhlersKamaInput, EhlersKamaOutput, EhlersKamaParams};
pub use ehlers_pma::{
    ehlers_pma, expand_grid as expand_grid_ehlers_pma, EhlersPmaBatchRange, EhlersPmaInput,
    EhlersPmaOutput, EhlersPmaParams,
};
pub use ehlers_undersampled_double_moving_average::{
    ehlers_undersampled_double_moving_average,
    expand_grid_ehlers_undersampled_double_moving_average,
    EhlersUndersampledDoubleMovingAverageBatchRange, EhlersUndersampledDoubleMovingAverageInput,
    EhlersUndersampledDoubleMovingAverageOutput, EhlersUndersampledDoubleMovingAverageParams,
};
pub use elastic_volume_weighted_moving_average::{
    elastic_volume_weighted_moving_average, expand_grid_elastic_volume_weighted_moving_average,
    ElasticVolumeWeightedMovingAverageBatchRange, ElasticVolumeWeightedMovingAverageInput,
    ElasticVolumeWeightedMovingAverageOutput, ElasticVolumeWeightedMovingAverageParams,
};
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use logarithmic_moving_average::logarithmic_moving_average_into;
pub use logarithmic_moving_average::{
    expand_grid_logarithmic_moving_average, logarithmic_moving_average,
    logarithmic_moving_average_batch_par_slice, logarithmic_moving_average_batch_slice,
    logarithmic_moving_average_batch_with_kernel, logarithmic_moving_average_into_slice,
    logarithmic_moving_average_with_kernel, LogarithmicMovingAverageBatchBuilder,
    LogarithmicMovingAverageBatchOutput, LogarithmicMovingAverageBatchRange,
    LogarithmicMovingAverageBuilder, LogarithmicMovingAverageData, LogarithmicMovingAverageError,
    LogarithmicMovingAverageInput, LogarithmicMovingAverageOutput, LogarithmicMovingAverageParams,
    LogarithmicMovingAverageStream,
};
pub use uma::{uma, UmaInput, UmaOutput, UmaParams};
pub use volatility_adjusted_ma::{
    vama as volatility_adjusted_ma, VamaInput as VolatilityAdjustedMaInput,
    VamaOutput as VolatilityAdjustedMaOutput, VamaParams as VolatilityAdjustedMaParams,
};
pub use volume_adjusted_ma::{
    VolumeAdjustedMa as volume_adjusted_ma, VolumeAdjustedMaInput, VolumeAdjustedMaOutput,
    VolumeAdjustedMaParams,
};

pub use ehma::{
    ehma, ehma_batch_inner_into, ehma_batch_par_slice, ehma_batch_slice, ehma_batch_with_kernel,
    ehma_batch_with_kernel_slice, ehma_into_slice, ehma_with_kernel, EhmaBatchBuilder,
    EhmaBatchOutput, EhmaBatchRange, EhmaBuilder, EhmaData, EhmaError, EhmaInput, EhmaOutput,
    EhmaParams, EhmaStream,
};

pub use nama::{
    nama, nama_batch_with_kernel, nama_into_slice, nama_with_kernel, NamaBatchBuilder,
    NamaBatchOutput, NamaBatchRange, NamaBuilder, NamaData, NamaError, NamaInput, NamaOutput,
    NamaParams, NamaStream,
};

pub use n_order_ema::{
    expand_grid_n_order_ema, n_order_ema, n_order_ema_batch_from_input_with_kernel,
    n_order_ema_batch_with_kernel, n_order_ema_into_slice, n_order_ema_with_kernel,
    NOrderEmaBatchBuilder, NOrderEmaBatchOutput, NOrderEmaBatchRange, NOrderEmaBuilder,
    NOrderEmaData, NOrderEmaError, NOrderEmaIirStyle, NOrderEmaInput, NOrderEmaOutput,
    NOrderEmaParams, NOrderEmaStream, NOrderEmaStyle,
};

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub use n_order_ema::n_order_ema_into;

pub use sama::{
    sama, sama_batch_par_slice, sama_batch_slice, sama_batch_with_kernel, sama_into_slice,
    sama_with_kernel, SamaBatchBuilder, SamaBatchOutput, SamaBatchRange, SamaBuilder, SamaData,
    SamaError, SamaInput, SamaOutput, SamaParams, SamaStream,
};

pub use sgf::{
    expand_grid as expand_grid_sgf, sgf, sgf_batch_into_slice, sgf_batch_par_slice,
    sgf_batch_slice, sgf_batch_with_kernel, sgf_into_slice, sgf_with_kernel, SgfBatchBuilder,
    SgfBatchOutput, SgfBatchRange, SgfBuilder, SgfData, SgfError, SgfInput, SgfOutput, SgfParams,
    SgfStream,
};

#[cfg(feature = "python")]
pub use dma::{dma_batch_py, dma_py, DmaStreamPy};

#[cfg(feature = "python")]
pub use ehma::{ehma_batch_py, ehma_py, EhmaStreamPy};

#[cfg(feature = "python")]
pub use corrected_moving_average::{
    corrected_moving_average_batch_py, corrected_moving_average_py, CorrectedMovingAverageStreamPy,
};

#[cfg(feature = "python")]
pub use nama::{nama_batch_py, nama_py, NamaStreamPy};

#[cfg(feature = "python")]
pub use n_order_ema::{
    n_order_ema_batch_py, n_order_ema_py, register_n_order_ema_module, NOrderEmaStreamPy,
};

#[cfg(feature = "python")]
pub use sama::{sama_batch_py, sama_py, SamaStreamPy};

#[cfg(feature = "python")]
pub use sgf::{sgf_batch_py, sgf_py, SgfStreamPy};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use corrected_moving_average::{
    corrected_moving_average_alloc, corrected_moving_average_batch_into,
    corrected_moving_average_batch_js, corrected_moving_average_free,
    corrected_moving_average_into, corrected_moving_average_js,
};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use nama::{nama_alloc, nama_batch_unified_js, nama_free, nama_into, nama_js};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub use n_order_ema::{
    n_order_ema_alloc, n_order_ema_batch_into, n_order_ema_batch_js, n_order_ema_free,
    n_order_ema_into, n_order_ema_js,
};
