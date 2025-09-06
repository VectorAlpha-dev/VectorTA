// Other indicators module exports

pub mod buff_averages;
pub use buff_averages::{
    buff_averages, buff_averages_with_kernel, buff_averages_into_slices,
    buff_averages_batch_with_kernel, BuffAveragesInput, BuffAveragesOutput, 
    BuffAveragesParams, BuffAveragesData, BuffAveragesBuilder, BuffAveragesStream, 
    BuffAveragesError, BuffAveragesBatchRange, BuffAveragesBatchOutput, 
    BuffAveragesBatchBuilder,
};

#[cfg(feature = "python")]
pub use buff_averages::{buff_averages_py, buff_averages_batch_py, BuffAveragesStreamPy};

#[cfg(feature = "wasm")]
pub use buff_averages::{
    buff_averages_js, buff_averages_unified_js, buff_averages_into, 
    buff_averages_alloc, buff_averages_free, BuffAveragesJsResult,
};

pub mod qqe;
pub use qqe::{
    qqe, qqe_with_kernel, qqe_into_slices,
    qqe_batch_with_kernel, qqe_batch_slice, qqe_batch_par_slice,
    QqeInput, QqeOutput, QqeParams, QqeData, QqeBuilder, 
    QqeStream, QqeError, QqeBatchRange, QqeBatchOutput, 
    QqeBatchBuilder,
};

#[cfg(feature = "python")]
pub use qqe::{qqe_py, qqe_batch_py, QqeStreamPy};

#[cfg(feature = "wasm")]
pub use qqe::{
    qqe_js, qqe_unified_js, qqe_into, qqe_batch_unified_js, qqe_batch_into,
    qqe_alloc, qqe_free, QqeJsResult, QqeBatchConfig, QqeBatchJsOutput,
};

pub mod volume_adjusted_ma;
pub use volume_adjusted_ma::{
    VolumeAdjustedMa, VolumeAdjustedMa_with_kernel, VolumeAdjustedMa_into_slice,
    VolumeAdjustedMa_batch_with_kernel, VolumeAdjustedMa_batch_par_slice,
    VolumeAdjustedMaInput, VolumeAdjustedMaOutput, VolumeAdjustedMaParams, VolumeAdjustedMaData, VolumeAdjustedMaBuilder,
    VolumeAdjustedMaStream, VolumeAdjustedMaError, VolumeAdjustedMaBatchRange, VolumeAdjustedMaBatchOutput,
    VolumeAdjustedMaBatchBuilder,
};

#[cfg(feature = "python")]
pub use volume_adjusted_ma::{volume_adjusted_ma_py, volume_adjusted_ma_batch_py, VolumeAdjustedMaStreamPy};

#[cfg(feature = "wasm")]
pub use volume_adjusted_ma::{
    VolumeAdjustedMa_js, VolumeAdjustedMa_unified_js, VolumeAdjustedMa_into, VolumeAdjustedMa_batch_unified_js,
    VolumeAdjustedMa_alloc, VolumeAdjustedMa_free, VolumeAdjustedMaJsResult, VolumeAdjustedMaBatchConfig, VolumeAdjustedMaBatchJsOutput,
};

pub mod nadaraya_watson_envelope;
pub use nadaraya_watson_envelope::{
    nadaraya_watson_envelope, nadaraya_watson_envelope_with_kernel, nadaraya_watson_envelope_into_slices,
    nadaraya_watson_envelope_batch_with_kernel, nwe_batch_with_kernel,
    NweInput, NweOutput, NweParams, NweData, NweBuilder, NweBatchBuilder,
    NweStream, NweError, NweBatchRange, NweBatchOutput,
};

#[cfg(feature = "python")]
pub use nadaraya_watson_envelope::{nadaraya_watson_envelope_py, nadaraya_watson_envelope_batch_py, NweStreamPy};

#[cfg(feature = "wasm")]
pub use nadaraya_watson_envelope::{
    nadaraya_watson_envelope_js, nadaraya_watson_envelope_unified_js, nadaraya_watson_envelope_into,
    nadaraya_watson_envelope_batch_unified_js,
    nadaraya_watson_envelope_alloc, nadaraya_watson_envelope_free, NweJsResult, NweBatchJsOutput,
};

pub mod beardy_squeeze_pro;
pub use beardy_squeeze_pro::{
    beardy_squeeze_pro, beardy_squeeze_pro_with_kernel,
    BeardySqueezeProInput, BeardySqueezeProOutput, BeardySqueezeProParams, BeardySqueezeProData,
    BeardySqueezeProError,
};

#[cfg(feature = "python")]
pub use beardy_squeeze_pro::beardy_squeeze_pro_py;

#[cfg(feature = "wasm")]
pub use beardy_squeeze_pro::{beardy_squeeze_pro_js, BeardySqueezeProJsResult};