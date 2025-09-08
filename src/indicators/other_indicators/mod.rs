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

pub mod ttm_squeeze;
pub use ttm_squeeze::{
    ttm_squeeze, ttm_squeeze_with_kernel, ttm_squeeze_into_slices,
    ttm_squeeze_batch_with_kernel,
    TtmSqueezeInput, TtmSqueezeOutput, TtmSqueezeParams, TtmSqueezeData,
    TtmSqueezeBuilder, TtmSqueezeStream, TtmSqueezeError,
    TtmSqueezeBatchRange, TtmSqueezeBatchOutput, TtmSqueezeBatchBuilder,
};

#[cfg(feature = "python")]
pub use ttm_squeeze::{ttm_squeeze_py, ttm_squeeze_batch_py, TtmSqueezeStreamPy};

#[cfg(feature = "wasm")]
pub use ttm_squeeze::{
    ttm_squeeze_js, ttm_squeeze_into_js, ttm_squeeze_into_js_ptrs,
    ttm_squeeze_alloc, ttm_squeeze_free, ttm_squeeze_batch_unified_js,
    TtmSqueezeJsResult, TtmSqueezeBatchConfig, TtmSqueezeBatchJsOutput,
};

pub mod mod_god_mode;
pub use mod_god_mode::{
    mod_god_mode, mod_god_mode_with_kernel, mod_god_mode_auto, mod_god_mode_into_slices,
    mod_god_mode_batch_with_kernel,
    ModGodModeInput, ModGodModeOutput, ModGodModeParams, ModGodModeData,
    ModGodModeMode, ModGodModeError,
    ModGodModeBuilder, ModGodModeStream, ModGodModeBatchBuilder,
    ModGodModeBatchRange, ModGodModeBatchOutput,
};

#[cfg(feature = "python")]
pub use mod_god_mode::{mod_god_mode_py, mod_god_mode_batch_py, ModGodModeStreamPy};

#[cfg(feature = "wasm")]
pub use mod_god_mode::{
    mod_god_mode_wasm, mod_god_mode_alloc, mod_god_mode_free, mod_god_mode_into,
    mod_god_mode_js_flat, ModGodModeJsFlat
};