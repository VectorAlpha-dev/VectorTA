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

pub mod vama;
pub use vama::{
    vama, vama_with_kernel, vama_into_slice, vama_batch_with_kernel,
    VamaInput, VamaOutput, VamaParams, VamaData, VamaError,
    VamaBuilder, VamaStream, VamaBatchRange, VamaBatchOutput, VamaBatchBuilder,
};

#[cfg(feature = "python")]
pub use vama::{vama_py, vama_batch_py, VamaStreamPy};

#[cfg(feature = "wasm")]
pub use vama::{
    vama_js, vama_alloc, vama_free, vama_into, 
    vama_batch_unified_js, VamaBatchConfig, VamaBatchJsOutput,
};