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