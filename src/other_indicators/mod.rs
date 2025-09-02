pub mod avsl;
pub use avsl::{
    avsl, avsl_with_kernel, AvslInput, AvslOutput, AvslParams, AvslError, 
    AvslData, AvslBuilder, avsl_into_slice,
    // Batch API exports
    AvslBatchRange, AvslBatchBuilder, AvslBatchOutput, avsl_batch_with_kernel,
};

// Python exports
#[cfg(feature = "python")]
pub use avsl::{avsl_py, avsl_batch_py};