pub mod nama;
pub use nama::{
    nama, nama_with_kernel, nama_into_slice, NamaInput, NamaOutput,
    NamaParams, NamaError, NamaData, NamaBuilder, NamaStream,
    // Batch API exports
    NamaBatchRange, NamaBatchOutput, NamaBatchBuilder, nama_batch_with_kernel,
};

// Python exports
#[cfg(feature = "python")]
pub use nama::{nama_py, nama_batch_py, NamaStreamPy};

// WASM exports
#[cfg(feature = "wasm")]
pub use nama::{nama_js, nama_batch_unified_js, nama_alloc, nama_free, nama_into};