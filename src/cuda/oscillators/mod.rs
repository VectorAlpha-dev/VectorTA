#[cfg(feature = "cuda")]
pub mod willr_wrapper;

#[cfg(feature = "cuda")]
pub use willr_wrapper::{CudaWillr, CudaWillrError};
