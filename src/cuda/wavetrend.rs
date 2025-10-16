// File module that routes wavetrend CUDA wrapper located in `src/cuda/`.
// This replaces the former `src/cuda/wavetrend/mod.rs` directory module.

#[path = "wavetrend_wrapper.rs"]
pub mod wavetrend_wrapper;

pub use wavetrend_wrapper::{CudaWavetrend, CudaWavetrendBatch, CudaWavetrendError};

