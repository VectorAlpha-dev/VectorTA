#![cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]

pub use my_project::cuda::moving_averages::vram_ma::{supports_vram_kernel_ma, VramMaComputer};
pub use my_project::cuda::moving_averages::vram_ma::VramMaInputs;
