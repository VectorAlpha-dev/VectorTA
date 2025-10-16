#[cfg(feature = "cuda")]
pub mod willr_wrapper;
pub mod acosc_wrapper;
pub mod aroonosc_wrapper;
pub mod cfo_wrapper;
pub mod dpo_wrapper;
pub mod fosc_wrapper;
pub mod kvo_wrapper;
pub mod lrsi_wrapper;
pub mod tsi_wrapper;
pub mod ppo_wrapper;
pub mod stoch_wrapper;

#[cfg(feature = "cuda")]
pub use willr_wrapper::{CudaWillr, CudaWillrError};
pub use acosc_wrapper::{CudaAcosc, CudaAcoscError};
pub use aroonosc_wrapper::{CudaAroonOsc, CudaAroonOscError};
pub use cfo_wrapper::{CudaCfo, CudaCfoError};
pub use dpo_wrapper::{CudaDpo, CudaDpoError};
pub use fosc_wrapper::{CudaFosc, CudaFoscError};
pub use kvo_wrapper::{CudaKvo, CudaKvoError};
pub use lrsi_wrapper::{CudaLrsi, CudaLrsiError};
pub use tsi_wrapper::{CudaTsi, CudaTsiError};
pub use ppo_wrapper::{CudaPpo, CudaPpoError};
pub use stoch_wrapper::{CudaStoch, CudaStochError};
