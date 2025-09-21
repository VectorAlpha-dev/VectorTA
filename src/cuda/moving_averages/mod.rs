#[cfg(feature = "cuda")]
pub mod alma_wrapper;
#[cfg(feature = "cuda")]
pub mod dema_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_itrend_wrapper;
#[cfg(feature = "cuda")]
pub mod ema_wrapper;
#[cfg(feature = "cuda")]
pub mod highpass2_wrapper;
#[cfg(feature = "cuda")]
pub mod jsa_wrapper;
#[cfg(feature = "cuda")]
pub mod mwdx_wrapper;
#[cfg(feature = "cuda")]
pub mod sama_wrapper;
#[cfg(feature = "cuda")]
pub mod srwma_wrapper;
#[cfg(feature = "cuda")]
pub mod tilson_wrapper;
#[cfg(feature = "cuda")]
pub mod vama_wrapper;
#[cfg(feature = "cuda")]
pub mod wilders_wrapper;

#[cfg(feature = "cuda")]
pub use alma_wrapper::{CudaAlma, DeviceArrayF32};
#[cfg(feature = "cuda")]
pub use dema_wrapper::{CudaDema, CudaDemaError};
#[cfg(feature = "cuda")]
pub use ehlers_itrend_wrapper::{CudaEhlersITrend, CudaEhlersITrendError};
#[cfg(feature = "cuda")]
pub use ema_wrapper::{CudaEma, CudaEmaError};
#[cfg(feature = "cuda")]
pub use highpass2_wrapper::{CudaHighPass2, CudaHighPass2Error};
#[cfg(feature = "cuda")]
pub use jsa_wrapper::{CudaJsa, CudaJsaError};
#[cfg(feature = "cuda")]
pub use mwdx_wrapper::{CudaMwdx, CudaMwdxError};
#[cfg(feature = "cuda")]
pub use sama_wrapper::{CudaSama, CudaSamaError};
#[cfg(feature = "cuda")]
pub use srwma_wrapper::{CudaSrwma, CudaSrwmaError};
#[cfg(feature = "cuda")]
pub use tilson_wrapper::{CudaTilson, CudaTilsonError};
#[cfg(feature = "cuda")]
pub use vama_wrapper::{CudaVama, CudaVamaError};
#[cfg(feature = "cuda")]
pub use wilders_wrapper::{CudaWilders, CudaWildersError};
