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

#[cfg(feature = "cuda")]#[cfg(feature = "cuda")]
pub mod cwma_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_ecema_wrapper;
#[cfg(feature = "cuda")]
pub mod epma_wrapper;
#[cfg(feature = "cuda")]
pub mod highpass_wrapper;
#[cfg(feature = "cuda")]
pub mod kama_wrapper;
#[cfg(feature = "cuda")]
pub mod nama_wrapper;
#[cfg(feature = "cuda")]
pub mod sinwma_wrapper;
#[cfg(feature = "cuda")]
pub mod supersmoother_3_pole_wrapper;
#[cfg(feature = "cuda")]
pub mod tradjema_wrapper;
#[cfg(feature = "cuda")]
pub mod volume_adjusted_ma_wrapper;
#[cfg(feature = "cuda")]
pub mod wma_wrapper;
pub mod buff_averages_wrapper;
pub mod frama_wrapper;
pub mod hma_wrapper;
pub mod linreg_wrapper;
pub mod nma_wrapper;
pub mod sma_wrapper;
pub mod supersmoother_wrapper;
pub mod trendflex_wrapper;
pub mod vpwma_wrapper;
pub mod zlema_wrapper;

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
#[cfg(feature = "cuda")]
pub use cwma_wrapper::CudaCwma;
#[cfg(feature = "cuda")]
pub use ehlers_ecema_wrapper::CudaEhlersEcema;
#[cfg(feature = "cuda")]
pub use epma_wrapper::CudaEpma;
#[cfg(feature = "cuda")]
pub use highpass_wrapper::CudaHighpass;
#[cfg(feature = "cuda")]
pub use kama_wrapper::CudaKama;
#[cfg(feature = "cuda")]
pub use nama_wrapper::CudaNama;
#[cfg(feature = "cuda")]
pub use sinwma_wrapper::CudaSinwma;
#[cfg(feature = "cuda")]
pub use supersmoother_3_pole_wrapper::CudaSupersmoother3Pole;
#[cfg(feature = "cuda")]
pub use tradjema_wrapper::CudaTradjema;
#[cfg(feature = "cuda")]
pub use volume_adjusted_ma_wrapper::CudaVama;
#[cfg(feature = "cuda")]
pub use wma_wrapper::CudaWma;
pub use buff_averages_wrapper::{CudaBuffAverages, CudaBuffAveragesError};
pub use frama_wrapper::{CudaFrama, CudaFramaError};
pub use hma_wrapper::{CudaHma, CudaHmaError};
pub use linreg_wrapper::{CudaLinreg, CudaLinregError};
pub use nma_wrapper::{CudaNma, CudaNmaError};
pub use sma_wrapper::{CudaSma, CudaSmaError};
pub use supersmoother_wrapper::{CudaSuperSmoother, CudaSuperSmootherError};
pub use trendflex_wrapper::{CudaTrendflex, CudaTrendflexError};
pub use vpwma_wrapper::{CudaVpwma, CudaVpwmaError};
pub use zlema_wrapper::{CudaZlema, CudaZlemaError};
