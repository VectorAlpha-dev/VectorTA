#[cfg(feature = "cuda")]
pub mod alma_wrapper;
#[cfg(feature = "cuda")]
pub mod dema_wrapper;
pub mod dma_wrapper;
pub mod edcf_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_itrend_wrapper;
pub mod ehlers_kama_wrapper;
pub mod ehlers_pma_wrapper;
#[cfg(feature = "cuda")]
pub mod pma_wrapper;
pub mod ehma_wrapper;
#[cfg(feature = "cuda")]
pub mod ema_wrapper;
pub mod fwma_wrapper;
pub mod gaussian_wrapper;
#[cfg(feature = "cuda")]
pub mod highpass2_wrapper;
pub mod hwma_wrapper;
pub mod jma_wrapper;
#[cfg(feature = "cuda")]
pub mod jsa_wrapper;
pub mod maaq_wrapper;
pub mod mama_wrapper;
#[cfg(feature = "cuda")]
pub mod mwdx_wrapper;
pub mod pwma_wrapper;
pub mod reflex_wrapper;
#[cfg(feature = "cuda")]
pub mod sama_wrapper;
pub mod smma_wrapper;
pub mod sqwma_wrapper;
#[cfg(feature = "cuda")]
pub mod srwma_wrapper;
pub mod swma_wrapper;
pub mod tema_wrapper;
#[cfg(feature = "cuda")]
pub mod tilson_wrapper;
pub mod trima_wrapper;
pub mod uma_wrapper;
#[cfg(feature = "cuda")]
pub mod vama_wrapper;
pub mod vwap_wrapper;
pub mod vwma_wrapper;
#[cfg(feature = "cuda")]
pub mod wilders_wrapper;

pub mod buff_averages_wrapper;
#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
pub mod cwma_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_ecema_wrapper;
#[cfg(feature = "cuda")]
pub mod epma_wrapper;
pub mod frama_wrapper;
#[cfg(feature = "cuda")]
pub mod highpass_wrapper;
pub mod hma_wrapper;
#[cfg(feature = "cuda")]
pub mod kama_wrapper;
pub mod linreg_wrapper;
#[cfg(feature = "cuda")]
pub mod ma_selector;
#[cfg(feature = "cuda")]
pub mod nama_wrapper;
pub mod nma_wrapper;
#[cfg(feature = "cuda")]
pub mod sinwma_wrapper;
pub mod sma_wrapper;
#[cfg(feature = "cuda")]
pub mod supersmoother_3_pole_wrapper;
pub mod supersmoother_wrapper;
#[cfg(feature = "cuda")]
pub mod tradjema_wrapper;
pub mod trendflex_wrapper;
#[cfg(feature = "cuda")]
pub mod volume_adjusted_ma_wrapper;
pub mod vpwma_wrapper;
#[cfg(feature = "cuda")]
pub mod wma_wrapper;
pub mod zlema_wrapper;
#[cfg(feature = "cuda")]
pub mod ott_wrapper;
#[cfg(feature = "cuda")]
pub mod tsf_wrapper;

pub use alma_wrapper::{CudaAlma, DeviceArrayF32};
pub use buff_averages_wrapper::{CudaBuffAverages, CudaBuffAveragesError};
#[cfg(feature = "cuda")]
pub use cwma_wrapper::{
    BatchKernelPolicy, BatchThreadsPerOutput, CudaCwma, CudaCwmaPolicy, ManySeriesKernelPolicy,
};
#[cfg(feature = "cuda")]
pub use dema_wrapper::{CudaDema, CudaDemaError};
pub use dma_wrapper::CudaDma;
pub use edcf_wrapper::CudaEdcf;
#[cfg(feature = "cuda")]
pub use ehlers_ecema_wrapper::CudaEhlersEcema;
#[cfg(feature = "cuda")]
pub use ehlers_itrend_wrapper::{
    BatchKernelPolicy as EhlersItrendBatchKernelPolicy,
    BatchThreadsPerOutput as EhlersItrendBatchThreadsPerOutput, CudaEhlersITrend,
    CudaEhlersITrendError, CudaEhlersITrendPolicy,
    ManySeriesKernelPolicy as EhlersItrendManySeriesKernelPolicy,
};
pub use ehlers_kama_wrapper::CudaEhlersKama;
pub use ehlers_pma_wrapper::{CudaEhlersPma, DeviceEhlersPmaPair};
#[cfg(feature = "cuda")]
pub use pma_wrapper::{benches as pma_benches, CudaPma, CudaPmaError, DevicePmaPair};
pub use ehma_wrapper::CudaEhma;
#[cfg(feature = "cuda")]
pub use ema_wrapper::{CudaEma, CudaEmaError};
#[cfg(feature = "cuda")]
pub use epma_wrapper::CudaEpma;
pub use frama_wrapper::{CudaFrama, CudaFramaError};
pub use fwma_wrapper::CudaFwma;
pub use gaussian_wrapper::CudaGaussian;
#[cfg(feature = "cuda")]
pub use highpass2_wrapper::{CudaHighPass2, CudaHighPass2Error};
#[cfg(feature = "cuda")]
pub use highpass_wrapper::CudaHighpass;
pub use hma_wrapper::{CudaHma, CudaHmaError};
pub use hwma_wrapper::CudaHwma;
pub use jma_wrapper::CudaJma;
#[cfg(feature = "cuda")]
pub use jsa_wrapper::{CudaJsa, CudaJsaError};
#[cfg(feature = "cuda")]
pub use kama_wrapper::CudaKama;
pub use linreg_wrapper::{CudaLinreg, CudaLinregError};
#[cfg(feature = "cuda")]
pub use ma_selector::{CudaMaData, CudaMaSelector, CudaMaSelectorError};
pub use maaq_wrapper::CudaMaaq;
pub use mama_wrapper::{CudaMama, DeviceMamaPair};
#[cfg(feature = "cuda")]
pub use mwdx_wrapper::{CudaMwdx, CudaMwdxError};
#[cfg(feature = "cuda")]
pub use nama_wrapper::CudaNama;
pub use nma_wrapper::{CudaNma, CudaNmaError};
pub use pwma_wrapper::CudaPwma;
pub use reflex_wrapper::CudaReflex;
#[cfg(feature = "cuda")]
pub use sama_wrapper::{CudaSama, CudaSamaError};
#[cfg(feature = "cuda")]
pub use sinwma_wrapper::CudaSinwma;
pub use sma_wrapper::{CudaSma, CudaSmaError};
pub use smma_wrapper::CudaSmma;
pub use sqwma_wrapper::CudaSqwma;
#[cfg(feature = "cuda")]
pub use srwma_wrapper::{CudaSrwma, CudaSrwmaError};
#[cfg(feature = "cuda")]
pub use supersmoother_3_pole_wrapper::CudaSupersmoother3Pole;
pub use supersmoother_wrapper::{CudaSuperSmoother, CudaSuperSmootherError};
pub use swma_wrapper::CudaSwma;
pub use tema_wrapper::CudaTema;
#[cfg(feature = "cuda")]
pub use tilson_wrapper::{CudaTilson, CudaTilsonError};
#[cfg(feature = "cuda")]
pub use tradjema_wrapper::CudaTradjema;
pub use trendflex_wrapper::{CudaTrendflex, CudaTrendflexError};
pub use trima_wrapper::CudaTrima;
pub use uma_wrapper::{
    BatchKernelPolicy as UmaBatchKernelPolicy, CudaUma, CudaUmaPolicy,
    ManySeriesKernelPolicy as UmaManySeriesKernelPolicy,
};
#[cfg(feature = "cuda")]
pub use vama_wrapper::{
    BatchKernelPolicy as VamaBatchKernelPolicy, CudaVama, CudaVamaError, CudaVamaPolicy,
    ManySeriesKernelPolicy as VamaManySeriesKernelPolicy,
};
#[cfg(feature = "cuda")]
pub use volume_adjusted_ma_wrapper::{
    CudaVama as CudaVolumeAdjustedMa, CudaVamaError as CudaVolumeAdjustedMaError,
};
pub use vpwma_wrapper::{CudaVpwma, CudaVpwmaError};
pub use vwap_wrapper::CudaVwap;
pub use vwma_wrapper::CudaVwma;
#[cfg(feature = "cuda")]
pub use wilders_wrapper::{CudaWilders, CudaWildersError};
#[cfg(feature = "cuda")]
pub use wma_wrapper::CudaWma;
pub use zlema_wrapper::{CudaZlema, CudaZlemaError};
#[cfg(feature = "cuda")]
pub use ott_wrapper::{benches as ott_benches, CudaOtt, CudaOttError};
pub use tsf_wrapper::{CudaTsf, CudaTsfError};
