#[cfg(feature = "cuda")]
pub mod alma_wrapper;
#[cfg(feature = "cuda")]
pub mod apo_wrapper;
#[cfg(feature = "cuda")]
pub mod decycler_wrapper;
#[cfg(feature = "cuda")]
pub mod dema_wrapper;
pub mod dma_wrapper;
pub mod edcf_wrapper;
#[cfg(feature = "cuda")]
pub mod ehlers_itrend_wrapper;
pub mod ehlers_kama_wrapper;
pub mod ehlers_pma_wrapper;
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
#[cfg(feature = "cuda")]
pub mod otto_wrapper;
#[cfg(feature = "cuda")]
pub mod pma_wrapper;
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
pub mod cora_wave_wrapper;
#[cfg(feature = "cuda")]
pub mod correlation_cycle_wrapper;
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
#[cfg(feature = "cuda")]
pub mod linearreg_intercept_wrapper;
#[cfg(feature = "cuda")]
pub mod linearreg_slope_wrapper;
pub mod linreg_wrapper;
#[cfg(feature = "cuda")]
pub mod ma_selector;
#[cfg(feature = "cuda")]
pub mod mab_wrapper;
#[cfg(feature = "cuda")]
pub mod macz_wrapper;
#[cfg(feature = "cuda")]
pub mod nama_wrapper;
pub mod nma_wrapper;
#[cfg(feature = "cuda")]
pub mod ott_wrapper;
#[cfg(feature = "cuda")]
pub mod rsmk_wrapper;
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
pub mod trix_wrapper;
#[cfg(feature = "cuda")]
pub mod tsf_wrapper;
#[cfg(feature = "cuda")]
pub mod vidya_wrapper;
#[cfg(feature = "cuda")]
pub mod vlma_wrapper;
#[cfg(feature = "cuda")]
pub mod volume_adjusted_ma_wrapper;
pub mod vpwma_wrapper;
#[cfg(feature = "cuda")]
pub mod wclprice_wrapper;
#[cfg(feature = "cuda")]
pub mod wma_wrapper;
pub mod zlema_wrapper;

pub use alma_wrapper::{CudaAlma, DeviceArrayF32};
#[cfg(feature = "cuda")]
pub use apo_wrapper::{CudaApo, CudaApoError};
pub use buff_averages_wrapper::{CudaBuffAverages, CudaBuffAveragesError};
#[cfg(feature = "cuda")]
pub use cora_wave_wrapper::{CudaCoraWave, CudaCoraWaveError};
#[cfg(feature = "cuda")]
pub use correlation_cycle_wrapper::{
    BatchKernelPolicy as CorrelationCycleBatchKernelPolicy, CudaCorrelationCycle,
    CudaCorrelationCyclePolicy, DeviceCorrelationCycleQuad,
    ManySeriesKernelPolicy as CorrelationCycleManySeriesKernelPolicy,
};
#[cfg(feature = "cuda")]
pub use cwma_wrapper::{
    BatchKernelPolicy, BatchThreadsPerOutput, CudaCwma, CudaCwmaPolicy, ManySeriesKernelPolicy,
};
#[cfg(feature = "cuda")]
pub use decycler_wrapper::{CudaDecycler, CudaDecyclerError};
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
#[cfg(feature = "cuda")]
pub use linearreg_intercept_wrapper::{CudaLinregIntercept, CudaLinregInterceptError};
#[cfg(feature = "cuda")]
pub use linearreg_slope_wrapper::{CudaLinearregSlope, CudaLinearregSlopeError};
pub use linreg_wrapper::{CudaLinreg, CudaLinregError};
#[cfg(feature = "cuda")]
pub use ma_selector::{CudaMaData, CudaMaSelector, CudaMaSelectorError};
pub use maaq_wrapper::CudaMaaq;
#[cfg(feature = "cuda")]
pub use mab_wrapper::{CudaMab, CudaMabError};
pub use mama_wrapper::{CudaMama, DeviceMamaPair};
#[cfg(feature = "cuda")]
pub use mwdx_wrapper::{CudaMwdx, CudaMwdxError};
#[cfg(feature = "cuda")]
pub use nama_wrapper::CudaNama;
pub use nma_wrapper::{CudaNma, CudaNmaError};
#[cfg(feature = "cuda")]
pub use otto_wrapper::{CudaOtto, CudaOttoError, CudaOttoPolicy};
#[cfg(feature = "cuda")]
pub use pma_wrapper::{benches as pma_benches, CudaPma, CudaPmaError, DevicePmaPair};
pub use pwma_wrapper::CudaPwma;
pub use reflex_wrapper::CudaReflex;
#[cfg(feature = "cuda")]
pub use rsmk_wrapper::{CudaRsmk, CudaRsmkError};
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
pub use vidya_wrapper::{CudaVidya, CudaVidyaError};
#[cfg(feature = "cuda")]
pub use volume_adjusted_ma_wrapper::{
    CudaVama as CudaVolumeAdjustedMa, CudaVamaError as CudaVolumeAdjustedMaError,
};
pub use vpwma_wrapper::{CudaVpwma, CudaVpwmaError};
pub use vwap_wrapper::CudaVwap;
pub use vwma_wrapper::CudaVwma;
#[cfg(feature = "cuda")]
pub use wclprice_wrapper::CudaWclprice;
#[cfg(feature = "cuda")]
pub use wilders_wrapper::{CudaWilders, CudaWildersError};
#[cfg(feature = "cuda")]
pub use wma_wrapper::{CudaWma, CudaWmaError};
pub use zlema_wrapper::{CudaZlema, CudaZlemaError};
// (duplicate export removed)
#[cfg(feature = "cuda")]
pub use macz_wrapper::{CudaMacz, CudaMaczError};
#[cfg(feature = "cuda")]
pub use ott_wrapper::{benches as ott_benches, CudaOtt, CudaOttError};
#[cfg(feature = "cuda")]
pub use trix_wrapper::{CudaTrix, CudaTrixError};
pub use tsf_wrapper::{CudaTsf, CudaTsfError};
#[cfg(feature = "cuda")]
pub use vlma_wrapper::{CudaVlma, CudaVlmaError};
