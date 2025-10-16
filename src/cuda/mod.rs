//! CUDA integration scaffolding (cust-based)
//!
//! This module is built only when the `cuda` feature is enabled. It provides
//! runtime detection helpers and submodules for GPU-accelerated indicators.

#[cfg(feature = "cuda")]
pub mod bench;
#[cfg(feature = "cuda")]
pub mod moving_averages;
#[cfg(feature = "cuda")]
pub mod wavetrend;
#[cfg(feature = "cuda")]
pub mod wclprice;
#[cfg(feature = "cuda")]
pub mod di_wrapper;
#[cfg(feature = "cuda")]
pub mod atr_wrapper;
#[cfg(feature = "cuda")]
pub mod chande_wrapper;
#[cfg(feature = "cuda")]
pub mod cvi_wrapper;
#[cfg(feature = "cuda")]
pub mod keltner_wrapper;

#[cfg(feature = "cuda")]
pub use bench::{CudaBenchScenario, CudaBenchState};
#[cfg(feature = "cuda")]
pub use moving_averages::{
    CudaAlma, CudaDma, CudaEhlersPma, CudaGaussian, CudaJma, CudaMama, CudaReflex, CudaSqwma,
    CudaTema, CudaVwma, DeviceArrayF32, DeviceEhlersPmaPair, DeviceMamaPair,
};
#[cfg(feature = "cuda")]
pub use wclprice::CudaWclprice;
#[cfg(feature = "cuda")]
pub use di_wrapper::{CudaDi, CudaDiError, DeviceArrayF32Pair};
#[cfg(feature = "cuda")]
pub use atr_wrapper::CudaAtr;
#[cfg(feature = "cuda")]
pub use chande_wrapper::CudaChande;
#[cfg(feature = "cuda")]
pub use cvi_wrapper::{CudaCvi, CudaCviError};
#[cfg(feature = "cuda")]
pub use keltner_wrapper::{CudaKeltner, CudaKeltnerError, CudaKeltnerBatchResult, DeviceKeltnerTriplet};
#[cfg(feature = "cuda")]
pub mod oscillators;
#[cfg(feature = "cuda")]
pub mod wto_wrapper;
#[cfg(feature = "cuda")]
pub mod dvdiqqe_wrapper;
#[cfg(feature = "cuda")]
pub mod er_wrapper;
#[cfg(feature = "cuda")]
pub mod pfe_wrapper;
#[cfg(feature = "cuda")]
pub mod nvi_wrapper;
#[cfg(feature = "cuda")]
pub mod pvi_wrapper;
#[cfg(feature = "cuda")]
pub mod supertrend_wrapper;
#[cfg(feature = "cuda")]
pub mod ttm_trend_wrapper;
#[cfg(feature = "cuda")]
pub mod vpt_wrapper;

#[cfg(feature = "cuda")]
pub use moving_averages::cwma_wrapper::CudaCwma;
#[cfg(feature = "cuda")]
pub use moving_averages::ehlers_ecema_wrapper::CudaEhlersEcema;
#[cfg(feature = "cuda")]
pub use moving_averages::epma_wrapper::CudaEpma;
#[cfg(feature = "cuda")]
pub use moving_averages::highpass_wrapper::CudaHighpass;
#[cfg(feature = "cuda")]
pub use moving_averages::kama_wrapper::CudaKama;
#[cfg(feature = "cuda")]
pub use moving_averages::nama_wrapper::CudaNama;
#[cfg(feature = "cuda")]
pub use moving_averages::sinwma_wrapper::CudaSinwma;
#[cfg(feature = "cuda")]
pub use moving_averages::supersmoother_3_pole_wrapper::CudaSupersmoother3Pole;
#[cfg(feature = "cuda")]
pub use moving_averages::tradjema_wrapper::CudaTradjema;
#[cfg(feature = "cuda")]
pub use moving_averages::wma_wrapper::CudaWma;
#[cfg(feature = "cuda")]
pub use wto_wrapper::{CudaWto, CudaWtoBatchResult, DeviceArrayF32Triplet};
#[cfg(feature = "cuda")]
pub use dvdiqqe_wrapper::{CudaDvdiqqe, CudaDvdiqqeError};
#[cfg(feature = "cuda")]
pub use er_wrapper::{CudaEr, CudaErError};
#[cfg(feature = "cuda")]
pub use pfe_wrapper::{CudaPfe, CudaPfeError};
#[cfg(feature = "cuda")]
pub use nvi_wrapper::{CudaNvi, CudaNviError};
#[cfg(feature = "cuda")]
pub use pvi_wrapper::{CudaPvi, CudaPviError};
#[cfg(feature = "cuda")]
pub use supertrend_wrapper::{CudaSupertrend, CudaSupertrendError};
#[cfg(feature = "cuda")]
pub use ttm_trend_wrapper::{CudaTtmTrend, CudaTtmTrendError};
#[cfg(feature = "cuda")]
pub use vpt_wrapper::{CudaVpt, CudaVptError};
#[cfg(feature = "cuda")]
pub mod wad_wrapper;
#[cfg(feature = "cuda")]
pub mod zscore_wrapper;
#[cfg(feature = "cuda")]
pub mod medium_ad_wrapper;

#[cfg(feature = "cuda")]
pub use moving_averages::{
    CudaBuffAverages, CudaBuffAveragesError, CudaFrama, CudaFramaError, CudaHma, CudaHmaError,
    CudaLinreg, CudaLinregError, CudaLinregIntercept, CudaLinregInterceptError, CudaNma, CudaNmaError, CudaSma, CudaSmaError, CudaSuperSmoother,
    CudaSuperSmootherError, CudaTrendflex, CudaTrendflexError, CudaVolumeAdjustedMa,
    CudaVolumeAdjustedMaError, CudaVpwma, CudaVpwmaError, CudaZlema, CudaZlemaError, CudaVidya, CudaVidyaError,
};
#[cfg(feature = "cuda")]
pub use wad_wrapper::{CudaWad, CudaWadError};
#[cfg(feature = "cuda")]
pub use zscore_wrapper::{CudaZscore, CudaZscoreError};
#[cfg(feature = "cuda")]
pub use medium_ad_wrapper::{CudaMediumAd, CudaMediumAdError};
#[cfg(feature = "cuda")]
pub use oscillators::adosc_wrapper::{CudaAdosc, CudaAdoscError};
#[cfg(feature = "cuda")]
pub use oscillators::ao_wrapper::{CudaAo, CudaAoError};
#[cfg(feature = "cuda")]
pub use oscillators::coppock_wrapper::{CudaCoppock, CudaCoppockError};
#[cfg(feature = "cuda")]
pub use oscillators::gatorosc_wrapper::{CudaGatorOsc, CudaGatorOscError};
#[cfg(feature = "cuda")]
pub use oscillators::macd_wrapper::{CudaMacd, CudaMacdError};
#[cfg(feature = "cuda")]
pub use chande_wrapper::CudaChandeError;

/// Returns true if a CUDA device is available and the driver API can be initialized.
#[inline]
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        use cust::{device::Device, prelude::CudaFlags};
        // Initialize the CUDA driver and query devices. Keep this defensive so
        // it never panics when CUDA is missing.
        if cust::init(CudaFlags::empty()).is_err() {
            return false;
        }
        match Device::num_devices() {
            Ok(n) => n > 0,
            Err(_) => false,
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Returns the number of CUDA devices available (0 on error or when disabled).
#[inline]
pub fn cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        use cust::{device::Device, prelude::CudaFlags};
        if cust::init(CudaFlags::empty()).is_err() {
            return 0;
        }
        match Device::num_devices() {
            Ok(n) => n as usize,
            Err(_) => 0,
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}
