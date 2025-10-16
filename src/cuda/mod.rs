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
pub mod medprice_wrapper;
#[cfg(feature = "cuda")]
pub mod eri_wrapper;
#[cfg(feature = "cuda")]
pub mod adx_wrapper;
#[cfg(feature = "cuda")]
pub mod avsl_wrapper;
#[cfg(feature = "cuda")]
pub mod dm_wrapper;
#[cfg(feature = "cuda")]
pub mod dx_wrapper;

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
pub use medprice_wrapper::CudaMedprice;
#[cfg(feature = "cuda")]
pub use eri_wrapper::{CudaEri, CudaEriError};
#[cfg(feature = "cuda")]
pub use adx_wrapper::{CudaAdx, CudaAdxError};
#[cfg(feature = "cuda")]
pub use avsl_wrapper::{CudaAvsl, CudaAvslError};
#[cfg(feature = "cuda")]
pub use dm_wrapper::{CudaDm, CudaDmError};
#[cfg(feature = "cuda")]
pub use dx_wrapper::{CudaDx, CudaDxError};
#[cfg(feature = "cuda")]
pub mod oscillators;
#[cfg(feature = "cuda")]
pub use oscillators::msw_wrapper::{CudaMsw, CudaMswError};
#[cfg(feature = "cuda")]
pub use oscillators::qqe_wrapper::{CudaQqe, CudaQqeError};
#[cfg(feature = "cuda")]
pub use oscillators::rvi_wrapper::{CudaRvi, CudaRviError};
#[cfg(feature = "cuda")]
pub use oscillators::stc_wrapper::{CudaStc, CudaStcError};
#[cfg(feature = "cuda")]
pub mod wto_wrapper;
#[cfg(feature = "cuda")]
pub mod vwmacd_wrapper;

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
pub use vwmacd_wrapper::{CudaVwmacd, CudaVwmacdError};
#[cfg(feature = "cuda")]
pub mod wad_wrapper;
#[cfg(feature = "cuda")]
pub mod zscore_wrapper;
#[cfg(feature = "cuda")]
pub mod chandelier_exit_wrapper;
#[cfg(feature = "cuda")]
pub mod damiani_volatmeter_wrapper;
#[cfg(feature = "cuda")]
pub mod halftrend_wrapper;
#[cfg(feature = "cuda")]
pub mod obv_wrapper;
#[cfg(feature = "cuda")]
pub mod pivot_wrapper;
#[cfg(feature = "cuda")]
pub mod ui_wrapper;

#[cfg(feature = "cuda")]
pub use moving_averages::{
    CudaBuffAverages, CudaBuffAveragesError, CudaFrama, CudaFramaError, CudaHma, CudaHmaError,
    CudaLinreg, CudaLinregError, CudaNma, CudaNmaError, CudaSma, CudaSmaError, CudaSuperSmoother,
    CudaSuperSmootherError, CudaTrendflex, CudaTrendflexError, CudaVolumeAdjustedMa,
    CudaVolumeAdjustedMaError, CudaVpwma, CudaVpwmaError, CudaZlema, CudaZlemaError, CudaApo, CudaVlma,
    CudaLinearregSlope, CudaLinearregSlopeError,
};
#[cfg(feature = "cuda")]
pub use wad_wrapper::{CudaWad, CudaWadError};
#[cfg(feature = "cuda")]
pub use zscore_wrapper::{CudaZscore, CudaZscoreError};
#[cfg(feature = "cuda")]
pub use chandelier_exit_wrapper::{CudaChandelierExit, CudaCeError};
#[cfg(feature = "cuda")]
pub use damiani_volatmeter_wrapper::{CudaDamianiVolatmeter, CudaDamianiError};
#[cfg(feature = "cuda")]
pub use halftrend_wrapper::{CudaHalftrend, CudaHalftrendError};
#[cfg(feature = "cuda")]
pub use obv_wrapper::{CudaObv, CudaObvError};
#[cfg(feature = "cuda")]
pub use pivot_wrapper::{CudaPivot, CudaPivotError};
#[cfg(feature = "cuda")]
pub use ui_wrapper::{CudaUi, CudaUiError};

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
