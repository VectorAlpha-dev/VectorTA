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
pub mod marketefi_wrapper;
#[cfg(feature = "cuda")]
pub mod bandpass_wrapper;
#[cfg(feature = "cuda")]
pub mod adxr_wrapper;
#[cfg(feature = "cuda")]
pub mod aroon_wrapper;
#[cfg(feature = "cuda")]
pub mod donchian_wrapper;
#[cfg(feature = "cuda")]
pub mod qstick_wrapper;
#[cfg(feature = "cuda")]
pub mod rocr_wrapper;

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
pub use bandpass_wrapper::{CudaBandpass, CudaBandpassBatchResult, DeviceArrayF32Quad};
#[cfg(feature = "cuda")]
pub use adxr_wrapper::{CudaAdxr, CudaAdxrError};
#[cfg(feature = "cuda")]
pub use aroon_wrapper::{CudaAroon, CudaAroonError};
#[cfg(feature = "cuda")]
pub use donchian_wrapper::{CudaDonchian, CudaDonchianError};
#[cfg(feature = "cuda")]
pub use qstick_wrapper::{
    BatchKernelPolicy as QsBatchKernelPolicy, CudaQstick, CudaQstickError,
    CudaQstickPolicy, ManySeriesKernelPolicy as QsManySeriesKernelPolicy,
};
#[cfg(feature = "cuda")]
pub use rocr_wrapper::{CudaRocr, CudaRocrError};
#[cfg(feature = "cuda")]
pub use marketefi_wrapper::{CudaMarketefi, CudaMarketefiError};
#[cfg(feature = "cuda")]
pub mod oscillators;
#[cfg(feature = "cuda")]
pub mod wto_wrapper;
#[cfg(feature = "cuda")]
pub mod nadaraya_watson_envelope_wrapper;

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
pub use nadaraya_watson_envelope_wrapper::{CudaNwe, CudaNweError, DeviceNwePair};
#[cfg(feature = "cuda")]
pub mod wad_wrapper;
#[cfg(feature = "cuda")]
pub mod zscore_wrapper;
#[cfg(feature = "cuda")]
pub mod lpc_wrapper;
#[cfg(feature = "cuda")]
pub mod correl_hl_wrapper;
#[cfg(feature = "cuda")]
pub mod efi_wrapper;
#[cfg(feature = "cuda")]
pub mod kurtosis_wrapper;
#[cfg(feature = "cuda")]
pub mod safezonestop_wrapper;
#[cfg(feature = "cuda")]
pub mod stddev_wrapper;
#[cfg(feature = "cuda")]
pub mod vosc_wrapper;

#[cfg(feature = "cuda")]
pub use oscillators::{CudaDecOsc, CudaDecOscError};
#[cfg(feature = "cuda")]
pub use oscillators::{CudaFisher, CudaFisherError};
#[cfg(feature = "cuda")]
pub use oscillators::{CudaIftRsi, CudaIftRsiError};
#[cfg(feature = "cuda")]
pub use oscillators::{CudaMfi, CudaMfiError};

#[cfg(feature = "cuda")]
pub use moving_averages::{
    CudaBuffAverages, CudaBuffAveragesError, CudaFrama, CudaFramaError, CudaHma, CudaHmaError,
    CudaLinreg, CudaLinregError, CudaNma, CudaNmaError, CudaSma, CudaSmaError, CudaSuperSmoother,
    CudaSuperSmootherError, CudaTrendflex, CudaTrendflexError, CudaVolumeAdjustedMa,
    CudaVolumeAdjustedMaError, CudaVpwma, CudaVpwmaError, CudaZlema, CudaZlemaError, CudaTsf,
    CudaTsfError,
};
#[cfg(feature = "cuda")]
pub use wad_wrapper::{CudaWad, CudaWadError};
#[cfg(feature = "cuda")]
pub use zscore_wrapper::{CudaZscore, CudaZscoreError};
#[cfg(feature = "cuda")]
pub use correl_hl_wrapper::{CudaCorrelHl, CudaCorrelHlError};
#[cfg(feature = "cuda")]
pub use efi_wrapper::{CudaEfi, CudaEfiError};
#[cfg(feature = "cuda")]
pub use kurtosis_wrapper::{CudaKurtosis, CudaKurtosisError};
#[cfg(feature = "cuda")]
pub use lpc_wrapper::{
    BatchKernelPolicy as LpcBatchKernelPolicy, CudaLpc, CudaLpcError, CudaLpcPolicy,
    ManySeriesKernelPolicy as LpcManySeriesKernelPolicy,
};
#[cfg(feature = "cuda")]
pub use safezonestop_wrapper::{CudaSafeZoneStop, CudaSafeZoneStopError};
#[cfg(feature = "cuda")]
pub use stddev_wrapper::{CudaStddev, CudaStddevError};
#[cfg(feature = "cuda")]
pub use vosc_wrapper::{
    BatchKernelPolicy as VoscBatchKernelPolicy, CudaVosc, CudaVoscError,
    CudaVoscPolicy, ManySeriesKernelPolicy as VoscManySeriesKernelPolicy,
};

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
