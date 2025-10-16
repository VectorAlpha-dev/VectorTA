//! CUDA integration scaffolding (cust-based)
//!
//! This module is built only when the `cuda` feature is enabled. It provides
//! runtime detection helpers and submodules for GPU-accelerated indicators.

#[cfg(feature = "cuda")]
pub mod bench;
#[cfg(feature = "cuda")]
pub mod moving_averages;
#[cfg(feature = "cuda")]
pub mod alligator_wrapper;
#[cfg(feature = "cuda")]
pub mod wavetrend;
#[cfg(feature = "cuda")]
pub mod ad_wrapper;
#[cfg(feature = "cuda")]
pub mod alphatrend_wrapper;

#[cfg(feature = "cuda")]
pub use bench::{CudaBenchScenario, CudaBenchState};
#[cfg(feature = "cuda")]
pub use moving_averages::{
    CudaAlma, CudaDma, CudaEhlersPma, CudaGaussian, CudaJma, CudaMama, CudaReflex, CudaSqwma,
    CudaTema, CudaVwma, DeviceArrayF32, DeviceEhlersPmaPair, DeviceMamaPair,
};
#[cfg(feature = "cuda")]
pub use alligator_wrapper::{CudaAlligator, CudaAlligatorBatchResult, CudaAlligatorError, DeviceArrayF32Trio};
#[cfg(feature = "cuda")]
pub use moving_averages::rsmk_wrapper::{CudaRsmk, CudaRsmkError};
#[cfg(feature = "cuda")]
pub use moving_averages::wclprice_wrapper::CudaWclprice;
#[cfg(feature = "cuda")]
pub use ad_wrapper::{CudaAd, CudaAdError};
#[cfg(feature = "cuda")]
pub use alphatrend_wrapper::{CudaAlphaTrend, CudaAlphaTrendError};
#[cfg(feature = "cuda")]
pub mod oscillators;
#[cfg(feature = "cuda")]
pub mod wto_wrapper;
#[cfg(feature = "cuda")]
pub mod bollinger_bands_wrapper;

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
pub mod wad_wrapper;
#[cfg(feature = "cuda")]
pub mod zscore_wrapper;
#[cfg(feature = "cuda")]
pub mod bollinger_bands_width_wrapper;
#[cfg(feature = "cuda")]
pub mod cksp_wrapper;
#[cfg(feature = "cuda")]
pub mod deviation_wrapper;
#[cfg(feature = "cuda")]
pub mod var_wrapper;
#[cfg(feature = "cuda")]
pub mod emd_wrapper;
#[cfg(feature = "cuda")]
pub mod kaufmanstop_wrapper;
#[cfg(feature = "cuda")]
pub mod mass_wrapper;
#[cfg(feature = "cuda")]
pub mod minmax_wrapper;
#[cfg(feature = "cuda")]
pub mod natr_wrapper;
#[cfg(feature = "cuda")]
pub mod range_filter_wrapper;
#[cfg(feature = "cuda")]
pub mod sar_wrapper;
#[cfg(feature = "cuda")]
pub mod voss_wrapper;
#[cfg(feature = "cuda")]
pub mod devstop_wrapper;
#[cfg(feature = "cuda")]
pub mod fvg_trailing_stop_wrapper;
#[cfg(feature = "cuda")]
pub mod mean_ad_wrapper;
#[cfg(feature = "cuda")]
pub mod mod_god_mode_wrapper;
#[cfg(feature = "cuda")]
pub mod net_myrsi_wrapper;
#[cfg(feature = "cuda")]
pub mod percentile_nearest_rank_wrapper;
#[cfg(feature = "cuda")]
pub mod prb_wrapper;
#[cfg(feature = "cuda")]
pub mod vi_wrapper;
#[cfg(feature = "cuda")]
pub mod vpci_wrapper;

#[cfg(feature = "cuda")]
pub use moving_averages::{
    CudaBuffAverages, CudaBuffAveragesError, CudaFrama, CudaFramaError, CudaHma, CudaHmaError,
    CudaLinreg, CudaLinregError, CudaNma, CudaNmaError, CudaSma, CudaSmaError, CudaSuperSmoother,
    CudaSuperSmootherError, CudaTrendflex, CudaTrendflexError, CudaVolumeAdjustedMa,
    CudaVolumeAdjustedMaError, CudaVpwma, CudaVpwmaError, CudaZlema, CudaZlemaError,
};
#[cfg(feature = "cuda")]
pub use oscillators::cfo_wrapper::{CudaCfo, CudaCfoError};
#[cfg(feature = "cuda")]
pub use oscillators::dpo_wrapper::{CudaDpo, CudaDpoError};
#[cfg(feature = "cuda")]
pub use oscillators::fosc_wrapper::{CudaFosc, CudaFoscError};
#[cfg(feature = "cuda")]
pub use oscillators::kvo_wrapper::{CudaKvo, CudaKvoError};
#[cfg(feature = "cuda")]
pub use oscillators::tsi_wrapper::{CudaTsi, CudaTsiError};
#[cfg(feature = "cuda")]
pub use oscillators::ppo_wrapper::{CudaPpo, CudaPpoError};
#[cfg(feature = "cuda")]
pub use wad_wrapper::{CudaWad, CudaWadError};
#[cfg(feature = "cuda")]
pub use zscore_wrapper::{CudaZscore, CudaZscoreError};
#[cfg(feature = "cuda")]
pub use bollinger_bands_width_wrapper::{CudaBbw, CudaBbwError};
#[cfg(feature = "cuda")]
pub use cksp_wrapper::{CudaCksp, CudaCkspError};
#[cfg(feature = "cuda")]
pub use deviation_wrapper::{CudaDeviation, CudaDeviationError};
#[cfg(feature = "cuda")]
pub use var_wrapper::{CudaVar, CudaVarError};
#[cfg(feature = "cuda")]
pub use emd_wrapper::{CudaEmd, CudaEmdBatchResult, CudaEmdError, DeviceArrayF32Triple};
#[cfg(feature = "cuda")]
pub use kaufmanstop_wrapper::{CudaKaufmanstop, CudaKaufmanstopError};
#[cfg(feature = "cuda")]
pub use mass_wrapper::{CudaMass, CudaMassError};
#[cfg(feature = "cuda")]
pub use minmax_wrapper::{CudaMinmax, CudaMinmaxError};
#[cfg(feature = "cuda")]
pub use natr_wrapper::{CudaNatr, CudaNatrError};
#[cfg(feature = "cuda")]
pub use range_filter_wrapper::{CudaRangeFilter, CudaRangeFilterError, DeviceRangeFilterTrio};
#[cfg(feature = "cuda")]
pub use sar_wrapper::{CudaSar, CudaSarError};
#[cfg(feature = "cuda")]
pub use voss_wrapper::{CudaVoss, CudaVossError};
#[cfg(feature = "cuda")]
pub use devstop_wrapper::{CudaDevStop, CudaDevStopError};
#[cfg(feature = "cuda")]
pub use fvg_trailing_stop_wrapper::{CudaFvgTs, CudaFvgTsError};
#[cfg(feature = "cuda")]
pub use mean_ad_wrapper::{CudaMeanAd, CudaMeanAdError};
#[cfg(feature = "cuda")]
pub use mod_god_mode_wrapper::{CudaModGodMode, CudaModGodModeBatchResult};
#[cfg(feature = "cuda")]
pub use net_myrsi_wrapper::{CudaNetMyrsi, CudaNetMyrsiError};
#[cfg(feature = "cuda")]
pub use percentile_nearest_rank_wrapper::{CudaPercentileNearestRank, CudaPnrError};
#[cfg(feature = "cuda")]
pub use prb_wrapper::{CudaPrb, CudaPrbError};
#[cfg(feature = "cuda")]
pub use vi_wrapper::{CudaVi, CudaViError};
#[cfg(feature = "cuda")]
pub use vpci_wrapper::{CudaVpci, CudaVpciError};
#[cfg(feature = "cuda")]
pub mod linearreg_angle_wrapper;
#[cfg(feature = "cuda")]
pub use linearreg_angle_wrapper::{CudaLinearregAngle, CudaLinearregAngleError};
#[cfg(feature = "cuda")]
pub use oscillators::cg_wrapper::{CudaCg, CudaCgError};
#[cfg(feature = "cuda")]
pub use oscillators::cmo_wrapper::{CudaCmo, CudaCmoError};
#[cfg(feature = "cuda")]
pub use oscillators::dti_wrapper::{CudaDti, CudaDtiError};
#[cfg(feature = "cuda")]
pub use oscillators::emv_wrapper::{CudaEmv, CudaEmvError};
#[cfg(feature = "cuda")]
pub use oscillators::kdj_wrapper::{CudaKdj, CudaKdjError};
#[cfg(feature = "cuda")]
pub use oscillators::stochf_wrapper::{CudaStochf, CudaStochfError};
#[cfg(feature = "cuda")]
pub use oscillators::reverse_rsi_wrapper::{CudaReverseRsi, CudaReverseRsiError};
#[cfg(feature = "cuda")]
pub use oscillators::squeeze_momentum_wrapper::{CudaSqueezeMomentum, CudaSmiError};
#[cfg(feature = "cuda")]
pub use oscillators::ttm_squeeze_wrapper::{CudaTtmSqueeze, CudaTtmSqueezeError};
#[cfg(feature = "cuda")]
pub use bollinger_bands_wrapper::{CudaBollingerBands, CudaBollingerError};

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
