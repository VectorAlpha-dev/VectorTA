//! CUDA integration scaffolding (cust-based)
//!
//! This module is built only when the `cuda` feature is enabled. It provides
//! runtime detection helpers and submodules for GPU-accelerated indicators.

#[cfg(feature = "cuda")]
pub mod ad_wrapper;
#[cfg(feature = "cuda")]
pub mod adx_wrapper;
#[cfg(feature = "cuda")]
pub mod adxr_wrapper;
#[cfg(feature = "cuda")]
pub mod alligator_wrapper;
#[cfg(feature = "cuda")]
pub mod alphatrend_wrapper;
#[cfg(feature = "cuda")]
pub mod aroon_wrapper;
#[cfg(feature = "cuda")]
pub mod atr_wrapper;
#[cfg(feature = "cuda")]
pub mod avsl_wrapper;
#[cfg(feature = "cuda")]
pub mod bandpass_wrapper;
#[cfg(feature = "cuda")]
pub mod bench;
#[cfg(feature = "cuda")]
pub mod chande_wrapper;
#[cfg(feature = "cuda")]
pub mod cvi_wrapper;
#[cfg(feature = "cuda")]
pub mod di_wrapper;
#[cfg(feature = "cuda")]
pub mod dm_wrapper;
#[cfg(feature = "cuda")]
pub mod donchian_wrapper;
#[cfg(feature = "cuda")]
pub mod dx_wrapper;
#[cfg(feature = "cuda")]
pub mod eri_wrapper;
#[cfg(feature = "cuda")]
pub mod keltner_wrapper;
#[cfg(feature = "cuda")]
pub mod marketefi_wrapper;
#[cfg(feature = "cuda")]
pub mod medprice_wrapper;
#[cfg(feature = "cuda")]
pub mod moving_averages;
#[cfg(feature = "cuda")]
pub mod qstick_wrapper;
#[cfg(feature = "cuda")]
pub mod rocr_wrapper;
#[cfg(feature = "cuda")]
pub mod wavetrend;

#[cfg(feature = "cuda")]
pub use ad_wrapper::{CudaAd, CudaAdError};
#[cfg(feature = "cuda")]
pub use adx_wrapper::{CudaAdx, CudaAdxError};
#[cfg(feature = "cuda")]
pub use adxr_wrapper::{CudaAdxr, CudaAdxrError};
#[cfg(feature = "cuda")]
pub use alligator_wrapper::{
    CudaAlligator, CudaAlligatorBatchResult, CudaAlligatorError, DeviceArrayF32Trio,
};
#[cfg(feature = "cuda")]
pub use alphatrend_wrapper::{CudaAlphaTrend, CudaAlphaTrendError};
#[cfg(feature = "cuda")]
pub use aroon_wrapper::{CudaAroon, CudaAroonError};
#[cfg(feature = "cuda")]
pub use atr_wrapper::CudaAtr;
#[cfg(feature = "cuda")]
pub use avsl_wrapper::{CudaAvsl, CudaAvslError};
#[cfg(feature = "cuda")]
pub use bandpass_wrapper::{CudaBandpass, CudaBandpassBatchResult, DeviceArrayF32Quad};
#[cfg(feature = "cuda")]
pub use bench::{CudaBenchScenario, CudaBenchState};
#[cfg(feature = "cuda")]
pub use chande_wrapper::CudaChande;
#[cfg(feature = "cuda")]
pub use cvi_wrapper::{CudaCvi, CudaCviError};
#[cfg(feature = "cuda")]
pub use di_wrapper::{CudaDi, CudaDiError, DeviceArrayF32Pair};
#[cfg(feature = "cuda")]
pub use dm_wrapper::{CudaDm, CudaDmError};
#[cfg(feature = "cuda")]
pub use donchian_wrapper::{CudaDonchian, CudaDonchianError};
#[cfg(feature = "cuda")]
pub use dx_wrapper::{CudaDx, CudaDxError};
#[cfg(feature = "cuda")]
pub use eri_wrapper::{CudaEri, CudaEriError};
#[cfg(feature = "cuda")]
pub use keltner_wrapper::{
    CudaKeltner, CudaKeltnerBatchResult, CudaKeltnerError, DeviceKeltnerTriplet,
};
#[cfg(feature = "cuda")]
pub use marketefi_wrapper::{CudaMarketefi, CudaMarketefiError};
#[cfg(feature = "cuda")]
pub use medprice_wrapper::CudaMedprice;
#[cfg(feature = "cuda")]
pub use moving_averages::rsmk_wrapper::{CudaRsmk, CudaRsmkError};
#[cfg(feature = "cuda")]
pub use moving_averages::wclprice_wrapper::CudaWclprice;
#[cfg(feature = "cuda")]
pub use moving_averages::{
    CudaAlma, CudaDma, CudaEhlersPma, CudaGaussian, CudaJma, CudaMama, CudaReflex, CudaSqwma,
    CudaTema, CudaVwma, DeviceArrayF32, DeviceEhlersPmaPair, DeviceMamaPair,
};
#[cfg(feature = "cuda")]
pub use qstick_wrapper::{
    BatchKernelPolicy as QsBatchKernelPolicy, CudaQstick, CudaQstickError, CudaQstickPolicy,
    ManySeriesKernelPolicy as QsManySeriesKernelPolicy,
};
#[cfg(feature = "cuda")]
pub use rocr_wrapper::{CudaRocr, CudaRocrError};
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
pub mod bollinger_bands_wrapper;
#[cfg(feature = "cuda")]
pub mod dvdiqqe_wrapper;
#[cfg(feature = "cuda")]
pub mod er_wrapper;
#[cfg(feature = "cuda")]
pub mod nadaraya_watson_envelope_wrapper;
#[cfg(feature = "cuda")]
pub mod nvi_wrapper;
#[cfg(feature = "cuda")]
pub mod pfe_wrapper;
#[cfg(feature = "cuda")]
pub mod pvi_wrapper;
#[cfg(feature = "cuda")]
pub mod supertrend_wrapper;
#[cfg(feature = "cuda")]
pub mod ttm_trend_wrapper;
#[cfg(feature = "cuda")]
pub mod vpt_wrapper;
#[cfg(feature = "cuda")]
pub mod vwmacd_wrapper;
#[cfg(feature = "cuda")]
pub mod wto_wrapper;

#[cfg(feature = "cuda")]
pub use dvdiqqe_wrapper::{CudaDvdiqqe, CudaDvdiqqeError};
#[cfg(feature = "cuda")]
pub use er_wrapper::{CudaEr, CudaErError};
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
pub use nadaraya_watson_envelope_wrapper::{CudaNwe, CudaNweError, DeviceNwePair};
#[cfg(feature = "cuda")]
pub use nvi_wrapper::{CudaNvi, CudaNviError};
#[cfg(feature = "cuda")]
pub use pfe_wrapper::{CudaPfe, CudaPfeError};
#[cfg(feature = "cuda")]
pub use pvi_wrapper::{CudaPvi, CudaPviError};
#[cfg(feature = "cuda")]
pub use supertrend_wrapper::{CudaSupertrend, CudaSupertrendError};
#[cfg(feature = "cuda")]
pub use ttm_trend_wrapper::{CudaTtmTrend, CudaTtmTrendError};
#[cfg(feature = "cuda")]
pub use vpt_wrapper::{CudaVpt, CudaVptError};
#[cfg(feature = "cuda")]
pub use vwmacd_wrapper::{CudaVwmacd, CudaVwmacdError};
#[cfg(feature = "cuda")]
pub use wto_wrapper::{CudaWto, CudaWtoBatchResult, DeviceArrayF32Triplet};
#[cfg(feature = "cuda")]
pub mod bollinger_bands_width_wrapper;
#[cfg(feature = "cuda")]
pub mod chandelier_exit_wrapper;
#[cfg(feature = "cuda")]
pub mod cksp_wrapper;
#[cfg(feature = "cuda")]
pub mod correl_hl_wrapper;
#[cfg(feature = "cuda")]
pub mod damiani_volatmeter_wrapper;
#[cfg(feature = "cuda")]
pub mod deviation_wrapper;
#[cfg(feature = "cuda")]
pub mod devstop_wrapper;
#[cfg(feature = "cuda")]
pub mod efi_wrapper;
#[cfg(feature = "cuda")]
pub mod emd_wrapper;
#[cfg(feature = "cuda")]
pub mod fvg_trailing_stop_wrapper;
#[cfg(feature = "cuda")]
pub mod halftrend_wrapper;
#[cfg(feature = "cuda")]
pub mod kaufmanstop_wrapper;
#[cfg(feature = "cuda")]
pub mod kurtosis_wrapper;
#[cfg(feature = "cuda")]
pub mod lpc_wrapper;
#[cfg(feature = "cuda")]
pub mod mass_wrapper;
#[cfg(feature = "cuda")]
pub mod mean_ad_wrapper;
#[cfg(feature = "cuda")]
pub mod medium_ad_wrapper;
#[cfg(feature = "cuda")]
pub mod minmax_wrapper;
#[cfg(feature = "cuda")]
pub mod mod_god_mode_wrapper;
#[cfg(feature = "cuda")]
pub mod natr_wrapper;
#[cfg(feature = "cuda")]
pub mod net_myrsi_wrapper;
#[cfg(feature = "cuda")]
pub mod obv_wrapper;
#[cfg(feature = "cuda")]
pub mod percentile_nearest_rank_wrapper;
#[cfg(feature = "cuda")]
pub mod pivot_wrapper;
#[cfg(feature = "cuda")]
pub mod prb_wrapper;
#[cfg(feature = "cuda")]
pub mod range_filter_wrapper;
#[cfg(feature = "cuda")]
pub mod safezonestop_wrapper;
#[cfg(feature = "cuda")]
pub mod sar_wrapper;
#[cfg(feature = "cuda")]
pub mod stddev_wrapper;
#[cfg(feature = "cuda")]
pub mod ui_wrapper;
#[cfg(feature = "cuda")]
pub mod var_wrapper;
#[cfg(feature = "cuda")]
pub mod vi_wrapper;
#[cfg(feature = "cuda")]
pub mod vosc_wrapper;
#[cfg(feature = "cuda")]
pub mod voss_wrapper;
#[cfg(feature = "cuda")]
pub mod vpci_wrapper;
#[cfg(feature = "cuda")]
pub mod wad_wrapper;
#[cfg(feature = "cuda")]
pub mod zscore_wrapper;

#[cfg(feature = "cuda")]
pub use oscillators::{CudaDecOsc, CudaDecOscError};
#[cfg(feature = "cuda")]
pub use oscillators::{CudaFisher, CudaFisherError};
#[cfg(feature = "cuda")]
pub use oscillators::{CudaIftRsi, CudaIftRsiError};
#[cfg(feature = "cuda")]
pub use oscillators::{CudaMfi, CudaMfiError};

#[cfg(feature = "cuda")]
pub use bollinger_bands_width_wrapper::{CudaBbw, CudaBbwError};
#[cfg(feature = "cuda")]
pub use chande_wrapper::CudaChandeError;
#[cfg(feature = "cuda")]
pub use cksp_wrapper::{CudaCksp, CudaCkspError};
#[cfg(feature = "cuda")]
pub use deviation_wrapper::{CudaDeviation, CudaDeviationError};
#[cfg(feature = "cuda")]
pub use devstop_wrapper::{CudaDevStop, CudaDevStopError};
#[cfg(feature = "cuda")]
pub use emd_wrapper::{CudaEmd, CudaEmdBatchResult, CudaEmdError, DeviceArrayF32Triple};
#[cfg(feature = "cuda")]
pub use fvg_trailing_stop_wrapper::{CudaFvgTs, CudaFvgTsError};
#[cfg(feature = "cuda")]
pub use kaufmanstop_wrapper::{CudaKaufmanstop, CudaKaufmanstopError};
#[cfg(feature = "cuda")]
pub use mass_wrapper::{CudaMass, CudaMassError};
#[cfg(feature = "cuda")]
pub use mean_ad_wrapper::{CudaMeanAd, CudaMeanAdError};
#[cfg(feature = "cuda")]
pub use medium_ad_wrapper::{CudaMediumAd, CudaMediumAdError};
#[cfg(feature = "cuda")]
pub use minmax_wrapper::{CudaMinmax, CudaMinmaxError};
#[cfg(feature = "cuda")]
pub use mod_god_mode_wrapper::{CudaModGodMode, CudaModGodModeBatchResult};
#[cfg(feature = "cuda")]
pub use moving_averages::{
    CudaApo, CudaBuffAverages, CudaBuffAveragesError, CudaFrama, CudaFramaError, CudaHma,
    CudaHmaError, CudaLinearregSlope, CudaLinearregSlopeError, CudaLinreg, CudaLinregError,
    CudaLinregIntercept, CudaLinregInterceptError, CudaNma, CudaNmaError, CudaSma, CudaSmaError,
    CudaSuperSmoother, CudaSuperSmootherError, CudaTrendflex, CudaTrendflexError, CudaTsf,
    CudaTsfError, CudaVidya, CudaVidyaError, CudaVlma, CudaVolumeAdjustedMa,
    CudaVolumeAdjustedMaError, CudaVpwma, CudaVpwmaError, CudaZlema, CudaZlemaError,
};
#[cfg(feature = "cuda")]
pub use natr_wrapper::{CudaNatr, CudaNatrError};
#[cfg(feature = "cuda")]
pub use net_myrsi_wrapper::{CudaNetMyrsi, CudaNetMyrsiError};
#[cfg(feature = "cuda")]
pub use oscillators::adosc_wrapper::{CudaAdosc, CudaAdoscError};
#[cfg(feature = "cuda")]
pub use oscillators::ao_wrapper::{CudaAo, CudaAoError};
#[cfg(feature = "cuda")]
pub use oscillators::cfo_wrapper::{CudaCfo, CudaCfoError};
#[cfg(feature = "cuda")]
pub use oscillators::coppock_wrapper::{CudaCoppock, CudaCoppockError};
#[cfg(feature = "cuda")]
pub use oscillators::dpo_wrapper::{CudaDpo, CudaDpoError};
#[cfg(feature = "cuda")]
pub use oscillators::fosc_wrapper::{CudaFosc, CudaFoscError};
#[cfg(feature = "cuda")]
pub use oscillators::gatorosc_wrapper::{CudaGatorOsc, CudaGatorOscError};
#[cfg(feature = "cuda")]
pub use oscillators::kvo_wrapper::{CudaKvo, CudaKvoError};
#[cfg(feature = "cuda")]
pub use oscillators::macd_wrapper::{CudaMacd, CudaMacdError};
#[cfg(feature = "cuda")]
pub use oscillators::ppo_wrapper::{CudaPpo, CudaPpoError};
#[cfg(feature = "cuda")]
pub use oscillators::tsi_wrapper::{CudaTsi, CudaTsiError};
#[cfg(feature = "cuda")]
pub use percentile_nearest_rank_wrapper::{CudaPercentileNearestRank, CudaPnrError};
#[cfg(feature = "cuda")]
pub use prb_wrapper::{CudaPrb, CudaPrbError};
#[cfg(feature = "cuda")]
pub use range_filter_wrapper::{CudaRangeFilter, CudaRangeFilterError, DeviceRangeFilterTrio};
#[cfg(feature = "cuda")]
pub use sar_wrapper::{CudaSar, CudaSarError};
#[cfg(feature = "cuda")]
pub use var_wrapper::{CudaVar, CudaVarError};
#[cfg(feature = "cuda")]
pub use vi_wrapper::{CudaVi, CudaViError};
#[cfg(feature = "cuda")]
pub use voss_wrapper::{CudaVoss, CudaVossError};
#[cfg(feature = "cuda")]
pub use vpci_wrapper::{CudaVpci, CudaVpciError};
#[cfg(feature = "cuda")]
pub use wad_wrapper::{CudaWad, CudaWadError};
#[cfg(feature = "cuda")]
pub use zscore_wrapper::{CudaZscore, CudaZscoreError};
#[cfg(feature = "cuda")]
pub mod linearreg_angle_wrapper;
#[cfg(feature = "cuda")]
pub use bollinger_bands_wrapper::{CudaBollingerBands, CudaBollingerError};
#[cfg(feature = "cuda")]
pub use chandelier_exit_wrapper::{CudaCeError, CudaChandelierExit};
#[cfg(feature = "cuda")]
pub use correl_hl_wrapper::{CudaCorrelHl, CudaCorrelHlError};
#[cfg(feature = "cuda")]
pub use damiani_volatmeter_wrapper::{CudaDamianiError, CudaDamianiVolatmeter};
#[cfg(feature = "cuda")]
pub use efi_wrapper::{CudaEfi, CudaEfiError};
#[cfg(feature = "cuda")]
pub use halftrend_wrapper::{CudaHalftrend, CudaHalftrendError};
#[cfg(feature = "cuda")]
pub use kurtosis_wrapper::{CudaKurtosis, CudaKurtosisError};
#[cfg(feature = "cuda")]
pub use linearreg_angle_wrapper::{CudaLinearregAngle, CudaLinearregAngleError};
#[cfg(feature = "cuda")]
pub use lpc_wrapper::{
    BatchKernelPolicy as LpcBatchKernelPolicy, CudaLpc, CudaLpcError, CudaLpcPolicy,
    ManySeriesKernelPolicy as LpcManySeriesKernelPolicy,
};
#[cfg(feature = "cuda")]
pub use obv_wrapper::{CudaObv, CudaObvError};
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
pub use oscillators::reverse_rsi_wrapper::{CudaReverseRsi, CudaReverseRsiError};
#[cfg(feature = "cuda")]
pub use oscillators::squeeze_momentum_wrapper::{CudaSmiError, CudaSqueezeMomentum};
#[cfg(feature = "cuda")]
pub use oscillators::stochf_wrapper::{CudaStochf, CudaStochfError};
#[cfg(feature = "cuda")]
pub use oscillators::ttm_squeeze_wrapper::{CudaTtmSqueeze, CudaTtmSqueezeError};
#[cfg(feature = "cuda")]
pub use pivot_wrapper::{CudaPivot, CudaPivotError};
#[cfg(feature = "cuda")]
pub use safezonestop_wrapper::{CudaSafeZoneStop, CudaSafeZoneStopError};
#[cfg(feature = "cuda")]
pub use stddev_wrapper::{CudaStddev, CudaStddevError};
#[cfg(feature = "cuda")]
pub use ui_wrapper::{CudaUi, CudaUiError};
#[cfg(feature = "cuda")]
pub use vosc_wrapper::{
    BatchKernelPolicy as VoscBatchKernelPolicy, CudaVosc, CudaVoscError, CudaVoscPolicy,
    ManySeriesKernelPolicy as VoscManySeriesKernelPolicy,
};

/// Returns true if a CUDA device is available and the driver API can be initialized.
#[inline]
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Local iteration safety: when building with placeholder PTX or when explicitly
        // asked to skip CUDA probes, report unavailable to let tests skip gracefully.
        if std::env::var("CUDA_PLACEHOLDER_ON_FAIL").ok().as_deref() == Some("1")
            || std::env::var("CUDA_FORCE_SKIP").ok().as_deref() == Some("1")
        {
            return false;
        }
        use cust::{device::Device, function::BlockSize, function::GridSize, module::Module, prelude::CudaFlags, stream::{Stream, StreamFlags}};
        // Initialize the CUDA driver and query devices. Keep this defensive so
        // it never panics when CUDA is missing.
        if cust::init(CudaFlags::empty()).is_err() {
            return false;
        }
        let ndev = match Device::num_devices() { Ok(n) => n, Err(_) => 0 };
        if ndev == 0 { return false; }
        // Probe a minimal kernel launch so test suites can confidently run.
        // Some environments expose a device but cannot JIT/launch PTX (e.g., mismatched drivers).
        // Launch a no-op kernel via a tiny PTX module; if it fails, treat CUDA as unavailable.
        const PROBE_PTX: &str = r#"
            .version 7.0
            .target compute_52
            .address_size 64
            .visible .entry probe() {
                ret;
            }
        "#;
        let device = match Device::get_device(0) { Ok(d) => d, Err(_) => return false };
        let context = match cust::context::Context::new(device) { Ok(c) => c, Err(_) => return false };
        let module = match Module::from_ptx(PROBE_PTX, &[]) { Ok(m) => m, Err(_) => return false };
        let func = match module.get_function("probe") { Ok(f) => f, Err(_) => return false };
        let stream = match Stream::new(StreamFlags::NON_BLOCKING, None) { Ok(s) => s, Err(_) => return false };
        unsafe {
            let args: &mut [*mut std::ffi::c_void] = &mut [];
            if stream.launch(&func, GridSize::xy(1, 1), BlockSize::xyz(1, 1, 1), 0, args).is_err() {
                return false;
            }
        }
        if stream.synchronize().is_err() {
            return false;
        }
        drop(context);
        true
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
