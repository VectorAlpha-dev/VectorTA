// File module that routes oscillator CUDA wrappers located in `src/cuda/`.
// This replaces the former `src/cuda/oscillators/mod.rs` directory module.

#[cfg(feature = "cuda")]
#[path = "willr_wrapper.rs"]
pub mod willr_wrapper;

#[cfg(feature = "cuda")]
#[path = "aso_wrapper.rs"]
pub mod aso_wrapper;

#[cfg(feature = "cuda")]
#[path = "cg_wrapper.rs"]
pub mod cg_wrapper;

#[cfg(feature = "cuda")]
#[path = "cmo_wrapper.rs"]
pub mod cmo_wrapper;

#[path = "acosc_wrapper.rs"]
pub mod acosc_wrapper;
#[path = "aroonosc_wrapper.rs"]
pub mod aroonosc_wrapper;
#[path = "cci_cycle_wrapper.rs"]
pub mod cci_cycle_wrapper;
#[path = "cfo_wrapper.rs"]
pub mod cfo_wrapper;
#[path = "dpo_wrapper.rs"]
pub mod dpo_wrapper;
#[cfg(feature = "cuda")]
#[path = "dti_wrapper.rs"]
pub mod dti_wrapper;
#[cfg(feature = "cuda")]
#[path = "emv_wrapper.rs"]
pub mod emv_wrapper;
#[path = "fosc_wrapper.rs"]
pub mod fosc_wrapper;
#[cfg(feature = "cuda")]
#[path = "kdj_wrapper.rs"]
pub mod kdj_wrapper;
#[cfg(feature = "cuda")]
#[path = "kst_wrapper.rs"]
pub mod kst_wrapper;
#[path = "kvo_wrapper.rs"]
pub mod kvo_wrapper;
#[path = "lrsi_wrapper.rs"]
pub mod lrsi_wrapper;
#[cfg(feature = "cuda")]
#[path = "msw_wrapper.rs"]
pub mod msw_wrapper;
#[path = "ppo_wrapper.rs"]
pub mod ppo_wrapper;
#[cfg(feature = "cuda")]
#[path = "qqe_wrapper.rs"]
pub mod qqe_wrapper;
#[cfg(feature = "cuda")]
#[path = "reverse_rsi_wrapper.rs"]
pub mod reverse_rsi_wrapper;
#[cfg(feature = "cuda")]
#[path = "rocp_wrapper.rs"]
pub mod rocp_wrapper;
#[cfg(feature = "cuda")]
#[path = "rvi_wrapper.rs"]
pub mod rvi_wrapper;
#[cfg(feature = "cuda")]
#[path = "squeeze_momentum_wrapper.rs"]
pub mod squeeze_momentum_wrapper;
#[cfg(feature = "cuda")]
#[path = "stc_wrapper.rs"]
pub mod stc_wrapper;
#[path = "stoch_wrapper.rs"]
pub mod stoch_wrapper;
#[cfg(feature = "cuda")]
#[path = "stochf_wrapper.rs"]
pub mod stochf_wrapper;
#[path = "tsi_wrapper.rs"]
pub mod tsi_wrapper;
#[cfg(feature = "cuda")]
#[path = "ttm_squeeze_wrapper.rs"]
pub mod ttm_squeeze_wrapper;

#[cfg(feature = "cuda")]
#[path = "cci_wrapper.rs"]
pub mod cci_wrapper;

#[cfg(feature = "cuda")]
#[path = "chop_wrapper.rs"]
pub mod chop_wrapper;

#[cfg(feature = "cuda")]
#[path = "dec_osc_wrapper.rs"]
pub mod dec_osc_wrapper;
#[cfg(feature = "cuda")]
#[path = "fisher_wrapper.rs"]
pub mod fisher_wrapper;
#[cfg(feature = "cuda")]
#[path = "ift_rsi_wrapper.rs"]
pub mod ift_rsi_wrapper;
#[cfg(feature = "cuda")]
#[path = "mfi_wrapper.rs"]
pub mod mfi_wrapper;
#[cfg(feature = "cuda")]
#[path = "ultosc_wrapper.rs"]
pub mod ultosc_wrapper;

#[cfg(feature = "cuda")]
#[path = "adosc_wrapper.rs"]
pub mod adosc_wrapper;

#[cfg(feature = "cuda")]
#[path = "ao_wrapper.rs"]
pub mod ao_wrapper;

#[cfg(feature = "cuda")]
#[path = "bop_wrapper.rs"]
pub mod bop_wrapper;

#[cfg(feature = "cuda")]
#[path = "coppock_wrapper.rs"]
pub mod coppock_wrapper;

#[cfg(feature = "cuda")]
#[path = "gatorosc_wrapper.rs"]
pub mod gatorosc_wrapper;

#[cfg(feature = "cuda")]
#[path = "macd_wrapper.rs"]
pub mod macd_wrapper;

#[cfg(feature = "cuda")]
#[path = "mom_wrapper.rs"]
pub mod mom_wrapper;

#[cfg(feature = "cuda")]
#[path = "roc_wrapper.rs"]
pub mod roc_wrapper;

#[cfg(feature = "cuda")]
#[path = "rsx_wrapper.rs"]
pub mod rsx_wrapper;

#[cfg(feature = "cuda")]
#[path = "rsi_wrapper.rs"]
pub mod rsi_wrapper;
#[cfg(feature = "cuda")]
#[path = "srsi_wrapper.rs"]
pub mod srsi_wrapper;

// Re-exports
pub use acosc_wrapper::{CudaAcosc, CudaAcoscError, DeviceArrayF32Acosc};
#[cfg(feature = "cuda")]
pub use adosc_wrapper::{CudaAdosc, CudaAdoscError};
#[cfg(feature = "cuda")]
pub use ao_wrapper::{CudaAo, CudaAoError};
pub use aroonosc_wrapper::{CudaAroonOsc, CudaAroonOscError, DeviceArrayF32Aroonosc};
#[cfg(feature = "cuda")]
pub use aso_wrapper::{CudaAso, CudaAsoError};
#[cfg(feature = "cuda")]
pub use bop_wrapper::{CudaBop, CudaBopError};
pub use cci_cycle_wrapper::{CudaCciCycle, CudaCciCycleError};
#[cfg(feature = "cuda")]
pub use cci_wrapper::{CudaCci, CudaCciError};
pub use cfo_wrapper::{CudaCfo, CudaCfoError};
#[cfg(feature = "cuda")]
pub use cg_wrapper::{CudaCg, CudaCgError};
#[cfg(feature = "cuda")]
pub use chop_wrapper::{CudaChop, CudaChopError};
#[cfg(feature = "cuda")]
pub use cmo_wrapper::{CudaCmo, CudaCmoError};
#[cfg(feature = "cuda")]
pub use coppock_wrapper::{CudaCoppock, CudaCoppockError};
#[cfg(feature = "cuda")]
pub use dec_osc_wrapper::{CudaDecOsc, CudaDecOscError};
pub use dpo_wrapper::{CudaDpo, CudaDpoError};
#[cfg(feature = "cuda")]
pub use dti_wrapper::{CudaDti, CudaDtiError};
#[cfg(feature = "cuda")]
pub use emv_wrapper::{CudaEmv, CudaEmvError};
#[cfg(feature = "cuda")]
pub use fisher_wrapper::{CudaFisher, CudaFisherError};
pub use fosc_wrapper::{CudaFosc, CudaFoscError};
#[cfg(feature = "cuda")]
pub use gatorosc_wrapper::{CudaGatorOsc, CudaGatorOscError};
#[cfg(feature = "cuda")]
pub use ift_rsi_wrapper::{CudaIftRsi, CudaIftRsiError};
#[cfg(feature = "cuda")]
pub use kdj_wrapper::{CudaKdj, CudaKdjError};
#[cfg(feature = "cuda")]
pub use kst_wrapper::{CudaKst, CudaKstError, DeviceKstPair};
pub use kvo_wrapper::{CudaKvo, CudaKvoError};
pub use lrsi_wrapper::{CudaLrsi, CudaLrsiError};
#[cfg(feature = "cuda")]
pub use macd_wrapper::{CudaMacd, CudaMacdError};
#[cfg(feature = "cuda")]
pub use mfi_wrapper::{CudaMfi, CudaMfiError};
#[cfg(feature = "cuda")]
pub use mom_wrapper::{CudaMom, CudaMomError};
#[cfg(feature = "cuda")]
pub use msw_wrapper::{CudaMsw, CudaMswError};
pub use ppo_wrapper::{CudaPpo, CudaPpoError};
#[cfg(feature = "cuda")]
pub use qqe_wrapper::{CudaQqe, CudaQqeError};
#[cfg(feature = "cuda")]
pub use reverse_rsi_wrapper::{CudaReverseRsi, CudaReverseRsiError};
#[cfg(feature = "cuda")]
pub use roc_wrapper::{CudaRoc, CudaRocError};
#[cfg(feature = "cuda")]
pub use rocp_wrapper::{CudaRocp, CudaRocpError};
#[cfg(feature = "cuda")]
pub use rsi_wrapper::{CudaRsi, CudaRsiError};
#[cfg(feature = "cuda")]
pub use rsx_wrapper::{CudaRsx, CudaRsxError};
#[cfg(feature = "cuda")]
pub use rvi_wrapper::{CudaRvi, CudaRviError};
#[cfg(feature = "cuda")]
pub use squeeze_momentum_wrapper::{CudaSmiError, CudaSqueezeMomentum};
#[cfg(feature = "cuda")]
pub use srsi_wrapper::{CudaSrsi, CudaSrsiError};
#[cfg(feature = "cuda")]
pub use stc_wrapper::{CudaStc, CudaStcError};
pub use stoch_wrapper::{CudaStoch, CudaStochError};
#[cfg(feature = "cuda")]
pub use stochf_wrapper::{CudaStochf, CudaStochfError};
pub use tsi_wrapper::{CudaTsi, CudaTsiError};
#[cfg(feature = "cuda")]
pub use ttm_squeeze_wrapper::{CudaTtmSqueeze, CudaTtmSqueezeError};
#[cfg(feature = "cuda")]
pub use ultosc_wrapper::{
    benches as ultosc_benches, BatchKernelPolicy as UltoscBatchKernelPolicy, CudaUltosc,
    CudaUltoscError, CudaUltoscPolicy, ManySeriesKernelPolicy as UltoscManySeriesKernelPolicy,
};
#[cfg(feature = "cuda")]
pub use willr_wrapper::{CudaWillr, CudaWillrError};
