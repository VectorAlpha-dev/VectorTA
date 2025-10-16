#[cfg(feature = "cuda")]
pub mod willr_wrapper;

#[cfg(feature = "cuda")]
pub mod aso_wrapper;

#[cfg(feature = "cuda")]
pub mod cg_wrapper;

#[cfg(feature = "cuda")]
pub mod cmo_wrapper;

#[cfg(feature = "cuda")]
pub mod dti_wrapper;
#[cfg(feature = "cuda")]
pub mod emv_wrapper;
#[cfg(feature = "cuda")]
pub mod kdj_wrapper;
#[cfg(feature = "cuda")]
pub mod stochf_wrapper;
#[cfg(feature = "cuda")]
pub mod reverse_rsi_wrapper;
#[cfg(feature = "cuda")]
pub mod squeeze_momentum_wrapper;
#[cfg(feature = "cuda")]
pub mod ttm_squeeze_wrapper;
pub mod acosc_wrapper;
pub mod aroonosc_wrapper;
pub mod cfo_wrapper;
pub mod dpo_wrapper;
pub mod fosc_wrapper;
pub mod kvo_wrapper;
pub mod lrsi_wrapper;
pub mod tsi_wrapper;
pub mod ppo_wrapper;
pub mod stoch_wrapper;

#[cfg(feature = "cuda")]
pub use willr_wrapper::{CudaWillr, CudaWillrError};

#[cfg(feature = "cuda")]
pub use aso_wrapper::{CudaAso, CudaAsoError};

#[cfg(feature = "cuda")]
pub use cg_wrapper::{CudaCg, CudaCgError};

#[cfg(feature = "cuda")]
pub use cmo_wrapper::{CudaCmo, CudaCmoError};

#[cfg(feature = "cuda")]
pub use dti_wrapper::{CudaDti, CudaDtiError};
#[cfg(feature = "cuda")]
pub use emv_wrapper::{CudaEmv, CudaEmvError};
#[cfg(feature = "cuda")]
pub use kdj_wrapper::{CudaKdj, CudaKdjError};
#[cfg(feature = "cuda")]
pub use stochf_wrapper::{CudaStochf, CudaStochfError};
#[cfg(feature = "cuda")]
pub use reverse_rsi_wrapper::{CudaReverseRsi, CudaReverseRsiError};
#[cfg(feature = "cuda")]
pub use squeeze_momentum_wrapper::{CudaSqueezeMomentum, CudaSmiError};
#[cfg(feature = "cuda")]
pub use ttm_squeeze_wrapper::{CudaTtmSqueeze, CudaTtmSqueezeError};
pub use acosc_wrapper::{CudaAcosc, CudaAcoscError};
pub use aroonosc_wrapper::{CudaAroonOsc, CudaAroonOscError};
pub use cfo_wrapper::{CudaCfo, CudaCfoError};
pub use dpo_wrapper::{CudaDpo, CudaDpoError};
pub use fosc_wrapper::{CudaFosc, CudaFoscError};
pub use kvo_wrapper::{CudaKvo, CudaKvoError};
pub use lrsi_wrapper::{CudaLrsi, CudaLrsiError};
pub use tsi_wrapper::{CudaTsi, CudaTsiError};
pub use ppo_wrapper::{CudaPpo, CudaPpoError};
pub use stoch_wrapper::{CudaStoch, CudaStochError};
