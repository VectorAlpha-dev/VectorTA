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
