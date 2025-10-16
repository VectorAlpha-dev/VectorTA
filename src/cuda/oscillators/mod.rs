#[cfg(feature = "cuda")]
pub mod willr_wrapper;

#[cfg(feature = "cuda")]
pub use willr_wrapper::{CudaWillr, CudaWillrError};

#[cfg(feature = "cuda")]
pub mod adosc_wrapper;

#[cfg(feature = "cuda")]
pub use adosc_wrapper::{CudaAdosc, CudaAdoscError};

#[cfg(feature = "cuda")]
pub mod ao_wrapper;

#[cfg(feature = "cuda")]
pub use ao_wrapper::{CudaAo, CudaAoError};

#[cfg(feature = "cuda")]
pub mod bop_wrapper;

#[cfg(feature = "cuda")]
pub use bop_wrapper::{CudaBop, CudaBopError};

#[cfg(feature = "cuda")]
pub mod coppock_wrapper;

#[cfg(feature = "cuda")]
pub use coppock_wrapper::{CudaCoppock, CudaCoppockError};

#[cfg(feature = "cuda")]
pub mod gatorosc_wrapper;

#[cfg(feature = "cuda")]
pub use gatorosc_wrapper::{CudaGatorOsc, CudaGatorOscError};

#[cfg(feature = "cuda")]
pub mod macd_wrapper;

#[cfg(feature = "cuda")]
pub use macd_wrapper::{CudaMacd, CudaMacdError};

#[cfg(feature = "cuda")]
pub mod mom_wrapper;

#[cfg(feature = "cuda")]
pub use mom_wrapper::{CudaMom, CudaMomError};

#[cfg(feature = "cuda")]
pub mod roc_wrapper;

#[cfg(feature = "cuda")]
pub use roc_wrapper::{CudaRoc, CudaRocError};

#[cfg(feature = "cuda")]
pub mod rsx_wrapper;

#[cfg(feature = "cuda")]
pub use rsx_wrapper::{CudaRsx, CudaRsxError};

#[cfg(feature = "cuda")]
pub mod srsi_wrapper;

#[cfg(feature = "cuda")]
pub use srsi_wrapper::{CudaSrsi, CudaSrsiError};
