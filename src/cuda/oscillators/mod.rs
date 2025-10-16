#[cfg(feature = "cuda")]
pub mod willr_wrapper;

#[cfg(feature = "cuda")]
pub mod cci_wrapper;

#[cfg(feature = "cuda")]
pub mod chop_wrapper;

#[cfg(feature = "cuda")]
pub mod dec_osc_wrapper;
#[cfg(feature = "cuda")]
pub mod fisher_wrapper;
#[cfg(feature = "cuda")]
pub mod ift_rsi_wrapper;
#[cfg(feature = "cuda")]
pub mod mfi_wrapper;
#[cfg(feature = "cuda")]
pub mod ultosc_wrapper;

#[cfg(feature = "cuda")]
pub use willr_wrapper::{CudaWillr, CudaWillrError};

#[cfg(feature = "cuda")]
pub use cci_wrapper::{CudaCci, CudaCciError};

#[cfg(feature = "cuda")]
pub use chop_wrapper::{CudaChop, CudaChopError};

#[cfg(feature = "cuda")]
pub use dec_osc_wrapper::{CudaDecOsc, CudaDecOscError};
#[cfg(feature = "cuda")]
pub use fisher_wrapper::{CudaFisher, CudaFisherError};
#[cfg(feature = "cuda")]
pub use ift_rsi_wrapper::{CudaIftRsi, CudaIftRsiError};
#[cfg(feature = "cuda")]
pub use mfi_wrapper::{CudaMfi, CudaMfiError};
#[cfg(feature = "cuda")]
pub use ultosc_wrapper::{
    benches as ultosc_benches, BatchKernelPolicy as UltoscBatchKernelPolicy, CudaUltosc,
    CudaUltoscError, CudaUltoscPolicy, ManySeriesKernelPolicy as UltoscManySeriesKernelPolicy,
};
