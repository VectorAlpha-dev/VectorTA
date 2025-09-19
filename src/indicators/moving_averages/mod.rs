pub mod alma;
pub mod buff_averages;
pub mod cwma;
pub mod dema;
pub mod dma;
pub mod edcf;
pub mod ehlers_ecema;
pub mod ehlers_itrend;
pub mod ehlers_kama;
pub mod ehlers_pma;
pub mod ehma;
pub mod ema;
pub mod epma;
pub mod frama;
pub mod fwma;
pub mod gaussian;
pub mod highpass;
pub mod highpass_2_pole;
pub mod hma;
pub mod hwma;
pub mod jma;
pub mod jsa;
pub mod kama;
pub mod linreg;
pub mod ma;
pub mod ma_stream;
pub mod maaq;
pub mod mama;
pub mod mwdx;
pub mod nama;
pub mod nma;
pub mod pwma;
pub mod reflex;
pub mod sama;
pub mod sinwma;
pub mod sma;
pub mod smma;
pub mod sqwma;
pub mod srwma;
pub mod supersmoother;
pub mod supersmoother_3_pole;
pub mod swma;
pub mod tema;
pub mod tilson;
pub mod tradjema;
pub mod trendflex;
pub mod trima;
pub mod uma;
pub mod volatility_adjusted_ma;
pub mod volume_adjusted_ma;
pub mod vpwma;
pub mod vwap;
pub mod vwma;
pub mod wilders;
pub mod wma;
pub mod zlema;

// Exports for migrated moving averages
pub use cwma::{cwma, CwmaInput, CwmaOutput, CwmaParams};
pub use dma::{
    dma,
    dma_batch_with_kernel,
    dma_into_slice,
    dma_with_kernel,
    DmaBatchBuilder,
    DmaBatchOutput,
    // Batch API exports
    DmaBatchRange,
    DmaBuilder,
    DmaData,
    DmaError,
    DmaInput,
    DmaOutput,
    DmaParams,
    DmaStream,
};

pub use edcf::{edcf, EdcfInput, EdcfOutput, EdcfParams};
pub use ehlers_kama::{ehlers_kama, EhlersKamaInput, EhlersKamaOutput, EhlersKamaParams};
pub use ehlers_pma::{ehlers_pma, EhlersPmaInput, EhlersPmaOutput, EhlersPmaParams};
pub use uma::{uma, UmaInput, UmaOutput, UmaParams};
pub use volatility_adjusted_ma::{
    vama as volatility_adjusted_ma, VamaInput as VolatilityAdjustedMaInput,
    VamaOutput as VolatilityAdjustedMaOutput, VamaParams as VolatilityAdjustedMaParams,
};
pub use volume_adjusted_ma::{
    VolumeAdjustedMa as volume_adjusted_ma, VolumeAdjustedMaInput, VolumeAdjustedMaOutput,
    VolumeAdjustedMaParams,
};

pub use ehma::{
    ehma,
    ehma_batch_inner_into,
    ehma_batch_par_slice,
    ehma_batch_slice,
    ehma_batch_with_kernel,
    ehma_batch_with_kernel_slice,
    ehma_into_slice,
    ehma_with_kernel,
    EhmaBatchBuilder,
    EhmaBatchOutput,
    // Batch API exports
    EhmaBatchRange,
    EhmaBuilder,
    EhmaData,
    EhmaError,
    EhmaInput,
    EhmaOutput,
    EhmaParams,
    // Streaming API exports
    EhmaStream,
};

pub use nama::{
    nama,
    nama_batch_with_kernel,
    nama_into_slice,
    nama_with_kernel,
    NamaBatchBuilder,
    NamaBatchOutput,
    // Batch API exports
    NamaBatchRange,
    NamaBuilder,
    NamaData,
    NamaError,
    NamaInput,
    NamaOutput,
    NamaParams,
    NamaStream,
};

pub use sama::{
    sama,
    sama_batch_par_slice,
    sama_batch_slice,
    sama_batch_with_kernel,
    sama_into_slice,
    sama_with_kernel,
    SamaBatchBuilder,
    SamaBatchOutput,
    // Batch API exports
    SamaBatchRange,
    SamaBuilder,
    SamaData,
    SamaError,
    SamaInput,
    SamaOutput,
    SamaParams,
    // Streaming API exports
    SamaStream,
};

// Python exports for migrated moving averages
#[cfg(feature = "python")]
pub use dma::{dma_batch_py, dma_py, DmaStreamPy};

#[cfg(feature = "python")]
pub use ehma::{ehma_batch_py, ehma_py, EhmaStreamPy};

#[cfg(feature = "python")]
pub use nama::{nama_batch_py, nama_py, NamaStreamPy};

#[cfg(feature = "python")]
pub use sama::{sama_batch_py, sama_py, SamaStreamPy};

// WASM exports
#[cfg(feature = "wasm")]
pub use nama::{nama_alloc, nama_batch_unified_js, nama_free, nama_into, nama_js};
