pub mod avsl;
pub use avsl::{
    avsl, avsl_with_kernel, AvslInput, AvslOutput, AvslParams, AvslError, 
    AvslData, AvslBuilder, avsl_into_slice,
    // Batch API exports
    AvslBatchRange, AvslBatchBuilder, AvslBatchOutput, avsl_batch_with_kernel,
};

pub mod dma;
pub use dma::{
    dma, dma_with_kernel, DmaInput, DmaOutput, DmaParams, DmaError,
    DmaData, DmaBuilder, dma_into_slice, DmaStream,
    // Batch API exports
    DmaBatchRange, DmaBatchBuilder, DmaBatchOutput, dma_batch_with_kernel,
};

pub mod range_filter;
pub use range_filter::{
    range_filter, range_filter_with_kernel, range_filter_into_slice, RangeFilterInput, RangeFilterOutput, 
    RangeFilterParams, RangeFilterError, RangeFilterData, RangeFilterBuilder,
    // Batch API exports
    RangeFilterBatchRange, RangeFilterBatchBuilder, RangeFilterBatchOutput,
    range_filter_batch_slice, range_filter_batch_par_slice,
    // Streaming API exports
    RangeFilterStream,
};

pub mod sama;
pub use sama::{
    sama, sama_with_kernel, sama_into_slice, SamaInput, SamaOutput, 
    SamaParams, SamaError, SamaData, SamaBuilder,
    // Batch API exports
    SamaBatchRange, SamaBatchBuilder, SamaBatchOutput, sama_batch_with_kernel,
    sama_batch_slice, sama_batch_par_slice,
    // Streaming API exports
    SamaStream,
};

pub mod wto;
pub use wto::{
    wto, wto_with_kernel, wto_into_slices, WtoInput, WtoOutput, 
    WtoParams, WtoError, WtoData, WtoBuilder,
    // Batch API exports
    WtoBatchRange, WtoBatchBuilder, WtoBatchOutput, wto_batch_slice, wto_batch_candles,
    // Streaming API exports
    WtoStream,
};

pub mod ehma;
pub use ehma::{
    ehma, ehma_with_kernel, ehma_into_slice, EhmaInput, EhmaOutput,
    EhmaParams, EhmaError, EhmaData, EhmaBuilder,
    // Batch API exports
    EhmaBatchRange, EhmaBatchBuilder, EhmaBatchOutput, 
    ehma_batch_with_kernel, ehma_batch_with_kernel_slice,
    ehma_batch_slice, ehma_batch_par_slice, ehma_batch_inner_into,
    // Streaming API exports
    EhmaStream,
};

// Python exports
#[cfg(feature = "python")]
pub use avsl::{avsl_py, avsl_batch_py};

#[cfg(feature = "python")]
pub use dma::{dma_py, dma_batch_py, DmaStreamPy};

#[cfg(feature = "python")]
pub use range_filter::{range_filter_py, range_filter_batch_py, RangeFilterStreamPy};

#[cfg(feature = "python")]
pub use sama::{sama_py, sama_batch_py, SamaStreamPy};

#[cfg(feature = "python")]
pub use wto::{wto_py, wto_batch_py, WtoStreamPy};

#[cfg(feature = "python")]
pub use ehma::{ehma_py, ehma_batch_py, EhmaStreamPy};