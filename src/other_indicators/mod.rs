pub mod alphatrend;
pub use alphatrend::{
    alphatrend, AlphaTrendInput, AlphaTrendOutput, AlphaTrendParams
};

pub mod cora_wave;
pub use cora_wave::{
    cora_wave, CoraWaveInput, CoraWaveOutput, CoraWaveParams
};

pub mod ehlers_pma;
pub use ehlers_pma::{
    ehlers_pma, EhlersPmaInput, EhlersPmaOutput, EhlersPmaParams
};

pub mod chandelier_exit;
pub use chandelier_exit::{
    chandelier_exit, chandelier_exit_with_kernel, chandelier_exit_into_slices, chandelier_exit_into_flat,
    ChandelierExitInput, ChandelierExitOutput, ChandelierExitParams, ChandelierExitData,
    ChandelierExitBuilder, ChandelierExitError,
    CeBatchRange, CeBatchBuilder, CeBatchOutput, 
    ce_batch_with_kernel, ce_batch_slice, ce_batch_par_slice
};

pub mod percentile_nearest_rank;
pub use percentile_nearest_rank::{
    percentile_nearest_rank, percentile_nearest_rank_with_kernel, percentile_nearest_rank_into_slice,
    PercentileNearestRankInput, PercentileNearestRankOutput, 
    PercentileNearestRankParams, PercentileNearestRankData, PercentileNearestRankError,
    PercentileNearestRankBuilder, PercentileNearestRankStream,
    PercentileNearestRankBatchRange, PercentileNearestRankBatchBuilder, PercentileNearestRankBatchOutput,
    pnr_batch_with_kernel, pnr_batch_slice, pnr_batch_par_slice
};

pub mod uma;
pub use uma::{
    uma, uma_with_kernel, uma_into_slice,
    UmaInput, UmaOutput, UmaParams, UmaData, UmaError, UmaBuilder, UmaStream,
    UmaBatchRange, UmaBatchBuilder, UmaBatchOutput,
    uma_batch_with_kernel, uma_batch_slice, uma_batch_par_slice,
    expand_grid_uma
};