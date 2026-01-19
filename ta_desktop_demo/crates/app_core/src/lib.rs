pub mod progress;
pub mod double_ma;
#[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
pub mod gpu_backtest_kernel;
#[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
pub(crate) mod vram_ma;

pub use ta_strategies::double_ma::DoubleMaCurves;

pub use double_ma::{
    Backend,
    BackendOptimizationResult,
    BackendUsed,
    DoubleMaDrilldownRequest,
    DoubleMaParamsResolved,
    DoubleMaRequest,
    OptimizationModeResolved,
    compute_double_ma_drilldown_blocking_with_candles,
    run_double_ma_optimization_blocking_with_candles,
};
