pub mod grid;
pub mod stream;

use serde::{Deserialize, Serialize};
use ta_strategies::double_ma::{DoubleMaParams, Metrics};
use my_project::utilities::data_loader::Candles;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObjectiveKind {
    Pnl,
    Sharpe,
    MaxDrawdown,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimizationMode {
    Grid,
    CoarseToFine,
    Auto,
}

pub trait BacktestEngine {
    fn eval_batch(&self, combos: &[DoubleMaParams]) -> Vec<Metrics>;
}

pub struct CpuEngine<'a> {
    pub candles: &'a Candles,
    pub fast_ma_types: &'a [String],
    pub slow_ma_types: &'a [String],
    pub ma_source: &'a str,
}

impl<'a> BacktestEngine for CpuEngine<'a> {
    fn eval_batch(&self, combos: &[DoubleMaParams]) -> Vec<Metrics> {
        ta_strategies::double_ma::double_ma_batch_cpu(
            self.candles,
            combos,
            self.fast_ma_types,
            self.slow_ma_types,
            self.ma_source,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub best_params: DoubleMaParams,
    pub best_metrics: Metrics,
    pub top: Vec<(DoubleMaParams, Metrics)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub all: Option<Vec<(DoubleMaParams, Metrics)>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heatmap: Option<OptimizationHeatmap>,
    pub num_combos: usize,
    pub num_candles: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationHeatmap {
    pub bins_fast: usize,
    pub bins_slow: usize,
    pub fast_min: u16,
    pub fast_max: u16,
    pub slow_min: u16,
    pub slow_max: u16,
    pub values: Vec<Option<f64>>,
}
