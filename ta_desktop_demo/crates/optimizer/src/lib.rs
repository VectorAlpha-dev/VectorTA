pub mod grid;

use serde::{Deserialize, Serialize};
use ta_strategies::double_ma::{DoubleMaParams, Metrics};
use my_project::utilities::data_loader::Candles;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ObjectiveKind {
    Pnl,
    Sharpe,
    MaxDrawdown,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum OptimizationMode {
    Grid,
    Auto,
}

pub trait BacktestEngine {
    fn eval_batch(&self, combos: &[DoubleMaParams]) -> Vec<Metrics>;
}

pub struct CpuEngine<'a> {
    pub candles: &'a Candles,
}

impl<'a> BacktestEngine for CpuEngine<'a> {
    fn eval_batch(&self, combos: &[DoubleMaParams]) -> Vec<Metrics> {
        ta_strategies::double_ma::double_ma_batch_cpu(self.candles, combos)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub best_params: DoubleMaParams,
    pub best_metrics: Metrics,
    pub all: Vec<(DoubleMaParams, Metrics)>,
    pub num_combos: usize,
    pub num_candles: usize,
}
