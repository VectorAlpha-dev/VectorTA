use crate::{BacktestEngine, ObjectiveKind, OptimizationResult};
use ta_strategies::double_ma::{DoubleMaParams, Metrics};

fn score(metrics: &Metrics, objective: ObjectiveKind) -> f64 {
    match objective {
        ObjectiveKind::Pnl => metrics.pnl,
        ObjectiveKind::Sharpe => metrics.sharpe,
        ObjectiveKind::MaxDrawdown => -metrics.max_dd,
    }
}

pub fn grid_search<E: BacktestEngine>(
    engine: &E,
    combos: &[DoubleMaParams],
    objective: ObjectiveKind,
) -> Option<OptimizationResult> {
    if combos.is_empty() {
        return None;
    }
    let metrics = engine.eval_batch(combos);
    if metrics.is_empty() {
        return None;
    }

    let mut best_idx = 0usize;
    let mut best_score = score(&metrics[0], objective);

    for (i, m) in metrics.iter().enumerate().skip(1) {
        let s = score(m, objective);
        if s > best_score {
            best_score = s;
            best_idx = i;
        }
    }

    let best_params = combos[best_idx].clone();
    let best_metrics = metrics[best_idx].clone();
    let all = combos
        .iter()
        .cloned()
        .zip(metrics.into_iter())
        .collect();

    Some(OptimizationResult {
        best_params,
        best_metrics,
        all,
    })
}
