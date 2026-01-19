use crate::{BacktestEngine, ObjectiveKind, OptimizationResult};
use ta_strategies::double_ma::{DoubleMaParams, Metrics};
use std::cmp::Ordering;

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
    num_candles: usize,
    top_k: usize,
    include_all: bool,
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

    let top = if top_k == 0 {
        Vec::new()
    } else {
        let mut idx: Vec<usize> = (0..metrics.len()).collect();
        idx.sort_by(|&a, &b| {
            let sa = score(&metrics[a], objective);
            let sb = score(&metrics[b], objective);
            sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
        });
        idx.truncate(top_k.min(metrics.len()));
        idx.into_iter()
            .map(|i| (combos[i].clone(), metrics[i].clone()))
            .collect()
    };

    let all = if include_all {
        Some(
            combos
                .iter()
                .cloned()
                .zip(metrics.into_iter())
                .collect(),
        )
    } else {
        None
    };

    Some(OptimizationResult {
        best_params,
        best_metrics,
        top,
        all,
        heatmap: None,
        num_combos: combos.len(),
        num_candles,
    })
}
