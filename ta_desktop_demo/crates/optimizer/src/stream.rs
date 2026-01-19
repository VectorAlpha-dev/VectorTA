use crate::{ObjectiveKind, OptimizationHeatmap, OptimizationResult};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use ta_strategies::double_ma::{DoubleMaParams, Metrics};

fn score(metrics: &Metrics, objective: ObjectiveKind) -> f64 {
    let s = match objective {
        ObjectiveKind::Pnl => metrics.pnl,
        ObjectiveKind::Sharpe => metrics.sharpe,
        ObjectiveKind::MaxDrawdown => -metrics.max_dd,
    };
    if s.is_finite() { s } else { f64::NEG_INFINITY }
}

#[derive(Clone, Debug)]
struct ScoredCombo {
    score: f64,
    params: DoubleMaParams,
    metrics: Metrics,
}

#[derive(Clone, Debug)]
struct HeatmapAgg {
    bins_fast: usize,
    bins_slow: usize,
    fast_min: u16,
    fast_max: u16,
    slow_min: u16,
    slow_max: u16,
    values: Vec<f64>,
}

impl HeatmapAgg {
    fn new(
        bins_fast: usize,
        bins_slow: usize,
        fast_min: u16,
        fast_max: u16,
        slow_min: u16,
        slow_max: u16,
    ) -> Self {
        let bins_fast = bins_fast.max(1);
        let bins_slow = bins_slow.max(1);
        let len = bins_fast.saturating_mul(bins_slow);
        Self {
            bins_fast,
            bins_slow,
            fast_min,
            fast_max,
            slow_min,
            slow_max,
            values: vec![f64::NEG_INFINITY; len],
        }
    }

    fn bin_period(v: u16, v_min: u16, v_max: u16, bins: usize) -> usize {
        if bins <= 1 {
            return 0;
        }
        let denom = v_max.saturating_sub(v_min) as usize;
        if denom == 0 {
            return 0;
        }
        let num = v.saturating_sub(v_min) as usize;
        (num.saturating_mul(bins - 1)) / denom
    }

    fn push(&mut self, fast_len: u16, slow_len: u16, score: f64) {
        if !score.is_finite() || self.values.is_empty() {
            return;
        }

        let f_bin = Self::bin_period(fast_len, self.fast_min, self.fast_max, self.bins_fast)
            .min(self.bins_fast - 1);
        let s_bin = Self::bin_period(slow_len, self.slow_min, self.slow_max, self.bins_slow)
            .min(self.bins_slow - 1);

        let idx = f_bin.saturating_mul(self.bins_slow).saturating_add(s_bin);
        if let Some(cell) = self.values.get_mut(idx) {
            if score > *cell {
                *cell = score;
            }
        }
    }

    fn finalize(self) -> OptimizationHeatmap {
        let values: Vec<Option<f64>> = self
            .values
            .into_iter()
            .map(|v| if v.is_finite() && v != f64::NEG_INFINITY { Some(v) } else { None })
            .collect();
        OptimizationHeatmap {
            bins_fast: self.bins_fast,
            bins_slow: self.bins_slow,
            fast_min: self.fast_min,
            fast_max: self.fast_max,
            slow_min: self.slow_min,
            slow_max: self.slow_max,
            values,
        }
    }
}

impl PartialEq for ScoredCombo {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for ScoredCombo {}

impl PartialOrd for ScoredCombo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredCombo {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score)
    }
}

pub struct StreamAggregator {
    objective: ObjectiveKind,
    top_k: usize,
    num_candles: usize,

    num_combos: usize,
    best: Option<ScoredCombo>,
    heap: BinaryHeap<Reverse<ScoredCombo>>,
    all: Option<Vec<(DoubleMaParams, Metrics)>>,
    heatmap: Option<HeatmapAgg>,
}

impl StreamAggregator {
    pub fn new(
        objective: ObjectiveKind,
        top_k: usize,
        include_all: bool,
        num_candles: usize,
    ) -> Self {
        Self {
            objective,
            top_k,
            num_candles,
            num_combos: 0,
            best: None,
            heap: BinaryHeap::new(),
            all: if include_all { Some(Vec::new()) } else { None },
            heatmap: None,
        }
    }

    pub fn with_heatmap(
        mut self,
        bins_fast: usize,
        bins_slow: usize,
        fast_min: u16,
        fast_max: u16,
        slow_min: u16,
        slow_max: u16,
    ) -> Self {
        self.heatmap = Some(HeatmapAgg::new(
            bins_fast,
            bins_slow,
            fast_min,
            fast_max,
            slow_min,
            slow_max,
        ));
        self
    }

    pub fn push(&mut self, params: DoubleMaParams, metrics: Metrics) {
        self.num_combos = self.num_combos.saturating_add(1);
        let s = score(&metrics, self.objective);

        if let Some(all) = self.all.as_mut() {
            all.push((params.clone(), metrics.clone()));
        }

        if let Some(h) = self.heatmap.as_mut() {
            h.push(params.fast_len, params.slow_len, s);
        }

        match self.best.as_ref() {
            None => {
                self.best = Some(ScoredCombo {
                    score: s,
                    params: params.clone(),
                    metrics: metrics.clone(),
                });
            }
            Some(b) if s > b.score => {
                self.best = Some(ScoredCombo {
                    score: s,
                    params: params.clone(),
                    metrics: metrics.clone(),
                });
            }
            _ => {}
        }

        if self.top_k == 0 {
            return;
        }

        let entry = ScoredCombo {
            score: s,
            params,
            metrics,
        };
        if self.heap.len() < self.top_k {
            self.heap.push(Reverse(entry));
            return;
        }

        if let Some(min_top) = self.heap.peek() {
            if entry.score > min_top.0.score {
                let _ = self.heap.pop();
                self.heap.push(Reverse(entry));
            }
        }
    }

    pub fn num_combos(&self) -> usize {
        self.num_combos
    }

    pub fn merge(&mut self, other: StreamAggregator) {
        if self.objective != other.objective || self.num_candles != other.num_candles {
            return;
        }

        if self.top_k != other.top_k {
            return;
        }

        self.num_combos = self.num_combos.saturating_add(other.num_combos);

        if let (Some(a), Some(mut b)) = (self.all.as_mut(), other.all) {
            a.append(&mut b);
        }

        match (self.best.take(), other.best) {
            (None, None) => {}
            (None, Some(b)) => self.best = Some(b),
            (Some(a), None) => self.best = Some(a),
            (Some(a), Some(b)) => {
                self.best = Some(if b.score > a.score { b } else { a });
            }
        }

        if self.top_k > 0 {
            for rev in other.heap {
                let entry = rev.0;
                if self.heap.len() < self.top_k {
                    self.heap.push(Reverse(entry));
                    continue;
                }
                if let Some(min_top) = self.heap.peek() {
                    if entry.score > min_top.0.score {
                        let _ = self.heap.pop();
                        self.heap.push(Reverse(entry));
                    }
                }
            }
        }

        match (self.heatmap.as_mut(), other.heatmap) {
            (Some(a), Some(b)) => {
                if a.bins_fast != b.bins_fast
                    || a.bins_slow != b.bins_slow
                    || a.fast_min != b.fast_min
                    || a.fast_max != b.fast_max
                    || a.slow_min != b.slow_min
                    || a.slow_max != b.slow_max
                {
                    return;
                }
                for (dst, src) in a.values.iter_mut().zip(b.values.into_iter()) {
                    if src > *dst {
                        *dst = src;
                    }
                }
            }
            (None, Some(b)) => {
                self.heatmap = Some(b);
            }
            _ => {}
        }
    }

    pub fn finalize(self) -> Option<OptimizationResult> {
        let best = self.best?;

        let mut top: Vec<ScoredCombo> = self.heap.into_iter().map(|r| r.0).collect();
        top.sort_by(|a, b| b.score.total_cmp(&a.score));
        let top = top.into_iter().map(|x| (x.params, x.metrics)).collect();

        Some(OptimizationResult {
            best_params: best.params,
            best_metrics: best.metrics,
            top,
            all: self.all,
            heatmap: self.heatmap.map(|h| h.finalize()),
            num_combos: self.num_combos,
            num_candles: self.num_candles,
        })
    }
}
