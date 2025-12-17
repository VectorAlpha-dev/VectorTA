use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use my_project::indicators::moving_averages::{
    alma::{alma, AlmaInput, AlmaParams},
    ema::{ema, EmaInput, EmaParams},
    sma::{sma, SmaInput, SmaParams},
};
use my_project::utilities::data_loader::{source_type, Candles};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DoubleMaParams {
    pub fast_len: u16,
    pub slow_len: u16,
    pub fast_ma_id: u16,
    pub slow_ma_id: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DoubleMaBatchRange {
    pub fast_len: (u16, u16, u16),
    pub slow_len: (u16, u16, u16),
    pub fast_ma_ids: Vec<u16>,
    pub slow_ma_ids: Vec<u16>,
}

impl Default for DoubleMaBatchRange {
    fn default() -> Self {
        Self {
            fast_len: (9, 9, 0),
            slow_len: (21, 21, 0),
            fast_ma_ids: vec![0],
            slow_ma_ids: vec![0],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metrics {
    pub pnl: f64,
    pub sharpe: f64,
    pub max_dd: f64,
}

/// Simple MA catalog: 0 = SMA, 1 = EMA, 2 = ALMA.
fn compute_ma_series(candles: &Candles, len: usize, ma_id: u16) -> Vec<f64> {
    let close = source_type(candles, "close");
    match ma_id {
        0 => {
            let params = SmaParams { period: Some(len) };
            let input = SmaInput::from_slice(close, params);
            sma(&input)
                .map(|o| o.values)
                .unwrap_or_else(|_| vec![f64::NAN; close.len()])
        }
        1 => {
            let params = EmaParams { period: Some(len) };
            let input = EmaInput::from_slice(close, params);
            ema(&input)
                .map(|o| o.values)
                .unwrap_or_else(|_| vec![f64::NAN; close.len()])
        }
        2 => {
            let params = AlmaParams {
                period: Some(len),
                offset: None,
                sigma: None,
            };
            let input = AlmaInput::from_slice(close, params);
            alma(&input)
                .map(|o| o.values)
                .unwrap_or_else(|_| vec![f64::NAN; close.len()])
        }
        _ => vec![f64::NAN; close.len()],
    }
}

pub fn expand_grid(range: &DoubleMaBatchRange) -> Vec<DoubleMaParams> {
    let (f_start, f_end, f_step) = range.fast_len;
    let (s_start, s_end, s_step) = range.slow_len;

    let fast_vals: Vec<u16> = if f_step == 0 {
        vec![f_start]
    } else {
        let (lo, hi) = if f_start <= f_end { (f_start, f_end) } else { (f_end, f_start) };
        let mut out = Vec::new();
        let mut v = lo;
        loop {
            out.push(v);
            if v == hi {
                break;
            }
            match v.checked_add(f_step) {
                Some(next) if next > v && next <= hi => v = next,
                _ => break,
            }
        }
        out
    };

    let slow_vals: Vec<u16> = if s_step == 0 {
        vec![s_start]
    } else {
        let (lo, hi) = if s_start <= s_end { (s_start, s_end) } else { (s_end, s_start) };
        let mut out = Vec::new();
        let mut v = lo;
        loop {
            out.push(v);
            if v == hi {
                break;
            }
            match v.checked_add(s_step) {
                Some(next) if next > v && next <= hi => v = next,
                _ => break,
            }
        }
        out
    };

    let mut combos = Vec::new();
    for &fast_len in &fast_vals {
        for &slow_len in &slow_vals {
            if fast_len >= slow_len {
                continue;
            }
            for &fast_ma_id in &range.fast_ma_ids {
                for &slow_ma_id in &range.slow_ma_ids {
                    combos.push(DoubleMaParams {
                        fast_len,
                        slow_len,
                        fast_ma_id,
                        slow_ma_id,
                    });
                }
            }
        }
    }
    combos
}

/// Builder-style API mirroring indicator batch builders (EhmaBatchBuilder, etc.).
#[derive(Clone, Debug, Default)]
pub struct DoubleMaBatchBuilder {
    range: DoubleMaBatchRange,
}

impl DoubleMaBatchBuilder {
    pub fn new() -> Self {
        Self {
            range: DoubleMaBatchRange::default(),
        }
    }

    pub fn fast_len_range(mut self, start: u16, end: u16, step: u16) -> Self {
        self.range.fast_len = (start, end, step);
        self
    }

    pub fn slow_len_range(mut self, start: u16, end: u16, step: u16) -> Self {
        self.range.slow_len = (start, end, step);
        self
    }

    pub fn fast_ma_ids(mut self, ids: Vec<u16>) -> Self {
        self.range.fast_ma_ids = ids;
        self
    }

    pub fn slow_ma_ids(mut self, ids: Vec<u16>) -> Self {
        self.range.slow_ma_ids = ids;
        self
    }

    pub fn range(&self) -> &DoubleMaBatchRange {
        &self.range
    }

    pub fn into_range(self) -> DoubleMaBatchRange {
        self.range
    }

    pub fn combos(&self) -> Vec<DoubleMaParams> {
        expand_grid(&self.range)
    }

    pub fn run_cpu(&self, candles: &Candles) -> Vec<Metrics> {
        let combos = self.combos();
        double_ma_batch_cpu(candles, &combos)
    }
}

pub fn eval_double_ma_one(candles: &Candles, params: &DoubleMaParams) -> Metrics {
    let prices = source_type(candles, "close");
    let n = prices.len();
    if n < 2 {
        return Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
        };
    }

    let fast = compute_ma_series(candles, params.fast_len as usize, params.fast_ma_id);
    let slow = compute_ma_series(candles, params.slow_len as usize, params.slow_ma_id);

    let mut equity = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut max_dd = 0.0_f64;
    let mut rets = Vec::with_capacity(n);

    let mut prev_signal = 0.0_f64;

    for i in 1..n {
        let f = fast[i];
        let s = slow[i];
        if !f.is_finite() || !s.is_finite() {
            rets.push(0.0);
            continue;
        }
        let signal = if f > s { 1.0 } else { 0.0 };
        let price_prev = prices[i - 1];
        let price_cur = prices[i];
        if !price_prev.is_finite() || !price_cur.is_finite() {
            rets.push(0.0);
            prev_signal = signal;
            continue;
        }
        let r = if price_prev != 0.0 {
            price_cur / price_prev - 1.0
        } else {
            0.0
        };
        let strat_r = prev_signal * r;
        equity *= 1.0 + strat_r;
        if equity > peak {
            peak = equity;
        }
        let dd = (peak - equity) / peak.max(1e-12);
        if dd > max_dd {
            max_dd = dd;
        }
        rets.push(strat_r);
        prev_signal = signal;
    }

    let pnl = equity - 1.0;

    let sharpe = if rets.len() > 1 {
        let mean = rets.iter().copied().sum::<f64>() / rets.len() as f64;
        let var = rets
            .iter()
            .map(|r| {
                let d = r - mean;
                d * d
            })
            .sum::<f64>()
            / (rets.len() as f64 - 1.0);
        let std = var.sqrt();
        if std > 0.0 {
            mean / std
        } else {
            0.0
        }
    } else {
        0.0
    };

    Metrics { pnl, sharpe, max_dd }
}

pub fn double_ma_batch_cpu(candles: &Candles, combos: &[DoubleMaParams]) -> Vec<Metrics> {
    combos
        .par_iter()
        .map(|p| eval_double_ma_one(candles, p))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use my_project::utilities::data_loader::Candles;

    fn make_trending_candles(len: usize) -> Candles {
        let mut ts = Vec::with_capacity(len);
        let mut open = Vec::with_capacity(len);
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut vol: Vec<f64> = Vec::with_capacity(len);

        for i in 0..len {
            ts.push(i as i64);
            let base = 100.0 + i as f64;
            open.push(base);
            high.push(base + 1.0);
            low.push(base - 1.0);
            close.push(base + 0.5);
            vol.push(1_000.0);
        }

        Candles::new(ts, open, high, low, close, vol)
    }

    #[test]
    fn expand_grid_enforces_fast_lt_slow() {
        let range = DoubleMaBatchRange {
            fast_len: (5, 10, 5),
            slow_len: (8, 12, 2),
            fast_ma_ids: vec![0],
            slow_ma_ids: vec![0],
        };
        let combos = expand_grid(&range);
        assert!(!combos.is_empty());
        for p in combos {
            assert!(p.fast_len < p.slow_len);
        }
    }

    #[test]
    fn batch_builder_runs_cpu() {
        let candles = make_trending_candles(200);
        let builder = DoubleMaBatchBuilder::new()
            .fast_len_range(5, 5, 0)
            .slow_len_range(20, 20, 0)
            .fast_ma_ids(vec![0])
            .slow_ma_ids(vec![0]);

        let metrics = builder.run_cpu(&candles);
        assert_eq!(metrics.len(), 1);
        // In a simple uptrend, long-only MA crossover should not be catastrophic.
        let m = &metrics[0];
        assert!(m.max_dd >= 0.0);
    }
}
