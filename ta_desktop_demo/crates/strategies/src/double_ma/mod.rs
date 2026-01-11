use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use my_project::indicators::moving_averages::ma::{ma, MaData};
use my_project::indicators::moving_averages::ma_batch::ma_batch;
use my_project::utilities::data_loader::{source_type, Candles};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DoubleMaParams {
    pub fast_len: u16,
    pub slow_len: u16,
    /// Index into `DoubleMaBatchRange.fast_ma_types`.
    pub fast_ma_id: u16,
    /// Index into `DoubleMaBatchRange.slow_ma_types`.
    pub slow_ma_id: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DoubleMaBatchRange {
    pub fast_len: (u16, u16, u16),
    pub slow_len: (u16, u16, u16),
    pub fast_ma_types: Vec<String>,
    pub slow_ma_types: Vec<String>,
}

impl Default for DoubleMaBatchRange {
    fn default() -> Self {
        Self {
            fast_len: (9, 9, 0),
            slow_len: (21, 21, 0),
            fast_ma_types: vec!["sma".to_string()],
            slow_ma_types: vec!["sma".to_string()],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metrics {
    pub pnl: f64,
    pub sharpe: f64,
    pub max_dd: f64,
}

fn compute_ma_series(candles: &Candles, len: usize, ma_type: &str, source: &str) -> Vec<f64> {
    let n = candles.close.len();
    ma(
        ma_type,
        MaData::Candles {
            candles,
            source,
        },
        len,
    )
    .unwrap_or_else(|_| vec![f64::NAN; n])
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
            let max_fast: u16 = if range.fast_ma_types.len() > u16::MAX as usize {
                u16::MAX
            } else {
                range.fast_ma_types.len() as u16
            };
            let max_slow: u16 = if range.slow_ma_types.len() > u16::MAX as usize {
                u16::MAX
            } else {
                range.slow_ma_types.len() as u16
            };
            for fast_ma_id in 0..max_fast {
                for slow_ma_id in 0..max_slow {
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

    pub fn fast_ma_types(mut self, types: Vec<String>) -> Self {
        self.range.fast_ma_types = types;
        self
    }

    pub fn slow_ma_types(mut self, types: Vec<String>) -> Self {
        self.range.slow_ma_types = types;
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
        double_ma_batch_cpu(
            candles,
            &combos,
            &self.range.fast_ma_types,
            &self.range.slow_ma_types,
            "close",
        )
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

    // Backwards-compat: treat ids as a minimal catalog.
    // Prefer `eval_double_ma_one_with_types` for selector-based MA dispatch.
    let fast = match params.fast_ma_id {
        0 => compute_ma_series(candles, params.fast_len as usize, "sma", "close"),
        1 => compute_ma_series(candles, params.fast_len as usize, "ema", "close"),
        2 => compute_ma_series(candles, params.fast_len as usize, "alma", "close"),
        _ => vec![f64::NAN; n],
    };
    let slow = match params.slow_ma_id {
        0 => compute_ma_series(candles, params.slow_len as usize, "sma", "close"),
        1 => compute_ma_series(candles, params.slow_len as usize, "ema", "close"),
        2 => compute_ma_series(candles, params.slow_len as usize, "alma", "close"),
        _ => vec![f64::NAN; n],
    };

    eval_double_ma_from_series(prices, &fast, &slow)
}

fn eval_double_ma_from_series(prices: &[f64], fast: &[f64], slow: &[f64]) -> Metrics {
    let n = prices.len();
    if n < 2 || fast.len() != n || slow.len() != n {
        return Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
        };
    }

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

pub fn eval_double_ma_one_with_types(
    candles: &Candles,
    params: &DoubleMaParams,
    fast_ma_types: &[String],
    slow_ma_types: &[String],
) -> Metrics {
    let prices = source_type(candles, "close");
    if prices.len() < 2 {
        return Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
        };
    }

    let fast_type = fast_ma_types
        .get(params.fast_ma_id as usize)
        .map(|s| s.as_str());
    let slow_type = slow_ma_types
        .get(params.slow_ma_id as usize)
        .map(|s| s.as_str());
    let (Some(fast_type), Some(slow_type)) = (fast_type, slow_type) else {
        return Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
        };
    };

    let fast = compute_ma_series(candles, params.fast_len as usize, fast_type, "close");
    let slow = compute_ma_series(candles, params.slow_len as usize, slow_type, "close");

    eval_double_ma_from_series(prices, &fast, &slow)
}

pub fn double_ma_batch_cpu(
    candles: &Candles,
    combos: &[DoubleMaParams],
    fast_ma_types: &[String],
    slow_ma_types: &[String],
    ma_source: &str,
) -> Vec<Metrics> {
    if combos.is_empty() {
        return Vec::new();
    }

    let prices = source_type(candles, "close");
    let n = prices.len();
    if n < 2 {
        return combos
            .iter()
            .map(|_| Metrics {
                pnl: 0.0,
                sharpe: 0.0,
                max_dd: 0.0,
            })
            .collect();
    }

    let matrices = build_ma_matrices_cpu(candles, combos, fast_ma_types, slow_ma_types, n, ma_source);
    let nan_row: Vec<f64> = vec![f64::NAN; n];

    combos
        .par_iter()
        .map(|p| {
            let fast_type = fast_ma_types
                .get(p.fast_ma_id as usize)
                .map(|s| s.as_str());
            let slow_type = slow_ma_types
                .get(p.slow_ma_id as usize)
                .map(|s| s.as_str());

            let (Some(fast_type), Some(slow_type)) = (fast_type, slow_type) else {
                return Metrics {
                    pnl: 0.0,
                    sharpe: 0.0,
                    max_dd: 0.0,
                };
            };

            let fast = matrices
                .get(fast_type)
                .and_then(|m| m.row(p.fast_len, n))
                .unwrap_or(&nan_row);
            let slow = matrices
                .get(slow_type)
                .and_then(|m| m.row(p.slow_len, n))
                .unwrap_or(&nan_row);

            eval_double_ma_from_series(prices, fast, slow)
        })
        .collect()
}

#[derive(Clone, Debug)]
struct PeriodMatrixF64 {
    period_start: u16,
    period_end: u16,
    values: Vec<f64>, // row-major: period-major rows, time-major cols
}

impl PeriodMatrixF64 {
    fn row(&self, period: u16, cols: usize) -> Option<&[f64]> {
        if period < self.period_start || period > self.period_end {
            return None;
        }
        let row = (period - self.period_start) as usize;
        let offset = row.checked_mul(cols)?;
        let end = offset.checked_add(cols)?;
        self.values.get(offset..end)
    }
}

fn update_minmax<'a>(map: &mut HashMap<&'a str, (u16, u16)>, ma_type: &'a str, period: u16) {
    let entry = map.entry(ma_type).or_insert((period, period));
    if period < entry.0 {
        entry.0 = period;
    }
    if period > entry.1 {
        entry.1 = period;
    }
}

fn build_ma_matrices_cpu<'a>(
    candles: &'a Candles,
    combos: &[DoubleMaParams],
    fast_ma_types: &'a [String],
    slow_ma_types: &'a [String],
    series_len: usize,
    ma_source: &str,
) -> HashMap<&'a str, PeriodMatrixF64> {
    let max_period_supported: u16 = if series_len >= u16::MAX as usize {
        u16::MAX
    } else {
        series_len as u16
    };

    let mut minmax: HashMap<&str, (u16, u16)> = HashMap::new();
    for p in combos {
        if p.fast_len > 0 {
            if let Some(t) = fast_ma_types.get(p.fast_ma_id as usize) {
                update_minmax(&mut minmax, t.as_str(), p.fast_len);
            }
        }
        if p.slow_len > 0 {
            if let Some(t) = slow_ma_types.get(p.slow_ma_id as usize) {
                update_minmax(&mut minmax, t.as_str(), p.slow_len);
            }
        }
    }

    let mut out: HashMap<&str, PeriodMatrixF64> = HashMap::new();
    for (&ma_type, &(min_p, max_p)) in &minmax {
        if min_p == 0 || max_p == 0 {
            continue;
        }

        let start = min_p;
        let end = max_p.min(max_period_supported);
        if start > end {
            continue;
        }

        let n_periods = (end as u32)
            .saturating_sub(start as u32)
            .saturating_add(1) as usize;
        let expected_len = match n_periods.checked_mul(series_len) {
            Some(v) => v,
            None => continue,
        };

        let mut values: Vec<f64> = vec![f64::NAN; expected_len];

        // Prefer the batch dispatcher (SIMD + scalar batch kernels). If unsupported, fall back
        // to per-period `ma()` calls but still cache results per MA type.
        let batch = ma_batch(
            ma_type,
            MaData::Candles {
                candles,
                source: ma_source,
            },
            (start as usize, end as usize, 1),
        );
        match batch {
            Ok(b) if b.cols == series_len => {
                for (src_row, &period) in b.periods.iter().enumerate() {
                    if period < start as usize || period > end as usize {
                        continue;
                    }
                    let dst_row = period - start as usize;
                    let dst_off = match dst_row.checked_mul(series_len) {
                        Some(v) => v,
                        None => continue,
                    };
                    let src_off = match src_row.checked_mul(b.cols) {
                        Some(v) => v,
                        None => continue,
                    };
                    if src_off + series_len > b.values.len() || dst_off + series_len > values.len() {
                        continue;
                    }
                    values[dst_off..dst_off + series_len]
                        .copy_from_slice(&b.values[src_off..src_off + series_len]);
                }
            }
            _ => {
                for (row, period) in (start..=end).enumerate() {
                    let series = compute_ma_series(candles, period as usize, ma_type, ma_source);
                    if series.len() != series_len {
                        continue;
                    }
                    let off = match row.checked_mul(series_len) {
                        Some(v) => v,
                        None => continue,
                    };
                    if off + series_len > values.len() {
                        continue;
                    }
                    values[off..off + series_len].copy_from_slice(&series);
                }
            }
        }

        out.insert(
            ma_type,
            PeriodMatrixF64 {
                period_start: start,
                period_end: end,
                values,
            },
        );
    }

    out
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
            fast_ma_types: vec!["sma".to_string()],
            slow_ma_types: vec!["sma".to_string()],
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
            .fast_ma_types(vec!["sma".to_string()])
            .slow_ma_types(vec!["sma".to_string()]);

        let metrics = builder.run_cpu(&candles);
        assert_eq!(metrics.len(), 1);
        // In a simple uptrend, long-only MA crossover should not be catastrophic.
        let m = &metrics[0];
        assert!(m.max_dd >= 0.0);
    }
}
