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

    pub fast_ma_id: u16,

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
    pub trades: u32,
    pub exposure: f64,
    pub net_exposure: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trade {
    pub entry_t: usize,
    pub exit_t: usize,
    pub direction: i32,
    pub pnl: f64,
    pub bars: u32,
    pub open: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeStats {
    pub closed_trades: usize,
    pub open_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyConfig {
    pub long_only: bool,
    pub allow_flip: bool,
    pub trade_on_next_bar: bool,
    pub commission: f64,
    pub eps_rel: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            long_only: true,
            allow_flip: true,
            trade_on_next_bar: true,
            commission: 0.0,
            eps_rel: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MinMaxBins {
    pub start_t: usize,
    pub end_t: usize,
    pub bins: usize,
    pub minmax: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DoubleMaCurves {
    pub equity: MinMaxBins,
    pub drawdown: MinMaxBins,
    #[serde(default)]
    pub trades: Vec<Trade>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trade_stats: Option<TradeStats>,
}

fn clamp_bins(requested: usize, n_steps: usize) -> usize {
    if n_steps == 0 {
        return 0;
    }
    let req = requested.clamp(1, 8192);
    req.min(n_steps)
}

pub fn eval_double_ma_curves_from_log_returns_and_mas_f64(
    lr: &[f64],
    fast: &[f64],
    slow: &[f64],
    start_t: usize,
    cfg: &StrategyConfig,
    bins: usize,
) -> DoubleMaCurves {
    let n_total = lr.len();
    if n_total < 2 || fast.len() != n_total || slow.len() != n_total || start_t >= n_total {
        return DoubleMaCurves {
            equity: MinMaxBins {
                start_t,
                end_t: n_total,
                bins: 0,
                minmax: Vec::new(),
            },
            drawdown: MinMaxBins {
                start_t,
                end_t: n_total,
                bins: 0,
                minmax: Vec::new(),
            },
            trades: Vec::new(),
            trade_stats: None,
        };
    }

    let trade_offset = if cfg.trade_on_next_bar { 1usize } else { 0usize };
    let log_comm = if cfg.commission > 0.0 {
        (1.0 - cfg.commission).ln()
    } else {
        0.0
    };

    let n_steps = n_total.saturating_sub(start_t);
    let bins = clamp_bins(bins, n_steps);
    if bins == 0 {
        return DoubleMaCurves {
            equity: MinMaxBins {
                start_t,
                end_t: n_total,
                bins: 0,
                minmax: Vec::new(),
            },
            drawdown: MinMaxBins {
                start_t,
                end_t: n_total,
                bins: 0,
                minmax: Vec::new(),
            },
            trades: Vec::new(),
            trade_stats: None,
        };
    }

    let mut eq_min: Vec<f64> = vec![f64::INFINITY; bins];
    let mut eq_max: Vec<f64> = vec![f64::NEG_INFINITY; bins];
    let mut dd_min: Vec<f64> = vec![f64::INFINITY; bins];
    let mut dd_max: Vec<f64> = vec![f64::NEG_INFINITY; bins];

    let mut pos: i32 = 0;
    let mut log_eq = 0.0_f64;
    let mut log_peak = 0.0_f64;

    let mut trades: Vec<Trade> = Vec::new();
    let mut cur_entry_t: usize = 0;
    let mut cur_dir: i32 = 0;
    let mut cur_log_ret: f64 = 0.0;

    for t in start_t..n_total {
        let t_sig = t.saturating_sub(trade_offset);
        let f = fast[t_sig];
        let s = slow[t_sig];
        let eps = if cfg.eps_rel > 0.0 {
            cfg.eps_rel * 1.0_f64.max(s.abs())
        } else {
            0.0
        };

        let mut sign: i32 = (f > s + eps) as i32 - (f < s - eps) as i32;
        if cfg.long_only && sign < 0 {
            sign = 0;
        }

        if sign != pos {
            if pos == 0 || sign == 0 {
                if cfg.commission > 0.0 {
                    log_eq += log_comm;
                }
                if pos != 0 && cur_dir != 0 {
                    if cfg.commission > 0.0 {
                        cur_log_ret += log_comm;
                    }
                    trades.push(Trade {
                        entry_t: cur_entry_t,
                        exit_t: t,
                        direction: cur_dir,
                        pnl: cur_log_ret.exp() - 1.0,
                        bars: t.saturating_sub(cur_entry_t) as u32,
                        open: false,
                    });
                    cur_dir = 0;
                    cur_log_ret = 0.0;
                } else if pos == 0 && sign != 0 {
                    cur_entry_t = t;
                    cur_dir = sign;
                    cur_log_ret = if cfg.commission > 0.0 { log_comm } else { 0.0 };
                }
                pos = if sign == 0 { 0 } else { sign };
            } else if !cfg.allow_flip {
                if cfg.commission > 0.0 {
                    log_eq += log_comm;
                }
                if cur_dir != 0 {
                    if cfg.commission > 0.0 {
                        cur_log_ret += log_comm;
                    }
                    trades.push(Trade {
                        entry_t: cur_entry_t,
                        exit_t: t,
                        direction: cur_dir,
                        pnl: cur_log_ret.exp() - 1.0,
                        bars: t.saturating_sub(cur_entry_t) as u32,
                        open: false,
                    });
                    cur_dir = 0;
                    cur_log_ret = 0.0;
                }
                pos = 0;
            } else {
                if cfg.commission > 0.0 {
                    log_eq += 2.0 * log_comm;
                }
                if cur_dir != 0 {
                    if cfg.commission > 0.0 {
                        cur_log_ret += log_comm;
                    }
                    trades.push(Trade {
                        entry_t: cur_entry_t,
                        exit_t: t,
                        direction: cur_dir,
                        pnl: cur_log_ret.exp() - 1.0,
                        bars: t.saturating_sub(cur_entry_t) as u32,
                        open: false,
                    });
                }
                cur_entry_t = t;
                cur_dir = sign;
                cur_log_ret = if cfg.commission > 0.0 { log_comm } else { 0.0 };
                pos = sign;
            }
        }

        log_eq += (pos as f64) * lr[t];
        if cur_dir != 0 {
            cur_log_ret += (pos as f64) * lr[t];
        }
        if log_eq > log_peak {
            log_peak = log_eq;
        }
        let dd_log = log_peak - log_eq;

        let eq = log_eq.exp();
        let dd = if dd_log > 0.0 { -(-dd_log).exp_m1() } else { 0.0 };

        let i = t - start_t;
        let b = (i.saturating_mul(bins)) / n_steps.max(1);
        let b = b.min(bins - 1);

        if eq < eq_min[b] {
            eq_min[b] = eq;
        }
        if eq > eq_max[b] {
            eq_max[b] = eq;
        }
        if dd < dd_min[b] {
            dd_min[b] = dd;
        }
        if dd > dd_max[b] {
            dd_max[b] = dd;
        }
    }

    let mut eq_mm: Vec<f32> = Vec::with_capacity(bins.saturating_mul(2));
    let mut dd_mm: Vec<f32> = Vec::with_capacity(bins.saturating_mul(2));
    for i in 0..bins {
        let (a, b) = (eq_min[i], eq_max[i]);
        if a.is_finite() && b.is_finite() {
            eq_mm.push(a as f32);
            eq_mm.push(b as f32);
        } else {
            eq_mm.push(f32::NAN);
            eq_mm.push(f32::NAN);
        }

        let (a, b) = (dd_min[i], dd_max[i]);
        if a.is_finite() && b.is_finite() {
            dd_mm.push(a as f32);
            dd_mm.push(b as f32);
        } else {
            dd_mm.push(f32::NAN);
            dd_mm.push(f32::NAN);
        }
    }

    if cur_dir != 0 {
        let exit_t = n_total.saturating_sub(1);
        trades.push(Trade {
            entry_t: cur_entry_t,
            exit_t,
            direction: cur_dir,
            pnl: cur_log_ret.exp() - 1.0,
            bars: exit_t.saturating_sub(cur_entry_t) as u32,
            open: true,
        });
    }

    let trade_stats: Option<TradeStats> = {
        let mut open_trades: usize = 0;
        let mut closed_trades: usize = 0;
        let mut wins: usize = 0;
        let mut losses: usize = 0;
        let mut sum_win: f64 = 0.0;
        let mut sum_loss_abs: f64 = 0.0;
        let mut sum_win_count: usize = 0;
        let mut sum_loss_count: usize = 0;
        let mut sum_all: f64 = 0.0;

        for tr in trades.iter() {
            if tr.open {
                open_trades += 1;
                continue;
            }
            closed_trades += 1;
            let r = tr.pnl;
            sum_all += r;
            if r > 0.0 {
                wins += 1;
                sum_win += r;
                sum_win_count += 1;
            } else if r < 0.0 {
                losses += 1;
                sum_loss_abs += -r;
                sum_loss_count += 1;
            }
        }

        if closed_trades == 0 && open_trades == 0 {
            None
        } else {
            let denom = wins + losses;
            let win_rate = if denom > 0 { wins as f64 / denom as f64 } else { 0.0 };
            let avg_win = if sum_win_count > 0 { sum_win / sum_win_count as f64 } else { 0.0 };
            let avg_loss = if sum_loss_count > 0 { -(sum_loss_abs / sum_loss_count as f64) } else { 0.0 };
            let profit_factor = if sum_loss_abs > 0.0 {
                sum_win / sum_loss_abs
            } else if sum_win > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            let expectancy = if closed_trades > 0 { sum_all / closed_trades as f64 } else { 0.0 };

            Some(TradeStats {
                closed_trades,
                open_trades,
                wins,
                losses,
                win_rate,
                avg_win,
                avg_loss,
                profit_factor,
                expectancy,
            })
        }
    };

    DoubleMaCurves {
        equity: MinMaxBins {
            start_t,
            end_t: n_total,
            bins,
            minmax: eq_mm,
        },
        drawdown: MinMaxBins {
            start_t,
            end_t: n_total,
            bins,
            minmax: dd_mm,
        },
        trades,
        trade_stats,
    }
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
            trades: 0,
            exposure: 0.0,
            net_exposure: 0.0,
        };
    }



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

    let first_valid = first_finite_idx(prices).unwrap_or(0);
    eval_double_ma_from_series_cfg(
        prices,
        &fast,
        &slow,
        params.fast_len,
        params.slow_len,
        first_valid,
        &StrategyConfig::default(),
    )
}

fn eval_double_ma_from_series(prices: &[f64], fast: &[f64], slow: &[f64]) -> Metrics {
    let n = prices.len();
    if n < 2 || fast.len() != n || slow.len() != n {
        return Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
            trades: 0,
            exposure: 0.0,
            net_exposure: 0.0,
        };
    }

    let first_valid = first_finite_idx(prices).unwrap_or(0);
    let t_sig0 = fast
        .iter()
        .zip(slow.iter())
        .position(|(f, s)| f.is_finite() && s.is_finite())
        .unwrap_or(n);
    let start_t = t_sig0.saturating_add(if StrategyConfig::default().trade_on_next_bar { 1 } else { 0 });

    eval_double_ma_from_log_returns_and_mas_f64(
        &compute_log_returns(prices),
        fast,
        slow,
        start_t.max(first_valid),
        &StrategyConfig::default(),
    )
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
            trades: 0,
            exposure: 0.0,
            net_exposure: 0.0,
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
            trades: 0,
            exposure: 0.0,
            net_exposure: 0.0,
        };
    };

    let fast = compute_ma_series(candles, params.fast_len as usize, fast_type, "close");
    let slow = compute_ma_series(candles, params.slow_len as usize, slow_type, "close");

    let first_valid = first_finite_idx(prices).unwrap_or(0);
    eval_double_ma_from_series_cfg(
        prices,
        &fast,
        &slow,
        params.fast_len,
        params.slow_len,
        first_valid,
        &StrategyConfig::default(),
    )
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
                trades: 0,
                exposure: 0.0,
                net_exposure: 0.0,
            })
            .collect();
    }

    let cfg = StrategyConfig::default();
    let lr = compute_log_returns(prices);
    let first_valid = first_finite_idx(prices).unwrap_or(0);
    let trade_offset = if cfg.trade_on_next_bar { 1usize } else { 0usize };

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
                    trades: 0,
                    exposure: 0.0,
                    net_exposure: 0.0,
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

            let max_p = (p.fast_len as usize).max(p.slow_len as usize).max(1);
            let t_valid = first_valid.saturating_add(max_p.saturating_sub(1));
            let start_t = t_valid.saturating_add(trade_offset);
            eval_double_ma_from_log_returns_and_mas_f64(&lr, fast, slow, start_t, &cfg)
        })
        .collect()
}

#[inline]
fn first_finite_idx(series: &[f64]) -> Option<usize> {
    series.iter().position(|v| v.is_finite())
}

fn compute_log_returns(prices: &[f64]) -> Vec<f64> {
    let n = prices.len();
    let mut out = vec![0.0_f64; n];
    if n < 2 {
        return out;
    }
    for t in 1..n {
        let p = prices[t];
        let pm = prices[t - 1];
        out[t] = if p > 0.0 && pm > 0.0 && p.is_finite() && pm.is_finite() {
            p.ln() - pm.ln()
        } else {
            0.0
        };
    }
    out
}

pub fn eval_double_ma_from_series_cfg(
    prices: &[f64],
    fast: &[f64],
    slow: &[f64],
    fast_period: u16,
    slow_period: u16,
    first_valid: usize,
    cfg: &StrategyConfig,
) -> Metrics {
    if prices.len() < 2 || fast.len() != prices.len() || slow.len() != prices.len() {
        return Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
            trades: 0,
            exposure: 0.0,
            net_exposure: 0.0,
        };
    }
    let lr = compute_log_returns(prices);
    let trade_offset = if cfg.trade_on_next_bar { 1usize } else { 0usize };
    let max_p = (fast_period as usize).max(slow_period as usize).max(1);
    let t_valid = first_valid.saturating_add(max_p.saturating_sub(1));
    let start_t = t_valid.saturating_add(trade_offset);
    eval_double_ma_from_log_returns_and_mas_f64(&lr, fast, slow, start_t, cfg)
}

pub fn eval_double_ma_from_log_returns_and_mas_f64(
    lr: &[f64],
    fast: &[f64],
    slow: &[f64],
    start_t: usize,
    cfg: &StrategyConfig,
) -> Metrics {
    eval_double_ma_from_log_returns_and_mas_f64_impl::<true>(lr, fast, slow, start_t, cfg)
}

pub fn eval_double_ma_from_log_returns_and_mas_f64_fast(
    lr: &[f64],
    fast: &[f64],
    slow: &[f64],
    start_t: usize,
    cfg: &StrategyConfig,
) -> Metrics {
    eval_double_ma_from_log_returns_and_mas_f64_impl::<false>(lr, fast, slow, start_t, cfg)
}

fn eval_double_ma_from_log_returns_and_mas_f64_impl<const COMPUTE_SHARPE: bool>(
    lr: &[f64],
    fast: &[f64],
    slow: &[f64],
    start_t: usize,
    cfg: &StrategyConfig,
) -> Metrics {
    let n_total = lr.len();
    if n_total < 2 || fast.len() != n_total || slow.len() != n_total {
        return Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
            trades: 0,
            exposure: 0.0,
            net_exposure: 0.0,
        };
    }

    if start_t >= n_total {
        return Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
            trades: 0,
            exposure: 0.0,
            net_exposure: 0.0,
        };
    }

    if !(0.0..1.0).contains(&cfg.commission) && cfg.commission != 0.0 {
        return Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
            trades: 0,
            exposure: 0.0,
            net_exposure: 0.0,
        };
    }

    let obs: u64 = (n_total - start_t) as u64;

    let log_comm = if cfg.commission > 0.0 {
        (1.0 - cfg.commission).ln()
    } else {
        0.0
    };
    let trade_offset = if cfg.trade_on_next_bar { 1usize } else { 0usize };

    let mut pos: i32 = 0;
    let mut log_eq = 0.0_f64;
    let mut log_peak = 0.0_f64;
    let mut max_log_dd = 0.0_f64;
    let mut trades: u32 = 0;

    let mut count: u64 = 0;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;

    let mut sum_abs_pos: i64 = 0;
    let mut sum_pos: i64 = 0;

    for t in start_t..n_total {
        let t_sig = t.saturating_sub(trade_offset);
        let f = fast[t_sig];
        let s = slow[t_sig];
        let eps = if cfg.eps_rel > 0.0 {
            cfg.eps_rel * 1.0_f64.max(s.abs())
        } else {
            0.0
        };

        let mut sign: i32 = (f > s + eps) as i32 - (f < s - eps) as i32;
        if cfg.long_only && sign < 0 {
            sign = 0;
        }

        if sign != pos {
            if pos == 0 || sign == 0 {
                if cfg.commission > 0.0 {
                    log_eq += log_comm;
                }
                trades = trades.saturating_add(1);
                pos = if sign == 0 { 0 } else { sign };
            } else if !cfg.allow_flip {
                if cfg.commission > 0.0 {
                    log_eq += log_comm;
                }
                trades = trades.saturating_add(1);
                pos = 0;
            } else {
                if cfg.commission > 0.0 {
                    log_eq += 2.0 * log_comm;
                }
                trades = trades.saturating_add(2);
                pos = sign;
            }
        }

        let lr_t = lr[t];
        log_eq += (pos as f64) * lr_t;
        if log_eq > log_peak {
            log_peak = log_eq;
        }
        let dd_log = log_peak - log_eq;
        if dd_log > max_log_dd {
            max_log_dd = dd_log;
        }

        if COMPUTE_SHARPE {
            let step_r = if pos == 0 { 0.0 } else { (pos as f64) * lr_t.exp_m1() };
            count += 1;
            let delta = step_r - mean;
            mean += delta / count as f64;
            let delta2 = step_r - mean;
            m2 += delta * delta2;
        }

        sum_abs_pos += (pos as i64).abs();
        sum_pos += pos as i64;
    }

    let sharpe = if COMPUTE_SHARPE {
        let variance = if count > 1 { m2 / (count as f64 - 1.0) } else { 0.0 };
        let std = variance.sqrt();
        if std > 0.0 { mean / std } else { 0.0 }
    } else {
        0.0
    };
    let denom = if COMPUTE_SHARPE { count } else { obs };
    let exposure = if denom > 0 { sum_abs_pos as f64 / denom as f64 } else { 0.0 };
    let net_exposure = if denom > 0 { sum_pos as f64 / denom as f64 } else { 0.0 };
    let max_dd = if max_log_dd > 0.0 {
        -(-max_log_dd).exp_m1()
    } else {
        0.0
    };

    Metrics {
        pnl: log_eq.exp() - 1.0,
        sharpe,
        max_dd,
        trades,
        exposure,
        net_exposure,
    }
}

pub fn eval_double_ma_from_log_returns_and_mas_f32(
    lr: &[f64],
    fast: &[f32],
    slow: &[f32],
    start_t: usize,
    cfg: &StrategyConfig,
) -> Metrics {
    eval_double_ma_from_log_returns_and_mas_f32_impl::<true>(lr, fast, slow, start_t, cfg)
}

pub fn eval_double_ma_from_log_returns_and_mas_f32_fast(
    lr: &[f64],
    fast: &[f32],
    slow: &[f32],
    start_t: usize,
    cfg: &StrategyConfig,
) -> Metrics {
    eval_double_ma_from_log_returns_and_mas_f32_impl::<false>(lr, fast, slow, start_t, cfg)
}

fn eval_double_ma_from_log_returns_and_mas_f32_impl<const COMPUTE_SHARPE: bool>(
    lr: &[f64],
    fast: &[f32],
    slow: &[f32],
    start_t: usize,
    cfg: &StrategyConfig,
) -> Metrics {
    let n_total = lr.len();
    if n_total < 2 || fast.len() != n_total || slow.len() != n_total {
        return Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
            trades: 0,
            exposure: 0.0,
            net_exposure: 0.0,
        };
    }

    if start_t >= n_total {
        return Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
            trades: 0,
            exposure: 0.0,
            net_exposure: 0.0,
        };
    }

    if !(0.0..1.0).contains(&cfg.commission) && cfg.commission != 0.0 {
        return Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
            trades: 0,
            exposure: 0.0,
            net_exposure: 0.0,
        };
    }

    let obs: u64 = (n_total - start_t) as u64;

    let log_comm = if cfg.commission > 0.0 {
        (1.0 - cfg.commission).ln()
    } else {
        0.0
    };
    let trade_offset = if cfg.trade_on_next_bar { 1usize } else { 0usize };

    let mut pos: i32 = 0;
    let mut log_eq = 0.0_f64;
    let mut log_peak = 0.0_f64;
    let mut max_log_dd = 0.0_f64;
    let mut trades: u32 = 0;

    let mut count: u64 = 0;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;

    let mut sum_abs_pos: i64 = 0;
    let mut sum_pos: i64 = 0;

    for t in start_t..n_total {
        let t_sig = t.saturating_sub(trade_offset);
        let f = fast[t_sig] as f64;
        let s = slow[t_sig] as f64;
        let eps = if cfg.eps_rel > 0.0 {
            cfg.eps_rel * 1.0_f64.max(s.abs())
        } else {
            0.0
        };

        let mut sign: i32 = (f > s + eps) as i32 - (f < s - eps) as i32;
        if cfg.long_only && sign < 0 {
            sign = 0;
        }

        if sign != pos {
            if pos == 0 || sign == 0 {
                if cfg.commission > 0.0 {
                    log_eq += log_comm;
                }
                trades = trades.saturating_add(1);
                pos = if sign == 0 { 0 } else { sign };
            } else if !cfg.allow_flip {
                if cfg.commission > 0.0 {
                    log_eq += log_comm;
                }
                trades = trades.saturating_add(1);
                pos = 0;
            } else {
                if cfg.commission > 0.0 {
                    log_eq += 2.0 * log_comm;
                }
                trades = trades.saturating_add(2);
                pos = sign;
            }
        }

        let lr_t = lr[t];
        log_eq += (pos as f64) * lr_t;
        if log_eq > log_peak {
            log_peak = log_eq;
        }
        let dd_log = log_peak - log_eq;
        if dd_log > max_log_dd {
            max_log_dd = dd_log;
        }

        if COMPUTE_SHARPE {
            let step_r = if pos == 0 { 0.0 } else { (pos as f64) * lr_t.exp_m1() };
            count += 1;
            let delta = step_r - mean;
            mean += delta / count as f64;
            let delta2 = step_r - mean;
            m2 += delta * delta2;
        }

        sum_abs_pos += (pos as i64).abs();
        sum_pos += pos as i64;
    }

    let sharpe = if COMPUTE_SHARPE {
        let variance = if count > 1 { m2 / (count as f64 - 1.0) } else { 0.0 };
        let std = variance.sqrt();
        if std > 0.0 { mean / std } else { 0.0 }
    } else {
        0.0
    };
    let denom = if COMPUTE_SHARPE { count } else { obs };
    let exposure = if denom > 0 { sum_abs_pos as f64 / denom as f64 } else { 0.0 };
    let net_exposure = if denom > 0 { sum_pos as f64 / denom as f64 } else { 0.0 };
    let max_dd = if max_log_dd > 0.0 {
        -(-max_log_dd).exp_m1()
    } else {
        0.0
    };

    Metrics {
        pnl: log_eq.exp() - 1.0,
        sharpe,
        max_dd,
        trades,
        exposure,
        net_exposure,
    }
}

#[derive(Clone, Debug)]
struct PeriodMatrixF64 {
    period_start: u16,
    period_end: u16,
    values: Vec<f64>,
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

        let m = &metrics[0];
        assert!(m.max_dd >= 0.0);
    }
}
