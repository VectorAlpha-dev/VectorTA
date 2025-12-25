use serde::{Deserialize, Serialize};
use ta_optimizer::{grid, CpuEngine, ObjectiveKind, OptimizationMode, OptimizationResult};
use ta_strategies::double_ma::{expand_grid, DoubleMaBatchRange};

use crate::state::AppState;

#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, moving_averages::CudaAlma, moving_averages::CudaEma, moving_averages::CudaSma};
#[cfg(feature = "cuda")]
use my_project::indicators::moving_averages::alma::AlmaBatchRange;
#[cfg(feature = "cuda")]
use my_project::indicators::moving_averages::ema::EmaBatchRange;
#[cfg(feature = "cuda")]
use my_project::indicators::moving_averages::sma::SmaBatchRange;
#[cfg(feature = "cuda")]
use rayon::prelude::*;
#[cfg(feature = "cuda")]
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
#[serde(tag = "backend")]
pub enum Backend {
    CpuOnly,
    GpuOnly { device_id: u32 },
}

#[derive(Debug, Deserialize)]
pub struct DoubleMaRequest {
    pub backend: Backend,
    pub data_id: String,
    pub fast_range: (u32, u32, u32),
    pub slow_range: (u32, u32, u32),
    pub fast_ma_ids: Vec<u16>,
    pub slow_ma_ids: Vec<u16>,
    pub objective: ObjectiveKind,
    pub mode: OptimizationMode,
    pub top_k: Option<usize>,
    pub include_all: Option<bool>,
}

#[derive(Debug, Serialize)]
pub enum OptimizationModeResolved {
    Grid,
    // CmaEs can be added later
}

#[derive(Debug, Serialize)]
pub enum BackendUsed {
    Cpu,
    Gpu,
}

#[derive(Debug, Serialize)]
pub struct BackendOptimizationResult {
    pub best_params: ta_strategies::double_ma::DoubleMaParams,
    pub best_metrics: ta_strategies::double_ma::Metrics,
    pub mode_used: OptimizationModeResolved,
    pub backend_used: BackendUsed,
    pub num_combos: usize,
    pub num_candles: usize,
    pub runtime_ms: u64,
    pub top: Vec<(ta_strategies::double_ma::DoubleMaParams, ta_strategies::double_ma::Metrics)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub all: Option<Vec<(ta_strategies::double_ma::DoubleMaParams, ta_strategies::double_ma::Metrics)>>,
}

#[tauri::command]
pub fn load_price_data(path: String, state: tauri::State<AppState>) -> Result<String, String> {
    state.load_price_data(&path)
}

#[tauri::command]
pub async fn run_double_ma_optimization(
    req: DoubleMaRequest,
    state: tauri::State<'_, AppState>,
) -> Result<BackendOptimizationResult, String> {
    let candles = state.get_candles(&req.data_id)?;
    let num_candles = candles.close.len();

    let fast = req.fast_range;
    let slow = req.slow_range;

    let range = DoubleMaBatchRange {
        fast_len: (fast.0 as u16, fast.1 as u16, fast.2 as u16),
        slow_len: (slow.0 as u16, slow.1 as u16, slow.2 as u16),
        fast_ma_ids: req.fast_ma_ids.clone(),
        slow_ma_ids: req.slow_ma_ids.clone(),
    };

    let combos = expand_grid(&range);
    if combos.is_empty() {
        return Err("no valid parameter combinations (check ranges)".to_string());
    }

    let start = std::time::Instant::now();
    let objective = req.objective;
    let top_k = req.top_k.unwrap_or(0);
    let include_all = req.include_all.unwrap_or(false);

    let backend = req.backend;
    let (result, backend_used): (OptimizationResult, BackendUsed) = match backend {
        Backend::CpuOnly => {
            let result: OptimizationResult = tauri::async_runtime::spawn_blocking(move || {
                let engine = CpuEngine { candles: &candles };
                grid::grid_search(&engine, &combos, objective, num_candles, top_k, include_all)
            })
            .await
            .map_err(|e| e.to_string())?
            .ok_or_else(|| "grid search produced no result".to_string())?;
            (result, BackendUsed::Cpu)
        }
        Backend::GpuOnly { device_id } => {
            let result: OptimizationResult = tauri::async_runtime::spawn_blocking(move || {
                grid_search_double_ma_gpu(&candles, &combos, objective, num_candles, top_k, include_all, device_id)
            })
            .await
            .map_err(|e| e.to_string())??;
            (result, BackendUsed::Gpu)
        }
    };
    let runtime_ms = start.elapsed().as_millis() as u64;

    Ok(BackendOptimizationResult {
        best_params: result.best_params,
        best_metrics: result.best_metrics,
        mode_used: OptimizationModeResolved::Grid,
        backend_used,
        num_combos: result.num_combos,
        num_candles: result.num_candles,
        runtime_ms,
        top: result.top,
        all: result.all,
    })
}

#[cfg(not(feature = "cuda"))]
fn grid_search_double_ma_gpu(
    _candles: &my_project::utilities::data_loader::Candles,
    _combos: &[ta_strategies::double_ma::DoubleMaParams],
    _objective: ObjectiveKind,
    _num_candles: usize,
    _top_k: usize,
    _include_all: bool,
    _device_id: u32,
) -> Result<OptimizationResult, String> {
    Err("GPU backend requires building `ta_desktop_demo_app` with `--features cuda`".to_string())
}

#[cfg(feature = "cuda")]
fn grid_search_double_ma_gpu(
    candles: &my_project::utilities::data_loader::Candles,
    combos: &[ta_strategies::double_ma::DoubleMaParams],
    objective: ObjectiveKind,
    num_candles: usize,
    top_k: usize,
    include_all: bool,
    device_id: u32,
) -> Result<OptimizationResult, String> {
    if !cuda_available() {
        return Err("CUDA device not available (cuda_available() == false)".to_string());
    }
    if combos.is_empty() {
        return Err("no parameter combinations".to_string());
    }

    let prices = my_project::utilities::data_loader::source_type(candles, "close");
    let n = prices.len();
    if n < 2 {
        return Err("not enough candles (need at least 2)".to_string());
    }

    let prices_f32: Vec<f32> = prices.iter().map(|&x| x as f32).collect();
    let nan_row: Vec<f32> = vec![f32::NAN; n];

    let matrices = build_ma_matrices_cuda(&prices_f32, combos, device_id, n)?;

    struct CachedEngine<'a> {
        prices: &'a [f64],
        n: usize,
        nan_row: &'a [f32],
        matrices: &'a HashMap<u16, PeriodMatrix>,
    }

    impl<'a> ta_optimizer::BacktestEngine for CachedEngine<'a> {
        fn eval_batch(&self, combos: &[ta_strategies::double_ma::DoubleMaParams]) -> Vec<ta_strategies::double_ma::Metrics> {
            combos
                .par_iter()
                .map(|p| {
                    let fast = self
                        .matrices
                        .get(&p.fast_ma_id)
                        .and_then(|m| m.row(p.fast_len, self.n))
                        .unwrap_or(self.nan_row);
                    let slow = self
                        .matrices
                        .get(&p.slow_ma_id)
                        .and_then(|m| m.row(p.slow_len, self.n))
                        .unwrap_or(self.nan_row);
                    eval_double_ma_metrics(self.prices, fast, slow)
                })
                .collect()
        }
    }

    let engine = CachedEngine {
        prices,
        n,
        nan_row: &nan_row,
        matrices: &matrices,
    };

    grid::grid_search(&engine, combos, objective, num_candles, top_k, include_all)
        .ok_or_else(|| "grid search produced no result".to_string())
}

#[cfg(feature = "cuda")]
struct PeriodMatrix {
    period_start: u16,
    period_end: u16,
    values: Vec<f32>, // row-major: period-major rows, time-major cols
}

#[cfg(feature = "cuda")]
impl PeriodMatrix {
    fn row(&self, period: u16, cols: usize) -> Option<&[f32]> {
        if period < self.period_start || period > self.period_end {
            return None;
        }
        let row = (period - self.period_start) as usize;
        let offset = row.checked_mul(cols)?;
        let end = offset.checked_add(cols)?;
        self.values.get(offset..end)
    }
}

#[cfg(feature = "cuda")]
fn build_ma_matrices_cuda(
    prices_f32: &[f32],
    combos: &[ta_strategies::double_ma::DoubleMaParams],
    device_id: u32,
    series_len: usize,
) -> Result<HashMap<u16, PeriodMatrix>, String> {
    let mut minmax: HashMap<u16, (u16, u16)> = HashMap::new();

    for p in combos {
        if p.fast_len > 0 {
            update_minmax(&mut minmax, p.fast_ma_id, p.fast_len);
        }
        if p.slow_len > 0 {
            update_minmax(&mut minmax, p.slow_ma_id, p.slow_len);
        }
    }

    let max_period_supported: u16 = if series_len >= u16::MAX as usize {
        u16::MAX
    } else {
        series_len as u16
    };

    let mut out: HashMap<u16, PeriodMatrix> = HashMap::new();

    for (&ma_id, &(min_p, max_p)) in &minmax {
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
        let expected_len = n_periods
            .checked_mul(series_len)
            .ok_or_else(|| "output length overflow".to_string())?;

        match ma_id {
            0 => {
                let cuda = CudaSma::new(device_id as usize).map_err(|e| e.to_string())?;
                let sweep = SmaBatchRange {
                    period: (start as usize, end as usize, 1),
                };
                let mut host = vec![0.0_f32; expected_len];
                cuda.sma_batch_into_host_f32(prices_f32, &sweep, &mut host)
                    .map_err(|e| e.to_string())?;
                out.insert(
                    ma_id,
                    PeriodMatrix {
                        period_start: start,
                        period_end: end,
                        values: host,
                    },
                );
            }
            1 => {
                let cuda = CudaEma::new(device_id as usize).map_err(|e| e.to_string())?;
                let sweep = EmaBatchRange {
                    period: (start as usize, end as usize, 1),
                };
                let mut host = vec![0.0_f32; expected_len];
                cuda.ema_batch_into_host_f32(prices_f32, &sweep, &mut host)
                    .map_err(|e| e.to_string())?;
                out.insert(
                    ma_id,
                    PeriodMatrix {
                        period_start: start,
                        period_end: end,
                        values: host,
                    },
                );
            }
            2 => {
                let cuda = CudaAlma::new(device_id as usize).map_err(|e| e.to_string())?;
                let sweep = AlmaBatchRange {
                    period: (start as usize, end as usize, 1),
                    offset: (0.85, 0.85, 0.0),
                    sigma: (6.0, 6.0, 0.0),
                };
                let mut host = vec![0.0_f32; expected_len];
                cuda.alma_batch_into_host_f32(prices_f32, &sweep, &mut host)
                    .map(|_| ())
                    .map_err(|e| e.to_string())?;
                out.insert(
                    ma_id,
                    PeriodMatrix {
                        period_start: start,
                        period_end: end,
                        values: host,
                    },
                );
            }
            _ => {
                return Err(format!("unsupported ma_id for GPU backend: {ma_id}"));
            }
        }
    }

    Ok(out)
}

#[cfg(feature = "cuda")]
fn update_minmax(map: &mut HashMap<u16, (u16, u16)>, ma_id: u16, period: u16) {
    let entry = map.entry(ma_id).or_insert((period, period));
    if period < entry.0 {
        entry.0 = period;
    }
    if period > entry.1 {
        entry.1 = period;
    }
}

#[cfg(feature = "cuda")]
fn eval_double_ma_metrics(
    prices: &[f64],
    fast: &[f32],
    slow: &[f32],
) -> ta_strategies::double_ma::Metrics {
    let n = prices.len();
    if n < 2 || fast.len() != n || slow.len() != n {
        return ta_strategies::double_ma::Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
        };
    }

    let mut equity = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut max_dd = 0.0_f64;

    let mut prev_signal = 0.0_f64;

    let mut count = 0_u64;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;

    for i in 1..n {
        let f = fast[i] as f64;
        let s = slow[i] as f64;
        if !f.is_finite() || !s.is_finite() {
            // Treat warmup/invalid as zero return; do not change signal.
            count += 1;
            let delta = 0.0 - mean;
            mean += delta / count as f64;
            let delta2 = 0.0 - mean;
            m2 += delta * delta2;
            continue;
        }

        let signal = if f > s { 1.0 } else { 0.0 };
        let price_prev = prices[i - 1];
        let price_cur = prices[i];
        if !price_prev.is_finite() || !price_cur.is_finite() {
            // Zero return, but update signal to match CPU semantics.
            prev_signal = signal;
            count += 1;
            let delta = 0.0 - mean;
            mean += delta / count as f64;
            let delta2 = 0.0 - mean;
            m2 += delta * delta2;
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

        count += 1;
        let delta = strat_r - mean;
        mean += delta / count as f64;
        let delta2 = strat_r - mean;
        m2 += delta * delta2;

        prev_signal = signal;
    }

    let pnl = equity - 1.0;
    let sharpe = if count > 1 {
        let var = m2 / (count as f64 - 1.0);
        let std = var.sqrt();
        if std > 0.0 {
            mean / std
        } else {
            0.0
        }
    } else {
        0.0
    };

    ta_strategies::double_ma::Metrics { pnl, sharpe, max_dd }
}
