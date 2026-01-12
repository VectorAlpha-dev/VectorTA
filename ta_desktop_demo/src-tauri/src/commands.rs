use serde::{Deserialize, Serialize};
use ta_optimizer::{grid, CpuEngine, ObjectiveKind, OptimizationMode, OptimizationResult};
use ta_strategies::double_ma::{expand_grid, DoubleMaBatchRange};

use crate::state::AppState;

#[cfg(feature = "cuda")]
use my_project::cuda::{
    cuda_available,
    moving_averages::{CudaMaData, CudaMaSelector},
};
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
    pub fast_ma_types: Vec<String>,
    pub slow_ma_types: Vec<String>,
    #[serde(default = "default_ma_source")]
    pub ma_source: String,
    pub objective: ObjectiveKind,
    pub mode: OptimizationMode,
    pub top_k: Option<usize>,
    pub include_all: Option<bool>,
}

fn default_ma_source() -> String {
    "close".to_string()
}

#[derive(Debug, Serialize)]
pub enum OptimizationModeResolved {
    Grid,

}

#[derive(Debug, Serialize)]
pub enum BackendUsed {
    Cpu,
    Gpu,
}

#[derive(Debug, Serialize)]
pub struct DoubleMaParamsResolved {
    pub fast_len: u16,
    pub slow_len: u16,
    pub fast_ma_type: String,
    pub slow_ma_type: String,
}

#[derive(Debug, Serialize)]
pub struct BackendOptimizationResult {
    pub best_params: DoubleMaParamsResolved,
    pub best_metrics: ta_strategies::double_ma::Metrics,
    pub mode_used: OptimizationModeResolved,
    pub backend_used: BackendUsed,
    pub num_combos: usize,
    pub num_candles: usize,
    pub runtime_ms: u64,
    pub top: Vec<(DoubleMaParamsResolved, ta_strategies::double_ma::Metrics)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub all: Option<Vec<(DoubleMaParamsResolved, ta_strategies::double_ma::Metrics)>>,
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

    let fast_ma_types = req.fast_ma_types.clone();
    let slow_ma_types = req.slow_ma_types.clone();
    let ma_source = req.ma_source.clone();

    let range = DoubleMaBatchRange {
        fast_len: (fast.0 as u16, fast.1 as u16, fast.2 as u16),
        slow_len: (slow.0 as u16, slow.1 as u16, slow.2 as u16),
        fast_ma_types: fast_ma_types.clone(),
        slow_ma_types: slow_ma_types.clone(),
    };

    let combos = expand_grid(&range);
    if combos.is_empty() {
        return Err("no valid parameter combinations (check ranges)".to_string());
    }

    let start = std::time::Instant::now();
    let objective = req.objective;
    let top_k = req.top_k.unwrap_or(0);
    let include_all = req.include_all.unwrap_or(false);
    let mode_used = match req.mode {
        OptimizationMode::Grid => OptimizationModeResolved::Grid,
        OptimizationMode::Auto => OptimizationModeResolved::Grid,
    };

    fn resolve_params(
        p: &ta_strategies::double_ma::DoubleMaParams,
        fast_ma_types: &[String],
        slow_ma_types: &[String],
    ) -> DoubleMaParamsResolved {
        let fast_ma_type = fast_ma_types
            .get(p.fast_ma_id as usize)
            .cloned()
            .unwrap_or_else(|| "<unknown>".to_string());
        let slow_ma_type = slow_ma_types
            .get(p.slow_ma_id as usize)
            .cloned()
            .unwrap_or_else(|| "<unknown>".to_string());

        DoubleMaParamsResolved {
            fast_len: p.fast_len,
            slow_len: p.slow_len,
            fast_ma_type,
            slow_ma_type,
        }
    }

    let backend = req.backend;
    let (result, backend_used): (OptimizationResult, BackendUsed) = match backend {
        Backend::CpuOnly => {
            let fast_ma_types = fast_ma_types.clone();
            let slow_ma_types = slow_ma_types.clone();
            let ma_source = ma_source.clone();
            let result: OptimizationResult = tauri::async_runtime::spawn_blocking(move || {
                let engine = CpuEngine {
                    candles: &candles,
                    fast_ma_types: &fast_ma_types,
                    slow_ma_types: &slow_ma_types,
                    ma_source: &ma_source,
                };
                grid::grid_search(&engine, &combos, objective, num_candles, top_k, include_all)
            })
            .await
            .map_err(|e| e.to_string())?
            .ok_or_else(|| "grid search produced no result".to_string())?;
            (result, BackendUsed::Cpu)
        }
        Backend::GpuOnly { device_id } => {
            let fast_ma_types = fast_ma_types.clone();
            let slow_ma_types = slow_ma_types.clone();
            let ma_source = ma_source.clone();
            let result: OptimizationResult = tauri::async_runtime::spawn_blocking(move || {
                grid_search_double_ma_gpu(
                    &candles,
                    &combos,
                    &fast_ma_types,
                    &slow_ma_types,
                    objective,
                    num_candles,
                    top_k,
                    include_all,
                    device_id,
                    &ma_source,
                )
            })
            .await
            .map_err(|e| e.to_string())??;
            (result, BackendUsed::Gpu)
        }
    };
    let runtime_ms = start.elapsed().as_millis() as u64;

    let best_params = resolve_params(&result.best_params, &fast_ma_types, &slow_ma_types);
    let top = result
        .top
        .into_iter()
        .map(|(p, m)| (resolve_params(&p, &fast_ma_types, &slow_ma_types), m))
        .collect();
    let all = result.all.map(|rows| {
        rows.into_iter()
            .map(|(p, m)| (resolve_params(&p, &fast_ma_types, &slow_ma_types), m))
            .collect()
    });

    Ok(BackendOptimizationResult {
        best_params,
        best_metrics: result.best_metrics,
        mode_used,
        backend_used,
        num_combos: result.num_combos,
        num_candles: result.num_candles,
        runtime_ms,
        top,
        all,
    })
}

#[cfg(not(feature = "cuda"))]
fn grid_search_double_ma_gpu(
    _candles: &my_project::utilities::data_loader::Candles,
    _combos: &[ta_strategies::double_ma::DoubleMaParams],
    _fast_ma_types: &[String],
    _slow_ma_types: &[String],
    _objective: ObjectiveKind,
    _num_candles: usize,
    _top_k: usize,
    _include_all: bool,
    _device_id: u32,
    _ma_source: &str,
) -> Result<OptimizationResult, String> {
    Err("GPU backend requires building `ta_desktop_demo_app` with `--features cuda`".to_string())
}

#[cfg(feature = "cuda")]
fn grid_search_double_ma_gpu(
    candles: &my_project::utilities::data_loader::Candles,
    combos: &[ta_strategies::double_ma::DoubleMaParams],
    fast_ma_types: &[String],
    slow_ma_types: &[String],
    objective: ObjectiveKind,
    num_candles: usize,
    top_k: usize,
    include_all: bool,
    device_id: u32,
    ma_source: &str,
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

    let ma_prices = my_project::utilities::data_loader::source_type(candles, ma_source);
    let prices_f32: Vec<f32> = ma_prices.iter().map(|&x| x as f32).collect();
    let nan_row: Vec<f32> = vec![f32::NAN; n];

    let matrices =
        build_ma_matrices_cuda(candles, &prices_f32, combos, fast_ma_types, slow_ma_types, device_id, ma_source)?;

    struct CachedEngine<'a> {
        prices: &'a [f64],
        n: usize,
        nan_row: &'a [f32],
        matrices: &'a HashMap<String, PeriodMatrix>,
        fast_ma_types: &'a [String],
        slow_ma_types: &'a [String],
    }

    impl<'a> ta_optimizer::BacktestEngine for CachedEngine<'a> {
        fn eval_batch(&self, combos: &[ta_strategies::double_ma::DoubleMaParams]) -> Vec<ta_strategies::double_ma::Metrics> {
            combos
                .par_iter()
                .map(|p| {
                    let fast_type = self
                        .fast_ma_types
                        .get(p.fast_ma_id as usize)
                        .map(|s| s.as_str());
                    let slow_type = self
                        .slow_ma_types
                        .get(p.slow_ma_id as usize)
                        .map(|s| s.as_str());

                    let fast = fast_type
                        .and_then(|t| self.matrices.get(t))
                        .and_then(|m| m.row(p.fast_len, self.n))
                        .unwrap_or(self.nan_row);
                    let slow = slow_type
                        .and_then(|t| self.matrices.get(t))
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
        fast_ma_types,
        slow_ma_types,
    };

    grid::grid_search(&engine, combos, objective, num_candles, top_k, include_all)
        .ok_or_else(|| "grid search produced no result".to_string())
}

#[cfg(feature = "cuda")]
struct PeriodMatrix {
    period_start: u16,
    period_end: u16,
    values: Vec<f32>,
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
    candles: &my_project::utilities::data_loader::Candles,
    prices_f32: &[f32],
    combos: &[ta_strategies::double_ma::DoubleMaParams],
    fast_ma_types: &[String],
    slow_ma_types: &[String],
    device_id: u32,
    ma_source: &str,

) -> Result<HashMap<String, PeriodMatrix>, String> {
    let series_len = prices_f32.len();
    let max_period_supported: u16 = if series_len >= u16::MAX as usize {
        u16::MAX
    } else {
        series_len as u16
    };

    let mut minmax: HashMap<&str, (u16, u16)> = HashMap::new();
    for p in combos {
        if p.fast_len > 0 {
            let t = fast_ma_types
                .get(p.fast_ma_id as usize)
                .ok_or_else(|| format!("fast_ma_id out of range: {}", p.fast_ma_id))?;
            update_minmax(&mut minmax, t.as_str(), p.fast_len);
        }
        if p.slow_len > 0 {
            let t = slow_ma_types
                .get(p.slow_ma_id as usize)
                .ok_or_else(|| format!("slow_ma_id out of range: {}", p.slow_ma_id))?;
            update_minmax(&mut minmax, t.as_str(), p.slow_len);
        }
    }

    let selector = CudaMaSelector::new(device_id as usize);

    let mut out: HashMap<String, PeriodMatrix> = HashMap::new();
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
        let expected_len = n_periods
            .checked_mul(series_len)
            .ok_or_else(|| "output length overflow".to_string())?;

        let requires_candles = ma_type.eq_ignore_ascii_case("vwap")
            || ma_type.eq_ignore_ascii_case("vwma")
            || ma_type.eq_ignore_ascii_case("vpwma");

        let data = if requires_candles {
            CudaMaData::Candles {
                candles,
                source: ma_source,
            }
        } else {
            CudaMaData::SliceF32(prices_f32)
        };

        let host = match selector.ma_sweep_to_host_f32(ma_type, data, start as usize, end as usize, 1) {
            Ok((values, rows, cols)) => {
                if rows != n_periods || cols != series_len {
                    return Err(format!(
                        "cuda ma sweep '{ma_type}' returned shape ({rows},{cols}) (expected ({n_periods},{series_len}))"
                    ));
                }
                if values.len() != expected_len {
                    return Err(format!(
                        "cuda ma sweep '{ma_type}' returned len {} (expected {expected_len})",
                        values.len()
                    ));
                }
                values
            }
            Err(sweep_err) => {

                let mut host = vec![f32::NAN; expected_len];
                for (row, period) in (start..=end).enumerate() {
                    let data = if requires_candles {
                        CudaMaData::Candles {
                            candles,
                            source: "close",
                        }
                    } else {
                        CudaMaData::SliceF32(prices_f32)
                    };
                    let series = selector.ma_to_host_f32(ma_type, data, period as usize).map_err(|e| {
                        format!(
                            "cuda ma sweep '{ma_type}' failed: {sweep_err}; fallback period {period} failed: {e}"
                        )
                    })?;
                    if series.len() != series_len {
                        return Err(format!(
                            "cuda ma '{ma_type}' period {period} returned len {} (expected {series_len})",
                            series.len()
                        ));
                    }
                    let offset = row
                        .checked_mul(series_len)
                        .ok_or_else(|| "output offset overflow".to_string())?;
                    let end = offset
                        .checked_add(series_len)
                        .ok_or_else(|| "output end overflow".to_string())?;
                    host[offset..end].copy_from_slice(&series);
                }
                host
            }
        };

        out.insert(
            ma_type.to_string(),
            PeriodMatrix {
                period_start: start,
                period_end: end,
                values: host,
            },
        );
    }

    Ok(out)
}

#[cfg(feature = "cuda")]
fn update_minmax<'a>(map: &mut HashMap<&'a str, (u16, u16)>, ma_type: &'a str, period: u16) {
    let entry = map.entry(ma_type).or_insert((period, period));
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
