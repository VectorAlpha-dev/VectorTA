use serde::{Deserialize, Serialize};
use ta_optimizer::{grid, CpuEngine, ObjectiveKind, OptimizationMode, OptimizationResult};
use ta_strategies::double_ma::{expand_grid, DoubleMaBatchRange};

use crate::state::AppState;

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
    pub all: Vec<(ta_strategies::double_ma::DoubleMaParams, ta_strategies::double_ma::Metrics)>,
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
    match req.backend {
        Backend::GpuOnly { .. } => {
            return Err("GPU backend not implemented yet".to_string());
        }
        Backend::CpuOnly => {}
    }

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
    let result: OptimizationResult = tauri::async_runtime::spawn_blocking(move || {
        let engine = CpuEngine { candles: &candles };
        grid::grid_search(&engine, &combos, objective, num_candles)
    })
    .await
    .map_err(|e| e.to_string())?
    .ok_or_else(|| "grid search produced no result".to_string())?;
    let runtime_ms = start.elapsed().as_millis() as u64;

    Ok(BackendOptimizationResult {
        best_params: result.best_params,
        best_metrics: result.best_metrics,
        mode_used: OptimizationModeResolved::Grid,
        backend_used: BackendUsed::Cpu,
        num_combos: result.num_combos,
        num_candles: result.num_candles,
        runtime_ms,
        all: result.all,
    })
}
