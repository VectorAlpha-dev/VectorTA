use serde::{Deserialize, Serialize};
use ta_optimizer::{ObjectiveKind, OptimizationHeatmap, OptimizationMode, OptimizationResult};
use ta_optimizer::stream::StreamAggregator;
use ta_strategies::double_ma::{DoubleMaCurves, DoubleMaParams, Metrics, StrategyConfig};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use rayon::prelude::*;

use crate::progress::ProgressSink;

#[cfg(feature = "cuda")]
use my_project::cuda::{
    cuda_available,
    moving_averages::{CudaAlma, CudaMaData, CudaMaSelector},
};

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "backend")]
pub enum Backend {
    Auto { device_id: u32 },
    CpuOnly,
    GpuOnly { device_id: u32 },
    GpuKernel { device_id: u32 },
}

#[derive(Clone, Debug, Deserialize)]
pub struct DoubleMaRequest {
    pub backend: Backend,
    pub data_id: String,
    pub fast_range: (u32, u32, u32),
    pub slow_range: (u32, u32, u32),
    pub fast_ma_types: Vec<String>,
    pub slow_ma_types: Vec<String>,
    #[serde(default = "default_ma_source")]
    pub ma_source: String,
    pub export_all_csv_path: Option<String>,
    pub fast_ma_params: Option<HashMap<String, f64>>,
    pub slow_ma_params: Option<HashMap<String, f64>>,
    #[serde(default)]
    pub strategy: StrategyConfig,
    pub objective: ObjectiveKind,
    pub mode: OptimizationMode,
    pub top_k: Option<usize>,
    pub include_all: Option<bool>,
    pub heatmap_bins: Option<u16>,
}

fn default_ma_source() -> String {
    "close".to_string()
}

#[derive(Debug, Serialize)]
pub enum OptimizationModeResolved {
    Grid,
    CoarseToFine,

}

#[derive(Debug, Serialize)]
pub enum BackendUsed {
    Cpu,
    Gpu,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DoubleMaParamsResolved {
    pub fast_len: u16,
    pub slow_len: u16,
    pub fast_ma_type: String,
    pub slow_ma_type: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DoubleMaDrilldownRequest {
    pub data_id: String,
    pub params: DoubleMaParamsResolved,
    #[serde(default = "default_ma_source")]
    pub ma_source: String,
    pub fast_ma_params: Option<HashMap<String, f64>>,
    pub slow_ma_params: Option<HashMap<String, f64>>,
    #[serde(default)]
    pub strategy: StrategyConfig,
    pub bins: Option<usize>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heatmap: Option<OptimizationHeatmap>,
}

pub fn compute_double_ma_drilldown_blocking_with_candles(
    req: DoubleMaDrilldownRequest,
    candles: &my_project::utilities::data_loader::Candles,
) -> Result<DoubleMaCurves, String> {
    let n = candles.close.len();
    if n < 2 {
        return Err("not enough candles (need at least 2)".to_string());
    }

    let fast_ma = req.params.fast_ma_type.trim();
    let slow_ma = req.params.slow_ma_type.trim();
    validate_ma_kind(fast_ma)?;
    validate_ma_kind(slow_ma)?;

    let fast_len = req.params.fast_len as usize;
    let slow_len = req.params.slow_len as usize;
    if fast_len == 0 || slow_len == 0 {
        return Err("periods must be > 0".to_string());
    }
    if fast_len > n || slow_len > n {
        return Err(format!(
            "period exceeds series length: fast_len={fast_len} slow_len={slow_len} n={n}"
        ));
    }

    validate_strategy_cfg(&req.strategy)?;

    let close = my_project::utilities::data_loader::source_type(candles, "close");
    let ma_prices = my_project::utilities::data_loader::source_type(candles, &req.ma_source);
    if close.len() != ma_prices.len() {
        return Err("ma source length mismatch".to_string());
    }
    let fv_close = first_finite_idx(close).ok_or_else(|| "close series all non-finite".to_string())?;
    let fv_ma = first_finite_idx(ma_prices).ok_or_else(|| "ma source series all non-finite".to_string())?;
    let first_valid = fv_close.max(fv_ma);

    let trade_offset = if req.strategy.trade_on_next_bar { 1usize } else { 0usize };
    let max_p = fast_len.max(slow_len).max(1);
    let t_valid = first_valid.saturating_add(max_p.saturating_sub(1));
    let start_t = t_valid.saturating_add(trade_offset);
    if start_t >= n {
        return Err("not enough candles after warmup for analysis".to_string());
    }

    let (fast_alma_offset, fast_alma_sigma) = resolve_alma_params_for_side(
        "fast",
        fast_ma,
        req.fast_ma_params.as_ref(),
    )?;
    let (slow_alma_offset, slow_alma_sigma) = resolve_alma_params_for_side(
        "slow",
        slow_ma,
        req.slow_ma_params.as_ref(),
    )?;

    let fast_periods = [req.params.fast_len];
    let (fast_mat, cols) = ma_matrix_cpu_f64(
        candles,
        fast_ma,
        &req.ma_source,
        &fast_periods,
        req.fast_ma_params.as_ref(),
        fast_alma_offset,
        fast_alma_sigma,
    )?;
    if cols != n || fast_mat.len() != n {
        return Err("fast MA series length mismatch".to_string());
    }

    let slow_periods = [req.params.slow_len];
    let (slow_mat, cols2) = ma_matrix_cpu_f64(
        candles,
        slow_ma,
        &req.ma_source,
        &slow_periods,
        req.slow_ma_params.as_ref(),
        slow_alma_offset,
        slow_alma_sigma,
    )?;
    if cols2 != n || slow_mat.len() != n {
        return Err("slow MA series length mismatch".to_string());
    }

    let lr = compute_log_returns_f64(close);
    let bins = req.bins.unwrap_or(1024);
    Ok(ta_strategies::double_ma::eval_double_ma_curves_from_log_returns_and_mas_f64(
        &lr,
        &fast_mat,
        &slow_mat,
        start_t,
        &req.strategy,
        bins,
    ))
}

pub fn run_double_ma_optimization_blocking_with_candles(
    req: DoubleMaRequest,
    candles: &my_project::utilities::data_loader::Candles,
    cancel: &AtomicBool,
    progress: Option<&dyn ProgressSink>,
) -> Result<BackendOptimizationResult, String> {
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".to_string());
    }

    let fast = req.fast_range;
    let slow = req.slow_range;

    let fast_ma_types = req.fast_ma_types;
    let slow_ma_types = req.slow_ma_types;
    let ma_source = req.ma_source;
    let export_all_csv_path = req.export_all_csv_path.and_then(|s| {
        let t = s.trim().to_string();
        if t.is_empty() { None } else { Some(t) }
    });
    let fast_ma_params = req.fast_ma_params;
    let slow_ma_params = req.slow_ma_params;
    let strategy = req.strategy;
    let heatmap_bins = req.heatmap_bins.unwrap_or(0) as usize;
    validate_strategy_cfg(&strategy)?;
    if heatmap_bins > 512 {
        return Err("heatmap_bins too large (max 512)".to_string());
    }

    if fast_ma_types.len() != 1 || slow_ma_types.len() != 1 {
        return Err("Select exactly one fast MA and one slow MA.".to_string());
    }
    let fast_ma = fast_ma_types[0].trim().to_ascii_lowercase();
    let slow_ma = slow_ma_types[0].trim().to_ascii_lowercase();
    if fast_ma.is_empty() || slow_ma.is_empty() {
        return Err("Fast/slow MA type cannot be empty.".to_string());
    }

    validate_ma_kind(&fast_ma)?;
    validate_ma_kind(&slow_ma)?;

    let (fast_alma_offset, fast_alma_sigma) = resolve_alma_params_for_side(
        "fast",
        &fast_ma,
        fast_ma_params.as_ref(),
    )?;
    let (slow_alma_offset, slow_alma_sigma) = resolve_alma_params_for_side(
        "slow",
        &slow_ma,
        slow_ma_params.as_ref(),
    )?;

    let fast_periods = expand_u16_range(fast)?;
    let slow_periods = expand_u16_range(slow)?;
    if fast_periods.is_empty() || slow_periods.is_empty() {
        return Err("empty period range".to_string());
    }
    let total_pairs = estimate_pairs_fast_lt_slow(&fast_periods, &slow_periods);
    if total_pairs == 0 {
        return Err("no valid parameter combinations (fast must be < slow)".to_string());
    }

    let close = my_project::utilities::data_loader::source_type(candles, "close");
    let n = close.len();

    let start = std::time::Instant::now();
    let objective = req.objective;
    let top_k = req.top_k.unwrap_or(0);
    let include_all = req.include_all.unwrap_or(false);
    if export_all_csv_path.is_some() && include_all {
        return Err("export_all_csv_path requires include_all=false (stream export avoids huge RAM usage)".to_string());
    }
    if export_all_csv_path.is_some() && req.mode != OptimizationMode::Grid {
        return Err("export_all_csv_path requires mode=Grid".to_string());
    }
    if include_all && total_pairs > 2_000_000 {
        return Err(format!(
            "include_all=true would return {} rows; disable it or reduce ranges.",
            total_pairs
        ));
    }
    let mode_used = match req.mode {
        OptimizationMode::Grid => OptimizationModeResolved::Grid,
        OptimizationMode::CoarseToFine => OptimizationModeResolved::CoarseToFine,
        OptimizationMode::Auto => {
            let can_adapt = !include_all && export_all_csv_path.is_none() && heatmap_bins == 0;
            if can_adapt && total_pairs > 500_000 {
                OptimizationModeResolved::CoarseToFine
            } else {
                OptimizationModeResolved::Grid
            }
        }
    };
    if matches!(mode_used, OptimizationModeResolved::CoarseToFine) {
        if include_all {
            return Err("CoarseToFine mode requires include_all=false".to_string());
        }
        if heatmap_bins > 0 {
            return Err("CoarseToFine mode currently requires heatmap_bins=0".to_string());
        }
        if export_all_csv_path.is_some() {
            return Err("CoarseToFine mode cannot be used with export_all_csv_path".to_string());
        }
    }

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

    let mut backend = match req.backend {
        Backend::Auto { device_id } => {
            let export_requested = export_all_csv_path.is_some();

            let fast_info = lookup_ma_info(&fast_ma).ok_or_else(|| "unknown fast MA".to_string())?;
            let slow_info = lookup_ma_info(&slow_ma).ok_or_else(|| "unknown slow MA".to_string())?;
            let cpu_ok = fast_info.supports_cpu_single && slow_info.supports_cpu_single;

            #[cfg(feature = "cuda")]
            let cuda_ok = cuda_available()
                && fast_info.supports_cuda_single
                && slow_info.supports_cuda_single;
            #[cfg(not(feature = "cuda"))]
            let cuda_ok = false;

            #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
            let kernel_ok = cuda_available()
                && crate::vram_ma::supports_vram_kernel_ma(fast_ma.as_str())
                && crate::vram_ma::supports_vram_kernel_ma(slow_ma.as_str());
            #[cfg(not(all(feature = "cuda", feature = "cuda-backtest-kernel")))]
            let kernel_ok = false;

            if export_requested && !kernel_ok {
                return Err(
                    "export_all_csv_path requires the GPU (kernel) backend, which is not available for this MA selection/build.".to_string(),
                );
            }

            if kernel_ok {
                Backend::GpuKernel { device_id }
            } else if cuda_ok {
                Backend::GpuOnly { device_id }
            } else if cpu_ok {
                Backend::CpuOnly
            } else {
                return Err(format!(
                    "No supported backend for fast='{fast_ma}' slow='{slow_ma}' (try different MAs or enable CUDA)."
                ));
            }
        }
        other => other,
    };

    // If the user explicitly asks for the kernel backend but it's not available for this MA/build,
    // fall back to the best available backend (unless export is requested).
    let kernel_device_id = match &backend {
        Backend::GpuKernel { device_id } => Some(*device_id),
        _ => None,
    };
    if let Some(device_id) = kernel_device_id {
        let export_requested = export_all_csv_path.is_some();

        #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
        let kernel_ok = cuda_available()
            && crate::vram_ma::supports_vram_kernel_ma(fast_ma.as_str())
            && crate::vram_ma::supports_vram_kernel_ma(slow_ma.as_str());
        #[cfg(not(all(feature = "cuda", feature = "cuda-backtest-kernel")))]
        let kernel_ok = false;

        if export_requested && !kernel_ok {
            return Err(
                "export_all_csv_path requires the GPU (kernel) backend, which is not available for this MA selection/build.".to_string(),
            );
        }

        if !kernel_ok {
            let fast_info = lookup_ma_info(&fast_ma).ok_or_else(|| "unknown fast MA".to_string())?;
            let slow_info = lookup_ma_info(&slow_ma).ok_or_else(|| "unknown slow MA".to_string())?;
            let cpu_ok = fast_info.supports_cpu_single && slow_info.supports_cpu_single;

            #[cfg(feature = "cuda")]
            let cuda_ok = cuda_available()
                && fast_info.supports_cuda_single
                && slow_info.supports_cuda_single;
            #[cfg(not(feature = "cuda"))]
            let cuda_ok = false;

            backend = if cuda_ok {
                Backend::GpuOnly { device_id }
            } else if cpu_ok {
                Backend::CpuOnly
            } else {
                return Err(format!(
                    "No supported backend for fast='{fast_ma}' slow='{slow_ma}' (try different MAs or enable CUDA)."
                ));
            };
        }
    }

    let run_grid = |fast_periods: &[u16],
                        slow_periods: &[u16],
                        top_k: usize,
                        include_all: bool,
                        heatmap_bins: usize,
                        export_all_csv_path: Option<&str>,
                        total_pairs: usize|
     -> Result<(OptimizationResult, BackendUsed), String> {
        match &backend {
            Backend::CpuOnly => {
                if export_all_csv_path.is_some() {
                    return Err("export_all_csv_path is currently supported only for the GPU (kernel) backend".to_string());
                }
                let result = optimize_double_ma_cpu_tiled(
                    candles,
                    fast_periods,
                    slow_periods,
                    &fast_ma,
                    &slow_ma,
                    &ma_source,
                    fast_ma_params.as_ref(),
                    slow_ma_params.as_ref(),
                    &strategy,
                    objective,
                    top_k,
                    include_all,
                    fast_alma_offset,
                    fast_alma_sigma,
                    slow_alma_offset,
                    slow_alma_sigma,
                    total_pairs,
                    heatmap_bins,
                    cancel,
                    progress,
                )?;
                Ok((result, BackendUsed::Cpu))
            }
            Backend::GpuOnly { device_id } => {
                if export_all_csv_path.is_some() {
                    return Err("export_all_csv_path is currently supported only for the GPU (kernel) backend".to_string());
                }
                let result = optimize_double_ma_gpu_sweep_tiled(
                    candles,
                    *device_id,
                    fast_periods,
                    slow_periods,
                    &fast_ma,
                    &slow_ma,
                    &ma_source,
                    fast_ma_params.as_ref(),
                    slow_ma_params.as_ref(),
                    &strategy,
                    objective,
                    top_k,
                    include_all,
                    fast_alma_offset,
                    fast_alma_sigma,
                    slow_alma_offset,
                    slow_alma_sigma,
                    total_pairs,
                    heatmap_bins,
                    cancel,
                    progress,
                )?;
                Ok((result, BackendUsed::Gpu))
            }
            Backend::GpuKernel { device_id } => {
                let result = optimize_double_ma_gpu_kernel_tiled(
                    candles,
                    *device_id,
                    fast_periods,
                    slow_periods,
                    &fast_ma,
                    &slow_ma,
                    &ma_source,
                    fast_ma_params.as_ref(),
                    slow_ma_params.as_ref(),
                    &strategy,
                    objective,
                    top_k,
                    include_all,
                    fast_alma_offset,
                    fast_alma_sigma,
                    slow_alma_offset,
                    slow_alma_sigma,
                    export_all_csv_path,
                    total_pairs,
                    heatmap_bins,
                    cancel,
                    progress,
                )?;
                Ok((result, BackendUsed::Gpu))
            }
            Backend::Auto { .. } => unreachable!("Backend::Auto should be resolved before dispatch"),
        }
    };

    let (result, backend_used): (OptimizationResult, BackendUsed) = match mode_used {
        OptimizationModeResolved::Grid => run_grid(
            &fast_periods,
            &slow_periods,
            top_k,
            include_all,
            heatmap_bins,
            export_all_csv_path.as_deref(),
            total_pairs,
        )?,
        OptimizationModeResolved::CoarseToFine => {
            let target_pairs = 200_000usize;
            let mut factor = 1u32;
            if total_pairs > target_pairs {
                let ratio = (total_pairs as f64) / (target_pairs as f64);
                let est = ratio.sqrt().ceil() as u32;
                factor = est.max(2).min(64);
            }

            if factor <= 1 {
                run_grid(
                    &fast_periods,
                    &slow_periods,
                    top_k,
                    include_all,
                    heatmap_bins,
                    export_all_csv_path.as_deref(),
                    total_pairs,
                )?
            } else {
                let coarse_fast = (
                    fast.0,
                    fast.1,
                    if fast.2 == 0 { 0 } else { fast.2.saturating_mul(factor) },
                );
                let coarse_slow = (
                    slow.0,
                    slow.1,
                    if slow.2 == 0 { 0 } else { slow.2.saturating_mul(factor) },
                );
                let coarse_fast_periods = expand_u16_range(coarse_fast)?;
                let coarse_slow_periods = expand_u16_range(coarse_slow)?;
                let coarse_pairs = estimate_pairs_fast_lt_slow(&coarse_fast_periods, &coarse_slow_periods);
                if coarse_pairs == 0 {
                    return Err("coarse search produced no valid parameter combinations".to_string());
                }

                let (coarse_res, coarse_backend) = run_grid(
                    &coarse_fast_periods,
                    &coarse_slow_periods,
                    0,
                    false,
                    0,
                    None,
                    coarse_pairs,
                )?;

                let best_fast = coarse_res.best_params.fast_len as u32;
                let best_slow = coarse_res.best_params.slow_len as u32;

                let (fast_min, fast_max) = (fast.0.min(fast.1), fast.0.max(fast.1));
                let (slow_min, slow_max) = (slow.0.min(slow.1), slow.0.max(slow.1));

                let f_step = fast.2;
                let s_step = slow.2;
                let f_win = if f_step == 0 { 0 } else { f_step.saturating_mul(factor).saturating_mul(2).max(f_step) };
                let s_win = if s_step == 0 { 0 } else { s_step.saturating_mul(factor).saturating_mul(2).max(s_step) };

                let fine_fast = (
                    best_fast.saturating_sub(f_win).max(fast_min),
                    best_fast.saturating_add(f_win).min(fast_max),
                    f_step,
                );
                let fine_slow = (
                    best_slow.saturating_sub(s_win).max(slow_min),
                    best_slow.saturating_add(s_win).min(slow_max),
                    s_step,
                );

                let fine_fast_periods = expand_u16_range(fine_fast)?;
                let fine_slow_periods = expand_u16_range(fine_slow)?;
                let fine_pairs = estimate_pairs_fast_lt_slow(&fine_fast_periods, &fine_slow_periods);
                if fine_pairs == 0 {
                    return Err("fine search produced no valid parameter combinations".to_string());
                }

                let (mut fine_res, fine_backend) = run_grid(
                    &fine_fast_periods,
                    &fine_slow_periods,
                    top_k,
                    include_all,
                    heatmap_bins,
                    None,
                    fine_pairs,
                )?;
                fine_res.num_combos = coarse_res
                    .num_combos
                    .saturating_add(fine_res.num_combos);

                let backend_used = if matches!(coarse_backend, BackendUsed::Cpu) && matches!(fine_backend, BackendUsed::Cpu) {
                    BackendUsed::Cpu
                } else {
                    BackendUsed::Gpu
                };

                (fine_res, backend_used)
            }
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
    let heatmap = result.heatmap;

    Ok(BackendOptimizationResult {
        best_params,
        best_metrics: result.best_metrics,
        mode_used,
        backend_used,
        num_combos: result.num_combos,
        num_candles: n,
        runtime_ms,
        top,
        all,
        heatmap,
    })
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
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

#[cfg(not(all(feature = "cuda", feature = "cuda-backtest-kernel")))]
#[allow(dead_code)]
fn grid_search_double_ma_gpu_kernel(
    _candles: &my_project::utilities::data_loader::Candles,
    _combos: &[ta_strategies::double_ma::DoubleMaParams],
    _fast_range: (u32, u32, u32),
    _slow_range: (u32, u32, u32),
    _fast_ma_types: &[String],
    _slow_ma_types: &[String],
    _objective: ObjectiveKind,
    _num_candles: usize,
    _top_k: usize,
    _include_all: bool,
    _device_id: u32,
    _ma_source: &str,
    _alma_offset: Option<f64>,
    _alma_sigma: Option<f64>,
) -> Result<OptimizationResult, String> {
    Err("GPU kernel backend requires building `ta_desktop_demo_app` with `--features cuda,cuda-backtest-kernel`".to_string())
}

#[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
#[allow(dead_code)]
fn grid_search_double_ma_gpu_kernel(
    candles: &my_project::utilities::data_loader::Candles,
    combos: &[ta_strategies::double_ma::DoubleMaParams],
    fast_range: (u32, u32, u32),
    slow_range: (u32, u32, u32),
    fast_ma_types: &[String],
    slow_ma_types: &[String],
    objective: ObjectiveKind,
    num_candles: usize,
    top_k: usize,
    include_all: bool,
    device_id: u32,
    ma_source: &str,
    alma_offset: Option<f64>,
    alma_sigma: Option<f64>,
) -> Result<OptimizationResult, String> {
    if !cuda_available() {
        return Err("CUDA device not available (cuda_available() == false)".to_string());
    }

    if fast_ma_types.len() != 1 || slow_ma_types.len() != 1 {
        return Err(
            "GPU kernel backend currently supports one fast MA type and one slow MA type (set fast_ma_types and slow_ma_types to a single entry).".to_string(),
        );
    }
    let fast_ma = fast_ma_types[0].trim();
    let slow_ma = slow_ma_types[0].trim();

    let offset = alma_offset.unwrap_or(0.85);
    let sigma = alma_sigma.unwrap_or(6.0);
    let cfg = crate::gpu_backtest_kernel::KernelConfig {
        device_id,
        fast_alma_offset: offset,
        fast_alma_sigma: sigma,
        slow_alma_offset: offset,
        slow_alma_sigma: sigma,
    };
    let metrics = crate::gpu_backtest_kernel::eval_double_ma_batch_gpu_kernel(
        candles,
        combos,
        fast_range,
        slow_range,
        fast_ma,
        slow_ma,
        ma_source,
        None,
        None,
        cfg,
        &StrategyConfig::default(),
    )?;
    if metrics.len() != combos.len() {
        return Err(format!(
            "kernel backend returned {} metrics (expected {})",
            metrics.len(),
            combos.len()
        ));
    }

    fn score(metrics: &ta_strategies::double_ma::Metrics, objective: ObjectiveKind) -> f64 {
        match objective {
            ObjectiveKind::Pnl => metrics.pnl,
            ObjectiveKind::Sharpe => metrics.sharpe,
            ObjectiveKind::MaxDrawdown => -metrics.max_dd,
        }
    }

    if combos.is_empty() {
        return Err("no parameter combinations".to_string());
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
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
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

    Ok(OptimizationResult {
        best_params,
        best_metrics,
        top,
        all,
        heatmap: None,
        num_combos: combos.len(),
        num_candles,
    })
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
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

    ta_optimizer::grid::grid_search(&engine, combos, objective, num_candles, top_k, include_all)
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

        let requires_candles = lookup_ma_info(ma_type)
            .map(|m| m.requires_candles)
            .unwrap_or(false);

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
                            source: ma_source,
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
            trades: 0,
            exposure: 0.0,
            net_exposure: 0.0,
        };
    }

    let cfg = StrategyConfig::default();
    let lr = compute_log_returns_f64(prices);
    let first_valid = first_finite_idx(prices).unwrap_or(0);
    let t_sig0 = fast
        .iter()
        .zip(slow.iter())
        .position(|(f, s)| f.is_finite() && s.is_finite())
        .unwrap_or(n);
    let trade_offset = if cfg.trade_on_next_bar { 1usize } else { 0usize };
    let start_t = t_sig0
        .saturating_add(trade_offset)
        .max(first_valid);

    ta_strategies::double_ma::eval_double_ma_from_log_returns_and_mas_f32(
        &lr,
        fast,
        slow,
        start_t,
        &cfg,
    )
}

fn emit_progress(
    progress: Option<&dyn ProgressSink>,
    processed_pairs: usize,
    total_pairs: usize,
    phase: &'static str,
) {
    crate::progress::emit_double_ma_progress(progress, processed_pairs, total_pairs, phase);
}

fn lookup_ma_info(
    ma_type: &str,
) -> Option<&'static my_project::indicators::moving_averages::registry::MovingAverageInfo> {
    my_project::indicators::moving_averages::registry::list_moving_averages()
        .iter()
        .find(|m| m.id.eq_ignore_ascii_case(ma_type))
}

fn validate_ma_kind(ma_type: &str) -> Result<(), String> {
    let ma = ma_type.trim();
    let info = lookup_ma_info(ma).ok_or_else(|| format!("Unknown moving average type: '{ma}'"))?;
    if !info.period_based {
        return Err(format!(
            "MA '{ma}' is not period-based and cannot be used in the double-MA optimizer."
        ));
    }
    if !info.single_output {
        return Err(format!(
            "MA '{ma}' has multiple outputs and cannot be used in the double-MA optimizer."
        ));
    }
    Ok(())
}

fn validate_strategy_cfg(cfg: &StrategyConfig) -> Result<(), String> {
    if !cfg.commission.is_finite() || cfg.commission < 0.0 || cfg.commission >= 1.0 {
        return Err("commission must be finite and in [0, 1)".to_string());
    }
    if !cfg.eps_rel.is_finite() || cfg.eps_rel < 0.0 {
        return Err("eps_rel must be finite and >= 0".to_string());
    }
    Ok(())
}

fn resolve_alma_params_for_side(
    side: &'static str,
    ma_type: &str,
    ma_params: Option<&HashMap<String, f64>>,
) -> Result<(f64, f64), String> {
    let default_offset = 0.85;
    let default_sigma = 6.0;
    if !ma_type.trim().eq_ignore_ascii_case("alma") {
        return Ok((default_offset, default_sigma));
    }

    let mut offset = default_offset;
    let mut sigma = default_sigma;
    if let Some(params) = ma_params {
        if let Some(v) = params.get("offset") {
            offset = *v;
        }
        if let Some(v) = params.get("sigma") {
            sigma = *v;
        }
    }

    if !offset.is_finite() || offset < 0.0 || offset > 1.0 {
        return Err(format!(
            "ALMA offset for {side} MA must be finite and in [0, 1]"
        ));
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(format!("ALMA sigma for {side} MA must be finite and > 0"));
    }

    Ok((offset, sigma))
}

fn expand_u16_range((start, end, step): (u32, u32, u32)) -> Result<Vec<u16>, String> {
    let push_checked = |out: &mut Vec<u16>, v: u32| -> Result<(), String> {
        if v > u16::MAX as u32 {
            return Err(format!("period {v} exceeds u16::MAX"));
        }
        out.push(v as u16);
        Ok(())
    };

    if step == 0 || start == end {
        let mut out = Vec::with_capacity(1);
        push_checked(&mut out, start)?;
        return Ok(out);
    }

    let step = step.max(1);
    let (lo, hi) = if start <= end { (start, end) } else { (end, start) };

    let mut out = Vec::new();
    let mut v = lo;
    loop {
        push_checked(&mut out, v)?;
        if v == hi {
            break;
        }
        match v.checked_add(step) {
            Some(next) if next > v && next <= hi => v = next,
            _ => break,
        }
    }
    if out.is_empty() {
        return Err("empty period range".to_string());
    }
    Ok(out)
}

fn estimate_pairs_fast_lt_slow(fast_periods: &[u16], slow_periods: &[u16]) -> usize {
    if fast_periods.is_empty() || slow_periods.is_empty() {
        return 0;
    }
    let mut slow_sorted: Vec<u16> = slow_periods.to_vec();
    slow_sorted.sort_unstable();

    let s_len = slow_sorted.len();
    let mut total: usize = 0;
    for &f in fast_periods {
        let idx = slow_sorted.partition_point(|&s| s <= f);
        total = total.saturating_add(s_len.saturating_sub(idx));
    }
    total
}

fn tile_sizes_from_budget(
    total_fast: usize,
    total_slow: usize,
    series_len: usize,
    bytes_per_value: usize,
    env_mb: &str,
    default_mb: usize,
) -> (usize, usize) {
    let budget_mb = std::env::var(env_mb)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default_mb)
        .max(64);

    let mut budget = budget_mb.saturating_mul(1024).saturating_mul(1024);
    let headroom = 256usize * 1024 * 1024;
    if budget > headroom {
        budget = budget.saturating_sub(headroom);
    } else {
        budget = budget / 2;
    }

    let bytes_per_row = series_len.saturating_mul(bytes_per_value).max(1);
    let max_rows_total = (budget / bytes_per_row).max(2);
    let mut pf = (max_rows_total / 2).max(1);
    let mut ps = (max_rows_total / 2).max(1);

    pf = pf.clamp(1, total_fast.max(1));
    ps = ps.clamp(1, total_slow.max(1));
    (pf, ps)
}

fn cpu_tile_sizes(total_fast: usize, total_slow: usize, series_len: usize) -> (usize, usize) {
    tile_sizes_from_budget(
        total_fast,
        total_slow,
        series_len,
        std::mem::size_of::<f64>(),
        "VECTORBT_CPU_RAM_BUDGET_MB",
        1024,
    )
}

#[cfg(feature = "cuda")]
fn sweep_tile_sizes(total_fast: usize, total_slow: usize, series_len: usize) -> (usize, usize) {
    tile_sizes_from_budget(
        total_fast,
        total_slow,
        series_len,
        std::mem::size_of::<f32>(),
        "VECTORBT_SWEEP_RAM_BUDGET_MB",
        2048,
    )
}

fn eval_double_ma_metrics_f64(prices: &[f64], fast: &[f64], slow: &[f64]) -> Metrics {
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

    let cfg = StrategyConfig::default();
    let lr = compute_log_returns_f64(prices);
    let first_valid = first_finite_idx(prices).unwrap_or(0);
    let t_sig0 = fast
        .iter()
        .zip(slow.iter())
        .position(|(f, s)| f.is_finite() && s.is_finite())
        .unwrap_or(n);
    let trade_offset = if cfg.trade_on_next_bar { 1usize } else { 0usize };
    let start_t = t_sig0
        .saturating_add(trade_offset)
        .max(first_valid);

    ta_strategies::double_ma::eval_double_ma_from_log_returns_and_mas_f64(
        &lr,
        fast,
        slow,
        start_t,
        &cfg,
    )
}

fn ma_period_step(periods: &[u16]) -> usize {
    if periods.len() < 2 {
        return 0;
    }
    let a = periods[0] as i32;
    let b = periods[1] as i32;
    (b - a).abs().max(1) as usize
}

#[inline]
fn first_finite_idx(series: &[f64]) -> Option<usize> {
    series.iter().position(|v| v.is_finite())
}

fn compute_log_returns_f64(prices: &[f64]) -> Vec<f64> {
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

fn ma_matrix_cpu_f64(
    candles: &my_project::utilities::data_loader::Candles,
    ma_type: &str,
    ma_source: &str,
    periods: &[u16],
    ma_params: Option<&HashMap<String, f64>>,
    alma_offset: f64,
    alma_sigma: f64,
) -> Result<(Vec<f64>, usize), String> {
    let n = candles.close.len();
    if periods.is_empty() {
        return Err("empty period list".to_string());
    }
    let rows = periods.len();
    let expected_len = rows
        .checked_mul(n)
        .ok_or_else(|| "rows*cols overflow".to_string())?;

    let start = periods[0] as usize;
    let end = periods[rows - 1] as usize;
    let step = ma_period_step(periods);

    let mut values: Vec<f64> = vec![f64::NAN; expected_len];

    if ma_type.eq_ignore_ascii_case("alma") {
        let prices = my_project::utilities::data_loader::source_type(candles, ma_source);
        let sweep = my_project::indicators::moving_averages::alma::AlmaBatchRange {
            period: (start, end, step),
            offset: (alma_offset, alma_offset, 0.0),
            sigma: (alma_sigma, alma_sigma, 0.0),
        };
        let out = my_project::indicators::moving_averages::alma::alma_batch_with_kernel(
            prices,
            &sweep,
            my_project::utilities::enums::Kernel::Auto,
        )
        .map_err(|e| e.to_string())?;
        if out.cols != n {
            return Err(format!("alma batch cols {} != {n}", out.cols));
        }
        for (src_row, params) in out.combos.iter().enumerate() {
            let per = params.period.unwrap_or(start) as usize;
            let row = if step == 0 {
                0usize
            } else if per >= start && (per - start) % step == 0 {
                (per - start) / step
            } else {
                continue;
            };
            if row >= rows {
                continue;
            }
            let src_off = src_row.saturating_mul(out.cols);
            let dst_off = row.saturating_mul(n);
            if src_off + n > out.values.len() || dst_off + n > values.len() {
                continue;
            }
            values[dst_off..dst_off + n].copy_from_slice(&out.values[src_off..src_off + n]);
        }
        return Ok((values, n));
    }

    let data = my_project::indicators::moving_averages::ma::MaData::Candles {
        candles,
        source: ma_source,
    };
    let batch = my_project::indicators::moving_averages::ma_batch::ma_batch_with_kernel_and_params(
        ma_type,
        data,
        (start, end, step.max(1)),
        my_project::utilities::enums::Kernel::Auto,
        ma_params,
    );
    match batch {
        Ok(b) if b.cols == n => {
            for (src_row, &period) in b.periods.iter().enumerate() {
                let per = period as usize;
                let row = if step == 0 {
                    if per == start { 0usize } else { continue }
                } else if per >= start && (per - start) % step == 0 {
                    (per - start) / step
                } else {
                    continue;
                };
                if row >= rows {
                    continue;
                }
                let src_off = src_row.saturating_mul(b.cols);
                let dst_off = row.saturating_mul(n);
                if src_off + n > b.values.len() || dst_off + n > values.len() {
                    continue;
                }
                values[dst_off..dst_off + n].copy_from_slice(&b.values[src_off..src_off + n]);
            }
        }
        _ => {
            for (row, &period) in periods.iter().enumerate() {
                let series = my_project::indicators::moving_averages::ma::ma(
                    ma_type,
                    my_project::indicators::moving_averages::ma::MaData::Candles {
                        candles,
                        source: ma_source,
                    },
                    period as usize,
                )
                .unwrap_or_else(|_| vec![f64::NAN; n]);
                if series.len() != n {
                    continue;
                }
                let dst_off = row.saturating_mul(n);
                if dst_off + n > values.len() {
                    continue;
                }
                values[dst_off..dst_off + n].copy_from_slice(&series);
            }
        }
    }

    Ok((values, n))
}

fn optimize_double_ma_cpu_tiled(
    candles: &my_project::utilities::data_loader::Candles,
    fast_periods: &[u16],
    slow_periods: &[u16],
    fast_ma_type: &str,
    slow_ma_type: &str,
    ma_source: &str,
    fast_ma_params: Option<&HashMap<String, f64>>,
    slow_ma_params: Option<&HashMap<String, f64>>,
    strategy: &StrategyConfig,
    objective: ObjectiveKind,
    top_k: usize,
    include_all: bool,
    fast_alma_offset: f64,
    fast_alma_sigma: f64,
    slow_alma_offset: f64,
    slow_alma_sigma: f64,
    total_pairs: usize,
    heatmap_bins: usize,
    cancel: &AtomicBool,
    progress: Option<&dyn ProgressSink>,
) -> Result<OptimizationResult, String> {
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".to_string());
    }

    let n = candles.close.len();
    if n < 2 {
        return Err("not enough candles (need at least 2)".to_string());
    }

    let fast_info = lookup_ma_info(fast_ma_type).ok_or_else(|| "unknown fast MA".to_string())?;
    if !fast_info.supports_cpu_single {
        return Err(format!("CPU backend does not support MA '{}'", fast_ma_type));
    }
    let slow_info = lookup_ma_info(slow_ma_type).ok_or_else(|| "unknown slow MA".to_string())?;
    if !slow_info.supports_cpu_single {
        return Err(format!("CPU backend does not support MA '{}'", slow_ma_type));
    }

    let close = my_project::utilities::data_loader::source_type(candles, "close");
    let ma_prices = my_project::utilities::data_loader::source_type(candles, ma_source);
    if ma_prices.len() != close.len() {
        return Err("ma source length mismatch".to_string());
    }
    let fv_close = first_finite_idx(close).ok_or_else(|| "close series all non-finite".to_string())?;
    let fv_ma = first_finite_idx(ma_prices).ok_or_else(|| "ma source series all non-finite".to_string())?;
    let first_valid = fv_close.max(fv_ma);

    let lr = compute_log_returns_f64(close);
    let trade_offset = if strategy.trade_on_next_bar { 1usize } else { 0usize };

    let mut agg = StreamAggregator::new(objective, top_k, include_all, n);
    let heatmap_cfg = if heatmap_bins > 0 {
        Some((
            *fast_periods.iter().min().unwrap_or(&0),
            *fast_periods.iter().max().unwrap_or(&0),
            *slow_periods.iter().min().unwrap_or(&0),
            *slow_periods.iter().max().unwrap_or(&0),
        ))
    } else {
        None
    };
    if heatmap_bins > 0 {
        let (f_min, f_max, s_min, s_max) = heatmap_cfg.unwrap();
        agg = agg.with_heatmap(heatmap_bins, heatmap_bins, f_min, f_max, s_min, s_max);
    }
    let (pf_tile, ps_tile) = cpu_tile_sizes(fast_periods.len(), slow_periods.len(), n);

    let mut processed = 0usize;
    emit_progress(progress, processed, total_pairs, "cpu");

    for f_chunk in fast_periods.chunks(pf_tile) {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".to_string());
        }
        let (fast_mat, cols) =
            ma_matrix_cpu_f64(candles, fast_ma_type, ma_source, f_chunk, fast_ma_params, fast_alma_offset, fast_alma_sigma)?;
        for s_chunk in slow_periods.chunks(ps_tile) {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".to_string());
            }
            let (slow_mat, cols2) =
                ma_matrix_cpu_f64(candles, slow_ma_type, ma_source, s_chunk, slow_ma_params, slow_alma_offset, slow_alma_sigma)?;
            if cols2 != cols {
                return Err("MA matrix column mismatch".to_string());
            }

            if include_all {
                for (i, &fast_len) in f_chunk.iter().enumerate() {
                    let fast_row = &fast_mat[i * cols..(i + 1) * cols];
                    for (j, &slow_len) in s_chunk.iter().enumerate() {
                        if fast_len >= slow_len {
                            continue;
                        }
                        let slow_row = &slow_mat[j * cols..(j + 1) * cols];
                        let max_p = (fast_len as usize).max(slow_len as usize).max(1);
                        let t_valid = first_valid.saturating_add(max_p.saturating_sub(1));
                        let start_t = t_valid.saturating_add(trade_offset);
                        let metrics = ta_strategies::double_ma::eval_double_ma_from_log_returns_and_mas_f64(
                            &lr,
                            fast_row,
                            slow_row,
                            start_t,
                            strategy,
                        );
                        let params = DoubleMaParams {
                            fast_len,
                            slow_len,
                            fast_ma_id: 0,
                            slow_ma_id: 0,
                        };
                        agg.push(params, metrics);
                        processed = processed.saturating_add(1);
                        if (processed & 0x1fff) == 0 && cancel.load(Ordering::Relaxed) {
                            return Err("cancelled".to_string());
                        }
                    }
                }
            } else {
                let eval: fn(&[f64], &[f64], &[f64], usize, &StrategyConfig) -> Metrics =
                    if objective == ObjectiveKind::Sharpe {
                        ta_strategies::double_ma::eval_double_ma_from_log_returns_and_mas_f64
                    } else {
                        ta_strategies::double_ma::eval_double_ma_from_log_returns_and_mas_f64_fast
                    };
                let pf = f_chunk.len();
                let ps = s_chunk.len();
                let tile_pairs = pf
                    .checked_mul(ps)
                    .ok_or_else(|| "tile pairs overflow".to_string())?;

                let init = || {
                    let mut local = StreamAggregator::new(objective, top_k, false, n);
                    if let Some((f_min, f_max, s_min, s_max)) = heatmap_cfg {
                        local = local.with_heatmap(heatmap_bins, heatmap_bins, f_min, f_max, s_min, s_max);
                    }
                    local
                };

                let tile = (0..tile_pairs)
                    .into_par_iter()
                    .try_fold(init, |mut local, idx| {
                        if cancel.load(Ordering::Relaxed) {
                            return Err("cancelled".to_string());
                        }
                        let i = idx / ps;
                        let j = idx - i * ps;
                        let fast_len = f_chunk[i];
                        let slow_len = s_chunk[j];
                        if fast_len >= slow_len {
                            return Ok(local);
                        }

                        let fast_row = &fast_mat[i * cols..(i + 1) * cols];
                        let slow_row = &slow_mat[j * cols..(j + 1) * cols];

                        let max_p = (fast_len as usize).max(slow_len as usize).max(1);
                        let t_valid = first_valid.saturating_add(max_p.saturating_sub(1));
                        let start_t = t_valid.saturating_add(trade_offset);
                        let metrics = eval(&lr, fast_row, slow_row, start_t, strategy);
                        local.push(
                            DoubleMaParams {
                                fast_len,
                                slow_len,
                                fast_ma_id: 0,
                                slow_ma_id: 0,
                            },
                            metrics,
                        );
                        Ok(local)
                    })
                    .try_reduce(init, |mut a, b| {
                        a.merge(b);
                        Ok(a)
                    })?;

                processed = processed.saturating_add(tile.num_combos());
                agg.merge(tile);
            }

            emit_progress(progress, processed, total_pairs, "cpu");
        }
    }

    let mut result = agg
        .finalize()
        .ok_or_else(|| "grid search produced no result".to_string())?;

    if objective != ObjectiveKind::Sharpe && !include_all {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".to_string());
        }

        let mut fast_cache: std::collections::HashMap<u16, Vec<f64>> = std::collections::HashMap::new();
        let mut slow_cache: std::collections::HashMap<u16, Vec<f64>> = std::collections::HashMap::new();

        let cache_ma = |cache: &mut std::collections::HashMap<u16, Vec<f64>>, period: u16, ma_type: &str, ma_params: Option<&HashMap<String, f64>>, alma_offset: f64, alma_sigma: f64| -> Result<(), String> {
            if cache.contains_key(&period) {
                return Ok(());
            }
            let periods = [period];
            let (values, _) = ma_matrix_cpu_f64(candles, ma_type, ma_source, &periods, ma_params, alma_offset, alma_sigma)?;
            cache.insert(period, values);
            Ok(())
        };

        let recompute = |fast_len: u16, slow_len: u16, fast_cache: &mut std::collections::HashMap<u16, Vec<f64>>, slow_cache: &mut std::collections::HashMap<u16, Vec<f64>>| -> Result<Metrics, String> {
            cache_ma(fast_cache, fast_len, fast_ma_type, fast_ma_params, fast_alma_offset, fast_alma_sigma)?;
            cache_ma(slow_cache, slow_len, slow_ma_type, slow_ma_params, slow_alma_offset, slow_alma_sigma)?;

            let fast = fast_cache.get(&fast_len).ok_or_else(|| "fast MA cache miss".to_string())?;
            let slow = slow_cache.get(&slow_len).ok_or_else(|| "slow MA cache miss".to_string())?;

            let max_p = (fast_len as usize).max(slow_len as usize).max(1);
            let t_valid = first_valid.saturating_add(max_p.saturating_sub(1));
            let start_t = t_valid.saturating_add(trade_offset);
            Ok(ta_strategies::double_ma::eval_double_ma_from_log_returns_and_mas_f64(
                &lr,
                fast,
                slow,
                start_t,
                strategy,
            ))
        };

        result.best_metrics = recompute(
            result.best_params.fast_len,
            result.best_params.slow_len,
            &mut fast_cache,
            &mut slow_cache,
        )?;

        for (p, m) in result.top.iter_mut() {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".to_string());
            }
            *m = recompute(p.fast_len, p.slow_len, &mut fast_cache, &mut slow_cache)?;
        }
    }

    Ok(result)
}

#[cfg(not(feature = "cuda"))]
fn optimize_double_ma_gpu_sweep_tiled(
    _candles: &my_project::utilities::data_loader::Candles,
    _device_id: u32,
    _fast_periods: &[u16],
    _slow_periods: &[u16],
    _fast_ma_type: &str,
    _slow_ma_type: &str,
    _ma_source: &str,
    _fast_ma_params: Option<&HashMap<String, f64>>,
    _slow_ma_params: Option<&HashMap<String, f64>>,
    _strategy: &StrategyConfig,
    _objective: ObjectiveKind,
    _top_k: usize,
    _include_all: bool,
    _fast_alma_offset: f64,
    _fast_alma_sigma: f64,
    _slow_alma_offset: f64,
    _slow_alma_sigma: f64,
    _total_pairs: usize,
    _heatmap_bins: usize,
    _cancel: &AtomicBool,
    _progress: Option<&dyn ProgressSink>,
) -> Result<OptimizationResult, String> {
    Err("GPU backend requires building `ta_desktop_demo_app` with `--features cuda`".to_string())
}

#[cfg(feature = "cuda")]
fn ma_matrix_cuda_host_f32(
    selector: &CudaMaSelector,
    device_id: usize,
    candles: &my_project::utilities::data_loader::Candles,
    prices_f32: &[f32],
    ma_type: &str,
    ma_source: &str,
    periods: &[u16],
    ma_params: Option<&HashMap<String, f64>>,
    alma_offset: f64,
    alma_sigma: f64,
) -> Result<(Vec<f32>, usize), String> {
    let n = prices_f32.len();
    if periods.is_empty() {
        return Err("empty period list".to_string());
    }
    let rows = periods.len();
    let expected_len = rows
        .checked_mul(n)
        .ok_or_else(|| "rows*cols overflow".to_string())?;

    let start = periods[0] as usize;
    let end = periods[rows - 1] as usize;
    let step = ma_period_step(periods).max(1);

    if ma_type.eq_ignore_ascii_case("alma") {
        let alma = CudaAlma::new(device_id).map_err(|e| e.to_string())?;
        let sweep = my_project::indicators::moving_averages::alma::AlmaBatchRange {
            period: (start, end, step),
            offset: (alma_offset, alma_offset, 0.0),
            sigma: (alma_sigma, alma_sigma, 0.0),
        };
        let mut out = vec![f32::NAN; expected_len];
        let (r, c, combos) = alma
            .alma_batch_into_host_f32(prices_f32, &sweep, &mut out)
            .map_err(|e| e.to_string())?;
        if r != rows || c != n {
            return Err(format!(
                "cuda alma batch returned shape ({r},{c}) (expected ({rows},{n}))"
            ));
        }
        if combos.len() != rows {
            return Err(format!(
                "cuda alma batch returned {} combos (expected {})",
                combos.len(),
                rows
            ));
        }
        return Ok((out, n));
    }

    let requires_candles = lookup_ma_info(ma_type).map(|m| m.requires_candles).unwrap_or(false);
    let data = if requires_candles {
        CudaMaData::Candles {
            candles,
            source: ma_source,
        }
    } else {
        CudaMaData::SliceF32(prices_f32)
    };

    let sweep_res = match ma_params {
        Some(p) => selector.ma_sweep_to_host_f32_with_params(ma_type, data, start, end, step, p),
        None => selector.ma_sweep_to_host_f32(ma_type, data, start, end, step),
    };

    let host = match sweep_res {
        Ok((values, r, c)) => {
            if r != rows || c != n || values.len() != expected_len {
                return Err(format!(
                    "cuda ma sweep '{ma_type}' returned shape ({r},{c}) len {} (expected ({rows},{n}) len {expected_len})",
                    values.len()
                ));
            }
            values
        }
        Err(sweep_err) => {
            let mut out = vec![f32::NAN; expected_len];
            for (row, &period) in periods.iter().enumerate() {
                let data = if requires_candles {
                    CudaMaData::Candles {
                        candles,
                        source: ma_source,
                    }
                } else {
                    CudaMaData::SliceF32(prices_f32)
                };
                let series = match ma_params {
                    Some(p) => selector.ma_to_host_f32_with_params(ma_type, data, period as usize, p),
                    None => selector.ma_to_host_f32(ma_type, data, period as usize),
                }.map_err(|e| {
                        format!(
                            "cuda ma sweep '{ma_type}' failed: {sweep_err}; fallback period {period} failed: {e}"
                        )
                    })?;
                if series.len() != n {
                    return Err(format!(
                        "cuda ma '{ma_type}' period {period} returned len {} (expected {n})",
                        series.len()
                    ));
                }
                let dst_off = row.saturating_mul(n);
                out[dst_off..dst_off + n].copy_from_slice(&series);
            }
            out
        }
    };

    Ok((host, n))
}

#[cfg(feature = "cuda")]
fn optimize_double_ma_gpu_sweep_tiled(
    candles: &my_project::utilities::data_loader::Candles,
    device_id: u32,
    fast_periods: &[u16],
    slow_periods: &[u16],
    fast_ma_type: &str,
    slow_ma_type: &str,
    ma_source: &str,
    fast_ma_params: Option<&HashMap<String, f64>>,
    slow_ma_params: Option<&HashMap<String, f64>>,
    strategy: &StrategyConfig,
    objective: ObjectiveKind,
    top_k: usize,
    include_all: bool,
    fast_alma_offset: f64,
    fast_alma_sigma: f64,
    slow_alma_offset: f64,
    slow_alma_sigma: f64,
    total_pairs: usize,
    heatmap_bins: usize,
    cancel: &AtomicBool,
    progress: Option<&dyn ProgressSink>,
) -> Result<OptimizationResult, String> {
    if !cuda_available() {
        return Err("CUDA device not available (cuda_available() == false)".to_string());
    }

    let n = candles.close.len();
    if n < 2 {
        return Err("not enough candles (need at least 2)".to_string());
    }

    let fast_info = lookup_ma_info(fast_ma_type).ok_or_else(|| "unknown fast MA".to_string())?;
    if !fast_info.supports_cuda_single {
        return Err(format!(
            "GPU (MA sweep) backend does not support MA '{}'",
            fast_ma_type
        ));
    }
    let slow_info = lookup_ma_info(slow_ma_type).ok_or_else(|| "unknown slow MA".to_string())?;
    if !slow_info.supports_cuda_single {
        return Err(format!(
            "GPU (MA sweep) backend does not support MA '{}'",
            slow_ma_type
        ));
    }

    let ma_prices = my_project::utilities::data_loader::source_type(candles, ma_source);
    if ma_prices.len() != n {
        return Err("ma source length mismatch".to_string());
    }
    let close = my_project::utilities::data_loader::source_type(candles, "close");
    let fv_close = first_finite_idx(close).ok_or_else(|| "close series all non-finite".to_string())?;
    let fv_ma = first_finite_idx(ma_prices).ok_or_else(|| "ma source series all non-finite".to_string())?;
    let first_valid = fv_close.max(fv_ma);
    let lr = compute_log_returns_f64(close);
    let trade_offset = if strategy.trade_on_next_bar { 1usize } else { 0usize };

    let prices_f32: Vec<f32> = ma_prices.iter().map(|&x| x as f32).collect();

    let selector = CudaMaSelector::new(device_id as usize);
    let mut agg = StreamAggregator::new(objective, top_k, include_all, n);
    let heatmap_cfg = if heatmap_bins > 0 {
        Some((
            *fast_periods.iter().min().unwrap_or(&0),
            *fast_periods.iter().max().unwrap_or(&0),
            *slow_periods.iter().min().unwrap_or(&0),
            *slow_periods.iter().max().unwrap_or(&0),
        ))
    } else {
        None
    };
    if heatmap_bins > 0 {
        let (f_min, f_max, s_min, s_max) = heatmap_cfg.unwrap();
        agg = agg.with_heatmap(heatmap_bins, heatmap_bins, f_min, f_max, s_min, s_max);
    }
    let (pf_tile, ps_tile) = sweep_tile_sizes(fast_periods.len(), slow_periods.len(), n);

    let mut processed = 0usize;
    emit_progress(progress, processed, total_pairs, "gpu-sweep");

    for f_chunk in fast_periods.chunks(pf_tile) {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".to_string());
        }
        let (fast_mat, cols) = ma_matrix_cuda_host_f32(
            &selector,
            device_id as usize,
            candles,
            &prices_f32,
            fast_ma_type,
            ma_source,
            f_chunk,
            fast_ma_params,
            fast_alma_offset,
            fast_alma_sigma,
        )?;
        for s_chunk in slow_periods.chunks(ps_tile) {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".to_string());
            }
            let (slow_mat, cols2) = ma_matrix_cuda_host_f32(
                &selector,
                device_id as usize,
                candles,
                &prices_f32,
                slow_ma_type,
                ma_source,
                s_chunk,
                slow_ma_params,
                slow_alma_offset,
                slow_alma_sigma,
            )?;
            if cols2 != cols {
                return Err("MA matrix column mismatch".to_string());
            }

            if include_all {
                for (i, &fast_len) in f_chunk.iter().enumerate() {
                    let fast_row = &fast_mat[i * cols..(i + 1) * cols];
                    for (j, &slow_len) in s_chunk.iter().enumerate() {
                        if fast_len >= slow_len {
                            continue;
                        }
                        let slow_row = &slow_mat[j * cols..(j + 1) * cols];
                        let max_p = (fast_len as usize).max(slow_len as usize).max(1);
                        let t_valid = first_valid.saturating_add(max_p.saturating_sub(1));
                        let start_t = t_valid.saturating_add(trade_offset);
                        let metrics = ta_strategies::double_ma::eval_double_ma_from_log_returns_and_mas_f32(
                            &lr,
                            fast_row,
                            slow_row,
                            start_t,
                            strategy,
                        );
                        let params = DoubleMaParams {
                            fast_len,
                            slow_len,
                            fast_ma_id: 0,
                            slow_ma_id: 0,
                        };
                        agg.push(params, metrics);
                        processed = processed.saturating_add(1);
                        if (processed & 0x1fff) == 0 && cancel.load(Ordering::Relaxed) {
                            return Err("cancelled".to_string());
                        }
                    }
                }
            } else {
                let eval: fn(&[f64], &[f32], &[f32], usize, &StrategyConfig) -> Metrics =
                    if objective == ObjectiveKind::Sharpe {
                        ta_strategies::double_ma::eval_double_ma_from_log_returns_and_mas_f32
                    } else {
                        ta_strategies::double_ma::eval_double_ma_from_log_returns_and_mas_f32_fast
                    };
                let pf = f_chunk.len();
                let ps = s_chunk.len();
                let tile_pairs = pf
                    .checked_mul(ps)
                    .ok_or_else(|| "tile pairs overflow".to_string())?;

                let init = || {
                    let mut local = StreamAggregator::new(objective, top_k, false, n);
                    if let Some((f_min, f_max, s_min, s_max)) = heatmap_cfg {
                        local = local.with_heatmap(heatmap_bins, heatmap_bins, f_min, f_max, s_min, s_max);
                    }
                    local
                };

                let tile = (0..tile_pairs)
                    .into_par_iter()
                    .try_fold(init, |mut local, idx| {
                        if cancel.load(Ordering::Relaxed) {
                            return Err("cancelled".to_string());
                        }
                        let i = idx / ps;
                        let j = idx - i * ps;
                        let fast_len = f_chunk[i];
                        let slow_len = s_chunk[j];
                        if fast_len >= slow_len {
                            return Ok(local);
                        }

                        let fast_row = &fast_mat[i * cols..(i + 1) * cols];
                        let slow_row = &slow_mat[j * cols..(j + 1) * cols];

                        let max_p = (fast_len as usize).max(slow_len as usize).max(1);
                        let t_valid = first_valid.saturating_add(max_p.saturating_sub(1));
                        let start_t = t_valid.saturating_add(trade_offset);
                        let metrics = eval(&lr, fast_row, slow_row, start_t, strategy);
                        local.push(
                            DoubleMaParams {
                                fast_len,
                                slow_len,
                                fast_ma_id: 0,
                                slow_ma_id: 0,
                            },
                            metrics,
                        );
                        Ok(local)
                    })
                    .try_reduce(init, |mut a, b| {
                        a.merge(b);
                        Ok(a)
                    })?;

                processed = processed.saturating_add(tile.num_combos());
                agg.merge(tile);
            }

            emit_progress(progress, processed, total_pairs, "gpu-sweep");
        }
    }

    let mut result = agg
        .finalize()
        .ok_or_else(|| "grid search produced no result".to_string())?;

    if objective != ObjectiveKind::Sharpe && !include_all {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".to_string());
        }

        let mut fast_cache: std::collections::HashMap<u16, Vec<f32>> = std::collections::HashMap::new();
        let mut slow_cache: std::collections::HashMap<u16, Vec<f32>> = std::collections::HashMap::new();

        let cache_ma = |cache: &mut std::collections::HashMap<u16, Vec<f32>>, period: u16, ma_type: &str, ma_params: Option<&HashMap<String, f64>>, alma_offset: f64, alma_sigma: f64| -> Result<(), String> {
            if cache.contains_key(&period) {
                return Ok(());
            }
            let periods = [period];
            let (values, _) = ma_matrix_cuda_host_f32(
                &selector,
                device_id as usize,
                candles,
                &prices_f32,
                ma_type,
                ma_source,
                &periods,
                ma_params,
                alma_offset,
                alma_sigma,
            )?;
            cache.insert(period, values);
            Ok(())
        };

        let recompute = |fast_len: u16, slow_len: u16, fast_cache: &mut std::collections::HashMap<u16, Vec<f32>>, slow_cache: &mut std::collections::HashMap<u16, Vec<f32>>| -> Result<Metrics, String> {
            cache_ma(fast_cache, fast_len, fast_ma_type, fast_ma_params, fast_alma_offset, fast_alma_sigma)?;
            cache_ma(slow_cache, slow_len, slow_ma_type, slow_ma_params, slow_alma_offset, slow_alma_sigma)?;

            let fast = fast_cache.get(&fast_len).ok_or_else(|| "fast MA cache miss".to_string())?;
            let slow = slow_cache.get(&slow_len).ok_or_else(|| "slow MA cache miss".to_string())?;

            let max_p = (fast_len as usize).max(slow_len as usize).max(1);
            let t_valid = first_valid.saturating_add(max_p.saturating_sub(1));
            let start_t = t_valid.saturating_add(trade_offset);
            Ok(ta_strategies::double_ma::eval_double_ma_from_log_returns_and_mas_f32(
                &lr,
                fast,
                slow,
                start_t,
                strategy,
            ))
        };

        result.best_metrics = recompute(
            result.best_params.fast_len,
            result.best_params.slow_len,
            &mut fast_cache,
            &mut slow_cache,
        )?;

        for (p, m) in result.top.iter_mut() {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".to_string());
            }
            *m = recompute(p.fast_len, p.slow_len, &mut fast_cache, &mut slow_cache)?;
        }
    }

    Ok(result)
}

#[cfg(not(all(feature = "cuda", feature = "cuda-backtest-kernel")))]
fn optimize_double_ma_gpu_kernel_tiled(
    _candles: &my_project::utilities::data_loader::Candles,
    _device_id: u32,
    _fast_periods: &[u16],
    _slow_periods: &[u16],
    _fast_ma_type: &str,
    _slow_ma_type: &str,
    _ma_source: &str,
    _fast_ma_params: Option<&HashMap<String, f64>>,
    _slow_ma_params: Option<&HashMap<String, f64>>,
    _strategy: &StrategyConfig,
    _objective: ObjectiveKind,
    _top_k: usize,
    _include_all: bool,
    _fast_alma_offset: f64,
    _fast_alma_sigma: f64,
    _slow_alma_offset: f64,
    _slow_alma_sigma: f64,
    _export_all_csv_path: Option<&str>,
    _total_pairs: usize,
    _heatmap_bins: usize,
    _cancel: &AtomicBool,
    _progress: Option<&dyn ProgressSink>,
) -> Result<OptimizationResult, String> {
    Err("GPU kernel backend requires building `ta_desktop_demo_app` with `--features cuda,cuda-backtest-kernel`".to_string())
}

#[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
fn optimize_double_ma_gpu_kernel_tiled(
    candles: &my_project::utilities::data_loader::Candles,
    device_id: u32,
    fast_periods: &[u16],
    slow_periods: &[u16],
    fast_ma_type: &str,
    slow_ma_type: &str,
    ma_source: &str,
    fast_ma_params: Option<&HashMap<String, f64>>,
    slow_ma_params: Option<&HashMap<String, f64>>,
    strategy: &StrategyConfig,
    objective: ObjectiveKind,
    top_k: usize,
    include_all: bool,
    fast_alma_offset: f64,
    fast_alma_sigma: f64,
    slow_alma_offset: f64,
    slow_alma_sigma: f64,
    export_all_csv_path: Option<&str>,
    total_pairs: usize,
    heatmap_bins: usize,
    cancel: &AtomicBool,
    progress: Option<&dyn ProgressSink>,
) -> Result<OptimizationResult, String> {
    if !cuda_available() {
        return Err("CUDA device not available (cuda_available() == false)".to_string());
    }

    crate::gpu_backtest_kernel::optimize_double_ma_gpu_kernel(
        candles,
        fast_periods,
        slow_periods,
        fast_ma_type,
        slow_ma_type,
        ma_source,
        fast_ma_params,
        slow_ma_params,
        strategy,
        objective,
        top_k,
        include_all,
        crate::gpu_backtest_kernel::KernelConfig {
            device_id,
            fast_alma_offset,
            fast_alma_sigma,
            slow_alma_offset,
            slow_alma_sigma,
        },
        export_all_csv_path,
        total_pairs,
        heatmap_bins,
        cancel,
        progress,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trending_candles(len: usize) -> my_project::utilities::data_loader::Candles {
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

        my_project::utilities::data_loader::Candles::new(ts, open, high, low, close, vol)
    }

    #[test]
    fn cpu_smoke_small_grid_counts_and_shapes() {
        let candles = make_trending_candles(5000);
        let cancel = AtomicBool::new(false);

        let req = DoubleMaRequest {
            backend: Backend::CpuOnly,
            data_id: "test".to_string(),
            fast_range: (5, 7, 1),
            slow_range: (10, 11, 1),
            fast_ma_types: vec!["sma".to_string()],
            slow_ma_types: vec!["sma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Sharpe,
            mode: OptimizationMode::Grid,
            top_k: Some(5),
            include_all: Some(false),
            heatmap_bins: Some(16),
        };

        let res = run_double_ma_optimization_blocking_with_candles(req, &candles, &cancel, None)
            .expect("cpu run");

        assert_eq!(res.num_combos, 6);
        assert!(res.best_params.fast_len < res.best_params.slow_len);
        assert!(res.top.len() <= 5);
        assert!(res.heatmap.is_some());
    }

    #[test]
    fn cpu_ma_params_gaussian_poles_changes_output() {
        let candles = make_trending_candles(1024);
        let periods = [14u16];

        let mut p1: HashMap<String, f64> = HashMap::new();
        p1.insert("poles".to_string(), 1.0);
        let (out1, cols1) = ma_matrix_cpu_f64(
            &candles,
            "gaussian",
            "close",
            &periods,
            Some(&p1),
            0.85,
            6.0,
        )
        .expect("gaussian poles=1");

        let mut p4: HashMap<String, f64> = HashMap::new();
        p4.insert("poles".to_string(), 4.0);
        let (out4, cols4) = ma_matrix_cpu_f64(
            &candles,
            "gaussian",
            "close",
            &periods,
            Some(&p4),
            0.85,
            6.0,
        )
        .expect("gaussian poles=4");

        assert_eq!(cols1, cols4);
        assert_eq!(out1.len(), out4.len());

        let idx = cols1 - 1;
        let a = out1[idx];
        let b = out4[idx];
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() > 1e-9);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_ma_params_gaussian_poles_changes_output() {
        if !cuda_available() {
            return;
        }

        let candles = make_trending_candles(2048);
        let ma_source = "close";
        let prices_f32: Vec<f32> =
            my_project::utilities::data_loader::source_type(&candles, ma_source)
                .iter()
                .map(|&x| x as f32)
                .collect();
        let selector = CudaMaSelector::new(0);

        let periods = [14u16];

        let mut p1: HashMap<String, f64> = HashMap::new();
        p1.insert("poles".to_string(), 1.0);
        let (out1, cols1) = ma_matrix_cuda_host_f32(
            &selector,
            0,
            &candles,
            &prices_f32,
            "gaussian",
            ma_source,
            &periods,
            Some(&p1),
            0.85,
            6.0,
        )
        .expect("cuda gaussian poles=1");

        let mut p4: HashMap<String, f64> = HashMap::new();
        p4.insert("poles".to_string(), 4.0);
        let (out4, cols4) = ma_matrix_cuda_host_f32(
            &selector,
            0,
            &candles,
            &prices_f32,
            "gaussian",
            ma_source,
            &periods,
            Some(&p4),
            0.85,
            6.0,
        )
        .expect("cuda gaussian poles=4");

        assert_eq!(cols1, cols4);
        assert_eq!(out1.len(), out4.len());

        let idx = cols1 - 1;
        let a = out1[idx];
        let b = out4[idx];
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() > 1e-6);
    }

    #[test]
    fn drilldown_smoke_has_bins_and_values() {
        let candles = make_trending_candles(5000);

        let req = DoubleMaDrilldownRequest {
            data_id: "test".to_string(),
            params: DoubleMaParamsResolved {
                fast_len: 10,
                slow_len: 30,
                fast_ma_type: "sma".to_string(),
                slow_ma_type: "sma".to_string(),
            },
            ma_source: "close".to_string(),
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            bins: Some(256),
        };

        let curves = compute_double_ma_drilldown_blocking_with_candles(req, &candles)
            .expect("drilldown");

        assert!(curves.equity.bins > 0);
        assert_eq!(curves.equity.bins, curves.drawdown.bins);
        assert_eq!(curves.equity.minmax.len(), curves.equity.bins.saturating_mul(2));
        assert_eq!(
            curves.drawdown.minmax.len(),
            curves.drawdown.bins.saturating_mul(2)
        );

        assert!(curves.equity.minmax.iter().all(|v| v.is_finite()));
        assert!(curves
            .drawdown
            .minmax
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_best_matches_include_all_true() {
        if !cuda_available() {
            return;
        }

        let candles = make_trending_candles(5000);
        let cancel = AtomicBool::new(false);

        let req_all = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 20, 1),
            slow_range: (10, 60, 1),
            fast_ma_types: vec!["sma".to_string()],
            slow_ma_types: vec!["sma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(0),
            include_all: Some(true),
            heatmap_bins: Some(0),
        };

        let req_top = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 20, 1),
            slow_range: (10, 60, 1),
            fast_ma_types: vec!["sma".to_string()],
            slow_ma_types: vec!["sma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(1),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        let a = run_double_ma_optimization_blocking_with_candles(req_all, &candles, &cancel, None)
            .expect("gpu-kernel include_all=true");

        let b = run_double_ma_optimization_blocking_with_candles(req_top, &candles, &cancel, None)
            .expect("gpu-kernel include_all=false");

        assert_eq!(a.best_params.fast_len, b.best_params.fast_len);
        assert_eq!(a.best_params.slow_len, b.best_params.slow_len);

        let h = b.heatmap.expect("heatmap");
        assert_eq!(h.values.len(), h.bins_fast.saturating_mul(h.bins_slow));
        assert!(h.values.iter().any(|v| v.is_some()));
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_export_all_csv_writes_one_row_per_pair() {
        if !cuda_available() {
            return;
        }

        let candles = make_trending_candles(5000);
        let cancel = AtomicBool::new(false);

        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join("ta_desktop_demo_exports");
        std::fs::create_dir_all(&dir).expect("create temp export dir");
        let path = dir.join(format!("gpu_kernel_export_{t}.csv"));

        let req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 7, 1),
            slow_range: (10, 11, 1),
            fast_ma_types: vec!["sma".to_string()],
            slow_ma_types: vec!["sma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: Some(path.to_string_lossy().to_string()),
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Sharpe,
            mode: OptimizationMode::Grid,
            top_k: Some(1),
            include_all: Some(false),
            heatmap_bins: Some(0),
        };

        let res = run_double_ma_optimization_blocking_with_candles(req, &candles, &cancel, None)
            .expect("gpu-kernel export");

        let csv = std::fs::read_to_string(&path).expect("read export csv");
        let lines: Vec<&str> = csv.lines().collect();
        assert!(!lines.is_empty(), "expected non-empty csv");
        assert_eq!(
            lines[0],
            "fast_ma,slow_ma,fast_len,slow_len,pnl,sharpe,max_dd,trades,exposure,net_exposure,score"
        );
        assert_eq!(lines.len(), res.num_combos + 1);
        for (i, line) in lines.iter().enumerate().skip(1) {
            assert_eq!(line.split(',').count(), 11, "bad csv column count on line {i}");
        }

        let best_prefix = format!(
            "sma,sma,{},{}",
            res.best_params.fast_len,
            res.best_params.slow_len
        );
        assert!(
            lines.iter().skip(1).any(|l| l.starts_with(&best_prefix)),
            "expected csv to contain best params row prefix '{best_prefix}'"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["sma".to_string()],
            slow_ma_types: vec!["sma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_alma() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let mut alma_p: HashMap<String, f64> = HashMap::new();
        alma_p.insert("offset".to_string(), 0.85);
        alma_p.insert("sigma".to_string(), 6.0);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["alma".to_string()],
            slow_ma_types: vec!["alma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: Some(alma_p.clone()),
            slow_ma_params: Some(alma_p),
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_jsa() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["jsa".to_string()],
            slow_ma_types: vec!["jsa".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_cwma() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["cwma".to_string()],
            slow_ma_types: vec!["cwma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_cora_wave() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let mut p: HashMap<String, f64> = HashMap::new();
        p.insert("r_multi".to_string(), 3.0);
        p.insert("smooth".to_string(), 1.0);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["cora_wave".to_string()],
            slow_ma_types: vec!["cora_wave".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: Some(p.clone()),
            slow_ma_params: Some(p),
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched =
            run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
                .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_epma() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let mut epma_p: HashMap<String, f64> = HashMap::new();
        epma_p.insert("offset".to_string(), 3.0);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["epma".to_string()],
            slow_ma_types: vec!["epma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: Some(epma_p.clone()),
            slow_ma_params: Some(epma_p),
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_pwma() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["pwma".to_string()],
            slow_ma_types: vec!["pwma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_srwma() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["srwma".to_string()],
            slow_ma_types: vec!["srwma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_supersmoother() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["supersmoother".to_string()],
            slow_ma_types: vec!["supersmoother".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_zlema() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["zlema".to_string()],
            slow_ma_types: vec!["zlema".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_nma() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["nma".to_string()],
            slow_ma_types: vec!["nma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_hma() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["hma".to_string()],
            slow_ma_types: vec!["hma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_jma() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["jma".to_string()],
            slow_ma_types: vec!["jma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_edcf() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["edcf".to_string()],
            slow_ma_types: vec!["edcf".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_ehlers_itrend() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (10, 120, 1),
            slow_range: (20, 240, 1),
            fast_ma_types: vec!["ehlers_itrend".to_string()],
            slow_ma_types: vec!["ehlers_itrend".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_vwma() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["vwma".to_string()],
            slow_ma_types: vec!["vwma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_vpwma() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (5, 120, 1),
            slow_range: (10, 240, 1),
            fast_ma_types: vec!["vpwma".to_string()],
            slow_ma_types: vec!["vpwma".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }

    #[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
    #[test]
    fn gpu_kernel_batched_matches_unbatched_best_params_frama() {
        if !cuda_available() {
            return;
        }

        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned");

        let prev = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

        let candles = make_trending_candles(50_000);
        let cancel = AtomicBool::new(false);

        let base_req = DoubleMaRequest {
            backend: Backend::GpuKernel { device_id: 0 },
            data_id: "test".to_string(),
            fast_range: (10, 120, 1),
            slow_range: (20, 240, 1),
            fast_ma_types: vec!["frama".to_string()],
            slow_ma_types: vec!["frama".to_string()],
            ma_source: "close".to_string(),
            export_all_csv_path: None,
            fast_ma_params: None,
            slow_ma_params: None,
            strategy: StrategyConfig::default(),
            objective: ObjectiveKind::Pnl,
            mode: OptimizationMode::Grid,
            top_k: Some(10),
            include_all: Some(false),
            heatmap_bins: Some(32),
        };

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "8192");
        let single = run_double_ma_optimization_blocking_with_candles(
            base_req.clone(),
            &candles,
            &cancel,
            None,
        )
        .expect("unbatched gpu-kernel run");

        std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", "128");
        let batched = run_double_ma_optimization_blocking_with_candles(base_req, &candles, &cancel, None)
            .expect("batched gpu-kernel run");

        match prev {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        assert_eq!(single.num_combos, batched.num_combos);
        assert_eq!(single.best_params.fast_len, batched.best_params.fast_len);
        assert_eq!(single.best_params.slow_len, batched.best_params.slow_len);

        let h1 = single.heatmap.expect("single heatmap");
        let h2 = batched.heatmap.expect("batched heatmap");
        assert_eq!(h1.bins_fast, h2.bins_fast);
        assert_eq!(h1.bins_slow, h2.bins_slow);
        assert_eq!(h1.fast_min, h2.fast_min);
        assert_eq!(h1.fast_max, h2.fast_max);
        assert_eq!(h1.slow_min, h2.slow_min);
        assert_eq!(h1.slow_max, h2.slow_max);
        assert_eq!(h1.values, h2.values);
    }
}
