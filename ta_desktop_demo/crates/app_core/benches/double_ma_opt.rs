use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use ta_app_core::{Backend, DoubleMaRequest};
use ta_optimizer::{ObjectiveKind, OptimizationMode};
use ta_strategies::double_ma::StrategyConfig;

use my_project::utilities::data_loader::Candles;

#[derive(Clone, Copy)]
struct MaBenchCase {
    label: &'static str,
    fast_ma: &'static str,
    slow_ma: &'static str,
}

fn make_synth_candles(len: usize) -> Candles {
    let mut ts = Vec::with_capacity(len);
    let mut open = Vec::with_capacity(len);
    let mut high = Vec::with_capacity(len);
    let mut low = Vec::with_capacity(len);
    let mut close = Vec::with_capacity(len);
    let mut vol: Vec<f64> = Vec::with_capacity(len);

    let mut px = 100.0f64;
    for i in 0..len {
        ts.push(i as i64);
        let drift = 0.00005;
        let noise = (i as f64 * 0.017).sin() * 0.001;
        px *= 1.0 + drift + noise;

        open.push(px);
        high.push(px * 1.001);
        low.push(px * 0.999);
        close.push(px);
        vol.push(1_000.0);
    }

    Candles::new(ts, open, high, low, close, vol)
}

fn alma_params_default() -> HashMap<String, f64> {
    let mut p: HashMap<String, f64> = HashMap::new();
    p.insert("offset".to_string(), 0.85);
    p.insert("sigma".to_string(), 6.0);
    p
}

fn epma_params_default() -> HashMap<String, f64> {
    let mut p: HashMap<String, f64> = HashMap::new();
    p.insert("offset".to_string(), 3.0);
    p
}

fn expand_range_u16((start, end, step): (u32, u32, u32)) -> Vec<u16> {
    let push_checked = |out: &mut Vec<u16>, v: u32| {
        if v <= u16::MAX as u32 {
            out.push(v as u16);
        }
    };

    if step == 0 || start == end {
        let mut out = Vec::with_capacity(1);
        push_checked(&mut out, start);
        return out;
    }

    let step = step.max(1);
    let (lo, hi) = if start <= end { (start, end) } else { (end, start) };

    let mut out = Vec::new();
    let mut v = lo;
    loop {
        push_checked(&mut out, v);
        if v == hi {
            break;
        }
        match v.checked_add(step) {
            Some(next) if next > v && next <= hi => v = next,
            _ => break,
        }
    }
    out
}

fn count_pairs_fast_lt_slow(fast_periods: &[u16], slow_periods: &[u16]) -> usize {
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

fn bench_double_ma_cpu(c: &mut Criterion) {
    let candles = make_synth_candles(200_000);
    let cancel = AtomicBool::new(false);

    let ma_cases: &[MaBenchCase] = &[
        MaBenchCase {
            label: "sma_sma",
            fast_ma: "sma",
            slow_ma: "sma",
        },
        MaBenchCase {
            label: "alma_alma",
            fast_ma: "alma",
            slow_ma: "alma",
        },
    ];

    let cases: &[(&str, (u32, u32, u32), (u32, u32, u32))] = &[
        (
            "small",
            (5, 50, 1),
            (20, 200, 5),
        ),
        (
            "medium",
            (5, 100, 1),
            (10, 200, 1),
        ),
        (
            "large",
            (5, 200, 1),
            (10, 400, 1),
        ),
    ];

    for ma in ma_cases {
        let mut group = c.benchmark_group(format!("double_ma/cpu/{}/200k", ma.label));
        group.measurement_time(Duration::from_secs(8));
        group.sample_size(if ma.fast_ma == "alma" || ma.slow_ma == "alma" { 10 } else { 20 });

        for (label, fast_range, slow_range) in cases.iter().copied() {
            let fast_periods = expand_range_u16(fast_range);
            let slow_periods = expand_range_u16(slow_range);
            let pairs = count_pairs_fast_lt_slow(&fast_periods, &slow_periods);
            let name = format!("{label}/{pairs}pairs");

            let fast_ma_params = if ma.fast_ma == "alma" { Some(alma_params_default()) } else { None };
            let slow_ma_params = if ma.slow_ma == "alma" { Some(alma_params_default()) } else { None };

            let req = DoubleMaRequest {
                backend: Backend::CpuOnly,
                data_id: "bench".to_string(),
                fast_range,
                slow_range,
                fast_ma_types: vec![ma.fast_ma.to_string()],
                slow_ma_types: vec![ma.slow_ma.to_string()],
                ma_source: "close".to_string(),
                export_all_csv_path: None,
                fast_ma_params,
                slow_ma_params,
                strategy: StrategyConfig::default(),
                objective: ObjectiveKind::Sharpe,
                mode: OptimizationMode::Grid,
                top_k: Some(50),
                include_all: Some(false),
                heatmap_bins: Some(64),
            };

            group.bench_function(&name, |b| {
                b.iter(|| {
                    let req2 = req.clone();
                    let _ = ta_app_core::run_double_ma_optimization_blocking_with_candles(
                        req2,
                        &candles,
                        &cancel,
                        None,
                    )
                    .expect("cpu run");
                })
            });
        }

        group.finish();
    }
}

#[cfg(feature = "cuda")]
fn bench_double_ma_gpu_sweep(c: &mut Criterion) {
    if !my_project::cuda::cuda_available() {
        return;
    }
    let candles = make_synth_candles(200_000);
    let cancel = AtomicBool::new(false);

    let ma_cases: &[MaBenchCase] = &[
        MaBenchCase {
            label: "sma_sma",
            fast_ma: "sma",
            slow_ma: "sma",
        },
        MaBenchCase {
            label: "alma_alma",
            fast_ma: "alma",
            slow_ma: "alma",
        },
    ];

    let cases: &[(&str, (u32, u32, u32), (u32, u32, u32))] = &[
        (
            "small",
            (5, 50, 1),
            (20, 200, 5),
        ),
        (
            "medium",
            (5, 100, 1),
            (10, 200, 1),
        ),
        (
            "large",
            (5, 200, 1),
            (10, 400, 1),
        ),
    ];

    for ma in ma_cases {
        let mut group = c.benchmark_group(format!("double_ma/gpu_sweep/{}/200k", ma.label));
        group.measurement_time(Duration::from_secs(8));
        group.sample_size(if ma.fast_ma == "alma" || ma.slow_ma == "alma" { 10 } else { 20 });

        for (label, fast_range, slow_range) in cases.iter().copied() {
            let fast_periods = expand_range_u16(fast_range);
            let slow_periods = expand_range_u16(slow_range);
            let pairs = count_pairs_fast_lt_slow(&fast_periods, &slow_periods);
            let name = format!("{label}/{pairs}pairs");

            let fast_ma_params = if ma.fast_ma == "alma" { Some(alma_params_default()) } else { None };
            let slow_ma_params = if ma.slow_ma == "alma" { Some(alma_params_default()) } else { None };

            let req = DoubleMaRequest {
                backend: Backend::GpuOnly { device_id: 0 },
                data_id: "bench".to_string(),
                fast_range,
                slow_range,
                fast_ma_types: vec![ma.fast_ma.to_string()],
                slow_ma_types: vec![ma.slow_ma.to_string()],
                ma_source: "close".to_string(),
                export_all_csv_path: None,
                fast_ma_params,
                slow_ma_params,
                strategy: StrategyConfig::default(),
                objective: ObjectiveKind::Sharpe,
                mode: OptimizationMode::Grid,
                top_k: Some(50),
                include_all: Some(false),
                heatmap_bins: Some(64),
            };

            group.bench_function(&name, |b| {
                b.iter(|| {
                    let req2 = req.clone();
                    let _ = ta_app_core::run_double_ma_optimization_blocking_with_candles(
                        req2,
                        &candles,
                        &cancel,
                        None,
                    )
                    .expect("gpu sweep run");
                })
            });
        }

        group.finish();
    }
}

#[cfg(not(feature = "cuda"))]
fn bench_double_ma_gpu_sweep(_c: &mut Criterion) {}

#[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
fn bench_double_ma_gpu_kernel(c: &mut Criterion) {
    if !my_project::cuda::cuda_available() {
        return;
    }
    let candles = make_synth_candles(200_000);
    let cancel = AtomicBool::new(false);

    let ma_cases: &[MaBenchCase] = &[
        MaBenchCase {
            label: "sma_sma",
            fast_ma: "sma",
            slow_ma: "sma",
        },
        MaBenchCase {
            label: "alma_alma",
            fast_ma: "alma",
            slow_ma: "alma",
        },
    ];

    let cases: &[(&str, (u32, u32, u32), (u32, u32, u32), Option<&'static str>)] = &[
        (
            "small",
            (5, 50, 1),
            (20, 200, 5),
            None,
        ),
        (
            "medium",
            (5, 100, 1),
            (10, 200, 1),
            None,
        ),
        (
            "large",
            (5, 200, 1),
            (10, 400, 1),
            None,
        ),
        (
            "xlarge",
            (5, 500, 1),
            (10, 1000, 1),
            None,
        ),
        (
            "xlarge_batched_1gb",
            (5, 500, 1),
            (10, 1000, 1),
            Some("1024"),
        ),
        (
            "xxlarge",
            (5, 1000, 1),
            (10, 2000, 1),
            None,
        ),
        (
            "xxlarge_batched_1gb",
            (5, 1000, 1),
            (10, 2000, 1),
            Some("1024"),
        ),
        (
            "large_batched_1gb",
            (5, 200, 1),
            (10, 400, 1),
            Some("1024"),
        ),
    ];

    let prev_budget = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

    for ma in ma_cases {
        let mut group = c.benchmark_group(format!("double_ma/gpu_kernel/{}/200k", ma.label));
        group.measurement_time(Duration::from_secs(8));
        group.sample_size(if ma.fast_ma == "alma" || ma.slow_ma == "alma" { 10 } else { 20 });

        for (label, fast_range, slow_range, budget_mb) in cases.iter().copied() {
            match budget_mb {
                Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
                None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
            }

            let fast_periods = expand_range_u16(fast_range);
            let slow_periods = expand_range_u16(slow_range);
            let pairs = count_pairs_fast_lt_slow(&fast_periods, &slow_periods);
            let name = format!("{label}/{pairs}pairs");

            let fast_ma_params = if ma.fast_ma == "alma" { Some(alma_params_default()) } else { None };
            let slow_ma_params = if ma.slow_ma == "alma" { Some(alma_params_default()) } else { None };

            let req = DoubleMaRequest {
                backend: Backend::GpuKernel { device_id: 0 },
                data_id: "bench".to_string(),
                fast_range,
                slow_range,
                fast_ma_types: vec![ma.fast_ma.to_string()],
                slow_ma_types: vec![ma.slow_ma.to_string()],
                ma_source: "close".to_string(),
                export_all_csv_path: None,
                fast_ma_params,
                slow_ma_params,
                strategy: StrategyConfig::default(),
                objective: ObjectiveKind::Sharpe,
                mode: OptimizationMode::Grid,
                top_k: Some(50),
                include_all: Some(false),
                heatmap_bins: Some(64),
            };

            group.bench_function(&name, |b| {
                b.iter(|| {
                    let req2 = req.clone();
                    let _ = ta_app_core::run_double_ma_optimization_blocking_with_candles(
                        req2,
                        &candles,
                        &cancel,
                        None,
                    )
                    .expect("gpu kernel run");
                })
            });
        }

        group.finish();
    }

    match prev_budget {
        Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
        None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
    }
}

#[cfg(not(all(feature = "cuda", feature = "cuda-backtest-kernel")))]
fn bench_double_ma_gpu_kernel(_c: &mut Criterion) {}

#[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
fn bench_double_ma_gpu_kernel_extra_ma(c: &mut Criterion) {
    if !my_project::cuda::cuda_available() {
        return;
    }
    let candles = make_synth_candles(200_000);
    let cancel = AtomicBool::new(false);

    let ma_cases: &[MaBenchCase] = &[
        MaBenchCase {
            label: "jsa_jsa",
            fast_ma: "jsa",
            slow_ma: "jsa",
        },
        MaBenchCase {
            label: "vwma_vwma",
            fast_ma: "vwma",
            slow_ma: "vwma",
        },
        MaBenchCase {
            label: "vpwma_vpwma",
            fast_ma: "vpwma",
            slow_ma: "vpwma",
        },
        MaBenchCase {
            label: "frama_frama",
            fast_ma: "frama",
            slow_ma: "frama",
        },
        MaBenchCase {
            label: "nma_nma",
            fast_ma: "nma",
            slow_ma: "nma",
        },
        MaBenchCase {
            label: "cwma_cwma",
            fast_ma: "cwma",
            slow_ma: "cwma",
        },
        MaBenchCase {
            label: "cora_wave_cora_wave",
            fast_ma: "cora_wave",
            slow_ma: "cora_wave",
        },
        MaBenchCase {
            label: "hma_hma",
            fast_ma: "hma",
            slow_ma: "hma",
        },
        MaBenchCase {
            label: "epma_epma",
            fast_ma: "epma",
            slow_ma: "epma",
        },
        MaBenchCase {
            label: "jma_jma",
            fast_ma: "jma",
            slow_ma: "jma",
        },
        MaBenchCase {
            label: "edcf_edcf",
            fast_ma: "edcf",
            slow_ma: "edcf",
        },
        MaBenchCase {
            label: "supersmoother_supersmoother",
            fast_ma: "supersmoother",
            slow_ma: "supersmoother",
        },
        MaBenchCase {
            label: "zlema_zlema",
            fast_ma: "zlema",
            slow_ma: "zlema",
        },
        MaBenchCase {
            label: "ehlers_itrend_ehlers_itrend",
            fast_ma: "ehlers_itrend",
            slow_ma: "ehlers_itrend",
        },
    ];

    let cases: &[(&str, (u32, u32, u32), (u32, u32, u32), Option<&'static str>)] = &[
        (
            "small",
            (5, 50, 1),
            (20, 200, 5),
            None,
        ),
        (
            "medium",
            (5, 100, 1),
            (10, 200, 1),
            None,
        ),
        (
            "large",
            (5, 200, 1),
            (10, 400, 1),
            None,
        ),
        (
            "large_batched_1gb",
            (5, 200, 1),
            (10, 400, 1),
            Some("1024"),
        ),
    ];

    let prev_budget = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();

    for ma in ma_cases {
        let mut group = c.benchmark_group(format!("double_ma/gpu_kernel_extra/{}/200k", ma.label));
        group.measurement_time(Duration::from_secs(8));
        group.sample_size(10);

        for (label, fast_range, slow_range, budget_mb) in cases.iter().copied() {
            match budget_mb {
                Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
                None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
            }

            let fast_periods = expand_range_u16(fast_range);
            let slow_periods = expand_range_u16(slow_range);
            let pairs = count_pairs_fast_lt_slow(&fast_periods, &slow_periods);
            let name = format!("{label}/{pairs}pairs");

            let fast_ma_params = if ma.fast_ma == "epma" { Some(epma_params_default()) } else { None };
            let slow_ma_params = if ma.slow_ma == "epma" { Some(epma_params_default()) } else { None };

            let req = DoubleMaRequest {
                backend: Backend::GpuKernel { device_id: 0 },
                data_id: "bench".to_string(),
                fast_range,
                slow_range,
                fast_ma_types: vec![ma.fast_ma.to_string()],
                slow_ma_types: vec![ma.slow_ma.to_string()],
                ma_source: "close".to_string(),
                export_all_csv_path: None,
                fast_ma_params,
                slow_ma_params,
                strategy: StrategyConfig::default(),
                objective: ObjectiveKind::Sharpe,
                mode: OptimizationMode::Grid,
                top_k: Some(50),
                include_all: Some(false),
                heatmap_bins: Some(64),
            };

            group.bench_function(&name, |b| {
                b.iter(|| {
                    let req2 = req.clone();
                    let _ = ta_app_core::run_double_ma_optimization_blocking_with_candles(
                        req2,
                        &candles,
                        &cancel,
                        None,
                    )
                    .expect("gpu kernel run");
                })
            });
        }

        group.finish();
    }

    match prev_budget {
        Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
        None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
    }
}

#[cfg(not(all(feature = "cuda", feature = "cuda-backtest-kernel")))]
fn bench_double_ma_gpu_kernel_extra_ma(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_double_ma_cpu,
    bench_double_ma_gpu_sweep,
    bench_double_ma_gpu_kernel,
    bench_double_ma_gpu_kernel_extra_ma
);
criterion_main!(benches);
