use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use ta_app_core::{Backend, DoubleMaRequest};
use ta_optimizer::{ObjectiveKind, OptimizationMode};
use ta_strategies::double_ma::StrategyConfig;

use my_project::utilities::data_loader::Candles;

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

#[cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]
fn bench_gpu_kernel_metrics_toggle(c: &mut Criterion) {
    if !my_project::cuda::cuda_available() {
        return;
    }

    let candles = make_synth_candles(200_000);
    let cancel = AtomicBool::new(false);

    let prev_budget = std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB").ok();
    let prev_metrics = std::env::var("VECTORBT_KERNEL_METRICS_COUNT").ok();
    let prev_skip = std::env::var("VECTORBT_KERNEL_SKIP_RECOMPUTE").ok();
    std::env::set_var("VECTORBT_KERNEL_SKIP_RECOMPUTE", "1");

    let ma_label = "alma_alma";
    let fast_ma = "alma";
    let slow_ma = "alma";

    let mut alma_p: HashMap<String, f64> = HashMap::new();
    alma_p.insert("offset".to_string(), 0.85);
    alma_p.insert("sigma".to_string(), 6.0);

    let cases: &[(&str, (u32, u32, u32), (u32, u32, u32), Option<&'static str>)] = &[
        ("small", (5, 50, 1), (20, 200, 5), None),
        ("medium", (5, 100, 1), (10, 200, 1), None),
        ("large", (5, 200, 1), (10, 400, 1), None),
        ("large_batched_1gb", (5, 200, 1), (10, 400, 1), Some("1024")),
    ];

    let metrics_variants: &[(usize, &str)] = &[(1, "m1"), (3, "m3"), (5, "m5"), (7, "m7")];

    let mut group = c.benchmark_group(format!("double_ma/gpu_kernel_metrics/{ma_label}/200k"));
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(10);

    for (case_label, fast_range, slow_range, budget_mb) in cases.iter().copied() {
        match budget_mb {
            Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
            None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
        }

        let fast_periods = expand_range_u16(fast_range);
        let slow_periods = expand_range_u16(slow_range);
        let pairs = count_pairs_fast_lt_slow(&fast_periods, &slow_periods);

        for (m, m_label) in metrics_variants.iter().copied() {
            std::env::set_var("VECTORBT_KERNEL_METRICS_COUNT", m.to_string());

            let req = DoubleMaRequest {
                backend: Backend::GpuKernel { device_id: 0 },
                data_id: "bench".to_string(),
                fast_range,
                slow_range,
                fast_ma_types: vec![fast_ma.to_string()],
                slow_ma_types: vec![slow_ma.to_string()],
                ma_source: "close".to_string(),
                export_all_csv_path: None,
                fast_ma_params: Some(alma_p.clone()),
                slow_ma_params: Some(alma_p.clone()),
                strategy: StrategyConfig::default(),
                objective: ObjectiveKind::Pnl,
                mode: OptimizationMode::Grid,
                top_k: Some(50),
                include_all: Some(false),
                heatmap_bins: Some(64),
            };

            let name = format!("{case_label}/{pairs}pairs/{m_label}");
            group.bench_function(&name, |b| {
                b.iter(|| {
                    let _ = ta_app_core::run_double_ma_optimization_blocking_with_candles(
                        req.clone(),
                        &candles,
                        &cancel,
                        None,
                    )
                    .expect("gpu kernel run");
                })
            });
        }
    }

    match prev_budget {
        Some(v) => std::env::set_var("VECTORBT_KERNEL_VRAM_BUDGET_MB", v),
        None => std::env::remove_var("VECTORBT_KERNEL_VRAM_BUDGET_MB"),
    }
    match prev_metrics {
        Some(v) => std::env::set_var("VECTORBT_KERNEL_METRICS_COUNT", v),
        None => std::env::remove_var("VECTORBT_KERNEL_METRICS_COUNT"),
    }
    match prev_skip {
        Some(v) => std::env::set_var("VECTORBT_KERNEL_SKIP_RECOMPUTE", v),
        None => std::env::remove_var("VECTORBT_KERNEL_SKIP_RECOMPUTE"),
    }
}

#[cfg(not(all(feature = "cuda", feature = "cuda-backtest-kernel")))]
fn bench_gpu_kernel_metrics_toggle(_c: &mut Criterion) {}

criterion_group!(benches, bench_gpu_kernel_metrics_toggle);
criterion_main!(benches);
