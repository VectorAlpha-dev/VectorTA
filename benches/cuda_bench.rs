#![cfg(feature = "cuda")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cust::memory::mem_get_info;
use my_project::cuda::{self, CudaBenchScenario};
use std::time::{Duration, Instant};

fn collect_registered_profiles() -> Vec<CudaBenchScenario> {
    let mut v = Vec::new();
    // Register wrappers that already expose bench profiles.
    v.extend(my_project::cuda::moving_averages::buff_averages_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::alma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::sma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::dema_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::dma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::fwma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::epma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::ehma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::wma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::ema_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::hma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::jma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::hwma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::jsa_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::ehlers_itrend_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::ehlers_kama_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::ehlers_ecema_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::vpwma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::vama_wrapper::benches::bench_profiles());
    v.extend(
        my_project::cuda::moving_averages::volume_adjusted_ma_wrapper::benches::bench_profiles(),
    );
    v.extend(my_project::cuda::moving_averages::vwma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::vwap_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::edcf_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::pwma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::nma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::kama_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::mwdx_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::nama_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::reflex_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::sinwma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::swma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::smma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::cwma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::sama_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::tema_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::tilson_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::gaussian_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::highpass_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::highpass2_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::trendflex_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::supersmoother_wrapper::benches::bench_profiles());
    v.extend(
        my_project::cuda::moving_averages::supersmoother_3_pole_wrapper::benches::bench_profiles(),
    );
    v.extend(my_project::cuda::moving_averages::mama_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::ehlers_pma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::pma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::linreg_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::tsf_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::trima_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::zlema_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::uma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::wilders_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::ott_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::maaq_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::srwma_wrapper::benches::bench_profiles());
    // Oscillators
    v.extend(my_project::cuda::oscillators::mfi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::chop_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::sqwma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::frama_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::tradjema_wrapper::benches::bench_profiles());

    // Non-MA wrappers
    v.extend(my_project::cuda::zscore_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::stddev_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::vosc_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::qstick_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::kurtosis_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::wto_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::wclprice::benches::bench_profiles());
    v.extend(my_project::cuda::wad_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::wavetrend::wavetrend_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::willr_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::adxr_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::aroon_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::bandpass_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::efi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::cci_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::dec_osc_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::fisher_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::ift_rsi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::ultosc_wrapper::benches::bench_profiles());
    // Correlation HL
    v.extend(my_project::cuda::correl_hl_wrapper::benches::bench_profiles());
    // Donchian
    v.extend(my_project::cuda::donchian_wrapper::benches::bench_profiles());
    // LPC
    v.extend(my_project::cuda::lpc_wrapper::benches::bench_profiles());
    // MarketEFI
    v.extend(my_project::cuda::marketefi_wrapper::benches::bench_profiles());
    // Nadaraya–Watson Envelope
    v.extend(my_project::cuda::nadaraya_watson_envelope_wrapper::benches::bench_profiles());
    // ROCR
    v.extend(my_project::cuda::rocr_wrapper::benches::bench_profiles());
    // SafeZoneStop
    v.extend(my_project::cuda::safezonestop_wrapper::benches::bench_profiles());
    v
}

fn run_registered_benches(c: &mut Criterion) {
    if !cuda::cuda_available() {
        // No device; register a tiny no-op bench so Criterion runs cleanly.
        let mut group = c.benchmark_group("cuda_unavailable");
        group.bench_with_input(BenchmarkId::new("skip", "no_device"), &0, |b, _| {
            b.iter(|| 0)
        });
        group.finish();
        return;
    }

    // Optional: make kernel launches synchronous for easier host-side timing
    // If the user has set CUDA_LAUNCH_BLOCKING externally, respect it. Otherwise,
    // leave it unset; benches already synchronize after each launch.
    // std::env::set_var("CUDA_LAUNCH_BLOCKING", "1");

    // Helper: active warm-up to stabilize clocks (GPU boost) and caches.
    // Default 1500 ms; overridable via CUDA_BENCH_WARMUP_MS.
    fn active_warmup<S: cuda::CudaBenchState + ?Sized>(state: &mut S) {
        let warm_ms: u64 = std::env::var("CUDA_BENCH_WARMUP_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1500);
        if warm_ms == 0 {
            return;
        }
        let t0 = Instant::now();
        while t0.elapsed().as_millis() < warm_ms as u128 {
            state.launch();
        }
    }

    for scen in collect_registered_profiles() {
        let mut group = c.benchmark_group(scen.group);
        // Group-level timing knobs (overridable via env for experimentation)
        let g_warm_ms: u64 = std::env::var("CUDA_BENCH_GROUP_WARMUP_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(300);
        let g_meas_ms: u64 = std::env::var("CUDA_BENCH_GROUP_MEASURE_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1200);
        group.warm_up_time(Duration::from_millis(g_warm_ms));
        group.measurement_time(Duration::from_millis(g_meas_ms));
        if let Some(n) = scen.sample_size {
            let n = n.max(10);
            group.sample_size(n);
        }
        // Optional VRAM check
        if let Some(req) = scen.mem_required {
            if let Ok((free, _total)) = mem_get_info() {
                if req > free {
                    let id = scen.skip_label.unwrap_or_else(|| scen.group).to_string();
                    group.bench_with_input(
                        BenchmarkId::new("skipped_insufficient_vram", id),
                        &0,
                        |b, _| b.iter(|| 0),
                    );
                    group.finish();
                    continue;
                }
            }
        }
        let prep = scen.prep;
        let mut state = prep();
        // Pre-warm: run for a short, configurable time to stabilize clocks.
        active_warmup(&mut *state);
        let inner = scen.inner_iters.unwrap_or(1);
        if inner > 1 {
            // Normalize to per-launch timing using iter_custom (divide elapsed by `inner`).
            group.bench_function(BenchmarkId::new(scen.bench_id, scen.indicator), |b| {
                b.iter_custom(|iters| {
                    let total = iters.saturating_mul(inner as u64);
                    let start = Instant::now();
                    for _ in 0..total {
                        state.launch();
                    }
                    let elapsed = start.elapsed();
                    // Return average per-iteration time, which we scale down by `inner`
                    // so Criterion reports time per single kernel launch.
                    let nanos = elapsed.as_nanos() / (inner as u128).max(1);
                    Duration::from_nanos(nanos as u64)
                })
            });
        } else {
            group.bench_function(BenchmarkId::new(scen.bench_id, scen.indicator), |b| {
                b.iter(|| state.launch())
            });
        }
        group.finish();
    }
}

criterion_group!(cuda_benches, run_registered_benches);
criterion_main!(cuda_benches);
