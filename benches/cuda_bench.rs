#![cfg(feature = "cuda")]

extern crate vector_ta as my_project;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cust::memory::mem_get_info;
use my_project::cuda::{self, CudaBenchScenario};
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};

// Work around broken-pipe panics when piping `-- --list` output into tools like
// `Select-Object -First ...` (downstream closes stdout early).
#[cfg(not(target_arch = "wasm32"))]
#[ctor::ctor]
fn __install_broken_pipe_panic_hook() {
    use std::panic;

    let default = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| info.payload().downcast_ref::<String>().map(|s| s.as_str()))
            .unwrap_or("");

        let is_stdout_broken_pipe = msg.contains("failed printing to stdout")
            && (msg.contains("The pipe is being closed")
                || msg.contains("Broken pipe")
                || msg.contains("os error 232")
                || msg.contains("os error 32"));

        if is_stdout_broken_pipe {
            std::process::exit(0);
        }

        default(info);
    }));
}

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
    v.extend(my_project::cuda::moving_averages::apo_wrapper::benches::bench_profiles());
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
    v.extend(my_project::cuda::moving_averages::vidya_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::kama_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::mwdx_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::nama_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::reflex_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::sinwma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::swma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::smma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::cwma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::cora_wave_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::sama_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::tema_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::tilson_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::gaussian_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::highpass_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::highpass2_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::decycler_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::trendflex_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::supersmoother_wrapper::benches::bench_profiles());
    v.extend(
        my_project::cuda::moving_averages::supersmoother_3_pole_wrapper::benches::bench_profiles(),
    );
    v.extend(my_project::cuda::moving_averages::mama_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::ehlers_pma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::pma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::linreg_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::linearreg_slope_wrapper::benches::bench_profiles());
    v.extend(
        my_project::cuda::moving_averages::linearreg_intercept_wrapper::benches::bench_profiles(),
    );
    v.extend(my_project::cuda::moving_averages::tsf_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::trima_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::zlema_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::uma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::vlma_wrapper::benches::bench_profiles());
    // UI (Ulcer Index)
    v.extend(my_project::cuda::ui_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::trix_wrapper::benches::bench_profiles());
    // Oscillators / others
    v.extend(my_project::cuda::oscillators::rocp_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::rvi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::stc_wrapper::benches::bench_profiles());
    // Oscillators
    v.extend(my_project::cuda::oscillators::qqe_wrapper::benches::bench_profiles());
    // Non-MAs
    v.extend(my_project::cuda::pivot_wrapper::benches::bench_profiles());
    // OBV
    v.extend(my_project::cuda::obv_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::msw_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::macz_wrapper::benches::bench_profiles());
    // Oscillators
    v.extend(my_project::cuda::oscillators::willr_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::wilders_wrapper::benches::bench_profiles());
    // VWMACD
    v.extend(my_project::cuda::vwmacd_wrapper::benches::bench_profiles());
    // Non-MA wrappers
    v.extend(my_project::cuda::oscillators::kst_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::halftrend_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::vpci_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::nvi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::pvi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::vpt_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::supertrend_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::ttm_trend_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::ott_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::maaq_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::squeeze_momentum_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::ttm_squeeze_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::rsmk_wrapper::benches::bench_profiles());
    // Non-MA wrappers
    v.extend(my_project::cuda::linearreg_angle_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::percentile_nearest_rank_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::prb_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::mab_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::kdj_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::stochf_wrapper::benches::bench_profiles());
    // non-MA wrappers registered here
    v.extend(my_project::cuda::devstop_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::fvg_trailing_stop_wrapper::benches::bench_profiles());
    // Composite/non-MA
    v.extend(my_project::cuda::keltner_wrapper::benches::bench_profiles());
    // Non-MA wrappers
    v.extend(my_project::cuda::dvdiqqe_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::er_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::pfe_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::srwma_wrapper::benches::bench_profiles());
    // Non-MA wrappers
    v.extend(my_project::cuda::sar_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::range_filter_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::mass_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::lrsi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::kaufmanstop_wrapper::benches::bench_profiles());
    v.extend(
        my_project::cuda::moving_averages::correlation_cycle_wrapper::benches::bench_profiles(),
    );
    v.extend(my_project::cuda::moving_averages::otto_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::cg_wrapper::benches::bench_profiles());
    // Oscillators
    v.extend(my_project::cuda::oscillators::mfi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::chop_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::sqwma_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::frama_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::tradjema_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::alligator_wrapper::benches::bench_profiles());

    // Non-MA wrappers
    v.extend(my_project::cuda::di_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::zscore_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::deviation_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::bollinger_bands_width_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::medium_ad_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::stddev_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::vosc_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::qstick_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::kurtosis_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::wto_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::moving_averages::wclprice_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::medprice_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::wad_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::wavetrend::wavetrend_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::cci_cycle_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::adx_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::dx_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::avsl_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::dm_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::chandelier_exit_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::damiani_volatmeter_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::eri_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::acosc_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::aroonosc_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::cfo_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::dpo_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::fosc_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::kvo_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::ppo_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::tsi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::stoch_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::cksp_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::emd_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::minmax_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::natr_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::var_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::voss_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::aso_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::cmo_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::dti_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::emv_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::reverse_rsi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::ad_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::alphatrend_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::bollinger_bands_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::mod_god_mode_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::mean_ad_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::net_myrsi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::vi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::adosc_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::ao_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::bop_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::coppock_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::gatorosc_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::macd_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::mom_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::roc_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::rsi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::rsx_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::oscillators::srsi_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::atr_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::chande_wrapper::benches::bench_profiles());
    v.extend(my_project::cuda::cvi_wrapper::benches::bench_profiles());
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

    fn panic_message(payload: &Box<dyn Any + Send>) -> String {
        if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic payload".to_string()
        }
    }

    let vram_extra_headroom_bytes: usize = std::env::var("CUDA_BENCH_VRAM_HEADROOM_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .and_then(|mb| mb.checked_mul(1024 * 1024))
        .or_else(|| {
            std::env::var("CUDA_BENCH_VRAM_HEADROOM_BYTES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
        })
        .unwrap_or(0);
    let vram_max_free_fraction: Option<f64> = std::env::var("CUDA_BENCH_VRAM_MAX_FREE_FRACTION")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|f| f.is_finite() && *f > 0.0 && *f <= 1.0);

    let scenario_timeout_ms: u64 = std::env::var("CUDA_BENCH_SCENARIO_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .and_then(|s| s.checked_mul(1000))
        .or_else(|| {
            std::env::var("CUDA_BENCH_SCENARIO_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
        })
        .unwrap_or(0);

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
        let min_samples: usize = std::env::var("CUDA_BENCH_MIN_SAMPLES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10);
        let forced_samples: Option<usize> = std::env::var("CUDA_BENCH_SAMPLE_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let default_samples: usize = std::env::var("CUDA_BENCH_DEFAULT_SAMPLES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10);

        let n = forced_samples.or(scen.sample_size).unwrap_or(default_samples);
        let sample_size = n.max(min_samples);
        group.sample_size(sample_size);

        let group_name = scen.group;
        let bench_id = scen.bench_id;
        let indicator = scen.indicator;
        let skip_label = scen.skip_label.unwrap_or(group_name);
        let prep = scen.prep;
        let mem_required = scen.mem_required;
        let inner = scen.inner_iters.unwrap_or(1);
        let scenario_timeout_ms = scenario_timeout_ms;
        let vram_extra_headroom_bytes = vram_extra_headroom_bytes;
        let vram_max_free_fraction = vram_max_free_fraction;
        let sample_size = sample_size;

        group.bench_function(BenchmarkId::new(bench_id, indicator), move |b| {
            // Optional VRAM check (cheap, and only run for selected benches).
            if let Some(req) = mem_required {
                if let Ok((free, _total)) = mem_get_info() {
                    let mut usable = free;
                    if vram_extra_headroom_bytes > 0 {
                        usable = usable.saturating_sub(vram_extra_headroom_bytes);
                    }
                    if let Some(frac) = vram_max_free_fraction {
                        let cap = (free as f64 * frac).floor();
                        if cap.is_finite() && cap > 0.0 {
                            usable = usable.min(cap as usize);
                        }
                    }
                    if req > usable {
                        eprintln!(
                            "[cuda_bench] skipped {}/{}: insufficient VRAM (required={}B, free={}B, usable={}B, headroom={}B, cap_fraction={:?})",
                            group_name,
                            skip_label,
                            req,
                            free,
                            usable,
                            vram_extra_headroom_bytes,
                            vram_max_free_fraction
                        );
                        b.iter(|| 0);
                        return;
                    }
                }
            }

            let mut state = match catch_unwind(AssertUnwindSafe(|| prep())) {
                Ok(s) => s,
                Err(panic) => {
                    let msg = panic_message(&panic);
                    let msg_lc = msg.to_ascii_lowercase();
                    if msg_lc.contains("outofmemory") || msg_lc.contains("out of memory") {
                        eprintln!(
                            "[cuda_bench] skipped {} {}/{}: {}",
                            group_name, bench_id, indicator, msg
                        );
                        b.iter(|| 0);
                        return;
                    }
                    std::panic::resume_unwind(panic)
                }
            };

            // Run the optional active warmup outside Criterion's measurement loop, and treat any
            // CUDA/runtime panic here as a skipped scenario so the rest of the suite can proceed.
            if let Err(panic) = catch_unwind(AssertUnwindSafe(|| active_warmup(&mut *state))) {
                let msg = panic_message(&panic);
                eprintln!(
                    "[cuda_bench] skipped {} {}/{}: warmup panicked: {}",
                    group_name, bench_id, indicator, msg
                );
                b.iter(|| 0);
                return;
            }

            // Optional per-scenario timeout: if the *minimum* possible runtime (sample_size samples,
            // each requiring at least 1 iteration) would exceed the budget, skip the bench.
            //
            // Note: we cannot safely interrupt a running CUDA kernel from within the same process, so
            // this is a pre-flight guard to avoid starting obviously-too-long scenarios.
            if scenario_timeout_ms > 0 {
                let launch_time = match catch_unwind(AssertUnwindSafe(|| {
                    let t0 = Instant::now();
                    state.launch();
                    t0.elapsed()
                })) {
                    Ok(d) => d,
                    Err(panic) => {
                        let msg = panic_message(&panic);
                        eprintln!(
                            "[cuda_bench] skipped {} {}/{}: preflight launch panicked: {}",
                            group_name, bench_id, indicator, msg
                        );
                        b.iter(|| 0);
                        return;
                    }
                };

                let timeout_nanos: u128 = (scenario_timeout_ms as u128).saturating_mul(1_000_000);
                let per_launch_nanos = launch_time.as_nanos();
                let inner_nanos = per_launch_nanos.saturating_mul(inner as u128);
                // At least one warmup iteration + `sample_size` measurement iterations.
                let min_total_nanos =
                    inner_nanos.saturating_mul((sample_size as u128).saturating_add(1));

                if min_total_nanos > timeout_nanos {
                    eprintln!(
                        "[cuda_bench] skipped {} {}/{}: timeout (min {:.2}s > {:.2}s; inner={}, samples={})",
                        group_name,
                        bench_id,
                        indicator,
                        (min_total_nanos as f64) / 1e9,
                        (timeout_nanos as f64) / 1e9,
                        inner,
                        sample_size
                    );
                    b.iter(|| 0);
                    return;
                }
            }

            if inner > 1 {
                // Normalize to per-launch timing using iter_custom (divide elapsed by `inner`).
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
            } else {
                b.iter(|| state.launch())
            }
        });
        group.finish();
    }
}

criterion_group!(cuda_benches, run_registered_benches);
criterion_main!(cuda_benches);
