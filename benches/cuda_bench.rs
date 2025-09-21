#![cfg(feature = "cuda")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cust::memory::DeviceBuffer;

use my_project::cuda::cuda_available;
use my_project::cuda::moving_averages::{
    CudaAlma, CudaDma, CudaEhlersPma, CudaGaussian, CudaJma, CudaMama, CudaReflex, CudaSqwma,
    CudaTema, CudaUma, CudaVwma,
};
use my_project::cuda::CudaWclprice;
use my_project::indicators::moving_averages::alma::{AlmaBatchRange, AlmaParams};
use my_project::indicators::moving_averages::dma::DmaBatchRange;
use my_project::indicators::moving_averages::ehlers_pma::EhlersPmaBatchRange;
use my_project::indicators::moving_averages::gaussian::{GaussianBatchRange, GaussianParams};
use my_project::indicators::moving_averages::jma::{expand_grid_jma, JmaBatchRange, JmaParams};
use my_project::indicators::moving_averages::mama::MamaBatchRange;
use my_project::indicators::moving_averages::reflex::ReflexBatchRange;
use my_project::indicators::moving_averages::sqwma::SqwmaBatchRange;
use my_project::indicators::moving_averages::uma::{expand_grid_uma, UmaBatchRange, UmaParams};
use my_project::indicators::moving_averages::vwma::VwmaBatchRange;

// A light macro approach: define a bench by providing a `prep` function that
// returns state, and a `launch` function that runs one kernel invocation using
// that state. This keeps the harness reusable across CUDA indicators.

macro_rules! cuda_bench {
    ( $group_name:literal, $prep_fn:ident, $launch_fn:ident ) => {
        fn $prep_fn() -> impl Sized {
            $prep_fn()
        }
        fn bench_impl(c: &mut Criterion) {
            if !cuda_available() {
                eprintln!("[bench] skipping CUDA (no device)");
                return;
            }
            let mut group = c.benchmark_group($group_name);
            let mut state = $prep_fn();
            group.bench_with_input(BenchmarkId::from_parameter("default"), &0, |b, _| {
                b.iter(|| {
                    $launch_fn(&mut state);
                    black_box(0)
                })
            });
            group.finish();
        }
        bench_impl
    };
}

fn gen_series(len: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        v[i] = (x * 0.001).sin() + 0.0001 * x;
    }
    v
}

fn gen_time_major_f32(num_series: usize, series_len: usize) -> Vec<f32> {
    let mut v = vec![f32::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in j..series_len {
            let x = (t as f32) + (j as f32) * 0.1;
            v[t * num_series + j] = (x * 0.003).cos() + 0.001 * x;
        }
    }
    v
}

fn compute_weights_cpu_f32(period: usize, offset: f64, sigma: f64) -> (Vec<f32>, f32) {
    let m = (offset * (period as f64 - 1.0)) as f32;
    let s = (period as f64 / sigma) as f32;
    let s2 = 2.0f32 * s * s;
    let mut w = vec![0.0f32; period];
    let mut norm = 0.0f32;
    for i in 0..period {
        let diff = i as f32 - m;
        let wi = (-(diff * diff) / s2).exp();
        w[i] = wi;
        norm += wi;
    }
    (w, 1.0f32 / norm)
}

fn compute_jma_consts(period: usize, phase: f64, power: u32) -> (f32, f32, f32) {
    let phase_ratio = if phase < -100.0 {
        0.5
    } else if phase > 100.0 {
        2.5
    } else {
        phase / 100.0 + 1.5
    };
    let numerator = 0.45 * (period as f64 - 1.0);
    let denominator = numerator + 2.0;
    let beta = numerator / denominator;
    let alpha = beta.powi(power as i32);
    let one_minus_beta = 1.0 - beta;
    (alpha as f32, one_minus_beta as f32, phase_ratio as f32)
}

#[inline]
fn device_free_bytes() -> Option<usize> {
    unsafe {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let _ = cust::init(cust::prelude::CudaFlags::empty());
        let res = cust::sys::cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize);
        if res == cust::sys::CUresult::CUDA_SUCCESS {
            Some(free)
        } else {
            None
        }
    }
}

// ──────────────────────────────────────────────────────────────
// WCLPRICE: single series (device-resident)
// ──────────────────────────────────────────────────────────────

struct WclpriceState {
    cuda: CudaWclprice,
    d_high: DeviceBuffer<f32>,
    d_low: DeviceBuffer<f32>,
    d_close: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    len: usize,
    first_valid: usize,
}

fn prep_wclprice_state() -> WclpriceState {
    let cuda = CudaWclprice::new(0).expect("cuda wclprice");
    let len = 1_000_000usize;
    let first_valid = 8usize;

    let mut high = vec![f32::NAN; len];
    let mut low = vec![f32::NAN; len];
    let mut close = vec![f32::NAN; len];

    for i in first_valid..len {
        let x = i as f32;
        let base = (x * 0.001).sin() + 0.0001 * x;
        close[i] = base;
        high[i] = base + 0.6f32;
        low[i] = base - 0.6f32;
    }

    let d_high = DeviceBuffer::from_slice(&high).expect("copy high");
    let d_low = DeviceBuffer::from_slice(&low).expect("copy low");
    let d_close = DeviceBuffer::from_slice(&close).expect("copy close");
    let d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(len) }.expect("alloc out");

    WclpriceState {
        cuda,
        d_high,
        d_low,
        d_close,
        d_out,
        len,
        first_valid,
    }
}

fn launch_wclprice(state: &mut WclpriceState) {
    state
        .cuda
        .wclprice_device(
            &state.d_high,
            &state.d_low,
            &state.d_close,
            state.len,
            state.first_valid,
            &mut state.d_out,
        )
        .expect("launch wclprice kernel");
}

fn wclprice_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let len = 1_000_000usize;
    let bytes_each = len * std::mem::size_of::<f32>();
    let approx = bytes_each * 4 + 32 * 1024 * 1024; // high + low + close + out + headroom
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip WCLPRICE (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("wclprice_cuda_single_series");
    group.sample_size(10);
    let mut state = prep_wclprice_state();
    group.bench_function("1m_points", |b| {
        b.iter(|| {
            launch_wclprice(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// ALMA: one series × many params (device buffers reused)
// ──────────────────────────────────────────────────────────────
struct AlmaOneSeriesState {
    cuda: CudaAlma,
    d_prices: DeviceBuffer<f32>,
    d_weights: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_inv_norms: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    max_period: i32,
    series_len: i32,
    n_combos: i32,
    first_valid: i32,
}

fn prep_alma_one_series_many_params() -> AlmaOneSeriesState {
    let cuda = CudaAlma::new(0).expect("cuda alma");
    let series_len = 50_000usize;
    let data = gen_series(series_len);
    let sweep = AlmaBatchRange {
        period: (9, 240, 12),
        offset: (0.05, 0.95, 0.10),
        sigma: (1.5, 11.0, 0.5),
    };
    let mut combos: Vec<(usize, f64, f64)> = Vec::new();
    for p in (sweep.period.0..=sweep.period.1).step_by(sweep.period.2) {
        for oi in 0..10 {
            let o = sweep.offset.0 + oi as f64 * sweep.offset.2;
            for si in 0..20 {
                let s = sweep.sigma.0 + si as f64 * sweep.sigma.2;
                combos.push((p, o, s));
            }
        }
    }
    let max_period = combos.iter().map(|(p, _, _)| *p).max().unwrap();
    let n_combos = combos.len();
    let host_prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let mut weights_flat = vec![0f32; n_combos * max_period];
    let mut inv_norms = vec![0f32; n_combos];
    let mut periods_i32 = vec![0i32; n_combos];
    for (i, (p, o, s)) in combos.iter().enumerate() {
        let (w, inv) = compute_weights_cpu_f32(*p, *o, *s);
        periods_i32[i] = *p as i32;
        inv_norms[i] = inv;
        let base = i * max_period;
        weights_flat[base..base + *p].copy_from_slice(&w);
    }
    let first_valid = host_prices_f32
        .iter()
        .position(|v| !v.is_nan())
        .unwrap_or(0) as i32;
    let d_prices = DeviceBuffer::from_slice(&host_prices_f32).unwrap();
    let d_weights = DeviceBuffer::from_slice(&weights_flat).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_inv_norms = DeviceBuffer::from_slice(&inv_norms).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.unwrap();
    AlmaOneSeriesState {
        cuda,
        d_prices,
        d_weights,
        d_periods,
        d_inv_norms,
        d_out,
        max_period: max_period as i32,
        series_len: series_len as i32,
        n_combos: n_combos as i32,
        first_valid,
    }
}

fn launch_alma_one_series_many_params(state: &mut AlmaOneSeriesState) {
    state
        .cuda
        .alma_batch_device(
            &state.d_prices,
            &state.d_weights,
            &state.d_periods,
            &state.d_inv_norms,
            state.max_period,
            state.series_len,
            state.n_combos,
            state.first_valid,
            &mut state.d_out,
        )
        .unwrap();
}

// ──────────────────────────────────────────────────────────────
// ALMA: many series × one param (time-major)
// ──────────────────────────────────────────────────────────────
struct AlmaManySeriesState {
    cuda: CudaAlma,
    d_prices_tm: DeviceBuffer<f32>,
    d_weights: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    inv: f32,
    num_series: i32,
    series_len: i32,
}

fn prep_alma_many_series_one_param() -> AlmaManySeriesState {
    let cuda = CudaAlma::new(0).expect("cuda alma");
    let num_series = 4000usize;
    let series_len = 50_000usize;
    let host_tm_f32 = gen_time_major_f32(num_series, series_len);
    let params = AlmaParams {
        period: Some(14),
        offset: Some(0.85),
        sigma: Some(6.0),
    };
    let (w, inv) = compute_weights_cpu_f32(
        params.period.unwrap(),
        params.offset.unwrap(),
        params.sigma.unwrap(),
    );
    let mut first_valids = vec![0i32; num_series];
    for j in 0..num_series {
        first_valids[j] = (0..series_len)
            .position(|t| !host_tm_f32[t * num_series + j].is_nan())
            .unwrap_or(0) as i32;
    }
    let d_prices_tm = DeviceBuffer::from_slice(&host_tm_f32).unwrap();
    let d_weights = DeviceBuffer::from_slice(&w).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();
    AlmaManySeriesState {
        cuda,
        d_prices_tm,
        d_weights,
        d_first_valids,
        d_out_tm,
        period: params.period.unwrap() as i32,
        inv,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_alma_many_series_one_param(state: &mut AlmaManySeriesState) {
    state
        .cuda
        .alma_multi_series_one_param_device(
            &state.d_prices_tm,
            &state.d_weights,
            state.period,
            state.inv,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .unwrap();
}

// Instantiate benches using the small macro wrapper
fn alma_one_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("alma_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_alma_one_series_many_params();
    group.bench_with_input(BenchmarkId::from_parameter("50k_x_4kparams"), &0, |b, _| {
        b.iter(|| {
            launch_alma_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// New: 1,000,000 samples × 240 params (period sweep only)
fn prep_alma_one_series_1m_x_240() -> AlmaOneSeriesState {
    let cuda = CudaAlma::new(0).expect("cuda alma");
    let series_len = 1_000_000usize;
    let data = gen_series(series_len);
    let host_prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let periods: Vec<usize> = (1..=240).collect();
    let n_combos = periods.len();
    let max_period = *periods.iter().max().unwrap();
    let mut weights_flat = vec![0f32; n_combos * max_period];
    let mut inv_norms = vec![0f32; n_combos];
    let mut periods_i32 = vec![0i32; n_combos];
    for (i, &p) in periods.iter().enumerate() {
        let (w, inv) = compute_weights_cpu_f32(p, 0.85, 6.0);
        periods_i32[i] = p as i32;
        inv_norms[i] = inv;
        let base = i * max_period;
        weights_flat[base..base + p].copy_from_slice(&w);
    }
    let first_valid = host_prices_f32
        .iter()
        .position(|v| !v.is_nan())
        .unwrap_or(0) as i32;
    let d_prices = DeviceBuffer::from_slice(&host_prices_f32).unwrap();
    let d_weights = DeviceBuffer::from_slice(&weights_flat).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_inv_norms = DeviceBuffer::from_slice(&inv_norms).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.unwrap();
    AlmaOneSeriesState {
        cuda,
        d_prices,
        d_weights,
        d_periods,
        d_inv_norms,
        d_out,
        max_period: max_period as i32,
        series_len: series_len as i32,
        n_combos: n_combos as i32,
        first_valid,
    }
}

fn alma_one_series_bench_1m_240(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    // VRAM guard
    let series_len = 1_000_000usize;
    let n_combos = 240usize;
    let max_period = 240usize;
    let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
    let weights_bytes = n_combos * max_period * std::mem::size_of::<f32>();
    let input_bytes = series_len * std::mem::size_of::<f32>();
    let approx = out_bytes + weights_bytes + input_bytes + 64 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip 1M x 240 (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }
    let mut group = c.benchmark_group("alma_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_alma_one_series_1m_x_240();
    group.bench_with_input(BenchmarkId::from_parameter("1m_x_240params"), &0, |b, _| {
        b.iter(|| {
            launch_alma_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// New: 250k samples × ~4k params (period×offset×sigma ~ 3840 combos)
fn prep_alma_one_series_250k_x_4k() -> AlmaOneSeriesState {
    let cuda = CudaAlma::new(0).expect("cuda alma");
    let series_len = 250_000usize;
    let data = gen_series(series_len);
    let host_prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let periods: Vec<usize> = (1..=240).collect();
    let offsets: [f64; 4] = [0.25, 0.5, 0.75, 0.85];
    let sigmas: [f64; 4] = [3.0, 6.0, 9.0, 12.0];
    let mut combos: Vec<(usize, f64, f64)> =
        Vec::with_capacity(periods.len() * offsets.len() * sigmas.len());
    for &p in &periods {
        for &o in &offsets {
            for &s in &sigmas {
                combos.push((p, o, s));
            }
        }
    }
    let max_period = periods.iter().copied().max().unwrap();
    let n_combos = combos.len();
    let mut weights_flat = vec![0f32; n_combos * max_period];
    let mut inv_norms = vec![0f32; n_combos];
    let mut periods_i32 = vec![0i32; n_combos];
    for (i, (p, o, s)) in combos.iter().enumerate() {
        let (w, inv) = compute_weights_cpu_f32(*p, *o, *s);
        periods_i32[i] = *p as i32;
        inv_norms[i] = inv;
        let base = i * max_period;
        weights_flat[base..base + *p].copy_from_slice(&w);
    }
    let first_valid = host_prices_f32
        .iter()
        .position(|v| !v.is_nan())
        .unwrap_or(0) as i32;
    let d_prices = DeviceBuffer::from_slice(&host_prices_f32).unwrap();
    let d_weights = DeviceBuffer::from_slice(&weights_flat).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_inv_norms = DeviceBuffer::from_slice(&inv_norms).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.unwrap();
    AlmaOneSeriesState {
        cuda,
        d_prices,
        d_weights,
        d_periods,
        d_inv_norms,
        d_out,
        max_period: max_period as i32,
        series_len: series_len as i32,
        n_combos: n_combos as i32,
        first_valid,
    }
}

fn alma_one_series_bench_250k_4k(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    // VRAM guard
    let series_len = 250_000usize;
    let n_combos = 240usize * 4usize * 4usize; // ~3840
    let max_period = 240usize;
    let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
    let weights_bytes = n_combos * max_period * std::mem::size_of::<f32>();
    let input_bytes = series_len * std::mem::size_of::<f32>();
    let approx = out_bytes + weights_bytes + input_bytes + 128 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip 250k x ~4k (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }
    let mut group = c.benchmark_group("alma_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_alma_one_series_250k_x_4k();
    group.bench_with_input(
        BenchmarkId::from_parameter("250k_x_~4kparams"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_alma_one_series_many_params(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}
fn alma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("alma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_alma_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("4000series_x_50k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_alma_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// keep one unified group + main at the bottom (below multi-stream registration)

// ──────────────────────────────────────────────────────────────
// DMA: one series × many params (device-resident batching)
// ──────────────────────────────────────────────────────────────

fn axis_values((start, end, step): (usize, usize, usize)) -> Vec<usize> {
    if step == 0 || start == end {
        vec![start]
    } else {
        (start..=end).step_by(step).collect()
    }
}

struct DmaManySeriesState {
    cuda: CudaDma,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    hull_length: i32,
    ema_length: i32,
    ema_gain_limit: i32,
    hull_type: i32,
    series_len: usize,
    num_series: usize,
}

fn prep_dma_many_series_state() -> DmaManySeriesState {
    let cuda = CudaDma::new(0).expect("cuda dma");
    let num_series = 2048usize;
    let series_len = 20_000usize;
    let host_tm_f32 = gen_time_major_f32(num_series, series_len);
    let mut first_valids = vec![0i32; num_series];
    for j in 0..num_series {
        let mut fv = 0i32;
        for t in 0..series_len {
            if !host_tm_f32[t * num_series + j].is_nan() {
                fv = t as i32;
                break;
            }
        }
        first_valids[j] = fv;
    }
    let d_prices_tm = DeviceBuffer::from_slice(&host_tm_f32).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    DmaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        hull_length: 21,
        ema_length: 34,
        ema_gain_limit: 30,
        hull_type: 0,
        series_len,
        num_series,
    }
}

fn launch_dma_many_series(state: &mut DmaManySeriesState) {
    state
        .cuda
        .dma_many_series_one_param_device(
            &state.d_prices_tm,
            state.hull_length,
            state.ema_length,
            state.ema_gain_limit,
            state.hull_type,
            state.series_len,
            state.num_series,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .unwrap();
}

struct DmaBatchState {
    cuda: CudaDma,
    d_prices: DeviceBuffer<f32>,
    d_hulls: DeviceBuffer<i32>,
    d_emas: DeviceBuffer<i32>,
    d_gain_limits: DeviceBuffer<i32>,
    d_types: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    first_valid: usize,
    max_sqrt_len: usize,
}

fn prep_dma_batch_state() -> DmaBatchState {
    let cuda = CudaDma::new(0).expect("cuda dma");
    let series_len = 50_000usize;
    let data = gen_series(series_len);
    let range = DmaBatchRange {
        hull_length: (7, 63, 7),
        ema_length: (14, 42, 7),
        ema_gain_limit: (10, 40, 5),
        hull_ma_type: "WMA".to_string(),
    };

    let hull_vals = axis_values(range.hull_length);
    let ema_vals = axis_values(range.ema_length);
    let gain_vals = axis_values(range.ema_gain_limit);
    let hull_type_tag = if range.hull_ma_type.eq_ignore_ascii_case("EMA") {
        1
    } else {
        0
    };

    let mut hull_lengths = Vec::new();
    let mut ema_lengths = Vec::new();
    let mut gain_limits = Vec::new();
    let mut types = Vec::new();
    let mut max_sqrt_len = 1usize;

    for &h in &hull_vals {
        for &e in &ema_vals {
            for &g in &gain_vals {
                hull_lengths.push(h as i32);
                ema_lengths.push(e as i32);
                gain_limits.push(g as i32);
                types.push(hull_type_tag);
                let sqrt_len = ((h as f64).sqrt().round()) as usize;
                max_sqrt_len = max_sqrt_len.max(sqrt_len.max(1));
            }
        }
    }

    let n_combos = hull_lengths.len();
    assert!(n_combos > 0, "DMA benchmark requires combos");

    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap_or(0);

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let d_prices = DeviceBuffer::from_slice(&data_f32).unwrap();
    let d_hulls = DeviceBuffer::from_slice(&hull_lengths).unwrap();
    let d_emas = DeviceBuffer::from_slice(&ema_lengths).unwrap();
    let d_gain_limits = DeviceBuffer::from_slice(&gain_limits).unwrap();
    let d_types = DeviceBuffer::from_slice(&types).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.unwrap();

    DmaBatchState {
        cuda,
        d_prices,
        d_hulls,
        d_emas,
        d_gain_limits,
        d_types,
        d_out,
        series_len,
        n_combos,
        first_valid,
        max_sqrt_len,
    }
}

fn launch_dma_batch(state: &mut DmaBatchState) {
    state
        .cuda
        .dma_batch_device(
            &state.d_prices,
            &state.d_hulls,
            &state.d_emas,
            &state.d_gain_limits,
            &state.d_types,
            state.series_len,
            state.n_combos,
            state.first_valid,
            state.max_sqrt_len,
            &mut state.d_out,
        )
        .unwrap();
}

fn dma_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let mut group = c.benchmark_group("dma_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_dma_batch_state();
    group.bench_function("default", |b| {
        b.iter(|| {
            launch_dma_batch(&mut state);
            black_box(())
        });
    });
    group.finish();
}

fn dma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 2048usize;
    let series_len = 20_000usize;
    let bytes_prices = num_series * series_len * std::mem::size_of::<f32>();
    let bytes_out = bytes_prices;
    let bytes_first_valids = num_series * std::mem::size_of::<i32>();
    let approx = bytes_prices + bytes_out + bytes_first_valids + 64 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip DMA many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("dma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_dma_many_series_state();
    group.bench_function("time_major_default", |b| {
        b.iter(|| {
            launch_dma_many_series(&mut state);
            black_box(())
        })
    });
    group.finish();
}

const GAUSSIAN_COEFF_STRIDE: usize = 5;

struct GaussianBatchState {
    cuda: CudaGaussian,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_poles: DeviceBuffer<i32>,
    d_coeffs: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    first_valid: usize,
}

fn prep_gaussian_batch_state() -> GaussianBatchState {
    let cuda = CudaGaussian::new(0).expect("cuda gaussian");

    let series_len = 1_000_000usize;
    let first_valid = 12usize;
    let mut prices = vec![f32::NAN; series_len];
    for i in first_valid..series_len {
        let x = i as f32;
        prices[i] = (x * 0.0013).sin() + 0.00028 * x;
    }

    let sweep = GaussianBatchRange {
        period: (8, 64, 8),
        poles: (1, 4, 1),
    };
    let combos = expand_grid_gaussian(&sweep);
    assert!(
        !combos.is_empty(),
        "Gaussian sweep produced no parameter combinations"
    );

    let mut periods = Vec::with_capacity(combos.len());
    let mut poles = Vec::with_capacity(combos.len());
    let mut coeffs = Vec::with_capacity(combos.len() * GAUSSIAN_COEFF_STRIDE);
    for prm in &combos {
        let period = prm.period.unwrap();
        let pole = prm.poles.unwrap();
        let coeff = gaussian_coeffs(period, pole);
        periods.push(period as i32);
        poles.push(pole as i32);
        coeffs.extend_from_slice(&coeff);
    }

    let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
    let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
    let d_poles = DeviceBuffer::from_slice(&poles).expect("d_poles");
    let d_coeffs = DeviceBuffer::from_slice(&coeffs).expect("d_coeffs");
    let mut d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * combos.len()).expect("d_out") };

    GaussianBatchState {
        cuda,
        d_prices,
        d_periods,
        d_poles,
        d_coeffs,
        d_out,
        series_len,
        n_combos: combos.len(),
        first_valid,
    }
}

fn launch_gaussian_batch(state: &mut GaussianBatchState) {
    state
        .cuda
        .gaussian_batch_device(
            &state.d_prices,
            &state.d_periods,
            &state.d_poles,
            &state.d_coeffs,
            state.series_len,
            state.n_combos,
            state.first_valid,
            &mut state.d_out,
        )
        .expect("launch gaussian batch");
}

fn gaussian_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let sweep = GaussianBatchRange {
        period: (8, 64, 8),
        poles: (1, 4, 1),
    };
    let combos = expand_grid_gaussian(&sweep);
    let series_len = 1_000_000usize;
    let price_bytes = series_len * std::mem::size_of::<f32>();
    let param_bytes = combos.len()
        * (2 * std::mem::size_of::<i32>() + GAUSSIAN_COEFF_STRIDE * std::mem::size_of::<f32>());
    let out_bytes = combos.len() * series_len * std::mem::size_of::<f32>();
    let approx = price_bytes + param_bytes + out_bytes + 64 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip Gaussian batch (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("gaussian_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_gaussian_batch_state();
    group.bench_function("default", |b| {
        b.iter(|| {
            launch_gaussian_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct GaussianManySeriesState {
    cuda: CudaGaussian,
    d_prices_tm: DeviceBuffer<f32>,
    d_coeffs: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: usize,
    poles: usize,
    num_series: usize,
    series_len: usize,
}

fn prep_gaussian_many_series_state(
    num_series: usize,
    series_len: usize,
    params: &GaussianParams,
) -> GaussianManySeriesState {
    let cuda = CudaGaussian::new(0).expect("cuda gaussian");

    let mut prices_tm = vec![f32::NAN; num_series * series_len];
    let mut first_valids = vec![0i32; num_series];
    for series in 0..num_series {
        let first_valid = (series % 16) + 12;
        first_valids[series] = first_valid as i32;
        for t in first_valid..series_len {
            let idx = t * num_series + series;
            let x = (t as f32) + (series as f32) * 0.19;
            prices_tm[idx] = (x * 0.0018).sin() + 0.00026 * x;
        }
    }

    let coeffs = gaussian_coeffs(params.period.unwrap(), params.poles.unwrap());

    let d_prices_tm = DeviceBuffer::from_slice(&prices_tm).expect("d_prices_tm");
    let d_coeffs = DeviceBuffer::from_slice(&coeffs).expect("d_coeffs");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
    let mut d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(prices_tm.len()).expect("d_out_tm") };

    GaussianManySeriesState {
        cuda,
        d_prices_tm,
        d_coeffs,
        d_first_valids,
        d_out_tm,
        period: params.period.unwrap(),
        poles: params.poles.unwrap(),
        num_series,
        series_len,
    }
}

fn launch_gaussian_many_series(state: &mut GaussianManySeriesState) {
    state
        .cuda
        .gaussian_many_series_one_param_device(
            &state.d_prices_tm,
            &state.d_coeffs,
            state.period,
            state.poles,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .expect("launch gaussian many-series");
}

fn gaussian_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 1024usize;
    let series_len = 20_000usize;
    let price_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let coeff_bytes = GAUSSIAN_COEFF_STRIDE * std::mem::size_of::<f32>();
    let fv_bytes = num_series * std::mem::size_of::<i32>();
    let out_bytes = price_bytes;
    let approx = price_bytes + coeff_bytes + fv_bytes + out_bytes + 64 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip Gaussian many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let params = GaussianParams {
        period: Some(18),
        poles: Some(3),
    };
    let mut state = prep_gaussian_many_series_state(num_series, series_len, &params);

    let mut group = c.benchmark_group("gaussian_cuda_many_series_one_param");
    group.sample_size(10);
    group.bench_function("time_major_default", |b| {
        b.iter(|| {
            launch_gaussian_many_series(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct EhlersPmaBatchState {
    cuda: CudaEhlersPma,
    d_prices: DeviceBuffer<f32>,
    d_predict: DeviceBuffer<f32>,
    d_trigger: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    first_valid: usize,
}

fn prep_ehlers_pma_batch_state(
    series_len: usize,
    first_valid: usize,
    n_combos: usize,
) -> EhlersPmaBatchState {
    let cuda = CudaEhlersPma::new(0).expect("cuda ehlers_pma");

    let mut prices = vec![f32::NAN; series_len];
    for i in first_valid..series_len {
        let x = i as f32;
        prices[i] = (x * 0.0014).sin() + (x * 0.0009).cos() * 0.35 + 0.00028 * x;
    }

    let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
    let mut d_predict: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos).expect("d_predict") };
    let mut d_trigger: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos).expect("d_trigger") };

    EhlersPmaBatchState {
        cuda,
        d_prices,
        d_predict,
        d_trigger,
        series_len,
        n_combos,
        first_valid,
    }
}

fn launch_ehlers_pma_batch(state: &mut EhlersPmaBatchState) {
    state
        .cuda
        .ehlers_pma_batch_device(
            &state.d_prices,
            state.series_len,
            state.n_combos,
            state.first_valid,
            &mut state.d_predict,
            &mut state.d_trigger,
        )
        .expect("launch ehlers_pma batch");
}

fn ehlers_pma_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let series_len = 1_000_000usize;
    let first_valid = 12usize;
    let sweep = EhlersPmaBatchRange { combos: 256 };
    let n_combos = sweep.combos;
    let price_bytes = series_len * std::mem::size_of::<f32>();
    let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
    let approx = price_bytes + 2 * out_bytes + 48 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip Ehlers PMA batch (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut state = prep_ehlers_pma_batch_state(series_len, first_valid, n_combos);
    let mut group = c.benchmark_group("ehlers_pma_cuda_one_series_many_params");
    group.sample_size(10);
    group.bench_function("periodless_sweep", |b| {
        b.iter(|| {
            launch_ehlers_pma_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct EhlersPmaManySeriesState {
    cuda: CudaEhlersPma,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_predict_tm: DeviceBuffer<f32>,
    d_trigger_tm: DeviceBuffer<f32>,
    num_series: usize,
    series_len: usize,
}

fn prep_ehlers_pma_many_series_state(
    num_series: usize,
    series_len: usize,
) -> EhlersPmaManySeriesState {
    let cuda = CudaEhlersPma::new(0).expect("cuda ehlers_pma");

    let mut prices_tm = vec![f32::NAN; num_series * series_len];
    let mut first_valids = vec![0i32; num_series];
    for series in 0..num_series {
        let fv = (series % 8) as usize + 5;
        first_valids[series] = fv as i32;
        for row in fv..series_len {
            let idx = row * num_series + series;
            let x = row as f32 + (series as f32) * 0.23;
            prices_tm[idx] = (x * 0.0017).cos() + (x * 0.0012).sin() * 0.42 + 0.00019 * x;
        }
    }

    let d_prices_tm = DeviceBuffer::from_slice(&prices_tm).expect("d_prices_tm");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
    let mut d_predict_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len).expect("d_predict_tm") };
    let mut d_trigger_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len).expect("d_trigger_tm") };

    EhlersPmaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_predict_tm,
        d_trigger_tm,
        num_series,
        series_len,
    }
}

fn launch_ehlers_pma_many_series(state: &mut EhlersPmaManySeriesState) {
    state
        .cuda
        .ehlers_pma_many_series_one_param_device(
            &state.d_prices_tm,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_predict_tm,
            &mut state.d_trigger_tm,
        )
        .expect("launch ehlers_pma many-series");
}

fn ehlers_pma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 512usize;
    let series_len = 200_000usize;
    let prices_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let first_valid_bytes = num_series * std::mem::size_of::<i32>();
    let out_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let approx = prices_bytes + first_valid_bytes + 2 * out_bytes + 48 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip Ehlers PMA many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut state = prep_ehlers_pma_many_series_state(num_series, series_len);
    let mut group = c.benchmark_group("ehlers_pma_cuda_many_series_one_param");
    group.sample_size(10);
    group.bench_function("time_major_default", |b| {
        b.iter(|| {
            launch_ehlers_pma_many_series(&mut state);
            black_box(())
        })
    });
    group.finish();
}

fn gaussian_coeffs(period: usize, poles: usize) -> [f32; GAUSSIAN_COEFF_STRIDE] {
    use std::f64::consts::PI;

    let period_f = period as f64;
    let poles_f = poles as f64;
    let beta_num = 1.0 - (2.0 * PI / period_f).cos();
    let beta_den = (2.0f64).powf(1.0 / poles_f) - 1.0;
    let beta = beta_num / beta_den;
    let discr = beta * beta + 2.0 * beta;
    let alpha = -beta + discr.sqrt();
    let one = 1.0 - alpha;

    let mut coeffs = [0f32; GAUSSIAN_COEFF_STRIDE];
    match poles {
        1 => {
            coeffs[0] = alpha as f32;
            coeffs[1] = one as f32;
        }
        2 => {
            coeffs[0] = (alpha * alpha) as f32;
            coeffs[1] = (2.0 * one) as f32;
            coeffs[2] = (-(one * one)) as f32;
        }
        3 => {
            let one_sq = one * one;
            coeffs[0] = (alpha * alpha * alpha) as f32;
            coeffs[1] = (3.0 * one) as f32;
            coeffs[2] = (-3.0 * one_sq) as f32;
            coeffs[3] = (one_sq * one) as f32;
        }
        4 => {
            let one_sq = one * one;
            let one_cu = one_sq * one;
            coeffs[0] = (alpha * alpha * alpha * alpha) as f32;
            coeffs[1] = (4.0 * one) as f32;
            coeffs[2] = (-6.0 * one_sq) as f32;
            coeffs[3] = (4.0 * one_cu) as f32;
            coeffs[4] = (-(one_cu * one)) as f32;
        }
        _ => unreachable!(),
    }
    coeffs
}

fn expand_grid_gaussian(range: &GaussianBatchRange) -> Vec<GaussianParams> {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        }
    }

    let periods = axis(range.period);
    let poles = axis(range.poles);
    let mut combos = Vec::with_capacity(periods.len() * poles.len());
    for &p in &periods {
        for &k in &poles {
            combos.push(GaussianParams {
                period: Some(p),
                poles: Some(k),
            });
        }
    }
    combos
}

struct UmaBatchState {
    cuda: CudaUma,
    d_prices: DeviceBuffer<f32>,
    d_volumes: Option<DeviceBuffer<f32>>,
    d_accels: DeviceBuffer<f32>,
    d_min_lengths: DeviceBuffer<i32>,
    d_max_lengths: DeviceBuffer<i32>,
    d_smooth_lengths: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    first_valid: usize,
    has_volume: bool,
}

fn prep_uma_batch_state(
    combos: &[UmaParams],
    series_len: usize,
    first_valid: usize,
    has_volume: bool,
) -> UmaBatchState {
    let cuda = CudaUma::new(0).expect("cuda uma");

    let mut prices = vec![f32::NAN; series_len];
    let mut volumes_opt = if has_volume {
        Some(vec![f32::NAN; series_len])
    } else {
        None
    };

    for i in first_valid..series_len {
        let x = i as f32;
        prices[i] = (x * 0.0011).sin() + 0.0004 * x;
        if let Some(vols) = volumes_opt.as_mut() {
            vols[i] = 600.0 + (x * 0.0019).cos() * 150.0;
        }
    }

    let accelerators: Vec<f32> = combos
        .iter()
        .map(|c| c.accelerator.unwrap_or(1.0) as f32)
        .collect();
    let min_lengths: Vec<i32> = combos
        .iter()
        .map(|c| c.min_length.unwrap_or(5) as i32)
        .collect();
    let max_lengths: Vec<i32> = combos
        .iter()
        .map(|c| c.max_length.unwrap_or(50) as i32)
        .collect();
    let smooth_lengths: Vec<i32> = combos
        .iter()
        .map(|c| c.smooth_length.unwrap_or(4) as i32)
        .collect();

    let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
    let d_volumes = if let Some(vols) = volumes_opt {
        Some(DeviceBuffer::from_slice(&vols).expect("d_volumes"))
    } else {
        None
    };
    let d_accels = DeviceBuffer::from_slice(&accelerators).expect("d_accels");
    let d_min_lengths = DeviceBuffer::from_slice(&min_lengths).expect("d_min_lengths");
    let d_max_lengths = DeviceBuffer::from_slice(&max_lengths).expect("d_max_lengths");
    let d_smooth_lengths = DeviceBuffer::from_slice(&smooth_lengths).expect("d_smooth_lengths");
    let mut d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * combos.len()).expect("d_out") };

    UmaBatchState {
        cuda,
        d_prices,
        d_volumes,
        d_accels,
        d_min_lengths,
        d_max_lengths,
        d_smooth_lengths,
        d_out,
        series_len,
        n_combos: combos.len(),
        first_valid,
        has_volume,
    }
}

fn launch_uma_batch(state: &mut UmaBatchState) {
    let volume_ref = state.d_volumes.as_ref();
    state
        .cuda
        .uma_batch_device(
            &state.d_prices,
            volume_ref,
            &state.d_accels,
            &state.d_min_lengths,
            &state.d_max_lengths,
            &state.d_smooth_lengths,
            state.series_len,
            state.n_combos,
            state.first_valid,
            state.has_volume,
            &mut state.d_out,
        )
        .expect("launch uma batch");
}

fn uma_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let sweep = UmaBatchRange {
        accelerator: (1.0, 1.5, 0.5),
        min_length: (5, 9, 2),
        max_length: (18, 30, 4),
        smooth_length: (2, 4, 1),
    };
    let combos = expand_grid_uma(&sweep);
    if combos.is_empty() {
        eprintln!("[bench] UMA sweep produced no parameter combinations");
        return;
    }

    let series_len = 1_000_000usize;
    let first_valid = 16usize;
    let has_volume = true;
    let n_combos = combos.len();

    let price_bytes = series_len * std::mem::size_of::<f32>();
    let volume_bytes = if has_volume {
        series_len * std::mem::size_of::<f32>()
    } else {
        0
    };
    let param_bytes = n_combos * (std::mem::size_of::<f32>() + 3 * std::mem::size_of::<i32>());
    let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
    let approx = price_bytes + volume_bytes + param_bytes + 2 * out_bytes + 64 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip UMA batch (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut state = prep_uma_batch_state(&combos, series_len, first_valid, has_volume);
    let mut group = c.benchmark_group("uma_cuda_one_series_many_params");
    group.sample_size(10);
    group.bench_function("default", |b| {
        b.iter(|| {
            launch_uma_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct UmaManySeriesState {
    cuda: CudaUma,
    d_prices_tm: DeviceBuffer<f32>,
    d_volumes_tm: Option<DeviceBuffer<f32>>,
    d_first_valids: DeviceBuffer<i32>,
    d_raw_tm: DeviceBuffer<f32>,
    d_out_tm: DeviceBuffer<f32>,
    accelerator: f32,
    min_length: i32,
    max_length: i32,
    smooth_length: i32,
    num_series: usize,
    series_len: usize,
    has_volume: bool,
}

fn prep_uma_many_series_state() -> UmaManySeriesState {
    let cuda = CudaUma::new(0).expect("cuda uma");
    let num_series = 64usize;
    let series_len = 200_000usize;

    let mut prices_tm = vec![f32::NAN; num_series * series_len];
    let mut volumes_tm = vec![f32::NAN; num_series * series_len];
    let mut first_valids = vec![0i32; num_series];

    for series in 0..num_series {
        let fv = (series % 12) + 12;
        first_valids[series] = fv as i32;
        for row in fv..series_len {
            let idx = row * num_series + series;
            let x = row as f32 + (series as f32) * 0.33;
            prices_tm[idx] = (x * 0.0014).sin() + 0.00035 * x;
            volumes_tm[idx] = 250.0 + (x * 0.0011).cos() * (series as f32 + 1.0) * 6.0;
        }
    }

    let d_prices_tm = DeviceBuffer::from_slice(&prices_tm).expect("d_prices_tm");
    let d_volumes_tm = DeviceBuffer::from_slice(&volumes_tm).expect("d_volumes_tm");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
    let mut d_raw_tm =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len).expect("d_raw_tm") };
    let mut d_out_tm =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len).expect("d_out_tm") };

    UmaManySeriesState {
        cuda,
        d_prices_tm,
        d_volumes_tm: Some(d_volumes_tm),
        d_first_valids,
        d_raw_tm,
        d_out_tm,
        accelerator: 1.25,
        min_length: 8,
        max_length: 26,
        smooth_length: 3,
        num_series,
        series_len,
        has_volume: true,
    }
}

fn launch_uma_many_series(state: &mut UmaManySeriesState) {
    state
        .cuda
        .uma_many_series_one_param_device(
            &state.d_prices_tm,
            state.d_volumes_tm.as_ref(),
            state.accelerator,
            state.min_length,
            state.max_length,
            state.smooth_length,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            state.has_volume,
            &mut state.d_raw_tm,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn uma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 64usize;
    let series_len = 200_000usize;
    let prices_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let volume_bytes = prices_bytes;
    let first_valid_bytes = num_series * std::mem::size_of::<i32>();
    let out_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let approx = prices_bytes + volume_bytes + first_valid_bytes + 2 * out_bytes + 64 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip UMA many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("uma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_uma_many_series_state();
    group.bench_function("time_major_default", |b| {
        b.iter(|| {
            launch_uma_many_series(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// MAMA: two-output adaptive moving average kernels
// ──────────────────────────────────────────────────────────────

struct MamaBatchState {
    cuda: CudaMama,
    d_prices: DeviceBuffer<f32>,
    d_fast_limits: DeviceBuffer<f32>,
    d_slow_limits: DeviceBuffer<f32>,
    d_out_mama: DeviceBuffer<f32>,
    d_out_fama: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    first_valid: usize,
}

fn expand_mama_ranges(fast: (f64, f64, f64), slow: (f64, f64, f64)) -> (Vec<f32>, Vec<f32>) {
    let mut fast_vals = Vec::new();
    let mut slow_vals = Vec::new();

    let (f_start, f_end, f_step) = fast;
    let (s_start, s_end, s_step) = slow;

    if f_step.abs() < 1e-12 || (f_start - f_end).abs() < 1e-12 {
        let fast_value = f_start as f32;
        if s_step.abs() < 1e-12 || (s_start - s_end).abs() < 1e-12 {
            fast_vals.push(fast_value);
            slow_vals.push(s_start as f32);
        } else {
            let mut s = s_start;
            while s <= s_end + 1e-12 {
                fast_vals.push(fast_value);
                slow_vals.push(s as f32);
                s += s_step;
            }
        }
        return (fast_vals, slow_vals);
    }

    let mut f = f_start;
    while f <= f_end + 1e-12 {
        if s_step.abs() < 1e-12 || (s_start - s_end).abs() < 1e-12 {
            fast_vals.push(f as f32);
            slow_vals.push(s_start as f32);
        } else {
            let mut s = s_start;
            while s <= s_end + 1e-12 {
                fast_vals.push(f as f32);
                slow_vals.push(s as f32);
                s += s_step;
            }
        }
        f += f_step;
    }

    (fast_vals, slow_vals)
}

fn prep_mama_batch_state() -> MamaBatchState {
    let cuda = CudaMama::new(0).expect("cuda mama");
    let series_len = 1_000_000usize;
    let first_valid = 10usize;
    let fast_range = (0.30f64, 0.70f64, 0.05f64);
    let slow_range = (0.02f64, 0.08f64, 0.02f64);

    let mut prices = vec![f32::NAN; series_len];
    for i in first_valid..series_len {
        let x = i as f32;
        prices[i] = (x * 0.0013).sin() + 0.00042 * x + (x * 0.00057).cos() * 0.28;
    }

    let (fast_limits, slow_limits) = expand_mama_ranges(fast_range, slow_range);
    let n_combos = fast_limits.len();

    let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
    let d_fast_limits = DeviceBuffer::from_slice(&fast_limits).expect("d_fast_limits");
    let d_slow_limits = DeviceBuffer::from_slice(&slow_limits).expect("d_slow_limits");
    let mut d_out_mama: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos).expect("d_out_mama") };
    let mut d_out_fama: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos).expect("d_out_fama") };

    MamaBatchState {
        cuda,
        d_prices,
        d_fast_limits,
        d_slow_limits,
        d_out_mama,
        d_out_fama,
        series_len,
        n_combos,
        first_valid,
    }
}

fn launch_mama_batch(state: &mut MamaBatchState) {
    state
        .cuda
        .mama_batch_device(
            &state.d_prices,
            &state.d_fast_limits,
            &state.d_slow_limits,
            state.series_len,
            state.n_combos,
            state.first_valid,
            &mut state.d_out_mama,
            &mut state.d_out_fama,
        )
        .expect("launch mama batch");
}

fn mama_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let series_len = 1_000_000usize;
    let fast_range = (0.30f64, 0.70f64, 0.05f64);
    let slow_range = (0.02f64, 0.08f64, 0.02f64);
    let (fast_limits, slow_limits) = expand_mama_ranges(fast_range, slow_range);
    let n_combos = fast_limits.len();

    let prices_bytes = series_len * std::mem::size_of::<f32>();
    let fast_bytes = n_combos * std::mem::size_of::<f32>();
    let slow_bytes = fast_bytes;
    let out_bytes = series_len * n_combos * std::mem::size_of::<f32>();
    let approx = prices_bytes + fast_bytes + slow_bytes + 2 * out_bytes + 64 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip MAMA batch (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut state = prep_mama_batch_state();
    let mut group = c.benchmark_group("mama_cuda_one_series_many_params");
    group.sample_size(10);
    group.bench_function("default", |b| {
        b.iter(|| {
            launch_mama_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct MamaManySeriesState {
    cuda: CudaMama,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_mama_tm: DeviceBuffer<f32>,
    d_out_fama_tm: DeviceBuffer<f32>,
    num_series: usize,
    series_len: usize,
    fast_limit: f32,
    slow_limit: f32,
}

fn prep_mama_many_series_state() -> MamaManySeriesState {
    let cuda = CudaMama::new(0).expect("cuda mama");
    let num_series = 48usize;
    let series_len = 200_000usize;
    let fast_limit = 0.48f32;
    let slow_limit = 0.055f32;

    let mut prices_tm = vec![f32::NAN; num_series * series_len];
    let mut first_valids = vec![0i32; num_series];
    for series in 0..num_series {
        let fv = (series % 6) as usize;
        first_valids[series] = fv as i32;
        for row in fv..series_len {
            let idx = row * num_series + series;
            let x = row as f32 + (series as f32) * 0.33;
            prices_tm[idx] = (x * 0.0016).sin() + 0.00048 * x + (x * 0.00041).cos() * 0.4;
        }
    }

    let d_prices_tm = DeviceBuffer::from_slice(&prices_tm).expect("d_prices_tm");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
    let mut d_out_mama_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(prices_tm.len()).expect("d_out_mama_tm") };
    let mut d_out_fama_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(prices_tm.len()).expect("d_out_fama_tm") };

    MamaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_mama_tm,
        d_out_fama_tm,
        num_series,
        series_len,
        fast_limit,
        slow_limit,
    }
}

fn launch_mama_many_series(state: &mut MamaManySeriesState) {
    state
        .cuda
        .mama_many_series_one_param_device(
            &state.d_prices_tm,
            state.fast_limit,
            state.slow_limit,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_mama_tm,
            &mut state.d_out_fama_tm,
        )
        .expect("launch mama many-series");
}

fn mama_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 48usize;
    let series_len = 200_000usize;
    let prices_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let first_valid_bytes = num_series * std::mem::size_of::<i32>();
    let out_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let approx = prices_bytes + first_valid_bytes + 2 * out_bytes + 48 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip MAMA many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut state = prep_mama_many_series_state();
    let mut group = c.benchmark_group("mama_cuda_many_series_one_param");
    group.sample_size(10);
    group.bench_function("time_major_default", |b| {
        b.iter(|| {
            launch_mama_many_series(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// TEMA: one series × many params (sequential GPU recurrence)
// ──────────────────────────────────────────────────────────────

struct JmaBatchState {
    cuda: CudaJma,
    d_prices: DeviceBuffer<f32>,
    d_alphas: DeviceBuffer<f32>,
    d_one_minus_betas: DeviceBuffer<f32>,
    d_phase_ratios: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    first_valid: usize,
}

fn prep_jma_batch_state() -> JmaBatchState {
    let cuda = CudaJma::new(0).expect("cuda jma");

    let series_len = 1_000_000usize;
    let first_valid = 6usize;

    let mut prices = vec![f32::NAN; series_len];
    for i in first_valid..series_len {
        let x = i as f32;
        prices[i] = (x * 0.0009).sin() + 0.00025 * x;
    }

    let sweep = JmaBatchRange {
        period: (8, 64, 8),
        phase: (-40.0, 40.0, 40.0),
        power: (1, 3, 1),
    };
    let combos = expand_grid_jma(&sweep);
    assert!(
        !combos.is_empty(),
        "JMA sweep produced no parameter combinations"
    );

    let mut alphas = Vec::with_capacity(combos.len());
    let mut one_minus_betas = Vec::with_capacity(combos.len());
    let mut phase_ratios = Vec::with_capacity(combos.len());
    for prm in &combos {
        let (alpha, one_minus_beta, phase_ratio) =
            compute_jma_consts(prm.period.unwrap(), prm.phase.unwrap(), prm.power.unwrap());
        alphas.push(alpha);
        one_minus_betas.push(one_minus_beta);
        phase_ratios.push(phase_ratio);
    }

    let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
    let d_alphas = DeviceBuffer::from_slice(&alphas).expect("d_alphas");
    let d_one_minus_betas = DeviceBuffer::from_slice(&one_minus_betas).expect("d_one_minus_betas");
    let d_phase_ratios = DeviceBuffer::from_slice(&phase_ratios).expect("d_phase_ratios");
    let mut d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * combos.len()) }.expect("d_out");

    JmaBatchState {
        cuda,
        d_prices,
        d_alphas,
        d_one_minus_betas,
        d_phase_ratios,
        d_out,
        series_len,
        n_combos: combos.len(),
        first_valid,
    }
}

fn launch_jma_batch(state: &mut JmaBatchState) {
    state
        .cuda
        .jma_batch_device(
            &state.d_prices,
            &state.d_alphas,
            &state.d_one_minus_betas,
            &state.d_phase_ratios,
            state.series_len,
            state.n_combos,
            state.first_valid,
            &mut state.d_out,
        )
        .expect("launch jma batch");
}

fn jma_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let mut group = c.benchmark_group("jma_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_jma_batch_state();
    group.bench_function("default", |b| {
        b.iter(|| {
            launch_jma_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct JmaManySeriesState {
    cuda: CudaJma,
    d_prices_tm: DeviceBuffer<f32>,
    alpha: f32,
    one_minus_beta: f32,
    phase_ratio: f32,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    num_series: usize,
    series_len: usize,
}

fn prep_jma_many_series_state(
    params: &JmaParams,
    num_series: usize,
    series_len: usize,
) -> JmaManySeriesState {
    let cuda = CudaJma::new(0).expect("cuda jma");

    let mut prices_tm = vec![f32::NAN; num_series * series_len];
    let mut first_valids = vec![0i32; num_series];
    for series in 0..num_series {
        let first_valid = 6 + (series % 8);
        first_valids[series] = first_valid as i32;
        for t in first_valid..series_len {
            let idx = t * num_series + series;
            let x = (t as f32) + (series as f32) * 0.21;
            prices_tm[idx] = (x * 0.0011).sin() + 0.0003 * x;
        }
    }

    let (alpha, one_minus_beta, phase_ratio) = compute_jma_consts(
        params.period.unwrap(),
        params.phase.unwrap(),
        params.power.unwrap(),
    );

    let d_prices_tm = DeviceBuffer::from_slice(&prices_tm).expect("d_prices_tm");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
    let mut d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(prices_tm.len()) }.expect("d_out_tm");

    JmaManySeriesState {
        cuda,
        d_prices_tm,
        alpha,
        one_minus_beta,
        phase_ratio,
        d_first_valids,
        d_out_tm,
        num_series,
        series_len,
    }
}

fn launch_jma_many_series(state: &mut JmaManySeriesState) {
    state
        .cuda
        .jma_many_series_one_param_device(
            &state.d_prices_tm,
            state.alpha,
            state.one_minus_beta,
            state.phase_ratio,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .expect("launch jma many-series");
}

fn jma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 1024usize;
    let series_len = 20_000usize;
    let price_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let first_valid_bytes = num_series * std::mem::size_of::<i32>();
    let out_bytes = price_bytes;
    let approx = price_bytes + first_valid_bytes + out_bytes + 64 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip JMA many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let params = JmaParams {
        period: Some(18),
        phase: Some(25.0),
        power: Some(2),
    };
    let mut state = prep_jma_many_series_state(&params, num_series, series_len);

    let mut group = c.benchmark_group("jma_cuda_many_series_one_param");
    group.sample_size(10);
    group.bench_function("time_major_default", |b| {
        b.iter(|| {
            launch_jma_many_series(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct TemaBatchState {
    cuda: CudaTema,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    first_valid: usize,
}

fn prep_tema_batch_state() -> TemaBatchState {
    let cuda = CudaTema::new(0).expect("cuda tema");
    let series_len = 1_000_000usize;
    let first_valid = 8usize;
    let period_range = (8usize, 96usize, 4usize);

    let mut prices = vec![f32::NAN; series_len];
    for i in first_valid..series_len {
        let x = i as f32;
        prices[i] = (x * 0.0009).sin() + 0.00018 * x;
    }

    let periods: Vec<i32> = if period_range.2 == 0 || period_range.0 == period_range.1 {
        vec![period_range.0 as i32]
    } else {
        (period_range.0..=period_range.1)
            .step_by(period_range.2)
            .map(|p| p as i32)
            .collect()
    };
    let n_combos = periods.len();

    let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
    let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
    let mut d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos).expect("d_out") };

    TemaBatchState {
        cuda,
        d_prices,
        d_periods,
        d_out,
        series_len,
        n_combos,
        first_valid,
    }
}

fn launch_tema_batch(state: &mut TemaBatchState) {
    state
        .cuda
        .tema_batch_device(
            &state.d_prices,
            &state.d_periods,
            state.series_len,
            state.n_combos,
            state.first_valid,
            &mut state.d_out,
        )
        .expect("launch tema batch");
}

fn tema_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let series_len = 1_000_000usize;
    let n_combos = ((96usize - 8usize) / 4usize) + 1usize;
    let price_bytes = series_len * std::mem::size_of::<f32>();
    let period_bytes = n_combos * std::mem::size_of::<i32>();
    let out_bytes = series_len * n_combos * std::mem::size_of::<f32>();
    let approx = price_bytes + period_bytes + out_bytes + 64 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip TEMA batch (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut state = prep_tema_batch_state();
    let mut group = c.benchmark_group("tema_cuda_one_series_many_params");
    group.sample_size(10);
    group.bench_function("default", |b| {
        b.iter(|| {
            launch_tema_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct TemaManySeriesState {
    cuda: CudaTema,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    num_series: usize,
    series_len: usize,
    period: usize,
}

fn prep_tema_many_series_state() -> TemaManySeriesState {
    let cuda = CudaTema::new(0).expect("cuda tema");
    let num_series = 48usize;
    let series_len = 200_000usize;
    let period = 28usize;

    let mut prices_tm = vec![f32::NAN; num_series * series_len];
    let mut first_valids = vec![0i32; num_series];

    for series in 0..num_series {
        let fv = (series % 6) as usize;
        first_valids[series] = fv as i32;
        for row in fv..series_len {
            let idx = row * num_series + series;
            let x = row as f32 + (series as f32) * 0.21;
            prices_tm[idx] = (x * 0.0013).sin() + 0.00026 * x;
        }
    }

    let d_prices_tm = DeviceBuffer::from_slice(&prices_tm).expect("d_prices_tm");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
    let mut d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len).expect("d_out_tm") };

    TemaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        num_series,
        series_len,
        period,
    }
}

fn launch_tema_many_series(state: &mut TemaManySeriesState) {
    state
        .cuda
        .tema_many_series_one_param_device(
            &state.d_prices_tm,
            state.period,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .expect("launch tema many-series");
}

fn tema_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 48usize;
    let series_len = 200_000usize;
    let prices_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let first_valid_bytes = num_series * std::mem::size_of::<i32>();
    let out_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let approx = prices_bytes + first_valid_bytes + out_bytes + 48 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip TEMA many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut state = prep_tema_many_series_state();
    let mut group = c.benchmark_group("tema_cuda_many_series_one_param");
    group.sample_size(10);
    group.bench_function("time_major_default", |b| {
        b.iter(|| {
            launch_tema_many_series(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// VWMA: one series × many params (prefix-sharing kernel)
// ──────────────────────────────────────────────────────────────

struct SqwmaBatchState {
    cuda: CudaSqwma,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    first_valid: usize,
    max_period: usize,
}

fn prep_sqwma_batch_state() -> SqwmaBatchState {
    let cuda = CudaSqwma::new(0).expect("cuda sqwma");
    let series_len = 1_000_000usize;
    let first_valid = 6usize;
    let period_range = (6usize, 96usize, 4usize);

    let mut prices = vec![f32::NAN; series_len];
    for i in first_valid..series_len {
        let x = i as f32;
        prices[i] = (x * 0.0012).sin() + 0.00021 * x;
    }

    let periods: Vec<i32> = if period_range.2 == 0 || period_range.0 == period_range.1 {
        vec![period_range.0 as i32]
    } else {
        (period_range.0..=period_range.1)
            .step_by(period_range.2)
            .map(|p| p as i32)
            .collect()
    };
    let n_combos = periods.len();
    let max_period = periods
        .iter()
        .copied()
        .map(|p| p as usize)
        .max()
        .unwrap_or(2);

    let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
    let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
    let mut d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos).expect("d_out") };

    SqwmaBatchState {
        cuda,
        d_prices,
        d_periods,
        d_out,
        series_len,
        n_combos,
        first_valid,
        max_period,
    }
}

fn launch_sqwma_batch(state: &mut SqwmaBatchState) {
    state
        .cuda
        .sqwma_batch_device(
            &state.d_prices,
            &state.d_periods,
            state.series_len,
            state.n_combos,
            state.first_valid,
            state.max_period,
            &mut state.d_out,
        )
        .expect("launch sqwma batch");
}

fn sqwma_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let series_len = 1_000_000usize;
    let period_range = (6usize, 96usize, 4usize);
    let n_combos = if period_range.2 == 0 || period_range.0 == period_range.1 {
        1
    } else {
        ((period_range.1 - period_range.0) / period_range.2) + 1
    };
    let price_bytes = series_len * std::mem::size_of::<f32>();
    let period_bytes = n_combos * std::mem::size_of::<i32>();
    let out_bytes = series_len * n_combos * std::mem::size_of::<f32>();
    let approx = price_bytes + period_bytes + out_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip SQWMA batch (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("sqwma_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_sqwma_batch_state();
    group.bench_function("1m_x_period_sweep", |b| {
        b.iter(|| {
            launch_sqwma_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct SqwmaManySeriesState {
    cuda: CudaSqwma,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    num_series: usize,
    series_len: usize,
    period: usize,
}

fn prep_sqwma_many_series_state() -> SqwmaManySeriesState {
    let cuda = CudaSqwma::new(0).expect("cuda sqwma");
    let num_series = 48usize;
    let series_len = 200_000usize;
    let period = 24usize;

    let mut prices_tm = vec![f32::NAN; num_series * series_len];
    let mut first_valids = vec![0i32; num_series];

    for series in 0..num_series {
        let fv = (series % 8) as usize;
        first_valids[series] = fv as i32;
        for row in fv..series_len {
            let idx = row * num_series + series;
            let x = row as f32 + (series as f32) * 0.19;
            prices_tm[idx] = (x * 0.0011).cos() + 0.00024 * x;
        }
    }

    let d_prices_tm = DeviceBuffer::from_slice(&prices_tm).expect("d_prices_tm");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
    let mut d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len).expect("d_out_tm") };

    SqwmaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        num_series,
        series_len,
        period,
    }
}

fn launch_sqwma_many_series(state: &mut SqwmaManySeriesState) {
    state
        .cuda
        .sqwma_many_series_one_param_device(
            &state.d_prices_tm,
            state.period,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .expect("launch sqwma many-series");
}

fn sqwma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 48usize;
    let series_len = 200_000usize;
    let prices_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let first_valid_bytes = num_series * std::mem::size_of::<i32>();
    let out_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let approx = prices_bytes + first_valid_bytes + out_bytes + 48 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip SQWMA many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("sqwma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_sqwma_many_series_state();
    group.bench_function("time_major_default", |b| {
        b.iter(|| {
            launch_sqwma_many_series(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct ReflexBatchState {
    cuda: CudaReflex,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    first_valid: usize,
    max_period: usize,
}

fn expand_reflex_periods(range: &ReflexBatchRange) -> Vec<usize> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return vec![start];
    }
    (start..=end).step_by(step).collect()
}

fn prep_reflex_batch_state() -> ReflexBatchState {
    let cuda = CudaReflex::new(0).expect("cuda reflex");
    let series_len = 1_000_000usize;
    let first_valid = 6usize;
    let range = ReflexBatchRange {
        period: (4, 128, 4),
    };

    let mut prices = vec![f32::NAN; series_len];
    for i in first_valid..series_len {
        let x = i as f32;
        prices[i] = (x * 0.0009).sin() + 0.00012 * x + (x * 0.0003).cos() * 0.35;
    }

    let combos = expand_reflex_periods(&range);
    let n_combos = combos.len();
    let periods_i32: Vec<i32> = combos.iter().map(|&p| p as i32).collect();
    let max_period = combos.iter().copied().max().unwrap_or(0);

    let d_prices = DeviceBuffer::from_slice(&prices).expect("d_prices");
    let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("d_periods");
    let mut d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos).expect("d_out") };

    ReflexBatchState {
        cuda,
        d_prices,
        d_periods,
        d_out,
        series_len,
        n_combos,
        first_valid,
        max_period,
    }
}

fn launch_reflex_batch(state: &mut ReflexBatchState) {
    state
        .cuda
        .reflex_batch_device(
            &state.d_prices,
            &state.d_periods,
            state.series_len,
            state.n_combos,
            state.first_valid,
            state.max_period,
            &mut state.d_out,
        )
        .expect("launch reflex batch");
}

fn reflex_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let series_len = 1_000_000usize;
    let range = ReflexBatchRange {
        period: (4, 128, 4),
    };
    let combos = expand_reflex_periods(&range);
    let n_combos = combos.len();
    let prices_bytes = series_len * std::mem::size_of::<f32>();
    let periods_bytes = n_combos * std::mem::size_of::<i32>();
    let out_bytes = series_len * n_combos * std::mem::size_of::<f32>();
    let approx = prices_bytes + periods_bytes + out_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip Reflex batch (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("reflex_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_reflex_batch_state();
    group.bench_function("1m_x_period_sweep", |b| {
        b.iter(|| {
            launch_reflex_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct ReflexManySeriesState {
    cuda: CudaReflex,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    num_series: usize,
    series_len: usize,
    period: usize,
}

fn prep_reflex_many_series_state() -> ReflexManySeriesState {
    let cuda = CudaReflex::new(0).expect("cuda reflex");
    let num_series = 48usize;
    let series_len = 180_000usize;
    let period = 24usize;

    let mut prices_tm = vec![f32::NAN; num_series * series_len];
    let mut first_valids = vec![0i32; num_series];

    for series in 0..num_series {
        let fv = (series % 6) as usize;
        first_valids[series] = fv as i32;
        for row in fv..series_len {
            let idx = row * num_series + series;
            let x = row as f32 + (series as f32) * 0.27;
            prices_tm[idx] = (x * 0.0007).cos() + 0.00018 * x;
        }
    }

    let d_prices_tm = DeviceBuffer::from_slice(&prices_tm).expect("d_prices_tm");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
    let mut d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len).expect("d_out_tm") };

    ReflexManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        num_series,
        series_len,
        period,
    }
}

fn launch_reflex_many_series(state: &mut ReflexManySeriesState) {
    state
        .cuda
        .reflex_many_series_one_param_device(
            &state.d_prices_tm,
            state.period,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .expect("launch reflex many-series");
}

fn reflex_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 48usize;
    let series_len = 180_000usize;
    let prices_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let first_valid_bytes = num_series * std::mem::size_of::<i32>();
    let out_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let approx = prices_bytes + first_valid_bytes + out_bytes + 48 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip Reflex many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("reflex_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_reflex_many_series_state();
    group.bench_function("time_major_default", |b| {
        b.iter(|| {
            launch_reflex_many_series(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct VwmaBatchState {
    cuda: CudaVwma,
    d_pv_prefix: DeviceBuffer<f32>,
    d_vol_prefix: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    first_valid: usize,
}

fn compute_vwma_prefixes(
    prices: &[f32],
    volumes: &[f32],
    first_valid: usize,
) -> (Vec<f32>, Vec<f32>) {
    let len = prices.len();
    let mut pv = vec![0f32; len];
    let mut vol = vec![0f32; len];
    let mut acc_pv = 0f32;
    let mut acc_vol = 0f32;
    for i in first_valid..len {
        let p = prices[i];
        let v = volumes[i];
        if p.is_nan() || v.is_nan() || acc_pv.is_nan() || acc_vol.is_nan() {
            acc_pv = f32::NAN;
            acc_vol = f32::NAN;
        } else {
            acc_pv += p * v;
            acc_vol += v;
        }
        pv[i] = acc_pv;
        vol[i] = acc_vol;
    }
    (pv, vol)
}

fn compute_vwma_prefixes_time_major(
    prices_tm: &[f32],
    volumes_tm: &[f32],
    num_series: usize,
    series_len: usize,
    first_valids: &[i32],
) -> (Vec<f32>, Vec<f32>) {
    let mut pv = vec![0f32; prices_tm.len()];
    let mut vol = vec![0f32; volumes_tm.len()];
    for series_idx in 0..num_series {
        let fv = first_valids[series_idx].max(0) as usize;
        let mut acc_pv = 0f32;
        let mut acc_vol = 0f32;
        for row in 0..series_len {
            let idx = row * num_series + series_idx;
            if row >= fv {
                let p = prices_tm[idx];
                let v = volumes_tm[idx];
                if p.is_nan() || v.is_nan() || acc_pv.is_nan() || acc_vol.is_nan() {
                    acc_pv = f32::NAN;
                    acc_vol = f32::NAN;
                } else {
                    acc_pv += p * v;
                    acc_vol += v;
                }
            }
            pv[idx] = acc_pv;
            vol[idx] = acc_vol;
        }
    }
    (pv, vol)
}

fn prep_vwma_batch_state() -> VwmaBatchState {
    let cuda = CudaVwma::new(0).expect("cuda vwma");
    let series_len = 1_000_000usize;
    let first_valid = 6usize;

    let mut prices = vec![f32::NAN; series_len];
    let mut volumes = vec![f32::NAN; series_len];
    for i in first_valid..series_len {
        let x = i as f32;
        prices[i] = (x * 0.0008).sin() + 0.0003 * x;
        volumes[i] = 80.0 + (x * 0.0005).cos();
    }

    let sweep = VwmaBatchRange { period: (6, 96, 6) };
    let periods: Vec<i32> = {
        let (start, end, step) = sweep.period;
        if step == 0 || start == end {
            vec![start as i32]
        } else {
            (start..=end).step_by(step).map(|p| p as i32).collect()
        }
    };
    let n_combos = periods.len();

    let (pv_prefix, vol_prefix) = compute_vwma_prefixes(&prices, &volumes, first_valid);

    let d_pv_prefix = DeviceBuffer::from_slice(&pv_prefix).expect("d_pv_prefix");
    let d_vol_prefix = DeviceBuffer::from_slice(&vol_prefix).expect("d_vol_prefix");
    let d_periods = DeviceBuffer::from_slice(&periods).expect("d_periods");
    let mut d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len).expect("d_out") };

    VwmaBatchState {
        cuda,
        d_pv_prefix,
        d_vol_prefix,
        d_periods,
        d_out,
        series_len,
        n_combos,
        first_valid,
    }
}

fn launch_vwma_batch(state: &mut VwmaBatchState) {
    state
        .cuda
        .vwma_batch_device(
            &state.d_pv_prefix,
            &state.d_vol_prefix,
            &state.d_periods,
            state.series_len,
            state.n_combos,
            state.first_valid,
            &mut state.d_out,
        )
        .expect("launch vwma batch");
}

fn vwma_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let series_len = 1_000_000usize;
    let sweep = VwmaBatchRange { period: (6, 96, 6) };
    let n_combos = if sweep.period.1 == sweep.period.0 || sweep.period.2 == 0 {
        1
    } else {
        ((sweep.period.1 - sweep.period.0) / sweep.period.2) + 1
    };
    let prefix_bytes = 2 * series_len * std::mem::size_of::<f32>();
    let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
    let approx = prefix_bytes + out_bytes + 64 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip VWMA batch (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("vwma_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_vwma_batch_state();
    group.bench_function("1m_x_period_sweep", |b| {
        b.iter(|| {
            launch_vwma_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct VwmaManySeriesState {
    cuda: CudaVwma,
    d_pv_prefix_tm: DeviceBuffer<f32>,
    d_vol_prefix_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: usize,
    num_series: usize,
    series_len: usize,
}

fn prep_vwma_many_series_state() -> VwmaManySeriesState {
    let cuda = CudaVwma::new(0).expect("cuda vwma");
    let series_len = 200_000usize;
    let num_series = 64usize;
    let period = 20usize;

    let mut prices_tm = vec![f32::NAN; num_series * series_len];
    let mut volumes_tm = vec![f32::NAN; num_series * series_len];
    let mut first_valids = vec![0i32; num_series];

    for series_idx in 0..num_series {
        let fv = series_idx.min(10);
        first_valids[series_idx] = fv as i32;
        for row in fv..series_len {
            let idx = row * num_series + series_idx;
            let x = (row as f32) + (series_idx as f32) * 0.3;
            prices_tm[idx] = (x * 0.0015).sin() + 0.0002 * x;
            volumes_tm[idx] = 100.0 + (x * 0.002).cos() + series_idx as f32 * 0.05;
        }
    }

    let (pv_prefix_tm, vol_prefix_tm) = compute_vwma_prefixes_time_major(
        &prices_tm,
        &volumes_tm,
        num_series,
        series_len,
        &first_valids,
    );

    let d_pv_prefix_tm = DeviceBuffer::from_slice(&pv_prefix_tm).expect("d_pv_prefix_tm");
    let d_vol_prefix_tm = DeviceBuffer::from_slice(&vol_prefix_tm).expect("d_vol_prefix_tm");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("d_first_valids");
    let mut d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len).expect("d_out_tm") };

    VwmaManySeriesState {
        cuda,
        d_pv_prefix_tm,
        d_vol_prefix_tm,
        d_first_valids,
        d_out_tm,
        period,
        num_series,
        series_len,
    }
}

fn launch_vwma_many_series(state: &mut VwmaManySeriesState) {
    state
        .cuda
        .vwma_many_series_one_param_device(
            &state.d_pv_prefix_tm,
            &state.d_vol_prefix_tm,
            state.period,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .expect("launch vwma many-series");
}

fn vwma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 64usize;
    let series_len = 200_000usize;
    let prefix_bytes = 2 * num_series * series_len * std::mem::size_of::<f32>();
    let first_valid_bytes = num_series * std::mem::size_of::<i32>();
    let out_bytes = num_series * series_len * std::mem::size_of::<f32>();
    let approx = prefix_bytes + first_valid_bytes + out_bytes + 64 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip VWMA many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("vwma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_vwma_many_series_state();
    group.bench_function("time_major_default", |b| {
        b.iter(|| {
            launch_vwma_many_series(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// ALMA: one series × many params (multi-stream, very large inputs)
// ──────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    alma_one_series_bench,
    alma_one_series_bench_1m_240,
    alma_one_series_bench_250k_4k,
    alma_many_series_bench,
    dma_batch_bench,
    dma_many_series_bench,
    gaussian_batch_bench,
    gaussian_many_series_bench,
    ehlers_pma_batch_bench,
    ehlers_pma_many_series_bench,
    uma_batch_bench,
    uma_many_series_bench,
    jma_batch_bench,
    jma_many_series_bench,
    mama_batch_bench,
    mama_many_series_bench,
    tema_batch_bench,
    tema_many_series_bench,
    sqwma_batch_bench,
    sqwma_many_series_bench,
    reflex_batch_bench,
    reflex_many_series_bench,
    vwma_batch_bench,
    vwma_many_series_bench,
    wclprice_bench
);
criterion_main!(benches);
