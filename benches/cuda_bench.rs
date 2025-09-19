#![cfg(feature = "cuda")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cust::memory::DeviceBuffer;

use my_project::cuda::cuda_available;
use my_project::cuda::moving_averages::CudaAlma;
use my_project::indicators::moving_averages::alma::{AlmaBatchRange, AlmaParams};

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
    let periods: Vec<usize> = (1..=240).collect();
    let n_combos = periods.len();
    let max_period = *periods.iter().max().unwrap();
    let host_prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
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
    let periods: Vec<usize> = (1..=240).collect(); // 240
    let offsets: [f64; 4] = [0.25, 0.5, 0.75, 0.85]; // 4
    let sigmas: [f64; 4] = [3.0, 6.0, 9.0, 12.0]; // 4
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
    let n_combos = combos.len(); // ~3840

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
// ALMA: one series × many params (multi-stream, very large inputs)
// ──────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    alma_one_series_bench,
    alma_one_series_bench_1m_240,
    alma_one_series_bench_250k_4k,
    alma_many_series_bench
);
criterion_main!(benches);
