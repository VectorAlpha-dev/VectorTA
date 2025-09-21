#![cfg(feature = "cuda")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cust::memory::DeviceBuffer;

use my_project::cuda::cuda_available;
use my_project::cuda::moving_averages::{
    CudaAlma, CudaBuffAverages, CudaFrama, CudaHma, CudaLinreg, CudaNma, CudaSma,
    CudaSuperSmoother, CudaTrendflex, CudaVpwma, CudaZlema,
};
use my_project::cuda::{CudaWad, CudaZscore};
use my_project::indicators::moving_averages::alma::{AlmaBatchRange, AlmaParams};
use my_project::indicators::moving_averages::buff_averages::BuffAveragesBatchRange;
use my_project::indicators::moving_averages::frama::{FramaBatchRange, FramaParams};
use my_project::indicators::moving_averages::hma::{HmaBatchRange, HmaParams};
use my_project::indicators::moving_averages::linreg::{LinRegBatchRange, LinRegParams};
use my_project::indicators::moving_averages::nma::{NmaBatchRange, NmaParams};
use my_project::indicators::moving_averages::sma::{SmaBatchRange, SmaParams};
use my_project::indicators::moving_averages::supersmoother::{
    SuperSmootherBatchRange, SuperSmootherParams,
};
use my_project::indicators::moving_averages::trendflex::{TrendFlexBatchRange, TrendFlexParams};
use my_project::indicators::moving_averages::vpwma::VpwmaBatchRange;
use my_project::indicators::moving_averages::zlema::ZlemaBatchRange;
use my_project::indicators::zscore::ZscoreBatchRange;

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

fn gen_positive_series_f32(len: usize) -> Vec<f32> {
    let mut v = vec![f32::NAN; len];
    for i in 10..len {
        let x = i as f32 * 0.0017;
        v[i] = 1.35 + 0.55 * x.sin() + 0.25 * x.cos();
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

fn gen_positive_time_major_f32(num_series: usize, series_len: usize) -> Vec<f32> {
    let mut v = vec![f32::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in (j + 6)..series_len {
            let x = t as f32 * 0.0024 + j as f32 * 0.11;
            v[t * num_series + j] = 1.28 + 0.48 * x.cos() + 0.18 * x.sin();
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

fn compute_vpwma_weights(period: usize, power: f64) -> (Vec<f32>, f32) {
    let win_len = period.saturating_sub(1);
    let mut weights = vec![0f32; win_len];
    let mut norm = 0.0f64;
    for k in 0..win_len {
        let w = (period as f64 - k as f64).powf(power);
        weights[k] = w as f32;
        norm += w;
    }
    let inv = if norm == 0.0 {
        0.0
    } else {
        (1.0 / norm) as f32
    };
    (weights, inv)
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

struct BuffAveragesBatchState {
    cuda: CudaBuffAverages,
    price: Vec<f32>,
    volume: Vec<f32>,
    sweep: BuffAveragesBatchRange,
}

fn prep_buff_averages_batch() -> BuffAveragesBatchState {
    let cuda = CudaBuffAverages::new(0).expect("cuda buff averages");
    let len = 60_000usize;
    let mut price = vec![f32::NAN; len];
    let mut volume = vec![f32::NAN; len];
    for i in 3..len {
        let x = i as f32;
        price[i] = (x * 0.001).sin() + 0.0001 * x;
        volume[i] = (x * 0.0007).cos().abs() + 0.6;
    }
    let sweep = BuffAveragesBatchRange {
        fast_period: (4, 28, 4),
        slow_period: (32, 128, 16),
    };
    BuffAveragesBatchState {
        cuda,
        price,
        volume,
        sweep,
    }
}

fn launch_buff_averages_batch(state: &mut BuffAveragesBatchState) {
    let (fast_dev, slow_dev) = state
        .cuda
        .buff_averages_batch_dev(&state.price, &state.volume, &state.sweep)
        .unwrap();
    drop(fast_dev);
    drop(slow_dev);
}

fn buff_averages_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("buff_averages_cuda_batch_dev");
    group.sample_size(10);
    let mut state = prep_buff_averages_batch();
    group.bench_with_input(BenchmarkId::from_parameter("60k_x_49combos"), &0, |b, _| {
        b.iter(|| {
            launch_buff_averages_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct LinregBatchState {
    cuda: CudaLinreg,
    data: Vec<f32>,
    sweep: LinRegBatchRange,
}

fn prep_linreg_batch() -> LinregBatchState {
    let cuda = CudaLinreg::new(0).expect("cuda linreg");
    let data_f64 = gen_series(60_000);
    let data = data_f64.iter().map(|&v| v as f32).collect();
    let sweep = LinRegBatchRange { period: (8, 72, 4) };
    LinregBatchState { cuda, data, sweep }
}

fn launch_linreg_batch(state: &mut LinregBatchState) {
    let (dev, _) = state
        .cuda
        .linreg_batch_dev(&state.data, &state.sweep)
        .expect("linreg_cuda_batch_dev");
    drop(dev);
}

fn linreg_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("linreg_cuda_batch_dev");
    group.sample_size(10);
    let mut state = prep_linreg_batch();
    group.bench_with_input(BenchmarkId::from_parameter("60k_x_17combos"), &0, |b, _| {
        b.iter(|| {
            launch_linreg_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct LinregManySeriesState {
    cuda: CudaLinreg,
    data_tm: Vec<f32>,
    cols: usize,
    rows: usize,
    params: LinRegParams,
}

fn prep_linreg_many_series() -> LinregManySeriesState {
    let cuda = CudaLinreg::new(0).expect("cuda linreg");
    let cols = 384usize;
    let rows = 32_000usize;
    let data_tm = gen_time_major_f32(cols, rows);
    let params = LinRegParams { period: Some(18) };
    LinregManySeriesState {
        cuda,
        data_tm,
        cols,
        rows,
        params,
    }
}

fn launch_linreg_many_series(state: &mut LinregManySeriesState) {
    let dev = state
        .cuda
        .linreg_multi_series_one_param_time_major_dev(
            &state.data_tm,
            state.cols,
            state.rows,
            &state.params,
        )
        .expect("linreg_multi_series_one_param_time_major_dev");
    drop(dev);
}

fn linreg_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("linreg_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_linreg_many_series();
    group.bench_with_input(
        BenchmarkId::from_parameter("384series_x_32k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_linreg_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct SmaBatchState {
    cuda: CudaSma,
    data: Vec<f32>,
    sweep: SmaBatchRange,
}

fn prep_sma_batch() -> SmaBatchState {
    let cuda = CudaSma::new(0).expect("cuda sma");
    let data_f64 = gen_series(60_000);
    let data = data_f64.iter().map(|&v| v as f32).collect();
    let sweep = SmaBatchRange {
        period: (5, 125, 5),
    };
    SmaBatchState { cuda, data, sweep }
}

fn launch_sma_batch(state: &mut SmaBatchState) {
    let (dev, _) = state
        .cuda
        .sma_batch_dev(&state.data, &state.sweep)
        .expect("sma_cuda_batch_dev");
    drop(dev);
}

fn sma_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("sma_cuda_batch_dev");
    group.sample_size(10);
    let mut state = prep_sma_batch();
    group.bench_with_input(BenchmarkId::from_parameter("60k_x_25combos"), &0, |b, _| {
        b.iter(|| {
            launch_sma_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct SmaManySeriesState {
    cuda: CudaSma,
    data_tm: Vec<f32>,
    cols: usize,
    rows: usize,
    params: SmaParams,
}

fn prep_sma_many_series() -> SmaManySeriesState {
    let cuda = CudaSma::new(0).expect("cuda sma");
    let cols = 512usize;
    let rows = 25_000usize;
    let data_tm = gen_time_major_f32(cols, rows);
    let params = SmaParams { period: Some(18) };
    SmaManySeriesState {
        cuda,
        data_tm,
        cols,
        rows,
        params,
    }
}

fn launch_sma_many_series(state: &mut SmaManySeriesState) {
    let dev = state
        .cuda
        .sma_multi_series_one_param_time_major_dev(
            &state.data_tm,
            state.cols,
            state.rows,
            &state.params,
        )
        .expect("sma_multi_series_one_param_time_major_dev");
    drop(dev);
}

fn sma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("sma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_sma_many_series();
    group.bench_with_input(
        BenchmarkId::from_parameter("512series_x_25k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_sma_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct HmaBatchState {
    cuda: CudaHma,
    data: Vec<f32>,
    sweep: HmaBatchRange,
}

fn prep_hma_batch() -> HmaBatchState {
    let cuda = CudaHma::new(0).expect("cuda hma");
    let data_f64 = gen_series(60_000);
    let data = data_f64.iter().map(|&v| v as f32).collect();
    let sweep = HmaBatchRange {
        period: (12, 108, 8),
    };
    HmaBatchState { cuda, data, sweep }
}

fn launch_hma_batch(state: &mut HmaBatchState) {
    let (dev, _) = state
        .cuda
        .hma_batch_dev(&state.data, &state.sweep)
        .expect("hma_cuda_batch_dev");
    drop(dev);
}

fn hma_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("hma_cuda_batch_dev");
    group.sample_size(10);
    let mut state = prep_hma_batch();
    group.bench_with_input(BenchmarkId::from_parameter("60k_x_13combos"), &0, |b, _| {
        b.iter(|| {
            launch_hma_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct HmaManySeriesState {
    cuda: CudaHma,
    data_tm: Vec<f32>,
    cols: usize,
    rows: usize,
    params: HmaParams,
}

fn prep_hma_many_series() -> HmaManySeriesState {
    let cuda = CudaHma::new(0).expect("cuda hma");
    let cols = 512usize;
    let rows = 25_000usize;
    let data_tm = gen_time_major_f32(cols, rows);
    let params = HmaParams { period: Some(24) };
    HmaManySeriesState {
        cuda,
        data_tm,
        cols,
        rows,
        params,
    }
}

fn launch_hma_many_series(state: &mut HmaManySeriesState) {
    let dev = state
        .cuda
        .hma_multi_series_one_param_time_major_dev(
            &state.data_tm,
            state.cols,
            state.rows,
            &state.params,
        )
        .expect("hma_multi_series_one_param_time_major_dev");
    drop(dev);
}

fn hma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("hma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_hma_many_series();
    group.bench_with_input(
        BenchmarkId::from_parameter("512series_x_25k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_hma_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct NmaBatchState {
    cuda: CudaNma,
    data: Vec<f32>,
    sweep: NmaBatchRange,
}

fn prep_nma_batch() -> NmaBatchState {
    let cuda = CudaNma::new(0).expect("cuda nma");
    let data = gen_positive_series_f32(60_000);
    let sweep = NmaBatchRange { period: (6, 96, 6) };
    NmaBatchState { cuda, data, sweep }
}

fn launch_nma_batch(state: &mut NmaBatchState) {
    let (dev, _) = state
        .cuda
        .nma_batch_dev(&state.data, &state.sweep)
        .expect("nma_cuda_batch_dev");
    drop(dev);
}

fn nma_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("nma_cuda_batch_dev");
    group.sample_size(10);
    let mut state = prep_nma_batch();
    group.bench_with_input(BenchmarkId::from_parameter("60k_x_16combos"), &0, |b, _| {
        b.iter(|| {
            launch_nma_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct NmaManySeriesState {
    cuda: CudaNma,
    data_tm: Vec<f32>,
    cols: usize,
    rows: usize,
    params: NmaParams,
}

fn prep_nma_many_series() -> NmaManySeriesState {
    let cuda = CudaNma::new(0).expect("cuda nma");
    let cols = 512usize;
    let rows = 25_000usize;
    let data_tm = gen_positive_time_major_f32(cols, rows);
    let params = NmaParams { period: Some(24) };
    NmaManySeriesState {
        cuda,
        data_tm,
        cols,
        rows,
        params,
    }
}

fn launch_nma_many_series(state: &mut NmaManySeriesState) {
    let dev = state
        .cuda
        .nma_multi_series_one_param_time_major_dev(
            &state.data_tm,
            state.cols,
            state.rows,
            &state.params,
        )
        .expect("nma_multi_series_one_param_time_major_dev");
    drop(dev);
}

fn nma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("nma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_nma_many_series();
    group.bench_with_input(
        BenchmarkId::from_parameter("512series_x_25k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_nma_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct ZlemaBatchState {
    cuda: CudaZlema,
    data: Vec<f32>,
    sweep: ZlemaBatchRange,
}

fn prep_zlema_batch() -> ZlemaBatchState {
    let cuda = CudaZlema::new(0).expect("cuda zlema");
    let data_f64 = gen_series(60_000);
    let data = data_f64.iter().map(|&v| v as f32).collect();
    let sweep = ZlemaBatchRange {
        period: (5, 125, 5),
    };
    ZlemaBatchState { cuda, data, sweep }
}

fn launch_zlema_batch(state: &mut ZlemaBatchState) {
    let (dev, _) = state
        .cuda
        .zlema_batch_dev(&state.data, &state.sweep)
        .expect("zlema_cuda_batch_dev");
    drop(dev);
}

fn zlema_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("zlema_cuda_batch_dev");
    group.sample_size(10);
    let mut state = prep_zlema_batch();
    group.bench_with_input(BenchmarkId::from_parameter("60k_x_25combos"), &0, |b, _| {
        b.iter(|| {
            launch_zlema_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct TrendflexBatchState {
    cuda: CudaTrendflex,
    data: Vec<f32>,
    sweep: TrendFlexBatchRange,
}

fn prep_trendflex_batch() -> TrendflexBatchState {
    let cuda = CudaTrendflex::new(0).expect("cuda trendflex");
    let data_f64 = gen_series(60_000);
    let data = data_f64.iter().map(|&v| v as f32).collect();
    let sweep = TrendFlexBatchRange {
        period: (6, 126, 6),
    };
    TrendflexBatchState { cuda, data, sweep }
}

fn launch_trendflex_batch(state: &mut TrendflexBatchState) {
    let (dev, _) = state
        .cuda
        .trendflex_batch_dev(&state.data, &state.sweep)
        .expect("trendflex_cuda_batch_dev");
    drop(dev);
}

fn trendflex_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("trendflex_cuda_batch_dev");
    group.sample_size(10);
    let mut state = prep_trendflex_batch();
    group.bench_with_input(BenchmarkId::from_parameter("60k_x_21combos"), &0, |b, _| {
        b.iter(|| {
            launch_trendflex_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct TrendflexManySeriesState {
    cuda: CudaTrendflex,
    data_tm: Vec<f32>,
    cols: usize,
    rows: usize,
    params: TrendFlexParams,
}

fn prep_trendflex_many_series() -> TrendflexManySeriesState {
    let cuda = CudaTrendflex::new(0).expect("cuda trendflex");
    let cols = 512usize;
    let rows = 25_000usize;
    let data_tm = gen_time_major_f32(cols, rows);
    let params = TrendFlexParams { period: Some(18) };
    TrendflexManySeriesState {
        cuda,
        data_tm,
        cols,
        rows,
        params,
    }
}

fn launch_trendflex_many_series(state: &mut TrendflexManySeriesState) {
    let dev = state
        .cuda
        .trendflex_multi_series_one_param_time_major_dev(
            &state.data_tm,
            state.cols,
            state.rows,
            &state.params,
        )
        .expect("trendflex_multi_series_one_param_time_major_dev");
    drop(dev);
}

fn trendflex_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("trendflex_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_trendflex_many_series();
    group.bench_with_input(
        BenchmarkId::from_parameter("512series_x_25k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_trendflex_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct FramaBatchState {
    cuda: CudaFrama,
    high: Vec<f32>,
    low: Vec<f32>,
    close: Vec<f32>,
    sweep: FramaBatchRange,
}

fn prep_frama_batch() -> FramaBatchState {
    let cuda = CudaFrama::new(0).expect("cuda frama");
    let close_f64 = gen_series(60_000);
    let mut high = Vec::with_capacity(close_f64.len());
    let mut low = Vec::with_capacity(close_f64.len());
    for (idx, &val) in close_f64.iter().enumerate() {
        if val.is_nan() {
            high.push(f32::NAN);
            low.push(f32::NAN);
        } else {
            let t = idx as f32 * 0.0013;
            let offset = (0.55f32 + 0.04f32 * t.sin()) as f64;
            high.push((val + offset) as f32);
            low.push((val - offset) as f32);
        }
    }
    let close: Vec<f32> = close_f64.iter().map(|&v| v as f32).collect();
    let sweep = FramaBatchRange {
        window: (10, 34, 6),
        sc: (200, 320, 40),
        fc: (1, 2, 1),
    };
    FramaBatchState {
        cuda,
        high,
        low,
        close,
        sweep,
    }
}

fn launch_frama_batch(state: &mut FramaBatchState) {
    let (dev, _) = state
        .cuda
        .frama_batch_dev(&state.high, &state.low, &state.close, &state.sweep)
        .expect("frama_cuda_batch_dev");
    drop(dev);
}

fn frama_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("frama_cuda_batch_dev");
    group.sample_size(10);
    let mut state = prep_frama_batch();
    group.bench_with_input(BenchmarkId::from_parameter("60k_x_12combos"), &0, |b, _| {
        b.iter(|| {
            launch_frama_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct FramaManySeriesState {
    cuda: CudaFrama,
    high_tm: Vec<f32>,
    low_tm: Vec<f32>,
    close_tm: Vec<f32>,
    cols: usize,
    rows: usize,
    params: FramaParams,
}

fn prep_frama_many_series() -> FramaManySeriesState {
    let cuda = CudaFrama::new(0).expect("cuda frama");
    let cols = 512usize;
    let rows = 25_000usize;
    let close_tm = gen_time_major_f32(cols, rows);
    let mut high_tm = Vec::with_capacity(close_tm.len());
    let mut low_tm = Vec::with_capacity(close_tm.len());
    for (idx, &val) in close_tm.iter().enumerate() {
        if val.is_nan() {
            high_tm.push(f32::NAN);
            low_tm.push(f32::NAN);
        } else {
            let row = (idx / cols) as f32;
            let col = (idx % cols) as f32;
            let offset = 0.48f32 + 0.03f32 * (row * 0.002 + col * 0.01).cos();
            high_tm.push(val + offset);
            low_tm.push(val - offset - 0.05f32);
        }
    }
    let params = FramaParams {
        window: Some(18),
        sc: Some(240),
        fc: Some(2),
    };
    FramaManySeriesState {
        cuda,
        high_tm,
        low_tm,
        close_tm,
        cols,
        rows,
        params,
    }
}

fn launch_frama_many_series(state: &mut FramaManySeriesState) {
    let dev = state
        .cuda
        .frama_many_series_one_param_time_major_dev(
            &state.high_tm,
            &state.low_tm,
            &state.close_tm,
            state.cols,
            state.rows,
            &state.params,
        )
        .expect("frama_many_series_one_param_time_major_dev");
    drop(dev);
}

fn frama_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("frama_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_frama_many_series();
    group.bench_with_input(
        BenchmarkId::from_parameter("512series_x_25k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_frama_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct SupersmootherBatchState {
    cuda: CudaSuperSmoother,
    data: Vec<f32>,
    sweep: SuperSmootherBatchRange,
}

fn prep_supersmoother_batch() -> SupersmootherBatchState {
    let cuda = CudaSuperSmoother::new(0).expect("cuda supersmoother");
    let data_f64 = gen_series(60_000);
    let data = data_f64.iter().map(|&v| v as f32).collect();
    let sweep = SuperSmootherBatchRange {
        period: (5, 125, 5),
    };
    SupersmootherBatchState { cuda, data, sweep }
}

fn launch_supersmoother_batch(state: &mut SupersmootherBatchState) {
    let (dev, _) = state
        .cuda
        .supersmoother_batch_dev(&state.data, &state.sweep)
        .expect("supersmoother_cuda_batch_dev");
    drop(dev);
}

fn supersmoother_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("supersmoother_cuda_batch_dev");
    group.sample_size(10);
    let mut state = prep_supersmoother_batch();
    group.bench_with_input(BenchmarkId::from_parameter("60k_x_25combos"), &0, |b, _| {
        b.iter(|| {
            launch_supersmoother_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct SupersmootherManySeriesState {
    cuda: CudaSuperSmoother,
    data_tm: Vec<f32>,
    cols: usize,
    rows: usize,
    params: SuperSmootherParams,
}

fn prep_supersmoother_many_series() -> SupersmootherManySeriesState {
    let cuda = CudaSuperSmoother::new(0).expect("cuda supersmoother");
    let cols = 512usize;
    let rows = 25_000usize;
    let data_tm = gen_time_major_f32(cols, rows);
    let params = SuperSmootherParams { period: Some(18) };
    SupersmootherManySeriesState {
        cuda,
        data_tm,
        cols,
        rows,
        params,
    }
}

fn launch_supersmoother_many_series(state: &mut SupersmootherManySeriesState) {
    let dev = state
        .cuda
        .supersmoother_multi_series_one_param_time_major_dev(
            &state.data_tm,
            state.cols,
            state.rows,
            &state.params,
        )
        .expect("supersmoother_multi_series_one_param_time_major_dev");
    drop(dev);
}

fn supersmoother_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("supersmoother_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_supersmoother_many_series();
    group.bench_with_input(
        BenchmarkId::from_parameter("512series_x_25k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_supersmoother_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct VpwmaBatchState {
    cuda: CudaVpwma,
    data: Vec<f32>,
    sweep: VpwmaBatchRange,
}

fn prep_vpwma_batch() -> VpwmaBatchState {
    let cuda = CudaVpwma::new(0).expect("cuda vpwma");
    let data_f64 = gen_series(60_000);
    let data = data_f64.iter().map(|&v| v as f32).collect();
    let sweep = VpwmaBatchRange {
        period: (5, 125, 5),
        power: (0.2, 1.0, 0.2),
    };
    VpwmaBatchState { cuda, data, sweep }
}

fn launch_vpwma_batch(state: &mut VpwmaBatchState) {
    let (dev, _) = state
        .cuda
        .vpwma_batch_dev(&state.data, &state.sweep)
        .expect("vpwma_cuda_batch_dev");
    drop(dev);
}

fn vpwma_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("vpwma_cuda_batch_dev");
    group.sample_size(10);
    let mut state = prep_vpwma_batch();
    group.bench_with_input(
        BenchmarkId::from_parameter("60k_x_125combos"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_vpwma_batch(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct VpwmaManySeriesState {
    cuda: CudaVpwma,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_weights: DeviceBuffer<f32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    inv: f32,
    num_series: i32,
    series_len: i32,
}

fn prep_vpwma_many_series_one_param() -> VpwmaManySeriesState {
    let cuda = CudaVpwma::new(0).expect("cuda vpwma");
    let num_series = 4000usize;
    let series_len = 50_000usize;
    let host_tm = gen_time_major_f32(num_series, series_len);
    let period = 14usize;
    let power = 0.6f64;
    let (weights, inv) = compute_vpwma_weights(period, power);
    let mut first_valids = vec![0i32; num_series];
    for j in 0..num_series {
        let mut first = 0i32;
        for row in 0..series_len {
            let val = host_tm[row * num_series + j];
            if !val.is_nan() {
                first = row as i32;
                break;
            }
        }
        first_valids[j] = first;
    }
    let d_prices_tm = DeviceBuffer::from_slice(&host_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_weights = DeviceBuffer::from_slice(&weights).unwrap();
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();
    VpwmaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_weights,
        d_out_tm,
        period: period as i32,
        inv,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_vpwma_many_series_one_param(state: &mut VpwmaManySeriesState) {
    state
        .cuda
        .vpwma_multi_series_one_param_device(
            &state.d_prices_tm,
            &state.d_first_valids,
            state.period,
            state.inv,
            state.num_series,
            state.series_len,
            &state.d_weights,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn vpwma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("vpwma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_vpwma_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("4000series_x_50k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_vpwma_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct ZscoreBatchState {
    cuda: CudaZscore,
    data: Vec<f32>,
    sweep: ZscoreBatchRange,
}

fn prep_zscore_batch() -> ZscoreBatchState {
    let cuda = CudaZscore::new(0).expect("cuda zscore");
    let len = 60_000usize;
    let mut data = vec![f32::NAN; len];
    for i in 4..len {
        let x = i as f32;
        let base = (x * 0.00043).sin() + (x * 0.00021).cos();
        data[i] = base + 0.0005 * ((i % 13) as f32 - 6.0);
    }
    let sweep = ZscoreBatchRange {
        period: (10, 50, 10),
        ma_type: ("sma".to_string(), "sma".to_string(), "".to_string()),
        nbdev: (0.5, 2.0, 0.5),
        devtype: (0, 0, 0),
    };
    ZscoreBatchState { cuda, data, sweep }
}

fn launch_zscore_batch(state: &mut ZscoreBatchState) {
    let (dev, _) = state
        .cuda
        .zscore_batch_dev(&state.data, &state.sweep)
        .unwrap();
    drop(dev);
}

fn zscore_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("zscore_cuda_batch_dev");
    group.sample_size(10);
    let mut state = prep_zscore_batch();
    group.bench_with_input(BenchmarkId::from_parameter("60k_x_9combos"), &0, |b, _| {
        b.iter(|| {
            launch_zscore_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct WadSeriesState {
    cuda: CudaWad,
    high: Vec<f32>,
    low: Vec<f32>,
    close: Vec<f32>,
}

fn prep_wad_series() -> WadSeriesState {
    let cuda = CudaWad::new(0).expect("cuda wad");
    let len = 60_000usize;
    let mut high = vec![0.0f32; len];
    let mut low = vec![0.0f32; len];
    let mut close = vec![0.0f32; len];

    let mut price = 101.0f32;
    close[0] = price;
    high[0] = price + 0.6;
    low[0] = price - 0.6;
    for i in 1..len {
        let t = i as f32;
        let delta = (t * 0.0043).sin() * 0.65 + (t * 0.0027).cos() * 0.38;
        let c = price + delta;
        close[i] = c;
        high[i] = c + 0.62 + 0.03 * ((i % 7) as f32);
        low[i] = c - 0.61 - 0.02 * ((i % 5) as f32);
        price = c;
    }

    WadSeriesState {
        cuda,
        high,
        low,
        close,
    }
}

fn launch_wad_series(state: &mut WadSeriesState) {
    let dev = state
        .cuda
        .wad_series_dev(&state.high, &state.low, &state.close)
        .unwrap();
    drop(dev);
}

fn wad_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("wad_cuda_dev");
    group.sample_size(10);
    let mut state = prep_wad_series();
    group.bench_with_input(BenchmarkId::from_parameter("60k"), &0, |b, _| {
        b.iter(|| {
            launch_wad_series(&mut state);
            black_box(())
        })
    });
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
    alma_many_series_bench,
    buff_averages_batch_bench,
    linreg_batch_bench,
    sma_batch_bench,
    hma_batch_bench,
    linreg_many_series_bench,
    sma_many_series_bench,
    hma_many_series_bench,
    nma_batch_bench,
    nma_many_series_bench,
    frama_batch_bench,
    frama_many_series_bench,
    trendflex_batch_bench,
    trendflex_many_series_bench,
    supersmoother_batch_bench,
    supersmoother_many_series_bench,
    vpwma_batch_bench,
    vpwma_many_series_bench,
    zlema_batch_bench,
    wad_series_bench,
    zscore_batch_bench
);
criterion_main!(benches);
