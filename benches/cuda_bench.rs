#![cfg(feature = "cuda")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cust::memory::DeviceBuffer;

use my_project::cuda::cuda_available;
use my_project::cuda::moving_averages::{
    CudaAlma, CudaDema, CudaEhlersITrend, CudaEma, CudaHighPass2, CudaJsa, CudaMwdx, CudaSama,
    CudaSrwma, CudaTilson, CudaVama, CudaWilders,
};
use my_project::cuda::oscillators::CudaWillr;
use my_project::indicators::moving_averages::alma::{AlmaBatchRange, AlmaParams};
use my_project::indicators::moving_averages::dema::DemaBatchRange;
use my_project::indicators::moving_averages::ehlers_itrend::EhlersITrendBatchRange;
use my_project::indicators::moving_averages::ema::EmaBatchRange;
use my_project::indicators::moving_averages::highpass_2_pole::{
    HighPass2BatchRange, HighPass2Params,
};
use my_project::indicators::moving_averages::jsa::JsaBatchRange;
use my_project::indicators::moving_averages::mwdx::MwdxBatchRange;
use my_project::indicators::moving_averages::wilders::WildersBatchRange;
use my_project::indicators::willr::{build_willr_gpu_tables, WillrBatchRange};

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

fn compute_highpass2_coefficients(period: usize, k: f64) -> (f32, f32, f32, f32) {
    use std::f64::consts::PI;

    let theta = 2.0 * PI * k / period as f64;
    let sin_v = theta.sin();
    let cos_v = theta.cos();
    let alpha = 1.0 + ((sin_v - 1.0) / cos_v);
    let c = (1.0 - 0.5 * alpha).powi(2);
    let cm2 = -2.0 * c;
    let one_minus_alpha = 1.0 - alpha;
    let two_1m = 2.0 * one_minus_alpha;
    let neg_oma_sq = -(one_minus_alpha * one_minus_alpha);

    (c as f32, cm2 as f32, two_1m as f32, neg_oma_sq as f32)
}

fn expand_highpass2_grid(range: &HighPass2BatchRange) -> Vec<HighPass2Params> {
    fn axis_usize(axis: (usize, usize, usize)) -> Vec<usize> {
        let (start, end, step) = axis;
        if step == 0 || start == end {
            return vec![start];
        }
        (start..=end).step_by(step).collect()
    }

    fn axis_f64(axis: (f64, f64, f64)) -> Vec<f64> {
        let (start, end, step) = axis;
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            return vec![start];
        }
        let mut values = Vec::new();
        let mut current = start;
        while current <= end + 1e-12 {
            values.push(current);
            current += step;
        }
        values
    }

    let periods = axis_usize(range.period);
    let ks = axis_f64(range.k);
    let mut out = Vec::with_capacity(periods.len() * ks.len());
    for &p in &periods {
        for &k in &ks {
            out.push(HighPass2Params {
                period: Some(p),
                k: Some(k),
            });
        }
    }
    out
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

// ──────────────────────────────────────────────────────────────
// DEMA: one series × many periods (device buffers reused)
// ──────────────────────────────────────────────────────────────
struct DemaBatchState {
    cuda: CudaDema,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: i32,
    n_combos: i32,
    first_valid: i32,
}

fn prep_dema_batch_many_periods() -> DemaBatchState {
    let cuda = CudaDema::new(0).expect("cuda dema");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let sweep = DemaBatchRange {
        period: (9, 240, 3),
    };
    let periods: Vec<i32> = (sweep.period.0..=sweep.period.1)
        .step_by(sweep.period.2.max(1))
        .map(|p| p as i32)
        .collect();
    let n_combos = periods.len();
    let first_valid = data_f32.iter().position(|v| !v.is_nan()).unwrap_or(0) as i32;
    let d_prices = DeviceBuffer::from_slice(&data_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();
    DemaBatchState {
        cuda,
        d_prices,
        d_periods,
        d_out,
        series_len: series_len as i32,
        n_combos: n_combos as i32,
        first_valid,
    }
}

fn launch_dema_batch_many_periods(state: &mut DemaBatchState) {
    state
        .cuda
        .dema_batch_device(
            &state.d_prices,
            &state.d_periods,
            state.series_len,
            state.first_valid,
            state.n_combos,
            &mut state.d_out,
        )
        .unwrap();
}

fn dema_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("dema_cuda_one_series_many_periods");
    group.sample_size(10);
    let mut state = prep_dema_batch_many_periods();
    group.bench_with_input(BenchmarkId::from_parameter("200k_x_periods"), &0, |b, _| {
        b.iter(|| {
            launch_dema_batch_many_periods(&mut state);
            black_box(())
        })
    });
    group.finish();
}

fn expand_ema_periods(range: &EmaBatchRange) -> Vec<usize> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return vec![start];
    }
    (start..=end).step_by(step).collect()
}

struct EmaBatchState {
    cuda: CudaEma,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_alphas: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    first_valid: usize,
    n_combos: usize,
}

fn prep_ema_one_series_many_params() -> EmaBatchState {
    let cuda = CudaEma::new(0).expect("cuda ema");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap_or(0);
    let prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let sweep = EmaBatchRange { period: (5, 65, 5) };
    let periods = expand_ema_periods(&sweep);
    let n_combos = periods.len();
    let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
    let alphas: Vec<f32> = periods
        .iter()
        .map(|&p| 2.0f32 / (p as f32 + 1.0f32))
        .collect();

    let d_prices = DeviceBuffer::from_slice(&prices_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_alphas = DeviceBuffer::from_slice(&alphas).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();

    EmaBatchState {
        cuda,
        d_prices,
        d_periods,
        d_alphas,
        d_out,
        series_len,
        first_valid,
        n_combos,
    }
}

fn launch_ema_one_series_many_params(state: &mut EmaBatchState) {
    state
        .cuda
        .ema_batch_device(
            &state.d_prices,
            &state.d_periods,
            &state.d_alphas,
            state.series_len,
            state.first_valid,
            state.n_combos,
            &mut state.d_out,
        )
        .unwrap();
}

fn ema_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("ema_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_ema_one_series_many_params();
    group.bench_with_input(BenchmarkId::from_parameter("200k_x_periods"), &0, |b, _| {
        b.iter(|| {
            launch_ema_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct EmaManySeriesState {
    cuda: CudaEma,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    alpha: f32,
    num_series: usize,
    series_len: usize,
}

fn prep_ema_many_series_one_param() -> EmaManySeriesState {
    let cuda = CudaEma::new(0).expect("cuda ema");
    let num_series = 1024usize;
    let series_len = 50_000usize;
    let data_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = None;
        for t in 0..series_len {
            let value = data_tm[t * num_series + series];
            if value.is_finite() {
                fv = Some(t as i32);
                break;
            }
        }
        let idx = fv.expect("each series must contain finite values");
        first_valids.push(idx);
    }

    let period = 21i32;
    let alpha = 2.0f32 / (period as f32 + 1.0f32);

    let d_prices_tm = DeviceBuffer::from_slice(&data_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    EmaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        period,
        alpha,
        num_series,
        series_len,
    }
}

fn launch_ema_many_series_one_param(state: &mut EmaManySeriesState) {
    state
        .cuda
        .ema_many_series_one_param_device(
            &state.d_prices_tm,
            &state.d_first_valids,
            state.period,
            state.alpha,
            state.num_series,
            state.series_len,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn ema_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("ema_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_ema_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("50k_len_x_1k_series"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_ema_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct EhlersBatchState {
    cuda: CudaEhlersITrend,
    d_prices: DeviceBuffer<f32>,
    d_warmups: DeviceBuffer<i32>,
    d_max_dcs: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    first_valid: usize,
    n_combos: usize,
    max_shared_dc: usize,
}

fn prep_ehlers_one_series_many_params() -> EhlersBatchState {
    let cuda = CudaEhlersITrend::new(0).expect("cuda ehlers");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let first_valid = data.iter().position(|v| v.is_finite()).unwrap_or(0);

    let sweep = EhlersITrendBatchRange {
        warmup_bars: (8, 20, 4),
        max_dc_period: (30, 60, 10),
    };

    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        }
    }

    let warmups_axis = axis(sweep.warmup_bars);
    let max_dc_axis = axis(sweep.max_dc_period);
    let mut warmups = Vec::with_capacity(warmups_axis.len() * max_dc_axis.len());
    let mut max_dcs = Vec::with_capacity(warmups_axis.len() * max_dc_axis.len());
    let mut max_shared_dc = 0usize;
    for &w in &warmups_axis {
        for &m in &max_dc_axis {
            assert!(w > 0 && m > 0, "warmup and max_dc must be positive");
            assert!(
                series_len - first_valid >= w,
                "not enough data for warmup {} (len {} first_valid {})",
                w,
                series_len,
                first_valid
            );
            warmups.push(w as i32);
            max_dcs.push(m as i32);
            max_shared_dc = max_shared_dc.max(m);
        }
    }

    let d_prices = DeviceBuffer::from_slice(&data_f32).unwrap();
    let d_warmups = DeviceBuffer::from_slice(&warmups).unwrap();
    let d_max_dcs = DeviceBuffer::from_slice(&max_dcs).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * warmups.len()) }.unwrap();

    EhlersBatchState {
        cuda,
        d_prices,
        d_warmups,
        d_max_dcs,
        d_out,
        series_len,
        first_valid,
        n_combos: warmups.len(),
        max_shared_dc,
    }
}

fn launch_ehlers_one_series_many_params(state: &mut EhlersBatchState) {
    state
        .cuda
        .ehlers_itrend_batch_device(
            &state.d_prices,
            &state.d_warmups,
            &state.d_max_dcs,
            state.series_len,
            state.first_valid,
            state.n_combos,
            state.max_shared_dc,
            &mut state.d_out,
        )
        .unwrap();
}

fn ehlers_one_series_many_params_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("ehlers_itrend_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_ehlers_one_series_many_params();
    group.bench_with_input(
        BenchmarkId::from_parameter("200k_x_param_pairs"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_ehlers_one_series_many_params(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct EhlersManySeriesState {
    cuda: CudaEhlersITrend,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    num_series: usize,
    series_len: usize,
    warmup: usize,
    max_dc: usize,
}

fn prep_ehlers_many_series_one_param() -> EhlersManySeriesState {
    let cuda = CudaEhlersITrend::new(0).expect("cuda ehlers");
    let num_series = 1024usize;
    let series_len = 50_000usize;
    let data_tm = gen_time_major_f32(num_series, series_len);

    let warmup = 12usize;
    let max_dc = 48usize;

    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = None;
        for t in 0..series_len {
            let value = data_tm[t * num_series + series];
            if value.is_finite() {
                fv = Some(t as i32);
                break;
            }
        }
        let idx = fv.expect("each series must contain finite values");
        assert!(
            series_len - idx as usize >= warmup,
            "series {} insufficient data for warmup {}",
            series,
            warmup
        );
        first_valids.push(idx);
    }

    let d_prices_tm = DeviceBuffer::from_slice(&data_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    EhlersManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        num_series,
        series_len,
        warmup,
        max_dc,
    }
}

fn launch_ehlers_many_series_one_param(state: &mut EhlersManySeriesState) {
    state
        .cuda
        .ehlers_itrend_many_series_one_param_device(
            &state.d_prices_tm,
            &state.d_first_valids,
            state.num_series,
            state.series_len,
            state.warmup,
            state.max_dc,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn ehlers_many_series_one_param_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("ehlers_itrend_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_ehlers_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("50k_len_x_1k_series"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_ehlers_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct WildersBatchState {
    cuda: CudaWilders,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_alphas: DeviceBuffer<f32>,
    d_warm: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: i32,
    n_combos: i32,
    first_valid: i32,
}

fn prep_wilders_batch_many_periods() -> WildersBatchState {
    let cuda = CudaWilders::new(0).expect("cuda wilders");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let sweep = WildersBatchRange {
        period: (5, 240, 3),
    };
    let (start, end, step) = sweep.period;
    let periods: Vec<i32> = if step == 0 || start == end {
        vec![start as i32]
    } else {
        (start..=end).step_by(step).map(|p| p as i32).collect()
    };
    let n_combos = periods.len();
    let first_valid = data_f32.iter().position(|v| !v.is_nan()).unwrap_or(0) as i32;

    let mut alphas = Vec::with_capacity(n_combos);
    let mut warm = Vec::with_capacity(n_combos);
    for &p_i32 in &periods {
        let period = p_i32 as usize;
        let warm_idx = first_valid as usize + period - 1;
        assert!(
            warm_idx < series_len,
            "insufficient data for period {}",
            period
        );
        alphas.push(1.0f32 / (period as f32));
        warm.push(warm_idx as i32);
    }

    let d_prices = DeviceBuffer::from_slice(&data_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods).unwrap();
    let d_alphas = DeviceBuffer::from_slice(&alphas).unwrap();
    let d_warm = DeviceBuffer::from_slice(&warm).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();

    WildersBatchState {
        cuda,
        d_prices,
        d_periods,
        d_alphas,
        d_warm,
        d_out,
        series_len: series_len as i32,
        n_combos: n_combos as i32,
        first_valid,
    }
}

fn launch_wilders_batch_many_periods(state: &mut WildersBatchState) {
    state
        .cuda
        .wilders_batch_device(
            &state.d_prices,
            &state.d_periods,
            &state.d_alphas,
            &state.d_warm,
            state.series_len,
            state.first_valid,
            state.n_combos,
            &mut state.d_out,
        )
        .unwrap();
}

fn wilders_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("wilders_cuda_one_series_many_periods");
    group.sample_size(10);
    let mut state = prep_wilders_batch_many_periods();
    group.bench_with_input(BenchmarkId::from_parameter("200k_x_periods"), &0, |b, _| {
        b.iter(|| {
            launch_wilders_batch_many_periods(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct VamaBatchState {
    cuda: CudaVama,
    d_prices: DeviceBuffer<f32>,
    d_base: DeviceBuffer<i32>,
    d_vol: DeviceBuffer<i32>,
    d_alphas: DeviceBuffer<f32>,
    d_betas: DeviceBuffer<f32>,
    d_ema: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    series_len: i32,
    n_combos: i32,
    first_valid: i32,
}

fn prep_vama_batch_many_params() -> VamaBatchState {
    fn axis((start, end, step): (usize, usize, usize)) -> Vec<usize> {
        if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        }
    }

    let cuda = CudaVama::new(0).expect("cuda vama");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let base_axis = axis((9, 96, 5));
    let vol_axis = axis((5, 45, 5));

    let mut base_periods = Vec::with_capacity(base_axis.len() * vol_axis.len());
    let mut vol_periods = Vec::with_capacity(base_axis.len() * vol_axis.len());
    let mut alphas = Vec::with_capacity(base_axis.len() * vol_axis.len());
    let mut betas = Vec::with_capacity(base_axis.len() * vol_axis.len());
    for &b in &base_axis {
        for &v in &vol_axis {
            base_periods.push(b as i32);
            vol_periods.push(v as i32);
            let alpha = 2.0f32 / (b as f32 + 1.0f32);
            alphas.push(alpha);
            betas.push(1.0f32 - alpha);
        }
    }
    let n_combos = base_periods.len();
    let first_valid = data_f32.iter().position(|v| !v.is_nan()).unwrap_or(0) as i32;

    let d_prices = DeviceBuffer::from_slice(&data_f32).unwrap();
    let d_base = DeviceBuffer::from_slice(&base_periods).unwrap();
    let d_vol = DeviceBuffer::from_slice(&vol_periods).unwrap();
    let d_alphas = DeviceBuffer::from_slice(&alphas).unwrap();
    let d_betas = DeviceBuffer::from_slice(&betas).unwrap();
    let d_ema: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();

    VamaBatchState {
        cuda,
        d_prices,
        d_base,
        d_vol,
        d_alphas,
        d_betas,
        d_ema,
        d_out,
        series_len: series_len as i32,
        n_combos: n_combos as i32,
        first_valid,
    }
}

fn launch_vama_batch_many_params(state: &mut VamaBatchState) {
    state
        .cuda
        .vama_batch_device(
            &state.d_prices,
            &state.d_base,
            &state.d_vol,
            &state.d_alphas,
            &state.d_betas,
            state.series_len as usize,
            state.first_valid as usize,
            state.n_combos as usize,
            &mut state.d_ema,
            &mut state.d_out,
        )
        .unwrap();
}

fn vama_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("vama_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_vama_batch_many_params();
    group.bench_with_input(
        BenchmarkId::from_parameter("200k_x_param_pairs"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_vama_batch_many_params(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct VamaManySeriesState {
    cuda: CudaVama,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_ema: DeviceBuffer<f32>,
    d_out_tm: DeviceBuffer<f32>,
    alpha: f32,
    beta: f32,
    base_period: i32,
    vol_period: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_vama_many_series_one_param() -> VamaManySeriesState {
    let cuda = CudaVama::new(0).expect("cuda vama");
    let num_series = 1024usize;
    let series_len = 50_000usize;
    let data_tm_f32 = gen_time_major_f32(num_series, series_len);

    let base_period = 21usize;
    let vol_period = 13usize;
    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = None;
        for t in 0..series_len {
            let value = data_tm_f32[t * num_series + series];
            if value.is_finite() {
                fv = Some(t);
                break;
            }
        }
        let idx = fv.unwrap_or(series_len);
        assert!(idx < series_len, "series {} all NaN", series);
        assert!(
            series_len - idx >= base_period.max(vol_period),
            "series {} insufficient data",
            series
        );
        first_valids.push(idx as i32);
    }

    let alpha = 2.0f32 / (base_period as f32 + 1.0f32);
    let beta = 1.0f32 - alpha;

    let d_prices_tm = DeviceBuffer::from_slice(&data_tm_f32).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_ema: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    VamaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_ema,
        d_out_tm,
        alpha,
        beta,
        base_period: base_period as i32,
        vol_period: vol_period as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_vama_many_series_one_param(state: &mut VamaManySeriesState) {
    state
        .cuda
        .vama_many_series_one_param_device(
            &state.d_prices_tm,
            &state.d_first_valids,
            state.base_period,
            state.vol_period,
            state.alpha,
            state.beta,
            state.num_series,
            state.series_len,
            &mut state.d_ema,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn vama_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("vama_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_vama_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("50k_len_x_1k_series"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_vama_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct TilsonBatchState {
    cuda: CudaTilson,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_ks: DeviceBuffer<f32>,
    d_c1: DeviceBuffer<f32>,
    d_c2: DeviceBuffer<f32>,
    d_c3: DeviceBuffer<f32>,
    d_c4: DeviceBuffer<f32>,
    d_lookbacks: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    first_valid: usize,
}

fn prep_tilson_one_series_many_params() -> TilsonBatchState {
    let cuda = CudaTilson::new(0).expect("cuda tilson");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap_or(0);
    let prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let mut periods_i32 = Vec::new();
    let mut ks_f32 = Vec::new();
    let mut c1_f32 = Vec::new();
    let mut c2_f32 = Vec::new();
    let mut c3_f32 = Vec::new();
    let mut c4_f32 = Vec::new();
    let mut lookbacks_i32 = Vec::new();

    let volume_factors: Vec<f32> = (0..=8).map(|i| i as f32 * 0.1f32).collect();

    for period in (5usize..=101usize).step_by(4) {
        let lookback = 6 * (period.saturating_sub(1));
        if first_valid + lookback >= series_len {
            continue;
        }
        let k = 2.0f32 / (period as f32 + 1.0f32);
        for &vf in &volume_factors {
            let temp = vf * vf;
            let c1 = -(temp * vf);
            let c2 = 3.0f32 * (temp - c1);
            let c3 = -6.0f32 * temp - 3.0f32 * (vf - c1);
            let c4 = 1.0f32 + 3.0f32 * vf - c1 + 3.0f32 * temp;

            periods_i32.push(period as i32);
            ks_f32.push(k);
            c1_f32.push(c1);
            c2_f32.push(c2);
            c3_f32.push(c3);
            c4_f32.push(c4);
            lookbacks_i32.push((lookback) as i32);
        }
    }

    let n_combos = periods_i32.len();
    assert!(n_combos > 0, "no tilson combos generated for bench");

    let d_prices = DeviceBuffer::from_slice(&prices_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_ks = DeviceBuffer::from_slice(&ks_f32).unwrap();
    let d_c1 = DeviceBuffer::from_slice(&c1_f32).unwrap();
    let d_c2 = DeviceBuffer::from_slice(&c2_f32).unwrap();
    let d_c3 = DeviceBuffer::from_slice(&c3_f32).unwrap();
    let d_c4 = DeviceBuffer::from_slice(&c4_f32).unwrap();
    let d_lookbacks = DeviceBuffer::from_slice(&lookbacks_i32).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();

    TilsonBatchState {
        cuda,
        d_prices,
        d_periods,
        d_ks,
        d_c1,
        d_c2,
        d_c3,
        d_c4,
        d_lookbacks,
        d_out,
        series_len,
        n_combos,
        first_valid,
    }
}

fn launch_tilson_one_series_many_params(state: &mut TilsonBatchState) {
    state
        .cuda
        .tilson_batch_device(
            &state.d_prices,
            &state.d_periods,
            &state.d_ks,
            &state.d_c1,
            &state.d_c2,
            &state.d_c3,
            &state.d_c4,
            &state.d_lookbacks,
            state.series_len,
            state.first_valid,
            state.n_combos,
            &mut state.d_out,
        )
        .unwrap();
}

fn tilson_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("tilson_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_tilson_one_series_many_params();
    group.bench_with_input(
        BenchmarkId::from_parameter("200k_x_param_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_tilson_one_series_many_params(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct TilsonManySeriesState {
    cuda: CudaTilson,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    k: f32,
    c1: f32,
    c2: f32,
    c3: f32,
    c4: f32,
    lookback: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_tilson_many_series_one_param() -> TilsonManySeriesState {
    let cuda = CudaTilson::new(0).expect("cuda tilson");
    let num_series = 1024usize;
    let series_len = 50_000usize;
    let data_tm_f32 = gen_time_major_f32(num_series, series_len);

    let period = 15usize;
    let volume_factor = 0.35f64;
    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = None;
        for t in 0..series_len {
            let value = data_tm_f32[t * num_series + series];
            if value.is_finite() {
                fv = Some(t);
                break;
            }
        }
        let idx = fv.unwrap_or(series_len);
        assert!(idx < series_len, "series {} all NaN", series);
        assert!(
            series_len - idx > 6 * (period.saturating_sub(1)),
            "series {} insufficient data",
            series
        );
        first_valids.push(idx as i32);
    }

    let k = 2.0f32 / (period as f32 + 1.0f32);
    let vf = volume_factor as f32;
    let temp = vf * vf;
    let c1 = -(temp * vf);
    let c2 = 3.0f32 * (temp - c1);
    let c3 = -6.0f32 * temp - 3.0f32 * (vf - c1);
    let c4 = 1.0f32 + 3.0f32 * vf - c1 + 3.0f32 * temp;
    let lookback = (6 * (period.saturating_sub(1))) as i32;

    let d_prices_tm = DeviceBuffer::from_slice(&data_tm_f32).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    TilsonManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        period: period as i32,
        k,
        c1,
        c2,
        c3,
        c4,
        lookback,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_tilson_many_series_one_param(state: &mut TilsonManySeriesState) {
    state
        .cuda
        .tilson_many_series_one_param_device(
            &state.d_prices_tm,
            &state.d_first_valids,
            state.period as usize,
            state.k,
            state.c1,
            state.c2,
            state.c3,
            state.c4,
            state.lookback,
            state.num_series as usize,
            state.series_len as usize,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn tilson_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("tilson_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_tilson_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("50k_len_x_1k_series"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_tilson_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct SamaBatchState {
    cuda: CudaSama,
    d_prices: DeviceBuffer<f32>,
    d_lengths: DeviceBuffer<i32>,
    d_min_alphas: DeviceBuffer<f32>,
    d_maj_alphas: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
}

fn prep_sama_one_series_many_params() -> SamaBatchState {
    let cuda = CudaSama::new(0).expect("cuda sama");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap_or(0);
    let prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let lengths: Vec<usize> = (32..=160).step_by(16).collect();
    let maj_lengths: Vec<usize> = (10..=34).step_by(6).collect();
    let min_lengths: Vec<usize> = (4..=16).step_by(4).collect();
    let n_combos = lengths.len() * maj_lengths.len() * min_lengths.len();

    let mut lengths_i32 = Vec::with_capacity(n_combos);
    let mut min_alphas = Vec::with_capacity(n_combos);
    let mut maj_alphas = Vec::with_capacity(n_combos);
    let mut first_valids = Vec::with_capacity(n_combos);

    for &length in &lengths {
        for &maj in &maj_lengths {
            for &min in &min_lengths {
                lengths_i32.push(length as i32);
                min_alphas.push(2.0f32 / (min as f32 + 1.0f32));
                maj_alphas.push(2.0f32 / (maj as f32 + 1.0f32));
                first_valids.push(first_valid as i32);
            }
        }
    }

    let d_prices = DeviceBuffer::from_slice(&prices_f32).unwrap();
    let d_lengths = DeviceBuffer::from_slice(&lengths_i32).unwrap();
    let d_min = DeviceBuffer::from_slice(&min_alphas).unwrap();
    let d_maj = DeviceBuffer::from_slice(&maj_alphas).unwrap();
    let d_first = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();

    SamaBatchState {
        cuda,
        d_prices,
        d_lengths,
        d_min_alphas: d_min,
        d_maj_alphas: d_maj,
        d_first_valids: d_first,
        d_out,
        series_len,
        n_combos,
    }
}

fn launch_sama_one_series_many_params(state: &mut SamaBatchState) {
    state
        .cuda
        .sama_batch_device(
            &state.d_prices,
            &state.d_lengths,
            &state.d_min_alphas,
            &state.d_maj_alphas,
            &state.d_first_valids,
            state.series_len,
            state.n_combos,
            &mut state.d_out,
        )
        .unwrap();
}

fn sama_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("sama_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_sama_one_series_many_params();
    group.bench_with_input(
        BenchmarkId::from_parameter("200k_x_param_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_sama_one_series_many_params(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct SamaManySeriesState {
    cuda: CudaSama,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    length: i32,
    min_alpha: f32,
    maj_alpha: f32,
    num_series: i32,
    series_len: i32,
}

fn prep_sama_many_series_one_param() -> SamaManySeriesState {
    let cuda = CudaSama::new(0).expect("cuda sama");
    let num_series = 1024usize;
    let series_len = 50_000usize;
    let data_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = None;
        for t in 0..series_len {
            let value = data_tm[t * num_series + series];
            if value.is_finite() {
                fv = Some(t);
                break;
            }
        }
        let idx = fv.unwrap_or(series_len);
        assert!(idx < series_len, "series {} all NaN", series);
        first_valids.push(idx as i32);
    }

    let length = 64i32;
    let maj_length = 18usize;
    let min_length = 8usize;
    let min_alpha = 2.0f32 / (min_length as f32 + 1.0f32);
    let maj_alpha = 2.0f32 / (maj_length as f32 + 1.0f32);

    let d_prices_tm = DeviceBuffer::from_slice(&data_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    SamaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        length,
        min_alpha,
        maj_alpha,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_sama_many_series_one_param(state: &mut SamaManySeriesState) {
    state
        .cuda
        .sama_many_series_one_param_device(
            &state.d_prices_tm,
            &state.d_first_valids,
            state.length,
            state.min_alpha,
            state.maj_alpha,
            state.num_series,
            state.series_len,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn sama_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("sama_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_sama_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("50k_len_x_1k_series"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_sama_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct HighPass2BatchState {
    cuda: CudaHighPass2,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_c: DeviceBuffer<f32>,
    d_cm2: DeviceBuffer<f32>,
    d_two: DeviceBuffer<f32>,
    d_neg: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    first_valid: usize,
    n_combos: usize,
}

fn prep_highpass2_one_series_many_params() -> HighPass2BatchState {
    let cuda = CudaHighPass2::new(0).expect("cuda highpass2");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap_or(0);
    let prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let sweep = HighPass2BatchRange {
        period: (6, 120, 6),
        k: (0.2, 0.9, 0.1),
    };
    let combos = expand_highpass2_grid(&sweep);
    let n_combos = combos.len();

    let mut periods_i32 = Vec::with_capacity(n_combos);
    let mut c_vals = Vec::with_capacity(n_combos);
    let mut cm2_vals = Vec::with_capacity(n_combos);
    let mut two_vals = Vec::with_capacity(n_combos);
    let mut neg_vals = Vec::with_capacity(n_combos);

    for params in &combos {
        let period = params.period.unwrap();
        let k = params.k.unwrap();
        debug_assert!(series_len - first_valid >= period);
        let (c, cm2, two_1m, neg_oma_sq) = compute_highpass2_coefficients(period, k);
        periods_i32.push(period as i32);
        c_vals.push(c);
        cm2_vals.push(cm2);
        two_vals.push(two_1m);
        neg_vals.push(neg_oma_sq);
    }

    let d_prices = DeviceBuffer::from_slice(&prices_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_c = DeviceBuffer::from_slice(&c_vals).unwrap();
    let d_cm2 = DeviceBuffer::from_slice(&cm2_vals).unwrap();
    let d_two = DeviceBuffer::from_slice(&two_vals).unwrap();
    let d_neg = DeviceBuffer::from_slice(&neg_vals).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();

    HighPass2BatchState {
        cuda,
        d_prices,
        d_periods,
        d_c,
        d_cm2,
        d_two,
        d_neg,
        d_out,
        series_len,
        first_valid,
        n_combos,
    }
}

fn launch_highpass2_one_series_many_params(state: &mut HighPass2BatchState) {
    state
        .cuda
        .highpass2_batch_device(
            &state.d_prices,
            &state.d_periods,
            &state.d_c,
            &state.d_cm2,
            &state.d_two,
            &state.d_neg,
            state.series_len,
            state.n_combos,
            state.first_valid,
            &mut state.d_out,
        )
        .unwrap();
}

fn highpass2_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("highpass2_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_highpass2_one_series_many_params();
    group.bench_with_input(
        BenchmarkId::from_parameter("200k_x_param_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_highpass2_one_series_many_params(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct HighPass2ManySeriesState {
    cuda: CudaHighPass2,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    c: f32,
    cm2: f32,
    two_1m: f32,
    neg_oma_sq: f32,
    num_series: usize,
    series_len: usize,
}

fn prep_highpass2_many_series_one_param() -> HighPass2ManySeriesState {
    let cuda = CudaHighPass2::new(0).expect("cuda highpass2");
    let num_series = 1024usize;
    let series_len = 50_000usize;
    let data_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = None;
        for t in 0..series_len {
            let value = data_tm[t * num_series + series];
            if value.is_finite() {
                fv = Some(t as i32);
                break;
            }
        }
        first_valids.push(fv.expect("series all NaN"));
    }

    let period = 48usize;
    let k = 0.707f64;
    let (c, cm2, two_1m, neg_oma_sq) = compute_highpass2_coefficients(period, k);

    let d_prices_tm = DeviceBuffer::from_slice(&data_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    HighPass2ManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        period: period as i32,
        c,
        cm2,
        two_1m,
        neg_oma_sq,
        num_series,
        series_len,
    }
}

fn launch_highpass2_many_series_one_param(state: &mut HighPass2ManySeriesState) {
    state
        .cuda
        .highpass2_many_series_one_param_time_major_device(
            &state.d_prices_tm,
            &state.d_first_valids,
            state.period,
            state.c,
            state.cm2,
            state.two_1m,
            state.neg_oma_sq,
            state.num_series,
            state.series_len,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn highpass2_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("highpass2_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_highpass2_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("50k_len_x_1k_series"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_highpass2_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct SrwmaBatchState {
    cuda: CudaSrwma,
    d_prices: DeviceBuffer<f32>,
    d_weights: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_warm: DeviceBuffer<i32>,
    d_inv: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    max_wlen: usize,
    n_combos: usize,
}

fn prep_srwma_one_series_many_params() -> SrwmaBatchState {
    let cuda = CudaSrwma::new(0).expect("cuda srwma");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap_or(0);
    let prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let periods: Vec<usize> = (5..=101).step_by(4).collect();
    let max_wlen = periods.iter().map(|p| p - 1).max().unwrap_or(1);
    let n_combos = periods.len();

    let mut weights_flat = vec![0f32; n_combos * max_wlen];
    let mut periods_i32 = Vec::with_capacity(n_combos);
    let mut warm_indices = Vec::with_capacity(n_combos);
    let mut inv_norms = Vec::with_capacity(n_combos);

    for (idx, &period) in periods.iter().enumerate() {
        let wlen = period - 1;
        let mut norm = 0f32;
        for k in 0..wlen {
            let weight = ((period - k) as f32).sqrt();
            weights_flat[idx * max_wlen + k] = weight;
            norm += weight;
        }
        periods_i32.push(period as i32);
        warm_indices.push((first_valid + period + 1) as i32);
        inv_norms.push(1.0f32 / norm);
    }

    let d_prices = DeviceBuffer::from_slice(&prices_f32).unwrap();
    let d_weights = DeviceBuffer::from_slice(&weights_flat).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_warm = DeviceBuffer::from_slice(&warm_indices).unwrap();
    let d_inv = DeviceBuffer::from_slice(&inv_norms).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();

    SrwmaBatchState {
        cuda,
        d_prices,
        d_weights,
        d_periods,
        d_warm,
        d_inv,
        d_out,
        series_len,
        max_wlen,
        n_combos,
    }
}

fn launch_srwma_one_series_many_params(state: &mut SrwmaBatchState) {
    state
        .cuda
        .srwma_batch_device(
            &state.d_prices,
            &state.d_weights,
            &state.d_periods,
            &state.d_warm,
            &state.d_inv,
            state.series_len,
            0,
            state.max_wlen,
            state.n_combos,
            &mut state.d_out,
        )
        .unwrap();
}

fn srwma_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("srwma_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_srwma_one_series_many_params();
    group.bench_with_input(
        BenchmarkId::from_parameter("200k_x_param_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_srwma_one_series_many_params(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct SrwmaManySeriesState {
    cuda: CudaSrwma,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_weights: DeviceBuffer<f32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    inv_norm: f32,
    num_series: i32,
    series_len: i32,
}

fn prep_srwma_many_series_one_param() -> SrwmaManySeriesState {
    let cuda = CudaSrwma::new(0).expect("cuda srwma");
    let num_series = 1024usize;
    let series_len = 50_000usize;
    let data_tm_f32 = gen_time_major_f32(num_series, series_len);

    let period = 21usize;
    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = None;
        for t in 0..series_len {
            let value = data_tm_f32[t * num_series + series];
            if value.is_finite() {
                fv = Some(t);
                break;
            }
        }
        let idx = fv.unwrap_or(series_len);
        assert!(idx < series_len, "series {} all NaN", series);
        assert!(
            series_len - idx >= period + 1,
            "series {} insufficient data",
            series
        );
        first_valids.push(idx as i32);
    }

    let wlen = period - 1;
    let mut weights = Vec::with_capacity(wlen);
    let mut norm = 0f32;
    for k in 0..wlen {
        let weight = ((period - k) as f32).sqrt();
        weights.push(weight);
        norm += weight;
    }
    let inv_norm = 1.0f32 / norm;

    let d_prices_tm = DeviceBuffer::from_slice(&data_tm_f32).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_weights = DeviceBuffer::from_slice(&weights).unwrap();
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    SrwmaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_weights,
        d_out_tm,
        period: period as i32,
        inv_norm,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_srwma_many_series_one_param(state: &mut SrwmaManySeriesState) {
    state
        .cuda
        .srwma_many_series_one_param_device(
            &state.d_prices_tm,
            &state.d_first_valids,
            &state.d_weights,
            state.period,
            state.inv_norm,
            state.num_series,
            state.series_len,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn srwma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("srwma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_srwma_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("50k_len_x_1k_series"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_srwma_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct MwdxBatchState {
    cuda: CudaMwdx,
    d_prices: DeviceBuffer<f32>,
    d_factors: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    first_valid: usize,
    n_combos: usize,
}

fn expand_mwdx_factors(range: (f64, f64, f64)) -> Vec<f32> {
    let (start, end, step) = range;
    if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
        return vec![start as f32];
    }
    let mut out = Vec::new();
    let mut value = start;
    while value <= end + 1e-12 {
        out.push(value as f32);
        value += step;
    }
    out
}

fn prep_mwdx_one_series_many_params() -> MwdxBatchState {
    let cuda = CudaMwdx::new(0).expect("cuda mwdx");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let first_valid = data
        .iter()
        .position(|v| !v.is_nan())
        .expect("series must contain finite values");
    let prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let sweep = MwdxBatchRange {
        factor: (0.05, 0.95, 0.05),
    };
    let factors = expand_mwdx_factors(sweep.factor);
    let n_combos = factors.len();

    let d_prices = DeviceBuffer::from_slice(&prices_f32).unwrap();
    let d_factors = DeviceBuffer::from_slice(&factors).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();

    MwdxBatchState {
        cuda,
        d_prices,
        d_factors,
        d_out,
        series_len,
        first_valid,
        n_combos,
    }
}

fn launch_mwdx_one_series_many_params(state: &mut MwdxBatchState) {
    state
        .cuda
        .mwdx_batch_device(
            &state.d_prices,
            &state.d_factors,
            state.series_len,
            state.first_valid,
            state.n_combos,
            &mut state.d_out,
        )
        .unwrap();
}

fn mwdx_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("mwdx_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_mwdx_one_series_many_params();
    group.bench_with_input(
        BenchmarkId::from_parameter("200k_x_factor_sweep"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_mwdx_one_series_many_params(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct MwdxManySeriesState {
    cuda: CudaMwdx,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    factor: f32,
    num_series: usize,
    series_len: usize,
}

fn prep_mwdx_many_series_one_param() -> MwdxManySeriesState {
    let cuda = CudaMwdx::new(0).expect("cuda mwdx");
    let num_series = 1024usize;
    let series_len = 50_000usize;
    let data_tm_f32 = gen_time_major_f32(num_series, series_len);

    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = None;
        for t in 0..series_len {
            let value = data_tm_f32[t * num_series + series];
            if value.is_finite() {
                fv = Some(t as i32);
                break;
            }
        }
        let idx = fv.expect("each series must contain finite values");
        first_valids.push(idx);
    }

    let factor = 0.27f32;

    let d_prices_tm = DeviceBuffer::from_slice(&data_tm_f32).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    MwdxManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        factor,
        num_series,
        series_len,
    }
}

fn launch_mwdx_many_series_one_param(state: &mut MwdxManySeriesState) {
    state
        .cuda
        .mwdx_many_series_one_param_device(
            &state.d_prices_tm,
            &state.d_first_valids,
            state.factor,
            state.num_series,
            state.series_len,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn mwdx_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("mwdx_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_mwdx_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("50k_len_x_1k_series"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_mwdx_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

fn expand_jsa_periods(range: (usize, usize, usize)) -> Vec<i32> {
    let (start, end, step) = range;
    if start > end {
        return Vec::new();
    }
    if step == 0 || start == end {
        return vec![start as i32];
    }
    let mut out = Vec::new();
    let mut value = start;
    while value <= end {
        out.push(value as i32);
        value = value.saturating_add(step);
    }
    out
}

struct JsaBatchState {
    cuda: CudaJsa,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_warm: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    first_valid: usize,
    n_combos: usize,
}

fn prep_jsa_one_series_many_params() -> JsaBatchState {
    let cuda = CudaJsa::new(0).expect("cuda jsa");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let first_valid = data
        .iter()
        .position(|v| !v.is_nan())
        .expect("series must contain finite values");

    let sweep = JsaBatchRange {
        period: (8, 128, 8),
    };
    let periods = expand_jsa_periods(sweep.period);
    let n_combos = periods.len();
    assert!(n_combos > 0, "empty JSA period sweep");

    let mut warm_indices = Vec::with_capacity(n_combos);
    for &period_i in &periods {
        let period = period_i as usize;
        assert!(
            series_len - first_valid >= period,
            "period {} requires more valid samples",
            period
        );
        warm_indices.push((first_valid + period) as i32);
    }

    let prices_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let d_prices = DeviceBuffer::from_slice(&prices_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods).unwrap();
    let d_warm = DeviceBuffer::from_slice(&warm_indices).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();

    JsaBatchState {
        cuda,
        d_prices,
        d_periods,
        d_warm,
        d_out,
        series_len,
        first_valid,
        n_combos,
    }
}

fn launch_jsa_one_series_many_params(state: &mut JsaBatchState) {
    state
        .cuda
        .jsa_batch_device(
            &state.d_prices,
            &state.d_periods,
            &state.d_warm,
            state.series_len,
            state.first_valid,
            state.n_combos,
            &mut state.d_out,
        )
        .unwrap();
}

fn jsa_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("jsa_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_jsa_one_series_many_params();
    group.bench_with_input(
        BenchmarkId::from_parameter("200k_x_period_sweep"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_jsa_one_series_many_params(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct JsaManySeriesState {
    cuda: CudaJsa,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_warm: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    num_series: usize,
    series_len: usize,
}

fn prep_jsa_many_series_one_param() -> JsaManySeriesState {
    let cuda = CudaJsa::new(0).expect("cuda jsa");
    let num_series = 1024usize;
    let series_len = 50_000usize;
    let data_tm_f32 = gen_time_major_f32(num_series, series_len);

    let period = 32usize;
    let mut first_valids = Vec::with_capacity(num_series);
    let mut warm_indices = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = None;
        for t in 0..series_len {
            let value = data_tm_f32[t * num_series + series];
            if value.is_finite() {
                fv = Some(t);
                break;
            }
        }
        let fv = fv.expect("each series must contain finite values");
        assert!(
            series_len - fv >= period,
            "series {} needs at least {} valid samples",
            series,
            period
        );
        first_valids.push(fv as i32);
        warm_indices.push((fv + period) as i32);
    }

    let d_prices_tm = DeviceBuffer::from_slice(&data_tm_f32).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_warm = DeviceBuffer::from_slice(&warm_indices).unwrap();
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    JsaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_warm,
        d_out_tm,
        period: period as i32,
        num_series,
        series_len,
    }
}

fn launch_jsa_many_series_one_param(state: &mut JsaManySeriesState) {
    state
        .cuda
        .jsa_many_series_one_param_device(
            &state.d_prices_tm,
            &state.d_first_valids,
            &state.d_warm,
            state.period,
            state.num_series,
            state.series_len,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn jsa_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("jsa_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_jsa_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("50k_len_x_1k_series"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_jsa_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct WillrBatchState {
    cuda: CudaWillr,
    d_close: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_log2: DeviceBuffer<i32>,
    d_offsets: DeviceBuffer<i32>,
    d_st_max: DeviceBuffer<f32>,
    d_st_min: DeviceBuffer<f32>,
    d_nan_psum: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: i32,
    n_combos: i32,
    first_valid: i32,
}

fn prep_willr_batch_many_periods() -> WillrBatchState {
    let cuda = CudaWillr::new(0).expect("cuda willr");
    let series_len = 200_000usize;
    let base = gen_series(series_len);

    let mut high_f32 = Vec::with_capacity(series_len);
    let mut low_f32 = Vec::with_capacity(series_len);
    let mut close_f32 = Vec::with_capacity(series_len);
    for &val in &base {
        if val.is_nan() {
            high_f32.push(f32::NAN);
            low_f32.push(f32::NAN);
            close_f32.push(f32::NAN);
        } else {
            let v = val as f32;
            high_f32.push(v + 0.9f32);
            low_f32.push(v - 0.8f32);
            close_f32.push(v);
        }
    }

    let sweep = WillrBatchRange {
        period: (9, 240, 3),
    };
    let periods: Vec<i32> = (sweep.period.0..=sweep.period.1)
        .step_by(sweep.period.2.max(1))
        .map(|p| p as i32)
        .collect();
    let n_combos = periods.len();
    let first_valid = close_f32.iter().position(|v| !v.is_nan()).unwrap_or(0) as i32;

    let tables = build_willr_gpu_tables(&high_f32, &low_f32);
    let d_close = DeviceBuffer::from_slice(&close_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods).unwrap();
    let d_log2 = DeviceBuffer::from_slice(&tables.log2).unwrap();
    let d_offsets = DeviceBuffer::from_slice(&tables.level_offsets).unwrap();
    let d_st_max = DeviceBuffer::from_slice(&tables.st_max).unwrap();
    let d_st_min = DeviceBuffer::from_slice(&tables.st_min).unwrap();
    let d_nan_psum = DeviceBuffer::from_slice(&tables.nan_psum).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();

    WillrBatchState {
        cuda,
        d_close,
        d_periods,
        d_log2,
        d_offsets,
        d_st_max,
        d_st_min,
        d_nan_psum,
        d_out,
        series_len: series_len as i32,
        n_combos: n_combos as i32,
        first_valid,
    }
}

fn launch_willr_batch_many_periods(state: &mut WillrBatchState) {
    state
        .cuda
        .willr_batch_device(
            &state.d_close,
            &state.d_periods,
            &state.d_log2,
            &state.d_offsets,
            &state.d_st_max,
            &state.d_st_min,
            &state.d_nan_psum,
            state.series_len,
            state.first_valid,
            state.n_combos,
            &mut state.d_out,
        )
        .unwrap();
}

fn willr_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("willr_cuda_one_series_many_periods");
    group.sample_size(10);
    let mut state = prep_willr_batch_many_periods();
    group.bench_with_input(BenchmarkId::from_parameter("200k_x_periods"), &0, |b, _| {
        b.iter(|| {
            launch_willr_batch_many_periods(&mut state);
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
    dema_batch_bench,
    ema_batch_bench,
    ema_many_series_bench,
    ehlers_one_series_many_params_bench,
    ehlers_many_series_one_param_bench,
    vama_batch_bench,
    vama_many_series_bench,
    tilson_batch_bench,
    tilson_many_series_bench,
    sama_batch_bench,
    sama_many_series_bench,
    highpass2_batch_bench,
    highpass2_many_series_bench,
    srwma_batch_bench,
    srwma_many_series_bench,
    mwdx_batch_bench,
    mwdx_many_series_bench,
    jsa_batch_bench,
    jsa_many_series_bench,
    wilders_batch_bench,
    willr_batch_bench
);
criterion_main!(benches);
