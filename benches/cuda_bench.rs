#![cfg(feature = "cuda")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cust::memory::DeviceBuffer;

use my_project::cuda::moving_averages::{
    CudaAlma, CudaCwma, CudaEhlersEcema, CudaEpma, CudaHighpass, CudaKama, CudaNama,     CudaSinwma, CudaSupersmoother3Pole, CudaTradjema, CudaVama, CudaWma,
};
use my_project::cuda::{cuda_available, CudaWto};
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

fn gen_series(len: usize) -> Vec<f32> {
    let mut v = vec![f32::NAN; len];
    for i in 3..len {
        let x = i as f32;
        v[i] = (x * 0.001).sin() + 0.0001 * x;
    }
    v
}

fn gen_volume(len: usize) -> Vec<f32> {
    let mut v = vec![f32::NAN; len];
    for i in 5..len {
        let x = i as f32;
        v[i] = ((x * 0.007).cos().abs() + 1.2) * 950.0;
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

fn gen_time_major_volume_f32(num_series: usize, series_len: usize) -> Vec<f32> {
    let mut v = vec![f32::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in j..series_len {
            let base = (t as f32) + (j as f32) * 0.2;
            v[t * num_series + j] = ((base * 0.008).sin().abs() + 0.9) * (300.0 + 20.0 * j as f32);
        }
    }
    v
}

fn gen_tradjema_ohlc(series_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut high = vec![f32::NAN; series_len];
    let mut low = vec![f32::NAN; series_len];
    let mut close = vec![f32::NAN; series_len];
    for i in 8..series_len {
        let t = i as f32;
        let trend = 0.0004 * t;
        let cycle = (t * 0.0023f32).sin() * 0.7;
        let base = trend + cycle;
        close[i] = base;
        high[i] = base + 0.28f32 + 0.012f32 * ((i % 11) as f32);
        low[i] = base - 0.30f32 - 0.015f32 * ((i % 7) as f32);
    }
    (high, low, close)
}

fn gen_tradjema_ohlc_tm(num_series: usize, series_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut high = vec![f32::NAN; num_series * series_len];
    let mut low = vec![f32::NAN; num_series * series_len];
    let mut close = vec![f32::NAN; num_series * series_len];
    for series in 0..num_series {
        let scale = 1.0 + 0.04 * series as f32;
        for t in (series + 6)..series_len {
            let time = t as f32;
            let wave = (time * 0.0035f32 + series as f32 * 0.25).sin();
            let drift = 0.00035f32 * time + 0.02 * series as f32;
            let base = (drift + wave * 0.6f32) * scale;
            let idx = t * num_series + series;
            close[idx] = base;
            high[idx] = base + 0.26f32 + 0.01f32 * series as f32;
            low[idx] = base - 0.29f32 - 0.008f32 * series as f32;
        }
    }
    (high, low, close)
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
    let m = offset * (period as f64 - 1.0);
    let s = (period as f64) / sigma;
    let s2 = 2.0 * s * s;
    let mut w = vec![0.0f32; period];
    let mut norm = 0.0f64;
    for i in 0..period {
        let diff = i as f64 - m;
        let wi = (-(diff * diff) / s2).exp();
        w[i] = wi as f32;
        norm += wi;
    }
    (w, (1.0 / norm) as f32)
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

fn build_vama_prefixes(prices: &[f32], volumes: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut prefix_vol = Vec::with_capacity(prices.len());
    let mut prefix_price_vol = Vec::with_capacity(prices.len());
    let mut accum_vol = 0.0f32;
    let mut accum_price_vol = 0.0f32;
    for (&p, &v) in prices.iter().zip(volumes.iter()) {
        let vol_nz = if v.is_nan() { 0.0f32 } else { v };
        let price_nz = if p.is_nan() { 0.0f32 } else { p };
        accum_vol += vol_nz;
        accum_price_vol += price_nz * vol_nz;
        prefix_vol.push(accum_vol);
        prefix_price_vol.push(accum_price_vol);
    }
    (prefix_vol, prefix_price_vol)
}

fn build_vama_prefixes_tm(
    prices_tm: &[f32],
    volumes_tm: &[f32],
    cols: usize,
    rows: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut prefix_vol = vec![0.0f32; cols * rows];
    let mut prefix_price_vol = vec![0.0f32; cols * rows];
    for series in 0..cols {
        let mut accum_vol = 0.0f32;
        let mut accum_price_vol = 0.0f32;
        for t in 0..rows {
            let idx = t * cols + series;
            let vol = volumes_tm[idx];
            let price = prices_tm[idx];
            let vol_nz = if vol.is_nan() { 0.0f32 } else { vol };
            let price_nz = if price.is_nan() { 0.0f32 } else { price };
            accum_vol += vol_nz;
            accum_price_vol += price_nz * vol_nz;
            prefix_vol[idx] = accum_vol;
            prefix_price_vol[idx] = accum_price_vol;
        }
    }
    (prefix_vol, prefix_price_vol)
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
    cwma_one_series_bench,
    cwma_many_series_bench,
    epma_one_series_bench,
    epma_many_series_bench,
    wma_one_series_bench,
    wma_many_series_bench,
    highpass_one_series_bench,
    highpass_many_series_bench,
    kama_one_series_bench,
    kama_many_series_bench,
    ehlers_ecema_one_series_bench,
    ehlers_ecema_many_series_bench,
    sinwma_one_series_bench,
    sinwma_many_series_bench,
    nama_one_series_bench,
    nama_many_series_bench,
    supersmoother3_pole_one_series_bench,
    supersmoother3_pole_many_series_bench,
    vama_one_series_bench,
    vama_many_series_bench,
    tradjema_one_series_bench,
    tradjema_many_series_bench,
    wto_one_series_bench,
    wto_many_series_bench
);
criterion_main!(benches);

// ──────────────────────────────────────────────────────────────
// CWMA: one series × many params
// ──────────────────────────────────────────────────────────────
struct CwmaOneSeriesState {
    cuda: CudaCwma,
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

fn prep_cwma_one_series_many_params() -> CwmaOneSeriesState {
    let cuda = CudaCwma::new(0).expect("cuda cwma");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let host_prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let periods: Vec<usize> = (6..=96).step_by(2).collect();
    let n_combos = periods.len();
    let max_period = periods.iter().copied().max().unwrap();

    let mut periods_i32 = vec![0i32; n_combos];
    let mut inv_norms = vec![0f32; n_combos];
    let mut weights_flat = vec![0f32; n_combos * max_period];
    for (idx, &p) in periods.iter().enumerate() {
        periods_i32[idx] = p as i32;
        let weight_len = p - 1;
        let mut norm = 0.0f32;
        for k in 0..weight_len {
            let w = ((p - k) as f32).powi(3);
            weights_flat[idx * max_period + k] = w;
            norm += w;
        }
        inv_norms[idx] = 1.0 / norm;
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

    CwmaOneSeriesState {
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

fn launch_cwma_one_series_many_params(state: &mut CwmaOneSeriesState) {
    state
        .cuda
        .cwma_batch_device(
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

fn cwma_one_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let series_len = 200_000usize;
    let n_combos = ((96 - 6) / 2 + 1) as usize;
    let max_period = 96usize;
    let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
    let weights_bytes = n_combos * max_period * std::mem::size_of::<f32>();
    let input_bytes = series_len * std::mem::size_of::<f32>();
    let approx = out_bytes + weights_bytes + input_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip cwma 200k x ~46 (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }
    let mut group = c.benchmark_group("cwma_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_cwma_one_series_many_params();
    group.bench_with_input(BenchmarkId::from_parameter("200k_x_periods"), &0, |b, _| {
        b.iter(|| {
            launch_cwma_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// CWMA: many series × one param (time-major)
// ──────────────────────────────────────────────────────────────
struct CwmaManySeriesState {
    cuda: CudaCwma,
    d_prices: DeviceBuffer<f32>,
    d_weights: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    period: i32,
    num_series: i32,
    series_len: i32,
    inv_norm: f32,
}

fn prep_cwma_many_series_one_param() -> CwmaManySeriesState {
    let cuda = CudaCwma::new(0).expect("cuda cwma");
    let num_series = 512usize;
    let series_len = 16_384usize;
    let period = 40usize;

    let mut data_tm = vec![f32::NAN; num_series * series_len];
    for series in 0..num_series {
        for t in (series / 2)..series_len {
            let x = (t as f32) + (series as f32) * 0.03;
            data_tm[t * num_series + series] = (x * 0.0025).sin() + 0.0001 * x;
        }
    }

    let mut first_valids = vec![0i32; num_series];
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len && data_tm[fv * num_series + series].is_nan() {
            fv += 1;
        }
        first_valids[series] = fv as i32;
    }

    let weight_len = period - 1;
    let mut weights = vec![0f32; weight_len];
    let mut norm = 0.0f32;
    for k in 0..weight_len {
        let w = ((period - k) as f32).powi(3);
        weights[k] = w;
        norm += w;
    }
    let inv_norm = 1.0 / norm;

    let d_prices = DeviceBuffer::from_slice(&data_tm).unwrap();
    let d_weights = DeviceBuffer::from_slice(&weights).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    CwmaManySeriesState {
        cuda,
        d_prices,
        d_weights,
        d_first_valids,
        d_out,
        period: period as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
        inv_norm,
    }
}

fn launch_cwma_many_series_one_param(state: &mut CwmaManySeriesState) {
    state
        .cuda
        .cwma_multi_series_one_param_device(
            &state.d_prices,
            &state.d_weights,
            state.period,
            state.inv_norm,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out,
        )
        .unwrap();
}

fn cwma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let num_series = 512usize;
    let series_len = 16_384usize;
    let period = 40usize;
    let total = num_series * series_len;
    let out_bytes = total * std::mem::size_of::<f32>();
    let weights_bytes = (period - 1) * std::mem::size_of::<f32>();
    let input_bytes = total * std::mem::size_of::<f32>();
    let approx = out_bytes + weights_bytes + input_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip cwma many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }
    let mut group = c.benchmark_group("cwma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_cwma_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("512series_x_16k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_cwma_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// WMA: one series × many params
// ──────────────────────────────────────────────────────────────
struct EpmaOneSeriesState {
    cuda: CudaEpma,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_offsets: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    first_valid: usize,
    max_period: usize,
}

fn prep_epma_one_series_many_params() -> EpmaOneSeriesState {
    const PERIOD_START: usize = 10;
    const PERIOD_END: usize = 160;
    const PERIOD_STEP: usize = 5;
    const OFFSET_START: usize = 1;
    const OFFSET_END: usize = 6;
    const OFFSET_STEP: usize = 1;

    let cuda = CudaEpma::new(0).expect("cuda epma");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let host_prices_f32: Vec<f32> = data;

    let periods: Vec<usize> = (PERIOD_START..=PERIOD_END).step_by(PERIOD_STEP).collect();
    let offsets: Vec<usize> = (OFFSET_START..=OFFSET_END).step_by(OFFSET_STEP).collect();
    let n_combos = periods.len() * offsets.len();
    let max_period = *periods.last().unwrap();

    let mut periods_i32 = Vec::with_capacity(n_combos);
    let mut offsets_i32 = Vec::with_capacity(n_combos);
    for &p in &periods {
        for &o in &offsets {
            periods_i32.push(p as i32);
            offsets_i32.push(o as i32);
        }
    }

    let first_valid = host_prices_f32
        .iter()
        .position(|v| !v.is_nan())
        .unwrap_or(0);

    let d_prices = DeviceBuffer::from_slice(&host_prices_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_offsets = DeviceBuffer::from_slice(&offsets_i32).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.unwrap();

    EpmaOneSeriesState {
        cuda,
        d_prices,
        d_periods,
        d_offsets,
        d_out,
        series_len,
        n_combos,
        first_valid,
        max_period,
    }
}

fn launch_epma_one_series_many_params(state: &mut EpmaOneSeriesState) {
    state
        .cuda
        .epma_batch_device(
            &state.d_prices,
            &state.d_periods,
            &state.d_offsets,
            state.series_len,
            state.n_combos,
            state.first_valid,
            state.max_period,
            &mut state.d_out,
        )
        .unwrap();
}

fn epma_one_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    const PERIOD_START: usize = 10;
    const PERIOD_END: usize = 160;
    const PERIOD_STEP: usize = 5;
    const OFFSET_START: usize = 1;
    const OFFSET_END: usize = 6;
    const OFFSET_STEP: usize = 1;

    let period_count = ((PERIOD_END - PERIOD_START) / PERIOD_STEP) + 1;
    let offset_count = ((OFFSET_END - OFFSET_START) / OFFSET_STEP) + 1;
    let n_combos = period_count * offset_count;
    let series_len = 200_000usize;
    let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
    let periods_bytes = n_combos * std::mem::size_of::<i32>();
    let offsets_bytes = n_combos * std::mem::size_of::<i32>();
    let input_bytes = series_len * std::mem::size_of::<f32>();
    let approx = out_bytes + periods_bytes + offsets_bytes + input_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip epma 200k x {} (need ~{} MB, free ~{} MB)",
                n_combos,
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("epma_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_epma_one_series_many_params();
    group.bench_with_input(BenchmarkId::from_parameter("200k_grid"), &0, |b, _| {
        b.iter(|| {
            launch_epma_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct EpmaManySeriesState {
    cuda: CudaEpma,
    d_prices: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    period: i32,
    offset: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_epma_many_series_one_param() -> EpmaManySeriesState {
    let cuda = CudaEpma::new(0).expect("cuda epma");
    let num_series = 512usize;
    let series_len = 16_384usize;
    let period = 48usize;
    let offset = 8usize;

    let data_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = vec![0i32; num_series];
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len && data_tm[fv * num_series + series].is_nan() {
            fv += 1;
        }
        if fv == series_len {
            panic!("series {} all NaN in epma bench", series);
        }
        first_valids[series] = fv as i32;
    }

    let d_prices = DeviceBuffer::from_slice(&data_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    EpmaManySeriesState {
        cuda,
        d_prices,
        d_first_valids,
        d_out,
        period: period as i32,
        offset: offset as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_epma_many_series_one_param(state: &mut EpmaManySeriesState) {
    state
        .cuda
        .epma_many_series_one_param_time_major_device(
            &state.d_prices,
            state.period,
            state.offset,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out,
        )
        .unwrap();
}

fn epma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 512usize;
    let series_len = 16_384usize;
    let total = num_series * series_len;
    let out_bytes = total * std::mem::size_of::<f32>();
    let input_bytes = total * std::mem::size_of::<f32>();
    let first_bytes = num_series * std::mem::size_of::<i32>();
    let approx = out_bytes + input_bytes + first_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip epma many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("epma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_epma_many_series_one_param();
    group.bench_with_input(BenchmarkId::from_parameter("512x16k"), &0, |b, _| {
        b.iter(|| {
            launch_epma_many_series_one_param(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct WmaOneSeriesState {
    cuda: CudaWma,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: i32,
    n_combos: i32,
    first_valid: i32,
    max_period: i32,
}

fn prep_wma_one_series_many_params() -> WmaOneSeriesState {
    const PERIOD_START: usize = 4;
    const PERIOD_END: usize = 192;
    const PERIOD_STEP: usize = 2;

    let cuda = CudaWma::new(0).expect("cuda wma");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let host_prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let periods: Vec<usize> = (PERIOD_START..=PERIOD_END).step_by(PERIOD_STEP).collect();
    let n_combos = periods.len();
    let max_period = *periods.last().unwrap();
    let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();

    let first_valid = host_prices_f32
        .iter()
        .position(|v| !v.is_nan())
        .unwrap_or(0) as i32;

    let d_prices = DeviceBuffer::from_slice(&host_prices_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.unwrap();

    WmaOneSeriesState {
        cuda,
        d_prices,
        d_periods,
        d_out,
        series_len: series_len as i32,
        n_combos: n_combos as i32,
        first_valid,
        max_period: max_period as i32,
    }
}

fn launch_wma_one_series_many_params(state: &mut WmaOneSeriesState) {
    state
        .cuda
        .wma_batch_device(
            &state.d_prices,
            &state.d_periods,
            state.series_len,
            state.n_combos,
            state.first_valid,
            state.max_period,
            &mut state.d_out,
        )
        .unwrap();
}

fn wma_one_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    const PERIOD_START: usize = 4;
    const PERIOD_END: usize = 192;
    const PERIOD_STEP: usize = 2;
    let series_len = 200_000usize;
    let n_combos = ((PERIOD_END - PERIOD_START) / PERIOD_STEP) + 1;
    let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
    let periods_bytes = n_combos * std::mem::size_of::<i32>();
    let input_bytes = series_len * std::mem::size_of::<f32>();
    let approx = out_bytes + periods_bytes + input_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip wma 200k x ~{} (need ~{} MB, free ~{} MB)",
                n_combos,
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("wma_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_wma_one_series_many_params();
    group.bench_with_input(BenchmarkId::from_parameter("200k_x_periods"), &0, |b, _| {
        b.iter(|| {
            launch_wma_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// WMA: many series × one param
// ──────────────────────────────────────────────────────────────
struct WmaManySeriesState {
    cuda: CudaWma,
    d_prices: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    period: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_wma_many_series_one_param() -> WmaManySeriesState {
    let cuda = CudaWma::new(0).expect("cuda wma");
    let num_series = 512usize;
    let series_len = 16_384usize;
    let period = 64usize;

    let data_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = vec![0i32; num_series];
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len && data_tm[fv * num_series + series].is_nan() {
            fv += 1;
        }
        first_valids[series] = fv as i32;
    }

    let d_prices = DeviceBuffer::from_slice(&data_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    WmaManySeriesState {
        cuda,
        d_prices,
        d_first_valids,
        d_out,
        period: period as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_wma_many_series_one_param(state: &mut WmaManySeriesState) {
    state
        .cuda
        .wma_multi_series_one_param_time_major_device(
            &state.d_prices,
            state.period,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out,
        )
        .unwrap();
}

fn wma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 512usize;
    let series_len = 16_384usize;
    let total = num_series * series_len;
    let out_bytes = total * std::mem::size_of::<f32>();
    let input_bytes = total * std::mem::size_of::<f32>();
    let first_bytes = num_series * std::mem::size_of::<i32>();
    let approx = out_bytes + input_bytes + first_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip wma many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("wma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_wma_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("512series_x_16k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_wma_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// HighPass: one series × many params
// ──────────────────────────────────────────────────────────────
const HIGHPASS_PERIOD_START: usize = 8;
const HIGHPASS_PERIOD_END: usize = 160;
const HIGHPASS_PERIOD_STEP: usize = 4;
const HIGHPASS_SERIES_LEN: usize = 200_000;

struct HighpassOneSeriesState {
    cuda: CudaHighpass,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: i32,
    n_combos: i32,
}

fn prep_highpass_one_series_many_params() -> HighpassOneSeriesState {
    let cuda = CudaHighpass::new(0).expect("cuda highpass");
    let data = gen_series(HIGHPASS_SERIES_LEN);
    let host_prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let periods: Vec<usize> = (HIGHPASS_PERIOD_START..=HIGHPASS_PERIOD_END)
        .step_by(HIGHPASS_PERIOD_STEP)
        .collect();
    let n_combos = periods.len();
    let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();

    let d_prices = DeviceBuffer::from_slice(&host_prices_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * HIGHPASS_SERIES_LEN) }.unwrap();

    HighpassOneSeriesState {
        cuda,
        d_prices,
        d_periods,
        d_out,
        series_len: HIGHPASS_SERIES_LEN as i32,
        n_combos: n_combos as i32,
    }
}

fn launch_highpass_one_series_many_params(state: &mut HighpassOneSeriesState) {
    state
        .cuda
        .highpass_batch_device(
            &state.d_prices,
            &state.d_periods,
            state.series_len,
            state.n_combos,
            &mut state.d_out,
        )
        .unwrap();
}

fn highpass_one_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let n_combos = ((HIGHPASS_PERIOD_END - HIGHPASS_PERIOD_START) / HIGHPASS_PERIOD_STEP) + 1;
    let out_bytes = n_combos * HIGHPASS_SERIES_LEN * std::mem::size_of::<f32>();
    let periods_bytes = n_combos * std::mem::size_of::<i32>();
    let prices_bytes = HIGHPASS_SERIES_LEN * std::mem::size_of::<f32>();
    let approx = out_bytes + periods_bytes + prices_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip highpass 200k x ~{} (need ~{} MB, free ~{} MB)",
                n_combos,
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("highpass_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_highpass_one_series_many_params();
    group.bench_with_input(BenchmarkId::from_parameter("200k_x_periods"), &0, |b, _| {
        b.iter(|| {
            launch_highpass_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// HighPass: many series × one param
// ──────────────────────────────────────────────────────────────
const HIGHPASS_NUM_SERIES: usize = 512;
const HIGHPASS_MANY_SERIES_LEN: usize = 16_384;
const HIGHPASS_PERIOD: usize = 48;

struct HighpassManySeriesState {
    cuda: CudaHighpass,
    d_prices: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    num_series: i32,
    series_len: i32,
    period: i32,
}

fn prep_highpass_many_series_one_param() -> HighpassManySeriesState {
    let cuda = CudaHighpass::new(0).expect("cuda highpass");
    let data_tm = gen_time_major_f32(HIGHPASS_NUM_SERIES, HIGHPASS_MANY_SERIES_LEN);

    let d_prices = DeviceBuffer::from_slice(&data_tm).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(HIGHPASS_NUM_SERIES * HIGHPASS_MANY_SERIES_LEN) }
            .unwrap();

    HighpassManySeriesState {
        cuda,
        d_prices,
        d_out,
        num_series: HIGHPASS_NUM_SERIES as i32,
        series_len: HIGHPASS_MANY_SERIES_LEN as i32,
        period: HIGHPASS_PERIOD as i32,
    }
}

fn launch_highpass_many_series_one_param(state: &mut HighpassManySeriesState) {
    state
        .cuda
        .highpass_many_series_one_param_time_major_device(
            &state.d_prices,
            state.period,
            state.num_series,
            state.series_len,
            &mut state.d_out,
        )
        .unwrap();
}

fn highpass_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let total = HIGHPASS_NUM_SERIES * HIGHPASS_MANY_SERIES_LEN;
    let out_bytes = total * std::mem::size_of::<f32>();
    let input_bytes = total * std::mem::size_of::<f32>();
    let approx = out_bytes + input_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip highpass many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("highpass_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_highpass_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("512series_x_16k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_highpass_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// KAMA: one series × many params
// ──────────────────────────────────────────────────────────────
struct KamaOneSeriesState {
    cuda: CudaKama,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: i32,
    n_combos: i32,
    first_valid: i32,
    max_period: i32,
}

fn prep_kama_one_series_many_params() -> KamaOneSeriesState {
    const PERIOD_START: usize = 5;
    const PERIOD_END: usize = 160;
    const PERIOD_STEP: usize = 5;

    let cuda = CudaKama::new(0).expect("cuda kama");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let host_prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let periods: Vec<usize> = (PERIOD_START..=PERIOD_END).step_by(PERIOD_STEP).collect();
    let n_combos = periods.len();
    let max_period = *periods.last().unwrap();
    let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();

    let first_valid = host_prices_f32
        .iter()
        .position(|v| !v.is_nan())
        .unwrap_or(0) as i32;

    let d_prices = DeviceBuffer::from_slice(&host_prices_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.unwrap();

    KamaOneSeriesState {
        cuda,
        d_prices,
        d_periods,
        d_out,
        series_len: series_len as i32,
        n_combos: n_combos as i32,
        first_valid,
        max_period: max_period as i32,
    }
}

fn launch_kama_one_series_many_params(state: &mut KamaOneSeriesState) {
    state
        .cuda
        .kama_batch_device(
            &state.d_prices,
            &state.d_periods,
            state.series_len,
            state.n_combos,
            state.first_valid,
            state.max_period,
            &mut state.d_out,
        )
        .unwrap();
}

fn kama_one_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    const PERIOD_START: usize = 5;
    const PERIOD_END: usize = 160;
    const PERIOD_STEP: usize = 5;
    let series_len = 200_000usize;
    let n_combos = ((PERIOD_END - PERIOD_START) / PERIOD_STEP) + 1;
    let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
    let periods_bytes = n_combos * std::mem::size_of::<i32>();
    let input_bytes = series_len * std::mem::size_of::<f32>();
    let approx = out_bytes + periods_bytes + input_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip kama 200k x ~{} (need ~{} MB, free ~{} MB)",
                n_combos,
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("kama_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_kama_one_series_many_params();
    group.bench_with_input(BenchmarkId::from_parameter("200k_x_periods"), &0, |b, _| {
        b.iter(|| {
            launch_kama_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// KAMA: many series × one param
// ──────────────────────────────────────────────────────────────
struct KamaManySeriesState {
    cuda: CudaKama,
    d_prices: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    period: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_kama_many_series_one_param() -> KamaManySeriesState {
    let cuda = CudaKama::new(0).expect("cuda kama");
    let num_series = 512usize;
    let series_len = 16_384usize;
    let period = 42usize;

    let data_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = vec![0i32; num_series];
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len && data_tm[fv * num_series + series].is_nan() {
            fv += 1;
        }
        first_valids[series] = fv as i32;
    }

    let d_prices = DeviceBuffer::from_slice(&data_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    KamaManySeriesState {
        cuda,
        d_prices,
        d_first_valids,
        d_out,
        period: period as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_kama_many_series_one_param(state: &mut KamaManySeriesState) {
    state
        .cuda
        .kama_many_series_one_param_time_major_device(
            &state.d_prices,
            state.period,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out,
        )
        .unwrap();
}

fn kama_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 512usize;
    let series_len = 16_384usize;
    let total = num_series * series_len;
    let out_bytes = total * std::mem::size_of::<f32>();
    let input_bytes = total * std::mem::size_of::<f32>();
    let first_bytes = num_series * std::mem::size_of::<i32>();
    let approx = out_bytes + input_bytes + first_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip kama many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("kama_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_kama_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("512series_x_16k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_kama_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// Ehlers ECEMA: one series × many params
// ──────────────────────────────────────────────────────────────
struct EcemaOneSeriesState {
    cuda: CudaEhlersEcema,
    d_prices: DeviceBuffer<f32>,
    d_lengths: DeviceBuffer<i32>,
    d_gain_limits: DeviceBuffer<i32>,
    d_pine: DeviceBuffer<u8>,
    d_confirmed: DeviceBuffer<u8>,
    d_out: DeviceBuffer<f32>,
    series_len: i32,
    n_combos: i32,
    first_valid: i32,
}

fn prep_ecema_one_series_many_params() -> EcemaOneSeriesState {
    const LENGTH_START: usize = 6;
    const LENGTH_END: usize = 126;
    const LENGTH_STEP: usize = 6;
    const GAIN_START: usize = 10;
    const GAIN_END: usize = 70;
    const GAIN_STEP: usize = 10;

    let cuda = CudaEhlersEcema::new(0).expect("cuda ecema");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let host_prices: Vec<f32> = data.iter().copied().collect();

    let lengths: Vec<usize> = (LENGTH_START..=LENGTH_END).step_by(LENGTH_STEP).collect();
    let gains: Vec<usize> = (GAIN_START..=GAIN_END).step_by(GAIN_STEP).collect();
    let n_combos = lengths.len() * gains.len();

    let mut lengths_i32 = Vec::with_capacity(n_combos);
    let mut gains_i32 = Vec::with_capacity(n_combos);
    for &len in &lengths {
        for &gain in &gains {
            lengths_i32.push(len as i32);
            gains_i32.push(gain as i32);
        }
    }
    let pine_flags = vec![0u8; n_combos];
    let confirmed_flags = vec![0u8; n_combos];

    let first_valid = host_prices
        .iter()
        .position(|v| !v.is_nan())
        .unwrap_or(0) as i32;

    let d_prices = DeviceBuffer::from_slice(&host_prices).unwrap();
    let d_lengths = DeviceBuffer::from_slice(&lengths_i32).unwrap();
    let d_gain_limits = DeviceBuffer::from_slice(&gains_i32).unwrap();
    let d_pine = DeviceBuffer::from_slice(&pine_flags).unwrap();
    let d_confirmed = DeviceBuffer::from_slice(&confirmed_flags).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.unwrap();

    EcemaOneSeriesState {
        cuda,
        d_prices,
        d_lengths,
        d_gain_limits,
        d_pine,
        d_confirmed,
        d_out,
        series_len: series_len as i32,
        n_combos: n_combos as i32,
        first_valid,
    }
}

fn launch_ecema_one_series_many_params(state: &mut EcemaOneSeriesState) {
    state
        .cuda
        .ehlers_ecema_batch_device(
            &state.d_prices,
            &state.d_lengths,
            &state.d_gain_limits,
            &state.d_pine,
            &state.d_confirmed,
            state.series_len,
            state.n_combos,
            state.first_valid,
            &mut state.d_out,
        )
        .unwrap();
}

fn ehlers_ecema_one_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    const LENGTH_START: usize = 6;
    const LENGTH_END: usize = 126;
    const LENGTH_STEP: usize = 6;
    const GAIN_START: usize = 10;
    const GAIN_END: usize = 70;
    const GAIN_STEP: usize = 10;
    let series_len = 200_000usize;
    let length_count = ((LENGTH_END - LENGTH_START) / LENGTH_STEP) + 1;
    let gain_count = ((GAIN_END - GAIN_START) / GAIN_STEP) + 1;
    let n_combos = length_count * gain_count;

    let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
    let length_bytes = n_combos * std::mem::size_of::<i32>();
    let gain_bytes = n_combos * std::mem::size_of::<i32>();
    let flag_bytes = n_combos * std::mem::size_of::<u8>() * 2;
    let input_bytes = series_len * std::mem::size_of::<f32>();
    let approx = out_bytes
        + length_bytes
        + gain_bytes
        + flag_bytes
        + input_bytes
        + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip ecema 200k x {} (need ~{} MB, free ~{} MB)",
                n_combos,
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("ecema_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_ecema_one_series_many_params();
    group.bench_with_input(BenchmarkId::from_parameter("200k_x_params"), &0, |b, _| {
        b.iter(|| {
            launch_ecema_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// Ehlers ECEMA: many series × one param
// ──────────────────────────────────────────────────────────────
struct EcemaManySeriesState {
    cuda: CudaEhlersEcema,
    d_prices: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    cols: i32,
    rows: i32,
    length: i32,
    gain_limit: i32,
    pine_flag: u8,
    confirmed_flag: u8,
}

fn prep_ecema_many_series_one_param() -> EcemaManySeriesState {
    let cuda = CudaEhlersEcema::new(0).expect("cuda ecema");
    let cols = 256usize;
    let rows = 32_768usize;
    let length = 26i32;
    let gain_limit = 50i32;

    let data_tm = gen_time_major_f32(cols, rows);
    let mut first_valids = vec![0i32; cols];
    for series in 0..cols {
        let mut fv = 0usize;
        while fv < rows && data_tm[fv * cols + series].is_nan() {
            fv += 1;
        }
        first_valids[series] = fv as i32;
    }

    let d_prices = DeviceBuffer::from_slice(&data_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(cols * rows) }.unwrap();

    EcemaManySeriesState {
        cuda,
        d_prices,
        d_first_valids,
        d_out,
        cols: cols as i32,
        rows: rows as i32,
        length,
        gain_limit,
        pine_flag: 0,
        confirmed_flag: 0,
    }
}

fn launch_ecema_many_series_one_param(state: &mut EcemaManySeriesState) {
    state
        .cuda
        .ehlers_ecema_many_series_one_param_time_major_device(
            &state.d_prices,
            state.cols,
            state.rows,
            state.length,
            state.gain_limit,
            state.pine_flag,
            state.confirmed_flag,
            &state.d_first_valids,
            &mut state.d_out,
        )
        .unwrap();
}

fn ehlers_ecema_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let cols = 256usize;
    let rows = 32_768usize;
    let input_bytes = cols * rows * std::mem::size_of::<f32>();
    let first_valid_bytes = cols * std::mem::size_of::<i32>();
    let out_bytes = cols * rows * std::mem::size_of::<f32>();
    let approx = input_bytes + first_valid_bytes + out_bytes + 16 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip ecema many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("ecema_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_ecema_many_series_one_param();
    group.bench_with_input(BenchmarkId::from_parameter("256x32768"), &0, |b, _| {
        b.iter(|| {
            launch_ecema_many_series_one_param(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// SINWMA: one series × many params
// ──────────────────────────────────────────────────────────────
struct SinwmaOneSeriesState {
    cuda: CudaSinwma,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: i32,
    n_combos: i32,
    first_valid: i32,
    max_period: i32,
}

fn prep_sinwma_one_series_many_params() -> SinwmaOneSeriesState {
    const PERIOD_START: usize = 4;
    const PERIOD_END: usize = 160;
    const PERIOD_STEP: usize = 2;

    let cuda = CudaSinwma::new(0).expect("cuda sinwma");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let host_prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let periods: Vec<usize> = (PERIOD_START..=PERIOD_END).step_by(PERIOD_STEP).collect();
    let n_combos = periods.len();
    let max_period = *periods.last().unwrap();
    let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();

    let first_valid = host_prices_f32
        .iter()
        .position(|v| !v.is_nan())
        .unwrap_or(0) as i32;

    let d_prices = DeviceBuffer::from_slice(&host_prices_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.unwrap();

    SinwmaOneSeriesState {
        cuda,
        d_prices,
        d_periods,
        d_out,
        series_len: series_len as i32,
        n_combos: n_combos as i32,
        first_valid,
        max_period: max_period as i32,
    }
}

fn launch_sinwma_one_series_many_params(state: &mut SinwmaOneSeriesState) {
    state
        .cuda
        .sinwma_batch_device(
            &state.d_prices,
            &state.d_periods,
            state.series_len,
            state.n_combos,
            state.first_valid,
            state.max_period,
            &mut state.d_out,
        )
        .unwrap();
}

fn sinwma_one_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    const PERIOD_START: usize = 4;
    const PERIOD_END: usize = 160;
    const PERIOD_STEP: usize = 2;
    let series_len = 200_000usize;
    let n_combos = ((PERIOD_END - PERIOD_START) / PERIOD_STEP) + 1;
    let out_bytes = n_combos * series_len * std::mem::size_of::<f32>();
    let periods_bytes = n_combos * std::mem::size_of::<i32>();
    let input_bytes = series_len * std::mem::size_of::<f32>();
    let approx = out_bytes + periods_bytes + input_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip sinwma 200k x ~{} (need ~{} MB, free ~{} MB)",
                n_combos,
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("sinwma_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_sinwma_one_series_many_params();
    group.bench_with_input(BenchmarkId::from_parameter("200k_x_periods"), &0, |b, _| {
        b.iter(|| {
            launch_sinwma_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// SINWMA: many series × one param
// ──────────────────────────────────────────────────────────────
struct SinwmaManySeriesState {
    cuda: CudaSinwma,
    d_prices: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    period: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_sinwma_many_series_one_param() -> SinwmaManySeriesState {
    let cuda = CudaSinwma::new(0).expect("cuda sinwma");
    let num_series = 512usize;
    let series_len = 16_384usize;
    let period = 48usize;

    let data_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = vec![0i32; num_series];
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len && data_tm[fv * num_series + series].is_nan() {
            fv += 1;
        }
        first_valids[series] = fv as i32;
    }

    let d_prices = DeviceBuffer::from_slice(&data_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    SinwmaManySeriesState {
        cuda,
        d_prices,
        d_first_valids,
        d_out,
        period: period as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_sinwma_many_series_one_param(state: &mut SinwmaManySeriesState) {
    state
        .cuda
        .sinwma_many_series_one_param_time_major_device(
            &state.d_prices,
            state.period,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out,
        )
        .unwrap();
}

fn sinwma_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let num_series = 512usize;
    let series_len = 16_384usize;
    let total = num_series * series_len;
    let out_bytes = total * std::mem::size_of::<f32>();
    let input_bytes = total * std::mem::size_of::<f32>();
    let first_bytes = num_series * std::mem::size_of::<i32>();
    let approx = out_bytes + input_bytes + first_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip sinwma many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }
    let mut group = c.benchmark_group("sinwma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_sinwma_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("512series_x_16k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_sinwma_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// NAMA: one series × many params
// ──────────────────────────────────────────────────────────────
const NAMA_PERIOD_START: usize = 6;
const NAMA_PERIOD_END: usize = 120;
const NAMA_PERIOD_STEP: usize = 3;
const NAMA_SERIES_LEN: usize = 200_000;

struct NamaOneSeriesState {
    cuda: CudaNama,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: i32,
    n_combos: i32,
    first_valid: i32,
    max_period: i32,
}

fn prep_nama_one_series_many_params() -> NamaOneSeriesState {
    let cuda = CudaNama::new(0).expect("cuda nama");
    let data = gen_series(NAMA_SERIES_LEN);
    let host_prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let periods: Vec<usize> = (NAMA_PERIOD_START..=NAMA_PERIOD_END)
        .step_by(NAMA_PERIOD_STEP)
        .collect();
    let n_combos = periods.len();
    let max_period = *periods.last().unwrap();
    let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();

    let first_valid = host_prices_f32
        .iter()
        .position(|v| !v.is_nan())
        .unwrap_or(0) as i32;

    let d_prices = DeviceBuffer::from_slice(&host_prices_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * NAMA_SERIES_LEN) }.unwrap();

    NamaOneSeriesState {
        cuda,
        d_prices,
        d_periods,
        d_out,
        series_len: NAMA_SERIES_LEN as i32,
        n_combos: n_combos as i32,
        first_valid,
        max_period: max_period as i32,
    }
}

fn launch_nama_one_series_many_params(state: &mut NamaOneSeriesState) {
    state
        .cuda
        .nama_batch_device(
            &state.d_prices,
            None::<&DeviceBuffer<f32>>,
            None::<&DeviceBuffer<f32>>,
            None::<&DeviceBuffer<f32>>,
            &state.d_periods,
            state.series_len,
            state.n_combos,
            state.first_valid,
            state.max_period,
            false,
            &mut state.d_out,
        )
        .unwrap();
}

fn nama_one_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let n_combos = ((NAMA_PERIOD_END - NAMA_PERIOD_START) / NAMA_PERIOD_STEP) + 1;
    let out_bytes = n_combos * NAMA_SERIES_LEN * std::mem::size_of::<f32>();
    let periods_bytes = n_combos * std::mem::size_of::<i32>();
    let input_bytes = NAMA_SERIES_LEN * std::mem::size_of::<f32>();
    let approx = out_bytes + periods_bytes + input_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip nama 200k x ~{} (need ~{} MB, free ~{} MB)",
                n_combos,
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("nama_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_nama_one_series_many_params();
    group.bench_with_input(BenchmarkId::from_parameter("200k_x_periods"), &0, |b, _| {
        b.iter(|| {
            launch_nama_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// NAMA: many series × one param
// ──────────────────────────────────────────────────────────────
const NAMA_NUM_SERIES: usize = 512;
const NAMA_MANY_SERIES_LEN: usize = 16_384;
const NAMA_PERIOD: usize = 40;

struct NamaManySeriesState {
    cuda: CudaNama,
    d_prices: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    num_series: i32,
    series_len: i32,
    period: i32,
}

fn prep_nama_many_series_one_param() -> NamaManySeriesState {
    let cuda = CudaNama::new(0).expect("cuda nama");
    let data_tm = gen_time_major_f32(NAMA_NUM_SERIES, NAMA_MANY_SERIES_LEN);

    let mut first_valids = vec![0i32; NAMA_NUM_SERIES];
    for series in 0..NAMA_NUM_SERIES {
        let mut fv = 0usize;
        while fv < NAMA_MANY_SERIES_LEN && data_tm[fv * NAMA_NUM_SERIES + series].is_nan() {
            fv += 1;
        }
        first_valids[series] = fv as i32;
    }

    let d_prices = DeviceBuffer::from_slice(&data_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(NAMA_NUM_SERIES * NAMA_MANY_SERIES_LEN) }.unwrap();

    NamaManySeriesState {
        cuda,
        d_prices,
        d_first_valids,
        d_out,
        num_series: NAMA_NUM_SERIES as i32,
        series_len: NAMA_MANY_SERIES_LEN as i32,
        period: NAMA_PERIOD as i32,
    }
}

fn launch_nama_many_series_one_param(state: &mut NamaManySeriesState) {
    state
        .cuda
        .nama_many_series_one_param_time_major_device(
            &state.d_prices,
            None::<&DeviceBuffer<f32>>,
            None::<&DeviceBuffer<f32>>,
            None::<&DeviceBuffer<f32>>,
            state.num_series,
            state.series_len,
            state.period,
            &state.d_first_valids,
            false,
            &mut state.d_out,
        )
        .unwrap();
}

fn nama_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let total = NAMA_NUM_SERIES * NAMA_MANY_SERIES_LEN;
    let out_bytes = total * std::mem::size_of::<f32>();
    let input_bytes = total * std::mem::size_of::<f32>();
    let first_bytes = NAMA_NUM_SERIES * std::mem::size_of::<i32>();
    let approx = out_bytes + input_bytes + first_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip nama many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("nama_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_nama_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("512series_x_16k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_nama_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// SuperSmoother 3-Pole: one series × many params
// ──────────────────────────────────────────────────────────────
const SS3P_PERIOD_START: usize = 6;
const SS3P_PERIOD_END: usize = 96;
const SS3P_PERIOD_STEP: usize = 2;
const SS3P_SERIES_LEN: usize = 200_000;

struct Supersmoother3PoleOneSeriesState {
    cuda: CudaSupersmoother3Pole,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    first_valid: usize,
}

fn prep_supersmoother3_pole_one_series_many_params() -> Supersmoother3PoleOneSeriesState {
    let cuda = CudaSupersmoother3Pole::new(0).expect("cuda supersmoother_3_pole");
    let data = gen_series(SS3P_SERIES_LEN);
    let host_prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let periods: Vec<usize> = (SS3P_PERIOD_START..=SS3P_PERIOD_END)
        .step_by(SS3P_PERIOD_STEP)
        .collect();
    let n_combos = periods.len();
    let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();

    let first_valid = host_prices_f32
        .iter()
        .position(|v| !v.is_nan())
        .unwrap_or(0);

    let d_prices = DeviceBuffer::from_slice(&host_prices_f32).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * SS3P_SERIES_LEN) }.unwrap();

    Supersmoother3PoleOneSeriesState {
        cuda,
        d_prices,
        d_periods,
        d_out,
        series_len: SS3P_SERIES_LEN,
        n_combos,
        first_valid,
    }
}

fn launch_supersmoother3_pole_one_series_many_params(state: &mut Supersmoother3PoleOneSeriesState) {
    state
        .cuda
        .supersmoother_3_pole_batch_device(
            &state.d_prices,
            &state.d_periods,
            state.series_len,
            state.n_combos,
            state.first_valid,
            &mut state.d_out,
        )
        .unwrap();
}

fn supersmoother3_pole_one_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let n_combos = ((SS3P_PERIOD_END - SS3P_PERIOD_START) / SS3P_PERIOD_STEP) + 1;
    let out_bytes = n_combos * SS3P_SERIES_LEN * std::mem::size_of::<f32>();
    let periods_bytes = n_combos * std::mem::size_of::<i32>();
    let input_bytes = SS3P_SERIES_LEN * std::mem::size_of::<f32>();
    let approx = out_bytes + periods_bytes + input_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip supersmoother3pole 200k x ~{} (need ~{} MB, free ~{} MB)",
                n_combos,
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("supersmoother3pole_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_supersmoother3_pole_one_series_many_params();
    group.bench_with_input(BenchmarkId::from_parameter("200k_x_periods"), &0, |b, _| {
        b.iter(|| {
            launch_supersmoother3_pole_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// SuperSmoother 3-Pole: many series × one param
// ──────────────────────────────────────────────────────────────
const SS3P_NUM_SERIES: usize = 256;
const SS3P_MANY_SERIES_LEN: usize = 16_384;
const SS3P_PERIOD: usize = 24;

struct Supersmoother3PoleManySeriesState {
    cuda: CudaSupersmoother3Pole,
    d_prices: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    cols: usize,
    rows: usize,
    period: usize,
}

fn prep_supersmoother3_pole_many_series_one_param() -> Supersmoother3PoleManySeriesState {
    let cuda = CudaSupersmoother3Pole::new(0).expect("cuda supersmoother_3_pole");
    let data_tm = gen_time_major_f32(SS3P_NUM_SERIES, SS3P_MANY_SERIES_LEN);

    let mut first_valids = vec![0i32; SS3P_NUM_SERIES];
    for series in 0..SS3P_NUM_SERIES {
        let mut fv = 0usize;
        while fv < SS3P_MANY_SERIES_LEN && data_tm[fv * SS3P_NUM_SERIES + series].is_nan() {
            fv += 1;
        }
        first_valids[series] = fv as i32;
    }

    let d_prices = DeviceBuffer::from_slice(&data_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(SS3P_NUM_SERIES * SS3P_MANY_SERIES_LEN) }.unwrap();

    Supersmoother3PoleManySeriesState {
        cuda,
        d_prices,
        d_first_valids,
        d_out,
        cols: SS3P_NUM_SERIES,
        rows: SS3P_MANY_SERIES_LEN,
        period: SS3P_PERIOD,
    }
}

fn launch_supersmoother3_pole_many_series_one_param(state: &mut Supersmoother3PoleManySeriesState) {
    state
        .cuda
        .supersmoother_3_pole_many_series_one_param_device(
            &state.d_prices,
            state.period,
            state.cols,
            state.rows,
            &state.d_first_valids,
            &mut state.d_out,
        )
        .unwrap();
}

fn supersmoother3_pole_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let total = SS3P_NUM_SERIES * SS3P_MANY_SERIES_LEN;
    let out_bytes = total * std::mem::size_of::<f32>();
    let input_bytes = total * std::mem::size_of::<f32>();
    let first_bytes = SS3P_NUM_SERIES * std::mem::size_of::<i32>();
    let approx = out_bytes + input_bytes + first_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip supersmoother3pole many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("supersmoother3pole_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_supersmoother3_pole_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("256series_x_16k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_supersmoother3_pole_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// VAMA: one series × many params
// ──────────────────────────────────────────────────────────────
const VAMA_LENGTHS: [usize; 8] = [6, 12, 18, 24, 30, 36, 42, 48];
const VAMA_VI_FACTORS: [f32; 5] = [0.5, 0.65, 0.8, 0.95, 1.1];
const VAMA_SAMPLE_PERIODS: [i32; 3] = [0, 8, 16];
const VAMA_SERIES_LEN: usize = 100_000;

struct VamaOneSeriesState {
    cuda: CudaVama,
    d_prices: DeviceBuffer<f32>,
    d_volumes: DeviceBuffer<f32>,
    d_prefix_volumes: DeviceBuffer<f32>,
    d_prefix_price_volumes: DeviceBuffer<f32>,
    d_lengths: DeviceBuffer<i32>,
    d_vi_factors: DeviceBuffer<f32>,
    d_sample_periods: DeviceBuffer<i32>,
    d_strict_flags: DeviceBuffer<u8>,
    d_out: DeviceBuffer<f32>,
    series_len: i32,
    n_combos: i32,
    first_valid: i32,
}

fn prep_vama_one_series_many_params() -> VamaOneSeriesState {
    let cuda = CudaVama::new(0).expect("cuda vama");
    let prices = gen_series(VAMA_SERIES_LEN);
    let volumes = gen_volume(VAMA_SERIES_LEN);
    let prices_f32: Vec<f32> = prices.iter().map(|&x| x as f32).collect();
    let volumes_f32: Vec<f32> = volumes.iter().map(|&x| x as f32).collect();
    let (prefix_vol, prefix_price_vol) = build_vama_prefixes(&prices_f32, &volumes_f32);

    let mut lengths_i32 = Vec::new();
    let mut vi_factors_f32 = Vec::new();
    let mut sample_periods_i32 = Vec::new();
    let mut strict_flags = Vec::new();
    for &len in &VAMA_LENGTHS {
        for &vf in &VAMA_VI_FACTORS {
            for &sp in &VAMA_SAMPLE_PERIODS {
                for &strict in &[true, false] {
                    lengths_i32.push(len as i32);
                    vi_factors_f32.push(vf);
                    sample_periods_i32.push(sp);
                    strict_flags.push(if strict { 1u8 } else { 0u8 });
                }
            }
        }
    }
    let n_combos = lengths_i32.len();
    let first_valid = prices_f32.iter().position(|v| !v.is_nan()).unwrap_or(0) as i32;

    let d_prices = DeviceBuffer::from_slice(&prices_f32).unwrap();
    let d_volumes = DeviceBuffer::from_slice(&volumes_f32).unwrap();
    let d_prefix_volumes = DeviceBuffer::from_slice(&prefix_vol).unwrap();
    let d_prefix_price_volumes = DeviceBuffer::from_slice(&prefix_price_vol).unwrap();
    let d_lengths = DeviceBuffer::from_slice(&lengths_i32).unwrap();
    let d_vi_factors = DeviceBuffer::from_slice(&vi_factors_f32).unwrap();
    let d_sample_periods = DeviceBuffer::from_slice(&sample_periods_i32).unwrap();
    let d_strict_flags = DeviceBuffer::from_slice(&strict_flags).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * VAMA_SERIES_LEN) }.unwrap();

    VamaOneSeriesState {
        cuda,
        d_prices,
        d_volumes,
        d_prefix_volumes,
        d_prefix_price_volumes,
        d_lengths,
        d_vi_factors,
        d_sample_periods,
        d_strict_flags,
        d_out,
        series_len: VAMA_SERIES_LEN as i32,
        n_combos: n_combos as i32,
        first_valid,
    }
}

fn launch_vama_one_series_many_params(state: &mut VamaOneSeriesState) {
    state
        .cuda
        .vama_batch_device(
            &state.d_prices,
            &state.d_volumes,
            &state.d_prefix_volumes,
            &state.d_prefix_price_volumes,
            &state.d_lengths,
            &state.d_vi_factors,
            &state.d_sample_periods,
            &state.d_strict_flags,
            state.series_len,
            state.n_combos,
            state.first_valid,
            &mut state.d_out,
        )
        .unwrap();
}

fn vama_one_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let n_combos = VAMA_LENGTHS.len() * VAMA_VI_FACTORS.len() * VAMA_SAMPLE_PERIODS.len() * 2;
    let out_bytes = n_combos * VAMA_SERIES_LEN * std::mem::size_of::<f32>();
    let param_bytes = n_combos
        * (std::mem::size_of::<i32>() * 2 + std::mem::size_of::<f32>() + std::mem::size_of::<u8>());
    let prefix_bytes = 2 * VAMA_SERIES_LEN * std::mem::size_of::<f32>();
    let input_bytes = 2 * VAMA_SERIES_LEN * std::mem::size_of::<f32>();
    let approx = out_bytes + param_bytes + prefix_bytes + input_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip vama one-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("vama_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_vama_one_series_many_params();
    group.bench_with_input(BenchmarkId::from_parameter("100k_x_grid"), &0, |b, _| {
        b.iter(|| {
            launch_vama_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

struct VamaManySeriesState {
    cuda: CudaVama,
    d_prices_tm: DeviceBuffer<f32>,
    d_volumes_tm: DeviceBuffer<f32>,
    d_prefix_volumes_tm: DeviceBuffer<f32>,
    d_prefix_price_volumes_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    vi_factor: f32,
    sample_period: i32,
    strict: bool,
    num_series: i32,
    series_len: i32,
}

fn prep_vama_many_series_one_param() -> VamaManySeriesState {
    let cuda = CudaVama::new(0).expect("cuda vama");
    let num_series = 256usize;
    let series_len = 16_384usize;
    let period = 24usize;
    let vi_factor = 0.66f32;
    let sample_period = 16usize;
    let strict = false;

    let prices_tm = gen_time_major_f32(num_series, series_len);
    let volumes_tm = gen_time_major_volume_f32(num_series, series_len);
    let (prefix_vol, prefix_price_vol) =
        build_vama_prefixes_tm(&prices_tm, &volumes_tm, num_series, series_len);

    let mut first_valids = vec![0i32; num_series];
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len && prices_tm[fv * num_series + series].is_nan() {
            fv += 1;
        }
        first_valids[series] = fv as i32;
    }

    let d_prices_tm = DeviceBuffer::from_slice(&prices_tm).unwrap();
    let d_volumes_tm = DeviceBuffer::from_slice(&volumes_tm).unwrap();
    let d_prefix_volumes_tm = DeviceBuffer::from_slice(&prefix_vol).unwrap();
    let d_prefix_price_volumes_tm = DeviceBuffer::from_slice(&prefix_price_vol).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    VamaManySeriesState {
        cuda,
        d_prices_tm,
        d_volumes_tm,
        d_prefix_volumes_tm,
        d_prefix_price_volumes_tm,
        d_first_valids,
        d_out_tm,
        period: period as i32,
        vi_factor,
        sample_period: sample_period as i32,
        strict,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_vama_many_series_one_param(state: &mut VamaManySeriesState) {
    state
        .cuda
        .vama_multi_series_one_param_time_major_device(
            &state.d_prices_tm,
            &state.d_volumes_tm,
            &state.d_prefix_volumes_tm,
            &state.d_prefix_price_volumes_tm,
            state.period,
            state.vi_factor,
            state.sample_period,
            state.strict,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn vama_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 256usize;
    let series_len = 16_384usize;
    let total = num_series * series_len;
    let base_bytes = 2 * total * std::mem::size_of::<f32>();
    let prefix_bytes = 2 * total * std::mem::size_of::<f32>();
    let first_bytes = num_series * std::mem::size_of::<i32>();
    let out_bytes = total * std::mem::size_of::<f32>();
    let approx = base_bytes + prefix_bytes + first_bytes + out_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip vama many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("vama_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_vama_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("256series_x_16k"),
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

// ──────────────────────────────────────────────────────────────
// WTO: one series × many params
// ──────────────────────────────────────────────────────────────
struct WtoOneSeriesState {
    cuda: CudaWto,
    d_prices: DeviceBuffer<f32>,
    d_channel: DeviceBuffer<i32>,
    d_average: DeviceBuffer<i32>,
    d_wt1: DeviceBuffer<f32>,
    d_wt2: DeviceBuffer<f32>,
    d_hist: DeviceBuffer<f32>,
    series_len: i32,
    n_combos: i32,
    first_valid: i32,
}

fn prep_wto_one_series_many_params() -> WtoOneSeriesState {
    let cuda = CudaWto::new(0).expect("cuda wto");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let prices_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();

    let channels: Vec<usize> = (8..=32).step_by(2).collect();
    let averages: Vec<usize> = (18..=38).step_by(4).collect();
    let n_combos = channels.len() * averages.len();

    let mut channel_i32 = Vec::with_capacity(n_combos);
    let mut average_i32 = Vec::with_capacity(n_combos);
    for &ch in &channels {
        for &av in &averages {
            channel_i32.push(ch as i32);
            average_i32.push(av as i32);
        }
    }

    let d_prices = DeviceBuffer::from_slice(&prices_f32).unwrap();
    let d_channel = DeviceBuffer::from_slice(&channel_i32).unwrap();
    let d_average = DeviceBuffer::from_slice(&average_i32).unwrap();
    let mut d_wt1: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();
    let mut d_wt2: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();
    let mut d_hist: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len * n_combos) }.unwrap();

    let first_valid = prices_f32.iter().position(|v| !v.is_nan()).unwrap_or(0) as i32;

    WtoOneSeriesState {
        cuda,
        d_prices,
        d_channel,
        d_average,
        d_wt1,
        d_wt2,
        d_hist,
        series_len: series_len as i32,
        n_combos: n_combos as i32,
        first_valid,
    }
}

fn launch_wto_one_series_many_params(state: &mut WtoOneSeriesState) {
    state
        .cuda
        .wto_batch_device(
            &state.d_prices,
            &state.d_channel,
            &state.d_average,
            state.series_len,
            state.n_combos,
            state.first_valid,
            &mut state.d_wt1,
            &mut state.d_wt2,
            &mut state.d_hist,
        )
        .unwrap();
}

struct TradjemaOneSeriesState {
    cuda: CudaTradjema,
    d_high: DeviceBuffer<f32>,
    d_low: DeviceBuffer<f32>,
    d_close: DeviceBuffer<f32>,
    d_lengths: DeviceBuffer<i32>,
    d_mults: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    first_valid: i32,
    series_len: i32,
    n_combos: i32,
    max_length: i32,
}

fn prep_tradjema_one_series_many_params() -> TradjemaOneSeriesState {
    let cuda = CudaTradjema::new(0).expect("cuda tradjema");
    const LENGTH_START: usize = 16;
    const LENGTH_END: usize = 128;
    const LENGTH_STEP: usize = 8;
    const MULT_START: f32 = 5.0;
    const MULT_END: f32 = 15.0;
    const MULT_STEP: f32 = 1.0;
    let series_len = 150_000usize;

    let (high, low, close) = gen_tradjema_ohlc(series_len);
    let first_valid = close.iter().position(|v| !v.is_nan()).unwrap_or(series_len) as i32;

    let length_vals: Vec<usize> = (LENGTH_START..=LENGTH_END).step_by(LENGTH_STEP).collect();
    let mult_steps = ((MULT_END - MULT_START) / MULT_STEP) as usize + 1;
    let mut lengths_host = Vec::with_capacity(length_vals.len() * mult_steps);
    let mut mults_host = Vec::with_capacity(length_vals.len() * mult_steps);
    for &len in &length_vals {
        let mut m = MULT_START;
        while m <= MULT_END + 1e-6 {
            lengths_host.push(len as i32);
            mults_host.push(m);
            m += MULT_STEP;
        }
    }

    let n_combos = lengths_host.len();
    let max_length = *length_vals.iter().max().unwrap_or(&2) as i32;

    let d_high = DeviceBuffer::from_slice(&high).unwrap();
    let d_low = DeviceBuffer::from_slice(&low).unwrap();
    let d_close = DeviceBuffer::from_slice(&close).unwrap();
    let d_lengths = DeviceBuffer::from_slice(&lengths_host).unwrap();
    let d_mults = DeviceBuffer::from_slice(&mults_host).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.unwrap();

    TradjemaOneSeriesState {
        cuda,
        d_high,
        d_low,
        d_close,
        d_lengths,
        d_mults,
        d_out,
        first_valid,
        series_len: series_len as i32,
        n_combos: n_combos as i32,
        max_length,
    }
}

fn launch_tradjema_one_series_many_params(state: &mut TradjemaOneSeriesState) {
    state
        .cuda
        .tradjema_batch_device(
            &state.d_high,
            &state.d_low,
            &state.d_close,
            &state.d_lengths,
            &state.d_mults,
            state.series_len,
            state.n_combos,
            state.first_valid,
            state.max_length,
            &mut state.d_out,
        )
        .unwrap();
}

fn tradjema_one_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    const LENGTH_START: usize = 16;
    const LENGTH_END: usize = 128;
    const LENGTH_STEP: usize = 8;
    const MULT_START: f64 = 5.0;
    const MULT_END: f64 = 15.0;
    const MULT_STEP: f64 = 1.0;
    let series_len = 150_000usize;

    let length_count = ((LENGTH_END - LENGTH_START) / LENGTH_STEP) + 1;
    let mult_count = ((MULT_END - MULT_START) / MULT_STEP) as usize + 1;
    let combos = length_count * mult_count;
    let out_bytes = combos * series_len * std::mem::size_of::<f32>();
    let input_bytes = series_len * std::mem::size_of::<f64>() * 3;
    let param_bytes = combos * (std::mem::size_of::<i32>() + std::mem::size_of::<f64>());
    let approx = out_bytes + input_bytes + param_bytes + 64 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip tradjema one-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("tradjema_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_tradjema_one_series_many_params();
    group.bench_with_input(
        BenchmarkId::from_parameter("150k_x_param_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_tradjema_one_series_many_params(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct TradjemaManySeriesState {
    cuda: CudaTradjema,
    d_high: DeviceBuffer<f32>,
    d_low: DeviceBuffer<f32>,
    d_close: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    num_series: i32,
    series_len: i32,
    length: i32,
    mult: f32,
}

fn prep_tradjema_many_series_one_param() -> TradjemaManySeriesState {
    let cuda = CudaTradjema::new(0).expect("cuda tradjema");
    let num_series = 384usize;
    let series_len = 12_288usize;
    let (high_tm, low_tm, close_tm) = gen_tradjema_ohlc_tm(num_series, series_len);

    let mut first_valids = vec![0i32; num_series];
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len && close_tm[fv * num_series + series].is_nan() {
            fv += 1;
        }
        first_valids[series] = fv as i32;
    }

    let d_high = DeviceBuffer::from_slice(&high_tm).unwrap();
    let d_low = DeviceBuffer::from_slice(&low_tm).unwrap();
    let d_close = DeviceBuffer::from_slice(&close_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    TradjemaManySeriesState {
        cuda,
        d_high,
        d_low,
        d_close,
        d_first_valids,
        d_out,
        num_series: num_series as i32,
        series_len: series_len as i32,
        length: 34,
        mult: 8.0,
    }
}

fn launch_tradjema_many_series_one_param(state: &mut TradjemaManySeriesState) {
    state
        .cuda
        .tradjema_many_series_one_param_device(
            &state.d_high,
            &state.d_low,
            &state.d_close,
            state.num_series,
            state.series_len,
            state.length,
            state.mult,
            &state.d_first_valids,
            &mut state.d_out,
        )
        .unwrap();
}

fn tradjema_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }

    let num_series = 384usize;
    let series_len = 12_288usize;
    let total = num_series * series_len;
    let input_bytes = total * std::mem::size_of::<f64>() * 3;
    let out_bytes = total * std::mem::size_of::<f32>();
    let first_bytes = num_series * std::mem::size_of::<i32>();
    let approx = input_bytes + out_bytes + first_bytes + 64 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip tradjema many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("tradjema_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_tradjema_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("384series_x_12k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_tradjema_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

fn wto_one_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let series_len = 200_000usize;
    let channels = ((8usize..=32).step_by(2)).count();
    let averages = ((18usize..=38).step_by(4)).count();
    let n_combos = channels * averages;
    let out_bytes = 3 * n_combos * series_len * std::mem::size_of::<f32>();
    let param_bytes = 2 * n_combos * std::mem::size_of::<i32>();
    let input_bytes = series_len * std::mem::size_of::<f32>();
    let approx = out_bytes + param_bytes + input_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip wto one-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("wto_cuda_one_series_many_params");
    group.sample_size(10);
    let mut state = prep_wto_one_series_many_params();
    group.bench_with_input(BenchmarkId::from_parameter("200k_x_grid"), &0, |b, _| {
        b.iter(|| {
            launch_wto_one_series_many_params(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// WTO: many series × one param
// ──────────────────────────────────────────────────────────────
struct WtoManySeriesState {
    cuda: CudaWto,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_wt1: DeviceBuffer<f32>,
    d_wt2: DeviceBuffer<f32>,
    d_hist: DeviceBuffer<f32>,
    cols: i32,
    rows: i32,
    channel: i32,
    average: i32,
}

fn prep_wto_many_series_one_param() -> WtoManySeriesState {
    let cuda = CudaWto::new(0).expect("cuda wto");
    let cols = 256usize;
    let rows = 16_384usize;
    let prices_tm = gen_time_major_f32(cols, rows);

    let mut first_valids = vec![0i32; cols];
    for series in 0..cols {
        first_valids[series] = (0..rows)
            .position(|t| !prices_tm[t * cols + series].is_nan())
            .unwrap_or(0) as i32;
    }

    let d_prices_tm = DeviceBuffer::from_slice(&prices_tm).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let mut d_wt1: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.unwrap();
    let mut d_wt2: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(cols * rows) }.unwrap();
    let mut d_hist: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(cols * rows) }.unwrap();

    WtoManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_wt1,
        d_wt2,
        d_hist,
        cols: cols as i32,
        rows: rows as i32,
        channel: 9,
        average: 21,
    }
}

fn launch_wto_many_series_one_param(state: &mut WtoManySeriesState) {
    state
        .cuda
        .wto_many_series_one_param_device(
            &state.d_prices_tm,
            state.cols,
            state.rows,
            state.channel,
            state.average,
            &state.d_first_valids,
            &mut state.d_wt1,
            &mut state.d_wt2,
            &mut state.d_hist,
        )
        .unwrap();
}

fn wto_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let cols = 256usize;
    let rows = 16_384usize;
    let total = cols * rows;
    let out_bytes = 3 * total * std::mem::size_of::<f32>();
    let input_bytes = total * std::mem::size_of::<f32>();
    let first_bytes = cols * std::mem::size_of::<i32>();
    let approx = out_bytes + input_bytes + first_bytes + 32 * 1024 * 1024;
    if let Some(free) = device_free_bytes() {
        if approx > free {
            eprintln!(
                "[bench] skip wto many-series (need ~{} MB, free ~{} MB)",
                approx / (1024 * 1024),
                free / (1024 * 1024)
            );
            return;
        }
    }

    let mut group = c.benchmark_group("wto_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_wto_many_series_one_param();
    group.bench_with_input(
        BenchmarkId::from_parameter("256series_x_16k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_wto_many_series_one_param(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}
