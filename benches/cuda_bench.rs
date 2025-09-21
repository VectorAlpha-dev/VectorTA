#![cfg(feature = "cuda")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cust::memory::DeviceBuffer;

use my_project::cuda::cuda_available;
use my_project::cuda::moving_averages::{
    CudaAlma, CudaEdcf, CudaEhlersKama, CudaEhma, CudaFwma, CudaHwma, CudaMaaq, CudaPwma, CudaSmma,
    CudaSwma, CudaTrima, CudaVwap,
};
use my_project::cuda::wavetrend::CudaWavetrend;
use my_project::indicators::moving_averages::alma::{AlmaBatchRange, AlmaParams};
use my_project::indicators::moving_averages::edcf::{EdcfBatchRange, EdcfParams};
use my_project::indicators::moving_averages::ehlers_kama::EhlersKamaBatchRange;
use my_project::indicators::moving_averages::ehma::{
    expand_grid as ehma_expand_grid, EhmaBatchRange,
};
use my_project::indicators::moving_averages::fwma::{FwmaBatchRange, FwmaParams};
use my_project::indicators::moving_averages::hwma::{
    expand_grid as hwma_expand_grid, HwmaBatchRange,
};
use my_project::indicators::moving_averages::maaq::{
    expand_grid as maaq_expand_grid, MaaqBatchRange,
};
use my_project::indicators::moving_averages::pwma::{
    expand_grid as pwma_expand_grid, PwmaBatchRange,
};
use my_project::indicators::moving_averages::smma::{
    expand_grid as smma_expand_grid, SmmaBatchRange,
};
use my_project::indicators::moving_averages::swma::SwmaBatchRange;
use my_project::indicators::moving_averages::trima::TrimaBatchRange;
use my_project::indicators::wavetrend::{WavetrendBatchRange, WavetrendParams};

fn gen_series(len: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        v[i] = (x * 0.001).sin() + 0.0001 * x;
    }
    v
}

fn expand_swma_periods(range: &SwmaBatchRange) -> Vec<usize> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return vec![start];
    }
    if start > end {
        return Vec::new();
    }
    (start..=end).step_by(step).collect()
}

fn compute_swma_weights(period: usize) -> Vec<f32> {
    if period == 0 {
        return Vec::new();
    }
    let norm = if period <= 2 {
        period as f32
    } else if period % 2 == 0 {
        let half = (period / 2) as f32;
        half * (half + 1.0)
    } else {
        let half_plus = ((period + 1) / 2) as f32;
        half_plus * half_plus
    };
    let inv_norm = 1.0f32 / norm.max(f32::EPSILON);
    let mut weights = vec![0.0f32; period];
    for idx in 0..period {
        let left = idx + 1;
        let right = period - idx;
        weights[idx] = left.min(right) as f32 * inv_norm;
    }
    weights
}

fn fwma_expand_grid(range: &FwmaBatchRange) -> Vec<FwmaParams> {
    let (start, end, step) = range.period;
    let periods = if step == 0 || start == end {
        vec![start]
    } else if start > end {
        Vec::new()
    } else {
        (start..=end).step_by(step).collect()
    };
    periods
        .into_iter()
        .map(|p| FwmaParams { period: Some(p) })
        .collect()
}

fn fibonacci_weights_f32(period: usize) -> Vec<f32> {
    if period == 0 {
        return Vec::new();
    }
    if period == 1 {
        return vec![1.0f32];
    }
    let mut fib = vec![1.0f64; period];
    for i in 2..period {
        fib[i] = fib[i - 1] + fib[i - 2];
    }
    let sum: f64 = fib.iter().sum();
    if sum == 0.0 {
        return vec![0.0f32; period];
    }
    let inv = 1.0 / sum;
    fib.into_iter().map(|v| (v * inv) as f32).collect()
}

fn ehma_normalized_weights(period: usize) -> Vec<f32> {
    if period == 0 {
        return Vec::new();
    }
    let mut weights = vec![0.0f32; period];
    let mut sum = 0.0f32;
    let pi = std::f32::consts::PI;
    for idx in 0..period {
        let i = (period - idx) as f32;
        let angle = (2.0f32 * pi * i) / (period as f32 + 1.0f32);
        let wt = 1.0f32 - angle.cos();
        weights[idx] = wt;
        sum += wt;
    }
    if sum > 0.0 {
        let inv = 1.0f32 / sum;
        for w in &mut weights {
            *w *= inv;
        }
    }
    weights
}

fn expand_wavetrend_range(range: &WavetrendBatchRange) -> Vec<WavetrendParams> {
    fn axis_usize(axis: (usize, usize, usize)) -> Vec<usize> {
        let (start, end, step) = axis;
        if step == 0 || start == end {
            vec![start]
        } else {
            (start..=end).step_by(step).collect()
        }
    }
    fn axis_f64(axis: (f64, f64, f64)) -> Vec<f64> {
        let (start, end, step) = axis;
        if step.abs() < 1e-12 || (start - end).abs() < 1e-12 {
            vec![start]
        } else {
            let mut out = Vec::new();
            let mut v = start;
            while v <= end + 1e-12 {
                out.push(v);
                v += step;
            }
            out
        }
    }

    let channels = axis_usize(range.channel_length);
    let averages = axis_usize(range.average_length);
    let mas = axis_usize(range.ma_length);
    let factors = axis_f64(range.factor);

    let mut combos =
        Vec::with_capacity(channels.len() * averages.len() * mas.len() * factors.len());
    for &ch in &channels {
        for &avg in &averages {
            for &ma in &mas {
                for &factor in &factors {
                    combos.push(WavetrendParams {
                        channel_length: Some(ch),
                        average_length: Some(avg),
                        ma_length: Some(ma),
                        factor: Some(factor),
                    });
                }
            }
        }
    }
    combos
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

fn pascal_weights_f32(period: usize) -> Vec<f32> {
    assert!(period > 0, "period must be positive");
    let n = period - 1;
    let mut row = Vec::with_capacity(period);
    let mut sum = 0.0f64;
    for r in 0..=n {
        let mut val = 1.0f64;
        for i in 0..r {
            val *= (n - i) as f64;
            val /= (i + 1) as f64;
        }
        row.push(val);
        sum += val;
    }
    let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
    row.into_iter().map(|v| (v * inv) as f32).collect()
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

fn gen_vwap_inputs(len: usize) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
    let mut timestamps = Vec::with_capacity(len);
    let mut volumes = Vec::with_capacity(len);
    let mut prices = Vec::with_capacity(len);
    let base_ts = 1_600_000_000_000i64;
    for i in 0..len {
        let idx = i as f64;
        timestamps.push(base_ts + (i as i64) * 60_000); // 1-minute spacing
        prices.push(100.0 + (idx * 0.0007).sin() * 8.0 + 0.05 * (idx * 0.003).cos());
        let vol = if i % 19 == 0 {
            0.0
        } else {
            1.0 + (idx * 0.01).sin().abs() + (idx * 0.005).cos().abs()
        };
        volumes.push(vol);
    }
    (timestamps, volumes, prices)
}

fn expand_trima_periods(range: &TrimaBatchRange) -> Vec<usize> {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        return vec![start];
    }
    (start..=end).step_by(step).collect()
}

fn compute_trima_weights(period: usize) -> Vec<f32> {
    let mut weights = vec![0.0f32; period];
    let m1 = (period + 1) / 2;
    let m2 = period - m1 + 1;
    let norm = (m1 * m2) as f32;
    for i in 0..period {
        let w = if i < m1 {
            (i + 1) as f32
        } else if i < m2 {
            m1 as f32
        } else {
            (m1 + m2 - 1 - i) as f32
        };
        weights[i] = w / norm;
    }
    weights
}

fn parse_anchor_basic(anchor: &str) -> (i32, i32, i64) {
    let split = anchor
        .find(|c: char| !c.is_ascii_digit())
        .expect("anchor must contain unit");
    let (num_part, unit_part) = anchor.split_at(split);
    let count: i32 = num_part.parse().expect("invalid anchor count");
    let unit_char = unit_part
        .chars()
        .next()
        .expect("missing anchor unit")
        .to_ascii_lowercase();
    let (code, scale) = match unit_char {
        'm' => (0, 60_000i64),
        'h' => (1, 3_600_000i64),
        'd' => (2, 86_400_000i64),
        other => panic!("unsupported anchor unit '{}' in bench", other),
    };
    (count, code, (count as i64) * scale)
}

fn first_valid_index_basic(timestamps: &[i64], volumes: &[f64], count: i32, unit_code: i32) -> i32 {
    if timestamps.is_empty() {
        return 0;
    }
    let denom = match unit_code {
        0 => (count as i64) * 60_000,
        1 => (count as i64) * 3_600_000,
        2 => (count as i64) * 86_400_000,
        _ => 1,
    };

    let mut cur_gid = i64::MIN;
    let mut vsum = 0.0f64;
    for (idx, (&ts, &vol)) in timestamps.iter().zip(volumes.iter()).enumerate() {
        let gid = ts / denom;
        if gid != cur_gid {
            cur_gid = gid;
            vsum = 0.0;
        }
        vsum += vol;
        if vsum > 0.0 {
            return idx as i32;
        }
    }
    0
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
// EDCF: one series × many periods (batch)
// ──────────────────────────────────────────────────────────────
struct EdcfBatchState {
    cuda: CudaEdcf,
    combos: Vec<EdcfParams>,
    d_prices: DeviceBuffer<f32>,
    d_dist: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    first_valid: usize,
    series_len: usize,
}

fn prep_edcf_batch_state() -> EdcfBatchState {
    let cuda = CudaEdcf::new(0).expect("cuda edcf");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap_or(0);
    let sweep = EdcfBatchRange { period: (5, 65, 5) };
    let combos: Vec<EdcfParams> = (sweep.period.0..=sweep.period.1)
        .step_by(sweep.period.2)
        .map(|p| EdcfParams { period: Some(p) })
        .collect();

    let host_prices: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let d_prices = DeviceBuffer::from_slice(&host_prices).expect("upload prices");
    let d_dist: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len) }.expect("allocate dist buffer");
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(combos.len() * series_len) }
            .expect("allocate output buffer");

    EdcfBatchState {
        cuda,
        combos,
        d_prices,
        d_dist,
        d_out,
        first_valid,
        series_len,
    }
}

fn launch_edcf_batch(state: &mut EdcfBatchState) {
    state
        .cuda
        .edcf_batch_device(
            &state.d_prices,
            &state.combos,
            state.first_valid,
            state.series_len,
            &mut state.d_dist,
            &mut state.d_out,
        )
        .expect("launch edcf batch kernel");
}

fn edcf_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("edcf_cuda_batch");
    group.sample_size(10);
    let mut state = prep_edcf_batch_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("series_200k_period_5_to_65"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_edcf_batch(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// WaveTrend: one series × many parameter combinations (batch)
// ──────────────────────────────────────────────────────────────
struct WavetrendBatchState {
    cuda: CudaWavetrend,
    combos: Vec<WavetrendParams>,
    d_prices: DeviceBuffer<f32>,
    d_wt1: DeviceBuffer<f32>,
    d_wt2: DeviceBuffer<f32>,
    d_wt_diff: DeviceBuffer<f32>,
    first_valid: usize,
    series_len: usize,
}

fn prep_wavetrend_batch_state() -> WavetrendBatchState {
    let cuda = CudaWavetrend::new(0).expect("cuda wavetrend");
    let series_len = 180_000usize;
    let data = gen_series(series_len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap_or(0);
    let sweep = WavetrendBatchRange {
        channel_length: (6, 24, 6),
        average_length: (8, 20, 4),
        ma_length: (3, 6, 1),
        factor: (0.010, 0.020, 0.0025),
    };
    let combos = expand_wavetrend_range(&sweep);

    let host_prices: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let d_prices = DeviceBuffer::from_slice(&host_prices).expect("upload prices");
    let d_wt1: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(combos.len() * series_len) }
            .expect("allocate wt1 buffer");
    let d_wt2: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(combos.len() * series_len) }
            .expect("allocate wt2 buffer");
    let d_wt_diff: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(combos.len() * series_len) }
            .expect("allocate diff buffer");

    WavetrendBatchState {
        cuda,
        combos,
        d_prices,
        d_wt1,
        d_wt2,
        d_wt_diff,
        first_valid,
        series_len,
    }
}

fn launch_wavetrend_batch(state: &mut WavetrendBatchState) {
    state
        .cuda
        .wavetrend_batch_device(
            &state.d_prices,
            &state.combos,
            state.first_valid,
            state.series_len,
            &mut state.d_wt1,
            &mut state.d_wt2,
            &mut state.d_wt_diff,
        )
        .expect("launch wavetrend batch kernel");
}

fn wavetrend_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("wavetrend_cuda_batch");
    group.sample_size(10);
    let mut state = prep_wavetrend_batch_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("series_180k_param_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_wavetrend_batch(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// VWAP: one series × many anchors (batch path)
// ──────────────────────────────────────────────────────────────
struct VwapBatchState {
    cuda: CudaVwap,
    d_timestamps: DeviceBuffer<i64>,
    d_volumes: DeviceBuffer<f32>,
    d_prices: DeviceBuffer<f32>,
    d_counts: DeviceBuffer<i32>,
    d_unit_codes: DeviceBuffer<i32>,
    d_divisors: DeviceBuffer<i64>,
    d_first_valids: DeviceBuffer<i32>,
    d_month_ids: Option<DeviceBuffer<i32>>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
}

fn prep_vwap_batch_state() -> VwapBatchState {
    let cuda = CudaVwap::new(0).expect("cuda vwap");
    let series_len = 200_000usize;
    let (timestamps, volumes, prices) = gen_vwap_inputs(series_len);
    let volumes_f32: Vec<f32> = volumes.iter().map(|&v| v as f32).collect();
    let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();

    let anchors = ["1m", "2m", "5m", "10m", "30m", "1h", "4h", "1d"];
    let n_combos = anchors.len();

    let mut counts = Vec::with_capacity(n_combos);
    let mut unit_codes = Vec::with_capacity(n_combos);
    let mut divisors = Vec::with_capacity(n_combos);
    let mut first_valids = Vec::with_capacity(n_combos);
    for &anchor in &anchors {
        let (count, unit_code, divisor) = parse_anchor_basic(anchor);
        counts.push(count);
        unit_codes.push(unit_code);
        divisors.push(divisor);
        first_valids.push(first_valid_index_basic(
            &timestamps,
            &volumes,
            count,
            unit_code,
        ));
    }

    let d_timestamps = DeviceBuffer::from_slice(&timestamps).expect("upload timestamps");
    let d_volumes = DeviceBuffer::from_slice(&volumes_f32).expect("upload volumes");
    let d_prices = DeviceBuffer::from_slice(&prices_f32).expect("upload prices");
    let d_counts = DeviceBuffer::from_slice(&counts).expect("upload counts");
    let d_unit_codes = DeviceBuffer::from_slice(&unit_codes).expect("upload unit codes");
    let d_divisors = DeviceBuffer::from_slice(&divisors).expect("upload divisors");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("upload first valids");
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.expect("alloc vwap out");

    VwapBatchState {
        cuda,
        d_timestamps,
        d_volumes,
        d_prices,
        d_counts,
        d_unit_codes,
        d_divisors,
        d_first_valids,
        d_month_ids: None,
        d_out,
        series_len,
        n_combos,
    }
}

fn launch_vwap_batch(state: &mut VwapBatchState) {
    let month_ids = state.d_month_ids.as_ref();
    state
        .cuda
        .vwap_batch_device_with_params(
            &state.d_timestamps,
            &state.d_volumes,
            &state.d_prices,
            &state.d_counts,
            &state.d_unit_codes,
            &state.d_divisors,
            &state.d_first_valids,
            month_ids,
            state.series_len,
            state.n_combos,
            &mut state.d_out,
        )
        .expect("launch vwap cuda batch kernel");
}

// ──────────────────────────────────────────────────────────────
// MAAQ: one series × many parameter combinations (batch path)
struct MaaqBatchState {
    cuda: CudaMaaq,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_fast_scs: DeviceBuffer<f32>,
    d_slow_scs: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    first_valid: usize,
    series_len: usize,
    n_combos: usize,
    max_period: usize,
}

fn prep_maaq_batch_state() -> MaaqBatchState {
    let cuda = CudaMaaq::new(0).expect("cuda maaq");
    let series_len = 200_000usize;
    let host_series = gen_series(series_len);
    let host_f32: Vec<f32> = host_series.into_iter().map(|v| v as f32).collect();
    let first_valid = host_f32.iter().position(|x| !x.is_nan()).unwrap_or(0);

    let sweep = MaaqBatchRange {
        period: (5, 115, 5),
        fast_period: (2, 10, 2),
        slow_period: (20, 80, 10),
    };
    let combos = maaq_expand_grid(&sweep);
    let n_combos = combos.len();
    let max_period = combos.iter().map(|p| p.period.unwrap()).max().unwrap_or(1);

    let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
    let fast_scs: Vec<f32> = combos
        .iter()
        .map(|p| 2.0f32 / (p.fast_period.unwrap() as f32 + 1.0f32))
        .collect();
    let slow_scs: Vec<f32> = combos
        .iter()
        .map(|p| 2.0f32 / (p.slow_period.unwrap() as f32 + 1.0f32))
        .collect();

    let d_prices = DeviceBuffer::from_slice(&host_f32).expect("upload maaq prices");
    let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("upload maaq periods");
    let d_fast_scs = DeviceBuffer::from_slice(&fast_scs).expect("upload maaq fast scs");
    let d_slow_scs = DeviceBuffer::from_slice(&slow_scs).expect("upload maaq slow scs");
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.expect("alloc maaq out");

    MaaqBatchState {
        cuda,
        d_prices,
        d_periods,
        d_fast_scs,
        d_slow_scs,
        d_out,
        first_valid,
        series_len,
        n_combos,
        max_period,
    }
}

fn launch_maaq_batch(state: &mut MaaqBatchState) {
    state
        .cuda
        .maaq_batch_device(
            &state.d_prices,
            &state.d_periods,
            &state.d_fast_scs,
            &state.d_slow_scs,
            state.first_valid,
            state.series_len,
            state.n_combos,
            state.max_period,
            &mut state.d_out,
        )
        .unwrap();
}

fn maaq_cuda_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("maaq_cuda_batch");
    group.sample_size(10);
    let mut state = prep_maaq_batch_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("series_200k_param_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_maaq_batch(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// MAAQ: many series × one parameter combination (time-major)
struct MaaqManySeriesState {
    cuda: CudaMaaq,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    fast_sc: f32,
    slow_sc: f32,
    period: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_maaq_many_series_state() -> MaaqManySeriesState {
    let cuda = CudaMaaq::new(0).expect("cuda maaq");
    let num_series = 64usize;
    let series_len = 8_192usize;
    let host_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len {
            let idx = fv * num_series + series;
            if !host_tm[idx].is_nan() {
                break;
            }
            fv += 1;
        }
        first_valids.push(fv as i32);
    }

    let period = 32usize;
    let fast_period = 4usize;
    let slow_period = 48usize;
    let fast_sc = 2.0f32 / (fast_period as f32 + 1.0f32);
    let slow_sc = 2.0f32 / (slow_period as f32 + 1.0f32);

    let d_prices_tm = DeviceBuffer::from_slice(&host_tm).expect("upload maaq tm prices");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("upload maaq first valids");
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.expect("alloc maaq tm out");

    MaaqManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        fast_sc,
        slow_sc,
        period: period as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_maaq_many_series(state: &mut MaaqManySeriesState) {
    state
        .cuda
        .maaq_multi_series_one_param_device(
            &state.d_prices_tm,
            state.period,
            state.fast_sc,
            state.slow_sc,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn maaq_cuda_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("maaq_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_maaq_many_series_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("time_major_series64_len8192"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_maaq_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// PWMA: one series × many periods (batch path)
struct PwmaBatchState {
    cuda: CudaPwma,
    d_prices: DeviceBuffer<f32>,
    d_weights: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_warms: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    max_period: usize,
}

fn prep_pwma_batch_state() -> PwmaBatchState {
    let cuda = CudaPwma::new(0).expect("cuda pwma");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let data_f32: Vec<f32> = data.into_iter().map(|v| v as f32).collect();
    let first_valid = data_f32.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let sweep = PwmaBatchRange {
        period: (5, 160, 5),
    };
    let combos = pwma_expand_grid(&sweep);
    let n_combos = combos.len();
    let max_period = combos.iter().map(|p| p.period.unwrap()).max().unwrap_or(1);

    let mut weights_flat = vec![0.0f32; n_combos * max_period];
    for (row, combo) in combos.iter().enumerate() {
        let weights = pascal_weights_f32(combo.period.unwrap());
        let base = row * max_period;
        for (idx, w) in weights.iter().enumerate() {
            weights_flat[base + idx] = *w;
        }
    }

    let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
    let warms_i32: Vec<i32> = combos
        .iter()
        .map(|p| (first_valid + p.period.unwrap() - 1) as i32)
        .collect();

    let d_prices = DeviceBuffer::from_slice(&data_f32).expect("upload pwma prices");
    let d_weights = DeviceBuffer::from_slice(&weights_flat).expect("upload pwma weights");
    let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("upload pwma periods");
    let d_warms = DeviceBuffer::from_slice(&warms_i32).expect("upload pwma warms");
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.expect("alloc pwma out");

    PwmaBatchState {
        cuda,
        d_prices,
        d_weights,
        d_periods,
        d_warms,
        d_out,
        series_len,
        n_combos,
        max_period,
    }
}

fn launch_pwma_batch(state: &mut PwmaBatchState) {
    state
        .cuda
        .pwma_batch_device(
            &state.d_prices,
            &state.d_weights,
            &state.d_periods,
            &state.d_warms,
            state.series_len,
            state.n_combos,
            state.max_period,
            &mut state.d_out,
        )
        .unwrap();
}

// FWMA: one series × many periods (batch path)
struct FwmaBatchState {
    cuda: CudaFwma,
    d_prices: DeviceBuffer<f32>,
    d_weights: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_warms: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    max_period: usize,
}

fn prep_fwma_batch_state() -> FwmaBatchState {
    let cuda = CudaFwma::new(0).expect("cuda fwma");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let data_f32: Vec<f32> = data.into_iter().map(|v| v as f32).collect();
    let first_valid = data_f32.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let sweep = FwmaBatchRange {
        period: (5, 160, 5),
    };
    let combos = fwma_expand_grid(&sweep);
    let n_combos = combos.len();
    let max_period = combos.iter().map(|p| p.period.unwrap()).max().unwrap_or(1);

    let mut weights_flat = vec![0.0f32; n_combos * max_period];
    for (row, combo) in combos.iter().enumerate() {
        let weights = fibonacci_weights_f32(combo.period.unwrap());
        let base = row * max_period;
        for (idx, w) in weights.iter().enumerate() {
            weights_flat[base + idx] = *w;
        }
    }

    let periods_i32: Vec<i32> = combos.iter().map(|p| p.period.unwrap() as i32).collect();
    let warms_i32: Vec<i32> = combos
        .iter()
        .map(|p| (first_valid + p.period.unwrap() - 1) as i32)
        .collect();

    let d_prices = DeviceBuffer::from_slice(&data_f32).expect("upload fwma prices");
    let d_weights = DeviceBuffer::from_slice(&weights_flat).expect("upload fwma weights");
    let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("upload fwma periods");
    let d_warms = DeviceBuffer::from_slice(&warms_i32).expect("upload fwma warms");
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.expect("alloc fwma out");

    FwmaBatchState {
        cuda,
        d_prices,
        d_weights,
        d_periods,
        d_warms,
        d_out,
        series_len,
        n_combos,
        max_period,
    }
}

fn launch_fwma_batch(state: &mut FwmaBatchState) {
    state
        .cuda
        .fwma_batch_device(
            &state.d_prices,
            &state.d_weights,
            &state.d_periods,
            &state.d_warms,
            state.series_len,
            state.n_combos,
            state.max_period,
            &mut state.d_out,
        )
        .unwrap();
}

fn fwma_cuda_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("fwma_cuda_batch");
    group.sample_size(10);
    let mut state = prep_fwma_batch_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("series_200k_period_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_fwma_batch(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// FWMA: many series × one period (time-major)
struct FwmaManySeriesState {
    cuda: CudaFwma,
    d_prices_tm: DeviceBuffer<f32>,
    d_weights: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_fwma_many_series_state() -> FwmaManySeriesState {
    let cuda = CudaFwma::new(0).expect("cuda fwma");
    let num_series = 64usize;
    let series_len = 8_192usize;
    let host_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len {
            let idx = fv * num_series + series;
            if !host_tm[idx].is_nan() {
                break;
            }
            fv += 1;
        }
        first_valids.push(fv as i32);
    }

    let period = 32usize;
    let weights = fibonacci_weights_f32(period);

    let d_prices_tm = DeviceBuffer::from_slice(&host_tm).expect("upload fwma tm prices");
    let d_weights = DeviceBuffer::from_slice(&weights).expect("upload fwma tm weights");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("upload fwma first valids");
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.expect("alloc fwma tm out");

    FwmaManySeriesState {
        cuda,
        d_prices_tm,
        d_weights,
        d_first_valids,
        d_out_tm,
        period: period as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_fwma_many_series(state: &mut FwmaManySeriesState) {
    state
        .cuda
        .fwma_multi_series_one_param_device(
            &state.d_prices_tm,
            &state.d_weights,
            &state.d_first_valids,
            state.period,
            state.num_series,
            state.series_len,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn fwma_cuda_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("fwma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_fwma_many_series_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("time_major_series64_len8192"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_fwma_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

fn pwma_cuda_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("pwma_cuda_batch");
    group.sample_size(10);
    let mut state = prep_pwma_batch_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("series_200k_period_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_pwma_batch(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// PWMA: many series × one period (time-major)
struct PwmaManySeriesState {
    cuda: CudaPwma,
    d_prices_tm: DeviceBuffer<f32>,
    d_weights: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_pwma_many_series_state() -> PwmaManySeriesState {
    let cuda = CudaPwma::new(0).expect("cuda pwma");
    let num_series = 64usize;
    let series_len = 8_192usize;
    let host_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len {
            let idx = fv * num_series + series;
            if !host_tm[idx].is_nan() {
                break;
            }
            fv += 1;
        }
        first_valids.push(fv as i32);
    }

    let period = 32usize;
    let weights = pascal_weights_f32(period);

    let d_prices_tm = DeviceBuffer::from_slice(&host_tm).expect("upload pwma tm prices");
    let d_weights = DeviceBuffer::from_slice(&weights).expect("upload pwma tm weights");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("upload pwma first valids");
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.expect("alloc pwma tm out");

    PwmaManySeriesState {
        cuda,
        d_prices_tm,
        d_weights,
        d_first_valids,
        d_out_tm,
        period: period as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_pwma_many_series(state: &mut PwmaManySeriesState) {
    state
        .cuda
        .pwma_multi_series_one_param_device(
            &state.d_prices_tm,
            &state.d_weights,
            &state.d_first_valids,
            state.period,
            state.num_series,
            state.series_len,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn pwma_cuda_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("pwma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_pwma_many_series_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("time_major_series64_len8192"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_pwma_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// EHMA: one series × many periods (batch path)
// ──────────────────────────────────────────────────────────────
struct EhmaBatchState {
    cuda: CudaEhma,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_warms: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    max_period: usize,
}

fn prep_ehma_batch_state() -> EhmaBatchState {
    let cuda = CudaEhma::new(0).expect("cuda ehma");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let data_f32: Vec<f32> = data.into_iter().map(|v| v as f32).collect();
    let first_valid = data_f32.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let sweep = EhmaBatchRange {
        period: (5, 160, 5),
    };
    let combos = ehma_expand_grid(&sweep);
    let n_combos = combos.len();
    let max_period = combos.iter().map(|c| c.period.unwrap()).max().unwrap_or(0);
    let periods_i32: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
    let warms_i32: Vec<i32> = combos
        .iter()
        .map(|c| (first_valid + c.period.unwrap() - 1) as i32)
        .collect();

    let d_prices = DeviceBuffer::from_slice(&data_f32).expect("upload ehma prices");
    let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("upload ehma periods");
    let d_warms = DeviceBuffer::from_slice(&warms_i32).expect("upload ehma warms");
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.expect("alloc ehma out");

    EhmaBatchState {
        cuda,
        d_prices,
        d_periods,
        d_warms,
        d_out,
        series_len,
        n_combos,
        max_period,
    }
}

fn launch_ehma_batch(state: &mut EhmaBatchState) {
    state
        .cuda
        .ehma_batch_device(
            &state.d_prices,
            &state.d_periods,
            &state.d_warms,
            state.series_len,
            state.n_combos,
            state.max_period,
            &mut state.d_out,
        )
        .unwrap();
}

fn ehma_cuda_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("ehma_cuda_batch");
    group.sample_size(10);
    let mut state = prep_ehma_batch_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("series_200k_period_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_ehma_batch(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// EHMA: many series × one period (time-major)
// ──────────────────────────────────────────────────────────────
struct EhmaManySeriesState {
    cuda: CudaEhma,
    d_prices_tm: DeviceBuffer<f32>,
    d_weights: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_ehma_many_series_state() -> EhmaManySeriesState {
    let cuda = CudaEhma::new(0).expect("cuda ehma");
    let num_series = 64usize;
    let series_len = 8_192usize;
    let host_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len {
            let idx = fv * num_series + series;
            if !host_tm[idx].is_nan() {
                break;
            }
            fv += 1;
        }
        first_valids.push(fv as i32);
    }

    let period = 24usize;
    let weights = ehma_normalized_weights(period);

    let d_prices_tm = DeviceBuffer::from_slice(&host_tm).expect("upload ehma tm prices");
    let d_weights = DeviceBuffer::from_slice(&weights).expect("upload ehma tm weights");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("upload ehma first valids");
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.expect("alloc ehma tm out");

    EhmaManySeriesState {
        cuda,
        d_prices_tm,
        d_weights,
        d_first_valids,
        d_out_tm,
        period: period as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_ehma_many_series(state: &mut EhmaManySeriesState) {
    state
        .cuda
        .ehma_multi_series_one_param_device(
            &state.d_prices_tm,
            &state.d_weights,
            state.period,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn ehma_cuda_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("ehma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_ehma_many_series_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("time_major_series64_len8192"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_ehma_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// EHLERS KAMA: one series × many periods (batch path)
// ──────────────────────────────────────────────────────────────
struct EhlersKamaBatchState {
    cuda: CudaEhlersKama,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    first_valid: usize,
    series_len: usize,
    n_combos: usize,
}

fn prep_ehlers_kama_batch_state() -> EhlersKamaBatchState {
    let cuda = CudaEhlersKama::new(0).expect("cuda ehlers kama");
    let series_len = 200_000usize;
    let host = gen_series(series_len);
    let host_f32: Vec<f32> = host.into_iter().map(|v| v as f32).collect();
    let first_valid = host_f32.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let sweep = EhlersKamaBatchRange {
        period: (5, 160, 5),
    };
    let (start, end, step) = sweep.period;
    let periods: Vec<usize> = if step == 0 || start == end {
        vec![start]
    } else {
        (start..=end).step_by(step).collect()
    };
    let n_combos = periods.len();
    let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();

    let d_prices = DeviceBuffer::from_slice(&host_f32).expect("upload ehlers kama prices");
    let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("upload ehlers kama periods");
    let d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }
        .expect("alloc ehlers kama out");

    EhlersKamaBatchState {
        cuda,
        d_prices,
        d_periods,
        d_out,
        first_valid,
        series_len,
        n_combos,
    }
}

fn launch_ehlers_kama_batch(state: &mut EhlersKamaBatchState) {
    state
        .cuda
        .ehlers_kama_batch_device(
            &state.d_prices,
            &state.d_periods,
            state.first_valid,
            state.series_len,
            state.n_combos,
            &mut state.d_out,
        )
        .unwrap();
}

fn ehlers_kama_cuda_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("ehlers_kama_cuda_batch");
    group.sample_size(10);
    let mut state = prep_ehlers_kama_batch_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("series_200k_period_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_ehlers_kama_batch(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// EHLERS KAMA: many series × one period (time-major)
// ──────────────────────────────────────────────────────────────
struct EhlersKamaManySeriesState {
    cuda: CudaEhlersKama,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_ehlers_kama_many_series_state() -> EhlersKamaManySeriesState {
    let cuda = CudaEhlersKama::new(0).expect("cuda ehlers kama");
    let num_series = 64usize;
    let series_len = 8_192usize;
    let host_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len {
            let idx = fv * num_series + series;
            if !host_tm[idx].is_nan() {
                break;
            }
            fv += 1;
        }
        first_valids.push(fv as i32);
    }

    let period = 30usize;
    let d_prices_tm = DeviceBuffer::from_slice(&host_tm).expect("upload ehlers kama tm prices");
    let d_first_valids =
        DeviceBuffer::from_slice(&first_valids).expect("upload ehlers kama first valids");
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }
            .expect("alloc ehlers kama tm out");

    EhlersKamaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        period: period as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_ehlers_kama_many_series(state: &mut EhlersKamaManySeriesState) {
    state
        .cuda
        .ehlers_kama_multi_series_one_param_device(
            &state.d_prices_tm,
            state.period,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn ehlers_kama_cuda_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("ehlers_kama_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_ehlers_kama_many_series_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("time_major_series64_len8192"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_ehlers_kama_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// HWMA: one series × many parameter triples (batch path)
// ──────────────────────────────────────────────────────────────
struct HwmaBatchState {
    cuda: CudaHwma,
    d_prices: DeviceBuffer<f32>,
    d_nas: DeviceBuffer<f32>,
    d_nbs: DeviceBuffer<f32>,
    d_ncs: DeviceBuffer<f32>,
    d_out: DeviceBuffer<f32>,
    first_valid: usize,
    series_len: usize,
    n_combos: usize,
}

fn prep_hwma_batch_state() -> HwmaBatchState {
    let cuda = CudaHwma::new(0).expect("cuda hwma");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let data_f32: Vec<f32> = data.into_iter().map(|v| v as f32).collect();
    let first_valid = data_f32.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let sweep = HwmaBatchRange {
        na: (0.10, 0.40, 0.05),
        nb: (0.05, 0.25, 0.05),
        nc: (0.05, 0.20, 0.05),
    };
    let combos = hwma_expand_grid(&sweep);
    let n_combos = combos.len();
    let nas: Vec<f32> = combos.iter().map(|p| p.na.unwrap() as f32).collect();
    let nbs: Vec<f32> = combos.iter().map(|p| p.nb.unwrap() as f32).collect();
    let ncs: Vec<f32> = combos.iter().map(|p| p.nc.unwrap() as f32).collect();

    let d_prices = DeviceBuffer::from_slice(&data_f32).expect("upload hwma prices");
    let d_nas = DeviceBuffer::from_slice(&nas).expect("upload hwma na");
    let d_nbs = DeviceBuffer::from_slice(&nbs).expect("upload hwma nb");
    let d_ncs = DeviceBuffer::from_slice(&ncs).expect("upload hwma nc");
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.expect("alloc hwma out");

    HwmaBatchState {
        cuda,
        d_prices,
        d_nas,
        d_nbs,
        d_ncs,
        d_out,
        first_valid,
        series_len,
        n_combos,
    }
}

fn launch_hwma_batch(state: &mut HwmaBatchState) {
    state
        .cuda
        .hwma_batch_device(
            &state.d_prices,
            &state.d_nas,
            &state.d_nbs,
            &state.d_ncs,
            state.first_valid,
            state.series_len,
            state.n_combos,
            &mut state.d_out,
        )
        .unwrap();
}

fn hwma_cuda_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("hwma_cuda_batch");
    group.sample_size(10);
    let mut state = prep_hwma_batch_state();
    group.bench_with_input(BenchmarkId::from_parameter("series_len200k"), &0, |b, _| {
        b.iter(|| {
            launch_hwma_batch(&mut state);
            black_box(())
        })
    });
    group.finish();
}

// HWMA: many series × one parameter triple (time-major path)
// ──────────────────────────────────────────────────────────────
struct HwmaManySeriesState {
    cuda: CudaHwma,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    na: f32,
    nb: f32,
    nc: f32,
    num_series: i32,
    series_len: i32,
}

fn prep_hwma_many_series_state() -> HwmaManySeriesState {
    let cuda = CudaHwma::new(0).expect("cuda hwma");
    let num_series = 64usize;
    let series_len = 8_192usize;
    let host_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len {
            let idx = fv * num_series + series;
            if !host_tm[idx].is_nan() {
                break;
            }
            fv += 1;
        }
        first_valids.push(fv as i32);
    }

    let na = 0.18f32;
    let nb = 0.12f32;
    let nc = 0.08f32;

    let d_prices_tm = DeviceBuffer::from_slice(&host_tm).expect("upload hwma tm prices");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("upload hwma first valids");
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.expect("alloc hwma tm out");

    HwmaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        na,
        nb,
        nc,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_hwma_many_series(state: &mut HwmaManySeriesState) {
    state
        .cuda
        .hwma_multi_series_one_param_device(
            &state.d_prices_tm,
            state.na,
            state.nb,
            state.nc,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn hwma_cuda_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("hwma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_hwma_many_series_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("tm_series64_len8k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_hwma_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// SMMA: one series × many periods (batch path)
// ──────────────────────────────────────────────────────────────
struct SmmaBatchState {
    cuda: CudaSmma,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_warms: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    first_valid: usize,
    series_len: usize,
    n_combos: usize,
}

fn prep_smma_batch_state() -> SmmaBatchState {
    let cuda = CudaSmma::new(0).expect("cuda smma");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let data_f32: Vec<f32> = data.into_iter().map(|v| v as f32).collect();
    let first_valid = data_f32.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let sweep = SmmaBatchRange {
        period: (5, 160, 5),
    };
    let combos = smma_expand_grid(&sweep);
    let n_combos = combos.len();
    let periods_i32: Vec<i32> = combos.iter().map(|c| c.period.unwrap() as i32).collect();
    let warms_i32: Vec<i32> = combos
        .iter()
        .map(|c| (first_valid + c.period.unwrap() - 1) as i32)
        .collect();

    let d_prices = DeviceBuffer::from_slice(&data_f32).expect("upload smma prices");
    let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("upload smma periods");
    let d_warms = DeviceBuffer::from_slice(&warms_i32).expect("upload smma warms");
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.expect("alloc smma out");

    SmmaBatchState {
        cuda,
        d_prices,
        d_periods,
        d_warms,
        d_out,
        first_valid,
        series_len,
        n_combos,
    }
}

fn launch_smma_batch(state: &mut SmmaBatchState) {
    state
        .cuda
        .smma_batch_device(
            &state.d_prices,
            &state.d_periods,
            &state.d_warms,
            state.first_valid,
            state.series_len,
            state.n_combos,
            &mut state.d_out,
        )
        .unwrap();
}

fn smma_cuda_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("smma_cuda_batch");
    group.sample_size(10);
    let mut state = prep_smma_batch_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("series_200k_period_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_smma_batch(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// SMMA: many series × one period (time-major)
// ──────────────────────────────────────────────────────────────
struct SmmaManySeriesState {
    cuda: CudaSmma,
    d_prices_tm: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_smma_many_series_state() -> SmmaManySeriesState {
    let cuda = CudaSmma::new(0).expect("cuda smma");
    let num_series = 64usize;
    let series_len = 8_192usize;
    let host_tm = gen_time_major_f32(num_series, series_len);

    let mut first_valids = Vec::with_capacity(num_series);
    for series in 0..num_series {
        let mut fv = 0usize;
        while fv < series_len {
            let idx = fv * num_series + series;
            if !host_tm[idx].is_nan() {
                break;
            }
            fv += 1;
        }
        first_valids.push(fv as i32);
    }

    let period = 24usize;
    let d_prices_tm = DeviceBuffer::from_slice(&host_tm).expect("upload smma tm prices");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("upload smma first valids");
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.expect("alloc smma tm out");

    SmmaManySeriesState {
        cuda,
        d_prices_tm,
        d_first_valids,
        d_out_tm,
        period: period as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_smma_many_series(state: &mut SmmaManySeriesState) {
    state
        .cuda
        .smma_multi_series_one_param_device(
            &state.d_prices_tm,
            state.period,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn smma_cuda_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("smma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_smma_many_series_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("tm_series64_len8k"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_smma_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// SWMA: one series × many periods (batch path)
// ──────────────────────────────────────────────────────────────
struct SwmaBatchState {
    cuda: CudaSwma,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_warms: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    max_period: usize,
}

fn prep_swma_batch_state() -> SwmaBatchState {
    let cuda = CudaSwma::new(0).expect("cuda swma");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let data_f32: Vec<f32> = data.into_iter().map(|v| v as f32).collect();
    let sweep = SwmaBatchRange {
        period: (5, 180, 5),
    };
    let periods = expand_swma_periods(&sweep);
    let first_valid = data_f32.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let max_period = *periods.iter().max().unwrap();
    let n_combos = periods.len();
    let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
    let warms_i32: Vec<i32> = periods
        .iter()
        .map(|&p| (first_valid + p - 1) as i32)
        .collect();

    let d_prices = DeviceBuffer::from_slice(&data_f32).expect("upload swma prices");
    let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("upload swma periods");
    let d_warms = DeviceBuffer::from_slice(&warms_i32).expect("upload swma warms");
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.expect("alloc swma out");

    SwmaBatchState {
        cuda,
        d_prices,
        d_periods,
        d_warms,
        d_out,
        series_len,
        n_combos,
        max_period,
    }
}

fn launch_swma_batch(state: &mut SwmaBatchState) {
    state
        .cuda
        .swma_batch_device(
            &state.d_prices,
            &state.d_periods,
            &state.d_warms,
            state.series_len,
            state.n_combos,
            state.max_period,
            &mut state.d_out,
        )
        .unwrap();
}

fn swma_cuda_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("swma_cuda_batch");
    group.sample_size(10);
    let mut state = prep_swma_batch_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("series_200k_period_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_swma_batch(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// SWMA: many series × one period (time-major)
// ──────────────────────────────────────────────────────────────
struct SwmaManySeriesState {
    cuda: CudaSwma,
    d_prices_tm: DeviceBuffer<f32>,
    d_weights: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_swma_many_series_state() -> SwmaManySeriesState {
    let cuda = CudaSwma::new(0).expect("cuda swma");
    let num_series = 2048usize;
    let series_len = 50_000usize;
    let host_tm = gen_time_major_f32(num_series, series_len);
    let period = 24usize;
    let weights = compute_swma_weights(period);
    let mut first_valids = vec![0i32; num_series];
    for series in 0..num_series {
        first_valids[series] = (0..series_len)
            .position(|row| !host_tm[row * num_series + series].is_nan())
            .unwrap_or(0) as i32;
    }

    let d_prices_tm = DeviceBuffer::from_slice(&host_tm).expect("upload swma tm prices");
    let d_weights = DeviceBuffer::from_slice(&weights).expect("upload swma weights");
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).expect("upload swma first valids");
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.expect("alloc swma tm out");

    SwmaManySeriesState {
        cuda,
        d_prices_tm,
        d_weights,
        d_first_valids,
        d_out_tm,
        period: period as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_swma_many_series(state: &mut SwmaManySeriesState) {
    state
        .cuda
        .swma_multi_series_one_param_device(
            &state.d_prices_tm,
            &state.d_weights,
            state.period,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn swma_cuda_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("swma_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_swma_many_series_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("time_major_50k_x_2kseries"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_swma_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

// ──────────────────────────────────────────────────────────────
// TRIMA: one series × many periods (batch path)
// ──────────────────────────────────────────────────────────────
struct TrimaBatchState {
    cuda: CudaTrima,
    d_prices: DeviceBuffer<f32>,
    d_periods: DeviceBuffer<i32>,
    d_warms: DeviceBuffer<i32>,
    d_out: DeviceBuffer<f32>,
    series_len: usize,
    n_combos: usize,
    max_period: usize,
}

fn prep_trima_batch_state() -> TrimaBatchState {
    let cuda = CudaTrima::new(0).expect("cuda trima");
    let series_len = 200_000usize;
    let data = gen_series(series_len);
    let data_f32: Vec<f32> = data.into_iter().map(|v| v as f32).collect();
    let sweep = TrimaBatchRange {
        period: (8, 192, 4),
    };
    let periods = expand_trima_periods(&sweep);
    let first_valid = data_f32.iter().position(|x| !x.is_nan()).unwrap_or(0);
    let max_period = *periods.iter().max().unwrap();
    let n_combos = periods.len();
    let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();
    let warms_i32: Vec<i32> = periods
        .iter()
        .map(|&p| (first_valid + p - 1) as i32)
        .collect();

    let d_prices = DeviceBuffer::from_slice(&data_f32).expect("upload trima prices");
    let d_periods = DeviceBuffer::from_slice(&periods_i32).expect("upload trima periods");
    let d_warms = DeviceBuffer::from_slice(&warms_i32).expect("upload trima warms");
    let d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.expect("alloc trima out");

    TrimaBatchState {
        cuda,
        d_prices,
        d_periods,
        d_warms,
        d_out,
        series_len,
        n_combos,
        max_period,
    }
}

fn launch_trima_batch(state: &mut TrimaBatchState) {
    state
        .cuda
        .trima_batch_device(
            &state.d_prices,
            &state.d_periods,
            &state.d_warms,
            state.series_len,
            state.n_combos,
            state.max_period,
            &mut state.d_out,
        )
        .unwrap();
}

fn trima_cuda_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("trima_cuda_batch");
    group.sample_size(10);
    let mut state = prep_trima_batch_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("series_200k_period_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_trima_batch(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

struct TrimaManySeriesState {
    cuda: CudaTrima,
    d_prices_tm: DeviceBuffer<f32>,
    d_weights: DeviceBuffer<f32>,
    d_first_valids: DeviceBuffer<i32>,
    d_out_tm: DeviceBuffer<f32>,
    period: i32,
    num_series: i32,
    series_len: i32,
}

fn prep_trima_many_series_state() -> TrimaManySeriesState {
    let cuda = CudaTrima::new(0).expect("cuda trima");
    let num_series = 2048usize;
    let series_len = 50_000usize;
    let host_tm = gen_time_major_f32(num_series, series_len);
    let period = 30usize;
    let weights = compute_trima_weights(period);
    let mut first_valids = vec![0i32; num_series];
    for series in 0..num_series {
        first_valids[series] = (0..series_len)
            .position(|row| !host_tm[row * num_series + series].is_nan())
            .unwrap_or(0) as i32;
    }

    let d_prices_tm = DeviceBuffer::from_slice(&host_tm).expect("upload trima tm prices");
    let d_weights = DeviceBuffer::from_slice(&weights).expect("upload trima weights");
    let d_first_valids =
        DeviceBuffer::from_slice(&first_valids).expect("upload trima first valids");
    let d_out_tm: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(num_series * series_len) }
            .expect("alloc trima tm out");

    TrimaManySeriesState {
        cuda,
        d_prices_tm,
        d_weights,
        d_first_valids,
        d_out_tm,
        period: period as i32,
        num_series: num_series as i32,
        series_len: series_len as i32,
    }
}

fn launch_trima_many_series(state: &mut TrimaManySeriesState) {
    state
        .cuda
        .trima_multi_series_one_param_device(
            &state.d_prices_tm,
            &state.d_weights,
            state.period,
            state.num_series,
            state.series_len,
            &state.d_first_valids,
            &mut state.d_out_tm,
        )
        .unwrap();
}

fn trima_cuda_many_series_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("trima_cuda_many_series_one_param");
    group.sample_size(10);
    let mut state = prep_trima_many_series_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("time_major_50k_x_2kseries"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_trima_many_series(&mut state);
                black_box(())
            })
        },
    );
    group.finish();
}

fn vwap_cuda_batch_bench(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA (no device)");
        return;
    }
    let mut group = c.benchmark_group("vwap_cuda_batch");
    group.sample_size(10);
    let mut state = prep_vwap_batch_state();
    group.bench_with_input(
        BenchmarkId::from_parameter("series_200k_anchor_grid"),
        &0,
        |b, _| {
            b.iter(|| {
                launch_vwap_batch(&mut state);
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
    alma_many_series_bench,
    edcf_batch_bench,
    ehma_cuda_batch_bench,
    ehma_cuda_many_series_bench,
    ehlers_kama_cuda_batch_bench,
    ehlers_kama_cuda_many_series_bench,
    fwma_cuda_batch_bench,
    fwma_cuda_many_series_bench,
    hwma_cuda_batch_bench,
    hwma_cuda_many_series_bench,
    maaq_cuda_batch_bench,
    maaq_cuda_many_series_bench,
    pwma_cuda_batch_bench,
    pwma_cuda_many_series_bench,
    smma_cuda_batch_bench,
    smma_cuda_many_series_bench,
    swma_cuda_batch_bench,
    swma_cuda_many_series_bench,
    trima_cuda_batch_bench,
    trima_cuda_many_series_bench,
    vwap_cuda_batch_bench,
    wavetrend_batch_bench
);
criterion_main!(benches);
