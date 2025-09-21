#![cfg(feature = "cuda")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cust::memory::DeviceBuffer;

use my_project::cuda::moving_averages::{
    CudaAlma, CudaCwma, CudaEhlersEcema, CudaEpma, CudaHighpass, CudaKama, CudaNama,     CudaSinwma, CudaSupersmoother3Pole, CudaTradjema, CudaVama, CudaWma,
};
use my_project::cuda::{cuda_available, CudaWto};
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
