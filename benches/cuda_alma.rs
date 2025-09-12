#![cfg(feature = "cuda")]

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use my_project::cuda::{cuda_available};
use my_project::cuda::moving_averages::CudaAlma;
use cust::memory::DeviceBuffer;
use my_project::indicators::moving_averages::alma::{AlmaBatchRange, AlmaParams};

fn gen_series(len: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        v[i] = (x * 0.001).sin() + 0.0001 * x;
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

fn gen_time_major(num_series: usize, series_len: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in j..series_len {
            let x = (t as f64) + (j as f64) * 0.1;
            v[t * num_series + j] = (x * 0.003).cos() + 0.001 * x;
        }
    }
    v
}

fn bench_one_series_many_params(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA ALMA (no device)");
        return;
    }
    let cuda = CudaAlma::new(0).expect("cuda alma");

    // 50,000 points, ~4,000 parameter combinations
    let series_len = 50_000usize;
    let data = gen_series(series_len);
    // We'll create ~4000 combos by combining:
    // - periods: 20 values in [9..=240] step 12 (max period <= 240)
    // - offsets: 10 values in [0.05..=0.95] step 0.10
    // - sigmas: 20 values in [1.5..=11.0] step 0.5
    let sweep = AlmaBatchRange { period: (9, 240, 12), offset: (0.05, 0.95, 0.10), sigma: (1.5, 11.0, 0.5) };

    let mut group = c.benchmark_group("alma_cuda_one_series_many_params");
    // Prepare device buffers once to avoid allocation and copies in the hot loop
    // Build explicit combos grid (~4000)
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
    // Host-side fp32 buffers
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
    // Device buffers
    let d_prices = DeviceBuffer::from_slice(&host_prices_f32).unwrap();
    let d_weights = DeviceBuffer::from_slice(&weights_flat).unwrap();
    let d_periods = DeviceBuffer::from_slice(&periods_i32).unwrap();
    let d_inv_norms = DeviceBuffer::from_slice(&inv_norms).unwrap();
    let mut d_out: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(n_combos * series_len) }.unwrap();

    group.bench_with_input(BenchmarkId::from_parameter("50k_x_4000params"), &series_len, |b, &_n| {
        b.iter(|| {
            cuda.alma_batch_device(
                &d_prices,
                &d_weights,
                &d_periods,
                &d_inv_norms,
                max_period as i32,
                series_len as i32,
                n_combos as i32,
                first_valid,
                &mut d_out,
            )
            .unwrap();
            black_box(d_out.as_device_ptr());
        })
    });
    group.finish();
}

fn bench_many_series_one_param(c: &mut Criterion) {
    if !cuda_available() {
        eprintln!("[bench] skipping CUDA ALMA (no device)");
        return;
    }
    let cuda = CudaAlma::new(0).expect("cuda alma");

    let num_series = 256usize;
    let series_len = 50_000usize;
    let data_tm = gen_time_major(num_series, series_len);
    let params = AlmaParams { period: Some(14), offset: Some(0.85), sigma: Some(6.0) };

    let mut group = c.benchmark_group("alma_cuda_many_series_one_param");
    // Prepare device buffers once
    let host_tm_f32: Vec<f32> = data_tm.iter().map(|&x| x as f32).collect();
    let (w, inv) = compute_weights_cpu_f32(params.period.unwrap(), params.offset.unwrap(), params.sigma.unwrap());
    let mut first_valids = vec![0i32; num_series];
    for j in 0..num_series {
        first_valids[j] = (0..series_len)
            .position(|t| !host_tm_f32[t * num_series + j].is_nan())
            .unwrap_or(0) as i32;
    }
    let d_prices_tm = DeviceBuffer::from_slice(&host_tm_f32).unwrap();
    let d_weights = DeviceBuffer::from_slice(&w).unwrap();
    let d_first_valids = DeviceBuffer::from_slice(&first_valids).unwrap();
    let mut d_out_tm: DeviceBuffer<f32> = unsafe { DeviceBuffer::uninitialized(num_series * series_len) }.unwrap();

    group.bench_with_input(BenchmarkId::from_parameter("256series_x_50k"), &series_len, |b, &_n| {
        b.iter(|| {
            cuda.alma_multi_series_one_param_device(
                &d_prices_tm,
                &d_weights,
                params.period.unwrap() as i32,
                inv,
                num_series as i32,
                series_len as i32,
                &d_first_valids,
                &mut d_out_tm,
            )
            .unwrap();
            black_box(d_out_tm.as_device_ptr());
        })
    });
    group.finish();
}

criterion_group!(benches, bench_one_series_many_params, bench_many_series_one_param);
criterion_main!(benches);
