extern crate vector_ta;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use vector_ta::indicators::moving_averages::ma::MaData;
use vector_ta::indicators::moving_averages::ma_batch::{
    ma_batch_with_kernel, ma_batch_with_kernel_and_typed_params, MaBatchParamKV,
};
use vector_ta::indicators::moving_averages::sma::{sma_batch_with_kernel, SmaBatchRange};
use vector_ta::utilities::enums::Kernel;

fn env_usize(name: &str, default_v: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().replace('_', "").parse::<usize>().ok())
        .unwrap_or(default_v)
}

fn gen_prices_f64(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i as f64) * 0.001).sin() + ((i as f64) * 0.0001).cos() + 100.0)
        .collect()
}

fn bench_cpu_sma_dispatch(c: &mut Criterion) {
    let len = env_usize("MA_DISPATCH_BENCH_LEN", 1_000_000);
    let start = env_usize("MA_DISPATCH_BENCH_START", 2);
    let end = env_usize("MA_DISPATCH_BENCH_END", 251);
    let step = env_usize("MA_DISPATCH_BENCH_STEP", 1);
    let range = (start, end, step);
    let sweep = SmaBatchRange { period: range };
    let prices = gen_prices_f64(len);
    let empty_typed: [MaBatchParamKV<'static>; 0] = [];

    let mut group = c.benchmark_group("ma_dispatch_cpu_sma");
    let warmup_ms = env_usize("MA_DISPATCH_BENCH_WARMUP_MS", 300) as u64;
    let measure_ms = env_usize("MA_DISPATCH_BENCH_MEASURE_MS", 1200) as u64;
    let sample_size = env_usize("MA_DISPATCH_BENCH_SAMPLE_SIZE", 10).max(10);
    group.warm_up_time(Duration::from_millis(warmup_ms));
    group.measurement_time(Duration::from_millis(measure_ms));
    group.sample_size(sample_size);

    group.bench_with_input(
        BenchmarkId::new(
            "direct_sma_batch_with_kernel",
            format!("{}x{}", len, end - start + 1),
        ),
        &prices,
        |b, p| {
            b.iter(|| {
                let out =
                    sma_batch_with_kernel(black_box(p.as_slice()), black_box(&sweep), Kernel::Auto)
                        .unwrap();
                black_box((out.rows, out.cols));
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "generic_ma_batch_with_kernel",
            format!("{}x{}", len, end - start + 1),
        ),
        &prices,
        |b, p| {
            b.iter(|| {
                let out = ma_batch_with_kernel(
                    "sma",
                    MaData::Slice(black_box(p.as_slice())),
                    black_box(range),
                    Kernel::Auto,
                )
                .unwrap();
                black_box((out.rows, out.cols));
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "generic_ma_batch_with_typed_params",
            format!("{}x{}", len, end - start + 1),
        ),
        &prices,
        |b, p| {
            b.iter(|| {
                let out = ma_batch_with_kernel_and_typed_params(
                    "sma",
                    MaData::Slice(black_box(p.as_slice())),
                    black_box(range),
                    Kernel::Auto,
                    &empty_typed,
                )
                .unwrap();
                black_box((out.rows, out.cols));
            })
        },
    );

    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_cuda_sma_dispatch(c: &mut Criterion) {
    use vector_ta::cuda::moving_averages::{CudaMaData, CudaMaSelector, CudaSma};

    if !vector_ta::cuda::cuda_available() {
        let mut group = c.benchmark_group("ma_dispatch_cuda_sma");
        group.bench_function("skip_no_cuda", |b| b.iter(|| 0usize));
        group.finish();
        return;
    }

    let len = env_usize("MA_DISPATCH_BENCH_LEN", 1_000_000);
    let start = env_usize("MA_DISPATCH_BENCH_START", 2);
    let end = env_usize("MA_DISPATCH_BENCH_END", 251);
    let step = env_usize("MA_DISPATCH_BENCH_STEP", 1);
    let range = (start, end, step);
    let sweep = SmaBatchRange { period: range };
    let prices_f32: Vec<f32> = gen_prices_f64(len).into_iter().map(|v| v as f32).collect();
    let selector = CudaMaSelector::new(0);
    let cuda = CudaSma::new(0).unwrap();

    let mut group = c.benchmark_group("ma_dispatch_cuda_sma");
    let warmup_ms = env_usize("MA_DISPATCH_BENCH_WARMUP_MS", 300) as u64;
    let measure_ms = env_usize("MA_DISPATCH_BENCH_MEASURE_MS", 1200) as u64;
    let sample_size = env_usize("MA_DISPATCH_BENCH_SAMPLE_SIZE", 10).max(10);
    group.warm_up_time(Duration::from_millis(warmup_ms));
    group.measurement_time(Duration::from_millis(measure_ms));
    group.sample_size(sample_size);

    group.bench_with_input(
        BenchmarkId::new(
            "direct_cuda_sma_batch_dev",
            format!("{}x{}", len, end - start + 1),
        ),
        &prices_f32,
        |b, p| {
            b.iter(|| {
                let (dev, _combos) = cuda
                    .sma_batch_dev(black_box(p.as_slice()), black_box(&sweep))
                    .unwrap();
                black_box((dev.rows, dev.cols));
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "generic_cuda_selector_sweep",
            format!("{}x{}", len, end - start + 1),
        ),
        &prices_f32,
        |b, p| {
            b.iter(|| {
                let dev = selector
                    .ma_sweep_to_device(
                        "sma",
                        CudaMaData::SliceF32(black_box(p.as_slice())),
                        start,
                        end,
                        step,
                    )
                    .unwrap();
                black_box((dev.rows, dev.cols));
            })
        },
    );

    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn bench_cuda_sma_dispatch(_c: &mut Criterion) {}

criterion_group!(
    ma_dispatch_benches,
    bench_cpu_sma_dispatch,
    bench_cuda_sma_dispatch
);
criterion_main!(ma_dispatch_benches);
