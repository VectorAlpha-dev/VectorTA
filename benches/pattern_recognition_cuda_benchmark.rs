#![cfg(feature = "cuda")]

extern crate vector_ta;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cust::memory::DeviceBuffer;
use std::time::Duration;
use vector_ta::cuda::pattern_recognition_wrapper::CudaPatternRecognition;
use vector_ta::indicators::pattern_recognition::list_patterns;

fn env_usize(name: &str, default_v: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().replace('_', "").parse::<usize>().ok())
        .unwrap_or(default_v)
}

fn sample_ohlc(len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut open = Vec::with_capacity(len);
    let mut high = Vec::with_capacity(len);
    let mut low = Vec::with_capacity(len);
    let mut close = Vec::with_capacity(len);

    let mut prev_close: f32 = 100.0;
    for i in 0..len {
        let x = i as f32 * 0.013;
        let o = prev_close + x.sin() * 0.7;
        let c = o + (x * 1.3).cos() * 0.4;
        let h = o.max(c) + 0.6 + (x * 0.7).sin().abs() * 0.2;
        let l = o.min(c) - 0.6 - (x * 0.5).cos().abs() * 0.2;
        open.push(o);
        high.push(h);
        low.push(l);
        close.push(c);
        prev_close = c;
    }

    (open, high, low, close)
}

fn native_row_map() -> Vec<(&'static str, usize)> {
    CudaPatternRecognition::native_supported_pattern_ids()
        .iter()
        .map(|id| {
            let row = list_patterns()
                .iter()
                .find(|spec| spec.id == *id)
                .map(|spec| spec.row_index)
                .expect("pattern row");
            (*id, row)
        })
        .collect()
}

fn bench_pattern_recognition_cuda(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_recognition_cuda");

    if !vector_ta::cuda::cuda_available() {
        group.bench_function("skip_no_cuda", |b| b.iter(|| 0usize));
        group.finish();
        return;
    }

    let len = env_usize("PATTERN_RECOG_CUDA_BENCH_LEN", 100_000);
    let warmup_ms = env_usize("PATTERN_RECOG_CUDA_BENCH_WARMUP_MS", 150) as u64;
    let measure_ms = env_usize("PATTERN_RECOG_CUDA_BENCH_MEASURE_MS", 500) as u64;
    let sample_size = env_usize("PATTERN_RECOG_CUDA_BENCH_SAMPLE_SIZE", 10).max(10);

    let (open, high, low, close) = sample_ohlc(len);
    let row_map = native_row_map();
    let rows = list_patterns().len();
    let cols = len;
    let words_per_row = cols.div_ceil(64);

    let cuda = CudaPatternRecognition::new(0).expect("cuda runtime");
    let d_open = DeviceBuffer::from_slice(open.as_slice()).expect("d_open");
    let d_high = DeviceBuffer::from_slice(high.as_slice()).expect("d_high");
    let d_low = DeviceBuffer::from_slice(low.as_slice()).expect("d_low");
    let d_close = DeviceBuffer::from_slice(close.as_slice()).expect("d_close");
    let features = cuda
        .compute_features_device(&open, &high, &low, &close)
        .expect("feature baseline");
    let matrix = cuda
        .compute_native_matrix_device(&features, rows, cols, row_map.as_slice())
        .expect("matrix baseline");
    cuda.synchronize().expect("baseline sync");

    group.warm_up_time(Duration::from_millis(warmup_ms));
    group.measurement_time(Duration::from_millis(measure_ms));
    group.sample_size(sample_size);

    group.bench_function(BenchmarkId::new("features_host_to_device", len), |b| {
        b.iter(|| {
            let out = cuda
                .compute_features_device(&open, &high, &low, &close)
                .expect("features");
            black_box(out.len());
        })
    });

    group.bench_function(
        BenchmarkId::new("native_matrix_device", format!("{rows}x{cols}")),
        |b| {
            b.iter(|| {
                let out = cuda
                    .compute_native_matrix_device(&features, rows, cols, row_map.as_slice())
                    .expect("matrix");
                cuda.synchronize().expect("matrix sync");
                black_box(out.len());
            })
        },
    );

    group.bench_function(
        BenchmarkId::new("native_matrix_host", format!("{rows}x{cols}")),
        |b| {
            b.iter(|| {
                let out = cuda
                    .compute_native_matrix_host(&features, rows, cols, row_map.as_slice())
                    .expect("matrix host");
                black_box(out.len());
            })
        },
    );

    group.bench_function(
        BenchmarkId::new("pack_u8_to_u64_device", format!("{rows}x{cols}")),
        |b| {
            b.iter(|| {
                let mut words = unsafe { DeviceBuffer::<u64>::uninitialized(rows * words_per_row) }
                    .expect("pack alloc");
                cuda.pack_matrix_u8_device_into(&matrix, rows, cols, &mut words)
                    .expect("pack");
                cuda.synchronize().expect("pack sync");
                black_box(words.len());
            })
        },
    );

    group.bench_function(
        BenchmarkId::new("u8_to_f32_device", format!("{rows}x{cols}")),
        |b| {
            b.iter(|| {
                let out = cuda
                    .matrix_u8_to_f32_device(&matrix, rows, cols)
                    .expect("u8->f32");
                cuda.synchronize().expect("u8->f32 sync");
                black_box((out.rows, out.cols, out.buf.len()));
            })
        },
    );

    group.bench_function(
        BenchmarkId::new("end_to_end_host_matrix", format!("{rows}x{cols}")),
        |b| {
            b.iter(|| {
                let feats = cuda
                    .compute_features_device(&open, &high, &low, &close)
                    .expect("features");
                let out = cuda
                    .compute_native_matrix_host(&feats, rows, cols, row_map.as_slice())
                    .expect("matrix host");
                black_box(out.len());
            })
        },
    );

    group.bench_function(
        BenchmarkId::new("end_to_end_device_inputs_u8", format!("{rows}x{cols}")),
        |b| {
            b.iter(|| {
                let out = cuda
                    .compute_native_matrix_device_from_device_inputs(
                        &d_open, &d_high, &d_low, &d_close, len,
                    )
                    .expect("device inputs u8");
                cuda.synchronize().expect("device inputs u8 sync");
                black_box(out.len());
            })
        },
    );

    group.bench_function(
        BenchmarkId::new("end_to_end_device_inputs_f32", format!("{rows}x{cols}")),
        |b| {
            b.iter(|| {
                let out = cuda
                    .compute_native_matrix_f32_device_from_device_inputs(
                        &d_open, &d_high, &d_low, &d_close, len,
                    )
                    .expect("device inputs f32");
                cuda.synchronize().expect("device inputs f32 sync");
                black_box((out.rows, out.cols, out.buf.len()));
            })
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "end_to_end_device_inputs_packed_u64",
            format!("{rows}x{cols}"),
        ),
        |b| {
            b.iter(|| {
                let out = cuda
                    .compute_native_matrix_bitmask_u64_device_from_device_inputs(
                        &d_open, &d_high, &d_low, &d_close, len,
                    )
                    .expect("device inputs packed u64");
                cuda.synchronize().expect("device inputs packed u64 sync");
                black_box((out.rows, out.cols, out.words_per_row, out.buf.len()));
            })
        },
    );

    group.finish();
}

criterion_group!(benches, bench_pattern_recognition_cuda);
criterion_main!(benches);
