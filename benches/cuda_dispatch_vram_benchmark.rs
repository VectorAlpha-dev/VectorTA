#![cfg(feature = "cuda")]

extern crate vector_ta;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cust::context::CurrentContext;
use std::time::Duration;
use vector_ta::cuda::{CudaDeviceSliceF32Ref, CudaRuntime};
use vector_ta::indicators::dispatch::{
    compute_cuda, compute_cuda_device, CudaOutputTarget, IndicatorCudaDataRef,
    IndicatorCudaDeviceDataRef, IndicatorCudaDeviceRequest, IndicatorCudaRequest,
    IndicatorCudaSeries, ParamKV, ParamValue,
};
use vector_ta::utilities::enums::Kernel;

fn env_usize(name: &str, default_v: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().replace('_', "").parse::<usize>().ok())
        .unwrap_or(default_v)
}

fn gen_prices_f32(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as f64) * 0.001).sin() + ((i as f64) * 0.0001).cos() + 100.0) as f32)
        .collect()
}

fn black_box_cuda_output(out: vector_ta::indicators::dispatch::IndicatorCudaOutput) {
    match out.series {
        IndicatorCudaSeries::HostF32(values) => {
            black_box((out.rows, out.cols, values.len()));
        }
        IndicatorCudaSeries::DeviceF32(values) => {
            black_box((out.rows, out.cols, values.device_ptr, values.device_id));
        }
    }
}

fn synchronize_cuda() {
    CurrentContext::synchronize().expect("cuda context sync");
}

fn device_matrix_row_view(
    matrix: &vector_ta::indicators::dispatch::DeviceMatrixF32,
    row: usize,
) -> CudaDeviceSliceF32Ref {
    assert!(row < matrix.rows, "row out of range");
    let elem_offset = row.checked_mul(matrix.cols).expect("row offset overflow");
    let byte_offset = elem_offset
        .checked_mul(std::mem::size_of::<f32>())
        .expect("byte offset overflow") as u64;
    unsafe {
        CudaDeviceSliceF32Ref::from_raw_parts(
            matrix.device_ptr + byte_offset,
            matrix.cols,
            matrix.device_id,
        )
        .expect("row view")
    }
}

fn bench_cuda_dispatch_vram_resident(c: &mut Criterion) {
    if !vector_ta::cuda::cuda_available() {
        let mut group = c.benchmark_group("cuda_dispatch_vram_resident");
        group.bench_function("skip_no_cuda", |b| b.iter(|| 0usize));
        group.finish();
        return;
    }

    let len = env_usize("CUDA_DISPATCH_VRAM_BENCH_LEN", 200_000);
    let fast_period = env_usize("CUDA_DISPATCH_VRAM_FAST_PERIOD", 50);
    let slow_period = env_usize("CUDA_DISPATCH_VRAM_SLOW_PERIOD", 200);
    let sweep = env_usize("CUDA_DISPATCH_VRAM_SWEEP", 8).max(1);
    let start_period = fast_period;
    let end_period = start_period + sweep.saturating_sub(1);
    let measure_ms = env_usize("CUDA_DISPATCH_VRAM_BENCH_MEASURE_MS", 400) as u64;
    let warmup_ms = env_usize("CUDA_DISPATCH_VRAM_BENCH_WARMUP_MS", 150) as u64;
    let sample_size = env_usize("CUDA_DISPATCH_VRAM_BENCH_SAMPLE_SIZE", 10).max(10);

    let prices = gen_prices_f32(len);
    let runtime = CudaRuntime::new(0).expect("runtime");
    let d_prices = runtime.upload_f32(&prices).expect("upload prices");

    let sma_range_host_params = [
        ParamKV {
            key: "period_start",
            value: ParamValue::Int(start_period as i64),
        },
        ParamKV {
            key: "period_end",
            value: ParamValue::Int(end_period as i64),
        },
        ParamKV {
            key: "period_step",
            value: ParamValue::Int(1),
        },
    ];
    let sma_range_device_params = [
        ParamKV {
            key: "period_start",
            value: ParamValue::Int(start_period as i64),
        },
        ParamKV {
            key: "period_end",
            value: ParamValue::Int(end_period as i64),
        },
        ParamKV {
            key: "period_step",
            value: ParamValue::Int(1),
        },
        ParamKV {
            key: "first_valid",
            value: ParamValue::Int(0),
        },
    ];
    let sma_host_params = [ParamKV {
        key: "period",
        value: ParamValue::Int(fast_period as i64),
    }];
    let sma_device_params = [
        ParamKV {
            key: "period",
            value: ParamValue::Int(fast_period as i64),
        },
        ParamKV {
            key: "first_valid",
            value: ParamValue::Int(0),
        },
    ];
    let ema_host_params = [ParamKV {
        key: "period",
        value: ParamValue::Int(slow_period as i64),
    }];
    let ema_device_params = [
        ParamKV {
            key: "period",
            value: ParamValue::Int(slow_period as i64),
        },
        ParamKV {
            key: "first_valid",
            value: ParamValue::Int(0),
        },
    ];

    let mut group = c.benchmark_group("cuda_dispatch_vram_resident");
    group.warm_up_time(Duration::from_millis(warmup_ms));
    group.measurement_time(Duration::from_millis(measure_ms));
    group.sample_size(sample_size);

    group.bench_function(
        BenchmarkId::new(
            "compat_host_sma_range_host_output",
            format!("{len}x{sweep}"),
        ),
        |b| {
            b.iter(|| {
                let out = compute_cuda(IndicatorCudaRequest {
                    indicator_id: "sma",
                    output_id: None,
                    data: IndicatorCudaDataRef::Slice { values: &prices },
                    params: &sma_range_host_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::HostF32,
                })
                .expect("compat host sma range");
                black_box((out.rows, out.cols));
            })
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "device_core_sma_range_host_output",
            format!("{len}x{sweep}"),
        ),
        |b| {
            b.iter(|| {
                let out = compute_cuda_device(IndicatorCudaDeviceRequest {
                    indicator_id: "sma",
                    output_id: None,
                    data: IndicatorCudaDeviceDataRef::Slice {
                        values: d_prices.as_view(),
                    },
                    params: &sma_range_device_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::HostF32,
                })
                .expect("device sma range");
                black_box((out.rows, out.cols));
            })
        },
    );

    group.bench_function(
        BenchmarkId::new("runtime_upload_prices_only", format!("{len}x1")),
        |b| {
            b.iter(|| {
                let dev = runtime.upload_f32(&prices).expect("upload only");
                synchronize_cuda();
                black_box((dev.device_ptr(), dev.len(), dev.device_id()));
            })
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "compat_host_sma_range_device_output",
            format!("{len}x{sweep}"),
        ),
        |b| {
            b.iter(|| {
                let out = compute_cuda(IndicatorCudaRequest {
                    indicator_id: "sma",
                    output_id: None,
                    data: IndicatorCudaDataRef::Slice { values: &prices },
                    params: &sma_range_host_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::DeviceF32,
                })
                .expect("compat host sma range device output");
                synchronize_cuda();
                black_box_cuda_output(out);
            })
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "device_core_sma_range_device_output",
            format!("{len}x{sweep}"),
        ),
        |b| {
            b.iter(|| {
                let out = compute_cuda_device(IndicatorCudaDeviceRequest {
                    indicator_id: "sma",
                    output_id: None,
                    data: IndicatorCudaDeviceDataRef::Slice {
                        values: d_prices.as_view(),
                    },
                    params: &sma_range_device_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::DeviceF32,
                })
                .expect("device sma range device output");
                synchronize_cuda();
                black_box_cuda_output(out);
            })
        },
    );

    group.bench_function(
        BenchmarkId::new("compat_host_double_ma_pipeline", format!("{len}x1")),
        |b| {
            b.iter(|| {
                let sma = compute_cuda(IndicatorCudaRequest {
                    indicator_id: "sma",
                    output_id: None,
                    data: IndicatorCudaDataRef::Slice { values: &prices },
                    params: &sma_host_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::HostF32,
                })
                .expect("compat sma");
                let ema = compute_cuda(IndicatorCudaRequest {
                    indicator_id: "ema",
                    output_id: None,
                    data: IndicatorCudaDataRef::Slice { values: &prices },
                    params: &ema_host_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::HostF32,
                })
                .expect("compat ema");
                black_box((sma.rows, sma.cols, ema.rows, ema.cols));
            })
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "compat_host_double_ma_pipeline_device_output",
            format!("{len}x1"),
        ),
        |b| {
            b.iter(|| {
                let sma = compute_cuda(IndicatorCudaRequest {
                    indicator_id: "sma",
                    output_id: None,
                    data: IndicatorCudaDataRef::Slice { values: &prices },
                    params: &sma_host_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::DeviceF32,
                })
                .expect("compat sma device");
                let ema = compute_cuda(IndicatorCudaRequest {
                    indicator_id: "ema",
                    output_id: None,
                    data: IndicatorCudaDataRef::Slice { values: &prices },
                    params: &ema_host_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::DeviceF32,
                })
                .expect("compat ema device");
                synchronize_cuda();
                black_box_cuda_output(sma);
                black_box_cuda_output(ema);
            })
        },
    );

    group.bench_function(
        BenchmarkId::new("device_core_double_ma_pipeline", format!("{len}x1")),
        |b| {
            b.iter(|| {
                let sma = compute_cuda_device(IndicatorCudaDeviceRequest {
                    indicator_id: "sma",
                    output_id: None,
                    data: IndicatorCudaDeviceDataRef::Slice {
                        values: d_prices.as_view(),
                    },
                    params: &sma_device_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::HostF32,
                })
                .expect("device sma");
                let ema = compute_cuda_device(IndicatorCudaDeviceRequest {
                    indicator_id: "ema",
                    output_id: None,
                    data: IndicatorCudaDeviceDataRef::Slice {
                        values: d_prices.as_view(),
                    },
                    params: &ema_device_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::HostF32,
                })
                .expect("device ema");
                black_box((sma.rows, sma.cols, ema.rows, ema.cols));
            })
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "compat_host_chained_sma_row_to_ema_host_output",
            format!("{len}x1"),
        ),
        |b| {
            b.iter(|| {
                let first = compute_cuda(IndicatorCudaRequest {
                    indicator_id: "sma",
                    output_id: None,
                    data: IndicatorCudaDataRef::Slice { values: &prices },
                    params: &sma_range_host_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::HostF32,
                })
                .expect("compat first-stage sma range");

                let first_row = match first.series {
                    IndicatorCudaSeries::HostF32(values) => values,
                    other => panic!("expected host output, got {other:?}"),
                };
                let row_slice = &first_row[..len];

                let chained = compute_cuda(IndicatorCudaRequest {
                    indicator_id: "ema",
                    output_id: None,
                    data: IndicatorCudaDataRef::Slice { values: row_slice },
                    params: &ema_host_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::HostF32,
                })
                .expect("compat chained ema");
                black_box_cuda_output(chained);
            })
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "device_core_chained_sma_row_to_ema_host_output",
            format!("{len}x1"),
        ),
        |b| {
            b.iter(|| {
                let first = compute_cuda_device(IndicatorCudaDeviceRequest {
                    indicator_id: "sma",
                    output_id: None,
                    data: IndicatorCudaDeviceDataRef::Slice {
                        values: d_prices.as_view(),
                    },
                    params: &sma_range_device_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::DeviceF32,
                })
                .expect("device first-stage sma range");

                let first_matrix = match first.series {
                    IndicatorCudaSeries::DeviceF32(values) => values,
                    other => panic!("expected device output, got {other:?}"),
                };
                let row_view = device_matrix_row_view(&first_matrix, 0);
                let chained_params = [
                    ParamKV {
                        key: "period",
                        value: ParamValue::Int(slow_period as i64),
                    },
                    ParamKV {
                        key: "first_valid",
                        value: ParamValue::Int((start_period.saturating_sub(1)) as i64),
                    },
                ];
                let chained = compute_cuda_device(IndicatorCudaDeviceRequest {
                    indicator_id: "ema",
                    output_id: None,
                    data: IndicatorCudaDeviceDataRef::Slice { values: row_view },
                    params: &chained_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::HostF32,
                })
                .expect("device chained ema");
                black_box((
                    first_matrix.device_ptr,
                    first_matrix.rows,
                    first_matrix.cols,
                ));
                black_box_cuda_output(chained);
            })
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "device_core_chained_sma_row_to_ema_device_output",
            format!("{len}x1"),
        ),
        |b| {
            b.iter(|| {
                let first = compute_cuda_device(IndicatorCudaDeviceRequest {
                    indicator_id: "sma",
                    output_id: None,
                    data: IndicatorCudaDeviceDataRef::Slice {
                        values: d_prices.as_view(),
                    },
                    params: &sma_range_device_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::DeviceF32,
                })
                .expect("device first-stage sma range");

                let first_matrix = match first.series {
                    IndicatorCudaSeries::DeviceF32(values) => values,
                    other => panic!("expected device output, got {other:?}"),
                };
                let row_view = device_matrix_row_view(&first_matrix, 0);
                let chained_params = [
                    ParamKV {
                        key: "period",
                        value: ParamValue::Int(slow_period as i64),
                    },
                    ParamKV {
                        key: "first_valid",
                        value: ParamValue::Int((start_period.saturating_sub(1)) as i64),
                    },
                ];
                let chained = compute_cuda_device(IndicatorCudaDeviceRequest {
                    indicator_id: "ema",
                    output_id: None,
                    data: IndicatorCudaDeviceDataRef::Slice { values: row_view },
                    params: &chained_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::DeviceF32,
                })
                .expect("device chained ema");
                synchronize_cuda();
                black_box((
                    first_matrix.device_ptr,
                    first_matrix.rows,
                    first_matrix.cols,
                ));
                black_box_cuda_output(chained);
            })
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "device_core_double_ma_pipeline_device_output",
            format!("{len}x1"),
        ),
        |b| {
            b.iter(|| {
                let sma = compute_cuda_device(IndicatorCudaDeviceRequest {
                    indicator_id: "sma",
                    output_id: None,
                    data: IndicatorCudaDeviceDataRef::Slice {
                        values: d_prices.as_view(),
                    },
                    params: &sma_device_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::DeviceF32,
                })
                .expect("device sma device");
                let ema = compute_cuda_device(IndicatorCudaDeviceRequest {
                    indicator_id: "ema",
                    output_id: None,
                    data: IndicatorCudaDeviceDataRef::Slice {
                        values: d_prices.as_view(),
                    },
                    params: &ema_device_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::DeviceF32,
                })
                .expect("device ema device");
                synchronize_cuda();
                black_box_cuda_output(sma);
                black_box_cuda_output(ema);
            })
        },
    );

    group.bench_function(
        BenchmarkId::new(
            "upload_then_device_core_double_ma_pipeline_device_output",
            format!("{len}x1"),
        ),
        |b| {
            b.iter(|| {
                let uploaded = runtime.upload_f32(&prices).expect("upload per iter");
                let data_ref = IndicatorCudaDeviceDataRef::Slice {
                    values: uploaded.as_view(),
                };
                let sma = compute_cuda_device(IndicatorCudaDeviceRequest {
                    indicator_id: "sma",
                    output_id: None,
                    data: data_ref,
                    params: &sma_device_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::DeviceF32,
                })
                .expect("upload+device sma");
                let ema = compute_cuda_device(IndicatorCudaDeviceRequest {
                    indicator_id: "ema",
                    output_id: None,
                    data: data_ref,
                    params: &ema_device_params,
                    kernel: Kernel::Auto,
                    target: CudaOutputTarget::DeviceF32,
                })
                .expect("upload+device ema");
                synchronize_cuda();
                black_box_cuda_output(sma);
                black_box_cuda_output(ema);
            })
        },
    );

    group.finish();
}

criterion_group!(
    cuda_dispatch_vram_benches,
    bench_cuda_dispatch_vram_resident
);
criterion_main!(cuda_dispatch_vram_benches);
