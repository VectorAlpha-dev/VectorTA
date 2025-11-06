// Integration tests covering ALMA CUDA kernels via explicit policy.

use my_project::indicators::moving_averages::alma::{
    alma_batch_with_kernel, AlmaBatchRange, AlmaBuilder, AlmaParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::alma_wrapper::{
    BatchKernelPolicy, BatchThreadsPerOutput, CudaAlma, CudaAlmaPolicy, ManySeriesKernelPolicy,
};

#[cfg(feature = "cuda")]
fn should_force_skip_cuda() -> bool {
    std::env::var("SKIP_CUDA_TESTS").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false)
}

fn approx_eq(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    diff <= atol + rtol * a.abs().max(b.abs())
}

#[test]
fn cuda_feature_off_compiles() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
fn gen_series_f64(len: usize, nan_lead: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; len];
    for i in nan_lead..len {
        let x = i as f64;
        v[i] = (x * 0.001).sin() + 0.0001 * x;
    }
    v
}

#[cfg(feature = "cuda")]
fn gen_time_major_f64(cols: usize, rows: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; cols * rows];
    for j in 0..cols {
        for t in j..rows {
            let x = t as f64 + j as f64 * 0.1;
            v[t * cols + j] = (x * 0.003).cos() + 0.001 * x;
        }
    }
    v
}

#[cfg(feature = "cuda")]
fn compare_batch(
    policy: CudaAlmaPolicy,
    series_len: usize,
    start_period: usize,
    end_period: usize,
) {
    if should_force_skip_cuda() || !cuda_available() {
        eprintln!("[compare_batch] skipped - no CUDA device");
        return;
    }
    let data = gen_series_f64(series_len, 3);
    let sweep = AlmaBatchRange {
        period: (start_period, end_period, 1),
        offset: (0.85, 0.85, 0.0),
        sigma: (6.0, 6.0, 0.0),
    };
    let cpu = alma_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch).expect("cpu alma batch");
    let cuda = CudaAlma::new_with_policy(0, policy).expect("cuda alma");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let dev = cuda
        .alma_batch_dev(&data_f32, &sweep)
        .expect("gpu alma batch");
    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host).expect("copy D2H");
    let (atol, rtol) = (1e-5, 1e-5);
    for i in 0..host.len() {
        assert!(approx_eq(cpu.values[i], host[i] as f64, atol, rtol));
    }
}

#[cfg(feature = "cuda")]
fn compare_many_series(policy: CudaAlmaPolicy, cols: usize, rows: usize, period: usize) {
    if should_force_skip_cuda() || !cuda_available() {
        eprintln!("[compare_many_series] skipped - no CUDA device");
        return;
    }
    let tm = gen_time_major_f64(cols, rows);
    let params = AlmaParams {
        period: Some(period),
        offset: Some(0.85),
        sigma: Some(6.0),
    };
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for j in 0..cols {
        let mut s = vec![f64::NAN; rows];
        for t in 0..rows {
            s[t] = tm[t * cols + j];
        }
        let out = AlmaBuilder::default()
            .period(period)
            .offset(0.85)
            .sigma(6.0)
            .apply_slice(&s)
            .expect("cpu alma series");
        for t in 0..rows {
            cpu_tm[t * cols + j] = out.values[t];
        }
    }
    let cuda = CudaAlma::new_with_policy(0, policy).expect("cuda alma");
    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let dev = cuda
        .alma_multi_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("gpu many-series");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host).expect("copy D2H");
    let (atol, rtol) = (1e-3, 1e-3);
    for i in 0..host.len() {
        assert!(approx_eq(cpu_tm[i], host[i] as f64, atol, rtol));
    }
}

// --------- Tests per-kernel variant ---------

#[cfg(feature = "cuda")]
#[test]
fn alma_batch_plain_matches_cpu() {
    let policy = CudaAlmaPolicy {
        batch: BatchKernelPolicy::Plain { block_x: 256 },
        many_series: ManySeriesKernelPolicy::Auto,
    };
    compare_batch(policy, 16384, 12, 48);
}

#[cfg(feature = "cuda")]
#[test]
fn alma_batch_tiled_1x_matches_cpu() {
    let policy = CudaAlmaPolicy {
        batch: BatchKernelPolicy::Tiled {
            tile: 256,
            per_thread: BatchThreadsPerOutput::One,
        },
        many_series: ManySeriesKernelPolicy::Auto,
    };
    compare_batch(policy, 131072, 12, 96);
}

#[cfg(feature = "cuda")]
#[test]
fn alma_batch_tiled_2x_matches_cpu() {
    let policy = CudaAlmaPolicy {
        batch: BatchKernelPolicy::Tiled {
            tile: 256,
            per_thread: BatchThreadsPerOutput::Two,
        },
        many_series: ManySeriesKernelPolicy::Auto,
    };
    compare_batch(policy, 131072, 12, 128);
}

#[cfg(feature = "cuda")]
#[test]
fn alma_many_series_1d_matches_cpu() {
    let policy = CudaAlmaPolicy {
        batch: BatchKernelPolicy::Auto,
        many_series: ManySeriesKernelPolicy::OneD { block_x: 128 },
    };
    compare_many_series(policy, 16, 8192, 32);
}

#[cfg(feature = "cuda")]
#[test]
fn alma_many_series_2d_ty4_matches_cpu() {
    let policy = CudaAlmaPolicy {
        batch: BatchKernelPolicy::Auto,
        many_series: ManySeriesKernelPolicy::Tiled2D { tx: 128, ty: 4 },
    };
    compare_many_series(policy, 32, 32768, 64);
}

#[cfg(feature = "cuda")]
#[test]
fn alma_many_series_2d_ty2_matches_cpu() {
    let policy = CudaAlmaPolicy {
        batch: BatchKernelPolicy::Auto,
        many_series: ManySeriesKernelPolicy::Tiled2D { tx: 128, ty: 2 },
    };
    compare_many_series(policy, 18, 32768, 64);
}
