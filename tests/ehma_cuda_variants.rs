// Integration tests covering EHMA CUDA kernel variants via explicit policy.

use my_project::indicators::moving_averages::ehma::{
    ehma_batch_with_kernel, EhmaBatchRange, EhmaBuilder, EhmaParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::ehma_wrapper::{
    BatchKernelPolicy, BatchThreadsPerOutput, CudaEhma, CudaEhmaPolicy, ManySeriesKernelPolicy,
};

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
            // stagger first_valids
            let x = t as f64 + j as f64 * 0.1;
            v[t * cols + j] = (x * 0.003).cos() + 0.001 * x;
        }
    }
    v
}

#[cfg(feature = "cuda")]
fn compare_batch(
    policy: CudaEhmaPolicy,
    series_len: usize,
    start_period: usize,
    end_period: usize,
) {
    if !cuda_available() {
        eprintln!("[ehma compare_batch] skipped - no CUDA device");
        return;
    }

    let data = gen_series_f64(series_len, 3);
    let sweep = EhmaBatchRange {
        period: (start_period, end_period, 1),
    };

    let cpu = ehma_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch).expect("cpu ehma batch");

    let mut cuda = CudaEhma::new_with_policy(0, policy).expect("cuda ehma");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let dev = cuda
        .ehma_batch_dev(&data_f32, &sweep)
        .expect("gpu ehma batch");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host).expect("copy D2H");

    // EHMA uses FP32 accumulation on GPU vs CPU f64; allow modest tolerance
    let (atol, rtol) = (1.0e-4, 1.0e-4);
    for i in 0..host.len() {
        let a = cpu.values[i];
        let b = host[i] as f64;
        assert!(
            approx_eq(a, b, atol, rtol),
            "batch mismatch at {}: {} vs {}",
            i,
            a,
            b
        );
    }
}

#[cfg(feature = "cuda")]
fn compare_many_series(policy: CudaEhmaPolicy, cols: usize, rows: usize, period: usize) {
    if !cuda_available() {
        eprintln!("[ehma compare_many_series] skipped - no CUDA device");
        return;
    }

    let tm = gen_time_major_f64(cols, rows);
    let params = EhmaParams {
        period: Some(period),
    };

    // CPU per-series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for j in 0..cols {
        let mut s = vec![f64::NAN; rows];
        for t in 0..rows {
            s[t] = tm[t * cols + j];
        }
        let out = EhmaBuilder::default()
            .period(period)
            .apply_slice(&s)
            .expect("cpu ehma series");
        for t in 0..rows {
            cpu_tm[t * cols + j] = out.values[t];
        }
    }

    let mut cuda = CudaEhma::new_with_policy(0, policy).expect("cuda ehma");
    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let dev = cuda
        .ehma_multi_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("gpu many-series");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host).expect("copy D2H");

    let (atol, rtol) = (5.0e-4, 5.0e-4);
    for i in 0..host.len() {
        let a = cpu_tm[i];
        let b = host[i] as f64;
        assert!(
            approx_eq(a, b, atol, rtol),
            "many-series mismatch at {}: {} vs {}",
            i,
            a,
            b
        );
    }
}

// --------- Tests per-kernel variant ---------

#[cfg(feature = "cuda")]
#[test]
fn ehma_batch_plain_matches_cpu() {
    let policy = CudaEhmaPolicy {
        batch: BatchKernelPolicy::Plain { block_x: 256 },
        many_series: ManySeriesKernelPolicy::Auto,
    };
    compare_batch(policy, 16384, 12, 48);
}

#[cfg(feature = "cuda")]
#[test]
fn ehma_batch_tiled_2x_matches_cpu() {
    let policy = CudaEhmaPolicy {
        batch: BatchKernelPolicy::Tiled {
            tile: 256,
            per_thread: BatchThreadsPerOutput::Two,
        },
        many_series: ManySeriesKernelPolicy::Auto,
    };
    compare_batch(policy, 131072, 12, 96);
}

#[cfg(feature = "cuda")]
#[test]
fn ehma_many_series_1d_matches_cpu() {
    let policy = CudaEhmaPolicy {
        batch: BatchKernelPolicy::Auto,
        many_series: ManySeriesKernelPolicy::OneD { block_x: 128 },
    };
    compare_many_series(policy, 16, 8192, 32);
}

#[cfg(feature = "cuda")]
#[test]
fn ehma_many_series_2d_ty4_matches_cpu() {
    let policy = CudaEhmaPolicy {
        batch: BatchKernelPolicy::Auto,
        many_series: ManySeriesKernelPolicy::Tiled2D { tx: 128, ty: 4 },
    };
    compare_many_series(policy, 32, 32768, 64);
}

#[cfg(feature = "cuda")]
#[test]
fn ehma_many_series_2d_ty2_matches_cpu() {
    let policy = CudaEhmaPolicy {
        batch: BatchKernelPolicy::Auto,
        many_series: ManySeriesKernelPolicy::Tiled2D { tx: 128, ty: 2 },
    };
    compare_many_series(policy, 18, 32768, 64);
}
