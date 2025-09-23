// Integration tests covering DMA CUDA kernel variants via explicit policy selection.

use my_project::indicators::moving_averages::dma::{
    dma_batch_with_kernel, DmaBatchRange, DmaBuilder, DmaParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::dma_wrapper::{
    BatchKernelPolicy, BatchThreadsPerOutput, CudaDma, CudaDmaPolicy, ManySeriesKernelPolicy,
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
        v[i] = (x * 0.0015).sin() + 0.0002 * x;
    }
    v
}

#[cfg(feature = "cuda")]
fn gen_time_major_f64(cols: usize, rows: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; cols * rows];
    for j in 0..cols {
        for t in j..rows {
            let x = t as f64 + j as f64 * 0.25;
            v[t * cols + j] = (x * 0.0025).sin() + 0.0002 * x;
        }
    }
    v
}

#[cfg(feature = "cuda")]
fn compare_batch(policy: CudaDmaPolicy, series_len: usize, hull_len: usize, ema_len: usize) {
    if !cuda_available() {
        eprintln!("[dma.compare_batch] skipped - no CUDA device");
        return;
    }

    let data = gen_series_f64(series_len, 5);
    // Build a moderately sized grid to force variant selection when needed.
    let sweep = DmaBatchRange {
        hull_length: (hull_len, hull_len + 6, 3),  // 3 combos
        ema_length: (ema_len, ema_len + 8, 4),     // 3 combos
        ema_gain_limit: (10, 30, 10),              // 3 combos
        hull_ma_type: "WMA".to_string(),
    };

    let cpu = dma_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch).expect("cpu dma batch");

    let mut cuda = CudaDma::new_with_policy(0, policy).expect("cuda dma");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let dev = cuda.dma_batch_dev(&data_f32, &sweep).expect("gpu dma batch");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host).expect("copy D2H");

    // DMA mixes recursive EMA with WMA-based hull; GPU uses fp32 while CPU baseline
    // uses f64. For long series and tiled paths, allow a slightly looser tolerance.
    let (atol, rtol) = (5e-3, 6e-4);
    for i in 0..host.len() {
        let a = cpu.values[i];
        let b = host[i] as f64;
        assert!(approx_eq(a, b, atol, rtol), "batch mismatch at {}: {} vs {}", i, a, b);
    }
}

#[cfg(feature = "cuda")]
fn compare_many_series(policy: CudaDmaPolicy, cols: usize, rows: usize, hull_len: usize, ema_len: usize) {
    if !cuda_available() {
        eprintln!("[dma.compare_many_series] skipped - no CUDA device");
        return;
    }

    let tm = gen_time_major_f64(cols, rows);
    let params = DmaParams {
        hull_length: Some(hull_len),
        ema_length: Some(ema_len),
        ema_gain_limit: Some(30),
        hull_ma_type: Some("WMA".to_string()),
    };

    // CPU baseline per-series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for j in 0..cols {
        let mut s = vec![f64::NAN; rows];
        for t in 0..rows { s[t] = tm[t * cols + j]; }
        let out = DmaBuilder::new()
            .hull_length(hull_len)
            .ema_length(ema_len)
            .ema_gain_limit(30)
            .hull_ma_type("WMA".to_string())
            .apply_slice(&s)
            .expect("cpu dma series");
        for t in 0..rows { cpu_tm[t * cols + j] = out.values[t]; }
    }

    let mut cuda = CudaDma::new_with_policy(0, policy).expect("cuda dma");
    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let dev = cuda
        .dma_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("gpu many-series");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host).expect("copy D2H");

    let (atol, rtol) = (4e-4, 6e-4);
    for i in 0..host.len() {
        let a = cpu_tm[i];
        let b = host[i] as f64;
        assert!(approx_eq(a, b, atol, rtol), "many-series mismatch at {}: {} vs {}", i, a, b);
    }
}

// --------- Tests per-kernel variant ---------

#[cfg(feature = "cuda")]
#[test]
fn dma_batch_plain_matches_cpu() {
    let policy = CudaDmaPolicy { batch: BatchKernelPolicy::Plain { block_x: 256 }, many_series: ManySeriesKernelPolicy::Auto };
    // small combos to steer into plain kernel easily
    compare_batch(policy, 32768, 12, 16);
}

#[cfg(feature = "cuda")]
#[test]
fn dma_batch_tiled_matches_cpu() {
    let policy = CudaDmaPolicy { batch: BatchKernelPolicy::Tiled { tile: 128, per_thread: BatchThreadsPerOutput::One }, many_series: ManySeriesKernelPolicy::Auto };
    // larger combos steer into tiled path
    compare_batch(policy, 131072, 12, 24);
}

#[cfg(feature = "cuda")]
#[test]
fn dma_many_series_1d_matches_cpu() {
    let policy = CudaDmaPolicy { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::OneD { block_x: 128 } };
    compare_many_series(policy, 8, 16384, 21, 34);
}

#[cfg(feature = "cuda")]
#[test]
fn dma_many_series_2d_ty4_matches_cpu() {
    let policy = CudaDmaPolicy { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Tiled2D { tx: 1, ty: 4 } };
    compare_many_series(policy, 32, 32768, 21, 34);
}

#[cfg(feature = "cuda")]
#[test]
fn dma_many_series_2d_ty2_matches_cpu() {
    let policy = CudaDmaPolicy { batch: BatchKernelPolicy::Auto, many_series: ManySeriesKernelPolicy::Tiled2D { tx: 1, ty: 2 } };
    compare_many_series(policy, 18, 32768, 21, 34);
}
