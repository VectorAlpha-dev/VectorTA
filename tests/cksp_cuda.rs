// Integration tests for CUDA CKSP kernels

use my_project::indicators::cksp::{CkspBatchBuilder, CkspBatchRange, CkspParams};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cksp_wrapper::CudaCksp;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cksp_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cksp_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let n = 4096usize;
    let mut high = vec![f64::NAN; n];
    let mut low = vec![f64::NAN; n];
    let mut close = vec![f64::NAN; n];
    for i in 10..n {
        let x = i as f64;
        let base = (x * 0.003).sin() + 0.0005 * x;
        high[i] = base + 0.8;
        low[i] = base - 0.7;
        close[i] = base;
    }

    let sweep = CkspBatchRange {
        p: (5, 29, 4),
        x: (1.0, 1.0, 0.0),
        q: (7, 13, 3),
    };

    let cpu = CkspBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .p_range(sweep.p.0, sweep.p.1, sweep.p.2)
        .x_static(1.0)
        .q_range(sweep.q.0, sweep.q.1, sweep.q.2)
        .apply_slices(&high, &low, &close)?;

    let cuda = CudaCksp::new(0).expect("CudaCksp::new");
    let hf: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let (dev_pair, combos) = cuda
        .cksp_batch_dev(&hf, &lf, &cf, &sweep)
        .expect("cksp batch dev");
    assert_eq!(combos.len(), cpu.rows);
    assert_eq!(dev_pair.long.rows, cpu.rows);
    assert_eq!(dev_pair.long.cols, cpu.cols);
    assert_eq!(dev_pair.short.rows, cpu.rows);
    assert_eq!(dev_pair.short.cols, cpu.cols);

    let mut long_host = vec![0f32; dev_pair.long.len()];
    let mut short_host = vec![0f32; dev_pair.short.len()];
    dev_pair.long.buf.copy_to(&mut long_host).unwrap();
    dev_pair.short.buf.copy_to(&mut short_host).unwrap();

    let tol = 5e-4; // FP32 device vs f64 scalar
    for idx in 0..(cpu.rows * cpu.cols) {
        let cpu_l = cpu.long_values[idx];
        let cpu_s = cpu.short_values[idx];
        let gpu_l = long_host[idx] as f64;
        let gpu_s = short_host[idx] as f64;
        assert!(
            approx_eq(cpu_l, gpu_l, tol),
            "long mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_l,
            gpu_l
        );
        assert!(
            approx_eq(cpu_s, gpu_s, tol),
            "short mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_s,
            gpu_s
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cksp_cuda_large_p_sweep_smoke() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cksp_cuda_large_p_sweep_smoke] skipped - no CUDA device");
        return Ok(());
    }

    let n = 4096usize;
    let mut high = vec![f64::NAN; n];
    let mut low = vec![f64::NAN; n];
    let mut close = vec![f64::NAN; n];
    for i in 10..n {
        let x = i as f64;
        let base = (x * 0.003).sin() + 0.0005 * x;
        high[i] = base + 0.8;
        low[i] = base - 0.7;
        close[i] = base;
    }

    // Wider p sweep than the accuracy test; this is a smoke test to catch
    // large-sweep launch/indexing issues without changing reference outputs.
    let sweep = CkspBatchRange {
        p: (10, 137, 1),
        x: (1.0, 1.0, 0.0),
        q: (9, 9, 0),
    };

    let cuda = CudaCksp::new(0).expect("CudaCksp::new");
    let hf: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let (dev_pair, combos) = cuda
        .cksp_batch_dev(&hf, &lf, &cf, &sweep)
        .expect("cksp batch dev (smoke)");
    assert_eq!(combos.len(), 128);
    assert_eq!(dev_pair.long.rows, combos.len());
    assert_eq!(dev_pair.short.rows, combos.len());
    assert_eq!(dev_pair.long.cols, n);
    assert_eq!(dev_pair.short.cols, n);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cksp_cuda_large_series_smoke() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cksp_cuda_large_series_smoke] skipped - no CUDA device");
        return Ok(());
    }

    let n = 20_000usize;
    let mut high = vec![f64::NAN; n];
    let mut low = vec![f64::NAN; n];
    let mut close = vec![f64::NAN; n];
    for i in 10..n {
        let x = i as f64;
        let base = (x * 0.003).sin() + 0.0005 * x;
        high[i] = base + 0.8;
        low[i] = base - 0.7;
        close[i] = base;
    }

    let sweep = CkspBatchRange {
        p: (10, 137, 1),
        x: (1.0, 1.0, 0.0),
        q: (9, 9, 0),
    };

    let cuda = CudaCksp::new(0).expect("CudaCksp::new");
    let hf: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let (dev_pair, combos) = cuda
        .cksp_batch_dev(&hf, &lf, &cf, &sweep)
        .expect("cksp batch dev (large series smoke)");
    assert_eq!(combos.len(), 128);
    assert_eq!(dev_pair.long.rows, combos.len());
    assert_eq!(dev_pair.short.rows, combos.len());
    assert_eq!(dev_pair.long.cols, n);
    assert_eq!(dev_pair.short.cols, n);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cksp_cuda_single_combo_smoke() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cksp_cuda_single_combo_smoke] skipped - no CUDA device");
        return Ok(());
    }

    let n = 4096usize;
    let mut high = vec![f64::NAN; n];
    let mut low = vec![f64::NAN; n];
    let mut close = vec![f64::NAN; n];
    for i in 10..n {
        let x = i as f64;
        let base = (x * 0.003).sin() + 0.0005 * x;
        high[i] = base + 0.8;
        low[i] = base - 0.7;
        close[i] = base;
    }

    // Single combo: exercises non-preTR path.
    let sweep = CkspBatchRange {
        p: (10, 10, 0),
        x: (1.0, 1.0, 0.0),
        q: (9, 9, 0),
    };

    let cuda = CudaCksp::new(0).expect("CudaCksp::new");
    let hf: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let (dev_pair, combos) = cuda
        .cksp_batch_dev(&hf, &lf, &cf, &sweep)
        .expect("cksp batch dev (single combo)");
    assert_eq!(combos.len(), 1);
    assert_eq!(dev_pair.long.rows, 1);
    assert_eq!(dev_pair.short.rows, 1);
    assert_eq!(dev_pair.long.cols, n);
    assert_eq!(dev_pair.short.cols, n);
    Ok(())
}
