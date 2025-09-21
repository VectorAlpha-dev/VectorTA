// Integration tests for CUDA WILLR kernels

use my_project::indicators::willr::{WillrBatchBuilder, WillrBatchRange};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaWillr;

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
fn willr_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[willr_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut high = vec![f64::NAN; series_len];
    let mut low = vec![f64::NAN; series_len];
    let mut close = vec![f64::NAN; series_len];

    for i in 5..series_len {
        let x = i as f64;
        let base = (x * 0.002).sin() + 0.001 * x;
        high[i] = base + 0.75;
        low[i] = base - 0.65;
        close[i] = base;
    }

    let sweep = WillrBatchRange { period: (9, 48, 3) };

    let cpu = WillrBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slices(&high, &low, &close)?;

    let cuda = CudaWillr::new(0).expect("CudaWillr::new");
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();

    let gpu_handle = cuda
        .willr_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
        .expect("cuda willr_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda willr batch result to host");

    let tol = 5e-5;
    for idx in 0..(cpu.rows * cpu.cols) {
        let cpu_val = cpu.values[idx];
        let gpu_val = gpu_host[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    Ok(())
}
