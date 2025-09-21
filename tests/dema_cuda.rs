// Integration tests for CUDA DEMA kernels

use my_project::indicators::moving_averages::dema::{DemaBatchBuilder, DemaBatchRange};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaDema;

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
fn dema_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dema_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 2048usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 3..series_len {
        let x = i as f64;
        data[i] = (x * 0.002).cos() + 0.0003 * x;
    }

    let sweep = DemaBatchRange { period: (9, 64, 1) };

    let cpu = DemaBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slice(&data)?;

    let cuda = CudaDema::new(0).expect("CudaDema::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .dema_batch_dev(&data_f32, &sweep)
        .expect("cuda dema_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda dema batch result to host");

    let tol = 1e-5;
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
