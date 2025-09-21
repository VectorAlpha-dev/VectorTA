// CUDA integration tests for Buff Averages indicator.

use my_project::indicators::moving_averages::buff_averages::{
    buff_averages_batch_with_kernel, BuffAveragesBatchRange,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaBuffAverages;

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
fn buff_averages_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[buff_averages_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut price = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        price[i] = (x * 0.001).sin() + 0.0001 * x;
        volume[i] = (x * 0.0007).cos().abs() + 0.5;
    }

    let sweep = BuffAveragesBatchRange {
        fast_period: (5, 25, 5),
        slow_period: (30, 90, 15),
    };

    let cpu = buff_averages_batch_with_kernel(&price, &volume, &sweep, Kernel::ScalarBatch)?;
    let cpu_fast_f32: Vec<f32> = cpu.fast.iter().map(|&v| v as f32).collect();
    let cpu_slow_f32: Vec<f32> = cpu.slow.iter().map(|&v| v as f32).collect();

    let cuda = CudaBuffAverages::new(0).expect("CudaBuffAverages::new");
    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let (fast_dev, slow_dev) = cuda
        .buff_averages_batch_dev(&price_f32, &volume_f32, &sweep)
        .expect("buff_averages_cuda");

    assert_eq!(cpu.rows, fast_dev.rows);
    assert_eq!(cpu.cols, fast_dev.cols);
    assert_eq!(fast_dev.rows, slow_dev.rows);
    assert_eq!(fast_dev.cols, slow_dev.cols);

    let mut fast_gpu = vec![0f32; fast_dev.len()];
    let mut slow_gpu = vec![0f32; slow_dev.len()];
    fast_dev
        .buf
        .copy_to(&mut fast_gpu)
        .expect("copy fast cuda results");
    slow_dev
        .buf
        .copy_to(&mut slow_gpu)
        .expect("copy slow cuda results");

    let tol = 2e-4;
    for (idx, (&cpu_fast, &gpu_fast)) in cpu_fast_f32.iter().zip(fast_gpu.iter()).enumerate() {
        assert!(
            approx_eq(cpu_fast as f64, gpu_fast as f64, tol),
            "fast mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_fast,
            gpu_fast
        );
    }
    for (idx, (&cpu_slow, &gpu_slow)) in cpu_slow_f32.iter().zip(slow_gpu.iter()).enumerate() {
        assert!(
            approx_eq(cpu_slow as f64, gpu_slow as f64, tol),
            "slow mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_slow,
            gpu_slow
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn buff_averages_cuda_host_copy_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[buff_averages_cuda_host_copy_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut price = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 2..len {
        let x = i as f64;
        price[i] = (x * 0.0009).cos() + 0.0002 * x;
        volume[i] = (x * 0.0005).sin().abs() + 0.75;
    }

    let sweep = BuffAveragesBatchRange {
        fast_period: (4, 10, 3),
        slow_period: (12, 24, 6),
    };

    let cpu = buff_averages_batch_with_kernel(&price, &volume, &sweep, Kernel::ScalarBatch)?;
    let cpu_fast_f32: Vec<f32> = cpu.fast.iter().map(|&v| v as f32).collect();
    let cpu_slow_f32: Vec<f32> = cpu.slow.iter().map(|&v| v as f32).collect();

    let cuda = CudaBuffAverages::new(0).expect("CudaBuffAverages::new");
    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let mut fast_gpu = vec![0f32; cpu.fast.len()];
    let mut slow_gpu = vec![0f32; cpu.slow.len()];
    let (rows, cols, combos) = cuda
        .buff_averages_batch_into_host_f32(
            &price_f32,
            &volume_f32,
            &sweep,
            &mut fast_gpu,
            &mut slow_gpu,
        )
        .expect("buff_averages_cuda_host_copy");

    assert_eq!(rows, cpu.rows);
    assert_eq!(cols, cpu.cols);
    assert_eq!(combos, cpu.combos);

    let tol = 2e-4;
    for (idx, (&cpu_fast, &gpu_fast)) in cpu_fast_f32.iter().zip(fast_gpu.iter()).enumerate() {
        assert!(
            approx_eq(cpu_fast as f64, gpu_fast as f64, tol),
            "fast mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_fast,
            gpu_fast
        );
    }
    for (idx, (&cpu_slow, &gpu_slow)) in cpu_slow_f32.iter().zip(slow_gpu.iter()).enumerate() {
        assert!(
            approx_eq(cpu_slow as f64, gpu_slow as f64, tol),
            "slow mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_slow,
            gpu_slow
        );
    }

    Ok(())
}
