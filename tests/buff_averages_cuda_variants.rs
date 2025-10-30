// Variant coverage tests for Buff Averages CUDA wrapper

#![cfg(feature = "cuda")]

use cust::memory::CopyDestination;
use my_project::cuda::cuda_available;
use my_project::cuda::moving_averages::buff_averages_wrapper::{
    BatchKernelPolicy, CudaBuffAverages, CudaBuffPolicy, ManySeriesKernelPolicy,
};
use my_project::indicators::moving_averages::buff_averages::{
    buff_averages_batch_with_kernel, BuffAveragesBatchRange,
};
use my_project::utilities::enums::Kernel;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

fn using_nvcc_stub() -> bool {
    std::env::var("NVCC")
        .map(|p| p.contains("nvcc_stub.sh"))
        .unwrap_or(false)
}

fn synth(len: usize) -> (Vec<f64>, Vec<f64>) {
    let mut price = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        price[i] = (x * 0.001).sin() + 0.0001 * x;
        volume[i] = (x * 0.0007).cos().abs() + 0.6;
    }
    (price, volume)
}

fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

#[test]
fn buff_averages_cuda_plain_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if using_nvcc_stub() {
        eprintln!("[buff_averages_cuda_plain_matches_cpu] skipped - NVCC stub in use (placeholder PTX)");
        return Ok(());
    }
    if !cuda_available() {
        eprintln!("[buff_averages_cuda_plain_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 12_000usize; // ensure plenty of room
    let (price, volume) = synth(len);
    let sweep = BuffAveragesBatchRange {
        fast_period: (4, 20, 4),
        slow_period: (24, 80, 14),
    };
    let cpu = buff_averages_batch_with_kernel(&price, &volume, &sweep, Kernel::ScalarBatch)?;
    let cpu_fast_f32: Vec<f32> = cpu.fast.iter().map(|&v| v as f32).collect();
    let cpu_slow_f32: Vec<f32> = cpu.slow.iter().map(|&v| v as f32).collect();

    let mut cuda = CudaBuffAverages::new(0)?;
    cuda.set_policy(CudaBuffPolicy {
        batch: BatchKernelPolicy::Plain { block_x: 256 },
        many_series: ManySeriesKernelPolicy::Auto,
    });
    let (fast_dev, slow_dev) =
        cuda.buff_averages_batch_dev(&to_f32(&price), &to_f32(&volume), &sweep)?;
    assert_eq!(cpu.rows, fast_dev.rows);
    assert_eq!(cpu.cols, fast_dev.cols);
    assert_eq!(fast_dev.rows, slow_dev.rows);
    assert_eq!(fast_dev.cols, slow_dev.cols);

    let mut fast_gpu = vec![0f32; fast_dev.len()];
    let mut slow_gpu = vec![0f32; slow_dev.len()];
    fast_dev.buf.copy_to(&mut fast_gpu)?;
    slow_dev.buf.copy_to(&mut slow_gpu)?;

    let tol = 2e-3f32;
    for (idx, (&cf, &gf)) in cpu_fast_f32.iter().zip(fast_gpu.iter()).enumerate() {
        assert!(
            approx_eq(cf, gf, tol),
            "plain fast mismatch at {}: cpu={} gpu={}",
            idx,
            cf,
            gf
        );
    }
    for (idx, (&cs, &gs)) in cpu_slow_f32.iter().zip(slow_gpu.iter()).enumerate() {
        assert!(
            approx_eq(cs, gs, tol),
            "plain slow mismatch at {}: cpu={} gpu={}",
            idx,
            cs,
            gs
        );
    }
    Ok(())
}

#[test]
fn buff_averages_cuda_tiled128_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if using_nvcc_stub() {
        eprintln!("[buff_averages_cuda_tiled128_matches_cpu] skipped - NVCC stub in use (placeholder PTX)");
        return Ok(());
    }
    if !cuda_available() {
        eprintln!("[buff_averages_cuda_tiled128_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 20_000usize;
    let (price, volume) = synth(len);
    let sweep = BuffAveragesBatchRange {
        fast_period: (4, 20, 4),
        slow_period: (24, 80, 14),
    };
    let cpu = buff_averages_batch_with_kernel(&price, &volume, &sweep, Kernel::ScalarBatch)?;
    let cpu_fast_f32: Vec<f32> = cpu.fast.iter().map(|&v| v as f32).collect();
    let cpu_slow_f32: Vec<f32> = cpu.slow.iter().map(|&v| v as f32).collect();

    let mut cuda = CudaBuffAverages::new(0)?;
    cuda.set_policy(CudaBuffPolicy {
        batch: BatchKernelPolicy::Tiled { tile: 128 },
        many_series: ManySeriesKernelPolicy::Auto,
    });
    let (fast_dev, slow_dev) =
        cuda.buff_averages_batch_dev(&to_f32(&price), &to_f32(&volume), &sweep)?;

    let mut fast_gpu = vec![0f32; fast_dev.len()];
    let mut slow_gpu = vec![0f32; slow_dev.len()];
    fast_dev.buf.copy_to(&mut fast_gpu)?;
    slow_dev.buf.copy_to(&mut slow_gpu)?;

    let tol = 2e-3f32;
    for (idx, (&cf, &gf)) in cpu_fast_f32.iter().zip(fast_gpu.iter()).enumerate() {
        assert!(
            approx_eq(cf, gf, tol),
            "tiled128 fast mismatch at {}: cpu={} gpu={}",
            idx,
            cf,
            gf
        );
    }
    for (idx, (&cs, &gs)) in cpu_slow_f32.iter().zip(slow_gpu.iter()).enumerate() {
        assert!(
            approx_eq(cs, gs, tol),
            "tiled128 slow mismatch at {}: cpu={} gpu={}",
            idx,
            cs,
            gs
        );
    }
    Ok(())
}

#[test]
fn buff_averages_cuda_tiled256_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if using_nvcc_stub() {
        eprintln!("[buff_averages_cuda_tiled256_matches_cpu] skipped - NVCC stub in use (placeholder PTX)");
        return Ok(());
    }
    if !cuda_available() {
        eprintln!("[buff_averages_cuda_tiled256_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 20_000usize;
    let (price, volume) = synth(len);
    let sweep = BuffAveragesBatchRange {
        fast_period: (4, 20, 4),
        slow_period: (24, 80, 14),
    };
    let cpu = buff_averages_batch_with_kernel(&price, &volume, &sweep, Kernel::ScalarBatch)?;
    let cpu_fast_f32: Vec<f32> = cpu.fast.iter().map(|&v| v as f32).collect();
    let cpu_slow_f32: Vec<f32> = cpu.slow.iter().map(|&v| v as f32).collect();

    let mut cuda = CudaBuffAverages::new(0)?;
    cuda.set_policy(CudaBuffPolicy {
        batch: BatchKernelPolicy::Tiled { tile: 256 },
        many_series: ManySeriesKernelPolicy::Auto,
    });
    let (fast_dev, slow_dev) =
        cuda.buff_averages_batch_dev(&to_f32(&price), &to_f32(&volume), &sweep)?;

    let mut fast_gpu = vec![0f32; fast_dev.len()];
    let mut slow_gpu = vec![0f32; slow_dev.len()];
    fast_dev.buf.copy_to(&mut fast_gpu)?;
    slow_dev.buf.copy_to(&mut slow_gpu)?;

    let tol = 2e-3f32;
    for (idx, (&cf, &gf)) in cpu_fast_f32.iter().zip(fast_gpu.iter()).enumerate() {
        assert!(
            approx_eq(cf, gf, tol),
            "tiled256 fast mismatch at {}: cpu={} gpu={}",
            idx,
            cf,
            gf
        );
    }
    for (idx, (&cs, &gs)) in cpu_slow_f32.iter().zip(slow_gpu.iter()).enumerate() {
        assert!(
            approx_eq(cs, gs, tol),
            "tiled256 slow mismatch at {}: cpu={} gpu={}",
            idx,
            cs,
            gs
        );
    }
    Ok(())
}
