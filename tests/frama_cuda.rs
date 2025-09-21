// Integration tests for CUDA FRAMA kernels

use my_project::indicators::moving_averages::frama::{
    frama_batch_with_kernel, FramaBatchRange, FramaBuilder, FramaParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaFrama;

fn approx_eq(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs());
    diff <= atol + rtol * scale
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
fn frama_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[frama_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];

    for i in 12..len {
        let t = i as f64 * 0.01;
        let base = (t * 0.4).sin() + 0.05 * (t * 0.7).cos();
        close[i] = base;
        high[i] = base + 0.6 + 0.03 * t.sin();
        low[i] = base - 0.6 - 0.02 * t.cos();
    }

    let sweep = FramaBatchRange {
        window: (10, 18, 4),
        sc: (200, 300, 50),
        fc: (1, 2, 1),
    };

    let cpu = frama_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    let cuda = CudaFrama::new(0).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let (dev, combos) = cuda
        .frama_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    assert_eq!(combos.len(), cpu.rows);
    assert_eq!(dev.rows, cpu.rows);
    assert_eq!(dev.cols, cpu.cols);

    for (idx, combo) in combos.iter().enumerate() {
        let cpu_combo = &cpu.combos[idx];
        assert_eq!(combo.window, cpu_combo.window);
        assert_eq!(combo.sc, cpu_combo.sc);
        assert_eq!(combo.fc, cpu_combo.fc);
    }

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    // FRAMA involves fractal dimension and exp/log operations; the GPU path
    // runs in f32 while the CPU reference uses f64. Allow a slightly looser
    // absolute tolerance to account for expected precision drift.
    // The time-major many-series kernel accumulates in f32 and exhibits a bit
    // more drift than the single-series batch path. Use a slightly wider
    // tolerance band here.
    let atol = 1.5e-2;
    let rtol = 2.5e-2;
    for (idx, gpu_v) in gpu_flat.iter().enumerate() {
        let cpu_v = cpu.values[idx];
        assert!(
            approx_eq(cpu_v, *gpu_v as f64, atol, rtol),
            "Mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_v,
            gpu_v
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn frama_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[frama_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 4usize;
    let rows = 2048usize;
    let mut high_tm = vec![f64::NAN; rows * cols];
    let mut low_tm = vec![f64::NAN; rows * cols];
    let mut close_tm = vec![f64::NAN; rows * cols];

    for row in 15..rows {
        let t = row as f64 * 0.02;
        for col in 0..cols {
            let phase = col as f64 * 0.3;
            let base = (t * 0.35 + phase).sin() + 0.02 * (t * 0.15).cos();
            let stride_idx = row * cols + col;
            close_tm[stride_idx] = base;
            high_tm[stride_idx] = base + 0.4 + 0.01 * (t + phase).sin();
            low_tm[stride_idx] = base - 0.4 - 0.015 * (t + phase).cos();
        }
    }

    let params = FramaParams {
        window: Some(16),
        sc: Some(250),
        fc: Some(2),
    };

    let mut cpu_tm = vec![f64::NAN; rows * cols];
    for col in 0..cols {
        let mut high_series = vec![f64::NAN; rows];
        let mut low_series = vec![f64::NAN; rows];
        let mut close_series = vec![f64::NAN; rows];
        for row in 0..rows {
            let idx = row * cols + col;
            high_series[row] = high_tm[idx];
            low_series[row] = low_tm[idx];
            close_series[row] = close_tm[idx];
        }
        let out = FramaBuilder::new()
            .window(params.window.unwrap())
            .sc(params.sc.unwrap())
            .fc(params.fc.unwrap())
            .apply_slices(&high_series, &low_series, &close_series)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        for row in 0..rows {
            cpu_tm[row * cols + col] = out.values[row];
        }
    }

    let cuda = CudaFrama::new(0).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let dev = cuda
        .frama_many_series_one_param_time_major_dev(
            &high_tm_f32,
            &low_tm_f32,
            &close_tm_f32,
            cols,
            rows,
            &params,
        )
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    // See note above on precision differences between f32 (GPU) and f64 (CPU).
    // The many-series/time-major variant accrues slightly more rounding error.
    let atol = 1.5e-2;
    let rtol = 2.5e-2;
    for (idx, gpu_v) in gpu_flat.iter().enumerate() {
        let cpu_v = cpu_tm[idx];
        assert!(
            approx_eq(cpu_v, *gpu_v as f64, atol, rtol),
            "Mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_v,
            gpu_v
        );
    }

    Ok(())
}
