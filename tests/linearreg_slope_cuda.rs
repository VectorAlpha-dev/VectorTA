#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaLinearregSlope;

use my_project::indicators::linearreg_slope::{
    linearreg_slope_batch_with_kernel, LinearRegSlopeBatchRange, LinearRegSlopeBuilder,
};
use my_project::utilities::enums::Kernel;

fn make_test_series(len: usize) -> Vec<f64> {
    let mut data = vec![f64::NAN; len];
    for i in 12..len {
        let t = i as f64;
        let trend = 0.0015 * t;
        let curve = (t * 0.015).sin() * 0.6 + (t * 0.005).cos() * 0.4;
        let noise = ((i * 17 % 29) as f64) as f64 * 0.0005;
        data[i] = trend + curve + noise;
    }
    data
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
fn compare_rows(cpu: &[f64], gpu: &[f64], periods: &[usize], len: usize, first_valid: usize) {
    for (row_idx, &period) in periods.iter().enumerate() {
        let warm = first_valid + period - 1;
        for col in 0..len {
            let idx = row_idx * len + col;
            let expected = cpu[idx];
            let actual = gpu[idx];
            if col < warm {
                assert!(expected.is_nan(), "CPU warmup NaN missing at row {row_idx} col {col}");
                assert!(actual.is_nan(), "CUDA warmup mismatch at row {row_idx} col {col}");
            } else {
                let diff = (expected - actual).abs();
                let tol = 3.0e-3 + expected.abs() * 8.0e-4; // slope tolerances
                assert!(
                    diff <= tol,
                    "row {row_idx} col {col} expected {expected} got {actual} diff {diff} tol {tol}"
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn linearreg_slope_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[linearreg_slope_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096;
    let data = make_test_series(len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap();
    let sweep = LinearRegSlopeBatchRange { period: (8, 36, 4) };

    let cpu = linearreg_slope_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let periods: Vec<usize> = cpu
        .combos
        .iter()
        .map(|p| p.period.unwrap())
        .collect();

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaLinearregSlope::new(0).expect("CudaLinearregSlope::new");
    let (dev, combos_gpu) = cuda
        .linearreg_slope_batch_dev(&data_f32, &sweep)
        .expect("linearreg_slope_batch_dev");

    for (c, g) in cpu.combos.iter().zip(&combos_gpu) {
        assert_eq!(c.period, g.period);
    }

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_flat)?;
    let gpu_flat_f64: Vec<f64> = gpu_flat.iter().map(|&v| v as f64).collect();

    compare_rows(&cpu.values, &gpu_flat_f64, &periods, len, first_valid);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn linearreg_slope_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[linearreg_slope_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let rows = 1536usize;
    let cols = 5usize;
    let period = 18usize;

    let mut data_tm = vec![f64::NAN; rows * cols];
    for col in 0..cols {
        for row in (col * 4)..rows {
            let t = row as f64 + col as f64 * 0.91;
            let drift = 0.00087 * t;
            let wave = (t * 0.011).sin() * 0.55 + (t * 0.007).cos() * 0.45;
            data_tm[row * cols + col] = drift + wave;
        }
    }

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; rows * cols];
    for col in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for row in 0..rows {
            series[row] = data_tm[row * cols + col];
        }
        let out = LinearRegSlopeBuilder::new().period(period).apply_slice(&series)?;
        for row in 0..rows {
            cpu_tm[row * cols + col] = out.values[row];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let params = my_project::indicators::linearreg_slope::LinearRegSlopeParams {
        period: Some(period),
    };
    let cuda = CudaLinearregSlope::new(0).expect("CudaLinearregSlope::new");
    let dev = cuda
        .linearreg_slope_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("linearreg_slope_many_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_tm)?;
    let gpu_tm_f64: Vec<f64> = gpu_tm.iter().map(|&v| v as f64).collect();

    for idx in 0..rows * cols {
        let expected = cpu_tm[idx];
        let actual = gpu_tm_f64[idx];
        if expected.is_nan() {
            assert!(actual.is_nan(), "CUDA warmup mismatch at idx {idx}");
        } else {
            let diff = (expected - actual).abs();
            let tol = 3.0e-3 + expected.abs() * 9.0e-4;
            assert!(
                diff <= tol,
                "idx {idx} expected {expected} got {actual} diff {diff} tol {tol}"
            );
        }
    }

    Ok(())
}

