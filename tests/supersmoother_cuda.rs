#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaSuperSmoother;
use vector_ta::indicators::moving_averages::supersmoother::{
    expand_grid_supersmoother, supersmoother_batch_inner_into, SuperSmootherBatchRange,
    SuperSmootherBuilder, SuperSmootherParams,
};
use vector_ta::utilities::enums::Kernel;

fn make_test_series(len: usize) -> Vec<f64> {
    let mut data = vec![f64::NAN; len];
    for i in 8..len {
        let x = i as f64;
        let trend = 0.00042 * x;
        let wave = (x * 0.0035).sin() + (x * 0.0017).cos();
        let noise = ((i * 13 % 23) as f64) * 0.00009;
        data[i] = wave * 0.75 + trend + noise;
    }
    data
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
fn compare_rows(
    cpu: &[f64],
    gpu: &[f64],
    combos: &[SuperSmootherParams],
    len: usize,
    first_valid: usize,
) {
    for (row_idx, combo) in combos.iter().enumerate() {
        let period = combo.period.unwrap();
        let warm = first_valid + period - 1;
        for col in 0..len {
            let idx = row_idx * len + col;
            let expected = cpu[idx];
            let actual = gpu[idx];
            if col < warm {
                assert!(
                    expected.is_nan(),
                    "CPU warmup NaN missing at row {row_idx} col {col}"
                );
                assert!(
                    actual.is_nan(),
                    "CUDA warmup mismatch at row {row_idx} col {col}"
                );
            } else {
                let diff = (expected - actual).abs();
                let tol = 9e-4 + expected.abs() * 2.5e-4;
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
fn supersmoother_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[supersmoother_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096;
    let data = make_test_series(len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap();
    let sweep = SuperSmootherBatchRange { period: (6, 36, 5) };

    let mut cpu_out = vec![f64::NAN; expand_grid_supersmoother(&sweep).len() * len];
    let combos_cpu =
        supersmoother_batch_inner_into(&data, &sweep, Kernel::Scalar, false, &mut cpu_out)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaSuperSmoother::new(0).expect("CudaSuperSmoother::new");
    let (dev, combos_gpu) = cuda
        .supersmoother_batch_dev(&data_f32, &sweep)
        .expect("supersmoother_cuda_batch_dev");

    assert_eq!(combos_cpu.len(), combos_gpu.len());
    for (cpu, gpu) in combos_cpu.iter().zip(&combos_gpu) {
        assert_eq!(cpu.period, gpu.period);
    }

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .expect("copy supersmoother cuda results");
    let gpu_flat_f64: Vec<f64> = gpu_flat.iter().map(|&v| v as f64).collect();

    compare_rows(&cpu_out, &gpu_flat_f64, &combos_cpu, len, first_valid);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn supersmoother_cuda_host_copy_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[supersmoother_cuda_host_copy_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096;
    let data = make_test_series(len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap();
    let sweep = SuperSmootherBatchRange { period: (5, 45, 5) };

    let mut cpu_out = vec![f64::NAN; expand_grid_supersmoother(&sweep).len() * len];
    let combos_cpu =
        supersmoother_batch_inner_into(&data, &sweep, Kernel::Scalar, false, &mut cpu_out)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaSuperSmoother::new(0).expect("CudaSuperSmoother::new");

    let mut gpu_flat = vec![0f32; cpu_out.len()];
    let (rows, cols, combos_gpu) = cuda
        .supersmoother_batch_into_host_f32(&data_f32, &sweep, &mut gpu_flat)
        .expect("supersmoother_cuda_batch_into_host_f32");

    assert_eq!(rows, combos_cpu.len());
    assert_eq!(cols, len);
    assert_eq!(combos_cpu.len(), combos_gpu.len());
    for (cpu, gpu) in combos_cpu.iter().zip(&combos_gpu) {
        assert_eq!(cpu.period, gpu.period);
    }

    let gpu_flat_f64: Vec<f64> = gpu_flat.iter().map(|&v| v as f64).collect();
    compare_rows(&cpu_out, &gpu_flat_f64, &combos_cpu, len, first_valid);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn supersmoother_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[supersmoother_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let rows = 2048usize;
    let cols = 6usize;
    let period = 18usize;

    let mut data_tm = vec![f64::NAN; rows * cols];
    for col in 0..cols {
        for row in (col * 3)..rows {
            let x = row as f64 + col as f64 * 0.37;
            let base = (x * 0.0027).sin() + (x * 0.0013).cos();
            let drift = 0.00031 * x;
            data_tm[row * cols + col] = base * 0.68 + drift;
        }
    }

    let mut cpu_tm = vec![f64::NAN; rows * cols];
    for col in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for row in 0..rows {
            series[row] = data_tm[row * cols + col];
        }
        let out = SuperSmootherBuilder::new()
            .period(period)
            .apply_slice(&series)?;
        for row in 0..rows {
            cpu_tm[row * cols + col] = out.values[row];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let params = SuperSmootherParams {
        period: Some(period),
    };

    let cuda = CudaSuperSmoother::new(0).expect("CudaSuperSmoother::new");
    let dev = cuda
        .supersmoother_multi_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("supersmoother_multi_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_tm)
        .expect("copy supersmoother many-series");
    let gpu_tm_f64: Vec<f64> = gpu_tm.iter().map(|&v| v as f64).collect();

    for col in 0..cols {
        let first_valid = (0..rows)
            .find(|&row| !data_tm[row * cols + col].is_nan())
            .unwrap();
        let warm = first_valid + period - 1;
        for row in 0..rows {
            let idx = row * cols + col;
            let expected = cpu_tm[idx];
            let actual = gpu_tm_f64[idx];
            if row < warm {
                assert!(
                    expected.is_nan(),
                    "CPU warmup NaN missing at row {row} col {col}"
                );
                assert!(
                    actual.is_nan(),
                    "CUDA warmup mismatch at row {row} col {col}"
                );
            } else {
                let diff = (expected - actual).abs();
                let tol = 1.1e-3 + expected.abs() * 3e-4;
                assert!(
                    diff <= tol,
                    "row {row} col {col} expected {expected} got {actual} diff {diff} tol {tol}"
                );
            }
        }
    }

    Ok(())
}
