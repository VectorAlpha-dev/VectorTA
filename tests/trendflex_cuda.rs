#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaTrendflex;
use vector_ta::indicators::moving_averages::trendflex::{
    expand_grid_trendflex, trendflex_batch_inner_into, TrendFlexBatchRange, TrendFlexBuilder,
    TrendFlexParams,
};
use vector_ta::utilities::enums::Kernel;

fn make_test_series(len: usize) -> Vec<f64> {
    let mut data = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        let base = (x * 0.00419).sin() + (x * 0.00237).cos();
        let trend = 0.00071 * x;
        let noise = ((i * 11 % 17) as f64) * 0.00008;
        data[i] = base * 0.6 + trend + noise;
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
    combos: &[TrendFlexParams],
    len: usize,
    first_valid: usize,
) {
    for (row_idx, combo) in combos.iter().enumerate() {
        let period = combo.period.unwrap();
        let warm = first_valid + period;
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
                let tol = 8e-4 + expected.abs() * 2e-4;
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
fn trendflex_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[trendflex_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096;
    let data = make_test_series(len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap();
    let sweep = TrendFlexBatchRange { period: (5, 45, 5) };

    let mut cpu_out = vec![f64::NAN; expand_grid_trendflex(&sweep).len() * len];
    let combos_cpu =
        trendflex_batch_inner_into(&data, &sweep, Kernel::Scalar, false, &mut cpu_out)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaTrendflex::new(0).expect("CudaTrendflex::new");
    let (dev, combos_gpu) = cuda
        .trendflex_batch_dev(&data_f32, &sweep)
        .expect("trendflex_cuda_batch_dev");

    assert_eq!(combos_cpu.len(), combos_gpu.len());
    for (cpu, gpu) in combos_cpu.iter().zip(&combos_gpu) {
        assert_eq!(cpu.period, gpu.period);
    }

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .expect("copy trendflex cuda results");
    let gpu_flat_f64: Vec<f64> = gpu_flat.iter().map(|&v| v as f64).collect();

    compare_rows(&cpu_out, &gpu_flat_f64, &combos_cpu, len, first_valid);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn trendflex_cuda_host_copy_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[trendflex_cuda_host_copy_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096;
    let data = make_test_series(len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap();
    let sweep = TrendFlexBatchRange { period: (7, 37, 5) };

    let mut cpu_out = vec![f64::NAN; expand_grid_trendflex(&sweep).len() * len];
    let combos_cpu =
        trendflex_batch_inner_into(&data, &sweep, Kernel::Scalar, false, &mut cpu_out)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaTrendflex::new(0).expect("CudaTrendflex::new");

    let mut gpu_flat = vec![0f32; cpu_out.len()];
    let (rows, cols, combos_gpu) = cuda
        .trendflex_batch_into_host_f32(&data_f32, &sweep, &mut gpu_flat)
        .expect("trendflex_cuda_batch_into_host_f32");

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
fn trendflex_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[trendflex_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let rows = 1024usize;
    let cols = 5usize;
    let period = 18usize;

    let mut data_tm = vec![f64::NAN; rows * cols];
    for col in 0..cols {
        for row in col..rows {
            let x = row as f64 + col as f64 * 0.23;
            let base = (x * 0.0031).sin() + (x * 0.0019).cos();
            let drift = 0.00054 * x;
            data_tm[row * cols + col] = base * 0.7 + drift;
        }
    }

    let mut cpu_tm = vec![f64::NAN; rows * cols];
    for col in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for row in 0..rows {
            series[row] = data_tm[row * cols + col];
        }
        let out = TrendFlexBuilder::new()
            .period(period)
            .apply_slice(&series)?;
        for row in 0..rows {
            cpu_tm[row * cols + col] = out.values[row];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let params = TrendFlexParams {
        period: Some(period),
    };

    let cuda = CudaTrendflex::new(0).expect("CudaTrendflex::new");
    let dev = cuda
        .trendflex_multi_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("trendflex_multi_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_tm)
        .expect("copy trendflex many-series");
    let gpu_tm_f64: Vec<f64> = gpu_tm.iter().map(|&v| v as f64).collect();

    for col in 0..cols {
        let first_valid = (0..rows)
            .find(|&row| !data_tm[row * cols + col].is_nan())
            .unwrap();
        let warm = first_valid + period;
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
                let tol = 8e-4 + expected.abs() * 2e-4;
                assert!(
                    diff <= tol,
                    "row {row} col {col} expected {expected} got {actual} diff {diff}"
                );
            }
        }
    }

    Ok(())
}
