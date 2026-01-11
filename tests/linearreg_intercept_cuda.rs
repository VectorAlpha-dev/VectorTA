#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaLinregIntercept;
use vector_ta::indicators::linearreg_intercept::{
    linearreg_intercept_batch_slice, LinearRegInterceptBatchOutput, LinearRegInterceptBatchRange,
    LinearRegInterceptBuilder, LinearRegInterceptParams,
};
use vector_ta::utilities::enums::Kernel;

fn make_test_series(len: usize) -> Vec<f64> {
    let mut data = vec![f64::NAN; len];
    for i in 16..len {
        let t = i as f64;
        let trend = 0.0012 * t;
        let curve = (t * 0.013).sin() * 0.55 + (t * 0.006).cos() * 0.45;
        let noise = ((i * 13 % 37) as f64) * 0.00045;
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
fn compare_rows(
    cpu: &[f64],
    gpu: &[f64],
    combos: &[LinearRegInterceptParams],
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
                let tol = 2.5e-3 + expected.abs() * 7.5e-4;
                assert!(diff <= tol, "row {row_idx} col {col} expected {expected} got {actual} diff {diff} tol {tol}");
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn linearreg_intercept_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[linearreg_intercept_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096;
    let data = make_test_series(len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap();
    let sweep = LinearRegInterceptBatchRange { period: (8, 36, 4) };

    let cpu_out_struct: LinearRegInterceptBatchOutput =
        linearreg_intercept_batch_slice(&data, &sweep, Kernel::Scalar)?;
    let combos_cpu = cpu_out_struct.combos.clone();
    let cpu_out = cpu_out_struct.values;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaLinregIntercept::new(0).expect("CudaLinregIntercept::new");
    let (dev, combos_gpu) = cuda
        .linearreg_intercept_batch_dev(&data_f32, &sweep)
        .expect("linearreg_intercept_batch_dev");

    for (cpu, gpu) in combos_cpu.iter().zip(&combos_gpu) {
        assert_eq!(cpu.period, gpu.period);
    }

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .expect("copy lri cuda results");
    let gpu_flat_f64: Vec<f64> = gpu_flat.iter().map(|&v| v as f64).collect();

    compare_rows(&cpu_out, &gpu_flat_f64, &combos_cpu, len, first_valid);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn linearreg_intercept_cuda_host_copy_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[linearreg_intercept_cuda_host_copy_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 3072;
    let data = make_test_series(len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap();
    let sweep = LinearRegInterceptBatchRange { period: (6, 30, 6) };
    let cpu_out_struct: LinearRegInterceptBatchOutput =
        linearreg_intercept_batch_slice(&data, &sweep, Kernel::Scalar)?;
    let combos_cpu = cpu_out_struct.combos.clone();
    let cpu_out = cpu_out_struct.values;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaLinregIntercept::new(0).expect("CudaLinregIntercept::new");
    let mut gpu_flat = vec![0f32; cpu_out.len()];
    let (rows, cols, combos_gpu) = cuda
        .linearreg_intercept_batch_into_host_f32(&data_f32, &sweep, &mut gpu_flat)
        .expect("linearreg_intercept_batch_into_host_f32");

    assert_eq!(rows, combos_cpu.len());
    assert_eq!(cols, len);
    for (cpu, gpu) in combos_cpu.iter().zip(&combos_gpu) {
        assert_eq!(cpu.period, gpu.period);
    }

    let gpu_flat_f64: Vec<f64> = gpu_flat.iter().map(|&v| v as f64).collect();
    compare_rows(&cpu_out, &gpu_flat_f64, &combos_cpu, len, first_valid);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn linearreg_intercept_cuda_many_series_one_param_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[linearreg_intercept_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let rows = 1536usize;
    let cols = 5usize;
    let period = 18usize;

    let mut data_tm = vec![f64::NAN; rows * cols];
    for col in 0..cols {
        for row in (col * 3)..rows {
            let t = row as f64 + (col as f64) * 0.77;
            let drift = 0.00075 * t;
            let wave = (t * 0.011).sin() * 0.50 + (t * 0.007).cos() * 0.40;
            data_tm[row * cols + col] = drift + wave;
        }
    }

    let mut cpu_tm = vec![f64::NAN; rows * cols];
    for col in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for row in 0..rows {
            series[row] = data_tm[row * cols + col];
        }
        let out = LinearRegInterceptBuilder::new()
            .period(period)
            .apply_slice(&series)?;
        for row in 0..rows {
            cpu_tm[row * cols + col] = out.values[row];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let params = LinearRegInterceptParams {
        period: Some(period),
    };

    let cuda = CudaLinregIntercept::new(0).expect("CudaLinregIntercept::new");
    let dev = cuda
        .linearreg_intercept_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("linearreg_intercept_many_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_tm).expect("copy lri many-series");
    let gpu_tm_f64: Vec<f64> = gpu_tm.iter().map(|&v| v as f64).collect();

    for idx in 0..rows * cols {
        let expected = cpu_tm[idx];
        let actual = gpu_tm_f64[idx];
        if expected.is_nan() {
            assert!(actual.is_nan(), "CUDA warmup mismatch at idx {idx}");
        } else {
            let diff = (expected - actual).abs();
            let tol = 2.5e-3 + expected.abs() * 7.5e-4;
            assert!(
                diff <= tol,
                "idx {idx} expected {expected} got {actual} diff {diff} tol {tol}"
            );
        }
    }
    Ok(())
}
