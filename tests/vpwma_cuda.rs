#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaVpwma;
use my_project::indicators::moving_averages::vpwma::{
    expand_grid_vpwma, vpwma_batch_inner_into, VpwmaBatchRange, VpwmaBuilder, VpwmaParams,
};
use my_project::utilities::enums::Kernel;

fn make_test_series(len: usize) -> Vec<f64> {
    let mut data = vec![f64::NAN; len];
    for i in 9..len {
        let x = i as f64;
        let base = (x * 0.003214).sin() + (x * 0.001732).cos();
        let trend = 0.00057 * x;
        let noise = ((i * 17 % 13) as f64) * 0.0001;
        data[i] = base * 0.65 + trend + noise;
    }
    data
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
fn compare_rows(cpu: &[f64], gpu: &[f64], combos: &[VpwmaParams], len: usize, first_valid: usize) {
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
                    "CPU warmup should be NaN at row {} col {}",
                    row_idx,
                    col
                );
                assert!(
                    actual.is_nan(),
                    "CUDA warmup mismatch at row {} col {}",
                    row_idx,
                    col
                );
            } else {
                let diff = (expected - actual).abs();
                let tol = 5e-4 + expected.abs() * 1e-4;
                assert!(
                    diff <= tol,
                    "row {} col {} expected {} got {} diff {}",
                    row_idx,
                    col,
                    expected,
                    actual,
                    diff
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn vpwma_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vpwma_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 3276;
    let data = make_test_series(len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap();
    let sweep = VpwmaBatchRange {
        period: (5, 35, 5),
        power: (0.2, 0.8, 0.2),
    };

    let combos_cpu = expand_grid_vpwma(&sweep);
    let combo_count = combos_cpu.len();
    let mut cpu_out = vec![f64::NAN; combo_count * len];
    let combos_cpu = vpwma_batch_inner_into(&data, &sweep, Kernel::Scalar, false, &mut cpu_out)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaVpwma::new(0).expect("CudaVpwma::new");
    let (dev, combos_gpu) = cuda
        .vpwma_batch_dev(&data_f32, &sweep)
        .expect("vpwma_cuda_batch_dev");

    assert_eq!(combos_cpu.len(), combos_gpu.len());
    for (cpu, gpu) in combos_cpu.iter().zip(&combos_gpu) {
        assert_eq!(cpu.period, gpu.period);
        assert!((cpu.power.unwrap() - gpu.power.unwrap()).abs() < 1e-9);
    }

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .expect("copy cuda vpwma results");
    let gpu_flat_f64: Vec<f64> = gpu_flat.iter().map(|&v| v as f64).collect();

    compare_rows(&cpu_out, &gpu_flat_f64, &combos_cpu, len, first_valid);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn vpwma_cuda_host_copy_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vpwma_cuda_host_copy_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096;
    let data = make_test_series(len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap();
    let sweep = VpwmaBatchRange {
        period: (7, 31, 6),
        power: (0.3, 0.9, 0.3),
    };

    let combos_cpu = expand_grid_vpwma(&sweep);
    let mut cpu_out = vec![f64::NAN; combos_cpu.len() * len];
    let combos_cpu = vpwma_batch_inner_into(&data, &sweep, Kernel::Scalar, false, &mut cpu_out)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaVpwma::new(0).expect("CudaVpwma::new");

    let mut gpu_flat = vec![0f32; cpu_out.len()];
    let (rows, cols, combos_gpu) = cuda
        .vpwma_batch_into_host_f32(&data_f32, &sweep, &mut gpu_flat)
        .expect("vpwma_cuda_batch_into_host_f32");

    assert_eq!(rows, combos_cpu.len());
    assert_eq!(cols, len);
    assert_eq!(combos_cpu.len(), combos_gpu.len());
    for (cpu, gpu) in combos_cpu.iter().zip(&combos_gpu) {
        assert_eq!(cpu.period, gpu.period);
        assert!((cpu.power.unwrap() - gpu.power.unwrap()).abs() < 1e-9);
    }

    let gpu_flat_f64: Vec<f64> = gpu_flat.iter().map(|&v| v as f64).collect();
    compare_rows(&cpu_out, &gpu_flat_f64, &combos_cpu, len, first_valid);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn vpwma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vpwma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let rows = 1024usize;
    let cols = 4usize;
    let period = 14usize;
    let power = 0.55f64;

    let mut data_tm = vec![f64::NAN; rows * cols];
    for col in 0..cols {
        for row in col..rows {
            let x = row as f64 + col as f64 * 0.17;
            data_tm[row * cols + col] = (x * 0.0037).sin() + 0.00042 * x;
        }
    }

    let mut cpu_tm = vec![f64::NAN; rows * cols];
    for col in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for row in 0..rows {
            series[row] = data_tm[row * cols + col];
        }
        let out = VpwmaBuilder::new()
            .period(period)
            .power(power)
            .apply_slice(&series)?;
        for row in 0..rows {
            cpu_tm[row * cols + col] = out.values[row];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let params = VpwmaParams {
        period: Some(period),
        power: Some(power),
    };
    let cuda = CudaVpwma::new(0).expect("CudaVpwma::new");
    let dev = cuda
        .vpwma_multi_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("vpwma_multi_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_tm)
        .expect("copy cuda vpwma many-series");
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
                    "CPU warmup should be NaN at row {} col {}",
                    row,
                    col
                );
                assert!(
                    actual.is_nan(),
                    "CUDA warmup mismatch at row {} col {}",
                    row,
                    col
                );
            } else {
                let diff = (expected - actual).abs();
                let tol = 5e-4 + expected.abs() * 1e-4;
                assert!(
                    diff <= tol,
                    "row {} col {} expected {} got {} diff {}",
                    row,
                    col,
                    expected,
                    actual,
                    diff
                );
            }
        }
    }

    Ok(())
}
