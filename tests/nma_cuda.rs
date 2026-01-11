

use vector_ta::indicators::moving_averages::nma::{
    nma_batch_with_kernel, NmaBatchRange, NmaBuilder, NmaParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaNma;

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
fn nma_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[nma_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 12..series_len {
        let x = i as f64 * 0.0013;
        data[i] = 1.5 + 0.45 * x.sin() + 0.2 * x.cos();
    }

    let sweep = NmaBatchRange { period: (5, 35, 5) };

    let cpu = nma_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    let cuda = CudaNma::new(0).expect("CudaNma::new failed");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let (dev, combos) = cuda
        .nma_batch_dev(&data_f32, &sweep)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    assert_eq!(dev.rows, cpu.rows);
    assert_eq!(dev.cols, cpu.cols);
    assert_eq!(combos.len(), cpu.combos.len());

    for (lhs, rhs) in combos.iter().zip(cpu.combos.iter()) {
        assert_eq!(lhs.period, rhs.period);
    }

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    let tol = 1e-4;
    for idx in 0..gpu_flat.len() {
        let a = cpu.values[idx];
        let b = gpu_flat[idx] as f64;
        assert!(
            approx_eq(a, b, tol),
            "Mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn nma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[nma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 4usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in (j + 8)..series_len {
            let base = t as f64 * 0.0021 + j as f64 * 0.17;
            data_tm[t * num_series + j] = 1.4 + 0.6 * base.sin();
        }
    }

    let period = 18usize;

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut slice = vec![f64::NAN; series_len];
        for row in 0..series_len {
            slice[row] = data_tm[row * num_series + series];
        }
        let out = NmaBuilder::new()
            .period(period)
            .apply_slice(&slice)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        for row in 0..series_len {
            cpu_tm[row * num_series + series] = out.values[row];
        }
    }

    let cuda = CudaNma::new(0).expect("CudaNma::new failed");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let params = NmaParams {
        period: Some(period),
    };

    let dev = cuda
        .nma_multi_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    assert_eq!(dev.rows, series_len);
    assert_eq!(dev.cols, num_series);

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    let tol = 1e-4;
    for idx in 0..gpu_flat.len() {
        let a = cpu_tm[idx];
        let b = gpu_flat[idx] as f64;
        assert!(
            approx_eq(a, b, tol),
            "Mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}
