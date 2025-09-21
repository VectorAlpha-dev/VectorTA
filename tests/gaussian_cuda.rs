use my_project::indicators::moving_averages::gaussian::{
    gaussian_batch_with_kernel, GaussianBatchRange, GaussianBuilder, GaussianParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaGaussian;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
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
fn gaussian_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[gaussian_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let first_valid = 12usize;
    let mut data = vec![f64::NAN; series_len];
    for i in first_valid..series_len {
        let x = i as f64;
        data[i] = (x * 0.0031).sin() + 0.0005 * x;
    }

    let sweep = GaussianBatchRange {
        period: (8, 40, 8),
        poles: (1, 4, 1),
    };

    let cpu = gaussian_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaGaussian::new(0).expect("CudaGaussian::new");
    let gpu = cuda
        .gaussian_batch_dev(&data_f32, &sweep)
        .expect("cuda gaussian_batch_dev");

    assert_eq!(gpu.rows, cpu.rows);
    assert_eq!(gpu.cols, cpu.cols);

    let mut gpu_flat = vec![0f32; gpu.len()];
    gpu.buf
        .copy_to(&mut gpu_flat)
        .expect("copy gaussian batch result");

    let tol = 3e-4f64;
    for idx in 0..gpu_flat.len() {
        let a = cpu.values[idx];
        let b = gpu_flat[idx] as f64;
        assert!(
            approx_eq(a, b, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn gaussian_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[gaussian_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 5usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];

    for series in 0..num_series {
        let first_valid = 10 + series;
        for t in first_valid..series_len {
            let idx = t * num_series + series;
            let x = (t as f64) + (series as f64) * 0.27;
            data_tm[idx] = (x * 0.0025).cos() + 0.00035 * x;
        }
    }

    let period = 18usize;
    let poles = 3usize;
    let params = GaussianParams {
        period: Some(period),
        poles: Some(poles),
    };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut series_prices = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series_prices[t] = data_tm[t * num_series + series];
        }
        let output = GaussianBuilder::new()
            .period(period)
            .poles(poles)
            .kernel(Kernel::Scalar)
            .apply_slice(&series_prices)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = output.values[t];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaGaussian::new(0).expect("CudaGaussian::new");
    let gpu = cuda
        .gaussian_many_series_one_param_time_major_dev(
            &data_tm_f32,
            num_series,
            series_len,
            &params,
        )
        .expect("cuda gaussian_many_series_one_param_time_major_dev");

    assert_eq!(gpu.rows, series_len);
    assert_eq!(gpu.cols, num_series);

    let mut gpu_flat = vec![0f32; gpu.len()];
    gpu.buf
        .copy_to(&mut gpu_flat)
        .expect("copy gaussian many-series result");

    let tol = 3e-4f64;
    for idx in 0..gpu_flat.len() {
        let a = cpu_tm[idx];
        let b = gpu_flat[idx] as f64;
        assert!(
            approx_eq(a, b, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}
