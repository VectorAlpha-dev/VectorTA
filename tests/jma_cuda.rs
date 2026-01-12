use vector_ta::indicators::moving_averages::jma::{
    jma_batch_with_kernel, JmaBatchRange, JmaBuilder, JmaParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaJma;

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
fn jma_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[jma_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 2048usize;
    let first_valid = 6usize;
    let mut data = vec![f64::NAN; series_len];
    for i in first_valid..series_len {
        let x = i as f64;
        data[i] = (x * 0.0021).sin() + 0.0004 * x;
    }

    let sweep = JmaBatchRange {
        period: (8, 16, 4),
        phase: (-40.0, 40.0, 40.0),
        power: (1, 3, 1),
    };

    let cpu = match jma_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch) {
        Ok(out) => out,
        Err(e) => return Err(Box::new(e)),
    };

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaJma::new(0).expect("CudaJma::new");
    let gpu_handle = cuda
        .jma_batch_dev(&data_f32, &sweep)
        .expect("cuda jma_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_flat = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_flat)
        .expect("copy cuda jma batch result");

    let tol = 2e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
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
fn jma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[jma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 4usize;
    let series_len = 1024usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];

    for series in 0..num_series {
        let first_valid = 4 + series;
        for t in first_valid..series_len {
            let x = (t as f64) + (series as f64) * 0.37;
            data_tm[t * num_series + series] = (x * 0.0017).cos() + 0.0003 * x;
        }
    }

    let params = JmaParams {
        period: Some(12),
        phase: Some(30.0),
        power: Some(2),
    };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut series_prices = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series_prices[t] = data_tm[t * num_series + series];
        }
        let out = JmaBuilder::new()
            .period(params.period.unwrap())
            .phase(params.phase.unwrap())
            .power(params.power.unwrap())
            .kernel(Kernel::Scalar)
            .apply_slice(&series_prices)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = out.values[t];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaJma::new(0).expect("CudaJma::new");
    let gpu_handle = cuda
        .jma_many_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)
        .expect("cuda jma_many_series_one_param_time_major_dev");

    assert_eq!(gpu_handle.rows, series_len);
    assert_eq!(gpu_handle.cols, num_series);

    let mut gpu_tm = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy cuda jma many-series result");

    let tol = 2e-4;
    for idx in 0..(num_series * series_len) {
        let a = cpu_tm[idx];
        let b = gpu_tm[idx] as f64;
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
