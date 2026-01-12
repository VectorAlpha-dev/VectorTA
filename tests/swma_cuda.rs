use vector_ta::indicators::moving_averages::swma::{
    swma_batch_with_kernel, SwmaBatchRange, SwmaBuilder, SwmaParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaSwma;

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
fn swma_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[swma_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 10..series_len {
        let x = i as f64;
        data[i] = (x * 0.0013).sin() + 0.0005 * x;
    }

    let sweep = SwmaBatchRange {
        period: (5, 160, 5),
    };

    let cpu = swma_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaSwma::new(0).expect("CudaSwma::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu = cuda
        .swma_batch_dev(&data_f32, &sweep)
        .expect("cuda swma_batch_dev");

    assert_eq!(cpu.rows, gpu.rows);
    assert_eq!(cpu.cols, gpu.cols);

    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda swma batch result to host");

    let tol = 2e-4f64;
    for idx in 0..(cpu.rows * cpu.cols) {
        let a = cpu.values[idx];
        let b = gpu_host[idx] as f64;
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
fn swma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[swma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 4usize;
    let series_len = 1024usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];

    for j in 0..num_series {
        for t in j..series_len {
            let idx = t * num_series + j;
            let x = (t as f64) + (j as f64) * 0.15;
            data_tm[idx] = (x * 0.0023).cos() + 0.0007 * x;
        }
    }

    let period = 24;
    let params = SwmaParams {
        period: Some(period),
    };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        let mut series = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series[t] = data_tm[t * num_series + j];
        }
        let out = SwmaBuilder::new().period(period).apply_slice(&series)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + j] = out.values[t];
        }
    }

    let cuda = CudaSwma::new(0).expect("CudaSwma::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let gpu = cuda
        .swma_multi_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)
        .expect("cuda swma_multi_series_one_param_time_major_dev");

    assert_eq!(gpu.rows, series_len);
    assert_eq!(gpu.cols, num_series);

    let mut gpu_tm = vec![0f32; gpu.len()];
    gpu.buf
        .copy_to(&mut gpu_tm)
        .expect("copy cuda swma many-series result");

    let tol = 2e-4f64;
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
