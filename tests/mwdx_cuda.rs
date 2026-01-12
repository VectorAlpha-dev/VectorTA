use vector_ta::indicators::moving_averages::mwdx::{
    mwdx_batch_with_kernel, mwdx_with_kernel, MwdxBatchRange, MwdxInput, MwdxParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaMwdx;

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
fn mwdx_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mwdx_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 5..series_len {
        let x = i as f64;
        data[i] = (x * 0.00137).sin() + 0.00043 * x.cos();
    }

    let sweep = MwdxBatchRange {
        factor: (0.05, 0.95, 0.1),
    };

    let cpu = mwdx_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaMwdx::new(0).expect("CudaMwdx::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .mwdx_batch_dev(&data_f32, &sweep)
        .expect("cuda mwdx_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda mwdx batch result to host");

    let tol = 1e-5;
    for idx in 0..gpu_host.len() {
        let cpu_val = cpu.values[idx];
        let gpu_val = gpu_host[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
            "batch mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn mwdx_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mwdx_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 6usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        for t in series..series_len {
            let x = t as f64 + (series as f64) * 0.31;
            data_tm[t * num_series + series] = (x * 0.0023).cos() + 0.00057 * x.sin();
        }
    }

    let factor = 0.27;
    let params = MwdxParams {
        factor: Some(factor),
    };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut series_slice = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series_slice[t] = data_tm[t * num_series + series];
        }
        let input = MwdxInput::from_slice(&series_slice, params.clone());
        let cpu_vals = mwdx_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = cpu_vals.values[t];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaMwdx::new(0).expect("CudaMwdx::new");
    let handle = cuda
        .mwdx_many_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)
        .expect("cuda mwdx many-series dev");

    assert_eq!(handle.rows, series_len);
    assert_eq!(handle.cols, num_series);

    let mut gpu_tm = vec![0f32; handle.len()];
    handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy cuda mwdx many-series result to host");

    let tol = 1e-5;
    for idx in 0..gpu_tm.len() {
        let cpu_val = cpu_tm[idx];
        let gpu_val = gpu_tm[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
            "many-series mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    Ok(())
}
