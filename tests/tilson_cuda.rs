use vector_ta::indicators::moving_averages::tilson::{
    tilson_batch_with_kernel, tilson_with_kernel, TilsonBatchRange, TilsonInput, TilsonParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaTilson;

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
fn tilson_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[tilson_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 7..series_len {
        let x = i as f64;
        data[i] = (x * 0.0017).sin() + 0.0009 * x;
    }

    let sweep = TilsonBatchRange {
        period: (5, 28, 3),
        volume_factor: (0.0, 0.7, 0.2),
    };

    let cpu = tilson_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaTilson::new(0).expect("CudaTilson::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .tilson_batch_dev(&data_f32, &sweep)
        .expect("cuda tilson_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda tilson batch to host");

    let tol = 2e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let cpu_val = cpu.values[idx];
        let gpu_val = gpu_host[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn tilson_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[tilson_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 6usize;
    let series_len = 1536usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        for t in series..series_len {
            let base = (t as f64) * 0.003 + (series as f64) * 0.05;
            data_tm[t * num_series + series] = (base).cos() + 0.0004 * (t as f64);
        }
    }

    let params = TilsonParams {
        period: Some(12),
        volume_factor: Some(0.4),
    };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut series_slice = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series_slice[t] = data_tm[t * num_series + series];
        }
        let input = TilsonInput::from_slice(&series_slice, params.clone());
        let cpu_vals = tilson_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = cpu_vals.values[t];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaTilson::new(0).expect("CudaTilson::new");
    let handle = cuda
        .tilson_many_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)
        .expect("cuda tilson_many_series_one_param_time_major_dev");

    assert_eq!(handle.rows, series_len);
    assert_eq!(handle.cols, num_series);

    let mut gpu_tm = vec![0f32; handle.len()];
    handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy cuda tilson many-series result to host");

    let tol = 2e-4;
    for idx in 0..gpu_tm.len() {
        let cpu_val = cpu_tm[idx];
        let gpu_val = gpu_tm[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    Ok(())
}
