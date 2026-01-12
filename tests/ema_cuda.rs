use vector_ta::indicators::moving_averages::ema::{
    ema_batch_with_kernel, ema_with_kernel, EmaBatchRange, EmaInput, EmaParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaEma;

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
fn ema_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ema_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 5..series_len {
        let x = i as f64;
        data[i] = (x * 0.00173).sin() + 0.00029 * x.cos();
    }

    let sweep = EmaBatchRange { period: (5, 45, 5) };

    let cpu = ema_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaEma::new(0).expect("CudaEma::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let handle = cuda
        .ema_batch_dev(&data_f32, &sweep)
        .expect("ema cuda batch dev");

    assert_eq!(cpu.rows, handle.rows);
    assert_eq!(cpu.cols, handle.cols);

    let mut gpu_flat = vec![0f32; handle.len()];
    handle
        .buf
        .copy_to(&mut gpu_flat)
        .expect("copy ema cuda batch result to host");

    let tol = 5e-5;
    for idx in 0..cpu.values.len() {
        let cpu_val = cpu.values[idx];
        let gpu_val = gpu_flat[idx] as f64;
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
fn ema_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ema_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 6usize;
    let series_len = 2048usize;
    let period = 21usize;

    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        for t in series..series_len {
            let base = (t as f64) + (series as f64) * 0.37;
            data_tm[t * num_series + series] = (base * 0.00191).cos() + 0.00041 * base;
        }
    }

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    let params = EmaParams {
        period: Some(period),
    };
    for series in 0..num_series {
        let mut single = vec![f64::NAN; series_len];
        for t in 0..series_len {
            single[t] = data_tm[t * num_series + series];
        }
        let input = EmaInput::from_slice(&single, params.clone());
        let cpu_vals = ema_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = cpu_vals.values[t];
        }
    }

    let cuda = CudaEma::new(0).expect("CudaEma::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let handle = cuda
        .ema_many_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)
        .expect("ema cuda many-series dev");

    assert_eq!(handle.rows, series_len);
    assert_eq!(handle.cols, num_series);

    let mut gpu_tm = vec![0f32; handle.len()];
    handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy ema cuda many-series result to host");

    let tol = 5e-5;
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
