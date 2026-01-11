

use vector_ta::indicators::moving_averages::highpass::{
    highpass_batch_with_kernel, highpass_with_kernel, HighPassBatchRange, HighPassInput,
    HighPassParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaHighpass;

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
fn highpass_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[highpass_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![0.0f64; series_len];
    for i in 0..series_len {
        let t = i as f64;
        data[i] = (t * 0.0023).sin() + 0.00027 * t;
    }

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let data_quant: Vec<f64> = data_f32.iter().map(|&v| v as f64).collect();

    let sweep = HighPassBatchRange { period: (8, 96, 4) };

    let cpu = highpass_batch_with_kernel(&data_quant, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaHighpass::new(0)?;
    let handle = cuda.highpass_batch_dev(&data_f32, &sweep)?;

    assert_eq!(handle.rows, cpu.rows);
    assert_eq!(handle.cols, cpu.cols);

    let mut gpu_host = vec![0f32; handle.len()];
    handle.buf.copy_to(&mut gpu_host)?;

    let tol = 1e-5;
    for idx in 0..gpu_host.len() {
        let cpu_v = cpu.values[idx];
        let gpu_v = gpu_host[idx] as f64;
        assert!(
            approx_eq(cpu_v, gpu_v, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_v,
            gpu_v
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn highpass_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[highpass_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 5usize;
    let series_len = 2048usize;
    let mut data_tm = vec![0.0f64; num_series * series_len];
    for series in 0..num_series {
        for t in 0..series_len {
            let time = t as f64;
            let phase = series as f64 * 0.35;
            data_tm[t * num_series + series] = (time * 0.0029 + phase).sin() + 0.00041 * time;
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let data_tm_quant: Vec<f64> = data_tm_f32.iter().map(|&v| v as f64).collect();

    let params = HighPassParams { period: Some(48) };

    let mut cpu_tm = vec![0.0f64; num_series * series_len];
    for series in 0..num_series {
        let mut slice = vec![0.0f64; series_len];
        for t in 0..series_len {
            slice[t] = data_tm_quant[t * num_series + series];
        }
        let input = HighPassInput::from_slice(&slice, params.clone());
        let out = highpass_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = out.values[t];
        }
    }

    let cuda = CudaHighpass::new(0)?;
    let handle = cuda.highpass_many_series_one_param_time_major_dev(
        &data_tm_f32,
        num_series,
        series_len,
        &params,
    )?;

    assert_eq!(handle.rows, series_len);
    assert_eq!(handle.cols, num_series);

    let mut gpu_tm = vec![0f32; handle.len()];
    handle.buf.copy_to(&mut gpu_tm)?;

    let tol = 1e-5;
    for idx in 0..gpu_tm.len() {
        let cpu_v = cpu_tm[idx];
        let gpu_v = gpu_tm[idx] as f64;
        assert!(
            approx_eq(cpu_v, gpu_v, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_v,
            gpu_v
        );
    }

    Ok(())
}
