use vector_ta::indicators::moving_averages::supersmoother_3_pole::{
    supersmoother_3_pole_batch_with_kernel, supersmoother_3_pole_with_kernel,
    SuperSmoother3PoleBatchRange, SuperSmoother3PoleInput, SuperSmoother3PoleParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaSupersmoother3Pole;

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
fn supersmoother_3_pole_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[supersmoother_3_pole_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 6..series_len {
        let t = i as f64;
        data[i] = (t * 0.002).sin() + 0.00025 * t;
    }

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let data_quant: Vec<f64> = data_f32.iter().map(|&v| v as f64).collect();

    let sweep = SuperSmoother3PoleBatchRange { period: (6, 64, 2) };

    let cpu = supersmoother_3_pole_batch_with_kernel(&data_quant, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaSupersmoother3Pole::new(0)?;
    let handle = cuda.supersmoother_3_pole_batch_dev(&data_f32, &sweep)?;

    assert_eq!(handle.rows, cpu.rows);
    assert_eq!(handle.cols, cpu.cols);

    let mut gpu_host = vec![0f32; handle.len()];
    handle.buf.copy_to(&mut gpu_host)?;

    let tol = 1e-5;
    for idx in 0..gpu_host.len() {
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
fn supersmoother_3_pole_cuda_many_series_one_param_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[supersmoother_3_pole_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let num_series = 6usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];

    for series in 0..num_series {
        for t in (series + 4)..series_len {
            let time = t as f64;
            let scale = 1.0 + 0.05 * series as f64;
            let base = (time * 0.0025 + series as f64 * 0.3).sin() * 0.7 * scale;
            let drift = 0.0004 * time + 0.015 * series as f64;
            data_tm[t * num_series + series] = base + drift;
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let data_tm_quant: Vec<f64> = data_tm_f32.iter().map(|&v| v as f64).collect();

    let params = SuperSmoother3PoleParams { period: Some(20) };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut series_data = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series_data[t] = data_tm_quant[t * num_series + series];
        }
        let input = SuperSmoother3PoleInput::from_slice(&series_data, params.clone());
        let output = supersmoother_3_pole_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = output.values[t];
        }
    }

    let cuda = CudaSupersmoother3Pole::new(0)?;
    let handle = cuda.supersmoother_3_pole_many_series_one_param_time_major_dev(
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
