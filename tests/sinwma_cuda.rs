use vector_ta::indicators::moving_averages::sinwma::{
    sinwma_batch_with_kernel, sinwma_with_kernel, SinWmaBatchRange, SinWmaInput, SinWmaParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaSinwma;

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
fn sinwma_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[sinwma_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 6..series_len {
        let t = i as f64;
        data[i] = (t * 0.0021).sin() + 0.00023 * t;
    }

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let data_quant: Vec<f64> = data_f32.iter().map(|&v| v as f64).collect();

    let sweep = SinWmaBatchRange { period: (6, 96, 3) };

    let cpu = sinwma_batch_with_kernel(&data_quant, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaSinwma::new(0)?;
    let handle = cuda.sinwma_batch_dev(&data_f32, &sweep)?;

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
fn sinwma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[sinwma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 5usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];

    for series in 0..num_series {
        for t in (series + 3)..series_len {
            let time = t as f64;
            let base = (time * 0.0028 + series as f64 * 0.17).sin();
            let drift = 0.00031 * time + 0.015 * series as f64;
            data_tm[t * num_series + series] = base + drift;
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let data_tm_quant: Vec<f64> = data_tm_f32.iter().map(|&v| v as f64).collect();

    let params = SinWmaParams { period: Some(28) };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut slice = vec![f64::NAN; series_len];
        for t in 0..series_len {
            slice[t] = data_tm_quant[t * num_series + series];
        }
        let input = SinWmaInput::from_slice(&slice, params.clone());
        let out = sinwma_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = out.values[t];
        }
    }

    let cuda = CudaSinwma::new(0)?;
    let handle = cuda.sinwma_many_series_one_param_time_major_dev(
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
