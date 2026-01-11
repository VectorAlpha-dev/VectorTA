

#[cfg(feature = "cuda")]
use vector_ta::indicators::moving_averages::dma::{
    dma_batch_with_kernel, DmaBatchRange, DmaBuilder, DmaParams,
};
#[cfg(feature = "cuda")]
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaDma;

#[cfg(feature = "cuda")]
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
fn dma_cuda_batch_wma_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dma_cuda_batch_wma_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 5..series_len {
        let x = i as f64;
        data[i] = (x * 0.002).sin() + 0.0002 * x;
    }

    let sweep = DmaBatchRange {
        hull_length: (7, 21, 2),
        ema_length: (16, 28, 4),
        ema_gain_limit: (10, 40, 10),
        hull_ma_type: "WMA".to_string(),
    };

    let cpu = match dma_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch) {
        Ok(v) => v,
        Err(e) => return Err(Box::new(e)),
    };

    let cuda = CudaDma::new(0).expect("CudaDma::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .dma_batch_dev(&data_f32, &sweep)
        .expect("cuda dma_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy dma gpu results");

    let tol = 3e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let a = cpu.values[idx];
        let b = gpu_host[idx] as f64;
        assert!(
            approx_eq(a, b, tol),
            "WMA mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn dma_cuda_batch_ema_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dma_cuda_batch_ema_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 3072usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 4..series_len {
        let x = i as f64;
        data[i] = (x * 0.0015).cos() + 0.00015 * x;
    }

    let sweep = DmaBatchRange {
        hull_length: (12, 24, 4),
        ema_length: (18, 30, 6),
        ema_gain_limit: (5, 25, 5),
        hull_ma_type: "EMA".to_string(),
    };

    let cpu = match dma_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch) {
        Ok(v) => v,
        Err(e) => return Err(Box::new(e)),
    };

    let cuda = CudaDma::new(0).expect("CudaDma::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .dma_batch_dev(&data_f32, &sweep)
        .expect("cuda dma_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy dma gpu results");

    let tol = 3e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let a = cpu.values[idx];
        let b = gpu_host[idx] as f64;
        assert!(
            approx_eq(a, b, tol),
            "EMA mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn dma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 4usize;
    let series_len = 1024usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in j..series_len {
            let x = (t as f64) + (j as f64) * 0.25;
            data_tm[t * num_series + j] = (x * 0.0025).sin() + 0.0002 * x;
        }
    }

    let hull_length = 21usize;
    let ema_length = 34usize;
    let ema_gain_limit = 30usize;
    let hull_ma_type = "WMA".to_string();

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        let mut series = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series[t] = data_tm[t * num_series + j];
        }
        let out = DmaBuilder::new()
            .hull_length(hull_length)
            .ema_length(ema_length)
            .ema_gain_limit(ema_gain_limit)
            .hull_ma_type(hull_ma_type.clone())
            .apply_slice(&series)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + j] = out.values[t];
        }
    }

    let cuda = CudaDma::new(0).expect("CudaDma::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let params = DmaParams {
        hull_length: Some(hull_length),
        ema_length: Some(ema_length),
        ema_gain_limit: Some(ema_gain_limit),
        hull_ma_type: Some(hull_ma_type.clone()),
    };
    let gpu_handle = cuda
        .dma_many_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)
        .expect("cuda dma_many_series_one_param_time_major_dev");

    assert_eq!(gpu_handle.rows, series_len);
    assert_eq!(gpu_handle.cols, num_series);

    let mut gpu_tm = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy dma many-series result to host");

    let tol = 1e-3;
    for i in 0..(num_series * series_len) {
        let a = cpu_tm[i];
        let b = gpu_tm[i] as f64;
        assert!(
            approx_eq(a, b, tol),
            "DMA many-series mismatch at {}: cpu={} gpu={}",
            i,
            a,
            b
        );
    }

    Ok(())
}
