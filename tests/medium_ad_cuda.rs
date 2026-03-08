use vector_ta::indicators::medium_ad::{
    medium_ad_batch_with_kernel, MediumAdBatchRange, MediumAdBuilder,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{CudaMediumAd, CudaRuntime};

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
fn medium_ad_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[medium_ad_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 6..len {
        let x = i as f64;
        let base = (x * 0.00039).sin() + (x * 0.00023).cos();
        data[i] = base + 0.001 * (i % 9) as f64;
        if i % 257 == 0 {
            data[i] = f64::NAN;
        }
    }

    let sweep = MediumAdBatchRange { period: (5, 25, 5) };
    let cpu = medium_ad_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();

    let cuda = CudaMediumAd::new(0).expect("CudaMediumAd::new");
    let dev_arr = cuda
        .medium_ad_batch_dev(&data_f32, &sweep)
        .expect("medium_ad_cuda_batch_dev");

    assert_eq!(dev_arr.rows, cpu.rows);
    assert_eq!(dev_arr.cols, cpu.cols);

    let mut gpu = vec![0f32; dev_arr.len()];
    dev_arr
        .buf
        .copy_to(&mut gpu)
        .expect("copy medium_ad cuda results");

    let tol = 5e-4;
    for (idx, (&cpu_val, &gpu_val)) in cpu.values.iter().zip(gpu.iter()).enumerate() {
        assert!(
            approx_eq(cpu_val, gpu_val as f64, tol),
            "MEDIUM_AD mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn medium_ad_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[medium_ad_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 5usize;
    let rows = 1024usize;
    let mut data_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.17;
            data_tm[t * cols + s] = (x * 0.0021).sin() + 0.0003 * x.cos();
        }
    }

    let period = 9usize;
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = data_tm[t * cols + s];
        }
        let out = MediumAdBuilder::new().period(period).apply_slice(&series)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let cuda = CudaMediumAd::new(0).expect("CudaMediumAd::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let handle = cuda
        .medium_ad_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, period)
        .expect("medium_ad_many_series_one_param_time_major_dev");

    assert_eq!(handle.rows, rows);
    assert_eq!(handle.cols, cols);

    let mut gpu_tm = vec![0f32; handle.len()];
    handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy medium_ad many-series results");

    let tol = 1e-3;
    for i in 0..(cols * rows) {
        let a = cpu_tm[i];
        let b = gpu_tm[i] as f64;
        assert!(
            approx_eq(a, b, tol),
            "MEDIUM_AD many-series mismatch at {}: cpu={} gpu={}",
            i,
            a,
            b
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn medium_ad_cuda_device_inputs_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[medium_ad_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        data[i] = (x * 0.00107).sin() + 0.00011 * x.cos();
    }
    let sweep = MediumAdBatchRange { period: (7, 21, 7) };

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let first_valid = data_f32
        .iter()
        .position(|v| v.is_finite())
        .expect("first_valid");
    let runtime = CudaRuntime::new(0).expect("runtime");
    let d_prices = runtime.upload_f32(&data_f32).expect("upload");

    let cuda = CudaMediumAd::new(0).expect("CudaMediumAd::new");
    let legacy_dev = cuda.medium_ad_batch_dev(&data_f32, &sweep)?;
    let device_dev = cuda.medium_ad_batch_dev_from_device_prices(
        d_prices.buffer(),
        data_f32.len(),
        first_valid,
        &sweep,
    )?;

    assert_eq!(legacy_dev.rows, device_dev.rows);
    assert_eq!(legacy_dev.cols, device_dev.cols);

    let mut legacy_host = vec![0f32; legacy_dev.len()];
    legacy_dev.buf.copy_to(&mut legacy_host)?;
    let mut device_host = vec![0f32; device_dev.len()];
    device_dev.buf.copy_to(&mut device_host)?;

    let tol = 1e-4;
    for idx in 0..legacy_host.len() {
        assert!(
            approx_eq(legacy_host[idx] as f64, device_host[idx] as f64, tol),
            "MEDIUM_AD device-input mismatch at {}: legacy={} device={}",
            idx,
            legacy_host[idx],
            device_host[idx]
        );
    }

    Ok(())
}
