use vector_ta::indicators::adxr::{
    adxr_batch_slice, adxr_with_kernel, AdxrBatchRange, AdxrInput, AdxrParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaAdxr;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaRuntime};

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
fn adxr_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[adxr_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 20_000usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 1..len {
        let x = i as f64 * 0.0021;
        let base = (x).sin() + 0.0002 * (i as f64);
        let hi = base + 0.6 + 0.05 * (x * 3.0).cos();
        let lo = base - 0.6 - 0.04 * (x * 1.7).sin();
        high[i] = hi;
        low[i] = lo;
        close[i] = (hi + lo) * 0.5;
    }
    let sweep = AdxrBatchRange { period: (5, 40, 5) };

    let cpu = adxr_batch_slice(&high, &low, &close, &sweep, Kernel::Scalar)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaAdxr::new(0).expect("CudaAdxr::new");
    let (dev, _combos) = cuda
        .adxr_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
        .expect("cuda adxr_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 2e-1;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn adxr_cuda_batch_opt_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[adxr_cuda_batch_opt_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 1..len {
        let x = i as f64 * 0.0031;
        let base = (x).sin() + 0.0002 * (i as f64);
        let hi = base + 0.6 + 0.05 * (x * 3.0).cos();
        let lo = base - 0.6 - 0.04 * (x * 1.7).sin();
        high[i] = hi;
        low[i] = lo;
        close[i] = (hi + lo) * 0.5;
    }

    let sweep = AdxrBatchRange {
        period: (5, 320, 5),
    };

    let cpu = adxr_batch_slice(&high, &low, &close, &sweep, Kernel::Scalar)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaAdxr::new(0).expect("CudaAdxr::new");
    let (dev, _combos) = cuda
        .adxr_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
        .expect("cuda adxr_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 2e-1;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn adxr_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[adxr_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 4096usize;

    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s + 1)..rows {
            let x = (t as f64) * 0.002 + (s as f64) * 0.1;
            let base = (x).cos() + 0.0001 * (t as f64);
            let hi = base + 0.5 + 0.03 * (x * 2.1).cos();
            let lo = base - 0.5 - 0.02 * (x * 1.3).sin();
            high_tm[t * cols + s] = hi;
            low_tm[t * cols + s] = lo;
            close_tm[t * cols + s] = (hi + lo) * 0.5;
        }
    }

    let period = 14usize;

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = high_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
            c[t] = close_tm[t * cols + s];
        }
        let params = AdxrParams {
            period: Some(period),
        };
        let input = AdxrInput::from_slices(&h, &l, &c, params);
        let out = adxr_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaAdxr::new(0).expect("CudaAdxr::new");
    let dev = cuda
        .adxr_many_series_one_param_time_major_dev(
            &high_tm_f32,
            &low_tm_f32,
            &close_tm_f32,
            cols,
            rows,
            period,
        )
        .expect("adxr_many_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let tol = 5e-2;
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu_tm[idx], host[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn adxr_cuda_device_inputs_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[adxr_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 2..len {
        let x = i as f64 * 0.0023;
        let base = x.sin() + 0.00015 * (i as f64);
        let hi = base + 0.55 + 0.04 * (x * 2.7).cos();
        let lo = base - 0.55 - 0.03 * (x * 1.5).sin();
        high[i] = hi;
        low[i] = lo;
        close[i] = (hi + lo) * 0.5;
    }
    let sweep = AdxrBatchRange { period: (6, 24, 6) };

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let first_valid = (0..len)
        .find(|&i| high_f32[i].is_finite() && low_f32[i].is_finite() && close_f32[i].is_finite())
        .expect("first_valid");
    let runtime = CudaRuntime::new(0).expect("runtime");
    let d_high = runtime.upload_f32(&high_f32).expect("upload high");
    let d_low = runtime.upload_f32(&low_f32).expect("upload low");
    let d_close = runtime.upload_f32(&close_f32).expect("upload close");

    let cuda = CudaAdxr::new(0).expect("CudaAdxr::new");
    let (legacy_dev, legacy_combos) =
        cuda.adxr_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)?;
    let (device_dev, device_combos) = cuda.adxr_batch_dev_from_device_inputs(
        d_high.buffer(),
        d_low.buffer(),
        d_close.buffer(),
        len,
        first_valid,
        &sweep,
    )?;

    assert_eq!(legacy_dev.rows, device_dev.rows);
    assert_eq!(legacy_dev.cols, device_dev.cols);
    assert_eq!(legacy_combos.len(), device_combos.len());

    let mut legacy_host = vec![0f32; legacy_dev.len()];
    legacy_dev.buf.copy_to(&mut legacy_host)?;
    let mut device_host = vec![0f32; device_dev.len()];
    device_dev.buf.copy_to(&mut device_host)?;

    let tol = 1e-4;
    for idx in 0..legacy_host.len() {
        assert!(
            approx_eq(legacy_host[idx] as f64, device_host[idx] as f64, tol),
            "ADXR device-input mismatch at {}: legacy={} device={}",
            idx,
            legacy_host[idx],
            device_host[idx]
        );
    }

    Ok(())
}
