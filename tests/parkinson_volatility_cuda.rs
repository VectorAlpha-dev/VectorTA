use vector_ta::indicators::parkinson_volatility::{
    parkinson_volatility_batch_with_kernel, parkinson_volatility_with_kernel,
    ParkinsonVolatilityBatchRange, ParkinsonVolatilityInput, ParkinsonVolatilityParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{CudaParkinsonVolatility, CudaRuntime};

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
fn parkinson_volatility_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[parkinson_volatility_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 6..len {
        let x = i as f64;
        let base = 100.0 + 0.01 * x + (x * 0.0021).sin();
        let spread = 0.8 + (x * 0.0017).cos().abs() * 0.2;
        high[i] = base + spread;
        low[i] = (base - spread).max(0.01);
    }

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let high_q: Vec<f64> = high_f32.iter().map(|&v| v as f64).collect();
    let low_q: Vec<f64> = low_f32.iter().map(|&v| v as f64).collect();
    let sweep = ParkinsonVolatilityBatchRange { period: (8, 32, 8) };

    let cpu = parkinson_volatility_batch_with_kernel(&high_q, &low_q, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaParkinsonVolatility::new(0)?;
    let gpu = cuda.parkinson_volatility_batch_dev(&high_f32, &low_f32, &sweep)?;

    assert_eq!(cpu.rows, gpu.outputs.rows());
    assert_eq!(cpu.cols, gpu.outputs.cols());
    assert_eq!(cpu.combos.len(), gpu.combos.len());

    let mut vol_gpu = vec![0f32; gpu.outputs.volatility.len()];
    let mut var_gpu = vec![0f32; gpu.outputs.variance.len()];
    gpu.outputs.volatility.buf.copy_to(&mut vol_gpu)?;
    gpu.outputs.variance.buf.copy_to(&mut var_gpu)?;

    let tol = 1e-4;
    for i in 0..cpu.volatility.len() {
        assert!(
            approx_eq(cpu.volatility[i], vol_gpu[i] as f64, tol),
            "vol mismatch at {}: cpu={} gpu={}",
            i,
            cpu.volatility[i],
            vol_gpu[i]
        );
        assert!(
            approx_eq(cpu.variance[i], var_gpu[i] as f64, tol),
            "variance mismatch at {}: cpu={} gpu={}",
            i,
            cpu.variance[i],
            var_gpu[i]
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn parkinson_volatility_cuda_many_series_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[parkinson_volatility_cuda_many_series_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 6usize;
    let rows = 2048usize;
    let period = 14usize;
    let mut high_tm = vec![f32::NAN; cols * rows];
    let mut low_tm = vec![f32::NAN; cols * rows];

    for s in 0..cols {
        for t in s..rows {
            let x = t as f64 + s as f64 * 0.5;
            let base = 80.0 + 0.02 * x + (x * 0.0031).sin();
            let spread = 0.6 + (x * 0.0013).cos().abs() * 0.15;
            let idx = t * cols + s;
            high_tm[idx] = (base + spread) as f32;
            low_tm[idx] = (base - spread).max(0.01) as f32;
        }
    }

    let mut cpu_vol_tm = vec![f64::NAN; cols * rows];
    let mut cpu_var_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut high = vec![f64::NAN; rows];
        let mut low = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            high[t] = high_tm[idx] as f64;
            low[t] = low_tm[idx] as f64;
        }
        let params = ParkinsonVolatilityParams {
            period: Some(period),
        };
        let input = ParkinsonVolatilityInput::from_slices(&high, &low, params);
        let out = parkinson_volatility_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let idx = t * cols + s;
            cpu_vol_tm[idx] = out.volatility[t];
            cpu_var_tm[idx] = out.variance[t];
        }
    }

    let cuda = CudaParkinsonVolatility::new(0)?;
    let dev = cuda.parkinson_volatility_many_series_one_param_time_major_dev(
        &high_tm, &low_tm, cols, rows, period,
    )?;
    let mut vol_gpu = vec![0f32; dev.volatility.len()];
    let mut var_gpu = vec![0f32; dev.variance.len()];
    dev.volatility.buf.copy_to(&mut vol_gpu)?;
    dev.variance.buf.copy_to(&mut var_gpu)?;

    let tol = 1e-4;
    for i in 0..vol_gpu.len() {
        assert!(
            approx_eq(cpu_vol_tm[i], vol_gpu[i] as f64, tol),
            "vol mismatch at {}: cpu={} gpu={}",
            i,
            cpu_vol_tm[i],
            vol_gpu[i]
        );
        assert!(
            approx_eq(cpu_var_tm[i], var_gpu[i] as f64, tol),
            "var mismatch at {}: cpu={} gpu={}",
            i,
            cpu_var_tm[i],
            var_gpu[i]
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn parkinson_volatility_cuda_device_inputs_match_legacy_batch(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[parkinson_volatility_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 8192usize;
    let first_valid = 6usize;
    let mut high_f32 = vec![f32::NAN; len];
    let mut low_f32 = vec![f32::NAN; len];
    for i in first_valid..len {
        let x = i as f32;
        let base = 75.0 + 0.01 * x + (x * 0.0021).sin();
        let spread = 0.4 + (x * 0.0017).cos().abs() * 0.1;
        high_f32[i] = base + spread;
        low_f32[i] = (base - spread).max(0.01);
    }
    let sweep = ParkinsonVolatilityBatchRange { period: (8, 32, 8) };

    let cuda = CudaParkinsonVolatility::new(0)?;
    let legacy = cuda.parkinson_volatility_batch_dev(&high_f32, &low_f32, &sweep)?;

    let runtime = CudaRuntime::new(0)?;
    let d_high = runtime.upload_f32(&high_f32)?;
    let d_low = runtime.upload_f32(&low_f32)?;
    let device = cuda.parkinson_volatility_batch_dev_from_device_inputs(
        d_high.buffer(),
        d_low.buffer(),
        len,
        first_valid,
        &sweep,
    )?;
    cuda.synchronize()?;

    assert_eq!(legacy.outputs.rows(), device.outputs.rows());
    assert_eq!(legacy.outputs.cols(), device.outputs.cols());
    assert_eq!(legacy.combos.len(), device.combos.len());

    let mut legacy_vol = vec![0f32; legacy.outputs.volatility.len()];
    let mut legacy_var = vec![0f32; legacy.outputs.variance.len()];
    legacy.outputs.volatility.buf.copy_to(&mut legacy_vol)?;
    legacy.outputs.variance.buf.copy_to(&mut legacy_var)?;

    let mut device_vol = vec![0f32; device.outputs.volatility.len()];
    let mut device_var = vec![0f32; device.outputs.variance.len()];
    device.outputs.volatility.buf.copy_to(&mut device_vol)?;
    device.outputs.variance.buf.copy_to(&mut device_var)?;

    let tol = 1e-5;
    for i in 0..legacy_vol.len() {
        assert!(
            approx_eq(legacy_vol[i] as f64, device_vol[i] as f64, tol),
            "vol mismatch at {}: legacy={} device={}",
            i,
            legacy_vol[i],
            device_vol[i]
        );
        assert!(
            approx_eq(legacy_var[i] as f64, device_var[i] as f64, tol),
            "var mismatch at {}: legacy={} device={}",
            i,
            legacy_var[i],
            device_var[i]
        );
    }

    Ok(())
}
