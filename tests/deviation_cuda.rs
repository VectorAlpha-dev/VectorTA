use vector_ta::utilities::enums::Kernel;

use vector_ta::indicators::deviation::{
    deviation_batch_with_kernel, DeviationBatchRange, DeviationBuilder, DeviationParams,
};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaDeviation, CudaRuntime};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs());
    diff <= tol + scale * (5.0 * tol)
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
fn deviation_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[deviation_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 6..len {
        let x = i as f64;
        let base = (x * 0.00041).sin() - (x * 0.00017).cos();
        data[i] = base + 0.0013 * ((i % 9) as f64 - 4.0);
        if i % 257 == 0 {
            data[i] = f64::NAN;
        }
    }

    let sweep = DeviationBatchRange {
        period: (10, 40, 10),
        devtype: (0, 0, 0),
    };
    let cpu = deviation_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaDeviation::new(0).expect("CudaDeviation::new");
    let (dev_arr, combos) = cuda
        .deviation_batch_dev(&data_f32, &sweep)
        .expect("deviation_cuda_batch_dev");

    assert_eq!(dev_arr.rows, cpu.rows);
    assert_eq!(dev_arr.cols, cpu.cols);
    assert_eq!(combos.len(), cpu.combos.len());
    for (c, p) in combos.iter().zip(cpu.combos.iter()) {
        assert_eq!(c.period.unwrap(), p.period.unwrap());
        assert_eq!(c.devtype.unwrap(), 0);
    }

    let mut gpu = vec![0f32; dev_arr.rows * dev_arr.cols];
    dev_arr.buf.copy_to(&mut gpu).expect("copy results");

    let tol = 5e-4;
    for (idx, (&cpu_v, &gpu_v)) in cpu.values.iter().zip(gpu.iter()).enumerate() {
        assert!(
            approx_eq(cpu_v, gpu_v as f64, tol),
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
fn deviation_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[deviation_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let rows = 2048usize;
    let cols = 5usize;
    let mut data_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        for t in (s + 3)..rows {
            let x = t as f64 + (s as f64) * 0.1;
            data_tm[t * cols + s] = (x * 0.0031).cos() + 0.0007 * x;
            if t % 211 == 0 {
                data_tm[t * cols + s] = f64::NAN;
            }
        }
    }
    let period = 14usize;
    let params = DeviationParams {
        period: Some(period),
        devtype: Some(0),
    };

    let mut cpu = vec![f32::NAN; rows * cols];
    for s in 0..cols {
        let mut col = vec![f64::NAN; rows];
        for t in 0..rows {
            col[t] = data_tm[t * cols + s];
        }
        let out = DeviationBuilder::default()
            .period(period)
            .devtype(0)
            .apply_slice(&col)?
            .values;
        for t in 0..rows {
            cpu[t * cols + s] = out[t] as f32;
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaDeviation::new(0).expect("CudaDeviation::new");
    let dev = cuda
        .deviation_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("deviation many-series");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut gpu = vec![0f32; rows * cols];
    dev.buf.copy_to(&mut gpu).expect("copy many-series");

    let tol = 5e-4;
    for i in 0..(rows * cols) {
        assert!(
            approx_eq(cpu[i] as f64, gpu[i] as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            i,
            cpu[i],
            gpu[i]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn deviation_cuda_device_prices_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[deviation_cuda_device_prices_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let first_valid = 6usize;
    let mut data_f32 = vec![f32::NAN; len];
    for (i, value) in data_f32.iter_mut().enumerate().skip(first_valid) {
        let x = i as f32;
        let base = (x * 0.00041).sin() - (x * 0.00017).cos();
        *value = base + 0.0013 * ((i % 9) as f32 - 4.0);
        if i % 257 == 0 {
            *value = f32::NAN;
        }
    }

    let sweep = DeviationBatchRange {
        period: (10, 40, 10),
        devtype: (0, 0, 0),
    };

    let cuda = CudaDeviation::new(0).expect("CudaDeviation::new");
    let (legacy, legacy_params) = cuda
        .deviation_batch_dev(&data_f32, &sweep)
        .expect("legacy batch");

    let runtime = CudaRuntime::new(0).expect("runtime");
    let device_prices = runtime.upload_f32(&data_f32).expect("upload prices");
    let (device, device_params) = cuda
        .deviation_batch_dev_from_device_prices(device_prices.buffer(), len, first_valid, &sweep)
        .expect("device prices");
    cuda.synchronize().expect("sync");

    assert_eq!(legacy.rows, device.rows);
    assert_eq!(legacy.cols, device.cols);
    assert_eq!(legacy_params.len(), device_params.len());
    for (lhs, rhs) in legacy_params.iter().zip(device_params.iter()) {
        assert_eq!(lhs.period, rhs.period);
        assert_eq!(lhs.devtype, rhs.devtype);
    }

    let mut legacy_host = vec![0f32; legacy.len()];
    legacy.buf.copy_to(&mut legacy_host)?;
    let mut device_host = vec![0f32; device.len()];
    device.buf.copy_to(&mut device_host)?;

    for (idx, (&lhs, &rhs)) in legacy_host.iter().zip(device_host.iter()).enumerate() {
        assert!(
            approx_eq(lhs as f64, rhs as f64, 5e-4),
            "mismatch at {}: legacy={} device={}",
            idx,
            lhs,
            rhs
        );
    }

    Ok(())
}
