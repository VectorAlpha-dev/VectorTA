use vector_ta::indicators::rocp::{
    rocp_batch_with_kernel, rocp_with_kernel, RocpBatchRange, RocpInput, RocpParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::CudaRocp;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaRuntime};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() || b.is_infinite() {
        return a.is_infinite() && b.is_infinite() && a.is_sign_positive() == b.is_sign_positive();
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
fn rocp_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rocp_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 65536usize;
    let mut data = vec![f64::NAN; len];
    for i in 100..len {
        let x = i as f64;
        data[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = RocpBatchRange {
        period: (4, 128, 5),
    };

    let cpu = rocp_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaRocp::new(0).expect("CudaRocp::new");
    let (dev, combos) = cuda
        .rocp_batch_dev(&data_f32, &sweep)
        .expect("rocp_batch_dev");
    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    assert_eq!(cpu.rows, combos.len());

    let mut gpu = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu)?;

    let tol = 1e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.values[idx], gpu[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn rocp_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rocp_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize;
    let rows = 4096usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = t as f64 + (s as f64) * 0.25;
            tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let period = 14usize;

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = tm[t * cols + s];
        }
        let input = RocpInput::from_slice(
            &series,
            RocpParams {
                period: Some(period),
            },
        );
        let out = rocp_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaRocp::new(0).expect("CudaRocp::new");
    let dev = cuda
        .rocp_many_series_one_param_time_major_dev(&tm_f32, cols, rows, period)
        .expect("rocp_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_tm)?;

    let tol = 1e-4;
    for idx in 0..gpu_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], gpu_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn rocp_cuda_device_inputs_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rocp_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 100..len {
        let x = i as f64;
        data[i] = (x * 0.00111).sin() + 0.00015 * x + 2.0;
    }
    let sweep = RocpBatchRange { period: (5, 25, 5) };

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let first_valid = data_f32
        .iter()
        .position(|v| v.is_finite())
        .expect("first_valid");
    let runtime = CudaRuntime::new(0).expect("runtime");
    let d_prices = runtime.upload_f32(&data_f32).expect("upload");

    let cuda = CudaRocp::new(0).expect("CudaRocp::new");
    let (legacy_dev, legacy_combos) = cuda.rocp_batch_dev(&data_f32, &sweep)?;
    let (device_dev, device_combos) = cuda.rocp_batch_dev_from_device_prices(
        d_prices.buffer(),
        data_f32.len(),
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

    let tol = 1e-5;
    for idx in 0..legacy_host.len() {
        assert!(
            approx_eq(legacy_host[idx] as f64, device_host[idx] as f64, tol),
            "ROCP device-input mismatch at {}: legacy={} device={}",
            idx,
            legacy_host[idx],
            device_host[idx]
        );
    }

    Ok(())
}
