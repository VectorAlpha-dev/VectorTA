use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{CudaRuntime, CudaUi};

use vector_ta::indicators::ui::{ui_batch_slice, ui_with_kernel, UiBatchRange, UiInput, UiParams};

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
fn ui_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ui_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 32_768usize;
    let mut price = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        price[i] = (x * 0.0017).sin() + 0.00019 * x;
    }
    let sweep = UiBatchRange {
        period: (10, 28, 3),
        scalar: (50.0, 150.0, 25.0),
    };

    let cpu = ui_batch_slice(&price, &sweep, Kernel::Scalar)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaUi::new(0).expect("CudaUi::new");
    let (gpu_dev, combos) = cuda.ui_batch_dev(&price_f32, &sweep)?;

    assert_eq!(cpu.rows, gpu_dev.rows);
    assert_eq!(cpu.cols, gpu_dev.cols);
    assert_eq!(combos.len(), cpu.rows);

    let mut gpu_host = vec![0f32; gpu_dev.len()];
    gpu_dev.buf.copy_to(&mut gpu_host)?;

    let tol = 2e-3;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = gpu_host[idx] as f64;
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
fn ui_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ui_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize;
    let rows = 8192usize;
    let mut price_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in 0..rows {
            let x = t as f64 * 0.002 + s as f64 * 0.01;
            price_tm[t * cols + s] = (x * 0.81).sin() + 0.0005 * x;
        }
    }
    let period = 14usize;
    let scalar = 100.0f64;

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = price_tm[t * cols + s];
        }
        let params = UiParams {
            period: Some(period),
            scalar: Some(scalar),
        };
        let input = UiInput {
            data: vector_ta::indicators::ui::UiData::Slice(&series),
            params,
        };
        let out = ui_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let price_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaUi::new(0).expect("CudaUi::new");
    let params = UiParams {
        period: Some(period),
        scalar: Some(scalar),
    };
    let gpu_dev_tm = cuda
        .ui_many_series_one_param_time_major_dev(&price_tm_f32, cols, rows, &params)
        .expect("ui_many_series_one_param_time_major_dev");

    assert_eq!(gpu_dev_tm.rows, rows);
    assert_eq!(gpu_dev_tm.cols, cols);
    let mut gpu_tm = vec![0f32; gpu_dev_tm.len()];
    gpu_dev_tm.buf.copy_to(&mut gpu_tm)?;

    let tol = 2e-3;
    for idx in 0..gpu_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], gpu_tm[idx] as f64, tol),
            "many-series mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ui_cuda_device_inputs_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ui_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 16_384usize;
    let mut price = vec![f64::NAN; len];
    for i in 9..len {
        let x = i as f64;
        price[i] = (x * 0.0011).sin() + 0.00015 * x;
    }
    let sweep = UiBatchRange {
        period: (10, 22, 4),
        scalar: (80.0, 120.0, 20.0),
    };

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let runtime = CudaRuntime::new(0).expect("runtime");
    let d_prices = runtime.upload_f32(&price_f32).expect("upload prices");
    let first_valid = price_f32
        .iter()
        .position(|v| v.is_finite())
        .expect("first_valid");

    let cuda = CudaUi::new(0).expect("CudaUi::new");
    let (legacy_dev, legacy_combos) = cuda.ui_batch_dev(&price_f32, &sweep)?;
    let (device_dev, device_combos) = cuda.ui_batch_dev_from_device_prices(
        d_prices.buffer(),
        price_f32.len(),
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

    let tol = 2e-3;
    for idx in 0..legacy_host.len() {
        assert!(
            approx_eq(legacy_host[idx] as f64, device_host[idx] as f64, tol),
            "mismatch at {}: legacy={} device={}",
            idx,
            legacy_host[idx],
            device_host[idx]
        );
    }

    Ok(())
}
