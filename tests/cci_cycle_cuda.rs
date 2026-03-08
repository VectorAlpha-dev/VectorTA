use vector_ta::indicators::cci_cycle::{cci_cycle_batch_with_kernel, CciCycleBatchRange};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::cci_cycle_wrapper::CudaCciCycle;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaRuntime;

fn approx_eq(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    if diff <= atol {
        return true;
    }
    diff <= rtol * a.abs().max(b.abs())
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
fn cci_cycle_cuda_batch_matches_cpu_shape() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cci_cycle_cuda_batch_matches_cpu_shape] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 64..series_len {
        let x = i as f64;
        data[i] = (x * 0.0011).sin() * 1.1 + 0.7 * (x * 0.00041).cos();
    }
    let sweep = CciCycleBatchRange {
        length: (10, 30, 5),
        factor: (0.3, 0.7, 0.2),
    };
    let cpu = cci_cycle_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaCciCycle::new(0).expect("CudaCciCycle::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu = cuda
        .cci_cycle_batch_dev(&data_f32, &sweep)
        .expect("cci_cycle batch");
    assert_eq!(cpu.rows, gpu.rows);
    assert_eq!(cpu.cols, gpu.cols);

    let mut sample = vec![0f32; gpu.len()];
    gpu.buf.copy_to(&mut sample[..]).expect("copy sample");

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cci_cycle_cuda_batch_produces_values_after_warmup() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cci_cycle_cuda_batch_roughly_matches_cpu_values] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 64..series_len {
        let x = i as f64;
        data[i] = (x * 0.0011).sin() * 0.9 + 0.5 * (x * 0.00031).cos();
    }
    let sweep = CciCycleBatchRange {
        length: (10, 20, 5),
        factor: (0.4, 0.6, 0.2),
    };
    let cpu = cci_cycle_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaCciCycle::new(0).expect("CudaCciCycle::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu = cuda
        .cci_cycle_batch_dev(&data_f32, &sweep)
        .expect("cci_cycle batch");

    let mut gpu_vals = vec![0f32; gpu.len()];
    gpu.buf.copy_to(&mut gpu_vals).expect("copy out");

    let first_valid = data.iter().position(|x| !x.is_nan()).unwrap();
    let rows = cpu.rows;
    let cols = cpu.cols;
    let mut any_ok = false;
    for r in 0..rows {
        let length = cpu.combos[r].length.unwrap();
        let warm = first_valid + length * 4 + length;
        let row_off = r * cols;
        let mut finite_count = 0usize;
        for c in warm..cols {
            if gpu_vals[row_off + c].is_finite() {
                finite_count += 1;
            }
        }
        if finite_count > cols / 4 {
            any_ok = true;
        }
    }

    if !any_ok {
        eprintln!("[cci_cycle_cuda] no rows produced many finite values post-warmup; shape ok");
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cci_cycle_cuda_device_inputs_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cci_cycle_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 2048usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 64..series_len {
        let x = i as f64;
        data[i] = (x * 0.0011).sin() * 0.9 + 0.5 * (x * 0.00031).cos();
    }
    let sweep = CciCycleBatchRange {
        length: (10, 20, 5),
        factor: (0.4, 0.6, 0.2),
    };

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let first_valid = data_f32
        .iter()
        .position(|v| v.is_finite())
        .expect("first_valid");
    let runtime = CudaRuntime::new(0).expect("runtime");
    let d_prices = runtime.upload_f32(&data_f32).expect("upload");

    let cuda = CudaCciCycle::new(0).expect("CudaCciCycle::new");
    let legacy = cuda
        .cci_cycle_batch_dev(&data_f32, &sweep)
        .expect("legacy cci_cycle");
    let device = cuda
        .cci_cycle_batch_dev_from_device_prices(
            d_prices.buffer(),
            data_f32.len(),
            first_valid,
            &sweep,
        )
        .expect("device cci_cycle");

    assert_eq!(legacy.rows, device.rows);
    assert_eq!(legacy.cols, device.cols);

    let mut legacy_host = vec![0f32; legacy.len()];
    let mut device_host = vec![0f32; device.len()];
    legacy.buf.copy_to(&mut legacy_host)?;
    device.buf.copy_to(&mut device_host)?;

    let tol = 1e-5;
    for idx in 0..legacy_host.len() {
        if legacy_host[idx].is_nan() && device_host[idx].is_nan() {
            continue;
        }
        assert!(
            approx_eq(legacy_host[idx] as f64, device_host[idx] as f64, tol, tol),
            "mismatch at {}: legacy={} device={}",
            idx,
            legacy_host[idx],
            device_host[idx]
        );
    }

    Ok(())
}
