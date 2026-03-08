use vector_ta::indicators::cfo::{
    cfo_batch_with_kernel, cfo_with_kernel, CfoBatchRange, CfoInput, CfoParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::cfo_wrapper::CudaCfo;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaRuntime;

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
fn cfo_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cfo_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 16_384usize;
    let mut price = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00011 * x;
    }
    let sweep = CfoBatchRange {
        period: (8, 8 + 63, 1),
        scalar: (100.0, 100.0, 0.0),
    };

    let cpu = cfo_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaCfo::new(0).expect("CudaCfo::new");
    let dev = cuda
        .cfo_batch_dev(&price_f32, &sweep)
        .expect("cfo_batch_dev");
    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 3e-3;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "batch mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cfo_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cfo_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 12usize;
    let rows = 8192usize;
    let mut data_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s % 7)..rows {
            let x = (t as f64) + (s as f64) * 0.37;
            data_tm[t * cols + s] = (x * 0.0019).sin() + 0.00021 * x;
        }
    }

    let period = 14usize;
    let scalar = 100.0f64;

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = data_tm[t * cols + s];
        }
        let params = CfoParams {
            period: Some(period),
            scalar: Some(scalar),
        };
        let input = CfoInput::from_slice(&p, params);
        let out = cfo_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaCfo::new(0).expect("CudaCfo::new");
    let dev = cuda
        .cfo_many_series_one_param_time_major_dev(
            &data_tm_f32,
            cols,
            rows,
            &CfoParams {
                period: Some(period),
                scalar: Some(scalar),
            },
        )
        .expect("cfo_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 3e-3;
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu_tm[idx], host[idx] as f64, tol),
            "many-series mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cfo_cuda_device_inputs_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cfo_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut price = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00011 * x;
    }
    let sweep = CfoBatchRange {
        period: (8, 40, 8),
        scalar: (100.0, 100.0, 0.0),
    };

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let first_valid = price_f32
        .iter()
        .position(|v| v.is_finite())
        .expect("first_valid");
    let runtime = CudaRuntime::new(0).expect("runtime");
    let d_price = runtime.upload_f32(&price_f32).expect("upload");

    let cuda = CudaCfo::new(0).expect("CudaCfo::new");
    let legacy = cuda.cfo_batch_dev(&price_f32, &sweep).expect("legacy cfo");
    let device = cuda
        .cfo_batch_dev_from_device_prices(d_price.buffer(), price_f32.len(), first_valid, &sweep)
        .expect("device cfo");

    assert_eq!(legacy.rows, device.rows);
    assert_eq!(legacy.cols, device.cols);

    let mut legacy_host = vec![0f32; legacy.len()];
    let mut device_host = vec![0f32; device.len()];
    legacy.buf.copy_to(&mut legacy_host)?;
    device.buf.copy_to(&mut device_host)?;

    let tol = 1e-5;
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
