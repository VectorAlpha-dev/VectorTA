use vector_ta::indicators::percentile_nearest_rank as pnr;
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{percentile_nearest_rank_wrapper::CudaPercentileNearestRank, CudaRuntime};

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
fn percentile_nearest_rank_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[pnr_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut price = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        price[i] = (x * 0.00131).sin() + 0.00017 * x;
    }

    let sweep = pnr::PercentileNearestRankBatchRange {
        length: (5, 25, 5),
        percentage: (25.0, 75.0, 25.0),
    };
    let cpu = pnr::pnr_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaPercentileNearestRank::new(0).expect("CudaPercentileNearestRank::new");
    let (dev, combos) = cuda
        .pnr_batch_dev(&price_f32, &sweep)
        .expect("pnr_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    assert_eq!(combos.len(), cpu.rows);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 1e-4;
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
fn percentile_nearest_rank_cuda_many_series_one_param_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[pnr_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 1024usize;
    let mut price_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s + 2)..rows {
            let x = t as f64 + s as f64 * 0.2;
            price_tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }

    let length = 15usize;
    let percentage = 50.0f64;

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut col = vec![f64::NAN; rows];
        for t in 0..rows {
            col[t] = price_tm[t * cols + s];
        }
        let params = pnr::PercentileNearestRankParams {
            length: Some(length),
            percentage: Some(percentage),
        };
        let input = pnr::PercentileNearestRankInput::from_slice(&col, params);
        let out = pnr::percentile_nearest_rank(&input).unwrap();
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let price_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaPercentileNearestRank::new(0).expect("CudaPercentileNearestRank::new");
    let dev = cuda
        .pnr_many_series_one_param_time_major_dev(&price_tm_f32, cols, rows, length, percentage)
        .expect("pnr_many_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 1e-4;
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
fn percentile_nearest_rank_cuda_device_inputs_match_legacy_batch(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[pnr_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut price = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        price[i] = (x * 0.00117).sin() + 0.00012 * x;
    }
    let sweep = pnr::PercentileNearestRankBatchRange {
        length: (10, 20, 5),
        percentage: (25.0, 75.0, 25.0),
    };

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let first_valid = price_f32
        .iter()
        .position(|v| !v.is_nan())
        .expect("first_valid");
    let runtime = CudaRuntime::new(0).expect("runtime");
    let d_prices = runtime.upload_f32(&price_f32).expect("upload");

    let cuda = CudaPercentileNearestRank::new(0).expect("CudaPercentileNearestRank::new");
    let (legacy_dev, legacy_combos) = cuda.pnr_batch_dev(&price_f32, &sweep)?;
    let (device_dev, device_combos) = cuda.pnr_batch_dev_from_device_prices(
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

    let tol = 1e-4;
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
