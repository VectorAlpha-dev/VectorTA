// Integration tests for CUDA PPO kernels

use my_project::indicators::ppo::{
    ppo_batch_with_kernel, ppo_with_kernel, PpoBatchRange, PpoData, PpoInput, PpoParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::ppo_wrapper::CudaPpo;

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
fn make_prices(len: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; len];
    for i in 7..len {
        let x = i as f64;
        v[i] = (x * 0.00131).sin() + 0.00009 * x;
    }
    v
}

#[cfg(feature = "cuda")]
#[test]
fn ppo_cuda_batch_matches_cpu_sma() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ppo_cuda_batch_matches_cpu_sma] skipped - no CUDA device");
        return Ok(());
    }
    let len = 24_000usize;
    let price = make_prices(len);
    let sweep = PpoBatchRange {
        fast_period: (4, 22, 3),
        slow_period: (8, 30, 4),
        ma_type: "sma".into(),
    };

    let cpu = ppo_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaPpo::new(0).expect("CudaPpo::new");
    let (dev, combos) = cuda
        .ppo_batch_dev(&price_f32, &sweep)
        .expect("ppo_batch_dev");
    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    assert_eq!(combos.len(), cpu.rows);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let tol = 8e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.values[idx], host[idx] as f64, tol),
            "batch sma mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ppo_cuda_batch_matches_cpu_ema() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ppo_cuda_batch_matches_cpu_ema] skipped - no CUDA device");
        return Ok(());
    }
    let len = 20_000usize;
    let price = make_prices(len);
    let sweep = PpoBatchRange {
        fast_period: (5, 20, 5),
        slow_period: (10, 30, 10),
        ma_type: "ema".into(),
    };

    let cpu = ppo_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaPpo::new(0).expect("CudaPpo::new");
    let (dev, _combos) = cuda
        .ppo_batch_dev(&price_f32, &sweep)
        .expect("ppo_batch_dev");
    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let tol = 1.5e-3; // EMA recurrence fp32 vs fp64
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.values[idx], host[idx] as f64, tol),
            "batch ema mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ppo_cuda_many_series_one_param_matches_cpu_sma() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ppo_cuda_many_series_one_param_matches_cpu_sma] skipped - no CUDA device");
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
    let params = PpoParams {
        fast_period: Some(12),
        slow_period: Some(26),
        ma_type: Some("sma".into()),
    };

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = data_tm[t * cols + s];
        }
        let input = PpoInput {
            data: PpoData::Slice(&p),
            params: params.clone(),
        };
        let out = ppo_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaPpo::new(0).expect("CudaPpo::new");
    let dev = cuda
        .ppo_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("ppo many-series");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let tol = 8e-4;
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu_tm[idx], host[idx] as f64, tol),
            "many-series sma mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ppo_cuda_many_series_one_param_matches_cpu_ema() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ppo_cuda_many_series_one_param_matches_cpu_ema] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 10usize;
    let rows = 8192usize;
    let mut data_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s % 5)..rows {
            let x = (t as f64) + (s as f64) * 0.23;
            data_tm[t * cols + s] = (x * 0.0017).sin() + 0.00019 * x;
        }
    }
    let params = PpoParams {
        fast_period: Some(10),
        slow_period: Some(21),
        ma_type: Some("ema".into()),
    };

    // CPU baseline
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = data_tm[t * cols + s];
        }
        let input = PpoInput {
            data: PpoData::Slice(&p),
            params: params.clone(),
        };
        let out = ppo_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaPpo::new(0).expect("CudaPpo::new");
    let dev = cuda
        .ppo_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("ppo many-series ema");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let tol = 1.5e-3;
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu_tm[idx], host[idx] as f64, tol),
            "many-series ema mismatch at {}",
            idx
        );
    }
    Ok(())
}
