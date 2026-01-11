

use vector_ta::indicators::ppo::{
    ppo_batch_with_kernel, ppo_with_kernel, PpoBatchRange, PpoData, PpoInput, PpoParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::ppo_wrapper::CudaPpo;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

fn approx_eq_absrel(a: f64, b: f64, abs_tol: f64, rel_tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    let scale = a.abs().max(1.0);
    diff <= abs_tol.max(rel_tol * scale)
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

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let price32_as_f64: Vec<f64> = price_f32.iter().map(|&v| v as f64).collect();
    let cpu = ppo_batch_with_kernel(&price32_as_f64, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaPpo::new(0).expect("CudaPpo::new");
    let (dev, combos) = cuda
        .ppo_batch_dev(&price_f32, &sweep)
        .expect("ppo_batch_dev");
    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    assert_eq!(combos.len(), cpu.rows);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let tol = 3e-3;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "batch sma mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
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

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let price32_as_f64: Vec<f64> = price_f32.iter().map(|&v| v as f64).collect();
    let cpu = ppo_batch_with_kernel(&price32_as_f64, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaPpo::new(0).expect("CudaPpo::new");
    let (dev, _combos) = cuda
        .ppo_batch_dev(&price_f32, &sweep)
        .expect("ppo_batch_dev");
    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let abs_tol = 2.0;
    let rel_tol = 1e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq_absrel(c, g, abs_tol, rel_tol),
            "batch ema mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
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

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();

    
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = data_tm_f32[t * cols + s] as f64;
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

    let cuda = CudaPpo::new(0).expect("CudaPpo::new");
    let dev = cuda
        .ppo_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("ppo many-series");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let tol = 3e-3;
    for idx in 0..host.len() {
        let c = cpu_tm[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "many-series sma mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
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

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();

    
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = data_tm_f32[t * cols + s] as f64;
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

    let cuda = CudaPpo::new(0).expect("CudaPpo::new");
    let dev = cuda
        .ppo_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("ppo many-series ema");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let abs_tol = 2.0;
    let rel_tol = 1e-4;
    for idx in 0..host.len() {
        assert!(
            approx_eq_absrel(cpu_tm[idx], host[idx] as f64, abs_tol, rel_tol),
            "many-series ema mismatch at {}",
            idx
        );
    }
    Ok(())
}
