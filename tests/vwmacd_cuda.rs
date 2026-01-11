

use vector_ta::indicators::vwmacd::{
    vwmacd_batch_with_kernel, VwmacdBatchRange, VwmacdBuilder, VwmacdParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaVwmacd;

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
fn vwmacd_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vwmacd_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut close = vec![f64::NAN; series_len];
    let mut volume = vec![f64::NAN; series_len];
    for i in 10..series_len {
        let x = i as f64;
        close[i] = (x * 0.0021).sin() + 0.0003 * x;
        volume[i] = (x * 0.01).cos().abs() * 120.0 + 10.0;
    }

    let sweep = VwmacdBatchRange {
        fast: (8, 16, 4),
        slow: (20, 28, 4),
        signal: (9, 9, 0),
        fast_ma_type: "sma".into(),
        slow_ma_type: "sma".into(),
        signal_ma_type: "ema".into(),
    };

    let cpu = vwmacd_batch_with_kernel(&close, &volume, &sweep, Kernel::ScalarBatch)?;
    let cpu_macd: Vec<f32> = cpu.macd.iter().map(|&v| v as f32).collect();
    let cpu_signal: Vec<f32> = cpu.signal.iter().map(|&v| v as f32).collect();
    let cpu_hist: Vec<f32> = cpu.hist.iter().map(|&v| v as f32).collect();

    let cuda = CudaVwmacd::new(0).map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let (dev, combos) = cuda
        .vwmacd_batch_dev(&close_f32, &volume_f32, &sweep)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    assert_eq!(cpu.rows, dev.rows());
    assert_eq!(cpu.cols, dev.cols());

    let mut macd_gpu = vec![0f32; cpu.rows * cpu.cols];
    let mut signal_gpu = vec![0f32; cpu.rows * cpu.cols];
    let mut hist_gpu = vec![0f32; cpu.rows * cpu.cols];
    dev.macd.buf.copy_to(&mut macd_gpu)?;
    dev.signal.buf.copy_to(&mut signal_gpu)?;
    dev.hist.buf.copy_to(&mut hist_gpu)?;

    let tol = 1e-4;
    for i in 0..macd_gpu.len() {
        assert!(
            approx_eq(cpu_macd[i] as f64, macd_gpu[i] as f64, tol),
            "MACD mismatch at {}",
            i
        );
        assert!(
            approx_eq(cpu_signal[i] as f64, signal_gpu[i] as f64, tol),
            "Signal mismatch at {}",
            i
        );
        assert!(
            approx_eq(cpu_hist[i] as f64, hist_gpu[i] as f64, tol),
            "Hist mismatch at {}",
            i
        );
    }

    
    let target = VwmacdParams {
        fast_period: Some(12),
        slow_period: Some(24),
        signal_period: Some(9),
        fast_ma_type: Some("sma".into()),
        slow_ma_type: Some("sma".into()),
        signal_ma_type: Some("ema".into()),
    };
    let cpu_single = VwmacdBuilder::new()
        .fast(12)
        .slow(24)
        .signal(9)
        .apply_slices(&close, &volume)?
        .macd;
    let row = combos
        .iter()
        .position(|p| {
            p.fast_period == target.fast_period
                && p.slow_period == target.slow_period
                && p.signal_period == target.signal_period
        })
        .unwrap();
    let gpu_row = &macd_gpu[row * series_len..(row + 1) * series_len];
    for j in 0..series_len {
        assert!(approx_eq(cpu_single[j], gpu_row[j] as f64, 1e-4));
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn vwmacd_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vwmacd_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 5usize; 
    let rows = 2048usize; 
    let mut prices_tm = vec![f64::NAN; rows * cols];
    let mut volumes_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        for r in (s + 8)..rows {
            let x = r as f64 + (s as f64) * 0.5;
            prices_tm[r * cols + s] = (x * 0.003).cos() + 0.0004 * x;
            volumes_tm[r * cols + s] = (x * 0.02).sin().abs() * 80.0 + 20.0 + s as f64;
        }
    }

    let f = 12usize;
    let sl = 26usize;
    let g = 9usize;

    
    let mut cpu_macd = vec![f32::NAN; rows * cols];
    let mut cpu_signal = vec![f32::NAN; rows * cols];
    let mut cpu_hist = vec![f32::NAN; rows * cols];
    for s in 0..cols {
        let mut c = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for r in 0..rows {
            let idx = r * cols + s;
            c[r] = prices_tm[idx];
            v[r] = volumes_tm[idx];
        }
        let out = VwmacdBuilder::new()
            .fast(f)
            .slow(sl)
            .signal(g)
            .apply_slices(&c, &v)?;
        for r in 0..rows {
            let idx = r * cols + s;
            cpu_macd[idx] = out.macd[r] as f32;
            cpu_signal[idx] = out.signal[r] as f32;
            cpu_hist[idx] = out.hist[r] as f32;
        }
    }

    
    let cuda = CudaVwmacd::new(0).map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    let p32: Vec<f32> = prices_tm.iter().map(|&x| x as f32).collect();
    let v32: Vec<f32> = volumes_tm.iter().map(|&x| x as f32).collect();
    let prm = VwmacdParams {
        fast_period: Some(f),
        slow_period: Some(sl),
        signal_period: Some(g),
        fast_ma_type: Some("sma".into()),
        slow_ma_type: Some("sma".into()),
        signal_ma_type: Some("ema".into()),
    };
    let dev = cuda
        .vwmacd_many_series_one_param_time_major_dev(&p32, &v32, cols, rows, &prm)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    let mut macd_gpu = vec![0f32; rows * cols];
    let mut signal_gpu = vec![0f32; rows * cols];
    let mut hist_gpu = vec![0f32; rows * cols];
    dev.macd.buf.copy_to(&mut macd_gpu)?;
    dev.signal.buf.copy_to(&mut signal_gpu)?;
    dev.hist.buf.copy_to(&mut hist_gpu)?;

    let tol = 1e-4;
    for i in 0..macd_gpu.len() {
        assert!(
            approx_eq(cpu_macd[i] as f64, macd_gpu[i] as f64, tol),
            "MACD mismatch at {}",
            i
        );
        assert!(
            approx_eq(cpu_signal[i] as f64, signal_gpu[i] as f64, tol),
            "Signal mismatch at {}",
            i
        );
        assert!(
            approx_eq(cpu_hist[i] as f64, hist_gpu[i] as f64, tol),
            "Hist mismatch at {}",
            i
        );
    }
    Ok(())
}
