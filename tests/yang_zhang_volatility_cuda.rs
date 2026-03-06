use vector_ta::indicators::yang_zhang_volatility::{
    yang_zhang_volatility_batch_with_kernel, yang_zhang_volatility_with_kernel,
    YangZhangVolatilityBatchRange, YangZhangVolatilityInput, YangZhangVolatilityParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaYangZhangVolatility;

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
fn yang_zhang_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[yang_zhang_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut open = vec![0.0f64; len];
    let mut high = vec![0.0f64; len];
    let mut low = vec![0.0f64; len];
    let mut close = vec![0.0f64; len];
    let mut prev = 1000.0f64;
    for i in 0..len {
        let x = i as f64;
        let o = (prev + 0.0002 * x + (x * 0.0017).sin() * 2.0 + (x * 0.00031).cos()).max(1.0);
        let c = (o + (x * 0.0013).sin() * 0.8).max(1.0);
        let h = o.max(c) + 0.4 + (x * 0.00091).cos().abs() * 0.05;
        let l = (o.min(c) - 0.4 - (x * 0.00111).sin().abs() * 0.05).max(0.01);
        open[i] = o;
        high[i] = h;
        low[i] = l;
        close[i] = c.max(0.01);
        prev = close[i];
    }

    let open_f32: Vec<f32> = open.iter().map(|&v| v as f32).collect();
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();

    let open_q: Vec<f64> = open_f32.iter().map(|&v| v as f64).collect();
    let high_q: Vec<f64> = high_f32.iter().map(|&v| v as f64).collect();
    let low_q: Vec<f64> = low_f32.iter().map(|&v| v as f64).collect();
    let close_q: Vec<f64> = close_f32.iter().map(|&v| v as f64).collect();

    let sweep = YangZhangVolatilityBatchRange {
        lookback: (10, 30, 10),
        k_override: false,
        k: (0.34, 0.34, 0.0),
    };

    let cpu = yang_zhang_volatility_batch_with_kernel(
        &open_q,
        &high_q,
        &low_q,
        &close_q,
        &sweep,
        Kernel::ScalarBatch,
    )?;

    let cuda = CudaYangZhangVolatility::new(0)?;
    let gpu_res =
        cuda.yang_zhang_volatility_batch_dev(&open_f32, &high_f32, &low_f32, &close_f32, &sweep)?;

    assert_eq!(gpu_res.outputs.rows(), cpu.rows);
    assert_eq!(gpu_res.outputs.cols(), cpu.cols);
    assert_eq!(gpu_res.combos.len(), cpu.combos.len());

    let mut yz_gpu = vec![0f32; gpu_res.outputs.yz.len()];
    let mut rs_gpu = vec![0f32; gpu_res.outputs.rs.len()];
    gpu_res.outputs.yz.buf.copy_to(&mut yz_gpu)?;
    gpu_res.outputs.rs.buf.copy_to(&mut rs_gpu)?;

    let tol = 2e-3;
    for i in 0..cpu.yz.len() {
        assert!(
            approx_eq(cpu.yz[i], yz_gpu[i] as f64, tol),
            "yz mismatch at {}: cpu={} gpu={}",
            i,
            cpu.yz[i],
            yz_gpu[i]
        );
        assert!(
            approx_eq(cpu.rs[i], rs_gpu[i] as f64, tol),
            "rs mismatch at {}: cpu={} gpu={}",
            i,
            cpu.rs[i],
            rs_gpu[i]
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn yang_zhang_cuda_many_series_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[yang_zhang_cuda_many_series_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let lookback = 20usize;
    let mut open_tm = vec![0.0f32; cols * rows];
    let mut high_tm = vec![0.0f32; cols * rows];
    let mut low_tm = vec![0.0f32; cols * rows];
    let mut close_tm = vec![0.0f32; cols * rows];

    for s in 0..cols {
        let mut prev = 800.0f64 + (s as f64) * 20.0;
        for t in 0..rows {
            let x = t as f64 + (s as f64) * 0.25;
            let o = (prev + (x * 0.0031).sin() * 1.5 + 0.0001 * x).max(1.0);
            let c = (o + (x * 0.0023).cos() * 0.6).max(1.0);
            let h = o.max(c) + 0.25 + (x * 0.0017).cos().abs() * 0.04;
            let l = (o.min(c) - 0.25 - (x * 0.0019).sin().abs() * 0.04).max(0.01);
            let idx = t * cols + s;
            open_tm[idx] = o as f32;
            high_tm[idx] = h as f32;
            low_tm[idx] = l as f32;
            close_tm[idx] = c as f32;
            prev = c;
        }
    }

    let mut cpu_yz_tm = vec![f64::NAN; cols * rows];
    let mut cpu_rs_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut open = vec![0.0f64; rows];
        let mut high = vec![0.0f64; rows];
        let mut low = vec![0.0f64; rows];
        let mut close = vec![0.0f64; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            open[t] = open_tm[idx] as f64;
            high[t] = high_tm[idx] as f64;
            low[t] = low_tm[idx] as f64;
            close[t] = close_tm[idx] as f64;
        }
        let params = YangZhangVolatilityParams {
            lookback: Some(lookback),
            k_override: Some(false),
            k: Some(0.34),
        };
        let input = YangZhangVolatilityInput::from_slices(&open, &high, &low, &close, params);
        let out = yang_zhang_volatility_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let idx = t * cols + s;
            cpu_yz_tm[idx] = out.yz[t];
            cpu_rs_tm[idx] = out.rs[t];
        }
    }

    let cuda = CudaYangZhangVolatility::new(0)?;
    let dev = cuda.yang_zhang_volatility_many_series_one_param_time_major_dev(
        &open_tm, &high_tm, &low_tm, &close_tm, cols, rows, lookback, false, 0.34,
    )?;
    let mut yz_gpu = vec![0f32; dev.yz.len()];
    let mut rs_gpu = vec![0f32; dev.rs.len()];
    dev.yz.buf.copy_to(&mut yz_gpu)?;
    dev.rs.buf.copy_to(&mut rs_gpu)?;

    let tol = 2e-3;
    for i in 0..yz_gpu.len() {
        assert!(
            approx_eq(cpu_yz_tm[i], yz_gpu[i] as f64, tol),
            "yz mismatch at {}: cpu={} gpu={}",
            i,
            cpu_yz_tm[i],
            yz_gpu[i]
        );
        assert!(
            approx_eq(cpu_rs_tm[i], rs_gpu[i] as f64, tol),
            "rs mismatch at {}: cpu={} gpu={}",
            i,
            cpu_rs_tm[i],
            rs_gpu[i]
        );
    }

    Ok(())
}
