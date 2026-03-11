use vector_ta::indicators::garman_klass_volatility::{
    garman_klass_volatility_batch_with_kernel, garman_klass_volatility_with_kernel,
    GarmanKlassVolatilityBatchRange, GarmanKlassVolatilityInput, GarmanKlassVolatilityParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaGarmanKlassVolatility;

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
fn garman_klass_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[garman_klass_cuda_batch_matches_cpu] skipped - no CUDA device");
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

    let sweep = GarmanKlassVolatilityBatchRange {
        lookback: (10, 30, 10),
    };

    let cpu = garman_klass_volatility_batch_with_kernel(
        &open_q,
        &high_q,
        &low_q,
        &close_q,
        &sweep,
        Kernel::ScalarBatch,
    )?;

    let cuda = CudaGarmanKlassVolatility::new(0)?;
    let gpu_res =
        cuda.garman_klass_volatility_batch_dev(&open_f32, &high_f32, &low_f32, &close_f32, &sweep)?;

    assert_eq!(gpu_res.outputs.rows, cpu.rows);
    assert_eq!(gpu_res.outputs.cols, cpu.cols);
    assert_eq!(gpu_res.combos.len(), cpu.combos.len());

    let mut gpu = vec![0f32; gpu_res.outputs.len()];
    gpu_res.outputs.buf.copy_to(&mut gpu)?;

    let tol = 2e-3;
    for i in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[i], gpu[i] as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            i,
            cpu.values[i],
            gpu[i]
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn garman_klass_cuda_many_series_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[garman_klass_cuda_many_series_matches_cpu] skipped - no CUDA device");
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

    let mut cpu_tm = vec![f64::NAN; cols * rows];
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
        let params = GarmanKlassVolatilityParams {
            lookback: Some(lookback),
        };
        let input = GarmanKlassVolatilityInput::from_slices(&open, &high, &low, &close, params);
        let out = garman_klass_volatility_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let idx = t * cols + s;
            cpu_tm[idx] = out.values[t];
        }
    }

    let cuda = CudaGarmanKlassVolatility::new(0)?;
    let dev = cuda.garman_klass_volatility_many_series_one_param_time_major_dev(
        &open_tm, &high_tm, &low_tm, &close_tm, cols, rows, lookback,
    )?;
    let mut gpu = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu)?;

    let tol = 2e-3;
    for i in 0..gpu.len() {
        assert!(
            approx_eq(cpu_tm[i], gpu[i] as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            i,
            cpu_tm[i],
            gpu[i]
        );
    }

    Ok(())
}
