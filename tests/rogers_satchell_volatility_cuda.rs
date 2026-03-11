use vector_ta::indicators::rogers_satchell_volatility::{
    rogers_satchell_volatility_batch_with_kernel, rogers_satchell_volatility_with_kernel,
    RogersSatchellVolatilityBatchRange, RogersSatchellVolatilityInput,
    RogersSatchellVolatilityParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaRogersSatchellVolatility;

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
fn rogers_satchell_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rogers_satchell_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut open = vec![0.0f64; len];
    let mut high = vec![0.0f64; len];
    let mut low = vec![0.0f64; len];
    let mut close = vec![0.0f64; len];
    let mut prev = 900.0f64;
    for i in 0..len {
        let x = i as f64;
        let o = (prev + 0.00015 * x + (x * 0.0014).sin() * 1.8 + (x * 0.00029).cos()).max(1.0);
        let c = (o + (x * 0.0011).sin() * 0.7).max(1.0);
        let h = o.max(c) + 0.35 + (x * 0.00087).cos().abs() * 0.05;
        let l = (o.min(c) - 0.35 - (x * 0.00107).sin().abs() * 0.05).max(0.01);
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

    let sweep = RogersSatchellVolatilityBatchRange {
        lookback: (10, 30, 10),
        signal_length: (4, 8, 4),
    };

    let cpu = rogers_satchell_volatility_batch_with_kernel(
        &open_q,
        &high_q,
        &low_q,
        &close_q,
        &sweep,
        Kernel::ScalarBatch,
    )?;

    let cuda = CudaRogersSatchellVolatility::new(0)?;
    let gpu_res = cuda
        .rogers_satchell_volatility_batch_dev(&open_f32, &high_f32, &low_f32, &close_f32, &sweep)?;

    assert_eq!(gpu_res.outputs.rs.rows, cpu.rows);
    assert_eq!(gpu_res.outputs.rs.cols, cpu.cols);
    assert_eq!(gpu_res.outputs.signal.rows, cpu.rows);
    assert_eq!(gpu_res.outputs.signal.cols, cpu.cols);
    assert_eq!(gpu_res.combos.len(), cpu.combos.len());

    let mut rs_gpu = vec![0f32; gpu_res.outputs.rs.len()];
    let mut signal_gpu = vec![0f32; gpu_res.outputs.signal.len()];
    gpu_res.outputs.rs.buf.copy_to(&mut rs_gpu)?;
    gpu_res.outputs.signal.buf.copy_to(&mut signal_gpu)?;

    let tol = 2e-3;
    for i in 0..cpu.rs.len() {
        assert!(
            approx_eq(cpu.rs[i], rs_gpu[i] as f64, tol),
            "rs mismatch at {}: cpu={} gpu={}",
            i,
            cpu.rs[i],
            rs_gpu[i]
        );
        assert!(
            approx_eq(cpu.signal[i], signal_gpu[i] as f64, tol),
            "signal mismatch at {}: cpu={} gpu={}",
            i,
            cpu.signal[i],
            signal_gpu[i]
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn rogers_satchell_cuda_many_series_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rogers_satchell_cuda_many_series_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let lookback = 20usize;
    let signal_length = 6usize;
    let mut open_tm = vec![0.0f32; cols * rows];
    let mut high_tm = vec![0.0f32; cols * rows];
    let mut low_tm = vec![0.0f32; cols * rows];
    let mut close_tm = vec![0.0f32; cols * rows];

    for s in 0..cols {
        let mut prev = 700.0f64 + (s as f64) * 25.0;
        for t in 0..rows {
            let x = t as f64 + (s as f64) * 0.31;
            let o = (prev + (x * 0.0029).sin() * 1.4 + 0.00012 * x).max(1.0);
            let c = (o + (x * 0.0021).cos() * 0.55).max(1.0);
            let h = o.max(c) + 0.22 + (x * 0.0015).cos().abs() * 0.04;
            let l = (o.min(c) - 0.22 - (x * 0.0017).sin().abs() * 0.04).max(0.01);
            let idx = t * cols + s;
            open_tm[idx] = o as f32;
            high_tm[idx] = h as f32;
            low_tm[idx] = l as f32;
            close_tm[idx] = c as f32;
            prev = c;
        }
    }

    let mut cpu_rs_tm = vec![f64::NAN; cols * rows];
    let mut cpu_signal_tm = vec![f64::NAN; cols * rows];
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
        let params = RogersSatchellVolatilityParams {
            lookback: Some(lookback),
            signal_length: Some(signal_length),
        };
        let input = RogersSatchellVolatilityInput::from_slices(&open, &high, &low, &close, params);
        let out = rogers_satchell_volatility_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let idx = t * cols + s;
            cpu_rs_tm[idx] = out.rs[t];
            cpu_signal_tm[idx] = out.signal[t];
        }
    }

    let cuda = CudaRogersSatchellVolatility::new(0)?;
    let dev = cuda.rogers_satchell_volatility_many_series_one_param_time_major_dev(
        &open_tm,
        &high_tm,
        &low_tm,
        &close_tm,
        cols,
        rows,
        lookback,
        signal_length,
    )?;
    let mut rs_gpu = vec![0f32; dev.rs.len()];
    let mut signal_gpu = vec![0f32; dev.signal.len()];
    dev.rs.buf.copy_to(&mut rs_gpu)?;
    dev.signal.buf.copy_to(&mut signal_gpu)?;

    let tol = 2e-3;
    for i in 0..rs_gpu.len() {
        assert!(
            approx_eq(cpu_rs_tm[i], rs_gpu[i] as f64, tol),
            "rs mismatch at {}: cpu={} gpu={}",
            i,
            cpu_rs_tm[i],
            rs_gpu[i]
        );
        assert!(
            approx_eq(cpu_signal_tm[i], signal_gpu[i] as f64, tol),
            "signal mismatch at {}: cpu={} gpu={}",
            i,
            cpu_signal_tm[i],
            signal_gpu[i]
        );
    }

    Ok(())
}
