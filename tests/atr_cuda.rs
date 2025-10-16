use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaAtr;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    (a - b).abs() <= tol
}

fn synth_hlc_from_close(close: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut high = close.to_vec();
    let mut low = close.to_vec();
    for i in 0..close.len() {
        let v = close[i];
        if v.is_nan() { continue; }
        let x = i as f64 * 0.002;
        let off = (0.004 * x.sin()).abs() + 0.12;
        high[i] = v + off;
        low[i] = v - off;
    }
    (high, low)
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
fn atr_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[atr_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut close = vec![f64::NAN; len];
    for i in 4..len { let x = i as f64; close[i] = (x * 0.00123).sin() + 0.00017 * x; }
    let (high, low) = synth_hlc_from_close(&close);

    let sweep = my_project::indicators::atr::AtrBatchRange { length: (4, 64, 5) };
    let cpu = my_project::indicators::atr::atr_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaAtr::new(0).expect("CudaAtr::new");
    let dev = cuda.atr_batch_dev(&high_f32, &low_f32, &close_f32, &sweep).expect("atr_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 1e-4;
    for idx in 0..host.len() {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(approx_eq(c, g, tol), "mismatch at {}: cpu={} gpu={}", idx, c, g);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn atr_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[atr_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; // series
    let rows = 2048usize; // length
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.25;
            close_tm[t * cols + s] = (x * 0.0027).sin() + 0.00011 * x;
        }
    }
    let (mut high_tm, mut low_tm) = (close_tm.clone(), close_tm.clone());
    for s in 0..cols {
        for t in 0..rows {
            let v = close_tm[t * cols + s];
            if v.is_nan() { continue; }
            let x = (t as f64) * 0.002;
            let off = (0.0035 * x.cos()).abs() + 0.1;
            high_tm[t * cols + s] = v + off;
            low_tm[t * cols + s] = v - off;
        }
    }

    let period = 14usize;
    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows { h[t] = high_tm[t * cols + s]; l[t] = low_tm[t * cols + s]; c[t] = close_tm[t * cols + s]; }
        let out = my_project::indicators::atr::AtrBuilder::new().length(period).apply_slices(&h, &l, &c)?;
        for t in 0..rows { cpu_tm[t * cols + s] = out.values[t]; }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaAtr::new(0).expect("CudaAtr::new");
    let dev = cuda
        .atr_many_series_one_param_time_major_dev(&high_tm_f32, &low_tm_f32, &close_tm_f32, cols, rows, period)
        .expect("atr many-series dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_tm)?;

    let tol = 1e-4;
    for idx in 0..gpu_tm.len() {
        let c = cpu_tm[idx];
        let g = gpu_tm[idx] as f64;
        assert!(approx_eq(c, g, tol), "many-series mismatch at {}: cpu={} gpu={}", idx, c, g);
    }
    Ok(())
}

