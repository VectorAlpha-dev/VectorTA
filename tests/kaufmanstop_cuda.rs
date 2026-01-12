use vector_ta::indicators::kaufmanstop::{
    kaufmanstop_batch_with_kernel, KaufmanstopBatchBuilder, KaufmanstopBatchRange,
    KaufmanstopBuilder, KaufmanstopParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaKaufmanstop;

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
fn kaufmanstop_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[kaufmanstop_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        let v = (x * 0.00123).sin() * 0.5 + 0.00017 * x;
        let r = 0.5 + (x * 0.00037).cos().abs();
        high[i] = v + r * 0.5;
        low[i] = v - r * 0.5;
    }
    let sweep = KaufmanstopBatchRange {
        period: (10, 40, 5),
        mult: (2.0, 2.0, 0.0),
        direction: ("long".to_string(), "long".to_string(), 0.0),
        ma_type: ("sma".to_string(), "sma".to_string(), 0.0),
    };

    let cpu = kaufmanstop_batch_with_kernel(&high, &low, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cuda = CudaKaufmanstop::new(0).expect("CudaKaufmanstop::new");
    let (dev, _combos) = cuda
        .kaufmanstop_batch_dev(&high_f32, &low_f32, &sweep)
        .expect("cuda kaufmanstop_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut g = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut g)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let d = g[idx] as f64;
        assert!(
            approx_eq(c, d, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            d
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn kaufmanstop_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[kaufmanstop_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 6usize;
    let rows = 1024usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) * 0.5 + (s as f64) * 0.2;
            let r = 0.5 + (x * 0.00077).cos().abs();
            let v = (x * 0.002).sin() + 0.0003 * x;
            high_tm[t * cols + s] = v + r * 0.5;
            low_tm[t * cols + s] = v - r * 0.5;
        }
    }

    let params = KaufmanstopParams {
        period: Some(22),
        mult: Some(2.0),
        direction: Some("long".to_string()),
        ma_type: Some("sma".to_string()),
    };

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = high_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
        }
        let out = KaufmanstopBuilder::new()
            .period(params.period.unwrap())
            .mult(params.mult.unwrap())
            .direction(params.direction.as_deref().unwrap())
            .ma_type(params.ma_type.as_deref().unwrap())
            .apply_slices(&h, &l)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaKaufmanstop::new(0).expect("CudaKaufmanstop::new");
    let dev_tm = cuda
        .kaufmanstop_many_series_one_param_time_major_dev(
            &high_tm_f32,
            &low_tm_f32,
            cols,
            rows,
            &params,
        )
        .expect("kaufmanstop many-series");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);
    let mut g_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut g_tm)?;

    let tol = 5e-4;
    for idx in 0..g_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], g_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }

    Ok(())
}
