// Integration tests for CUDA MarketEFI kernels

use my_project::indicators::marketefi::{marketefi, MarketefiInput, MarketefiParams, MarketefiData};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, CudaMarketefi};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
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
fn marketefi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[marketefi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut vol = vec![f64::NAN; len];
    for i in 0..len {
        let x = i as f64;
        high[i] = (x * 0.00123).sin() + 1.0;
        low[i] = high[i] - 0.5_f64.abs();
        vol[i] = (x * 0.00077).cos().abs() + 0.5;
    }

    let input = MarketefiInput { data: MarketefiData::Slices { high: &high, low: &low, volume: &vol }, params: MarketefiParams };
    let cpu = marketefi(&input)?.values;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let vol_f32: Vec<f32> = vol.iter().map(|&v| v as f32).collect();
    let cuda = CudaMarketefi::new(0).expect("CudaMarketefi::new");
    let dev = cuda.marketefi_batch_dev(&high_f32, &low_f32, &vol_f32).expect("marketefi_batch_dev");

    assert_eq!(dev.rows, 1);
    assert_eq!(dev.cols, len);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 1e-5; // simple arithmetic; FP32 is fine here
    for i in 0..len {
        assert!(approx_eq(cpu[i], host[i] as f64, tol), "mismatch at {}: cpu={} gpu={}", i, cpu[i], host[i]);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn marketefi_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[marketefi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; // num_series
    let rows = 2048usize; // series_len
    let mut h_tm = vec![f64::NAN; cols * rows];
    let mut l_tm = vec![f64::NAN; cols * rows];
    let mut v_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in 0..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            h_tm[t * cols + s] = (x * 0.002).sin() + 1.0;
            l_tm[t * cols + s] = h_tm[t * cols + s] - 0.4_f64.abs();
            v_tm[t * cols + s] = (x * 0.001).cos().abs() + 0.4;
        }
    }

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for t in 0..rows { h[t] = h_tm[t * cols + s]; l[t] = l_tm[t * cols + s]; v[t] = v_tm[t * cols + s]; }
        let input = MarketefiInput { data: MarketefiData::Slices { high: &h, low: &l, volume: &v }, params: MarketefiParams };
        let out = marketefi(&input)?.values;
        for t in 0..rows { cpu_tm[t * cols + s] = out[t]; }
    }

    let h_tm_f32: Vec<f32> = h_tm.iter().map(|&v| v as f32).collect();
    let l_tm_f32: Vec<f32> = l_tm.iter().map(|&v| v as f32).collect();
    let v_tm_f32: Vec<f32> = v_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaMarketefi::new(0).expect("CudaMarketefi::new");
    let dev_tm = cuda
        .marketefi_many_series_one_param_time_major_dev(&h_tm_f32, &l_tm_f32, &v_tm_f32, cols, rows)
        .expect("marketefi_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);

    let mut host_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut host_tm)?;

    let tol = 1e-5;
    for idx in 0..host_tm.len() {
        assert!(approx_eq(cpu_tm[idx], host_tm[idx] as f64, tol), "mismatch at {}", idx);
    }
    Ok(())
}
