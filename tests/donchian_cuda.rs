// Integration tests for CUDA Donchian kernels (upper, middle, lower)

use my_project::indicators::donchian::{
    donchian_batch_with_kernel, donchian_with_kernel, DonchianBatchRange, DonchianData,
    DonchianInput, DonchianParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaDonchian;

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
fn donchian_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[donchian_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 16384usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 16..len {
        let x = i as f64;
        high[i] = (x * 0.0023).sin() + 0.0002 * x;
        low[i]  = (x * 0.0017).cos() + 0.0001 * x;
    }
    let sweep = DonchianBatchRange { period: (5, 48, 1) };

    // Downcast to f32 for fairness, then compute CPU baseline on quantized data
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32>  = low.iter().map(|&v| v as f32).collect();
    let high_q: Vec<f64> = high_f32.iter().map(|&v| v as f64).collect();
    let low_q: Vec<f64>  = low_f32.iter().map(|&v| v as f64).collect();
    let cpu = donchian_batch_with_kernel(&high_q, &low_q, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaDonchian::new(0).expect("CudaDonchian::new");
    let (dev, _combos) = cuda
        .donchian_batch_dev(&high_f32, &low_f32, &sweep)
        .expect("donchian_cuda_batch_dev");

    assert_eq!(cpu.rows, dev.rows());
    assert_eq!(cpu.cols, dev.cols());

    let mut g_u = vec![0f32; dev.wt1.len()];
    let mut g_m = vec![0f32; dev.wt2.len()];
    let mut g_l = vec![0f32; dev.hist.len()];
    dev.wt1.buf.copy_to(&mut g_u)?;
    dev.wt2.buf.copy_to(&mut g_m)?;
    dev.hist.buf.copy_to(&mut g_l)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(approx_eq(cpu.upper[idx], g_u[idx] as f64, tol), "upper mismatch at {}", idx);
        assert!(approx_eq(cpu.middle[idx], g_m[idx] as f64, tol), "middle mismatch at {}", idx);
        assert!(approx_eq(cpu.lower[idx], g_l[idx] as f64, tol), "lower mismatch at {}", idx);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn donchian_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[donchian_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let cols = 6usize;
    let rows = 4096usize;
    let mut high_tm = vec![f64::NAN; rows * cols];
    let mut low_tm  = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.25;
            high_tm[t * cols + s] = (x * 0.0021).sin() + 0.0002 * x;
            low_tm[t * cols + s]  = (x * 0.0017).cos() + 0.0001 * x;
        }
    }
    let period = 21usize;

    // CPU baseline per series
    let mut up_tm = vec![f64::NAN; rows * cols];
    let mut mid_tm = vec![f64::NAN; rows * cols];
    let mut lo_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        for t in 0..rows { h[t] = high_tm[t * cols + s]; l[t] = low_tm[t * cols + s]; }
        let params = DonchianParams { period: Some(period) };
        let input = DonchianInput { data: DonchianData::Slices { high: &h, low: &l }, params };
        let out = donchian_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            up_tm[t * cols + s] = out.upperband[t];
            mid_tm[t * cols + s] = out.middleband[t];
            lo_tm[t * cols + s] = out.lowerband[t];
        }
    }

    // GPU
    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32>  = low_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaDonchian::new(0).expect("CudaDonchian::new");
    let triplet = cuda
        .donchian_many_series_one_param_time_major_dev(&high_tm_f32, &low_tm_f32, cols, rows, &DonchianParams { period: Some(period) })
        .expect("donchian many-series");

    assert_eq!(triplet.rows(), rows);
    assert_eq!(triplet.cols(), cols);
    let mut g_u = vec![0f32; rows * cols];
    let mut g_m = vec![0f32; rows * cols];
    let mut g_l = vec![0f32; rows * cols];
    triplet.wt1.buf.copy_to(&mut g_u)?;
    triplet.wt2.buf.copy_to(&mut g_m)?;
    triplet.hist.buf.copy_to(&mut g_l)?;

    let tol = 5e-4;
    for i in 0..(rows * cols) {
        assert!(approx_eq(up_tm[i], g_u[i] as f64, tol), "upper mismatch at {}", i);
        assert!(approx_eq(mid_tm[i], g_m[i] as f64, tol), "middle mismatch at {}", i);
        assert!(approx_eq(lo_tm[i], g_l[i] as f64, tol), "lower mismatch at {}", i);
    }
    Ok(())
}

