use vector_ta::indicators::vi::{
    vi_batch_with_kernel, vi_with_kernel, ViBatchRange, ViInput, ViParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::vi_wrapper::CudaVi;

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
fn vi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 32_768usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        high[i] = (x * 0.00123).sin() + 0.0007 * x;
        low[i] = high[i] - 0.6;
        close[i] = (0.5 * x).cos() + 0.0009 * x;
    }
    let sweep = ViBatchRange { period: (7, 33, 2) };

    let cpu = vi_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;

    let h32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let l32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let c32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaVi::new(0).expect("CudaVi::new");
    let (pair, combos) = cuda
        .vi_batch_dev(&h32, &l32, &c32, &sweep)
        .expect("vi_batch_dev");

    assert_eq!(cpu.rows, combos.len());
    assert_eq!(cpu.rows, pair.rows());
    assert_eq!(cpu.cols, pair.cols());

    let mut plus_g = vec![0f32; pair.a.len()];
    let mut minus_g = vec![0f32; pair.b.len()];
    pair.a.buf.copy_to(&mut plus_g)?;
    pair.b.buf.copy_to(&mut minus_g)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.plus[idx], plus_g[idx] as f64, tol),
            "plus mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.minus[idx], minus_g[idx] as f64, tol),
            "minus mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn vi_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 16usize;
    let rows = 8192usize;
    let mut high_tm = vec![f64::NAN; rows * cols];
    let mut low_tm = vec![f64::NAN; rows * cols];
    let mut close_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        for r in s..rows {
            let idx = r * cols + s;
            let x = (r as f64) * 0.002 + (s as f64) * 0.01;
            high_tm[idx] = x.sin() + 0.01 * x;
            low_tm[idx] = high_tm[idx] - 0.5;
            close_tm[idx] = (0.5 * x).cos() + 0.02 * x;
        }
    }
    let period = 14usize;

    let mut plus_cpu_tm = vec![f64::NAN; rows * cols];
    let mut minus_cpu_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for r in 0..rows {
            let idx = r * cols + s;
            h[r] = high_tm[idx];
            l[r] = low_tm[idx];
            c[r] = close_tm[idx];
        }
        let params = ViParams {
            period: Some(period),
        };
        let input = ViInput::from_slices(&h, &l, &c, params);
        let out = vi_with_kernel(&input, Kernel::Scalar)?;
        for r in 0..rows {
            let idx = r * cols + s;
            plus_cpu_tm[idx] = out.plus[r];
            minus_cpu_tm[idx] = out.minus[r];
        }
    }

    let h32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let l32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let c32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaVi::new(0).expect("CudaVi::new");
    let pair = cuda
        .vi_many_series_one_param_time_major_dev(
            &h32,
            &l32,
            &c32,
            cols,
            rows,
            &ViParams {
                period: Some(period),
            },
        )
        .expect("vi_many_series_one_param_time_major_dev");

    assert_eq!(pair.rows(), rows);
    assert_eq!(pair.cols(), cols);

    let mut plus_g = vec![0f32; rows * cols];
    let mut minus_g = vec![0f32; rows * cols];
    pair.a.buf.copy_to(&mut plus_g)?;
    pair.b.buf.copy_to(&mut minus_g)?;

    let tol = 5e-4;
    for idx in 0..(rows * cols) {
        assert!(
            approx_eq(plus_cpu_tm[idx], plus_g[idx] as f64, tol),
            "plus mismatch at {}",
            idx
        );
        assert!(
            approx_eq(minus_cpu_tm[idx], minus_g[idx] as f64, tol),
            "minus mismatch at {}",
            idx
        );
    }
    Ok(())
}
