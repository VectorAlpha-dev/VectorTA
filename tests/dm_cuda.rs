use vector_ta::indicators::dm::{
    dm_batch_with_kernel, dm_with_kernel, DmBatchRange, DmInput, DmParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::dm_wrapper::CudaDm;

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
fn dm_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dm_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        let base = (x * 0.00123).sin() + 0.00017 * x;
        let off = (0.001 * x.cos()).abs() + 0.12;
        high[i] = base + off;
        low[i] = base - off;
    }
    let sweep = DmBatchRange { period: (4, 64, 8) };

    let cpu = dm_batch_with_kernel(&high, &low, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cuda = CudaDm::new(0).expect("CudaDm::new");
    let pair = cuda
        .dm_batch_dev(&high_f32, &low_f32, &sweep)
        .expect("dm_batch_dev")
        .0;

    assert_eq!(cpu.rows, pair.rows());
    assert_eq!(cpu.cols, pair.cols());

    let mut plus_host = vec![0f32; pair.plus.len()];
    let mut minus_host = vec![0f32; pair.minus.len()];
    pair.plus.buf.copy_to(&mut plus_host)?;
    pair.minus.buf.copy_to(&mut minus_host)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.plus[idx], plus_host[idx] as f64, tol),
            "plus mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.minus[idx], minus_host[idx] as f64, tol),
            "minus mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn dm_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dm_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 1024usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            let base = (x * 0.002).sin() + 0.0003 * x;
            let off = (0.001 * x.cos()).abs() + 0.15;
            high_tm[t * cols + s] = base + off;
            low_tm[t * cols + s] = base - off;
        }
    }
    let period = 14usize;

    let mut cpu_plus_tm = vec![f64::NAN; cols * rows];
    let mut cpu_minus_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = high_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
        }
        let params = DmParams {
            period: Some(period),
        };
        let input = DmInput::from_slices(&h, &l, params);
        let out = dm_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_plus_tm[t * cols + s] = out.plus[t];
            cpu_minus_tm[t * cols + s] = out.minus[t];
        }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaDm::new(0).expect("CudaDm::new");
    let pair_tm = cuda
        .dm_many_series_one_param_time_major_dev(&high_tm_f32, &low_tm_f32, cols, rows, period)
        .expect("dm_many_series_one_param_time_major_dev");

    assert_eq!(pair_tm.rows(), rows);
    assert_eq!(pair_tm.cols(), cols);

    let mut g_plus_tm = vec![0f32; pair_tm.plus.len()];
    let mut g_minus_tm = vec![0f32; pair_tm.minus.len()];
    pair_tm.plus.buf.copy_to(&mut g_plus_tm)?;
    pair_tm.minus.buf.copy_to(&mut g_minus_tm)?;

    let tol = 1e-4;
    for idx in 0..g_plus_tm.len() {
        assert!(
            approx_eq(cpu_plus_tm[idx], g_plus_tm[idx] as f64, tol),
            "plus mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_minus_tm[idx], g_minus_tm[idx] as f64, tol),
            "minus mismatch at {}",
            idx
        );
    }
    Ok(())
}
