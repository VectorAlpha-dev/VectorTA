use vector_ta::indicators::minmax::{
    minmax_batch_with_kernel, minmax_with_kernel, MinmaxBatchRange, MinmaxInput, MinmaxParams,
};

use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::minmax_wrapper::CudaMinmax;

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
fn minmax_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[minmax_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        let base = (x * 0.0013).sin() + 0.00011 * x;
        let spread = (x * 0.00073).cos().abs() * 0.2 + 0.2;
        low[i] = base;
        high[i] = base * (1.0 + spread);
    }
    let sweep = MinmaxBatchRange { order: (3, 31, 4) };

    let cpu = minmax_batch_with_kernel(&high, &low, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cuda = CudaMinmax::new(0).expect("CudaMinmax::new");
    let (quad_dev, combos) = cuda
        .minmax_batch_dev(&high_f32, &low_f32, &sweep)
        .expect("cuda minmax_batch_dev");

    assert_eq!(cpu.rows, combos.len());
    assert_eq!(cpu.cols, len);
    assert_eq!(quad_dev.rows, combos.len());
    assert_eq!(quad_dev.cols, len);

    let mut is_min_host = vec![0f32; combos.len() * len];
    let mut is_max_host = vec![0f32; combos.len() * len];
    let mut last_min_host = vec![0f32; combos.len() * len];
    let mut last_max_host = vec![0f32; combos.len() * len];
    quad_dev.is_min.copy_to(&mut is_min_host)?;
    quad_dev.is_max.copy_to(&mut is_max_host)?;
    quad_dev.last_min.copy_to(&mut last_min_host)?;
    quad_dev.last_max.copy_to(&mut last_max_host)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c_min = cpu.is_min[idx];
        let c_max = cpu.is_max[idx];
        let c_lmin = cpu.last_min[idx];
        let c_lmax = cpu.last_max[idx];
        let g_min = is_min_host[idx] as f64;
        let g_max = is_max_host[idx] as f64;
        let g_lmin = last_min_host[idx] as f64;
        let g_lmax = last_max_host[idx] as f64;
        assert!(
            approx_eq(c_min, g_min, tol),
            "is_min mismatch at {}: cpu={} gpu={}",
            idx,
            c_min,
            g_min
        );
        assert!(
            approx_eq(c_max, g_max, tol),
            "is_max mismatch at {}: cpu={} gpu={}",
            idx,
            c_max,
            g_max
        );
        assert!(
            approx_eq(c_lmin, g_lmin, tol),
            "last_min mismatch at {}: cpu={} gpu={}",
            idx,
            c_lmin,
            g_lmin
        );
        assert!(
            approx_eq(c_lmax, g_lmax, tol),
            "last_max mismatch at {}: cpu={} gpu={}",
            idx,
            c_lmax,
            g_lmax
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn minmax_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[minmax_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            let base = (x * 0.002).sin() + 0.0003 * x;
            let spread = (x * 0.001).cos().abs() * 0.15 + 0.15;
            let idx = t * cols + s;
            low_tm[idx] = base;
            high_tm[idx] = base * (1.0 + spread);
        }
    }
    let order = 9usize;

    let mut cpu_is_min = vec![f64::NAN; cols * rows];
    let mut cpu_is_max = vec![f64::NAN; cols * rows];
    let mut cpu_last_min = vec![f64::NAN; cols * rows];
    let mut cpu_last_max = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            h[t] = high_tm[idx];
            l[t] = low_tm[idx];
        }
        let params = MinmaxParams { order: Some(order) };
        let input = MinmaxInput::from_slices(&h, &l, params);
        let out = minmax_with_kernel(&input, Kernel::Scalar).unwrap();
        for t in 0..rows {
            let idx = t * cols + s;
            cpu_is_min[idx] = out.is_min[t];
            cpu_is_max[idx] = out.is_max[t];
            cpu_last_min[idx] = out.last_min[t];
            cpu_last_max[idx] = out.last_max[t];
        }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaMinmax::new(0).expect("CudaMinmax::new");
    let quad_tm = cuda
        .minmax_many_series_one_param_time_major_dev(
            &high_tm_f32,
            &low_tm_f32,
            cols,
            rows,
            &MinmaxParams { order: Some(order) },
        )
        .expect("minmax many_series device");

    assert_eq!(quad_tm.rows, rows);
    assert_eq!(quad_tm.cols, cols);

    let mut g_is_min = vec![0f32; cols * rows];
    let mut g_is_max = vec![0f32; cols * rows];
    let mut g_last_min = vec![0f32; cols * rows];
    let mut g_last_max = vec![0f32; cols * rows];
    quad_tm.is_min.copy_to(&mut g_is_min)?;
    quad_tm.is_max.copy_to(&mut g_is_max)?;
    quad_tm.last_min.copy_to(&mut g_last_min)?;
    quad_tm.last_max.copy_to(&mut g_last_max)?;

    let tol = 1e-4;
    for idx in 0..g_is_min.len() {
        assert!(
            approx_eq(cpu_is_min[idx], g_is_min[idx] as f64, tol),
            "is_min mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_is_max[idx], g_is_max[idx] as f64, tol),
            "is_max mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_last_min[idx], g_last_min[idx] as f64, tol),
            "last_min mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_last_max[idx], g_last_max[idx] as f64, tol),
            "last_max mismatch at {}",
            idx
        );
    }

    Ok(())
}
