// CUDA integration tests for Pivot indicator

use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;

#[cfg(feature = "cuda")]
#[test]
fn pivot_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[pivot_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    // Synthesize OHLC with initial NaNs to exercise warmup
    let len = 16384usize;
    let mut h = vec![f64::NAN; len];
    let mut l = vec![f64::NAN; len];
    let mut c = vec![f64::NAN; len];
    let mut o = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64 * 0.0015;
        let base = (x * 0.9).sin() + 0.001 * x;
        let range = 0.2 + 0.03 * (x * 0.37).cos().abs();
        c[i] = base;
        o[i] = base + 0.01 * (x * 0.23).sin();
        l[i] = base - range;
        h[i] = base + range;
    }

    let sweep = my_project::indicators::pivot::PivotBatchRange { mode: (0, 4, 1) };

    let cpu = my_project::indicators::pivot::pivot_batch_flat_with_kernel(
        &h,
        &l,
        &c,
        &o,
        &sweep,
        Kernel::ScalarBatch,
    )?;

    let hf: Vec<f32> = h.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = l.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = c.iter().map(|&v| v as f32).collect();
    let of: Vec<f32> = o.iter().map(|&v| v as f32).collect();

    let cuda = my_project::cuda::pivot_wrapper::CudaPivot::new(0).expect("CudaPivot::new");
    let (dev, combos) = cuda
        .pivot_batch_dev(&hf, &lf, &cf, &of, &sweep)
        .expect("pivot batch dev");

    assert_eq!(combos.len(), cpu.combos.len());
    assert_eq!(dev.rows, 9 * combos.len());
    assert_eq!(dev.cols, cpu.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-4;
    for idx in 0..host.len() {
        let g = host[idx] as f64;
        let cval = cpu.values[idx];
        if cval.is_nan() && g.is_nan() {
            continue;
        }
        assert!(
            (g - cval).abs() <= tol,
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cval,
            g
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn pivot_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[pivot_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 32usize;
    let rows = 4096usize;
    let mut h_tm = vec![f64::NAN; cols * rows];
    let mut l_tm = vec![f64::NAN; cols * rows];
    let mut c_tm = vec![f64::NAN; cols * rows];
    let mut o_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in 3..rows {
            let idx = t * cols + s;
            let x = (t as f64) * 0.002 + (s as f64) * 0.01;
            let base = (x * 0.61).sin() + 0.002 * x;
            let range = 0.15 + 0.02 * (x * 0.17).cos().abs();
            c_tm[idx] = base;
            o_tm[idx] = base + 0.01 * (x * 0.13).sin();
            l_tm[idx] = base - range;
            h_tm[idx] = base + range;
        }
    }

    // CPU baseline for mode 3 (Camarilla)
    let mode = 3usize;
    let mut expected = vec![f64::NAN; 9 * cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        let mut o = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            h[t] = h_tm[idx];
            l[t] = l_tm[idx];
            c[t] = c_tm[idx];
            o[t] = o_tm[idx];
        }
        let params = my_project::indicators::pivot::PivotParams { mode: Some(mode) };
        let input = my_project::indicators::pivot::PivotInput::from_slices(&h, &l, &c, &o, params);
        let out = my_project::indicators::pivot::pivot_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            expected[(0 * rows + t) * cols + s] = out.r4[t];
            expected[(1 * rows + t) * cols + s] = out.r3[t];
            expected[(2 * rows + t) * cols + s] = out.r2[t];
            expected[(3 * rows + t) * cols + s] = out.r1[t];
            expected[(4 * rows + t) * cols + s] = out.pp[t];
            expected[(5 * rows + t) * cols + s] = out.s1[t];
            expected[(6 * rows + t) * cols + s] = out.s2[t];
            expected[(7 * rows + t) * cols + s] = out.s3[t];
            expected[(8 * rows + t) * cols + s] = out.s4[t];
        }
    }

    let hf: Vec<f32> = h_tm.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = l_tm.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = c_tm.iter().map(|&v| v as f32).collect();
    let of: Vec<f32> = o_tm.iter().map(|&v| v as f32).collect();

    let cuda = my_project::cuda::pivot_wrapper::CudaPivot::new(0).expect("CudaPivot::new");
    let dev = cuda
        .pivot_many_series_one_param_time_major_dev(&hf, &lf, &cf, &of, cols, rows, mode)
        .expect("pivot many-series dev");
    assert_eq!(dev.rows, 9 * rows);
    assert_eq!(dev.cols, cols);

    let mut g = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut g)?;
    let tol = 5e-4;
    for idx in 0..g.len() {
        let gg = g[idx] as f64;
        let ee = expected[idx];
        if ee.is_nan() && gg.is_nan() {
            continue;
        }
        assert!((gg - ee).abs() <= tol, "mismatch at {}", idx);
    }
    Ok(())
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}
