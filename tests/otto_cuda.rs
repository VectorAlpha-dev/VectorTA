// Integration tests for CUDA OTTO kernels

use my_project::indicators::otto::{otto_with_kernel, OttoInput, OttoParams};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaOtto;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop_otto() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn otto_cuda_batch_matches_cpu_var_only() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[otto_cuda_batch_matches_cpu_var_only] skipped - no CUDA device");
        return Ok(());
    }
    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 1..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = my_project::indicators::otto::OttoBatchRange {
        ott_period: (2, 14, 3),
        ott_percent: (0.6, 0.6, 0.0),
        fast_vidya: (10, 10, 0),
        slow_vidya: (25, 25, 0),
        correcting_constant: (100000.0, 100000.0, 0.0),
        ma_types: vec!["VAR".into()],
    };

    // CPU baseline for each combo
    let combos = {
        let mut v = Vec::new();
        for p in (sweep.ott_period.0..=sweep.ott_period.1).step_by(sweep.ott_period.2) {
            v.push(OttoParams { ott_period: Some(p), ..Default::default() });
        }
        v
    };

    let mut cpu_hott = Vec::new();
    let mut cpu_lott = Vec::new();
    for params in &combos {
        let input = OttoInput::from_slice(&price, params.clone());
        let out = otto_with_kernel(&input, Kernel::Scalar)?;
        cpu_hott.extend_from_slice(&out.hott);
        cpu_lott.extend_from_slice(&out.lott);
    }

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaOtto::new(0).expect("CudaOtto::new");
    let (hott_dev, lott_dev, _combos) =
        cuda.otto_batch_dev(&price_f32, &sweep).expect("otto_batch_dev");

    assert_eq!(hott_dev.rows, combos.len());
    assert_eq!(hott_dev.cols, len);
    assert_eq!(lott_dev.rows, combos.len());
    assert_eq!(lott_dev.cols, len);

    let mut g_hott = vec![0f32; hott_dev.len()];
    let mut g_lott = vec![0f32; lott_dev.len()];
    hott_dev.buf.copy_to(&mut g_hott)?;
    lott_dev.buf.copy_to(&mut g_lott)?;

    let tol = 1e-2; // OTTO is sensitive; allow a modest tolerance
    for idx in 0..(len * combos.len()) {
        assert!(approx_eq(cpu_hott[idx], g_hott[idx] as f64, tol), "hott mismatch at {}", idx);
        assert!(approx_eq(cpu_lott[idx], g_lott[idx] as f64, tol), "lott mismatch at {}", idx);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn otto_cuda_many_series_one_param_matches_cpu_var_only() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[otto_cuda_many_series_one_param_matches_cpu_var_only] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 300usize; // satisfy warmup requirement slow*fast + 10
    let rows = 256usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..rows { for t in 0..cols { let x = (t as f64) + (s as f64) * 0.07; tm[t * rows + s] = (x * 0.002).sin() + 0.00021 * x; } }
    let params = OttoParams::default();

    // CPU baseline per series
    let mut cpu_hott_tm = vec![f64::NAN; cols * rows];
    let mut cpu_lott_tm = vec![f64::NAN; cols * rows];
    for s in 0..rows {
        let mut one = vec![f64::NAN; cols];
        for t in 0..cols { one[t] = tm[t * rows + s]; }
        let input = OttoInput::from_slice(&one, params.clone());
        let out = otto_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..cols {
            cpu_hott_tm[t * rows + s] = out.hott[t];
            cpu_lott_tm[t * rows + s] = out.lott[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaOtto::new(0).expect("CudaOtto::new");
    let (hott_tm, lott_tm) = cuda
        .otto_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("ms otto dev");
    assert_eq!(hott_tm.rows, rows); assert_eq!(hott_tm.cols, cols);

    let mut g_hott_tm = vec![0f32; hott_tm.len()];
    let mut g_lott_tm = vec![0f32; lott_tm.len()];
    hott_tm.buf.copy_to(&mut g_hott_tm)?;
    lott_tm.buf.copy_to(&mut g_lott_tm)?;
    let tol = 3e-3;
    for idx in 0..(cols * rows) {
        assert!(approx_eq(cpu_hott_tm[idx], g_hott_tm[idx] as f64, tol), "hott tm mismatch at {}", idx);
        assert!(approx_eq(cpu_lott_tm[idx], g_lott_tm[idx] as f64, tol), "lott tm mismatch at {}", idx);
    }
    Ok(())
}
