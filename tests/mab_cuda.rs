// Integration tests for CUDA MAB kernels

use my_project::indicators::mab::{mab, mab_batch, MabBatchRange, MabInput, MabParams};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaMab;

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
fn mab_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mab_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 60..len { let x = i as f64; price[i] = (x*0.00123).sin() + 0.00017*x; }
    let sweep = MabBatchRange{
        fast_period: (10, 22, 4),
        slow_period: (50, 74, 12),
        devup: (1.0, 1.0, 0.0),
        devdn: (1.0, 1.0, 0.0),
        fast_ma_type: ("sma".to_string(), "sma".to_string(), String::new()),
        slow_ma_type: ("sma".to_string(), "sma".to_string(), String::new()),
    };
    let cpu = mab_batch(&price, &sweep)?; // CPU reference

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaMab::new(0).expect("CudaMab::new");
    let (trip, combos) = cuda.mab_batch_dev(&price_f32, &sweep).expect("cuda mab batch");

    assert_eq!(combos.len(), cpu.rows);
    assert_eq!(trip.rows(), cpu.rows);
    assert_eq!(trip.cols(), cpu.cols);

    let mut up_g = vec![0f32; trip.upper.len()];
    let mut mid_g = vec![0f32; trip.middle.len()];
    let mut lo_g = vec![0f32; trip.lower.len()];
    trip.upper.buf.copy_to(&mut up_g)?;
    trip.middle.buf.copy_to(&mut mid_g)?;
    trip.lower.buf.copy_to(&mut lo_g)?;

    let tol = 1e-3;
    for i in 0..(cpu.rows * cpu.cols) {
        assert!(approx_eq(cpu.upperbands[i], up_g[i] as f64, tol), "upper mismatch at {}", i);
        assert!(approx_eq(cpu.middlebands[i], mid_g[i] as f64, tol), "middle mismatch at {}", i);
        assert!(approx_eq(cpu.lowerbands[i], lo_g[i] as f64, tol), "lower mismatch at {}", i);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn mab_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mab_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize; // series
    let rows = 1024usize; // length
    let mut tm = vec![f64::NAN; cols*rows];
    for s in 0..cols { for t in s..rows { let x = (t as f64) + 0.15*(s as f64); tm[t*cols + s] = (x*0.002).sin() + 0.0003*x; }}
    let params = MabParams{ fast_period: Some(10), slow_period: Some(50), devup: Some(1.0), devdn: Some(1.0), fast_ma_type: Some("sma".into()), slow_ma_type: Some("sma".into()) };

    // CPU per series
    let mut up_cpu = vec![f64::NAN; cols*rows];
    let mut mid_cpu = vec![f64::NAN; cols*rows];
    let mut lo_cpu = vec![f64::NAN; cols*rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows { series[t] = tm[t*cols + s]; }
        let out = mab(&MabInput::from_slice(&series, params.clone()))?;
        for t in 0..rows { let idx = t*cols + s; up_cpu[idx] = out.upperband[t]; mid_cpu[idx] = out.middleband[t]; lo_cpu[idx] = out.lowerband[t]; }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaMab::new(0).expect("CudaMab::new");
    let trip = cuda
        .mab_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("mab many-series");

    assert_eq!(trip.rows(), rows);
    assert_eq!(trip.cols(), cols);

    let mut up_g = vec![0f32; trip.upper.len()];
    let mut mid_g = vec![0f32; trip.middle.len()];
    let mut lo_g = vec![0f32; trip.lower.len()];
    trip.upper.buf.copy_to(&mut up_g)?;
    trip.middle.buf.copy_to(&mut mid_g)?;
    trip.lower.buf.copy_to(&mut lo_g)?;

    let tol = 1e-3;
    for i in 0..up_g.len() {
        assert!(approx_eq(up_cpu[i], up_g[i] as f64, tol), "upper mismatch at {}", i);
        assert!(approx_eq(mid_cpu[i], mid_g[i] as f64, tol), "middle mismatch at {}", i);
        assert!(approx_eq(lo_cpu[i], lo_g[i] as f64, tol), "lower mismatch at {}", i);
    }
    Ok(())
}

