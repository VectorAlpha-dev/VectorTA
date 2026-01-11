

use vector_ta::indicators::medprice::{medprice_with_kernel, MedpriceInput, MedpriceParams};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaMedprice};

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
fn medprice_cuda_dev_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[medprice_cuda_dev_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        high[i] = (x * 0.00123).sin() + 100.0;
        low[i] = high[i] - ((x * 0.00011).cos().abs() + 0.25);
    }

    let input = MedpriceInput::from_slices(&high, &low, MedpriceParams::default());
    let cpu_out = medprice_with_kernel(&input, Kernel::Scalar)?.values;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cuda = CudaMedprice::new(0).expect("CudaMedprice::new");
    let dev = cuda
        .medprice_dev(&high_f32, &low_f32)
        .expect("medprice_dev");

    assert_eq!(dev.rows, 1);
    assert_eq!(dev.cols, len);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 1e-5;
    for i in 0..len {
        assert!(
            approx_eq(cpu_out[i], host[i] as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            i,
            cpu_out[i],
            host[i]
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn medprice_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[medprice_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 1024usize;
    let mut high_tm = vec![f32::NAN; cols * rows];
    let mut low_tm = vec![f32::NAN; cols * rows];
    
    for s in 0..cols {
        for t in (s + 2)..rows {
            let x = (t as f32) * 0.002 + (s as f32) * 0.01;
            let base = x.sin() + 100.0;
            high_tm[t * cols + s] = base + 0.2;
            low_tm[t * cols + s] = base - 0.2;
        }
    }
    
    let mut cpu = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            h[t] = high_tm[idx] as f64;
            l[t] = low_tm[idx] as f64;
        }
        let input = MedpriceInput::from_slices(&h, &l, MedpriceParams::default());
        let out = medprice_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            cpu[t * cols + s] = out[t];
        }
    }
    let cuda = CudaMedprice::new(0).expect("CudaMedprice::new");
    let dev = cuda
        .medprice_many_series_one_param_time_major_dev(&high_tm, &low_tm, cols, rows)
        .expect("many_series_one_param");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut got = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got)?;
    let tol = 1e-5;
    for i in 0..got.len() {
        if cpu[i].is_nan() {
            assert!(got[i].is_nan());
        } else {
            assert!((cpu[i] - got[i] as f64).abs() <= tol);
        }
    }
    Ok(())
}
