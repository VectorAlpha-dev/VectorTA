

use vector_ta::indicators::stochf::{
    stochf_batch_with_kernel, StochfBatchRange, StochfInput, StochfParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaStochf;

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
fn stochf_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[stochf_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    
    let mut close = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        let c = (x * 0.00123).sin() + 0.00017 * x;
        let off = (0.09 + 0.01 * (x * 0.0007).cos().abs());
        close[i] = c;
        high[i] = c + off;
        low[i] = c - off;
    }

    let sweep = StochfBatchRange {
        fastk_period: (5, 11, 2),
        fastd_period: (3, 3, 0),
    };
    let cpu = stochf_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaStochf::new(0).expect("CudaStochf::new");
    let (pair, _combos) = cuda
        .stochf_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
        .expect("stochf_batch_dev");

    assert_eq!(cpu.rows, pair.rows());
    assert_eq!(cpu.cols, pair.cols());

    let mut k_host = vec![0f32; pair.a.len()];
    pair.a.buf.copy_to(&mut k_host)?;
    let mut d_host = vec![0f32; pair.b.len()];
    pair.b.buf.copy_to(&mut d_host)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let ck = cpu.k[idx];
        let cd = cpu.d[idx];
        let gk = k_host[idx] as f64;
        let gd = d_host[idx] as f64;
        assert!(
            approx_eq(ck, gk, tol),
            "K mismatch at {}: cpu={} gpu={}",
            idx,
            ck,
            gk
        );
        assert!(
            approx_eq(cd, gd, tol),
            "D mismatch at {}: cpu={} gpu={}",
            idx,
            cd,
            gd
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn stochf_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[stochf_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 1024usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.19;
            let c = (x * 0.0021).sin() + 0.00031 * x;
            let off = (0.07 + 0.01 * (x * 0.0013).cos().abs());
            close_tm[t * cols + s] = c;
            high_tm[t * cols + s] = c + off;
            low_tm[t * cols + s] = c - off;
        }
    }

    let fk = 9usize;
    let fd = 3usize;
    let mt = 0usize;

    
    let mut cpu_k_tm = vec![f64::NAN; cols * rows];
    let mut cpu_d_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            h[t] = high_tm[idx];
            l[t] = low_tm[idx];
            c[t] = close_tm[idx];
        }
        let input = StochfInput::from_slices(
            &h,
            &l,
            &c,
            StochfParams {
                fastk_period: Some(fk),
                fastd_period: Some(fd),
                fastd_matype: Some(mt),
            },
        );
        let out = vector_ta::indicators::stochf::stochf(&input)?;
        for t in 0..rows {
            let idx = t * cols + s;
            cpu_k_tm[idx] = out.k[t];
            cpu_d_tm[idx] = out.d[t];
        }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaStochf::new(0).expect("CudaStochf::new");
    let params = StochfParams {
        fastk_period: Some(fk),
        fastd_period: Some(fd),
        fastd_matype: Some(mt),
    };
    let (k_dev_tm, d_dev_tm) = cuda
        .stochf_many_series_one_param_time_major_dev(
            &high_tm_f32,
            &low_tm_f32,
            &close_tm_f32,
            cols,
            rows,
            &params,
        )
        .expect("stochf_many_series_one_param_time_major_dev");

    assert_eq!(k_dev_tm.rows, rows);
    assert_eq!(k_dev_tm.cols, cols);
    assert_eq!(d_dev_tm.rows, rows);
    assert_eq!(d_dev_tm.cols, cols);

    let mut gk_tm = vec![0f32; k_dev_tm.len()];
    let mut gd_tm = vec![0f32; d_dev_tm.len()];
    k_dev_tm.buf.copy_to(&mut gk_tm)?;
    d_dev_tm.buf.copy_to(&mut gd_tm)?;

    let tol = 1e-3; 
    for idx in 0..gk_tm.len() {
        assert!(
            approx_eq(cpu_k_tm[idx], gk_tm[idx] as f64, tol),
            "K mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_d_tm[idx], gd_tm[idx] as f64, tol),
            "D mismatch at {}",
            idx
        );
    }

    Ok(())
}
