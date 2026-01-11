

use vector_ta::indicators::cvi::{
    cvi_batch_with_kernel, cvi_with_kernel, CviBatchRange, CviData, CviInput, CviParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaCvi;

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
fn cvi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cvi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 16384usize;
    let mut close = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        close[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let mut high = close.clone();
    let mut low = close.clone();
    for i in 0..len {
        let v = close[i];
        if v.is_nan() {
            continue;
        }
        let x = i as f64 * 0.002;
        let off = (0.004 * x.sin()).abs() + 0.12;
        high[i] = v + off;
        low[i] = v - off;
    }

    let sweep = CviBatchRange { period: (4, 64, 4) };
    let cpu = cvi_batch_with_kernel(&high, &low, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cuda = CudaCvi::new(0).expect("CudaCvi::new");
    let dev = cuda
        .cvi_batch_dev(&high_f32, &low_f32, &sweep)
        .expect("cvi_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-4;
    for idx in 0..host.len() {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cvi_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cvi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let (cols, rows) = (16usize, 4096usize);
    let mut base = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            base[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let mut high_tm = base.clone();
    let mut low_tm = base.clone();
    for s in 0..cols {
        for t in 0..rows {
            let v = base[t * cols + s];
            if v.is_nan() {
                continue;
            }
            let x = (t as f64) * 0.002;
            let off = (0.004 * x.cos()).abs() + 0.11;
            high_tm[t * cols + s] = v + off;
            low_tm[t * cols + s] = v - off;
        }
    }

    let period = 10usize;

    
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = high_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
        }
        let params = CviParams {
            period: Some(period),
        };
        let input = CviInput {
            data: CviData::Slices { high: &h, low: &l },
            params,
        };
        let out = cvi_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    
    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaCvi::new(0).expect("CudaCvi::new");
    let dev = cuda
        .cvi_many_series_one_param_time_major_dev(&high_tm_f32, &low_tm_f32, cols, rows, period)
        .expect("cvi_many_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 1e-4;
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu_tm[idx], host[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
