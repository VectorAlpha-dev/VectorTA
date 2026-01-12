use vector_ta::indicators::aso::{aso_batch_with_kernel, AsoBatchRange, AsoParams};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::CudaAso;

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
fn aso_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[aso_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1024usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        let base = (x * 0.0021).sin() + 0.0002 * x;
        open[i] = base - 0.05;
        high[i] = base + 0.12;
        low[i] = base - 0.11;
        close[i] = base + 0.02;
    }

    let sweep = AsoBatchRange {
        period: (5, 17, 4),
        mode: (0, 2, 1),
    };

    let cpu = aso_batch_with_kernel(&open, &high, &low, &close, &sweep, Kernel::ScalarBatch)?;

    let of: Vec<f32> = open.iter().map(|&v| v as f32).collect();
    let hf: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaAso::new(0).expect("CudaAso::new");
    let (dev_bulls, dev_bears) = cuda
        .aso_batch_dev(&of, &hf, &lf, &cf, &sweep)
        .expect("aso_batch_dev");

    assert_eq!(dev_bulls.rows, cpu.rows);
    assert_eq!(dev_bulls.cols, cpu.cols);
    assert_eq!(dev_bears.rows, cpu.rows);
    assert_eq!(dev_bears.cols, cpu.cols);

    let mut gb = vec![0f32; dev_bulls.len()];
    let mut ge = vec![0f32; dev_bears.len()];
    dev_bulls.buf.copy_to(&mut gb)?;
    dev_bears.buf.copy_to(&mut ge)?;

    let tol = 7e-3;
    for i in 0..gb.len() {
        assert!(
            approx_eq(cpu.bulls[i], gb[i] as f64, tol),
            "bulls mismatch at {}",
            i
        );
        assert!(
            approx_eq(cpu.bears[i], ge[i] as f64, tol),
            "bears mismatch at {}",
            i
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn aso_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[aso_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 1024usize;
    let fast_p = 10usize;
    let mode = 0usize;
    let mut o_tm = vec![f64::NAN; cols * rows];
    let mut h_tm = vec![f64::NAN; cols * rows];
    let mut l_tm = vec![f64::NAN; cols * rows];
    let mut c_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let idx = t * cols + s;
            let x = (t as f64) + (s as f64) * 0.3;
            let base = (x * 0.0025).sin() + 0.00015 * x;
            o_tm[idx] = base - 0.03;
            h_tm[idx] = base + 0.09;
            l_tm[idx] = base - 0.08;
            c_tm[idx] = base + 0.015;
        }
    }

    let mut cpu_b = vec![f64::NAN; cols * rows];
    let mut cpu_e = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut o = vec![0.0; rows];
        let mut h = vec![0.0; rows];
        let mut l = vec![0.0; rows];
        let mut c = vec![0.0; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            o[t] = o_tm[idx];
            h[t] = h_tm[idx];
            l[t] = l_tm[idx];
            c[t] = c_tm[idx];
        }
        let params = AsoParams {
            period: Some(fast_p),
            mode: Some(mode),
        };
        let input = vector_ta::indicators::aso::AsoInput::from_slices(&o, &h, &l, &c, params);
        let out = vector_ta::indicators::aso::aso_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let idx = t * cols + s;
            cpu_b[idx] = out.bulls[t];
            cpu_e[idx] = out.bears[t];
        }
    }
    let o32: Vec<f32> = o_tm.iter().map(|&v| v as f32).collect();
    let h32: Vec<f32> = h_tm.iter().map(|&v| v as f32).collect();
    let l32: Vec<f32> = l_tm.iter().map(|&v| v as f32).collect();
    let c32: Vec<f32> = c_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaAso::new(0).expect("CudaAso::new");
    let (db, de) = cuda
        .aso_many_series_one_param_time_major_dev(&o32, &h32, &l32, &c32, cols, rows, fast_p, mode)
        .expect("aso_many_series");
    assert_eq!(db.rows, rows);
    assert_eq!(db.cols, cols);
    let mut gb = vec![0f32; db.len()];
    db.buf.copy_to(&mut gb)?;
    let mut ge = vec![0f32; de.len()];
    de.buf.copy_to(&mut ge)?;
    let tol = 7e-3;
    for i in 0..gb.len() {
        assert!(
            approx_eq(cpu_b[i], gb[i] as f64, tol),
            "bulls mismatch at {}",
            i
        );
        assert!(
            approx_eq(cpu_e[i], ge[i] as f64, tol),
            "bears mismatch at {}",
            i
        );
    }
    Ok(())
}
