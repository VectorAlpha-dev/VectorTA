

use vector_ta::indicators::gatorosc::{
    gatorosc_batch_with_kernel, gatorosc_with_kernel, GatorOscBatchRange, GatorOscInput,
    GatorOscParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::gatorosc_wrapper::CudaGatorOsc;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn gatorosc_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[gatorosc_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut close = vec![f64::NAN; len];
    for i in 10..len {
        let x = i as f64;
        close[i] = (x * 0.00123).sin() * 3.0 + 0.00017 * x;
    }
    let sweep = GatorOscBatchRange {
        jaws_length: (10, 14, 2),
        jaws_shift: (4, 8, 2),
        teeth_length: (7, 9, 1),
        teeth_shift: (3, 5, 1),
        lips_length: (4, 6, 1),
        lips_shift: (1, 3, 1),
    };

    let cpu = gatorosc_batch_with_kernel(&close, &sweep, Kernel::ScalarBatch)?;

    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaGatorOsc::new(0).expect("CudaGatorOsc::new");
    let dev = cuda
        .gatorosc_batch_dev(&close_f32, &sweep)
        .expect("gatorosc_batch_dev");

    assert_eq!(cpu.rows, dev.upper.rows);
    assert_eq!(cpu.cols, dev.upper.cols);

    let mut u = vec![0f32; dev.upper.rows * dev.upper.cols];
    let mut l = vec![0f32; dev.lower.rows * dev.lower.cols];
    let mut uc = vec![0f32; dev.upper_change.rows * dev.upper_change.cols];
    let mut lc = vec![0f32; dev.lower_change.rows * dev.lower_change.cols];
    dev.upper.buf.copy_to(&mut u)?;
    dev.lower.buf.copy_to(&mut l)?;
    dev.upper_change.buf.copy_to(&mut uc)?;
    dev.lower_change.buf.copy_to(&mut lc)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.upper[idx], u[idx] as f64, tol),
            "upper mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.lower[idx], l[idx] as f64, tol),
            "lower mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.upper_change[idx], uc[idx] as f64, tol),
            "upper_change mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.lower_change[idx], lc[idx] as f64, tol),
            "lower_change mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn gatorosc_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[gatorosc_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; 
    let rows = 2048usize; 
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.37;
            close_tm[t * cols + s] = (x * 0.002).cos() * 1.7 + 0.0003 * x;
        }
    }

    let (jl, js, tl, ts, ll, ls) = (13usize, 8usize, 8usize, 5usize, 5usize, 3usize);

    
    let mut cpu_u = vec![f64::NAN; cols * rows];
    let mut cpu_l = vec![f64::NAN; cols * rows];
    let mut cpu_uc = vec![f64::NAN; cols * rows];
    let mut cpu_lc = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = close_tm[t * cols + s];
        }
        let params = GatorOscParams {
            jaws_length: Some(jl),
            jaws_shift: Some(js),
            teeth_length: Some(tl),
            teeth_shift: Some(ts),
            lips_length: Some(ll),
            lips_shift: Some(ls),
        };
        let input = GatorOscInput::from_slice(&series, params);
        let out = gatorosc_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_u[t * cols + s] = out.upper[t];
            cpu_l[t * cols + s] = out.lower[t];
            cpu_uc[t * cols + s] = out.upper_change[t];
            cpu_lc[t * cols + s] = out.lower_change[t];
        }
    }

    
    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaGatorOsc::new(0).expect("CudaGatorOsc::new");
    let dev = cuda
        .gatorosc_many_series_one_param_time_major_dev(
            &close_tm_f32,
            cols,
            rows,
            jl,
            js,
            tl,
            ts,
            ll,
            ls,
        )
        .expect("gatorosc_many_series_one_param_time_major_dev");
    assert_eq!(dev.upper.rows, rows);
    assert_eq!(dev.upper.cols, cols);
    let mut u = vec![0f32; dev.upper.len()];
    let mut l = vec![0f32; dev.lower.len()];
    let mut uc = vec![0f32; dev.upper_change.len()];
    let mut lc = vec![0f32; dev.lower_change.len()];
    dev.upper.buf.copy_to(&mut u)?;
    dev.lower.buf.copy_to(&mut l)?;
    dev.upper_change.buf.copy_to(&mut uc)?;
    dev.lower_change.buf.copy_to(&mut lc)?;

    let tol = 1e-3;
    for idx in 0..u.len() {
        assert!(
            approx_eq(cpu_u[idx], u[idx] as f64, tol),
            "upper mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_l[idx], l[idx] as f64, tol),
            "lower mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_uc[idx], uc[idx] as f64, tol),
            "upper_change mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_lc[idx], lc[idx] as f64, tol),
            "lower_change mismatch at {}",
            idx
        );
    }
    Ok(())
}
