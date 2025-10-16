// CUDA integration tests for Momentum (MOM)

use my_project::utilities::enums::Kernel;
use my_project::indicators::mom::{mom_with_kernel, MomBatchRange, MomInput, MomParams};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::mom_wrapper::CudaMom;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { true } else { (a - b).abs() <= tol }
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn mom_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mom_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 16384usize;
    let mut data = vec![f64::NAN; len];
    for i in 8..len {
        let x = i as f64 * 0.00123;
        data[i] = (x).sin() + 0.0002 * x;
    }
    let sweep = MomBatchRange { period: (2, 64, 3) };

    // CPU
    let cpu = my_project::indicators::mom::mom_batch_slice(&data, &sweep, Kernel::ScalarBatch)?;

    // GPU
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaMom::new(0).expect("CudaMom::new");
    let dev = cuda.mom_batch_dev(&data_f32, &sweep).expect("mom_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    let mut got = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(approx_eq(cpu.values[idx], got[idx] as f64, tol), "mismatch at {}", idx);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn mom_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mom_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 11usize; // series
    let rows = 2048usize; // time
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows { // stagger first_valid per series
            let x = (t as f64) + (s as f64) * 0.2;
            tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let period = 14usize;

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows { series[t] = tm[t * cols + s]; }
        let params = MomParams { period: Some(period) };
        let input = MomInput::from_slice(&series, params);
        let out = mom_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows { cpu_tm[t * cols + s] = out.values[t]; }
    }

    // GPU
    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaMom::new(0).expect("CudaMom::new");
    let dev = cuda
        .mom_many_series_one_param_time_major_dev(&tm_f32, cols, rows, period)
        .expect("mom_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut got_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got_tm)?;

    let tol = 1e-4;
    for idx in 0..got_tm.len() {
        assert!(approx_eq(cpu_tm[idx], got_tm[idx] as f64, tol), "mismatch at {}", idx);
    }
    Ok(())
}

