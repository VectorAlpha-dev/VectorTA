// CUDA integration tests for Awesome Oscillator (AO)

use my_project::indicators::ao::{
    ao_batch_with_kernel, ao_with_kernel, AoBatchRange, AoInput, AoParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::ao_wrapper::CudaAo;

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
fn ao_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ao_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut hl2 = vec![f64::NAN; len];
    for i in 10..len {
        let x = i as f64;
        hl2[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = AoBatchRange {
        short_period: (4, 28, 4),
        long_period: (32, 128, 16),
    };

    let cpu = ao_batch_with_kernel(&hl2, &sweep, Kernel::ScalarBatch)?;

    let hl2_f32: Vec<f32> = hl2.iter().map(|&v| v as f32).collect();
    let cuda = CudaAo::new(0).expect("CudaAo::new");
    let dev = cuda.ao_batch_dev(&hl2_f32, &sweep).expect("ao_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    let mut got = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.values[idx], got[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ao_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ao_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; // series
    let rows = 1024usize; // time
    let mut hl2_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            hl2_tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }

    let short = 5usize;
    let long = 34usize;

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = hl2_tm[t * cols + s];
        }
        let params = AoParams {
            short_period: Some(short),
            long_period: Some(long),
        };
        let input = AoInput::from_slice(&series, params);
        let out = ao_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    // GPU
    let hl2_tm_f32: Vec<f32> = hl2_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaAo::new(0).expect("CudaAo::new");
    let dev = cuda
        .ao_many_series_one_param_time_major_dev(&hl2_tm_f32, cols, rows, short, long)
        .expect("ao_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut got_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got_tm)?;

    let tol = 1e-4;
    for idx in 0..got_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], got_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
