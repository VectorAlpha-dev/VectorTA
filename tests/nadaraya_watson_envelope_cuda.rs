

use vector_ta::indicators::nadaraya_watson_envelope::{
    nadaraya_watson_envelope_batch_with_kernel, nadaraya_watson_envelope_with_kernel,
    NweBatchRange, NweInput, NweParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaNwe};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_ok() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn nwe_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[nwe_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 12..series_len {
        let x = i as f64;
        data[i] = (x * 0.0013).sin() + 0.001 * (x * 0.0009).cos();
    }

    let sweep = NweBatchRange {
        bandwidth: (6.0, 12.0, 2.0),
        multiplier: (2.0, 3.0, 0.5),
        lookback: (128, 256, 64),
    };
    let cpu = nadaraya_watson_envelope_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaNwe::new(0).expect("CudaNwe::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let (pair, combos) = cuda
        .nwe_batch_dev(&data_f32, &sweep)
        .expect("cuda nwe_batch_dev");
    assert_eq!(combos.len(), cpu.rows);
    assert_eq!(pair.rows(), cpu.rows);
    assert_eq!(pair.cols(), cpu.cols);

    let mut upper_gpu = vec![0f32; pair.upper.len()];
    let mut lower_gpu = vec![0f32; pair.lower.len()];
    pair.upper.buf.copy_to(&mut upper_gpu).expect("copy upper");
    pair.lower.buf.copy_to(&mut lower_gpu).expect("copy lower");

    let tol = 2e-3; 
    for idx in 0..(cpu.rows * cpu.cols) {
        let a = cpu.values_upper[idx];
        let b = upper_gpu[idx] as f64;
        assert!(
            approx_eq(a, b, tol) || (a.is_nan() && b.is_nan()),
            "upper mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }
    for idx in 0..(cpu.rows * cpu.cols) {
        let a = cpu.values_lower[idx];
        let b = lower_gpu[idx] as f64;
        assert!(
            approx_eq(a, b, tol) || (a.is_nan() && b.is_nan()),
            "lower mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn nwe_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[nwe_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 6usize; 
    let rows = 2048usize; 
    let mut data_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s + 12)..rows {
            let base = (t as f64) * 0.002 + (s as f64) * 0.05;
            data_tm[t * cols + s] = base.sin() + 0.0007 * base.cos();
        }
    }

    let params = NweParams {
        bandwidth: Some(8.0),
        multiplier: Some(3.0),
        lookback: Some(200),
    };

    
    let mut upper_cpu = vec![f64::NAN; cols * rows];
    let mut lower_cpu = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = data_tm[t * cols + s];
        }
        let input = NweInput::from_slice(&series, params.clone());
        if let Ok(out) = nadaraya_watson_envelope_with_kernel(&input, Kernel::Scalar) {
            for t in 0..rows {
                upper_cpu[t * cols + s] = out.upper[t];
                lower_cpu[t * cols + s] = out.lower[t];
            }
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaNwe::new(0).expect("CudaNwe::new");
    let pair = cuda
        .nwe_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("cuda many-series");
    assert_eq!(pair.rows(), rows);
    assert_eq!(pair.cols(), cols);
    let mut up_gpu = vec![0f32; pair.upper.len()];
    let mut lo_gpu = vec![0f32; pair.lower.len()];
    pair.upper.buf.copy_to(&mut up_gpu).expect("copy up");
    pair.lower.buf.copy_to(&mut lo_gpu).expect("copy lo");

    let tol = 3e-3;
    for i in 0..(cols * rows) {
        let a = upper_cpu[i];
        let b = up_gpu[i] as f64;
        assert!(
            approx_eq(a, b, tol) || (a.is_nan() && b.is_nan()),
            "upper mismatch at {}: cpu={} gpu={}",
            i,
            a,
            b
        );
    }
    for i in 0..(cols * rows) {
        let a = lower_cpu[i];
        let b = lo_gpu[i] as f64;
        assert!(
            approx_eq(a, b, tol) || (a.is_nan() && b.is_nan()),
            "lower mismatch at {}: cpu={} gpu={}",
            i,
            a,
            b
        );
    }
    Ok(())
}
