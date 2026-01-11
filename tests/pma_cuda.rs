

use vector_ta::indicators::pma::{pma, PmaInput, PmaParams};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaPma;

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
fn pma_cuda_one_series_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[pma_cuda_one_series_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let n = 4096usize;
    let mut data = vec![f64::NAN; n];
    for i in 0..n {
        if i >= 6 {
            let x = i as f64;
            data[i] = (x * 0.00123).sin() + 0.00017 * x;
        }
    }

    let input = PmaInput::from_slice(&data, PmaParams {});
    let cpu = pma(&input)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaPma::new(0).expect("CudaPma::new");
    let pair = cuda
        .pma_batch_dev(
            &data_f32,
            &vector_ta::indicators::pma::PmaBatchRange::default(),
        )
        .expect("pma_batch_dev");

    assert_eq!(pair.rows(), 1);
    assert_eq!(pair.cols(), n);

    let mut gpu_predict = vec![0f32; pair.predict.len()];
    let mut gpu_trigger = vec![0f32; pair.trigger.len()];
    pair.predict.buf.copy_to(&mut gpu_predict)?;
    pair.trigger.buf.copy_to(&mut gpu_trigger)?;

    
    let tol = 2e-4;
    for idx in 0..n {
        assert!(
            approx_eq(cpu.predict[idx], gpu_predict[idx] as f64, tol),
            "predict mismatch at {}: cpu={} gpu={}",
            idx,
            cpu.predict[idx],
            gpu_predict[idx]
        );
        assert!(
            approx_eq(cpu.trigger[idx], gpu_trigger[idx] as f64, tol),
            "trigger mismatch at {}: cpu={} gpu={}",
            idx,
            cpu.trigger[idx],
            gpu_trigger[idx]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn pma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[pma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 7usize; 
    let rows = 2048usize; 
    let mut data_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for r in s..rows {
            
            let x = r as f64 + (s as f64) * 0.37;
            data_tm[r * cols + s] = (x * 0.0016).cos() + (x * 0.0011).sin() * 0.4 + 0.0003 * x;
        }
    }

    let mut cpu_predict_tm = vec![f64::NAN; cols * rows];
    let mut cpu_trigger_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for r in 0..rows {
            series[r] = data_tm[r * cols + s];
        }
        let out = pma(&PmaInput::from_slice(&series, PmaParams {}))?;
        for r in 0..rows {
            let idx = r * cols + s;
            cpu_predict_tm[idx] = out.predict[r];
            cpu_trigger_tm[idx] = out.trigger[r];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaPma::new(0).expect("CudaPma::new");
    let pair = cuda
        .pma_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows)
        .expect("pma_many_series_one_param_time_major_dev");
    assert_eq!(pair.rows(), rows);
    assert_eq!(pair.cols(), cols);

    let mut gpu_predict_tm = vec![0f32; pair.predict.len()];
    let mut gpu_trigger_tm = vec![0f32; pair.trigger.len()];
    pair.predict.buf.copy_to(&mut gpu_predict_tm)?;
    pair.trigger.buf.copy_to(&mut gpu_trigger_tm)?;

    let tol = 2e-4;
    for idx in 0..(cols * rows) {
        assert!(
            approx_eq(cpu_predict_tm[idx], gpu_predict_tm[idx] as f64, tol),
            "predict mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_trigger_tm[idx], gpu_trigger_tm[idx] as f64, tol),
            "trigger mismatch at {}",
            idx
        );
    }
    Ok(())
}
