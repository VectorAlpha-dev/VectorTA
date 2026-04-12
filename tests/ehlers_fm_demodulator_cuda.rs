use vector_ta::indicators::ehlers_fm_demodulator::{
    ehlers_fm_demodulator_batch_with_kernel, EhlersFmDemodulatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaEhlersFmDemodulator};

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
fn ehlers_fm_demodulator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ehlers_fm_demodulator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1536usize;
    let mut open = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 90.0f64;
    for i in 4..len {
        base += (i as f64 * 0.011).sin() * 0.37 + (i as f64 * 0.004).cos() * 0.15;
        open[i] = base - 0.24 + (i as f64 * 0.005).sin() * 0.13;
        close[i] = base + 0.18 + (i as f64 * 0.009).cos() * 0.17;
    }
    open[512] = f64::NAN;
    close[512] = f64::NAN;
    open[1024] = f64::NAN;
    close[1024] = f64::NAN;

    let sweep = EhlersFmDemodulatorBatchRange {
        period: (12, 18, 3),
    };
    let cpu = ehlers_fm_demodulator_batch_with_kernel(&open, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaEhlersFmDemodulator::new(0).expect("CudaEhlersFmDemodulator::new");
    let result = cuda.batch_dev(&open, &close, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-9),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
