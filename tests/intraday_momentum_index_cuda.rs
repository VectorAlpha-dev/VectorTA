use vector_ta::indicators::intraday_momentum_index::{
    intraday_momentum_index_batch_with_kernel, IntradayMomentumIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaIntradayMomentumIndex};

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
fn intraday_momentum_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[intraday_momentum_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2080usize;
    let mut open = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 88.0f64;
    for i in 12..len {
        let x = i as f64;
        base += (x * 0.014).sin() * 0.42 + (x * 0.005).cos() * 0.21;
        open[i] = base + (x * 0.018).sin() * 0.54;
        close[i] = open[i] + (x * 0.061).sin() * 2.8 + (x * 0.023).cos() * 1.4;
    }
    for i in (500..580).step_by(8) {
        open[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1490..1560).step_by(10) {
        open[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = IntradayMomentumIndexBatchRange {
        length: (12, 14, 2),
        length_ma: (5, 7, 2),
        mult: (1.5, 2.0, 0.5),
        length_bb: (16, 18, 2),
        apply_smoothing: Some(false),
        low_band: (10, 10, 0),
    };
    let cpu =
        intraday_momentum_index_batch_with_kernel(&open, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaIntradayMomentumIndex::new(0).expect("CudaIntradayMomentumIndex::new");
    let result = cuda.batch_dev(&open, &close, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_imi = vec![0.0f64; result.outputs.imi.len()];
    let mut got_upper = vec![0.0f64; result.outputs.upper_hit.len()];
    let mut got_lower = vec![0.0f64; result.outputs.lower_hit.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    result.outputs.imi.buf.copy_to(&mut got_imi)?;
    result.outputs.upper_hit.buf.copy_to(&mut got_upper)?;
    result.outputs.lower_hit.buf.copy_to(&mut got_lower)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.imi.len() {
        assert!(
            approx_eq(cpu.imi[idx], got_imi[idx], 1e-6),
            "imi mismatch at {idx}: cpu={} cuda={}",
            cpu.imi[idx],
            got_imi[idx]
        );
        assert!(
            approx_eq(cpu.upper_hit[idx], got_upper[idx], 1e-6),
            "upper_hit mismatch at {idx}: cpu={} cuda={}",
            cpu.upper_hit[idx],
            got_upper[idx]
        );
        assert!(
            approx_eq(cpu.lower_hit[idx], got_lower[idx], 1e-6),
            "lower_hit mismatch at {idx}: cpu={} cuda={}",
            cpu.lower_hit[idx],
            got_lower[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-6),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
    }

    Ok(())
}
