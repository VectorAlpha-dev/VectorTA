use vector_ta::indicators::decisionpoint_breadth_swenlin_trading_oscillator::{
    decisionpoint_breadth_swenlin_trading_oscillator_batch_with_kernel,
    DecisionPointBreadthSwenlinTradingOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaDecisionPointBreadthSwenlinTradingOscillator};

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
fn decisionpoint_breadth_swenlin_trading_oscillator_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[decisionpoint_breadth_swenlin_trading_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 1536usize;
    let mut advancing = vec![f64::NAN; len];
    let mut declining = vec![f64::NAN; len];
    for i in 6..len {
        let x = i as f64;
        advancing[i] = 1500.0 + x * 0.8 + (x * 0.07).sin() * 120.0 + 40.0;
        declining[i] = 1300.0 + x * 0.5 + (x * 0.05).cos() * 95.0 + 30.0;
    }
    advancing[480] = 0.0;
    declining[480] = 0.0;
    advancing[960] = f64::NAN;
    declining[960] = f64::NAN;

    let sweep = DecisionPointBreadthSwenlinTradingOscillatorBatchRange;
    let cpu = decisionpoint_breadth_swenlin_trading_oscillator_batch_with_kernel(
        &advancing,
        &declining,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaDecisionPointBreadthSwenlinTradingOscillator::new(0)
        .expect("CudaDecisionPointBreadthSwenlinTradingOscillator::new");
    let result = cuda
        .batch_dev(&advancing, &declining, &sweep)
        .expect("batch_dev");

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
