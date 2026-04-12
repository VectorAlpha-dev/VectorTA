use vector_ta::indicators::relative_strength_index_wave_indicator::{
    relative_strength_index_wave_indicator_batch_with_kernel,
    RelativeStrengthIndexWaveIndicatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaRelativeStrengthIndexWaveIndicator};

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
fn relative_strength_index_wave_indicator_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[relative_strength_index_wave_indicator_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2112usize;
    let mut source = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut base = 95.0f64;
    for i in 10..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.31 + (x * 0.004).cos() * 0.17;
        source[i] = base + (x * 0.019).sin() * 0.63 + (x * 0.006).cos() * 0.24;
        high[i] = source[i] + 0.71 + (x * 0.013).sin().abs() * 0.18;
        low[i] = source[i] - 0.69 - (x * 0.012).cos().abs() * 0.16;
    }
    for i in (430..520).step_by(11) {
        source[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
    }
    for i in (1330..1410).step_by(13) {
        source[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
    }

    let sweep = RelativeStrengthIndexWaveIndicatorBatchRange {
        rsi_length: (10, 12, 2),
        length1: (2, 2, 0),
        length2: (4, 6, 2),
        length3: (8, 8, 0),
        length4: (12, 14, 2),
    };
    let cpu = relative_strength_index_wave_indicator_batch_with_kernel(
        &source,
        &high,
        &low,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaRelativeStrengthIndexWaveIndicator::new(0)
        .expect("CudaRelativeStrengthIndexWaveIndicator::new");
    let result = cuda
        .batch_dev(&source, &high, &low, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_ma1 = vec![0.0f64; result.outputs.rsi_ma1.len()];
    let mut got_ma2 = vec![0.0f64; result.outputs.rsi_ma2.len()];
    let mut got_ma3 = vec![0.0f64; result.outputs.rsi_ma3.len()];
    let mut got_ma4 = vec![0.0f64; result.outputs.rsi_ma4.len()];
    let mut got_state = vec![0.0f64; result.outputs.state.len()];
    result.outputs.rsi_ma1.buf.copy_to(&mut got_ma1)?;
    result.outputs.rsi_ma2.buf.copy_to(&mut got_ma2)?;
    result.outputs.rsi_ma3.buf.copy_to(&mut got_ma3)?;
    result.outputs.rsi_ma4.buf.copy_to(&mut got_ma4)?;
    result.outputs.state.buf.copy_to(&mut got_state)?;

    for idx in 0..cpu.rsi_ma1.len() {
        assert!(
            approx_eq(cpu.rsi_ma1[idx], got_ma1[idx], 1e-9),
            "rsi_ma1 mismatch at {idx}: cpu={} cuda={}",
            cpu.rsi_ma1[idx],
            got_ma1[idx]
        );
        assert!(
            approx_eq(cpu.rsi_ma2[idx], got_ma2[idx], 1e-9),
            "rsi_ma2 mismatch at {idx}: cpu={} cuda={}",
            cpu.rsi_ma2[idx],
            got_ma2[idx]
        );
        assert!(
            approx_eq(cpu.rsi_ma3[idx], got_ma3[idx], 1e-9),
            "rsi_ma3 mismatch at {idx}: cpu={} cuda={}",
            cpu.rsi_ma3[idx],
            got_ma3[idx]
        );
        assert!(
            approx_eq(cpu.rsi_ma4[idx], got_ma4[idx], 1e-9),
            "rsi_ma4 mismatch at {idx}: cpu={} cuda={}",
            cpu.rsi_ma4[idx],
            got_ma4[idx]
        );
        assert!(
            approx_eq(cpu.state[idx], got_state[idx], 1e-12),
            "state mismatch at {idx}: cpu={} cuda={}",
            cpu.state[idx],
            got_state[idx]
        );
    }

    Ok(())
}
