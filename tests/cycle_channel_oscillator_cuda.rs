use vector_ta::indicators::cycle_channel_oscillator::{
    cycle_channel_oscillator_batch_with_kernel, CycleChannelOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaCycleChannelOscillator};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_inputs(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut source = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut prev = 108.0f64;

    for i in 12..len {
        let x = i as f64;
        let o = prev + (x * 0.051).sin() * 0.44;
        let c = o + (x * 0.083).cos() * 0.76 - (x * 0.013).sin() * 0.22;
        let h = o.max(c) + 0.52 + (x * 0.017).sin().abs() * 0.21;
        let l = o.min(c) - 0.49 - (x * 0.019).cos().abs() * 0.19;
        source[i] = (h + l + c) / 3.0;
        high[i] = h;
        low[i] = l;
        close[i] = c;
        prev = c;
    }

    for i in (220..280).step_by(13) {
        source[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (780..860).step_by(17) {
        source[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    (source, high, low, close)
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn cycle_channel_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cycle_channel_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let (source, high, low, close) = sample_inputs(1408);
    let sweep = CycleChannelOscillatorBatchRange {
        short_cycle_length: (10, 14, 4),
        medium_cycle_length: (30, 34, 4),
        short_multiplier: (1.0, 1.0, 0.0),
        medium_multiplier: (2.0, 3.0, 1.0),
    };
    let cpu = cycle_channel_oscillator_batch_with_kernel(
        &source,
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaCycleChannelOscillator::new(0).expect("CudaCycleChannelOscillator::new");
    let result = cuda
        .batch_dev(&source, &high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_fast = vec![0.0f64; result.outputs.fast.len()];
    let mut got_slow = vec![0.0f64; result.outputs.slow.len()];
    result.outputs.fast.buf.copy_to(&mut got_fast)?;
    result.outputs.slow.buf.copy_to(&mut got_slow)?;

    for idx in 0..cpu.fast.len() {
        assert!(
            approx_eq(cpu.fast[idx], got_fast[idx], 1e-9),
            "fast mismatch at {idx}: cpu={} cuda={}",
            cpu.fast[idx],
            got_fast[idx]
        );
        assert!(
            approx_eq(cpu.slow[idx], got_slow[idx], 1e-9),
            "slow mismatch at {idx}: cpu={} cuda={}",
            cpu.slow[idx],
            got_slow[idx]
        );
    }

    Ok(())
}
