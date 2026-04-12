use vector_ta::indicators::keltner_channel_width_oscillator::{
    keltner_channel_width_oscillator_batch_with_kernel, KeltnerChannelWidthOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaKeltnerChannelWidthOscillator};

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
fn keltner_channel_width_oscillator_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[keltner_channel_width_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2304usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut source = vec![f64::NAN; len];
    let mut base = 108.0f64;
    for i in 18..len {
        let x = i as f64;
        base += (x * 0.009).sin() * 0.34 + (x * 0.004).cos() * 0.16;
        close[i] = base + (x * 0.020).sin() * 0.64 + (x * 0.006).cos() * 0.21;
        high[i] = close[i] + 0.94 + (x * 0.013).sin().abs() * 0.23;
        low[i] = close[i] - 0.89 - (x * 0.012).cos().abs() * 0.25;
        source[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    let sweep = KeltnerChannelWidthOscillatorBatchRange {
        length: (12, 16, 4),
        multiplier: (1.5, 2.5, 1.0),
        atr_length: (6, 10, 4),
        use_exponential: Some(false),
        bands_style: Some("Average True Range".to_string()),
    };
    let cpu = keltner_channel_width_oscillator_batch_with_kernel(
        &high,
        &low,
        &close,
        &source,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda =
        CudaKeltnerChannelWidthOscillator::new(0).expect("CudaKeltnerChannelWidthOscillator::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &source, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_kbw = vec![0.0f64; result.outputs.kbw.len()];
    let mut got_kbw_sma = vec![0.0f64; result.outputs.kbw_sma.len()];
    result.outputs.kbw.buf.copy_to(&mut got_kbw)?;
    result.outputs.kbw_sma.buf.copy_to(&mut got_kbw_sma)?;

    for idx in 0..cpu.kbw.len() {
        assert!(
            approx_eq(cpu.kbw[idx], got_kbw[idx], 1e-6),
            "kbw mismatch at {idx}: cpu={} cuda={}",
            cpu.kbw[idx],
            got_kbw[idx]
        );
        assert!(
            approx_eq(cpu.kbw_sma[idx], got_kbw_sma[idx], 1e-6),
            "kbw_sma mismatch at {idx}: cpu={} cuda={}",
            cpu.kbw_sma[idx],
            got_kbw_sma[idx]
        );
    }

    Ok(())
}
