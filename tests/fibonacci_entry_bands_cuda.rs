use vector_ta::indicators::fibonacci_entry_bands::{
    fibonacci_entry_bands_batch_with_kernel, FibonacciEntryBandsBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaFibonacciEntryBands};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let close: Vec<f64> = (0..len)
        .map(|i| {
            let x = i as f64;
            101.0 + x * 0.06 + (x * 0.019).sin() * 1.1 + (x * 0.011).cos() * 0.42
        })
        .collect();
    let open: Vec<f64> = close
        .iter()
        .enumerate()
        .map(|(i, &c)| c - ((i as f64) * 0.027).cos() * 0.55)
        .collect();
    let high: Vec<f64> = open
        .iter()
        .zip(close.iter())
        .enumerate()
        .map(|(i, (&o, &c))| o.max(c) + 0.32 + ((i as f64) * 0.013).sin().abs() * 0.23)
        .collect();
    let low: Vec<f64> = open
        .iter()
        .zip(close.iter())
        .enumerate()
        .map(|(i, (&o, &c))| o.min(c) - 0.31 - ((i as f64) * 0.017).cos().abs() * 0.21)
        .collect();
    (open, high, low, close)
}

#[cfg(feature = "cuda")]
fn run_case(sweep: FibonacciEntryBandsBatchRange) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[fibonacci_entry_bands_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let (mut open, mut high, mut low, mut close) = sample_ohlc(420);
    open[211] = f64::NAN;
    high[211] = f64::NAN;
    low[211] = f64::NAN;
    close[211] = f64::NAN;

    let cpu = fibonacci_entry_bands_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaFibonacciEntryBands::new(0)?;
    let result = cuda.batch_dev(&open, &high, &low, &close, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_basis = vec![0.0f64; result.outputs.basis.len()];
    let mut got_trend = vec![0.0f64; result.outputs.trend.len()];
    let mut got_upper_0618 = vec![0.0f64; result.outputs.upper_0618.len()];
    let mut got_upper_1000 = vec![0.0f64; result.outputs.upper_1000.len()];
    let mut got_upper_1618 = vec![0.0f64; result.outputs.upper_1618.len()];
    let mut got_upper_2618 = vec![0.0f64; result.outputs.upper_2618.len()];
    let mut got_lower_0618 = vec![0.0f64; result.outputs.lower_0618.len()];
    let mut got_lower_1000 = vec![0.0f64; result.outputs.lower_1000.len()];
    let mut got_lower_1618 = vec![0.0f64; result.outputs.lower_1618.len()];
    let mut got_lower_2618 = vec![0.0f64; result.outputs.lower_2618.len()];
    let mut got_tp_long_band = vec![0.0f64; result.outputs.tp_long_band.len()];
    let mut got_tp_short_band = vec![0.0f64; result.outputs.tp_short_band.len()];
    let mut got_long_entry = vec![0.0f64; result.outputs.long_entry.len()];
    let mut got_short_entry = vec![0.0f64; result.outputs.short_entry.len()];
    let mut got_rejection_long = vec![0.0f64; result.outputs.rejection_long.len()];
    let mut got_rejection_short = vec![0.0f64; result.outputs.rejection_short.len()];
    let mut got_long_bounce = vec![0.0f64; result.outputs.long_bounce.len()];
    let mut got_short_bounce = vec![0.0f64; result.outputs.short_bounce.len()];
    result.outputs.basis.buf.copy_to(&mut got_basis)?;
    result.outputs.trend.buf.copy_to(&mut got_trend)?;
    result.outputs.upper_0618.buf.copy_to(&mut got_upper_0618)?;
    result.outputs.upper_1000.buf.copy_to(&mut got_upper_1000)?;
    result.outputs.upper_1618.buf.copy_to(&mut got_upper_1618)?;
    result.outputs.upper_2618.buf.copy_to(&mut got_upper_2618)?;
    result.outputs.lower_0618.buf.copy_to(&mut got_lower_0618)?;
    result.outputs.lower_1000.buf.copy_to(&mut got_lower_1000)?;
    result.outputs.lower_1618.buf.copy_to(&mut got_lower_1618)?;
    result.outputs.lower_2618.buf.copy_to(&mut got_lower_2618)?;
    result
        .outputs
        .tp_long_band
        .buf
        .copy_to(&mut got_tp_long_band)?;
    result
        .outputs
        .tp_short_band
        .buf
        .copy_to(&mut got_tp_short_band)?;
    result.outputs.long_entry.buf.copy_to(&mut got_long_entry)?;
    result
        .outputs
        .short_entry
        .buf
        .copy_to(&mut got_short_entry)?;
    result
        .outputs
        .rejection_long
        .buf
        .copy_to(&mut got_rejection_long)?;
    result
        .outputs
        .rejection_short
        .buf
        .copy_to(&mut got_rejection_short)?;
    result
        .outputs
        .long_bounce
        .buf
        .copy_to(&mut got_long_bounce)?;
    result
        .outputs
        .short_bounce
        .buf
        .copy_to(&mut got_short_bounce)?;

    for idx in 0..cpu.basis.len() {
        assert!(approx_eq(cpu.basis[idx], got_basis[idx], 1e-6));
        assert!(approx_eq(cpu.trend[idx], got_trend[idx], 1e-6));
        assert!(approx_eq(cpu.upper_0618[idx], got_upper_0618[idx], 1e-6));
        assert!(approx_eq(cpu.upper_1000[idx], got_upper_1000[idx], 1e-6));
        assert!(approx_eq(cpu.upper_1618[idx], got_upper_1618[idx], 1e-6));
        assert!(approx_eq(cpu.upper_2618[idx], got_upper_2618[idx], 1e-6));
        assert!(approx_eq(cpu.lower_0618[idx], got_lower_0618[idx], 1e-6));
        assert!(approx_eq(cpu.lower_1000[idx], got_lower_1000[idx], 1e-6));
        assert!(approx_eq(cpu.lower_1618[idx], got_lower_1618[idx], 1e-6));
        assert!(approx_eq(cpu.lower_2618[idx], got_lower_2618[idx], 1e-6));
        assert!(approx_eq(
            cpu.tp_long_band[idx],
            got_tp_long_band[idx],
            1e-6
        ));
        assert!(approx_eq(
            cpu.tp_short_band[idx],
            got_tp_short_band[idx],
            1e-6
        ));
        assert!(approx_eq(cpu.long_entry[idx], got_long_entry[idx], 1e-6));
        assert!(approx_eq(cpu.short_entry[idx], got_short_entry[idx], 1e-6));
        assert!(approx_eq(
            cpu.rejection_long[idx],
            got_rejection_long[idx],
            1e-6
        ));
        assert!(approx_eq(
            cpu.rejection_short[idx],
            got_rejection_short[idx],
            1e-6
        ));
        assert!(approx_eq(cpu.long_bounce[idx], got_long_bounce[idx], 1e-6));
        assert!(approx_eq(
            cpu.short_bounce[idx],
            got_short_bounce[idx],
            1e-6
        ));
    }

    Ok(())
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn fibonacci_entry_bands_cuda_atr_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    run_case(FibonacciEntryBandsBatchRange {
        length: (21, 25, 4),
        atr_length: (14, 18, 4),
        source: "hlc3".to_string(),
        use_atr: true,
        tp_aggressiveness: "medium".to_string(),
    })
}

#[cfg(feature = "cuda")]
#[test]
fn fibonacci_entry_bands_cuda_stdev_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    run_case(FibonacciEntryBandsBatchRange {
        length: (18, 22, 4),
        atr_length: (14, 14, 0),
        source: "ohlc4".to_string(),
        use_atr: false,
        tp_aggressiveness: "high".to_string(),
    })
}
