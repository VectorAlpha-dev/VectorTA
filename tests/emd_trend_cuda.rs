use vector_ta::indicators::emd_trend::{emd_trend_batch_with_kernel, EmdTrendBatchRange};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaEmdTrend};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 104.0f64;
    for i in 16..len {
        let x = i as f64;
        base += (x * 0.007).sin() * 0.24 + (x * 0.0015).cos() * 0.09;
        let c = base + (x * 0.023).sin() * 0.96 + (x * 0.014).cos() * 0.33;
        let o = c - (x * 0.019).cos() * 0.42;
        let span = 0.88 + (x * 0.017).sin().abs() * 0.31;
        open[i] = o;
        close[i] = c;
        high[i] = o.max(c) + span;
        low[i] = o.min(c) - span * (0.81 + (x * 0.012).cos().abs() * 0.19);
    }
    (open, high, low, close)
}

#[cfg(feature = "cuda")]
fn run_case(
    source: &str,
    avg_type: &str,
    sweep: EmdTrendBatchRange,
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[emd_trend_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let (mut open, mut high, mut low, mut close) = sample_ohlc(440);
    open[211] = f64::NAN;
    high[211] = f64::NAN;
    low[211] = f64::NAN;
    close[211] = f64::NAN;

    let cpu = emd_trend_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &sweep,
        source,
        avg_type,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaEmdTrend::new(0)?;
    let result = cuda.batch_dev(&open, &high, &low, &close, &sweep, source, avg_type)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_direction = vec![0.0f64; result.outputs.direction.len()];
    let mut got_average = vec![0.0f64; result.outputs.average.len()];
    let mut got_upper = vec![0.0f64; result.outputs.upper.len()];
    let mut got_lower = vec![0.0f64; result.outputs.lower.len()];
    result.outputs.direction.buf.copy_to(&mut got_direction)?;
    result.outputs.average.buf.copy_to(&mut got_average)?;
    result.outputs.upper.buf.copy_to(&mut got_upper)?;
    result.outputs.lower.buf.copy_to(&mut got_lower)?;

    for idx in 0..cpu.direction.len() {
        assert!(approx_eq(cpu.direction[idx], got_direction[idx], 1e-6));
        assert!(approx_eq(cpu.average[idx], got_average[idx], 1e-6));
        assert!(approx_eq(cpu.upper[idx], got_upper[idx], 1e-6));
        assert!(approx_eq(cpu.lower[idx], got_lower[idx], 1e-6));
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
fn emd_trend_cuda_ema_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    run_case(
        "ohlc4",
        "EMA",
        EmdTrendBatchRange {
            length: (20, 24, 4),
            mult: (0.8, 1.2, 0.4),
        },
    )
}

#[cfg(feature = "cuda")]
#[test]
fn emd_trend_cuda_frama_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    run_case(
        "hlc3",
        "FRAMA",
        EmdTrendBatchRange {
            length: (21, 23, 2),
            mult: (1.0, 1.0, 0.0),
        },
    )
}
