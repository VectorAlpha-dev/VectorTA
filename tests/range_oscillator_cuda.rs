use vector_ta::indicators::range_oscillator::{
    range_oscillator_batch_with_kernel, RangeOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaRangeOscillator};

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
fn range_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[range_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2304usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 104.0f64;
    for i in 15..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.29 + (x * 0.003).cos() * 0.17;
        close[i] = base + (x * 0.017).sin() * 0.61 + (x * 0.005).cos() * 0.23;
        high[i] = close[i] + 0.92 + (x * 0.013).sin().abs() * 0.24;
        low[i] = close[i] - 0.88 - (x * 0.012).cos().abs() * 0.21;
    }
    for i in (460..560).step_by(13) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1490..1580).step_by(12) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = RangeOscillatorBatchRange {
        length: (40, 60, 20),
        mult: (1.5, 2.5, 1.0),
    };
    let cpu = range_oscillator_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaRangeOscillator::new(0).expect("CudaRangeOscillator::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_oscillator = vec![0.0f64; result.outputs.oscillator.len()];
    let mut got_ma = vec![0.0f64; result.outputs.ma.len()];
    let mut got_upper_band = vec![0.0f64; result.outputs.upper_band.len()];
    let mut got_lower_band = vec![0.0f64; result.outputs.lower_band.len()];
    let mut got_range_width = vec![0.0f64; result.outputs.range_width.len()];
    let mut got_in_range = vec![0.0f64; result.outputs.in_range.len()];
    let mut got_trend = vec![0.0f64; result.outputs.trend.len()];
    let mut got_break_up = vec![0.0f64; result.outputs.break_up.len()];
    let mut got_break_down = vec![0.0f64; result.outputs.break_down.len()];
    result.outputs.oscillator.buf.copy_to(&mut got_oscillator)?;
    result.outputs.ma.buf.copy_to(&mut got_ma)?;
    result.outputs.upper_band.buf.copy_to(&mut got_upper_band)?;
    result.outputs.lower_band.buf.copy_to(&mut got_lower_band)?;
    result
        .outputs
        .range_width
        .buf
        .copy_to(&mut got_range_width)?;
    result.outputs.in_range.buf.copy_to(&mut got_in_range)?;
    result.outputs.trend.buf.copy_to(&mut got_trend)?;
    result.outputs.break_up.buf.copy_to(&mut got_break_up)?;
    result.outputs.break_down.buf.copy_to(&mut got_break_down)?;

    for idx in 0..cpu.oscillator.len() {
        assert!(
            approx_eq(cpu.oscillator[idx], got_oscillator[idx], 1e-6),
            "oscillator mismatch at {idx}: cpu={} cuda={}",
            cpu.oscillator[idx],
            got_oscillator[idx]
        );
        assert!(
            approx_eq(cpu.ma[idx], got_ma[idx], 1e-6),
            "ma mismatch at {idx}: cpu={} cuda={}",
            cpu.ma[idx],
            got_ma[idx]
        );
        assert!(
            approx_eq(cpu.upper_band[idx], got_upper_band[idx], 1e-6),
            "upper_band mismatch at {idx}: cpu={} cuda={}",
            cpu.upper_band[idx],
            got_upper_band[idx]
        );
        assert!(
            approx_eq(cpu.lower_band[idx], got_lower_band[idx], 1e-6),
            "lower_band mismatch at {idx}: cpu={} cuda={}",
            cpu.lower_band[idx],
            got_lower_band[idx]
        );
        assert!(
            approx_eq(cpu.range_width[idx], got_range_width[idx], 1e-6),
            "range_width mismatch at {idx}: cpu={} cuda={}",
            cpu.range_width[idx],
            got_range_width[idx]
        );
        assert!(
            approx_eq(cpu.in_range[idx], got_in_range[idx], 1e-9),
            "in_range mismatch at {idx}: cpu={} cuda={}",
            cpu.in_range[idx],
            got_in_range[idx]
        );
        assert!(
            approx_eq(cpu.trend[idx], got_trend[idx], 1e-9),
            "trend mismatch at {idx}: cpu={} cuda={}",
            cpu.trend[idx],
            got_trend[idx]
        );
        assert!(
            approx_eq(cpu.break_up[idx], got_break_up[idx], 1e-9),
            "break_up mismatch at {idx}: cpu={} cuda={}",
            cpu.break_up[idx],
            got_break_up[idx]
        );
        assert!(
            approx_eq(cpu.break_down[idx], got_break_down[idx], 1e-9),
            "break_down mismatch at {idx}: cpu={} cuda={}",
            cpu.break_down[idx],
            got_break_down[idx]
        );
    }

    Ok(())
}
