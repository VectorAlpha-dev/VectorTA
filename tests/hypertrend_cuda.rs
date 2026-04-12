use vector_ta::indicators::hypertrend::{hypertrend_batch_with_kernel, HyperTrendBatchRange};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaHyperTrend};

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
fn hypertrend_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[hypertrend_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 720usize;
    let mut source = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut base = 115.0f64;
    for i in 24..len {
        let x = i as f64;
        base += (x * 0.009).sin() * 0.31 + (x * 0.0021).cos() * 0.08;
        source[i] = base + (x * 0.047).sin() * 1.9 + (x * 0.029).cos() * 0.7;
        let spread = 1.25 + (x * 0.018).sin().abs() * 0.8;
        high[i] = source[i] + spread;
        low[i] = source[i] - spread * (0.85 + (x * 0.011).cos().abs() * 0.2);
    }
    source[366] = f64::NAN;
    high[366] = f64::NAN;
    low[366] = f64::NAN;

    let sweep = HyperTrendBatchRange {
        factor: (3.0, 5.0, 2.0),
        slope: (10.0, 14.0, 4.0),
        width_percent: (60.0, 80.0, 20.0),
    };

    let cpu = hypertrend_batch_with_kernel(&high, &low, &source, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaHyperTrend::new(0).expect("CudaHyperTrend::new");
    let result = cuda
        .batch_dev(&high, &low, &source, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_upper = vec![0.0f64; result.outputs.upper.len()];
    let mut got_average = vec![0.0f64; result.outputs.average.len()];
    let mut got_lower = vec![0.0f64; result.outputs.lower.len()];
    let mut got_trend = vec![0.0f64; result.outputs.trend.len()];
    let mut got_changed = vec![0.0f64; result.outputs.changed.len()];
    result.outputs.upper.buf.copy_to(&mut got_upper)?;
    result.outputs.average.buf.copy_to(&mut got_average)?;
    result.outputs.lower.buf.copy_to(&mut got_lower)?;
    result.outputs.trend.buf.copy_to(&mut got_trend)?;
    result.outputs.changed.buf.copy_to(&mut got_changed)?;

    for idx in 0..cpu.upper.len() {
        assert!(
            approx_eq(cpu.upper[idx], got_upper[idx], 1e-6),
            "upper mismatch at {idx}: cpu={} cuda={}",
            cpu.upper[idx],
            got_upper[idx]
        );
        assert!(
            approx_eq(cpu.average[idx], got_average[idx], 1e-6),
            "average mismatch at {idx}: cpu={} cuda={}",
            cpu.average[idx],
            got_average[idx]
        );
        assert!(
            approx_eq(cpu.lower[idx], got_lower[idx], 1e-6),
            "lower mismatch at {idx}: cpu={} cuda={}",
            cpu.lower[idx],
            got_lower[idx]
        );
        assert!(
            approx_eq(cpu.trend[idx], got_trend[idx], 1e-6),
            "trend mismatch at {idx}: cpu={} cuda={}",
            cpu.trend[idx],
            got_trend[idx]
        );
        assert!(
            approx_eq(cpu.changed[idx], got_changed[idx], 1e-6),
            "changed mismatch at {idx}: cpu={} cuda={}",
            cpu.changed[idx],
            got_changed[idx]
        );
    }

    Ok(())
}
