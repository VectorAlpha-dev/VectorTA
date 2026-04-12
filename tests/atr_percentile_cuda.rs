use vector_ta::indicators::atr_percentile::{
    atr_percentile_batch_with_kernel, AtrPercentileBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaAtrPercentile};

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
fn atr_percentile_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[atr_percentile_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 3072usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 100.0f64;
    for i in 9..len {
        base += (i as f64 * 0.010).sin() * 0.65 + (i as f64 * 0.017).cos() * 0.22;
        let c = base + (i as f64 * 0.005).sin() * 0.18;
        let h = c + 0.9 + (i as f64 * 0.013).sin().abs() * 0.25;
        let l = c - 0.8 - (i as f64 * 0.011).cos().abs() * 0.24;
        high[i] = h;
        low[i] = l;
        close[i] = c;
    }
    for i in (900..980).step_by(17) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1800..1860).step_by(19) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = AtrPercentileBatchRange {
        atr_length: (5, 7, 2),
        percentile_length: (9, 11, 2),
    };
    let cpu = atr_percentile_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaAtrPercentile::new(0).expect("CudaAtrPercentile::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-10),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
