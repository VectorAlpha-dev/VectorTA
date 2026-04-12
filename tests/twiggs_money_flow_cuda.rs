use vector_ta::indicators::twiggs_money_flow::{
    twiggs_money_flow_batch_with_kernel, TwiggsMoneyFlowBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaTwiggsMoneyFlow};

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
fn twiggs_money_flow_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[twiggs_money_flow_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2176usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    let mut base = 102.0f64;
    for i in 7..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.38 + (x * 0.004).cos() * 0.17;
        close[i] = base + (x * 0.015).sin() * 0.49;
        high[i] = close[i] + 0.82 + (x * 0.013).sin().abs() * 0.24;
        low[i] = close[i] - 0.79 - (x * 0.012).cos().abs() * 0.19;
        volume[i] = 20_000.0 + (x * 0.017).sin() * 2_700.0 + (x % 23.0) * 111.0;
    }
    for i in (390..470).step_by(10) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
    }
    for i in (1180..1260).step_by(12) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
        volume[i] = f64::NAN;
    }

    let sweep = TwiggsMoneyFlowBatchRange {
        length: (5, 7, 2),
        smoothing_length: (4, 5, 1),
        ma_type: "WMA".to_string(),
    };
    let cpu = twiggs_money_flow_batch_with_kernel(
        &high,
        &low,
        &close,
        &volume,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaTwiggsMoneyFlow::new(0).expect("CudaTwiggsMoneyFlow::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &volume, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_tmf = vec![0.0f64; result.outputs.tmf.len()];
    let mut got_smoothed = vec![0.0f64; result.outputs.smoothed.len()];
    result.outputs.tmf.buf.copy_to(&mut got_tmf)?;
    result.outputs.smoothed.buf.copy_to(&mut got_smoothed)?;

    for idx in 0..cpu.tmf.len() {
        assert!(
            approx_eq(cpu.tmf[idx], got_tmf[idx], 1e-9),
            "tmf mismatch at {idx}: cpu={} cuda={}",
            cpu.tmf[idx],
            got_tmf[idx]
        );
        assert!(
            approx_eq(cpu.smoothed[idx], got_smoothed[idx], 1e-9),
            "smoothed mismatch at {idx}: cpu={} cuda={}",
            cpu.smoothed[idx],
            got_smoothed[idx]
        );
    }

    Ok(())
}
