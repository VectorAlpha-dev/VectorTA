use vector_ta::indicators::ehlers_linear_extrapolation_predictor::{
    ehlers_linear_extrapolation_predictor_batch_with_kernel,
    EhlersLinearExtrapolationPredictorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaEhlersLinearExtrapolationPredictor};

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
fn ehlers_linear_extrapolation_predictor_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[ehlers_linear_extrapolation_predictor_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2304usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 112.0f64;
    for i in 18..len {
        let x = i as f64;
        base += (x * 0.008).sin() * 0.37 + (x * 0.003).cos() * 0.14;
        data[i] = base + (x * 0.023).sin() * 0.91 + (x * 0.005).cos() * 0.33;
    }

    let sweep = EhlersLinearExtrapolationPredictorBatchRange {
        high_pass_length: (32, 48, 16),
        low_pass_length: (6, 10, 4),
        gain: (0.7, 1.1, 0.4),
        bars_forward: (0, 4, 4),
        signal_mode: Some("predict_filter_crosses".to_string()),
    };
    let cpu = ehlers_linear_extrapolation_predictor_batch_with_kernel(
        &data,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaEhlersLinearExtrapolationPredictor::new(0)
        .expect("CudaEhlersLinearExtrapolationPredictor::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_prediction = vec![0.0f64; result.outputs.prediction.len()];
    let mut got_filter = vec![0.0f64; result.outputs.filter.len()];
    let mut got_state = vec![0.0f64; result.outputs.state.len()];
    let mut got_go_long = vec![0.0f64; result.outputs.go_long.len()];
    let mut got_go_short = vec![0.0f64; result.outputs.go_short.len()];
    result.outputs.prediction.buf.copy_to(&mut got_prediction)?;
    result.outputs.filter.buf.copy_to(&mut got_filter)?;
    result.outputs.state.buf.copy_to(&mut got_state)?;
    result.outputs.go_long.buf.copy_to(&mut got_go_long)?;
    result.outputs.go_short.buf.copy_to(&mut got_go_short)?;

    for idx in 0..cpu.prediction.len() {
        assert!(
            approx_eq(cpu.prediction[idx], got_prediction[idx], 1e-6),
            "prediction mismatch at {idx}: cpu={} cuda={}",
            cpu.prediction[idx],
            got_prediction[idx]
        );
        assert!(
            approx_eq(cpu.filter[idx], got_filter[idx], 1e-6),
            "filter mismatch at {idx}: cpu={} cuda={}",
            cpu.filter[idx],
            got_filter[idx]
        );
        assert!(
            approx_eq(cpu.state[idx], got_state[idx], 1e-9),
            "state mismatch at {idx}: cpu={} cuda={}",
            cpu.state[idx],
            got_state[idx]
        );
        assert!(
            approx_eq(cpu.go_long[idx], got_go_long[idx], 1e-9),
            "go_long mismatch at {idx}: cpu={} cuda={}",
            cpu.go_long[idx],
            got_go_long[idx]
        );
        assert!(
            approx_eq(cpu.go_short[idx], got_go_short[idx], 1e-9),
            "go_short mismatch at {idx}: cpu={} cuda={}",
            cpu.go_short[idx],
            got_go_short[idx]
        );
    }

    Ok(())
}
