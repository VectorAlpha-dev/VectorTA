use vector_ta::indicators::forward_backward_exponential_oscillator::{
    forward_backward_exponential_oscillator_batch_with_kernel,
    ForwardBackwardExponentialOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaForwardBackwardExponentialOscillator};

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
fn forward_backward_exponential_oscillator_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[forward_backward_exponential_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2304usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 92.0f64;
    for i in 5..len {
        let x = i as f64;
        value += (x * 0.011).sin() * 0.56 + (x * 0.004).cos() * 0.23;
        data[i] = value + (x * 0.023).sin() * 0.31;
    }
    for i in (700..760).step_by(8) {
        data[i] = f64::NAN;
    }
    for i in (1680..1760).step_by(10) {
        data[i] = f64::NAN;
    }

    let sweep = ForwardBackwardExponentialOscillatorBatchRange {
        length: (18, 20, 2),
        smooth: (6, 8, 2),
    };
    let cpu = forward_backward_exponential_oscillator_batch_with_kernel(
        &data,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaForwardBackwardExponentialOscillator::new(0)
        .expect("CudaForwardBackwardExponentialOscillator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_forward_backward = vec![0.0f64; result.outputs.forward_backward.len()];
    let mut got_backward = vec![0.0f64; result.outputs.backward.len()];
    let mut got_histogram = vec![0.0f64; result.outputs.histogram.len()];
    result
        .outputs
        .forward_backward
        .buf
        .copy_to(&mut got_forward_backward)?;
    result.outputs.backward.buf.copy_to(&mut got_backward)?;
    result.outputs.histogram.buf.copy_to(&mut got_histogram)?;

    for idx in 0..cpu.forward_backward.len() {
        assert!(
            approx_eq(cpu.forward_backward[idx], got_forward_backward[idx], 1e-10),
            "forward_backward mismatch at {idx}: cpu={} cuda={}",
            cpu.forward_backward[idx],
            got_forward_backward[idx]
        );
        assert!(
            approx_eq(cpu.backward[idx], got_backward[idx], 1e-10),
            "backward mismatch at {idx}: cpu={} cuda={}",
            cpu.backward[idx],
            got_backward[idx]
        );
        assert!(
            approx_eq(cpu.histogram[idx], got_histogram[idx], 1e-10),
            "histogram mismatch at {idx}: cpu={} cuda={}",
            cpu.histogram[idx],
            got_histogram[idx]
        );
    }

    Ok(())
}
