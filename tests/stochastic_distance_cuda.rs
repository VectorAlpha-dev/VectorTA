use vector_ta::indicators::stochastic_distance::{
    stochastic_distance_batch_with_kernel, StochasticDistanceBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaStochasticDistance};

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
fn stochastic_distance_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[stochastic_distance_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1792usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 120.0f64;
    for i in 5..len {
        let x = i as f64;
        value += (x * 0.013).sin() * 0.62 + (x * 0.006).cos() * 0.25;
        data[i] = value + (x * 0.019).sin() * 0.28;
    }

    let sweep = StochasticDistanceBatchRange {
        lookback_length: (40, 50, 10),
        length1: (8, 10, 2),
        length2: (3, 3, 0),
        ob_level: (40, 40, 0),
        os_level: (-40, -40, 0),
    };
    let cpu = stochastic_distance_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaStochasticDistance::new(0).expect("CudaStochasticDistance::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_oscillator = vec![0.0f64; result.outputs.oscillator.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    result.outputs.oscillator.buf.copy_to(&mut got_oscillator)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.oscillator.len() {
        assert!(
            approx_eq(cpu.oscillator[idx], got_oscillator[idx], 1e-10),
            "oscillator mismatch at {idx}: cpu={} cuda={}",
            cpu.oscillator[idx],
            got_oscillator[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-10),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
    }

    Ok(())
}
