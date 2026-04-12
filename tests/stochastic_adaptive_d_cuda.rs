use vector_ta::indicators::stochastic_adaptive_d::{
    stochastic_adaptive_d_batch_with_kernel, StochasticAdaptiveDBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaStochasticAdaptiveD};

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
fn stochastic_adaptive_d_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[stochastic_adaptive_d_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2016usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 108.0f64;
    for i in 9..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.39 + (x * 0.004).cos() * 0.22;
        let center = base + (x * 0.015).sin() * 0.44;
        close[i] = center + (x * 0.013).cos() * 0.18;
        high[i] = close[i] + 0.93 + (x * 0.012).sin().abs() * 0.17;
        low[i] = close[i] - 0.89 - (x * 0.011).cos().abs() * 0.15;
    }
    for i in (520..590).step_by(10) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1450..1520).step_by(12) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = StochasticAdaptiveDBatchRange {
        k_length: (18, 20, 2),
        d_smoothing: (7, 9, 2),
        pre_smooth: (18, 20, 2),
        attenuation: (1.5, 2.0, 0.5),
    };
    let cpu =
        stochastic_adaptive_d_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaStochasticAdaptiveD::new(0).expect("CudaStochasticAdaptiveD::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_standard = vec![0.0f64; result.outputs.standard_d.len()];
    let mut got_adaptive = vec![0.0f64; result.outputs.adaptive_d.len()];
    let mut got_difference = vec![0.0f64; result.outputs.difference.len()];
    result.outputs.standard_d.buf.copy_to(&mut got_standard)?;
    result.outputs.adaptive_d.buf.copy_to(&mut got_adaptive)?;
    result.outputs.difference.buf.copy_to(&mut got_difference)?;

    for idx in 0..cpu.standard_d.len() {
        assert!(
            approx_eq(cpu.standard_d[idx], got_standard[idx], 1e-9),
            "standard_d mismatch at {idx}: cpu={} cuda={}",
            cpu.standard_d[idx],
            got_standard[idx]
        );
        assert!(
            approx_eq(cpu.adaptive_d[idx], got_adaptive[idx], 1e-9),
            "adaptive_d mismatch at {idx}: cpu={} cuda={}",
            cpu.adaptive_d[idx],
            got_adaptive[idx]
        );
        assert!(
            approx_eq(cpu.difference[idx], got_difference[idx], 1e-9),
            "difference mismatch at {idx}: cpu={} cuda={}",
            cpu.difference[idx],
            got_difference[idx]
        );
    }

    Ok(())
}
