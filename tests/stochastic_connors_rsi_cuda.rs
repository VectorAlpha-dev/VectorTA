use vector_ta::indicators::stochastic_connors_rsi::{
    stochastic_connors_rsi_batch_with_kernel, StochasticConnorsRsiBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaStochasticConnorsRsi};

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
fn stochastic_connors_rsi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[stochastic_connors_rsi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2368usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 104.0f64;
    for i in 9..len {
        let x = i as f64;
        base += (x * 0.008).sin() * 0.46 + (x * 0.003).cos() * 0.22;
        data[i] = base + (x * 0.021).sin() * 0.91 + (x * 0.007).cos() * 0.29;
    }
    for i in (470..560).step_by(10) {
        data[i] = f64::NAN;
    }
    for i in (1520..1600).step_by(12) {
        data[i] = f64::NAN;
    }

    let sweep = StochasticConnorsRsiBatchRange {
        stoch_length: (3, 5, 2),
        smooth_k: (2, 2, 0),
        smooth_d: (2, 3, 1),
        rsi_length: (3, 4, 1),
        updown_length: (2, 2, 0),
        roc_length: (60, 80, 20),
    };
    let cpu = stochastic_connors_rsi_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaStochasticConnorsRsi::new(0).expect("CudaStochasticConnorsRsi::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_k = vec![0.0f64; result.outputs.k.len()];
    let mut got_d = vec![0.0f64; result.outputs.d.len()];
    result.outputs.k.buf.copy_to(&mut got_k)?;
    result.outputs.d.buf.copy_to(&mut got_d)?;

    for idx in 0..cpu.k.len() {
        assert!(
            approx_eq(cpu.k[idx], got_k[idx], 1e-9),
            "k mismatch at {idx}: cpu={} cuda={}",
            cpu.k[idx],
            got_k[idx]
        );
        assert!(
            approx_eq(cpu.d[idx], got_d[idx], 1e-9),
            "d mismatch at {idx}: cpu={} cuda={}",
            cpu.d[idx],
            got_d[idx]
        );
    }

    Ok(())
}
