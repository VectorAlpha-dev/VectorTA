use vector_ta::indicators::stochastic_money_flow_index::{
    stochastic_money_flow_index_batch_with_kernel, StochasticMoneyFlowIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaStochasticMoneyFlowIndex};

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
fn stochastic_money_flow_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[stochastic_money_flow_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 3072usize;
    let mut source = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    let mut value = 120.0f64;
    for i in 8..len {
        value += (i as f64 * 0.016).sin() * 0.58 + (i as f64 * 0.006).cos() * 0.24;
        source[i] = value;
        volume[i] = 1_000.0 + (i as f64 * 0.011).sin() * 140.0 + (i % 17) as f64 * 9.0;
    }
    for i in (1400..1470).step_by(13) {
        source[i] = f64::NAN;
        volume[i] = f64::NAN;
    }

    let sweep = StochasticMoneyFlowIndexBatchRange {
        stoch_k_length: (12, 14, 2),
        stoch_k_smooth: (3, 3, 0),
        stoch_d_smooth: (3, 5, 2),
        mfi_length: (12, 14, 2),
    };
    let cpu = stochastic_money_flow_index_batch_with_kernel(
        &source,
        &volume,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaStochasticMoneyFlowIndex::new(0).expect("CudaStochasticMoneyFlowIndex::new");
    let result = cuda.batch_dev(&source, &volume, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_k = vec![0f64; result.outputs.k.len()];
    let mut got_d = vec![0f64; result.outputs.d.len()];
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
