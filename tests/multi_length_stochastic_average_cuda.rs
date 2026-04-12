use vector_ta::indicators::multi_length_stochastic_average::{
    multi_length_stochastic_average_batch_with_kernel, MultiLengthStochasticAverageBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaMultiLengthStochasticAverage};

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
fn multi_length_stochastic_average_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[multi_length_stochastic_average_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 512usize;
    let mut data = vec![0.0f64; len];
    let mut base = 103.0f64;
    for i in 0..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.27 + (x * 0.0025).cos() * 0.09;
        data[i] = base + (x * 0.016).sin() * 0.71 + (x * 0.005).cos() * 0.33;
    }
    data[208] = f64::NAN;

    let method_pairs = [
        ("none", "sma"),
        ("sma", "lsma"),
        ("tma", "tma"),
        ("lsma", "none"),
    ];
    let cuda =
        CudaMultiLengthStochasticAverage::new(0).expect("CudaMultiLengthStochasticAverage::new");

    for (premethod, postmethod) in method_pairs {
        let sweep = MultiLengthStochasticAverageBatchRange {
            length: (12, 14, 2),
            presmooth: (5, 5, 0),
            postsmooth: (4, 4, 0),
            premethod: Some(premethod.to_string()),
            postmethod: Some(postmethod.to_string()),
        };

        let cpu =
            multi_length_stochastic_average_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
        let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

        assert_eq!(result.outputs.rows, cpu.rows);
        assert_eq!(result.outputs.cols, cpu.cols);
        assert_eq!(result.combos.len(), cpu.combos.len());

        let mut got = vec![0.0f64; result.outputs.len()];
        result.outputs.buf.copy_to(&mut got)?;

        for idx in 0..cpu.values.len() {
            assert!(
                approx_eq(cpu.values[idx], got[idx], 1e-9),
                "value mismatch for premethod={premethod} postmethod={postmethod} at {idx}: cpu={} cuda={}",
                cpu.values[idx],
                got[idx]
            );
        }
    }

    Ok(())
}
