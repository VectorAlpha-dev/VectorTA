use vector_ta::indicators::ehlers_simple_cycle_indicator::{
    ehlers_simple_cycle_indicator_batch_with_kernel, EhlersSimpleCycleIndicatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaEhlersSimpleCycleIndicator};

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
fn ehlers_simple_cycle_indicator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[ehlers_simple_cycle_indicator_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 1536usize;
    let mut data = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        data[i] = 95.0 + x * 0.06 + (x * 0.1).sin() * 1.9 + (x * 0.027).cos() * 0.6;
    }
    data[500] = f64::NAN;
    data[900] = f64::NAN;

    let sweep = EhlersSimpleCycleIndicatorBatchRange {
        alpha: (0.05, 0.09, 0.02),
    };
    let cpu = ehlers_simple_cycle_indicator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaEhlersSimpleCycleIndicator::new(0).expect("CudaEhlersSimpleCycleIndicator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.cycle.rows, cpu.rows);
    assert_eq!(result.outputs.cycle.cols, cpu.cols);
    assert_eq!(result.outputs.trigger.rows, cpu.rows);
    assert_eq!(result.outputs.trigger.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_cycle = vec![0.0f64; result.outputs.cycle.len()];
    let mut got_trigger = vec![0.0f64; result.outputs.trigger.len()];
    result.outputs.cycle.buf.copy_to(&mut got_cycle)?;
    result.outputs.trigger.buf.copy_to(&mut got_trigger)?;

    for idx in 0..cpu.cycle.len() {
        assert!(
            approx_eq(cpu.cycle[idx], got_cycle[idx], 1e-9),
            "cycle mismatch at {idx}: cpu={} cuda={}",
            cpu.cycle[idx],
            got_cycle[idx]
        );
        assert!(
            approx_eq(cpu.trigger[idx], got_trigger[idx], 1e-9),
            "trigger mismatch at {idx}: cpu={} cuda={}",
            cpu.trigger[idx],
            got_trigger[idx]
        );
    }

    Ok(())
}
