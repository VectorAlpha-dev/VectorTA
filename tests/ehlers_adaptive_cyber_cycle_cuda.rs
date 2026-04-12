use vector_ta::indicators::ehlers_adaptive_cyber_cycle::{
    ehlers_adaptive_cyber_cycle_batch_with_kernel, EhlersAdaptiveCyberCycleBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaEhlersAdaptiveCyberCycle};

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
fn ehlers_adaptive_cyber_cycle_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ehlers_adaptive_cyber_cycle_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2096usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 88.0f64;
    for i in 8..len {
        let x = i as f64;
        base += (x * 0.014).sin() * 0.42 + (x * 0.003).cos() * 0.15;
        data[i] = base + (x * 0.016).sin() * 0.63 + (x * 0.005).cos() * 0.24;
    }
    for i in (410..470).step_by(9) {
        data[i] = f64::NAN;
    }
    for i in (1230..1310).step_by(11) {
        data[i] = f64::NAN;
    }

    let sweep = EhlersAdaptiveCyberCycleBatchRange {
        alpha: (0.05, 0.09, 0.02),
    };
    let cpu = ehlers_adaptive_cyber_cycle_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaEhlersAdaptiveCyberCycle::new(0).expect("CudaEhlersAdaptiveCyberCycle::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_cycle = vec![0.0f64; result.outputs.cycle.len()];
    let mut got_trigger = vec![0.0f64; result.outputs.trigger.len()];
    result.outputs.cycle.buf.copy_to(&mut got_cycle)?;
    result.outputs.trigger.buf.copy_to(&mut got_trigger)?;

    for idx in 0..cpu.cycle.len() {
        assert!(
            approx_eq(cpu.cycle[idx], got_cycle[idx], 1e-6),
            "cycle mismatch at {idx}: cpu={} cuda={}",
            cpu.cycle[idx],
            got_cycle[idx]
        );
        assert!(
            approx_eq(cpu.trigger[idx], got_trigger[idx], 1e-6),
            "trigger mismatch at {idx}: cpu={} cuda={}",
            cpu.trigger[idx],
            got_trigger[idx]
        );
    }

    Ok(())
}
