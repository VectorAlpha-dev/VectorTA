use vector_ta::indicators::ehlers_adaptive_cg::{
    ehlers_adaptive_cg_batch_with_kernel, EhlersAdaptiveCgBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaEhlersAdaptiveCg};

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
fn ehlers_adaptive_cg_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ehlers_adaptive_cg_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 103.0f64;
    for i in 9..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.31 + (x * 0.004).cos() * 0.18;
        data[i] = base + (x * 0.023).sin() * 1.35 + (x * 0.007).cos() * 0.82;
    }
    for i in (410..470).step_by(9) {
        data[i] = f64::NAN;
    }
    for i in (1320..1390).step_by(11) {
        data[i] = f64::NAN;
    }

    let sweep = EhlersAdaptiveCgBatchRange {
        alpha: (0.07, 0.09, 0.02),
    };
    let cpu = ehlers_adaptive_cg_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaEhlersAdaptiveCg::new(0).expect("CudaEhlersAdaptiveCg::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_cg = vec![0.0f64; result.outputs.cg.len()];
    let mut got_trigger = vec![0.0f64; result.outputs.trigger.len()];
    result.outputs.cg.buf.copy_to(&mut got_cg)?;
    result.outputs.trigger.buf.copy_to(&mut got_trigger)?;

    for idx in 0..cpu.cg.len() {
        assert!(
            approx_eq(cpu.cg[idx], got_cg[idx], 1e-6),
            "cg mismatch at {idx}: cpu={} cuda={}",
            cpu.cg[idx],
            got_cg[idx]
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
