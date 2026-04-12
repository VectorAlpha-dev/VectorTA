use vector_ta::indicators::l1_ehlers_phasor::{
    l1_ehlers_phasor_batch_with_kernel, L1EhlersPhasorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaL1EhlersPhasor};

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
fn l1_ehlers_phasor_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[l1_ehlers_phasor_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1536usize;
    let mut data = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        data[i] = 100.0 + x * 0.09 + (x * 0.12).sin() * 1.7 + (x * 0.03).cos() * 0.8;
    }
    data[640] = f64::NAN;
    data[641] = f64::NAN;
    data[1200] = f64::NAN;

    let sweep = L1EhlersPhasorBatchRange {
        domestic_cycle_length: (15, 19, 2),
    };
    let cpu = l1_ehlers_phasor_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaL1EhlersPhasor::new(0).expect("CudaL1EhlersPhasor::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-9),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
