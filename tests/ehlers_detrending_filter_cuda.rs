use vector_ta::indicators::ehlers_detrending_filter::{
    ehlers_detrending_filter_batch_with_kernel, EhlersDetrendingFilterBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaEhlersDetrendingFilter};

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
fn ehlers_detrending_filter_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ehlers_detrending_filter_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1856usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 98.0f64;
    for i in 8..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.28 + (x * 0.004).cos() * 0.17;
        data[i] = base + (x * 0.021).sin() * 1.15 + (x * 0.008).cos() * 0.54;
    }
    for i in (370..430).step_by(10) {
        data[i] = f64::NAN;
    }
    for i in (1110..1180).step_by(9) {
        data[i] = f64::NAN;
    }

    let sweep = EhlersDetrendingFilterBatchRange {
        length: (10, 14, 2),
    };
    let cpu = ehlers_detrending_filter_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaEhlersDetrendingFilter::new(0).expect("CudaEhlersDetrendingFilter::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_edf = vec![0.0f64; result.outputs.edf.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    result.outputs.edf.buf.copy_to(&mut got_edf)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.edf.len() {
        assert!(
            approx_eq(cpu.edf[idx], got_edf[idx], 1e-6),
            "edf mismatch at {idx}: cpu={} cuda={}",
            cpu.edf[idx],
            got_edf[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-6),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
    }

    Ok(())
}
