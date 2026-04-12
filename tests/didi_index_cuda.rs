use vector_ta::indicators::didi_index::{didi_index_batch_with_kernel, DidiIndexBatchRange};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaDidiIndex};

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
fn didi_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[didi_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1792usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 94.0f64;
    for i in 7..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.44 + (x * 0.003).cos() * 0.18;
        data[i] = base + (x * 0.015).cos() * 0.57 + (x * 0.007).sin() * 0.24;
    }
    for i in (360..430).step_by(11) {
        data[i] = f64::NAN;
    }
    for i in (1080..1160).step_by(9) {
        data[i] = f64::NAN;
    }

    let sweep = DidiIndexBatchRange {
        short_length: (3, 5, 2),
        medium_length: (8, 10, 2),
        long_length: (18, 22, 2),
    };
    let cpu = didi_index_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaDidiIndex::new(0).expect("CudaDidiIndex::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_short = vec![0.0f64; result.outputs.short.len()];
    let mut got_long = vec![0.0f64; result.outputs.long.len()];
    let mut got_crossover = vec![0.0f64; result.outputs.crossover.len()];
    let mut got_crossunder = vec![0.0f64; result.outputs.crossunder.len()];
    result.outputs.short.buf.copy_to(&mut got_short)?;
    result.outputs.long.buf.copy_to(&mut got_long)?;
    result.outputs.crossover.buf.copy_to(&mut got_crossover)?;
    result.outputs.crossunder.buf.copy_to(&mut got_crossunder)?;

    for idx in 0..cpu.short.len() {
        assert!(
            approx_eq(cpu.short[idx], got_short[idx], 1e-10),
            "short mismatch at {idx}: cpu={} cuda={}",
            cpu.short[idx],
            got_short[idx]
        );
        assert!(
            approx_eq(cpu.long[idx], got_long[idx], 1e-10),
            "long mismatch at {idx}: cpu={} cuda={}",
            cpu.long[idx],
            got_long[idx]
        );
        assert!(
            approx_eq(cpu.crossover[idx], got_crossover[idx], 1e-10),
            "crossover mismatch at {idx}: cpu={} cuda={}",
            cpu.crossover[idx],
            got_crossover[idx]
        );
        assert!(
            approx_eq(cpu.crossunder[idx], got_crossunder[idx], 1e-10),
            "crossunder mismatch at {idx}: cpu={} cuda={}",
            cpu.crossunder[idx],
            got_crossunder[idx]
        );
    }

    Ok(())
}
