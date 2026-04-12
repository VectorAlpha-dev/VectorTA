use vector_ta::indicators::kairi_relative_index::{
    kairi_relative_index_batch_with_kernel, KairiRelativeIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaKairiRelativeIndex};

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
fn kairi_relative_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[kairi_relative_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut source = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    let mut base = 101.0f64;
    for i in 12..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.34 + (x * 0.004).cos() * 0.19;
        source[i] = base + (x * 0.016).sin() * 0.57 + (x * 0.007).cos() * 0.21;
        volume[i] = 9_000.0 + (x * 0.013).cos() * 1_700.0 + (i % 17) as f64 * 43.0;
    }
    for i in (360..440).step_by(10) {
        source[i] = f64::NAN;
        volume[i] = f64::NAN;
    }
    for i in (1180..1260).step_by(11) {
        source[i] = f64::NAN;
        volume[i] = f64::NAN;
    }

    let ma_types = [
        "SMA", "EMA", "WMA", "TMA", "VIDYA", "WWMA", "ZLEMA", "TSF", "HMA", "VWMA",
    ];
    let cuda = CudaKairiRelativeIndex::new(0).expect("CudaKairiRelativeIndex::new");

    for ma_type in ma_types {
        let sweep = KairiRelativeIndexBatchRange {
            length: (5, 9, 2),
            ma_type: ma_type.to_string(),
        };
        let cpu =
            kairi_relative_index_batch_with_kernel(&source, &volume, &sweep, Kernel::ScalarBatch)?;
        let result = cuda.batch_dev(&source, &volume, &sweep).expect("batch_dev");

        assert_eq!(result.outputs.rows, cpu.rows, "rows mismatch for {ma_type}");
        assert_eq!(result.outputs.cols, cpu.cols, "cols mismatch for {ma_type}");
        assert_eq!(
            result.combos.len(),
            cpu.combos.len(),
            "combos mismatch for {ma_type}"
        );

        let mut got = vec![0.0f64; result.outputs.len()];
        result.outputs.buf.copy_to(&mut got)?;

        for idx in 0..cpu.values.len() {
            assert!(
                approx_eq(cpu.values[idx], got[idx], 1e-6),
                "value mismatch for {ma_type} at {idx}: cpu={} cuda={}",
                cpu.values[idx],
                got[idx]
            );
        }
    }

    Ok(())
}
