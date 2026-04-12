use vector_ta::indicators::volume_weighted_relative_strength_index::{
    volume_weighted_relative_strength_index_batch_with_kernel,
    VolumeWeightedRelativeStrengthIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaVolumeWeightedRelativeStrengthIndex};

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
fn volume_weighted_relative_strength_index_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[volume_weighted_relative_strength_index_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2304usize;
    let mut source = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    let mut base = 88.0f64;
    for i in 10..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.41 + (x * 0.003).cos() * 0.16;
        source[i] = base + (x * 0.015).cos() * 0.52 + (x * 0.006).sin() * 0.28;
        volume[i] = 7_500.0 + (x * 0.014).sin() * 1_200.0 + (i % 29) as f64 * 37.0;
    }
    for i in (520..600).step_by(13) {
        source[i] = f64::NAN;
        volume[i] = f64::NAN;
    }
    for i in (1510..1580).step_by(17) {
        source[i] = f64::NAN;
        volume[i] = f64::NAN;
    }

    let ma_types = ["EMA", "SMA", "HMA", "RMA", "WMA", "VWMA"];
    let cuda = CudaVolumeWeightedRelativeStrengthIndex::new(0)
        .expect("CudaVolumeWeightedRelativeStrengthIndex::new");

    for ma_type in ma_types {
        let sweep = VolumeWeightedRelativeStrengthIndexBatchRange {
            rsi_length: (10, 12, 2),
            range_length: (10, 10, 0),
            ma_length: (6, 8, 2),
            ma_type: ma_type.to_string(),
        };
        let cpu = volume_weighted_relative_strength_index_batch_with_kernel(
            &source,
            &volume,
            &sweep,
            Kernel::ScalarBatch,
        )?;
        let result = cuda.batch_dev(&source, &volume, &sweep).expect("batch_dev");

        assert_eq!(
            result.outputs.rows(),
            cpu.rows,
            "rows mismatch for {ma_type}"
        );
        assert_eq!(
            result.outputs.cols(),
            cpu.cols,
            "cols mismatch for {ma_type}"
        );
        assert_eq!(
            result.combos.len(),
            cpu.combos.len(),
            "combos mismatch for {ma_type}"
        );

        let mut got_rsi = vec![0.0f64; result.outputs.rsi.len()];
        let mut got_consolidation = vec![0.0f64; result.outputs.consolidation_strength.len()];
        let mut got_rsi_ma = vec![0.0f64; result.outputs.rsi_ma.len()];
        let mut got_bearish = vec![0.0f64; result.outputs.bearish_tp.len()];
        let mut got_bullish = vec![0.0f64; result.outputs.bullish_tp.len()];
        result.outputs.rsi.buf.copy_to(&mut got_rsi)?;
        result
            .outputs
            .consolidation_strength
            .buf
            .copy_to(&mut got_consolidation)?;
        result.outputs.rsi_ma.buf.copy_to(&mut got_rsi_ma)?;
        result.outputs.bearish_tp.buf.copy_to(&mut got_bearish)?;
        result.outputs.bullish_tp.buf.copy_to(&mut got_bullish)?;

        for idx in 0..cpu.rsi.len() {
            assert!(
                approx_eq(cpu.rsi[idx], got_rsi[idx], 1e-6),
                "rsi mismatch for {ma_type} at {idx}: cpu={} cuda={}",
                cpu.rsi[idx],
                got_rsi[idx]
            );
            assert!(
                approx_eq(
                    cpu.consolidation_strength[idx],
                    got_consolidation[idx],
                    1e-6
                ),
                "consolidation_strength mismatch for {ma_type} at {idx}: cpu={} cuda={}",
                cpu.consolidation_strength[idx],
                got_consolidation[idx]
            );
            assert!(
                approx_eq(cpu.rsi_ma[idx], got_rsi_ma[idx], 1e-6),
                "rsi_ma mismatch for {ma_type} at {idx}: cpu={} cuda={}",
                cpu.rsi_ma[idx],
                got_rsi_ma[idx]
            );
            assert!(
                approx_eq(cpu.bearish_tp[idx], got_bearish[idx], 1e-6),
                "bearish_tp mismatch for {ma_type} at {idx}: cpu={} cuda={}",
                cpu.bearish_tp[idx],
                got_bearish[idx]
            );
            assert!(
                approx_eq(cpu.bullish_tp[idx], got_bullish[idx], 1e-6),
                "bullish_tp mismatch for {ma_type} at {idx}: cpu={} cuda={}",
                cpu.bullish_tp[idx],
                got_bullish[idx]
            );
        }
    }

    Ok(())
}
