use vector_ta::indicators::price_moving_average_ratio_percentile::{
    price_moving_average_ratio_percentile_batch_with_kernel,
    PriceMovingAverageRatioPercentileBatchRange, PriceMovingAverageRatioPercentileLineMode,
    PriceMovingAverageRatioPercentileMaType,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaPriceMovingAverageRatioPercentile};

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
fn price_moving_average_ratio_percentile_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[price_moving_average_ratio_percentile_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2048usize;
    let mut price = vec![0.0f64; len];
    let mut volume = vec![0.0f64; len];
    let mut base = 102.0f64;
    for i in 0..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.37 + (x * 0.003).cos() * 0.14;
        price[i] = base + (x * 0.018).sin() * 0.49 + x * 0.002;
        volume[i] = 8_000.0 + (x * 0.017).cos() * 640.0 + (i % 31) as f64 * 23.0;
    }

    let ma_types = [
        PriceMovingAverageRatioPercentileMaType::Sma,
        PriceMovingAverageRatioPercentileMaType::Ema,
        PriceMovingAverageRatioPercentileMaType::Hma,
        PriceMovingAverageRatioPercentileMaType::Rma,
        PriceMovingAverageRatioPercentileMaType::Vwma,
    ];
    let cuda = CudaPriceMovingAverageRatioPercentile::new(0)
        .expect("CudaPriceMovingAverageRatioPercentile::new");

    for ma_type in ma_types {
        let sweep = PriceMovingAverageRatioPercentileBatchRange {
            ma_length: (20, 22, 2),
            pmarp_lookback: (30, 30, 0),
            signal_ma_length: (5, 5, 0),
            ma_type: Some(ma_type),
            signal_ma_type: Some(ma_type),
            line_mode: Some(PriceMovingAverageRatioPercentileLineMode::Pmarp),
        };
        let cpu = price_moving_average_ratio_percentile_batch_with_kernel(
            &price,
            &volume,
            &sweep,
            Kernel::ScalarBatch,
        )?;
        let result = cuda.batch_dev(&price, &volume, &sweep).expect("batch_dev");

        assert_eq!(result.outputs.rows(), cpu.rows);
        assert_eq!(result.outputs.cols(), cpu.cols);
        assert_eq!(result.combos.len(), cpu.combos.len());

        let mut got_pmar = vec![0.0f64; result.outputs.pmar.len()];
        let mut got_pmarp = vec![0.0f64; result.outputs.pmarp.len()];
        let mut got_plotline = vec![0.0f64; result.outputs.plotline.len()];
        let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
        let mut got_pmar_high = vec![0.0f64; result.outputs.pmar_high.len()];
        let mut got_pmar_low = vec![0.0f64; result.outputs.pmar_low.len()];
        let mut got_scaled_pmar = vec![0.0f64; result.outputs.scaled_pmar.len()];
        result.outputs.pmar.buf.copy_to(&mut got_pmar)?;
        result.outputs.pmarp.buf.copy_to(&mut got_pmarp)?;
        result.outputs.plotline.buf.copy_to(&mut got_plotline)?;
        result.outputs.signal.buf.copy_to(&mut got_signal)?;
        result.outputs.pmar_high.buf.copy_to(&mut got_pmar_high)?;
        result.outputs.pmar_low.buf.copy_to(&mut got_pmar_low)?;
        result
            .outputs
            .scaled_pmar
            .buf
            .copy_to(&mut got_scaled_pmar)?;

        for idx in 0..cpu.pmar.len() {
            assert!(
                approx_eq(cpu.pmar[idx], got_pmar[idx], 1e-6),
                "pmar mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.pmar[idx],
                got_pmar[idx]
            );
            assert!(
                approx_eq(cpu.pmarp[idx], got_pmarp[idx], 1e-6),
                "pmarp mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.pmarp[idx],
                got_pmarp[idx]
            );
            assert!(
                approx_eq(cpu.plotline[idx], got_plotline[idx], 1e-6),
                "plotline mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.plotline[idx],
                got_plotline[idx]
            );
            assert!(
                approx_eq(cpu.signal[idx], got_signal[idx], 1e-6),
                "signal mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.signal[idx],
                got_signal[idx]
            );
            assert!(
                approx_eq(cpu.pmar_high[idx], got_pmar_high[idx], 1e-6),
                "pmar_high mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.pmar_high[idx],
                got_pmar_high[idx]
            );
            assert!(
                approx_eq(cpu.pmar_low[idx], got_pmar_low[idx], 1e-6),
                "pmar_low mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.pmar_low[idx],
                got_pmar_low[idx]
            );
            assert!(
                approx_eq(cpu.scaled_pmar[idx], got_scaled_pmar[idx], 1e-6),
                "scaled_pmar mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.scaled_pmar[idx],
                got_scaled_pmar[idx]
            );
        }
    }

    Ok(())
}
