use vector_ta::indicators::moving_average_cross_probability::{
    moving_average_cross_probability_batch_with_kernel, MovingAverageCrossProbabilityBatchRange,
    MovingAverageCrossProbabilityMaType,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaMovingAverageCrossProbability};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

const TOL: f64 = 1e-6;

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn moving_average_cross_probability_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[moving_average_cross_probability_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2304usize;
    let mut data = vec![0.0f64; len];
    let mut base = 96.0f64;
    for (i, item) in data.iter_mut().enumerate() {
        let x = i as f64;
        base += (x * 0.009).sin() * 0.44 + (x * 0.003).cos() * 0.19;
        *item = base + (x * 0.016).sin() * 0.72 + (x * 0.005).cos() * 0.31;
    }

    let cuda =
        CudaMovingAverageCrossProbability::new(0).expect("CudaMovingAverageCrossProbability::new");
    let ma_types = [
        MovingAverageCrossProbabilityMaType::Ema,
        MovingAverageCrossProbabilityMaType::Sma,
    ];

    for ma_type in ma_types {
        let sweep = MovingAverageCrossProbabilityBatchRange {
            smoothing_window: (7, 8, 1),
            slow_length: (30, 30, 0),
            fast_length: (14, 14, 0),
            resolution: (50, 50, 0),
            ma_type,
        };
        let cpu =
            moving_average_cross_probability_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
        let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

        assert_eq!(result.outputs.rows(), cpu.rows);
        assert_eq!(result.outputs.cols(), cpu.cols);
        assert_eq!(result.combos.len(), cpu.combos.len());

        let mut got_value = vec![0.0f64; result.outputs.value.len()];
        let mut got_slow = vec![0.0f64; result.outputs.slow_ma.len()];
        let mut got_fast = vec![0.0f64; result.outputs.fast_ma.len()];
        let mut got_forecast = vec![0.0f64; result.outputs.forecast.len()];
        let mut got_upper = vec![0.0f64; result.outputs.upper.len()];
        let mut got_lower = vec![0.0f64; result.outputs.lower.len()];
        let mut got_direction = vec![0.0f64; result.outputs.direction.len()];
        result.outputs.value.buf.copy_to(&mut got_value)?;
        result.outputs.slow_ma.buf.copy_to(&mut got_slow)?;
        result.outputs.fast_ma.buf.copy_to(&mut got_fast)?;
        result.outputs.forecast.buf.copy_to(&mut got_forecast)?;
        result.outputs.upper.buf.copy_to(&mut got_upper)?;
        result.outputs.lower.buf.copy_to(&mut got_lower)?;
        result.outputs.direction.buf.copy_to(&mut got_direction)?;

        for idx in 0..cpu.value.len() {
            assert!(
                approx_eq(cpu.value[idx], got_value[idx], TOL),
                "value mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.value[idx],
                got_value[idx]
            );
            assert!(
                approx_eq(cpu.slow_ma[idx], got_slow[idx], TOL),
                "slow_ma mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.slow_ma[idx],
                got_slow[idx]
            );
            assert!(
                approx_eq(cpu.fast_ma[idx], got_fast[idx], TOL),
                "fast_ma mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.fast_ma[idx],
                got_fast[idx]
            );
            assert!(
                approx_eq(cpu.forecast[idx], got_forecast[idx], TOL),
                "forecast mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.forecast[idx],
                got_forecast[idx]
            );
            assert!(
                approx_eq(cpu.upper[idx], got_upper[idx], TOL),
                "upper mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.upper[idx],
                got_upper[idx]
            );
            assert!(
                approx_eq(cpu.lower[idx], got_lower[idx], TOL),
                "lower mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.lower[idx],
                got_lower[idx]
            );
            assert!(
                approx_eq(cpu.direction[idx], got_direction[idx], TOL),
                "direction mismatch for {:?} at {idx}: cpu={} cuda={}",
                ma_type,
                cpu.direction[idx],
                got_direction[idx]
            );
        }
    }

    Ok(())
}
