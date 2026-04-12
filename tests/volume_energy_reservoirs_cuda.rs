use vector_ta::indicators::volume_energy_reservoirs::{
    volume_energy_reservoirs_batch_with_kernel, VolumeEnergyReservoirsBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaVolumeEnergyReservoirs};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_series(length: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let close = (0..length)
        .map(|i| {
            let x = i as f64;
            100.0 + x * 0.04 + (x * 0.13).sin() * 2.4 + (x * 0.03).cos() * 0.8
        })
        .collect::<Vec<_>>();
    let high = close
        .iter()
        .enumerate()
        .map(|(i, value)| value + 0.7 + (i as f64 * 0.05).cos().abs() * 0.3)
        .collect::<Vec<_>>();
    let low = close
        .iter()
        .enumerate()
        .map(|(i, value)| value - 0.8 - (i as f64 * 0.07).sin().abs() * 0.25)
        .collect::<Vec<_>>();
    let volume = (0..length)
        .map(|i| {
            let x = i as f64;
            1_000.0 + x * 3.0 + (x * 0.11).sin() * 220.0 + (x * 0.021).cos() * 50.0
        })
        .collect::<Vec<_>>();
    (high, low, close, volume)
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn volume_energy_reservoirs_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[volume_energy_reservoirs_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let (mut high, mut low, mut close, mut volume) = sample_series(320);
    high[155] = f64::NAN;
    low[155] = f64::NAN;
    close[155] = f64::NAN;
    volume[155] = f64::NAN;

    let sweep = VolumeEnergyReservoirsBatchRange {
        length: (18, 22, 4),
        sensitivity: (1.5, 2.0, 0.5),
    };

    let cpu = volume_energy_reservoirs_batch_with_kernel(
        &high,
        &low,
        &close,
        &volume,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaVolumeEnergyReservoirs::new(0)?;
    let result = cuda.batch_dev(&high, &low, &close, &volume, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_momentum = vec![0.0f64; result.outputs.momentum.len()];
    let mut got_reservoir = vec![0.0f64; result.outputs.reservoir.len()];
    let mut got_squeeze_active = vec![0.0f64; result.outputs.squeeze_active.len()];
    let mut got_squeeze_start = vec![0.0f64; result.outputs.squeeze_start.len()];
    let mut got_range_high = vec![0.0f64; result.outputs.range_high.len()];
    let mut got_range_low = vec![0.0f64; result.outputs.range_low.len()];
    result.outputs.momentum.buf.copy_to(&mut got_momentum)?;
    result.outputs.reservoir.buf.copy_to(&mut got_reservoir)?;
    result
        .outputs
        .squeeze_active
        .buf
        .copy_to(&mut got_squeeze_active)?;
    result
        .outputs
        .squeeze_start
        .buf
        .copy_to(&mut got_squeeze_start)?;
    result.outputs.range_high.buf.copy_to(&mut got_range_high)?;
    result.outputs.range_low.buf.copy_to(&mut got_range_low)?;

    for idx in 0..cpu.momentum.len() {
        assert!(
            approx_eq(cpu.momentum[idx], got_momentum[idx], 1e-6),
            "momentum mismatch at {idx}: cpu={} cuda={}",
            cpu.momentum[idx],
            got_momentum[idx]
        );
        assert!(
            approx_eq(cpu.reservoir[idx], got_reservoir[idx], 1e-6),
            "reservoir mismatch at {idx}: cpu={} cuda={}",
            cpu.reservoir[idx],
            got_reservoir[idx]
        );
        assert!(
            approx_eq(cpu.squeeze_active[idx], got_squeeze_active[idx], 1e-6),
            "squeeze_active mismatch at {idx}: cpu={} cuda={}",
            cpu.squeeze_active[idx],
            got_squeeze_active[idx]
        );
        assert!(
            approx_eq(cpu.squeeze_start[idx], got_squeeze_start[idx], 1e-6),
            "squeeze_start mismatch at {idx}: cpu={} cuda={}",
            cpu.squeeze_start[idx],
            got_squeeze_start[idx]
        );
        assert!(
            approx_eq(cpu.range_high[idx], got_range_high[idx], 1e-6),
            "range_high mismatch at {idx}: cpu={} cuda={}",
            cpu.range_high[idx],
            got_range_high[idx]
        );
        assert!(
            approx_eq(cpu.range_low[idx], got_range_low[idx], 1e-6),
            "range_low mismatch at {idx}: cpu={} cuda={}",
            cpu.range_low[idx],
            got_range_low[idx]
        );
    }

    Ok(())
}
