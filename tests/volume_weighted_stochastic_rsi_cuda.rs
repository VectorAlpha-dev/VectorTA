use vector_ta::indicators::volume_weighted_stochastic_rsi::{
    volume_weighted_stochastic_rsi_batch_with_kernel, VolumeWeightedStochasticRsiBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaVolumeWeightedStochasticRsi};

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
fn volume_weighted_stochastic_rsi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[volume_weighted_stochastic_rsi_cuda_batch_matches_cpu] skipped - no CUDA device"
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

    let ma_types = ["WSMA", "SMA", "EMA", "WMA", "VWMA"];
    let cuda =
        CudaVolumeWeightedStochasticRsi::new(0).expect("CudaVolumeWeightedStochasticRsi::new");

    for ma_type in ma_types {
        let sweep = VolumeWeightedStochasticRsiBatchRange {
            rsi_length: (10, 12, 2),
            stoch_length: (8, 10, 2),
            k_length: (3, 4, 1),
            d_length: (2, 3, 1),
            ma_type: ma_type.to_string(),
        };
        let cpu = volume_weighted_stochastic_rsi_batch_with_kernel(
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

        let mut got_k = vec![0.0f64; result.outputs.k.len()];
        let mut got_d = vec![0.0f64; result.outputs.d.len()];
        result.outputs.k.buf.copy_to(&mut got_k)?;
        result.outputs.d.buf.copy_to(&mut got_d)?;

        for idx in 0..cpu.k.len() {
            assert!(
                approx_eq(cpu.k[idx], got_k[idx], 1e-6),
                "k mismatch for {ma_type} at {idx}: cpu={} cuda={}",
                cpu.k[idx],
                got_k[idx]
            );
            assert!(
                approx_eq(cpu.d[idx], got_d[idx], 1e-6),
                "d mismatch for {ma_type} at {idx}: cpu={} cuda={}",
                cpu.d[idx],
                got_d[idx]
            );
        }
    }

    Ok(())
}
