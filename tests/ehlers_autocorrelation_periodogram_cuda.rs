use vector_ta::indicators::ehlers_autocorrelation_periodogram::{
    ehlers_autocorrelation_periodogram_batch_with_kernel,
    EhlersAutocorrelationPeriodogramBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaEhlersAutocorrelationPeriodogram};

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
fn ehlers_autocorrelation_periodogram_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[ehlers_autocorrelation_periodogram_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 640usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 96.0f64;
    for i in 18..len {
        let x = i as f64;
        base += (x * 0.007).sin() * 0.22 + (x * 0.0019).cos() * 0.05;
        data[i] = base + (x * 0.051).sin() * 1.7 + (x * 0.033).cos() * 0.9;
    }
    data[302] = f64::NAN;

    let cuda = CudaEhlersAutocorrelationPeriodogram::new(0)
        .expect("CudaEhlersAutocorrelationPeriodogram::new");

    for enhance in [true, false] {
        let sweep = EhlersAutocorrelationPeriodogramBatchRange {
            min_period: (8, 10, 2),
            max_period: (18, 22, 4),
            avg_length: (0, 3, 3),
            enhance,
        };

        let cpu = ehlers_autocorrelation_periodogram_batch_with_kernel(
            &data,
            &sweep,
            Kernel::ScalarBatch,
        )?;
        let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

        assert_eq!(result.outputs.rows(), cpu.rows);
        assert_eq!(result.outputs.cols(), cpu.cols);
        assert_eq!(result.combos.len(), cpu.combos.len());

        let mut got_dominant_cycle = vec![0.0f64; result.outputs.dominant_cycle.len()];
        let mut got_normalized_power = vec![0.0f64; result.outputs.normalized_power.len()];
        result
            .outputs
            .dominant_cycle
            .buf
            .copy_to(&mut got_dominant_cycle)?;
        result
            .outputs
            .normalized_power
            .buf
            .copy_to(&mut got_normalized_power)?;

        for idx in 0..cpu.dominant_cycle.len() {
            assert!(
                approx_eq(cpu.dominant_cycle[idx], got_dominant_cycle[idx], 1e-6),
                "dominant_cycle mismatch for enhance={enhance} at {idx}: cpu={} cuda={}",
                cpu.dominant_cycle[idx],
                got_dominant_cycle[idx]
            );
            assert!(
                approx_eq(cpu.normalized_power[idx], got_normalized_power[idx], 1e-6),
                "normalized_power mismatch for enhance={enhance} at {idx}: cpu={} cuda={}",
                cpu.normalized_power[idx],
                got_normalized_power[idx]
            );
        }
    }

    Ok(())
}
