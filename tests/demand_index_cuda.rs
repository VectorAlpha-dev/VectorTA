use vector_ta::indicators::demand_index::{demand_index_batch_with_kernel, DemandIndexBatchRange};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaDemandIndex};

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
fn demand_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[demand_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 960usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    let mut base = 104.0f64;
    for i in 12..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.37 + (x * 0.002).cos() * 0.09;
        let center = base + (x * 0.016).sin() * 0.41;
        close[i] = center + (x * 0.013).cos() * 0.18;
        high[i] = close[i] + 0.77 + (x * 0.007).cos().abs() * 0.14;
        low[i] = close[i] - 0.73 - (x * 0.008).sin().abs() * 0.12;
        volume[i] = 8500.0 + x * 7.0 + (x * 0.021).sin() * 430.0 + (x * 0.011).cos() * 160.0;
    }
    for i in (380..460).step_by(17) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
        volume[i] = f64::NAN;
    }

    let cuda = CudaDemandIndex::new(0).expect("CudaDemandIndex::new");

    for ma_type in ["ema", "sma", "wma", "rma"] {
        let sweep = DemandIndexBatchRange {
            len_bs: (12, 14, 2),
            len_bs_ma: (10, 12, 2),
            len_di_ma: (8, 10, 2),
            ma_type: Some(ma_type.to_string()),
        };

        let cpu = demand_index_batch_with_kernel(
            &high,
            &low,
            &close,
            &volume,
            &sweep,
            Kernel::ScalarBatch,
        )?;
        let result = cuda
            .batch_dev(&high, &low, &close, &volume, &sweep)
            .expect("batch_dev");

        assert_eq!(result.outputs.rows(), cpu.rows);
        assert_eq!(result.outputs.cols(), cpu.cols);
        assert_eq!(result.combos.len(), cpu.combos.len());

        let mut got_di = vec![0.0f64; result.outputs.demand_index.len()];
        let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
        result.outputs.demand_index.buf.copy_to(&mut got_di)?;
        result.outputs.signal.buf.copy_to(&mut got_signal)?;

        for idx in 0..cpu.demand_index.len() {
            assert!(
                approx_eq(cpu.demand_index[idx], got_di[idx], 1e-10),
                "demand_index mismatch for ma_type={ma_type} at {idx}: cpu={} cuda={}",
                cpu.demand_index[idx],
                got_di[idx]
            );
            assert!(
                approx_eq(cpu.signal[idx], got_signal[idx], 1e-10),
                "signal mismatch for ma_type={ma_type} at {idx}: cpu={} cuda={}",
                cpu.signal[idx],
                got_signal[idx]
            );
        }
    }

    Ok(())
}
