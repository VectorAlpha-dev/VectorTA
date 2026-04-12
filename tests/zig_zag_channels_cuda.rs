use vector_ta::indicators::zig_zag_channels::{
    zig_zag_channels_batch_with_kernel, ZigZagChannelsBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaZigZagChannels};

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
fn zig_zag_channels_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[zig_zag_channels_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 840usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 110.0f64;
    for i in 18..len {
        let x = i as f64;
        base += (x * 0.012).sin() * 0.48 + (x * 0.0025).cos() * 0.11;
        open[i] = base - 0.22 + (x * 0.007).cos() * 0.09;
        close[i] = base + (x * 0.019).sin() * 0.34;
        high[i] = open[i].max(close[i]) + 0.61 + (x * 0.006).sin().abs() * 0.15;
        low[i] = open[i].min(close[i]) - 0.58 - (x * 0.005).cos().abs() * 0.17;
    }
    for i in (320..380).step_by(15) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let cuda = CudaZigZagChannels::new(0).expect("CudaZigZagChannels::new");

    for extend in [true, false] {
        let sweep = ZigZagChannelsBatchRange {
            length: (24, 28, 4),
            extend,
        };

        let cpu = zig_zag_channels_batch_with_kernel(
            &open,
            &high,
            &low,
            &close,
            &sweep,
            Kernel::ScalarBatch,
        )?;
        let result = cuda
            .batch_dev(&open, &high, &low, &close, &sweep)
            .expect("batch_dev");

        assert_eq!(result.outputs.rows(), cpu.rows);
        assert_eq!(result.outputs.cols(), cpu.cols);
        assert_eq!(result.combos.len(), cpu.combos.len());

        let mut got_middle = vec![0.0f64; result.outputs.middle.len()];
        let mut got_upper = vec![0.0f64; result.outputs.upper.len()];
        let mut got_lower = vec![0.0f64; result.outputs.lower.len()];
        result.outputs.middle.buf.copy_to(&mut got_middle)?;
        result.outputs.upper.buf.copy_to(&mut got_upper)?;
        result.outputs.lower.buf.copy_to(&mut got_lower)?;

        for idx in 0..cpu.middle.len() {
            assert!(
                approx_eq(cpu.middle[idx], got_middle[idx], 1e-10),
                "middle mismatch for extend={extend} at {idx}: cpu={} cuda={}",
                cpu.middle[idx],
                got_middle[idx]
            );
            assert!(
                approx_eq(cpu.upper[idx], got_upper[idx], 1e-10),
                "upper mismatch for extend={extend} at {idx}: cpu={} cuda={}",
                cpu.upper[idx],
                got_upper[idx]
            );
            assert!(
                approx_eq(cpu.lower[idx], got_lower[idx], 1e-10),
                "lower mismatch for extend={extend} at {idx}: cpu={} cuda={}",
                cpu.lower[idx],
                got_lower[idx]
            );
        }
    }

    Ok(())
}
