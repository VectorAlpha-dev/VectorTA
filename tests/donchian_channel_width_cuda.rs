use vector_ta::indicators::donchian_channel_width::{
    donchian_channel_width_batch_with_kernel, DonchianChannelWidthBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaDonchianChannelWidth};

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
fn donchian_channel_width_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[donchian_channel_width_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 10..len {
        let x = i as f64;
        let base = 100.0 + 0.02 * x + (x * 0.011).sin();
        let spread = 0.9 + (x * 0.017).cos().abs() * 0.25;
        high[i] = base + spread;
        low[i] = base - spread;
    }
    for i in (900..980).step_by(13) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
    }

    let sweep = DonchianChannelWidthBatchRange {
        period: (10, 30, 10),
    };
    let cpu = donchian_channel_width_batch_with_kernel(&high, &low, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaDonchianChannelWidth::new(0).expect("CudaDonchianChannelWidth::new");
    let result = cuda.batch_dev(&high, &low, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-10),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
