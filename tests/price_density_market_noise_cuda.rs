use vector_ta::indicators::price_density_market_noise::{
    price_density_market_noise_batch_with_kernel, PriceDensityMarketNoiseBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaPriceDensityMarketNoise};

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
fn price_density_market_noise_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[price_density_market_noise_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 120.0f64;
    for i in 8..len {
        let x = i as f64;
        base += (x * 0.009).sin() * 0.48 + (x * 0.003).cos() * 0.21;
        let center = base + (x * 0.015).sin() * 0.37;
        close[i] = center + (x * 0.011).cos() * 0.19;
        high[i] = close[i] + 0.92 + (x * 0.017).sin().abs() * 0.21;
        low[i] = close[i] - 0.87 - (x * 0.013).cos().abs() * 0.18;
    }
    for i in (420..520).step_by(13) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1370..1460).step_by(9) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = PriceDensityMarketNoiseBatchRange {
        length: (10, 14, 2),
        eval_period: (24, 32, 8),
    };
    let cpu = price_density_market_noise_batch_with_kernel(
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaPriceDensityMarketNoise::new(0).expect("CudaPriceDensityMarketNoise::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_price_density = vec![0.0f64; result.outputs.price_density.len()];
    let mut got_price_density_percent = vec![0.0f64; result.outputs.price_density_percent.len()];
    result
        .outputs
        .price_density
        .buf
        .copy_to(&mut got_price_density)?;
    result
        .outputs
        .price_density_percent
        .buf
        .copy_to(&mut got_price_density_percent)?;

    for idx in 0..cpu.price_density.len() {
        assert!(
            approx_eq(cpu.price_density[idx], got_price_density[idx], 1e-10),
            "price_density mismatch at {idx}: cpu={} cuda={}",
            cpu.price_density[idx],
            got_price_density[idx]
        );
        assert!(
            approx_eq(
                cpu.price_density_percent[idx],
                got_price_density_percent[idx],
                1e-10
            ),
            "price_density_percent mismatch at {idx}: cpu={} cuda={}",
            cpu.price_density_percent[idx],
            got_price_density_percent[idx]
        );
    }

    Ok(())
}
