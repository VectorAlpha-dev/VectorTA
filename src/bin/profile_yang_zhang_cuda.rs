#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("profile_yang_zhang_cuda requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
use std::error::Error;
#[cfg(feature = "cuda")]
use std::time::Instant;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaYangZhangVolatility;
#[cfg(feature = "cuda")]
use vector_ta::indicators::yang_zhang_volatility::YangZhangVolatilityBatchRange;

#[cfg(feature = "cuda")]
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

#[cfg(feature = "cuda")]
fn env_string(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

#[cfg(feature = "cuda")]
fn gen_ohlc(len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut open = vec![0.0f32; len];
    let mut high = vec![0.0f32; len];
    let mut low = vec![0.0f32; len];
    let mut close = vec![0.0f32; len];
    let mut prev = 1000.0f32;
    for i in 0..len {
        let x = i as f32;
        let drift = 0.0002f32 * x;
        let wave = (x * 0.0013f32).sin() * 2.0 + (x * 0.00037f32).cos() * 1.3;
        let o = (prev + drift + wave).max(1.0);
        let c = (o + (x * 0.0021f32).sin() * 0.7).max(1.0);
        let hi = o.max(c) + 0.35 + (x * 0.0011f32).cos().abs() * 0.08;
        let lo = (o.min(c) - 0.35 - (x * 0.0017f32).sin().abs() * 0.08).max(0.01);
        open[i] = o;
        high[i] = hi;
        low[i] = lo;
        close[i] = c.max(0.01);
        prev = close[i];
    }
    (open, high, low, close)
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn Error>> {
    let len = env_usize("YZ_LEN", 1_000_000);
    let params = env_usize("YZ_PARAMS", 250);
    let warmup = env_usize("YZ_WARMUP", 1);
    let iters = env_usize("YZ_ITERS", 5).max(1);
    let device = env_usize("YZ_DEVICE", 0);
    let mode = env_string("YZ_MODE", "prepared");

    let sweep = YangZhangVolatilityBatchRange {
        lookback: (10, 10 + params.saturating_sub(1), 1),
        k_override: false,
        k: (0.34, 0.34, 0.0),
    };
    let (open, high, low, close) = gen_ohlc(len);
    let cuda = CudaYangZhangVolatility::new(device)?;

    match mode.as_str() {
        "public" => {
            for _ in 0..warmup {
                let _ = cuda.yang_zhang_volatility_batch_dev(&open, &high, &low, &close, &sweep)?;
            }
            let started = Instant::now();
            let mut rows = 0usize;
            let mut cols = 0usize;
            for _ in 0..iters {
                let result =
                    cuda.yang_zhang_volatility_batch_dev(&open, &high, &low, &close, &sweep)?;
                rows = result.outputs.rows();
                cols = result.outputs.cols();
            }
            let elapsed = started.elapsed();
            println!(
                "mode=public len={} params={} rows={} cols={} total_ms={:.3} per_iter_ms={:.3}",
                len,
                params,
                rows,
                cols,
                elapsed.as_secs_f64() * 1e3,
                elapsed.as_secs_f64() * 1e3 / iters as f64
            );
        }
        "prepared" => {
            let prepared_series = cuda.prepare_batch_series(&open, &high, &low, &close)?;
            let mut prepared_batch = cuda.prepare_batch_run(&prepared_series, &sweep)?;
            for _ in 0..warmup {
                cuda.launch_prepared_batch(&prepared_series, &mut prepared_batch)?;
                cuda.synchronize()?;
            }
            let started = Instant::now();
            for _ in 0..iters {
                cuda.launch_prepared_batch(&prepared_series, &mut prepared_batch)?;
                cuda.synchronize()?;
            }
            let elapsed = started.elapsed();
            println!(
                "mode=prepared len={} params={} rows={} cols={} total_ms={:.3} per_iter_ms={:.3}",
                len,
                params,
                prepared_batch.rows(),
                prepared_batch.cols(),
                elapsed.as_secs_f64() * 1e3,
                elapsed.as_secs_f64() * 1e3 / iters as f64
            );
        }
        other => {
            return Err(format!("unsupported YZ_MODE: {other}").into());
        }
    }

    Ok(())
}
