use vector_ta::indicators::macd_wave_signal_pro::{
    macd_wave_signal_pro_batch_with_kernel, MacdWaveSignalProBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaMacdWaveSignalPro};

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
fn macd_wave_signal_pro_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[macd_wave_signal_pro_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 960usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 102.0f64;
    for i in 12..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.41 + (x * 0.003).cos() * 0.12;
        let center = base + (x * 0.015).sin() * 0.46;
        open[i] = center - 0.18 + (x * 0.006).cos() * 0.05;
        close[i] = center + (x * 0.013).sin() * 0.24;
        high[i] = open[i].max(close[i]) + 0.52 + (x * 0.009).cos().abs() * 0.11;
        low[i] = open[i].min(close[i]) - 0.49 - (x * 0.008).sin().abs() * 0.09;
    }
    for i in (420..500).step_by(13) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = MacdWaveSignalProBatchRange;
    let cpu = macd_wave_signal_pro_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaMacdWaveSignalPro::new(0).expect("CudaMacdWaveSignalPro::new");
    let result = cuda
        .batch_dev(&open, &high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_diff = vec![0.0f64; result.outputs.diff.len()];
    let mut got_dea = vec![0.0f64; result.outputs.dea.len()];
    let mut got_macd = vec![0.0f64; result.outputs.macd_histogram.len()];
    let mut got_line = vec![0.0f64; result.outputs.line_convergence.len()];
    let mut got_buy = vec![0.0f64; result.outputs.buy_signal.len()];
    let mut got_sell = vec![0.0f64; result.outputs.sell_signal.len()];
    result.outputs.diff.buf.copy_to(&mut got_diff)?;
    result.outputs.dea.buf.copy_to(&mut got_dea)?;
    result.outputs.macd_histogram.buf.copy_to(&mut got_macd)?;
    result.outputs.line_convergence.buf.copy_to(&mut got_line)?;
    result.outputs.buy_signal.buf.copy_to(&mut got_buy)?;
    result.outputs.sell_signal.buf.copy_to(&mut got_sell)?;

    for idx in 0..cpu.diff.len() {
        assert!(
            approx_eq(cpu.diff[idx], got_diff[idx], 1e-10),
            "diff mismatch at {idx}: cpu={} cuda={}",
            cpu.diff[idx],
            got_diff[idx]
        );
        assert!(
            approx_eq(cpu.dea[idx], got_dea[idx], 1e-10),
            "dea mismatch at {idx}: cpu={} cuda={}",
            cpu.dea[idx],
            got_dea[idx]
        );
        assert!(
            approx_eq(cpu.macd_histogram[idx], got_macd[idx], 1e-10),
            "macd_histogram mismatch at {idx}: cpu={} cuda={}",
            cpu.macd_histogram[idx],
            got_macd[idx]
        );
        assert!(
            approx_eq(cpu.line_convergence[idx], got_line[idx], 1e-10),
            "line_convergence mismatch at {idx}: cpu={} cuda={}",
            cpu.line_convergence[idx],
            got_line[idx]
        );
        assert!(
            approx_eq(cpu.buy_signal[idx], got_buy[idx], 1e-10),
            "buy_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.buy_signal[idx],
            got_buy[idx]
        );
        assert!(
            approx_eq(cpu.sell_signal[idx], got_sell[idx], 1e-10),
            "sell_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.sell_signal[idx],
            got_sell[idx]
        );
    }

    Ok(())
}
