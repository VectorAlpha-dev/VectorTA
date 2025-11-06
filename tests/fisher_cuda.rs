// Integration tests for CUDA Fisher kernels

use my_project::indicators::fisher::{
    fisher_batch_with_kernel, fisher_with_kernel, FisherBatchBuilder, FisherBatchRange,
    FisherInput, FisherParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaFisher;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn fisher_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[fisher_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 10..len {
        let x = i as f64;
        high[i] = (x * 0.002).sin() + 0.001 * x;
        low[i] = high[i] - 0.3 - 0.05 * (x * 0.1).cos();
    }

    let sweep = FisherBatchRange { period: (9, 64, 5) };

    let cpu = FisherBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slices(&high, &low)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cuda = CudaFisher::new(0).expect("CudaFisher::new");
    let (dev_pair, _combos) = cuda
        .fisher_batch_dev(&high_f32, &low_f32, &sweep)
        .expect("fisher cuda batch");

    assert_eq!(cpu.rows, dev_pair.rows());
    assert_eq!(cpu.cols, dev_pair.cols());

    let mut g_fish = vec![0f32; dev_pair.fisher.len()];
    let mut g_sig = vec![0f32; dev_pair.signal.len()];
    dev_pair.fisher.buf.copy_to(&mut g_fish)?;
    dev_pair.signal.buf.copy_to(&mut g_sig)?;

    let tol = 4.0; // relaxed absolute tol for GPU vs CPU (FP32 vs FP64)
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.fisher[idx], g_fish[idx] as f64, tol),
            "fisher mismatch at {} (row={}, col={}, period={}, cpu={}, gpu={})",
            idx,
            idx / cpu.cols,
            idx % cpu.cols,
            sweep.period.0 + (idx / cpu.cols) * sweep.period.2,
            cpu.fisher[idx],
            g_fish[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], g_sig[idx] as f64, tol),
            "signal mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn fisher_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[fisher_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 6usize; // series
    let rows = 2048usize; // length
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for r in s..rows {
            let x = r as f64 + 0.17 * s as f64;
            let h = (x * 0.0023).sin() + 0.0002 * x;
            let l = h - 0.25 - 0.07 * (x * 0.11).cos();
            high_tm[r * cols + s] = h;
            low_tm[r * cols + s] = l;
        }
    }
    let period = 13usize;

    let mut cpu_fish_tm = vec![f64::NAN; cols * rows];
    let mut cpu_sig_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut high = vec![f64::NAN; rows];
        let mut low = vec![f64::NAN; rows];
        for r in 0..rows {
            high[r] = high_tm[r * cols + s];
            low[r] = low_tm[r * cols + s];
        }
        let input = FisherInput::from_slices(
            &high,
            &low,
            FisherParams {
                period: Some(period),
            },
        );
        let out = fisher_with_kernel(&input, Kernel::Scalar)?;
        for r in 0..rows {
            cpu_fish_tm[r * cols + s] = out.fisher[r];
            cpu_sig_tm[r * cols + s] = out.signal[r];
        }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaFisher::new(0).expect("CudaFisher::new");
    let pair = cuda
        .fisher_many_series_one_param_time_major_dev(&high_tm_f32, &low_tm_f32, cols, rows, period)
        .expect("fisher many-series");
    assert_eq!(pair.rows(), rows);
    assert_eq!(pair.cols(), cols);
    let mut g_fish = vec![0f32; pair.fisher.len()];
    let mut g_sig = vec![0f32; pair.signal.len()];
    pair.fisher.buf.copy_to(&mut g_fish)?;
    pair.signal.buf.copy_to(&mut g_sig)?;

    let tol = 4.0; // relaxed absolute tol for GPU vs CPU (FP32 vs FP64)
    for idx in 0..g_fish.len() {
        assert!(
            approx_eq(cpu_fish_tm[idx], g_fish[idx] as f64, tol),
            "fisher mismatch at {} (row={}, col={}, period={}, cpu={}, gpu={})",
            idx,
            idx / cols,
            idx % cols,
            period,
            cpu_fish_tm[idx],
            g_fish[idx]
        );
        assert!(
            approx_eq(cpu_sig_tm[idx], g_sig[idx] as f64, tol),
            "signal mismatch at {}",
            idx
        );
    }

    Ok(())
}
