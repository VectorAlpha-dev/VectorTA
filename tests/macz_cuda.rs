// Integration tests for CUDA MAC-Z kernels

use my_project::indicators::macz::{
    macz_batch_with_kernel_vol, macz_with_kernel, MaczBatchRange, MaczInput, MaczParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaMacz;

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
fn macz_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[macz_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut price = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 60..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
        volume[i] = (x * 0.00077).cos().abs() + 0.5;
    }
    let sweep = MaczBatchRange {
        fast_length: (10, 14, 2),
        slow_length: (20, 30, 5),
        signal_length: (7, 11, 2),
        lengthz: (18, 22, 2),
        length_stdev: (20, 30, 5),
        a: (0.8, 1.2, 0.2),
        b: (0.8, 1.2, 0.2),
    };

    let cpu = macz_batch_with_kernel_vol(&price, Some(&volume), &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let cuda = CudaMacz::new(0).expect("CudaMacz::new");
    let (hist_dev, combos) = cuda
        .macz_batch_dev(&price_f32, Some(&volume_f32), &sweep)
        .expect("macz_batch_dev");

    assert_eq!(cpu.rows, hist_dev.rows);
    assert_eq!(cpu.cols, hist_dev.cols);
    assert_eq!(cpu.combos.len(), combos.len());

    let mut hist_host = vec![0f32; hist_dev.len()];
    hist_dev.buf.copy_to(&mut hist_host)?;

    let tol = 5e-3; // Allow FP32 drift (MAC-Z is numerically sensitive)
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = hist_host[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "hist mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn macz_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[macz_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 6usize;
    let rows = 1024usize;
    let mut price_tm = vec![f64::NAN; cols * rows];
    let mut volume_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            price_tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
            volume_tm[t * cols + s] = (x * 0.001).cos().abs() + 0.4;
        }
    }

    let params = MaczParams {
        fast_length: Some(12),
        slow_length: Some(25),
        signal_length: Some(9),
        lengthz: Some(20),
        length_stdev: Some(25),
        a: Some(1.0),
        b: Some(1.0),
        use_lag: Some(false),
        gamma: Some(0.02),
    };

    // CPU baseline per series
    let mut cpu_hist_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = price_tm[t * cols + s];
            v[t] = volume_tm[t * cols + s];
        }
        let input = MaczInput {
            data: my_project::indicators::macz::MaczData::SliceWithVolume {
                data: &p,
                volume: &v,
            },
            params: params.clone(),
        };
        let out = macz_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_hist_tm[t * cols + s] = out.values[t];
        }
    }

    let price_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let volume_tm_f32: Vec<f32> = volume_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaMacz::new(0).expect("CudaMacz::new");
    let dev_tm = cuda
        .macz_many_series_one_param_time_major_dev(
            &price_tm_f32,
            Some(&volume_tm_f32),
            cols,
            rows,
            &params,
        )
        .expect("macz_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);
    let mut g_hist_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut g_hist_tm)?;
    let tol = 5e-3;
    for idx in 0..g_hist_tm.len() {
        let c = cpu_hist_tm[idx];
        let g = g_hist_tm[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "hist mismatch at {}: cpu={} gpu={} diff={}",
            idx,
            c,
            g,
            (c - g).abs()
        );
    }
    Ok(())
}
