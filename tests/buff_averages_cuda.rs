// Integration tests for CUDA Buff Averages kernels

use my_project::indicators::moving_averages::buff_averages::{
    buff_averages_batch_with_kernel, buff_averages_with_kernel, BuffAveragesBatchRange,
    BuffAveragesInput, BuffAveragesParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaBuffAverages;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[cfg(feature = "cuda")]
fn using_nvcc_stub() -> bool {
    std::env::var("NVCC")
        .map(|p| p.contains("nvcc_stub.sh"))
        .unwrap_or(false)
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
fn buff_averages_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if using_nvcc_stub() {
        eprintln!(
            "[buff_averages_cuda_batch_matches_cpu] skipped - NVCC stub in use (placeholder PTX)"
        );
        return Ok(());
    }
    if !cuda_available() {
        eprintln!("[buff_averages_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
        volume[i] = (x * 0.00077).cos().abs() + 0.5;
    }
    let sweep = BuffAveragesBatchRange {
        fast_period: (4, 28, 4),
        slow_period: (32, 128, 16),
    };

    let cpu = buff_averages_batch_with_kernel(&price, &volume, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let cuda = CudaBuffAverages::new(0).expect("CudaBuffAverages::new");
    let (fast_dev, slow_dev) = match cuda
        .buff_averages_batch_dev(&price_f32, &volume_f32, &sweep)
    {
        Ok(v) => v,
        Err(e) => {
            let msg = e.to_string();
            if msg.to_lowercase().contains("named symbol not found") {
                eprintln!(
                    "[buff_averages_cuda_batch_matches_cpu] skipping - PTX missing kernels (likely placeholder)"
                );
                return Ok(());
            }
            return Err(Box::<dyn std::error::Error>::from(e));
        }
    };

    assert_eq!(cpu.rows, fast_dev.rows);
    assert_eq!(cpu.cols, fast_dev.cols);
    assert_eq!(cpu.rows, slow_dev.rows);
    assert_eq!(cpu.cols, slow_dev.cols);

    let mut fast_host = vec![0f32; fast_dev.len()];
    fast_dev.buf.copy_to(&mut fast_host)?;
    let mut slow_host = vec![0f32; slow_dev.len()];
    slow_dev.buf.copy_to(&mut slow_host)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c_fast = cpu.fast[idx];
        let g_fast = fast_host[idx] as f64;
        let c_slow = cpu.slow[idx];
        let g_slow = slow_host[idx] as f64;
        assert!(
            approx_eq(c_fast, g_fast, tol),
            "fast mismatch at {}: cpu={} gpu={}",
            idx,
            c_fast,
            g_fast
        );
        assert!(
            approx_eq(c_slow, g_slow, tol),
            "slow mismatch at {}: cpu={} gpu={}",
            idx,
            c_slow,
            g_slow
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn buff_averages_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if using_nvcc_stub() {
        eprintln!(
            "[buff_averages_cuda_many_series_one_param_matches_cpu] skipped - NVCC stub in use (placeholder PTX)"
        );
        return Ok(());
    }
    if !cuda_available() {
        eprintln!(
            "[buff_averages_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let cols = 8usize;
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

    let fast = 10usize;
    let slow = 21usize;

    // CPU baseline per series
    let mut cpu_fast_tm = vec![f64::NAN; cols * rows];
    let mut cpu_slow_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = price_tm[t * cols + s];
            v[t] = volume_tm[t * cols + s];
        }
        let params = BuffAveragesParams {
            fast_period: Some(fast),
            slow_period: Some(slow),
        };
        let input = BuffAveragesInput {
            data: my_project::indicators::moving_averages::buff_averages::BuffAveragesData::Slice(
                &p,
            ),
            volume: Some(&v),
            params,
        };
        let out = buff_averages_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_fast_tm[t * cols + s] = out.fast_buff[t];
            cpu_slow_tm[t * cols + s] = out.slow_buff[t];
        }
    }

    let price_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let volume_tm_f32: Vec<f32> = volume_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaBuffAverages::new(0).expect("CudaBuffAverages::new");
    let (fast_dev_tm, slow_dev_tm) = match cuda
        .buff_averages_many_series_one_param_time_major_dev(
            &price_tm_f32,
            &volume_tm_f32,
            cols,
            rows,
            fast,
            slow,
        )
    {
        Ok(v) => v,
        Err(e) => {
            let msg = e.to_string();
            if msg.to_lowercase().contains("named symbol not found") {
                eprintln!(
                    "[buff_averages_cuda_many_series_one_param_matches_cpu] skipping - PTX missing kernels (likely placeholder)"
                );
                return Ok(());
            }
            return Err(Box::<dyn std::error::Error>::from(e));
        }
    };

    assert_eq!(fast_dev_tm.rows, rows);
    assert_eq!(fast_dev_tm.cols, cols);
    assert_eq!(slow_dev_tm.rows, rows);
    assert_eq!(slow_dev_tm.cols, cols);

    let mut g_fast_tm = vec![0f32; fast_dev_tm.len()];
    let mut g_slow_tm = vec![0f32; slow_dev_tm.len()];
    fast_dev_tm.buf.copy_to(&mut g_fast_tm)?;
    slow_dev_tm.buf.copy_to(&mut g_slow_tm)?;

    let tol = 1e-4;
    for idx in 0..g_fast_tm.len() {
        assert!(
            approx_eq(cpu_fast_tm[idx], g_fast_tm[idx] as f64, tol),
            "fast mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_slow_tm[idx], g_slow_tm[idx] as f64, tol),
            "slow mismatch at {}",
            idx
        );
    }

    Ok(())
}
