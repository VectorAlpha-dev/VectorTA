

use vector_ta::indicators::moving_averages::vwma::{
    vwma_batch_with_kernel, VwmaBatchRange, VwmaBuilder,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaVwma;

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
fn vwma_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vwma_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 2048usize;
    let mut prices = vec![f64::NAN; series_len];
    let mut volumes = vec![f64::NAN; series_len];
    for i in 6..series_len {
        let x = i as f64;
        prices[i] = (x * 0.002).sin() + 0.0005 * x;
        volumes[i] = 50.0 + (x * 0.01).cos();
    }

    let sweep = VwmaBatchRange { period: (6, 30, 4) };

    
    let cpu = vwma_batch_with_kernel(&prices, &volumes, &sweep, Kernel::ScalarBatch)?;

    
    let cuda = CudaVwma::new(0).expect("CudaVwma::new");
    let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
    let volumes_f32: Vec<f32> = volumes.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .vwma_batch_dev(&prices_f32, &volumes_f32, &sweep)
        .expect("cuda vwma_batch_dev");

    assert_eq!(gpu_handle.rows, cpu.rows);
    assert_eq!(gpu_handle.cols, cpu.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda vwma batch result to host");

    let tol = 1e-5;
    for idx in 0..gpu_host.len() {
        let cpu_val = cpu.values[idx];
        let gpu_val = gpu_host[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    
    let cpu_single = VwmaBuilder::new()
        .period(14)
        .apply_slice(&prices, &volumes)?
        .values;
    let row_idx = cpu
        .combos
        .iter()
        .position(|p| p.period.unwrap() == 14)
        .expect("combo for period 14 not found");
    let gpu_row = &gpu_host[row_idx * series_len..(row_idx + 1) * series_len];
    for (c, g) in cpu_single.iter().zip(gpu_row.iter()) {
        assert!(approx_eq(*c, *g as f64, tol));
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn vwma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vwma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 4usize;
    let series_len = 1024usize;

    let mut prices_tm = vec![f64::NAN; num_series * series_len];
    let mut volumes_tm = vec![f64::NAN; num_series * series_len];

    for series_idx in 0..num_series {
        for row in series_idx..series_len {
            let x = (row as f64) + (series_idx as f64) * 0.25;
            let price = (x * 0.0025).sin() + 0.0007 * x;
            let volume = 80.0 + (x * 0.015).cos() + f64::from(series_idx as u32);
            let idx = row * num_series + series_idx;
            prices_tm[idx] = price;
            volumes_tm[idx] = volume;
        }
    }

    let period = 18usize;

    
    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series_idx in 0..num_series {
        let mut prices_series = vec![f64::NAN; series_len];
        let mut volumes_series = vec![f64::NAN; series_len];
        for row in 0..series_len {
            let idx = row * num_series + series_idx;
            prices_series[row] = prices_tm[idx];
            volumes_series[row] = volumes_tm[idx];
        }
        let cpu = VwmaBuilder::new()
            .period(period)
            .apply_slice(&prices_series, &volumes_series)?
            .values;
        for row in 0..series_len {
            let idx = row * num_series + series_idx;
            cpu_tm[idx] = cpu[row];
        }
    }

    
    let cuda = CudaVwma::new(0).expect("CudaVwma::new");
    let prices_f32: Vec<f32> = prices_tm.iter().map(|&v| v as f32).collect();
    let volumes_f32: Vec<f32> = volumes_tm.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .vwma_many_series_one_param_time_major_dev(
            &prices_f32,
            &volumes_f32,
            num_series,
            series_len,
            period,
        )
        .expect("cuda vwma_many_series_one_param_time_major_dev");

    assert_eq!(gpu_handle.rows, series_len);
    assert_eq!(gpu_handle.cols, num_series);

    let mut gpu_tm = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy vwma many-series result");

    let tol = 1e-5;
    for idx in 0..cpu_tm.len() {
        let cpu_val = cpu_tm[idx];
        let gpu_val = gpu_tm[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    Ok(())
}
