

use vector_ta::indicators::moving_averages::hma::{
    hma_batch_with_kernel, HmaBatchRange, HmaBuilder, HmaParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaHma;

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
fn hma_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[hma_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 10..series_len {
        let x = i as f64 * 0.01;
        data[i] = (x * 0.25).cos() + 0.0001 * x;
    }

    let sweep = HmaBatchRange {
        period: (10, 40, 5),
    };

    let cpu = hma_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    let cuda = CudaHma::new(0).expect("CudaHma::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let (dev, combos) = cuda
        .hma_batch_dev(&data_f32, &sweep)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    assert_eq!(combos.len(), cpu.rows);
    assert_eq!(dev.rows, cpu.rows);
    assert_eq!(dev.cols, cpu.cols);

    for (idx, combo) in combos.iter().enumerate() {
        assert_eq!(combo.period, cpu.combos[idx].period);
    }

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    let tol = 5e-3;
    for i in 0..gpu_flat.len() {
        let a = cpu.values[i];
        let b = gpu_flat[i] as f64;
        assert!(
            approx_eq(a, b, tol),
            "Mismatch at {}: cpu={} gpu={}",
            i,
            a,
            b
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn hma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[hma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 5usize;
    let series_len = 3072usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for row in 0..series_len {
        for col in 0..num_series {
            if row < 15 + col {
                continue;
            }
            let t = row as f64;
            let phase = col as f64 * 0.4;
            data_tm[row * num_series + col] = (0.003 * (t + phase)).sin() + 0.2 * (t % 10.0) / 10.0;
        }
    }

    let period = 21usize;
    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for col in 0..num_series {
        let mut series = vec![f64::NAN; series_len];
        for row in 0..series_len {
            series[row] = data_tm[row * num_series + col];
        }
        let out = HmaBuilder::new()
            .period(period)
            .apply_slice(&series)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        for row in 0..series_len {
            cpu_tm[row * num_series + col] = out.values[row];
        }
    }

    let cuda = CudaHma::new(0).expect("CudaHma::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let params = HmaParams {
        period: Some(period),
    };
    let dev = cuda
        .hma_multi_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    assert_eq!(dev.rows, series_len);
    assert_eq!(dev.cols, num_series);

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    let tol = 5e-3;
    for idx in 0..gpu_flat.len() {
        let a = cpu_tm[idx];
        let b = gpu_flat[idx] as f64;
        assert!(
            approx_eq(a, b, tol),
            "Mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}
