// Integration tests for CUDA WTO kernels

use my_project::indicators::wto::{
    wto_batch_all_outputs_with_kernel, WtoBatchRange, WtoBuilder, WtoParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::{CudaWto, CudaWtoBatchResult, DeviceArrayF32Triplet};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs());
    diff <= tol + scale * (5.0 * tol)
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
fn wto_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[wto_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 7..series_len {
        let x = i as f64;
        data[i] = (x * 0.002).sin() + 0.0002 * x;
    }

    let sweep = WtoBatchRange {
        channel: (8, 16, 2),
        average: (18, 30, 4),
    };

    let cpu = wto_batch_all_outputs_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cpu_wt1: Vec<f32> = cpu.wt1.iter().map(|&v| v as f32).collect();
    let cpu_wt2: Vec<f32> = cpu.wt2.iter().map(|&v| v as f32).collect();
    let cpu_hist: Vec<f32> = cpu.hist.iter().map(|&v| v as f32).collect();

    let cuda = CudaWto::new(0).map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let result = cuda
        .wto_batch_dev(&data_f32, &sweep)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    let CudaWtoBatchResult { outputs, combos } = result;
    let DeviceArrayF32Triplet { wt1, wt2, hist } = outputs;

    assert_eq!(cpu.rows, wt1.rows);
    assert_eq!(cpu.cols, wt1.cols);

    let mut wt1_gpu = vec![0f32; cpu.rows * cpu.cols];
    let mut wt2_gpu = vec![0f32; cpu.rows * cpu.cols];
    let mut hist_gpu = vec![0f32; cpu.rows * cpu.cols];
    wt1.buf
        .copy_to(&mut wt1_gpu)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    wt2.buf
        .copy_to(&mut wt2_gpu)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    hist.buf
        .copy_to(&mut hist_gpu)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    for idx in 0..(cpu.rows * cpu.cols) {
        let wt1_cpu = cpu_wt1[idx] as f64;
        let wt2_cpu = cpu_wt2[idx] as f64;
        let hist_cpu = cpu_hist[idx] as f64;

        let wt1_dev = wt1_gpu[idx] as f64;
        let wt2_dev = wt2_gpu[idx] as f64;
        let hist_dev = hist_gpu[idx] as f64;

        assert!(
            approx_eq(wt1_cpu, wt1_dev, 1e-4),
            "WT1 mismatch at {}: cpu={} gpu={}",
            idx,
            wt1_cpu,
            wt1_dev
        );
        assert!(
            approx_eq(wt2_cpu, wt2_dev, 1e-4),
            "WT2 mismatch at {}: cpu={} gpu={}",
            idx,
            wt2_cpu,
            wt2_dev
        );
        assert!(
            approx_eq(hist_cpu, hist_dev, 1e-4),
            "Hist mismatch at {}: cpu={} gpu={}",
            idx,
            hist_cpu,
            hist_dev
        );
    }

    let cpu_channels: Vec<_> = cpu
        .combos
        .iter()
        .map(|p| p.channel_length.unwrap())
        .collect();
    let cpu_averages: Vec<_> = cpu
        .combos
        .iter()
        .map(|p| p.average_length.unwrap())
        .collect();
    let gpu_channels: Vec<_> = combos.iter().map(|p| p.channel_length.unwrap()).collect();
    let gpu_averages: Vec<_> = combos.iter().map(|p| p.average_length.unwrap()).collect();

    assert_eq!(cpu_channels, gpu_channels);
    assert_eq!(cpu_averages, gpu_averages);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn wto_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[wto_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let rows = 2048usize;
    let cols = 6usize;
    let mut data_tm = vec![f64::NAN; rows * cols];
    for series in 0..cols {
        for t in (series + 4)..rows {
            let x = (t as f64) + (series as f64) * 0.05;
            data_tm[t * cols + series] = (x * 0.003).cos() + 0.0003 * x;
        }
    }

    let params = WtoParams {
        channel_length: Some(9),
        average_length: Some(21),
    };

    let mut cpu_wt1_f32 = vec![f32::NAN; rows * cols];
    let mut cpu_wt2_f32 = vec![f32::NAN; rows * cols];
    let mut cpu_hist_f32 = vec![f32::NAN; rows * cols];

    for series in 0..cols {
        let mut column = vec![f64::NAN; rows];
        for t in 0..rows {
            column[t] = data_tm[t * cols + series];
        }
        let out = WtoBuilder::default()
            .channel_length(params.channel_length.unwrap())
            .average_length(params.average_length.unwrap())
            .apply_slice(&column)?;
        for t in 0..rows {
            cpu_wt1_f32[t * cols + series] = out.wavetrend1[t] as f32;
            cpu_wt2_f32[t * cols + series] = out.wavetrend2[t] as f32;
            cpu_hist_f32[t * cols + series] = out.histogram[t] as f32;
        }
    }

    let cuda = CudaWto::new(0).map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let result = cuda
        .wto_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    let DeviceArrayF32Triplet { wt1, wt2, hist } = result;

    assert_eq!(wt1.rows, rows);
    assert_eq!(wt1.cols, cols);

    let total = rows * cols;
    let mut wt1_gpu = vec![0f32; total];
    let mut wt2_gpu = vec![0f32; total];
    let mut hist_gpu = vec![0f32; total];
    wt1.buf
        .copy_to(&mut wt1_gpu)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    wt2.buf
        .copy_to(&mut wt2_gpu)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    hist.buf
        .copy_to(&mut hist_gpu)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    for idx in 0..total {
        let wt1_dev = wt1_gpu[idx] as f64;
        let wt2_dev = wt2_gpu[idx] as f64;
        let hist_dev = hist_gpu[idx] as f64;
        let wt1_cpu = cpu_wt1_f32[idx] as f64;
        let wt2_cpu = cpu_wt2_f32[idx] as f64;
        let hist_cpu = cpu_hist_f32[idx] as f64;
        assert!(
            approx_eq(wt1_cpu, wt1_dev, 1e-4),
            "WT1 mismatch at {}: cpu={} gpu={}",
            idx,
            wt1_cpu,
            wt1_dev
        );
        assert!(
            approx_eq(wt2_cpu, wt2_dev, 1e-4),
            "WT2 mismatch at {}: cpu={} gpu={}",
            idx,
            wt2_cpu,
            wt2_dev
        );
        assert!(
            approx_eq(hist_cpu, hist_dev, 1e-4),
            "Hist mismatch at {}: cpu={} gpu={}",
            idx,
            hist_cpu,
            hist_dev
        );
    }

    Ok(())
}
