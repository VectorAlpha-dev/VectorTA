

use vector_ta::indicators::wto::{
    wto_batch_all_outputs_with_kernel, WtoBatchRange, WtoBuilder, WtoParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{CudaWto, CudaWtoBatchResult, DeviceArrayF32Triplet};

fn approx_ratio(a: f64, b: f64, tol: f64) -> f64 {
    if a.is_nan() && b.is_nan() {
        return 0.0;
    }
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs());
    let allowed = tol + scale * (5.0 * tol);
    if allowed == 0.0 {
        if diff == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        diff / allowed
    }
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

    
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let data_cpu: Vec<f64> = data_f32.iter().map(|&v| v as f64).collect();

    let cpu = wto_batch_all_outputs_with_kernel(&data_cpu, &sweep, Kernel::ScalarBatch)?;
    let cpu_wt1: Vec<f32> = cpu.wt1.iter().map(|&v| v as f32).collect();
    let cpu_wt2: Vec<f32> = cpu.wt2.iter().map(|&v| v as f32).collect();
    let cpu_hist: Vec<f32> = cpu.hist.iter().map(|&v| v as f32).collect();

    let cuda = CudaWto::new(0).map_err(|e| Box::<dyn std::error::Error>::from(e))?;
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

    let tol = 3e-4;
    let mut wt1_max_ratio = 0.0f64;
    let mut wt1_worst = 0usize;
    let mut wt2_max_ratio = 0.0f64;
    let mut wt2_worst = 0usize;
    let mut hist_max_ratio = 0.0f64;
    let mut hist_worst = 0usize;

    for idx in 0..(cpu.rows * cpu.cols) {
        let wt1_cpu = cpu_wt1[idx] as f64;
        let wt2_cpu = cpu_wt2[idx] as f64;
        let hist_cpu = cpu_hist[idx] as f64;

        let wt1_dev = wt1_gpu[idx] as f64;
        let wt2_dev = wt2_gpu[idx] as f64;
        let hist_dev = hist_gpu[idx] as f64;

        let r1 = approx_ratio(wt1_cpu, wt1_dev, tol);
        if r1 > wt1_max_ratio {
            wt1_max_ratio = r1;
            wt1_worst = idx;
        }
        let r2 = approx_ratio(wt2_cpu, wt2_dev, tol);
        if r2 > wt2_max_ratio {
            wt2_max_ratio = r2;
            wt2_worst = idx;
        }
        let r3 = approx_ratio(hist_cpu, hist_dev, tol);
        if r3 > hist_max_ratio {
            hist_max_ratio = r3;
            hist_worst = idx;
        }
    }

    assert!(
        wt1_max_ratio <= 1.0,
        "WT1 max mismatch at {}: cpu={} gpu={} ratio={} tol={}",
        wt1_worst,
        cpu_wt1[wt1_worst] as f64,
        wt1_gpu[wt1_worst] as f64,
        wt1_max_ratio,
        tol
    );
    assert!(
        wt2_max_ratio <= 1.0,
        "WT2 max mismatch at {}: cpu={} gpu={} ratio={} tol={}",
        wt2_worst,
        cpu_wt2[wt2_worst] as f64,
        wt2_gpu[wt2_worst] as f64,
        wt2_max_ratio,
        tol
    );
    assert!(
        hist_max_ratio <= 1.0,
        "Hist max mismatch at {}: cpu={} gpu={} ratio={} tol={}",
        hist_worst,
        cpu_hist[hist_worst] as f64,
        hist_gpu[hist_worst] as f64,
        hist_max_ratio,
        tol
    );

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

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let data_tm_cpu: Vec<f64> = data_tm_f32.iter().map(|&v| v as f64).collect();

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
            column[t] = data_tm_cpu[t * cols + series];
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

    let tol = 3e-4;
    let mut wt1_max_ratio = 0.0f64;
    let mut wt1_worst = 0usize;
    let mut wt2_max_ratio = 0.0f64;
    let mut wt2_worst = 0usize;
    let mut hist_max_ratio = 0.0f64;
    let mut hist_worst = 0usize;

    for idx in 0..total {
        let wt1_dev = wt1_gpu[idx] as f64;
        let wt2_dev = wt2_gpu[idx] as f64;
        let hist_dev = hist_gpu[idx] as f64;
        let wt1_cpu = cpu_wt1_f32[idx] as f64;
        let wt2_cpu = cpu_wt2_f32[idx] as f64;
        let hist_cpu = cpu_hist_f32[idx] as f64;

        let r1 = approx_ratio(wt1_cpu, wt1_dev, tol);
        if r1 > wt1_max_ratio {
            wt1_max_ratio = r1;
            wt1_worst = idx;
        }
        let r2 = approx_ratio(wt2_cpu, wt2_dev, tol);
        if r2 > wt2_max_ratio {
            wt2_max_ratio = r2;
            wt2_worst = idx;
        }
        let r3 = approx_ratio(hist_cpu, hist_dev, tol);
        if r3 > hist_max_ratio {
            hist_max_ratio = r3;
            hist_worst = idx;
        }
    }

    assert!(
        wt1_max_ratio <= 1.0,
        "WT1 max mismatch at {}: cpu={} gpu={} ratio={} tol={}",
        wt1_worst,
        cpu_wt1_f32[wt1_worst] as f64,
        wt1_gpu[wt1_worst] as f64,
        wt1_max_ratio,
        tol
    );
    assert!(
        wt2_max_ratio <= 1.0,
        "WT2 max mismatch at {}: cpu={} gpu={} ratio={} tol={}",
        wt2_worst,
        cpu_wt2_f32[wt2_worst] as f64,
        wt2_gpu[wt2_worst] as f64,
        wt2_max_ratio,
        tol
    );
    assert!(
        hist_max_ratio <= 1.0,
        "Hist max mismatch at {} (t={} series={}): cpu={} gpu={} ratio={} tol={} (cpu_wt1={} cpu_wt2={} cpu_hist_calc={} cpu_wt2_sma4={} gpu_wt1={} gpu_wt2={} gpu_hist_calc={} gpu_wt2_sma4={})",
        hist_worst,
        hist_worst / cols,
        hist_worst % cols,
        cpu_hist_f32[hist_worst] as f64,
        hist_gpu[hist_worst] as f64,
        hist_max_ratio,
        tol,
        cpu_wt1_f32[hist_worst] as f64,
        cpu_wt2_f32[hist_worst] as f64,
        (cpu_wt1_f32[hist_worst] - cpu_wt2_f32[hist_worst]) as f64,
        {
            let t = hist_worst / cols;
            let s = hist_worst % cols;
            let mut sum = 0.0f32;
            for dt in 0..4 {
                sum += cpu_wt1_f32[(t - 3 + dt) * cols + s];
            }
            0.25f32 * sum
        } as f64,
        wt1_gpu[hist_worst] as f64,
        wt2_gpu[hist_worst] as f64,
        (wt1_gpu[hist_worst] - wt2_gpu[hist_worst]) as f64,
        {
            let t = hist_worst / cols;
            let s = hist_worst % cols;
            let mut sum = 0.0f32;
            for dt in 0..4 {
                sum += wt1_gpu[(t - 3 + dt) * cols + s];
            }
            0.25f32 * sum
        } as f64
    );

    Ok(())
}
