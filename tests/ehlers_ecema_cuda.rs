

use vector_ta::indicators::moving_averages::ehlers_ecema::{
    ehlers_ecema_batch_with_kernel, ehlers_ecema_with_kernel, EhlersEcemaBatchRange,
    EhlersEcemaInput, EhlersEcemaParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaEhlersEcema;

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
fn ehlers_ecema_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ehlers_ecema_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 15..len {
        let t = i as f64;
        data[i] = (t * 0.0027).sin() + 0.00031 * t + (t * 0.0043).cos() * 0.2;
    }

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let data_quant: Vec<f64> = data_f32.iter().map(|&v| v as f64).collect();

    let sweep = EhlersEcemaBatchRange {
        length: (6, 30, 6),
        gain_limit: (10, 50, 20),
    };

    let cpu = ehlers_ecema_batch_with_kernel(&data_quant, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaEhlersEcema::new(0)?;
    let params = EhlersEcemaParams {
        length: None,
        gain_limit: None,
        pine_compatible: Some(false),
        confirmed_only: Some(false),
    };
    let handle = cuda.ehlers_ecema_batch_dev(&data_f32, &sweep, &params)?;

    assert_eq!(handle.rows, cpu.rows);
    assert_eq!(handle.cols, cpu.cols);

    let mut gpu_host = vec![0f32; handle.len()];
    handle.buf.copy_to(&mut gpu_host)?;

    
    let tol = 1e-4;
    for idx in 0..gpu_host.len() {
        let cpu_v = cpu.values[idx];
        let gpu_v = gpu_host[idx] as f64;
        assert!(
            approx_eq(cpu_v, gpu_v, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_v,
            gpu_v
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ehlers_ecema_cuda_batch_pine_confirmed_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ehlers_ecema_cuda_batch_pine_confirmed_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut data = vec![f64::NAN; len];
    for i in 8..len {
        let t = i as f64;
        data[i] = (t * 0.003).sin() * 0.7 + (t * 0.0017).cos() * 0.4 + 0.0005 * t;
    }

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let data_quant: Vec<f64> = data_f32.iter().map(|&v| v as f64).collect();

    let sweep = EhlersEcemaBatchRange {
        length: (10, 18, 4),
        gain_limit: (20, 40, 10),
    };

    
    let lengths: Vec<usize> = (sweep.length.0..=sweep.length.1)
        .step_by(sweep.length.2.max(1))
        .collect();
    let gains: Vec<usize> = (sweep.gain_limit.0..=sweep.gain_limit.1)
        .step_by(sweep.gain_limit.2.max(1))
        .collect();

    let rows = lengths.len() * gains.len();
    let cols = len;
    let mut cpu = vec![f64::NAN; rows * cols];

    let mut row = 0usize;
    for &length in &lengths {
        for &gain in &gains {
            let params = EhlersEcemaParams {
                length: Some(length),
                gain_limit: Some(gain),
                pine_compatible: Some(true),
                confirmed_only: Some(true),
            };
            let input = EhlersEcemaInput::from_slice(&data_quant, params);
            let out = ehlers_ecema_with_kernel(&input, Kernel::Scalar)?;
            cpu[row * cols..(row + 1) * cols].copy_from_slice(&out.values);
            row += 1;
        }
    }

    let cuda = CudaEhlersEcema::new(0)?;
    let params = EhlersEcemaParams {
        length: None,
        gain_limit: None,
        pine_compatible: Some(true),
        confirmed_only: Some(true),
    };
    let handle = cuda.ehlers_ecema_batch_dev(&data_f32, &sweep, &params)?;

    assert_eq!(handle.rows, rows);
    assert_eq!(handle.cols, cols);

    let mut gpu_host = vec![0f32; handle.len()];
    handle.buf.copy_to(&mut gpu_host)?;

    let tol = 3e-5;
    for idx in 0..gpu_host.len() {
        let cpu_v = cpu[idx];
        let gpu_v = gpu_host[idx] as f64;
        assert!(
            approx_eq(cpu_v, gpu_v, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_v,
            gpu_v
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ehlers_ecema_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ehlers_ecema_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 5usize;
    let rows = 1536usize;
    let mut data_tm = vec![f64::NAN; cols * rows];
    for series in 0..cols {
        for t in (series + 6)..rows {
            let time = t as f64;
            let base = (time * 0.0023 + series as f64 * 0.41).sin();
            let modulated = (time * 0.0009).cos() * (0.2 + 0.05 * series as f64);
            data_tm[t * cols + series] = base + modulated + 0.0007 * time;
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let data_tm_quant: Vec<f64> = data_tm_f32.iter().map(|&v| v as f64).collect();

    let scenarios = [
        (false, false, 24usize, 40usize),
        (true, true, 18usize, 30usize),
    ];

    let cuda = CudaEhlersEcema::new(0)?;
    let tol = 3e-5;

    for &(pine, confirmed, length, gain_limit) in &scenarios {
        let params = EhlersEcemaParams {
            length: Some(length),
            gain_limit: Some(gain_limit),
            pine_compatible: Some(pine),
            confirmed_only: Some(confirmed),
        };

        
        let mut cpu_tm = vec![f64::NAN; cols * rows];
        for series in 0..cols {
            let mut series_data = vec![f64::NAN; rows];
            for t in 0..rows {
                series_data[t] = data_tm_quant[t * cols + series];
            }
            let input = EhlersEcemaInput::from_slice(&series_data, params.clone());
            let out = ehlers_ecema_with_kernel(&input, Kernel::Scalar)?;
            for t in 0..rows {
                cpu_tm[t * cols + series] = out.values[t];
            }
        }

        let handle = cuda.ehlers_ecema_many_series_one_param_time_major_dev(
            &data_tm_f32,
            cols,
            rows,
            &params,
        )?;

        assert_eq!(handle.rows, rows);
        assert_eq!(handle.cols, cols);

        let mut gpu_tm = vec![0f32; handle.len()];
        handle.buf.copy_to(&mut gpu_tm)?;

        for idx in 0..gpu_tm.len() {
            let cpu_v = cpu_tm[idx];
            let gpu_v = gpu_tm[idx] as f64;
            assert!(
                approx_eq(cpu_v, gpu_v, tol),
                "mismatch at scenario {:?} idx {}: cpu={} gpu={}",
                (pine, confirmed, length, gain_limit),
                idx,
                cpu_v,
                gpu_v
            );
        }
    }

    Ok(())
}
