#![cfg(feature = "cuda")]

use cust::memory::CopyDestination;
use vector_ta::cuda::cuda_available;
use vector_ta::cuda::moving_averages::ehlers_ecema_wrapper::{
    BatchKernelPolicy, BatchThreadsPerOutput, CudaEhlersEcemaPolicy, ManySeriesKernelPolicy,
};
use vector_ta::cuda::moving_averages::CudaEhlersEcema;
use vector_ta::indicators::moving_averages::ehlers_ecema::{
    ehlers_ecema_batch_with_kernel, ehlers_ecema_with_kernel, EhlersEcemaBatchRange,
    EhlersEcemaInput, EhlersEcemaParams,
};
use vector_ta::utilities::enums::Kernel;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

fn gen_series(n: usize, offset: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; n];
    for i in offset..n {
        let t = i as f64;
        v[i] = (t * 0.0027).sin() + 0.00031 * t + (t * 0.0043).cos() * 0.2;
    }
    v
}

#[test]
fn ecema_cuda_variants_batch_plain_and_tiled_match_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ecema variants] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let data = gen_series(len, 20);
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let data_quant: Vec<f64> = data_f32.iter().map(|&x| x as f64).collect();

    let sweep = EhlersEcemaBatchRange {
        length: (6, 30, 6),
        gain_limit: (10, 50, 20),
    };
    let cpu = ehlers_ecema_batch_with_kernel(&data_quant, &sweep, Kernel::ScalarBatch)?;

    let mut cuda_plain = CudaEhlersEcema::new_with_policy(
        0,
        CudaEhlersEcemaPolicy {
            batch: BatchKernelPolicy::Plain { block_x: 1 },
            many_series: ManySeriesKernelPolicy::Auto,
        },
    )?;
    let params = EhlersEcemaParams {
        length: None,
        gain_limit: None,
        pine_compatible: Some(false),
        confirmed_only: Some(false),
    };
    let handle_plain = cuda_plain.ehlers_ecema_batch_dev(&data_f32, &sweep, &params)?;
    let mut gpu_plain = vec![0f32; handle_plain.len()];
    handle_plain.buf.copy_to(&mut gpu_plain)?;

    let mut cuda_tiled = CudaEhlersEcema::new_with_policy(
        0,
        CudaEhlersEcemaPolicy {
            batch: BatchKernelPolicy::Tiled {
                tile: 128,
                per_thread: BatchThreadsPerOutput::One,
            },
            many_series: ManySeriesKernelPolicy::Auto,
        },
    )?;
    let handle_tiled = cuda_tiled.ehlers_ecema_batch_dev(&data_f32, &sweep, &params)?;
    let mut gpu_tiled = vec![0f32; handle_tiled.len()];
    handle_tiled.buf.copy_to(&mut gpu_tiled)?;

    assert_eq!(cpu.rows, handle_plain.rows);
    assert_eq!(cpu.rows, handle_tiled.rows);
    assert_eq!(cpu.cols, handle_plain.cols);
    assert_eq!(cpu.cols, handle_tiled.cols);

    let tol_plain = 3e-5;
    let tol_tiled = 1e-4;
    for i in 0..cpu.values.len() {
        let c = cpu.values[i];
        let p = gpu_plain[i] as f64;
        let t = gpu_tiled[i] as f64;
        assert!(
            approx_eq(c, p, tol_plain),
            "plain mismatch at {}: {} vs {}",
            i,
            c,
            p
        );
        assert!(
            approx_eq(c, t, tol_tiled),
            "tiled mismatch at {}: {} vs {}",
            i,
            c,
            t
        );
    }

    Ok(())
}

#[test]
fn ecema_cuda_variants_many_series_1d_and_2d_match_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ecema variants] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 7usize;
    let rows = 2048usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s + 8)..rows {
            let time = t as f64;
            tm[t * cols + s] = (time * 0.0023 + s as f64 * 0.41).sin()
                + (time * 0.0009).cos() * (0.2 + 0.05 * s as f64)
                + 0.0007 * time;
        }
    }
    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let tm_quant: Vec<f64> = tm_f32.iter().map(|&v| v as f64).collect();

    let params = EhlersEcemaParams {
        length: Some(24),
        gain_limit: Some(40),
        pine_compatible: Some(false),
        confirmed_only: Some(false),
    };

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = tm_quant[t * cols + s];
        }
        let input = EhlersEcemaInput::from_slice(&series, params.clone());
        let out = ehlers_ecema_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let mut cuda_1d = CudaEhlersEcema::new_with_policy(
        0,
        CudaEhlersEcemaPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::OneD { block_x: 128 },
        },
    )?;
    let handle_1d =
        cuda_1d.ehlers_ecema_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)?;
    let mut gpu_1d = vec![0f32; handle_1d.len()];
    handle_1d.buf.copy_to(&mut gpu_1d)?;

    let mut cuda_2d = CudaEhlersEcema::new_with_policy(
        0,
        CudaEhlersEcemaPolicy {
            batch: BatchKernelPolicy::Auto,
            many_series: ManySeriesKernelPolicy::Tiled2D { tx: 128, ty: 2 },
        },
    )?;
    let handle_2d =
        cuda_2d.ehlers_ecema_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)?;
    let mut gpu_2d = vec![0f32; handle_2d.len()];
    handle_2d.buf.copy_to(&mut gpu_2d)?;

    assert_eq!(handle_1d.rows, rows);
    assert_eq!(handle_2d.rows, rows);
    assert_eq!(handle_1d.cols, cols);
    assert_eq!(handle_2d.cols, cols);

    let tol = 3e-5;
    for i in 0..cpu_tm.len() {
        let c = cpu_tm[i];
        let a = gpu_1d[i] as f64;
        let b = gpu_2d[i] as f64;
        assert!(approx_eq(c, a, tol), "1D mismatch at {}: {} vs {}", i, c, a);
        assert!(approx_eq(c, b, tol), "2D mismatch at {}: {} vs {}", i, c, b);
    }

    Ok(())
}
