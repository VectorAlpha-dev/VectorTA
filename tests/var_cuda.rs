

use vector_ta::indicators::var::{var_batch_with_kernel, VarBatchRange, VarBuilder, VarParams};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaVar;

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
fn var_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[var_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 10..len {
        let x = i as f64;
        let base = (x * 0.00037).sin() - (x * 0.00021).cos();
        data[i] = base + 0.0011 * ((i % 11) as f64 - 5.0);
    }

    let sweep = VarBatchRange {
        period: (10, 40, 10),
        nbdev: (1.0, 2.0, 1.0),
    };
    let cpu = var_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaVar::new(0).expect("CudaVar::new");
    let (dev_arr, combos) = cuda
        .var_batch_dev(&data_f32, &sweep)
        .expect("var_cuda_batch_dev");

    assert_eq!(dev_arr.rows, cpu.rows);
    assert_eq!(dev_arr.cols, cpu.cols);
    assert_eq!(combos.len(), cpu.combos.len());
    for (c, p) in combos.iter().zip(cpu.combos.iter()) {
        assert_eq!(c.period.unwrap(), p.period.unwrap());
        assert!((c.nbdev.unwrap() - p.nbdev.unwrap()).abs() < 1e-12);
    }

    let mut gpu = vec![0f32; dev_arr.rows * dev_arr.cols];
    dev_arr.buf.copy_to(&mut gpu).expect("copy results");

    let tol = 5e-4;
    for (idx, (&cpu_v, &gpu_v)) in cpu.values.iter().zip(gpu.iter()).enumerate() {
        assert!(
            approx_eq(cpu_v, gpu_v as f64, tol),
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
fn var_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[var_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let rows = 2048usize;
    let cols = 6usize;
    let mut data_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        for t in (s + 5)..rows {
            let x = t as f64 + (s as f64) * 0.13;
            data_tm[t * cols + s] = (x * 0.0027).cos() + 0.0009 * x;
        }
    }
    let period = 14usize;
    let nbdev = 1.5f64;
    let params = VarParams {
        period: Some(period),
        nbdev: Some(nbdev),
    };

    
    let mut cpu = vec![f32::NAN; rows * cols];
    for s in 0..cols {
        let mut col = vec![f64::NAN; rows];
        for t in 0..rows {
            col[t] = data_tm[t * cols + s];
        }
        let out = VarBuilder::new()
            .period(period)
            .nbdev(nbdev)
            .apply_slice(&col)?
            .values;
        for t in 0..rows {
            cpu[t * cols + s] = out[t] as f32;
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaVar::new(0).expect("CudaVar::new");
    let dev = cuda
        .var_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("var many-series");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut gpu = vec![0f32; rows * cols];
    dev.buf.copy_to(&mut gpu).expect("copy many-series");

    let tol = 5e-4;
    for i in 0..(rows * cols) {
        assert!(
            approx_eq(cpu[i] as f64, gpu[i] as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            i,
            cpu[i],
            gpu[i]
        );
    }
    Ok(())
}
