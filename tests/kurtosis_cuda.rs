// CUDA integration tests for the Kurtosis indicator.

use my_project::indicators::kurtosis::{
    kurtosis_batch_with_kernel, kurtosis_with_kernel, KurtosisBatchRange, KurtosisInput,
    KurtosisParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaKurtosis;

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
fn kurtosis_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[kurtosis_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 6..len {
        let x = i as f64;
        // Use a higher-variance signal to avoid ill-conditioned moment cancellation in FP32.
        let base = (x * 0.2).sin() + 0.25 * (x * 0.13).cos();
        data[i] = base + 0.01 * ((i % 13) as f64 - 6.0);
        if i % 257 == 0 {
            data[i] = f64::NAN;
        }
    }

    let sweep = KurtosisBatchRange { period: (8, 32, 8) };
    let cpu = kurtosis_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();

    let cuda = CudaKurtosis::new(0).expect("CudaKurtosis::new");
    let (dev_arr, combos) = cuda
        .kurtosis_batch_dev(&data_f32, &sweep)
        .expect("kurtosis_cuda_batch_dev");

    assert_eq!(dev_arr.rows, cpu.rows);
    assert_eq!(dev_arr.cols, cpu.cols);
    assert_eq!(combos.len(), cpu.combos.len());
    for (c, p) in combos.iter().zip(cpu.combos.iter()) {
        assert_eq!(c.period.unwrap(), p.period.unwrap());
    }

    let mut gpu = vec![0f32; dev_arr.len()];
    dev_arr.buf.copy_to(&mut gpu)?;

    let tol = 1.5e-1; // FP32 output vs FP64 CPU baseline (kurtosis can be numerically sensitive)
    for (idx, (&cpu_val, &gpu_val)) in cpu.values.iter().zip(gpu.iter()).enumerate() {
        assert!(
            approx_eq(cpu_val, gpu_val as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn kurtosis_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[kurtosis_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 1024usize;
    let mut data_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.17;
            data_tm[t * cols + s] = (x * 0.2).sin() + 0.25 * (x * 0.13).cos();
        }
    }
    let period = 15usize;

    // CPU per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = data_tm[t * cols + s];
        }
        let input = KurtosisInput::from_slice(
            &series,
            KurtosisParams {
                period: Some(period),
            },
        );
        let out = kurtosis_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaKurtosis::new(0).expect("CudaKurtosis::new");
    let dev = cuda
        .kurtosis_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, period)
        .expect("kurtosis_many_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_tm)?;

    let tol = 1.5e-1;
    for i in 0..gpu_tm.len() {
        let a = cpu_tm[i];
        let b = gpu_tm[i] as f64;
        assert!(
            approx_eq(a, b, tol),
            "mismatch at {}: cpu={} gpu={}",
            i,
            a,
            b
        );
    }
    Ok(())
}
