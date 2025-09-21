// CUDA integration tests for the Zscore indicator (SMA/stddev path).

use my_project::indicators::zscore::{zscore_batch_with_kernel, ZscoreBatchRange};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaZscore;

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
fn zscore_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[zscore_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        let base = (x * 0.00037).sin() + (x * 0.00021).cos();
        data[i] = base + 0.001 * (i % 7) as f64;
        if i % 251 == 0 {
            data[i] = f64::NAN;
        }
    }

    let sweep = ZscoreBatchRange {
        period: (10, 40, 10),
        ma_type: ("sma".to_string(), "sma".to_string(), "".to_string()),
        nbdev: (0.5, 1.5, 0.5),
        devtype: (0, 0, 0),
    };

    let cpu = zscore_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();

    let cuda = CudaZscore::new(0).expect("CudaZscore::new");
    let (dev_arr, combos_meta) = cuda
        .zscore_batch_dev(&data_f32, &sweep)
        .expect("zscore_cuda_batch_dev");

    assert_eq!(dev_arr.rows, cpu.rows);
    assert_eq!(dev_arr.cols, cpu.cols);
    assert_eq!(combos_meta.len(), cpu.combos.len());

    for (combo, params) in combos_meta.iter().zip(cpu.combos.iter()) {
        assert_eq!(combo.0, params.period.unwrap());
        assert!((combo.1 as f64 - params.nbdev.unwrap()).abs() < 1e-6);
        assert_eq!(params.ma_type.as_ref().unwrap(), "sma");
        assert_eq!(params.devtype.unwrap(), 0);
    }

    let mut gpu = vec![0f32; dev_arr.len()];
    dev_arr
        .buf
        .copy_to(&mut gpu)
        .expect("copy zscore cuda results");

    let tol = 5e-4;
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
fn zscore_cuda_host_copy_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[zscore_cuda_host_copy_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut data = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        let base = (x * 0.00051).cos() - (x * 0.00013).sin();
        data[i] = base + 0.0007 * ((i % 11) as f64 - 5.0);
        if i % 199 == 0 {
            data[i] = f64::NAN;
        }
    }

    let sweep = ZscoreBatchRange {
        period: (8, 16, 4),
        ma_type: ("sma".to_string(), "sma".to_string(), "".to_string()),
        nbdev: (0.25, 1.0, 0.25),
        devtype: (0, 0, 0),
    };

    let cpu = zscore_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();

    let cuda = CudaZscore::new(0).expect("CudaZscore::new");
    let mut gpu = vec![0f32; cpu.values.len()];
    let (rows, cols, combos_meta) = cuda
        .zscore_batch_into_host_f32(&data_f32, &sweep, &mut gpu)
        .expect("zscore_cuda_batch_into_host_f32");

    assert_eq!(rows, cpu.rows);
    assert_eq!(cols, cpu.cols);
    assert_eq!(combos_meta.len(), cpu.combos.len());

    for (combo, params) in combos_meta.iter().zip(cpu.combos.iter()) {
        assert_eq!(combo.0, params.period.unwrap());
        assert!((combo.1 as f64 - params.nbdev.unwrap()).abs() < 1e-6);
    }

    let tol = 5e-4;
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
