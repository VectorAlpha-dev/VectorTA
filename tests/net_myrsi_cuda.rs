// CUDA integration tests for NET MyRSI

use my_project::indicators::net_myrsi::{
    net_myrsi_batch_with_kernel, net_myrsi_with_kernel, NetMyrsiBatchRange, NetMyrsiInput,
    NetMyrsiParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaNetMyrsi;

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
fn net_myrsi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[net_myrsi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut data = vec![f64::NAN; len];
    for i in 6..len {
        let x = i as f64;
        // Inject a few NaNs to exercise handling
        data[i] = (x * 0.00071).sin() + 0.0003 * (i % 11) as f64;
        if i % 997 == 0 {
            data[i] = f64::NAN;
        }
    }

    let sweep = NetMyrsiBatchRange {
        period: (10, 50, 10),
    };
    let cpu = net_myrsi_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaNetMyrsi::new(0).expect("CudaNetMyrsi::new");
    let (dev, combos) = cuda
        .net_myrsi_batch_dev(&data_f32, &sweep)
        .expect("net_myrsi_cuda_batch_dev");

    assert_eq!(dev.rows, cpu.rows);
    assert_eq!(dev.cols, cpu.cols);
    assert_eq!(combos.len(), cpu.combos.len());
    for (c, p) in combos.iter().zip(cpu.combos.iter()) {
        assert_eq!(c.period.unwrap(), p.period.unwrap());
    }

    let mut gpu = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu)?;

    let tol = 1e-3; // recurrence path; small FP32 differences are expected
    for (idx, (&a, &b)) in cpu.values.iter().zip(gpu.iter()).enumerate() {
        assert!(
            approx_eq(a, b as f64, tol),
            "mismatch at {}: {} vs {}",
            idx,
            a,
            b
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn net_myrsi_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[net_myrsi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize;
    let rows = 4096usize;
    let mut data_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for r in s..rows {
            let idx = r * cols + s;
            let x = r as f64 + 0.1 * s as f64;
            data_tm[idx] = (x * 0.00123).cos() + 0.00017 * x;
        }
    }

    // CPU per-series baseline
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    let p = NetMyrsiParams { period: Some(14) };
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for r in 0..rows {
            series[r] = data_tm[r * cols + s];
        }
        let out = net_myrsi_with_kernel(
            &NetMyrsiInput::from_slice(&series, p.clone()),
            Kernel::Scalar,
        )?
        .values;
        for r in 0..rows {
            cpu_tm[r * cols + s] = out[r];
        }
    }

    let data_tm32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaNetMyrsi::new(0).expect("CudaNetMyrsi::new");
    let dev = cuda
        .net_myrsi_many_series_one_param_time_major_dev(&data_tm32, cols, rows, &p)
        .expect("net_myrsi_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_tm)?;

    let tol = 1e-3;
    for idx in 0..gpu_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], gpu_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
