use vector_ta::indicators::net_myrsi::{
    net_myrsi_batch_with_kernel, net_myrsi_with_kernel, NetMyrsiBatchRange, NetMyrsiInput,
    NetMyrsiParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{CudaNetMyrsi, CudaRuntime};

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

    let tol = 1e-3;
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

    let data_tm32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    let p = NetMyrsiParams { period: Some(14) };
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for r in 0..rows {
            series[r] = data_tm32[r * cols + s] as f64;
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
        let c = cpu_tm[idx];
        let g = gpu_tm[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "mismatch at {}: cpu={} gpu={} diff={}",
            idx,
            c,
            g,
            (c - g).abs()
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn net_myrsi_cuda_device_prices_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[net_myrsi_cuda_device_prices_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f32::NAN; len];
    let first_valid = 6usize;
    for (i, value) in data.iter_mut().enumerate().skip(first_valid) {
        let x = i as f32;
        *value = (x * 0.00071).sin() + 0.0003 * (i % 11) as f32;
        if i % 997 == 0 {
            *value = f32::NAN;
        }
    }

    let sweep = NetMyrsiBatchRange {
        period: (10, 50, 10),
    };

    let cuda = CudaNetMyrsi::new(0).expect("CudaNetMyrsi::new");
    let (legacy_dev, legacy_combos) = cuda.net_myrsi_batch_dev(&data, &sweep).expect("legacy");

    let runtime = CudaRuntime::new(0).expect("CudaRuntime::new");
    let device_prices = runtime.upload_f32(&data).expect("upload prices");
    let (device_dev, device_combos) = cuda
        .net_myrsi_batch_dev_from_device_prices(device_prices.buffer(), len, first_valid, &sweep)
        .expect("device inputs");

    assert_eq!(legacy_combos.len(), device_combos.len());
    for (legacy, device) in legacy_combos.iter().zip(device_combos.iter()) {
        assert_eq!(legacy.period, device.period);
    }
    assert_eq!(legacy_dev.rows, device_dev.rows);
    assert_eq!(legacy_dev.cols, device_dev.cols);

    let mut legacy = vec![0.0f32; legacy_dev.len()];
    legacy_dev.buf.copy_to(legacy.as_mut_slice())?;
    let mut device = vec![0.0f32; device_dev.len()];
    device_dev.buf.copy_to(device.as_mut_slice())?;

    for (idx, (lhs, rhs)) in legacy.iter().zip(device.iter()).enumerate() {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!(
            (lhs - rhs).abs() < 1e-6,
            "mismatch at {idx}: lhs={lhs} rhs={rhs}"
        );
    }
    Ok(())
}
