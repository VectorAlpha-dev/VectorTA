// Integration tests for CUDA WaveTrend kernels

use my_project::indicators::wavetrend::{
    wavetrend_batch_with_kernel, WavetrendBatchRange, WavetrendParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer};
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::wavetrend::CudaWavetrend;

fn approx_eq(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    if diff <= atol {
        return true;
    }
    diff <= rtol * a.abs().max(b.abs())
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
fn wavetrend_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[wavetrend_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 6..series_len {
        let x = i as f64;
        data[i] = (x * 0.0012).sin() * 1.5 + (x * 0.0007).cos();
    }

    let sweep = WavetrendBatchRange {
        channel_length: (6, 12, 3),
        average_length: (8, 14, 3),
        ma_length: (3, 5, 1),
        factor: (0.010, 0.020, 0.005),
    };

    let cpu = wavetrend_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cpu_wt1_f32: Vec<f32> = cpu.wt1.iter().map(|&x| x as f32).collect();
    let cpu_wt2_f32: Vec<f32> = cpu.wt2.iter().map(|&x| x as f32).collect();
    let cpu_diff_f32: Vec<f32> = cpu.wt_diff.iter().map(|&x| x as f32).collect();

    let cuda = CudaWavetrend::new(0).expect("CudaWavetrend::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let batch = cuda
        .wavetrend_batch_dev(&data_f32, &sweep)
        .expect("cuda wavetrend_batch_dev");

    assert_eq!(cpu.rows, batch.wt1.rows);
    assert_eq!(cpu.cols, batch.wt1.cols);

    let mut gpu_wt1 = vec![0f32; batch.wt1.len()];
    let mut gpu_wt2 = vec![0f32; batch.wt2.len()];
    let mut gpu_diff = vec![0f32; batch.wt_diff.len()];
    batch
        .wt1
        .buf
        .copy_to(&mut gpu_wt1)
        .expect("copy wt1 to host");
    batch
        .wt2
        .buf
        .copy_to(&mut gpu_wt2)
        .expect("copy wt2 to host");
    batch
        .wt_diff
        .buf
        .copy_to(&mut gpu_diff)
        .expect("copy wt_diff to host");

    let tol = 1.5;
    let rtol = 0.1;
    for idx in 0..gpu_wt1.len() {
        let a = cpu_wt1_f32[idx] as f64;
        let b = gpu_wt1[idx] as f64;
        assert!(
            approx_eq(a, b, tol, rtol) || (a.is_nan() && b.is_nan()),
            "WT1 mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }
    for idx in 0..gpu_wt2.len() {
        let a = cpu_wt2_f32[idx] as f64;
        let b = gpu_wt2[idx] as f64;
        assert!(
            approx_eq(a, b, tol, rtol) || (a.is_nan() && b.is_nan()),
            "WT2 mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }
    for idx in 0..gpu_diff.len() {
        let a = cpu_diff_f32[idx] as f64;
        let b = gpu_diff[idx] as f64;
        assert!(
            approx_eq(a, b, tol, rtol) || (a.is_nan() && b.is_nan()),
            "WT diff mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn wavetrend_cuda_batch_device_reuse_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[wavetrend_cuda_batch_device_reuse_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 3072usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 8..series_len {
        let x = i as f64;
        data[i] = (x * 0.0009).sin() + 0.002 * x.cos();
    }

    let sweep = WavetrendBatchRange {
        channel_length: (5, 11, 3),
        average_length: (7, 13, 3),
        ma_length: (2, 4, 1),
        factor: (0.010, 0.015, 0.0025),
    };

    let cpu = wavetrend_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let combos = {
        let mut out = Vec::new();
        for ch in
            (sweep.channel_length.0..=sweep.channel_length.1).step_by(sweep.channel_length.2.max(1))
        {
            for avg in (sweep.average_length.0..=sweep.average_length.1)
                .step_by(sweep.average_length.2.max(1))
            {
                for ma in (sweep.ma_length.0..=sweep.ma_length.1).step_by(sweep.ma_length.2.max(1))
                {
                    let mut factor = sweep.factor.0;
                    while factor <= sweep.factor.1 + 1e-12 {
                        out.push(WavetrendParams {
                            channel_length: Some(ch),
                            average_length: Some(avg),
                            ma_length: Some(ma),
                            factor: Some(factor),
                        });
                        factor += sweep.factor.2;
                        if sweep.factor.2.abs() < 1e-12 {
                            break;
                        }
                    }
                }
            }
        }
        out
    };

    let first_valid = data
        .iter()
        .position(|x| !x.is_nan())
        .expect("non-NaN value expected");

    let cuda = CudaWavetrend::new(0).expect("CudaWavetrend::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let d_prices = DeviceBuffer::from_slice(&data_f32).expect("upload prices");
    let mut d_wt1: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(combos.len() * series_len) }
            .expect("allocate wt1 buffer");
    let mut d_wt2: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(combos.len() * series_len) }
            .expect("allocate wt2 buffer");
    let mut d_wt_diff: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(combos.len() * series_len) }
            .expect("allocate diff buffer");

    cuda.wavetrend_batch_device(
        &d_prices,
        &combos,
        first_valid,
        series_len,
        &mut d_wt1,
        &mut d_wt2,
        &mut d_wt_diff,
    )
    .expect("cuda wavetrend_batch_device");

    let mut gpu_wt1 = vec![0f32; combos.len() * series_len];
    let mut gpu_wt2 = vec![0f32; combos.len() * series_len];
    let mut gpu_diff = vec![0f32; combos.len() * series_len];
    d_wt1.copy_to(&mut gpu_wt1).expect("copy wt1");
    d_wt2.copy_to(&mut gpu_wt2).expect("copy wt2");
    d_wt_diff.copy_to(&mut gpu_diff).expect("copy wt_diff");

    let tol = 2e-2;
    let rtol = 5e-4;
    for idx in 0..gpu_wt1.len() {
        let a = cpu.wt1[idx];
        let b = gpu_wt1[idx] as f64;
        assert!(
            approx_eq(a, b, tol, rtol) || (a.is_nan() && b.is_nan()),
            "WT1 mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }
    for idx in 0..gpu_wt2.len() {
        let a = cpu.wt2[idx];
        let b = gpu_wt2[idx] as f64;
        assert!(
            approx_eq(a, b, tol, rtol) || (a.is_nan() && b.is_nan()),
            "WT2 mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }
    for idx in 0..gpu_diff.len() {
        let a = cpu.wt_diff[idx];
        let b = gpu_diff[idx] as f64;
        assert!(
            approx_eq(a, b, tol, rtol) || (a.is_nan() && b.is_nan()),
            "WT diff mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}
