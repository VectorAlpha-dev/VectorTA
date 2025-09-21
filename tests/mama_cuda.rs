// Integration tests for CUDA MAMA kernels

use my_project::indicators::moving_averages::mama::{
    mama_batch_with_kernel, MamaBatchRange, MamaBuilder,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::{CudaMama, DeviceMamaPair};

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
fn mama_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mama_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![0.0f64; series_len];
    for i in 0..series_len {
        let x = i as f64;
        data[i] = (x * 0.0017).sin() + 0.00045 * x + (x * 0.0003).cos() * 0.3;
    }

    let sweep = MamaBatchRange {
        fast_limit: (0.25, 0.65, 0.1),
        slow_limit: (0.02, 0.08, 0.02),
    };

    let cpu = mama_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaMama::new(0).expect("CudaMama::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let pair = cuda
        .mama_batch_dev(&data_f32, &sweep)
        .expect("cuda mama_batch_dev");

    let DeviceMamaPair {
        mama: gpu_m,
        fama: gpu_f,
    } = pair;

    assert_eq!(cpu.rows, gpu_m.rows);
    assert_eq!(cpu.cols, gpu_m.cols);
    assert_eq!(cpu.rows, gpu_f.rows);
    assert_eq!(cpu.cols, gpu_f.cols);

    let mut gpu_m_host = vec![0f32; gpu_m.len()];
    let mut gpu_f_host = vec![0f32; gpu_f.len()];
    gpu_m
        .buf
        .copy_to(&mut gpu_m_host)
        .expect("copy mama results to host");
    gpu_f
        .buf
        .copy_to(&mut gpu_f_host)
        .expect("copy fama results to host");

    let tol = 1e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let cpu_m = cpu.mama_values[idx];
        let gpu_m_val = gpu_m_host[idx] as f64;
        assert!(
            approx_eq(cpu_m, gpu_m_val, tol),
            "mismatch in MAMA at {}: cpu={} gpu={}",
            idx,
            cpu_m,
            gpu_m_val
        );

        let cpu_f = cpu.fama_values[idx];
        let gpu_f_val = gpu_f_host[idx] as f64;
        assert!(
            approx_eq(cpu_f, gpu_f_val, tol),
            "mismatch in FAMA at {}: cpu={} gpu={}",
            idx,
            cpu_f,
            gpu_f_val
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn mama_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mama_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 4usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in 0..series_len {
            let x = t as f64 + (j as f64) * 0.37;
            data_tm[t * num_series + j] = (x * 0.0019).sin() + 0.0006 * x + (x * 0.0004).cos();
        }
    }

    let fast_limit = 0.45f64;
    let slow_limit = 0.06f64;

    let mut cpu_m_tm = vec![f64::NAN; num_series * series_len];
    let mut cpu_f_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        let mut series = vec![0.0f64; series_len];
        for t in 0..series_len {
            series[t] = data_tm[t * num_series + j];
        }
        let out = MamaBuilder::default()
            .fast_limit(fast_limit)
            .slow_limit(slow_limit)
            .apply_slice(&series)?;
        for t in 0..series_len {
            cpu_m_tm[t * num_series + j] = out.mama_values[t];
            cpu_f_tm[t * num_series + j] = out.fama_values[t];
        }
    }

    let cuda = CudaMama::new(0).expect("CudaMama::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let pair = cuda
        .mama_many_series_one_param_time_major_dev(
            &data_tm_f32,
            num_series,
            series_len,
            fast_limit as f32,
            slow_limit as f32,
        )
        .expect("cuda mama_many_series_one_param_time_major_dev");

    let DeviceMamaPair {
        mama: gpu_m,
        fama: gpu_f,
    } = pair;

    assert_eq!(gpu_m.rows, series_len);
    assert_eq!(gpu_m.cols, num_series);
    assert_eq!(gpu_f.rows, series_len);
    assert_eq!(gpu_f.cols, num_series);

    let mut gpu_m_tm = vec![0f32; gpu_m.len()];
    let mut gpu_f_tm = vec![0f32; gpu_f.len()];
    gpu_m
        .buf
        .copy_to(&mut gpu_m_tm)
        .expect("copy gpu mama time-major");
    gpu_f
        .buf
        .copy_to(&mut gpu_f_tm)
        .expect("copy gpu fama time-major");

    let tol = 1e-4;
    for idx in 0..(num_series * series_len) {
        let cpu_m = cpu_m_tm[idx];
        let gpu_m_val = gpu_m_tm[idx] as f64;
        assert!(
            approx_eq(cpu_m, gpu_m_val, tol),
            "MAMA mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_m,
            gpu_m_val
        );

        let cpu_f = cpu_f_tm[idx];
        let gpu_f_val = gpu_f_tm[idx] as f64;
        assert!(
            approx_eq(cpu_f, gpu_f_val, tol),
            "FAMA mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_f,
            gpu_f_val
        );
    }

    Ok(())
}
