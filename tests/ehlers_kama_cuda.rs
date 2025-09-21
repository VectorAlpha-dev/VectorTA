// Integration tests for CUDA Ehlers KAMA kernels

use my_project::indicators::moving_averages::ehlers_kama::{
    ehlers_kama_batch_with_kernel, EhlersKamaBatchRange, EhlersKamaBuilder, EhlersKamaParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaEhlersKama;

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
fn ehlers_kama_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ehlers_kama_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 32..series_len {
        let x = i as f64;
        data[i] = (x * 0.0023).sin() + 0.0007 * x;
    }

    let sweep = EhlersKamaBatchRange {
        period: (5, 120, 5),
    };

    let cpu = ehlers_kama_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaEhlersKama::new(0).expect("CudaEhlersKama::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu = cuda
        .ehlers_kama_batch_dev(&data_f32, &sweep)
        .expect("cuda ehlers_kama_batch_dev");

    assert_eq!(cpu.rows, gpu.rows);
    assert_eq!(cpu.cols, gpu.cols);

    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda ehlers_kama batch result");

    let tol = 4.0e-4f64;
    for idx in 0..(cpu.rows * cpu.cols) {
        let a = cpu.values[idx];
        let b = gpu_host[idx] as f64;
        assert!(
            approx_eq(a, b, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ehlers_kama_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ehlers_kama_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 5usize;
    let series_len = 1536usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];

    for j in 0..num_series {
        for t in (j * 4)..series_len {
            let idx = t * num_series + j;
            let x = (t as f64) * 0.75 + (j as f64) * 0.19;
            data_tm[idx] = (x * 0.0019).cos() + 0.0005 * x;
        }
    }

    let period = 34;
    let params = EhlersKamaParams {
        period: Some(period),
    };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        let mut series = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series[t] = data_tm[t * num_series + j];
        }
        let out = EhlersKamaBuilder::new()
            .period(period)
            .apply_slice(&series)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + j] = out.values[t];
        }
    }

    let cuda = CudaEhlersKama::new(0).expect("CudaEhlersKama::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let gpu = cuda
        .ehlers_kama_multi_series_one_param_time_major_dev(
            &data_tm_f32,
            num_series,
            series_len,
            &params,
        )
        .expect("cuda ehlers_kama_many_series_one_param_dev");

    assert_eq!(gpu.rows, series_len);
    assert_eq!(gpu.cols, num_series);

    let mut gpu_tm = vec![0f32; gpu.len()];
    gpu.buf
        .copy_to(&mut gpu_tm)
        .expect("copy cuda ehlers_kama many-series result");

    let tol = 4.0e-4f64;
    for idx in 0..(num_series * series_len) {
        let a = cpu_tm[idx];
        let b = gpu_tm[idx] as f64;
        assert!(
            approx_eq(a, b, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}
