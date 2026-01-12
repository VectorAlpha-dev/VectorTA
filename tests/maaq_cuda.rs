use vector_ta::indicators::moving_averages::maaq::{
    maaq_batch_with_kernel, MaaqBatchRange, MaaqBuilder, MaaqParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaMaaq;

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
fn maaq_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[maaq_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 24..series_len {
        let x = i as f64;
        data[i] = (x * 0.0019).sin() + 0.00035 * x.cos();
    }

    let sweep = MaaqBatchRange {
        period: (5, 55, 5),
        fast_period: (2, 6, 2),
        slow_period: (20, 60, 10),
    };

    let cpu = maaq_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaMaaq::new(0).expect("CudaMaaq::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu = cuda
        .maaq_batch_dev(&data_f32, &sweep)
        .expect("cuda maaq_batch_dev");

    assert_eq!(cpu.rows, gpu.rows);
    assert_eq!(cpu.cols, gpu.cols);

    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda maaq batch result");

    let tol = 5e-4f64;
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
fn maaq_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[maaq_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 4usize;
    let series_len = 1024usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];

    for series in 0..num_series {
        let offset = 8 + series;
        for t in offset..series_len {
            let idx = t * num_series + series;
            let x = (t as f64) + (series as f64) * 0.37;
            data_tm[idx] = (x * 0.0015).sin() + 0.00045 * x.cos();
        }
    }

    let period = 28usize;
    let fast_period = 4usize;
    let slow_period = 40usize;

    let builder = MaaqBuilder::new()
        .period(period)
        .fast_period(fast_period)
        .slow_period(slow_period)
        .kernel(Kernel::Scalar);

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut series_vec = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series_vec[t] = data_tm[t * num_series + series];
        }
        let out = builder
            .apply_slice(&series_vec)
            .expect("maaq builder cpu result");
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = out.values[t];
        }
    }

    let cuda = CudaMaaq::new(0).expect("CudaMaaq::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let params = MaaqParams {
        period: Some(period),
        fast_period: Some(fast_period),
        slow_period: Some(slow_period),
    };
    let gpu = cuda
        .maaq_multi_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)
        .expect("cuda maaq multi-series result");

    assert_eq!(gpu.rows, series_len);
    assert_eq!(gpu.cols, num_series);

    let mut gpu_tm = vec![0f32; gpu.len()];
    gpu.buf
        .copy_to(&mut gpu_tm)
        .expect("copy cuda maaq multi-series result");

    let tol = 5e-4f64;
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
