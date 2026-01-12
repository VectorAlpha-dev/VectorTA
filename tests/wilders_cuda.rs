use vector_ta::indicators::moving_averages::wilders::{
    wilders_batch_with_kernel, WildersBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaWilders;

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
fn wilders_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[wilders_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 4..series_len {
        let x = i as f64;
        data[i] = (x * 0.0015).sin() + 0.0002 * x;
    }

    let sweep = WildersBatchRange { period: (5, 48, 3) };

    let cpu = wilders_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaWilders::new(0).expect("CudaWilders::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .wilders_batch_dev(&data_f32, &sweep)
        .expect("cuda wilders_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda wilders batch result to host");

    let tol = 1e-5;
    for idx in 0..(cpu.rows * cpu.cols) {
        let cpu_val = cpu.values[idx];
        let gpu_val = gpu_host[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
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
fn wilders_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[wilders_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 1024usize;
    let mut data_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.25;
            data_tm[t * cols + s] = (x * 0.0027).sin() + 0.00019 * x;
        }
    }

    let period = 14usize;

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = data_tm[t * cols + s];
        }
        let params = vector_ta::indicators::moving_averages::wilders::WildersParams {
            period: Some(period),
        };
        let input = vector_ta::indicators::moving_averages::wilders::WildersInput::from_slice(
            &series, params,
        );
        let out = vector_ta::indicators::moving_averages::wilders::wilders_with_kernel(
            &input,
            Kernel::Scalar,
        )?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaWilders::new(0).expect("CudaWilders::new");
    let params = vector_ta::indicators::moving_averages::wilders::WildersParams {
        period: Some(period),
    };
    let dev = cuda
        .wilders_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("wilders many-series dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_tm)?;

    let tol = 1e-5;
    for idx in 0..(cols * rows) {
        let c = cpu_tm[idx];
        let g = gpu_tm[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "many-series mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
        );
    }

    Ok(())
}
