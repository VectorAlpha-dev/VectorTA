use vector_ta::indicators::moving_averages::edcf::{
    edcf_batch_with_kernel, EdcfBatchRange, EdcfParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer};
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaEdcf;

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
fn edcf_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[edcf_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 3..series_len {
        let x = i as f64;
        data[i] = (x * 0.002).sin() + 0.0003 * x;
    }

    let sweep = EdcfBatchRange { period: (5, 32, 3) };

    let cpu = edcf_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaEdcf::new(0).expect("CudaEdcf::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .edcf_batch_dev(&data_f32, &sweep)
        .expect("cuda edcf_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda edcf batch result to host");

    let tol = 1e-4;
    for idx in 0..gpu_host.len() {
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
fn edcf_cuda_batch_device_reuse_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[edcf_cuda_batch_device_reuse_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 2048usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 5..series_len {
        let x = i as f64;
        data[i] = (x * 0.0015).cos() + 0.0002 * x;
    }

    let sweep = EdcfBatchRange { period: (9, 21, 6) };
    let combos: Vec<EdcfParams> = (sweep.period.0..=sweep.period.1)
        .step_by(sweep.period.2)
        .map(|p| EdcfParams { period: Some(p) })
        .collect();

    let cpu = edcf_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    assert_eq!(cpu.rows, combos.len());
    assert_eq!(cpu.cols, series_len);

    let first_valid = data
        .iter()
        .position(|x| !x.is_nan())
        .expect("non-NaN value expected");

    let cuda = CudaEdcf::new(0).expect("CudaEdcf::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let d_prices = DeviceBuffer::from_slice(&data_f32).expect("price upload");
    let mut d_dist: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(series_len) }.expect("dist buffer");
    let mut d_out: DeviceBuffer<f32> =
        unsafe { DeviceBuffer::uninitialized(combos.len() * series_len) }.expect("out buffer");

    cuda.edcf_batch_device(
        &d_prices,
        &combos,
        first_valid,
        series_len,
        &mut d_dist,
        &mut d_out,
    )
    .expect("cuda edcf_batch_device");

    let mut gpu_host = vec![0f32; combos.len() * series_len];
    d_out.copy_to(&mut gpu_host).expect("copy device output");

    let tol = 1e-4;
    for idx in 0..gpu_host.len() {
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
