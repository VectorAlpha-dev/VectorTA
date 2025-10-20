// Integration tests for CUDA Alligator kernels

use my_project::indicators::alligator::{
    alligator_batch_with_kernel, AlligatorBatchRange, AlligatorBuilder, AlligatorParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaAlligator;

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
fn alligator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[alligator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 20..series_len {
        let x = i as f64;
        data[i] = (x * 0.0015).sin() + 0.0002 * x;
    }

    let sweep = AlligatorBatchRange {
        jaw_period: (10, 18, 4),
        jaw_offset: (3, 6, 1),
        teeth_period: (6, 14, 4),
        teeth_offset: (2, 5, 1),
        lips_period: (3, 9, 3),
        lips_offset: (1, 3, 1),
    };

    let cpu = alligator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaAlligator::new(0).expect("CudaAlligator::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu = cuda
        .alligator_batch_dev(&data_f32, &sweep)
        .expect("cuda alligator_batch_dev");

    assert_eq!(cpu.rows, gpu.outputs.rows());
    assert_eq!(cpu.cols, gpu.outputs.cols());

    let mut jaw_host = vec![0f32; gpu.outputs.rows() * gpu.outputs.cols()];
    let mut teeth_host = vec![0f32; jaw_host.len()];
    let mut lips_host = vec![0f32; jaw_host.len()];
    gpu.outputs.jaw.buf.copy_to(&mut jaw_host).unwrap();
    gpu.outputs.teeth.buf.copy_to(&mut teeth_host).unwrap();
    gpu.outputs.lips.buf.copy_to(&mut lips_host).unwrap();

    let tol = 3.0e-4f64;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.jaw[idx], jaw_host[idx] as f64, tol),
            "jaw mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.teeth[idx], teeth_host[idx] as f64, tol),
            "teeth mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.lips[idx], lips_host[idx] as f64, tol),
            "lips mismatch at {}",
            idx
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn alligator_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[alligator_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 4usize;
    let rows = 1536usize;
    let mut data_tm = vec![f64::NAN; cols * rows];
    for t in 12..rows {
        for j in 0..cols {
            let idx = t * cols + j;
            let x = (t as f64) + (j as f64) * 0.09;
            data_tm[idx] = (x * 0.0023).cos() + 0.0007 * x;
        }
    }

    let params = AlligatorParams::default();

    // CPU reference in time-major form
    let mut jaw_tm = vec![f64::NAN; cols * rows];
    let mut teeth_tm = vec![f64::NAN; cols * rows];
    let mut lips_tm = vec![f64::NAN; cols * rows];
    for j in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = data_tm[t * cols + j];
        }
        let out = AlligatorBuilder::new().apply_slice(&series)?;
        for t in 0..rows {
            let idx = t * cols + j;
            jaw_tm[idx] = out.jaw[t];
            teeth_tm[idx] = out.teeth[t];
            lips_tm[idx] = out.lips[t];
        }
    }

    let cuda = CudaAlligator::new(0).expect("CudaAlligator::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let out = cuda
        .alligator_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("cuda alligator ms1p");

    assert_eq!(out.rows(), rows);
    assert_eq!(out.cols(), cols);

    let mut jaw_gpu = vec![0f32; cols * rows];
    let mut teeth_gpu = vec![0f32; cols * rows];
    let mut lips_gpu = vec![0f32; cols * rows];
    out.jaw.buf.copy_to(&mut jaw_gpu).unwrap();
    out.teeth.buf.copy_to(&mut teeth_gpu).unwrap();
    out.lips.buf.copy_to(&mut lips_gpu).unwrap();

    let tol = 3.0e-4f64;
    for idx in 0..(cols * rows) {
        assert!(
            approx_eq(jaw_tm[idx], jaw_gpu[idx] as f64, tol),
            "jaw mismatch at {}",
            idx
        );
        assert!(
            approx_eq(teeth_tm[idx], teeth_gpu[idx] as f64, tol),
            "teeth mismatch at {}",
            idx
        );
        assert!(
            approx_eq(lips_tm[idx], lips_gpu[idx] as f64, tol),
            "lips mismatch at {}",
            idx
        );
    }

    Ok(())
}
