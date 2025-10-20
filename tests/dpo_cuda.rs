// Integration tests for CUDA DPO kernels

use my_project::indicators::dpo::{
    dpo_batch_with_kernel, dpo_with_kernel, DpoBatchRange, DpoBuilder, DpoInput, DpoParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::dpo_wrapper::CudaDpo;

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
fn dpo_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dpo_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 16_384usize;
    let mut price = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00011 * x;
    }
    let sweep = DpoBatchRange {
        period: (8, 8 + 63, 1),
    };

    let cpu = dpo_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaDpo::new(0).expect("CudaDpo::new");
    let dev = cuda
        .dpo_batch_dev(&price_f32, &sweep)
        .expect("dpo_batch_dev");
    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 7e-4; // FP32 kernel vs FP64 CPU
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "batch mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn dpo_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dpo_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 12usize; // series count
    let rows = 8192usize; // series length
    let mut data_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s % 7)..rows {
            let x = (t as f64) + (s as f64) * 0.37;
            data_tm[t * cols + s] = (x * 0.0019).sin() + 0.00021 * x;
        }
    }

    let period = 14usize;

    // CPU baseline per series (row-major flatten)
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = data_tm[t * cols + s];
        }
        let params = DpoParams {
            period: Some(period),
        };
        let input = DpoInput::from_slice(&p, params);
        let out = dpo_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaDpo::new(0).expect("CudaDpo::new");
    let dev = cuda
        .dpo_many_series_one_param_time_major_dev(
            &data_tm_f32,
            cols,
            rows,
            &DpoParams {
                period: Some(period),
            },
        )
        .expect("dpo_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 7e-4;
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu_tm[idx], host[idx] as f64, tol),
            "many-series mismatch at {}",
            idx
        );
    }
    Ok(())
}
