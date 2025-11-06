// Integration tests for CUDA Aroon kernels

use my_project::utilities::enums::Kernel;

use my_project::indicators::aroon::{
    aroon_batch_with_kernel, aroon_with_kernel, AroonBatchRange, AroonData, AroonInput, AroonParams,
};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaAroon;

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
fn aroon_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[aroon_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let n = 50_000usize;
    let mut high = vec![f64::NAN; n];
    let mut low = vec![f64::NAN; n];
    for i in 7..n {
        let x = i as f64 * 0.0023;
        let base = x.sin() * 0.6 + 0.0002 * (i as f64);
        high[i] = base + 1.3 + 0.02 * (x * 1.9).cos();
        low[i] = base - 1.3 - 0.015 * (x * 1.2).sin();
    }
    let sweep = AroonBatchRange { length: (5, 60, 5) };
    // Quantize baseline to f32 like the device path to avoid precision/tie drift
    // emulate device precision: cast to f32 then back to f64
    let high_q: Vec<f64> = high.iter().map(|&v| (v as f32) as f64).collect();
    let low_q: Vec<f64> = low.iter().map(|&v| (v as f32) as f64).collect();
    let cpu = aroon_batch_with_kernel(&high_q, &low_q, &sweep, Kernel::ScalarBatch)?;

    let hf32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let lf32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cuda = CudaAroon::new(0).expect("CudaAroon::new");
    let out = cuda
        .aroon_batch_dev(&hf32, &lf32, &sweep)
        .expect("aroon_batch_dev");

    assert_eq!(cpu.rows, out.outputs.rows());
    assert_eq!(cpu.cols, out.outputs.cols());

    let mut up_host = vec![0f32; out.outputs.first.len()];
    let mut dn_host = vec![0f32; out.outputs.second.len()];
    out.outputs.first.buf.copy_to(&mut up_host)?;
    out.outputs.second.buf.copy_to(&mut dn_host)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.up[idx], up_host[idx] as f64, tol),
            "up mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.down[idx], dn_host[idx] as f64, tol),
            "down mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn aroon_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[aroon_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 16usize; // series count
    let rows = 4096usize; // time
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s % 5)..rows {
            let x = (t as f64) * 0.003 + (s as f64) * 0.01;
            let base = x.sin() * 0.5 + 0.0003 * (t as f64);
            high_tm[t * cols + s] = base + 0.9 + 0.02 * (x * 1.3).cos();
            low_tm[t * cols + s] = base - 0.9 - 0.015 * (x * 1.7).sin();
        }
    }
    let length = 25usize;

    // CPU baseline per series
    let mut up_cpu = vec![f64::NAN; cols * rows];
    let mut dn_cpu = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = high_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
        }
        let params = AroonParams {
            length: Some(length),
        };
        let input = AroonInput {
            data: AroonData::SlicesHL { high: &h, low: &l },
            params,
        };
        let out = aroon_with_kernel(&input, Kernel::Scalar).expect("cpu aroon");
        for t in 0..rows {
            up_cpu[t * cols + s] = out.aroon_up[t];
            dn_cpu[t * cols + s] = out.aroon_down[t];
        }
    }

    let hf32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let lf32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaAroon::new(0).expect("CudaAroon::new");
    let dev = cuda
        .aroon_many_series_one_param_time_major_dev(&hf32, &lf32, cols, rows, length)
        .expect("aroon_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows(), rows);
    assert_eq!(dev.cols(), cols);
    let mut up_g = vec![0f32; dev.first.len()];
    let mut dn_g = vec![0f32; dev.second.len()];
    dev.first.buf.copy_to(&mut up_g)?;
    dev.second.buf.copy_to(&mut dn_g)?;
    let tol = 1e-4;
    for idx in 0..up_g.len() {
        assert!(
            approx_eq(up_cpu[idx], up_g[idx] as f64, tol),
            "up mismatch at {}",
            idx
        );
        assert!(
            approx_eq(dn_cpu[idx], dn_g[idx] as f64, tol),
            "down mismatch at {}",
            idx
        );
    }
    Ok(())
}
