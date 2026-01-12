use vector_ta::indicators::rsx::{
    rsx_batch_with_kernel, rsx_with_kernel, RsxBatchRange, RsxInput, RsxParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::rsx_wrapper::CudaRsx;

fn approx_eq(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    let thresh = atol.max(rtol * a.abs().max(b.abs()));
    diff <= thresh
}

#[test]
fn cuda_feature_off_noop_rsx() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn rsx_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rsx_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 8..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = RsxBatchRange { period: (4, 60, 4) };

    let cpu = rsx_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaRsx::new(0).expect("CudaRsx::new");
    let dev = cuda
        .rsx_batch_dev(&price_f32, &sweep)
        .expect("rsx_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let atol = 6e-1;
    let rtol = 1e-1;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, atol, rtol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn rsx_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rsx_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize;
    let rows = 2048usize;
    let mut price_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            price_tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }

    let period = 14usize;

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = price_tm[t * cols + s];
        }
        let params = RsxParams {
            period: Some(period),
        };
        let input = RsxInput::from_slice(&series, params);
        let out = rsx_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let prices_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaRsx::new(0).expect("CudaRsx::new");
    let dev_tm = cuda
        .rsx_many_series_one_param_time_major_dev(&prices_tm_f32, cols, rows, period)
        .expect("rsx many series");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);

    let mut g_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut g_tm)?;

    let atol = 6e-1;
    let rtol = 1e-1;
    for idx in 0..g_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], g_tm[idx] as f64, atol, rtol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
