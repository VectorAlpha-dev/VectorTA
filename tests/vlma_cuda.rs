use vector_ta::indicators::vlma::{
    vlma_batch_with_kernel, vlma_with_kernel, VlmaBatchBuilder, VlmaBatchRange, VlmaBuilder,
    VlmaInput, VlmaParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaVlma;

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
fn vlma_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vlma_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut price = vec![f64::NAN; len];
    for i in 10..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }

    let sweep = VlmaBatchRange {
        min_period: (5, 7, 1),
        max_period: (20, 28, 4),
        matype: ("sma".to_string(), "sma".to_string(), "".to_string()),
        devtype: (0, 0, 0),
    };

    let cpu = vlma_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let mut cuda = CudaVlma::new(0).expect("CudaVlma::new");
    let dev = cuda
        .vlma_batch_dev(&price_f32, &sweep)
        .expect("vlma_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 2e-3;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
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
fn vlma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vlma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 6usize;
    let rows = 1200usize;
    let mut tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.25;
            tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let min_p = 5usize;
    let max_p = 27usize;

    let mut cpu = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = tm[t * cols + s];
        }
        let params = VlmaParams {
            min_period: Some(min_p),
            max_period: Some(max_p),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        };
        let input = VlmaInput::from_slice(&series, params);
        let out = vlma_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            cpu[t * cols + s] = out[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let params = VlmaParams {
        min_period: Some(min_p),
        max_period: Some(max_p),
        matype: Some("sma".to_string()),
        devtype: Some(0),
    };
    let mut cuda = CudaVlma::new(0).expect("CudaVlma::new");
    let dev_tm = cuda
        .vlma_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("vlma_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);

    let mut g_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut g_tm)?;

    let tol = 2e-3;
    for idx in 0..g_tm.len() {
        assert!(
            approx_eq(cpu[idx], g_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }

    Ok(())
}
