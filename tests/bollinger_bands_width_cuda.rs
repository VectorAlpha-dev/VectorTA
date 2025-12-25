// Integration tests for CUDA Bollinger Bands Width (BBW)

use my_project::indicators::bollinger_bands_width::{
    bollinger_bands_width_batch_with_kernel, bollinger_bands_width_with_kernel,
    BollingerBandsWidthBatchRange, BollingerBandsWidthData, BollingerBandsWidthInput,
    BollingerBandsWidthParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::bollinger_bands_width_wrapper::CudaBbw;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;

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
fn bollinger_bands_width_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[bbw_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 16384usize;
    let mut price = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = BollingerBandsWidthBatchRange {
        period: (10, 40, 5),
        devup: (2.0, 2.0, 0.0),
        devdn: (2.0, 2.0, 0.0),
    };

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let price_quant: Vec<f64> = price_f32.iter().map(|&v| v as f64).collect();
    let cpu = bollinger_bands_width_batch_with_kernel(&price_quant, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaBbw::new(0).expect("CudaBbw::new");
    let (dev, _meta) = cuda
        .bbw_batch_dev(&price_f32, &sweep)
        .expect("bbw_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-4;
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
fn bollinger_bands_width_cuda_many_series_one_param_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[bbw_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize;
    let rows = 8192usize;
    let mut price_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.3;
            price_tm[t * cols + s] = (x * 0.002).sin() + 0.0002 * x;
        }
    }

    let period = 20usize;
    let devup = 2.0f64;
    let devdn = 2.0f64;

    let price_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    // CPU baseline per series (quantize inputs to match GPU FP32 path)
    let price_tm_quant: Vec<f64> = price_tm_f32.iter().map(|&v| v as f64).collect();
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = price_tm_quant[t * cols + s];
        }
        let params = BollingerBandsWidthParams {
            period: Some(period),
            devup: Some(devup),
            devdn: Some(devdn),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        };
        let input = BollingerBandsWidthInput {
            data: BollingerBandsWidthData::Slice(&p),
            params,
        };
        let out = bollinger_bands_width_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let cuda = CudaBbw::new(0).expect("CudaBbw::new");
    let dev_tm = cuda
        .bbw_many_series_one_param_time_major_dev(
            &price_tm_f32,
            cols,
            rows,
            period,
            devup as f32,
            devdn as f32,
        )
        .expect("bbw_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);

    let mut g_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut g_tm)?;

    let tol = 1e-4;
    for idx in 0..g_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], g_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
