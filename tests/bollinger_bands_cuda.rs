use vector_ta::indicators::bollinger_bands::{
    bollinger_bands_batch_with_kernel, BollingerBandsBatchRange,
};
use vector_ta::indicators::bollinger_bands::{
    bollinger_bands_with_kernel, BollingerBandsInput, BollingerBandsParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaBollingerBands;

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
fn bollinger_bands_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[bollinger_bands_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        data[i] = (x * 0.00037).sin() + 0.00021 * x;
        if i % 253 == 0 {
            data[i] = f64::NAN;
        }
    }

    let sweep = BollingerBandsBatchRange {
        period: (10, 30, 10),
        devup: (1.5, 2.5, 0.5),
        devdn: (1.5, 2.5, 0.5),
        matype: ("sma".to_string(), "sma".to_string(), 0),
        devtype: (0, 0, 0),
    };

    let cpu = bollinger_bands_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();

    let cuda = CudaBollingerBands::new(0).expect("CudaBollingerBands::new");
    let (dev_up, dev_mid, dev_lo) = cuda
        .bollinger_bands_batch_dev(&data_f32, &sweep)
        .expect("bollinger_bands_cuda_batch_dev");

    assert_eq!(dev_up.rows, cpu.rows);
    assert_eq!(dev_mid.rows, cpu.rows);
    assert_eq!(dev_lo.rows, cpu.rows);
    assert_eq!(dev_up.cols, cpu.cols);
    assert_eq!(dev_mid.cols, cpu.cols);
    assert_eq!(dev_lo.cols, cpu.cols);

    let mut up = vec![0f32; dev_up.len()];
    let mut mid = vec![0f32; dev_mid.len()];
    let mut lo = vec![0f32; dev_lo.len()];
    dev_up.buf.copy_to(&mut up)?;
    dev_mid.buf.copy_to(&mut mid)?;
    dev_lo.buf.copy_to(&mut lo)?;

    let tol = 1e-3;
    for i in 0..(cpu.rows * cpu.cols) {
        if !approx_eq(cpu.upper[i], up[i] as f64, tol) {
            eprintln!(
                "DEBUG mismatch upper idx {}: cpu={} gpu={}",
                i, cpu.upper[i], up[i]
            );
            panic!("upper mismatch at {}", i);
        }
        if !approx_eq(cpu.middle[i], mid[i] as f64, tol) {
            eprintln!(
                "DEBUG mismatch middle idx {}: cpu={} gpu={}",
                i, cpu.middle[i], mid[i]
            );
            panic!("middle mismatch at {}", i);
        }
        if !approx_eq(cpu.lower[i], lo[i] as f64, tol) {
            eprintln!(
                "DEBUG mismatch lower idx {}: cpu={} gpu={}",
                i, cpu.lower[i], lo[i]
            );
            panic!("lower mismatch at {}", i);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn bollinger_bands_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[bollinger_bands_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let cols = 8usize;
    let rows = 1024usize;
    let period = 20usize;
    let devup = 2.0f64;
    let devdn = 2.0f64;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + 0.1 * (s as f64);
            tm[t * cols + s] = (x * 0.003).sin() + 0.0002 * x;
        }
    }

    let mut cpu_up_tm = vec![f64::NAN; cols * rows];
    let mut cpu_mid_tm = vec![f64::NAN; cols * rows];
    let mut cpu_lo_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = tm[t * cols + s];
        }
        let params = BollingerBandsParams {
            period: Some(period),
            devup: Some(devup),
            devdn: Some(devdn),
            matype: Some("sma".into()),
            devtype: Some(0),
        };
        let input = BollingerBandsInput::from_slice(&series, params);
        let out = bollinger_bands_with_kernel(&input, Kernel::Scalar).unwrap();
        for t in 0..rows {
            cpu_up_tm[t * cols + s] = out.upper_band[t];
            cpu_mid_tm[t * cols + s] = out.middle_band[t];
            cpu_lo_tm[t * cols + s] = out.lower_band[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaBollingerBands::new(0).expect("CudaBollingerBands::new");
    let (dev_up_tm, dev_mid_tm, dev_lo_tm) = cuda
        .bollinger_bands_many_series_one_param_time_major_dev(
            &tm_f32,
            cols,
            rows,
            period,
            devup as f32,
            devdn as f32,
        )
        .expect("bollinger_bands_many_series_one_param_time_major_dev");

    assert_eq!(dev_up_tm.rows, rows);
    assert_eq!(dev_mid_tm.rows, rows);
    assert_eq!(dev_lo_tm.rows, rows);
    assert_eq!(dev_up_tm.cols, cols);
    assert_eq!(dev_mid_tm.cols, cols);
    assert_eq!(dev_lo_tm.cols, cols);

    let mut up_tm = vec![0f32; dev_up_tm.len()];
    let mut mid_tm = vec![0f32; dev_mid_tm.len()];
    let mut lo_tm = vec![0f32; dev_lo_tm.len()];
    dev_up_tm.buf.copy_to(&mut up_tm)?;
    dev_mid_tm.buf.copy_to(&mut mid_tm)?;
    dev_lo_tm.buf.copy_to(&mut lo_tm)?;

    let tol = 1e-4;
    for idx in 0..(cols * rows) {
        assert!(
            approx_eq(cpu_up_tm[idx], up_tm[idx] as f64, tol),
            "upper mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_mid_tm[idx], mid_tm[idx] as f64, tol),
            "middle mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_lo_tm[idx], lo_tm[idx] as f64, tol),
            "lower mismatch at {}",
            idx
        );
    }
    Ok(())
}
