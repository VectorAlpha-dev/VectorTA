use vector_ta::indicators::vidya::{
    vidya_batch_with_kernel, vidya_with_kernel, VidyaBatchRange, VidyaData, VidyaInput, VidyaParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaVidya;

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
fn vidya_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vidya_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 8..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = VidyaBatchRange {
        short_period: (2, 2, 0),
        long_period: (10, 64, 3),
        alpha: (0.2, 0.2, 0.0),
    };

    let cpu = vidya_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaVidya::new(0).expect("CudaVidya::new");
    let dev = cuda
        .vidya_batch_dev(&price_f32, &sweep)
        .expect("vidya_batch_dev");

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
fn vidya_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vidya_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 4096usize;
    let mut price_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            price_tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let sp = 2usize;
    let lp = 21usize;
    let alpha = 0.2f64;

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = price_tm[t * cols + s];
        }
        let params = VidyaParams {
            short_period: Some(sp),
            long_period: Some(lp),
            alpha: Some(alpha),
        };
        let input = VidyaInput {
            data: VidyaData::Slice(&p),
            params,
        };
        let out = vidya_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
    }

    let price_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaVidya::new(0).expect("CudaVidya::new");
    let params = VidyaParams {
        short_period: Some(sp),
        long_period: Some(lp),
        alpha: Some(alpha),
    };
    let dev_tm = cuda
        .vidya_many_series_one_param_time_major_dev(&price_tm_f32, cols, rows, &params)
        .expect("vidya many series dev");

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
