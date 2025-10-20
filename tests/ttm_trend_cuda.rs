use my_project::indicators::ttm_trend::{
    ttm_trend_batch_with_kernel, ttm_trend_with_kernel, TtmTrendBatchBuilder, TtmTrendBatchRange,
    TtmTrendInput, TtmTrendParams,
};
use my_project::utilities::data_loader::Candles;
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, CudaTtmTrend};

fn make_series(len: usize) -> (Vec<f64>, Vec<f64>) {
    let mut src = vec![f64::NAN; len];
    let mut cls = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        src[i] = (x * 0.0013).sin() + 0.00021 * x;
        cls[i] = src[i] + 0.05 * (x * 0.00071).cos();
    }
    (src, cls)
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
fn ttm_trend_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ttm_trend_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 4096usize;
    let (src, cls) = make_series(len);
    let sweep = TtmTrendBatchRange { period: (5, 64, 7) };

    // CPU baseline (bool -> f32 mapping)
    let cpu = ttm_trend_batch_with_kernel(&src, &cls, &sweep, Kernel::ScalarBatch)?;
    let cpu_f32: Vec<f32> = cpu
        .values
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect();

    // GPU
    let src_f32: Vec<f32> = src.iter().map(|&v| v as f32).collect();
    let cls_f32: Vec<f32> = cls.iter().map(|&v| v as f32).collect();
    let cuda = CudaTtmTrend::new(0).expect("CudaTtmTrend::new");
    let dev = cuda
        .ttm_trend_batch_dev(&src_f32, &cls_f32, &sweep)
        .expect("cuda ttm_trend batch");

    assert_eq!(dev.rows, cpu.rows);
    assert_eq!(dev.cols, cpu.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    for i in 0..host.len() {
        assert!(
            (host[i] - cpu_f32[i]).abs() < 0.5,
            "mismatch at {}: gpu={} cpu={}",
            i,
            host[i],
            cpu_f32[i]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ttm_trend_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ttm_trend_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; // number of series
    let rows = 2048usize; // length per series
    let mut src_tm = vec![f64::NAN; cols * rows];
    let mut cls_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.13;
            let v = (x * 0.0021).sin() + 0.00017 * x;
            src_tm[t * cols + s] = v;
            cls_tm[t * cols + s] = v + 0.05 * (x * 0.00077).cos();
        }
    }

    let period = 13usize;

    // CPU per-series
    let mut cpu_bool_tm = vec![false; cols * rows];
    for s in 0..cols {
        let mut src = vec![f64::NAN; rows];
        let mut cls = vec![f64::NAN; rows];
        for t in 0..rows {
            src[t] = src_tm[t * cols + s];
            cls[t] = cls_tm[t * cols + s];
        }
        let input = TtmTrendInput::from_slices(
            &src,
            &cls,
            TtmTrendParams {
                period: Some(period),
            },
        );
        let out = ttm_trend_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_bool_tm[t * cols + s] = out.values[t];
        }
    }
    let cpu_f32_tm: Vec<f32> = cpu_bool_tm
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect();

    // GPU
    let src_tm_f32: Vec<f32> = src_tm.iter().map(|&v| v as f32).collect();
    let cls_tm_f32: Vec<f32> = cls_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaTtmTrend::new(0).expect("CudaTtmTrend::new");
    let dev = cuda
        .ttm_trend_many_series_one_param_time_major_dev(
            &src_tm_f32,
            &cls_tm_f32,
            cols,
            rows,
            period,
        )
        .expect("cuda ttm_trend many-series");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    for i in 0..host.len() {
        assert!((host[i] - cpu_f32_tm[i]).abs() < 0.5, "mismatch at {}", i);
    }
    Ok(())
}
