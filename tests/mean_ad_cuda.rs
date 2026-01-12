use vector_ta::indicators::mean_ad::{
    mean_ad_batch_with_kernel, mean_ad_with_kernel, MeanAdBatchRange, MeanAdData, MeanAdInput,
    MeanAdParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaMeanAd;

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
fn mean_ad_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mean_ad_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 8192usize;
    let mut data = vec![f64::NAN; len];
    for i in 8..len {
        let x = i as f64;
        data[i] = (x * 0.00123).sin() + 0.00011 * x;
    }

    let sweep = MeanAdBatchRange { period: (5, 25, 5) };

    let cpu = mean_ad_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaMeanAd::new(0).expect("CudaMeanAd::new");
    let dev = cuda
        .mean_ad_batch_dev(&data_f32, &sweep)
        .expect("mean_ad_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-4;
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu.values[idx], host[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn mean_ad_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mean_ad_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 2048usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.123;
            tm[t * cols + s] = (x * 0.002).sin() + 0.0007 * x;
        }
    }

    let period = 15usize;

    let mut ref_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = tm[t * cols + s];
        }
        let input = MeanAdInput {
            data: MeanAdData::Slice(&series),
            params: MeanAdParams {
                period: Some(period),
            },
        };
        let out = mean_ad_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            ref_tm[t * cols + s] = out[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaMeanAd::new(0).expect("CudaMeanAd::new");
    let dev = cuda
        .mean_ad_many_series_one_param_time_major_dev(
            &tm_f32,
            cols,
            rows,
            &MeanAdParams {
                period: Some(period),
            },
        )
        .expect("mean_ad_many_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut g_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut g_tm)?;

    let tol = 5e-4;
    for idx in 0..g_tm.len() {
        assert!(
            approx_eq(ref_tm[idx], g_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
