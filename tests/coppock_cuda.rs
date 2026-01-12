use vector_ta::indicators::coppock::{
    coppock_batch_with_kernel, coppock_with_kernel, CoppockBatchRange, CoppockInput, CoppockParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::coppock_wrapper::CudaCoppock;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn coppock_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[coppock_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 8192usize;
    let mut close = vec![f64::NAN; len];
    for i in 50..len {
        let x = i as f64;
        close[i] = 100.0 + (x * 0.001).sin() * 2.0 + 0.001 * x;
    }

    let sweep = CoppockBatchRange {
        short: (8, 16, 4),
        long: (14, 28, 7),
        ma: (8, 12, 2),
    };

    let cpu = coppock_batch_with_kernel(&close, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaCoppock::new(0).expect("CudaCoppock::new");
    let dev = cuda
        .coppock_batch_dev(&price_f32, &sweep)
        .expect("coppock_batch_dev");
    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    let mut got = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got)?;

    let tol = 8e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.values[idx], got[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn coppock_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[coppock_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 6usize;
    let rows = 2048usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.3;
            tm[t * cols + s] = 100.0 + (x * 0.0023).sin() + 0.0007 * x;
        }
    }

    let short = 11usize;
    let long = 14usize;
    let ma_p = 10usize;

    let mut cpu = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = tm[t * cols + s];
        }
        let params = CoppockParams {
            short_roc_period: Some(short),
            long_roc_period: Some(long),
            ma_period: Some(ma_p),
            ma_type: Some("wma".to_string()),
        };
        let input = CoppockInput::from_slice(&series, params);
        let out = coppock_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu[t * cols + s] = out.values[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaCoppock::new(0).expect("CudaCoppock::new");
    let dev = cuda
        .coppock_many_series_one_param_time_major_dev(&tm_f32, cols, rows, short, long, ma_p)
        .expect("coppock_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut got = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got)?;

    let tol = 8e-4;
    for idx in 0..got.len() {
        assert!(
            approx_eq(cpu[idx], got[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
