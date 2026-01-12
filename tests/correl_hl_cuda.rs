use vector_ta::indicators::correl_hl::{
    correl_hl_batch_with_kernel, correl_hl_with_kernel, CorrelHlBatchRange, CorrelHlData,
    CorrelHlInput, CorrelHlParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaCorrelHl;

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
fn correl_hl_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[correl_hl_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 32768usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 8..len {
        let x = i as f64;
        high[i] = (x * 0.00123).sin() + 0.0001 * x;
        low[i] = (x * 0.00079).cos() + 0.00005 * x;
    }
    let sweep = CorrelHlBatchRange { period: (9, 64, 1) };

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let high_q: Vec<f64> = high_f32.iter().map(|&v| v as f64).collect();
    let low_q: Vec<f64> = low_f32.iter().map(|&v| v as f64).collect();
    let cpu = correl_hl_batch_with_kernel(&high_q, &low_q, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaCorrelHl::new(0).expect("CudaCorrelHl::new");
    let (dev, _combos) = cuda
        .correl_hl_batch_dev(&high_f32, &low_f32, &sweep)
        .expect("correl_hl_cuda_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.rows * dev.cols];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-3;
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
fn correl_hl_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[correl_hl_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 4096usize;
    let mut high_tm = vec![f64::NAN; rows * cols];
    let mut low_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        for t in (s)..rows {
            let x = (t as f64) + (s as f64) * 0.5;
            high_tm[t * cols + s] = (x * 0.0021).sin() + 0.0002 * x;
            low_tm[t * cols + s] = (x * 0.0017).cos() + 0.0001 * x;
        }
    }

    let period = 21usize;

    let mut cpu_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = high_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
        }
        let params = CorrelHlParams {
            period: Some(period),
        };
        let input = CorrelHlInput {
            data: CorrelHlData::Slices { high: &h, low: &l },
            params,
        };
        let out = correl_hl_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaCorrelHl::new(0).expect("CudaCorrelHl::new");
    let dev_tm = cuda
        .correl_hl_many_series_one_param_time_major_dev(
            &high_tm_f32,
            &low_tm_f32,
            cols,
            rows,
            period,
        )
        .expect("correl_hl_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);
    let mut g_tm = vec![0f32; rows * cols];
    dev_tm.buf.copy_to(&mut g_tm)?;

    let tol = 5e-3;
    for idx in 0..g_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], g_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }

    Ok(())
}
