use vector_ta::indicators::natr::{NatrBatchBuilder, NatrBatchRange};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaNatr;

fn approx_eq(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs());
    diff <= atol + rtol * scale
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
fn natr_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[natr_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        let base = (x * 0.0021).sin() + 0.0002 * x;
        high[i] = base + 0.8;
        low[i] = base - 0.7;
        close[i] = base;
    }

    let sweep = NatrBatchRange { period: (7, 64, 3) };

    let cpu = NatrBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slices(&high, &low, &close)?;

    let mut cuda = CudaNatr::new(0).expect("CudaNatr::new");
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let gpu = cuda
        .natr_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
        .expect("natr_batch_dev");

    assert_eq!(cpu.rows, gpu.rows);
    assert_eq!(cpu.cols, gpu.cols);

    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf.copy_to(&mut gpu_host)?;

    let atol = 5e-3;
    let rtol = 1e-7;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.values[idx], gpu_host[idx] as f64, atol, rtol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu.values[idx],
            gpu_host[idx]
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn natr_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[natr_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.31;
            let base = (x * 0.0023).sin() + 0.00019 * x;
            high_tm[t * cols + s] = base + 0.6;
            low_tm[t * cols + s] = base - 0.55;
            close_tm[t * cols + s] = base;
        }
    }
    let period = 14usize;

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = high_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
            c[t] = close_tm[t * cols + s];
        }
        let out = vector_ta::indicators::natr::natr(
            &vector_ta::indicators::natr::NatrInput::from_slices(
                &h,
                &l,
                &c,
                vector_ta::indicators::natr::NatrParams {
                    period: Some(period),
                },
            ),
        )?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let mut cuda = CudaNatr::new(0).expect("CudaNatr::new");
    let hf: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let gpu_tm = cuda
        .natr_many_series_one_param_time_major_dev(&hf, &lf, &cf, cols, rows, period)
        .expect("natr many-series");

    assert_eq!(gpu_tm.rows, rows);
    assert_eq!(gpu_tm.cols, cols);
    let mut g = vec![0f32; gpu_tm.len()];
    gpu_tm.buf.copy_to(&mut g)?;

    let atol = 5e-3;
    let rtol = 1e-7;
    for idx in 0..(cols * rows) {
        assert!(
            approx_eq(cpu_tm[idx], g[idx] as f64, atol, rtol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_tm[idx],
            g[idx]
        );
    }

    Ok(())
}
