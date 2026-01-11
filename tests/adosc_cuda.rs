

use vector_ta::indicators::adosc::{
    adosc_batch_with_kernel, AdoscBatchRange, AdoscInput, AdoscParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaAdosc;

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
fn adosc_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[adosc_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 16_384usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 0..len {
        let x = i as f64;
        let base = (x * 0.0007).sin() + 0.0002 * x;
        let off = (x * 0.0013).cos().abs() + 0.2;
        close[i] = base;
        high[i] = base + off;
        low[i] = base - off;
        volume[i] = (x * 0.0021).cos().abs() * 1000.0 + 10.0;
    }
    let sweep = AdoscBatchRange {
        short_period: (3, 7, 2),
        long_period: (10, 20, 5),
    };

    let cpu = adosc_batch_with_kernel(&high, &low, &close, &volume, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();

    let cuda = CudaAdosc::new(0).expect("CudaAdosc::new");
    let dev = cuda
        .adosc_batch_dev(&high_f32, &low_f32, &close_f32, &volume_f32, &sweep)
        .expect("adosc_batch_dev");

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
fn adosc_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[adosc_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 4096usize;
    let mut high_tm = vec![0f64; rows * cols];
    let mut low_tm = vec![0f64; rows * cols];
    let mut close_tm = vec![0f64; rows * cols];
    let mut volume_tm = vec![0f64; rows * cols];
    for s in 0..cols {
        for t in 0..rows {
            let x = t as f64 + s as f64 * 0.33;
            let base = (x * 0.001).sin() + 0.0001 * x;
            let off = (x * 0.002).cos().abs() + 0.1;
            let idx = t * cols + s;
            close_tm[idx] = base;
            high_tm[idx] = base + off;
            low_tm[idx] = base - off;
            volume_tm[idx] = (x * 0.003).cos().abs() * 800.0 + 5.0;
        }
    }

    let short = 5usize;
    let long = 21usize;

    
    let mut cpu_tm = vec![0.0f64; rows * cols];
    for s in 0..cols {
        let mut h = vec![0.0f64; rows];
        let mut l = vec![0.0f64; rows];
        let mut c = vec![0.0f64; rows];
        let mut v = vec![0.0f64; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            h[t] = high_tm[idx];
            l[t] = low_tm[idx];
            c[t] = close_tm[idx];
            v[t] = volume_tm[idx];
        }
        let params = AdoscParams {
            short_period: Some(short),
            long_period: Some(long),
        };
        let input = AdoscInput::from_slices(&h, &l, &c, &v, params);
        let out = vector_ta::indicators::adosc::adosc(&input)?.values;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let volume_tm_f32: Vec<f32> = volume_tm.iter().map(|&v| v as f32).collect();

    let cuda = CudaAdosc::new(0).expect("cuda adosc");
    let dev_tm = cuda
        .adosc_many_series_one_param_time_major_dev(
            &high_tm_f32,
            &low_tm_f32,
            &close_tm_f32,
            &volume_tm_f32,
            cols,
            rows,
            short,
            long,
        )
        .expect("adosc many-series");
    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);

    let mut gpu_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut gpu_tm)?;

    let tol = 2e-3;
    for idx in 0..gpu_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], gpu_tm[idx] as f64, tol),
            "tm mismatch at {}",
            idx
        );
    }

    Ok(())
}
