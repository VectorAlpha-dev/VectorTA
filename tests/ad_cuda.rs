// CUDA integration tests for Chaikin Accumulation/Distribution (AD)

use my_project::indicators::ad::{ad_with_kernel, AdData, AdInput, AdParams};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, CudaAd};

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
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn ad_cuda_series_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ad_cuda_series_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let n = 4096usize;
    let mut high = vec![0.0f64; n];
    let mut low = vec![0.0f64; n];
    let mut close = vec![0.0f64; n];
    let mut volume = vec![0.0f64; n];
    for i in 0..n {
        let x = i as f64;
        let base = (x * 0.004).sin() + 0.0005 * x;
        let off = (0.0031 * (x * 0.17).cos()).abs() + 0.12;
        high[i] = base + off;
        low[i] = base - off;
        close[i] = base + 0.25 * off * (x * 0.01).sin();
        volume[i] = ((x * 0.009).cos().abs() + 0.9) * 1200.0;
    }

    let input = AdInput {
        data: AdData::Slices {
            high: &high,
            low: &low,
            close: &close,
            volume: &volume,
        },
        params: AdParams::default(),
    };
    let cpu = ad_with_kernel(&input, Kernel::Scalar)?;

    let cuda = CudaAd::new(0).map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    let hf: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let vf: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let dev = cuda.ad_series_dev(&hf, &lf, &cf, &vf)?;
    assert_eq!(dev.rows, 1);
    assert_eq!(dev.cols, n);
    let mut gpu = vec![0f32; n];
    dev.buf.copy_to(&mut gpu)?;

    let (atol, rtol) = (2e-2, 2e-3); // single-series permits slightly looser tolerance
    for i in 0..n {
        assert!(
            approx_eq(cpu.values[i], gpu[i] as f64, atol, rtol),
            "mismatch at {}",
            i
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ad_cuda_many_series_time_major_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ad_cuda_many_series_time_major_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 7usize; // num series
    let rows = 1024usize; // series length
    let mut high_tm = vec![0.0f64; cols * rows];
    let mut low_tm = vec![0.0f64; cols * rows];
    let mut close_tm = vec![0.0f64; cols * rows];
    let mut volume_tm = vec![0.0f64; cols * rows];
    for t in 0..rows {
        for s in 0..cols {
            let idx = t * cols + s;
            let x = (t as f64) + (s as f64) * 0.3;
            let base = (x * 0.003).sin() + 0.0004 * x;
            let off = (0.0029 * (x * 0.17).cos()).abs() + 0.11;
            high_tm[idx] = base + off;
            low_tm[idx] = base - off;
            close_tm[idx] = base + 0.2 * off * (x * 0.01).sin();
            volume_tm[idx] = ((x * 0.007).cos().abs() + 0.8) * 900.0;
        }
    }

    // CPU baseline by rows
    let mut cpu_tm = vec![0.0f64; cols * rows];
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
        let input = AdInput {
            data: AdData::Slices {
                high: &h,
                low: &l,
                close: &c,
                volume: &v,
            },
            params: AdParams::default(),
        };
        let out = ad_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let cuda = CudaAd::new(0).map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    let hf: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let vf: Vec<f32> = volume_tm.iter().map(|&v| v as f32).collect();
    let dev = cuda.ad_many_series_one_param_time_major_dev(&hf, &lf, &cf, &vf, cols, rows)?;
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut gpu_tm = vec![0f32; cols * rows];
    dev.buf.copy_to(&mut gpu_tm)?;

    let (atol, rtol) = (1e-2, 5e-4);
    for idx in 0..(cols * rows) {
        assert!(
            approx_eq(cpu_tm[idx], gpu_tm[idx] as f64, atol, rtol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
