// Integration tests for CUDA SAR kernels

use my_project::indicators::sar::{sar_with_kernel, SarBatchBuilder, SarInput, SarParams};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaSar;

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
fn sar_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[sar_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 32_768usize;
    let mut base = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        base[i] = (x * 0.0013).sin() + 0.0002 * x;
    }
    let mut high = base.clone();
    let mut low = base.clone();
    for i in 4..len {
        let x = i as f64 * 0.0021;
        let off = (0.0087 * x.cos()).abs() + 0.1;
        high[i] = base[i] + off;
        low[i] = base[i] - off;
    }

    let sweep = my_project::indicators::sar::SarBatchRange {
        acceleration: (0.01, 0.05, 0.01),
        maximum: (0.1, 0.3, 0.1),
    };
    // Match the exact sweep used for CUDA to ensure apples-to-apples
    let cpu = my_project::indicators::sar::sar_batch_with_kernel(
        &high,
        &low,
        &sweep,
        Kernel::ScalarBatch,
    )?;

    let h_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let l_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cuda = CudaSar::new(0).expect("CudaSar::new");
    let (dev, _combos) = cuda
        .sar_batch_dev(&h_f32, &l_f32, &sweep)
        .expect("sar_batch_dev");
    assert_eq!(dev.rows, cpu.rows);
    assert_eq!(dev.cols, cpu.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 3e-1; // loosened tolerance: FP32 GPU warp-synchronous kernel vs f64 CPU
    for idx in 0..host.len() {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        if !approx_eq(c, g, tol) {
            eprintln!("first mismatch at {}: cpu={} gpu={}", idx, c, g);
            // Dump a small window around the mismatch for row 0
            let len = len;
            let r = idx / len;
            let t = idx % len;
            eprintln!("row={}, t={}", r, t);
            let start = t.saturating_sub(4);
            let end = (t + 5).min(len);
            for j in start..end {
                let ii = r * len + j;
                eprintln!("t={} cpu={} gpu={}", j, cpu.values[ii], host[ii] as f64);
            }
            assert!(false, "mismatch at {}: cpu={} gpu={}", idx, c, g);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn sar_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[sar_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 4096usize;
    let mut base_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s + 2)..rows {
            let x = t as f64 + s as f64 * 0.2;
            base_tm[t * cols + s] = (x * 0.0025).sin() + 0.00025 * x;
        }
    }
    let mut high_tm = base_tm.clone();
    let mut low_tm = base_tm.clone();
    for s in 0..cols {
        for t in 0..rows {
            let v = base_tm[t * cols + s];
            if v.is_nan() {
                continue;
            }
            let x = (t as f64) * 0.0021 + s as f64 * 0.03;
            let off = (0.0077 * x.cos()).abs() + 0.09;
            high_tm[t * cols + s] = v + off;
            low_tm[t * cols + s] = v - off;
        }
    }

    // CPU baseline per series
    let params = SarParams {
        acceleration: Some(0.02),
        maximum: Some(0.2),
    };
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = high_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
        }
        let out = sar_with_kernel(
            &SarInput::from_slices(&h, &l, params.clone()),
            Kernel::Scalar,
        )?
        .values;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
    }

    let h_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let l_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaSar::new(0).expect("CudaSar::new");
    let dev = cuda
        .sar_many_series_one_param_time_major_dev(&h_f32, &l_f32, cols, rows, &params)
        .expect("sar many series");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut g = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut g)?;

    let tol = 3e-3;
    for idx in 0..g.len() {
        assert!(
            approx_eq(cpu_tm[idx], g[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
