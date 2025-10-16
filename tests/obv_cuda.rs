// Integration tests for CUDA OBV kernels

use my_project::indicators::obv::{obv, ObvInput, ObvParams, ObvData};
// use my_project::utilities::enums::Kernel; // not needed in these tests

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaObv;

fn approx_eq_absrel(a: f64, b: f64, abs_tol: f64, rel_tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    let scale = a.abs().max(1.0);
    diff <= abs_tol.max(rel_tol * scale)
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
fn obv_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[obv_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        close[i] = (x * 0.00123).sin() + 0.00017 * x;
        volume[i] = (x * 0.00077).cos().abs() * 800.0 + 1.0;
    }
    // Build CPU baseline on FP32-rounded data to match GPU inputs
    let close32_as_f64: Vec<f64> = close.iter().map(|&v| (v as f32) as f64).collect();
    let volume32_as_f64: Vec<f64> = volume.iter().map(|&v| (v as f32) as f64).collect();
    let cpu = obv(&ObvInput {
        data: ObvData::Slices { close: &close32_as_f64, volume: &volume32_as_f64 },
        params: ObvParams::default(),
    })?;

    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let cuda = CudaObv::new(0).expect("CudaObv::new");
    let dev = cuda
        .obv_batch_dev(&close_f32, &volume_f32)
        .expect("cuda obv_batch_dev");

    assert_eq!(dev.rows, 1);
    assert_eq!(dev.cols, len);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    for i in 0..len {
        let c = cpu.values[i];
        let g = host[i] as f64;
        assert!(
            approx_eq_absrel(c, g, 1e-3, 1e-7),
            "mismatch at {}: cpu={} gpu={}",
            i,
            c,
            g
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn obv_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[obv_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let cols = 16usize; // series
    let rows = 2048usize; // length
    let mut close_tm = vec![f64::NAN; cols * rows];
    let mut volume_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in 3..rows {
            let x = (t as f64) * 0.002 + (s as f64) * 0.05;
            close_tm[t * cols + s] = (x * 0.7).sin() + 0.0005 * x;
            volume_tm[t * cols + s] = (x * 0.3).cos().abs() * 500.0 + 1.0;
        }
    }

    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let volume_tm_f32: Vec<f32> = volume_tm.iter().map(|&v| v as f32).collect();

    // CPU baseline per series (computed on FP32-rounded data to match GPU inputs)
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut c = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for t in 0..rows {
            c[t] = close_tm_f32[t * cols + s] as f64;
            v[t] = volume_tm_f32[t * cols + s] as f64;
        }
        let input = ObvInput { data: ObvData::Slices { close: &c, volume: &v }, params: ObvParams::default() };
        let out = obv(&input)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }
    let cuda = CudaObv::new(0).expect("CudaObv::new");
    let dev_tm = cuda
        .obv_many_series_one_param_time_major_dev(&close_tm_f32, &volume_tm_f32, cols, rows)
        .expect("obv many-series tm");

    assert_eq!(dev_tm.cols, cols);
    assert_eq!(dev_tm.rows, rows);

    let mut g_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut g_tm)?;

    for idx in 0..g_tm.len() {
        assert!(approx_eq_absrel(cpu_tm[idx], g_tm[idx] as f64, 1e-3, 1e-7), "mismatch at {}", idx);
    }

    Ok(())
}
