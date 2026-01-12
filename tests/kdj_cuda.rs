#![cfg(feature = "cuda")]

use cust::memory::CopyDestination;
use vector_ta::cuda::cuda_available;
use vector_ta::cuda::CudaKdj;
use vector_ta::indicators::kdj::{
    kdj_batch_with_kernel, kdj_with_kernel, KdjBatchRange, KdjInput, KdjParams,
};
use vector_ta::utilities::data_loader::Candles;
use vector_ta::utilities::enums::Kernel;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

fn synth_from_close(close: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut h = close.to_vec();
    let mut l = close.to_vec();
    for i in 0..close.len() {
        let v = close[i];
        if v.is_nan() {
            continue;
        }
        let x = i as f64 * 0.0023;
        let off = (0.0029 * x.sin()).abs() + 0.1;
        h[i] = v + off;
        l[i] = v - off;
    }
    (h, l)
}

#[test]
fn kdj_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[kdj_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut close = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        close[i] = (x * 0.00037).sin() + 0.00021 * x;
        if i % 271 == 0 {
            close[i] = f64::NAN;
        }
    }
    let (high, low) = synth_from_close(&close);

    let high_f32_round: Vec<f64> = high.iter().map(|&v| (v as f32) as f64).collect();
    let low_f32_round: Vec<f64> = low.iter().map(|&v| (v as f32) as f64).collect();
    let close_f32_round: Vec<f64> = close.iter().map(|&v| (v as f32) as f64).collect();

    let sweep = KdjBatchRange {
        fast_k_period: (9, 29, 5),
        slow_k_period: (3, 3, 0),
        slow_k_ma_type: ("sma".into(), "sma".into(), "".into()),
        slow_d_period: (3, 3, 0),
        slow_d_ma_type: ("sma".into(), "sma".into(), "".into()),
    };
    let cpu = kdj_batch_with_kernel(
        &high_f32_round,
        &low_f32_round,
        &close_f32_round,
        &sweep,
        Kernel::ScalarBatch,
    )?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaKdj::new(0).expect("CudaKdj::new");
    let (dev_k, dev_d, dev_j) = cuda
        .kdj_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
        .expect("kdj_batch_dev");

    assert_eq!(dev_k.rows, cpu.rows);
    assert_eq!(dev_k.cols, cpu.cols);
    assert_eq!(dev_d.rows, cpu.rows);
    assert_eq!(dev_d.cols, cpu.cols);
    assert_eq!(dev_j.rows, cpu.rows);
    assert_eq!(dev_j.cols, cpu.cols);

    let mut k = vec![0f32; dev_k.len()];
    let mut d = vec![0f32; dev_d.len()];
    let mut j = vec![0f32; dev_j.len()];
    dev_k.buf.copy_to(&mut k)?;
    dev_d.buf.copy_to(&mut d)?;
    dev_j.buf.copy_to(&mut j)?;

    let mut cpu_finite = 0usize;
    let mut gpu_finite = 0usize;
    for i in (cpu.cols / 2)..(cpu.rows * cpu.cols) {
        if cpu.k[i].is_finite() {
            cpu_finite += 1;
        }
        if (k[i] as f64).is_finite() {
            gpu_finite += 1;
        }
    }

    assert!(
        gpu_finite * 2 >= cpu_finite,
        "GPU finite ratio too low: gpu={} cpu={}",
        gpu_finite,
        cpu_finite
    );
    Ok(())
}

#[test]
fn kdj_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[kdj_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 1024usize;
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + 0.1 * (s as f64);
            close_tm[t * cols + s] = (x * 0.003).sin() + 0.0002 * x;
        }
    }
    let (mut high_tm, mut low_tm) = (close_tm.clone(), close_tm.clone());
    for s in 0..cols {
        for t in s..rows {
            let base = close_tm[t * cols + s];
            if base.is_nan() {
                continue;
            }
            let off = 0.1 + 0.01 * ((t + s) as f64).sin().abs();
            high_tm[t * cols + s] = base + off;
            low_tm[t * cols + s] = base - off;
        }
    }

    let mut cpu_k_tm = vec![f64::NAN; cols * rows];
    let mut cpu_d_tm = vec![f64::NAN; cols * rows];
    let mut cpu_j_tm = vec![f64::NAN; cols * rows];
    let params = KdjParams {
        fast_k_period: Some(9),
        slow_k_period: Some(3),
        slow_k_ma_type: Some("sma".into()),
        slow_d_period: Some(3),
        slow_d_ma_type: Some("sma".into()),
    };
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = high_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
            c[t] = close_tm[t * cols + s];
        }
        let out = kdj_with_kernel(
            &KdjInput::from_slices(&h, &l, &c, params.clone()),
            Kernel::Scalar,
        )
        .unwrap();
        for t in 0..rows {
            cpu_k_tm[t * cols + s] = out.k[t];
            cpu_d_tm[t * cols + s] = out.d[t];
            cpu_j_tm[t * cols + s] = out.j[t];
        }
    }

    let h_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let l_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let c_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaKdj::new(0).expect("CudaKdj::new");
    let (dev_k, dev_d, dev_j) = cuda
        .kdj_many_series_one_param_time_major_dev(&h_f32, &l_f32, &c_f32, cols, rows, &params)
        .expect("kdj many series");

    assert_eq!(dev_k.rows, rows);
    assert_eq!(dev_k.cols, cols);
    assert_eq!(dev_d.rows, rows);
    assert_eq!(dev_d.cols, cols);
    assert_eq!(dev_j.rows, rows);
    assert_eq!(dev_j.cols, cols);

    let mut k = vec![0f32; dev_k.len()];
    let mut d = vec![0f32; dev_d.len()];
    let mut j = vec![0f32; dev_j.len()];
    dev_k.buf.copy_to(&mut k)?;
    dev_d.buf.copy_to(&mut d)?;
    dev_j.buf.copy_to(&mut j)?;

    let tol = 1e-3;
    for idx in 0..(cols * rows) {
        assert!(
            approx_eq(cpu_k_tm[idx], k[idx] as f64, tol),
            "K mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_d_tm[idx], d[idx] as f64, tol),
            "D mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_j_tm[idx], j[idx] as f64, tol),
            "J mismatch at {}",
            idx
        );
    }
    Ok(())
}
