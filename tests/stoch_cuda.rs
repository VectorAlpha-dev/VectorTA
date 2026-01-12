#![cfg(feature = "cuda")]

use cust::memory::CopyDestination;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use vector_ta::cuda::cuda_available;
use vector_ta::cuda::oscillators::CudaStoch;
use vector_ta::indicators::stoch::{
    stoch_batch_with_kernel, stoch_with_kernel, StochBatchRange, StochInput, StochParams,
};
use vector_ta::utilities::data_loader::Candles;
use vector_ta::utilities::enums::Kernel;

static CUDA_TEST_MUTEX: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

fn gen_series(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64 * 0.01;
            (x.sin() * 2.0 + x.cos() * 0.5) + 100.0
        })
        .collect()
}

fn synth_hlc(close: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut high = close.to_vec();
    let mut low = close.to_vec();
    for i in 0..close.len() {
        let v = close[i];
        let x = i as f64 * 0.0031;
        let off = (0.0077 * x.sin()).abs() + 0.2;
        high[i] = v + off;
        low[i] = v - off;
    }
    (high, low)
}

#[test]
fn stoch_cuda_batch_matches_cpu() {
    let _guard = CUDA_TEST_MUTEX.lock().unwrap();
    if !cuda_available() {
        eprintln!("CUDA not available; skipping stoch_cuda_batch_matches_cpu");
        return;
    }
    let n = 5000;
    let close = gen_series(n);
    let (high, low) = synth_hlc(&close);
    let high32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close32: Vec<f32> = close.iter().map(|&v| v as f32).collect();

    let sweep = StochBatchRange {
        fastk_period: (10, 22, 3),
        slowk_period: (3, 5, 2),
        slowk_ma_type: ("sma".into(), "sma".into(), 0.0),
        slowd_period: (3, 4, 1),
        slowd_ma_type: ("ema".into(), "ema".into(), 0.0),
    };

    let cuda = CudaStoch::new(0).expect("cuda stoch");
    let batch = cuda
        .stoch_batch_dev(&high32, &low32, &close32, &sweep)
        .expect("stoch batch dev");

    let cpu = stoch_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)
        .expect("cpu batch");

    let mut k_gpu = vec![0f32; batch.k.rows * batch.k.cols];
    let mut d_gpu = vec![0f32; batch.d.rows * batch.d.cols];
    batch.k.buf.copy_to(&mut k_gpu).unwrap();
    batch.d.buf.copy_to(&mut d_gpu).unwrap();

    assert_eq!(cpu.rows * cpu.cols, k_gpu.len());
    assert_eq!(cpu.rows * cpu.cols, d_gpu.len());

    for i in 0..(cpu.rows * cpu.cols) {
        let k_ref = cpu.k[i] as f32;
        let d_ref = cpu.d[i] as f32;
        let k_v = k_gpu[i];
        let d_v = d_gpu[i];
        if k_ref.is_finite() && k_v.is_finite() {
            assert!(
                (k_ref - k_v).abs() <= 1e-2 || k_ref.to_bits().abs_diff(k_v.to_bits()) <= 32,
                "K mismatch at {}: {} vs {}",
                i,
                k_v,
                k_ref
            );
        }
        if d_ref.is_finite() && d_v.is_finite() {
            assert!(
                (d_ref - d_v).abs() <= 1e-2 || d_ref.to_bits().abs_diff(d_v.to_bits()) <= 32,
                "D mismatch at {}: {} vs {}",
                i,
                d_v,
                d_ref
            );
        }
    }
}

#[test]
fn stoch_cuda_many_series_time_major_matches_cpu() {
    let _guard = CUDA_TEST_MUTEX.lock().unwrap();
    if !cuda_available() {
        eprintln!("CUDA not available; skipping stoch_cuda_many_series_time_major_matches_cpu");
        return;
    }
    let cols = 8usize;
    let rows = 4096usize;
    let mut close_tm = vec![f32::NAN; cols * rows];
    let mut high_tm = vec![f32::NAN; cols * rows];
    let mut low_tm = vec![f32::NAN; cols * rows];
    for s in 0..cols {
        let c = gen_series(rows);
        let (h, l) = synth_hlc(&c);
        for r in 0..rows {
            let idx = r * cols + s;
            close_tm[idx] = c[r] as f32;
            high_tm[idx] = h[r] as f32;
            low_tm[idx] = l[r] as f32;
        }
    }
    let params = StochParams {
        fastk_period: Some(14),
        slowk_period: Some(3),
        slowk_ma_type: Some("sma".into()),
        slowd_period: Some(3),
        slowd_ma_type: Some("sma".into()),
    };
    let cuda = CudaStoch::new(0).expect("cuda stoch");
    let (k_tm, d_tm) = cuda
        .stoch_many_series_one_param_time_major_dev(
            &high_tm, &low_tm, &close_tm, cols, rows, &params,
        )
        .expect("stoch many-series dev");
    let mut k_gpu = vec![0f32; cols * rows];
    let mut d_gpu = vec![0f32; cols * rows];
    k_tm.buf.copy_to(&mut k_gpu).unwrap();
    d_tm.buf.copy_to(&mut d_gpu).unwrap();

    for s in 0..cols {
        let mut close = vec![0f64; rows];
        let mut high = vec![0f64; rows];
        let mut low = vec![0f64; rows];
        for r in 0..rows {
            let idx = r * cols + s;
            close[r] = close_tm[idx] as f64;
            high[r] = high_tm[idx] as f64;
            low[r] = low_tm[idx] as f64;
        }
        let input = StochInput::from_slices(&high, &low, &close, params.clone());
        let out = stoch_with_kernel(&input, Kernel::Scalar).expect("cpu stoch");
        let warm = {
            let first = (0..rows)
                .find(|&i| high[i].is_finite() && low[i].is_finite() && close[i].is_finite())
                .unwrap();
            first + params.fastk_period.unwrap() - 1
        };
        for r in warm..rows {
            let k_ref = out.k[r] as f32;
            let d_ref = out.d[r] as f32;
            let k_v = k_gpu[r * cols + s];
            let d_v = d_gpu[r * cols + s];
            if k_ref.is_finite() && k_v.is_finite() {
                assert!(
                    (k_ref - k_v).abs() <= 1e-3,
                    "K col{} row{}: {} vs {}",
                    s,
                    r,
                    k_v,
                    k_ref
                );
            }
            if d_ref.is_finite() && d_v.is_finite() {
                assert!(
                    (d_ref - d_v).abs() <= 1e-3,
                    "D col{} row{}: {} vs {}",
                    s,
                    r,
                    d_v,
                    d_ref
                );
            }
        }
    }
}
