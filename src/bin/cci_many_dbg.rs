#![cfg(feature = "cuda")]

use cust::memory::CopyDestination;
use my_project::cuda::cuda_available;
use my_project::cuda::oscillators::CudaCci;
use my_project::indicators::cci::{CciInput, CciParams};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    (a - b).abs() <= tol
}

fn main() {
    if !cuda_available() {
        eprintln!("No CUDA device; exiting");
        return;
    }
    let cols = 8usize;
    let rows = 2048usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for r in s..rows {
            let x = r as f64 + 0.13 * s as f64;
            tm[r * cols + s] = (x * 0.0023).sin() + 0.0002 * x;
        }
    }
    let period = 14usize;

    // CPU per-series baseline
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut col = vec![f64::NAN; rows];
        for r in 0..rows { col[r] = tm[r * cols + s]; }
        let input = CciInput::from_slice(&col, CciParams { period: Some(period) });
        let out = my_project::indicators::cci::cci_with_kernel(&input, my_project::utilities::enums::Kernel::Scalar).unwrap().values;
        for r in 0..rows { cpu_tm[r * cols + s] = out[r]; }
    }

    // GPU
    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaCci::new(0).expect("CudaCci::new");
    let dev = cuda
        .cci_many_series_one_param_time_major_dev(&tm_f32, cols, rows, period)
        .expect("cci many-series");
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host).expect("copy");

    let mut max_diff = 0.0f64;
    let mut max_idx = 0usize;
    for idx in 0..host.len() {
        let y = host[idx] as f64;
        let r = cpu_tm[idx];
        let d = (r - y).abs();
        if d > max_diff { max_diff = d; max_idx = idx; }
    }
    let s = max_idx % cols; let t = max_idx / cols;
    println!(
        "max diff = {} at idx {} (s={}, t={}) cpu={} gpu={}",
        max_diff, max_idx, s, t, cpu_tm[max_idx], host[max_idx]
    );
}
