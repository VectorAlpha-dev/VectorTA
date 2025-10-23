#![cfg(feature = "cuda")]

use my_project::cuda::cuda_available;
use cust::memory::CopyDestination;
use my_project::cuda::oscillators::CudaCci;
use my_project::indicators::cci::{CciBatchBuilder, CciBatchRange, CciBuilder, CciInput, CciParams};
use my_project::utilities::enums::Kernel;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    (a - b).abs() <= tol
}

fn main() {
    if !cuda_available() {
        eprintln!("No CUDA device; exiting");
        return;
    }

    // Reproduce test input
    let len = 4096usize;
    let mut price = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        price[i] = (x * 0.002).sin() + 0.0007 * x;
    }
    let sweep = CciBatchRange { period: (9, 64, 5) };

    // CPU baseline
    let cpu = CciBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slice(&price)
        .expect("cpu batch");

    // GPU
    let cuda = CudaCci::new(0).expect("cuda cci");
    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let dev = cuda.cci_batch_dev(&price_f32, &sweep).expect("gpu batch");
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host).expect("copy");

    // Optional: CPU f32 emulation for first row (period=9) to gauge fp32 vs fp64 gap
    let period = 9usize;
    let first_valid = price_f32.iter().position(|v| !v.is_nan()).unwrap();
    let warm = first_valid + period - 1;
    let mut out_f32 = vec![f32::NAN; len];
    let mut sum = 0.0f32;
    for k in 0..period { sum += price_f32[first_valid + k]; }
    let mut sma = sum / (period as f32);
    // first output
    let mut sum_abs = 0.0f32;
    for k in 0..period { sum_abs += (price_f32[warm - period + 1 + k] - sma).abs(); }
    let denom = 0.015f32 * (sum_abs / (period as f32));
    out_f32[warm] = if denom == 0.0 { 0.0 } else { (price_f32[warm] - sma) / denom };
    for t in (warm + 1)..len {
        sum += price_f32[t];
        sum -= price_f32[t - period];
        sma = sum / (period as f32);
        let mut sabs = 0.0f32;
        for k in 0..period { sabs += (price_f32[t - period + 1 + k] - sma).abs(); }
        let d = 0.015f32 * (sabs / (period as f32));
        out_f32[t] = if d == 0.0 { 0.0 } else { (price_f32[t] - sma) / d };
    }

    // Scan for first mismatch and print diagnostic
    let tol = 6e-4;
    for idx in 0..host.len() {
        let y = host[idx] as f64;
        let r = cpu.values[idx];
        if !approx_eq(r, y, tol) {
            let row = idx / len;
            let col = idx % len;
            println!("first mismatch @ idx={} (row={}, t={}): cpu={} gpu={} diff={} tol={}", idx, row, col, r, y, (r - y).abs(), tol);
            // Print a tiny neighborhood
            for dt in -2..=2 {
                let t = (col as isize + dt) as usize;
                let i2 = row * len + t;
                if i2 < host.len() {
                    println!(
                        "  t={:5} cpu={:.8e} gpu={:.8e} diff={:.3e}",
                        t,
                        cpu.values[i2],
                        host[i2] as f64,
                        (cpu.values[i2] - host[i2] as f64).abs()
                    );
                }
            }
            // Also compare fp32 emulation for period=9 on the same series
            println!("\nfp32 emu vs gpu (row 0):");
            for dt in -2..=2 {
                let t = (col as isize + dt) as usize;
                if t < len {
                    println!(
                        "  t={:5} emu={:.8e} gpu={:.8e} diff={:.3e}",
                        t,
                        out_f32[t] as f64,
                        host[t] as f64,
                        (out_f32[t] as f64 - host[t] as f64).abs()
                    );
                }
            }
            return;
        }
    }
    println!("No mismatch within tolerance");
}
