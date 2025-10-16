// Integration tests for CUDA Band-Pass kernels

use my_project::utilities::enums::Kernel;
use my_project::indicators::bandpass::{
    bandpass_batch_with_kernel, bandpass_with_kernel, BandPassBatchRange, BandPassInput,
    BandPassParams,
};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::bandpass_wrapper::CudaBandpass;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
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
fn bandpass_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[bandpass_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 4..len { let x = i as f64; price[i] = (x * 0.0031).sin() + 0.00013 * x; }
    let sweep = BandPassBatchRange { period: (12, 24, 3), bandwidth: (0.2, 0.4, 0.1) };

    let cpu = bandpass_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaBandpass::new(0).expect("CudaBandpass::new");
    let res = cuda.bandpass_batch_dev(&price_f32, &sweep).expect("cuda bandpass_batch_dev");
    let out = res.outputs;

    assert_eq!(cpu.rows, out.rows());
    assert_eq!(cpu.cols, out.cols());

    let mut bp_host = vec![0f32; out.first.len()];
    let mut bpn_host = vec![0f32; out.second.len()];
    let mut sig_host = vec![0f32; out.third.len()];
    let mut trg_host = vec![0f32; out.fourth.len()];
    out.first.buf.copy_to(&mut bp_host)?;
    out.second.buf.copy_to(&mut bpn_host)?;
    out.third.buf.copy_to(&mut sig_host)?;
    out.fourth.buf.copy_to(&mut trg_host)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(approx_eq(cpu.bp[idx], bp_host[idx] as f64, tol), "bp mismatch at {}", idx);
        assert!(
            approx_eq(cpu.bp_normalized[idx], bpn_host[idx] as f64, tol),
            "bpn mismatch at {}",
            idx
        );
        assert!(approx_eq(cpu.signal[idx], sig_host[idx] as f64, 1e-3), "sig mismatch at {}", idx);
        assert!(approx_eq(cpu.trigger[idx], trg_host[idx] as f64, tol), "trg mismatch at {}", idx);
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn bandpass_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[bandpass_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let cols = 8usize; // series
    let rows = 2048usize; // length
    let mut price_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.17;
            price_tm[t * cols + s] = (x * 0.003).sin() + 0.0002 * x;
        }
    }
    let period = 20usize;
    let bandwidth = 0.3f64;

    // CPU baseline per series
    let mut cpu_bp_tm = vec![f64::NAN; cols * rows];
    let mut cpu_bpn_tm = vec![f64::NAN; cols * rows];
    let mut cpu_sig_tm = vec![f64::NAN; cols * rows];
    let mut cpu_trg_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        for t in 0..rows { p[t] = price_tm[t * cols + s]; }
        let params = BandPassParams { period: Some(period), bandwidth: Some(bandwidth) };
        let input = BandPassInput::from_slice(&p, params);
        let out = bandpass_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_bp_tm[t * cols + s] = out.bp[t];
            cpu_bpn_tm[t * cols + s] = out.bp_normalized[t];
            cpu_sig_tm[t * cols + s] = out.signal[t];
            cpu_trg_tm[t * cols + s] = out.trigger[t];
        }
    }

    let price_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaBandpass::new(0).expect("CudaBandpass::new");
    let out = cuda
        .bandpass_many_series_one_param_time_major_dev(
            &price_tm_f32,
            cols,
            rows,
            &BandPassParams { period: Some(period), bandwidth: Some(bandwidth) },
        )
        .expect("bandpass many-series");

    assert_eq!(out.rows(), rows);
    assert_eq!(out.cols(), cols);
    let mut g_bp = vec![0f32; out.first.len()];
    let mut g_bpn = vec![0f32; out.second.len()];
    let mut g_sig = vec![0f32; out.third.len()];
    let mut g_trg = vec![0f32; out.fourth.len()];
    out.first.buf.copy_to(&mut g_bp)?;
    out.second.buf.copy_to(&mut g_bpn)?;
    out.third.buf.copy_to(&mut g_sig)?;
    out.fourth.buf.copy_to(&mut g_trg)?;

    let tol = 5e-4;
    for idx in 0..g_bp.len() {
        assert!(approx_eq(cpu_bp_tm[idx], g_bp[idx] as f64, tol), "bp mismatch at {}", idx);
        assert!(approx_eq(cpu_bpn_tm[idx], g_bpn[idx] as f64, tol), "bpn mismatch at {}", idx);
        assert!(approx_eq(cpu_sig_tm[idx], g_sig[idx] as f64, 1e-3), "sig mismatch at {}", idx);
        assert!(approx_eq(cpu_trg_tm[idx], g_trg[idx] as f64, tol), "trg mismatch at {}", idx);
    }

    Ok(())
}

