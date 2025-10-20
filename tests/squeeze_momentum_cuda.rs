// CUDA integration tests for Squeeze Momentum Indicator

use my_project::indicators::squeeze_momentum::{
    squeeze_momentum_batch_with_kernel, squeeze_momentum_with_kernel, SqueezeMomentumBatchRange,
    SqueezeMomentumBuilder, SqueezeMomentumData, SqueezeMomentumInput, SqueezeMomentumParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, CudaSqueezeMomentum};

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
fn squeeze_momentum_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[squeeze_momentum_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 2..len {
        let x = i as f64;
        high[i] = (x * 0.001).sin() + 0.5;
        low[i] = high[i] - 1.0;
        close[i] = (x * 0.0007).cos() + 0.1;
    }

    let sweep = SqueezeMomentumBatchRange {
        length_bb: (10, 28, 6),
        mult_bb: (2.0, 2.0, 0.0),
        length_kc: (12, 24, 6),
        mult_kc: (1.5, 1.5, 0.0),
    };
    let cpu = squeeze_momentum_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;

    let h32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let l32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let c32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaSqueezeMomentum::new(0).expect("CudaSqueezeMomentum::new");
    let (sq_dev, mo_dev, si_dev) = cuda
        .squeeze_momentum_batch_dev(&h32, &l32, &c32, &sweep)
        .expect("smi batch dev");

    assert_eq!(cpu.rows, sq_dev.rows);
    assert_eq!(cpu.cols, sq_dev.cols);
    assert_eq!(cpu.rows, mo_dev.rows);
    assert_eq!(cpu.cols, mo_dev.cols);
    assert_eq!(cpu.rows, si_dev.rows);
    assert_eq!(cpu.cols, si_dev.cols);

    let mut sq_g = vec![0f32; sq_dev.len()];
    sq_dev.buf.copy_to(&mut sq_g)?;
    let mut mo_g = vec![0f32; mo_dev.len()];
    mo_dev.buf.copy_to(&mut mo_g)?;
    let mut si_g = vec![0f32; si_dev.len()];
    si_dev.buf.copy_to(&mut si_g)?;

    let tol = 2e-3; // allow some FP32 drift
    for idx in 0..cpu.rows * cpu.cols {
        if !approx_eq(cpu.squeeze[idx], sq_g[idx] as f64, 1e-6) {
            eprintln!(
                "First squeeze mismatch at {}: cpu={} gpu={}",
                idx, cpu.squeeze[idx], sq_g[idx]
            );
            // Dump a small window for debugging
            let start = idx.saturating_sub(5);
            let end = (idx + 6).min(cpu.rows * cpu.cols);
            for k in start..end {
                eprintln!("  [{}] cpu={} gpu={}", k, cpu.squeeze[k], sq_g[k]);
            }
            assert!(false, "squeeze mismatch at {}", idx);
        }
        let cm = cpu.momentum[idx];
        let gm = mo_g[idx] as f64;
        let cs = cpu.signal[idx];
        let gs = si_g[idx] as f64;
        if !(cm.is_nan() || gm.is_nan()) {
            assert!(
                approx_eq(cm, gm, tol),
                "momentum mismatch at {} (cpu={} gpu={})",
                idx,
                cm,
                gm
            );
        }
        if !(cs.is_nan() || gs.is_nan()) {
            assert!(approx_eq(cs, gs, 1e-5), "signal mismatch at {}", idx);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn squeeze_momentum_cuda_many_series_one_param_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[squeeze_momentum_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let cols = 6usize;
    let rows = 2048usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + 0.1 * (s as f64);
            let h = (x * 0.002).sin() + 1.0;
            high_tm[t * cols + s] = h;
            low_tm[t * cols + s] = h - 1.0;
            close_tm[t * cols + s] = (x * 0.0013).cos() + 0.2;
        }
    }

    let lbb = 20usize;
    let mbb = 2.0f64;
    let lkc = 20usize;
    let mkc = 1.5f64;

    // CPU baseline (per series)
    let mut sq_cpu = vec![f64::NAN; cols * rows];
    let mut mo_cpu = vec![f64::NAN; cols * rows];
    let mut si_cpu = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            h[t] = high_tm[idx];
            l[t] = low_tm[idx];
            c[t] = close_tm[idx];
        }
        let input = SqueezeMomentumInput {
            data: SqueezeMomentumData::Slices {
                high: &h,
                low: &l,
                close: &c,
            },
            params: SqueezeMomentumParams {
                length_bb: Some(lbb),
                mult_bb: Some(mbb),
                length_kc: Some(lkc),
                mult_kc: Some(mkc),
            },
        };
        let out = squeeze_momentum_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let idx = t * cols + s;
            sq_cpu[idx] = out.squeeze[t];
            mo_cpu[idx] = out.momentum[t];
            si_cpu[idx] = out.momentum_signal[t];
        }
    }

    // GPU path
    let h32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let l32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let c32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaSqueezeMomentum::new(0).expect("CudaSqueezeMomentum");
    let (sq_tm, mo_tm, si_tm) = cuda
        .squeeze_momentum_many_series_one_param_time_major_dev(
            &h32, &l32, &c32, cols, rows, lbb, mbb as f32, lkc, mkc as f32,
        )
        .expect("smi many series");

    assert_eq!(sq_tm.rows, rows);
    assert_eq!(sq_tm.cols, cols);
    let mut sq_g = vec![0f32; sq_tm.len()];
    sq_tm.buf.copy_to(&mut sq_g)?;
    let mut mo_g = vec![0f32; mo_tm.len()];
    mo_tm.buf.copy_to(&mut mo_g)?;
    let mut si_g = vec![0f32; si_tm.len()];
    si_tm.buf.copy_to(&mut si_g)?;

    let tol = 2e-3;
    for idx in 0..rows * cols {
        if !approx_eq(sq_cpu[idx], sq_g[idx] as f64, 1e-6) {
            eprintln!(
                "First squeeze mismatch (many-series) at {}: cpu={} gpu={}",
                idx, sq_cpu[idx], sq_g[idx]
            );
            let start = idx.saturating_sub(5);
            let end = (idx + 6).min(rows * cols);
            for k in start..end {
                eprintln!("  [{}] cpu={} gpu={}", k, sq_cpu[k], sq_g[k]);
            }
            assert!(false, "squeeze mismatch at {}", idx);
        }
        let cm = mo_cpu[idx];
        let gm = mo_g[idx] as f64;
        if !(cm.is_nan() || gm.is_nan()) {
            assert!(approx_eq(cm, gm, tol), "momentum mismatch at {}", idx);
        }
        let cs = si_cpu[idx];
        let gs = si_g[idx] as f64;
        if !(cs.is_nan() || gs.is_nan()) {
            assert!(approx_eq(cs, gs, 1e-5), "signal mismatch at {}", idx);
        }
    }
    Ok(())
}
