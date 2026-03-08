use vector_ta::indicators::ttm_squeeze::{
    ttm_squeeze_batch_with_kernel, ttm_squeeze_with_kernel, TtmSqueezeBatchRange, TtmSqueezeData,
    TtmSqueezeInput, TtmSqueezeParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaRuntime, CudaTtmSqueeze};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_ohlc_with_leading_nans_f64(
    len: usize,
    first_valid: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut high: Vec<f64> = (0..len)
        .map(|i| 100.0f64 + (i as f64 * 0.1) + ((i as f64) * 0.03).sin() + 1.0)
        .collect();
    let mut low: Vec<f64> = high
        .iter()
        .enumerate()
        .map(|(i, v)| v - 1.6 - ((i as f64) * 0.02).sin().abs() * 0.2)
        .collect();
    let mut close: Vec<f64> = low
        .iter()
        .zip(high.iter())
        .enumerate()
        .map(|(i, (l, h))| (l + h) * 0.5 + ((i as f64) * 0.05).sin() * 0.2)
        .collect();
    for series in [&mut high, &mut low, &mut close] {
        for value in series.iter_mut().take(first_valid.min(len)) {
            *value = f64::NAN;
        }
    }
    (high, low, close)
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
fn ttm_squeeze_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ttm_squeeze_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let (high, low, close) = sample_ohlc_with_leading_nans_f64(len, 2);
    let sweep = TtmSqueezeBatchRange {
        length: (10, 28, 6),
        bb_mult: (2.0, 2.0, 0.0),
        kc_high: (1.0, 1.0, 0.0),
        kc_mid: (1.5, 1.5, 0.0),
        kc_low: (2.0, 2.0, 0.0),
    };
    let cpu = ttm_squeeze_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;

    let h32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let l32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let c32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaTtmSqueeze::new(0).expect("CudaTtmSqueeze::new");
    let (mo_dev, sq_dev) = cuda
        .ttm_squeeze_batch_dev(&h32, &l32, &c32, &sweep)
        .expect("ttm squeeze batch dev");

    assert_eq!(cpu.rows, mo_dev.rows);
    assert_eq!(cpu.cols, mo_dev.cols);
    assert_eq!(cpu.rows, sq_dev.rows);
    assert_eq!(cpu.cols, sq_dev.cols);

    let mut mo_g = vec![0f32; mo_dev.len()];
    mo_dev.buf.copy_to(&mut mo_g)?;
    let mut sq_g = vec![0f32; sq_dev.len()];
    sq_dev.buf.copy_to(&mut sq_g)?;

    let tol = 1e-2;
    for idx in 0..cpu.rows * cpu.cols {
        let cm = cpu.momentum[idx];
        let gm = mo_g[idx] as f64;
        let cs = cpu.squeeze[idx];
        let gs = sq_g[idx] as f64;
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
            assert!(
                approx_eq(cs, gs, 1e-6),
                "squeeze mismatch at {} (cpu={} gpu={})",
                idx,
                cs,
                gs
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ttm_squeeze_cuda_device_inputs_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ttm_squeeze_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let (high_f64, low_f64, close_f64) = sample_ohlc_with_leading_nans_f64(len, 2);
    let high: Vec<f32> = high_f64.iter().map(|&v| v as f32).collect();
    let low: Vec<f32> = low_f64.iter().map(|&v| v as f32).collect();
    let close: Vec<f32> = close_f64.iter().map(|&v| v as f32).collect();
    let sweep = TtmSqueezeBatchRange {
        length: (10, 28, 6),
        bb_mult: (2.0, 2.0, 0.0),
        kc_high: (1.0, 1.0, 0.0),
        kc_mid: (1.5, 1.5, 0.0),
        kc_low: (2.0, 2.0, 0.0),
    };

    let cuda = CudaTtmSqueeze::new(0).expect("CudaTtmSqueeze::new");
    let (legacy_momentum, legacy_squeeze) = cuda
        .ttm_squeeze_batch_dev(&high, &low, &close, &sweep)
        .expect("legacy batch");

    let runtime = CudaRuntime::new(0).expect("runtime");
    let d_high = runtime.upload_f32(&high).expect("upload high");
    let d_low = runtime.upload_f32(&low).expect("upload low");
    let d_close = runtime.upload_f32(&close).expect("upload close");
    let (device_momentum, device_squeeze) = cuda
        .ttm_squeeze_batch_dev_from_device_inputs(
            d_high.buffer(),
            d_low.buffer(),
            d_close.buffer(),
            len,
            2,
            &sweep,
        )
        .expect("device inputs");

    for (legacy, device, label, tol) in [
        (legacy_momentum, device_momentum, "momentum", 1e-6f32),
        (legacy_squeeze, device_squeeze, "squeeze", 1e-6f32),
    ] {
        assert_eq!(legacy.rows, device.rows, "{label} rows");
        assert_eq!(legacy.cols, device.cols, "{label} cols");
        let mut legacy_host = vec![0f32; legacy.len()];
        legacy.buf.copy_to(&mut legacy_host)?;
        let mut device_host = vec![0f32; device.len()];
        device.buf.copy_to(&mut device_host)?;
        for (idx, (&lhs, &rhs)) in legacy_host.iter().zip(device_host.iter()).enumerate() {
            if lhs.is_nan() && rhs.is_nan() {
                continue;
            }
            assert!(
                (lhs - rhs).abs() < tol,
                "{label} mismatch at {idx}: legacy={lhs} device={rhs}"
            );
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ttm_squeeze_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ttm_squeeze_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
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

    let length = 20usize;
    let bb_mult = 2.0f64;
    let kc_high = 1.0f64;
    let kc_mid = 1.5f64;
    let kc_low = 2.0f64;

    let mut mo_cpu = vec![f64::NAN; cols * rows];
    let mut sq_cpu = vec![f64::NAN; cols * rows];
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
        let input = TtmSqueezeInput {
            data: TtmSqueezeData::Slices {
                high: &h,
                low: &l,
                close: &c,
            },
            params: TtmSqueezeParams {
                length: Some(length),
                bb_mult: Some(bb_mult),
                kc_mult_high: Some(kc_high),
                kc_mult_mid: Some(kc_mid),
                kc_mult_low: Some(kc_low),
            },
        };
        let out = ttm_squeeze_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let idx = t * cols + s;
            mo_cpu[idx] = out.momentum[t];
            sq_cpu[idx] = out.squeeze[t];
        }
    }

    let h32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let l32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let c32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaTtmSqueeze::new(0).expect("CudaTtmSqueeze");
    let (mo_tm, sq_tm) = cuda
        .ttm_squeeze_many_series_one_param_time_major_dev(
            &h32,
            &l32,
            &c32,
            cols,
            rows,
            length,
            bb_mult as f32,
            kc_high as f32,
            kc_mid as f32,
            kc_low as f32,
        )
        .expect("ttm many series");

    assert_eq!(mo_tm.rows, rows);
    assert_eq!(mo_tm.cols, cols);
    let mut mo_g = vec![0f32; mo_tm.len()];
    mo_tm.buf.copy_to(&mut mo_g)?;
    let mut sq_g = vec![0f32; sq_tm.len()];
    sq_tm.buf.copy_to(&mut sq_g)?;

    let tol = 1e-2;
    for idx in 0..rows * cols {
        let cm = mo_cpu[idx];
        let gm = mo_g[idx] as f64;
        if !(cm.is_nan() || gm.is_nan()) {
            assert!(approx_eq(cm, gm, tol), "momentum mismatch at {}", idx);
        }
        let cs = sq_cpu[idx];
        let gs = sq_g[idx] as f64;
        if !(cs.is_nan() || gs.is_nan()) {
            assert!(approx_eq(cs, gs, 1e-6), "squeeze mismatch at {}", idx);
        }
    }
    Ok(())
}
