use vector_ta::indicators::supertrend::{
    supertrend_batch_with_kernel, supertrend_with_kernel, SuperTrendBatchRange, SuperTrendData,
    SuperTrendInput, SuperTrendParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaRuntime, CudaSupertrend};

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
fn supertrend_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[supertrend_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        let c = (x * 0.00123).sin() + 0.00017 * x;
        let off = (0.004 * (x * 0.002).sin()).abs() + 0.12;
        close[i] = c;
        high[i] = c + off;
        low[i] = c - off;
    }
    let sweep = SuperTrendBatchRange {
        period: (7, 23, 4),
        factor: (2.0, 3.5, 0.5),
    };

    let cpu = supertrend_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;

    let h32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let l32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let c32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaSupertrend::new(0).expect("CudaSupertrend::new");
    let (trend_dev, changed_dev, combos) = cuda
        .supertrend_batch_dev(&h32, &l32, &c32, &sweep)
        .expect("supertrend_cuda_batch_dev");

    assert_eq!(cpu.rows, combos.len());
    assert_eq!(cpu.rows, trend_dev.rows);
    assert_eq!(cpu.cols, trend_dev.cols);
    assert_eq!(cpu.rows, changed_dev.rows);
    assert_eq!(cpu.cols, changed_dev.cols);

    let mut g_trend = vec![0f32; trend_dev.len()];
    trend_dev.buf.copy_to(&mut g_trend)?;
    let mut g_changed = vec![0f32; changed_dev.len()];
    changed_dev.buf.copy_to(&mut g_changed)?;

    let tol_trend = 7e-4;
    let tol_changed = 1e-6;
    for idx in 0..(cpu.rows * cpu.cols) {
        let ct = cpu.trend[idx];
        let gt = g_trend[idx] as f64;
        let cc = cpu.changed[idx];
        let gc = g_changed[idx] as f64;
        assert!(
            approx_eq(ct, gt, tol_trend),
            "trend mismatch at {}: cpu={} gpu={}",
            idx,
            ct,
            gt
        );
        assert!(
            approx_eq(cc, gc, tol_changed),
            "changed mismatch at {}: cpu={} gpu={}",
            idx,
            cc,
            gc
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn supertrend_cuda_device_inputs_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[supertrend_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut high = vec![f32::NAN; len];
    let mut low = vec![f32::NAN; len];
    let mut close = vec![f32::NAN; len];
    for i in 4..len {
        let x = i as f32;
        let c = (x * 0.00123).sin() + 0.00017 * x;
        let off = (0.004 * (x * 0.002).sin()).abs() + 0.12;
        close[i] = c;
        high[i] = c + off;
        low[i] = c - off;
    }
    let sweep = SuperTrendBatchRange {
        period: (7, 23, 4),
        factor: (2.0, 3.5, 0.5),
    };

    let cuda = CudaSupertrend::new(0).expect("CudaSupertrend::new");
    let (legacy_trend, legacy_changed, legacy_combos) = cuda
        .supertrend_batch_dev(&high, &low, &close, &sweep)
        .expect("legacy batch");

    let runtime = CudaRuntime::new(0).expect("runtime");
    let d_high = runtime.upload_f32(&high).expect("upload high");
    let d_low = runtime.upload_f32(&low).expect("upload low");
    let d_close = runtime.upload_f32(&close).expect("upload close");
    let (device_trend, device_changed, device_combos) = cuda
        .supertrend_batch_dev_from_device_inputs(
            d_high.buffer(),
            d_low.buffer(),
            d_close.buffer(),
            len,
            4,
            &sweep,
        )
        .expect("device inputs");

    assert_eq!(legacy_combos.len(), device_combos.len(), "combos len");
    for (idx, (legacy, device)) in legacy_combos.iter().zip(device_combos.iter()).enumerate() {
        assert_eq!(legacy.period, device.period, "combo period at {idx}");
        assert_eq!(legacy.factor, device.factor, "combo factor at {idx}");
    }
    for (legacy, device, label, tol) in [
        (legacy_trend, device_trend, "trend", 1e-6f32),
        (legacy_changed, device_changed, "changed", 1e-6f32),
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
fn supertrend_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[supertrend_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            let c = (x * 0.002).sin() + 0.0003 * x;
            let off = (0.004 * (x * 0.0017).cos()).abs() + 0.11;
            close_tm[t * cols + s] = c;
            high_tm[t * cols + s] = c + off;
            low_tm[t * cols + s] = c - off;
        }
    }

    let period = 10usize;
    let factor = 3.0f64;

    let mut cpu_trend_tm = vec![f64::NAN; cols * rows];
    let mut cpu_changed_tm = vec![f64::NAN; cols * rows];
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
        let params = SuperTrendParams {
            period: Some(period),
            factor: Some(factor),
        };
        let input = SuperTrendInput {
            data: SuperTrendData::Slices {
                high: &h,
                low: &l,
                close: &c,
            },
            params,
        };
        let out = supertrend_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_trend_tm[t * cols + s] = out.trend[t];
            cpu_changed_tm[t * cols + s] = out.changed[t];
        }
    }

    let h32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let l32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let c32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaSupertrend::new(0).expect("CudaSupertrend::new");
    let dev = cuda
        .supertrend_many_series_one_param_time_major_dev(
            &h32,
            &l32,
            &c32,
            cols,
            rows,
            period,
            factor as f32,
        )
        .expect("supertrend_many_series_one_param_time_major_dev");

    assert_eq!(dev.plus.rows, rows);
    assert_eq!(dev.plus.cols, cols);
    assert_eq!(dev.minus.rows, rows);
    assert_eq!(dev.minus.cols, cols);

    let mut g_trend_tm = vec![0f32; dev.plus.len()];
    let mut g_changed_tm = vec![0f32; dev.minus.len()];
    dev.plus.buf.copy_to(&mut g_trend_tm)?;
    dev.minus.buf.copy_to(&mut g_changed_tm)?;

    let tol_trend = 7e-4;
    let tol_changed = 1e-6;
    for idx in 0..g_trend_tm.len() {
        assert!(
            approx_eq(cpu_trend_tm[idx], g_trend_tm[idx] as f64, tol_trend),
            "trend mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_changed_tm[idx], g_changed_tm[idx] as f64, tol_changed),
            "changed mismatch at {}",
            idx
        );
    }

    Ok(())
}
