#![cfg(feature = "cuda")]

use vector_ta::indicators::dvdiqqe::{
    dvdiqqe_batch_with_kernel, dvdiqqe_with_kernel, DvdiqqeBatchRange, DvdiqqeInput, DvdiqqeParams,
};
use vector_ta::utilities::enums::Kernel;

use cust::memory::CopyDestination;
use vector_ta::cuda::{cuda_available, CudaDvdiqqe, CudaRuntime};

fn approx(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[test]
fn dvdiqqe_cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[test]
fn dvdiqqe_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dvdiqqe_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut close = vec![f64::NAN; len];
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64 * 0.0021;
        let c = (x).sin() + 0.0002 * (i as f64);
        close[i] = c;
        open[i] = c - 0.15 + (x * 0.3).cos();
        high[i] = c + 0.2;
        low[i] = c - 0.2;
        volume[i] = (0.6 + (x * 0.7).cos().abs()).max(0.0);
    }
    let sweep = DvdiqqeBatchRange {
        period: (10, 28, 3),
        smoothing_period: (4, 10, 3),
        fast_multiplier: (1.5, 3.0, 0.5),
        slow_multiplier: (3.0, 6.0, 1.0),
    };

    let cpu = dvdiqqe_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        Some(&volume),
        &sweep,
        Kernel::ScalarBatch,
        "default",
        "dynamic",
        0.01,
    )?;

    let o_f32: Vec<f32> = open.iter().map(|&v| v as f32).collect();
    let c_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let v_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();

    let cuda = CudaDvdiqqe::new(0).unwrap();
    let gpu = cuda
        .dvdiqqe_batch_dev(
            &o_f32,
            &c_f32,
            Some(&v_f32),
            &sweep,
            "default",
            "dynamic",
            0.01,
        )
        .unwrap();

    assert_eq!(gpu.dvdi.rows, cpu.rows);
    assert_eq!(gpu.dvdi.cols, cpu.cols);
    let plane = cpu.rows * cpu.cols;
    let mut g_dvdi = vec![0f32; plane];
    let mut g_fast = vec![0f32; plane];
    let mut g_slow = vec![0f32; plane];
    let mut g_cent = vec![0f32; plane];
    gpu.dvdi.buf.copy_to(&mut g_dvdi)?;
    gpu.fast.buf.copy_to(&mut g_fast)?;
    gpu.slow.buf.copy_to(&mut g_slow)?;
    gpu.center.buf.copy_to(&mut g_cent)?;

    for r in 0..cpu.rows {
        let period = cpu.combos[r].period.unwrap();
        let warm = close.iter().position(|x| x.is_finite()).unwrap() + (2 * period - 1);
        for c in (warm + 1)..cpu.cols {
            let idx = r * cpu.cols + c;

            assert!(
                approx(cpu.dvdi_values[idx], g_dvdi[idx] as f64, 1e-1),
                "dvdi mismatch at r={}, c={}",
                r,
                c
            );
            assert!(
                approx(cpu.fast_tl_values[idx], g_fast[idx] as f64, 1e-1),
                "fast mismatch at r={}, c={}",
                r,
                c
            );
            assert!(
                approx(cpu.slow_tl_values[idx], g_slow[idx] as f64, 1e-1),
                "slow mismatch at r={}, c={}",
                r,
                c
            );
            assert!(
                approx(cpu.center_values[idx], g_cent[idx] as f64, 3e-2),
                "center mismatch at r={}, c={}",
                r,
                c
            );
        }
    }
    Ok(())
}

#[test]
fn dvdiqqe_cuda_device_inputs_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dvdiqqe_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut open = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 6..len {
        let x = i as f64 * 0.0024;
        let base = 100.0 + (x * 0.73).sin() * 0.8 + 0.0004 * i as f64;
        close[i] = base + (x * 0.41).cos() * 0.03;
        open[i] = base - 0.12 + (x * 0.17).sin() * 0.05;
        volume[i] = ((x * 0.91).cos().abs() + 1.0) * 750.0;
    }
    let sweep = DvdiqqeBatchRange {
        period: (10, 22, 4),
        smoothing_period: (4, 8, 2),
        fast_multiplier: (1.5, 2.5, 0.5),
        slow_multiplier: (3.0, 5.0, 1.0),
    };

    let open_f32: Vec<f32> = open.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let first_valid = close_f32
        .iter()
        .position(|value| value.is_finite())
        .expect("first valid");

    let runtime = CudaRuntime::new(0).expect("CudaRuntime::new");
    let d_open = runtime.upload_f32(&open_f32).expect("upload open");
    let d_close = runtime.upload_f32(&close_f32).expect("upload close");
    let d_volume = runtime.upload_f32(&volume_f32).expect("upload volume");
    let cuda = CudaDvdiqqe::new(0).expect("CudaDvdiqqe::new");

    let legacy = cuda
        .dvdiqqe_batch_dev(
            &open_f32,
            &close_f32,
            Some(&volume_f32),
            &sweep,
            "default",
            "dynamic",
            0.01,
        )
        .expect("legacy dvdiqqe");
    let device = cuda
        .dvdiqqe_batch_dev_from_device_inputs(
            d_open.buffer(),
            d_close.buffer(),
            Some(d_volume.buffer()),
            len,
            first_valid,
            &sweep,
            "default",
            "dynamic",
            0.01,
        )
        .expect("device dvdiqqe");
    cuda.synchronize().expect("dvdiqqe sync");

    assert_eq!(legacy.dvdi.rows, device.dvdi.rows);
    assert_eq!(legacy.dvdi.cols, device.dvdi.cols);

    let mut legacy_dvdi = vec![0f32; legacy.dvdi.len()];
    let mut legacy_fast = vec![0f32; legacy.fast.len()];
    let mut legacy_slow = vec![0f32; legacy.slow.len()];
    let mut legacy_center = vec![0f32; legacy.center.len()];
    legacy.dvdi.buf.copy_to(&mut legacy_dvdi)?;
    legacy.fast.buf.copy_to(&mut legacy_fast)?;
    legacy.slow.buf.copy_to(&mut legacy_slow)?;
    legacy.center.buf.copy_to(&mut legacy_center)?;

    let mut device_dvdi = vec![0f32; device.dvdi.len()];
    let mut device_fast = vec![0f32; device.fast.len()];
    let mut device_slow = vec![0f32; device.slow.len()];
    let mut device_center = vec![0f32; device.center.len()];
    device.dvdi.buf.copy_to(&mut device_dvdi)?;
    device.fast.buf.copy_to(&mut device_fast)?;
    device.slow.buf.copy_to(&mut device_slow)?;
    device.center.buf.copy_to(&mut device_center)?;

    for (lhs, rhs) in legacy_dvdi.iter().zip(device_dvdi.iter()) {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!((lhs - rhs).abs() <= 1e-6, "dvdi lhs={lhs} rhs={rhs}");
    }
    for (lhs, rhs) in legacy_fast.iter().zip(device_fast.iter()) {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!((lhs - rhs).abs() <= 1e-6, "fast lhs={lhs} rhs={rhs}");
    }
    for (lhs, rhs) in legacy_slow.iter().zip(device_slow.iter()) {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!((lhs - rhs).abs() <= 1e-6, "slow lhs={lhs} rhs={rhs}");
    }
    for (lhs, rhs) in legacy_center.iter().zip(device_center.iter()) {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!((lhs - rhs).abs() <= 1e-6, "center lhs={lhs} rhs={rhs}");
    }

    Ok(())
}

#[test]
fn dvdiqqe_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dvdiqqe_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let mut open_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            let c = (x * 0.0021).sin() + 0.0002 * x;
            close_tm[t * cols + s] = c;
            open_tm[t * cols + s] = c - 0.14 + (x * 0.31).cos();
        }
    }

    let (period, smoothing, fast, slow) = (13usize, 6usize, 2.618f64, 4.236f64);
    let mut dvdi_tm = vec![f64::NAN; cols * rows];
    let mut fast_tm = vec![f64::NAN; cols * rows];
    let mut slow_tm = vec![f64::NAN; cols * rows];
    let mut cent_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut o = vec![f64::NAN; rows];
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            o[t] = open_tm[idx];
            h[t] = close_tm[idx] + 0.2;
            l[t] = close_tm[idx] - 0.2;
            c[t] = close_tm[idx];
        }
        let params = DvdiqqeParams {
            period: Some(period),
            smoothing_period: Some(smoothing),
            fast_multiplier: Some(fast),
            slow_multiplier: Some(slow),
            volume_type: Some("default".into()),
            center_type: Some("dynamic".into()),
            tick_size: Some(0.01),
        };
        let input = DvdiqqeInput::from_slices(&o, &h, &l, &c, None, params);
        let out = dvdiqqe_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let idx = t * cols + s;
            dvdi_tm[idx] = out.dvdi[t];
            fast_tm[idx] = out.fast_tl[t];
            slow_tm[idx] = out.slow_tl[t];
            cent_tm[idx] = out.center_line[t];
        }
    }

    let o_f32: Vec<f32> = open_tm.iter().map(|&v| v as f32).collect();
    let c_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaDvdiqqe::new(0).unwrap();
    let gpu = cuda
        .dvdiqqe_many_series_one_param_time_major_dev(
            &o_f32, &c_f32, None, cols, rows, period, smoothing, 2.618, 4.236, "default",
            "dynamic", 0.01,
        )
        .unwrap();
    assert_eq!(gpu.dvdi.rows, rows);
    assert_eq!(gpu.dvdi.cols, cols);
    let plane = rows * cols;
    let mut gd = vec![0f32; plane];
    let mut gf = vec![0f32; plane];
    let mut gs = vec![0f32; plane];
    let mut gc = vec![0f32; plane];
    gpu.dvdi.buf.copy_to(&mut gd)?;
    gpu.fast.buf.copy_to(&mut gf)?;
    gpu.slow.buf.copy_to(&mut gs)?;
    gpu.center.buf.copy_to(&mut gc)?;
    let tol = 2.5e-2;

    let first = c_f32.iter().position(|x| x.is_finite()).unwrap_or(0);
    let warm = first + (2 * period - 1);
    for s in 0..cols {
        for t in warm..rows {
            let idx = t * cols + s;
            assert!(approx(dvdi_tm[idx], gd[idx] as f64, tol));
            assert!(approx(fast_tm[idx], gf[idx] as f64, tol));
            assert!(approx(slow_tm[idx], gs[idx] as f64, tol));
            assert!(approx(cent_tm[idx], gc[idx] as f64, tol));
        }
    }
    Ok(())
}
