use vector_ta::indicators::aso::{aso_batch_with_kernel, AsoBatchRange, AsoParams};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::CudaAso;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaRuntime;

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
fn aso_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[aso_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1024usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        let base = (x * 0.0021).sin() + 0.0002 * x;
        open[i] = base - 0.05;
        high[i] = base + 0.12;
        low[i] = base - 0.11;
        close[i] = base + 0.02;
    }

    let sweep = AsoBatchRange {
        period: (5, 17, 4),
        mode: (0, 2, 1),
    };

    let cpu = aso_batch_with_kernel(&open, &high, &low, &close, &sweep, Kernel::ScalarBatch)?;

    let of: Vec<f32> = open.iter().map(|&v| v as f32).collect();
    let hf: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaAso::new(0).expect("CudaAso::new");
    let (dev_bulls, dev_bears) = cuda
        .aso_batch_dev(&of, &hf, &lf, &cf, &sweep)
        .expect("aso_batch_dev");

    assert_eq!(dev_bulls.rows, cpu.rows);
    assert_eq!(dev_bulls.cols, cpu.cols);
    assert_eq!(dev_bears.rows, cpu.rows);
    assert_eq!(dev_bears.cols, cpu.cols);

    let mut gb = vec![0f32; dev_bulls.len()];
    let mut ge = vec![0f32; dev_bears.len()];
    dev_bulls.buf.copy_to(&mut gb)?;
    dev_bears.buf.copy_to(&mut ge)?;

    let tol = 7e-3;
    for i in 0..gb.len() {
        assert!(
            approx_eq(cpu.bulls[i], gb[i] as f64, tol),
            "bulls mismatch at {}",
            i
        );
        assert!(
            approx_eq(cpu.bears[i], ge[i] as f64, tol),
            "bears mismatch at {}",
            i
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn aso_cuda_device_inputs_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[aso_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 6..len {
        let x = i as f64 * 0.0023;
        let base = (x * 0.91).sin() + 0.00017 * i as f64;
        open[i] = base - 0.04 + (x * 0.31).cos() * 0.01;
        high[i] = base + 0.11 + (x * 0.27).sin().abs() * 0.02;
        low[i] = base - 0.10 - (x * 0.19).cos().abs() * 0.015;
        close[i] = base + 0.015 + (x * 0.43).sin() * 0.01;
    }
    let sweep = AsoBatchRange {
        period: (5, 17, 4),
        mode: (0, 2, 1),
    };

    let open_f32: Vec<f32> = open.iter().map(|&v| v as f32).collect();
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let first_valid = close_f32
        .iter()
        .position(|value| value.is_finite())
        .expect("first valid");

    let runtime = CudaRuntime::new(0).expect("CudaRuntime::new");
    let d_open = runtime.upload_f32(&open_f32).expect("upload open");
    let d_high = runtime.upload_f32(&high_f32).expect("upload high");
    let d_low = runtime.upload_f32(&low_f32).expect("upload low");
    let d_close = runtime.upload_f32(&close_f32).expect("upload close");
    let cuda = CudaAso::new(0).expect("CudaAso::new");

    let legacy = cuda
        .aso_batch_dev(&open_f32, &high_f32, &low_f32, &close_f32, &sweep)
        .expect("legacy aso");
    let device = cuda
        .aso_batch_dev_from_device_inputs(
            d_open.buffer(),
            d_high.buffer(),
            d_low.buffer(),
            d_close.buffer(),
            len,
            first_valid,
            &sweep,
        )
        .expect("device aso");
    cuda.synchronize().expect("aso sync");

    assert_eq!(legacy.0.rows, device.0.rows);
    assert_eq!(legacy.0.cols, device.0.cols);
    assert_eq!(legacy.1.rows, device.1.rows);
    assert_eq!(legacy.1.cols, device.1.cols);

    let mut legacy_bulls = vec![0f32; legacy.0.len()];
    let mut legacy_bears = vec![0f32; legacy.1.len()];
    legacy.0.buf.copy_to(&mut legacy_bulls)?;
    legacy.1.buf.copy_to(&mut legacy_bears)?;

    let mut device_bulls = vec![0f32; device.0.len()];
    let mut device_bears = vec![0f32; device.1.len()];
    device.0.buf.copy_to(&mut device_bulls)?;
    device.1.buf.copy_to(&mut device_bears)?;

    for (lhs, rhs) in legacy_bulls.iter().zip(device_bulls.iter()) {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!((lhs - rhs).abs() <= 1e-6, "bulls lhs={lhs} rhs={rhs}");
    }
    for (lhs, rhs) in legacy_bears.iter().zip(device_bears.iter()) {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!((lhs - rhs).abs() <= 1e-6, "bears lhs={lhs} rhs={rhs}");
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn aso_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[aso_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 1024usize;
    let fast_p = 10usize;
    let mode = 0usize;
    let mut o_tm = vec![f64::NAN; cols * rows];
    let mut h_tm = vec![f64::NAN; cols * rows];
    let mut l_tm = vec![f64::NAN; cols * rows];
    let mut c_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let idx = t * cols + s;
            let x = (t as f64) + (s as f64) * 0.3;
            let base = (x * 0.0025).sin() + 0.00015 * x;
            o_tm[idx] = base - 0.03;
            h_tm[idx] = base + 0.09;
            l_tm[idx] = base - 0.08;
            c_tm[idx] = base + 0.015;
        }
    }

    let mut cpu_b = vec![f64::NAN; cols * rows];
    let mut cpu_e = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut o = vec![0.0; rows];
        let mut h = vec![0.0; rows];
        let mut l = vec![0.0; rows];
        let mut c = vec![0.0; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            o[t] = o_tm[idx];
            h[t] = h_tm[idx];
            l[t] = l_tm[idx];
            c[t] = c_tm[idx];
        }
        let params = AsoParams {
            period: Some(fast_p),
            mode: Some(mode),
        };
        let input = vector_ta::indicators::aso::AsoInput::from_slices(&o, &h, &l, &c, params);
        let out = vector_ta::indicators::aso::aso_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let idx = t * cols + s;
            cpu_b[idx] = out.bulls[t];
            cpu_e[idx] = out.bears[t];
        }
    }
    let o32: Vec<f32> = o_tm.iter().map(|&v| v as f32).collect();
    let h32: Vec<f32> = h_tm.iter().map(|&v| v as f32).collect();
    let l32: Vec<f32> = l_tm.iter().map(|&v| v as f32).collect();
    let c32: Vec<f32> = c_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaAso::new(0).expect("CudaAso::new");
    let (db, de) = cuda
        .aso_many_series_one_param_time_major_dev(&o32, &h32, &l32, &c32, cols, rows, fast_p, mode)
        .expect("aso_many_series");
    assert_eq!(db.rows, rows);
    assert_eq!(db.cols, cols);
    let mut gb = vec![0f32; db.len()];
    db.buf.copy_to(&mut gb)?;
    let mut ge = vec![0f32; de.len()];
    de.buf.copy_to(&mut ge)?;
    let tol = 7e-3;
    for i in 0..gb.len() {
        assert!(
            approx_eq(cpu_b[i], gb[i] as f64, tol),
            "bulls mismatch at {}",
            i
        );
        assert!(
            approx_eq(cpu_e[i], ge[i] as f64, tol),
            "bears mismatch at {}",
            i
        );
    }
    Ok(())
}
