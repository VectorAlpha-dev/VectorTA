use vector_ta::indicators::prb::{
    prb_batch_with_kernel, prb_with_kernel, PrbBatchRange, PrbData, PrbInput, PrbParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaPrb};

fn approx_close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs());
    diff <= atol + rtol * scale
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
fn prb_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[prb_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        data[i] = (x * 0.00123).sin() + 0.00011 * x;
    }
    data[521] = f64::NAN;
    data[1000] = f64::NAN;

    let sweep = PrbBatchRange {
        smooth_period: (10, 10, 0),
        regression_period: (50, 80, 15),
        polynomial_order: (2, 2, 0),
        regression_offset: (0, 0, 0),
    };
    let cpu = prb_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch, false)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaPrb::new(0).expect("CudaPrb::new");
    let (dev_main, dev_up, dev_lo) = cuda
        .prb_batch_dev(&data_f32, &sweep, false)
        .expect("prb_batch_dev");

    assert_eq!(cpu.rows, dev_main.rows);
    assert_eq!(cpu.cols, dev_main.cols);
    let mut g_main = vec![0f32; dev_main.len()];
    let mut g_up = vec![0f32; dev_up.len()];
    let mut g_lo = vec![0f32; dev_lo.len()];
    dev_main.buf.copy_to(&mut g_main)?;
    dev_up.buf.copy_to(&mut g_up)?;
    dev_lo.buf.copy_to(&mut g_lo)?;

    let rtol = 2e-2f64;
    let atol = 1e-3f64;
    for idx in 0..(cpu.rows * cpu.cols) {
        if !approx_close(cpu.values[idx], g_main[idx] as f64, rtol, atol) {
            eprintln!(
                "first mismatch idx {} cpu={} gpu={}",
                idx, cpu.values[idx], g_main[idx] as f64
            );
            assert!(false, "main mismatch at {}", idx);
        }
        assert!(
            approx_close(cpu.upper_band[idx], g_up[idx] as f64, rtol, atol),
            "up mismatch at {}",
            idx
        );
        assert!(
            approx_close(cpu.lower_band[idx], g_lo[idx] as f64, rtol, atol),
            "lo mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn prb_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[prb_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 6usize;
    let rows = 1024usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s + 3)..rows {
            let x = (t as f64) + (s as f64) * 0.17;
            tm[t * cols + s] = (x * 0.0021).sin() + 0.0002 * x;
        }
    }

    let params = PrbParams {
        smooth_data: Some(false),
        smooth_period: Some(10),
        regression_period: Some(64),
        polynomial_order: Some(2),
        regression_offset: Some(0),
        ndev: Some(2.0),
        equ_from: Some(0),
    };

    let mut cpu_m = vec![f64::NAN; cols * rows];
    let mut cpu_u = vec![f64::NAN; cols * rows];
    let mut cpu_l = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut col = vec![0.0f64; rows];
        for t in 0..rows {
            col[t] = tm[t * cols + s];
        }
        let input = PrbInput {
            data: PrbData::Slice(&col),
            params: params.clone(),
        };
        let out = prb_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_m[t * cols + s] = out.values[t];
            cpu_u[t * cols + s] = out.upper_band[t];
            cpu_l[t * cols + s] = out.lower_band[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();

    let cuda = CudaPrb::new(0).expect("CudaPrb::new");
    let (dev_m, dev_u, dev_l) = cuda
        .prb_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("prb_many_series_one_param");
    let mut g_m = vec![0f32; dev_m.len()];
    let mut g_u = vec![0f32; dev_u.len()];
    let mut g_l = vec![0f32; dev_l.len()];
    dev_m.buf.copy_to(&mut g_m)?;
    dev_u.buf.copy_to(&mut g_u)?;
    dev_l.buf.copy_to(&mut g_l)?;

    let rtol = 2e-2f64;
    let atol = 1e-3f64;
    for idx in 0..g_m.len() {
        if !approx_close(cpu_m[idx], g_m[idx] as f64, rtol, atol) {
            eprintln!(
                "first mismatch idx {} cpu={} gpu={}",
                idx, cpu_m[idx], g_m[idx] as f64
            );
            assert!(false, "m mismatch at {}", idx);
        }
        assert!(
            approx_close(cpu_u[idx], g_u[idx] as f64, rtol, atol),
            "u mismatch at {}",
            idx
        );
        assert!(
            approx_close(cpu_l[idx], g_l[idx] as f64, rtol, atol),
            "l mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn prb_cuda_device_inputs_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[prb_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut data = vec![f32::NAN; len];
    for i in 7..len {
        let x = i as f32;
        data[i] = (x * 0.0023).sin() + 0.0004 * x;
    }
    data[311] = f32::NAN;
    data[777] = f32::NAN;

    let sweep = PrbBatchRange {
        smooth_period: (10, 14, 4),
        regression_period: (48, 64, 16),
        polynomial_order: (2, 2, 0),
        regression_offset: (0, 0, 0),
    };
    let first_valid = data.iter().position(|v| !v.is_nan()).expect("first_valid");
    let cuda = CudaPrb::new(0).expect("CudaPrb::new");
    let d_prices = cust::memory::DeviceBuffer::from_slice(&data)?;

    let (legacy_main, legacy_up, legacy_lo) = cuda
        .prb_batch_dev(&data, &sweep, true)
        .expect("legacy batch");
    let (device_main, device_up, device_lo) = cuda
        .prb_batch_dev_from_device_prices(&d_prices, data.len(), first_valid, &sweep, true)
        .expect("device batch");
    cuda.synchronize().expect("sync");

    let mut legacy_main_host = vec![0f32; legacy_main.len()];
    let mut legacy_up_host = vec![0f32; legacy_up.len()];
    let mut legacy_lo_host = vec![0f32; legacy_lo.len()];
    let mut device_main_host = vec![0f32; device_main.len()];
    let mut device_up_host = vec![0f32; device_up.len()];
    let mut device_lo_host = vec![0f32; device_lo.len()];
    legacy_main.buf.copy_to(&mut legacy_main_host)?;
    legacy_up.buf.copy_to(&mut legacy_up_host)?;
    legacy_lo.buf.copy_to(&mut legacy_lo_host)?;
    device_main.buf.copy_to(&mut device_main_host)?;
    device_up.buf.copy_to(&mut device_up_host)?;
    device_lo.buf.copy_to(&mut device_lo_host)?;

    for (lhs, rhs) in legacy_main_host.iter().zip(device_main_host.iter()) {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!(
            (*lhs - *rhs).abs() < 1e-4,
            "main mismatch lhs={lhs} rhs={rhs}"
        );
    }
    for (lhs, rhs) in legacy_up_host.iter().zip(device_up_host.iter()) {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!(
            (*lhs - *rhs).abs() < 1e-4,
            "upper mismatch lhs={lhs} rhs={rhs}"
        );
    }
    for (lhs, rhs) in legacy_lo_host.iter().zip(device_lo_host.iter()) {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!(
            (*lhs - *rhs).abs() < 1e-4,
            "lower mismatch lhs={lhs} rhs={rhs}"
        );
    }

    Ok(())
}
