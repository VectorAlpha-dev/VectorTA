use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::CudaChop;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaRuntime};
use vector_ta::indicators::chop::{ChopBatchBuilder, ChopBatchRange};

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
fn chop_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[chop_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let n = 8192usize;
    let mut h = vec![f64::NAN; n];
    let mut l = vec![f64::NAN; n];
    let mut c = vec![f64::NAN; n];
    for i in 1..n {
        let x = i as f64 * 0.00123;
        let base = x.sin() + 0.0002 * (i as f64);
        let hi = base + 0.6 + 0.05 * (x * 2.1).cos();
        let lo = base - 0.6 - 0.04 * (x * 1.7).sin();
        h[i] = hi;
        l[i] = lo;
        c[i] = (hi + lo) * 0.5;
    }
    let sweep = ChopBatchRange {
        period: (5, 25, 5),
        scalar: (100.0, 100.0, 0.0),
        drift: (1, 3, 1),
    };

    let cpu = ChopBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(5, 25, 5)
        .scalar_static(100.0)
        .drift_range(1, 3, 1)
        .apply_slices(&h, &l, &c)?;

    let hf: Vec<f32> = h.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = l.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = c.iter().map(|&v| v as f32).collect();
    let cuda = CudaChop::new(0).expect("CudaChop::new");
    let (dev, combos) = cuda
        .chop_batch_dev(&hf, &lf, &cf, &sweep)
        .expect("cuda chop");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    assert_eq!(cpu.combos.len(), combos.len());

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-3;
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu.values[idx], host[idx] as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu.values[idx],
            host[idx]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn chop_cuda_device_inputs_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[chop_cuda_device_inputs_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let n = 4096usize;
    let mut h = vec![f64::NAN; n];
    let mut l = vec![f64::NAN; n];
    let mut c = vec![f64::NAN; n];
    for i in 4..n {
        let x = i as f64 * 0.0017;
        let base = x.sin() * 0.9 + 0.0003 * i as f64;
        let hi = base + 0.55 + 0.04 * (x * 1.9).cos();
        let lo = base - 0.55 - 0.03 * (x * 1.3).sin();
        h[i] = hi;
        l[i] = lo;
        c[i] = (hi + lo) * 0.5;
    }

    let hf: Vec<f32> = h.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = l.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = c.iter().map(|&v| v as f32).collect();
    let first_valid = hf
        .iter()
        .zip(lf.iter())
        .zip(cf.iter())
        .position(|((h, l), c)| h.is_finite() && l.is_finite() && c.is_finite())
        .expect("first valid");
    let sweep = ChopBatchRange {
        period: (5, 25, 5),
        scalar: (100.0, 100.0, 0.0),
        drift: (1, 3, 1),
    };

    let cuda = CudaChop::new(0).expect("CudaChop::new");
    let (legacy, legacy_combos) = cuda
        .chop_batch_dev(&hf, &lf, &cf, &sweep)
        .expect("legacy chop");

    let runtime = CudaRuntime::new(0).expect("runtime");
    let d_high = runtime.upload_f32(&hf).expect("upload high");
    let d_low = runtime.upload_f32(&lf).expect("upload low");
    let d_close = runtime.upload_f32(&cf).expect("upload close");
    let (device, device_combos) = cuda
        .chop_batch_dev_from_device_inputs(
            d_high.buffer(),
            d_low.buffer(),
            d_close.buffer(),
            n,
            first_valid,
            &sweep,
        )
        .expect("device chop");
    cuda.synchronize().expect("chop sync");

    assert_eq!(legacy.rows, device.rows);
    assert_eq!(legacy.cols, device.cols);
    assert_eq!(legacy_combos.len(), device_combos.len());

    let mut legacy_host = vec![0f32; legacy.len()];
    legacy.buf.copy_to(&mut legacy_host)?;
    let mut device_host = vec![0f32; device.len()];
    device.buf.copy_to(&mut device_host)?;

    for (lhs, rhs) in legacy_host.iter().zip(device_host.iter()) {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!((lhs - rhs).abs() <= 1e-6, "lhs={lhs} rhs={rhs}");
    }

    Ok(())
}
