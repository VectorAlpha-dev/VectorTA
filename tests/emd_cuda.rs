use vector_ta::indicators::emd::{
    emd_batch_with_kernel, emd_with_kernel, EmdBatchRange, EmdInput, EmdParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{CudaEmd, CudaMedprice, CudaRuntime};

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
fn emd_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[emd_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 8192usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 2..len {
        let x = i as f64;
        high[i] = (x * 0.00123).sin() + 0.00017 * x + 0.4;
        low[i] = (x * 0.00123).sin() + 0.00017 * x - 0.4;
    }
    let sweep = EmdBatchRange {
        period: (10, 22, 4),
        delta: (0.3, 0.7, 0.2),
        fraction: (0.05, 0.15, 0.05),
    };

    let cpu = emd_batch_with_kernel(&high, &low, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cuda = CudaEmd::new(0).expect("CudaEmd::new");
    let res = cuda
        .emd_batch_dev(&high_f32, &low_f32, &sweep)
        .expect("cuda emd batch");
    let outputs = res.outputs;

    assert_eq!(cpu.rows, outputs.rows());
    assert_eq!(cpu.cols, outputs.cols());

    let mut g_upper = vec![0f32; outputs.upper.len()];
    let mut g_middle = vec![0f32; outputs.middle.len()];
    let mut g_lower = vec![0f32; outputs.lower.len()];
    outputs.upper.buf.copy_to(&mut g_upper)?;
    outputs.middle.buf.copy_to(&mut g_middle)?;
    outputs.lower.buf.copy_to(&mut g_lower)?;

    let tol = 2e-3;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.upperband[idx], g_upper[idx] as f64, tol),
            "upper mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.middleband[idx], g_middle[idx] as f64, tol),
            "middle mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.lowerband[idx], g_lower[idx] as f64, tol),
            "lower mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn emd_cuda_device_prices_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[emd_cuda_device_prices_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64 * 0.0019;
        let mid = 100.0 + x.sin() * 0.6 + 0.0003 * i as f64;
        let width = 0.35 + (x * 0.37).cos().abs() * 0.05;
        high[i] = mid + width;
        low[i] = mid - width;
    }
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let first_valid = high_f32
        .iter()
        .zip(low_f32.iter())
        .position(|(high, low)| high.is_finite() && low.is_finite())
        .expect("first valid");
    let sweep = EmdBatchRange {
        period: (10, 22, 4),
        delta: (0.3, 0.7, 0.2),
        fraction: (0.05, 0.15, 0.05),
    };

    let runtime = CudaRuntime::new(0).expect("CudaRuntime::new");
    let d_high = runtime.upload_f32(&high_f32).expect("upload high");
    let d_low = runtime.upload_f32(&low_f32).expect("upload low");
    let medprice = CudaMedprice::new(0).expect("CudaMedprice::new");
    let mut d_prices =
        unsafe { cust::memory::DeviceBuffer::<f32>::uninitialized(len) }.expect("prices alloc");
    medprice
        .medprice_device(
            d_high.buffer(),
            d_low.buffer(),
            len,
            first_valid,
            &mut d_prices,
        )
        .expect("medprice_device");
    let cuda = CudaEmd::new(0).expect("CudaEmd::new");

    let legacy = cuda
        .emd_batch_dev(&high_f32, &low_f32, &sweep)
        .expect("legacy emd");
    let device = cuda
        .emd_batch_dev_from_device_prices(&d_prices, len, first_valid, &sweep)
        .expect("device emd");
    cuda.synchronize().expect("emd sync");

    let mut legacy_upper = vec![0f32; legacy.outputs.upper.len()];
    let mut legacy_middle = vec![0f32; legacy.outputs.middle.len()];
    let mut legacy_lower = vec![0f32; legacy.outputs.lower.len()];
    let mut device_upper = vec![0f32; device.outputs.upper.len()];
    let mut device_middle = vec![0f32; device.outputs.middle.len()];
    let mut device_lower = vec![0f32; device.outputs.lower.len()];
    legacy.outputs.upper.buf.copy_to(&mut legacy_upper)?;
    legacy.outputs.middle.buf.copy_to(&mut legacy_middle)?;
    legacy.outputs.lower.buf.copy_to(&mut legacy_lower)?;
    device.outputs.upper.buf.copy_to(&mut device_upper)?;
    device.outputs.middle.buf.copy_to(&mut device_middle)?;
    device.outputs.lower.buf.copy_to(&mut device_lower)?;

    for (lhs, rhs) in legacy_upper.iter().zip(device_upper.iter()) {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!((lhs - rhs).abs() <= 1e-6, "upper lhs={lhs} rhs={rhs}");
    }
    for (lhs, rhs) in legacy_middle.iter().zip(device_middle.iter()) {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!((lhs - rhs).abs() <= 1e-6, "middle lhs={lhs} rhs={rhs}");
    }
    for (lhs, rhs) in legacy_lower.iter().zip(device_lower.iter()) {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!((lhs - rhs).abs() <= 1e-6, "lower lhs={lhs} rhs={rhs}");
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn emd_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[emd_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 4096usize;
    let mut mid_tm = vec![f64::NAN; cols * rows];

    for s in 0..cols {
        for t in 2..rows {
            let x = (t as f64) + (s as f64) * 0.5;

            mid_tm[t * cols + s] = (x * 0.002).sin() + 0.0002 * x;
        }
    }

    let period = 18usize;
    let delta = 0.5f64;
    let fraction = 0.1f64;

    let mut cpu_upper_tm = vec![f64::NAN; cols * rows];
    let mut cpu_middle_tm = vec![f64::NAN; cols * rows];
    let mut cpu_lower_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut high = vec![f64::NAN; rows];
        let mut low = vec![f64::NAN; rows];
        for t in 0..rows {
            let m = mid_tm[t * cols + s];
            high[t] = if m.is_nan() { f64::NAN } else { m + 0.5 };
            low[t] = if m.is_nan() { f64::NAN } else { m - 0.5 };
        }
        let params = EmdParams {
            period: Some(period),
            delta: Some(delta),
            fraction: Some(fraction),
        };
        let input = EmdInput::from_slices(&high, &low, &[], &[], params);
        let out = emd_with_kernel(&input, Kernel::Scalar).unwrap();
        for t in 0..rows {
            cpu_upper_tm[t * cols + s] = out.upperband[t];
            cpu_middle_tm[t * cols + s] = out.middleband[t];
            cpu_lower_tm[t * cols + s] = out.lowerband[t];
        }
    }

    let mid_tm_f32: Vec<f32> = mid_tm.iter().map(|&v| v as f32).collect();

    let mut first_valids = vec![0i32; cols];
    for s in 0..cols {
        let mut fv = 0i32;
        for t in 0..rows {
            if mid_tm_f32[t * cols + s].is_finite() {
                fv = t as i32;
                break;
            }
        }
        first_valids[s] = fv;
    }

    let cuda = CudaEmd::new(0).expect("CudaEmd::new");
    let trio = cuda
        .emd_many_series_one_param_time_major_dev(
            &mid_tm_f32,
            cols,
            rows,
            &EmdParams {
                period: Some(period),
                delta: Some(delta),
                fraction: Some(fraction),
            },
            &first_valids,
        )
        .expect("emd many series");
    assert_eq!(trio.rows(), rows);
    assert_eq!(trio.cols(), cols);

    let mut g_upper = vec![0f32; trio.upper.len()];
    let mut g_middle = vec![0f32; trio.middle.len()];
    let mut g_lower = vec![0f32; trio.lower.len()];
    trio.upper.buf.copy_to(&mut g_upper)?;
    trio.middle.buf.copy_to(&mut g_middle)?;
    trio.lower.buf.copy_to(&mut g_lower)?;

    let tol = 2e-3;
    for idx in 0..(rows * cols) {
        assert!(
            approx_eq(cpu_upper_tm[idx], g_upper[idx] as f64, tol),
            "upper mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_middle_tm[idx], g_middle[idx] as f64, tol),
            "middle mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_lower_tm[idx], g_lower[idx] as f64, tol),
            "lower mismatch at {}",
            idx
        );
    }
    Ok(())
}
