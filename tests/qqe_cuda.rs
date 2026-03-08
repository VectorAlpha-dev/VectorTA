use vector_ta::indicators::qqe::{
    qqe_batch_with_kernel, qqe_with_kernel, QqeBatchRange, QqeData, QqeInput, QqeParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::qqe_wrapper::CudaQqe;
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
fn qqe_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[qqe_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 64..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = QqeBatchRange {
        rsi_period: (8, 32, 4),
        smoothing_factor: (5, 9, 2),
        fast_factor: (4.236, 4.236, 0.0),
    };

    let cpu = qqe_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaQqe::new(0).expect("CudaQqe::new");
    let (dev, combos) = cuda
        .qqe_batch_dev(&price_f32, &sweep)
        .expect("qqe_batch_dev");

    assert_eq!(dev.cols, cpu.cols);
    assert_eq!(dev.rows, 2 * combos.len());
    assert_eq!(cpu.rows, combos.len());

    let mut out = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut out)?;

    let tol = 1.5e-1;
    let cols = cpu.cols;
    for row in 0..cpu.rows {
        let g_row_fast = 2 * row;
        let g_row_slow = 2 * row + 1;
        let mut printed = 0;
        for t in 0..cols {
            let c_fast = cpu.fast_values[row * cols + t];
            let c_slow = cpu.slow_values[row * cols + t];
            let g_fast = out[g_row_fast * cols + t] as f64;
            let g_slow = out[g_row_slow * cols + t] as f64;
            if row == 0 && ((1388..1400).contains(&t) || (6495..6505).contains(&t)) && printed < 24
            {
                eprintln!(
                    "dbg row0 t={} cpu_fast={} gpu_fast={} cpu_slow={} gpu_slow={}",
                    t, c_fast, g_fast, c_slow, g_slow
                );
                printed += 1;
            }
            if !approx_eq(c_fast, g_fast, tol) && printed < 3 {
                eprintln!(
                    "batch diff fast row={}, t={} cpu={} gpu={} abs={} tol={}",
                    row,
                    t,
                    c_fast,
                    g_fast,
                    (c_fast - g_fast).abs(),
                    tol
                );
                printed += 1;
            }
            if !approx_eq(c_slow, g_slow, tol) && printed < 3 {
                eprintln!(
                    "batch diff slow row={}, t={} cpu={} gpu={} abs={} tol={}",
                    row,
                    t,
                    c_slow,
                    g_slow,
                    (c_slow - g_slow).abs(),
                    tol
                );
                printed += 1;
            }
            assert!(
                approx_eq(c_fast, g_fast, tol),
                "fast mismatch at (row={}, t={})",
                row,
                t
            );
            assert!(
                approx_eq(c_slow, g_slow, tol),
                "slow mismatch at (row={}, t={})",
                row,
                t
            );
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn qqe_cuda_device_prices_match_legacy_batch() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[qqe_cuda_device_prices_match_legacy_batch] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![f32::NAN; len];
    for i in 64..len {
        let x = i as f32;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = QqeBatchRange {
        rsi_period: (8, 32, 4),
        smoothing_factor: (5, 9, 2),
        fast_factor: (4.236, 4.236, 0.0),
    };

    let cuda = CudaQqe::new(0).expect("CudaQqe::new");
    let (legacy, legacy_combos) = cuda.qqe_batch_dev(&price, &sweep).expect("legacy batch");

    let runtime = CudaRuntime::new(0).expect("runtime");
    let d_price = runtime.upload_f32(&price).expect("upload prices");
    let device = cuda
        .qqe_batch_dev_from_device_prices(d_price.buffer(), len, 64, &sweep)
        .expect("device prices");
    assert_eq!(legacy.rows, device.rows, "rows");
    assert_eq!(legacy.cols, device.cols, "cols");
    assert!(!legacy_combos.is_empty(), "combos");

    let mut legacy_host = vec![0f32; legacy.len()];
    legacy.buf.copy_to(&mut legacy_host)?;
    let mut device_host = vec![0f32; device.len()];
    device.buf.copy_to(&mut device_host)?;
    for (idx, (&lhs, &rhs)) in legacy_host.iter().zip(device_host.iter()).enumerate() {
        if lhs.is_nan() && rhs.is_nan() {
            continue;
        }
        assert!(
            (lhs - rhs).abs() < 1e-6,
            "qqe mismatch at {idx}: legacy={lhs} device={rhs}"
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn qqe_cuda_device_output_selection_matches_legacy_batch() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[qqe_cuda_device_output_selection_matches_legacy_batch] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 8192usize;
    let first_valid = 64usize;
    let mut price = vec![f32::NAN; len];
    for i in first_valid..len {
        let x = i as f32;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = QqeBatchRange {
        rsi_period: (8, 32, 4),
        smoothing_factor: (5, 9, 2),
        fast_factor: (4.236, 4.236, 0.0),
    };

    let cuda = CudaQqe::new(0).expect("CudaQqe::new");
    let (legacy, legacy_combos) = cuda.qqe_batch_dev(&price, &sweep).expect("legacy batch");

    let runtime = CudaRuntime::new(0).expect("runtime");
    let d_price = runtime.upload_f32(&price).expect("upload prices");
    let (fast_dev, fast_combos) = cuda
        .qqe_batch_output_dev_from_device_prices(d_price.buffer(), len, first_valid, &sweep, 0)
        .expect("fast output");
    let (slow_dev, slow_combos) = cuda
        .qqe_batch_output_dev_from_device_prices(d_price.buffer(), len, first_valid, &sweep, 1)
        .expect("slow output");
    cuda.synchronize().expect("sync");

    assert_eq!(legacy_combos.len(), fast_combos.len());
    assert_eq!(legacy_combos.len(), slow_combos.len());
    for ((lhs, rhs_fast), rhs_slow) in legacy_combos
        .iter()
        .zip(fast_combos.iter())
        .zip(slow_combos.iter())
    {
        assert_eq!(lhs.rsi_period, rhs_fast.rsi_period);
        assert_eq!(lhs.smoothing_factor, rhs_fast.smoothing_factor);
        assert_eq!(lhs.fast_factor, rhs_fast.fast_factor);
        assert_eq!(lhs.rsi_period, rhs_slow.rsi_period);
        assert_eq!(lhs.smoothing_factor, rhs_slow.smoothing_factor);
        assert_eq!(lhs.fast_factor, rhs_slow.fast_factor);
    }

    let mut legacy_host = vec![0f32; legacy.len()];
    legacy.buf.copy_to(&mut legacy_host)?;
    let mut fast_host = vec![0f32; fast_dev.len()];
    fast_dev.buf.copy_to(&mut fast_host)?;
    let mut slow_host = vec![0f32; slow_dev.len()];
    slow_dev.buf.copy_to(&mut slow_host)?;

    let rows = fast_dev.rows;
    let cols = fast_dev.cols;
    assert_eq!(rows, slow_dev.rows);
    assert_eq!(cols, slow_dev.cols);
    for row in 0..rows {
        for col in 0..cols {
            let legacy_fast = legacy_host[(2 * row) * cols + col];
            let legacy_slow = legacy_host[(2 * row + 1) * cols + col];
            let fast = fast_host[row * cols + col];
            let slow = slow_host[row * cols + col];
            if !(legacy_fast.is_nan() && fast.is_nan()) {
                assert!(
                    (legacy_fast - fast).abs() < 1e-6,
                    "fast mismatch at ({row}, {col}): legacy={legacy_fast} device={fast}"
                );
            }
            if !(legacy_slow.is_nan() && slow.is_nan()) {
                assert!(
                    (legacy_slow - slow).abs() < 1e-6,
                    "slow mismatch at ({row}, {col}): legacy={legacy_slow} device={slow}"
                );
            }
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn qqe_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[qqe_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 1024usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.17;
            tm[t * cols + s] = (x * 0.002).sin() + 0.00031 * x;
        }
    }
    let params = QqeParams {
        rsi_period: Some(14),
        smoothing_factor: Some(5),
        fast_factor: Some(4.236),
    };

    let mut cpu_fast_tm = vec![f64::NAN; cols * rows];
    let mut cpu_slow_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];

        for t in 0..rows {
            series[t] = tm[t * cols + s];
        }
        let input = QqeInput {
            data: QqeData::Slice(&series),
            params: params.clone(),
        };
        let out = qqe_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_fast_tm[t * cols + s] = out.fast[t];
            cpu_slow_tm[t * cols + s] = out.slow[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaQqe::new(0).expect("CudaQqe::new");
    let dev = cuda
        .qqe_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("qqe_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, 2 * cols);

    let mut out = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut out)?;

    let tol = 7.5e-3;
    for s in 0..cols {
        let mut printed = 0;
        for t in 0..rows {
            let g_fast = out[t * (2 * cols) + s] as f64;
            let g_slow = out[t * (2 * cols) + (s + cols)] as f64;
            let c_fast = cpu_fast_tm[t * cols + s];
            let c_slow = cpu_slow_tm[t * cols + s];
            if !approx_eq(c_fast, g_fast, tol) && printed < 3 {
                eprintln!(
                    "many diff fast s={}, t={} cpu={} gpu={} abs={} tol={}",
                    s,
                    t,
                    c_fast,
                    g_fast,
                    (c_fast - g_fast).abs(),
                    tol
                );
                printed += 1;
            }
            if !approx_eq(c_slow, g_slow, tol) && printed < 3 {
                eprintln!(
                    "many diff slow s={}, t={} cpu={} gpu={} abs={} tol={}",
                    s,
                    t,
                    c_slow,
                    g_slow,
                    (c_slow - g_slow).abs(),
                    tol
                );
                printed += 1;
            }
            assert!(
                approx_eq(c_fast, g_fast, tol),
                "fast mismatch at (s={}, t={})",
                s,
                t
            );
            assert!(
                approx_eq(c_slow, g_slow, tol),
                "slow mismatch at (s={}, t={})",
                s,
                t
            );
        }
    }

    Ok(())
}
