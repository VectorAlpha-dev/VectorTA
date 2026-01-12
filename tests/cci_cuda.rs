use vector_ta::indicators::cci::{CciBatchBuilder, CciBatchRange, CciInput, CciParams};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::CudaCci;

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
fn cci_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cci_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut price = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        price[i] = (x * 0.002).sin() + 0.0007 * x;
    }

    let sweep = CciBatchRange { period: (9, 64, 5) };
    let mut periods = Vec::new();
    {
        let (start, end, step) = sweep.period;
        if step == 0 || start == end {
            periods.push(start);
        } else if start < end {
            let mut v = start;
            while v <= end {
                periods.push(v);
                match v.checked_add(step) {
                    Some(next) if next != v => v = next,
                    _ => break,
                }
            }
        } else {
            let mut v = start;
            loop {
                periods.push(v);
                if v <= end {
                    break;
                }
                match v.checked_sub(step) {
                    Some(next) if next != v => v = next,
                    _ => break,
                }
            }
        }
    }

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let price_quant: Vec<f64> = price_f32.iter().map(|&v| v as f64).collect();

    let cpu = CciBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slice(&price_quant)?;

    let cuda = CudaCci::new(0).expect("CudaCci::new");
    let dev = cuda
        .cci_batch_dev(&price_f32, &sweep)
        .expect("cci cuda batch");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 7e-1;
    let mut max_diff = 0.0f64;
    let mut max_idx = 0usize;
    let mut max_cpu = 0.0f64;
    let mut max_gpu = 0.0f64;
    let mut mismatches = 0usize;
    for idx in 0..(cpu.rows * cpu.cols) {
        let a = cpu.values[idx];
        let b = host[idx] as f64;
        if !approx_eq(a, b, tol) {
            mismatches += 1;
        }
        let diff = if a.is_nan() || b.is_nan() {
            if a.is_nan() && b.is_nan() {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (a - b).abs()
        };
        if diff > max_diff {
            max_diff = diff;
            max_idx = idx;
            max_cpu = a;
            max_gpu = b;
        }
    }
    if mismatches != 0 {
        let row = max_idx / cpu.cols;
        let col = max_idx % cpu.cols;
        let p = periods.get(row).copied().unwrap_or(usize::MAX);
        panic!(
            "cci cuda batch mismatch: mismatches={} max_diff={} at idx={} row={} col={} period={} cpu={} gpu={} tol={}",
            mismatches,
            max_diff,
            max_idx,
            row,
            col,
            p,
            max_cpu,
            max_gpu,
            tol
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cci_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cci_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for r in s..rows {
            let x = r as f64 + 0.13 * s as f64;
            tm[r * cols + s] = (x * 0.0023).sin() + 0.0002 * x;
        }
    }
    let period = 14usize;

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let tm_quant: Vec<f64> = tm_f32.iter().map(|&v| v as f64).collect();
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut col = vec![f64::NAN; rows];
        for r in 0..rows {
            col[r] = tm_quant[r * cols + s];
        }
        let input = CciInput::from_slice(
            &col,
            CciParams {
                period: Some(period),
            },
        );
        let out = vector_ta::indicators::cci::cci_with_kernel(&input, Kernel::Scalar)?.values;
        for r in 0..rows {
            cpu_tm[r * cols + s] = out[r];
        }
    }

    let cuda = CudaCci::new(0).expect("CudaCci::new");
    let dev = cuda
        .cci_many_series_one_param_time_major_dev(&tm_f32, cols, rows, period)
        .expect("cci many-series");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 7e-1;
    let mut max_diff = 0.0f64;
    let mut max_idx = 0usize;
    let mut max_cpu = 0.0f64;
    let mut max_gpu = 0.0f64;
    let mut mismatches = 0usize;
    for idx in 0..host.len() {
        let a = cpu_tm[idx];
        let b = host[idx] as f64;
        if !approx_eq(a, b, tol) {
            mismatches += 1;
        }
        let diff = if a.is_nan() || b.is_nan() {
            if a.is_nan() && b.is_nan() {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (a - b).abs()
        };
        if diff > max_diff {
            max_diff = diff;
            max_idx = idx;
            max_cpu = a;
            max_gpu = b;
        }
    }
    if mismatches != 0 {
        let row = max_idx / cols;
        let col = max_idx % cols;
        panic!(
            "cci cuda many-series mismatch: mismatches={} max_diff={} at idx={} row={} col={} period={} cpu={} gpu={} tol={}",
            mismatches,
            max_diff,
            max_idx,
            row,
            col,
            period,
            max_cpu,
            max_gpu,
            tol
        );
    }

    Ok(())
}
