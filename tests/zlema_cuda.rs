#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaZlema;
use my_project::indicators::moving_averages::zlema::{
    zlema_batch_inner_into, zlema_with_kernel, ZlemaBatchRange, ZlemaInput, ZlemaParams,
};
use my_project::utilities::enums::Kernel;

fn make_test_series(len: usize) -> Vec<f64> {
    let mut data = vec![f64::NAN; len];
    for i in 7..len {
        let x = i as f64;
        let base = (x * 0.004321).sin() + (x * 0.002137).cos();
        let trend = 0.00082 * x;
        data[i] = base * 0.6 + trend;
    }
    data
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
fn compare_rows(cpu: &[f64], gpu: &[f64], combos: &[ZlemaParams], len: usize, first_valid: usize) {
    for (row_idx, combo) in combos.iter().enumerate() {
        let period = combo.period.unwrap();
        let warm = first_valid + period - 1;
        for col in 0..len {
            let idx = row_idx * len + col;
            let expected = cpu[idx];
            let actual = gpu[idx];
            if col < warm {
                assert!(
                    expected.is_nan(),
                    "CPU warmup should be NaN at row {}, col {}",
                    row_idx,
                    col
                );
                assert!(
                    actual.is_nan(),
                    "CUDA warmup mismatch at row {}, col {}",
                    row_idx,
                    col
                );
            } else {
                let diff = (expected - actual).abs();
                let tol = 3e-4 + expected.abs() * 1e-4;
                assert!(
                    diff <= tol,
                    "row {}, col {} expected {} got {} diff {}",
                    row_idx,
                    col,
                    expected,
                    actual,
                    diff
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn zlema_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[zlema_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2450;
    let data = make_test_series(len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap();
    let sweep = ZlemaBatchRange { period: (5, 29, 6) };

    let combo_count = expand_grid_count(&sweep);
    let mut cpu_out = vec![f64::NAN; combo_count * len];
    let combos_cpu = zlema_batch_inner_into(&data, &sweep, Kernel::Scalar, false, &mut cpu_out)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaZlema::new(0).expect("CudaZlema::new");
    let (dev, combos_gpu) = cuda
        .zlema_batch_dev(&data_f32, &sweep)
        .expect("zlema_cuda_batch_dev");

    assert_eq!(combos_cpu.len(), combos_gpu.len());
    for (cpu, gpu) in combos_cpu.iter().zip(&combos_gpu) {
        assert_eq!(cpu.period, gpu.period);
    }

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .expect("copy cuda zlema results");
    let gpu_flat_f64: Vec<f64> = gpu_flat.iter().map(|&v| v as f64).collect();

    compare_rows(&cpu_out, &gpu_flat_f64, &combos_cpu, len, first_valid);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn zlema_cuda_host_copy_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[zlema_cuda_host_copy_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096;
    let data = make_test_series(len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap();
    let sweep = ZlemaBatchRange { period: (7, 31, 6) };

    let mut cpu_out = vec![f64::NAN; expand_grid_count(&sweep) * len];
    let combos_cpu = zlema_batch_inner_into(&data, &sweep, Kernel::Scalar, false, &mut cpu_out)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaZlema::new(0).expect("CudaZlema::new");

    let mut gpu_flat = vec![0f32; cpu_out.len()];
    let (rows, cols, combos_gpu) = cuda
        .zlema_batch_into_host_f32(&data_f32, &sweep, &mut gpu_flat)
        .expect("zlema_cuda_batch_into_host_f32");

    assert_eq!(rows, combos_cpu.len());
    assert_eq!(cols, len);
    assert_eq!(combos_cpu.len(), combos_gpu.len());
    for (cpu, gpu) in combos_cpu.iter().zip(&combos_gpu) {
        assert_eq!(cpu.period, gpu.period);
    }

    let gpu_flat_f64: Vec<f64> = gpu_flat.iter().map(|&v| v as f64).collect();
    compare_rows(&cpu_out, &gpu_flat_f64, &combos_cpu, len, first_valid);

    Ok(())
}

#[cfg(feature = "cuda")]
fn expand_grid_count(range: &ZlemaBatchRange) -> usize {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        1
    } else {
        ((end - start) / step) + 1
    }
}

#[cfg(feature = "cuda")]
#[test]
fn zlema_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[zlema_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    // Build time-major input (rows = time, cols = series)
    let cols = 8usize;
    let rows = 1024usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.17;
            tm[t * cols + s] = (x * 0.00321).sin() + 0.00051 * x;
        }
    }

    let period = 13usize;

    // CPU baseline per-series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = tm[t * cols + s];
        }
        let params = ZlemaParams {
            period: Some(period),
        };
        let input = ZlemaInput::from_slice(&series, params);
        let out = zlema_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaZlema::new(0).expect("CudaZlema::new");
    let params = ZlemaParams {
        period: Some(period),
    };
    let dev = cuda
        .zlema_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("zlema_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_tm)?;

    // Compare with tolerance, honoring warmup per series
    let tol = 1e-4;
    for s in 0..cols {
        // compute first_valid for series
        let first_valid = (0..rows)
            .find(|&t| !tm[t * cols + s].is_nan())
            .unwrap_or(rows);
        let warm = first_valid + period - 1;
        for t in 0..rows {
            let idx = t * cols + s;
            let c = cpu_tm[idx];
            let g = gpu_tm[idx] as f64;
            if t < warm {
                assert!(c.is_nan());
                assert!(g.is_nan());
            } else {
                assert!(
                    (c - g).abs() <= tol + c.abs() * 1e-4,
                    "mismatch at ({}, {}): cpu={} gpu={}",
                    t,
                    s,
                    c,
                    g
                );
            }
        }
    }

    Ok(())
}
