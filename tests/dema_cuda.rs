#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaDema;
use vector_ta::indicators::moving_averages::dema::{dema_batch_slice, DemaBatchRange, DemaParams};
use vector_ta::utilities::enums::Kernel;

fn make_test_series(len: usize) -> Vec<f64> {
    let mut data = vec![f64::NAN; len];
    for i in 11..len {
        let x = i as f64;
        let base = (x * 0.00371).sin() + (x * 0.00153).cos();
        let trend = 0.00047 * x;
        data[i] = base * 0.7 + trend;
    }
    data
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
fn compare_rows(cpu: &[f64], gpu: &[f64], combos: &[DemaParams], len: usize, first_valid: usize) {
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

                let tol = 5e-6 + expected.abs() * 5e-6;
                assert!(
                    diff <= tol,
                    "row {}, col {} expected {} got {} diff {} (tol {})",
                    row_idx,
                    col,
                    expected,
                    actual,
                    diff,
                    tol
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn dema_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dema_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 3072;
    let data = make_test_series(len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap();
    let sweep = DemaBatchRange { period: (5, 33, 7) };

    let cpu = dema_batch_slice(&data, &sweep, Kernel::Scalar)?;
    let combos_cpu = cpu.combos.clone();
    let cpu_out = cpu.values;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaDema::new(0).expect("CudaDema::new");
    let dev = cuda
        .dema_batch_dev(&data_f32, &sweep)
        .expect("dema_cuda_batch_dev");

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .expect("copy cuda dema results");
    let gpu_flat_f64: Vec<f64> = gpu_flat.iter().map(|&v| v as f64).collect();

    compare_rows(&cpu_out, &gpu_flat_f64, &combos_cpu, len, first_valid);

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn dema_cuda_host_copy_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dema_cuda_host_copy_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096;
    let data = make_test_series(len);
    let first_valid = data.iter().position(|v| !v.is_nan()).unwrap();
    let sweep = DemaBatchRange { period: (7, 31, 6) };

    let cpu = dema_batch_slice(&data, &sweep, Kernel::Scalar)?;
    let combos_cpu = cpu.combos.clone();
    let cpu_out = cpu.values;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaDema::new(0).expect("CudaDema::new");

    let mut gpu_flat = vec![0f32; cpu_out.len()];
    cuda.dema_batch_into_host_f32(&data_f32, &sweep, &mut gpu_flat)
        .expect("dema_cuda_batch_into_host_f32");

    assert_eq!(gpu_flat.len(), combos_cpu.len() * len);

    let gpu_flat_f64: Vec<f64> = gpu_flat.iter().map(|&v| v as f64).collect();
    compare_rows(&cpu_out, &gpu_flat_f64, &combos_cpu, len, first_valid);

    Ok(())
}

#[cfg(feature = "cuda")]
fn expand_grid_count(range: &DemaBatchRange) -> usize {
    let (start, end, step) = range.period;
    if step == 0 || start == end {
        1
    } else {
        ((end - start) / step) + 1
    }
}

#[cfg(feature = "cuda")]
#[test]
fn dema_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dema_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 1536usize;
    let mut price_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (4 + s)..rows {
            let x = (t as f64) + (s as f64) * 0.25;
            price_tm[t * cols + s] = (x * 0.00237).sin() + (x * 0.00071).cos() * 0.5 + 0.00021 * x;
        }
    }

    let period = 24usize;

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = price_tm[t * cols + s];
        }
        let params = DemaParams {
            period: Some(period),
        };
        let input = vector_ta::indicators::moving_averages::dema::DemaInput {
            data: vector_ta::indicators::moving_averages::dema::DemaData::Slice(&series),
            params,
        };
        let out =
            vector_ta::indicators::moving_averages::dema::dema_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let price_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaDema::new(0).expect("CudaDema::new");
    let dev = cuda
        .dema_many_series_one_param_time_major_dev(
            &price_tm_f32,
            cols,
            rows,
            &DemaParams {
                period: Some(period),
            },
        )
        .expect("dema_many_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_tm)?;

    let tol = 5e-6;
    for idx in 0..gpu_tm.len() {
        let c = cpu_tm[idx];
        let g = gpu_tm[idx] as f64;
        if c.is_nan() {
            assert!(g.is_nan(), "warmup NaN mismatch at {}", idx);
        } else {
            let diff = (c - g).abs();
            let rtol = 5e-6 * c.abs();
            assert!(
                diff <= tol + rtol,
                "mismatch at {}: cpu={} gpu={} diff={} tol={} rtol={}",
                idx,
                c,
                g,
                diff,
                tol,
                rtol
            );
        }
    }

    Ok(())
}
