#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaDema;
use my_project::indicators::moving_averages::dema::{
    dema_batch_slice, DemaBatchRange, DemaParams,
};
use my_project::utilities::enums::Kernel;

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
                assert!(expected.is_nan(), "CPU warmup should be NaN at row {}, col {}", row_idx, col);
                assert!(actual.is_nan(), "CUDA warmup mismatch at row {}, col {}", row_idx, col);
            } else {
                let diff = (expected - actual).abs();
                // Tighter tolerance: benefits from FMA delta form in kernel
                let tol = 5e-6 + expected.abs() * 5e-6; // ~5 ppm relative + small absolute floor
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
    cuda
        .dema_batch_into_host_f32(&data_f32, &sweep, &mut gpu_flat)
        .expect("dema_cuda_batch_into_host_f32");
    // Sanity: output length should match rows*cols
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
