// CUDA integration tests for EMD (Empirical Mode Decomposition)

use my_project::indicators::emd::{
    emd_batch_with_kernel, emd_with_kernel, EmdBatchRange, EmdInput, EmdParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaEmd;

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

    let tol = 2e-3; // f32 kernels vs f64 scalar
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
fn emd_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[emd_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 4096usize;
    let mut mid_tm = vec![f64::NAN; cols * rows];
    // Build close-like mid price for testing: not needed by scalar (we use high/low pairs),
    // but for GPU many-series we supply a single series of midpoint prices in one plane.
    for s in 0..cols {
        for t in 2..rows {
            let x = (t as f64) + (s as f64) * 0.5;
            // reconstruct high/low later centered around this mid
            mid_tm[t * cols + s] = (x * 0.002).sin() + 0.0002 * x;
        }
    }
    // Derive high/low per series for the scalar CPU baseline with a fixed spread
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
    // Compute first_valids per series for GPU wrapper
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
