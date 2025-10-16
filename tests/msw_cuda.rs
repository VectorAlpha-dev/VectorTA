// CUDA integration tests for MSW (Mesa Sine Wave)

use my_project::indicators::msw::{msw_batch_with_kernel, msw_with_kernel, MswBatchRange, MswInput, MswParams};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, CudaMsw};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn msw_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[msw_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 16..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }

    let sweep = MswBatchRange { period: (5, 40, 5) };

    // CPU baseline
    let cpu = msw_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    // GPU
    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaMsw::new(0).expect("CudaMsw::new");
    let (dev, combos) = cuda.msw_batch_dev(&price_f32, &sweep).expect("msw_batch_dev");
    assert_eq!(combos.len(), cpu.rows);
    assert_eq!(dev.rows, 2 * cpu.rows);
    assert_eq!(dev.cols, cpu.cols);

    let mut out = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut out)?;

    // Compare row by row (GPU rows: 2 per period -> sine, lead)
    let tol = 1.5e-3; // f32 vs f64 allowance
    for (r, prm) in combos.iter().enumerate() {
        let base = r * cpu.cols;
        // GPU sine and lead rows
        let g_sine = &out[(2*r)*cpu.cols .. (2*r+1)*cpu.cols];
        let g_lead = &out[(2*r+1)*cpu.cols .. (2*r+2)*cpu.cols];
        let c_sine = &cpu.sine[base .. base + cpu.cols];
        let c_lead = &cpu.lead[base .. base + cpu.cols];
        for i in 0..cpu.cols {
            assert!(
                approx_eq(c_sine[i], g_sine[i] as f64, tol),
                "sine mismatch (period={} idx={}): cpu={} gpu={}",
                prm.period.unwrap(), i, c_sine[i], g_sine[i]
            );
            assert!(
                approx_eq(c_lead[i], g_lead[i] as f64, tol),
                "lead mismatch (period={} idx={}): cpu={} gpu={}",
                prm.period.unwrap(), i, c_lead[i], g_lead[i]
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn msw_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[msw_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize; // number of series
    let rows = 1024usize; // timesteps
    let mut price_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows { // stagger validity
            let x = (t as f64) + (s as f64) * 0.25;
            price_tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let period = 21usize;

    // CPU baseline per series
    let mut cpu_sine_tm = vec![f64::NAN; cols * rows];
    let mut cpu_lead_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows { series[t] = price_tm[t * cols + s]; }
        let params = MswParams { period: Some(period) };
        let out = msw_with_kernel(&MswInput::from_slice(&series, params), Kernel::Scalar)?;
        for t in 0..rows {
            cpu_sine_tm[t * cols + s] = out.sine[t];
            cpu_lead_tm[t * cols + s] = out.lead[t];
        }
    }

    // GPU
    let price_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaMsw::new(0).expect("CudaMsw::new");
    let dev = cuda
        .msw_many_series_one_param_time_major_dev(&price_tm_f32, cols, rows, &MswParams { period: Some(period) })
        .expect("msw_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, 2 * cols);
    let mut out = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut out)?;

    let tol = 1.5e-3;
    for t in 0..rows {
        for s in 0..cols {
            let g_sine = out[t * (2*cols) + s] as f64;
            let g_lead = out[t * (2*cols) + (cols + s)] as f64;
            let c_sine = cpu_sine_tm[t * cols + s];
            let c_lead = cpu_lead_tm[t * cols + s];
            assert!(approx_eq(c_sine, g_sine, tol), "sine mismatch t={} s={}", t, s);
            assert!(approx_eq(c_lead, g_lead, tol), "lead mismatch t={} s={}", t, s);
        }
    }
    Ok(())
}

