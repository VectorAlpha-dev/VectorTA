

use vector_ta::indicators::mass::{
    mass_batch_with_kernel, mass_with_kernel, MassBatchRange, MassData, MassInput, MassParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaMass};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop_mass() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn mass_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mass_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 24_000usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 20..len {
        let x = i as f64;
        let h = (x * 0.0023).sin().abs() + 1.0;
        let l = h - (0.5 + (x * 0.0017).cos().abs());
        high[i] = h;
        low[i] = l;
    }

    let sweep = MassBatchRange { period: (2, 18, 2) };
    let cpu = mass_batch_with_kernel(&high, &low, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();

    let mut cuda = CudaMass::new(0).expect("CudaMass::new");
    let (dev, combos) = cuda
        .mass_batch_dev(&high_f32, &low_f32, &sweep)
        .expect("cuda mass batch");
    assert_eq!(cpu.rows, combos.len());
    assert_eq!(dev.rows, combos.len());
    assert_eq!(dev.cols, len);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 1e-3;
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
fn mass_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mass_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 4096usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            let h = (x * 0.002).sin().abs() + 1.2;
            let l = h - (0.4 + (x * 0.001).cos().abs());
            high_tm[t * cols + s] = h;
            low_tm[t * cols + s] = l;
        }
    }

    let period = 9usize;

    
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = high_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
        }
        let params = MassParams {
            period: Some(period),
        };
        let input = MassInput {
            data: MassData::Slices { high: &h, low: &l },
            params,
        };
        let out = mass_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let mut cuda = CudaMass::new(0).expect("CudaMass::new");
    let dev = cuda
        .mass_many_series_one_param_time_major_dev(
            &high_tm_f32,
            &low_tm_f32,
            cols,
            rows,
            &MassParams {
                period: Some(period),
            },
        )
        .expect("mass many-series");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 1e-3;
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu_tm[idx], host[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
