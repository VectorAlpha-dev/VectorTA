use my_project::utilities::enums::Kernel;

use my_project::indicators::mfi::{
    mfi_batch_with_kernel, mfi_with_kernel, MfiBatchRange, MfiData, MfiInput, MfiParams,
};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaMfi;

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
fn mfi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mfi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let n = 50_000usize;
    let mut tp = vec![f64::NAN; n];
    let mut vol = vec![f64::NAN; n];
    for i in 8..n {
        let x = i as f64 * 0.0021;
        let base = 100.0 + x.sin() * 0.7 + 0.0003 * (i as f64);
        tp[i] = base;
        vol[i] = ((x * 0.9).cos().abs() + 1.1) * 1_000.0;
    }
    let sweep = MfiBatchRange { period: (5, 60, 5) };
    // CPU baseline using f32-rounded inputs to match GPU math path
    let tp32: Vec<f32> = tp.iter().map(|&v| v as f32).collect();
    let vol32: Vec<f32> = vol.iter().map(|&v| v as f32).collect();
    let tp_cpu: Vec<f64> = tp32.iter().map(|&v| v as f64).collect();
    let vol_cpu: Vec<f64> = vol32.iter().map(|&v| v as f64).collect();
    let cpu = mfi_batch_with_kernel(&tp_cpu, &vol_cpu, &sweep, Kernel::ScalarBatch)?;

    // No debug: end local checks
    let cuda = CudaMfi::new(0).expect("CudaMfi::new");
    let (dev, combos) = cuda
        .mfi_batch_dev(&tp32, &vol32, &sweep)
        .expect("mfi_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    assert_eq!(cpu.combos.len(), combos.len());

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-3;
    for idx in 0..(cpu.rows * cpu.cols) {
        if !approx_eq(cpu.values[idx], host[idx] as f64, tol) {
            let row = idx / cpu.cols;
            let col = idx % cpu.cols;
            let p = cpu.combos[row].period.unwrap();
            eprintln!(
                "row {} col {} (idx {}) period {} cpu={} gpu={}",
                row, col, idx, p, cpu.values[idx], host[idx]
            );
            assert!(false, "mismatch at {}", idx);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn mfi_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[mfi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 16usize; // series count
    let rows = 4096usize; // time
    let mut tp_tm = vec![f64::NAN; cols * rows];
    let mut vol_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s % 7)..rows {
            let x = t as f64 * 0.003 + s as f64 * 0.011;
            tp_tm[t * cols + s] = 100.0 + x.cos() * 0.6 + 0.0002 * (t as f64);
            vol_tm[t * cols + s] = ((x * 0.7).sin().abs() + 0.9) * (900.0 + 10.0 * s as f64);
        }
    }
    let period = 25usize;

    // CPU baseline per series (with f32-rounded inputs)
    let mut cpu = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut tp = vec![f64::NAN; rows];
        let mut vol = vec![f64::NAN; rows];
        for t in 0..rows {
            tp[t] = (tp_tm[t * cols + s] as f32) as f64;
            vol[t] = (vol_tm[t * cols + s] as f32) as f64;
        }
        let params = MfiParams {
            period: Some(period),
        };
        let input = MfiInput {
            data: MfiData::Slices {
                typical_price: &tp,
                volume: &vol,
            },
            params,
        };
        let out = mfi_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu[t * cols + s] = out.values[t];
        }
    }

    let tp32: Vec<f32> = tp_tm.iter().map(|&v| v as f32).collect();
    let vol32: Vec<f32> = vol_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaMfi::new(0).expect("CudaMfi::new");
    let dev = cuda
        .mfi_many_series_one_param_time_major_dev(&tp32, &vol32, cols, rows, period)
        .expect("mfi_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut got = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got)?;
    let tol = 5e-3;
    for idx in 0..got.len() {
        if !approx_eq(cpu[idx], got[idx] as f64, tol) {
            eprintln!("idx {} cpu={} gpu={}", idx, cpu[idx], got[idx]);
            assert!(false, "mismatch at {}", idx);
        }
    }
    Ok(())
}
