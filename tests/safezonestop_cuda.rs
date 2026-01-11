

use vector_ta::indicators::safezonestop::{
    safezonestop_batch_with_kernel, safezonestop_with_kernel, SafeZoneStopBatchRange,
    SafeZoneStopBuilder, SafeZoneStopInput, SafeZoneStopParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaSafeZoneStop;

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
fn safezonestop_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[safezonestop_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 6000usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        let base = (x * 0.00123).sin() + 0.00021 * x;
        high[i] = base + 0.5;
        low[i] = base - 0.5;
    }
    let sweep = SafeZoneStopBatchRange {
        period: (10, 22, 6),
        mult: (1.5, 3.0, 0.75),
        max_lookback: (3, 5, 1),
    };

    let cpu = safezonestop_batch_with_kernel(&high, &low, &sweep, "long", Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cuda = CudaSafeZoneStop::new(0).expect("CudaSafeZoneStop::new");
    let (dev, combos) = cuda
        .safezonestop_batch_dev(&high_f32, &low_f32, "long", &sweep)
        .expect("cuda szz batch");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    assert_eq!(cpu.combos.len(), combos.len());

    use cust::memory::CopyDestination;
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.values[idx], host[idx] as f64, tol),
            "mismatch at {}: cpu={}, gpu={}",
            idx,
            cpu.values[idx],
            host[idx]
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn safezonestop_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[safezonestop_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize; 
    let rows = 4000usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            let base = (x * 0.002).sin() + 0.0003 * x;
            high_tm[t * cols + s] = base + 0.5;
            low_tm[t * cols + s] = base - 0.5;
        }
    }

    let period = 22usize;
    let mult = 2.5f64;
    let lb = 3usize;

    
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut high = vec![f64::NAN; rows];
        let mut low = vec![f64::NAN; rows];
        for t in 0..rows {
            high[t] = high_tm[t * cols + s];
            low[t] = low_tm[t * cols + s];
        }
        let params = SafeZoneStopParams {
            period: Some(period),
            mult: Some(mult),
            max_lookback: Some(lb),
        };
        let input = SafeZoneStopInput::from_slices(&high, &low, "long", params);
        let out = safezonestop_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaSafeZoneStop::new(0).expect("CudaSafeZoneStop::new");
    let dev_tm = cuda
        .safezonestop_many_series_one_param_time_major_dev(
            &high_tm_f32,
            &low_tm_f32,
            cols,
            rows,
            period,
            mult as f32,
            lb,
            "long",
        )
        .expect("szz many-series");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);

    use cust::memory::CopyDestination;
    let mut gpu_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut gpu_tm)?;

    let tol = 7e-4;
    for idx in 0..gpu_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], gpu_tm[idx] as f64, tol),
            "ms mismatch at {}: cpu={}, gpu={}",
            idx,
            cpu_tm[idx],
            gpu_tm[idx]
        );
    }

    Ok(())
}
