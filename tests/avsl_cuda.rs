use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaAvsl};
use vector_ta::indicators::avsl::{
    avsl_batch_with_kernel, avsl_with_kernel, AvslBatchRange, AvslData, AvslInput, AvslParams,
};
use vector_ta::utilities::data_loader::Candles;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_guard() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn avsl_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[avsl_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 8192usize;
    let mut close = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 64..len {
        let x = i as f64;
        close[i] = (x * 0.00123).sin() + 0.00017 * x;
        low[i] = close[i] - 0.25 * (0.5 + (x * 0.01).cos().abs());
        volume[i] = (x * 0.00077).cos().abs() + 0.5;
    }
    let sweep = AvslBatchRange {
        fast_period: (4, 28, 4),
        slow_period: (32, 96, 16),
        multiplier: (2.0, 2.0, 0.0),
    };

    let cpu = avsl_batch_with_kernel(&close, &low, &volume, &sweep, Kernel::ScalarBatch)?;

    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let cuda = CudaAvsl::new(0).expect("CudaAvsl::new");
    let (dev, _combos) = cuda
        .avsl_batch_dev(&close_f32, &low_f32, &volume_f32, &sweep)
        .expect("avsl_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut gpu_vals = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_vals)?;

    let tol = 1.5e-2;
    for idx in 0..gpu_vals.len() {
        assert!(
            approx_eq(cpu.values[idx], gpu_vals[idx] as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu.values[idx],
            gpu_vals[idx]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn avsl_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[avsl_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 2048usize;
    let mut close_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    let mut vol_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.11;
            let idx = t * cols + s;
            close_tm[idx] = (x * 0.0023).sin() + 0.00019 * x;
            low_tm[idx] = close_tm[idx] - 0.2 * (0.4 + (x * 0.01).sin().abs());
            vol_tm[idx] = (x * 0.0010).cos().abs() + 0.4;
        }
    }

    let fast = 10usize;
    let slow = 26usize;
    let mult = 2.0;

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut c = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for t in 0..rows {
            c[t] = close_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
            v[t] = vol_tm[t * cols + s];
        }
        let params = AvslParams {
            fast_period: Some(fast),
            slow_period: Some(slow),
            multiplier: Some(mult),
        };
        let input = AvslInput {
            data: AvslData::Slices {
                close: &c,
                low: &l,
                volume: &v,
            },
            params,
        };
        let out = avsl_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let vol_tm_f32: Vec<f32> = vol_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaAvsl::new(0).expect("CudaAvsl::new");
    let dev = cuda
        .avsl_many_series_one_param_time_major_dev(
            &close_tm_f32,
            &low_tm_f32,
            &vol_tm_f32,
            cols,
            rows,
            &AvslParams {
                fast_period: Some(fast),
                slow_period: Some(slow),
                multiplier: Some(mult),
            },
        )
        .expect("avsl_many_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_tm)?;
    let tol = 1.5e-2;
    for idx in 0..gpu_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], gpu_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }

    Ok(())
}
