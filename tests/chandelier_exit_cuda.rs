// CUDA tests for Chandelier Exit (CE)

use my_project::indicators::chandelier_exit::{
    ce_batch_with_kernel, CeBatchRange, ChandelierExitBuilder, ChandelierExitData,
    ChandelierExitInput, ChandelierExitParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, CudaChandelierExit};

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
fn chandelier_exit_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[chandelier_exit_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    // Synthesize H/L/C with valid prefix
    let len = 8192usize;
    let mut close = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        close[i] = (x * 0.002).sin() + 0.0003 * x;
    }
    let mut high = close.clone();
    let mut low = close.clone();
    for i in 4..len {
        let v = close[i];
        let off = (0.15 + 0.02 * (i as f64).sin()).abs();
        high[i] = v + off;
        low[i] = v - off;
    }

    let sweep = CeBatchRange {
        period: (10, 30, 10),
        mult: (2.0, 3.0, 1.0),
        use_close: (true, true, false),
    };
    let cpu = ce_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;

    // GPU
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaChandelierExit::new(0).expect("cuda ce");
    let (dev, combos) = cuda
        .chandelier_exit_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
        .expect("ce batch dev");
    assert_eq!(combos.len() * 2, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    // Copy back and compare
    let mut got = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got)?;
    let tol = 5e-4;
    for r in 0..cpu.rows {
        let base = r * cpu.cols;
        for i in 0..cpu.cols {
            assert!(
                approx_eq(cpu.values[base + i], got[base + i] as f64, tol),
                "mismatch at row {}, col {}: cpu={} gpu={}",
                r,
                i,
                cpu.values[base + i],
                got[base + i]
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn chandelier_exit_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[chandelier_exit_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            close_tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let mut high_tm = close_tm.clone();
    let mut low_tm = close_tm.clone();
    for s in 0..cols {
        for t in s..rows {
            let v = close_tm[t * cols + s];
            let off = (0.15 + 0.02 * (t as f64).cos()).abs();
            high_tm[t * cols + s] = v + off;
            low_tm[t * cols + s] = v - off;
        }
    }

    let period = 22usize;
    let mult = 3.0f64;
    let use_close = true;

    // CPU baseline per series
    let mut cpu_long_tm = vec![f64::NAN; cols * rows];
    let mut cpu_short_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = high_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
            c[t] = close_tm[t * cols + s];
        }
        let params = ChandelierExitParams {
            period: Some(period),
            mult: Some(mult),
            use_close: Some(use_close),
        };
        let input = ChandelierExitInput {
            data: ChandelierExitData::Slices {
                high: &h,
                low: &l,
                close: &c,
            },
            params,
        };
        let out = ChandelierExitBuilder::new()
            .period(period)
            .mult(mult)
            .use_close(use_close)
            .apply_slices(&h, &l, &c)
            .unwrap();
        for t in 0..rows {
            cpu_long_tm[t * cols + s] = out.long_stop[t];
            cpu_short_tm[t * cols + s] = out.short_stop[t];
        }
    }

    // GPU
    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaChandelierExit::new(0).expect("cuda ce");
    let dev = cuda
        .chandelier_exit_many_series_one_param_time_major_dev(
            &high_tm_f32,
            &low_tm_f32,
            &close_tm_f32,
            cols,
            rows,
            period,
            mult as f32,
            use_close,
        )
        .expect("ce many-series dev");
    assert_eq!(dev.rows, 2 * rows);
    assert_eq!(dev.cols, cols);
    let mut got = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got)?;
    let got_long = &got[..rows * cols];
    let got_short = &got[rows * cols..];
    let tol = 5e-4;
    for idx in 0..(rows * cols) {
        assert!(
            approx_eq(cpu_long_tm[idx], got_long[idx] as f64, tol),
            "long mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_short_tm[idx], got_short[idx] as f64, tol),
            "short mismatch at {}",
            idx
        );
    }
    Ok(())
}
