// CUDA DI (+DI, -DI) integration tests

use my_project::indicators::di::{
    di_batch_with_kernel, di_with_kernel, DiBatchRange, DiData, DiInput, DiParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, CudaDi};

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
fn di_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[di_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 16_384usize;
    let mut close = vec![f64::NAN; len];
    for i in 6..len {
        let x = i as f64;
        close[i] = (x * 0.00137).sin() + 0.00009 * x;
    }
    let mut high = close.clone();
    let mut low = close.clone();
    for i in 0..len {
        if close[i].is_nan() {
            continue;
        }
        let off = 0.12 + ((i as f64) * 0.0031).cos().abs() * 0.02;
        high[i] = close[i] + off;
        low[i] = close[i] - off;
    }

    let sweep = DiBatchRange { period: (5, 40, 5) };
    let cpu = di_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;

    let h_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let l_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let c_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaDi::new(0).expect("CudaDi::new");
    let (plus_dev, minus_dev, combos) = cuda
        .di_batch_dev(&h_f32, &l_f32, &c_f32, &sweep)
        .expect("di cuda batch dev");

    assert_eq!(combos.len(), cpu.rows);
    assert_eq!(plus_dev.rows, cpu.rows);
    assert_eq!(plus_dev.cols, cpu.cols);
    assert_eq!(minus_dev.rows, cpu.rows);
    assert_eq!(minus_dev.cols, cpu.cols);

    let mut g_plus = vec![0f32; plus_dev.len()];
    let mut g_minus = vec![0f32; minus_dev.len()];
    plus_dev.buf.copy_to(&mut g_plus)?;
    minus_dev.buf.copy_to(&mut g_minus)?;

    let tol = 1e-3; // FP32 path
    for idx in 0..(cpu.rows * cpu.cols) {
        let c_pl = cpu.plus[idx];
        let g_pl = g_plus[idx] as f64;
        let c_mi = cpu.minus[idx];
        let g_mi = g_minus[idx] as f64;
        assert!(
            approx_eq(c_pl, g_pl, tol),
            "plus mismatch at {}: cpu={} gpu={}",
            idx,
            c_pl,
            g_pl
        );
        assert!(
            approx_eq(c_mi, g_mi, tol),
            "minus mismatch at {}: cpu={} gpu={}",
            idx,
            c_mi,
            g_mi
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn di_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[di_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize; // series count
    let rows = 4096usize; // points per series
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.3;
            close_tm[t * cols + s] = (x * 0.0023).sin() + 0.0002 * x;
        }
    }
    let mut high_tm = close_tm.clone();
    let mut low_tm = close_tm.clone();
    for s in 0..cols {
        for t in 0..rows {
            let idx = t * cols + s;
            if close_tm[idx].is_nan() {
                continue;
            }
            let off = 0.1 + ((t as f64) * 0.0029).cos().abs() * 0.02;
            high_tm[idx] = close_tm[idx] + off;
            low_tm[idx] = close_tm[idx] - off;
        }
    }

    let period = 14usize;

    // CPU baseline per series
    let mut cpu_plus_tm = vec![f64::NAN; cols * rows];
    let mut cpu_minus_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            h[t] = high_tm[idx];
            l[t] = low_tm[idx];
            c[t] = close_tm[idx];
        }
        let input = DiInput {
            data: DiData::Slices {
                high: &h,
                low: &l,
                close: &c,
            },
            params: DiParams {
                period: Some(period),
            },
        };
        let out = di_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let idx = t * cols + s;
            cpu_plus_tm[idx] = out.plus[t];
            cpu_minus_tm[idx] = out.minus[t];
        }
    }

    let h_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let l_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let c_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaDi::new(0).expect("CudaDi::new");
    let pair = cuda
        .di_many_series_one_param_time_major_dev(&h_f32, &l_f32, &c_f32, cols, rows, period)
        .expect("di many series");
    assert_eq!(pair.rows(), rows);
    assert_eq!(pair.cols(), cols);
    let mut g_plus_tm = vec![0f32; pair.plus.len()];
    let mut g_minus_tm = vec![0f32; pair.minus.len()];
    pair.plus.buf.copy_to(&mut g_plus_tm)?;
    pair.minus.buf.copy_to(&mut g_minus_tm)?;

    let tol = 1e-3;
    for idx in 0..(cols * rows) {
        assert!(
            approx_eq(cpu_plus_tm[idx], g_plus_tm[idx] as f64, tol),
            "plus mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu_minus_tm[idx], g_minus_tm[idx] as f64, tol),
            "minus mismatch at {}",
            idx
        );
    }
    Ok(())
}
