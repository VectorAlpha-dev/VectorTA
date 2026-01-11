

use vector_ta::indicators::fvg_trailing_stop::{
    fvg_trailing_stop_batch_with_kernel, fvg_trailing_stop_with_kernel, FvgTrailingStopInput,
    FvgTrailingStopParams, FvgTsBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::fvg_trailing_stop_wrapper::CudaFvgTs;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

#[test]
fn cuda_feature_guard() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn fvg_ts_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[fvg_ts_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    
    let len = 4096usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        let base = (x * 0.0021).sin() + 0.0002 * x;
        high[i] = base + 0.15;
        low[i] = base - 0.14;
        close[i] = base + 0.01;
    }
    let sweep = FvgTsBatchRange {
        lookback: (3, 9, 3),
        smoothing: (5, 15, 5),
        reset_on_cross: (true, false),
    };

    let cpu =
        fvg_trailing_stop_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;

    let hf: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaFvgTs::new(0).expect("CudaFvgTs::new");
    let batch = cuda
        .fvg_ts_batch_dev(&hf, &lf, &cf, &sweep)
        .expect("fvg_ts_batch_dev");
    assert_eq!(batch.upper.rows, cpu.rows);
    assert_eq!(batch.upper.cols, cpu.cols);
    assert_eq!(batch.lower.rows, cpu.rows);
    assert_eq!(batch.lower.cols, cpu.cols);
    assert_eq!(batch.upper_ts.rows, cpu.rows);
    assert_eq!(batch.upper_ts.cols, cpu.cols);
    assert_eq!(batch.lower_ts.rows, cpu.rows);
    assert_eq!(batch.lower_ts.cols, cpu.cols);

    let mut gu = vec![0f32; batch.upper.len()];
    let mut gl = vec![0f32; batch.lower.len()];
    let mut gut = vec![0f32; batch.upper_ts.len()];
    let mut glt = vec![0f32; batch.lower_ts.len()];
    batch.upper.buf.copy_to(&mut gu)?;
    batch.lower.buf.copy_to(&mut gl)?;
    batch.upper_ts.buf.copy_to(&mut gut)?;
    batch.lower_ts.buf.copy_to(&mut glt)?;

    let tol = 1e-2; 
                    
    let rows = cpu.rows;
    let cols = cpu.cols;
    for r in 0..rows {
        let base = r * 4 * cols;
        for c in 0..cols {
            let iu = base + c;
            let il = base + cols + c;
            let iut = base + 2 * cols + c;
            let ilt = base + 3 * cols + c;
            assert!(
                approx(cpu.values[iu], gu[r * cols + c] as f64, tol),
                "upper r={}, c={}",
                r,
                c
            );
            assert!(
                approx(cpu.values[il], gl[r * cols + c] as f64, tol),
                "lower r={}, c={}",
                r,
                c
            );
            assert!(
                approx(cpu.values[iut], gut[r * cols + c] as f64, tol),
                "upper_ts r={}, c={}",
                r,
                c
            );
            assert!(
                approx(cpu.values[ilt], glt[r * cols + c] as f64, tol),
                "lower_ts r={}, c={}",
                r,
                c
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn fvg_ts_cuda_many_series_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[fvg_ts_cuda_many_series_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 1024usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in 0..rows {
            let idx = t * cols + s;
            let x = (t as f64) + (s as f64) * 0.33;
            let base = (x * 0.0024).cos() + 0.00012 * x;
            high_tm[idx] = base + 0.11;
            low_tm[idx] = base - 0.10;
            close_tm[idx] = base + 0.02;
        }
    }
    let params = FvgTrailingStopParams {
        unmitigated_fvg_lookback: Some(5),
        smoothing_length: Some(9),
        reset_on_cross: Some(false),
    };
    
    let mut cpu_u = vec![f64::NAN; cols * rows];
    let mut cpu_l = vec![f64::NAN; cols * rows];
    let mut cpu_ut = vec![f64::NAN; cols * rows];
    let mut cpu_lt = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![0.0; rows];
        let mut l = vec![0.0; rows];
        let mut c = vec![0.0; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            h[t] = high_tm[idx];
            l[t] = low_tm[idx];
            c[t] = close_tm[idx];
        }
        let input = FvgTrailingStopInput::from_slices(&h, &l, &c, params.clone());
        let out = fvg_trailing_stop_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let idx = t * cols + s;
            cpu_u[idx] = out.upper[t];
            cpu_l[idx] = out.lower[t];
            cpu_ut[idx] = out.upper_ts[t];
            cpu_lt[idx] = out.lower_ts[t];
        }
    }
    let hf: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaFvgTs::new(0).expect("CudaFvgTs::new");
    let (du, dl, dut, dlt) = cuda
        .fvg_ts_many_series_one_param_time_major_dev(&hf, &lf, &cf, cols, rows, &params)
        .expect("many_series");
    let mut gu = vec![0f32; du.len()];
    du.buf.copy_to(&mut gu)?;
    let mut gl = vec![0f32; dl.len()];
    dl.buf.copy_to(&mut gl)?;
    let mut gut = vec![0f32; dut.len()];
    dut.buf.copy_to(&mut gut)?;
    let mut glt = vec![0f32; dlt.len()];
    dlt.buf.copy_to(&mut glt)?;
    let tol = 1e-2;
    for i in 0..gu.len() {
        assert!(
            approx(cpu_u[i], gu[i] as f64, tol),
            "upper mismatch at {}",
            i
        );
        assert!(
            approx(cpu_l[i], gl[i] as f64, tol),
            "lower mismatch at {}",
            i
        );
        assert!(
            approx(cpu_ut[i], gut[i] as f64, tol),
            "upper_ts mismatch at {}",
            i
        );
        assert!(
            approx(cpu_lt[i], glt[i] as f64, tol),
            "lower_ts mismatch at {}",
            i
        );
    }
    Ok(())
}
