use vector_ta::indicators::rsmk::{
    rsmk_batch_with_kernel, rsmk_with_kernel, RsmkBatchRange, RsmkData, RsmkInput, RsmkParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaRsmk;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a == b {
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
fn rsmk_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rsmk_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut main = vec![f64::NAN; len];
    let mut comp = vec![f64::NAN; len];
    for i in 10..len {
        let x = i as f64;
        main[i] = (x * 0.00123).sin() + 0.00017 * x;
        comp[i] = (x * 0.00077).cos().abs() + 0.5;
    }
    let sweep = RsmkBatchRange {
        lookback: (30, 42, 6),
        period: (3, 9, 3),
        signal_period: (10, 22, 6),
    };
    let cpu = rsmk_batch_with_kernel(&main, &comp, &sweep, Kernel::ScalarBatch)?;

    let main_f32: Vec<f32> = main.iter().map(|&v| v as f32).collect();
    let comp_f32: Vec<f32> = comp.iter().map(|&v| v as f32).collect();
    let cuda = CudaRsmk::new(0).expect("CudaRsmk::new");
    let (pair, _combos) = cuda
        .rsmk_batch_dev(&main_f32, &comp_f32, &sweep)
        .expect("rsmk batch dev");

    assert_eq!(cpu.rows, pair.a.rows);
    assert_eq!(cpu.cols, pair.a.cols);
    assert_eq!(cpu.rows, pair.b.rows);
    assert_eq!(cpu.cols, pair.b.cols);

    let mut ind_host = vec![0f32; pair.a.len()];
    let mut sig_host = vec![0f32; pair.b.len()];
    pair.a.buf.copy_to(&mut ind_host)?;
    pair.b.buf.copy_to(&mut sig_host)?;

    let tol = 6e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.indicator[idx], ind_host[idx] as f64, tol),
            "indicator mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.signal[idx], sig_host[idx] as f64, tol),
            "signal mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn rsmk_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rsmk_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let mut main_tm = vec![f64::NAN; cols * rows];
    let mut comp_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            main_tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
            comp_tm[t * cols + s] = (x * 0.001).cos().abs() + 0.4;
        }
    }

    let lookback = 30usize;
    let period = 3usize;
    let sigp = 10usize;

    let mut cpu_ind_tm = vec![f64::NAN; cols * rows];
    let mut cpu_sig_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut main = vec![f64::NAN; rows];
        let mut comp = vec![f64::NAN; rows];
        for t in 0..rows {
            main[t] = main_tm[t * cols + s];
            comp[t] = comp_tm[t * cols + s];
        }
        let input = RsmkInput {
            data: RsmkData::Slices {
                main: &main,
                compare: &comp,
            },
            params: RsmkParams {
                lookback: Some(lookback),
                period: Some(period),
                signal_period: Some(sigp),
                matype: Some("ema".into()),
                signal_matype: Some("ema".into()),
            },
        };
        let out = rsmk_with_kernel(&input, Kernel::Scalar).unwrap();
        for t in 0..rows {
            cpu_ind_tm[t * cols + s] = out.indicator[t];
            cpu_sig_tm[t * cols + s] = out.signal[t];
        }
    }

    let main_tm_f32: Vec<f32> = main_tm.iter().map(|&v| v as f32).collect();
    let comp_tm_f32: Vec<f32> = comp_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaRsmk::new(0).expect("CudaRsmk::new");
    let pair = cuda
        .rsmk_many_series_one_param_time_major_dev(
            &main_tm_f32,
            &comp_tm_f32,
            cols,
            rows,
            &RsmkParams {
                lookback: Some(lookback),
                period: Some(period),
                signal_period: Some(sigp),
                matype: Some("ema".into()),
                signal_matype: Some("ema".into()),
            },
        )
        .expect("rsmk many series");

    assert_eq!(pair.a.rows, rows);
    assert_eq!(pair.a.cols, cols);
    assert_eq!(pair.b.rows, rows);
    assert_eq!(pair.b.cols, cols);
    let mut g_ind_tm = vec![0f32; pair.a.len()];
    let mut g_sig_tm = vec![0f32; pair.b.len()];
    pair.a.buf.copy_to(&mut g_ind_tm)?;
    pair.b.buf.copy_to(&mut g_sig_tm)?;

    let tol = 1e-4;
    for idx in 0..g_ind_tm.len() {
        assert!(
            approx_eq(cpu_ind_tm[idx], g_ind_tm[idx] as f64, tol),
            "indicator mismatch at {} (cpu={}, gpu={})",
            idx,
            cpu_ind_tm[idx],
            g_ind_tm[idx]
        );
        assert!(
            approx_eq(cpu_sig_tm[idx], g_sig_tm[idx] as f64, tol),
            "signal mismatch at {} (cpu={}, gpu={})",
            idx,
            cpu_sig_tm[idx],
            g_sig_tm[idx]
        );
    }
    Ok(())
}
