use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::wavetrend::CudaWavetrend;

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
fn wavetrend_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[wavetrend_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    use vector_ta::indicators::wavetrend::{wavetrend_batch_with_kernel, WavetrendBatchRange};

    let len = 8192usize;
    let mut data = vec![f64::NAN; len];
    for i in 8..len {
        let x = i as f64;
        data[i] = (x * 0.00137).sin() + 0.00029 * x;
    }
    let sweep = WavetrendBatchRange {
        channel_length: (6, 20, 7),
        average_length: (9, 21, 6),
        ma_length: (3, 5, 1),
        factor: (0.015, 0.015, 0.0),
    };

    let cpu = wavetrend_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaWavetrend::new(0).expect("CudaWavetrend::new");
    let dev = cuda
        .wavetrend_batch_dev(&data_f32, &sweep)
        .expect("wavetrend_batch_dev");

    assert_eq!(dev.combos.len(), cpu.rows);
    assert_eq!(dev.wt1.rows, cpu.rows);
    assert_eq!(dev.wt1.cols, cpu.cols);
    assert_eq!(dev.wt2.rows, cpu.rows);
    assert_eq!(dev.wt2.cols, cpu.cols);
    assert_eq!(dev.wt_diff.rows, cpu.rows);
    assert_eq!(dev.wt_diff.cols, cpu.cols);

    let mut wt1_g = vec![0f32; dev.wt1.len()];
    let mut wt2_g = vec![0f32; dev.wt2.len()];
    let mut diff_g = vec![0f32; dev.wt_diff.len()];
    dev.wt1.buf.copy_to(&mut wt1_g)?;
    dev.wt2.buf.copy_to(&mut wt2_g)?;
    dev.wt_diff.buf.copy_to(&mut diff_g)?;

    let mut n1 = 0usize;
    let mut sse1 = 0.0f64;
    let mut n2 = 0usize;
    let mut sse2 = 0.0f64;
    let mut n3 = 0usize;
    let mut sse3 = 0.0f64;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c1 = cpu.wt1[idx];
        let g1 = wt1_g[idx] as f64;
        if c1.is_finite() && g1.is_finite() {
            n1 += 1;
            sse1 += (c1 - g1) * (c1 - g1);
        }
        let c2 = cpu.wt2[idx];
        let g2 = wt2_g[idx] as f64;
        if c2.is_finite() && g2.is_finite() {
            n2 += 1;
            sse2 += (c2 - g2) * (c2 - g2);
        }
        let c3 = cpu.wt_diff[idx];
        let g3 = diff_g[idx] as f64;
        if c3.is_finite() && g3.is_finite() {
            n3 += 1;
            sse3 += (c3 - g3) * (c3 - g3);
        }
    }
    let rmse1 = (sse1 / (n1.max(1) as f64)).sqrt();
    let rmse2 = (sse2 / (n2.max(1) as f64)).sqrt();
    let rmse3 = (sse3 / (n3.max(1) as f64)).sqrt();
    assert!(rmse1 <= 0.15, "wt1 RMSE too high: {} (n={})", rmse1, n1);
    assert!(rmse2 <= 0.15, "wt2 RMSE too high: {} (n={})", rmse2, n2);
    assert!(rmse3 <= 0.10, "wt_diff RMSE too high: {} (n={})", rmse3, n3);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn wavetrend_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[wavetrend_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    use vector_ta::indicators::wavetrend::{
        wavetrend_with_kernel, WavetrendData, WavetrendInput, WavetrendParams,
    };

    let cols = 16usize;
    let rows = 4096usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (8 + s)..rows {
            let x = t as f64 + (s as f64) * 0.13;
            tm[t * cols + s] = (x * 0.0023).cos() + 0.00037 * x;
        }
    }
    let params = WavetrendParams {
        channel_length: Some(10),
        average_length: Some(21),
        ma_length: Some(4),
        factor: Some(0.015),
    };

    let mut wt1_cpu_tm = vec![f64::NAN; cols * rows];
    let mut wt2_cpu_tm = vec![f64::NAN; cols * rows];
    let mut diff_cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = tm[t * cols + s];
        }
        let input = WavetrendInput {
            data: WavetrendData::Slice(&series),
            params: params.clone(),
        };
        let out = wavetrend_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            wt1_cpu_tm[t * cols + s] = out.wt1[t];
            wt2_cpu_tm[t * cols + s] = out.wt2[t];
            diff_cpu_tm[t * cols + s] = out.wt_diff[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaWavetrend::new(0).expect("CudaWavetrend::new");
    let (dev_wt1_tm, dev_wt2_tm, dev_diff_tm) = cuda
        .wavetrend_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("wavetrend many-series one-param");

    assert_eq!(dev_wt1_tm.rows, rows);
    assert_eq!(dev_wt1_tm.cols, cols);
    assert_eq!(dev_wt2_tm.rows, rows);
    assert_eq!(dev_wt2_tm.cols, cols);
    assert_eq!(dev_diff_tm.rows, rows);
    assert_eq!(dev_diff_tm.cols, cols);

    let mut wt1_g = vec![0f32; dev_wt1_tm.len()];
    let mut wt2_g = vec![0f32; dev_wt2_tm.len()];
    let mut diff_g = vec![0f32; dev_diff_tm.len()];
    dev_wt1_tm.buf.copy_to(&mut wt1_g)?;
    dev_wt2_tm.buf.copy_to(&mut wt2_g)?;
    dev_diff_tm.buf.copy_to(&mut diff_g)?;

    let tol = 5e-2;
    for idx in 0..wt1_g.len() {
        assert!(
            approx_eq(wt1_cpu_tm[idx], wt1_g[idx] as f64, tol),
            "wt1 mismatch at {}",
            idx
        );
        assert!(
            approx_eq(wt2_cpu_tm[idx], wt2_g[idx] as f64, tol),
            "wt2 mismatch at {}",
            idx
        );
        assert!(
            approx_eq(diff_cpu_tm[idx], diff_g[idx] as f64, tol),
            "wt_diff mismatch at {}",
            idx
        );
    }
    Ok(())
}
