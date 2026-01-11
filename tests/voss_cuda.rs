

use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaVoss;

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
fn voss_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[voss_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    use vector_ta::indicators::voss::{voss_batch_with_kernel, VossBatchRange};

    let len = 8192usize;
    let mut data = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        data[i] = (x * 0.00123).sin() + 0.00037 * x;
    }
    let sweep = VossBatchRange {
        period: (10, 34, 4),
        predict: (1, 4, 1),
        bandwidth: (0.10, 0.40, 0.10),
    };

    let cpu = voss_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaVoss::new(0).expect("CudaVoss::new");
    let (dev_voss, dev_filt, combos) = cuda
        .voss_batch_dev(&data_f32, &sweep)
        .expect("voss_batch_dev");

    assert_eq!(combos.len(), cpu.rows);
    assert_eq!(dev_voss.rows, cpu.rows);
    assert_eq!(dev_voss.cols, cpu.cols);
    assert_eq!(dev_filt.rows, cpu.rows);
    assert_eq!(dev_filt.cols, cpu.cols);

    let mut voss_g = vec![0f32; dev_voss.len()];
    let mut filt_g = vec![0f32; dev_filt.len()];
    dev_voss.buf.copy_to(&mut voss_g)?;
    dev_filt.buf.copy_to(&mut filt_g)?;

    let tol = 1e-2; 
    for idx in 0..(cpu.rows * cpu.cols) {
        let cv = cpu.voss[idx];
        let gv = voss_g[idx] as f64;
        let cf = cpu.filt[idx];
        let gf = filt_g[idx] as f64;
        assert!(
            approx_eq(cv, gv, tol),
            "voss mismatch at {}: cpu={} gpu={}",
            idx,
            cv,
            gv
        );
        assert!(
            approx_eq(cf, gf, tol),
            "filt mismatch at {}: cpu={} gpu={}",
            idx,
            cf,
            gf
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn voss_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[voss_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    use vector_ta::indicators::voss::{voss_with_kernel, VossData, VossInput, VossParams};

    let cols = 8usize; 
    let rows = 2048usize; 
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (2 + s)..rows {
            
            let x = t as f64 + (s as f64) * 0.1;
            tm[t * cols + s] = (x * 0.0021).sin() + 0.00021 * x;
        }
    }
    let params = VossParams {
        period: Some(20),
        predict: Some(3),
        bandwidth: Some(0.25),
    };

    
    let mut voss_cpu_tm = vec![f64::NAN; cols * rows];
    let mut filt_cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = tm[t * cols + s];
        }
        let input = VossInput {
            data: VossData::Slice(&series),
            params: params.clone(),
        };
        let out = voss_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            voss_cpu_tm[t * cols + s] = out.voss[t];
            filt_cpu_tm[t * cols + s] = out.filt[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaVoss::new(0).expect("CudaVoss::new");
    let (dev_voss_tm, dev_filt_tm) = cuda
        .voss_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("voss_many_series_one_param_time_major_dev");

    assert_eq!(dev_voss_tm.rows, rows);
    assert_eq!(dev_voss_tm.cols, cols);
    assert_eq!(dev_filt_tm.rows, rows);
    assert_eq!(dev_filt_tm.cols, cols);

    let mut voss_g = vec![0f32; dev_voss_tm.len()];
    let mut filt_g = vec![0f32; dev_filt_tm.len()];
    dev_voss_tm.buf.copy_to(&mut voss_g)?;
    dev_filt_tm.buf.copy_to(&mut filt_g)?;

    let tol = 1e-2;
    for idx in 0..voss_g.len() {
        assert!(
            approx_eq(voss_cpu_tm[idx], voss_g[idx] as f64, tol),
            "voss mismatch at {}",
            idx
        );
        assert!(
            approx_eq(filt_cpu_tm[idx], filt_g[idx] as f64, tol),
            "filt mismatch at {}",
            idx
        );
    }
    Ok(())
}
