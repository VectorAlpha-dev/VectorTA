use vector_ta::indicators::dec_osc::{
    dec_osc_batch_with_kernel, dec_osc_with_kernel, DecOscBatchBuilder, DecOscBatchRange,
    DecOscInput, DecOscParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::CudaDecOsc;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop_dec_osc() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn dec_osc_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dec_osc_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut data = vec![f64::NAN; len];
    for i in 0..len {
        if i >= 2 {
            let x = i as f64;
            data[i] = (x * 0.00123).sin() + 0.00017 * x + 50.0;
        }
    }
    let sweep = DecOscBatchRange {
        hp_period: (50, 90, 10),
        k: (0.5, 1.5, 0.5),
    };

    let cpu = dec_osc_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaDecOsc::new(0).expect("CudaDecOsc::new");
    let dev = cuda
        .dec_osc_batch_dev(&data_f32, &sweep)
        .expect("dec_osc_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 3e-3;
    for idx in 0..host.len() {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "mismatch at {}: cpu={}, gpu={}",
            idx,
            c,
            g
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn dec_osc_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dec_osc_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in 0..rows {
            if t >= 2 {
                let x = (t as f64) + (s as f64) * 0.2;
                tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x + 100.0;
            }
        }
    }
    let params = DecOscParams {
        hp_period: Some(64),
        k: Some(1.0),
    };

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = tm[t * cols + s];
        }
        let input = DecOscInput::from_slice(&series, params.clone());
        let out = dec_osc_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
    }

    let cuda = CudaDecOsc::new(0).expect("CudaDecOsc::new");
    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let dev = cuda
        .dec_osc_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("dec_osc_many_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 3e-3;
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu_tm[idx], host[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
