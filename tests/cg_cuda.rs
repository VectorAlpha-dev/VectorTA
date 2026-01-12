use vector_ta::indicators::cg::{
    cg_batch_with_kernel, cg_with_kernel, CgBatchRange, CgInput, CgParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::CudaCg;

fn approx_close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    diff <= (rtol * a.abs().max(b.abs())).max(atol)
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
fn cg_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cg_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 8192usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 40..series_len {
        let x = i as f64;
        data[i] = (x * 0.0031).sin() + 0.0003 * x;
    }

    let sweep = CgBatchRange {
        period: (5, 45, 10),
    };
    let cpu = cg_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaCg::new(0).expect("CudaCg::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu = cuda
        .cg_batch_dev(&data_f32, &sweep)
        .expect("cuda cg_batch_dev");

    assert_eq!(cpu.rows, gpu.rows);
    assert_eq!(cpu.cols, gpu.cols);

    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda cg batch result");

    let rtol = 1.0e-3f64;
    let atol = 6.0e-2f64;
    for idx in 0..(cpu.rows * cpu.cols) {
        let a = cpu.values[idx];
        let b = gpu_host[idx] as f64;
        assert!(
            approx_close(a, b, rtol, atol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cg_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cg_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 7usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for s in 0..num_series {
        for t in (s * 3)..series_len {
            let idx = t * num_series + s;
            let x = (t as f64) * 0.51 + (s as f64) * 0.11;
            data_tm[idx] = (x * 0.0023).cos() + 0.0004 * x;
        }
    }

    let period = 20usize;
    let params = CgParams {
        period: Some(period),
    };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for s in 0..num_series {
        let mut series = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series[t] = data_tm[t * num_series + s];
        }
        let out = cg_with_kernel(
            &CgInput::from_slice(&series, params.clone()),
            Kernel::Scalar,
        )?;
        for t in 0..series_len {
            cpu_tm[t * num_series + s] = out.values[t];
        }
    }

    let cuda = CudaCg::new(0).expect("CudaCg::new");
    let tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let gpu = cuda
        .cg_many_series_one_param_time_major_dev(&tm_f32, num_series, series_len, &params)
        .expect("cuda cg_many_series_one_param_time_major_dev");
    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda cg many-series result");

    let rtol = 1.0e-3f64;
    let atol = 6.0e-2f64;
    for i in 0..(num_series * series_len) {
        let a = cpu_tm[i];
        let b = gpu_host[i] as f64;
        assert!(
            approx_close(a, b, rtol, atol),
            "mismatch at {}: cpu={} gpu={}",
            i,
            a,
            b
        );
    }
    Ok(())
}
