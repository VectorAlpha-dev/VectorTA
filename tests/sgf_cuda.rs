use vector_ta::indicators::moving_averages::sgf::{
    sgf_batch_with_kernel, SgfBatchRange, SgfBuilder, SgfParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaSgf;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    diff <= tol || diff <= 2e-7 * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn sgf_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[sgf_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 12..series_len {
        let x = i as f64;
        data[i] = 4.0 + 0.5 * x - 0.01 * x * x;
    }

    let sweep = SgfBatchRange {
        period: (7, 13, 2),
        poly_order: (2, 4, 2),
    };

    let cpu = sgf_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let gpu = CudaSgf::new(0)
        .expect("CudaSgf::new")
        .sgf_batch_dev(&data.iter().map(|&v| v as f32).collect::<Vec<_>>(), &sweep)
        .expect("sgf_batch_dev");

    assert_eq!(cpu.rows, gpu.rows);
    assert_eq!(cpu.cols, gpu.cols);

    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf.copy_to(&mut gpu_host).expect("copy batch");

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], gpu_host[idx] as f64, 5e-4),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu.values[idx],
            gpu_host[idx]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn sgf_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[sgf_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 4usize;
    let series_len = 1024usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        for t in (series + 6)..series_len {
            let x = t as f64 + series as f64 * 0.2;
            data_tm[t * num_series + series] = 1.0 + 0.25 * x - 0.005 * x * x;
        }
    }

    let params = SgfParams {
        period: Some(9),
        poly_order: Some(2),
    };
    let period = params.period.unwrap();

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut row = vec![f64::NAN; series_len];
        for t in 0..series_len {
            row[t] = data_tm[t * num_series + series];
        }
        let out = SgfBuilder::new()
            .period(period)
            .poly_order(2)
            .apply_slice(&row)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = out.values[t];
        }
    }

    let gpu = CudaSgf::new(0)
        .expect("CudaSgf::new")
        .sgf_multi_series_one_param_time_major_dev(
            &data_tm.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            num_series,
            series_len,
            &params,
        )
        .expect("sgf_multi_series_one_param_time_major_dev");

    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf.copy_to(&mut gpu_host).expect("copy many-series");

    for idx in 0..cpu_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], gpu_host[idx] as f64, 5e-4),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_tm[idx],
            gpu_host[idx]
        );
    }
    Ok(())
}
