use vector_ta::indicators::moving_averages::ehlers_pma::{
    ehlers_pma, EhlersPmaBatchRange, EhlersPmaInput, EhlersPmaParams,
};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaEhlersPma;

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
fn ehlers_pma_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ehlers_pma_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 2048usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 6..series_len {
        let x = i as f64;
        data[i] = (x * 0.0013).sin() + (x * 0.0007).cos() * 0.25 + 0.0002 * x;
    }

    let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams);
    let cpu = ehlers_pma(&input)?;

    let sweep = EhlersPmaBatchRange { combos: 3 };
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();

    let cuda = CudaEhlersPma::new(0).expect("CudaEhlersPma::new");
    let pair = cuda
        .ehlers_pma_batch_dev(&data_f32, &sweep)
        .expect("cuda ehlers_pma_batch_dev");

    assert_eq!(pair.rows(), sweep.combos);
    assert_eq!(pair.cols(), series_len);

    let mut gpu_predict = vec![0f32; pair.predict.len()];
    let mut gpu_trigger = vec![0f32; pair.trigger.len()];
    pair.predict
        .buf
        .copy_to(&mut gpu_predict)
        .expect("copy predict");
    pair.trigger
        .buf
        .copy_to(&mut gpu_trigger)
        .expect("copy trigger");

    let tol = 2e-5;
    for combo in 0..sweep.combos {
        for idx in 0..series_len {
            let cpu_p = cpu.predict[idx];
            let gpu_p = gpu_predict[combo * series_len + idx] as f64;
            assert!(
                approx_eq(cpu_p, gpu_p, tol),
                "predict mismatch combo {} idx {}: cpu={} gpu={}",
                combo,
                idx,
                cpu_p,
                gpu_p
            );

            let cpu_t = cpu.trigger[idx];
            let gpu_t = gpu_trigger[combo * series_len + idx] as f64;
            assert!(
                approx_eq(cpu_t, gpu_t, tol),
                "trigger mismatch combo {} idx {}: cpu={} gpu={}",
                combo,
                idx,
                cpu_t,
                gpu_t
            );
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ehlers_pma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ehlers_pma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 5usize;
    let series_len = 1024usize;

    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let start = (series % 6) as usize;
        for row in start..series_len {
            let idx = row * num_series + series;
            let x = row as f64 + (series as f64) * 0.37;
            data_tm[idx] = (x * 0.0016).cos() + (x * 0.0011).sin() * 0.4 + 0.0003 * x;
        }
    }

    let mut cpu_predict_tm = vec![f64::NAN; num_series * series_len];
    let mut cpu_trigger_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut series_data = vec![f64::NAN; series_len];
        for row in 0..series_len {
            series_data[row] = data_tm[row * num_series + series];
        }
        let input = EhlersPmaInput::from_slice(&series_data, EhlersPmaParams);
        let out = ehlers_pma(&input)?;
        for row in 0..series_len {
            let idx = row * num_series + series;
            cpu_predict_tm[idx] = out.predict[row];
            cpu_trigger_tm[idx] = out.trigger[row];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaEhlersPma::new(0).expect("CudaEhlersPma::new");
    let pair = cuda
        .ehlers_pma_many_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len)
        .expect("cuda ehlers_pma_many_series_one_param_time_major_dev");

    assert_eq!(pair.rows(), series_len);
    assert_eq!(pair.cols(), num_series);

    let mut gpu_predict_tm = vec![0f32; pair.predict.len()];
    let mut gpu_trigger_tm = vec![0f32; pair.trigger.len()];
    pair.predict
        .buf
        .copy_to(&mut gpu_predict_tm)
        .expect("copy predict tm");
    pair.trigger
        .buf
        .copy_to(&mut gpu_trigger_tm)
        .expect("copy trigger tm");

    let tol = 2e-5;
    for idx in 0..num_series * series_len {
        let cpu_p = cpu_predict_tm[idx];
        let gpu_p = gpu_predict_tm[idx] as f64;
        assert!(
            approx_eq(cpu_p, gpu_p, tol),
            "predict mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_p,
            gpu_p
        );

        let cpu_t = cpu_trigger_tm[idx];
        let gpu_t = gpu_trigger_tm[idx] as f64;
        assert!(
            approx_eq(cpu_t, gpu_t, tol),
            "trigger mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_t,
            gpu_t
        );
    }

    Ok(())
}
