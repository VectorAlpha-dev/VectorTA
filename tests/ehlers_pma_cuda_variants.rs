

use vector_ta::indicators::moving_averages::ehlers_pma::{
    ehlers_pma, EhlersPmaBatchRange, EhlersPmaInput, EhlersPmaParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::ehlers_pma_wrapper::{
    BatchKernelPolicy, BatchThreadsPerOutput, CudaEhlersPma, CudaEhlersPmaPolicy,
    ManySeriesKernelPolicy,
};

fn approx_eq(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    diff <= atol + rtol * a.abs().max(b.abs())
}

#[test]
fn cuda_feature_off_compiles() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
fn gen_series_f64(len: usize, nan_lead: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; len];
    for i in nan_lead..len {
        let x = i as f64;
        v[i] = (x * 0.0013).sin() + (x * 0.0007).cos() * 0.25 + 0.0002 * x;
    }
    v
}

#[cfg(feature = "cuda")]
fn gen_time_major_f64(cols: usize, rows: usize) -> Vec<f64> {
    let mut v = vec![f64::NAN; cols * rows];
    for j in 0..cols {
        let start = (j % 6) as usize;
        for t in start..rows {
            let x = t as f64 + j as f64 * 0.37;
            v[t * cols + j] = (x * 0.0016).cos() + (x * 0.0011).sin() * 0.4 + 0.0003 * x;
        }
    }
    v
}

#[cfg(feature = "cuda")]
fn compare_batch(policy: CudaEhlersPmaPolicy, series_len: usize, combos: usize) {
    if !cuda_available() {
        eprintln!("[ehlers_pma.compare_batch] skipped - no CUDA device");
        return;
    }

    let data = gen_series_f64(series_len, 7);
    let input = EhlersPmaInput::from_slice(&data, EhlersPmaParams);
    let cpu = ehlers_pma(&input).expect("cpu ehlers_pma");

    let sweep = EhlersPmaBatchRange { combos };
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();

    let mut cuda = CudaEhlersPma::new_with_policy(0, policy).expect("cuda pma");
    let pair = cuda
        .ehlers_pma_batch_dev(&data_f32, &sweep)
        .expect("gpu pma batch");
    assert_eq!(pair.rows(), combos);
    assert_eq!(pair.cols(), series_len);

    let mut gpu_predict = vec![0f32; pair.predict.len()];
    let mut gpu_trigger = vec![0f32; pair.trigger.len()];
    pair.predict.buf.copy_to(&mut gpu_predict).unwrap();
    pair.trigger.buf.copy_to(&mut gpu_trigger).unwrap();

    let (atol, rtol) = (2e-5, 2e-5);
    for combo in 0..combos {
        for idx in 0..series_len {
            let a = cpu.predict[idx];
            let b = gpu_predict[combo * series_len + idx] as f64;
            assert!(
                approx_eq(a, b, atol, rtol),
                "predict mismatch at ({},{})",
                combo,
                idx
            );
            let a2 = cpu.trigger[idx];
            let b2 = gpu_trigger[combo * series_len + idx] as f64;
            assert!(
                approx_eq(a2, b2, atol, rtol),
                "trigger mismatch at ({},{})",
                combo,
                idx
            );
        }
    }
}

#[cfg(feature = "cuda")]
fn compare_many_series(policy: CudaEhlersPmaPolicy, cols: usize, rows: usize) {
    if !cuda_available() {
        eprintln!("[ehlers_pma.compare_many_series] skipped - no CUDA device");
        return;
    }
    let tm = gen_time_major_f64(cols, rows);
    
    let mut cpu_predict_tm = vec![f64::NAN; cols * rows];
    let mut cpu_trigger_tm = vec![f64::NAN; cols * rows];
    for j in 0..cols {
        let mut s = vec![f64::NAN; rows];
        for t in 0..rows {
            s[t] = tm[t * cols + j];
        }
        let input = EhlersPmaInput::from_slice(&s, EhlersPmaParams);
        let out = ehlers_pma(&input).expect("cpu ehlers_pma per series");
        for t in 0..rows {
            cpu_predict_tm[t * cols + j] = out.predict[t];
            cpu_trigger_tm[t * cols + j] = out.trigger[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let mut cuda = CudaEhlersPma::new_with_policy(0, policy).expect("cuda pma");
    let pair = cuda
        .ehlers_pma_many_series_one_param_time_major_dev(&tm_f32, cols, rows)
        .expect("gpu pma many-series");
    assert_eq!(pair.rows(), rows);
    assert_eq!(pair.cols(), cols);
    let mut gpu_predict_tm = vec![0f32; pair.predict.len()];
    let mut gpu_trigger_tm = vec![0f32; pair.trigger.len()];
    pair.predict.buf.copy_to(&mut gpu_predict_tm).unwrap();
    pair.trigger.buf.copy_to(&mut gpu_trigger_tm).unwrap();

    let (atol, rtol) = (2e-5, 2e-5);
    for i in 0..gpu_predict_tm.len() {
        let a = cpu_predict_tm[i];
        let b = gpu_predict_tm[i] as f64;
        assert!(
            approx_eq(a, b, atol, rtol),
            "many-series predict mismatch at {}",
            i
        );
        let a2 = cpu_trigger_tm[i];
        let b2 = gpu_trigger_tm[i] as f64;
        assert!(
            approx_eq(a2, b2, atol, rtol),
            "many-series trigger mismatch at {}",
            i
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn ehlers_pma_batch_plain_matches_cpu() {
    let policy = CudaEhlersPmaPolicy {
        batch: BatchKernelPolicy::Plain { block_x: 256 },
        many_series: ManySeriesKernelPolicy::Auto,
    };
    compare_batch(policy, 32768, 3);
}

#[cfg(feature = "cuda")]
#[test]
fn ehlers_pma_batch_tiled_matches_cpu() {
    let policy = CudaEhlersPmaPolicy {
        batch: BatchKernelPolicy::Tiled {
            tile: 128,
            per_thread: BatchThreadsPerOutput::One,
        },
        many_series: ManySeriesKernelPolicy::Auto,
    };
    compare_batch(policy, 65536, 5);
}

#[cfg(feature = "cuda")]
#[test]
fn ehlers_pma_many_series_1d_matches_cpu() {
    let policy = CudaEhlersPmaPolicy {
        batch: BatchKernelPolicy::Auto,
        many_series: ManySeriesKernelPolicy::OneD { block_x: 1 },
    };
    compare_many_series(policy, 8, 16384);
}

#[cfg(feature = "cuda")]
#[test]
fn ehlers_pma_many_series_2d_ty4_matches_cpu() {
    let policy = CudaEhlersPmaPolicy {
        batch: BatchKernelPolicy::Auto,
        many_series: ManySeriesKernelPolicy::Tiled2D { tx: 1, ty: 4 },
    };
    compare_many_series(policy, 32, 4096);
}

#[cfg(feature = "cuda")]
#[test]
fn ehlers_pma_many_series_2d_ty2_matches_cpu() {
    let policy = CudaEhlersPmaPolicy {
        batch: BatchKernelPolicy::Auto,
        many_series: ManySeriesKernelPolicy::Tiled2D { tx: 1, ty: 2 },
    };
    compare_many_series(policy, 18, 4096);
}
