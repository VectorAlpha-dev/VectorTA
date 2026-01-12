use vector_ta::indicators::aroonosc::{
    aroon_osc_with_kernel, AroonOscBatchBuilder, AroonOscBatchRange, AroonOscData, AroonOscInput,
    AroonOscParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::CudaAroonOsc;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn aroonosc_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[aroonosc_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 8192usize;
    let mut high = vec![f64::NAN; series_len];
    let mut low = vec![f64::NAN; series_len];
    for i in 20..series_len {
        let x = i as f64;
        let base = (x * 0.002).sin() + 0.0008 * x;
        high[i] = base + 0.9;
        low[i] = base - 0.7;
    }

    let sweep = AroonOscBatchRange { length: (9, 60, 5) };

    let cpu = AroonOscBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .length_range(sweep.length.0, sweep.length.1, sweep.length.2)
        .apply_slices(&high, &low)?;

    let cuda = CudaAroonOsc::new(0).expect("CudaAroonOsc::new");
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();

    let gpu_handle = cuda
        .aroonosc_batch_dev(&high_f32, &low_f32, &sweep)
        .expect("cuda aroonosc_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle.buf.copy_to(&mut gpu_host).unwrap();

    let tol = 7e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let cpu_val = cpu.values[idx];
        let gpu_val = gpu_host[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn aroonosc_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[aroonosc_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let rows = 48usize;
    let cols = 2048usize;
    let length = 14usize;
    let mut high_tm = vec![f32::NAN; rows * cols];
    let mut low_tm = vec![f32::NAN; rows * cols];

    for s in 0..rows {
        for t in 20..cols {
            let x = (t as f64) + (s as f64) * 0.013;
            let base = (x * 0.002).sin() + 0.0009 * x;
            let h = (base + 0.85) as f32;
            let l = (base - 0.65) as f32;
            high_tm[t * rows + s] = h;
            low_tm[t * rows + s] = l;
        }
    }

    let cuda = CudaAroonOsc::new(0).expect("CudaAroonOsc::new");
    let handle = cuda
        .aroonosc_many_series_one_param_time_major_dev(&high_tm, &low_tm, cols, rows, length)
        .expect("aroonosc many-series");
    assert_eq!(handle.rows, rows);
    assert_eq!(handle.cols, cols);

    let mut gpu_host = vec![0f32; handle.len()];
    handle.buf.copy_to(&mut gpu_host).unwrap();

    let tol = 7e-4;
    for &s in &[0usize, 7, 17, 31, 47] {
        let mut high = vec![f64::NAN; cols];
        let mut low = vec![f64::NAN; cols];
        for t in 0..cols {
            let idx = t * rows + s;
            high[t] = high_tm[idx] as f64;
            low[t] = low_tm[idx] as f64;
        }
        let inp = AroonOscInput {
            data: AroonOscData::SlicesHL {
                high: &high,
                low: &low,
            },
            params: AroonOscParams {
                length: Some(length),
            },
        };
        let cpu = aroon_osc_with_kernel(&inp, Kernel::Scalar)?;
        for t in 0..cols {
            let idx = t * rows + s;
            assert!(
                approx_eq(cpu.values[t], gpu_host[idx] as f64, tol),
                "series {} [{}]",
                s,
                t
            );
        }
    }

    Ok(())
}
