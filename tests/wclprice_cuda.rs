// Integration tests for CUDA WCLPRICE kernel

#[cfg(feature = "cuda")]
use my_project::indicators::wclprice::{wclprice_with_kernel, WclpriceInput};
#[cfg(feature = "cuda")]
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, CudaWclprice};

#[cfg(feature = "cuda")]
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
fn wclprice_cuda_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[wclprice_cuda_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];

    for i in 5..len {
        let x = i as f64;
        let base = (x * 0.002).sin() * 0.5 + 0.001 * x;
        close[i] = base;
        high[i] = base + 0.75;
        low[i] = base - 0.6;
    }

    let input = WclpriceInput::from_slices(&high, &low, &close);
    let cpu = wclprice_with_kernel(&input, Kernel::Scalar)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();

    let cuda = CudaWclprice::new(0)?;
    let gpu_handle = cuda.wclprice_dev(&high_f32, &low_f32, &close_f32)?;

    assert_eq!(gpu_handle.rows, 1);
    assert_eq!(gpu_handle.cols, cpu.values.len());

    let mut gpu = vec![0f32; gpu_handle.len()];
    gpu_handle.buf.copy_to(&mut gpu)?;

    let tol = 1e-5;
    for (idx, (&cpu_val, &gpu_val)) in cpu.values.iter().zip(gpu.iter()).enumerate() {
        assert!(
            approx_eq(cpu_val, gpu_val as f64, tol),
            "Mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    Ok(())
}
