use my_project::indicators::wad::{wad, WadInput};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, CudaWad};

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
fn build_series(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut high = vec![0.0; len];
    let mut low = vec![0.0; len];
    let mut close = vec![0.0; len];

    let mut price = 100.0f64;
    close[0] = price;
    high[0] = price + 0.6;
    low[0] = price - 0.6;

    for i in 1..len {
        let t = i as f64;
        let delta = (t * 0.0041).sin() * 0.7 + (t * 0.0023).cos() * 0.4;
        let c = price + delta;
        close[i] = c;
        high[i] = c + 0.65 + 0.05 * (i % 5) as f64;
        low[i] = c - 0.64 - 0.04 * (i % 3) as f64;
        price = c;
    }

    (high, low, close)
}

#[cfg(feature = "cuda")]
#[test]
fn wad_cuda_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[wad_cuda_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let (high, low, close) = build_series(len);

    let input = WadInput::from_slices(&high, &low, &close);
    let cpu = wad(&input)?.values;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();

    let cuda = CudaWad::new(0).expect("CudaWad::new");
    let dev = cuda
        .wad_series_dev(&high_f32, &low_f32, &close_f32)
        .expect("wad_series_dev");

    assert_eq!(dev.rows, 1);
    assert_eq!(dev.cols, len);

    let mut gpu = vec![0f32; len];
    dev.buf.copy_to(&mut gpu).expect("copy wad cuda results");

    let tol = 1e-3;
    for (idx, (&cpu_v, &gpu_v)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            approx_eq(cpu_v, gpu_v as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_v,
            gpu_v
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn wad_cuda_into_host_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[wad_cuda_into_host_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let (high, low, close) = build_series(len);

    let input = WadInput::from_slices(&high, &low, &close);
    let cpu = wad(&input)?.values;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();

    let mut gpu = vec![0f32; len];
    let cuda = CudaWad::new(0).expect("CudaWad::new");
    let written = cuda
        .wad_into_host_f32(&high_f32, &low_f32, &close_f32, &mut gpu)
        .expect("wad_into_host_f32");
    assert_eq!(written, len);

    let tol = 1e-3;
    for (idx, (&cpu_v, &gpu_v)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            approx_eq(cpu_v, gpu_v as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_v,
            gpu_v
        );
    }

    Ok(())
}
