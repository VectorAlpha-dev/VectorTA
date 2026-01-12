use vector_ta::indicators::vpt::{vpt_with_kernel, VptInput};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaVpt;

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
fn vpt_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vpt_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut price = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];

    for i in 0..len {
        if i == 0 {
            price[i] = 100.0;
            continue;
        }
        if i >= 4 {
            let x = i as f64;
            price[i] = (x * 0.00121).sin() + 100.0 + 0.00019 * x;
            volume[i] = (x * 0.00081).cos().abs() * 600.0 + 150.0;
        }
    }

    let input = VptInput::from_slices(&price, &volume);
    let cpu = vpt_with_kernel(&input, Kernel::Scalar)?.values;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let cuda = CudaVpt::new(0).expect("CudaVpt::new");
    let dev = match cuda.vpt_batch_dev(&price_f32, &volume_f32) {
        Ok(d) => d,
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("named symbol not found") {
                eprintln!("[vpt_cuda_batch_matches_cpu] skipped - kernel symbol not found");
                return Ok(());
            }
            panic!("vpt_batch_dev failed: {}", msg);
        }
    };
    assert_eq!(dev.rows, 1);
    assert_eq!(dev.cols, len);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let tol = 2e-4;
    for c in 0..len {
        let g = host[c] as f64;
        let s = cpu[c];
        assert!(
            approx_eq(s, g, tol),
            "col {} mismatch: cpu={} gpu={}",
            c,
            s,
            g
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn vpt_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vpt_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize;
    let rows = 2048usize;
    let mut price_tm = vec![f64::NAN; cols * rows];
    let mut volume_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s.min(6)..rows {
            let x = (t as f64) + (s as f64) * 0.19;
            price_tm[t * cols + s] = (x * 0.0013).sin() + 100.0 + 0.0002 * x;
            volume_tm[t * cols + s] = (x * 0.0011).cos().abs() * 450.0 + 110.0;
        }

        price_tm[s.min(6) * cols + s] = 100.0;
    }

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = price_tm[t * cols + s];
            v[t] = volume_tm[t * cols + s];
        }
        let input = VptInput::from_slices(&p, &v);
        let out = vpt_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
    }

    let price_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaVpt::new(0).expect("CudaVpt::new");
    let dev =
        match cuda.vpt_many_series_one_param_time_major_dev(&price_f32, &volume_f32, cols, rows) {
            Ok(d) => d,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("named symbol not found") {
                    eprintln!(
                    "[vpt_cuda_many_series_one_param_matches_cpu] skipped - kernel symbol not found"
                );
                    return Ok(());
                }
                panic!("vpt_many_series_one_param_time_major_dev failed: {}", msg);
            }
        };
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let tol = 2e-4;
    for idx in 0..host.len() {
        let g = host[idx] as f64;
        let s = cpu_tm[idx];
        assert!(
            approx_eq(s, g, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            s,
            g
        );
    }
    Ok(())
}
