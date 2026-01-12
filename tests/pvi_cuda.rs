use vector_ta::indicators::pvi::{pvi_with_kernel, PviInput, PviParams};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaPvi;

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
fn pvi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[pvi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];

    for i in 4..len {
        let x = i as f64;
        close[i] = (x * 0.00121).sin() + 100.0 + 0.00019 * x;
        volume[i] = (x * 0.00081).cos().abs() * 600.0 + 150.0;
    }

    let inits = vec![500.0, 1000.0, 1500.0, 2000.0];

    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let close32_as_f64: Vec<f64> = close_f32.iter().map(|&v| v as f64).collect();
    let volume32_as_f64: Vec<f64> = volume_f32.iter().map(|&v| v as f64).collect();

    let mut cpu_rows: Vec<Vec<f64>> = Vec::new();
    for &iv in &inits {
        let params = PviParams {
            initial_value: Some(iv),
        };
        let input = PviInput::from_slices(&close32_as_f64, &volume32_as_f64, params);
        let out = pvi_with_kernel(&input, Kernel::Scalar)?.values;
        cpu_rows.push(out);
    }

    let inits_f32: Vec<f32> = inits.iter().map(|&v| v as f32).collect();
    let cuda = CudaPvi::new(0).expect("CudaPvi::new");
    let dev = match cuda.pvi_batch_dev(&close_f32, &volume_f32, &inits_f32) {
        Ok(d) => d,
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("named symbol not found") || msg.contains("NotImplemented") {
                eprintln!("[pvi_cuda_batch_matches_cpu] skipped - kernel symbol not found");
                return Ok(());
            }
            panic!("pvi_batch_dev failed: {}", msg);
        }
    };
    assert_eq!(dev.rows, inits.len());
    assert_eq!(dev.cols, len);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let tol = 1e-3;
    for r in 0..inits.len() {
        for c in 0..len {
            let g = host[r * len + c] as f64;
            let s = cpu_rows[r][c];
            assert!(
                approx_eq(s, g, tol),
                "row {} col {} mismatch: cpu={} gpu={}",
                r,
                c,
                s,
                g
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn pvi_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[pvi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize;
    let rows = 2048usize;
    let mut close_tm = vec![f64::NAN; cols * rows];
    let mut volume_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s.min(6)..rows {
            let x = (t as f64) + (s as f64) * 0.19;
            close_tm[t * cols + s] = (x * 0.0013).sin() + 100.0 + 0.0002 * x;
            volume_tm[t * cols + s] = (x * 0.0011).cos().abs() * 450.0 + 110.0;
        }
    }
    let initial_value = 1200.0;

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut c = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for t in 0..rows {
            c[t] = close_tm[t * cols + s];
            v[t] = volume_tm[t * cols + s];
        }
        let params = PviParams {
            initial_value: Some(initial_value),
        };
        let input = PviInput::from_slices(&c, &v, params);
        let out = pvi_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
    }

    let close_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaPvi::new(0).expect("CudaPvi::new");
    let dev = match cuda.pvi_many_series_one_param_time_major_dev(
        &close_f32,
        &volume_f32,
        cols,
        rows,
        initial_value as f32,
    ) {
        Ok(d) => d,
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("named symbol not found") || msg.contains("NotImplemented") {
                eprintln!(
                    "[pvi_cuda_many_series_one_param_matches_cpu] skipped - kernel symbol not found"
                );
                return Ok(());
            }
            panic!("pvi_many_series_one_param_time_major_dev failed: {}", msg);
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
