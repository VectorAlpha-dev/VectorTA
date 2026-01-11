

use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::CudaChop;
use vector_ta::indicators::chop::{ChopBatchBuilder, ChopBatchRange, ChopParams};

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
fn chop_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[chop_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let n = 8192usize;
    let mut h = vec![f64::NAN; n];
    let mut l = vec![f64::NAN; n];
    let mut c = vec![f64::NAN; n];
    for i in 1..n {
        let x = i as f64 * 0.00123;
        let base = x.sin() + 0.0002 * (i as f64);
        let hi = base + 0.6 + 0.05 * (x * 2.1).cos();
        let lo = base - 0.6 - 0.04 * (x * 1.7).sin();
        h[i] = hi;
        l[i] = lo;
        c[i] = (hi + lo) * 0.5;
    }
    let sweep = ChopBatchRange {
        period: (5, 25, 5),
        scalar: (100.0, 100.0, 0.0),
        drift: (1, 3, 1),
    };

    
    let cpu = ChopBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(5, 25, 5)
        .scalar_static(100.0)
        .drift_range(1, 3, 1)
        .apply_slices(&h, &l, &c)?;

    
    let hf: Vec<f32> = h.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = l.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = c.iter().map(|&v| v as f32).collect();
    let cuda = CudaChop::new(0).expect("CudaChop::new");
    let (dev, combos) = cuda
        .chop_batch_dev(&hf, &lf, &cf, &sweep)
        .expect("cuda chop");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    assert_eq!(cpu.combos.len(), combos.len());

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-3; 
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu.values[idx], host[idx] as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu.values[idx],
            host[idx]
        );
    }
    Ok(())
}
