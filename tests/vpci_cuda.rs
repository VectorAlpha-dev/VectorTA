

use vector_ta::indicators::vpci::{
    vpci_batch_with_kernel, vpci_with_kernel, VpciBatchRange, VpciInput, VpciParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::vpci_wrapper::CudaVpci;

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
fn vpci_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vpci_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 32_768usize;
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        close[i] = (x * 0.00091).sin() * 1.5 + 0.001 * x;
        volume[i] = ((x * 0.00037).cos().abs() + 0.1) * 1000.0;
    }
    let sweep = VpciBatchRange {
        short_range: (5, 13, 2),
        long_range: (16, 40, 4),
    };

    let cpu = vpci_batch_with_kernel(&close, &volume, &sweep, Kernel::ScalarBatch)?;

    let c32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let v32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let cuda = CudaVpci::new(0).expect("CudaVpci::new");
    let (pair, combos) = cuda
        .vpci_batch_dev(&c32, &v32, &sweep)
        .expect("vpci_batch_dev");

    assert_eq!(cpu.rows, combos.len());
    assert_eq!(cpu.rows, pair.rows());
    assert_eq!(cpu.cols, pair.cols());

    let mut vpci_g = vec![0f32; pair.a.len()];
    let mut vpcis_g = vec![0f32; pair.b.len()];
    pair.a.buf.copy_to(&mut vpci_g)?;
    pair.b.buf.copy_to(&mut vpcis_g)?;

    let tol = 5e-2; 
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.vpci[idx], vpci_g[idx] as f64, tol),
            "vpci mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.vpcis[idx], vpcis_g[idx] as f64, tol),
            "vpcis mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn vpci_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vpci_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 12usize; 
    let rows = 8192usize; 
    let mut close_tm = vec![f64::NAN; rows * cols];
    let mut volume_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        for r in (s + 10)..rows {
            
            let idx = r * cols + s;
            let x = (r as f64) * 0.002 + (s as f64) * 0.01;
            close_tm[idx] = (x).sin() + 0.01 * x;
            volume_tm[idx] = (x * 0.33).cos().abs() * 800.0 + 50.0;
        }
    }
    let short = 7usize;
    let long = 24usize;

    
    let mut vpci_cpu_tm = vec![f64::NAN; rows * cols];
    let mut vpcis_cpu_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        let mut c = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for r in 0..rows {
            let idx = r * cols + s;
            c[r] = close_tm[idx];
            v[r] = volume_tm[idx];
        }
        let params = VpciParams {
            short_range: Some(short),
            long_range: Some(long),
        };
        let input = VpciInput::from_slices(&c, &v, params);
        let out = vpci_with_kernel(&input, Kernel::Scalar)?;
        for r in 0..rows {
            let idx = r * cols + s;
            vpci_cpu_tm[idx] = out.vpci[r];
            vpcis_cpu_tm[idx] = out.vpcis[r];
        }
    }

    let c32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let v32: Vec<f32> = volume_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaVpci::new(0).expect("CudaVpci::new");
    let pair = cuda
        .vpci_many_series_one_param_time_major_dev(
            &c32,
            &v32,
            cols,
            rows,
            &VpciParams {
                short_range: Some(short),
                long_range: Some(long),
            },
        )
        .expect("vpci_many_series_one_param_time_major_dev");

    assert_eq!(pair.rows(), rows);
    assert_eq!(pair.cols(), cols);

    let mut vpci_g = vec![0f32; rows * cols];
    let mut vpcis_g = vec![0f32; rows * cols];
    pair.a.buf.copy_to(&mut vpci_g)?;
    pair.b.buf.copy_to(&mut vpcis_g)?;

    let tol = 5e-2;
    for idx in 0..(rows * cols) {
        assert!(
            approx_eq(vpci_cpu_tm[idx], vpci_g[idx] as f64, tol),
            "vpci mismatch at {}",
            idx
        );
        assert!(
            approx_eq(vpcis_cpu_tm[idx], vpcis_g[idx] as f64, tol),
            "vpcis mismatch at {}",
            idx
        );
    }
    Ok(())
}
