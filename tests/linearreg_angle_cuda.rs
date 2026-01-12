use vector_ta::indicators::linearreg_angle::{
    linearreg_angle_batch_with_kernel, Linearreg_angleBatchRange, Linearreg_angleBuilder,
    Linearreg_angleParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaLinearregAngle;

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
fn linearreg_angle_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[linearreg_angle_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![0.0f64; len];
    for i in 0..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = Linearreg_angleBatchRange { period: (4, 64, 3) };

    let cpu = linearreg_angle_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaLinearregAngle::new(0).expect("CudaLinearregAngle::new");
    let dev = cuda
        .linearreg_angle_batch_dev(&price_f32, &sweep)
        .expect("linearreg_angle_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn linearreg_angle_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[linearreg_angle_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let mut data_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for r in s..rows {
            let x = (r as f64) + (s as f64) * 0.25;
            data_tm[r * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let period = 21usize;

    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for r in 0..rows {
            series[r] = data_tm[r * cols + s];
        }
        let out = Linearreg_angleBuilder::new()
            .period(period)
            .kernel(Kernel::Scalar)
            .apply_slice(&series)?;
        for r in 0..rows {
            cpu_tm[r * cols + s] = out.values[r];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let params = Linearreg_angleParams {
        period: Some(period),
    };
    let cuda = CudaLinearregAngle::new(0).expect("CudaLinearregAngle::new");
    let dev_tm = cuda
        .linearreg_angle_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("linearreg_angle_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);

    let mut g_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut g_tm)?;

    let tol = 5e-4;
    for idx in 0..g_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], g_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
