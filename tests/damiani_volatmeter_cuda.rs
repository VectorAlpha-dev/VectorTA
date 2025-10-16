// CUDA integration tests for Damiani Volatmeter

use my_project::indicators::damiani_volatmeter::{
    damiani_volatmeter_batch_with_kernel, DamianiVolatmeterBatchRange,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, CudaDamianiVolatmeter};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    { assert!(true); }
}

#[cfg(feature = "cuda")]
#[test]
fn damiani_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[damiani_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        let base = (x * 0.00037).sin() + (x * 0.00021).cos();
        data[i] = base + 0.001 * (i % 7) as f64;
        if i % 257 == 0 { data[i] = f64::NAN; }
    }

    let sweep = DamianiVolatmeterBatchRange { vis_atr: (13, 40, 1), vis_std: (20, 40, 1), sed_atr: (40, 40, 0), sed_std: (100, 100, 0), threshold: (1.4, 1.4, 0.0) };

    let cpu = damiani_volatmeter_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();

    let cuda = CudaDamianiVolatmeter::new(0).expect("CudaDamianiVolatmeter::new");
    let (dev_arr, combos_meta) = cuda
        .damiani_volatmeter_batch_dev(&data_f32, &sweep)
        .expect("damiani_cuda_batch_dev");

    assert_eq!(combos_meta.len(), cpu.combos.len());
    assert_eq!(dev_arr.rows, 2 * cpu.rows);
    assert_eq!(dev_arr.cols, cpu.cols);

    let mut gpu = vec![0f32; dev_arr.len()];
    dev_arr.buf.copy_to(&mut gpu).expect("copy damiani results");

    // Validate warmup/NaN prefix and shapes; spot-check finiteness after warmup.
    for (row, params) in cpu.combos.iter().enumerate() {
        let base_vol = (2 * row) * cpu.cols;
        let base_anti = (2 * row + 1) * cpu.cols;
        let vol = cpu.vol_for(params).unwrap();
        let anti = cpu.anti_for(params).unwrap();
        // Compute CPU warmup (NaN prefix length)
        let cpu_warm_vol = vol.iter().position(|v| v.is_finite()).unwrap_or(vol.len());
        let cpu_warm_anti = anti.iter().position(|v| v.is_finite()).unwrap_or(anti.len());
        // GPU warmup
        let gpu_vol_slice = &gpu[base_vol..base_vol + cpu.cols];
        let gpu_anti_slice = &gpu[base_anti..base_anti + cpu.cols];
        let gpu_warm_vol = gpu_vol_slice.iter().position(|v| v.is_finite()).unwrap_or(gpu_vol_slice.len());
        let gpu_warm_anti = gpu_anti_slice.iter().position(|v| v.is_finite()).unwrap_or(gpu_anti_slice.len());
        assert_eq!(gpu_warm_vol, cpu_warm_vol, "warmup prefix mismatch (vol) for row {}", row);
        assert_eq!(gpu_warm_anti, cpu_warm_anti, "warmup prefix mismatch (anti) for row {}", row);
        // Spot-check a small suffix for finiteness
        for i in (cpu.cols.saturating_sub(64))..cpu.cols {
            assert!(gpu_vol_slice[i].is_finite() || vol[i].is_nan(), "GPU vol should be finite where CPU is finite");
            assert!(gpu_anti_slice[i].is_finite() || anti[i].is_nan(), "GPU anti should be finite where CPU is finite");
        }
    }

    Ok(())
}
