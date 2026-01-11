

use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaRangeFilter;

use vector_ta::indicators::range_filter::{
    range_filter_batch_with_kernel, range_filter_with_kernel, RangeFilterBatchRange,
    RangeFilterInput, RangeFilterParams,
};

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
fn range_filter_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[range_filter_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 20_000usize;
    let mut data = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        data[i] = (x * 0.0019).sin() + 0.00023 * x;
    }
    let sweep = RangeFilterBatchRange {
        range_size: (2.0, 3.0, 0.2),
        range_period: (8, 32, 4),
        smooth_range: Some(true),
        smooth_period: Some(27),
    };

    let cpu = range_filter_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaRangeFilter::new(0).expect("CudaRangeFilter::new");
    let (dev, combos) = cuda
        .range_filter_batch_dev(&data_f32, &sweep)
        .expect("range_filter_batch_dev");

    assert_eq!(cpu.rows, combos.len());
    assert_eq!(cpu.cols, data.len());

    let mut g_f = vec![0f32; dev.len()];
    let mut g_h = vec![0f32; dev.len()];
    let mut g_l = vec![0f32; dev.len()];
    dev.filter.copy_to(&mut g_f)?;
    dev.high.copy_to(&mut g_h)?;
    dev.low.copy_to(&mut g_l)?;

    let tol = 1.5e-3; 
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.filter_values[idx], g_f[idx] as f64, tol),
            "filter mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.high_band_values[idx], g_h[idx] as f64, tol),
            "high mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.low_band_values[idx], g_l[idx] as f64, tol),
            "low mismatch at {}",
            idx
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn range_filter_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[range_filter_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 10usize; 
    let rows = 8_192usize; 
    let mut data_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        for t in s..rows {
            let x = t as f64 + s as f64 * 0.1;
            data_tm[t * cols + s] = (x * 0.0027).sin() + 0.00019 * x;
        }
    }
    let range_size = 2.618f64;
    let range_period = 14usize;
    let smooth_range = true;
    let smooth_period = 27usize;

    
    let mut f_cpu_tm = vec![f64::NAN; rows * cols];
    let mut h_cpu_tm = vec![f64::NAN; rows * cols];
    let mut l_cpu_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = data_tm[t * cols + s];
        }
        let params = RangeFilterParams {
            range_size: Some(range_size),
            range_period: Some(range_period),
            smooth_range: Some(smooth_range),
            smooth_period: Some(smooth_period),
        };
        let input = RangeFilterInput::from_slice(&series, params);
        let out = range_filter_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            let idx = t * cols + s;
            f_cpu_tm[idx] = out.filter[t];
            h_cpu_tm[idx] = out.high_band[t];
            l_cpu_tm[idx] = out.low_band[t];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaRangeFilter::new(0).expect("CudaRangeFilter::new");
    let dev = cuda
        .range_filter_many_series_one_param_time_major_dev(
            &data_tm_f32,
            cols,
            rows,
            &RangeFilterParams {
                range_size: Some(range_size),
                range_period: Some(range_period),
                smooth_range: Some(smooth_range),
                smooth_period: Some(smooth_period),
            },
        )
        .expect("range_filter_many_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut g_f_tm = vec![0f32; dev.len()];
    let mut g_h_tm = vec![0f32; dev.len()];
    let mut g_l_tm = vec![0f32; dev.len()];
    dev.filter.copy_to(&mut g_f_tm)?;
    dev.high.copy_to(&mut g_h_tm)?;
    dev.low.copy_to(&mut g_l_tm)?;

    let tol = 1.5e-3;
    for idx in 0..g_f_tm.len() {
        assert!(
            approx_eq(f_cpu_tm[idx], g_f_tm[idx] as f64, tol),
            "filter mismatch at {}",
            idx
        );
        assert!(
            approx_eq(h_cpu_tm[idx], g_h_tm[idx] as f64, tol),
            "high mismatch at {}",
            idx
        );
        assert!(
            approx_eq(l_cpu_tm[idx], g_l_tm[idx] as f64, tol),
            "low mismatch at {}",
            idx
        );
    }

    Ok(())
}
