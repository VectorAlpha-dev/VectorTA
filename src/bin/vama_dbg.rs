use my_project::indicators::moving_averages::volatility_adjusted_ma::{vama_with_kernel, VamaInput, VamaParams};
use my_project::indicators::moving_averages::volume_adjusted_ma::{
    VolumeAdjustedMaBatchRange, VolumeAdjustedMa_batch_with_kernel,
};
use my_project::utilities::enums::Kernel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        data[i] = (x * 0.00123).sin() + 0.00017 * x;
    }

    let base_period = 9usize;
    let vol_period = 5usize;
    let params = VamaParams {
        base_period: Some(base_period),
        vol_period: Some(vol_period),
        smoothing: Some(false),
        smooth_type: Some(3),
        smooth_period: Some(5),
    };

    let idx = 2562usize;

    let out_f64 = vama_with_kernel(&VamaInput::from_slice(&data, params.clone()), Kernel::Scalar)?;
    let data32_as_f64: Vec<f64> = data.iter().map(|&v| (v as f32) as f64).collect();
    let out_f32 = vama_with_kernel(&VamaInput::from_slice(&data32_as_f64, params), Kernel::Scalar)?;

    eprintln!(
        "[vama] idx={idx} cpu(f64)={} cpu(f32-rounded)={} diff={}",
        out_f64.values[idx],
        out_f32.values[idx],
        out_f32.values[idx] - out_f64.values[idx]
    );

    // ---- VolumeAdjustedMa batch rounding check (matches tests/volume_adjusted_ma_cuda.rs) ----
    let series_len = 4096usize;
    let mut prices = vec![f64::NAN; series_len];
    let mut volumes = vec![f64::NAN; series_len];
    for i in 12..series_len {
        let x = i as f64;
        prices[i] = (x * 0.004).sin() + 0.0004 * x;
        volumes[i] = ((x * 0.006).cos().abs() + 1.5) * 750.0;
    }
    let sweep = VolumeAdjustedMaBatchRange {
        length: (5, 21, 4),
        vi_factor: (0.45, 1.05, 0.2),
        sample_period: (0, 12, 4),
        strict: None,
    };
    let cpu64 = VolumeAdjustedMa_batch_with_kernel(&prices, &volumes, &sweep, Kernel::ScalarBatch)?;
    let prices32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
    let volumes32: Vec<f32> = volumes.iter().map(|&v| v as f32).collect();
    let prices32_as_f64: Vec<f64> = prices32.iter().map(|&v| v as f64).collect();
    let volumes32_as_f64: Vec<f64> = volumes32.iter().map(|&v| v as f64).collect();
    let cpu32 = VolumeAdjustedMa_batch_with_kernel(
        &prices32_as_f64,
        &volumes32_as_f64,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let idx2 = 157540usize;
    eprintln!(
        "[volume_adjusted_ma] idx={idx2} cpu(f64)={} cpu(f32-rounded)={} diff={}",
        cpu64.values[idx2],
        cpu32.values[idx2],
        cpu32.values[idx2] - cpu64.values[idx2]
    );

    Ok(())
}
