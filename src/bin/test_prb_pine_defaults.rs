use my_project::other_indicators::prb::{prb, PrbInput, PrbParams};
use my_project::utilities::data_loader::read_candles_from_csv;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load test data
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;
    
    // Test with exact Pine Script defaults
    println!("Testing PRB with exact Pine Script default parameters:");
    println!("=========================================");
    
    // Pine defaults:
    // use_filt = true (smooth_data)
    // filt_per = 10 (smooth_period)
    // per = 100 (regression_period)
    // order = 2 (polynomial_order)
    // calc_offs = 0 (regression_offset)
    // ndev = 2.0
    // equ_from = 0
    
    let params = PrbParams {
        smooth_data: Some(true),   // Pine default: true
        smooth_period: Some(10),   // Pine default: 10
        regression_period: Some(100),
        polynomial_order: Some(2),
        regression_offset: Some(0),
        ndev: Some(2.0),
        equ_from: Some(0),
    };
    
    let input = PrbInput::from_candles(&candles, "close", params);
    let output = prb(&input)?;
    
    // Get the last 10 non-NaN values
    println!("\nWith smoothing enabled (Pine default):");
    let non_nan_values: Vec<(usize, f64)> = output.values
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, v)| (i, *v))
        .collect();
    
    println!("Last 10 main regression values:");
    let start = non_nan_values.len().saturating_sub(10);
    for (i, val) in &non_nan_values[start..] {
        let close_val = candles.close[*i];
        let diff = val - close_val;
        println!("  Index {}: PRB={:.2}, Close={:.2}, Diff={:.2}", i, val, close_val, diff);
    }
    
    // Now test WITHOUT smoothing
    println!("\n=========================================");
    println!("Without smoothing:");
    
    let params_no_smooth = PrbParams {
        smooth_data: Some(false),  // Disable smoothing
        smooth_period: Some(10),
        regression_period: Some(100),
        polynomial_order: Some(2),
        regression_offset: Some(0),
        ndev: Some(2.0),
        equ_from: Some(0),
    };
    
    let input_no_smooth = PrbInput::from_candles(&candles, "close", params_no_smooth);
    let output_no_smooth = prb(&input_no_smooth)?;
    
    let non_nan_no_smooth: Vec<(usize, f64)> = output_no_smooth.values
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, v)| (i, *v))
        .collect();
    
    println!("Last 10 main regression values:");
    let start = non_nan_no_smooth.len().saturating_sub(10);
    for (i, val) in &non_nan_no_smooth[start..] {
        let close_val = candles.close[*i];
        let diff = val - close_val;
        println!("  Index {}: PRB={:.2}, Close={:.2}, Diff={:.2}", i, val, close_val, diff);
    }
    
    // Compare smoothed vs unsmoothed
    println!("\n=========================================");
    println!("Difference between smoothed and unsmoothed:");
    let start = non_nan_values.len().saturating_sub(5);
    for i in start..non_nan_values.len() {
        let (idx, smoothed) = non_nan_values[i];
        let unsmoothed = non_nan_no_smooth[i].1;
        let diff = smoothed - unsmoothed;
        println!("  Index {}: Smoothed={:.2}, Unsmoothed={:.2}, Diff={:.2}", 
                 idx, smoothed, unsmoothed, diff);
    }
    
    Ok(())
}