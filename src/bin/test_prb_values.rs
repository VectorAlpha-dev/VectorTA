use my_project::other_indicators::prb::{prb, PrbInput, PrbParams};
use my_project::utilities::data_loader::read_candles_from_csv;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load test data
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;
    
    // Create PRB input with default parameters
    let params = PrbParams {
        smooth_data: Some(false),  // No smoothing as per user request
        smooth_period: None,
        regression_period: Some(100),  // Default period of 100
        polynomial_order: Some(2),
        regression_offset: Some(0),
        ndev: Some(2.0),
        equ_from: Some(0),
    };
    
    let input = PrbInput::from_candles(&candles, "close", params);
    let output = prb(&input)?;
    
    // Get the last 5 non-NaN values
    println!("PRB Reference Values (no smoothing, period=100):");
    let non_nan_values: Vec<(usize, f64)> = output.values
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, v)| (i, *v))
        .collect();
    
    println!("Last 5 main values:");
    let start = non_nan_values.len().saturating_sub(5);
    for (i, val) in &non_nan_values[start..] {
        println!("  Index {}: {:.8}", i, val);
    }
    
    // Also get upper and lower bands
    let non_nan_upper: Vec<(usize, f64)> = output.upper_band
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, v)| (i, *v))
        .collect();
    
    println!("\nLast 5 upper band values:");
    let start = non_nan_upper.len().saturating_sub(5);
    for (i, val) in &non_nan_upper[start..] {
        println!("  Index {}: {:.8}", i, val);
    }
    
    let non_nan_lower: Vec<(usize, f64)> = output.lower_band
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, v)| (i, *v))
        .collect();
    
    println!("\nLast 5 lower band values:");
    let start = non_nan_lower.len().saturating_sub(5);
    for (i, val) in &non_nan_lower[start..] {
        println!("  Index {}: {:.8}", i, val);
    }
    
    Ok(())
}