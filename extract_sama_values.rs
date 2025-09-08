use my_project::other_indicators::sama::{sama, SamaInput, SamaParams};
use my_project::utilities::data_loader::read_candles_from_csv;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;
    
    let params = SamaParams::default(); // length=200, maj_length=14, min_length=6
    let input = SamaInput::from_candles(&candles, "close", params);
    let output = sama(&input)?;
    
    // Find non-NaN values
    let valid_values: Vec<f64> = output.values.iter()
        .filter(|&&v| !v.is_nan())
        .copied()
        .collect();
    
    println!("Total values: {}", output.values.len());
    println!("Valid values: {}", valid_values.len());
    
    if valid_values.len() >= 5 {
        println!("\nLast 5 valid values:");
        for i in valid_values.len().saturating_sub(5)..valid_values.len() {
            println!("  {:.8}", valid_values[i]);
        }
    }
    
    // Also test with smaller params for more output
    let params2 = SamaParams {
        length: Some(50),
        maj_length: Some(14),
        min_length: Some(6),
    };
    let input2 = SamaInput::from_candles(&candles, "close", params2);
    let output2 = sama(&input2)?;
    
    let valid_values2: Vec<f64> = output2.values.iter()
        .filter(|&&v| !v.is_nan())
        .copied()
        .collect();
    
    println!("\nWith length=50:");
    println!("Valid values: {}", valid_values2.len());
    if valid_values2.len() >= 5 {
        println!("Last 5 valid values:");
        for i in valid_values2.len().saturating_sub(5)..valid_values2.len() {
            println!("  {:.8}", valid_values2[i]);
        }
    }
    
    Ok(())
}