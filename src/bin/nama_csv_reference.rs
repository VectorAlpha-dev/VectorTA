use my_project::indicators::moving_averages::nama::{nama, NamaInput, NamaParams};
use my_project::utilities::data_loader::read_candles_from_csv;

fn main() {
    // Load the same CSV data used in tests
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path).unwrap();
    
    println!("CSV data loaded: {} candles", candles.close.len());
    println!("Last 5 close prices:");
    let len = candles.close.len();
    for i in (len-5)..len {
        println!("  [{}]: {}", i, candles.close[i]);
    }
    
    // Calculate NAMA with default period=30
    let params = NamaParams { period: Some(30) };
    let input = NamaInput::from_candles(&candles, "close", params);
    let result = nama(&input).unwrap();
    
    println!("\nNAMA (period=30) last 5 values:");
    for i in (len-5)..len {
        if result.values[i].is_nan() {
            println!("  [{}]: NaN", i);
        } else {
            println!("  [{}]: {:.8}", i, result.values[i]);
        }
    }
    
    // Also show the exact last 5 values for comparison with user's values
    println!("\nLast 5 NAMA values (formatted for comparison):");
    let last_5: Vec<String> = result.values[(len-5)..len]
        .iter()
        .map(|v| if v.is_nan() { 
            "NaN".to_string() 
        } else { 
            format!("{:.8}", v) 
        })
        .collect();
    println!("{}", last_5.join(", "));
    
    // User's provided values for comparison:
    println!("\nUser's reference values:");
    println!("59,309.14340744, 59,304.88975909, 59,283.51109653, 59,243.52850894, 59,228.86200178");
}