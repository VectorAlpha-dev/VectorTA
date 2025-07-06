use my_project::indicators::moving_averages::fwma::{FwmaInput, FwmaParams, fwma};
use my_project::utilities::data_loader::read_candles_from_csv;

fn main() {
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path).unwrap();
    
    println!("Candle data info:");
    println!("Total candles: {}", candles.close.len());
    println!("Last 10 close prices:");
    let close_len = candles.close.len();
    for i in (close_len - 10)..close_len {
        println!("  [{}]: {}", i, candles.close[i]);
    }
    
    // Run FWMA with default params (period = 5)
    let params = FwmaParams { period: None }; // Default is 5
    let input = FwmaInput::from_candles(&candles, "close", params);
    let result = fwma(&input).unwrap();
    
    println!("\nFWMA results (period = 5):");
    println!("Last 5 values:");
    let result_len = result.values.len();
    for i in (result_len - 5)..result_len {
        println!("  [{}]: {:.12}", i - (result_len - 5), result.values[i]);
    }
    
    // Also check what the expected values should be
    println!("\nExpected values from test:");
    let expected = [
        59273.583333333336,
        59252.5,
        59167.083333333336,
        59151.0,
        58940.333333333336,
    ];
    for (i, val) in expected.iter().enumerate() {
        println!("  [{}]: {:.12}", i, val);
    }
}