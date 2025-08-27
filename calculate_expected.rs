use my_project::indicators::moving_averages::srwma::{srwma, SrwmaInput, SrwmaParams};
use my_project::utilities::data_loader::read_candles_from_csv;

fn main() {
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path).unwrap();
    let input = SrwmaInput::from_candles(&candles, "close", SrwmaParams::default());
    let result = srwma(&input).unwrap();
    
    println!("Last 5 SRWMA values with period=14:");
    let start = result.values.len().saturating_sub(5);
    for (i, &val) in result.values[start..].iter().enumerate() {
        println!("  [{}]: {:.11}", i, val);
    }
}