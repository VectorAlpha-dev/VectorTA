use my_project::indicators::moving_averages::trima::{trima, TrimaInput, TrimaParams};
use my_project::utilities::data_loader::read_candles_from_csv;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path)?;

    // First pass with period=30 (matching test_trima.py)
    let first_params = TrimaParams { period: Some(30) };
    let first_input = TrimaInput::from_candles(&candles, "close", first_params);
    let first_result = trima(&first_input)?;

    // Second pass with period=10
    let second_params = TrimaParams { period: Some(10) };
    let second_input = TrimaInput::from_slice(&first_result.values, second_params);
    let second_result = trima(&second_input)?;

    println!("Reinput last 5 values (period=30 then period=10):");
    let len = second_result.values.len();
    for i in (len - 5)..len {
        println!("  {}", second_result.values[i]);
    }

    Ok(())
}