use my_project::indicators::er::{er, ErInput, ErParams};
use my_project::utilities::data_loader::read_candles_from_csv;

fn main() {
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path).expect("Failed to load data");
    
    // Default parameters for ER
    let params = ErParams { period: Some(5) };
    let input = ErInput::from_candles(&candles, "close", params);
    let output = er(&input).expect("ER calculation failed");
    
    let len = output.values.len();
    println!("Total length: {}", len);
    
    // Print last 5 values
    println!("\nLast 5 values:");
    for i in (len-5)..len {
        println!("  {}", output.values[i]);
    }
    
    // Print some middle values for secondary validation
    println!("\nValues at indices 100-104:");
    for i in 100..105 {
        if i < len {
            println!("  [{}]: {}", i, output.values[i]);
        }
    }
    
    // Test with trending data
    let trending = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let params = ErParams { period: Some(5) };
    let input = ErInput::from_slice(&trending, params);
    let output = er(&input).expect("ER calculation failed");
    
    println!("\nTrending data ER values:");
    for (i, val) in output.values.iter().enumerate() {
        println!("  [{}]: {}", i, val);
    }
    
    // Test with choppy data  
    let choppy = vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0, 5.0, 9.0];
    let params2 = ErParams { period: Some(5) };
    let input = ErInput::from_slice(&choppy, params2);
    let output = er(&input).expect("ER calculation failed");
    
    println!("\nChoppy data ER values:");
    for (i, val) in output.values.iter().enumerate() {
        println!("  [{}]: {}", i, val);
    }
}