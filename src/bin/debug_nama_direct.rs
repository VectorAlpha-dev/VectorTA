use my_project::indicators::moving_averages::nama::{nama, nama_into_slice, NamaInput, NamaParams};
use my_project::utilities::data_loader::read_candles_from_csv;
use my_project::utilities::enums::Kernel;

fn main() {
    // Load CSV data
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path).unwrap();
    
    println!("Loaded {} candles", candles.close.len());
    println!("Last 5 close prices: {:?}", &candles.close[candles.close.len()-5..]);
    
    // Test 1: Using nama function
    let params = NamaParams { period: Some(30) };
    let input1 = NamaInput::from_candles(&candles, "close", params);
    let result1 = nama(&input1).unwrap();
    
    println!("\nTest 1 - nama() function:");
    println!("Last 5 values:");
    for i in (candles.close.len()-5)..candles.close.len() {
        println!("  [{}]: {:.8}", i, result1.values[i]);
    }
    
    // Test 2: Using nama_into_slice (what Python uses)
    let mut output = vec![0.0; candles.close.len()];
    let input2 = NamaInput::from_slice(&candles.close, params);
    nama_into_slice(&mut output, &input2, Kernel::Auto).unwrap();
    
    println!("\nTest 2 - nama_into_slice() function:");
    println!("Last 5 values:");
    for i in (candles.close.len()-5)..candles.close.len() {
        println!("  [{}]: {:.8}", i, output[i]);
    }
    
    // Compare
    println!("\nComparison:");
    for i in (candles.close.len()-5)..candles.close.len() {
        let diff = result1.values[i] - output[i];
        if diff.abs() > 1e-10 {
            println!("  [{}]: DIFFERENT! {:.8} vs {:.8}, diff: {:.8}", 
                     i, result1.values[i], output[i], diff);
        } else {
            println!("  [{}]: Same", i);
        }
    }
}