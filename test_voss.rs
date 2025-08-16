use my_project::indicators::voss::{voss, VossInput, VossParams};
use my_project::utilities::data_loader::read_candles_from_csv;

fn main() {
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path).unwrap();
    
    let params = VossParams {
        period: Some(20),
        predict: Some(3),
        bandwidth: Some(0.25),
    };
    let input = VossInput::from_candles(&candles, "close", params);
    let output = voss(&input).unwrap();
    
    let len = output.voss.len();
    println!("Total length: {}", len);
    println!("Last 10 VOSS values:");
    for i in (len-10)..len {
        println!("  [{}]: voss={}, filt={}", i, output.voss[i], output.filt[i]);
    }
    
    // Check for first non-NaN
    for (i, val) in output.voss.iter().enumerate() {
        if !val.is_nan() {
            println!("First non-NaN voss at index {}: {}", i, val);
            break;
        }
    }
}
