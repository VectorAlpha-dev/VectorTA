use my_project::other_indicators::ehlers_pma::{ehlers_pma, EhlersPmaInput, EhlersPmaParams};
use my_project::utilities::data_loader::read_candles_from_csv;

fn main() {
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path).unwrap();
    println!("Loaded {} candles", candles.close.len());
    
    let input = EhlersPmaInput::from_candles(&candles, "close", EhlersPmaParams::default());
    let out = ehlers_pma(&input).unwrap();
    
    println!("Last 10 close prices:");
    for i in (candles.close.len() - 10)..candles.close.len() {
        println!("  [{}]: {}", i, candles.close[i]);
    }
    
    println!("\nLast 10 predict values:");
    for i in (out.predict.len() - 10)..out.predict.len() {
        println!("  [{}]: {}", i, out.predict[i]);
    }
    
    println!("\nLast 5 predict values (what test checks):");
    let start = out.predict.len().saturating_sub(5);
    for (i, val) in out.predict[start..].iter().enumerate() {
        println!("  [{}]: {}", i, val);
    }
    
    println!("\nLast 5 trigger values (what test checks):");
    for (i, val) in out.trigger[start..].iter().enumerate() {
        println!("  [{}]: {}", i, val);
    }
}