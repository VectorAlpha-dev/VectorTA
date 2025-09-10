use my_project::other_indicators::tradjema::{tradjema, TradjemaInput, TradjemaParams};

fn main() {
    // Sample OHLC data
    let high = vec![
        100.5, 101.2, 100.8, 101.5, 102.3, 101.8, 100.9, 101.6, 102.1, 101.4,
        100.7, 101.3, 102.0, 101.1, 100.6, 101.9, 102.4, 101.0, 100.4, 101.7,
        102.2, 100.3, 101.8, 102.5, 101.2, 100.8, 102.0, 101.5, 100.9, 102.3,
        101.1, 100.5, 102.1, 101.6, 100.7, 102.4, 101.3, 100.2, 101.9, 102.6,
        100.1, 101.4, 102.2, 100.6, 101.0,
    ];
    
    let low = vec![
        99.5, 100.2, 99.8, 100.5, 101.3, 100.8, 99.9, 100.6, 101.1, 100.4,
        99.7, 100.3, 101.0, 100.1, 99.6, 100.9, 101.4, 100.0, 99.4, 100.7,
        101.2, 99.3, 100.8, 101.5, 100.2, 99.8, 101.0, 100.5, 99.9, 101.3,
        100.1, 99.5, 101.1, 100.6, 99.7, 101.4, 100.3, 99.2, 100.9, 101.6,
        99.1, 100.4, 101.2, 99.6, 100.0,
    ];
    
    let close = vec![
        100.0, 100.7, 100.3, 101.0, 101.8, 101.3, 100.4, 101.1, 101.6, 100.9,
        100.2, 100.8, 101.5, 100.6, 100.1, 101.4, 101.9, 100.5, 99.9, 101.2,
        101.7, 99.8, 101.3, 102.0, 100.7, 100.3, 101.5, 101.0, 100.4, 101.8,
        100.6, 100.0, 101.6, 101.1, 100.2, 101.9, 100.8, 99.7, 101.4, 102.1,
        99.6, 100.9, 101.7, 100.1, 100.5,
    ];
    
    let params = TradjemaParams::default(); // length: 40, mult: 10.0
    let input = TradjemaInput::from_slices(&high, &low, &close, params);
    
    match tradjema(&input) {
        Ok(output) => {
            println!("TRADJEMA calculation successful!");
            println!("Input data length: {}", close.len());
            println!("Output values length: {}", output.values.len());
            
            // Show last 5 values
            println!("\nLast 5 TRADJEMA values:");
            let start = output.values.len().saturating_sub(5);
            for (i, value) in output.values[start..].iter().enumerate() {
                let idx = start + i;
                if value.is_finite() {
                    println!("  [{:2}] Close: {:.2}, TRADJEMA: {:.8}", idx, close[idx], value);
                } else {
                    println!("  [{:2}] Close: {:.2}, TRADJEMA: NaN (warmup)", idx, close[idx]);
                }
            }
            
            // Show how many valid values we got
            let valid_count = output.values.iter().filter(|v| v.is_finite()).count();
            println!("\nValid values: {} out of {}", valid_count, output.values.len());
            println!("Warmup period: {} bars", output.values.len() - valid_count);
        }
        Err(e) => {
            println!("Error calculating TRADJEMA: {}", e);
        }
    }
}