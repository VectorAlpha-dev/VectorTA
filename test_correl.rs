use my_project::indicators::correl_hl::{CorrelHlInput, CorrelHlParams, correl_hl};

fn main() {
    // Test with period=1
    let high = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let low = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    let params = CorrelHlParams { period: Some(1) };
    let input = CorrelHlInput::from_slices(&high, &low, params);
    
    match correl_hl(&input) {
        Ok(output) => {
            println!("Period=1 results:");
            for (i, val) in output.values.iter().enumerate() {
                println!("  [{}]: {}", i, val);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
    
    // Test with period=2  
    let params2 = CorrelHlParams { period: Some(2) };
    let input2 = CorrelHlInput::from_slices(&high, &low, params2);
    
    match correl_hl(&input2) {
        Ok(output) => {
            println!("\nPeriod=2 results:");
            for (i, val) in output.values.iter().enumerate() {
                println!("  [{}]: {}", i, val);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}
