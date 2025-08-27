use my_project::indicators::moving_averages::cwma::{cwma, CwmaInput, CwmaParams};

fn main() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let params = CwmaParams { period: Some(3) };
    let input = CwmaInput::from_slice(&data, params);
    
    match cwma(&input) {
        Ok(output) => {
            println!("CWMA output: {:?}", output.values);
            
            // Position 2 should be ~2.7222 (calculated manually above)
            if output.values.len() > 2 {
                let val = output.values[2];
                let expected = 2.7222222222222223;
                let diff = (val - expected).abs();
                println!("At position 2: got {}, expected {}, diff = {}", val, expected, diff);
                if diff < 1e-10 {
                    println!("✓ Correct!");
                } else {
                    println!("✗ Mismatch!");
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}
