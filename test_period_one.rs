// Test that period=1 is rejected
use my_project::indicators::linearreg_angle::{linearreg_angle, Linearreg_angleInput, Linearreg_angleParams};

fn main() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let params = Linearreg_angleParams { period: Some(1) };
    let input = Linearreg_angleInput::from_slice(&data, params);
    
    match linearreg_angle(&input) {
        Ok(_) => {
            println!("ERROR: period=1 should be rejected!");
            std::process::exit(1);
        }
        Err(e) => {
            println!("✓ period=1 correctly rejected with error: {}", e);
            
            // Test period=2 should work
            let params2 = Linearreg_angleParams { period: Some(2) };
            let input2 = Linearreg_angleInput::from_slice(&data, params2);
            match linearreg_angle(&input2) {
                Ok(output) => {
                    println!("✓ period=2 works correctly, output length: {}", output.values.len());
                    
                    // Check that we don't get all NaN (which would happen with divisor=0)
                    let non_nan_count = output.values.iter().filter(|v| !v.is_nan()).count();
                    if non_nan_count > 0 {
                        println!("✓ Got {} non-NaN values, no division by zero!", non_nan_count);
                    } else {
                        println!("ERROR: All values are NaN, possible division by zero!");
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    println!("ERROR: period=2 should work but got error: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}