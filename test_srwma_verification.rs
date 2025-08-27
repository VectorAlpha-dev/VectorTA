use my_project::indicators::moving_averages::srwma::{srwma, SrwmaInput, SrwmaParams};

fn main() {
    // Test with simple data and period=3
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let params = SrwmaParams { period: Some(3) };
    let input = SrwmaInput::from_slice(&data, params);
    
    let result = srwma(&input).unwrap();
    
    println!("SRWMA results:");
    for (i, val) in result.values.iter().enumerate() {
        if val.is_nan() {
            println!("  [{}]: NaN (warmup)", i);
        } else {
            println!("  [{}]: {}", i, val);
        }
    }
    
    // Check if the first valid value (at index 2) matches our calculation
    let expected_at_2 = 2.1765567128029324;
    let expected_at_3 = 3.1765567128029324;
    let expected_at_4 = 4.176556712802932;
    
    println!("\nValidation:");
    println!("  At index 2: expected {}, got {}", expected_at_2, result.values[2]);
    println!("  At index 3: expected {}, got {}", expected_at_3, result.values[3]);
    println!("  At index 4: expected {}, got {}", expected_at_4, result.values[4]);
    
    let eps = 1e-10;
    assert!((result.values[2] - expected_at_2).abs() < eps, "Mismatch at index 2");
    assert!((result.values[3] - expected_at_3).abs() < eps, "Mismatch at index 3");
    assert!((result.values[4] - expected_at_4).abs() < eps, "Mismatch at index 4");
    
    println!("\n✅ All values match! The implementation is correct.");
}