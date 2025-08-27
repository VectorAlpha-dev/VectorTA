use my_project::indicators::moving_averages::srwma::{srwma_with_kernel, SrwmaInput, SrwmaParams};
use my_project::utilities::enums::Kernel;

fn main() {
    // Test with simple data and period=3
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let params = SrwmaParams { period: Some(3) };
    let input = SrwmaInput::from_slice(&data, params);
    
    println!("Input data: {:?}", data);
    println!("Period: 3");
    println!("Expected first valid index: 2 (first=0 + period=3 - 1)");
    
    let result = srwma_with_kernel(&input, Kernel::Scalar).unwrap();
    
    println!("\nSRWMA results:");
    for (i, val) in result.values.iter().enumerate() {
        if val.is_nan() {
            println!("  [{}]: NaN", i);
        } else {
            println!("  [{}]: {}", i, val);
        }
    }
    
    // Now test with period=2 to see if the pattern holds
    println!("\n--- Testing with period=2 ---");
    let params2 = SrwmaParams { period: Some(2) };
    let input2 = SrwmaInput::from_slice(&data, params2);
    let result2 = srwma_with_kernel(&input2, Kernel::Scalar).unwrap();
    
    println!("Expected first valid index: 1 (first=0 + period=2 - 1)");
    println!("\nSRWMA results:");
    for (i, val) in result2.values.iter().enumerate() {
        if val.is_nan() {
            println!("  [{}]: NaN", i);
        } else {
            println!("  [{}]: {}", i, val);
        }
    }
    
    // Test with period=4
    println!("\n--- Testing with period=4 ---");
    let params3 = SrwmaParams { period: Some(4) };
    let input3 = SrwmaInput::from_slice(&data, params3);
    let result3 = srwma_with_kernel(&input3, Kernel::Scalar).unwrap();
    
    println!("Expected first valid index: 3 (first=0 + period=4 - 1)");
    println!("\nSRWMA results:");
    for (i, val) in result3.values.iter().enumerate() {
        if val.is_nan() {
            println!("  [{}]: NaN", i);
        } else {
            println!("  [{}]: {}", i, val);
        }
    }
}