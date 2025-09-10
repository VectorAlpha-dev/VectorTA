use my_project::indicators::moving_averages::nama::{nama, NamaInput, NamaParams};

fn main() {
    // Test data matching Python/WASM tests
    let data = vec![
        100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0,
        107.0, 109.0, 111.0, 110.0, 112.0, 114.0, 113.0, 115.0,
    ];
    
    let params = NamaParams { period: Some(5) };
    let input = NamaInput::from_slice(&data, params);
    let result = nama(&input).unwrap();
    
    println!("NAMA Reference Values (period=5):");
    println!("Input data: {:?}", data);
    println!("Output values:");
    for (i, val) in result.values.iter().enumerate() {
        if val.is_nan() {
            println!("  [{}]: NaN", i);
        } else {
            println!("  [{}]: {:.10}", i, val);
        }
    }
    
    // Also get last 5 non-NaN values for easy comparison
    let non_nan_values: Vec<f64> = result.values.iter()
        .filter(|v| !v.is_nan())
        .copied()
        .collect();
    
    if non_nan_values.len() >= 5 {
        let last_5_start = non_nan_values.len() - 5;
        println!("\nLast 5 non-NaN values:");
        println!("[");
        for val in &non_nan_values[last_5_start..] {
            println!("    {:.10},", val);
        }
        println!("]");
    }
    
    // Test with period=30 for larger dataset check
    let large_data: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64).sin() * 10.0).collect();
    let params30 = NamaParams { period: Some(30) };
    let input30 = NamaInput::from_slice(&large_data, params30);
    let result30 = nama(&input30).unwrap();
    
    // Get last 5 values for period=30
    let last_5_vals: Vec<f64> = result30.values.iter()
        .rev()
        .take(5)
        .rev()
        .copied()
        .collect();
    
    println!("\nNAMA with period=30, last 5 values:");
    println!("[");
    for val in last_5_vals {
        if val.is_nan() {
            println!("    NaN,");
        } else {
            println!("    {:.10},", val);
        }
    }
    println!("]");
}