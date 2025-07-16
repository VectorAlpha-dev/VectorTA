use my_project::indicators::adxr::{adxr, AdxrInput, AdxrParams};
use my_project::utilities::data_loader::load_from_csv;

fn main() {
    // Load test data
    let data = load_from_csv("./tests/integration_test_data.csv")
        .expect("Failed to load test data");
    
    // Test with period 15
    let params = AdxrParams { period: Some(15) };
    let input = AdxrInput::from_candles(&data, params);
    let output = adxr(&input).expect("ADXR calculation failed");
    
    println!("ADXR with period 15:");
    println!("Values at indices 25-35:");
    for i in 25..36 {
        println!("  [{}] = {}", i, output.values[i]);
    }
    
    println!("\nFirst non-NaN value:");
    for (i, val) in output.values.iter().enumerate() {
        if !val.is_nan() && *val != 0.0 {
            println!("  Index {} = {}", i, val);
            break;
        }
    }
}