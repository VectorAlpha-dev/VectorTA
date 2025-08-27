use my_project::indicators::moving_averages::hma::{
    hma, hma_into_slice, HmaInput, HmaParams, HmaStream,
};

fn main() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let period = 5;
    let params = HmaParams { period: Some(period) };
    let input = HmaInput::from_slice(&data, params.clone());

    // Test 1: hma (returns Vec)
    let result1 = hma(&input).unwrap();
    let first_valid1 = result1.values.iter().position(|x| !x.is_nan());
    println!("hma() first non-NaN at index: {:?}", first_valid1);

    // Test 2: hma_into_slice
    let mut result2 = vec![0.0; data.len()];
    hma_into_slice(&mut result2, &input, my_project::utilities::enums::Kernel::Auto).unwrap();
    let first_valid2 = result2.iter().position(|x| !x.is_nan());
    println!("hma_into_slice() first non-NaN at index: {:?}", first_valid2);

    // Test 3: streaming
    let mut stream = HmaStream::try_new(params).unwrap();
    let mut result3 = Vec::new();
    for &val in &data {
        result3.push(stream.update(val).unwrap_or(f64::NAN));
    }
    let first_valid3 = result3.iter().position(|x| !x.is_nan());
    println!("HmaStream first non-NaN at index: {:?}", first_valid3);

    // Verify they all match
    assert_eq!(first_valid1, first_valid2, "hma() and hma_into_slice() mismatch!");
    assert_eq!(first_valid1, first_valid3, "hma() and HmaStream mismatch!");

    // Calculate expected first_out
    let first = 0; // no NaN in input
    let sqrt_len = (period as f64).sqrt().floor() as usize;
    let expected_first_out = first + period + sqrt_len - 2;
    println!("Expected first_out = {} + {} + {} - 2 = {}", first, period, sqrt_len, expected_first_out);

    assert_eq!(first_valid1, Some(expected_first_out), "First non-NaN doesn't match expected!");

    println!("\nAll warmup checks passed! First non-NaN consistently at index {}", expected_first_out);
}