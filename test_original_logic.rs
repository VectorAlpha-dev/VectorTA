// Test to understand the original SRWMA logic that produced the expected values
fn main() {
    // Original implementation used wlen = period - 1
    // With period=14, that means 13 weights
    let period = 14;
    let wlen = period - 1;  // This was the original
    
    println!("Original implementation:");
    println!("Period: {}", period);
    println!("Weight length: {} (period - 1)", wlen);
    
    // Original weights: sqrt(14), sqrt(13), ..., sqrt(2)
    // Note: it STOPPED at sqrt(2), not sqrt(1)
    let mut weights_orig = Vec::new();
    let mut sum_orig = 0.0;
    for i in 0..wlen {
        let w = ((period - i) as f64).sqrt();
        weights_orig.push(w);
        sum_orig += w;
        if i < 3 || i >= wlen - 3 {
            println!("  Weight[{}] = sqrt({}) = {}", i, period - i, w);
        } else if i == 3 {
            println!("  ...");
        }
    }
    println!("Sum of weights: {}", sum_orig);
    
    println!("\nNew implementation (what the guide suggests):");
    println!("Period: {}", period);
    println!("Weight length: {} (period)", period);
    
    // New weights: sqrt(14), sqrt(13), ..., sqrt(1)
    // This INCLUDES sqrt(1)
    let mut weights_new = Vec::new();
    let mut sum_new = 0.0;
    for i in 0..period {
        let w = ((period - i) as f64).sqrt();
        weights_new.push(w);
        sum_new += w;
        if i < 3 || i >= period - 3 {
            println!("  Weight[{}] = sqrt({}) = {}", i, period - i, w);
        } else if i == 3 {
            println!("  ...");
        }
    }
    println!("Sum of weights: {}", sum_new);
    
    println!("\nDifference:");
    println!("New implementation adds weight sqrt(1) = 1.0");
    println!("This changes the sum from {} to {}", sum_orig, sum_new);
}