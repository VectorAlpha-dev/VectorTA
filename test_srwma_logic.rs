// Test to verify SRWMA logic with period weights
fn main() {
    // Test with simple data and period=3
    // Data: [1.0, 2.0, 3.0, 4.0, 5.0]
    // Period: 3
    // Weights should be: sqrt(3), sqrt(2), sqrt(1) = 1.732, 1.414, 1.0
    // Sum of weights: 1.732 + 1.414 + 1.0 = 4.146
    
    let period = 3;
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    // Calculate weights
    let mut weights = Vec::new();
    let mut sum_weights = 0.0;
    for i in 0..period {
        let w = ((period - i) as f64).sqrt();
        weights.push(w);
        sum_weights += w;
        println!("Weight[{}] = sqrt({}) = {}", i, period - i, w);
    }
    println!("Sum of weights: {}", sum_weights);
    
    // Calculate SRWMA at index 2 (first valid output with period=3)
    // Window: data[0], data[1], data[2] = 1.0, 2.0, 3.0
    // SRWMA = (3.0*sqrt(3) + 2.0*sqrt(2) + 1.0*sqrt(1)) / sum_weights
    //       = (3.0*1.732 + 2.0*1.414 + 1.0*1.0) / 4.146
    //       = (5.196 + 2.828 + 1.0) / 4.146
    //       = 9.024 / 4.146
    //       = 2.177
    
    let idx = 2;
    let mut srwma_val = 0.0;
    for k in 0..period {
        srwma_val += data[idx - k] * weights[k];
    }
    srwma_val /= sum_weights;
    
    println!("\nAt index {}: window = [{}, {}, {}]", idx, data[0], data[1], data[2]);
    println!("SRWMA = {}", srwma_val);
    println!("Expected ~2.177");
    
    // Calculate SRWMA at index 3
    // Window: data[1], data[2], data[3] = 2.0, 3.0, 4.0
    let idx = 3;
    let mut srwma_val = 0.0;
    for k in 0..period {
        srwma_val += data[idx - k] * weights[k];
    }
    srwma_val /= sum_weights;
    
    println!("\nAt index {}: window = [{}, {}, {}]", idx, data[1], data[2], data[3]);
    println!("SRWMA = {}", srwma_val);
    
    // Calculate SRWMA at index 4
    // Window: data[2], data[3], data[4] = 3.0, 4.0, 5.0
    let idx = 4;
    let mut srwma_val = 0.0;
    for k in 0..period {
        srwma_val += data[idx - k] * weights[k];
    }
    srwma_val /= sum_weights;
    
    println!("\nAt index {}: window = [{}, {}, {}]", idx, data[2], data[3], data[4]);
    println!("SRWMA = {}", srwma_val);
}