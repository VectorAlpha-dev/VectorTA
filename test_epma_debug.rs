use my_project::indicators::moving_averages::epma::*;

fn main() {
    // Create test data similar to what might cause the issue
    let mut data = vec![0.0; 100];
    data[50] = 772733.5359199807; // Large spike
    
    let period = 10;
    let offset = 0;
    
    let params = EpmaParams {
        period: Some(period),
        offset: Some(offset),
    };
    
    let input = EpmaInput::from_slice(&data, params);
    let output = epma(&input).unwrap();
    
    println!("Testing EPMA with period={}, offset={}", period, offset);
    println!("Data has spike at index 50: {}", data[50]);
    
    // Check the window around the spike
    for i in 45..65 {
        if i >= period + 1 {
            let start_idx = i + 1 - (period - 1);
            let window = &data[start_idx..=i];
            let lo = window.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            
            println!("Index {}: window[{}..={}], lo={}, hi={}, output={}", 
                     i, start_idx, i, lo, hi, output.values[i]);
            
            if !output.values[i].is_nan() && (output.values[i] < lo - 1e-9 || output.values[i] > hi + 1e-9) {
                println!("  ERROR: Output {} is outside bounds [{}, {}]", output.values[i], lo, hi);
            }
        }
    }
    
    // Let's also manually calculate what EPMA should produce
    println!("\nManual calculation:");
    let p1 = period - 1;
    let mut weights = Vec::with_capacity(p1);
    for i in 0..p1 {
        let w = (period as i32 - i as i32 - offset as i32) as f64;
        weights.push(w);
        println!("  weight[{}] = {}", i, w);
    }
    let weight_sum: f64 = weights.iter().sum();
    println!("  weight_sum = {}", weight_sum);
    
    // Calculate for index 55 (after the spike)
    let j = 55;
    let mut sum = 0.0;
    println!("\nFor index {}:", j);
    for i in 0..p1 {
        let val = data[j - i];
        println!("  data[{}] * weight[{}] = {} * {} = {}", j-i, i, val, weights[i], val * weights[i]);
        sum += val * weights[i];
    }
    println!("  sum = {}, result = {}", sum, sum / weight_sum);
}