// Test that CWMA uses period weights correctly
fn main() {
    // Simple test data
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let period = 3;
    
    // Weights should be: [3³, 2³, 1³] = [27, 8, 1]
    // For position 2 (value 3.0), looking back at [1.0, 2.0, 3.0]:
    // CWMA = (3*27 + 2*8 + 1*1) / (27+8+1) = (81 + 16 + 1) / 36 = 98/36 = 2.7222...
    
    let weights_sum = 27.0 + 8.0 + 1.0; // 36
    let weighted_sum = 3.0 * 27.0 + 2.0 * 8.0 + 1.0 * 1.0; // 98
    let expected = weighted_sum / weights_sum;
    
    println!("For data [1,2,3,4,5] with period 3:");
    println!("Weights: [27, 8, 1]");
    println!("At position 2 (value 3.0):");
    println!("Expected CWMA = {}", expected);
}
