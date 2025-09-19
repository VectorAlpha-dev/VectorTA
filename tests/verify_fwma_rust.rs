// Temporary test file to verify FWMA calculations using Rust
use my_project::indicators::moving_averages::fwma::{fwma, FwmaInput, FwmaParams};

fn main() {
    println!("Verifying FWMA calculations in Rust\n");

    // Test 1: Same as Python/WASM tests
    println!("Test 1: Period=5, Data=[1,2,3,4,5]");
    let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let params1 = FwmaParams { period: Some(5) };
    let input1 = FwmaInput::from_slice(&data1, params1);
    let result1 = fwma(&input1).unwrap();

    println!("Fibonacci sequence for period=5: [1, 1, 2, 3, 5]");
    println!("Sum = 12, Normalized weights: [1/12, 1/12, 2/12, 3/12, 5/12]");
    println!("Expected: (1*1 + 2*1 + 3*2 + 4*3 + 5*5) / 12 = 46/12 = 3.833333...");
    println!("Rust result: {:?}", result1.values);
    println!("Last value: {:.10}", result1.values[4]);

    // Test 2: Same as WASM consistency test
    println!("\nTest 2: Period=4, Data=[10,20,30,40,50,60,70,80,90,100]");
    let data2 = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
    let params2 = FwmaParams { period: Some(4) };
    let input2 = FwmaInput::from_slice(&data2, params2);
    let result2 = fwma(&input2).unwrap();

    println!("Fibonacci sequence for period=4: [1, 1, 2, 3]");
    println!("Sum = 7, Normalized weights: [1/7, 1/7, 2/7, 3/7]");
    println!("Expected at index 3: (10*1 + 20*1 + 30*2 + 40*3) / 7 = 210/7 = 30");
    println!("Expected at index 4: (20*1 + 30*1 + 40*2 + 50*3) / 7 = 280/7 = 40");
    println!("Rust results:");
    for i in 0..6 {
        println!("  [{}]: {:.10}", i, result2.values[i]);
    }

    // Manual calculation verification
    println!("\nManual calculation verification:");
    let fib = vec![1.0, 1.0, 2.0, 3.0];
    let fib_sum = 7.0;
    let weights: Vec<f64> = fib.iter().map(|&f| f / fib_sum).collect();
    println!("Weights: {:?}", weights);

    // Calculate for index 3
    let calc3 = 10.0 * weights[0] + 20.0 * weights[1] + 30.0 * weights[2] + 40.0 * weights[3];
    println!("Manual calc for index 3: {:.10}", calc3);

    // Calculate for index 4
    let calc4 = 20.0 * weights[0] + 30.0 * weights[1] + 40.0 * weights[2] + 50.0 * weights[3];
    println!("Manual calc for index 4: {:.10}", calc4);
}
