use my_project::indicators::marketefi::{marketefi_batch_with_kernel, MarketefiBatchOutput};
use my_project::utilities::enums::Kernel;

fn main() {
    let high = vec![100.0, 105.0, 110.0, 108.0, 112.0];
    let low = vec![95.0, 98.0, 102.0, 104.0, 106.0];
    let volume = vec![1000.0, 1500.0, 2000.0, 1200.0, 1800.0];
    
    let result: MarketefiBatchOutput = marketefi_batch_with_kernel(&high, &low, &volume, Kernel::Auto)
        .expect("Batch calculation failed");
    
    println!("Batch output:");
    println!("  Rows: {}", result.rows);
    println!("  Cols: {}", result.cols);
    println!("  Values length: {}", result.values.len());
    println!("  Combos length: {}", result.combos.len());
    
    assert_eq!(result.rows, 1, "Should have 1 row");
    assert_eq!(result.cols, 5, "Should have 5 columns");
    assert_eq!(result.values.len(), 5, "Should have 5 values");
    assert_eq!(result.combos.len(), 1, "Should have 1 combo (MarketefiParams)");
    
    println!("\n✓ All assertions passed! MarketefiBatchOutput includes combos field correctly.");
}