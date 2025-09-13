// Quick test for new MA selectors
use my_project::indicators::moving_averages::ma::{ma, MaData};
use my_project::indicators::moving_averages::ma_stream::ma_stream;

fn main() {
    // Test data
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let period = 3;
    
    // Test new MAs in batch selector
    println!("Testing batch selector (ma):");
    
    let test_mas = vec![
        "dma", "ehlers_ecema", "ehlers_kama", "ehma", "nama", "sama", "vama"
    ];
    
    for ma_type in test_mas {
        match ma(ma_type, MaData::Slice(&data), period) {
            Ok(result) => {
                println!("  {} ✓ - Result length: {}", ma_type, result.len());
            }
            Err(e) => {
                println!("  {} ✗ - Error: {}", ma_type, e);
            }
        }
    }
    
    // Test new MAs in streaming selector
    println!("\nTesting streaming selector (ma_stream):");
    
    let test_stream_mas = vec![
        "dma", "ehlers_ecema", "ehlers_kama", "ehma", "nama", "sama", "vama", "volatility_adjusted_ma"
    ];
    
    for ma_type in test_stream_mas {
        match ma_stream(ma_type, period) {
            Ok(mut stream) => {
                // Test a few updates
                let val1 = stream.update(1.0);
                let val2 = stream.update(2.0);
                let val3 = stream.update(3.0);
                let val4 = stream.update(4.0);
                println!("  {} ✓ - Stream created, last value: {:?}", ma_type, val4);
            }
            Err(e) => {
                println!("  {} ✗ - Error: {}", ma_type, e);
            }
        }
    }
    
    println!("\n✅ Test complete!");
}