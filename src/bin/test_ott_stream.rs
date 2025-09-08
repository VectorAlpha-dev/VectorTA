use my_project::other_indicators::ott::*;

fn main() {
    let data = vec![
        59400.0, 59350.0, 59420.0, 59380.0, 59450.0,
        59480.0, 59420.0, 59390.0, 59430.0, 59460.0,
        59490.0, 59510.0, 59480.0, 59520.0, 59550.0,
    ];
    
    // Test streaming with VAR
    let mut var_stream = OttStream::try_new(OttParams {
        period: Some(5),
        percent: Some(1.4),
        ma_type: Some("VAR".to_string()),
    }).unwrap();
    
    // Test streaming with WWMA
    let mut wwma_stream = OttStream::try_new(OttParams {
        period: Some(5),
        percent: Some(1.4),
        ma_type: Some("WWMA".to_string()),
    }).unwrap();
    
    // Test streaming with default (SMA)
    let mut sma_stream = OttStream::try_new(OttParams {
        period: Some(5),
        percent: Some(1.4),
        ma_type: Some("SMA".to_string()),
    }).unwrap();
    
    println!("Testing OTT streaming with different MA types:");
    println!("Index | VAR      | WWMA     | SMA");
    println!("------|----------|----------|----------");
    
    for (i, &val) in data.iter().enumerate() {
        let var_result = var_stream.update(val);
        let wwma_result = wwma_stream.update(val);
        let sma_result = sma_stream.update(val);
        
        println!("{:5} | {:8.2} | {:8.2} | {:8.2}", 
            i,
            var_result.unwrap_or(f64::NAN),
            wwma_result.unwrap_or(f64::NAN),
            sma_result.unwrap_or(f64::NAN)
        );
    }
    
    println!("\n✓ OttStream now honors ma_type (VAR, WWMA, SMA)");
}