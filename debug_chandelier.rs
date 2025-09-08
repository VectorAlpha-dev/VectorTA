use my_project::other_indicators::chandelier_exit::{chandelier_exit, ChandelierExitInput, ChandelierExitParams};
use my_project::utilities::data_loader::read_candles_from_csv;

fn main() {
    let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
    
    println!("Total candles: {}", candles.close.len());
    
    // Find where prices are around 68k
    let mut indices = Vec::new();
    for (i, &close) in candles.close.iter().enumerate() {
        if close > 65000.0 && close < 70000.0 {
            indices.push(i);
        }
    }
    
    if !indices.is_empty() {
        println!("Found {} bars with close between 65k-70k", indices.len());
        println!("First at index: {}", indices[0]);
        println!("Last at index: {}", indices[indices.len()-1]);
        
        // Test with data up to the last 68k price point
        let end_idx = indices[indices.len()-1] + 1;
        println!("\nTesting with data up to index {}", end_idx);
        
        let high_subset = &candles.high[..end_idx];
        let low_subset = &candles.low[..end_idx];
        let close_subset = &candles.close[..end_idx];
        
        let params = ChandelierExitParams {
            period: Some(22),
            mult: Some(3.0),
            use_close: Some(true),
        };
        
        let input = ChandelierExitInput::from_slices(high_subset, low_subset, close_subset, params);
        let result = chandelier_exit(&input).expect("Chandelier Exit calculation failed");
        
        // Get last 5 non-NaN short_stop values
        let mut non_nan_short_stops = Vec::new();
        for i in (0..result.short_stop.len()).rev() {
            if !result.short_stop[i].is_nan() {
                non_nan_short_stops.push(result.short_stop[i]);
                if non_nan_short_stops.len() == 5 {
                    break;
                }
            }
        }
        non_nan_short_stops.reverse();
        
        println!("\nLast 5 non-NaN short_stop values:");
        for (i, val) in non_nan_short_stops.iter().enumerate() {
            println!("  [{}]: {:.8}", i, val);
        }
        
        println!("\nExpected values:");
        let expected = vec![68719.23648167, 68705.54391432, 68244.42828185, 67599.49972358, 66883.02246342];
        for (i, val) in expected.iter().enumerate() {
            println!("  [{}]: {:.8}", i, val);
        }
    }
}