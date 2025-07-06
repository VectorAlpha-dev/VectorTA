#[cfg(test)]
mod test_actual_values {
    use super::super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn print_actual_fwma_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        
        println!("\nCandle data info:");
        println!("Total candles: {}", candles.close.len());
        println!("Last 10 close prices:");
        let close_len = candles.close.len();
        for i in (close_len - 10)..close_len {
            println!("  [{}]: {}", i - (close_len - 10), candles.close[i]);
        }
        
        let input = FwmaInput::with_default_candles(&candles);
        let result = fwma(&input).unwrap();
        
        println!("\nFWMA results (period = 5):");
        println!("Last 5 values:");
        let result_len = result.values.len();
        for i in (result_len - 5)..result_len {
            println!("  [{}]: {:.12}", i - (result_len - 5), result.values[i]);
        }
        
        // Check against expected
        let expected_last_five = [
            59273.583333333336,
            59252.5,
            59167.083333333336,
            59151.0,
            58940.333333333336,
        ];
        
        println!("\nExpected values:");
        for (i, val) in expected_last_five.iter().enumerate() {
            println!("  [{}]: {:.12}", i, val);
        }
        
        println!("\nDifferences:");
        for i in 0..5 {
            let actual = result.values[result_len - 5 + i];
            let expected = expected_last_five[i];
            println!("  [{}]: {:.12} (diff: {:.2e})", i, actual - expected, (actual - expected).abs());
        }
    }
}