#[cfg(test)]
mod debug_supertrend {
    use crate::indicators::supertrend::{supertrend, SuperTrendInput, SuperTrendParams};
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn debug_supertrend_values() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path)?;
        
        let params = SuperTrendParams { period: Some(10), factor: Some(3.0) };
        let input = SuperTrendInput::from_candles(&candles, params);
        let st_result = supertrend(&input)?;
        
        println!("\nData length: {}", candles.close.len());
        println!("Last 10 close prices:");
        let start = candles.close.len().saturating_sub(10);
        for i in start..candles.close.len() {
            println!("  [{}] close: {}, high: {}, low: {}", 
                     i, candles.close[i], candles.high[i], candles.low[i]);
        }
        
        println!("\nLast 10 trend values:");
        let start = st_result.trend.len().saturating_sub(10);
        for i in start..st_result.trend.len() {
            println!("  [{}] trend: {}, changed: {}", i, st_result.trend[i], st_result.changed[i]);
        }
        
        println!("\nExpected last 5 trend values:");
        let expected_last_five_trend = [
            61811.479454208165,
            61721.73150878735,
            61459.10835790861,
            61351.59752211775,
            61033.18776990598,
        ];
        let start_idx = st_result.trend.len() - 5;
        for (i, &exp) in expected_last_five_trend.iter().enumerate() {
            let idx = start_idx + i;
            let actual = st_result.trend[idx];
            println!("  [{}] expected: {}, actual: {}, diff: {}", 
                     idx, exp, actual, (actual - exp).abs());
        }
        
        Ok(())
    }
}