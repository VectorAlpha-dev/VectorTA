use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct SrwmaParams {
    pub period: Option<usize>,
}

impl Default for SrwmaParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SrwmaInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: SrwmaParams,
}

impl<'a> SrwmaInput<'a> {
    #[inline]
    pub fn new(candles: &'a Candles, source: &'a str, params: SrwmaParams) -> Self {
        Self { candles, source, params }
    }

    #[inline]
    pub fn with_default_params(candles: &'a Candles) -> Self {
        Self {
            candles,
            source: "close",
            params: SrwmaParams::default(),
        }
    }

    #[inline]
    fn get_period(&self) -> usize {
        self.params.period.unwrap_or_else(|| SrwmaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct SrwmaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn srwma(input: &SrwmaInput) -> Result<SrwmaOutput, Box<dyn Error>> {
    let data: &[f64] = source_type(input.candles, input.source);
    let period: usize = input.get_period();

    if data.is_empty() {
        return Ok(SrwmaOutput { values: vec![] });
    }
    if period == 0 {
        return Err("SRWMA period must be >= 1.".into());
    }

    let len = data.len();
    if period + 1 > len {
        return Ok(SrwmaOutput {
            values: data.to_vec(),
        });
    }

    let mut weights = Vec::with_capacity(period - 1);
    for i in 0..(period - 1) {
        weights.push((period as f64 - i as f64).sqrt());
    }
    let sum_of_weights: f64 = weights.iter().sum();

    let mut srwma_values = data.to_vec();

    for j in (period + 1)..len {
        let mut my_sum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            my_sum += data[j - i] * w;
        }
        srwma_values[j] = my_sum / sum_of_weights;
    }

    Ok(SrwmaOutput {
        values: srwma_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_srwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles.select_candle_field("close").expect("Failed to extract close");
        let params = SrwmaParams { period: Some(14) };
        let input = SrwmaInput::new(&candles, "close", params);
        let result = srwma(&input).expect("SRWMA calculation failed");
        let vals = &result.values;
        assert_eq!(vals.len(), close_prices.len());
        let expected_last_five = [
            59344.28384704595,
            59282.09151629659,
            59192.76580529367,
            59178.04767548977,
            59110.03801260874,
        ];
        assert!(vals.len() >= 5);
        let start_index = vals.len() - 5;
        let last_five = &vals[start_index..];
        for (i, (&actual, &expected)) in last_five.iter().zip(expected_last_five.iter()).enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "SRWMA mismatch at index {}: expected {:.14}, got {:.14}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_srwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = SrwmaParams { period: None };
        let input = SrwmaInput::new(&candles, "close", default_params);
        let output = srwma(&input).expect("Failed SRWMA with default params");
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_14 = SrwmaParams { period: Some(14) };
        let input2 = SrwmaInput::new(&candles, "hl2", params_period_14);
        let output2 = srwma(&input2).expect("Failed SRWMA with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = SrwmaParams { period: Some(10) };
        let input3 = SrwmaInput::new(&candles, "hlc3", params_custom);
        let output3 = srwma(&input3).expect("Failed SRWMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }
}