use std::error::Error;

#[derive(Debug, Clone)]
pub struct SmaParams {
    pub period: Option<usize>,
}

impl Default for SmaParams {
    fn default() -> Self {
        SmaParams { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct SmaInput<'a> {
    pub data: &'a [f64],
    pub params: SmaParams,
}

impl<'a> SmaInput<'a> {
    pub fn new(data: &'a [f64], params: SmaParams) -> Self {
        SmaInput { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        SmaInput {
            data,
            params: SmaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SmaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct SmaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_sma(input: &SmaInput) -> Result<SmaOutput, Box<dyn Error>> {
    let data = input.data;
    let period = input.get_period();

    if period == 0 || period > data.len() {
        return Err("Invalid period specified for SMA calculation.".into());
    }

    let len = data.len();
    let mut sma_values = vec![f64::NAN; len];

    let inv_period = 1.0 / period as f64;
    let mut sum: f64 = 0.0;

    for i in 0..period {
        sum += data[i];
    }
    sma_values[period - 1] = sum * inv_period;

    for i in period..len {
        sum += data[i] - data[i - period];
        sma_values[i] = sum * inv_period;
    }

    Ok(SmaOutput { values: sma_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::data_loader::read_candles_from_csv;

    #[test]
    fn test_sma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = SmaParams { period: Some(9) };
        let input = SmaInput::new(close_prices, params);
        let sma_result = calculate_sma(&input).expect("Failed to calculate SMA");

        assert_eq!(
            sma_result.values.len(),
            close_prices.len(),
            "SMA values count should match the input data length"
        );

        let expected_last_five_sma = [59180.8, 59175.0, 59129.4, 59085.4, 59133.7];
        assert!(
            sma_result.values.len() >= 5,
            "Not enough SMA values for the test"
        );

        let start_index = sma_result.values.len() - 5;
        let result_last_five_sma = &sma_result.values[start_index..];

        for (i, &value) in result_last_five_sma.iter().enumerate() {
            let expected_value = expected_last_five_sma[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "SMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period = input.get_period();
        for i in 0..(period - 1) {
            let val = sma_result.values[i];
            assert!(
                val.is_nan(),
                "Expected NaN for early index {}, got {}",
                i,
                val
            );
        }

        let default_input = SmaInput::with_default_params(close_prices);
        let default_sma_result =
            calculate_sma(&default_input).expect("Failed to calculate SMA with defaults");
        assert!(
            !default_sma_result.values.is_empty(),
            "Should produce some SMA values with default params"
        );
    }
}
