use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct EmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EmaParams {
    pub period: Option<usize>,
}

impl Default for EmaParams {
    fn default() -> Self {
        EmaParams { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct EmaInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: EmaParams,
}

impl<'a> EmaInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: EmaParams) -> Self {
        EmaInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        EmaInput {
            candles,
            source: "close",
            params: EmaParams::default(),
        }
    }

    #[inline]
    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| EmaParams::default().period.unwrap())
    }
}

#[inline]
pub fn ema(input: &EmaInput) -> Result<EmaOutput, Box<dyn Error>> {
    let data = source_type(input.candles, input.source);
    let len = data.len();
    let period = input.get_period();

    if period == 0 || period > data.len() {
        return Err("Invalid period specified for EMA calculation.".into());
    }

    let len = data.len();
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut ema_values = Vec::with_capacity(len);

    let mut last_ema = data[0];
    ema_values.push(last_ema);

    for i in 1..len {
        last_ema = alpha * data[i] + (1.0 - alpha) * last_ema;
        ema_values.push(last_ema);
    }

    Ok(EmaOutput { values: ema_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ema_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let close_prices = &candles.close;
        let params = EmaParams { period: Some(9) };

        let old_style_input = EmaInput {
            candles: &candles,
            source: "close",
            params,
        };
        let ema_result = ema(&old_style_input).expect("Failed to calculate EMA");

        let expected_last_five_ema = [59302.2, 59277.9, 59230.2, 59215.1, 59103.1];

        assert!(
            ema_result.values.len() >= 5,
            "Not enough EMA values for the test"
        );
        assert_eq!(
            ema_result.values.len(),
            close_prices.len(),
            "EMA output length does not match input length"
        );

        let start_index = ema_result.values.len().saturating_sub(5);
        let result_last_five_ema = &ema_result.values[start_index..];

        for (i, &value) in result_last_five_ema.iter().enumerate() {
            assert!(
                (value - expected_last_five_ema[i]).abs() < 1e-1,
                "EMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_ema[i],
                value
            );
        }

        let default_input = EmaInput::with_default_params(&candles);
        let default_ema_result = ema(&default_input).expect("Failed to calculate default EMA");
        assert!(
            !default_ema_result.values.is_empty(),
            "Should produce EMA values with default params"
        );
    }
}
