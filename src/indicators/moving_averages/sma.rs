use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct SmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SmaParams {
    pub period: Option<usize>,
}

impl SmaParams {
    pub fn with_default_params() -> Self {
        SmaParams { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct SmaInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: SmaParams,
}

impl<'a> SmaInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: SmaParams) -> Self {
        SmaInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        SmaInput {
            candles,
            source: "close",
            params: SmaParams::with_default_params(),
        }
    }
}
#[inline]
pub fn sma(input: &SmaInput) -> Result<SmaOutput, Box<dyn Error>> {
    let data: &[f64] = source_type(input.candles, input.source);
    let len: usize = data.len();
    let period = input.params.period.unwrap_or(9);
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
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_sma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = SmaParams { period: None };
        let input_default = SmaInput::new(&candles, "close", default_params);
        let output_default = sma(&input_default).expect("Failed SMA with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = SmaParams { period: Some(14) };
        let input_period_14 = SmaInput::new(&candles, "hl2", params_period_14);
        let output_period_14 =
            sma(&input_period_14).expect("Failed SMA with period=14, source=hl2");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = SmaParams { period: Some(20) };
        let input_custom = SmaInput::new(&candles, "hlc3", params_custom);
        let output_custom = sma(&input_custom).expect("Failed SMA fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_sma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = SmaParams { period: Some(9) };
        let input = SmaInput::new(&candles, "close", params);
        let sma_result = sma(&input).expect("Failed to calculate SMA");

        assert_eq!(
            sma_result.values.len(),
            close_prices.len(),
            "SMA length mismatch"
        );

        let expected_last_five_sma = [59180.8, 59175.0, 59129.4, 59085.4, 59133.7];
        assert!(sma_result.values.len() >= 5, "SMA length too short");
        let start_index = sma_result.values.len() - 5;
        let result_last_five_sma = &sma_result.values[start_index..];
        for (i, &value) in result_last_five_sma.iter().enumerate() {
            let expected_value = expected_last_five_sma[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "SMA mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period: usize = 9;
        for i in 0..(period - 1) {
            assert!(sma_result.values[i].is_nan());
        }

        let default_input = SmaInput::with_default_params(&candles);
        let default_sma_result = sma(&default_input).expect("Failed to calculate SMA defaults");
        assert_eq!(default_sma_result.values.len(), close_prices.len());
    }
}
