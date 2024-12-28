use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct HighPassOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HighPassParams {
    pub period: Option<usize>,
}

impl HighPassParams {
    pub fn with_default_params() -> Self {
        HighPassParams { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct HighPassInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: HighPassParams,
}

impl<'a> HighPassInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: HighPassParams) -> Self {
        HighPassInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        HighPassInput {
            candles,
            source: "close",
            params: HighPassParams::with_default_params(),
        }
    }
}

#[inline]
pub fn highpass(input: &HighPassInput) -> Result<HighPassOutput, Box<dyn Error>> {
    let data = source_type(input.candles, input.source);
    let period = input.params.period.unwrap_or(48);
    let len = data.len();
    if len <= 2 || period == 0 || period > len {
        return Err("Invalid period or insufficient data for highpass calculation.".into());
    }

    let k = 1.0;
    let two_pi_k_div = 2.0 * PI * k / (period as f64);
    let sin_val = two_pi_k_div.sin();
    let cos_val = two_pi_k_div.cos();
    let alpha = 1.0 + (sin_val - 1.0) / cos_val;

    let one_minus_half_alpha = 1.0 - alpha / 2.0;
    let one_minus_alpha = 1.0 - alpha;

    let mut newseries = vec![0.0; len];
    newseries[0] = data[0];

    for i in 1..len {
        let val = one_minus_half_alpha * data[i] - one_minus_half_alpha * data[i - 1]
            + one_minus_alpha * newseries[i - 1];
        newseries[i] = val;
    }

    Ok(HighPassOutput { values: newseries })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_highpass_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = HighPassParams { period: None };
        let input_default = HighPassInput::new(&candles, "close", default_params);
        let output_default = highpass(&input_default).expect("Failed highpass with default params");
        assert_eq!(output_default.values.len(), candles.close.len());
        let params_period = HighPassParams { period: Some(36) };
        let input_period = HighPassInput::new(&candles, "hl2", params_period);
        let output_period =
            highpass(&input_period).expect("Failed highpass with period=36, source=hl2");
        assert_eq!(output_period.values.len(), candles.close.len());
    }

    #[test]
    fn test_highpass_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HighPassInput::with_default_params(&candles);
        let result = highpass(&input).expect("Failed to calculate highpass");
        let expected_last_five = [
            -265.1027020005024,
            -330.0916060058495,
            -422.7478979710918,
            -261.87532144673423,
            -698.9026088956363,
        ];
        assert_eq!(result.values.len(), candles.close.len());
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            assert!(
                (val - expected_last_five[i]).abs() < 1e-6,
                "Highpass mismatch at {}: expected {}, got {}",
                i,
                expected_last_five[i],
                val
            );
        }
        for val in &result.values {
            assert!(val.is_finite());
        }
    }
}
