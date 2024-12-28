use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum SmmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SmmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SmmaParams {
    pub period: Option<usize>,
}

impl SmmaParams {
    pub fn with_default_params() -> Self {
        Self { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct SmmaInput<'a> {
    pub data: SmmaData<'a>,
    pub params: SmmaParams,
}

impl<'a> SmmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: SmmaParams) -> Self {
        Self {
            data: SmmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: SmmaParams) -> Self {
        Self {
            data: SmmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SmmaData::Candles {
                candles,
                source: "close",
            },
            params: SmmaParams::with_default_params(),
        }
    }
}

#[inline]
pub fn smma(input: &SmmaInput) -> Result<SmmaOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        SmmaData::Candles { candles, source } => source_type(candles, source),
        SmmaData::Slice(slice) => slice,
    };
    let len: usize = data.len();
    let period: usize = input.params.period.unwrap_or(7);

    if period == 0 || period > len {
        return Err("Invalid period specified for SMMA calculation.".into());
    }

    let mut smma_values = Vec::with_capacity(len);

    for _ in 0..(period - 1) {
        smma_values.push(f64::NAN);
    }

    let sum_first_period: f64 = data[..period].iter().sum();
    let first_smma = sum_first_period / (period as f64);
    smma_values.push(first_smma);

    let mut prev_smma = first_smma;
    for &value in &data[period..] {
        let new_smma = (prev_smma * (period as f64 - 1.0) + value) / (period as f64);
        smma_values.push(new_smma);
        prev_smma = new_smma;
    }

    Ok(SmmaOutput {
        values: smma_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_smma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = SmmaParams { period: None };
        let input_default = SmmaInput::from_candles(&candles, "close", default_params);
        let output_default = smma(&input_default).expect("Failed SMMA with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_custom = SmmaParams { period: Some(10) };
        let input_custom = SmmaInput::from_candles(&candles, "hl2", params_custom);
        let output_custom = smma(&input_custom).expect("Failed SMMA with period=10, source=hl2");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_smma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = SmmaParams { period: Some(7) };
        let input = SmmaInput::from_candles(&candles, "close", params);
        let result = smma(&input).expect("Failed to calculate SMMA");
        assert_eq!(result.values.len(), close_prices.len());

        let expected_last_five = [59434.4, 59398.2, 59346.9, 59319.4, 59224.5];
        assert!(result.values.len() >= 5);
        let start_index = result.values.len().saturating_sub(5);
        let actual_last_five = &result.values[start_index..];
        for (i, &actual) in actual_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-1,
                "SMMA mismatch at index {}: expected {}, got {}, diff={}",
                i,
                expected,
                actual,
                diff
            );
        }
    }
}
