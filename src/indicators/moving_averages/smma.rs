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

impl Default for SmmaParams {
    fn default() -> Self {
        Self { period: Some(7) }
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
            params: SmmaParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SmmaParams::default().period.unwrap())
    }
}

#[inline]
pub fn smma(input: &SmmaInput) -> Result<SmmaOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        SmmaData::Candles { candles, source } => source_type(candles, source),
        SmmaData::Slice(slice) => slice,
    };
    let first_valid_idx: usize = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err("All values in input data are NaN.".into()),
    };
    let len: usize = data.len();
    let period = input.get_period();
    if period == 0 || period > len {
        return Err("Invalid period specified for SMMA calculation.".into());
    }
    if (len - first_valid_idx) < period {
        return Err("Not enough valid data points to compute SMMA.".into());
    }
    if data[first_valid_idx..].iter().any(|&v| v.is_nan()) {
        return Err("NaN found in data after the first valid index.".into());
    }
    let mut smma_values: Vec<f64> = vec![f64::NAN; len];
    let start: usize = first_valid_idx;
    let end: usize = start + period;
    let sum_first_period: f64 = data[start..end].iter().sum();
    let first_smma: f64 = sum_first_period / period as f64;
    smma_values[end - 1] = first_smma;
    let mut prev_smma = first_smma;
    for i in end..len {
        let new_smma = (prev_smma * (period as f64 - 1.0) + data[i]) / (period as f64);
        smma_values[i] = new_smma;
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
    #[test]
    fn test_smma_params_with_default_params() {
        let params = SmmaParams::default();
        assert_eq!(params.period, Some(7));
    }

    #[test]
    fn test_smma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = SmmaInput::with_default_candles(&candles);
        match input.data {
            SmmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected SmmaData::Candles variant"),
        }
    }

    #[test]
    fn test_smma_zero_period() {
        let data = [10.0, 20.0, 30.0];
        let params = SmmaParams { period: Some(0) };
        let input = SmmaInput::from_slice(&data, params);
        let result = smma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_smma_period_exceeding_data_length() {
        let data = [10.0, 20.0, 30.0];
        let params = SmmaParams { period: Some(10) };
        let input = SmmaInput::from_slice(&data, params);
        let result = smma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_smma_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params_first = SmmaParams { period: Some(7) };
        let input_first = SmmaInput::from_candles(&candles, "close", params_first);
        let result_first = smma(&input_first).expect("Failed first SMMA");
        assert_eq!(result_first.values.len(), candles.close.len());
        let params_second = SmmaParams { period: Some(5) };
        let input_second = SmmaInput::from_slice(&result_first.values, params_second);
        let result_second = smma(&input_second).expect("Failed second SMMA");
        assert_eq!(result_second.values.len(), result_first.values.len());
        if result_second.values.len() > 240 {
            for i in 240..result_second.values.len() {
                assert!(result_second.values[i].is_finite());
            }
        }
    }

    #[test]
    fn test_smma_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = SmmaParams { period: Some(7) };
        let input = SmmaInput::from_candles(&candles, "close", params);
        let result = smma(&input).expect("Failed SMMA calculation");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
