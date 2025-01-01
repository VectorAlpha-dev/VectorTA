use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum SrwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

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
    pub data: SrwmaData<'a>,
    pub params: SrwmaParams,
}

impl<'a> SrwmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: SrwmaParams) -> Self {
        Self {
            data: SrwmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: SrwmaParams) -> Self {
        Self {
            data: SrwmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SrwmaData::Candles {
                candles,
                source: "close",
            },
            params: SrwmaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SrwmaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct SrwmaOutput {
    pub values: Vec<f64>,
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SrwmaError {
    #[error("Data slice is empty for SRWMA calculation.")]
    EmptyData,
    #[error("Invalid period for SRWMA calculation. period = {period}")]
    InvalidPeriod { period: usize },
}

#[inline]
pub fn srwma(input: &SrwmaInput) -> Result<SrwmaOutput, SrwmaError> {
    let data: &[f64] = match &input.data {
        SrwmaData::Candles { candles, source } => source_type(candles, source),
        SrwmaData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(SrwmaError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 {
        return Err(SrwmaError::InvalidPeriod { period });
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
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close");
        let params = SrwmaParams { period: Some(14) };
        let input = SrwmaInput::from_candles(&candles, "close", params);
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
        let input = SrwmaInput::from_candles(&candles, "close", default_params);
        let output = srwma(&input).expect("Failed SRWMA with default params");
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_14 = SrwmaParams { period: Some(14) };
        let input2 = SrwmaInput::from_candles(&candles, "hl2", params_period_14);
        let output2 = srwma(&input2).expect("Failed SRWMA with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = SrwmaParams { period: Some(10) };
        let input3 = SrwmaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = srwma(&input3).expect("Failed SRWMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }
    #[test]
    fn test_srwma_params_with_default() {
        let default_params = SrwmaParams::default();
        assert_eq!(default_params.period, Some(14));
    }

    #[test]
    fn test_srwma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = SrwmaInput::with_default_candles(&candles);
        match input.data {
            SrwmaData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected SrwmaData::Candles variant"),
        }
        assert_eq!(input.params.period, Some(14));
    }

    #[test]
    fn test_srwma_with_zero_period() {
        let data = [10.0, 20.0, 30.0, 40.0];
        let params = SrwmaParams { period: Some(0) };
        let input = SrwmaInput::from_slice(&data, params);
        let result = srwma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_srwma_with_period_exceeding_data_length() {
        let data = [10.0, 20.0, 30.0];
        let params = SrwmaParams { period: Some(10) };
        let input = SrwmaInput::from_slice(&data, params);
        let result = srwma(&input).expect("Should handle period > data.len()");
        assert_eq!(result.values, data);
    }

    #[test]
    fn test_srwma_very_small_data_set() {
        let data = [42.0, 52.0];
        let params = SrwmaParams { period: Some(3) };
        let input = SrwmaInput::from_slice(&data, params);
        let result = srwma(&input).expect("Should handle data smaller than period");
        assert_eq!(result.values.len(), data.len());
        assert_eq!(result.values, data);
    }

    #[test]
    fn test_srwma_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = SrwmaParams { period: Some(14) };
        let first_input = SrwmaInput::from_candles(&candles, "close", first_params);
        let first_result = srwma(&first_input).expect("Failed to calculate first SRWMA");
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = SrwmaParams { period: Some(5) };
        let second_input = SrwmaInput::from_slice(&first_result.values, second_params);
        let second_result = srwma(&second_input).expect("Failed to calculate second SRWMA");
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 240..second_result.values.len() {
            assert!(second_result.values[i].is_finite());
        }
    }

    #[test]
    fn test_srwma_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = SrwmaParams { period: Some(14) };
        let input = SrwmaInput::from_candles(&candles, "close", params);
        let result = srwma(&input).expect("Failed to calculate SRWMA");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 50 {
            for i in 50..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
