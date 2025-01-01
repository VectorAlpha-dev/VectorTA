use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum WmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct WmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WmaParams {
    pub period: Option<usize>,
}

impl Default for WmaParams {
    fn default() -> Self {
        Self { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct WmaInput<'a> {
    pub data: WmaData<'a>,
    pub params: WmaParams,
}

impl<'a> WmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: WmaParams) -> Self {
        Self {
            data: WmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: WmaParams) -> Self {
        Self {
            data: WmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: WmaData::Candles {
                candles,
                source: "close",
            },
            params: WmaParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| WmaParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WmaError {
    #[error("Data slice is empty, cannot compute WMA.")]
    EmptyData,
    #[error("Period {period} is greater than data length {data_len}.")]
    PeriodExceedsDataLen { period: usize, data_len: usize },
    #[error("Invalid period for WMA calculation, must be >= 2. period = {period}")]
    InvalidPeriod { period: usize },
}

#[inline]
pub fn wma(input: &WmaInput) -> Result<WmaOutput, WmaError> {
    let data: &[f64] = match &input.data {
        WmaData::Candles { candles, source } => source_type(candles, source),
        WmaData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(WmaError::EmptyData);
    }

    let len = data.len();
    let period = input.get_period();

    if period > len {
        return Err(WmaError::PeriodExceedsDataLen {
            period,
            data_len: len,
        });
    }

    if period < 2 {
        return Err(WmaError::InvalidPeriod { period });
    }

    let mut values = vec![f64::NAN; len];

    let lookback = period - 1;
    let sum_of_weights = (period * (period + 1)) >> 1;
    let divider = sum_of_weights as f64;

    let mut weighted_sum = 0.0;
    let mut plain_sum = 0.0;

    for (i, &val) in data.iter().take(lookback).enumerate() {
        weighted_sum += (i as f64 + 1.0) * val;
        plain_sum += val;
    }

    for i in lookback..len {
        let val = data[i];
        weighted_sum += (period as f64) * val;
        plain_sum += val;
        values[i] = weighted_sum / divider;
        weighted_sum -= plain_sum;
        let old_val = data[i - lookback];
        plain_sum -= old_val;
    }
    Ok(WmaOutput { values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_wma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = WmaParams { period: None };
        let input = WmaInput::from_candles(&candles, "close", default_params);
        let output = wma(&input).expect("Failed WMA with default params");
        assert_eq!(output.values.len(), candles.close.len());

        let params_period_14 = WmaParams { period: Some(14) };
        let input2 = WmaInput::from_candles(&candles, "hl2", params_period_14);
        let output2 = wma(&input2).expect("Failed WMA with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());

        let params_custom = WmaParams { period: Some(20) };
        let input3 = WmaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = wma(&input3).expect("Failed WMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }

    #[test]
    fn test_wma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let data = &candles.close;
        let default_params = WmaParams::default();
        let input = WmaInput::from_candles(&candles, "close", default_params);
        let result = wma(&input).expect("Failed to calculate WMA");

        let expected_last_five = [
            59638.52903225806,
            59563.7376344086,
            59489.4064516129,
            59432.02580645162,
            59350.58279569892,
        ];

        assert!(result.values.len() >= 5, "Not enough WMA values");
        assert_eq!(
            result.values.len(),
            data.len(),
            "WMA output length should match input length"
        );

        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];

        for (i, &value) in last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-6,
                "WMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }

        let period = input.params.period.unwrap_or(30);
        for val in result.values.iter().skip(period - 1) {
            if !val.is_nan() {
                assert!(val.is_finite(), "WMA output should be finite");
            }
        }
    }
    #[test]
    fn test_wma_params_with_default() {
        let default_params = WmaParams::default();
        assert_eq!(default_params.period, Some(30));
    }

    #[test]
    fn test_wma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = WmaInput::with_default_candles(&candles);
        match input.data {
            WmaData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected WmaData::Candles variant"),
        }
    }

    #[test]
    fn test_wma_with_zero_or_one_period() {
        let data = [10.0, 20.0, 30.0];
        for &p in &[0, 1] {
            let params = WmaParams { period: Some(p) };
            let input = WmaInput::from_slice(&data, params);
            let result = wma(&input);
            assert!(result.is_err());
            if let Err(e) = result {
                assert!(
                    e.to_string().contains("Invalid period"),
                    "Unexpected error: {}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_wma_with_period_exceeding_data_length() {
        let data = [10.0, 20.0, 30.0];
        let params = WmaParams { period: Some(5) };
        let input = WmaInput::from_slice(&data, params);
        let result = wma(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("greater than data length"),
                "Unexpected error: {}",
                e
            );
        }
    }

    #[test]
    fn test_wma_very_small_data_set() {
        let data = [42.0, 50.0];
        let params = WmaParams { period: Some(2) };
        let input = WmaInput::from_slice(&data, params);
        let result = wma(&input).expect("Should handle data with exact period length");
        assert_eq!(result.values.len(), data.len());
    }

    #[test]
    fn test_wma_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = WmaParams { period: Some(14) };
        let first_input = WmaInput::from_candles(&candles, "close", first_params);
        let first_result = wma(&first_input).expect("Failed to calculate first WMA");
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = WmaParams { period: Some(5) };
        let second_input = WmaInput::from_slice(&first_result.values, second_params);
        let second_result = wma(&second_input).expect("Failed to calculate second WMA");
        assert_eq!(second_result.values.len(), first_result.values.len());
        for val in &second_result.values[240..] {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_wma_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = WmaParams { period: Some(14) };
        let input = WmaInput::from_candles(&candles, "close", params);
        let result = wma(&input).expect("Failed to calculate WMA");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 50 {
            for i in 50..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
