use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum RsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RsiParams {
    pub period: Option<usize>,
}

impl Default for RsiParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct RsiInput<'a> {
    pub data: RsiData<'a>,
    pub params: RsiParams,
}

impl<'a> RsiInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: RsiParams) -> Self {
        Self {
            data: RsiData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: RsiParams) -> Self {
        Self {
            data: RsiData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: RsiData::Candles {
                candles,
                source: "close",
            },
            params: RsiParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| RsiParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct RsiOutput {
    pub values: Vec<f64>,
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RsiError {
    #[error("No data provided for RSI calculation.")]
    NoData,

    #[error("All values in input data are NaN for RSI calculation.")]
    AllValuesNaN,

    #[error("Not enough data points to compute RSI. Needed at least {needed}, found {found}")]
    NotEnoughData { needed: usize, found: usize },

    #[error("Invalid period specified for RSI calculation. period={period}, data_len={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
}

#[inline]
pub fn rsi(input: &RsiInput) -> Result<RsiOutput, RsiError> {
    let data: &[f64] = match &input.data {
        RsiData::Candles { candles, source } => source_type(candles, source),
        RsiData::Slice(slice) => slice,
    };
    let period = input.get_period();
    let len = data.len();
    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(RsiError::AllValuesNaN),
    };

    if len == 0 {
        return Err(RsiError::NoData);
    }
    if len < period {
        return Err(RsiError::NotEnoughData {
            needed: period,
            found: len,
        });
    }
    if period == 0 || period > len {
        return Err(RsiError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let mut rsi_values = vec![f64::NAN; len];

    let inv_period = 1.0 / period as f64;
    let beta = 1.0 - inv_period;

    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    for i in (first_valid_idx + 1)..=period {
        let delta = data[i] - data[i - 1];
        if delta > 0.0 {
            avg_gain += delta;
        } else {
            avg_loss += -delta;
        }
    }

    avg_gain *= inv_period;
    avg_loss *= inv_period;

    let initial_rsi = if avg_gain + avg_loss == 0.0 {
        50.0
    } else {
        100.0 * avg_gain / (avg_gain + avg_loss)
    };
    rsi_values[first_valid_idx + period] = initial_rsi;

    for i in (first_valid_idx + period + 1)..len {
        let delta = data[i] - data[i - 1];
        let gain = if delta > 0.0 { delta } else { 0.0 };
        let loss = if delta < 0.0 { -delta } else { 0.0 };

        avg_gain = inv_period * gain + beta * avg_gain;
        avg_loss = inv_period * loss + beta * avg_loss;

        let current_rsi = if avg_gain + avg_loss == 0.0 {
            50.0
        } else {
            100.0 * avg_gain / (avg_gain + avg_loss)
        };

        rsi_values[i] = current_rsi;
    }

    Ok(RsiOutput { values: rsi_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_rsi_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let partial_params = RsiParams { period: None };
        let input = RsiInput::from_candles(&candles, "close", partial_params);
        let result = rsi(&input).expect("Failed RSI with partial params");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_rsi_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = RsiParams { period: Some(14) };
        let input = RsiInput::from_candles(&candles, "close", params);
        let rsi_result = rsi(&input).expect("Failed to calculate RSI");

        let expected_last_five_rsi = [43.42, 42.68, 41.62, 42.86, 39.01];

        assert!(
            rsi_result.values.len() >= 5,
            "Not enough RSI values for the test"
        );

        assert_eq!(
            rsi_result.values.len(),
            close_prices.len(),
            "RSI values count should match input data count"
        );

        let start_index = rsi_result.values.len().saturating_sub(5);
        let result_last_five_rsi = &rsi_result.values[start_index..];

        for (i, &value) in result_last_five_rsi.iter().enumerate() {
            assert!(
                (value - expected_last_five_rsi[i]).abs() < 1e-2,
                "RSI value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five_rsi[i],
                value
            );
        }

        let default_input = RsiInput::with_default_candles(&candles);
        let default_rsi_result =
            rsi(&default_input).expect("Failed to calculate RSI with defaults");
        assert!(
            !default_rsi_result.values.is_empty(),
            "Should produce RSI values with default params"
        );
    }
    #[test]
    fn test_rsi_params_with_default_params() {
        let default_params = RsiParams::default();
        assert_eq!(default_params.period, Some(14));
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = RsiInput::from_candles(&candles, "close", default_params);
        let result = rsi(&input).expect("Failed RSI with default params");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_rsi_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = RsiInput::with_default_candles(&candles);
        match input.data {
            RsiData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected RsiData::Candles variant"),
        }
        let result = rsi(&input).expect("Failed RSI with default candles");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_rsi_with_zero_period() {
        let slice = [10.0, 11.0, 12.0];
        let params = RsiParams { period: Some(0) };
        let input = RsiInput::from_slice(&slice, params);
        let result = rsi(&input);
        assert!(result.is_err(), "Expected an error for zero period");
    }

    #[test]
    fn test_rsi_with_period_exceeding_data_length() {
        let slice = [10.0, 11.0, 12.0];
        let params = RsiParams { period: Some(10) };
        let input = RsiInput::from_slice(&slice, params);
        let result = rsi(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_rsi_very_small_data_set() {
        let slice = [42.0];
        let params = RsiParams { period: Some(14) };
        let input = RsiInput::from_slice(&slice, params);
        let result = rsi(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_rsi_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = RsiParams { period: Some(14) };
        let first_input = RsiInput::from_candles(&candles, "close", first_params);
        let first_result = rsi(&first_input).expect("Failed first RSI");
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = RsiParams { period: Some(5) };
        let second_input = RsiInput::from_slice(&first_result.values, second_params);
        let second_result = rsi(&second_input).expect("Failed second RSI");
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(
                    !second_result.values[i].is_nan(),
                    "Found NaN in RSI at {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_rsi_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = RsiParams { period: Some(14) };
        let input = RsiInput::from_candles(&candles, "close", params);
        let result = rsi(&input).expect("Failed RSI calculation");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan(), "Found NaN in RSI at {}", i);
            }
        }
    }
}
