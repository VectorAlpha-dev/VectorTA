use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum EmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

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
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct EmaInput<'a> {
    pub data: EmaData<'a>,
    pub params: EmaParams,
}

impl<'a> EmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: EmaParams) -> Self {
        Self {
            data: EmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: EmaParams) -> Self {
        Self {
            data: EmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: EmaData::Candles {
                candles,
                source: "close",
            },
            params: EmaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| EmaParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmaError {
    #[error("All input data are NaN.")]
    AllValuesNaN,
    #[error("Invalid period for EMA: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("Not enough valid data to compute EMA: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn ema(input: &EmaInput) -> Result<EmaOutput, EmaError> {
    let data: &[f64] = match &input.data {
        EmaData::Candles { candles, source } => source_type(candles, source),
        EmaData::Slice(slice) => slice,
    };
    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(EmaError::AllValuesNaN),
    };
    let len: usize = data.len();
    let period: usize = input.get_period();

    if period == 0 || period > len {
        return Err(EmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    if (len - first_valid_idx) < period {
        return Err(EmaError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }

    let mut ema_values = vec![f64::NAN; len];
    let alpha = 2.0 / (period as f64 + 1.0);
    ema_values[first_valid_idx] = data[first_valid_idx];
    for i in (first_valid_idx + 1)..len {
        ema_values[i] = alpha * data[i] + (1.0 - alpha) * ema_values[i - 1];
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
        let candles = read_candles_from_csv(file_path).unwrap();
        let close_prices = &candles.close;
        let params = EmaParams { period: Some(9) };
        let input = EmaInput::from_candles(&candles, "close", params);
        let result = ema(&input).unwrap();
        let expected_last_five = [59302.2, 59277.9, 59230.2, 59215.1, 59103.1];
        assert!(result.values.len() >= 5, "Result length is less than 5.");
        assert_eq!(
            result.values.len(),
            close_prices.len(),
            "Result length is not equal to close prices length."
        );
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            assert!(
                (val - expected_last_five[i]).abs() < 1e-1,
                "Mismatch at index {}.",
                i
            );
        }
        let default_input = EmaInput::with_default_candles(&candles);
        let default_result = ema(&default_input).unwrap();
        assert!(
            !default_result.values.is_empty(),
            "Default result is empty."
        );
    }

    #[test]
    fn test_ema_params_with_default_period() {
        let params = EmaParams::default();
        assert_eq!(params.period, Some(9), "Default period is not 9.");
    }

    #[test]
    fn test_ema_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let input = EmaInput::with_default_candles(&candles);
        match input.data {
            EmaData::Candles { source, .. } => assert_eq!(source, "close", "Source is not close."),
            _ => panic!(),
        }
        assert_eq!(input.params.period, Some(9), "Period is not 9.");
    }

    #[test]
    fn test_ema_with_zero_period() {
        let data = [10.0, 20.0, 30.0];
        let params = EmaParams { period: Some(0) };
        let input = EmaInput::from_slice(&data, params);
        let result = ema(&input);
        assert!(result.is_err(), "Result is not an error.");
    }

    #[test]
    fn test_ema_period_exceeding_data_length() {
        let data = [10.0, 20.0, 30.0];
        let params = EmaParams { period: Some(10) };
        let input = EmaInput::from_slice(&data, params);
        let result = ema(&input);
        assert!(result.is_err(), "Result is not an error.");
    }

    #[test]
    fn test_ema_very_small_data_set() {
        let data = [42.0];
        let params = EmaParams { period: Some(9) };
        let input = EmaInput::from_slice(&data, params);
        let result = ema(&input);
        assert!(result.is_err(), "Result is not an error.");
    }

    #[test]
    fn test_ema_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let params_first = EmaParams { period: Some(9) };
        let input_first = EmaInput::from_candles(&candles, "close", params_first);
        let result_first = ema(&input_first).unwrap();
        assert_eq!(
            result_first.values.len(),
            candles.close.len(),
            "Result length mismatch."
        );
        let params_second = EmaParams { period: Some(5) };
        let input_second = EmaInput::from_slice(&result_first.values, params_second);
        let result_second = ema(&input_second).unwrap();
        assert_eq!(
            result_second.values.len(),
            result_first.values.len(),
            "Result length mismatch."
        );
        if result_second.values.len() > 240 {
            for i in 240..result_second.values.len() {
                assert!(
                    !result_second.values[i].is_nan(),
                    "Found NaN at index {}.",
                    i
                );
            }
        }
    }

    #[test]
    fn test_ema_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let params = EmaParams { period: Some(9) };
        let input = EmaInput::from_candles(&candles, "close", params);
        let result = ema(&input).unwrap();
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "Result length mismatch."
        );
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan(), "Found NaN at index {}.", i);
            }
        }
    }
}
