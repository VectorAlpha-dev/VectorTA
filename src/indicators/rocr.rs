/// # Rate of Change Ratio (ROCR)
///
/// The Rate of Change Ratio (ROCR) measures the ratio between the current price
/// and the price `period` bars ago. This ratio is typically > 0 (for positive prices),
/// and is centered around 1. A value of 1 means no change, >1 means an increase,
/// and <1 indicates a decrease.
///
/// \[Formula\]
/// \[ ROCR[i] = price[i] / price[i - period] \]
///
/// ## Parameters
/// - **period**: The lookback window (number of data points). Defaults to 9.
///
/// ## Errors
/// - **EmptyData**: rocr: Input data slice is empty.
/// - **InvalidPeriod**: rocr: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: rocr: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: rocr: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(RocrOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the first valid ROCR value.
/// - **`Err(RocrError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum RocrData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RocrOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RocrParams {
    pub period: Option<usize>,
}

impl Default for RocrParams {
    fn default() -> Self {
        // Default period = 9
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct RocrInput<'a> {
    pub data: RocrData<'a>,
    pub params: RocrParams,
}

impl<'a> RocrInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: RocrParams) -> Self {
        Self {
            data: RocrData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: RocrParams) -> Self {
        Self {
            data: RocrData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: RocrData::Candles {
                candles,
                source: "close",
            },
            params: RocrParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| RocrParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum RocrError {
    #[error("rocr: Empty data provided.")]
    EmptyData,
    #[error("rocr: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("rocr: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("rocr: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn rocr(input: &RocrInput) -> Result<RocrOutput, RocrError> {
    let data: &[f64] = match &input.data {
        RocrData::Candles { candles, source } => source_type(candles, source),
        RocrData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(RocrError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(RocrError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(RocrError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(RocrError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut rocr_values = vec![f64::NAN; data.len()];

    let start_idx = first_valid_idx + period;
    for i in start_idx..data.len() {
        let current = data[i];
        let past = data[i - period];
        if past == 0.0 || past.is_nan() {
            rocr_values[i] = 0.0;
        } else {
            rocr_values[i] = current / past;
        }
    }

    Ok(RocrOutput {
        values: rocr_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_rocr_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = RocrParams { period: None };
        let input_default = RocrInput::from_candles(&candles, "close", default_params);
        let output_default = rocr(&input_default).expect("Failed ROCR with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = RocrParams { period: Some(14) };
        let input_period_14 = RocrInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 =
            rocr(&input_period_14).expect("Failed ROCR with period=14, source=hl2");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = RocrParams { period: Some(20) };
        let input_custom = RocrInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = rocr(&input_custom).expect("Failed ROCR fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_rocr_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = RocrParams { period: Some(10) };
        let input = RocrInput::from_candles(&candles, "close", params);
        let rocr_result = rocr(&input).expect("Failed to calculate ROCR");

        assert_eq!(
            rocr_result.values.len(),
            close_prices.len(),
            "ROCR length mismatch"
        );

        let expected_last_five = [
            0.9977448290950706,
            0.9944380965183492,
            0.9967247986764135,
            0.9950545846019277,
            0.984954072979463,
        ];
        assert!(
            rocr_result.values.len() >= 5,
            "Not enough data for the final five checks"
        );
        let start_idx = rocr_result.values.len() - 5;
        let actual_last_five = &rocr_result.values[start_idx..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-8,
                "ROCR mismatch at index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        let period = input.get_period();
        for i in 0..(period - 1) {
            assert!(
                rocr_result.values[i].is_nan(),
                "Expected leading NaN at index {}",
                i
            );
        }

        let default_input = RocrInput::with_default_candles(&candles);
        let default_rocr = rocr(&default_input).expect("Failed default ROCR");
        assert_eq!(default_rocr.values.len(), close_prices.len());
    }

    #[test]
    fn test_rocr_params_with_default_params() {
        let default_params = RocrParams::default();
        assert_eq!(
            default_params.period,
            Some(9),
            "Expected period=9 in default params"
        );
    }

    #[test]
    fn test_rocr_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = RocrInput::with_default_candles(&candles);
        match input.data {
            RocrData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected RocrData::Candles variant"),
        }
    }

    #[test]
    fn test_rocr_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = RocrParams { period: Some(0) };
        let input = RocrInput::from_slice(&input_data, params);

        let result = rocr(&input);
        assert!(result.is_err(), "Expected error for zero period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_rocr_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = RocrParams { period: Some(10) };
        let input = RocrInput::from_slice(&input_data, params);

        let result = rocr(&input);
        assert!(result.is_err(), "Expected error for period > data.len()");
    }

    #[test]
    fn test_rocr_very_small_data_set() {
        let input_data = [42.0];
        let params = RocrParams { period: Some(9) };
        let input = RocrInput::from_slice(&input_data, params);

        let result = rocr(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_rocr_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = RocrParams { period: Some(14) };
        let first_input = RocrInput::from_candles(&candles, "close", first_params);
        let first_result = rocr(&first_input).expect("Failed first ROCR calculation");
        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First ROCR output length mismatch"
        );

        let second_params = RocrParams { period: Some(14) };
        let second_input = RocrInput::from_slice(&first_result.values, second_params);
        let second_result = rocr(&second_input).expect("Failed second ROCR calculation");
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second ROCR output length mismatch"
        );

        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 28, but found NaN at index {}",
                i
            );
        }
    }

    #[test]
    fn test_rocr_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 9;
        let params = RocrParams {
            period: Some(period),
        };
        let input = RocrInput::from_candles(&candles, "close", params);
        let rocr_result = rocr(&input).expect("Failed to calculate ROCR for nan-check");

        assert_eq!(
            rocr_result.values.len(),
            close_prices.len(),
            "ROCR length mismatch"
        );

        if rocr_result.values.len() > 240 {
            for i in 240..rocr_result.values.len() {
                assert!(
                    !rocr_result.values[i].is_nan(),
                    "Expected no NaN after index 240, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
