/// # Rate of Change Percentage (ROCP)
///
/// The Rate of Change Percentage (ROCP) calculates the relative change in value between
/// the current price and the price `period` bars ago:
///
/// \[ ROCP[i] = (price[i] - price[i - period]) / price[i - period] \]
///
/// ROCP is often interpreted as a normalized momentum measure centered around zero.
/// Positive values indicate increasing price, negative values indicate a decrease.
///
/// ## Parameters
/// - **period**: The lookback window (number of data points). Defaults to 9.
///
/// ## Errors
/// - **EmptyData**: rocp: Input data slice is empty.
/// - **InvalidPeriod**: rocp: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: rocp: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: rocp: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(RocpOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the first valid ROCP value.
/// - **`Err(RocpError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum RocpData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RocpOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RocpParams {
    pub period: Option<usize>,
}

impl Default for RocpParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct RocpInput<'a> {
    pub data: RocpData<'a>,
    pub params: RocpParams,
}

impl<'a> RocpInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: RocpParams) -> Self {
        Self {
            data: RocpData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: RocpParams) -> Self {
        Self {
            data: RocpData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: RocpData::Candles {
                candles,
                source: "close",
            },
            params: RocpParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| RocpParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum RocpError {
    #[error("rocp: Empty data provided.")]
    EmptyData,
    #[error("rocp: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("rocp: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("rocp: All values are NaN.")]
    AllValuesNaN,
}

pub fn rate_of_change_percentage(input: &RocpInput) -> Result<RocpOutput, RocpError> {
    let data: &[f64] = match &input.data {
        RocpData::Candles { candles, source } => source_type(candles, source),
        RocpData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(RocpError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(RocpError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(RocpError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(RocpError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut rocp_values = vec![f64::NAN; data.len()];

    let start_idx = first_valid_idx + period;
    for i in start_idx..data.len() {
        let current = data[i];
        let past = data[i - period];
        if past == 0.0 || past.is_nan() {
            rocp_values[i] = 0.0;
        } else {
            rocp_values[i] = (current - past) / past;
        }
    }

    Ok(RocpOutput {
        values: rocp_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_rocp_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = RocpParams { period: None };
        let input_default = RocpInput::from_candles(&candles, "close", default_params);
        let output_default =
            rate_of_change_percentage(&input_default).expect("Failed ROCP with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = RocpParams { period: Some(14) };
        let input_period_14 = RocpInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 = rate_of_change_percentage(&input_period_14)
            .expect("Failed ROCP with period=14, source=hl2");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = RocpParams { period: Some(20) };
        let input_custom = RocpInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom =
            rate_of_change_percentage(&input_custom).expect("Failed ROCP fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_rocp_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = RocpParams { period: Some(10) };
        let input = RocpInput::from_candles(&candles, "close", params);
        let rocp_result = rate_of_change_percentage(&input).expect("Failed to calculate ROCP");

        assert_eq!(
            rocp_result.values.len(),
            close_prices.len(),
            "ROCP length mismatch"
        );

        let expected_last_five = [
            -0.0022551709049293996,
            -0.005561903481650759,
            -0.003275201323586514,
            -0.004945415398072297,
            -0.015045927020537019,
        ];
        assert!(
            rocp_result.values.len() >= 5,
            "Not enough output data to check last five"
        );
        let start_idx = rocp_result.values.len() - 5;
        let actual_last_five = &rocp_result.values[start_idx..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-9,
                "ROCP mismatch at index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }

        let period = input.get_period();
        for i in 0..(period - 1) {
            assert!(
                rocp_result.values[i].is_nan(),
                "Expected leading NaN at index {}",
                i
            );
        }

        let default_input = RocpInput::with_default_candles(&candles);
        let default_rocp_result =
            rate_of_change_percentage(&default_input).expect("Failed to compute ROCP defaults");
        assert_eq!(
            default_rocp_result.values.len(),
            close_prices.len(),
            "Default input mismatch"
        );
    }

    #[test]
    fn test_rocp_params_with_default_params() {
        let default_params = RocpParams::default();
        assert_eq!(
            default_params.period,
            Some(9),
            "Expected period=9 in default params"
        );
    }

    #[test]
    fn test_rocp_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = RocpInput::with_default_candles(&candles);
        match input.data {
            RocpData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected RocpData::Candles variant"),
        }
    }

    #[test]
    fn test_rocp_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = RocpParams { period: Some(0) };
        let input = RocpInput::from_slice(&input_data, params);

        let result = rate_of_change_percentage(&input);
        assert!(result.is_err(), "Expected error for zero period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' message, got {}",
                e
            );
        }
    }

    #[test]
    fn test_rocp_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = RocpParams { period: Some(10) };
        let input = RocpInput::from_slice(&input_data, params);

        let result = rate_of_change_percentage(&input);
        assert!(result.is_err(), "Expected error for period > data.len()");
    }

    #[test]
    fn test_rocp_very_small_data_set() {
        let input_data = [42.0];
        let params = RocpParams { period: Some(9) };
        let input = RocpInput::from_slice(&input_data, params);

        let result = rate_of_change_percentage(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_rocp_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = RocpParams { period: Some(14) };
        let first_input = RocpInput::from_candles(&candles, "close", first_params);
        let first_result =
            rate_of_change_percentage(&first_input).expect("Failed first ROCP calculation");
        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First ROCP output length mismatch"
        );

        let second_params = RocpParams { period: Some(14) };
        let second_input = RocpInput::from_slice(&first_result.values, second_params);
        let second_result =
            rate_of_change_percentage(&second_input).expect("Failed second ROCP calculation");
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second ROCP output length mismatch"
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
    fn test_rocp_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 9;
        let params = RocpParams {
            period: Some(period),
        };
        let input = RocpInput::from_candles(&candles, "close", params);
        let rocp_result = rate_of_change_percentage(&input).expect("Failed to calculate ROCP");

        assert_eq!(
            rocp_result.values.len(),
            close_prices.len(),
            "ROCP length mismatch"
        );

        if rocp_result.values.len() > 240 {
            for i in 240..rocp_result.values.len() {
                assert!(
                    !rocp_result.values[i].is_nan(),
                    "Expected no NaN after index 240, found NaN at index {}",
                    i
                );
            }
        }
    }
}
