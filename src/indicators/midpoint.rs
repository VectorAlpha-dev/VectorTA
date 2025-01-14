/// # Midpoint
///
/// The midpoint indicator calculates the midpoint of the highest and lowest values
/// over a specified window (period). It is defined as:
/// \
/// MIDPOINT = (Highest Value + Lowest Value) / 2
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: midpoint: Input data slice is empty.
/// - **InvalidPeriod**: midpoint: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: midpoint: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: midpoint: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(MidpointOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the period window is filled.
/// - **`Err(MidpointError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MidpointData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MidpointOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MidpointParams {
    pub period: Option<usize>,
}

impl Default for MidpointParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct MidpointInput<'a> {
    pub data: MidpointData<'a>,
    pub params: MidpointParams,
}

impl<'a> MidpointInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: MidpointParams) -> Self {
        Self {
            data: MidpointData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: MidpointParams) -> Self {
        Self {
            data: MidpointData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MidpointData::Candles {
                candles,
                source: "close",
            },
            params: MidpointParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| MidpointParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum MidpointError {
    #[error("midpoint: Empty data provided.")]
    EmptyData,
    #[error("midpoint: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("midpoint: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("midpoint: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn midpoint(input: &MidpointInput) -> Result<MidpointOutput, MidpointError> {
    let data: &[f64] = match &input.data {
        MidpointData::Candles { candles, source } => source_type(candles, source),
        MidpointData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(MidpointError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(MidpointError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(MidpointError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(MidpointError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut midpoint_values = vec![f64::NAN; data.len()];

    for i in (first_valid_idx + period - 1)..data.len() {
        let window = &data[(i + 1 - period)..=i];
        let mut highest = f64::MIN;
        let mut lowest = f64::MAX;
        for &val in window {
            if val > highest {
                highest = val;
            }
            if val < lowest {
                lowest = val;
            }
        }
        midpoint_values[i] = (highest + lowest) / 2.0;
    }

    Ok(MidpointOutput {
        values: midpoint_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_midpoint_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = MidpointParams { period: None };
        let input_default = MidpointInput::from_candles(&candles, "close", default_params);
        let output_default = midpoint(&input_default).expect("Failed MIDPOINT with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_10 = MidpointParams { period: Some(10) };
        let input_period_10 = MidpointInput::from_candles(&candles, "hl2", params_period_10);
        let output_period_10 =
            midpoint(&input_period_10).expect("Failed MIDPOINT with period=10, source=hl2");
        assert_eq!(output_period_10.values.len(), candles.close.len());

        let params_custom = MidpointParams { period: Some(20) };
        let input_custom = MidpointInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = midpoint(&input_custom).expect("Failed MIDPOINT fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_midpoint_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = MidpointParams { period: Some(14) };
        let input = MidpointInput::from_candles(&candles, "close", params);
        let midpoint_result = midpoint(&input).expect("Failed to calculate MIDPOINT");

        assert_eq!(
            midpoint_result.values.len(),
            close_prices.len(),
            "MIDPOINT length mismatch"
        );

        let expected_last_five_midpoint = [59578.5, 59578.5, 59578.5, 58886.0, 58886.0];
        assert!(
            midpoint_result.values.len() >= 5,
            "MIDPOINT length too short"
        );
        let start_index = midpoint_result.values.len() - 5;
        let result_last_five_midpoint = &midpoint_result.values[start_index..];
        for (i, &value) in result_last_five_midpoint.iter().enumerate() {
            let expected_value = expected_last_five_midpoint[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "MIDPOINT mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period: usize = 14;
        for i in 0..(period - 1) {
            assert!(
                midpoint_result.values[i].is_nan(),
                "Expected NaN for leading values in MIDPOINT"
            );
        }

        let default_input = MidpointInput::with_default_candles(&candles);
        let default_midpoint_result =
            midpoint(&default_input).expect("Failed to calculate MIDPOINT defaults");
        assert_eq!(
            default_midpoint_result.values.len(),
            close_prices.len(),
            "Default MIDPOINT length mismatch"
        );
    }

    #[test]
    fn test_midpoint_params_with_default_params() {
        let default_params = MidpointParams::default();
        assert_eq!(
            default_params.period,
            Some(14),
            "Expected period to be Some(14) in default parameters"
        );
    }

    #[test]
    fn test_midpoint_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = MidpointInput::with_default_candles(&candles);
        match input.data {
            MidpointData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected MidpointData::Candles variant"),
        }
    }

    #[test]
    fn test_midpoint_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = MidpointParams { period: Some(0) };
        let input = MidpointInput::from_slice(&input_data, params);

        let result = midpoint(&input);
        assert!(result.is_err(), "Expected an error for zero period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_midpoint_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = MidpointParams { period: Some(10) };
        let input = MidpointInput::from_slice(&input_data, params);

        let result = midpoint(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_midpoint_very_small_data_set() {
        let input_data = [42.0];
        let params = MidpointParams { period: Some(9) };
        let input = MidpointInput::from_slice(&input_data, params);

        let result = midpoint(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_midpoint_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = MidpointParams { period: Some(14) };
        let first_input = MidpointInput::from_candles(&candles, "close", first_params);
        let first_result = midpoint(&first_input).expect("Failed to calculate first MIDPOINT");
        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First MIDPOINT output length mismatch"
        );

        let second_params = MidpointParams { period: Some(14) };
        let second_input = MidpointInput::from_slice(&first_result.values, second_params);
        let second_result = midpoint(&second_input).expect("Failed to calculate second MIDPOINT");
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second MIDPOINT output length mismatch"
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
    fn test_midpoint_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 14;
        let params = MidpointParams {
            period: Some(period),
        };
        let input = MidpointInput::from_candles(&candles, "close", params);
        let midpoint_result = midpoint(&input).expect("Failed to calculate MIDPOINT");

        assert_eq!(
            midpoint_result.values.len(),
            close_prices.len(),
            "MIDPOINT length mismatch"
        );

        if midpoint_result.values.len() > 240 {
            for i in 240..midpoint_result.values.len() {
                assert!(
                    !midpoint_result.values[i].is_nan(),
                    "Expected no NaN after index 240, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
