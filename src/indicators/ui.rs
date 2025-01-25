use crate::indicators::utility_functions::max_rolling;
/// # Ulcer Index (UI)
///
/// The Ulcer Index (UI) is a volatility indicator that measures price drawdown from recent highs
/// and focuses on downside risk. It is calculated as the square root of the average of the squared
/// percentage drawdowns from the rolling maximum price within a specified window.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 14.
/// - **scalar**: The multiplier applied to the drawdown (defaults to 100).
///
/// ## Errors
/// - **EmptyData**: ui: Input data slice is empty.
/// - **InvalidPeriod**: ui: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: ui: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: ui: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(UiOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the UI window is filled.
/// - **`Err(UiError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum UiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct UiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct UiParams {
    pub period: Option<usize>,
    pub scalar: Option<f64>,
}

impl Default for UiParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            scalar: Some(100.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UiInput<'a> {
    pub data: UiData<'a>,
    pub params: UiParams,
}

impl<'a> UiInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: UiParams) -> Self {
        Self {
            data: UiData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: UiParams) -> Self {
        Self {
            data: UiData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: UiData::Candles {
                candles,
                source: "close",
            },
            params: UiParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| UiParams::default().period.unwrap())
    }

    pub fn get_scalar(&self) -> f64 {
        self.params
            .scalar
            .unwrap_or_else(|| UiParams::default().scalar.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum UiError {
    #[error("ui: Empty data provided.")]
    EmptyData,
    #[error("ui: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("ui: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ui: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn ui(input: &UiInput) -> Result<UiOutput, UiError> {
    let data: &[f64] = match &input.data {
        UiData::Candles { candles, source } => source_type(candles, source),
        UiData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(UiError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(UiError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let scalar = input.get_scalar();
    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(UiError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(UiError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let rolling_max = max_rolling(data, period).map_err(|_| UiError::NotEnoughValidData {
        needed: period,
        valid: data.len() - first_valid_idx,
    })?;

    let second_valid_idx = first_valid_idx + period - 1;

    let mut ui_values = vec![f64::NAN; data.len()];
    let mut squared_drawdowns = vec![f64::NAN; data.len()];

    for i in second_valid_idx..data.len() {
        if !rolling_max[i].is_nan() && !data[i].is_nan() {
            let dd = scalar * (data[i] - rolling_max[i]) / rolling_max[i];
            squared_drawdowns[i] = dd * dd;
        }
    }

    let mut sum_of_squares = 0.0;
    for i in second_valid_idx..(second_valid_idx + period) {
        sum_of_squares += squared_drawdowns[i];
    }
    ui_values[second_valid_idx + period - 1] = (sum_of_squares / period as f64).sqrt();

    for i in (second_valid_idx + period)..data.len() {
        sum_of_squares += squared_drawdowns[i] - squared_drawdowns[i - period];
        ui_values[i] = (sum_of_squares / period as f64).sqrt();
    }

    Ok(UiOutput { values: ui_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ui_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = UiParams {
            period: None,
            scalar: None,
        };
        let input_default = UiInput::from_candles(&candles, "close", default_params);
        let output_default = ui(&input_default).expect("Failed UI with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_10 = UiParams {
            period: Some(10),
            scalar: None,
        };
        let input_period_10 = UiInput::from_candles(&candles, "close", params_period_10);
        let output_period_10 = ui(&input_period_10).expect("Failed UI with period=10");
        assert_eq!(output_period_10.values.len(), candles.close.len());

        let params_custom = UiParams {
            period: Some(20),
            scalar: Some(50.0),
        };
        let input_custom = UiInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = ui(&input_custom).expect("Failed UI fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_ui_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = UiParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let input = UiInput::from_candles(&candles, "close", params);
        let ui_result = ui(&input).expect("Failed to calculate UI");

        assert_eq!(
            ui_result.values.len(),
            candles.close.len(),
            "UI length mismatch"
        );

        let expected_last_five_ui = [
            3.514342861283708,
            3.304986039846459,
            3.2011859814326304,
            3.1308860017483373,
            2.909612553474519,
        ];
        assert!(ui_result.values.len() >= 5, "UI length too short");
        let start_index = ui_result.values.len() - 5;
        let result_last_five_ui = &ui_result.values[start_index..];
        for (i, &value) in result_last_five_ui.iter().enumerate() {
            let expected_value = expected_last_five_ui[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "UI mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period = 14;
        for i in 0..(period - 1) {
            assert!(
                ui_result.values[i].is_nan(),
                "Expected leading NaNs before the first valid UI value"
            );
        }
    }

    #[test]
    fn test_ui_params_with_default_params() {
        let default_params = UiParams::default();
        assert_eq!(
            default_params.period,
            Some(14),
            "Expected default period to be 14"
        );
        assert_eq!(
            default_params.scalar,
            Some(100.0),
            "Expected default scalar to be 100.0"
        );
    }

    #[test]
    fn test_ui_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = UiInput::with_default_candles(&candles);
        match input.data {
            UiData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected UiData::Candles variant"),
        }
    }

    #[test]
    fn test_ui_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = UiParams {
            period: Some(0),
            scalar: None,
        };
        let input = UiInput::from_slice(&input_data, params);
        let result = ui(&input);
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
    fn test_ui_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = UiParams {
            period: Some(10),
            scalar: Some(100.0),
        };
        let input = UiInput::from_slice(&input_data, params);
        let result = ui(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_ui_very_small_data_set() {
        let input_data = [42.0];
        let params = UiParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let input = UiInput::from_slice(&input_data, params);
        let result = ui(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }
}
