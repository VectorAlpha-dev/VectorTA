/// # Rate of Change (ROC)
///
/// The Rate of Change (ROC) measures the percentage change in price between
/// the current bar and the bar `period` bars ago. This is often used to identify
/// momentum shifts and overbought/oversold conditions.
///
/// \[Formula\]
/// \[ ROC[i] = ((price[i] / price[i - period]) - 1) * 100 \]
///
/// ## Parameters
/// - **period**: The lookback window (number of data points). Defaults to 9.
///
/// ## Errors
/// - **EmptyData**: roc: Input data slice is empty.
/// - **InvalidPeriod**: roc: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: roc: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: roc: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(RocOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the first valid ROC value.
/// - **`Err(RocError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum RocData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RocOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RocParams {
    pub period: Option<usize>,
}

impl Default for RocParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct RocInput<'a> {
    pub data: RocData<'a>,
    pub params: RocParams,
}

impl<'a> RocInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: RocParams) -> Self {
        Self {
            data: RocData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: RocParams) -> Self {
        Self {
            data: RocData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: RocData::Candles {
                candles,
                source: "close",
            },
            params: RocParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| RocParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum RocError {
    #[error("roc: Empty data provided.")]
    EmptyData,
    #[error("roc: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("roc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("roc: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn roc(input: &RocInput) -> Result<RocOutput, RocError> {
    let data: &[f64] = match &input.data {
        RocData::Candles { candles, source } => source_type(candles, source),
        RocData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(RocError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(RocError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(RocError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(RocError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut roc_values = vec![f64::NAN; data.len()];

    let start_idx = first_valid_idx + period;
    if start_idx > data.len() - 1 {
        return Err(RocError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    for i in start_idx..data.len() {
        let current = data[i];
        let prev = data[i - period];
        if prev == 0.0 || prev.is_nan() {
            roc_values[i] = 0.0;
        } else {
            roc_values[i] = ((current / prev) - 1.0) * 100.0;
        }
    }

    Ok(RocOutput { values: roc_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_roc_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = RocParams { period: None };
        let input_default = RocInput::from_candles(&candles, "close", default_params);
        let output_default = roc(&input_default).expect("Failed ROC with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = RocParams { period: Some(14) };
        let input_period_14 = RocInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 =
            roc(&input_period_14).expect("Failed ROC with period=14, source=hl2");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = RocParams { period: Some(20) };
        let input_custom = RocInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = roc(&input_custom).expect("Failed ROC fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_roc_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = RocParams { period: Some(10) };
        let input = RocInput::from_candles(&candles, "close", params);
        let roc_result = roc(&input).expect("Failed to calculate ROC");

        assert_eq!(
            roc_result.values.len(),
            close_prices.len(),
            "ROC length mismatch"
        );

        let expected_last_five_roc = [
            -0.22551709049294377,
            -0.5561903481650754,
            -0.32752013235864963,
            -0.49454153980722504,
            -1.5045927020536976,
        ];
        assert!(
            roc_result.values.len() >= 5,
            "Not enough output data to check last five"
        );
        let start_index = roc_result.values.len() - 5;
        let result_last_five_roc = &roc_result.values[start_index..];
        for (i, &value) in result_last_five_roc.iter().enumerate() {
            let expected_value = expected_last_five_roc[i];
            assert!(
                (value - expected_value).abs() < 1e-7,
                "ROC mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period = input.get_period();
        for i in 0..(period - 1) {
            assert!(
                roc_result.values[i].is_nan(),
                "Expected leading NaN at index {}",
                i
            );
        }

        let default_input = RocInput::with_default_candles(&candles);
        let default_roc_result = roc(&default_input).expect("Failed to calculate ROC");
        assert_eq!(
            default_roc_result.values.len(),
            close_prices.len(),
            "Default ROC length mismatch"
        );
    }

    #[test]
    fn test_roc_params_with_default_params() {
        let default_params = RocParams::default();
        assert_eq!(
            default_params.period,
            Some(9),
            "Expected period=9 in default parameters"
        );
    }

    #[test]
    fn test_roc_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = RocInput::with_default_candles(&candles);
        match input.data {
            RocData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected RocData::Candles variant"),
        }
    }

    #[test]
    fn test_roc_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = RocParams { period: Some(0) };
        let input = RocInput::from_slice(&input_data, params);

        let result = roc(&input);
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
    fn test_roc_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = RocParams { period: Some(10) };
        let input = RocInput::from_slice(&input_data, params);

        let result = roc(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_roc_very_small_data_set() {
        let input_data = [42.0];
        let params = RocParams { period: Some(9) };
        let input = RocInput::from_slice(&input_data, params);

        let result = roc(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_roc_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = RocParams { period: Some(14) };
        let first_input = RocInput::from_candles(&candles, "close", first_params);
        let first_result = roc(&first_input).expect("Failed to calculate first ROC");
        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First ROC output length mismatch"
        );

        let second_params = RocParams { period: Some(14) };
        let second_input = RocInput::from_slice(&first_result.values, second_params);
        let second_result = roc(&second_input).expect("Failed to calculate second ROC");
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second ROC output length mismatch"
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
    fn test_roc_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let close_prices = &candles.close;

        let period = 9;
        let params = RocParams {
            period: Some(period),
        };
        let input = RocInput::from_candles(&candles, "close", params);
        let roc_result = roc(&input).expect("Failed to calculate ROC");

        assert_eq!(
            roc_result.values.len(),
            close_prices.len(),
            "ROC length mismatch"
        );

        if roc_result.values.len() > 240 {
            for i in 240..roc_result.values.len() {
                assert!(
                    !roc_result.values[i].is_nan(),
                    "Expected no NaN after index 240, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
