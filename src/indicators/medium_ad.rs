/// # Median Absolute Deviation (MEDIUM_AD)
///
/// A robust measure of dispersion that calculates the median of the absolute
/// deviations from the median for a specified `period`. This indicator is less
/// sensitive to outliers compared to standard deviation.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: medium_ad: Input data slice is empty.
/// - **InvalidPeriod**: medium_ad: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: medium_ad: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: medium_ad: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(MediumAdOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the MEDIUM_AD window is filled.
/// - **`Err(MediumAdError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MediumAdData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MediumAdOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MediumAdParams {
    pub period: Option<usize>,
}

impl Default for MediumAdParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct MediumAdInput<'a> {
    pub data: MediumAdData<'a>,
    pub params: MediumAdParams,
}

impl<'a> MediumAdInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: MediumAdParams) -> Self {
        Self {
            data: MediumAdData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: MediumAdParams) -> Self {
        Self {
            data: MediumAdData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MediumAdData::Candles {
                candles,
                source: "close",
            },
            params: MediumAdParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| MediumAdParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum MediumAdError {
    #[error("medium_ad: Empty data provided for MEDIUM_AD.")]
    EmptyData,
    #[error("medium_ad: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("medium_ad: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("medium_ad: All values are NaN.")]
    AllValuesNaN,
}

pub fn medium_ad(input: &MediumAdInput) -> Result<MediumAdOutput, MediumAdError> {
    let data: &[f64] = match &input.data {
        MediumAdData::Candles { candles, source } => source_type(candles, source),
        MediumAdData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(MediumAdError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(MediumAdError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(MediumAdError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(MediumAdError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut output_values = vec![f64::NAN; data.len()];

    for i in (first_valid_idx + period - 1)..data.len() {
        let window = &data[i + 1 - period..=i];
        if window.iter().any(|&v| v.is_nan()) {
            output_values[i] = f64::NAN;
            continue;
        }
        let mut sorted_window: Vec<f64> = window.to_vec();
        sorted_window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_val = if period % 2 == 1 {
            sorted_window[period / 2]
        } else {
            0.5 * (sorted_window[period / 2 - 1] + sorted_window[period / 2])
        };
        let mut abs_devs: Vec<f64> = sorted_window
            .iter()
            .map(|&v| (v - median_val).abs())
            .collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if period % 2 == 1 {
            abs_devs[period / 2]
        } else {
            0.5 * (abs_devs[period / 2 - 1] + abs_devs[period / 2])
        };
        output_values[i] = mad;
    }

    Ok(MediumAdOutput {
        values: output_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_medium_ad_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = MediumAdParams { period: None };
        let input_default = MediumAdInput::from_candles(&candles, "close", default_params);
        let output_default =
            medium_ad(&input_default).expect("Failed MEDIUM_AD with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_10 = MediumAdParams { period: Some(10) };
        let input_period_10 = MediumAdInput::from_candles(&candles, "hl2", params_period_10);
        let output_period_10 =
            medium_ad(&input_period_10).expect("Failed MEDIUM_AD with period=10, source=hl2");
        assert_eq!(output_period_10.values.len(), candles.close.len());

        let params_custom = MediumAdParams { period: Some(20) };
        let input_custom = MediumAdInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = medium_ad(&input_custom).expect("Failed MEDIUM_AD fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_medium_ad_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let params = MediumAdParams { period: Some(5) };
        let input = MediumAdInput::from_candles(&candles, "hl2", params);
        let mad_result = medium_ad(&input).expect("Failed to calculate MEDIUM_AD");

        assert_eq!(
            mad_result.values.len(),
            close_prices.len(),
            "Length mismatch"
        );

        let expected_last_five = [220.0, 78.5, 126.5, 48.0, 28.5];
        assert!(mad_result.values.len() >= 5, "MEDIUM_AD length too short");
        let start_index = mad_result.values.len() - 5;
        let result_last_five = &mad_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "MEDIUM_AD mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period: usize = 5;
        for i in 0..(period - 1) {
            assert!(mad_result.values[i].is_nan());
        }

        let default_input = MediumAdInput::with_default_candles(&candles);
        let default_mad_result = medium_ad(&default_input).expect("Failed to calculate MEDIUM_AD");
        assert_eq!(default_mad_result.values.len(), close_prices.len());
    }

    #[test]
    fn test_medium_ad_params_with_default_params() {
        let default_params = MediumAdParams::default();
        assert_eq!(
            default_params.period,
            Some(5),
            "Expected period to be Some(5) in default parameters"
        );
    }

    #[test]
    fn test_medium_ad_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = MediumAdInput::with_default_candles(&candles);
        match input.data {
            MediumAdData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected MediumAdData::Candles variant"),
        }
    }

    #[test]
    fn test_medium_ad_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = MediumAdParams { period: Some(0) };
        let input = MediumAdInput::from_slice(&input_data, params);
        let result = medium_ad(&input);
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
    fn test_medium_ad_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = MediumAdParams { period: Some(10) };
        let input = MediumAdInput::from_slice(&input_data, params);
        let result = medium_ad(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_medium_ad_very_small_data_set() {
        let input_data = [42.0];
        let params = MediumAdParams { period: Some(5) };
        let input = MediumAdInput::from_slice(&input_data, params);
        let result = medium_ad(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_medium_ad_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = MediumAdParams { period: Some(5) };
        let first_input = MediumAdInput::from_candles(&candles, "close", first_params);
        let first_result = medium_ad(&first_input).expect("Failed to calculate first MEDIUM_AD");
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = MediumAdParams { period: Some(5) };
        let second_input = MediumAdInput::from_slice(&first_result.values, second_params);
        let second_result = medium_ad(&second_input).expect("Failed to calculate second MEDIUM_AD");
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in (5 * 2 - 1)..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after sufficient indices, but found NaN at index {}",
                i
            );
        }
    }

    #[test]
    fn test_medium_ad_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 5;
        let params = MediumAdParams {
            period: Some(period),
        };
        let input = MediumAdInput::from_candles(&candles, "close", params);
        let mad_result = medium_ad(&input).expect("Failed to calculate MEDIUM_AD");
        assert_eq!(mad_result.values.len(), close_prices.len());
        if mad_result.values.len() > 60 {
            for i in 60..mad_result.values.len() {
                assert!(
                    !mad_result.values[i].is_nan(),
                    "Expected no NaN after index 60, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
