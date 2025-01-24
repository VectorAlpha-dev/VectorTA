/// # Standard Deviation (STDDEV)
///
/// The Standard Deviation (STDDEV) indicator measures the dispersion of data points
/// relative to their mean. It computes the rolling standard deviation over a specified
/// `period`, multiplying the result by `nbdev` (standard deviations to include).
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 5.
/// - **nbdev**: The factor by which to multiply the standard deviation. Defaults to 1.0.
///
/// ## Errors
/// - **EmptyData**: stddev: Input data slice is empty.
/// - **InvalidPeriod**: stddev: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: stddev: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: stddev: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(StdDevOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the rolling window is filled.
/// - **`Err(StdDevError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum StdDevData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct StdDevOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct StdDevParams {
    pub period: Option<usize>,
    pub nbdev: Option<f64>,
}

impl Default for StdDevParams {
    fn default() -> Self {
        Self {
            period: Some(5),
            nbdev: Some(1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StdDevInput<'a> {
    pub data: StdDevData<'a>,
    pub params: StdDevParams,
}

impl<'a> StdDevInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: StdDevParams) -> Self {
        Self {
            data: StdDevData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: StdDevParams) -> Self {
        Self {
            data: StdDevData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: StdDevData::Candles {
                candles,
                source: "close",
            },
            params: StdDevParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| StdDevParams::default().period.unwrap())
    }

    pub fn get_nbdev(&self) -> f64 {
        self.params
            .nbdev
            .unwrap_or_else(|| StdDevParams::default().nbdev.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum StdDevError {
    #[error("stddev: Empty data provided for STDDEV.")]
    EmptyData,
    #[error("stddev: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("stddev: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("stddev: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn stddev(input: &StdDevInput) -> Result<StdDevOutput, StdDevError> {
    let data: &[f64] = match &input.data {
        StdDevData::Candles { candles, source } => source_type(candles, source),
        StdDevData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(StdDevError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(StdDevError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let nbdev = input.get_nbdev();
    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(StdDevError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(StdDevError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut stddev_values = vec![f64::NAN; data.len()];
    let mut sum = 0.0;
    let mut sum_sqr = 0.0;

    for &value in data[first_valid_idx..(first_valid_idx + period)].iter() {
        sum += value;
        sum_sqr += value * value;
    }

    let compute_stddev = |sum: f64, sum_sqr: f64, period: f64, nbdev: f64| {
        let mean = sum / period;
        let variance = (sum_sqr / period) - (mean * mean);
        if variance <= 0.0 {
            0.0
        } else {
            variance.sqrt() * nbdev
        }
    };

    stddev_values[first_valid_idx + period - 1] =
        compute_stddev(sum, sum_sqr, period as f64, nbdev);

    for i in (first_valid_idx + period)..data.len() {
        let old_value = data[i - period];
        let new_value = data[i];
        sum += new_value - old_value;
        sum_sqr += (new_value * new_value) - (old_value * old_value);
        stddev_values[i] = compute_stddev(sum, sum_sqr, period as f64, nbdev);
    }

    Ok(StdDevOutput {
        values: stddev_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_stddev_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = StdDevParams {
            period: None,
            nbdev: None,
        };
        let input_default = StdDevInput::from_candles(&candles, "close", default_params);
        let output_default = stddev(&input_default).expect("Failed STDDEV with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = StdDevParams {
            period: Some(14),
            nbdev: Some(2.0),
        };
        let input_period_14 = StdDevInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 =
            stddev(&input_period_14).expect("Failed STDDEV with period=14, nbdev=2, source=hl2");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = StdDevParams {
            period: Some(20),
            nbdev: Some(1.5),
        };
        let input_custom = StdDevInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = stddev(&input_custom).expect("Failed STDDEV fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_stddev_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = StdDevParams {
            period: Some(5),
            nbdev: Some(1.0),
        };
        let input = StdDevInput::from_candles(&candles, "close", params);
        let stddev_result = stddev(&input).expect("Failed to calculate STDDEV");

        assert_eq!(
            stddev_result.values.len(),
            close_prices.len(),
            "STDDEV length mismatch"
        );

        // Example final five test values (placeholder comparison)
        let expected_last_five_stddev = [
            180.12506767314034,
            77.7395652441455,
            127.16225857341935,
            89.40156600773197,
            218.50034325919697,
        ];
        assert!(stddev_result.values.len() >= 5, "STDDEV length too short");
        let start_index = stddev_result.values.len() - 5;
        let result_last_five_stddev = &stddev_result.values[start_index..];
        for (i, &value) in result_last_five_stddev.iter().enumerate() {
            let expected_value = expected_last_five_stddev[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "STDDEV mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period: usize = 5;
        for i in 0..(period - 1) {
            assert!(stddev_result.values[i].is_nan());
        }

        let default_input = StdDevInput::with_default_candles(&candles);
        let default_stddev_result =
            stddev(&default_input).expect("Failed to calculate STDDEV defaults");
        assert_eq!(default_stddev_result.values.len(), close_prices.len());
    }

    #[test]
    fn test_stddev_params_with_default_params() {
        let default_params = StdDevParams::default();
        assert_eq!(
            default_params.period,
            Some(5),
            "Expected period to be Some(5) in default parameters"
        );
        assert_eq!(
            default_params.nbdev,
            Some(1.0),
            "Expected nbdev to be Some(1.0) in default parameters"
        );
    }

    #[test]
    fn test_stddev_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = StdDevInput::with_default_candles(&candles);
        match input.data {
            StdDevData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected StdDevData::Candles variant"),
        }
    }

    #[test]
    fn test_stddev_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = StdDevParams {
            period: Some(0),
            nbdev: Some(1.0),
        };
        let input = StdDevInput::from_slice(&input_data, params);

        let result = stddev(&input);
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
    fn test_stddev_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = StdDevParams {
            period: Some(10),
            nbdev: Some(1.0),
        };
        let input = StdDevInput::from_slice(&input_data, params);

        let result = stddev(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_stddev_very_small_data_set() {
        let input_data = [42.0];
        let params = StdDevParams {
            period: Some(5),
            nbdev: Some(1.0),
        };
        let input = StdDevInput::from_slice(&input_data, params);

        let result = stddev(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_stddev_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = StdDevParams {
            period: Some(10),
            nbdev: Some(1.0),
        };
        let first_input = StdDevInput::from_candles(&candles, "close", first_params);
        let first_result = stddev(&first_input).expect("Failed to calculate first STDDEV");

        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First STDDEV output length mismatch"
        );

        let second_params = StdDevParams {
            period: Some(10),
            nbdev: Some(1.0),
        };
        let second_input = StdDevInput::from_slice(&first_result.values, second_params);
        let second_result = stddev(&second_input).expect("Failed to calculate second STDDEV");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second STDDEV output length mismatch"
        );

        for i in 19..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 19, but found NaN at index {}",
                i
            );
        }
    }

    #[test]
    fn test_stddev_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 5;
        let params = StdDevParams {
            period: Some(period),
            nbdev: Some(1.0),
        };
        let input = StdDevInput::from_candles(&candles, "close", params);
        let stddev_result = stddev(&input).expect("Failed to calculate STDDEV");

        assert_eq!(
            stddev_result.values.len(),
            close_prices.len(),
            "STDDEV length mismatch"
        );

        if stddev_result.values.len() > (period * 2) {
            for i in (period * 2)..stddev_result.values.len() {
                assert!(
                    !stddev_result.values[i].is_nan(),
                    "Expected no NaN after index {}, but found NaN",
                    i
                );
            }
        }
    }
}
