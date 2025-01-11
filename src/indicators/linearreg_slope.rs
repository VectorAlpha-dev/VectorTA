/// # Linear Regression Slope
///
/// Computes the slope (coefficient `b`) of the linear regression line over the most recent `period` values.
/// The slope is calculated for each valid index in the input, matching the input's length, with leading `NaN`s
/// until enough data points (equal to `period`) are available after the first valid non-`NaN` value.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: linearreg_slope: Input data slice is empty.
/// - **InvalidPeriod**: linearreg_slope: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: linearreg_slope: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: linearreg_slope: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(LinearRegSlopeOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the linear regression window is filled.
/// - **`Err(LinearRegSlopeError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum LinearRegSlopeData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct LinearRegSlopeOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LinearRegSlopeParams {
    pub period: Option<usize>,
}

impl Default for LinearRegSlopeParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct LinearRegSlopeInput<'a> {
    pub data: LinearRegSlopeData<'a>,
    pub params: LinearRegSlopeParams,
}

impl<'a> LinearRegSlopeInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: LinearRegSlopeParams,
    ) -> Self {
        Self {
            data: LinearRegSlopeData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: LinearRegSlopeParams) -> Self {
        Self {
            data: LinearRegSlopeData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: LinearRegSlopeData::Candles {
                candles,
                source: "close",
            },
            params: LinearRegSlopeParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| LinearRegSlopeParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum LinearRegSlopeError {
    #[error("linearreg_slope: Empty data provided.")]
    EmptyData,
    #[error("linearreg_slope: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("linearreg_slope: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("linearreg_slope: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn linearreg_slope(
    input: &LinearRegSlopeInput,
) -> Result<LinearRegSlopeOutput, LinearRegSlopeError> {
    let data: &[f64] = match &input.data {
        LinearRegSlopeData::Candles { candles, source } => source_type(candles, source),
        LinearRegSlopeData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(LinearRegSlopeError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(LinearRegSlopeError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(LinearRegSlopeError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(LinearRegSlopeError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut slope_values = vec![f64::NAN; data.len()];

    let n = period as f64;
    let sum_x = (period - 1) as f64 * n / 2.0;
    let sum_x2 = (period - 1) as f64 * n * (2.0 * (period - 1) as f64 + 1.0) / 6.0;

    let mut prefix_sum_data = vec![0.0; data.len() + 1];
    let mut prefix_sum_data_k = vec![0.0; data.len() + 1];
    for i in 0..data.len() {
        prefix_sum_data[i + 1] = prefix_sum_data[i] + data[i];
        prefix_sum_data_k[i + 1] = prefix_sum_data_k[i] + (i as f64) * data[i];
    }

    for i in (first_valid_idx + period - 1)..data.len() {
        let end_idx = i + 1;
        let start_idx = end_idx - period;

        let sum_y = prefix_sum_data[end_idx] - prefix_sum_data[start_idx];

        let total_kd = prefix_sum_data_k[end_idx] - prefix_sum_data_k[start_idx];
        let sum_xy = total_kd - (start_idx as f64) * sum_y;

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = n * sum_x2 - sum_x * sum_x;
        slope_values[i] = if denominator.abs() < f64::EPSILON {
            f64::NAN
        } else {
            numerator / denominator
        };
    }

    Ok(LinearRegSlopeOutput {
        values: slope_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_linearreg_slope_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = LinearRegSlopeParams { period: None };
        let input_default = LinearRegSlopeInput::from_candles(&candles, "close", default_params);
        let output_default =
            linearreg_slope(&input_default).expect("Failed linearreg_slope with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_20 = LinearRegSlopeParams { period: Some(20) };
        let input_period_20 =
            LinearRegSlopeInput::from_candles(&candles, "close", params_period_20);
        let output_period_20 = linearreg_slope(&input_period_20)
            .expect("Failed linearreg_slope with period=20, source=close");
        assert_eq!(output_period_20.values.len(), candles.close.len());
    }

    #[test]
    fn test_linearreg_slope_accuracy() {
        let input_data = [100.0, 98.0, 95.0, 90.0, 85.0, 80.0, 78.0, 77.0, 79.0, 81.0];
        let params = LinearRegSlopeParams { period: Some(5) };
        let input = LinearRegSlopeInput::from_slice(&input_data, params);
        let result = linearreg_slope(&input).expect("Failed to calculate linearreg_slope");

        assert_eq!(result.values.len(), input_data.len());
        for val in &result.values[4..] {
            assert!(
                !val.is_nan(),
                "Expected valid slope values after period-1 index"
            );
        }
    }

    #[test]
    fn test_linearreg_slope_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = LinearRegSlopeParams { period: Some(0) };
        let input = LinearRegSlopeInput::from_slice(&input_data, params);

        let result = linearreg_slope(&input);
        assert!(
            result.is_err(),
            "Expected an error for zero period in linearreg_slope"
        );
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_linearreg_slope_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = LinearRegSlopeParams { period: Some(10) };
        let input = LinearRegSlopeInput::from_slice(&input_data, params);

        let result = linearreg_slope(&input);
        assert!(
            result.is_err(),
            "Expected an error for period > data.len() in linearreg_slope"
        );
    }

    #[test]
    fn test_linearreg_slope_all_nan_data() {
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = LinearRegSlopeParams { period: Some(3) };
        let input = LinearRegSlopeInput::from_slice(&input_data, params);

        let result = linearreg_slope(&input);
        assert!(result.is_err(), "Expected error for all-NaN data");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("All values are NaN"),
                "Expected 'All values are NaN' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_linearreg_slope_not_enough_valid_data() {
        let input_data = [f64::NAN, 10.0, f64::NAN];
        let params = LinearRegSlopeParams { period: Some(3) };
        let input = LinearRegSlopeInput::from_slice(&input_data, params);

        let result = linearreg_slope(&input);
        assert!(result.is_err(), "Expected error for not enough valid data");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Not enough valid data"),
                "Expected 'Not enough valid data' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_linearreg_slope_known_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = LinearRegSlopeParams { period: Some(14) };
        let input = LinearRegSlopeInput::from_candles(&candles, "close", params);
        let result = linearreg_slope(&input).expect("Failed to calculate linearreg_slope");

        let expected_last_five = [
            -82.42637362637363,
            -80.5934065934066,
            -64.28571428571429,
            -16.753846153846155,
            -25.69010989010989,
        ];

        assert!(result.values.len() >= 5);
        let start_index = result.values.len() - 5;
        let last_five = &result.values[start_index..];
        for (i, (&actual, &expected)) in last_five.iter().zip(expected_last_five.iter()).enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at last_five[{}]: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_linearreg_slope_slice_reinput() {
        let input_data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let first_params = LinearRegSlopeParams { period: Some(3) };
        let first_input = LinearRegSlopeInput::from_slice(&input_data, first_params);
        let first_result = linearreg_slope(&first_input).expect("Failed to calculate first slope");

        let second_params = LinearRegSlopeParams { period: Some(3) };
        let second_input = LinearRegSlopeInput::from_slice(&first_result.values, second_params);
        let second_result =
            linearreg_slope(&second_input).expect("Failed to calculate second slope");

        assert_eq!(second_result.values.len(), first_result.values.len());
    }
}
