/// # Linear Regression Intercept (LINEARREG_INTERCEPT)
///
/// This indicator calculates the y-value of the linear regression line at the last point
/// (period - 1) of each regression window. It effectively gives the "intercept" value if
/// you consider the last bar in each window as the reference point.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: linearreg_intercept: Input data slice is empty.
/// - **InvalidPeriod**: linearreg_intercept: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: linearreg_intercept: Fewer than `period` valid (non-`NaN`) data
///   points remain after the first valid index.
/// - **AllValuesNaN**: linearreg_intercept: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(LinearRegInterceptOutput)`** on success, containing a `Vec<f64>` matching the
///   input length, with leading `NaN`s until the regression window is filled.
/// - **`Err(LinearRegInterceptError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum LinearRegInterceptData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct LinearRegInterceptOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LinearRegInterceptParams {
    pub period: Option<usize>,
}

impl Default for LinearRegInterceptParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct LinearRegInterceptInput<'a> {
    pub data: LinearRegInterceptData<'a>,
    pub params: LinearRegInterceptParams,
}

impl<'a> LinearRegInterceptInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: LinearRegInterceptParams,
    ) -> Self {
        Self {
            data: LinearRegInterceptData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: LinearRegInterceptParams) -> Self {
        Self {
            data: LinearRegInterceptData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: LinearRegInterceptData::Candles {
                candles,
                source: "close",
            },
            params: LinearRegInterceptParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| LinearRegInterceptParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum LinearRegInterceptError {
    #[error("linearreg_intercept: Empty data provided.")]
    EmptyData,
    #[error("linearreg_intercept: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("linearreg_intercept: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("linearreg_intercept: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn linearreg_intercept(
    input: &LinearRegInterceptInput,
) -> Result<LinearRegInterceptOutput, LinearRegInterceptError> {
    let data: &[f64] = match &input.data {
        LinearRegInterceptData::Candles { candles, source } => source_type(candles, source),
        LinearRegInterceptData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(LinearRegInterceptError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(LinearRegInterceptError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(LinearRegInterceptError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(LinearRegInterceptError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut output_values = vec![f64::NAN; data.len()];

    let mut x = 0.0;
    let mut x2 = 0.0;
    let mut y = 0.0;
    let mut xy = 0.0;

    {
        for i in 0..(period - 1) {
            let val = data[first_valid_idx + i];
            let idx = (i + 1) as f64;
            x += idx;
            x2 += idx * idx;
            xy += val * idx;
            y += val;
        }
        let p_idx = period as f64;
        x += p_idx;
        x2 += p_idx * p_idx;
    }

    let denom = (period as f64) * x2 - x * x;
    if denom.abs() < f64::EPSILON {
        return Ok(LinearRegInterceptOutput {
            values: output_values,
        });
    }
    let bd = 1.0 / denom;

    let start_i = first_valid_idx + period - 1;
    for i in start_i..data.len() {
        let val = data[i];
        xy += val * (period as f64);
        y += val;

        let b = ((period as f64) * xy - x * y) * bd;
        let a = (y - b * x) / (period as f64);
        output_values[i] = a + b;

        let remove_idx = i as isize - (period as isize) + 1;
        if remove_idx >= 0 && (remove_idx as usize) < data.len() {
            xy -= y;
            y -= data[remove_idx as usize];
        }
    }

    Ok(LinearRegInterceptOutput {
        values: output_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_linearreg_intercept_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = LinearRegInterceptParams { period: None };
        let input_default =
            LinearRegInterceptInput::from_candles(&candles, "close", default_params);
        let output_default = linearreg_intercept(&input_default)
            .expect("Failed linearreg_intercept with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_10 = LinearRegInterceptParams { period: Some(10) };
        let input_period_10 =
            LinearRegInterceptInput::from_candles(&candles, "hl2", params_period_10);
        let output_period_10 = linearreg_intercept(&input_period_10)
            .expect("Failed linearreg_intercept with period=10, source=hl2");
        assert_eq!(output_period_10.values.len(), candles.close.len());

        let params_custom = LinearRegInterceptParams { period: Some(20) };
        let input_custom = LinearRegInterceptInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom =
            linearreg_intercept(&input_custom).expect("Failed linearreg_intercept fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_linearreg_intercept_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = LinearRegInterceptParams { period: Some(14) };
        let input = LinearRegInterceptInput::from_candles(&candles, "close", params);
        let linreg_result =
            linearreg_intercept(&input).expect("Failed to calculate linear regression intercept");

        assert_eq!(
            linreg_result.values.len(),
            close_prices.len(),
            "Length mismatch"
        );

        let expected_last_five = [
            60000.91428571429,
            59947.142857142855,
            59754.57142857143,
            59318.4,
            59321.91428571429,
        ];
        assert!(
            linreg_result.values.len() >= 5,
            "Indicator length is too short for last five check"
        );
        let start_index = linreg_result.values.len() - 5;
        let result_last_five = &linreg_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period: usize = 14;
        for i in 0..(period - 1) {
            assert!(linreg_result.values[i].is_nan());
        }

        let default_input = LinearRegInterceptInput::with_default_candles(&candles);
        let default_linreg_result = linearreg_intercept(&default_input)
            .expect("Failed to calculate linear regression intercept defaults");
        assert_eq!(default_linreg_result.values.len(), close_prices.len());
    }

    #[test]
    fn test_linearreg_intercept_params_with_default() {
        let default_params = LinearRegInterceptParams::default();
        assert_eq!(
            default_params.period,
            Some(14),
            "Expected period=14 in default parameters"
        );
    }

    #[test]
    fn test_linearreg_intercept_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = LinearRegInterceptInput::with_default_candles(&candles);
        match input.data {
            LinearRegInterceptData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected LinearRegInterceptData::Candles variant"),
        }
    }

    #[test]
    fn test_linearreg_intercept_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = LinearRegInterceptParams { period: Some(0) };
        let input = LinearRegInterceptInput::from_slice(&input_data, params);

        let result = linearreg_intercept(&input);
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
    fn test_linearreg_intercept_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = LinearRegInterceptParams { period: Some(10) };
        let input = LinearRegInterceptInput::from_slice(&input_data, params);

        let result = linearreg_intercept(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_linearreg_intercept_very_small_data_set() {
        let input_data = [42.0];
        let params = LinearRegInterceptParams { period: Some(14) };
        let input = LinearRegInterceptInput::from_slice(&input_data, params);

        let result = linearreg_intercept(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_linearreg_intercept_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = LinearRegInterceptParams { period: Some(10) };
        let first_input = LinearRegInterceptInput::from_candles(&candles, "close", first_params);
        let first_result =
            linearreg_intercept(&first_input).expect("Failed to calculate first intercept");

        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First intercept output length mismatch"
        );

        let second_params = LinearRegInterceptParams { period: Some(10) };
        let second_input = LinearRegInterceptInput::from_slice(&first_result.values, second_params);
        let second_result =
            linearreg_intercept(&second_input).expect("Failed to calculate second intercept");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second intercept output length mismatch"
        );

        for i in 20..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 20, but found NaN at index {}",
                i
            );
        }
    }

    #[test]
    fn test_linearreg_intercept_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 14;
        let params = LinearRegInterceptParams {
            period: Some(period),
        };
        let input = LinearRegInterceptInput::from_candles(&candles, "close", params);
        let linreg_result =
            linearreg_intercept(&input).expect("Failed to calculate linear regression intercept");

        assert_eq!(
            linreg_result.values.len(),
            close_prices.len(),
            "Length mismatch"
        );

        if linreg_result.values.len() > 50 {
            for i in 50..linreg_result.values.len() {
                assert!(
                    !linreg_result.values[i].is_nan(),
                    "Expected no NaN after index 50, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
