/// # Linear Regression Angle (LRA)
///
/// Computes the angle (in degrees) of the linear regression line for a given period.
/// The calculation is based on the "least squares method" to find the best-fit line.
/// At each data point (starting after the first valid index plus `period - 1`),
/// the angle is derived from the slope `m` of the line: `angle = atan(m) * (180 / PI)`.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: Linearreg_angle: Input data slice is empty.
/// - **InvalidPeriod**: Linearreg_angle: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: Linearreg_angle: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: Linearreg_angle: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(Linearreg_angleOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the calculation window is filled.
/// - **`Err(Linearreg_angleError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use std::f64::consts::PI;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum Linearreg_angleData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct Linearreg_angleOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Linearreg_angleParams {
    pub period: Option<usize>,
}

impl Default for Linearreg_angleParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct Linearreg_angleInput<'a> {
    pub data: Linearreg_angleData<'a>,
    pub params: Linearreg_angleParams,
}

impl<'a> Linearreg_angleInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: Linearreg_angleParams,
    ) -> Self {
        Self {
            data: Linearreg_angleData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: Linearreg_angleParams) -> Self {
        Self {
            data: Linearreg_angleData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: Linearreg_angleData::Candles {
                candles,
                source: "close",
            },
            params: Linearreg_angleParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| Linearreg_angleParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum Linearreg_angleError {
    #[error("Linearreg_angle: Empty data provided for Linear Regression Angle.")]
    EmptyData,
    #[error("Linearreg_angle: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("Linearreg_angle: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("Linearreg_angle: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn linearreg_angle(
    input: &Linearreg_angleInput,
) -> Result<Linearreg_angleOutput, Linearreg_angleError> {
    let data: &[f64] = match &input.data {
        Linearreg_angleData::Candles { candles, source } => source_type(candles, source),
        Linearreg_angleData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(Linearreg_angleError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(Linearreg_angleError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(Linearreg_angleError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(Linearreg_angleError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut lra_values = vec![f64::NAN; data.len()];

    let sum_x = (period * (period - 1)) as f64 * 0.5;
    let sum_x_sqr = (period * (period - 1) * (2 * period - 1)) as f64 / 6.0;
    let divisor = sum_x * sum_x - (period as f64) * sum_x_sqr;

    let n = data.len();
    let mut prefix_data = vec![0.0; n + 1];
    let mut prefix_id = vec![0.0; n + 1];

    for i in 0..n {
        prefix_data[i + 1] = prefix_data[i] + data[i];
        prefix_id[i + 1] = prefix_id[i] + (i as f64) * data[i];
    }

    for i in (first_valid_idx + period - 1)..n {
        let sum_y = prefix_data[i + 1] - prefix_data[i + 1 - period];
        let sum_kd = prefix_id[i + 1] - prefix_id[i + 1 - period];
        let sum_xy = (i as f64) * sum_y - sum_kd;
        let slope = ((period as f64) * sum_xy - sum_x * sum_y) / divisor;
        lra_values[i] = slope.atan() * (180.0 / PI);
    }

    Ok(Linearreg_angleOutput { values: lra_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_Linearreg_angle_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = Linearreg_angleParams { period: None };
        let input_default = Linearreg_angleInput::from_candles(&candles, "close", default_params);
        let output_default =
            linearreg_angle(&input_default).expect("Failed LRA with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_10 = Linearreg_angleParams { period: Some(10) };
        let input_period_10 =
            Linearreg_angleInput::from_candles(&candles, "hlc3", params_period_10);
        let output_period_10 =
            linearreg_angle(&input_period_10).expect("Failed LRA with period=10, source=hlc3");
        assert_eq!(output_period_10.values.len(), candles.close.len());
    }

    #[test]
    fn test_Linearreg_angle_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = Linearreg_angleParams { period: Some(14) };
        let input = Linearreg_angleInput::from_candles(&candles, "close", params);
        let Linearreg_angle_result = linearreg_angle(&input).expect("Failed to calculate LRA");

        assert_eq!(
            Linearreg_angle_result.values.len(),
            close_prices.len(),
            "LRA length mismatch"
        );

        let expected_last_five_Linearreg_angle = [
            -89.30491945492733,
            -89.28911257342405,
            -89.1088041965075,
            -86.58419429159467,
            -87.77085937059316,
        ];
        assert!(
            Linearreg_angle_result.values.len() >= 5,
            "LRA length too short"
        );
        let start_index = Linearreg_angle_result.values.len() - 5;
        let result_last_five_Linearreg_angle = &Linearreg_angle_result.values[start_index..];
        for (i, &value) in result_last_five_Linearreg_angle.iter().enumerate() {
            let expected_value = expected_last_five_Linearreg_angle[i];
            let diff = (value - expected_value).abs();
            assert!(
                diff < 1e-5,
                "LRA mismatch at index {}: expected {}, got {} (diff {})",
                i,
                expected_value,
                value,
                diff
            );
        }

        let period: usize = 14;
        for i in 0..(period - 1) {
            assert!(Linearreg_angle_result.values[i].is_nan());
        }
    }

    #[test]
    fn test_Linearreg_angle_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = Linearreg_angleParams { period: Some(0) };
        let input = Linearreg_angleInput::from_slice(&input_data, params);

        let result = linearreg_angle(&input);
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
    fn test_Linearreg_angle_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = Linearreg_angleParams { period: Some(10) };
        let input = Linearreg_angleInput::from_slice(&input_data, params);

        let result = linearreg_angle(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_Linearreg_angle_all_nan() {
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = Linearreg_angleParams { period: Some(2) };
        let input = Linearreg_angleInput::from_slice(&input_data, params);

        let result = linearreg_angle(&input);
        assert!(result.is_err(), "Expected an error for all NaN data");
    }

    #[test]
    fn test_Linearreg_angle_not_enough_valid_data() {
        let input_data = [f64::NAN, f64::NAN, 42.0];
        let params = Linearreg_angleParams { period: Some(3) };
        let input = Linearreg_angleInput::from_slice(&input_data, params);

        let result = linearreg_angle(&input);
        assert!(result.is_err(), "Expected not enough valid data error");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Not enough valid data"),
                "Expected 'Not enough valid data' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_Linearreg_angle_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = Linearreg_angleInput::with_default_candles(&candles);
        match input.data {
            Linearreg_angleData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected Linearreg_angleData::Candles variant"),
        }
    }

    #[test]
    fn test_Linearreg_angle_reinput_as_slice() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = Linearreg_angleParams { period: Some(14) };
        let first_input = Linearreg_angleInput::from_candles(&candles, "close", first_params);
        let first_result = linearreg_angle(&first_input).expect("Failed to calculate first LRA");

        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First LRA output length mismatch"
        );

        let second_params = Linearreg_angleParams { period: Some(14) };
        let second_input = Linearreg_angleInput::from_slice(&first_result.values, second_params);
        let second_result = linearreg_angle(&second_input).expect("Failed to calculate second LRA");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second LRA output length mismatch"
        );
    }
}
