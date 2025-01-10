/// # Forecast Oscillator (FOSC)
///
/// The Forecast Oscillator attempts to determine the difference (in percentage) between
/// the current price and a linear-regression-based “forecast” of the price. A positive
/// FOSC suggests the current price is above its expected regression forecast, while a
/// negative FOSC indicates it is below. This can help traders gauge when the price may
/// be overextended or underextended relative to its short-term trend.
///
/// ## Formula
/// FOSC(i) = 100 * [ Price(i) - Forecast(i+1) ] / Price(i)
///
/// where Forecast(i+1) is obtained via a linear regression over the last `period` values,
/// extrapolated one bar ahead (i+1).
///
/// ## Parameters
/// - **period**: The window size for regression. Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: fosc: Input data slice is empty.
/// - **InvalidPeriod**: fosc: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: fosc: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: fosc: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(FoscOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until enough data is available to compute the regression.
/// - **`Err(FoscError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum FoscData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct FoscOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FoscParams {
    pub period: Option<usize>,
}

impl Default for FoscParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct FoscInput<'a> {
    pub data: FoscData<'a>,
    pub params: FoscParams,
}

impl<'a> FoscInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: FoscParams) -> Self {
        Self {
            data: FoscData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: FoscParams) -> Self {
        Self {
            data: FoscData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: FoscData::Candles {
                candles,
                source: "close",
            },
            params: FoscParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| FoscParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum FoscError {
    #[error("fosc: Empty data provided for FOSC.")]
    EmptyData,
    #[error("fosc: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("fosc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("fosc: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn fosc(input: &FoscInput) -> Result<FoscOutput, FoscError> {
    let data: &[f64] = match &input.data {
        FoscData::Candles { candles, source } => source_type(candles, source),
        FoscData::Slice(slice) => slice,
    };

    let size = data.len();
    if size == 0 {
        return Err(FoscError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > size {
        return Err(FoscError::InvalidPeriod {
            period,
            data_len: size,
        });
    }

    if data.iter().all(|v| v.is_nan()) {
        return Err(FoscError::AllValuesNaN);
    }

    let mut output_values = vec![f64::NAN; size];

    let mut x = 0.0;
    let mut x2 = 0.0;
    let mut y = 0.0;
    let mut xy = 0.0;
    let p = 1.0 / (period as f64);
    let mut tsf = 0.0;

    for i in 0..(period - 1) {
        x += (i + 1) as f64;
        x2 += ((i + 1) as f64) * ((i + 1) as f64);
        xy += data[i] * ((i + 1) as f64);
        y += data[i];
    }

    x += period as f64;
    x2 += (period as f64) * (period as f64);

    let bd = {
        let denom = (period as f64) * x2 - x * x;
        if denom.abs() < f64::EPSILON {
            0.0
        } else {
            1.0 / denom
        }
    };

    for i in (period - 1)..size {
        xy += data[i] * (period as f64);
        y += data[i];

        let b = (period as f64 * xy - x * y) * bd;
        let a = (y - b * x) * p;

        if i >= period {
            if !data[i].is_nan() && data[i] != 0.0 {
                output_values[i] = 100.0 * (data[i] - tsf) / data[i];
            } else {
                output_values[i] = f64::NAN;
            }
        }

        tsf = a + b * ((period + 1) as f64);

        xy -= y;
        let old_idx = i as isize - (period as isize) + 1;
        y -= data[old_idx as usize];
    }

    Ok(FoscOutput {
        values: output_values,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_fosc_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = FoscParams { period: None };
        let input_default = FoscInput::from_candles(&candles, "close", default_params);
        let output_default = fosc(&input_default).expect("Failed FOSC with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_10 = FoscParams { period: Some(10) };
        let input_10 = FoscInput::from_candles(&candles, "close", params_period_10);
        let output_10 = fosc(&input_10).expect("Failed FOSC with period=10");
        assert_eq!(output_10.values.len(), candles.close.len());
    }

    #[test]
    fn test_fosc_basic_accuracy_check() {
        let test_data = [
            81.59, 81.06, 82.87, 83.00, 83.61, 83.15, 82.84, 82.84, 83.99, 84.55, 84.36, 85.53,
        ];
        let period = 5;
        let input = FoscInput::from_slice(
            &test_data,
            FoscParams {
                period: Some(period),
            },
        );
        let result = fosc(&input).expect("Failed to compute FOSC");

        assert_eq!(result.values.len(), test_data.len());
        for i in 0..(period - 1) {
            assert!(result.values[i].is_nan());
        }
    }

    #[test]
    fn test_fosc_with_nan_data() {
        let input_data = [f64::NAN, f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0];
        let params = FoscParams { period: Some(3) };
        let input = FoscInput::from_slice(&input_data, params);
        let result = fosc(&input).expect("Failed to calculate FOSC with NaN data");
        assert_eq!(result.values.len(), input_data.len());
    }

    #[test]
    fn test_fosc_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = FoscParams { period: Some(0) };
        let input = FoscInput::from_slice(&input_data, params);

        let result = fosc(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_fosc_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = FoscParams { period: Some(10) };
        let input = FoscInput::from_slice(&input_data, params);

        let result = fosc(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_fosc_very_small_data_set() {
        let input_data = [42.0];
        let params = FoscParams { period: Some(5) };
        let input = FoscInput::from_slice(&input_data, params);

        let result = fosc(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_fosc_all_values_nan() {
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = FoscParams { period: Some(2) };
        let input = FoscInput::from_slice(&input_data, params);

        let result = fosc(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("All values are NaN"),
                "Expected 'All values are NaN' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_fosc_known_values_length() {
        let input_data = [100.0, 101.0, 102.0, 99.0, 98.0, 97.0, 100.0];
        let params = FoscParams { period: Some(5) };
        let input = FoscInput::from_slice(&input_data, params);
        let result = fosc(&input).expect("Failed to compute FOSC");
        assert_eq!(result.values.len(), input_data.len());
    }

    #[test]
    fn test_fosc_expected_values_reference() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let expected_last_five = [
            -0.8904444627923475,
            -0.4763353099245297,
            -0.2379782851444668,
            0.292790128025632,
            -0.6597902988090389,
        ];
        let params = FoscParams { period: Some(5) };
        let input = FoscInput::from_candles(&candles, "close", params);
        let result = fosc(&input).expect("Failed to compute FOSC");
        let valid_len = result.values.len();
        assert!(valid_len >= 5);

        let output_slice = &result.values[valid_len - 5..valid_len];
        for (i, &val) in output_slice.iter().enumerate() {
            let exp: f64 = expected_last_five[i];
            if exp.is_nan() {
                assert!(val.is_nan());
            } else {
                assert!(
                    (val - exp).abs() < 1e-7,
                    "Mismatch at index {}: expected {}, got {}",
                    i,
                    exp,
                    val
                );
            }
        }
    }
}
