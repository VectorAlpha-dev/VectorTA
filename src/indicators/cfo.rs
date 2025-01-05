/// # Chande Forecast Oscillator (CFO)
///
/// Calculates `scalar * ((source - LinReg(source, period)) / source)`.
///
/// ## Parameters
/// - **period**: Window size for the internal linear regression. Defaults to 14.
/// - **scalar**: Multiplier for the final ratio. Defaults to 100.0.
///
/// ## Errors
/// - **AllValuesNaN**: cfo: All input data values are `NaN`.
/// - **InvalidPeriod**: cfo: Invalid or zero `period`.
/// - **NoData**: cfo: Input data slice is empty.
///
/// ## Returns
/// - **`Ok(CfoOutput)`** on success, containing a `Vec<f64>` matching the input length.
///   Leading values may be `NaN` until enough data is gathered.
/// - **`Err(CfoError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum CfoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CfoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CfoParams {
    pub period: Option<usize>,
    pub scalar: Option<f64>,
}

impl Default for CfoParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            scalar: Some(100.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CfoInput<'a> {
    pub data: CfoData<'a>,
    pub params: CfoParams,
}

impl<'a> CfoInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: CfoParams) -> Self {
        Self {
            data: CfoData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: CfoParams) -> Self {
        Self {
            data: CfoData::Slices(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: CfoData::Candles {
                candles,
                source: "close",
            },
            params: CfoParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params.period.unwrap_or(14)
    }

    pub fn get_scalar(&self) -> f64 {
        self.params.scalar.unwrap_or(100.0)
    }
}

#[derive(Debug, Error)]
pub enum CfoError {
    #[error("cfo: All values are NaN.")]
    AllValuesNaN,
    #[error("cfo: Invalid period: {period}.")]
    InvalidPeriod { period: usize },
    #[error("cfo: No data provided.")]
    NoData,
}

#[inline]
pub fn cfo(input: &CfoInput) -> Result<CfoOutput, CfoError> {
    let data = match &input.data {
        CfoData::Candles { candles, source } => source_type(candles, source),
        CfoData::Slices(slice) => slice,
    };

    let size = data.len();
    let period = input.get_period();
    let scalar = input.get_scalar();

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(CfoError::AllValuesNaN),
    };

    if period < 1 {
        return Err(CfoError::InvalidPeriod { period });
    }
    if size == 0 {
        return Err(CfoError::NoData);
    }
    if size < period {
        return Ok(CfoOutput {
            values: vec![f64::NAN; size],
        });
    }

    let mut cfo_values = vec![f64::NAN; size];

    let x = (period * (period + 1)) / 2;
    let x2 = (period * (period + 1) * (2 * period + 1)) / 6;
    let x_f = x as f64;
    let x2_f = x2 as f64;
    let period_f = period as f64;
    let bd = 1.0 / (period_f * x2_f - x_f * x_f);

    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;

    for i in 0..(period - 1) {
        let x_i = (i + 1) as f64;
        let val = data[i + first_valid_idx];
        sum_y += val;
        sum_xy += val * x_i;
    }

    for i in (first_valid_idx + period - 1)..size {
        let val = data[i];
        sum_xy += val * period_f;
        sum_y += val;

        let b = (period_f * sum_xy - x_f * sum_y) * bd;
        let a = (sum_y - b * x_f) / period_f;
        let forecast = a + b * period_f;

        if !val.is_nan() {
            cfo_values[i] = scalar * (val - forecast) / val;
        }

        sum_xy -= sum_y;
        let oldest_idx = i - (period - 1);
        let oldest_val = data[oldest_idx];
        sum_y -= oldest_val;
    }

    Ok(CfoOutput { values: cfo_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_cfo_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = CfoParams {
            period: None,
            scalar: None,
        };
        let input_default = CfoInput::from_candles(&candles, "close", default_params);
        let output_default = cfo(&input_default).expect("Failed CFO with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_10 = CfoParams {
            period: Some(10),
            scalar: Some(100.0),
        };
        let input_period_10 = CfoInput::from_candles(&candles, "high", params_period_10);
        let output_period_10 =
            cfo(&input_period_10).expect("Failed CFO with period=10, source=high");
        assert_eq!(output_period_10.values.len(), candles.close.len());

        let params_custom = CfoParams {
            period: Some(20),
            scalar: Some(50.0),
        };
        let input_custom = CfoInput::from_candles(&candles, "low", params_custom);
        let output_custom = cfo(&input_custom).expect("Failed CFO fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_cfo_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let params = CfoParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let input = CfoInput::from_candles(&candles, "close", params);
        let cfo_result = cfo(&input).expect("Failed to calculate CFO");

        assert_eq!(
            cfo_result.values.len(),
            close_prices.len(),
            "CFO length mismatch"
        );

        let expected_last_five = [
            0.5998626489475746,
            0.47578011282578453,
            0.20349744599816233,
            0.0919617952835795,
            -0.5676291145560617,
        ];
        assert!(cfo_result.values.len() >= 5, "CFO length too short");
        let start_index = cfo_result.values.len() - 5;
        let result_last_five = &cfo_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "CFO mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let default_input = CfoInput::with_default_candles(&candles);
        let default_cfo_result = cfo(&default_input).expect("Failed to calculate CFO defaults");
        assert_eq!(default_cfo_result.values.len(), close_prices.len());
    }

    #[test]
    fn test_cfo_params_with_default_params() {
        let default_params = CfoParams::default();
        assert_eq!(default_params.period, Some(14));
        assert_eq!(default_params.scalar, Some(100.0));
    }

    #[test]
    fn test_cfo_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = CfoInput::with_default_candles(&candles);
        match input.data {
            CfoData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected CfoData::Candles variant"),
        }
    }

    #[test]
    fn test_cfo_with_zero_period() {
        let data = [10.0, 20.0, 30.0];
        let params = CfoParams {
            period: Some(0),
            scalar: Some(100.0),
        };
        let input = CfoInput::from_slice(&data, params);

        let result = cfo(&input);
        assert!(result.is_err(), "Expected error for zero period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_cfo_with_period_exceeding_data_length() {
        let data = [10.0, 20.0, 30.0];
        let params = CfoParams {
            period: Some(10),
            scalar: Some(100.0),
        };
        let input = CfoInput::from_slice(&data, params);

        let result = cfo(&input);
        assert!(
            result.is_ok(),
            "Should return Ok with all NaNs if size < period"
        );
        let output = result.unwrap();
        assert_eq!(output.values.len(), data.len());
        for &val in &output.values {
            assert!(val.is_nan(), "Expected NaN for insufficient data length");
        }
    }

    #[test]
    fn test_cfo_very_small_data_set() {
        let data = [42.0];
        let params = CfoParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let input = CfoInput::from_slice(&data, params);

        let result = cfo(&input).expect("Should not panic even if data < period");
        assert_eq!(result.values.len(), data.len());
        assert!(
            result.values[0].is_nan(),
            "Expected NaN for insufficient data"
        );
    }

    #[test]
    fn test_cfo_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = CfoParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let first_input = CfoInput::from_candles(&candles, "close", first_params);
        let first_result = cfo(&first_input).expect("Failed to calculate first CFO");
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = CfoParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let second_input = CfoInput::from_slice(&first_result.values, second_params);
        let second_result = cfo(&second_input).expect("Failed to calculate second CFO");
        assert_eq!(second_result.values.len(), first_result.values.len());

        for i in 240..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 28, found NaN at {}",
                i
            );
        }
    }

    #[test]
    fn test_cfo_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let params = CfoParams {
            period: Some(14),
            scalar: Some(100.0),
        };
        let input = CfoInput::from_candles(&candles, "close", params);
        let cfo_result = cfo(&input).expect("Failed to calculate CFO");

        assert_eq!(
            cfo_result.values.len(),
            close_prices.len(),
            "CFO length mismatch"
        );

        if cfo_result.values.len() > 240 {
            for i in 240..cfo_result.values.len() {
                assert!(
                    !cfo_result.values[i].is_nan(),
                    "Expected no NaN after index 240, found NaN at {}",
                    i
                );
            }
        }
    }
}
