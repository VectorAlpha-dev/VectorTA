/// # High-Pass Filter (HP)
///
/// A digital filter that attenuates low-frequency components of the input data,
/// allowing higher-frequency fluctuations to pass through. This helps to remove
/// or reduce slow-moving trends or bias.
///
/// ## Parameters
/// - **period**: The size of the window (number of data points). Defaults to 48.
///
/// ## Errors
/// - **AllValuesNaN**: highpass: All input data values are `NaN`.
/// - **InvalidPeriod**: highpass: `period` is zero, or exceeds the data length, or the data length
///   is insufficient for filter calculations.
/// - **InvalidAlpha**: highpass: The cosine value (`cos_val`) is too close to zero, preventing
///   a valid computation of the alpha parameter.
///
/// ## Returns
/// - **`Ok(HighPassOutput)`** on success, containing a `Vec<f64>` matching the input length.
/// - **`Err(HighPassError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum HighPassData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HighPassParams {
    pub period: Option<usize>,
}

impl Default for HighPassParams {
    fn default() -> Self {
        Self { period: Some(48) }
    }
}

#[derive(Debug, Clone)]
pub struct HighPassOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HighPassInput<'a> {
    pub data: HighPassData<'a>,
    pub params: HighPassParams,
}

impl<'a> HighPassInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: HighPassParams) -> Self {
        Self {
            data: HighPassData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: HighPassParams) -> Self {
        Self {
            data: HighPassData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: HighPassData::Candles {
                candles,
                source: "close",
            },
            params: HighPassParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| HighPassParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum HighPassError {
    #[error("highpass: All values are NaN.")]
    AllValuesNaN,

    #[error("highpass: Invalid period or insufficient data for highpass calculation: period = {period}, data length = {data_len}.")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error(
        "highpass: Invalid alpha calculation. cos_val is too close to zero: cos_val = {cos_val}"
    )]
    InvalidAlpha { cos_val: f64 },
}

#[inline]
pub fn highpass(input: &HighPassInput) -> Result<HighPassOutput, HighPassError> {
    let data: &[f64] = match &input.data {
        HighPassData::Candles { candles, source } => source_type(candles, source),
        HighPassData::Slice(slice) => slice,
    };

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(HighPassError::AllValuesNaN),
    };

    let len = data.len();
    let period = input.params.period.unwrap_or(48);
    if len <= 2 || period == 0 || period > len {
        return Err(HighPassError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let k = 1.0;
    let two_pi_k_div = 2.0 * std::f64::consts::PI * k / (period as f64);
    let sin_val = two_pi_k_div.sin();
    let cos_val = two_pi_k_div.cos();

    if cos_val.abs() < 1e-15 {
        return Err(HighPassError::InvalidAlpha { cos_val });
    }

    let alpha = 1.0 + (sin_val - 1.0) / cos_val;

    let one_minus_half_alpha = 1.0 - alpha / 2.0;
    let one_minus_alpha = 1.0 - alpha;

    let mut newseries = vec![0.0; len];
    newseries[0] = data[0];

    for i in 1..len {
        let val = one_minus_half_alpha * data[i] - one_minus_half_alpha * data[i - 1]
            + one_minus_alpha * newseries[i - 1];
        newseries[i] = val;
    }

    Ok(HighPassOutput { values: newseries })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_highpass_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = HighPassParams { period: None };
        let input_default = HighPassInput::from_candles(&candles, "close", default_params);
        let output_default = highpass(&input_default).expect("Failed highpass with default params");
        assert_eq!(output_default.values.len(), candles.close.len());
        let params_period = HighPassParams { period: Some(36) };
        let input_period = HighPassInput::from_candles(&candles, "hl2", params_period);
        let output_period =
            highpass(&input_period).expect("Failed highpass with period=36, source=hl2");
        assert_eq!(output_period.values.len(), candles.close.len());
    }

    #[test]
    fn test_highpass_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HighPassInput::with_default_candles(&candles);
        let result = highpass(&input).expect("Failed to calculate highpass");
        let expected_last_five = [
            -265.1027020005024,
            -330.0916060058495,
            -422.7478979710918,
            -261.87532144673423,
            -698.9026088956363,
        ];
        assert_eq!(result.values.len(), candles.close.len());
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            assert!(
                (val - expected_last_five[i]).abs() < 1e-6,
                "Highpass mismatch at {}: expected {}, got {}",
                i,
                expected_last_five[i],
                val
            );
        }
        for val in &result.values {
            assert!(val.is_finite());
        }
    }
    #[test]
    fn test_highpass_params_with_default_params() {
        let default_params = HighPassParams::default();
        assert_eq!(default_params.period, Some(48));
    }

    #[test]
    fn test_highpass_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let input = HighPassInput::with_default_candles(&candles);
        match input.data {
            HighPassData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Unexpected data variant"),
        }
    }

    #[test]
    fn test_highpass_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = HighPassParams { period: Some(0) };
        let input = HighPassInput::from_slice(&input_data, params);
        let result = highpass(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("Invalid period or insufficient data"));
        }
    }

    #[test]
    fn test_highpass_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = HighPassParams { period: Some(48) };
        let input = HighPassInput::from_slice(&input_data, params);
        let result = highpass(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_highpass_very_small_data_set() {
        let input_data = [42.0, 43.0];
        let params = HighPassParams { period: Some(2) };
        let input = HighPassInput::from_slice(&input_data, params);
        let result = highpass(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("Invalid period or insufficient data"));
        }
    }

    #[test]
    fn test_highpass_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let first_params = HighPassParams { period: Some(36) };
        let first_input = HighPassInput::from_candles(&candles, "close", first_params);
        let first_result = highpass(&first_input).unwrap();
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = HighPassParams { period: Some(24) };
        let second_input = HighPassInput::from_slice(&first_result.values, second_params);
        let second_result = highpass(&second_input).unwrap();
        assert_eq!(second_result.values.len(), first_result.values.len());
        for val in &second_result.values[240..] {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_highpass_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let params = HighPassParams { period: Some(48) };
        let input = HighPassInput::from_candles(&candles, "close", params);
        let highpass_result = highpass(&input).unwrap();
        for val in &highpass_result.values {
            assert!(!val.is_nan());
        }
    }
}
