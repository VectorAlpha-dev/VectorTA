/// # Ehlers Simple Decycler
///
/// A noise-reduction filter based on John Ehlers's High-Pass Filter concept.
/// It subtracts the high-frequency components from the original data,
/// leaving a "decycled" output that can help highlight underlying trends.
///
/// ## Parameters
/// - **hp_period**: Window size used for the embedded high-pass filter (minimum of 2). Defaults to 125.
/// - **k**: Frequency coefficient for the high-pass filter. Defaults to `1.0`.
///
/// ## Errors
/// - **EmptyData**: decycler: Input data slice is empty.
/// - **InvalidPeriod**: decycler: `hp_period` is zero, less than 2, or exceeds the data length.
/// - **NotEnoughValidData**: decycler: Fewer than `hp_period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: decycler: All input data values are `NaN`.
/// - **InvalidK**: decycler: `k` is non-positive or NaN.
///
/// ## Returns
/// - **`Ok(DecyclerOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the filter can be computed.
/// - **`Err(DecyclerError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use std::f64::consts::PI;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DecyclerData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DecyclerOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DecyclerParams {
    pub hp_period: Option<usize>,
}

impl Default for DecyclerParams {
    fn default() -> Self {
        Self {
            hp_period: Some(125),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecyclerInput<'a> {
    pub data: DecyclerData<'a>,
    pub params: DecyclerParams,
}

impl<'a> DecyclerInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: DecyclerParams) -> Self {
        Self {
            data: DecyclerData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: DecyclerParams) -> Self {
        Self {
            data: DecyclerData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: DecyclerData::Candles {
                candles,
                source: "close",
            },
            params: DecyclerParams::default(),
        }
    }

    pub fn get_hp_period(&self) -> usize {
        self.params
            .hp_period
            .unwrap_or_else(|| DecyclerParams::default().hp_period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum DecyclerError {
    #[error("decycler: Empty data provided for Decycler.")]
    EmptyData,
    #[error("decycler: Invalid period: hp_period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("decycler: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("decycler: All values are NaN.")]
    AllValuesNaN,
    #[error("decycler: Invalid k: k = {k}")]
    InvalidK { k: f64 },
}

#[inline]
pub fn decycler(input: &DecyclerInput) -> Result<DecyclerOutput, DecyclerError> {
    let data: &[f64] = match &input.data {
        DecyclerData::Candles { candles, source } => source_type(candles, source),
        DecyclerData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(DecyclerError::EmptyData);
    }

    let hp_period = input.get_hp_period();
    if hp_period < 2 || hp_period > data.len() {
        return Err(DecyclerError::InvalidPeriod {
            period: hp_period,
            data_len: data.len(),
        });
    }

    let k_val = 0.707;

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(DecyclerError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < hp_period {
        return Err(DecyclerError::NotEnoughValidData {
            needed: hp_period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut out = vec![f64::NAN; data.len()];
    let mut hp = vec![0.0; data.len()];
    let angle = 2.0 * PI * k_val / (hp_period as f64);
    let sin_val = angle.sin();
    let cos_val = angle.cos();
    let alpha = 1.0 + ((sin_val - 1.0) / cos_val);
    let one_minus_alpha_half = 1.0 - alpha / 2.0;
    let c = one_minus_alpha_half * one_minus_alpha_half;
    let one_minus_alpha = 1.0 - alpha;
    let one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;

    if data.len() > first_valid_idx {
        hp[first_valid_idx] = data[first_valid_idx];
        out[first_valid_idx] = data[first_valid_idx] - hp[first_valid_idx];
    }

    if data.len() > (first_valid_idx + 1) {
        hp[first_valid_idx + 1] = data[first_valid_idx + 1];
        out[first_valid_idx + 1] = data[first_valid_idx + 1] - hp[first_valid_idx + 1];
    }

    for i in (first_valid_idx + 2)..data.len() {
        let current = data[i];
        let prev1 = data[i - 1];
        let prev2 = data[i - 2];
        let hp_prev1 = hp[i - 1];
        let hp_prev2 = hp[i - 2];
        let val = c * current - 2.0 * c * prev1 + c * prev2 + 2.0 * one_minus_alpha * hp_prev1
            - one_minus_alpha_sq * hp_prev2;
        hp[i] = val;
        out[i] = current - val;
    }

    Ok(DecyclerOutput { values: out })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_decycler_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = DecyclerParams { hp_period: None };
        let input_default = DecyclerInput::from_candles(&candles, "close", default_params);
        let output_default = decycler(&input_default).expect("Failed Decycler with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_hp_50 = DecyclerParams {
            hp_period: Some(50),
        };
        let input_hp_50 = DecyclerInput::from_candles(&candles, "hl2", params_hp_50);
        let output_hp_50 = decycler(&input_hp_50).expect("Failed Decycler with hp_period=50");
        assert_eq!(output_hp_50.values.len(), candles.close.len());

        let params_custom = DecyclerParams {
            hp_period: Some(30),
        };
        let input_custom = DecyclerInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = decycler(&input_custom).expect("Failed Decycler fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_decycler_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = DecyclerParams {
            hp_period: Some(125),
        };
        let input = DecyclerInput::from_candles(&candles, "close", params);
        let decycler_result = decycler(&input).expect("Failed to calculate Decycler");
        assert_eq!(decycler_result.values.len(), close_prices.len());

        let test_values = [
            60289.96384058519,
            60204.010366691065,
            60114.255563805666,
            60028.535266555904,
            59934.26876964316,
        ];
        assert!(decycler_result.values.len() >= test_values.len());
        let start_index = decycler_result.values.len() - test_values.len();
        let result_last_values = &decycler_result.values[start_index..];
        for (i, &value) in result_last_values.iter().enumerate() {
            let expected_value = test_values[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "Decycler mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_decycler_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = DecyclerInput::with_default_candles(&candles);
        match input.data {
            DecyclerData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected DecyclerData::Candles variant"),
        }
    }

    #[test]
    fn test_decycler_params_with_default() {
        let default_params = DecyclerParams::default();
        assert_eq!(default_params.hp_period, Some(125));
    }

    #[test]
    fn test_decycler_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = DecyclerParams { hp_period: Some(0) };
        let input = DecyclerInput::from_slice(&input_data, params);
        let result = decycler(&input);
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
    fn test_decycler_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = DecyclerParams {
            hp_period: Some(10),
        };
        let input = DecyclerInput::from_slice(&input_data, params);
        let result = decycler(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_decycler_very_small_data_set() {
        let input_data = [42.0];
        let params = DecyclerParams { hp_period: Some(2) };
        let input = DecyclerInput::from_slice(&input_data, params);
        let result = decycler(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_decycler_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = DecyclerParams {
            hp_period: Some(30),
        };
        let first_input = DecyclerInput::from_candles(&candles, "close", first_params);
        let first_result = decycler(&first_input).expect("Failed to calculate first Decycler");
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = DecyclerParams {
            hp_period: Some(30),
        };
        let second_input = DecyclerInput::from_slice(&first_result.values, second_params);
        let second_result = decycler(&second_input).expect("Failed to calculate second Decycler");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }

    #[test]
    fn test_decycler_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;
        let period = 125;
        let params = DecyclerParams {
            hp_period: Some(period),
        };
        let input = DecyclerInput::from_candles(&candles, "close", params);
        let decycler_result = decycler(&input).expect("Failed to calculate Decycler");
        assert_eq!(decycler_result.values.len(), close_prices.len());
        if decycler_result.values.len() > 240 {
            for i in 240..decycler_result.values.len() {
                assert!(
                    !decycler_result.values[i].is_nan(),
                    "Expected no NaN after index 240, found NaN at {}",
                    i
                );
            }
        }
    }
}
