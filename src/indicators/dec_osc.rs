/// # Decycler Oscillator (DEC_OSC)
///
/// An oscillator that applies two sequential high-pass filters (2-pole) to remove
/// cyclical components from the data (e.g., price). The residual is then scaled by
/// `k` and expressed as a percentage of the original input series.
///
/// ## Parameters
/// - **hp_period**: The period used for the primary high-pass filter. Defaults to 125.
/// - **k**: Multiplier for the final oscillator values. Defaults to 1.0.
///
/// ## Errors
/// - **EmptyData**: dec_osc: Input data slice is empty.
/// - **InvalidPeriod**: dec_osc: `hp_period` is below 2 or exceeds the data length.
/// - **NotEnoughValidData**: dec_osc: Fewer than 2 valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: dec_osc: All input data values are `NaN`.
/// - **InvalidK**: dec_osc: `k` is `NaN` or non-positive.
///
/// ## Returns
/// - **`Ok(DecOscOutput)`** on success, containing a `Vec<f64>` matching the input length.
///   The first few values will be `NaN` until enough points are available (2-pole filter).
/// - **`Err(DecOscError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use std::f64::consts::PI;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DecOscData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DecOscOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DecOscParams {
    pub hp_period: Option<usize>,
    pub k: Option<f64>,
}

impl Default for DecOscParams {
    fn default() -> Self {
        Self {
            hp_period: Some(125),
            k: Some(1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecOscInput<'a> {
    pub data: DecOscData<'a>,
    pub params: DecOscParams,
}

impl<'a> DecOscInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: DecOscParams) -> Self {
        Self {
            data: DecOscData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: DecOscParams) -> Self {
        Self {
            data: DecOscData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: DecOscData::Candles {
                candles,
                source: "close",
            },
            params: DecOscParams::default(),
        }
    }

    pub fn get_hp_period(&self) -> usize {
        self.params
            .hp_period
            .unwrap_or_else(|| DecOscParams::default().hp_period.unwrap())
    }

    pub fn get_k(&self) -> f64 {
        self.params
            .k
            .unwrap_or_else(|| DecOscParams::default().k.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum DecOscError {
    #[error("dec_osc: Empty data provided.")]
    EmptyData,
    #[error("dec_osc: All values are NaN.")]
    AllValuesNaN,
    #[error("dec_osc: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("dec_osc: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("dec_osc: Invalid K: k = {k}")]
    InvalidK { k: f64 },
}

#[inline]
pub fn dec_osc(input: &DecOscInput) -> Result<DecOscOutput, DecOscError> {
    let data: &[f64] = match &input.data {
        DecOscData::Candles { candles, source } => source_type(candles, source),
        DecOscData::Slice(slice) => slice,
    };
    let len = data.len();

    if len == 0 {
        return Err(DecOscError::EmptyData);
    }

    let hp_period = input.get_hp_period();
    let k_val = input.get_k();

    if hp_period < 2 || hp_period > len {
        return Err(DecOscError::InvalidPeriod {
            period: hp_period,
            data_len: len,
        });
    }
    if k_val <= 0.0 || k_val.is_nan() {
        return Err(DecOscError::InvalidK { k: k_val });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(DecOscError::AllValuesNaN),
    };

    if (len - first_valid_idx) < 2 {
        return Err(DecOscError::NotEnoughValidData {
            needed: 2,
            valid: len - first_valid_idx,
        });
    }

    let mut out = vec![f64::NAN; len];

    let half_period = (hp_period as f64) * 0.5;

    let angle1 = 2.0 * PI * 0.707 / (hp_period as f64);
    let sin1 = angle1.sin();
    let cos1 = angle1.cos();
    let alpha1 = 1.0 + ((sin1 - 1.0) / cos1);
    let c1 = (1.0 - alpha1 / 2.0) * (1.0 - alpha1 / 2.0);
    let one_minus_alpha1 = 1.0 - alpha1;
    let one_minus_alpha1_sq = one_minus_alpha1 * one_minus_alpha1;

    let angle2 = 2.0 * PI * 0.707 / half_period;
    let sin2 = angle2.sin();
    let cos2 = angle2.cos();
    let alpha2 = 1.0 + ((sin2 - 1.0) / cos2);
    let c2 = (1.0 - alpha2 / 2.0) * (1.0 - alpha2 / 2.0);
    let one_minus_alpha2 = 1.0 - alpha2;
    let one_minus_alpha2_sq = one_minus_alpha2 * one_minus_alpha2;

    let mut hp_prev_2;
    let mut hp_prev_1;
    let mut decosc_prev_2;
    let mut decosc_prev_1;

    {
        let val0 = data[first_valid_idx];
        out[first_valid_idx] = f64::NAN;
        hp_prev_2 = val0;
        hp_prev_1 = val0;
        decosc_prev_2 = 0.0;
        decosc_prev_1 = 0.0;
    }

    if first_valid_idx + 1 < len {
        let val1 = data[first_valid_idx + 1];
        out[first_valid_idx + 1] = f64::NAN;
        hp_prev_2 = hp_prev_1;
        hp_prev_1 = val1;

        let dec = val1 - hp_prev_1;
        decosc_prev_2 = decosc_prev_1;
        decosc_prev_1 = dec;
    }
    for i in (first_valid_idx + 2)..len {
        let d0 = data[i];

        let d1 = data[i - 1];
        let d2 = data[i - 2];

        let hp0 = c1 * d0 - 2.0 * c1 * d1 + c1 * d2 + 2.0 * one_minus_alpha1 * hp_prev_1
            - one_minus_alpha1_sq * hp_prev_2;

        let dec = d0 - hp0;

        let d_dec1 = d1 - hp_prev_1;
        let d_dec2 = d2 - hp_prev_2;

        let decosc0 =
            c2 * dec - 2.0 * c2 * d_dec1 + c2 * d_dec2 + 2.0 * one_minus_alpha2 * decosc_prev_1
                - one_minus_alpha2_sq * decosc_prev_2;

        out[i] = 100.0 * k_val * decosc0 / d0;

        hp_prev_2 = hp_prev_1;
        hp_prev_1 = hp0;
        decosc_prev_2 = decosc_prev_1;
        decosc_prev_1 = decosc0;
    }

    Ok(DecOscOutput { values: out })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_dec_osc_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = DecOscParams {
            hp_period: None,
            k: None,
        };
        let input_default = DecOscInput::from_candles(&candles, "close", default_params);
        let output_default = dec_osc(&input_default).expect("Failed dec_osc with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_50 = DecOscParams {
            hp_period: Some(50),
            k: Some(1.0),
        };
        let input_period_50 = DecOscInput::from_candles(&candles, "hl2", params_period_50.clone());
        let output_period_50 =
            dec_osc(&input_period_50).expect("Failed dec_osc with period=50, source=hl2");
        assert_eq!(output_period_50.values.len(), candles.close.len());

        let params_custom = DecOscParams {
            hp_period: Some(100),
            k: Some(2.0),
        };
        let input_custom = DecOscInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = dec_osc(&input_custom).expect("Failed dec_osc fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_dec_osc_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let source_data = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = DecOscParams {
            hp_period: Some(125),
            k: Some(1.0),
        };
        let input = DecOscInput::from_candles(&candles, "close", params);
        let dec_osc_result = dec_osc(&input).expect("Failed to calculate dec_osc");

        assert_eq!(
            dec_osc_result.values.len(),
            source_data.len(),
            "dec_osc length mismatch"
        );

        if dec_osc_result.values.len() > 5 {
            let expected_last_five = [
                -1.5036367540303395,
                -1.4037875172207006,
                -1.3174199471429475,
                -1.2245874070642693,
                -1.1638422627265639,
            ];
            let start_index = dec_osc_result.values.len() - 5;
            let actual_last_five = &dec_osc_result.values[start_index..];
            for (i, &value) in actual_last_five.iter().enumerate() {
                let expected_value = expected_last_five[i];
                let diff = (value - expected_value).abs();
                assert!(
                    diff < 1e-7,
                    "DEC_OSC mismatch at index {}: expected {}, got {}",
                    i,
                    expected_value,
                    value
                );
            }
        }
    }

    #[test]
    fn test_dec_osc_params_with_default_params() {
        let default_params = DecOscParams::default();
        assert_eq!(
            default_params.hp_period,
            Some(125),
            "Expected hp_period=125 in default parameters"
        );
        assert_eq!(
            default_params.k,
            Some(1.0),
            "Expected k=1.0 in default parameters"
        );
    }

    #[test]
    fn test_dec_osc_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = DecOscInput::with_default_candles(&candles);
        match input.data {
            DecOscData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected DecOscData::Candles variant"),
        }
    }

    #[test]
    fn test_dec_osc_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = DecOscParams {
            hp_period: Some(0),
            k: Some(1.0),
        };
        let input = DecOscInput::from_slice(&input_data, params);

        let result = dec_osc(&input);
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
    fn test_dec_osc_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = DecOscParams {
            hp_period: Some(10),
            k: Some(1.0),
        };
        let input = DecOscInput::from_slice(&input_data, params);

        let result = dec_osc(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_dec_osc_very_small_data_set() {
        let input_data = [42.0];
        let params = DecOscParams {
            hp_period: Some(125),
            k: Some(1.0),
        };
        let input = DecOscInput::from_slice(&input_data, params);

        let result = dec_osc(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period or not enough points"
        );
    }

    #[test]
    fn test_dec_osc_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = DecOscParams {
            hp_period: Some(50),
            k: Some(1.0),
        };
        let first_input = DecOscInput::from_candles(&candles, "close", first_params);
        let first_result = dec_osc(&first_input).expect("Failed to calculate first dec_osc");

        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First dec_osc output length mismatch"
        );

        let second_params = DecOscParams {
            hp_period: Some(50),
            k: Some(1.0),
        };
        let second_input = DecOscInput::from_slice(&first_result.values, second_params);
        let second_result = dec_osc(&second_input).expect("Failed to calculate second dec_osc");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second dec_osc output length mismatch"
        );
    }
}
