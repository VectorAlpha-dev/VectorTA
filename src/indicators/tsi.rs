use crate::indicators::ema::{ema, EmaData, EmaError, EmaInput, EmaOutput, EmaParams};
/// # True Strength Index (TSI)
///
/// A momentum oscillator that applies a double smoothing (using EMA) to the raw momentum
/// (price change) and its absolute value. TSI measures trend direction and magnitude,
/// oscillating between positive and negative values.
///
/// ## Formula
/// TSI = 100 * [ EMA( EMA( MOM(source, 1), long_period ), short_period ) ]
///             / [ EMA( EMA( abs( MOM(source, 1) ), long_period ), short_period ) ]
///
/// ## Parameters
/// - **long_period**: Default = 25
/// - **short_period**: Default = 13
///
/// ## Errors
/// - **EmptyData**: tsi: Input data slice is empty.
/// - **InvalidPeriod**: tsi: One or both periods are zero or exceed the data length.
/// - **NotEnoughValidData**: tsi: Fewer than `1 + long_period + short_period` valid
///   (non-`NaN`) data points remain after the first valid index.
/// - **AllValuesNaN**: tsi: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(TsiOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the indicator can be computed.
/// - **`Err(TsiError)`** otherwise.
use crate::indicators::mom::{mom, MomData, MomError, MomInput, MomOutput, MomParams};
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum TsiData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TsiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TsiParams {
    pub long_period: Option<usize>,
    pub short_period: Option<usize>,
}

impl Default for TsiParams {
    fn default() -> Self {
        Self {
            long_period: Some(25),
            short_period: Some(13),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TsiInput<'a> {
    pub data: TsiData<'a>,
    pub params: TsiParams,
}

impl<'a> TsiInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: TsiParams) -> Self {
        Self {
            data: TsiData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: TsiParams) -> Self {
        Self {
            data: TsiData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: TsiData::Candles {
                candles,
                source: "close",
            },
            params: TsiParams::default(),
        }
    }

    fn get_long_period(&self) -> usize {
        self.params
            .long_period
            .unwrap_or_else(|| TsiParams::default().long_period.unwrap())
    }

    fn get_short_period(&self) -> usize {
        self.params
            .short_period
            .unwrap_or_else(|| TsiParams::default().short_period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum TsiError {
    #[error("tsi: Empty data provided.")]
    EmptyData,
    #[error("tsi: One or both periods are invalid: long_period = {long_period}, short_period = {short_period}, data length = {data_len}")]
    InvalidPeriod {
        long_period: usize,
        short_period: usize,
        data_len: usize,
    },
    #[error("tsi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("tsi: All values are NaN.")]
    AllValuesNaN,
    #[error("tsi: Momentum sub-calculation error: {0}")]
    MomSubError(#[from] MomError),
    #[error("tsi: EMA sub-calculation error: {0}")]
    EmaSubError(#[from] EmaError),
}

#[inline]
pub fn tsi(input: &TsiInput) -> Result<TsiOutput, TsiError> {
    let data: &[f64] = match &input.data {
        TsiData::Candles { candles, source } => source_type(candles, source),
        TsiData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(TsiError::EmptyData);
    }

    let long_period = input.get_long_period();
    let short_period = input.get_short_period();

    if long_period == 0
        || short_period == 0
        || long_period > data.len()
        || short_period > data.len()
    {
        return Err(TsiError::InvalidPeriod {
            long_period,
            short_period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(TsiError::AllValuesNaN),
    };

    let needed = 1 + long_period + short_period;
    if (data.len() - first_valid_idx) < needed {
        return Err(TsiError::NotEnoughValidData {
            needed,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut tsi_values = vec![f64::NAN; data.len()];

    let mom_input = MomInput::from_slice(&data[first_valid_idx..], MomParams { period: Some(1) });
    let mom_output: MomOutput = mom(&mom_input)?;
    let abs_mom_values: Vec<f64> = mom_output
        .values
        .iter()
        .map(|v| if v.is_nan() { f64::NAN } else { v.abs() })
        .collect();

    let ema_long_input_numer = EmaInput::from_slice(
        &mom_output.values,
        EmaParams {
            period: Some(long_period),
        },
    );
    let ema_long_numer: EmaOutput = ema(&ema_long_input_numer)?;

    let ema_short_input_numer = EmaInput::from_slice(
        &ema_long_numer.values,
        EmaParams {
            period: Some(short_period),
        },
    );
    let ema_short_numer: EmaOutput = ema(&ema_short_input_numer)?;

    let ema_long_input_denom = EmaInput::from_slice(
        &abs_mom_values,
        EmaParams {
            period: Some(long_period),
        },
    );
    let ema_long_denom: EmaOutput = ema(&ema_long_input_denom)?;

    let ema_short_input_denom = EmaInput::from_slice(
        &ema_long_denom.values,
        EmaParams {
            period: Some(short_period),
        },
    );
    let ema_short_denom: EmaOutput = ema(&ema_short_input_denom)?;

    for i in 0..data.len() {
        if i < first_valid_idx {
            tsi_values[i] = f64::NAN;
        } else {
            let idx = i - first_valid_idx;
            let numer = ema_short_numer.values[idx];
            let denom = ema_short_denom.values[idx];
            if numer.is_nan() || denom.is_nan() || denom == 0.0 {
                tsi_values[i] = f64::NAN;
            } else {
                tsi_values[i] = 100.0 * (numer / denom);
            }
        }
    }

    Ok(TsiOutput { values: tsi_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_tsi_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = TsiParams {
            long_period: None,
            short_period: None,
        };
        let input_default = TsiInput::from_candles(&candles, "close", default_params);
        let output_default = tsi(&input_default).expect("Failed TSI with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_long_30 = TsiParams {
            long_period: Some(30),
            short_period: None,
        };
        let input_long_30 = TsiInput::from_candles(&candles, "hl2", params_long_30);
        let output_long_30 = tsi(&input_long_30).expect("Failed TSI with long=30");
        assert_eq!(output_long_30.values.len(), candles.close.len());

        let params_custom = TsiParams {
            long_period: Some(20),
            short_period: Some(10),
        };
        let input_custom = TsiInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = tsi(&input_custom).expect("Failed TSI fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_tsi_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = TsiParams {
            long_period: Some(25),
            short_period: Some(13),
        };
        let input = TsiInput::from_candles(&candles, "close", params);
        let tsi_result = tsi(&input).expect("Failed to calculate TSI");

        assert_eq!(tsi_result.values.len(), close_prices.len());

        let expected_last_five_tsi = [
            -17.757654061849838,
            -17.367527062626184,
            -17.305577681249513,
            -16.937565646991143,
            -17.61825617316731,
        ];
        assert!(tsi_result.values.len() >= 5);
        let start_index = tsi_result.values.len() - 5;
        let result_last_five_tsi = &tsi_result.values[start_index..];
        for (i, &value) in result_last_five_tsi.iter().enumerate() {
            let expected_value = expected_last_five_tsi[i];
            assert!(
                (value - expected_value).abs() < 1e-7,
                "TSI mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_tsi_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = TsiParams {
            long_period: Some(0),
            short_period: Some(13),
        };
        let input = TsiInput::from_slice(&input_data, params);

        let result = tsi(&input);
        assert!(result.is_err(), "Expected an error for zero long_period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("One or both periods are invalid"),
                "Expected 'InvalidPeriod' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_tsi_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = TsiParams {
            long_period: Some(25),
            short_period: Some(13),
        };
        let input = TsiInput::from_slice(&input_data, params);

        let result = tsi(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_tsi_very_small_data_set() {
        let input_data = [42.0];
        let params = TsiParams {
            long_period: Some(25),
            short_period: Some(13),
        };
        let input = TsiInput::from_slice(&input_data, params);

        let result = tsi(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than needed chain length"
        );
    }

    #[test]
    fn test_tsi_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = TsiInput::with_default_candles(&candles);
        match input.data {
            TsiData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected TsiData::Candles variant"),
        }
    }

    #[test]
    fn test_tsi_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = TsiParams {
            long_period: Some(25),
            short_period: Some(13),
        };
        let first_input = TsiInput::from_candles(&candles, "close", first_params);
        let first_result = tsi(&first_input).expect("Failed to calculate first TSI");
        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First TSI output length mismatch"
        );

        let second_params = TsiParams {
            long_period: Some(25),
            short_period: Some(13),
        };
        let second_input = TsiInput::from_slice(&first_result.values, second_params);
        let second_result = tsi(&second_input).expect("Failed to calculate second TSI");
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second TSI output length mismatch"
        );
    }

    #[test]
    fn test_tsi_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let params = TsiParams {
            long_period: Some(25),
            short_period: Some(13),
        };
        let input = TsiInput::from_candles(&candles, "close", params);
        let tsi_result = tsi(&input).expect("Failed to calculate TSI");

        assert_eq!(tsi_result.values.len(), close_prices.len());

        if tsi_result.values.len() > 240 {
            for i in 240..tsi_result.values.len() {
                assert!(
                    !tsi_result.values[i].is_nan(),
                    "Expected no NaN after index 240, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
