/// # Polarized Fractal Efficiency (PFE)
///
/// A technical indicator that measures how efficiently price moves from one point to another
/// over a given period. It outputs positive values for upward movement efficiency and negative
/// values for downward movement efficiency, then smooths the result with an EMA.
///
/// ## Parameters
/// - **period**: The lookback window size. Defaults to 10.
/// - **smoothing**: The smoothing period for the EMA. Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: pfe: Input data slice is empty.
/// - **InvalidPeriod**: pfe: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: pfe: Fewer than `period` valid (non-`NaN`) data points remain after the first valid index.
/// - **AllValuesNaN**: pfe: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(PfeOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN` until the indicator can be calculated.
/// - **`Err(PfeError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use std::f64;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum PfeData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PfeOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PfeParams {
    pub period: Option<usize>,
    pub smoothing: Option<usize>,
}

impl Default for PfeParams {
    fn default() -> Self {
        Self {
            period: Some(10),
            smoothing: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PfeInput<'a> {
    pub data: PfeData<'a>,
    pub params: PfeParams,
}

impl<'a> PfeInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: PfeParams) -> Self {
        Self {
            data: PfeData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: PfeParams) -> Self {
        Self {
            data: PfeData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: PfeData::Candles {
                candles,
                source: "close",
            },
            params: PfeParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| PfeParams::default().period.unwrap())
    }

    pub fn get_smoothing(&self) -> usize {
        self.params
            .smoothing
            .unwrap_or_else(|| PfeParams::default().smoothing.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum PfeError {
    #[error("pfe: Empty data provided.")]
    EmptyData,
    #[error("pfe: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("pfe: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("pfe: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn pfe(input: &PfeInput) -> Result<PfeOutput, PfeError> {
    let data: &[f64] = match &input.data {
        PfeData::Candles { candles, source } => source_type(candles, source),
        PfeData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(PfeError::EmptyData);
    }

    let period = input.get_period();
    let smoothing = input.get_smoothing();

    if period == 0 || period > data.len() {
        return Err(PfeError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(PfeError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(PfeError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let ln = period.saturating_sub(1);

    let diff_len = data.len().saturating_sub(ln);
    let mut diff_array = vec![f64::NAN; diff_len];
    for i in 0..diff_len {
        diff_array[i] = data[i + ln] - data[i];
    }

    let mut a_array = vec![f64::NAN; diff_len];
    for i in 0..diff_len {
        let d = diff_array[i];
        a_array[i] = (d.powi(2) + (period as f64).powi(2)).sqrt();
    }

    let mut b_array = vec![f64::NAN; diff_len];
    for i in 0..diff_len {
        let start = i;
        let end = i + ln;
        let mut b_sum = 0.0;
        for j in start..end {
            let step_diff = data[j + 1] - data[j];
            b_sum += (1.0 + step_diff.powi(2)).sqrt();
        }
        b_array[i] = b_sum;
    }

    let mut pfe_tmp = vec![f64::NAN; diff_len];
    for i in 0..diff_len {
        if b_array[i].abs() < f64::EPSILON {
            pfe_tmp[i] = 0.0;
        } else {
            pfe_tmp[i] = 100.0 * a_array[i] / b_array[i];
        }
    }

    let mut signed_pfe = vec![f64::NAN; diff_len];
    for i in 0..diff_len {
        let d = diff_array[i];
        if d.is_nan() {
            signed_pfe[i] = f64::NAN;
        } else if d > 0.0 {
            signed_pfe[i] = pfe_tmp[i];
        } else {
            signed_pfe[i] = -pfe_tmp[i];
        }
    }

    let alpha = 2.0 / (smoothing as f64 + 1.0);
    let mut ema_array = vec![f64::NAN; diff_len];
    let mut started = false;
    let mut ema_val = 0.0;
    for i in 0..diff_len {
        let val = signed_pfe[i];
        if val.is_nan() {
            ema_array[i] = f64::NAN;
        } else if !started {
            ema_val = val;
            ema_array[i] = val;
            started = true;
        } else {
            ema_val = alpha * val + (1.0 - alpha) * ema_val;
            ema_array[i] = ema_val;
        }
    }
    let mut pfe_values = vec![f64::NAN; data.len()];
    for (i, &val) in ema_array.iter().enumerate() {
        let out_idx = i + ln;
        if out_idx < pfe_values.len() {
            pfe_values[out_idx] = val;
        }
    }

    Ok(PfeOutput { values: pfe_values })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_pfe_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = PfeParams {
            period: None,
            smoothing: None,
        };
        let input_default = PfeInput::from_candles(&candles, "close", default_params);
        let output_default = pfe(&input_default).expect("Failed PFE with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = PfeParams {
            period: Some(14),
            smoothing: Some(5),
        };
        let input_period_14 = PfeInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 =
            pfe(&input_period_14).expect("Failed PFE with period=14, source=hl2");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = PfeParams {
            period: Some(20),
            smoothing: Some(10),
        };
        let input_custom = PfeInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = pfe(&input_custom).expect("Failed PFE fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    #[ignore]
    fn test_pfe_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = PfeParams {
            period: Some(10),
            smoothing: Some(5),
        };
        let input = PfeInput::from_candles(&candles, "close", params);
        let pfe_result = pfe(&input).expect("Failed to calculate PFE");

        assert_eq!(
            pfe_result.values.len(),
            close_prices.len(),
            "PFE length mismatch"
        );

        let expected_last_five_pfe = [
            464.4835119128518,
            -311.47775707009305,
            63.47691006853603,
            -122.09984956859148,
            76.97379946575279,
        ];
        assert!(pfe_result.values.len() >= 5, "PFE length too short");
        let start_index = pfe_result.values.len() - 5;
        let result_last_five_pfe = &pfe_result.values[start_index..];
        for (i, &value) in result_last_five_pfe.iter().enumerate() {
            let expected_value = expected_last_five_pfe[i];
            assert!(
                (value - expected_value).abs() < 1e-8,
                "PFE mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        for i in 0..(10 - 1) {
            assert!(pfe_result.values[i].is_nan());
        }

        let default_input = PfeInput::with_default_candles(&candles);
        let default_pfe_result = pfe(&default_input).expect("Failed to calculate PFE defaults");
        assert_eq!(default_pfe_result.values.len(), close_prices.len());
    }

    #[test]
    fn test_pfe_params_with_default_params() {
        let default_params = PfeParams::default();
        assert_eq!(
            default_params.period,
            Some(10),
            "Expected period to default to 10"
        );
        assert_eq!(
            default_params.smoothing,
            Some(5),
            "Expected smoothing to default to 5"
        );
    }

    #[test]
    fn test_pfe_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = PfeInput::with_default_candles(&candles);
        match input.data {
            PfeData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected PfeData::Candles variant"),
        }
    }

    #[test]
    fn test_pfe_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = PfeParams {
            period: Some(0),
            smoothing: Some(5),
        };
        let input = PfeInput::from_slice(&input_data, params);

        let result = pfe(&input);
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
    fn test_pfe_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = PfeParams {
            period: Some(10),
            smoothing: Some(2),
        };
        let input = PfeInput::from_slice(&input_data, params);

        let result = pfe(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_pfe_very_small_data_set() {
        let input_data = [42.0];
        let params = PfeParams {
            period: Some(10),
            smoothing: Some(2),
        };
        let input = PfeInput::from_slice(&input_data, params);

        let result = pfe(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_pfe_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = PfeParams {
            period: Some(10),
            smoothing: Some(5),
        };
        let first_input = PfeInput::from_candles(&candles, "close", first_params);
        let first_result = pfe(&first_input).expect("Failed to calculate first PFE");

        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First PFE output length mismatch"
        );

        let second_params = PfeParams {
            period: Some(10),
            smoothing: Some(5),
        };
        let second_input = PfeInput::from_slice(&first_result.values, second_params);
        let second_result = pfe(&second_input).expect("Failed to calculate second PFE");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second PFE output length mismatch"
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
    fn test_pfe_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 10;
        let params = PfeParams {
            period: Some(period),
            smoothing: Some(5),
        };
        let input = PfeInput::from_candles(&candles, "close", params);
        let pfe_result = pfe(&input).expect("Failed to calculate PFE");

        assert_eq!(
            pfe_result.values.len(),
            close_prices.len(),
            "PFE length mismatch"
        );

        if pfe_result.values.len() > 240 {
            for i in 240..pfe_result.values.len() {
                assert!(
                    !pfe_result.values[i].is_nan(),
                    "Expected no NaN after index 240, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
