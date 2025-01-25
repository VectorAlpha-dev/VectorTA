/// # Variance (VAR)
///
/// Computes the rolling variance of an input data set over a specified window (`period`), with
/// an optional standard deviation factor (`nbdev`). The rolling variance at each point is
/// calculated as the average of the squared values minus the square of the average:
/// \[ VAR = (sum(x^2)/period) - (sum(x)/period)^2 \]
/// multiplied by `nbdev^2`.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 14.
/// - **nbdev**: The standard deviation factor (multiplied as `nbdev^2`). Defaults to 1.0.
///
/// ## Errors
/// - **EmptyData**: var: Input data slice is empty.
/// - **InvalidPeriod**: var: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: var: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: var: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(VarOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the rolling variance window is filled.
/// - **`Err(VarError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum VarData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VarOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VarParams {
    pub period: Option<usize>,
    pub nbdev: Option<f64>,
}

impl Default for VarParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            nbdev: Some(1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VarInput<'a> {
    pub data: VarData<'a>,
    pub params: VarParams,
}

impl<'a> VarInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: VarParams) -> Self {
        Self {
            data: VarData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: VarParams) -> Self {
        Self {
            data: VarData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VarData::Candles {
                candles,
                source: "close",
            },
            params: VarParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| VarParams::default().period.unwrap())
    }

    pub fn get_nbdev(&self) -> f64 {
        self.params
            .nbdev
            .unwrap_or_else(|| VarParams::default().nbdev.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum VarError {
    #[error("var: Empty data provided for VAR.")]
    EmptyData,
    #[error("var: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("var: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("var: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn var(input: &VarInput) -> Result<VarOutput, VarError> {
    let data: &[f64] = match &input.data {
        VarData::Candles { candles, source } => source_type(candles, source),
        VarData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(VarError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(VarError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let nbdev = input.get_nbdev();
    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(VarError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(VarError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut var_values = vec![f64::NAN; data.len()];
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &value in data[first_valid_idx..(first_valid_idx + period)].iter() {
        sum += value;
        sum_sq += value * value;
    }

    let period_f = period as f64;
    var_values[first_valid_idx + period - 1] =
        (sum_sq / period_f - (sum / period_f) * (sum / period_f)) * nbdev * nbdev;

    for i in (first_valid_idx + period)..data.len() {
        let old_val = data[i - period];
        let new_val = data[i];
        sum += new_val - old_val;
        sum_sq += new_val * new_val - old_val * old_val;
        var_values[i] = (sum_sq / period_f - (sum / period_f) * (sum / period_f)) * nbdev * nbdev;
    }

    Ok(VarOutput { values: var_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_var_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = VarParams {
            period: None,
            nbdev: None,
        };
        let input_default = VarInput::from_candles(&candles, "close", default_params);
        let output_default = var(&input_default).expect("Failed VAR with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = VarParams {
            period: Some(14),
            nbdev: Some(1.0),
        };
        let input_period_14 = VarInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 =
            var(&input_period_14).expect("Failed VAR with period=14, source=hl2");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = VarParams {
            period: Some(20),
            nbdev: Some(2.0),
        };
        let input_custom = VarInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = var(&input_custom).expect("Failed VAR fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_var_accuracy_small_data() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = VarParams {
            period: Some(5),
            nbdev: Some(1.0),
        };
        let input = VarInput::from_slice(&data, params);
        let output = var(&input).expect("VAR calc failed");
        assert_eq!(output.values.len(), 5);
        for i in 0..4 {
            assert!(output.values[i].is_nan());
        }
        assert!((output.values[4] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_var_accuracy() {
        // These values are for demonstration; in practice, use real data comparisons.
        // The user-provided reference values might come from a known dataset:
        // "350987.4081501961, 348493.9183540344, 302611.06121110916, 106092.2499871254, 121941.35202789307"
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = VarParams {
            period: Some(14),
            nbdev: Some(1.0),
        };
        let input = VarInput::from_candles(&candles, "close", params);
        let var_result = var(&input).expect("Failed to calculate VAR");

        assert_eq!(var_result.values.len(), candles.close.len());

        let expected_last_five_var = [
            350987.4081501961,
            348493.9183540344,
            302611.06121110916,
            106092.2499871254,
            121941.35202789307,
        ];
        assert!(
            var_result.values.len() >= 5,
            "VAR length too short for checking last 5"
        );
        let start_index = var_result.values.len() - 5;
        let result_last_five_var = &var_result.values[start_index..];
        for (i, &value) in result_last_five_var.iter().enumerate() {
            let expected_value = expected_last_five_var[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "VAR mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_var_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = VarInput::with_default_candles(&candles);
        match input.data {
            VarData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected VarData::Candles variant"),
        }
    }

    #[test]
    fn test_var_with_zero_period() {
        let data = [10.0, 20.0, 30.0];
        let params = VarParams {
            period: Some(0),
            nbdev: Some(1.0),
        };
        let input = VarInput::from_slice(&data, params);
        let result = var(&input);
        assert!(result.is_err(), "Expected error for zero period");
    }

    #[test]
    fn test_var_with_period_exceeding_data_length() {
        let data = [10.0, 20.0, 30.0];
        let params = VarParams {
            period: Some(10),
            nbdev: Some(1.0),
        };
        let input = VarInput::from_slice(&data, params);
        let result = var(&input);
        assert!(result.is_err(), "Expected error for period > data.len()");
    }

    #[test]
    fn test_var_very_small_data_set() {
        let data = [42.0];
        let params = VarParams {
            period: Some(14),
            nbdev: Some(1.0),
        };
        let input = VarInput::from_slice(&data, params);
        let result = var(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_var_all_nan_values() {
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let params = VarParams {
            period: Some(2),
            nbdev: Some(1.0),
        };
        let input = VarInput::from_slice(&data, params);
        let result = var(&input);
        assert!(result.is_err(), "Expected error for all NaN values");
    }

    #[test]
    fn test_var_with_slice_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = VarParams {
            period: Some(14),
            nbdev: Some(1.0),
        };
        let first_input = VarInput::from_candles(&candles, "close", first_params);
        let first_result = var(&first_input).expect("Failed to calculate first VAR");
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = VarParams {
            period: Some(14),
            nbdev: Some(1.0),
        };
        let second_input = VarInput::from_slice(&first_result.values, second_params);
        let second_result = var(&second_input).expect("Failed to calculate second VAR");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }
}
