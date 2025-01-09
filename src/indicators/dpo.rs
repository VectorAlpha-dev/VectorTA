/// # Detrended Price Oscillator (DPO)
///
/// The Detrended Price Oscillator (DPO) is used to identify cycles in price data.
/// It subtracts the moving average (based on the `period`) from the price at a
/// shifted index (period/2 + 1 bars back). This "detrends" the price series, making
/// cycles more visible.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: dpo: Input data slice is empty.
/// - **InvalidPeriod**: dpo: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: dpo: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: dpo: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(DpoOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the DPO window is filled.
/// - **`Err(DpoError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum DpoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DpoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DpoParams {
    pub period: Option<usize>,
}

impl Default for DpoParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct DpoInput<'a> {
    pub data: DpoData<'a>,
    pub params: DpoParams,
}

impl<'a> DpoInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: DpoParams) -> Self {
        Self {
            data: DpoData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: DpoParams) -> Self {
        Self {
            data: DpoData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: DpoData::Candles {
                candles,
                source: "close",
            },
            params: DpoParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| DpoParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DpoError {
    #[error("dpo: Empty data provided for DPO.")]
    EmptyData,
    #[error("dpo: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("dpo: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("dpo: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn dpo(input: &DpoInput) -> Result<DpoOutput, DpoError> {
    let data: &[f64] = match &input.data {
        DpoData::Candles { candles, source } => source_type(candles, source),
        DpoData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(DpoError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(DpoError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(DpoError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(DpoError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let back = period / 2 + 1;
    let mut dpo_values = vec![f64::NAN; data.len()];
    let mut sum = 0.0;
    let scale = 1.0 / (period as f64);

    for i in first_valid_idx..data.len() {
        sum += data[i];
        if i >= first_valid_idx + period {
            sum -= data[i - period];
        }
        if i >= first_valid_idx + period - 1 && i >= back {
            dpo_values[i] = data[i - back] - (sum * scale);
        }
    }

    Ok(DpoOutput { values: dpo_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_dpo_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = DpoParams { period: None };
        let input_default = DpoInput::from_candles(&candles, "close", default_params);
        let output_default = dpo(&input_default).expect("Failed DPO with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_10 = DpoParams { period: Some(10) };
        let input_period_10 = DpoInput::from_candles(&candles, "hl2", params_period_10);
        let output_period_10 =
            dpo(&input_period_10).expect("Failed DPO with period=10, source=hl2");
        assert_eq!(output_period_10.values.len(), candles.close.len());

        let params_custom = DpoParams { period: Some(14) };
        let input_custom = DpoInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = dpo(&input_custom).expect("Failed DPO fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_dpo_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = DpoParams { period: Some(5) };
        let input = DpoInput::from_candles(&candles, "close", params);
        let dpo_result = dpo(&input).expect("Failed to calculate DPO");

        assert_eq!(
            dpo_result.values.len(),
            close_prices.len(),
            "DPO length mismatch"
        );

        let expected_last_five_dpo = [
            65.3999999999287,
            131.3999999999287,
            32.599999999925785,
            98.3999999999287,
            117.99999999992724,
        ];
        assert!(
            dpo_result.values.len() >= 5,
            "DPO result is too short for this test"
        );

        let start_index = dpo_result.values.len() - 5;
        let result_last_five = &dpo_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five_dpo[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "DPO mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        for i in 0..(5 - 1) {
            assert!(dpo_result.values[i].is_nan());
        }
    }

    #[test]
    fn test_dpo_params_with_default_params() {
        let default_params = DpoParams::default();
        assert_eq!(
            default_params.period,
            Some(5),
            "Expected period to be 5 in default parameters"
        );
    }

    #[test]
    fn test_dpo_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = DpoInput::with_default_candles(&candles);
        match input.data {
            DpoData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected DpoData::Candles variant"),
        }
    }

    #[test]
    fn test_dpo_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = DpoParams { period: Some(0) };
        let input = DpoInput::from_slice(&input_data, params);

        let result = dpo(&input);
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
    fn test_dpo_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = DpoParams { period: Some(10) };
        let input = DpoInput::from_slice(&input_data, params);

        let result = dpo(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_dpo_very_small_data_set() {
        let input_data = [42.0];
        let params = DpoParams { period: Some(5) };
        let input = DpoInput::from_slice(&input_data, params);

        let result = dpo(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_dpo_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = DpoParams { period: Some(5) };
        let first_input = DpoInput::from_candles(&candles, "close", first_params);
        let first_result = dpo(&first_input).expect("Failed to calculate first DPO");

        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First DPO output length mismatch"
        );

        let second_params = DpoParams { period: Some(5) };
        let second_input = DpoInput::from_slice(&first_result.values, second_params);
        let second_result = dpo(&second_input).expect("Failed to calculate second DPO");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second DPO output length mismatch"
        );
    }

    #[test]
    fn test_dpo_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 5;
        let params = DpoParams {
            period: Some(period),
        };
        let input = DpoInput::from_candles(&candles, "close", params);
        let dpo_result = dpo(&input).expect("Failed to calculate DPO");

        assert_eq!(
            dpo_result.values.len(),
            close_prices.len(),
            "DPO length mismatch"
        );

        if dpo_result.values.len() > 20 {
            for i in 20..dpo_result.values.len() {
                assert!(
                    !dpo_result.values[i].is_nan(),
                    "Expected no NaN after index 20, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
