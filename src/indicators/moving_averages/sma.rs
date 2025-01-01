use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum SmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SmaParams {
    pub period: Option<usize>,
}

impl Default for SmaParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct SmaInput<'a> {
    pub data: SmaData<'a>,
    pub params: SmaParams,
}

impl<'a> SmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: SmaParams) -> Self {
        Self {
            data: SmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: SmaParams) -> Self {
        Self {
            data: SmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SmaData::Candles {
                candles,
                source: "close",
            },
            params: SmaParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SmaParams::default().period.unwrap())
    }
}
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SmaError {
    #[error("Empty data provided for SMA.")]
    EmptyData,
    #[error("All values are NaN.")]
    AllValuesNaN,
    #[error("Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn sma(input: &SmaInput) -> Result<SmaOutput, SmaError> {
    let data: &[f64] = match &input.data {
        SmaData::Candles { candles, source } => source_type(candles, source),
        SmaData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(SmaError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(SmaError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(SmaError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(SmaError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut sma_values = vec![f64::NAN; data.len()];
    let mut sum = 0.0;
    for &value in data[first_valid_idx..(first_valid_idx + period)].iter() {
        sum += value;
    }

    let inv_period = 1.0 / (period as f64);
    sma_values[first_valid_idx + period - 1] = sum * inv_period;

    for i in (first_valid_idx + period)..data.len() {
        sum += data[i] - data[i - period];
        sma_values[i] = sum * inv_period;
    }

    Ok(SmaOutput { values: sma_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_sma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = SmaParams { period: None };
        let input_default = SmaInput::from_candles(&candles, "close", default_params);
        let output_default = sma(&input_default).expect("Failed SMA with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = SmaParams { period: Some(14) };
        let input_period_14 = SmaInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 =
            sma(&input_period_14).expect("Failed SMA with period=14, source=hl2");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = SmaParams { period: Some(20) };
        let input_custom = SmaInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = sma(&input_custom).expect("Failed SMA fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_sma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = SmaParams { period: Some(9) };
        let input = SmaInput::from_candles(&candles, "close", params);
        let sma_result = sma(&input).expect("Failed to calculate SMA");

        assert_eq!(
            sma_result.values.len(),
            close_prices.len(),
            "SMA length mismatch"
        );

        let expected_last_five_sma = [59180.8, 59175.0, 59129.4, 59085.4, 59133.7];
        assert!(sma_result.values.len() >= 5, "SMA length too short");
        let start_index = sma_result.values.len() - 5;
        let result_last_five_sma = &sma_result.values[start_index..];
        for (i, &value) in result_last_five_sma.iter().enumerate() {
            let expected_value = expected_last_five_sma[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "SMA mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period: usize = 9;
        for i in 0..(period - 1) {
            assert!(sma_result.values[i].is_nan());
        }

        let default_input = SmaInput::with_default_candles(&candles);
        let default_sma_result = sma(&default_input).expect("Failed to calculate SMA defaults");
        assert_eq!(default_sma_result.values.len(), close_prices.len());
    }

    #[test]
    fn test_sma_params_with_default_params() {
        let default_params = SmaParams::default();
        assert_eq!(
            default_params.period,
            Some(9),
            "Expected period to be None in default parameters"
        );
    }

    #[test]
    fn test_sma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = SmaInput::with_default_candles(&candles);
        match input.data {
            SmaData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected SmaData::Candles variant"),
        }
    }

    #[test]
    fn test_sma_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = SmaParams { period: Some(0) };
        let input = SmaInput::from_slice(&input_data, params);

        let result = sma(&input);
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
    fn test_sma_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = SmaParams { period: Some(10) };
        let input = SmaInput::from_slice(&input_data, params);

        let result = sma(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_sma_very_small_data_set() {
        let input_data = [42.0];
        let params = SmaParams { period: Some(9) };
        let input = SmaInput::from_slice(&input_data, params);

        let result = sma(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_sma_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = SmaParams { period: Some(14) };
        let first_input = SmaInput::from_candles(&candles, "close", first_params);
        let first_result = sma(&first_input).expect("Failed to calculate first SMA");

        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First SMA output length mismatch"
        );

        let second_params = SmaParams { period: Some(14) };
        let second_input = SmaInput::from_slice(&first_result.values, second_params);
        let second_result = sma(&second_input).expect("Failed to calculate second SMA");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second SMA output length mismatch"
        );

        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 14, but found NaN at index {}",
                i
            );
        }
    }

    #[test]
    fn test_sma_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 9;
        let params = SmaParams {
            period: Some(period),
        };
        let input = SmaInput::from_candles(&candles, "close", params);
        let sma_result = sma(&input).expect("Failed to calculate SMA");

        assert_eq!(
            sma_result.values.len(),
            close_prices.len(),
            "SMA length mismatch"
        );

        if sma_result.values.len() > 240 {
            for i in 240..sma_result.values.len() {
                assert!(
                    !sma_result.values[i].is_nan(),
                    "Expected no NaN after index 240, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
