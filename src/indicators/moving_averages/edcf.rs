use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum EdcfData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct EdcfParams {
    pub period: Option<usize>,
}

impl Default for EdcfParams {
    fn default() -> Self {
        EdcfParams { period: Some(15) }
    }
}

#[derive(Debug, Clone)]
pub struct EdcfOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EdcfInput<'a> {
    pub data: EdcfData<'a>,
    pub params: EdcfParams,
}

impl<'a> EdcfInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: EdcfParams) -> Self {
        Self {
            data: EdcfData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: EdcfParams) -> Self {
        Self {
            data: EdcfData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: EdcfData::Candles {
                candles,
                source: "close",
            },
            params: EdcfParams::default(),
        }
    }

    #[inline]
    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| EdcfParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EdcfError {
    #[error("No data provided to EDCF filter.")]
    NoData,

    #[error("All values are NaN.")]
    AllValuesNaN,

    #[error("Invalid period: period = {period}. Period must be > 0.")]
    InvalidPeriod { period: usize },

    #[error("Not enough valid data points to compute EDCF. Need at least {needed} valid points after index {idx}.")]
    NotEnoughValidData { needed: usize, idx: usize },

    #[error("NaN found in data after the first valid index.")]
    NaNFound,
}

#[inline]
pub fn edcf(input: &EdcfInput) -> Result<EdcfOutput, EdcfError> {
    let data: &[f64] = match &input.data {
        EdcfData::Candles { candles, source } => source_type(candles, source),
        EdcfData::Slice(slice) => slice,
    };

    let len = data.len();
    if len == 0 {
        return Err(EdcfError::NoData);
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(EdcfError::AllValuesNaN),
    };

    let period = input.get_period();
    if period == 0 {
        return Err(EdcfError::InvalidPeriod { period });
    }

    let needed = 2 * period;
    if (len - first_valid_idx) < needed {
        return Err(EdcfError::NotEnoughValidData {
            needed,
            idx: first_valid_idx,
        });
    }

    if data[first_valid_idx..].iter().any(|&v| v.is_nan()) {
        return Err(EdcfError::NaNFound);
    }

    let mut newseries = vec![0.0; len];
    let mut dist = vec![0.0; len];

    let dist_start = first_valid_idx + period;
    for k in dist_start..len {
        let xk = data[k];
        let mut sum_sq = 0.0;
        for lb in 1..period {
            let diff = xk - data[k - lb];
            sum_sq += diff * diff;
        }
        dist[k] = sum_sq;
    }

    let start_j = first_valid_idx + needed;
    for j in start_j..len {
        let mut num = 0.0;
        let mut coef_sum = 0.0;
        for i in 0..period {
            let k = j - i;
            let w = dist[k];
            num += w * data[k];
            coef_sum += w;
        }
        if coef_sum != 0.0 {
            newseries[j] = num / coef_sum;
        }
    }

    Ok(EdcfOutput { values: newseries })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_edcf_accuracy_last_five_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = EdcfInput::from_candles(&candles, "hl2", EdcfParams { period: Some(15) });
        let edcf_result = edcf(&input).expect("EDCF calculation failed");
        let edcf_values = &edcf_result.values;

        assert_eq!(
            edcf_values.len(),
            candles.close.len(),
            "EDCF output length does not match input length!"
        );

        let expected_last_five = [
            59593.332275678375,
            59731.70263288801,
            59766.41512339413,
            59655.66162110993,
            59332.492883847,
        ];

        assert!(
            edcf_values.len() >= expected_last_five.len(),
            "Not enough EDCF values for the test"
        );

        let start_index = edcf_values.len() - expected_last_five.len();
        let actual_last_five = &edcf_values[start_index..];

        for (i, (&actual, &expected)) in actual_last_five
            .iter()
            .zip(expected_last_five.iter())
            .enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "EDCF mismatch at index {}: expected {:.14}, got {:.14}",
                start_index + i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_edcf_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = EdcfInput::with_default_candles(&candles);
        match input.data {
            EdcfData::Candles { source, .. } => {
                assert_eq!(source, "close", "Default source should be 'close'.");
            }
            _ => panic!("Expected EdcfData::Candles variant"),
        }
        let period = input.get_period();
        assert_eq!(period, 15, "Default period should be 15.");
    }

    #[test]
    fn test_edcf_with_default_params() {
        let default_params = EdcfParams::default();
        assert_eq!(
            default_params.period,
            Some(15),
            "Expected default period to be Some(15)."
        );
    }

    #[test]
    fn test_edcf_with_zero_period() {
        let data = [10.0, 20.0, 30.0];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(0) });
        let result = edcf(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_edcf_with_no_data() {
        let data: [f64; 0] = [];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(15) });
        let result = edcf(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_edcf_with_period_exceeding_data_length() {
        let data = [10.0, 20.0, 30.0];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(10) });
        let result = edcf(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_edcf_very_small_data_set() {
        let data = [42.0];
        let input = EdcfInput::from_slice(&data, EdcfParams { period: Some(15) });
        let result = edcf(&input);
        assert!(result.is_err(), "Expected an error for insufficient data");
    }

    #[test]
    fn test_edcf_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_input =
            EdcfInput::from_candles(&candles, "close", EdcfParams { period: Some(15) });
        let first_result = edcf(&first_input).expect("First EDCF failed");
        let first_values = first_result.values;

        assert_eq!(
            first_values.len(),
            candles.close.len(),
            "First EDCF output length mismatch!"
        );

        let second_input = EdcfInput::from_slice(&first_values, EdcfParams { period: Some(5) });
        let second_result = edcf(&second_input).expect("Second EDCF failed");

        assert_eq!(
            second_result.values.len(),
            first_values.len(),
            "Second EDCF output length mismatch!"
        );

        if second_result.values.len() > 240 {
            for (i, &val) in second_result.values.iter().enumerate().skip(240) {
                assert!(
                    !val.is_nan(),
                    "Found NaN in second EDCF output at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_edcf_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = EdcfInput::from_candles(&candles, "close", EdcfParams { period: None });
        let result = edcf(&input).expect("EDCF calculation failed");
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "EDCF output length mismatch"
        );
    }

    #[test]
    fn test_edcf_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let period = 15;
        let input = EdcfInput::from_candles(
            &candles,
            "close",
            EdcfParams {
                period: Some(period),
            },
        );
        let result = edcf(&input).expect("EDCF calculation failed");
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "EDCF output length mismatch"
        );

        let start_index = 2 * period;
        if result.values.len() > start_index {
            for (i, &val) in result.values.iter().enumerate().skip(start_index) {
                assert!(
                    !val.is_nan(),
                    "Found NaN in EDCF output at index {} (>= 2 * period)",
                    i
                );
            }
        }
    }
}
