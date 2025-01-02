use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum PwmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PwmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PwmaParams {
    pub period: Option<usize>,
}

impl Default for PwmaParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct PwmaInput<'a> {
    pub data: PwmaData<'a>,
    pub params: PwmaParams,
}

impl<'a> PwmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: PwmaParams) -> Self {
        Self {
            data: PwmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: PwmaParams) -> Self {
        Self {
            data: PwmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: PwmaData::Candles {
                candles,
                source: "close",
            },
            params: PwmaParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| PwmaParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PwmaError {
    #[error("All values are NaN.")]
    AllValuesNaN,
    #[error("Invalid period specified for PWMA calculation: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("Pascal weights sum to zero for period = {period}")]
    PascalWeightsSumZero { period: usize },
}

#[inline]
pub fn pwma(input: &PwmaInput) -> Result<PwmaOutput, PwmaError> {
    let data: &[f64] = match &input.data {
        PwmaData::Candles { candles, source } => source_type(candles, source),
        PwmaData::Slice(slice) => slice,
    };

    let period = input.get_period();
    let len = data.len();

    if period == 0 || period > len {
        return Err(PwmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let weights = pascal_weights(period)?;

    let mut output = vec![f64::NAN; len];

    for (i, output_val) in output.iter_mut().enumerate().skip(period - 1) {
        let mut weighted_sum = 0.0;
        for (k, &weight) in weights.iter().enumerate() {
            let idx = i - k;
            weighted_sum += data[idx] * weight;
        }
        *output_val = weighted_sum;
    }

    Ok(PwmaOutput { values: output })
}

#[inline]
fn pascal_weights(period: usize) -> Result<Vec<f64>, PwmaError> {
    let n = period - 1;
    let mut row = Vec::with_capacity(period);

    for r in 0..=n {
        let c = combination_f64(n, r);
        row.push(c);
    }

    let sum: f64 = row.iter().sum();
    if sum == 0.0 {
        return Err(PwmaError::PascalWeightsSumZero { period });
    }

    for val in row.iter_mut() {
        *val /= sum;
    }

    Ok(row)
}
#[inline]
fn combination_f64(n: usize, r: usize) -> f64 {
    let r = r.min(n - r);
    if r == 0 {
        return 1.0;
    }

    let mut result = 1.0;
    for i in 0..r {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_pwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = PwmaParams { period: None };
        let input_default = PwmaInput::from_candles(&candles, "close", default_params);
        let output_default = pwma(&input_default).expect("Failed PWMA with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_custom = PwmaParams { period: Some(8) };
        let input_custom = PwmaInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = pwma(&input_custom).expect("Failed PWMA with custom params");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_pwma_accuracy() {
        let expected_last_five_pwma = [59313.25, 59309.6875, 59249.3125, 59175.625, 59094.875];
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let params = PwmaParams { period: Some(5) };
        let input = PwmaInput::from_candles(&candles, "close", params);
        let result = pwma(&input).expect("Failed to calculate PWMA");
        assert_eq!(result.values.len(), close_prices.len());
        assert!(result.values.len() >= 5);
        let start_index = result.values.len() - 5;
        let result_last_five = &result.values[start_index..];
        for (i, &val) in result_last_five.iter().enumerate() {
            let expected_val = expected_last_five_pwma[i];
            assert!(
                (val - expected_val).abs() < 1e-3,
                "PWMA mismatch at index {}, expected {}, got {}",
                i,
                expected_val,
                val
            );
        }
    }
    #[test]
    fn test_pwma_params_with_default_params() {
        let default_params = PwmaParams::default();
        assert_eq!(default_params.period, Some(5));
    }

    #[test]
    fn test_pwma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let input = PwmaInput::with_default_candles(&candles);
        match input.data {
            PwmaData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Unexpected data variant"),
        }
    }

    #[test]
    fn test_pwma_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = PwmaParams { period: Some(0) };
        let input = PwmaInput::from_slice(&input_data, params);
        let result = pwma(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("Invalid period specified for PWMA calculation"));
        }
    }

    #[test]
    fn test_pwma_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0];
        let params = PwmaParams { period: Some(5) };
        let input = PwmaInput::from_slice(&input_data, params);
        let result = pwma(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("Invalid period specified for PWMA calculation"));
        }
    }

    #[test]
    fn test_pwma_very_small_data_set() {
        let input_data = [42.0, 43.0];
        let params = PwmaParams { period: Some(2) };
        let input = PwmaInput::from_slice(&input_data, params);
        let result = pwma(&input).unwrap();
        assert_eq!(result.values.len(), input_data.len());
        assert!(result.values[0].is_nan());
        assert!(!result.values[1].is_nan());
    }

    #[test]
    fn test_pwma_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let first_params = PwmaParams { period: Some(5) };
        let first_input = PwmaInput::from_candles(&candles, "close", first_params);
        let first_result = pwma(&first_input).unwrap();
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = PwmaParams { period: Some(3) };
        let second_input = PwmaInput::from_slice(&first_result.values, second_params);
        let second_result = pwma(&second_input).unwrap();
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 240..second_result.values.len() {
            assert!(second_result.values[i].is_finite());
        }
    }

    #[test]
    fn test_pwma_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).unwrap();
        let params = PwmaParams { period: Some(5) };
        let input = PwmaInput::from_candles(&candles, "close", params);
        let pwma_result = pwma(&input).unwrap();
        for &val in &pwma_result.values {
            if !val.is_nan() {
                assert!(val.is_finite());
            }
        }
    }
}
