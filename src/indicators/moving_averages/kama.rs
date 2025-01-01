use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum KamaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct KamaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct KamaParams {
    pub period: Option<usize>,
}

impl Default for KamaParams {
    fn default() -> Self {
        KamaParams { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct KamaInput<'a> {
    pub data: KamaData<'a>,
    pub params: KamaParams,
}

impl<'a> KamaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: KamaParams) -> Self {
        Self {
            data: KamaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: KamaParams) -> Self {
        Self {
            data: KamaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: KamaData::Candles {
                candles,
                source: "close",
            },
            params: KamaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params.period.unwrap_or(30)
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum KamaError {
    #[error("No data provided for KAMA.")]
    NoData,

    #[error("All data is NaN.")]
    AllValuesNaN,

    #[error("Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("Not enough data to compute KAMA: needed = {needed}, valid = {valid}")]
    NotEnoughData { needed: usize, valid: usize },
}

#[inline]
pub fn kama(input: &KamaInput) -> Result<KamaOutput, KamaError> {
    let data: &[f64] = match &input.data {
        KamaData::Candles { candles, source } => source_type(candles, source),
        KamaData::Slice(slice) => slice,
    };

    let len: usize = data.len();
    if len == 0 {
        return Err(KamaError::NoData);
    }

    let period: usize = input.get_period();
    if period == 0 || period > len {
        return Err(KamaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let lookback = period.saturating_sub(1);
    if lookback >= len {
        return Err(KamaError::NotEnoughData {
            needed: lookback + 1,
            valid: len,
        });
    }
    let mut values = vec![f64::NAN; len];
    let const_max = 2.0 / (30.0 + 1.0);
    let const_diff = (2.0 / (2.0 + 1.0)) - const_max;
    let start_idx = lookback;
    let mut sum_roc1 = 0.0;
    let mut today = start_idx - lookback;
    let mut i = period;
    while i > 0 {
        i -= 1;
        let temp = data[today + 1] - data[today];
        sum_roc1 += temp.abs();
        today += 1;
    }
    let mut prev_kama = data[today];
    values[today] = prev_kama;
    let mut out_idx = 1;
    let mut trailing_idx = start_idx - lookback;
    let mut trailing_value = data[trailing_idx];
    today += 1;
    while today <= start_idx {
        let price = data[today];
        let temp_real = (price - data[trailing_idx]).abs();
        sum_roc1 -= (data[trailing_idx + 1] - trailing_value).abs();
        sum_roc1 += (price - data[today - 1]).abs();
        trailing_value = data[trailing_idx + 1];
        trailing_idx += 1;
        let direction = temp_real;
        let er = if sum_roc1 == 0.0 {
            0.0
        } else {
            direction / sum_roc1
        };
        let sc = (er * const_diff + const_max) * (er * const_diff + const_max);
        prev_kama += (price - prev_kama) * sc;
        today += 1;
    }
    if today <= len {
        values[0] = f64::NAN;
        for i in 1..out_idx {
            values[i] = f64::NAN;
        }
    }
    values[0] = f64::NAN;
    let output_beg = today - 1;
    values[output_beg] = prev_kama;
    out_idx = 1;
    while today < len {
        let price = data[today];
        sum_roc1 -= (data[trailing_idx + 1] - trailing_value).abs();
        sum_roc1 += (price - data[today - 1]).abs();
        trailing_value = data[trailing_idx + 1];
        trailing_idx += 1;
        let direction = (price - data[trailing_idx]).abs();
        let er = if sum_roc1 == 0.0 {
            0.0
        } else {
            direction / sum_roc1
        };
        let sc = (er * const_diff + const_max) * (er * const_diff + const_max);
        prev_kama += (price - prev_kama) * sc;
        values[output_beg + out_idx] = prev_kama;
        out_idx += 1;
        today += 1;
    }
    Ok(KamaOutput { values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_kama_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = KamaInput::with_default_candles(&candles);

        let result = kama(&input).expect("Failed to calculate KAMA");

        let expected_last_five = [
            60234.925553804125,
            60176.838757545665,
            60115.177367962766,
            60071.37070833558,
            59992.79386218023,
        ];

        assert!(
            result.values.len() >= 5,
            "Expected at least 5 values to compare"
        );
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "KAMA output length does not match input length"
        );

        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];

        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-6,
                "KAMA mismatch at last-five index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
    #[test]
    fn test_kama_with_default_params() {
        let default_params = KamaParams::default();
        assert_eq!(default_params.period, Some(30));
    }

    #[test]
    fn test_kama_with_no_data() {
        let data: [f64; 0] = [];
        let input = KamaInput::from_slice(&data, KamaParams { period: Some(30) });
        let result = kama(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_kama_very_small_data_set() {
        let data = [42.0];
        let input = KamaInput::from_slice(&data, KamaParams { period: Some(30) });
        let result = kama(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_kama_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_input =
            KamaInput::from_candles(&candles, "close", KamaParams { period: Some(30) });
        let first_result = kama(&first_input).expect("First KAMA failed");
        let second_input =
            KamaInput::from_slice(&first_result.values, KamaParams { period: Some(10) });
        let second_result = kama(&second_input).expect("Second KAMA failed");
        assert_eq!(second_result.values.len(), first_result.values.len());
        for val in second_result.values.iter().skip(240) {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_kama_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = KamaInput::from_candles(&candles, "close", KamaParams { period: None });
        let result = kama(&input).expect("KAMA calculation failed with partial params");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_kama_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = KamaInput::from_candles(&candles, "close", KamaParams { period: Some(30) });
        let result = kama(&input).expect("KAMA calculation failed");
        assert_eq!(result.values.len(), candles.close.len());
        for val in result.values.iter().skip(30) {
            assert!(val.is_finite());
        }
    }
}
