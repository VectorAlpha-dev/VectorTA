use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum ZlemaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ZlemaParams {
    pub period: Option<usize>,
}

impl Default for ZlemaParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct ZlemaInput<'a> {
    pub data: ZlemaData<'a>,
    pub params: ZlemaParams,
}

impl<'a> ZlemaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: ZlemaParams) -> Self {
        Self {
            data: ZlemaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: ZlemaParams) -> Self {
        Self {
            data: ZlemaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: ZlemaData::Candles {
                candles,
                source: "close",
            },
            params: ZlemaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| ZlemaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct ZlemaOutput {
    pub values: Vec<f64>,
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ZlemaError {
    #[error("Empty data provided for ZLEMA.")]
    EmptyData,
    #[error("All values are NaN during ZLEMA calculation.")]
    AllValuesNaN,
    #[error("ZLEMA Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
}

#[inline]
pub fn zlema(input: &ZlemaInput) -> Result<ZlemaOutput, ZlemaError> {
    let data: &[f64] = match &input.data {
        ZlemaData::Candles { candles, source } => source_type(candles, source),
        ZlemaData::Slice(slice) => slice,
    };

    let len = data.len();
    if len == 0 {
        return Err(ZlemaError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(ZlemaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first_valid_idx = data.iter().position(|&x| !x.is_nan());
    if first_valid_idx.is_none() {
        return Err(ZlemaError::AllValuesNaN);
    }

    let lag = (period - 1) / 2;
    let alpha = 2.0 / (period as f64 + 1.0);

    let mut zlema_values = vec![f64::NAN; len];

    let mut last_ema = data[first_valid_idx.unwrap()];
    zlema_values[first_valid_idx.unwrap()] = last_ema;

    for i in (first_valid_idx.unwrap() + 1)..len {
        let val = if i < lag {
            data[i]
        } else {
            2.0 * data[i] - data[i - lag]
        };

        last_ema = alpha * val + (1.0 - alpha) * last_ema;
        zlema_values[i] = last_ema;
    }

    Ok(ZlemaOutput {
        values: zlema_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_zlema_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close");
        let params = ZlemaParams { period: Some(14) };
        let input = ZlemaInput::from_candles(&candles, "close", params);
        let result = zlema(&input).expect("Failed to calculate ZLEMA");
        let expected_last_five = [59015.1, 59165.2, 59168.1, 59147.0, 58978.9];
        assert!(result.values.len() >= 5);
        assert_eq!(result.values.len(), close_prices.len());
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &value) in last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "ZLEMA mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }
        for val in result.values.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_zlema_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = ZlemaParams { period: None };
        let input = ZlemaInput::from_candles(&candles, "close", default_params);
        let output = zlema(&input).expect("Failed ZLEMA with default params");
        assert_eq!(output.values.len(), candles.close.len());
        let params_10 = ZlemaParams { period: Some(10) };
        let input2 = ZlemaInput::from_candles(&candles, "hl2", params_10);
        let output2 = zlema(&input2).expect("Failed ZLEMA with period=10, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = ZlemaParams { period: Some(20) };
        let input3 = ZlemaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = zlema(&input3).expect("Failed ZLEMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }
    #[test]
    fn test_zlema_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = ZlemaInput::with_default_candles(&candles);
        match input.data {
            ZlemaData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected ZlemaData::Candles variant"),
        }
        let period = input.get_period();
        assert_eq!(period, 14);
    }

    #[test]
    fn test_zlema_with_default_params() {
        let default_params = ZlemaParams::default();
        assert_eq!(default_params.period, Some(14));
    }

    #[test]
    fn test_zlema_with_no_data() {
        let data: [f64; 0] = [];
        let params = ZlemaParams { period: Some(14) };
        let input = ZlemaInput::from_slice(&data, params);
        let result = zlema(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_zlema_very_small_data_set() {
        let data = [42.0];
        let params = ZlemaParams { period: Some(14) };
        let input = ZlemaInput::from_slice(&data, params);
        let result = zlema(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_zlema_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_input =
            ZlemaInput::from_candles(&candles, "close", ZlemaParams { period: Some(21) });
        let first_result = zlema(&first_input).expect("Failed ZLEMA on first input");

        let second_input =
            ZlemaInput::from_slice(&first_result.values, ZlemaParams { period: Some(14) });
        let second_result = zlema(&second_input).expect("Failed ZLEMA on second input");
        assert_eq!(second_result.values.len(), first_result.values.len());
        for (idx, &val) in second_result.values.iter().enumerate().skip(240) {
            assert!(val.is_finite(), "NaN found at index {}", idx);
        }
    }

    #[test]
    fn test_zlema_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = ZlemaInput::from_candles(&candles, "close", ZlemaParams { period: Some(14) });
        let result = zlema(&input).expect("Failed ZLEMA calculation");
        assert_eq!(result.values.len(), candles.close.len());
        for (idx, &val) in result.values.iter().enumerate().skip(14) {
            assert!(val.is_finite(), "NaN found at index {}", idx);
        }
    }
}
