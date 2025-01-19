use crate::indicators::moving_averages::ma::{ma, MaData};
/// # Percentage Price Oscillator (PPO)
///
/// The PPO is similar to MACD, but expresses the difference between two moving averages
/// as a percentage of the slower moving average.
///
/// ## Parameters
/// - **fast_period**: The short-term moving average period. Defaults to 12.
/// - **slow_period**: The long-term moving average period. Defaults to 26.
/// - **ma_type**: The type of moving average to use (e.g., "sma", "ema", etc.). Defaults to "sma".
/// - **source**: The candle source to use (e.g. "close"). Defaults to "close".
///
/// ## Errors
/// - **EmptyData**: ppo: Input data slice is empty.
/// - **InvalidPeriod**: ppo: `fast_period` or `slow_period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: ppo: Fewer than `slow_period` valid (non-`NaN`) data points remain after the first valid index.
/// - **AllValuesNaN**: ppo: All input data values are `NaN`.
/// - **MaError**: ppo: Error returned from the internal moving average function.
///
/// ## Returns
/// - **`Ok(PpoOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the slower period window is filled.
/// - **`Err(PpoError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum PpoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PpoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PpoParams {
    pub fast_period: Option<usize>,
    pub slow_period: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for PpoParams {
    fn default() -> Self {
        Self {
            fast_period: Some(12),
            slow_period: Some(26),
            ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PpoInput<'a> {
    pub data: PpoData<'a>,
    pub params: PpoParams,
}

impl<'a> PpoInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: PpoParams) -> Self {
        Self {
            data: PpoData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: PpoParams) -> Self {
        Self {
            data: PpoData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: PpoData::Candles {
                candles,
                source: "close",
            },
            params: PpoParams::default(),
        }
    }

    pub fn get_fast_period(&self) -> usize {
        self.params
            .fast_period
            .unwrap_or_else(|| PpoParams::default().fast_period.unwrap())
    }

    pub fn get_slow_period(&self) -> usize {
        self.params
            .slow_period
            .unwrap_or_else(|| PpoParams::default().slow_period.unwrap())
    }

    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .clone()
            .unwrap_or_else(|| PpoParams::default().ma_type.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum PpoError {
    #[error("ppo: Empty data provided.")]
    EmptyData,
    #[error("ppo: Invalid period: fast = {fast}, slow = {slow}, data length = {data_len}")]
    InvalidPeriod {
        fast: usize,
        slow: usize,
        data_len: usize,
    },
    #[error("ppo: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ppo: All values are NaN.")]
    AllValuesNaN,
    #[error("ppo: MA error: {0}")]
    MaError(String),
}

pub fn ppo(input: &PpoInput) -> Result<PpoOutput, PpoError> {
    let data: &[f64] = match &input.data {
        PpoData::Candles { candles, source } => source_type(candles, source),
        PpoData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(PpoError::EmptyData);
    }

    let fast_period = input.get_fast_period();
    let slow_period = input.get_slow_period();
    if fast_period == 0 || slow_period == 0 || fast_period > data.len() || slow_period > data.len()
    {
        return Err(PpoError::InvalidPeriod {
            fast: fast_period,
            slow: slow_period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(PpoError::AllValuesNaN),
    };

    let needed = slow_period;
    let valid = data.len() - first_valid_idx;
    if valid < needed {
        return Err(PpoError::NotEnoughValidData { needed, valid });
    }

    let mut ppo_values = vec![f64::NAN; data.len()];

    let ma_type = input.get_ma_type();
    let fast_ma = ma(&ma_type, MaData::Slice(&data), fast_period)
        .map_err(|e| PpoError::MaError(e.to_string()))?;
    let slow_ma = ma(&ma_type, MaData::Slice(&data), slow_period)
        .map_err(|e| PpoError::MaError(e.to_string()))?;

    for i in first_valid_idx..data.len() {
        let sf = slow_ma[i];
        let ff = fast_ma[i];
        if sf.is_nan() || ff.is_nan() || sf == 0.0 {
            ppo_values[i] = f64::NAN;
        } else {
            ppo_values[i] = 100.0 * (ff - sf) / sf;
        }
    }

    Ok(PpoOutput { values: ppo_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ppo_with_defaults() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = PpoInput::with_default_candles(&candles);
        let output = ppo(&input).expect("Failed to calculate PPO with default params");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_ppo_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = PpoParams {
            fast_period: Some(12),
            slow_period: Some(26),
            ma_type: Some("sma".to_string()),
        };
        let input = PpoInput::from_candles(&candles, "close", params);
        let result = ppo(&input).expect("Failed to calculate PPO");

        assert_eq!(result.values.len(), candles.close.len());
        let expected_last_five = [
            -0.8532313608928664,
            -0.8537562894550523,
            -0.6821291938174874,
            -0.5620008722078592,
            -0.4101724140910927,
        ];
        assert!(result.values.len() >= 5);
        let start_index = result.values.len().saturating_sub(5);
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let diff = (val - expected_last_five[i]).abs();
            assert!(
                diff < 1e-7,
                "Mismatch at {}: expected {}, got {}",
                i,
                expected_last_five[i],
                val
            );
        }
    }

    #[test]
    fn test_ppo_empty_data() {
        let input_data = [];
        let params = PpoParams::default();
        let input = PpoInput::from_slice(&input_data, params);
        let res = ppo(&input);
        assert!(res.is_err());
        if let Err(e) = res {
            assert!(
                e.to_string().contains("Empty data"),
                "Expected empty data error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_ppo_period_exceeds_data_length() {
        let input_data = [1.0, 2.0, 3.0];
        let params = PpoParams {
            fast_period: Some(12),
            slow_period: Some(26),
            ma_type: Some("sma".to_string()),
        };
        let input = PpoInput::from_slice(&input_data, params);
        let res = ppo(&input);
        assert!(res.is_err());
        if let Err(e) = res {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected invalid period error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_ppo_all_nans() {
        let input_data = [f64::NAN, f64::NAN];
        let params = PpoParams::default();
        let input = PpoInput::from_slice(&input_data, params);
        let res = ppo(&input);
        assert!(res.is_err());
    }

    #[test]
    fn test_ppo_not_enough_valid_data() {
        let input_data = [f64::NAN, 50.0];
        let params = PpoParams {
            fast_period: Some(12),
            slow_period: Some(26),
            ma_type: Some("sma".to_string()),
        };
        let input = PpoInput::from_slice(&input_data, params);
        let res = ppo(&input);
        assert!(res.is_err());
    }
}
