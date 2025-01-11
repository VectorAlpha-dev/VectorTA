use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::indicators::utility_functions::{max_rolling, min_rolling, RollingError};
use crate::utilities::data_loader::read_candles_from_csv;
use crate::utilities::data_loader::Candles;
/// # KDJ (Stochastic Oscillator with MA smoothing)
///
/// KDJ is derived from the Stochastic Oscillator (K, D) with an additional line J,
/// where `J = 3 * K - 2 * D`. This indicator highlights momentum and potential
/// overbought/oversold conditions.
///
/// ## Parameters
/// - **fast_k_period**: The window for the fast stochastic calculation. Defaults to 9.
/// - **slow_k_period**: The smoothing period for K. Defaults to 3.
/// - **slow_k_ma_type**: MA type for smoothing K (0 = SMA, 1 = EMA, etc.). Defaults to 0 (SMA).
/// - **slow_d_period**: The smoothing period for D. Defaults to 3.
/// - **slow_d_ma_type**: MA type for smoothing D (0 = SMA, 1 = EMA, etc.). Defaults to 0 (SMA).
///
/// ## Errors
/// - **EmptyData**: kdj: Input data slice is empty.
/// - **InvalidPeriod**: kdj: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: kdj: Fewer valid (non-`NaN`) data points remain after the first valid index than required.
/// - **AllValuesNaN**: kdj: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(KdjOutput)`** on success, containing three `Vec<f64>` (k, d, j) matching the input length,
///   with leading `NaN`s until the indicator is filled.
/// - **`Err(KdjError)`** otherwise.
use std::collections::VecDeque;
use std::error::Error;
use thiserror::Error;
#[derive(Debug, Clone)]
pub enum KdjData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct KdjParams {
    pub fast_k_period: Option<usize>,
    pub slow_k_period: Option<usize>,
    pub slow_k_ma_type: Option<String>,
    pub slow_d_period: Option<usize>,
    pub slow_d_ma_type: Option<String>,
}

impl Default for KdjParams {
    fn default() -> Self {
        Self {
            fast_k_period: Some(9),
            slow_k_period: Some(3),
            slow_k_ma_type: Some("sma".to_string()),
            slow_d_period: Some(3),
            slow_d_ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KdjInput<'a> {
    pub data: KdjData<'a>,
    pub params: KdjParams,
}

impl<'a> KdjInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: KdjParams) -> Self {
        Self {
            data: KdjData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: KdjParams,
    ) -> Self {
        Self {
            data: KdjData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: KdjData::Candles { candles },
            params: KdjParams::default(),
        }
    }

    pub fn get_fast_k_period(&self) -> usize {
        self.params.fast_k_period.unwrap_or(9)
    }

    pub fn get_slow_k_period(&self) -> usize {
        self.params.slow_k_period.unwrap_or(3)
    }

    pub fn get_slow_k_ma_type(&self) -> String {
        self.params
            .slow_k_ma_type
            .clone()
            .unwrap_or_else(|| "sma".to_string())
    }

    pub fn get_slow_d_period(&self) -> usize {
        self.params.slow_d_period.unwrap_or(3)
    }

    pub fn get_slow_d_ma_type(&self) -> String {
        self.params
            .slow_d_ma_type
            .clone()
            .unwrap_or_else(|| "sma".to_string())
    }
}

#[derive(Debug, Clone)]
pub struct KdjOutput {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
    pub j: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum KdjError {
    #[error("kdj: Empty data provided.")]
    EmptyData,
    #[error("kdj: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("kdj: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("kdj: All values are NaN.")]
    AllValuesNaN,
    #[error("kdj: Rolling error {0}")]
    RollingError(#[from] RollingError),
    #[error("kdj: MA error {0}")]
    MaError(#[from] Box<dyn Error>),
}

#[inline]
pub fn kdj(input: &KdjInput) -> Result<KdjOutput, KdjError> {
    let (high, low, close) = match &input.data {
        KdjData::Candles { candles } => {
            let high = candles.select_candle_field("high")?;
            let low = candles.select_candle_field("low")?;
            let close = candles.select_candle_field("close")?;
            (high, low, close)
        }
        KdjData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(KdjError::EmptyData);
    }

    let fast_k_period = input.get_fast_k_period();
    let slow_k_period = input.get_slow_k_period();
    let slow_k_ma_type_string = input.get_slow_k_ma_type();
    let slow_k_ma_type = slow_k_ma_type_string.as_str();
    let slow_d_period = input.get_slow_d_period();
    let slow_d_ma_type_string = input.get_slow_d_ma_type();
    let slow_d_ma_type = slow_d_ma_type_string.as_str();

    if fast_k_period == 0 || fast_k_period > high.len() {
        return Err(KdjError::InvalidPeriod {
            period: fast_k_period,
            data_len: high.len(),
        });
    }

    let first_valid_idx = match high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .position(|((&h, &l), &c)| !h.is_nan() && !l.is_nan() && !c.is_nan())
    {
        Some(idx) => idx,
        None => return Err(KdjError::AllValuesNaN),
    };

    if (high.len() - first_valid_idx) < fast_k_period {
        return Err(KdjError::NotEnoughValidData {
            needed: fast_k_period,
            valid: high.len() - first_valid_idx,
        });
    }

    let hh = max_rolling(high, fast_k_period)?;
    let ll = min_rolling(low, fast_k_period)?;

    let mut stoch = vec![f64::NAN; high.len()];
    for i in first_valid_idx..high.len() {
        if i < (first_valid_idx + fast_k_period - 1) {
            continue;
        }
        let denom = hh[i] - ll[i];
        if denom == 0.0 || denom.is_nan() {
            stoch[i] = f64::NAN;
        } else {
            stoch[i] = 100.0 * ((close[i] - ll[i]) / denom);
        }
    }

    let k = ma(slow_k_ma_type, MaData::Slice(&stoch), slow_k_period)?;
    let d = ma(slow_d_ma_type, MaData::Slice(&k), slow_d_period)?;

    let mut j = vec![f64::NAN; high.len()];
    for i in 0..high.len() {
        if k[i].is_nan() || d[i].is_nan() {
            j[i] = f64::NAN;
        } else {
            j[i] = 3.0 * k[i] - 2.0 * d[i];
        }
    }

    Ok(KdjOutput { k, d, j })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_kdj_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let partial_params = KdjParams {
            fast_k_period: None,
            slow_k_period: Some(4),
            slow_k_ma_type: None,
            slow_d_period: None,
            slow_d_ma_type: None,
        };
        let input = KdjInput::from_candles(&candles, partial_params);
        let output = kdj(&input).expect("Failed KDJ with partial params");
        assert_eq!(output.k.len(), candles.close.len());
        assert_eq!(output.d.len(), candles.close.len());
        assert_eq!(output.j.len(), candles.close.len());
    }

    #[test]
    fn test_kdj_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = KdjParams {
            fast_k_period: Some(9),
            slow_k_period: Some(3),
            slow_k_ma_type: Some("sma".to_string()),
            slow_d_period: Some(3),
            slow_d_ma_type: Some("sma".to_string()),
        };

        let input = KdjInput::from_candles(&candles, params);
        let result = kdj(&input).expect("Failed to calculate KDJ");

        assert_eq!(result.k.len(), candles.close.len(), "K length mismatch");
        assert_eq!(result.d.len(), candles.close.len(), "D length mismatch");
        assert_eq!(result.j.len(), candles.close.len(), "J length mismatch");

        let expected_k = [
            58.04341315415984,
            61.56034740940419,
            58.056304282719545,
            56.10961365678364,
            51.43992326447119,
        ];
        let expected_d = [
            49.57659409278555,
            56.81719223571944,
            59.22002161542779,
            58.57542178296905,
            55.20194706799139,
        ];
        let expected_j = [
            74.97705127690843,
            71.04665775677368,
            55.72886961730306,
            51.17799740441281,
            43.91587565743079,
        ];

        let len = result.k.len();
        assert!(len >= 5, "Not enough data to test last 5 values");
        let start_idx = len - 5;

        for i in 0..5 {
            let k_val = result.k[start_idx + i];
            let d_val = result.d[start_idx + i];
            let j_val = result.j[start_idx + i];
            let k_diff = (k_val - expected_k[i]).abs();
            let d_diff = (d_val - expected_d[i]).abs();
            let j_diff = (j_val - expected_j[i]).abs();
            assert!(
                k_diff < 1e-4,
                "Mismatch in K at index {}: expected {}, got {}",
                i,
                expected_k[i],
                k_val
            );
            assert!(
                d_diff < 1e-4,
                "Mismatch in D at index {}: expected {}, got {}",
                i,
                expected_d[i],
                d_val
            );
            assert!(
                j_diff < 1e-4,
                "Mismatch in J at index {}: expected {}, got {}",
                i,
                expected_j[i],
                j_val
            );
        }
    }

    #[test]
    fn test_kdj_params_with_default_params() {
        let default_params = KdjParams::default();
        assert_eq!(
            default_params.fast_k_period,
            Some(9),
            "Expected fast_k_period = 9"
        );
        assert_eq!(
            default_params.slow_k_period,
            Some(3),
            "Expected slow_k_period = 3"
        );
        assert_eq!(
            default_params.slow_k_ma_type,
            Some("sma".to_string()),
            "Expected slow_k_ma_type = 0"
        );
        assert_eq!(
            default_params.slow_d_period,
            Some(3),
            "Expected slow_d_period = 3"
        );
        assert_eq!(
            default_params.slow_d_ma_type,
            Some("sma".to_string()),
            "Expected slow_d_ma_type = 0"
        );
    }

    #[test]
    fn test_kdj_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = KdjInput::with_default_candles(&candles);
        match input.data {
            KdjData::Candles { .. } => {}
            _ => panic!("Expected KdjData::Candles variant"),
        }
        let output = kdj(&input).expect("Failed KDJ with default candles");
        assert_eq!(output.k.len(), candles.close.len());
    }

    #[test]
    fn test_kdj_with_zero_fastk_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = KdjParams {
            fast_k_period: Some(0),
            ..Default::default()
        };
        let input = KdjInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = kdj(&input);
        assert!(result.is_err(), "Expected an error for zero fast_k_period");
    }

    #[test]
    fn test_kdj_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = KdjParams {
            fast_k_period: Some(10),
            ..Default::default()
        };
        let input = KdjInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = kdj(&input);
        assert!(
            result.is_err(),
            "Expected an error for fast_k_period > data.len()"
        );
    }

    #[test]
    fn test_kdj_very_small_data_set() {
        let input_data = [42.0];
        let params = KdjParams {
            fast_k_period: Some(9),
            ..Default::default()
        };
        let input = KdjInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = kdj(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_kdj_all_nan_data() {
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = KdjParams::default();
        let input = KdjInput::from_slices(&input_data, &input_data, &input_data, params);
        let result = kdj(&input);
        assert!(result.is_err(), "Expected error for all-NaN data");
    }

    #[test]
    fn test_kdj_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = KdjParams {
            fast_k_period: Some(9),
            slow_k_period: Some(3),
            slow_k_ma_type: Some("sma".to_string()),
            slow_d_period: Some(3),
            slow_d_ma_type: Some("sma".to_string()),
        };
        let first_input = KdjInput::from_candles(&candles, first_params);
        let first_result = kdj(&first_input).expect("Failed to calculate first KDJ");
        assert_eq!(first_result.k.len(), candles.close.len());

        let second_params = KdjParams {
            fast_k_period: Some(9),
            slow_k_period: Some(3),
            slow_k_ma_type: Some("sma".to_string()),
            slow_d_period: Some(3),
            slow_d_ma_type: Some("sma".to_string()),
        };
        let second_input = KdjInput::from_slices(
            &first_result.k,
            &first_result.k,
            &first_result.k,
            second_params,
        );
        let second_result = kdj(&second_input).expect("Failed to calculate second KDJ");
        assert_eq!(second_result.k.len(), first_result.k.len());
        for i in 240..second_result.k.len() {
            assert!(
                !second_result.k[i].is_nan(),
                "Expected no NaN in second KDJ at index {}",
                i
            );
        }
    }

    #[test]
    fn test_kdj_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = KdjParams::default();
        let input = KdjInput::from_candles(&candles, params);
        let result = kdj(&input).expect("Failed to calculate KDJ");
        if result.k.len() > 50 {
            for i in 50..result.k.len() {
                assert!(
                    !result.k[i].is_nan(),
                    "Expected no NaN in K after index 50 at i={}",
                    i
                );
            }
        }
    }
}
