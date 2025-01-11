use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
/// # Keltner Channels
///
/// A volatility-based envelope indicator. The middle band is typically a moving average (MA) of a
/// user-specified `source`, and the upper and lower bands are derived by adding or subtracting a
/// multiple of an internally computed Average True Range (ATR).
///
/// ## Parameters
/// - **period**: The lookback length for both the moving average and the ATR. Defaults to 20.
/// - **multiplier**: The ATR multiplier for constructing upper/lower bands. Defaults to 2.0.
/// - **ma_type**: The moving average type to use (e.g., `"ema"`, `"sma"`, `"wma"`, etc.). Defaults to `"ema"`.
///
/// ## Errors
/// - **KeltnerEmptyData**: keltner: Input data is empty.
/// - **KeltnerInvalidPeriod**: keltner: `period` is zero or exceeds data length.
/// - **KeltnerNotEnoughValidData**: keltner: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **KeltnerAllValuesNaN**: keltner: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(KeltnerOutput)`** on success, containing three `Vec<f64>` matching the input length:
///   (`upper_band`, `middle_band`, `lower_band`) with leading `NaN`s until the first valid index.
/// - **`Err(KeltnerError)`** otherwise.
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum KeltnerData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
}

#[derive(Debug, Clone)]
pub struct KeltnerParams {
    pub period: Option<usize>,
    pub multiplier: Option<f64>,
    pub ma_type: Option<String>,
}

impl Default for KeltnerParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            multiplier: Some(2.0),
            ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KeltnerInput<'a> {
    pub data: KeltnerData<'a>,
    pub params: KeltnerParams,
}

impl<'a> KeltnerInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: KeltnerParams) -> Self {
        Self {
            data: KeltnerData::Candles { candles, source },
            params,
        }
    }
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, "close", KeltnerParams::default())
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| KeltnerParams::default().period.unwrap())
    }

    pub fn get_multiplier(&self) -> f64 {
        self.params
            .multiplier
            .unwrap_or_else(|| KeltnerParams::default().multiplier.unwrap())
    }

    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .as_ref()
            .map(|s| s.to_lowercase())
            .unwrap_or_else(|| "sma".to_string())
    }
}

#[derive(Debug, Clone)]
pub struct KeltnerOutput {
    pub upper_band: Vec<f64>,
    pub middle_band: Vec<f64>,
    pub lower_band: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum KeltnerError {
    #[error("keltner: empty data provided.")]
    KeltnerEmptyData,
    #[error("keltner: invalid period: period = {period}, data length = {data_len}")]
    KeltnerInvalidPeriod { period: usize, data_len: usize },
    #[error("keltner: not enough valid data: needed = {needed}, valid = {valid}")]
    KeltnerNotEnoughValidData { needed: usize, valid: usize },
    #[error("keltner: all values are NaN.")]
    KeltnerAllValuesNaN,
    #[error("keltner: MA error: {0}")]
    KeltnerMaError(#[from] Box<dyn Error>),
}

#[inline]
pub fn keltner(input: &KeltnerInput) -> Result<KeltnerOutput, KeltnerError> {
    let period = input.get_period();
    if period == 0 {
        return Err(KeltnerError::KeltnerInvalidPeriod {
            period,
            data_len: 0,
        });
    }

    let (high, low, close, source_slice) = match &input.data {
        KeltnerData::Candles { candles, source } => {
            let high = candles.select_candle_field("high")?;
            let low = candles.select_candle_field("low")?;
            let close = candles.select_candle_field("close")?;
            let source_slice = source_type(candles, source);
            (high, low, close, source_slice)
        }
    };

    let len = close.len();
    if len == 0 {
        return Err(KeltnerError::KeltnerEmptyData);
    }
    if period > len {
        return Err(KeltnerError::KeltnerInvalidPeriod {
            period,
            data_len: len,
        });
    }

    let mut atr_values = vec![f64::NAN; len];
    let alpha = 1.0 / (period as f64);
    let mut sum_tr = 0.0;
    let mut rma = f64::NAN;

    for i in 0..len {
        let tr = if i == 0 {
            high[0] - low[0]
        } else {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            hl.max(hc).max(lc)
        };

        if i < period {
            sum_tr += tr;
            if i == period - 1 {
                rma = sum_tr / (period as f64);
                atr_values[i] = rma;
            }
        } else {
            rma += alpha * (tr - rma);
            atr_values[i] = rma;
        }
    }

    let ma_values = ma(&input.get_ma_type(), MaData::Slice(&source_slice), period)
        .map_err(KeltnerError::KeltnerMaError)?;

    if ma_values.len() != len {
        return Err(KeltnerError::KeltnerInvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first_valid_ma = ma_values.iter().position(|&v| !v.is_nan());
    let first_valid_atr = atr_values.iter().position(|&v| !v.is_nan());
    let first_valid_idx = match (first_valid_ma, first_valid_atr) {
        (Some(m), Some(a)) => m.max(a),
        _ => return Err(KeltnerError::KeltnerAllValuesNaN),
    };

    if (len - first_valid_idx) < period {
        return Err(KeltnerError::KeltnerNotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }

    let multiplier = input.get_multiplier();
    let mut upper_band = vec![f64::NAN; len];
    let mut middle_band = vec![f64::NAN; len];
    let mut lower_band = vec![f64::NAN; len];

    for i in first_valid_idx..len {
        let ma_v = ma_values[i];
        let atr_v = atr_values[i];
        if ma_v.is_nan() || atr_v.is_nan() {
            continue;
        }
        middle_band[i] = ma_v;
        upper_band[i] = ma_v + multiplier * atr_v;
        lower_band[i] = ma_v - multiplier * atr_v;
    }

    Ok(KeltnerOutput {
        upper_band,
        middle_band,
        lower_band,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_keltner_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = KeltnerParams {
            period: Some(20),
            multiplier: Some(2.0),
            ma_type: Some("ema".to_string()),
        };
        let input = KeltnerInput::from_candles(&candles, "close", params);
        let result = keltner(&input).expect("Failed to calculate Keltner channels");

        assert_eq!(result.upper_band.len(), candles.close.len());
        assert_eq!(result.middle_band.len(), candles.close.len());
        assert_eq!(result.lower_band.len(), candles.close.len());

        let last_five_index = candles.close.len().saturating_sub(5);
        let expected_upper = [
            61619.504155205745,
            61503.56119134791,
            61387.47897150178,
            61286.61078267451,
            61206.25688331261,
        ];
        let expected_middle = [
            59758.339871629956,
            59703.35512195091,
            59640.083205574636,
            59593.884805043715,
            59504.46720456336,
        ];
        let expected_lower = [
            57897.17558805417,
            57903.14905255391,
            57892.68743964749,
            57901.158827412924,
            57802.67752581411,
        ];

        let last_five_upper = &result.upper_band[last_five_index..];
        let last_five_middle = &result.middle_band[last_five_index..];
        let last_five_lower = &result.lower_band[last_five_index..];

        for i in 0..5 {
            let diff_u = (last_five_upper[i] - expected_upper[i]).abs();
            let diff_m = (last_five_middle[i] - expected_middle[i]).abs();
            let diff_l = (last_five_lower[i] - expected_lower[i]).abs();

            assert!(
                diff_u < 1e-1,
                "Upper band mismatch at index {}: expected {}, got {}",
                i,
                expected_upper[i],
                last_five_upper[i]
            );
            assert!(
                diff_m < 1e-1,
                "Middle band mismatch at index {}: expected {}, got {}",
                i,
                expected_middle[i],
                last_five_middle[i]
            );
            assert!(
                diff_l < 1e-1,
                "Lower band mismatch at index {}: expected {}, got {}",
                i,
                expected_lower[i],
                last_five_lower[i]
            );
        }
    }

    #[test]
    fn test_keltner_with_default_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = KeltnerParams::default();
        let input = KeltnerInput::from_candles(&candles, "close", default_params);
        let result = keltner(&input).expect("Failed to calculate Keltner default params");

        assert_eq!(result.upper_band.len(), candles.close.len());
        assert_eq!(result.middle_band.len(), candles.close.len());
        assert_eq!(result.lower_band.len(), candles.close.len());
    }

    #[test]
    fn test_keltner_zero_period() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = KeltnerParams {
            period: Some(0),
            multiplier: Some(2.0),
            ma_type: Some("ema".to_string()),
        };
        let input = KeltnerInput::from_candles(&candles, "close", params);
        let result = keltner(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("invalid period"),
                "Expected invalid period error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_keltner_large_period() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = KeltnerParams {
            period: Some(999999),
            multiplier: Some(2.0),
            ma_type: Some("ema".to_string()),
        };
        let input = KeltnerInput::from_candles(&candles, "close", params);
        let result = keltner(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("invalid period"),
                "Expected invalid period error, got: {}",
                e
            );
        }
    }
}
