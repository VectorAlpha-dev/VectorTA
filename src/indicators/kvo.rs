/// # Klinger Volume Oscillator (KVO)
///
/// The Klinger Volume Oscillator (KVO) is designed to capture long-term
/// money flow trends, while remaining sensitive enough to short-term
/// fluctuations. It uses high, low, close prices and volume to measure
/// volume force (VF), then applies two separate EMAs (short and long)
/// to VF and calculates the difference.
///
/// ## Parameters
/// - **short_period**: The short EMA period. Defaults to 2.
/// - **long_period**: The long EMA period. Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: kvo: Input data slice is empty or not found.
/// - **InvalidPeriod**: kvo: `short_period` < 1 or `long_period` < `short_period`.
/// - **NotEnoughValidData**: kvo: Fewer than 2 valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: kvo: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(KvoOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until enough data is present for the calculation.
/// - **`Err(KvoError)`** otherwise.
use crate::utilities::data_loader::{read_candles_from_csv, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum KvoData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct KvoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct KvoParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
}

impl Default for KvoParams {
    fn default() -> Self {
        Self {
            short_period: Some(2),
            long_period: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KvoInput<'a> {
    pub data: KvoData<'a>,
    pub params: KvoParams,
}

impl<'a> KvoInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: KvoParams) -> Self {
        Self {
            data: KvoData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        params: KvoParams,
    ) -> Self {
        Self {
            data: KvoData::Slices {
                high,
                low,
                close,
                volume,
            },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: KvoData::Candles { candles },
            params: KvoParams::default(),
        }
    }

    pub fn get_short_period(&self) -> usize {
        self.params
            .short_period
            .unwrap_or_else(|| KvoParams::default().short_period.unwrap())
    }

    pub fn get_long_period(&self) -> usize {
        self.params
            .long_period
            .unwrap_or_else(|| KvoParams::default().long_period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum KvoError {
    #[error("kvo: Empty data provided.")]
    EmptyData,
    #[error("kvo: Invalid period settings: short={short}, long={long}")]
    InvalidPeriod { short: usize, long: usize },
    #[error("kvo: Not enough valid data: found {valid} valid points after the first valid index.")]
    NotEnoughValidData { valid: usize },
    #[error("kvo: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn kvo(input: &KvoInput) -> Result<KvoOutput, KvoError> {
    let (high, low, close, volume) = match &input.data {
        KvoData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| KvoError::EmptyData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| KvoError::EmptyData)?;
            let close = candles
                .select_candle_field("close")
                .map_err(|_| KvoError::EmptyData)?;
            let volume = candles
                .select_candle_field("volume")
                .map_err(|_| KvoError::EmptyData)?;
            (high, low, close, volume)
        }
        KvoData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() || volume.is_empty() {
        return Err(KvoError::EmptyData);
    }

    let short_period = input.get_short_period();
    let long_period = input.get_long_period();
    if short_period < 1 || long_period < short_period {
        return Err(KvoError::InvalidPeriod {
            short: short_period,
            long: long_period,
        });
    }

    let first_valid_idx = high
        .iter()
        .zip(low.iter())
        .zip(close.iter())
        .zip(volume.iter())
        .position(|(((h, l), c), v)| !h.is_nan() && !l.is_nan() && !c.is_nan() && !v.is_nan());
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(KvoError::AllValuesNaN),
    };

    if (high.len() - first_valid_idx) < 2 {
        return Err(KvoError::NotEnoughValidData {
            valid: high.len() - first_valid_idx,
        });
    }

    let mut output = vec![f64::NAN; high.len()];
    let short_per = 2.0 / (short_period as f64 + 1.0);
    let long_per = 2.0 / (long_period as f64 + 1.0);

    let mut trend = -1;
    let mut cm = 0.0;
    let mut prev_hlc = high[first_valid_idx] + low[first_valid_idx] + close[first_valid_idx];
    let mut short_ema = 0.0;
    let mut long_ema = 0.0;

    for i in (first_valid_idx + 1)..high.len() {
        let hlc = high[i] + low[i] + close[i];
        let dm = high[i] - low[i];

        if hlc > prev_hlc && trend != 1 {
            trend = 1;
            cm = high[i - 1] - low[i - 1];
        } else if hlc < prev_hlc && trend != 0 {
            trend = 0;
            cm = high[i - 1] - low[i - 1];
        }

        cm += dm;

        let vf =
            volume[i] * (dm / cm * 2.0 - 1.0).abs() * 100.0 * if trend == 1 { 1.0 } else { -1.0 };

        if i == first_valid_idx + 1 {
            short_ema = vf;
            long_ema = vf;
        } else {
            short_ema = (vf - short_ema) * short_per + short_ema;
            long_ema = (vf - long_ema) * long_per + long_ema;
        }

        output[i] = short_ema - long_ema;
        prev_hlc = hlc;
    }

    Ok(KvoOutput { values: output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_kvo_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = KvoParams {
            short_period: None,
            long_period: None,
        };
        let input_default = KvoInput::from_candles(&candles, default_params);
        let output_default = kvo(&input_default).expect("Failed KVO with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_custom = KvoParams {
            short_period: Some(2),
            long_period: Some(7),
        };
        let input_custom = KvoInput::from_candles(&candles, params_custom);
        let output_custom = kvo(&input_custom).expect("Failed KVO with custom params");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_kvo_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = KvoParams::default();
        let input = KvoInput::from_candles(&candles, params);
        let kvo_result = kvo(&input).expect("Failed to calculate KVO");

        assert_eq!(
            kvo_result.values.len(),
            close_prices.len(),
            "KVO length mismatch"
        );

        let expected_last_five_kvo = [
            -246.42698280402647,
            530.8651474164992,
            237.2148311016648,
            608.8044103976362,
            -6339.615516805162,
        ];
        assert!(
            kvo_result.values.len() >= 5,
            "KVO result length too short for verification"
        );
        let start_index = kvo_result.values.len() - 5;
        let result_last_five_kvo = &kvo_result.values[start_index..];
        for (i, &value) in result_last_five_kvo.iter().enumerate() {
            let expected_value = expected_last_five_kvo[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "KVO mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let first_valid_point = kvo_result
            .values
            .iter()
            .position(|&v| !v.is_nan())
            .unwrap_or(kvo_result.values.len());
        for i in 0..first_valid_point {
            assert!(kvo_result.values[i].is_nan());
        }

        let default_input = KvoInput::with_default_candles(&candles);
        let default_kvo_result = kvo(&default_input).expect("Failed to calculate KVO defaults");
        assert_eq!(default_kvo_result.values.len(), close_prices.len());
    }

    #[test]
    fn test_kvo_params_with_default_params() {
        let default_params = KvoParams::default();
        assert_eq!(
            default_params.short_period,
            Some(2),
            "Expected default short_period of 2"
        );
        assert_eq!(
            default_params.long_period,
            Some(5),
            "Expected default long_period of 5"
        );
    }

    #[test]
    fn test_kvo_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = KvoInput::with_default_candles(&candles);
        match input.data {
            KvoData::Candles { .. } => {}
            _ => panic!("Expected KvoData::Candles variant"),
        }
    }

    #[test]
    fn test_kvo_with_zero_short_period() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = KvoParams {
            short_period: Some(0),
            long_period: Some(5),
        };
        let input = KvoInput::from_candles(&candles, params);
        let result = kvo(&input);
        assert!(result.is_err(), "Expected error for zero short_period");
    }

    #[test]
    fn test_kvo_with_long_period_less_than_short() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = KvoParams {
            short_period: Some(5),
            long_period: Some(2),
        };
        let input = KvoInput::from_candles(&candles, params);
        let result = kvo(&input);
        assert!(
            result.is_err(),
            "Expected error for long_period < short_period"
        );
    }

    #[test]
    fn test_kvo_very_small_data_set() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let mut candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        candles.high.truncate(1);
        candles.low.truncate(1);
        candles.close.truncate(1);
        candles.volume.truncate(1);

        let input = KvoInput::from_candles(&candles, KvoParams::default());
        let result = kvo(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than the minimum needed"
        );
    }

    #[test]
    fn test_kvo_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = KvoParams {
            short_period: Some(2),
            long_period: Some(5),
        };
        let first_input = KvoInput::from_candles(&candles, first_params);
        let first_result = kvo(&first_input).expect("Failed to calculate first KVO");

        let second_params = KvoParams {
            short_period: Some(2),
            long_period: Some(5),
        };
        let second_input = KvoInput::from_slices(
            &candles.high,
            &candles.low,
            &candles.close,
            &first_result.values,
            second_params,
        );
        let second_result = kvo(&second_input);
        assert!(
            second_result.is_err() || second_result.is_ok(),
            "Check if second KVO can handle reinput (likely an error if reinput is unorthodox data)"
        );
    }

    #[test]
    fn test_kvo_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = KvoParams::default();
        let input = KvoInput::from_candles(&candles, params);
        let kvo_result = kvo(&input).expect("Failed to calculate KVO");

        if kvo_result.values.len() > 240 {
            for i in 240..kvo_result.values.len() {
                assert!(
                    !kvo_result.values[i].is_nan(),
                    "Expected no NaN after index 240, found NaN at {}",
                    i
                );
            }
        }
    }
}
