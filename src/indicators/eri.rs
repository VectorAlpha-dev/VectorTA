use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
/// # Elder Ray Index (ERI)
///
/// The Elder-Ray Index (ERI) measures bullish and bearish pressure using an MA (e.g., EMA) as a baseline.
/// `bull` represents the difference between the candle's high and the selected MA, and `bear` represents
/// the difference between the candle's low and the selected MA.
///
/// ## Parameters
/// - **period**: The window size for the MA. Defaults to 13.
/// - **ma_type**: Type of MA to use (e.g., "ema", "sma", etc.). Defaults to "ema".
///
/// ## Errors
/// - **EmptyData**: eri: Input data slice is empty.
/// - **InvalidPeriod**: eri: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: eri: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: eri: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(EriOutput)`** on success, containing `bull` and `bear` each as a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the moving average window is filled.
/// - **`Err(EriError)`** otherwise.
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum EriData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        source: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct EriOutput {
    pub bull: Vec<f64>,
    pub bear: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EriParams {
    pub period: Option<usize>,
    pub ma_type: Option<String>,
}

impl Default for EriParams {
    fn default() -> Self {
        Self {
            period: Some(13),
            ma_type: Some("ema".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EriInput<'a> {
    pub data: EriData<'a>,
    pub params: EriParams,
}

impl<'a> EriInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: EriParams) -> Self {
        Self {
            data: EriData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        source: &'a [f64],
        params: EriParams,
    ) -> Self {
        Self {
            data: EriData::Slices { high, low, source },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: EriData::Candles {
                candles,
                source: "close",
            },
            params: EriParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| EriParams::default().period.unwrap())
    }

    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .clone()
            .unwrap_or_else(|| EriParams::default().ma_type.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum EriError {
    #[error("eri: Empty data provided.")]
    EmptyData,
    #[error("eri: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("eri: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("eri: All values are NaN.")]
    AllValuesNaN,
    #[error("eri: MA calculation error: {0}")]
    MaCalculationError(String),
}

#[inline]
pub fn eri(input: &EriInput) -> Result<EriOutput, EriError> {
    let (high, low, source_data) = match &input.data {
        EriData::Candles { candles, source } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| EriError::EmptyData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| EriError::EmptyData)?;
            let src = source_type(candles, source);
            (high, low, src)
        }
        EriData::Slices { high, low, source } => (*high, *low, *source),
    };

    if source_data.is_empty() || high.is_empty() || low.is_empty() {
        return Err(EriError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > source_data.len() {
        return Err(EriError::InvalidPeriod {
            period,
            data_len: source_data.len(),
        });
    }

    let mut first_valid_idx = None;
    for i in 0..source_data.len() {
        if !(source_data[i].is_nan() || high[i].is_nan() || low[i].is_nan()) {
            first_valid_idx = Some(i);
            break;
        }
    }
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(EriError::AllValuesNaN),
    };

    if (source_data.len() - first_valid_idx) < period {
        return Err(EriError::NotEnoughValidData {
            needed: period,
            valid: source_data.len() - first_valid_idx,
        });
    }

    let ma_type = input.get_ma_type();
    let full_ma = ma(&ma_type, MaData::Slice(&source_data), period)
        .map_err(|e| EriError::MaCalculationError(e.to_string()))?;

    let mut bull = vec![f64::NAN; source_data.len()];
    let mut bear = vec![f64::NAN; source_data.len()];

    for i in (first_valid_idx + period - 1)..source_data.len() {
        bull[i] = high[i] - full_ma[i];
        bear[i] = low[i] - full_ma[i];
    }

    Ok(EriOutput { bull, bear })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_eri_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = EriParams {
            period: None,
            ma_type: None,
        };
        let input_default = EriInput::from_candles(&candles, "close", default_params);
        let output_default = eri(&input_default).expect("Failed ERI with default params");
        assert_eq!(output_default.bull.len(), candles.close.len());
        assert_eq!(output_default.bear.len(), candles.close.len());

        let params_period_14 = EriParams {
            period: Some(14),
            ma_type: Some("ema".to_string()),
        };
        let input_period_14 = EriInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 =
            eri(&input_period_14).expect("Failed ERI with period=14, source=hl2");
        assert_eq!(output_period_14.bull.len(), candles.close.len());
        assert_eq!(output_period_14.bear.len(), candles.close.len());

        let params_custom = EriParams {
            period: Some(20),
            ma_type: Some("sma".to_string()),
        };
        let input_custom = EriInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = eri(&input_custom).expect("Failed ERI fully custom");
        assert_eq!(output_custom.bull.len(), candles.close.len());
        assert_eq!(output_custom.bear.len(), candles.close.len());
    }

    #[test]
    fn test_eri_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = EriParams {
            period: Some(13),
            ma_type: Some("ema".to_string()),
        };
        let input = EriInput::from_candles(&candles, "close", params);
        let eri_result = eri(&input).expect("Failed to calculate ERI");

        assert_eq!(
            eri_result.bull.len(),
            close_prices.len(),
            "ERI bull length mismatch"
        );
        assert_eq!(
            eri_result.bear.len(),
            close_prices.len(),
            "ERI bear length mismatch"
        );

        let expected_bull_last_five = [
            -103.35343557205488,
            6.839912366813223,
            -42.851503685589705,
            -9.444146016219747,
            11.476446271808527,
        ];
        let expected_bear_last_five = [
            -433.3534355720549,
            -314.1600876331868,
            -414.8515036855897,
            -336.44414601621975,
            -925.5235537281915,
        ];
        assert!(eri_result.bull.len() >= 5, "ERI length too short for bull");
        assert!(eri_result.bear.len() >= 5, "ERI length too short for bear");

        let start_index = eri_result.bull.len() - 5;
        for i in 0..5 {
            let actual_bull = eri_result.bull[start_index + i];
            let actual_bear = eri_result.bear[start_index + i];
            let expected_bull = expected_bull_last_five[i];
            let expected_bear = expected_bear_last_five[i];
            assert!(
                (actual_bull - expected_bull).abs() < 1e-2,
                "ERI bull mismatch at index {}: expected {}, got {}",
                i,
                expected_bull,
                actual_bull
            );
            assert!(
                (actual_bear - expected_bear).abs() < 1e-2,
                "ERI bear mismatch at index {}: expected {}, got {}",
                i,
                expected_bear,
                actual_bear
            );
        }
    }

    #[test]
    fn test_eri_params_with_default_params() {
        let default_params = EriParams::default();
        assert_eq!(
            default_params.period,
            Some(13),
            "Expected default period to be 13"
        );
        assert_eq!(
            default_params.ma_type.as_deref(),
            Some("ema"),
            "Expected default ma_type to be 'ema'"
        );
    }

    #[test]
    fn test_eri_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = EriInput::with_default_candles(&candles);
        match input.data {
            EriData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected EriData::Candles variant"),
        }
    }

    #[test]
    fn test_eri_with_zero_period() {
        let high = [10.0, 20.0, 30.0];
        let low = [8.0, 18.0, 28.0];
        let src = [9.0, 19.0, 29.0];
        let params = EriParams {
            period: Some(0),
            ma_type: Some("ema".to_string()),
        };
        let input = EriInput::from_slices(&high, &low, &src, params);

        let result = eri(&input);
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
    fn test_eri_with_period_exceeding_data_length() {
        let high = [10.0, 20.0, 30.0];
        let low = [8.0, 18.0, 28.0];
        let src = [9.0, 19.0, 29.0];
        let params = EriParams {
            period: Some(10),
            ma_type: Some("ema".to_string()),
        };
        let input = EriInput::from_slices(&high, &low, &src, params);

        let result = eri(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_eri_very_small_data_set() {
        let high = [42.0];
        let low = [40.0];
        let src = [41.0];
        let params = EriParams {
            period: Some(9),
            ma_type: Some("ema".to_string()),
        };
        let input = EriInput::from_slices(&high, &low, &src, params);

        let result = eri(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_eri_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = EriParams {
            period: Some(14),
            ma_type: Some("ema".to_string()),
        };
        let first_input = EriInput::from_candles(&candles, "close", first_params);
        let first_result = eri(&first_input).expect("Failed to calculate first ERI");

        assert_eq!(
            first_result.bull.len(),
            candles.close.len(),
            "First ERI bull output length mismatch"
        );
        assert_eq!(
            first_result.bear.len(),
            candles.close.len(),
            "First ERI bear output length mismatch"
        );

        let second_params = EriParams {
            period: Some(14),
            ma_type: Some("ema".to_string()),
        };
        let second_input = EriInput::from_slices(
            &first_result.bull,
            &first_result.bear,
            &first_result.bull,
            second_params,
        );
        let second_result = eri(&second_input).expect("Failed to calculate second ERI");

        assert_eq!(
            second_result.bull.len(),
            first_result.bull.len(),
            "Second ERI bull output length mismatch"
        );
        assert_eq!(
            second_result.bear.len(),
            first_result.bear.len(),
            "Second ERI bear output length mismatch"
        );

        for i in 28..second_result.bull.len() {
            assert!(
                !second_result.bull[i].is_nan(),
                "Expected no NaN in bull after index 28, but found NaN at index {}",
                i
            );
            assert!(
                !second_result.bear[i].is_nan(),
                "Expected no NaN in bear after index 28, but found NaN at index {}",
                i
            );
        }
    }

    #[test]
    fn test_eri_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 13;
        let params = EriParams {
            period: Some(period),
            ma_type: Some("ema".to_string()),
        };
        let input = EriInput::from_candles(&candles, "close", params);
        let eri_result = eri(&input).expect("Failed to calculate ERI");

        assert_eq!(
            eri_result.bull.len(),
            close_prices.len(),
            "ERI bull length mismatch"
        );
        assert_eq!(
            eri_result.bear.len(),
            close_prices.len(),
            "ERI bear length mismatch"
        );

        if eri_result.bull.len() > 240 {
            for i in 240..eri_result.bull.len() {
                assert!(
                    !eri_result.bull[i].is_nan(),
                    "Expected no NaN after index 240 for bull, but found NaN at index {}",
                    i
                );
            }
        }
        if eri_result.bear.len() > 240 {
            for i in 240..eri_result.bear.len() {
                assert!(
                    !eri_result.bear[i].is_nan(),
                    "Expected no NaN after index 240 for bear, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
