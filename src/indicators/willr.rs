/// # Williams' %R (WILLR)
///
/// Williams' %R is a momentum indicator that measures overbought/oversold levels.
/// It compares the current closing price to the recent trading range (highest high
/// and lowest low over a specified period).
///
/// ## Formula
/// \[ \text{%R} = \frac{\text{Highest\_High} - \text{Close}}{\text{Highest\_High} - \text{Lowest\_Low}} \times (-100) \]
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: willr: Input data slice is empty.
/// - **InvalidPeriod**: willr: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: willr: Fewer than `period` valid data points remain
///   after the first valid index (where all high, low, and close are non-`NaN`).
/// - **AllValuesNaN**: willr: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(WillrOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the calculation window is filled.
/// - **`Err(WillrError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum WillrData<'a> {
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
pub struct WillrOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WillrParams {
    pub period: Option<usize>,
}

impl Default for WillrParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct WillrInput<'a> {
    pub data: WillrData<'a>,
    pub params: WillrParams,
}

impl<'a> WillrInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: WillrParams) -> Self {
        Self {
            data: WillrData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: WillrParams,
    ) -> Self {
        Self {
            data: WillrData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: WillrData::Candles { candles },
            params: WillrParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| WillrParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WillrError {
    #[error("willr: Empty data provided for WILLR.")]
    EmptyData,
    #[error("willr: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("willr: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("willr: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn willr(input: &WillrInput) -> Result<WillrOutput, WillrError> {
    let (high, low, close) = match &input.data {
        WillrData::Candles { candles } => {
            let h = candles.select_candle_field("high").unwrap();
            let l = candles.select_candle_field("low").unwrap();
            let c = candles.select_candle_field("close").unwrap();
            (h, l, c)
        }
        WillrData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WillrError::EmptyData);
    }

    let length = high.len();
    if low.len() != length || close.len() != length {
        return Err(WillrError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > length {
        return Err(WillrError::InvalidPeriod {
            period,
            data_len: length,
        });
    }

    let first_valid_idx =
        match (0..length).find(|&i| !(high[i].is_nan() || low[i].is_nan() || close[i].is_nan())) {
            Some(idx) => idx,
            None => return Err(WillrError::AllValuesNaN),
        };

    if (length - first_valid_idx) < period {
        return Err(WillrError::NotEnoughValidData {
            needed: period,
            valid: length - first_valid_idx,
        });
    }

    let mut willr_values = vec![f64::NAN; length];

    for i in (first_valid_idx + period - 1)..length {
        let start = i + 1 - period;
        let mut highest = f64::NEG_INFINITY;
        let mut lowest = f64::INFINITY;
        let mut has_nan = false;

        for j in start..=i {
            if high[j].is_nan() || low[j].is_nan() || close[i].is_nan() {
                has_nan = true;
                break;
            }
            if high[j] > highest {
                highest = high[j];
            }
            if low[j] < lowest {
                lowest = low[j];
            }
        }

        if has_nan || highest.is_infinite() || lowest.is_infinite() {
            willr_values[i] = f64::NAN;
        } else {
            let denom = highest - lowest;
            if denom == 0.0 {
                willr_values[i] = 0.0;
            } else {
                willr_values[i] = (highest - close[i]) / denom * -100.0;
            }
        }
    }

    Ok(WillrOutput {
        values: willr_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_willr_default_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = WillrParams::default();
        let input_default = WillrInput::from_candles(&candles, default_params);
        let output_default = willr(&input_default).expect("Failed WILLR with default params");
        assert_eq!(output_default.values.len(), candles.close.len());
    }

    #[test]
    fn test_willr_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = WillrParams { period: Some(14) };
        let input = WillrInput::from_candles(&candles, params);
        let willr_result = willr(&input).expect("Failed to calculate WILLR");

        assert_eq!(
            willr_result.values.len(),
            candles.close.len(),
            "WILLR length mismatch"
        );

        let expected_last_five = [
            -58.72876391329818,
            -61.77504393673111,
            -65.93438781487991,
            -60.27950310559006,
            -65.00449236298293,
        ];

        assert!(willr_result.values.len() >= 5, "WILLR length too short");
        let start_index = willr_result.values.len() - 5;
        let result_last_five = &willr_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-8,
                "WILLR mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_willr_with_slice_data() {
        let high = [1.0, 2.0, 3.0, 4.0];
        let low = [0.5, 1.5, 2.5, 3.5];
        let close = [0.75, 1.75, 2.75, 3.75];
        let params = WillrParams { period: Some(2) };
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let result = willr(&input).expect("Failed to calculate WILLR on slices");
        assert_eq!(result.values.len(), 4);
    }

    #[test]
    fn test_willr_with_zero_period() {
        let high = [1.0, 2.0];
        let low = [0.8, 1.8];
        let close = [1.0, 2.0];
        let params = WillrParams { period: Some(0) };
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let result = willr(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_willr_with_period_exceeding_data_length() {
        let high = [1.0, 2.0, 3.0];
        let low = [0.5, 1.5, 2.5];
        let close = [1.0, 2.0, 3.0];
        let params = WillrParams { period: Some(10) };
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let result = willr(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_willr_all_nan() {
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let close = [f64::NAN, f64::NAN];
        let params = WillrParams::default();
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let result = willr(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_willr_not_enough_valid_data() {
        let high = [f64::NAN, 2.0];
        let low = [f64::NAN, 1.0];
        let close = [f64::NAN, 1.5];
        let params = WillrParams { period: Some(3) };
        let input = WillrInput::from_slices(&high, &low, &close, params);
        let result = willr(&input);
        assert!(result.is_err());
    }
}
