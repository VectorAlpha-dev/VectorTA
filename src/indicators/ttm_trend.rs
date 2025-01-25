/// # TTM Trend
///
/// Compares the current candle's close price to a rolling average of a chosen price source
/// (e.g., "close", "hl2", "hlc3", etc.) over a specified `period`. If `close > average`, the
/// trend is considered `true`; otherwise, it is `false`.
///
/// ## Parameters
/// - **period**: The lookback window size for computing the average. Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: ttm_trend: Input data slice is empty.
/// - **InvalidPeriod**: ttm_trend: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: ttm_trend: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index (considering both close and source).
/// - **AllValuesNaN**: ttm_trend: No valid (non-`NaN`) pair of values found in `close` and `source`.
///
/// ## Returns
/// - **`Ok(TtmTrendOutput)`** on success, containing a `Vec<bool>` matching the input length,
///   with initial values set to `false` until enough data is accumulated.
/// - **`Err(TtmTrendError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum TtmTrendData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slices {
        source: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct TtmTrendOutput {
    pub values: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct TtmTrendParams {
    pub period: Option<usize>,
}

impl Default for TtmTrendParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct TtmTrendInput<'a> {
    pub data: TtmTrendData<'a>,
    pub params: TtmTrendParams,
}

impl<'a> TtmTrendInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: TtmTrendParams) -> Self {
        Self {
            data: TtmTrendData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slices(source: &'a [f64], close: &'a [f64], params: TtmTrendParams) -> Self {
        Self {
            data: TtmTrendData::Slices { source, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: TtmTrendData::Candles {
                candles,
                source: "hl2",
            },
            params: TtmTrendParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| TtmTrendParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum TtmTrendError {
    #[error("ttm_trend: Empty data provided.")]
    EmptyData,
    #[error("ttm_trend: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("ttm_trend: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("ttm_trend: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn ttm_trend(input: &TtmTrendInput) -> Result<TtmTrendOutput, TtmTrendError> {
    let (source, close) = match &input.data {
        TtmTrendData::Candles { candles, source } => {
            let s = source_type(candles, source);
            let c = source_type(candles, "close");
            (s, c)
        }
        TtmTrendData::Slices { source, close } => (*source, *close),
    };

    if source.is_empty() || close.is_empty() {
        return Err(TtmTrendError::EmptyData);
    }
    let data_len = source.len().min(close.len());
    let period = input.get_period();

    if period == 0 || period > data_len {
        return Err(TtmTrendError::InvalidPeriod { period, data_len });
    }

    let first_valid_idx = match source
        .iter()
        .zip(close.iter())
        .position(|(&src, &cl)| !src.is_nan() && !cl.is_nan())
    {
        Some(idx) => idx,
        None => return Err(TtmTrendError::AllValuesNaN),
    };

    if (data_len - first_valid_idx) < period {
        return Err(TtmTrendError::NotEnoughValidData {
            needed: period,
            valid: data_len - first_valid_idx,
        });
    }

    let mut trend_values = vec![false; data_len];
    let mut sum = 0.0;
    for &val in &source[first_valid_idx..(first_valid_idx + period)] {
        sum += val;
    }
    let inv_period = 1.0 / (period as f64);
    let mut idx = first_valid_idx + period - 1;
    if close[idx] > sum * inv_period {
        trend_values[idx] = true;
    }
    idx += 1;

    while idx < data_len {
        sum += source[idx] - source[idx - period];
        trend_values[idx] = close[idx] > sum * inv_period;
        idx += 1;
    }

    Ok(TtmTrendOutput {
        values: trend_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_ttm_trend_default_params() {
        let default_params = TtmTrendParams::default();
        assert_eq!(default_params.period, Some(5));
    }

    #[test]
    fn test_ttm_trend_with_zero_period() {
        let src_data = [10.0, 20.0, 30.0];
        let close_data = [12.0, 22.0, 32.0];
        let params = TtmTrendParams { period: Some(0) };
        let input = TtmTrendInput::from_slices(&src_data, &close_data, params);
        let result = ttm_trend(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_ttm_trend_with_period_exceeding_data_length() {
        let src_data = [1.0, 2.0, 3.0];
        let close_data = [1.0, 2.0, 3.0];
        let params = TtmTrendParams { period: Some(10) };
        let input = TtmTrendInput::from_slices(&src_data, &close_data, params);
        let result = ttm_trend(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_ttm_trend_not_enough_valid_data() {
        let src_data = [f64::NAN, f64::NAN, 3.0, 4.0, 5.0];
        let close_data = [f64::NAN, f64::NAN, 3.0, 4.0, 5.0];
        let params = TtmTrendParams { period: Some(5) };
        let input = TtmTrendInput::from_slices(&src_data, &close_data, params);
        let result = ttm_trend(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_ttm_trend_all_values_nan() {
        let src_data = [f64::NAN, f64::NAN, f64::NAN];
        let close_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = TtmTrendParams { period: Some(5) };
        let input = TtmTrendInput::from_slices(&src_data, &close_data, params);
        let result = ttm_trend(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_ttm_trend_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = TtmTrendParams { period: Some(5) };
        let input = TtmTrendInput::from_candles(&candles, "hl2", params);
        let trend_result = ttm_trend(&input).expect("Failed TTM Trend calculation");

        assert_eq!(
            trend_result.values.len(),
            close_prices.len(),
            "TTM Trend length mismatch"
        );

        let expected_last_five = [true, false, false, false, false];
        assert!(trend_result.values.len() >= 5, "TTM Trend length too short");
        let start_index = trend_result.values.len() - 5;
        let result_last_five = &trend_result.values[start_index..];
        for (i, &val) in result_last_five.iter().enumerate() {
            assert_eq!(
                val, expected_last_five[i],
                "TTM Trend mismatch at index {}: expected {}, got {}",
                i, expected_last_five[i], val
            );
        }
    }
}
