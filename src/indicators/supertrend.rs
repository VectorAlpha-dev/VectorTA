use crate::indicators::atr::{atr, AtrData, AtrError, AtrInput, AtrOutput, AtrParams};
/// # SuperTrend
///
/// A trend-following indicator that uses an ATR-based calculation to determine
/// dynamic support and resistance levels (referred to here as "bands"). SuperTrend
/// alternates between these bands depending on price action, indicating bullish
/// or bearish market conditions. A "changed" flag can signal switches between
/// these conditions.
///
/// ## Parameters
/// - **period**: The lookback window size for the underlying ATR calculation. Defaults to 10.
/// - **factor**: Multiplier applied to the ATR for offsetting the bands. Defaults to 3.0.
///
/// ## Errors
/// - **EmptyData**: supertrend: Input data slice is empty.
/// - **InvalidPeriod**: supertrend: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: supertrend: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: supertrend: All input data values (high, low, close) are `NaN`.
///
/// ## Returns
/// - **`Ok(SuperTrendOutput)`** on success, containing two `Vec<f64>` (trend and changed),
///   each matching the input length, with leading `NaN`s for `trend` (and `0.0` for `changed`)
///   until the calculation window is filled.
/// - **`Err(SuperTrendError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum SuperTrendData<'a> {
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
pub struct SuperTrendParams {
    pub period: Option<usize>,
    pub factor: Option<f64>,
}

impl Default for SuperTrendParams {
    fn default() -> Self {
        Self {
            period: Some(10),
            factor: Some(3.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SuperTrendInput<'a> {
    pub data: SuperTrendData<'a>,
    pub params: SuperTrendParams,
}

impl<'a> SuperTrendInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: SuperTrendParams) -> Self {
        Self {
            data: SuperTrendData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: SuperTrendParams,
    ) -> Self {
        Self {
            data: SuperTrendData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SuperTrendData::Candles { candles },
            params: SuperTrendParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SuperTrendParams::default().period.unwrap())
    }

    fn get_factor(&self) -> f64 {
        self.params
            .factor
            .unwrap_or_else(|| SuperTrendParams::default().factor.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct SuperTrendOutput {
    pub trend: Vec<f64>,
    pub changed: Vec<f64>,
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SuperTrendError {
    #[error("supertrend: Empty data provided.")]
    EmptyData,
    #[error("supertrend: All values are NaN.")]
    AllValuesNaN,
    #[error("supertrend: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("supertrend: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(transparent)]
    AtrError(#[from] AtrError),
}

#[inline]
pub fn supertrend(input: &SuperTrendInput) -> Result<SuperTrendOutput, SuperTrendError> {
    let (high, low, close) = match &input.data {
        SuperTrendData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            (high, low, close)
        }
        SuperTrendData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(SuperTrendError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(SuperTrendError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }

    let factor = input.get_factor();

    let len = high.len();
    let mut first_valid_idx = None;
    for i in 0..len {
        if !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan() {
            first_valid_idx = Some(i);
            break;
        }
    }
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(SuperTrendError::AllValuesNaN),
    };
    if (len - first_valid_idx) < period {
        return Err(SuperTrendError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }

    let atr_input = AtrInput::from_slices(
        &high[first_valid_idx..],
        &low[first_valid_idx..],
        &close[first_valid_idx..],
        AtrParams {
            length: Some(period),
        },
    );
    let AtrOutput { values: atr_values } = atr(&atr_input)?;

    let mut trend = vec![f64::NAN; len];
    let mut changed = vec![0.0; len];

    let mut upper_basic = vec![f64::NAN; len - first_valid_idx];
    let mut lower_basic = vec![f64::NAN; len - first_valid_idx];
    let mut upper_band = vec![f64::NAN; len - first_valid_idx];
    let mut lower_band = vec![f64::NAN; len - first_valid_idx];

    for i in 0..(len - first_valid_idx) {
        let half_range = (high[first_valid_idx + i] + low[first_valid_idx + i]) / 2.0;
        upper_basic[i] = half_range + factor * atr_values[i];
        lower_basic[i] = half_range - factor * atr_values[i];
        upper_band[i] = upper_basic[i];
        lower_band[i] = lower_basic[i];
    }

    for i in period..(len - first_valid_idx) {
        let prev_close = close[first_valid_idx + i - 1];
        let prev_upper_band = upper_band[i - 1];
        let prev_lower_band = lower_band[i - 1];
        let curr_upper_basic = upper_basic[i];
        let curr_lower_basic = lower_basic[i];

        if prev_close <= prev_upper_band {
            upper_band[i] = f64::min(curr_upper_basic, prev_upper_band);
        }
        if prev_close >= prev_lower_band {
            lower_band[i] = f64::max(curr_lower_basic, prev_lower_band);
        }

        if prev_close <= prev_upper_band {
            trend[first_valid_idx + i - 1] = prev_upper_band;
        } else {
            trend[first_valid_idx + i - 1] = prev_lower_band;
        }
    }

    for i in period..(len - first_valid_idx) {
        let prev_close = close[first_valid_idx + i - 1];
        let prev_upper_band = upper_band[i - 1];
        let curr_upper_band = upper_band[i];
        let prev_lower_band = lower_band[i - 1];
        let curr_lower_band = lower_band[i];
        let prev_trend = trend[first_valid_idx + i - 1];

        if (prev_trend - prev_upper_band).abs() < f64::EPSILON {
            if close[first_valid_idx + i] <= curr_upper_band {
                trend[first_valid_idx + i] = curr_upper_band;
                changed[first_valid_idx + i] = 0.0;
            } else {
                trend[first_valid_idx + i] = curr_lower_band;
                changed[first_valid_idx + i] = 1.0;
            }
        } else if (prev_trend - prev_lower_band).abs() < f64::EPSILON {
            if close[first_valid_idx + i] >= curr_lower_band {
                trend[first_valid_idx + i] = curr_lower_band;
                changed[first_valid_idx + i] = 0.0;
            } else {
                trend[first_valid_idx + i] = curr_upper_band;
                changed[first_valid_idx + i] = 1.0;
            }
        }
    }

    Ok(SuperTrendOutput { trend, changed })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_supertrend_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = SuperTrendParams {
            period: None,
            factor: None,
        };
        let input_default = SuperTrendInput::from_candles(&candles, default_params);
        let output_default =
            supertrend(&input_default).expect("Failed SuperTrend with default params");
        assert_eq!(output_default.trend.len(), candles.close.len());
        assert_eq!(output_default.changed.len(), candles.close.len());

        let custom_params = SuperTrendParams {
            period: Some(7),
            factor: Some(2.5),
        };
        let input_custom = SuperTrendInput::from_candles(&candles, custom_params);
        let output_custom =
            supertrend(&input_custom).expect("Failed SuperTrend with custom params");
        assert_eq!(output_custom.trend.len(), candles.close.len());
        assert_eq!(output_custom.changed.len(), candles.close.len());
    }

    #[test]
    fn test_supertrend_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = SuperTrendParams {
            period: Some(10),
            factor: Some(3.0),
        };
        let input = SuperTrendInput::from_candles(&candles, params);
        let st_result = supertrend(&input).expect("Failed to calculate SuperTrend");

        assert_eq!(
            st_result.trend.len(),
            candles.close.len(),
            "SuperTrend length mismatch"
        );
        assert_eq!(
            st_result.changed.len(),
            candles.close.len(),
            "Changed length mismatch"
        );

        let expected_last_five_trend = [
            61811.479454208165,
            61721.73150878735,
            61459.10835790861,
            61351.59752211775,
            61033.18776990598,
        ];
        let expected_last_five_changed = [0.0, 0.0, 0.0, 0.0, 0.0];

        assert!(st_result.trend.len() >= 5, "SuperTrend length too short");
        let start_index = st_result.trend.len() - 5;
        let result_last_five_trend = &st_result.trend[start_index..];
        let result_last_five_changed = &st_result.changed[start_index..];

        for (i, &value) in result_last_five_trend.iter().enumerate() {
            let expected_value = expected_last_five_trend[i];
            assert!(
                (value - expected_value).abs() < 1e-4,
                "Trend mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        for (i, &value) in result_last_five_changed.iter().enumerate() {
            let expected_value = expected_last_five_changed[i];
            assert!(
                (value - expected_value).abs() < 1e-9,
                "Changed mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_supertrend_params_with_default_params() {
        let default_params = SuperTrendParams::default();
        assert_eq!(default_params.period, Some(10));
        assert_eq!(default_params.factor, Some(3.0));
    }

    #[test]
    fn test_supertrend_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = SuperTrendInput::with_default_candles(&candles);
        match input.data {
            SuperTrendData::Candles { .. } => {}
            _ => panic!("Expected SuperTrendData::Candles variant"),
        }
        assert_eq!(input.get_period(), 10);
        assert_eq!(input.get_factor(), 3.0);
    }

    #[test]
    fn test_supertrend_with_zero_period() {
        let high = [10.0, 12.0, 13.0];
        let low = [9.0, 11.0, 12.5];
        let close = [9.5, 11.5, 13.0];
        let params = SuperTrendParams {
            period: Some(0),
            factor: Some(3.0),
        };
        let input = SuperTrendInput::from_slices(&high, &low, &close, params);

        let result = supertrend(&input);
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
    fn test_supertrend_with_period_exceeding_data_length() {
        let high = [10.0, 12.0, 13.0];
        let low = [9.0, 11.0, 12.5];
        let close = [9.5, 11.5, 13.0];
        let params = SuperTrendParams {
            period: Some(10),
            factor: Some(3.0),
        };
        let input = SuperTrendInput::from_slices(&high, &low, &close, params);

        let result = supertrend(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_supertrend_very_small_data_set() {
        let high = [42.0];
        let low = [40.0];
        let close = [41.0];
        let params = SuperTrendParams {
            period: Some(10),
            factor: Some(3.0),
        };
        let input = SuperTrendInput::from_slices(&high, &low, &close, params);

        let result = supertrend(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_supertrend_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = SuperTrendParams {
            period: Some(10),
            factor: Some(3.0),
        };
        let first_input = SuperTrendInput::from_candles(&candles, first_params);
        let first_result = supertrend(&first_input).expect("Failed to calculate first SuperTrend");

        assert_eq!(
            first_result.trend.len(),
            candles.close.len(),
            "First SuperTrend trend length mismatch"
        );
        assert_eq!(
            first_result.changed.len(),
            candles.close.len(),
            "First SuperTrend changed length mismatch"
        );

        let second_params = SuperTrendParams {
            period: Some(5),
            factor: Some(2.0),
        };
        let second_input = SuperTrendInput::from_slices(
            &first_result.trend,
            &first_result.trend,
            &first_result.trend,
            second_params,
        );
        let second_result =
            supertrend(&second_input).expect("Failed to calculate second SuperTrend");

        assert_eq!(
            second_result.trend.len(),
            first_result.trend.len(),
            "Second SuperTrend trend length mismatch"
        );
        assert_eq!(
            second_result.changed.len(),
            first_result.trend.len(),
            "Second SuperTrend changed length mismatch"
        );
    }

    #[test]
    fn test_supertrend_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let params = SuperTrendParams {
            period: Some(10),
            factor: Some(3.0),
        };
        let input = SuperTrendInput::from_candles(&candles, params);
        let st_result = supertrend(&input).expect("Failed to calculate SuperTrend");

        assert_eq!(
            st_result.trend.len(),
            close_prices.len(),
            "SuperTrend length mismatch"
        );
        assert_eq!(
            st_result.changed.len(),
            close_prices.len(),
            "Changed length mismatch"
        );

        if st_result.trend.len() > 50 {
            for i in 50..st_result.trend.len() {
                assert!(
                    !st_result.trend[i].is_nan(),
                    "Expected no NaN after index 50 for trend, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
