/// # Midprice
///
/// The midpoint price over a specified period, calculated as `(highest high + lowest low) / 2`.
/// Useful for identifying average price levels in a range and potential support/resistance.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: midprice: Input data slice is empty.
/// - **InvalidPeriod**: midprice: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: midprice: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: midprice: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(MidpriceOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the window is filled.
/// - **`Err(MidpriceError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MidpriceData<'a> {
    Candles {
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MidpriceOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MidpriceParams {
    pub period: Option<usize>,
}

impl Default for MidpriceParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct MidpriceInput<'a> {
    pub data: MidpriceData<'a>,
    pub params: MidpriceParams,
}

impl<'a> MidpriceInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
        params: MidpriceParams,
    ) -> Self {
        Self {
            data: MidpriceData::Candles {
                candles,
                high_src,
                low_src,
            },
            params,
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: MidpriceParams) -> Self {
        Self {
            data: MidpriceData::Slices { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MidpriceData::Candles {
                candles,
                high_src: "high",
                low_src: "low",
            },
            params: MidpriceParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| MidpriceParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum MidpriceError {
    #[error("midprice: Empty data provided.")]
    EmptyData,
    #[error("midprice: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("midprice: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("midprice: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn midprice(input: &MidpriceInput) -> Result<MidpriceOutput, MidpriceError> {
    let (high, low) = match &input.data {
        MidpriceData::Candles {
            candles,
            high_src,
            low_src,
        } => {
            let h = source_type(candles, high_src);
            let l = source_type(candles, low_src);
            (h, l)
        }
        MidpriceData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(MidpriceError::EmptyData);
    }

    if high.len() != low.len() {
        return Err(MidpriceError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(MidpriceError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }

    let first_valid_idx = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(MidpriceError::AllValuesNaN),
    };

    if (high.len() - first_valid_idx) < period {
        return Err(MidpriceError::NotEnoughValidData {
            needed: period,
            valid: high.len() - first_valid_idx,
        });
    }

    let mut mid_values = vec![f64::NAN; high.len()];
    for i in (first_valid_idx + period - 1)..high.len() {
        let window_start = i + 1 - period;
        let mut highest = f64::NEG_INFINITY;
        let mut lowest = f64::INFINITY;
        for j in window_start..=i {
            if high[j] > highest {
                highest = high[j];
            }
            if low[j] < lowest {
                lowest = low[j];
            }
        }
        mid_values[i] = (highest + lowest) / 2.0;
    }

    Ok(MidpriceOutput { values: mid_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_midprice_default_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = MidpriceInput::with_default_candles(&candles);
        let output = midprice(&input).expect("Midprice calculation failed");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_midprice_custom_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params_period_7 = MidpriceParams { period: Some(7) };
        let input_period_7 = MidpriceInput::from_candles(&candles, "high", "low", params_period_7);
        let output_period_7 = midprice(&input_period_7).expect("Failed Midprice with period=7");
        assert_eq!(output_period_7.values.len(), candles.close.len());

        let params_custom = MidpriceParams { period: Some(20) };
        let input_custom = MidpriceInput::from_candles(&candles, "high", "low", params_custom);
        let output_custom = midprice(&input_custom).expect("Failed Midprice with period=20");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_midprice_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let high_prices = candles
            .select_candle_field("high")
            .expect("Failed to extract high prices");
        let low_prices = candles
            .select_candle_field("low")
            .expect("Failed to extract low prices");

        let params = MidpriceParams { period: Some(14) };
        let input = MidpriceInput::from_candles(&candles, "high", "low", params);
        let mid_result = midprice(&input).expect("Failed to calculate Midprice");

        assert_eq!(
            mid_result.values.len(),
            high_prices.len(),
            "Midprice length mismatch"
        );

        let expected_last_five_midprice = [59583.0, 59583.0, 59583.0, 59486.0, 58989.0];
        assert!(
            mid_result.values.len() >= 5,
            "Midprice output length is too short"
        );
        let start_index = mid_result.values.len() - 5;
        let result_last_five = &mid_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five_midprice[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "Midprice mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period: usize = 14;
        let first_valid_idx = match (0..high_prices.len())
            .find(|&i| !high_prices[i].is_nan() && !low_prices[i].is_nan())
        {
            Some(idx) => idx,
            None => 0,
        };
        for i in first_valid_idx..(first_valid_idx + period - 1) {
            assert!(
                mid_result.values[i].is_nan(),
                "Expected NaN in the warm-up period at index {}",
                i
            );
        }
    }

    #[test]
    fn test_midprice_with_zero_period() {
        let highs = [10.0, 14.0, 12.0];
        let lows = [5.0, 6.0, 7.0];
        let params = MidpriceParams { period: Some(0) };
        let input = MidpriceInput::from_slices(&highs, &lows, params);
        let result = midprice(&input);
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
    fn test_midprice_with_period_exceeding_data_length() {
        let highs = [10.0, 14.0, 12.0];
        let lows = [5.0, 6.0, 7.0];
        let params = MidpriceParams { period: Some(10) };
        let input = MidpriceInput::from_slices(&highs, &lows, params);
        let result = midprice(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_midprice_very_small_data_set() {
        let highs = [42.0];
        let lows = [36.0];
        let params = MidpriceParams { period: Some(14) };
        let input = MidpriceInput::from_slices(&highs, &lows, params);
        let result = midprice(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_midprice_slice_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params_primary = MidpriceParams { period: Some(10) };
        let input_primary = MidpriceInput::from_candles(&candles, "high", "low", params_primary);
        let primary_result = midprice(&input_primary).expect("Failed to calculate Midprice");

        let params_secondary = MidpriceParams { period: Some(10) };
        let input_secondary = MidpriceInput::from_slices(
            &primary_result.values,
            &primary_result.values,
            params_secondary,
        );
        let secondary_result = midprice(&input_secondary).expect("Failed to calculate Midprice");

        assert_eq!(secondary_result.values.len(), primary_result.values.len());
    }

    #[test]
    fn test_midprice_all_nan() {
        let highs = [f64::NAN, f64::NAN, f64::NAN];
        let lows = [f64::NAN, f64::NAN, f64::NAN];
        let params = MidpriceParams { period: Some(2) };
        let input = MidpriceInput::from_slices(&highs, &lows, params);
        let result = midprice(&input);
        assert!(result.is_err(), "Expected error for all NaN values");
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
    }
}
