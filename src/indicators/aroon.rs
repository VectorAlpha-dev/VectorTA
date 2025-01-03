/// # Aroon
///
/// A trend-following indicator designed by Tushar Chande that measures the strength and
/// potential direction of a market trend based on the recent highs and lows over a specified
/// `length`. It provides two outputs:
/// - **`aroon_up`**: How close the most recent highest high is to the current bar (as a percentage).
/// - **`aroon_down`**: How close the most recent lowest low is to the current bar (as a percentage).
///
/// ## Parameters
/// - **length**: The lookback period used to determine the highest high and lowest low
///   (defaults to 14).
///
/// ## Errors
/// - **NoCandlesAvailable**: aroon: No candle data was found.
/// - **EmptySlices**: aroon: One or both of the `high`/`low` slices are empty.
/// - **MismatchSliceLength**: aroon: `high` and `low` slices differ in length.
/// - **NotEnoughData**: aroon: The data length is smaller than the specified `length`.
/// - **ZeroLength**: aroon: `length` is zero (invalid).
///
/// ## Returns
/// - **`Ok(AroonOutput)`** on success, containing:
///   - `aroon_up`: A `Vec<f64>` representing the Aroon Up values.
///   - `aroon_down`: A `Vec<f64>` representing the Aroon Down values.
/// - **`Err(AroonError)`** otherwise.
use crate::utilities::data_loader::Candles;

#[derive(Debug, Clone)]
pub enum AroonData<'a> {
    Candles { candles: &'a Candles },
    SlicesHL { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct AroonParams {
    pub length: Option<usize>,
}

impl Default for AroonParams {
    fn default() -> Self {
        Self { length: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct AroonInput<'a> {
    pub data: AroonData<'a>,
    pub params: AroonParams,
}

impl<'a> AroonInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: AroonParams) -> Self {
        Self {
            data: AroonData::Candles { candles },
            params,
        }
    }

    pub fn from_slices_hl(high: &'a [f64], low: &'a [f64], params: AroonParams) -> Self {
        Self {
            data: AroonData::SlicesHL { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AroonData::Candles { candles },
            params: AroonParams::default(),
        }
    }

    pub fn get_length(&self) -> usize {
        self.params
            .length
            .unwrap_or_else(|| AroonParams::default().length.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct AroonOutput {
    pub aroon_up: Vec<f64>,
    pub aroon_down: Vec<f64>,
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AroonError {
    #[error(transparent)]
    CandleFieldError(#[from] Box<dyn std::error::Error>),

    #[error("No candles available for Aroon.")]
    NoCandlesAvailable,

    #[error("One or both slices for Aroon are empty: high_len={high_len}, low_len={low_len}")]
    EmptySlices { high_len: usize, low_len: usize },

    #[error("Aroon: Mismatch in high/low slice length: high_len={high_len}, low_len={low_len}")]
    MismatchSliceLength { high_len: usize, low_len: usize },

    #[error(
        "Aroon: Not enough data points for Aroon: data length={data_len}, required={required}"
    )]
    NotEnoughData { data_len: usize, required: usize },

    #[error("Invalid length specified for Aroon calculation (length=0).")]
    ZeroLength,
}

#[inline]
pub fn aroon(input: &AroonInput) -> Result<AroonOutput, AroonError> {
    let (high, low) = match &input.data {
        AroonData::Candles { candles } => {
            if candles.close.is_empty() {
                return Err(AroonError::NoCandlesAvailable);
            }
            (
                candles.select_candle_field("high")?,
                candles.select_candle_field("low")?,
            )
        }
        AroonData::SlicesHL { high, low } => {
            let h_len = high.len();
            let l_len = low.len();

            if h_len == 0 || l_len == 0 {
                return Err(AroonError::EmptySlices {
                    high_len: h_len,
                    low_len: l_len,
                });
            }
            if h_len != l_len {
                return Err(AroonError::MismatchSliceLength {
                    high_len: h_len,
                    low_len: l_len,
                });
            }
            (*high, *low)
        }
    };

    let length = input.get_length();
    let data_len = high.len();
    if data_len < length {
        return Err(AroonError::NotEnoughData {
            data_len,
            required: length,
        });
    }
    if length == 0 {
        return Err(AroonError::ZeroLength);
    }
    let len = low.len();
    if length == 0 {
        return Err(AroonError::ZeroLength);
    }
    let mut aroon_up = vec![f64::NAN; len];
    let mut aroon_down = vec![f64::NAN; len];

    let window = length + 1;
    let inv_length = 1.0 / length as f64;

    for i in (window - 1)..len {
        let start = i + 1 - window;
        let mut highest_val = high[start];
        let mut lowest_val = low[start];
        let mut highest_idx = start;
        let mut lowest_idx = start;

        for j in (start + 1)..=i {
            let h_val = high[j];
            if h_val > highest_val {
                highest_val = h_val;
                highest_idx = j;
            }
            let l_val = low[j];
            if l_val < lowest_val {
                lowest_val = l_val;
                lowest_idx = j;
            }
        }

        let offset_highest = i - highest_idx;
        let offset_lowest = i - lowest_idx;

        aroon_up[i] = (length as f64 - offset_highest as f64) * inv_length * 100.0;
        aroon_down[i] = (length as f64 - offset_lowest as f64) * inv_length * 100.0;
    }

    Ok(AroonOutput {
        aroon_up,
        aroon_down,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_aroon_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AroonInput::with_default_candles(&candles);
        let result = aroon(&input).expect("Failed to calculate Aroon");

        let expected_up_last_five = [21.43, 14.29, 7.14, 0.0, 0.0];
        let expected_down_last_five = [71.43, 64.29, 57.14, 50.0, 42.86];

        assert!(
            result.aroon_up.len() >= 5 && result.aroon_down.len() >= 5,
            "Not enough Aroon values"
        );

        assert_eq!(
            result.aroon_up.len(),
            result.aroon_down.len(),
            "Aroon Up and Down lengths mismatch"
        );

        assert_eq!(
            result.aroon_up.len(),
            candles.close.len(),
            "Aroon output length does not match input length"
        );

        let start_index = result.aroon_up.len().saturating_sub(5);

        let up_last_five = &result.aroon_up[start_index..];
        let down_last_five = &result.aroon_down[start_index..];

        for (i, &value) in up_last_five.iter().enumerate() {
            assert!(
                (value - expected_up_last_five[i]).abs() < 1e-2,
                "Aroon Up mismatch at index {}: expected {}, got {}",
                i,
                expected_up_last_five[i],
                value
            );
        }

        for (i, &value) in down_last_five.iter().enumerate() {
            assert!(
                (value - expected_down_last_five[i]).abs() < 1e-2,
                "Aroon Down mismatch at index {}: expected {}, got {}",
                i,
                expected_down_last_five[i],
                value
            );
        }

        let length = 14;
        for val in result.aroon_up.iter().skip(length) {
            if !val.is_nan() {
                assert!(
                    val.is_finite(),
                    "Aroon Up should be finite after enough data"
                );
            }
        }
        for val in result.aroon_down.iter().skip(length) {
            if !val.is_nan() {
                assert!(
                    val.is_finite(),
                    "Aroon Down should be finite after enough data"
                );
            }
        }
    }

    #[test]
    fn test_aroon_params_with_default_params() {
        let default_params = AroonParams::default();
        assert_eq!(default_params.length, Some(14));
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AroonInput::from_candles(&candles, default_params);
        let result = aroon(&input).expect("Failed Aroon with default params");
        assert_eq!(result.aroon_up.len(), candles.close.len());
        assert_eq!(result.aroon_down.len(), candles.close.len());
    }

    #[test]
    fn test_aroon_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AroonInput::with_default_candles(&candles);
        match input.data {
            AroonData::Candles { .. } => {}
            _ => panic!("Expected AroonData::Candles variant"),
        }
        let result = aroon(&input).expect("Failed to calculate Aroon with default candles");
        assert_eq!(result.aroon_up.len(), candles.close.len());
        assert_eq!(result.aroon_down.len(), candles.close.len());
    }

    #[test]
    fn test_aroon_with_zero_length() {
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 10.0, 11.0];
        let params = AroonParams { length: Some(0) };
        let input = AroonInput::from_slices_hl(&high, &low, params);
        let result = aroon(&input);
        assert!(result.is_err(), "Expected error for zero length");
    }

    #[test]
    fn test_aroon_with_length_exceeding_data_length() {
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 10.0, 11.0];
        let params = AroonParams { length: Some(14) };
        let input = AroonInput::from_slices_hl(&high, &low, params);
        let result = aroon(&input);
        assert!(result.is_err(), "Expected error for length > data.len()");
    }

    #[test]
    fn test_aroon_very_small_data_set() {
        let high = [100.0];
        let low = [99.5];
        let params = AroonParams { length: Some(14) };
        let input = AroonInput::from_slices_hl(&high, &low, params);
        let result = aroon(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than length"
        );
    }

    #[test]
    fn test_aroon_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = AroonParams { length: Some(14) };
        let first_input = AroonInput::from_candles(&candles, first_params);
        let first_result = aroon(&first_input).expect("Failed Aroon calculation");
        assert_eq!(first_result.aroon_up.len(), candles.close.len());
        assert_eq!(first_result.aroon_down.len(), candles.close.len());
        let second_params = AroonParams { length: Some(5) };
        let second_input = AroonInput::from_slices_hl(&candles.high, &candles.low, second_params);
        let second_result = aroon(&second_input).expect("Failed second Aroon calculation");
        assert_eq!(second_result.aroon_up.len(), candles.close.len());
        assert_eq!(second_result.aroon_down.len(), candles.close.len());
        if first_result.aroon_up.len() > 240 {
            for i in 240..first_result.aroon_up.len() {
                assert!(
                    !first_result.aroon_up[i].is_nan(),
                    "Found NaN in aroon_up at {}",
                    i
                );
                assert!(
                    !first_result.aroon_down[i].is_nan(),
                    "Found NaN in aroon_down at {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_aroon_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = AroonParams { length: Some(14) };
        let input = AroonInput::from_candles(&candles, params);
        let result = aroon(&input).expect("Failed Aroon calculation");
        assert_eq!(result.aroon_up.len(), candles.close.len());
        assert_eq!(result.aroon_down.len(), candles.close.len());
        if result.aroon_up.len() > 240 {
            for i in 240..result.aroon_up.len() {
                assert!(
                    !result.aroon_up[i].is_nan(),
                    "Found NaN in aroon_up at {}",
                    i
                );
                assert!(
                    !result.aroon_down[i].is_nan(),
                    "Found NaN in aroon_down at {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_aroon_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let partial_params = AroonParams { length: None };
        let input = AroonInput::from_candles(&candles, partial_params);
        let result = aroon(&input).expect("Failed Aroon with partial params");
        assert_eq!(result.aroon_up.len(), candles.close.len());
        assert_eq!(result.aroon_down.len(), candles.close.len());
    }
}
