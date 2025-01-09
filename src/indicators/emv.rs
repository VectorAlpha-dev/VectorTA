/// # Ease of Movement (EMV)
///
/// EMV measures the "ease" with which price moves by combining midpoint changes (based on
/// high and low) and volume. This indicator helps visualize how much volume is required
/// to move price. A large absolute EMV value typically indicates more significant price
/// movement relative to volume.
///
/// ## Parameters
/// *No parameters* (This version computes EMV over the entire dataset.)
///
/// ## Errors
/// - **EmptyData**: emv: Input data slice is empty.
/// - **NotEnoughData**: emv: Fewer than 2 valid (non-`NaN`) data points are available.
/// - **AllValuesNaN**: emv: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(EmvOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s before the first valid EMV calculation.
/// - **`Err(EmvError)`** otherwise.
use crate::utilities::data_loader::{read_candles_from_csv, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum EmvData<'a> {
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
pub struct EmvOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct EmvParams;

#[derive(Debug, Clone)]
pub struct EmvInput<'a> {
    pub data: EmvData<'a>,
    pub params: EmvParams,
}

impl<'a> EmvInput<'a> {
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: EmvData::Candles { candles },
            params: EmvParams::default(),
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    ) -> Self {
        Self {
            data: EmvData::Slices {
                high,
                low,
                close,
                volume,
            },
            params: EmvParams::default(),
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles)
    }
}

#[derive(Debug, Error)]
pub enum EmvError {
    #[error("emv: Empty data provided.")]
    EmptyData,
    #[error("emv: Not enough data: needed at least 2 valid points, found {valid}.")]
    NotEnoughData { valid: usize },
    #[error("emv: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn emv(input: &EmvInput) -> Result<EmvOutput, EmvError> {
    let (high, low, _close, volume) = match &input.data {
        EmvData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| EmvError::EmptyData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| EmvError::EmptyData)?;
            let close = candles
                .select_candle_field("close")
                .map_err(|_| EmvError::EmptyData)?;
            let volume = candles
                .select_candle_field("volume")
                .map_err(|_| EmvError::EmptyData)?;
            (high, low, close, volume)
        }
        EmvData::Slices {
            high,
            low,
            close,
            volume,
        } => (*high, *low, *close, *volume),
    };

    if high.is_empty() || low.is_empty() || volume.is_empty() {
        return Err(EmvError::EmptyData);
    }
    let len = high.len().min(low.len()).min(volume.len());
    if len == 0 {
        return Err(EmvError::EmptyData);
    }

    let mut emv_values = vec![f64::NAN; len];

    let first_valid_idx =
        (0..len).find(|&i| !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()));
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(EmvError::AllValuesNaN),
    };

    let mut valid_count = 0_usize;
    for i in first_valid_idx..len {
        if !(high[i].is_nan() || low[i].is_nan() || volume[i].is_nan()) {
            valid_count += 1;
        }
    }
    if valid_count < 2 {
        return Err(EmvError::NotEnoughData { valid: valid_count });
    }

    let mut last_mid = 0.5 * (high[first_valid_idx] + low[first_valid_idx]);
    for i in (first_valid_idx + 1)..len {
        if high[i].is_nan() || low[i].is_nan() || volume[i].is_nan() {
            emv_values[i] = f64::NAN;
            continue;
        }
        let current_mid = 0.5 * (high[i] + low[i]);
        let range = high[i] - low[i];
        if range == 0.0 {
            emv_values[i] = f64::NAN;
            last_mid = current_mid;
            continue;
        }
        let br = volume[i] / 10000.0 / range;
        emv_values[i] = (current_mid - last_mid) / br;
        last_mid = current_mid;
    }

    Ok(EmvOutput { values: emv_values })
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emv_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let emv_input = EmvInput::from_candles(&candles);
        let emv_output = emv(&emv_input).expect("Failed to calculate EMV");
        assert_eq!(emv_output.values.len(), candles.close.len());

        let expected_last_five_emv = [
            -6488905.579799851,
            2371436.7401001123,
            -3855069.958128531,
            1051939.877943717,
            -8519287.22257077,
        ];
        assert!(
            emv_output.values.len() >= 5,
            "EMV length is too short to check last five values."
        );
        let start_index: usize = emv_output.values.len() - 5;
        let actual_last_five: &[f64] = &emv_output.values[start_index..];

        for (i, &value) in actual_last_five.iter().enumerate() {
            let expected_value: f64 = expected_last_five_emv[i];
            let tolerance = expected_value.abs() * 0.0001;
            assert!(
                (value - expected_value).abs() <= tolerance,
                "EMV mismatch at index {}: expected {}, got {}, tolerance ±{}",
                i,
                expected_value,
                value,
                tolerance
            );
        }
    }

    #[test]
    fn test_emv_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = EmvInput::with_default_candles(&candles);
        let output = emv(&input).expect("Failed EMV with default candles");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_emv_empty_data() {
        let empty: [f64; 0] = [];
        let input = EmvInput::from_slices(&empty, &empty, &empty, &empty);
        let result = emv(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_emv_all_nan() {
        let nan_arr = [f64::NAN, f64::NAN];
        let input = EmvInput::from_slices(&nan_arr, &nan_arr, &nan_arr, &nan_arr);
        let result = emv(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_emv_not_enough_data() {
        let high = [10000.0, f64::NAN];
        let low = [9990.0, f64::NAN];
        let close = [9995.0, f64::NAN];
        let volume = [1_000_000.0, f64::NAN];
        let input = EmvInput::from_slices(&high, &low, &close, &volume);

        let result = emv(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_emv_basic_calculation() {
        let high = [10.0, 12.0, 13.0, 15.0];
        let low = [5.0, 7.0, 8.0, 10.0];
        let close = [7.5, 9.0, 10.5, 12.5];
        let volume = [10000.0, 20000.0, 25000.0, 30000.0];

        let input = EmvInput::from_slices(&high, &low, &close, &volume);
        let output = emv(&input).expect("Failed EMV calculation");
        assert_eq!(output.values.len(), 4);
        assert!(output.values[0].is_nan());
        for &val in &output.values[1..] {
            assert!(!val.is_nan());
        }
    }

    fn first_valid_non_nan(values: &[f64]) -> usize {
        values
            .iter()
            .position(|v| !v.is_nan())
            .unwrap_or(values.len())
    }
}
