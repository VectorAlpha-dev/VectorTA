/// # Negative Volume Index (NVI)
///
/// The NVI (Negative Volume Index) focuses on days when the volume decreases from the previous day.
/// It assumes that the “smart money” is trading on these days. A rising NVI suggests that smart money
/// is becoming more bullish, while a declining NVI suggests the opposite.
///
/// This implementation follows the Tulip Indicators reference and does not take any parameters.
/// It initializes NVI with a value of `1000.0` at the first valid (non-`NaN`) data point, then updates
/// only on days when `volume[i] < volume[i-1]`.
///
/// ## Errors
/// - **EmptyData**: nvi: Input data slice(s) is empty.
/// - **AllCloseValuesNaN**: nvi: All close input values are `NaN`.
/// - **AllVolumeValuesNaN**: nvi: All volume input values are `NaN`.
/// - **NotEnoughValidData**: nvi: Fewer than 2 valid (non-`NaN`) data points remain after the first valid index.
///
/// ## Returns
/// - **`Ok(NviOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the first valid index.
/// - **`Err(NviError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum NviData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct NviOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct NviInput<'a> {
    pub data: NviData<'a>,
}

impl<'a> NviInput<'a> {
    pub fn from_candles(candles: &'a Candles, close_source: &'a str) -> Self {
        Self {
            data: NviData::Candles {
                candles,
                close_source,
            },
        }
    }

    pub fn from_slices(close: &'a [f64], volume: &'a [f64]) -> Self {
        Self {
            data: NviData::Slices { close, volume },
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: NviData::Candles {
                candles,
                close_source: "close",
            },
        }
    }
}

#[derive(Debug, Error)]
pub enum NviError {
    #[error("nvi: Empty data provided.")]
    EmptyData,
    #[error("nvi: All close values are NaN.")]
    AllCloseValuesNaN,
    #[error("nvi: All volume values are NaN.")]
    AllVolumeValuesNaN,
    #[error("nvi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn nvi(input: &NviInput) -> Result<NviOutput, NviError> {
    let (close, volume) = match &input.data {
        NviData::Candles {
            candles,
            close_source,
        } => {
            let close = source_type(candles, close_source);
            let volume = candles
                .select_candle_field("volume")
                .map_err(|_| NviError::EmptyData)?;
            (close, volume)
        }
        NviData::Slices { close, volume } => (*close, *volume),
    };

    if close.is_empty() || volume.is_empty() {
        return Err(NviError::EmptyData);
    }

    let first_valid_idx = match close
        .iter()
        .zip(volume.iter())
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
    {
        Some(idx) => idx,
        None => {
            if close.iter().all(|&c| c.is_nan()) {
                return Err(NviError::AllCloseValuesNaN);
            } else {
                return Err(NviError::AllVolumeValuesNaN);
            }
        }
    };

    if (close.len() - first_valid_idx) < 2 {
        return Err(NviError::NotEnoughValidData {
            needed: 2,
            valid: close.len() - first_valid_idx,
        });
    }

    let mut output = vec![f64::NAN; close.len()];
    let mut nvi_val = 1000.0;
    output[0] = volume[0] + nvi_val;

    for i in (1)..close.len() {
        if volume[i] < volume[i - 1] {
            let pct_change = (close[i] - close[i - 1]) / close[i - 1];
            nvi_val += (nvi_val * pct_change);
        }
        output[i] = nvi_val;
    }

    Ok(NviOutput { values: output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_nvi_with_slices() {
        let close_data = [10.0, 10.5, 10.3, 10.7, 10.2];
        let volume_data = [100.0, 90.0, 80.0, 120.0, 70.0];
        let input = NviInput::from_slices(&close_data, &volume_data);
        let result = nvi(&input).expect("Failed NVI with slices");
        assert_eq!(result.values.len(), close_data.len());
        assert!(
            result.values.iter().any(|&x| !x.is_nan()),
            "Expected some non-NaN values in NVI output"
        );
    }

    #[test]
    fn test_nvi_with_empty_data() {
        let close_data: [f64; 0] = [];
        let volume_data: [f64; 0] = [];
        let input = NviInput::from_slices(&close_data, &volume_data);
        let result = nvi(&input);
        assert!(result.is_err(), "Expected 'Empty data' error");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Empty data"),
                "Expected 'Empty data' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_nvi_not_enough_valid_data() {
        let close_data = [f64::NAN, 100.0];
        let volume_data = [f64::NAN, 120.0];
        let input = NviInput::from_slices(&close_data, &volume_data);
        let result = nvi(&input);
        assert!(result.is_err(), "Expected 'Not enough valid data' error");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Not enough valid data"),
                "Expected 'Not enough valid data' error, got: {}",
                e
            );
        }
    }

    #[test]
    #[ignore]
    fn test_nvi_accuracy_with_csv() {
        let expected_last_five = [
            17555.49871646325,
            17524.70219345554,
            17524.70219345554,
            17559.13477961792,
            17559.13477961792,
        ];

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let input = NviInput::with_default_candles(&candles);
        let nvi_result = nvi(&input).expect("Failed to calculate NVI");
        assert_eq!(
            nvi_result.values.len(),
            close_prices.len(),
            "NVI length mismatch"
        );

        assert!(
            nvi_result.values.len() >= 5,
            "NVI length too short to compare last five values"
        );

        let start_idx = nvi_result.values.len() - 5;
        let end_section = &nvi_result.values[start_idx..];
        for (i, &value) in end_section.iter().enumerate() {
            let expected_val = expected_last_five[i];
            assert!(
                (value - expected_val).abs() < 1e-5,
                "NVI mismatch at index {}, expected {}, got {}",
                i,
                expected_val,
                value
            );
        }
    }

    #[test]
    fn test_nvi_with_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = NviInput::with_default_candles(&candles);
        let output = nvi(&input).expect("Failed NVI calculation with candles");
        assert_eq!(output.values.len(), candles.close.len());
    }
}
