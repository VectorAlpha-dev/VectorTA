/// # Predictive Moving Average (PMA)
///
/// Ehlers’ Predictive Moving Average calculates a smoothed value (`predict`)
/// and a signal line (`trigger`) by applying a series of weighted moving averages
/// and transformations to the input data. This indicator aims to predict future
/// price movements more responsively than standard moving averages.
///
/// ## Parameters
/// - **source**: The data field to be used from the candles (e.g., "close", "hl2", etc.).
///   Defaults to "close" when using `with_default_candles`.
///
/// ## Errors
/// - **EmptyData**: pma: Input data slice is empty.
/// - **NotEnoughValidData**: pma: Fewer than 7 valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: pma: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(PmaOutput)`** on success, containing two `Vec<f64>` (`predict` and `trigger`),
///   each matching the input length and filled with leading `NaN`s until enough data
///   points have accumulated.
/// - **`Err(PmaError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum PmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct PmaOutput {
    pub predict: Vec<f64>,
    pub trigger: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PmaParams;

impl Default for PmaParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct PmaInput<'a> {
    pub data: PmaData<'a>,
    pub params: PmaParams,
}

impl<'a> PmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: PmaParams) -> Self {
        Self {
            data: PmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: PmaParams) -> Self {
        Self {
            data: PmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: PmaData::Candles {
                candles,
                source: "close",
            },
            params: PmaParams::default(),
        }
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PmaError {
    #[error("pma: Empty data provided.")]
    EmptyData,
    #[error("pma: Not enough valid data: needed = 7, valid = {valid}")]
    NotEnoughValidData { valid: usize },
    #[error("pma: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn pma(input: &PmaInput) -> Result<PmaOutput, PmaError> {
    let data: &[f64] = match &input.data {
        PmaData::Candles { candles, source } => source_type(candles, source),
        PmaData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(PmaError::EmptyData);
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(PmaError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < 7 {
        return Err(PmaError::NotEnoughValidData {
            valid: data.len() - first_valid_idx,
        });
    }

    let mut predict = vec![f64::NAN; data.len()];
    let mut trigger = vec![f64::NAN; data.len()];
    let mut wma1 = vec![0.0; data.len()];

    for j in (first_valid_idx + 6)..data.len() {
        let wma1_j = ((7.0 * data[j])
            + (6.0 * data[j - 1])
            + (5.0 * data[j - 2])
            + (4.0 * data[j - 3])
            + (3.0 * data[j - 4])
            + (2.0 * data[j - 5])
            + data[j - 6])
            / 28.0;
        wma1[j] = wma1_j;

        let wma2 = ((7.0 * wma1[j])
            + (6.0 * wma1[j - 1])
            + (5.0 * wma1[j - 2])
            + (4.0 * wma1[j - 3])
            + (3.0 * wma1[j - 4])
            + (2.0 * wma1[j - 5])
            + wma1[j - 6])
            / 28.0;

        let predict_j = (2.0 * wma1_j) - wma2;
        predict[j] = predict_j;

        let trigger_j =
            ((4.0 * predict_j) + (3.0 * predict[j - 1]) + (2.0 * predict[j - 2]) + predict[j - 3])
                / 10.0;
        trigger[j] = trigger_j;
    }

    Ok(PmaOutput { predict, trigger })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_pma_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = PmaInput::with_default_candles(&candles);
        let output = pma(&input).expect("Failed PMA with default candles");
        assert_eq!(output.predict.len(), candles.close.len());
        assert_eq!(output.trigger.len(), candles.close.len());
    }

    #[test]
    fn test_pma_with_slice() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let input = PmaInput::from_slice(&data, PmaParams {});
        let output = pma(&input).expect("Failed PMA with slice");
        assert_eq!(output.predict.len(), data.len());
        assert_eq!(output.trigger.len(), data.len());
    }

    #[test]
    fn test_pma_not_enough_data() {
        let data = [10.0, 20.0, 30.0];
        let input = PmaInput::from_slice(&data, PmaParams {});
        let result = pma(&input);
        assert!(result.is_err(), "Expected error for not enough data");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Not enough valid data"),
                "Expected 'Not enough valid data' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_pma_all_values_nan() {
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let input = PmaInput::from_slice(&data, PmaParams {});
        let result = pma(&input);
        assert!(result.is_err(), "Expected error for all values NaN");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("All values are NaN"),
                "Expected 'All values are NaN' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_pma_expected_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = PmaInput::from_candles(&candles, "hl2", PmaParams {});
        let result = pma(&input).expect("Failed to calculate PMA");

        assert_eq!(
            result.predict.len(),
            candles.close.len(),
            "Predict length mismatch"
        );
        assert_eq!(
            result.trigger.len(),
            candles.close.len(),
            "Trigger length mismatch"
        );

        let expected_predict = [
            59208.18749999999,
            59233.83609693878,
            59213.19132653061,
            59199.002551020414,
            58993.318877551,
        ];
        let expected_trigger = [
            59157.70790816327,
            59208.60076530612,
            59218.6763392857,
            59211.1443877551,
            59123.05019132652,
        ];

        assert!(
            result.predict.len() >= 5,
            "Output length too short for checking"
        );
        let start_idx = result.predict.len() - 5;
        for i in 0..5 {
            let calc_val = result.predict[start_idx + i];
            let exp_val = expected_predict[i];
            assert!(
                (calc_val - exp_val).abs() < 1e-1,
                "Mismatch in predict at index {}: expected {}, got {}",
                start_idx + i,
                exp_val,
                calc_val
            );
        }

        assert!(
            result.trigger.len() >= 5,
            "Output length too short for checking"
        );
        for i in 0..5 {
            let calc_val = result.trigger[start_idx + i];
            let exp_val = expected_trigger[i];
            assert!(
                (calc_val - exp_val).abs() < 1e-1,
                "Mismatch in trigger at index {}: expected {}, got {}",
                start_idx + i,
                exp_val,
                calc_val
            );
        }
    }
}
