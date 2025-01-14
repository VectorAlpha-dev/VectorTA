/// # Median Price (MEDPRICE)
///
/// The median price is calculated as `(high + low) / 2.0` for each data point.
/// This indicator uses the provided high and low price sources and returns a
/// vector of median prices. Leading `NaN` values will be produced until the
/// first valid (non-`NaN`) values of both `high` and `low` are encountered.
///
/// ## Parameters
/// *None*
///
/// ## Errors
/// - **EmptyData**: medprice: Input data slices are empty.
/// - **DifferentLength**: medprice: `high` and `low` data slices have different lengths.
/// - **AllValuesNaN**: medprice: All input data values (high or low) are `NaN`.
///
/// ## Returns
/// - **`Ok(MedpriceOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN` until the first valid high/low pair is encountered.
/// - **`Err(MedpriceError)`** otherwise.
use crate::utilities::data_loader::{read_candles_from_csv, source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MedpriceData<'a> {
    Candles {
        candles: &'a Candles,
        high_source: &'a str,
        low_source: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MedpriceOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MedpriceParams;

impl Default for MedpriceParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct MedpriceInput<'a> {
    pub data: MedpriceData<'a>,
    pub params: MedpriceParams,
}

impl<'a> MedpriceInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        high_source: &'a str,
        low_source: &'a str,
        params: MedpriceParams,
    ) -> Self {
        Self {
            data: MedpriceData::Candles {
                candles,
                high_source,
                low_source,
            },
            params,
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: MedpriceParams) -> Self {
        Self {
            data: MedpriceData::Slices { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MedpriceData::Candles {
                candles,
                high_source: "high",
                low_source: "low",
            },
            params: MedpriceParams::default(),
        }
    }
}

#[derive(Debug, Error)]
pub enum MedpriceError {
    #[error("medprice: Empty data provided.")]
    EmptyData,
    #[error("medprice: Different lengths for high ({high_len}) and low ({low_len}).")]
    DifferentLength { high_len: usize, low_len: usize },
    #[error("medprice: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn medprice(input: &MedpriceInput) -> Result<MedpriceOutput, MedpriceError> {
    let (high, low) = match &input.data {
        MedpriceData::Candles {
            candles,
            high_source,
            low_source,
        } => (
            source_type(candles, high_source),
            source_type(candles, low_source),
        ),
        MedpriceData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(MedpriceError::EmptyData);
    }
    if high.len() != low.len() {
        return Err(MedpriceError::DifferentLength {
            high_len: high.len(),
            low_len: low.len(),
        });
    }

    let mut med_values = vec![f64::NAN; high.len()];

    let first_valid_idx = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(MedpriceError::AllValuesNaN),
    };

    for i in first_valid_idx..high.len() {
        if high[i].is_nan() || low[i].is_nan() {
            continue;
        }
        med_values[i] = (high[i] + low[i]) / 2.0;
    }

    Ok(MedpriceOutput { values: med_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_medprice_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = MedpriceInput::with_default_candles(&candles);
        let output = medprice(&input).expect("Failed to compute medprice with default candles");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_medprice_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = MedpriceInput::from_candles(&candles, "high", "low", MedpriceParams);
        let result = medprice(&input).expect("Failed to compute medprice");

        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "Output length mismatch"
        );

        let expected_last_five = [59166.0, 59244.5, 59118.0, 59146.5, 58767.5];
        assert!(result.values.len() >= 5, "Not enough data for comparison");
        let start_index = result.values.len() - 5;
        let actual_last_five = &result.values[start_index..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (val - expected).abs() < 1e-1,
                "Mismatch at last five index {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }
    }

    #[test]
    fn test_medprice_empty_data() {
        let high = [];
        let low = [];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice(&input);
        assert!(result.is_err(), "Expected error for empty data");
    }

    #[test]
    fn test_medprice_different_length() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice(&input);
        assert!(
            result.is_err(),
            "Expected error for different slice lengths"
        );
    }

    #[test]
    fn test_medprice_all_values_nan() {
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice(&input);
        assert!(result.is_err(), "Expected error for all NaN data");
    }

    #[test]
    fn test_medprice_nan_handling() {
        let high = [f64::NAN, 100.0, 110.0];
        let low = [f64::NAN, 80.0, 90.0];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice(&input).expect("Failed to compute medprice with partial NaNs");
        assert_eq!(result.values.len(), 3);
        assert!(result.values[0].is_nan());
        assert_eq!(result.values[1], 90.0);
        assert_eq!(result.values[2], 100.0);
    }

    #[test]
    fn test_medprice_late_nan_handling() {
        let high = [100.0, 110.0, f64::NAN];
        let low = [80.0, 90.0, f64::NAN];
        let input = MedpriceInput::from_slices(&high, &low, MedpriceParams);
        let result = medprice(&input).expect("Failed to compute medprice with late NaNs");
        assert_eq!(result.values.len(), 3);
        assert_eq!(result.values[0], 90.0);
        assert_eq!(result.values[1], 100.0);
        assert!(result.values[2].is_nan());
    }
}
