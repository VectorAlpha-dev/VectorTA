/// # Positive Volume Index (PVI)
///
/// The Positive Volume Index (PVI) starts at an initial value (commonly 1000) and tracks changes in price
/// when volume increases compared to the previous bar. If volume does not increase, the PVI remains
/// unchanged. It is often used in conjunction with the Negative Volume Index (NVI) to analyze market trends.
///
/// ## Parameters
/// - **initial_value**: The starting PVI value. Defaults to 1000.
///
/// ## Errors
/// - **EmptyData**: pvi: Input data slice is empty.
/// - **AllValuesNaN**: pvi: All input data values are `NaN`.
/// - **MismatchedLength**: pvi: Close and volume data have different lengths.
/// - **NotEnoughValidData**: pvi: Fewer than 2 valid (non-`NaN`) data points remain after the first valid index.
///
/// ## Returns
/// - **`Ok(PviOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the first valid (non-`NaN`) data point.
/// - **`Err(PviError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum PviData<'a> {
    Candles {
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
    },
    Slices {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct PviOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct PviParams {
    pub initial_value: Option<f64>,
}

impl Default for PviParams {
    fn default() -> Self {
        Self {
            initial_value: Some(1000.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PviInput<'a> {
    pub data: PviData<'a>,
    pub params: PviParams,
}

impl<'a> PviInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        close_source: &'a str,
        volume_source: &'a str,
        params: PviParams,
    ) -> Self {
        Self {
            data: PviData::Candles {
                candles,
                close_source,
                volume_source,
            },
            params,
        }
    }

    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: PviParams) -> Self {
        Self {
            data: PviData::Slices { close, volume },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: PviData::Candles {
                candles,
                close_source: "close",
                volume_source: "volume",
            },
            params: PviParams::default(),
        }
    }

    pub fn get_initial_value(&self) -> f64 {
        self.params
            .initial_value
            .unwrap_or_else(|| PviParams::default().initial_value.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PviError {
    #[error("pvi: Empty data provided.")]
    EmptyData,
    #[error("pvi: All values are NaN.")]
    AllValuesNaN,
    #[error("pvi: Close and volume data have different lengths.")]
    MismatchedLength,
    #[error("pvi: Not enough valid data: needed at least 2 valid data points.")]
    NotEnoughValidData,
}

#[inline]
pub fn pvi(input: &PviInput) -> Result<PviOutput, PviError> {
    let (close, volume) = match &input.data {
        PviData::Candles {
            candles,
            close_source,
            volume_source,
        } => {
            let c = source_type(candles, close_source);
            let v = source_type(candles, volume_source);
            (c, v)
        }
        PviData::Slices { close, volume } => (*close, *volume),
    };

    if close.is_empty() || volume.is_empty() {
        return Err(PviError::EmptyData);
    }

    if close.len() != volume.len() {
        return Err(PviError::MismatchedLength);
    }

    let first_valid_idx = match close
        .iter()
        .zip(volume.iter())
        .position(|(&c, &v)| !c.is_nan() && !v.is_nan())
    {
        Some(idx) => idx,
        None => return Err(PviError::AllValuesNaN),
    };

    if (close.len() - first_valid_idx) < 2 {
        return Err(PviError::NotEnoughValidData);
    }

    let mut pvi_values = vec![f64::NAN; close.len()];
    let mut pvi_current = input.get_initial_value();
    pvi_values[first_valid_idx] = pvi_current;

    for i in (first_valid_idx + 1)..close.len() {
        if !close[i].is_nan()
            && !volume[i].is_nan()
            && !close[i - 1].is_nan()
            && !volume[i - 1].is_nan()
        {
            if volume[i] > volume[i - 1] {
                pvi_current += ((close[i] - close[i - 1]) / close[i - 1]) * pvi_current;
            }
            pvi_values[i] = pvi_current;
        }
    }

    Ok(PviOutput { values: pvi_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_pvi_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = PviInput::with_default_candles(&candles);
        let output = pvi(&input).expect("Failed PVI with default parameters");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_pvi_custom_sources() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = PviParams {
            initial_value: Some(1500.0),
        };
        let input = PviInput::from_candles(&candles, "close", "volume", params);
        let output = pvi(&input).expect("Failed PVI with custom sources");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_pvi_with_slices() {
        let close_data = [100.0, 101.0, 102.0, 101.0, 103.0];
        let volume_data = [500.0, 600.0, 700.0, 650.0, 800.0];
        let params = PviParams::default();
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let output = pvi(&input).expect("Failed PVI with slices");
        assert_eq!(output.values.len(), close_data.len());
        assert!(!output.values[4].is_nan());
    }

    #[test]
    fn test_pvi_empty_data() {
        let close_data = [];
        let volume_data = [];
        let params = PviParams::default();
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let result = pvi(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Empty data"));
        }
    }

    #[test]
    fn test_pvi_mismatched_length() {
        let close_data = [100.0, 101.0];
        let volume_data = [500.0];
        let params = PviParams::default();
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let result = pvi(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e
                .to_string()
                .contains("Close and volume data have different lengths"));
        }
    }

    #[test]
    fn test_pvi_all_values_nan() {
        let close_data = [f64::NAN, f64::NAN, f64::NAN];
        let volume_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = PviParams::default();
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let result = pvi(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
    }

    #[test]
    fn test_pvi_not_enough_valid_data() {
        let close_data = [f64::NAN, 100.0];
        let volume_data = [f64::NAN, 500.0];
        let params = PviParams::default();
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let result = pvi(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Not enough valid data"));
        }
    }

    #[test]
    fn test_pvi_calculation_logic() {
        let close_data = [100.0, 102.0, 101.0, 103.0, 103.0, 105.0];
        let volume_data = [500.0, 600.0, 500.0, 700.0, 680.0, 900.0];
        let params = PviParams {
            initial_value: Some(1000.0),
        };
        let input = PviInput::from_slices(&close_data, &volume_data, params);
        let output = pvi(&input).unwrap();
        assert_eq!(output.values.len(), close_data.len());
        assert!(output.values[0].abs() - 1000.0 < 1e-6);
        assert!((output.values[1] - 1000.0 - ((102.0 - 100.0) / 100.0) * 1000.0).abs() < 1e-6);
        assert!((output.values[2] - output.values[1]).abs() < 1e-6);
        assert!(output.values[3] > output.values[2]);
        assert!((output.values[4] - output.values[3]).abs() < 1e-6);
        assert!(output.values[5] > output.values[4]);
    }
}
