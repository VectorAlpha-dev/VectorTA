/// # Market Facilitation Index (MarketFI)
///
/// Market Facilitation Index (MarketFI) is calculated by taking the difference
/// between the high and low of each data point and dividing by the volume.
///
/// Unlike moving averages, MarketFI does not require a period parameter. It
/// provides insight into how efficiently the price moves relative to trading
/// activity.
///
/// ## Parameters
/// - *(none)*: No adjustable parameters are required; calculation is direct.
///
/// ## Errors
/// - **EmptyData**: marketfi: Input data slice is empty.
/// - **MismatchedDataLength**: marketfi: `high`, `low`, and `volume` slices do not have the same length.
/// - **AllValuesNaN**: marketfi: All input data values are `NaN`.
/// - **NotEnoughValidData**: marketfi: No calculable values remain after the first valid index.
/// - **ZeroOrNaNVolume**: marketfi: Volume is zero or NaN at a valid index.
///
/// ## Returns
/// - **`Ok(MarketFiOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the first valid index.
/// - **`Err(MarketFiError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MarketFiData<'a> {
    Candles {
        candles: &'a Candles,
        source_high: &'a str,
        source_low: &'a str,
        source_volume: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        volume: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MarketFiParams;

impl Default for MarketFiParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct MarketFiInput<'a> {
    pub data: MarketFiData<'a>,
    pub params: MarketFiParams,
}

impl<'a> MarketFiInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        source_high: &'a str,
        source_low: &'a str,
        source_volume: &'a str,
        params: MarketFiParams,
    ) -> Self {
        Self {
            data: MarketFiData::Candles {
                candles,
                source_high,
                source_low,
                source_volume,
            },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        volume: &'a [f64],
        params: MarketFiParams,
    ) -> Self {
        Self {
            data: MarketFiData::Slices { high, low, volume },
            params,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarketFiOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum MarketFiError {
    #[error("marketfi: Empty data provided.")]
    EmptyData,
    #[error("marketfi: Mismatched data length among high, low, and volume.")]
    MismatchedDataLength,
    #[error("marketfi: All values are NaN.")]
    AllValuesNaN,
    #[error("marketfi: Not enough valid data to calculate.")]
    NotEnoughValidData,
    #[error("marketfi: Zero or NaN volume at a valid index.")]
    ZeroOrNaNVolume,
}

#[inline]
pub fn marketfi(input: &MarketFiInput) -> Result<MarketFiOutput, MarketFiError> {
    let (high, low, volume) = match &input.data {
        MarketFiData::Candles {
            candles,
            source_high,
            source_low,
            source_volume,
        } => (
            source_type(candles, source_high),
            source_type(candles, source_low),
            source_type(candles, source_volume),
        ),
        MarketFiData::Slices { high, low, volume } => (*high, *low, *volume),
    };

    if high.is_empty() || low.is_empty() || volume.is_empty() {
        return Err(MarketFiError::EmptyData);
    }

    if high.len() != low.len() || low.len() != volume.len() {
        return Err(MarketFiError::MismatchedDataLength);
    }

    let mut output_values = vec![f64::NAN; high.len()];
    let first_valid_idx = match (0..high.len()).find(|&i| {
        let h = high[i];
        let l = low[i];
        let v = volume[i];
        !(h.is_nan() || l.is_nan() || v.is_nan())
    }) {
        Some(idx) => idx,
        None => return Err(MarketFiError::AllValuesNaN),
    };

    let mut valid_count = 0;
    for i in first_valid_idx..high.len() {
        let h = high[i];
        let l = low[i];
        let v = volume[i];
        if h.is_nan() || l.is_nan() || v.is_nan() {
            output_values[i] = f64::NAN;
            continue;
        }
        if v == 0.0 {
            output_values[i] = f64::NAN;
            continue;
        }
        output_values[i] = (h - l) / v;
        valid_count += 1;
    }

    if valid_count == 0 {
        return Err(MarketFiError::NotEnoughValidData);
    }

    if output_values[first_valid_idx..]
        .iter()
        .all(|&val| val.is_nan())
    {
        return Err(MarketFiError::ZeroOrNaNVolume);
    }

    Ok(MarketFiOutput {
        values: output_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_marketfi_empty_data() {
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let volume: [f64; 0] = [];

        let params = MarketFiParams;
        let input = MarketFiInput::from_slices(&high, &low, &volume, params);
        let result = marketfi(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Empty data"),
                "Expected 'EmptyData' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_marketfi_all_values_nan() {
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let volume = [f64::NAN, f64::NAN];

        let params = MarketFiParams;
        let input = MarketFiInput::from_slices(&high, &low, &volume, params);
        let result = marketfi(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("All values are NaN"),
                "Expected 'AllValuesNaN' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_marketfi_mismatched_length() {
        let high = [2.0, 3.0, 4.0];
        let low = [1.0, 2.0];
        let volume = [10.0, 10.0, 10.0];

        let params = MarketFiParams;
        let input = MarketFiInput::from_slices(&high, &low, &volume, params);
        let result = marketfi(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Mismatched data length"),
                "Expected 'MismatchedDataLength' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_marketfi_zero_volume() {
        let high = [2.0, 3.0, 4.0];
        let low = [1.0, 2.0, 3.0];
        let volume = [10.0, 0.0, 10.0];

        let params = MarketFiParams;
        let input = MarketFiInput::from_slices(&high, &low, &volume, params);
        let result = marketfi(&input).expect("Failed to calculate MarketFI");
        assert_eq!(result.values.len(), 3);
        assert!(result.values[1].is_nan());
    }

    #[test]
    fn test_marketfi_not_enough_valid_data() {
        let high = [f64::NAN, 3.0];
        let low = [f64::NAN, 1.0];
        let volume = [f64::NAN, 0.0];

        let params = MarketFiParams;
        let input = MarketFiInput::from_slices(&high, &low, &volume, params);
        let result = marketfi(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Not enough valid data")
                    || e.to_string().contains("Zero or NaN volume"),
                "Expected 'NotEnoughValidData' or 'ZeroOrNaNVolume', got: {}",
                e
            );
        }
    }

    #[test]
    #[ignore]
    fn test_marketfi_accuracy_with_csv_data() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = MarketFiInput::from_candles(
            &candles,
            "high",
            "low",
            "volume",
            MarketFiParams::default(),
        );

        let result = marketfi(&input).expect("Failed to calculate MarketFI");
        assert_eq!(result.values.len(), candles.close.len());

        let expected_last_five = [
            2.8460112192104607,
            3.020938522420525,
            3.0474861329079292,
            3.691017115591989,
            2.2478330402561397,
        ];
        assert!(result.values.len() >= 5);
        let start_index = result.values.len() - 5;
        let last_five = &result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-8,
                "MarketFI mismatch at index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
}
