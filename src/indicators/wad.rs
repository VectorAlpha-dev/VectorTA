/// # Williams Accumulation/Distribution (WAD)
///
/// Williams Accumulation/Distribution (WAD) is a cumulative measure of buying and selling pressure
/// based on the relationship between the current close, the previous close, and the high and low price
/// ranges. It helps to identify potential divergences and confirm trend strength.
///
/// ## Parameters
/// - None (WAD does not use a period).
///
/// ## Errors
/// - **EmptyData**: wad: Input data slice is empty.
/// - **AllValuesNaN**: wad: All input data values for high, low, or close are `NaN`.
///
/// ## Returns
/// - **`Ok(WadOutput)`** on success, containing a `Vec<f64>` matching the input length.
/// - **`Err(WadError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum WadData<'a> {
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
pub struct WadOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WadParams;

impl Default for WadParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct WadInput<'a> {
    pub data: WadData<'a>,
    pub params: WadParams,
}

impl<'a> WadInput<'a> {
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: WadData::Candles { candles },
            params: WadParams::default(),
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> Self {
        Self {
            data: WadData::Slices { high, low, close },
            params: WadParams::default(),
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: WadData::Candles { candles },
            params: WadParams::default(),
        }
    }
}

#[derive(Debug, Error)]
pub enum WadError {
    #[error("wad: Empty data provided.")]
    EmptyData,
    #[error("wad: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn wad(input: &WadInput) -> Result<WadOutput, WadError> {
    let (high, low, close) = match &input.data {
        WadData::Candles { candles } => {
            let high = candles.select_candle_field("high").unwrap();
            let low = candles.select_candle_field("low").unwrap();
            let close = candles.select_candle_field("close").unwrap();
            (high, low, close)
        }
        WadData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WadError::EmptyData);
    }

    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(WadError::EmptyData);
    }

    if high.iter().all(|x| x.is_nan())
        || low.iter().all(|x| x.is_nan())
        || close.iter().all(|x| x.is_nan())
    {
        return Err(WadError::AllValuesNaN);
    }

    let mut ad = vec![0.0; len];
    ad[0] = 0.0;

    for i in 1..len {
        if close[i] > close[i - 1] {
            ad[i] = close[i] - low[i].min(close[i - 1]);
        } else if close[i] < close[i - 1] {
            ad[i] = close[i] - high[i].max(close[i - 1]);
        } else {
            ad[i] = 0.0;
        }
    }

    let mut wad_values = vec![0.0; len];
    let mut running_sum = 0.0;
    for i in 0..len {
        running_sum += ad[i];
        wad_values[i] = running_sum;
    }

    Ok(WadOutput { values: wad_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_wad_empty_data() {
        let input = WadInput::from_slices(&[], &[], &[]);
        let result = wad(&input);
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), WadError::EmptyData));
    }

    #[test]
    fn test_wad_all_values_nan() {
        let nan_slice = [f64::NAN, f64::NAN, f64::NAN];
        let input = WadInput::from_slices(&nan_slice, &nan_slice, &nan_slice);
        let result = wad(&input);
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), WadError::AllValuesNaN));
    }

    #[test]
    fn test_wad_basic_slice() {
        let high = [10.0, 11.0, 11.0, 12.0];
        let low = [9.0, 9.0, 10.0, 10.0];
        let close = [9.5, 10.5, 10.5, 11.5];
        let input = WadInput::from_slices(&high, &low, &close);
        let output = wad(&input).expect("Failed to calculate WAD");
        assert_eq!(output.values.len(), 4);
        assert!((output.values[0] - 0.0).abs() < 1e-10);
        assert!((output.values[1] - 1.5).abs() < 1e-10);
        assert!((output.values[2] - 1.5).abs() < 1e-10);
        assert!((output.values[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    #[ignore]
    fn test_wad_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = WadInput::from_candles(&candles);
        let output = wad(&input).expect("Failed to calculate WAD");
        assert_eq!(output.values.len(), candles.close.len());

        let expected_last_five_wad = [
            166650.5139999995,
            166851.5139999995,
            166729.5139999995,
            166458.5139999995,
            167314.5139999995,
        ];

        let start_idx = output.values.len() - 5;
        let slice = &output.values[start_idx..];
        for (i, &val) in slice.iter().enumerate() {
            let exp = expected_last_five_wad[i];
            assert!(
                (val - exp).abs() < 1e-4,
                "WAD mismatch at index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
}
