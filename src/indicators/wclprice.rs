/// # Weighted Close Price (WCLPRICE)
///
/// The Weighted Close Price is calculated for each data point as:
/// `(high + low + 2*close) / 4`.
///
/// This indicator takes one or more candle fields (or slices of f64) and returns
/// a new set of values corresponding to each input element. The result will be
/// `NaN` for indices preceding the first valid (non-`NaN`) set of values for high,
/// low, and close.
///
/// ## Errors
/// - **EmptyData**: wclprice: Input data is empty.
/// - **AllValuesNaN**: wclprice: All input data values (in any required field) are `NaN`.
///
/// ## Returns
/// - **`Ok(WclpriceOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the first valid index.
/// - **`Err(WclpriceError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum WclpriceData<'a> {
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
pub struct WclpriceOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WclpriceParams;

impl Default for WclpriceParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct WclpriceInput<'a> {
    pub data: WclpriceData<'a>,
    pub params: WclpriceParams,
}

impl<'a> WclpriceInput<'a> {
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: WclpriceData::Candles { candles },
            params: WclpriceParams::default(),
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> Self {
        Self {
            data: WclpriceData::Slices { high, low, close },
            params: WclpriceParams::default(),
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: WclpriceData::Candles { candles },
            params: WclpriceParams::default(),
        }
    }
}

#[derive(Debug, Error)]
pub enum WclpriceError {
    #[error("wclprice: Empty data provided.")]
    EmptyData,
    #[error("wclprice: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn wclprice(input: &WclpriceInput) -> Result<WclpriceOutput, WclpriceError> {
    let (high, low, close) = match &input.data {
        WclpriceData::Candles { candles } => {
            let high = candles.select_candle_field("high").unwrap();
            let low = candles.select_candle_field("low").unwrap();
            let close = candles.select_candle_field("close").unwrap();
            (high, low, close)
        }
        WclpriceData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(WclpriceError::EmptyData);
    }

    let len = high.len().min(low.len()).min(close.len());
    if len == 0 {
        return Err(WclpriceError::EmptyData);
    }

    let mut out_values = vec![f64::NAN; len];
    let first_valid_idx = (0..len).find(|&i| {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        !h.is_nan() && !l.is_nan() && !c.is_nan()
    });

    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(WclpriceError::AllValuesNaN),
    };

    for i in first_valid_idx..len {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        if h.is_nan() || l.is_nan() || c.is_nan() {
            out_values[i] = f64::NAN;
        } else {
            out_values[i] = (h + l + 2.0 * c) / 4.0;
        }
    }

    Ok(WclpriceOutput { values: out_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_wclprice_with_slices() {
        let high = vec![59230.0, 59220.0, 59077.0, 59160.0, 58717.0];
        let low = vec![59222.0, 59211.0, 59077.0, 59143.0, 58708.0];
        let close = vec![59225.0, 59210.0, 59080.0, 59150.0, 58710.0];

        let input = WclpriceInput::from_slices(&high, &low, &close);
        let output = wclprice(&input).expect("Failed to calculate WCLPRICE");

        let expected = vec![59225.5, 59212.75, 59078.5, 59150.75, 58711.25];

        assert_eq!(output.values.len(), expected.len());
        for (i, &val) in output.values.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-2,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }
    }

    #[test]
    fn test_wclprice_with_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = WclpriceInput::from_candles(&candles);
        let output = wclprice(&input).expect("Failed to calculate WCLPRICE with candles");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_wclprice_empty_data() {
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let close: [f64; 0] = [];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let result = wclprice(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Empty data"));
        }
    }

    #[test]
    fn test_wclprice_all_nan() {
        let high = vec![f64::NAN, f64::NAN];
        let low = vec![f64::NAN, f64::NAN];
        let close = vec![f64::NAN, f64::NAN];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let result = wclprice(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
    }

    #[test]
    fn test_wclprice_partial_nan() {
        let high = vec![f64::NAN, 59000.0];
        let low = vec![f64::NAN, 58950.0];
        let close = vec![f64::NAN, 58975.0];
        let input = WclpriceInput::from_slices(&high, &low, &close);
        let output = wclprice(&input).expect("Failed to calculate WCLPRICE with partial NaN");
        assert!(output.values[0].is_nan());
        assert_eq!(output.values[1], (59000.0 + 58950.0 + 2.0 * 58975.0) / 4.0);
    }
}
