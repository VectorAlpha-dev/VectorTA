/// # On Balance Volume (OBV)
///
/// OBV is a cumulative indicator that measures buying and selling pressure by
/// adding or subtracting volume based on the price movement. If the closing price
/// is above the previous close, the current volume is added to the cumulative total;
/// if it is below, the current volume is subtracted.
///
/// This implementation follows TA-Lib's OBV logic:
/// 1. Set `prev_obv` to the volume of the first valid candle.
/// 2. Output that as the first OBV value.
/// 3. For each subsequent candle, if its close is higher than the previous close,
///    add that candle's volume; if it's lower, subtract that candle's volume.
///
/// ## Parameters
/// *(none)*
///
/// ## Errors
/// - **EmptyData**: obv: Input data slice is empty.
/// - **DataLengthMismatch**: obv: Mismatch in data lengths (close vs. volume).
/// - **AllValuesNaN**: obv: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(ObvOutput)`** on success, containing a `Vec<f64>` matching the input length.
/// - **`Err(ObvError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ObvData<'a> {
    Candles { candles: &'a Candles },
    Slices { close: &'a [f64], volume: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct ObvOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ObvParams;

impl Default for ObvParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct ObvInput<'a> {
    pub data: ObvData<'a>,
    pub params: ObvParams,
}

impl<'a> ObvInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: ObvParams) -> Self {
        Self {
            data: ObvData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(close: &'a [f64], volume: &'a [f64], params: ObvParams) -> Self {
        Self {
            data: ObvData::Slices { close, volume },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: ObvData::Candles { candles },
            params: ObvParams::default(),
        }
    }
}

#[derive(Debug, Error)]
pub enum ObvError {
    #[error("obv: Empty data provided.")]
    EmptyData,
    #[error("obv: Data length mismatch: close_len = {close_len}, volume_len = {volume_len}")]
    DataLengthMismatch { close_len: usize, volume_len: usize },
    #[error("obv: All values are NaN.")]
    AllValuesNaN,
}

impl From<Box<dyn std::error::Error>> for ObvError {
    fn from(error: Box<dyn std::error::Error>) -> Self {
        ObvError::EmptyData
    }
}

#[inline]
pub fn obv(input: &ObvInput) -> Result<ObvOutput, ObvError> {
    let (close, volume) = match &input.data {
        ObvData::Candles { candles } => {
            let close = candles.select_candle_field("close")?;
            let volume = candles.select_candle_field("volume")?;
            (close, volume)
        }
        ObvData::Slices { close, volume } => (*close, *volume),
    };

    if close.is_empty() || volume.is_empty() {
        return Err(ObvError::EmptyData);
    }

    if close.len() != volume.len() {
        return Err(ObvError::DataLengthMismatch {
            close_len: close.len(),
            volume_len: volume.len(),
        });
    }

    let first_valid_idx = match close
        .iter()
        .zip(volume.iter())
        .position(|(c, v)| !c.is_nan() && !v.is_nan())
    {
        Some(idx) => idx,
        None => return Err(ObvError::AllValuesNaN),
    };

    let mut obv_values = vec![f64::NAN; close.len()];
    let mut prev_obv = volume[first_valid_idx];
    let mut prev_close = close[first_valid_idx];
    obv_values[first_valid_idx] = prev_obv;

    for i in (first_valid_idx + 1)..close.len() {
        if close[i] > prev_close {
            prev_obv += volume[i];
        } else if close[i] < prev_close {
            prev_obv -= volume[i];
        }
        obv_values[i] = prev_obv;
        prev_close = close[i];
    }

    Ok(ObvOutput { values: obv_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_obv_empty_data() {
        let close: [f64; 0] = [];
        let volume: [f64; 0] = [];
        let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
        let result = obv(&input);
        assert!(result.is_err(), "Expected error for empty data");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Empty data"),
                "Expected 'Empty data' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_obv_data_length_mismatch() {
        let close = [1.0, 2.0, 3.0];
        let volume = [100.0, 200.0];
        let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
        let result = obv(&input);
        assert!(result.is_err(), "Expected error for mismatched data length");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Data length mismatch"),
                "Expected 'Data length mismatch' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_obv_all_nan() {
        let close = [f64::NAN, f64::NAN];
        let volume = [f64::NAN, f64::NAN];
        let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
        let result = obv(&input);
        assert!(result.is_err(), "Expected error for all NaN data");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("All values are NaN"),
                "Expected 'All values are NaN' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_obv_basic_slices() {
        let close = [10.0, 11.0, 11.0, 12.0, 10.0];
        let volume = [100.0, 200.0, 300.0, 400.0, 500.0];
        let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
        let output = obv(&input).expect("Failed to calculate OBV");

        assert_eq!(
            output.values.len(),
            close.len(),
            "OBV output length mismatch with input length"
        );

        let expected = [100.0, 300.0, 300.0, 700.0, 200.0];
        for (i, &val) in output.values.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-10,
                "OBV mismatch at index {}: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }
    }

    #[test]
    fn test_obv_replicate_talib() {
        let close = [100.0, 101.0, 99.0, 99.0, 102.0];
        let volume = [500.0, 100.0, 200.0, 300.0, 400.0];
        let input = ObvInput::from_slices(&close, &volume, ObvParams::default());
        let output = obv(&input).expect("Failed to calculate OBV");

        let expected = [500.0, 600.0, 400.0, 400.0, 800.0];
        for (i, &val) in output.values.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-10,
                "OBV mismatch at index {}: expected {}, got {}",
                i,
                expected[i],
                val
            );
        }
    }

    #[test]
    #[ignore]
    fn test_obv_csv_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let volume_data = candles
            .select_candle_field("volume")
            .expect("Failed to extract volume data");

        let input = ObvInput::from_candles(&candles, ObvParams::default());
        let obv_result = obv(&input).expect("Failed to calculate OBV");

        assert_eq!(
            obv_result.values.len(),
            close_prices.len(),
            "OBV length mismatch"
        );

        let last_five_expected = [
            -364431.9459467806,
            -364538.2043157006,
            -364660.27213940065,
            -364571.6786732206,
            -364988.52457913064,
        ];
        assert!(obv_result.values.len() >= 5, "OBV length too short");
        let start_idx = obv_result.values.len() - 5;
        let result_tail = &obv_result.values[start_idx..];
        for (i, &val) in result_tail.iter().enumerate() {
            let exp_val = last_five_expected[i];
            let diff = (val - exp_val).abs();
            assert!(
                diff < 1e-6,
                "OBV mismatch at tail index {}: expected {}, got {}",
                i,
                exp_val,
                val
            );
        }

        let default_input = ObvInput::with_default_candles(&candles);
        let default_obv_result =
            obv(&default_input).expect("Failed to calculate OBV with default candles");
        assert_eq!(
            default_obv_result.values.len(),
            close_prices.len(),
            "OBV default length mismatch"
        );

        let slice_input = ObvInput::from_slices(&close_prices, &volume_data, ObvParams::default());
        let slice_obv_result = obv(&slice_input).expect("Failed to calculate OBV with slices");
        assert_eq!(
            slice_obv_result.values.len(),
            close_prices.len(),
            "OBV slice-based length mismatch"
        );

        for ((idx, &candles_val), &slices_val) in obv_result
            .values
            .iter()
            .enumerate()
            .zip(slice_obv_result.values.iter())
        {
            let difference = (candles_val - slices_val).abs();
            assert!(
                difference < f64::EPSILON,
                "Mismatch at {}: candles_val={}, slices_val={}",
                idx,
                candles_val,
                slices_val
            );
        }
    }
}
