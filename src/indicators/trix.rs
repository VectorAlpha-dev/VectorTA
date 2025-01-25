/// # TRIX (Triple Exponential Average Oscillator)
///
/// TRIX is a momentum oscillator derived from a triple-smoothed Exponential Moving Average (EMA),
/// then taking the 1-day Rate-Of-Change (ROC) of that triple EMA (multiplied by 100).
/// This version forces an EMA warm-up that matches the standard TA-Lib approach:
/// The first EMA output at `first_valid_idx + period - 1` is the average of those `period` bars.
/// Each subsequent pass for EMA2 and EMA3 also uses that same initialization pattern, ensuring
/// accurate alignment for the final TRIX values.
///
/// ## Parameters
/// - **period**: The EMA window size. Defaults to 18.
///
/// ## Errors
/// - **EmptyData**: trix: Input data slice is empty.
/// - **InvalidPeriod**: trix: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: trix: Fewer than `3*(period - 1) + 1` valid data points remain
///   after the first valid index for triple-EMA + 1-bar ROC.
/// - **AllValuesNaN**: trix: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(TrixOutput)`** on success, matching the input length,
///   with `NaN` until triple-EMA is fully initialized plus 1 bar for the ROC.
/// - **`Err(TrixError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum TrixData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TrixOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TrixParams {
    pub period: Option<usize>,
}

impl Default for TrixParams {
    fn default() -> Self {
        Self { period: Some(18) }
    }
}

#[derive(Debug, Clone)]
pub struct TrixInput<'a> {
    pub data: TrixData<'a>,
    pub params: TrixParams,
}

impl<'a> TrixInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: TrixParams) -> Self {
        Self {
            data: TrixData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: TrixParams) -> Self {
        Self {
            data: TrixData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: TrixData::Candles {
                candles,
                source: "close",
            },
            params: TrixParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| TrixParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum TrixError {
    #[error("trix: Empty data provided.")]
    EmptyData,
    #[error("trix: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("trix: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("trix: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
fn compute_standard_ema(data: &[f64], period: usize, first_valid_idx: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    let alpha = 2.0 / (period as f64 + 1.0);

    let mut sum = 0.0;
    for &val in &data[first_valid_idx..(first_valid_idx + period)] {
        sum += val;
    }
    let initial_ema = sum / (period as f64);
    out[first_valid_idx + period - 1] = initial_ema;

    for i in (first_valid_idx + period)..data.len() {
        let prev = out[i - 1];
        if !prev.is_nan() && !data[i].is_nan() {
            out[i] = alpha * data[i] + (1.0 - alpha) * prev;
        }
    }

    out
}

#[inline]
fn compute_triple_ema(data: &[f64], period: usize, first_valid_idx: usize) -> Vec<f64> {
    let ema1 = compute_standard_ema(data, period, first_valid_idx);
    let ema2 = compute_standard_ema(&ema1, period, first_valid_idx + period - 1);
    compute_standard_ema(&ema2, period, first_valid_idx + 2 * (period - 1))
}

#[inline]
pub fn trix(input: &TrixInput) -> Result<TrixOutput, TrixError> {
    let data: &[f64] = match &input.data {
        TrixData::Candles { candles, source } => source_type(candles, source),
        TrixData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(TrixError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(TrixError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(TrixError::AllValuesNaN),
    };

    let needed = 3 * (period - 1) + 1;
    let valid_len = data.len() - first_valid_idx;
    if valid_len < needed {
        return Err(TrixError::NotEnoughValidData {
            needed,
            valid: valid_len,
        });
    }

    let triple_ema = compute_triple_ema(data, period, first_valid_idx);
    let mut trix_values = vec![f64::NAN; data.len()];
    let triple_ema_start = first_valid_idx + 3 * (period - 1);

    for i in (triple_ema_start + 1)..data.len() {
        let prev = triple_ema[i - 1];
        let curr = triple_ema[i];
        if !prev.is_nan() && !curr.is_nan() && prev != 0.0 {
            trix_values[i] = (curr / prev - 1.0) * 100.0;
        }
    }

    Ok(TrixOutput {
        values: trix_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_trix_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = TrixParams { period: None };
        let input_default = TrixInput::from_candles(&candles, "close", default_params);
        let output_default = trix(&input_default).expect("Failed TRIX with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = TrixParams { period: Some(14) };
        let input_period_14 = TrixInput::from_candles(&candles, "hl2", params_period_14);
        let output_period_14 =
            trix(&input_period_14).expect("Failed TRIX with period=14, source=hl2");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = TrixParams { period: Some(20) };
        let input_custom = TrixInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = trix(&input_custom).expect("Failed TRIX fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    #[ignore]
    fn test_trix_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = TrixParams { period: Some(18) };
        let input = TrixInput::from_candles(&candles, "close", params);
        let trix_result = trix(&input).expect("Failed to calculate TRIX");

        assert_eq!(
            trix_result.values.len(),
            close_prices.len(),
            "TRIX length mismatch"
        );

        let expected_last_five = [
            -16.03083789275206,
            -15.93477668222043,
            -15.794825711480387,
            -15.587573840557534,
            -15.416073398576424,
        ];
        assert!(trix_result.values.len() >= 5, "TRIX length too short");
        let start_index = trix_result.values.len() - 5;
        let result_last_five = &trix_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "TRIX mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_trix_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = TrixInput::with_default_candles(&candles);
        match input.data {
            TrixData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected TrixData::Candles variant"),
        }
    }

    #[test]
    fn test_trix_params_with_default() {
        let default_params = TrixParams::default();
        assert_eq!(
            default_params.period,
            Some(18),
            "Expected default TRIX period to be 18"
        );
    }

    #[test]
    fn test_trix_empty_data() {
        let params = TrixParams { period: Some(18) };
        let input_data: [f64; 0] = [];
        let input = TrixInput::from_slice(&input_data, params);
        let result = trix(&input);
        assert!(result.is_err(), "Expected error on empty data");
    }

    #[test]
    fn test_trix_zero_period() {
        let params = TrixParams { period: Some(0) };
        let input_data = [1.0, 2.0, 3.0];
        let input = TrixInput::from_slice(&input_data, params);
        let result = trix(&input);
        assert!(result.is_err(), "Expected error for zero period");
    }

    #[test]
    fn test_trix_period_exceeds_length() {
        let params = TrixParams { period: Some(100) };
        let input_data = [1.0, 2.0, 3.0];
        let input = TrixInput::from_slice(&input_data, params);
        let result = trix(&input);
        assert!(result.is_err(), "Expected error when period > data length");
    }

    #[test]
    fn test_trix_all_nan() {
        let params = TrixParams { period: Some(18) };
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let input = TrixInput::from_slice(&input_data, params);
        let result = trix(&input);
        assert!(result.is_err(), "Expected error when all data is NaN");
    }

    #[test]
    fn test_trix_not_enough_valid_data() {
        let params = TrixParams { period: Some(18) };
        let input_data = [f64::NAN; 30];
        let mut valid_data = input_data.clone();
        valid_data[25] = 50.0;
        let input = TrixInput::from_slice(&valid_data, params);
        let result = trix(&input);
        assert!(
            result.is_err(),
            "Expected error for insufficient valid data"
        );
    }

    #[test]
    fn test_trix_small_dataset() {
        let params = TrixParams { period: Some(18) };
        let input_data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let input = TrixInput::from_slice(&input_data, params);
        let result = trix(&input);
        assert!(result.is_err(), "Expected error on small dataset for TRIX");
    }

    #[test]
    fn test_trix_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = TrixParams { period: Some(10) };
        let input = TrixInput::from_candles(&candles, "close", params);
        let first_result = trix(&input).expect("First TRIX calculation failed");
        let second_input =
            TrixInput::from_slice(&first_result.values, TrixParams { period: Some(10) });
        let second_result = trix(&second_input).expect("Second TRIX calculation failed");
        assert_eq!(first_result.values.len(), second_result.values.len());
    }
}
