/// # DX (Directional Movement Index)
///
/// The DX (Directional Movement Index) measures the trend strength by comparing
/// the smoothed +DI (Positive Directional Indicator) and -DI (Negative Directional
/// Indicator). The DX is the absolute difference between +DI and -DI, divided by
/// the sum of +DI and -DI, multiplied by 100. This indicator is based on the work
/// of J. Welles Wilder.
///
/// ## Parameters
/// - **period**: The time period for computing the DX (typically 14). Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: dx: Input data slices are empty.
/// - **SelectCandleFieldError**: dx: Failed to select candle field from `Candles`.
/// - **InvalidPeriod**: dx: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: dx: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: dx: All high, low, and close values are `NaN`.
///
/// ## Returns
/// - **`Ok(DxOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the DX window is filled.
/// - **`Err(DxError)`** otherwise.
use crate::utilities::data_loader::{read_candles_from_csv, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DxData<'a> {
    Candles {
        candles: &'a Candles,
    },
    HlcSlices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct DxOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DxParams {
    pub period: Option<usize>,
}

impl Default for DxParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct DxInput<'a> {
    pub data: DxData<'a>,
    pub params: DxParams,
}

impl<'a> DxInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: DxParams) -> Self {
        Self {
            data: DxData::Candles { candles },
            params,
        }
    }

    pub fn from_hlc_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: DxParams,
    ) -> Self {
        Self {
            data: DxData::HlcSlices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: DxData::Candles { candles },
            params: DxParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| DxParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum DxError {
    #[error("dx: Empty data provided for DX.")]
    EmptyData,
    #[error("dx: Could not select candle field: {0}")]
    SelectCandleFieldError(String),
    #[error("dx: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("dx: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("dx: All high, low, and close values are NaN.")]
    AllValuesNaN,
}

#[inline]
fn true_range(prev_close: f64, current_high: f64, current_low: f64) -> f64 {
    let tr1 = current_high - current_low;
    let tr2 = (current_high - prev_close).abs();
    let tr3 = (current_low - prev_close).abs();
    tr1.max(tr2).max(tr3)
}

#[inline]
pub fn dx(input: &DxInput) -> Result<DxOutput, DxError> {
    let (high, low, close) = match &input.data {
        DxData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|e| DxError::SelectCandleFieldError(e.to_string()))?;
            let low = candles
                .select_candle_field("low")
                .map_err(|e| DxError::SelectCandleFieldError(e.to_string()))?;
            let close = candles
                .select_candle_field("close")
                .map_err(|e| DxError::SelectCandleFieldError(e.to_string()))?;
            (high, low, close)
        }
        DxData::HlcSlices { high, low, close } => (*high, *low, *close),
    };
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(DxError::EmptyData);
    }
    let len = high.len().min(low.len()).min(close.len());
    let period = input.get_period();
    if period == 0 || period > len {
        return Err(DxError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let first_valid_idx =
        (0..len).find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan());
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(DxError::AllValuesNaN),
    };
    if (len - first_valid_idx) < period {
        return Err(DxError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }
    let mut dx_values = vec![f64::NAN; len];
    let mut prev_high = high[first_valid_idx];
    let mut prev_low = low[first_valid_idx];
    let mut prev_close = close[first_valid_idx];
    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;
    let mut tr_sum = 0.0;
    let mut initial_count = 0;
    for i in (first_valid_idx + 1)..len {
        if high[i].is_nan() || low[i].is_nan() || close[i].is_nan() {
            dx_values[i] = if i > 0 { dx_values[i - 1] } else { f64::NAN };
            prev_high = high[i];
            prev_low = low[i];
            prev_close = close[i];
            continue;
        }
        let up_move = high[i] - prev_high;
        let down_move = prev_low - low[i];
        let mut plus_dm = 0.0;
        let mut minus_dm = 0.0;
        if up_move > 0.0 && up_move > down_move {
            plus_dm = up_move;
        } else if down_move > 0.0 && down_move > up_move {
            minus_dm = down_move;
        }
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - prev_close).abs();
        let tr3 = (low[i] - prev_close).abs();
        let tr = tr1.max(tr2).max(tr3);
        if initial_count < (period - 1) {
            plus_dm_sum += plus_dm;
            minus_dm_sum += minus_dm;
            tr_sum += tr;
            initial_count += 1;
            if initial_count == (period - 1) {
                let plus_di = (plus_dm_sum / tr_sum) * 100.0;
                let minus_di = (minus_dm_sum / tr_sum) * 100.0;
                let sum_di = plus_di + minus_di;
                dx_values[i] = if sum_di != 0.0 {
                    100.0 * ((plus_di - minus_di).abs() / sum_di)
                } else {
                    0.0
                };
            }
        } else {
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period as f64) + plus_dm;
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period as f64) + minus_dm;
            tr_sum = tr_sum - (tr_sum / period as f64) + tr;
            let plus_di = if tr_sum != 0.0 {
                (plus_dm_sum / tr_sum) * 100.0
            } else {
                0.0
            };
            let minus_di = if tr_sum != 0.0 {
                (minus_dm_sum / tr_sum) * 100.0
            } else {
                0.0
            };
            let sum_di = plus_di + minus_di;
            dx_values[i] = if sum_di != 0.0 {
                100.0 * ((plus_di - minus_di).abs() / sum_di)
            } else {
                dx_values[i - 1]
            };
        }
        prev_high = high[i];
        prev_low = low[i];
        prev_close = close[i];
    }
    Ok(DxOutput { values: dx_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_dx_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = DxParams { period: None };
        let input_default = DxInput::from_candles(&candles, default_params);
        let output_default = dx(&input_default).expect("Failed DX with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_14 = DxParams { period: Some(14) };
        let input_14 = DxInput::from_candles(&candles, params_14);
        let output_14 = dx(&input_14).expect("Failed DX with period=14");
        assert_eq!(output_14.values.len(), candles.close.len());

        let params_20 = DxParams { period: Some(20) };
        let input_20 = DxInput::from_candles(&candles, params_20);
        let output_20 = dx(&input_20).expect("Failed DX with period=20");
        assert_eq!(output_20.values.len(), candles.close.len());
    }

    #[test]
    fn test_dx_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = DxParams { period: Some(14) };
        let input = DxInput::from_candles(&candles, params);
        let dx_result = dx(&input).expect("Failed to calculate DX");
        assert_eq!(dx_result.values.len(), candles.close.len());
        let expected_last_five_dx = [
            43.72121533411883,
            41.47251493226443,
            43.43041386436222,
            43.22673458811955,
            51.65514026197179,
        ];
        assert!(dx_result.values.len() >= 5);
        let start_index = dx_result.values.len() - 5;
        let last_five = &dx_result.values[start_index..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five_dx[i];
            assert!(
                (val - exp).abs() < 1e-4,
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }

    #[test]
    fn test_dx_empty_data() {
        let high: [f64; 0] = [];
        let low: [f64; 0] = [];
        let close: [f64; 0] = [];
        let params = DxParams { period: Some(14) };
        let input = DxInput::from_hlc_slices(&high, &low, &close, params);
        let result = dx(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Empty data"),
                "Expected EmptyData error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_dx_all_nan() {
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let close = [f64::NAN, f64::NAN, f64::NAN];
        let params = DxParams { period: Some(14) };
        let input = DxInput::from_hlc_slices(&high, &low, &close, params);
        let result = dx(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_dx_period_too_large() {
        let high = [3.0, 4.0];
        let low = [2.0, 3.0];
        let close = [2.5, 3.5];
        let params = DxParams { period: Some(14) };
        let input = DxInput::from_hlc_slices(&high, &low, &close, params);
        let result = dx(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected InvalidPeriod error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_dx_zero_period() {
        let high = [2.0, 2.5, 3.0];
        let low = [1.0, 1.2, 2.1];
        let close = [1.5, 2.3, 2.2];
        let params = DxParams { period: Some(0) };
        let input = DxInput::from_hlc_slices(&high, &low, &close, params);
        let result = dx(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_dx_small_data_set() {
        let high = [3.0];
        let low = [2.0];
        let close = [2.5];
        let params = DxParams { period: Some(14) };
        let input = DxInput::from_hlc_slices(&high, &low, &close, params);
        let result = dx(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_dx_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = DxInput::with_default_candles(&candles);
        let result = dx(&input).expect("Failed to calculate DX with default");
        assert_eq!(result.values.len(), candles.close.len());
    }

    #[test]
    fn test_dx_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = DxParams { period: Some(14) };
        let first_input = DxInput::from_candles(&candles, first_params);
        let first_result = dx(&first_input).expect("Failed to calculate first DX");
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = DxParams { period: Some(14) };
        let second_input = DxInput::from_hlc_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            second_params,
        );
        let second_result = dx(&second_input).expect("Failed to calculate second DX");
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 28, but found NaN at index {}",
                i
            );
        }
    }

    #[test]
    fn test_dx_nan_check_after_certain_index() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = DxParams { period: Some(14) };
        let input = DxInput::from_candles(&candles, params);
        let dx_result = dx(&input).expect("Failed to calculate DX");
        assert_eq!(dx_result.values.len(), candles.close.len());
        if dx_result.values.len() > 50 {
            for i in 50..dx_result.values.len() {
                assert!(
                    !dx_result.values[i].is_nan(),
                    "Expected no NaN after index 50, but found NaN at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_dx_params_with_default() {
        let default_params = DxParams::default();
        assert_eq!(default_params.period, Some(14));
    }

    #[test]
    fn test_dx_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = DxInput::with_default_candles(&candles);
        match input.data {
            DxData::Candles { .. } => {}
            _ => panic!("Expected DxData::Candles variant"),
        }
    }
}
