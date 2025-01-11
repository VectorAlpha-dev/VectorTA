use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::utilities::data_loader::{source_type, Candles};
/// # Kaufmanstop
///
/// Perry Kaufman's Stop indicator computes an adaptive price stop based on the
/// average true range of price (here simplified as the difference between high
/// and low). By applying a moving average (of type `ma_type`) on the range and
/// multiplying it by `mult`, it places a trailing stop either above or below
/// the price, depending on the direction.
///
/// ## Parameters
/// - **period**: The window size for the range average (number of data points). Defaults to 22.
/// - **mult**: A multiplier for the averaged range. Defaults to 2.0.
/// - **direction**: Whether to calculate a stop for "long" (below price) or "short" (above price). Defaults to "long".
/// - **ma_type**: The type of moving average to use. Defaults to "sma".
///
/// ## Errors
/// - **EmptyData**: kaufmanstop: Input data slice or fields are empty.
/// - **InvalidPeriod**: kaufmanstop: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: kaufmanstop: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: kaufmanstop: All relevant input values (`high` or `low`) are `NaN`.
///
/// ## Returns
/// - **`Ok(KaufmanstopOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the moving average window is filled.
/// - **`Err(KaufmanstopError)`** otherwise.
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum KaufmanstopData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct KaufmanstopOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct KaufmanstopParams {
    pub period: Option<usize>,
    pub mult: Option<f64>,
    pub direction: Option<String>,
    pub ma_type: Option<String>,
}

impl Default for KaufmanstopParams {
    fn default() -> Self {
        Self {
            period: Some(22),
            mult: Some(2.0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KaufmanstopInput<'a> {
    pub data: KaufmanstopData<'a>,
    pub params: KaufmanstopParams,
}

impl<'a> KaufmanstopInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: KaufmanstopParams) -> Self {
        Self {
            data: KaufmanstopData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: KaufmanstopParams) -> Self {
        Self {
            data: KaufmanstopData::Slices { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: KaufmanstopData::Candles { candles },
            params: KaufmanstopParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| KaufmanstopParams::default().period.unwrap())
    }

    pub fn get_mult(&self) -> f64 {
        self.params
            .mult
            .unwrap_or_else(|| KaufmanstopParams::default().mult.unwrap())
    }

    pub fn get_direction(&self) -> String {
        self.params
            .direction
            .clone()
            .unwrap_or_else(|| KaufmanstopParams::default().direction.unwrap())
    }

    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .clone()
            .unwrap_or_else(|| KaufmanstopParams::default().ma_type.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum KaufmanstopError {
    #[error("kaufmanstop: Empty data provided.")]
    EmptyData,
    #[error("kaufmanstop: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("kaufmanstop: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("kaufmanstop: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn kaufmanstop(input: &KaufmanstopInput) -> Result<KaufmanstopOutput, KaufmanstopError> {
    let (high, low) = match &input.data {
        KaufmanstopData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| KaufmanstopError::EmptyData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| KaufmanstopError::EmptyData)?;
            (high, low)
        }
        KaufmanstopData::Slices { high, low } => {
            if high.is_empty() || low.is_empty() {
                return Err(KaufmanstopError::EmptyData);
            }
            (*high, *low)
        }
    };

    if high.is_empty() || low.is_empty() {
        return Err(KaufmanstopError::EmptyData);
    }

    let period = input.get_period();
    let mult = input.get_mult();
    let direction = input.get_direction();
    let ma_type = input.get_ma_type();

    if period == 0 || period > high.len() || period > low.len() {
        return Err(KaufmanstopError::InvalidPeriod {
            period,
            data_len: high.len().min(low.len()),
        });
    }

    let mut first_valid_idx = None;
    for (i, (&h, &l)) in high.iter().zip(low.iter()).enumerate() {
        if !h.is_nan() && !l.is_nan() {
            first_valid_idx = Some(i);
            break;
        }
    }
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(KaufmanstopError::AllValuesNaN),
    };

    if (high.len() - first_valid_idx) < period || (low.len() - first_valid_idx) < period {
        return Err(KaufmanstopError::NotEnoughValidData {
            needed: period,
            valid: high.len() - first_valid_idx,
        });
    }

    let mut hl_diff = vec![f64::NAN; high.len()];
    for i in first_valid_idx..high.len() {
        if high[i].is_nan() || low[i].is_nan() {
            hl_diff[i] = f64::NAN;
        } else {
            hl_diff[i] = high[i] - low[i];
        }
    }

    let ma_input = MaData::Slice(&hl_diff[first_valid_idx..]);
    let hl_diff_ma = ma(&ma_type, ma_input, period).map_err(|_| KaufmanstopError::AllValuesNaN)?;

    let mut kaufmanstop_values = vec![f64::NAN; high.len()];
    for (i, &val) in hl_diff_ma.iter().enumerate() {
        let actual_idx = first_valid_idx + i;
        if actual_idx < high.len() {
            if direction.eq_ignore_ascii_case("long") {
                kaufmanstop_values[actual_idx] = low[actual_idx] - val * mult;
            } else {
                kaufmanstop_values[actual_idx] = high[actual_idx] + val * mult;
            }
        }
    }

    Ok(KaufmanstopOutput {
        values: kaufmanstop_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_kaufmanstop_default_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = KaufmanstopInput::with_default_candles(&candles);
        let output = kaufmanstop(&input).expect("Failed Kaufmanstop with default params");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_kaufmanstop_custom_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let custom_params = KaufmanstopParams {
            period: Some(10),
            mult: Some(2.5),
            direction: Some("short".to_string()),
            ma_type: Some("ema".to_string()),
        };
        let input = KaufmanstopInput::from_candles(&candles, custom_params);
        let output = kaufmanstop(&input).expect("Failed Kaufmanstop with custom params");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_kaufmanstop_accuracy_with_last_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = KaufmanstopParams {
            period: Some(22),
            mult: Some(2.0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = KaufmanstopInput::from_candles(&candles, params);
        let result = kaufmanstop(&input).expect("Failed Kaufmanstop known-values test");
        assert_eq!(result.values.len(), candles.high.len());

        let last_five = &result.values[result.values.len().saturating_sub(5)..];
        let expected = [
            56711.545454545456,
            57132.72727272727,
            57015.72727272727,
            57137.18181818182,
            56516.09090909091,
        ];
        assert_eq!(last_five.len(), expected.len());
        for (idx, &val) in last_five.iter().enumerate() {
            assert!(
                (val - expected[idx]).abs() < 1e-1,
                "Mismatch at index {}: expected {}, got {}",
                idx,
                expected[idx],
                val
            );
        }
    }

    #[test]
    fn test_kaufmanstop_period_too_big() {
        let high = vec![10.0, 20.0, 30.0];
        let low = vec![5.0, 15.0, 25.0];
        let params = KaufmanstopParams {
            period: Some(10),
            mult: Some(2.0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = KaufmanstopInput::from_slices(&high, &low, params);
        let result = kaufmanstop(&input);
        assert!(result.is_err(), "Expected period too big error");
    }

    #[test]
    fn test_kaufmanstop_zero_period() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let mut params = KaufmanstopParams::default();
        params.period = Some(0);
        let input = KaufmanstopInput::from_candles(&candles, params);
        let result = kaufmanstop(&input);
        assert!(result.is_err(), "Expected zero period error");
    }

    #[test]
    fn test_kaufmanstop_not_enough_valid_data() {
        let high = vec![50.0, 60.0];
        let low = vec![49.0, 59.0];
        let params = KaufmanstopParams {
            period: Some(3),
            mult: Some(2.0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = KaufmanstopInput::from_slices(&high, &low, params);
        let result = kaufmanstop(&input);
        assert!(result.is_err(), "Expected not enough valid data error");
    }

    #[test]
    fn test_kaufmanstop_slice_vs_candle_equivalence() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let high = candles
            .select_candle_field("high")
            .expect("Failed to get high");
        let low = candles
            .select_candle_field("low")
            .expect("Failed to get low");

        let cdl_input = KaufmanstopInput::with_default_candles(&candles);
        let cdl_result = kaufmanstop(&cdl_input).expect("Failed Kaufmanstop from candles");
        let slice_params = KaufmanstopParams::default();
        let slice_input = KaufmanstopInput::from_slices(&high, &low, slice_params);
        let slice_result = kaufmanstop(&slice_input).expect("Failed Kaufmanstop from slices");
        assert_eq!(cdl_result.values.len(), slice_result.values.len());
    }
}
