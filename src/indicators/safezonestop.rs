/// # SafeZoneStop
///
/// The SafeZoneStop indicator attempts to place stop-loss levels based on
/// the concept of directional movement and volatility, using TA-Lib's
/// `MINUS_DM` or `PLUS_DM` logic under the hood. Depending on the `direction`
/// ("long" or "short"), it calculates a shifted low or high (`last_low` or
/// `last_high`) minus or plus a multiplier (`mult`) times the Wilder-smoothed
/// directional movement measure over a specified `period`. A final rolling
/// maximum or minimum (over `max_lookback` bars) is then taken to produce the
/// stop values.
///
/// This implementation closely replicates the following Python pseudocode:
///
/// ```python
/// if direction == "long":
///     res = talib.MAX(last_low - mult * talib.MINUS_DM(high, low, timeperiod=period), max_lookback)
/// else:
///     res = talib.MIN(last_high + mult * talib.PLUS_DM(high, low, timeperiod=period), max_lookback)
/// ```
///
/// ## Parameters
/// - **period**: The time period used for calculating `MINUS_DM` or `PLUS_DM` (Wilder's smoothing). Defaults to 22.
/// - **mult**: Multiplier for the directional movement measure. Defaults to 2.5.
/// - **max_lookback**: Rolling window size for the final max or min operation. Defaults to 3.
/// - **direction**: `"long"` or `"short"`, determines which DM measure is used and the sign of the final computation. Defaults to `"long"`.
///
/// ## Errors
/// - **EmptyData**: safezonestop: Input data slice is empty.
/// - **InvalidPeriod**: safezonestop: `period` is zero or exceeds data length.
/// - **AllValuesNaN**: safezonestop: All input data values are `NaN`.
/// - **MismatchedLengths**: safezonestop: Input slices have different lengths.
/// - **InvalidDirection**: safezonestop: Direction must be either `"long"` or `"short"`.
///
/// ## Returns
/// - **`Ok(SafeZoneStopOutput)`** on success, containing a `Vec<f64>` of length equal to the input,
///   with leading `NaN`s in places where not enough data is available.
/// - **`Err(SafeZoneStopError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum SafeZoneStopData<'a> {
    Candles {
        candles: &'a Candles,
        direction: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        direction: &'a str,
    },
}

#[derive(Debug, Clone)]
pub struct SafeZoneStopOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SafeZoneStopParams {
    pub period: Option<usize>,
    pub mult: Option<f64>,
    pub max_lookback: Option<usize>,
}

impl Default for SafeZoneStopParams {
    fn default() -> Self {
        Self {
            period: Some(22),
            mult: Some(2.5),
            max_lookback: Some(3),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SafeZoneStopInput<'a> {
    pub data: SafeZoneStopData<'a>,
    pub params: SafeZoneStopParams,
}

impl<'a> SafeZoneStopInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        direction: &'a str,
        params: SafeZoneStopParams,
    ) -> Self {
        Self {
            data: SafeZoneStopData::Candles { candles, direction },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        direction: &'a str,
        params: SafeZoneStopParams,
    ) -> Self {
        Self {
            data: SafeZoneStopData::Slices {
                high,
                low,
                direction,
            },
            params,
        }
    }

    pub fn with_default_candles_long(candles: &'a Candles) -> Self {
        Self {
            data: SafeZoneStopData::Candles {
                candles,
                direction: "long",
            },
            params: SafeZoneStopParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SafeZoneStopParams::default().period.unwrap())
    }

    pub fn get_mult(&self) -> f64 {
        self.params
            .mult
            .unwrap_or_else(|| SafeZoneStopParams::default().mult.unwrap())
    }

    pub fn get_max_lookback(&self) -> usize {
        self.params
            .max_lookback
            .unwrap_or_else(|| SafeZoneStopParams::default().max_lookback.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum SafeZoneStopError {
    #[error("safezonestop: Empty data provided.")]
    EmptyData,
    #[error("safezonestop: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("safezonestop: All values are NaN.")]
    AllValuesNaN,
    #[error("safezonestop: Input slices have mismatched lengths.")]
    MismatchedLengths,
    #[error("safezonestop: Invalid direction. Must be 'long' or 'short'.")]
    InvalidDirection,
}

#[inline]
fn shift_left(input: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; input.len()];
    for i in 1..input.len() {
        out[i] = input[i - 1];
    }
    out
}

fn minus_dm_talib(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let mut raw = vec![0.0; high.len()];
    for i in 1..high.len() {
        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];
        if down_move > up_move && down_move > 0.0 {
            raw[i] = down_move;
        }
    }
    let mut output = vec![f64::NAN; high.len()];

    if period < high.len() {
        let mut sum = 0.0;
        for i in 1..=period {
            sum += raw[i];
        }
        output[period] = sum;

        for i in (period + 1)..high.len() {
            output[i] = output[i - 1] - (output[i - 1] / (period as f64)) + raw[i];
        }
    }
    output
}

fn plus_dm_talib(high: &[f64], low: &[f64], period: usize) -> Vec<f64> {
    let mut raw = vec![0.0; high.len()];
    for i in 1..high.len() {
        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];
        if up_move > down_move && up_move > 0.0 {
            raw[i] = up_move;
        }
    }
    let mut output = vec![f64::NAN; high.len()];

    if period < high.len() {
        let mut sum = 0.0;
        for i in 1..=period {
            sum += raw[i];
        }
        output[period] = sum;

        for i in (period + 1)..high.len() {
            output[i] = output[i - 1] - (output[i - 1] / (period as f64)) + raw[i];
        }
    }
    output
}

#[inline]
pub fn safezonestop(input: &SafeZoneStopInput) -> Result<SafeZoneStopOutput, SafeZoneStopError> {
    let (high, low, direction) = match &input.data {
        SafeZoneStopData::Candles { candles, direction } => {
            let h = source_type(candles, "high");
            let l = source_type(candles, "low");
            if h.is_empty() || l.is_empty() {
                return Err(SafeZoneStopError::EmptyData);
            }
            (h, l, *direction)
        }
        SafeZoneStopData::Slices {
            high,
            low,
            direction,
        } => {
            if high.len() != low.len() {
                return Err(SafeZoneStopError::MismatchedLengths);
            }
            if high.is_empty() || low.is_empty() {
                return Err(SafeZoneStopError::EmptyData);
            }
            (*high, *low, *direction)
        }
    };

    if direction != "long" && direction != "short" {
        return Err(SafeZoneStopError::InvalidDirection);
    }

    let period = input.get_period();
    let mult = input.get_mult();
    let max_lookback = input.get_max_lookback();
    let len = high.len();

    if period == 0 || period > len {
        return Err(SafeZoneStopError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let has_any_non_nan = high.iter().any(|&v| !v.is_nan()) || low.iter().any(|&v| !v.is_nan());
    if !has_any_non_nan {
        return Err(SafeZoneStopError::AllValuesNaN);
    }

    let last_low = shift_left(&low);
    let last_high = shift_left(&high);

    let minus_dm = minus_dm_talib(high, low, period);
    let plus_dm = plus_dm_talib(high, low, period);

    let mut intermediate = vec![f64::NAN; len];
    if direction == "long" {
        for i in 0..len {
            if !minus_dm[i].is_nan() && !last_low[i].is_nan() {
                intermediate[i] = last_low[i] - mult * minus_dm[i];
            }
        }
    } else {
        for i in 0..len {
            if !plus_dm[i].is_nan() && !last_high[i].is_nan() {
                intermediate[i] = last_high[i] + mult * plus_dm[i];
            }
        }
    }

    let mut output = vec![f64::NAN; len];
    for i in 0..len {
        if i + 1 < max_lookback {
            continue;
        }
        let start_idx = i + 1 - max_lookback;
        if direction == "long" {
            let mut mx = f64::NAN;
            for j in start_idx..=i {
                let val = intermediate[j];
                if val.is_nan() {
                    continue;
                }
                if mx.is_nan() || val > mx {
                    mx = val;
                }
            }
            output[i] = mx;
        } else {
            let mut mn = f64::NAN;
            for j in start_idx..=i {
                let val = intermediate[j];
                if val.is_nan() {
                    continue;
                }
                if mn.is_nan() || val < mn {
                    mn = val;
                }
            }
            output[i] = mn;
        }
    }

    Ok(SafeZoneStopOutput { values: output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_safezonestop_default_long() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = SafeZoneStopInput::with_default_candles_long(&candles);
        let output = safezonestop(&input).expect("Failed SafeZoneStop calculation");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_safezonestop_partial_params_short() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = SafeZoneStopParams {
            period: Some(14),
            mult: None,
            max_lookback: None,
        };
        let input = SafeZoneStopInput::from_candles(&candles, "short", params);
        let output = safezonestop(&input).expect("Failed SafeZoneStop short");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_safezonestop_last_five_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = SafeZoneStopParams {
            period: Some(22),
            mult: Some(2.5),
            max_lookback: Some(3),
        };
        let input = SafeZoneStopInput::from_candles(&candles, "long", params);
        let output = safezonestop(&input).expect("Failed SafeZoneStop long");
        assert_eq!(output.values.len(), candles.close.len());
        if output.values.len() >= 5 {
            let last_five = &output.values[output.values.len() - 5..];
            let expected = [
                45331.180007991,
                45712.94455308232,
                46019.94707339676,
                46461.767660969635,
                46461.767660969635,
            ];
            for (i, &val) in last_five.iter().enumerate() {
                let diff = (val - expected[i]).abs();
                assert!(
                    diff < 1e-4,
                    "Mismatch at index {}: got {}, expected {}",
                    i,
                    val,
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn test_safezonestop_with_zero_period() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = SafeZoneStopParams {
            period: Some(0),
            mult: Some(2.5),
            max_lookback: Some(3),
        };
        let input = SafeZoneStopInput::from_slices(&high, &low, "long", params);
        let result = safezonestop(&input);
        assert!(result.is_err(), "Expected an error for zero period");
    }

    #[test]
    fn test_safezonestop_with_invalid_direction() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = SafeZoneStopParams {
            period: Some(22),
            mult: Some(2.5),
            max_lookback: Some(3),
        };
        let input = SafeZoneStopInput::from_slices(&high, &low, "nonsense", params);
        let result = safezonestop(&input);
        assert!(result.is_err(), "Expected an error for invalid direction");
    }

    #[test]
    fn test_safezonestop_with_mismatched_lengths() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0];
        let params = SafeZoneStopParams::default();
        let input = SafeZoneStopInput::from_slices(&high, &low, "long", params);
        let result = safezonestop(&input);
        assert!(result.is_err(), "Expected an error for mismatched lengths");
    }
}
