/// # Heikin Ashi Candles
///
/// Heikin Ashi Candles reduce noise from standard candlestick charts by applying
/// an averaging formula to both the current candle and the previous Heikin Ashi
/// candle. This helps in identifying and visualizing trends more clearly.
///
/// ## Parameters
/// - *(None)*: No user-configurable parameters. The transformation applies to
///   the full open, high, low, and close data sets.
///
/// ## Errors
/// - **EmptyData**: heikin_ashi_candles: Input data slice(s) are empty.
/// - **AllValuesNaN**: heikin_ashi_candles: All input data values are `NaN`.
/// - **NotEnoughValidData**: heikin_ashi_candles: Fewer than 2 valid (non-`NaN`) data points remain
///   after the first valid index.
///
/// ## Returns
/// - **`Ok(HeikinAshiOutput)`** on success, containing `Vec<f64>` for open,
///   high, low, and close matching the input length, with leading `NaN`s
///   until the first valid index.
/// - **`Err(HeikinAshiError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum HeikinAshiData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct HeikinAshiOutput {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HeikinAshiParams;

impl Default for HeikinAshiParams {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct HeikinAshiInput<'a> {
    pub data: HeikinAshiData<'a>,
    pub params: HeikinAshiParams,
}

impl<'a> HeikinAshiInput<'a> {
    pub fn from_candles(candles: &'a Candles) -> Self {
        Self {
            data: HeikinAshiData::Candles { candles },
            params: HeikinAshiParams::default(),
        }
    }

    pub fn from_slices(open: &'a [f64], high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> Self {
        Self {
            data: HeikinAshiData::Slices {
                open,
                high,
                low,
                close,
            },
            params: HeikinAshiParams::default(),
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: HeikinAshiData::Candles { candles },
            params: HeikinAshiParams::default(),
        }
    }
}

#[derive(Debug, Error)]
pub enum HeikinAshiError {
    #[error("heikin_ashi_candles: Empty data provided.")]
    EmptyData,
    #[error("heikin_ashi_candles: All values are NaN.")]
    AllValuesNaN,
    #[error("heikin_ashi_candles: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn heikin_ashi_candles(input: &HeikinAshiInput) -> Result<HeikinAshiOutput, HeikinAshiError> {
    let (open, high, low, close) = match &input.data {
        HeikinAshiData::Candles { candles } => {
            let o = candles.select_candle_field("open").unwrap();
            let h = candles.select_candle_field("high").unwrap();
            let l = candles.select_candle_field("low").unwrap();
            let c = candles.select_candle_field("close").unwrap();
            (o, h, l, c)
        }
        HeikinAshiData::Slices {
            open,
            high,
            low,
            close,
        } => (*open, *high, *low, *close),
    };

    if open.is_empty() || high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(HeikinAshiError::EmptyData);
    }

    let len = open.len();
    if len != high.len() || len != low.len() || len != close.len() {
        return Err(HeikinAshiError::EmptyData);
    }

    let first_valid_idx = match (0..len)
        .find(|&i| !open[i].is_nan() && !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan())
    {
        Some(idx) => idx,
        None => return Err(HeikinAshiError::AllValuesNaN),
    };

    if (len - first_valid_idx) < 2 {
        return Err(HeikinAshiError::NotEnoughValidData {
            needed: 2,
            valid: len - first_valid_idx,
        });
    }

    let mut ha_open = vec![f64::NAN; len];
    let mut ha_high = vec![f64::NAN; len];
    let mut ha_low = vec![f64::NAN; len];
    let mut ha_close = vec![f64::NAN; len];

    ha_open[first_valid_idx] = open[first_valid_idx];
    ha_close[first_valid_idx] = (open[first_valid_idx]
        + high[first_valid_idx]
        + low[first_valid_idx]
        + close[first_valid_idx])
        / 4.0;
    ha_high[first_valid_idx] = high[first_valid_idx]
        .max(ha_open[first_valid_idx])
        .max(ha_close[first_valid_idx]);
    ha_low[first_valid_idx] = low[first_valid_idx]
        .min(ha_open[first_valid_idx])
        .min(ha_close[first_valid_idx]);

    for i in (first_valid_idx + 1)..len {
        let prev_idx = i - 1;
        ha_open[i] = (open[prev_idx] + close[prev_idx]) / 2.0;
        ha_close[i] = (open[i] + high[i] + low[i] + close[i]) / 4.0;
        ha_high[i] = high[i].max(ha_open[i]).max(ha_close[i]);
        ha_low[i] = low[i].min(ha_open[i]).min(ha_close[i]);
    }

    Ok(HeikinAshiOutput {
        open: ha_open,
        high: ha_high,
        low: ha_low,
        close: ha_close,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_heikin_ashi_empty_data() {
        let input = HeikinAshiInput::from_slices(&[], &[], &[], &[]);
        let result = heikin_ashi_candles(&input);
        assert!(result.is_err(), "Expected empty data error");
    }

    #[test]
    fn test_heikin_ashi_all_nan() {
        let open = [f64::NAN, f64::NAN];
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let close = [f64::NAN, f64::NAN];
        let input = HeikinAshiInput::from_slices(&open, &high, &low, &close);
        let result = heikin_ashi_candles(&input);
        assert!(result.is_err(), "Expected all NaN error");
    }

    #[test]
    fn test_heikin_ashi_not_enough_data() {
        let open = [60000.0];
        let high = [60100.0];
        let low = [59900.0];
        let close = [60050.0];
        let input = HeikinAshiInput::from_slices(&open, &high, &low, &close);
        let result = heikin_ashi_candles(&input);
        assert!(result.is_err(), "Expected not enough data error");
    }

    #[test]
    fn test_heikin_ashi_basic_calculation() {
        let open = [1.0, 2.0, 4.0];
        let high = [2.0, 5.0, 5.0];
        let low = [0.5, 1.5, 3.0];
        let close = [1.5, 3.0, 4.0];
        let input = HeikinAshiInput::from_slices(&open, &high, &low, &close);
        let result = heikin_ashi_candles(&input).expect("Failed to calculate Heikin Ashi");
        assert_eq!(result.open.len(), 3);
        assert_eq!(result.close.len(), 3);
        assert_eq!(result.high.len(), 3);
        assert_eq!(result.low.len(), 3);
        assert!(result.open[0].abs() - 1.0 < 1e-10);
        assert!(result.close[0].abs() - 1.625 < 1e-10);
        assert!(!result.open[1].is_nan());
        assert!(!result.open[2].is_nan());
    }

    #[test]
    fn test_heikin_ashi_from_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HeikinAshiInput::from_candles(&candles);
        let output = heikin_ashi_candles(&input).expect("Failed Heikin Ashi from candles");
        assert_eq!(output.open.len(), candles.close.len());
        assert_eq!(output.high.len(), candles.close.len());
        assert_eq!(output.low.len(), candles.close.len());
        assert_eq!(output.close.len(), candles.close.len());
    }

    #[test]
    fn test_heikin_ashi_accuracy_sample() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HeikinAshiInput::from_candles(&candles);
        let output = heikin_ashi_candles(&input).expect("Failed Heikin Ashi calculation");
        assert_eq!(output.open.len(), candles.close.len());
        let len = output.open.len();
        if len >= 5 {
            let expected_last_five_high = [59348.5, 59405.0, 59304.0, 59310.0, 59236.0];
            let expected_last_five_low = [59001.0, 59084.0, 58932.0, 58983.0, 58299.0];
            let expected_last_five_close = [59221.75, 59238.75, 59114.25, 59121.75, 58836.25];
            let expected_last_five_open = [59348.5, 59277.5, 59233.0, 59110.5, 59097.0];
            for i in 0..5 {
                let idx = len - 5 + i;
                let diff_high = (output.high[idx] - expected_last_five_high[i]).abs();
                let diff_low = (output.low[idx] - expected_last_five_low[i]).abs();
                let diff_close = (output.close[idx] - expected_last_five_close[i]).abs();
                let diff_open = (output.open[idx] - expected_last_five_open[i]).abs();
                assert!(
                    diff_high < 1.0,
                    "Heikin Ashi high mismatch at {}: got {}, expected {}",
                    idx,
                    output.high[idx],
                    expected_last_five_high[i]
                );
                assert!(
                    diff_low < 1.0,
                    "Heikin Ashi low mismatch at {}: got {}, expected {}",
                    idx,
                    output.low[idx],
                    expected_last_five_low[i]
                );
                assert!(
                    diff_close < 1.0,
                    "Heikin Ashi close mismatch at {}: got {}, expected {}",
                    idx,
                    output.close[idx],
                    expected_last_five_close[i]
                );
                assert!(
                    diff_open < 1.0,
                    "Heikin Ashi open mismatch at {}: got {}, expected {}",
                    idx,
                    output.open[idx],
                    expected_last_five_open[i]
                );
            }
        }
    }
}
