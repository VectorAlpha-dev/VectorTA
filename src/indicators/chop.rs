use crate::indicators::atr::{atr, AtrData, AtrInput, AtrParams};
use crate::indicators::utility_functions::{max_rolling, min_rolling, sum_rolling, RollingError};
/// # Choppiness Index (CHOP)
///
/// The Choppiness Index is a volatility indicator designed to quantify whether
/// the market is choppy (range-bound) or trending. A higher CHOP value implies
/// more choppiness (sideways movement), while a lower value implies trending behavior.
///
///
/// ```ignore
/// atr_sum = SUM(ATR(high, low, close, drift), period)
/// hh = MAX(high, period)
/// ll = MIN(low, period)
/// chop = (scalar * (log10(atr_sum) - log10(hh - ll))) / log10(period)
/// ```
///
/// The leading values in the output will be `NaN` until at least `period` bars of valid
/// data are available for the rolling computations.
///
/// ## Parameters
/// - **period**: Rolling window length for summation, highest-high, and lowest-low. Defaults to 14.
/// - **scalar**: Multiplicative factor for scaling (commonly 100). Defaults to 100.
/// - **drift**: ATR period for calculating the rolling ATR. Defaults to 1.
///
/// ## Errors
/// - **EmptyData**: chop: Input data slice is empty.
/// - **InvalidPeriod**: chop: `period` is zero or exceeds the data length.
/// - **AllValuesNaN**: chop: All relevant data (high, low, close) are `NaN`.
/// - **NotEnoughValidData**: chop: Fewer than `period` valid data points remain after the first valid index.
/// - **UnderlyingFunctionFailed**: If any rolling computations (e.g., ATR, sum, max, or min) fail internally.
///
/// ## Returns
/// - **`Ok(ChopOutput)`** on success, containing a `Vec<f64>` with the same length as the input,
///   filled with leading `NaN` until the rolling window is fully available.
/// - **`Err(ChopError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ChopData<'a> {
    Candles(&'a Candles),
    Slice {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct ChopParams {
    pub period: Option<usize>,
    pub scalar: Option<f64>,
    pub drift: Option<usize>,
}

impl Default for ChopParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            scalar: Some(100.0),
            drift: Some(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChopInput<'a> {
    pub data: ChopData<'a>,
    pub params: ChopParams,
}

impl<'a> ChopInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: ChopParams) -> Self {
        Self {
            data: ChopData::Candles(candles),
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: ChopParams,
    ) -> Self {
        Self {
            data: ChopData::Slice { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: ChopData::Candles(candles),
            params: ChopParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| ChopParams::default().period.unwrap())
    }

    pub fn get_scalar(&self) -> f64 {
        self.params
            .scalar
            .unwrap_or_else(|| ChopParams::default().scalar.unwrap())
    }

    pub fn get_drift(&self) -> usize {
        self.params
            .drift
            .unwrap_or_else(|| ChopParams::default().drift.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct ChopOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum ChopError {
    #[error("chop: Empty data provided.")]
    EmptyData,
    #[error("chop: Invalid period: period={period}, data length={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("chop: All relevant data (high/low/close) are NaN.")]
    AllValuesNaN,
    #[error("chop: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("chop: Underlying function failed: {0}")]
    UnderlyingFunctionFailed(String),
}
#[inline]
pub fn chop(input: &ChopInput) -> Result<ChopOutput, ChopError> {
    let (high, low, close) = match &input.data {
        ChopData::Candles(candles) => (
            candles.high.as_slice(),
            candles.low.as_slice(),
            candles.close.as_slice(),
        ),
        ChopData::Slice { high, low, close } => (*high, *low, *close),
    };

    let len = close.len();
    if len == 0 {
        return Err(ChopError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(ChopError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first_valid_idx = match (0..len).find(|&i| {
        let (h, l, c) = (high[i], low[i], close[i]);
        !(h.is_nan() || l.is_nan() || c.is_nan())
    }) {
        Some(idx) => idx,
        None => return Err(ChopError::AllValuesNaN),
    };

    if (len - first_valid_idx) < period {
        return Err(ChopError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }

    let drift = input.get_drift();
    let scalar = input.get_scalar();

    let atr_input = AtrInput::from_slices(
        high,
        low,
        close,
        AtrParams {
            length: Some(drift),
        },
    );
    let atr_output =
        atr(&atr_input).map_err(|e| ChopError::UnderlyingFunctionFailed(e.to_string()))?;
    let atr_values = atr_output.values;

    let atr_sum = sum_rolling(&atr_values, period)
        .map_err(|e| ChopError::UnderlyingFunctionFailed(e.to_string()))?;
    let hh = max_rolling(high, period)
        .map_err(|e| ChopError::UnderlyingFunctionFailed(e.to_string()))?;
    let ll =
        min_rolling(low, period).map_err(|e| ChopError::UnderlyingFunctionFailed(e.to_string()))?;

    let mut chop_values = vec![f64::NAN; len];
    let log_period = (period as f64).log10();
    let start_idx = first_valid_idx + period - 1;

    for i in start_idx..len {
        let sum_val = atr_sum[i];
        let hi_val = hh[i];
        let lo_val = ll[i];
        if !(sum_val.is_nan() || hi_val.is_nan() || lo_val.is_nan()) {
            let rng = hi_val - lo_val;
            if rng > 0.0 && sum_val > 0.0 {
                chop_values[i] = (scalar * (sum_val.log10() - rng.log10())) / log_period;
            }
        }
    }

    Ok(ChopOutput {
        values: chop_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_chop_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let partial_params = ChopParams {
            period: Some(30),
            scalar: None,
            drift: None,
        };
        let input_partial = ChopInput::from_candles(&candles, partial_params);
        let output_partial = chop(&input_partial).expect("Failed CHOP with partial params");
        assert_eq!(output_partial.values.len(), candles.close.len());
    }

    #[test]
    fn test_chop_accuracy() {
        let expected_final_5 = [
            49.98214330294626,
            48.90450693742312,
            46.63648608318844,
            46.19823574588033,
            56.22876423352909,
        ];

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = ChopInput::with_default_candles(&candles);
        let result = chop(&input).expect("CHOP default calculation failed");

        assert!(result.values.len() >= 5);
        let start_idx = result.values.len() - 5;
        for (i, &exp) in expected_final_5.iter().enumerate() {
            let idx = start_idx + i;
            let got = result.values[idx];
            assert!(
                (got - exp).abs() < 1e-4,
                "Mismatch in CHOP at idx={} => expected={}, got={}",
                idx,
                exp,
                got
            );
        }
    }

    #[test]
    fn test_chop_params_default() {
        let defaults = ChopParams::default();
        assert_eq!(defaults.period, Some(14));
        assert_eq!(defaults.scalar, Some(100.0));
        assert_eq!(defaults.drift, Some(1));
    }

    #[test]
    fn test_chop_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = ChopInput::with_default_candles(&candles);
        match input.data {
            ChopData::Candles(_) => {}
            _ => panic!("Expected ChopData::Candles variant"),
        }
    }

    #[test]
    fn test_chop_zero_period() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = ChopParams {
            period: Some(0),
            ..Default::default()
        };
        let input = ChopInput::from_candles(&candles, params);
        let result = chop(&input);
        assert!(result.is_err(), "Expected error for zero period");
    }

    #[test]
    fn test_chop_period_exceeding_data_length() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = ChopParams {
            period: Some(999999),
            ..Default::default()
        };
        let input = ChopInput::from_candles(&candles, params);
        let result = chop(&input);
        assert!(result.is_err(), "Expected error for huge period");
    }

    #[test]
    fn test_chop_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = ChopInput::with_default_candles(&candles);
        let result = chop(&input).expect("Failed CHOP calculation");

        let check_index = 240;
        if result.values.len() > check_index {
            let all_nan = result.values[check_index..].iter().all(|&x| x.is_nan());
            assert!(
                !all_nan,
                "All CHOP values from index {} onward are NaN.",
                check_index
            );
        }
    }
}
