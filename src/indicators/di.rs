/// # Directional Indicator (DI)
///
/// Calculates both +DI (plus directional indicator) and -DI (minus directional indicator)
/// using the same approach as Wilder's DMI, which measures trend strength and direction
/// by comparing upward and downward price movements over a specified period.
///
/// ## Parameters
/// - **period**: The smoothing window size. Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: di: Input data slice is empty.
/// - **InvalidPeriod**: di: `period` is zero or exceeds data length.
/// - **NotEnoughValidData**: di: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: di: All high/low/close values are `NaN`.
///
/// ## Returns
/// - **`Ok(DiOutput)`** on success, containing two `Vec<f64>` matching the input length,
///   with leading `NaN`s until the calculation window is filled.
/// - **`Err(DiError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum DiData<'a> {
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
pub struct DiOutput {
    pub plus: Vec<f64>,
    pub minus: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DiParams {
    pub period: Option<usize>,
}

impl Default for DiParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct DiInput<'a> {
    pub data: DiData<'a>,
    pub params: DiParams,
}

impl<'a> DiInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: DiParams) -> Self {
        Self {
            data: DiData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: DiParams,
    ) -> Self {
        Self {
            data: DiData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: DiData::Candles { candles },
            params: DiParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| DiParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DiError {
    #[error("di: Empty data provided for DI.")]
    EmptyData,
    #[error("di: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("di: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("di: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn di(input: &DiInput) -> Result<DiOutput, DiError> {
    let (high, low, close) = match &input.data {
        DiData::Candles { candles } => {
            let h = source_type(candles, "high");
            let l = source_type(candles, "low");
            let c = source_type(candles, "close");
            (h, l, c)
        }
        DiData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(DiError::EmptyData);
    }

    let n = high.len();
    if low.len() != n || close.len() != n {
        return Err(DiError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > n {
        return Err(DiError::InvalidPeriod {
            period,
            data_len: n,
        });
    }

    let first_valid_idx = (0..n).find(|&i| {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        !(h.is_nan() || l.is_nan() || c.is_nan())
    });
    let first_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(DiError::AllValuesNaN),
    };

    if (n - first_idx) < period {
        return Err(DiError::NotEnoughValidData {
            needed: period,
            valid: n - first_idx,
        });
    }

    let mut plus_di = vec![f64::NAN; n];
    let mut minus_di = vec![f64::NAN; n];

    let mut prev_high = high[first_idx];
    let mut prev_low = low[first_idx];
    let mut prev_close = close[first_idx];
    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;
    let mut tr_sum = 0.0;

    for i in (first_idx + 1)..(first_idx + period) {
        let diff_p = high[i] - prev_high;
        let diff_m = prev_low - low[i];
        prev_high = high[i];
        prev_low = low[i];
        let tr = true_range(high[i], low[i], prev_close);
        prev_close = close[i];
        if diff_p > 0.0 && diff_p > diff_m {
            plus_dm_sum += diff_p;
        }
        if diff_m > 0.0 && diff_m > diff_p {
            minus_dm_sum += diff_m;
        }
        tr_sum += tr;
    }

    let mut idx = first_idx + period - 1;
    let mut current_plus_dm = plus_dm_sum;
    let mut current_minus_dm = minus_dm_sum;
    let mut current_tr = tr_sum;

    plus_di[idx] = if current_tr == 0.0 {
        0.0
    } else {
        (current_plus_dm / current_tr) * 100.0
    };
    minus_di[idx] = if current_tr == 0.0 {
        0.0
    } else {
        (current_minus_dm / current_tr) * 100.0
    };

    idx += 1;

    while idx < n {
        let diff_p = high[idx] - prev_high;
        let diff_m = prev_low - low[idx];
        prev_high = high[idx];
        prev_low = low[idx];
        let tr = true_range(high[idx], low[idx], prev_close);
        prev_close = close[idx];
        if diff_p > 0.0 && diff_p > diff_m {
            current_plus_dm = current_plus_dm - (current_plus_dm / (period as f64)) + diff_p;
        } else {
            current_plus_dm = current_plus_dm - (current_plus_dm / (period as f64));
        }
        if diff_m > 0.0 && diff_m > diff_p {
            current_minus_dm = current_minus_dm - (current_minus_dm / (period as f64)) + diff_m;
        } else {
            current_minus_dm = current_minus_dm - (current_minus_dm / (period as f64));
        }
        current_tr = current_tr - (current_tr / (period as f64)) + tr;
        plus_di[idx] = if current_tr == 0.0 {
            0.0
        } else {
            (current_plus_dm / current_tr) * 100.0
        };
        minus_di[idx] = if current_tr == 0.0 {
            0.0
        } else {
            (current_minus_dm / current_tr) * 100.0
        };
        idx += 1;
    }

    Ok(DiOutput {
        plus: plus_di,
        minus: minus_di,
    })
}

#[inline]
fn true_range(current_high: f64, current_low: f64, prev_close: f64) -> f64 {
    let mut tr1 = current_high - current_low;
    let tr2 = (current_high - prev_close).abs();
    let tr3 = (current_low - prev_close).abs();
    if tr2 > tr1 {
        tr1 = tr2;
    }
    if tr3 > tr1 {
        tr1 = tr3;
    }
    tr1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_di_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = DiParams { period: None };
        let input_default = DiInput::from_candles(&candles, default_params);
        let output_default = di(&input_default).expect("Failed DI with default params");
        assert_eq!(output_default.plus.len(), candles.close.len());
        assert_eq!(output_default.minus.len(), candles.close.len());
        let params_period_10 = DiParams { period: Some(10) };
        let input_period_10 = DiInput::from_candles(&candles, params_period_10);
        let output_period_10 = di(&input_period_10).expect("Failed DI with period=10");
        assert_eq!(output_period_10.plus.len(), candles.close.len());
        assert_eq!(output_period_10.minus.len(), candles.close.len());
    }

    #[test]
    fn test_di_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = DiParams { period: Some(14) };
        let input = DiInput::from_candles(&candles, params);
        let di_result = di(&input).expect("Failed to calculate DI");
        assert_eq!(di_result.plus.len(), candles.close.len());
        assert_eq!(di_result.minus.len(), candles.close.len());
        let test_plus = [
            10.99067007335658,
            11.306993269828585,
            10.948661818939213,
            10.683207768215592,
            9.802180952619183,
        ];
        let test_minus = [
            28.06728094177839,
            27.331240567633152,
            27.759989125359493,
            26.951434842917386,
            30.748897303623057,
        ];
        if di_result.plus.len() > 5 {
            let plus_tail = &di_result.plus[di_result.plus.len() - 5..];
            let minus_tail = &di_result.minus[di_result.minus.len() - 5..];
            for i in 0..5 {
                assert!(
                    (plus_tail[i] - test_plus[i]).abs() < 1e-6,
                    "Mismatch in +DI at tail index {}: expected {}, got {}",
                    i,
                    test_plus[i],
                    plus_tail[i]
                );
                assert!(
                    (minus_tail[i] - test_minus[i]).abs() < 1e-6,
                    "Mismatch in -DI at tail index {}: expected {}, got {}",
                    i,
                    test_minus[i],
                    minus_tail[i]
                );
            }
        }
    }

    #[test]
    fn test_di_params_with_default_params() {
        let default_params = DiParams::default();
        assert_eq!(default_params.period, Some(14));
    }

    #[test]
    fn test_di_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = DiInput::with_default_candles(&candles);
        match input.data {
            DiData::Candles { .. } => {}
            _ => panic!("Expected DiData::Candles variant"),
        }
    }

    #[test]
    fn test_di_with_zero_period() {
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 8.0, 7.0];
        let close = [9.5, 10.0, 11.0];
        let params = DiParams { period: Some(0) };
        let input = DiInput::from_slices(&high, &low, &close, params);
        let result = di(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_di_with_period_exceeding_data_length() {
        let high = [10.0, 11.0, 12.0];
        let low = [9.0, 8.0, 7.0];
        let close = [9.5, 10.0, 11.0];
        let params = DiParams { period: Some(10) };
        let input = DiInput::from_slices(&high, &low, &close, params);
        let result = di(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_di_very_small_data_set() {
        let high = [42.0];
        let low = [41.0];
        let close = [41.5];
        let params = DiParams { period: Some(14) };
        let input = DiInput::from_slices(&high, &low, &close, params);
        let result = di(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_di_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = DiParams { period: Some(14) };
        let first_input = DiInput::from_candles(&candles, first_params);
        let first_result = di(&first_input).expect("Failed to calculate first DI");
        assert_eq!(first_result.plus.len(), candles.close.len());
        assert_eq!(first_result.minus.len(), candles.close.len());
        let second_params = DiParams { period: Some(14) };
        let second_input = DiInput::from_slices(
            &first_result.plus,
            &first_result.minus,
            &candles.close,
            second_params,
        );
        let second_result = di(&second_input).expect("Failed to calculate second DI");
        assert_eq!(second_result.plus.len(), first_result.plus.len());
        assert_eq!(second_result.minus.len(), first_result.minus.len());
    }

    #[test]
    fn test_di_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = DiParams { period: Some(14) };
        let input = DiInput::from_candles(&candles, params);
        let di_result = di(&input).expect("Failed to calculate DI");
        assert_eq!(di_result.plus.len(), candles.close.len());
        assert_eq!(di_result.minus.len(), candles.close.len());
        if di_result.plus.len() > 40 {
            for i in 40..di_result.plus.len() {
                assert!(!di_result.plus[i].is_nan());
                assert!(!di_result.minus[i].is_nan());
            }
        }
    }
}
