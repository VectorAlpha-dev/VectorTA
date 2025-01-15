/// # Normalized Average True Range (NATR)
///
/// A volatility indicator that normalizes the Average True Range (ATR) by the
/// closing price, expressed as a percentage. NATR is useful for comparing volatility
/// across different assets or time periods where price scales differ.
///
/// ## Parameters
/// - **period**: The number of data points to consider for the ATR calculation (Wilder's method). Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: natr: Input data slice is empty.
/// - **InvalidPeriod**: natr: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: natr: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: natr: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(NatrOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the ATR window is filled.
/// - **`Err(NatrError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum NatrData<'a> {
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
pub struct NatrOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct NatrParams {
    pub period: Option<usize>,
}

impl Default for NatrParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct NatrInput<'a> {
    pub data: NatrData<'a>,
    pub params: NatrParams,
}

impl<'a> NatrInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: NatrParams) -> Self {
        Self {
            data: NatrData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: NatrParams,
    ) -> Self {
        Self {
            data: NatrData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: NatrData::Candles { candles },
            params: NatrParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| NatrParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum NatrError {
    #[error("natr: Empty data provided for NATR.")]
    EmptyData,
    #[error("natr: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("natr: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("natr: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn natr(input: &NatrInput) -> Result<NatrOutput, NatrError> {
    let (high, low, close) = match &input.data {
        NatrData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            (high, low, close)
        }
        NatrData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(NatrError::EmptyData);
    }

    let period = input.get_period();
    let len = high.len().min(low.len()).min(close.len());
    if period == 0 || period > len {
        return Err(NatrError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let first_valid_idx = {
        let first_valid_idx_h = high.iter().position(|&x| !x.is_nan());
        let first_valid_idx_l = low.iter().position(|&x| !x.is_nan());
        let first_valid_idx_c = close.iter().position(|&x| !x.is_nan());

        match (first_valid_idx_h, first_valid_idx_l, first_valid_idx_c) {
            (Some(h), Some(l), Some(c)) => Some(h.max(l).max(c)),
            _ => None,
        }
    };

    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(NatrError::AllValuesNaN),
    };

    if (len - first_valid_idx) < period {
        return Err(NatrError::NotEnoughValidData {
            needed: period,
            valid: len - first_valid_idx,
        });
    }

    let mut natr_values = vec![f64::NAN; len];

    let mut sum_tr = 0.0;
    let mut prev_atr = 0.0;
    let mut count_since_first = 0usize;

    for i in first_valid_idx..len {
        let tr = if i == first_valid_idx {
            high[i] - low[i]
        } else {
            let tr_curr = high[i] - low[i];
            let tr_prev_close_high = (high[i] - close[i - 1]).abs();
            let tr_prev_close_low = (low[i] - close[i - 1]).abs();
            tr_curr.max(tr_prev_close_high).max(tr_prev_close_low)
        };

        if count_since_first < period {
            sum_tr += tr;
            if count_since_first == period - 1 {
                prev_atr = sum_tr / (period as f64);
                let c = close[i];
                if c.is_finite() && c != 0.0 {
                    natr_values[i] = (prev_atr / c) * 100.0;
                } else {
                    natr_values[i] = 0.0;
                }
            }
        } else {
            let new_atr = ((prev_atr * ((period - 1) as f64)) + tr) / (period as f64);
            prev_atr = new_atr;

            let c = close[i];
            if c.is_finite() && c != 0.0 {
                natr_values[i] = (new_atr / c) * 100.0;
            } else {
                natr_values[i] = 0.0;
            }
        }

        count_since_first += 1;
    }

    Ok(NatrOutput {
        values: natr_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_natr_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = NatrParams { period: None };
        let input_default = NatrInput::from_candles(&candles, default_params);
        let output_default = natr(&input_default).expect("Failed NATR with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_7 = NatrParams { period: Some(7) };
        let input_period_7 = NatrInput::from_candles(&candles, params_period_7);
        let output_period_7 = natr(&input_period_7).expect("Failed NATR with period=7");
        assert_eq!(output_period_7.values.len(), candles.close.len());
    }

    #[test]
    fn test_natr_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = NatrParams { period: Some(14) };
        let input = NatrInput::from_candles(&candles, params);
        let natr_result = natr(&input).expect("Failed to calculate NATR");

        assert_eq!(
            natr_result.values.len(),
            close_prices.len(),
            "NATR length mismatch"
        );

        let expected_last_five = [
            1.5465877404905772,
            1.4773840355794576,
            1.4201627494720954,
            1.3556212509014807,
            1.3836271128536142,
        ];
        assert!(
            natr_result.values.len() >= 5,
            "NATR length too short to check last five"
        );
        let start_index = natr_result.values.len() - 5;
        let result_last_five = &natr_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-8,
                "NATR mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
        let params2 = NatrParams { period: Some(14) };
        let period = params2.period.unwrap();
        for i in 0..(period - 1) {
            assert!(
                natr_result.values[i].is_nan(),
                "Expected NATR values before the period window to be NaN, got {} at index {}",
                natr_result.values[i],
                i
            );
        }

        let default_input = NatrInput::with_default_candles(&candles);
        let default_natr_result = natr(&default_input).expect("Failed to calculate NATR defaults");
        assert_eq!(default_natr_result.values.len(), close_prices.len());
    }

    #[test]
    fn test_natr_params_with_default_params() {
        let default_params = NatrParams::default();
        assert_eq!(
            default_params.period,
            Some(14),
            "Expected period to be Some(14) in default parameters"
        );
    }

    #[test]
    fn test_natr_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = NatrInput::with_default_candles(&candles);
        match input.data {
            NatrData::Candles { .. } => {}
            _ => panic!("Expected NatrData::Candles variant"),
        }
    }

    #[test]
    fn test_natr_with_zero_period() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 10.0, 15.0];
        let close = [7.0, 14.0, 25.0];
        let params = NatrParams { period: Some(0) };
        let input = NatrInput::from_slices(&high, &low, &close, params);

        let result = natr(&input);
        assert!(result.is_err(), "Expected an error for zero period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error message, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_natr_with_period_exceeding_data_length() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 10.0, 15.0];
        let close = [7.0, 14.0, 25.0];
        let params = NatrParams { period: Some(10) };
        let input = NatrInput::from_slices(&high, &low, &close, params);

        let result = natr(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_natr_very_small_data_set() {
        let high = [42.0];
        let low = [40.0];
        let close = [41.0];
        let params = NatrParams { period: Some(14) };
        let input = NatrInput::from_slices(&high, &low, &close, params);

        let result = natr(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_natr_all_values_nan() {
        let high = [f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN];
        let close = [f64::NAN, f64::NAN];
        let params = NatrParams { period: Some(2) };
        let input = NatrInput::from_slices(&high, &low, &close, params);

        let result = natr(&input);
        assert!(result.is_err(), "Expected an error for all NaN data");
    }

    #[test]
    fn test_natr_not_enough_valid_data() {
        let high = [f64::NAN, 10.0];
        let low = [f64::NAN, 5.0];
        let close = [f64::NAN, 7.0];
        let params = NatrParams { period: Some(5) };
        let input = NatrInput::from_slices(&high, &low, &close, params);

        let result = natr(&input);
        assert!(result.is_err(), "Expected error for not enough valid data");
    }

    #[test]
    fn test_natr_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = NatrParams { period: Some(14) };
        let first_input = NatrInput::from_candles(&candles, first_params);
        let first_result = natr(&first_input).expect("Failed to calculate first NATR");
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = NatrParams { period: Some(14) };
        let second_input = NatrInput::from_slices(
            &first_result.values,
            &first_result.values,
            &first_result.values,
            second_params,
        );
        let second_result = natr(&second_input).expect("Failed to calculate second NATR");
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
    fn test_natr_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = NatrParams { period: Some(14) };
        let input = NatrInput::from_candles(&candles, params);
        let natr_result = natr(&input).expect("Failed to calculate NATR");

        assert_eq!(natr_result.values.len(), candles.close.len());
        if natr_result.values.len() > 30 {
            for i in 30..natr_result.values.len() {
                assert!(
                    !natr_result.values[i].is_nan(),
                    "Expected no NaN after index 30, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
