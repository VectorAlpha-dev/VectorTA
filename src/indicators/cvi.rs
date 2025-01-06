/// # Chaikin's Volatility (CVI)
///
/// Chaikin's Volatility (CVI) measures the volatility of a financial instrument by calculating
/// the percentage difference between two exponentially smoothed averages of the trading range
/// (high-low) over a given period. A commonly used default period is 10. Higher values for the
/// period will smooth out short-term fluctuations, while lower values will track rapid changes
/// more closely.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 10.
///
/// ## Errors
/// - **EmptyData**: cvi: Input data (high/low) is empty.
/// - **InvalidPeriod**: cvi: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: cvi: Fewer than `2*period - 1` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: cvi: All input high/low values are `NaN`.
///
/// ## Returns
/// - **`Ok(CviOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the first calculable index (at `2*period - 1` from the first
///   valid data point).
/// - **`Err(CviError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum CviData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct CviOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CviParams {
    pub period: Option<usize>,
}

impl Default for CviParams {
    fn default() -> Self {
        Self { period: Some(10) }
    }
}

#[derive(Debug, Clone)]
pub struct CviInput<'a> {
    pub data: CviData<'a>,
    pub params: CviParams,
}

impl<'a> CviInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: CviParams) -> Self {
        Self {
            data: CviData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: CviParams) -> Self {
        Self {
            data: CviData::Slices { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: CviData::Candles { candles },
            params: CviParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| CviParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum CviError {
    #[error("cvi: Empty data provided for CVI.")]
    EmptyData,
    #[error("cvi: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("cvi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("cvi: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn cvi(input: &CviInput) -> Result<CviOutput, CviError> {
    let (high, low) = match &input.data {
        CviData::Candles { candles } => {
            if candles.high.is_empty() || candles.low.is_empty() {
                return Err(CviError::EmptyData);
            }
            (&candles.high[..], &candles.low[..])
        }
        CviData::Slices { high, low } => {
            if high.is_empty() || low.is_empty() {
                return Err(CviError::EmptyData);
            }
            (*high, *low)
        }
    };

    if high.len() != low.len() {
        return Err(CviError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(CviError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }

    let first_valid_idx = match (0..high.len()).find(|&i| !high[i].is_nan() && !low[i].is_nan()) {
        Some(idx) => idx,
        None => return Err(CviError::AllValuesNaN),
    };

    let needed = 2 * period - 1;
    if (high.len() - first_valid_idx) < needed {
        return Err(CviError::NotEnoughValidData {
            needed,
            valid: high.len() - first_valid_idx,
        });
    }

    let mut cvi_values = vec![f64::NAN; high.len()];
    let alpha = 2.0 / (period as f64 + 1.0);

    let mut val = high[first_valid_idx] - low[first_valid_idx];
    let mut lag_buffer = Vec::with_capacity(period);
    lag_buffer.push(val);

    for i in (first_valid_idx + 1)..(first_valid_idx + needed) {
        let range = high[i] - low[i];
        val += (range - val) * alpha;
        if lag_buffer.len() < period {
            lag_buffer.push(val);
        } else {
            lag_buffer.remove(0);
            lag_buffer.push(val);
        }
    }

    for i in (first_valid_idx + needed)..high.len() {
        let range = high[i] - low[i];
        val += (range - val) * alpha;
        let old = lag_buffer.remove(0);
        cvi_values[i] = 100.0 * (val - old) / old;
        lag_buffer.push(val);
    }

    Ok(CviOutput { values: cvi_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_cvi_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = CviParams { period: None };
        let input_default = CviInput::from_candles(&candles, default_params);
        let output_default = cvi(&input_default).expect("Failed CVI with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = CviParams { period: Some(14) };
        let input_period_14 = CviInput::from_candles(&candles, params_period_14);
        let output_period_14 = cvi(&input_period_14).expect("Failed CVI with period=14");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = CviParams { period: Some(20) };
        let input_custom = CviInput::from_candles(&candles, params_custom);
        let output_custom = cvi(&input_custom).expect("Failed CVI fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_cvi_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = CviParams { period: Some(5) };
        let input = CviInput::from_candles(&candles, params);
        let cvi_result = cvi(&input).expect("Failed to calculate CVI");

        let expected_last_five_cvi = [
            -52.96320026271643,
            -64.39616778235792,
            -59.4830094380472,
            -52.4690724045071,
            -11.858704179539174,
        ];
        assert!(
            cvi_result.values.len() >= 5,
            "CVI result length is too short for validation"
        );
        let start_index = cvi_result.values.len() - 5;
        let result_last_five = &cvi_result.values[start_index..];
        for (i, &val) in result_last_five.iter().enumerate() {
            let expected = expected_last_five_cvi[i];
            assert!(
                (val - expected).abs() < 1e-6,
                "CVI mismatch at index {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }
    }

    #[test]
    fn test_cvi_params_with_default_params() {
        let default_params = CviParams::default();
        assert_eq!(
            default_params.period,
            Some(10),
            "Expected default period to be Some(10)"
        );
    }

    #[test]
    fn test_cvi_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = CviInput::with_default_candles(&candles);
        match input.data {
            CviData::Candles { .. } => {}
            _ => panic!("Expected CviData::Candles variant"),
        }
    }

    #[test]
    fn test_cvi_with_zero_period() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = CviParams { period: Some(0) };
        let input = CviInput::from_slices(&high, &low, params);

        let result = cvi(&input);
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
    fn test_cvi_with_period_exceeding_data_length() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = CviParams { period: Some(10) };
        let input = CviInput::from_slices(&high, &low, params);

        let result = cvi(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_cvi_very_small_data_set() {
        let high = [5.0];
        let low = [2.0];
        let params = CviParams { period: Some(10) };
        let input = CviInput::from_slices(&high, &low, params);

        let result = cvi(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_cvi_with_nan_data() {
        let high = [f64::NAN, 20.0, 30.0];
        let low = [5.0, 15.0, f64::NAN];
        let input = CviInput::from_slices(&high, &low, CviParams { period: Some(2) });

        let result = cvi(&input);
        assert!(result.is_err(), "Expected an error due to trailing NaN");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Not enough valid data")
                    || e.to_string().contains("All values are NaN"),
                "Expected 'Not enough valid data' or 'All values are NaN', got: {}",
                e
            );
        }
    }

    #[test]
    fn test_cvi_slice_reinput() {
        let high = [
            10.0, 12.0, 12.5, 12.2, 13.0, 14.0, 15.0, 16.0, 16.5, 17.0, 17.5, 18.0,
        ];
        let low = [
            9.0, 10.0, 11.5, 11.0, 12.0, 13.5, 14.0, 14.5, 15.5, 16.0, 16.5, 17.0,
        ];
        let first_input = CviInput::from_slices(&high, &low, CviParams { period: Some(3) });
        let first_result = cvi(&first_input).expect("Failed to calculate first CVI");

        let second_input =
            CviInput::from_slices(&first_result.values, &low, CviParams { period: Some(3) });
        let second_result = cvi(&second_input).expect("Failed to calculate second CVI");

        assert_eq!(
            second_result.values.len(),
            low.len(),
            "Second CVI output length mismatch"
        );
    }
}
