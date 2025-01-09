/// # Directional Movement (DM)
///
/// Measures the strength of upward and downward price movements based on changes
/// between consecutive high and low values. +DM is computed when the positive
/// range (current high minus previous high) exceeds the negative range (previous
/// low minus current low), while -DM is computed in the opposite case. Both
/// values can be optionally smoothed over the specified `period`.
///
/// ## Parameters
/// - **period**: The smoothing window size (number of data points). Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: dm: Input high/low slices are empty or mismatched in length.
/// - **InvalidPeriod**: dm: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: dm: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: dm: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(DmOutput)`** on success, containing two `Vec<f64>` matching the input length,
///   with leading `NaN`s until the smoothing window is filled.
/// - **`Err(DmError)`** otherwise.
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DmData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct DmOutput {
    pub plus: Vec<f64>,
    pub minus: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DmParams {
    pub period: Option<usize>,
}

impl Default for DmParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct DmInput<'a> {
    pub data: DmData<'a>,
    pub params: DmParams,
}

impl<'a> DmInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: DmParams) -> Self {
        Self {
            data: DmData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: DmParams) -> Self {
        Self {
            data: DmData::Slices { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: DmData::Candles { candles },
            params: DmParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| DmParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum DmError {
    #[error("dm: Empty data provided or mismatched high/low lengths.")]
    EmptyData,
    #[error("dm: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("dm: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("dm: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn dm(input: &DmInput) -> Result<DmOutput, DmError> {
    let (high, low) = match &input.data {
        DmData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| DmError::EmptyData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| DmError::EmptyData)?;
            (high, low)
        }
        DmData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() || high.len() != low.len() {
        return Err(DmError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > high.len() {
        return Err(DmError::InvalidPeriod {
            period,
            data_len: high.len(),
        });
    }

    let first_valid_idx = match high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !h.is_nan() && !l.is_nan())
    {
        Some(idx) => idx,
        None => return Err(DmError::AllValuesNaN),
    };

    if (high.len() - first_valid_idx) < period {
        return Err(DmError::NotEnoughValidData {
            needed: period,
            valid: high.len() - first_valid_idx,
        });
    }

    let mut plus_dm = vec![f64::NAN; high.len()];
    let mut minus_dm = vec![f64::NAN; high.len()];

    let mut prev_high = high[first_valid_idx];
    let mut prev_low = low[first_valid_idx];
    let mut sum_plus = 0.0;
    let mut sum_minus = 0.0;

    let end_init = first_valid_idx + period - 1;
    for i in (first_valid_idx + 1)..=end_init {
        let diff_p = high[i] - prev_high;
        let diff_m = prev_low - low[i];
        prev_high = high[i];
        prev_low = low[i];

        let plus_val = if diff_p > 0.0 && diff_p > diff_m {
            diff_p
        } else {
            0.0
        };
        let minus_val = if diff_m > 0.0 && diff_m > diff_p {
            diff_m
        } else {
            0.0
        };

        sum_plus += plus_val;
        sum_minus += minus_val;
    }

    plus_dm[end_init] = sum_plus;
    minus_dm[end_init] = sum_minus;

    let inv_period = 1.0 / (period as f64);

    for i in (end_init + 1)..high.len() {
        let diff_p = high[i] - prev_high;
        let diff_m = prev_low - low[i];
        prev_high = high[i];
        prev_low = low[i];

        let plus_val = if diff_p > 0.0 && diff_p > diff_m {
            diff_p
        } else {
            0.0
        };
        let minus_val = if diff_m > 0.0 && diff_m > diff_p {
            diff_m
        } else {
            0.0
        };

        sum_plus = sum_plus - (sum_plus * inv_period) + plus_val;
        sum_minus = sum_minus - (sum_minus * inv_period) + minus_val;

        plus_dm[i] = sum_plus;
        minus_dm[i] = sum_minus;
    }

    Ok(DmOutput {
        plus: plus_dm,
        minus: minus_dm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_dm_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = DmParams { period: None };
        let input_default = DmInput::from_candles(&candles, default_params);
        let output_default = dm(&input_default).expect("Failed DM with default params");
        assert_eq!(output_default.plus.len(), candles.high.len());
        assert_eq!(output_default.minus.len(), candles.high.len());

        let params_custom = DmParams { period: Some(10) };
        let input_custom = DmInput::from_candles(&candles, params_custom);
        let output_custom = dm(&input_custom).expect("Failed DM with period=10");
        assert_eq!(output_custom.plus.len(), candles.high.len());
        assert_eq!(output_custom.minus.len(), candles.high.len());
    }

    #[test]
    fn test_dm_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = DmInput::with_default_candles(&candles);
        let result = dm(&input).expect("Failed DM with defaults");
        assert_eq!(result.plus.len(), candles.high.len());
        assert_eq!(result.minus.len(), candles.high.len());
    }

    #[test]
    fn test_dm_with_slice_data() {
        let high_values = [8000.0, 8050.0, 8100.0, 8075.0, 8110.0, 8050.0];
        let low_values = [7800.0, 7900.0, 7950.0, 7950.0, 8000.0, 7950.0];
        let params = DmParams { period: Some(3) };
        let input = DmInput::from_slices(&high_values, &low_values, params);
        let result = dm(&input).expect("Failed DM from slices");
        assert_eq!(result.plus.len(), 6);
        assert_eq!(result.minus.len(), 6);

        for i in 0..2 {
            assert!(result.plus[i].is_nan());
            assert!(result.minus[i].is_nan());
        }
    }

    #[test]
    fn test_dm_zero_period() {
        let high_values = [100.0, 110.0, 120.0];
        let low_values = [90.0, 100.0, 110.0];
        let params = DmParams { period: Some(0) };
        let input = DmInput::from_slices(&high_values, &low_values, params);
        let result = dm(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid period"));
        }
    }

    #[test]
    fn test_dm_period_exceeds_data_length() {
        let high_values = [100.0, 110.0, 120.0];
        let low_values = [90.0, 100.0, 110.0];
        let params = DmParams { period: Some(10) };
        let input = DmInput::from_slices(&high_values, &low_values, params);
        let result = dm(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_dm_not_enough_valid_data() {
        let high_values = [f64::NAN, f64::NAN, 100.0, 101.0, 102.0];
        let low_values = [f64::NAN, f64::NAN, 90.0, 89.0, 88.0];
        let params = DmParams { period: Some(5) };
        let input = DmInput::from_slices(&high_values, &low_values, params);
        let result = dm(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_dm_all_values_nan() {
        let high_values = [f64::NAN, f64::NAN, f64::NAN];
        let low_values = [f64::NAN, f64::NAN, f64::NAN];
        let params = DmParams { period: Some(3) };
        let input = DmInput::from_slices(&high_values, &low_values, params);
        let result = dm(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
    }

    #[test]
    fn test_dm_with_slice_reinput() {
        let high_values = [9000.0, 9100.0, 9050.0, 9200.0, 9150.0, 9300.0];
        let low_values = [8900.0, 9000.0, 8950.0, 9000.0, 9050.0, 9100.0];
        let params = DmParams { period: Some(2) };
        let input_first = DmInput::from_slices(&high_values, &low_values, params.clone());
        let result_first = dm(&input_first).expect("Failed first DM calculation");
        let input_second = DmInput::from_slices(&result_first.plus, &result_first.minus, params);
        let result_second = dm(&input_second).expect("Failed second DM calculation");
        assert_eq!(result_second.plus.len(), high_values.len());
        assert_eq!(result_second.minus.len(), high_values.len());
    }

    #[test]
    fn test_dm_known_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = DmParams { period: Some(14) };
        let input = DmInput::from_candles(&candles, params);
        let output = dm(&input).expect("Failed DM");

        let slice_size = 5;
        assert!(output.plus.len() >= slice_size);
        assert!(output.minus.len() >= slice_size);

        let last_plus_slice = &output.plus[output.plus.len() - slice_size..];
        let last_minus_slice = &output.minus[output.minus.len() - slice_size..];

        let expected_plus = [
            1410.819956368491,
            1384.04710234217,
            1285.186595032015,
            1199.3875525297283,
            1113.7170130633192,
        ];
        let expected_minus = [
            3602.8631384045057,
            3345.5157713756125,
            3258.5503591344973,
            3025.796762053462,
            3493.668421906786,
        ];

        for i in 0..slice_size {
            let diff_plus = (last_plus_slice[i] - expected_plus[i]).abs();
            let diff_minus = (last_minus_slice[i] - expected_minus[i]).abs();
            assert!(
                diff_plus < 1e-6,
                "Mismatch in +DM at index {}: expected {}, got {}",
                i,
                expected_plus[i],
                last_plus_slice[i]
            );
            assert!(
                diff_minus < 1e-6,
                "Mismatch in -DM at index {}: expected {}, got {}",
                i,
                expected_minus[i],
                last_minus_slice[i]
            );
        }
    }
}
