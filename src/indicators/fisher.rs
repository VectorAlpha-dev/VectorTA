/// # Fisher Transform
///
/// The Fisher Transform helps identify potential price reversals by normalizing price extremes
/// via the Fisher Transform function.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 9.
///
/// ## Errors
/// - **EmptyData**: fisher: Input data slice is empty.
/// - **InvalidPeriod**: fisher: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: fisher: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: fisher: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(FisherOutput)`** on success, containing `fisher` and `signal` vectors matching
///   the input length, with leading `NaN`s until the transform window is filled.
/// - **`Err(FisherError)`** otherwise.
use crate::utilities::data_loader::{read_candles_from_csv, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum FisherData<'a> {
    Candles { candles: &'a Candles },
    Slices { high: &'a [f64], low: &'a [f64] },
}

#[derive(Debug, Clone)]
pub struct FisherOutput {
    pub fisher: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FisherParams {
    pub period: Option<usize>,
}

impl Default for FisherParams {
    fn default() -> Self {
        Self { period: Some(9) }
    }
}

#[derive(Debug, Clone)]
pub struct FisherInput<'a> {
    pub data: FisherData<'a>,
    pub params: FisherParams,
}

impl<'a> FisherInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: FisherParams) -> Self {
        Self {
            data: FisherData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: FisherParams) -> Self {
        Self {
            data: FisherData::Slices { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: FisherData::Candles { candles },
            params: FisherParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| FisherParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum FisherError {
    #[error("fisher: Empty data provided.")]
    EmptyData,
    #[error("fisher: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("fisher: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("fisher: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn fisher(input: &FisherInput) -> Result<FisherOutput, FisherError> {
    let (high, low) = match &input.data {
        FisherData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| FisherError::EmptyData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| FisherError::EmptyData)?;
            (high, low)
        }
        FisherData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(FisherError::EmptyData);
    }

    let period = input.get_period();
    let data_len = high.len().min(low.len());
    if period == 0 || period > data_len {
        return Err(FisherError::InvalidPeriod { period, data_len });
    }

    let mut merged = vec![f64::NAN; data_len];
    for i in 0..data_len {
        merged[i] = 0.5 * (high[i] + low[i]);
    }

    let first_valid_idx = match merged.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(FisherError::AllValuesNaN),
    };

    if (data_len - first_valid_idx) < period {
        return Err(FisherError::NotEnoughValidData {
            needed: period,
            valid: data_len - first_valid_idx,
        });
    }

    let mut fisher_vals = vec![f64::NAN; data_len];
    let mut signal_vals = vec![f64::NAN; data_len];
    let mut prev_fish = 0.0;
    let mut val1 = 0.0;

    for i in first_valid_idx..data_len {
        if i < first_valid_idx + period - 1 {
            continue;
        }
        let start = i + 1 - period;
        let (min_val, max_val) = {
            let window = &merged[start..=i];
            let mut local_min = f64::MAX;
            let mut local_max = f64::MIN;
            for &v in window.iter() {
                if v < local_min {
                    local_min = v;
                }
                if v > local_max {
                    local_max = v;
                }
            }
            (local_min, local_max)
        };
        let range = (max_val - min_val).max(0.001);
        let current_hl = merged[i];
        val1 = 0.33 * 2.0 * ((current_hl - min_val) / range - 0.5) + 0.67 * val1;
        if val1 > 0.99 {
            val1 = 0.999;
        } else if val1 < -0.99 {
            val1 = -0.999;
        }

        signal_vals[i] = prev_fish;
        let new_fish = 0.5 * ((1.0 + val1) / (1.0 - val1)).ln() + 0.5 * prev_fish;
        fisher_vals[i] = new_fish;
        prev_fish = new_fish;
    }

    Ok(FisherOutput {
        fisher: fisher_vals,
        signal: signal_vals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = FisherParams { period: Some(9) };
        let input = FisherInput::from_candles(&candles, params);
        let fisher_result = fisher(&input).expect("Failed to calculate Fisher Transform");

        assert_eq!(
            fisher_result.fisher.len(),
            close_prices.len(),
            "Fisher output length mismatch"
        );
        assert_eq!(
            fisher_result.signal.len(),
            close_prices.len(),
            "Signal output length mismatch"
        );

        let expected_last_five_fisher = [
            -0.4720164683904261,
            -0.23467530106650444,
            -0.14879388501136784,
            -0.026651419122953053,
            -0.2569225042442664,
        ];
        let expected_last_five_signal = [
            -0.7742705746872902,
            -0.4720164683904261,
            -0.23467530106650444,
            -0.14879388501136784,
            -0.026651419122953053,
        ];

        assert!(
            fisher_result.fisher.len() >= 5,
            "Fisher result length too short"
        );
        let start_index = fisher_result.fisher.len() - 5;
        let result_last_five_fisher = &fisher_result.fisher[start_index..];
        let result_last_five_signal = &fisher_result.signal[start_index..];

        for (i, &value) in result_last_five_fisher.iter().enumerate() {
            let expected_value = expected_last_five_fisher[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "Fisher mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        for (i, &value) in result_last_five_signal.iter().enumerate() {
            let expected_value = expected_last_five_signal[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "Signal mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period: usize = 9;
        for i in 0..(period - 1) {
            assert!(
                fisher_result.fisher[i].is_nan(),
                "Expected NaN in fisher_values at index {}",
                i
            );
            assert!(
                fisher_result.signal[i].is_nan(),
                "Expected NaN in signal_values at index {}",
                i
            );
        }

        let default_input = FisherInput::with_default_candles(&candles);
        let default_result = fisher(&default_input).expect("Failed default Fisher");
        assert_eq!(
            default_result.fisher.len(),
            close_prices.len(),
            "Default Fisher length mismatch"
        );
        assert_eq!(
            default_result.signal.len(),
            close_prices.len(),
            "Default Signal length mismatch"
        );
    }
    #[test]
    fn test_fisher_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = FisherParams { period: None };
        let input_default = FisherInput::from_candles(&candles, default_params);
        let output_default = fisher(&input_default).expect("Failed Fisher with default params");
        assert_eq!(output_default.fisher.len(), candles.close.len());
        assert_eq!(output_default.signal.len(), candles.close.len());

        let params_period_14 = FisherParams { period: Some(14) };
        let input_period_14 = FisherInput::from_candles(&candles, params_period_14);
        let output_period_14 = fisher(&input_period_14).expect("Failed Fisher with period=14");
        assert_eq!(output_period_14.fisher.len(), candles.close.len());
        assert_eq!(output_period_14.signal.len(), candles.close.len());
    }

    #[test]
    fn test_fisher_accuracy_check() {
        let high = [10.0, 12.0, 14.0, 16.0, 18.0];
        let low = [5.0, 6.0, 7.0, 8.0, 9.0];
        let params = FisherParams { period: Some(3) };
        let input = FisherInput::from_slices(&high, &low, params);
        let output = fisher(&input).expect("Fisher calculation failed");
        assert_eq!(output.fisher.len(), 5);
        assert_eq!(output.signal.len(), 5);
    }

    #[test]
    fn test_fisher_with_zero_period() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = FisherParams { period: Some(0) };
        let input = FisherInput::from_slices(&high, &low, params);
        let result = fisher(&input);
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
    fn test_fisher_period_exceeding_length() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = FisherParams { period: Some(10) };
        let input = FisherInput::from_slices(&high, &low, params);
        let result = fisher(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_fisher_all_nan() {
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let params = FisherParams { period: Some(3) };
        let input = FisherInput::from_slices(&high, &low, params);
        let result = fisher(&input);
        assert!(result.is_err(), "Expected error for all NaN inputs");
    }

    #[test]
    fn test_fisher_too_small_data() {
        let high = [10.0];
        let low = [5.0];
        let params = FisherParams { period: Some(9) };
        let input = FisherInput::from_slices(&high, &low, params);
        let result = fisher(&input);
        assert!(
            result.is_err(),
            "Expected an error for data smaller than period"
        );
    }

    #[test]
    fn test_fisher_reinput() {
        let high = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let low = [5.0, 7.0, 9.0, 10.0, 13.0, 15.0];
        let first_params = FisherParams { period: Some(3) };
        let first_input = FisherInput::from_slices(&high, &low, first_params);
        let first_result = fisher(&first_input).expect("Failed first fisher transform");

        let second_params = FisherParams { period: Some(3) };
        let second_input =
            FisherInput::from_slices(&first_result.fisher, &first_result.signal, second_params);
        let second_result = fisher(&second_input).expect("Failed second fisher transform");

        assert_eq!(first_result.fisher.len(), second_result.fisher.len());
        assert_eq!(first_result.signal.len(), second_result.signal.len());
    }

    #[test]
    fn test_fisher_default_input() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = FisherInput::with_default_candles(&candles);
        let result = fisher(&input).expect("Failed default fisher transform");
        assert_eq!(result.fisher.len(), candles.close.len());
        assert_eq!(result.signal.len(), candles.close.len());
    }
}
