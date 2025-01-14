/// # Momentum (MOM)
///
/// MOM measures the amount that a security’s price has changed over a given time span.
/// It is calculated by subtracting the previous price (from the chosen period) from the
/// current price, i.e., `momentum[i] = data[i] - data[i - period]`.
///
/// ## Parameters
/// - **period**: The lookback window size (number of data points). Defaults to 10.
///
/// ## Errors
/// - **EmptyData**: mom: Input data slice is empty.
/// - **InvalidPeriod**: mom: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: mom: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: mom: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(MomOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the momentum window is filled.
/// - **`Err(MomError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MomData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MomOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MomParams {
    pub period: Option<usize>,
}

impl Default for MomParams {
    fn default() -> Self {
        Self { period: Some(10) }
    }
}

#[derive(Debug, Clone)]
pub struct MomInput<'a> {
    pub data: MomData<'a>,
    pub params: MomParams,
}

impl<'a> MomInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: MomParams) -> Self {
        Self {
            data: MomData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: MomParams) -> Self {
        Self {
            data: MomData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MomData::Candles {
                candles,
                source: "close",
            },
            params: MomParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| MomParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum MomError {
    #[error("mom: Empty data provided for Momentum.")]
    EmptyData,
    #[error("mom: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("mom: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("mom: All values are NaN.")]
    AllValuesNaN,
}

#[inline(always)]
pub fn mom(input: &MomInput) -> Result<MomOutput, MomError> {
    let data: &[f64] = match &input.data {
        MomData::Candles { candles, source } => source_type(candles, source),
        MomData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(MomError::EmptyData);
    }

    let period = input.get_period();
    let data_len = data.len();
    if period == 0 || period > data_len {
        return Err(MomError::InvalidPeriod { period, data_len });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(MomError::AllValuesNaN),
    };

    let valid_count = data_len - first_valid_idx;
    if valid_count < period {
        return Err(MomError::NotEnoughValidData {
            needed: period,
            valid: valid_count,
        });
    }

    let mut mom_values = vec![f64::NAN; data_len];
    let offset = first_valid_idx + period;
    for i in offset..data_len {
        mom_values[i] = data[i] - data[i - period];
    }

    Ok(MomOutput { values: mom_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_mom_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = MomParams { period: None };
        let input_default = MomInput::from_candles(&candles, "close", default_params);
        let output_default = mom(&input_default).expect("Failed MOM with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_14 = MomParams { period: Some(14) };
        let input_period_14 = MomInput::from_candles(&candles, "high", params_period_14);
        let output_period_14 =
            mom(&input_period_14).expect("Failed MOM with period=14, source=high");
        assert_eq!(output_period_14.values.len(), candles.close.len());

        let params_custom = MomParams { period: Some(20) };
        let input_custom = MomInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = mom(&input_custom).expect("Failed MOM fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_mom_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = MomParams { period: Some(10) };
        let input = MomInput::from_candles(&candles, "close", params);
        let mom_result = mom(&input).expect("Failed to calculate MOM");

        assert_eq!(
            mom_result.values.len(),
            close_prices.len(),
            "MOM length mismatch"
        );

        let expected_last_five_mom = [-134.0, -331.0, -194.0, -294.0, -896.0];
        assert!(mom_result.values.len() >= 5, "MOM length too short");
        let start_index = mom_result.values.len() - 5;
        let result_last_five_mom = &mom_result.values[start_index..];
        for (i, &value) in result_last_five_mom.iter().enumerate() {
            let expected_value = expected_last_five_mom[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "MOM mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        for i in 0..10 {
            assert!(mom_result.values[i].is_nan());
        }

        let default_input = MomInput::with_default_candles(&candles);
        let default_mom_result = mom(&default_input).expect("Failed to calculate MOM defaults");
        assert_eq!(default_mom_result.values.len(), close_prices.len());
    }

    #[test]
    fn test_mom_params_with_default_params() {
        let default_params = MomParams::default();
        assert_eq!(
            default_params.period,
            Some(10),
            "Expected period to be Some(10) in default parameters"
        );
    }

    #[test]
    fn test_mom_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = MomInput::with_default_candles(&candles);
        match input.data {
            MomData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected MomData::Candles variant"),
        }
    }

    #[test]
    fn test_mom_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = MomParams { period: Some(0) };
        let input = MomInput::from_slice(&input_data, params);

        let result = mom(&input);
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
    fn test_mom_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = MomParams { period: Some(10) };
        let input = MomInput::from_slice(&input_data, params);

        let result = mom(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_mom_very_small_data_set() {
        let input_data = [42.0];
        let params = MomParams { period: Some(9) };
        let input = MomInput::from_slice(&input_data, params);

        let result = mom(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_mom_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = MomParams { period: Some(14) };
        let first_input = MomInput::from_candles(&candles, "close", first_params);
        let first_result = mom(&first_input).expect("Failed to calculate first MOM");

        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First MOM output length mismatch"
        );

        let second_params = MomParams { period: Some(14) };
        let second_input = MomInput::from_slice(&first_result.values, second_params);
        let second_result = mom(&second_input).expect("Failed to calculate second MOM");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second MOM output length mismatch"
        );

        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 28, but found NaN at index {}",
                i
            );
        }
    }

    #[test]
    fn test_mom_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 10;
        let params = MomParams {
            period: Some(period),
        };
        let input = MomInput::from_candles(&candles, "close", params);
        let mom_result = mom(&input).expect("Failed to calculate MOM");

        assert_eq!(
            mom_result.values.len(),
            close_prices.len(),
            "MOM length mismatch"
        );

        if mom_result.values.len() > 240 {
            for i in 240..mom_result.values.len() {
                assert!(
                    !mom_result.values[i].is_nan(),
                    "Expected no NaN after index 240, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
