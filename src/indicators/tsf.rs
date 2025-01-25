/// # Time Series Forecast (TSF)
///
/// TSF is based on linear regression, attempting to fit a line through the previous `period` data points.
/// The line is used to forecast the next value, effectively returning `b + m * period` for each point,
/// where `b` is the intercept and `m` is the slope of the regression line.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: tsf: Input data slice is empty.
/// - **InvalidPeriod**: tsf: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: tsf: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: tsf: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(TsfOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the forecast window is filled.
/// - **`Err(TsfError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum TsfData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct TsfOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TsfParams {
    pub period: Option<usize>,
}

impl Default for TsfParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct TsfInput<'a> {
    pub data: TsfData<'a>,
    pub params: TsfParams,
}

impl<'a> TsfInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: TsfParams) -> Self {
        Self {
            data: TsfData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: TsfParams) -> Self {
        Self {
            data: TsfData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: TsfData::Candles {
                candles,
                source: "close",
            },
            params: TsfParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| TsfParams::default().period.unwrap())
    }
}
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TsfError {
    #[error("tsf: Empty data provided for TSF.")]
    EmptyData,
    #[error("tsf: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("tsf: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("tsf: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn tsf(input: &TsfInput) -> Result<TsfOutput, TsfError> {
    let data: &[f64] = match &input.data {
        TsfData::Candles { candles, source } => source_type(candles, source),
        TsfData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(TsfError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(TsfError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(TsfError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(TsfError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut tsf_values = vec![f64::NAN; data.len()];

    let sum_x = (0..period).map(|x| x as f64).sum::<f64>();
    let sum_x_sqr = (0..period).map(|x| (x as f64) * (x as f64)).sum::<f64>();
    let divisor = (period as f64 * sum_x_sqr) - (sum_x * sum_x);

    for i in (first_valid_idx + period - 1)..data.len() {
        let mut sum_xy = 0.0;
        let mut sum_y = 0.0;
        for j in 0..period {
            let val = data[i - j];
            sum_y += val;
            sum_xy += (j as f64) * val;
        }

        let m = ((period as f64) * sum_xy - sum_x * sum_y) / divisor;
        let b = (sum_y - m * sum_x) / (period as f64);
        tsf_values[i] = b + m * (period as f64);
    }

    Ok(TsfOutput { values: tsf_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_tsf_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = TsfParams { period: None };
        let input_default = TsfInput::from_candles(&candles, "close", default_params);
        let output_default = tsf(&input_default).expect("Failed TSF with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_20 = TsfParams { period: Some(20) };
        let input_period_20 = TsfInput::from_candles(&candles, "hl2", params_period_20);
        let output_period_20 =
            tsf(&input_period_20).expect("Failed TSF with period=20, source=hl2");
        assert_eq!(output_period_20.values.len(), candles.close.len());
    }

    #[test]
    #[ignore]
    fn test_tsf_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = TsfParams { period: Some(14) };
        let input = TsfInput::from_candles(&candles, "close", params);
        let tsf_result = tsf(&input).expect("Failed to calculate TSF");
        assert_eq!(tsf_result.values.len(), close_prices.len());

        let expected_last_five_tsf = [
            58846.945054945056,
            58818.83516483516,
            58854.57142857143,
            59083.846153846156,
            58962.25274725275,
        ];
        assert!(tsf_result.values.len() >= 5, "TSF length too short");
        let start_index = tsf_result.values.len() - 5;
        let result_last_five = &tsf_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five_tsf[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "TSF mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        for i in 0..13 {
            assert!(tsf_result.values[i].is_nan());
        }

        let default_input = TsfInput::with_default_candles(&candles);
        let default_result = tsf(&default_input).expect("Failed TSF defaults");
        assert_eq!(default_result.values.len(), close_prices.len());
    }

    #[test]
    fn test_tsf_params_with_default_params() {
        let default_params = TsfParams::default();
        assert_eq!(
            default_params.period,
            Some(14),
            "Expected period to be Some(14) in default parameters"
        );
    }

    #[test]
    fn test_tsf_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = TsfInput::with_default_candles(&candles);
        match input.data {
            TsfData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected TsfData::Candles variant"),
        }
    }

    #[test]
    fn test_tsf_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = TsfParams { period: Some(0) };
        let input = TsfInput::from_slice(&input_data, params);
        let result = tsf(&input);
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
    fn test_tsf_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = TsfParams { period: Some(10) };
        let input = TsfInput::from_slice(&input_data, params);
        let result = tsf(&input);
        assert!(result.is_err(), "Expected error for period > data.len()");
    }

    #[test]
    fn test_tsf_very_small_data_set() {
        let input_data = [42.0];
        let params = TsfParams { period: Some(9) };
        let input = TsfInput::from_slice(&input_data, params);
        let result = tsf(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_tsf_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = TsfParams { period: Some(14) };
        let first_input = TsfInput::from_candles(&candles, "close", first_params);
        let first_result = tsf(&first_input).expect("Failed to calculate first TSF");
        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First TSF output length mismatch"
        );

        let second_params = TsfParams { period: Some(14) };
        let second_input = TsfInput::from_slice(&first_result.values, second_params);
        let second_result = tsf(&second_input).expect("Failed to calculate second TSF");
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second TSF output length mismatch"
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
    fn test_tsf_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 14;
        let params = TsfParams {
            period: Some(period),
        };
        let input = TsfInput::from_candles(&candles, "close", params);
        let tsf_result = tsf(&input).expect("Failed to calculate TSF");
        assert_eq!(tsf_result.values.len(), close_prices.len());

        if tsf_result.values.len() > 240 {
            for i in 240..tsf_result.values.len() {
                assert!(
                    !tsf_result.values[i].is_nan(),
                    "Expected no NaN after index 240, but found NaN at index {}",
                    i
                );
            }
        }
    }
}
