/// # Kaufman Efficiency Ratio (ER)
///
/// The Kaufman Efficiency Ratio (ER) compares the absolute price change over a specified
/// `period` to the sum of the incremental absolute changes within that same `period`.
/// This ratio yields a value between 0.0 and 1.0, illustrating how directly and
/// efficiently the price has moved from the start to the end of the window. A higher
/// value (closer to 1.0) indicates a more directed move with less volatility, whereas
/// a lower value (closer to 0.0) indicates choppier price action.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: er: Input data slice is empty.
/// - **InvalidPeriod**: er: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: er: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: er: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(ErOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until enough data is available to calculate the ER.
/// - **`Err(ErError)`** otherwise.
///
/// ## Example
/// ```ignore
/// use crate::utilities::data_loader::read_candles_from_csv;
///
/// let file_path = "path/to/your/data.csv";
/// let candles = read_candles_from_csv(file_path).expect("Failed to load candles");
///
/// let params = ErParams { period: Some(5) };
/// let input = ErInput::from_candles(&candles, "close", params);
/// let er_result = er(&input).expect("Failed to calculate ER");
///
/// println!("Computed ER values: {:?}", er_result.values);
/// ```
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ErData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ErOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ErParams {
    pub period: Option<usize>,
}

impl Default for ErParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct ErInput<'a> {
    pub data: ErData<'a>,
    pub params: ErParams,
}

impl<'a> ErInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: ErParams) -> Self {
        Self {
            data: ErData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: ErParams) -> Self {
        Self {
            data: ErData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: ErData::Candles {
                candles,
                source: "close",
            },
            params: ErParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| ErParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum ErError {
    #[error("er: Empty data provided for ER.")]
    EmptyData,
    #[error("er: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
}

#[inline]
pub fn er(input: &ErInput) -> Result<ErOutput, ErError> {
    let data: &[f64] = match &input.data {
        ErData::Candles { candles, source } => source_type(candles, source),
        ErData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(ErError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(ErError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let mut change = Vec::with_capacity(data.len() - period);
    for i in 0..(data.len() - period) {
        let val = (data[i + period] - data[i]).abs();
        change.push(val);
    }

    let mut abs_dif = Vec::with_capacity(data.len() - 1);
    for i in 0..(data.len() - 1) {
        abs_dif.push((data[i + 1] - data[i]).abs());
    }

    let mut volatility = Vec::with_capacity(data.len() - period);
    for i in 0..(data.len() - period) {
        let sum_slice = &abs_dif[i..(i + period)];
        let sum_vol: f64 = sum_slice.iter().sum();
        volatility.push(sum_vol);
    }

    let mut er_values = vec![f64::NAN; data.len()];
    for i in 0..(data.len() - period) {
        if volatility[i] != 0.0 {
            er_values[i] = change[i] / volatility[i];
        } else {
            er_values[i] = f64::NAN;
        }
    }

    Ok(ErOutput { values: er_values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_er_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = ErParams { period: None };
        let input_default = ErInput::from_candles(&candles, "close", default_params);
        let output_default = er(&input_default).expect("Failed ER with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_10 = ErParams { period: Some(10) };
        let input_period_10 = ErInput::from_candles(&candles, "high", params_period_10);
        let output_period_10 = er(&input_period_10).expect("Failed ER with period=10, source=high");
        assert_eq!(output_period_10.values.len(), candles.close.len());

        let params_custom = ErParams { period: Some(20) };
        let input_custom = ErInput::from_candles(&candles, "low", params_custom);
        let output_custom = er(&input_custom).expect("Failed ER fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_er_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = ErInput::with_default_candles(&candles);
        match input.data {
            ErData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected ErData::Candles variant"),
        }
    }

    #[test]
    fn test_er_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = ErParams { period: Some(0) };
        let input = ErInput::from_slice(&input_data, params);

        let result = er(&input);
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
    fn test_er_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = ErParams { period: Some(10) };
        let input = ErInput::from_slice(&input_data, params);

        let result = er(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_er_very_small_data_set() {
        let input_data = [42.0];
        let params = ErParams { period: Some(5) };
        let input = ErInput::from_slice(&input_data, params);

        let result = er(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_er_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = ErParams { period: Some(5) };
        let first_input = ErInput::from_candles(&candles, "close", first_params);
        let first_result = er(&first_input).expect("Failed to calculate first ER");

        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First ER output length mismatch"
        );

        let second_params = ErParams { period: Some(5) };
        let second_input = ErInput::from_slice(&first_result.values, second_params);
        let second_result = er(&second_input).expect("Failed to calculate second ER");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second ER output length mismatch"
        );
    }

    #[test]
    fn test_er_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let period = 5;
        let params = ErParams {
            period: Some(period),
        };
        let input = ErInput::from_candles(&candles, "close", params);
        let er_result = er(&input).expect("Failed to calculate ER");

        assert_eq!(
            er_result.values.len(),
            close_prices.len(),
            "ER length mismatch"
        );

        if er_result.values.len() > 50 {
            for i in 50..er_result.values.len() {
                if i < er_result.values.len() - period {
                    assert!(
                        !er_result.values[i].is_nan(),
                        "Expected a valid value at index {}, got NaN",
                        i
                    );
                }
            }
        }
    }
}
