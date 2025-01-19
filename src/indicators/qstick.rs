/// # Qstick
///
/// Qstick measures the average difference between the Close and Open over a specified period.
/// A positive Qstick indicates that, on average, the market closes above its open, while a
/// negative Qstick indicates the opposite.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: qstick: Input data slice is empty.
/// - **InvalidPeriod**: qstick: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: qstick: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: qstick: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(QstickOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the Qstick window is filled.
/// - **`Err(QstickError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum QstickData<'a> {
    Candles {
        candles: &'a Candles,
        open_source: &'a str,
        close_source: &'a str,
    },
    Slices {
        open: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct QstickOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct QstickParams {
    pub period: Option<usize>,
}

impl Default for QstickParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct QstickInput<'a> {
    pub data: QstickData<'a>,
    pub params: QstickParams,
}

impl<'a> QstickInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        open_source: &'a str,
        close_source: &'a str,
        params: QstickParams,
    ) -> Self {
        Self {
            data: QstickData::Candles {
                candles,
                open_source,
                close_source,
            },
            params,
        }
    }

    pub fn from_slices(open: &'a [f64], close: &'a [f64], params: QstickParams) -> Self {
        Self {
            data: QstickData::Slices { open, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: QstickData::Candles {
                candles,
                open_source: "open",
                close_source: "close",
            },
            params: QstickParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| QstickParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum QstickError {
    #[error("qstick: Empty data provided.")]
    EmptyData,
    #[error("qstick: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("qstick: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("qstick: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn qstick(input: &QstickInput) -> Result<QstickOutput, QstickError> {
    let (open, close) = match &input.data {
        QstickData::Candles {
            candles,
            open_source,
            close_source,
        } => {
            let open = source_type(candles, open_source);
            let close = source_type(candles, close_source);
            (open, close)
        }
        QstickData::Slices { open, close } => (*open, *close),
    };

    if open.is_empty() || close.is_empty() {
        return Err(QstickError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > open.len() || period > close.len() {
        return Err(QstickError::InvalidPeriod {
            period,
            data_len: open.len().min(close.len()),
        });
    }

    let data_len = open.len().min(close.len());
    let diff: Vec<f64> = open
        .iter()
        .zip(close.iter())
        .map(|(&o, &c)| c - o)
        .collect();

    let first_valid_idx = match diff.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(QstickError::AllValuesNaN),
    };

    if (data_len - first_valid_idx) < period {
        return Err(QstickError::NotEnoughValidData {
            needed: period,
            valid: data_len - first_valid_idx,
        });
    }

    let mut output_values = vec![f64::NAN; data_len];
    let mut sum = 0.0;
    for &value in diff[first_valid_idx..(first_valid_idx + period)].iter() {
        sum += value;
    }

    let inv_period = 1.0 / (period as f64);
    output_values[first_valid_idx + period - 1] = sum * inv_period;

    for i in (first_valid_idx + period)..data_len {
        sum += diff[i] - diff[i - period];
        output_values[i] = sum * inv_period;
    }

    Ok(QstickOutput {
        values: output_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_qstick_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = QstickParams { period: None };
        let input_default = QstickInput::from_candles(&candles, "open", "close", default_params);
        let output_default = qstick(&input_default).expect("Failed Qstick with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_7 = QstickParams { period: Some(7) };
        let input_period_7 = QstickInput::from_candles(&candles, "open", "close", params_period_7);
        let output_period_7 = qstick(&input_period_7).expect("Failed Qstick with period=7");
        assert_eq!(output_period_7.values.len(), candles.close.len());

        let params_custom = QstickParams { period: Some(10) };
        let input_custom = QstickInput::from_candles(&candles, "open", "close", params_custom);
        let output_custom = qstick(&input_custom).expect("Failed Qstick fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_qstick_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let open_prices = candles
            .select_candle_field("open")
            .expect("Failed to extract open prices");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = QstickParams { period: Some(5) };
        let input = QstickInput::from_candles(&candles, "open", "close", params);
        let qstick_result = qstick(&input).expect("Failed to calculate Qstick");
        assert_eq!(qstick_result.values.len(), open_prices.len());

        let expected_last_five_qstick = [219.4, 61.6, -51.8, -53.4, -123.2];
        assert!(qstick_result.values.len() >= 5);
        let start_index = qstick_result.values.len() - 5;
        let result_last_five = &qstick_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five_qstick[i];
            assert!(
                (value - expected_value).abs() < 1e-1,
                "Qstick mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period: usize = 5;
        for i in 0..(period - 1) {
            assert!(qstick_result.values[i].is_nan());
        }

        let default_input = QstickInput::with_default_candles(&candles);
        let default_result = qstick(&default_input).expect("Failed to calculate Qstick defaults");
        assert_eq!(default_result.values.len(), close_prices.len());
    }

    #[test]
    fn test_qstick_params_with_default_params() {
        let default_params = QstickParams::default();
        assert_eq!(
            default_params.period,
            Some(5),
            "Expected period=5 in default parameters"
        );
    }

    #[test]
    fn test_qstick_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = QstickInput::with_default_candles(&candles);
        match input.data {
            QstickData::Candles {
                open_source,
                close_source,
                ..
            } => {
                assert_eq!(
                    open_source, "open",
                    "Expected default open source to be 'open'"
                );
                assert_eq!(
                    close_source, "close",
                    "Expected default close source to be 'close'"
                );
            }
            _ => panic!("Expected QstickData::Candles variant"),
        }
    }

    #[test]
    fn test_qstick_with_zero_period() {
        let open_data = [10.0, 20.0, 30.0];
        let close_data = [15.0, 25.0, 35.0];
        let params = QstickParams { period: Some(0) };
        let input = QstickInput::from_slices(&open_data, &close_data, params);
        let result = qstick(&input);
        assert!(result.is_err(), "Expected an error for zero period");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Invalid period"),
                "Expected 'Invalid period' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_qstick_with_period_exceeding_data_length() {
        let open_data = [10.0, 20.0, 30.0];
        let close_data = [15.0, 25.0, 35.0];
        let params = QstickParams { period: Some(10) };
        let input = QstickInput::from_slices(&open_data, &close_data, params);
        let result = qstick(&input);
        assert!(
            result.is_err(),
            "Expected an error for period > data length"
        );
    }

    #[test]
    fn test_qstick_very_small_data_set() {
        let open_data = [50.0];
        let close_data = [55.0];
        let params = QstickParams { period: Some(5) };
        let input = QstickInput::from_slices(&open_data, &close_data, params);
        let result = qstick(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_qstick_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = QstickParams { period: Some(5) };
        let first_input = QstickInput::from_candles(&candles, "open", "close", first_params);
        let first_result = qstick(&first_input).expect("Failed to calculate Qstick first run");
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = QstickParams { period: Some(5) };
        let second_input =
            QstickInput::from_slices(&first_result.values, &first_result.values, second_params);
        let second_result = qstick(&second_input).expect("Failed to calculate Qstick second run");
        assert_eq!(second_result.values.len(), first_result.values.len());

        for i in 10..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 10, found NaN at index {}",
                i
            );
        }
    }

    #[test]
    fn test_qstick_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let period = 5;
        let params = QstickParams {
            period: Some(period),
        };
        let input = QstickInput::from_candles(&candles, "open", "close", params);
        let qstick_result = qstick(&input).expect("Failed to calculate Qstick");
        assert_eq!(qstick_result.values.len(), candles.close.len());

        if qstick_result.values.len() > 50 {
            for i in 50..qstick_result.values.len() {
                assert!(
                    !qstick_result.values[i].is_nan(),
                    "Expected no NaN after index 50, found NaN at index {}",
                    i
                );
            }
        }
    }
}
