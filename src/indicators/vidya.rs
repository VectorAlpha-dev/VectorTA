/// # Variable Index Dynamic Average (VIDYA)
///
/// VIDYA is a moving average calculation that dynamically adjusts its smoothing factor
/// based on the ratio of short-term to long-term standard deviations. This allows the
/// average to become more responsive to volatility changes while still providing a
/// smoothed signal.
///
/// ## Parameters
/// - **short_period**: The short look-back period for standard deviation. Defaults to 2.
/// - **long_period**: The long look-back period for standard deviation. Defaults to 5.
/// - **alpha**: A smoothing factor between 0.0 and 1.0. Defaults to 0.2.
///
/// ## Errors
/// - **EmptyData**: vidya: Input data slice is empty.
/// - **AllValuesNaN**: vidya: All input data values are `NaN`.
/// - **NotEnoughValidData**: vidya: Fewer than `long_period` valid data points remain
///   after the first valid index.
/// - **InvalidParameters**: vidya: Invalid `short_period`, `long_period`, or `alpha`.
///
/// ## Returns
/// - **`Ok(VidyaOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the computation can start.
/// - **`Err(VidyaError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum VidyaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VidyaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VidyaParams {
    pub short_period: Option<usize>,
    pub long_period: Option<usize>,
    pub alpha: Option<f64>,
}

impl Default for VidyaParams {
    fn default() -> Self {
        Self {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VidyaInput<'a> {
    pub data: VidyaData<'a>,
    pub params: VidyaParams,
}

impl<'a> VidyaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: VidyaParams) -> Self {
        Self {
            data: VidyaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: VidyaParams) -> Self {
        Self {
            data: VidyaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VidyaData::Candles {
                candles,
                source: "close",
            },
            params: VidyaParams::default(),
        }
    }

    pub fn get_short_period(&self) -> usize {
        self.params
            .short_period
            .unwrap_or_else(|| VidyaParams::default().short_period.unwrap())
    }

    pub fn get_long_period(&self) -> usize {
        self.params
            .long_period
            .unwrap_or_else(|| VidyaParams::default().long_period.unwrap())
    }

    pub fn get_alpha(&self) -> f64 {
        self.params
            .alpha
            .unwrap_or_else(|| VidyaParams::default().alpha.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VidyaError {
    #[error("vidya: Empty data provided.")]
    EmptyData,
    #[error("vidya: All values are NaN.")]
    AllValuesNaN,
    #[error("vidya: Not enough valid data to compute VIDYA. needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "vidya: Invalid short/long period or alpha. short={short}, long={long}, alpha={alpha}"
    )]
    InvalidParameters {
        short: usize,
        long: usize,
        alpha: f64,
    },
}

#[inline]
pub fn vidya(input: &VidyaInput) -> Result<VidyaOutput, VidyaError> {
    let data: &[f64] = match &input.data {
        VidyaData::Candles { candles, source } => source_type(candles, source),
        VidyaData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(VidyaError::EmptyData);
    }

    let short_period = input.get_short_period();
    let long_period = input.get_long_period();
    let alpha = input.get_alpha();

    if short_period < 1
        || long_period < short_period
        || long_period < 2
        || alpha < 0.0
        || alpha > 1.0
        || long_period > data.len()
    {
        return Err(VidyaError::InvalidParameters {
            short: short_period,
            long: long_period,
            alpha,
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(VidyaError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < long_period {
        return Err(VidyaError::NotEnoughValidData {
            needed: long_period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut output = vec![f64::NAN; data.len()];
    let mut long_sum = 0.0;
    let mut long_sum2 = 0.0;
    let mut short_sum = 0.0;
    let mut short_sum2 = 0.0;

    for i in first_valid_idx..(first_valid_idx + long_period) {
        long_sum += data[i];
        long_sum2 += data[i] * data[i];
        if i >= (first_valid_idx + long_period - short_period) {
            short_sum += data[i];
            short_sum2 += data[i] * data[i];
        }
    }

    let mut val = data[first_valid_idx + long_period - 2];
    output[first_valid_idx + long_period - 2] = val;

    if first_valid_idx + long_period - 1 < data.len() {
        let sp = short_period as f64;
        let lp = long_period as f64;
        let short_div = 1.0 / sp;
        let long_div = 1.0 / lp;
        let short_stddev =
            (short_sum2 * short_div - (short_sum * short_div) * (short_sum * short_div)).sqrt();
        let long_stddev =
            (long_sum2 * long_div - (long_sum * long_div) * (long_sum * long_div)).sqrt();
        let mut k = short_stddev / long_stddev;
        if k.is_nan() {
            k = 0.0;
        }
        k *= alpha;
        val = (data[first_valid_idx + long_period - 1] - val) * k + val;
        output[first_valid_idx + long_period - 1] = val;
    }

    for i in (first_valid_idx + long_period)..data.len() {
        long_sum += data[i];
        long_sum2 += data[i] * data[i];
        short_sum += data[i];
        short_sum2 += data[i] * data[i];

        let remove_long = i - long_period;
        let remove_short = i - short_period;
        long_sum -= data[remove_long];
        long_sum2 -= data[remove_long] * data[remove_long];
        short_sum -= data[remove_short];
        short_sum2 -= data[remove_short] * data[remove_short];

        let sp = short_period as f64;
        let lp = long_period as f64;
        let short_div = 1.0 / sp;
        let long_div = 1.0 / lp;
        let short_stddev =
            (short_sum2 * short_div - (short_sum * short_div) * (short_sum * short_div)).sqrt();
        let long_stddev =
            (long_sum2 * long_div - (long_sum * long_div) * (long_sum * long_div)).sqrt();
        let mut k = short_stddev / long_stddev;
        if k.is_nan() {
            k = 0.0;
        }
        k *= alpha;
        val = (data[i] - val) * k + val;
        output[i] = val;
    }

    Ok(VidyaOutput { values: output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_vidya_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = VidyaParams {
            short_period: None,
            long_period: Some(10),
            alpha: None,
        };
        let input_default = VidyaInput::from_candles(&candles, "close", default_params);
        let output_default = vidya(&input_default).expect("Failed VIDYA with partial params");
        assert_eq!(output_default.values.len(), candles.close.len());
    }

    #[test]
    fn test_vidya_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = VidyaParams {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let input = VidyaInput::from_candles(&candles, "close", params);
        let vidya_result = vidya(&input).expect("Failed to calculate VIDYA");

        assert_eq!(
            vidya_result.values.len(),
            close_prices.len(),
            "VIDYA length mismatch"
        );

        if vidya_result.values.len() >= 5 {
            let expected_last_five = [
                59553.42785306692,
                59503.60445032524,
                59451.72283651444,
                59413.222561244685,
                59239.716526894175,
            ];
            let start_index = vidya_result.values.len() - 5;
            let result_last_five = &vidya_result.values[start_index..];
            for (i, &value) in result_last_five.iter().enumerate() {
                let expected_value = expected_last_five[i];
                assert!(
                    (value - expected_value).abs() < 1e-1,
                    "VIDYA mismatch at index {}: expected {}, got {}",
                    i,
                    expected_value,
                    value
                );
            }
        }
    }

    #[test]
    fn test_vidya_params_with_default_params() {
        let default_params = VidyaParams::default();
        assert_eq!(default_params.short_period, Some(2));
        assert_eq!(default_params.long_period, Some(5));
        assert_eq!(default_params.alpha, Some(0.2));
    }

    #[test]
    fn test_vidya_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = VidyaInput::with_default_candles(&candles);
        match input.data {
            VidyaData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected VidyaData::Candles variant"),
        }
    }

    #[test]
    fn test_vidya_with_invalid_params() {
        let data = [10.0, 20.0, 30.0];
        let params = VidyaParams {
            short_period: Some(0),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let input = VidyaInput::from_slice(&data, params);
        let result = vidya(&input);
        assert!(
            result.is_err(),
            "Expected an error for invalid short period"
        );
    }

    #[test]
    fn test_vidya_exceeding_data_length() {
        let data = [10.0, 20.0, 30.0];
        let params = VidyaParams {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let input = VidyaInput::from_slice(&data, params);
        let result = vidya(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_vidya_very_small_data_set() {
        let data = [42.0, 43.0];
        let params = VidyaParams {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let input = VidyaInput::from_slice(&data, params);
        let result = vidya(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than long period"
        );
    }

    #[test]
    fn test_vidya_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = VidyaParams {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let first_input = VidyaInput::from_candles(&candles, "close", first_params);
        let first_result = vidya(&first_input).expect("Failed to calculate first VIDYA");

        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First VIDYA output length mismatch"
        );

        let second_params = VidyaParams {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let second_input = VidyaInput::from_slice(&first_result.values, second_params);
        let second_result = vidya(&second_input).expect("Failed to calculate second VIDYA");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second VIDYA output length mismatch"
        );
    }

    #[test]
    fn test_vidya_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = &candles.close;

        let params = VidyaParams {
            short_period: Some(2),
            long_period: Some(5),
            alpha: Some(0.2),
        };
        let input = VidyaInput::from_candles(&candles, "close", params);
        let vidya_result = vidya(&input).expect("Failed to calculate VIDYA");
        assert_eq!(vidya_result.values.len(), close_prices.len());

        if vidya_result.values.len() > 10 {
            for i in 10..vidya_result.values.len() {
                assert!(!vidya_result.values[i].is_nan());
            }
        }
    }
}
