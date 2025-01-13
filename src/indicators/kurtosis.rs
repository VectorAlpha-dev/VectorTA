/// # Kurtosis
///
/// Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued
/// random variable. We compute the sample values over a sliding window of size `period`,
/// using the uncorrected moment-based formula for excess kurtosis:
///
/// \[ k = \frac{\frac{1}{n}\sum\limits_{i=1}^{n}(x_i - \mu)^4}{\left(\frac{1}{n}\sum\limits_{i=1}^{n}(x_i - \mu)^2\right)^2} - 3 \]
///
/// where `n = period`, `\mu` is the mean of the window, and `x_i` are the values in the window.
/// If any value in the window is `NaN`, the kurtosis for that window is `NaN`.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: kurtosis: Input data slice is empty.
/// - **InvalidPeriod**: kurtosis: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: kurtosis: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: kurtosis: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(KurtosisOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the moving window is filled.
/// - **`Err(KurtosisError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum KurtosisData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct KurtosisOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct KurtosisParams {
    pub period: Option<usize>,
}

impl Default for KurtosisParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct KurtosisInput<'a> {
    pub data: KurtosisData<'a>,
    pub params: KurtosisParams,
}

impl<'a> KurtosisInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: KurtosisParams) -> Self {
        Self {
            data: KurtosisData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: KurtosisParams) -> Self {
        Self {
            data: KurtosisData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: KurtosisData::Candles {
                candles,
                source: "hl2",
            },
            params: KurtosisParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| KurtosisParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum KurtosisError {
    #[error("kurtosis: Empty data provided.")]
    EmptyData,
    #[error("kurtosis: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("kurtosis: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("kurtosis: All values are NaN.")]
    AllValuesNaN,
}
#[inline]
pub fn kurtosis(input: &KurtosisInput) -> Result<KurtosisOutput, KurtosisError> {
    let data: &[f64] = match &input.data {
        KurtosisData::Candles { candles, source } => source_type(candles, source),
        KurtosisData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(KurtosisError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(KurtosisError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(KurtosisError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(KurtosisError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut kurtosis_values = vec![f64::NAN; data.len()];

    for i in (first_valid_idx + period - 1)..data.len() {
        let start_idx = i + 1 - period;
        let window = &data[start_idx..=i];

        if window.iter().any(|x| x.is_nan()) {
            kurtosis_values[i] = f64::NAN;
            continue;
        }

        let n = window.len() as f64;
        let mean = window.iter().sum::<f64>() / n;
        let mut m2 = 0.0;
        let mut m4 = 0.0;
        for &val in window {
            let diff = val - mean;
            m2 += diff * diff;
            m4 += diff.powi(4);
        }
        m2 /= n;
        m4 /= n;

        if m2.abs() < f64::EPSILON {
            kurtosis_values[i] = f64::NAN;
        } else {
            kurtosis_values[i] = (m4 / (m2 * m2)) - 3.0;
        }
    }

    Ok(KurtosisOutput {
        values: kurtosis_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_kurtosis_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = KurtosisParams { period: None };
        let input_default = KurtosisInput::from_candles(&candles, "close", default_params);
        let output_default = kurtosis(&input_default).expect("Failed kurtosis with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_7 = KurtosisParams { period: Some(7) };
        let input_period_7 = KurtosisInput::from_candles(&candles, "hl2", params_period_7);
        let output_period_7 = kurtosis(&input_period_7).expect("Failed kurtosis with period=7");
        assert_eq!(output_period_7.values.len(), candles.close.len());

        let params_custom = KurtosisParams { period: Some(10) };
        let input_custom = KurtosisInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom = kurtosis(&input_custom).expect("Failed kurtosis fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_kurtosis_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let hl2 = &candles.hl2;

        let params = KurtosisParams { period: Some(5) };
        let input = KurtosisInput::from_candles(&candles, "hl2", params);
        let kurtosis_result = kurtosis(&input).expect("Failed to calculate Kurtosis");

        assert_eq!(
            kurtosis_result.values.len(),
            hl2.len(),
            "Kurtosis length mismatch"
        );

        let expected_last_five = [
            -0.5438903789933454,
            -1.6848139264816433,
            -1.6331336745945797,
            -0.6130805596586351,
            -0.027802601135927585,
        ];
        assert!(
            kurtosis_result.values.len() >= 5,
            "Kurtosis length too short"
        );

        let start_index = kurtosis_result.values.len() - 5;
        let result_last_five = &kurtosis_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "Kurtosis mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }

        let period: usize = 5;
        for i in 0..(period - 1) {
            assert!(
                kurtosis_result.values[i].is_nan(),
                "Expected NaN at index {}",
                i
            );
        }

        let default_input = KurtosisInput::with_default_candles(&candles);
        let default_kurtosis_result =
            kurtosis(&default_input).expect("Failed to calculate defaults");
        assert_eq!(default_kurtosis_result.values.len(), hl2.len());
    }

    #[test]
    fn test_kurtosis_params_with_default_params() {
        let default_params = KurtosisParams::default();
        assert_eq!(
            default_params.period,
            Some(5),
            "Expected period to be Some(5) in default parameters"
        );
    }

    #[test]
    fn test_kurtosis_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = KurtosisInput::with_default_candles(&candles);
        match input.data {
            KurtosisData::Candles { source, .. } => {
                assert_eq!(source, "hl2", "Expected default source to be 'hl2'");
            }
            _ => panic!("Expected KurtosisData::Candles variant"),
        }
    }

    #[test]
    fn test_kurtosis_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = KurtosisParams { period: Some(0) };
        let input = KurtosisInput::from_slice(&input_data, params);

        let result = kurtosis(&input);
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
    fn test_kurtosis_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = KurtosisParams { period: Some(10) };
        let input = KurtosisInput::from_slice(&input_data, params);

        let result = kurtosis(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_kurtosis_very_small_data_set() {
        let input_data = [42.0];
        let params = KurtosisParams { period: Some(5) };
        let input = KurtosisInput::from_slice(&input_data, params);

        let result = kurtosis(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_kurtosis_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = KurtosisParams { period: Some(5) };
        let first_input = KurtosisInput::from_candles(&candles, "close", first_params);
        let first_result = kurtosis(&first_input).expect("Failed to calculate first Kurtosis");

        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First Kurtosis output length mismatch"
        );

        let second_params = KurtosisParams { period: Some(5) };
        let second_input = KurtosisInput::from_slice(&first_result.values, second_params);
        let second_result = kurtosis(&second_input).expect("Failed to calculate second Kurtosis");

        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second Kurtosis output length mismatch"
        );

        for i in (5 * 2 - 1)..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index {}",
                i
            );
        }
    }

    #[test]
    fn test_kurtosis_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let hl2 = &candles.hl2;

        let period = 5;
        let params = KurtosisParams {
            period: Some(period),
        };
        let input = KurtosisInput::from_candles(&candles, "hl2", params);
        let kurtosis_result = kurtosis(&input).expect("Failed to calculate Kurtosis");

        assert_eq!(
            kurtosis_result.values.len(),
            hl2.len(),
            "Kurtosis length mismatch"
        );

        if kurtosis_result.values.len() > 50 {
            for i in 50..kurtosis_result.values.len() {
                assert!(
                    !kurtosis_result.values[i].is_nan(),
                    "Expected no NaN after index {}, but found NaN",
                    i
                );
            }
        }
    }
}
