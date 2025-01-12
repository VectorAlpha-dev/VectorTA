/// # Mean Absolute Deviation (MeanAd)
///
/// A two-pass rolling statistic. First computes a rolling average of the last `period` values,
/// then computes the rolling average of the absolute deviation from that rolling average. This
/// process helps quantify the variability or dispersion in the data.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 5.
///
/// ## Errors
/// - **EmptyData**: mean_ad: Input data slice is empty.
/// - **InvalidPeriod**: mean_ad: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: mean_ad: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: mean_ad: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(MeanAdOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until enough data points are accumulated for both passes.
/// - **`Err(MeanAdError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MeanAdData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct MeanAdOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MeanAdParams {
    pub period: Option<usize>,
}

impl Default for MeanAdParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct MeanAdInput<'a> {
    pub data: MeanAdData<'a>,
    pub params: MeanAdParams,
}

impl<'a> MeanAdInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: MeanAdParams) -> Self {
        Self {
            data: MeanAdData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: MeanAdParams) -> Self {
        Self {
            data: MeanAdData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MeanAdData::Candles {
                candles,
                source: "close",
            },
            params: MeanAdParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| MeanAdParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum MeanAdError {
    #[error("mean_ad: Empty data provided.")]
    EmptyData,
    #[error("mean_ad: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("mean_ad: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("mean_ad: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn mean_ad(input: &MeanAdInput) -> Result<MeanAdOutput, MeanAdError> {
    let data: &[f64] = match &input.data {
        MeanAdData::Candles { candles, source } => source_type(candles, source),
        MeanAdData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(MeanAdError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(MeanAdError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(MeanAdError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(MeanAdError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let rolling_mean_data = rolling_mean(data, period, first_valid_idx);

    let first_valid_idx_rm = match rolling_mean_data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => {
            return Err(MeanAdError::NotEnoughValidData {
                needed: period,
                valid: 0,
            });
        }
    };

    let mut abs_diff = vec![f64::NAN; data.len()];
    for i in first_valid_idx_rm..data.len() {
        if !rolling_mean_data[i].is_nan() {
            abs_diff[i] = (data[i] - rolling_mean_data[i]).abs();
        }
    }

    let mad_values = rolling_mean(&abs_diff, period, first_valid_idx_rm);

    Ok(MeanAdOutput { values: mad_values })
}

fn rolling_mean(values: &[f64], period: usize, start_idx: usize) -> Vec<f64> {
    let mut output = vec![f64::NAN; values.len()];

    if start_idx + period > values.len() {
        return output;
    }

    let mut sum = 0.0;
    for &v in &values[start_idx..(start_idx + period)] {
        sum += v;
    }
    output[start_idx + period - 1] = sum / (period as f64);

    for i in (start_idx + period)..values.len() {
        sum += values[i] - values[i - period];
        output[i] = sum / (period as f64);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_mean_ad_with_default_params() {
        let default_params = MeanAdParams::default();
        assert_eq!(default_params.period, Some(5));
    }

    #[test]
    fn test_mean_ad_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = MeanAdInput::with_default_candles(&candles);
        match input.data {
            MeanAdData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected MeanAdData::Candles"),
        }
    }

    #[test]
    fn test_mean_ad_zero_period() {
        let data = [10.0, 20.0, 30.0];
        let params = MeanAdParams { period: Some(0) };
        let input = MeanAdInput::from_slice(&data, params);
        let result = mean_ad(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid period"));
        }
    }

    #[test]
    fn test_mean_ad_period_exceeds_length() {
        let data = [10.0, 20.0, 30.0];
        let params = MeanAdParams { period: Some(10) };
        let input = MeanAdInput::from_slice(&data, params);
        let result = mean_ad(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_mean_ad_all_nan() {
        let data = [f64::NAN, f64::NAN];
        let params = MeanAdParams { period: Some(2) };
        let input = MeanAdInput::from_slice(&data, params);
        let result = mean_ad(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
    }

    #[test]
    fn test_mean_ad_not_enough_valid_data() {
        let data = [f64::NAN, 10.0, f64::NAN];
        let params = MeanAdParams { period: Some(3) };
        let input = MeanAdInput::from_slice(&data, params);
        let result = mean_ad(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Not enough valid data"));
        }
    }

    #[test]
    fn test_mean_ad_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = MeanAdParams { period: Some(5) };
        let input = MeanAdInput::from_candles(&candles, "hl2", params);
        let result = mean_ad(&input).expect("Failed to calculate MeanAd");
        assert_eq!(result.values.len(), candles.close.len());

        let expected_last_five = [
            199.71999999999971,
            104.14000000000087,
            133.4,
            100.54000000000087,
            117.98000000000029,
        ];
        assert!(result.values.len() >= 5);
        let start_index = result.values.len() - 5;
        let actual_last_five = &result.values[start_index..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (val - expected).abs() < 1e-1,
                "MeanAd mismatch at index {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }
    }

    #[test]
    fn test_mean_ad_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = MeanAdParams { period: Some(5) };
        let input = MeanAdInput::from_candles(&candles, "close", params);
        let first_result = mean_ad(&input).expect("Failed to calculate first MeanAd");
        assert_eq!(first_result.values.len(), candles.close.len());
        let params2 = MeanAdParams { period: Some(3) };
        let second_input = MeanAdInput::from_slice(&first_result.values, params2);
        let second_result = mean_ad(&second_input).expect("Failed to calculate second MeanAd");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }

    #[test]
    fn test_mean_ad_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = MeanAdParams { period: Some(5) };
        let input = MeanAdInput::from_candles(&candles, "close", params);
        let result = mean_ad(&input).expect("Failed to calculate MeanAd");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan(), "Unexpected NaN at index {}", i);
            }
        }
    }
}
