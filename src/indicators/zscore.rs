use crate::indicators::deviation::{deviation, DevError, DevInput, DevParams};
use crate::indicators::moving_averages::ma::{ma, MaData};
/// # Z-Score (Zscore)
///
/// A statistical measurement that describes a value's relationship to the mean of a group of values,
/// measured in terms of standard deviations. A Z-Score of 0 indicates the value is identical to the mean,
/// while positive/negative Z-Scores indicate how many standard deviations above/below the mean the value is.
///
/// ## Parameters
/// - **period**: The window size (number of data points). Defaults to 14.
/// - **ma_type**: Type of moving average to use for the mean. Defaults to `"sma"`.
/// - **nbdev**: The multiplier for the standard/mean/median absolute deviation. Defaults to `1.0`.
/// - **devtype**: Which deviation function to use:
///   - `0` = Standard Deviation
///   - `1` = Mean Absolute Deviation
///   - `2` = Median Absolute Deviation
///   Defaults to `0`.
///
/// ## Errors
/// - **EmptyData**: zscore: Input data slice is empty.
/// - **InvalidPeriod**: zscore: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: zscore: Fewer than `period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: zscore: All input data values are `NaN`.
/// - **DevError**: zscore: Underlying error from the deviation function.
/// - **MaError**: zscore: Underlying error from the moving average function.
///
/// ## Returns
/// - **`Ok(ZscoreOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the required window is filled.
/// - **`Err(ZscoreError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ZscoreData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ZscoreOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ZscoreParams {
    pub period: Option<usize>,
    pub ma_type: Option<String>,
    pub nbdev: Option<f64>,
    pub devtype: Option<usize>,
}

impl Default for ZscoreParams {
    fn default() -> Self {
        Self {
            period: Some(14),
            ma_type: Some("sma".to_string()),
            nbdev: Some(1.0),
            devtype: Some(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZscoreInput<'a> {
    pub data: ZscoreData<'a>,
    pub params: ZscoreParams,
}

impl<'a> ZscoreInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: ZscoreParams) -> Self {
        Self {
            data: ZscoreData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: ZscoreParams) -> Self {
        Self {
            data: ZscoreData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: ZscoreData::Candles {
                candles,
                source: "close",
            },
            params: ZscoreParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| ZscoreParams::default().period.unwrap())
    }

    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .clone()
            .unwrap_or_else(|| ZscoreParams::default().ma_type.unwrap())
    }

    pub fn get_nbdev(&self) -> f64 {
        self.params
            .nbdev
            .unwrap_or_else(|| ZscoreParams::default().nbdev.unwrap())
    }

    pub fn get_devtype(&self) -> usize {
        self.params
            .devtype
            .unwrap_or_else(|| ZscoreParams::default().devtype.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum ZscoreError {
    #[error("zscore: Empty data provided for Zscore.")]
    EmptyData,
    #[error("zscore: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("zscore: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("zscore: All values are NaN.")]
    AllValuesNaN,
    #[error("zscore: DevError {0}")]
    DevError(#[from] DevError),
    #[error("zscore: MaError {0}")]
    MaError(String),
}

#[inline]
pub fn zscore(input: &ZscoreInput) -> Result<ZscoreOutput, ZscoreError> {
    let data: &[f64] = match &input.data {
        ZscoreData::Candles { candles, source } => source_type(candles, source),
        ZscoreData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(ZscoreError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(ZscoreError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(ZscoreError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(ZscoreError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let ma_type = input.get_ma_type();
    let nbdev = input.get_nbdev();
    let devtype = input.get_devtype();

    let means = ma(&ma_type, MaData::Slice(data), period)
        .map_err(|e: Box<dyn Error>| ZscoreError::MaError(e.to_string()))?;
    let dev_input = DevInput {
        data,
        params: DevParams {
            period: Some(period),
            devtype: Some(devtype),
        },
    };
    let mut sigmas = deviation(&dev_input).map_err(ZscoreError::DevError)?;
    for val in &mut sigmas {
        *val *= nbdev;
    }

    let mut zscore_values = vec![f64::NAN; data.len()];
    for i in first_valid_idx..data.len() {
        let offset = i - first_valid_idx;
        if offset < (period - 1) {
            continue;
        }
        let mean = means[offset];
        let sigma = sigmas[offset];
        let value = data[i];
        zscore_values[i] = if sigma == 0.0 || sigma.is_nan() {
            f64::NAN
        } else {
            (value - mean) / sigma
        };
    }

    Ok(ZscoreOutput {
        values: zscore_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_zscore_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = ZscoreParams {
            period: None,
            ma_type: None,
            nbdev: None,
            devtype: None,
        };
        let input_default = ZscoreInput::from_candles(&candles, "close", default_params);
        let output_default = zscore(&input_default).expect("Failed Zscore with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let custom_params = ZscoreParams {
            period: Some(20),
            ma_type: Some("sma".to_string()),
            nbdev: Some(2.0),
            devtype: Some(0),
        };
        let input_custom = ZscoreInput::from_candles(&candles, "hlc3", custom_params);
        let output_custom = zscore(&input_custom).expect("Failed Zscore fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    #[ignore]
    fn test_zscore_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = ZscoreParams {
            period: Some(14),
            ma_type: Some("sma".to_string()),
            nbdev: Some(1.0),
            devtype: Some(0),
        };
        let input = ZscoreInput::from_candles(&candles, "close", params);
        let zscore_result = zscore(&input).expect("Failed to calculate Zscore");
        assert_eq!(zscore_result.values.len(), close_prices.len());

        let expected_last_five = [
            -0.48296332772534434,
            -0.7213074913423706,
            -0.8458037396726564,
            -0.18072921072693846,
            -1.670775998772587,
        ];
        assert!(zscore_result.values.len() >= 5);
        let start_index = zscore_result.values.len() - 5;
        let result_last_five = &zscore_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (value - expected).abs() < 1e-6,
                "Zscore mismatch at index {}: expected {}, got {}",
                i,
                expected,
                value
            );
        }
    }

    #[test]
    fn test_zscore_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = ZscoreParams {
            period: Some(0),
            ma_type: Some("sma".to_string()),
            nbdev: Some(1.0),
            devtype: Some(0),
        };
        let input = ZscoreInput::from_slice(&input_data, params);
        let result = zscore(&input);
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
    fn test_zscore_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = ZscoreParams {
            period: Some(10),
            ma_type: Some("sma".to_string()),
            nbdev: Some(1.0),
            devtype: Some(0),
        };
        let input = ZscoreInput::from_slice(&input_data, params);
        let result = zscore(&input);
        assert!(result.is_err(), "Expected an error for period > data.len()");
    }

    #[test]
    fn test_zscore_very_small_data_set() {
        let input_data = [42.0];
        let params = ZscoreParams {
            period: Some(14),
            ma_type: Some("sma".to_string()),
            nbdev: Some(1.0),
            devtype: Some(0),
        };
        let input = ZscoreInput::from_slice(&input_data, params);
        let result = zscore(&input);
        assert!(
            result.is_err(),
            "Expected error for data smaller than period"
        );
    }

    #[test]
    fn test_zscore_all_nan() {
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = ZscoreParams::default();
        let input = ZscoreInput::from_slice(&input_data, params);
        let result = zscore(&input);
        assert!(result.is_err(), "Expected error when all values are NaN");
    }

    #[test]
    fn test_zscore_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = ZscoreInput::with_default_candles(&candles);
        match input.data {
            ZscoreData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected default source to be 'close'");
            }
            _ => panic!("Expected ZscoreData::Candles variant"),
        }
    }
}
