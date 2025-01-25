use crate::indicators::deviation::{deviation, DevInput, DevParams};
use crate::indicators::moving_averages::ma::{ma, MaData};
/// # Variable Length Moving Average (VLMA)
///
/// A moving average whose period adapts based on how close the current value is
/// to certain deviation thresholds of a long-term average. The period is clamped
/// between `min_period` and `max_period`. This can be used to slow down or speed
/// up the average’s responsiveness based on market volatility.
///
/// ## Parameters
/// - **min_period**: The minimum period. Defaults to 5.
/// - **max_period**: The maximum period. Defaults to 50.
/// - **matype**: The type of moving average used internally. Defaults to "sma".
/// - **devtype**: The type of deviation used:
///   - 0 = Standard Deviation
///   - 1 = Mean Absolute Deviation
///   - 2 = Median Absolute Deviation
///   Defaults to 0 (Standard Deviation).
///
/// ## Errors
/// - **EmptyData**: vlma: Input data slice is empty.
/// - **InvalidPeriodRange**: vlma: min_period > max_period.
/// - **InvalidPeriod**: vlma: `max_period` is zero or exceeds the data length.
/// - **AllValuesNaN**: vlma: All input data values are `NaN`.
/// - **NotEnoughValidData**: vlma: Fewer than `max_period` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **MaError**: vlma: Error calculating the internal moving average.
/// - **DevError**: vlma: Error calculating the internal deviation.
///
/// ## Returns
/// - **`Ok(VlmaOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the initial valid window is filled.
/// - **`Err(VlmaError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum VlmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct VlmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct VlmaParams {
    pub min_period: Option<usize>,
    pub max_period: Option<usize>,
    pub matype: Option<String>,
    pub devtype: Option<usize>,
}

impl Default for VlmaParams {
    fn default() -> Self {
        Self {
            min_period: Some(5),
            max_period: Some(50),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VlmaInput<'a> {
    pub data: VlmaData<'a>,
    pub params: VlmaParams,
}

impl<'a> VlmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: VlmaParams) -> Self {
        Self {
            data: VlmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: VlmaParams) -> Self {
        Self {
            data: VlmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: VlmaData::Candles {
                candles,
                source: "close",
            },
            params: VlmaParams::default(),
        }
    }

    pub fn get_min_period(&self) -> usize {
        self.params
            .min_period
            .unwrap_or_else(|| VlmaParams::default().min_period.unwrap())
    }

    pub fn get_max_period(&self) -> usize {
        self.params
            .max_period
            .unwrap_or_else(|| VlmaParams::default().max_period.unwrap())
    }

    pub fn get_matype(&self) -> String {
        self.params
            .matype
            .clone()
            .unwrap_or_else(|| VlmaParams::default().matype.unwrap())
    }

    pub fn get_devtype(&self) -> usize {
        self.params
            .devtype
            .unwrap_or_else(|| VlmaParams::default().devtype.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VlmaError {
    #[error("vlma: Empty data provided.")]
    EmptyData,
    #[error("vlma: min_period={min_period} is greater than max_period={max_period}.")]
    InvalidPeriodRange {
        min_period: usize,
        max_period: usize,
    },
    #[error("vlma: Invalid period: min_period={min_period}, max_period={max_period}, data length={data_len}.")]
    InvalidPeriod {
        min_period: usize,
        max_period: usize,
        data_len: usize,
    },
    #[error("vlma: All values are NaN.")]
    AllValuesNaN,
    #[error("vlma: Not enough valid data: needed={needed}, valid={valid}.")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vlma: Error in MA calculation: {0}")]
    MaError(String),
    #[error("vlma: Error in Deviation calculation: {0}")]
    DevError(String),
}

#[inline]
pub fn vlma(input: &VlmaInput) -> Result<VlmaOutput, VlmaError> {
    let data: Vec<f64> = match &input.data {
        VlmaData::Candles { candles, source } => source_type(candles, source).to_vec(),
        VlmaData::Slice(slice) => slice.to_vec(),
    };

    if data.is_empty() {
        return Err(VlmaError::EmptyData);
    }

    let min_period = input.get_min_period();
    let max_period = input.get_max_period();
    if min_period > max_period {
        return Err(VlmaError::InvalidPeriodRange {
            min_period,
            max_period,
        });
    }

    if max_period == 0 || max_period > data.len() {
        return Err(VlmaError::InvalidPeriod {
            min_period,
            max_period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(VlmaError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < max_period {
        return Err(VlmaError::NotEnoughValidData {
            needed: max_period,
            valid: data.len() - first_valid_idx,
        });
    }

    let matype = input.get_matype();
    let devtype = input.get_devtype();

    let mean = ma(&matype, MaData::Slice(&data), max_period)
        .map_err(|e| VlmaError::MaError(e.to_string()))?;

    let dev_params = DevParams {
        period: Some(max_period),
        devtype: Some(devtype),
    };
    let dev_input = DevInput::from_slice(&data, dev_params);
    let dev = deviation(&dev_input).map_err(|e| VlmaError::DevError(e.to_string()))?;

    let mut a = vec![f64::NAN; data.len()];
    let mut b = vec![f64::NAN; data.len()];
    let mut c = vec![f64::NAN; data.len()];
    let mut d = vec![f64::NAN; data.len()];

    for i in 0..data.len() {
        if !mean[i].is_nan() && !dev[i].is_nan() {
            a[i] = mean[i] - 1.75 * dev[i];
            b[i] = mean[i] - 0.25 * dev[i];
            c[i] = mean[i] + 0.25 * dev[i];
            d[i] = mean[i] + 1.75 * dev[i];
        }
    }

    let mut vlma_values = vec![f64::NAN; data.len()];
    let mut periods = vec![0.0; data.len()];

    let mut last_val = data[first_valid_idx];
    vlma_values[first_valid_idx] = last_val;
    periods[first_valid_idx] = max_period as f64;

    for i in (first_valid_idx + 1)..data.len() {
        if data[i].is_nan() {
            vlma_values[i] = f64::NAN;
            continue;
        }
        let prev_period = if periods[i - 1] == 0.0 {
            max_period as f64
        } else {
            periods[i - 1]
        };

        let mut new_period = if !a[i].is_nan() && !b[i].is_nan() && !c[i].is_nan() && !d[i].is_nan()
        {
            if data[i] < a[i] || data[i] > d[i] {
                prev_period - 1.0
            } else if data[i] >= b[i] && data[i] <= c[i] {
                prev_period + 1.0
            } else {
                prev_period
            }
        } else {
            prev_period
        };

        if new_period < min_period as f64 {
            new_period = min_period as f64;
        } else if new_period > max_period as f64 {
            new_period = max_period as f64;
        }

        let sc = 2.0 / (new_period + 1.0);
        let new_val = data[i] * sc + (1.0 - sc) * last_val;
        periods[i] = new_period;
        last_val = new_val;

        if i >= first_valid_idx + max_period - 1 {
            vlma_values[i] = new_val;
        }
    }

    Ok(VlmaOutput {
        values: vlma_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_vlma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = VlmaParams {
            min_period: None,
            max_period: None,
            matype: None,
            devtype: None,
        };
        let input_default = VlmaInput::from_candles(&candles, "close", default_params);
        let output_default = vlma(&input_default).expect("Failed VLMA with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_custom = VlmaParams {
            min_period: Some(10),
            max_period: Some(30),
            matype: Some("ema".to_string()),
            devtype: Some(2),
        };
        let input_custom = VlmaInput::from_candles(&candles, "close", params_custom);
        let output_custom = vlma(&input_custom).expect("Failed VLMA with custom params");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_vlma_accuracy_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = VlmaParams {
            min_period: Some(5),
            max_period: Some(50),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        };
        let input = VlmaInput::from_candles(&candles, "close", params);
        let vlma_result = vlma(&input).expect("Failed to calculate VLMA");

        assert_eq!(vlma_result.values.len(), close_prices.len());

        let required_len = 5;
        assert!(
            vlma_result.values.len() >= required_len,
            "VLMA length is too short"
        );

        let test_vals = [
            59376.252799490234,
            59343.71066624187,
            59292.92555520155,
            59269.93796266796,
            59167.4483022233,
        ];
        let start_idx = vlma_result.values.len() - test_vals.len();
        let actual_slice = &vlma_result.values[start_idx..];

        for (i, &val) in actual_slice.iter().enumerate() {
            let expected = test_vals[i];
            if !val.is_nan() {
                assert!(
                    (val - expected).abs() < 1e-1,
                    "Mismatch at index {}: expected {}, got {}",
                    i,
                    expected,
                    val
                );
            }
        }
    }

    #[test]
    fn test_vlma_zero_or_inverted_periods() {
        let input_data = [10.0, 20.0, 30.0, 40.0];
        let params_min_greater = VlmaParams {
            min_period: Some(10),
            max_period: Some(5),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        };
        let input_min_greater = VlmaInput::from_slice(&input_data, params_min_greater);
        let result = vlma(&input_min_greater);
        assert!(result.is_err());

        let params_zero_max = VlmaParams {
            min_period: Some(5),
            max_period: Some(0),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        };
        let input_zero_max = VlmaInput::from_slice(&input_data, params_zero_max);
        let result2 = vlma(&input_zero_max);
        assert!(result2.is_err());
    }

    #[test]
    fn test_vlma_not_enough_data() {
        let input_data = [10.0, 20.0, 30.0];
        let params = VlmaParams {
            min_period: Some(5),
            max_period: Some(10),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        };
        let input = VlmaInput::from_slice(&input_data, params);
        let result = vlma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_vlma_all_nan() {
        let input_data = [f64::NAN, f64::NAN, f64::NAN];
        let params = VlmaParams {
            min_period: Some(2),
            max_period: Some(3),
            matype: Some("sma".to_string()),
            devtype: Some(0),
        };
        let input = VlmaInput::from_slice(&input_data, params);
        let result = vlma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_vlma_slice_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = VlmaParams {
            min_period: Some(5),
            max_period: Some(20),
            matype: Some("ema".to_string()),
            devtype: Some(1),
        };
        let first_input = VlmaInput::from_candles(&candles, "close", first_params);
        let first_result = vlma(&first_input).expect("Failed to calculate first VLMA");

        let second_params = VlmaParams {
            min_period: Some(5),
            max_period: Some(20),
            matype: Some("ema".to_string()),
            devtype: Some(1),
        };
        let second_input = VlmaInput::from_slice(&first_result.values, second_params);
        let second_result = vlma(&second_input).expect("Failed to calculate second VLMA");

        assert_eq!(second_result.values.len(), first_result.values.len());
    }
}
