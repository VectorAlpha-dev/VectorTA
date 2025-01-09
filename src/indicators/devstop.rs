use crate::indicators::deviation::{deviation, DevInput, DevParams};
use crate::indicators::moving_averages::ma::{ma, MaData};
use crate::indicators::utility_functions::{max_rolling, min_rolling};
use crate::utilities::data_loader::{source_type, Candles};
/// # DevStop
///
/// A stop indicator that uses a volatility measure to determine stop levels.
/// Similar in logic to Kase Dev Stops, it calculates the difference between
/// rolling highs and lows, applies a moving average (`ma_type`) to that range,
/// computes a chosen deviation type (standard, mean absolute, or median
/// absolute), and then offsets the current high or low by a factor (`mult` ×
/// deviation) before finally taking a rolling maximum or minimum over
/// `period`.
///
/// ## Parameters
/// - **period**: The rolling window size. Defaults to 20.
/// - **mult**: The multiplier for the deviation. Defaults to 0.0.
/// - **devtype**: The type of deviation to compute.  
///   \- `0`: Standard Deviation  
///   \- `1`: Mean Absolute Deviation  
///   \- `2`: Median Absolute Deviation  
///   Defaults to 0.
/// - **direction**: Determines if the stop is based on "long" or "short".
///   Defaults to "long".
/// - **ma_type**: The type of moving average used for the average true range
///   calculation. Examples: `"sma"`, `"ema"`, etc. Defaults to `"sma"`.
///
/// ## Errors
/// - **EmptyData**: devstop: Input data slice is empty.
/// - **InvalidPeriod**: devstop: `period` is zero or exceeds the data length.
/// - **AllValuesNaN**: devstop: All values for high or low are NaN.
/// - **NotEnoughValidData**: devstop: Not enough valid (non-NaN) data points
///   remain after the first valid index.
/// - **DevStopCalculation**: devstop: Underlying calculation error, including
///   invalid `devtype`.
///
/// ## Returns
/// - **`Ok(DevStopOutput)`** on success, containing a `Vec<f64>` matching the
///   input length, with leading `NaN`s until the rolling windows can be
///   calculated.
/// - **`Err(DevStopError)`** otherwise.
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum DevStopData<'a> {
    Candles {
        candles: &'a Candles,
        source_high: &'a str,
        source_low: &'a str,
    },
    SliceHL(&'a [f64], &'a [f64]),
}

#[derive(Debug, Clone)]
pub struct DevStopParams {
    pub period: Option<usize>,
    pub mult: Option<f64>,
    pub devtype: Option<usize>,
    pub direction: Option<String>,
    pub ma_type: Option<String>,
}

impl Default for DevStopParams {
    fn default() -> Self {
        Self {
            period: Some(20),
            mult: Some(0.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DevStopInput<'a> {
    pub data: DevStopData<'a>,
    pub params: DevStopParams,
}

impl<'a> DevStopInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        source_high: &'a str,
        source_low: &'a str,
        params: DevStopParams,
    ) -> Self {
        Self {
            data: DevStopData::Candles {
                candles,
                source_high,
                source_low,
            },
            params,
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: DevStopParams) -> Self {
        Self {
            data: DevStopData::SliceHL(high, low),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: DevStopData::Candles {
                candles,
                source_high: "high",
                source_low: "low",
            },
            params: DevStopParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| DevStopParams::default().period.unwrap())
    }

    pub fn get_mult(&self) -> f64 {
        self.params
            .mult
            .unwrap_or_else(|| DevStopParams::default().mult.unwrap())
    }

    pub fn get_devtype(&self) -> usize {
        self.params
            .devtype
            .unwrap_or_else(|| DevStopParams::default().devtype.unwrap())
    }

    pub fn get_direction(&self) -> String {
        self.params
            .direction
            .clone()
            .unwrap_or_else(|| DevStopParams::default().direction.unwrap())
    }

    pub fn get_ma_type(&self) -> String {
        self.params
            .ma_type
            .clone()
            .unwrap_or_else(|| DevStopParams::default().ma_type.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct DevStopOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum DevStopError {
    #[error("devstop: Empty data provided.")]
    EmptyData,
    #[error("devstop: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("devstop: All values are NaN for high or low.")]
    AllValuesNaN,
    #[error("devstop: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("devstop: Calculation error: {0}")]
    DevStopCalculation(String),
}

#[inline]
pub fn devstop(input: &DevStopInput) -> Result<DevStopOutput, DevStopError> {
    let (high, low) = match &input.data {
        DevStopData::Candles {
            candles,
            source_high,
            source_low,
        } => {
            let h = source_type(candles, source_high);
            let l = source_type(candles, source_low);
            (h, l)
        }
        DevStopData::SliceHL(h, l) => (*h, *l),
    };

    if high.is_empty() || low.is_empty() {
        return Err(DevStopError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > high.len() || period > low.len() {
        return Err(DevStopError::InvalidPeriod {
            period,
            data_len: high.len().min(low.len()),
        });
    }

    let first_valid_high = high.iter().position(|&x| !x.is_nan());
    let first_valid_low = low.iter().position(|&x| !x.is_nan());
    let first_valid_idx = match (first_valid_high, first_valid_low) {
        (Some(h_idx), Some(l_idx)) => h_idx.min(l_idx),
        _ => return Err(DevStopError::AllValuesNaN),
    };

    if (high.len() - first_valid_idx) < period || (low.len() - first_valid_idx) < period {
        return Err(DevStopError::NotEnoughValidData {
            needed: period,
            valid: (high.len() - first_valid_idx).min(low.len() - first_valid_idx),
        });
    }

    let high2 = match crate::indicators::utility_functions::max_rolling(high, 2) {
        Ok(v) => v,
        Err(e) => return Err(DevStopError::DevStopCalculation(e.to_string())),
    };

    let low2 = match crate::indicators::utility_functions::min_rolling(low, 2) {
        Ok(v) => v,
        Err(e) => return Err(DevStopError::DevStopCalculation(e.to_string())),
    };

    let mut range = vec![f64::NAN; high.len()];
    for i in 0..high.len() {
        if !high2[i].is_nan() && !low2[i].is_nan() {
            range[i] = high2[i] - low2[i];
        }
    }

    let avtr = match ma(&input.get_ma_type(), MaData::Slice(&range), period) {
        Ok(v) => v,
        Err(e) => return Err(DevStopError::DevStopCalculation(e.to_string())),
    };

    for i in 0..avtr.len() {
        if !avtr[i].is_nan() {
            break;
        }
        if i == avtr.len() - 1 {
            return Err(DevStopError::DevStopCalculation(
                "All values are NaN for average true range.".to_string(),
            ));
        }
    }
    let dev_values = {
        let dev_input = DevInput::from_slice(
            &range,
            DevParams {
                period: Some(period),
                devtype: Some(input.get_devtype()),
            },
        );
        match deviation(&dev_input) {
            Ok(v) => v,
            Err(e) => return Err(DevStopError::DevStopCalculation(e.to_string())),
        }
    };

    let mult = input.get_mult();
    let direction = input.get_direction();

    let mut base = vec![f64::NAN; high.len()];
    for i in 0..high.len() {
        if direction.eq_ignore_ascii_case("long") {
            if !high[i].is_nan() && !avtr[i].is_nan() && !dev_values[i].is_nan() {
                base[i] = high[i] - avtr[i] - mult * dev_values[i];
            }
        } else {
            if !low[i].is_nan() && !avtr[i].is_nan() && !dev_values[i].is_nan() {
                base[i] = low[i] + avtr[i] + mult * dev_values[i];
            }
        }
    }

    let final_values = if direction.eq_ignore_ascii_case("long") {
        match max_rolling(&base, period) {
            Ok(v) => v,
            Err(e) => return Err(DevStopError::DevStopCalculation(e.to_string())),
        }
    } else {
        match min_rolling(&base, period) {
            Ok(v) => v,
            Err(e) => return Err(DevStopError::DevStopCalculation(e.to_string())),
        }
    };

    Ok(DevStopOutput {
        values: final_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_devstop_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = DevStopParams {
            period: None,
            mult: None,
            devtype: None,
            direction: None,
            ma_type: None,
        };
        let input_default = DevStopInput::from_candles(&candles, "high", "low", default_params);
        let output_default = devstop(&input_default).expect("Failed devstop with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_custom = DevStopParams {
            period: Some(20),
            mult: Some(1.0),
            devtype: Some(2),
            direction: Some("short".to_string()),
            ma_type: Some("ema".to_string()),
        };
        let input_custom = DevStopInput::from_candles(&candles, "high", "low", params_custom);
        let output_custom = devstop(&input_custom).expect("Failed devstop custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_devstop_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let high = &candles.high;
        let low = &candles.low;

        let params = DevStopParams {
            period: Some(20),
            mult: Some(0.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = DevStopInput::from_slices(high, low, params);
        let result = devstop(&input).expect("Failed to calculate devstop");

        assert_eq!(result.values.len(), candles.close.len());
        assert!(result.values.len() >= 5);
        let last_five = &result.values[result.values.len() - 5..];
        for &val in last_five {
            println!("Indicator values {}", val);
        }
    }

    #[test]
    fn test_devstop_params_with_default_params() {
        let default_params = DevStopParams::default();
        assert_eq!(default_params.period, Some(20));
        assert_eq!(default_params.mult, Some(0.0));
        assert_eq!(default_params.devtype, Some(0));
        assert_eq!(default_params.direction, Some("long".to_string()));
        assert_eq!(default_params.ma_type, Some("sma".to_string()));
    }

    #[test]
    fn test_devstop_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = DevStopInput::with_default_candles(&candles);
        match input.data {
            DevStopData::Candles {
                source_high,
                source_low,
                ..
            } => {
                assert_eq!(source_high, "high");
                assert_eq!(source_low, "low");
            }
            _ => panic!("Expected DevStopData::Candles"),
        }
    }

    #[test]
    fn test_devstop_with_zero_period() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = DevStopParams {
            period: Some(0),
            mult: Some(1.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = DevStopInput::from_slices(&high, &low, params);
        let result = devstop(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid period"));
        }
    }

    #[test]
    fn test_devstop_with_period_exceeding_data_length() {
        let high = [10.0, 20.0, 30.0];
        let low = [5.0, 15.0, 25.0];
        let params = DevStopParams {
            period: Some(10),
            mult: Some(1.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = DevStopInput::from_slices(&high, &low, params);
        let result = devstop(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_devstop_very_small_data_set() {
        let high = [100.0];
        let low = [90.0];
        let params = DevStopParams {
            period: Some(20),
            mult: Some(2.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = DevStopInput::from_slices(&high, &low, params);
        let result = devstop(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_devstop_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = DevStopParams {
            period: Some(20),
            mult: Some(1.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = DevStopInput::from_candles(&candles, "high", "low", params);
        let first_result = devstop(&input).expect("Failed first devstop");

        assert_eq!(first_result.values.len(), candles.close.len());

        let reinput_params = DevStopParams {
            period: Some(20),
            mult: Some(0.5),
            devtype: Some(2),
            direction: Some("short".to_string()),
            ma_type: Some("ema".to_string()),
        };
        let second_input =
            DevStopInput::from_slices(&first_result.values, &first_result.values, reinput_params);
        let second_result = devstop(&second_input).expect("Failed second devstop");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }

    #[test]
    fn test_devstop_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let high = &candles.high;
        let low = &candles.low;

        let params = DevStopParams {
            period: Some(20),
            mult: Some(0.0),
            devtype: Some(0),
            direction: Some("long".to_string()),
            ma_type: Some("sma".to_string()),
        };
        let input = DevStopInput::from_slices(high, low, params);
        let result = devstop(&input).expect("Failed devstop");

        assert_eq!(result.values.len(), high.len());

        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
