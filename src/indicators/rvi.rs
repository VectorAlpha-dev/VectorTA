/// # Relative Volatility Index (RVI)
///
/// The RVI measures the direction of volatility. It uses a "dev" measure (standard deviation,
/// mean absolute deviation, or median absolute deviation) split into "up" or "down" components
/// based on price changes, then smooths these values using a moving average (SMA or EMA).
///
/// ## Parameters
/// - **period**: The window size used for the volatility calculation. Defaults to 10.
/// - **ma_len**: The window size used for smoothing the "up" and "down" arrays. Defaults to 14.
/// - **matype**: Determines the smoothing type (0=SMA, 1=EMA). Defaults to 1 (EMA).
/// - **devtype**: Determines the volatility measure (0=StdDev, 1=MeanAbsDev, 2=MedianAbsDev).
///   Defaults to 0 (StdDev).
/// - **source**: Which field to use for candles (e.g. "close"). Defaults to "close".
///
/// ## Errors
/// - **EmptyData**: rvi: Input data slice is empty.
/// - **InvalidPeriod**: rvi: `period` or `ma_len` is zero or exceeds data length.
/// - **NotEnoughValidData**: rvi: Not enough valid data remains after the first valid index
///   to compute RVI.
/// - **AllValuesNaN**: rvi: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(RviOutput)`** on success, containing a `Vec<f64>` matching the input length,
///   with leading `NaN`s until the indicator can be computed.
/// - **`Err(RviError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum RviData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct RviOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RviParams {
    pub period: Option<usize>,
    pub ma_len: Option<usize>,
    pub matype: Option<usize>,
    pub devtype: Option<usize>,
}

impl Default for RviParams {
    fn default() -> Self {
        Self {
            period: Some(10),
            ma_len: Some(14),
            matype: Some(1),
            devtype: Some(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RviInput<'a> {
    pub data: RviData<'a>,
    pub params: RviParams,
}

impl<'a> RviInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: RviParams) -> Self {
        Self {
            data: RviData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: RviParams) -> Self {
        Self {
            data: RviData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: RviData::Candles {
                candles,
                source: "close",
            },
            params: RviParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| RviParams::default().period.unwrap())
    }

    pub fn get_ma_len(&self) -> usize {
        self.params
            .ma_len
            .unwrap_or_else(|| RviParams::default().ma_len.unwrap())
    }

    pub fn get_matype(&self) -> usize {
        self.params
            .matype
            .unwrap_or_else(|| RviParams::default().matype.unwrap())
    }

    pub fn get_devtype(&self) -> usize {
        self.params
            .devtype
            .unwrap_or_else(|| RviParams::default().devtype.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RviError {
    #[error("rvi: Empty data provided.")]
    EmptyData,
    #[error("rvi: Invalid period or ma_len: period = {period}, ma_len = {ma_len}, data length = {data_len}")]
    InvalidPeriod {
        period: usize,
        ma_len: usize,
        data_len: usize,
    },
    #[error("rvi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("rvi: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn rvi(input: &RviInput) -> Result<RviOutput, RviError> {
    let data: &[f64] = match &input.data {
        RviData::Candles { candles, source } => source_type(candles, source),
        RviData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(RviError::EmptyData);
    }

    let period = input.get_period();
    let ma_len = input.get_ma_len();
    let matype = input.get_matype();
    let devtype = input.get_devtype();

    if period == 0 || ma_len == 0 || period > data.len() || ma_len > data.len() {
        return Err(RviError::InvalidPeriod {
            period,
            ma_len,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(RviError::AllValuesNaN),
    };

    let max_needed = period.saturating_sub(1) + ma_len.saturating_sub(1);
    if (data.len() - first_valid_idx) <= max_needed {
        return Err(RviError::NotEnoughValidData {
            needed: max_needed + 1,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut rvi_values = vec![f64::NAN; data.len()];

    let dev_array = compute_dev(data, period, devtype);
    let diff_array = compute_diff_same_length(data);
    let up_array = compute_up_array(&diff_array, &dev_array);
    let down_array = compute_down_array(&diff_array, &dev_array);

    let up_smoothed = compute_rolling_ma(&up_array, ma_len, matype);
    let down_smoothed = compute_rolling_ma(&down_array, ma_len, matype);

    let start_idx = first_valid_idx + period.saturating_sub(1) + ma_len.saturating_sub(1);
    for i in start_idx..data.len() {
        let up_val = up_smoothed[i];
        let down_val = down_smoothed[i];
        if up_val.is_nan() || down_val.is_nan() {
            rvi_values[i] = f64::NAN;
        } else if (up_val + down_val).abs() < f64::EPSILON {
            rvi_values[i] = f64::NAN;
        } else {
            rvi_values[i] = 100.0 * (up_val / (up_val + down_val));
        }
    }

    Ok(RviOutput { values: rvi_values })
}

fn compute_diff_same_length(data: &[f64]) -> Vec<f64> {
    let mut diff = vec![0.0; data.len()];
    for i in 1..data.len() {
        let prev = data[i - 1];
        let curr = data[i];
        if prev.is_nan() || curr.is_nan() {
            diff[i] = f64::NAN;
        } else {
            diff[i] = curr - prev;
        }
    }
    diff
}

fn compute_up_array(diff: &[f64], dev: &[f64]) -> Vec<f64> {
    let mut up = vec![f64::NAN; diff.len()];
    for i in 0..diff.len() {
        let d = diff[i];
        let dv = dev[i];
        if d.is_nan() || dv.is_nan() {
            up[i] = f64::NAN;
        } else if d <= 0.0 {
            up[i] = 0.0;
        } else {
            up[i] = dv;
        }
    }
    up
}

fn compute_down_array(diff: &[f64], dev: &[f64]) -> Vec<f64> {
    let mut down = vec![f64::NAN; diff.len()];
    for i in 0..diff.len() {
        let d = diff[i];
        let dv = dev[i];
        if d.is_nan() || dv.is_nan() {
            down[i] = f64::NAN;
        } else if d > 0.0 {
            down[i] = 0.0;
        } else {
            down[i] = dv;
        }
    }
    down
}

fn compute_dev(data: &[f64], period: usize, devtype: usize) -> Vec<f64> {
    match devtype {
        1 => rolling_mean_abs_dev(data, period),
        2 => rolling_median_abs_dev(data, period),
        _ => rolling_std_dev(data, period),
    }
}

fn rolling_std_dev(data: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    if period == 0 || period > data.len() {
        return out;
    }
    let mut window_sum = 0.0;
    let mut window_sumsq = 0.0;
    for i in 0..period {
        let x = data[i];
        if x.is_nan() {
            window_sum = f64::NAN;
            break;
        }
        window_sum += x;
        window_sumsq += x * x;
    }
    if !window_sum.is_nan() {
        let mean = window_sum / (period as f64);
        let mean_sq = window_sumsq / (period as f64);
        out[period - 1] = (mean_sq - mean * mean).sqrt();
    }
    for i in period..data.len() {
        let leaving = data[i - period];
        let incoming = data[i];
        if leaving.is_nan() || incoming.is_nan() || window_sum.is_nan() {
            out[i] = f64::NAN;
            window_sum = f64::NAN;
            continue;
        }
        window_sum += incoming - leaving;
        window_sumsq += incoming * incoming - leaving * leaving;
        let mean = window_sum / (period as f64);
        let mean_sq = window_sumsq / (period as f64);
        out[i] = (mean_sq - mean * mean).sqrt();
    }
    out
}

fn rolling_mean_abs_dev(data: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    if period == 0 || period > data.len() {
        return out;
    }
    use std::collections::VecDeque;
    let mut window = VecDeque::with_capacity(period);
    let mut current_sum = 0.0;
    for i in 0..data.len() {
        let x = data[i];
        if x.is_nan() {
            out[i] = f64::NAN;
            window.clear();
            current_sum = 0.0;
        } else {
            window.push_back(x);
            current_sum += x;
            if window.len() > period {
                if let Some(old) = window.pop_front() {
                    current_sum -= old;
                }
            }
            if window.len() == period {
                let mean = current_sum / (period as f64);
                let mut abs_sum = 0.0;
                for &val in &window {
                    abs_sum += (val - mean).abs();
                }
                out[i] = abs_sum / (period as f64);
            }
        }
    }
    out
}

fn rolling_median_abs_dev(data: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    if period == 0 || period > data.len() {
        return out;
    }
    use std::collections::VecDeque;
    let mut window = VecDeque::with_capacity(period);
    for i in 0..data.len() {
        let x = data[i];
        if x.is_nan() {
            out[i] = f64::NAN;
            window.clear();
        } else {
            window.push_back(x);
            if window.len() > period {
                window.pop_front();
            }
            if window.len() == period {
                let mut tmp: Vec<f64> = window.iter().copied().collect();
                tmp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = if period % 2 == 1 {
                    tmp[period / 2]
                } else {
                    (tmp[period / 2 - 1] + tmp[period / 2]) / 2.0
                };
                let mut abs_sum = 0.0;
                for &val in &tmp {
                    abs_sum += (val - median).abs();
                }
                out[i] = abs_sum / (period as f64);
            }
        }
    }
    out
}

fn compute_rolling_ma(data: &[f64], period: usize, matype: usize) -> Vec<f64> {
    match matype {
        0 => rolling_sma(data, period),
        _ => rolling_ema(data, period),
    }
}

fn rolling_sma(data: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    if period == 0 || period > data.len() {
        return out;
    }
    let mut window_sum = 0.0;
    let mut count = 0;
    for i in 0..data.len() {
        let x = data[i];
        if x.is_nan() {
            out[i] = f64::NAN;
            window_sum = 0.0;
            count = 0;
        } else {
            window_sum += x;
            count += 1;
            if i >= period {
                let old = data[i - period];
                if !old.is_nan() {
                    window_sum -= old;
                    count -= 1;
                } else {
                    out[i] = f64::NAN;
                    continue;
                }
            }
            if i + 1 >= period {
                out[i] = window_sum / (period as f64);
            }
        }
    }
    out
}

fn rolling_ema(data: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; data.len()];
    if period == 0 || period > data.len() {
        return out;
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut prev_ema = 0.0;
    let mut started = false;
    for i in 0..data.len() {
        let x = data[i];
        if x.is_nan() {
            out[i] = f64::NAN;
            continue;
        }
        if !started {
            let first_window_end = if i + 1 < period { i + 1 } else { period };
            if i + 1 < period {
                // Not enough data to start EMA yet
                out[i] = f64::NAN;
                prev_ema += x;
                if i + 1 == first_window_end {
                    prev_ema /= period as f64;
                }
            } else {
                prev_ema += x;
                prev_ema /= period as f64;
                out[i] = prev_ema;
                started = true;
            }
        } else {
            prev_ema = alpha * x + (1.0 - alpha) * prev_ema;
            out[i] = prev_ema;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_rvi_default_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = RviInput::with_default_candles(&candles);
        let output = rvi(&input).expect("Failed RVI with default params");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_rvi_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let partial_params = RviParams {
            period: Some(10),
            ma_len: None,
            matype: None,
            devtype: None,
        };
        let input = RviInput::from_candles(&candles, "close", partial_params);
        let output = rvi(&input).expect("Failed RVI with partial params");
        assert_eq!(output.values.len(), candles.close.len());
    }

    #[test]
    fn test_rvi_error_zero_period() {
        let data = [10.0, 20.0, 30.0, 40.0];
        let params = RviParams {
            period: Some(0),
            ma_len: Some(14),
            matype: Some(1),
            devtype: Some(0),
        };
        let input = RviInput::from_slice(&data, params);
        let result = rvi(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_rvi_error_zero_ma_len() {
        let data = [10.0, 20.0, 30.0, 40.0];
        let params = RviParams {
            period: Some(10),
            ma_len: Some(0),
            matype: Some(1),
            devtype: Some(0),
        };
        let input = RviInput::from_slice(&data, params);
        let result = rvi(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_rvi_error_period_exceeds_data_length() {
        let data = [10.0, 20.0, 30.0];
        let params = RviParams {
            period: Some(10),
            ma_len: Some(14),
            matype: Some(1),
            devtype: Some(0),
        };
        let input = RviInput::from_slice(&data, params);
        let result = rvi(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_rvi_all_nan_input() {
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let params = RviParams::default();
        let input = RviInput::from_slice(&data, params);
        let result = rvi(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_rvi_not_enough_valid_data() {
        let data = [f64::NAN, 1.0, 2.0, 3.0];
        let params = RviParams {
            period: Some(3),
            ma_len: Some(5),
            matype: Some(1),
            devtype: Some(0),
        };
        let input = RviInput::from_slice(&data, params);
        let result = rvi(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_rvi_example_values() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = RviParams {
            period: Some(10),
            ma_len: Some(14),
            matype: Some(1),
            devtype: Some(0),
        };
        let input = RviInput::from_candles(&candles, "close", params);
        let output = rvi(&input).expect("Failed to calculate RVI");
        assert_eq!(output.values.len(), candles.close.len());
        let last_five = &output.values[output.values.len().saturating_sub(5)..];
        let expected = [
            67.48579363423423,
            62.03322230763894,
            56.71819195768154,
            60.487299747927636,
            55.022521428674175,
        ];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected[i];
            assert!(
                val.is_finite(),
                "Expected a finite RVI value, got NaN at index {}",
                i
            );
            let diff = (val - exp).abs();
            assert!(
                diff < 1e-1,
                "Mismatch at index {} -> got: {}, expected: {}, diff: {}",
                i,
                val,
                exp,
                diff
            );
        }
    }
}
