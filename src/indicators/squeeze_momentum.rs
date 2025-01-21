/// # Squeeze Momentum Indicator (SMI)
///
/// Also known as the "LazyBear Squeeze". Combines Bollinger Bands (BB) and Keltner Channels (KC)
/// to detect volatility squeezes, and uses a momentum calculation with linear regression smoothing
/// for signal generation.
///
/// ## Parameters
/// - **length_bb**: The lookback window for Bollinger Bands. Defaults to 20.
/// - **mult_bb**: The multiplier for the Bollinger Bands' standard deviation. Defaults to 2.0.
/// - **length_kc**: The lookback window for Keltner Channels. Defaults to 20.
/// - **mult_kc**: The multiplier for the Keltner Channels' True Range factor. Defaults to 1.5.
///
/// ## Errors
/// - **EmptyData**: smi: No valid data provided.
/// - **InvalidLength**: smi: A provided length parameter is zero or exceeds data length.
/// - **InconsistentDataLength**: smi: High, low, and close data have different lengths.
/// - **AllValuesNaN**: smi: All values in high/low/close are NaN.
/// - **NotEnoughValidData**: smi: Not enough valid data after the first valid index.
///
/// ## Returns
/// - **`Ok(SqueezeMomentumOutput)`** on success, containing:
///   - `squeeze`: Vec<f64> with squeeze state (-1, 0, +1),
///   - `momentum`: Vec<f64> with the linear-regression-smoothed momentum values,
///   - `momentum_signal`: Vec<f64> with the momentum signals (±1, ±2).
/// - **`Err(SqueezeMomentumError)`** otherwise.
use crate::indicators::sma::{sma, SmaData, SmaInput, SmaParams};
use crate::utilities::data_loader::Candles;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum SqueezeMomentumData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct SqueezeMomentumParams {
    pub length_bb: Option<usize>,
    pub mult_bb: Option<f64>,
    pub length_kc: Option<usize>,
    pub mult_kc: Option<f64>,
}

impl Default for SqueezeMomentumParams {
    fn default() -> Self {
        Self {
            length_bb: Some(20),
            mult_bb: Some(2.0),
            length_kc: Some(20),
            mult_kc: Some(1.5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SqueezeMomentumInput<'a> {
    pub data: SqueezeMomentumData<'a>,
    pub params: SqueezeMomentumParams,
}

impl<'a> SqueezeMomentumInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: SqueezeMomentumParams) -> Self {
        Self {
            data: SqueezeMomentumData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: SqueezeMomentumParams,
    ) -> Self {
        Self {
            data: SqueezeMomentumData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SqueezeMomentumData::Candles { candles },
            params: SqueezeMomentumParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SqueezeMomentumOutput {
    pub squeeze: Vec<f64>,
    pub momentum: Vec<f64>,
    pub momentum_signal: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum SqueezeMomentumError {
    #[error("smi: Empty data provided for Squeeze Momentum.")]
    EmptyData,
    #[error("smi: Invalid length parameter: length = {length}, data length = {data_len}")]
    InvalidLength { length: usize, data_len: usize },
    #[error("smi: High/low/close arrays have inconsistent lengths.")]
    InconsistentDataLength,
    #[error("smi: All values are NaN.")]
    AllValuesNaN,
    #[error("smi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn squeeze_momentum(
    input: &SqueezeMomentumInput,
) -> Result<SqueezeMomentumOutput, SqueezeMomentumError> {
    let (high, low, close) = match &input.data {
        SqueezeMomentumData::Candles { candles } => {
            let high = candles
                .select_candle_field("high")
                .map_err(|_| SqueezeMomentumError::EmptyData)?;
            let low = candles
                .select_candle_field("low")
                .map_err(|_| SqueezeMomentumError::EmptyData)?;
            let close = candles
                .select_candle_field("close")
                .map_err(|_| SqueezeMomentumError::EmptyData)?;
            (high, low, close)
        }
        SqueezeMomentumData::Slices { high, low, close } => (*high, *low, *close),
    };
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(SqueezeMomentumError::EmptyData);
    }
    if high.len() != low.len() || low.len() != close.len() {
        return Err(SqueezeMomentumError::InconsistentDataLength);
    }
    let length_bb = input
        .params
        .length_bb
        .unwrap_or_else(|| SqueezeMomentumParams::default().length_bb.unwrap());
    let mult_bb = input
        .params
        .mult_bb
        .unwrap_or_else(|| SqueezeMomentumParams::default().mult_bb.unwrap());
    let length_kc = input
        .params
        .length_kc
        .unwrap_or_else(|| SqueezeMomentumParams::default().length_kc.unwrap());
    let mult_kc = input
        .params
        .mult_kc
        .unwrap_or_else(|| SqueezeMomentumParams::default().mult_kc.unwrap());
    if length_bb == 0 || length_bb > close.len() {
        return Err(SqueezeMomentumError::InvalidLength {
            length: length_bb,
            data_len: close.len(),
        });
    }
    if length_kc == 0 || length_kc > close.len() {
        return Err(SqueezeMomentumError::InvalidLength {
            length: length_kc,
            data_len: close.len(),
        });
    }
    let first_valid_idx = match (0..close.len()).find(|&i| {
        let h = high[i];
        let l = low[i];
        let c = close[i];
        !(h.is_nan() || l.is_nan() || c.is_nan())
    }) {
        Some(idx) => idx,
        None => return Err(SqueezeMomentumError::AllValuesNaN),
    };
    let needed = length_bb.max(length_kc);
    if (high.len() - first_valid_idx) < needed {
        return Err(SqueezeMomentumError::NotEnoughValidData {
            needed,
            valid: high.len() - first_valid_idx,
        });
    }
    let bb_sma_params = SmaParams {
        period: Some(length_bb),
    };
    let bb_sma_input = SmaInput::from_slice(&close, bb_sma_params);
    let bb_sma_output =
        crate::indicators::sma::sma(&bb_sma_input).map_err(|_| SqueezeMomentumError::EmptyData)?;
    let basis = &bb_sma_output.values;
    let dev = stddev_slice(&close, length_bb);
    let mut upper_bb = vec![f64::NAN; close.len()];
    let mut lower_bb = vec![f64::NAN; close.len()];
    for i in first_valid_idx..close.len() {
        if i + 1 >= length_bb && !basis[i].is_nan() && !dev[i].is_nan() {
            upper_bb[i] = basis[i] + mult_bb * dev[i];
            lower_bb[i] = basis[i] - mult_bb * dev[i];
        }
    }
    let kc_sma_params = SmaParams {
        period: Some(length_kc),
    };
    let kc_sma_input = SmaInput::from_slice(&close, kc_sma_params.clone());
    let kc_sma_output =
        crate::indicators::sma::sma(&kc_sma_input).map_err(|_| SqueezeMomentumError::EmptyData)?;
    let kc_ma = &kc_sma_output.values;
    let true_range = true_range_slice(&high, &low, &close);
    let tr_sma_input = SmaInput::from_slice(&true_range, kc_sma_params.clone());
    let tr_sma_output =
        crate::indicators::sma::sma(&tr_sma_input).map_err(|_| SqueezeMomentumError::EmptyData)?;
    let tr_ma = &tr_sma_output.values;
    let mut upper_kc = vec![f64::NAN; close.len()];
    let mut lower_kc = vec![f64::NAN; close.len()];
    for i in first_valid_idx..close.len() {
        if i + 1 >= length_kc && !kc_ma[i].is_nan() && !tr_ma[i].is_nan() {
            upper_kc[i] = kc_ma[i] + tr_ma[i] * mult_kc;
            lower_kc[i] = kc_ma[i] - tr_ma[i] * mult_kc;
        }
    }
    let mut squeeze = vec![f64::NAN; close.len()];
    for i in first_valid_idx..close.len() {
        if !lower_bb[i].is_nan()
            && !upper_bb[i].is_nan()
            && !lower_kc[i].is_nan()
            && !upper_kc[i].is_nan()
        {
            let sqz_on = lower_bb[i] > lower_kc[i] && upper_bb[i] < upper_kc[i];
            let sqz_off = lower_bb[i] < lower_kc[i] && upper_bb[i] > upper_kc[i];
            let no_sqz = !sqz_on && !sqz_off;
            squeeze[i] = if no_sqz {
                0.0
            } else if sqz_on {
                -1.0
            } else {
                1.0
            };
        }
    }
    let mut highest_vals = rolling_high_slice(&high, length_kc);
    let mut lowest_vals = rolling_low_slice(&low, length_kc);
    let sma_kc_input = SmaInput::from_slice(&close, kc_sma_params);
    let sma_kc_output =
        crate::indicators::sma::sma(&sma_kc_input).map_err(|_| SqueezeMomentumError::EmptyData)?;
    let ma_kc = &sma_kc_output.values;
    let mut momentum_raw = vec![f64::NAN; close.len()];
    for i in first_valid_idx..close.len() {
        if i + 1 >= length_kc
            && !close[i].is_nan()
            && !highest_vals[i].is_nan()
            && !lowest_vals[i].is_nan()
            && !ma_kc[i].is_nan()
        {
            let mid = (highest_vals[i] + lowest_vals[i]) / 2.0;
            momentum_raw[i] = close[i] - (mid + ma_kc[i]) / 2.0;
        }
    }
    let momentum = linearreg_slice(&momentum_raw, length_kc);
    let mut momentum_signal = vec![f64::NAN; close.len()];
    for i in first_valid_idx..(close.len().saturating_sub(1)) {
        if !momentum[i].is_nan() && !momentum[i + 1].is_nan() {
            let next = momentum[i + 1];
            let curr = momentum[i];
            if next > 0.0 {
                momentum_signal[i + 1] = if next > curr { 1.0 } else { 2.0 };
            } else {
                momentum_signal[i + 1] = if next < curr { -1.0 } else { -2.0 };
            }
        }
    }
    Ok(SqueezeMomentumOutput {
        squeeze,
        momentum,
        momentum_signal,
    })
}

fn stddev_slice(data: &[f64], period: usize) -> Vec<f64> {
    let mut output = vec![f64::NAN; data.len()];
    if period == 0 || period > data.len() {
        return output;
    }
    let mut window_sum = 0.0;
    let mut window_sumsq = 0.0;
    for i in 0..period {
        let v = data[i];
        if v.is_finite() {
            window_sum += v;
            window_sumsq += v * v;
        }
    }
    let mut count = period;
    if count > 0 {
        output[period - 1] = variance_to_stddev(window_sum, window_sumsq, count);
    }
    for i in period..data.len() {
        let old_v = data[i - period];
        let new_v = data[i];
        if old_v.is_finite() {
            window_sum -= old_v;
            window_sumsq -= old_v * old_v;
        }
        if new_v.is_finite() {
            window_sum += new_v;
            window_sumsq += new_v * new_v;
        }
        output[i] = variance_to_stddev(window_sum, window_sumsq, count);
    }
    output
}

fn variance_to_stddev(sum: f64, sumsq: f64, count: usize) -> f64 {
    if count < 2 {
        return f64::NAN;
    }
    let mean = sum / (count as f64);
    let var = (sumsq / (count as f64)) - (mean * mean);
    if var.is_sign_negative() {
        f64::NAN
    } else {
        var.sqrt()
    }
}

fn true_range_slice(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    if high.len() != low.len() || low.len() != close.len() {
        return vec![];
    }
    let mut output = vec![f64::NAN; high.len()];
    let mut prev_close = close[0];
    output[0] = high[0].max(low[0]) - low[0].min(high[0]);
    for i in 1..high.len() {
        if !high[i].is_nan() && !low[i].is_nan() && !prev_close.is_nan() {
            let tr1 = high[i] - low[i];
            let tr2 = (high[i] - prev_close).abs();
            let tr3 = (low[i] - prev_close).abs();
            output[i] = tr1.max(tr2).max(tr3);
        }
        prev_close = close[i];
    }
    output
}

fn rolling_high_slice(data: &[f64], period: usize) -> Vec<f64> {
    let mut output = vec![f64::NAN; data.len()];
    if period == 0 || period > data.len() {
        return output;
    }
    let mut deque = Vec::new();
    for i in 0..data.len() {
        if !data[i].is_nan() {
            deque.push(data[i]);
        } else {
            deque.push(f64::NAN);
        }
        if i + 1 >= period {
            if i + 1 > period {
                deque.remove(0);
            }
            output[i] = deque.iter().copied().fold(f64::NAN, |a, b| a.max(b));
        }
    }
    output
}

fn rolling_low_slice(data: &[f64], period: usize) -> Vec<f64> {
    let mut output = vec![f64::NAN; data.len()];
    if period == 0 || period > data.len() {
        return output;
    }
    let mut deque = Vec::new();
    for i in 0..data.len() {
        if !data[i].is_nan() {
            deque.push(data[i]);
        } else {
            deque.push(f64::NAN);
        }
        if i + 1 >= period {
            if i + 1 > period {
                deque.remove(0);
            }
            let mut mn = f64::NAN;
            for &v in &deque {
                if !v.is_nan() && (mn.is_nan() || v < mn) {
                    mn = v;
                }
            }
            output[i] = mn;
        }
    }
    output
}

fn linearreg_slice(data: &[f64], period: usize) -> Vec<f64> {
    let mut output = vec![f64::NAN; data.len()];
    if period == 0 || period > data.len() {
        return output;
    }
    for i in (period - 1)..data.len() {
        let subset = &data[i + 1 - period..=i];
        if subset.iter().all(|x| x.is_finite()) {
            output[i] = linear_regression_last_point(subset);
        }
    }
    output
}

fn linear_regression_last_point(window: &[f64]) -> f64 {
    let n = window.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    for (i, &val) in window.iter().enumerate() {
        let x = (i + 1) as f64;
        sum_x += x;
        sum_y += val;
        sum_xy += x * val;
        sum_x2 += x * x;
    }
    let n_f = n as f64;
    let denom = (n_f * sum_x2) - (sum_x * sum_x);
    if denom.abs() < f64::EPSILON {
        return f64::NAN;
    }
    let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n_f;
    let x_last = n_f;
    intercept + slope * x_last
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_squeeze_momentum_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = SqueezeMomentumInput::with_default_candles(&candles);
        let result = squeeze_momentum(&input).expect("Failed to compute squeeze momentum");
        assert_eq!(result.squeeze.len(), candles.close.len());
        assert_eq!(result.momentum.len(), candles.close.len());
        assert_eq!(result.momentum_signal.len(), candles.close.len());
    }

    #[test]
    fn test_squeeze_momentum_with_slices() {
        let high = vec![10.0, 12.0, 14.0, 11.0, 15.0];
        let low = vec![5.0, 6.0, 7.0, 6.5, 7.0];
        let close = vec![7.0, 11.0, 10.0, 10.5, 14.0];
        let params = SqueezeMomentumParams {
            length_bb: Some(2),
            mult_bb: Some(2.0),
            length_kc: Some(2),
            mult_kc: Some(1.5),
        };
        let input = SqueezeMomentumInput::from_slices(&high, &low, &close, params);
        let output = squeeze_momentum(&input).expect("Failed to compute squeeze momentum slices");
        assert_eq!(output.squeeze.len(), close.len());
        assert_eq!(output.momentum.len(), close.len());
        assert_eq!(output.momentum_signal.len(), close.len());
    }

    #[test]
    fn test_squeeze_momentum_nan_and_error_checks() {
        let high = vec![];
        let low = vec![];
        let close = vec![];
        let params = SqueezeMomentumParams::default();
        let input = SqueezeMomentumInput::from_slices(&high, &low, &close, params);
        let result = squeeze_momentum(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Empty data"));
        }
    }

    #[test]
    fn test_squeeze_momentum_inconsistent_data_length() {
        let high = vec![1.0, 2.0, 3.0];
        let low = vec![1.0, 2.0];
        let close = vec![1.0, 2.0, 3.0];
        let params = SqueezeMomentumParams::default();
        let input = SqueezeMomentumInput::from_slices(&high, &low, &close, params);
        let result = squeeze_momentum(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("inconsistent lengths"));
        }
    }

    #[test]
    fn test_squeeze_momentum_minimum_valid_data() {
        let high = vec![10.0, 12.0, 14.0];
        let low = vec![5.0, 6.0, 7.0];
        let close = vec![7.0, 11.0, 10.0];
        let params = SqueezeMomentumParams {
            length_bb: Some(5),
            mult_bb: Some(2.0),
            length_kc: Some(5),
            mult_kc: Some(1.5),
        };
        let input = SqueezeMomentumInput::from_slices(&high, &low, &close, params);
        let result = squeeze_momentum(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_squeeze_momentum_accuracy_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = SqueezeMomentumParams::default();
        let input = SqueezeMomentumInput::from_candles(&candles, params);
        let output = squeeze_momentum(&input).expect("Failed to compute squeeze momentum");
        assert_eq!(output.squeeze.len(), candles.close.len());
        assert_eq!(output.momentum.len(), candles.close.len());
        assert_eq!(output.momentum_signal.len(), candles.close.len());
        let expected_last_five = [-170.9, -155.4, -65.3, -61.1, -178.1];
        if output.momentum.len() >= 5 {
            let start_index = output.momentum.len() - 5;
            for (i, &val) in output.momentum[start_index..].iter().enumerate() {
                let exp = expected_last_five[i];
                assert!(
                    (val - exp).abs() < 1e-1,
                    "Mismatch at {}: expected {}, got {}",
                    i,
                    exp,
                    val
                );
            }
        }
    }
}
