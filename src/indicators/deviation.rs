use std::error::Error;

/// Compute rolling standard deviation, mean absolute deviation, or median absolute deviation.
///
/// # Arguments
///
/// * `data` - Input slice of data.
/// * `period` - Size of the window for each deviation calculation.
/// * `devtype`:
///   - `0` => Standard Deviation (sample)
///   - `1` => Mean Absolute Deviation
///   - `2` => Median Absolute Deviation
///
/// # Returns
///
/// `Ok(Vec<f64>)` containing the computed rolling deviation values, or an error if invalid arguments.
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct DevParams {
    pub period: Option<usize>,
    pub devtype: Option<usize>,
}

impl Default for DevParams {
    fn default() -> Self {
        Self {
            period: Some(9),
            devtype: Some(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DevInput<'a> {
    pub data: &'a [f64],
    pub params: DevParams,
}

impl<'a> DevInput<'a> {
    pub fn from_slice(data: &'a [f64], params: DevParams) -> Self {
        Self { data, params }
    }

    pub fn with_defaults(data: &'a [f64]) -> Self {
        Self {
            data,
            params: DevParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| DevParams::default().period.unwrap())
    }

    pub fn get_devtype(&self) -> usize {
        self.params
            .devtype
            .unwrap_or_else(|| DevParams::default().devtype.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum DevError {
    #[error("dev: Empty data provided.")]
    EmptyData,
    #[error("dev: Invalid period: period={period}, data length={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("dev: Invalid devtype (must be 0, 1, or 2). devtype={devtype}")]
    InvalidDevType { devtype: usize },
    #[error("dev: Could not compute rolling deviation. {0}")]
    CalculationError(String),
}

#[inline]
pub fn deviation(input: &DevInput) -> Result<Vec<f64>, DevError> {
    let data = input.data;
    if data.is_empty() {
        return Err(DevError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(DevError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let devtype = input.get_devtype();
    if !(0..=2).contains(&devtype) {
        return Err(DevError::InvalidDevType { devtype });
    }

    let out = match devtype {
        0 => standard_deviation_rolling(data, period)
            .map_err(|e| DevError::CalculationError(e.to_string()))?,
        1 => mean_absolute_deviation_rolling(data, period)
            .map_err(|e| DevError::CalculationError(e.to_string()))?,
        2 => median_absolute_deviation_rolling(data, period)
            .map_err(|e| DevError::CalculationError(e.to_string()))?,
        _ => unreachable!("We already validated devtype <= 2."),
    };

    Ok(out)
}

#[inline]
fn standard_deviation_rolling(data: &[f64], period: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    if period < 2 {
        return Err("Period must be >= 2 for standard deviation.".into());
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err("All values are NaN.".into()),
    };

    if data.len() - first_valid_idx < period {
        return Err(format!(
            "Not enough valid data: need {}, but only {} valid from index {}.",
            period,
            data.len() - first_valid_idx,
            first_valid_idx
        )
        .into());
    }

    let mut result = vec![f64::NAN; data.len()];

    let mut sum = 0.0;
    let mut sumsq = 0.0;

    for &val in &data[first_valid_idx..(first_valid_idx + period)] {
        sum += val;
        sumsq += val * val;
    }

    let mut idx = first_valid_idx + period - 1;
    let mean = sum / (period as f64);
    let var = (sumsq / (period as f64)) - mean * mean;
    result[idx] = var.sqrt();

    for i in (idx + 1)..data.len() {
        let val_in = data[i];
        let val_out = data[i - period];
        sum += val_in - val_out;
        sumsq += val_in * val_in - val_out * val_out;

        let mean = sum / (period as f64);
        let var = (sumsq / (period as f64)) - mean * mean;
        result[i] = var.sqrt();
    }

    Ok(result)
}

#[inline]
fn mean_absolute_deviation_rolling(
    data: &[f64],
    period: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err("All values are NaN.".into()),
    };

    if data.len() - first_valid_idx < period {
        return Err(format!(
            "Not enough valid data: need {}, but only {} valid from index {}.",
            period,
            data.len() - first_valid_idx,
            first_valid_idx
        )
        .into());
    }

    let mut result = vec![f64::NAN; data.len()];

    let start_window_end = first_valid_idx + period - 1;
    for i in start_window_end..data.len() {
        let window_start = i + 1 - period;
        if window_start < first_valid_idx {
            continue;
        }

        let window = &data[window_start..=i];
        let mean = window.iter().sum::<f64>() / (period as f64);
        let abs_sum = window.iter().fold(0.0, |acc, &x| acc + (x - mean).abs());
        result[i] = abs_sum / (period as f64);
    }

    Ok(result)
}
#[inline]
fn median_absolute_deviation_rolling(
    data: &[f64],
    period: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err("All values are NaN.".into()),
    };

    if data.len() - first_valid_idx < period {
        return Err(format!(
            "Not enough valid data: need {}, but only {} valid from index {}.",
            period,
            data.len() - first_valid_idx,
            first_valid_idx
        )
        .into());
    }

    let mut result = vec![f64::NAN; data.len()];

    let start_window_end = first_valid_idx + period - 1;
    for i in start_window_end..data.len() {
        let window_start = i + 1 - period;
        if window_start < first_valid_idx {
            continue;
        }

        let window = &data[window_start..=i];
        let median = find_median(window);
        let mut abs_devs: Vec<f64> = window.iter().map(|&x| (x - median).abs()).collect();
        result[i] = find_median(&abs_devs);
    }

    Ok(result)
}

#[inline]
fn find_median(slice: &[f64]) -> f64 {
    if slice.is_empty() {
        return f64::NAN;
    }
    let mut sorted = slice.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_median_odd() {
        let data = [3.0, 1.0, 2.0];
        let median = find_median(&data);
        assert!((median - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_find_median_even() {
        let data: [f64; 4] = [4.0, 1.0, 3.0, 2.0];
        let median: f64 = find_median(&data);
        assert!((median - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_find_median_empty() {
        let data: [f64; 0] = [];
        let median = find_median(&data);
        assert!(median.is_nan());
    }

    #[test]
    fn test_standard_deviation_rolling_length_and_nans() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let period = 3;

        let result = standard_deviation_rolling(&data, period).unwrap();
        assert_eq!(result.len(), data.len());

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        let expected = 0.816496580927726;
        for &val in &result[2..] {
            assert!((val - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_standard_deviation_rolling_error_too_small_period() {
        let data = [1.0, 2.0, 3.0];
        let period = 1;
        let result = standard_deviation_rolling(&data, period);
        assert!(result.is_err());
    }

    #[test]
    fn test_mean_absolute_deviation_rolling_length_and_nans() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let period = 3;
        let result = mean_absolute_deviation_rolling(&data, period).unwrap();
        assert_eq!(result.len(), data.len());

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        for &val in &result[2..] {
            assert!((val - (2.0 / 3.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_median_absolute_deviation_rolling_length_and_nans() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let period = 3;
        let result = median_absolute_deviation_rolling(&data, period).unwrap();
        assert_eq!(result.len(), data.len());

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        for &val in &result[2..] {
            assert!((val - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_deviation_std_dev_matches_input_length() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = DevParams {
            period: Some(3),
            devtype: Some(0),
        };
        let input = DevInput::from_slice(&data, params);
        let result = deviation(&input).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        let expected = 0.816496580927726;
        for &val in &result[2..] {
            assert!((val - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_deviation_mean_abs_dev_matches_input_length() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = DevParams {
            period: Some(3),
            devtype: Some(1),
        };
        let input = DevInput::from_slice(&data, params);
        let result = deviation(&input).unwrap();
        assert_eq!(result.len(), data.len());
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        for &val in &result[2..] {
            assert!((val - (2.0 / 3.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_deviation_median_abs_dev_matches_input_length() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let params = DevParams {
            period: Some(3),
            devtype: Some(2),
        };
        let input = DevInput::from_slice(&data, params);
        let result = deviation(&input).unwrap();
        assert_eq!(result.len(), data.len());
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        for &val in &result[2..] {
            assert!((val - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_deviation_empty_data() {
        let data: [f64; 0] = [];
        let input = DevInput::with_defaults(&data);
        let result = deviation(&input);
        assert!(matches!(result, Err(DevError::EmptyData)));
    }

    #[test]
    fn test_deviation_invalid_period() {
        let data = [1.0, 2.0, 3.0];

        let params = DevParams {
            period: Some(0),
            devtype: Some(0),
        };
        let input = DevInput::from_slice(&data, params);
        let result = deviation(&input);
        assert!(matches!(result, Err(DevError::InvalidPeriod { .. })));

        let params2 = DevParams {
            period: Some(10),
            devtype: Some(0),
        };
        let input2 = DevInput::from_slice(&data, params2);
        let result2 = deviation(&input2);
        assert!(matches!(result2, Err(DevError::InvalidPeriod { .. })));
    }

    #[test]
    fn test_deviation_invalid_devtype() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let params = DevParams {
            period: Some(2),
            devtype: Some(999),
        };
        let input = DevInput::from_slice(&data, params);
        let result = deviation(&input);
        assert!(matches!(result, Err(DevError::InvalidDevType { .. })));
    }
}
