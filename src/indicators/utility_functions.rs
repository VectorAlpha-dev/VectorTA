use std::collections::VecDeque;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RollingError {
    #[error("rolling: Empty data provided.")]
    EmptyData,
    #[error("rolling: Invalid period: period={period}, data length={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("rolling: All values are NaN.")]
    AllValuesNaN,
    #[error("rolling: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn sum_rolling(data: &[f64], period: usize) -> Result<Vec<f64>, RollingError> {
    if data.is_empty() {
        return Err(RollingError::EmptyData);
    }
    if period == 0 || period > data.len() {
        return Err(RollingError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(RollingError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(RollingError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let valid_len = data.len() - first_valid_idx;
    let mut prefix = Vec::with_capacity(valid_len + 1);
    prefix.push(0.0);

    for &val in &data[first_valid_idx..] {
        let prev = *prefix.last().unwrap();
        prefix.push(prev + val);
    }

    let mut output = vec![f64::NAN; data.len()];
    let start_idx = first_valid_idx + period - 1;
    for i in start_idx..data.len() {
        let prefix_end = i - first_valid_idx + 1;
        let prefix_start = prefix_end - period;
        output[i] = prefix[prefix_end] - prefix[prefix_start];
    }

    Ok(output)
}

#[inline]
pub fn max_rolling(data: &[f64], period: usize) -> Result<Vec<f64>, RollingError> {
    if data.is_empty() {
        return Err(RollingError::EmptyData);
    }
    if period == 0 || period > data.len() {
        return Err(RollingError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(RollingError::AllValuesNaN),
    };
    if (data.len() - first_valid_idx) < period {
        return Err(RollingError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut output = vec![f64::NAN; data.len()];
    let mut deque: VecDeque<usize> = VecDeque::with_capacity(period);

    let start_idx = first_valid_idx + period - 1;

    for i in 0..data.len() {
        if i < first_valid_idx {
            continue;
        }
        let window_start = i.saturating_sub(period - 1);

        while let Some(&front_idx) = deque.front() {
            if front_idx < window_start {
                deque.pop_front();
            } else {
                break;
            }
        }

        let val = data[i];
        while let Some(&back_idx) = deque.back() {
            if data[back_idx] <= val {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back(i);

        if i >= start_idx {
            let max_idx = *deque.front().unwrap();
            output[i] = data[max_idx];
        }
    }

    Ok(output)
}

#[inline]
pub fn min_rolling(data: &[f64], period: usize) -> Result<Vec<f64>, RollingError> {
    if data.is_empty() {
        return Err(RollingError::EmptyData);
    }
    if period == 0 || period > data.len() {
        return Err(RollingError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(RollingError::AllValuesNaN),
    };

    if (data.len() - first_valid_idx) < period {
        return Err(RollingError::NotEnoughValidData {
            needed: period,
            valid: data.len() - first_valid_idx,
        });
    }

    let mut output = vec![f64::NAN; data.len()];
    let mut deque: VecDeque<usize> = VecDeque::with_capacity(period);

    let start_idx = first_valid_idx + period - 1;

    for i in 0..data.len() {
        if i < first_valid_idx {
            continue;
        }
        let window_start = i.saturating_sub(period - 1);

        while let Some(&front) = deque.front() {
            if front < window_start {
                deque.pop_front();
            } else {
                break;
            }
        }

        let val = data[i];
        while let Some(&back_idx) = deque.back() {
            if data[back_idx] >= val {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back(i);

        if i >= start_idx {
            let min_idx = *deque.front().unwrap();
            output[i] = data[min_idx];
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_rolling_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let period = 3;
        let result = sum_rolling(&data, period).unwrap();
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 6.0);
        assert_eq!(result[3], 9.0);
        assert_eq!(result[4], 12.0);
    }

    #[test]
    fn test_sum_rolling_zero_period() {
        let data = [1.0, 2.0, 3.0];
        let period = 0;
        let err = sum_rolling(&data, period).unwrap_err();
        assert!(
            err.to_string().contains("Invalid period"),
            "Expected InvalidPeriod error, got: {}",
            err
        );
    }

    #[test]
    fn test_sum_rolling_period_exceeding_data_length() {
        let data = [1.0, 2.0];
        let period = 5;
        let err = sum_rolling(&data, period).unwrap_err();
        assert!(
            err.to_string().contains("Invalid period"),
            "Expected InvalidPeriod error, got: {}",
            err
        );
    }

    #[test]
    fn test_max_rolling_basic() {
        let data = [2.0, 5.0, 3.0, 8.0, 1.0];
        let period = 2;
        let result = max_rolling(&data, period).unwrap();
        assert!(result[0].is_nan());
        assert_eq!(result[1], 5.0);
        assert_eq!(result[2], 5.0);
        assert_eq!(result[3], 8.0);
        assert_eq!(result[4], 8.0);
    }

    #[test]
    fn test_max_rolling_all_nan() {
        let data = [f64::NAN, f64::NAN, f64::NAN];
        let err = max_rolling(&data, 2).unwrap_err();
        assert!(
            err.to_string().contains("All values are NaN"),
            "Expected AllValuesNaN, got {}",
            err
        );
    }

    #[test]
    fn test_min_rolling_basic() {
        let data = [5.0, 2.0, 3.0, 1.0, 4.0];
        let period = 2;
        let result = min_rolling(&data, period).unwrap();
        assert!(result[0].is_nan());
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 1.0);
    }

    #[test]
    fn test_min_rolling_nan_handling() {
        let data = [f64::NAN, 5.0, 2.0];
        let period = 2;
        let result = min_rolling(&data, period).unwrap();
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0);
    }

    #[test]
    fn test_min_rolling_empty_data() {
        let data: [f64; 0] = [];
        let period = 3;
        let err = min_rolling(&data, period).unwrap_err();
        assert!(
            err.to_string().contains("Empty data provided"),
            "Expected EmptyData, got {}",
            err
        );
    }
}
