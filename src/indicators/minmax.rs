/// # MinMax (Local Extrema)
///
/// Identifies local minima and maxima over a specified `order` range.
/// Similar to the logic of `argrelextrema` in Python's SciPy, this function
/// looks for points that are strictly less (for minima) or strictly greater
/// (for maxima) than the neighboring `order` data points. The results are
/// stored in vectors marking where minima and maxima occur, and "last" vectors
/// which forward-fill the most recent extrema values.
///
/// ## Parameters
/// - **order**: The number of points on each side to use for the comparison.
///   Defaults to 3.
///
/// ## Errors
/// - **EmptyData**: minmax: Input data slice is empty.
/// - **InvalidOrder**: minmax: `order` is zero or exceeds the data length.
/// - **NotEnoughValidData**: minmax: Fewer than `order` valid (non-`NaN`) data points remain
///   after the first valid index.
/// - **AllValuesNaN**: minmax: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(MinmaxOutput)`** on success, containing:
///   - `is_min`: A `Vec<f64>` where minima positions hold their `low` value, else `NaN`.
///   - `is_max`: A `Vec<f64>` where maxima positions hold their `high` value, else `NaN`.
///   - `last_min`: Forward-filled `is_min`.
///   - `last_max`: Forward-filled `is_max`.
/// - **`Err(MinmaxError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum MinmaxData<'a> {
    Candles {
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct MinmaxOutput {
    pub is_min: Vec<f64>,
    pub is_max: Vec<f64>,
    pub last_min: Vec<f64>,
    pub last_max: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MinmaxParams {
    pub order: Option<usize>,
}

impl Default for MinmaxParams {
    fn default() -> Self {
        Self { order: Some(3) }
    }
}

#[derive(Debug, Clone)]
pub struct MinmaxInput<'a> {
    pub data: MinmaxData<'a>,
    pub params: MinmaxParams,
}

impl<'a> MinmaxInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        high_src: &'a str,
        low_src: &'a str,
        params: MinmaxParams,
    ) -> Self {
        Self {
            data: MinmaxData::Candles {
                candles,
                high_src,
                low_src,
            },
            params,
        }
    }

    pub fn from_slices(high: &'a [f64], low: &'a [f64], params: MinmaxParams) -> Self {
        Self {
            data: MinmaxData::Slices { high, low },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: MinmaxData::Candles {
                candles,
                high_src: "high",
                low_src: "low",
            },
            params: MinmaxParams::default(),
        }
    }

    pub fn get_order(&self) -> usize {
        self.params
            .order
            .unwrap_or_else(|| MinmaxParams::default().order.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum MinmaxError {
    #[error("minmax: Empty data provided.")]
    EmptyData,
    #[error("minmax: Invalid order: order = {order}, data length = {data_len}")]
    InvalidOrder { order: usize, data_len: usize },
    #[error("minmax: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("minmax: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn minmax(input: &MinmaxInput) -> Result<MinmaxOutput, MinmaxError> {
    let (high, low) = match &input.data {
        MinmaxData::Candles {
            candles,
            high_src,
            low_src,
        } => {
            let h = source_type(candles, high_src);
            let l = source_type(candles, low_src);
            (h, l)
        }
        MinmaxData::Slices { high, low } => (*high, *low),
    };

    if high.is_empty() || low.is_empty() {
        return Err(MinmaxError::EmptyData);
    }

    let order = input.get_order();
    let len_data = high.len();
    if order == 0 || order > len_data {
        return Err(MinmaxError::InvalidOrder {
            order,
            data_len: len_data,
        });
    }

    let first_valid_idx = match high
        .iter()
        .zip(low.iter())
        .position(|(&h, &l)| !(h.is_nan() || l.is_nan()))
    {
        Some(idx) => idx,
        None => return Err(MinmaxError::AllValuesNaN),
    };

    if (len_data - first_valid_idx) < order {
        return Err(MinmaxError::NotEnoughValidData {
            needed: order,
            valid: len_data - first_valid_idx,
        });
    }

    let mut is_min = vec![f64::NAN; len_data];
    let mut is_max = vec![f64::NAN; len_data];
    let mut last_min = vec![f64::NAN; len_data];
    let mut last_max = vec![f64::NAN; len_data];

    let mut last_min_val = f64::NAN;
    let mut last_max_val = f64::NAN;

    for i in first_valid_idx..len_data {
        let center_low = low[i];
        let center_high = high[i];

        if i >= order && i + order < len_data && !center_low.is_nan() && !center_high.is_nan() {
            let mut less_than_neighbors = true;
            let mut greater_than_neighbors = true;

            for o in 1..=order {
                if center_low >= low[i - o] || center_low >= low[i + o] {
                    less_than_neighbors = false;
                }
                if center_high <= high[i - o] || center_high <= high[i + o] {
                    greater_than_neighbors = false;
                }
                if !less_than_neighbors && !greater_than_neighbors {
                    break;
                }
            }

            if less_than_neighbors {
                is_min[i] = center_low;
            }
            if greater_than_neighbors {
                is_max[i] = center_high;
            }
        }

        if i == first_valid_idx {
            last_min_val = is_min[i];
            last_max_val = is_max[i];
        } else {
            if !is_min[i].is_nan() {
                last_min_val = is_min[i];
            }
            if !is_max[i].is_nan() {
                last_max_val = is_max[i];
            }
        }

        last_min[i] = last_min_val;
        last_max[i] = last_max_val;
    }

    Ok(MinmaxOutput {
        is_min,
        is_max,
        last_min,
        last_max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_minmax_default_params() {
        let params = MinmaxParams::default();
        assert_eq!(params.order, Some(3));
    }

    #[test]
    fn test_minmax_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = MinmaxInput::with_default_candles(&candles);
        let result = minmax(&input);
        assert!(result.is_ok(), "Minmax failed with default params");
        let output = result.unwrap();
        assert_eq!(output.is_min.len(), candles.close.len());
        assert_eq!(output.is_max.len(), candles.close.len());
        assert_eq!(output.last_min.len(), candles.close.len());
        assert_eq!(output.last_max.len(), candles.close.len());
    }

    #[test]
    fn test_minmax_with_zero_order() {
        let high = [10.0, 20.0, 30.0];
        let low = [1.0, 2.0, 3.0];
        let params = MinmaxParams { order: Some(0) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let result = minmax(&input);
        assert!(result.is_err(), "Expected error for zero order");
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid order"));
        }
    }

    #[test]
    fn test_minmax_with_order_exceeding_data() {
        let high = [10.0, 20.0, 30.0];
        let low = [1.0, 2.0, 3.0];
        let params = MinmaxParams { order: Some(10) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let result = minmax(&input);
        assert!(result.is_err(), "Expected error for order > data.len()");
    }

    #[test]
    fn test_minmax_nan_data() {
        let high = [f64::NAN, f64::NAN, f64::NAN];
        let low = [f64::NAN, f64::NAN, f64::NAN];
        let params = MinmaxParams { order: Some(1) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let result = minmax(&input);
        assert!(result.is_err(), "Expected error for all NaN data");
        if let Err(e) = result {
            assert!(e.to_string().contains("All values are NaN"));
        }
    }

    #[test]
    fn test_minmax_not_enough_valid_data() {
        let high = [f64::NAN, 10.0];
        let low = [f64::NAN, 5.0];
        let params = MinmaxParams { order: Some(3) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let result = minmax(&input);
        assert!(result.is_err(), "Expected error for not enough valid data");
    }

    #[test]
    fn test_minmax_basic_slices() {
        let high = [50.0, 55.0, 60.0, 55.0, 50.0, 45.0, 50.0, 55.0];
        let low = [40.0, 38.0, 35.0, 38.0, 40.0, 42.0, 41.0, 39.0];
        let params = MinmaxParams { order: Some(2) };
        let input = MinmaxInput::from_slices(&high, &low, params);
        let output = minmax(&input).expect("Failed to calculate Minmax");
        assert_eq!(output.is_min.len(), 8);
        assert_eq!(output.is_max.len(), 8);
        assert_eq!(output.last_min.len(), 8);
        assert_eq!(output.last_max.len(), 8);
    }

    #[test]
    fn test_minmax_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = MinmaxParams { order: Some(3) };
        let input = MinmaxInput::from_candles(&candles, "high", "low", params);
        let minmax_result = minmax(&input).expect("Failed to calculate MinMax indicator");

        assert_eq!(
            minmax_result.is_min.len(),
            candles.close.len(),
            "MinMax length mismatch"
        );

        let count = minmax_result.is_min.len();
        assert!(count >= 5, "Not enough data to check the last 5 elements.");

        let start_index = count - 5;
        for &val in &minmax_result.is_min[start_index..] {
            assert!(
                val.is_nan(),
                "Expected the last 5 is_min values to be NaN, but got {}",
                val
            );
        }
        for &val in &minmax_result.is_max[start_index..] {
            assert!(
                val.is_nan(),
                "Expected the last 5 is_max values to be NaN, but got {}",
                val
            );
        }

        let expected_last_five_min = [57876.0, 57876.0, 57876.0, 57876.0, 57876.0];
        let last_min_slice = &minmax_result.last_min[start_index..];
        for (i, &val) in last_min_slice.iter().enumerate() {
            let expected_val = expected_last_five_min[i];
            assert!(
                (val - expected_val).abs() < 1e-1,
                "MinMax last_min mismatch at index {}: expected {}, got {}",
                i,
                expected_val,
                val
            );
        }

        let expected_last_five_max = [60102.0, 60102.0, 60102.0, 60102.0, 60102.0];
        let last_max_slice = &minmax_result.last_max[start_index..];
        for (i, &val) in last_max_slice.iter().enumerate() {
            let expected_val = expected_last_five_max[i];
            assert!(
                (val - expected_val).abs() < 1e-1,
                "MinMax last_max mismatch at index {}: expected {}, got {}",
                i,
                expected_val,
                val
            );
        }
    }
}
