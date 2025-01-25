/// # Vortex Indicator (VI)
///
/// The Vortex Indicator consists of two lines: the positive Vortex (VI+) and
/// the negative Vortex (VI-). It is computed based on True Range (TR), and the
/// absolute movement of the current High relative to the previous Low (for VI+),
/// and the current Low relative to the previous High (for VI-). The sums of
/// these movements over a specified `period` are normalized by the sum of TR
/// over the same `period`.
///
/// ## Parameters
/// - **period**: The lookback window size. Defaults to 14.
///
/// ## Errors
/// - **EmptyData**: vi: Input data slices are empty.
/// - **InvalidPeriod**: vi: `period` is zero or exceeds the data length.
/// - **NotEnoughValidData**: vi: Fewer than `period` valid data points remain
///   after the first valid index.
/// - **AllValuesNaN**: vi: All input data values are `NaN`.
///
/// ## Returns
/// - **`Ok(ViOutput)`** on success, containing `plus` and `minus` vectors
///   matching the input length, with leading `NaN`s until the period window is filled.
/// - **`Err(ViError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use std::f64;
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum ViData<'a> {
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
pub struct ViOutput {
    pub plus: Vec<f64>,
    pub minus: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ViParams {
    pub period: Option<usize>,
}

impl Default for ViParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct ViInput<'a> {
    pub data: ViData<'a>,
    pub params: ViParams,
}

impl<'a> ViInput<'a> {
    pub fn from_candles(candles: &'a Candles, params: ViParams) -> Self {
        Self {
            data: ViData::Candles { candles },
            params,
        }
    }

    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: ViParams,
    ) -> Self {
        Self {
            data: ViData::Slices { high, low, close },
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: ViData::Candles { candles },
            params: ViParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| ViParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum ViError {
    #[error("vi: Empty data provided.")]
    EmptyData,
    #[error("vi: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("vi: Not enough valid data: needed = {needed}, valid = {valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error("vi: All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn vi(input: &ViInput) -> Result<ViOutput, ViError> {
    let (high, low, close) = match &input.data {
        ViData::Candles { candles } => {
            let high = source_type(candles, "high");
            let low = source_type(candles, "low");
            let close = source_type(candles, "close");
            (high, low, close)
        }
        ViData::Slices { high, low, close } => (*high, *low, *close),
    };

    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(ViError::EmptyData);
    }

    let length = high.len();
    if length != low.len() || length != close.len() {
        return Err(ViError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > length {
        return Err(ViError::InvalidPeriod {
            period,
            data_len: length,
        });
    }

    let first_valid_idx =
        (0..length).find(|&i| !high[i].is_nan() && !low[i].is_nan() && !close[i].is_nan());
    let first_valid_idx = match first_valid_idx {
        Some(idx) => idx,
        None => return Err(ViError::AllValuesNaN),
    };

    if (length - first_valid_idx) < period {
        return Err(ViError::NotEnoughValidData {
            needed: period,
            valid: length - first_valid_idx,
        });
    }

    let mut tr = vec![0.0; length];
    let mut vp = vec![0.0; length];
    let mut vm = vec![0.0; length];

    tr[first_valid_idx] = high[first_valid_idx] - low[first_valid_idx];

    for i in (first_valid_idx + 1)..length {
        tr[i] = (high[i] - low[i])
            .max((high[i] - close[i - 1]).abs())
            .max((low[i] - close[i - 1]).abs());
        vp[i] = (high[i] - low[i - 1]).abs();
        vm[i] = (low[i] - high[i - 1]).abs();
    }

    let mut plus = vec![f64::NAN; length];
    let mut minus = vec![f64::NAN; length];
    let mut sum_tr = 0.0;
    let mut sum_vp = 0.0;
    let mut sum_vm = 0.0;

    for i in first_valid_idx..(first_valid_idx + period) {
        sum_tr += tr[i];
        sum_vp += vp[i];
        sum_vm += vm[i];
    }
    plus[first_valid_idx + period - 1] = sum_vp / sum_tr;
    minus[first_valid_idx + period - 1] = sum_vm / sum_tr;

    for i in (first_valid_idx + period)..length {
        sum_tr += tr[i] - tr[i - period];
        sum_vp += vp[i] - vp[i - period];
        sum_vm += vm[i] - vm[i - period];
        plus[i] = sum_vp / sum_tr;
        minus[i] = sum_vm / sum_tr;
    }

    Ok(ViOutput { plus, minus })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_vi_basic() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = ViParams { period: Some(14) };
        let input = ViInput::from_candles(&candles, params);
        let output = vi(&input).expect("Failed to calculate VI");

        assert_eq!(output.plus.len(), candles.close.len());
        assert_eq!(output.minus.len(), candles.close.len());
    }

    #[test]
    fn test_vi_accuracy_last_five() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = ViParams { period: Some(14) };
        let input = ViInput::from_candles(&candles, params);
        let output = vi(&input).expect("Failed to calculate VI");

        let expected_last_five_plus = [
            0.9970238095238095,
            0.9871071716357775,
            0.9464453759945247,
            0.890897412369242,
            0.9206478557604156,
        ];
        let expected_last_five_minus = [
            1.0097117794486214,
            1.04174053182917,
            1.1152365471811105,
            1.181684712791338,
            1.1894672506875827,
        ];

        let n = output.plus.len();
        assert!(n >= 5);

        let plus_slice = &output.plus[n - 5..];
        let minus_slice = &output.minus[n - 5..];

        for (i, &val) in plus_slice.iter().enumerate() {
            let expected = expected_last_five_plus[i];
            assert!(
                (val - expected).abs() < 1e-8,
                "Mismatch in VI+ at index {}, expected {}, got {}",
                i,
                expected,
                val
            );
        }

        for (i, &val) in minus_slice.iter().enumerate() {
            let expected = expected_last_five_minus[i];
            assert!(
                (val - expected).abs() < 1e-8,
                "Mismatch in VI- at index {}, expected {}, got {}",
                i,
                expected,
                val
            );
        }
    }

    #[test]
    fn test_vi_with_default_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = ViInput::with_default_candles(&candles);
        let result = vi(&input).expect("Failed to calculate VI with default params");
        assert_eq!(result.plus.len(), candles.close.len());
        assert_eq!(result.minus.len(), candles.close.len());
    }

    #[test]
    fn test_vi_error_handling() {
        let empty_data: [f64; 0] = [];
        let params = ViParams { period: Some(14) };

        let input = ViInput::from_slices(&empty_data, &empty_data, &empty_data, params.clone());
        let result = vi(&input);
        assert!(result.is_err());

        let single_val = [10.0];
        let input_small = ViInput::from_slices(&single_val, &single_val, &single_val, params);
        let result_small = vi(&input_small);
        assert!(result_small.is_err());
    }
}
