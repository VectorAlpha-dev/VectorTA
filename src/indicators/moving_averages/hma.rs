/// # Hull Moving Average (HMA)
///
/// The Hull Moving Average (HMA) is a moving average technique that aims to
/// minimize lag while providing smooth output. It combines Weighted Moving
/// Averages of different lengths—namely `period/2` and `period`—to form an
/// intermediate difference. A final Weighted Moving Average is then applied
/// using the integer part of `sqrt(period)`, yielding a responsive trend
/// indication with reduced lag.
///
/// ## Parameters
/// - **period**: Window size (number of data points). (defaults to 5)
///
/// ## Errors
/// - **NoData**: hma: No data provided.
/// - **AllValuesNaN**: hma: All input data values are `NaN`.
/// - **InvalidPeriod**: hma: `period` is zero or exceeds the data length.
/// - **ZeroHalf**: hma: Cannot calculate half of period.
/// - **ZeroSqrtPeriod**: hma: Cannot calculate sqrt of period.
///
/// ## Returns
/// - **`Ok(HmaOutput)`** on success, containing a `Vec<f64>` of length matching the input.
/// - **`Err(HmaError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};

#[derive(Debug, Clone)]
pub enum HmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HmaParams {
    pub period: Option<usize>,
}

impl Default for HmaParams {
    fn default() -> Self {
        Self { period: Some(5) }
    }
}

#[derive(Debug, Clone)]
pub struct HmaInput<'a> {
    pub data: HmaData<'a>,
    pub params: HmaParams,
}

impl<'a> HmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: HmaParams) -> Self {
        Self {
            data: HmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: HmaParams) -> Self {
        Self {
            data: HmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: HmaData::Candles {
                candles,
                source: "close",
            },
            params: HmaParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| HmaParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum HmaError {
    #[error("hma: No data provided.")]
    NoData,

    #[error("hma: All values are NaN.")]
    AllValuesNaN,

    #[error("hma: Invalid period: period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },

    #[error("hma: Cannot calculate half of period: period = {period}")]
    ZeroHalf { period: usize },

    #[error("hma: Cannot calculate sqrt of period: period = {period}")]
    ZeroSqrtPeriod { period: usize },
}

#[inline]
pub fn hma(input: &HmaInput) -> Result<HmaOutput, HmaError> {
    let data = match &input.data {
        HmaData::Candles { candles, source } => source_type(candles, source),
        HmaData::Slice(slice) => slice,
    };

    let len = data.len();
    if len == 0 {
        return Err(HmaError::NoData);
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => {
            return Ok(HmaOutput {
                values: vec![f64::NAN; len],
            });
        }
    };

    let period = input.get_period();
    if period == 0 || period > len {
        return Err(HmaError::InvalidPeriod {
            period,
            data_len: len,
        });
    }

    let mut values = vec![f64::NAN; len];

    let half = period / 2;
    if half == 0 {
        return Err(HmaError::ZeroHalf { period });
    }

    let sqrtp = (period as f64).sqrt().floor() as usize;
    if sqrtp == 0 {
        return Err(HmaError::ZeroSqrtPeriod { period });
    }

    let m = len - first_valid_idx;
    if period > m {
        return Ok(HmaOutput { values });
    }

    let sum_w_half = (half * (half + 1)) >> 1;
    let denom_half = sum_w_half as f64;

    let sum_w_full = (period * (period + 1)) >> 1;
    let denom_full = sum_w_full as f64;

    let sum_w_sqrt = (sqrtp * (sqrtp + 1)) >> 1;
    let denom_sqrt = sum_w_sqrt as f64;

    let lookback_half = half - 1;
    let lookback_full = period - 1;

    let half_f = half as f64;
    let period_f = period as f64;
    let sqrtp_f = sqrtp as f64;

    let mut wma_half = vec![f64::NAN; len];
    let mut wma_full = vec![f64::NAN; len];

    let mut period_sub_half = 0.0;
    let mut period_sum_half = 0.0;
    let mut in_idx = 0;
    let mut i_half = 1;

    while in_idx < lookback_half {
        let val = data[first_valid_idx + in_idx];
        period_sub_half += val;
        period_sum_half += val * (i_half as f64);
        in_idx += 1;
        i_half += 1;
    }

    let mut period_sub_full = 0.0;
    let mut period_sum_full = 0.0;
    let mut in_idx_full = 0;
    let mut i_full = 1;

    while in_idx_full < lookback_full {
        let val = data[first_valid_idx + in_idx_full];
        period_sub_full += val;
        period_sum_full += val * (i_full as f64);
        in_idx_full += 1;
        i_full += 1;
    }

    if in_idx < m {
        let val = data[first_valid_idx + in_idx];
        in_idx += 1;
        period_sub_half += val;
        period_sum_half += val * half_f;

        wma_half[first_valid_idx + lookback_half] = period_sum_half / denom_half;
        period_sum_half -= period_sub_half;

        let mut trailing_idx_half = 1;
        let mut trailing_value_half = data[first_valid_idx];

        if in_idx_full < m {
            let valf = data[first_valid_idx + in_idx_full];
            in_idx_full += 1;
            period_sub_full += valf;
            period_sum_full += valf * period_f;

            wma_full[first_valid_idx + lookback_full] = period_sum_full / denom_full;
            period_sum_full -= period_sub_full;

            let mut trailing_idx_full = 1;
            let mut trailing_value_full = data[first_valid_idx];

            while in_idx < m || in_idx_full < m {
                if in_idx < m {
                    let new_val = data[first_valid_idx + in_idx];
                    in_idx += 1;

                    period_sub_half += new_val;
                    period_sub_half -= trailing_value_half;
                    period_sum_half += new_val * half_f;

                    trailing_value_half = data[first_valid_idx + trailing_idx_half];
                    trailing_idx_half += 1;

                    wma_half[first_valid_idx + (in_idx - 1)] = period_sum_half / denom_half;
                    period_sum_half -= period_sub_half;
                }

                if in_idx_full < m {
                    let new_valf = data[first_valid_idx + in_idx_full];
                    in_idx_full += 1;

                    period_sub_full += new_valf;
                    period_sub_full -= trailing_value_full;
                    period_sum_full += new_valf * period_f;

                    trailing_value_full = data[first_valid_idx + trailing_idx_full];
                    trailing_idx_full += 1;

                    wma_full[first_valid_idx + (in_idx_full - 1)] = period_sum_full / denom_full;
                    period_sum_full -= period_sub_full;
                }
            }
        }
    }

    let mut diff = vec![f64::NAN; len];
    for i in 0..len {
        let a = wma_half[i];
        let b = wma_full[i];
        if a.is_finite() && b.is_finite() {
            diff[i] = 2.0 * a - b;
        }
    }

    let mut wma_sqrt = vec![f64::NAN; len];
    {
        let lookback_sqrt = sqrtp - 1;
        let mut period_sub_sqrt = 0.0;
        let mut period_sum_sqrt = 0.0;
        let mut in_idx_sqrt = 0;
        let mut i_s = 1;

        while in_idx_sqrt < lookback_sqrt {
            let val = diff[first_valid_idx + in_idx_sqrt];
            if val.is_finite() {
                period_sub_sqrt += val;
                period_sum_sqrt += val * (i_s as f64);
            }
            in_idx_sqrt += 1;
            i_s += 1;
        }

        if in_idx_sqrt < m {
            let val = diff[first_valid_idx + in_idx_sqrt];
            in_idx_sqrt += 1;
            if val.is_finite() {
                period_sub_sqrt += val;
                period_sum_sqrt += val * sqrtp_f;
            }
            let mut trailing_idx_sqrt = 1;
            let mut trailing_value_sqrt = diff[first_valid_idx];

            wma_sqrt[first_valid_idx + lookback_sqrt] = if trailing_value_sqrt.is_finite() {
                period_sum_sqrt / denom_sqrt
            } else {
                f64::NAN
            };
            period_sum_sqrt -= period_sub_sqrt;

            while in_idx_sqrt < m {
                let new_val = diff[first_valid_idx + in_idx_sqrt];
                in_idx_sqrt += 1;

                if new_val.is_finite() {
                    period_sub_sqrt += new_val;
                }
                if trailing_value_sqrt.is_finite() {
                    period_sub_sqrt -= trailing_value_sqrt;
                }
                if new_val.is_finite() {
                    period_sum_sqrt += new_val * sqrtp_f;
                }

                trailing_value_sqrt = diff[first_valid_idx + trailing_idx_sqrt];
                trailing_idx_sqrt += 1;

                wma_sqrt[first_valid_idx + (in_idx_sqrt - 1)] = if period_sub_sqrt != 0.0 {
                    period_sum_sqrt / denom_sqrt
                } else {
                    f64::NAN
                };
                period_sum_sqrt -= period_sub_sqrt;
            }
        }
    }

    values.copy_from_slice(&wma_sqrt);

    Ok(HmaOutput { values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_hma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = HmaParams { period: None };
        let input_default = HmaInput::from_candles(&candles, "close", default_params);
        let output_default = hma(&input_default).expect("Failed hma with default params");
        assert_eq!(output_default.values.len(), candles.close.len());
        let params_period = HmaParams { period: Some(10) };
        let input_period = HmaInput::from_candles(&candles, "hl2", params_period);
        let output_period = hma(&input_period).expect("Failed hma with period=10");
        assert_eq!(output_period.values.len(), candles.close.len());
    }

    #[test]
    fn test_hma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HmaInput::with_default_candles(&candles);
        let result = hma(&input).expect("Failed hma");
        let expected_last_five = [
            59334.13333336847,
            59201.4666667018,
            59047.77777781293,
            59048.71111114628,
            58803.44444447962,
        ];
        assert!(result.values.len() >= 5);
        assert_eq!(result.values.len(), candles.close.len());
        let start = result.values.len() - 5;
        let last_five = &result.values[start..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!((val - exp).abs() < 1e-3);
        }
    }

    #[test]
    fn test_hma_params_with_default_params() {
        let default_params = HmaParams::default();
        assert_eq!(default_params.period, Some(5));
    }

    #[test]
    fn test_hma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HmaInput::with_default_candles(&candles);
        match input.data {
            HmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected HmaData::Candles variant"),
        }
        assert_eq!(input.params.period, Some(5));
    }

    #[test]
    fn test_hma_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = HmaParams { period: Some(0) };
        let input = HmaInput::from_slice(&input_data, params);
        let result = hma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_hma_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = HmaParams { period: Some(10) };
        let input = HmaInput::from_slice(&input_data, params);
        let result = hma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_hma_very_small_data_set() {
        let input_data = [42.0];
        let params = HmaParams { period: Some(5) };
        let input = HmaInput::from_slice(&input_data, params);
        let result = hma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_hma_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = HmaParams { period: Some(5) };
        let first_input = HmaInput::from_candles(&candles, "close", first_params);
        let first_result = hma(&first_input).expect("Failed first hma");
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = HmaParams { period: Some(3) };
        let second_input = HmaInput::from_slice(&first_result.values, second_params);
        let second_result = hma(&second_input).expect("Failed second hma");
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(!second_result.values[i].is_nan());
            }
        }
    }

    #[test]
    fn test_hma_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = HmaParams::default();
        let input = HmaInput::from_candles(&candles, "close", params);
        let result = hma(&input).expect("Failed hma");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 50 {
            for i in 50..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
