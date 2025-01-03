/// # Sine Weighted Moving Average (SINWMA)
///
/// A specialized weighted moving average that applies sine coefficients to
/// the most recent data points. The sine values decrease from `sin(π/(period+1))`
/// at the earliest point up to `sin(π * period / (period+1))` at the most recent
/// point in the window, emphasizing nearer data. This approach can offer a smooth
/// yet responsive curve.
///
/// ## Parameters
/// - **period**: Number of data points to include in each weighted sum (defaults to 14).
///
/// ## Errors
/// - **EmptyData**: sinwma: The input data slice is empty.
/// - **InvalidPeriod**: sinwma: `period` is zero or greater than the data length.
/// - **ZeroSumSines**: sinwma: Sum of the sine coefficients is zero or extremely close to zero,
///   preventing a valid weighted average.
///
/// ## Returns
/// - **`Ok(SinWmaOutput)`** on success, containing a `Vec<f64>` that mirrors the input length.
/// - **`Err(SinWmaError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub enum SinWmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SinWmaParams {
    pub period: Option<usize>,
}

impl Default for SinWmaParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SinWmaInput<'a> {
    pub data: SinWmaData<'a>,
    pub params: SinWmaParams,
}

impl<'a> SinWmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: SinWmaParams) -> Self {
        Self {
            data: SinWmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: SinWmaParams) -> Self {
        Self {
            data: SinWmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SinWmaData::Candles {
                candles,
                source: "close",
            },
            params: SinWmaParams::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SinWmaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct SinWmaOutput {
    pub values: Vec<f64>,
}

#[inline(always)]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SinWmaError {
    #[error("sinwma: Data slice is empty for SINWMA calculation.")]
    EmptyData,
    #[error("sinwma: Invalid period for SINWMA calculation. period = {period}, data length = {data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error(
        "sinwma: Sum of sines is zero or too close to zero, cannot compute SINWMA. sum_sines = {sum_sines}"
    )]
    ZeroSumSines { sum_sines: f64 },
}

#[inline]
pub fn sinwma(input: &SinWmaInput) -> Result<SinWmaOutput, SinWmaError> {
    let data: &[f64] = match &input.data {
        SinWmaData::Candles { candles, source } => source_type(candles, source),
        SinWmaData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(SinWmaError::EmptyData);
    }

    let period = input.get_period();
    if period == 0 || period > data.len() {
        return Err(SinWmaError::InvalidPeriod {
            period,
            data_len: data.len(),
        });
    }

    let mut sines = Vec::with_capacity(period);
    let mut sum_sines = 0.0;
    for k in 0..period {
        let angle = (k as f64 + 1.0) * PI / (period as f64 + 1.0);
        let val = angle.sin();
        sum_sines += val;
        sines.push(val);
    }

    if sum_sines.abs() < f64::EPSILON {
        return Err(SinWmaError::ZeroSumSines { sum_sines });
    }

    let inv_sum = 1.0 / sum_sines;
    for w in &mut sines {
        *w *= inv_sum;
    }

    let len = data.len();
    let mut sinwma_values = vec![f64::NAN; len];

    for i in (period - 1)..len {
        let start_idx = i + 1 - period;
        let data_window = &data[start_idx..(start_idx + period)];
        let value = dot_product(data_window, &sines);
        sinwma_values[i] = value;
    }

    Ok(SinWmaOutput {
        values: sinwma_values,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_sinwma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let params = SinWmaParams { period: Some(14) };
        let input = SinWmaInput::from_candles(&candles, "close", params);
        let result = sinwma(&input).expect("Failed to calculate SINWMA");
        assert_eq!(result.values.len(), close_prices.len());
        let expected_last_five = [
            59376.72903536103,
            59300.76862770367,
            59229.27622157621,
            59178.48781774477,
            59154.66580703081,
        ];
        assert!(result.values.len() >= 5);
        let start_index = result.values.len() - 5;
        let last_five = &result.values[start_index..];
        for (i, &value) in last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "SINWMA mismatch at {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }

    #[test]
    fn test_sinwma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = SinWmaParams { period: None };
        let input = SinWmaInput::from_candles(&candles, "close", default_params);
        let output = sinwma(&input).expect("Failed SINWMA with default params");
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_10 = SinWmaParams { period: Some(10) };
        let input2 = SinWmaInput::from_candles(&candles, "hl2", params_period_10);
        let output2 = sinwma(&input2).expect("Failed SINWMA with period=10, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = SinWmaParams { period: Some(20) };
        let input3 = SinWmaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = sinwma(&input3).expect("Failed SINWMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }

    #[test]
    fn test_sinwma_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = SinWmaInput::with_default_candles(&candles);
        match input.data {
            SinWmaData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected SinWmaData::Candles"),
        }
        let period = input.get_period();
        assert_eq!(period, 14);
    }

    #[test]
    fn test_sinwma_with_default_params() {
        let default_params = SinWmaParams::default();
        assert_eq!(default_params.period, Some(14));
    }

    #[test]
    fn test_sinwma_with_no_data() {
        let data: [f64; 0] = [];
        let params = SinWmaParams { period: Some(14) };
        let input = SinWmaInput::from_slice(&data, params);
        let result = sinwma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_sinwma_very_small_data_set() {
        let data = [42.0];
        let params = SinWmaParams { period: Some(14) };
        let input = SinWmaInput::from_slice(&data, params);
        let result = sinwma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_sinwma_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_input =
            SinWmaInput::from_candles(&candles, "close", SinWmaParams { period: Some(14) });
        let first_result = sinwma(&first_input).expect("First SINWMA failed");
        let second_input =
            SinWmaInput::from_slice(&first_result.values, SinWmaParams { period: Some(5) });
        let second_result = sinwma(&second_input).expect("Second SINWMA failed");
        assert_eq!(second_result.values.len(), first_result.values.len());
        for &val in second_result.values.iter().skip(240) {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_sinwma_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = SinWmaInput::from_candles(&candles, "close", SinWmaParams { period: Some(14) });
        let result = sinwma(&input).expect("Failed to calculate SINWMA");
        assert_eq!(result.values.len(), candles.close.len());
        for val in result.values.iter().skip(14) {
            assert!(val.is_finite());
        }
    }
}
