/// # Chande Momentum Oscillator (CMO)
///
/// CMO is very similar to RSI, except for the final calculation step:
///
/// RSI = 100 * (avgGain / (avgGain + avgLoss))
/// CMO = 100 * ((avgGain - avgLoss) / (avgGain + avgLoss))
///
/// Gains and losses are typically calculated using Wilder's smoothing.
///
/// ## Parameters
/// - **period**: Length of the window for computing average gains/losses (defaults to 14).
/// - **source**: Candle field (e.g., `"close"`). Defaults to `"close"`.
///
/// ## Errors
/// - **EmptyData**: cmo: Input data slice is empty.
/// - **InvalidPeriod**: cmo: Period is zero or exceeds data length.
/// - **AllValuesNaN**: cmo: All input data values are `NaN`.
/// - **NotEnoughValidData**: cmo: Fewer than `period` valid data points remain after the first non-`NaN`.
///
/// ## Returns
/// - **`Ok(CmoOutput)`** on success, containing a `Vec<f64>` with the same length as the input data
///   (leading `NaN`s until there's enough data to compute CMO).
/// - **`Err(CmoError)`** otherwise.
use crate::utilities::data_loader::{source_type, Candles};
use thiserror::Error;

#[derive(Debug, Clone)]
pub enum CmoData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct CmoOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CmoParams {
    pub period: Option<usize>,
}

impl Default for CmoParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct CmoInput<'a> {
    pub data: CmoData<'a>,
    pub params: CmoParams,
}

impl<'a> CmoInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: CmoParams) -> Self {
        Self {
            data: CmoData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: CmoParams) -> Self {
        Self {
            data: CmoData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: CmoData::Candles {
                candles,
                source: "close",
            },
            params: CmoParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| CmoParams::default().period.unwrap())
    }
}

#[derive(Debug, Error)]
pub enum CmoError {
    #[error("cmo: Empty data provided.")]
    EmptyData,
    #[error("cmo: Invalid period: period={period}, data_len={data_len}")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("cmo: All values are NaN.")]
    AllValuesNaN,
    #[error("cmo: Not enough valid data: needed={needed}, valid={valid}")]
    NotEnoughValidData { needed: usize, valid: usize },
}

#[inline]
pub fn cmo(input: &CmoInput) -> Result<CmoOutput, CmoError> {
    let data: &[f64] = match &input.data {
        CmoData::Candles { candles, source } => source_type(candles, source),
        CmoData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(CmoError::EmptyData);
    }

    let period = input.get_period();
    let data_len = data.len();
    if period == 0 || period > data_len {
        return Err(CmoError::InvalidPeriod { period, data_len });
    }

    let first_valid_idx = match data.iter().position(|&x| !x.is_nan()) {
        Some(idx) => idx,
        None => return Err(CmoError::AllValuesNaN),
    };

    if (data_len - first_valid_idx) < period {
        return Err(CmoError::NotEnoughValidData {
            needed: period,
            valid: data_len - first_valid_idx,
        });
    }

    let mut cmo_values = vec![f64::NAN; data_len];
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    let mut prev_price = data[first_valid_idx];

    let start_loop = first_valid_idx + 1;
    let init_end = first_valid_idx + period;

    let period_f = period as f64;
    let period_m1 = (period - 1) as f64;
    let inv_period = 1.0 / period_f;

    for i in start_loop..data_len {
        let curr = data[i];
        let diff = curr - prev_price;
        prev_price = curr;

        let abs_diff = diff.abs();
        let gain = 0.5 * (diff + abs_diff);
        let loss = 0.5 * (abs_diff - diff);

        if i <= init_end {
            avg_gain += gain;
            avg_loss += loss;

            if i == init_end {
                avg_gain *= inv_period;
                avg_loss *= inv_period;

                let sum_gl = avg_gain + avg_loss;
                cmo_values[i] = if sum_gl != 0.0 {
                    100.0 * ((avg_gain - avg_loss) / sum_gl)
                } else {
                    0.0
                };
            }
        } else {
            avg_gain *= period_m1;
            avg_loss *= period_m1;

            avg_gain += gain;
            avg_loss += loss;

            avg_gain *= inv_period;
            avg_loss *= inv_period;

            let sum_gl = avg_gain + avg_loss;
            cmo_values[i] = if sum_gl != 0.0 {
                100.0 * ((avg_gain - avg_loss) / sum_gl)
            } else {
                0.0
            };
        }
    }

    Ok(CmoOutput { values: cmo_values })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_cmo_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = CmoParams { period: None };
        let input_default = CmoInput::from_candles(&candles, "close", default_params);
        let output_default = cmo(&input_default).expect("Failed CMO with default params");
        assert_eq!(output_default.values.len(), candles.close.len());
        let params_10 = CmoParams { period: Some(10) };
        let input_10 = CmoInput::from_candles(&candles, "hl2", params_10);
        let output_10 = cmo(&input_10).expect("Failed CMO with period=10, source=hl2");
        assert_eq!(output_10.values.len(), candles.close.len());
    }

    #[test]
    fn test_cmo_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let params = CmoParams { period: Some(14) };
        let input = CmoInput::from_candles(&candles, "close", params);
        let cmo_result = cmo(&input).expect("Failed to calculate CMO");

        assert_eq!(
            cmo_result.values.len(),
            candles.close.len(),
            "CMO length mismatch"
        );

        let expected_last_five = [
            -13.152504931406101,
            -14.649876201213106,
            -16.760170709240303,
            -14.274505732779227,
            -21.984038127126716,
        ];

        assert!(
            cmo_result.values.len() >= 5,
            "Not enough data to test the final 5 values"
        );

        let start_idx = cmo_result.values.len() - 5;
        let last_five = &cmo_result.values[start_idx..];
        for (i, &actual) in last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (actual - expected).abs() < 1e-6,
                "CMO mismatch at final 5 index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_cmo_params_with_default_params() {
        let default_params = CmoParams::default();
        assert_eq!(
            default_params.period,
            Some(14),
            "Expected period=14 in default params"
        );
    }

    #[test]
    fn test_cmo_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let input = CmoInput::with_default_candles(&candles);
        match input.data {
            CmoData::Candles { source, .. } => {
                assert_eq!(source, "close", "Expected 'close' as default source");
            }
            _ => panic!("Expected CmoData::Candles variant"),
        }
    }

    #[test]
    fn test_cmo_with_invalid_period() {
        let data = [10.0, 20.0, 30.0];
        let params_zero = CmoParams { period: Some(0) };
        let input_zero = CmoInput::from_slice(&data, params_zero);
        let result_zero = cmo(&input_zero);
        assert!(result_zero.is_err(), "Expected error for period=0");

        let params_big = CmoParams { period: Some(10) };
        let input_big = CmoInput::from_slice(&data, params_big);
        let result_big = cmo(&input_big);
        assert!(result_big.is_err(), "Expected error for period>data.len()");
    }

    #[test]
    fn test_cmo_all_nan() {
        let all_nan = [f64::NAN, f64::NAN, f64::NAN];
        let params = CmoParams::default();
        let input_nan = CmoInput::from_slice(&all_nan, params);
        let result = cmo(&input_nan);
        assert!(result.is_err(), "Expected AllValuesNaN error");
    }

    #[test]
    fn test_cmo_not_enough_valid_data() {
        let data = [f64::NAN, f64::NAN, 10.0, 11.0, 12.0];
        let params = CmoParams { period: Some(5) };
        let input = CmoInput::from_slice(&data, params);
        let result = cmo(&input);
        assert!(
            result.is_err(),
            "Expected not enough valid data for period=5"
        );
    }

    #[test]
    fn test_cmo_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = CmoParams { period: Some(14) };
        let first_input = CmoInput::from_candles(&candles, "close", first_params);
        let first_result = cmo(&first_input).expect("Failed to calculate first CMO");
        assert_eq!(
            first_result.values.len(),
            candles.close.len(),
            "First CMO length mismatch"
        );

        let second_params = CmoParams { period: Some(14) };
        let second_input = CmoInput::from_slice(&first_result.values, second_params);
        let second_result = cmo(&second_input).expect("Failed second CMO calculation");
        assert_eq!(
            second_result.values.len(),
            first_result.values.len(),
            "Second CMO length mismatch"
        );

        for i in 28..second_result.values.len() {
            assert!(
                !second_result.values[i].is_nan(),
                "Expected no NaN after index 28, found NaN at {}",
                i
            );
        }
    }
}
