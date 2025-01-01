use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub enum ReflexData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct ReflexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ReflexParams {
    pub period: Option<usize>,
}

impl Default for ReflexParams {
    fn default() -> Self {
        Self { period: Some(20) }
    }
}

#[derive(Debug, Clone)]
pub struct ReflexInput<'a> {
    pub data: ReflexData<'a>,
    pub params: ReflexParams,
}

impl<'a> ReflexInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: ReflexParams) -> Self {
        Self {
            data: ReflexData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: ReflexParams) -> Self {
        Self {
            data: ReflexData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: ReflexData::Candles {
                candles,
                source: "close",
            },
            params: ReflexParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| ReflexParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReflexError {
    #[error("No data available for Reflex.")]
    NoData,

    #[error("Reflex period must be >=2. Provided period was {period}")]
    InvalidPeriod { period: usize },

    #[error("Not enough data: needed {needed}, found {found}")]
    NotEnoughData { needed: usize, found: usize },

    #[error("All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn reflex(input: &ReflexInput) -> Result<ReflexOutput, ReflexError> {
    let data: &[f64] = match &input.data {
        ReflexData::Candles { candles, source } => source_type(candles, source),
        ReflexData::Slice(slice) => slice,
    };
    let len = data.len();
    let period = input.get_period();

    if len == 0 {
        return Err(ReflexError::NoData);
    }
    if !data.iter().any(|&x| !x.is_nan()) {
        return Err(ReflexError::AllValuesNaN);
    }
    if period < 2 {
        return Err(ReflexError::InvalidPeriod { period });
    }
    if period > len {
        return Err(ReflexError::NotEnoughData {
            needed: period,
            found: len,
        });
    }

    let half_period = (period / 2).max(1);
    let a = (-1.414_f64 * PI / half_period as f64).exp();
    let a_sq = a * a;
    let b = 2.0 * a * (1.414_f64 * PI / half_period as f64).cos();
    let c = (1.0 + a_sq - b) * 0.5;

    let mut ssf = vec![0.0; len];
    let mut reflex = vec![0.0; len];
    let mut ms = vec![0.0; len];
    let mut sums = vec![0.0; len];

    if len > 0 {
        ssf[0] = data[0];
    }
    if len > 1 {
        ssf[1] = data[1];
    }

    let period_f = period as f64;

    for i in 2..len {
        let d_i = data[i];
        let d_im1 = data[i - 1];
        let prev_ssf1 = ssf[i - 1];
        let prev_ssf2 = ssf[i - 2];
        let ssf_i = c * (d_i + d_im1) + b * prev_ssf1 - a_sq * prev_ssf2;
        ssf[i] = ssf_i;

        if i >= period {
            let slope = (ssf[i - period] - ssf_i) / period_f;

            let mut my_sum = 0.0;
            for t in 1..=period {
                let pred = ssf_i + slope * (t as f64);
                let past = ssf[i - t];
                my_sum += pred - past;
            }
            my_sum /= period_f;
            sums[i] = my_sum;
            let ms_im1 = ms[i - 1];
            let my_sum_sq = my_sum * my_sum;
            let ms_i = 0.04 * my_sum_sq + 0.96 * ms_im1;
            ms[i] = ms_i;

            reflex[i] = if ms_i > 0.0 {
                my_sum / ms_i.sqrt()
            } else {
                0.0
            };
        }
    }

    Ok(ReflexOutput { values: reflex })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_reflex_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = ReflexParams { period: None };
        let input = ReflexInput::from_candles(&candles, "close", default_params);
        let output = reflex(&input).expect("Failed Reflex with default params");
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_14 = ReflexParams { period: Some(14) };
        let input2 = ReflexInput::from_candles(&candles, "hl2", params_period_14);
        let output2 = reflex(&input2).expect("Failed Reflex with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = ReflexParams { period: Some(30) };
        let input3 = ReflexInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = reflex(&input3).expect("Failed Reflex fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }

    #[test]
    fn test_reflex_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = ReflexParams::default();
        let input = ReflexInput::from_candles(&candles, "close", default_params);
        let result = reflex(&input).expect("Failed to calculate Reflex");
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "Output size mismatch"
        );
        let len = result.values.len();
        let expected_last_five = [
            0.8085220962465361,
            0.445264715886137,
            0.13861699036615063,
            -0.03598639652007061,
            -0.224906760543743,
        ];
        assert!(len >= 5, "Not enough data for the test");
        let start_idx = len - 5;
        let last_five = &result.values[start_idx..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-7,
                "Reflex mismatch at offset {}, expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }

    #[test]
    fn test_reflex_params_with_default_params() {
        let default_params = ReflexParams::default();
        assert_eq!(default_params.period, Some(20));
    }

    #[test]
    fn test_reflex_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = ReflexInput::with_default_candles(&candles);
        match input.data {
            ReflexData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected ReflexData::Candles variant"),
        }
    }

    #[test]
    fn test_reflex_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = ReflexParams { period: Some(0) };
        let input = ReflexInput::from_slice(&input_data, params);
        let result = reflex(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Reflex period must be >=2"),
                "Unexpected error: {}",
                e
            );
        }
    }

    #[test]
    fn test_reflex_with_period_less_than_two() {
        let input_data = [10.0, 20.0, 30.0];
        let params = ReflexParams { period: Some(1) };
        let input = ReflexInput::from_slice(&input_data, params);
        let result = reflex(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_reflex_very_small_data_set() {
        let input_data = [42.0];
        let params = ReflexParams { period: Some(2) };
        let input = ReflexInput::from_slice(&input_data, params);
        let result = reflex(&input);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.values.len(), 1);
        assert_eq!(output.values[0], 0.0);
    }

    #[test]
    fn test_reflex_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = ReflexParams { period: Some(14) };
        let first_input = ReflexInput::from_candles(&candles, "close", first_params);
        let first_result = reflex(&first_input).expect("Failed to calculate first Reflex");
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = ReflexParams { period: Some(10) };
        let second_input = ReflexInput::from_slice(&first_result.values, second_params);
        let second_result = reflex(&second_input).expect("Failed to calculate second Reflex");
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 240..second_result.values.len() {
            assert!(second_result.values[i].is_finite());
        }
    }

    #[test]
    fn test_reflex_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let period = 14;
        let params = ReflexParams {
            period: Some(period),
        };
        let input = ReflexInput::from_candles(&candles, "close", params);
        let result = reflex(&input).expect("Failed to calculate Reflex");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > period {
            for i in period..result.values.len() {
                assert!(
                    result.values[i].is_finite(),
                    "Unexpected NaN at index {}",
                    i
                );
            }
        }
    }
}
