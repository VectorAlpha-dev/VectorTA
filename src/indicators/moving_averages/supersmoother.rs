use crate::utilities::data_loader::{source_type, Candles};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub enum SuperSmootherData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SuperSmootherOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SuperSmootherParams {
    pub period: Option<usize>,
}

impl Default for SuperSmootherParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SuperSmootherInput<'a> {
    pub data: SuperSmootherData<'a>,
    pub params: SuperSmootherParams,
}

impl<'a> SuperSmootherInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: SuperSmootherParams,
    ) -> Self {
        Self {
            data: SuperSmootherData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: SuperSmootherParams) -> Self {
        Self {
            data: SuperSmootherData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SuperSmootherData::Candles {
                candles,
                source: "close",
            },
            params: SuperSmootherParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SuperSmootherParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SuperSmootherError {
    #[error("Swma: All input data for Super Smoother are NaN.")]
    AllValuesNaN,
    #[error("Smwa: Invalid period for Super Smoother: period = {period}, data length = {data_len}. Period must be >= 1 and no greater than the data length.")]
    InvalidPeriod { period: usize, data_len: usize },
    #[error("Swma: Empty data provided for Super Smoother.")]
    EmptyData,
}

#[inline]
pub fn supersmoother(
    input: &SuperSmootherInput,
) -> Result<SuperSmootherOutput, SuperSmootherError> {
    let data: &[f64] = match &input.data {
        SuperSmootherData::Candles { candles, source } => source_type(candles, source),
        SuperSmootherData::Slice(slice) => slice,
    };

    if data.is_empty() {
        return Err(SuperSmootherError::EmptyData);
    }

    let period = input.get_period();
    let len = data.len();

    if period < 1 || period > len {
        return Err(SuperSmootherError::InvalidPeriod {
            period,
            data_len: len,
        });
    }
    let mut output_values = vec![0.0; len];

    let a = (-1.414_f64 * PI / (period as f64)).exp();
    let a_sq = a * a;
    let b = 2.0 * a * (1.414_f64 * PI / (period as f64)).cos();
    let c = (1.0 + a_sq - b) * 0.5;

    output_values[0] = data[0];
    if len > 1 {
        output_values[1] = data[1];
    }

    for i in 2..len {
        let prev_1 = output_values[i - 1];
        let prev_2 = output_values[i - 2];
        let d_i = data[i];
        let d_im1 = data[i - 1];
        output_values[i] = c * (d_i + d_im1) + b * prev_1 - a_sq * prev_2;
    }

    Ok(SuperSmootherOutput {
        values: output_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_supersmoother_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = SuperSmootherParams { period: None };
        let input_default = SuperSmootherInput::from_candles(&candles, "close", default_params);
        let output_default =
            supersmoother(&input_default).expect("Failed supersmoother with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let custom_params = SuperSmootherParams { period: Some(20) };
        let input_custom = SuperSmootherInput::from_candles(&candles, "hlc3", custom_params);
        let output_custom =
            supersmoother(&input_custom).expect("Failed supersmoother with period=20, source=hlc3");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_supersmoother_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = SuperSmootherParams { period: Some(14) };
        let input = SuperSmootherInput::from_candles(&candles, "close", params);
        let result = supersmoother(&input).expect("Failed to calculate SuperSmoother");

        let out_vals = &result.values;
        let expected_last_five = [
            59140.98229179739,
            59172.03593376982,
            59179.40342783722,
            59171.22758152845,
            59127.859841077094,
        ];

        assert!(out_vals.len() >= 5);
        assert_eq!(out_vals.len(), close_prices.len());

        let start_idx = out_vals.len() - 5;
        let actual_last_five = &out_vals[start_idx..];
        for (i, (&actual, &expected)) in
            actual_last_five.iter().zip(&expected_last_five).enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "Mismatch at index {}: expected {}, got {}, diff={}",
                i,
                expected,
                actual,
                diff
            );
        }
    }
    #[test]
    fn test_supersmoother_params_with_default_params() {
        let params = SuperSmootherParams::default();
        assert_eq!(params.period, Some(14));
    }

    #[test]
    fn test_supersmoother_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = SuperSmootherInput::with_default_candles(&candles);
        match input.data {
            SuperSmootherData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected SuperSmootherData::Candles variant"),
        }
    }

    #[test]
    fn test_supersmoother_invalid_period() {
        let data = [10.0, 20.0, 30.0];
        let params = SuperSmootherParams { period: Some(0) };
        let input = SuperSmootherInput::from_slice(&data, params);
        let result = supersmoother(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_supersmoother_empty_data() {
        let data: [f64; 0] = [];
        let params = SuperSmootherParams { period: Some(14) };
        let input = SuperSmootherInput::from_slice(&data, params);
        let result = supersmoother(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_supersmoother_small_data() {
        let data = [42.0];
        let params = SuperSmootherParams { period: Some(14) };
        let input = SuperSmootherInput::from_slice(&data, params);
        let result = supersmoother(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_supersmoother_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params_1 = SuperSmootherParams { period: Some(14) };
        let input_1 = SuperSmootherInput::from_candles(&candles, "close", params_1);
        let result_1 = supersmoother(&input_1).expect("Failed first smoother");
        assert_eq!(result_1.values.len(), candles.close.len());
        let params_2 = SuperSmootherParams { period: Some(10) };
        let input_2 = SuperSmootherInput::from_slice(&result_1.values, params_2);
        let result_2 = supersmoother(&input_2).expect("Failed second smoother");
        assert_eq!(result_2.values.len(), result_1.values.len());
        if result_2.values.len() > 240 {
            for i in 240..result_2.values.len() {
                assert!(result_2.values[i].is_finite());
            }
        }
    }

    #[test]
    fn test_supersmoother_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = SuperSmootherParams { period: Some(14) };
        let input = SuperSmootherInput::from_candles(&candles, "close", params);
        let result = supersmoother(&input).expect("Failed supersmoother");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(result.values[i].is_finite());
            }
        }
    }
}
