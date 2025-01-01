use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub enum SuperSmoother3PoleData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleParams {
    pub period: Option<usize>,
}

impl Default for SuperSmoother3PoleParams {
    fn default() -> Self {
        Self { period: Some(14) }
    }
}

#[derive(Debug, Clone)]
pub struct SuperSmoother3PoleInput<'a> {
    pub data: SuperSmoother3PoleData<'a>,
    pub params: SuperSmoother3PoleParams,
}

impl<'a> SuperSmoother3PoleInput<'a> {
    pub fn from_candles(
        candles: &'a Candles,
        source: &'a str,
        params: SuperSmoother3PoleParams,
    ) -> Self {
        Self {
            data: SuperSmoother3PoleData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: SuperSmoother3PoleParams) -> Self {
        Self {
            data: SuperSmoother3PoleData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: SuperSmoother3PoleData::Candles {
                candles,
                source: "close",
            },
            params: SuperSmoother3PoleParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| SuperSmoother3PoleParams::default().period.unwrap())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SuperSmoother3PoleError {
    #[error("No data provided for 3-pole SuperSmoother.")]
    EmptyData,
    #[error("Period must be >= 1 for 3-pole SuperSmoother: period = {period}")]
    InvalidPeriod { period: usize },
    #[error("All values are NaN.")]
    AllValuesNaN,
}

#[inline]
pub fn supersmoother_3_pole(
    input: &SuperSmoother3PoleInput,
) -> Result<SuperSmoother3PoleOutput, SuperSmoother3PoleError> {
    let data: &[f64] = match &input.data {
        SuperSmoother3PoleData::Candles { candles, source } => source_type(candles, source),
        SuperSmoother3PoleData::Slice(slice) => slice,
    };

    let n = data.len();
    let period = input.get_period();

    if data.is_empty() {
        return Err(SuperSmoother3PoleError::EmptyData);
    }

    if period < 1 {
        return Err(SuperSmoother3PoleError::InvalidPeriod { period });
    }

    let first_valid_idx = data.iter().position(|&x| !x.is_nan());
    if first_valid_idx.is_none() {
        return Err(SuperSmoother3PoleError::AllValuesNaN);
    }

    let mut output = vec![0.0; n];

    let a = (-PI / period as f64).exp();
    let b = 2.0 * a * (1.738_f64 * PI / period as f64).cos();
    let c = a * a;

    if n > 0 {
        output[0] = data[0];
    }
    if n > 1 {
        output[1] = data[1];
    }
    if n > 2 {
        output[2] = data[2];
    }

    let coef_source = 1.0 - c * c - b + b * c;
    let coef_prev1 = b + c;
    let coef_prev2 = -c - b * c;
    let coef_prev3 = c * c;

    for i in 3..n {
        let d_i = data[i];
        let o_im1 = output[i - 1];
        let o_im2 = output[i - 2];
        let o_im3 = output[i - 3];

        output[i] =
            coef_source * d_i + coef_prev1 * o_im1 + coef_prev2 * o_im2 + coef_prev3 * o_im3;
    }

    Ok(SuperSmoother3PoleOutput { values: output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_supersmoother_3_pole_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = SuperSmoother3PoleParams { period: None };
        let input_default =
            SuperSmoother3PoleInput::from_candles(&candles, "close", default_params);
        let output_default = supersmoother_3_pole(&input_default)
            .expect("Failed 3-pole SuperSmoother with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_period_20 = SuperSmoother3PoleParams { period: Some(20) };
        let input_period_20 =
            SuperSmoother3PoleInput::from_candles(&candles, "hl2", params_period_20);
        let output_period_20 = supersmoother_3_pole(&input_period_20)
            .expect("Failed 3-pole SuperSmoother with period=20, source=hl2");
        assert_eq!(output_period_20.values.len(), candles.close.len());

        let params_custom = SuperSmoother3PoleParams { period: Some(10) };
        let input_custom = SuperSmoother3PoleInput::from_candles(&candles, "hlc3", params_custom);
        let output_custom =
            supersmoother_3_pole(&input_custom).expect("Failed 3-pole SuperSmoother fully custom");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_supersmoother_3_pole_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let params = SuperSmoother3PoleParams { period: Some(14) };
        let input = SuperSmoother3PoleInput::from_candles(&candles, "close", params);
        let result = supersmoother_3_pole(&input).expect("Failed 3-pole SS calculation");
        let values = &result.values;
        let expected_last_five = [
            59072.13481064446,
            59089.08032603,
            59111.35711851466,
            59133.14402399381,
            59121.91820047289,
        ];
        assert!(values.len() >= 5);
        assert_eq!(values.len(), close_prices.len());
        let start_idx = values.len() - 5;
        let last_five = &values[start_idx..];
        for (i, (&actual, &expected)) in last_five.iter().zip(expected_last_five.iter()).enumerate()
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-8,
                "3-pole SuperSmoother mismatch at index {}: expected {}, got {}, diff {}",
                i,
                expected,
                actual,
                diff
            );
        }
    }
    #[test]
    fn test_supersmoother_3_pole_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = SuperSmoother3PoleInput::with_default_candles(&candles);
        match input.data {
            SuperSmoother3PoleData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected SuperSmoother3PoleData::Candles"),
        }
    }

    #[test]
    fn test_supersmoother_3_pole_with_default_params() {
        let default_params = SuperSmoother3PoleParams::default();
        assert_eq!(default_params.period, Some(14));
    }

    #[test]
    fn test_supersmoother_3_pole_with_no_data() {
        let data: [f64; 0] = [];
        let input = SuperSmoother3PoleInput::from_slice(
            &data,
            SuperSmoother3PoleParams { period: Some(14) },
        );
        let result = supersmoother_3_pole(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_supersmoother_3_pole_very_small_data_set() {
        let data = [10.0, 20.0];
        let input = SuperSmoother3PoleInput::from_slice(
            &data,
            SuperSmoother3PoleParams { period: Some(14) },
        );
        let result =
            supersmoother_3_pole(&input).expect("Very small data 3-pole SS should succeed");
        assert_eq!(result.values.len(), data.len());
        assert_eq!(result.values[0], 10.0);
        if result.values.len() > 1 {
            assert_eq!(result.values[1], 20.0);
        }
    }

    #[test]
    fn test_supersmoother_3_pole_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_input = SuperSmoother3PoleInput::from_candles(
            &candles,
            "close",
            SuperSmoother3PoleParams { period: Some(14) },
        );
        let first_result = supersmoother_3_pole(&first_input).expect("First 3-pole SS failed");
        let second_input = SuperSmoother3PoleInput::from_slice(
            &first_result.values,
            SuperSmoother3PoleParams { period: Some(7) },
        );
        let second_result = supersmoother_3_pole(&second_input).expect("Second 3-pole SS failed");
        assert_eq!(second_result.values.len(), first_result.values.len());
        if second_result.values.len() > 240 {
            for i in 240..second_result.values.len() {
                assert!(second_result.values[i].is_finite());
            }
        }
    }

    #[test]
    fn test_supersmoother_3_pole_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = SuperSmoother3PoleInput::from_candles(
            &candles,
            "close",
            SuperSmoother3PoleParams { period: Some(14) },
        );
        let result = supersmoother_3_pole(&input).expect("3-pole SS calculation failed");
        assert_eq!(result.values.len(), candles.close.len());
        for (idx, &val) in result.values.iter().enumerate() {
            assert!(val.is_finite(), "NaN found at index {}", idx);
        }
    }
}
