use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub enum HighPass2Data<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct HighPass2Output {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct HighPass2Params {
    pub period: Option<usize>,
    pub k: Option<f64>,
}

impl HighPass2Params {
    pub fn with_default_params() -> Self {
        HighPass2Params {
            period: None,
            k: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HighPass2Input<'a> {
    pub data: HighPass2Data<'a>,
    pub params: HighPass2Params,
}

impl<'a> HighPass2Input<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: HighPass2Params) -> Self {
        Self {
            data: HighPass2Data::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: HighPass2Params) -> Self {
        Self {
            data: HighPass2Data::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: HighPass2Data::Candles {
                candles,
                source: "close",
            },
            params: HighPass2Params::with_default_params(),
        }
    }
}

pub fn high_pass_2_pole(input: &HighPass2Input) -> Result<HighPass2Output, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        HighPass2Data::Candles { candles, source } => source_type(candles, source),
        HighPass2Data::Slice(slice) => slice,
    };
    let len: usize = data.len();
    let period: usize = input.params.period.unwrap_or(48);
    let k: f64 = input.params.k.unwrap_or(0.707);
    if period < 2 || len == 0 {
        return Err("Invalid period (<2) or no data for 2-pole high-pass.".into());
    }

    let angle = 2.0 * PI * k / (period as f64);
    let sin_val = angle.sin();
    let cos_val = angle.cos();
    let alpha = 1.0 + ((sin_val - 1.0) / cos_val);

    let one_minus_alpha_half = 1.0 - alpha / 2.0;
    let c = one_minus_alpha_half * one_minus_alpha_half;

    let one_minus_alpha = 1.0 - alpha;
    let one_minus_alpha_sq = one_minus_alpha * one_minus_alpha;

    let mut out = vec![0.0; len];

    if len > 0 {
        out[0] = data[0];
    }
    if len > 1 {
        out[1] = data[1];
    }

    for i in 2..len {
        out[i] = c * data[i] - 2.0 * c * data[i - 1]
            + c * data[i - 2]
            + 2.0 * one_minus_alpha * out[i - 1]
            - one_minus_alpha_sq * out[i - 2];
    }

    Ok(HighPass2Output { values: out })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_high_pass_2_pole_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");
        let params = HighPass2Params {
            period: Some(48),
            k: Some(0.707),
        };
        let input = HighPass2Input::from_candles(&candles, "close", params);
        let result = high_pass_2_pole(&input).expect("Failed to calculate 2-pole high pass filter");
        let expected_last_five = [
            445.29073821108943,
            359.51467478973296,
            250.7236793408186,
            394.04381266217234,
            -52.65414073315134,
        ];
        assert!(result.values.len() >= 5);
        assert_eq!(result.values.len(), close_prices.len());
        let start_index = result.values.len() - 5;
        let actual_last_five = &result.values[start_index..];
        for (i, &actual) in actual_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            let diff = (actual - expected).abs();
            assert!(diff < 1e-6);
        }
    }
    #[test]
    fn test_high_pass_2_pole_params_with_default_params() {
        let params = HighPass2Params::with_default_params();
        assert_eq!(params.period, None);
        assert_eq!(params.k, None);
    }

    #[test]
    fn test_high_pass_2_pole_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = HighPass2Input::with_default_candles(&candles);
        match input.data {
            HighPass2Data::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected HighPass2Data::Candles"),
        }
        assert_eq!(input.params.period, None);
        assert_eq!(input.params.k, None);
    }

    #[test]
    fn test_high_pass_2_pole_invalid_period() {
        let data = [10.0, 20.0, 30.0];
        let params = HighPass2Params {
            period: Some(1),
            k: Some(0.707),
        };
        let input = HighPass2Input::from_slice(&data, params);
        let result = high_pass_2_pole(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_high_pass_2_pole_no_data() {
        let data: [f64; 0] = [];
        let params = HighPass2Params {
            period: Some(48),
            k: Some(0.707),
        };
        let input = HighPass2Input::from_slice(&data, params);
        let result = high_pass_2_pole(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_high_pass_2_pole_very_small_data_set() {
        let data = [42.0];
        let params = HighPass2Params {
            period: Some(2),
            k: Some(0.707),
        };
        let input = HighPass2Input::from_slice(&data, params);
        let result = high_pass_2_pole(&input).expect("Should handle single data with period=2");
        assert_eq!(result.values.len(), data.len());
        assert_eq!(result.values[0], data[0]);
    }

    #[test]
    fn test_high_pass_2_pole_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = HighPass2Params {
            period: Some(48),
            k: Some(0.707),
        };
        let first_input = HighPass2Input::from_candles(&candles, "close", first_params);
        let first_result = high_pass_2_pole(&first_input).expect("Failed first pass");
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = HighPass2Params {
            period: Some(32),
            k: Some(0.9),
        };
        let second_input = HighPass2Input::from_slice(&first_result.values, second_params);
        let second_result = high_pass_2_pole(&second_input).expect("Failed second pass");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }

    #[test]
    fn test_high_pass_2_pole_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = HighPass2Params {
            period: Some(48),
            k: Some(0.707),
        };
        let input = HighPass2Input::from_candles(&candles, "close", params);
        let result = high_pass_2_pole(&input).expect("Failed to calculate 2-pole high pass filter");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
