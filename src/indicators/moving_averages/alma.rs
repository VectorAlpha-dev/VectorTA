use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum AlmaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct AlmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct AlmaParams {
    pub period: Option<usize>,
    pub offset: Option<f64>,
    pub sigma: Option<f64>,
}

impl AlmaParams {
    pub fn with_default_params() -> Self {
        AlmaParams {
            period: None,
            offset: None,
            sigma: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlmaInput<'a> {
    pub data: AlmaData<'a>,
    pub params: AlmaParams,
}

impl<'a> AlmaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: AlmaParams) -> Self {
        Self {
            data: AlmaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: AlmaParams) -> Self {
        Self {
            data: AlmaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: AlmaData::Candles {
                candles,
                source: "close",
            },
            params: AlmaParams::with_default_params(),
        }
    }
}

#[inline]
pub fn alma(input: &AlmaInput) -> Result<AlmaOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        AlmaData::Candles { candles, source } => source_type(candles, source),
        AlmaData::Slice(slice) => slice,
    };
    let len: usize = data.len();

    let period: usize = input.params.period.unwrap_or(9);
    let offset: f64 = input.params.offset.unwrap_or(0.85);
    let sigma: f64 = input.params.sigma.unwrap_or(6.0);

    if period == 0 || period > len {
        return Err("Invalid period specified for ALMA calculation.".into());
    }

    let m: f64 = offset * (period - 1) as f64;
    let s: f64 = period as f64 / sigma;
    let s_sq: f64 = s * s;
    let den: f64 = 2.0 * s_sq;

    let mut weights: Vec<f64> = Vec::with_capacity(period);
    let mut norm: f64 = 0.0;

    for i in 0..period {
        let diff: f64 = i as f64 - m;
        let num: f64 = diff * diff;
        let w: f64 = (-num / den).exp();
        weights.push(w);
        norm += w;
    }

    let inv_norm: f64 = 1.0 / norm;
    let mut alma_values: Vec<f64> = vec![f64::NAN; len];

    alma_values
        .iter_mut()
        .enumerate()
        .skip(period - 1)
        .for_each(|(i, value)| {
            let start: usize = i + 1 - period;
            let mut sum: f64 = 0.0;
            for (idx, &w) in weights.iter().enumerate() {
                sum += data[start + idx] * w;
            }
            *value = sum * inv_norm;
        });

    Ok(AlmaOutput {
        values: alma_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_alma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = AlmaParams {
            period: None,
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_candles(&candles, "close", default_params);
        let output = alma(&input).expect("Failed ALMA with default params");
        assert_eq!(output.values.len(), candles.close.len());

        let params_period_14 = AlmaParams {
            period: Some(14),
            offset: None,
            sigma: None,
        };
        let input2 = AlmaInput::from_candles(&candles, "hl2", params_period_14);
        let output2 = alma(&input2).expect("Failed ALMA with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());

        let params_custom = AlmaParams {
            period: Some(10),
            offset: Some(0.9),
            sigma: Some(5.0),
        };
        let input3 = AlmaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = alma(&input3).expect("Failed ALMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }

    #[test]
    fn test_alma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = AlmaParams::with_default_params();

        let input = AlmaInput::from_candles(&candles, "close", default_params);
        let result = alma(&input).expect("Failed to calculate ALMA");

        let expected_last_five = [59286.7222, 59273.5343, 59204.3729, 59155.9338, 59026.9253];

        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "ALMA output length does not match input length!"
        );

        let start_index = result.values.len().saturating_sub(5);
        let result_last_five = &result.values[start_index..];

        for (i, &value) in result_last_five.iter().enumerate() {
            assert!(
                (value - expected_last_five[i]).abs() < 1e-1,
                "ALMA value mismatch at index {}: expected {}, got {}",
                i,
                expected_last_five[i],
                value
            );
        }

        for val in result.values.iter() {
            if !val.is_nan() {
                assert!(val.is_finite(), "ALMA output should be finite");
            }
        }
    }
    #[test]
    fn test_alma_params_with_default_params() {
        let params = AlmaParams::with_default_params();
        assert_eq!(params.period, None);
        assert_eq!(params.offset, None);
        assert_eq!(params.sigma, None);
    }

    #[test]
    fn test_alma_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = AlmaInput::with_default_candles(&candles);
        match input.data {
            AlmaData::Candles { source, .. } => assert_eq!(source, "close"),
            _ => panic!("Expected AlmaData::Candles variant"),
        }
        assert_eq!(input.params.period, None);
        assert_eq!(input.params.offset, None);
        assert_eq!(input.params.sigma, None);
    }

    #[test]
    fn test_alma_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = AlmaParams {
            period: Some(0),
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_slice(&input_data, params);
        let result = alma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_alma_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = AlmaParams {
            period: Some(10),
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_slice(&input_data, params);
        let result = alma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_alma_very_small_data_set() {
        let input_data = [42.0];
        let params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_slice(&input_data, params);
        let result = alma(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_alma_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let first_params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let first_input = AlmaInput::from_candles(&candles, "close", first_params);
        let first_result = alma(&first_input).expect("Failed first ALMA");
        assert_eq!(first_result.values.len(), candles.close.len());
        let second_params = AlmaParams {
            period: Some(5),
            offset: Some(0.8),
            sigma: Some(4.0),
        };
        let second_input = AlmaInput::from_slice(&first_result.values, second_params);
        let second_result = alma(&second_input).expect("Failed second ALMA");
        assert_eq!(second_result.values.len(), first_result.values.len());
    }

    #[test]
    fn test_alma_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let params = AlmaParams {
            period: Some(9),
            offset: None,
            sigma: None,
        };
        let input = AlmaInput::from_candles(&candles, "close", params);
        let result = alma(&input).expect("Failed ALMA calculation");
        assert_eq!(result.values.len(), candles.close.len());
        if result.values.len() > 240 {
            for i in 240..result.values.len() {
                assert!(!result.values[i].is_nan());
            }
        }
    }
}
