use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub enum JsaData<'a> {
    Candles {
        candles: &'a Candles,
        source: &'a str,
    },
    Slice(&'a [f64]),
}

#[derive(Debug, Clone)]
pub struct JsaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct JsaParams {
    pub period: Option<usize>,
}

impl Default for JsaParams {
    fn default() -> Self {
        Self { period: Some(30) }
    }
}

#[derive(Debug, Clone)]
pub struct JsaInput<'a> {
    pub data: JsaData<'a>,
    pub params: JsaParams,
}

impl<'a> JsaInput<'a> {
    pub fn from_candles(candles: &'a Candles, source: &'a str, params: JsaParams) -> Self {
        Self {
            data: JsaData::Candles { candles, source },
            params,
        }
    }

    pub fn from_slice(slice: &'a [f64], params: JsaParams) -> Self {
        Self {
            data: JsaData::Slice(slice),
            params,
        }
    }

    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self {
            data: JsaData::Candles {
                candles,
                source: "close",
            },
            params: JsaParams::default(),
        }
    }

    pub fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| JsaParams::default().period.unwrap())
    }
}

#[inline]
pub fn jsa(input: &JsaInput) -> Result<JsaOutput, Box<dyn Error>> {
    let data: &[f64] = match &input.data {
        JsaData::Candles { candles, source } => source_type(candles, source),
        JsaData::Slice(slice) => slice,
    };
    let len: usize = data.len();
    if len == 0 {
        return Err("No data provided for JSA calculation.".into());
    }

    let period = input.get_period();
    if period == 0 {
        return Err("JSA period must be > 0.".into());
    }
    if period >= len {
        let output = vec![f64::NAN; len];
        return Ok(JsaOutput { values: output });
    }

    let mut output = vec![f64::NAN; len];

    for i in period..len {
        output[i] = (data[i] + data[i - period]) * 0.5;
    }

    Ok(JsaOutput { values: output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_jsa_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = JsaParams { period: None };
        let input = JsaInput::from_candles(&candles, "close", default_params);
        let output = jsa(&input).expect("Failed JSA with default params");
        assert_eq!(output.values.len(), candles.close.len());

        let params_period_14 = JsaParams { period: Some(14) };
        let input2 = JsaInput::from_candles(&candles, "hl2", params_period_14);
        let output2 = jsa(&input2).expect("Failed JSA with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());

        let params_custom = JsaParams { period: Some(10) };
        let input3 = JsaInput::from_candles(&candles, "hlc3", params_custom);
        let output3 = jsa(&input3).expect("Failed JSA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }

    #[test]
    fn test_jsa_accuracy() {
        let expected_last_five = [61640.0, 61418.0, 61240.0, 61060.5, 60889.5];
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = JsaParams::default();
        let input = JsaInput::from_candles(&candles, "close", default_params);
        let result = jsa(&input).expect("Failed to calculate JSA");
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "JSA result length mismatch"
        );
        assert!(
            result.values.len() >= 5,
            "Not enough data to compare last 5 JSA values"
        );
        let start_idx = result.values.len() - 5;
        let actual_last_five = &result.values[start_idx..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!(
                (val - expected).abs() < 1e-5,
                "JSA mismatch at index {} => expected {}, got {}",
                i,
                expected,
                val
            );
        }
        for val in result.values.iter() {
            if !val.is_nan() {
                assert!(val.is_finite(), "JSA output should be finite");
            }
        }
        let default_input = JsaInput::with_default_candles(&candles);
        let default_result = jsa(&default_input).expect("Failed to calculate JSA default");
        assert_eq!(
            default_result.values.len(),
            candles.close.len(),
            "Default JSA result length mismatch"
        );
    }
    #[test]
    fn test_jsa_params_with_default_params() {
        let default_params = JsaParams::default();
        assert_eq!(
            default_params.period,
            Some(30),
            "Default period should be 30"
        );
    }

    #[test]
    fn test_jsa_input_with_default_candles() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let input = JsaInput::with_default_candles(&candles);
        match input.data {
            JsaData::Candles { source, .. } => {
                assert_eq!(source, "close");
            }
            _ => panic!("Expected JsaData::Candles variant"),
        }
    }

    #[test]
    fn test_jsa_with_zero_period() {
        let input_data = [10.0, 20.0, 30.0];
        let params = JsaParams { period: Some(0) };
        let input = JsaInput::from_slice(&input_data, params);
        let result = jsa(&input);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string().contains("JSA period must be > 0"),
                "Expected 'JSA period must be > 0' error, got: {}",
                e
            );
        }
    }

    #[test]
    fn test_jsa_with_period_exceeding_data_length() {
        let input_data = [10.0, 20.0, 30.0];
        let params = JsaParams { period: Some(10) };
        let input = JsaInput::from_slice(&input_data, params);
        let result = jsa(&input).expect("Should not panic with large period");
        for &val in &result.values {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn test_jsa_very_small_data_set() {
        let input_data = [42.0];
        let params = JsaParams { period: Some(5) };
        let input = JsaInput::from_slice(&input_data, params);
        let result = jsa(&input).expect("Should not panic on small data");
        for &val in &result.values {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn test_jsa_with_slice_data_reinput() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let first_params = JsaParams { period: Some(10) };
        let first_input = JsaInput::from_candles(&candles, "close", first_params);
        let first_result = jsa(&first_input).expect("Failed to calculate first JSA");
        assert_eq!(first_result.values.len(), candles.close.len());

        let second_params = JsaParams { period: Some(5) };
        let second_input = JsaInput::from_slice(&first_result.values, second_params);
        let second_result = jsa(&second_input).expect("Failed to calculate second JSA");
        assert_eq!(second_result.values.len(), first_result.values.len());
        for i in 240..second_result.values.len() {
            assert!(
                second_result.values[i].is_finite(),
                "NaN found at index {}",
                i
            );
        }
    }

    #[test]
    fn test_jsa_accuracy_nan_check() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let period = 10;
        let params = JsaParams {
            period: Some(period),
        };
        let input = JsaInput::from_candles(&candles, "close", params);
        let result = jsa(&input).expect("Failed to calculate JSA");
        assert_eq!(result.values.len(), candles.close.len());

        if result.values.len() > period {
            for i in period..result.values.len() {
                assert!(!result.values[i].is_nan(), "Unexpected NaN at index {}", i);
            }
        }
    }
}
