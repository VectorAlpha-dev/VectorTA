use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct JsaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct JsaParams {
    pub period: Option<usize>,
}

impl JsaParams {
    pub fn with_default_params() -> Self {
        JsaParams { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct JsaInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: JsaParams,
}

impl<'a> JsaInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: JsaParams) -> Self {
        JsaInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        JsaInput {
            candles,
            source: "close",
            params: JsaParams::with_default_params(),
        }
    }
}

pub fn jsa(input: &JsaInput) -> Result<JsaOutput, Box<dyn Error>> {
    let data: &[f64] = source_type(input.candles, input.source);
    let len: usize = data.len();
    if n == 0 {
        return Err("No data provided for JSA calculation.".into());
    }

    let period = input.get_period();
    if period == 0 {
        return Err("JSA period must be > 0.".into());
    }
    if period >= n {
        let output = vec![f64::NAN; n];
        return Ok(JsaOutput { values: output });
    }

    let mut output = vec![f64::NAN; n];

    for i in period..n {
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
        let input = JsaInput::new(&candles, "close", default_params);
        let output = jsa(&input).expect("Failed JSA with default params");
        assert_eq!(output.values.len(), candles.close.len());

        let params_period_14 = JsaParams { period: Some(14) };
        let input2 = JsaInput::new(&candles, "hl2", params_period_14);
        let output2 = jsa(&input2).expect("Failed JSA with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());

        let params_custom = JsaParams { period: Some(10) };
        let input3 = JsaInput::new(&candles, "hlc3", params_custom);
        let output3 = jsa(&input3).expect("Failed JSA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }

    #[test]
    fn test_jsa_accuracy() {
        let expected_last_five = [61640.0, 61418.0, 61240.0, 61060.5, 60889.5];
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = JsaParams::with_default_params();
        let input = JsaInput::new(&candles, "close", default_params);
        let result = jsa(&input).expect("Failed to calculate JSA");
        assert_eq!(result.values.len(), candles.close.len(), "JSA result length mismatch");
        assert!(result.values.len() >= 5, "Not enough data to compare last 5 JSA values");
        let start_idx = result.values.len() - 5;
        let actual_last_five = &result.values[start_idx..];
        for (i, &val) in actual_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            assert!((val - expected).abs() < 1e-5, "JSA mismatch at index {} => expected {}, got {}", i, expected, val);
        }
        for val in result.values.iter() {
            if !val.is_nan() {
                assert!(val.is_finite(), "JSA output should be finite");
            }
        }
        let default_input = JsaInput::with_default_params(&candles);
        let default_result = jsa(&default_input).expect("Failed to calculate JSA default");
        assert_eq!(default_result.values.len(), candles.close.len(), "Default JSA result length mismatch");
    }
}
