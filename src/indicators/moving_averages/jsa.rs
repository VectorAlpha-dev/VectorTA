use std::error::Error;

#[derive(Debug, Clone)]
pub struct JsaParams {
    pub period: Option<usize>,
}

impl Default for JsaParams {
    fn default() -> Self {
        JsaParams {
            period: Some(30),
        }
    }
}

#[derive(Debug, Clone)]
pub struct JsaInput<'a> {
    pub data: &'a [f64],
    pub params: JsaParams,
}

impl<'a> JsaInput<'a> {
    #[inline]
    pub fn new(data: &'a [f64], params: JsaParams) -> Self {
        Self { data, params }
    }

    #[inline]
    pub fn with_default_params(data: &'a [f64]) -> Self {
        Self {
            data,
            params: JsaParams::default(),
        }
    }

    #[inline]
    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| JsaParams::default().period.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct JsaOutput {
    pub values: Vec<f64>,
}

#[inline]
pub fn calculate_jsa(input: &JsaInput) -> Result<JsaOutput, Box<dyn Error>> {
    let data = input.data;
    let n = data.len();
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
    fn test_jsa_accuracy() {
        let expected_last_five = [61640.0, 61418.0, 61240.0, 61060.5, 60889.5];

        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let source = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let params = JsaParams {
            period: Some(30),
        };
        let input = JsaInput::new(source, params);

        let result = calculate_jsa(&input).expect("Failed to calculate JSA");

        assert_eq!(
            result.values.len(),
            source.len(),
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

        let default_input = JsaInput::with_default_params(source);
        let default_result = calculate_jsa(&default_input).expect("Failed to calculate JSA default");
        assert_eq!(
            default_result.values.len(),
            source.len(),
            "Default JSA result length mismatch"
        );
    }
}
