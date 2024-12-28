use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

#[derive(Debug, Clone)]
pub struct JmaOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct JmaParams {
    pub period: Option<usize>,
    pub phase: Option<f64>,
    pub power: Option<u32>,
}

impl JmaParams {
    pub fn with_default_params() -> Self {
        JmaParams {
            period: None,
            phase: None,
            power: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct JmaInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: JmaParams,
}

impl<'a> JmaInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: JmaParams) -> Self {
        JmaInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        JmaInput {
            candles,
            source: "close",
            params: JmaParams::with_default_params(),
        }
    }
}

#[inline]
pub fn jma(input: &JmaInput) -> Result<JmaOutput, Box<dyn Error>> {
    let data: &[f64] = source_type(input.candles, input.source);
    let len: usize = data.len();

    if len == 0 {
        return Err("JMA calculation: input data is empty.".into());
    }

    let period = input.get_period();
    let phase = input.get_phase();
    let power = input.get_power();

    let phase_ratio = if phase < -100.0 {
        0.5
    } else if phase > 100.0 {
        2.5
    } else {
        (phase / 100.0) + 1.5
    };

    let beta = {
        let numerator = 0.45 * (period as f64 - 1.0);
        let denominator = numerator + 2.0;
        if denominator.abs() < f64::EPSILON {
            0.0
        } else {
            numerator / denominator
        }
    };
    let alpha = beta.powi(power as i32);

    let mut e0 = vec![0.0; len];
    let mut e1 = vec![0.0; len];
    let mut e2 = vec![0.0; len];
    let mut jma_val = vec![0.0; len];

    e0[0] = data[0];
    e1[0] = 0.0;
    e2[0] = 0.0;
    jma_val[0] = data[0];

    for i in 1..len {
        let src_i = data[i];

        e0[i] = (1.0 - alpha) * src_i + alpha * e0[i - 1];
        e1[i] = (src_i - e0[i]) * (1.0 - beta) + beta * e1[i - 1];
        let diff = e0[i] + phase_ratio * e1[i] - jma_val[i - 1];
        e2[i] = diff * (1.0 - alpha).powi(2) + alpha.powi(2) * e2[i - 1];
        jma_val[i] = e2[i] + jma_val[i - 1];
    }

    Ok(JmaOutput { values: jma_val })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_jma_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = JmaParams {
            period: None,
            phase: None,
            power: None,
        };
        let input_default = JmaInput::new(&candles, "close", default_params);
        let output_default = jma(&input_default).expect("Failed JMA with default params");
        assert_eq!(output_default.values.len(), candles.close.len());

        let params_custom = JmaParams {
            period: Some(10),
            phase: Some(0.0),
            power: Some(1),
        };
        let input_custom = JmaInput::new(&candles, "hlc3", params_custom);
        let output_custom = jma(&input_custom).expect("Failed JMA with custom params");
        assert_eq!(output_custom.values.len(), candles.close.len());
    }

    #[test]
    fn test_jma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let close_prices: &[f64] = candles
            .select_candle_field("close")
            .expect("Failed to extract close prices");

        let jma_params = JmaParams {
            period: Some(7),
            phase: Some(50.0),
            power: Some(2),
        };
        let input = JmaInput::new(&candles,"close", jma_params);
        let jma_result = jma(&input).expect("Failed to calculate JMA");

        let expected_last_five = [
            59305.04794668568,
            59261.270455005455,
            59156.791263606865,
            59128.30656791065,
            58918.89223153998,
        ];

        assert!(jma_result.values.len() >= 5, "Not enough JMA values for the test");
        assert_eq!(jma_result.values.len(), close_prices.len(), "JMA values count mismatch");
        let start_index = jma_result.values.len() - 5;
        let result_last_five = &jma_result.values[start_index..];
        for (i, &value) in result_last_five.iter().enumerate() {
            let expected_value = expected_last_five[i];
            assert!(
                (value - expected_value).abs() < 1e-6,
                "JMA mismatch at index {}: expected {}, got {}",
                i,
                expected_value,
                value
            );
        }
    }
}