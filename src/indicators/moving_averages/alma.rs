use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;

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
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: AlmaParams,
}

impl<'a> AlmaInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: AlmaParams) -> Self {
        AlmaInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        AlmaInput {
            candles,
            source: "close",
            params: AlmaParams::with_default_params(),
        }
    }
}

pub fn alma(input: &AlmaInput) -> Result<AlmaOutput, Box<dyn Error>> {
    let data = source_type(input.candles, input.source);
    let len = data.len();

    let period = input.params.period.unwrap_or(9);
    let offset = input.params.offset.unwrap_or(0.85);
    let sigma = input.params.sigma.unwrap_or(6.0);

    if period == 0 || period > len {
        return Err("Invalid period specified for ALMA calculation.".into());
    }

    let m = offset * (period - 1) as f64;
    let s = period as f64 / sigma;
    let s_sq = s * s;
    let den = 2.0 * s_sq;

    let mut weights = Vec::with_capacity(period);
    let mut norm = 0.0;

    for i in 0..period {
        let diff = i as f64 - m;
        let w = (-diff * diff / den).exp();
        weights.push(w);
        norm += w;
    }

    let inv_norm = 1.0 / norm;
    let mut alma_values = vec![f64::NAN; len];

    for i in (period - 1)..len {
        let start = i + 1 - period;
        let mut sum = 0.0;
        for (idx, &w) in weights.iter().enumerate() {
            sum += data[start + idx] * w;
        }
        alma_values[i] = sum * inv_norm;
    }

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
        let input = AlmaInput::new(&candles, "close", default_params);
        let output = alma(&input).expect("Failed ALMA with default params");
        assert_eq!(output.values.len(), candles.close.len());

        let params_period_14 = AlmaParams {
            period: Some(14),
            offset: None,
            sigma: None,
        };
        let input2 = AlmaInput::new(&candles, "hl2", params_period_14);
        let output2 = alma(&input2).expect("Failed ALMA with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());

        let params_custom = AlmaParams {
            period: Some(10),
            offset: Some(0.9),
            sigma: Some(5.0),
        };
        let input3 = AlmaInput::new(&candles, "hlc3", params_custom);
        let output3 = alma(&input3).expect("Failed ALMA fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }

    #[test]
    fn test_alma_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");

        let default_params = AlmaParams::with_default_params();

        let input = AlmaInput::new(&candles, "close", default_params);
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
}
