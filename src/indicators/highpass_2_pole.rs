use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct HighPass2Params {
    pub period: Option<usize>,
    pub k: Option<f64>,
}

impl Default for HighPass2Params {
    fn default() -> Self {
        HighPass2Params {
            period: Some(48),
            k: Some(0.707),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HighPass2Input<'a> {
    pub data: &'a [f64],
    pub params: HighPass2Params,
}

impl<'a> HighPass2Input<'a> {
    pub fn new(data: &'a [f64], params: HighPass2Params) -> Self {
        HighPass2Input { data, params }
    }

    pub fn with_default_params(data: &'a [f64]) -> Self {
        HighPass2Input {
            data,
            params: HighPass2Params::default(),
        }
    }

    fn get_period(&self) -> usize {
        self.params
            .period
            .unwrap_or_else(|| HighPass2Params::default().period.unwrap())
    }

    fn get_k(&self) -> f64 {
        self.params
            .k
            .unwrap_or_else(|| HighPass2Params::default().k.unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct HighPass2Output {
    pub values: Vec<f64>,
}

pub fn calculate_high_pass_2_pole(
    input: &HighPass2Input,
) -> Result<HighPass2Output, Box<dyn Error>> {
    let data = input.data;
    let len = data.len();

    let period = input.get_period();
    let k = input.get_k();

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
    use crate::indicators::data_loader::{read_candles_from_csv, Candles};

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
        let input = HighPass2Input::new(&close_prices, params);
        let result = calculate_high_pass_2_pole(&input)
            .expect("Failed to calculate 2-pole high pass filter");

        let expected_last_five = vec![
            445.29073821108943,
            359.51467478973296,
            250.7236793408186,
            394.04381266217234,
            -52.65414073315134,
        ];

        assert!(
            result.values.len() >= 5,
            "Not enough high-pass 2 pole values for test"
        );

        let start_index = result.values.len() - 5;
        let actual_last_five = &result.values[start_index..];

        for (i, &actual) in actual_last_five.iter().enumerate() {
            let expected = expected_last_five[i];
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-6,
                "Mismatch at index {}: expected {}, got {}, diff={}",
                i,
                expected,
                actual,
                diff
            );
        }
    }
}
