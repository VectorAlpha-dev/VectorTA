use crate::utilities::data_loader::{source_type, Candles};
use std::error::Error;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct ReflexOutput {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ReflexParams {
    pub period: Option<usize>,
}

impl ReflexParams {
    pub fn with_default_params() -> Self {
        ReflexParams { period: None }
    }
}

#[derive(Debug, Clone)]
pub struct ReflexInput<'a> {
    pub candles: &'a Candles,
    pub source: &'a str,
    pub params: ReflexParams,
}

impl<'a> ReflexInput<'a> {
    pub fn new(candles: &'a Candles, source: &'a str, params: ReflexParams) -> Self {
        ReflexInput {
            candles,
            source,
            params,
        }
    }

    pub fn with_default_params(candles: &'a Candles) -> Self {
        ReflexInput {
            candles,
            source: "close",
            params: ReflexParams::with_default_params(),
        }
    }
}

pub fn reflex(input: &ReflexInput) -> Result<ReflexOutput, Box<dyn Error>> {
    let data: &[f64] = source_type(input.candles, input.source);
    let len: usize = data.len();
    let period: usize = input.params.period.unwrap_or(20);

    if len == 0 {
        return Err("No data available for Reflex.".into());
    }
    if period < 2 {
        return Err("Reflex period must be >=2.".into());
    }

    let half_period = (period / 2).max(1);
    let a = (-1.414_f64 * PI / half_period as f64).exp();
    let a_sq = a * a;
    let b = 2.0 * a * (1.414_f64 * PI / half_period as f64).cos();
    let c = (1.0 + a_sq - b) * 0.5;

    let mut ssf = vec![0.0; len];
    let mut reflex = vec![0.0; len];
    let mut ms = vec![0.0; len];
    let mut sums = vec![0.0; len];

    if len > 0 {
        ssf[0] = data[0];
    }
    if len > 1 {
        ssf[1] = data[1];
    }

    let period_f = period as f64;

    for i in 2..len {
        let d_i = data[i];
        let d_im1 = data[i - 1];
        let prev_ssf1 = ssf[i - 1];
        let prev_ssf2 = ssf[i - 2];
        let ssf_i = c * (d_i + d_im1) + b * prev_ssf1 - a_sq * prev_ssf2;
        ssf[i] = ssf_i;

        if i >= period {
            let slope = (ssf[i - period] - ssf_i) / period_f;

            let mut my_sum = 0.0;
            for t in 1..=period {
                let pred = ssf_i + slope * (t as f64);
                let past = ssf[i - t];
                my_sum += pred - past;
            }
            my_sum /= period_f;
            sums[i] = my_sum;
            let ms_im1 = ms[i - 1];
            let my_sum_sq = my_sum * my_sum;
            let ms_i = 0.04 * my_sum_sq + 0.96 * ms_im1;
            ms[i] = ms_i;

            reflex[i] = if ms_i > 0.0 {
                my_sum / ms_i.sqrt()
            } else {
                0.0
            };
        }
    }

    Ok(ReflexOutput { values: reflex })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::data_loader::read_candles_from_csv;

    #[test]
    fn test_reflex_partial_params() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = ReflexParams { period: None };
        let input = ReflexInput::new(&candles, "close", default_params);
        let output = reflex(&input).expect("Failed Reflex with default params");
        assert_eq!(output.values.len(), candles.close.len());
        let params_period_14 = ReflexParams { period: Some(14) };
        let input2 = ReflexInput::new(&candles, "hl2", params_period_14);
        let output2 = reflex(&input2).expect("Failed Reflex with period=14, source=hl2");
        assert_eq!(output2.values.len(), candles.close.len());
        let params_custom = ReflexParams { period: Some(30) };
        let input3 = ReflexInput::new(&candles, "hlc3", params_custom);
        let output3 = reflex(&input3).expect("Failed Reflex fully custom");
        assert_eq!(output3.values.len(), candles.close.len());
    }

    #[test]
    fn test_reflex_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
        let candles = read_candles_from_csv(file_path).expect("Failed to load test candles");
        let default_params = ReflexParams::with_default_params();
        let input = ReflexInput::new(&candles, "close", default_params);
        let result = reflex(&input).expect("Failed to calculate Reflex");
        assert_eq!(
            result.values.len(),
            candles.close.len(),
            "Output size mismatch"
        );
        let len = result.values.len();
        let expected_last_five = [
            0.8085220962465361,
            0.445264715886137,
            0.13861699036615063,
            -0.03598639652007061,
            -0.224906760543743,
        ];
        assert!(len >= 5, "Not enough data for the test");
        let start_idx = len - 5;
        let last_five = &result.values[start_idx..];
        for (i, &val) in last_five.iter().enumerate() {
            let exp = expected_last_five[i];
            assert!(
                (val - exp).abs() < 1e-7,
                "Reflex mismatch at offset {}, expected {}, got {}",
                i,
                exp,
                val
            );
        }
    }
}